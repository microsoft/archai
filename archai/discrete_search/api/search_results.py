# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import re
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.search_objectives import SearchObjectives
from archai.discrete_search.api.search_space import DiscreteSearchSpace
from archai.discrete_search.utils.multi_objective import (
    _find_pareto_frontier_points,
    get_pareto_frontier,
)


class SearchResults:
    """Discrete search results.

    This class implements search results, which consists in producing data frames
    and plots with information regarding the search.

    """

    def __init__(self, search_space: DiscreteSearchSpace, objectives: SearchObjectives) -> None:
        """Initialize the search results.

        Args:
            search_space: Search space.
            objectives: Search objectives.

        """

        self.search_space = search_space
        self.objectives = objectives

        self.iteration_num = 0
        self.init_time = time()
        self.search_walltimes = []
        self.results = []

    @property
    def all_evaluated_objs(self) -> Dict[str, np.array]:
        """Return all evaluated objectives."""

        return {
            obj_name: np.array([r for iter_results in self.results for r in iter_results[obj_name]], dtype=np.float32)
            for obj_name in self.objectives.objectives
        }

    def add_iteration_results(
        self,
        models: List[ArchaiModel],
        evaluation_results: Dict[str, np.ndarray],
        extra_model_data: Optional[Dict[str, List]] = None,
    ) -> None:
        """Store results of the current search iteration.

        Args:
            models: Models evaluated in the search iteration.
            evaluation_results: Evaluation results from `SearchObjectives.eval_all_objs()`.
            extra_model_data: Additional model information to be stored in the search state
                file. Must be a list of the same size as `models` and csv-serializable.

        """

        assert all(obj_name in evaluation_results for obj_name in self.objectives.objectives)
        assert all(len(r) == len(models) for r in evaluation_results.values())

        extra_model_data = copy.deepcopy(extra_model_data) or dict()

        if extra_model_data:
            assert all(len(v) == len(models) for v in extra_model_data.values())

        evaluation_results = copy.deepcopy(evaluation_results)
        evaluation_results.update(extra_model_data)

        self.results.append(
            {
                "archid": [m.archid for m in models],
                "models": [m for m in models],  # To avoid creating a reference to `models` variable
                **evaluation_results,
            }
        )

        # Adds current search duration in hours
        self.search_walltimes += [(time() - self.init_time) / 3600] * len(models)
        self.iteration_num += 1

    def get_pareto_frontier(
        self, start_iteration: Optional[int] = 0, end_iteration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get the pareto-frontier using the search results from iterations `start_iteration`
        to `end_iteration`. If `end_iteration=None`, uses the last iteration.

        Args:
            start_iteration: Start search iteration.
            end_iteration: End search iteration. If `None`, uses the last iteration.

        Returns:
            Dictionary containing 'models', 'evaluation_results', 'indices' and
                'iteration_nums' for all pareto-frontier members.

        """

        end_iteration = end_iteration or self.iteration_num

        all_models = [model for it in range(start_iteration, end_iteration) for model in self.results[it]["models"]]

        all_results = {
            obj_name: np.concatenate(
                [self.results[it][obj_name] for it in range(start_iteration, end_iteration)], axis=0
            )
            for obj_name in self.objectives.objective_names
        }

        all_iteration_nums = np.array(
            [it for it in range(start_iteration, end_iteration) for _ in range(len(self.results[it]["models"]))]
        )

        pareto_frontier = get_pareto_frontier(all_models, all_results, self.objectives)
        pareto_frontier.update({"iteration_nums": all_iteration_nums[pareto_frontier["indices"]]})

        return pareto_frontier

    def get_search_state_df(self) -> pd.DataFrame:
        """Get the search state data frame.

        Returns:
            Search state data frame.

        """

        state_df = pd.concat(
            [pd.DataFrame(it_results).assign(iteration_num=it) for it, it_results in enumerate(self.results)], axis=0
        ).reset_index(drop=True)
        state_df["search_walltime_hours"] = self.search_walltimes

        pareto_frontier = self.get_pareto_frontier()

        state_df["is_pareto"] = False
        state_df.loc[pareto_frontier["indices"], "is_pareto"] = True

        return state_df.drop(["models"], axis=1)

    def save_search_state(self, file_path: Union[str, Path]) -> None:
        """Save the search state to a .csv file.

        Args:
            file_path: File path to save the search state.

        """

        state_df = self.get_search_state_df()
        state_df.to_csv(file_path, index=False)

    def save_pareto_frontier_models(self, directory: str, save_weights: Optional[bool] = False) -> None:
        """Save the pareto-frontier models to a directory.

        Args:
            directory: Directory to save the models.
            save_weights: If `True`, saves the model weights. Otherwise, only saves the architecture.

        """
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True, parents=True)

        pareto_frontier = self.get_pareto_frontier()
        for model in pareto_frontier["models"]:
            self.search_space.save_arch(model, str(dir_path / f"{model.archid}"))

            if save_weights:
                self.search_space.save_model_weights(model, str(dir_path / f"{model.archid}_weights.pt"))

    def plot_2d_pareto_evolution(
        self, objective_names: Tuple[str, str], figsize: Optional[Tuple[int, int]] = (10, 5)
    ) -> plt.Figure:
        """Plot the evolution of the pareto-frontier in 2D.

        Args:
            objective_names: Names of the objectives to plot.
            figsize: Figure size.

        Returns:
            2D pareto-frontier evolution figure.

        """

        obj_x, obj_y = objective_names
        status_df = self.get_search_state_df().copy()

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')
        status_range = range(0, self.iteration_num + 1)

        # Transforms dimensions to be decreasing if necessary
        max_x, max_y = status_df[obj_x].max(), status_df[obj_y].max()
        status_df["x"], status_df["y"] = status_df[obj_x], status_df[obj_y]

        if self.objectives.objectives[obj_x].higher_is_better:
            status_df["x"] = max_x - status_df["x"]

        if self.objectives.objectives[obj_y].higher_is_better:
            status_df["y"] = max_y - status_df["y"]

        colors = plt.cm.plasma(np.linspace(0, 1, self.iteration_num + 1))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=self.iteration_num + 1))

        for s in status_range:
            generation_df = status_df.query(f"iteration_num <= {s}").copy()

            points = generation_df[["x", "y"]].values
            pareto_df = generation_df.iloc[_find_pareto_frontier_points(points)].copy()
            pareto_df = pareto_df.sort_values("x")

            ax.step(pareto_df[obj_x], pareto_df[obj_y], where="post", color=colors[s])
            ax.plot(pareto_df[obj_x], pareto_df[obj_y], "o", color=colors[s])

        ax.set_xlabel(obj_x)
        ax.set_ylabel(obj_y)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Iteration number", rotation=270, labelpad=15)

        ax.set_title("Evolution of Pareto Frontier (2D projection)")
        plt.close()
        return fig

    def save_2d_pareto_evolution_plot(self, objective_names: Tuple[str, str], file_path: str) -> None:
        """Save the evolution of the pareto-frontier in 2D.

        Args:
            objective_names: Names of the objectives to plot.
            file_path: Path to save the plot.

        """

        fig = self.plot_2d_pareto_evolution(objective_names)
        fig.savefig(file_path)

    def save_all_2d_pareto_evolution_plots(self, directory: Union[str, Path]) -> None:
        """Save all the 2D pareto-frontier evolution plots.

        Args:
            directory: Directory to save the plots.

        """

        path = Path(directory)
        path.mkdir(exist_ok=True, parents=True)

        objective_names = list(self.objectives.objective_names)
        plots = []

        for i, obj_x in enumerate(objective_names):
            for obj_y in objective_names[(i + 1) :]:
                # Sanitizes filename
                fname = f"pareto_{obj_x}_vs_{obj_y}.png".strip().replace(" ", "_")
                fname = re.sub(r"(?u)[^-\w.]", "", fname)

                plots.append(self.save_2d_pareto_evolution_plot((obj_x, obj_y), str(path / fname)))
