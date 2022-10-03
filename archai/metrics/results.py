from typing import Union, List, Optional, Dict, Tuple, Any
from pathlib import Path
import re
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider
from archai.metrics.base import BaseMetric, BaseAsyncMetric
from archai.search_spaces.discrete.base import DiscreteSearchSpaceBase
from archai.metrics.utils import get_pareto_frontier, _find_pareto_frontier_points


class SearchResults():
    def __init__(self, search_space: DiscreteSearchSpaceBase,
                 objectives: Dict[str, Union[BaseMetric, BaseAsyncMetric]]):
        self.search_space = search_space
        self.objectives = objectives
        self.iteration_num = 0
        self.results = []

    def add_iteration_results(self, models: List[ArchWithMetaData],
                              evaluation_results: Dict[str, np.ndarray],
                              extra_model_data: Optional[Dict[str, List]] = None):
        """Stores results of the current search iteration.

        Args:
            models (List[ArchWithMetaData]): Models evaluated in the search iteration
            evaluation_results (Dict[str, np.ndarray]): Evaluation results from `archai.metrics.evaluate_models()`
            extra_model_data (Dict[str, List], optional): Additional model information to be
                stored in the search state file. Must be a list of the same size as `models` and
                csv-serializable.
        """
        assert len(self.objectives) == len(evaluation_results)
        assert all(len(r) == len(models) for r in evaluation_results.values())

        extra_model_data = extra_model_data or dict()
        
        if extra_model_data:
            assert all(len(v) == len(models) for v in extra_model_data.values())
        
        evaluation_results = copy.deepcopy(evaluation_results)
        evaluation_results.update(extra_model_data)

        self.results.append({
            'archid': [m.metadata['archid'] for m in models],
            'models': [m for m in models], # To avoid creating a reference to `models` variable
            **evaluation_results
        })

        self.iteration_num += 1

    def get_pareto_frontier(self, start_iteration: int = 0, end_iteration: Optional[int] = None) -> Dict:
        """Gets the pareto-frontier using the search results from iterations `start_iteration` to `end_iteration`.
        If `end_iteration=None`, uses the last iteration. 

        Args:
            start_iteration (int, optional): Start search iteration. Defaults to 0
            end_iteration (Optional[int], optional): End search iteration. If `None`, uses
                the last iteration. Defaults to None. 

        Returns:
            Dict: Dictionary containing 'models', 'evaluation_results', 
             'indices' and 'iteration_nums' for all pareto-frontier members.
        """        
        end_iteration = end_iteration or self.iteration_num

        all_models = [
            model 
            for it in range(start_iteration, end_iteration)
            for model in self.results[it]['models']
        ]
        
        all_results = {
            obj_name: np.concatenate([
                self.results[it][obj_name]
                for it in range(start_iteration, end_iteration)
            ], axis=0)
            for obj_name in self.objectives.keys()
        }

        all_iteration_nums = np.array([
            it 
            for it in range(start_iteration, end_iteration)
            for _ in range(len(self.results[it]['models']))
        ])

        pareto_frontier = get_pareto_frontier(all_models, all_results, self.objectives)
        pareto_frontier.update({'iteration_nums': all_iteration_nums[pareto_frontier['indices']]})

        return pareto_frontier

    def get_search_state_df(self) -> pd.DataFrame:
        """Gets the search state pd.DataFrame

        Returns:
            pd.DataFrame: search state DataFrame.
        """        
        state_df = pd.concat([
            pd.DataFrame(it_results).assign(iteration_num=it)
            for it, it_results in enumerate(self.results)
        ], axis=0).reset_index(drop=True)

        pareto_frontier = self.get_pareto_frontier()

        state_df['is_pareto'] = False
        state_df.loc[pareto_frontier['indices'], 'is_pareto'] = True

        return state_df

    def save_search_state(self, file: str) -> None:
        state_df = self.get_search_state_df()
        state_df.to_csv(file, index=False)

    def save_pareto_frontier_models(self, directory: str, save_weights: bool = False):
        if save_weights:
            raise NotImplementedError
        
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True, parents=True)

        pareto_frontier = self.get_pareto_frontier()
        for model in pareto_frontier['models']:
            self.search_space.save_arch(model, str(dir_path / f'{model.metadata["archid"]}'))
 
    def save_2d_pareto_evolution_plot(self, objective_names: Tuple[str, str], path: str) -> Any:
        obj_x, obj_y = objective_names
        status_df = self.get_search_state_df().copy()

        fig, ax = plt.subplots(figsize=(10, 5))
        status_range = range(0, self.iteration_num + 1)
        
        # Transforms dimensions to be decreasing if necessary
        max_x, max_y = status_df[obj_x].max(), status_df[obj_y].max()
        status_df['x'], status_df['y'] = status_df[obj_x], status_df[obj_y]

        if self.objectives[obj_x].higher_is_better:
            status_df['x'] = (max_x - status_df['x'])
        
        if self.objectives[obj_y].higher_is_better:
            status_df['y'] = (max_y - status_df['y'])

        colors = plt.cm.plasma(np.linspace(0, 1, self.iteration_num + 1))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=self.iteration_num + 1))

        for s in status_range:
            generation_df = status_df.query(f'iteration_num <= {s}').copy()
            
            points = generation_df[['x', 'y']].values
            pareto_df = generation_df.iloc[_find_pareto_frontier_points(points)].copy()
            pareto_df = pareto_df.sort_values('x')

            ax.step(
                pareto_df[obj_x], pareto_df[obj_y],
                where='post',
                color=colors[s]
            )
            ax.plot(pareto_df[obj_x], pareto_df[obj_y], 'o', color=colors[s])
        
        ax.set_xlabel(obj_x)
        ax.set_ylabel(obj_y)
        fig.colorbar(sm, ax=ax)
        fig.savefig(path)

        return fig

    def save_all_2d_pareto_evolution_plots(self, directory: str) -> List[Any]:
        path = Path(directory)
        path.mkdir(exist_ok=True, parents=True)

        objective_names = list(self.objectives.keys())
        plots = []

        for i, obj_x in enumerate(objective_names):
            for obj_y in objective_names[(i + 1):]:
                # Sanitizes filename
                fname = f'pareto_{obj_x}_vs_{obj_y}.png'.strip().replace(' ', '_')
                fname = re.sub(r'(?u)[^-\w.]', '', fname)

                plots.append(
                    self.save_2d_pareto_evolution_plot((obj_x, obj_y), str(path / fname))
                )

        return plots
