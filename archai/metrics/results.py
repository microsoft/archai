from typing import Union, List, Optional, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider
from archai.metrics.base import BaseMetric, BaseAsyncMetric
from archai.search_spaces.discrete.base import DiscreteSearchSpaceBase
from archai.metrics.utils import get_pareto_frontier, _find_pareto_frontier_points


class SearchResults():
    def __init__(self, search_space: DiscreteSearchSpaceBase,
                 objectives: List[Union[BaseMetric, BaseAsyncMetric]]):
        self.search_space = search_space
        self.objectives = objectives
        self.iteration_num = 0
        self.results = []

    def add_iteration_results(self, models: List[ArchWithMetaData],
                              evaluation_results: np.ndarray,
                              extra_model_data: Optional[Dict[str, List]] = None):
        """Stores results of the current search iteration.

        Args:
            models (List[ArchWithMetaData]): Models evaluated in the search iteration
            evaluation_results (np.ndarray): Evaluation results
            extra_model_data (Dict[str, List], optional): Additional model information to be
                stored in the search state file. Must be a list of the same size as `models` and
                csv-serializable.
        """
        assert len(self.objectives) == evaluation_results.shape[1]
        assert len(models) == evaluation_results.shape[0]

        extra_model_data = extra_model_data or dict()
        
        if extra_model_data:
            assert all(len(v) == len(models) for v in extra_model_data.values())

        self.results.append({
            'models': [m for m in models], # To avoid creating a reference to `models` variable
            'evaluation_results': evaluation_results.tolist(),
            **extra_model_data
        })

        self.iteration_num += 1

    def get_pareto_frontier(self, iteration: Optional[int] = None) -> Dict:
        """Gets the pareto-frontier from `iteration`. If `iteration=None`,
        gets the latest pareto-frontier.

        Args:
            iteration (Optional[int], optional): Search iteration. If `None`, gets
                the latest pareto-frontier. Defaults to None.

        Returns:
            Dict: Dictionary containing 'models', 'evaluation_results', 
             'indices' and 'iteration_nums' for all pareto-frontier members.
        """        
        iteration = iteration or self.iteration_num

        all_models = [
            model 
            for it in range(iteration)
            for model in self.results[it]['models']
        ]
        
        all_results = np.concatenate([
            np.array(self.results[it]['evaluation_results'])
            for it in range(iteration)
        ], axis=0)

        all_iteration_nums = np.array([
            it 
            for it in range(iteration)
            for _ in range(len(self.results[it]['evaluation_results']))
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

        # Spread objectives in separate columns
        objective_names = [obj.__class__.__name__ for obj in self.objectives]
        state_df = pd.concat([
            state_df, 
            pd.DataFrame(state_df['evaluation_results'].tolist(), columns=objective_names)
        ], axis=1)

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
            self.search_space.save_arch(model, str(dir_path / f'{model["archid"]}'))
 
