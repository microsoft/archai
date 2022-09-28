from typing import Union, List, Optional, Dict, Tuple
from pathlib import Path
import re

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
            self.search_space.save_arch(model, str(dir_path / f'{model["archid"]}'))
 
