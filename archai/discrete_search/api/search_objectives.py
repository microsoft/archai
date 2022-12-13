# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union, Optional, Tuple, List, Dict, Callable

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.objective import Objective, AsyncObjective
from archai.discrete_search.api.dataset import DatasetProvider

import numpy as np
from tqdm import tqdm


class SearchObjectives():
    def __init__(self, cache_objective_evaluation: bool = True, progress_bar: bool = True) -> None:
        self.cheap_objs = {}
        self.exp_objs = {}
        self.proxy_objs = {}

        self.progress_bar = progress_bar
        self.cache_objective_evaluation = cache_objective_evaluation
        
        # Cache key: (obj_name, is_proxy, archid, dataset obj, budget)
        self.cache: Dict[Tuple[str, bool, str, DatasetProvider, Optional[float]], Optional[float]] = {}

    def add_cheap_objective(self, objective_name: str, objective: Union[Objective, AsyncObjective],
                            higher_is_better: bool,
                            constraint: Optional[Tuple[float, float]] = None) -> None:
        assert isinstance(objective, (AsyncObjective, Objective))
        assert objective_name not in dict(self.cheap_objs, **self.exp_objs),\
            f'There is already an objective named {objective_name}.'

        self.cheap_objs[objective_name] = {
            'objective': objective,
            'higher_is_better': higher_is_better,
            'constraint': constraint or [-float('-inf'), float('+inf')],
            'proxy': False
        }
    
    def add_expensive_objective(self, objective_name: str,
                                objective: Union[Objective, AsyncObjective],
                                higher_is_better: bool,
                                constraint: Optional[Tuple[float, float]] = None,
                                proxy_constraint: Optional[Tuple[Union[Objective, AsyncObjective], float, float]] = None) -> None:
        assert isinstance(objective, (AsyncObjective, Objective))
        assert objective_name not in dict(self.cheap_objs, **self.exp_objs),\
            f'There is already an objective named {objective_name}.'

        self.exp_objs[objective_name] = {
            'objective': objective,
            'higher_is_better': higher_is_better,
            'constraint': constraint or [-float('-inf'), float('+inf')],
            'proxy': False
        }

        if proxy_constraint:
            proxy_objective, *p_constraint = proxy_constraint

            self.proxy_objs[objective_name] = {
                'objective': proxy_objective,
                'higher_is_better': higher_is_better,
                'constraint': p_constraint,
                'proxy': True
            }

    def _filter_objs(self, objs: Dict[str, Dict], field_name: str, query_fn: Callable) -> Dict[str, Dict]:
        return {
            obj_name: obj_dict
            for obj_name, obj_dict in objs.items()
            if query_fn(obj_dict[field_name])
        }

    def _eval_objs(self,
                   objs: Dict[str, Dict],
                   models: List[ArchaiModel], 
                   dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                   budgets: Optional[Dict[str, List]] = None,
                   progress_bar: bool = False) -> Dict[str, np.ndarray]:
        # Sets `None` budget for objectives not specified in `budgets`
        budgets = budgets or {}
        budgets = {
            obj_name: budgets.get(obj_name, [None] * len(models))
            for obj_name in objs
        }

        # Casts dataset_providers to a list if necessary
        if not isinstance(dataset_providers, list):
            dataset_providers = [dataset_providers] * len(models)
        
        # Splits `objs` in sync and async
        sync_objs = self._filter_objs(objs, 'objective', lambda x: isinstance(x, Objective))
        async_objs = self._filter_objs(objs, 'objective', lambda x: isinstance(x, AsyncObjective))

        assert all(len(dataset_providers) == len(models) == b for b in budgets.values())

        # Initializes evaluation results with cached results
        eval_results = {
            obj_name: [
                self.cache.get(
                    (obj_name, obj_d['proxy'], model.archid, data, budget)
                )
                for model, data, budget in zip(models, dataset_providers, budgets[obj_name])
            ]
            for obj_name, obj_d in objs.items()
        }

        # Saves model indices that are not in the cache and need to be evaluated
        eval_indices = {
            obj_name: [i for i, result in enumerate(obj_results) if result is None]
            for obj_name, obj_results in eval_results.items()
        }

        # Dispatches jobs for all async objectives first
        for obj_name, obj_d in async_objs.items():
            pbar = (
                tqdm(eval_indices[obj_name], desc=f'Dispatching jobs for "{obj_name}"...')
                if progress_bar else eval_indices[obj_name]
            )

            for i in pbar:
                obj_d['objective'].send(
                    models[i], dataset_providers[i], budgets[obj_name][i]
                )

        # Calculates synchronous objectives in order
        for obj_name, obj_d in sync_objs.items():
            pbar = (
                tqdm(eval_indices[obj_name], desc=f'Calculating "{obj_name}"...')
                if progress_bar else eval_indices[obj_name]
            )

            for i in pbar:
                eval_results[obj_name][i] = obj_d['objective'].evaluate(
                    models[i], dataset_providers[i], budgets[obj_name][i]
                )

        # Gets results from async objectives
        pbar = (
            tqdm(async_objs.items(), desc=f'Gathering results from async objectives...')
            if progress_bar
            else async_objs.items()
        )

        for obj_name, obj_d in pbar:
            results = obj_d['objective'].fetch_all()
            
            assert len(eval_indices[obj_name]) == len(results), \
                'Received more results than expected.'
            
            for result_i, eval_i in enumerate(eval_indices[obj_name]):
                eval_results[obj_name][eval_i] = results[result_i]
        
        # Updates cache
        if self.cache_objective_evaluation:
            for obj_name, obj_d in objs.items():
                for i in eval_indices[obj_name]:
                    cache_tuple = (
                        obj_name, obj_d['proxy'],
                        models[i].archid, dataset_providers[i],
                        budgets[obj_name][i]
                    )

                    self.cache[cache_tuple] = eval_results[obj_name][i]

        assert len(set(len(r) for r in eval_results.values())) == 1

        return {
            obj_name: np.ndarray(obj_results, dtype=np.float64)
            for obj_name, obj_results in eval_results.items()
        }

    def _get_valid_arch_indices(self, objs: Dict[str, Dict], 
                                results: Dict[str, np.ndarray]) -> np.ndarray:
        valid_mask = np.logical_and.reduce([
            (obj_r >= objs[obj_name]['constraint'][0]) &\
            (obj_r <= objs[obj_name]['constraint'][1])
            for obj_name, obj_r in results.items()
        ])

        return np.where(valid_mask)[0]

    def eval_cheap_objs(self, models: List[ArchaiModel], 
                        dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                        budgets: Optional[Dict[str, List]] = None,
                        progress_bar: bool = False
                        ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        results = self._eval_objs(
            self.cheap_objs, models, dataset_providers, budgets, progress_bar
        )
        valid_archs = self._get_valid_arch_indices(self.cheap_objs, results)

        return results, valid_archs

    def eval_cheap_objs_and_proxies(self, models: List[ArchaiModel], 
                                    dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                                    budgets: Union[Dict[str, float], Dict[str, List[float]], None] = None,
                                    progress_bar: bool = False
                                    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        selected_objs = dict(self.cheap_objs, **self.proxy_objs)
        
        results = self._eval_objs(
            selected_objs, models, dataset_providers, budgets, progress_bar
        )
        valid_archs = self._get_valid_arch_indices(selected_objs, results)

        return results, valid_archs

    def eval_expensive_objs(self, models: List[ArchaiModel], 
                            dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                            budgets: Union[Dict[str, float], Dict[str, List[float]], None] = None,
                            progress_bar: bool = False) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        results = self._eval_objs(
            self.exp_objs, models, dataset_providers, budgets, progress_bar
        )
        valid_archs = self._get_valid_arch_indices(self.exp_objs, results)

        return results, valid_archs

    def eval_all_objs(self, models: List[ArchaiModel], 
                  dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                  budgets: Union[Dict[str, float], Dict[str, List[float]], None] = None,
                  progress_bar: bool = False):
        selected_objs = dict(self.exp_objs, **self.cheap_objs)

        results = self._eval_objs(
            selected_objs, models, dataset_providers, budgets, progress_bar
        )
        valid_archs = self._get_valid_arch_indices(selected_objs, results)

        return results, valid_archs
