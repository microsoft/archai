# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union, Optional, Tuple, List, Dict, Callable

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.objective import Objective, AsyncObjective
from archai.discrete_search.api.dataset import DatasetProvider

import numpy as np
from tqdm import tqdm


class SearchObjectives():
    def __init__(self, cache_objective_evaluation: bool = True) -> None:
        """Class used by search algorithms to evaluate and cache search objectives and constraints.
        
        Search objectives can be either registered as `cheap` or `expensive`. This distinction is used
        to signal search algorithms on how to optimize their execution. Objectives labeled as 'cheap' can 
        be evaluated multiple times by a search algorithm to look for suitable candidate architectures, while expensive
        objectives will be evaluated much less frequently. NAS algorithms based on surrogate models
        will typically only predict the value of expensive objectives and evaluate cheap objectives
        directly.

        To see how this distinction is used by each search algorithm, please refer to the documentation
        of the search algorithm.

        Additionally to objectives, this class can also hold search constraints. Constraints are used to
        filter out candidate architectures that do not satisfy a given constraint (e.g number of parameters, FLOPs, etc.).
        Constraints are tipically evaluated multiple times by search algorithms to look for suitable candidate and
        should not be computationally expensive to evaluate.

        Args:
            cache_objective_evaluation (bool, optional): If True, objective evaluations are cached.
                Defaults to True.
        """        
        self.cache_objective_evaluation = cache_objective_evaluation

        self.cheap_objs = {}
        self.exp_objs = {}
        self.extra_constraints = {}

        # Cache key: (obj_name, archid, dataset provider name, budget)
        self.cache: Dict[Tuple[str, str, str, Optional[float]], Optional[float]] = {}

    @property
    def objs(self) -> Dict[str, Dict]:
        return dict(self.cheap_objs, **self.exp_objs)

    @property
    def objs_and_constraints(self) -> Dict[str, Dict]:
        return dict(self.objs, **self.extra_constraints)

    def add_cheap_objective(self, name: str, objective: Union[Objective, AsyncObjective],
                            higher_is_better: bool, constraint: Optional[Tuple[float, float]] = None) -> None:
        assert isinstance(objective, (AsyncObjective, Objective))
        assert name not in dict(self.objs, **self.extra_constraints),\
            f'There is already an objective or constraint named {name}.'

        self.cheap_objs[name] = {
            'objective': objective,
            'higher_is_better': higher_is_better,
            'constraint': constraint
        }
    
    def add_expensive_objective(self, name: str, objective: Union[Objective, AsyncObjective],
                                higher_is_better: bool) -> None:
        assert isinstance(objective, (AsyncObjective, Objective))
        assert name not in dict(self.objs, **self.extra_constraints),\
            f'There is already an objective or constraint named {name}.'

        self.exp_objs[name] = {
            'objective': objective,
            'higher_is_better': higher_is_better
        }

    def add_extra_constraint(self, name: str, constraint_fn: Union[Objective, AsyncObjective],
                             constraint: Tuple[float, float]):
        assert isinstance(constraint_fn, (AsyncObjective, Objective))
        assert name not in dict(self.extra_constraints, **self.objs),\
            f'There is already an objective or constraint named {name}.'

        self.extra_constraints[name] = {
            'objective': constraint_fn,
            'constraint': constraint,
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
        if not objs or not models:
            return {}
        
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

        assert all(len(dataset_providers) == len(models) == len(b) for b in budgets.values())

        # Initializes evaluation results with cached results
        eval_results = {
            obj_name: [
                self.cache.get(
                    (obj_name, model.archid, data.__class__.__name__, budget)
                )
                for model, data, budget in zip(models, dataset_providers, budgets[obj_name])
            ]
            for obj_name in objs
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
                'Received a different amount of results than expected.'
            
            for result_i, eval_i in enumerate(eval_indices[obj_name]):
                eval_results[obj_name][eval_i] = results[result_i]
        
        # Updates cache
        if self.cache_objective_evaluation:
            for obj_name in objs:
                for i in eval_indices[obj_name]:
                    cache_tuple = (
                        obj_name, models[i].archid,
                        dataset_providers[i].__class__.__name__,
                        budgets[obj_name][i]
                    )

                    self.cache[cache_tuple] = eval_results[obj_name][i]

        assert len(set(len(r) for r in eval_results.values())) == 1

        return {
            obj_name: np.array(obj_results, dtype=np.float64)
            for obj_name, obj_results in eval_results.items()
        }

    def _get_valid_arch_indices(self, objs_or_constraints: Dict[str, Dict], results: Dict[str, np.ndarray]) -> np.ndarray:
        eval_lens = {len(r) for r in results.values()}
        assert len(eval_lens) == 1

        if list(eval_lens)[0] == 0:
            return np.array([])
        
        valid_mask = np.logical_and.reduce([
            (obj_r >= objs_or_constraints[obj_name]['constraint'][0]) &\
            (obj_r <= objs_or_constraints[obj_name]['constraint'][1])
            for obj_name, obj_r in results.items()
        ])

        return np.where(valid_mask)[0]

    def eval_constraints(self, models: List[ArchaiModel],
                         dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                         progress_bar: bool = False
                         ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Evaluate constraints for a list of models and return a tuple of
        results and valid model indices.

        Args:
            models (List[ArchaiModel]): List of models to evaluate.
            dataset_providers (Union[DatasetProvider, List[DatasetProvider]]): Dataset provider or list of dataset providers.
            progress_bar (bool, optional): Whether to show progress bar. Defaults to False.

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray]: Tuple of results and indices of valid models.
        """        
        # Gets all constraints from cheap_objectives and extra_constraints
        constraints = dict(
            self.extra_constraints,
            **self._filter_objs(self.cheap_objs, 'constraint', lambda x: x is not None)
        )

        if not constraints:
            return {}, np.arange(len(models))

        eval_results = self._eval_objs(
            constraints, models,
            dataset_providers,
            budgets=None, progress_bar=progress_bar
        )

        return eval_results, self._get_valid_arch_indices(constraints, eval_results)

    def check_model_valid(self, model: ArchaiModel, dataset_provider: DatasetProvider) -> bool:
        """Checks if a model is valid.

        Args:
            model (ArchaiModel): Model to check.
            dataset_provider (DatasetProvider): Dataset provider.

        Returns:
            bool: True if model is valid, False otherwise.
        """        
        _, idx = self.eval_constraints([model], dataset_provider, progress_bar=False)
        return len(idx) > 0

    def eval_cheap_objs(self, models: List[ArchaiModel], 
                        dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                        budgets: Optional[Dict[str, List]] = None,
                        progress_bar: bool = False) -> Dict[str, np.ndarray]:
        """Evaluates all registered cheap objectives for a list of models and
        dataset provider(s).

        Args:
            models (List[ArchaiModel]): List of models to evaluate.
            
            dataset_providers (Union[DatasetProvider, List[DatasetProvider]]): Dataset provider or 
                list of dataset providers.
            
            budgets (Optional[Dict[str, List]], optional): Budgets for each objective. Defaults to None.
            
            progress_bar (bool, optional): Weather to show progress bar. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Dictionary with evaluation results.
        """        
        return self._eval_objs(
            self.cheap_objs, models, dataset_providers, budgets, progress_bar
        )

    def eval_expensive_objs(self, models: List[ArchaiModel], 
                            dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                            budgets: Optional[Dict[str, List]] = None,
                            progress_bar: bool = False) -> Dict[str, np.ndarray]:
        """Evaluates all registered expensive objectives for a list of models and
        dataset provider(s).

        Args:
            models (List[ArchaiModel]): List of models to evaluate.
            
            dataset_providers (Union[DatasetProvider, List[DatasetProvider]]): Dataset provider or 
                list of dataset providers.

            budgets (Optional[Dict[str, List]], optional): Budgets for each objective.
                Defaults to None.

            progress_bar (bool, optional): Weather to show progress bar. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Dictionary with evaluation results.
        """        
        return self._eval_objs(
            self.exp_objs, models, dataset_providers, budgets, progress_bar
        )

    def eval_all_objs(self, models: List[ArchaiModel], 
                      dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
                      budgets: Optional[Dict[str, List]] = None,
                      progress_bar: bool = False) -> Dict[str, np.ndarray]:
        """Evaluates all registered objectives for a list of models and
        dataset provider(s).

        Args:
            models (List[ArchaiModel]): List of models to evaluate.
            
            dataset_providers (Union[DatasetProvider, List[DatasetProvider]]): Dataset provider or
                list of dataset providers.

            budgets (Optional[Dict[str, List]], optional): Budgets for each objective.
                Defaults to None.
            
            progress_bar (bool, optional): Weather to show progress bar. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Dictionary with evaluation results.
        """        
        return self._eval_objs(
            dict(self.exp_objs, **self.cheap_objs), models, dataset_providers,
            budgets, progress_bar
        )
