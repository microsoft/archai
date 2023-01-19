# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from tqdm import tqdm

from archai.api.archai_model import ArchaiModel
from archai.api.dataset_provider import DatasetProvider
from archai.api.model_evaluator import AsyncModelEvaluator, ModelEvaluator


class SearchObjectives:
    def __init__(self, cache_objective_evaluation: bool = True) -> None:
        """Class used to create, evaluate and cache search objectives and constraints for search algorithms.

        Besides objectives, this class also supports registering search constraints, which are used to filter out candidate
        architectures that do not meet certain criteria (e.g., number of parameters, FLOPs). Constraints are typically evaluated
        multiple times by search algorithms and should not be computationally expensive to evaluate.

        Args:
            cache_objective_evaluation (bool, optional): If `True`, objective evaluations are cached.
                using the tuple `(obj_name, archid, dataset_provider_name, budget)` as key. Defaults to `True`.
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

    def add_objective(
        self,
        name: str,
        model_evaluator: Union[ModelEvaluator, AsyncModelEvaluator],
        higher_is_better: bool,
        compute_intensive: bool = True,
        constraint: Optional[Tuple[float, float]] = None,
    ):
        """Adds an objective function to the `SearchObjectives` object.

        Args:
            name (str): The name of the objective.
            model_evaluator (Union[ModelEvaluator, AsyncModelEvaluator]): The model evaluator responsible
                for evaluating the objective.

            higher_is_better (bool): Weather the objective should be maximized (`True`) or minimized (`False`).

            compute_intensive (bool, optional): If `True`, the objective is considered computationally expensive and
                will be estimated using surrogate models when possible. Defaults to `True`.

            constraint (Optional[Tuple[float, float]], optional): Optional objective constraint used to filter
                out candidate architectures. Expects `(lower_bound, upper_bound)` tuple. Can only be set
                if `compute_intensive` is set to `False`. Defaults to None.
        """
        assert isinstance(model_evaluator, (ModelEvaluator, AsyncModelEvaluator))
        assert name not in dict(
            self.objs, **self.extra_constraints
        ), f"There is already an objective or constraint named {name}."

        obj = {"evaluator": model_evaluator, "higher_is_better": higher_is_better, "constraint": constraint}

        if compute_intensive:
            assert constraint is None, "Constraints can only be set for cheap objectives (compute_intensive=False)."
            self.exp_objs[name] = obj
        else:
            self.cheap_objs[name] = obj

    def add_constraint(
        self, name: str, model_evaluator: Union[ModelEvaluator, AsyncModelEvaluator], constraint: Tuple[float, float]
    ):
        """Adds a search constraint to the `SearchObjectives` object. Constraints are typically evaluated
        multiple times by search algorithms to validate candidate architectures and should not be computationally
        expensive to evaluate.

        Args:
            name (str): The name of the constraint.
            model_evaluator (Union[ModelEvaluator, AsyncModelEvaluator]): The model evaluator responsible
                for evaluating the constraint.

            constraint (Tuple[float, float]): The valid range of the constraint. Expects
                a `(lower_bound, upper_bound)` tuple.
        """
        assert isinstance(model_evaluator, (ModelEvaluator, AsyncModelEvaluator))
        assert name not in dict(
            self.extra_constraints, **self.objs
        ), f"There is already an objective or constraint named {name}."

        self.extra_constraints[name] = {
            "evaluator": model_evaluator,
            "constraint": constraint,
        }

    def _filter_objs(self, objs: Dict[str, Dict], field_name: str, query_fn: Callable) -> Dict[str, Dict]:
        return {obj_name: obj_dict for obj_name, obj_dict in objs.items() if query_fn(obj_dict[field_name])}

    def _eval_objs(
        self,
        objs: Dict[str, Dict],
        models: List[ArchaiModel],
        dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
        budgets: Optional[Dict[str, List]] = None,
        progress_bar: bool = False,
    ) -> Dict[str, np.ndarray]:
        if not objs or not models:
            return {}

        # Sets `None` budget for objectives not specified in `budgets`
        budgets = budgets or {}
        budgets = {obj_name: budgets.get(obj_name, [None] * len(models)) for obj_name in objs}

        # Casts dataset_providers to a list if necessary
        if not isinstance(dataset_providers, list):
            dataset_providers = [dataset_providers] * len(models)

        # Splits `objs` in sync and async
        sync_objs = self._filter_objs(objs, "evaluator", lambda x: isinstance(x, ModelEvaluator))
        async_objs = self._filter_objs(objs, "evaluator", lambda x: isinstance(x, AsyncModelEvaluator))

        assert all(len(dataset_providers) == len(models) == len(b) for b in budgets.values())

        # Initializes evaluation results with cached results
        eval_results = {
            obj_name: [
                self.cache.get((obj_name, model.archid, data.__class__.__name__, budget))
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
                if progress_bar
                else eval_indices[obj_name]
            )

            for i in pbar:
                obj_d["evaluator"].send(models[i], dataset_providers[i], budgets[obj_name][i])

        # Calculates synchronous objectives in order
        for obj_name, obj_d in sync_objs.items():
            pbar = (
                tqdm(eval_indices[obj_name], desc=f'Calculating "{obj_name}"...')
                if progress_bar
                else eval_indices[obj_name]
            )

            for i in pbar:
                eval_results[obj_name][i] = obj_d["evaluator"].evaluate(
                    models[i], dataset_providers[i], budgets[obj_name][i]
                )

        # Gets results from async objectives
        pbar = (
            tqdm(async_objs.items(), desc="Gathering results from async objectives...")
            if progress_bar
            else async_objs.items()
        )

        for obj_name, obj_d in pbar:
            results = obj_d["evaluator"].fetch_all()

            assert len(eval_indices[obj_name]) == len(results), "Received a different amount of results than expected."

            for result_i, eval_i in enumerate(eval_indices[obj_name]):
                eval_results[obj_name][eval_i] = results[result_i]

        # Updates cache
        if self.cache_objective_evaluation:
            for obj_name in objs:
                for i in eval_indices[obj_name]:
                    cache_tuple = (
                        obj_name,
                        models[i].archid,
                        dataset_providers[i].__class__.__name__,
                        budgets[obj_name][i],
                    )

                    self.cache[cache_tuple] = eval_results[obj_name][i]

        assert len(set(len(r) for r in eval_results.values())) == 1

        return {obj_name: np.array(obj_results, dtype=np.float64) for obj_name, obj_results in eval_results.items()}

    def _get_valid_arch_indices(
        self, objs_or_constraints: Dict[str, Dict], results: Dict[str, np.ndarray]
    ) -> np.ndarray:
        eval_lens = {len(r) for r in results.values()}
        assert len(eval_lens) == 1

        if list(eval_lens)[0] == 0:
            return np.array([])

        valid_mask = np.logical_and.reduce(
            [
                (obj_r >= objs_or_constraints[obj_name]["constraint"][0])
                & (obj_r <= objs_or_constraints[obj_name]["constraint"][1])
                for obj_name, obj_r in results.items()
            ]
        )

        return np.where(valid_mask)[0]

    def validate_constraints(
        self,
        models: List[ArchaiModel],
        dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
        progress_bar: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Evaluates constraints for a list of models and returns the indices of models that satisfy all constraints.

        Args:
            models (List[ArchaiModel]): List of models to evaluate.
            dataset_providers (Union[DatasetProvider, List[DatasetProvider]]): Dataset provider or list of dataset providers.
            progress_bar (bool, optional): Whether to show progress bar. Defaults to False.

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray]: Tuple containing evaluation results
                and indices of models that satisfy all constraints.
        """
        # Gets all constraints from cheap_objectives and extra_constraints
        constraints = dict(
            self.extra_constraints, **self._filter_objs(self.cheap_objs, "constraint", lambda x: x is not None)
        )

        if not constraints:
            return {}, np.arange(len(models))

        eval_results = self._eval_objs(constraints, models, dataset_providers, budgets=None, progress_bar=progress_bar)

        return eval_results, self._get_valid_arch_indices(constraints, eval_results)

    def is_model_valid(self, model: ArchaiModel, dataset_provider: DatasetProvider) -> bool:
        """Checks if a model satisfies all constraints.

        Args:
            model (ArchaiModel): Model to check.
            dataset_provider (DatasetProvider): Dataset provider.

        Returns:
            bool: True if model is valid, False otherwise.
        """
        _, idx = self.validate_constraints([model], dataset_provider, progress_bar=False)
        return len(idx) > 0

    def eval_cheap_objs(
        self,
        models: List[ArchaiModel],
        dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
        budgets: Optional[Dict[str, List]] = None,
        progress_bar: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Evaluates all cheap objectives for a list of models and dataset provider(s).

        Args:
            models (List[ArchaiModel]): List of models to evaluate.

            dataset_providers (Union[DatasetProvider, List[DatasetProvider]]): Dataset provider or
                list of dataset providers.

            budgets (Optional[Dict[str, List]], optional): Budgets for each objective. Defaults to None.

            progress_bar (bool, optional): Weather to show progress bar. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Dictionary with evaluation results.
        """
        return self._eval_objs(self.cheap_objs, models, dataset_providers, budgets, progress_bar)

    def eval_expensive_objs(
        self,
        models: List[ArchaiModel],
        dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
        budgets: Optional[Dict[str, List]] = None,
        progress_bar: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Evaluates all expensive objective functions for a list of models and dataset provider(s).

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
        return self._eval_objs(self.exp_objs, models, dataset_providers, budgets, progress_bar)

    def eval_all_objs(
        self,
        models: List[ArchaiModel],
        dataset_providers: Union[DatasetProvider, List[DatasetProvider]],
        budgets: Optional[Dict[str, List]] = None,
        progress_bar: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Evaluates all objective functions for a list of models and dataset provider(s).

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
        return self._eval_objs(dict(self.exp_objs, **self.cheap_objs), models, dataset_providers, budgets, progress_bar)

    def save_cache(self, path: str) -> None:
        """Saves the state of the SearchObjectives object to a YAML file.

        Args:
            path (str): Path to save YAML file.
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.cache, f)

    def load_cache(self, path: str) -> None:
        """Loads the state of the SearchObjectives object from a YAML file.

        Args:
            path (str): Path to YAML file.
        """
        with open(path, "r", encoding="utf-8") as f:
            self.cache = yaml.load(f)
