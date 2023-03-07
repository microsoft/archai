# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from tqdm import tqdm

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import (
    AsyncModelEvaluator,
    ModelEvaluator,
)


class SearchConstraint:
    def __init__(self, name, evaluator, constraint):
        self.name = name
        self.evaluator = evaluator
        self.constraint = constraint


class SearchObjective:
    def __init__(self, name, model_evaluator, higher_is_better, compute_intensive, constraint):
        self.name = name
        self.evaluator = model_evaluator
        self.higher_is_better = higher_is_better
        self.compute_intensive = compute_intensive
        self.constraint = constraint


class SearchObjectives:
    """Search objectives and constraints."""

    def __init__(self, cache_objective_evaluation: Optional[bool] = True) -> None:
        """Create, evaluate and cache search objectives and constraints for search algorithms.

        Besides objectives, this class also supports registering search constraints,
        which are used to filter out candidate architectures that do not meet certain
        criteria (e.g., number of parameters, FLOPs). Constraints are typically evaluated
        multiple times by search algorithms and should not be computationally expensive to
        evaluate.

        Args:
            cache_objective_evaluation: If `True`, objective evaluations are cached using the
                tuple `(obj_name, archid, dataset_provider_name, budget)` as key.

        """

        self._cache_objective_evaluation = cache_objective_evaluation

        self._objs = {}
        self._extra_constraints = {}

        # Cache key: (obj_name, archid, budget)
        self._cache: Dict[Tuple[str, str, Optional[float]], Optional[float]] = {}

    @property
    def objective_names(self) -> List[str]:
        """Return a list of all objective names."""
        return list(self._objs.keys())

    @property
    def cheap_objective_names(self) -> List[str]:
        """Return a list of cheap objective names."""
        return list(self.cheap_objectives.keys())

    @property
    def expensive_objective_names(self) -> List[str]:
        """Return a list of expensive objective names."""
        return list(self.expensive_objectives.keys())

    @property
    def objectives(self) -> Dict[str, SearchObjective]:
        """Return a dictionary of all objectives."""
        return self._objs

    @property
    def cheap_objectives(self) -> Dict[str, SearchObjective]:
        """Return a dictionary of cheap objectives."""
        return self._filter_objs(self._objs, lambda x: not x.compute_intensive)

    @property
    def expensive_objectives(self) -> Dict[str, SearchObjective]:
        """Return a dictionary of expensive objectives."""
        return self._filter_objs(self._objs, lambda x: x.compute_intensive)

    @property
    def constraints(self) -> Dict[str, SearchConstraint]:
        """Return a dictionary of all the constraints."""

        return self._extra_constraints

    def add_objective(
        self,
        name: str,
        model_evaluator: Union[ModelEvaluator, AsyncModelEvaluator],
        higher_is_better: bool,
        compute_intensive: Optional[bool] = True,
        constraint: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Add an objective function to the `SearchObjectives` object.

        Args:
            name: The name of the objective.
            model_evaluator: The model evaluator responsible for evaluating the objective.
            higher_is_better: Whether the objective should be maximized (`True`) or minimized (`False`).
            compute_intensive: If `True`, the objective is considered computationally expensive
                and will be estimated using surrogate models when possible.
            constraint: Objective constraint used to filter out candidate architectures.
                Expects `(lower_bound, upper_bound)` tuple. Can only be set if
                `compute_intensive` is set to `False`.

        """

        assert isinstance(model_evaluator, (ModelEvaluator, AsyncModelEvaluator))
        assert name not in self._objs, f"There is already an objective {name}."
        assert name not in self._extra_constraints, f"There is already an constraint named {name}."

        obj = SearchObjective(name, model_evaluator, higher_is_better, compute_intensive, constraint)

        if compute_intensive:
            assert constraint is None, "Constraints can only be set for cheap objectives (compute_intensive=False)."

        self._objs[name] = obj

    def add_constraint(
        self, name: str, model_evaluator: Union[ModelEvaluator, AsyncModelEvaluator], constraint: Tuple[float, float]
    ) -> None:
        """Add a search constraint to the `SearchObjectives` object.

        Constraints are typically evaluated multiple times by search algorithms to validate
        candidate architectures and should not be computationally expensive to evaluate.

        Args:
            name: The name of the constraint.
            model_evaluator: The model evaluator responsible for evaluating the constraint.
            constraint: The valid range of the constraint. Expects a `(lower_bound, upper_bound)`
                tuple.

        """

        assert isinstance(model_evaluator, (ModelEvaluator, AsyncModelEvaluator))
        assert name not in self._objs, f"There is already an objective {name}."
        assert name not in self._extra_constraints, f"There is already an constraint named {name}."

        self._extra_constraints[name] = SearchConstraint(name, model_evaluator, constraint)

    def _filter_objs(self, objs: Dict[str, Dict], query_fn: Callable) -> Dict[str, Dict]:
        return {obj_name: obj_dict for obj_name, obj_dict in objs.items() if query_fn(obj_dict)}

    def _eval_objs(
        self,
        objs: Dict[str, Dict],
        models: List[ArchaiModel],
        budgets: Optional[Dict[str, List[Any]]] = None,
        progress_bar: Optional[bool] = False,
    ) -> Dict[str, np.ndarray]:
        if not objs or not models:
            return {}

        # Sets `None` budget for objectives not specified in `budgets`
        budgets = budgets or {}
        budgets = {obj_name: budgets.get(obj_name, [None] * len(models)) for obj_name in objs}

        # Splits `objs` in sync and async
        sync_objs = self._filter_objs(objs, lambda x: isinstance(x.evaluator, ModelEvaluator))
        async_objs = self._filter_objs(objs, lambda x: isinstance(x.evaluator, AsyncModelEvaluator))

        # Initializes evaluation results with cached results
        eval_results = {
            obj_name: [
                self._cache.get((obj_name, model.archid, budget))
                for model, budget in zip(models, budgets[obj_name])
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
                obj_d.evaluator.send(models[i], budgets[obj_name][i])

        # Calculates synchronous objectives in order
        for obj_name, obj_d in sync_objs.items():
            pbar = (
                tqdm(eval_indices[obj_name], desc=f'Calculating "{obj_name}"...')
                if progress_bar
                else eval_indices[obj_name]
            )

            for i in pbar:
                eval_results[obj_name][i] = obj_d.evaluator.evaluate(
                    models[i], budgets[obj_name][i]
                )

        # Gets results from async objectives
        pbar = (
            tqdm(async_objs.items(), desc="Gathering results from async objectives...")
            if progress_bar
            else async_objs.items()
        )

        for obj_name, obj_d in pbar:
            results = obj_d.evaluator.fetch_all()

            assert len(eval_indices[obj_name]) == len(results), "Received a different amount of results than expected."

            for result_i, eval_i in enumerate(eval_indices[obj_name]):
                eval_results[obj_name][eval_i] = results[result_i]

        # Updates cache
        if self._cache_objective_evaluation:
            for obj_name in objs:
                for i in eval_indices[obj_name]:
                    cache_tuple = (
                        obj_name,
                        models[i].archid,
                        budgets[obj_name][i],
                    )

                    self._cache[cache_tuple] = eval_results[obj_name][i]

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
                (obj_r >= objs_or_constraints[obj_name].constraint[0])
                & (obj_r <= objs_or_constraints[obj_name].constraint[1])
                for obj_name, obj_r in results.items()
            ]
        )

        return np.where(valid_mask)[0]

    def validate_constraints(
        self,
        models: List[ArchaiModel],
        progress_bar: Optional[bool] = False,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Evaluate constraints for a list of models and returns the indices of models that
        satisfy all constraints.

        Args:
            models: List of models to evaluate.
            progress_bar: Whether to show progress bar.

        Returns:
            Evaluation results and indices of models that satisfy all constraints.

        """

        # Gets all constraints from cheap_objectives that have constraints and extra_constraints
        constraints = dict(
            self._extra_constraints, **self._filter_objs(self.cheap_objectives, lambda x: x.constraint is not None)
        )

        if not constraints:
            return {}, np.arange(len(models))

        eval_results = self._eval_objs(constraints, models, budgets=None, progress_bar=progress_bar)

        return eval_results, self._get_valid_arch_indices(constraints, eval_results)

    def is_model_valid(self, model: ArchaiModel) -> bool:
        """Check if a model satisfies all constraints.

        Args:
            model: Model to check.

        Returns:
            `True` if model is valid, `False` otherwise.

        """

        _, idx = self.validate_constraints([model], progress_bar=False)
        return len(idx) > 0

    def eval_cheap_objs(
        self,
        models: List[ArchaiModel],
        budgets: Optional[Dict[str, List]] = None,
        progress_bar: Optional[bool] = False,
    ) -> Dict[str, np.ndarray]:
        """Evaluate all cheap objectives for a list of models.

        Args:
            models: List of models to evaluate.
            budgets: Budgets for each objective.
            progress_bar: Whether to show progress bar.

        Returns:
            Dictionary with evaluation results.

        """
        return self._eval_objs(self.cheap_objectives, models, budgets, progress_bar)

    def eval_expensive_objs(
        self,
        models: List[ArchaiModel],
        budgets: Optional[Dict[str, List]] = None,
        progress_bar: Optional[bool] = False,
    ) -> Dict[str, np.ndarray]:
        """Evaluate all expensive objective functions for a list of models.

        Args:
            models: List of models to evaluate.
            budgets: Budgets for each objective.
            progress_bar: Whether to show progress bar. Defaults to False.

        Returns:
            Dictionary with evaluation results.

        """

        return self._eval_objs(self.expensive_objectives, models, budgets, progress_bar)

    def eval_all_objs(
        self,
        models: List[ArchaiModel],
        budgets: Optional[Dict[str, List]] = None,
        progress_bar: Optional[bool] = False,
    ) -> Dict[str, np.ndarray]:
        """Evaluate all objective functions for a list of models.

        Args:
            models: List of models to evaluate.
            budgets: Budgets for each objective.
            progress_bar: Whether to show progress bar.

        Returns:
            Dictionary with evaluation results.

        """

        return self._eval_objs(self._objs, models, budgets, progress_bar)

    def save_cache(self, file_path: str) -> None:
        """Save the state of the `SearchObjectives` object to a YAML file.

        Args:
            file_path: Path to save YAML file.

        """

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self._cache, f)

    def load_cache(self, file_path: str) -> None:
        """Load the state of the `SearchObjectives` object from a YAML file.

        Args:
            file_path: Path to YAML file.

        """

        with open(file_path, "r", encoding="utf-8") as f:
            self._cache = yaml.load(f)

    def lookup_cache(self, obj_name: str, arch_id: str, budget: int) -> Optional[float]:
        """Look up the cache for a specific objective, architecture and budget.

        Args:
            obj_name: Name of objective.
            arch_id: Architecture ID.
            budget: Budget.

        Returns:
            Evaluation result if found in cache, `None` otherwise.

        """

        return self._cache.get((obj_name, arch_id, budget), None)