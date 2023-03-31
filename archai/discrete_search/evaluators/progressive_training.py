# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ray
from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import (
    AsyncModelEvaluator,
    ModelEvaluator,
)
from archai.discrete_search.api.search_space import DiscreteSearchSpace
from archai.common.file_utils import TemporaryFiles


def _ray_wrap_training_fn(training_fn) -> Callable:
    def _stateful_training_fn(
        arch: ArchaiModel, dataset: DatasetProvider, budget: float, training_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[ArchaiModel, float, Dict[str, Any]]:
        metric_result, training_state = training_fn(arch, dataset, budget, training_state)
        return arch, metric_result, training_state

    return _stateful_training_fn


class ProgressiveTraining(ModelEvaluator):
    """Progressive training evaluator."""

    def __init__(self, search_space: DiscreteSearchSpace, dataset: DatasetProvider, training_fn: Callable) -> None:
        """Initialize the evaluator.

        Args:
            search_space: Search space.
            training_fn: Training function.

        """

        self.search_space = search_space
        self.training_fn = training_fn
        self.dataset = dataset

        # Training state buffer (e.g optimizer state) for each architecture id
        self.training_states = {}

    @overrides
    def evaluate(self, arch: ArchaiModel, budget: Optional[float] = None) -> float:
        # Tries to retrieve previous training state
        tr_state = self.training_states.get(arch.archid, None)

        # Computes metric and updates training state
        metric_result, updated_tr_state = self.training_fn(arch, self.dataset, budget, tr_state)
        self.training_states[arch.archid] = updated_tr_state

        return metric_result


class RayProgressiveTraining(AsyncModelEvaluator):
    """Progressive training evaluator using Ray."""

    def __init__(
        self,
        search_space: DiscreteSearchSpace,
        dataset: DatasetProvider,
        training_fn: Callable,
        timeout: Optional[float] = None,
        force_stop: Optional[bool] = False,
        **ray_kwargs
    ) -> None:
        """Initialize the evaluator.

        Args:
            search_space: Search space.
            training_fn: Training function.
            timeout: Timeout (seconds) for fetching results.
            force_stop: If True, forces to stop all training jobs when fetching results.

        """

        self.search_space = search_space
        self.dataset = dataset

        if ray_kwargs:
            self.compute_fn = ray.remote(**ray_kwargs)(_ray_wrap_training_fn(training_fn))
        else:
            self.compute_fn = ray.remote(_ray_wrap_training_fn(training_fn))

        self.timeout = timeout
        self.force_stop = force_stop

        # Buffer that stores original model references from `send` calls
        # to update weights after training is complete
        self.models = []

        # Ray training job object refs
        self.results_ref = []

        # Training state buffer (e.g optimizer state) for each architecture id
        self.training_states = {}

    @overrides
    def send(self, arch: ArchaiModel, budget: Optional[float] = None) -> None:
        # Stores original model reference
        self.models.append(arch)

        current_tr_state = self.training_states.get(arch.archid, None)
        self.results_ref.append(self.compute_fn.remote(arch, self.dataset, budget, current_tr_state))

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        results = [None] * len(self.results_ref)

        # Fetchs training job results
        if not self.timeout:
            results = ray.get(self.results_ref, timeout=self.timeout)
        else:
            # Maps each object from the object_refs list to its index
            ref2idx = {ref: i for i, ref in enumerate(self.results_ref)}

            # Gets all results available within `self.timeout` seconds.
            complete_objs, incomplete_objs = ray.wait(
                self.results_ref, timeout=self.timeout, num_returns=len(self.results_ref)
            )
            partial_results = ray.get(complete_objs)

            for ref, result in zip(complete_objs, partial_results):
                results[ref2idx[ref]] = result

            for incomplete_obj in incomplete_objs:
                ray.cancel(incomplete_obj, force=self.force_stop)

        # Gathers metrics and syncs local references
        metric_results = []

        for job_id, job_results in enumerate(results):
            if job_results:
                trained_model, job_metric, training_state = job_results

                # Syncs model weights
                # On windows you cannot open a named temporary file a second time.
                temp_file_name = None
                with TemporaryFiles() as tmp:
                    temp_file_name = tmp.get_temp_file()
                    self.search_space.save_model_weights(trained_model, temp_file_name)
                    self.search_space.load_model_weights(self.models[job_id], temp_file_name)

                # Syncs training state
                self.training_states[trained_model.archid] = training_state

                metric_results.append(job_metric)

        # Resets model and job buffers
        self.models = []
        self.results_ref = []

        return metric_results
