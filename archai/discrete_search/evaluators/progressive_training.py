# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ray
import torch
from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import (
    AsyncModelEvaluator, ModelEvaluator, ProgressiveModelEvaluator
)
from archai.discrete_search.api.search_space import DiscreteSearchSpace


def _ray_wrap_training_fn(training_fn) -> Callable:
    def _stateful_training_fn(
        arch: ArchaiModel, dataset: DatasetProvider, budget: float, training_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[ArchaiModel, float, Dict[str, Any]]:
        metric_result, training_state = training_fn(arch, dataset, budget, training_state)
        return arch, metric_result, training_state

    return _stateful_training_fn


class ProgressiveTraining(ModelEvaluator, ProgressiveModelEvaluator):
    """Progressive training evaluator."""

    def __init__(self, training_fn: Callable, output_dir: str) -> None:
        """Initialize the evaluator.

        Args:
            training_fn: Training function.

        """

        self.training_fn = training_fn
        self.output_dir = Path(output_dir)

        assert not self.output_dir.is_file(), f'Output directory {self.output_dir} is a file'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state buffer (e.g optimizer state) for each architecture id
        self.training_states = {}

    @overrides
    def save_checkpoint(self) -> str:
        for archid, tr_state in self.training_states.items():
            torch.save(tr_state, self.output_dir / f'{archid}.pt')
        
        return str(self.output_dir)

    @overrides
    def load_checkpoint(self, checkpoint: str) -> None:
        checkpoint_dir = Path(checkpoint)
        assert checkpoint_dir.is_dir(), f'Checkpoint {checkpoint} is not a directory'

        for checkpoint_file in checkpoint_dir.glob('*.pt'):
            archid = checkpoint_file.stem
            tr_state = torch.load(checkpoint_file)
            self.training_states[archid] = tr_state        

    @overrides
    def evaluate(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> float:
        # Tries to retrieve previous training state
        tr_state = self.training_states.get(arch.archid, None)

        # Computes metric and updates training state
        metric_result, updated_tr_state = self.training_fn(arch, dataset, budget, tr_state)
        self.training_states[arch.archid] = updated_tr_state

        return metric_result


class RayProgressiveTraining(AsyncModelEvaluator, ProgressiveModelEvaluator):
    """Progressive training evaluator using Ray."""

    def __init__(
        self,
        training_fn: Callable,
        timeout: Optional[float] = None,
        force_stop: Optional[bool] = False,
        **ray_kwargs
    ) -> None:
        """Initialize the evaluator.

        Args:
            training_fn: Training function.
            timeout: Timeout (seconds) for fetching results.
            force_stop: If True, forces to stop all training jobs when fetching results.

        """

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
    def save_checkpoint(self) -> str:
        for archid, tr_state in self.training_states.items():
            torch.save(tr_state, self.output_dir / f'{archid}.pt')
        
        return str(self.output_dir)
    
    @overrides
    def load_checkpoint(self, checkpoint: str) -> None:
        checkpoint_dir = Path(checkpoint)
        assert checkpoint_dir.is_dir(), f'Checkpoint {checkpoint} is not a directory'

        for checkpoint_file in checkpoint_dir.glob('*.pt'):
            archid = checkpoint_file.stem
            tr_state = torch.load(checkpoint_file)
            self.training_states[archid] = tr_state

    @overrides
    def send(self, arch: ArchaiModel, dataset: DatasetProvider, budget: Optional[float] = None) -> None:
        # Stores original model reference
        self.models.append(arch)

        current_tr_state = self.training_states.get(arch.archid, None)
        self.results_ref.append(self.compute_fn.remote(arch, dataset, budget, current_tr_state))

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

                # Syncs training state
                self.training_states[trained_model.archid] = training_state
                metric_results.append(job_metric)

        # Resets model and job buffers
        self.models = []
        self.results_ref = []

        return metric_results
