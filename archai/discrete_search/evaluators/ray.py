# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, List, Optional, Union

import ray
from overrides import overrides

from archai.discrete_search.api.archai_model import ArchaiModel
from archai.discrete_search.api.model_evaluator import (
    AsyncModelEvaluator,
    ModelEvaluator,
)


def _wrap_metric_calculate(class_method) -> Callable:
    def _calculate(arch: ArchaiModel, budget: Optional[float] = None) -> Callable:
        return class_method(arch, budget)

    return _calculate


class RayParallelEvaluator(AsyncModelEvaluator):
    """Wraps a `ModelEvaluator` object into an `AsyncModelEvaluator` with parallel execution using Ray.

    `RayParallelEvaluator` expects a stateless objective function as input, meaning that any
    `ModelEvaluator.evaluate(arch, ...)` will not alter the state of `obj` or `arch` in any way.

    """

    def __init__(
        self, obj: ModelEvaluator, timeout: Optional[float] = None, force_stop: Optional[bool] = False, **ray_kwargs
    ) -> None:
        """Initialize the evaluator.

        Args:
            obj: A `ModelEvaluator` object.
            timeout: Timeout for receiving results from Ray. If None, then Ray will wait
                indefinitely for results. If timeout is reached, then incomplete tasks
                are canceled and returned as None.
            force_stop: If incomplete tasks (within `timeout` seconds) should be force-killed.
                If set to `False`, Ray will just send a `KeyboardInterrupt` signal to the process.
            **ray_kwargs: Key-value arguments for ray.remote(), e.g: num_gpus, num_cpus, max_task_retries.

        """

        assert isinstance(obj, ModelEvaluator)

        # Wraps metric.calculate as a standalone function. This only works with stateless metrics
        if ray_kwargs:
            self.compute_fn = ray.remote(**ray_kwargs)(_wrap_metric_calculate(obj.evaluate))
        else:
            self.compute_fn = ray.remote(_wrap_metric_calculate(obj.evaluate))

        self.timeout = timeout
        self.force_stop = force_stop
        self.object_refs = []

    @overrides
    def send(self, arch: ArchaiModel, budget: Optional[float] = None) -> None:
        self.object_refs.append(self.compute_fn.remote(arch, budget))

    @overrides
    def fetch_all(self) -> List[Union[float, None]]:
        results = [None] * len(self.object_refs)

        if not self.timeout:
            results = ray.get(self.object_refs, timeout=self.timeout)
        else:
            # Maps each object from the object_refs list to its index
            ref2idx = {ref: i for i, ref in enumerate(self.object_refs)}

            # Gets all results available within `self.timeout` seconds.
            complete_objs, incomplete_objs = ray.wait(
                self.object_refs, timeout=self.timeout, num_returns=len(self.object_refs)
            )
            partial_results = ray.get(complete_objs)

            # Update results with the partial results fetched
            for ref, result in zip(complete_objs, partial_results):
                results[ref2idx[ref]] = result

            # Cancels incomplete jobs
            for incomplete_obj in incomplete_objs:
                ray.cancel(incomplete_obj, force=self.force_stop)

        # Resets metric state
        self.object_refs = []

        return results
