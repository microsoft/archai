from typing import List, Optional, Union

import ray
from overrides import overrides

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider
from archai.metrics.base import BaseMetric, BaseAsyncMetric


def wrap_metric_calculate(class_method):
    def calculate(arch: ArchWithMetaData, dataset: DatasetProvider, budget: Optional[float] = None):
        return class_method(arch, dataset, budget)
    return calculate


class RayParallelMetric(BaseAsyncMetric):
    def __init__(self, metric: BaseMetric, timeout: Optional[float] = None, force_stop: bool = False, 
                 **ray_kwargs):
        """Wraps a synchronous `BaseMetric` as an asynchronous metric to be computed in parallel using Ray.
        `RayParallelMetric` expects a stateless metric, meaning that any `metric.compute` call cannot alter 
        the state of `metric` in any way.

        Args:
            metric (BaseMetric): A metric object
            timeout (Optional[float], optional): Timeout for `receive_all`. Jobs not finished after the time limit
                are canceled and returned as None. Defaults to None.
            force_stop (bool, optional): If incomplete tasks (within `timeout` seconds) should be force-killed. If 
                set to `False`, Ray will just send a `KeyboardInterrupt` signal to the process.
            **ray_kwargs: Key-value arguments for ray.remote(), e.g: num_gpus, num_cpus, max_task_retries.
        """        
        assert isinstance(metric, BaseMetric)

        # Wraps metric.calculate as a standalone function. This only works with stateless metrics
        self.compute_fn = ray.remote(**ray_kwargs)(wrap_metric_calculate(metric.compute))
        self.higher_is_better = metric.higher_is_better
        self.timeout = timeout
        self.force_stop = force_stop
        self.object_refs = []

    @overrides
    def send(self, arch: ArchWithMetaData, dataset: DatasetProvider,
             budget: Optional[float] = None) -> None:
        self.object_refs.append(self.compute_fn.remote(arch, dataset, budget))

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
                self.object_refs, timeout=self.timeout,
                num_returns=len(self.object_refs)
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
