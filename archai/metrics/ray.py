from typing import List, Optional

import ray
from overrides import overrides

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider
from archai.metrics.base import BaseMetric, BaseAsyncMetric


def wrap_metric_calculate(class_method):
    def calculate(arch: ArchWithMetaData, dataset: DatasetProvider):
        return class_method(arch, dataset)
    return calculate


class RayParallelMetric(BaseAsyncMetric):
    def __init__(self, metric: BaseMetric, timeout: Optional[float] = None,  **ray_kwargs):
        """Wraps a synchronous `BaseMetric` as an asynchronous metric to be computed in parallel using Ray.
        `RayParallelMetric` expects a stateless metric, meaning that any `metric.compute` call cannot alter 
        the state of `metric` in any way.

        Args:
            metric (BaseMetric): A metric object
            timeout (Optional[float], optional): Timeout for `receive_all`. Jobs not finished after the time limit
            are canceled and returned as None. Defaults to None.
        """        
        assert isinstance(metric, BaseMetric)

        # Wraps metric.calculate as a standalone function. This only works with stateless metrics
        self.compute_fn = ray.remote(**ray_kwargs)(wrap_metric_calculate(metric.compute))
        self.higher_is_better = metric.higher_is_better
        self.timeout = timeout
        self.object_refs = []

    @overrides
    def send(self, arch: ArchWithMetaData, dataset: DatasetProvider) -> None:
        self.object_refs.append(self.compute_fn.remote(arch, dataset))

    @overrides
    def receive_all(self) -> List[float]:
        ''' Receives results from all previous `.send` calls and resets state. '''
        results = ray.get(self.object_refs, timeout=self.timeout)
        self.object_refs = []
        return results
