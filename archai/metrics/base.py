import copy
from abc import abstractmethod
from typing import List, Optional
from overrides import EnforceOverrides

from archai.nas.nas_model import NasModel
from archai.datasets.dataset_provider import DatasetProvider


class BaseMetric(EnforceOverrides):
    higher_is_better: bool = False

    @abstractmethod
    def compute(self, arch: NasModel, dataset: DatasetProvider,
                budget: Optional[float] = None) -> float:
        pass

    def __neg__(self) -> 'BaseMetric':
        neg_metric = copy.deepcopy(self)
        neg_metric.higher_is_better = not neg_metric.higher_is_better
        return neg_metric


class BaseAsyncMetric(EnforceOverrides):
    higher_is_better: bool = False

    @abstractmethod
    def send(self, arch: NasModel, dataset: DatasetProvider,
             budget: Optional[float] = None) -> None:
        pass

    @abstractmethod
    def fetch_all(self) -> List[Optional[float]]:
        ''' Fetch results from all previous `.send` calls and resets the state. '''
        pass

    def __neg__(self) -> 'BaseAsyncMetric':
        neg_metric = copy.deepcopy(self)
        neg_metric.higher_is_better = not neg_metric.higher_is_better
        return neg_metric
