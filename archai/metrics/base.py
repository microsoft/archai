from abc import abstractmethod
from overrides import EnforceOverrides

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider


class BaseMetric(EnforceOverrides):
    higher_is_better: bool = False

    @abstractmethod
    def compute(self, arch: ArchWithMetaData, dataset: DatasetProvider) -> float:
        pass
