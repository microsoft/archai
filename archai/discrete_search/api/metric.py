# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from abc import abstractmethod
from typing import List, Optional
from overrides import EnforceOverrides

from archai.discrete_search.api.model import NasModel
from archai.discrete_search.api.dataset import DatasetProvider


class Metric(EnforceOverrides):
    higher_is_better: bool = False

    @abstractmethod
    def compute(self, arch: NasModel, dataset: DatasetProvider,
                budget: Optional[float] = None) -> float:
        pass

    def __neg__(self) -> 'Metric':
        neg_metric = copy.deepcopy(self)
        neg_metric.higher_is_better = not neg_metric.higher_is_better
        return neg_metric


class AsyncMetric(EnforceOverrides):
    higher_is_better: bool = False

    @abstractmethod
    def send(self, arch: NasModel, dataset: DatasetProvider,
             budget: Optional[float] = None) -> None:
        pass

    @abstractmethod
    def fetch_all(self) -> List[Optional[float]]:
        ''' Fetch results from all previous `.send` calls and resets the state. '''
        pass

    def __neg__(self) -> 'AsyncMetric':
        neg_metric = copy.deepcopy(self)
        neg_metric.higher_is_better = not neg_metric.higher_is_better
        return neg_metric
