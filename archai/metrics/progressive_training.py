from typing import Callable, List, Dict, Optional, Union
import tempfile

import ray
from overrides import overrides

from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider
from archai.search_spaces.discrete.base import DiscreteSearchSpaceBase
from archai.metrics.base import BaseMetric, BaseAsyncMetric


class ProgressiveTrainingMetric(BaseMetric):
    def __init__(self, search_space: DiscreteSearchSpaceBase, 
                 training_fn: Callable, higher_is_better: bool = False):
        self.search_space = search_space
        self.training_fn = training_fn
        self.higher_is_better = higher_is_better

        # Training state buffer (e.g optimizer state) for each architecture id
        self.training_states = {}

    @overrides
    def compute(self, arch: ArchWithMetaData, dataset: DatasetProvider,
                budget: Optional[float] = None) -> float:
        # Tries to retrieve previous training state
        tr_state = self.training_states.get(arch.metadata['archid'], None)

        # Computes metric and updates training state
        metric_result, updated_tr_state = self.training_fn(arch, dataset, budget, tr_state)
        self.training_states[arch.metadata['archid']] = updated_tr_state

        return metric_result

