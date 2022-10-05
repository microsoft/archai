from pathlib import Path
from typing import Optional, Dict, Any
from overrides import overrides
import re

import nats_bench

from archai.metrics.base import BaseMetric
from archai.nas.arch_meta import ArchWithMetaData
from archai.search_spaces.discrete.natsbench_tss.search_space import NatsbenchTssSearchSpace
from archai.datasets.dataset_provider import DatasetProvider


class NatsBenchMetric(BaseMetric):
    def __init__(self, search_space: NatsbenchTssSearchSpace,
                 metric_name: str, higher_is_better: bool,
                 epochs: Optional[int] = None,
                 raise_not_found: bool = True, 
                 more_info_kwargs: Optional[Dict[str, Any]] = None,
                 cost_info_kwargs: Optional[Dict[str, Any]] = None):
        assert isinstance(search_space, NatsbenchTssSearchSpace), \
            'This metric only works with architectures from NatsbenchTssSearchSpace'

        self.search_space = search_space
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.epochs = epochs
    
        self.archid_pattern = re.compile(f'natsbench-tss-([0-9]+)')
        self.api = nats_bench.create(
            str(self.search_space.natsbench_location),
            'tss', fast_mode=True, verbose=False
        )

        self.raise_not_found = raise_not_found
        self.more_info_kwargs = more_info_kwargs or dict()
        self.cost_info_kwargs = cost_info_kwargs or dict()

    @overrides
    def compute(self, arch: ArchWithMetaData, dataset: DatasetProvider,
                budget: Optional[float] = None) -> Optional[float]:
        archid = arch.metadata['archid']
        natsbench_id = self.archid_pattern.match(archid)
        budget = int(budget) if budget else budget

        if not natsbench_id:
            if self.raise_not_found:
                raise ValueError(
                    f'Architecture {archid} does not belong to the NatsBench search space. '
                    'Please refer to `archai.search_spaces.discrete.NatsbenchSearchSpace` to '
                    'use the Natsbench search space.'
                )
            
            return None
        
        info = self.api.get_more_info(
            int(natsbench_id.group(1)), dataset=self.search_space.base_dataset,
            iepoch=budget or self.epochs, **self.more_info_kwargs
        )

        info.update(self.api.get_cost_info(
            int(natsbench_id.group(1)), dataset=self.search_space.base_dataset,
            **self.cost_info_kwargs
        ))

        if self.metric_name not in info:
            raise KeyError(
                f'`metric_name` {self.metric_name} not found. Available metrics = {str(list(info.keys()))}'
            )

        return info[self.metric_name]

