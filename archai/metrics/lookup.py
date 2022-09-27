from pathlib import Path
from typing import Optional, Dict, Any
from overrides import overrides
import re

import nats_bench

from archai.metrics.base import BaseMetric
from archai.nas.arch_meta import ArchWithMetaData
from archai.datasets.dataset_provider import DatasetProvider


class NatsBenchMetric(BaseMetric):
    def __init__(self, natsbench_location: str, dataset_name: str,
                 metric_name: str, higher_is_better: bool,
                 epochs: Optional[int] = None, search_space_type: str = 'tss',
                 raise_not_found: bool = True, 
                 more_info_kwargs: Optional[Dict[str, Any]] = None,
                 cost_info_kwargs: Optional[Dict[str, Any]] = None):
        
        self.natsbench_location = Path(natsbench_location)
        assert self.natsbench_location.exists(), \
            'The provided path to `natsbench_location` does not exist'
        
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.dataset_name = dataset_name
        assert dataset_name in ['cifar10', 'cifar100', 'ImageNet16-120'], \
            "`dataset_name` must be one of ['cifar10', 'cifar100', 'ImageNet16-120']"

        self.epochs = epochs
        self.search_space_type = search_space_type
        assert search_space_type in ['tss', 'sss'], \
            "`search_space_type` must be one of ['tss', 'sss']"
    
        self.archid_pattern = re.compile(f'natsbench-{self.search_space_type}-([0-9]+)')
        self.api = nats_bench.create(
            natsbench_location, search_space_type, fast_mode=True, verbose=False
        )

        self.raise_not_found = raise_not_found
        self.more_info_kwargs = more_info_kwargs or dict()
        self.cost_info_kwargs = cost_info_kwargs or dict()

    @overrides
    def compute(self, arch: ArchWithMetaData, dataset: DatasetProvider) -> Optional[float]:
        archid = arch.metadata['archid']
        natsbench_id = self.archid_pattern.match(archid)

        if not natsbench_id:
            if self.raise_not_found:
                raise ValueError(
                    f'Architecture {archid} does not belong to the NatsBench search space. '
                    'Please refer to `archai.search_spaces.discrete.NatsbenchSearchSpace` to '
                    'use the Natsbench search space.'
                )
            
            return None
        
        info = self.api.get_more_info(
            int(natsbench_id.group(1)), dataset=self.dataset_name,
            iepoch=self.epochs, **self.more_info_kwargs
        )

        info.update(self.api.get_cost_info(
            int(natsbench_id.group(1)), dataset=self.dataset_name,
            **self.cost_info_kwargs
        ))

        if self.metric_name not in info:
            raise KeyError(
                f'`metric_name` {self.metric_name} not found. Available metrics = {str(list(info.keys()))}'
            )

        return info[self.metric_name]

