# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers.data.data_collator import DataCollatorForLanguageModeling

from archai.nlp.datasets_v2 import distributed_utils

class ArchaiIterator(DataLoader):
    """
    """

    def __init__(self, dataset, collate_fn=None, batch_size=None):
        self.world_size = distributed_utils.get_world_size()
        self.rank = distributed_utils.get_rank()

        print(self.world_size, self.rank)

        sampler = DistributedSampler(dataset,
                                     num_replicas=self.world_size,
                                     rank=self.rank,
                                     shuffle=False)

        def collate(examples):
            input_ids =  torch.stack([torch.tensor(e['input_ids'][:-1], dtype=torch.long) for e in examples], dim=0)
            labels = torch.stack([torch.tensor(e['input_ids'][1:], dtype=torch.long) for e in examples], dim=0)
            return input_ids, labels, 192, False

        super().__init__(dataset,
                         collate_fn=collate,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0,
                         sampler=sampler,
                         pin_memory=True)
