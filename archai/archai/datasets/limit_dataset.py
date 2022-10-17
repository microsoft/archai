# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional

import torch
from torch.utils.data import \
    SubsetRandomSampler, Sampler, Subset, ConcatDataset, Dataset, random_split

class LimitDataset(Dataset):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n
        if hasattr(dataset, 'targets'):
            self.targets = dataset.targets[:n]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.dataset[i]

DatasetLike = Optional[Union[Dataset, Subset, ConcatDataset, LimitDataset]]
