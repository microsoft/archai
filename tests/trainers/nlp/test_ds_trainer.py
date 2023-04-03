# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest
if os.name == "nt":
    pytest.skip(allow_module_level=True)

from torch.utils.data import Dataset
from archai.trainers.nlp.ds_trainer import StatefulDistributedSampler


class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


def test_stateful_distributed_sampler():
    dataset = DummyDataset(100)

    # Assert that the correct subset of indices is returned
    sampler = StatefulDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False, total_consumed_samples=50)
    expected_indices = [i for i in range(50, 100)]
    assert list(iter(sampler)) == expected_indices

    # Assert that the correct subset of indices is returned with more than one replica
    sampler = StatefulDistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False, total_consumed_samples=80)
    expected_indices = [i for i in range(80, 100, 2)]
    assert list(iter(sampler)) == expected_indices
