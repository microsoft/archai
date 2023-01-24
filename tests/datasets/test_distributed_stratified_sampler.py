# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import random
import time
from collections import Counter

import numpy as np
from torch.utils.data import Dataset

from archai.common.config import Config
from archai.datasets.distributed_stratified_sampler import DistributedStratifiedSampler
from archai.supergraph.datasets import data


class ListDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.targets = self.y = np.array(y)
        self.transform = transform

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def _dist_no_val(rep_count: int, data_len=1000, labels_len=2, val_ratio=0.0):
    x = np.random.randint(-data_len, data_len, data_len)
    labels = np.array(range(labels_len))
    y = np.repeat(labels, math.ceil(float(data_len) / labels_len))[:data_len]
    np.random.shuffle(y)
    dataset = ListDataset(x, y)

    train_samplers, val_samplers = [], []
    for i in range(rep_count):
        train_samplers.append(
            DistributedStratifiedSampler(dataset, world_size=rep_count, rank=i, val_ratio=val_ratio, is_val=False)
        )
        val_samplers.append(
            DistributedStratifiedSampler(dataset, world_size=rep_count, rank=i, val_ratio=val_ratio, is_val=True)
        )
    tl = [list(iter(s)) for s in train_samplers]
    vl = [list(iter(s)) for s in val_samplers]

    c_l = [tli + vli for tli, vli in zip(tl, vl)]  # combile train val
    all_len = sum((len(li) for li in c_l))
    u = set(i for li in c_l for i in li)

    # verify stratification
    for vli, tli in zip(vl, tl):
        vlic = Counter(dataset.targets[vli])
        assert len(vlic.keys()) == labels_len
        assert max(vlic.values()) - min(vlic.values()) <= 2
        tlic = Counter(dataset.targets[tli])
        assert len(tlic.keys()) == labels_len
        assert max(tlic.values()) - min(tlic.values()) <= 2

    # below means all indices are equally divided between shards
    assert len(set((len(li) for li in c_l))) == 1  # all shards equal
    assert all((len(li) >= len(dataset) / rep_count for li in c_l))
    assert all((len(li) <= len(dataset) / rep_count + 1 for li in c_l))
    assert min(u) == 0
    assert max(u) == len(x) - 1
    assert len(u) == len(x)
    assert all((float(len(vli)) / (len(vli) + len(tli)) >= val_ratio for vli, tli in zip(vl, tl)))
    assert all(((len(vli) - 1.0) / (len(vli) + len(tli)) <= val_ratio for vli, tli in zip(vl, tl)))
    assert all((len(set(vli).union(tli)) == len(vli + tli) for vli, tli in zip(vl, tl)))
    assert all_len <= math.ceil(len(x) / rep_count) * rep_count


def test_combinations():
    st = time.time()
    labels_len = 2
    combs = 0
    random.seed(0)
    for data_len in (100, 1001, 17777):
        max_rep = int(math.sqrt(data_len) * 3)
        for rep_count in range(1, max_rep, max(1, max_rep // 17)):
            for val_num in range(0, random.randint(0, 5)):
                combs += 1
                val_ratio = val_num / 11.0  # good to have prime numbers
                if math.floor(val_ratio * data_len / rep_count) >= labels_len:
                    _dist_no_val(rep_count=rep_count, val_ratio=val_ratio, data_len=data_len, labels_len=labels_len)
    elapsed = time.time() - st
    print("elapsed", elapsed, "combs", combs)


def imagenet_test():
    conf = Config(
        "confs/algos/darts.yaml;confs/datasets/imagenet.yaml",
    )
    conf_loader = conf["nas"]["eval"]["loader"]
    _ = data.get_data(conf_loader)


def exclusion_test(data_len=32, labels_len=2, val_ratio=0.5):
    x = np.array(range(data_len))
    labels = np.array(range(labels_len))
    y = np.repeat(labels, math.ceil(float(data_len) / labels_len))[:data_len]
    np.random.shuffle(y)
    dataset = ListDataset(x, y)

    train_sampler = DistributedStratifiedSampler(
        dataset, val_ratio=val_ratio, is_val=False, shuffle=True, max_items=-1, world_size=1, rank=0
    )

    valid_sampler = DistributedStratifiedSampler(
        dataset, val_ratio=val_ratio, is_val=True, shuffle=True, max_items=-1, world_size=1, rank=0
    )
    tidx = list(train_sampler)
    vidx = list(valid_sampler)

    assert len(tidx) == len(vidx) == 16
    assert all(ti not in vidx for ti in tidx)
    # print(len(tidx), tidx)
    # print(len(vidx), vidx)


exclusion_test()
_dist_no_val(1, 100, val_ratio=0.1)
test_combinations()
