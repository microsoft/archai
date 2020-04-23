import math
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Sampler
import torch.distributed as dist
from torch.utils.data.dataset import Dataset

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

class DistributedStratifiedSampler(Sampler):
    def __init__(self, dataset:Dataset, num_replicas:Optional[int]=None,
                 rank:Optional[int]=None, shuffle=True,
                 val_ratio=0.0, is_val=False, auto_epoch=True,
                 max_items:Optional[int]=None):
        """Performs stratified sampling of dataset for each replica in the distributed as well as non-distributed setting. If validation split is needed then yet another stratified sampling within replica's split is performed to further obtain the train/validation splits.

        This sampler works in distributed as well as non-distributed setting with no panelty in either mode and is replacement for built-in torch.util.data.DistributedSampler. In distributed setting, many instances of the same code runs as process known as replicas. Each replica has sequential number assigned by the launcher, starting from 0 to uniquely identify it. This is known as global rank or simply rank. The number of replicas is known as the world size. For non-distributed setting, world_size=1 and rank=0.

        To perform stratified sampling we need labels. This sampler assumes that labels for each datapoint is available in dataset.targets property which should be array like containing as many values as length of the dataset. This is availalble already for many popular datasets such as cifar and, with newer PyTorch versions, ImageFolder as well as DatasetFolder. If you are using custom dataset, you can usually create this property with one line of code such as `dataset.targets = [yi for _, yi in dataset]`.

        Generally, to do distributed sampling, each replica must shuffle with same seed as all other replicas with every epoch and then chose some subset of dataset for itself. Traditionally, we use epoch number as seed for shuffling for each replica. However, this then requires that training code calls sampler.set_epoch(epoch) to set seed at every epoch. This breaks many training code which doesn't have access to the sampler. However, this is unnecessory as well. Sampler knows when each new iteration is requested, so it can automatically increment epoch and make call to set_epoch by itself. This is what this sampler does. This makes things transparent to users in distributed or non-distributed setting. If you don't want this behaviour then pass auto_epoch=False.

        Arguments:
            dataset -- PyTorch dataset like object

        Keyword Arguments:
            num_replicas -- Total number of replicas running in distributed setting, if None then auto-detect, 1 for non distributed setting (default: {None})
            rank -- Global rank of this replica, if None then auto-detect, 0 for non distributed setting (default: {None})
            shuffle {bool} -- If True then suffle at every epoch (default: {True})
            val_ratio {float} -- If you want to create validation split then set to > 0 (default: {0.0})
            is_val {bool} -- If True then validation split is returned set to val_ratio otherwise main split is returned (default: {False})
            auto_epoch {bool} -- if True then automatically count epoch for each new iteration eliminating the need to call set_epoch() in distributed setting (default: {True})
            max_items -- if >= 0 then dataset will be trimmed to these many items for each replica (useful to test on smaller dataset)
        """


        # cifar10 amd DatasetFolder has this attribute, for others it may be easy to add from outside
        assert hasattr(dataset, 'targets') and dataset.targets is not None, 'dataset needs to have targets attribute to work with this sampler'

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        assert num_replicas >= 1
        assert rank >= 0 and rank < num_replicas
        assert val_ratio < 1.0 and val_ratio >= 0.0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = -1
        self.auto_epoch = auto_epoch
        self.shuffle = shuffle
        self.data_len = len(self.dataset)
        self.max_items = max_items if max_items is not None and max_items >= 0 else None
        assert self.data_len == len(dataset.targets)
        self.val_ratio = val_ratio
        self.is_val = is_val

        # computing duplications we needs
        self.replica_len = self.replica_len_full =  int(math.ceil(float(self.data_len)/self.num_replicas))
        self.total_size = self.replica_len_full * self.num_replicas
        assert self.total_size >= self.data_len

        if self.max_items is not None:
            self.replica_len = min(self.replica_len_full, self.max_items)

        self.main_split_len = int(math.floor(self.replica_len*(1-val_ratio)))
        self.val_split_len = self.replica_len - self.main_split_len
        self._len = self.val_split_len if self.is_val else self.main_split_len


    def __iter__(self):
        # every time new iterator is requested, we increase epoch count
        # assuming all replicas do same thing, this avoids need to call set_epoch
        if self.auto_epoch:
            self.epoch += 1

        # get shuffled indices, dataset is extended if needed to divide equally
        # between replicas
        indices, targets = self._indices()

        # get the fold which we will assign to current replica
        indices, targets = self._replica_fold(indices, targets)

        indices, targets = self._limit(indices, targets, self.max_items)

        # split current replica's fold between train and val
        # return indices depending on if we are val or train split
        indices, _ = self._split(indices, targets, self.val_split_len, self.is_val)
        assert len(indices) == self._len

        return iter(indices)

    def _replica_fold(self, indices:np.ndarray, targets:np.ndarray)\
            ->Tuple[np.ndarray, np.ndarray]:

        if self.num_replicas > 1:
            replica_fold_idxs = None
            rfolder = StratifiedKFold(n_splits=self.num_replicas, shuffle=False)
            folds = rfolder.split(indices, targets)
            for _ in range(self.rank + 1):
                other_fold_idxs, replica_fold_idxs = next(folds)

            assert replica_fold_idxs is not None and \
                    len(replica_fold_idxs)==self.replica_len_full

            return indices[replica_fold_idxs], targets[replica_fold_idxs]
        else:
            assert self.num_replicas == 1
            return indices, targets


    def _indices(self)->Tuple[np.ndarray, np.ndarray]:
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.data_len, generator=g).numpy()
        else:
            indices = np.arange(self.data_len)

        # add extra samples to make it evenly divisible
        # this is neccesory because we have __len__ which must return same
        # number consistently
        if self.total_size > self.data_len:
            indices = np.append(indices, indices[:(self.total_size - self.data_len)])
        else:
            assert self.total_size == self.data_len, 'total_size cannot be less than dataset size!'

        targets = np.array(list(self.dataset.targets[i] for i in indices))
        assert len(indices) == self.total_size

        return indices, targets

    def _limit(self, indices:np.ndarray, targets:np.ndarray, max_items:Optional[int])\
            ->Tuple[np.ndarray, np.ndarray]:
        if max_items is not None:
            return self._split(indices, targets, max_items, True)
        return indices, targets

    def _split(self, indices:np.ndarray, targets:np.ndarray, test_size:int,
               return_test_split:bool)->Tuple[np.ndarray, np.ndarray]:
        if test_size:
            assert isinstance(test_size, int) # othewise next call assumes ratio instead of count
            vfolder = StratifiedShuffleSplit(n_splits=1,
                                             test_size=test_size,
                                             random_state=self.epoch)
            vfolder = vfolder.split(indices, targets)
            train_idx, valid_idx = next(vfolder)

            idxs = valid_idx if return_test_split else train_idx
            return indices[idxs], targets[idxs]
        else:
            return indices, targets

    def __len__(self):
        return self._len

    def set_epoch(self, epoch):
        self.epoch = epoch