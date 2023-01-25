# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset


class DistributedStratifiedSampler(Sampler):
    """Distributed stratified sampling of dataset.

    This sampler works in distributed as well as non-distributed setting with no penalty in
    either mode and is a replacement for built-in `torch.util.data.DistributedSampler`.

    In distributed setting, many instances of the same code runs as process
    known as replicas. Each replica has sequential number assigned by the launcher,
    starting from 0 to uniquely identify it. This is known as global rank or simply rank.
    The number of replicas is known as the world size. For non-distributed setting,
    `world_size=1` and `rank=0`.

    This sampler assumes that labels for each datapoint is available in dataset.targets
    property which should be array like containing as many values as length of the dataset.
    This is availalble already for many popular datasets such as cifar and, with newer
    PyTorch versions, `ImageFolder` as well as `DatasetFolder`. If you are using custom dataset,
    you can usually create this property with one line of code such as
    `dataset.targets = [yi for _, yi in dataset]`.

    To do distributed sampling, each replica must shuffle with same seed as all other
    replicas with every epoch and then chose some subset of dataset for itself.
    Traditionally, we use epoch number as seed for shuffling for each replica.
    However, this then requires that training code calls `sampler.set_epoch(epoch)`
    to set seed at every epoch.

    """

    def __init__(
        self,
        dataset: Dataset,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: Optional[bool] = True,
        val_ratio: Optional[float] = 0.0,
        is_val_split: Optional[bool] = False,
        max_samples: Optional[int] = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            dataset: Input dataset.
            world_size: Total number of replicas. If `None` then auto-detect,
                while 1 for non-distributed setting.
            rank: Global rank of this replica. If `None` then auto-detect,
                while 0 for non-distributed setting.
            shuffle: Whether shuffling should be applied for every epoch.
            val_ratio: Creates a validation split when set to > 0.
            is_val_split: Whether the validation split should be returned.
            max_samples: Maximum number of samples for each replica.

        """

        # CIFAR-10 amd DatasetFolder has this attribute
        # For others it may be easy to add from outside
        assert (
            hasattr(dataset, "targets") and dataset.targets is not None
        ), "dataset needs to have targets attribute to work with this sampler"

        if world_size is None:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        if val_ratio is None:
            val_ratio = 0.0

        assert world_size >= 1
        assert rank >= 0 and rank < world_size
        assert val_ratio < 1.0 and val_ratio >= 0.0

        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0  # Used as a seed

        self.shuffle = shuffle
        self.data_len = len(self.dataset)
        self.max_samples = max_samples if max_samples is not None and max_samples >= 0 else None
        assert self.data_len == len(dataset.targets)

        self.val_ratio = val_ratio
        self.is_val_split = is_val_split

        # Computes duplications of dataset to make it divisible by world_size
        self.replica_len = self.replica_len_full = int(math.ceil(float(self.data_len) / self.world_size))
        self.total_size = self.replica_len_full * self.world_size
        assert self.total_size >= self.data_len

        if self.max_samples is not None:
            self.replica_len = min(self.replica_len_full, self.max_samples)

        self.main_split_len = int(math.floor(self.replica_len * (1 - val_ratio)))
        self.val_split_len = self.replica_len - self.main_split_len
        self._len = self.val_split_len if self.is_val_split else self.main_split_len

    def __len__(self) -> int:
        """Return the length of the current replica's split."""

        return self._len

    def __iter__(self) -> Iterable:
        """Return an iterator over the current replica's split."""

        indices, targets = self._get_indices()
        indices, targets = self._split_rank(indices, targets)
        indices, targets = self._limit_indices(indices, targets, self.max_samples)

        indices, _ = self._split_indices(indices, targets, self.val_split_len, self.is_val_split)
        assert len(indices) == self._len

        if self.shuffle and self.val_ratio > 0.0 and self.epoch > 0:
            np.random.shuffle(indices)

        return iter(indices)

    def _split_rank(self, indices: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split the indices into `world_size` folds and return the one for current rank.

        Args:
            indices: Indices to split.
            targets: Targets to split.

        Returns:
            Indices and targets for current rank.

        """

        if self.world_size > 1:
            replica_fold_idxs = None

            rfolder = StratifiedKFold(n_splits=self.world_size, shuffle=False)
            folds = rfolder.split(indices, targets)

            for _ in range(self.rank + 1):
                _, replica_fold_idxs = next(folds)

            assert replica_fold_idxs is not None and len(replica_fold_idxs) == self.replica_len_full

            return indices[replica_fold_idxs], targets[replica_fold_idxs]

        assert self.world_size == 1
        return indices, targets

    def _get_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices for the current epoch.

        Returns:
            Indices and targets for the current epoch.

        """

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self._get_seed())
            indices = torch.randperm(self.data_len, generator=g).numpy()
        else:
            indices = np.arange(self.data_len)

        if self.total_size > self.data_len:
            indices = np.append(indices, indices[: (self.total_size - self.data_len)])
        else:
            assert self.total_size == self.data_len, "`total_size` cannot be less than dataset size."

        targets = np.array(list(self.dataset.targets[i] for i in indices))
        assert len(indices) == self.total_size

        return indices, targets

    def _limit_indices(
        self, indices: np.ndarray, targets: np.ndarray, max_samples: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Limit the number of indices to `max_samples`.

        Args:
            indices: Indices to limit.
            targets: Targets to limit.

        Returns:
            Limited indices and targets.

        """

        if max_samples is not None:
            return self._split_indices(indices, targets, len(indices) - max_samples, False)

        return indices, targets

    def _get_seed(self) -> int:
        """Get the seed for the current epoch.

        Returns:
            Seed for the current epoch.

        """

        return self.epoch if self.val_ratio == 0.0 else 0

    def _split_indices(
        self, indices: np.ndarray, targets: np.ndarray, val_size: int, return_val_split: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split the indices into train and validation sets.

        Args:
            indices: Indices to split.
            targets: Targets to split.
            val_size: Number of samples to use for validation.
            return_val_split: Whether to return the validation split.

        Returns:
            Train/validation indices and targets.

        """

        if val_size:
            assert isinstance(val_size, int)

            vfolder = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=self._get_seed())
            vfolder = vfolder.split(indices, targets)

            train_idx, valid_idx = next(vfolder)

            idxs = valid_idx if return_val_split else train_idx
            return indices[idxs], targets[idxs]

        return indices, targets

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the current replica, which is used to seed the shuffling.

        Args:
            epoch: Epoch number.

        """

        self.epoch = epoch
