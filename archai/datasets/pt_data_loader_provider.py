# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Iterable, Optional, Union

from overrides import overrides
from torch import Generator
from torch.utils.data import DataLoader, Dataset, Sampler

from archai.api.data_loader_provider import DataLoaderProvider


class TorchDataLoaderProvider(DataLoaderProvider):
    """PyTorch data loader provider."""

    def __init__(
        self,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        sampler: Optional[Union[Sampler, Iterable]] = None,
        batch_sampler: Optional[Union[Sampler, Iterable]] = None,
        num_workers: Optional[int] = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: Optional[bool] = False,
        drop_last: Optional[bool] = False,
        timeout: Optional[int] = 0,
        worker_init_fn: Optional[Callable] = None,
        generator: Optional[Generator] = None,
        prefetch_factor: Optional[int] = 2,
        persistent_workers: Optional[bool] = False,
        pin_memory_device: Optional[str] = "",
    ) -> None:
        """Initialize PyTorch data loader provider.

        Args:
            batch_size: Amount of samples per batch.
            shuffle: Whether to re-shuffle the data at every epoch.
            sampler: Strategy to draw samples from the dataset.
            batch_sampler: Samples a batch of indices at a time.
            num_workers: Number of subprocesses to use for data loading.
            collate_fn: Function to merge list of samples into mini-batches.
            pin_memory: Whether to copy tensors into `pin_memory_device` or
                CUDA before returning them.
            drop_last: Whether to drop the last incomplete batch.
            timeout: Timeout for collecting a batch from workers.
            worker_init_fn: Function called on each worker subprocess before data loading.
            generator: RNG used to generate indices for the sampler.
            prefetch_factor: Number of batches loaded in advance by each worker.
            persistent_workers: Whether to not shutdown consumed workers after each epoch.
            pin_memory_device: Name of device to copy tensors to.

        """

        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.generator = generator
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device

    @overrides
    def get_data_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            pin_memory_device=self.pin_memory_device,
        )
