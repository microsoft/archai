# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Optional

from overrides import overrides
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor

from archai.api.dataset_provider import DatasetProvider
from archai.common.ordered_dict_logger import OrderedDictLogger

logger = OrderedDictLogger(source=__name__)


class CifarDatasetProvider(DatasetProvider):
    """CIFAR-based dataset provider."""

    SUPPORTED_DATASETS = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
    }

    def __init__(
        self,
        dataset: Optional[str] = "cifar10",
        root: Optional[str] = "dataroot",
    ) -> None:
        """Initialize CIFAR-based dataset provider.

        Args:
            dataset: Name of dataset.
            root: Root directory of dataset where is saved.

        """

        super().__init__()

        assert dataset in self.SUPPORTED_DATASETS, f"`dataset` should be one of: {list(self.SUPPORTED_DATASETS)}"
        self.dataset = dataset
        self.root = root

    @overrides
    def get_train_dataset(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return self.SUPPORTED_DATASETS[self.dataset](
            self.root,
            train=True,
            download=True,
            transform=transform or ToTensor(),
            target_transform=target_transform,
        )

    @overrides
    def get_val_dataset(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return self.SUPPORTED_DATASETS[self.dataset](
            self.root,
            train=False,
            download=True,
            transform=transform or ToTensor(),
            target_transform=target_transform,
        )

    @overrides
    def get_test_dataset(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        logger.warn(f"Testing set not available for `{self.dataset}`. Returning validation set ...")
        return self.get_val_dataset(transform=transform, target_transform=target_transform)
