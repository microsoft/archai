# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Optional

from overrides import overrides
from torch.utils.data import Dataset
from torchvision.datasets import StanfordCars
from torchvision.transforms import ToTensor

from archai.api.dataset_provider import DatasetProvider
from archai.common.logger import Logger

logger = Logger(source=__name__)


class StanfordCarsDatasetProvider(DatasetProvider):
    """Stanford Cars dataset provider."""

    def __init__(
        self,
        root: Optional[str] = "dataroot",
    ) -> None:
        """Initialize Stanford Cars dataset provider.

        Args:
            root: Root directory of dataset where is saved.

        """

        super().__init__()

        self.root = root

    @overrides
    def get_train_dataset(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return StanfordCars(
            self.root,
            split="train",
            transform=transform or ToTensor(),
            target_transform=target_transform,
            download=True,
        )

    @overrides
    def get_val_dataset(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        logger.warn("Validation set not available. Returning training set ...")
        return self.get_train_dataset(transform=transform, target_transform=target_transform)

    @overrides
    def get_test_dataset(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return StanfordCars(
            self.root,
            split="test",
            transform=transform or ToTensor(),
            target_transform=target_transform,
            download=True,
        )
