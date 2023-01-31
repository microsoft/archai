# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Optional

from overrides import overrides
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor

from archai.api.dataset_provider import DatasetProvider


class CityscapesDatasetProvider(DatasetProvider):
    """Cityscapes dataset provider."""

    def __init__(
        self,
        root: Optional[str] = "dataroot",
    ) -> None:
        """Initialize Cityscapes dataset provider.

        Args:
            root: Root directory of dataset where is saved.

        """

        super().__init__()

        self.root = root

    @overrides
    def get_train_dataset(
        self,
        target_type: Optional[str] = "instance",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return Cityscapes(
            self.root,
            split="train",
            mode="fine",
            target_type=target_type,
            transform=transform or ToTensor(),
            target_transform=target_transform,
        )

    @overrides
    def get_val_dataset(
        self,
        target_type: Optional[str] = "instance",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return Cityscapes(
            self.root,
            split="val",
            mode="fine",
            target_type=target_type,
            transform=transform or ToTensor(),
            target_transform=target_transform,
        )

    @overrides
    def get_test_dataset(
        self,
        target_type: Optional[str] = "instance",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return Cityscapes(
            self.root,
            split="test",
            mode="fine",
            target_type=target_type,
            transform=transform or ToTensor(),
            target_transform=target_transform,
        )
