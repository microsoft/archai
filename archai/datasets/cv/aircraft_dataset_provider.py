# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Optional

from overrides import overrides
from torch.utils.data import Dataset
from torchvision.datasets import FGVCAircraft
from torchvision.transforms import ToTensor

from archai.api.dataset_provider import DatasetProvider


class AircraftDatasetProvider(DatasetProvider):
    """FGVC Aircraft dataset provider."""

    def __init__(
        self,
        root: Optional[str] = "dataroot",
    ) -> None:
        """Initializes FGVC Aircraft dataset provider.

        Args:
            root: Root directory of dataset where is saved.

        """

        super().__init__()

        self.root = root

    @overrides
    def get_train_dataset(
        self,
        annotation_level: Optional[str] = "variant",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return FGVCAircraft(
            self.root,
            split="train",
            annotation_level=annotation_level,
            transform=transform or ToTensor(),
            target_transform=target_transform,
            download=True,
        )

    @overrides
    def get_val_dataset(
        self,
        annotation_level: Optional[str] = "variant",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return FGVCAircraft(
            self.root,
            split="val",
            annotation_level=annotation_level,
            transform=transform or ToTensor(),
            target_transform=target_transform,
            download=True,
        )

    @overrides
    def get_test_dataset(
        self,
        annotation_level: Optional[str] = "variant",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return FGVCAircraft(
            self.root,
            split="test",
            annotation_level=annotation_level,
            transform=transform or ToTensor(),
            target_transform=target_transform,
            download=True,
        )
