# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Optional

from overrides import overrides
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions, CocoDetection
from torchvision.transforms import ToTensor

from archai.api.dataset_provider import DatasetProvider
from archai.common.logger import Logger

logger = Logger(source=__name__)


class CocoDatasetProvider(DatasetProvider):
    """COCO-based dataset provider."""

    SUPPORTED_DATASETS = {
        "coco_captions": CocoCaptions,
        "coco_detection": CocoDetection,
    }

    def __init__(
        self,
        dataset: Optional[str] = "coco_captions",
        root: Optional[str] = "dataroot",
    ) -> None:
        """Initialize COCO-based dataset provider.

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
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        return self.SUPPORTED_DATASETS[self.dataset](
            self.root,
            ann_file,
            transform=transform or ToTensor(),
            target_transform=target_transform,
        )

    @overrides
    def get_val_dataset(
        self,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        logger.warn(f"Validation set not available for `{self.dataset}`. Returning training set ...")
        return self.get_train_dataset(ann_file, transform=transform, target_transform=target_transform)

    @overrides
    def get_test_dataset(
        self,
        ann_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        logger.warn(f"Testing set not available for `{self.dataset}`. Returning validation set ...")
        return self.get_val_dataset(ann_file, transform=transform, target_transform=target_transform)
