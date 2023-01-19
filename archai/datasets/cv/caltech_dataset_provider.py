# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, List, Optional, Union

from overrides import overrides
from torch.utils.data import Dataset
from torchvision.datasets import Caltech101, Caltech256
from torchvision.transforms import ToTensor

from archai.api.dataset_provider import DatasetProvider
from archai.common.logger import Logger

logger = Logger(source=__name__)


class CaltechDatasetProvider(DatasetProvider):
    """Caltech-based dataset provider."""

    SUPPORTED_DATASETS = {
        "caltech101": Caltech101,
        "caltech256": Caltech256,
    }

    def __init__(
        self,
        dataset: Optional[str] = "caltech101",
        root: Optional[str] = "dataroot",
    ) -> None:
        """Initializes Caltech-based dataset provider.

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
        target_type: Optional[Union[str, List[str]]] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        kwargs = {"target_type": target_type} if self.dataset == "caltech101" else {}
        return self.SUPPORTED_DATASETS[self.dataset](
            self.root, download=True, transform=transform or ToTensor(), target_transform=target_transform, **kwargs
        )

    @overrides
    def get_val_dataset(
        self,
        target_type: Optional[Union[str, List[str]]] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        logger.warn(f"Validation set not available for `{self.dataset}`. Returning training set ...")
        return self.get_train_dataset(target_type=target_type, transform=transform, target_transform=target_transform)

    @overrides
    def get_test_dataset(
        self,
        target_type: Optional[Union[str, List[str]]] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        logger.warn(f"Testing set not available for `{self.dataset}`. Returning training set ...")
        return self.get_train_dataset(target_type=target_type, transform=transform, target_transform=target_transform)
