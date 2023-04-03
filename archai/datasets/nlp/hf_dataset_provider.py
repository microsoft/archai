# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Optional, Union

from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk as hf_load_from_disk
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from datasets.splits import Split
from datasets.utils.version import Version
from overrides import overrides

from archai.api.dataset_provider import DatasetProvider
from archai.common.ordered_dict_logger import OrderedDictLogger
from archai.datasets.nlp.hf_dataset_provider_utils import should_refresh_cache

logger = OrderedDictLogger(source=__name__)


class HfHubDatasetProvider(DatasetProvider):
    """Hugging Face Hub dataset provider."""

    def __init__(
        self,
        dataset_name: str,
        dataset_config_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[Union[str, Version]] = None,
    ) -> None:
        """Initialize Hugging Face Hub dataset provider.

        Args:
            dataset_name: Name of the dataset.
            dataset_config_name: Name of the dataset configuration.
            data_dir: Path to the data directory.
            data_files: Path(s) to the data file(s).
            cache_dir: Path to the read/write cache directory.
            revision: Version of the dataset to load.

        """

        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.data_dir = data_dir
        self.data_files = data_files
        self.cache_dir = cache_dir
        self.revision = revision

    def get_dataset(
        self,
        split: Optional[Union[str, Split]] = None,
        refresh_cache: Optional[bool] = False,
        keep_in_memory: Optional[bool] = False,
        streaming: Optional[bool] = False,
    ) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        return hf_load_dataset(
            self.dataset_name,
            name=self.dataset_config_name,
            data_dir=self.data_dir,
            data_files=self.data_files,
            split=split,
            cache_dir=self.cache_dir,
            download_mode=should_refresh_cache(refresh_cache),
            keep_in_memory=keep_in_memory,
            revision=self.revision,
            streaming=streaming,
        )

    @overrides
    def get_train_dataset(
        self,
        split: Optional[Union[str, Split]] = "train",
        refresh_cache: Optional[bool] = False,
        keep_in_memory: Optional[bool] = False,
        streaming: Optional[bool] = False,
    ) -> Union[Dataset, IterableDataset]:
        return self.get_dataset(
            split=split, refresh_cache=refresh_cache, keep_in_memory=keep_in_memory, streaming=streaming
        )

    @overrides
    def get_val_dataset(
        self,
        split: Optional[Union[str, Split]] = "validation",
        refresh_cache: Optional[bool] = False,
        keep_in_memory: Optional[bool] = False,
        streaming: Optional[bool] = False,
    ) -> Union[Dataset, IterableDataset]:
        try:
            return self.get_dataset(
                split=split, refresh_cache=refresh_cache, keep_in_memory=keep_in_memory, streaming=streaming
            )
        except ValueError:
            logger.warn(f"Validation set not available for `{self.dataset}`. Returning full training set ...")
            return self.get_dataset(
                split="train", refresh_cache=refresh_cache, keep_in_memory=keep_in_memory, streaming=streaming
            )

    @overrides
    def get_test_dataset(
        self,
        split: Optional[Union[str, Split]] = "test",
        refresh_cache: Optional[bool] = False,
        keep_in_memory: Optional[bool] = False,
        streaming: Optional[bool] = False,
    ) -> Union[Dataset, IterableDataset]:
        try:
            return self.get_dataset(
                split=split, refresh_cache=refresh_cache, keep_in_memory=keep_in_memory, streaming=streaming
            )
        except ValueError:
            logger.warn(f"Testing set not available for `{self.dataset}`. Returning full validation set ...")
            return self.get_dataset(
                split="validation", refresh_cache=refresh_cache, keep_in_memory=keep_in_memory, streaming=streaming
            )


class HfDiskDatasetProvider(DatasetProvider):
    """Hugging Face disk-saved dataset provider."""

    def __init__(
        self,
        data_dir: str,
        keep_in_memory: Optional[bool] = False,
    ) -> None:
        """Initialize Hugging Face disk-saved dataset provider.

        Args:
            data_dir: Path to the disk-saved dataset.
            keep_in_memory: Whether to keep the dataset in memory.

        """

        super().__init__()

        self.data_dir = data_dir
        self.keep_in_memory = keep_in_memory

        # Pre-loads the dataset when class is instantiated to avoid loading it multiple times
        self.dataset = hf_load_from_disk(self.data_dir, keep_in_memory=keep_in_memory)

    @overrides
    def get_train_dataset(self) -> Dataset:
        if isinstance(self.dataset, DatasetDict):
            return self.dataset["train"]
        return self.dataset

    @overrides
    def get_val_dataset(self) -> Dataset:
        try:
            if isinstance(self.dataset, DatasetDict):
                return self.dataset["validation"]
        except:
            logger.warn("Validation set not available. Returning training set ...")
            return self.get_train_dataset()

    @overrides
    def get_test_dataset(self) -> Dataset:
        try:
            if isinstance(self.dataset, DatasetDict):
                return self.dataset["test"]
        except:
            logger.warn("Testing set not available. Returning validation set ...")
            return self.get_val_dataset()
