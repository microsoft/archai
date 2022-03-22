# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Auxiliary functions that allows loading datasets from files or hub.
"""

from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from datasets.utils.download_manager import DownloadMode


def _should_refresh_cache(refresh: bool) -> DownloadMode:
    """Refreshes the cached dataset by re-downloading/re-creating it.

    Args:
        refresh: Whether the dataset cache should be refreshed or not.

    Returns:
        (DownloadMode): Enumerator that defines whether cache should refresh or not.

    """

    if refresh:
        return DownloadMode.FORCE_REDOWNLOAD

    return DownloadMode.REUSE_DATASET_IF_EXISTS


def load_file_dataset(data_dir: str,
                      cache_dir: str,
                      data_files: Optional[Union[Dict[str, Any], List[str]]] = None,
                      split: Optional[List[str]] = None,
                      features: Optional[List[str]] = None,
                      from_stream: Optional[bool] = False,
                      refresh_cache: Optional[bool] = False
                      ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Loads a dataset from local files.

    Args:
        data_dir: Directory of the dataset to be loaded.
        cache_dir: Directory of where the cache should be stored.
        data_files: Files that should be loaded from `data_dir`.
        split: Specific splits that should be loaded (`train`, `val` or `test`).
        features: Custom features (column names) that should be loaded.
        from_stream: Whether dataset should be streamed or not.
        refresh_cache: Whether cache should be refreshed or not.

    Returns:
        (Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]): An instance of the
            loaded and cached dataset.

    """

    return load_dataset(data_dir,
                        cache_dir=cache_dir,
                        data_files=data_files,
                        split=split,
                        download_mode=_should_refresh_cache(refresh_cache),
                        features=features,
                        streaming=from_stream)


def load_hub_dataset(data_dir: str,
                     data_config_name: str,
                     cache_dir: str,
                     split: Optional[List[str]] = None,
                     revision: Optional[List[str]] = None,
                     from_stream: Optional[bool] = False,
                     refresh_cache: Optional[bool] = False
                     ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Loads a dataset from Huggingface's datasets hub.

    Args:
        data_dir: Directory of the dataset to be loaded.
        data_config_name: Name of dataset's configuration to be loaded.
        cache_dir: Directory of where the cache should be stored.
        split: Specific splits that should be loaded (`train`, `val` or `test`).
        revision: Version of the dataset to be loaded.
        from_stream: Whether dataset should be streamed or not.
        refresh_cache: Whether cache should be refreshed or not.

    Returns:
        (Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]): An instance of the
            downloaded and cached dataset.

    """

    return load_dataset(data_dir,
                        name=data_config_name,
                        cache_dir=cache_dir,
                        split=split,
                        download_mode=_should_refresh_cache(refresh_cache),
                        revision=revision,
                        streaming=from_stream)
