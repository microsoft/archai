# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for loading and encoding datasets.
"""

import os
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk as hf_load_from_disk
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.download.download_manager import DownloadMode
from transformers.models.auto.tokenization_auto import AutoTokenizer

from archai.common.utils import map_to_list
from archai.nlp.datasets.hf.processors import (
    map_dataset_to_dict,
    resize_dataset,
    shuffle_dataset,
    tokenize_dataset,
)
from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)


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


def load_dataset(
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    dataset_files: Optional[Union[Dict[str, Any], List[str]]] = None,
    dataset_split: Optional[Union[str, List[str]]] = None,
    dataset_cache: Optional[str] = None,
    dataset_keep_in_memory: Optional[bool] = None,
    dataset_revision: Optional[List[str]] = None,
    dataset_disk: Optional[str] = "",
    dataset_stream: Optional[bool] = False,
    dataset_refresh_cache: Optional[bool] = False,
    random_seed: Optional[int] = 42,
    n_samples: Optional[Union[int, List[int]]] = -1,
) -> Union[DatasetDict, IterableDatasetDict]:
    """Loads a dataset from Huggingface's Hub or local files.

    Args:
        dataset_name: Name of dataset to be downloaded.
        dataset_config_name: Name of configuration of dataset to be downloaded.
        dataset_dir: Path to manually downloaded files.
        dataset_files: Files that should be loaded from `dataset_name` (in case it's a folder).
        dataset_split: Split to be retrieved. `None` defaults to all splits.
        dataset_cache: Folder where cache should be stored/loaded.
        dataset_keep_in_memory: Whether dataset should be directly loaded in memory or not.
        dataset_revision: Version of the dataset to be loaded.
        dataset_disk: Folder where dataset should be stored/loaded (if supplied).
        dataset_stream: Whether dataset should be streamed or not.
        dataset_refresh_cache: Whether cache should be refreshed or not.
        random_seed: Fixes the order of samples.
        n_samples: Subsamples into a fixed amount of samples.

    Returns:
        (Union[DatasetDict, IterableDatasetDict]): Loaded dataset.

    """

    if os.path.exists(dataset_disk):
        dataset = hf_load_from_disk(dataset_disk, keep_in_memory=dataset_keep_in_memory)
    else:
        dataset = hf_load_dataset(
            dataset_name,
            name=dataset_config_name,
            data_dir=dataset_dir,
            data_files=dataset_files,
            download_mode=_should_refresh_cache(dataset_refresh_cache),
            split=dataset_split,
            cache_dir=dataset_cache,
            keep_in_memory=dataset_keep_in_memory,
            revision=dataset_revision,
            streaming=dataset_stream,
        )

    if not isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        dataset = map_dataset_to_dict(dataset, splits=dataset_split)

    n_samples_list = map_to_list(n_samples, len(dataset.items()))
    for split, n_samples in zip(dataset.keys(), n_samples_list):
        dataset[split] = shuffle_dataset(dataset[split], random_seed)
        dataset[split] = resize_dataset(dataset[split], n_samples)

    return dataset


def encode_dataset(
    tokenizer: Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast],
    dataset: Union[DatasetDict, IterableDatasetDict],
    mapping_column_name: Optional[Union[str, List[str]]] = "text",
    next_sentence_prediction: Optional[bool] = False,
    truncate: Optional[bool] = True,
    padding: Optional[str] = "max_length",
    batched: Optional[bool] = True,
    batch_size: Optional[int] = 1000,
    writer_batch_size: Optional[int] = 1000,
    num_proc: Optional[int] = None,
    format_column_name: Optional[Union[str, List[str]]] = None,
) -> Union[DatasetDict, IterableDatasetDict]:
    """Encodes a dataset.

    Args:
        tokenizer: Tokenizer to transform text into tokens.
        dataset: Dataset to be encoded.
        mapping_column_name: Column to be tokenized.
        next_sentence_prediction: Whether next sentence prediction labels should exist or not.
        truncate: Whether samples should be truncated or not.
        padding: Strategy used to pad samples that do not have the proper size.
        batched: Whether mapping should be batched or not.
        batch_size: Number of examples per batch.
        writer_batch_size: Number of examples per write operation to cache.
        num_proc: Number of processes for multi-processing.
        format_column_name: Columns that should be available on dataset.

    Returns:
        (Union[DatasetDict, IterableDatasetDict]): Encoded dataset.

    """

    dataset = tokenize_dataset(
        dataset,
        tokenizer,
        mapping_column_name,
        next_sentence_prediction,
        truncate,
        padding,
        batched,
        batch_size,
        writer_batch_size,
        num_proc,
    )

    if isinstance(dataset, DatasetDict):
        dataset.set_format(type="torch", columns=format_column_name)
    elif isinstance(dataset, IterableDatasetDict):
        dataset = dataset.with_format(type="torch")

    return dataset
