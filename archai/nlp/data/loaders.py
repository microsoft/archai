# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities that reduces the amount of coding needed to load a dataset.
"""

import os
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk as hf_load_from_disk
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.download.download_manager import DownloadMode

from archai.nlp import logging_utils
from archai.nlp.attribute_utils import map_to_list
from archai.nlp.data.processors import (
    map_dataset_to_dict,
    resize_dataset,
    shuffle_dataset,
    tokenize_dataset,
)
from archai.nlp.tokenizer import (
    ArchaiPreTrainedTokenizer,
    ArchaiPreTrainedTokenizerFast,
)

logger = logging_utils.get_logger(__name__)


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
    dataset_files: Optional[Union[Dict[str, Any], List[str]]] = None,
    dataset_split: Optional[Union[str, List[str]]] = None,
    dataset_cache: Optional[str] = None,
    dataset_revision: Optional[List[str]] = None,
    dataset_disk: Optional[str] = "",
    dataset_stream: Optional[bool] = False,
    dataset_refresh_cache: Optional[bool] = False,
    random_seed: Optional[int] = 42,
    n_samples: Optional[Union[int, List[int]]] = -1,
) -> Union[DatasetDict, IterableDatasetDict]:
    """Loads a single dataset.

    Args:
        dataset_name: Name of dataset to be downloaded.
        dataset_config_name: Name of configuration of dataset to be downloaded.
        dataset_split: Split to be retrieved. `None` defaults to all splits.
        dataset_cache: Folder where cache should be stored/loaded.
        dataset_files: Files that should be loaded from `dataset_name` (in case it's a folder).
        dataset_revision: Version of the dataset to be loaded.
        dataset_disk: Folder where dataset should be stored/loaded (if supplied).
        dataset_stream: Whether dataset should be streamed or not.
        dataset_refresh_cache: Whether cache should be refreshed or not.
        random_seed: Fixes the order of samples.
        n_samples: Subsamples into a fixed amount of samples.

    Returns:
        (Union[DatasetDict, IterableDatasetDict]): Loaded dataset.

    """

    # Loads dataset either from Huggingface's disk file, hub or raw files
    if os.path.exists(dataset_disk):
        logger.info(f"Loading dataset from disk: {dataset_disk}")
        dataset = hf_load_from_disk(dataset_disk)
    else:
        logger.info(f"Loading dataset: {dataset_name}/{dataset_config_name}")
        dataset = hf_load_dataset(
            dataset_name,
            name=dataset_config_name,
            data_files=dataset_files,
            download_mode=_should_refresh_cache(dataset_refresh_cache),
            split=dataset_split,
            cache_dir=dataset_cache,
            revision=dataset_revision,
            streaming=dataset_stream,
        )

    # Asserts that dataset is always a dictionary
    if not isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        dataset = map_dataset_to_dict(dataset, splits=dataset_split)

    # Shuffling and resizing
    # Asserts that number of samples is the same length of number of splits
    n_samples_list = map_to_list(n_samples, len(dataset.items()))
    for split, n_samples in zip(dataset.keys(), n_samples_list):
        dataset[split] = shuffle_dataset(dataset[split], random_seed)
        dataset[split] = resize_dataset(dataset[split], n_samples)

    logger.info("Dataset loaded.")

    return dataset


def prepare_dataset(
    tokenizer: Union[ArchaiPreTrainedTokenizer, ArchaiPreTrainedTokenizerFast],
    dataset: Union[DatasetDict, IterableDatasetDict],
    encoded_dataset_path: Optional[str] = "",
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
    """Loads and prepares a dataset.

    Args:
        tokenizer: Tokenizer to transform text into tokens.
        dataset: Dataset to be prepared.
        encoded_dataset_path: Path to where encoded dataset should be stored/loaded (if supplied).
        mapping_column_name: Defines column to be tokenized.
        next_sentence_prediction: Whether next sentence prediction labels should exist or not.
        truncate: Whether samples should be truncated or not.
        padding: Strategy used to pad samples that do not have the proper size.
        batched: Whether mapping should be batched or not.
        batch_size: Number of examples per batch.
        writer_batch_size: Number of examples per write operation to cache.
        num_proc: Number of processes for multiprocessing.
        format_column_name: Defines columns that should be available on dataset.

    Returns:
        (Union[DatasetDict, IterableDatasetDict]): An already tokenized dataset, ready to be used.

    """

    # Loads already-encoded dataset from disk
    if os.path.exists(encoded_dataset_path):
        logger.info(f"Loading prepared dataset: {encoded_dataset_path}")
        return hf_load_from_disk(encoded_dataset_path)

    logger.info(f"Preparing dataset from columns: {mapping_column_name}")

    # Tokenization
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

    # Assures that dataset is in PyTorch format
    if isinstance(dataset, DatasetDict):
        dataset.set_format(type="torch", columns=format_column_name)
    elif isinstance(dataset, IterableDatasetDict):
        dataset = dataset.with_format(type="torch")

    # Saves already-encoded dataset to disk
    if encoded_dataset_path:
        logger.info(f"Saving prepared dataset: {encoded_dataset_path}")
        dataset.save_to_disk(encoded_dataset_path)

    logger.info("Dataset prepared.")

    return dataset
