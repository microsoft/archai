# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for loading and encoding datasets."""

import os
from typing import Any, Callable, Dict, List, Optional, Union

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
    """Determine whether to refresh the cached dataset.

    This function determines whether the cached dataset should be refreshed by
    re-downloading or re-creating it based on the value of the `refresh`
    parameter.

    Args:
        refresh: If `True`, the cache will be refreshed. If `False`, the
            existing cache will be used if it exists.

    Returns:
        An enumerator indicating whether the cache should be refreshed or not.

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
    """Load a dataset from Hugging Face's Hub or local files.

    This function loads a dataset from either Hugging Face's Hub or local
    files, depending on whether the `dataset_disk` parameter is provided. It
    also has options for subsampling the dataset, shuffling the order of
    samples, and keeping the dataset in memory.

    Args:
        dataset_name: Name of the dataset to be downloaded.
        dataset_config_name: Name of the configuration of the dataset to be downloaded.
        dataset_dir: Path to manually downloaded files.
        dataset_files: Files that should be loaded from `dataset_name` (in case it's a folder).
        dataset_split: Split to be retrieved. If `None`, all splits will be retrieved.
        dataset_cache: Folder where cache should be stored/loaded.
        dataset_keep_in_memory: Whether the dataset should be directly loaded in memory or not.
        dataset_revision: Version of the dataset to be loaded.
        dataset_disk: Folder where dataset should be stored/loaded (if supplied).
        dataset_stream: Whether the dataset should be streamed or not.
        dataset_refresh_cache: Whether the cache should be refreshed or not.
        random_seed: Fixes the order of samples.
        n_samples: Subsamples into a fixed amount of samples.
            If `int`, the same number of samples will be used for all splits.
            If `List[int]`, a different number of samples can be specified for each split.

    Returns:
        The loaded dataset.

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
        dataset = map_dataset_to_dict(dataset, dataset_split)

    n_samples_list = map_to_list(n_samples, len(dataset.items()))
    for split, n_samples in zip(dataset.keys(), n_samples_list):
        dataset[split] = shuffle_dataset(dataset[split], random_seed)
        dataset[split] = resize_dataset(dataset[split], n_samples)

    return dataset


def encode_dataset(
    dataset: Union[DatasetDict, IterableDatasetDict],
    tokenizer: Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast],
    mapping_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    mapping_fn_kwargs: Optional[Dict[str, Any]] = None,
    mapping_column_name: Optional[Union[str, List[str]]] = "text",
    batched: Optional[bool] = True,
    batch_size: Optional[int] = 1000,
    writer_batch_size: Optional[int] = 1000,
    num_proc: Optional[int] = None,
    format_column_name: Optional[Union[str, List[str]]] = None,
) -> Union[DatasetDict, IterableDatasetDict]:
    """Encode a dataset using a tokenizer.

    Args:
        dataset: The dataset to be encoded.
        tokenizer: The tokenizer to use for encoding.
        mapping_fn: A function that maps the dataset. If not provided,
            the default `tokenize_dataset` function will be used.
        mapping_fn_kwargs: Keyword arguments to pass to `mapping_fn`.
        mapping_column_name: The columns in the dataset to be tokenized.
            If `str`, only one column will be tokenized.
            If `List[str]`, multiple columns will be tokenized.
        batched: Whether the mapping should be done in batches or not.
        batch_size: The number of examples per batch when mapping in batches.
        writer_batch_size: The number of examples per write operation to cache.
        num_proc: The number of processes to use for multi-processing.
        format_column_name: The columns that should be available on the resulting dataset.
            If `str`, only one column will be available.
            If `List[str]`, multiple columns will be available.

    Returns:
        The encoded dataset.

    """

    if not mapping_fn:
        mapping_fn = tokenize_dataset

    if isinstance(mapping_column_name, str):
        mapping_column_name = (mapping_column_name,)
    elif isinstance(mapping_column_name, list):
        mapping_column_name = tuple(mapping_column_name)

    fn_kwargs = mapping_fn_kwargs or {}
    fn_kwargs["tokenizer"] = tokenizer
    fn_kwargs["mapping_column_name"] = mapping_column_name

    remove_columns = [v.column_names for _, v in dataset.items()]
    assert all([c[0] for c in remove_columns])

    mapping_kwargs = {"batched": batched}
    if isinstance(dataset, DatasetDict):
        mapping_kwargs["remove_columns"] = remove_columns[0]
        mapping_kwargs["batch_size"] = batch_size
        mapping_kwargs["writer_batch_size"] = writer_batch_size
        mapping_kwargs["num_proc"] = num_proc

    dataset = dataset.map(mapping_fn, fn_kwargs=fn_kwargs, **mapping_kwargs)

    if isinstance(dataset, DatasetDict):
        dataset.set_format(type="torch", columns=format_column_name)
    elif isinstance(dataset, IterableDatasetDict):
        dataset = dataset.with_format(type="torch")

    return dataset
