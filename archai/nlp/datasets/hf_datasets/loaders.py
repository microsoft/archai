# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

import os
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk as hf_load_from_disk
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.download.download_manager import DownloadMode
from transformers.models.auto.tokenization_auto import AutoTokenizer
from archai.nlp.datasets.hf_datasets.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast
)
from archai.nlp.datasets.hf_datasets.processors import (
    map_dataset_to_dict,
    resize_dataset,
    shuffle_dataset,
    tokenize_dataset,
)
from archai.common.common import map_to_list


def _should_refresh_cache(refresh: bool) -> DownloadMode:
    """
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
    """
    """

    if os.path.exists(dataset_disk):
        dataset = hf_load_from_disk(dataset_disk)
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


def prepare_dataset(
    tokenizer: Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast],
    dataset: Union[DatasetDict, IterableDatasetDict],
    encoded_dataset_path: Optional[str] = "",
    encoded_dataset_keep_in_memory: Optional[bool] = None,
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
    """
    """

    if os.path.exists(encoded_dataset_path):
        return hf_load_from_disk(
            encoded_dataset_path, keep_in_memory=encoded_dataset_keep_in_memory
        )

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

    if encoded_dataset_path:
        dataset.save_to_disk(encoded_dataset_path)

    return dataset
