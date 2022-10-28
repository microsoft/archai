# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for processing datasets, such as tokenization, shuffling, among others.
"""

import random
import re
from typing import Any, Dict, List, Optional, Union

from datasets import concatenate_datasets as hf_concatenate_datasets
from datasets import interleave_datasets as hf_interleave_datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

from archai.nlp.datasets.hf_datasets.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)


def map_dataset_to_dict(
    dataset: Union[Dataset, IterableDataset, List[Dataset], List[IterableDataset]],
    splits: Optional[Union[str, List[str]]] = None,
) -> Union[DatasetDict, IterableDatasetDict]:
    """Maps either an instance of dataset or list of datasets to a dictionary.

    Args:
        dataset: Input dataset.
        splits: Splits used to create the keys of dictionary.

    Returns:
        (Union[DatasetDict, IterableDatasetDict]): Dataset mapped as dictionary.

    """

    dataset = dataset if isinstance(dataset, list) else [dataset]
    splits = splits if isinstance(splits, list) else [splits]
    assert len(dataset) == len(splits), "`dataset` and `splits` should have same length."

    def _adjust_split_name(split_name: str) -> str:
        if not split_name:
            return "train"

        split_name = split_name.lower()

        split_name_match = re.search("train|validation|test", split_name)
        if split_name_match:
            return split_name_match.group()

        raise ValueError(f"split_name: {split_name} could not be matched.")

    splits = [_adjust_split_name(split) for split in splits]

    if isinstance(dataset[0], Dataset):
        dataset = DatasetDict({s: d for s, d in zip(splits, dataset)})
    elif isinstance(dataset[0], IterableDataset):
        dataset = IterableDatasetDict({s: d for s, d in zip(splits, dataset)})

    return dataset


def merge_datasets(
    datasets: Union[List[DatasetDict], List[IterableDatasetDict]]
) -> Union[DatasetDict, IterableDatasetDict]:
    """Merges a list of datasets.

    Args:
        datasets: Input datasets.

    Returns:
        (Union[DatasetDict, IterableDatasetDict]): Merged dataset.

    """

    available_splits = [list(dataset) for dataset in datasets]
    assert all(
        [available_splits[0] == splits for splits in available_splits]
    ), f"`datasets` must have identical splits: {available_splits}."

    available_typing = [type(dataset) for dataset in datasets]
    assert all(
        [available_typing[0] == typing for typing in available_typing]
    ), f"`datasets` must have identical types: {available_typing}."

    if isinstance(datasets[0], DatasetDict):
        dataset = DatasetDict(
            {split: hf_concatenate_datasets([dataset[split] for dataset in datasets]) for split in available_splits[0]}
        )

    if isinstance(datasets[0], IterableDatasetDict):
        dataset = IterableDatasetDict(
            {split: hf_interleave_datasets([dataset[split] for dataset in datasets]) for split in available_splits[0]}
        )

    return dataset


def resize_dataset(dataset: Dataset, n_samples: int) -> Dataset:
    """Resizes a dataset according to a supplied size.

    Args:
        dataset: Input dataset.
        n_samples: Amount of samples.

    Returns:
        (Dataset): Resized dataset.

    """

    if n_samples > -1:
        dataset = dataset.select(range(n_samples))

    return dataset


def shuffle_dataset(dataset: Union[Dataset, IterableDataset], seed: int) -> Union[Dataset, IterableDataset]:
    """Shuffles a dataset according to a supplied seed.

    Args:
        dataset: Input dataset.
        seed: Random seed.

    Returns:
        (Union[Dataset, IterableDataset]): Shuffled dataset.

    """

    if seed > -1:
        dataset = dataset.shuffle(seed)

    return dataset


def tokenize_dataset(
    dataset: Union[DatasetDict, IterableDatasetDict],
    tokenizer: Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast],
    mapping_column_name: Optional[str] = "text",
    next_sentence_prediction: Optional[bool] = False,
    truncate: Optional[bool] = True,
    padding: Optional[str] = "max_length",
    batched: Optional[bool] = True,
    batch_size: Optional[int] = 1000,
    writer_batch_size: Optional[int] = 1000,
    num_proc: Optional[int] = None,
) -> Union[DatasetDict, IterableDatasetDict]:
    """Tokenizes a dataset according to supplied tokenizer and constraints.

    Args:
        dataset: Input dataset.
        tokenizer: Input tokenizer.
        mapping_column_name: Defines column to be tokenized.
        next_sentence_prediction: Whether next sentence prediction labels should be created or not.
        truncate: Whether samples should be truncated or not.
        padding: Strategy used to pad samples that do not have the proper size.
        batched: Whether mapping should be batched or not.
        batch_size: Number of examples per batch.
        writer_batch_size: Number of examples per write operation to cache.
        num_proc: Number of processes for multiprocessing.

    Returns:
        (Union[DatasetDict, IterableDatasetDict]): Tokenized dataset.

    """

    if isinstance(mapping_column_name, str):
        mapping_column_name = (mapping_column_name,)
    elif isinstance(mapping_column_name, list):
        mapping_column_name = tuple(mapping_column_name)

    def _tokenize(examples: List[str]) -> Dict[str, Any]:
        examples_mapping = tuple(examples[column_name] for column_name in mapping_column_name)
        return tokenizer(*examples_mapping, truncation=truncate, padding=padding)

    def _tokenize_nsp(examples: List[str]) -> Dict[str, Any]:
        assert (
            len(mapping_column_name) == 1
        ), "next_sentence_prediction requires a single value for mapping_column_name."
        examples_mapping = examples[mapping_column_name[0]]

        examples, next_sentence_labels = [], []
        for i in range(len(examples_mapping)):
            if random.random() < 0.5:
                examples.append(examples_mapping[i])
                next_sentence_labels.append(0)
            else:
                examples.append(random.choices(examples_mapping, k=2))
                next_sentence_labels.append(1)

        tokenized_examples = tokenizer(examples, truncation=truncate, padding=padding)
        tokenized_examples["next_sentence_label"] = next_sentence_labels
        return tokenized_examples

    map_kwargs = {"batched": batched}
    if isinstance(dataset, DatasetDict):
        map_kwargs["batch_size"] = batch_size
        map_kwargs["writer_batch_size"] = writer_batch_size
        map_kwargs["num_proc"] = num_proc

    if next_sentence_prediction:
        return dataset.map(
            _tokenize_nsp,
            **map_kwargs,
        )

    return dataset.map(_tokenize, **map_kwargs)
