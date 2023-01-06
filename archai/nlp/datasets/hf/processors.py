# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Utilities for processing datasets, such as tokenization, shuffling, among others."""

import random
import re
from itertools import chain
from typing import Any, Dict, List, Optional, Union

from datasets import concatenate_datasets as hf_concatenate_datasets
from datasets import interleave_datasets as hf_interleave_datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

from archai.nlp.datasets.hf.tokenizer_utils.pre_trained_tokenizer import (
    ArchaiPreTrainedTokenizerFast,
)


def map_dataset_to_dict(
    dataset: Union[Dataset, IterableDataset, List[Dataset], List[IterableDataset]],
    splits: Union[str, List[str]],
) -> Union[DatasetDict, IterableDatasetDict]:
    """Map a dataset or list of datasets to a dictionary.

    This function maps either a single dataset or a list of datasets to a
    dictionary, using the provided `splits` as the keys. If `splits` is not
    provided, the keys will be determined by the type of the dataset(s)
    (e.g. 'train', 'validation', 'test').

    Args:
        dataset: The input dataset(s).
        splits: The splits to use as the keys of the dictionary.
            If `str`, only one split will be used.
            If `List[str]`, multiple splits will be used.

    Returns:
        Dataset mapped as dictionary.

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
    """Merge a list of datasets.

    This function merges a list of datasets, which must have identical splits
    and be of the same type (either `DatasetDict` or `IterableDatasetDict`).
    The resulting dataset will be of the same type as the input datasets.

    Args:
        datasets: The input datasets to be merged.

    Returns:
        The merged dataset.

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
    """Resize a dataset to a specified number of samples.

    This function resizes a dataset to the specified number of samples, by
    selecting a subset of the original dataset. If `n_samples` is negative,
    the original dataset will be returned unmodified.

    Args:
        dataset: The input dataset to be resized.
        n_samples: The number of samples to retain in the resized dataset.

    Returns:
        The resized dataset.

    """

    if n_samples > -1:
        dataset = dataset.select(range(n_samples))

    return dataset


def shuffle_dataset(dataset: Union[Dataset, IterableDataset], seed: int) -> Union[Dataset, IterableDataset]:
    """Shuffle a dataset using a specified random seed.

    This function shuffles a dataset using the provided random seed. If
    `seed` is negative, the original dataset will be returned unmodified.
    The resulting dataset will be of the same type as the input dataset.

    Args:
        dataset: The input dataset to be shuffled.
        seed: The random seed to use for shuffling the dataset.

    Returns:
        The shuffled dataset.

    """

    if seed > -1:
        dataset = dataset.shuffle(seed)

    return dataset


def tokenize_dataset(
    examples: List[str],
    tokenizer: Optional[Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast]] = None,
    mapping_column_name: Optional[List[str]] = None,
    truncate: Optional[Union[bool, str]] = True,
    padding: Optional[Union[bool, str]] = "max_length",
    **kwargs,
) -> Dict[str, Any]:
    """Tokenize a list of examples using a specified tokenizer.

    Args:
        examples: A list of examples to be tokenized.
        tokenizer: The tokenizer to use.
        mapping_column_name: The columns in `examples` that should be tokenized.
        truncate: Whether truncation should be applied.
        padding: Whether padding should be applied.

    Returns:
        Tokenized examples.

    """

    if mapping_column_name is None:
        mapping_column_name = ["text"]

    examples_mapping = tuple(examples[column_name] for column_name in mapping_column_name)

    return tokenizer(*examples_mapping, truncation=truncate, padding=padding)


def tokenize_contiguous_dataset(
    examples: List[str],
    tokenizer: Optional[Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast]] = None,
    mapping_column_name: Optional[List[str]] = None,
    model_max_length: Optional[int] = 1024,
    **kwargs,
) -> Dict[str, Any]:
    """Tokenize a list of examples using a specified tokenizer and
    with contiguous-length batches (no truncation nor padding).

    Args:
        examples: A list of examples to be tokenized.
        tokenizer: The tokenizer to use.
        mapping_column_name: The columns in `examples` that should be tokenized.
        model_max_length: Maximum length of sequences.

    Returns:
        Contiguous-length tokenized examples.

    """

    examples = tokenize_dataset(
        examples, mapping_column_name=mapping_column_name, tokenizer=tokenizer, truncate=False, padding=False
    )

    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= model_max_length:
        total_length = (total_length // model_max_length) * model_max_length

    result = {
        k: [t[i : i + model_max_length] for i in range(0, total_length, model_max_length)]
        for k, t in concatenated_examples.items()
    }

    return result


def tokenize_nsp_dataset(
    examples: List[str],
    tokenizer: Optional[Union[AutoTokenizer, ArchaiPreTrainedTokenizerFast]] = None,
    mapping_column_name: Optional[List[str]] = None,
    truncate: Optional[Union[bool, str]] = True,
    padding: Optional[Union[bool, str]] = "max_length",
    **kwargs,
) -> Dict[str, Any]:
    """Tokenizes a list of examples using a specified tokenizer and
    with next-sentence prediction (NSP).

    Args:
        examples: A list of examples to be tokenized.
        tokenizer: The tokenizer to use.
        mapping_column_name: The columns in `examples` that should be tokenized.
        truncate: Whether truncation should be applied.
        padding: Whether padding should be applied.

    Returns:
        Tokenized examples with NSP labels.

    """

    if mapping_column_name is None:
        mapping_column_name = ["text"]

    assert len(mapping_column_name) == 1, "`mapping_column_name` must have a single value."
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
