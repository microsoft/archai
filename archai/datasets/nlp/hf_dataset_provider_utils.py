# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.download.download_manager import DownloadMode
from datasets.iterable_dataset import IterableDataset
from transformers.models.auto.tokenization_auto import AutoTokenizer


def should_refresh_cache(refresh: bool) -> DownloadMode:
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


def tokenize_dataset(
    examples: Dict[str, List[str]],
    tokenizer: Optional[AutoTokenizer] = None,
    mapping_column_name: Optional[List[str]] = None,
    use_eos_token: Optional[bool] = False,
    truncate: Optional[Union[bool, str]] = True,
    padding: Optional[Union[bool, str]] = "max_length",
) -> Dict[str, Any]:
    """Tokenize a list of examples using a specified tokenizer.

    Args:
        examples: A list of examples to be tokenized.
        tokenizer: The tokenizer to use.
        mapping_column_name: The columns in `examples` that should be tokenized.
        use_eos_token: Whether to append the EOS token to each example.
        truncate: Whether truncation should be applied.
        padding: Whether padding should be applied.

    Returns:
        Tokenized examples.

    """

    def _add_eos_token(examples: List[str]) -> List[str]:
        return [example + tokenizer.eos_token if example else example for example in examples]

    if mapping_column_name is None:
        mapping_column_name = ["text"]

    examples_mapping = tuple(
        _add_eos_token(examples[column_name]) if use_eos_token else examples[column_name]
        for column_name in mapping_column_name
    )

    return tokenizer(*examples_mapping, truncation=truncate, padding=padding)


def tokenize_concatenated_dataset(
    examples: Dict[str, List[str]],
    tokenizer: Optional[AutoTokenizer] = None,
    mapping_column_name: Optional[List[str]] = None,
    use_eos_token: Optional[bool] = False,
    dtype: Optional[np.dtype] = None,
) -> Dict[str, Any]:
    """Tokenize a list of examples using a specified tokenizer and
    with concatenated batches (no truncation nor padding).

    Args:
        examples: A list of examples to be tokenized.
        tokenizer: The tokenizer to use.
        mapping_column_name: The columns in `examples` that should be tokenized.
        use_eos_token: Whether to append the EOS token to each example.
        dtype: Numpy data type of the tokenized examples.

    Returns:
        Concatenated tokenized examples.

    """

    examples = tokenize_dataset(
        examples,
        tokenizer=tokenizer,
        mapping_column_name=mapping_column_name,
        use_eos_token=use_eos_token,
        truncate=False,
        padding=False,
    )
    tokenized_examples = np.fromiter(chain(*examples["input_ids"]), dtype=dtype)

    return {"input_ids": [tokenized_examples], "length": [len(tokenized_examples)]}


def tokenize_contiguous_dataset(
    examples: Dict[str, List[str]],
    tokenizer: Optional[AutoTokenizer] = None,
    mapping_column_name: Optional[List[str]] = None,
    model_max_length: Optional[int] = 1024,
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
    examples: Dict[str, List[str]],
    tokenizer: Optional[AutoTokenizer] = None,
    mapping_column_name: Optional[List[str]] = None,
    truncate: Optional[Union[bool, str]] = True,
    padding: Optional[Union[bool, str]] = "max_length",
) -> Dict[str, Any]:
    """Tokenize a list of examples using a specified tokenizer and
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


def encode_dataset(
    dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
    tokenizer: AutoTokenizer,
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

    mapping_kwargs = {"batched": batched}
    if isinstance(dataset, DatasetDict):
        remove_columns = [v.column_names for _, v in dataset.items()]
        assert all([c[0] for c in remove_columns])

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
