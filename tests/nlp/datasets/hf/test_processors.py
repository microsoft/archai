# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from transformers import AutoTokenizer

from archai.nlp.datasets.hf.loaders import (
    DatasetDict,
    IterableDatasetDict,
    load_dataset,
)
from archai.nlp.datasets.hf.processors import (
    map_dataset_to_dict,
    merge_datasets,
    resize_dataset,
    shuffle_dataset,
    tokenize_contiguous_dataset,
    tokenize_dataset,
    tokenize_nsp_dataset,
)


@pytest.fixture
def dataset():
    return load_dataset(
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        dataset_refresh_cache=True,
    )


@pytest.fixture
def dataset_train_set():
    return load_dataset(
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        dataset_refresh_cache=True,
        dataset_split="train",
    )


@pytest.fixture
def dataset_val_set():
    return load_dataset(
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        dataset_refresh_cache=True,
        dataset_split="validation",
    )


@pytest.fixture
def iterable_dataset():
    return load_dataset(
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        dataset_refresh_cache=True,
        dataset_stream=True,
    )


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=8)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_map_dataset_to_dict(dataset):
    # Assert mapping single dataset to dictionary
    dataset = dataset["test"]
    dataset_dict = map_dataset_to_dict(dataset, "test")
    assert isinstance(dataset_dict, (DatasetDict, IterableDatasetDict))

    # Assert mapping multiple datasets to dictionary
    datasets = [dataset for _ in range(3)]
    dataset_dict = map_dataset_to_dict(datasets, ["test", "test", "test"])
    assert isinstance(dataset_dict, (DatasetDict, IterableDatasetDict))


def test_merge_datasets(dataset, dataset_train_set, dataset_val_set, iterable_dataset):
    # Assert that dataset can be merged
    datasets = [dataset for _ in range(3)]
    merged_dataset = merge_datasets(datasets)
    assert isinstance(merged_dataset, (DatasetDict, IterableDatasetDict))
    assert len(merged_dataset) == 3
    assert len(list(merged_dataset.values())[0]) == len(list(dataset.values())[0]) * 3

    # Assert that dataset can not be merged with different splits
    datasets = [dataset_train_set, dataset_val_set]
    with pytest.raises(AssertionError):
        merged_dataset = merge_datasets(datasets)

    # Assert that dataset can not be merged with different types
    datasets = [dataset, iterable_dataset]
    with pytest.raises(AssertionError):
        merged_dataset = merge_datasets(datasets)


def test_resize_dataset(dataset):
    dataset = dataset["train"]

    # Assert resizing dataset to smaller size
    resized_dataset = resize_dataset(dataset, 10)
    assert len(resized_dataset) == 10

    # Assert resizing dataset to larger size
    resized_dataset = resize_dataset(dataset, 10000)
    assert len(resized_dataset) == 10000

    # Assert resizing dataset to same size
    resized_dataset = resize_dataset(dataset, len(dataset))
    assert len(resized_dataset) == len(dataset)


def test_shuffle_dataset(dataset):
    dataset = dataset["train"]

    # Assert shuffling dataset with positive seed
    shuffled_dataset = shuffle_dataset(dataset, 42)
    assert len(shuffled_dataset) == len(dataset)
    assert isinstance(shuffled_dataset, type(dataset))

    # Assert shuffling dataset with negative seed
    shuffled_dataset = shuffle_dataset(dataset, -1)
    assert len(shuffled_dataset) == len(dataset)
    assert isinstance(shuffled_dataset, type(dataset))


def test_tokenize_dataset(tokenizer):
    # Assert that examples can be tokenized
    examples = {"text": ["Hello, this is a test.", "This is another test."]}
    expected_output = {
        "input_ids": [[15496, 11, 428, 318, 257, 1332, 13, 50256], [1212, 318, 1194, 1332, 13, 50256, 50256, 50256]],
        "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 0]],
    }

    output = tokenize_dataset(examples, tokenizer=tokenizer)
    assert output == expected_output


def test_tokenize_contiguous_dataset(tokenizer):
    # Assert that examples can be contiguously tokenized
    examples = {"text": ["This is a test example.", "This is another test example.", "And yet another test example."]}
    output = tokenize_contiguous_dataset(examples, tokenizer=tokenizer, model_max_length=8)
    assert len(output["input_ids"][0]) == 8
    assert len(output["input_ids"][1]) == 8
    with pytest.raises(IndexError):
        assert len(output["input_ids"][2]) == 8


def test_tokenize_nsp_dataset(tokenizer):
    # Assert that a single example can be tokenized
    examples = {"text": ["This is a single example."]}
    tokenized_examples = tokenize_nsp_dataset(examples, tokenizer=tokenizer)
    assert tokenized_examples["next_sentence_label"][0] in [0, 1]

    # Assert that multiple examples can be tokenized
    examples = {"text": ["This is the first example.", "This is the second example."]}
    tokenized_examples = tokenize_nsp_dataset(examples, tokenizer=tokenizer)
    assert tokenized_examples["next_sentence_label"][0] in [0, 1]
    assert tokenized_examples["next_sentence_label"][1] in [0, 1]
