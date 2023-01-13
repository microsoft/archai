# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import AutoTokenizer

from archai.nlp.datasets.hf.loaders import (
    DatasetDict,
    DownloadMode,
    IterableDatasetDict,
    _should_refresh_cache,
    encode_dataset,
    load_dataset,
)


def test_should_refresh_cache():
    # Test that the function returns FORCE_REDOWNLOAD when refresh is True
    assert _should_refresh_cache(True) == DownloadMode.FORCE_REDOWNLOAD

    # Test that the function returns REUSE_DATASET_IF_EXISTS when refresh is False
    assert _should_refresh_cache(False) == DownloadMode.REUSE_DATASET_IF_EXISTS


def test_load_dataset():
    dataset_name = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"

    # Assert loading dataset from Hugging Face Hub
    dataset = load_dataset(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        dataset_refresh_cache=True,
    )
    assert isinstance(dataset, (DatasetDict, IterableDatasetDict))

    # Assert that subsampling works
    n_samples = 10
    dataset = dataset = load_dataset(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        dataset_refresh_cache=True,
        n_samples=n_samples,
    )
    assert all(len(split) == n_samples for split in dataset.values())


def test_encode_dataset():
    dataset = load_dataset(
        dataset_name="wikitext", dataset_config_name="wikitext-2-raw-v1", dataset_refresh_cache=True, n_samples=10
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Assert that dataaset can be encoded
    encoded_dataset = encode_dataset(dataset, tokenizer)
    assert isinstance(encoded_dataset, (DatasetDict, IterableDatasetDict))

    # Assert that dataset can be encoded with custom mapping function
    def custom_mapping_fn(example, tokenizer, mapping_column_name=None):
        example["text2"] = example["text"]
        return example

    encoded_dataset = encode_dataset(dataset, tokenizer, mapping_fn=custom_mapping_fn)
    assert isinstance(encoded_dataset, (DatasetDict, IterableDatasetDict))
