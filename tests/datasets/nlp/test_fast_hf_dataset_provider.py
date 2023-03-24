# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil
from archai.datasets.nlp.fast_hf_dataset_provider import FastHfDatasetProvider


def test_fast_hf_dataset_provider_from_hub():
    dataset_provider = FastHfDatasetProvider.from_hub(
        "glue",
        dataset_config_name="sst2",
        tokenizer_name="Salesforce/codegen-350M-mono",
        mapping_column_name=["sentence"],
        use_shared_memory=False,
    )

    # Assert that we can individually load training, validation and test datasets
    train_dataset = dataset_provider.get_train_dataset(seq_len=256)
    assert len(train_dataset) == 3514

    val_dataset = dataset_provider.get_val_dataset(seq_len=256)
    assert len(val_dataset) == 85

    test_dataset = dataset_provider.get_test_dataset(seq_len=256)
    assert len(test_dataset) == 169


def test_fast_hf_dataset_provider_from_cache():
    dataset_provider = FastHfDatasetProvider.from_cache("cache")

    # Assert that we can individually load training, validation and test datasets
    train_dataset = dataset_provider.get_train_dataset(seq_len=256)
    assert len(train_dataset) == 3514

    val_dataset = dataset_provider.get_val_dataset(seq_len=256)
    assert len(val_dataset) == 85

    test_dataset = dataset_provider.get_test_dataset(seq_len=256)
    assert len(test_dataset) == 169

    shutil.rmtree("cache")
