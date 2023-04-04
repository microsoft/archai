# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil
from archai.datasets.nlp.fast_hf_dataset_provider import FastHfDatasetProvider

TEST_CACHE_DIR='test_fast_hf_dataset_cache'

def test_fast_hf_dataset_provider_from_hub():
    dataset_provider = FastHfDatasetProvider.from_hub(
        "glue",
        dataset_config_name="sst2",
        tokenizer_name="Salesforce/codegen-350M-mono",
        mapping_column_name=["sentence"],
        use_shared_memory=False,
        cache_dir=TEST_CACHE_DIR
    )

    # Assert that we can individually load training, validation and test datasets
    with dataset_provider.get_train_dataset(seq_len=256) as train_dataset:
        assert len(train_dataset) == 3514

    with dataset_provider.get_val_dataset(seq_len=256) as val_dataset:
        assert len(val_dataset) == 85

    with dataset_provider.get_test_dataset(seq_len=256) as test_dataset:
        assert len(test_dataset) == 169


def test_fast_hf_dataset_provider_from_cache():
    dataset_provider = FastHfDatasetProvider.from_cache(TEST_CACHE_DIR)

    # Assert that we can individually load training, validation and test datasets
    with dataset_provider.get_train_dataset(seq_len=256) as train_dataset:
        assert len(train_dataset) == 3514

    with dataset_provider.get_val_dataset(seq_len=256) as val_dataset:
        assert len(val_dataset) == 85

    with dataset_provider.get_test_dataset(seq_len=256) as test_dataset:
        assert len(test_dataset) == 169

    shutil.rmtree(TEST_CACHE_DIR)
