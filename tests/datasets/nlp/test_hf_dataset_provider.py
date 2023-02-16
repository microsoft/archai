# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil

from archai.datasets.nlp.hf_dataset_provider import (
    HfDiskDatasetProvider,
    HfHubDatasetProvider,
)


def test_hf_hub_dataset_provider():
    dataset_provider = HfHubDatasetProvider("glue", dataset_config_name="sst2")

    # Assert that we can individually load training, validation and test datasets
    train_dataset = dataset_provider.get_train_dataset()
    assert len(train_dataset) == 67349

    val_dataset = dataset_provider.get_val_dataset()
    assert len(val_dataset) == 872

    test_dataset = dataset_provider.get_test_dataset()
    assert len(test_dataset) == 1821


def test_hf_disk_dataset_provider():
    dataset_hub_provider = HfHubDatasetProvider("glue", dataset_config_name="sst2")
    train_dataset = dataset_hub_provider.get_train_dataset()
    train_dataset.save_to_disk("dataroot")

    # Assert that we can load the dataset from disk
    dataset_provider = HfDiskDatasetProvider("dataroot")
    train_dataset = dataset_provider.get_train_dataset()
    assert len(train_dataset) == 67349

    shutil.rmtree("dataroot")
