# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil

from archai.datasets.nlp.nvidia_dataset_provider import NvidiaDatasetProvider


def test_nvidia_dataset_provider():
    os.makedirs("dataroot/olx_tmp")
    with open("dataroot/olx_tmp/train.txt", "w") as f:
        f.write("train")
    with open("dataroot/olx_tmp/valid.txt", "w") as f:
        f.write("valid")
    with open("dataroot/olx_tmp/test.txt", "w") as f:
        f.write("test")

    # Assert that we can individually load training, validation and test datasets
    dataset_provider = NvidiaDatasetProvider("olx_tmp", dataset_dir="dataroot/olx_tmp", refresh_cache=True)

    train_dataset = dataset_provider.get_train_dataset()
    assert len(train_dataset) == 7

    val_dataset = dataset_provider.get_val_dataset()
    assert len(val_dataset) == 7

    test_dataset = dataset_provider.get_test_dataset()
    assert len(test_dataset) == 6

    shutil.rmtree("cache")
    shutil.rmtree("dataroot")
