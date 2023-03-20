# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import shutil

from archai.datasets.cv.mnist_dataset_provider import MnistDatasetProvider


def test_mnist_dataset_provider():
    dataset_provider = MnistDatasetProvider()

    # Assert that we can individually load training, validation and test datasets
    train_dataset = dataset_provider.get_train_dataset()
    assert len(train_dataset) > 0
    assert isinstance(train_dataset[0][0], torch.Tensor)
    assert isinstance(train_dataset[0][1], int)

    val_dataset = dataset_provider.get_val_dataset()
    assert len(val_dataset) > 0
    assert isinstance(val_dataset[0][0], torch.Tensor)
    assert isinstance(val_dataset[0][1], int)

    test_dataset = dataset_provider.get_test_dataset()
    assert len(test_dataset) > 0
    assert isinstance(test_dataset[0][0], torch.Tensor)
    assert isinstance(test_dataset[0][1], int)

    shutil.rmtree("dataroot")
