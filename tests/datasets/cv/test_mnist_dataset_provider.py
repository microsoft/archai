# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import shutil

from archai.datasets.cv.mnist_dataset_provider import MnistDatasetProvider


def test_mnist_dataset_provider():
    # make sure tests can run in parallel and not clobber each other's dataroot.
    unique_data_root = 'test_mnist_dataset_provider_dataroot'
    dataset_provider = MnistDatasetProvider(root=unique_data_root)

    # Assert that we can individually load training, validation and test datasets
    train_dataset = dataset_provider.get_train_dataset()
    assert len(train_dataset) == 60000
    assert isinstance(train_dataset[0][0], torch.Tensor)
    assert isinstance(train_dataset[0][1], int)

    val_dataset = dataset_provider.get_val_dataset()
    assert len(val_dataset) == 10000
    assert isinstance(val_dataset[0][0], torch.Tensor)
    assert isinstance(val_dataset[0][1], int)

    test_dataset = dataset_provider.get_test_dataset()
    assert len(test_dataset) == 10000
    assert isinstance(test_dataset[0][0], torch.Tensor)
    assert isinstance(test_dataset[0][1], int)

    shutil.rmtree(unique_data_root)
