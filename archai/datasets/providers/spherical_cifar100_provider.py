# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional
import os
import gzip
import pickle
import numpy as np 

from overrides import overrides, EnforceOverrides
import torch
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config
from archai.common import utils


def load_spherical_data(path, val_split=0.0):
    ''' Modified from 
    https://github.com/rtu715/NAS-Bench-360/blob/main/backbone/utils_pt.py '''

    data_file = os.path.join(path, 's2_cifar100.gz')
    with gzip.open(data_file, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32))
    train_data = torch.squeeze(train_data)
    # DEBUG
    train_data = torch.nn.functional.interpolate(train_data, size=(32, 32))

    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))

    all_train_dataset = data_utils.TensorDataset(train_data, train_labels)
    if val_split == 0.0:
        val_dataset = None
        train_dataset = all_train_dataset
        train_dataset.targets = train_labels # compatibility with stratified sampler
    else:
        ntrain = int((1-val_split) * len(all_train_dataset))
        train_dataset = data_utils.TensorDataset(train_data[:ntrain], train_labels[:ntrain])
        train_dataset.targets = train_labels[:ntrain] # compatibility with stratified sampler
        val_dataset = data_utils.TensorDataset(train_data[ntrain:], train_labels[ntrain:])
        val_dataset.targets = train_labels[ntrain:] # compatibility with stratified sampler
        
    test_data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_data = torch.squeeze(test_data)

    # DEBUG
    test_data = torch.nn.functional.interpolate(test_data, size=(32,32))

    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    # compatibility with stratified sampler
    test_dataset.targets = test_labels

    return train_dataset, val_dataset, test_dataset


class SphericalCifar100Provider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        path_to_data = os.path.join(self._dataroot, 'sphericalcifar100')

        # load the dataset but without any validation split
        trainset, _, testset = load_spherical_data(path_to_data, val_split=0.0)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        # return empty transforms since we have preprocessed data
        train_transf = []
        test_transf = []

        train_transform = transforms.Compose(train_transf)
        test_transform = transforms.Compose(test_transf)
        return train_transform, test_transform

register_dataset_provider('sphericalcifar100', SphericalCifar100Provider)