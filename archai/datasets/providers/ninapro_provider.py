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


def load_ninapro_data(path, train=True):
    ''' Modified from 
    https://github.com/rtu715/NAS-Bench-360/blob/ba7ff6bd0762073d1ce49207b95245c5c742b567/backbone/data_utils/load_data.py#L396 '''

    trainset = load_ninapro(path, 'train')
    valset = load_ninapro(path, 'val')
    testset = load_ninapro(path, 'test')

    if train:
        return trainset, valset, testset
    else:
        targets = torch.cat((trainset.targets, valset.targets))
        trainset = data_utils.ConcatDataset([trainset, valset])
        trainset.targets = targets # for compatibility with stratified sampler

    return trainset, None, testset


def load_ninapro(path, whichset):
    ''' Modified from 
    https://github.com/rtu715/NAS-Bench-360/blob/ba7ff6bd0762073d1ce49207b95245c5c742b567/backbone/data_utils/load_data.py#L396 '''

    data_str = 'ninapro_' + whichset + '.npy'
    label_str = 'label_' + whichset + '.npy'

    data = np.load(os.path.join(path, data_str),
                             encoding="bytes", allow_pickle=True)
    labels = np.load(os.path.join(path, label_str), encoding="bytes", allow_pickle=True)

    data = np.transpose(data, (0, 2, 1))
    data = data[:, None, :, :]
    data = torch.from_numpy(data.astype(np.float32))
    labels = torch.from_numpy(labels.astype(np.int64))

    all_data = data_utils.TensorDataset(data, labels)
    all_data.targets = labels # for compatibility with stratified data sampler
    return all_data



class NinaproProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        path_to_data = os.path.join(self._dataroot, 'ninapro')

        # load the dataset but without any validation split
        trainset, _, testset = load_ninapro_data(path_to_data, train=False)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        # return empty transforms since we have preprocessed data
        train_transf = []
        test_transf = []

        train_transform = transforms.Compose(train_transf)
        test_transform = transforms.Compose(test_transf)
        return train_transform, test_transform

register_dataset_provider('ninapro', NinaproProvider)