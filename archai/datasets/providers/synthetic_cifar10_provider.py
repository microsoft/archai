# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional
import os

from overrides import overrides, EnforceOverrides

import torch
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision.transforms import transforms

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config
from archai.common import utils


class SyntheticCifar10Provider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainpath = os.path.join(self._dataroot, 'synthetic_cifar10', 'train')
            trainset = torchvision.datasets.DatasetFolder(trainpath, loader=torch.load, extensions='.pt' ,transform=None)
        if load_test:
            testpath = os.path.join(self._dataroot, 'synthetic_cifar10', 'test')
            testset = torchvision.datasets.DatasetFolder(testpath, loader=torch.load, extensions='.pt', transform=None)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        MEAN = [0.4323, 0.4323, 0.4323]
        STD = [0.3192, 0.3192, 0.3192]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]

        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]

        train_transform = transforms.Compose(transf + normalize)
        test_transform = transforms.Compose(normalize)

        return train_transform, test_transform

register_dataset_provider('synthetic_cifar10', SyntheticCifar10Provider)