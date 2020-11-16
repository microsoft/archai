# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional

from overrides import overrides, EnforceOverrides
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms
from torch.utils.data import ConcatDataset

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config
from archai.common import utils


class SvhnProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = torchvision.datasets.SVHN(root=self._dataroot, split='train',
                download=True, transform=transform_train)
            extraset = torchvision.datasets.SVHN(root=self._dataroot, split='extra',
                download=True, transform=transform_train)
            trainset = ConcatDataset([trainset, extraset])
        if load_test:
            testset = torchvision.datasets.SVHN(root=self._dataroot, split='test',
                download=True, transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        MEAN = [0.4914, 0.4822, 0.4465]
        STD = [0.2023, 0.1994, 0.20100]
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

register_dataset_provider('svhn', SvhnProvider)