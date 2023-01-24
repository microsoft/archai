# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torchvision
from overrides import overrides
from torchvision.transforms import transforms

from archai.common.config import Config
from archai.supergraph.datasets.dataset_provider import (
    DatasetProvider,
    ImgSize,
    TrainTestDatasets,
    register_dataset_provider,
)
from archai.supergraph.utils import utils


class Cifar10Provider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = torchvision.datasets.CIFAR10(root=self._dataroot,
                train=True, download=True, transform=transform_train)
        if load_test:
            testset = torchvision.datasets.CIFAR10(root=self._dataroot,
                train=False, download=True, transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self, img_size:ImgSize)->tuple:
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
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

register_dataset_provider('cifar10', Cifar10Provider)