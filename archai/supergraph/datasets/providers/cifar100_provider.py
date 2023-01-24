# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

import torchvision
from torchvision.transforms import transforms

from archai.supergraph.datasets.dataset_provider import DatasetProvider, ImgSize, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config
from archai.supergraph.utils import utils


class Cifar100Provider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = torchvision.datasets.CIFAR100(root=self._dataroot, train=True,
                download=True, transform=transform_train)
        if load_test:
            testset = torchvision.datasets.CIFAR100(root=self._dataroot, train=False,
                download=True, transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self, img_size:ImgSize)->tuple:
        MEAN = [0.507, 0.487, 0.441]
        STD = [0.267, 0.256, 0.276]
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

register_dataset_provider('cifar100', Cifar100Provider)