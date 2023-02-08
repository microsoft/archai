# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torchvision
from overrides import overrides
from torchvision.transforms import transforms

from archai.common import utils
from archai.common.config import Config
from archai.supergraph.datasets.dataset_provider import (
    DatasetProvider,
    ImgSize,
    TrainTestDatasets,
    register_dataset_provider,
)


class MnistProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = torchvision.datasets.MNIST(root=self._dataroot, train=True,
                download=True, transform=transform_train)
        if load_test:
            testset = torchvision.datasets.MNIST(root=self._dataroot, train=False,
                download=True, transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self, img_size:ImgSize)->tuple:
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                scale=(0.9, 1.1), shear=0.1)
        ]

        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]

        train_transform = transforms.Compose(transf + normalize)
        test_transform = transforms.Compose(normalize)

        return train_transform, test_transform

register_dataset_provider('mnist', MnistProvider)