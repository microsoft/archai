# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torchvision
from overrides import overrides
from torch.utils.data import ConcatDataset
from torchvision.transforms import transforms

from archai.common import utils
from archai.common.config import Config
from archai.supergraph.datasets.dataset_provider import (
    DatasetProvider,
    ImgSize,
    TrainTestDatasets,
    register_dataset_provider,
)


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
    def get_transforms(self, img_size:ImgSize)->tuple:
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