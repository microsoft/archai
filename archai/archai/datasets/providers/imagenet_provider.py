# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional
import os
import shutil
import torch

import torchvision
from torchvision.transforms import transforms
from torchvision import datasets
from torchvision.datasets.utils import check_integrity, download_url

from PIL import Image

from overrides import overrides, EnforceOverrides

from archai.common.common import logger
from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config
from archai.common import utils
from archai.datasets.transforms.lighting import Lighting
from .imagenet_folder import ImageNetFolder

class ImagenetProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                    transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = datasets.ImageFolder(root=os.path.join(self._dataroot, 'ImageNet', 'train'),
                transform=transform_train)
            # compatibility with older PyTorch
            if not hasattr(trainset, 'targets'):
                trainset.targets = [lb for _, lb in trainset.samples]
        if load_test:
            testset = datasets.ImageFolder(root=os.path.join(self._dataroot, 'ImageNet', 'val'),
                transform=transform_test)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        _IMAGENET_PCA = {
            'eigval': [0.2175, 0.0188, 0.0045],
            'eigvec': [
                [-0.5675,  0.7192,  0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948,  0.4203],
            ]
        }

        transform_train, transform_test = None, None

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224,
                scale=(0.08, 1.0), # TODO: these two params are normally not specified
                interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2
            ),
            transforms.ToTensor(),
            # TODO: Lighting is not used in original darts paper
            # Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

        return transform_train, transform_test

register_dataset_provider('imagenet', ImagenetProvider)
