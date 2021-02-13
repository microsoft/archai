# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional
import os

from overrides import overrides, EnforceOverrides
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.datasets.providers.imagenet16 import ImageNet16
from archai.common.config import Config
from archai.common import utils


class ImageNet16120Provider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        # the ImageNet16 class returns PIL images and we 
        # want to have tensors for computing model stats in pre_fit
        loader = transforms.Compose([transforms.ToTensor()])  

        if load_train:
            trainpath = os.path.join(self._dataroot, 'imagenet16')
            trainset = ImageNet16(trainpath, True, loader, 120)
        if load_test:
            testpath = os.path.join(self._dataroot, 'imagenet16')
            testset = ImageNet16(testpath, False, loader, 120)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        # MEAN, STD from Xuanyi Dong [Github D-X-Y]        
        MEAN = [x / 255 for x in [122.68, 116.66, 104.01]]
        STD  = [x / 255 for x in [63.22,  61.26 , 65.09]]

        # transformations match that in
        
        train_transf = [
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(16, padding=2)            
        ]

        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]

        train_transform = transforms.Compose(train_transf + normalize)
        test_transform = transforms.Compose(normalize)

        return train_transform, test_transform

register_dataset_provider('ImageNet16-120', ImageNet16120Provider)