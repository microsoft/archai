# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Tuple, Type, Optional
from abc import abstractmethod

from overrides import overrides, EnforceOverrides

from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from ..common.config import Config


TrainTestDatasets = Tuple[Optional[Dataset], Optional[Dataset]]

class DatasetProvider(EnforceOverrides):
    def __init__(self, conf_dataset:Config):
        super().__init__()
        pass

    @abstractmethod
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        pass

    @abstractmethod
    def get_transforms(self)->tuple: # of transforms
        pass

DatasetProviderType = type(DatasetProvider)
_providers: Dict[str, DatasetProviderType] = {}

def register_dataset_provider(name:str, class_type:DatasetProviderType)->None:
    global _providers
    if name in _providers:
        raise KeyError(f'dataset provider with name {name} has already been registered')
    _providers[name] = class_type

def get_provider_type(name:str)->DatasetProviderType:
    global _providers
    return _providers[name]