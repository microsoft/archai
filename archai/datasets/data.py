from typing import List, Tuple, Union, Optional

import os
import sys
import math

import torch
import torchvision
from PIL import Image

from torch.utils.data import \
    SubsetRandomSampler, Sampler, Subset, ConcatDataset, Dataset, random_split,\
    DataLoader
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.distributed import DistributedSampler

from .augmentation import add_named_augs
from ..common.common import logger
from ..common.common import utils
from archai.datasets.dataset_provider import DatasetProvider, get_provider_type
from ..common.config import Config
from .limit_dataset import LimitDataset, DatasetLike
from .distributed_stratified_sampler import DistributedStratifiedSampler

def get_data(conf_loader:Config)\
        -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    # region conf vars
    # dataset
    conf_data = conf_loader['dataset']
    max_batches = conf_data['max_batches']

    aug = conf_loader['aug']
    cutout = conf_loader['cutout']
    val_ratio = conf_loader['val_ratio']
    val_fold = conf_loader['val_fold']
    load_train = conf_loader['load_train']
    train_batch = conf_loader['train_batch']
    train_workers = conf_loader['train_workers']
    load_test = conf_loader['load_test']
    test_batch = conf_loader['test_batch']
    test_workers = conf_loader['test_workers']
    # endregion

    ds_provider = create_dataset_provider(conf_data)

    train_dl, val_dl, test_dl = get_dataloaders(ds_provider,
        load_train=load_train, train_batch_size=train_batch,
        load_test=load_test, test_batch_size=test_batch,
        aug=aug, cutout=cutout,  val_ratio=val_ratio, val_fold=val_fold,
        train_workers=train_workers, test_workers=test_workers,
        max_batches=max_batches)

    assert train_dl is not None
    return train_dl, val_dl, test_dl

def create_dataset_provider(conf_data:Config)->DatasetProvider:
    ds_name = conf_data['name']
    dataroot = conf_data['dataroot']
    storage_name = conf_data['storage_name']

    logger.info({'ds_name': ds_name, 'dataroot':dataroot, 'storage_name':storage_name})

    ds_provider_type = get_provider_type(ds_name)
    return ds_provider_type(conf_data)

def get_dataloaders(ds_provider:DatasetProvider,
    load_train:bool, train_batch_size:int,
    load_test:bool, test_batch_size:int,
    aug, cutout:int, val_ratio:float, val_fold=0,
    train_workers:Optional[int]=None, test_workers:Optional[int]=None,
    target_lb=-1, max_batches:int=-1) \
        -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:

    # if debugging in vscode, workers > 0 gets termination
    if utils.is_debugging():
        train_workers = test_workers = 0
        logger.warn({'debugger': True})
    if train_workers is None:
        train_workers = 4
    if test_workers is None:
        test_workers = 4
    logger.info({'train_workers': train_workers, 'test_workers':test_workers})

    transform_train, transform_test = ds_provider.get_transforms()
    add_named_augs(transform_train, aug, cutout)

    trainset, testset = _get_datasets(ds_provider,
        load_train, load_test, transform_train, transform_test)

    # TODO: below will never get executed, set_preaug does not exist in PyTorch
    # if total_aug is not None and augs is not None:
    #     trainset.set_preaug(augs, total_aug)
    #     logger.info('set_preaug-')

    trainloader, validloader, testloader, train_sampler = None, None, None, None

    max_train_fold = max_batches*train_batch_size if max_batches else None
    max_test_fold = max_batches*test_batch_size if max_batches else None
    logger.info({'val_ratio': val_ratio, 'max_batches': max_batches,
                    'max_train_fold': max_train_fold, 'max_test_fold': max_test_fold})

    if trainset:
        # sample validation set from trainset if cv_ratio > 0
        train_sampler, valid_sampler = _get_sampler(trainset, val_ratio=val_ratio,
                                                    shuffle=True,
                                                    max_items=max_train_fold)

        # shuffle is performed by sampler at each epoch
        trainloader = DataLoader(trainset,
            batch_size=train_batch_size, shuffle=False,
            num_workers=round((1-val_ratio)*train_workers),
            pin_memory=True,
            sampler=train_sampler, drop_last=False) # TODO: original paper has this True

        if val_ratio > 0.0:
            validloader = DataLoader(trainset,
                batch_size=train_batch_size, shuffle=False,
                num_workers=round(val_ratio*train_workers),  # if val_ratio = 0.5, then both sets re same
                pin_memory=True, #TODO: set n_workers per ratio?
                sampler=valid_sampler, drop_last=False)
        # else validloader is left as None
    if testset:
        test_sampler, _ = _get_sampler(testset, val_ratio=0.0,
                                       shuffle=False,
                                       max_items=max_test_fold)
        testloader = DataLoader(testset,
            batch_size=test_batch_size, shuffle=False,
            num_workers=test_workers,
            pin_memory=True,
            sampler=test_sampler, drop_last=False
    )

    assert val_ratio > 0.0 or validloader is None

    logger.info({
        'train_batch_size': train_batch_size, 'test_batch_size': test_batch_size,
        'train_batches': len(trainloader) if trainloader is not None else None,
        'val_batches': len(validloader) if validloader is not None else None,
        'test_batches': len(testloader) if testloader is not None else None
    })

    return trainloader, validloader, testloader


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

def _get_datasets(ds_provider:DatasetProvider, load_train:bool, load_test:bool,
        transform_train, transform_test)\
            ->Tuple[DatasetLike, DatasetLike]:
    trainset, testset = ds_provider.get_datasets(load_train, load_test,
                                                 transform_train, transform_test)

    return  trainset, testset

# target_lb allows to filter dataset for a specific class, not used
def _get_sampler(dataset:Dataset, val_ratio:Optional[float], shuffle:bool,
                 max_items:Optional[int])->Tuple[Sampler, Optional[Sampler]]:
    # we cannot not shuffle just for train or just val because of in distributed mode both must come from same shrad
    train_sampler = DistributedStratifiedSampler(dataset,
                        val_ratio=val_ratio, is_val=False, shuffle=shuffle,
                        max_items=max_items)
    valid_sampler = DistributedStratifiedSampler(dataset,
                        val_ratio=val_ratio, is_val=True, shuffle=shuffle,
                        max_items=max_items) \
                    if val_ratio is not None else None


    return train_sampler, valid_sampler


