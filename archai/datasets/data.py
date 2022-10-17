# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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

from filelock import FileLock

from .augmentation import add_named_augs
from archai.common import common
from archai.common.common import logger
from archai.common import utils, apex_utils
from archai.datasets.dataset_provider import DatasetProvider, get_provider_type
from archai.common.config import Config
from archai.datasets.limit_dataset import LimitDataset, DatasetLike
from archai.datasets.distributed_stratified_sampler import DistributedStratifiedSampler


class DataLoaders:
    def __init__(self, train_dl:Optional[DataLoader]=None,
                 val_dl:Optional[DataLoader]=None,
                 test_dl:Optional[DataLoader]=None) -> None:
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

def get_data(conf_loader:Config)->DataLoaders:
    logger.pushd('data')

    # region conf vars
    # dataset
    conf_dataset = conf_loader['dataset']
    max_batches = conf_dataset['max_batches']

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
    conf_apex  = conf_loader['apex']
    # endregion

    ds_provider = create_dataset_provider(conf_dataset)

    apex = apex_utils.ApexUtils(conf_apex, logger)

    train_dl, val_dl, test_dl = get_dataloaders(ds_provider,
        load_train=load_train, train_batch_size=train_batch,
        load_test=load_test, test_batch_size=test_batch,
        aug=aug, cutout=cutout,  val_ratio=val_ratio, val_fold=val_fold,
        train_workers=train_workers, test_workers=test_workers,
        max_batches=max_batches, apex=apex)

    assert train_dl is not None

    logger.popd()

    return DataLoaders(train_dl=train_dl, val_dl=val_dl, test_dl=test_dl)

def create_dataset_provider(conf_dataset:Config)->DatasetProvider:
    ds_name = conf_dataset['name']
    dataroot = utils.full_path(conf_dataset['dataroot'])
    storage_name = conf_dataset['storage_name']

    logger.info({'ds_name': ds_name, 'dataroot':dataroot, 'storage_name':storage_name})

    ds_provider_type = get_provider_type(ds_name)
    return ds_provider_type(conf_dataset)

def get_dataloaders(ds_provider:DatasetProvider,
    load_train:bool, train_batch_size:int,
    load_test:bool, test_batch_size:int,
    aug, cutout:int, val_ratio:float, apex:apex_utils.ApexUtils,
    val_fold=0, train_workers:Optional[int]=None, test_workers:Optional[int]=None,
    target_lb=-1, max_batches:int=-1) \
        -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:

    # if debugging in vscode, workers > 0 gets termination
    default_workers = 4
    if utils.is_debugging():
        train_workers = test_workers = 0
        logger.warn({'debugger': True})
    if train_workers is None:
        train_workers = default_workers # following NVidia DeepLearningExamples
    if test_workers is None:
        test_workers = default_workers

    train_workers = round((1-val_ratio)*train_workers)
    val_workers = round(val_ratio*train_workers)
    logger.info({'train_workers': train_workers, 'val_workers': val_workers,
                 'test_workers':test_workers})

    transform_train, transform_test = ds_provider.get_transforms()
    add_named_augs(transform_train, aug, cutout)

    trainset, testset = _get_datasets(ds_provider,
        load_train, load_test, transform_train, transform_test)

    # TODO: below will never get executed, set_preaug does not exist in PyTorch
    # if total_aug is not None and augs is not None:
    #     trainset.set_preaug(augs, total_aug)
    #     logger.info('set_preaug-')

    trainloader, validloader, testloader, train_sampler = None, None, None, None

    if trainset:
        max_train_fold = min(len(trainset), max_batches*train_batch_size) if max_batches else None
        logger.info({'val_ratio': val_ratio, 'max_train_batches': max_batches,
                    'max_train_fold': max_train_fold})

        # sample validation set from trainset if cv_ratio > 0
        train_sampler, valid_sampler = _get_sampler(trainset, val_ratio=val_ratio,
                                                    shuffle=True, apex=apex,
                                                    max_items=max_train_fold)
        logger.info({'train_sampler_world_size':train_sampler.world_size,
                    'train_sampler_rank':train_sampler.rank,
                    'train_sampler_len': len(train_sampler)})
        if valid_sampler:
            logger.info({'valid_sampler_world_size':valid_sampler.world_size,
                        'valid_sampler_rank':valid_sampler.rank,
                        'valid_sampler_len': len(valid_sampler)
                        })

        # shuffle is performed by sampler at each epoch
        trainloader = DataLoader(trainset,
            batch_size=train_batch_size, shuffle=False,
            num_workers=train_workers,
            pin_memory=True,
            sampler=train_sampler, drop_last=False) # TODO: original paper has this True

        if val_ratio > 0.0:
            validloader = DataLoader(trainset,
                batch_size=train_batch_size, shuffle=False,
                num_workers=val_workers,
                pin_memory=True,
                sampler=valid_sampler, drop_last=False)
        # else validloader is left as None
    if testset:
        max_test_fold = min(len(testset), max_batches*test_batch_size) if max_batches else None
        logger.info({'max_test_batches': max_batches,
                    'max_test_fold': max_test_fold})

        test_sampler, test_val_sampler = _get_sampler(testset, val_ratio=None,
                                       shuffle=False, apex=apex,
                                       max_items=max_test_fold)
        logger.info({'test_sampler_world_size':test_sampler.world_size,
                    'test_sampler_rank':test_sampler.rank,
                    'test_sampler_len': len(test_sampler)})
        assert test_val_sampler is None

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
    """Samples elements from a given list of indices, without replacement.

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
                 max_items:Optional[int], apex:apex_utils.ApexUtils)\
        ->Tuple[DistributedStratifiedSampler, Optional[DistributedStratifiedSampler]]:

    world_size, global_rank = apex.world_size, apex.global_rank

    # we cannot not shuffle just for train or just val because of in distributed mode both must come from same shrad
    train_sampler = DistributedStratifiedSampler(dataset,
                        val_ratio=val_ratio, is_val=False, shuffle=shuffle,
                        max_items=max_items, world_size=world_size, rank=global_rank)

    valid_sampler = DistributedStratifiedSampler(dataset,
                        val_ratio=val_ratio, is_val=True, shuffle=shuffle,
                        max_items=max_items, world_size=world_size, rank=global_rank) \
                    if val_ratio is not None else None

    return train_sampler, valid_sampler


