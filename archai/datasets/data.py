from typing import List, Tuple, Union, Optional

import os
import sys


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
    horovod = conf_loader['horovod']
    load_train = conf_loader['load_train']
    train_batch = conf_loader['train_batch']
    train_workers = conf_loader['train_workers']
    load_test = conf_loader['load_test']
    test_batch = conf_loader['test_batch']
    test_workers = conf_loader['test_workers']
    # endregion

    ds_provider = create_dataset_provider(conf_data)

    train_dl, val_dl, test_dl, *_ = get_dataloaders(ds_provider,
        load_train=load_train, train_batch_size=train_batch,
        load_test=load_test, test_batch_size=test_batch,
        aug=aug, cutout=cutout,  val_ratio=val_ratio, val_fold=val_fold,
        train_workers=train_workers, test_workers=test_workers, horovod=horovod,
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
    horovod=False, target_lb=-1, max_batches:int=-1) \
        -> Tuple[Optional[DataLoader], Optional[DataLoader],
                 Optional[DataLoader], Optional[Sampler]]:

    # if debugging in vscode, workers > 0 gets termination
    if utils.is_debugging():
        train_workers = test_workers = 0
        logger.warn({'debugger': True})
    if train_workers is None:
        train_workers = torch.cuda.device_count() * 4
    if test_workers is None:
        test_workers = torch.cuda.device_count() * 4
    logger.info({'train_workers': train_workers, 'test_workers':test_workers})

    transform_train, transform_test = ds_provider.get_transforms()
    add_named_augs(transform_train, aug, cutout)

    trainset, testset = _get_datasets(ds_provider,
        load_train, load_test, transform_train, transform_test,
        train_max_size=max_batches*train_batch_size,
        test_max_size=max_batches*test_batch_size)

    # TODO: below will never get executed, set_preaug does not exist in PyTorch
    # if total_aug is not None and augs is not None:
    #     trainset.set_preaug(augs, total_aug)
    #     logger.info('set_preaug-')

    trainloader, validloader, testloader, train_sampler = None, None, None, None

    if trainset:
        # sample validation set from trainset if cv_ratio > 0
        train_sampler, valid_sampler = _get_train_sampler(val_ratio, val_fold,
            trainset, horovod, target_lb)
        trainloader = DataLoader(trainset,
            batch_size=train_batch_size, shuffle=True if train_sampler is None else False,
            num_workers=train_workers, pin_memory=True,
            sampler=train_sampler, drop_last=False) # TODO: original paper has this True
        if train_sampler is not None:
            validloader = DataLoader(trainset,
                batch_size=train_batch_size, shuffle=False,
                num_workers=train_workers, pin_memory=True, #TODO: set n_workers per ratio?
                sampler=valid_sampler, drop_last=False)
        # else validloader is left as None
    if testset:
        testloader = DataLoader(testset,
            batch_size=test_batch_size, shuffle=False,
            num_workers=test_workers, pin_memory=True,
            sampler=None, drop_last=False
    )

    assert val_ratio > 0.0 or validloader is None

    logger.info({
        'train_batches': len(trainloader) if trainloader is not None else None,
        'val_batches': len(validloader) if validloader is not None else None,
        'test_batches': len(testloader) if testloader is not None else None
    })

    # we have to return train_sampler because of horovod
    return trainloader, validloader, testloader, train_sampler


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
        transform_train, transform_test, train_max_size:int, test_max_size:int)\
            ->Tuple[DatasetLike, DatasetLike]:
    trainset, testset = ds_provider.get_datasets(load_train, load_test,
                                                 transform_train, transform_test)

    if train_max_size > 0:
        logger.warn({'train_max_size': train_max_size})
        trainset = LimitDataset(trainset, train_max_size)
    if test_max_size > 0:
        logger.warn({'test_max_size': test_max_size})
        testset = LimitDataset(testset, test_max_size)

    return  trainset, testset

# target_lb allows to filter dataset for a specific class, not used
def _get_train_sampler(val_ratio:float, val_fold:int, trainset, horovod,
        target_lb:int=-1)->Tuple[Optional[Sampler], Sampler]:
    """Splits train set into train, validation sets, stratified rand sampling.

    Arguments:
        val_ratio {float} -- % of data to put in valid set
        val_fold {int} -- Total of 5 folds are created, val_fold specifies which
            one to use
        target_lb {int} -- If >= 0 then trainset is filtered for only that
            target class ID
    """
    assert val_fold >= 0

    train_sampler, valid_sampler = None, None
    logger.info({'val_ratio': val_ratio})
    if val_ratio > 0.0: # if val_ratio is not specified then sampler is empty
        """stratified shuffle val_ratio will yield return total of n_splits,
        each val_ratio containing tuple of train and valid set with valid set
        size portion = val_ratio, while samples for each class having same
        proportions as original dataset"""


        # TODO: random_state should be None so np.random is used
        # TODO: keep hardcoded n_splits=5?
        sss = StratifiedShuffleSplit(n_splits=5, test_size=val_ratio,
                                     random_state=0)
        sss = sss.split(list(range(len(trainset))), trainset.targets)

        # we have 5 plits, but will select only one of them by val_fold
        for _ in range(val_fold + 1):
            train_idx, valid_idx = next(sss)

        if target_lb >= 0:
            train_idx = [i for i in train_idx if trainset.targets[i] == target_lb]
            valid_idx = [i for i in valid_idx if trainset.targets[i] == target_lb]

        # NOTE: we apply random sampler for validation set as well because
        #       this set is used for training alphas for darts
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        if horovod: # train sampler for horovod
            import horovod.torch as hvd
            train_sampler = DistributedSampler(
                    train_sampler, num_replicas=hvd.size(), rank=hvd.rank())
    else:
        # this means no sampling, validation set would be empty
        valid_sampler = SubsetSampler([])

        if horovod: # train sampler for horovod
            import horovod.torch as hvd
            train_sampler = DistributedSampler(
                    valid_sampler, num_replicas=hvd.size(), rank=hvd.rank())
        # else train_sampler is None
    return train_sampler, valid_sampler


