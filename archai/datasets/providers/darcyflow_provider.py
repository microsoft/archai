# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional
import os
import gzip
import pickle
import numpy as np
from scipy.io import loadmat
import h5py 

from overrides import overrides, EnforceOverrides
import torch
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset

import torchvision
from torchvision.transforms import transforms

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config
from archai.common import utils


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float



# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


def create_grid(sub):
    '''construct a grid for pde data'''
    s = int(((421 - 1) / sub) + 1)
    grids = []
    grids.append(np.linspace(0, 1, s))
    grids.append(np.linspace(0, 1, s))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1, s, s, 2)
    grid = torch.tensor(grid, dtype=torch.float)

    return grid, s



def load_darcyflow(path_to_data:str, sub:int):
    TRAIN_PATH = os.path.join(path_to_data, 'piececonst_r421_N1024_smooth1.mat')
    reader = MatReader(TRAIN_PATH)
    grid, s = create_grid(sub)
    r = sub
    ntrain = 1000

    x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]
    y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), grid.repeat(ntrain, 1, 1, 1)], dim=3)
    train_data = torch.utils.data.TensorDataset(x_train, y_train)

    ntest = 100 # according to the procedure of https://arxiv.org/abs/2010.08895
    TEST_PATH = os.path.join(path_to_data, 'piececonst_r421_N1024_smooth2.mat')
    reader = MatReader(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
    y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

    x_test = x_normalizer.encode(x_test)
    x_test = torch.cat([x_test.reshape(ntest, s, s, 1), grid.repeat(ntest, 1, 1, 1)], dim=3)
    # also note that y_test is not 
    # encoded according to the procedure of 
    # https://arxiv.org/abs/2010.08895
    test_data = torch.utils.data.TensorDataset(x_test, y_test)

    return train_data, test_data


class DarcyflowProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self._sub = conf_dataset['sub']

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        path_to_data = os.path.join(self._dataroot, 'darcyflow')

        # load the dataset but without any validation split
        trainset, testset = load_darcyflow(path_to_data, self._sub)

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        # return empty transforms since we have preprocessed data
        train_transf = []
        test_transf = []

        train_transform = transforms.Compose(train_transf)
        test_transform = transforms.Compose(test_transf)
        return train_transform, test_transform

register_dataset_provider('darcyflow', DarcyflowProvider)