# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Optional, Any
from pathlib import Path
import os

import lmdb
import torch
import msgpack
import torchvision
import numpy as np
import cv2
from overrides import overrides

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config
from archai.common import utils


class TensorpackLmdbImageDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_dir: str, img_key: str, label_key: str,
                 serializer: str = 'msgpack', img_format: str = 'numpy'):
        
        self.db = lmdb.open(
            lmdb_dir, subdir=False, readonly=True, lock=False,
            readahead=True, map_size=1099511627776 * 2, max_readers=100
        )
        self.img_key = img_key
        self.label_key = label_key
        self.txn = self.db.begin()
        self.keys = [k for k, _ in self.txn.cursor()]
        self.serializer = serializer
        self.img_format = img_format

    def __getitem__(self, idx)->Tuple[torch.Tensor, Any]:
        key = self.keys[idx]
        value = self.txn.get(key)

        if self.serializer == 'msgpack':
            sample = msgpack.loads(value)
        else:
            raise NotImplementedError(f'unsupported serializer {self.serializer}')

        if self.img_format == 'numpy':
            img = cv2.imdecode(
                np.frombuffer(sample[self.img_key], dtype=np.uint8),
                cv2.IMREAD_COLOR
            )[..., ::-1]
            sample[self.img_key] = (img/255.0).transpose(2, 1, 0) # HWC to CHW

        return torch.tensor(sample[self.img_key]/255.0), sample[self.label_key]

    def __len__(self)->int:
        return len(self.keys)


class TensorpackLmdbImageProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        super().__init__(conf_dataset)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self._name = conf_dataset['name']
        self.label_key = conf_dataset.get('label_key', 'sequence')
        self.img_key = conf_dataset.get('img_key', 'img_data')
        self.serializer = conf_dataset.get('serializer', 'msgpack')

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainpath = os.path.join(self._dataroot, self._name)
            trainset = TensorpackLmdbImageDataset(
                trainpath, img_key=self.img_key, label_key=self.label_key,
                serializer=self.serializer
            )

        if load_test:
            testpath = os.path.join(self._dataroot, self._name)
            testset = TensorpackLmdbImageDataset(
                testpath, img_key=self.img_key, label_key=self.label_key,
                serializer=self.serializer
            )

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        return tuple()

register_dataset_provider('tensorpack_lmdb_image', TensorpackLmdbImageProvider)
