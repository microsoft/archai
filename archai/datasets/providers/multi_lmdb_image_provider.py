# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import os

import torch
import msgpack
import numpy as np
import cv2
from overrides import overrides

from torch.utils.data import Dataset, Subset

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.datasets.providers.lmdb_image_provider import TensorpackLmdbImageDataset
from archai.common.config import Config
from archai.common.common import logger
from archai.common import utils


class MultiTensorpackLmdbImageProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        """Multiple Tensorpack LMDB datasets provider. 
        Expects a configuration file with a list of multiple LMDBs:

        ```
        dataroot: [dataroot]
        name: [name of the dataset collection]
        val_split: [validation split (optional, defaults to 2%)]
        random_seed: [random seed, optional]

        datasets:
            - name: [name of the first dataset]
              tr_lmdb: [path to train LMDB of the first dataset]
              te_lmdb: [path to test LMDB of the first dataset (optional)]
              img_key: [key of image in LMDB]
              mask_key: 
            
            - name: [name of the second dataset]
              ...
        ```

        kwargs from the TensorpackLmdbImageDataset constructor (img_key, mask_key, is_bgr, ...) might be specified
        for each dataset differently.

        Args:
            conf_dataset (Config): configuration for the dataset
        """        
        super().__init__(conf_dataset)
        self.conf_dataset = conf_dataset
        self._dataroot = Path(conf_dataset['dataroot']).absolute()

        assert 'datasets' in conf_dataset and isinstance(conf_dataset['datasets'], list), \
            '`datasets` must be a list of datasets'

        self.datasets = conf_dataset['datasets']
        self.val_split = conf_dataset.get('val_split', 0.02)

        for k in ['img_key', 'tr_lmdb']:
            assert all(k in d for d in self.datasets), f'`{k}` must be specified for all datasets'

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None
        
        if load_train:
            trainset = torch.utils.data.ConcatDataset([
                TensorpackLmdbImageDataset(str(self._dataroot / d['tr_lmdb']), **d)
                for d in self.datasets
            ])

        if load_test:
            testset = torch.utils.data.ConcatDataset([
                TensorpackLmdbImageDataset(str(self._dataroot / d['te_lmdb']), **d)
                for d in self.datasets
            ])

        return trainset, testset

    def get_train_val_datasets(self) -> Tuple[Dataset, Dataset]:
        """Returns train and validation datasets."""

        if 'random_seed' in self.conf_dataset:
            np.random.seed(self.conf_dataset['random_seed'])

        base_tr_dataset = torch.utils.data.ConcatDataset([
            TensorpackLmdbImageDataset(str(self._dataroot / d['tr_lmdb']), **d)
            for d in self.datasets
        ])

        idxs = np.arange(len(base_tr_dataset))
        split_point = int(len(base_tr_dataset) * (1 - self.val_split))
        np.random.shuffle(idxs)

        tr_subset = Subset(base_tr_dataset, idxs[:split_point])
        val_subset = Subset(base_tr_dataset, idxs[split_point:])

        return tr_subset, val_subset

    @overrides
    def get_transforms(self)->tuple:
        return tuple()

register_dataset_provider('multi_tensorpack_lmdb_image', MultiTensorpackLmdbImageProvider)
