# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Tuple, Dict, Optional, Callable
from pathlib import Path
import os

import torch
import numpy as np
from tqdm import tqdm
from overrides import overrides

from torch.utils.data import Dataset, Subset

from archai.discrete_search.datasets.dataset_provider import DatasetProvider
from archai.datasets.providers.lmdb_image_provider import TensorpackLmdbImageDataset


class MultiTensorpackLmdbImageProvider(DatasetProvider):
    def __init__(self, conf_dataset: Dict):
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
        self.conf_dataset = conf_dataset
        self._dataroot = Path(conf_dataset['dataroot']).absolute()

        assert 'datasets' in conf_dataset and isinstance(conf_dataset['datasets'], list), \
            '`datasets` must be a list of datasets'

        self.datasets = conf_dataset['datasets']
        self.val_split = conf_dataset.get('val_split', 0.02)

        for k in ['img_key', 'tr_lmdb']:
            assert all(k in d for d in self.datasets), f'`{k}` must be specified for all datasets'

    @overrides
    def get_train_val_datasets(self) -> Tuple[Dataset, Dataset]:
        tr_d = torch.utils.data.ConcatDataset([
            TensorpackLmdbImageDataset(
                str(self._dataroot / d['tr_lmdb']), **d, augmentation_fn=transform_train
            ) for d in self.datasets
        ])

        te_d = torch.utils.data.ConcatDataset([
            TensorpackLmdbImageDataset(
                str(self._dataroot / d['te_lmdb']), **d, augmentation_fn=transform_test
            ) for d in self.datasets
        ])

        return tr_d, te_d

    def get_train_val_datasets(self, transform_train: Optional[Callable] = None,
                               transform_val: Optional[Callable] = None) -> Tuple[Dataset, Dataset]:
        """Returns train and validation datasets using the `val_split` parameter."""
        # Creates two copies of the dataset using different transforms
        tr_dataset = torch.utils.data.ConcatDataset([
            TensorpackLmdbImageDataset(
                str(self._dataroot / d['tr_lmdb']), **d, augmentation_fn=transform_train
            ) for d in tqdm(self.datasets, desc='Loading LMDB datasets...')
        ])

        val_dataset = torch.utils.data.ConcatDataset([
            TensorpackLmdbImageDataset(
                str(self._dataroot / d['tr_lmdb']), **d, augmentation_fn=transform_val
            ) for d in tqdm(self.datasets, desc='Loading LMDB datasets...')
        ])

        # Performs train-validation split
        indices = np.arange(len(tr_dataset))
        np.random.shuffle(indices)
        split_point = int(len(tr_dataset) * (1 - self.val_split))

        tr_subset = Subset(
            tr_dataset, indices[:split_point]
        )

        val_subset = Subset(
            val_dataset, indices[split_point:]
        )

        assert len(tr_subset) + len(val_subset) == len(tr_dataset)
        assert len(set(tr_subset.indices).intersection(set(val_subset.indices))) == 0  

        return tr_subset, val_subset

    @overrides
    def get_transforms(self) -> tuple:
        return tuple()
