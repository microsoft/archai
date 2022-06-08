# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from re import X
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import os

import lmdb
import torch
import msgpack
import numpy as np
import cv2
from overrides import overrides

from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common.config import Config
from archai.common.common import logger
from archai.common import utils


class TensorpackLmdbImageDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path: str, img_key: str,
                 mask_key: Optional[str] = None, serializer: str = 'msgpack',
                 img_format: str = 'numpy', ones_mask: bool = False,
                 zeroes_mask: bool = False, raise_errors: bool = True,
                 is_bgr: bool = True, valid_resolutions: Optional[List[Tuple]] = None,
                 **kwargs):
        """Tensorpack LMDB torch Dataset.

        Args:
            lmdb_path (str): path to LMDB
            img_key (str): key of image in LMDB
            mask_key (Optional[str], optional): key of mask in LMDB. Defaults to None.
            serializer (str, optional): serialization method. Defaults to 'msgpack'.
            img_format (str, optional): image format. Defaults to 'numpy'.
            ones_mask (bool, optional): if mask is all ones. Defaults to False.
            zeroes_mask (bool, optional): if mask is all zeroes. Defaults to False.
            raise_errors (bool, optional): if errors should be raised. Defaults to True.
            is_bgr (bool, optional): if image is in BGR format. Defaults to True.
            valid_resolutions (Optional[List[Tuple]], optional): list of valid resolutions to validate inputs.
            Defaults to None.
        """
        self.lmdb_path = lmdb_path
        self.db = lmdb.open(
            lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=True, map_size=1099511627776 * 2, max_readers=100
        )
        self.img_key = img_key
        self.mask_key = mask_key
        self.txn = self.db.begin()
        self.keys = [k for k, _ in self.txn.cursor()]
        self.serializer = serializer
        self.img_format = img_format

        self.ones_mask = ones_mask
        self.zeroes_mask = zeroes_mask
        
        assert not (self.ones_mask and self.zeroes_mask), \
            'ones_mask and zeroes_mask are mutually exclusive'
        
        if self.mask_key is None:
            assert self.ones_mask or self.zeroes_mask, \
                'ones_mask or zeroes_mask must be True if mask_key is None'

        self.is_bgr = is_bgr
        self.raise_errors = raise_errors
        self.valid_resolutions = valid_resolutions

    def _get_datapoint(self, idx)->Dict:
        key = self.keys[idx]
        value = self.txn.get(key)

        if self.serializer == 'msgpack':
            sample = msgpack.loads(value)
        else:
            raise NotImplementedError(f'unsupported serializer {self.serializer}')

        for d_key in [self.img_key, self.mask_key]:
            if d_key and d_key not in sample:
                raise KeyError(
                    f'{d_key} not found in sample. '
                    f'Available keys: {sample.keys()}'
                )

            if d_key and isinstance(sample[d_key], dict) and b'data' in sample[d_key]:
                sample[d_key] = sample[d_key][b'data']

        return sample

    def __getitem__(self, idx)->Tuple[torch.Tensor, Any]:
        sample = self._get_datapoint(idx)

        try:
            if self.img_format == 'numpy':
                img = np.frombuffer(sample[self.img_key], dtype=np.uint8).reshape((-1, 1))
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = img[..., ::-1].copy() if self.is_bgr else img

                if self.ones_mask:
                    mask = np.ones(img.shape[:2], dtype=np.uint8)
                elif self.zeroes_mask:
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                else:
                    mask_cv2_buf = np.frombuffer(sample[self.mask_key], dtype=np.uint8).reshape((-1, 1))
                    mask = cv2.imdecode(mask_cv2_buf, cv2.IMREAD_GRAYSCALE)
                    
                if self.valid_resolutions:
                    assert img.shape[:2] in self.valid_resolutions
                    assert mask.shape[:2] in self.valid_resolutions

            return torch.tensor(img.transpose(2, 1, 0) / 255.0), torch.tensor(mask, dtype=torch.long)

        except Exception as e:
            if self.raise_errors:
                raise e
            else:
                logger.info(
                    f'Error while loading sample {idx} from dataset '
                    f' {self.lmdb_path}: {e}'
                )

                return None, None

    def __len__(self)->int:
        return len(self.keys)


class TensorpackLmdbImageProvider(DatasetProvider):
    def __init__(self, conf_dataset:Config):
        """Tensorpack LMDB dataset provider. 
        Expects a configuration file with the following keys:
        - tr_lmdb: relative path to the dataroot of the training LMDB 
        - te_lmdb: relative path to the dataroot of the test LMDB  (optional)
        - kwargs from the TensorpackLmdbImageDataset constructor

        Expects training and test datasets to be in distinct directories following
        the pattern: `{dataset-name}_{subset}` where `subset` is either `train` or `test`.

        Args:
            conf_dataset (Config): configuration for the dataset
        """        
        super().__init__(conf_dataset)
        self.conf_dataset = conf_dataset
        self._dataroot = utils.full_path(conf_dataset['dataroot'])

        self.tr_lmdb = Path(self._dataroot) / conf_dataset['tr_lmdb']
        self.te_lmdb = Path(self._dataroot) / conf_dataset.get('te_lmdb', '')

        assert self.tr_lmdb.exists(), f'{self.tr_lmdb} must exist.'

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test)->TrainTestDatasets:
        trainset, testset = None, None

        if load_train:
            trainset = TensorpackLmdbImageDataset(
                str(self.tr_lmdb), **self.conf_dataset
            )

        if load_test:
            testset = TensorpackLmdbImageDataset(
                str(self.te_lmdb), **self.conf_dataset
            )

        return trainset, testset

    @overrides
    def get_transforms(self)->tuple:
        return tuple()

register_dataset_provider('tensorpack_lmdb_image', TensorpackLmdbImageProvider)
