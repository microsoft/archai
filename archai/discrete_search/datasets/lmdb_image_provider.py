# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Optional, Dict, Any, Callable
from pathlib import Path

import lmdb
import torch
import msgpack
import numpy as np
import cv2
from overrides import overrides

from torch.utils.data import Dataset

from archai.discrete_search import DatasetProvider
from archai.common.config import Config
from archai.common.common import logger
from archai.common import utils


class TensorpackLmdbImageDataset(Dataset):
    def __init__(self, lmdb_path: str, img_key: str,
                 mask_key: Optional[str] = None, 
                 serializer: str = 'msgpack', img_size: Optional[Tuple[int]] = None,
                 img_format: str = 'numpy', ones_mask: bool = False,
                 zeroes_mask: bool = False, raise_errors: bool = True,
                 is_bgr: bool = True, valid_resolutions: Optional[List[Tuple]] = None,
                 augmentation_fn: Optional[Callable] = None,
                 mask_interpolation_method: int = cv2.INTER_NEAREST,
                 **kwargs):
        """Tensorpack LMDB torch Dataset.
        Args:
            lmdb_path (str): path to LMDB
            img_key (str): key of image in LMDB
            mask_key (Optional[str], optional): key of mask in LMDB. Defaults to None.
            serializer (str, optional): serialization method. Defaults to 'msgpack'.
            img_size (Optional[Tuple[int]], optional): size of image to resize. Defaults to None.
            img_format (str, optional): image format. Defaults to 'numpy'.
            ones_mask (bool, optional): if mask is all ones. Defaults to False.
            zeroes_mask (bool, optional): if mask is all zeroes. Defaults to False.
            raise_errors (bool, optional): if errors should be raised. Defaults to True.
            is_bgr (bool, optional): if image is in BGR format. Defaults to True.
            valid_resolutions (Optional[List[Tuple]], optional): list of valid resolutions to validate inputs.
            Defaults to None.
            augmentation_fn (Optional[Callable], optional): augmentation function of format
            aug_fn(image: np.ndarray, mask: np.ndarray) and returns a dictionary with
            'image' and 'mask' keys. Defaults to None.
            mask_interpolation_method (int, optional): interpolation method for mask. Defaults to cv2.INTER_NEAREST.
        """
        self.lmdb_path = lmdb_path
        self.db = lmdb.open(
            lmdb_path, subdir=False, readonly=True, lock=False,
            readahead=True, map_size=1099511627776 * 2, max_readers=100
        )
        self.img_key = img_key
        self.mask_key = mask_key
        self.txn = self.db.begin()
        self.keys = [k for k, _ in self.txn.cursor() if k != b'__keys__']
        self.img_size = img_size
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
        self.augmentation_fn = augmentation_fn
        self.mask_interpolation_method = mask_interpolation_method

    def _get_datapoint(self, idx) -> Dict:
        key = self.keys[idx]
        value = self.txn.get(key)

        if self.serializer == 'msgpack':
            sample = msgpack.loads(value)
        else:
            raise NotImplementedError(f'unsupported serializer {self.serializer}')

        for d_key in [self.img_key, self.mask_key]:
            if d_key and d_key not in sample:
                available_keys = sample.keys() if isinstance(sample, dict) else []
                raise KeyError(
                    f'{d_key} not found in sample. '
                    f'Available keys: {available_keys}'
                )

            if d_key and isinstance(sample[d_key], dict) and b'data' in sample[d_key]:
                sample[d_key] = sample[d_key][b'data']

        return sample

    def __getitem__(self, idx: int) -> Optional[Dict]:
        try:
            sample = self._get_datapoint(idx)

            if self.img_format == 'numpy':
                img = np.frombuffer(sample[self.img_key], dtype=np.uint8).reshape((-1, 1))
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = img[..., ::-1].copy() if self.is_bgr else img

                if self.ones_mask:
                    mask = np.ones(img.shape[:2], dtype=np.uint8)
                elif self.zeroes_mask or len(sample[self.mask_key]) == 0:
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                else:
                    mask_cv2_buf = np.frombuffer(sample[self.mask_key], dtype=np.uint8).reshape((-1, 1))
                    mask = cv2.imdecode(mask_cv2_buf, cv2.IMREAD_GRAYSCALE)
                
                sample = {'image': img, 'mask': mask}

                if self.augmentation_fn:
                    sample = self.augmentation_fn(**sample)

                if self.img_size:
                    sample['image'] = cv2.resize(sample['image'], self.img_size)
                    sample['mask'] = cv2.resize(
                        sample['mask'], self.img_size,
                        interpolation=self.mask_interpolation_method
                    )

                if self.valid_resolutions:
                    assert img.shape[:2] in self.valid_resolutions
                    assert mask.shape[:2] in self.valid_resolutions

            else:
                raise NotImplementedError(f'unsupported image format {self.img_format}')


            return {
                'image': torch.tensor(
                    sample['image'].transpose(2, 0, 1) / 255.0, dtype=torch.float
                ),
                'mask': torch.tensor(sample['mask'], dtype=torch.long),
                'dataset_path': self.lmdb_path,
                'key': self.keys[idx]
            }

        except Exception as e:
            if self.raise_errors:
                raise e
            else:
                logger.info(
                    f'Error while loading sample {idx} from dataset '
                    f' {self.lmdb_path}: {e}'
                )

    def __len__(self) -> int:
        return len(self.keys)


class TensorpackLmdbImageProvider(DatasetProvider):
    def __init__(self, conf_dataset: Config):
        """Tensorpack LMDB dataset provider. 
        Expects a configuration file with the following keys:
        - tr_lmdb: relative path to the dataroot of the training LMDB 
        - te_lmdb: relative path to the dataroot of the test LMDB  (optional)
        - kwargs from the TensorpackLmdbImageDataset constructor (img_key, mask_key, ...)
        Args:
            conf_dataset (Config): configuration for the dataset
        """        
        self.conf_dataset = conf_dataset
        assert 'img_key' in conf_dataset, 'img_key must be specified'

        self._dataroot = utils.full_path(conf_dataset['dataroot'])

        self.tr_lmdb = Path(self._dataroot) / conf_dataset['tr_lmdb']
        self.te_lmdb = Path(self._dataroot) / conf_dataset.get('te_lmdb', '')

        self.val_split = conf_dataset.get('val_split', 0.02)
        assert self.tr_lmdb.exists(), f'{self.tr_lmdb} must exist.'

    @overrides
    def get_train_val_datasets(self) -> Tuple[Dataset, Dataset]:
        tr_d = TensorpackLmdbImageDataset(str(self.tr_lmdb), **self.conf_dataset)
        te_d = TensorpackLmdbImageDataset(str(self.te_lmdb), **self.conf_dataset)

        return tr_d, te_d
