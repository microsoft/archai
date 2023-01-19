# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import lmdb
import msgpack
import numpy as np
import torch
from torch.utils.data import Dataset

from archai.common.logger import Logger

logger = Logger(source=__name__)


class TensorpackLmdbDataset(Dataset):
    """Tensorpack LMDB dataset."""

    def __init__(
        self,
        lmdb_file_path: str,
        img_key: str,
        mask_key: Optional[str] = None,
        serializer: Optional[str] = "msgpack",
        img_size: Optional[Tuple[int, ...]] = None,
        img_format: Optional[str] = "numpy",
        ones_mask: Optional[bool] = False,
        zeroes_mask: Optional[bool] = False,
        raise_errors: Optional[bool] = True,
        is_bgr: Optional[bool] = True,
        valid_resolutions: Optional[List[Tuple]] = None,
        augmentation_fn: Optional[Callable] = None,
        mask_interpolation_method: int = cv2.INTER_NEAREST,
    ) -> None:
        """Initialize Tensorpack LMDB dataset.

        Args:
            lmdb_file_path: Path to the LMDB file.
            img_key: Image key in LMDB file.
            mask_key: Mask key in LMDB file.
            serializer: Serializer used to serialize data in LMDB file.
            img_size: Image size.
            img_format: Image format.
            ones_mask: Whether mask is composed of ones.
            zeroes_mask: Whether mask is composed of zeroes.
            raise_errors: Whether to raise errors.
            is_bgr: Whether image is in BGR format.
            valid_resolutions: Valid resolutions.
            augmentation_fn: Augmentation function.
            mask_interpolation_method: Mask interpolation method.

        """

        self.lmdb_file_path = lmdb_file_path
        self.db = lmdb.open(
            lmdb_file_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            map_size=1099511627776 * 2,
            max_readers=100,
        )
        self.img_key = img_key
        self.mask_key = mask_key
        self.txn = self.db.begin()
        self.keys = [k for k, _ in self.txn.cursor() if k != b"__keys__"]
        self.img_size = img_size
        self.serializer = serializer
        self.img_format = img_format

        self.ones_mask = ones_mask
        self.zeroes_mask = zeroes_mask

        assert not (self.ones_mask and self.zeroes_mask), "`ones_mask` and `zeroes_mask` are mutually exclusive."

        if self.mask_key is None:
            assert (
                self.ones_mask or self.zeroes_mask
            ), "`ones_mask` or `zeroes_mask` must be True if `mask_key` is None."

        self.is_bgr = is_bgr
        self.raise_errors = raise_errors
        self.valid_resolutions = valid_resolutions
        self.augmentation_fn = augmentation_fn
        self.mask_interpolation_method = mask_interpolation_method

    def __len__(self) -> int:
        return len(self.keys)

    def _get_datapoint(self, idx: int) -> Dict[str, Any]:
        key = self.keys[idx]
        value = self.txn.get(key)

        if self.serializer == "msgpack":
            sample = msgpack.loads(value)
        else:
            raise NotImplementedError(f"Unsupported serializer {self.serializer}")

        for d_key in [self.img_key, self.mask_key]:
            if d_key and d_key not in sample:
                available_keys = sample.keys() if isinstance(sample, dict) else []
                raise KeyError(f"{d_key} not found in sample. Available keys: {available_keys}")

            if d_key and isinstance(sample[d_key], dict) and b"data" in sample[d_key]:
                sample[d_key] = sample[d_key][b"data"]

        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            sample = self._get_datapoint(idx)

            if self.img_format == "numpy":
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

                sample = {"image": img, "mask": mask}

                if self.augmentation_fn:
                    sample = self.augmentation_fn(**sample)

                if self.img_size:
                    sample["image"] = cv2.resize(sample["image"], self.img_size)
                    sample["mask"] = cv2.resize(
                        sample["mask"], self.img_size, interpolation=self.mask_interpolation_method
                    )

                if self.valid_resolutions:
                    assert img.shape[:2] in self.valid_resolutions
                    assert mask.shape[:2] in self.valid_resolutions

            else:
                raise NotImplementedError(f"Unsupported image format: {self.img_format}")

            return {
                "image": torch.tensor(sample["image"].transpose(2, 0, 1) / 255.0, dtype=torch.float),
                "mask": torch.tensor(sample["mask"], dtype=torch.long),
                "dataset_path": self.lmdb_file_path,
                "key": self.keys[idx],
            }

        except Exception as e:
            if self.raise_errors:
                raise e
            else:
                logger.error(f"Sample {idx} from dataset {self.lmdb_file_path} could not be loaded.")
