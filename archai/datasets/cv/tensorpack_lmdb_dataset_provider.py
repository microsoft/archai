# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, List, Optional, Tuple

import cv2
from overrides import overrides
from torch.utils.data import ConcatDataset

from archai.api.dataset_provider import DatasetProvider
from archai.datasets.cv.tensorpack_lmdb_dataset_provider_utils import (
    TensorpackLmdbDataset,
)


class TensorpackLmdbDatasetProvider(DatasetProvider):
    """Tensorpack LMDB dataset provider."""

    def __init__(
        self,
        img_key: str,
        train_lmdb_file_path: str,
        val_lmdb_file_path: Optional[str] = None,
        test_lmdb_file_path: Optional[str] = None,
    ) -> None:
        """Initialize Tensorpack LMDB dataset provider.

        Args:
            img_key: Image key in the LMDB file.
            train_lmdb_file_path: Path to the training LMDB file.
            val_lmdb_file_path: Path to the validation LMDB file.
            test_lmdb_file_path: Path to the testing LMDB file.

        """

        self.img_key = img_key

        assert train_lmdb_file_path.exists(), f"File {train_lmdb_file_path} must exists."
        self.train_lmdb_file_path = train_lmdb_file_path

        if val_lmdb_file_path is not None:
            assert val_lmdb_file_path.exists(), f"File {val_lmdb_file_path} must exists."
        self.val_lmdb_file_path = val_lmdb_file_path

        if test_lmdb_file_path is not None:
            assert test_lmdb_file_path.exists(), f"File {test_lmdb_file_path} must exists."
        self.test_lmdb_file_path = test_lmdb_file_path

    @overrides
    def get_train_dataset(
        self,
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
    ) -> TensorpackLmdbDataset:
        return TensorpackLmdbDataset(
            self.train_lmdb_file_path,
            self.img_key,
            mask_key=mask_key,
            serializer=serializer,
            img_size=img_size,
            img_format=img_format,
            ones_mask=ones_mask,
            zeroes_mask=zeroes_mask,
            raise_errors=raise_errors,
            is_bgr=is_bgr,
            valid_resolutions=valid_resolutions,
            augmentation_fn=augmentation_fn,
            mask_interpolation_method=mask_interpolation_method,
        )

    @overrides
    def get_val_dataset(
        self,
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
    ) -> TensorpackLmdbDataset:
        try:
            return TensorpackLmdbDataset(
                self.val_lmdb_file_path,
                self.img_key,
                mask_key=mask_key,
                serializer=serializer,
                img_size=img_size,
                img_format=img_format,
                ones_mask=ones_mask,
                zeroes_mask=zeroes_mask,
                raise_errors=raise_errors,
                is_bgr=is_bgr,
                valid_resolutions=valid_resolutions,
                augmentation_fn=augmentation_fn,
                mask_interpolation_method=mask_interpolation_method,
            )
        except:
            print("Warning: validation set not available. Returning training set ...")
            return self.get_train_dataset(
                mask_key=mask_key,
                serializer=serializer,
                img_size=img_size,
                img_format=img_format,
                ones_mask=ones_mask,
                zeroes_mask=zeroes_mask,
                raise_errors=raise_errors,
                is_bgr=is_bgr,
                valid_resolutions=valid_resolutions,
                augmentation_fn=augmentation_fn,
                mask_interpolation_method=mask_interpolation_method,
            )

    @overrides
    def get_test_dataset(
        self,
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
    ) -> TensorpackLmdbDataset:
        try:
            return TensorpackLmdbDataset(
                self.test_lmdb_file_path,
                self.img_key,
                mask_key=mask_key,
                serializer=serializer,
                img_size=img_size,
                img_format=img_format,
                ones_mask=ones_mask,
                zeroes_mask=zeroes_mask,
                raise_errors=raise_errors,
                is_bgr=is_bgr,
                valid_resolutions=valid_resolutions,
                augmentation_fn=augmentation_fn,
                mask_interpolation_method=mask_interpolation_method,
            )
        except:
            print("Warning: testing set not available. Returning validation set ...")
            return self.get_val_dataset(
                mask_key=mask_key,
                serializer=serializer,
                img_size=img_size,
                img_format=img_format,
                ones_mask=ones_mask,
                zeroes_mask=zeroes_mask,
                raise_errors=raise_errors,
                is_bgr=is_bgr,
                valid_resolutions=valid_resolutions,
                augmentation_fn=augmentation_fn,
                mask_interpolation_method=mask_interpolation_method,
            )


class MultiFileTensorpackLmdbDatasetProvider(DatasetProvider):
    """Multi-file Tensorpack LMDB dataset provider."""

    def __init__(
        self,
        img_key: List[str],
        train_lmdb_file_path: List[str],
        val_lmdb_file_path: Optional[List[str]] = None,
        test_lmdb_file_path: Optional[List[str]] = None,
    ) -> None:
        """Initialize multi-file Tensorpack LMDB dataset provider.

        Args:
            img_key: Image keys in the LMDB files.
            train_lmdb_file_path: Path to the training LMDB files.
            val_lmdb_file_path: Path to the validation LMDB files.
            test_lmdb_file_path: Path to the testing LMDB files.

        """

        assert len(img_key) == len(
            train_lmdb_file_path
        ), "Number of image keys must be equal to the number of training LMDB files."
        self.img_key = img_key

        for file_path in train_lmdb_file_path:
            assert file_path.exists(), f"File {file_path} must exists."
        self.train_lmdb_file_path = train_lmdb_file_path

        if val_lmdb_file_path is not None:
            for file_path in val_lmdb_file_path:
                assert file_path.exists(), f"File {file_path} must exists."
            assert len(img_key) == len(
                val_lmdb_file_path
            ), "Number of image keys must be equal to the number of validation LMDB files."
        self.val_lmdb_file_path = val_lmdb_file_path

        if test_lmdb_file_path is not None:
            for file_path in test_lmdb_file_path:
                assert file_path.exists(), f"File {file_path} must exists."
            assert len(img_key) == len(
                val_lmdb_file_path
            ), "Number of image keys must be equal to the number of testing LMDB files."
        self.test_lmdb_file_path = test_lmdb_file_path

    @overrides
    def get_train_dataset(
        self,
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
    ) -> ConcatDataset:
        return ConcatDataset(
            [
                TensorpackLmdbDataset(
                    file_path,
                    img_key,
                    mask_key=mask_key,
                    serializer=serializer,
                    img_size=img_size,
                    img_format=img_format,
                    ones_mask=ones_mask,
                    zeroes_mask=zeroes_mask,
                    raise_errors=raise_errors,
                    is_bgr=is_bgr,
                    valid_resolutions=valid_resolutions,
                    augmentation_fn=augmentation_fn,
                    mask_interpolation_method=mask_interpolation_method,
                )
                for file_path, img_key in zip(self.train_lmdb_file_path, self.img_key)
            ]
        )

    @overrides
    def get_val_dataset(
        self,
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
    ) -> ConcatDataset:
        try:
            return ConcatDataset(
                [
                    TensorpackLmdbDataset(
                        file_path,
                        img_key,
                        mask_key=mask_key,
                        serializer=serializer,
                        img_size=img_size,
                        img_format=img_format,
                        ones_mask=ones_mask,
                        zeroes_mask=zeroes_mask,
                        raise_errors=raise_errors,
                        is_bgr=is_bgr,
                        valid_resolutions=valid_resolutions,
                        augmentation_fn=augmentation_fn,
                        mask_interpolation_method=mask_interpolation_method,
                    )
                    for file_path, img_key in zip(self.val_lmdb_file_path, self.img_key)
                ]
            )
        except:
            print("Warning: validation set not available. Returning training set ...")
            return self.get_train_dataset(
                mask_key=mask_key,
                serializer=serializer,
                img_size=img_size,
                img_format=img_format,
                ones_mask=ones_mask,
                zeroes_mask=zeroes_mask,
                raise_errors=raise_errors,
                is_bgr=is_bgr,
                valid_resolutions=valid_resolutions,
                augmentation_fn=augmentation_fn,
                mask_interpolation_method=mask_interpolation_method,
            )

    @overrides
    def get_test_dataset(
        self,
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
    ) -> ConcatDataset:
        try:
            return ConcatDataset(
                [
                    TensorpackLmdbDataset(
                        file_path,
                        img_key,
                        mask_key=mask_key,
                        serializer=serializer,
                        img_size=img_size,
                        img_format=img_format,
                        ones_mask=ones_mask,
                        zeroes_mask=zeroes_mask,
                        raise_errors=raise_errors,
                        is_bgr=is_bgr,
                        valid_resolutions=valid_resolutions,
                        augmentation_fn=augmentation_fn,
                        mask_interpolation_method=mask_interpolation_method,
                    )
                    for file_path, img_key in zip(self.test_lmdb_file_path, self.img_key)
                ]
            )
        except:
            print("Warning: testing set not available. Returning validation set ...")
            return self.get_val_dataset(
                mask_key=mask_key,
                serializer=serializer,
                img_size=img_size,
                img_format=img_format,
                ones_mask=ones_mask,
                zeroes_mask=zeroes_mask,
                raise_errors=raise_errors,
                is_bgr=is_bgr,
                valid_resolutions=valid_resolutions,
                augmentation_fn=augmentation_fn,
                mask_interpolation_method=mask_interpolation_method,
            )
