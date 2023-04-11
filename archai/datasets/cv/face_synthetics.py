from pathlib import Path
from typing import Callable, Optional, Tuple

from overrides import overrides
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image

from archai.api.dataset_provider import DatasetProvider
from archai.common.utils import download_and_extract_zip


class FaceSyntheticsDataset(torch.utils.data.Dataset):
    CLASSES = ['background', 'skin', 'nose', 'right_eye', 'left_eye', 'right_brow', 'left_brow',
               'right_ear', 'left_ear', 'mouth_interior', 'top_lip', 'bottom_lip', 'neck', 'hair',
               'beard', 'clothing', 'glasses', 'headwear', 'facewear']

    def __init__(self, dataset_dir: str, img_size: Tuple[int, int] = (256, 256),
                 subset: str = 'train', val_size: int = 2000, ignore_index: int = 255,
                 mask_size: Optional[Tuple[int, int]] = None,
                 augmentation: Optional[Callable] = None):
        """Face Synthetics Dataset

        Args:
            dataset_dir (str): Dataset directory.
            img_size (Tuple[int, int]): Image size (width, height). Defaults to (256, 256).
            subset (str, optional): Subset ['train', 'test', 'validation']. Defaults to 'train'.
            val_size (int, optional): Validation set size. Defaults to 2000.

            mask_size (Optional[Tuple[int, int]], optional): Segmentation mask size (width, height). If `None`,
                `img_size` is used. Defaults to None.

            augmentation (Optional[Callable], optional): Augmentation function. Expects a callable object
                with named arguments 'image' and 'mask' that returns a dictionary with 'image' and 'mask' as
                keys. Defaults to None.
        """
        dataset_dir = Path(dataset_dir)
        assert dataset_dir.is_dir()
        assert isinstance(img_size, tuple)

        zip_url = "https://facesyntheticspubwedata.blob.core.windows.net/iccv-2021/dataset_1000.zip"
        self.img_size = img_size
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.mask_size = mask_size
        self.augmentation = augmentation

        all_seg_files = [str(f) for f in sorted(self.dataset_dir.glob('*_seg.png'))]
        if len(all_seg_files) == 0:
            download_and_extract_zip(zip_url, self.dataset_dir)
            all_seg_files = [str(f) for f in sorted(self.dataset_dir.glob('*_seg.png'))]

        train_subset, test_subset = all_seg_files[:90_000], all_seg_files[90_000:]

        if subset == 'train':
            self.seg_files = train_subset[:-val_size] if val_size > 0 else train_subset
        elif subset == 'validation':
            self.seg_files = train_subset[-val_size:] if val_size > 0 else None
        elif subset == 'test':
            self.seg_files = test_subset

        self.img_files = [s.replace("_seg.png",".png") for s in self.seg_files]
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        sample = {
            'image': read_image(self.img_files[idx]),
            'mask': read_image(self.seg_files[idx]).long()
        }

        if self.augmentation and self.subset == 'train':
            sample = self.augmentation(**sample)

        sample['image'] = sample['image']/255

        mask_size = self.mask_size if self.mask_size else self.img_size
        sample['mask'] = F.resize(
            sample['mask'], mask_size[::-1],
            interpolation=F.InterpolationMode.NEAREST
        )
        sample['image'] = F.resize(sample['image'], self.img_size[::-1])

        return sample


class FaceSyntheticsDatasetProvider(DatasetProvider):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        assert self.dataset_dir.is_dir()

    @overrides
    def get_train_dataset(self, **kwargs) -> torch.utils.data.Dataset:
        return FaceSyntheticsDataset(
            self.dataset_dir, subset='train', **kwargs
        )

    @overrides
    def get_test_dataset(self, **kwargs) -> torch.utils.data.Dataset:
        return FaceSyntheticsDataset(
            self.dataset_dir, subset='test', **kwargs
        )

    @overrides
    def get_val_dataset(self, **kwargs) -> torch.utils.data.Dataset:
        return FaceSyntheticsDataset(
            self.dataset_dir, subset='validation', **kwargs
        )
