from typing import Tuple, Optional, Dict
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from overrides import overrides
import albumentations as A

from archai.common.config import Config
from archai.datasets.dataset_provider import DatasetProvider, register_dataset_provider, TrainTestDatasets
from archai.common import utils


def face_hflip_image(image, **kwargs):
    return cv2.flip(image, 1)

def face_hflip_mask(mask, swap_indices=((3, 4), (5, 6), (7, 8)), **kwargs):
    mask = cv2.flip(mask, 1)

    for index_left, index_right in swap_indices:
        right_mask = (mask == index_right)
        
        mask[mask == index_left] = index_right
        mask[right_mask] = index_left
    
    return mask


class FaceSyntheticsDataset(torch.utils.data.Dataset):
    CLASSES = ['background', 'skin', 'nose', 'right_eye', 'left_eye', 'right_brow', 'left_brow',
               'right_ear', 'left_ear', 'mouth_interior', 'top_lip', 'bottom_lip', 'neck', 'hair',
               'beard', 'clothing', 'glasses', 'headwear', 'facewear']

    AUGMENTATIONS = {
        'none': None,
        'flip': A.Compose([
            A.Lambda(name='FaceHflip', image=face_hflip_image, mask=face_hflip_mask, p=0.5),
        ], p=1)
    }

    def __init__(self, root: str, img_size: Tuple[int, int],
                 subset: str = 'train', val_size: int = 2000,
                 ignore_index: int = 255, augmentation: str = 'none',
                 mask_size: Optional[Tuple[int, int]] = None,
                 debug: bool = False, **kwargs):

        root = Path(root)
        assert root.is_dir(), f'{root} does not exist'
        assert isinstance(img_size, (list, tuple))

        assert augmentation in self.AUGMENTATIONS
        augmentation = self.AUGMENTATIONS[augmentation]

        self.img_size = img_size
        self.root = root
        self.subset = subset
        self.augmentation = augmentation
        self.mask_size = mask_size

        all_seg_files = [str(f) for f in sorted(self.root.glob('*_seg.png'))]
        train_subset, test_subset = all_seg_files[:90_000], all_seg_files[90_000:]

        if not debug:
            assert subset in ['train', 'test', 'validation']
            assert len(all_seg_files) == 100_000
            assert val_size < 90_00

        if subset == 'train':
            self.seg_files = train_subset[:-val_size] if val_size > 0 else train_subset
        elif subset == 'validation':
            self.seg_files = train_subset[-val_size:] if val_size > 0 else None
        elif subset == 'test':
            self.seg_files = test_subset

        self.img_files = [s.replace("_seg.png",".png") for s in self.seg_files]
        self.ignore_index = ignore_index
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = {
            'image': cv2.imread(self.img_files[idx]),
            'mask': cv2.imread(self.seg_files[idx], cv2.IMREAD_GRAYSCALE)
        }

        sample['image'] = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB) / 255.
        
        if self.augmentation and self.subset == 'train':
            sample = self.augmentation(**sample)
        
        mask_size = self.mask_size or self.img_size 
        sample['mask'] = cv2.resize(sample['mask'], mask_size, interpolation=cv2.INTER_NEAREST)
        sample['image'] = cv2.resize(sample['image'], self.img_size, interpolation=cv2.INTER_LINEAR)

        sample['mask'] = torch.tensor(sample['mask'], dtype=torch.long)
        sample['image'] = torch.tensor(sample['image'].transpose(2, 0, 1), dtype=torch.float32)

        return sample


class FaceSyntheticsProvider(DatasetProvider):    
    def __init__(self, conf_dataset: Config):
        super().__init__(conf_dataset)

        self.conf_dataset = conf_dataset
        self.conf_dataset['val_size'] = conf_dataset.get('val_size', 2_000)
        self._dataroot = utils.full_path(conf_dataset['dataroot'])
        self.datadir = Path(self._dataroot) / 'face_synthetics'

        assert self.datadir.is_dir(), f'{self.datadir} does not exist or is not a directory.'

    @overrides
    def get_datasets(self, load_train:bool, load_test:bool,
                     transform_train, transform_test) -> TrainTestDatasets:
        train_dataset, test_dataset = None, None

        if load_train:
            train_dataset = FaceSyntheticsDataset(
                self.datadir, subset='train', **self.conf_dataset
            )
    
        if load_test:
            test_dataset = FaceSyntheticsDataset(
                self.datadir, subset='test', **self.conf_dataset
            )

        return train_dataset, test_dataset

    @overrides
    def get_transforms(self) -> tuple:
        return None, None

    def get_train_val_datasets(self) -> Tuple[Dataset, Dataset]:
        tr_dataset = FaceSyntheticsDataset(
            self.datadir, subset='train', **self.conf_dataset
        )

        val_dataset = FaceSyntheticsDataset(
            self.datadir, subset='validation', **self.conf_dataset
        )

        return tr_dataset, val_dataset

register_dataset_provider('face_synthetics', FaceSyntheticsProvider)
