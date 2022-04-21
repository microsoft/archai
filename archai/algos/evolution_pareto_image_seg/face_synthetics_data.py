import cv2
import torch
from pathlib import Path
import albumentations as A
from .augmentations import AUGMENTATIONS


class FaceSynthetics(torch.utils.data.Dataset):
    CLASSES = ['background', 'skin', 'nose', 'right_eye', 'left_eye', 'right_brow', 'left_brow',
               'right_ear', 'left_ear', 'mouth_interior', 'top_lip', 'bottom_lip', 'neck', 'hair',
               'beard', 'clothing', 'glasses', 'headwear', 'facewear']

    def __init__(self, root, img_size, subset='train',
                 val_size=2000, ignore_index=255,
                 augmentation='none', preprocessing=False,
                 mask_size=None, debug=False):

        root = Path(root)
        assert root.is_dir()
        assert isinstance(img_size, tuple)

        assert augmentation in AUGMENTATIONS
        augmentation = AUGMENTATIONS[augmentation]

        self.img_size = img_size
        self.root = root
        self.subset = subset
        self.augmentation = augmentation
        self.preprocessing = preprocessing
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
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        sample = {
            'image': cv2.imread(self.img_files[idx]),
            'mask': cv2.imread(self.seg_files[idx], cv2.IMREAD_GRAYSCALE)
        }

        sample['image'] = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB)
        
        if self.augmentation and self.subset == 'train':
            sample = self.augmentation(**sample)
        
        if self.preprocessing:
            sample['image'] = self.preprocessing(sample['image'])
        else:
            sample['image'] = sample['image']/255

        
        mask_size = self.mask_size if self.mask_size else self.img_size 
        sample['mask'] = cv2.resize(sample['mask'], mask_size, interpolation=cv2.INTER_NEAREST)
        sample['image'] = cv2.resize(sample['image'], self.img_size, interpolation=cv2.INTER_LINEAR)

        sample['mask'] = torch.tensor(sample['mask'], dtype=torch.long)
        sample['image'] = torch.tensor(sample['image'].transpose(2, 0, 1), dtype=torch.float32)

        return sample
