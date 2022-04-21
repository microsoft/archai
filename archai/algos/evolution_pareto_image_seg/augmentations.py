import random
import cv2
import numpy as np
import albumentations as A
from typing import Tuple


def face_hflip_image(image, **kwargs):
    return cv2.flip(image, 1)

def face_hflip_mask(mask, swap_indices=((3, 4), (5, 6), (7, 8)), **kwargs):
    mask = cv2.flip(mask, 1)

    for index_left, index_right in swap_indices:
        right_mask = (mask == index_right)
        
        mask[mask == index_left] = index_right
        mask[right_mask] = index_left
    
    return mask

flip_aug = A.Compose([
    A.Lambda(name='FaceHflip', image=face_hflip_image, mask=face_hflip_mask, p=0.5),
], p=1)


light_aug = A.Compose([
    A.Lambda(name='FaceHflip', image=face_hflip_image, mask=face_hflip_mask, p=0.5),
    A.ShiftScaleRotate(always_apply=False, p=0.5, shift_limit_x=(-0.1, 0.1), shift_limit_y=(-0.1, 0.1), scale_limit=(0.0, 0.10000000000000009), rotate_limit=(-0.2, 0.2), interpolation=1, border_mode=0, value=None, mask_value=None),
    A.Perspective(always_apply=False, p=1, scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1),
    A.RandomBrightnessContrast(p=0.2)
], p=1)


AUGMENTATIONS = {
    'flip': flip_aug,
    'light': light_aug,
    'none': None
}