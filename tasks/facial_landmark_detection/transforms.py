import math
from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional as F

from torchvision.transforms import ToTensor, Compose
import numpy as np
import cv2

class Sample():
    """A sample of an image and its landmarks."""
    def __init__(self, image=None, landmarks=None):

        self.image = np.array(image)
        self.landmarks = landmarks
    
class ExtractRegionOfInterest():
    """Extracts a region of interest from an image and its landmarks."""

    class Rectangle:
        """A rectangle defined by its top-left and bottom-right corners."""
        def __init__(self, top_left, bottom_right):
            assert isinstance(top_left, np.ndarray)
            assert isinstance(bottom_right, np.ndarray)
            self.top_left = top_left
            self.bottom_right = bottom_right

        @property
        def corners(self):
            top_right = np.array([self.bottom_right[0], self.top_left[1]])
            bottom_left = np.array([self.top_left[0], self.bottom_right[1]])
            return np.array([self.top_left, top_right, bottom_left, self.bottom_right])

    def __init__(self, roi_size, scale = 2):

        self.roi_size = roi_size
        self.rect_dst = self.Rectangle(np.array([0, 0]), np.array([self.roi_size, self.roi_size]))
        self.scale = scale

    @property
    def src_to_dst_mapping(self):
        """Returns the homography matrix that maps the source ROI to the destination ROI."""
        return cv2.findHomography(self.rect_src.corners, self.rect_dst.corners)[0][:3, :3]

    def transform_image(self, image):
        """Transforms an image to the destination ROI."""
        return cv2.warpPerspective(image, self.src_to_dst_mapping, (self.roi_size, self.roi_size)) 

    def transform_points(self, points):
        """Transforms points to the destination ROI."""
        assert points.ndim == 2 and points.shape[-1] == 2, "Expecting a 2D array of points."

        points_h = np.hstack([points, np.ones((points.shape[0], 1))]) # Homogenize
        points_h = points_h.dot(self.src_to_dst_mapping.T)
        return points_h[:, :2] / points_h[:, 2][..., None] # Dehomogenize

    def find_src_roi(self, sample: Sample):
        """Finds the source ROI that encloses the landmarks. Enlarged with scale a factor"""
        bbox = self._get_bbox(sample.landmarks)
        center = np.mean(bbox.corners, axis=0)
        M = cv2.getRotationMatrix2D(center, angle=0, scale=self.scale)
        
        corners = np.hstack([bbox.corners, np.ones((bbox.corners.shape[0], 1))])
        corners_scaled = corners.dot(M.T)
        self.rect_src = self.Rectangle(corners_scaled[0], corners_scaled[3])

        return
    
    def _get_bbox(self, points):
        """Gets the square bounding box that enclose points."""

        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        size = max(max_point - min_point)
        center = (min_point + max_point) / 2

        top_left = center - size / 2
        bottom_right = center + size / 2
        return self.Rectangle(top_left, bottom_right)    

    def __call__(self, sample : tuple):
        """Extracts a region of interest from an image and its landmarks."""
        assert sample.image is not None
        assert sample.landmarks is not None

        self.find_src_roi(sample)

        sample.image = self.transform_image(sample.image)
        sample.landmarks = self.transform_points(sample.landmarks)

        return sample

class NormalizeCoordinates():
    """Normalize coordinates from pixel units to [-1, 1]."""
    def __call__(self, sample: Sample):
        assert (sample.landmarks is not None)
        roi_size = torch.tensor(sample.image.shape[-2::], dtype=sample.landmarks.dtype) 
        sample.landmarks =  (sample.landmarks - (roi_size / 2)) / (roi_size / 2)
        return sample
    
class SampleToTensor():
    """ Turns a NumPy data in a Sample into PyTorch data """

    def __call__(self, sample: Sample):
        sample.image = torch.from_numpy(np.transpose(sample.image, (2, 0, 1)))
        sample.image = sample.image / 255.0 
        sample.image = sample.image.float() 

        if sample.landmarks is not None:
            sample.landmarks = torch.from_numpy(sample.landmarks).float()

        return sample

class FaceLandmarkTransform:
    """Transforms a sample of an image and its landmarks."""
    def __init__(
        self,
        crop_size,
    ):
        self.transform = Compose(
            [
                ExtractRegionOfInterest(roi_size = crop_size),
                SampleToTensor(),
                NormalizeCoordinates()
            ]
        )

    def __call__(self, sample: Sample):
        return self.transform(sample)


class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = F.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s
