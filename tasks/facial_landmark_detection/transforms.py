import math
from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional as F

from torchvision.transforms import ToTensor, Compose
import numpy as np
import cv2

class Sample():

    # pylint: disable=too-many-arguments
    def __init__(self, image=None, landmarks=None):

        self.image = np.array(image)
        self.landmarks = landmarks
        self.warp_region = None

def get_bounds(points):
    """Returns the bounds of a set of N-dimensional points, e.g. 2D or 3D."""
    return np.min(points, axis=0), np.max(points, axis=0)

def get_bounds_middle(points):
    """Gets the middle of the bounds of a set of N-dimensional points."""
    return np.mean(np.stack(get_bounds(points)), axis=0)

def get_square_bounds(points):
    """Gets the square bounds that enclose a set of points."""

    bounds = get_bounds(points)
    size = np.max(bounds[1] - bounds[0])
    center = get_bounds_middle(bounds)

    return np.array([center - size / 2, center + size / 2])

class WarpRegion():
    """A warpable Region Of Interest (ROI) within an image."""

    def __init__(self, top_left, bottom_right, roi_size):
        """Create a warpable region of interest from an axis-aligned bounding box.

        Args:
            top_left: The top-left corner of the source bounding box.
            bottom_right: The bottom-right corner of the source bounding box.
            roi_size: the size in pixels of the desired output ROI image.
        """

        # Source and destination points ordered: [Top Left, Top Right, Bottom Right, Bottom Left]

        tl_x, tl_y = top_left
        br_x, br_y = bottom_right
        self.src_pts = np.array([[tl_x, tl_y], [br_x, tl_y], [br_x, br_y], [tl_x, br_y]], dtype=float)

        self.roi_size = dst_w, dst_h = roi_size
        self.dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=float)

    @property
    def src_pts_h(self):
        """Destination points in homogenous form."""
        return np.hstack([self.src_pts, np.ones((len(self.src_pts), 1))])

    @property
    def src_pts_center(self):
        """The center of the destination points."""
        return tuple(get_bounds_middle(self.src_pts))

    @property
    def matrix(self):
        """The 3x3 perspective matrix for warping source points to destination points."""
        return cv2.findHomography(self.src_pts, self.dst_pts)[0][:3, :3]

    @property
    def bounds(self):
        """The rectangular bounds of the source region: top left and bottom right."""
        return (self.src_pts[0], self.src_pts[2])

    @property
    def size(self):
        """The average size of the region in the source image, in pixels."""
        other_corners = np.roll(self.src_pts, -1, axis=0)
        dists = np.linalg.norm(self.src_pts - other_corners, axis=1)
        return np.mean(dists)

    def shift(self, offset):
        """Shift the source points by some offset, scaled by the size of the region."""
        self.src_pts += offset * self.size

    def scale(self, scale):
        """Uniformly scale the source points about their center."""
        scale_mat = cv2.getRotationMatrix2D(self.src_pts_center, angle=0, scale=scale)
        self.src_pts = self.src_pts_h.dot(scale_mat.T)

    def rotate(self, angle):
        """Rotate the region anti-clockwise."""
        rot_mat = cv2.getRotationMatrix2D(self.src_pts_center, angle=angle, scale=1)
        self.src_pts = self.src_pts_h.dot(rot_mat.T)

    def random_shift(self, amount):
        """Randomly rigidly shift the entire region."""
        self.shift(np.random.uniform(-amount, amount, size=2))

    def random_jiggle(self, amount):
        """Shift each corner of the source quad independently."""
        self.shift(np.random.uniform(-amount, amount, size=self.src_pts.shape))

    def random_scale(self, amount):
        """Randomly scale the source region up or down by a given amount."""
        self.scale(1.0 + np.random.uniform(-amount, amount))

    def random_squash(self, amount_range):
        """Randomly squash along horizontal or vertical axis by a given amount."""

        amount = -np.random.uniform(amount_range[0], amount_range[1])
        # horizontal/vertical
        if np.random.uniform() < 0.5:
            self.dst_pts += np.array([[amount, 0], [-amount, 0], [-amount, 0], [amount, 0]]) * self.roi_size
        else:
            self.dst_pts += np.array([[0, amount], [0, amount], [0, -amount], [0, -amount]]) * self.roi_size

    def random_rotate(self, amount):
        """Randomly rotate the source region by up to a given amount, in degrees."""
        self.rotate(np.random.uniform(-amount, amount))

    def extract_from_image(self, image, **kwargs):
        """Extract this region from an image. Pass additional OpenCV warping arguments via kwargs."""
        return cv2.warpPerspective(image, self.matrix, self.roi_size, **kwargs)

    def transform_points(self, points):
        """Transform a set of 2D points using this ROI's matrix."""

        assert points.ndim == 2 and points.shape[-1] == 2, "Expecting a 2D array of points."

        points_h = np.hstack([points, np.ones((points.shape[0], 1))]) # Homogenize
        points_h = points_h.dot(self.matrix.T)
        return points_h[:, :2] / points_h[:, 2][..., None] # Dehomogenize

    def composite_roi_onto_image(self, roi, image, interp_mode=cv2.INTER_NEAREST):
        """Composites a ROI-sized image onto a full image."""

        assert roi.shape[:2] == self.roi_size

        dsize = image.shape[1], image.shape[0]
        cv2.warpPerspective(roi, self.matrix, dsize, dst=image, borderMode=cv2.BORDER_TRANSPARENT,
                            flags=interp_mode|cv2.WARP_INVERSE_MAP)

    def roi_to_full_image(self, points):
        """Transform 2D points in ROI space to full-frame space."""
        assert points.ndim == 2 and points.shape[-1] == 2, "Expecting a 2D array of points."
        points_h = np.hstack([points, np.ones((points.shape[0], 1))]) # Homogenize
        points_h = points_h.dot(np.linalg.inv(self.matrix).T)
        return points_h[:, :2] / points_h[:, 2][..., None] # Dehomogenize

class GetWarpRegion():
    """Builds a warp region for the face, with square bounds enclosing the given 2D landmarks."""

    def __init__(self, roi_size, scale=2.0, landmarks_definition=None):
        self.roi_size = roi_size
        self.scale = scale
        self.landmarks_definition = landmarks_definition

    def __call__(self, sample: Sample):

        assert sample.landmarks is not None

        ldmks_2d = sample.landmarks
        if self.landmarks_definition:
            ldmks_2d = self.landmarks_definition.apply(sample.landmarks)

        sample.warp_region = WarpRegion(*get_square_bounds(ldmks_2d), self.roi_size)
        sample.warp_region.scale(self.scale)

        return sample

class ExtractWarpRegion():
    """Extract the Warp Region from a sample."""

    def __init__(self, invalid_depth_value=0.0, keep_unwarped=False):
        self.kwargs_bgr = {"flags": cv2.INTER_AREA, "borderMode": cv2.BORDER_REPLICATE}
        self.keep_unwarped = keep_unwarped

    def __call__(self, sample : tuple):

        assert sample.image is not None
        assert sample.warp_region is not None

        warp_region = sample.warp_region

        if self.keep_unwarped:
            # Useful for visualizations and debugging
            sample.image_unwarped = np.copy(sample.image)

        sample.image = warp_region.extract_from_image(sample.image, **self.kwargs_bgr)

        if sample.landmarks is not None:
            sample.landmarks = warp_region.transform_points(sample.landmarks)

        return sample

def normalize_coordinates(coords: torch.Tensor, width: int, height: int):
    """Normalize coordinates from pixel units to [-1, 1]."""
    roi_size = torch.tensor([width, height], device=coords.device, dtype=coords.dtype)
    return (coords - (roi_size / 2)) / (roi_size / 2)


class NormalizeCoordinates():
    """Normalize coordinates from pixel units to [-1, 1]."""
    def __call__(self, sample: Sample):

        assert (sample.landmarks is not None)
        width, height = sample.image.shape[-2::]
        sample.landmarks =  normalize_coordinates(sample.landmarks, width, height)

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
    def __init__(
        self,
        crop_size,
    ):
        self.transform = Compose(
            [
                GetWarpRegion(roi_size = crop_size),
                ExtractWarpRegion(),
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
