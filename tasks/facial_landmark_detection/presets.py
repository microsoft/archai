import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToTensor, Compose
import numpy as np
import cv2

class ClassificationPresetTrain:
    def __init__(
        self,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)

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
