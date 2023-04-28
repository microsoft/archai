# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Tuple

import cv2
import numpy as np
import torch

from torch import Tensor
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import functional as F

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
