"""This module contains methods and callbacks for visualizing training or validation data."""

from heapq import heapify, heappush, heappushpop
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch

#from face_synthetics_training.core.utils import add_text, draw_landmark_connectivity, draw_landmarks, make_image_grid
#from face_synthetics_training.training.utils import bgr_img_to_np, to_np

UINT8_MAX = 255
def bgr_img_to_np(bgr_img_pt):
    """Converts a PyTorch (C, H, W) BGR float image [0, 1] into a NumPy (H, W, C) UINT8 image [0, 255]."""

    assert isinstance(bgr_img_pt, torch.Tensor)
    bgr_img_np = np.clip(np.transpose(bgr_img_pt.cpu().detach().numpy(), (1, 2, 0)), 0, 1)
    return np.ascontiguousarray((bgr_img_np * UINT8_MAX).astype(np.uint8))


def to_np(tensor: torch.Tensor):
    """Convenience function for NumPy-ifying a PyTorch tensor."""
    isinstance(tensor, torch.Tensor)
    return tensor.cpu().detach().numpy()

def draw_landmarks(img, ldmks_2d, thickness=1, thicknesses=None, color=(255, 255, 255),
                   colors=None, ldmks_visibility=None):
    """Drawing dots on an image."""
    # pylint: disable=too-many-arguments
    assert img.dtype == np.uint8

    img_size = (img.shape[1], img.shape[0])

    for i, ldmk in enumerate(ldmks_2d.astype(int)):
        if ldmks_visibility is not None and ldmks_visibility[i] == 0:
            continue
        if np.all(ldmk > 0) and np.all(ldmk < img_size):
            l_c = tuple(colors[i] if colors is not None else color)
            t = thicknesses[i] if thicknesses is not None else thickness
            cv2.circle(img, tuple(ldmk+1), t, 0, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(ldmk), t, l_c, -1, cv2.LINE_AA)


def visualize_landmarks(
    color_image: np.ndarray,
    label_landmarks: Optional[np.ndarray] = None,
    predicted_landmarks: Optional[np.ndarray] = None,
    name: Optional[str] = None,
    connectivity: Optional[np.ndarray] = None,
    error=None,
    include_original_image: bool = False,
) -> np.ndarray:
    """Creates a visualization of landmarks on a training image.

    Args:
        color_image (np.ndarray): The color image, e.g. an image of a face.
        label_landmarks (Optional[np.ndarray], optional): The label or GT landmarks. Defaults to None.
        predicted_landmarks (Optional[np.ndarray], optional): The landmarks predicted by a network. Defaults to None.
        name (Optional[str], optional): The name of the image. Defaults to None.
        connectivity (Optional[np.ndarray], optional): The connectivity between landmark pairs. Defaults to None.
        error ([type], optional): The average Euclidean landmark error. Defaults to None.
        include_original_image (bool, optional): If true, also include the original image without annotation.

    Returns:
        np.ndarray: [description]
    """
    # pylint: disable=too-many-arguments
    vis_img = color_image.copy()
    if connectivity is not None:
        if label_landmarks is not None:
            draw_landmark_connectivity(vis_img, label_landmarks, connectivity, color=(0, 255, 0))
        if predicted_landmarks is not None:
            draw_landmark_connectivity(vis_img, predicted_landmarks, connectivity)
    else:
        if label_landmarks is not None:
            draw_landmarks(vis_img, label_landmarks, color=(0, 255, 0))
        if predicted_landmarks is not None:
            draw_landmarks(vis_img, predicted_landmarks, color=(0, 165, 255))

    return np.vstack([color_image, vis_img]) if include_original_image else vis_img

def unnormalize_coordinates(coords: np.array, img_size: Tuple[int, int]):
    """Unnormalize coordinates from [-1, 1] to pixel units."""
    img_size = np.divide(img_size, 2)
    coords_pixels = np.add(img_size, np.multiply(coords, img_size))
    return coords_pixels


def visualize_batch_data(
    img_file_prefix: str,
    epoch: int, #epoch number
    outputs,
    batch: Any, #image, label tuple
    batch_idx: int = 0
):
    """At the end of each training batch, dump an image visualizing label and predicted landmarks."""
    # We are overriding, and do not use all arguments, so:
    # pylint: disable=too-many-arguments,signature-differs,unused-argument

    if batch_idx == 0:  # Visualize first batch only
        vis_imgs = []

        batch_size = batch[0].shape[0]
        num_images = 1 
        for img_idx in range(num_images):
            color_image = bgr_img_to_np(batch[0][img_idx]).copy()
            label_coordinates = to_np(batch[1][img_idx])
            predicted_coords = to_np(outputs[img_idx])

            img_size = color_image.shape[0:2]
            label_coordinates_unnormalized = unnormalize_coordinates(label_coordinates, img_size)
            predicted_coords_unnormalized = unnormalize_coordinates(predicted_coords, img_size)

            vis_img = visualize_landmarks(
                color_image=color_image,
                label_landmarks=label_coordinates_unnormalized,
                predicted_landmarks=predicted_coords_unnormalized)
            vis_imgs.append(vis_img)

        cv2.imwrite(f"{img_file_prefix}_{epoch:04d}.jpg", vis_imgs[0])
        #batch_visualization = make_image_grid(vis_imgs, min(num_images, 8))
        #cv2.imwrite(str(self.log_dir / f"vis_img_train_{epoch:04d}.jpg"), batch_visualization)
    """
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        vis_imgs = []

        # Sort all samples by score, and visualize them
        for _, error, _, sample in sorted(self.samples):
            vis_imgs.append(visualize_landmarks(**sample, connectivity=self.connectivity, error=error))

        batch_visualization = make_image_grid(vis_imgs, min(len(self.samples), 8))
        cv2.imwrite(str(self.log_dir / f"{self.name}_{epoch:04d}.jpg"), batch_visualization)
    """

