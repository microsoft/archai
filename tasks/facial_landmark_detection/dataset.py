"""Datasets for Microsoft Face Synthetics dataset."""

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import transforms

class FaceLandmarkDataset(Dataset):
    """Dataset class for Microsoft Face Synthetics dataset.

    Args:
        directory (str): Path to the directory containing the PNG images and landmarks files.
        limit (int, optional): Maximum number of samples to load from the dataset. Defaults to None.
        crop_size (int, optional): Size of the square crop to apply to the images. Defaults to 128.

    Attributes:
        png_files (list): List of paths to the PNG image files in the dataset.
        transform (FaceLandmarkTransform): Transform to apply to the samples.
        _num_landmarks (int): Number of landmarks in each sample.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the image and landmarks of the sample at the given index.
        num_landmarks(): Returns the number of landmarks in each sample.
    """
    def __init__(self, directory, limit=None, crop_size = 128):

        pattern = os.path.join(directory, "[0-9][0-9][0-9][0-9][0-9][0-9].png") #don't load *_seg.png files
        self.png_files = glob.glob(pattern) 
        assert len(self.png_files) > 0, f"Can't find any PNG image in folder: {directory}"
        if limit is not None:
            self.png_files = self.png_files[:limit]
        self.transform = transforms.FaceLandmarkTransform(crop_size = crop_size)
        self._num_landmarks = None

    def __len__(self):

        return len(self.png_files)

    def __getitem__(self, index):
        """
        Returns the image and landmarks of the sample at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and landmarks of the sample.
        """
        png_file = self.png_files[index]
        image = Image.open(png_file)
        label_file = png_file.replace(".png", "_ldmks.txt")
        label = np.loadtxt(label_file, dtype=np.single) 
        assert label.size > 0, "Can't find data in landmarks file: f{label_file}"
        #label[:, 1] = image.height - label[:, 1]  #flip due to the landmarks Y definition

        sample = transforms.Sample(image=image, landmarks=label)
        assert sample is not None
        sample_transformed = self.transform(sample)
        assert sample_transformed is not None

        return sample_transformed.image, sample_transformed.landmarks
    
    @property
    def num_landmarks(self):
        """
        Returns the number of landmarks in each sample.

        Returns:
            int: The number of landmarks in each sample.
        """
        if self._num_landmarks is None:
            _, label = self.__getitem__(0)
            self._num_landmarks = torch.numel(label)
        return self._num_landmarks
