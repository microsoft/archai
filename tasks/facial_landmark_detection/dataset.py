"""Datasets for Microsoft Face Synthetics dataset."""

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import presets


class FaceLandmarkDataset(Dataset):
    def __init__(self, directory, limit=None, crop_size = 128):
        """ initialize """
        pattern = os.path.join(directory, "[0-9][0-9][0-9][0-9][0-9][0-9].png") #don't load *_seg.png files
        self.png_files = glob.glob(pattern) 
        assert len(self.png_files) > 0, f"Can't find any PNG image in folder: {directory}"
        if limit is not None:
            self.png_files = self.png_files[:limit]
        self.transform = presets.FaceLandmarkTransform(crop_size = (crop_size, crop_size))
        self._num_landmarks = None

    def __len__(self):

        return len(self.png_files)

    def __getitem__(self, index):
        """get a sample"""
        png_file = self.png_files[index]
        image = Image.open(png_file)
        label_file = png_file.replace(".png", "_ldmks.txt")
        label = np.loadtxt(label_file, dtype=np.single) 
        assert label.size > 0, "Can't find data in landmarks file: f{label_file}"
        #label[:, 1] = image.height - label[:, 1]  #flip due to the landmarks Y definition

        sample = presets.Sample(bgr_img=image, ldmks_2d=label)
        assert sample is not None
        sample_transformed = self.transform(sample)
        assert sample_transformed is not None

        return sample_transformed.bgr_img, sample_transformed.ldmks_2d
    
    @property
    def num_landmarks(self):
        """ number of landmarks in each sample"""
        if self._num_landmarks is None:
            _, label = self.__getitem__(0)
            self._num_landmarks = torch.numel(label)
        return self._num_landmarks
    
