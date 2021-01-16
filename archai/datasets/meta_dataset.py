from typing import List, Optional
import torch
from torch.utils.data import Dataset

class MetaDataset(Dataset):
    def __init__(self, source:Dataset, transform=None, target_transform=None) -> None:
        self._source = source
        self.transform = transform
        self.target_transform = target_transform

        self._meta = [{'idx':i} for i in range(len(source))]

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx):
        t = self._source[idx]
        return tuple(*t, self._meta[idx])

