from typing import List, Optional
import torch
from torch.utils.data import Dataset

class MetaDataset(Dataset):
    def __init__(self, source:Dataset, transform=None, target_transform=None) -> None:
        self._source = source
        self.transform = transform if transform is not None else lambda x: x
        self.target_transform = target_transform if target_transform is not None else lambda x: x

        self._meta = [{'idx':i} for i in range(len(source))]

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx):
        x, y = self._source[idx]
        return (self.transform(x), self.target_transform(y), self._meta[idx])

