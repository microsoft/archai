# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from archai.datasets.cv.mnist_dataset_provider import MnistDatasetProvider
import torch
import pytorch_lightning as pl


class MNistDataModule(pl.LightningDataModule):
    def __init__(self, path):
        super().__init__()
        self.root = path

    def prepare_data(self):
        self.dataset_provider = MnistDatasetProvider(root=self.root)
        self.tr_data = self.dataset_provider.get_train_dataset()
        self.val_data = self.dataset_provider.get_val_dataset()
        self.input_shape = self.tr_data.data[0].shape

    def prepare_data_per_node(self):
        self.prepare_data()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.tr_data, batch_size=16, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=16, shuffle=False, num_workers=4)

    def test_dataloader(self):
        # MNIST doesn't have a test dataset, so just reuse the validation dataset.
        return torch.utils.data.DataLoader(self.val_data, batch_size=16, shuffle=False, num_workers=4)
