# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from archai.datasets.cv.mnist_dataset_provider import MnistDatasetProvider
from archai.trainers.cv.pl_trainer import PlTrainer


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)

        x_hat = self.linear(x)
        loss = F.cross_entropy(x_hat, y)

        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)

        x_hat = self.linear(x)
        loss = F.cross_entropy(x_hat, y)

        self.log("val_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = x.view(x.size(0), -1)

        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def test_pl_trainer():
    # make sure tests can run in parallel and not clobber each other's dataroot.
    unique_data_root = 'test_pl_trainer_dataroot'
    model = Model()
    trainer = PlTrainer(max_steps=1, limit_train_batches=1, limit_test_batches=1, limit_predict_batches=1)

    dataset_provider = MnistDatasetProvider(root=unique_data_root)
    train_dataset = dataset_provider.get_train_dataset()
    val_dataset = dataset_provider.get_val_dataset()

    # Assert that the trainer correctly calls `train`, `evaluate` and `predict`
    trainer.train(model, DataLoader(train_dataset))
    trainer.evaluate(model, DataLoader(val_dataset))
    trainer.predict(model, DataLoader(val_dataset))

    shutil.rmtree(unique_data_root)
    shutil.rmtree("lightning_logs")
