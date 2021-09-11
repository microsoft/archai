
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBarBase
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
import sys
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import shutil
import subprocess


class TransformerLightning(pl.LightningModule):
    def __init__(self, model:nn.Module, dataset:Dataset, hparams:dict):
        super(TransformerLightning, self).__init__()
        self.model, self.dataset, self.hparams = \
            model, dataset, hparams

    def forward(self, inputs):
        return self.model(**inputs, return_dict=False)

    def training_step(self, batch, batch_num):
        "Compute loss and log."

        outputs = self.forward({"input_ids": batch, "labels": batch})
        loss = outputs[0]

        return {"loss": loss}

    def train_dataloader(self):
        "Load datasets. Called after prepare data."

        return DataLoader(
            self.dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            pin_memory=self.hparams["pin_memory"],
            num_workers=self.hparams["num_workers"],
        )

    def configure_optimizers(self):
        "Prepare optimizer"

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams["learning_rate"],
            eps=self.hparams["adam_epsilon"],
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams["warmup_steps"],
            num_training_steps=self.hparams["num_steps"],
        )

        return [optimizer], [scheduler]