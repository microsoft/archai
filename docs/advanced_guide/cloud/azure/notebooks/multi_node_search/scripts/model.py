# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from torch import nn
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from archai.discrete_search.search_spaces.config import ArchConfig


class MyModel(pl.LightningModule):
    """ This is a simple CNN model that can learn how to do MNIST classification.
    It is parametrized by a given ArchConfig that defines number of layers, the
    convolution kernel size, and the number of features.
    """

    def __init__(self, arch_config: ArchConfig, lr: float = 1e-4):
        super().__init__()

        self.lr = lr
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.nb_layers = arch_config.pick('nb_layers')
        self.kernel_size = arch_config.pick('kernel_size')
        self.hidden_dim = arch_config.pick('hidden_dim')

        layer_list = []

        for i in range(self.nb_layers):
            in_ch = (1 if i == 0 else self.hidden_dim)

            layer_list += [
                nn.Conv2d(in_ch, self.hidden_dim, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU(),
            ]

        layer_list += [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(self.hidden_dim, 10, kernel_size=1)
        ]

        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x).squeeze()

    def get_archid(self):
        return f'({self.nb_layers}, {self.kernel_size}, {self.hidden_dim})'

    def export_onnx(self, input_shape, path):
        # add batch and channel dimensions
        shape = [1, 1] + list(input_shape)
        dummy_input = torch.randn(shape, device="cpu")
        torch.onnx.export(self.model, dummy_input, path,
                          input_names=['input'],
                          output_names=['output'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        device = x.device
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
        acc = accuracy(logits, y)
        self.log('accuracy', acc)
        return loss

    @staticmethod
    def from_archid(model_id):
        nb_layers,  kernel_size, hidden_dim = eval(model_id)
        return MyModel(ArchConfig({
            "nb_layers": nb_layers,
            "kernel_size": kernel_size,
            "hidden_dim": hidden_dim
        }))
