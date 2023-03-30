from pathlib import Path
from typing import Tuple

import torch
from torch import nn
import pytorch_lightning as pl

from .metrics import get_confusion_matrix, get_iou, get_f1_scores


class SegmentationTrainingLoop(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 2e-4, ignore_mask_value: int = 255):
        super().__init__()
        
        self.model = model
        self.num_classes = model.num_classes
        self.in_channels = model.in_channels
        self.ignore_mask_value = ignore_mask_value
        self.lr = lr

        self.save_hyperparameters(ignore=['model'])
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, stage='train'):
        image = batch['image']
        mask = batch['mask']
        batch_size, _, height, width = image.shape

        assert image.ndim == 4
        logits_mask = self.forward(image) # (N, C, H, W)
        logits_mask = logits_mask.view(batch_size, self.num_classes, -1)
        
        loss = self.loss(logits_mask, mask.view(batch_size, -1))

        pred_classes = logits_mask.argmax(axis=1)
        confusion_matrix = get_confusion_matrix(
            pred_classes.cpu(), mask.cpu(), self.num_classes, self.ignore_mask_value
        )

        iou_dict = get_iou(confusion_matrix)
        f1_dict = get_f1_scores(confusion_matrix)

        results = {
            f'{stage}_loss': loss,
            f'{stage}_mIOU': iou_dict['mIOU'],
            f'{stage}_macro_f1': f1_dict['macro_f1'],
            f'{stage}_weighted_f1': f1_dict['weighted_f1']
        }

        return results

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, stage='train')
        self.log_dict(results, sync_dist=True, on_step=True, on_epoch=True)
        return results['train_loss']

    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch, stage='validation')
        self.log_dict(results, sync_dist=True, on_step=False, on_epoch=True)
        return results

    def predict(self, image):
        with torch.no_grad():
            return self.model.predict(image)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]
