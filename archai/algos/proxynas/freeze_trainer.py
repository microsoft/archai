# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Callable, Type
import os

import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides, EnforceOverrides

from archai.common.config import Config
from archai.common import common, utils
from archai.nas.model import Model
from archai.nas.model_desc import ModelDesc
from archai.nas.arch_trainer import ArchTrainer
from archai.common.trainer import Trainer
from archai.nas.vis_model_desc import draw_model_desc
from archai.common.checkpoint import CheckPoint

TFreezeTrainer = Optional[Type['FreezeTrainer']]


class FreezeTrainer(ArchTrainer, EnforceOverrides):
    def __init__(self, conf_train: Config, model: nn.Module,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint) 

        # region config vars specific to freeze trainer
        self.conf_train = conf_train
        self._val_top1_acc = conf_train['val_top1_acc_threshold']
        # endregion

    @overrides
    def post_epoch(self, train_dl: DataLoader, val_dl: Optional[DataLoader]) -> None:
        super().post_epoch(train_dl, val_dl)

        # if current validation accuracy is above
        # freeze everything other than the last layer
        best_val_top1_avg = self._metrics.best_val_top1()

        if best_val_top1_avg >= self._val_top1_acc:

            # freeze everything other than the last layer
            self.freeze_but_last_layer()

            # reset optimizer
            del self._multi_optim
            self._multi_optim = self.create_multi_optim(len(train_dl))


    def freeze_but_last_layer(self) -> None:
        # first freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # NOTE: assumption here is that the last 
        # layer has the word 'logits' in the name string
        # e.g. logits_op._op.weight, logits_op._op.bias
        # e.g. _aux_towers.13.logits_op.weight, _aux_towers.13.logits_op.bias
        # TODO: confirm from Shital that this is good!
        
        for name, param in self.model.named_parameters():
            if 'logits' in name:
                param.requires_grad = True


            