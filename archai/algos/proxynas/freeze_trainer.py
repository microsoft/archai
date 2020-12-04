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
from archai.common.common import logger
from archai.nas.model import Model
from archai.nas.model_desc import ModelDesc
from archai.nas.arch_trainer import ArchTrainer
from archai.common.trainer import Trainer
from archai.nas.vis_model_desc import draw_model_desc
from archai.common.checkpoint import CheckPoint
from archai.common.ml_utils import set_optim_lr

TFreezeTrainer = Optional[Type['FreezeTrainer']]


class FreezeTrainer(ArchTrainer, EnforceOverrides):
    def __init__(self, conf_train: Config, model: nn.Module,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint) 

        # region config vars specific to freeze trainer
        self._val_top1_acc = conf_train['proxynas']['val_top1_acc_threshold']
        self._in_freeze_mode = False
        # endregion

        self._epoch_freeze_started = None
        self._max_epochs = None

    @overrides
    def post_epoch(self, train_dl: DataLoader, val_dl: Optional[DataLoader]) -> None:
        super().post_epoch(train_dl, val_dl)

        # if current validation accuracy is above threshold
        # freeze everything other than the last layer
        best_val_top1_avg = self._metrics.best_val_top1()

        if best_val_top1_avg >= self._val_top1_acc and not self._in_freeze_mode:

            # freeze everything other than the last layer
            self.freeze_but_last_layer()

            # reset optimizer
            del self._multi_optim
            
            self.conf_optim['lr'] = self.conf_train['proxynas']['freeze_lr']
            self.conf_optim['decay'] = self.conf_train['proxynas']['freeze_decay']
            self.conf_optim['momentum'] = self.conf_train['proxynas']['freeze_momentum']
            self.conf_sched = Config()
            self._aux_weight = self.conf_train['proxynas']['aux_weight']

            self.model.zero_grad()
            self._multi_optim = self.create_multi_optim(len(train_dl))
            # before checkpoint restore, convert to amp
            self.model = self._apex.to_amp(self.model, self._multi_optim,
                                           batch_size=train_dl.batch_size)
        
            self._in_freeze_mode = True
            self._epoch_freeze_started = self._metrics.epochs()
            self._max_epochs = self._epoch_freeze_started + self.conf_train['proxynas']['freeze_epochs']
            logger.info('-----------Entered freeze training-----------------')


    def freeze_but_last_layer(self) -> None:
        
        # NOTE: assumption here is that the last 
        # layer has the word 'logits' in the name string
        # e.g. logits_op._op.weight, logits_op._op.bias
        # e.g. _aux_towers.13.logits_op.weight, _aux_towers.13.logits_op.bias

        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        for name, param in self.model.named_parameters():
            # TODO: Make the layer names to be updated a config value
            # 'fc' for resnet18
            # 'logits_op._op' for darts search space
            for identifier in self.conf_train['proxynas']['identifiers_to_unfreeze']:
                if identifier in name:
                    param.requires_grad = True

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(f'{name} requires grad')

            