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
from archai.datasets import data

TFreezeTrainer = Optional[Type['FreezeTrainer']]


class FreezeTrainer(ArchTrainer, EnforceOverrides):
    def __init__(self, conf_train: Config, model: nn.Module,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)


    @overrides
    def pre_fit(self, data_loaders:data.DataLoaders) -> None:
        super().pre_fit(data_loaders)

        # freeze everything other than the last layer
        if not self.conf_train['bypass_freeze']:
            # addup parameters which are not frozen
            num_frozen_params = 0
            for l in model_stats.layer_stats:
                for identifier in self.conf_train['identifiers_to_unfreeze']:
                    if identifier in l.name:
                        num_frozen_params += l.parameters            
            ratio_unfrozen = num_frozen_params / model_stats.parameters 
            logger.info(f'unfrozen parameters ratio {ratio_unfrozen}')

            self._freeze_but_last_layer()
        else:
            logger.info(f'Bypassing freezing!')


    def _freeze_but_last_layer(self) -> None:

        # Do it via parameters
        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            for identifier in self.conf_train['identifiers_to_unfreeze']:
                if identifier in name:
                    param.requires_grad = True

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(f'{name} requires grad')