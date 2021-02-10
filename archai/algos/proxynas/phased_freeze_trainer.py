# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Callable, Type, List
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
from archai.nas.nas_utils import get_model_stats
from archai.nas.model_desc import ModelDesc
from archai.nas.arch_trainer import ArchTrainer
from archai.common.trainer import Trainer
from archai.nas.vis_model_desc import draw_model_desc
from archai.common.checkpoint import CheckPoint
from archai.common.ml_utils import set_optim_lr
from archai.datasets import data

TFreezeTrainer = Optional[Type['PhasedFreezeTrainer']]


class PhasedFreezeTrainer(ArchTrainer, EnforceOverrides):
    def __init__(self, conf_train: Config, model: nn.Module,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)

        # region config vars specific to freeze trainer
        self.phase_identifiers:List[List] = self.conf_train['phase_identifiers']
        self.num_epochs_per_phase = self.conf_train['num_epochs_per_phase']
        # end region

        # track freeze phases
        self.phase_index = 1

        # flag to terminate training
        self._terminate = False


    @overrides
    def _should_terminate(self):
        if self._terminate:
            return True


    @overrides 
    def post_epoch(self, data_loaders:data.DataLoaders)->None:
        super().post_epoch(data_loaders)

        # freeze all layers up to current phase
        if self.epoch() % self.num_epochs_per_phase == 0:
            self._freeze_specific_layers(self.phase_identifiers[:self.phase_index])
            self.phase_index += 1

        # if phase_index is whole network stop training
        if self.phase_index == len(self.phase_identifiers):
            self._terminate = True       


    def _freeze_specific_layers(self, layer_identifiers:List[List])->None:

        for name, param in self.model.named_parameters():
            for id_list in layer_identifiers:
                for identifier in id_list:
                    if identifier in name:
                        param.requires_grad = False

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                logger.info(f'{name} frozen')

        






    
