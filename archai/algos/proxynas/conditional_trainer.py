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
from archai.nas.nas_utils import get_model_stats
from archai.nas.model_desc import ModelDesc
from archai.nas.arch_trainer import ArchTrainer
from archai.common.trainer import Trainer
from archai.nas.vis_model_desc import draw_model_desc
from archai.common.checkpoint import CheckPoint
from archai.common.ml_utils import set_optim_lr
from archai.datasets import data

TFreezeTrainer = Optional[Type['ConditionalTrainer']]


class ConditionalTrainer(ArchTrainer, EnforceOverrides):
    def __init__(self, conf_train: Config, model: nn.Module,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)

        # region config vars specific to freeze trainer
        self._val_top1_acc = conf_train['val_top1_acc_threshold']
        # endregion

    @overrides
    def _should_terminate(self):
        # if current validation accuracy is above threshold
        # terminate training
        best_val_top1_avg = self._metrics.best_val_top1()

        if best_val_top1_avg >= self._val_top1_acc:
            logger.info(f'terminating at {best_val_top1_avg}')
            logger.info('----------terminating regular training---------')
            return True
        else:
            return False


    @overrides
    def pre_fit(self, data_loaders:data.DataLoaders)->None:
        super().pre_fit(data_loaders)