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
                 checkpoint:Optional[CheckPoint], max_duration_secs:Optional[float]) -> None:
        super().__init__(conf_train, model, checkpoint)

        # region config vars specific to freeze trainer
        self._top1_acc_threshold = conf_train['top1_acc_threshold']
        self._use_val = conf_train['use_val']
        self._max_duration_secs = max_duration_secs
        # endregion

    @overrides
    def _should_terminate(self):
        # if current validation accuracy is above threshold
        # terminate training
        if self._use_val:
            best_top1_avg = self._metrics.best_val_top1()
        else:
            best_top1_avg = self._metrics.best_train_top1()

        should_terminate = False

        if best_top1_avg >= self._top1_acc_threshold:
            logger.info(f'terminating at {best_top1_avg}')
            logger.info('----------terminating regular training---------')
            should_terminate = True
        
        # terminate if maximum training duration 
        # threshold is exceeded
        if self._max_duration_secs:
            if self._metrics.total_training_time >= self._max_duration_secs:
                should_terminate = True
        
        return should_terminate

    @overrides
    def pre_fit(self, data_loaders:data.DataLoaders)->None:
        super().pre_fit(data_loaders)