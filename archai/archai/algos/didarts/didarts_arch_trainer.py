# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Mapping, Optional, Union, Tuple
import copy

import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn, autograd
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides

from archai.common.config import Config
from archai.nas.arch_trainer import ArchTrainer
from archai.common import utils, ml_utils
from archai.nas.model import Model
from archai.common.checkpoint import CheckPoint
from archai.common.common import logger
from archai.common.multi_optim import MultiOptim, OptimSched

class DidartsArchTrainer(ArchTrainer):
    """Train network using different optimizers for alphas and other parameters"""

    def __init__(self, conf_train: Config, model: Model,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)

        self._conf_alpha_optim = conf_train['alpha_optimizer']
        self._conf_alpha_sched = conf_train['alpha_lr_schedule']


    @overrides
    def create_multi_optim(self, train_len:int)->MultiOptim:
        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state specific to each run
        optim = self.create_optimizer(self.conf_optim, self.model.nonarch_params(recurse=True))
        # create scheduler for optim before applying amp
        sched, sched_on_epoch = self.create_scheduler(self.conf_sched, optim, train_len)

        alpha_optim = self.create_optimizer(self._conf_alpha_optim,
                                            self.model.all_owned().param_by_kind(None))
        alpha_sched, alpha_sched_on_epoch = self.create_scheduler(self._conf_alpha_sched, alpha_optim, train_len)

        multi_optim = MultiOptim()
        multi_optim.append(OptimSched(optim, sched, sched_on_epoch))
        multi_optim.append(OptimSched(alpha_optim, alpha_sched, alpha_sched_on_epoch))

        logger.info({'multi_optim_len': len(multi_optim)})

        return multi_optim



