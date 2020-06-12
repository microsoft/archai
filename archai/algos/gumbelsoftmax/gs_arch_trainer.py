# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Mapping, Optional, Union
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


class GsArchTrainer(ArchTrainer):
    @overrides
    def create_optimizer(self, conf_optim:Config, params) -> Optimizer:
        # in this case we don't need to differentiate between arch_params and weights
        # as the same optimizer will update both
        arch_params = list(self.model.all_owned().param_by_kind('alphas'))
        nonarch_params = list(self.model.nonarch_params(recurse=True))
        param_groups = [{'params': nonarch_params}, {'params': arch_params}]
        return ml_utils.create_optimizer(conf_optim, param_groups)
