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
    def __init__(self, conf_train: Config, model: Model, device,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, device, checkpoint)

        self._conf_w_optim = conf_train['optimizer']
        # self._conf_w_lossfn = conf_train['lossfn']

    @overrides
    def create_optimizer(self) -> Optimizer:
        # in this case we don't need to differentiate between alphas and weights
        # as the same optimizer will update both
        param_groups = [{'params': self.model.weights()}, {'params': self.model.alphas()}]
        return ml_utils.create_optimizer(self._conf_w_optim, param_groups)
