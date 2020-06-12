# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Callable, Type
import os

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides, EnforceOverrides

from ..common.config import Config
from ..common import common, utils
from ..nas.model import Model
from ..nas.model_desc import ModelDesc
from ..common.trainer import Trainer
from ..nas.vis_model_desc import draw_model_desc
from ..common.checkpoint import CheckPoint

TArchTrainer = Optional[Type['ArchTrainer']]

class ArchTrainer(Trainer, EnforceOverrides):
    def __init__(self, conf_train: Config, model: Model,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)

        self._l1_alphas = conf_train['l1_alphas']
        self._plotsdir = conf_train['plotsdir']

        # if l1 regularization is needed then cache alphas
        if self._l1_alphas > 0.0:
            self._alphas = list(self.model.all_owned().param_by_kind('alphas'))

    @overrides
    def compute_loss(self, lossfn: Callable,
                     y: Tensor, logits: Tensor,
                     aux_weight: float, aux_logits: Optional[Tensor]) -> Tensor:
        loss = super().compute_loss(lossfn, y, logits,
                                    aux_weight, aux_logits)
        # add L1 alpha regularization
        if self._l1_alphas > 0.0:
            l_extra = sum(torch.sum(a.abs()) for a in self._alphas)
            loss += self._l1_alphas * l_extra
        return loss

    @overrides
    def post_epoch(self, train_dl: DataLoader, val_dl: Optional[DataLoader])->None:
        super().post_epoch(train_dl, val_dl)
        self._draw_model()

    # TODO: move this outside as utility
    def _draw_model(self) -> None:
        if not self._plotsdir:
            return
        train_metrics = self.get_metrics()
        if train_metrics:
            best_train, best_val = train_metrics.run_metrics.best_epoch()
            is_best = best_val and best_val==train_metrics.cur_epoch()
            is_best = is_best or best_train==train_metrics.cur_epoch()
            if is_best:
                # log model_desc as a image
                plot_filepath = utils.full_path(os.path.join(
                                    self._plotsdir,
                                    f"EP{train_metrics.cur_epoch().index:03d}"),
                                create=True)
                draw_model_desc(self.model.finalize(), filepath=plot_filepath,
                                caption=f"Epoch {train_metrics.cur_epoch().index}")
