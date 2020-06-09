from typing import Mapping, Optional, Union
import copy
import math as ma

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
from archai.nas.model_desc import CellType
from archai.common.checkpoint import CheckPoint
from archai.common.common import logger
from archai.common.utils import zip_eq
from archai.common.common import get_conf
from .xnas_op import XnasOp



class XnasArchTrainer(ArchTrainer):
    def __init__(self, conf_train: Config, model: Model,
                 checkpoint: Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)

        self._conf_w_lossfn = conf_train['lossfn']

    @overrides
    def create_optimizer(self, conf_optim: Config, params) -> Optimizer:
        # return optim that only operates on w, not alphas
        return ml_utils.create_optimizer(conf_optim,
                                         self.model.nonarch_params(recurse=True))

    @overrides
    def pre_fit(self, train_dl: DataLoader, val_dl: Optional[DataLoader]) -> None:
        super().pre_fit(train_dl, val_dl)

        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state
        assert val_dl is not None
        lossfn = ml_utils.get_lossfn(self._conf_w_lossfn).to(self.get_device())

        conf = get_conf()
        self._train_batch = conf['nas']['search']['loader']['train_batch']
        num_val_examples = len(val_dl) * self._train_batch
        num_cells = conf['nas']['search']['model_desc']['n_cells']
        num_reduction_cells = conf['nas']['search']['model_desc']['n_reductions']
        num_normal_cells = num_cells - num_reduction_cells
        num_primitives = len(XnasOp.PRIMITIVES)

        assert num_cells > 0
        assert num_reduction_cells > 0
        assert num_normal_cells > 0
        assert num_primitives > 0

        self._normal_cell_effective_t = num_val_examples * self._epochs * num_normal_cells
        self._reduction_cell_effective_t = num_val_examples * self._epochs * num_reduction_cells
    
        self._normal_cell_lr = ma.sqrt(2 * ma.log(num_primitives) / (self._normal_cell_effective_t * self._grad_clip * self._grad_clip))
        self._reduction_cell_lr = ma.sqrt(2 * ma.log(num_primitives) / (self._reduction_cell_effective_t * self._grad_clip * self._grad_clip))

        self._xnas_optim = _XnasOptimizer(self._normal_cell_lr, self._reduction_cell_lr, self._normal_cell_effective_t, self._reduction_cell_effective_t, self._train_batch, self.model, lossfn)

    @overrides
    def post_fit(self, train_dl: DataLoader, val_dl: Optional[DataLoader]) -> None:
        # delete state we created in pre_fit
        del self._xnas_optim
        return super().post_fit(train_dl, val_dl)

    @overrides
    def pre_epoch(self, train_dl: DataLoader, val_dl: Optional[DataLoader]) -> None:
        super().pre_epoch(train_dl, val_dl)

        # prep val set to train alphas
        self._valid_iter = iter(val_dl)  # type: ignore

    @overrides
    def post_epoch(self, train_dl: DataLoader, val_dl: Optional[DataLoader]) -> None:
        del self._valid_iter  # clean up
        super().post_epoch(train_dl, val_dl)

    @overrides
    def pre_step(self, x: Tensor, y: Tensor) -> None:
        super().pre_step(x, y)

        # reset val loader if we exhausted it
        try:
            x_val, y_val = next(self._valid_iter)
        except StopIteration:
            # reinit iterator
            self._valid_iter = iter(self._val_dl)
            x_val, y_val = next(self._valid_iter)

        x_val, y_val = x_val.to(self.get_device()), y_val.to(
            self.get_device(), non_blocking=True)

        # update alphas

        self._multi_optim.zero_grad()
        self._xnas_optim.step(x, y, x_val, y_val,
                              self.epoch(), self._epochs, self._grad_clip, self._multi_optim)
        self._multi_optim.zero_grad()

    @overrides
    def update_checkpoint(self, checkpoint: CheckPoint) -> None:
        super().update_checkpoint(checkpoint)


class _XnasOptimizer:
    def __init__(self, ncell_lr: float, rcell_lr: float,
                 ncell_effective_t: float, rcell_effective_t: float, train_batch: int,
                 model: Model, lossfn: _Loss) -> None:
        self._ncell_lr = ncell_lr
        self._rcell_lr = rcell_lr
        self._ncell_effective_t = ncell_effective_t
        self._rcell_effective_t = rcell_effective_t
        self._train_batch = train_batch

        self._lossfn = lossfn
        self._model = model  # main model with respect to w and alpha

    @staticmethod
    def _get_loss(model, lossfn, x, y):
        logits, *_ = model(x)  # might also return aux tower logits
        return lossfn(logits, y)

    def step(self, x_train: Tensor, y_train: Tensor, x_valid: Tensor, y_valid: Tensor, epoch: int, epochs: int, grad_clip: float, optim) -> None:
        # put model in train mode just to be safe
        self._model.train()

        # put model through val data
        loss = self._get_loss(self._model, self._lossfn, x_valid, y_valid)

        # compute gradients
        loss.backward()

        # do grad clip
        self._apex.clip_grad(grad_clip, model, optim)

        # for each op in the model update alphas        
        for cell in self._model.cells:
            if cell.desc.cell_type  == CellType.Reduction:
                lr = self._rcell_lr
                T = self._rcell_effective_t
            elif cell.desc.cell_type == CellType.Regular:
                lr = self._ncell_lr
                T = self._ncell_effective_t
            else:
                raise NotImplementedError

            t = epoch * self._train_batch
            for op in cell.ops():
                op.update_alphas(lr, t, T, grad_clip)