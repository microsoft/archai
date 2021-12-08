# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Tuple, Optional, Type

import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from overrides import EnforceOverrides, overrides

from archai.common.metrics import Metrics
from archai.common.tester import Tester
from archai.common.config import Config
from archai.common import utils, ml_utils
from archai.common.common import logger
from archai.datasets import data
from archai.common.checkpoint import CheckPoint
from archai.common.apex_utils import ApexUtils
from archai.common.multi_optim import MultiOptim, OptimSched
from archai.nas.nas_utils import get_model_stats
from archai.nas.arch_trainer import ArchTrainer
from archai.datasets.providers.darcyflow_provider import UnitGaussianNormalizer

TDarcyflowTrainer = Optional[Type['DarcyflowTrainer']]


class DarcyflowTrainer(ArchTrainer, EnforceOverrides):
    def __init__(self, conf_train: Config, model: nn.Module,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)

        # region config vars specific to Darcyflow trainer
        
        # endregion

    @overrides
    def _train_epoch(self, train_dl: DataLoader)->None:
        steps = len(train_dl)
        self.model.train()

        logger.pushd('steps')
        for step, (x, y) in enumerate(train_dl):
            logger.pushd(step)
            assert self.model.training # derived class might alter the mode

            # TODO: please check that no algorithm is invalidated by swapping prestep with zero grad
            self._multi_optim.zero_grad()

            self.pre_step(x, y)

            # divide batch in to chunks if needed so it fits in GPU RAM
            if self.batch_chunks > 1:
                x_chunks, y_chunks = torch.chunk(x, self.batch_chunks), torch.chunk(y, self.batch_chunks)
            else:
                x_chunks, y_chunks = (x,), (y,)

            logits_chunks = []
            loss_sum, loss_count = 0.0, 0
            for xc, yc in zip(x_chunks, y_chunks):
                xc, yc = xc.to(self.get_device(), non_blocking=True), yc.to(self.get_device(), non_blocking=True)

                logits_c, aux_logits = self.model(xc), None
                tupled_out = isinstance(logits_c, Tuple) and len(logits_c) >=2
                if tupled_out: # then we are using model created by desc
                    logits_c, aux_logits = logits_c[0], logits_c[1]

                # darcyflow specific line
                logits_c = logits_c.squeeze()
                # WARNING, DEBUG: Making code run through for now
                # this is missing all the y's decoding

                loss_c = self.compute_loss(self._lossfn, yc, logits_c,
                                        self._aux_weight, aux_logits)

                self._apex.backward(loss_c, self._multi_optim)

                loss_sum += loss_c.item() * len(logits_c)
                loss_count += len(logits_c)
                logits_chunks.append(logits_c.detach().cpu())

            # TODO: original darts clips alphas as well but pt.darts doesn't
            self._apex.clip_grad(self._grad_clip, self.model, self._multi_optim)

            self._multi_optim.step()

            # TODO: we possibly need to sync so all replicas are upto date
            self._apex.sync_devices()

            self.post_step(x, y,
                           ml_utils.join_chunks(logits_chunks),
                           torch.tensor(loss_sum/loss_count),
                           steps)
            logger.popd()

            # end of step

        self._multi_optim.epoch()
        logger.popd()

        
