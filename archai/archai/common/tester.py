# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Optional

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from overrides import EnforceOverrides

from .metrics import Metrics
from .config import Config
from . import utils, ml_utils
from .common import logger
from archai.common.apex_utils import ApexUtils

class Tester(EnforceOverrides):
    def __init__(self, conf_val:Config, model:nn.Module, apex:ApexUtils)->None:
        self._title = conf_val['title']
        self._logger_freq = conf_val['logger_freq']
        conf_lossfn = conf_val['lossfn']
        self.batch_chunks = conf_val['batch_chunks']

        self._apex = apex
        self.model = model
        self._lossfn = ml_utils.get_lossfn(conf_lossfn).to(apex.device)
        self._metrics = None

    def test(self, test_dl: DataLoader)->Metrics:
        logger.pushd(self._title)

        self._metrics = self._create_metrics()

        # recreate metrics for this run
        self._pre_test()
        self._test_epoch(test_dl)
        self._post_test()

        logger.popd()
        return self.get_metrics() # type: ignore

    def _test_epoch(self, test_dl: DataLoader)->None:
        self._metrics.pre_epoch()
        self.model.eval()
        steps = len(test_dl)

        with torch.no_grad(), logger.pushd('steps'):
            for step, (x, y) in enumerate(test_dl):
                # derived class might alter the mode through pre/post hooks
                assert not self.model.training
                logger.pushd(step)

                self._pre_step(x, y, self._metrics)

                # divide batch in to chunks if needed so it fits in GPU RAM
                if self.batch_chunks > 1:
                    x_chunks, y_chunks = torch.chunk(x, self.batch_chunks), torch.chunk(y, self.batch_chunks)
                else:
                    x_chunks, y_chunks = (x,), (y,)

                logits_chunks = []
                loss_sum, loss_count = 0.0, 0
                for xc, yc in zip(x_chunks, y_chunks):
                    xc, yc = xc.to(self.get_device(), non_blocking=True), yc.to(self.get_device(), non_blocking=True)

                    logits_c = self.model(xc)
                    tupled_out = isinstance(logits_c, Tuple) and len(logits_c) >=2
                    if tupled_out:
                        logits_c = logits_c[0]
                    loss_c = self._lossfn(logits_c, yc)

                    loss_sum += loss_c.item() * len(logits_c)
                    loss_count += len(logits_c)
                    logits_chunks.append(logits_c.detach().cpu())

                self._post_step(x, y,
                                ml_utils.join_chunks(logits_chunks),
                                torch.tensor(loss_sum/loss_count),
                                steps, self._metrics)

                # TODO: we possibly need to sync so all replicas are upto date
                self._apex.sync_devices()

                logger.popd()
        self._metrics.post_epoch() # no "val" dataset for the test phase

    def get_metrics(self)->Optional[Metrics]:
        return self._metrics

    def state_dict(self)->dict:
        return {
            'metrics': self._metrics.state_dict()
        }

    def get_device(self):
        return self._apex.device

    def load_state_dict(self, state_dict:dict)->None:
        self._metrics.load_state_dict(state_dict['metrics'])

    def _pre_test(self)->None:
        self._metrics.pre_run()

    def _post_test(self)->None:
        self._metrics.post_run()

    def _pre_step(self, x:Tensor, y:Tensor, metrics:Metrics)->None:
        metrics.pre_step(x, y)

    def _post_step(self, x:Tensor, y:Tensor, logits:Tensor, loss:Tensor,
                  steps:int, metrics:Metrics)->None:
        metrics.post_step(x, y, logits, loss, steps)

    def _create_metrics(self)->Metrics:
        return Metrics(self._title, self._apex, logger_freq=self._logger_freq)

