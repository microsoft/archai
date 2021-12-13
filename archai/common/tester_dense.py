from typing import Tuple, Optional
from overrides import EnforceOverrides, overrides

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from .config import Config
from . import utils, ml_utils
from .common import logger
from archai.common.apex_utils import ApexUtils
from .tester import Tester
from .metrics_dense import MetricsDense


class TesterDense(Tester, EnforceOverrides):

    def __init__(self, conf_val: Config, model: nn.Module, apex: ApexUtils) -> None:
        super().__init__(conf_val, model, apex)
    

    @overrides
    def _create_metrics(self) -> MetricsDense:
        return MetricsDense(self._title, self._apex, logger_freq=self._logger_freq)

    @overrides
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

                    # darcyflow specific line
                    logits_c = logits_c.squeeze()
                    # WARNING, DEBUG: Making code run through for now
                    # this is missing all the y's decoding

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
