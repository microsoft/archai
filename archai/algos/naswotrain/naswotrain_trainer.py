# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Callable, Tuple, Type
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides, EnforceOverrides

from archai.common.metrics import Metrics
from archai.common.config import Config
from archai.common import common, utils
from archai.common.common import logger
from archai.nas.model import Model
from archai.nas.model_desc import ModelDesc
from archai.nas.arch_trainer import ArchTrainer
from archai.common.trainer import Trainer
from archai.nas.vis_model_desc import draw_model_desc
from archai.common.checkpoint import CheckPoint
from archai.common.ml_utils import set_optim_lr

from .naswotrain_metrics import NaswoTrainMetrics

TNaswotrainTrainer = Optional[Type['NaswotrainTrainer']]


class NaswotrainTrainer(ArchTrainer, EnforceOverrides):

    @overrides
    def fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->Metrics:
        logger.pushd(self._title)

        self._metrics = NaswoTrainMetrics(self._title, self._apex, logger_freq=self._logger_freq)     

        # create optimizers and schedulers (we don't need it only to make to_amp call pass)
        self._multi_optim = self.create_multi_optim(len(train_dl))

        # before checkpoint restore, convert to amp
        self.model = self._apex.to_amp(self.model, self._multi_optim,
                                       batch_size=train_dl.batch_size)

        # score the model with one minibatch of data 
        # as in the paper "Neural Architecture Search without Training", Mellor et al. 2020
        # modified from https://github.com/BayesWatch/nas-without-training/blob/master/search.py
        self.model.train()
        data_iterator = iter(train_dl)
        x, target = next(data_iterator)
        x, target = x.to(self.get_device()), target.to(self.get_device())

        jacobs = self._get_batch_jacobian(x)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
        score = self._eval_score(jacobs)
        self._metrics.naswotraining_score = score
        logger.info(f'nas without training score: {score} using batch size: {train_dl.batch_size}')
        logger.info({'naswithouttraining':float(score)})
        logger.info({'naswithouttraining_batch_size':train_dl.batch_size})

        # make sure we don't keep references to the graph
        del self._multi_optim

        logger.popd()
        return self.get_metrics()


    def _get_batch_jacobian(self, x):
        ''' Modified from https://github.com/BayesWatch/nas-without-training/blob/master/search.py '''
        self.model.zero_grad()
        x.requires_grad_(True)
        logits = self.model(x)
        # Manual models only return logits, 
        # whereas DARTS space models return logits, aux_logits
        if isinstance(logits, tuple):
            logits = logits[0]
        logits.backward(torch.ones_like(logits))
        jacob = x.grad.detach()
        return jacob


    def _eval_score(self, jacob):
        ''' Modified from https://github.com/BayesWatch/nas-without-training/blob/master/search.py '''
        corrs = np.corrcoef(jacob)
        v, _  = np.linalg.eig(corrs)
        k = 1e-5
        return -np.sum(np.log(v + k) + 1./(v + k))