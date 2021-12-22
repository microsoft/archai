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
from archai.common.common import get_conf
from archai.common.config import Config
from archai.common import common, utils
from archai.common.common import logger
from archai.nas.model import Model
from archai.nas.model_desc import ModelDesc
from archai.nas.arch_trainer import ArchTrainer
from archai.datasets import data
from archai.common.trainer import Trainer
from archai.nas.vis_model_desc import draw_model_desc
from archai.common.checkpoint import CheckPoint
from archai.common.ml_utils import get_lossfn, set_optim_lr

import archai.algos.zero_cost_measures.pruners.predictive as predictive


class ZeroCostTrainer(ArchTrainer, EnforceOverrides):

    @overrides
    def fit(self, data_loaders:data.DataLoaders, num_classes:int)->Metrics:
        logger.pushd(self._title)

        self._metrics = Metrics(self._title, self._apex, logger_freq=self._logger_freq)

        # TODO: Move to conf/obtain from the right place
        dataload:str = 'random' # 'random or grasp supported'
        dataload_info = 1 # 'number of batches to use for random dataload or number of samples per class for grasp dataload'

        # get the loss_fn needed. datasets like darcyflow 
        # can have different loss function.
        conf = get_conf()
        conf_lossfn = conf['nas']['eval']['trainer']['lossfn']
        loss_fn = get_lossfn(conf_lossfn=conf_lossfn)
    
        measures = predictive.find_measures(self.model, 
                                            data_loaders.train_dl, 
                                            (dataload, dataload_info, num_classes),
                                            self.get_device(), 
                                            loss_fn=loss_fn)
        
        for k, v in measures.items():
            logger.info({k:v})

        logger.popd()

        return self._metrics