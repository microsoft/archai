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
from archai.datasets import data
from archai.common.trainer import Trainer
from archai.nas.vis_model_desc import draw_model_desc
from archai.common.checkpoint import CheckPoint
from archai.common.ml_utils import set_optim_lr

import archai.algos.zero_cost_measures.pruners.predictive as predictive


class ZeroCostTrainer(ArchTrainer, EnforceOverrides):

    @overrides
    def fit(self, data_loaders:data.DataLoaders)->Metrics:
        logger.pushd(self._title)

        self._metrics = Metrics(self._title, self._apex, logger_freq=self._logger_freq)

        # TODO: Move to conf/obtain from the right place
        dataload:str = 'random' # 'random or grasp supported'
        dataload_info = 1 # 'number of batches to use for random dataload or number of samples per class for grasp dataload'
        num_classes = 10 # we have to get these from somewhere

        measures = predictive.find_measures(self.model, 
                                            data_loaders.train_dl, 
                                            (dataload, dataload_info, num_classes),
                                            self.get_device())
        
        for k, v in measures:
            logger.info({k:v})

        logger.popd()