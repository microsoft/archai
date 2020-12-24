# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nas.evaluater import Evaluater
from typing import Optional, Tuple
import importlib
import sys
import string
import os

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from overrides import overrides, EnforceOverrides

from archai.common.trainer import Trainer
from archai.common.config import Config
from archai.common.common import logger
from archai.datasets import data
from archai.nas.model_desc import ModelDesc
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas import nas_utils
from archai.common import ml_utils, utils
from archai.common.metrics import EpochMetrics, Metrics
from archai.nas.model import Model
from archai.common.checkpoint import CheckPoint
from archai.algos.proxynas.conditional_trainer import ConditionalTrainer

from .freeze_trainer import FreezeTrainer

class FreezeEvaluator(Evaluater, EnforceOverrides):
    @overrides
    def train_model(self,  conf_train:Config, model:nn.Module,
                    checkpoint:Optional[CheckPoint])->Metrics:
            
        conf_loader = conf_train['loader']
        conf_train_cond = conf_train['trainer']
        conf_train_freeze = conf_train['freeze_trainer']

        logger.pushd('conditional_training')
        # get data
        train_dl, test_dl = self.get_data(conf_loader)
        # first regular train until certain accuracy is achieved
        cond_trainer = ConditionalTrainer(conf_train_cond, model, checkpoint)
        cond_trainer_metrics = cond_trainer.fit(train_dl, test_dl)
        logger.popd()

        # get data with new batch size for freeze training
        # NOTE: important to create copy and modify as otherwise get_data will return 
        # a cached data loader by hashing the id of conf_loader
        conf_loader_freeze = deepcopy(conf_loader)
        conf_loader_freeze['train_batch'] = conf_loader['freeze_loader']['train_batch']

        logger.pushd('freeze_data')
        train_dl, test_dl = self.get_data(conf_loader_freeze)
        trainer = FreezeTrainer(conf_train_freeze, model, checkpoint)
        freeze_train_metrics = trainer.fit(train_dl, test_dl)
        logger.popd()

        return freeze_train_metrics