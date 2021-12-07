# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Optional
import importlib
import sys
import string
import os

from overrides import overrides

import torch
from torch import nn

from overrides import overrides, EnforceOverrides

from archai.common.trainer import Trainer
from archai.common.config import Config
from archai.common.common import logger
from archai.datasets import data
from archai.nas.model_desc import ModelDesc
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.nas_utils import create_nb301_genotype_from_desc
from archai.nas import nas_utils
from archai.common import ml_utils, utils
from archai.common.metrics import EpochMetrics, Metrics
from archai.nas.model import Model
from archai.common.checkpoint import CheckPoint
from archai.nas.evaluater import Evaluater
from archai.algos.proxynas.freeze_trainer import FreezeTrainer
from archai.algos.proxynas.conditional_trainer import ConditionalTrainer
from archai.algos.random_sample_darts_space.constant_darts_space_sampler import ConstantDartsSpaceSampler
from archai.algos.random_sample_darts_space.random_model_desc_builder import RandomModelDescBuilder
from archai.algos.random_sample_darts_space.darts_space_evaluater import DartsSpaceEvaluater
from archai.nas.darcyflow_trainer import DarcyflowTrainer

class DartsSpaceDarcyflowEvaluater(DartsSpaceEvaluater):

    @overrides 
    def train_model(self, conf_train:Config, model:nn.Module,
                    checkpoint:Optional[CheckPoint])->Metrics:
        conf_loader = conf_train['loader']
        conf_train = conf_train['trainer']

        # only darcyflow works with this evaluater    
        if conf_loader['dataset']['name'] != 'darcyflow':
            raise TypeError

        # get data
        data_loaders = self.get_data(conf_loader)

        # the trainer class is the only difference
        trainer = DarcyflowTrainer(conf_train, model, checkpoint)
        train_metrics = trainer.fit(data_loaders)
        return train_metrics
    