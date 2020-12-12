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

from .naswotrain_trainer import NaswotrainTrainer


class NaswotrainEvaluator(Evaluater, EnforceOverrides):
    @overrides
    def train_model(self,  conf_train:Config, model:nn.Module,
                    checkpoint:Optional[CheckPoint])->Metrics:
        conf_loader = conf_train['loader']
        conf_train = conf_train['trainer']

        # get data
        train_dl, test_dl = self.get_data(conf_loader)

        trainer = NaswotrainTrainer(conf_train, model, checkpoint)
        train_metrics = trainer.fit(train_dl, test_dl)
        return train_metrics