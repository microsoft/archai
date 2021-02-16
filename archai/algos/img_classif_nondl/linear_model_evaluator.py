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
from archai.nas import nas_utils
from archai.common import ml_utils, utils
from archai.common.metrics import EpochMetrics, Metrics
from archai.nas.model import Model
from archai.common.checkpoint import CheckPoint
from archai.nas.evaluater import Evaluater
from archai.algos.proxynas.freeze_trainer import FreezeTrainer
from archai.algos.proxynas.conditional_trainer import ConditionalTrainer

from nats_bench import create
from archai.algos.natsbench.lib.models import get_cell_based_tiny_net

class LinearEvaluater(Evaluater):
    @overrides
    def create_model(self, conf_eval:Config, model_desc_builder:ModelDescBuilder,
                      final_desc_filename=None, full_desc_filename=None)->nn.Module:
        # region conf vars
        dataset_name = conf_eval['loader']['dataset']['name']

        # if explicitly passed in then don't get from conf
        if not final_desc_filename:
            final_desc_filename = conf_eval['final_desc_filename']

        dataroot = utils.full_path(conf_eval['loader']['dataset']['dataroot'])    
        # endregion

        # create linear model

        # problem: what is the input size of the image features?


        return self._model_from_natsbench(arch_index, dataset_name, natsbench_location)
