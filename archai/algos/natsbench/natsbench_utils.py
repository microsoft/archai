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



def model_from_natsbench_tss(arch_index:int, dataset_name:str, natsbench_location:str)->Model:

        # create natsbench api
        api = create(natsbench_location, 'tss', fast_mode=True, verbose=True)

        if arch_index > 15625 or arch_index < 0:
            logger.warn(f'architecture id {arch_index} is invalid ')

        supported_datasets = {'cifar10', 'cifar100', 'ImageNet16-120'}
        
        if dataset_name not in supported_datasets:
            raise NotImplementedError

        config = api.get_net_config(arch_index, dataset_name)
        
        model = get_cell_based_tiny_net(config)

        return model