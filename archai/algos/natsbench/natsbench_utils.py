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

from nats_bench import create
from nats_bench.api_size import NATSsize
from nats_bench.api_topology import NATStopology
from archai.algos.natsbench.lib.models import get_cell_based_tiny_net


def create_natsbench_tss_api(natsbench_location:str)->NATStopology:
    # create natsbench api
    api = create(natsbench_location, 'tss', fast_mode=True, verbose=True)
    return api

def model_from_natsbench_tss(arch_index:int, dataset_name:str, api:NATStopology)->Model:
    if arch_index >= 15625 or arch_index < 0:
        logger.warn(f'architecture id {arch_index} is invalid ')

    supported_datasets = {'cifar10', 'cifar100', 'ImageNet16-120'}
    
    if dataset_name not in supported_datasets:
        raise NotImplementedError

    config = api.get_net_config(arch_index, dataset_name)
    
    model = get_cell_based_tiny_net(config)

    return model