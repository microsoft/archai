# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
from archai.nas.evaluater import EvalResult, Evaluater
from archai.algos.proxynas.freeze_trainer import FreezeTrainer

from nats_bench import create
from archai.algos.natsbench.lib.models import get_cell_based_tiny_net

from .zero_cost_trainer import ZeroCostTrainer

class ZeroCostNatsbenchEvaluater(Evaluater):
    @overrides
    def create_model(self, conf_eval:Config, model_desc_builder:ModelDescBuilder,
                      final_desc_filename=None, full_desc_filename=None)->nn.Module:
        # region conf vars
        dataset_name = conf_eval['loader']['dataset']['name']

        # if explicitly passed in then don't get from conf
        if not final_desc_filename:
            final_desc_filename = conf_eval['final_desc_filename']
        arch_index = conf_eval['natsbench']['arch_index']
        dataroot = utils.full_path(conf_eval['loader']['dataset']['dataroot'])
        natsbench_location = os.path.join(dataroot, 'natsbench', conf_eval['natsbench']['natsbench_tss_fast'])
        # endregion

        assert arch_index
        assert natsbench_location

        return self._model_from_natsbench(arch_index, dataset_name, natsbench_location)

    def _model_from_natsbench(self, arch_index:int, dataset_name:str, natsbench_location:str)->Model:

        # create natsbench api
        api = create(natsbench_location, 'tss', fast_mode=True, verbose=True)

        if arch_index > 15625 or arch_index < 0:
            logger.warn(f'architecture id {arch_index} is invalid ')

        if dataset_name not in {'cifar10', 'cifar100', 'ImageNet16-120'}:
            logger.warn(f'dataset {dataset_name} is not part of natsbench')
            raise NotImplementedError()

        config = api.get_net_config(arch_index, dataset_name)
        # network is a nn.Module subclass. the last few modules have names
        # lastact, lastact.0, lastact.1, global_pooling, classifier
        # which we can freeze train as usual
        model = get_cell_based_tiny_net(config)

        return model

    @overrides
    def train_model(self,  conf_train:Config, model:nn.Module,
                    checkpoint:Optional[CheckPoint])->Metrics:
        conf_loader = conf_train['loader']
        
        # get data
        data_loaders = self.get_data(conf_loader)

        trainer = ZeroCostTrainer(conf_train, model, checkpoint)
        train_metrics = trainer.fit(data_loaders)
        return train_metrics

