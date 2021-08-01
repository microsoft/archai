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
from archai.algos.proxynas.freeze_ratio_trainer import FreezeRatioTrainer
from archai.algos.proxynas.conditional_trainer import ConditionalTrainer

from nats_bench import create
from archai.algos.natsbench.lib.models import get_cell_based_tiny_net

class FreezeNatsbenchSSSEvaluater(Evaluater):
    @overrides
    def create_model(self, conf_eval:Config, model_desc_builder:ModelDescBuilder,
                      final_desc_filename=None, full_desc_filename=None)->nn.Module:
        # region conf vars
        dataset_name = conf_eval['loader']['dataset']['name']
        self.num_classes = conf_eval['loader']['dataset']['n_classes']

        # if explicitly passed in then don't get from conf
        if not final_desc_filename:
            final_desc_filename = conf_eval['final_desc_filename']
        arch_index = conf_eval['natsbench']['arch_index']

        dataroot = utils.full_path(conf_eval['loader']['dataset']['dataroot'])
        natsbench_location = os.path.join(dataroot, 'natsbench', conf_eval['natsbench']['natsbench_sss_fast'])
        # endregion

        assert natsbench_location

        return self._model_from_natsbench(arch_index, dataset_name, natsbench_location)

    def _model_from_natsbench(self, arch_index:int, dataset_name:str, natsbench_location:str)->Model:

        # create natsbench api
        api = create(natsbench_location, 'sss', fast_mode=True, verbose=True)

        if arch_index > 32767 or arch_index < 0:
            logger.warn(f'architecture id {arch_index} is invalid ')

        supported_datasets = {'cifar10', 'cifar100', 'ImageNet16-120'}
        
        # force natsbench to use cifar10 archs
        # since it doesn't know about other datasets
        if dataset_name not in supported_datasets:
            dataset_name = 'cifar10'

        config = api.get_net_config(arch_index, dataset_name)

        # fill in the num classes from conf again
        # to account for unsupported datasets may 
        # have number of classes different from cifar10
        config['num_classes'] = self.num_classes
        
        # network is a nn.Module subclass. the last few modules have names
        # lastact, lastact.0, lastact.1, global_pooling, classifier 
        # which we can freeze train as usual
        model = get_cell_based_tiny_net(config)

        return model

    @overrides
    def train_model(self, conf_train:Config, model:nn.Module,
                    checkpoint:Optional[CheckPoint])->Metrics:
        conf_loader = conf_train['loader']
        conf_train_cond = conf_train['trainer']
        conf_train_freeze = conf_train['freeze_trainer']

        # NOTE: we don't pass checkpoint to the trainers
        # as it creates complications and we don't need it
        # as these trainers are quite fast
        checkpoint = None

        logger.pushd('conditional_training')
        data_loaders = self.get_data(conf_loader)
        # first regular train until certain accuracy is achieved
        cond_trainer = ConditionalTrainer(conf_train_cond, model, checkpoint, None)
        cond_trainer_metrics = cond_trainer.fit(data_loaders)
        logger.popd()

        # get data with new batch size for freeze training
        # NOTE: important to create copy and modify as otherwise get_data will return
        # a cached data loader by hashing the id of conf_loader
        conf_loader_freeze = deepcopy(conf_loader)
        conf_loader_freeze['train_batch'] = conf_loader['freeze_loader']['train_batch']

        logger.pushd('freeze_training')
        data_loaders = self.get_data(conf_loader_freeze)
        # now just finetune the last few layers
        checkpoint = None

        if conf_train_freeze['use_ratio']:
            trainer = FreezeRatioTrainer(conf_train_freeze, model, checkpoint)
        else:
            trainer = FreezeTrainer(conf_train_freeze, model, checkpoint)
        freeze_train_metrics = trainer.fit(data_loaders)
        logger.popd()

        return freeze_train_metrics