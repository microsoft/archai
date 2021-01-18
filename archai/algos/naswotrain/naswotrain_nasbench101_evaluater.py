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
from archai.algos.nasbench101.nasbench101_dataset import Nasbench101Dataset

from .naswotrain_trainer import NaswotrainTrainer

class NaswotrainNasbench101Evaluater(Evaluater):
    @overrides
    def create_model(self, conf_eval:Config, model_desc_builder:ModelDescBuilder,
                      final_desc_filename=None, full_desc_filename=None)->nn.Module:
        # region conf vars
        dataset_name = conf_eval['loader']['dataset']['name']

        if dataset_name != 'cifar10':
            logger.warn(f'dataset {dataset_name} is not part of nasbench101')
            raise NotImplementedError()

        # if explicitly passed in then don't get from conf
        if not final_desc_filename:
            final_desc_filename = conf_eval['final_desc_filename']
        arch_index = conf_eval['nasbench101']['arch_index']

        dataroot = utils.full_path(conf_eval['loader']['dataset']['dataroot'])
        nasbench101_location = os.path.join(dataroot, 'nasbench_ds', 'nasbench_only108.tfrecord.pkl')             
        # endregion

        assert arch_index
        assert nasbench101_location

        return self._model_from_nasbench101(arch_index, dataset_name, nasbench101_location)


    def _model_from_nasbench101(self, arch_index:int, dataset_name:str, nasbench101_location:str)->Model:

        # create the nasbench101 api
        nsds = Nasbench101Dataset(nasbench101_location)

        # there are 423624 architectures total
        if arch_index < 0 or arch_index > 423623:
            logger.warn(f'architecture id {arch_index} is invalid')
            raise NotImplementedError()

        model = nsds.create_model(arch_index)
        return model


    @overrides
    def train_model(self,  conf_train:Config, model:nn.Module,
                    checkpoint:Optional[CheckPoint])->Metrics:
        conf_loader = conf_train['loader']
        # change the loader batch size to that desired for computing score
        conf_loader['train_batch'] = conf_loader['naswotrain']['train_batch']
        conf_train = conf_train['trainer']

        # get data
        train_dl, test_dl = self.get_data(conf_loader)

        trainer = NaswotrainTrainer(conf_train, model, checkpoint)
        train_metrics = trainer.fit(train_dl, test_dl)
        return train_metrics

    