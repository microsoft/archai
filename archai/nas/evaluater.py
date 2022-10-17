# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Optional, Tuple
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


class EvalResult:
    def __init__(self, train_metrics:Metrics) -> None:
        self.train_metrics = train_metrics

class Evaluater(EnforceOverrides):
    def evaluate(self, conf_eval:Config, model_desc_builder:ModelDescBuilder)->EvalResult:
        logger.pushd('eval_arch')

        # region conf vars
        conf_checkpoint = conf_eval['checkpoint']
        resume = conf_eval['resume']

        model_filename    = conf_eval['model_filename']
        metric_filename    = conf_eval['metric_filename']
        # endregion

        model = self.create_model(conf_eval, model_desc_builder)

        checkpoint = nas_utils.create_checkpoint(conf_checkpoint, resume)
        train_metrics = self.train_model(conf_eval, model, checkpoint)
        train_metrics.save(metric_filename)

        # save model
        if model_filename:
            model_filename = utils.full_path(model_filename)
            ml_utils.save_model(model, model_filename)

        logger.info({'model_save_path': model_filename})

        logger.popd()

        return EvalResult(train_metrics)

    def train_model(self, conf_train:Config, model:nn.Module,
                    checkpoint:Optional[CheckPoint])->Metrics:
        conf_loader = conf_train['loader']
        conf_train = conf_train['trainer']

        # get data
        data_loaders = self.get_data(conf_loader)

        trainer = Trainer(conf_train, model, checkpoint)
        train_metrics = trainer.fit(data_loaders)
        return train_metrics

    def get_data(self, conf_loader:Config)->data.DataLoaders:

        # this dict caches the dataset objects per dataset config so we don't have to reload
        # the reason we do dynamic attribute is so that any dependent methods
        # can do ray.remote
        if not hasattr(self, '_data_cache'):
            self._data_cache:Dict[int, data.DataLoaders] = {}

        # first get from cache
        if id(conf_loader) in self._data_cache:
            data_loaders = self._data_cache[id(conf_loader)]
        else:
            data_loaders = data.get_data(conf_loader)
            self._data_cache[id(conf_loader)] = data_loaders

        return data_loaders

    def _default_module_name(self, dataset_name:str, function_name:str)->str:
        """Select PyTorch pre-defined network to support manual mode"""
        module_name = ''
        # TODO: below detection code is too week, need to improve, possibly encode image size in yaml and use that instead
        if dataset_name.startswith('cifar'):
            if function_name.startswith('res'): # support resnext as well
                module_name = 'archai.cifar10_models.resnet'
            elif function_name.startswith('dense'):
                module_name = 'archai.cifar10_models.densenet'
        elif dataset_name.startswith('imagenet') or dataset_name.startswith('sport8'):
            module_name = 'torchvision.models'
        if not module_name:
            raise NotImplementedError(f'Cannot get default module for {function_name} and dataset {dataset_name} because it is not supported yet')
        return module_name

    def create_model(self, conf_eval:Config, model_desc_builder:ModelDescBuilder,
                      final_desc_filename=None, full_desc_filename=None)->nn.Module:

        assert model_desc_builder is not None, 'Default evaluater requires model_desc_builder'

        # region conf vars
        # if explicitly passed in then don't get from conf
        if not final_desc_filename:
            final_desc_filename = conf_eval['final_desc_filename']
            full_desc_filename = conf_eval['full_desc_filename']
        conf_model_desc   = conf_eval['model_desc']
        # endregion

        # load model desc file to get template model
        template_model_desc = ModelDesc.load(final_desc_filename)
        model_desc = model_desc_builder.build(conf_model_desc,
                                            template=template_model_desc)

        # save desc for reference
        model_desc.save(full_desc_filename)

        model = self.model_from_desc(model_desc)

        logger.info({'model_factory':False,
                    'cells_len':len(model.desc.cell_descs()),
                    'init_node_ch': conf_model_desc['model_stems']['init_node_ch'],
                    'n_cells': conf_model_desc['n_cells'],
                    'n_reductions': conf_model_desc['n_reductions'],
                    'n_nodes': conf_model_desc['cell']['n_nodes']})

        return model

    def model_from_desc(self, model_desc)->Model:
        return Model(model_desc, droppath=True, affine=True)
