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
from archai.algos.random_sample_darts_space.constant_darts_space_sampler import ConstantDartsSpaceSampler
from archai.algos.random_sample_darts_space.random_model_desc_builder import RandomModelDescBuilder

from .zero_cost_trainer import ZeroCostTrainer

class ZeroCostDartsSpaceConstantRandomEvaluator(Evaluater):
    @overrides
    def create_model(self, conf_eval:Config, model_desc_builder:RandomModelDescBuilder,
                      final_desc_filename=None, full_desc_filename=None)->nn.Module:
        
        assert model_desc_builder is not None, 'ZeroCostDartsSpaceConstantRandomEvaluator requires model_desc_builder'
        assert final_desc_filename is None, 'ZeroCostDartsSpaceConstantRandomEvaluator creates its own model desc based on arch index'
        assert type(model_desc_builder) == RandomModelDescBuilder, 'DartsSpaceEvaluater requires RandomModelDescBuilder'

        # region conf vars
        # if not explicitly passed in then get from conf
        if not final_desc_filename:
            final_desc_filename = conf_eval['final_desc_filename']
            full_desc_filename = conf_eval['full_desc_filename']
        conf_model_desc   = conf_eval['model_desc']
        arch_index = conf_eval['dartsspace']['arch_index']
        self.num_classes = conf_eval['loader']['dataset']['n_classes']
        # endregion

        assert arch_index >= 0

        # get random seed from constant sampler
        # to get deterministic model creation
        self.constant_sampler = ConstantDartsSpaceSampler()
        random_seed_for_model_construction = self.constant_sampler.get_archid(arch_index)

        # we don't load template model desc file from disk
        # as we are creating model based on arch_index
        model_desc = model_desc_builder.build(conf_model_desc, 
                                            seed=random_seed_for_model_construction)

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

    @overrides
    def train_model(self,  conf_train:Config, model:nn.Module,
                    checkpoint:Optional[CheckPoint])->Metrics:
        conf_loader = conf_train['loader']
        conf_train = conf_train['trainer']
        
        # get data
        data_loaders = self.get_data(conf_loader)

        trainer = ZeroCostTrainer(conf_train, model, checkpoint)
        train_metrics = trainer.fit(data_loaders, self.num_classes)
        return train_metrics

