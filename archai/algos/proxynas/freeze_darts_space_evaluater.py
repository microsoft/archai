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
from archai.algos.random_sample_darts_space.constant_darts_space_sampler import ConstantDartsSpaceSampler
from archai.algos.random_sample_darts_space.random_model_desc_builder import RandomModelDescBuilder

class FreezeDartsSpaceEvaluater(Evaluater):
    @overrides
    def create_model(self, conf_eval:Config, model_desc_builder:RandomModelDescBuilder,
                      final_desc_filename=None, full_desc_filename=None)->nn.Module:
        
        assert model_desc_builder is not None, 'FreezeDartsSpaceEvaluater requires model_desc_builder'
        assert final_desc_filename is None, 'FreezeDartsSpaceEvaluater creates its own model desc based on arch index'
        assert type(model_desc_builder) == RandomModelDescBuilder, 'FreezeDartsSpaceEvaluater requires RandomModelDescBuilder'

        # region conf vars
        # if not explicitly passed in then get from conf
        if not final_desc_filename:
            final_desc_filename = conf_eval['final_desc_filename']
            full_desc_filename = conf_eval['full_desc_filename']
        conf_model_desc   = conf_eval['model_desc']
        arch_index = conf_eval['dartsspace']['arch_index']
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
        trainer = FreezeTrainer(conf_train_freeze, model, checkpoint)
        freeze_train_metrics = trainer.fit(data_loaders)
        logger.popd()

        return freeze_train_metrics