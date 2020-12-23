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

        # Need a large batch size for getting good estimate of correlations 
        # amongst the dataset
        conf_loader['train_batch'] = conf_loader['naswotrain']['train_batch']

        # get data
        train_dl, test_dl = self.get_data(conf_loader)

        trainer = NaswotrainTrainer(conf_train, model, checkpoint)
        train_metrics = trainer.fit(train_dl, test_dl)
        return train_metrics


    @overrides
    def create_model(self, conf_eval: Config, model_desc_builder: ModelDescBuilder, 
                    final_desc_filename=None, full_desc_filename=None) -> nn.Module:

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

        # NOTE: Changing the number of classes to 1
        # to make the function scalar valued as it is necessary
        # for the score function to conform to the paper 
        model_desc.logits_op.params['n_classes'] = 1

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


