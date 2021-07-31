# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Optional, Callable, Type, List
import os

import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides, EnforceOverrides

from archai.common.config import Config
from archai.common import common, utils
from archai.common.common import logger
from archai.nas.model import Model
from archai.nas.model_desc import ModelDesc
from archai.nas.arch_trainer import ArchTrainer
from archai.common.trainer import Trainer
from archai.nas.vis_model_desc import draw_model_desc
from archai.common.checkpoint import CheckPoint
from archai.common.ml_utils import set_optim_lr
from archai.datasets import data
from archai.nas.nas_utils import get_model_stats

TFreezeRatioTrainer = Optional[Type['FreezeRatioTrainer']]


class FreezeRatioTrainer(ArchTrainer, EnforceOverrides):
    def __init__(self, conf_train: Config, model: nn.Module,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)


    @overrides
    def pre_fit(self, data_loaders:data.DataLoaders) -> None:
        super().pre_fit(data_loaders)

        train_dl = data_loaders.train_dl
        assert train_dl is not None

        # compute model stats per minibatch of training data
        data_iterator = iter(train_dl)
        x, target = next(data_iterator)
        x_shape = list(x.shape)
        x_shape[0] = 1 # to prevent overflow errors with large batch size we will use a batch size of 1
        model_stats = get_model_stats(self.model, input_tensor_shape=x_shape, clone_model=True)

        # get number of parameters per 
        layer_basenames = self.conf_train['layer_basenames']
        layer_numparams = [0] * len(layer_basenames)
        for l in model_stats.layer_stats:
            for lb, j in enumerate(layer_basenames):
                if lb in l.name:
                    layer_numparams[j] += l.parameters

        assert len(layer_basenames) == len(layer_numparams)

        # freeze up to the layer which satisfies the ratio
        num_params_unfreeze = self.conf_train['desired_ratio_unfreeze'] * model_stats.parameters
        identifiers_to_unfreeze = []
        runsum = 0
        for i in range(len(layer_basenames)-1, -1, -1):
            runsum += layer_numparams[i]
            identifiers_to_unfreeze.append(layer_basenames[i])
            if runsum >= num_params_unfreeze:
                break
                
        # print diagnostics
        logger.info(f'layers unfrozen {identifiers_to_unfreeze}')
        logger.info(dict({'achieved_ratio_unfreeze': runsum/model_stats.parameters}))

        # freeze everything other than the last layer
        self._freeze_but_last_layer(identifiers_to_unfreeze)


    def _freeze_but_last_layer(self, identifiers_to_unfreeze:List[str]) -> None:

        # Do it via parameters
        for param in self.model.parameters():
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            for identifier in identifiers_to_unfreeze:
                if identifier in name:
                    param.requires_grad = True

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(f'{name} requires grad')