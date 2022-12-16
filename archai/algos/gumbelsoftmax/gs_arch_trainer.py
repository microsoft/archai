# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Mapping, Optional, Union
import copy

import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn, autograd
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F

from overrides import overrides

from archai.common.config import Config
from archai.nas.arch_trainer import ArchTrainer
from archai.common import utils, ml_utils
from archai.nas.model import Model
from archai.common.checkpoint import CheckPoint
from archai.common.common import logger, get_conf
from archai.algos.gumbelsoftmax.gs_op import GsOp

class GsArchTrainer(ArchTrainer):
    def __init__(self, conf_train: Config, model: nn.Module, checkpoint: Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)

        conf = get_conf()
        self._gs_num_sample = conf['nas']['search']['model_desc']['cell']['gs']['num_sample']

    @overrides
    def create_optimizer(self, conf_optim:Config, params) -> Optimizer:
        # in this case we don't need to differentiate between arch_params and weights
        # as the same optimizer will update both
        arch_params = list(self.model.all_owned().param_by_kind('alphas'))
        nonarch_params = list(self.model.nonarch_params(recurse=True))
        # TODO: do we need different param groups? Check in paper if they are using different optimizers for alphas or not.
        param_groups = [{'params': nonarch_params}, {'params': arch_params}]
        return ml_utils.create_optimizer(conf_optim, param_groups)


    @overrides
    def pre_step(self, x:Tensor, y:Tensor)->None:
        super().pre_step(x, y)

        # TODO: is it a good idea to ensure model is in training mode here?

        # for each node in a cell, get the alphas of each incoming edge
        # concatenate them all together, sample from them via GS
        # push the resulting weights to the corresponding edge ops
        # for use in their respective forward

        for _, cell in enumerate(self.model.cells):
            for _, node in enumerate(cell.dag):
                # collect all alphas for all edges in to node
                node_alphas = []
                for edge in node:
                    if hasattr(edge._op, 'PRIMITIVES') and type(edge._op) == GsOp:
                        node_alphas.extend(alpha for op, alpha in edge._op.ops())

                # TODO: will creating a tensor from a list of tensors preserve the graph?
                node_alphas = torch.Tensor(node_alphas)

                if node_alphas.nelement() > 0:
                    # sample ops via gumbel softmax
                    sample_storage = []
                    for _ in range(self._gs_num_sample):
                        sampled = F.gumbel_softmax(node_alphas, tau=1, hard=False, eps=1e-10, dim=-1)
                        sample_storage.append(sampled)

                    samples_summed = torch.sum(torch.stack(sample_storage, dim=0), dim=0)
                    samples = samples_summed / torch.sum(samples_summed)

                    # TODO: should we be normalizing the sampled weights?
                    # TODO: do gradients blow up as number of samples increases?

                    # send the sampled op weights to their respective edges
                    # to be used in forward
                    counter = 0
                    for _, edge in enumerate(node):
                        if hasattr(edge._op, 'PRIMITIVES') and type(edge._op) == GsOp:
                            this_edge_sampled_weights = samples[counter:counter+len(edge._op.PRIMITIVES)]
                            edge._op.set_op_sampled_weights(this_edge_sampled_weights)
                            counter += len(edge._op.PRIMITIVES)
