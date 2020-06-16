# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import defaultdict
from typing import Iterable, Optional, Tuple, List
import copy
import math as ma
from itertools import count
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from overrides import overrides

from archai.nas.model_desc import OpDesc
from archai.nas.operations import Op
from archai.nas.arch_params import ArchParams
from archai.common.utils import zip_eq
from archai.common.common import get_conf
from archai.common.common import get_expdir


# TODO: reduction cell might have output reduced by 2^1=2X due to
#   stride 2 through input nodes however FactorizedReduce does only
#   4X reduction. Is this correct?


class XnasOp(Op):
    """The output of XnasOp is weighted output of all allowed primitives.
    """

    PRIMITIVES = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',  # identity
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'none'  # this must be at the end so top1 doesn't chose it
    ]

    def __init__(self, op_desc:OpDesc, arch_params:Optional[ArchParams],
                 affine:bool):
        super().__init__()

        # assume last PRIMITIVE is 'none'
        assert XnasOp.PRIMITIVES[-1] == 'none'
        
        self._ops = nn.ModuleList()
        for primitive in XnasOp.PRIMITIVES:
            op = Op.create(
                OpDesc(primitive, op_desc.params, in_len=1, trainables=None),
                affine=affine, arch_params=None)
            self._ops.append(op)

        # for getting gradients to non-leaf node
        self._grad = None

        # we do this at the end so that we can capture all arch params registered by
        # any previous child modules
        self._setup_arch_params(arch_params)

    def update_alphas(self, eta:float, current_t:int, total_t:int, grad_clip:float):       
        grad_flat = torch.flatten(self._grad)        
        rewards = torch.tensor([-torch.dot(grad_flat, torch.flatten(activ)) for activ in self._activs])
        exprewards = torch.exp(eta * rewards).cuda()
        # NOTE: Will this remain registered?
        self._alphas[0] = torch.mul(self._alphas[0], exprewards)

        # weak learner eviction
        conf = get_conf()
        to_evict = conf['nas']['search']['xnas']['to_evict']
        if to_evict:
            theta = max(self._alphas[0]) * ma.exp(-2 * eta * grad_clip * (total_t - current_t))
            assert len(self._ops) == self._alphas[0].shape[0]
            to_keep_mask = self._alphas[0] >= theta
            num_ops_kept = torch.sum(to_keep_mask).item()
            assert num_ops_kept > 0
            # zero out the weights which are evicted
            self._alphas[0] = torch.mul(self._alphas[0], to_keep_mask)

        # save some debugging info
        expdir = get_expdir()
        filename = os.path.join(expdir, str(id(self)) + '.txt')

        # save debug info to file
        alphas = [str(self._alphas[0][i].item()) for i in range(self._alphas[0].shape[0])]
        with open(filename, 'a') as f:
            f.write(str(alphas))
            f.write('\n')


        

    def _save_grad(self):
        def hook(grad):
            self._grad = copy.deepcopy(grad)
        return hook

    @overrides
    def forward(self, x):
        self._activs = [op(x) for op in self._ops]
        numer = sum(w * activ for w, activ in zip_eq(self._alphas[0], self._activs))
        denom = sum(self._alphas[0])
        self.pt = torch.div(numer, denom)

        # register hook to save gradients 
        # NOTE: it has to be done every forward call
        # otherwise the hook doesn't remain registered
        # for subsequent loss.backward calls
        if self.training:
            self.pt.register_hook(self._save_grad())

        return self.pt

    @overrides
    def finalize(self) -> Tuple[OpDesc, Optional[float]]:
        with torch.no_grad():
            # select except 'none' op
            val, i = torch.topk(self._alphas[0][:-1], 1)
            desc, _ = self._ops[i].finalize()
            return desc, float(val.item())

    @overrides
    def can_drop_path(self) -> bool:
        return False

    def _setup_arch_params(self, arch_params:Optional[ArchParams])->None:
        # do we have shared arch params?
        if arch_params is None:
            # create our own arch params
            # the alphas are updated by exponentiated gradient descent
            # and not by gradients from backprop. so we don't require grad. 
            new_p = nn.Parameter(torch.ones(len(XnasOp.PRIMITIVES)), requires_grad=False)
            self.create_arch_params([('alphas', new_p)])
        else:
            assert arch_params.has_kind('alphas')
            self.set_arch_params(arch_params)

        # we store alphas in list so Pytorch don't register them
        self._alphas = list(self.arch_params().param_by_kind('alphas'))
        assert len(self._alphas)==1
