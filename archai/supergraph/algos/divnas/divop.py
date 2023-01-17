# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Optional, Tuple, List, Iterator
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import math

from overrides import overrides

from archai.nas.model_desc import OpDesc
from archai.nas.operations import Op
from archai.common.common import get_conf
from archai.nas.arch_params import ArchParams
from archai.common.utils import zip_eq

# TODO: reduction cell might have output reduced by 2^1=2X due to
#   stride 2 through input nodes however FactorizedReduce does only
#   4X reduction. Is this correct?


class DivOp(Op):
    """The output of DivOp is weighted output of all allowed primitives.
    """

    PRIMITIVES = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',  # identity
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'none'  # this must be at the end so top1 doesn't choose it
    ]

    # def _indices_of_notallowed(self):
    #     ''' computes indices of notallowed ops in PRIMITIVES '''
    #     self._not_allowed_indices = []
    #     for op_name in self.NOTALLOWED:
    #         self._not_allowed_indices.append(self.PRIMITIVES.index(op_name))
    #     self._not_allowed_indices = sorted(self._not_allowed_indices, reverse=True)

    # def _create_mapping_valid_to_orig(self):
    #     ''' Creates a list with indices of the valid ops to the original list '''
    #     self._valid_to_orig = []
    #     for i, prim in enumerate(self.PRIMITIVES):
    #         if prim in self.NOTALLOWED:
    #             continue
    #         else:
    #             self._valid_to_orig.append(i)

    def __init__(self, op_desc:OpDesc, arch_params:Optional[ArchParams],
                 affine:bool):
        super().__init__()

        # assume last PRIMITIVE is 'none'
        assert DivOp.PRIMITIVES[-1] == 'none'

        conf = get_conf()
        trainer = conf['nas']['search']['divnas']['archtrainer']
        finalizer = conf['nas']['search']['finalizer']

        if trainer == 'noalpha' and finalizer == 'default':
            raise NotImplementedError('noalpha trainer is not implemented for the default finalizer')

        if trainer != 'noalpha':
            self._setup_arch_params(arch_params)
        else:
            self._alphas = None

        self._ops = nn.ModuleList()
        for primitive in DivOp.PRIMITIVES:
            op = Op.create(
                OpDesc(primitive, op_desc.params, in_len=1, trainables=None),
                affine=affine, arch_params=None)
            self._ops.append(op)

        # various state variables for diversity
        self._collect_activations = False
        self._forward_counter = 0
        self._batch_activs = None
        #self._indices_of_notallowed()
        #self._create_mapping_valid_to_orig()

    @property
    def collect_activations(self)->bool:
        return self._collect_activations

    @collect_activations.setter
    def collect_activations(self, to_collect:bool)->None:
        self._collect_activations = to_collect

    @property
    def activations(self)->Optional[List[np.array]]:
        return self._batch_activs

    @property
    def num_primitive_ops(self)->int:
        return len(self.PRIMITIVES)

    @overrides
    def forward(self, x):

        # save activations to object
        if self._collect_activations:
            self._forward_counter += 1
            activs = [op(x) for op in self._ops]
            # delete the activation for none type
            # as we don't consider it
            activs = activs[:-1]
            self._batch_activs = [t.cpu().detach().numpy() for t in activs]
            
        if self._alphas:
            asm = F.softmax(self._alphas[0], dim=0)
            result = sum(w * op(x) for w, op in zip(asm, self._ops))
        else:
            result = sum(op(x) for op in self._ops)

        return result

    @overrides
    def ops(self)->Iterator[Tuple['Op', float]]: # type: ignore
        return iter(sorted(zip_eq(self._ops,
                                  self._alphas[0] if self._alphas is not None else [math.nan for _ in range(len(self._ops))]),
                           key=lambda t:t[1], reverse=True))

    # def get_valid_op_desc(self, index:int)->OpDesc:
    #     ''' index: index in the valid index list '''
    #     assert index <= self.num_valid_div_ops
    #     orig_index = self._valid_to_orig[index]        
    #     desc, _ = self._ops[orig_index].finalize()
    #     return desc

    @overrides
    def finalize(self) -> Tuple[OpDesc, Optional[float]]:
        ''' Divnas with default finalizer option needs this override else 
        the finalizer in base class returns the whole divop '''
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
            new_p = nn.Parameter(  # TODO: use better init than uniform random?
                1.0e-3*torch.randn(len(self.PRIMITIVES)), requires_grad=True)
            self.create_arch_params([('alphas', new_p)])
        else:
            assert arch_params.has_kind('alphas')
            self.set_arch_params(arch_params)

        # we store alphas in list so Pytorch don't register them
        self._alphas = list(self.arch_params().param_by_kind('alphas'))
        assert len(self._alphas)==1