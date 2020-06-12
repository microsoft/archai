# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.common.utils import AverageMeter
from collections import defaultdict
from typing import Iterable, Optional, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from overrides import overrides

from archai.nas.model_desc import OpDesc
from archai.nas.operations import Op
from archai.nas.arch_params import ArchParams
from archai.common.utils import zip_eq

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
        self._is_first_call = True
        self._avg_grad_meter = AverageMeter()

        # we do this at the end so that we can capture all arch params registered by
        # any previous child modules
        self._setup_arch_params(arch_params)

    def get_avg_grad(self)->torch.Tensor:
        return self._avg_grad_meter.avg

    def update_alphas(self, eta:float):
        grad_flat = torch.flatten(self._avg_grad_meter.avg)
        rewards = torch.tensor([-torch.dot(grad_flat, torch.flatten(activ)) for activ in self._activs])
        exprewards = torch.exp(eta * rewards).cuda()
        # TODO: Will this remain registered?
        self._alphas[0] = torch.mul(self._alphas[0], exprewards)
        # TODO: Implement the weak learner eviction

    def _save_grad(self):
        def hook(grad):
            # TODO: Note that we have to reduce the minibatch to 1 finally
            self._avg_grad_meter.update(grad, n=1)
        return hook

    @overrides
    def forward(self, x):
        self._activs = [op(x) for op in self._ops]
        numer = sum(w * activ for w, activ in zip_eq(self._alphas[0], self._activs))
        denom = sum(self._alphas[0])
        self.pt = torch.div(numer, denom)

        # register gradient hook if first time
        if self._is_first_call:
            self.pt.register_hook(self._save_grad())
            self._is_first_call = False

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
            # TODO: dey: why requires_grad = False?
            new_p = nn.Parameter(  # TODO: use better init than uniform random?
                1.0e-3*torch.randn(len(XnasOp.PRIMITIVES)), requires_grad=False)
            self.create_arch_params([('alphas', new_p)])
        else:
            assert arch_params.has_kind('alphas')
            self.set_arch_params(arch_params)

        # we store alphas in list so Pytorch don't register them
        self._alphas = list(self.arch_params().param_by_kind('alphas'))
        assert len(self._alphas)==1
