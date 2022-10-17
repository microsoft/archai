# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Optional, Tuple, List, Iterator

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


class MixedOp(Op):
    """The output of MixedOp is weighted output of all allowed primitives.
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
        assert MixedOp.PRIMITIVES[-1] == 'none'

        self._ops = nn.ModuleList()
        for primitive in MixedOp.PRIMITIVES:
            op = Op.create(
                OpDesc(primitive, op_desc.params, in_len=1, trainables=None),
                affine=affine, arch_params=None)
            self._ops.append(op)
        # we do this at the end so that we can capture all arch params registered by
        # any previous child modules
        self._setup_arch_params(arch_params)

    @overrides
    def forward(self, x):
        asm = F.softmax(self._alphas[0], dim=0)
        return sum(w * op(x) for w, op in zip(asm, self._ops))

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

    @overrides
    def ops(self)->Iterator[Tuple['Op', float]]: # type: ignore
        return iter(sorted(zip_eq(self._ops, self._alphas[0]),
                           key=lambda t:t[1], reverse=True))

    def _setup_arch_params(self, arch_params:Optional[ArchParams])->None:
        # do we have shared arch params?
        if arch_params is None:
            # create our own arch params
            new_p = nn.Parameter(  # TODO: use better init than uniform random?
                1.0e-3*torch.randn(len(MixedOp.PRIMITIVES)), requires_grad=True)
            self.create_arch_params([('alphas', new_p)])
        else:
            assert arch_params.has_kind('alphas')
            self.set_arch_params(arch_params)

        # we store alphas in list so Pytorch don't register them
        self._alphas = list(self.arch_params().param_by_kind('alphas'))
        assert len(self._alphas)==1
