# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from typing import Iterable, List, Optional, Iterator, Tuple, Mapping
import heapq
import copy

import torch
from torch import Tensor, nn

from overrides import overrides

from archai.nas.model_desc import ConvMacroParams, OpDesc
from archai.nas.operations import Identity, Op, FactorizedReduce
from archai.common.utils import zip_eq
from archai.nas.arch_params import ArchParams


class NasBench101Op(Op):
    def __init__(self, op_desc:OpDesc, arch_params: Optional[ArchParams], affine:bool):
        super().__init__()

        vertex_op_name = op_desc.params['vertex_op']
        proj_first = op_desc.params['proj_first'] # first input needs projection

        self._vertex_op = Op.create(OpDesc(vertex_op_name, params=op_desc.params,
                                                in_len=1, trainables=None),
                                        affine=affine, arch_params=None)

        self._in_len = op_desc.in_len


        self._proj_op = Op.create(OpDesc('convbnrelu_1x1', params=op_desc.params,
                                                in_len=1, trainables=None),
                                        affine=affine, arch_params=None) \
                        if proj_first else None

    @overrides
    def forward(self, x:List[Tensor]):
        assert not isinstance(x, torch.Tensor)
        assert len(x) == self._in_len

        x0 = x[0] if not self._proj_first else self._proj_op(x[0])
        s = sum(x[1:]) + x0
        out = self._vertex_op(s)

        return out


