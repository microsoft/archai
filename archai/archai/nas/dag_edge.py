# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from overrides import overrides

from archai.nas.operations import Op, DropPath_
from archai.nas.model_desc import EdgeDesc
from archai.nas.arch_module import ArchModule

class DagEdge(ArchModule):
    def __init__(self, desc:EdgeDesc, affine:bool, droppath:bool,
                 template_edge:Optional['DagEdge'])->None:
        super().__init__()
        # we may need to wrap op is droppath is needed
        self._wrapped = self._op = Op.create(desc.op_desc, affine,
                        template_edge.op().arch_params() if template_edge is not None else None)
        if droppath and self._op.can_drop_path():
            assert self.training
            self._wrapped = nn.Sequential(self._op, DropPath_())
        self.input_ids = desc.input_ids
        self.desc = desc

    @overrides
    def forward(self, inputs:List[torch.Tensor]):
        if len(self.input_ids)==1:
            return self._wrapped(inputs[self.input_ids[0]])
        elif len(self.input_ids) == len(inputs): # for perf
            return self._wrapped(inputs)
        else:
            return self._wrapped([inputs[i] for i in self.input_ids])

    def op(self)->Op:
        return self._op
