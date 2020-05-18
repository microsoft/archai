from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from overrides import overrides

from .operations import Op, DropPath_
from .model_desc import EdgeDesc

class DagEdge(nn.Module):
    def __init__(self, desc:EdgeDesc, affine:bool, droppath:bool,
                 alphas_edge:Optional['DagEdge'])->None:
        super().__init__()
        self._wrapped = self._op = Op.create(desc.op_desc, affine,
                        alphas_edge.alphas() if alphas_edge else [])
        if droppath and self._op.can_drop_path():
            assert self.training
            self._wrapped = nn.Sequential(self._op, DropPath_())
        self._input_ids = desc.input_ids
        self.desc = desc

    @overrides
    def forward(self, inputs:List[torch.Tensor]):
        if len(self._input_ids)==1:
            return self._wrapped(inputs[self._input_ids[0]])
        elif len(self._input_ids) == len(inputs): # for perf
            return self._wrapped(inputs)
        else:
            return self._wrapped([inputs[i] for i in self._input_ids])

    def finalize(self)->Tuple[EdgeDesc, Optional[float]]:
        op_desc, rank = self._op.finalize()
        return (EdgeDesc(op_desc, self._input_ids), rank)

    def alphas(self)->Iterable[nn.Parameter]:
        for alpha in self._op.alphas():
            if alpha is not None:
                yield alpha

    def weights(self)->Iterable[nn.Parameter]:
        for w in self._op.weights():
            yield w

    def op(self)->Op:
        return self._op
