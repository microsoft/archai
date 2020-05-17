from copy import deepcopy
from typing import Iterable, List, Optional, Sequence, Tuple, Mapping
import heapq
import copy

import torch
from torch import Tensor, nn

from overrides import overrides

from archai.nas.model_desc import ConvMacroParams, OpDesc
from archai.nas.operations import Identity, Op, FactorizedReduce


class StopForward(Op):
    def __init__(self):
        super().__init__()
        self._sg_op = StopGradient()

    @overrides
    def forward(self, x):
        y = x - self._sg_op(x)
        return y

class StopGradient(Op):
    @staticmethod
    def _zero_grad(grad):
        return torch.zeros_like(grad)

    @overrides
    def forward(self, x):
        y = x * 1
        if self.training: # TODO: check with Dey, without this search time validation doesn't work
            y.register_hook(StopGradient._zero_grad)
        return y

class StopForwardReductionOp(Op):
    def __init__(self, op_desc:OpDesc, affine:bool):
        super().__init__()
        self._op = nn.Sequential(
            StopForward(),
            FactorizedReduce(op_desc, affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)


class StopGradientReduction(Op):
    def __init__(self, op_desc:OpDesc, affine:bool):
        super().__init__()
        self._op = nn.Sequential(
            StopGradient(),
            FactorizedReduce(op_desc, affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

class TempIdentityOp(Identity):
    def __init__(self, op_desc) -> None:
        super().__init__(op_desc)

    @overrides
    def forward(self, x):
        return x

class PetridishOp(Op):
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

    def __init__(self, op_desc:OpDesc, alphas: Iterable[nn.Parameter],
                 reduction:bool, affine:bool):
        super().__init__()

        # assume last PRIMITIVE is 'none' (this is used for finalize)
        assert PetridishOp.PRIMITIVES[-1] == 'none'

        # create alphas for the op
        self._set_alphas(alphas, op_desc.in_len)

        # create edges for the op, each edge connects input state,
        # within each edge we will have all N primitives
        self._edges = nn.ModuleList()

        for i in range(op_desc.in_len):
            # edge contains all primitives with alphas
            edge = nn.ModuleList()
            self._edges.append(edge)

            # for each input stride could be different,
            # so we will make copy of our params and then set stride for this input
            params = deepcopy(op_desc.params)
            params['stride'] = op_desc.params['_strides'][i]

            # create primitives for the edge
            for primitive in PetridishOp.PRIMITIVES:
                primitive_op = Op.create(OpDesc(primitive, params=params,
                                                in_len=1, trainables=None),
                                        affine=affine, alphas=alphas)
                # wrap primitive with sg
                op = nn.Sequential(StopGradient(), primitive_op)
                edge.append(op)

        # TODO: check with Dey: Do we really need StopForwardReductionOp
        #   or StopGradientReductionOp because these two will only make sense
        #   for cell stems.
        # NOTE: Consider the case where prev_prev is normal, prev is reduction
        # then s_0 is twice as big in each dimension as s_1 and the number of channels
        # won't match. So you have to use StopGradientReductionOp on s_1 to make it match.
        self._sf = StopForward()

    @overrides
    def forward(self, x:List[Tensor]):
        assert not isinstance(x, torch.Tensor)
        s = 0.0
        # apply each input in the list to associated edge
        for i, (xi, edge) in enumerate(zip(x, self._edges)):
            # apply input to each primitive within edge
            # TODO: is avg better idea than sum here? sum can explode as
            #   number of primitives goes up
            s = sum(a * op(xi) for a, op in zip(self._alphas[i], edge)) + s
        return self._sf(s)

    @overrides
    def alphas(self) -> Iterable[nn.Parameter]:
        for alpha in self._alphas:
            yield alpha

    @overrides
    def weights(self) -> Iterable[nn.Parameter]:
        #TODO: cache this?
        for edge in self._edges:
            for op in edge:
                for w in op.parameters():
                    yield w

    @overrides
    def finalize(self) -> Tuple[OpDesc, Optional[float]]:
        with torch.no_grad(): # probably this is not needed
            # Create list of (alpha, input_id, op_desc), sort them, select top k.
            # Here op should be nn.Sequence of sg followed by primitive.
            # First for loop gets edge and associated alphas.
            # Second for loop gets op and associated alpha.
            l = ((a, i, op[1])                                         \
                for edge_alphas, i, edge in                                 \
                    zip(self._alphas, range(self.desc.in_len), self._edges) \
                for a, op in zip(edge_alphas, edge)) # op is nn.Sequence

            # select 3 largest ops by alpha
            sel = heapq.nlargest(3, l, key=lambda t: t[0])  # TODO: add config

        # multi_op needs to know each input and associated primitive
        final_op_desc = OpDesc(name='multi_op',
                                params={
                                    # copy convolution parameters
                                    'conv': self.desc.params['conv']
                                },
                                # Number of inputs remains same although only 3 of
                                # them will be used.
                                in_len=self.desc.in_len,
                                trainables=None,
                                # primitive's finalize call also records its
                                # weights in description. finalize call returns
                                # (desc, rank) where rank for primitive is None
                                children = [op.finalize()[0] for a,i,op in sel],
                                children_ins = [i for a,i,op in sel]
                               )

        # rank=None to indicate no further selection needed as in darts
        return final_op_desc, None

    def get_op_desc(self, index:int)->OpDesc:
        ''' index: index in the primitives list '''
        assert index < len(self.PRIMITIVES)
        desc, _ = self._ops[index].finalize()
        return desc

    def _set_alphas(self, alphas: Iterable[nn.Parameter], in_len:int) -> None:
        assert len(list(self.parameters()))==0 # must call before adding other ops

        # If we are using shared alphas from another cell, don't create our own
        self._alphas = list(alphas)
        if not len(self._alphas):
            # Each nn.Parameter is tensor with alphas for entire edge.
            # We will create same numbers of nn.Parameter as number of edges
            n_primitives = len(PetridishOp.PRIMITIVES)
            pl = nn.ParameterList((
                nn.Parameter(  # TODO: use better init than uniform random?
                    torch.FloatTensor(n_primitives).uniform_(-0.1, 0.1),
                    requires_grad=True)
                for _ in range(in_len)
            ))
            # register parameters with module
            self._reg_alphas = pl
            # save PyTorch registered alphas into list for later use
            self._alphas = [p for p in self.parameters()]
