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


class GsOp(Op):
    """The output of GsOp is weighted output of all allowed primitives.
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
        assert GsOp.PRIMITIVES[-1] == 'none'

        self._gs_num_sample = op_desc.params['gs_num_sample']

        self._ops = nn.ModuleList()
        for primitive in GsOp.PRIMITIVES:
            op = Op.create(
                OpDesc(primitive, op_desc.params, in_len=1, trainables=None),
                affine=affine, arch_params=None)
            self._ops.append(op)
        # we do this at the end so that we can capture all arch params registered by
        # any previous child modules
        self._setup_arch_params(arch_params)

    @overrides
    def forward(self, x):
        # soft sample from the categorical distribution
        # via gumbel softmax distribution
        # TODO: should we be normalizing the ensemble?
        #sampled = torch.zeros(alphas.size(), requires_grad=True)

        sample_storage = []
        for _ in range(self._gs_num_sample):
            sampled = F.gumbel_softmax(self._alphas[0], tau=1, hard=False, eps=1e-10, dim=-1)
            sample_storage.append(sampled)

        samples_summed = torch.sum(torch.stack(sample_storage, dim=0), dim=0)
        return sum(w * op(x) for w, op in zip(samples_summed, self._ops))

    @overrides
    def finalize(self) -> Tuple[OpDesc, Optional[float]]:
        # finalization where each edge gets to keep as many
        # unique operations that are sampled
        sample_storage = []
        for i in range(self._gs_num_sample):
            sampled = F.gumbel_softmax(self._alphas[0], tau=1, hard=True, eps=1e-10, dim=-1)
            sample_storage.append(sampled)

        samples_summed = torch.sum(torch.stack(sample_storage, dim=0), dim=0)
        greater_than_0 = samples_summed > 0
        children = []
        children_ins = []

        for i in range(greater_than_0.size()[0]):
            if greater_than_0[i]:
                children.append(self._ops[i].finalize()[0])
                # all the ops will operate on the single node input
                children_ins.append(0)

        final_op_desc = OpDesc(name='multi_op',
                                params={
                                    # copy convolution parameters
                                    'conv': self.desc.params['conv']
                                },
                                # number of inputs remains same and in this
                                # case should be 1
                                in_len=self.desc.in_len,
                                trainables=None,
                                # primitive's finalize call also records its
                                # weights in description. finalize call returns
                                # (desc, rank) where rank for primitive is None
                                children = children,
                                children_ins = children_ins
                               )

        return final_op_desc, None

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
                1.0e-3*torch.randn(len(GsOp.PRIMITIVES)), requires_grad=True)
            self.create_arch_params([('alphas', new_p)])
        else:
            assert arch_params.has_kind('alphas')
            self.set_arch_params(arch_params)

        # we store alphas in list so Pytorch don't register them
        self._alphas = list(self.arch_params().param_by_kind('alphas'))
        assert len(self._alphas)==1
