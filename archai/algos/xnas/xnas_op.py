from archai.common.utils import AverageMeter
from collections import defaultdict
from typing import Iterable, Optional, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from overrides import overrides

from archai.nas.model_desc import OpDesc
from archai.nas.operations import Op

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

    def __init__(self, op_desc:OpDesc, alphas: Iterable[nn.Parameter],
                 affine:bool):
        super().__init__()

        # assume last PRIMITIVE is 'none'
        assert XnasOp.PRIMITIVES[-1] == 'none'

        self._set_alphas(alphas)
        self._ops = nn.ModuleList()
        for primitive in XnasOp.PRIMITIVES:
            op = Op.create(
                OpDesc(primitive, op_desc.params, in_len=1, trainables=None),
                affine=affine, alphas=alphas)
            self._ops.append(op)

        # for getting gradients to non-leaf node
        self._is_first_call = True
        self._avg_grad_meter = AverageMeter()

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
        numer = sum(w * activ for w, activ in zip(self._alphas[0], self._activs))
        denom = sum(self._alphas[0])
        self.pt = torch.div(numer, denom)

        # register gradient hook if first time
        if self._is_first_call:
            self.pt.register_hook(self._save_grad())
            self._is_first_call = False

        return self.pt


    @overrides
    def alphas(self) -> Iterable[nn.Parameter]:
        for alpha in self._alphas:
            yield alpha

    @overrides
    def weights(self) -> Iterable[nn.Parameter]:
        for op in self._ops:
            for w in op.parameters():
                yield w

    @overrides
    def finalize(self) -> Tuple[OpDesc, Optional[float]]:
        # select except 'none' op
        with torch.no_grad():
            val, i = torch.topk(self._alphas[0][:-1], 1)
            desc, _ = self._ops[i].finalize()
            return desc, float(val.item())

    @overrides
    def can_drop_path(self) -> bool:
        return False

    # TODO: Do we even need alphas to be registered with Pytorch
    # since we don't have to compute gradients on them?
    def _set_alphas(self, alphas: Iterable[nn.Parameter]) -> None:
        # must call before adding other ops
        assert len(list(self.parameters())) == 0
        self._alphas = list(alphas)
        if not len(self._alphas):
            # TODO: Better initialization than random?
            new_p = nn.Parameter(1.0e-3*torch.randn(len(XnasOp.PRIMITIVES)), requires_grad=False)
            # NOTE: This is a way to register parameters with PyTorch.
            # One creates a dummy variable with the parameters and then
            # asks back for the parameters in the object from Pytorch
            # which automagically registers the just created parameters.
            self._reg_alphas = new_p
            self._alphas = [p for p in self.parameters()]
