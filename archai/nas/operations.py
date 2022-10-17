# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from argparse import ArgumentError
from typing import Callable, Iterable, Iterator, List, Mapping, Tuple, Dict, Optional, Union
from abc import ABC, abstractmethod
import copy
import math

from overrides import overrides, EnforceOverrides

import torch
from torch import affine_grid_generator, nn, Tensor, strided

from archai.common import utils, ml_utils
from archai.nas.model_desc import OpDesc, ConvMacroParams
from archai.nas.arch_params import ArchParams
from archai.nas.arch_module import ArchModule
from archai.common.utils import zip_eq

# type alias
OpFactoryFn = Callable[[OpDesc, Iterable[nn.Parameter]], 'Op']

# Each op is a unary tensor operator, all take same constructor params
# TODO: swap order of arch_params and affine to match with create signature
_ops_factory:Dict[str, Callable] = {
    'max_pool_3x3':     lambda op_desc, arch_params, affine:
                            PoolBN('max', op_desc, affine),
    'avg_pool_3x3':     lambda op_desc, arch_params, affine:
                            PoolBN('avg', op_desc, affine),
    'skip_connect':     lambda op_desc, arch_params, affine:
                            SkipConnect(op_desc, affine),
    'sep_conv_3x3':     lambda op_desc, arch_params, affine:
                            SepConv(op_desc, 3, 1, affine),
    'sep_conv_5x5':     lambda op_desc, arch_params, affine:
                            SepConv(op_desc, 5, 2, affine),
    'convbnrelu_3x3':     lambda op_desc, arch_params, affine: # used by NASBench-101
                            ConvBNReLU(op_desc, 3, 1, 1, affine),
    'convbnrelu_1x1':     lambda op_desc, arch_params, affine: # used by NASBench-101
                            ConvBNReLU(op_desc, 1, 1, 0, affine),
    'dil_conv_3x3':     lambda op_desc, arch_params, affine:
                            DilConv(op_desc, 3, op_desc.params['stride'], 2, 2, affine),
    'dil_conv_5x5':     lambda op_desc, arch_params, affine:
                            DilConv(op_desc, 5, op_desc.params['stride'], 4, 2, affine),
    'none':             lambda op_desc, arch_params, affine:
                            Zero(op_desc),
    'identity':         lambda op_desc, arch_params, affine:
                            Identity(op_desc),
    'sep_conv_7x7':     lambda op_desc, arch_params, affine:
                            SepConv(op_desc, 7, 3, affine),
    'conv_7x1_1x7':     lambda op_desc, arch_params, affine:
                            FacConv(op_desc, 7, 3, affine),
    'prepr_reduce':     lambda op_desc, arch_params, affine:
                            FactorizedReduce(op_desc, affine),
    'prepr_normal':     lambda op_desc, arch_params, affine:
                            ReLUConvBN(op_desc, 1, 1, 0, affine),
    'stem_conv3x3':       lambda op_desc, arch_params, affine:
                            StemConv3x3(op_desc, affine),
    'stem_conv3x3Relu':       lambda op_desc, arch_params, affine:
                            StemConv3x3Relu(op_desc, affine),
    'stem_conv3x3_s4':   lambda op_desc, arch_params, affine:
                            StemConv3x3S4(op_desc, affine),
    'stem_conv3x3_s4s2':   lambda op_desc, arch_params, affine:
                            StemConv3x3S4S2(op_desc, affine),
    'pool_adaptive_avg2d':       lambda op_desc, arch_params, affine:
                            PoolAdaptiveAvg2D(),
    'pool_avg2d7x7':    lambda op_desc, arch_params, affine:
                            AvgPool2d7x7(),
    'pool_mean_tensor':  lambda op_desc, arch_params, affine:
                            PoolMeanTensor(),
    'concate_channels':   lambda op_desc, arch_params, affine:
                            ConcateChannelsOp(op_desc, affine),
    'proj_channels':   lambda op_desc, arch_params, affine:
                            ProjectChannelsOp(op_desc, affine),
    'linear':           lambda op_desc, arch_params, affine:
                            LinearOp(op_desc),
    'multi_op':         lambda op_desc, arch_params, affine:
                            MultiOp(op_desc, affine)
}

class Op(ArchModule, ABC, EnforceOverrides):
    @staticmethod
    def create(op_desc:OpDesc, affine:bool, arch_params:Optional[ArchParams]=None)->'Op':
        global _ops_factory
        op = _ops_factory[op_desc.name](op_desc, arch_params, affine)

        # TODO: annotate as Final?
        op.desc = op_desc # type: ignore
        # load any pre-trained weights
        op.set_trainables(op_desc.trainables)
        return op

    def get_trainables(self)->Mapping:
        return {'name': self.desc.name, 'sd': self.state_dict()}

    def set_trainables(self, state_dict)->None:
        if state_dict is not None:
            assert state_dict['name'] == self.desc.name
            # At search time, batchnorms are not affine so when restoring
            # weights during pretraining we don't have those keys which is why
            # we use strict=False
            # TODO: should we assign op_desc uuid to enforce more strictness?
            self.load_state_dict(state_dict['sd'], strict=False)

    @staticmethod
    def register_op(name: str, factory_fn: Callable, exists_ok=True) -> None:
        global _ops_factory
        if name in _ops_factory:
            if not exists_ok:
                raise ArgumentError(argument=None, message=f'{name} is already registered in op factory')
            # else no need to register again
        else:
            _ops_factory[name] = factory_fn

    def finalize(self)->Tuple[OpDesc, Optional[float]]:
        """for trainable op, return final op and its rank"""

        # make copy because we are going to modify the trainables
        desc = self.desc.clone(clone_trainables=False)
        # make copy of trainables so we don't keep around references
        desc.trainables = copy.deepcopy(self.get_trainables())
        return desc, None # desc, rank (None means op is unranked and cannot be removed)

    def ops(self)->Iterator[Tuple['Op', float]]: # type: ignore
        """Return contituent ops, if this op is primitive just return self"""
        yield self, math.nan

    # if op should not be dropped during drop path then return False
    def can_drop_path(self)->bool:
        return True


class PoolBN(Op):
    """AvgPool or MaxPool - BN """

    def __init__(self, pool_type:str, op_desc:OpDesc, affine:bool):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in

        stride = op_desc.params['stride']
        kernel_size = op_desc.params.get('kernel_size', 3)
        padding = op_desc.params.get('padding', 1)

        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(
                kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        # TODO: pt.darts applies BN but original implementation doesn't
        # self.bn = nn.BatchNorm2d(ch_in, affine=affine)

    @overrides
    def forward(self, x):
        out = self.pool(x)
        #out = self.bn(out)
        return out

class SkipConnect(Op):
    def __init__(self, op_desc:OpDesc, affine) -> None:
        super().__init__()

        stride = op_desc.params['stride']
        self._op = Identity(op_desc) if stride == 1 \
                              else FactorizedReduce(op_desc, affine)

    @overrides
    def forward(self, x:Tensor)->Tensor:
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        # TODO: original darts drops path only for identity, not FactorizedReduce
        #   but that seems wrong. Here we drop path for skip connect.
        return False


class FacConv(Op):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """

    def __init__(self, op_desc:OpDesc, kernel_length:int, padding:int,
                 affine:bool):
        super().__init__()

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out

        stride = op_desc.params['stride']

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_in, (kernel_length, 1),
                      stride, padding, bias=False),
            nn.Conv2d(ch_in, ch_out, (1, kernel_length),
                      stride, padding, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self.net(x)


class ReLUConvBN(Op): # std DARTS op has BN at the end
    def __init__(self, op_desc:OpDesc, kernel_size:int, stride:int, padding:int,
                 affine:bool):
        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out

        super(ReLUConvBN, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self.op(x)


class ConvBNReLU(Op): # NAS bench op has BN in the middle
    def __init__(self, op_desc:OpDesc, kernel_size:int, stride:int, padding:int,
                 affine:bool):
        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out

        super(ConvBNReLU, self).__init__()

        self.op = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine),
            nn.ReLU(inplace=True) #TODO: review inplace
        )

    @overrides
    def forward(self, x):
        return self.op(x)

class DilConv(Op):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """

    def __init__(self, op_desc:OpDesc, kernel_size:int, stride:int,  padding:int,
                 dilation:int, affine:bool):
        super(DilConv, self).__init__()
        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out

        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation,
                      groups=ch_in, bias=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine),
        )

    @overrides
    def forward(self, x):
        return self.op(x)


class SepConv(Op):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2

    This is same as two DilConv stacked with dilation=1
    """

    def __init__(self, op_desc:OpDesc, kernel_size:int, padding:int,
                 affine:bool):
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            DilConv(op_desc, kernel_size, op_desc.params['stride'],
                    padding, dilation=1, affine=affine),
            DilConv(op_desc, kernel_size, 1,
                    padding, dilation=1, affine=affine))

    @overrides
    def forward(self, x):
        return self.op(x)


class Identity(Op):
    def __init__(self, op_desc:OpDesc):
        super().__init__()
        stride, conv_params = op_desc.params['stride'], op_desc.params['conv']
        assert stride == 1
        assert conv_params.ch_in == conv_params.ch_out

    @overrides
    def forward(self, x):
        return x

    @overrides
    def can_drop_path(self)->bool:
        return False


class Zero(Op):
    """Represents no connection. Zero op can be thought of 1x1 kernel with fixed zero weight.
    For stride=1, it will produce output of same dimension as input but with all 0s. Now with stride of 2, it will zero out every other pixel in output.
    """

    def __init__(self, op_desc:OpDesc):
        super().__init__()
        stride = op_desc.params['stride']
        self.stride = stride

    @overrides
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(Op):
    """
    reduce feature maps height/width by 2X while doubling channels using two 1x1 convs, each with stride=2.
    """
    # TODO: modify to take number of nodes in reduction cells where stride 2 was applied (currently only first two input nodes)

    def __init__(self, op_desc:OpDesc, affine:bool):
        super(FactorizedReduce, self).__init__()

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out

        assert ch_out % 2 == 0

        self.relu = nn.ReLU()
        # this conv layer operates on even pixels to produce half width, half channels
        self.conv_1 = nn.Conv2d(ch_in, ch_out // 2, 1,
                                stride=2, padding=0, bias=False)
        # this conv layer operates on odd pixels (because of code in forward()) to produce half width, half channels
        self.conv_2 = nn.Conv2d(ch_in, ch_out // 2, 1,
                                stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, affine=affine)

    @overrides
    def forward(self, x):
        x = self.relu(x)

        # x: torch.Size([32, 32, 32, 32])
        # conv1: [b, c_out//2, d//2, d//2]
        # conv2: []
        # out: torch.Size([32, 32, 16, 16])

        # concate two half channels to produce same number of channels as before but with output as only half the width
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class StemBase(Op):
    """Abstract base class for model stems that enforces reduction property
    indicating amount of spatial map reductions performed by stem, i.e., reduction=2 for each stride=2"""
    def __init__(self, reduction:int) -> None:
        super().__init__()
        self.reduction = reduction

class StemConv3x3(StemBase):
    def __init__(self, op_desc:OpDesc, affine:bool)->None:
        super().__init__(1)

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out

        self._op = nn.Sequential( # 3 => 48
            # batchnorm is added after each layer. Bias is turned off due to
            # BN in conv layer.
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class StemConv3x3Relu(StemBase): # used in NASbench-101
    def __init__(self, op_desc:OpDesc, affine:bool)->None:
        super().__init__(1)

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out

        self._op = nn.Sequential( # 3 => 48
            # batchnorm is added after each layer. Bias is turned off due to
            # BN in conv layer.
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine),
            nn.ReLU(inplace=True)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class StemConv3x3S4(StemBase):
    def __init__(self, op_desc, affine:bool)->None:
        super().__init__(4)

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out

        self._op = nn.Sequential(
            # keep in sync with StemConv3x3S4S2
            nn.Conv2d(ch_in, ch_out//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out//2, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out//2, ch_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class StemConv3x3S4S2(StemBase):
    def __init__(self, op_desc, affine:bool)->None:
        super().__init__(8)

        conv_params:ConvMacroParams = op_desc.params['conv']
        ch_in = conv_params.ch_in
        ch_out = conv_params.ch_out

        self._op = nn.Sequential(
            # s4 ops - keep in sync with StemConv3x3S4
            nn.Conv2d(ch_in, ch_out//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out//2, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out//2, ch_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine),

            # s2 ops
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch_out, affine=affine)
        )

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class AvgPool2d7x7(Op):
    def __init__(self)->None:
        super().__init__()
        self._op = nn.AvgPool2d(7)

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class PoolAdaptiveAvg2D(Op):
    def __init__(self)->None:
        super().__init__()
        self._op = nn.AdaptiveAvgPool2d(1)

    @overrides
    def forward(self, x):
        return self._op(x)

    @overrides
    def can_drop_path(self)->bool:
        return False

class PoolMeanTensor(Op): # used in Nasbench-101
    def __init__(self)->None:
        super().__init__()

    @overrides
    def forward(self, x):
        return torch.mean(x, (2, 3))

    @overrides
    def can_drop_path(self)->bool:
        return False

class LinearOp(Op):
    def __init__(self, op_desc:OpDesc)->None:
        super().__init__()

        n_ch = op_desc.params['n_ch']
        n_classes = op_desc.params['n_classes']

        self._op = nn.Linear(n_ch, n_classes)

    @overrides
    def forward(self, x:torch.Tensor):
        flattened = x.view(x.size(0), -1)
        return self._op(flattened)

    @overrides
    def can_drop_path(self)->bool:
        return False

class MergeOp(Op, ABC):
    def __init__(self, op_desc:OpDesc, affine:bool)->None:
        super().__init__()

        self.ch_in = op_desc.params['conv'].ch_in
        self.ch_out = op_desc.params['conv'].ch_out
        self.out_states = op_desc.params['out_states']

    @overrides
    def forward(self, states:List[torch.Tensor]):
        raise NotImplementedError()

    @overrides
    def can_drop_path(self)->bool:
        return False

class ConcateChannelsOp(MergeOp):
    def __init__(self, op_desc:OpDesc, affine:bool)->None:
        super().__init__(op_desc, affine)

    @overrides
    def forward(self, states:List[torch.Tensor]):
        return torch.cat(states[-self.out_states:], dim=1)

class ProjectChannelsOp(MergeOp):
    def __init__(self, op_desc:OpDesc, affine:bool)->None:
        super().__init__(op_desc, affine)

        self._op = nn.Sequential(
            nn.Conv2d(self.ch_in, self.ch_out, 1, # 1x1 conv
                    stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ch_out, affine=affine)
        )

    @overrides
    def forward(self, states:List[torch.Tensor]):
        concatenated = torch.cat(states[-self.out_states:], dim=1)
        return self._op(concatenated)


class DropPath_(nn.Module):
    """Replace values in tensor by 0. with probability p
        Ref: https://arxiv.org/abs/1605.07648
    """

    def __init__(self, p:float=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    @overrides
    def forward(self, x):
        return ml_utils.drop_path_(x, self.p, self.training)

class MultiOp(Op):
    def __init__(self, op_desc:OpDesc, affine:bool) -> None:
        """MultiOp combines multiple ops to one op. The set of ops to combine
            if passed through op_desc.children and each of children's inputs are
            passed through op_desc.children_ins. This op will receive list of
            inputs in forward() and each of the children works on one of these
            inputs and generates an output. All outputs of children are then
            combined using projection operation to produce final output of the
            overall op.
        """
        super().__init__()

        # get list of inputs and associated primitives
        iop_descs = op_desc.children
        ins = op_desc.children_ins
        assert iop_descs is not None and ins is not None and len(iop_descs) == len(ins)

        # conv params typically specified by macro builder
        conv_params:ConvMacroParams = op_desc.params['conv']

        self._ops = nn.ModuleList()
        self._ins:List[int] = []

        for i, iop_desc in zip(ins, iop_descs):
            iop_desc.params['conv'] = conv_params
            self._ops.append(Op.create(iop_desc, affine=affine))
            self._ins.append(i)

        # number of channels as we will concate output of ops
        ch_out_sum = conv_params.ch_out * len(self._ins)
        ch_adj_desc =  OpDesc('proj_channels',
            {
                'conv': ConvMacroParams(ch_out_sum, conv_params.ch_out),
                'out_states': len(self._ins)
            },
            in_len=1, trainables=None, children=None)
        self._ch_adj = Op.create(ch_adj_desc, affine=affine)


    @overrides
    def forward(self, x:Union[Tensor, List[Tensor]])->Tensor:
        # we may receive k=1..N tensors as inputs. Currently DagEdge will pass
        # tensor as-is if k=1 to support primitives and create list of tensors
        # if k > 1. So below we handle k = 1 case.
        if not isinstance(x, list):
            x = [x]
        return self._ch_adj([op(x[i]) for op, i in zip(self._ops, self._ins)])

