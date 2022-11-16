# Copyright (c) Microsoft Corporation (DeepSpeed).
# Licensed under the MIT License.

from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

module_flop_count = []
module_mac_count = []
old_functions = {}


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = torch.numel(input) * out_features
    return 2 * macs, macs


def _relu_flops_compute(input, inplace=False):
    return torch.numel(input), 0


def _prelu_flops_compute(input: Tensor, weight: Tensor):
    return torch.numel(input), 0


def _elu_flops_compute(input: Tensor, alpha: float = 1.0, inplace: bool = False):
    return torch.numel(input), 0


def _leaky_relu_flops_compute(input: Tensor,
                              negative_slope: float = 0.01,
                              inplace: bool = False):
    return torch.numel(input), 0


def _relu6_flops_compute(input: Tensor, inplace: bool = False):
    return torch.numel(input), 0


def _silu_flops_compute(input: Tensor, inplace: bool = False):
    return torch.numel(input), 0


def _gelu_flops_compute(input):
    return torch.numel(input), 0


def _pool_flops_compute(input,
                        kernel_size,
                        stride=None,
                        padding=0,
                        dilation=None,
                        ceil_mode=False,
                        count_include_pad=True,
                        divisor_override=None,
                        return_indices=None):
    return torch.numel(input), 0


def _conv_flops_compute(input,
                        weight,
                        bias=None,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1):
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding, ) * length
    strides = stride if type(stride) is tuple else (stride, ) * length
    dilations = dilation if type(dilation) is tuple else (dilation, ) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] -
                      (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(output_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _conv_trans_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding, ) * length
    strides = stride if type(stride) is tuple else (stride, ) * length
    dilations = dilation if type(dilation) is tuple else (dilation, ) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):

        output_dim = (input_dim + 2 * paddings[idx] -
                      (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(input_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(_prod(output_dims))

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _batch_norm_flops_compute(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    has_affine = weight is not None
    if training:
        # estimation
        return torch.numel(input) * (5 if has_affine else 4), 0
    flops = torch.numel(input) * (2 if has_affine else 1)
    return flops, 0


def _layer_norm_flops_compute(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return torch.numel(input) * (5 if has_affine else 4), 0


def _group_norm_flops_compute(input: Tensor,
                              num_groups: int,
                              weight: Optional[Tensor] = None,
                              bias: Optional[Tensor] = None,
                              eps: float = 1e-5):
    has_affine = weight is not None
    # estimation
    return torch.numel(input) * (5 if has_affine else 4), 0


def _instance_norm_flops_compute(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return torch.numel(input) * (5 if has_affine else 4), 0


def _upsample_flops_compute(input,
                            size=None,
                            scale_factor=None,
                            mode="nearest",
                            align_corners=None):
    if size is not None:
        if isinstance(size, tuple):
            return int(_prod(size)), 0
        else:
            return int(size), 0
    assert scale_factor is not None, "either size or scale_factor should be defined"
    flops = torch.numel(input)
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops * int(_prod(scale_factor))
    else:
        flops * scale_factor**len(input)
    return flops, 0


def _softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    return torch.numel(input), 0


def _embedding_flops_compute(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return 0, 0


def _dropout_flops_compute(input, p=0.5, training=True, inplace=False):
    return 0, 0


def _matmul_flops_compute(input, other, *, out=None):
    """
    Count flops for the matmul operation.
    """
    macs = _prod(input.shape) * other.shape[-1]
    return 2 * macs, macs


def _addmm_flops_compute(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(input.shape), macs


def _einsum_flops_compute(equation, *operands):
    """
    Count flops for the einsum operation.
    """
    equation = equation.replace(" ", "")
    input_shapes = [o.shape for o in operands]

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
    for line in optim.split("\n"):
        if "optimized flop" in line.lower():
            flop = int(float(line.split(":")[-1]))
            return flop, 0
    raise NotImplementedError("Unsupported einsum operation.")


def _tensor_addmm_flops_compute(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the tensor addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(self.shape), macs


def _mul_flops_compute(input, other, *, out=None):
    return _elementwise_flops_compute(input, other)


def _add_flops_compute(input, other, *, alpha=1, out=None):
    return _elementwise_flops_compute(input, other)


def _elementwise_flops_compute(input, other):
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return _prod(other.shape), 0
        else:
            return 1, 0
    elif not torch.is_tensor(other):
        return _prod(input.shape), 0
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = _prod(final_shape)
        return flops, 0


def wrapFunc(func, funcFlopCompute):
    oldFunc = func
    name = func.__name__
    old_functions[name] = oldFunc

    def newFunc(*args, **kwds):
        flops, macs = funcFlopCompute(*args, **kwds)
        if module_flop_count:
            module_flop_count[-1].append((name, flops))
        if module_mac_count and macs:
            module_mac_count[-1].append((name, macs))
        return oldFunc(*args, **kwds)

    newFunc.__name__ = func.__name__

    return newFunc


def _patch_functionals():
    # FC
    F.linear = wrapFunc(F.linear, _linear_flops_compute)

    # convolutions
    F.conv1d = wrapFunc(F.conv1d, _conv_flops_compute)
    F.conv2d = wrapFunc(F.conv2d, _conv_flops_compute)
    F.conv3d = wrapFunc(F.conv3d, _conv_flops_compute)

    # conv transposed
    F.conv_transpose1d = wrapFunc(F.conv_transpose1d, _conv_trans_flops_compute)
    F.conv_transpose2d = wrapFunc(F.conv_transpose2d, _conv_trans_flops_compute)
    F.conv_transpose3d = wrapFunc(F.conv_transpose3d, _conv_trans_flops_compute)

    # activations
    F.relu = wrapFunc(F.relu, _relu_flops_compute)
    F.prelu = wrapFunc(F.prelu, _prelu_flops_compute)
    F.elu = wrapFunc(F.elu, _elu_flops_compute)
    F.leaky_relu = wrapFunc(F.leaky_relu, _leaky_relu_flops_compute)
    F.relu6 = wrapFunc(F.relu6, _relu6_flops_compute)
    if hasattr(F, "silu"):
        F.silu = wrapFunc(F.silu, _silu_flops_compute)
    F.gelu = wrapFunc(F.gelu, _gelu_flops_compute)

    # Normalizations
    F.batch_norm = wrapFunc(F.batch_norm, _batch_norm_flops_compute)
    F.layer_norm = wrapFunc(F.layer_norm, _layer_norm_flops_compute)
    F.instance_norm = wrapFunc(F.instance_norm, _instance_norm_flops_compute)
    F.group_norm = wrapFunc(F.group_norm, _group_norm_flops_compute)

    # poolings
    F.avg_pool1d = wrapFunc(F.avg_pool1d, _pool_flops_compute)
    F.avg_pool2d = wrapFunc(F.avg_pool2d, _pool_flops_compute)
    F.avg_pool3d = wrapFunc(F.avg_pool3d, _pool_flops_compute)
    F.max_pool1d = wrapFunc(F.max_pool1d, _pool_flops_compute)
    F.max_pool2d = wrapFunc(F.max_pool2d, _pool_flops_compute)
    F.max_pool3d = wrapFunc(F.max_pool3d, _pool_flops_compute)
    F.adaptive_avg_pool1d = wrapFunc(F.adaptive_avg_pool1d, _pool_flops_compute)
    F.adaptive_avg_pool2d = wrapFunc(F.adaptive_avg_pool2d, _pool_flops_compute)
    F.adaptive_avg_pool3d = wrapFunc(F.adaptive_avg_pool3d, _pool_flops_compute)
    F.adaptive_max_pool1d = wrapFunc(F.adaptive_max_pool1d, _pool_flops_compute)
    F.adaptive_max_pool2d = wrapFunc(F.adaptive_max_pool2d, _pool_flops_compute)
    F.adaptive_max_pool3d = wrapFunc(F.adaptive_max_pool3d, _pool_flops_compute)

    # upsample
    F.upsample = wrapFunc(F.upsample, _upsample_flops_compute)
    F.interpolate = wrapFunc(F.interpolate, _upsample_flops_compute)

    # softmax
    F.softmax = wrapFunc(F.softmax, _softmax_flops_compute)

    # embedding
    F.embedding = wrapFunc(F.embedding, _embedding_flops_compute)


def _patch_tensor_methods():
    torch.matmul = wrapFunc(torch.matmul, _matmul_flops_compute)
    torch.Tensor.matmul = wrapFunc(torch.Tensor.matmul, _matmul_flops_compute)
    torch.mm = wrapFunc(torch.mm, _matmul_flops_compute)
    torch.Tensor.mm = wrapFunc(torch.Tensor.mm, _matmul_flops_compute)
    torch.bmm = wrapFunc(torch.bmm, _matmul_flops_compute)
    torch.Tensor.bmm = wrapFunc(torch.bmm, _matmul_flops_compute)

    torch.addmm = wrapFunc(torch.addmm, _addmm_flops_compute)
    torch.Tensor.addmm = wrapFunc(torch.Tensor.addmm, _tensor_addmm_flops_compute)

    torch.mul = wrapFunc(torch.mul, _mul_flops_compute)
    torch.Tensor.mul = wrapFunc(torch.Tensor.mul, _mul_flops_compute)

    torch.add = wrapFunc(torch.add, _add_flops_compute)
    torch.Tensor.add = wrapFunc(torch.Tensor.add, _add_flops_compute)

    torch.einsum = wrapFunc(torch.einsum, _einsum_flops_compute)


def _reload_functionals():
    # torch.nn.functional does not support importlib.reload()
    F.linear = old_functions[F.linear.__name__]
    F.conv1d = old_functions[F.conv1d.__name__]
    F.conv2d = old_functions[F.conv2d.__name__]
    F.conv3d = old_functions[F.conv3d.__name__]
    F.conv_transpose1d = old_functions[F.conv_transpose1d.__name__]
    F.conv_transpose2d = old_functions[F.conv_transpose2d.__name__]
    F.conv_transpose3d = old_functions[F.conv_transpose3d.__name__]
    F.relu = old_functions[F.relu.__name__]
    F.prelu = old_functions[F.prelu.__name__]
    F.elu = old_functions[F.elu.__name__]
    F.leaky_relu = old_functions[F.leaky_relu.__name__]
    F.relu6 = old_functions[F.relu6.__name__]
    F.batch_norm = old_functions[F.batch_norm.__name__]
    F.avg_pool1d = old_functions[F.avg_pool1d.__name__]
    F.avg_pool2d = old_functions[F.avg_pool2d.__name__]
    F.avg_pool3d = old_functions[F.avg_pool3d.__name__]
    F.max_pool1d = old_functions[F.max_pool1d.__name__]
    F.max_pool2d = old_functions[F.max_pool2d.__name__]
    F.max_pool3d = old_functions[F.max_pool3d.__name__]
    F.adaptive_avg_pool1d = old_functions[F.adaptive_avg_pool1d.__name__]
    F.adaptive_avg_pool2d = old_functions[F.adaptive_avg_pool2d.__name__]
    F.adaptive_avg_pool3d = old_functions[F.adaptive_avg_pool3d.__name__]
    F.adaptive_max_pool1d = old_functions[F.adaptive_max_pool1d.__name__]
    F.adaptive_max_pool2d = old_functions[F.adaptive_max_pool2d.__name__]
    F.adaptive_max_pool3d = old_functions[F.adaptive_max_pool3d.__name__]
    F.upsample = old_functions[F.upsample.__name__]
    F.interpolate = old_functions[F.interpolate.__name__]
    F.softmax = old_functions[F.softmax.__name__]
    F.embedding = old_functions[F.embedding.__name__]


def _reload_tensor_methods():
    torch.matmul = old_functions[torch.matmul.__name__]


def _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size * 3
        # last two hadamard _product and add
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size * 4
        # two hadamard _product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def _rnn_forward_hook(rnn_module, input, output):
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__("weight_ih_l" + str(i))
        w_hh = rnn_module.__getattr__("weight_hh_l" + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__("bias_ih_l" + str(i))
            b_hh = rnn_module.__getattr__("bias_hh_l" + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def _rnn_cell_forward_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = inp.shape[1]
    flops = _rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    # RNN
    nn.RNN: _rnn_forward_hook,
    nn.GRU: _rnn_forward_hook,
    nn.LSTM: _rnn_forward_hook,
    nn.RNNCell: _rnn_cell_forward_hook,
    nn.LSTMCell: _rnn_cell_forward_hook,
    nn.GRUCell: _rnn_cell_forward_hook,
}




def duration_to_string(duration, units=None, precision=2):
    if units is None:
        if duration > 1:
            return str(round(duration, precision)) + " s"
        elif duration * 10**3 > 1:
            return str(round(duration * 10**3, precision)) + " ms"
        elif duration * 10**6 > 1:
            return str(round(duration * 10**6, precision)) + " us"
        else:
            return str(duration)
    else:
        if units == "us":
            return str(round(duration * 10.0**6, precision)) + " " + units
        elif units == "ms":
            return str(round(duration * 10.0**3, precision)) + " " + units
        else:
            return str(round(duration, precision)) + " s"


    # can not iterate over all submodules using self.model.modules()
    # since modules() returns duplicate modules only once
def get_module_flops(module):
    sum = module.__flops__
    # iterate over immediate children modules
    for child in module.children():
        sum += get_module_flops(child)
    return sum


def get_module_macs(module):
    sum = module.__macs__
    # iterate over immediate children modules
    for child in module.children():
        sum += get_module_macs(child)
    return sum


def get_module_duration(module):
    duration = module.__duration__
    if duration == 0:  # e.g. ModuleList
        for m in module.children():
            duration += m.__duration__
    return duration



def num_to_string(num, precision=2):
    if num // 10**9 > 0:
        return str(round(num / 10.0**9, precision)) + " G"
    elif num // 10**6 > 0:
        return str(round(num / 10.0**6, precision)) + " M"
    elif num // 10**3 > 0:
        return str(round(num / 10.0**3, precision)) + " K"
    else:
        return str(num)


def macs_to_string(macs, units=None, precision=2):
    if units is None:
        if macs // 10**9 > 0:
            return str(round(macs / 10.0**9, precision)) + " GMACs"
        elif macs // 10**6 > 0:
            return str(round(macs / 10.0**6, precision)) + " MMACs"
        elif macs // 10**3 > 0:
            return str(round(macs / 10.0**3, precision)) + " KMACs"
        else:
            return str(macs) + " MACs"
    else:
        if units == "GMACs":
            return str(round(macs / 10.0**9, precision)) + " " + units
        elif units == "MMACs":
            return str(round(macs / 10.0**6, precision)) + " " + units
        elif units == "KMACs":
            return str(round(macs / 10.0**3, precision)) + " " + units
        else:
            return str(macs) + " MACs"


def number_to_string(num, units=None, precision=2):
    if units is None:
        if num // 10**9 > 0:
            return str(round(num / 10.0**9, precision)) + " G"
        elif num // 10**6 > 0:
            return str(round(num / 10.0**6, precision)) + " M"
        elif num // 10**3 > 0:
            return str(round(num / 10.0**3, precision)) + " K"
        else:
            return str(num) + " "
    else:
        if units == "G":
            return str(round(num / 10.0**9, precision)) + " " + units
        elif units == "M":
            return str(round(num / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(num / 10.0**3, precision)) + " " + units
        else:
            return str(num) + " "


def flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**12 > 0:
            return str(round(flops / 10.0**12, precision)) + " TFLOPS"
        if flops // 10**9 > 0:
            return str(round(flops / 10.0**9, precision)) + " GFLOPS"
        elif flops // 10**6 > 0:
            return str(round(flops / 10.0**6, precision)) + " MFLOPS"
        elif flops // 10**3 > 0:
            return str(round(flops / 10.0**3, precision)) + " KFLOPS"
        else:
            return str(flops) + " FLOPS"
    else:
        if units == "TFLOPS":
            return str(round(flops / 10.0**12, precision)) + " " + units
        if units == "GFLOPS":
            return str(round(flops / 10.0**9, precision)) + " " + units
        elif units == "MFLOPS":
            return str(round(flops / 10.0**6, precision)) + " " + units
        elif units == "KFLOPS":
            return str(round(flops / 10.0**3, precision)) + " " + units
        else:
            return str(flops) + " FLOPS"


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10**6 > 0:
            return str(round(params_num / 10**6, 2)) + " M"
        elif params_num // 10**3:
            return str(round(params_num / 10**3, 2)) + " k"
        else:
            return str(params_num)
    else:
        if units == "M":
            return str(round(params_num / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(params_num / 10.0**3, precision)) + " " + units
        else:
            return str(params_num)
