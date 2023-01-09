# Copyright (c) DeepSpeed Team - Microsoft Corporation.
# Licensed under the MIT License.
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler/profiler.py


from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

FLOPS = []
MACS = []
TORCH_FUNCTIONS = {}


def __shape_inner_product(dims: Tuple[int, ...]) -> int:
    p = 1
    for v in dims:
        p *= v

    return p


def _linear_hook(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    out_features = weight.shape[0]
    macs = torch.numel(input) * out_features

    return 2 * macs, macs


def _relu_hook(input: torch.Tensor, inplace: Optional[bool] = False) -> Tuple[int, int]:
    return torch.numel(input), 0


def _prelu_hook(input: torch.Tensor, weight: torch.Tensor) -> Tuple[int, int]:
    return torch.numel(input), 0


def _elu_hook(input: torch.Tensor, alpha: Optional[float] = 1.0, inplace: Optional[bool] = False) -> Tuple[int, int]:
    return torch.numel(input), 0


def _leakyrelu_hook(
    input: torch.Tensor, negative_slope: Optional[float] = 0.01, inplace: Optional[bool] = False
) -> Tuple[int, int]:
    return torch.numel(input), 0


def _relu6_hook(input: torch.Tensor, inplace: Optional[bool] = False) -> Tuple[int, int]:
    return torch.numel(input), 0


def _silu_hook(input: torch.Tensor, inplace: Optional[bool] = False) -> Tuple[int, int]:
    return torch.numel(input), 0


def _gelu_hook(input: torch.Tensor, approximate: str = 'none') -> Tuple[int, int]:
    return torch.numel(input), 0


def _pool_hook(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Optional[int] = 0,
    dilation: Optional[int] = None,
    ceil_mode: Optional[bool] = False,
    count_include_pad: Optional[bool] = True,
    divisor_override: Optional[int] = None,
    return_indices: Optional[bool] = None,
) -> Tuple[int, int]:
    return torch.numel(input), 0


def _conv_hook(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Optional[Union[int, Tuple[int, ...]]] = 1,
    padding: Optional[Union[int, str]] = 0,
    dilation: Optional[Union[int, Tuple[int, ...]]] = 1,
    groups: Optional[int] = 1,
) -> Tuple[int, int]:
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]

    in_channels = input.shape[1]
    out_channels = weight.shape[0]

    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding,) * length
    strides = stride if type(stride) is tuple else (stride,) * length
    dilations = dilation if type(dilation) is tuple else (dilation,) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(__shape_inner_product(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(__shape_inner_product(output_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _conv_transpose_hook(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Optional[Union[int, Tuple[int, ...]]] = 1,
    padding: Optional[Union[int, str]] = 0,
    output_padding: Optional[int] = 0,
    dilation: Optional[Union[int, Tuple[int, ...]]] = 1,
    groups: Optional[int] = 1,
) -> Tuple[int, int]:
    batch_size = input.shape[0]

    in_channels = input.shape[1]
    out_channels = weight.shape[0]

    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding,) * length
    strides = stride if type(stride) is tuple else (stride,) * length
    dilations = dilation if type(dilation) is tuple else (dilation,) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (input_dim + 2 * paddings[idx] - (dilations[idx] * (kernel_dims[idx] - 1) + 1)) // strides[idx] + 1
        output_dims.append(output_dim)

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(__shape_inner_product(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(__shape_inner_product(input_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(__shape_inner_product(output_dims))

    return int(overall_conv_flops + bias_flops), int(overall_conv_macs)


def _batch_norm_hook(
    input: torch.Tensor,
    running_mean: Optional[torch.Tensor] = None,
    running_var: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: Optional[bool] = False,
    momentum: Optional[float] = 0.1,
    eps: Optional[float] = 1e-05,
) -> Tuple[int, int]:
    has_affine = weight is not None

    if training:
        return torch.numel(input) * (5 if has_affine else 4), 0

    flops = torch.numel(input) * (2 if has_affine else 1)

    return flops, 0


def _layer_norm_hook(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-5,
) -> Tuple[int, int]:
    has_affine = weight is not None

    return torch.numel(input) * (5 if has_affine else 4), 0


def _instance_norm_hook(
    input: torch.Tensor,
    running_mean: Optional[torch.Tensor] = None,
    running_var: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_input_stats: Optional[bool] = True,
    momentum: Optional[float] = 0.1,
    eps: Optional[float] = 1e-5,
) -> Tuple[int, int]:
    has_affine = weight is not None

    return torch.numel(input) * (5 if has_affine else 4), 0


def _group_norm_hook(
    input: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-5,
) -> Tuple[int, int]:
    has_affine = weight is not None

    return torch.numel(input) * (5 if has_affine else 4), 0


def _upsample_hook(
    input: torch.Tensor,
    size: Optional[Union[int, Tuple[int, ...]]] = None,
    scale_factor: Optional[Union[float, Tuple[float]]] = None,
    mode: Optional[str] = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None
) -> Tuple[int, int]:
    if size is not None:
        if isinstance(size, tuple):
            return int(__shape_inner_product(size)), 0
        else:
            return int(size), 0

    assert scale_factor is not None, "Either `size` or `scale_factor` should be defined."

    flops = torch.numel(input)
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops * int(__shape_inner_product(scale_factor))
    else:
        flops * scale_factor ** len(input)

    return flops, 0


def _softmax_hook(
    input: torch.Tensor, dim: Optional[int] = None, _stacklevel: Optional[int] = 3, dtype: Optional[torch.dtype] = None
) -> Tuple[int, int]:
    return torch.numel(input), 0


def _embedding_hook(
    input: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: Optional[float] = 2.0,
    scale_grad_by_freq: Optional[bool] = False,
    sparse: Optional[bool] = False,
) -> Tuple[int, int]:
    return 0, 0


def _matmul_hook(input: torch.Tensor, other: torch.Tensor, *, out: Optional[Tuple[int, ...]] = None) -> Tuple[int, int]:
    macs = __shape_inner_product(input.shape) * other.shape[-1]

    return 2 * macs, macs


def _addmm_hook(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: Optional[int] = 1,
    alpha: Optional[int] = 1,
    out: Optional[Tuple[int, ...]] = None
) -> Tuple[int, int]:
    macs = __shape_inner_product(mat1.shape) * mat2.shape[-1]

    return 2 * macs + __shape_inner_product(input.shape), macs


def _einsum_hook(equation: str, *operands) -> Tuple[int, int]:
    equation = equation.replace(" ", "")

    # Fix for `opt_einsum.contract`
    if len(operands) == 1 and isinstance(operands[0], tuple):
        operands = operands[0]
        
    input_shapes = [o.shape for o in operands]

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


def __elementwise_hook(input: torch.Tensor, other: torch.Tensor) -> Tuple[int, int]:
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return __shape_inner_product(other.shape), 0
        else:
            return 1, 0

    elif not torch.is_tensor(other):
        return __shape_inner_product(input.shape), 0

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

        flops = __shape_inner_product(final_shape)

        return flops, 0


def _mul_hook(input: torch.Tensor, other: torch.Tensor, *, out: Optional[Tuple[int, ...]] = None) -> Tuple[int, int]:
    return __elementwise_hook(input, other)


def _add_hook(
    input: torch.Tensor, other: torch.Tensor, *, alpha: Optional[int] = 1, out: Optional[Tuple[int, ...]] = None
) -> Tuple[int, int]:
    return __elementwise_hook(input, other)


def _wrap_fn(fn: Callable, new_fn: Callable) -> Callable:
    """Wraps a function with another function.

    Args:
        fn: Current function.
        new_fn: New function.

    Returns:
        (Callable): Wrapped function.

    """

    old_fn = fn
    name = fn.__name__
    TORCH_FUNCTIONS[name] = old_fn

    def __wrap_fn(*args, **kwargs):
        flops, macs = new_fn(*args, **kwargs)

        if FLOPS:
            FLOPS[-1].append((name, flops))
        if MACS and macs:
            MACS[-1].append((name, macs))

        return old_fn(*args, **kwargs)

    __wrap_fn.__name__ = fn.__name__

    return __wrap_fn


def enable_functional_hooks() -> None:
    """Enables functional API profiler hooks."""

    F.linear = _wrap_fn(F.linear, _linear_hook)

    F.conv1d = _wrap_fn(F.conv1d, _conv_hook)
    F.conv2d = _wrap_fn(F.conv2d, _conv_hook)
    F.conv3d = _wrap_fn(F.conv3d, _conv_hook)
    F.conv_transpose1d = _wrap_fn(F.conv_transpose1d, _conv_transpose_hook)
    F.conv_transpose2d = _wrap_fn(F.conv_transpose2d, _conv_transpose_hook)
    F.conv_transpose3d = _wrap_fn(F.conv_transpose3d, _conv_transpose_hook)

    F.relu = _wrap_fn(F.relu, _relu_hook)
    F.prelu = _wrap_fn(F.prelu, _prelu_hook)
    F.elu = _wrap_fn(F.elu, _elu_hook)
    F.leaky_relu = _wrap_fn(F.leaky_relu, _leakyrelu_hook)
    F.relu6 = _wrap_fn(F.relu6, _relu6_hook)
    if hasattr(F, "silu"):
        F.silu = _wrap_fn(F.silu, _silu_hook)
    F.gelu = _wrap_fn(F.gelu, _gelu_hook)

    F.batch_norm = _wrap_fn(F.batch_norm, _batch_norm_hook)
    F.layer_norm = _wrap_fn(F.layer_norm, _layer_norm_hook)
    F.instance_norm = _wrap_fn(F.instance_norm, _instance_norm_hook)
    F.group_norm = _wrap_fn(F.group_norm, _group_norm_hook)

    F.avg_pool1d = _wrap_fn(F.avg_pool1d, _pool_hook)
    F.avg_pool2d = _wrap_fn(F.avg_pool2d, _pool_hook)
    F.avg_pool3d = _wrap_fn(F.avg_pool3d, _pool_hook)
    F.max_pool1d = _wrap_fn(F.max_pool1d, _pool_hook)
    F.max_pool2d = _wrap_fn(F.max_pool2d, _pool_hook)
    F.max_pool3d = _wrap_fn(F.max_pool3d, _pool_hook)
    F.adaptive_avg_pool1d = _wrap_fn(F.adaptive_avg_pool1d, _pool_hook)
    F.adaptive_avg_pool2d = _wrap_fn(F.adaptive_avg_pool2d, _pool_hook)
    F.adaptive_avg_pool3d = _wrap_fn(F.adaptive_avg_pool3d, _pool_hook)
    F.adaptive_max_pool1d = _wrap_fn(F.adaptive_max_pool1d, _pool_hook)
    F.adaptive_max_pool2d = _wrap_fn(F.adaptive_max_pool2d, _pool_hook)
    F.adaptive_max_pool3d = _wrap_fn(F.adaptive_max_pool3d, _pool_hook)

    F.upsample = _wrap_fn(F.upsample, _upsample_hook)
    F.interpolate = _wrap_fn(F.interpolate, _upsample_hook)
    F.softmax = _wrap_fn(F.softmax, _softmax_hook)
    F.embedding = _wrap_fn(F.embedding, _embedding_hook)


def disable_functional_hooks() -> None:
    """Disables functional API profiler hooks."""

    F.linear = TORCH_FUNCTIONS[F.linear.__name__]

    F.conv1d = TORCH_FUNCTIONS[F.conv1d.__name__]
    F.conv2d = TORCH_FUNCTIONS[F.conv2d.__name__]
    F.conv3d = TORCH_FUNCTIONS[F.conv3d.__name__]
    F.conv_transpose1d = TORCH_FUNCTIONS[F.conv_transpose1d.__name__]
    F.conv_transpose2d = TORCH_FUNCTIONS[F.conv_transpose2d.__name__]
    F.conv_transpose3d = TORCH_FUNCTIONS[F.conv_transpose3d.__name__]

    F.relu = TORCH_FUNCTIONS[F.relu.__name__]
    F.prelu = TORCH_FUNCTIONS[F.prelu.__name__]
    F.elu = TORCH_FUNCTIONS[F.elu.__name__]
    F.leaky_relu = TORCH_FUNCTIONS[F.leaky_relu.__name__]
    F.relu6 = TORCH_FUNCTIONS[F.relu6.__name__]

    F.batch_norm = TORCH_FUNCTIONS[F.batch_norm.__name__]
    F.layer_norm = TORCH_FUNCTIONS[F.layer_norm.__name__]
    F.instance_norm = TORCH_FUNCTIONS[F.instance_norm.__name__]
    F.group_norm = TORCH_FUNCTIONS[F.group_norm.__name__]

    F.avg_pool1d = TORCH_FUNCTIONS[F.avg_pool1d.__name__]
    F.avg_pool2d = TORCH_FUNCTIONS[F.avg_pool2d.__name__]
    F.avg_pool3d = TORCH_FUNCTIONS[F.avg_pool3d.__name__]
    F.max_pool1d = TORCH_FUNCTIONS[F.max_pool1d.__name__]
    F.max_pool2d = TORCH_FUNCTIONS[F.max_pool2d.__name__]
    F.max_pool3d = TORCH_FUNCTIONS[F.max_pool3d.__name__]
    F.adaptive_avg_pool1d = TORCH_FUNCTIONS[F.adaptive_avg_pool1d.__name__]
    F.adaptive_avg_pool2d = TORCH_FUNCTIONS[F.adaptive_avg_pool2d.__name__]
    F.adaptive_avg_pool3d = TORCH_FUNCTIONS[F.adaptive_avg_pool3d.__name__]
    F.adaptive_max_pool1d = TORCH_FUNCTIONS[F.adaptive_max_pool1d.__name__]
    F.adaptive_max_pool2d = TORCH_FUNCTIONS[F.adaptive_max_pool2d.__name__]
    F.adaptive_max_pool3d = TORCH_FUNCTIONS[F.adaptive_max_pool3d.__name__]

    F.upsample = TORCH_FUNCTIONS[F.upsample.__name__]
    F.interpolate = TORCH_FUNCTIONS[F.interpolate.__name__]
    F.softmax = TORCH_FUNCTIONS[F.softmax.__name__]
    F.embedding = TORCH_FUNCTIONS[F.embedding.__name__]


def enable_tensor_hooks() -> None:
    """Enables tensor-based operations profiler hooks."""

    torch.matmul = _wrap_fn(torch.matmul, _matmul_hook)
    torch.mm = _wrap_fn(torch.mm, _matmul_hook)
    torch.bmm = _wrap_fn(torch.bmm, _matmul_hook)
    torch.addmm = _wrap_fn(torch.addmm, _addmm_hook)
    torch.mul = _wrap_fn(torch.mul, _mul_hook)
    torch.add = _wrap_fn(torch.add, _add_hook)
    torch.einsum = _wrap_fn(torch.einsum, _einsum_hook)


def disable_tensor_hooks() -> None:
    """Disables tensor-based operations profiler hooks."""

    torch.matmul = TORCH_FUNCTIONS[torch.matmul.__name__]
    torch.mm = TORCH_FUNCTIONS[torch.mm.__name__]
    torch.bmm = TORCH_FUNCTIONS[torch.bmm.__name__]
    torch.addmm = TORCH_FUNCTIONS[torch.addmm.__name__]
    torch.mul = TORCH_FUNCTIONS[torch.mul.__name__]
    torch.add = TORCH_FUNCTIONS[torch.add.__name__]
    torch.einsum = TORCH_FUNCTIONS[torch.einsum.__name__]
