# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from opacus.grad_sample import register_grad_sampler
from transformers.modeling_utils import Conv1D

from archai.nlp.compression.quantization.modules import FakeDynamicQuantHFConv1DForOnnx, FakeDynamicQuantLinearForOnnx, FakeQuantEmbeddingForOnnx


@register_grad_sampler(FakeDynamicQuantHFConv1DForOnnx)
def compute_quant_transformers_conv1d_grad_sample(
    layer: FakeDynamicQuantHFConv1DForOnnx, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0):
    gs = (torch.einsum("n...i,n...j->nji", B, A) * layer.weight_fake_quant.mask).contiguous()
    ret = {layer.weight: gs}

    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", B)

    return ret

@register_grad_sampler(FakeDynamicQuantLinearForOnnx)
def compute_quant_lin_grad_sample(layer: FakeDynamicQuantLinearForOnnx, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0):

    gs = torch.einsum("n...i,n...j->nij", B, A) * layer.weight_fake_quant.mask
    ret = {layer.weight: gs}

    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", B)

    return ret

@register_grad_sampler(FakeQuantEmbeddingForOnnx)
def compute_quant_emb_grad_sample(layer: FakeQuantEmbeddingForOnnx, activations: torch.Tensor, backprops: torch.Tensor, batch_dim: int = 0):

    saved = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True

    batch_size = activations.shape[0]
    index = (
        activations.unsqueeze(-1)
        .expand(*activations.shape, layer.embedding_dim)
        .reshape(batch_size, -1, layer.embedding_dim)
    )
    grad_sample = torch.zeros(
        batch_size, *layer.weight.shape, device=layer.weight.device
    )
    grad_sample.scatter_add_(
        1, index, backprops.reshape(batch_size, -1, layer.embedding_dim)
    )

    grad_sample *= layer.weight_fake_quant.mask
    torch.backends.cudnn.deterministic = saved

    return {layer.weight: grad_sample}

@register_grad_sampler(Conv1D)
def compute_transformers_conv1d_grad_sample(layer: Conv1D, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0):
    gs = torch.einsum("n...i,n...j->nji", B, A).contiguous()
    ret = {layer.weight: gs}

    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", B)

    return ret