# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn

import opacus
from opacus.grad_sample.grad_sample_module import create_or_accumulate_grad_sample, promote_current_grad_sample

def _grad_sampler_get_attr(self, item):
    try:
        return super(opacus.GradSampleModule, self).__getattr__(item)
    except AttributeError as e:
        submodules = dict(self._module.named_modules())
        if item and item in submodules:
            return submodules[item]

        # Enable the Opacus wrapped model to access internal/ArchaiModel
        # model attributes
        if hasattr(self._module, item):
            return getattr(self._module, item)
        
        raise e

def _grad_sample_capture_backprops_hook(
    self,
    module: nn.Module,
    _forward_input: torch.Tensor,
    forward_output: torch.Tensor,
    loss_reduction: str,
    batch_first: bool,
):
    """
    Computes per sample gradients given the current backprops and activations
    stored by the associated forward hook. Computed per sample gradients are
    stored in ``grad_sample`` field in each parameter.

    For non-recurrent layers the process is straightforward: for each
    ``loss.backward()`` call this hook will be called exactly one. For recurrent
    layers, however, this is more complicated and the hook will be called multiple
    times, while still processing the same batch of data.

    For this reason we first accumulate the gradients from *the same batch* in
    ``p._current_grad_sample`` and then, when we detect the end of a full backward
    pass - we store accumulated result on ``p.grad_sample``.

    From there, ``p.grad_sample`` could be either a Tensor or a list of Tensors,
    if accumulated over multiple batches

    This function was patched to work with weight-sharing.

    Args:
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    """
    if not self.hooks_enabled:
        return

    backprops = forward_output[0].detach()
    activations, backprops = self.rearrange_grad_samples(
        module=module,
        backprops=backprops,
        loss_reduction=loss_reduction,
        batch_first=batch_first,
    )
    grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
    grad_samples = grad_sampler_fn(module, activations, backprops)

    for param, gs in grad_samples.items():
        create_or_accumulate_grad_sample(param=param, grad_sample=gs, layer=module)

    # We never update layers marked as shared and hence
    # we do not need a max_batch_len for them
    if hasattr(module, 'shared') and hasattr(module, "max_batch_len"):
        del module.max_batch_len

    # Never promote (save) layers with the attr shared so that the
    # gradient is always accumulated
    if len(module.activations) == 0 and not hasattr(module, 'shared'):
        if hasattr(module, "max_batch_len"):
            del module.max_batch_len

        for p in module.parameters():
            promote_current_grad_sample(p)

opacus.GradSampleModule.__getattr__ = _grad_sampler_get_attr
opacus.grad_sample.grad_sample_module.GradSampleModule.capture_backprops_hook = _grad_sample_capture_backprops_hook