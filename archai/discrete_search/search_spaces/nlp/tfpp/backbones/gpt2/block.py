import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from transformers.activations import ACT2FN
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from archai.discrete_search.search_spaces.config import ArchConfig

try:
    from flash_attn.modules.mlp import FusedMLP
except ImportError:
    FusedMLP = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

from ...utils import get_optim_flag
from ...mixed_op import MixedAttentionBlock


class GPT2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation=nn.functional.gelu):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, hidden_size)
        self.act = activation

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, arch_config: ArchConfig, hf_config: GPT2Config,
                 hidden_size, resid_dropout1: float = 0.0, resid_dropout2: float = 0.0,
                 layer_idx=None):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(hidden_size, eps=hf_config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(hidden_size, eps=hf_config.layer_norm_epsilon)
        self.attn = MixedAttentionBlock(arch_config, hf_config, hidden_size, layer_idx=layer_idx)
        
        self.inner_dim = arch_config.pick('d_inner')
        self.fused_mlp = get_optim_flag(hf_config, 'fused_mlp')

        if self.fused_mlp:
            assert FusedMLP is not None, 'Need to install fused_mlp'
            self.mlp = FusedMLP(hidden_size, self.inner_dim)
        else:
            self.mlp = GPT2MLP(hidden_size, self.inner_dim)
        
        self.resid_dropout1 = nn.Dropout(resid_dropout1)
        self.resid_dropout2 = nn.Dropout(resid_dropout2)

        self.residual_in_fp32 = getattr(hf_config, 'residual_in_fp32', False)
        self.fused_dropout_add_ln = get_optim_flag(hf_config, 'fused_dropout_add_ln')
        
        if self.fused_dropout_add_ln:
            assert dropout_add_layer_norm is not None, 'dropout_add_ln is not installed'
            assert isinstance(self.ln1, nn.LayerNorm) and isinstance(self.resid_dropout1, nn.Dropout)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        residual: Optional[Tuple[torch.FloatTensor]],
        mixer_subset=None, mixer_kwargs=None, **kwargs
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

        if not self.fused_dropout_add_ln:
            dropped = self.resid_dropout1(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.ln1(residual.to(dtype=self.ln1.weight.dtype))

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = dropout_add_layer_norm(
                hidden_states, residual, self.ln1.weight, self.ln1.bias,
                self.resid_dropout1.p if self.training else 0.0, self.ln1.eps,
                rowscale=None, prenorm=True, residual_in_fp32=self.residual_in_fp32
            )

        hidden_states, _ = self.attn(hidden_states, **kwargs)

        if not self.fused_dropout_add_ln:
            dropped = self.resid_dropout2(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.ln2(residual.to(dtype=self.ln2.weight.dtype))
            
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = dropout_add_layer_norm(
                hidden_states, residual, self.ln2.weight, self.ln2.bias,
                self.resid_dropout2.p if self.training else 0.0, self.ln2.eps,
                rowscale=None, prenorm=True, residual_in_fp32=self.residual_in_fp32
            )

        return self.mlp(hidden_states), residual
