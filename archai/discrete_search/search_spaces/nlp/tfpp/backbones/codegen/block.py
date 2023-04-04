from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PretrainedConfig
from archai.discrete_search.search_spaces.config import ArchConfig

try:
    from flash_attn.modules.mlp import FusedMLP
except ImportError:
    FusedMLP = None

from ...utils import get_optim_flag
from ...mixed_op import MixedAttentionBlock


# From https://github.com/HazyResearch/flash-attention (Copyright Tri Dao)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation=nn.functional.gelu,
                 return_residual=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class CodeGenBlock(nn.Module):
    def __init__(self, arch_config: ArchConfig, hf_config: PretrainedConfig,
                 hidden_size: int, layer_idx: Optional[int] = None):
        super().__init__()

        self.inner_dim = arch_config.pick('d_inner')
        self.attn = MixedAttentionBlock(arch_config, hf_config, hidden_size, layer_idx=layer_idx)
        self.fused_mlp = get_optim_flag(hf_config, 'fused_mlp')
        
        if self.fused_mlp:
            assert FusedMLP is not None, 'Need to install fused_mlp'
            self.mlp = FusedMLP(hidden_size, self.inner_dim)
        else:
            self.mlp = Mlp(hidden_size, self.inner_dim)

        self.resid_dropout = nn.Dropout(hf_config.resid_pdrop)
        self.norm = nn.LayerNorm(hidden_size)
        
        if getattr(hf_config, 'fused_dropout_add_ln', False):
            raise NotImplementedError

    def forward(self, hidden_states: Tensor, mixer_subset=None, mixer_kwargs=None, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        mixer_kwargs = mixer_kwargs or {}
        mixer_kwargs.update(**kwargs)

        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        
        attn_output, _ = self.attn(hidden_states, **mixer_kwargs)
        attn_output = self.resid_dropout(attn_output)
        mlp_output = self.resid_dropout(self.mlp(hidden_states))
        
        return residual + attn_output + mlp_output
