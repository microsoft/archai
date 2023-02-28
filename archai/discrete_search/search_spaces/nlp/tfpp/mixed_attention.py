from typing import Optional

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D
from transformers import PretrainedConfig

from archai.discrete_search.search_spaces.config import ArchConfig
from archai.discrete_search.search_spaces.nlp.tfpp.ops import OPS


class MixedAttentionBlock(nn.Module):
    def __init__(self, arch_config: ArchConfig, hf_config: PretrainedConfig,
                 hidden_size: int, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        
        self.total_heads = arch_config.pick('total_heads')
        self.op_allocation = {
            op_name: round(self.total_heads * op_prop)
            for op_name, op_prop in arch_config.pick('op_allocation')
        }

        self.hf_config = hf_config
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.head_size = hidden_size // self.total_heads

        assert hidden_size % self.total_heads == 0
        assert sum(list(self.op_allocation.values())) == self.total_heads, \
            'Invalid allocation'
        
        op_kwargs = {
            'hidden_size': self.hidden_size, 
            'total_heads': self.total_heads,
            'hf_config': self.hf_config,
            'layer_idx': self.layer_idx
        }
        
        self.ops = nn.ModuleList([
            OPS[op_name].cls(
                arch_config=arch_config.pick(op_name) if OPS[op_name].requires_extra_config else None,
                op_heads=self.op_allocation[op_name],
                **op_kwargs
            ) for op_name, op_heads in self.op_allocation.items()
            if op_heads > 0
        ])

        self.resid_dropout = nn.Dropout(self.hf_config.resid_pdrop)
        self.out_proj = Conv1D(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # Concatenates outputs from each op in the embedding dim
        output = [op(hidden_states, **kwargs)[0] for op in self.ops]
        output = torch.cat(output, dim=-1)

        # TODO: return present values
        return self.resid_dropout(self.out_proj(output)), None
