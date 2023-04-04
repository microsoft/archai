from typing import Optional

import torch
from torch import nn

from transformers.models.reformer.modeling_reformer import ReformerConfig
from .lsh_utils.modeling_reformer import ReformerAttention

from archai.discrete_search.search_spaces.config import ArchConfig


class LSHAttention(nn.Module):
    def __init__(self, arch_config: ArchConfig, hidden_size: int, total_heads: int,
                 op_heads: int, auto_pick_num_buckets: bool = True, autopad: bool = True, 
                 **kwargs):
        assert hidden_size % total_heads == 0, 'hidden size must be divisible by total heads'
        super().__init__()

        self.hidden_size = hidden_size
        self.total_heads = total_heads
        self.op_heads = op_heads
        self.op_size = (self.hidden_size // self.total_heads) * self.op_heads
        
        self.num_hashes = arch_config.pick('num_hashes')
        self.bucket_size = arch_config.pick('bucket_size')
        self.num_buckets = arch_config.pick('num_buckets') if not auto_pick_num_buckets else None
        self.autopad = autopad

        self.config = ReformerConfig(
            attn_layers=['lsh'],
            hidden_size=hidden_size,
            num_attention_heads=op_heads,
            num_hashes=self.num_hashes,
            lsh_attn_chunk_length=self.bucket_size,
            num_buckets=self.num_buckets,
            attention_head_size=(self.hidden_size // self.total_heads),
            axial_pos_embds=False,
            is_decoder=True,
            use_cache=False
        )

        self.attn = ReformerAttention(self.config)
        
        # Overrides the output layer to be identity to make it
        # return an output of `op_size` instead of `hidden_size`
        self.attn.output = nn.Identity()
        

    def forward(self, hidden_states, bin_attention_mask: Optional[torch.FloatTensor] = None, 
                past_buckets_states: Optional[torch.Tensor] = None,  use_cache: bool = False,
                *args, **kwargs):
        seq_len = hidden_states.size(1)

        # Pads input to be divisible by bucket size
        if self.autopad and seq_len % self.bucket_size != 0:
            pad_size = (self.bucket_size - seq_len % self.bucket_size) % self.bucket_size
            
            # Pads hidden states and attention mask with zeros so attn is not computed for padded tokens
            p_hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, pad_size, 0))
            p_bin_attention_mask = torch.nn.functional.pad(bin_attention_mask, (pad_size, 0))

            # Computes attention with padded input and unpads output
            output = self.attn(p_hidden_states, attention_mask=p_bin_attention_mask)
            return output[0][:, pad_size:], output[1:]
        
        return self.attn(hidden_states, attention_mask=bin_attention_mask)
