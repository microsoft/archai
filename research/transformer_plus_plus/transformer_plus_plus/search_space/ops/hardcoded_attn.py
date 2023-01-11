import torch
from torch import nn

from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class HardcodedAttention(nn.Module):
    def __init__(self, hf_config: GPT2Config, hidden_size: int,
                 total_heads: int, op_heads: int, **kwargs):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.total_heads = total_heads
        self.op_heads = op_heads
        self.head_size = (hidden_size // total_heads)

        # Splits between broadcast and association heads
        self.broad_heads = 0 # For now
        self.assoc_heads = op_heads - self.broad_heads
        self.v_proj = nn.Linear(hidden_size, self.op_heads * self.head_size)

        # Causal mask buffer
        max_position = hf_config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((max_position, max_position), dtype=torch.uint8)).view(
                1, 1, max_position, max_position
            ),
        )

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, hidden_states, assoc_hc_attn, broad_hc_attn, **kwargs):
        seq_len = hidden_states.size(1)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # Value (batch, seq, op_heads * head_size)
        value = self.v_proj(hidden_states) 
        
        # sp_value (batch, op_heads, seq, head_size)
        sp_value = self._split_heads(value, self.op_heads, self.head_size)

        attn_heads = torch.cat([
            broad_hc_attn.unsqueeze(1).repeat((1, self.broad_heads, 1, 1)),
            assoc_hc_attn.unsqueeze(1).repeat((1, self.assoc_heads, 1, 1))
        ], dim=1).float()

        zero_mask = torch.tensor([0.0], dtype=attn_heads.float().dtype).to(attn_heads.device)
        attn_heads = torch.where(causal_mask.bool(), attn_heads, zero_mask)

        # Re-normalize attn_heads
        attn_heads = attn_heads / (1e-6 + attn_heads.sum(dim=-1, keepdims=True))
        
        output = torch.matmul(attn_heads, sp_value)
        return self._merge_heads(output, self.op_heads, self.head_size), None
