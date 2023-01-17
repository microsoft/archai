import torch
from typing import Optional, Tuple, Union
from torch import nn

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from archai.discrete_search.search_spaces.config import ArchConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, arch_config: ArchConfig, hf_config: GPT2Config, hidden_size: int,
                 total_heads: int, op_heads: int, layer_idx=None, **kwargs):
        super().__init__()

        max_positions = hf_config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.hidden_size = hidden_size
        self.total_heads = total_heads
        self.op_heads = op_heads
        self.op_size = (self.hidden_size // total_heads) * op_heads
        self.max_positions = max_positions
        # self.attn_window_len = int(arch_config.pick('attn_window_prop') * hf_config.n_positions)

        # if self.attn_window_len < self.max_positions:
        #     print('Using causal attention window of length', self.attn_window_len)
        # else:
        #     print('Using full attention window')
        
        self.scale_attn_weights = hf_config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = hf_config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = hf_config.reorder_and_upcast_attn

        self.embed2qk = nn.Linear(self.hidden_size, 2 * self.op_size)
        self.embed2v = nn.Linear(self.hidden_size, self.op_size)

        self.attn_dropout = nn.Dropout(hf_config.attn_pdrop)
        self.resid_dropout = nn.Dropout(hf_config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.tensor(
                value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

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

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # Splits input according to attention window length
        # if self.attn_window_len < self.max_positions:
        #     outside_states = self.embed2v(hidden_states[:, : -self.attn_window_len, :])
        #     hidden_states = hidden_states[:, -self.attn_window_len :, :]
            
        #     if attention_mask is not None:
        #         attention_mask = attention_mask[..., -self.attn_window_len :]

        query, key = self.embed2qk(hidden_states).split(self.op_size, dim=-1)
        value = self.embed2v(hidden_states)
        
        head_size = (self.hidden_size // self.total_heads)
        query = self._split_heads(query, self.op_heads, head_size)
        key = self._split_heads(key, self.op_heads, head_size)
        value = self._split_heads(value, self.op_heads, head_size)

        attn_output, _ = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.op_heads, head_size)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # Re-concatenates truncated input to attention window
        # if self.attn_window_len < self.max_positions:
        #     attn_output = torch.cat([outside_states, attn_output], dim=1)

        return attn_output, present
