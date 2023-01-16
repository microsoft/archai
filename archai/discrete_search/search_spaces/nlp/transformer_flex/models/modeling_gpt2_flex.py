# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP,
    GPT2Attention,
    GPT2Block,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2PreTrainedModel,
)
from transformers.pytorch_utils import Conv1D

from archai.discrete_search.search_spaces.nlp.transformer_flex.models.configuration_gpt2_flex import (
    GPT2FlexConfig,
)


class GPT2FlexAttention(GPT2Attention):
    def __init__(
        self,
        config: GPT2FlexConfig,
        is_cross_attention: Optional[bool] = False,
        layer_idx: Optional[int] = None,
    ) -> None:
        nn.Module.__init__(self)

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads[layer_idx]
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()


class GPT2FlexMLP(GPT2MLP):
    def __init__(self, intermediate_size: int, config: GPT2FlexConfig) -> None:
        nn.Module.__init__(self)

        embed_dim = config.hidden_size

        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

        self.primer_square = config.primer_square

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)

        if self.primer_square:
            hidden_states = hidden_states**2

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class GPT2FlexBlock(GPT2Block):
    def __init__(self, config: GPT2FlexConfig, layer_idx: Optional[int] = None) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        inner_dim = config.n_inner[layer_idx]

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2FlexAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2FlexAttention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2FlexMLP(inner_dim, config)


class GPT2FlexModel(GPT2Model):
    config_class = GPT2FlexConfig

    def __init__(self, config: GPT2FlexConfig) -> None:
        GPT2PreTrainedModel.__init__(self, config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2FlexBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        self.post_init()


class GPT2FlexLMHeadModel(GPT2LMHeadModel):
    config_class = GPT2FlexConfig

    def __init__(self, config: GPT2FlexConfig) -> None:
        GPT2PreTrainedModel.__init__(self, config)

        self.transformer = GPT2FlexModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.model_parallel = False
        self.device_map = None

        self.post_init()
