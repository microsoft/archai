# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.models.codegen.configuration_codegen import CodeGenConfig
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention,
    CodeGenBlock,
    CodeGenForCausalLM,
    CodeGenMLP,
    CodeGenModel,
    CodeGenPreTrainedModel,
    apply_rotary_pos_emb,
    fixed_pos_embedding,
)

from archai.nlp import logging_utils

logger = logging_utils.get_logger(__name__)


class CodeGenConvAttConfig(CodeGenConfig):
    model_type = "codegen_conv_att"

    def __init__(
        self,
        *args,
        kernel_size: Optional[int] = 64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.kernel_size = kernel_size


class CodeGenConvAttAttention(CodeGenAttention):
    def __init__(self, config: CodeGenConvAttConfig) -> None:
        nn.Module.__init__(self)

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size

        # treat the config.num_attention_heads as total
        # number of desired attention heads
        self.num_total_attention_heads = config.num_attention_heads

        self.head_dim = self.embed_dim // self.num_total_attention_heads
        if self.head_dim * self.num_total_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )

        # half of the desired attention heads will be convolution
        self.conv_kernel_size = config.kernel_size
        self.num_conv_heads = self.num_total_attention_heads // 2
        self.conv_head_dim = self.head_dim * self.num_conv_heads
        self.conv_map_in = nn.Linear(self.embed_dim, self.conv_head_dim)
        # NOTE: critical that this has no padding aka 'valid'
        # we will be left padding the input during forward
        self.conv = nn.Conv1d(
            self.conv_head_dim,
            self.conv_head_dim,
            self.conv_kernel_size,
            padding="valid",
            groups=self.conv_head_dim,
        )
        self.conv_map_out = nn.Linear(self.conv_head_dim, self.conv_head_dim)

        # and rest (half of the head budget) will be regular self-attention
        self.num_attention_heads = self.num_total_attention_heads - self.num_conv_heads
        self.embed_dim_for_self_att = self.head_dim * self.num_attention_heads

        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim_for_self_att * 3, bias=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        # convolution pathway
        # ---------------------
        # NOTE: convolution pathway does not need
        if layer_past is not None:
            # get the past few hidden states
            # (up to kernel length since we
            # don't more than that for conv1d)
            padding = (0, 0, self.conv_kernel_size - 1, 0)
            hidden_states_for_conv = nn.functional.pad(layer_past[2], padding, mode="constant", value=0)
            left_start = hidden_states_for_conv.shape[1] - self.conv_kernel_size
            hidden_states_for_conv = hidden_states_for_conv[:, left_start:, :]
        else:
            # first pad from the left to ensure causal mask during training
            # refer to https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            padding = (0, 0, self.conv_kernel_size - 1, 0)
            hidden_states_for_conv = nn.functional.pad(hidden_states, padding, mode="constant", value=0)

        conv_output = self.conv(F.relu(self.conv_map_in(hidden_states_for_conv)).transpose(-1, -2)).transpose(-1, -2)
        conv_output = self.conv_map_out(conv_output)

        if layer_past is not None:
            # if everything was done correctly during generation
            # only one convolution should have happened
            assert conv_output.shape[1] == 1

        # attention
        qkv = self.qkv_proj(hidden_states)

        # NOTE: mp_num = 4 was originally to take advantage of TPU-v4
        # set map_num = 1 since we are using GPUs
        mp_num = 1
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        all_hidden_states = hidden_states

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            past_hidden_states = layer_past[2]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
            all_hidden_states = torch.cat((past_hidden_states, hidden_states), dim=-2)

        if use_cache is True:
            present = (key, value, all_hidden_states)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        # NOTE: the association and broadcast maps are just
        # passed in as they don't have to be computed.
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)

        # add the conv pathway
        attn_output = torch.cat([attn_output, conv_output], dim=-1)

        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class CodeGenConvAttBlock(CodeGenBlock):
    def __init__(self, config: CodeGenConvAttConfig) -> None:
        nn.Module.__init__(self)

        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CodeGenConvAttAttention(config)
        self.mlp = CodeGenMLP(inner_dim, config)


class CodeGenConvAttModel(CodeGenModel):
    config_class = CodeGenConvAttConfig

    def __init__(self, config: CodeGenConvAttConfig) -> None:
        CodeGenPreTrainedModel.__init__(self, config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.eos_id = config.eos_token_id
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CodeGenConvAttBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)

        self.gradient_checkpointing = False

        self.post_init()


class CodeGenConvAttForCausalLM(CodeGenForCausalLM):
    config_class = CodeGenConvAttConfig

    def __init__(self, config: CodeGenConvAttConfig) -> None:
        CodeGenPreTrainedModel.__init__(self, config)

        self.transformer = CodeGenConvAttModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.post_init()
