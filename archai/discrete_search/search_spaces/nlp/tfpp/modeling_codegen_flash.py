# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.codegen.configuration_codegen import CodeGenConfig
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention,
    CodeGenMLP,
    CodeGenPreTrainedModel,
    apply_rotary_pos_emb,
    fixed_pos_embedding,
)
from xformers.ops import LowerTriangularMask, memory_efficient_attention


class CodeGenFlashConfig(CodeGenConfig):
    model_type = "codegen-flash"

    def __init__(
        self,
        *args,
        pad_vocab_size_multiple: Optional[int] = 1,
        attn_type: Optional[str] = "default",
        use_fused_mlp: Optional[bool] = False,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.vocab_size = int(math.ceil(self.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
        assert attn_type in [
            "default",
            "flash",
            "xformer",
        ], "`attn_type` should be one of: `default`, `flash` or `xformer`."
        self.attn_type = attn_type
        self.use_fused_mlp = use_fused_mlp


class CodeGenFlashEmbedding(nn.Module):
    def __init__(self, config: CodeGenFlashConfig) -> None:
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)

        return hidden_states


class CodeGenXAttention(CodeGenAttention):
    def __init__(self, config):
        super().__init__(config)

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        # TODO(enijkamp): factor out number of logical TPU-v4 cores or make forward pass agnostic
        mp_num = 4
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        seq_len = key.shape[1]
        offset = 0

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

        # compute self-attention: V x Softmax(QK^T)
        attn_output = memory_efficient_attention(
            query.to(torch.float16),
            key.to(torch.float16),
            value.to(torch.float16),
            attn_bias=LowerTriangularMask(),
        )

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class CodeGenFlashBlock(nn.Module):
    def __init__(self, config: CodeGenFlashConfig) -> None:
        super().__init__()

        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)

        self.attn_type = config.attn_type
        self.use_fused_mlp = config.use_fused_mlp
        self.resid_pdrop = config.resid_pdrop
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        if self.attn_type == "default":
            self.attn = CodeGenAttention(config)
        elif self.attn_type == "flash":
            head_dim = config.n_embd // config.n_head
            self.attn = MHA(
                embed_dim=config.n_embd,
                num_heads=config.n_head,
                cross_attn=False,
                bias=True,
                dropout=config.attn_pdrop,
                softmax_scale=head_dim ** (-0.5),
                causal=True,
                rotary_emb_dim=rotary_dim,
                fused_bias_fc=True,
                use_flash_attn=True,
                return_residual=False,
            )
        elif self.attn_type == "xformer":
            self.attn = CodeGenXAttention(config)

        if not self.use_fused_mlp:
            self.mlp = CodeGenMLP(inner_dim, config)
        else:
            activation = (
                "gelu_approx" if config.activation_function in ["gelu_new", "gelu_fast", "gelu_approx"] else "relu"
            )
            self.mlp = FusedMLP(in_features=config.n_embd, hidden_features=inner_dim, activation=activation)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(hidden_states)
        if isinstance(attn_outputs, tuple):
            attn_outputs = attn_outputs[0]

        feed_forward_hidden_states = self.mlp(hidden_states)

        if self.attn_type == "flash":
            attn_outputs = nn.Dropout(self.resid_pdrop)(attn_outputs)
            feed_forward_hidden_states = nn.Dropout(self.resid_pdrop)(feed_forward_hidden_states)

        hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return hidden_states


class CodeGenFlashLMHead(nn.Module):
    def __init__(self, config: CodeGenFlashConfig) -> None:
        super().__init__()

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        return lm_logits


class LMHeadLoss(nn.Module):
    def __init__(self, shift_labels: Optional[bool] = False) -> None:
        super().__init__()

        self.shift_labels = shift_labels
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, lm_logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if self.shift_labels:
            lm_logits = lm_logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return loss


class CodeGenFlashSequential(CodeGenPreTrainedModel):
    def __init__(self, config: CodeGenFlashConfig) -> None:
        super().__init__(config)

        modules = [CodeGenFlashEmbedding(config)]
        for _ in range(config.n_layer):
            modules.append(CodeGenFlashBlock(config))
        modules.append(CodeGenFlashLMHead(config))

        self.layers = nn.Sequential(*modules)
        self.loss = LMHeadLoss()

        self.post_init()

    def forward(
        self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None, **kwargs
    ) -> CausalLMOutput:
        lm_logits = self.layers(input_ids)

        loss = None
        if labels is not None:
            loss = self.loss(lm_logits, labels)

        return CausalLMOutput(loss=loss, logits=lm_logits)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, torch.Tensor]:
        return {"input_ids": input_ids}

    def get_input_embeddings(self) -> torch.Tensor:
        return self.layers[0].wte

    def set_input_embeddings(self, new_embeddings: torch.Tensor) -> None:
        self.layers[0].wte = new_embeddings

    def get_output_embeddings(self) -> torch.Tensor:
        return self.layers[-1].lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor) -> None:
        self.layers[-1].lm_head = new_embeddings
