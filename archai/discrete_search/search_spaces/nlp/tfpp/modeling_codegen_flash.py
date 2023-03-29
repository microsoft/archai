# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.codegen.configuration_codegen import CodeGenConfig
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention,
    CodeGenMLP,
    CodeGenPreTrainedModel,
)


class CodeGenFlashConfig(CodeGenConfig):
    model_type = "codegen-flash"

    def __init__(
        self,
        *args,
        pad_vocab_size_multiple: Optional[int] = 1,
        use_flash_attn: Optional[bool] = False,
        use_flash_fused_mlp: Optional[bool] = False,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.vocab_size = int(math.ceil(self.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
        self.use_flash_attn = use_flash_attn
        self.use_flash_fused_mlp = use_flash_fused_mlp


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


class CodeGenFlashBlock(nn.Module):
    def __init__(self, config: CodeGenFlashConfig) -> None:
        super().__init__()

        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)

        self.use_flash_attn = config.use_flash_attn
        self.resid_pdrop = config.resid_pdrop
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        if not self.use_flash_attn:
            self.attn = CodeGenAttention(config)
            self.mlp = CodeGenMLP(inner_dim, config)
        else:
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

            if not config.use_flash_fused_mlp:
                self.mlp = Mlp(
                    in_features=config.n_embd, hidden_features=inner_dim, activation=ACT2FN[config.activation_function]
                )
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

        if self.use_flash_attn:
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
