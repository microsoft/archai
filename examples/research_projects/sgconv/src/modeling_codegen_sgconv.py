# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from src.gconv import GConv
from torch import nn
from transformers.models.codegen.configuration_codegen import CodeGenConfig
from transformers.models.codegen.modeling_codegen import (
    CodeGenBlock,
    CodeGenForCausalLM,
    CodeGenMLP,
    CodeGenModel,
    CodeGenPreTrainedModel,
)

from archai.nlp import logging_utils

logger = logging_utils.get_logger(__name__)


class CodeGenSGConvConfig(CodeGenConfig):
    model_type = "codegen_sgconv"

    def __init__(
        self,
        *args,
        sgconv_d_state: Optional[int] = 64,
        sgconv_channels: Optional[int] = 1,
        sgconv_kernel_dim: Optional[int] = 128,
        sgconv_decay_min: Optional[int] = 2,
        sgconv_decay_max: Optional[int] = 2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.sgconv_d_state = sgconv_d_state
        self.sgconv_channels = sgconv_channels
        self.sgconv_kernel_dim = sgconv_kernel_dim
        self.sgconv_decay_min = sgconv_decay_min
        self.sgconv_decay_max = sgconv_decay_max

        # Cache has not been implemented yet
        self.use_cache = False


class CodeGenSGConv(nn.Module):
    def __init__(self, config: CodeGenSGConvConfig) -> None:
        super().__init__()

        self.max_positions = config.max_position_embeddings

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads

        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )

        # Bidirectional must be false in autoregressive setting
        self.gconv = GConv(
            d_model=self.embed_dim,
            d_state=config.sgconv_d_state,
            l_max=self.max_positions,
            channels=config.sgconv_channels,
            bidirectional=False,
            transposed=False,
            kernel_dim=config.sgconv_kernel_dim,
            n_scales=None,
            decay_min=config.sgconv_decay_min,
            decay_max=config.sgconv_decay_max,
        )

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:

        # Potential BUG: causal leakage? Seems no because of FFT
        y, _ = self.gconv(hidden_states, return_kernel=False)

        return y


class CodeGenSGConvBlock(CodeGenBlock):
    def __init__(self, config: CodeGenSGConvConfig) -> None:
        nn.Module.__init__(self)

        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.sgconv = CodeGenSGConv(config)
        self.mlp = CodeGenMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        sgconv_outputs = self.sgconv(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        sgconv_output = sgconv_outputs[0]  # output_sgconv: a

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = sgconv_output + feed_forward_hidden_states + residual

        outputs = (hidden_states,)

        return outputs  # hidden_states


class CodeGenSGConvModel(CodeGenModel):
    config_class = CodeGenSGConvConfig

    def __init__(self, config: CodeGenSGConvConfig) -> None:
        CodeGenPreTrainedModel.__init__(self, config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CodeGenSGConvBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        self.post_init()


class CodeGenSGConvForCausalLM(CodeGenForCausalLM):
    config_class = CodeGenSGConvConfig

    def __init__(self, config: CodeGenSGConvConfig) -> None:
        CodeGenPreTrainedModel.__init__(self, config)

        self.transformer = CodeGenSGConvModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.post_init()
