# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2PreTrainedModel,
)
from transformers.pytorch_utils import Conv1D


class GPT2AttentionFlex(nn.Module):
    """Implements a GPT-2 Attention flexible layer."""

    def __init__(
        self,
        config: Dict[str, Any],
        is_cross_attention: Optional[bool] = False,
        layer_idx: Optional[int] = None,
    ) -> None:
        """Overrides with custom initialization.

        Args:
            config: Dictionary holding model's configuration.
            is_cross_attention: Whether attention is standard or cross-attention.
            layer_idx: Number of current layer (identifier).

        """

        super().__init__()

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

    def prune_heads(self, heads: List[int]) -> None:
        """Prunes a set of attention heads.

        Args:
            heads: Heads to be prunned.

        """

        return GPT2Attention.prune_heads(self, heads)

    def _attn(
        self,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Calculates the attention mechanism.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attention_mask: Attention mask.
            head_mask: Head mask.

        Returns:
            (Tuple[torch.FloatTensor, torch.FloatTensor]): Attention outputs and weights.

        """

        return GPT2Attention._attn(
            self, query, key, value, attention_mask=attention_mask, head_mask=head_mask
        )

    def _upcast_and_reordered_attn(
        self,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Applies auto-cast and re-orders attentions to correct shape.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attention_mask: Attention mask.
            head_mask: Head mask.

        Returns:
            (Tuple[torch.FloatTensor, torch.FloatTensor]): Attention outputs and weights.

        """

        return GPT2Attention._upcast_and_reordered_attn(
            self, query, key, value, attention_mask=attention_mask, head_mask=head_mask
        )

    def _split_heads(
        self, tensor: torch.FloatTensor, num_heads: int, attn_head_size: int
    ) -> torch.FloatTensor:
        """Splits a tensor into num_heads x attn_head_size.

        Args:
            tensor: Tensor to be split.
            num_heads: Number of attention heads.
            attn_head_size: Size of single attention head.

        Returns:
            (torch.FloatTensor): Split attention heads.

        """

        return GPT2Attention._split_heads(self, tensor, num_heads, attn_head_size)

    def _merge_heads(
        self, tensor: torch.FloatTensor, num_heads: int, attn_head_size: int
    ) -> torch.FloatTensor:
        """Merges a tensor into num_heads x attn_head_size.

        Args:
            tensor: Tensor to be merged.
            num_heads: Number of attention heads.
            attn_head_size: Size of single attention head.

        Returns:
            (torch.FloatTensor): Merged attention heads.

        """

        return GPT2Attention._merge_heads(self, tensor, num_heads, attn_head_size)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.FloatTensor, Tuple[torch.FloatTensor]], ...]:
        """Overrides forward method.

        Args:
            hidden_states: Input hidden states.
            layer_past: Input past key/values.
            attention_mask: Attention mask.
            head_mask: Head mask.
            encoder_hidden_states: Encoder hidden states.
            encoder_attention_mask: Encoder attention mask.
            use_cache: Whether to use and save past key/values states.
            output_attentions: Whether to return attention tensors.

        Returns:
            (Tuple[Union[torch.FloatTensor, Tuple[torch.FloatTensor]], ...]): Output, present states and attention weights.

        """

        return GPT2Attention.forward(
            self,
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


class GPT2MLPFlex(nn.Module):
    """Implements a GPT-2 Multi-Layer Perceptron flexible layer."""

    def __init__(self, intermediate_size: int, config: Dict[str, Any]) -> None:
        """Overrides with custom initialization.

        Args:
            intermediate_size: Size of inner dimension.
            config: Dictionary holding model's configuration.

        """

        super().__init__()

        embed_dim = config.hidden_size

        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

        self.primer_square = config.primer_square

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Overrides forward method to add PrimerEZ-square primitive.

        Args:
            hidden_states: Input hidden states.

        Returns:
            (torch.FloatTensor): Output hidden states.

        """

        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)

        if self.primer_square:
            hidden_states = hidden_states**2

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class GPT2BlockFlex(nn.Module):
    """Implements a GPT-2 flexible block."""

    def __init__(self, config: Dict[str, Any], layer_idx: Optional[int] = None) -> None:
        """Overrides with custom initialization.

        Args:
            config: Dictionary holding model's configuration.
            layer_idx: Number of current layer (identifier).

        """

        super().__init__()

        hidden_size = config.hidden_size
        inner_dim = (
            config.n_inner[layer_idx] if config.n_inner is not None else 4 * config.hidden_size
        )

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2AttentionFlex(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2AttentionFlex(
                config, is_cross_attention=True, layer_idx=layer_idx
            )
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLPFlex(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.FloatTensor],
        Optional[Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        """Overrides forward method.

        Args:
            hidden_states: Input hidden states.
            layer_past: Input past key/values.
            attention_mask: Attention mask.
            head_mask: Head mask.
            encoder_hidden_states: Encoder hidden states.
            encoder_attention_mask: Encoder attention mask.
            use_cache: Whether to use and save past key/values states.
            output_attentions: Whether to return attention tensors.

        Returns:
            Union[Tuple[torch.FloatTensor], Optional[Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]]]: Hidden states, present states and attention weights.

        """

        return GPT2Block.forward(
            self,
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


class GPT2ModelFlex(GPT2PreTrainedModel):
    """Implements a GPT-2 flexible model."""

    _keys_to_ignore_on_load_missing = GPT2Model._keys_to_ignore_on_load_missing

    def __init__(self, config: Dict[str, Any]) -> None:
        """Overrides with custom initialization.

        Args:
            config: Dictionary holding model's configuration.

        """

        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPT2BlockFlex(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        self.post_init()

    def parallelize(self, device_map: Dict[Any, List[int]]) -> None:
        """Parallelizes the module.

        Args:
            device_map: Map of devices.

        """

        return GPT2Model.parallelize(self, device_map=device_map)

    def deparallelize(self) -> None:
        """De-parallelizes the module."""

        return GPT2Model.deparallelize(self)

    def get_input_embeddings(self) -> torch.FloatTensor:
        """Gets the input embeddings.

        Returns:
            (torch.FloatTensor): Input embeddings.

        """

        return GPT2Model.get_input_embeddings(self)

    def set_input_embeddings(self, new_embeddings: torch.FloatTensor) -> None:
        """Sets new input embeddings.

        Args:
            new_embeddings: New embeddings to be set.

        """

        return GPT2Model.set_input_embeddings(self, new_embeddings)

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """Prunes a set of heads.

        Args:
            heads_to_prune: Heads to be prunned.

        """

        return GPT2Model._prune_heads(self, heads_to_prune)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """Overrides forward method.

        Args:
            input_ids: Input tokens identifiers.
            past_key_values: Past key/values states.
            attention_mask: Attention mask.
            token_type_ids: Token type identifers.
            position_ids: Position identifiers.
            head_mask: Head mask.
            input_embeds: Input embeddings.
            encoder_hidden_states: Encoder hidden states.
            encoder_attention_mask: Encoder attention mask.
            use_cache: Whether to use and save past key/values states.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return ModelOutput instead of tuple.

        Returnd:
            (Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]): Model's outputs.

        """

        return GPT2Model.forward(
            self,
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class GPT2LMHeadModelFlex(GPT2PreTrainedModel):
    """Implements a GPT-2 language modeling flexible head."""

    _keys_to_ignore_on_load_missing = GPT2LMHeadModel._keys_to_ignore_on_load_missing

    def __init__(self, config: Dict[str, Any]) -> None:
        """Overrides with custom initialization.

        Args:
            config: Dictionary holding model's configuration.

        """

        super().__init__(config)

        self.transformer = GPT2ModelFlex(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.model_parallel = False
        self.device_map = None

        self.post_init()

    def parallelize(self, device_map: Dict[Any, List[int]]) -> None:
        """Parallelizes the module.

        Args:
            device_map: Map of devices.

        """

        return GPT2LMHeadModel.parallelize(self, device_map=device_map)

    def deparallelize(self) -> None:
        """De-parallelizes the module."""

        return GPT2LMHeadModel.deparallelize(self)

    def get_output_embeddings(self) -> torch.FloatTensor:
        """Gets the output embeddings.

        Returns:
            (torch.FloatTensor): Output embeddings.

        """

        return GPT2LMHeadModel.get_output_embeddings(self)

    def set_output_embeddings(self, new_embeddings: torch.FloatTensor) -> None:
        """Sets new output embeddings.

        Args:
            new_embeddings: New embeddings to be set.

        """

        return GPT2LMHeadModel.set_output_embeddings(self, new_embeddings)

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past: Optional[torch.FloatTensor] = None, **kwargs
    ) -> Dict[str, Any]:
        """Prepares inputs for text generation.

        Args:
            input_ids: Inputs identifiers.
            past: Past key/values.

        Returns:
            (Dict[str, Any]): Inputs prepared for generation.

        """

        return GPT2LMHeadModel.prepare_inputs_for_generation(
            self, input_ids, past=past, **kwargs
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """Overrides forward method.

        Args:
            input_ids: Input tokens identifiers.
            past_key_values: Past key/values states.
            attention_mask: Attention mask.
            token_type_ids: Token type identifers.
            position_ids: Position identifiers.
            head_mask: Head mask.
            input_embeds: Input embeddings.
            encoder_hidden_states: Encoder hidden states.
            encoder_attention_mask: Encoder attention mask.
            labels: Labels.
            use_cache: Whether to use and save past key/values states.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return ModelOutput instead of tuple.

        Returnd:
            (Union[Tuple, CausalLMOutputWithCrossAttentions]): Model's outputs.

        """

        return GPT2LMHeadModel.forward(
            self,
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @staticmethod
    def _reorder_cache(
        self, past: Tuple[Tuple[torch.FloatTensor]], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.FloatTensor]]:
        """Re-orders the `past_key_values` cache.

        Args:
            past: Past key/values.
            beam_idx: Identifiers of beam sampling.

        Returns:
            (Tuple[Tuple[torch.FloatTensor]]): Re-ordered past key/values.

        """

        return GPT2LMHeadModel._reorder_cache(self, past, beam_idx)
