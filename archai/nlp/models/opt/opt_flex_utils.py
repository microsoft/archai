# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""OPT flexible-related classes and methods.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoder,
    OPTDecoderLayer,
    OPTForCausalLM,
    OPTLearnedPositionalEmbedding,
    OPTModel,
    OPTPreTrainedModel,
)


class OPTDecoderLayerFlex(nn.Module):
    """Implements an OPT Decoder flexible layer."""

    def __init__(self, config: Dict[str, Any], layer_idx: Optional[int] = None) -> None:
        """Overrides with custom initialization.

        Args:
            config: Dictionary holding model's configuration.
            layer_idx: Number of current layer (identifier).

        """

        super().__init__()

        self.embed_dim = config.hidden_size

        ffn_dim = config.ffn_dim[layer_idx]
        num_heads = config.num_attention_heads[layer_idx]

        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout

        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        layer_head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Overrides forward method.

        Args:
            hidden_states: Input hidden states.
            attention_mask: Attention mask.
            layer_head_mask: Head mask.
            output_attentions: Whether to return attention tensors.
            use_cache: Whether to use and save past key/values states.
            past_key_value: Input past key/values.

        Returns:
            (Tuple[Union[torch.FloatTensor, Tuple[torch.FloatTensor]], ...]): Output, attention weights and present states.

        """

        return OPTDecoderLayer.forward(
            self,
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )


class OPTDecoderFlex(OPTPreTrainedModel):
    """Implements an OPT Decoder flexible."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Overrides with custom initialization.

        Args:
            config: Dictionary holding model's configuration.

        """

        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size, config.word_embed_proj_dim, bias=False
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim, config.hidden_size, bias=False
            )
        else:
            self.project_in = None

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [OPTDecoderLayerFlex(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self) -> torch.FloatTensor:
        """Gets the input embeddings.

        Returns:
            (torch.FloatTensor): Input embeddings.

        """

        return OPTDecoder.get_input_embeddings(self)

    def set_input_embeddings(self, value: torch.FloatTensor) -> None:
        """Sets new input embeddings.

        Args:
            value: New embeddings to be set.

        """

        return OPTDecoder.set_input_embeddings(self, value)

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: torch.LongTensor,
        input_shape: Tuple[int, ...],
        inputs_embeds: torch.FloatTensor,
        past_key_values_length: int,
    ) -> torch.LongTensor:
        """Prepares the decoder attention mask.

        Args:
            attention_mask: Attention mask.
            input_shape: Input shape.
            input_embeds: Input embeddings.
            past_key_values_length: Length of past key/values.

        Returns:
            (torch.LongTensor): Decoder attention mask.

        """

        return OPTDecoder._prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Overrides forward method.

        Args:
            input_ids: Input tokens identifiers.
            attention_mask: Attention mask.
            head_mask: Head mask.
            past_key_values: Past key/values states.
            input_embeds: Input embeddings.
            use_cache: Whether to use and save past key/values states.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return ModelOutput instead of tuple.

        Returnd:
            (Union[Tuple, BaseModelOutputWithPast]): Model's outputs.

        """

        return OPTDecoder.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class OPTModelFlex(OPTPreTrainedModel):
    """Implements an OPT flexible model."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Overrides with custom initialization.

        Args:
            config: Dictionary holding model's configuration.

        """

        super().__init__(config)

        self.decoder = OPTDecoderFlex(config)

        self.post_init()

    def get_input_embeddings(self) -> torch.FloatTensor:
        """Gets the input embeddings.

        Returns:
            (torch.FloatTensor): Input embeddings.

        """

        return OPTModel.get_input_embeddings(self)

    def set_input_embeddings(self, value: torch.FloatTensor) -> None:
        """Sets new input embeddings.

        Args:
            value: New embeddings to be set.

        """

        return OPTModel.set_input_embeddings(self, value)

    def get_decoder(self) -> OPTDecoder:
        """Gets the decoder.

        Returns:
            (OPTDecoder): Decoder.

        """

        return OPTModel.get_decoder(self)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Overrides forward method.

        Args:
            input_ids: Input tokens identifiers.
            attention_mask: Attention mask.
            head_mask: Head mask.
            past_key_values: Past key/values states.
            input_embeds: Input embeddings.
            use_cache: Whether to use and save past key/values states.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return ModelOutput instead of tuple.

        Returnd:
            (Union[Tuple, BaseModelOutputWithPast]): Model's outputs.

        """

        return OPTModel.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class OPTForCausalLMFlex(OPTPreTrainedModel):
    """Implements an OPT flexible model for causal language modeling."""

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Overrides with custom initialization.

        Args:
            config: Dictionary holding model's configuration.

        """

        super().__init__(config)

        self.model = OPTModelFlex(config)
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self) -> torch.FloatTensor:
        """Gets the input embeddings.

        Returns:
            (torch.FloatTensor): Input embeddings.

        """

        return OPTForCausalLM.get_input_embeddings(self)

    def set_input_embeddings(self, value: torch.FloatTensor) -> None:
        """Sets new input embeddings.

        Args:
            value: New embeddings to be set.

        """

        return OPTForCausalLM.set_input_embeddings(self, value)

    def get_output_embeddings(self) -> torch.FloatTensor:
        """Gets the output embeddings.

        Returns:
            (torch.FloatTensor): Output embeddings.

        """

        return OPTForCausalLM.get_output_embeddings(self)

    def set_output_embeddings(self, new_embeddings: torch.FloatTensor) -> None:
        """Sets new output embeddings.

        Args:
            new_embeddings: New embeddings to be set.

        """

        return OPTForCausalLM.set_output_embeddings(self, new_embeddings)

    def set_decoder(self, decoder: OPTDecoder) -> None:
        """Sets new decoder.

        Args:
            decoder: Decoder to be set.

        """

        return OPTForCausalLM.set_decoder(self, decoder)

    def get_decoder(self) -> OPTDecoder:
        """Gets the decoder.

        Returns:
            (OPTDecoder): Decoder.

        """

        return OPTForCausalLM.get_decoder(self)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Overrides forward method.

        Args:
            input_ids: Input tokens identifiers.
            attention_mask: Attention mask.
            head_mask: Head mask.
            past_key_values: Past key/values states.
            input_embeds: Input embeddings.
            labels: Labels.
            use_cache: Whether to use and save past key/values states.
            token_type_ids: Token type identifers.
            output_attentions: Whether to return attention tensors.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return ModelOutput instead of tuple.

        Returnd:
            (Union[Tuple, CausalLMOutputWithPast]): Model's outputs.

        """

        return OPTForCausalLM.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepares inputs for text generation.

        Args:
            input_ids: Inputs identifiers.
            past: Past key/values.
            attention_mask: Attention mask.
            use_cache: Whether to use and save past key/values states.

        Returns:
            (Dict[str, Any]): Inputs prepared for generation.

        """

        return OPTForCausalLM.prepare_inputs_for_generation(
            self,
            input_ids,
            past=past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs
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

        return OPTForCausalLM._reorder_cache(self, past, beam_idx)
