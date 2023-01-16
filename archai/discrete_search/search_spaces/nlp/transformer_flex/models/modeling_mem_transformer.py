# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) 2018, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.file_utils import ModelOutput
from transformers.models.transfo_xl.modeling_transfo_xl import (
    TransfoXLModel,
    TransfoXLPreTrainedModel,
)

from archai.discrete_search.search_spaces.nlp.transformer_flex.models.configuration_mem_transformer import (
    MemTransformerConfig,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.models.mem_transformer_utils.adaptive_embedding import (
    AdaptiveEmbedding,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.models.mem_transformer_utils.positional_embedding import (
    PositionalEmbedding,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.models.mem_transformer_utils.projected_adaptive_log_softmax import (
    ProjectedAdaptiveLogSoftmax,
)
from archai.discrete_search.search_spaces.nlp.transformer_flex.models.mem_transformer_utils.rel_partial_learnable_decoder import (
    RelPartialLearnableDecoderLayer,
)


class MemTransformerBaseOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MemTransformerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_scores: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    mems: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    @property
    def logits(self) -> torch.FloatTensor:
        return self.prediction_scores


class MemTransformerModel(TransfoXLModel):
    config_class = MemTransformerConfig

    def __init__(self, config: MemTransformerConfig) -> None:
        super().__init__(config)

        self.word_emb = AdaptiveEmbedding(
            config.vocab_size,
            config.d_embed,
            config.d_model,
            config.cutoffs,
            div_val=config.div_val,
            fp16=config.fp16,
        )

        self.layers = nn.ModuleList()
        for _ in range(config.n_layer):
            layer_i = RelPartialLearnableDecoderLayer(
                config.n_head,
                config.d_model,
                config.d_head,
                config.d_inner,
                config.dropout,
                dropatt=config.dropatt,
                primer_conv=config.primer_conv,
                primer_square=config.primer_square,
                pre_lnorm=config.pre_lnorm,
                layer_norm_epsilon=config.layer_norm_epsilon,
                r_w_bias=None if config.untie_r else self.r_w_bias,
                r_r_bias=None if config.untie_r else self.r_r_bias,
            )
            self.layers.append(layer_i)

        self.pos_embeds = PositionalEmbedding(self.config.d_model)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        mems: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MemTransformerBaseOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Original Transformer-XL uses [q_length, batch_size], where
        # we prefer to use [batch_size, q_length]
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both `input_ids` and `inputs_embeds` at the same time")
        elif input_ids is not None:
            input_ids = input_ids.transpose(0, 1).contiguous()
            q_length, batch_size = input_ids.size()
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            q_length, batch_size = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either `input_ids` or `inputs_embeds`")

        if mems is None:
            mems = self.init_mems(batch_size)

        # (n_hidden_layers, q_length, k_length, batch_size, n_head)
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            # Guarantees 16-bit floating point compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.n_layer

        if inputs_embeds is not None:
            word_embeds = inputs_embeds
        else:
            word_embeds = self.word_emb(input_ids)

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.n_layer)
        else:
            past_length = past_key_values[0][0].size(0)

        mem_length = mems[0].size(0) if mems is not None else 0
        k_length = mem_length + q_length

        if self.same_length:
            all_ones = word_embeds.new_ones((q_length, k_length + past_length), dtype=torch.uint8)
            mask_length = k_length - self.mem_len

            if mask_length > 0:
                mask_shifted_length = q_length - mask_length
            else:
                mask_shifted_length = q_length

            dec_attn_mask = (
                torch.triu(all_ones, 1 + mem_length + past_length) + torch.tril(all_ones, -mask_shifted_length)
            )[:, :, None]
        else:
            dec_attn_mask = torch.triu(
                word_embeds.new_ones((q_length, k_length + past_length), dtype=torch.uint8),
                diagonal=1 + mem_length + past_length,
            )[:, :, None]

        hidden_states = []
        attentions = [] if output_attentions else None
        presents = () if use_cache else None

        pos_sequence = torch.arange(
            k_length + past_length - 1,
            past_length - 1,
            -1.0,
            device=word_embeds.device,
            dtype=word_embeds.dtype,
        )

        if self.clamp_len > 0:
            pos_sequence.clamp_(max=self.clamp_len)

        pos_embeds = self.pos_emb(pos_sequence)
        pos_embeds = self.drop(pos_embeds)

        output = self.drop(word_embeds)

        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            hidden_states.append(output)

            mems_i = None if mems is None else mems[i]

            layer_output = layer(
                output,
                pos_embeds,
                layer_past=layer_past,
                dec_attn_mask=dec_attn_mask,
                mems=mems_i,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            output = layer_output[0]

            if use_cache is True:
                presents = presents + (layer_output[1],)

            if output_attentions:
                attentions.append(layer_output[2])

        output = self.drop(output)
        new_mems = self._update_mems(hidden_states, mems, mem_length, q_length)

        if output_hidden_states:
            # (batch_size, length, d_model)
            hidden_states.append(output)
            hidden_states = tuple(t.transpose(0, 1).contiguous() for t in hidden_states)
        else:
            hidden_states = None

        if output_attentions:
            # (batch_size, n_heads, q_length, k_length)
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)

        # (batch_size, length, d_model)
        output = output.transpose(0, 1).contiguous()

        if not return_dict:
            return tuple(v for v in [output, presents, new_mems, hidden_states, attentions] if v is not None)

        return MemTransformerBaseOutput(
            last_hidden_state=output,
            past_key_values=presents,
            mems=new_mems,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class MemTransformerLMHeadModel(TransfoXLPreTrainedModel):
    config_class = MemTransformerConfig

    def __init__(self, config: MemTransformerConfig) -> None:
        super().__init__(config)

        self.transformer = MemTransformerModel(config)

        if self.config.tie_word_embeddings:
            emb_weights = [emb_layer.weight for emb_layer in self.transformer.word_emb.emb_layers]
        else:
            emb_weights = None
        emb_projs = self.transformer.word_emb.emb_projs

        self.crit = ProjectedAdaptiveLogSoftmax(
            config.vocab_size,
            config.d_embed,
            config.d_model,
            config.cutoffs,
            config.tie_projs,
            emb_projs=emb_projs,
            emb_weights=emb_weights,
            div_val=config.div_val,
        )

        self.init_weights()

    def tie_weights(self) -> None:
        # Mockup to disable weight tieing as it is already being done
        pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        mems: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MemTransformerOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size, target_length = input_ids.size(0), input_ids.size(1)
        elif inputs_embeds is not None:
            batch_size, target_length = inputs_embeds.size(0), inputs_embeds.size(1)
        else:
            raise ValueError("You have to specify either `input_ids` or `inputs_embeds`")

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = transformer_outputs[0]
        pred_hidden_state = last_hidden_state[:, -target_length:]

        if labels is not None:
            # Prevents all labels being -100 and throwing an error
            # when backwarding the loss
            miss_valid_label = labels[0, 1:].sum() == (labels.size(1) - 1) * -100
            if miss_valid_label:
                # Sets an <EOS> token, just to prevent loss from being NaN
                labels[0, 1] = self.config.eos_token_id

        softmax_output = self.crit(pred_hidden_state, labels)

        if labels is not None:
            prediction_scores = self.crit(pred_hidden_state, None).detach()
            prediction_scores = prediction_scores.view(batch_size, target_length, -1)

            loss = softmax_output.view(batch_size, target_length - 1)
            loss = loss[loss != 0].mean()
        else:
            prediction_scores = softmax_output.view(batch_size, target_length, -1)
            loss = None

        if not return_dict:
            output = (prediction_scores,) + transformer_outputs[1:]

            if loss is not None:
                return (loss,) + output

            return output

        return MemTransformerOutput(
            loss=loss,
            prediction_scores=prediction_scores,
            past_key_values=transformer_outputs.past_key_values,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
