# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
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


def make_asso_map(input_ids: torch.LongTensor, mask: torch.FloatTensor) -> torch.FloatTensor:
    T = input_ids.shape[1]
    tokens = input_ids.unsqueeze(1).float()
    v = torch.ones(1, 1, T, device=input_ids.device).float()
    asso_map = (
        v.transpose(-1, -2) @ tokens**2 + (tokens**2).transpose(-1, -2) @ v - 2 * tokens.transpose(-1, -2) @ tokens
    )
    asso_map = (asso_map.long() == 0).float()
    idx = torch.arange(T, device=input_ids.device)
    asso_map[:, idx, idx] = 0
    asso_map *= mask.unsqueeze(-1) * mask.unsqueeze(1)
    asso_map /= asso_map.sum(-1, keepdim=True) + 1e-6

    return asso_map.unsqueeze(1)


def make_broadcast_map(input_ids: torch.LongTensor, mask: torch.FloatTensor, eos_id: int) -> torch.FloatTensor:
    T = input_ids.shape[1]
    eos_map = (input_ids == eos_id).float()
    eos_map = eos_map.unsqueeze(1).expand(-1, T, -1)
    eos_mapp = eos_map * (mask.unsqueeze(-1) * mask.unsqueeze(1))
    eos_map = eos_mapp / (eos_map.sum(dim=-1, keepdim=True) + 1e-6)

    return eos_map.unsqueeze(1)


class CodeGenHardCodedConfig(CodeGenConfig):
    model_type = "codegen_hard_coded"

    def __init__(
        self,
        *args,
        kernel_size: Optional[int] = 64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.kernel_size = kernel_size


class CodeGenHardCodedAttention(CodeGenAttention):
    def __init__(self, config: CodeGenHardCodedConfig) -> None:
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

        # half - 2 of the desired attention heads will be convolution
        self.conv_kernel_size = config.kernel_size
        self.num_conv_heads = self.num_total_attention_heads // 2 - 2
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

        # of the other rest 2 will be hardcoded attention (association and broadcast attention)
        # and rest (half of the head budget) will be regular self-attention
        self.num_attention_heads = self.num_total_attention_heads - self.num_conv_heads - 2
        self.embed_dim_for_self_att = self.head_dim * self.num_attention_heads
        self.embed_dim_for_hardcode = self.head_dim * 2

        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim_for_self_att * 3, bias=False)
        self.vproj_hardcode = nn.Linear(self.embed_dim, self.embed_dim_for_hardcode, bias=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim

    def _attn(
        self,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        value_hardcode: torch.FloatTensor,
        asso_map: torch.FloatTensor,
        bro_map: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, ...]:
        # Compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # Get the relevant part of association and broadcast map
        asso_map = asso_map[:, :, key_length - query_length : key_length, :key_length]
        bro_map = bro_map[:, :, key_length - query_length : key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / self.scale_attn
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        # Causal mask the association and broadcast maps
        # with 0 since they are of 0-1 masks
        # NOTE: filling in with 0 since this does not go
        # through softmax
        # TODO: potentially re-normalize here
        asso_map_hardcode = torch.where(causal_mask, asso_map, 0)
        bro_map_hardcode = torch.where(causal_mask, bro_map, 0)

        if attention_mask is not None:
            # Apply the attention mask
            # NOTE: asso_map and bro_map don't need this
            # as they already take the attention_mask into account
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Concatenate the association map and broadcast map
        attn_weights = torch.cat([attn_weights, asso_map_hardcode, bro_map_hardcode], dim=1)
        value = torch.cat([value, value_hardcode], dim=1)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        asso_map: torch.FloatTensor,
        bro_map: torch.FloatTensor,
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
        if layer_past is not None:
            # get the past few hidden states
            # (up to kernel length since we
            # don't more than that for conv1d)
            padding = (0, 0, self.conv_kernel_size - 1, 0)
            hidden_states_for_conv = nn.functional.pad(layer_past[3], padding, mode="constant", value=0)
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

        # attention (self-attention and hardcoded-attention pathway)
        qkv = self.qkv_proj(hidden_states)

        # separate pathway for values corresponding to the two hardcoded attention pathways
        value_hardcode = self.vproj_hardcode(hidden_states).reshape(-1, hidden_states.shape[1], 2, self.head_dim)
        value_hardcode = value_hardcode.permute(0, 2, 1, 3)

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
            past_value_hardcode = layer_past[2]
            past_hidden_states = layer_past[3]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
            value_hardcode = torch.cat((past_value_hardcode, value_hardcode), dim=-2)
            all_hidden_states = torch.cat((past_hidden_states, hidden_states), dim=-2)

        if use_cache is True:
            present = (key, value, value_hardcode, all_hidden_states)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        # NOTE: the association and broadcast maps are just
        # passed in as they don't have to be computed.
        attn_output, attn_weights = self._attn(
            query, key, value, value_hardcode, asso_map, bro_map, attention_mask, head_mask
        )

        attn_output = self._merge_heads(attn_output, self.num_attention_heads + 2, self.head_dim)

        # add the conv pathway
        attn_output = torch.cat([attn_output, conv_output], dim=-1)

        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class CodeGenHardCodedBlock(CodeGenBlock):
    def __init__(self, config: CodeGenHardCodedConfig) -> None:
        nn.Module.__init__(self)

        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CodeGenHardCodedAttention(config)
        self.mlp = CodeGenMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        asso_map: Optional[torch.FloatTensor],
        bro_map: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            asso_map,
            bro_map,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class CodeGenHardCodedModel(CodeGenModel):
    config_class = CodeGenHardCodedConfig

    def __init__(self, config: CodeGenHardCodedConfig) -> None:
        CodeGenPreTrainedModel.__init__(self, config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.eos_id = config.eos_token_id
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CodeGenHardCodedBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)

        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        all_input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        device = input_ids.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Prepare association attention map
        # NOTE: important to call make_asso_map and make_broadcast_map before attention_mask
        # is transformed to [-negative_large_value, 0]
        # NOTE: during generation only the last generated input id is passed in
        # to leverage past state caching. But association and broadcast maps
        # need to see the entire past ids.
        input_ids_for_hardcoded = all_input_ids if all_input_ids is not None else input_ids
        asso_map = make_asso_map(input_ids_for_hardcoded, attention_mask)
        bro_map = make_broadcast_map(input_ids_for_hardcoded, attention_mask, eos_id=self.eos_id)

        # TODO: Causal mask asso and bro maps once here
        # itself since they don't change

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    asso_map,
                    bro_map,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CodeGenHardCodedForCausalLM(CodeGenForCausalLM):
    config_class = CodeGenHardCodedConfig

    def __init__(self, config: CodeGenHardCodedConfig) -> None:
        CodeGenPreTrainedModel.__init__(self, config)

        self.transformer = CodeGenHardCodedModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past: Optional[torch.FloatTensor] = None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        # association map and broadcast map need all ids
        # so making a copy of all input ids generated/input
        all_input_ids = input_ids
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            # only last token
            input_ids = input_ids[:, -1].unsqueeze(-1)

            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "all_input_ids": all_input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        all_input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            all_input_ids=all_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
