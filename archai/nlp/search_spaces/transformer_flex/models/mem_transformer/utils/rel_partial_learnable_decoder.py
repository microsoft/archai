# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) 2018, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.

"""Relational Partial-Learnable Decoder."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from archai.nlp.search_spaces.transformer_flex.models.mem_transformer.utils.depth_wise_convolution import (
    DepthWiseConvolution,
)
from archai.nlp.search_spaces.transformer_flex.models.mem_transformer.utils.position_wise_ff import (
    PositionWiseFF,
    PositionWiseFFPrimerEZ,
)


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        dropout: float,
        dropatt: Optional[float] = 0.0,
        primer_conv: Optional[bool] = False,
        pre_lnorm: Optional[bool] = False,
        r_w_bias: Optional[torch.FloatTensor] = None,
        r_r_bias: Optional[torch.FloatTensor] = None,
        layer_norm_epsilon: Optional[float] = 1e-5,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.primer_conv = primer_conv
        self.pre_lnorm = pre_lnorm
        self.scale = 1 / (d_head**0.5)

        self.qkv = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.o = nn.Linear(n_head * d_head, d_model, bias=False)
        self.r = nn.Linear(d_model, n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        if self.primer_conv:
            self.d_conv = DepthWiseConvolution(self.d_model)

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        # Bias are not shared, i.e., they are defined per layer
        if r_w_bias is None or r_r_bias is None:
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_w_bias = r_w_bias
            self.r_r_bias = r_r_bias

    def _relational_shift(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        pad_shape = (inputs.size(0), 1) + inputs.size()[2:]
        pad = torch.zeros(pad_shape, device=inputs.device, dtype=inputs.dtype)
        inputs_padded = torch.cat([pad, inputs], dim=1)

        inputs_padded_shape = (inputs.size(1) + 1, inputs.size(0)) + inputs.size()[2:]
        inputs_padded = inputs_padded.view(*inputs_padded_shape)

        output = inputs_padded[1:].view_as(inputs)

        return output

    def forward(
        self,
        w: torch.FloatTensor,
        r: torch.FloatTensor,
        layer_past: Optional[torch.FloatTensor] = None,
        attn_mask: Optional[torch.FloatTensor] = None,
        mems: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> torch.FloatTensor:
        q_length, r_length, batch_size = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            mems_w = torch.cat([mems, w], 0)

            if self.pre_lnorm:
                mems_w = self.layer_norm(mems_w)

            heads_w = self.qkv(mems_w)
            if self.primer_conv:
                heads_w = self.d_conv(heads_w)

            head_wq, head_wk, head_wv = torch.chunk(heads_w, 3, dim=-1)
            head_wq = head_wq[-q_length:]

            head_rk = self.r(r)
        else:
            if self.pre_lnorm:
                w = self.layer_norm(w)

            heads_w = self.qkv(w)
            if self.primer_conv:
                heads_w = self.d_conv(heads_w)

            head_wq, head_wk, head_wv = torch.chunk(heads_w, 3, dim=-1)

            head_rk = self.r(r)

        k_length = head_wk.size(0)

        # Changes the view according to required size
        # (head_length, batch_size, n_head, d_head)
        head_wq = head_wq.view(q_length, batch_size, self.n_head, self.d_head)
        head_wk = head_wk.view(k_length, batch_size, self.n_head, self.d_head)
        head_wv = head_wv.view(k_length, batch_size, self.n_head, self.d_head)

        # (head_length, n_head, d_head)
        head_rk = head_rk.view(r_length, self.n_head, self.d_head)

        if layer_past is not None:
            past_k, past_v, past_r = torch.unbind(layer_past, dim=0)
            past_r = past_r[:, 0, :, :]
            head_wk = torch.cat((past_k, head_wk), dim=0)
            head_wv = torch.cat((past_v, head_wv), dim=0)
            head_rk = torch.cat((head_rk, past_r), dim=0)

        if use_cache is True:
            _r_head_k = head_rk.unsqueeze(1).expand(-1, batch_size, -1, -1)
            present = torch.stack([head_wk, head_wv, _r_head_k], dim=0)
        else:
            present = None

        # Attention score
        # (q_length, batch_size, n_head, d_head)
        head_r_wq = head_wq + self.r_w_bias
        head_r_rq = head_wq + self.r_r_bias

        # (q_length, k_length, batch_size, n_head)
        AC = torch.einsum("ibnd,jbnd->ijbn", (head_r_wq, head_wk))
        BD = torch.einsum("ibnd,jnd->ijbn", (head_r_rq, head_rk))
        BD = self._relational_shift(BD)

        # (q_length, k_length, batch_size, h_head)
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # Attention probability
        if attn_mask is not None and torch.sum(attn_mask).item():
            # Switches to a boolean mask
            attn_mask = attn_mask == 1

            # Standard filling for 32-bit float precision
            fill = -1e30

            # If using 16-bit float precision, `fill` should be smaller
            if next(self.parameters()).dtype == torch.float16:
                fill = -65000

            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None], fill).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], fill).type_as(attn_score)

        # (q_length, k_length, batch_size, n_head)
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # Whether heads should be masked or not
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # Attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, head_wv))

        # (q_length, batch_size, n_head, d_head)
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o(attn_vec)
        attn_out = self.drop(attn_out)

        # Residual connection
        output = w + attn_out

        if not self.pre_lnorm:
            # Layer normalization
            output = self.layer_norm(output)

        output = [output, present]

        if output_attentions:
            output.append(attn_prob)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_head: int,
        d_inner: int,
        dropout: float,
        dropatt: Optional[float] = 0.0,
        primer_conv: Optional[bool] = False,
        primer_square: Optional[bool] = False,
        pre_lnorm: Optional[bool] = False,
        layer_norm_epsilon: Optional[float] = 1e-5,
        r_w_bias: Optional[torch.FloatTensor] = None,
        r_r_bias: Optional[torch.FloatTensor] = None,
    ) -> None:
        super().__init__()

        self.attn = RelPartialLearnableMultiHeadAttn(
            n_head,
            d_model,
            d_head,
            dropout,
            dropatt=dropatt,
            primer_conv=primer_conv,
            pre_lnorm=pre_lnorm,
            layer_norm_epsilon=layer_norm_epsilon,
            r_w_bias=r_w_bias,
            r_r_bias=r_r_bias,
        )

        if primer_square:
            self.pos_ff = PositionWiseFFPrimerEZ(
                d_model,
                d_inner,
                dropout,
                pre_lnorm=pre_lnorm,
                layer_norm_epsilon=layer_norm_epsilon,
            )
        else:
            self.pos_ff = PositionWiseFF(
                d_model,
                d_inner,
                dropout,
                pre_lnorm=pre_lnorm,
                layer_norm_epsilon=layer_norm_epsilon,
            )

    def forward(
        self,
        inputs: torch.FloatTensor,
        embeds: torch.FloatTensor,
        layer_past: Optional[torch.FloatTensor] = None,
        dec_attn_mask: Optional[torch.FloatTensor] = None,
        mems: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> torch.FloatTensor:
        attn_output = self.attn(
            inputs,
            embeds,
            layer_past=layer_past,
            attn_mask=dec_attn_mask,
            mems=mems,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        output = [self.pos_ff(attn_output[0])] + attn_output[1:]

        return output
