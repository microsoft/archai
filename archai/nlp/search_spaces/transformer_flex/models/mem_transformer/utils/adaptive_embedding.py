# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) 2018, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.

"""Adaptive Embedding layer."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        d_model: int,
        cutoffs: Tuple[int],
        div_val: Optional[int] = 1,
        sample_softmax: Optional[bool] = False,
        fp16: Optional[bool] = False,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_model = d_model
        self.div_val = div_val

        self.cutoffs = cutoffs + [vocab_size]
        self.cutoffs_ends = [0] + self.cutoffs

        self.n_clusters = len(self.cutoffs) - 1

        self.emb_scale = d_model**0.5
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()

        if div_val == 1:
            self.emb_layers.append(nn.Embedding(vocab_size, d_embed, sparse=sample_softmax > 0))

            if d_model != d_embed:
                self.emb_projs.append(nn.Parameter(torch.zeros(d_model, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                d_embed_i = d_embed // (div_val**i)
                d_out_i = self.cutoffs_ends[i + 1] - self.cutoffs_ends[i]

                self.emb_layers.append(nn.Embedding(d_out_i, d_embed_i))
                self.emb_projs.append(nn.Parameter(torch.zeros(d_model, d_embed_i)))

        if fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        if self.div_val == 1:
            embed = self.emb_layers[0](inputs)

            if self.d_model != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            inputs_flatten = inputs.view(-1)
            embed_flatten = torch.zeros(
                [inputs_flatten.size(0), self.d_model],
                dtype=self.dtype,
                device=inputs_flatten.device,
            )

            # Every cutoff should be considered for calculating final embeddings
            for i in range(len(self.cutoffs)):
                cutoff_start, cutoff_end = (
                    self.cutoffs_ends[i],
                    self.cutoffs_ends[i + 1],
                )

                # Gathers a mask of valid indexes
                mask_i = (inputs_flatten >= cutoff_start) & (inputs_flatten < cutoff_end)
                indexes_i = mask_i.nonzero().squeeze()

                if indexes_i.numel() == 0:
                    continue

                inputs_i = inputs_flatten.index_select(0, indexes_i) - cutoff_start

                embed_i = self.emb_layers[i](inputs_i)
                embed_i = F.linear(embed_i, self.emb_projs[i]).to(self.dtype)

                embed_flatten.index_copy_(0, indexes_i, embed_i)

            embed_shape = inputs.size() + (self.d_model,)
            embed = embed_flatten.view(embed_shape)

        embed.mul_(self.emb_scale)

        return embed
