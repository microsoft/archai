# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Copyright (c) 2018, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.

from typing import Optional

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()

        self.d_model = d_model

        inv_freq = 1 / (1e4 ** (torch.arange(0, d_model, 2) / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, inputs: torch.FloatTensor, batch_size: Optional[int] = None) -> torch.FloatTensor:
        # Generates a positional embedding through sinusoids
        inputs_sinusoid = torch.ger(inputs, self.inv_freq)
        embed_pos = torch.cat([inputs_sinusoid.sin(), inputs_sinusoid.cos()], dim=-1)

        # If a batch size is supplied, expand the tensor to comply with it
        if batch_size is not None:
            return embed_pos[:, None, :].expand(-1, batch_size, -1)
        else:
            return embed_pos[:, None, :]
