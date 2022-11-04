# Copyright (c) 2018 Google AI, Google Brain, Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.

"""Position-Wise Feed-Forward layer.
"""

from typing import Optional

import torch
import torch.nn as nn


class PositionWiseFF(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dropout: float,
        pre_lnorm: Optional[bool] = False,
        layer_norm_epsilon: Optional[float] = 1e-5,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        if self.pre_lnorm:
            output = self.ff(self.layer_norm(inputs))
            output += inputs
        else:
            output = self.ff(inputs)
            output = self.layer_norm(inputs + output)

        return output
