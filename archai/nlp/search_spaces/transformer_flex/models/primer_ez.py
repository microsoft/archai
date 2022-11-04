# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Primer-EZ primitives.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWiseConvPrimerEZ(nn.Module):
    """Depth-wise convolution: https://arxiv.org/abs/2109.08668.

    """

    def __init__(self,
                 d_model: int,
                 kernel_size: Optional[int] = 3) -> None:
        super().__init__()

        self.kernel_size = kernel_size

        # Depthwise convolution: groups == in_channels
        self.dconv = nn.Conv1d(d_model*3, d_model*3, kernel_size=kernel_size, groups=d_model*3)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # LxBxF -> BxFxL
        w_heads = inp.permute((1, 2, 0))

        # Pad kernel_size-1 to the left of the length so we have causal convolution (can't look forward)
        w_heads = F.pad(w_heads, (self.kernel_size-1, 0))
        w_heads = self.dconv(w_heads)

        # Permute back: BxFxL -> LxBxF
        w_heads = w_heads.permute((2, 0, 1))

        return w_heads


class PositionWiseFFPrimerEZ(nn.Module):
    """Squared ReLU: https://arxiv.org/abs/2109.08668.
    
    """

    def __init__(self,
                 d_model: int,
                 d_inner: int,
                 dropout: float,
                 pre_lnorm: Optional[bool] = False) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet1 = nn.Sequential(nn.Linear(d_model, d_inner), nn.ReLU(inplace=True))
        self.CoreNet2 = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(d_inner, d_model),
                                      nn.Dropout(dropout))

        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.pre_lnorm:
            inp = self.layer_norm(inp)

        core_out = self.CoreNet2(self.CoreNet1(inp) ** 2)
        output = core_out + inp

        if not self.pre_lnorm:
            output = self.layer_norm(output)

        return output
