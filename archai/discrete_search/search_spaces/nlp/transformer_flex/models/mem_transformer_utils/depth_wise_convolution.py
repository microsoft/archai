# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: Optional[int] = 3) -> None:
        super().__init__()

        # Depth-Wise Convolution: https://arxiv.org/abs/2109.08668
        self.kernel_size = kernel_size
        self.dconv = nn.Conv1d(d_model * 3, d_model * 3, kernel_size=kernel_size, groups=d_model * 3)

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        # LxBxF -> BxFxL
        w_heads = inputs.permute((1, 2, 0))

        # Pad kernel_size-1 to the left of the length
        # so we have causal convolution (can't look forward)
        w_heads = F.pad(w_heads, (self.kernel_size - 1, 0))
        w_heads = self.dconv(w_heads)

        # Permute back: BxFxL -> LxBxF
        w_heads = w_heads.permute((2, 0, 1))

        return w_heads
