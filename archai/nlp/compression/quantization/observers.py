# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Quantization-ready observers.
"""

import torch


class OnnxDynamicObserver:
    """Provides a DynamicObserver that is compliant with ONNX-based graphs.

    Note that usually `qint8` is used for symmetric quantization, while
    `quint8` is used for assymetric quantization.

    """

    def __init__(self, dtype: str) -> None:
        """Initializes the class by setting appropriate values for quantization bounds.

        Args:
            dtype: Type of quantization operators.

        """

        self.dtype = dtype
        self.eps = torch.finfo(torch.float32).eps

        assert dtype in (torch.quint8, torch.qint8)

        if dtype == torch.quint8:
            self.qmin, self.qmax = 0, 255
        else:
            self.qmin, self.qmax = -128, 127

    def __call__(self, x: torch.Tensor) -> None:
        """Performs a call to set minimum and maximum tensor values.

        Args:
            x: Input tensor.

        """

        x = x.detach().float()

        self.min_val, self.max_val = x.min().view(-1), x.max().view(-1)

    def calculate_qparams(self) -> None:
        """Calculates the quantization parameters.

        """

        if self.dtype == torch.qint8:
            scale = torch.max(self.max_val.clamp(min=0), -self.min_val.clamp(max=0)) / 127
            zero_point = torch.zeros_like(scale).to(torch.int64)

            return scale.clamp(min=self.eps), zero_point

        else:
            scale = (self.max_val - self.min_val) / float(self.qmax - self.qmin)
            scale = scale.clamp(min=self.eps)

            zero_point = self.qmin - torch.round(self.min_val / scale)
            zero_point = zero_point.clamp(min=self.qmin, max=self.qmax).to(torch.int64)

            return scale, zero_point
