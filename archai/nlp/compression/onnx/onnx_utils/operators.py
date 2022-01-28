# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Additional operators that are not natively supported by ONNX.
"""

from typing import Optional

import torch


def triu_onnx(inputs: torch.FloatTensor, diagonal: Optional[int] = 0) -> torch.FloatTensor:
    """Caveat to export a triu-based operator with ONNX.

    Args:
        inputs: Input tensor.
        diagonal: Value of diagonal.

    Returns:
        (torch.FloatTensor): Output tensor.

    """

    arange = torch.arange(inputs.size(0), device=inputs.device)
    arange2 = torch.arange(inputs.size(1), device=inputs.device)

    mask = arange.unsqueeze(-1).expand(-1, inputs.size(1)) <= (arange2 - diagonal)

    return inputs.masked_fill(mask == 0, 0)


def tril_onnx(inputs: torch.FloatTensor, diagonal: Optional[int] = 0) -> torch.FloatTensor:
    """Caveat to export a tril-based operator with ONNX.

    Args:
        inputs: Input tensor.
        diagonal: Value of diagonal.

    Returns:
        (torch.FloatTensor): Output tensor.

    """

    arange = torch.arange(inputs.size(0), device=inputs.device)
    arange2 = torch.arange(inputs.size(1), device=inputs.device)

    mask = arange.unsqueeze(-1).expand(-1, inputs.size(1)) >= (arange2 - diagonal)

    return inputs.masked_fill(mask == 0, 0)


def register_trilu_operator() -> None:
    """Register triu/tril operators to make them available at ORT.

    """

    def triu(g, self, diagonal):
        return g.op('com.microsoft::Trilu', self, diagonal)
    torch.onnx.register_custom_op_symbolic('::triu', triu, 1)

    def tril(g, self, diagonal):
        return g.op('com.microsoft::Trilu', self, diagonal, upper=False)
    torch.onnx.register_custom_op_symbolic('::tril', tril, 1)
