# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
from typing import Optional, Tuple

import torch

from archai.quantization.qat import prepare_with_qat


class MixedQAT(torch.nn.Module):
    """Mixed QAT (Quantization-Aware Training) model, which can be fine-tuned
    using a linear combination of regular and QAT losses.

    """

    def __init__(self, model: torch.nn.Module, qat_weight: Optional[float] = 0.2) -> None:
        """Initialize the class by creating standard and QAT-based attributes
        of the incoming model.

        Args:
            model: Instance of the model that will be fine-tuned with Mixed QAT.
            qat_weight: Amount of QAT-based loss that should be used in the linear combination.
                This value should be between 0 and 1.
        """

        super().__init__()

        if qat_weight < 0.0 or qat_weight > 1.0:
            raise ValueError(f"qat_weight: {qat_weight} should be between 0 and 1.")

        self.qat_weight = qat_weight
        self.regular_weight = 1.0 - qat_weight

        self.model = model
        self.qat_model = copy.deepcopy(model)

        # Shares all parameters
        for module, qat_module in zip(self.model.modules(), self.qat_model.modules()):
            if hasattr(qat_module, "weight"):
                qat_module.weight = module.weight
            if hasattr(qat_module, "bias"):
                qat_module.bias = module.bias

        # Adds fake quantization
        prepare_with_qat(self.qat_model, onnx_compatible=True)

        for param, qat_param in zip(self.model.parameters(), self.qat_model.parameters()):
            assert qat_param is param, "MixedQAT parameters are not fully shared."

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(*args, **kwargs)
        qat_outputs = self.qat_model(*args, **kwargs)

        # If training, returns the linear combination of losses
        if self.training:
            loss = outputs.loss * self.regular_weight + qat_outputs.loss * self.qat_weight
            return (loss,) + outputs[1:]

        return qat_outputs
