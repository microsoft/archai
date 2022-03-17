# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Mixed Quantization Aware-Training model.
"""

import copy
from typing import Dict, Optional, Tuple

import torch

from archai.nlp.compression.quantization.qat import prepare_with_qat
from archai.nlp.models.model_base import ArchaiModel


class MixedQATModel(ArchaiModel):
    """Implements a mixed QAT model, which can be fine-tuned using a linear combination of
        regular and QAT losses.

    """

    def __init__(self, model: torch.nn.Module, qat_weight: Optional[float] = 0.2) -> None:
        super(MixedQATModel, self).__init__()

        if qat_weight < 0.0 or qat_weight > 1.0:
            raise ValueError(f'QAT weight: {qat_weight} should be between 0 and 1.')

        self.qat_weight = qat_weight
        self.regular_weight = 1.0 - qat_weight

        self.model = model
        self.qat_model = copy.deepcopy(model)

        # Shares all parameters
        for module, qat_module in zip(self.model.modules(), self.qat_model.modules()):
            if hasattr(qat_module, 'weight'):
                qat_module.weight = module.weight
            if hasattr(qat_module, 'bias'):
                qat_module.bias = module.bias

        # Adds fake quantization
        self.qat_model = prepare_with_qat(self.qat_model, onnx_compatible=True)

        # Makes sure that all parameters are shared
        for param, qat_param in zip(self.model.parameters(), self.qat_model.parameters()):
            assert qat_param is param, 'Mixed QAT parameters are not fully shared.'

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        out = self.model(*args, **kwargs)
        qat_out = self.qat_model(*args, **kwargs)

        # If training, returns the linear combination of losses
        if self.training:
            return ((out[0] * self.regular_weight + qat_out[0] * self.qat_weight), out[1], out[2], out[3])
        
        return qat_out

    def reset_length(self, *args, **kwargs) -> None:
        return self.model.reset_length(*args, **kwargs)

    def get_params_from_layer(self, layer_type: str) -> int:
        return self.model.get_params_from_layer(layer_type)
        
    def get_params(self) -> Dict[str, int]:
        return self.model.get_params()
