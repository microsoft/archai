# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Mixed Quantization Aware-Training model.
"""

import copy
from typing import Optional

import torch

from archai.nlp.compression.quantization.qat import prepare_with_qat
from archai.nlp.models.model_base import ArchaiModel


class MixedQATModel(ArchaiModel):
    def __init__(self, model: torch.nn.Module, qat_weight: Optional[float] = 0.2) -> None:
        super(MixedQATModel, self).__init__()

        if qat_weight < 0.0 or qat_weight > 1.0:
            raise ValueError(f'QAT weight: {qat_weight} should be between 0 and 1.')

        self.qat_weight = qat_weight
        self.regular_weight = 1.0 - qat_weight

        self.model = model
        self.qat_model = copy.deepcopy(model)

        # Shares all parameters
        for module1, module2 in zip(self.model.modules(), self.qat_model.modules()):
            if hasattr(module2, 'weight'):
                module2.weight = module1.weight
            if hasattr(module2, 'bias'):
                module2.bias = module1.bias

        # Adds fake quantization
        self.qat_model = prepare_with_qat(self.qat_model, onnx_compatible=True)

        # Makes sure that all parameters are shared
        for param1, param2 in zip(self.model.parameters(), self.qat_model.parameters()):
            assert param2 is param1, 'Mixed QAT parameters are not fully shared.'

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        qat_out = self.qat_model(*args, **kwargs)

        # If we are not training, we only return the QAT loss
        if self.training:
            return ((out[0] * self.regular_weight + qat_out[0] * self.qat_weight), out[1], out[2], out[3])
        
        return qat_out

    def reset_length(self, *args, **kwargs):
        return self.model.reset_length(*args, **kwargs)

    def get_params(self):
        return self.model.get_params()
