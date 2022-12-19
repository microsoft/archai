# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Quantization-based classes, methods and definitions."""

from archai.nlp.quantization.mixed_qat import MixedQAT
from archai.nlp.quantization.ptq import (
    dynamic_quantization_onnx,
    dynamic_quantization_torch,
)
from archai.nlp.quantization.qat import prepare_with_qat
