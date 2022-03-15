# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Dictionary that allows the implementation and usage of Transformer-based models.
"""

from archai.nlp.models.hf_gpt2.config_hf_gpt2 import HfGPT2Config, HfGPT2FlexConfig
from archai.nlp.models.hf_gpt2.model_hf_gpt2 import HfGPT2, HfGPT2Flex
from archai.nlp.models.hf_gpt2.onnx_hf_gpt2 import HfGPT2OnnxConfig, HfGPT2OnnxModel

from archai.nlp.models.hf_transfo_xl.config_hf_transfo_xl import HfTransfoXLConfig
from archai.nlp.models.hf_transfo_xl.model_hf_transfo_xl import HfTransfoXL
from archai.nlp.models.hf_transfo_xl.onnx_hf_transfo_xl import HfTransfoXLOnnxConfig, HfTransfoXLOnnxModel

from archai.nlp.models.mem_transformer.config_mem_transformer import MemTransformerLMConfig
from archai.nlp.models.mem_transformer.model_mem_transformer import MemTransformerLM
from archai.nlp.models.mem_transformer.onnx_mem_transformer import MemTransformerLMOnnxConfig, MemTransformerLMOnnxModel

from archai.nlp.models.model_formulae import get_params_hf_gpt2_formula, get_params_hf_gpt2_flex_formula, get_params_hf_transfo_xl_formula, get_params_mem_transformer_formula

# Available models and their configurations
MODELS = {
    'hf_gpt2': HfGPT2,
    'hf_gpt2_flex': HfGPT2Flex,
    'hf_transfo_xl': HfTransfoXL,
    'mem_transformer': MemTransformerLM
}

MODELS_CONFIGS = {
    'hf_gpt2': HfGPT2Config,
    'hf_gpt2_flex': HfGPT2FlexConfig,
    'hf_transfo_xl': HfTransfoXLConfig,
    'mem_transformer': MemTransformerLMConfig
}

MODELS_PARAMS_FORMULAE = {
    'hf_gpt2': get_params_hf_gpt2_formula,
    'hf_gpt2_flex': get_params_hf_gpt2_flex_formula,
    'hf_transfo_xl': get_params_hf_transfo_xl_formula,
    'mem_transformer': get_params_mem_transformer_formula
}

# Available ONNX-based models and their configurations
ONNX_MODELS = {
    'hf_gpt2': HfGPT2OnnxModel,
    'hf_gpt2_flex': HfGPT2OnnxModel,
    'hf_transfo_xl': HfTransfoXLOnnxModel,
    'mem_transformer': MemTransformerLMOnnxModel
}

ONNX_MODELS_CONFIGS = {
    'hf_gpt2': HfGPT2OnnxConfig,
    'hf_gpt2_flex': HfGPT2OnnxConfig,
    'hf_transfo_xl': HfTransfoXLOnnxConfig,
    'mem_transformer': MemTransformerLMOnnxConfig
}
