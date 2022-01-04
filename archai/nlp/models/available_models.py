# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nlp.models.hf_gpt2 import HfGPT2
from archai.nlp.models.hf_transfo_xl import HfTransfoXL
from archai.nlp.models.mem_transformer import MemTransformerLM

# List of available models
AVAILABLE_MODELS = {
    'mem_transformer': MemTransformerLM,
    'hf_gpt2': HfGPT2,
    'hf_transfo_xl': HfTransfoXL
}
