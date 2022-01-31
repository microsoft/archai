# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from archai.nlp.models.hf_gpt2 import HfGPT2
from archai.nlp.models.hf_transfo_xl import HfTransfoXL
from archai.nlp.models.mem_transformer import MemTransformerLM
from archai.nlp.models.hf_gpt2_flex import HfGPT2Flex

# List of available models
AVAILABLE_MODELS = {
    'mem_transformer': MemTransformerLM,
    'hf_gpt2': HfGPT2,
    'hf_transfo_xl': HfTransfoXL,
    'hf_gpt2_flex': HfGPT2Flex
}
