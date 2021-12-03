from archai.nlp.nvidia_transformer_xl.models.mem_transformer import MemTransformerLM
from archai.nlp.nvidia_transformer_xl.models.hf_gpt2 import HfGPT2
from archai.nlp.nvidia_transformer_xl.models.hf_transfo_xl import HfTransfoXL

# Maps argument string/name to class
MODEL_LIST = {'mem_transformer': MemTransformerLM,
              'hf_gpt2': HfGPT2,
              'hf_transfo_xl': HfTransfoXL}
