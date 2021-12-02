from archai.nlp.nvidia_transformer_xl.models.mem_transformer import MemTransformerLM
from archai.nlp.nvidia_transformer_xl.models.hf_gpt2 import HfGPT2

# Maps argument string/name to class
MODEL_LIST = {'mem_transformer': MemTransformerLM,
              'hf_gpt2': HfGPT2}