from .codegen.model import CodeGenForCausalLM, CodeGenConfig
from .gpt2.model import GPT2LMHeadModel, GPT2Config

BACKBONES = {
    'codegen': CodeGenForCausalLM,
    'gpt2': GPT2LMHeadModel
}

CONFIGS = {
    'codegen': CodeGenConfig,
    'gpt2': GPT2Config
}
