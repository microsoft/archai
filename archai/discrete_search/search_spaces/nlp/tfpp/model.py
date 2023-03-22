from typing import Any

from torch import nn
from transformers import PretrainedConfig
from archai.discrete_search.search_spaces.config import ArchConfig

from .backbones import BACKBONES, CONFIGS


class LanguageModel(nn.Module):
    def __init__(self, arch_config: ArchConfig, **hf_config_kwargs):
        super().__init__()
        
        self.backbone = arch_config.pick('backbone', default='codegen')
        self.hf_config = LanguageModel.get_hf_config_cls(arch_config)(**hf_config_kwargs)
        self.model = BACKBONES[self.backbone](arch_config, self.hf_config)
    
    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    @staticmethod
    def get_hf_config_cls(arch_config: ArchConfig) -> PretrainedConfig:
        backbone = arch_config.pick('backbone', default='codegen', record_usage=False)
        return CONFIGS[backbone]
