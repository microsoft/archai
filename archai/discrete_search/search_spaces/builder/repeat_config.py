from typing import Dict, Any, List
from copy import deepcopy
from random import Random

from archai.discrete_search.search_spaces.builder.discrete_choice import DiscreteChoice
from archai.discrete_search.search_spaces.builder.arch_config import ArchConfig
from archai.discrete_search.search_spaces.builder.arch_param_tree import ArchParamTree


class RepeatConfig(ArchParamTree):
    def __init__(self, config_dict: Dict[str, Any], repeat_times: List[int], share_arch: bool = False):
        assert min(repeat_times) >= 0
        assert max(repeat_times) > 0

        self.repeat_times = repeat_times
        self.share_arch = share_arch
        
        config_dict = deepcopy(config_dict)

        # If arch params should be shared between blocks, re-uses the same reference to `config_dict`
        # instead of creating a new copy
        config_dict = {
            'configs': {
                i: config_dict if self.share_arch else deepcopy(config_dict)
                for i in range(max(repeat_times))
            },
            'share_arch': share_arch,
            'repeat_times': DiscreteChoice(repeat_times)
        }
        super().__init__(config_dict)
        self._blank_encoded_config = self._get_blank_encoded_config()

    def _get_blank_encoded_config(self):
        config = super()._sample_config(Random(), {})
        return super().encode_config(config)
        
    def _sample_config(self, rng: Random, ref_map: Dict[int, Any]):
        config = super()._sample_config(rng, ref_map)
        sampled_repeat_times = config.config_tree['repeat_times']
        sampled_block_configs = config.config_tree['configs'].config_tree

        # Return each block ArchConfig in order
        return [
            sampled_block_configs[i]
            for i in range(sampled_repeat_times)
        ]

    def encode_config(self, config_list: List[ArchConfig], drop_duplicates: bool = True) -> List[float]:
        encoder = super().encode_config
        e_configs = [encoder(config) for config in config_list]
        e_configs += [
            deepcopy(self._blank_encoded_config)
            for _ in range(len(config_list), max(self.repeat_times))
        ]

        return [
            e_param
            for e_config in e_configs
            for e_param in e_config
        ]
