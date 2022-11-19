from typing import Dict, Any, Callable, Optional, Union, List
from collections import OrderedDict
from copy import deepcopy
from random import Random

from archai.discrete_search.search_spaces.builder.discrete_choice import DiscreteChoice
from archai.discrete_search.search_spaces.builder.arch_config import ArchConfig


class ArchParamTree(object):
    def __init__(self, param_tree: Dict[str, Any]):
        self.config_tree = deepcopy(param_tree)
        self.params, self.constants = self._get_params_and_constants(param_tree)

    def _get_params_and_constants(self, config_tree: Dict[str, Any]):
        param_tree, constants = {}, {}

        # map from id(param) -> ArchParamTree | DiscreteChoice
        ref_map = {}

        for param_name, param in config_tree.items():
            # Preserves references to an object already added to the tree.
            # This makes sharing arch params possible
            if isinstance(param, (DiscreteChoice, dict)) and id(param) in ref_map:
                param_tree[param_name] = ref_map[id(param)]
            
            elif isinstance(param, (DiscreteChoice, ArchParamTree)):
                param_tree[param_name] = param
                ref_map[id(param)] = param
                
            elif isinstance(param, dict):
                param_tree[param_name] = ArchParamTree(param)
                ref_map[id(param)] = param

            else:
                constants[param_name] = param
        
        return param_tree, constants

    def _sample_config(self, rng: Random, ref_map: Dict[int, Any]):
        # Initializes empty dict with constants already set
        sample = deepcopy(self.constants)

        for param_name, param in self.params.items():
            if isinstance(param, ArchParamTree):
                sample[param_name] = param._sample_config(rng, ref_map)

            elif isinstance(param, DiscreteChoice):
                # Only samples params not sampled before
                if id(param) not in ref_map:
                    sampled_param = rng.choice(param.choices)
                    ref_map[id(param)] = sampled_param
                
                sample[param_name] = ref_map[id(param)]
        
        return ArchConfig(sample)

    def sample_config(self, rng: Optional[Random] = None):
        return self._sample_config(rng or Random(), {})
    
    def get_param_name_list(self, prefix: str = '') -> List[str]:
        param_names = []

        for param_name, param in self.params.items():
            if isinstance(param, ArchParamTree):
                subtree_prefix =  prefix + f'.{param_name}' if prefix else param_name
                param_names += param.get_param_name_list(subtree_prefix)
            else:
                param_names += [f'{prefix}.{param_name}' if prefix else param_name]

        return param_names

    def encode_config(self, config: ArchConfig, drop_duplicates: bool = True) -> List[float]:
        arch_vector = []
        used_params = config.get_used_params()

        for param_name, param in self.params.items():
            config_value = config.config_tree[param_name]

            if isinstance(param, ArchParamTree):
                arch_vector += param.encode_config(config_value)
            else:
                arch_vector += [config_value if used_params[param_name] else float('NaN')]
        
        return arch_vector
