from typing import Dict, Any, Callable, Optional, Union, List
from collections import OrderedDict
from copy import deepcopy
from random import Random

from archai.discrete_search.search_spaces.builder import utils
from archai.discrete_search.search_spaces.builder.repeat_config import RepeatConfig
from archai.discrete_search.search_spaces.builder.discrete_choice import DiscreteChoice
from archai.discrete_search.search_spaces.builder.arch_config import ARCH_CONFIGS, ArchConfig


class ArchParamTree(object):
    def __init__(self, config_tree: Dict[str, Any]):
        self.config_tree = deepcopy(config_tree)
        self.params, self.constants = self._init_tree(config_tree)
        self.free_params = self.to_dict(deduplicate_params=True)

    def _init_tree(self, config_tree: Dict[str, Any]) -> Tuple[OrderedDict, OrderedDict]:
        params, constants = OrderedDict(), OrderedDict()

        for param_name, param in config_tree.items():
            # Converts special config operations to config dict form
            param = param.to_config_dict() if isinstance(param, RepeatConfig) else param
            
            if isinstance(param, DiscreteChoice):
                params[param_name] = param
            elif isinstance(param, dict):
                params[param_name] = ArchParamTree(param)
            else:
                constants[param_name] = param
        
        return params, constants

    def _to_dict(self, prefix: str, flatten: bool, dedup_param_ids: Optional[set] = None) -> OrderedDict:
        prefix = f'{prefix}.' if prefix else prefix

        # Initializes output dictionary with constants
        output_dict = OrderedDict([
            (prefix + c_name, c_value)
            for c_name, c_value in deepcopy(self.constants).items()
        ])

        # Adds arch parameters to `output_dict`
        for param_name, param in self.params.items():
            param_name = prefix + str(param_name) if flatten else str(param_name)

            if isinstance(param, ArchParamTree):
                param_dict = param._to_dict(param_name, flatten, dedup_param_ids)

                if flatten:
                    output_dict.update(param_dict)
                else:
                    output_dict[param_name] = param_dict
            
            elif isinstance(param, DiscreteChoice):
                if dedup_param_ids is None or id(param) not in dedup_param_ids:
                    output_dict[param_name] = param
                    
                    if dedup_param_ids:
                        dedup_param_ids.add(id(param))

        return output_dict
    
    def to_dict(self, flatten: bool = False, deduplicate_params: bool = False) -> OrderedDict:
        return self._to_dict('', flatten, set() if deduplicate_params else None)

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
        
        config_type = sample.get('_config_type', 'default')
        return ARCH_CONFIGS[config_type](sample)

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
