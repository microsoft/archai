from typing import Dict, Any, Callable, Optional, Union, List, Tuple
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

    def _to_dict(self, prefix: str, flatten: bool, dedup_param_ids: Optional[set] = None,
                 remove_constants: bool = True) -> OrderedDict:
        prefix = f'{prefix}.' if prefix else prefix
        output_dict = OrderedDict()

        # if `remove_constants`, initializes the output dictionary with constants first
        if not remove_constants:
            output_dict = OrderedDict([
                (prefix + c_name if flatten else c_name, c_value)
                for c_name, c_value in deepcopy(self.constants).items()
            ])

        # Adds architecture parameters to the output dictionary
        for param_name, param in self.params.items():
            param_name = prefix + str(param_name) if flatten else str(param_name)

            if isinstance(param, ArchParamTree):
                param_dict = param._to_dict(
                    param_name, flatten,
                    dedup_param_ids, remove_constants
                )

                if flatten:
                    output_dict.update(param_dict)
                else:
                    output_dict[param_name] = param_dict
            
            elif isinstance(param, DiscreteChoice):
                if dedup_param_ids is None:
                    output_dict[param_name] = param
                elif id(param) not in dedup_param_ids:
                    output_dict[param_name] = param
                    dedup_param_ids.add(id(param))

        return output_dict
    
    def to_dict(self, flatten: bool = False, deduplicate_params: bool = False,
                remove_constants: bool = True) -> OrderedDict:
        """Converts the ArchParamTree to an ordered dictionary.

        Args:
            flatten (bool, optional): If the output dictionary should
                be flattened. Defaults to False.
            
            deduplicate_params (bool, optional): Removes duplicate architecture
                parameters. Defaults to False.
            
            remove_constants (bool, optional): Removes attributes that are not
                architecture params from the output dictionary. Defaults to True.

        Returns:
            OrderedDict
        """        
        return self._to_dict('', flatten, set() if deduplicate_params else None, remove_constants)

    def _sample_config(self, rng: Random, ref_map: Dict) -> ArchConfig:
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

    def sample_config(self, rng: Optional[Random] = None) -> ArchConfig:
        """Samples an architecture config from the search param tree.

        Args:
            rng (Optional[Random], optional): Random number generator used during sampling.
                If set to `None`, `random.Random()` is used. Defaults to None.

                
        Returns:
            ArchConfig: Sampled architecture config
        """        
        return self._sample_config(rng or Random(), {})
    
    def get_param_name_list(self) -> List[str]:
        param_dict = self.to_dict(flatten=True, deduplicate_params=True, remove_constants=True)
        return list(param_dict.keys())

    def encode_config(self, config: ArchConfig, track_unused_params: bool = True) -> List[float]:
        """Encodes an `ArchConfig` object into a fixed-length vector of features.
        This method should be used after the model object is created.

        Args:
            config (ArchConfig): Architecture configuration
            
            track_unused_params (bool): If `track_unused_params=True`, parameters
                not used during model creation (by calling `config.pick`) 
                will be represented as `float("NaN")`.

        Returns:
            List[float]
        """
        if not track_unused_params:
            raise NotImplementedError

        arch_vector = []
        used_params = config.get_used_params()

        for param_name, param in self.params.items():
            config_value = config.config_tree[param_name]

            if isinstance(param, ArchParamTree):
                arch_vector += param.encode_config(config_value)
            else:
                arch_vector += [config_value if used_params[param_name] else float('NaN')]
        
        return arch_vector
