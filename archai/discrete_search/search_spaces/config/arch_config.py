from typing import Dict, Union, Any
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import json
import yaml

def build_arch_config(config_dict: Dict) -> 'ArchConfig':
    """Builds an `ArchConfig` object from a sampled config dictionary.

    Args:
        config_dict (Dict): Config dictionary

    Returns:
        ArchConfig
    """    
    ARCH_CONFIGS = {
        'default': ArchConfig,
        'config_list': ArchConfigList
    }
    
    config_type = config_dict.get('_config_type', 'default')
    return ARCH_CONFIGS[config_type](config_dict)


class ArchConfig():
    def __init__(self, config_dict: Dict[str, Union[dict, float, int, str]]):
        """Stores architecture configs.

        Args:
            config_dict (Dict[str, Union[dict, float, int, str]]): Sampled configuration
        """        
        # Set that stores all parameters used to build the model instance
        self._used_params = set()
        
        # Original config dictionary
        self._config_dict = deepcopy(config_dict)

        # ArchConfig nodes
        self.nodes = OrderedDict()

        for param_name, param in self._config_dict.items():
            if isinstance(param, dict):
                self.nodes[param_name] = build_arch_config(param)
            else:
                self.nodes[param_name] = param
    
    def __repr__(self) -> str:
        class ArchConfigJsonEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, ArchConfig):
                    return o.to_dict(remove_metadata_info=True)
                
                return super().default(o)

        cls_name = self.__class__.__name__
        return f'{cls_name}({json.dumps(self, cls=ArchConfigJsonEncoder, indent=4)})'

    def get_used_params(self) -> Dict[str, Union[Dict, bool]]:
        """Gets the parameter usage tree. Terminal nodes with value `True`
        represent architecture parameters that were used by calling 
        `ArchConfig.pick(param_name)`.

        Returns:
            Dict[str, bool]: Used parameters
        """
        used_params = OrderedDict()
        
        for param_name, param in self.nodes.items():
            used_params[param_name] = param_name in self._used_params

            if isinstance(param, ArchConfig):
                used_params[param_name] = param.get_used_params()

        return used_params

    def pick(self, param_name: str, record_usage: bool = True) -> Any:
        """Picks an architecture parameter, possibly recording its usage.

        Args:
            param_name (str): Architecture parameter name
            
            record_usage (bool, optional): If this parameter should be recorded
                as 'used' in `ArchConfig._used_params`. Defaults to True.

        Returns:
            Any: Parameter value
        """
        param_value = self.nodes[param_name]
        
        if record_usage:
            self._used_params.add(param_name)

        return param_value
    
    def to_dict(self, remove_metadata_info: bool = False) -> OrderedDict:
        """Converts ArchConfig object to an ordered dictionary.

        Args:
            remove_metadata_info (bool, optional): If keys used to store
            extra metadata should be removed. Defaults to False.

        Returns:
            OrderedDict
        """        
        return OrderedDict(
            (k, v.to_dict(remove_metadata_info)) if isinstance(v, ArchConfig) else (k, v)
            for k, v in self.nodes.items()
            if not remove_metadata_info or not k.startswith('_')
        )

    def to_file(self, path: str) -> None:        
        d = self.to_dict()
        yaml.dump(d, open(path, 'w', encoding='utf-8'), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_file(cls, path: str) -> 'ArchConfig':
        path = Path(path)

        if path.suffix == '.json':
            # For compatibility with older versions
            d = json.load(open(path, 'r', encoding='utf-8'))
        elif path.suffix == '.yaml':
            d = yaml.load(open(path, 'r', encoding='utf-8'), Loader=yaml.Loader)
        
        return build_arch_config(d)


class ArchConfigList(ArchConfig):
    def __init__(self, config: OrderedDict):
        super().__init__(config)

        assert '_configs' in config
        assert '_repeat_times' in config

        self.max_size = config['_repeat_times']
    
    def __len__(self) -> int:
        self._used_params.add('_repeat_times')
        return self.max_size
    
    def __getitem__(self, idx: int) -> ArchConfig:
        if 0 <= idx < len(self):
            self._used_params.add('_repeat_times')
            return self.nodes['_configs'].pick(str(idx))
        raise IndexError

    def __iter__(self):
        yield from [self[i] for i in range(len(self))]

    def pick(self, param_name: str, record_usage: bool = True):
        raise ValueError(
            'Attempted to use .pick in an ArchConfigList instance. '
            'Select a config first using indexing (e.g `config_list[i]`).'
        )

    def to_dict(self, remove_metadata_info: bool = False):
        if remove_metadata_info:
            blocks = [
                self.nodes['_configs'].pick(str(i), record_usage=False).to_dict()
                for i in range(self.max_size)
            ]

            return blocks
        
        return super().to_dict(remove_metadata_info)
