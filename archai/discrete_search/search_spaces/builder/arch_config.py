from typing import Dict, Any, List, Optional, Union
from collections import OrderedDict
from copy import deepcopy
import json


def build_arch_config(choice_dict) -> 'ArchConfig':
    ARCH_CONFIGS = {
        'default': ArchConfig,
        'config_list': ArchConfigList
    }
    
    config_type = choice_dict.get('_config_type', 'default')
    return ARCH_CONFIGS[config_type](choice_dict)

    
class ArchConfig():
    def __init__(self, config: Dict[str, Union[dict, float, int, str]]):
        self._used_params = set()
        self._config_dict = deepcopy(config)
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
                    return o.to_dict(remove_config_keys=True)
                
                return super().default(o)

        cls_name = self.__class__.__name__
        return f'{cls_name}({json.dumps(self, cls=ArchConfigJsonEncoder, indent=4)})'

    def get_used_params(self) -> Dict[str, bool]:
        used_params = OrderedDict()
        
        for param_name, param in self.nodes.items():
            used_params[param_name] = param_name in self._used_params

            if isinstance(param, ArchConfig):
                used_params[param_name] = param.get_used_params()

        return used_params

    def pick(self, param_name: str, record_usage: bool = True):
        param_value = self.nodes[param_name]
        
        if record_usage:
            self._used_params.add(param_name)

        return param_value
    
    def to_dict(self, remove_config_keys: bool = False):
        return OrderedDict(
            (k, v.to_dict(remove_config_keys)) if isinstance(v, ArchConfig) else (k, v)
            for k, v in self.nodes.items()
            if not remove_config_keys or not k.startswith('_')
        )

    def to_json(self, path: str) -> None:
        d = self.to_dict()
        json.dump(d, open(path, 'w', encoding='utf-8'), indent=4)

    @classmethod
    def from_json(cls, path: str) -> 'ArchConfig':
        d = json.load(open(path, encoding='utf-8'), object_pairs_hook_=OrderedDict)
        return cls(d)


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

    def to_dict(self, remove_config_keys: bool = False):
        if remove_config_keys:
            blocks = [
                self.nodes['_configs'].pick(str(i), record_usage=False).to_dict()
                for i in range(self.max_size)
            ]

            return blocks
        
        return super().to_dict(remove_config_keys)
