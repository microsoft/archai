from typing import Dict, Any, List, Optional
from collections import OrderedDict
from copy import deepcopy
import json


class ArchConfig():
    def __init__(self, config_tree: OrderedDict):
        self.config_tree = deepcopy(config_tree)
        self._used_params = set()
    
    def to_dict(self):
        return OrderedDict(
            (k, v.to_dict()) if isinstance(v, ArchConfig) else (k, v)
            for k, v in self.config_tree.items()
        )

    def __repr__(self) -> str:
        return f'ArchConfig({json.dumps(self, indent=4, cls=ArchConfigJsonEncoder)})'

    def get_used_params(self) -> Dict[str, bool]:
        used_params = OrderedDict()
        
        for param_name, param in self.config_tree.items():
            used_params[param_name] = param_name in self._used_params

            if isinstance(param, ArchConfig):
                used_params[param_name] = param.get_used_params()

        return used_params

    def pick(self, param_name: str, record_usage: bool = True):
        param_value = self.config_tree[param_name]
        
        if record_usage:
            self._used_params.add(param_name)

        return param_value


class ArchConfigList(ArchConfig):
    def __init__(self, config_tree: Dict[str, Any]):
        super().__init__(config_tree)

        assert '_configs' in config_tree
        assert '_repeat_times' in config_tree

        self.max_size = config_tree['_repeat_times']

    def to_dict(self):
        return [
            self.config_tree['_configs'].config_tree[i].to_dict() 
            for i in range(self.max_size)
        ]

    def __repr__(self) -> str:
        return f'ArchConfigList({json.dumps(self, indent=4, cls=ArchConfigJsonEncoder)})'

    def __len__(self) -> int:
        return self.max_size
    
    def __getitem__(self, idx: int) -> ArchConfig:
        if 0 <= idx < len(self):
            self._used_params.add('_repeat_times')
            return self.config_tree['_configs'].config_tree[idx]
        raise IndexError

    def __iter__(self):
        yield from [self[i] for i in range(len(self))]

    def pick(self, param_name: str, record_usage: bool = True):
        raise ValueError(
            'Attempted to use .pick in an ArchConfigList instance. '
            'Select a config first using indexing (e.g `config_list[i]`).'
        )

    
class ArchConfigJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ArchConfig):
            return o.to_dict()
        
        return super().default(o)


ARCH_CONFIGS = {
    'default': ArchConfig,
    'config_list': ArchConfigList
}
