from typing import Union, List, Dict, Type, Optional
from collections import OrderedDict
from copy import deepcopy

from torch import nn
from archai.discrete_search.search_spaces.builder import DiscreteChoice

        
class Cell(nn.Module):
    _config: Dict
    _choices: Dict[str, DiscreteChoice]
    _cells: Dict[str, Union['Cell', List['Cell']]]
    _used_params: OrderedDict
    _lock: bool

    def __init__(self):
        super().__init__()
        
    def __getattribute__(self, attr_name):
        attr = super().__getattribute__(attr_name)
        
        # If attr is a DiscreteChoice object, gets
        # the specified value from `_config`.
        if isinstance(attr, DiscreteChoice):
            if not hasattr(self, '_config'):
                raise ValueError('Cell not initialized correctly. Did you try to use __init__()?')
            
            if not self._lock:
                self._used_params[attr_name] = True

            return self._config[attr_name]
        
        # If the user is trying to acess a Cell,
        # builds the module using `_config`
        if attr_name != '__class__' and isinstance(attr, type(Cell)):
            cell = self._cells[attr_name]  
            
            if not self._lock:          
                self._used_params[attr_name] = cell._used_params
            
            return cell
        
        if attr_name != '__class__' and isinstance(attr, RepeatCell):
            cell_list = self._cells[attr_name]
            
            if not self._lock:
                self._used_params[attr_name] = attr._get_used_params(
                    self._config[attr_name], cell_list
                )

            return cell_list

        return attr
    
    @classmethod
    def get_search_params(cls, blank: bool = False):
        valid_operations = (type(Cell), RepeatCell, DiscreteChoice)
        
        return OrderedDict([
            (k, getattr(cls, k).get_search_params(blank=blank))
            for k in sorted(vars(cls).keys())
            if isinstance(getattr(cls, k), valid_operations)
        ])
    
    def build(self, *args, **kwargs):
        self.__init__(*args, **kwargs)
        self._lock = True
        return self

    @classmethod
    def _prebuild(cls, config: Dict):
        # Creates object deferring initialization
        obj = cls.__new__(cls)
        obj._lock = False
        
        # Initializes empty access dict
        obj._used_params = cls.get_search_params(blank=True)

        # Sets config and choices atributes
        obj._config = deepcopy(config)
        obj._choices = {
            k: v for k, v in cls.__dict__.items() 
            if isinstance(v, DiscreteChoice)
        }
        
        # Prebuilds cells
        obj._cells = {
            k: v._prebuild(config[k]) 
            for k, v in cls.__dict__.items() 
            if isinstance(v, (type(Cell), RepeatCell))
        }

        return obj
    
    @classmethod
    def from_config(cls, config: Dict, *args, **kwargs):
        return cls._prebuild(config).build(*args, **kwargs)


class RepeatCell():
    def __init__(self, cell_cls: Type[Cell], repeat_times: List[int], share_arch: bool = False):
        self.cell_cls = cell_cls
        self.repeat_times = repeat_times
        self.share_arch = share_arch
        
    def _prebuild(self, config) -> List[Cell]:
        if self.share_arch:
            return [
                self.cell_cls._prebuild(config['cell'])
                for _ in range(config['repeat'])
            ]
        
        else:
            assert config['repeat'] in self.repeat_times
            assert len(config['cells']) == max(self.repeat_times)

            return [
                self.cell_cls._prebuild(config['cells'][i])
                for i in range(config['repeat'])
            ]

    def get_search_params(self, blank: bool = False) -> OrderedDict:
        repeat_choice = DiscreteChoice(self.repeat_times)
        
        if self.share_arch:
            return OrderedDict([
                ('repeat', repeat_choice.get_search_params(blank=blank)),
                ('cell', self.cell_cls.get_search_params(blank=blank)),
            ])
        
        return OrderedDict([
            ('repeat', repeat_choice.get_search_params(blank=blank)),
            ('cells', [
                self.cell_cls.get_search_params(blank=blank) 
                for _ in range(max(self.repeat_times))
            ])
        ])
    
    def from_config(self, config: Union[List[Dict], Dict],
                    cell_args: Optional[Dict] = None,
                    *args, **kwargs):
        
        cell_args = cell_args or dict()
        assert 'repeat' in config
        
        if self.share_arch:
            assert 'cell' in config and isinstance(config['cell'], dict)
            
            cell_list = [
                self.cell_cls.from_config(config['cell'], **cell_args)
                for _ in range(config['repeat'])
            ]
        
        else:
            assert 'cells' in config and isinstance(config['cells'], list)
            assert len(config['cells']) == max(self.repeat_times)
            
            cell_list = [
                self.cell_cls.from_config(config['cells'][i], **cell_args)
                for i in range(config['repeat'])
            ]

        return cell_list

    def _get_used_params(self, config: Dict, cell_list: List[Cell]):
        used_params = self.get_search_params(blank=True)
        used_params['repeat'] = True

        if self.share_arch:
            used_params['cell'] = (
                cell_list[0]._used_params if len(cell_list) > 0
                else self.cell_cls.get_search_params(blank=True)
            )
        else:
            for i, cell in enumerate(cell_list):
                used_params['cells'][i] = cell._used_params
                
        return used_params
