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
        # gets the prebuilt cell from `self._cells``
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
    def get_search_params(cls, blank: bool = False) -> OrderedDict:
        """Gets all architecture search parameters.

        Args:
            blank (bool, optional): If True, replaces DiscreteChoice
                objects with `False`. Defaults to False.

        Returns:
            OrderedDict: Search parameter tree
        """        
        valid_operations = (type(Cell), RepeatCell, DiscreteChoice)
        
        return OrderedDict([
            (k, getattr(cls, k).get_search_params(blank=blank))
            for k in sorted(vars(cls).keys())
            if isinstance(getattr(cls, k), valid_operations)
        ])
    
    def build(self, *args, **kwargs):
        """Builds a prebuilt object created using `Cell._prebuild`

        Returns:
            Cell: Cell instance
        """
        self.__init__(*args, **kwargs)
        self._lock = True
        return self

    @classmethod
    def _prebuild(cls, config: Dict):
        # Creates object deferring initialization
        obj = cls.__new__(cls)
        obj._lock = False
        
        # Initializes empty access dict to track
        # which params were used by the user __init__
        obj._used_params = cls.get_search_params(blank=True)

        # Sets config and choices atributes
        obj._config = deepcopy(config)
        obj._choices = {
            k: v for k, v in cls.__dict__.items() 
            if isinstance(v, DiscreteChoice)
        }
        
        # Prebuilds child cells
        obj._cells = {
            k: v._prebuild(config[k]) 
            for k, v in cls.__dict__.items() 
            if isinstance(v, (type(Cell), RepeatCell))
        }

        return obj
    
    @classmethod
    def from_config(cls, arch_config: Dict, *args, **kwargs) -> 'Cell':
        """Initializes the Cell using the architecture parameters
        from `config` and the positional args and kwargs for the
        class `__init__` method.

        Args:
            arch_config (Dict): Architecture parameters

        Returns:
            Cell: Initialized Cell object
        """        
        return cls._prebuild(arch_config).build(*args, **kwargs)


class RepeatCell():
    def __init__(self, cell_cls: Type[Cell], repeat_times: List[int], share_arch: bool = False):
        """Repeats a cell a variable number of times.

        Args:
            cell_cls (Type[Cell]): Cell class
            
            repeat_times (List[int]): List of possible values for the number of times
                the cell should be repeated. For instance [1, 2, 5], means
                the Cell will be repeated 1, 2 or 5 times.
            
            share_arch (bool, optional): If repetions of the same Cell should all
                share the same architecture (not model weights). Defaults to False.
        """
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
