from typing import Callable, Type
from copy import deepcopy
from collections import OrderedDict


def replace_param_tree_nodes(param_tree: OrderedDict, repl_fn: Callable) -> OrderedDict:
    def _search(d):
        if isinstance(d, OrderedDict):
            for k, v in d.items():
                d[k] = _search(v)
        elif isinstance(d, list):
            for i, v in enumerate(d):
                d[i] = _search(v)
        else:
            return repl_fn(d)
                
        return d
    
    return _search(deepcopy(param_tree))


def replace_param_tree_pair(param_tree_1: OrderedDict,
                            param_tree_2: OrderedDict,
                            repl_fn: Callable) -> OrderedDict:
    def _search(d1, d2):
        if isinstance(d1, OrderedDict):
            for k, v in d1.items():
                d1[k] = _search(v, d2[k])
        elif isinstance(d1, list):
            for i, v in enumerate(d1):
                d1[i] = _search(v, d2[i])
        else:
            return repl_fn(d1, d2)

        return d1
    
    return _search(
        deepcopy(param_tree_1),
        deepcopy(param_tree_2)
    )


def flatten_ordered_dict(d):
    fdict = OrderedDict()
    
    def _flatten(prefix, d):
        prefix = prefix + '.' if prefix else prefix
        
        if isinstance(d, OrderedDict):
            for k, v in d.items():
                flat_v = _flatten(prefix + k, v)
                
                if flat_v is not None:
                    fdict[prefix + k] = flat_v
        
        elif isinstance(d, list):
            for i, v in enumerate(d):
                flat_v = _flatten(prefix + str(i), v)
                
                if flat_v is not None:
                    fdict[prefix + str(i)] = flat_v
        else:
            return d
    
    _flatten('', d)
    return fdict
