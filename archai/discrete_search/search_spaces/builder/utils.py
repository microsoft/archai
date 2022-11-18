from typing import Dict, Callable, Type
from copy import deepcopy


def search_and_replace_node(param_tree: Dict, query: Type,
                            repl_fn: Callable) -> Dict:    
    def _search(d):
        if isinstance(d, dict):
            for k, v in d.items():
                d[k] = _search(v)
        
        elif isinstance(d, list):
            for i, v in enumerate(d):
                d[i] = _search(v)
        
        elif isinstance(d, query):
            return repl_fn(d)
        
        return d

    return _search(deepcopy(param_tree))


def replace_param_tree_pair(param_tree_1: dict, param_tree_2: dict, repl_fn: Callable) -> dict:
    def _search(d1, d2):
        if isinstance(d1, dict):
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
