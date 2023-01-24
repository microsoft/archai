# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Type, Union

from archai.discrete_search.search_spaces.config.discrete_choice import DiscreteChoice


def flatten_dict(odict: Dict) -> OrderedDict:
    fdict = OrderedDict()

    def _flatten(prefix, d):
        prefix = prefix + "." if prefix else prefix

        if isinstance(d, OrderedDict):
            for k, v in d.items():
                flat_v = _flatten(prefix + k, v)

                if flat_v is not None:
                    fdict[prefix + k] = flat_v
        else:
            return d

    _flatten("", odict)
    return fdict


def replace_ptree_choices(config_tree: Union[Dict, DiscreteChoice], repl_fn: Callable[[DiscreteChoice], Any]):
    def _replace_tree_nodes(node, repl_fn, ref_map):
        if isinstance(node, dict):
            output_tree = OrderedDict()

            for param_name, param in node.items():
                output_tree[param_name] = _replace_tree_nodes(param, repl_fn, ref_map)

        elif isinstance(node, DiscreteChoice):
            if id(node) not in ref_map:
                ref_map[id(node)] = repl_fn(node)
            return ref_map[id(node)]

        else:
            return node

        return output_tree

    return _replace_tree_nodes(config_tree, repl_fn, {})


def replace_ptree_pair_choices(
    query_tree: Union[Dict, DiscreteChoice], aux_tree: Union[Dict, Any], repl_fn: Callable[[DiscreteChoice, Any], Any]
):
    def _replace_tree_nodes(query_node, aux_node, repl_fn, ref_map):
        if isinstance(query_node, dict):
            output_tree = OrderedDict()

            for param_name, param in query_node.items():
                assert param_name in aux_node, "`aux_tree` must be identical to `query_tree` apart from terminal nodes"

                output_tree[param_name] = _replace_tree_nodes(param, aux_node[param_name], repl_fn, ref_map)

        elif isinstance(query_node, DiscreteChoice):
            if id(query_node) not in ref_map:
                ref_map[id(query_node)] = repl_fn(query_node, aux_node)

            return ref_map[id(query_node)]

        else:
            return query_node

        return output_tree

    return _replace_tree_nodes(query_tree, aux_tree, repl_fn, {})
