# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict
from typing import Any, Callable, Dict, Union

from archai.discrete_search.search_spaces.config.discrete_choice import DiscreteChoice


def flatten_dict(odict: Dict[str, Any]) -> OrderedDict:
    """Flatten a nested dictionary into a single level dictionary.

    Args:
        odict: Nested dictionary.

    Returns:
        Flattened dictionary.

    """

    fdict = OrderedDict()

    def _flatten(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
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


def replace_ptree_choices(
    config_tree: Union[Dict, DiscreteChoice], repl_fn: Callable[[DiscreteChoice], Any]
) -> OrderedDict:
    """Replace all DiscreteChoice nodes in a tree with the output of a function.

    Args:
        config_tree: Tree with DiscreteChoice nodes.
        repl_fn: Function to replace DiscreteChoice nodes.

    Returns:
        Replaced tree.

    """

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
) -> OrderedDict:
    """Replace all DiscreteChoice nodes in a tree with the output of a function and an auxilary tree.

    Args:
        query_tree: Tree with DiscreteChoice nodes.
        aux_tree: Auxiliary tree with DiscreteChoice nodes.
        repl_fn: Function that takes a `query_node` and an `aux_node` and returns a replacement for `query_node`.

    Returns:
        Replaced tree.

    """

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
