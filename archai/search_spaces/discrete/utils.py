import sys
import math
from typing import List, Dict, Tuple
from functools import lru_cache


def get_graph_ngrams(graph: Dict[str, Dict], node_vars: List[str],
                     n: int = 3, output_node_only: bool = False) -> List[Tuple[Tuple]]:
    """Lists all node n-grams from a graph with node attributes

    Args:
        graph (Dict[str, Dict]): Node name to node attributes mapping.
        node_vars (List[str]): List of node attributes that should be considered
        n (int, optional): n-gram length. Defaults to 5.
        output_node_only (bool, optional): If all of the node n-grams should end in 
        the output node. Defaults to False.

    Returns:
        List[List[Tuple]]: List of node n-grams (tuples of tuples) 
    """    

    @lru_cache(maxsize=20_000)
    def _ngrams_ending_in(node_name: str, n: int):
        node = graph[node_name]
        node_info = [tuple(node[var] for var in node_vars)]

        if n == 1 or (output_node_only and node_name == 'input'):
            return [node_info]

        if node_name == 'input' and not output_node_only:
            return [None]

        return [
            path + node_info
            for p_node in node['inputs']
            for path in _ngrams_ending_in(p_node, n-1)
            if path
        ]
    
    if output_node_only:
        return [tuple(p) for p in _ngrams_ending_in('output', n)]

    return [
        tuple(path) 
        for terminal_node in graph
        for path in _ngrams_ending_in(terminal_node, n)
        if path
    ]

def get_graph_paths(graph: Dict[str, Dict], node_vars: List[str]) -> List[Tuple[Tuple]]:
    """Lists all paths from a architecture graph. 

    Args:
        graph (Dict[str, Dict]): Node name to node attributes mapping.
        node_vars (List[str]): List of node attributes that should be considered
        the output node. Defaults to False.

    Returns:
        List[List[Tuple]]: List of paths (tuples of tuples) 
    """
    return get_graph_ngrams(graph, node_vars, n=sys.maxsize, output_node_only=True)


def graph_ngram_cossim(graph1: Dict, graph2: Dict, node_vars: List[str], 
                       n: int, output_node_only: bool = False):
    x, y = [get_graph_ngrams(g, node_vars, n, output_node_only) for g in [graph1, graph2]]
    x, y = set(x), set(y)
    norm_x, norm_y = math.sqrt(len(x)), math.sqrt(len(y))
    
    return (
        len(x.intersection(y)) / (norm_x * norm_y)
    )


def graph_path_cossim(graph1: Dict, graph2: Dict, node_vars: List[str]):
    return graph_ngram_cossim(graph1, graph2, node_vars, sys.maxsize, True)


def graph_ngram_jaccard(graph1: Dict, graph2: Dict, node_vars: List[str], 
                       n: int, output_node_only: bool = False):
    x, y = [get_graph_ngrams(g, node_vars, n, output_node_only) for g in [graph1, graph2]]
    x, y = set(x), set(y)
    
    return (
        len(x.intersection(y)) / (len(x) + len(y) - len(x.intersection(y)))
    )


def graph_path_jaccard(graph1: Dict, graph2: Dict, node_vars: List[str]):
    return graph_ngram_jaccard(graph1, graph2, node_vars, sys.maxsize, True)
