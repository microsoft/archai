import sys
import math
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from collections import defaultdict

from torch_geometric.data import Data as GraphData


def get_graph_ngrams(graph: GraphData, n: int = 3,
                     node_features: Optional[List[int]] = None,
                     output_node_only: bool = False) -> List[Tuple[Tuple]]:
    """Lists all node n-grams from a torch_geometric graph

    Args:
        graph (torch_geometric.data.Data): Torch geometric graph
        node_features (List[int]): List of node attributes that should be considered. 
    If None, all node attributes are considered.
        n (int, optional): n-gram length.
        output_node_only (bool, optional): If all of the listed node n-grams should end in 
    the output node. Defaults to False.

    Returns:
        List[Tuple[Tuple]]: List of node features n-grams (tuples of tuples) 
    """
    # Converts the edge list to a node dict
    edges = graph.edge_index.T.numpy().tolist()
    node_features = graph.x.numpy()[:, node_features] if node_features else graph.x.numpy()

    graph_dict = {
        node: {
            'inputs': [],
            'features': node_features[node]
        } for node in range(graph.num_nodes)
    }

    for in_node, out_node in edges:
        graph_dict[out_node]['inputs'].append(in_node)

    @lru_cache(maxsize=20_000)
    def _ngrams_ending_in(node_id: int, n: int):
        node = graph_dict[node_id]
        features = [tuple(node['features'].tolist())]

        if n == 1 or (output_node_only and node_id == 0):
            return [features]

        if node['inputs'] is None and not output_node_only:
            return [None]

        return [
            path + features
            for p_node in node['inputs']
            for path in _ngrams_ending_in(p_node, n-1)
            if path
        ]
    
    if output_node_only:
        return [tuple(p) for p in _ngrams_ending_in(len(node_features) - 1, n)]

    return [
        tuple(path)
        for terminal_node in graph_dict
        for path in _ngrams_ending_in(terminal_node, n)
        if path
    ]


def get_graph_paths(graph: GraphData, node_features: Optional[List[int]] = None) -> List[Tuple[Tuple]]:
    """Lists all paths from a architecture graph. 

    Args:
        graph (torch_geometric.data.Data): Torch geometric graph
        node_features (List[int]): List of node attributes that should be considered. 
    If None, all node attributes are considered.
        the output node. Defaults to False.

    Returns:
        List[Tuple[Tuple]]: List of node features n-grams (tuples of tuples) 
    """
    return get_graph_ngrams(
        graph, n=sys.maxsize, node_features=node_features,
        output_node_only=True
    )


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
