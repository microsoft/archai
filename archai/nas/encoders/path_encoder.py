from copy import deepcopy
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data as GraphData

from archai.search_spaces.discrete_search_spaces.utils import get_graph_ngrams, get_graph_paths
from archai.nas.discrete_search_space import EncodableDiscreteSearchSpace


class PathEncoder():
    def __init__(self, node_features: Optional[List[int]] = None, path_length: int = -1):
        self.vocab = None
        self.path_length = path_length
        self.node_features = node_features

    def encode(self, g: GraphData) -> List[Tuple[Tuple]]:
        if self.path_length == -1:
            return get_graph_paths(g, node_features=self.node_features)

        return get_graph_ngrams(g, node_features=self.node_features, n=self.path_length)

    def fit(self, arch_list: List[GraphData]) -> None:
        paths = {
            p for m in arch_list for p in self.encode(m)
        }

        self.vocab = list(paths)
        self.path2idx = {path: i for i, path in enumerate(self.vocab)}
    
    def transform(self, arch_list: List[GraphData]) -> torch.Tensor:
        if not self.vocab:
            raise ValueError('PathEncoder is not fitted.')

        arr = np.zeros((len(arch_list), len(self.vocab)))

        for model_idx, m in enumerate(arch_list):
            idxs = np.array([
                self.path2idx[p] for p in self.encode(m)
                if p in self.path2idx
            ])
            if idxs.any():
                arr[model_idx][idxs] = 1

        return arr

    def fit_transform(self, x: List[GraphData]) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)
