import numpy as np
import torch
from copy import deepcopy
from typing import List, Tuple, Dict
from tqdm import tqdm
import plotly.express as px

from archai.algos.evolution_pareto_image_seg.remote_benchmark import RemoteAzureBenchmark
from archai.algos.evolution_pareto_image_seg.model import SegmentationNasModel
from archai.algos.evolution_pareto_image_seg.segmentation_trainer import LightningModelWrapper
from archai.search_spaces.discrete_search_spaces.segmentation_search_spaces.discrete_search_space_segmentation import DiscreteSearchSpaceSegmentation
from archai.search_spaces.discrete_search_spaces.segmentation_search_spaces.discrete_search_space_segmentation import ArchWithMetaData
from archai.search_spaces.discrete_search_spaces.utils import get_graph_paths, get_graph_ngrams


def _get_graph_with_channels(m: ArchWithMetaData):
    g = deepcopy(m.arch.graph)
    ch_per_scale = m.arch.channels_per_scale
    
    # TODO: should the scale info be embedded
    # in ArchWithMetaData to avoid mismatch?
    scales = [1, 2, 4, 8, 16, 32]

    ch_per_scale = {
        scale: ch_per_scale['base_channels'] +\
               (scale_i)*ch_per_scale['delta_channels']
        for scale_i, scale in enumerate(scales)
    }
    
    for node_attrs in g.values():
        node_attrs['channels'] = ch_per_scale[node_attrs['scale']]
    
    return g


class PathEncoder():
    def __init__(self, node_vars: List, path_length: int = -1):
        self.vocab = None
        self.node_vars = node_vars
        self.path_length = path_length
        self.should_calc_channels = 'channels' in node_vars
            
    
    def fit(self, model_list: List[ArchWithMetaData]) -> None:
        paths = {
            p for m in model_list for p in self.encode(m)
        }
        
        self.vocab = list(paths)
        self.path2idx = {path: i for i, path in enumerate(self.vocab)}
    
    def transform(self, model_list: List[ArchWithMetaData]) -> torch.Tensor:
        if not self.vocab:
            raise ValueException('PathEncoder is not fitted.')
        
        arr = np.zeros((len(model_list), len(self.vocab)))
        
        for model_idx, m in enumerate(model_list):
            idxs = np.array([
                self.path2idx[p] for p in self.encode(m)
                if p in self.path2idx
            ])
            if idxs.any():
                arr[model_idx][idxs] = 1
            
        return arr

    def fit_transform(self, x: List[ArchWithMetaData]) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)
    
    def encode(self, m: ArchWithMetaData) -> List[Tuple[Tuple]]:
        g = _get_graph_with_channels(m) if self.should_calc_channels else m.arch.graph
            
        if self.path_length == -1:
            return get_graph_paths(g, node_vars=self.node_vars)
        
        return get_graph_ngrams(g, node_vars=self.node_vars, n=self.path_length)


def main():

    random_model = DiscreteSearchSpaceSegmentation('f', min_layers=3, max_layers=16)

    model_list = [random_model.random_sample() for _ in tqdm(range(10))]

    encoder1 = PathEncoder(['scale'], path_length=-1)
    encoder1.fit(model_list)

    encoder2 = PathEncoder(['op'], path_length=-1)
    encoder2.fit(model_list)

    X = np.concatenate([encoder1.transform(model_list), encoder2.transform(model_list)], axis=1)

    sim_matrix = (X @ X.T)/np.diag(X@X.T)
    fig = px.imshow(sim_matrix)
    fig.show()

    print('dummy')



if __name__ == '__main__':
    main()