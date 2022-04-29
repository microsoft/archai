import random
from typing import List
from overrides.overrides import overrides
import copy
import uuid
import sys

import torch

import tensorwatch as tw

from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.discrete_search_space import DiscreteSearchSpace

from archai.algos.evolution_pareto_image_seg.model import OPS, SegmentationNasModel


def random_neighbor(param_values: List[int], current_value: int):
    param_values = sorted(copy.deepcopy(param_values))
    param2idx = {param: idx for idx, param in enumerate(param_values)}

    current_idx = param2idx[current_value]
    offset = random.randint(
        a=-1 if current_idx > 0 else 0,
        b=1 if current_idx < len(param_values) - 1 else 0
    )
    
    return param_values[current_idx + offset]

class DiscreteSearchSpaceSegmentation(DiscreteSearchSpace):
    def __init__(self, datasetname:str, 
                 min_mac:int=0, 
                 max_mac:int=sys.maxsize,
                 min_layers:int=1,
                 max_layers:int=12,
                 max_downsample_factor:int=16,
                 skip_connections:bool=True,
                 max_skip_connection_length:int=3,
                 max_scale_delta:int=1,
                 max_post_upsample_layers:int=3,
                 min_base_channels:int=8,
                 max_base_channels:int=48,
                 base_channels_binwidth:int=8,
                 min_delta_channels:int=8,
                 max_delta_channels:int=48,
                 delta_channels_binwidth:int=8):
        super().__init__()
        self.datasetname = datasetname
        assert self.datasetname != ''

        self.operations = list(OPS.keys())
        assert len(self.operations) > 0

        self.min_mac = min_mac
        self.max_mac = max_mac
        assert self.min_mac <= self.max_mac

        self.min_layers = min_layers
        self.max_layers = max_layers
        assert self.min_layers <= self.max_layers

        self.max_downsample_factor = max_downsample_factor
        assert self.max_downsample_factor in set([2, 4, 8, 16])

        self.max_skip_connection_length = max_skip_connection_length
        assert self.max_skip_connection_length > 0

        self.max_scale_delta = max_scale_delta
        assert self.max_scale_delta in set([1, 2, 3])

        self.post_upsample_layers_list = list(range(1, max_post_upsample_layers + 1))
        assert len(self.post_upsample_layers_list) < 5

        self.base_channels_list = list(range(min_base_channels, max_base_channels + 1, base_channels_binwidth))
        assert min_base_channels < max_base_channels
        assert len(self.base_channels_list) > 1
        
        self.delta_channels_list = list(range(min_delta_channels, max_delta_channels + 1, delta_channels_binwidth))
        assert min_delta_channels < max_delta_channels
        assert len(self.delta_channels_list) > 1

        self.skip_connections = skip_connections
        

    @overrides
    def random_sample(self)->ArchWithMetaData:
        ''' Uniform random sample an architecture within the limits of min and max MAC '''

        found_valid = False
        while not found_valid:
            
            # randomly pick number of layers    
            num_layers = random.randint(self.min_layers, self.max_layers)

            model = SegmentationNasModel.sample_model(
                base_channels_list=self.base_channels_list,
                delta_channels_list=self.delta_channels_list,
                post_upsample_layer_list=self.post_upsample_layers_list,
                nb_layers=num_layers,                                                         
                max_downsample_factor=self.max_downsample_factor,
                skip_connections=self.skip_connections,
                max_skip_connection_length=self.max_skip_connection_length,             
                max_scale_delta=self.max_scale_delta
            )

            meta_data = {
                'datasetname': self.datasetname,
                'archid': model.to_hash(),
                'parent': None
            }
            arch_meta = ArchWithMetaData(model, meta_data)

            # check if the model is within desired bounds    
            input_tensor_shape = (1, 3, model.img_size, model.img_size)
            model_stats = tw.ModelStats(model, input_tensor_shape, clone_model=True)
            if model_stats.MAdd > self.min_mac and model_stats.MAdd < self.max_mac:
                found_valid = True
            
        return arch_meta
        

    @overrides
    def get_neighbors(self, base_model: ArchWithMetaData) -> List[ArchWithMetaData]:
        parent_id = base_model.metadata['archid']
        found_valid = False

        while not found_valid:    
            graph = copy.deepcopy(list(base_model.arch.graph.values()))
            channels_per_scale = copy.deepcopy(base_model.arch.channels_per_scale)

            # sanity check the graph
            assert len(graph) > 1
            assert graph[-1]['name'] == 'output'
            assert graph[0]['name'] == 'input'

            # `base_channels` and `delta_channels` mutation
            channels_per_scale = {
                'base_channels': random_neighbor(self.base_channels_list, channels_per_scale['base_channels']),
                'delta_channels': random_neighbor(self.delta_channels_list, channels_per_scale['delta_channels']),
            }

            # `post_upsample_layers` mutation
            post_upsample_layers = random_neighbor(
                self.post_upsample_layers_list,
                base_model.arch.post_upsample_layers
            )

            # pick a node at random (but not input node)
            # and change its operator at random
            # and its input sources
            chosen_node_idx = random.randint(1, len(graph) - 1)
            node = graph[chosen_node_idx]
            
            if node['name'] != 'output':
                node['op'] = random.choice(self.operations)
            
            # choose up to k inputs from previous nodes
            max_inputs = 3 # TODO: make config 

            if node['name'] != 'input':
                k = min(chosen_node_idx, random.randint(1, max_inputs))
                input_idxs = random.sample(range(chosen_node_idx), k)

                node['inputs'] = [graph[chosen_node_idx - 1]['name']]
                node['inputs'] += [
                    graph[idx]['name'] for idx in input_idxs
                    if graph[idx]['name'] != graph[chosen_node_idx-1]['name']
                ]

            # compile the model
            nbr_model = SegmentationNasModel(graph, channels_per_scale, post_upsample_layers)
            out_shape = nbr_model.validate_forward(
                torch.randn(1, 3, nbr_model.img_size, nbr_model.img_size)
            ).shape

            assert out_shape == torch.Size([1, 19, nbr_model.img_size, nbr_model.img_size])

            arch_meta = ArchWithMetaData(nbr_model, {
                'datasetname': self.datasetname,
                'archid': nbr_model.to_hash(),
                'parent': parent_id
            })
            
            # check if the model is within desired bounds    
            input_tensor_shape = (1, 3, nbr_model.img_size, nbr_model.img_size)
            model_stats = tw.ModelStats(nbr_model, input_tensor_shape, clone_model=True)
            if model_stats.MAdd > self.min_mac and model_stats.MAdd < self.max_mac:
                found_valid = True

        return [arch_meta]


    def load_from_file(self, config_file: str) -> ArchWithMetaData:
        model = SegmentationNasModel.from_file(config_file)
        
        return ArchWithMetaData(model, {
            'datasetname': self.datasetname,
            'archid': model.to_hash(),
            'parent': None
        })

