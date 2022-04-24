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
from archai.algos.evolution_pareto_image_seg.segmentation_search_space import SegmentationSearchSpace

class DiscreteSearchSpaceSegmentation(DiscreteSearchSpace):
    def __init__(self, datasetname:str, 
                min_mac:int=0, 
                max_mac:int=sys.maxsize,
                min_layers:int=1,
                max_layers:int=12,
                max_downsample_factor:int=32,
                skip_connections:bool=True,
                max_skip_connection_length:int=3,
                max_scale_delta:int=1):
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
        assert self.max_downsample_factor in set([2, 4, 8, 16, 32])

        self.max_skip_connection_length = max_skip_connection_length
        assert self.max_skip_connection_length > 0

        self.max_scale_delta = max_scale_delta
        assert self.max_scale_delta in set([1, 2, 3])

        self.skip_connections = skip_connections
        
        

    @overrides
    def random_sample(self)->ArchWithMetaData:
        ''' Uniform random sample an architecture within the limits of min and max MAC '''

        found_valid = False
        while not found_valid:
            
            # randomly pick number of layers    
            num_layers = random.randint(self.min_layers, self.max_layers)

            model, graph, channels_per_scale = \
                SegmentationSearchSpace.random_sample(nb_layers=num_layers,                                                         
                                                      max_downsample_factor=self.max_downsample_factor,
                                                      skip_connections=self.skip_connections,
                                                      max_skip_connection_length=self.max_skip_connection_length,             
                                                      max_scale_delta=self.max_scale_delta)

            meta_data = {
                'datasetname': self.datasetname,
                'graph': graph,
                'channels_per_scale': channels_per_scale,
                'archid': uuid.uuid4(), #TODO: need to replace with a string of the graph 
            }
            arch_meta = ArchWithMetaData(model, meta_data)

            # check if the model is within desired bounds    
            input_tensor_shape = (1, 3, model.img_size, model.img_size)
            model_stats = tw.ModelStats(model, input_tensor_shape, clone_model=True)
            if model_stats.MAdd > self.min_mac and model_stats.MAdd < self.max_mac:
                found_valid = True
            
        return arch_meta
        

    @overrides
    def get_neighbors(self, arch: ArchWithMetaData) -> List[ArchWithMetaData]:
        found_valid = False

        while not found_valid:    

            graph = copy.deepcopy(arch.metadata['graph'])
            channels_per_scale = copy.deepcopy(arch.metadata['channels_per_scale'])

            # sanity check the graph
            assert len(graph) > 1
            assert graph[-1]['name'] == 'output'
            assert graph[0]['name'] == 'input'
            
            # pick a node at random (but not input node)
            # and change its operator at random
            # and its input sources
            # WARNING: this can result in some nodes left hanging
            chosen_node_idx = random.randint(1, len(graph)-1)
            node = graph[chosen_node_idx]
            node['op'] = random.choice(self.operations)
            # choose up to k inputs from previous nodes
            max_inputs = 3 # TODO: make config 
            k = min(chosen_node_idx, random.randint(1, max_inputs))
            input_idxs = random.sample(range(chosen_node_idx), k)
            node['inputs'] = [graph[idx]['name'] for idx in input_idxs]

            # now go through every node in the graph (except output node)
            # and make sure it is being used as input in some node after it
            for i, node in enumerate(graph[:-1]):
                this_name = node['name']
                orphan = True
                # test whether not orphan
                for r in range(i+1, len(graph)):
                    if graph[r]['name'] == this_name:
                        orphan = False
                if orphan:
                    # choose a forward node to connect it with
                    chosen_forward_idx = random.randint(i+1, len(graph)-1)
                    graph[chosen_forward_idx]['inputs'].append(this_name)

            # compile the model
            model = SegmentationNasModel.from_config(graph, channels_per_scale)
            # TODO: these should come from config or elsewhere 
            # such that they are not hardcoded in here
            out_shape = model.validate_forward(torch.randn(1, 3, model.img_size, model.img_size)).shape
            assert out_shape == torch.Size([1, 19, model.img_size, model.img_size])
            extradata = {
                            'datasetname': self.datasetname,
                            'graph': graph,
                            'channels_per_scale': channels_per_scale,
                            'archid': uuid.uuid4(), #TODO: need to replace with a string of the graph 
                        }

            arch_meta = ArchWithMetaData(model, extradata)
            
            # check if the model is within desired bounds    
            input_tensor_shape = (1, 3, model.img_size, model.img_size)
            model_stats = tw.ModelStats(model, input_tensor_shape, clone_model=True)
            if model_stats.MAdd > self.min_mac and model_stats.MAdd < self.max_mac:
                found_valid = True

        return [arch_meta]



        

