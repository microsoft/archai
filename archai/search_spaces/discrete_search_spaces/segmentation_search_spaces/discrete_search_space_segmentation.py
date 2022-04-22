import random
from typing import List
from overrides.overrides import overrides
import copy
import uuid

import torch

from archai.nas.arch_meta import ArchWithMetaData
from archai.nas.discrete_search_space import DiscreteSearchSpace

from archai.algos.evolution_pareto_image_seg.model import OPS, SegmentationNasModel
from archai.algos.evolution_pareto_image_seg.segmentation_search_space import SegmentationSearchSpace

class DiscreteSearchSpaceSegmentation(DiscreteSearchSpace):
    def __init__(self, datasetname:str):
        super().__init__()
        self.datasetname = datasetname
        self.operations = list(OPS.keys())
        

    @overrides
    def random_sample(self)->ArchWithMetaData:
        ''' Uniform random sample an architecture '''
        # TODO: put choices in config so that we can 
        # randomize over the input choices as well
        model, graph, channels_per_scale = SegmentationSearchSpace.random_sample(nb_layers=12, 
                                                    max_downsample_factor=16, 
                                                    max_scale_delta=1)

        meta_data = {
            'datasetname': self.datasetname,
            'graph': graph,
            'channels_per_scale': channels_per_scale,
            'archid': uuid.uuid4(), #TODO: need to replace with a string of the graph 
        }
        arch_meta = ArchWithMetaData(model, meta_data)
        return arch_meta
        

    @overrides
    def get_neighbors(self, arch: ArchWithMetaData) -> List[ArchWithMetaData]:
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
        out_shape = model.validate_forward(torch.randn(1, 3, 256, 256)).shape
        assert out_shape == torch.Size([1, 19, 256, 256])
        extradata = {
                        'datasetname': self.datasetname,
                        'graph': graph,
                        'channels_per_scale': channels_per_scale,
                        'archid': uuid.uuid4(), #TODO: need to replace with a string of the graph 
                    }

        arch_meta = ArchWithMetaData(model, extradata)
        return [arch_meta]



        

