import random
from abc import abstractmethod
from typing import List, Dict, Optional, Set
from overrides import overrides

import torch.nn as nn
from .model import SegmentationNasModel, OPS

DEFAULT_CH_PER_SCALE = {
    1: 24,
    2: 60,
    4: 96,
    8: 132,
    16: 168
}

class SegmentationSearchSpace():

    @abstractmethod
    def random_sample(channels_per_scale: Optional[Dict] = None,
                      nb_layers: int = 24,
                      max_downsample_factor: int = 16,
                      skip_connections: bool = True,
                      max_skip_connection_length: int = 3,
                      operation_subset: Optional[Set] = None,
                      max_scale_delta: Optional[int] = None):
        '''Uniform random sample an architecture (nn.Module)'''
        operations = list(OPS.keys())

        if operation_subset:
            operations = [op_name for op_name in operations if op_name in operation_subset]

        if not channels_per_scale:
            channels_per_scale = DEFAULT_CH_PER_SCALE

        # Input node
        graph = [{'name': 'input', 'inputs': None, 'op': random.choice(operations), 'scale': 1}]
        node_list = ['input']

        # Used to control `max_scale_delta`
        idx2scale = list(channels_per_scale.keys())
        scale2idx = {scale: i for i, scale in enumerate(channels_per_scale.keys())}

        for layer in range(nb_layers):
            is_output = (layer == nb_layers - 1)
            last_layer = graph[-1]

            new_node = {
                'name': 'output' if is_output else f'layer_{layer}',
                'op': None if is_output else random.choice(operations),
                'inputs': [last_layer['name']]
            }

            # Choose scale
            if max_scale_delta:
                last_scale_idx = scale2idx[last_layer['scale']]

                # Samples a delta value for the current scale index
                scale_delta = random.randint(
                    max(-max_scale_delta, -last_scale_idx),
                    min(max_scale_delta, len(channels_per_scale) - last_scale_idx - 1)
                )

                # Assigns the new scale to the new node
                new_node['scale'] = idx2scale[last_scale_idx + scale_delta]
            else:
                new_node['scale'] = random.choice(idx2scale)

            # Choose inputs
            if len(node_list) > 1:
                for i in range(2, 1 + random.randint(2, min(len(node_list), max_skip_connection_length))):
                    if skip_connections and random.random() < 0.5:
                        new_node['inputs'].append(node_list[-i])

            # Adds node
            graph.append(new_node)
            node_list.append(new_node['name'])

        return SegmentationNasModel.from_config(graph, channels_per_scale)

    @abstractmethod
    def get_neighbors():
        pass
