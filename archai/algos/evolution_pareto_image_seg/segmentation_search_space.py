import random
from copy import deepcopy
from abc import abstractmethod
from typing import List, Dict, Optional, Set
from overrides import overrides

import torch.nn as nn
from .model import SegmentationNasModel, OPS

class SegmentationSearchSpace():

    @abstractmethod
    def random_sample(base_channels_list: List[int] = [8, 12, 24, 32, 36, 48],
                      delta_channels_list: List[int] = [8, 12, 24, 32, 36, 48],
                      post_upsample_layer_list: List[int] = [1, 2, 3],
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

        # Samples `base_channels` and `delta_channels`
        base_channels = random.choice(base_channels_list)
        delta_channels = random.choice(delta_channels_list)

        # Samples `post_upsample_layers`
        post_upsample_layers = random.choice(post_upsample_layer_list)

        # Builds channels per level map using the sampled `base_channels` and `delta_channels`
        ch_map = {
            scale: base_channels + i*delta_channels
            for i, scale in enumerate([1, 2, 4, 8, 16])
        }

        # Input node
        graph = [{'name': 'input', 'inputs': None, 'op': random.choice(operations), 'scale': 1}]
        node_list = ['input']

        # Used to control `max_scale_delta`
        idx2scale = list(ch_map.keys())
        scale2idx = {scale: i for i, scale in enumerate(ch_map.keys())}

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
                    min(max_scale_delta, len(ch_map) - last_scale_idx - 1)
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

        channels_per_scale = {'base_channels': base_channels, 'delta_channels': delta_channels}
        return (
            SegmentationNasModel.from_config(
                graph, channels_per_scale, post_upsample_layers=post_upsample_layers
            ), graph, channels_per_scale
        )
