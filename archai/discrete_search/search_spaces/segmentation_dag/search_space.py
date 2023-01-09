from random import Random
from typing import List, Optional, Dict, Tuple
from overrides.overrides import overrides
import copy
import sys

import torch
import numpy as np

import tensorwatch as tw

from archai.common.common import logger
from archai.discrete_search import ArchaiModel, EvolutionarySearchSpace
from archai.discrete_search.search_spaces.segmentation_dag.model import SegmentationDagModel, OPS


class SegmentationDagSearchSpace(EvolutionarySearchSpace):
    def __init__(self, 
                 nb_classes: int,
                 img_size: Tuple[int, int],
                 min_mac: int = 0, 
                 max_mac: int = sys.maxsize,
                 min_layers: int = 1,
                 max_layers: int = 12,
                 max_downsample_factor: int = 16,
                 skip_connections: bool = True,
                 max_skip_connection_length: int = 3,
                 max_scale_delta: int = 1,
                 max_post_upsample_layers: int = 3,
                 min_base_channels: int = 8,
                 max_base_channels: int = 48,
                 base_channels_binwidth: int = 8,
                 min_delta_channels: int = 8,
                 max_delta_channels: int = 48,
                 delta_channels_binwidth: int = 8,
                 downsample_prob_ratio: float = 1.5,
                 op_subset: Optional[str] = None,
                 mult_delta: bool = False,
                 seed: int = 1):
        super().__init__()

        self.nb_classes = nb_classes
        self.operations = list(OPS.keys())
        op_list = op_subset.split(',') if op_subset else []

        if op_list:
            self.operations = [op for op in self.operations if op in op_list]

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
        assert min_base_channels <= max_base_channels
        assert len(self.base_channels_list) > 1
        
        self.delta_channels_list = list(range(min_delta_channels, max_delta_channels + 1, delta_channels_binwidth))
        self.mult_delta = mult_delta
        assert min_delta_channels <= max_delta_channels
        assert len(self.delta_channels_list) >= 1

        self.skip_connections = skip_connections
        self.downsample_prob_ratio = downsample_prob_ratio
        self.img_size = img_size

        self.rng = Random(seed)

    def is_valid_model(self, model: torch.nn.Module) -> Tuple[bool, int]:
        ''' Utility method that checks if a model is valid and falls inside of the specified MAdds range. ''' 
        is_valid = True

        try:
            model.validate_forward(torch.randn(1, 3, *self.img_size[::-1]))
        except Exception:
            is_valid = False

        if is_valid:
            input_tensor_shape = (1, 3, *self.img_size)
            model_stats = tw.ModelStats(model, input_tensor_shape, clone_model=True)
            is_valid = (model_stats.MAdd >= self.min_mac and model_stats.MAdd <= self.max_mac)

        return is_valid, None if not is_valid else model_stats.MAdd

    def load_from_graph(self, graph: List[Dict], channels_per_scale: Dict,
                        post_upsample_layers: int = 1) -> ArchaiModel:
        ''' Utility method to create a SegmentationDagModel from a DAG ''' 
        model = SegmentationDagModel(
            graph, channels_per_scale, post_upsample_layers,
            img_size=self.img_size, nb_classes=self.nb_classes
        )
        
        return ArchaiModel(
            arch=model, archid=model.to_hash(),
            metadata={'parent': None}
        )

    def random_neighbor(self, param_values: List[int], current_value: int) -> int:
        ''' Utility method to sample a random neighbor from an element of a list'''
        param_values = sorted(copy.deepcopy(param_values))
        param2idx = {param: idx for idx, param in enumerate(param_values)}

        # Gets the index of the closest value to the current value
        if current_value in param2idx:
            current_idx = param2idx[current_value]
        else:
            current_idx = param2idx[min(param2idx, key=lambda k: abs(k - current_value))]

        offset = self.rng.randint(
            a=-1 if current_idx > 0 else 0,
            b=1 if current_idx < len(param_values) - 1 else 0
        )
        
        return param_values[current_idx + offset]

    def rename_dag_node_list(self, node_list: List[Dict], prefix: str = '',
                             rename_input_output: bool = True,
                             add_input_output: bool = False) -> List[Dict]:
        ''' Utility method to rename a list of nodes from a dag ''' 
        node_list = copy.deepcopy(node_list)
        prefix = prefix + '_' if prefix else ''

        rename_map = {}
        if not rename_input_output:
            rename_map = {'input': 'input', 'output': 'output'}

        for i, node in enumerate(node_list):
            if node['name'] not in rename_map:

                if add_input_output:
                    new_name = (
                        'input' if i == 0 else 
                        'output' if i == len(node_list) - 1 else
                        prefix + f'layer_{i}'
                    )
                else:
                    new_name = prefix + f'layer_{i}'

                rename_map[node['name']] = new_name
                node['name'] = new_name
            
            if node['inputs']:
                node['inputs'] = [
                    rename_map[inp_name] 
                    for inp_name in node['inputs']
                    if inp_name and inp_name in rename_map
                ]

        return node_list

    @overrides
    def save_arch(self, model: ArchaiModel, path: str) -> None:
        model.arch.to_file(path)

    @overrides
    def save_model_weights(self, model: ArchaiModel, path: str) -> None:
        torch.save(model.arch.to('cpu').state_dict(), path)

    @overrides
    def load_arch(self, path: str) -> ArchaiModel:
        model = SegmentationDagModel.from_file(
            path, img_size=self.img_size, nb_classes=self.nb_classes
        )
        
        return ArchaiModel(model, model.to_hash(), metadata={'parent': None})
    
    @overrides
    def load_model_weights(self, model: ArchaiModel, path: str) -> None:
        model.arch.load_state_dict(torch.load(path))

    @overrides
    def random_sample(self) -> ArchaiModel:
        nas_model = None

        while not nas_model:
            # randomly pick number of layers    
            nb_layers = self.rng.randint(self.min_layers, self.max_layers)

            # Samples `base_channels` and `delta_channels`
            ch_per_scale = {
                'base_channels': self.rng.choice(self.base_channels_list),
                'delta_channels': self.rng.choice(self.delta_channels_list),
                'mult_delta': self.mult_delta
            }

            # Samples `post_upsample_layers`
            post_upsample_layers = (
                self.rng.choice(self.post_upsample_layers_list) if self.post_upsample_layers_list else 1
            )

            # Builds channels per level map using the sampled `base_channels` and `delta_channels`
            ch_map = SegmentationDagModel._get_channels_per_scale(ch_per_scale, self.max_downsample_factor, True)

            # Input node
            graph = [{'name': 'input', 'inputs': None, 'op': self.rng.choice(self.operations), 'scale': 1}]
            node_list = ['input']

            # Used to control `max_scale_delta`
            idx2scale = list(ch_map.keys())
            scale2idx = {scale: i for i, scale in enumerate(ch_map.keys())}

            for layer in range(nb_layers):
                is_output = (layer == nb_layers - 1)
                last_layer = graph[-1]

                new_node = {
                    'name': 'output' if is_output else f'layer_{layer}',
                    'op': None if is_output else self.rng.choice(self.operations),
                    'inputs': [last_layer['name']]
                }

                # Choose scale
                last_scale_idx = scale2idx[last_layer['scale']]

                # Samples a delta value for the current scale index
                scale_options = list(range(
                    max(-self.max_scale_delta, -last_scale_idx),
                    1 + min(self.max_scale_delta, len(ch_map) - last_scale_idx - 1)
                ))

                sample_weights = np.array([
                    1 if delta < 0 else self.downsample_prob_ratio
                    for delta in scale_options
                ])
                scale_delta = self.rng.choices(scale_options, k=1, weights=sample_weights)[0]

                # Assigns the new scale to the new node
                new_node['scale'] = idx2scale[last_scale_idx + scale_delta]

                # Choose inputs
                if len(node_list) > 1:
                    for i in range(2, 1 + self.rng.randint(2, min(len(node_list), self.max_skip_connection_length))):
                        if self.skip_connections and self.rng.random() < 0.5:
                            new_node['inputs'].append(node_list[-i])

                # Adds node
                graph.append(new_node)
                node_list.append(new_node['name'])

            # Builds model
            model = SegmentationDagModel(
                graph, ch_per_scale, post_upsample_layers, 
                img_size=self.img_size, nb_classes=self.nb_classes
            )

            found_valid, macs = self.is_valid_model(model)
            
            if found_valid:
                nas_model = ArchaiModel(
                    model, model.to_hash(), {'parent': None, 'macs': macs}
                )
    
        return nas_model

    @overrides
    def mutate(self, base_model: ArchaiModel, patience: int = 5) -> ArchaiModel:
        parent_id = base_model.archid
        nb_tries = 0

        while nb_tries < patience:
            nb_tries += 1
            graph = copy.deepcopy(list(base_model.arch.graph.values()))
            channels_per_scale = copy.deepcopy(base_model.arch.channels_per_scale)

            # sanity check the graph
            assert len(graph) > 1
            assert graph[-1]['name'] == 'output'
            assert graph[0]['name'] == 'input'

            # `base_channels` and `delta_channels` mutation
            channels_per_scale = {
                'base_channels': self.random_neighbor(self.base_channels_list, channels_per_scale['base_channels']),
                'delta_channels': self.random_neighbor(self.delta_channels_list, channels_per_scale['delta_channels']),
                'mult_delta': self.mult_delta
            }

            # `post_upsample_layers` mutation
            post_upsample_layers = self.random_neighbor(
                self.post_upsample_layers_list,
                base_model.arch.post_upsample_layers
            )

            # pick a node at random (but not input node)
            # and change its operator at random
            # and its input sources
            chosen_node_idx = self.rng.randint(1, len(graph) - 1)
            node = graph[chosen_node_idx]
            
            if node['name'] != 'output':
                node['op'] = self.rng.choice(self.operations)
            
            # choose up to k inputs from previous nodes
            max_inputs = 3 # TODO: make config 

            # Gets the out connections for each node
            edges = [tuple(k.split('-')) for k in base_model.arch.edge_dict.keys()]
            out_degree = lambda x: len([(orig, dest) for orig, dest in edges if orig == x])

            if node['name'] != 'input':
                k = min(chosen_node_idx, self.rng.randint(1, max_inputs))
                input_idxs = self.rng.sample(range(chosen_node_idx), k)

                # Removes everything except inputs that have out degree == 1
                node['inputs'] = [input for input in node['inputs'] if out_degree(input) <= 1]

                # Adds `k` new inputs
                node['inputs'] += [
                    graph[idx]['name'] for idx in input_idxs
                    if graph[idx]['name'] not in node['inputs']
                ]

            # compile the model
            nbr_model = SegmentationDagModel(
                graph, channels_per_scale, post_upsample_layers,
                img_size=self.img_size, nb_classes=self.nb_classes
            )

            if not self.is_valid_model(nbr_model)[0]:
                logger.info(f'Neighbor generation {base_model.arch.to_hash()} -> {nbr_model.to_hash()} failed')
                continue

            return ArchaiModel(nbr_model, nbr_model.to_hash(), metadata={'parent': parent_id})


    @overrides
    def crossover(self, model_list: List[ArchaiModel], patience: int = 30) -> Optional[ArchaiModel]:
        if len(model_list) < 2:
            return

        # Chooses randomly left and right models
        left_m, right_m = self.rng.sample(model_list, 2)
        left_arch, right_arch = [list(m.arch.graph.values()) for m in [left_m, right_m]]

        # Renames nodes to avoid name collision
        left_arch = self.rename_dag_node_list(left_arch, 'left')
        right_arch = self.rename_dag_node_list(right_arch, 'right')

        # Stores node names
        left_n, right_n = [[n['name'] for n in g] for g in [left_arch, right_arch]]

        if len(left_n) <= 2 or len(right_n) <= 2:
            return

        # Tries to merge left_m and right_m
        result_g = None
        nb_tries = 0

        for nb_tries in range(patience):
            left_g, right_g = copy.deepcopy(left_arch), copy.deepcopy(right_arch)
            nb_tries += 1

            # Samples a pivot node from the left model
            left_pivot_idx = self.rng.randint(1, len(left_n) - 2)
            left_pivot = left_n[left_pivot_idx]

            # Samples a pivot node from the right model w/ the same scale as the left_pivot
            # excluding input and output nodes
            right_candidates = [
                i
                for i, fields in enumerate(right_g)
                if fields['scale'] == left_g[left_pivot_idx]['scale'] and\
                0 < i < (len(right_n) - 1)
            ]

            if len(right_candidates) > 0:
                # Picks a right pivot
                right_pivot_idx = self.rng.choice(right_candidates)

                # Splits right_g and left_g using the pivot nodes
                left_half = left_g[:left_pivot_idx + 1]
                right_half = right_g[right_pivot_idx:]

                # Gets node2idx for right model
                right_node2idx = {node: i for i, node in enumerate(right_n)}

                # Corrects connections from right_g
                for fields in right_half[::-1]:
                    for inp_idx, inp in enumerate(fields['inputs']):

                        # Checks if this connection falls outside of right_half
                        if inp not in right_n[right_pivot_idx:]:
                            # Finds a new starting node to connect this edge
                            # with the same original input scale
                            candidates = [
                                n['name'] for n in left_half
                                if n['scale'] == right_g[right_node2idx[inp]]['scale']
                            ]

                            fields['inputs'][inp_idx] = (
                                self.rng.choice(candidates) if len(candidates) > 0
                                else None
                            )
                            
                # Renames end node
                right_half[-1]['name'] = 'output'

                # Connects left_half and right_half
                if left_pivot not in right_half[0]['inputs']:
                    right_half[0]['inputs'].append(left_pivot)

                # Merge and rename nodes
                result_g = self.rename_dag_node_list(left_half + right_half, add_input_output=True)
                
                # Pick `channels_per_scale` and `post_upsample_layers` from left_m or right_m
                ch_map = self.rng.choice(
                    [copy.deepcopy(left_m.arch.channels_per_scale), copy.deepcopy(right_m.arch.channels_per_scale)]
                )

                post_upsample_layers = self.rng.choice(
                    [left_m.arch.post_upsample_layers, right_m.arch.post_upsample_layers]
                )

                try:
                    result_model = self.load_from_graph(
                        result_g,
                        {'base_channels': ch_map['base_channels'],
                        'delta_channels': ch_map['delta_channels'],
                        'mult_delta': ch_map['mult_delta']},
                        post_upsample_layers
                    )
                    
                    out_shape = result_model.arch.validate_forward(
                        torch.randn(1, 3, *result_model.arch.img_size[::-1])
                    ).shape

                    assert out_shape == torch.Size([1, self.nb_classes, *result_model.arch.img_size[::-1]])
                
                except Exception as e:
                    logger.info(
                        f'Crossover between {left_m.arch.to_hash()}, {right_m.arch.to_hash()} failed '
                        f'(nb_tries = {nb_tries})'
                    )
                    logger.info(str(e))
                    print(str(e))
                    continue
                
                result_model.metadata['parents'] = left_m.archid + ',' + right_m.archid
                return result_model
