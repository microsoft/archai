from collections import OrderedDict
from copy import deepcopy
from typing import Tuple, List, Dict, MutableMapping, Optional
from pathlib import Path
import random
import json
from hashlib import sha1
import math
import yaml

from torch import nn
import torch

from archai.algos.evolution_pareto_image_seg.ops import OPS, SeparableConvBlock, NormalConvBlock

class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, in_scale: int, out_scale: int, op_name: str):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.in_scale, self.out_scale = in_scale, out_scale
        self.op_name = op_name

        assert op_name in OPS
        assert (out_scale % in_scale == 0) or (in_scale % out_scale == 0)

        if out_scale >= in_scale:
            self.op = nn.Sequential(
                OPS[op_name](in_ch, out_ch, stride=int(out_scale // in_scale))
            )
        else:
            self.op = nn.Sequential(
                OPS[op_name](in_ch, out_ch, stride=1),
                nn.Upsample(scale_factor=int(in_scale // out_scale), mode='nearest')
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.op(input)


class SegmentationNasModel(torch.nn.Module):
    def __init__(self, graph: List[Dict], channels_per_scale: Dict, post_upsample_layers: int = 1,
                 stem_stride: int = 2, img_size: Tuple[int, int] = (256, 256), nb_classes: int = 19):
        """Creates a SegmentationNasModel from a configuration

        Args:
            graph (List[Dict]): List of dictionaries with the following keys:
                * name (str): Name of the node
                * op (str): Name of the operation used to process the node
                * inputs (List[str]): List of input nodes
                * scale (int): Scale of the node (higher means smaller resolutions)
            channels_per_scale (Dict): Dictionary with the number of channels that should be 
                used for each scale value, e.g: {1: 32, 2: 64, 4: 128} or a dictionary containing
                `base_channels`, `delta_channels` and optionally a `mult_delta` flag.
                For instance, {'base_channels': 24, 'delta_channels': 2}, is equivalent to
                {1: 24, 2: 26, 4: 28, 8: 30, 16: 32}, and {'base_channels': 24, 'delta_channels': 2,
                mult_delta: True} is equivalent to {1: 24, 2: 48, 4: 96, 8: 192, 16: 384}.
            post_upsample_layers (int): Number of post-upsample layers
            stem_strid (int): Stride of the first convolution
            img_size (int): Image size
            nb_classes (int): Number of classes for segmentation

        Returns:
            SegmentationNasModel: A SegmentationNasModel instance
        """

        super().__init__()
        if isinstance(img_size, int):
            assert img_size % 32 == 0, 'Image size must be a multiple of 32'
        else:
            assert img_size[0] % 32 == 0 and img_size[1] % 32 == 0, 'Image size must be a multiple of 32'

        self.graph = OrderedDict([(n['name'], n) for n in graph])
        self.node_names = [n['name'] for n in self.graph.values()]
        self.channels_per_scale = self._get_channels_per_scale(channels_per_scale)
        self.edge_dict = nn.ModuleDict(self._get_edge_list(self.graph, self.channels_per_scale))
        self.stem_stride = stem_stride
        self.img_size = img_size
        self.nb_classes = nb_classes
        self.post_upsample_layers = post_upsample_layers

        # Checks if the edges are in topological order        
        self._validate_edges(self.edge_dict)

        # Stem block
        stem_ch = self.channels_per_scale[self.graph['input']['scale']]
        self.stem_block = OPS['conv3x3'](3, stem_ch, stride=self.stem_stride)

        # Upsample layers
        w, h = self.img_size
        self.up = nn.Upsample(size=(h, w), mode='nearest')
        output_ch = self.channels_per_scale[self.graph['output']['scale']]

        self.post_upsample = nn.Sequential(
            *[
                OPS['conv3x3'](output_ch if i == 0 else self.channels_per_scale[1], self.channels_per_scale[1], stride=1)
                for i in range(self.post_upsample_layers)
            ]
        )

        # Classifier
        self.classifier = nn.Conv2d(
            stem_ch, self.nb_classes,
            kernel_size=1
        )

    @classmethod
    def _get_channels_per_scale(cls, ch_per_scale: Dict, max_downsample_factor: int = 16, 
                                remove_spec: bool = False) -> Dict:
        ch_per_scale = deepcopy(ch_per_scale)
        scales = [1, 2, 4, 8, 16]
        scales = [s for s in scales if s <= max_downsample_factor]
 
        # Builds `ch_per_scale` using `base_channels` and `delta_channels`
        ch_per_scale['mult_delta'] = ch_per_scale.get('mult_delta', False)
        assert 'base_channels' in ch_per_scale
        assert 'delta_channels' in ch_per_scale
        
        assert len(ch_per_scale.keys()) == 3, \
            'Must specify only `base_channels`, `delta_channels` and `mult_delta`'

        if ch_per_scale['mult_delta']:
            ch_per_scale.update({
                scale: ch_per_scale['base_channels'] * ch_per_scale['delta_channels']**i
                for i, scale in enumerate(scales)
            })
        else:
           ch_per_scale.update({
                scale: ch_per_scale['base_channels'] + ch_per_scale['delta_channels']*i
                for i, scale in enumerate(scales)
            }) 
        
        if remove_spec:
            ch_per_scale.pop('base_channels', None)
            ch_per_scale.pop('delta_channels', None)
            ch_per_scale.pop('mult_delta', None)
        
        return ch_per_scale

    def _get_edge_list(self, graph: 'OrderedDict[str, Dict]',
                         channels_per_scale: Dict) -> MutableMapping[Tuple[str, str], nn.Module]:
        assert 'input' in graph
        assert 'output' in graph

        edges = [(in_node, node['name']) for node in graph.values()
                 if node['name'] != 'input' for in_node in node['inputs']]

        # Returns an `OrderedDict` with the mapping "in_node-out_node": nn.Module
        return OrderedDict([
            (f'{i}-{o}', Block(
                in_ch=channels_per_scale[graph[i]['scale']],
                out_ch=channels_per_scale[graph[o]['scale']],
                in_scale=graph[i]['scale'],
                out_scale=graph[o]['scale'],
                op_name=graph[i]['op']
            )) for i, o in edges
        ])

    def _validate_edges(self, edge_dict: MutableMapping[Tuple[str, str], nn.Module]) -> None:
        '''Checks if the edges are in topological order '''
        visited_nodes = {'input'}

        for edge in edge_dict.keys():
            in_node, out_node = edge.split('-')
            visited_nodes.add(out_node)

            assert in_node in visited_nodes,\
                'SegmentationModel received a list of nodes that is not in topological order'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = {node_name: 0 for node_name in self.node_names}
        inputs['input'] = self.stem_block(x)

        for edge, module in self.edge_dict.items():
            in_node, out_node = edge.split('-')
            inputs[out_node] = inputs[out_node] + module(inputs[in_node])

        output = self.post_upsample(self.up(inputs['output']))
        return self.classifier(output)

    def validate_forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Checks if the constructed model is working as expected.'''
        in_nodes = set()
        resolution = (self.img_size // self.stem_stride)

        inputs = {node_name: 0 for node_name in self.node_names}
        inputs['input'] = self.stem_block(x)

        for edge, module in self.edge_dict.items():
            in_node, out_node = edge.split('-')
            in_nodes.add(in_node)

            # Checks if the resolution of each node is correct
            assert inputs[in_node].shape[3] == int(resolution // self.graph[in_node]['scale']),\
                'Input resolution does not match the node resolution.'

            inputs[out_node] = inputs[out_node] + module(inputs[in_node])

            assert inputs[out_node].shape[1] == self.channels_per_scale[self.graph[out_node]['scale']],\
                'Output channel does not match the node channel scale.'

        assert all(node in in_nodes for node in set(self.graph.keys()) - {'output'}),\
            f'Unused nodes were detected: {set(self.graph.keys()) - in_nodes - set(["output"])}.'

        output = self.post_upsample(self.up(inputs['output']))
        return self.classifier(output)

    @classmethod
    def from_file(cls, config_file: str, img_size: int = 256, nb_classes: int = 19) -> 'SegmentationNasModel':
        """Creates a SegmentationNasModel from a YAML config file

        Args:
            config_file (str): Path to the YAML config file, following the format:
                ```
                post_upsample_layers: 2
                channels_per_scale:
                    1: 32
                    2: 64
                architecture:
                    - name: input
                      scale: 1
                      op: conv3x3
                      inputs: null
                    - name: node0
                      scale: 2
                      op: conv5x5
                      inputs: [input]
                    - name: output
                      scale: 4
                      op: conv3x3
                      inputs: [node0, node1]
                ```
            img_size (int): The size of the input image.
            nb_classes (int): The number of classes in the dataset.
        Returns:
            SegmentationNasModel: A SegmentationNasModel instance
        """
        config_file = Path(config_file)
        assert config_file.is_file()
        assert config_file.suffix == '.yaml'

        config_dict = yaml.safe_load(open(config_file))
        return cls(config_dict['architecture'], config_dict['channels_per_scale'],
                   config_dict['post_upsample_layers'], img_size=img_size, nb_classes=nb_classes)

    def view(self):
        import graphviz
        scales = []
        dot = graphviz.Digraph('architecture', graph_attr={'splines': 'true', 'overlap': 'true'})
        dot.engine = 'neato'

        for i, node in enumerate(self.node_names):
            scales.append(self.graph[node]['scale'])
            dot.node(node, label=self.graph[node]['op'], pos=f'{i*1.5 + 2},-{math.log2(2*scales[-1])}!')

        for scale in sorted(list(set(scales))):
            dot.node(
                f'scale-{scale}', label=f'scale={2*scale}, ch={self.channels_per_scale[scale]}',
                pos=f'-1,-{math.log2(2*scale)}!'
            )

        for edge in self.edge_dict:
            in_node, out_node = edge.split('-')
            dot.edge(in_node, out_node)

        # Adds post upsample
        dot.node('upsample', label=f'Upsample + {self.post_upsample_layers} x Conv 3x3', pos=f'{i*1.5 + 2},0!')
        dot.edge('output', 'upsample')

        # Shows the graph 
        return dot

    def to_config(self) -> Dict:
        ch_map = self.channels_per_scale
        
        if 'base_channels' in ch_map:
            ch_map = {
                'base_channels': ch_map['base_channels'],
                'delta_channels': ch_map['delta_channels']
            }
        
            # We only put the `mult_delta` flag in config dict if it's active 
            if self.channels_per_scale['mult_delta']:
                ch_map['mult_delta'] = True

        return {
            'post_upsample_layers': self.post_upsample_layers,
            'channels_per_scale': ch_map,
            'architecture': list(self.graph.values())
        }

    def to_file(self, path: str) -> None:
        content = self.to_config()

        with open(path, 'w') as fp:
            fp.write(yaml.dump(content))

        m = SegmentationNasModel.from_file(path, self.img_size, self.nb_classes)
        assert content['architecture'] == list(m.graph.values())
        assert content['post_upsample_layers'] == len(self.post_upsample)
        assert all(
            m.channels_per_scale[k] == v 
            for k, v in content['channels_per_scale'].items()
        )

    def to_hash(self) -> str:
        config = self.to_config()
        arch_str = json.dumps(config, sort_keys=True, ensure_ascii=True)
        return sha1(arch_str.encode('ascii')).hexdigest()

    @classmethod
    def sample_model(cls, 
                    base_channels_list: List[int],
                    delta_channels_list: List[int],
                    post_upsample_layer_list: List[int],
                    max_downsample_factor: int = 16,
                    nb_layers: int = 24,
                    skip_connections: bool = True,
                    max_skip_connection_length: int = 3,
                    max_scale_delta: Optional[int] = None,
                    op_subset: Optional[List[str]] = None,
                    downsample_prob_ratio: float = 1.0,
                    mult_delta: bool = False,
                    img_size: int = 256):
        '''Uniform random sample an architecture (nn.Module)'''
        operations = list(OPS.keys())
        
        if op_subset:
            operations = [op for op in operations if op in op_subset]

        # Samples `base_channels` and `delta_channels`
        ch_per_scale = {
            'base_channels': random.choice(base_channels_list),
            'delta_channels': random.choice(delta_channels_list),
            'mult_delta': mult_delta
        }

        # Samples `post_upsample_layers`
        post_upsample_layers = (
            random.choice(post_upsample_layer_list) if post_upsample_layer_list else 1
        )

        # Builds channels per level map using the sampled `base_channels` and `delta_channels`
        ch_map = cls._get_channels_per_scale(ch_per_scale, max_downsample_factor, True)

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
                scale_options = list(range(
                    max(-max_scale_delta, -last_scale_idx),
                    1 + min(max_scale_delta, len(ch_map) - last_scale_idx - 1)
                ))
                sample_weights = [
                    1 if delta < 0 else downsample_prob_ratio
                    for delta in scale_options
                ]

                scale_delta = random.choices(scale_options, k=1, weights=sample_weights)[0]

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

        return SegmentationNasModel(graph, ch_per_scale, post_upsample_layers, img_size=img_size)
