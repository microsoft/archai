from collections import OrderedDict
from functools import partial
from typing import Tuple, List, Dict, MutableMapping, Optional
from pathlib import Path
import math
import yaml

from torch import nn
import torch

class NormalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = True, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(self.bn(self.conv(x)))


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, expand_ratio: float = 1.0, id_skip: bool = False,
                 bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.id_skip = id_skip

        # Expansion phase
        inp = in_channels  # number of input channels
        oup = int(in_channels * self.expand_ratio)  # number of output channels

        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=bias)
            self._bn0 = nn.BatchNorm2d(num_features=oup)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=kernel_size, stride=stride, bias=bias, padding=padding
        )
        self._bn1 = nn.BatchNorm2d(num_features=oup)

        # Output phase
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=out_channels, kernel_size=1, bias=bias)
        self._bn2 = nn.BatchNorm2d(num_features=out_channels)
        self._act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Expansion and Depthwise Convolution
        out = x

        if self.expand_ratio != 1:
            out = self._bn0(self._expand_conv(out))  # No activation function here
        out = self._act(self._bn1(self._depthwise_conv(out)))

        # Pointwise conv.
        out = self._bn2(self._project_conv(out))

        # Skip connection
        if self.id_skip and self.stride == 1 and self.in_channels == self.out_channels:
            out = out + x

        return out

OPS = {
    'conv3x3': partial(NormalConvBlock, kernel_size=3, padding=1),
    'conv5x5': partial(NormalConvBlock, kernel_size=5, padding=2),
    'conv7x7': partial(NormalConvBlock, kernel_size=7, padding=3),
    'mbconv3x3_e1': partial(SeparableConvBlock, kernel_size=3, padding=1),
    'mbconv3x3_e2': partial(SeparableConvBlock, kernel_size=3, padding=1, expand_ratio=2),
    'mbconv5x5_e1': partial(SeparableConvBlock, kernel_size=5, padding=2),
    'mbconv5x5_e2': partial(SeparableConvBlock, kernel_size=5, padding=2, expand_ratio=2),
}


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
    def __init__(self, node_names: List[str], edge_dict: MutableMapping[Tuple[str, str], nn.Module],
                 channels_per_scale: Dict, stem_stride: int = 2, img_size: int = 256,
                 nb_classes: int = 19, post_upsample_layers: int = 1, node_info: Optional[Dict] = None):
        '''SegmentationModel constructor. Should not be called directly, use `SegmentationModel.from_yaml` '''
        ''' or `SegmentationModel.from_config` instead'''
        super().__init__()
        assert img_size % 32 == 0, 'Image size must be a multiple of 32'

        self.node_names = node_names
        self.channels_per_scale = channels_per_scale
        self.node_info = node_info
        self.edge_dict = nn.ModuleDict(edge_dict)
        self.stem_stride = stem_stride
        self.img_size = img_size
        self.nb_classes = nb_classes
        self.post_upsample_layers = post_upsample_layers

        # Checks if the edges are in topological order        
        self._validate_edges(edge_dict)

        # Stem block
        stem_ch = next(iter(edge_dict.values())).in_ch
        self.stem_block = OPS['conv3x3'](3, stem_ch, stride=self.stem_stride)

        # Upsample layers
        self.up = nn.Upsample(size=(self.img_size, self.img_size), mode='nearest')
        output_ch = self.channels_per_scale[self.node_info['output']['scale']]

        self.post_upsample = nn.Sequential(
            *[
                OPS['conv3x3'](output_ch if i == 0 else stem_ch, stem_ch, stride=1)
                for i in range(self.post_upsample_layers)
            ]
        )

        # Classifier
        self.classifier = nn.Conv2d(
            stem_ch, self.nb_classes,
            kernel_size=1
        )

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
        resolution = (256 // self.stem_stride)

        inputs = {node_name: 0 for node_name in self.node_names}
        inputs['input'] = self.stem_block(x)

        for edge, module in self.edge_dict.items():
            in_node, out_node = edge.split('-')
            in_nodes.add(in_node)

            # Checks if the resolution of each node is correct
            assert inputs[in_node].shape[3] == int(resolution // self.node_info[in_node]['scale']),\
                'Input resolution does not match the node resolution.'

            inputs[out_node] = inputs[out_node] + module(inputs[in_node])

            assert inputs[out_node].shape[1] == self.channels_per_scale[self.node_info[out_node]['scale']],\
                'Output channel does not match the node channel scale.'

        assert all(node in in_nodes for node in set(self.node_info.keys()) - {'output'}),\
            f'Unused nodes were detected: {set(self.node_info.keys()) - in_nodes - set(["output"])}.'

        output = self.post_upsample(self.up(inputs['output']))
        return self.classifier(output)

    @classmethod
    def from_config(cls, node_list: List[Dict], channels_per_scale: Dict) -> 'SegmentationNasModel':
        """Creates a SegmentationModel from a config file

        Args:
            node_list (List[Dict]): List of dictionaries with the following keys:
                * name (str): Name of the node
                * op (str): Name of the operation used to process the node
                * inputs (List[str]): List of input nodes
                * scale (int): Scale of the node (higher means smaller resolutions)
            channels_per_scale (Dict): Dictionary with the number of channels that should be 
            used for each scale value, e.g: {1: 32, 2: 64, 4: 128}.

        Returns:
            SegmentationNasModel: A SegmentationNasModel instance
        """
        node_info = OrderedDict([(node['name'], node) for node in node_list])
        edges = [(in_node, node['name']) for node in node_list if node['name'] != 'input' for in_node in node['inputs']]

        assert 'input' in node_info
        assert 'output' in node_info

        # Builds an `OrderedDict` with the mapping "in_node-out_node": nn.Module
        module_dict = OrderedDict([
            (f'{i}-{o}', Block(
                in_ch=channels_per_scale[node_info[i]['scale']],
                out_ch=channels_per_scale[node_info[o]['scale']],
                in_scale=node_info[i]['scale'],
                out_scale=node_info[o]['scale'],
                op_name=node_info[i]['op']
            )) for i, o in edges
        ])

        return cls([n['name'] for n in node_list], module_dict, channels_per_scale, node_info=node_info)

    @classmethod
    def from_file(cls, config_file: str) -> 'SegmentationNasModel':
        """Creates a SegmentationNasModel from a YAML config file

        Args:
            config_file (str): Path to the YAML config file, following the format:
                ```
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
        Returns:
            SegmentationNasModel: A SegmentationNasModel instance
        """
        config_file = Path(config_file)
        assert config_file.is_file()
        assert config_file.suffix == '.yaml'

        config_dict = yaml.safe_load(open(config_file))
        return cls.from_config(config_dict['architecture'], config_dict['channels_per_scale'])

    def view(self):
        import graphviz
        scales = []
        dot = graphviz.Digraph('architecture')
        dot.engine = 'neato'

        for i, node in enumerate(self.node_names):
            scales.append(self.node_info[node]["scale"])
            n = dot.node(node, label=self.node_info[node]['op'], pos=f'{i*1.5 + 2},-{math.log2(scales[-1])}!')

        for scale in sorted(list(set(scales))):
            dot.node(f'scale-{scale}', label=f'scale={scale}', pos=f'0,-{math.log2(scale)}!')

        for edge in self.edge_dict:
            in_node, out_node = edge.split('-')
            dot.edge(in_node, out_node)

        # Shows the graph 
        return dot

    def to_file(self, path: str) -> None:
        content = {
            'channels_per_scale': self.channels_per_scale,
            'architecture': list(self.node_info.values())
        }

        with open(path, 'w') as fp:
            fp.write(yaml.dump(content))

        m = SegmentationNasModel.from_file(path)
        assert content['architecture'] == list(m.node_info.values())
        assert content['channels_per_scale'] == m.channels_per_scale
