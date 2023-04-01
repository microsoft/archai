# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from functools import partial
from typing import Tuple, Optional, List

import torch
from torch import nn

from archai.discrete_search.search_spaces.config import (
    ArchConfig, ArchParamTree, DiscreteChoice, ConfigSearchSpace, repeat_config
)
from .ops import ReluConv2d, Conv2dSamePadding, OPS


def hgnet_param_tree_factory(stem_strides: Tuple[int, ...] = (2, 4),
                             max_num_hourglass: int = 2,
                             share_hourglass_arch: bool = False,
                             base_channels: Tuple[int, ...] = (16, 24, 32, 48),
                             op_subset: Tuple[str, ...] = ('conv3x3', 'conv5x5', 'conv7x7'),
                             num_blocks: int = 4,
                             downsample_block_max_ops: int = 5,
                             post_upsample_max_ops: int = 3,
                             skip_block_max_ops: int = 3,
                             upsample_block_max_ops: int = 4):
    assert num_blocks > 1, 'num_blocks must be greater than 1'

    return ArchParamTree({
        'stem_stride': DiscreteChoice(stem_strides),
        'base_ch': DiscreteChoice(base_channels),

        'hourglasses': repeat_config({

            'downsample_blocks': repeat_config({
                'layers': repeat_config({
                    'op': DiscreteChoice(op_subset)
                }, repeat_times=range(1, downsample_block_max_ops + 1), share_arch=False),

                'ch_expansion_factor': DiscreteChoice([1.0, 1.2, 1.5, 1.6, 2.0, 2.2]),
            }, repeat_times=num_blocks),

            'skip_blocks': repeat_config({
                'layers': repeat_config({
                    'op': DiscreteChoice(op_subset)
                }, repeat_times=range(0, skip_block_max_ops+1), share_arch=False),
            }, repeat_times=num_blocks-1),

            'upsample_blocks': repeat_config({
                'layers': repeat_config({
                    'op': DiscreteChoice(op_subset)
                }, repeat_times=range(1, upsample_block_max_ops+1), share_arch=False),
            }, repeat_times=num_blocks-1),
        }, repeat_times=range(1, max_num_hourglass+1), share_arch=share_hourglass_arch),

        'post_upsample_layers': repeat_config({
            'op': DiscreteChoice(op_subset)
        }, repeat_times=range(0, post_upsample_max_ops+1), share_arch=False)
    })


class Hourglass(nn.Module):
    def __init__(self, arch_config: ArchConfig, base_channels: int):
        super().__init__()
        self.base_channels = base_channels

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.chs = [self.base_channels]

        # Calculates channels on each branch
        for block_cfg in arch_config.pick('downsample_blocks'):
            self.chs.append(
                int(self.chs[-1] * block_cfg.pick('ch_expansion_factor'))
            )

        self.nb_blocks = len(self.chs) - 1

        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        for block_idx, block_cfg in enumerate(arch_config.pick('downsample_blocks')):
            in_ch, out_ch = self.chs[block_idx], self.chs[block_idx + 1]

            down_block = [
                OPS[layer_cfg.pick('op')](
                    (in_ch if layer_idx == 0 else out_ch),
                    out_ch,
                    stride=(2 if (layer_idx == 0 and block_idx > 0) else 1)
                )
                for layer_idx, layer_cfg in enumerate(block_cfg.pick('layers'))
            ]

            self.down_blocks.append(nn.Sequential(*down_block))

        # Skip blocks
        self.skip_blocks = nn.ModuleList()
        for block_idx, block_cfg in enumerate(arch_config.pick('skip_blocks')):
            out_ch = self.chs[block_idx + 1]

            skip_block = [
                OPS.get(layer_cfg.pick('op'))(out_ch, out_ch)
                for layer_idx, layer_cfg in enumerate(block_cfg.pick('layers'))
            ]

            self.skip_blocks.append(nn.Sequential(*skip_block))

        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        for block_idx, block_cfg in enumerate(arch_config.pick('upsample_blocks')):
            in_ch, out_ch = self.chs[block_idx + 1], self.chs[block_idx + 2]

            up_block = [
                OPS.get(layer_cfg.pick('op'))(
                    (out_ch if layer_idx == 0 else in_ch), in_ch
                )
                for layer_idx, layer_cfg in enumerate(block_cfg.pick('layers'))
            ]

            self.up_blocks.append(nn.Sequential(*up_block))

        # Converts output to `base_channels`
        self.final_conv = nn.Conv2d(self.chs[1], base_channels, kernel_size=1)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        skip_connections = [0 for _ in range(self.nb_blocks - 1)]
        inp = x

        for i in range(self.nb_blocks - 1):
            out = self.down_blocks[i](inp)
            skip_connections[i] = self.skip_blocks[i](out)
            inp = out

        # Last downsample branch
        out = self.down_blocks[-1](inp)

        for i in range(self.nb_blocks - 1)[::-1]:
            out = skip_connections[i] + self.up_blocks[i](self.upsample(out))

        return self.final_conv(out)


class StackedHourglass(nn.Module):
    def __init__(self, arch_config: ArchConfig, num_classes: int, in_channels: int = 3):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.arch_config = arch_config
        self.base_channels = arch_config.pick('base_ch')

        # Classifier
        self.classifier = nn.Conv2d(self.base_channels, num_classes, kernel_size=1)

        # Stem convolution
        self.stem_stride = arch_config.pick('stem_stride')
        self.stem_conv = ReluConv2d(
            in_channels=in_channels, out_channels=self.base_channels,
            stride=self.stem_stride
        )

        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=self.stem_stride)

        self.hgs = nn.Sequential(*[
            Hourglass(hg_conf, self.base_channels)
            for hg_conf in arch_config.pick('hourglasses')
        ])

        self.post_upsample = nn.Sequential(*[
            OPS[layer_cfg.pick('op')](self.base_channels, self.base_channels)
            for layer_cfg in arch_config.pick('post_upsample_layers')
        ])

        self.classifier = Conv2dSamePadding(self.base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem_conv(x)
        out = self.hgs(out)
        out = self.post_upsample(self.final_upsample(out))

        return self.classifier(out)


class HgnetSegmentationSearchSpace(ConfigSearchSpace):
    def __init__(self,
                 num_classes: int,
                 img_size: Tuple[int, int],
                 in_channels: int = 3,
                 op_subset: Tuple[str, ...] = ('conv3x3', 'conv5x5', 'conv7x7'),
                 stem_strides: Tuple[int, ...] = (1, 2, 4),
                 num_blocks: int = 4,
                 downsample_block_max_ops: int = 4,
                 skip_block_max_ops: int = 2,
                 upsample_block_max_ops: int = 4,
                 post_upsample_max_ops: int = 3,
                 **ss_kwargs):

        possible_downsample_factors = [
            2**num_blocks * stem_stride for stem_stride in stem_strides
        ]

        w, h = img_size

        assert all(w % d_factor == 0 for d_factor in possible_downsample_factors), \
            f'Image width must be divisible by all possible downsample factors ({2**num_blocks} * stem_stride)'

        assert all(h % d_factor == 0 for d_factor in possible_downsample_factors), \
            f'Image height must be divisible by all possible downsample factors ({2**num_blocks} * stem_stride)'

        ss_kwargs['builder_kwargs'] = {
            'op_subset': op_subset,
            'stem_strides': stem_strides,
            'num_blocks': num_blocks,
            'downsample_block_max_ops': downsample_block_max_ops,
            'skip_block_max_ops': skip_block_max_ops,
            'upsample_block_max_ops': upsample_block_max_ops,
            'post_upsample_max_ops': post_upsample_max_ops
        }

        ss_kwargs['model_kwargs'] = {
            'num_classes': num_classes,
            'in_channels': in_channels,
        }

        self.img_size = img_size
        self.in_channels = in_channels

        super().__init__(StackedHourglass, hgnet_param_tree_factory, **ss_kwargs)
