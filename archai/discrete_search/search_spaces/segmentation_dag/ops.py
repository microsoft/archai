from functools import partial
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