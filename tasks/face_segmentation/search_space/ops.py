from functools import partial
from itertools import chain

import torch
from torch import nn

class Conv2dSamePadding(nn.Conv2d):
    __doc__ = nn.Conv2d.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pad_shape = tuple(chain(*[
            [k // 2 + (k - 2 * (k // 2)) - 1, k // 2]
            for k in self.kernel_size[::-1]
        ]))

        self.pad = nn.ZeroPad2d(pad_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(self.pad(x), self.weight, self.bias)


class ReluConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, 
                 bias: bool = False,  **kwargs):
        super().__init__()

        self.conv = Conv2dSamePadding(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, bias=bias
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))

OPS = {
    'conv3x3': partial(ReluConv2d, kernel_size=3),
    'conv5x5': partial(ReluConv2d, kernel_size=5),
    'conv7x7': partial(ReluConv2d, kernel_size=7),
}
