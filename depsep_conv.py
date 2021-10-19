import torch.nn as nn


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.batchnorm = nn.BatchNorm2d(in_channels)
        self.pointconv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, groups=1, bias=bias)

    def forward(self, x):
        x = self.depthconv(x)
        x = self.batchnorm(x)
        x = self.pointconv(x)
        return x

