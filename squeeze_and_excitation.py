from torch import nn


class SqueezeExcitationBlock(nn.Module):

    def __init__(self, num_channel, reduction=16, bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.Sequential(
            nn.Linear(num_channel, num_channel // reduction, bias=bias),
            nn.ReLU(),
            nn.Linear(num_channel // reduction, num_channel, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.layers(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        