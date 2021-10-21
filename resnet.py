"""ResNet15.
@inproceedings{tang2018deep,
  title={Deep residual learning for small-footprint keyword spotting},
  author={Tang, Raphael and Lin, Jimmy},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5484--5488},
  year={2018},
  organization={IEEE}
}
"""

# import os
import torch  # noqa: F401
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.linear import Linear
from depsep_conv import SeparableConv2d
from squeeze_and_excitation import SqueezeExcitationBlock


class ResNetBlock(nn.Module):
  
  def __init__(self, num_channels, kernel_size=3, dilation=1, activation=torch.nn.ReLU, bias=False, separable_conv=False):
    super().__init__()
    conv_layer = SeparableConv2d if separable_conv else nn.Conv2d
    dilation1 = int(2**(2*dilation // 3))
    dilation2 = int(2**((2*dilation+1) // 3))
    self.conv1 = conv_layer(num_channels, num_channels, kernel_size, padding=dilation1, dilation=dilation1, bias=bias)
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.activation1 = activation()
    self.conv2 = conv_layer(num_channels, num_channels, kernel_size, padding=dilation2, dilation=dilation2, bias=bias)
    self.bn2 = nn.BatchNorm2d(num_channels)
    self.activation2 = activation()
  
  def forward(self, X):
    Y = self.bn1(self.activation1(self.conv1(X)))
    Y = self.activation2(self.conv2(Y))
    Y += X
    Y = self.bn2(Y)
    return Y


class SpeechResModel(torch.nn.Module):
    """
    
    Arguments
    ---------
    device : str
        Device used e.g. "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    tdnn_blocks : int
        Number of time-delay neural (TDNN) layers.
    n_channels : list of ints
        Output channels for CNN layer.
    kernel_sizes : list of ints
        List of kernel sizes for each TDNN layer.
    dilations : list of ints
        List of dilations for kernels in each TDNN layer.
    lin_neurons : int
        Number of neurons in linear layers.
    
    Example
    -------
    >>> compute_xvect = Xvector('cpu')
    >>> input_feats = torch.rand([5, 10, 40])
    >>> outputs = compute_xvect(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.LeakyReLU,
        res_blocks=6,
        out_channels=45,
        kernel_sizes=3,
        in_channels=1,
        res_pool=True,
        bias=False,
        separable_conv=False,
        se_block=False,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()
        # first CNN layers
        self.blocks.extend(
             [
                 nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_sizes,
                        padding=1, 
                        bias=bias,
                    ),
                 activation(),
             ]
        )

        # add SE block
        if se_block:
            self.blocks.append(SqueezeExcitationBlock(out_channels))

        # add avg_pooling layer
        if res_pool:
            self.blocks.append(
                nn.AvgPool2d((2, 2))
            )
        # ResNet blocks
        for block_index in range(res_blocks):
            self.blocks.append(ResNetBlock(out_channels, kernel_size=kernel_sizes, dilation=block_index, activation=activation, bias=bias, separable_conv=separable_conv))
        # last CNN layers
        conv_layer = SeparableConv2d if separable_conv else nn.Conv2d
        self.blocks.extend(
            [
                conv_layer(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes,
                    padding=int(2 ** (2 * res_blocks // 3)),
                    dilation=int(2 ** (2 * res_blocks // 3)),
                    bias=bias,
                ),
                nn.BatchNorm2d(out_channels),
                activation(),
            ]
        )
        # avg-pooling
        self.blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        """Returns encoded vectors.

        Arguments
        ---------
        x : torch.Tensor
        """
        b, w, h = x.shape
        x = x.view(b, 1, w, h)
        for layer in self.blocks:
            x = layer(x)
            
        return x.view(b, -1)
                               

class Classifier(torch.nn.Module):
    """This class implements the last MLP on the top of xvector features.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of output neurons.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 40])
    >>> compute_xvect = Xvector()
    >>> xvects = compute_xvect(input_feats)
    >>> classify = Classifier(input_shape=xvects.shape)
    >>> output = classify(xvects)
    >>> output.shape
    torch.Size([5, 1, 1211])
    """

    def __init__(
        self,
        input_dim,
        out_neurons=12,
    ):
        super().__init__()

        # Final output classifier (without softmax)
        self.output = nn.Linear(input_dim, out_neurons)
        
    def forward(self, x):
        """Returns poster probability.

        Arguments
        ---------
        x : torch.Tensor
        """
            
        return self.output(x)
        
