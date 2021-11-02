import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from squeeze_and_excitation import SqueezeExcitationBlock
from positional_encoding import PositionalEncoding


class TcResNetBlock(nn.Module):
  
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=9, activation=torch.nn.ReLU, bias=False):
        super().__init__()
        if stride == 1:
            assert in_channels == out_channels
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, stride=stride,
            kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation1 = activation()

        if stride != 1:
            self.conv_res = nn.Conv1d(in_channels, out_channels, stride=stride,
            kernel_size=1, padding=0, bias=bias)
            self.bn_res = nn.BatchNorm1d(out_channels)
            self.activation_res = activation()

        self.conv2 = nn.Conv1d(out_channels, out_channels, stride=1,
            kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation2 = activation()

    def forward(self, X):
        Y = self.activation1(self.bn1(self.conv1(X)))
        if hasattr(self, 'conv_res'):
            X = self.activation_res(self.bn_res(self.conv_res(X)))
        Y = self.bn2(self.conv2(Y))
        return self.activation2(X + Y)


class TemporallyPooledAttention(nn.Module):

    def __init__(self, n_heads, n_feat, dropout_rate=0):
        super().__init__()
        assert n_feat % n_heads == 0
        self.d_k = n_feat // n_heads
        self.h = n_heads
        self.linear_qkv = nn.Linear(n_feat, n_feat, bias=False)
        # self.linear_out = nn.Linear(n_feat, n_feat, bias=False)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        # positional encoding layer
        self.posenc = PositionalEncoding(d_model=n_feat)

    def forward(self, x):

        x = self.posenc(x)
        b, c, t = x.shape
        x = x.transpose(1, 2)
        query = torch.mean(x, dim=1, keepdim=False)
        q = self.linear_qkv(query).view(b, -1, self.h, self.d_k) # (batch, 1, head, d_k)
        k_v = self.linear_qkv(x).view(b, -1, self.h, self.d_k) # (batch, t, head, d_k)
        q = q.transpose(1, 2)  # (batch, head, 1, d_k)
        k_v = k_v.transpose(1, 2)  # (batch, head, t, d_k)
        att_scores = torch.matmul(q, k_v.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, 1, t)
        self.attn = torch.softmax(att_scores, dim=-1)  # (batch, head, 1, t)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, k_v)  # (batch, head, 1, d_k)
        x = x.view(b, self.h * self.d_k)  # (batch, d_model)

        return x
        # return self.linear_out(x), self.attn  # (batch, time1, d_model)


class SpeechTcResModel(nn.Module):

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.LeakyReLU,
        res_blocks=3,
        out_channels=[16, 24, 32, 48],
        kernel_sizes=[3, 9, 9, 9],
        in_channels=40,
        pool=True,
        bias=False,
        se_block=False,
    ):

        super().__init__()
        assert res_blocks == len(out_channels) - 1

        self.blocks = nn.ModuleList()
        # first CNN layer
        self.blocks.extend(
            [
                nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels[0],
                        kernel_size=kernel_sizes[0],
                        padding=1, 
                        bias=bias,
                    ),
                # nn.BatchNorm1d(out_channels[0]),
                activation(),
                # positional encoding layer
                # PositionalEncoding(d_model=out_channels[0])
            ]
        )

        # add avg_pooling layer
        if pool:
            self.blocks.append(
                nn.AvgPool1d(2)
            )

        # ResNet blocks
        for block_index in range(res_blocks):

            inc = out_channels[block_index]
            ouc = out_channels[block_index + 1]
            stride = 1 if inc == ouc else 2
            self.blocks.extend([
                TcResNetBlock(inc, ouc, stride=stride, 
                kernel_size=kernel_sizes[block_index+1], activation=activation, bias=bias),
            ])

        # # avg-pooling
        # self.blocks.append(nn.AdaptiveAvgPool1d(1))
        # replace avg-pooling with temporally pooled attention
        self.blocks.append(TemporallyPooledAttention(1, out_channels[-1]))


    def forward(self, x):
        """

        Arguments
        ---------
        x : torch.Tensor #(B, T, C)
        """
        b, t, c = x.shape
        x = x.transpose(1, 2)

        for layer in self.blocks:
            x = layer(x)

        return x.view(b, -1)

