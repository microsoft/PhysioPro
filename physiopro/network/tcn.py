# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Optional
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import weight_norm

from .tsrnn import NETWORKS
from ..module import PositionEmbedding, Chomp1d


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


@NETWORKS.register_module("TCN")
class TemporalConvNet(nn.Module):
    def __init__(
            self,
            num_channels: List[int],
            position_embedding: bool,
            emb_type: str,
            kernel_size: int = 2,
            dropout: float = 0.2,
            max_length: Optional[int] = 100,
            input_size: Optional[int] = None,
            weight_file: Optional[Path] = None,
    ):
        """The implementation of TCN described in https://arxiv.org/abs/1803.01271.

        Args:
            num_channels: The number of convolutional channels in each layer.
            position_embedding: Whether to use position embedding.
            emb_type: The type of position embedding to use. Can be "learn" or "static".
            kernel_size: The kernel size of convolutional layers.
            dropout: Dropout rate.
        """
        super().__init__(self)
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.__output_size = num_channels[-1]
        self.__hidden_size = num_channels[-1]
        self._position_embedding = position_embedding

        if position_embedding:
            self.emb = PositionEmbedding(emb_type, input_size, max_length, dropout=dropout)

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, inputs):
        if self._position_embedding:
            inputs = self.emb(inputs)
        hiddens = self.network(inputs.transpose(1, 2)).transpose(1, 2)
        return hiddens, hiddens[:, -1, :]

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size


@NETWORKS.register_module("tcn_layer")
class TemporalConvNetLayer(nn.Module):
    def __init__(
            self,
            # num_channels: List[int],
            num_levels: int,
            channel: int,
            dilation: int,
            position_embedding: bool,
            emb_type: str,
            kernel_size: int = 2,
            dropout: float = 0.2,
            max_length: int = 100,
            input_size: Optional[int] = None,
            weight_file: Optional[Path] = None,
        ):
        """The implementation of TCN described in https://arxiv.org/abs/1803.01271.

        Args:
            num_channels: The number of convolutional channels in each layer.
            position_embedding: Whether to use position embedding.
            emb_type: The type of position embedding to use. Can be "learn" or "static".
            kernel_size: The kernel size of convolutional layers.
            dropout: Dropout rate.
        """
        super().__init__(self)
        layers = []
        num_channels = [channel] * num_levels
        if input_size is None:
            input_size = num_channels
        for i in range(num_levels):
            dilation_size = dilation**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.__output_size = num_channels[-1]
        self.__hidden_size = num_channels[-1]
        self._position_embedding = position_embedding

        if position_embedding:
            self.emb = PositionEmbedding(emb_type, input_size, max_length, dropout=dropout)

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, inputs: torch.Tensor):
        if self._position_embedding:
            inputs = self.emb(inputs)
        hiddens = self.network(inputs.transpose(1, 2)).transpose(1, 2)  # batch, channel, dim, T
        return hiddens, hiddens[:, -1, :]

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size
