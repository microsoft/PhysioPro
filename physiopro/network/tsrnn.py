# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from pathlib import Path

import torch
from torch import nn
from utilsd.config import Registry

from ..module import PositionEmbedding, get_cell


class NETWORKS(metaclass=Registry, name="network"):
    pass


@NETWORKS.register_module()
class TSRNN(nn.Module):
    def __init__(
            self,
            cell_type: str,
            emb_dim: int,
            emb_type: str,
            hidden_size: int,
            dropout: float,
            num_layers: int = 1,
            is_bidir: bool = False,
            max_length: Optional[int] = 100,
            input_size: Optional[int] = None,
            weight_file: Optional[Path] = None,
        ):
        """The RNN network for time-series prediction.

        Args:
            cell_type: RNN cell type, e.g. "lstm", "gru", "rnn".
            emb_dim: embedding dimension.
            emb_type: "static" or "learn", static or learnable embedding.
            hidden_size: hidden size of the RNN cell.
            dropout: Dropout rate.
            num_layers: Number of layers of the RNN cell.
            is_bidir: Whether to use bidirectional RNN.
            max_length: Maximum length of the input sequence.
            input_size: Input size of the time-series data.
            weight_file: Path to the pretrained model.

        Raises:
            ValueError: If `cell_type` is not supported.
            ValueError: If `emb_type` is not supported.
        """
        super().__init__()
        Cell = get_cell(cell_type)
        if input_size is not None:
            self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        else:
            self.encoder = None

        self.temporal_encoder = Cell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=is_bidir,
            dropout=dropout,
        )

        if emb_dim != 0:
            self.emb = PositionEmbedding(emb_type, input_size, max_length, dropout=dropout)

        self.emb_dim = emb_dim
        self.__output_size = hidden_size * 2 if is_bidir else hidden_size
        self.__hidden_size = hidden_size

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    @property
    def output_size(self):
        return self.__output_size

    @property
    def hidden_size(self):
        return self.__hidden_size

    def forward(self, inputs):
        # positional encoding
        if self.emb_dim > 0:
            inputs = self.emb(inputs)

        # non-regressive encoder
        if self.encoder is not None:
            z = self.encoder(inputs)
        else:
            z = inputs

        # regressive encoder
        rnn_outs, _ = self.temporal_encoder(z)

        return rnn_outs, rnn_outs[:, -1, :]
