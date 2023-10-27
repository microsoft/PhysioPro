# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

from torch import nn
import torch


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term_single = torch.exp(torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term_single)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PositionEmbedding(nn.Module):
    def __init__(self, emb_type: str, input_size: int, max_len: int = 5000, dropout=0.1):
        """Addictive position embedding on time-series data.

        Args:
            emb_type: The type of embedding.
            Can be learn or static. If learn, the embedding is learned as the model parameter.
            If static, apply the sinusoidal encoding.
            input_size: Dimension of the input.
            max_len: Maximum length of the input.

        Raises:
            ValueError: If emb_type is not learn or static.
        """
        super().__init__()
        self.emb_type = emb_type
        if emb_type == "learn":
            self.emb = nn.Embedding(max_len, input_size)
        elif emb_type == "static":
            self.emb = PositionalEncoding(input_size, max_len=max_len, dropout=dropout)
        else:
            raise ValueError("Unknown embedding type: {}".format(emb_type))

    def forward(self, x):
        if self.emb_type in ["", "learn"]:
            embedding = self.emb(torch.arange(end=x.size()[1], device=x.device))
            embedding = embedding.repeat([x.size()[0], 1, 1])
            x = x + embedding
        elif self.emb_type == "static":
            x = self.emb(x.transpose(0, 1)).transpose(0, 1)

        return x
