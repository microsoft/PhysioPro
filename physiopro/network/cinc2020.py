# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .tsrnn import NETWORKS
from ..module.positional_encoding import PositionalEncoding


@NETWORKS.register_module("DeepAndWide")
class DeepAndWide(nn.Module):
    def __init__(
            self,
            d_model: int = 256,   # embedding size
            nhead: int = 8,       # number of heads
            d_ff: int = 2048,     # feed forward layer size
            num_layers: int = 8,  # number of encoding layers
            dropout_rate: float = 0.2,
            deepfeat_sz: int = 64,
            nb_feats: int = 20,
            nb_demo: int = 2,
            num_class: int = 27,
            weight_file: Optional[Path] = None,
            partial_feature: Optional[str] = None,
            input_size: int = 0,
            max_length: int = 0,
    ):
        super().__init__()

        self.encoder = nn.Sequential(  # downsampling factor = 20
            nn.Conv1d(12, 128, kernel_size=14, stride=3, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        self.transformer = Transformer(d_model, nhead, d_ff, num_layers, dropout=0.1)
        self.fc1 = nn.Linear(d_model, deepfeat_sz)
        self.dropout = nn.Dropout(dropout_rate)
        self.partial_feature = partial_feature
        if partial_feature == 'deep':
            self.output_size = deepfeat_sz
        elif partial_feature == 'wide':
            self.output_size = nb_feats+nb_demo
        elif not partial_feature:
            self.output_size = deepfeat_sz+nb_feats+nb_demo
        else:
            raise ValueError
        print(f'ecg feature: {self.partial_feature}, dimension: {self.output_size}')

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def forward(self, data):
        x, wide_feats = data
        x, wide_feats = x.transpose(1, 0)[0].float(), wide_feats.squeeze(1).float()
        z = self.encoder(x)          # encoded sequence is batch_sz x nb_ch x seq_len, [8, 12, 7500] -> [8, 256, 183]
        out = self.transformer(z)    # transformer output is batch_sz x d_model
        out = self.dropout(F.relu(self.fc1(out)))
        # out = self.fc2(torch.cat([wide_feats, out], dim=1))
        if self.partial_feature == 'deep':
            return out, out
        if self.partial_feature == 'wide':
            return wide_feats, wide_feats
        if not self.partial_feature:
            return out, torch.cat([wide_feats, out], dim=1)
        raise ValueError


class Transformer(nn.Module):
    '''
    Transformer encoder processes convolved ECG samples
    Stacks a number of TransformerEncoderLayers
    '''

    def __init__(self, d_model, h, d_ff, num_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe = PositionalEncoding(d_model, dropout=0.1)

        encode_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.h,
            dim_feedforward=self.d_ff,
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encode_layer, self.num_layers)

    def forward(self, x):  # [8, 256, 183]
        out = x.permute(0, 2, 1)  # [8, 183, 256]
        out = self.pe(out)  # [8, 183, 256]
        out = out.permute(1, 0, 2)  # [183, 8, 256]
        out = self.transformer_encoder(out)  # [183, 8, 256]
        out = out.mean(0)  # global pooling [8, 256]
        return out
