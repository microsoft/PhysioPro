from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from .tsrnn import NETWORKS

from .utils import Block,get_3d_sincos_pos_embed,get_1d_sincos_pos_embed_fromlength,get_sph_sincos_pos_embed,get_2d_sincos_pos_embed


@NETWORKS.register_module()
class MMM_Encoder(nn.Module):
    def __init__(
        self, 
        in_chans: int = 5,
        channel_num: int = 79,
        encoder_dim: int = 16,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio : float = 4.,
        attn_mask: Optional[torch.Tensor] = None,
        pe_type: Optional[str] = None,
        pe_coordination: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.channel_num = channel_num
        self.embed_dim = encoder_dim
        self.in_chans = in_chans
        self.pe_type = pe_type
        self.pe_coordination = pe_coordination
        self.attn_mask = nn.parameter.Parameter(attn_mask,requires_grad=False) 
        self.patch_embed = nn.Linear(in_chans, encoder_dim, bias=True) 
        self.pos_embed = nn.Parameter(torch.zeros(self.channel_num + 1, encoder_dim), requires_grad=True)

        norm_layer = nn.LayerNorm
        self.cls_token = nn.parameter.Parameter(torch.zeros(1,1,encoder_dim))

        torch.nn.init.normal_(self.cls_token)

        self.blocks = nn.ModuleList(
            Block(encoder_dim,num_heads,mlp_ratio=mlp_ratio,qkv_bias=True,
                norm_layer=norm_layer) for _ in range(depth))

        self.norm = norm_layer(encoder_dim)
        self.fc_norm = norm_layer(encoder_dim) 
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):

        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.pe_type == '3d':
            pos_embed = get_3d_sincos_pos_embed(self.embed_dim,self.pe_coordination, cls_token=True)
        if self.pe_type == '2d':
            pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.pe_coordination, cls_token=True)
        elif self.pe_type == 'sph':
            pos_embed = get_sph_sincos_pos_embed(self.embed_dim,self.pe_coordination, cls_token=True)
        elif self.pe_type == '1d':
            pos_embed = get_1d_sincos_pos_embed_fromlength(self.embed_dim, self.channel_num, cls_token=True)
        else:
            pos_embed = get_1d_sincos_pos_embed_fromlength(self.embed_dim, 62, cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x:torch.tensor,head=True,region=17):
        x = self.patch_embed(x)
        x=x+self.pos_embed[:-1,:]
        cls=torch.repeat_interleave(self.cls_token+self.pos_embed[-1:,:],x.shape[0],0)
        x=torch.cat([x,cls],dim=1)

        # apply blocks
        for blk in self.blocks:
            x = blk(x,self.attn_mask)
        if head:
            return x[:,-(region + 1):]
        return x[:,:-1,:]

