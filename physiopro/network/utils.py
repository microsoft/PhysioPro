import torch
from torch import nn
import numpy as np


def get_1d_sincos_pos_embed_fromlength(embed_dim, patch_length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = np.arange(patch_length, dtype=float)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    if cls_token:
        pos_embed = np.concatenate([emb,np.zeros([1, embed_dim])], axis=0)
    else:
        pos_embed = emb
    return pos_embed

def get_sph_sincos_pos_embed(embed_dim, sph_coordination, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    sph_coordination = sph_coordination.reshape(2,-1)

    sph_the = sph_coordination[0]
    sph_phi = sph_coordination[1]
    # use half of dimensions to encode sph_theta
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, sph_the)  # (channel_number, D/2)
    # use half of dimensions to encode sph_phi
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, sph_phi)  # (channel_number, D/2)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1) # (channel_number, D)

    if cls_token:
        pos_embed = np.concatenate([pos_embed,np.zeros([1, embed_dim])], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed(embed_dim, coordination, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    if embed_dim == 18:
        # coordination = coordination.reshape(3,1,-1)
        emb_x = get_1d_sincos_pos_embed(embed_dim // 3, coordination[0])  # (channel_number, D/2)
        emb_y = get_1d_sincos_pos_embed(embed_dim // 3, coordination[1])  # (channel_number, D/2)
        emb_z = get_1d_sincos_pos_embed(embed_dim // 3, coordination[2])  # (channel_number, D/2)
    elif embed_dim == 16:
        emb_x = get_1d_sincos_pos_embed(6, coordination[0])  # (channel_number, D/2)
        emb_y = get_1d_sincos_pos_embed(6, coordination[1])  # (channel_number, D/2)
        emb_z = get_1d_sincos_pos_embed(4, coordination[2])  # (channel_number, D/2)


    pos_embed = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (channel_number, D)
    if cls_token:
        pos_embed = np.concatenate([pos_embed, np.zeros([1, embed_dim])], axis=0)
    return pos_embed

def get_4d_sincos_pos_embed(embed_dim, coordination, t_seq, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    # coordination = coordination.reshape(3,1,-1)
    emb_x = get_1d_sincos_pos_embed(embed_dim // 4, coordination[0])  # (channel_number, D/2)
    emb_y = get_1d_sincos_pos_embed(embed_dim // 4, coordination[1])  # (channel_number, D/2)
    emb_z = get_1d_sincos_pos_embed(embed_dim // 4, coordination[2])  # (channel_number, D/2)
    emb_t = get_1d_sincos_pos_embed(embed_dim // 4, t_seq)  # (channel_number, D/2)
    pos_embed = np.concatenate([emb_x, emb_y, emb_z,emb_t], axis=1) # (channel_number, D)
    if cls_token:
        pos_embed = np.concatenate([pos_embed, np.zeros([1, embed_dim])], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed(embed_dim, coordination, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 2 == 0

    emb_x = get_1d_sincos_pos_embed(embed_dim // 2, coordination[0])  # (channel_number, D/2)
    emb_y = get_1d_sincos_pos_embed(embed_dim // 2, coordination[1])  # (channel_number, D/2)

    pos_embed = np.concatenate([emb_x, emb_y], axis=1) # (channel_number, D)
    if cls_token:
        pos_embed = np.concatenate([pos_embed, np.zeros([1, embed_dim])], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class tcnformer_Block(nn.Module):
    def __init__(
        self,
        spectral_dim,
        spatial_dim=18,
        num_heads=6,
        mlp_ratio=4.0,
        drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.spectral_dim = spectral_dim
        self.spatial_dim = spatial_dim
        self.tcn_attn_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv1d(
                            spectral_dim,
                            spatial_dim,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=kernel_size - 1,
                            dilation=1,
                        ),
                        nn.MultiheadAttention(
                            spatial_dim,
                            num_heads=num_heads,
                            batch_first=True,
                        ),
                        nn.Linear(spatial_dim, spectral_dim),
                        nn.BatchNorm1d(spatial_dim),
                    ]
                )
                for kernel_size in [2,3,5]
            ]
        )

        self.act = nn.ReLU()
        self.norm1 = norm_layer(spectral_dim)
        self.norm2 = norm_layer(spectral_dim)
        mlp_hidden_dim = int(spectral_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=spectral_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x,attn_mask=None):
        N, C, T, D = x.shape
        result = torch.zeros_like(x)
        result = result + x
        for tcn, attn, spatial_proj, bn in self.tcn_attn_blocks:
            out = x.reshape(N * C, T, D).permute(0, 2, 1)
            out = self.act(bn(tcn(out)[:, :, :+T]))
            # TCN TODO!
            _, F, T = out.shape
            out = out.reshape(N, C, F, T).permute(0, 3, 1, 2)
            out = out.reshape(N * T, C, F)
            out, _ = attn(out,out,out,attn_mask=attn_mask)
            # out = nn.LayerNorm()(out) TODO!
            out = spatial_proj(out)
            out = out.reshape(N, T, C, D).permute(0, 2, 1, 3)
            result = result + out
        x = result
        x = self.act(self.norm1(x))
        x = self.act(self.norm2(self.mlp(x) + x))

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
                            dim,
                            num_heads=num_heads,
                            batch_first=True,
                        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,attn_mask=None):
        x = self.norm1(x)
        x = x + self.drop_path(self.attn(x,x,x,attn_mask=attn_mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
