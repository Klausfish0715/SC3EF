"""
SC3EF for Optical Flow Estimation.

This module integrates self-correlation extraction and cross-correspondence estimation,
hierarchical optical flow generation.
"""

from functools import partial
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from .convnext_isotropic import convnext_isotropic_small_2dec_v3SE
from .matching import global_correlation_softmax


# nec
def conv_blck(in_channels,
              out_channels,
              kernel_size=3,
              stride=1,
              padding=1,
              dilation=1,
              bn=False):
    """Standard convolutional block with optional BatchNorm and ReLU."""
    if bn:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation), nn.ReLU(inplace=True))

# nec
def conv_head(in_channels):
    """Output head to predict 2D flow maps."""
    return nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)


class IRB(nn.Module):
    """Inverted Residual Block with depthwise convolution and Hardswish activation."""
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 ksize=3,
                 act_layer=nn.Hardswish,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features,
                              hidden_features,
                              kernel_size=ksize,
                              padding=ksize // 2,
                              stride=1,
                              groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.reshape(B, C, -1).permute(0, 2, 1)


class Pooling_cross_Attention(nn.Module):
    """Cross-attention module with pyramid pooling for cross-modal feature interaction."""
    def __init__(self,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 pool_ratios=[1, 2, 3, 6]):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y, H, W, d_convs=None):
        B, N, C = x.shape

        q_x = self.q(x).reshape(B, N, self.num_heads,
                                C // self.num_heads).permute(0, 2, 1, 3)
        pools_x = []
        pools_y = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        y_ = y.permute(0, 2, 1).reshape(B, C, H, W)

        # pooling -> x
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            pool_x = F.adaptive_avg_pool2d(
                x_, (round(H / pool_ratio), round(W / pool_ratio)))
            pool_x = pool_x + l(
                pool_x
            )  # fix backward bug in higher torch versions when training
            pools_x.append(pool_x.view(B, C, -1))

        pools_x = torch.cat(pools_x, dim=2)
        pools_x = self.norm(pools_x.permute(0, 2, 1))

        kv_x = self.kv(pools_x).reshape(B, -1, 2, self.num_heads,
                                        C // self.num_heads).permute(
                                            2, 0, 3, 1, 4)
        k_x, v_x = kv_x[0], kv_x[1]

        # pooling -> y
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            pool_y = F.adaptive_avg_pool2d(
                y_, (round(H / pool_ratio), round(W / pool_ratio)))
            pool_y = pool_y + l(
                pool_y
            )  # fix backward bug in higher torch versions when training
            pools_y.append(pool_y.view(B, C, -1))

        pools_y = torch.cat(pools_y, dim=2)
        pools_y = self.norm(pools_y.permute(0, 2, 1))

        kv_y = self.kv(pools_y).reshape(B, -1, 2, self.num_heads,
                                        C // self.num_heads).permute(
                                            2, 0, 3, 1, 4)
        k_y, v_y = kv_y[0], kv_y[1]

        attn = (q_x @ k_y.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v_y)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)

        return x


class Pooling_self_Attention(nn.Module):
    """Self-attention module with pyramid pooling for self-modal feature learning."""
    def __init__(self,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 pool_ratios=[1, 2, 3, 6]):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, d_convs=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)
        # print('q', q.shape)
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            pool = F.adaptive_avg_pool2d(
                x_, (round(H / pool_ratio), round(W / pool_ratio)))
            pool = pool + l(
                pool
            )  # fix backward bug in higher torch versions when training
            pools.append(pool.view(B, C, -1))

        # print(len(pools), pools[0].shape, pools[1].shape, pools[2].shape,
        #       pools[3].shape)
        pools = torch.cat(pools, dim=2)
        # print(pools.shape)
        pools = self.norm(pools.permute(0, 2, 1))
        # print(pools.shape)
        # print(self.kv(pools).shape)

        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        # print('kv', k.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print('attn', attn.shape)

        attn = attn.softmax(dim=-1)
        # print('attn', attn.shape)
        x = (attn @ v)
        # print(x.shape)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        # print(x.shape)

        x = self.proj(x)
        # print(x.shape)
        # exit()

        return x

# nec
class Block_self(nn.Module):
    """Transformer block with self-attention and IRB."""
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 pool_ratios=[12, 16, 20, 24]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Pooling_self_Attention(dim,
                                           num_heads=num_heads,
                                           qkv_bias=qkv_bias,
                                           qk_scale=qk_scale,
                                           attn_drop=attn_drop,
                                           proj_drop=drop,
                                           pool_ratios=pool_ratios)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.Hardswish,
                       drop=drop,
                       ksize=3)

    def forward(self, x, H, W, d_convs=None):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, d_convs=d_convs))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class Block_cross(nn.Module):
    """Transformer block with cross-attention and IRB for dual-branch interaction."""
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 pool_ratios=[12, 16, 20, 24]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Pooling_cross_Attention(dim,
                                            num_heads=num_heads,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            attn_drop=attn_drop,
                                            proj_drop=drop,
                                            pool_ratios=pool_ratios)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.Hardswish,
                       drop=drop,
                       ksize=3)

    def forward(self, x, y, H, W, d_convs=None):
        x = x + self.drop_path(
            self.attn(self.norm1(x), y, H, W, d_convs=d_convs))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        y = y + self.drop_path(
            self.attn(self.norm1(y), x, H, W, d_convs=d_convs))
        y = y + self.drop_path(self.mlp(self.norm2(y), H, W))

        return x, y

# nec
class PatchEmbed(nn.Module):
    """(Overlapping) Patch embedding layer for image tokens."""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 kernel_size=3,
                 in_chans=3,
                 embed_dim=768,
                 overlap=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        if not overlap:
            self.proj = nn.Conv2d(in_chans,
                                  embed_dim,
                                  kernel_size=patch_size,
                                  stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans,
                                  embed_dim,
                                  kernel_size=kernel_size,
                                  stride=patch_size,
                                  padding=kernel_size // 2)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

# nec
class SC3EF(nn.Module):
    """
    Main SC3EF architecture.

    Args:
        img_size (int): Input image size.
        patch_size (int): Patch size for initial embedding.
        embed_dims (list): Embedding dimensions at each stage.
        num_heads (list): Number of attention heads at each stage.
        mlp_ratios (list): MLP expansion ratios.
        ...
    """
    def __init__(self,
                 img_size=256,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[48, 96, 240, 384],
                 num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 6, 3]):  #
        super().__init__()
        
         # Initialization, build patch embeds, blocks, convolutions etc. 
        self.num_classes = num_classes
        self.depths = depths

        self.embed_dims = embed_dims

        # pyramid pooling ratios for each stage
        pool_ratios = [[12, 16, 20, 24], [6, 8, 10, 12], [3, 4, 5, 6],
                       [1, 2, 3, 4]]

        self.patch_embed1 = PatchEmbed(img_size=img_size,
                                       patch_size=4,
                                       kernel_size=7,
                                       in_chans=in_chans,
                                       embed_dim=embed_dims[0],
                                       overlap=True)

        self.conv_bf_embed1 = conv_blck(2 * embed_dims[0],
                                        embed_dims[0],
                                        bn=True)

        self.patch_embed2 = PatchEmbed(img_size=img_size // 4,
                                       patch_size=2,
                                       in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1],
                                       overlap=True)

        self.conv_bf_embed2_hf = conv_blck(2 * embed_dims[1],
                                           embed_dims[1],
                                           bn=True)
        self.conv_bf_embed2_lf = conv_blck(2 * embed_dims[1],
                                           embed_dims[1],
                                           bn=True)

        self.conv_bf_embed3 = conv_blck(2 * embed_dims[2],
                                        embed_dims[2],
                                        bn=True)

        self.conv_bf_embed4 = conv_blck(2 * embed_dims[3],
                                        embed_dims[3],
                                        bn=True)

        self.patch_embed3_lf = PatchEmbed(img_size=img_size // 8,
                                          patch_size=2,
                                          in_chans=embed_dims[1],
                                          embed_dim=embed_dims[2],
                                          overlap=True)

        self.patch_embed4_lf = PatchEmbed(img_size=img_size // 16,
                                          patch_size=2,
                                          in_chans=embed_dims[2],
                                          embed_dim=embed_dims[3],
                                          overlap=True)

        self.d_convs1 = nn.ModuleList([
            nn.Conv2d(embed_dims[0],
                      embed_dims[0],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=embed_dims[0]) for temp in pool_ratios[0]
        ])
        self.d_convs2 = nn.ModuleList([
            nn.Conv2d(embed_dims[1],
                      embed_dims[1],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=embed_dims[1]) for temp in pool_ratios[1]
        ])
        self.d_convs3_lf = nn.ModuleList([
            nn.Conv2d(embed_dims[2],
                      embed_dims[2],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=embed_dims[2]) for temp in pool_ratios[2]
        ])

        self.d_convs4_lf = nn.ModuleList([
            nn.Conv2d(embed_dims[3],
                      embed_dims[3],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=embed_dims[3]) for temp in pool_ratios[3]
        ])

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        ksize = 3

        self.block1 = nn.ModuleList([
            Block_self(dim=embed_dims[0],
                       num_heads=num_heads[0],
                       mlp_ratio=mlp_ratios[0],
                       qkv_bias=qkv_bias,
                       qk_scale=qk_scale,
                       drop=drop_rate,
                       attn_drop=attn_drop_rate,
                       drop_path=dpr[cur + i],
                       norm_layer=norm_layer,
                       pool_ratios=pool_ratios[0]) for i in range(depths[0])
        ])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block_self(dim=embed_dims[1],
                       num_heads=num_heads[1],
                       mlp_ratio=mlp_ratios[1],
                       qkv_bias=qkv_bias,
                       qk_scale=qk_scale,
                       drop=drop_rate,
                       attn_drop=attn_drop_rate,
                       drop_path=dpr[cur + i],
                       norm_layer=norm_layer,
                       pool_ratios=pool_ratios[1]) for i in range(depths[1])
        ])

        cur += depths[1]

        self.block3_lf = nn.ModuleList([
            Block_cross(dim=embed_dims[2],
                        num_heads=num_heads[2],
                        mlp_ratio=mlp_ratios[2],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + i],
                        norm_layer=norm_layer,
                        pool_ratios=pool_ratios[2]) for i in range(depths[2])
        ])

        cur += depths[2]

        self.block4_lf = nn.ModuleList([
            Block_cross(dim=embed_dims[3],
                        num_heads=num_heads[3],
                        mlp_ratio=mlp_ratios[3],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + i],
                        norm_layer=norm_layer,
                        pool_ratios=pool_ratios[3]) for i in range(depths[3])
        ])

        # high-freq feature extraction
        self.hf_conv = convnext_isotropic_small_2dec_v3SE(
            dims_list=[48, 96, 240, 384])

        self.hf_conv3 = conv_blck(embed_dims[1],
                                  embed_dims[2],
                                  3,
                                  2,
                                  1,
                                  bn=True)
        self.hf_conv4 = conv_blck(embed_dims[2],
                                  embed_dims[3],
                                  3,
                                  2,
                                  1,
                                  bn=True)

        # fow estimation
        self.pred_conv_conv1 = conv_blck(2, 2, bn=True)

        self.conv16_0 = conv_blck(2, 64, bn=True)
        # self.conv16_1 = conv_blck(128, 96, padding=2, dilation=2, bn=True)
        # self.conv16_2 = conv_blck(96, 64, padding=3, dilation=3, bn=True)
        self.conv16_1 = conv_blck(64, 32, padding=4, dilation=4, bn=True)
        self.final_16 = conv_head(32)

        self.conv8_0 = conv_blck(2, 64, bn=True)
        # self.conv8_1 = conv_blck(128, 96, padding=2, dilation=2, bn=True)
        # self.conv8_2 = conv_blck(96, 64, padding=3, dilation=3, bn=True)
        self.conv8_1 = conv_blck(64, 32, padding=4, dilation=4, bn=True)
        self.final_8 = conv_head(32)

        self.conv4_0 = conv_blck(2, 64, bn=True)
        # self.conv4_1 = conv_blck(128, 96, padding=2, dilation=2, bn=True)
        # self.conv4_2 = conv_blck(96, 64, padding=3, dilation=3, bn=True)
        self.conv4_1 = conv_blck(64, 32, padding=4, dilation=4, bn=True)
        self.final_4 = conv_head(32)

        self.conv2_0 = conv_blck(2, 64, bn=True)
        # self.conv2_1 = conv_blck(128, 96, padding=2, dilation=2, bn=True)
        # self.conv2_2 = conv_blck(96, 64, padding=3, dilation=3, bn=True)
        self.conv2_1 = conv_blck(64, 32, padding=4, dilation=4, bn=True)
        self.final_2 = conv_head(32)

        self.conv1_0 = conv_blck(2, 64, bn=True)
        # self.conv1_1 = conv_blck(128, 96, padding=2, dilation=2, bn=True)
        # self.conv1_2 = conv_blck(96, 64, padding=3, dilation=3, bn=True)
        self.conv1_1 = conv_blck(64, 32, padding=4, dilation=4, bn=True)
        self.final_1 = conv_head(32)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def upsample_flow(
        self,
        flow,
        feature,
        bilinear=False,
        upsample_factor=8,
    ):
        if bilinear:
            up_flow = F.interpolate(flow,
                                    scale_factor=upsample_factor,
                                    mode='bilinear',
                                    align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor,
                             self.upsample_factor, h,
                             w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h,
                                   w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(
                b, flow_channel, self.upsample_factor * h,
                self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    def forward_features(self, x0, x1, y0, y1):
        """
        Forward feature extraction and decoding.

        Args:
            x0, y0: Low-frequency inputs.
            x1, y1: High-frequency inputs.

        Returns:
            dict: Dictionary containing multiscale flow predictions.
        """

        ## x0, y0 -> low-freq; x1, y1 -> high freq
        B = x0.shape[0]

        #### 1. self-modal ####
        # 1.1 stage 1
        # 1.1.2 high-freq
        x1_1, x1_2, x1_3, x1_4 = self.hf_conv(x1, y1)
        y1_1, y1_2, y1_3, y1_4 = self.hf_conv(y1, x1)
        
        # 1.1.1 low-freq
        x0, (H, W) = self.patch_embed1(x0)
        for idx, blk in enumerate(self.block1):
            x0 = blk(x0, H, W, self.d_convs1)
        x0 = x0.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x0 = torch.cat((x1_1, x0), 1)
        x0 = self.conv_bf_embed1(x0)

        y0, (H, W) = self.patch_embed1(y0)
        for idx, blk in enumerate(self.block1):
            y0 = blk(y0, H, W, self.d_convs1)
        y0 = y0.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        y0 = torch.cat((y1_1, y0), 1)
        y0 = self.conv_bf_embed1(y0)

        # stage 2
        x0, (H, W) = self.patch_embed2(x0)
        for idx, blk in enumerate(self.block2):
            x0 = blk(x0, H, W, self.d_convs2)
        x0 = x0.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x1 = torch.cat((x1_2, x0), 1)
        x1 = self.conv_bf_embed2_hf(x1)

        x0 = torch.cat((x0, x1_2), 1)
        x0 = self.conv_bf_embed2_lf(x0)

        y0, (H, W) = self.patch_embed2(y0)
        for idx, blk in enumerate(self.block2):
            y0 = blk(y0, H, W, self.d_convs2)
        y0 = y0.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        y1 = torch.cat((y1_2, y0), 1)
        y1 = self.conv_bf_embed2_hf(y1)

        y0 = torch.cat((y0, y1_2), 1)
        y0 = self.conv_bf_embed2_lf(y0)

        #### 2. cross-modal ####
        # stage 3
        # LF Decoder
        x0, (H, W) = self.patch_embed3_lf(x0)
        y0, (H, W) = self.patch_embed3_lf(y0)

        for idx, blk in enumerate(self.block3_lf):
            x0, y0 = blk(x0, y0, H, W, self.d_convs3_lf)
        x0 = x0.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        y0 = y0.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # HF Decoder
        x0 = torch.cat((x1_3, x0), 1)
        x0 = self.conv_bf_embed3(x0)

        # stage 4
        # LF Decoder
        x0, (H, W) = self.patch_embed4_lf(x0)
        y0, (H, W) = self.patch_embed4_lf(y0)

        for idx, blk in enumerate(self.block4_lf):
            x0, y0 = blk(x0, y0, H, W, self.d_convs4_lf)
        x0 = x0.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        y0 = y0.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # HF Decoder
        x0 = torch.cat((x1_4, x0), 1)
        x0 = self.conv_bf_embed4(x0)


        #### 3. hierarchical flow estimation ###
        # global corr
        results_dict = {}
        flow_preds = []

        flow_pred0 = global_correlation_softmax(x0, y0)[0]
        flow_pred1 = global_correlation_softmax(x1_4, y1_4)[0]
        flow_pred = flow_pred0 + 0.001 * flow_pred1

        flow_32 = self.pred_conv_conv1(flow_pred)
        flow_preds.append(flow_32)

        flow_up_16 = self.upsample_flow(flow_32,
                                        None,
                                        bilinear=True,
                                        upsample_factor=2)

        flow_16 = self.conv16_1(self.conv16_0(flow_up_16))
        flow_16 = self.final_16(flow_16)

        flow_preds.append(flow_16)

        flow_up_8 = self.upsample_flow(flow_16,
                                       None,
                                       bilinear=True,
                                       upsample_factor=2)
        flow_8 = self.conv8_1(self.conv8_0(flow_up_8))
        flow_8 = self.final_8(flow_8)

        flow_preds.append(flow_8)

        flow_up_4 = self.upsample_flow(flow_8,
                                       None,
                                       bilinear=True,
                                       upsample_factor=2)
        flow_4 = self.conv4_1(self.conv4_0(flow_up_4))
        flow_4 = self.final_4(flow_4)

        flow_preds.append(flow_4)

        flow_up_2 = self.upsample_flow(flow_4,
                                       None,
                                       bilinear=True,
                                       upsample_factor=2)
        flow_2 = self.conv2_1(self.conv2_0(flow_up_2))
        flow_2 = self.final_2(flow_2)

        flow_preds.append(flow_2)

        flow_up_1 = self.upsample_flow(flow_2,
                                       None,
                                       bilinear=True,
                                       upsample_factor=2)
        flow_1 = self.conv1_1(self.conv1_0(flow_up_1))
        flow_1 = self.final_1(flow_1)

        # output
        flow_preds.append(flow_1)
        results_dict.update({'flow_preds': flow_preds})

        return results_dict

    def forward(self, x0, x1, y0, y1):
        """
        Full forward pass.

        Args:
            x0, x1, y0, y1: Input image tensors (low and high frequency).

        Returns:
            dict: Output dictionary with flow predictions.
        """
        
        results_dict = self.forward_features(x0, x1, y0, y1)
        return results_dict
    
