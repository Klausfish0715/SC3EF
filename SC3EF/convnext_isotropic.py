# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from mmcv.ops import DeformConv2d
from .convnext import Block, LayerNorm, Selayer


class DeformableConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformableConvLayer, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, deformable_groups * 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

    def forward(self, x):
        offset = self.conv_offset(x)
        x = self.conv(x, offset)
        return x
    
    
class ConvNeXtIsotropic(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depth=18,
        dim=[48, 96],
        drop_path_rate=0.,
        layer_scale_init_value=0,
        head_init_scale=1.,
    ):
        super().__init__()

        self.stem1 = nn.Conv2d(in_chans, dim[0], kernel_size=8, stride=8)
        self.stem2 = nn.Conv2d(in_chans, dim[1], kernel_size=4, stride=4)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks1 = nn.Sequential(*[
            Block(dim=dim[0],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm1 = LayerNorm(dim[0], eps=1e-6)  # final norm layer

        self.blocks2 = nn.Sequential(*[
            Block(dim=dim[1],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm2 = LayerNorm(dim[1], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x2 = self.stem2(x)
        x2 = self.blocks2(x2)
        x1 = self.stem1(x)
        x1 = self.blocks1(x1)

        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = self.norm1(x1)
        x1 = x1.permute(0, 3, 1, 2)

        x2 = x2.permute(0, 2, 3, 1).contiguous()
        x2 = self.norm2(x2)
        x2 = x2.permute(0, 3, 1, 2)

        return x2, x1  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class ConvNeXtIsotropic_2dec(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depth=18,
        dim=384,
        drop_path_rate=0.,
        layer_scale_init_value=0,
        head_init_scale=1.,
    ):
        super().__init__()

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=32, stride=32)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=dim,
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm = LayerNorm(dim, eps=1e-6)  # final norm layer
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class ConvNeXtIsotropic_2dec_v3(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depth=18,
        dim=[48, 96, 240, 384],
        drop_path_rate=0.,
        layer_scale_init_value=0,
        head_init_scale=1.,
    ):
        super().__init__()

        self.stem1 = nn.Conv2d(in_chans, dim[0], kernel_size=4, stride=4)
        self.stem2 = nn.Conv2d(in_chans, dim[1], kernel_size=8, stride=8)
        self.stem3 = nn.Conv2d(in_chans, dim[2], kernel_size=16, stride=16)
        self.stem4 = nn.Conv2d(in_chans, dim[3], kernel_size=32, stride=32)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks1 = nn.Sequential(*[
            Block(dim=dim[0],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm1 = LayerNorm(dim[0], eps=1e-6)  # final norm layer

        self.blocks2 = nn.Sequential(*[
            Block(dim=dim[1],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm2 = LayerNorm(dim[1], eps=1e-6)  # final norm layer

        self.blocks3 = nn.Sequential(*[
            Block(dim=dim[2],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm3 = LayerNorm(dim[2], eps=1e-6)  # final norm layer

        self.blocks4 = nn.Sequential(*[
            Block(dim=dim[3],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm4 = LayerNorm(dim[3], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x1 = self.stem1(x)
        x2 = self.stem2(x)
        x3 = self.stem3(x)
        x4 = self.stem4(x)

        x1 = self.blocks1(x1)
        x2 = self.blocks2(x2)
        x3 = self.blocks3(x3)
        x4 = self.blocks4(x4)

        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = self.norm1(x1)
        x1 = x1.permute(0, 3, 1, 2)

        x2 = x2.permute(0, 2, 3, 1).contiguous()
        x2 = self.norm2(x2)
        x2 = x2.permute(0, 3, 1, 2)

        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.norm3(x3)
        x3 = x3.permute(0, 3, 1, 2)

        x4 = x4.permute(0, 2, 3, 1).contiguous()
        x4 = self.norm4(x4)
        x4 = x4.permute(0, 3, 1, 2)

        return x1, x2, x3, x4  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x1, x2, x3, x4 = self.forward_features(x)
        # x = self.head(x)
        return x1, x2, x3, x4


class ConvNeXtIsotropic_2dec_v3SE(nn.Module):

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depth=18,
        dim=[48, 96, 240, 384],
        drop_path_rate=0.,
        layer_scale_init_value=0,
        head_init_scale=1.,
    ):
        super().__init__()

        self.stem1 = nn.Conv2d(in_chans, dim[0], kernel_size=4, stride=4)
        self.stem2 = nn.Conv2d(in_chans, dim[1], kernel_size=8, stride=8)
        self.stem3 = nn.Conv2d(2 * in_chans, dim[2], kernel_size=16, stride=16)
        self.stem4 = nn.Conv2d(2 * in_chans, dim[3], kernel_size=32, stride=32)
        self.se3 = Selayer(dim[2])
        self.se4 = Selayer(dim[3])

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks1 = nn.Sequential(*[
            Block(dim=dim[0],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm1 = LayerNorm(dim[0], eps=1e-6)  # final norm layer

        self.blocks2 = nn.Sequential(*[
            Block(dim=dim[1],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm2 = LayerNorm(dim[1], eps=1e-6)  # final norm layer

        self.blocks3 = nn.Sequential(*[
            Block(dim=dim[2],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm3 = LayerNorm(dim[2], eps=1e-6)  # final norm layer

        self.blocks4 = nn.Sequential(*[
            Block(dim=dim[3],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm4 = LayerNorm(dim[3], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, y):
        x1 = self.stem1(x)
        x2 = self.stem2(x)
        x = torch.cat((x, y), 1)
        x3 = self.stem3(x)
        x4 = self.stem4(x)

        x1 = self.blocks1(x1)
        x2 = self.blocks2(x2)
        x3 = self.blocks3(x3)
        x3 = self.se3(x3)
        x4 = self.blocks4(x4)
        x4 = self.se4(x4)

        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = self.norm1(x1)
        x1 = x1.permute(0, 3, 1, 2)

        x2 = x2.permute(0, 2, 3, 1).contiguous()
        x2 = self.norm2(x2)
        x2 = x2.permute(0, 3, 1, 2)

        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.norm3(x3)
        x3 = x3.permute(0, 3, 1, 2)

        x4 = x4.permute(0, 2, 3, 1).contiguous()
        x4 = self.norm4(x4)
        x4 = x4.permute(0, 3, 1, 2)

        return x1, x2, x3, x4  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, y):
        x1, x2, x3, x4 = self.forward_features(x, y)
        # x = self.head(x)
        return x1, x2, x3, x4
    

class ConvNeXtIsotropic_2dec_v3SE_DCN(nn.Module):
    
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depth=18,
        dim=[48, 96, 240, 384],
        drop_path_rate=0.,
        layer_scale_init_value=0,
        head_init_scale=1.,
    ):
        super().__init__()

        self.stem1 = DeformableConvLayer(in_chans, dim[0], kernel_size=4, stride=4)
        self.stem2 = DeformableConvLayer(in_chans, dim[1], kernel_size=8, stride=8)
        self.stem3 = DeformableConvLayer(2 * in_chans, dim[2], kernel_size=16, stride=16)
        self.stem4 = DeformableConvLayer(2 * in_chans, dim[3], kernel_size=32, stride=32)
        self.se3 = Selayer(dim[2])
        self.se4 = Selayer(dim[3])

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks1 = nn.Sequential(*[
            Block(dim=dim[0],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm1 = LayerNorm(dim[0], eps=1e-6)  # final norm layer

        self.blocks2 = nn.Sequential(*[
            Block(dim=dim[1],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm2 = LayerNorm(dim[1], eps=1e-6)  # final norm layer

        self.blocks3 = nn.Sequential(*[
            Block(dim=dim[2],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm3 = LayerNorm(dim[2], eps=1e-6)  # final norm layer

        self.blocks4 = nn.Sequential(*[
            Block(dim=dim[3],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm4 = LayerNorm(dim[3], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, y):
        x1 = self.stem1(x)
        x2 = self.stem2(x)
        x = torch.cat((x, y), 1)
        x3 = self.stem3(x)
        x4 = self.stem4(x)

        x1 = self.blocks1(x1)
        x2 = self.blocks2(x2)
        x3 = self.blocks3(x3)
        x3 = self.se3(x3)
        x4 = self.blocks4(x4)
        x4 = self.se4(x4)

        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = self.norm1(x1)
        x1 = x1.permute(0, 3, 1, 2)

        x2 = x2.permute(0, 2, 3, 1).contiguous()
        x2 = self.norm2(x2)
        x2 = x2.permute(0, 3, 1, 2)

        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.norm3(x3)
        x3 = x3.permute(0, 3, 1, 2)

        x4 = x4.permute(0, 2, 3, 1).contiguous()
        x4 = self.norm4(x4)
        x4 = x4.permute(0, 3, 1, 2)

        return x1, x2, x3, x4  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, y):
        x1, x2, x3, x4 = self.forward_features(x, y)
        # x = self.head(x)
        return x1, x2, x3, x4


class ConvNeXtIsotropic_2dec_v4(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depth=18,
        dim=[48, 96, 240, 384],
        drop_path_rate=0.,
        layer_scale_init_value=0,
        head_init_scale=1.,
    ):
        super().__init__()

        self.stem1 = nn.Conv2d(in_chans, dim[0], kernel_size=4, stride=4)
        self.stem2 = nn.Conv2d(dim[0], dim[1], kernel_size=2, stride=2)
        self.stem3 = nn.Conv2d(dim[1], dim[2], kernel_size=2, stride=2)
        self.stem4 = nn.Conv2d(dim[2], dim[3], kernel_size=2, stride=2)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks1 = nn.Sequential(*[
            Block(dim=dim[0],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm1 = LayerNorm(dim[0], eps=1e-6)  # final norm layer

        self.blocks2 = nn.Sequential(*[
            Block(dim=dim[1],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm2 = LayerNorm(dim[1], eps=1e-6)  # final norm layer

        self.blocks3 = nn.Sequential(*[
            Block(dim=dim[2],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm3 = LayerNorm(dim[2], eps=1e-6)  # final norm layer

        self.blocks4 = nn.Sequential(*[
            Block(dim=dim[3],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm4 = LayerNorm(dim[3], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):

        x1 = self.stem1(x)
        x1 = self.blocks1(x1)
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = self.norm1(x1)
        x1 = x1.permute(0, 3, 1, 2)

        x2 = self.stem2(x1)
        x2 = self.blocks2(x2)
        x2 = x2.permute(0, 2, 3, 1).contiguous()
        x2 = self.norm2(x2)
        x2 = x2.permute(0, 3, 1, 2)

        x3 = self.stem3(x2)
        x3 = self.blocks3(x3)
        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.norm3(x3)
        x3 = x3.permute(0, 3, 1, 2)

        x4 = self.stem4(x3)
        x4 = self.blocks4(x4)
        x4 = x4.permute(0, 2, 3, 1).contiguous()
        x4 = self.norm4(x4)
        x4 = x4.permute(0, 3, 1, 2)

        return x1, x2, x3, x4  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x1, x2, x3, x4 = self.forward_features(x)
        # x = self.head(x)
        return x1, x2, x3, x4


class ConvNeXtIsotropic_2dec_v10(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depth=18,
        dim=[48, 96, 240, 384],
        drop_path_rate=0.,
        layer_scale_init_value=0,
        head_init_scale=1.,
    ):
        super().__init__()

        self.stem1 = nn.Conv2d(in_chans, dim[0], kernel_size=4, stride=4)
        self.stem2 = nn.Conv2d(in_chans, dim[1], kernel_size=8, stride=8)
        self.stem3 = nn.Conv2d(2 * in_chans, dim[2], kernel_size=16, stride=16)
        self.stem4 = nn.Conv2d(2 * in_chans, dim[3], kernel_size=32, stride=32)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks1 = nn.Sequential(*[
            Block(dim=dim[0],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm1 = LayerNorm(dim[0], eps=1e-6)  # final norm layer

        self.blocks2 = nn.Sequential(*[
            Block(dim=dim[1],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm2 = LayerNorm(dim[1], eps=1e-6)  # final norm layer

        self.blocks3 = nn.Sequential(*[
            Block(dim=dim[2],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm3 = LayerNorm(dim[2], eps=1e-6)  # final norm layer

        self.blocks4 = nn.Sequential(*[
            Block(dim=dim[3],
                  drop_path=dp_rates[i],
                  layer_scale_init_value=layer_scale_init_value)
            for i in range(depth)
        ])

        self.norm4 = LayerNorm(dim[3], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, y):
        x1 = self.stem1(x)
        x2 = self.stem2(x)
        x = torch.cat((x, y), 1)
        x3 = self.stem3(x)
        x4 = self.stem4(x)

        x1 = self.blocks1(x1)
        x2 = self.blocks2(x2)
        x3 = self.blocks3(x3)
        x4 = self.blocks4(x4)

        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = self.norm1(x1)
        x1 = x1.permute(0, 3, 1, 2)

        x2 = x2.permute(0, 2, 3, 1).contiguous()
        x2 = self.norm2(x2)
        x2 = x2.permute(0, 3, 1, 2)

        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.norm3(x3)
        x3 = x3.permute(0, 3, 1, 2)

        x4 = x4.permute(0, 2, 3, 1).contiguous()
        x4 = self.norm4(x4)
        x4 = x4.permute(0, 3, 1, 2)

        return x1, x2, x3, x4  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, y):
        x1, x2, x3, x4 = self.forward_features(x, y)
        # x = self.head(x)
        return x1, x2, x3, x4


@register_model
def convnext_isotropic_small(pretrained=False, **kwargs):
    model = ConvNeXtIsotropic(depth=9, dim=[48, 96], **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_isotropic_small_2dec(dims_list, pretrained=False, **kwargs):
    model = ConvNeXtIsotropic_2dec(depth=18, dim=dims_list, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_isotropic_small_2dec_v3(dims_list, pretrained=False, **kwargs):
    model = ConvNeXtIsotropic_2dec_v3(depth=9, dim=dims_list, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_isotropic_small_2dec_v3SE(dims_list, pretrained=False, **kwargs):
    model = ConvNeXtIsotropic_2dec_v3SE(depth=9, dim=dims_list, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_isotropic_small_2dec_v3SE_DCN(dims_list, pretrained=False, **kwargs):
    model = ConvNeXtIsotropic_2dec_v3SE_DCN(depth=9, dim=dims_list, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_isotropic_small_2dec_v4(dims_list, pretrained=False, **kwargs):
    model = ConvNeXtIsotropic_2dec_v4(depth=9, dim=dims_list, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_isotropic_small_2dec_v10(dims_list, pretrained=False, **kwargs):
    model = ConvNeXtIsotropic_2dec_v10(depth=9, dim=dims_list, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_isotropic_base(pretrained=False, **kwargs):
    model = ConvNeXtIsotropic(depth=18, dim=768, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_base_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_isotropic_large(pretrained=False, **kwargs):
    model = ConvNeXtIsotropic(depth=36, dim=1024, **kwargs)
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/convnext/convnext_iso_large_1k_224_ema.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url=url,
                                                        map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
