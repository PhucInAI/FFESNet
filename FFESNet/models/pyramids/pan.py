import numpy as np
import math
from typing import List, Optional
import torch
import torch.nn as nn

Numpy = np.array
Tensor = torch.Tensor


class Downsampler_Conv(nn.Module):
    __doc__ = r"""
        This module adjusts padding to get a desired feature size from the given size,
        and downsample a feature by nn.Conv2d.
        
        Args:
            in_size: 2d size of input feature map, assumed that the height and width are same
            out_size: 2d size of output feature map, assumed that the height and width are same
            in_channels: in_channels of nn.Conv2d
            out_channels: out_channels of nn.Conv2d
            kernel_size: kernel_size of nn.Conv2d
            stride: stride of nn.Conv2d
            dilation: dilation of nn.Conv2d
            groups: groups of nn.Conv2d
            bias: bias of nn.Conv2d
        
        Output:
            a downsampled tensor
        
        The error condition is same as for nn.Conv2d.
    """

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 **kwargs):

        super().__init__()
        padding = math.ceil((stride * (out_size - 1) - in_size + dilation * (kernel_size - 1) + 1) / 2)

        if padding < 0:
            raise ValueError('negative padding is not supported for Conv2d')
        if stride < 2:
            raise ValueError('downsampling stride must be greater than 1')

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, **kwargs)

    def forward(self, x):
        return self.conv(x)


class FPN(nn.Module):

    __doc__ = r"""
        paper: https://arxiv.org/abs/1612.03144
        
        * All list arguments and input, output feature maps are given in bottom-to-top.
          
        Args:
            num_levels: the number of feature maps
            in_channels: channels of each input feature maps in list
            out_channels: channels of output feature maps 
            sizes: 2d size of each feature maps in list
            up_mode: nn.Upsample mode
        
        Output:
            list of feature maps in the same number of channels
            
        If 'sizes' is not given, 'scale_factor' of every upsampling are set to 2.
        """

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: Optional[List] = None,
                 up_mode: str = 'nearest'):

        self.num_levels = num_levels

        assert len(in_channels) == num_levels, \
            'make len(in_channels) = num_levels'
        if sizes:
            assert len(sizes) == num_levels, \
                'make len(sizes) = num_levels'

        super().__init__()

        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])

        if sizes:
            self.upsamples = nn.ModuleList([nn.Upsample(size=size, mode=up_mode) for size in sizes[:-1]])
        else:
            self.upsamples = nn.ModuleList([nn.Upsample(scale_factor=2, mode=up_mode) for _ in range(num_levels - 1)])

        self.fuses = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True) for _ in range(num_levels)])


    def forward(self, features: List[Tensor]) -> List[Tensor]:
        p_features = []

        for i in range(self.num_levels - 1, -1, -1):
            p = self.laterals[i](features[i])

            if p_features:
                u = self.upsamples[i](p_features[-1])
                p += u

            p_features.append(p)

        p_features = p_features[::-1]
        p_features = [f(p) for f, p in zip(self.fuses, p_features)]

        return p_features


class BU_FPN(nn.Module):

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: Optional[List] = None,
                 strides: list = None):

        self.num_levels = num_levels

        assert len(in_channels) == num_levels, \
            'make len(in_channels) = num_levels'
        if sizes:
            assert len(sizes) == num_levels and len(strides) == num_levels - 1, \
                'make len(sizes) = num_levels, and len(strides) = num_levels - 1'

        super().__init__()

        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])

        if sizes and strides:
            self.downsamples = nn.ModuleList([Downsampler_Conv(sizes[i], sizes[i + 1], out_channels, out_channels, 1, strides[i], bias=True)
                                              for i in range(len(sizes) - 1)])
        else:
            self.downsamples = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 1, 2, padding=0, bias=True)
                                              for _ in range(num_levels - 1)])

        self.fuses = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True) for _ in range(num_levels)])


    def forward(self, features: List[Tensor]) -> List[Tensor]:
        p_features = []

        for i in range(self.num_levels):
            p = self.laterals[i](features[i])

            if p_features:
                d = self.downsamples[i - 1](p_features[-1])
                p += d

            p = self.fuses[i](p)
            p_features.append(p)

        return p_features


class PAN(nn.Module):

    __doc__ = r"""
        paper: https://arxiv.org/abs/1803.01534

        * All list arguments and input, output feature maps are given in bottom-to-top.  

        Args:
            num_levels: the number of feature maps
            in_channels: channels of each input feature maps in list
            out_channels: channels of output feature maps 
            sizes: 2d size of each feature maps in list
            strides: list of strides between two feature maps, of nn.Conv2d for downsampling
            up_mode: nn.Upsample mode

        Output:
            list of feature maps in the same number of channels

        If 'sizes' and 'strides' are not given, 'scale_factor' of every upsampling 
        and 'stride' of every downsampling are set to 2.
        """

    def __init__(self,
                 num_levels: int,
                 in_channels: list,
                 out_channels: int,
                 sizes: Optional[List] = None,
                 strides: Optional[List] = None,
                 up_mode: str = 'nearest'):

        super().__init__()

        self.top_down = FPN(num_levels, in_channels, out_channels, sizes, up_mode)
        self.bottom_up = BU_FPN(num_levels, len(in_channels) * [out_channels], out_channels, sizes, strides)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        features = self.top_down(features)
        features = self.bottom_up(features)

        return features


__all__ = [PAN]