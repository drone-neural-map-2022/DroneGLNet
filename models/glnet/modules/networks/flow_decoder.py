"""
Modified from https://github.com/yzcjtr/GeoNet/blob/master/geonet_nets.py
"""


import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x, size=None):
    """Upsample input tensor by a factor of 2
    """
    if size is not None:
        return F.interpolate(x, size=size, mode="nearest")
    else:
        return F.interpolate(x, scale_factor=2, mode="nearest")


class FlowDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_frames=1, use_skips=True):
        super(FlowDecoder, self).__init__()

        self.num_output_frames = num_output_frames
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.flow_scale = 0.1

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("flowconv", s)] = Conv3x3(self.num_ch_dec[s], 2 * self.num_output_frames)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            if i != 0:
                x = upsample(x, input_features[i - 1].shape[-2:])
            else:
                x = upsample(x)
            x = [self.convs[("upconv", i, 0)](x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("flow", i)] = self.flow_scale * self.convs[("flowconv", i)](x)

        return self.outputs

