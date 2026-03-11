"""
Adapted from:
https://github.com/Turoad/CLRNet/blob/main/clrnet/models/necks/fpn.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmdet.registry import MODELS


class FeatureEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureEnhancementModule, self).__init__()
        self.conv1x1_g = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv1x1_a = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3_a = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1x1_enh = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn_enh = nn.BatchNorm2d(in_channels)

    def forward(self, f_ori):
        # 1. Gain generation
        g = torch.sigmoid(self.conv1x1_g(f_ori))
        
        # 2. Feature amplification
        f_gain = (1 + g) * f_ori
        
        # 3. Spatial refinement
        a_special = torch.sigmoid(self.conv1x1_a(f_gain))
        f_attn = a_special * self.conv3x3_a(f_gain)
        
        # 4. Residual fusion
        f_enh = self.bn_enh(f_ori + self.conv1x1_enh(f_attn))
        
        return f_enh


@MODELS.register_module()
class CLRerNetFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs):
        """
        Feature pyramid network for CLRerNet.
        Args:
            in_channels (List[int]): Channel number list.
            out_channels (int): Number of output feature map channels.
            num_outs (int): Number of output feature map levels.
        """
        super(CLRerNetFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.backbone_end_level = self.num_ins
        self.start_level = 0
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fem = FeatureEnhancementModule(self.out_channels)

    def forward(self, inputs):
        """
        Args:
            inputs (List[torch.Tensor]): Input feature maps.
              Example of shapes:
                ([1, 64, 80, 200], [1, 128, 40, 100], [1, 256, 20, 50], [1, 512, 10, 25]).
        Returns:
            outputs (Tuple[torch.Tensor]): Output feature maps.
              The number of feature map levels and channels correspond to
               `num_outs` and `out_channels` respectively.
              Example of shapes:
                ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
        """
        if isinstance(inputs, tuple):
            inputs = list(inputs)

        assert len(inputs) >= len(self.in_channels)  # 4 > 3

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        outs[0] = self.fem(outs[0])
        return tuple(outs)
