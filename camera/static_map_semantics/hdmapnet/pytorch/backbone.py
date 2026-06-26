"""
Backbone feature extractors for HDMapNet.

Provides EfficientNet-B0 and ResNet-50 backbones with FPN neck,
returning a single-scale feature map suitable for view transformation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FPNNeck(nn.Module):
    """Feature Pyramid Network neck that fuses multi-scale features into a
    single-scale output."""

    def __init__(self, in_channels_list, out_channels=64):
        """
        Args:
            in_channels_list: List of channel dimensions for [C3, C4, C5].
            out_channels: Number of output channels after FPN fusion.
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            )
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        """
        Args:
            features: List of feature maps [C3, C4, C5] at different scales.

        Returns:
            Fused single-scale feature map at C3's spatial resolution.
        """
        laterals = []
        for i, feat in enumerate(features):
            laterals.append(self.lateral_convs[i](feat))

        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False
            )

        # Apply output convolutions
        outputs = []
        target_size = laterals[0].shape[2:]
        for i, lat in enumerate(laterals):
            out = self.output_convs[i](lat)
            if out.shape[2:] != target_size:
                out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
            outputs.append(out)

        # Concatenate and fuse
        fused = torch.cat(outputs, dim=1)
        fused = self.fuse_conv(fused)
        return fused


class EfficientNetB0Backbone(nn.Module):
    """EfficientNet-B0 backbone with FPN neck.

    Extracts multi-scale features at stages 3, 5, 8 (C3, C4, C5)
    with channel dimensions [40, 112, 320].
    """

    def __init__(self, pretrained=True, out_channels=64):
        """
        Args:
            pretrained: Whether to use ImageNet pretrained weights.
            out_channels: Number of output channels from FPN.
        """
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        efficientnet = models.efficientnet_b0(weights=weights)

        # EfficientNet-B0 feature blocks:
        # features[0]: Conv stem
        # features[1]: MBConv stage 1 (channels: 16)
        # features[2]: MBConv stage 2 (channels: 24)
        # features[3]: MBConv stage 3 (channels: 40) -> C3
        # features[4]: MBConv stage 4 (channels: 80)
        # features[5]: MBConv stage 5 (channels: 112) -> C4
        # features[6]: MBConv stage 6 (channels: 192)
        # features[7]: MBConv stage 7 (channels: 320) -> C5
        # features[8]: Conv head (channels: 1280)
        self.stage1 = nn.Sequential(*efficientnet.features[:4])  # up to C3
        self.stage2 = nn.Sequential(*efficientnet.features[4:6])  # up to C4
        self.stage3 = nn.Sequential(*efficientnet.features[6:8])  # up to C5

        self.in_channels_list = [40, 112, 320]
        self.fpn = FPNNeck(self.in_channels_list, out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Feature map (B, out_channels, H/8, W/8).
        """
        c3 = self.stage1(x)   # stride 8
        c4 = self.stage2(c3)  # stride 16
        c5 = self.stage3(c4)  # stride 32

        out = self.fpn([c3, c4, c5])
        return out


class ResNet50Backbone(nn.Module):
    """ResNet-50 backbone with FPN neck.

    Extracts multi-scale features from layer2, layer3, layer4 (C3, C4, C5)
    with channel dimensions [512, 1024, 2048].
    """

    def __init__(self, pretrained=True, out_channels=64):
        """
        Args:
            pretrained: Whether to use ImageNet pretrained weights.
            out_channels: Number of output channels from FPN.
        """
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # stride 4, channels 256
        self.layer2 = resnet.layer2  # stride 8, channels 512 -> C3
        self.layer3 = resnet.layer3  # stride 16, channels 1024 -> C4
        self.layer4 = resnet.layer4  # stride 32, channels 2048 -> C5

        self.in_channels_list = [512, 1024, 2048]
        self.fpn = FPNNeck(self.in_channels_list, out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Feature map (B, out_channels, H/8, W/8).
        """
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)   # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32

        out = self.fpn([c3, c4, c5])
        return out
