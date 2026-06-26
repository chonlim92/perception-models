"""ResNet-101 + FPN backbone for BEVFormer multi-scale feature extraction.

This module implements a Feature Pyramid Network on top of a pretrained
ResNet-101 backbone. It extracts multi-scale features from multiple camera
views simultaneously, producing 4 FPN levels (P3-P6) with 256 output channels.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights

__all__ = ["ResNetFPN"]


class FPN(nn.Module):
    """Feature Pyramid Network that produces multi-scale features from ResNet stages.

    Takes feature maps from ResNet layer2 (C3), layer3 (C4), layer4 (C5) and
    produces P3, P4, P5, P6 with a uniform channel dimension of 256.
    P6 is obtained via stride-2 max pooling on P5.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256) -> None:
        """Initialize FPN layers.

        Args:
            in_channels_list: Input channel dimensions for C3, C4, C5.
                For ResNet-101 these are [512, 1024, 2048].
            out_channels: Output channel dimension for all FPN levels.
        """
        super().__init__()
        self.out_channels = out_channels

        # Lateral 1x1 convolutions to reduce channel dimensions
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

        # Output 3x3 convolutions to reduce aliasing from upsampling
        self.output_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform and biases with zeros."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Forward pass through FPN.

        Args:
            features: List of [C3, C4, C5] feature maps from ResNet.
                Each has shape (B*N_cams, C_i, H_i, W_i).

        Returns:
            List of [P3, P4, P5, P6] feature maps, each with out_channels channels.
                P6 is obtained by stride-2 max pooling on P5.
        """
        assert len(features) == len(self.lateral_convs)

        # Build laterals
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]

        # Top-down pathway: add upsampled higher-level features to lower levels
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply output convolutions
        outputs = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

        # P6: stride-2 max pooling on P5
        p6 = F.max_pool2d(outputs[-1], kernel_size=2, stride=2)
        outputs.append(p6)

        return outputs


class ResNetFPN(nn.Module):
    """ResNet-101 backbone with Feature Pyramid Network for BEVFormer.

    Extracts multi-scale features from multiple camera images. Uses a pretrained
    ResNet-101 as backbone and builds an FPN on top of stages C3, C4, C5.

    Input shape:  (B, N_cams, 3, H, W)
    Output shape: List of (B*N_cams, 256, H_i, W_i) for 4 FPN levels.
        - P3: stride 8 relative to input
        - P4: stride 16 relative to input
        - P5: stride 32 relative to input
        - P6: stride 64 relative to input
    """

    def __init__(
        self,
        out_channels: int = 256,
        pretrained: bool = True,
        frozen_stages: int = 1,
    ) -> None:
        """Initialize ResNet-101 + FPN backbone.

        Args:
            out_channels: Number of output channels for each FPN level.
            pretrained: Whether to use ImageNet pretrained weights.
            frozen_stages: Number of ResNet stages to freeze (0-4).
                Stage 0 = stem (conv1 + bn1), stage 1 = layer1, etc.
        """
        super().__init__()
        self.out_channels = out_channels
        self.frozen_stages = frozen_stages

        # Load pretrained ResNet-101
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        resnet = resnet101(weights=weights)

        # Extract backbone stages
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # C2: stride 4, 256 channels
        self.layer2 = resnet.layer2  # C3: stride 8, 512 channels
        self.layer3 = resnet.layer3  # C4: stride 16, 1024 channels
        self.layer4 = resnet.layer4  # C5: stride 32, 2048 channels

        # FPN takes C3, C4, C5
        self.fpn = FPN(
            in_channels_list=[512, 1024, 2048],
            out_channels=out_channels,
        )

        # Freeze early stages
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze parameters in early ResNet stages for stable training."""
        if self.frozen_stages >= 0:
            for param in self.stem.parameters():
                param.requires_grad = False

        frozen_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(min(self.frozen_stages, 4)):
            layer = frozen_layers[i]
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale FPN features from multi-camera images.

        Args:
            images: Multi-camera images with shape (B, N_cams, 3, H, W).

        Returns:
            List of 4 feature maps [P3, P4, P5, P6], each with shape
            (B*N_cams, out_channels, H_i, W_i) where H_i, W_i decrease
            by factor 2 at each successive level.
        """
        batch_size, num_cams = images.shape[:2]

        # Reshape to process all camera views together
        # (B, N_cams, 3, H, W) -> (B*N_cams, 3, H, W)
        x = images.flatten(0, 1)

        # ResNet forward
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # FPN forward: produces [P3, P4, P5, P6]
        fpn_features = self.fpn([c3, c4, c5])

        return fpn_features

    def train(self, mode: bool = True) -> "ResNetFPN":
        """Override train to keep frozen stages in eval mode."""
        super().train(mode)
        if mode:
            self._freeze_stages()
            # Keep BatchNorm in eval mode for frozen stages
            if self.frozen_stages >= 0:
                for module in self.stem.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            frozen_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for i in range(min(self.frozen_stages, 4)):
                for module in frozen_layers[i].modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
        return self
