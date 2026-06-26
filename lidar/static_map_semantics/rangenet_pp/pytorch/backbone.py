"""DarkNet-53 backbone adapted for range image semantic segmentation.

Implements the DarkNet-53 encoder from:
  "RangeNet++: Fast and Accurate LiDAR Semantic Segmentation"
  (Milioto et al., IROS 2019)

The backbone processes range images (B, 5, H, W) and returns multi-scale
feature maps for the U-Net decoder.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class DarkNetBlock(nn.Module):
    """DarkNet residual block: Conv1x1 (reduce) -> Conv3x3 (restore) + skip."""

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        return out + residual


class DarkNetStage(nn.Module):
    """A DarkNet stage: downsample conv followed by N residual blocks."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        # Downsample with stride-2 convolution
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        # Residual blocks
        blocks = [DarkNetBlock(out_channels) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class DarkNet53Backbone(nn.Module):
    """DarkNet-53 backbone adapted for range images.

    Architecture:
        - Initial convolution: 5 -> 32 channels
        - Stage 1: 32 -> 64,  1 residual block
        - Stage 2: 64 -> 128, 2 residual blocks
        - Stage 3: 128 -> 256, 8 residual blocks
        - Stage 4: 256 -> 512, 8 residual blocks
        - Stage 5: 512 -> 1024, 4 residual blocks

    Returns multi-scale features [stage1, stage2, stage3, stage4, stage5]
    for the decoder skip connections.
    """

    def __init__(self, in_channels: int = 5):
        super().__init__()
        # Initial convolution: input channels -> 32
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # DarkNet-53 stages: [1, 2, 8, 8, 4] residual blocks
        # Channel progression: 32 -> 64 -> 128 -> 256 -> 512 -> 1024
        self.stage1 = DarkNetStage(32, 64, num_blocks=1)
        self.stage2 = DarkNetStage(64, 128, num_blocks=2)
        self.stage3 = DarkNetStage(128, 256, num_blocks=8)
        self.stage4 = DarkNetStage(256, 512, num_blocks=8)
        self.stage5 = DarkNetStage(512, 1024, num_blocks=4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning multi-scale features.

        Args:
            x: Input range image tensor (B, 5, H, W)

        Returns:
            List of feature maps at progressively reduced resolutions:
                [stage1_out, stage2_out, stage3_out, stage4_out, stage5_out]
                Channels: [64, 128, 256, 512, 1024]
                Spatial: [H/2, H/4, H/8, H/16, H/32] x [W/2, W/4, W/8, W/16, W/32]
        """
        x = self.initial_conv(x)  # (B, 32, H, W)

        s1 = self.stage1(x)   # (B, 64, H/2, W/2)
        s2 = self.stage2(s1)  # (B, 128, H/4, W/4)
        s3 = self.stage3(s2)  # (B, 256, H/8, W/8)
        s4 = self.stage4(s3)  # (B, 512, H/16, W/16)
        s5 = self.stage5(s4)  # (B, 1024, H/32, W/32)

        return [s1, s2, s3, s4, s5]

    def get_output_channels(self) -> List[int]:
        """Return output channel counts for each stage."""
        return [64, 128, 256, 512, 1024]
