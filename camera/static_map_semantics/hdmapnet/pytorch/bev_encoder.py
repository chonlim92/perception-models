"""
BEV (Bird's Eye View) encoder for HDMapNet.

Processes BEV feature maps using ResNet-style residual blocks with
a U-Net-like encoder-decoder structure (downsample then upsample)
to produce refined BEV features at the original spatial resolution.

Channel progression: 64 -> 128 -> 256 -> 128 -> 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions and a skip connection."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for the first convolution (used for downsampling).
            downsample: Optional downsampling module for the skip connection.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class DownBlock(nn.Module):
    """Downsampling block: stride-2 BasicBlock followed by another BasicBlock."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.block1 = BasicBlock(in_channels, out_channels, stride=2, downsample=downsample)
        self.block2 = BasicBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block: bilinear upsample + concat skip + two BasicBlocks."""

    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels: Channels from the deeper layer (before concat).
            skip_channels: Channels from the skip connection.
            out_channels: Output channels after fusion.
        """
        super().__init__()
        fused_channels = in_channels + skip_channels
        downsample_fuse = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.block1 = BasicBlock(fused_channels, out_channels, downsample=downsample_fuse)
        self.block2 = BasicBlock(out_channels, out_channels)

    def forward(self, x, skip):
        """
        Args:
            x: Feature map from deeper layer (lower resolution).
            skip: Feature map from the encoder (same resolution as target).
        """
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return x


class BEVEncoder(nn.Module):
    """BEV encoder with U-Net-like architecture.

    Encoder path: 64 -> 128 -> 256
    Decoder path: 256 -> 128 -> 64

    Uses skip connections from encoder to decoder at matching resolutions.
    """

    def __init__(self, in_channels=64, base_channels=64):
        """
        Args:
            in_channels: Number of input BEV feature channels.
            base_channels: Base number of channels (doubled at each encoder stage).
        """
        super().__init__()
        c1 = base_channels       # 64
        c2 = base_channels * 2   # 128
        c3 = base_channels * 4   # 256

        # Input projection if in_channels != c1
        if in_channels != c1:
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, c1, kernel_size=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_proj = nn.Identity()

        # Initial block at full resolution
        self.init_block = nn.Sequential(
            BasicBlock(c1, c1),
            BasicBlock(c1, c1),
        )

        # Encoder (downsampling)
        self.down1 = DownBlock(c1, c2)   # 64 -> 128, stride 2
        self.down2 = DownBlock(c2, c3)   # 128 -> 256, stride 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            BasicBlock(c3, c3),
            BasicBlock(c3, c3),
        )

        # Decoder (upsampling)
        self.up1 = UpBlock(c3, c2, c2)   # 256+128 -> 128
        self.up2 = UpBlock(c2, c1, c1)   # 128+64 -> 64

        # Output refinement
        self.output_block = nn.Sequential(
            BasicBlock(c1, c1),
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.out_channels = c1

    def forward(self, x):
        """
        Args:
            x: BEV feature map (B, C, bev_h, bev_w).

        Returns:
            Refined BEV feature map (B, 64, bev_h, bev_w).
        """
        x = self.input_proj(x)

        # Encoder
        s1 = self.init_block(x)     # (B, 64, H, W)
        s2 = self.down1(s1)         # (B, 128, H/2, W/2)
        s3 = self.down2(s2)         # (B, 256, H/4, W/4)

        # Bottleneck
        s3 = self.bottleneck(s3)    # (B, 256, H/4, W/4)

        # Decoder with skip connections
        d2 = self.up1(s3, s2)       # (B, 128, H/2, W/2)
        d1 = self.up2(d2, s1)       # (B, 64, H, W)

        # Output
        out = self.output_block(d1)  # (B, 64, H, W)
        return out
