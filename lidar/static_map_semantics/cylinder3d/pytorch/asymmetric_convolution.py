"""
Asymmetric Convolution Blocks for Cylinder3D.

Implements dimension-decomposed convolutions that exploit the anisotropic
nature of cylindrical voxel grids. The key insight is that rho, theta, and z
dimensions have very different resolutions and physical meanings, so
decomposing 3D convolutions into asymmetric kernels captures context more
efficiently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AsymmetricConvBlock(nn.Module):
    """
    Asymmetric 3D convolution block using three parallel branches.

    Each branch uses a different asymmetric kernel to capture directional context:
        - Branch 1: kernel (1, 3, 3) - captures theta-z plane context
        - Branch 2: kernel (3, 1, 3) - captures rho-z plane context
        - Branch 3: kernel (3, 3, 1) - captures rho-theta plane context

    The outputs are summed element-wise, followed by BatchNorm and LeakyReLU.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Convolution stride (applied to all branches). Default: 1
        padding_mode: Padding mode for convolutions. Default: 'zeros'
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        # Branch 1: kernel (1, 3, 3) - theta-z plane
        self.conv_branch1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=stride,
            padding=(0, 1, 1),
            bias=False,
            padding_mode=padding_mode,
        )

        # Branch 2: kernel (3, 1, 3) - rho-z plane
        self.conv_branch2 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            stride=stride,
            padding=(1, 0, 1),
            bias=False,
            padding_mode=padding_mode,
        )

        # Branch 3: kernel (3, 3, 1) - rho-theta plane
        self.conv_branch3 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 1),
            stride=stride,
            padding=(1, 1, 0),
            bias=False,
            padding_mode=padding_mode,
        )

        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D_rho, D_theta, D_z) input tensor

        Returns:
            (B, C_out, D_rho', D_theta', D_z') output tensor
        """
        out1 = self.conv_branch1(x)
        out2 = self.conv_branch2(x)
        out3 = self.conv_branch3(x)

        out = out1 + out2 + out3
        out = self.bn(out)
        out = self.activation(out)

        return out


class DDCMod(nn.Module):
    """
    Dimension-Decomposition Context Modeling (DDCMod).

    Decomposes 3D context modeling into three orthogonal 2D planes:
        - rho-theta plane (collapse z via average pooling)
        - rho-z plane (collapse theta via average pooling)
        - theta-z plane (collapse rho via average pooling)

    Each plane is processed with a 2D convolution, then the features are
    broadcast back and combined to produce the 3D output.

    Args:
        channels: Number of input/output channels (preserved)
    """

    def __init__(self, channels: int):
        super().__init__()

        # 2D convolution for rho-theta plane (viewing from top)
        self.conv_rho_theta = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 2D convolution for rho-z plane (viewing from side)
        self.conv_rho_z = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 2D convolution for theta-z plane (viewing from front)
        self.conv_theta_z = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Fusion layer to combine the three plane features
        self.fusion = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D_rho, D_theta, D_z) input 3D feature volume

        Returns:
            (B, C, D_rho, D_theta, D_z) context-enhanced feature volume
        """
        B, C, D_rho, D_theta, D_z = x.shape

        # Plane 1: rho-theta (average pool over z dimension)
        plane_rho_theta = x.mean(dim=4)  # (B, C, D_rho, D_theta)
        plane_rho_theta = self.conv_rho_theta(plane_rho_theta)  # (B, C, D_rho, D_theta)
        feat_rho_theta = plane_rho_theta.unsqueeze(4).expand_as(x)  # broadcast back

        # Plane 2: rho-z (average pool over theta dimension)
        plane_rho_z = x.mean(dim=3)  # (B, C, D_rho, D_z)
        plane_rho_z = self.conv_rho_z(plane_rho_z)  # (B, C, D_rho, D_z)
        feat_rho_z = plane_rho_z.unsqueeze(3).expand_as(x)  # broadcast back

        # Plane 3: theta-z (average pool over rho dimension)
        plane_theta_z = x.mean(dim=2)  # (B, C, D_theta, D_z)
        plane_theta_z = self.conv_theta_z(plane_theta_z)  # (B, C, D_theta, D_z)
        feat_theta_z = plane_theta_z.unsqueeze(2).expand_as(x)  # broadcast back

        # Combine: element-wise sum of three plane features + original
        combined = x + feat_rho_theta + feat_rho_z + feat_theta_z

        # Fusion with 1x1x1 conv
        out = self.fusion(combined)

        return out


class AsymmetricResBlock(nn.Module):
    """
    Asymmetric Residual Block.

    Consists of two AsymmetricConvBlocks with a skip connection.
    Supports both regular (stride=1) and downsampling (stride=2) modes.

    Architecture:
        x -> AsymConv1 -> AsymConv2 -> + -> out
        |                              ^
        +--- skip connection ----------+

    For downsampling, a strided 1x1x1 conv is used in the skip path.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for downsampling. Use 1 for regular, 2 for downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = AsymmetricConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = AsymmetricConvBlock(out_channels, out_channels, stride=1)

        # Skip connection
        self.use_projection = (in_channels != out_channels) or (stride != 1)
        if self.use_projection:
            self.skip = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D_rho, D_theta, D_z) input tensor

        Returns:
            (B, C_out, D_rho', D_theta', D_z') output tensor
        """
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity
        out = self.activation(out)

        return out


class AsymmetricDownBlock(nn.Module):
    """
    Downsampling block using stride-2 asymmetric convolution.

    Halves spatial dimensions while increasing channel count.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_blocks: Number of AsymmetricResBlocks at this resolution before downsampling
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2):
        super().__init__()

        blocks = []
        # First block handles channel change and downsampling
        blocks.append(AsymmetricResBlock(in_channels, out_channels, stride=2))
        # Additional blocks at the new resolution
        for _ in range(num_blocks - 1):
            blocks.append(AsymmetricResBlock(out_channels, out_channels, stride=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class AsymmetricUpBlock(nn.Module):
    """
    Upsampling block using transposed convolution with skip connections.

    Doubles spatial dimensions while decreasing channel count.
    Concatenates skip connection features from the encoder.

    Args:
        in_channels: Number of input channels (from previous decoder stage)
        skip_channels: Number of channels from the encoder skip connection
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.bn_up = nn.BatchNorm3d(in_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        # After concatenation with skip, input channels = in_channels + skip_channels
        self.conv_block = nn.Sequential(
            AsymmetricResBlock(in_channels + skip_channels, out_channels, stride=1),
            AsymmetricResBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D/2, H/2, W/2) input from previous decoder stage
            skip: (B, C_skip, D, H, W) encoder features at this resolution

        Returns:
            (B, C_out, D, H, W) upsampled and fused features
        """
        x = self.upsample(x)
        x = self.bn_up(x)
        x = self.activation(x)

        # Handle potential size mismatch from non-even dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Process concatenated features
        x = self.conv_block(x)

        return x
