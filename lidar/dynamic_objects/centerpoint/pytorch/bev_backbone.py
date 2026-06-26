"""
CenterPoint BEV (Bird's Eye View) Backbone.

Converts 3D sparse backbone output into a 2D BEV feature map, then processes
it with a ResNet-style 2D CNN backbone with multi-scale feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

# Import SparseTensor from local module
try:
    from .sparse_backbone import SparseTensor
except ImportError:
    from sparse_backbone import SparseTensor


class SparseToBEV(nn.Module):
    """Convert sparse 3D features to dense 2D BEV representation.

    Takes sparse features from the 3D backbone (at some stride) and collapses
    the Z dimension to produce a BEV feature map. Two modes:
    - 'concat': Concatenate features across Z bins along channel dimension.
    - 'sum': Sum features across the Z dimension.

    Args:
        in_channels: Number of input feature channels.
        spatial_shape: (D, H, W) spatial shape of the sparse volume at backbone output.
        mode: How to collapse Z dimension ('concat' or 'sum').
    """

    def __init__(
        self,
        in_channels: int = 128,
        spatial_shape: Tuple[int, int, int] = (5, 180, 180),
        mode: str = 'sum',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_shape = spatial_shape
        self.mode = mode

        if mode == 'concat':
            self.out_channels = in_channels * spatial_shape[0]
        else:
            self.out_channels = in_channels

    def forward(self, sparse_input: SparseTensor) -> torch.Tensor:
        """
        Args:
            sparse_input: SparseTensor with features at stride-8 resolution.

        Returns:
            bev_features: (B, C, H, W) dense BEV feature map.
        """
        features = sparse_input.features  # (N, C)
        indices = sparse_input.indices  # (N, 4) [batch, z, y, x]
        batch_size = sparse_input.batch_size
        D, H, W = sparse_input.spatial_shape
        device = features.device
        dtype = features.dtype

        if self.mode == 'concat':
            # Create dense BEV by concatenating Z slices
            bev = torch.zeros(
                batch_size, self.in_channels * D, H, W,
                dtype=dtype, device=device
            )
            batch_idx = indices[:, 0].long()
            z_idx = indices[:, 1].long()
            y_idx = indices[:, 2].long()
            x_idx = indices[:, 3].long()

            # Channel offset based on Z position
            channel_offset = z_idx * self.in_channels

            # Scatter features into BEV
            for i in range(features.shape[0]):
                b = batch_idx[i]
                ch_start = channel_offset[i]
                bev[b, ch_start:ch_start + self.in_channels, y_idx[i], x_idx[i]] += features[i]

        else:  # sum
            bev = torch.zeros(
                batch_size, self.in_channels, H, W,
                dtype=dtype, device=device
            )
            batch_idx = indices[:, 0].long()
            y_idx = indices[:, 2].long()
            x_idx = indices[:, 3].long()

            # Accumulate features by summing over Z
            # Use scatter_add for efficiency
            linear_bev_idx = batch_idx * (H * W) + y_idx * W + x_idx
            bev_flat = bev.view(batch_size * H * W, self.in_channels)

            # Ensure correct shapes for scatter_add
            expanded_idx = linear_bev_idx.unsqueeze(1).expand(-1, self.in_channels)
            bev_flat.scatter_add_(0, expanded_idx, features)

            bev = bev_flat.view(batch_size, self.in_channels, H, W)

        return bev


class BEVResBlock(nn.Module):
    """Residual block for BEV feature processing.

    Architecture: Conv2d -> BN -> ReLU -> Conv2d -> BN + shortcut.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride for the first convolution (for downsampling).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class BEVBackbone(nn.Module):
    """BEV feature backbone with two stages and multi-scale fusion.

    Architecture:
        - Stage 1: 128 channels, 3 residual blocks
        - Stage 2: 256 channels, 5 residual blocks (stride-2 entry)
        - Upsample Stage 1 and Stage 2 features to same resolution
        - Concatenate for final BEV feature map

    Args:
        in_channels: Number of input BEV feature channels.
        stage1_channels: Channel dimension for stage 1 (default: 128).
        stage2_channels: Channel dimension for stage 2 (default: 256).
        stage1_blocks: Number of residual blocks in stage 1 (default: 3).
        stage2_blocks: Number of residual blocks in stage 2 (default: 5).
        upsample_channels: Channel dimension after upsampling each stage (default: 128).
    """

    def __init__(
        self,
        in_channels: int = 128,
        stage1_channels: int = 128,
        stage2_channels: int = 256,
        stage1_blocks: int = 3,
        stage2_blocks: int = 5,
        upsample_channels: int = 128,
    ):
        super().__init__()

        # Stage 1: maintain resolution
        stage1_layers = []
        prev_ch = in_channels
        for i in range(stage1_blocks):
            out_ch = stage1_channels
            stride = 1
            stage1_layers.append(BEVResBlock(prev_ch, out_ch, stride=stride))
            prev_ch = out_ch
        self.stage1 = nn.Sequential(*stage1_layers)

        # Stage 2: downsample by 2x at entry, then maintain resolution
        stage2_layers = []
        prev_ch = stage1_channels
        for i in range(stage2_blocks):
            out_ch = stage2_channels
            stride = 2 if i == 0 else 1
            stage2_layers.append(BEVResBlock(prev_ch, out_ch, stride=stride))
            prev_ch = out_ch
        self.stage2 = nn.Sequential(*stage2_layers)

        # Upsampling: bring stage 1 and stage 2 to same resolution
        # Stage 1 is at 1x resolution -> upsample 1x (identity) then project
        self.upsample_stage1 = nn.Sequential(
            nn.ConvTranspose2d(
                stage1_channels, upsample_channels,
                kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(upsample_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 2 is at 0.5x resolution -> upsample 2x
        self.upsample_stage2 = nn.Sequential(
            nn.ConvTranspose2d(
                stage2_channels, upsample_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(upsample_channels),
            nn.ReLU(inplace=True),
        )

        # Output channels after concatenation
        self.out_channels = upsample_channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) BEV feature map from SparseToBEV.

        Returns:
            out: (B, C_out, H, W) fused multi-scale BEV features.
        """
        # Stage 1: full resolution
        s1 = self.stage1(x)  # (B, 128, H, W)

        # Stage 2: half resolution
        s2 = self.stage2(s1)  # (B, 256, H/2, W/2)

        # Upsample both stages
        up1 = self.upsample_stage1(s1)  # (B, 128, H, W)
        up2 = self.upsample_stage2(s2)  # (B, 128, H, W)

        # Ensure spatial dimensions match (handle odd sizes)
        if up2.shape[2:] != up1.shape[2:]:
            up2 = F.interpolate(up2, size=up1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate for final BEV feature map
        out = torch.cat([up1, up2], dim=1)  # (B, 256, H, W)

        return out


class BEVFeatureNet(nn.Module):
    """Complete BEV feature extraction pipeline.

    Combines sparse-to-BEV conversion with the 2D BEV backbone.

    Args:
        sparse_channels: Feature channels from the 3D sparse backbone.
        sparse_spatial_shape: (D, H, W) spatial shape of sparse backbone output.
        collapse_mode: How to collapse Z dimension ('sum' or 'concat').
        backbone_kwargs: Additional kwargs for BEVBackbone.
    """

    def __init__(
        self,
        sparse_channels: int = 128,
        sparse_spatial_shape: Tuple[int, int, int] = (5, 180, 180),
        collapse_mode: str = 'sum',
        **backbone_kwargs,
    ):
        super().__init__()
        self.sparse_to_bev = SparseToBEV(
            in_channels=sparse_channels,
            spatial_shape=sparse_spatial_shape,
            mode=collapse_mode,
        )

        bev_in_channels = self.sparse_to_bev.out_channels
        self.backbone = BEVBackbone(in_channels=bev_in_channels, **backbone_kwargs)
        self.out_channels = self.backbone.out_channels

    def forward(self, sparse_input: SparseTensor) -> torch.Tensor:
        """
        Args:
            sparse_input: SparseTensor from 3D backbone.

        Returns:
            bev_features: (B, C, H, W) final BEV feature map.
        """
        bev = self.sparse_to_bev(sparse_input)
        out = self.backbone(bev)
        return out
