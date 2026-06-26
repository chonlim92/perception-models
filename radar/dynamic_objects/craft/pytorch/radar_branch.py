"""
Radar Branch for CRAFT (Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer).

Implements PointPillar-style radar point cloud encoding and a BEV (Bird's Eye View)
backbone for extracting spatial features from sparse radar detections.

Pipeline:
    1. RadarPillarEncoder: Discretize radar points into pillars, encode with PointNet-like MLP
    2. Scatter to BEV pseudo-image
    3. RadarBEVBackbone: 2D CNN to extract multi-scale spatial features
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PillarFeatureNet(nn.Module):
    """PointNet-like feature extractor for points within each pillar.

    Applies shared MLPs (implemented as 1D convolutions across points) followed by
    max-pooling to produce a single feature vector per pillar.

    Args:
        in_channels: Number of input point features. For radar: x, y, z, vx, vy, rcs = 6,
                     plus augmented features (x_c, y_c, z_c, x_p, y_p) = 5.
                     Total default input = 11.
        hidden_channels: Hidden layer dimension.
        out_channels: Output feature dimension per pillar.
    """

    def __init__(
        self,
        in_channels: int = 11,
        hidden_channels: int = 64,
        out_channels: int = 128,
    ) -> None:
        super().__init__()

        # Shared MLP implemented as 1x1 convolutions over the points dimension
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, pillar_features: torch.Tensor, pillar_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode points within pillars into per-pillar feature vectors.

        Args:
            pillar_features: Point features within pillars.
                Shape: [B, max_pillars, max_points_per_pillar, in_channels]
            pillar_mask: Binary mask indicating valid points.
                Shape: [B, max_pillars, max_points_per_pillar]

        Returns:
            Per-pillar feature vectors after max pooling.
            Shape: [B, max_pillars, out_channels]
        """
        B, P, N, C = pillar_features.shape

        # Reshape for batch processing through linear layers
        # [B, P, N, C] -> [B*P*N, C]
        x = pillar_features.reshape(B * P * N, C)

        # Apply shared MLP
        x = self.mlp(x)

        # Reshape back: [B*P*N, out_channels] -> [B, P, N, out_channels]
        out_channels = x.shape[-1]
        x = x.reshape(B, P, N, out_channels)

        # Mask invalid points before max pooling
        # Expand mask: [B, P, N] -> [B, P, N, 1]
        mask_expanded = pillar_mask.unsqueeze(-1).float()
        x = x * mask_expanded

        # Replace masked positions with -inf for correct max pooling
        x = x.masked_fill(mask_expanded == 0, float("-inf"))

        # Max pooling over points in each pillar: [B, P, N, out_channels] -> [B, P, out_channels]
        x, _ = x.max(dim=2)

        # Handle fully-empty pillars (all points masked) - replace -inf with 0
        x = x.clamp(min=0.0)

        return x


class RadarPillarEncoder(nn.Module):
    """Encodes raw radar point clouds into BEV pillar features.

    Converts sparse radar points into a structured pillar representation:
    1. Discretize x-y plane into a grid of pillars (voxels in BEV)
    2. Assign points to pillars based on their x-y location
    3. Augment point features with relative offsets from pillar center
    4. Encode points within each pillar using PillarFeatureNet
    5. Scatter encoded pillars to form BEV pseudo-image

    Args:
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] in meters.
        voxel_size: [vx, vy, vz] pillar dimensions in meters.
        max_points_per_pillar: Maximum number of points to keep per pillar.
        max_num_pillars: Maximum number of non-empty pillars.
        in_channels: Raw radar point feature dimension (x, y, z, vx, vy, rcs = 6).
        pillar_feat_channels: Output channels of PillarFeatureNet.
    """

    def __init__(
        self,
        point_cloud_range: List[float] = None,
        voxel_size: List[float] = None,
        max_points_per_pillar: int = 20,
        max_num_pillars: int = 30000,
        in_channels: int = 6,
        pillar_feat_channels: int = 128,
    ) -> None:
        super().__init__()

        if point_cloud_range is None:
            point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if voxel_size is None:
            voxel_size = [0.2, 0.2, 8.0]

        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.max_points_per_pillar = max_points_per_pillar
        self.max_num_pillars = max_num_pillars
        self.in_channels = in_channels

        # Compute BEV grid dimensions
        self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
        self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
        self.vx, self.vy, self.vz = voxel_size

        self.grid_size_x = int(round((self.x_max - self.x_min) / self.vx))  # 512
        self.grid_size_y = int(round((self.y_max - self.y_min) / self.vy))  # 512

        # Augmented features: original (6) + offset from pillar center (x_c, y_c, z_c) +
        # offset from point cloud center (x_p, y_p) = 6 + 3 + 2 = 11
        augmented_channels = in_channels + 5

        self.pillar_feature_net = PillarFeatureNet(
            in_channels=augmented_channels,
            hidden_channels=64,
            out_channels=pillar_feat_channels,
        )

        self.pillar_feat_channels = pillar_feat_channels

    def _create_pillars(
        self, points: torch.Tensor, num_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Discretize point cloud into pillars.

        Args:
            points: Radar point clouds, shape [B, N_max, in_channels].
                    Padded with zeros for variable-length point clouds.
            num_points: Number of valid points per sample, shape [B].

        Returns:
            pillar_features: Augmented point features in pillars.
                Shape: [B, max_num_pillars, max_points_per_pillar, augmented_channels]
            pillar_coords: Grid coordinates (ix, iy) of each pillar.
                Shape: [B, max_num_pillars, 2]
            pillar_mask: Valid point mask.
                Shape: [B, max_num_pillars, max_points_per_pillar]
        """
        B = points.shape[0]
        device = points.device

        pillar_features_list = []
        pillar_coords_list = []
        pillar_mask_list = []

        for b in range(B):
            n_valid = int(num_points[b].item())
            pts = points[b, :n_valid]  # [n_valid, in_channels]

            if n_valid == 0:
                # Handle empty point cloud
                pf = torch.zeros(
                    self.max_num_pillars, self.max_points_per_pillar,
                    self.in_channels + 5, device=device
                )
                pc = torch.zeros(self.max_num_pillars, 2, dtype=torch.long, device=device)
                pm = torch.zeros(
                    self.max_num_pillars, self.max_points_per_pillar,
                    dtype=torch.bool, device=device
                )
                pillar_features_list.append(pf)
                pillar_coords_list.append(pc)
                pillar_mask_list.append(pm)
                continue

            # Filter points within range
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            valid = (
                (x >= self.x_min) & (x < self.x_max) &
                (y >= self.y_min) & (y < self.y_max) &
                (z >= self.z_min) & (z < self.z_max)
            )
            pts = pts[valid]

            if pts.shape[0] == 0:
                pf = torch.zeros(
                    self.max_num_pillars, self.max_points_per_pillar,
                    self.in_channels + 5, device=device
                )
                pc = torch.zeros(self.max_num_pillars, 2, dtype=torch.long, device=device)
                pm = torch.zeros(
                    self.max_num_pillars, self.max_points_per_pillar,
                    dtype=torch.bool, device=device
                )
                pillar_features_list.append(pf)
                pillar_coords_list.append(pc)
                pillar_mask_list.append(pm)
                continue

            # Compute pillar indices for each point
            ix = ((pts[:, 0] - self.x_min) / self.vx).long()
            iy = ((pts[:, 1] - self.y_min) / self.vy).long()

            # Clamp to valid range
            ix = ix.clamp(0, self.grid_size_x - 1)
            iy = iy.clamp(0, self.grid_size_y - 1)

            # Unique pillar index = ix * grid_size_y + iy
            pillar_idx = ix * self.grid_size_y + iy

            # Find unique pillars and assign points
            unique_pillars, inverse_indices = torch.unique(pillar_idx, return_inverse=True)
            n_pillars = min(unique_pillars.shape[0], self.max_num_pillars)

            # Initialize outputs for this sample
            pf = torch.zeros(
                self.max_num_pillars, self.max_points_per_pillar,
                self.in_channels + 5, device=device
            )
            pc = torch.zeros(self.max_num_pillars, 2, dtype=torch.long, device=device)
            pm = torch.zeros(
                self.max_num_pillars, self.max_points_per_pillar,
                dtype=torch.bool, device=device
            )

            # Fill pillars
            for p_idx in range(n_pillars):
                # Get points belonging to this pillar
                point_mask = inverse_indices == p_idx
                pillar_points = pts[point_mask]

                # Limit number of points per pillar
                n_pts_in_pillar = min(pillar_points.shape[0], self.max_points_per_pillar)
                pillar_points = pillar_points[:n_pts_in_pillar]

                # Compute pillar center in world coordinates
                pillar_ix = unique_pillars[p_idx] // self.grid_size_y
                pillar_iy = unique_pillars[p_idx] % self.grid_size_y
                pillar_center_x = self.x_min + (pillar_ix.float() + 0.5) * self.vx
                pillar_center_y = self.y_min + (pillar_iy.float() + 0.5) * self.vy
                pillar_center_z = (self.z_min + self.z_max) / 2.0

                # Augmented features: offset from pillar center + offset from point cloud mean
                offset_from_center = torch.zeros(n_pts_in_pillar, 3, device=device)
                offset_from_center[:, 0] = pillar_points[:, 0] - pillar_center_x
                offset_from_center[:, 1] = pillar_points[:, 1] - pillar_center_y
                offset_from_center[:, 2] = pillar_points[:, 2] - pillar_center_z

                # Offset from mean of points in pillar (x_p, y_p)
                mean_xy = pillar_points[:, :2].mean(dim=0, keepdim=True)
                offset_from_mean = pillar_points[:, :2] - mean_xy

                # Concatenate: [original_features, offset_from_center, offset_from_mean]
                augmented = torch.cat(
                    [pillar_points, offset_from_center, offset_from_mean], dim=-1
                )

                pf[p_idx, :n_pts_in_pillar] = augmented
                pm[p_idx, :n_pts_in_pillar] = True
                pc[p_idx, 0] = pillar_ix
                pc[p_idx, 1] = pillar_iy

            pillar_features_list.append(pf)
            pillar_coords_list.append(pc)
            pillar_mask_list.append(pm)

        pillar_features = torch.stack(pillar_features_list, dim=0)
        pillar_coords = torch.stack(pillar_coords_list, dim=0)
        pillar_mask = torch.stack(pillar_mask_list, dim=0)

        return pillar_features, pillar_coords, pillar_mask

    def forward(
        self, points: torch.Tensor, num_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode radar points into pillar features and scatter to BEV.

        Args:
            points: Radar point clouds [B, N_max, in_channels].
                    Features per point: [x, y, z, vx, vy, rcs]
            num_points: Number of valid points per sample [B].

        Returns:
            bev_features: BEV pseudo-image [B, pillar_feat_channels, grid_size_x, grid_size_y].
            pillar_coords: Grid coordinates of pillars [B, max_num_pillars, 2].
        """
        # Create pillars from raw points
        pillar_features, pillar_coords, pillar_mask = self._create_pillars(points, num_points)

        # Encode pillar features using PillarFeatureNet
        # [B, max_pillars, max_points, augmented_channels] -> [B, max_pillars, out_channels]
        encoded_pillars = self.pillar_feature_net(pillar_features, pillar_mask)

        # Scatter pillars to BEV pseudo-image
        B = points.shape[0]
        device = points.device
        bev_features = torch.zeros(
            B, self.pillar_feat_channels, self.grid_size_x, self.grid_size_y,
            device=device, dtype=encoded_pillars.dtype,
        )

        for b in range(B):
            coords = pillar_coords[b]  # [max_pillars, 2]
            features = encoded_pillars[b]  # [max_pillars, out_channels]

            # Only scatter non-zero pillars (valid ones have non-zero coordinates or features)
            valid_pillars = pillar_mask[b].any(dim=-1)  # [max_pillars]
            if valid_pillars.any():
                valid_coords = coords[valid_pillars]  # [n_valid, 2]
                valid_features = features[valid_pillars]  # [n_valid, out_channels]

                # Scatter: place each pillar feature at its grid location
                ix = valid_coords[:, 0].long()
                iy = valid_coords[:, 1].long()

                # Use scatter_ for efficient placement
                # valid_features: [n_valid, out_channels] -> transpose to [out_channels, n_valid]
                for i in range(valid_features.shape[0]):
                    bev_features[b, :, ix[i], iy[i]] = valid_features[i]

        return bev_features, pillar_coords


class ConvBlock(nn.Module):
    """Basic convolutional block with Conv2d -> BatchNorm -> ReLU.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class RadarBEVBackbone(nn.Module):
    """2D CNN backbone for processing BEV pseudo-images from pillar encoding.

    Architecture:
    - Multi-scale feature extraction with stride-2 downsampling blocks
    - Deconvolution (transposed convolution) upsampling to merge multi-scale features
    - Produces a unified BEV feature map

    The backbone consists of multiple stages:
    - Stage 1: 128 -> 128, stride 1 (same resolution)
    - Stage 2: 128 -> 128, stride 2 (2x downsample)
    - Stage 3: 128 -> 256, stride 2 (4x downsample)

    Each stage's features are upsampled to the same resolution and concatenated,
    then projected to the final output dimension.

    Args:
        in_channels: Input BEV feature channels (from pillar encoder).
        layer_nums: Number of convolution blocks per stage.
        layer_strides: Stride for the first conv in each stage.
        num_filters: Output channels for each stage.
        upsample_strides: Stride for deconvolution in each stage.
        upsample_filters: Output channels for deconvolution in each stage.
        out_channels: Final output feature dimension.
    """

    def __init__(
        self,
        in_channels: int = 128,
        layer_nums: List[int] = None,
        layer_strides: List[int] = None,
        num_filters: List[int] = None,
        upsample_strides: List[int] = None,
        upsample_filters: List[int] = None,
        out_channels: int = 256,
    ) -> None:
        super().__init__()

        if layer_nums is None:
            layer_nums = [3, 5, 5]
        if layer_strides is None:
            layer_strides = [1, 2, 2]
        if num_filters is None:
            num_filters = [128, 128, 256]
        if upsample_strides is None:
            upsample_strides = [1, 2, 4]
        if upsample_filters is None:
            upsample_filters = [128, 128, 128]

        assert len(layer_nums) == len(layer_strides) == len(num_filters), (
            "layer_nums, layer_strides, and num_filters must have the same length"
        )
        assert len(upsample_strides) == len(upsample_filters) == len(layer_nums), (
            "upsample_strides and upsample_filters must match number of stages"
        )

        self.num_stages = len(layer_nums)

        # Build downsampling stages
        self.stages = nn.ModuleList()
        current_channels = in_channels

        for stage_idx in range(self.num_stages):
            blocks = []

            # First block may have stride > 1 for downsampling
            blocks.append(ConvBlock(
                current_channels, num_filters[stage_idx],
                kernel_size=3, stride=layer_strides[stage_idx], padding=1,
            ))

            # Remaining blocks at stride 1
            for _ in range(1, layer_nums[stage_idx]):
                blocks.append(ConvBlock(
                    num_filters[stage_idx], num_filters[stage_idx],
                    kernel_size=3, stride=1, padding=1,
                ))

            self.stages.append(nn.Sequential(*blocks))
            current_channels = num_filters[stage_idx]

        # Build upsampling (deconvolution) layers
        self.deconv_layers = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            stride = upsample_strides[stage_idx]
            if stride >= 1:
                deconv = nn.Sequential(
                    nn.ConvTranspose2d(
                        num_filters[stage_idx],
                        upsample_filters[stage_idx],
                        kernel_size=stride,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(upsample_filters[stage_idx]),
                    nn.ReLU(inplace=True),
                )
            else:
                # For fractional strides, use regular convolution with stride
                actual_stride = int(round(1.0 / stride))
                deconv = nn.Sequential(
                    nn.Conv2d(
                        num_filters[stage_idx],
                        upsample_filters[stage_idx],
                        kernel_size=actual_stride,
                        stride=actual_stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(upsample_filters[stage_idx]),
                    nn.ReLU(inplace=True),
                )
            self.deconv_layers.append(deconv)

        # Final projection to output channels
        total_upsample_channels = sum(upsample_filters)
        self.output_proj = nn.Sequential(
            nn.Conv2d(total_upsample_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process BEV pseudo-image through multi-scale backbone.

        Args:
            x: BEV feature map from pillar encoder [B, in_channels, H_bev, W_bev].

        Returns:
            Multi-scale fused BEV features [B, out_channels, H_bev, W_bev].
        """
        # Extract multi-scale features
        stage_outputs = []
        for stage in self.stages:
            x = stage(x)
            stage_outputs.append(x)

        # Upsample each stage output to the same resolution and concatenate
        upsampled = []
        for stage_idx, (feat, deconv) in enumerate(
            zip(stage_outputs, self.deconv_layers)
        ):
            up = deconv(feat)
            upsampled.append(up)

        # Ensure all upsampled features have the same spatial size
        # Use the first upsampled feature's size as reference
        target_h, target_w = upsampled[0].shape[2:]
        aligned = []
        for feat in upsampled:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            aligned.append(feat)

        # Concatenate along channel dimension
        concat = torch.cat(aligned, dim=1)

        # Project to output channels
        output = self.output_proj(concat)

        return output


class RadarBranch(nn.Module):
    """Complete radar processing branch combining pillar encoding and BEV backbone.

    End-to-end pipeline:
        Raw radar points -> Pillar Encoding -> BEV Pseudo-image -> BEV Backbone -> Radar BEV Features

    Args:
        point_cloud_range: Spatial extent [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: Pillar dimensions [vx, vy, vz] in meters.
        max_points_per_pillar: Maximum points retained per pillar.
        max_num_pillars: Maximum number of non-empty pillars.
        in_channels: Number of raw radar point features (x, y, z, vx, vy, rcs).
        pillar_feat_channels: Intermediate pillar feature dimension.
        bev_out_channels: Final BEV feature map channel dimension.
    """

    def __init__(
        self,
        point_cloud_range: List[float] = None,
        voxel_size: List[float] = None,
        max_points_per_pillar: int = 20,
        max_num_pillars: int = 30000,
        in_channels: int = 6,
        pillar_feat_channels: int = 128,
        bev_out_channels: int = 256,
    ) -> None:
        super().__init__()

        if point_cloud_range is None:
            point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if voxel_size is None:
            voxel_size = [0.2, 0.2, 8.0]

        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        # Pillar encoder
        self.pillar_encoder = RadarPillarEncoder(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_points_per_pillar=max_points_per_pillar,
            max_num_pillars=max_num_pillars,
            in_channels=in_channels,
            pillar_feat_channels=pillar_feat_channels,
        )

        # BEV backbone
        self.bev_backbone = RadarBEVBackbone(
            in_channels=pillar_feat_channels,
            layer_nums=[3, 5, 5],
            layer_strides=[1, 2, 2],
            num_filters=[128, 128, 256],
            upsample_strides=[1, 2, 4],
            upsample_filters=[128, 128, 128],
            out_channels=bev_out_channels,
        )

        # Store grid dimensions for external reference
        self.grid_size_x = self.pillar_encoder.grid_size_x
        self.grid_size_y = self.pillar_encoder.grid_size_y
        self.bev_out_channels = bev_out_channels

    def forward(
        self, points: torch.Tensor, num_points: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Process radar point cloud to produce BEV feature map.

        Args:
            points: Radar point clouds [B, N_max, 6].
                    Per-point features: [x, y, z, vx, vy, rcs]
                    - x, y, z: 3D position in meters
                    - vx, vy: radial velocity components in m/s
                    - rcs: radar cross-section in dBsm
            num_points: Number of valid points per sample [B].

        Returns:
            Dictionary with:
                'bev_features': Radar BEV feature map [B, bev_out_channels, H_bev, W_bev]
                'pillar_coords': Pillar grid coordinates [B, max_num_pillars, 2]
        """
        # Encode points into BEV pseudo-image
        bev_pseudo_image, pillar_coords = self.pillar_encoder(points, num_points)

        # Process through BEV backbone
        bev_features = self.bev_backbone(bev_pseudo_image)

        return {
            "bev_features": bev_features,
            "pillar_coords": pillar_coords,
        }

    def get_output_info(self) -> Dict[str, any]:
        """Return metadata about the output feature maps.

        Returns:
            Dictionary with BEV grid dimensions and feature channels.
        """
        return {
            "bev_channels": self.bev_out_channels,
            "bev_height": self.grid_size_x,
            "bev_width": self.grid_size_y,
            "point_cloud_range": self.point_cloud_range,
            "voxel_size": self.voxel_size,
        }


def build_radar_branch(
    point_cloud_range: List[float] = None,
    voxel_size: List[float] = None,
    max_points_per_pillar: int = 20,
    max_num_pillars: int = 30000,
    in_channels: int = 6,
    pillar_feat_channels: int = 128,
    bev_out_channels: int = 256,
) -> RadarBranch:
    """Factory function to build the radar branch.

    Args:
        point_cloud_range: Spatial extent [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: Pillar dimensions [vx, vy, vz].
        max_points_per_pillar: Maximum points per pillar.
        max_num_pillars: Maximum non-empty pillars.
        in_channels: Number of radar point features.
        pillar_feat_channels: Pillar feature dimension.
        bev_out_channels: Output BEV feature channels.

    Returns:
        Configured RadarBranch instance.
    """
    return RadarBranch(
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_points_per_pillar=max_points_per_pillar,
        max_num_pillars=max_num_pillars,
        in_channels=in_channels,
        pillar_feat_channels=pillar_feat_channels,
        bev_out_channels=bev_out_channels,
    )


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_radar_branch(
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=[0.2, 0.2, 8.0],
        max_points_per_pillar=20,
        max_num_pillars=30000,
        in_channels=6,
        pillar_feat_channels=128,
        bev_out_channels=256,
    ).to(device)

    # Simulate radar input: batch of 2 samples with variable point counts
    batch_size = 2
    max_points = 500  # Radar is sparse, typically 100-1000 points per frame

    # Random radar points: [x, y, z, vx, vy, rcs]
    dummy_points = torch.zeros(batch_size, max_points, 6, device=device)

    # Sample 1: 300 points
    dummy_points[0, :300, 0] = torch.rand(300, device=device) * 102.4 - 51.2  # x
    dummy_points[0, :300, 1] = torch.rand(300, device=device) * 102.4 - 51.2  # y
    dummy_points[0, :300, 2] = torch.rand(300, device=device) * 8.0 - 5.0     # z
    dummy_points[0, :300, 3] = torch.randn(300, device=device) * 5.0           # vx
    dummy_points[0, :300, 4] = torch.randn(300, device=device) * 5.0           # vy
    dummy_points[0, :300, 5] = torch.rand(300, device=device) * 30.0           # rcs

    # Sample 2: 150 points
    dummy_points[1, :150, 0] = torch.rand(150, device=device) * 102.4 - 51.2
    dummy_points[1, :150, 1] = torch.rand(150, device=device) * 102.4 - 51.2
    dummy_points[1, :150, 2] = torch.rand(150, device=device) * 8.0 - 5.0
    dummy_points[1, :150, 3] = torch.randn(150, device=device) * 5.0
    dummy_points[1, :150, 4] = torch.randn(150, device=device) * 5.0
    dummy_points[1, :150, 5] = torch.rand(150, device=device) * 30.0

    num_points = torch.tensor([300, 150], device=device)

    print("Radar Branch Configuration:")
    print(f"  Point cloud range: {model.point_cloud_range}")
    print(f"  Voxel size: {model.voxel_size}")
    print(f"  BEV grid size: {model.grid_size_x} x {model.grid_size_y}")
    print(f"  Max pillars: {model.pillar_encoder.max_num_pillars}")
    print(f"  Max points/pillar: {model.pillar_encoder.max_points_per_pillar}")
    print()

    with torch.no_grad():
        output = model(dummy_points, num_points)

    print("Radar Branch Output:")
    print(f"  BEV features shape: {output['bev_features'].shape}")
    print(f"  Pillar coords shape: {output['pillar_coords'].shape}")

    # Verify output shape
    bev_feat = output["bev_features"]
    assert bev_feat.shape[0] == batch_size
    assert bev_feat.shape[1] == 256  # bev_out_channels
    print(f"  BEV spatial dimensions: {bev_feat.shape[2]} x {bev_feat.shape[3]}")

    print("\nAll checks passed!")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
