"""
Point Refinement Module for Cylinder3D.

After obtaining voxel-level semantic predictions from the backbone, this module
refines predictions at the individual point level. It extracts per-point features
from the voxel grid (via nearest-neighbor lookup or trilinear interpolation),
concatenates with original point features, and processes through an MLP to
produce refined per-point logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class PointRefinementModule(nn.Module):
    """
    Point-level prediction refinement.

    Takes voxel-level features from the backbone and original per-point features,
    extracts per-point voxel features using the point-to-voxel mapping, and
    produces refined per-point class logits through an MLP.

    Architecture:
        Input: [voxel_feature_at_point (C_voxel), point_features (C_point)]
        -> Linear(C_voxel + C_point, 256) -> BN -> ReLU
        -> Linear(256, 256) -> BN -> ReLU
        -> Linear(256, 128) -> BN -> ReLU
        -> Dropout(0.3)
        -> Linear(128, num_classes)

    Args:
        voxel_feature_dim: Dimension of per-voxel features from backbone decoder.
                           Default: 32 (matches decoder output channels)
        point_feature_dim: Dimension of original per-point features.
                           Default: 9 (from CylindricalPartition)
        num_classes: Number of semantic classes.
                     Default: 20
        hidden_dims: Hidden layer dimensions for the MLP.
                     Default: [256, 256, 128]
        dropout: Dropout probability before the final classification layer.
                 Default: 0.3
        use_interpolation: If True, use trilinear interpolation to extract
                          voxel features; otherwise use nearest-neighbor lookup.
                          Default: False (nearest neighbor is faster and simpler)
    """

    def __init__(
        self,
        voxel_feature_dim: int = 32,
        point_feature_dim: int = 9,
        num_classes: int = 20,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.3,
        use_interpolation: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        self.voxel_feature_dim = voxel_feature_dim
        self.point_feature_dim = point_feature_dim
        self.num_classes = num_classes
        self.use_interpolation = use_interpolation

        # Build MLP layers
        input_dim = voxel_feature_dim + point_feature_dim
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def extract_point_features_nearest(
        self,
        voxel_features_volume: torch.Tensor,
        point_to_voxel: torch.Tensor,
        voxel_coords: torch.Tensor,
        grid_size: list,
    ) -> torch.Tensor:
        """
        Extract per-point features from voxel volume using nearest-neighbor lookup.

        Uses the point_to_voxel mapping to directly index into voxel features.

        Args:
            voxel_features_volume: (B, C, D_rho, D_theta, D_z) dense feature volume
            point_to_voxel: (N,) mapping from points to voxel linear indices
            voxel_coords: (M, 3) coordinates [rho_idx, theta_idx, z_idx] of occupied voxels
            grid_size: [D_rho, D_theta, D_z]

        Returns:
            point_voxel_features: (N, C) extracted features for each point
        """
        B, C = voxel_features_volume.shape[0], voxel_features_volume.shape[1]
        N = point_to_voxel.shape[0]
        device = voxel_features_volume.device

        # For points with valid voxel mapping, extract features
        # Reshape volume for easy indexing: (B, C, D_rho, D_theta, D_z)
        # We assume batch size = 1 for point cloud processing
        volume = voxel_features_volume[0]  # (C, D_rho, D_theta, D_z)

        point_voxel_features = torch.zeros(N, C, device=device, dtype=volume.dtype)

        valid_mask = point_to_voxel >= 0
        valid_indices = point_to_voxel[valid_mask]

        if valid_indices.numel() > 0:
            # Get voxel coordinates for valid points
            valid_coords = voxel_coords[valid_indices]  # (N_valid, 3)

            # Index into the volume
            rho_idx = valid_coords[:, 0].long()
            theta_idx = valid_coords[:, 1].long()
            z_idx = valid_coords[:, 2].long()

            # Extract features: volume is (C, D_rho, D_theta, D_z)
            extracted = volume[:, rho_idx, theta_idx, z_idx]  # (C, N_valid)
            point_voxel_features[valid_mask] = extracted.t()  # (N_valid, C)

        return point_voxel_features

    def extract_point_features_interpolated(
        self,
        voxel_features_volume: torch.Tensor,
        points_cylindrical: torch.Tensor,
        grid_size: list,
        rho_range: list,
        theta_range: list,
        z_range: list,
    ) -> torch.Tensor:
        """
        Extract per-point features using trilinear interpolation.

        Maps each point's cylindrical coordinates to normalized grid coordinates
        and uses grid_sample for trilinear interpolation.

        Args:
            voxel_features_volume: (B, C, D_rho, D_theta, D_z) dense feature volume
            points_cylindrical: (N, 3) [rho, theta, z] for each point
            grid_size: [D_rho, D_theta, D_z]
            rho_range: [rho_min, rho_max]
            theta_range: [theta_min, theta_max]
            z_range: [z_min, z_max]

        Returns:
            point_voxel_features: (N, C) interpolated features for each point
        """
        B, C = voxel_features_volume.shape[0], voxel_features_volume.shape[1]
        N = points_cylindrical.shape[0]
        device = voxel_features_volume.device

        # Normalize coordinates to [-1, 1] for grid_sample
        rho_norm = 2.0 * (points_cylindrical[:, 0] - rho_range[0]) / (rho_range[1] - rho_range[0]) - 1.0
        theta_norm = 2.0 * (points_cylindrical[:, 1] - theta_range[0]) / (theta_range[1] - theta_range[0]) - 1.0
        z_norm = 2.0 * (points_cylindrical[:, 2] - z_range[0]) / (z_range[1] - z_range[0]) - 1.0

        # Clamp to valid range
        rho_norm = torch.clamp(rho_norm, -1.0, 1.0)
        theta_norm = torch.clamp(theta_norm, -1.0, 1.0)
        z_norm = torch.clamp(z_norm, -1.0, 1.0)

        # grid_sample expects grid of shape (B, D_out, H_out, W_out, 3)
        # We treat N points as a 1D grid: (1, 1, 1, N, 3)
        grid = torch.stack([z_norm, theta_norm, rho_norm], dim=1)  # (N, 3) - note: grid_sample uses (x,y,z) = (W,H,D)
        grid = grid.view(1, 1, 1, N, 3)

        # grid_sample on 5D input: (B, C, D, H, W) with grid (B, D_out, H_out, W_out, 3)
        sampled = F.grid_sample(
            voxel_features_volume,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        # sampled shape: (B, C, 1, 1, N)
        point_voxel_features = sampled[0, :, 0, 0, :].t()  # (N, C)

        return point_voxel_features

    def forward(
        self,
        voxel_features_volume: torch.Tensor,
        point_features: torch.Tensor,
        point_to_voxel: torch.Tensor,
        voxel_coords: torch.Tensor,
        grid_size: list,
        points_cylindrical: Optional[torch.Tensor] = None,
        partition_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Refine per-point predictions using voxel features and point features.

        Args:
            voxel_features_volume: (B, C_voxel, D_rho, D_theta, D_z) backbone decoder features
            point_features: (N, C_point) original per-point features
            point_to_voxel: (N,) mapping from points to voxel indices
            voxel_coords: (M, 3) voxel grid coordinates
            grid_size: [D_rho, D_theta, D_z]
            points_cylindrical: (N, 3) cylindrical coords, required if use_interpolation=True
            partition_params: dict with range info, required if use_interpolation=True

        Returns:
            point_logits: (N, num_classes) per-point class logits
        """
        # Extract voxel features for each point
        if self.use_interpolation and points_cylindrical is not None and partition_params is not None:
            point_voxel_feats = self.extract_point_features_interpolated(
                voxel_features_volume,
                points_cylindrical,
                grid_size,
                partition_params["rho_range"],
                partition_params["theta_range"],
                partition_params["z_range"],
            )
        else:
            point_voxel_feats = self.extract_point_features_nearest(
                voxel_features_volume,
                point_to_voxel,
                voxel_coords,
                grid_size,
            )

        # Concatenate voxel features with original point features
        combined = torch.cat([point_voxel_feats, point_features], dim=1)

        # MLP for refined prediction
        out = self.mlp(combined)
        out = self.dropout(out)
        point_logits = self.classifier(out)

        return point_logits
