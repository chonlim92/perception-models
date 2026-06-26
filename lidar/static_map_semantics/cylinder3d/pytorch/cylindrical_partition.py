"""
Cylindrical Partition Module for Cylinder3D.

Converts raw LiDAR point clouds from Cartesian (x, y, z, intensity) to
cylindrical coordinates (rho, theta, z) and performs voxelization into a
discrete 3D grid for downstream processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict


class CylindricalPartition(nn.Module):
    """
    Converts point clouds to cylindrical coordinate voxel representations.

    This module:
    1. Transforms (x, y, z, intensity) points to cylindrical coords (rho, theta, z)
    2. Maps continuous cylindrical coords to discrete grid indices
    3. Aggregates point features within each voxel via mean pooling
    4. Computes enriched per-point features including relative positions

    Args:
        grid_size: Voxel grid dimensions [rho_bins, theta_bins, z_bins].
                   Default: [480, 360, 32]
        rho_range: Min and max radial distance [rho_min, rho_max].
                   Default: [0.0, 50.0]
        theta_range: Min and max azimuth angle [theta_min, theta_max].
                     Default: [-pi, pi]
        z_range: Min and max height [z_min, z_max].
                 Default: [-3.0, 1.0]
        point_feature_dim: Number of enriched per-point features computed.
                           Default: 9 (x, y, z, intensity, rho, delta_x, delta_y, delta_z, dist_to_center)
    """

    def __init__(
        self,
        grid_size: List[int] = None,
        rho_range: List[float] = None,
        theta_range: List[float] = None,
        z_range: List[float] = None,
    ):
        super().__init__()

        if grid_size is None:
            grid_size = [480, 360, 32]
        if rho_range is None:
            rho_range = [0.0, 50.0]
        if theta_range is None:
            theta_range = [-np.pi, np.pi]
        if z_range is None:
            z_range = [-3.0, 1.0]

        self.grid_size = grid_size
        self.rho_range = rho_range
        self.theta_range = theta_range
        self.z_range = z_range

        # Compute voxel sizes for each dimension
        self.rho_voxel_size = (rho_range[1] - rho_range[0]) / grid_size[0]
        self.theta_voxel_size = (theta_range[1] - theta_range[0]) / grid_size[1]
        self.z_voxel_size = (z_range[1] - z_range[0]) / grid_size[2]

        # Per-point feature dimension:
        # [x, y, z, intensity, rho, delta_x, delta_y, delta_z, dist_to_voxel_center]
        self.point_feature_dim = 9

    def cartesian_to_cylindrical(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to cylindrical coordinates.

        Args:
            points: (N, 4+) tensor with columns [x, y, z, intensity, ...]

        Returns:
            rho: (N,) radial distance sqrt(x^2 + y^2)
            theta: (N,) azimuth angle atan2(y, x), in [-pi, pi]
            z: (N,) height (unchanged from input)
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        rho = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)

        return rho, theta, z

    def coords_to_voxel_indices(
        self, rho: torch.Tensor, theta: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map continuous cylindrical coordinates to discrete voxel grid indices.

        Points outside the defined ranges are clamped to the boundary voxels.

        Args:
            rho: (N,) radial distances
            theta: (N,) azimuth angles
            z: (N,) heights

        Returns:
            voxel_indices: (N, 3) integer tensor of [rho_idx, theta_idx, z_idx]
            valid_mask: (N,) boolean tensor indicating points within bounds
        """
        # Clamp to valid ranges
        rho_clamped = torch.clamp(rho, self.rho_range[0], self.rho_range[1] - 1e-6)
        theta_clamped = torch.clamp(theta, self.theta_range[0], self.theta_range[1] - 1e-6)
        z_clamped = torch.clamp(z, self.z_range[0], self.z_range[1] - 1e-6)

        # Compute valid mask (points within range before clamping)
        valid_mask = (
            (rho >= self.rho_range[0]) & (rho < self.rho_range[1]) &
            (theta >= self.theta_range[0]) & (theta < self.theta_range[1]) &
            (z >= self.z_range[0]) & (z < self.z_range[1])
        )

        # Compute grid indices
        rho_idx = ((rho_clamped - self.rho_range[0]) / self.rho_voxel_size).long()
        theta_idx = ((theta_clamped - self.theta_range[0]) / self.theta_voxel_size).long()
        z_idx = ((z_clamped - self.z_range[0]) / self.z_voxel_size).long()

        # Ensure indices are within bounds
        rho_idx = torch.clamp(rho_idx, 0, self.grid_size[0] - 1)
        theta_idx = torch.clamp(theta_idx, 0, self.grid_size[1] - 1)
        z_idx = torch.clamp(z_idx, 0, self.grid_size[2] - 1)

        voxel_indices = torch.stack([rho_idx, theta_idx, z_idx], dim=1)

        return voxel_indices, valid_mask

    def compute_voxel_centers(self, voxel_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute the Cartesian center coordinates of each voxel.

        Args:
            voxel_indices: (N, 3) tensor of [rho_idx, theta_idx, z_idx]

        Returns:
            centers_xyz: (N, 3) tensor of [center_x, center_y, center_z]
        """
        # Compute cylindrical center of each voxel
        rho_center = (
            self.rho_range[0]
            + (voxel_indices[:, 0].float() + 0.5) * self.rho_voxel_size
        )
        theta_center = (
            self.theta_range[0]
            + (voxel_indices[:, 1].float() + 0.5) * self.theta_voxel_size
        )
        z_center = (
            self.z_range[0]
            + (voxel_indices[:, 2].float() + 0.5) * self.z_voxel_size
        )

        # Convert back to Cartesian
        center_x = rho_center * torch.cos(theta_center)
        center_y = rho_center * torch.sin(theta_center)
        center_z = z_center

        return torch.stack([center_x, center_y, center_z], dim=1)

    def compute_point_features(
        self,
        points: torch.Tensor,
        rho: torch.Tensor,
        voxel_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute enriched per-point features.

        Features: [x, y, z, intensity, rho, delta_x, delta_y, delta_z, dist_to_voxel_center]
        where delta_* is the offset from the point to the voxel center and
        dist_to_voxel_center is the Euclidean distance to the voxel center.

        Args:
            points: (N, 4+) raw point cloud [x, y, z, intensity, ...]
            rho: (N,) radial distances
            voxel_indices: (N, 3) voxel indices for each point

        Returns:
            features: (N, 9) enriched point features
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3]

        # Compute voxel centers in Cartesian space
        voxel_centers = self.compute_voxel_centers(voxel_indices)

        # Offset from point to voxel center
        delta_x = x - voxel_centers[:, 0]
        delta_y = y - voxel_centers[:, 1]
        delta_z = z - voxel_centers[:, 2]

        # Euclidean distance to voxel center
        dist_to_center = torch.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

        features = torch.stack(
            [x, y, z, intensity, rho, delta_x, delta_y, delta_z, dist_to_center],
            dim=1,
        )

        return features

    def aggregate_voxel_features(
        self,
        point_features: torch.Tensor,
        voxel_indices: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate point features within each voxel using mean pooling.

        Args:
            point_features: (N, C) per-point features
            voxel_indices: (N, 3) voxel indices for each point
            valid_mask: (N,) boolean mask of valid points

        Returns:
            voxel_features: (M, C) aggregated features for M unique voxels
            unique_voxel_coords: (M, 3) coordinates of unique voxels
            point_to_voxel: (N,) mapping from each point to its voxel index in [0, M)
            num_voxels: number of unique occupied voxels
        """
        device = point_features.device
        feat_dim = point_features.shape[1]

        # Filter valid points
        valid_point_features = point_features[valid_mask]
        valid_voxel_indices = voxel_indices[valid_mask]

        # Create a linear index for each voxel
        linear_indices = (
            valid_voxel_indices[:, 0] * (self.grid_size[1] * self.grid_size[2])
            + valid_voxel_indices[:, 1] * self.grid_size[2]
            + valid_voxel_indices[:, 2]
        )

        # Find unique voxels and map points to them
        unique_linear, inverse_indices = torch.unique(
            linear_indices, return_inverse=True
        )
        num_voxels = unique_linear.shape[0]

        # Mean pooling: sum features per voxel then divide by count
        voxel_features = torch.zeros(
            num_voxels, feat_dim, device=device, dtype=point_features.dtype
        )
        voxel_counts = torch.zeros(num_voxels, device=device, dtype=point_features.dtype)

        voxel_features.scatter_add_(
            0, inverse_indices.unsqueeze(1).expand(-1, feat_dim), valid_point_features
        )
        voxel_counts.scatter_add_(
            0, inverse_indices, torch.ones(inverse_indices.shape[0], device=device, dtype=point_features.dtype)
        )

        # Mean pooling
        voxel_features = voxel_features / voxel_counts.unsqueeze(1).clamp(min=1.0)

        # Recover 3D coordinates from linear indices
        unique_rho_idx = unique_linear // (self.grid_size[1] * self.grid_size[2])
        remainder = unique_linear % (self.grid_size[1] * self.grid_size[2])
        unique_theta_idx = remainder // self.grid_size[2]
        unique_z_idx = remainder % self.grid_size[2]
        unique_voxel_coords = torch.stack(
            [unique_rho_idx, unique_theta_idx, unique_z_idx], dim=1
        )

        # Build full point_to_voxel mapping (including invalid points mapped to -1)
        point_to_voxel = torch.full(
            (point_features.shape[0],), -1, device=device, dtype=torch.long
        )
        point_to_voxel[valid_mask] = inverse_indices

        return voxel_features, unique_voxel_coords, point_to_voxel, num_voxels

    def forward(
        self, points: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full cylindrical partition pipeline.

        Args:
            points: (N, 4+) point cloud tensor with [x, y, z, intensity, ...]

        Returns:
            Dictionary containing:
                - 'voxel_features': (M, C) aggregated features for M occupied voxels
                - 'voxel_coords': (M, 3) grid coordinates of occupied voxels
                - 'point_to_voxel': (N,) mapping from points to voxel indices
                - 'point_features': (N, 9) enriched per-point features
                - 'num_voxels': int, number of occupied voxels
                - 'grid_size': list, the voxel grid dimensions
        """
        # Step 1: Convert to cylindrical coordinates
        rho, theta, z = self.cartesian_to_cylindrical(points)

        # Step 2: Map to voxel indices
        voxel_indices, valid_mask = self.coords_to_voxel_indices(rho, theta, z)

        # Step 3: Compute enriched per-point features
        point_features = self.compute_point_features(points, rho, voxel_indices)

        # Step 4: Aggregate features within voxels
        voxel_features, voxel_coords, point_to_voxel, num_voxels = (
            self.aggregate_voxel_features(point_features, voxel_indices, valid_mask)
        )

        return {
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "point_to_voxel": point_to_voxel,
            "point_features": point_features,
            "num_voxels": num_voxels,
            "grid_size": self.grid_size,
        }
