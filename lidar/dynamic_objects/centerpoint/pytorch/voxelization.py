"""
CenterPoint Voxelization Module.

Implements dynamic voxelization and pillar feature extraction for converting
raw point clouds into structured voxel/pillar representations.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


def points_to_voxel(
    points: torch.Tensor,
    voxel_size: List[float],
    point_cloud_range: List[float],
    max_voxels: int = 30000,
    max_points_per_voxel: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign points to voxels using scatter operations.

    Args:
        points: (N, C) tensor of point cloud features (x, y, z, intensity, ...).
        voxel_size: [vx, vy, vz] voxel dimensions in meters.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        max_voxels: Maximum number of voxels to keep.
        max_points_per_voxel: Maximum points per voxel (for padding, not used in dynamic mode).

    Returns:
        voxel_features: (M, C) mean features for each occupied voxel.
        voxel_coords: (M, 3) integer voxel coordinates (z, y, x order for sparse conv).
        voxel_num_points: (M,) number of points in each voxel.
    """
    device = points.device
    dtype = points.dtype

    voxel_size_tensor = torch.tensor(voxel_size, dtype=dtype, device=device)
    range_min = torch.tensor(point_cloud_range[:3], dtype=dtype, device=device)
    range_max = torch.tensor(point_cloud_range[3:], dtype=dtype, device=device)

    # Compute grid dimensions
    grid_size = ((range_max - range_min) / voxel_size_tensor).round().long()

    # Filter points outside the range
    mask = (
        (points[:, 0] >= range_min[0]) & (points[:, 0] < range_max[0]) &
        (points[:, 1] >= range_min[1]) & (points[:, 1] < range_max[1]) &
        (points[:, 2] >= range_min[2]) & (points[:, 2] < range_max[2])
    )
    points = points[mask]

    # Compute voxel indices for each point
    coords = ((points[:, :3] - range_min) / voxel_size_tensor).long()
    # Clamp to valid range
    coords[:, 0] = coords[:, 0].clamp(0, grid_size[0] - 1)
    coords[:, 1] = coords[:, 1].clamp(0, grid_size[1] - 1)
    coords[:, 2] = coords[:, 2].clamp(0, grid_size[2] - 1)

    # Linearize voxel indices for scatter
    # Use z, y, x ordering consistent with sparse convolution conventions
    linear_idx = (
        coords[:, 2] * (grid_size[1] * grid_size[0]) +
        coords[:, 1] * grid_size[0] +
        coords[:, 0]
    )

    # Get unique voxels and inverse mapping
    unique_indices, inverse_map, counts = torch.unique(
        linear_idx, return_inverse=True, return_counts=True
    )

    num_voxels = unique_indices.shape[0]

    # Limit number of voxels
    if num_voxels > max_voxels:
        # Keep voxels with more points (or just truncate)
        keep = torch.argsort(counts, descending=True)[:max_voxels]
        keep_mask = torch.zeros(num_voxels, dtype=torch.bool, device=device)
        keep_mask[keep] = True
        # Filter points belonging to kept voxels
        point_keep_mask = keep_mask[inverse_map]
        points = points[point_keep_mask]
        # Recompute
        linear_idx = linear_idx[mask][point_keep_mask] if False else linear_idx[point_keep_mask]
        unique_indices, inverse_map, counts = torch.unique(
            linear_idx, return_inverse=True, return_counts=True
        )
        num_voxels = unique_indices.shape[0]

    # Scatter mean: compute mean features for each voxel
    num_features = points.shape[1]
    voxel_features = torch.zeros(
        num_voxels, num_features, dtype=dtype, device=device
    )
    # Sum features per voxel
    voxel_features.scatter_add_(
        0, inverse_map.unsqueeze(1).expand(-1, num_features), points
    )
    # Divide by counts to get mean
    voxel_features = voxel_features / counts.unsqueeze(1).float()

    # Convert linear indices back to 3D coordinates (z, y, x order)
    voxel_z = unique_indices // (grid_size[1] * grid_size[0])
    remainder = unique_indices % (grid_size[1] * grid_size[0])
    voxel_y = remainder // grid_size[0]
    voxel_x = remainder % grid_size[0]

    # Sparse conv convention: (z, y, x)
    voxel_coords = torch.stack([voxel_z, voxel_y, voxel_x], dim=1).int()

    voxel_num_points = counts.int()

    return voxel_features, voxel_coords, voxel_num_points


class DynamicVoxelization(nn.Module):
    """Dynamic voxelization that assigns each point to a voxel without
    fixed-size buffers. Uses scatter operations for efficiency.

    Args:
        voxel_size: [vx, vy, vz] voxel dimensions in meters.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        max_voxels: Maximum number of voxels to retain.
        max_points_per_voxel: Maximum points per voxel (soft limit for dynamic mode).
    """

    def __init__(
        self,
        voxel_size: List[float] = [0.075, 0.075, 0.2],
        point_cloud_range: List[float] = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        max_voxels: int = 30000,
        max_points_per_voxel: int = 20,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_voxels = max_voxels
        self.max_points_per_voxel = max_points_per_voxel

        # Compute grid size
        range_min = torch.tensor(point_cloud_range[:3])
        range_max = torch.tensor(point_cloud_range[3:])
        voxel_size_t = torch.tensor(voxel_size)
        self.grid_size = ((range_max - range_min) / voxel_size_t).round().long().tolist()

    def forward(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 4+) raw point cloud [x, y, z, intensity, ...].

        Returns:
            voxel_features: (M, C) mean point features per voxel.
            voxel_coords: (M, 3) integer voxel coordinates (z, y, x).
            voxel_num_points: (M,) count of points per voxel.
        """
        return points_to_voxel(
            points,
            self.voxel_size,
            self.point_cloud_range,
            self.max_voxels,
            self.max_points_per_voxel,
        )

    def forward_batch(
        self, batch_points: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a batch of point clouds.

        Args:
            batch_points: List of (Ni, 4+) tensors, one per sample.

        Returns:
            voxel_features: (M_total, C) features for all voxels in the batch.
            voxel_coords: (M_total, 4) coordinates with batch index prepended: (batch_id, z, y, x).
            voxel_num_points: (M_total,) point counts.
        """
        all_features = []
        all_coords = []
        all_num_points = []

        for batch_idx, points in enumerate(batch_points):
            feats, coords, num_pts = self.forward(points)
            # Prepend batch index to coordinates
            batch_col = torch.full(
                (coords.shape[0], 1), batch_idx,
                dtype=coords.dtype, device=coords.device
            )
            coords_with_batch = torch.cat([batch_col, coords], dim=1)

            all_features.append(feats)
            all_coords.append(coords_with_batch)
            all_num_points.append(num_pts)

        voxel_features = torch.cat(all_features, dim=0)
        voxel_coords = torch.cat(all_coords, dim=0)
        voxel_num_points = torch.cat(all_num_points, dim=0)

        return voxel_features, voxel_coords, voxel_num_points


class PillarFeatureExtraction(nn.Module):
    """Pillar-based feature extraction (PointPillars variant).

    Groups points into vertical pillars, computes augmented features
    (original features + offsets from pillar center + offsets from point cloud center),
    and applies a PointNet-style shared MLP with max pooling per pillar.

    Args:
        in_channels: Number of input point features (default: 4 for x,y,z,r).
        feat_channels: List of MLP hidden dimensions.
        voxel_size: [vx, vy] pillar dimensions in x, y (z is full range).
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        max_points_per_pillar: Maximum points to keep per pillar.
        max_pillars: Maximum number of pillars.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: List[int] = [64],
        voxel_size: List[float] = [0.2, 0.2, 8.0],
        point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_points_per_pillar: int = 32,
        max_pillars: int = 30000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        # Augmented feature dimension:
        # original (in_channels) + offset from pillar center (3) + offset from PC center (3)
        augmented_channels = in_channels + 6

        # Build PointNet MLP
        layers = []
        prev_channels = augmented_channels
        for out_ch in feat_channels:
            layers.append(nn.Linear(prev_channels, out_ch, bias=False))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            prev_channels = out_ch
        self.pointnet = nn.ModuleList(layers)
        self.out_channels = feat_channels[-1]

        # Compute grid size (only X, Y matter for pillars)
        range_min = torch.tensor(point_cloud_range[:3])
        range_max = torch.tensor(point_cloud_range[3:])
        voxel_size_t = torch.tensor(voxel_size)
        self.grid_size = ((range_max - range_min) / voxel_size_t).round().long().tolist()

    def forward(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 4+) raw point features [x, y, z, intensity, ...].

        Returns:
            pillar_features: (M, C) output features per pillar after PointNet.
            pillar_coords: (M, 2) integer pillar coordinates (y, x).
        """
        device = points.device
        dtype = points.dtype

        voxel_size_tensor = torch.tensor(self.voxel_size[:2], dtype=dtype, device=device)
        range_min = torch.tensor(self.point_cloud_range[:2], dtype=dtype, device=device)
        range_max = torch.tensor(self.point_cloud_range[3:5], dtype=dtype, device=device)
        pc_center = torch.tensor(
            [(self.point_cloud_range[0] + self.point_cloud_range[3]) / 2.0,
             (self.point_cloud_range[1] + self.point_cloud_range[4]) / 2.0,
             (self.point_cloud_range[2] + self.point_cloud_range[5]) / 2.0],
            dtype=dtype, device=device
        )

        # Filter points in range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0]) &
            (points[:, 0] < self.point_cloud_range[3]) &
            (points[:, 1] >= self.point_cloud_range[1]) &
            (points[:, 1] < self.point_cloud_range[4]) &
            (points[:, 2] >= self.point_cloud_range[2]) &
            (points[:, 2] < self.point_cloud_range[5])
        )
        points = points[mask]

        # Compute pillar indices (2D grid)
        pillar_coords_float = (points[:, :2] - range_min) / voxel_size_tensor
        pillar_indices = pillar_coords_float.long()
        grid_x = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0])
        grid_y = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1])
        pillar_indices[:, 0] = pillar_indices[:, 0].clamp(0, grid_x - 1)
        pillar_indices[:, 1] = pillar_indices[:, 1].clamp(0, grid_y - 1)

        # Linearize pillar index
        linear_idx = pillar_indices[:, 1] * grid_x + pillar_indices[:, 0]

        # Get unique pillars
        unique_pillars, inverse_map, counts = torch.unique(
            linear_idx, return_inverse=True, return_counts=True
        )
        num_pillars = unique_pillars.shape[0]

        # Limit pillars
        if num_pillars > self.max_pillars:
            keep = torch.argsort(counts, descending=True)[:self.max_pillars]
            keep_mask = torch.zeros(num_pillars, dtype=torch.bool, device=device)
            keep_mask[keep] = True
            point_keep_mask = keep_mask[inverse_map]
            points = points[point_keep_mask]
            linear_idx = linear_idx[point_keep_mask]
            unique_pillars, inverse_map, counts = torch.unique(
                linear_idx, return_inverse=True, return_counts=True
            )
            num_pillars = unique_pillars.shape[0]

        # Compute pillar centers (mean of x, y, z per pillar)
        pillar_centers = torch.zeros(num_pillars, 3, dtype=dtype, device=device)
        pillar_centers.scatter_add_(
            0,
            inverse_map.unsqueeze(1).expand(-1, 3),
            points[:, :3]
        )
        pillar_centers = pillar_centers / counts.unsqueeze(1).float()

        # Compute augmented features for each point
        # offset from pillar center
        offset_pillar = points[:, :3] - pillar_centers[inverse_map]
        # offset from point cloud center
        offset_pc = points[:, :3] - pc_center.unsqueeze(0)

        # Augmented features: [original_features, offset_from_pillar_center, offset_from_pc_center]
        augmented = torch.cat([
            points[:, :self.in_channels],
            offset_pillar,
            offset_pc,
        ], dim=1)  # (N_filtered, in_channels + 6)

        # Apply PointNet MLP per point
        x = augmented
        for i, layer in enumerate(self.pointnet):
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            elif isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)

        # Max pool per pillar using scatter_reduce
        pillar_features = torch.zeros(
            num_pillars, self.out_channels, dtype=dtype, device=device
        )
        # Use scatter with max reduction
        expanded_inverse = inverse_map.unsqueeze(1).expand(-1, self.out_channels)
        # Initialize with very negative values for max
        pillar_features.fill_(float('-inf'))
        pillar_features.scatter_reduce_(
            0, expanded_inverse, x, reduce='amax', include_self=False
        )
        # Replace -inf with 0 for empty slots (shouldn't happen but safety)
        pillar_features[pillar_features == float('-inf')] = 0.0

        # Convert linear indices back to 2D coords (y, x)
        pillar_y = unique_pillars // grid_x
        pillar_x = unique_pillars % grid_x
        pillar_coords = torch.stack([pillar_y, pillar_x], dim=1).int()

        return pillar_features, pillar_coords

    def forward_to_bev(
        self, points: torch.Tensor
    ) -> torch.Tensor:
        """Generate a dense BEV pseudo-image from pillar features.

        Args:
            points: (N, 4+) raw point cloud.

        Returns:
            bev_image: (1, C, H, W) dense BEV feature map.
        """
        pillar_features, pillar_coords = self.forward(points)
        device = pillar_features.device

        grid_x = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0])
        grid_y = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1])

        bev = torch.zeros(
            1, self.out_channels, grid_y, grid_x,
            dtype=pillar_features.dtype, device=device
        )
        # Place pillar features into BEV grid
        y_idx = pillar_coords[:, 0].long()
        x_idx = pillar_coords[:, 1].long()
        bev[0, :, y_idx, x_idx] = pillar_features.t()

        return bev
