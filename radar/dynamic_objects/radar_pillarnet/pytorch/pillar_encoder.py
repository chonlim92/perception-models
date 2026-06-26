"""
Pillar encoding with radar-specific features for RadarPillarNet.

Implements the pillar-based point cloud encoding from PointPillars, adapted for
radar-specific input features. Radar pillars are larger (0.4m) than LiDAR pillars
(0.16m) due to the significantly sparser point density of radar sensors.

Input features per point (9 dims):
    - x, y, z: Absolute coordinates in ego frame
    - RCS: Radar cross section (dBsm)
    - vr: Compensated radial velocity (m/s)
    - dt: Time delta from accumulation (s)
    - x_c, y_c, z_c: Offsets from pillar center (geometric mean of points in pillar)

The encoder uses a simplified PointNet (single linear layer + BN + ReLU + max pool)
to produce a fixed-size feature vector per pillar, which is then scattered to a
2D BEV pseudo-image for subsequent 2D convolution processing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def create_pillars(
    points: np.ndarray,
    point_range: List[float],
    pillar_size: List[float],
    max_points_per_pillar: int,
    max_pillars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize point cloud into pillars with radar-specific features.

    Assigns each point to a pillar (vertical column in BEV grid), then pads/truncates
    to fixed sizes. Augments raw features with offsets from the pillar geometric center.

    Args:
        points: (N, 7) array with columns [x, y, z, RCS, vr, dt, ...].
            Only the first 6 columns are used as raw features.
        point_range: [x_min, y_min, z_min, x_max, y_max, z_max] detection range.
        pillar_size: [dx, dy, dz] size of each pillar in meters.
        max_points_per_pillar: Maximum number of points kept per pillar (truncated).
        max_pillars: Maximum number of non-empty pillars to process.

    Returns:
        Tuple of:
            pillars: (max_pillars, max_points_per_pillar, 9) float32 array.
                Features: [x, y, z, RCS, vr, dt, x_c, y_c, z_c]
            pillar_indices: (max_pillars, 3) int32 array of (batch_idx, grid_x, grid_y).
                batch_idx is always 0 here (single sample).
            num_points_per_pillar: (max_pillars,) int32 array, actual point count per pillar.
    """
    x_min, y_min, z_min = point_range[0], point_range[1], point_range[2]
    x_max, y_max, z_max = point_range[3], point_range[4], point_range[5]
    dx, dy, dz = pillar_size[0], pillar_size[1], pillar_size[2]

    # Compute grid dimensions
    grid_x = int(np.round((x_max - x_min) / dx))
    grid_y = int(np.round((y_max - y_min) / dy))

    # Filter points within range
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] < z_max)
    )
    points = points[mask]

    if points.shape[0] == 0:
        pillars = np.zeros(
            (max_pillars, max_points_per_pillar, 9), dtype=np.float32
        )
        pillar_indices = np.zeros((max_pillars, 3), dtype=np.int32)
        num_points_per_pillar = np.zeros(max_pillars, dtype=np.int32)
        return pillars, pillar_indices, num_points_per_pillar

    # Compute grid indices for each point
    grid_idx_x = np.floor((points[:, 0] - x_min) / dx).astype(np.int32)
    grid_idx_y = np.floor((points[:, 1] - y_min) / dy).astype(np.int32)

    # Clip to valid range
    grid_idx_x = np.clip(grid_idx_x, 0, grid_x - 1)
    grid_idx_y = np.clip(grid_idx_y, 0, grid_y - 1)

    # Create unique pillar ID for each point
    pillar_ids = grid_idx_x * grid_y + grid_idx_y

    # Find unique pillars and their indices
    unique_pillars, inverse_indices = np.unique(pillar_ids, return_inverse=True)
    n_pillars = min(len(unique_pillars), max_pillars)

    # If too many pillars, randomly sample
    if len(unique_pillars) > max_pillars:
        selected_idx = np.random.choice(
            len(unique_pillars), max_pillars, replace=False
        )
        selected_idx.sort()
        unique_pillars = unique_pillars[selected_idx]
        # Create a mask for points belonging to selected pillars
        selected_set = set(unique_pillars.tolist())
        point_mask = np.array(
            [pillar_ids[i] in selected_set for i in range(len(pillar_ids))]
        )
        points = points[point_mask]
        grid_idx_x = grid_idx_x[point_mask]
        grid_idx_y = grid_idx_y[point_mask]
        pillar_ids = pillar_ids[point_mask]
        # Recompute
        unique_pillars, inverse_indices = np.unique(pillar_ids, return_inverse=True)
        n_pillars = len(unique_pillars)

    # Allocate output arrays
    pillars = np.zeros((max_pillars, max_points_per_pillar, 9), dtype=np.float32)
    pillar_indices = np.zeros((max_pillars, 3), dtype=np.int32)
    num_points_per_pillar = np.zeros(max_pillars, dtype=np.int32)

    # Fill pillars
    for p_idx in range(n_pillars):
        pillar_id = unique_pillars[p_idx]
        point_mask = pillar_ids == pillar_id
        pillar_points = points[point_mask]

        # Truncate if too many points
        n_pts = min(pillar_points.shape[0], max_points_per_pillar)
        if pillar_points.shape[0] > max_points_per_pillar:
            # Random subsample
            sel = np.random.choice(
                pillar_points.shape[0], max_points_per_pillar, replace=False
            )
            pillar_points = pillar_points[sel]
            n_pts = max_points_per_pillar

        # Extract raw features: [x, y, z, RCS, vr, dt]
        raw_feats = pillar_points[:n_pts, :6]  # (n_pts, 6)

        # Compute pillar center (geometric mean of occupied points)
        center_x = np.mean(raw_feats[:, 0])
        center_y = np.mean(raw_feats[:, 1])
        center_z = np.mean(raw_feats[:, 2])

        # Compute offsets from center
        offset_x = raw_feats[:, 0] - center_x  # x_c
        offset_y = raw_feats[:, 1] - center_y  # y_c
        offset_z = raw_feats[:, 2] - center_z  # z_c

        # Concatenate: [x, y, z, RCS, vr, dt, x_c, y_c, z_c]
        augmented = np.column_stack(
            [raw_feats, offset_x, offset_y, offset_z]
        )  # (n_pts, 9)

        pillars[p_idx, :n_pts, :] = augmented
        num_points_per_pillar[p_idx] = n_pts

        # Store grid coordinates
        # Recover grid x, y from pillar_id
        gx = pillar_id // grid_y
        gy = pillar_id % grid_y
        pillar_indices[p_idx] = [0, gx, gy]  # batch_idx=0

    return pillars, pillar_indices, num_points_per_pillar


class PillarEncoder(nn.Module):
    """Encodes radar pillars using a simplified PointNet architecture.

    Processes variable-length point sets within each pillar through a shared MLP
    (Linear -> BatchNorm -> ReLU) followed by max-pooling to produce a single
    feature vector per pillar.

    Configuration (radar-adapted):
        - Pillar size: [0.4, 0.4, 8.0] m (larger than LiDAR due to sparsity)
        - Point range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] m
        - Max points per pillar: 20
        - Max pillars: 12000
        - Input features: 9 (x, y, z, RCS, vr, dt, x_c, y_c, z_c)
        - Output features: 64 per pillar
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 64,
        x_range: Tuple[float, float] = (-51.2, 51.2),
        y_range: Tuple[float, float] = (-51.2, 51.2),
        z_range: Tuple[float, float] = (-5.0, 3.0),
        pillar_size: Tuple[float, float, float] = (0.4, 0.4, 8.0),
        max_points_per_pillar: int = 20,
        max_pillars: int = 12000,
    ) -> None:
        """Initialize pillar encoder.

        Args:
            in_channels: Number of input features per point (default 9).
            out_channels: Number of output channels per pillar (default 64).
            x_range: (x_min, x_max) detection range in meters.
            y_range: (y_min, y_max) detection range in meters.
            z_range: (z_min, z_max) detection range in meters.
            pillar_size: (dx, dy, dz) pillar dimensions in meters.
            max_points_per_pillar: Maximum points kept per pillar.
            max_pillars: Maximum number of non-empty pillars.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.pillar_size = pillar_size
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        # Compute grid dimensions
        self.grid_x = int(round((x_range[1] - x_range[0]) / pillar_size[0]))
        self.grid_y = int(round((y_range[1] - y_range[0]) / pillar_size[1]))

        # PointNet: Linear -> BN -> ReLU
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        pillars: torch.Tensor,
        num_points_per_pillar: torch.Tensor,
    ) -> torch.Tensor:
        """Encode pillars through PointNet.

        Args:
            pillars: (B, max_pillars, max_points_per_pillar, in_channels) tensor.
                Zero-padded point features within each pillar.
            num_points_per_pillar: (B, max_pillars) tensor with actual point counts.
                Used to create masks for valid points.

        Returns:
            (B, max_pillars, out_channels) tensor of pillar feature vectors.
        """
        batch_size = pillars.shape[0]
        # pillars: (B, P, N, C) where P=max_pillars, N=max_points_per_pillar, C=in_channels

        # Reshape for linear layer: (B*P*N, C)
        x = pillars.reshape(-1, self.in_channels)  # (B*P*N, 9)

        # Apply linear transformation
        x = self.linear(x)  # (B*P*N, 64)

        # Apply batch norm (operates on feature dim)
        x = self.bn(x)  # (B*P*N, 64)

        # Apply ReLU
        x = self.relu(x)  # (B*P*N, 64)

        # Reshape back: (B, P, N, out_channels)
        x = x.reshape(
            batch_size, self.max_pillars, self.max_points_per_pillar, self.out_channels
        )

        # Create mask for valid points: (B, P, N, 1)
        # Points beyond num_points_per_pillar should be zeroed before max pool
        device = pillars.device
        point_indices = torch.arange(
            self.max_points_per_pillar, device=device
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        thresholds = num_points_per_pillar.unsqueeze(-1)  # (B, P, 1)
        mask = point_indices < thresholds  # (B, P, N)
        mask = mask.unsqueeze(-1)  # (B, P, N, 1)

        # Apply mask (set padded positions to large negative for max pool)
        x = x.masked_fill(~mask, float("-inf"))

        # Max pooling over points dimension: (B, P, out_channels)
        x, _ = x.max(dim=2)  # (B, P, 64)

        # Handle empty pillars (all -inf after max -> set to 0)
        empty_mask = num_points_per_pillar == 0  # (B, P)
        x[empty_mask] = 0.0

        return x  # (B, max_pillars, 64)

    def get_output_shape(self) -> Tuple[int, int, int]:
        """Return the BEV grid shape (channels, height, width).

        Returns:
            Tuple of (out_channels, grid_x, grid_y).
        """
        return (self.out_channels, self.grid_x, self.grid_y)


class PillarScatter(nn.Module):
    """Scatters pillar features to a 2D BEV pseudo-image.

    Takes encoded pillar features and their grid indices, and places them
    into a dense 2D grid (Bird's Eye View). Empty cells remain zero.
    This converts the sparse pillar representation into a regular grid
    that can be processed by standard 2D convolutions.
    """

    def __init__(
        self,
        in_channels: int = 64,
        grid_x: int = 256,
        grid_y: int = 256,
    ) -> None:
        """Initialize pillar scatter module.

        Args:
            in_channels: Number of channels per pillar feature vector.
            grid_x: Number of grid cells in x direction (height of BEV image).
            grid_y: Number of grid cells in y direction (width of BEV image).
        """
        super().__init__()
        self.in_channels = in_channels
        self.grid_x = grid_x
        self.grid_y = grid_y

    def forward(
        self,
        pillar_features: torch.Tensor,
        pillar_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Scatter pillar features to BEV grid.

        Args:
            pillar_features: (B, max_pillars, C) encoded pillar features.
            pillar_indices: (B, max_pillars, 3) grid coordinates [batch_idx, x, y].
                Note: batch_idx in column 0 is relative within the batch.
            batch_size: Number of samples in the batch.

        Returns:
            (B, C, grid_x, grid_y) dense BEV pseudo-image tensor.
        """
        device = pillar_features.device
        dtype = pillar_features.dtype

        # Initialize output canvas
        canvas = torch.zeros(
            (batch_size, self.in_channels, self.grid_x, self.grid_y),
            dtype=dtype,
            device=device,
        )  # (B, C, H, W)

        for b in range(batch_size):
            # Get features and indices for this batch element
            feats = pillar_features[b]  # (max_pillars, C)
            indices = pillar_indices[b]  # (max_pillars, 3)

            # Extract grid coordinates (ignore batch_idx column)
            gx = indices[:, 1].long()  # (max_pillars,)
            gy = indices[:, 2].long()  # (max_pillars,)

            # Filter valid pillars (non-zero indices or non-zero features)
            # A valid pillar has at least one non-zero feature
            valid_mask = feats.abs().sum(dim=-1) > 0  # (max_pillars,)

            # Also ensure indices are within bounds
            valid_mask &= (gx >= 0) & (gx < self.grid_x)
            valid_mask &= (gy >= 0) & (gy < self.grid_y)

            if valid_mask.sum() == 0:
                continue

            valid_feats = feats[valid_mask]  # (V, C)
            valid_gx = gx[valid_mask]  # (V,)
            valid_gy = gy[valid_mask]  # (V,)

            # Scatter features to canvas
            # canvas[b, :, gx, gy] = feat
            canvas[b, :, valid_gx, valid_gy] = valid_feats.T  # (C, V)

        return canvas  # (B, C, H, W)
