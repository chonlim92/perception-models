"""
PillarFeatureNet: Core feature extraction network from PointPillars.

Reference:
    Lang, A.H., Vora, S., Caesar, H., Zhou, L., Yang, J., & Beijbom, O. (2019).
    PointPillars: Fast Encoders for Object Detection from Point Clouds.
    CVPR 2019.

This module implements the pillar-based feature extraction that converts raw
LiDAR point clouds into a pseudo-image representation suitable for 2D
convolutional backbones.
"""

from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import numpy as np


class PillarFeatureNet(nn.Module):
    """Pillar Feature Network for PointPillars.

    Converts a raw point cloud into a set of pillar-wise feature vectors by:
    1. Voxelizing the point cloud into vertical pillars on an x-y grid.
    2. Augmenting each point's features with geometric offsets.
    3. Applying a simplified PointNet (shared MLP + max pooling) per pillar.

    Args:
        in_channels: Number of raw input features per point (default: 4 for x, y, z, intensity).
        out_channels: Number of output feature channels per pillar (default: 64).
        pillar_size: Pillar dimensions in meters [dx, dy] (default: [0.16, 0.16]).
        x_range: Point cloud x-axis range [min, max] in meters (default: [-39.68, 39.68]).
        y_range: Point cloud y-axis range [min, max] in meters (default: [0, 69.12]).
        z_range: Point cloud z-axis range [min, max] in meters (default: [-3, 1]).
        max_points_per_pillar: Maximum number of points sampled per pillar (default: 32).
        max_pillars: Maximum number of non-empty pillars to retain (default: 12000).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 64,
        pillar_size: List[float] = None,
        x_range: List[float] = None,
        y_range: List[float] = None,
        z_range: List[float] = None,
        max_points_per_pillar: int = 32,
        max_pillars: int = 12000,
    ) -> None:
        super().__init__()

        if pillar_size is None:
            pillar_size = [0.16, 0.16]
        if x_range is None:
            x_range = [-39.68, 39.68]
        if y_range is None:
            y_range = [0.0, 69.12]
        if z_range is None:
            z_range = [-3.0, 1.0]

        self.pillar_size = pillar_size
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        # Compute grid dimensions
        self.grid_x_size = int(round((x_range[1] - x_range[0]) / pillar_size[0]))
        self.grid_y_size = int(round((y_range[1] - y_range[0]) / pillar_size[1]))

        # Augmented feature dimension: original (x, y, z, intensity) + offsets (xc, yc, zc, xp, yp)
        # where xc/yc/zc = offset from pillar arithmetic mean, xp/yp = offset from pillar geometric center
        augmented_channels = in_channels + 5  # 9 total

        # Shared MLP: Linear -> BatchNorm -> ReLU (applied point-wise within each pillar)
        self.linear = nn.Linear(augmented_channels, out_channels, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.out_channels = out_channels

    def voxelize(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Divide raw point cloud into pillars on a 2D x-y grid.

        Each pillar is a vertical column of unlimited height. Points are assigned
        to pillars based on their x-y coordinates. If a pillar has more points
        than max_points_per_pillar, points are randomly sampled. If fewer, the
        pillar is zero-padded.

        Args:
            points: Raw point cloud tensor of shape (N, 4) with columns [x, y, z, intensity].
                    Must reside on the same device as the module.

        Returns:
            pillar_features: Tensor of shape (P, N_max, C) where P is the number of
                non-empty pillars (capped at max_pillars), N_max is max_points_per_pillar,
                and C is the augmented feature dimension (9).
            pillar_coords: Tensor of shape (P, 2) containing the [grid_x_idx, grid_y_idx]
                integer coordinates for each pillar.
            num_points_per_pillar: Tensor of shape (P,) with the actual number of points
                in each pillar (before padding, after sampling).
        """
        device = points.device
        N = points.shape[0]

        # Filter points within the configured range
        x_mask = (points[:, 0] >= self.x_range[0]) & (points[:, 0] < self.x_range[1])
        y_mask = (points[:, 1] >= self.y_range[0]) & (points[:, 1] < self.y_range[1])
        z_mask = (points[:, 2] >= self.z_range[0]) & (points[:, 2] < self.z_range[1])
        valid_mask = x_mask & y_mask & z_mask
        points = points[valid_mask]

        N_valid = points.shape[0]

        if N_valid == 0:
            # Return empty tensors with correct shapes
            pillar_features = torch.zeros(
                (0, self.max_points_per_pillar, 9), dtype=points.dtype, device=device
            )
            pillar_coords = torch.zeros((0, 2), dtype=torch.long, device=device)
            num_points_per_pillar = torch.zeros((0,), dtype=torch.long, device=device)
            return pillar_features, pillar_coords, num_points_per_pillar

        # Compute grid indices for each point
        grid_x_idx = ((points[:, 0] - self.x_range[0]) / self.pillar_size[0]).long()
        grid_y_idx = ((points[:, 1] - self.y_range[0]) / self.pillar_size[1]).long()

        # Clamp to valid grid range (handles floating point edge cases)
        grid_x_idx = grid_x_idx.clamp(0, self.grid_x_size - 1)
        grid_y_idx = grid_y_idx.clamp(0, self.grid_y_size - 1)

        # Create a unique linear index for each pillar
        pillar_linear_idx = grid_y_idx * self.grid_x_size + grid_x_idx

        # Find unique pillars and map each point to its pillar
        unique_pillars, inverse_indices = torch.unique(pillar_linear_idx, return_inverse=True)
        num_unique_pillars = unique_pillars.shape[0]

        # If there are more non-empty pillars than max_pillars, randomly sample
        if num_unique_pillars > self.max_pillars:
            perm = torch.randperm(num_unique_pillars, device=device)[: self.max_pillars]
            selected_pillars = unique_pillars[perm]
            # Create a mask for points belonging to selected pillars
            # Build a set membership check using a boolean tensor
            pillar_selected_mask = torch.zeros(
                num_unique_pillars, dtype=torch.bool, device=device
            )
            pillar_selected_mask[perm] = True
            point_keep_mask = pillar_selected_mask[inverse_indices]
            points = points[point_keep_mask]
            pillar_linear_idx = pillar_linear_idx[point_keep_mask]
            # Recompute unique pillars after filtering
            unique_pillars, inverse_indices = torch.unique(
                pillar_linear_idx, return_inverse=True
            )
            num_unique_pillars = unique_pillars.shape[0]

        P = num_unique_pillars

        # Recover 2D grid coordinates from linear index
        pillar_grid_y = unique_pillars // self.grid_x_size
        pillar_grid_x = unique_pillars % self.grid_x_size
        pillar_coords = torch.stack([pillar_grid_x, pillar_grid_y], dim=1)  # (P, 2)

        # Compute pillar geometric centers in metric coordinates
        pillar_center_x = (pillar_grid_x.float() + 0.5) * self.pillar_size[0] + self.x_range[0]
        pillar_center_y = (pillar_grid_y.float() + 0.5) * self.pillar_size[1] + self.y_range[0]

        # Allocate output tensors
        pillar_features = torch.zeros(
            (P, self.max_points_per_pillar, 9), dtype=points.dtype, device=device
        )
        num_points_per_pillar = torch.zeros((P,), dtype=torch.long, device=device)

        # Group points into pillars with sampling/padding
        # For each unique pillar, gather its points, sample if needed, and compute augmented features
        for pillar_idx in range(P):
            # Get all points belonging to this pillar
            point_mask = inverse_indices == pillar_idx
            pillar_points = points[point_mask]  # (K, 4)
            K = pillar_points.shape[0]

            # Sample or pad
            if K > self.max_points_per_pillar:
                sample_indices = torch.randperm(K, device=device)[: self.max_points_per_pillar]
                pillar_points = pillar_points[sample_indices]
                actual_count = self.max_points_per_pillar
            else:
                actual_count = K

            num_points_per_pillar[pillar_idx] = actual_count

            # Compute arithmetic mean of the points in this pillar
            mean_x = pillar_points[:actual_count, 0].mean()
            mean_y = pillar_points[:actual_count, 1].mean()
            mean_z = pillar_points[:actual_count, 2].mean()

            # Compute offsets from arithmetic mean (xc, yc, zc)
            xc = pillar_points[:actual_count, 0] - mean_x
            yc = pillar_points[:actual_count, 1] - mean_y
            zc = pillar_points[:actual_count, 2] - mean_z

            # Compute offsets from pillar geometric center (xp, yp)
            xp = pillar_points[:actual_count, 0] - pillar_center_x[pillar_idx]
            yp = pillar_points[:actual_count, 1] - pillar_center_y[pillar_idx]

            # Assemble augmented features: [x, y, z, intensity, xc, yc, zc, xp, yp]
            augmented = torch.stack(
                [
                    pillar_points[:actual_count, 0],
                    pillar_points[:actual_count, 1],
                    pillar_points[:actual_count, 2],
                    pillar_points[:actual_count, 3],
                    xc,
                    yc,
                    zc,
                    xp,
                    yp,
                ],
                dim=1,
            )  # (actual_count, 9)

            pillar_features[pillar_idx, :actual_count, :] = augmented

        return pillar_features, pillar_coords, num_points_per_pillar

    def forward(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract pillar features from a raw point cloud.

        Performs voxelization, feature augmentation, shared MLP encoding,
        and max pooling to produce one feature vector per non-empty pillar.

        Args:
            points: Raw point cloud of shape (N, 4) with columns [x, y, z, intensity].

        Returns:
            pillar_features: Tensor of shape (P, out_channels) containing the extracted
                feature vector for each non-empty pillar.
            pillar_coords: Tensor of shape (P, 2) containing the [grid_x_idx, grid_y_idx]
                for each pillar, useful for scattering features back to a pseudo-image.
        """
        # Step 1: Voxelize the point cloud into pillars with augmented features
        pillar_data, pillar_coords, num_points = self.voxelize(points)
        # pillar_data: (P, N_max, 9), pillar_coords: (P, 2), num_points: (P,)

        P = pillar_data.shape[0]

        if P == 0:
            empty_features = torch.zeros(
                (0, self.out_channels), dtype=points.dtype, device=points.device
            )
            empty_coords = torch.zeros((0, 2), dtype=torch.long, device=points.device)
            return empty_features, empty_coords

        N_max = self.max_points_per_pillar

        # Step 2: Apply shared MLP (Linear + BN + ReLU) to each point
        # Reshape to (P * N_max, 9) for efficient batch processing
        flat_features = pillar_data.reshape(P * N_max, -1)  # (P * N_max, 9)
        flat_features = self.linear(flat_features)  # (P * N_max, out_channels)
        # BatchNorm1d expects (batch, features) when input is 2D
        flat_features = self.batch_norm(flat_features)  # (P * N_max, out_channels)
        flat_features = self.relu(flat_features)  # (P * N_max, out_channels)

        # Reshape back to (P, N_max, out_channels)
        pillar_encoded = flat_features.reshape(P, N_max, self.out_channels)

        # Step 3: Create a mask to zero out padded positions before max pooling
        # Build a mask of shape (P, N_max, 1): True for valid points, False for padding
        point_indices = torch.arange(N_max, device=pillar_data.device).unsqueeze(0)  # (1, N_max)
        valid_mask = point_indices < num_points.unsqueeze(1)  # (P, N_max)
        valid_mask = valid_mask.unsqueeze(2)  # (P, N_max, 1)

        # Apply mask: set padded positions to very large negative so they don't affect max
        pillar_encoded = pillar_encoded.masked_fill(~valid_mask, float("-inf"))

        # Step 4: Max pooling across points dimension to get one vector per pillar
        pillar_output, _ = pillar_encoded.max(dim=1)  # (P, out_channels)

        # Handle pillars that might have all -inf (shouldn't happen if num_points > 0)
        # Replace any remaining -inf with 0
        pillar_output = pillar_output.masked_fill(
            pillar_output == float("-inf"), 0.0
        )

        return pillar_output, pillar_coords

    def create_pseudo_image(
        self,
        pillar_features: torch.Tensor,
        pillar_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter pillar features onto a 2D pseudo-image grid.

        This is a convenience method that places pillar features at their
        corresponding spatial locations to form a dense pseudo-image that
        can be processed by standard 2D CNN backbones.

        Args:
            pillar_features: Tensor of shape (P, C) with per-pillar feature vectors.
            pillar_coords: Tensor of shape (P, 2) with [grid_x_idx, grid_y_idx] coordinates.

        Returns:
            pseudo_image: Tensor of shape (1, C, grid_y_size, grid_x_size) in standard
                (batch, channels, height, width) format for CNN processing.
        """
        C = pillar_features.shape[1]
        device = pillar_features.device
        dtype = pillar_features.dtype

        pseudo_image = torch.zeros(
            (1, C, self.grid_y_size, self.grid_x_size), dtype=dtype, device=device
        )

        if pillar_coords.shape[0] > 0:
            x_indices = pillar_coords[:, 0].long()
            y_indices = pillar_coords[:, 1].long()
            # Scatter features: pseudo_image[0, :, y, x] = feature_vector
            pseudo_image[0, :, y_indices, x_indices] = pillar_features.t()

        return pseudo_image


class PillarFeatureNetBatch(nn.Module):
    """Batch-aware wrapper around PillarFeatureNet.

    Handles batched point clouds where each sample in the batch is represented
    separately, producing a batch of pseudo-images ready for backbone processing.

    Args:
        in_channels: Number of raw input features per point (default: 4).
        out_channels: Number of output feature channels per pillar (default: 64).
        pillar_size: Pillar dimensions in meters [dx, dy] (default: [0.16, 0.16]).
        x_range: Point cloud x-axis range [min, max] in meters.
        y_range: Point cloud y-axis range [min, max] in meters.
        z_range: Point cloud z-axis range [min, max] in meters.
        max_points_per_pillar: Maximum number of points per pillar (default: 32).
        max_pillars: Maximum number of pillars per sample (default: 12000).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 64,
        pillar_size: List[float] = None,
        x_range: List[float] = None,
        y_range: List[float] = None,
        z_range: List[float] = None,
        max_points_per_pillar: int = 32,
        max_pillars: int = 12000,
    ) -> None:
        super().__init__()

        self.pfn = PillarFeatureNet(
            in_channels=in_channels,
            out_channels=out_channels,
            pillar_size=pillar_size,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
        )

    @property
    def out_channels(self) -> int:
        """Output feature dimensionality."""
        return self.pfn.out_channels

    @property
    def grid_x_size(self) -> int:
        """Number of grid cells along the x-axis."""
        return self.pfn.grid_x_size

    @property
    def grid_y_size(self) -> int:
        """Number of grid cells along the y-axis."""
        return self.pfn.grid_y_size

    def forward(
        self,
        batch_points: List[torch.Tensor],
    ) -> torch.Tensor:
        """Process a batch of point clouds into pseudo-images.

        Args:
            batch_points: List of B tensors, each of shape (N_i, 4), representing
                individual point cloud samples in the batch.

        Returns:
            pseudo_images: Tensor of shape (B, out_channels, grid_y_size, grid_x_size)
                representing the batch of pseudo-images.
        """
        batch_size = len(batch_points)
        device = batch_points[0].device
        dtype = batch_points[0].dtype

        pseudo_images = torch.zeros(
            (batch_size, self.out_channels, self.grid_y_size, self.grid_x_size),
            dtype=dtype,
            device=device,
        )

        for i, points in enumerate(batch_points):
            pillar_features, pillar_coords = self.pfn(points)
            if pillar_coords.shape[0] > 0:
                x_indices = pillar_coords[:, 0].long()
                y_indices = pillar_coords[:, 1].long()
                pseudo_images[i, :, y_indices, x_indices] = pillar_features.t()

        return pseudo_images
