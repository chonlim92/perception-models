"""
PointNet++ Set Abstraction and Feature Propagation modules.

These are the core building blocks of PointNet++:
- PointNetSetAbstraction: Single-scale grouping (SSG)
- PointNetSetAbstractionMsg: Multi-scale grouping (MSG)
- PointNetFeaturePropagation: Feature upsampling via interpolation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampling import (
    farthest_point_sampling,
    ball_query,
    index_points,
    square_distance,
)


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction layer with single-scale grouping.

    Pipeline: FPS -> Ball Query -> Group Points -> Shared MLP -> Max Pool

    Args:
        npoint: Number of points to sample with FPS (output resolution).
                If None, aggregates all points (global feature).
        radius: Ball query radius
        nsample: Max number of points in each ball query group
        in_channel: Number of input feature channels (including xyz if
                    features are concatenated with coordinates)
        mlp: List of output channel sizes for the shared MLP layers
        group_all: If True, group all points into a single set (ignores
                   npoint, radius, nsample)
    """

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list,
        group_all: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # Build shared MLP as sequence of Conv1d + BN + ReLU
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor,
    ) -> tuple:
        """
        Forward pass.

        Args:
            xyz: Point coordinates, shape (B, N, 3)
            points: Point features, shape (B, N, C) or None

        Returns:
            new_xyz: Sampled point coordinates, shape (B, npoint, 3)
            new_points: Features for sampled points, shape (B, npoint, D)
        """
        if self.group_all:
            new_xyz, new_points = self._sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self._sample_and_group(xyz, points)

        # new_points shape: (B, npoint, nsample, C+3)
        # Transpose to (B, C+3, npoint*nsample) for conv1d, but we process per-group
        # Reshape to (B, npoint, nsample, C) -> (B*npoint, C, nsample)
        B, S, K, D = new_points.shape
        new_points = new_points.permute(0, 1, 3, 2)  # (B, S, D, K)
        new_points = new_points.reshape(B * S, D, K)  # (B*S, D, K)

        # Apply shared MLP
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        # Max pooling over the group dimension
        new_points = torch.max(new_points, dim=2)[0]  # (B*S, D_out)
        new_points = new_points.view(B, S, -1)  # (B, S, D_out)

        return new_xyz, new_points

    def _sample_and_group(self, xyz, points):
        """FPS + Ball Query + Grouping."""
        B, N, C = xyz.shape

        # Farthest point sampling
        fps_idx = farthest_point_sampling(xyz, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

        # Ball query
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, nsample)

        # Group points
        grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # Normalize to local coords

        if points is not None:
            grouped_points = index_points(points, idx)  # (B, npoint, nsample, C)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        return new_xyz, new_points

    def _sample_and_group_all(self, xyz, points):
        """Group all points into one set (for global feature extraction)."""
        B, N, C = xyz.shape

        new_xyz = torch.zeros(B, 1, C, device=xyz.device)
        grouped_xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)

        if points is not None:
            new_points = torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1)
        else:
            new_points = grouped_xyz

        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """
    PointNet++ Set Abstraction with Multi-Scale Grouping (MSG).

    Uses multiple radius/nsample/MLP configurations to capture features
    at different scales, then concatenates the results.

    Args:
        npoint: Number of points to sample with FPS
        radius_list: List of radii for each scale
        nsample_list: List of max samples for each scale
        in_channel: Number of input feature channels
        mlp_list: List of MLP configs (each is a list of output channels)
    """

    def __init__(
        self,
        npoint: int,
        radius_list: list,
        nsample_list: list,
        in_channel: int,
        mlp_list: list,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i, mlp in enumerate(mlp_list):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3  # +3 for local xyz coordinates
            for out_channel in mlp:
                convs.append(nn.Conv1d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm1d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor,
    ) -> tuple:
        """
        Forward pass.

        Args:
            xyz: Point coordinates, shape (B, N, 3)
            points: Point features, shape (B, N, C) or None

        Returns:
            new_xyz: Sampled point coordinates, shape (B, npoint, 3)
            new_points: Concatenated multi-scale features, shape (B, npoint, sum(mlp[-1]))
        """
        B, N, _ = xyz.shape

        # Farthest point sampling (shared across all scales)
        fps_idx = farthest_point_sampling(xyz, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

        new_points_list = []

        for i, (radius, nsample) in enumerate(
            zip(self.radius_list, self.nsample_list)
        ):
            # Ball query at this scale
            idx = ball_query(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)

            # Group points
            grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat(
                    [grouped_xyz_norm, grouped_points], dim=-1
                )
            else:
                grouped_points = grouped_xyz_norm

            # (B, npoint, nsample, C+3) -> (B*npoint, C+3, nsample)
            B_cur, S, K, D = grouped_points.shape
            grouped_points = grouped_points.permute(0, 1, 3, 2)
            grouped_points = grouped_points.reshape(B_cur * S, D, K)

            # Apply MLP for this scale
            for conv, bn in zip(self.conv_blocks[i], self.bn_blocks[i]):
                grouped_points = F.relu(bn(conv(grouped_points)))

            # Max pooling
            grouped_points = torch.max(grouped_points, dim=2)[0]  # (B*S, D_out)
            grouped_points = grouped_points.view(B_cur, S, -1)  # (B, npoint, D_out)

            new_points_list.append(grouped_points)

        # Concatenate features from all scales
        new_points = torch.cat(new_points_list, dim=-1)  # (B, npoint, sum_D)

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation layer.

    Interpolates features from subsampled points back to original resolution
    using inverse distance weighting (3 nearest neighbors), then applies MLP.

    Args:
        in_channel: Number of input channels (interpolated + skip connection)
        mlp: List of output channel sizes for the MLP layers
    """

    def __init__(self, in_channel: int, mlp: list):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: interpolate features from xyz2 to xyz1.

        Args:
            xyz1: Target (higher-res) coordinates, shape (B, N, 3)
            xyz2: Source (lower-res) coordinates, shape (B, S, 3)
            points1: Skip connection features at target resolution,
                     shape (B, N, C1) or None
            points2: Features at source resolution, shape (B, S, C2)

        Returns:
            Propagated features at target resolution, shape (B, N, D)
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # Only one source point: broadcast its features to all targets
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # Compute distances from each target point to all source points
            dists = square_distance(xyz1, xyz2)  # (B, N, S)

            # Find 3 nearest source points for each target point
            dists, idx = dists.sort(dim=-1)
            dists = dists[:, :, :3]  # (B, N, 3)
            idx = idx[:, :, :3]  # (B, N, 3)

            # Inverse distance weighting
            dist_recip = 1.0 / (dists + 1e-8)  # (B, N, 3)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # (B, N, 1)
            weight = dist_recip / norm  # (B, N, 3)

            # Gather source features and compute weighted sum
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.unsqueeze(-1), dim=2
            )  # (B, N, C2)

        # Concatenate with skip connection features
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # Apply MLP: (B, N, C) -> (B, C, N) for Conv1d
        new_points = new_points.permute(0, 2, 1)  # (B, C, N)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = new_points.permute(0, 2, 1)  # (B, N, C_out)

        return new_points
