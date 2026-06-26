"""BEVFormer Spatial Cross-Attention module.

Implements deformable attention from BEV queries to multi-camera image features.
For each BEV query, 3D reference points are generated at multiple heights,
projected to each camera's image plane, and multi-scale deformable attention
is applied to sample and aggregate features from visible cameras.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BEVFormerSpatialCrossAttention"]


class MSDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention mechanism.

    Learns sampling offsets around reference points and computes attention-weighted
    aggregation across multiple feature levels, attention heads, and sampling points.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
    ) -> None:
        """Initialize multi-scale deformable attention.

        Args:
            embed_dim: Embedding dimension of queries and output.
            num_heads: Number of attention heads.
            num_levels: Number of feature map levels to attend to.
            num_points: Number of sampling points per head per level.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Sampling offsets: predict 2D offsets for each head, level, point
        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )

        # Attention weights: one weight per head per level per point
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )

        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following deformable attention conventions."""
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)

        # Initialize offsets bias so initial sampling points form a grid
        # around the reference point
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * torch.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        )  # normalize to [-1, 1]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of multi-scale deformable attention.

        Args:
            query: BEV queries, shape (B, num_queries, embed_dim).
            value: Flattened multi-scale features, shape (B, sum(H_i*W_i), embed_dim).
            reference_points: Normalized reference points in [0, 1],
                shape (B, num_queries, num_levels, 2).
            spatial_shapes: Spatial dimensions of each level,
                shape (num_levels, 2) with (H_i, W_i).
            level_start_index: Start index of each level in the flattened value,
                shape (num_levels,).

        Returns:
            Output features, shape (B, num_queries, embed_dim).
        """
        batch_size, num_queries, _ = query.shape
        _, num_values, _ = value.shape

        # Project values
        value = self.value_proj(value)
        value = value.view(
            batch_size, num_values, self.num_heads, self.head_dim
        )

        # Compute sampling offsets
        # (B, num_queries, num_heads * num_levels * num_points * 2)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Compute attention weights
        # (B, num_queries, num_heads * num_levels * num_points)
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        # Compute sampling locations
        # reference_points: (B, num_queries, num_levels, 2)
        # -> (B, num_queries, 1, num_levels, 1, 2)
        ref_pts = reference_points[:, :, None, :, None, :]

        # Normalize offsets by spatial shape to get relative offsets in [0, 1] space
        # spatial_shapes: (num_levels, 2) -> (1, 1, 1, num_levels, 1, 2)
        offset_normalizer = spatial_shapes[None, None, None, :, None, :].flip(-1).float()
        sampling_locations = ref_pts + sampling_offsets / offset_normalizer

        # Bilinear sampling from multi-scale features
        output = self._sample_and_aggregate(
            value, sampling_locations, attention_weights, spatial_shapes, level_start_index
        )

        # Output projection
        output = self.output_proj(output)
        return output

    def _sample_and_aggregate(
        self,
        value: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Sample features at deformed locations and aggregate with attention weights.

        Args:
            value: (B, num_values, num_heads, head_dim)
            sampling_locations: (B, num_queries, num_heads, num_levels, num_points, 2)
                Normalized coordinates in [0, 1].
            attention_weights: (B, num_queries, num_heads, num_levels, num_points)
            spatial_shapes: (num_levels, 2)
            level_start_index: (num_levels,)

        Returns:
            Aggregated features, shape (B, num_queries, embed_dim).
        """
        batch_size, num_queries, num_heads, num_levels, num_points, _ = (
            sampling_locations.shape
        )

        output_list = []

        for level_idx in range(num_levels):
            h, w = spatial_shapes[level_idx]
            start_idx = level_start_index[level_idx]
            end_idx = (
                level_start_index[level_idx + 1]
                if level_idx < num_levels - 1
                else value.shape[1]
            )

            # Extract value for this level: (B, H*W, num_heads, head_dim)
            value_level = value[:, start_idx:end_idx, :, :]
            # Reshape to (B*num_heads, head_dim, H, W) for grid_sample
            value_level = (
                value_level.permute(0, 2, 3, 1)
                .reshape(batch_size * num_heads, self.head_dim, int(h), int(w))
            )

            # Sampling locations for this level: (B, num_queries, num_heads, num_points, 2)
            sampling_loc_level = sampling_locations[:, :, :, level_idx, :, :]
            # Convert from [0, 1] to [-1, 1] for grid_sample
            sampling_grid = 2.0 * sampling_loc_level - 1.0
            # Reshape to (B*num_heads, num_queries, num_points, 2)
            sampling_grid = (
                sampling_grid.permute(0, 2, 1, 3, 4)
                .reshape(batch_size * num_heads, num_queries, num_points, 2)
            )

            # Bilinear sampling: (B*num_heads, head_dim, num_queries, num_points)
            sampled = F.grid_sample(
                value_level,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            # -> (B, num_heads, head_dim, num_queries, num_points)
            sampled = sampled.view(
                batch_size, num_heads, self.head_dim, num_queries, num_points
            )
            output_list.append(sampled)

        # Stack levels and weight: (B, num_heads, head_dim, num_queries, num_levels, num_points)
        stacked = torch.stack(output_list, dim=-2)
        # attention_weights: (B, num_queries, num_heads, num_levels, num_points)
        # -> (B, num_heads, 1, num_queries, num_levels, num_points)
        weights = attention_weights.permute(0, 2, 1, 3, 4).unsqueeze(2)

        # Weighted sum over levels and points
        # (B, num_heads, head_dim, num_queries)
        output = (stacked * weights).sum(dim=[-1, -2])
        # -> (B, num_queries, num_heads, head_dim) -> (B, num_queries, embed_dim)
        output = output.permute(0, 3, 1, 2).reshape(
            batch_size, num_queries, self.embed_dim
        )

        return output


class BEVFormerSpatialCrossAttention(nn.Module):
    """Spatial cross-attention from BEV queries to multi-camera image features.

    For each BEV query position, 3D reference points are generated at multiple
    heights. These are projected to each camera's image plane, and deformable
    attention is used to sample features from cameras where the points are visible.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        num_cams: int = 6,
        num_ref_points: int = 4,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_h: int = 200,
        bev_w: int = 200,
    ) -> None:
        """Initialize BEVFormer spatial cross-attention.

        Args:
            embed_dim: Embedding dimension for queries and features.
            num_heads: Number of attention heads.
            num_levels: Number of multi-scale feature levels.
            num_points: Number of deformable sampling points per reference point.
            num_cams: Number of camera views.
            num_ref_points: Number of reference points along the z-axis per BEV query.
            pc_range: Point cloud range (x_min, y_min, z_min, x_max, y_max, z_max)
                defining the BEV spatial extent.
            bev_h: BEV grid height.
            bev_w: BEV grid width.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_cams = num_cams
        self.num_ref_points = num_ref_points
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Learnable z-offsets for reference point heights
        self.z_offsets = nn.Parameter(
            torch.linspace(0.0, 1.0, num_ref_points).view(1, 1, num_ref_points, 1)
        )

        # Deformable attention module
        self.deformable_attn = MSDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )

        # Layer norm and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

        # Projection to combine features from multiple cameras and reference points
        self.cam_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # Query projection
        self.query_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform and zeros for biases."""
        for module in [self.cam_embed, self.query_proj]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)

    def _generate_ref_points_3d(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate 3D reference points for all BEV query positions.

        Creates a grid of (x, y) positions in the BEV plane and adds multiple
        z-axis heights defined by learnable offsets within pc_range.

        Args:
            device: Device to create tensors on.
            dtype: Data type for tensors.

        Returns:
            Reference points of shape (1, bev_h*bev_w, num_ref_points, 3)
            with normalized coordinates in [0, 1] within pc_range.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range

        # Create BEV grid coordinates normalized to [0, 1]
        xs = torch.linspace(0.5, self.bev_w - 0.5, self.bev_w, device=device, dtype=dtype)
        ys = torch.linspace(0.5, self.bev_h - 0.5, self.bev_h, device=device, dtype=dtype)
        xs = xs / self.bev_w  # normalize to [0, 1]
        ys = ys / self.bev_h

        # Create 2D grid: (bev_h, bev_w, 2) -> (bev_h*bev_w, 2)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

        # Expand for reference points: (bev_h*bev_w, 1, 2)
        grid_xy = grid_xy.unsqueeze(1).expand(-1, self.num_ref_points, -1)

        # Z offsets normalized to [0, 1] via sigmoid of learnable parameters
        z_values = torch.sigmoid(self.z_offsets).to(device=device, dtype=dtype)
        z_values = z_values.expand(self.bev_h * self.bev_w, -1, -1).squeeze(-1)

        # Combine: (bev_h*bev_w, num_ref_points, 3)
        ref_points_3d = torch.cat(
            [grid_xy, z_values.unsqueeze(-1)], dim=-1
        )

        # Add batch dim: (1, bev_h*bev_w, num_ref_points, 3)
        return ref_points_3d.unsqueeze(0)

    def _project_to_image(
        self,
        ref_points_3d: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        img_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project 3D reference points to each camera's image plane.

        Args:
            ref_points_3d: Normalized 3D reference points (B, num_queries, num_ref_points, 3)
                in [0, 1] coordinate space.
            intrinsics: Camera intrinsic matrices (B, num_cams, 3, 3).
            extrinsics: Camera extrinsic matrices (B, num_cams, 4, 4) - world-to-camera.
            img_shape: Image dimensions (H, W).

        Returns:
            Tuple of:
                - Projected 2D points normalized to [0, 1]:
                  (B, num_cams, num_queries, num_ref_points, 2)
                - Visibility mask:
                  (B, num_cams, num_queries, num_ref_points)
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        batch_size = ref_points_3d.shape[0]
        num_queries = ref_points_3d.shape[1]
        img_h, img_w = img_shape

        # Denormalize reference points to world coordinates
        # ref_points_3d: (B, num_queries, num_ref_points, 3) in [0, 1]
        ref_world = ref_points_3d.clone()
        ref_world[..., 0] = ref_world[..., 0] * (x_max - x_min) + x_min
        ref_world[..., 1] = ref_world[..., 1] * (y_max - y_min) + y_min
        ref_world[..., 2] = ref_world[..., 2] * (z_max - z_min) + z_min

        # Convert to homogeneous: (B, num_queries, num_ref_points, 4)
        ones = torch.ones_like(ref_world[..., :1])
        ref_homo = torch.cat([ref_world, ones], dim=-1)

        # Reshape for projection against each camera
        # ref_homo: (B, 1, num_queries, num_ref_points, 4)
        ref_homo = ref_homo.unsqueeze(1)

        # extrinsics: (B, num_cams, 4, 4) -> transform points to camera frame
        # (B, num_cams, 1, 1, 4, 4) @ (B, 1, num_queries, num_ref_points, 4, 1)
        ref_homo_expand = ref_homo.unsqueeze(-1)  # (B, 1, Q, R, 4, 1)
        extrinsics_expand = extrinsics[:, :, None, None, :, :]  # (B, N, 1, 1, 4, 4)

        # Transform to camera coordinates
        pts_cam = torch.matmul(
            extrinsics_expand, ref_homo_expand
        ).squeeze(-1)  # (B, N, Q, R, 4)

        # Keep only xyz: (B, N, Q, R, 3)
        pts_cam = pts_cam[..., :3]

        # Depth check: points must be in front of camera
        depth = pts_cam[..., 2:3]  # (B, N, Q, R, 1)
        valid_depth = (depth > 0.1).squeeze(-1)  # (B, N, Q, R)

        # Avoid division by zero
        depth_safe = depth.clamp(min=0.1)

        # Project to pixel coordinates using intrinsics
        # intrinsics: (B, num_cams, 3, 3)
        pts_cam_norm = pts_cam / depth_safe  # (B, N, Q, R, 3)
        pts_cam_norm_expand = pts_cam_norm.unsqueeze(-1)  # (B, N, Q, R, 3, 1)
        intrinsics_expand = intrinsics[:, :, None, None, :, :]  # (B, N, 1, 1, 3, 3)

        pts_pixel = torch.matmul(
            intrinsics_expand, pts_cam_norm_expand
        ).squeeze(-1)  # (B, N, Q, R, 3)

        # Normalize to [0, 1] image coordinates
        pts_2d = pts_pixel[..., :2].clone()
        pts_2d[..., 0] = pts_2d[..., 0] / img_w
        pts_2d[..., 1] = pts_2d[..., 1] / img_h

        # Visibility check: within image bounds
        valid_x = (pts_2d[..., 0] >= 0.0) & (pts_2d[..., 0] <= 1.0)
        valid_y = (pts_2d[..., 1] >= 0.0) & (pts_2d[..., 1] <= 1.0)
        visibility_mask = valid_depth & valid_x & valid_y  # (B, N, Q, R)

        return pts_2d, visibility_mask

    def forward(
        self,
        bev_queries: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        img_shape: Tuple[int, int] = (900, 1600),
    ) -> torch.Tensor:
        """Forward pass of spatial cross-attention.

        Args:
            bev_queries: BEV query features, shape (B, bev_h*bev_w, embed_dim).
            multi_scale_features: List of multi-scale image features from FPN.
                Each has shape (B*num_cams, C, H_i, W_i).
            intrinsics: Camera intrinsic matrices, shape (B, num_cams, 3, 3).
            extrinsics: World-to-camera transformation matrices,
                shape (B, num_cams, 4, 4).
            img_shape: Image height and width (H, W) used for projection normalization.

        Returns:
            Updated BEV features, shape (B, bev_h*bev_w, embed_dim).
        """
        batch_size = bev_queries.shape[0]
        num_queries = bev_queries.shape[1]
        device = bev_queries.device
        dtype = bev_queries.dtype

        # Generate 3D reference points: (1, num_queries, num_ref_points, 3)
        ref_points_3d = self._generate_ref_points_3d(device, dtype)
        ref_points_3d = ref_points_3d.expand(batch_size, -1, -1, -1)

        # Project to each camera: (B, num_cams, num_queries, num_ref_points, 2)
        pts_2d, visibility_mask = self._project_to_image(
            ref_points_3d, intrinsics, extrinsics, img_shape
        )

        # Prepare multi-scale value features per camera
        # Compute spatial shapes for deformable attention
        spatial_shapes_list = []
        for feat in multi_scale_features:
            h, w = feat.shape[2], feat.shape[3]
            spatial_shapes_list.append((h, w))
        spatial_shapes = torch.tensor(
            spatial_shapes_list, device=device, dtype=torch.long
        )

        # Level start indices
        level_start_index = torch.zeros(
            len(spatial_shapes_list), device=device, dtype=torch.long
        )
        for i in range(1, len(spatial_shapes_list)):
            level_start_index[i] = level_start_index[i - 1] + (
                spatial_shapes_list[i - 1][0] * spatial_shapes_list[i - 1][1]
            )

        # Process each camera separately and aggregate
        # Reshape features: (B*N_cams, C, H, W) -> per camera (B, C, H, W)
        output = torch.zeros_like(bev_queries)
        cam_count = torch.zeros(
            batch_size, num_queries, 1, device=device, dtype=dtype
        )

        query_proj = self.query_proj(bev_queries)

        for cam_idx in range(self.num_cams):
            # Extract features for this camera across all levels
            # multi_scale_features[level]: (B*N_cams, C, H, W)
            value_list = []
            for level_feat in multi_scale_features:
                # Extract this camera's features: index by cam_idx within each batch
                feat_per_cam = level_feat.view(
                    batch_size, self.num_cams, self.embed_dim,
                    level_feat.shape[2], level_feat.shape[3]
                )[:, cam_idx]  # (B, C, H, W)
                # Flatten spatial: (B, H*W, C)
                feat_flat = feat_per_cam.flatten(2).permute(0, 2, 1)
                value_list.append(feat_flat)

            # Concatenate all levels: (B, sum(H_i*W_i), C)
            value = torch.cat(value_list, dim=1)

            # Reference points for this camera: (B, num_queries, num_ref_points, 2)
            cam_pts_2d = pts_2d[:, cam_idx]  # (B, Q, R, 2)
            cam_mask = visibility_mask[:, cam_idx]  # (B, Q, R)

            # Average visible reference points for deformable attention reference
            # Use mean of visible points as the reference per level
            # (B, num_queries, num_ref_points, 2) -> (B, num_queries, 2)
            # Mask invalid points
            cam_pts_masked = cam_pts_2d * cam_mask.unsqueeze(-1).float()
            num_visible = cam_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
            mean_ref_pts = cam_pts_masked.sum(dim=2) / num_visible  # (B, Q, 2)

            # Expand reference points for each level
            # (B, num_queries, num_levels, 2)
            ref_pts_per_level = mean_ref_pts.unsqueeze(2).expand(
                -1, -1, self.num_levels, -1
            )

            # Check if this camera has any visible points per query
            cam_visible = cam_mask.any(dim=-1)  # (B, Q)

            # Apply deformable attention
            cam_output = self.deformable_attn(
                query=query_proj,
                value=value,
                reference_points=ref_pts_per_level,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )

            # Apply camera embedding for view-dependent weighting
            cam_weight = self.cam_embed(cam_output)  # (B, Q, C)

            # Mask out contributions from cameras where point is not visible
            cam_visible_mask = cam_visible.unsqueeze(-1).float()  # (B, Q, 1)
            output = output + cam_output * cam_visible_mask
            cam_count = cam_count + cam_visible_mask

        # Average across visible cameras
        cam_count = cam_count.clamp(min=1.0)
        output = output / cam_count

        # Residual connection with normalization
        output = bev_queries + self.dropout(output)
        output = self.norm(output)

        return output
