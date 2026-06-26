"""BEVFormer Temporal Self-Attention module.

Implements temporal BEV-to-BEV attention with ego-motion alignment. Previous
BEV features are warped to the current frame using the ego-motion transformation
matrix, then deformable self-attention is applied between current queries and
the aligned historical features.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemporalSelfAttention"]


class DeformableSelfAttention(nn.Module):
    """Single-scale deformable self-attention for BEV features.

    Learns sampling offsets around reference points on a 2D BEV grid and
    computes attention-weighted feature aggregation.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
    ) -> None:
        """Initialize deformable self-attention.

        Args:
            embed_dim: Embedding/feature dimension.
            num_heads: Number of attention heads.
            num_points: Number of sampling points per head.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Sampling offsets: 2D offset per head per point
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)

        # Attention weights
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)

        # Value and output projections
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with proper defaults for deformable attention."""
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)

        # Initialize sampling offset biases in a circular pattern
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * torch.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 2).repeat(1, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, i, :] *= i + 1

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
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Forward pass of single-scale deformable self-attention.

        Args:
            query: Query features, shape (B, H*W, embed_dim).
            value: Value features to sample from, shape (B, H*W, embed_dim).
            reference_points: Normalized reference points in [0, 1],
                shape (B, H*W, 2).
            spatial_shape: Spatial dimensions (H, W) of the BEV grid.

        Returns:
            Output features, shape (B, H*W, embed_dim).
        """
        batch_size, num_queries, _ = query.shape
        bev_h, bev_w = spatial_shape

        # Project values: (B, H*W, embed_dim) -> (B, H*W, num_heads, head_dim)
        value = self.value_proj(value)
        value = value.view(batch_size, num_queries, self.num_heads, self.head_dim)
        # Rearrange to (B*num_heads, head_dim, H, W) for grid_sample
        value = (
            value.permute(0, 2, 3, 1)
            .reshape(batch_size * self.num_heads, self.head_dim, bev_h, bev_w)
        )

        # Compute sampling offsets: (B, H*W, num_heads * num_points * 2)
        offsets = self.sampling_offsets(query)
        offsets = offsets.view(
            batch_size, num_queries, self.num_heads, self.num_points, 2
        )

        # Compute attention weights
        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(
            batch_size, num_queries, self.num_heads, self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute sampling locations
        # reference_points: (B, H*W, 2) -> (B, H*W, 1, 1, 2)
        ref_pts = reference_points[:, :, None, None, :]

        # Normalize offsets by spatial dimensions
        offset_normalizer = torch.tensor(
            [[bev_w, bev_h]], device=query.device, dtype=query.dtype
        ).view(1, 1, 1, 1, 2)
        sampling_locations = ref_pts + offsets.unsqueeze(3) / offset_normalizer
        sampling_locations = sampling_locations.squeeze(3)  # (B, Q, H, P, 2)

        # Convert to grid_sample coordinates [-1, 1]
        sampling_grid = 2.0 * sampling_locations - 1.0
        # (B, num_queries, num_heads, num_points, 2) -> (B*num_heads, num_queries, num_points, 2)
        sampling_grid = (
            sampling_grid.permute(0, 2, 1, 3, 4)
            .reshape(batch_size * self.num_heads, num_queries, self.num_points, 2)
        )

        # Bilinear sampling: (B*num_heads, head_dim, num_queries, num_points)
        sampled = F.grid_sample(
            value,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Reshape: (B, num_heads, head_dim, num_queries, num_points)
        sampled = sampled.view(
            batch_size, self.num_heads, self.head_dim, num_queries, self.num_points
        )

        # Apply attention weights
        # attn_weights: (B, num_queries, num_heads, num_points)
        # -> (B, num_heads, 1, num_queries, num_points)
        weights = attn_weights.permute(0, 2, 1, 3).unsqueeze(2)

        # Weighted sum over points: (B, num_heads, head_dim, num_queries)
        output = (sampled * weights).sum(dim=-1)

        # Reshape: (B, num_queries, embed_dim)
        output = output.permute(0, 3, 1, 2).reshape(
            batch_size, num_queries, self.embed_dim
        )

        # Output projection
        output = self.output_proj(output)
        return output


class TemporalSelfAttention(nn.Module):
    """Temporal self-attention with ego-motion alignment for BEVFormer.

    Aligns previous BEV features to the current ego frame using the ego-motion
    transformation, then applies deformable self-attention between the current
    BEV queries and the aligned historical features.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        bev_h: int = 200,
        bev_w: int = 200,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        """Initialize temporal self-attention.

        Args:
            embed_dim: Feature/embedding dimension.
            num_heads: Number of attention heads for deformable attention.
            num_points: Number of sampling points per attention head.
            bev_h: Height of the BEV grid.
            bev_w: Width of the BEV grid.
            pc_range: Point cloud range (x_min, y_min, z_min, x_max, y_max, z_max).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range

        # Deformable self-attention for temporal fusion
        self.temporal_deformable_attn = DeformableSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_points=num_points,
        )

        # Self-attention fallback for first frame
        self.self_deformable_attn = DeformableSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_points=num_points,
        )

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

        # Projection to blend current and temporal features
        self.blend_proj = nn.Linear(embed_dim * 2, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.blend_proj.weight)
        nn.init.zeros_(self.blend_proj.bias)

    def _create_bev_grid(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Create BEV grid coordinates in world space.

        Returns:
            Grid coordinates of shape (bev_h, bev_w, 2) in world coordinates (x, y).
        """
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]

        # Create normalized grid [0, 1] then scale to world coordinates
        xs = torch.linspace(0.5, self.bev_w - 0.5, self.bev_w, device=device, dtype=dtype)
        ys = torch.linspace(0.5, self.bev_h - 0.5, self.bev_h, device=device, dtype=dtype)
        xs = xs / self.bev_w  # [0, 1]
        ys = ys / self.bev_h

        # Scale to world coordinates
        xs = xs * (x_max - x_min) + x_min
        ys = ys * (y_max - y_min) + y_min

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (bev_h, bev_w, 2)

        return grid

    def _get_reference_points(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate normalized reference points for the BEV grid.

        Returns:
            Reference points of shape (1, bev_h*bev_w, 2) normalized to [0, 1].
        """
        xs = torch.linspace(0.5, self.bev_w - 0.5, self.bev_w, device=device, dtype=dtype)
        ys = torch.linspace(0.5, self.bev_h - 0.5, self.bev_h, device=device, dtype=dtype)
        xs = xs / self.bev_w
        ys = ys / self.bev_h

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ref_points = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        ref_points = ref_points.reshape(1, -1, 2)  # (1, H*W, 2)

        return ref_points

    def _align_prev_bev(
        self,
        prev_bev: torch.Tensor,
        ego_motion: torch.Tensor,
    ) -> torch.Tensor:
        """Align previous BEV features to the current ego frame.

        Uses the ego-motion transformation to warp previous BEV features
        into the current coordinate system via grid_sample.

        Args:
            prev_bev: Previous BEV features, shape (B, bev_h*bev_w, embed_dim).
            ego_motion: Ego-motion transformation from previous to current frame,
                shape (B, 4, 4). This is T_curr_from_prev.

        Returns:
            Aligned previous BEV features, shape (B, bev_h*bev_w, embed_dim).
        """
        batch_size = prev_bev.shape[0]
        device = prev_bev.device
        dtype = prev_bev.dtype

        # Reshape prev_bev to spatial format: (B, embed_dim, bev_h, bev_w)
        prev_bev_spatial = prev_bev.permute(0, 2, 1).reshape(
            batch_size, self.embed_dim, self.bev_h, self.bev_w
        )

        # Create BEV grid in world coordinates: (bev_h, bev_w, 2)
        bev_grid = self._create_bev_grid(device, dtype)

        # Extend to 3D by adding z=0: (bev_h, bev_w, 3)
        z_zeros = torch.zeros(
            self.bev_h, self.bev_w, 1, device=device, dtype=dtype
        )
        bev_grid_3d = torch.cat([bev_grid, z_zeros], dim=-1)

        # Make homogeneous: (bev_h, bev_w, 4)
        ones = torch.ones(
            self.bev_h, self.bev_w, 1, device=device, dtype=dtype
        )
        bev_grid_homo = torch.cat([bev_grid_3d, ones], dim=-1)

        # Flatten: (bev_h*bev_w, 4)
        bev_grid_flat = bev_grid_homo.reshape(-1, 4)

        # Apply ego-motion inverse to find where current positions were in prev frame
        # ego_motion: T_curr_from_prev means prev_point = T_inv @ curr_point
        # We need to find where in the previous BEV each current position maps to
        ego_motion_inv = torch.inverse(ego_motion)  # (B, 4, 4)

        # Transform: (B, 4, 4) @ (4, N) -> (B, 4, N) -> (B, N, 4)
        bev_grid_expand = bev_grid_flat.T.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, 4, N)
        transformed = torch.bmm(ego_motion_inv, bev_grid_expand)  # (B, 4, N)
        transformed = transformed.permute(0, 2, 1)  # (B, N, 4)

        # Extract x, y coordinates in previous frame
        prev_xy = transformed[:, :, :2]  # (B, N, 2)

        # Normalize to [-1, 1] for grid_sample using pc_range
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]

        # Map from world coords to [-1, 1]
        grid_x = (prev_xy[:, :, 0] - x_min) / (x_max - x_min) * 2.0 - 1.0
        grid_y = (prev_xy[:, :, 1] - y_min) / (y_max - y_min) * 2.0 - 1.0

        # Reshape to grid format: (B, bev_h, bev_w, 2)
        sampling_grid = torch.stack([grid_x, grid_y], dim=-1).reshape(
            batch_size, self.bev_h, self.bev_w, 2
        )

        # Warp previous BEV features: (B, embed_dim, bev_h, bev_w)
        aligned_bev = F.grid_sample(
            prev_bev_spatial,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Reshape back to sequence format: (B, bev_h*bev_w, embed_dim)
        aligned_bev = aligned_bev.flatten(2).permute(0, 2, 1)

        return aligned_bev

    def forward(
        self,
        bev_queries: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
        ego_motion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of temporal self-attention.

        Args:
            bev_queries: Current BEV queries, shape (B, bev_h*bev_w, embed_dim).
            prev_bev: Previous frame BEV features, shape (B, bev_h*bev_w, embed_dim).
                None for the first frame.
            ego_motion: Ego-motion transformation from previous to current frame,
                shape (B, 4, 4). Required when prev_bev is provided.

        Returns:
            Updated BEV features with temporal information,
            shape (B, bev_h*bev_w, embed_dim).
        """
        batch_size = bev_queries.shape[0]
        device = bev_queries.device
        dtype = bev_queries.dtype

        # Generate reference points for the BEV grid
        ref_points = self._get_reference_points(device, dtype)
        ref_points = ref_points.expand(batch_size, -1, -1)

        spatial_shape = (self.bev_h, self.bev_w)

        if prev_bev is None:
            # First frame: apply self-attention on current queries only
            self_attn_output = self.self_deformable_attn(
                query=bev_queries,
                value=bev_queries,
                reference_points=ref_points,
                spatial_shape=spatial_shape,
            )
            output = bev_queries + self.dropout(self_attn_output)
            output = self.norm(output)
            return output

        # Align previous BEV to current frame
        assert ego_motion is not None, (
            "ego_motion is required when prev_bev is provided"
        )
        aligned_prev_bev = self._align_prev_bev(prev_bev, ego_motion)

        # Apply deformable attention: current queries attend to aligned previous BEV
        temporal_output = self.temporal_deformable_attn(
            query=bev_queries,
            value=aligned_prev_bev,
            reference_points=ref_points,
            spatial_shape=spatial_shape,
        )

        # Apply self-attention on current queries
        self_output = self.self_deformable_attn(
            query=bev_queries,
            value=bev_queries,
            reference_points=ref_points,
            spatial_shape=spatial_shape,
        )

        # Blend temporal and self-attention outputs
        combined = torch.cat([temporal_output, self_output], dim=-1)
        blended = self.blend_proj(combined)

        # Residual connection and normalization
        output = bev_queries + self.dropout(blended)
        output = self.norm(output)

        return output
