"""BEVFormer Encoder: stacked transformer encoder layers for BEV feature generation.

This module implements the BEVFormer encoder which iteratively refines BEV queries
by alternating between temporal self-attention (fusing historical BEV features
aligned via ego-motion) and spatial cross-attention (lifting multi-camera image
features into the BEV plane). Each encoder layer also includes a feed-forward
network with GELU activation.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import BEVFormerSpatialCrossAttention

__all__ = ["BEVFormerEncoder", "BEVFormerEncoderLayer"]


class BEVPositionalEncoding(nn.Module):
    """Learnable 2D sinusoidal positional encoding for the BEV grid.

    Generates a fixed sinusoidal embedding for the 2D BEV grid, augmented with
    a small learnable residual to allow task-specific positional adjustments.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        bev_h: int = 200,
        bev_w: int = 200,
    ) -> None:
        """Initialize BEV positional encoding.

        Args:
            embed_dims: Embedding dimension (must be divisible by 4 for 2D encoding).
            bev_h: Height of the BEV grid.
            bev_w: Width of the BEV grid.
        """
        super().__init__()
        assert embed_dims % 4 == 0, (
            f"embed_dims must be divisible by 4 for 2D sinusoidal encoding, got {embed_dims}"
        )
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Generate fixed sinusoidal positional encoding
        pe = self._generate_sinusoidal_2d(embed_dims, bev_h, bev_w)
        self.register_buffer("pe", pe)  # (1, bev_h*bev_w, embed_dims)

        # Learnable residual for fine-tuning positional information
        self.learnable_residual = nn.Parameter(
            torch.zeros(1, bev_h * bev_w, embed_dims)
        )
        nn.init.normal_(self.learnable_residual, std=0.01)

    @staticmethod
    def _generate_sinusoidal_2d(
        embed_dims: int, height: int, width: int
    ) -> torch.Tensor:
        """Generate 2D sinusoidal positional encoding.

        Creates separate sinusoidal embeddings for the x and y coordinates,
        each using embed_dims/2 dimensions, then concatenates them.

        Args:
            embed_dims: Total embedding dimension.
            height: Grid height.
            width: Grid width.

        Returns:
            Positional encoding tensor of shape (1, height*width, embed_dims).
        """
        half_dim = embed_dims // 2
        quarter_dim = embed_dims // 4

        # Create position indices
        y_pos = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # (H, 1)
        x_pos = torch.arange(width, dtype=torch.float32).unsqueeze(1)  # (W, 1)

        # Create frequency bands
        dim_t = torch.arange(quarter_dim, dtype=torch.float32)
        dim_t = 10000.0 ** (2.0 * (dim_t // 2) / quarter_dim)

        # Compute sinusoidal encoding for y
        y_embed = y_pos / dim_t.unsqueeze(0)  # (H, quarter_dim)
        y_sin = y_embed[:, 0::2].sin()  # (H, quarter_dim/2)
        y_cos = y_embed[:, 1::2].cos()  # (H, quarter_dim/2)
        y_pe = torch.stack([y_sin, y_cos], dim=-1).flatten(-2)  # (H, quarter_dim)

        # Pad if quarter_dim is odd
        if y_pe.shape[-1] < half_dim:
            y_pe = torch.cat(
                [y_pe, torch.zeros(height, half_dim - y_pe.shape[-1])], dim=-1
            )

        # Compute sinusoidal encoding for x
        x_embed = x_pos / dim_t.unsqueeze(0)  # (W, quarter_dim)
        x_sin = x_embed[:, 0::2].sin()  # (W, quarter_dim/2)
        x_cos = x_embed[:, 1::2].cos()  # (W, quarter_dim/2)
        x_pe = torch.stack([x_sin, x_cos], dim=-1).flatten(-2)  # (W, quarter_dim)

        if x_pe.shape[-1] < half_dim:
            x_pe = torch.cat(
                [x_pe, torch.zeros(width, half_dim - x_pe.shape[-1])], dim=-1
            )

        # Combine: expand y across width and x across height, then concatenate
        # y_pe: (H, half_dim) -> (H, W, half_dim)
        # x_pe: (W, half_dim) -> (H, W, half_dim)
        y_pe_2d = y_pe.unsqueeze(1).expand(-1, width, -1)
        x_pe_2d = x_pe.unsqueeze(0).expand(height, -1, -1)

        pe = torch.cat([y_pe_2d, x_pe_2d], dim=-1)  # (H, W, embed_dims)
        pe = pe.reshape(1, height * width, embed_dims)

        return pe

    def forward(self, bev_queries: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to BEV queries.

        Args:
            bev_queries: BEV query features, shape (B, bev_h*bev_w, embed_dims).

        Returns:
            Positionally-encoded BEV queries, same shape as input.
        """
        return bev_queries + self.pe + self.learnable_residual


class FFN(nn.Module):
    """Feed-Forward Network with two linear layers, GELU activation, and dropout.

    Standard transformer FFN block: Linear -> GELU -> Dropout -> Linear -> Dropout.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_dims: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        """Initialize feed-forward network.

        Args:
            embed_dims: Input and output dimension.
            feedforward_dims: Hidden dimension of the intermediate layer.
            dropout: Dropout rate applied after each linear layer.
        """
        super().__init__()
        self.linear1 = nn.Linear(embed_dims, feedforward_dims)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dims, embed_dims)
        self.dropout2 = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform and biases with zeros."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN.

        Args:
            x: Input tensor of shape (..., embed_dims).

        Returns:
            Output tensor of same shape as input.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class BEVFormerEncoderLayer(nn.Module):
    """Single BEVFormer encoder layer.

    Each layer consists of three sub-layers applied in order:
      1. Temporal Self-Attention: fuses current BEV queries with ego-motion-aligned
         previous BEV features using deformable self-attention.
      2. Spatial Cross-Attention: lifts multi-camera image features into BEV space
         via deformable cross-attention with 3D reference point projection.
      3. FFN: standard feed-forward network for non-linear feature transformation.

    Uses pre-norm architecture (LayerNorm before each sub-layer) with residual
    connections around the spatial cross-attention and FFN sub-layers. The temporal
    self-attention module handles its own normalization and residual internally.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_dims: int = 1024,
        dropout: float = 0.1,
        num_points_temporal: int = 4,
        num_points_spatial: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        num_ref_points: int = 4,
        bev_h: int = 200,
        bev_w: int = 200,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        """Initialize BEVFormer encoder layer.

        Args:
            embed_dims: Feature embedding dimension.
            num_heads: Number of attention heads.
            feedforward_dims: Hidden dimension of the FFN.
            dropout: Dropout rate.
            num_points_temporal: Sampling points for temporal deformable attention.
            num_points_spatial: Sampling points for spatial deformable attention.
            num_levels: Number of multi-scale feature levels from FPN.
            num_cams: Number of camera views.
            num_ref_points: Number of z-axis reference points per BEV query.
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            pc_range: Point cloud range (x_min, y_min, z_min, x_max, y_max, z_max).
        """
        super().__init__()

        # Sub-layer 1: Temporal Self-Attention
        self.temporal_self_attn = TemporalSelfAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            num_points=num_points_temporal,
            bev_h=bev_h,
            bev_w=bev_w,
            pc_range=pc_range,
        )

        # Sub-layer 2: Spatial Cross-Attention
        self.spatial_cross_attn = BEVFormerSpatialCrossAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points_spatial,
            num_cams=num_cams,
            num_ref_points=num_ref_points,
            pc_range=pc_range,
            bev_h=bev_h,
            bev_w=bev_w,
        )

        # Sub-layer 3: Feed-Forward Network
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_dims=feedforward_dims,
            dropout=dropout,
        )

        # Layer normalization (pre-norm for spatial cross-attention and FFN)
        self.norm_spatial = nn.LayerNorm(embed_dims)
        self.norm_ffn = nn.LayerNorm(embed_dims)

        # Dropout for residual connections
        self.dropout_spatial = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(
        self,
        bev_queries: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
        ego_motion: Optional[torch.Tensor] = None,
        img_shape: Tuple[int, int] = (900, 1600),
    ) -> torch.Tensor:
        """Forward pass of a single encoder layer.

        Args:
            bev_queries: BEV query features, shape (B, bev_h*bev_w, embed_dims).
            multi_scale_features: List of multi-scale FPN features.
                Each has shape (B*num_cams, embed_dims, H_i, W_i).
            intrinsics: Camera intrinsic matrices, shape (B, num_cams, 3, 3).
            extrinsics: World-to-camera matrices, shape (B, num_cams, 4, 4).
            prev_bev: Previous BEV features for temporal fusion,
                shape (B, bev_h*bev_w, embed_dims). None for first frame.
            ego_motion: Ego-motion transformation matrix (prev-to-current),
                shape (B, 4, 4). Required when prev_bev is provided.
            img_shape: Image dimensions (H, W) for projection normalization.

        Returns:
            Updated BEV features, shape (B, bev_h*bev_w, embed_dims).
        """
        # 1. Temporal Self-Attention (handles its own residual + norm)
        bev_queries = self.temporal_self_attn(
            bev_queries=bev_queries,
            prev_bev=prev_bev,
            ego_motion=ego_motion,
        )

        # 2. Spatial Cross-Attention with pre-norm and residual
        residual = bev_queries
        bev_queries = self.norm_spatial(bev_queries)
        bev_queries = self.spatial_cross_attn(
            bev_queries=bev_queries,
            multi_scale_features=multi_scale_features,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            img_shape=img_shape,
        )
        # Note: spatial_cross_attn already applies its own residual and norm,
        # so we skip the explicit residual here to avoid double-application.
        # The pre-norm above ensures stable gradient flow.

        # 3. FFN with pre-norm and residual
        residual = bev_queries
        bev_queries = self.norm_ffn(bev_queries)
        bev_queries = residual + self.dropout_ffn(self.ffn(bev_queries))

        return bev_queries


class BEVFormerEncoder(nn.Module):
    """BEVFormer Encoder: stacked BEVFormerEncoderLayer modules.

    Takes BEV queries and iteratively refines them through N encoder layers,
    each performing temporal self-attention, spatial cross-attention to multi-camera
    features, and feed-forward transformation. Outputs encoded BEV features that
    represent the 3D scene from a bird's-eye view perspective.
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_dims: int = 1024,
        dropout: float = 0.1,
        num_points_temporal: int = 4,
        num_points_spatial: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        num_ref_points: int = 4,
        bev_h: int = 200,
        bev_w: int = 200,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        """Initialize the BEVFormer encoder.

        Args:
            num_layers: Number of stacked encoder layers.
            embed_dims: Feature embedding dimension throughout the encoder.
            num_heads: Number of attention heads in all attention modules.
            feedforward_dims: Hidden dimension of FFN in each layer.
            dropout: Dropout rate used across all sub-layers.
            num_points_temporal: Deformable sampling points for temporal attention.
            num_points_spatial: Deformable sampling points for spatial attention.
            num_levels: Number of multi-scale feature levels from backbone FPN.
            num_cams: Number of camera views (e.g., 6 for nuScenes).
            num_ref_points: Number of z-axis reference points per BEV position.
            bev_h: BEV grid height in cells.
            bev_w: BEV grid width in cells.
            pc_range: Point cloud range defining the BEV spatial extent.
        """
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Learnable BEV queries: initialized per-position
        self.bev_queries = nn.Parameter(
            torch.zeros(1, bev_h * bev_w, embed_dims)
        )
        nn.init.xavier_uniform_(self.bev_queries)

        # BEV positional encoding
        self.bev_pos_encoding = BEVPositionalEncoding(
            embed_dims=embed_dims,
            bev_h=bev_h,
            bev_w=bev_w,
        )

        # Stacked encoder layers
        self.layers = nn.ModuleList([
            BEVFormerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_dims=feedforward_dims,
                dropout=dropout,
                num_points_temporal=num_points_temporal,
                num_points_spatial=num_points_spatial,
                num_levels=num_levels,
                num_cams=num_cams,
                num_ref_points=num_ref_points,
                bev_h=bev_h,
                bev_w=bev_w,
                pc_range=pc_range,
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(embed_dims)

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
        ego_motion: Optional[torch.Tensor] = None,
        img_shape: Tuple[int, int] = (900, 1600),
    ) -> torch.Tensor:
        """Forward pass through the full BEVFormer encoder.

        Args:
            multi_scale_features: Multi-scale image features from backbone FPN.
                List of tensors each with shape (B*num_cams, embed_dims, H_i, W_i).
            intrinsics: Camera intrinsic matrices, shape (B, num_cams, 3, 3).
            extrinsics: World-to-camera extrinsic matrices, shape (B, num_cams, 4, 4).
            prev_bev: Previous frame's encoded BEV features for temporal fusion,
                shape (B, bev_h*bev_w, embed_dims). None for the first frame in
                a sequence.
            ego_motion: Ego-motion transformation matrix from previous to current
                frame, shape (B, 4, 4). Required when prev_bev is provided.
            img_shape: Original image dimensions (H, W) for reference point
                projection normalization.

        Returns:
            Encoded BEV features of shape (B, bev_h*bev_w, embed_dims) representing
            the bird's-eye-view scene understanding.
        """
        # Infer batch size from camera parameters
        batch_size = intrinsics.shape[0]

        # Expand learnable BEV queries to batch: (1, H*W, C) -> (B, H*W, C)
        bev_queries = self.bev_queries.expand(batch_size, -1, -1)

        # Add positional encoding
        bev_queries = self.bev_pos_encoding(bev_queries)

        # Pass through stacked encoder layers
        for layer in self.layers:
            bev_queries = layer(
                bev_queries=bev_queries,
                multi_scale_features=multi_scale_features,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                prev_bev=prev_bev,
                ego_motion=ego_motion,
                img_shape=img_shape,
            )

        # Final normalization
        bev_queries = self.final_norm(bev_queries)

        return bev_queries
