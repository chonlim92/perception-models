"""Lane topology utilities for hierarchical lane positional embeddings.

Shared components used by all lane detection decoders (BEVFormer, DETR3D,
PETR, MapTR, StreamMapNet). Provides:
- HybridPointEmbedding: sinusoidal base + learned residual for point ordering
- build_alibi_intra_line_bias: ALiBi-style distance bias for intra-line attention
- lane_width_consistency_loss: geometric regularization for left-right parallelism
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridPointEmbedding(nn.Module):
    """Sinusoidal base + learned residual for point position encoding.

    Provides an inductive bias that adjacent point indices are close in
    embedding space, improving early training convergence and smoothness
    of predicted lane geometries. The learned residual allows the model
    to deviate from pure sinusoidal structure where needed.
    """

    def __init__(self, num_points: int, embed_dim: int, sinusoidal_scale: float = 0.02) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"
        self.num_points = num_points
        self.embed_dim = embed_dim

        # Fixed sinusoidal base (encodes ordinal structure), scaled to match learned embeddings
        pe = torch.zeros(num_points, embed_dim)
        position = torch.arange(0, num_points, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe * sinusoidal_scale
        self.register_buffer("sinusoidal_base", pe)

        # Learned residual (same scale as sinusoidal for balanced gradient flow)
        self.residual = nn.Embedding(num_points, embed_dim)
        nn.init.normal_(self.residual.weight, std=0.02)

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility property: returns full embedding table."""
        return self.sinusoidal_base + self.residual.weight

    def forward(self, point_ids: torch.Tensor) -> torch.Tensor:
        """Lookup hybrid embeddings by point index.

        Args:
            point_ids: (N,) long tensor of point indices in [0, num_points).

        Returns:
            (N, embed_dim) hybrid positional embeddings.
        """
        return self.sinusoidal_base[point_ids] + self.residual(point_ids)


def build_alibi_intra_line_bias(
    num_total_lines: int,
    points_per_line: int,
    num_heads: int,
    slope_scale: float = 0.5,
) -> torch.Tensor:
    """Build ALiBi-style attention bias for intra-line point pairs.

    Points on the same line get a negative bias proportional to their
    index distance: bias(i,j) = -slope * |i - j|. Points on different
    lines get zero (handled by decoupled mask's -inf).

    This encodes the prior that nearby points are more relevant for local
    geometry estimation while still allowing long-range attention.

    Args:
        num_total_lines: Total number of lines (num_lanes*2 + num_other_lines).
        points_per_line: Number of points per line.
        num_heads: Number of attention heads.
        slope_scale: Base scale for slopes (smaller = weaker locality bias).

    Returns:
        Attention bias tensor (num_heads, total_queries, total_queries).
    """
    total_queries = num_total_lines * points_per_line

    # Per-head slopes (geometric sequence as in original ALiBi paper)
    slopes = torch.pow(
        2.0,
        -torch.arange(1, num_heads + 1, dtype=torch.float32) * (8.0 / num_heads),
    ) * slope_scale

    # Distance matrix within each line block
    point_dists = torch.abs(
        torch.arange(points_per_line, dtype=torch.float32).unsqueeze(0)
        - torch.arange(points_per_line, dtype=torch.float32).unsqueeze(1)
    )

    # Build full bias: block-diagonal structure
    bias = torch.zeros(num_heads, total_queries, total_queries)
    for line_idx in range(num_total_lines):
        start = line_idx * points_per_line
        end = start + points_per_line
        for h in range(num_heads):
            bias[h, start:end, start:end] = -slopes[h] * point_dists

    return bias


def lane_width_consistency_loss(
    pred_points: torch.Tensor,
    num_lanes: int,
    points_per_line: int,
) -> torch.Tensor:
    """Encourage consistent lane width along each lane.

    Penalizes variation in the distance between left and right boundaries
    along the longitudinal direction. This encodes the geometric prior that
    lane width changes smoothly.

    Args:
        pred_points: (B, total_queries, 2) predicted BEV coordinates.
            Queries must be ordered: lane_0_left_pts, lane_0_right_pts,
            lane_1_left_pts, lane_1_right_pts, ...
        num_lanes: Number of lanes.
        points_per_line: Points per boundary line.

    Returns:
        Scalar loss penalizing non-smooth lane widths.
    """
    B = pred_points.shape[0]
    num_lane_queries = num_lanes * 2 * points_per_line

    lane_pts = pred_points[:, :num_lane_queries]
    lane_pts = lane_pts.view(B, num_lanes, 2, points_per_line, 2)

    left = lane_pts[:, :, 0, :, :]  # (B, num_lanes, points_per_line, 2)
    right = lane_pts[:, :, 1, :, :]  # (B, num_lanes, points_per_line, 2)

    # Width at each longitudinal station (eps for gradient stability at zero)
    widths = (left - right).pow(2).sum(-1).clamp(min=1e-6).sqrt()

    # Penalize width variation (Smooth-L1 for outlier robustness at merges/splits)
    width_diff = widths[:, :, 1:] - widths[:, :, :-1]
    return F.smooth_l1_loss(width_diff, torch.zeros_like(width_diff), beta=0.01)
