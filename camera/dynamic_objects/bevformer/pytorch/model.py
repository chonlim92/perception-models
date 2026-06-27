"""BEVFormer: Complete model combining backbone, BEV encoder, decoder, and detection head.

This module implements the full BEVFormer architecture for multi-camera 3D object
detection. It transforms surround-view camera images into a Bird's-Eye-View (BEV)
representation using spatiotemporal transformers, then performs object detection
via a DETR-style transformer decoder with iterative bounding box refinement.

Reference: BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera
           Images via Spatiotemporal Transformers (ECCV 2022)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .backbone import ResNetFPN
from .spatial_cross_attention import BEVFormerSpatialCrossAttention
from .temporal_self_attention import TemporalSelfAttention

__all__ = [
    "BEVFormer",
    "BEVEncoder",
    "TransformerDecoder",
    "HierarchicalLanePositionalEmbedding",
    "LaneDetectionDecoder",
    "DetectionHead",
    "HungarianMatcher",
    "BEVFormerLoss",
]


# =============================================================================
# BEV Encoder
# =============================================================================


class BEVEncoderLayer(nn.Module):
    """Single BEV encoder layer: temporal self-attention + spatial cross-attention + FFN."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_points_spatial: int = 4,
        num_points_temporal: int = 4,
        num_levels: int = 4,
        num_cams: int = 6,
        num_ref_points: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        bev_h: int = 200,
        bev_w: int = 200,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        """Initialize a BEV encoder layer.

        Args:
            embed_dim: Feature embedding dimension.
            num_heads: Number of attention heads.
            num_points_spatial: Sampling points for spatial cross-attention.
            num_points_temporal: Sampling points for temporal self-attention.
            num_levels: Number of multi-scale feature levels.
            num_cams: Number of camera views.
            num_ref_points: Number of 3D reference points along z-axis.
            ffn_dim: Feed-forward network hidden dimension.
            dropout: Dropout rate.
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            pc_range: Point cloud range (x_min, y_min, z_min, x_max, y_max, z_max).
        """
        super().__init__()

        self.temporal_self_attn = TemporalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_points=num_points_temporal,
            bev_h=bev_h,
            bev_w=bev_w,
            pc_range=pc_range,
        )

        self.spatial_cross_attn = BEVFormerSpatialCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points_spatial,
            num_cams=num_cams,
            num_ref_points=num_ref_points,
            pc_range=pc_range,
            bev_h=bev_h,
            bev_w=bev_w,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        bev_queries: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
        ego_motion: Optional[torch.Tensor],
        img_shape: Tuple[int, int] = (900, 1600),
    ) -> torch.Tensor:
        """Forward pass of one encoder layer.

        Args:
            bev_queries: BEV queries (B, bev_h*bev_w, embed_dim).
            multi_scale_features: Multi-scale image features from backbone.
            intrinsics: Camera intrinsics (B, num_cams, 3, 3).
            extrinsics: Camera extrinsics (B, num_cams, 4, 4).
            prev_bev: Previous BEV features (B, bev_h*bev_w, embed_dim) or None.
            ego_motion: Ego motion matrix (B, 4, 4) or None.
            img_shape: Image shape (H, W).

        Returns:
            Updated BEV features (B, bev_h*bev_w, embed_dim).
        """
        # Temporal self-attention
        bev_queries = self.temporal_self_attn(
            bev_queries=bev_queries,
            prev_bev=prev_bev,
            ego_motion=ego_motion,
        )

        # Spatial cross-attention
        bev_queries = self.spatial_cross_attn(
            bev_queries=bev_queries,
            multi_scale_features=multi_scale_features,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            img_shape=img_shape,
        )

        # Feed-forward network with residual
        residual = bev_queries
        bev_queries = self.ffn(bev_queries)
        bev_queries = residual + bev_queries
        bev_queries = self.ffn_norm(bev_queries)

        return bev_queries


class BEVEncoder(nn.Module):
    """BEV encoder with stacked temporal-spatial transformer layers.

    Transforms multi-camera image features into a unified BEV representation
    through iterative refinement with temporal fusion.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        num_points_spatial: int = 4,
        num_points_temporal: int = 4,
        num_levels: int = 4,
        num_cams: int = 6,
        num_ref_points: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        bev_h: int = 200,
        bev_w: int = 200,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        """Initialize BEV encoder.

        Args:
            embed_dim: Feature embedding dimension.
            num_encoder_layers: Number of stacked encoder layers.
            num_heads: Number of attention heads.
            num_points_spatial: Sampling points for spatial cross-attention.
            num_points_temporal: Sampling points for temporal self-attention.
            num_levels: Number of multi-scale feature levels.
            num_cams: Number of cameras.
            num_ref_points: Number of 3D reference points per BEV query.
            ffn_dim: FFN hidden dimension.
            dropout: Dropout rate.
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            pc_range: Point cloud range.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Learnable BEV query embeddings
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dim)

        # Stacked encoder layers
        self.layers = nn.ModuleList([
            BEVEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_points_spatial=num_points_spatial,
                num_points_temporal=num_points_temporal,
                num_levels=num_levels,
                num_cams=num_cams,
                num_ref_points=num_ref_points,
                ffn_dim=ffn_dim,
                dropout=dropout,
                bev_h=bev_h,
                bev_w=bev_w,
                pc_range=pc_range,
            )
            for _ in range(num_encoder_layers)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize BEV embedding with uniform distribution."""
        nn.init.uniform_(self.bev_embedding.weight, -1.0, 1.0)

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
        ego_motion: Optional[torch.Tensor],
        img_shape: Tuple[int, int] = (900, 1600),
    ) -> torch.Tensor:
        """Forward pass through BEV encoder.

        Args:
            multi_scale_features: Multi-scale image features from FPN.
            intrinsics: Camera intrinsics (B, num_cams, 3, 3).
            extrinsics: Camera extrinsics (B, num_cams, 4, 4).
            prev_bev: Previous BEV features or None for first frame.
            ego_motion: Ego motion transformation (B, 4, 4) or None.
            img_shape: Original image shape (H, W).

        Returns:
            BEV features (B, bev_h*bev_w, embed_dim).
        """
        batch_size = intrinsics.shape[0]

        # Initialize BEV queries from learnable embeddings
        bev_queries = self.bev_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, bev_h*bev_w, embed_dim)

        # Pass through encoder layers
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

        return bev_queries


# =============================================================================
# Transformer Decoder
# =============================================================================


class DecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """Initialize decoder layer.

        Args:
            embed_dim: Feature dimension.
            num_heads: Number of attention heads.
            ffn_dim: FFN hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        # Self-attention among object queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Cross-attention: object queries attend to BEV features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        bev_features: torch.Tensor,
        query_pos: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Object queries (B, num_queries, embed_dim).
            bev_features: BEV features (B, bev_h*bev_w, embed_dim).
            query_pos: Positional embeddings for queries (B, num_queries, embed_dim).
            self_attn_mask: Additive float mask (Q, Q) for self-attention. -inf blocks.

        Returns:
            Updated queries (B, num_queries, embed_dim).
        """
        # Self-attention
        q = k = query + query_pos
        residual = query
        query_out, _ = self.self_attn(q, k, query, attn_mask=self_attn_mask)
        query = residual + self.self_attn_dropout(query_out)
        query = self.self_attn_norm(query)

        # Cross-attention to BEV features
        residual = query
        q = query + query_pos
        cross_out, _ = self.cross_attn(q, bev_features, bev_features)
        query = residual + self.cross_attn_dropout(cross_out)
        query = self.cross_attn_norm(query)

        # FFN
        residual = query
        query = residual + self.ffn(query)
        query = self.ffn_norm(query)

        return query


class TransformerDecoder(nn.Module):
    """Transformer decoder with iterative bounding box refinement.

    Uses learnable object queries to detect objects from BEV features.
    Each decoder layer refines the predicted bounding boxes which serve
    as reference points for the next layer.
    """

    def __init__(
        self,
        num_decoder_layers: int = 6,
        num_queries: int = 900,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        iterative_bbox_refinement: bool = True,
        code_size: int = 10,
    ) -> None:
        """Initialize transformer decoder.

        Args:
            num_decoder_layers: Number of decoder layers.
            num_queries: Number of object queries (max detections).
            embed_dim: Feature dimension.
            num_heads: Attention heads.
            ffn_dim: FFN hidden dimension.
            dropout: Dropout rate.
            iterative_bbox_refinement: If True, refine boxes at each layer.
            code_size: Bounding box code size.
        """
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.iterative_bbox_refinement = iterative_bbox_refinement

        # Learnable object queries and positional embeddings
        self.query_embedding = nn.Embedding(num_queries, embed_dim)
        self.query_pos_embedding = nn.Embedding(num_queries, embed_dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        # Reference point generation from query position
        self.reference_points_proj = nn.Linear(embed_dim, 3)

        # Regression heads per layer for iterative refinement
        if iterative_bbox_refinement:
            self.reg_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dim, code_size),
                )
                for _ in range(num_decoder_layers)
            ])
        else:
            self.reg_branches = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.uniform_(self.query_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.query_pos_embedding.weight, -1.0, 1.0)
        nn.init.xavier_uniform_(self.reference_points_proj.weight)
        nn.init.zeros_(self.reference_points_proj.bias)

        if self.reg_branches is not None:
            for branch in self.reg_branches:
                for module in branch.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)

    def forward(
        self, bev_features: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass through decoder.

        Args:
            bev_features: BEV features (B, bev_h*bev_w, embed_dim).

        Returns:
            Tuple of:
                - List of intermediate query outputs, one per layer.
                  Each has shape (B, num_queries, embed_dim).
                - Final reference points (B, num_queries, 3) for loss computation.
        """
        batch_size = bev_features.shape[0]

        # Initialize queries
        query = self.query_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        query_pos = self.query_pos_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Initial reference points from query positions
        reference_points = self.reference_points_proj(query_pos).sigmoid()

        intermediate_outputs = []

        for layer_idx, layer in enumerate(self.layers):
            query = layer(query, bev_features, query_pos)
            intermediate_outputs.append(query)

            # Iterative refinement: update reference points
            if self.iterative_bbox_refinement and self.reg_branches is not None:
                reg_offset = self.reg_branches[layer_idx](query)
                # Update reference points with predicted center offsets
                new_reference = reference_points.clone()
                new_reference = new_reference + reg_offset[..., :3]
                reference_points = new_reference.detach()

        return intermediate_outputs, reference_points


# =============================================================================
# Hierarchical Lane Positional Embeddings
# =============================================================================


class HierarchicalLanePositionalEmbedding(nn.Module):
    """Hierarchical positional embeddings encoding lane -> line -> point structure.

    Encodes the structural relationship between lanes, their boundary lines,
    and individual points along each line. The final positional embedding for
    each query is the sum of its lane-level, line-level, and point-level
    embeddings, giving the transformer explicit awareness of the hierarchical
    map topology.

    Query layout (in order):
        [0, num_lane_queries): Lane queries organized as
            lane_0_left_pt0, ..., lane_0_left_pt19,
            lane_0_right_pt0, ..., lane_0_right_pt19,
            lane_1_left_pt0, ..., (25 lanes × 2 lines × 20 points = 1000)
        [num_lane_queries, total_queries): Other line queries organized as
            line_0_pt0, ..., line_0_pt19,
            line_1_pt0, ..., (num_other_lines × 20 points)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_lanes: int = 25,
        points_per_line: int = 20,
        num_other_lines: int = 0,
        pos_drop: float = 0.1,
    ) -> None:
        """Initialize hierarchical lane positional embeddings.

        Args:
            embed_dim: Embedding dimension (must match decoder d_model).
            num_lanes: Number of lanes (each has left + right boundary).
            points_per_line: Number of points sampled per line.
            num_other_lines: Number of additional non-lane lines (e.g.,
                road boundaries, crosswalks).
            pos_drop: Dropout rate on summed positional embedding.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_lanes = num_lanes
        self.points_per_line = points_per_line
        self.num_other_lines = num_other_lines

        self.num_lane_queries = num_lanes * 2 * points_per_line
        self.num_other_queries = num_other_lines * points_per_line
        self.total_queries = self.num_lane_queries + self.num_other_queries
        self.num_total_lines = num_lanes * 2 + num_other_lines

        # Lane-level embedding: which lane (0..num_lanes-1) or other-line group
        self.lane_embedding = nn.Embedding(num_lanes + num_other_lines, embed_dim)

        # Line-type embedding: 0=left boundary, 1=right boundary, 2=other
        self.line_type_embedding = nn.Embedding(3, embed_dim)

        # Point-position embedding: hybrid sinusoidal + learned for ordinal prior
        self._build_sinusoidal_base(points_per_line, embed_dim)
        self.point_residual = nn.Embedding(points_per_line, embed_dim)

        # Learnable content queries (one per structural slot)
        self.content_embedding = nn.Embedding(self.total_queries, embed_dim)

        # LayerNorm stabilizes the summed positional embedding
        self.pos_layer_norm = nn.LayerNorm(embed_dim)
        self.pos_dropout = nn.Dropout(pos_drop)

        self._cached_pos: Optional[torch.Tensor] = None
        self._init_weights()
        self._build_index_tables()

    def _build_sinusoidal_base(self, num_points: int, embed_dim: int) -> None:
        """Pre-compute fixed sinusoidal positional encoding for point ordering."""
        pe = torch.zeros(num_points, embed_dim)
        position = torch.arange(0, num_points, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:embed_dim // 2])
        self.register_buffer("point_sinusoidal", pe)

    def _point_embedding(self, point_ids: torch.Tensor) -> torch.Tensor:
        """Hybrid point embedding: sinusoidal base + learned residual."""
        return self.point_sinusoidal[point_ids] + self.point_residual(point_ids)

    def _init_weights(self) -> None:
        nn.init.normal_(self.lane_embedding.weight, std=0.02)
        nn.init.normal_(self.line_type_embedding.weight, std=0.02)
        nn.init.normal_(self.point_residual.weight, std=0.01)
        nn.init.normal_(self.content_embedding.weight, std=0.02)

    def _build_index_tables(self) -> None:
        """Pre-compute index tensors for efficient lookup."""
        lane_ids = []
        line_type_ids = []
        point_ids = []

        # Lane queries: 25 lanes × 2 lines × 20 points
        for lane_idx in range(self.num_lanes):
            for line_type in range(2):  # 0=left, 1=right
                for pt_idx in range(self.points_per_line):
                    lane_ids.append(lane_idx)
                    line_type_ids.append(line_type)
                    point_ids.append(pt_idx)

        # Other line queries
        for line_idx in range(self.num_other_lines):
            for pt_idx in range(self.points_per_line):
                lane_ids.append(self.num_lanes + line_idx)
                line_type_ids.append(2)  # type=other
                point_ids.append(pt_idx)

        self.register_buffer("lane_ids", torch.tensor(lane_ids, dtype=torch.long))
        self.register_buffer("line_type_ids", torch.tensor(line_type_ids, dtype=torch.long))
        self.register_buffer("point_ids", torch.tensor(point_ids, dtype=torch.long))

        # Pre-compute lane_mask as buffer (avoids per-call allocation)
        lane_mask = torch.zeros(self.total_queries, dtype=torch.bool)
        lane_mask[: self.num_lane_queries] = True
        self.register_buffer("lane_mask", lane_mask)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute hierarchical positional and content embeddings.

        Returns:
            Tuple of:
                - pos_embed: (total_queries, embed_dim) positional embeddings
                - content_embed: (total_queries, embed_dim) content queries
        """
        if not self.training and self._cached_pos is not None:
            return self._cached_pos, self.content_embedding.weight

        pos_embed = (
            self.lane_embedding(self.lane_ids)
            + self.line_type_embedding(self.line_type_ids)
            + self._point_embedding(self.point_ids)
        )
        pos_embed = self.pos_dropout(self.pos_layer_norm(pos_embed))

        if not self.training:
            self._cached_pos = pos_embed

        content_embed = self.content_embedding.weight
        return pos_embed, content_embed

    def get_lane_mask(self) -> torch.Tensor:
        """Return boolean mask identifying lane queries vs other-line queries."""
        return self.lane_mask


class LaneDetectionDecoder(nn.Module):
    """Transformer decoder with hierarchical lane-structured positional embeddings.

    Designed for lane detection from BEV features. Each query corresponds to a
    specific point on a specific line (left/right) of a specific lane, giving
    the transformer explicit structural knowledge of the output topology.

    Output organization:
        - 25 lanes × 2 boundary lines × 20 points = 1000 lane queries
        - Additional non-lane polylines × 20 points each
    """

    def __init__(
        self,
        num_decoder_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        num_lanes: int = 25,
        points_per_line: int = 20,
        num_other_lines: int = 0,
    ) -> None:
        """Initialize lane detection decoder.

        Args:
            num_decoder_layers: Number of transformer decoder layers.
            embed_dim: Feature dimension.
            num_heads: Number of attention heads.
            ffn_dim: FFN hidden dimension.
            dropout: Dropout rate.
            num_lanes: Number of lanes (each with left+right boundary).
            points_per_line: Points sampled per line (default 20).
            num_other_lines: Non-lane polylines (road edges, crosswalks).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_lanes = num_lanes
        self.points_per_line = points_per_line
        self.num_other_lines = num_other_lines

        # Hierarchical positional embeddings
        self.pos_embed = HierarchicalLanePositionalEmbedding(
            embed_dim=embed_dim,
            num_lanes=num_lanes,
            points_per_line=points_per_line,
            num_other_lines=num_other_lines,
        )

        self.total_queries = self.pos_embed.total_queries

        # Decoder layers (reuse DecoderLayer from TransformerDecoder)
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Per-point 2D coordinate regression (x, y in BEV)
        self.point_reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2),
        )

        # Per-lane confidence (pooled over all points of a lane)
        num_total_lines = num_lanes * 2 + num_other_lines
        self.lane_cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.point_reg_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.lane_cls_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_decoupled_self_attn_mask(self) -> torch.Tensor:
        """Build block-diagonal self-attention mask (decoupled attention).

        Points on the same line can attend to each other but not to other lines.
        """
        num_total_lines = self.num_lanes * 2 + self.num_other_lines
        total_q = num_total_lines * self.points_per_line
        mask = torch.full((total_q, total_q), float("-inf"))
        for line_idx in range(num_total_lines):
            start = line_idx * self.points_per_line
            end = start + self.points_per_line
            mask[start:end, start:end] = 0.0
        return mask

    def forward(
        self, bev_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for lane detection.

        Args:
            bev_features: BEV features (B, bev_h*bev_w, embed_dim).

        Returns:
            Dict with:
                'pred_points': (B, total_queries, 2) predicted BEV coordinates
                'lane_logits': (B, num_total_lines) per-line confidence scores
                'intermediate_points': list of (B, total_queries, 2) per layer
        """
        batch_size = bev_features.shape[0]

        # Get hierarchical positional and content embeddings
        query_pos, query_content = self.pos_embed()

        # Expand for batch
        query = query_content.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos_expanded = query_pos.unsqueeze(0).expand(batch_size, -1, -1)

        # Build decoupled self-attention mask
        self_attn_mask = self._build_decoupled_self_attn_mask().to(
            device=query.device
        )

        intermediate_points = []

        for layer in self.layers:
            query = layer(
                query, bev_features, query_pos_expanded,
                self_attn_mask=self_attn_mask,
            )
            # Intermediate point predictions for auxiliary loss
            pts = self.point_reg_head(self.norm(query)).sigmoid()
            intermediate_points.append(pts)

        # Final predictions
        query = self.norm(query)
        pred_points = self.point_reg_head(query).sigmoid()  # (B, Q, 2)

        # Per-line confidence: pool points belonging to each line (vectorized)
        num_total_lines = self.num_lanes * 2 + self.num_other_lines
        line_features = query.reshape(
            batch_size, num_total_lines, self.points_per_line, self.embed_dim
        ).mean(dim=2)  # (B, num_lines, embed_dim)

        lane_logits = self.lane_cls_head(line_features).squeeze(-1)  # (B, num_lines)

        return {
            "pred_points": pred_points,
            "lane_logits": lane_logits,
            "intermediate_points": intermediate_points,
        }


# =============================================================================
# Detection Head
# =============================================================================


class DetectionHead(nn.Module):
    """Detection head that predicts class scores and bounding box parameters.

    Applied to the output of each decoder layer to enable auxiliary supervision.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 10,
        code_size: int = 10,
        num_reg_fcs: int = 2,
    ) -> None:
        """Initialize detection head.

        Args:
            embed_dim: Input feature dimension.
            num_classes: Number of object classes.
            code_size: Bounding box parameterization size.
            num_reg_fcs: Number of FC layers in regression branch.
        """
        super().__init__()
        self.num_classes = num_classes
        self.code_size = code_size

        # Classification branch
        cls_layers: List[nn.Module] = []
        for _ in range(num_reg_fcs):
            cls_layers.append(nn.Linear(embed_dim, embed_dim))
            cls_layers.append(nn.LayerNorm(embed_dim))
            cls_layers.append(nn.ReLU(inplace=True))
        cls_layers.append(nn.Linear(embed_dim, num_classes))
        self.cls_branch = nn.Sequential(*cls_layers)

        # Regression branch
        reg_layers: List[nn.Module] = []
        for _ in range(num_reg_fcs):
            reg_layers.append(nn.Linear(embed_dim, embed_dim))
            reg_layers.append(nn.ReLU(inplace=True))
        reg_layers.append(nn.Linear(embed_dim, code_size))
        self.reg_branch = nn.Sequential(*reg_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with Xavier uniform and zero bias for final layers."""
        for module in self.cls_branch.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.reg_branch.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Bias initialization for classification (focal loss prior)
        final_cls = self.cls_branch[-1]
        assert isinstance(final_cls, nn.Linear)
        nn.init.constant_(final_cls.bias, -4.6)  # -log((1-0.01)/0.01)

    def forward(
        self, query_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            query_features: Decoder output (B, num_queries, embed_dim).

        Returns:
            Tuple of:
                - cls_scores: (B, num_queries, num_classes)
                - bbox_preds: (B, num_queries, code_size)
        """
        cls_scores = self.cls_branch(query_features)
        bbox_preds = self.reg_branch(query_features)
        return cls_scores, bbox_preds


# =============================================================================
# Loss Components
# =============================================================================


class HungarianMatcher(nn.Module):
    """Hungarian algorithm-based bipartite matcher between predictions and GT.

    Finds the optimal assignment between predictions and ground truth objects
    that minimizes the total matching cost (classification + bbox regression).
    """

    def __init__(
        self,
        cls_cost: float = 2.0,
        bbox_cost: float = 0.25,
        iou_cost: float = 0.0,
    ) -> None:
        """Initialize matcher.

        Args:
            cls_cost: Weight for classification cost.
            bbox_cost: Weight for L1 bbox cost.
            iou_cost: Weight for IoU cost (unused in base config).
        """
        super().__init__()
        self.cls_cost = cls_cost
        self.bbox_cost = bbox_cost
        self.iou_cost = iou_cost

    @torch.no_grad()
    def forward(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Perform Hungarian matching.

        Args:
            cls_scores: Predicted class logits (B, num_queries, num_classes).
            bbox_preds: Predicted bboxes (B, num_queries, code_size).
            gt_labels: GT labels, list-like or padded (B, max_gt).
            gt_bboxes: GT bboxes (B, max_gt, code_size).

        Returns:
            List of (pred_indices, gt_indices) tuples, one per batch sample.
        """
        batch_size, num_queries, _ = cls_scores.shape
        indices = []

        for b in range(batch_size):
            # Get valid GT for this sample
            valid_mask = gt_labels[b] >= 0
            num_gt = valid_mask.sum().item()

            if num_gt == 0:
                indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=cls_scores.device),
                        torch.tensor([], dtype=torch.long, device=cls_scores.device),
                    )
                )
                continue

            gt_lab = gt_labels[b][valid_mask]  # (num_gt,)
            gt_box = gt_bboxes[b][valid_mask]  # (num_gt, code_size)

            # Classification cost: focal-loss-based cost
            pred_scores = cls_scores[b].sigmoid()  # (num_queries, num_classes)
            # Negative focal cost for the target class
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (
                (1 - alpha) * (pred_scores ** gamma) * (-(1 - pred_scores + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - pred_scores) ** gamma) * (-(pred_scores + 1e-8).log())
            )
            # Cost for each pred-gt pair: (num_queries, num_gt)
            cls_cost_matrix = (
                pos_cost_class[:, gt_lab] - neg_cost_class[:, gt_lab]
            )

            # Bbox L1 cost: (num_queries, num_gt)
            bbox_cost_matrix = torch.cdist(
                bbox_preds[b], gt_box, p=1
            )

            # Total cost
            cost_matrix = (
                self.cls_cost * cls_cost_matrix + self.bbox_cost * bbox_cost_matrix
            )

            # Hungarian matching
            cost_np = cost_matrix.detach().cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(cost_np)
            indices.append(
                (
                    torch.tensor(pred_idx, dtype=torch.long, device=cls_scores.device),
                    torch.tensor(gt_idx, dtype=torch.long, device=cls_scores.device),
                )
            )

        return indices


class BEVFormerLoss(nn.Module):
    """Combined loss for BEVFormer: focal loss + L1 regression with Hungarian matching.

    Supports auxiliary losses from intermediate decoder layers.
    """

    def __init__(
        self,
        num_classes: int = 10,
        code_size: int = 10,
        cls_weight: float = 2.0,
        bbox_weight: float = 0.25,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_cost: float = 2.0,
        bbox_cost: float = 0.25,
    ) -> None:
        """Initialize loss module.

        Args:
            num_classes: Number of object classes.
            code_size: Bounding box code size.
            cls_weight: Classification loss weight.
            bbox_weight: Bbox regression loss weight.
            focal_alpha: Focal loss alpha.
            focal_gamma: Focal loss gamma.
            cls_cost: Hungarian matcher classification cost.
            bbox_cost: Hungarian matcher bbox cost.
        """
        super().__init__()
        self.num_classes = num_classes
        self.code_size = code_size
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.matcher = HungarianMatcher(
            cls_cost=cls_cost, bbox_cost=bbox_cost
        )

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Compute sigmoid focal loss.

        Args:
            pred: Predicted logits (N, num_classes).
            target: One-hot encoded targets (N, num_classes).
            alpha: Balancing factor.
            gamma: Focusing parameter.

        Returns:
            Scalar focal loss.
        """
        pred_sigmoid = pred.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        )
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * (
            (1 - p_t) ** gamma
        )
        loss = (focal_weight * ce_loss).sum()
        return loss

    def _compute_loss_single_layer(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for one decoder layer output.

        Args:
            cls_scores: (B, num_queries, num_classes).
            bbox_preds: (B, num_queries, code_size).
            gt_labels: (B, max_gt) with -1 for padding.
            gt_bboxes: (B, max_gt, code_size).

        Returns:
            Dict with 'cls_loss' and 'bbox_loss'.
        """
        batch_size, num_queries, _ = cls_scores.shape
        device = cls_scores.device

        # Hungarian matching
        indices = self.matcher(cls_scores, bbox_preds, gt_labels, gt_bboxes)

        # Classification loss
        total_cls_loss = torch.tensor(0.0, device=device)
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_num_pos = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            # Build classification target: all background by default
            cls_target = torch.zeros(
                num_queries, self.num_classes, device=device
            )
            if len(pred_idx) > 0:
                valid_mask = gt_labels[b] >= 0
                gt_lab = gt_labels[b][valid_mask]
                matched_labels = gt_lab[gt_idx]
                # One-hot encode matched labels
                cls_target[pred_idx] = F.one_hot(
                    matched_labels, self.num_classes
                ).float()
                total_num_pos += len(pred_idx)

            # Focal loss for this sample
            total_cls_loss = total_cls_loss + self.focal_loss(
                cls_scores[b], cls_target, self.focal_alpha, self.focal_gamma
            )

            # Bbox regression loss (only for matched pairs)
            if len(pred_idx) > 0:
                valid_mask = gt_labels[b] >= 0
                gt_box = gt_bboxes[b][valid_mask]
                matched_pred = bbox_preds[b][pred_idx]
                matched_gt = gt_box[gt_idx]
                total_bbox_loss = total_bbox_loss + F.l1_loss(
                    matched_pred, matched_gt, reduction="sum"
                )

        # Normalize by number of positive samples
        num_pos = max(total_num_pos, 1)
        cls_loss = total_cls_loss / num_pos
        bbox_loss = total_bbox_loss / num_pos

        return {"cls_loss": cls_loss, "bbox_loss": bbox_loss}

    def forward(
        self,
        all_cls_scores: List[torch.Tensor],
        all_bbox_preds: List[torch.Tensor],
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss including auxiliary losses from all decoder layers.

        Args:
            all_cls_scores: List of cls_scores per decoder layer.
            all_bbox_preds: List of bbox_preds per decoder layer.
            gt_labels: (B, max_gt) with -1 padding.
            gt_bboxes: (B, max_gt, code_size).

        Returns:
            Dict with total loss and individual loss components.
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        device = all_cls_scores[0].device
        total_loss = torch.tensor(0.0, device=device)

        for layer_idx, (cls_scores, bbox_preds) in enumerate(
            zip(all_cls_scores, all_bbox_preds)
        ):
            layer_losses = self._compute_loss_single_layer(
                cls_scores, bbox_preds, gt_labels, gt_bboxes
            )

            weighted_cls = self.cls_weight * layer_losses["cls_loss"]
            weighted_bbox = self.bbox_weight * layer_losses["bbox_loss"]

            loss_dict[f"cls_loss_layer{layer_idx}"] = weighted_cls
            loss_dict[f"bbox_loss_layer{layer_idx}"] = weighted_bbox
            total_loss = total_loss + weighted_cls + weighted_bbox

        loss_dict["total_loss"] = total_loss
        return loss_dict


# =============================================================================
# Full BEVFormer Model
# =============================================================================


class BEVFormer(nn.Module):
    """BEVFormer: Multi-camera 3D object detection via spatiotemporal BEV transformers.

    Combines a CNN backbone with FPN, a BEV encoder using spatial cross-attention
    and temporal self-attention, a transformer decoder for object detection, and
    a detection head with Hungarian matching loss.
    """

    def __init__(
        self,
        # Backbone
        backbone_out_channels: int = 256,
        backbone_pretrained: bool = True,
        backbone_frozen_stages: int = 1,
        # BEV Encoder
        embed_dim: int = 256,
        bev_h: int = 200,
        bev_w: int = 200,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        num_points_spatial: int = 4,
        num_points_temporal: int = 4,
        num_levels: int = 4,
        num_cams: int = 6,
        num_ref_points: int = 4,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        # Decoder
        num_decoder_layers: int = 6,
        num_queries: int = 900,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        iterative_bbox_refinement: bool = True,
        # Head
        num_classes: int = 10,
        code_size: int = 10,
        num_reg_fcs: int = 2,
        # Loss
        cls_weight: float = 2.0,
        bbox_weight: float = 0.25,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cls_cost: float = 2.0,
        bbox_cost: float = 0.25,
        # Inference
        score_threshold: float = 0.1,
        max_detections: int = 300,
    ) -> None:
        """Initialize BEVFormer.

        Args:
            backbone_out_channels: FPN output channels.
            backbone_pretrained: Use pretrained backbone.
            backbone_frozen_stages: Freeze N backbone stages.
            embed_dim: Transformer embedding dimension.
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            num_encoder_layers: BEV encoder depth.
            num_heads: Attention heads.
            num_points_spatial: Spatial sampling points.
            num_points_temporal: Temporal sampling points.
            num_levels: Multi-scale feature levels.
            num_cams: Number of cameras.
            num_ref_points: 3D reference points per query.
            pc_range: Point cloud range.
            num_decoder_layers: Decoder depth.
            num_queries: Object query count.
            ffn_dim: FFN hidden size.
            dropout: Dropout rate.
            iterative_bbox_refinement: Use iterative refinement.
            num_classes: Detection classes.
            code_size: Box parameterization size.
            num_reg_fcs: FC layers in head branches.
            cls_weight: Classification loss weight.
            bbox_weight: Regression loss weight.
            focal_alpha: Focal loss alpha.
            focal_gamma: Focal loss gamma.
            cls_cost: Matcher classification cost.
            bbox_cost: Matcher bbox cost.
            score_threshold: Inference score threshold.
            max_detections: Max detections per frame.
        """
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cams = num_cams
        self.embed_dim = embed_dim
        self.score_threshold = score_threshold
        self.max_detections = max_detections

        # Backbone: ResNet-101 + FPN
        self.backbone = ResNetFPN(
            out_channels=backbone_out_channels,
            pretrained=backbone_pretrained,
            frozen_stages=backbone_frozen_stages,
        )

        # Channel projection if backbone output differs from embed_dim
        if backbone_out_channels != embed_dim:
            self.input_proj = nn.Conv2d(
                backbone_out_channels, embed_dim, kernel_size=1
            )
        else:
            self.input_proj = None

        # BEV Encoder
        self.bev_encoder = BEVEncoder(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            num_points_spatial=num_points_spatial,
            num_points_temporal=num_points_temporal,
            num_levels=num_levels,
            num_cams=num_cams,
            num_ref_points=num_ref_points,
            ffn_dim=ffn_dim,
            dropout=dropout,
            bev_h=bev_h,
            bev_w=bev_w,
            pc_range=pc_range,
        )

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            num_decoder_layers=num_decoder_layers,
            num_queries=num_queries,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            iterative_bbox_refinement=iterative_bbox_refinement,
            code_size=code_size,
        )

        # Detection Head (shared across decoder layers for auxiliary losses)
        self.head = DetectionHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            code_size=code_size,
            num_reg_fcs=num_reg_fcs,
        )

        # Loss
        self.loss = BEVFormerLoss(
            num_classes=num_classes,
            code_size=code_size,
            cls_weight=cls_weight,
            bbox_weight=bbox_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            cls_cost=cls_cost,
            bbox_cost=bbox_cost,
        )

    def _extract_intrinsics_3x3(self, intrinsics: torch.Tensor) -> torch.Tensor:
        """Extract 3x3 intrinsic matrix from 4x4 if needed.

        Args:
            intrinsics: (B, num_cams, 3, 3) or (B, num_cams, 4, 4).

        Returns:
            (B, num_cams, 3, 3) intrinsic matrices.
        """
        if intrinsics.shape[-1] == 4:
            return intrinsics[:, :, :3, :3]
        return intrinsics

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        ego_motion: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass through the full BEVFormer model.

        Args:
            images: Multi-camera images (B, num_cams, 3, H, W).
            intrinsics: Camera intrinsics (B, num_cams, 4, 4) or (B, num_cams, 3, 3).
            extrinsics: Camera extrinsics (B, num_cams, 4, 4).
            ego_motion: Ego motion from previous to current frame (B, 4, 4).
            prev_bev: Previous BEV features (B, bev_h*bev_w, embed_dim) or None.

        Returns:
            Tuple of:
                - predictions dict with keys:
                    'all_cls_scores': List of (B, num_queries, num_classes) per layer
                    'all_bbox_preds': List of (B, num_queries, code_size) per layer
                    'cls_scores': final layer scores
                    'bbox_preds': final layer boxes
                - new_prev_bev: BEV features for next frame (B, bev_h*bev_w, embed_dim)
        """
        img_shape = (images.shape[3], images.shape[4])  # (H, W)

        # Extract multi-scale features from backbone
        multi_scale_features = self.backbone(images)

        # Project channels if needed
        if self.input_proj is not None:
            multi_scale_features = [
                self.input_proj(feat) for feat in multi_scale_features
            ]

        # Extract 3x3 intrinsics
        intrinsics_3x3 = self._extract_intrinsics_3x3(intrinsics)

        # BEV encoding with temporal fusion
        bev_features = self.bev_encoder(
            multi_scale_features=multi_scale_features,
            intrinsics=intrinsics_3x3,
            extrinsics=extrinsics,
            prev_bev=prev_bev,
            ego_motion=ego_motion if prev_bev is not None else None,
            img_shape=img_shape,
        )

        # Transformer decoder
        intermediate_outputs, reference_points = self.decoder(bev_features)

        # Apply detection head to each intermediate output
        all_cls_scores = []
        all_bbox_preds = []
        for layer_output in intermediate_outputs:
            cls_scores, bbox_preds = self.head(layer_output)
            all_cls_scores.append(cls_scores)
            all_bbox_preds.append(bbox_preds)

        predictions = {
            "all_cls_scores": all_cls_scores,
            "all_bbox_preds": all_bbox_preds,
            "cls_scores": all_cls_scores[-1],
            "bbox_preds": all_bbox_preds[-1],
            "reference_points": reference_points,
        }

        return predictions, bev_features

    def forward_train(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        ego_motion: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
        gt_bboxes_3d: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass for training with loss computation.

        Args:
            images: (B, num_cams, 3, H, W).
            intrinsics: (B, num_cams, 4, 4).
            extrinsics: (B, num_cams, 4, 4).
            ego_motion: (B, 4, 4).
            prev_bev: Previous BEV features or None.
            gt_bboxes_3d: GT boxes (B, max_gt, code_size) padded with zeros.
            gt_labels: GT labels (B, max_gt) with -1 for padding.

        Returns:
            Tuple of (loss_dict, new_prev_bev).
        """
        predictions, new_bev = self.forward(
            images, intrinsics, extrinsics, ego_motion, prev_bev
        )

        # Compute losses with auxiliary supervision
        loss_dict = self.loss(
            all_cls_scores=predictions["all_cls_scores"],
            all_bbox_preds=predictions["all_bbox_preds"],
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes_3d,
        )

        return loss_dict, new_bev

    @torch.no_grad()
    def forward_test(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        ego_motion: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass for inference with post-processing.

        Args:
            images: (B, num_cams, 3, H, W).
            intrinsics: (B, num_cams, 4, 4).
            extrinsics: (B, num_cams, 4, 4).
            ego_motion: (B, 4, 4).
            prev_bev: Previous BEV features or None.

        Returns:
            Tuple of (detections_dict, new_prev_bev) where detections_dict has:
                'scores': (B, K) detection scores
                'labels': (B, K) class labels
                'boxes': (B, K, code_size) bounding boxes
        """
        predictions, new_bev = self.forward(
            images, intrinsics, extrinsics, ego_motion, prev_bev
        )

        # Post-process: take final layer predictions
        cls_scores = predictions["cls_scores"].sigmoid()  # (B, Q, C)
        bbox_preds = predictions["bbox_preds"]  # (B, Q, code_size)

        batch_size = cls_scores.shape[0]
        results_scores = []
        results_labels = []
        results_boxes = []

        for b in range(batch_size):
            # Get max score per query across classes
            scores_per_query, labels_per_query = cls_scores[b].max(dim=-1)

            # Filter by score threshold
            keep = scores_per_query > self.score_threshold
            scores = scores_per_query[keep]
            labels = labels_per_query[keep]
            boxes = bbox_preds[b][keep]

            # Keep top-K
            if scores.numel() > self.max_detections:
                topk_indices = scores.topk(self.max_detections).indices
                scores = scores[topk_indices]
                labels = labels[topk_indices]
                boxes = boxes[topk_indices]

            results_scores.append(scores)
            results_labels.append(labels)
            results_boxes.append(boxes)

        # Pad to same length for batching
        max_dets = max(s.numel() for s in results_scores) if results_scores else 0
        max_dets = max(max_dets, 1)  # At least 1 to avoid empty tensors

        device = cls_scores.device
        padded_scores = torch.zeros(batch_size, max_dets, device=device)
        padded_labels = torch.zeros(batch_size, max_dets, dtype=torch.long, device=device)
        padded_boxes = torch.zeros(batch_size, max_dets, self.head.code_size, device=device)

        for b in range(batch_size):
            n = results_scores[b].numel()
            if n > 0:
                padded_scores[b, :n] = results_scores[b]
                padded_labels[b, :n] = results_labels[b]
                padded_boxes[b, :n] = results_boxes[b]

        detections = {
            "scores": padded_scores,
            "labels": padded_labels,
            "boxes": padded_boxes,
            "num_detections": torch.tensor(
                [s.numel() for s in results_scores], device=device
            ),
        }

        return detections, new_bev


if __name__ == "__main__":
    # Smoke test: instantiate model and run a dummy forward pass
    import sys

    print("Instantiating BEVFormer model...")
    model = BEVFormer(
        backbone_pretrained=False,  # Skip downloading weights for test
        embed_dim=64,  # Small for testing
        bev_h=10,
        bev_w=10,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_queries=50,
        num_heads=4,
        ffn_dim=128,
        num_levels=4,
        num_cams=6,
        num_classes=10,
        code_size=10,
    )
    model.eval()

    batch_size = 1
    num_cams = 6
    H, W = 224, 400

    images = torch.randn(batch_size, num_cams, 3, H, W)
    intrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, num_cams, -1, -1).clone()
    intrinsics[:, :, 0, 0] = 800.0  # fx
    intrinsics[:, :, 1, 1] = 800.0  # fy
    intrinsics[:, :, 0, 2] = W / 2  # cx
    intrinsics[:, :, 1, 2] = H / 2  # cy
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, num_cams, -1, -1).clone()
    ego_motion = torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1).clone()

    print(f"Input shapes: images={images.shape}, intrinsics={intrinsics.shape}")
    print("Running forward_test (first frame, no prev_bev)...")

    with torch.no_grad():
        detections, bev_features = model.forward_test(
            images, intrinsics, extrinsics, ego_motion, prev_bev=None
        )

    print(f"BEV features shape: {bev_features.shape}")
    print(f"Detections - scores: {detections['scores'].shape}, "
          f"boxes: {detections['boxes'].shape}")
    print(f"Number of detections: {detections['num_detections']}")

    print("\nRunning forward_test (second frame, with prev_bev)...")
    with torch.no_grad():
        detections2, bev_features2 = model.forward_test(
            images, intrinsics, extrinsics, ego_motion, prev_bev=bev_features.detach()
        )
    print(f"Second frame detections: {detections2['num_detections']}")

    # Test training forward
    print("\nRunning forward_train...")
    model.train()
    gt_bboxes = torch.randn(batch_size, 5, 10)  # 5 GT boxes
    gt_labels = torch.randint(0, 10, (batch_size, 5))

    loss_dict, new_bev = model.forward_train(
        images, intrinsics, extrinsics, ego_motion,
        prev_bev=None, gt_bboxes_3d=gt_bboxes, gt_labels=gt_labels
    )
    print(f"Losses: { {k: v.item() for k, v in loss_dict.items()} }")
    print("\nAll tests passed!")
    sys.exit(0)
