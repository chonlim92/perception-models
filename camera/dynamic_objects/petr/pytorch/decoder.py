"""
Transformer decoder for PETR.

Standard (non-deformable) transformer decoder with multi-head self-attention
among object queries, multi-head cross-attention from queries to position-aware
image features (global attention), FFN blocks, and support for iterative
bounding box refinement.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional key/value projections.

    Args:
        embed_dims: Total dimension of the model.
        num_heads: Number of attention heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        assert (
            self.head_dim * num_heads == embed_dims
        ), "embed_dims must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            query: Query tensor (B, Q, C).
            key: Key tensor (B, K, C).
            value: Value tensor (B, K, C).
            key_padding_mask: Bool mask (B, K) where True means ignore.
            attn_mask: Additive attention mask (Q, K) or (B*H, Q, K).

        Returns:
            Attention output (B, Q, C).
        """
        B, Q, _ = query.shape
        _, K, _ = key.shape
        H = self.num_heads

        # Project queries, keys, values
        q = self.q_proj(query).reshape(B, Q, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, K, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, K, H, self.head_dim).transpose(1, 2)
        # q, k, v: (B, H, Q/K, head_dim)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, Q, K)

        # Apply attention mask (additive)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                attn = attn + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, K) -> (B, 1, 1, K)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, Q, head_dim)
        out = out.transpose(1, 2).reshape(B, Q, self.embed_dims)
        out = self.out_proj(out)

        return out


class FFN(nn.Module):
    """Feed-Forward Network with two linear layers and activation.

    Args:
        embed_dims: Input and output dimension.
        feedforward_dims: Hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_dims: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dims, feedforward_dims)
        self.fc2 = nn.Linear(feedforward_dims, embed_dims)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Linear -> ReLU -> Dropout -> Linear -> Dropout.

        Args:
            x: Input tensor (B, Q, C).

        Returns:
            Output tensor (B, Q, C).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Single layer of the PETR transformer decoder.

    Consists of:
    1. Multi-head self-attention among object queries
    2. Multi-head cross-attention: queries attend to ALL position-aware
       image features (global attention, not deformable/local)
    3. Feed-forward network
    Each with residual connection and layer normalization (pre-norm style).

    Args:
        embed_dims: Feature dimension.
        num_heads: Number of attention heads.
        feedforward_dims: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_dims: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (global attention to all image features)
        self.cross_attn = MultiHeadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_dims=feedforward_dims,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through one decoder layer.

        Args:
            query: Object queries (B, Q, C).
            key: Image features / memory (B, K, C) where K = N_cams * D * H * W.
            value: Same as key (for standard attention).
            query_pos: Position embedding for queries (B, Q, C).
            key_pos: Position embedding for keys (B, K, C). If None, assumed
                already added to key.
            self_attn_mask: Mask for self-attention (Q, Q).
            cross_attn_mask: Mask for cross-attention (Q, K).
            key_padding_mask: Padding mask for keys (B, K).

        Returns:
            Updated query features (B, Q, C).
        """
        # --- Self-attention ---
        residual = query
        query_with_pos = query + query_pos
        query = self.self_attn(
            query=query_with_pos,
            key=query_with_pos,
            value=query,
            attn_mask=self_attn_mask,
        )
        query = residual + self.dropout1(query)
        query = self.norm1(query)

        # --- Cross-attention (global attention to all image features) ---
        residual = query
        query_with_pos = query + query_pos
        if key_pos is not None:
            key_with_pos = key + key_pos
        else:
            key_with_pos = key
        query = self.cross_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=cross_attn_mask,
        )
        query = residual + self.dropout2(query)
        query = self.norm2(query)

        # --- FFN ---
        residual = query
        query = self.ffn(query)
        query = residual + self.dropout3(query)
        query = self.norm3(query)

        return query


class PETRTransformerDecoder(nn.Module):
    """PETR Transformer Decoder with iterative bounding box refinement.

    Stack of TransformerDecoderLayers. Each layer's output produces
    intermediate predictions that are refined by subsequent layers.

    Args:
        num_layers: Number of decoder layers (default 6).
        embed_dims: Feature dimension.
        num_heads: Number of attention heads.
        feedforward_dims: FFN hidden dimension.
        dropout: Dropout probability.
        return_intermediate: Whether to return intermediate layer outputs
            (needed for auxiliary losses and iterative refinement).
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_dims: int = 2048,
        dropout: float = 0.1,
        return_intermediate: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.return_intermediate = return_intermediate

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_dims=feedforward_dims,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dims)

        # Post-initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        reg_branches: Optional[nn.ModuleList] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through all decoder layers with iterative refinement.

        Args:
            query: Initial object queries (B, Q, C).
            key: Position-aware image features (B, K, C).
            value: Image features (B, K, C).
            query_pos: Position embedding for queries (B, Q, C).
            key_pos: Position embedding for keys (B, K, C).
            reference_points: Initial 3D reference points (B, Q, 3), normalized.
            reg_branches: Optional list of regression heads (one per layer)
                for iterative refinement. Each takes (B, Q, C) -> (B, Q, code_size).
            self_attn_mask: Mask for self-attention.
            key_padding_mask: Padding mask for cross-attention keys.

        Returns:
            Tuple of:
                - Final query output (B, Q, C).
                - List of intermediate query outputs (one per layer if
                  return_intermediate, else just the last).
                - List of reference points after each layer's refinement.
        """
        intermediate_outputs = []
        intermediate_ref_pts = []

        output = query

        for layer_idx, layer in enumerate(self.layers):
            # Update query position embedding based on current reference points
            # (The position embedding can be recomputed if reference points change)
            output = layer(
                query=output,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                self_attn_mask=self_attn_mask,
                key_padding_mask=key_padding_mask,
            )

            # Iterative refinement of reference points
            if reg_branches is not None and reference_points is not None:
                # Get regression prediction for this layer
                reg_output = reg_branches[layer_idx](output)  # (B, Q, code_size)
                # Update reference points with predicted offsets
                # Only use the first 3 values (cx, cy, cz offsets)
                new_reference_points = reference_points + reg_output[..., :3]
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate_outputs.append(self.final_norm(output))
                if reference_points is not None:
                    intermediate_ref_pts.append(reference_points)

        if not self.return_intermediate:
            output = self.final_norm(output)
            intermediate_outputs.append(output)
            if reference_points is not None:
                intermediate_ref_pts.append(reference_points)

        # Stack intermediate outputs: (num_layers, B, Q, C)
        intermediate_outputs_stacked = torch.stack(intermediate_outputs, dim=0)

        return output, intermediate_outputs, intermediate_ref_pts


# =============================================================================
# Hierarchical Lane Positional Embeddings for PETR
# =============================================================================


class HierarchicalLanePositionalEmbedding(nn.Module):
    """Hierarchical positional embeddings encoding lane -> line -> point structure.

    Each query's positional embedding is the sum of lane-level, line-type,
    and point-position learned embeddings. Designed to replace flat query
    embeddings when using PETR for lane detection tasks.

    Query layout:
        [0, num_lane_queries): 25 lanes × 2 lines × 20 points = 1000
        [num_lane_queries, total): num_other_lines × 20 points
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_lanes: int = 25,
        points_per_line: int = 20,
        num_other_lines: int = 0,
        pos_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_lanes = num_lanes
        self.points_per_line = points_per_line
        self.num_other_lines = num_other_lines

        self.num_lane_queries = num_lanes * 2 * points_per_line
        self.num_other_queries = num_other_lines * points_per_line
        self.total_queries = self.num_lane_queries + self.num_other_queries
        self.num_total_lines = num_lanes * 2 + num_other_lines

        self.lane_embedding = nn.Embedding(num_lanes + num_other_lines, embed_dim)
        self.line_type_embedding = nn.Embedding(3, embed_dim)
        self._build_sinusoidal_base(points_per_line, embed_dim)
        self.point_residual = nn.Embedding(points_per_line, embed_dim)
        self.content_embedding = nn.Embedding(self.total_queries, embed_dim)

        self.pos_layer_norm = nn.LayerNorm(embed_dim)
        self.pos_dropout = nn.Dropout(pos_drop)

        self._cached_pos: Optional[torch.Tensor] = None
        self._init_weights()
        self._build_index_tables()

    def _build_sinusoidal_base(self, num_points: int, embed_dim: int) -> None:
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
        return self.point_sinusoidal[point_ids] + self.point_residual(point_ids)

    def _init_weights(self) -> None:
        nn.init.normal_(self.lane_embedding.weight, std=0.02)
        nn.init.normal_(self.line_type_embedding.weight, std=0.02)
        nn.init.normal_(self.point_residual.weight, std=0.01)
        nn.init.normal_(self.content_embedding.weight, std=0.02)

    def _build_index_tables(self) -> None:
        lane_ids, line_type_ids, point_ids = [], [], []
        for lane_idx in range(self.num_lanes):
            for line_type in range(2):
                for pt_idx in range(self.points_per_line):
                    lane_ids.append(lane_idx)
                    line_type_ids.append(line_type)
                    point_ids.append(pt_idx)
        for line_idx in range(self.num_other_lines):
            for pt_idx in range(self.points_per_line):
                lane_ids.append(self.num_lanes + line_idx)
                line_type_ids.append(2)
                point_ids.append(pt_idx)

        self.register_buffer("lane_ids", torch.tensor(lane_ids, dtype=torch.long))
        self.register_buffer("line_type_ids", torch.tensor(line_type_ids, dtype=torch.long))
        self.register_buffer("point_ids", torch.tensor(point_ids, dtype=torch.long))

        lane_mask = torch.zeros(self.total_queries, dtype=torch.bool)
        lane_mask[: self.num_lane_queries] = True
        self.register_buffer("lane_mask", lane_mask)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pos_embed, content_embed) each (total_queries, embed_dim)."""
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

        return pos_embed, self.content_embedding.weight

    def get_lane_mask(self) -> torch.Tensor:
        """Return boolean mask identifying lane queries vs other-line queries."""
        return self.lane_mask


class PETRLaneDecoder(nn.Module):
    """PETR decoder adapted for lane detection with hierarchical positional embeddings.

    Uses PETR's global cross-attention to 3D position-aware image features,
    but with lane-structured queries: 25 lanes × 2 lines × 20 points.
    Each query has a 3D reference point that is iteratively refined.

    PETR's key advantage for lane detection: position-aware features encode
    3D geometry directly, so lane queries can attend to relevant 3D locations
    without explicit projection (unlike DETR3D's feature sampling approach).
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_dims: int = 2048,
        dropout: float = 0.1,
        num_lanes: int = 25,
        points_per_line: int = 20,
        num_other_lines: int = 0,
        return_intermediate: bool = True,
    ) -> None:
        """Initialize PETR lane detection decoder.

        Args:
            num_layers: Number of decoder layers.
            embed_dims: Embedding dimension.
            num_heads: Number of attention heads.
            feedforward_dims: FFN hidden dimension.
            dropout: Dropout rate.
            num_lanes: Number of lanes (each with left+right boundary).
            points_per_line: Points per line (default 20).
            num_other_lines: Additional non-lane polylines.
            return_intermediate: Return all layer outputs for auxiliary loss.
        """
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_lanes = num_lanes
        self.points_per_line = points_per_line
        self.num_other_lines = num_other_lines
        self.num_total_lines = num_lanes * 2 + num_other_lines
        self.return_intermediate = return_intermediate

        # Hierarchical positional embeddings
        self.hierarchical_pos = HierarchicalLanePositionalEmbedding(
            embed_dim=embed_dims,
            num_lanes=num_lanes,
            points_per_line=points_per_line,
            num_other_lines=num_other_lines,
        )
        self.total_queries = self.hierarchical_pos.total_queries

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_dims=feedforward_dims,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Dynamic position injection: project reference points to query_pos space
        self.query_pos_proj = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        # 3D reference points (lanes are ground-plane structures)
        self.reference_points_proj = nn.Linear(embed_dims, 3)

        # Per-layer refinement heads
        self.reg_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 3),
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dims)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.reference_points_proj.weight)
        nn.init.zeros_(self.reference_points_proj.bias)
        for branch in self.reg_branches:
            for m in branch.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        for m in self.query_pos_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_decoupled_self_attn_mask(self) -> torch.Tensor:
        """Build block-diagonal self-attention mask for lane-structured queries."""
        total_q = self.total_queries
        mask = torch.full((total_q, total_q), float("-inf"))
        for line_idx in range(self.num_total_lines):
            start = line_idx * self.points_per_line
            end = start + self.points_per_line
            mask[start:end, start:end] = 0.0
        return mask

    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass for lane detection.

        Args:
            key: Position-aware image features (B, K, C).
            value: Image features (B, K, C).
            key_pos: Position embedding for keys (B, K, C), or None if
                already encoded in key.
            key_padding_mask: Padding mask for keys (B, K).

        Returns:
            query_output: Final queries (B, total_queries, embed_dims).
            intermediate_outputs: Per-layer normalized outputs.
            intermediate_ref_pts: Per-layer reference points (normalized).
        """
        batch_size = key.shape[0]

        # Get hierarchical embeddings
        hier_pos, hier_content = self.hierarchical_pos()

        # Initialize queries and positional embeddings
        query = hier_content.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos_static = hier_pos.unsqueeze(0).expand(batch_size, -1, -1)

        # Initialize 3D reference points
        combined = hier_content + hier_pos
        reference_points = self.reference_points_proj(combined).sigmoid()
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1)

        # Decoupled self-attention mask
        self_attn_mask = self._build_decoupled_self_attn_mask().to(
            device=query.device
        )

        intermediate_outputs = []
        intermediate_ref_pts = []

        for layer_idx, layer in enumerate(self.layers):
            # Dynamic position injection: combine static hierarchy with ref-point signal
            query_pos = query_pos_static + self.query_pos_proj(reference_points)

            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                self_attn_mask=self_attn_mask,
                key_padding_mask=key_padding_mask,
            )

            # Iterative refinement in inverse-sigmoid (logit) space, fp32 for safety
            ref_delta = self.reg_branches[layer_idx](query)
            ref_f32 = reference_points.float().clamp(1e-3, 1 - 1e-3)
            inv_ref = torch.log(ref_f32 / (1 - ref_f32))
            new_ref_pts = (inv_ref + ref_delta.float()).sigmoid().to(query.dtype)
            reference_points = new_ref_pts.detach()

            if self.return_intermediate:
                intermediate_outputs.append(self.final_norm(query))
                intermediate_ref_pts.append(new_ref_pts)

        if not self.return_intermediate:
            intermediate_outputs.append(self.final_norm(query))
            intermediate_ref_pts.append(new_ref_pts)

        return query, intermediate_outputs, intermediate_ref_pts
