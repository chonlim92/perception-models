"""MapTR Transformer Decoder with hierarchical queries and iterative refinement.

This module implements the decoder component of MapTR, which uses a hierarchical
query structure (instance queries + point queries) to predict vectorized HD map
elements. Each decoder layer performs self-attention among queries, cross-attention
to BEV features, and feedforward processing, with iterative coordinate refinement
across layers.

Reference: MapTR: Structured Modeling and Learning for Online Vectorized HD Map
Construction (Liao et al., ICLR 2023)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation_fn(activation: str) -> nn.Module:
    """Return an activation function module by name."""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for BEV feature maps.

    Generates separate sin/cos encodings for x and y coordinates, each using
    half of the embedding dimensions, then concatenates them.

    Args:
        embed_dims: Total embedding dimension (split equally between x and y).
        temperature: Temperature scaling factor for the frequency bands.
        normalize: Whether to normalize coordinates to [0, 1].
    """

    def __init__(
        self,
        embed_dims: int = 256,
        temperature: float = 10000.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.temperature = temperature
        self.normalize = normalize
        # Each spatial dimension uses half the embedding dims
        assert embed_dims % 2 == 0, "embed_dims must be even for 2D positional encoding"
        self.half_dims = embed_dims // 2

    def forward(self, bev_shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """Generate 2D positional encoding.

        Args:
            bev_shape: (H, W) of the BEV feature map.
            device: Device to create tensors on.

        Returns:
            Positional encoding tensor of shape [1, embed_dims, H, W].
        """
        h, w = bev_shape
        # Create coordinate grids
        y_coords = torch.arange(h, dtype=torch.float32, device=device)
        x_coords = torch.arange(w, dtype=torch.float32, device=device)

        if self.normalize:
            y_coords = y_coords / (h - 1 + 1e-6)
            x_coords = x_coords / (w - 1 + 1e-6)

        # Build frequency bands: dim_t = temperature^(2i/d) for i in [0, half_dims)
        dim_t = torch.arange(self.half_dims, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.half_dims)

        # Compute positional encodings for y: [H, half_dims]
        pos_y = y_coords.unsqueeze(1) / dim_t.unsqueeze(0)  # [H, half_dims]
        pos_y = torch.stack(
            [pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2
        ).flatten(1)  # [H, half_dims]

        # Compute positional encodings for x: [W, half_dims]
        pos_x = x_coords.unsqueeze(1) / dim_t.unsqueeze(0)  # [W, half_dims]
        pos_x = torch.stack(
            [pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2
        ).flatten(1)  # [W, half_dims]

        # Broadcast to full grid: [H, W, embed_dims]
        pos_y = pos_y.unsqueeze(1).expand(-1, w, -1)  # [H, W, half_dims]
        pos_x = pos_x.unsqueeze(0).expand(h, -1, -1)  # [H, W, half_dims]
        pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, embed_dims]

        # Reshape to [1, embed_dims, H, W]
        pos = pos.permute(2, 0, 1).unsqueeze(0)
        return pos


class MapDecoderLayer(nn.Module):
    """Single transformer decoder layer for MapTR.

    Each layer performs:
    1. Self-attention among all queries (instance x points)
    2. Cross-attention from queries to BEV features
    3. Feedforward network (FFN)

    All sub-layers use pre-layer-norm and residual connections.

    Args:
        embed_dims: Embedding dimension for queries and keys.
        num_heads: Number of attention heads.
        ffn_dims: Hidden dimension of the feedforward network.
        dropout: Dropout probability.
        activation: Activation function name for FFN.
        self_attn_mask_type: Type of self-attention mask.
            - "none": no mask (standard MapTR)
            - "decoupled": block-diagonal mask for MapTRv2 decoupled attention
              where instance queries attend only within the same instance.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_dims: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        self_attn_mask_type: str = "none",
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.self_attn_mask_type = self_attn_mask_type

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout2 = nn.Dropout(dropout)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_self_attn_mask(
        self, num_queries: int, num_points: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Build self-attention mask for decoupled attention (MapTRv2).

        In decoupled mode, point queries within the same instance can attend
        to each other, but not to points from other instances. Instance-level
        information is shared via the iterative refinement mechanism instead.

        Args:
            num_queries: Number of instance queries.
            num_points: Number of points per instance.
            device: Device to create the mask on.

        Returns:
            Attention mask of shape [num_queries*num_points, num_queries*num_points]
            where True means "do NOT attend" (additive -inf mask convention for
            nn.MultiheadAttention), or None if no masking.
        """
        if self.self_attn_mask_type == "none":
            return None

        total = num_queries * num_points
        # Start with all masked (True = blocked)
        mask = torch.ones(total, total, dtype=torch.bool, device=device)
        # Unmask within each instance block
        for i in range(num_queries):
            start = i * num_points
            end = start + num_points
            mask[start:end, start:end] = False

        # Convert bool mask to float mask: True -> -inf, False -> 0
        float_mask = torch.zeros(total, total, dtype=torch.float32, device=device)
        float_mask.masked_fill_(mask, float("-inf"))
        return float_mask

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        num_queries: int,
        num_points: int,
    ) -> torch.Tensor:
        """Forward pass of a single decoder layer.

        Args:
            query: Query features [B, num_queries*num_points, embed_dims].
            query_pos: Positional encoding for queries [B, num_queries*num_points, embed_dims].
            memory: BEV features (key/value for cross-attention) [B, H*W, embed_dims].
            memory_pos: Positional encoding for BEV features [B, H*W, embed_dims].
            num_queries: Number of instance queries.
            num_points: Number of points per instance.

        Returns:
            Updated query features [B, num_queries*num_points, embed_dims].
        """
        # --- Self-attention ---
        residual = query
        query = self.norm1(query)
        q = k = query + query_pos
        attn_mask = self._build_self_attn_mask(num_queries, num_points, query.device)
        query2, _ = self.self_attn(q, k, query, attn_mask=attn_mask)
        query = residual + self.dropout1(query2)

        # --- Cross-attention ---
        residual = query
        query = self.norm2(query)
        q = query + query_pos
        k = memory + memory_pos
        query2, _ = self.cross_attn(q, k, memory)
        query = residual + self.dropout2(query2)

        # --- FFN ---
        residual = query
        query = self.norm3(query)
        query = residual + self.ffn(query)

        return query


class MapDecoder(nn.Module):
    """MapTR transformer decoder with hierarchical queries and iterative refinement.

    The decoder uses a two-level query structure:
    - Instance queries: represent individual map elements (lane lines, boundaries, etc.)
    - Point queries: represent ordered vertices within each map element

    The combined query for each point is: instance_query[i] + point_query[j],
    where i indexes the instance and j indexes the point within that instance.

    Iterative refinement: each decoder layer predicts coordinate offsets that
    update the reference points, which in turn update the positional encodings
    for subsequent layers.

    Args:
        embed_dims: Embedding dimension.
        num_heads: Number of attention heads per layer.
        ffn_dims: FFN hidden dimension.
        num_layers: Number of decoder layers.
        num_queries: Number of instance queries (map elements to detect).
        num_points: Number of points per map element.
        dropout: Dropout probability.
        activation: Activation function name.
        self_attn_mask_type: Self-attention mask type ("none" or "decoupled").
        return_intermediate: Whether to return intermediate layer outputs.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_dims: int = 1024,
        num_layers: int = 6,
        num_queries: int = 50,
        num_points: int = 20,
        dropout: float = 0.1,
        activation: str = "relu",
        self_attn_mask_type: str = "none",
        return_intermediate: bool = True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_points = num_points
        self.return_intermediate = return_intermediate

        # Learnable instance queries: each represents a map element
        self.instance_queries = nn.Embedding(num_queries, embed_dims)
        # Learnable point queries: each represents a point position within an instance
        self.point_queries = nn.Embedding(num_points, embed_dims)

        # Learnable reference points for iterative refinement (normalized coords)
        # Initial reference points for each instance-point combination
        self.reference_points_embed = nn.Linear(embed_dims, 2)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                MapDecoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    ffn_dims=ffn_dims,
                    dropout=dropout,
                    activation=activation,
                    self_attn_mask_type=self_attn_mask_type,
                )
                for _ in range(num_layers)
            ]
        )

        # Per-layer refinement MLPs for iterative coordinate updates
        self.refinement_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dims, embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims, 2),
                )
                for _ in range(num_layers)
            ]
        )

        # Query positional encoding projection (from reference points to embed_dims)
        self.query_pos_proj = nn.Sequential(
            nn.Linear(2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        # 2D positional encoding for BEV features
        self.bev_pos_enc = PositionalEncoding2D(embed_dims)

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.instance_queries.weight)
        nn.init.xavier_uniform_(self.point_queries.weight)
        nn.init.xavier_uniform_(self.reference_points_embed.weight)
        nn.init.zeros_(self.reference_points_embed.bias)
        for mlp in self.refinement_mlps:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for layer in self.query_pos_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _build_combined_queries(self, batch_size: int) -> torch.Tensor:
        """Build combined hierarchical queries by broadcasting and adding.

        Combined query[i, j] = instance_query[i] + point_query[j]

        Args:
            batch_size: Batch size.

        Returns:
            Combined queries [B, num_queries*num_points, embed_dims].
        """
        # instance_queries: [num_queries, embed_dims] -> [1, num_queries, 1, embed_dims]
        inst_q = self.instance_queries.weight.unsqueeze(0).unsqueeze(2)
        # point_queries: [num_points, embed_dims] -> [1, 1, num_points, embed_dims]
        pt_q = self.point_queries.weight.unsqueeze(0).unsqueeze(1)

        # Broadcast add: [1, num_queries, num_points, embed_dims]
        combined = inst_q + pt_q
        # Expand to batch and flatten: [B, num_queries*num_points, embed_dims]
        combined = combined.expand(batch_size, -1, -1, -1)
        combined = combined.reshape(batch_size, self.num_queries * self.num_points, self.embed_dims)
        return combined

    def _get_initial_reference_points(self, query: torch.Tensor) -> torch.Tensor:
        """Compute initial reference points from queries.

        Args:
            query: Combined query features [B, num_queries*num_points, embed_dims].

        Returns:
            Reference points [B, num_queries*num_points, 2] in normalized [0, 1] coords.
        """
        ref_pts = self.reference_points_embed(query)
        ref_pts = ref_pts.sigmoid()
        return ref_pts

    def forward(
        self, bev_features: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass of the MapTR decoder.

        Args:
            bev_features: BEV feature map [B, C, H, W] where C == embed_dims.

        Returns:
            Tuple of:
                - intermediate_outputs: List of query features from each layer,
                  each of shape [B, num_queries, num_points, embed_dims].
                - intermediate_ref_pts: List of reference points from each layer,
                  each of shape [B, num_queries, num_points, 2].
        """
        batch_size, c, h, w = bev_features.shape
        assert c == self.embed_dims, (
            f"BEV feature channels ({c}) must match embed_dims ({self.embed_dims})"
        )

        # --- Prepare BEV memory ---
        # Flatten spatial dims: [B, C, H, W] -> [B, H*W, C]
        memory = bev_features.flatten(2).permute(0, 2, 1)

        # BEV positional encoding: [1, C, H, W] -> [1, H*W, C]
        bev_pos = self.bev_pos_enc((h, w), bev_features.device)
        memory_pos = bev_pos.flatten(2).permute(0, 2, 1)
        # Expand to batch size
        memory_pos = memory_pos.expand(batch_size, -1, -1)

        # --- Build hierarchical queries ---
        query = self._build_combined_queries(batch_size)

        # --- Initialize reference points ---
        reference_points = self._get_initial_reference_points(query)

        # --- Iterative decoding ---
        intermediate_outputs = []
        intermediate_ref_pts = []

        for layer_idx, (decoder_layer, refine_mlp) in enumerate(
            zip(self.layers, self.refinement_mlps)
        ):
            # Compute query positional encoding from current reference points
            query_pos = self.query_pos_proj(reference_points)

            # Apply decoder layer
            query = decoder_layer(
                query=query,
                query_pos=query_pos,
                memory=memory,
                memory_pos=memory_pos,
                num_queries=self.num_queries,
                num_points=self.num_points,
            )

            # Iterative refinement: predict coordinate offset and update reference points
            delta = refine_mlp(query)  # [B, num_queries*num_points, 2]
            # Apply offset in inverse-sigmoid space for stable training
            new_ref_pts = (
                torch.special.logit(reference_points.clamp(1e-5, 1 - 1e-5)) + delta
            ).sigmoid()
            reference_points = new_ref_pts.detach()  # Detach for next layer (stop gradient)

            # Store intermediate results
            if self.return_intermediate:
                normed_query = self.final_norm(query)
                # Reshape to [B, num_queries, num_points, embed_dims]
                out = normed_query.reshape(
                    batch_size, self.num_queries, self.num_points, self.embed_dims
                )
                intermediate_outputs.append(out)
                # Reshape reference points similarly
                ref = new_ref_pts.reshape(
                    batch_size, self.num_queries, self.num_points, 2
                )
                intermediate_ref_pts.append(ref)

        # If not returning intermediates, return only the final layer output
        if not self.return_intermediate:
            normed_query = self.final_norm(query)
            out = normed_query.reshape(
                batch_size, self.num_queries, self.num_points, self.embed_dims
            )
            intermediate_outputs.append(out)
            ref = new_ref_pts.reshape(
                batch_size, self.num_queries, self.num_points, 2
            )
            intermediate_ref_pts.append(ref)

        return intermediate_outputs, intermediate_ref_pts


# =============================================================================
# Hierarchical Lane Positional Embeddings for MapTR
# =============================================================================


class HierarchicalLanePositionalEmbedding(nn.Module):
    """Hierarchical positional embeddings encoding lane -> line -> point structure.

    Replaces MapTR's generic instance_queries + point_queries with an explicit
    lane topology: each query's position is the sum of its lane-level,
    line-type (left/right/other), and point-position embeddings.

    Query layout (in order):
        [0, num_lane_queries): Lane queries organized as
            lane_0_left_pt0, ..., lane_0_left_pt19,
            lane_0_right_pt0, ..., lane_0_right_pt19,
            lane_1_left_pt0, ...  (num_lanes × 2 lines × points_per_line)
        [num_lane_queries, total_queries): Other line queries organized as
            line_0_pt0, ..., line_0_pt19,
            line_1_pt0, ...  (num_other_lines × points_per_line)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_lanes: int = 25,
        points_per_line: int = 20,
        num_other_lines: int = 0,
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

        # Lane-level embedding: which lane or other-line group
        self.lane_embedding = nn.Embedding(num_lanes + num_other_lines, embed_dim)
        # Line-type embedding: 0=left boundary, 1=right boundary, 2=other
        self.line_type_embedding = nn.Embedding(3, embed_dim)
        # Point-position embedding: ordinal position along the line
        self.point_embedding = nn.Embedding(points_per_line, embed_dim)
        # Content queries (one per structural slot)
        self.content_embedding = nn.Embedding(self.total_queries, embed_dim)

        self._init_weights()
        self._build_index_tables()

    def _init_weights(self) -> None:
        nn.init.normal_(self.lane_embedding.weight, std=0.02)
        nn.init.normal_(self.line_type_embedding.weight, std=0.02)
        nn.init.normal_(self.point_embedding.weight, std=0.02)
        nn.init.normal_(self.content_embedding.weight, std=0.02)

    def _build_index_tables(self) -> None:
        lane_ids, line_type_ids, point_ids = [], [], []
        for lane_idx in range(self.num_lanes):
            for line_type in range(2):  # 0=left, 1=right
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

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (positional_embedding, content_embedding) each (total_queries, embed_dim)."""
        pos_embed = (
            self.lane_embedding(self.lane_ids)
            + self.line_type_embedding(self.line_type_ids)
            + self.point_embedding(self.point_ids)
        )
        return pos_embed, self.content_embedding.weight

    def get_lane_mask(self) -> torch.Tensor:
        """Return boolean mask identifying lane queries vs other-line queries."""
        mask = torch.zeros(
            self.total_queries, dtype=torch.bool, device=self.lane_ids.device
        )
        mask[: self.num_lane_queries] = True
        return mask


class HierarchicalLaneMapDecoder(nn.Module):
    """MapTR decoder with hierarchical lane-structured positional embeddings.

    Extends MapTR's decoder by replacing generic instance+point queries with
    explicit lane→line→point hierarchy (25 lanes × 2 lines × 20 points).
    Retains MapTR's iterative refinement and decoupled self-attention features.

    Compared to the base MapDecoder:
    - Instance queries are replaced by lane+line_type embeddings
    - Point queries are replaced by point-position embeddings
    - The combined positional embedding encodes the full topology
    - Iterative reference point refinement is preserved
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_dims: int = 1024,
        num_layers: int = 6,
        num_lanes: int = 25,
        points_per_line: int = 20,
        num_other_lines: int = 0,
        dropout: float = 0.1,
        activation: str = "relu",
        self_attn_mask_type: str = "decoupled",
        return_intermediate: bool = True,
    ):
        """Initialize hierarchical lane map decoder.

        Args:
            embed_dims: Embedding dimension.
            num_heads: Number of attention heads per layer.
            ffn_dims: FFN hidden dimension.
            num_layers: Number of decoder layers.
            num_lanes: Number of lanes (each with left+right boundary).
            points_per_line: Points sampled per line (default 20).
            num_other_lines: Non-lane polylines (road edges, crosswalks).
            dropout: Dropout probability.
            activation: Activation function name.
            self_attn_mask_type: "none" or "decoupled" (block-diagonal).
            return_intermediate: Return intermediate layer outputs.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_lanes = num_lanes
        self.points_per_line = points_per_line
        self.num_other_lines = num_other_lines
        self.return_intermediate = return_intermediate

        self.num_total_lines = num_lanes * 2 + num_other_lines

        # Hierarchical positional embeddings
        self.hierarchical_pos = HierarchicalLanePositionalEmbedding(
            embed_dim=embed_dims,
            num_lanes=num_lanes,
            points_per_line=points_per_line,
            num_other_lines=num_other_lines,
        )
        self.total_queries = self.hierarchical_pos.total_queries

        # Reference points initialization from combined query embeddings
        self.reference_points_embed = nn.Linear(embed_dims, 2)

        # Decoder layers
        self.layers = nn.ModuleList([
            MapDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                ffn_dims=ffn_dims,
                dropout=dropout,
                activation=activation,
                self_attn_mask_type=self_attn_mask_type,
            )
            for _ in range(num_layers)
        ])

        # Per-layer refinement MLPs
        self.refinement_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 2),
            )
            for _ in range(num_layers)
        ])

        # Query positional encoding projection (from 2D ref points to embed_dims)
        self.query_pos_proj = nn.Sequential(
            nn.Linear(2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        # BEV positional encoding
        self.bev_pos_enc = PositionalEncoding2D(embed_dims)

        # Final norm
        self.final_norm = nn.LayerNorm(embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.reference_points_embed.weight)
        nn.init.zeros_(self.reference_points_embed.bias)
        for mlp in self.refinement_mlps:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for layer in self.query_pos_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self, bev_features: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass of the hierarchical lane decoder.

        Args:
            bev_features: BEV feature map [B, C, H, W].

        Returns:
            Tuple of:
                - intermediate_outputs: List of query features per layer,
                  each [B, num_total_lines, points_per_line, embed_dims].
                - intermediate_ref_pts: List of reference points per layer,
                  each [B, num_total_lines, points_per_line, 2].
        """
        batch_size, c, h, w = bev_features.shape

        # Prepare BEV memory
        memory = bev_features.flatten(2).permute(0, 2, 1)
        bev_pos = self.bev_pos_enc((h, w), bev_features.device)
        memory_pos = bev_pos.flatten(2).permute(0, 2, 1).expand(batch_size, -1, -1)

        # Build hierarchical queries
        hier_pos, hier_content = self.hierarchical_pos()
        query = hier_content.unsqueeze(0).expand(batch_size, -1, -1)

        # Initialize reference points
        reference_points = self.reference_points_embed(
            hier_content + hier_pos
        ).sigmoid()
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1)

        # Iterative decoding
        intermediate_outputs = []
        intermediate_ref_pts = []

        for layer_idx, (decoder_layer, refine_mlp) in enumerate(
            zip(self.layers, self.refinement_mlps)
        ):
            # Positional encoding from reference points + hierarchical structure
            query_pos = self.query_pos_proj(reference_points) + hier_pos.unsqueeze(0)

            # Apply decoder layer
            query = decoder_layer(
                query=query,
                query_pos=query_pos,
                memory=memory,
                memory_pos=memory_pos,
                num_queries=self.num_total_lines,
                num_points=self.points_per_line,
            )

            # Iterative refinement
            delta = refine_mlp(query)
            new_ref_pts = (
                torch.special.logit(reference_points.clamp(1e-5, 1 - 1e-5)) + delta
            ).sigmoid()
            reference_points = new_ref_pts.detach()

            # Store intermediate results
            if self.return_intermediate:
                normed_query = self.final_norm(query)
                out = normed_query.reshape(
                    batch_size, self.num_total_lines, self.points_per_line, self.embed_dims
                )
                intermediate_outputs.append(out)
                ref = new_ref_pts.reshape(
                    batch_size, self.num_total_lines, self.points_per_line, 2
                )
                intermediate_ref_pts.append(ref)

        if not self.return_intermediate:
            normed_query = self.final_norm(query)
            out = normed_query.reshape(
                batch_size, self.num_total_lines, self.points_per_line, self.embed_dims
            )
            intermediate_outputs.append(out)
            ref = new_ref_pts.reshape(
                batch_size, self.num_total_lines, self.points_per_line, 2
            )
            intermediate_ref_pts.append(ref)

        return intermediate_outputs, intermediate_ref_pts
