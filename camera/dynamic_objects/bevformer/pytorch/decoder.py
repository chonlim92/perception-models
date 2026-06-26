"""BEVFormer DETR-style Transformer Decoder.

This module implements a transformer decoder that takes encoded BEV features
and refines learnable object queries through iterative self-attention among
queries and cross-attention to BEV features. It predicts 3D reference points
from object queries for deformable cross-attention, following the Deformable
DETR paradigm adapted to BEV space.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BEVFormerDecoder"]


class DecoderFFN(nn.Module):
    """Feed-Forward Network for decoder layers.

    Two-layer MLP with GELU activation and dropout.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_dims: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        """Initialize decoder FFN.

        Args:
            embed_dims: Input and output dimension.
            feedforward_dims: Hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.linear1 = nn.Linear(embed_dims, feedforward_dims)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dims, embed_dims)
        self.dropout2 = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with Xavier uniform for weights, zeros for biases."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., embed_dims).

        Returns:
            Output tensor of same shape.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class DecoderSelfAttention(nn.Module):
    """Multi-head self-attention among object queries.

    Standard scaled dot-product attention with learnable projections.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize self-attention module.

        Args:
            embed_dims: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Attention dropout probability.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads

        assert embed_dims % num_heads == 0, (
            f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})"
        )

        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with Xavier uniform for weights, zeros for biases."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of self-attention.

        Args:
            query: Object query features, shape (B, num_queries, embed_dims).
            query_pos: Positional encoding for queries, shape (B, num_queries, embed_dims).

        Returns:
            Self-attended features, shape (B, num_queries, embed_dims).
        """
        batch_size, num_queries, _ = query.shape

        # Add positional encoding to queries and keys
        q = self.q_proj(query + query_pos)
        k = self.k_proj(query + query_pos)
        v = self.v_proj(query)

        # Reshape for multi-head attention
        # (B, N, embed_dims) -> (B, num_heads, N, head_dim)
        q = q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, N, N)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, head_dim)

        # Reshape back: (B, H, N, head_dim) -> (B, N, embed_dims)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, num_queries, self.embed_dims
        )

        # Output projection
        output = self.out_proj(attn_output)
        return output


class DecoderCrossAttention(nn.Module):
    """Cross-attention from object queries to BEV features.

    Uses deformable-style attention where object queries attend to BEV features
    around predicted 3D reference points projected onto the 2D BEV grid.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize cross-attention module.

        Args:
            embed_dims: Embedding dimension.
            num_heads: Number of attention heads.
            num_points: Number of deformable sampling points per head.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dims // num_heads

        assert embed_dims % num_heads == 0

        # Sampling offsets: predict 2D offsets on BEV grid per head per point
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_points * 2)

        # Attention weights over sampling points
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_points)

        # Value and output projections
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following deformable attention conventions."""
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)

        # Initialize offsets in a circular pattern around reference points
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
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
        query_pos: torch.Tensor,
        bev_features: torch.Tensor,
        bev_pos: torch.Tensor,
        reference_points_2d: torch.Tensor,
        bev_h: int,
        bev_w: int,
    ) -> torch.Tensor:
        """Forward pass of deformable cross-attention to BEV features.

        Args:
            query: Object query features, shape (B, num_queries, embed_dims).
            query_pos: Query positional encoding, shape (B, num_queries, embed_dims).
            bev_features: Encoded BEV features, shape (B, bev_h*bev_w, embed_dims).
            bev_pos: BEV positional encoding, shape (B, bev_h*bev_w, embed_dims).
            reference_points_2d: Normalized 2D reference points on BEV grid,
                shape (B, num_queries, 2) in [0, 1].
            bev_h: BEV grid height.
            bev_w: BEV grid width.

        Returns:
            Cross-attended features, shape (B, num_queries, embed_dims).
        """
        batch_size, num_queries, _ = query.shape
        num_bev = bev_features.shape[1]

        # Project BEV values (add positional encoding to keys implicitly via value context)
        value = self.value_proj(bev_features + bev_pos)
        value = value.view(batch_size, num_bev, self.num_heads, self.head_dim)
        # Reshape to (B*num_heads, head_dim, bev_h, bev_w)
        value = (
            value.permute(0, 2, 3, 1)
            .reshape(batch_size * self.num_heads, self.head_dim, bev_h, bev_w)
        )

        # Compute sampling offsets from queries + position
        query_with_pos = query + query_pos
        offsets = self.sampling_offsets(query_with_pos)
        offsets = offsets.view(
            batch_size, num_queries, self.num_heads, self.num_points, 2
        )

        # Compute attention weights
        attn_weights = self.attention_weights(query_with_pos)
        attn_weights = attn_weights.view(
            batch_size, num_queries, self.num_heads, self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute sampling locations around reference points
        # reference_points_2d: (B, num_queries, 2) -> (B, num_queries, 1, 1, 2)
        ref_pts = reference_points_2d[:, :, None, None, :]

        # Normalize offsets by BEV dimensions
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

        # Reshape: (B, num_queries, embed_dims)
        output = output.permute(0, 3, 1, 2).reshape(
            batch_size, num_queries, self.embed_dims
        )

        # Output projection
        output = self.output_proj(output)
        return output


class BEVFormerDecoderLayer(nn.Module):
    """Single BEVFormer decoder layer.

    Consists of three sub-layers with pre-norm residual connections:
      1. Self-attention among object queries
      2. Cross-attention from queries to BEV features (deformable)
      3. Feed-forward network
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        feedforward_dims: int = 1024,
        dropout: float = 0.1,
        num_points: int = 4,
    ) -> None:
        """Initialize decoder layer.

        Args:
            embed_dims: Feature embedding dimension.
            num_heads: Number of attention heads.
            feedforward_dims: FFN hidden dimension.
            dropout: Dropout rate.
            num_points: Number of deformable sampling points for cross-attention.
        """
        super().__init__()

        # Self-attention among queries
        self.self_attn = DecoderSelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention to BEV features
        self.cross_attn = DecoderCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points,
            dropout=dropout,
        )

        # FFN
        self.ffn = DecoderFFN(
            embed_dims=embed_dims,
            feedforward_dims=feedforward_dims,
            dropout=dropout,
        )

        # Layer norms (pre-norm)
        self.norm_self_attn = nn.LayerNorm(embed_dims)
        self.norm_cross_attn = nn.LayerNorm(embed_dims)
        self.norm_ffn = nn.LayerNorm(embed_dims)

        # Dropout for residual connections
        self.dropout_self_attn = nn.Dropout(dropout)
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        bev_features: torch.Tensor,
        bev_pos: torch.Tensor,
        reference_points_2d: torch.Tensor,
        bev_h: int,
        bev_w: int,
    ) -> torch.Tensor:
        """Forward pass of a single decoder layer.

        Args:
            query: Object query features, shape (B, num_queries, embed_dims).
            query_pos: Query positional encoding, shape (B, num_queries, embed_dims).
            bev_features: Encoded BEV features, shape (B, bev_h*bev_w, embed_dims).
            bev_pos: BEV positional encoding, shape (B, bev_h*bev_w, embed_dims).
            reference_points_2d: 2D reference points on BEV grid,
                shape (B, num_queries, 2) in [0, 1].
            bev_h: BEV grid height.
            bev_w: BEV grid width.

        Returns:
            Updated query features, shape (B, num_queries, embed_dims).
        """
        # 1. Self-attention with pre-norm and residual
        residual = query
        query = self.norm_self_attn(query)
        query = residual + self.dropout_self_attn(
            self.self_attn(query, query_pos)
        )

        # 2. Cross-attention with pre-norm and residual
        residual = query
        query = self.norm_cross_attn(query)
        query = residual + self.dropout_cross_attn(
            self.cross_attn(
                query=query,
                query_pos=query_pos,
                bev_features=bev_features,
                bev_pos=bev_pos,
                reference_points_2d=reference_points_2d,
                bev_h=bev_h,
                bev_w=bev_w,
            )
        )

        # 3. FFN with pre-norm and residual
        residual = query
        query = self.norm_ffn(query)
        query = residual + self.dropout_ffn(self.ffn(query))

        return query


class BEVFormerDecoder(nn.Module):
    """BEVFormer DETR-style Transformer Decoder.

    Decodes object queries by iteratively refining them through self-attention
    and cross-attention to BEV features. Predicts 3D reference points from
    queries for spatially-aware deformable cross-attention. Outputs features
    from all decoder layers for auxiliary loss computation.
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_queries: int = 900,
        feedforward_dims: int = 1024,
        dropout: float = 0.1,
        num_points: int = 4,
        bev_h: int = 200,
        bev_w: int = 200,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        """Initialize BEVFormer decoder.

        Args:
            num_layers: Number of stacked decoder layers.
            embed_dims: Feature embedding dimension.
            num_heads: Number of attention heads.
            num_queries: Number of learnable object queries.
            feedforward_dims: FFN hidden dimension.
            dropout: Dropout rate.
            num_points: Deformable sampling points for cross-attention.
            bev_h: BEV grid height.
            bev_w: BEV grid width.
            pc_range: Point cloud range (x_min, y_min, z_min, x_max, y_max, z_max).
        """
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range

        # Learnable object queries
        self.query_embedding = nn.Embedding(num_queries, embed_dims)

        # Learnable query positional encoding
        self.query_pos_embedding = nn.Embedding(num_queries, embed_dims)

        # Reference point prediction: predicts 3D reference points from queries
        # The 3D reference point (x, y, z) is in normalized [0, 1] coordinates
        # within pc_range, then projected to 2D BEV for cross-attention
        self.reference_point_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 3),  # 3D reference point (x, y, z)
        )

        # BEV positional encoding (sinusoidal for keys)
        self.bev_pos_embed = nn.Parameter(
            self._generate_bev_pos_embed(embed_dims, bev_h, bev_w)
        )

        # Stacked decoder layers
        self.layers = nn.ModuleList([
            BEVFormerDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_dims=feedforward_dims,
                dropout=dropout,
                num_points=num_points,
            )
            for _ in range(num_layers)
        ])

        # Final normalization for each layer output (for auxiliary losses)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dims) for _ in range(num_layers)
        ])

        self._init_weights()

    @staticmethod
    def _generate_bev_pos_embed(
        embed_dims: int, bev_h: int, bev_w: int
    ) -> torch.Tensor:
        """Generate fixed sinusoidal positional encoding for BEV keys.

        Args:
            embed_dims: Embedding dimension.
            bev_h: BEV grid height.
            bev_w: BEV grid width.

        Returns:
            Positional encoding of shape (1, bev_h*bev_w, embed_dims).
        """
        half_dim = embed_dims // 2

        # Temperature for sinusoidal encoding
        temperature = 10000.0
        dim_t = torch.arange(half_dim // 2, dtype=torch.float32)
        dim_t = temperature ** (2.0 * (dim_t // 2) / (half_dim // 2))

        # Y positions
        y_pos = torch.arange(bev_h, dtype=torch.float32).unsqueeze(1) / bev_h
        y_embed = y_pos / dim_t.unsqueeze(0)
        y_sin = y_embed[:, 0::2].sin()
        y_cos = y_embed[:, 1::2].cos()
        y_pe = torch.zeros(bev_h, half_dim)
        y_pe[:, 0::2] = y_sin[:, :y_pe[:, 0::2].shape[1]]
        y_pe[:, 1::2] = y_cos[:, :y_pe[:, 1::2].shape[1]]

        # X positions
        x_pos = torch.arange(bev_w, dtype=torch.float32).unsqueeze(1) / bev_w
        x_embed = x_pos / dim_t.unsqueeze(0)
        x_sin = x_embed[:, 0::2].sin()
        x_cos = x_embed[:, 1::2].cos()
        x_pe = torch.zeros(bev_w, half_dim)
        x_pe[:, 0::2] = x_sin[:, :x_pe[:, 0::2].shape[1]]
        x_pe[:, 1::2] = x_cos[:, :x_pe[:, 1::2].shape[1]]

        # Combine: (H, half) x (W, half) -> (H, W, embed_dims)
        y_pe_2d = y_pe.unsqueeze(1).expand(-1, bev_w, -1)
        x_pe_2d = x_pe.unsqueeze(0).expand(bev_h, -1, -1)
        pe = torch.cat([y_pe_2d, x_pe_2d], dim=-1)

        return pe.reshape(1, bev_h * bev_w, embed_dims)

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform and zeros for biases."""
        # Query embeddings
        nn.init.xavier_uniform_(
            self.query_embedding.weight.unsqueeze(0)
        ).squeeze(0)
        nn.init.xavier_uniform_(
            self.query_pos_embedding.weight.unsqueeze(0)
        ).squeeze(0)

        # Reference point head
        for module in self.reference_point_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize reference point output bias to center of BEV
        # This helps with convergence by starting reference points near the center
        with torch.no_grad():
            self.reference_point_head[-1].bias.fill_(0.0)

    def forward(
        self,
        bev_features: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Forward pass through the BEVFormer decoder.

        Args:
            bev_features: Encoded BEV features from the encoder,
                shape (B, bev_h*bev_w, embed_dims).

        Returns:
            List of decoded features from each decoder layer, each with shape
            (B, num_queries, embed_dims). Used for auxiliary losses during training.
            The last element corresponds to the final decoder layer output.
        """
        batch_size = bev_features.shape[0]
        device = bev_features.device

        # Initialize object queries: (num_queries, embed_dims) -> (B, num_queries, embed_dims)
        query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos = self.query_pos_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # BEV positional encoding: (1, bev_h*bev_w, embed_dims) -> (B, bev_h*bev_w, embed_dims)
        bev_pos = self.bev_pos_embed.expand(batch_size, -1, -1)

        # Collect outputs from each layer for auxiliary losses
        intermediate_outputs: List[torch.Tensor] = []

        for layer_idx, (layer, norm) in enumerate(
            zip(self.layers, self.layer_norms)
        ):
            # Predict 3D reference points from current queries
            # reference_points_3d: (B, num_queries, 3) normalized to [0, 1]
            reference_points_3d = torch.sigmoid(
                self.reference_point_head(query + query_pos)
            )

            # Project 3D reference points to 2D BEV coordinates
            # BEV uses only x, y; z is used for height-aware reasoning
            reference_points_2d = reference_points_3d[..., :2]  # (B, num_queries, 2)

            # Apply decoder layer
            query = layer(
                query=query,
                query_pos=query_pos,
                bev_features=bev_features,
                bev_pos=bev_pos,
                reference_points_2d=reference_points_2d,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
            )

            # Normalize and store intermediate output
            intermediate_outputs.append(norm(query))

        return intermediate_outputs
