"""StreamMapNet Transformer Decoder for Map Element Prediction.

This module implements a multi-layer transformer decoder that takes BEV (Bird's
Eye View) features and uses learnable map element queries to predict structured
map elements such as lane dividers, road boundaries, and pedestrian crossings.

Architecture:
    - Learnable queries (50 per class x 3 classes = 150 queries)
    - 6-layer transformer decoder with self-attention, cross-attention, and FFN
    - Deformable cross-attention with reference point prediction
    - 2D sinusoidal positional encoding for BEV features

Reference:
    Yuan et al., "StreamMapNet: Streaming Mapping Network for Vectorized Online
    HD Map Construction", WACV 2024.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for BEV feature maps.

    Generates fixed positional encodings using sine and cosine functions
    at different frequencies for both spatial dimensions.

    Args:
        d_model: Embedding dimension. Must be divisible by 4 (half for x, half
            for y, each split into sin and cos).
        max_h: Maximum height of the feature map.
        max_w: Maximum width of the feature map.
        temperature: Temperature for the frequency scaling.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_h: int = 200,
        max_w: int = 100,
        temperature: float = 10000.0,
    ):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D pos enc"
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        self.temperature = temperature

        # Pre-compute positional encodings
        pe = self._build_encoding(max_h, max_w)
        self.register_buffer("pe", pe)

    def _build_encoding(self, h: int, w: int) -> torch.Tensor:
        """Build 2D sinusoidal positional encoding.

        Args:
            h: Height dimension.
            w: Width dimension.

        Returns:
            Tensor of shape (1, d_model, h, w).
        """
        half_d = self.d_model // 2
        quarter_d = self.d_model // 4

        # Create position grids
        y_pos = torch.arange(h, dtype=torch.float32).unsqueeze(1).expand(h, w)
        x_pos = torch.arange(w, dtype=torch.float32).unsqueeze(0).expand(h, w)

        # Frequency dimensions
        dim = torch.arange(quarter_d, dtype=torch.float32)
        div_term = self.temperature ** (2 * (dim // 2) / half_d)

        # Compute encodings for y
        pe_y_sin = torch.sin(y_pos.unsqueeze(-1) / div_term)  # (h, w, quarter_d)
        pe_y_cos = torch.cos(y_pos.unsqueeze(-1) / div_term)  # (h, w, quarter_d)

        # Compute encodings for x
        pe_x_sin = torch.sin(x_pos.unsqueeze(-1) / div_term)  # (h, w, quarter_d)
        pe_x_cos = torch.cos(x_pos.unsqueeze(-1) / div_term)  # (h, w, quarter_d)

        # Concatenate: [y_sin, y_cos, x_sin, x_cos] -> (h, w, d_model)
        pe = torch.cat([pe_y_sin, pe_y_cos, pe_x_sin, pe_x_cos], dim=-1)

        # Reshape to (1, d_model, h, w)
        pe = pe.permute(2, 0, 1).unsqueeze(0)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input features.

        Args:
            x: Input BEV features of shape (B, C, H, W).

        Returns:
            Positional encoding of shape (1, C, H, W), broadcastable to input.
        """
        _, _, h, w = x.shape
        return self.pe[:, :, :h, :w]


class DeformableCrossAttention(nn.Module):
    """Simplified deformable cross-attention for map queries attending to BEV features.

    Each query predicts sampling offsets around its reference point in the BEV
    space and aggregates features from those sampled locations using learned
    attention weights.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_points: Number of sampling points per attention head.
        dropout: Dropout rate for attention weights.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # Sampling offsets: predict 2D offsets for each head and sampling point
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)

        # Attention weights over sampling points
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)

        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with small offsets."""
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)

        # Initialize offsets to form a grid pattern
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, i, :] *= (i + 1) * 0.5
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Deformable cross-attention forward pass.

        Args:
            query: Query embeddings of shape (B, N_queries, d_model).
            value: BEV feature values of shape (B, H*W, d_model).
            reference_points: Normalized reference points in [0,1] of shape
                (B, N_queries, 2).
            spatial_shape: Tuple of (H, W) for the BEV feature map.

        Returns:
            Output of shape (B, N_queries, d_model).
        """
        B, N_q, _ = query.shape
        _, N_v, _ = value.shape
        H, W = spatial_shape

        # Project values
        value = self.value_proj(value)  # (B, H*W, d_model)
        value = value.view(B, N_v, self.n_heads, self.head_dim)  # (B, H*W, n_heads, head_dim)

        # Predict sampling offsets: (B, N_q, n_heads * n_points * 2)
        offsets = self.sampling_offsets(query)
        offsets = offsets.view(B, N_q, self.n_heads, self.n_points, 2)

        # Predict attention weights: (B, N_q, n_heads * n_points)
        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(B, N_q, self.n_heads, self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, N_q, n_heads, n_points)

        # Compute sampling locations
        # reference_points: (B, N_q, 2) -> (B, N_q, 1, 1, 2)
        ref = reference_points.unsqueeze(2).unsqueeze(3)
        # Normalize offsets to [-1, 1] range relative to spatial dims
        offset_normalizer = torch.tensor(
            [W, H], dtype=torch.float32, device=query.device
        )
        sampling_locations = ref + offsets / offset_normalizer  # (B, N_q, n_heads, n_points, 2)

        # Bilinear sampling from value map
        # Reshape value to (B, n_heads, head_dim, H, W) for grid_sample
        value_map = value.permute(0, 2, 3, 1).view(B, self.n_heads, self.head_dim, H, W)

        # Sample features at predicted locations
        output = torch.zeros(
            B, self.n_heads, self.head_dim, N_q, device=query.device, dtype=query.dtype
        )

        for h_idx in range(self.n_heads):
            # Get sampling grid for this head: (B, N_q, n_points, 2)
            grid = sampling_locations[:, :, h_idx, :, :]  # (B, N_q, n_points, 2)

            # Convert from [0,1] to [-1,1] for grid_sample
            grid = 2.0 * grid - 1.0

            # Reshape for grid_sample: (B, head_dim, H, W) and grid (B, N_q, n_points, 2)
            feat_map = value_map[:, h_idx, :, :, :]  # (B, head_dim, H, W)

            # grid_sample expects grid of shape (B, H_out, W_out, 2)
            # Treat N_q as H_out and n_points as W_out
            sampled = F.grid_sample(
                feat_map,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # (B, head_dim, N_q, n_points)

            # Weight by attention weights for this head
            weights = attn_weights[:, :, h_idx, :]  # (B, N_q, n_points)
            weights = weights.unsqueeze(1)  # (B, 1, N_q, n_points)

            # Weighted sum over sampling points
            output[:, h_idx, :, :] = (sampled * weights).sum(dim=-1)  # (B, head_dim, N_q)

        # Reshape: (B, n_heads, head_dim, N_q) -> (B, N_q, d_model)
        output = output.permute(0, 3, 1, 2).contiguous().view(B, N_q, self.d_model)

        output = self.output_proj(output)
        output = self.dropout(output)

        return output


class MapDecoderLayer(nn.Module):
    """Single transformer decoder layer for map element prediction.

    Each layer consists of:
        1. Self-attention among map queries
        2. Deformable cross-attention from queries to BEV features
        3. Feed-forward network

    All sub-layers include residual connections and layer normalization.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads for self-attention.
        n_deform_heads: Number of heads for deformable cross-attention.
        n_points: Number of sampling points per deformable attention head.
        dim_feedforward: Hidden dimension of FFN.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_deform_heads: int = 8,
        n_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Deformable cross-attention
        self.cross_attn = DeformableCrossAttention(
            d_model=d_model,
            n_heads=n_deform_heads,
            n_points=n_points,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        bev_features: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shape: Tuple[int, int],
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a single decoder layer.

        Args:
            query: Map element query embeddings of shape (B, N_queries, d_model).
            bev_features: Flattened BEV features of shape (B, H*W, d_model).
            reference_points: Normalized reference points of shape (B, N_queries, 2).
            spatial_shape: Tuple of (H, W) for the BEV feature map.
            query_pos: Optional positional embedding for queries of shape
                (B, N_queries, d_model).

        Returns:
            Refined query embeddings of shape (B, N_queries, d_model).
        """
        # Self-attention with positional encoding added to Q and K
        q = k = query + query_pos if query_pos is not None else query
        sa_output, _ = self.self_attn(q, k, query)
        query = query + self.dropout1(sa_output)
        query = self.norm1(query)

        # Deformable cross-attention
        ca_input = query + query_pos if query_pos is not None else query
        ca_output = self.cross_attn(ca_input, bev_features, reference_points, spatial_shape)
        query = query + self.dropout2(ca_output)
        query = self.norm2(query)

        # Feed-forward network
        ffn_output = self.ffn(query)
        query = query + ffn_output
        query = self.norm3(query)

        return query


class MapTransformerDecoder(nn.Module):
    """Multi-layer Transformer Decoder for StreamMapNet.

    Decodes BEV features into structured map element representations using
    learnable queries. Each query represents a potential map element (lane
    divider, road boundary, or pedestrian crossing).

    The decoder uses deformable attention where each query predicts a reference
    point in BEV space and cross-attends to features sampled around that point.

    Args:
        d_model: Model/embedding dimension.
        n_heads: Number of self-attention heads.
        n_deform_heads: Number of deformable cross-attention heads.
        n_points: Number of sampling points per deformable attention head.
        num_layers: Number of decoder layers.
        dim_feedforward: Hidden dimension of the FFN.
        dropout: Dropout rate.
        num_queries_per_class: Number of learnable queries per map class.
        num_classes: Number of map element classes (default 3: lane_divider,
            road_boundary, ped_crossing).
        bev_h: BEV feature map height.
        bev_w: BEV feature map width.
        return_intermediate: Whether to return intermediate layer outputs for
            iterative refinement.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_deform_heads: int = 8,
        n_points: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_queries_per_class: int = 50,
        num_classes: int = 3,
        bev_h: int = 200,
        bev_w: int = 100,
        return_intermediate: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_queries = num_queries_per_class * num_classes
        self.num_classes = num_classes
        self.num_queries_per_class = num_queries_per_class
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.return_intermediate = return_intermediate

        # Learnable map element queries
        self.query_embedding = nn.Embedding(self.num_queries, d_model)
        self.query_pos_embedding = nn.Embedding(self.num_queries, d_model)

        # Reference point prediction: each query predicts its initial reference point
        self.reference_point_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

        # Positional encoding for BEV features
        self.bev_pos_encoding = SinusoidalPositionalEncoding2D(
            d_model=d_model, max_h=bev_h, max_w=bev_w
        )

        # BEV feature projection (in case input dim differs from d_model)
        self.input_proj = nn.Conv2d(d_model, d_model, kernel_size=1)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                MapDecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_deform_heads=n_deform_heads,
                    n_points=n_points,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize learnable parameters."""
        nn.init.normal_(self.query_embedding.weight, std=0.02)
        nn.init.normal_(self.query_pos_embedding.weight, std=0.02)

        for p in self.reference_point_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        bev_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the map transformer decoder.

        Args:
            bev_features: BEV feature map of shape (B, C, H, W) where C=d_model.

        Returns:
            Tuple of:
                - query_embeddings: If return_intermediate=True, shape
                  (num_layers, B, N_queries, d_model); otherwise
                  (B, N_queries, d_model).
                - reference_points: Predicted reference points of shape
                  (B, N_queries, 2) in normalized [0,1] BEV coordinates.
                - query_classes: Class indices for each query of shape
                  (N_queries,), indicating which class each query belongs to.
        """
        B, C, H, W = bev_features.shape

        # Project BEV features
        bev_feat = self.input_proj(bev_features)  # (B, d_model, H, W)

        # Add positional encoding to BEV features
        bev_pos = self.bev_pos_encoding(bev_feat)  # (1, d_model, H, W)
        bev_feat_with_pos = bev_feat + bev_pos  # (B, d_model, H, W)

        # Flatten BEV features to sequence: (B, H*W, d_model)
        bev_flat = bev_feat_with_pos.flatten(2).permute(0, 2, 1)  # (B, H*W, d_model)
        spatial_shape = (H, W)

        # Initialize queries: (N_queries, d_model) -> (B, N_queries, d_model)
        query = self.query_embedding.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos_embedding.weight.unsqueeze(0).expand(B, -1, -1)

        # Predict reference points from query positional embeddings
        reference_points = self.reference_point_head(
            self.query_pos_embedding.weight
        )  # (N_queries, 2)
        reference_points = reference_points.unsqueeze(0).expand(B, -1, -1)  # (B, N_queries, 2)

        # Create class assignment indices for each query
        # Queries 0..49 -> class 0, 50..99 -> class 1, 100..149 -> class 2
        query_classes = torch.arange(self.num_queries, device=bev_features.device)
        query_classes = query_classes // self.num_queries_per_class

        # Run through decoder layers
        intermediate_outputs = []

        for layer in self.layers:
            query = layer(
                query=query,
                bev_features=bev_flat,
                reference_points=reference_points,
                spatial_shape=spatial_shape,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate_outputs.append(self.norm(query))

        if self.return_intermediate:
            # Stack intermediate outputs: (num_layers, B, N_queries, d_model)
            output = torch.stack(intermediate_outputs, dim=0)
        else:
            output = self.norm(query)  # (B, N_queries, d_model)

        return output, reference_points, query_classes
