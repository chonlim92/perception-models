"""
DETR3D Transformer Decoder with feature sampling cross-attention.

Implements the decoder from "DETR3D: 3D Object Detection from Multi-view
Images via 3D-to-2D Queries". Each decoder layer performs self-attention
among object queries, cross-attention via 3D-to-2D feature sampling, and
a feed-forward network. Reference points are iteratively refined.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from typing import List, Tuple, Optional

from .feature_sampling import DETR3DFeatureSampler


class DETR3DCrossAttention(nn.Module):
    """Cross-attention via 3D-to-2D feature sampling.

    Instead of standard cross-attention over flattened features, this module
    uses 3D reference points associated with each query, projects them to
    camera views, samples features via bilinear interpolation, and then
    performs weighted attention over the sampled features.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.feature_sampler = DETR3DFeatureSampler(embed_dims=embed_dims)

        # Attention projections
        self.query_proj = nn.Linear(embed_dims, embed_dims)
        self.key_proj = nn.Linear(embed_dims, embed_dims)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        for proj in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.constant_(proj.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            query: Object query embeddings, shape (B, N, embed_dims).
            reference_points: 3D reference points, shape (B, N, 3).
            multi_scale_features: Multi-scale feature maps from backbone+FPN.
            intrinsics: Camera intrinsics, shape (B, num_cams, 3, 3).
            extrinsics: Camera extrinsics, shape (B, num_cams, 4, 4).
            image_shape: (H, W) of input images.

        Returns:
            Attended features, shape (B, N, embed_dims).
        """
        batch_size, num_queries, _ = query.shape

        # Sample features from multi-view images at reference point locations
        sampled_features = self.feature_sampler(
            reference_points, multi_scale_features, intrinsics, extrinsics, image_shape
        )  # (B, N, embed_dims)

        # Compute attention between queries and sampled features
        Q = self.query_proj(query)  # (B, N, embed_dims)
        K = self.key_proj(sampled_features)  # (B, N, embed_dims)
        V = self.value_proj(sampled_features)  # (B, N, embed_dims)

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, num_queries, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, num_queries, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, num_queries, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention (each query attends to its own sampled feature)
        # For DETR3D, we use a simpler approach: direct projection of sampled features
        # weighted by attention scores between query and sampled key
        attn_weights = (Q * K).sum(dim=-1, keepdim=True) * self.scale  # (B, H, N, 1)
        attn_weights = torch.sigmoid(attn_weights)  # Use sigmoid for per-query gating
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        output = attn_weights * V  # (B, H, N, head_dim)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, num_queries, self.embed_dims)

        # Output projection
        output = self.output_proj(output)

        return output


class DETR3DTransformerDecoderLayer(nn.Module):
    """Single decoder layer for DETR3D.

    Consists of:
    1. Self-attention among object queries
    2. Cross-attention via 3D-to-2D feature sampling
    3. Feed-forward network (FFN)

    Each sub-layer has LayerNorm and residual connections.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_dims: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dims: Query/feature embedding dimension.
            num_heads: Number of attention heads.
            ffn_dims: Hidden dimension of the feed-forward network.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embed_dims = embed_dims

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(embed_dims)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Cross-attention via feature sampling
        self.cross_attn = DETR3DCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dims)
        self.cross_attn_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.ffn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            query: Object queries, shape (B, N, embed_dims).
            query_pos: Positional embeddings for queries, shape (B, N, embed_dims).
            reference_points: 3D reference points, shape (B, N, 3).
            multi_scale_features: Multi-scale features from backbone.
            intrinsics: Camera intrinsics, shape (B, num_cams, 3, 3).
            extrinsics: Camera extrinsics, shape (B, num_cams, 4, 4).
            image_shape: (H, W) of input images.

        Returns:
            Updated query features, shape (B, N, embed_dims).
        """
        # 1. Self-attention with residual and LayerNorm
        q = k = query + query_pos
        self_attn_out, _ = self.self_attn(q, k, query)
        query = query + self.self_attn_dropout(self_attn_out)
        query = self.self_attn_norm(query)

        # 2. Cross-attention via feature sampling with residual and LayerNorm
        cross_attn_input = query + query_pos
        cross_attn_out = self.cross_attn(
            cross_attn_input, reference_points,
            multi_scale_features, intrinsics, extrinsics, image_shape
        )
        query = query + self.cross_attn_dropout(cross_attn_out)
        query = self.cross_attn_norm(query)

        # 3. FFN with residual and LayerNorm
        ffn_out = self.ffn(query)
        query = query + self.ffn_dropout(ffn_out)
        query = self.ffn_norm(query)

        return query


class DETR3DTransformerDecoder(nn.Module):
    """Full DETR3D Transformer Decoder.

    Stacks N decoder layers with iterative reference point refinement.
    Produces intermediate outputs from each layer for auxiliary supervision.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_dims: int = 1024,
        num_layers: int = 6,
        dropout: float = 0.1,
        num_queries: int = 900,
        pc_range: Optional[List[float]] = None,
    ):
        """
        Args:
            embed_dims: Embedding dimension for queries and features.
            num_heads: Number of attention heads.
            ffn_dims: FFN hidden dimension.
            num_layers: Number of decoder layers.
            dropout: Dropout rate.
            num_queries: Number of object queries.
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
                      Used to convert normalized reference points to absolute coords.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.num_queries = num_queries

        if pc_range is None:
            # Default range for autonomous driving (in meters)
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.pc_range = pc_range

        # Decoder layers
        self.layers = nn.ModuleList([
            DETR3DTransformerDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                ffn_dims=ffn_dims,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Learnable query embeddings
        self.query_embedding = nn.Embedding(num_queries, embed_dims)

        # Learnable positional encoding for queries
        self.query_pos_embedding = nn.Embedding(num_queries, embed_dims)

        # Learnable 3D reference points (initialized in normalized [0, 1] space)
        self.reference_points_embed = nn.Embedding(num_queries, 3)

        # Reference point refinement layers (one per decoder layer)
        self.ref_point_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 3),
            )
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        # Initialize query embeddings
        nn.init.normal_(self.query_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.query_pos_embedding.weight, mean=0, std=0.02)

        # Initialize reference points uniformly in [0, 1]
        nn.init.uniform_(self.reference_points_embed.weight, 0, 1)

        # Initialize refinement heads
        for head in self.ref_point_heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def denormalize_reference_points(self, ref_points: torch.Tensor) -> torch.Tensor:
        """Convert normalized [0,1] reference points to absolute coordinates.

        Args:
            ref_points: Normalized points, shape (..., 3) in [0, 1].

        Returns:
            Absolute 3D coordinates in world frame, shape (..., 3).
        """
        pc_range = torch.tensor(self.pc_range, device=ref_points.device, dtype=ref_points.dtype)
        min_bound = pc_range[:3]
        max_bound = pc_range[3:]
        return ref_points * (max_bound - min_bound) + min_bound

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            multi_scale_features: List of L feature maps from backbone+FPN,
                                  each (B, num_cams, C, H_l, W_l).
            intrinsics: Camera intrinsics, shape (B, num_cams, 3, 3).
            extrinsics: Camera extrinsics, shape (B, num_cams, 4, 4).
            image_shape: (H, W) of the input images.

        Returns:
            query_outputs: Final query features, shape (B, N, embed_dims).
            intermediate_outputs: List of query features from each layer
                                  (for auxiliary losses), each (B, N, embed_dims).
            intermediate_ref_points: List of reference points from each layer
                                     (normalized), each (B, N, 3).
        """
        batch_size = multi_scale_features[0].shape[0]
        device = multi_scale_features[0].device

        # Initialize queries (expand to batch size)
        query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos = self.query_pos_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Initialize reference points (sigmoid to ensure [0, 1])
        reference_points = torch.sigmoid(
            self.reference_points_embed.weight
        ).unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, 3)

        intermediate_outputs = []
        intermediate_ref_points = []

        for layer_idx, layer in enumerate(self.layers):
            # Convert normalized reference points to absolute 3D coords for projection
            ref_points_3d = self.denormalize_reference_points(reference_points)

            # Run decoder layer
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=ref_points_3d,
                multi_scale_features=multi_scale_features,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_shape=image_shape,
            )

            # Refine reference points
            ref_point_delta = self.ref_point_heads[layer_idx](query)
            # Add delta to current reference points (in normalized space)
            # Use inverse_sigmoid for numerical stability
            new_ref_points = torch.sigmoid(
                self._inverse_sigmoid(reference_points) + ref_point_delta
            )
            reference_points = new_ref_points.detach()  # Detach for next layer

            # Store intermediate outputs
            intermediate_outputs.append(query)
            intermediate_ref_points.append(new_ref_points)

        return query, intermediate_outputs, intermediate_ref_points

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Inverse of sigmoid function, clamped for numerical stability."""
        x = x.clamp(min=eps, max=1 - eps)
        return torch.log(x / (1 - x))
