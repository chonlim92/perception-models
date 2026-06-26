"""
Geometry-guided Kernel Transformer (GKT) for MapTR.

Projects perspective image features into BEV (Bird's Eye View) space using
camera intrinsics/extrinsics to guide cross-attention from BEV queries to
image feature maps.

Key idea: For each BEV grid cell, project it back to image planes and apply
deformable-attention-like kernels around projected locations to aggregate features.

Input:  Multi-scale image features [B, N_cams, C, H, W], camera parameters
Output: BEV features [B, C_out, bev_h, bev_w]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class FFN(nn.Module):
    """Feed-Forward Network with GELU activation and dropout."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.activation(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x


class GeometryProjection(nn.Module):
    """
    Projects BEV grid coordinates onto image planes using camera parameters.

    For each BEV (x, y) cell, computes the corresponding 2D pixel location in
    each camera image using the camera intrinsic and extrinsic matrices.
    """

    def __init__(
        self,
        bev_h: int,
        bev_w: int,
        bev_x_range: Tuple[float, float] = (-50.0, 50.0),
        bev_y_range: Tuple[float, float] = (-50.0, 50.0),
        bev_z_range: Tuple[float, float] = (-5.0, 3.0),
        num_z_anchors: int = 4,
    ):
        """
        Args:
            bev_h: BEV grid height
            bev_w: BEV grid width
            bev_x_range: (min_x, max_x) in meters
            bev_y_range: (min_y, max_y) in meters
            bev_z_range: (min_z, max_z) height range for anchor points
            num_z_anchors: Number of height anchors to sample per BEV cell
        """
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_z_anchors = num_z_anchors

        # Create BEV grid coordinates in world frame
        xs = torch.linspace(bev_x_range[0], bev_x_range[1], bev_w)
        ys = torch.linspace(bev_y_range[0], bev_y_range[1], bev_h)
        zs = torch.linspace(bev_z_range[0], bev_z_range[1], num_z_anchors)

        # Grid shape: [bev_h, bev_w, num_z, 3]
        grid_y, grid_x, grid_z = torch.meshgrid(ys, xs, zs, indexing="ij")
        # Stack as (x, y, z) world coords
        bev_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [bev_h, bev_w, num_z, 3]
        self.register_buffer("bev_coords", bev_coords)

    def forward(
        self,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project BEV grid points onto each camera's image plane.

        Args:
            intrinsics: Camera intrinsic matrices [B, N_cams, 3, 3]
            extrinsics: Camera extrinsic matrices [B, N_cams, 4, 4] (world-to-camera)
            img_h: Feature map height
            img_w: Feature map width

        Returns:
            proj_coords: Normalized 2D coordinates [B, N_cams, bev_h, bev_w, num_z, 2] in [-1, 1]
            valid_mask: Boolean mask for points within image bounds [B, N_cams, bev_h, bev_w, num_z]
        """
        B, N_cams = intrinsics.shape[:2]
        device = intrinsics.device

        # BEV coords: [bev_h, bev_w, num_z, 3]
        coords = self.bev_coords.to(device)
        num_points = self.bev_h * self.bev_w * self.num_z_anchors

        # Flatten to [num_points, 3]
        coords_flat = coords.reshape(-1, 3)

        # Convert to homogeneous: [num_points, 4]
        ones = torch.ones(num_points, 1, device=device, dtype=coords_flat.dtype)
        coords_homo = torch.cat([coords_flat, ones], dim=-1)

        # Extract rotation and translation from extrinsics
        # extrinsics: [B, N_cams, 4, 4] world-to-camera transform
        # Transform world points to camera frame
        # coords_homo: [num_points, 4] -> [1, 1, num_points, 4]
        coords_homo = coords_homo.unsqueeze(0).unsqueeze(0)

        # extrinsics: [B, N_cams, 4, 4] -> we need [B, N_cams, 4, 4] @ [1, 1, 4, num_points]
        coords_cam = torch.matmul(
            extrinsics, coords_homo.permute(0, 1, 3, 2)
        )  # [B, N_cams, 4, num_points]
        coords_cam = coords_cam[:, :, :3, :]  # [B, N_cams, 3, num_points]

        # Project to image plane: intrinsics @ camera_coords
        # intrinsics: [B, N_cams, 3, 3], coords_cam: [B, N_cams, 3, num_points]
        proj = torch.matmul(intrinsics, coords_cam)  # [B, N_cams, 3, num_points]

        # Perspective divide
        depth = proj[:, :, 2:3, :]  # [B, N_cams, 1, num_points]
        depth = depth.clamp(min=1e-5)
        pixel_coords = proj[:, :, :2, :] / depth  # [B, N_cams, 2, num_points]

        # Normalize to [-1, 1] for grid_sample compatibility
        pixel_coords_norm = pixel_coords.clone()
        pixel_coords_norm[:, :, 0, :] = (pixel_coords[:, :, 0, :] / img_w) * 2 - 1
        pixel_coords_norm[:, :, 1, :] = (pixel_coords[:, :, 1, :] / img_h) * 2 - 1

        # Reshape to [B, N_cams, bev_h, bev_w, num_z, 2]
        pixel_coords_norm = pixel_coords_norm.permute(0, 1, 3, 2)  # [B, N_cams, num_points, 2]
        pixel_coords_norm = pixel_coords_norm.reshape(
            B, N_cams, self.bev_h, self.bev_w, self.num_z_anchors, 2
        )

        # Validity mask: points in front of camera and within normalized bounds
        depth_flat = depth.squeeze(2).reshape(B, N_cams, self.bev_h, self.bev_w, self.num_z_anchors)
        valid_mask = (
            (depth_flat > 0)
            & (pixel_coords_norm[..., 0] >= -1)
            & (pixel_coords_norm[..., 0] <= 1)
            & (pixel_coords_norm[..., 1] >= -1)
            & (pixel_coords_norm[..., 1] <= 1)
        )

        return pixel_coords_norm, valid_mask


class KernelAttention(nn.Module):
    """
    Kernel-based deformable attention around projected reference points.

    For each projected BEV location in the image, samples features at learned
    offset positions around that point and computes weighted aggregation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_points: int = 8,
        num_z_anchors: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Feature dimension
            num_heads: Number of attention heads
            num_points: Number of sampling points per reference location
            num_z_anchors: Number of z-anchor heights per BEV cell
            dropout: Attention dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_z_anchors = num_z_anchors
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Learned sampling offsets: predict 2D offsets for each point
        self.offset_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_heads * num_z_anchors * num_points * 2),
        )

        # Attention weights for combining sampled points
        self.attention_weights_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_heads * num_z_anchors * num_points),
        )

        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self._initialize_offsets()

    def _initialize_offsets(self):
        """Initialize offsets to form a regular grid pattern."""
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

        # Initialize attention weights uniformly
        nn.init.zeros_(self.attention_weights_net[-1].weight)
        nn.init.zeros_(self.attention_weights_net[-1].bias)

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: BEV queries [B, Q, C] where Q = bev_h * bev_w
            value: Image features [B, N_cams, H*W, C]
            reference_points: Projected coords [B, N_cams, Q, num_z, 2] normalized to [-1, 1]
            valid_mask: [B, N_cams, Q, num_z] boolean mask

        Returns:
            Attended features [B, Q, C]
        """
        B, Q, C = query.shape
        _, N_cams, HW, _ = value.shape

        # Compute sampling offsets from queries
        offsets = self.offset_net(query)  # [B, Q, num_heads * num_z * num_points * 2]
        offsets = offsets.reshape(B, Q, self.num_heads, self.num_z_anchors, self.num_points, 2)
        # Scale offsets to be small perturbations
        offsets = offsets * 0.05

        # Compute attention weights
        attn_weights = self.attention_weights_net(query)  # [B, Q, num_heads * num_z * num_points]
        attn_weights = attn_weights.reshape(
            B, Q, self.num_heads, self.num_z_anchors, self.num_points
        )

        # Apply validity mask to attention weights before softmax
        # reference_points: [B, N_cams, Q, num_z, 2]
        # valid_mask: [B, N_cams, Q, num_z]

        # Project values
        value_proj = self.value_proj(value)  # [B, N_cams, HW, C]

        # Determine spatial dims from HW
        feat_h = feat_w = int(math.sqrt(HW))
        if feat_h * feat_w != HW:
            # Non-square: try to infer - default to closest factors
            feat_h = int(math.sqrt(HW))
            feat_w = HW // feat_h

        # Reshape value for grid_sample: [B * N_cams, C, H, W]
        value_spatial = value_proj.permute(0, 1, 3, 2).reshape(
            B * N_cams, C, feat_h, feat_w
        )

        # For each camera, sample features at offset reference points
        # reference_points: [B, N_cams, Q, num_z, 2]
        # Add offsets: we average over heads and expand reference points
        # reference_expanded: [B, N_cams, Q, num_heads, num_z, num_points, 2]
        ref_expanded = reference_points.unsqueeze(3).unsqueeze(5).expand(
            B, N_cams, Q, self.num_heads, self.num_z_anchors, self.num_points, 2
        )

        # offsets: [B, Q, num_heads, num_z, num_points, 2] -> expand for N_cams
        offsets_expanded = offsets.unsqueeze(1).expand(
            B, N_cams, Q, self.num_heads, self.num_z_anchors, self.num_points, 2
        )

        sampling_locations = ref_expanded + offsets_expanded
        # Clamp to valid range
        sampling_locations = sampling_locations.clamp(-1, 1)

        # Sample features using grid_sample for each camera
        # Flatten sampling locations: [B * N_cams, Q * num_heads * num_z * num_points, 1, 2]
        num_sample_total = Q * self.num_heads * self.num_z_anchors * self.num_points
        sampling_grid = sampling_locations.reshape(B * N_cams, num_sample_total, 1, 2)

        # grid_sample expects grid in [B, H_out, W_out, 2]
        sampled = F.grid_sample(
            value_spatial,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [B * N_cams, C, num_sample_total, 1]

        sampled = sampled.squeeze(-1)  # [B * N_cams, C, num_sample_total]
        sampled = sampled.reshape(
            B, N_cams, C, Q, self.num_heads, self.num_z_anchors, self.num_points
        )
        # Rearrange: [B, Q, num_heads, head_dim, N_cams, num_z, num_points]
        sampled = sampled.permute(0, 3, 4, 1, 5, 6, 2)
        # [B, Q, num_heads, N_cams, num_z, num_points, C]
        # Split C into heads: C = num_heads * head_dim
        sampled = sampled.reshape(
            B, Q, self.num_heads, N_cams, self.num_z_anchors, self.num_points, self.head_dim
        )

        # Compute attention weights across cameras, z-anchors, and points
        # valid_mask: [B, N_cams, Q, num_z] -> [B, Q, N_cams, num_z]
        mask_transposed = valid_mask.permute(0, 2, 1, 3)
        # Expand mask: [B, Q, 1, N_cams, num_z, 1]
        mask_expanded = mask_transposed.unsqueeze(2).unsqueeze(-1)

        # attn_weights: [B, Q, num_heads, num_z, num_points]
        # -> expand for cameras: [B, Q, num_heads, N_cams, num_z, num_points]
        attn_expanded = attn_weights.unsqueeze(3).expand(
            B, Q, self.num_heads, N_cams, self.num_z_anchors, self.num_points
        )

        # Apply mask: set invalid locations to large negative
        attn_expanded = attn_expanded.masked_fill(~mask_expanded.squeeze(-1), float("-inf"))

        # Softmax over (N_cams, num_z, num_points) jointly
        attn_flat = attn_expanded.reshape(
            B, Q, self.num_heads, N_cams * self.num_z_anchors * self.num_points
        )
        attn_flat = F.softmax(attn_flat, dim=-1)
        attn_flat = self.dropout(attn_flat)

        # Handle case where all weights are -inf (all invalid)
        attn_flat = attn_flat.nan_to_num(0.0)

        attn_final = attn_flat.reshape(
            B, Q, self.num_heads, N_cams, self.num_z_anchors, self.num_points
        )

        # Weighted sum: [B, Q, num_heads, head_dim]
        # sampled: [B, Q, num_heads, N_cams, num_z, num_points, head_dim]
        # attn_final: [B, Q, num_heads, N_cams, num_z, num_points]
        output = torch.einsum(
            "bqhczp,bqhczpd->bqhd", attn_final, sampled
        )

        # Merge heads: [B, Q, C]
        output = output.reshape(B, Q, C)
        output = self.output_proj(output)

        return output


class GKTLayer(nn.Module):
    """
    Single GKT layer: geometry-guided cross-attention + FFN with residual connections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_points: int = 8,
        num_z_anchors: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = KernelAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_points=num_points,
            num_z_anchors=num_z_anchors,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Q, C]
            value: [B, N_cams, HW, C]
            reference_points: [B, N_cams, Q, num_z, 2]
            valid_mask: [B, N_cams, Q, num_z]

        Returns:
            Updated query features [B, Q, C]
        """
        # Pre-norm cross-attention with residual
        residual = query
        query_normed = self.norm1(query)
        attn_out = self.cross_attn(query_normed, value, reference_points, valid_mask)
        query = residual + self.dropout1(attn_out)

        # Pre-norm FFN with residual
        residual = query
        query_normed = self.norm2(query)
        ffn_out = self.ffn(query_normed)
        query = residual + self.dropout2(ffn_out)

        return query


class GKT(nn.Module):
    """
    Geometry-guided Kernel Transformer.

    Transforms multi-camera perspective image features into BEV (Bird's Eye View)
    representation using geometry-guided cross-attention.

    The module:
    1. Initializes learnable BEV positional embeddings as queries
    2. Projects BEV grid coordinates onto camera image planes
    3. Applies kernel-based deformable cross-attention at projected locations
    4. Iteratively refines BEV features through multiple transformer layers
    """

    def __init__(
        self,
        embed_dim: int = 256,
        bev_h: int = 200,
        bev_w: int = 200,
        num_heads: int = 8,
        num_points: int = 8,
        num_z_anchors: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        bev_x_range: Tuple[float, float] = (-50.0, 50.0),
        bev_y_range: Tuple[float, float] = (-50.0, 50.0),
        bev_z_range: Tuple[float, float] = (-5.0, 3.0),
        input_feat_channels: Optional[List[int]] = None,
    ):
        """
        Args:
            embed_dim: BEV query and output feature dimension
            bev_h: BEV grid height
            bev_w: BEV grid width
            num_heads: Number of attention heads in cross-attention
            num_points: Number of deformable sampling points per reference
            num_z_anchors: Number of height anchors for 3D-to-2D projection
            num_layers: Number of GKT transformer layers
            ffn_dim: Hidden dimension in feed-forward networks
            dropout: Dropout rate
            bev_x_range: BEV x-axis range in meters (forward)
            bev_y_range: BEV y-axis range in meters (lateral)
            bev_z_range: Height range in meters for anchor sampling
            input_feat_channels: Channel dims of multi-scale input features (for projection)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_layers = num_layers

        # Learnable BEV positional embedding
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dim)
        nn.init.normal_(self.bev_embedding.weight, std=0.02)

        # Geometry projection module
        self.geometry_proj = GeometryProjection(
            bev_h=bev_h,
            bev_w=bev_w,
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
            bev_z_range=bev_z_range,
            num_z_anchors=num_z_anchors,
        )

        # Multi-scale feature projection layers
        if input_feat_channels is None:
            input_feat_channels = [256]
        self.input_proj = nn.ModuleList()
        for in_ch in input_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=True),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                )
            )

        # Level embedding to distinguish multi-scale features
        self.level_embed = nn.Parameter(
            torch.zeros(len(input_feat_channels), embed_dim)
        )
        nn.init.normal_(self.level_embed, std=0.02)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GKTLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_points=num_points,
                    num_z_anchors=num_z_anchors,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
            )

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # Output convolution to refine BEV features spatially
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for input projections and output convolution."""
        for proj in self.input_proj:
            for m in proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        for m in self.output_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _prepare_multi_scale_features(
        self, multi_scale_feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Project and concatenate multi-scale features into a unified representation.

        Args:
            multi_scale_feats: List of [B * N_cams, C_i, H_i, W_i] feature maps

        Returns:
            Unified features [B_N, HW_total, C]
            feat_h, feat_w of the finest scale (for reference point computation)
        """
        projected_feats = []
        for i, (feat, proj) in enumerate(zip(multi_scale_feats, self.input_proj)):
            # Project to embed_dim
            feat_proj = proj(feat)  # [B_N, C, H_i, W_i]
            BN, C, H, W = feat_proj.shape
            # Add level embedding
            feat_flat = feat_proj.flatten(2).permute(0, 2, 1)  # [B_N, H_i*W_i, C]
            feat_flat = feat_flat + self.level_embed[i].unsqueeze(0).unsqueeze(0)
            projected_feats.append(feat_flat)

        # Use the finest level's spatial dims for projection
        finest_feat = multi_scale_feats[0]
        feat_h, feat_w = finest_feat.shape[2], finest_feat.shape[3]

        # Concatenate all levels: [B_N, sum(H_i*W_i), C]
        unified = torch.cat(projected_feats, dim=1)

        return unified, feat_h, feat_w

    def forward(
        self,
        multi_scale_feats: List[torch.Tensor],
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        bev_queries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Transform perspective image features to BEV representation.

        Args:
            multi_scale_feats: List of feature maps [B * N_cams, C_i, H_i, W_i]
                from FPN backbone. If single-scale, pass a list with one element.
            camera_intrinsics: [B, N_cams, 3, 3] camera intrinsic matrices
            camera_extrinsics: [B, N_cams, 4, 4] world-to-camera extrinsic matrices
            bev_queries: Optional pre-computed BEV queries [B, bev_h*bev_w, C].
                If None, uses learnable positional embeddings.

        Returns:
            BEV features [B, C, bev_h, bev_w]
        """
        B, N_cams = camera_intrinsics.shape[:2]
        device = camera_intrinsics.device

        # Prepare multi-scale features
        # Input feats are [B * N_cams, C_i, H_i, W_i], keep only the levels we have projections for
        num_levels = min(len(multi_scale_feats), len(self.input_proj))
        feats_to_use = multi_scale_feats[:num_levels]

        unified_feats, feat_h, feat_w = self._prepare_multi_scale_features(feats_to_use)
        # unified_feats: [B * N_cams, HW_total, C]

        # Reshape to [B, N_cams, HW_total, C]
        HW_total = unified_feats.shape[1]
        value = unified_feats.reshape(B, N_cams, HW_total, self.embed_dim)

        # Initialize BEV queries
        Q = self.bev_h * self.bev_w
        if bev_queries is None:
            indices = torch.arange(Q, device=device)
            bev_queries = self.bev_embedding(indices)  # [Q, C]
            bev_queries = bev_queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]
        else:
            assert bev_queries.shape == (B, Q, self.embed_dim)

        # Compute geometry-guided reference points
        # Scale intrinsics to feature map resolution
        # The feature map is at some stride relative to the original image
        # We pass feat_h, feat_w so the projection normalizes correctly
        proj_coords, valid_mask = self.geometry_proj(
            camera_intrinsics, camera_extrinsics, feat_h, feat_w
        )
        # proj_coords: [B, N_cams, bev_h, bev_w, num_z, 2]
        # valid_mask: [B, N_cams, bev_h, bev_w, num_z]

        # Reshape for attention: [B, N_cams, Q, num_z, 2]
        reference_points = proj_coords.reshape(
            B, N_cams, Q, self.geometry_proj.num_z_anchors, 2
        )
        valid_mask_flat = valid_mask.reshape(
            B, N_cams, Q, self.geometry_proj.num_z_anchors
        )

        # Apply GKT layers iteratively
        query = bev_queries
        for layer in self.layers:
            query = layer(query, value, reference_points, valid_mask_flat)

        # Final normalization
        query = self.final_norm(query)

        # Reshape to spatial BEV grid: [B, C, bev_h, bev_w]
        bev_feat = query.permute(0, 2, 1).reshape(B, self.embed_dim, self.bev_h, self.bev_w)

        # Spatial refinement
        bev_feat = bev_feat + self.output_conv(bev_feat)

        return bev_feat
