"""Spatio-Contextual Fusion Transformer for CRAFT.

Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer.

This module implements the core fusion architecture that combines radar BEV
features with multi-view camera features through cross-attention mechanisms,
leveraging radar spatial context (range, azimuth, velocity, RCS) to produce
enriched fused BEV representations for downstream 3D detection heads.

Reference: CRAFT - Camera-Radar 3D Object Detection with Spatio-Contextual
Fusion Transformer (AAAI 2023).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RadarToImageProjection(nn.Module):
    """Projects radar 3D points onto 2D image planes using camera parameters.

    Given radar points in the ego vehicle coordinate frame, this module applies
    the extrinsic transformation (ego -> camera frame) followed by the intrinsic
    projection (camera frame -> pixel coordinates).

    Args:
        image_height: Height of the target image in pixels.
        image_width: Width of the target image in pixels.
    """

    def __init__(self, image_height: int, image_width: int) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width

    def forward(
        self,
        points: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Project radar points to image pixel coordinates.

        Args:
            points: Radar 3D points in ego frame [B, N_points, 3] (x, y, z).
            intrinsics: Camera intrinsic matrices [B, 3, 3].
            extrinsics: Ego-to-camera extrinsic matrices [B, 4, 4].

        Returns:
            pixel_coords: Projected pixel coordinates [B, N_points, 2] as (u, v).
            valid_mask: Boolean mask [B, N_points] indicating points that project
                within the image boundaries and are in front of the camera (z > 0).
        """
        B, N, _ = points.shape

        # Convert to homogeneous coordinates [B, N, 4]
        ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=-1)  # [B, N, 4]

        # Apply extrinsic transformation: ego -> camera frame
        # extrinsics: [B, 4, 4], points_homo: [B, N, 4]
        # Result: [B, N, 4] in camera coordinates
        points_cam = torch.einsum("bij,bnj->bni", extrinsics, points_homo)  # [B, N, 4]

        # Extract camera-frame x, y, z (discard homogeneous w)
        points_cam_xyz = points_cam[..., :3]  # [B, N, 3]
        depth = points_cam_xyz[..., 2]  # [B, N]

        # Apply intrinsic projection: camera -> pixel
        # intrinsics: [B, 3, 3], points_cam_xyz: [B, N, 3]
        points_pixel = torch.einsum("bij,bnj->bni", intrinsics, points_cam_xyz)  # [B, N, 3]

        # Normalize by depth (perspective division)
        # Avoid division by zero by clamping depth
        depth_safe = depth.clamp(min=1e-5)
        u = points_pixel[..., 0] / depth_safe  # [B, N]
        v = points_pixel[..., 1] / depth_safe  # [B, N]

        pixel_coords = torch.stack([u, v], dim=-1)  # [B, N, 2]

        # Valid mask: point is in front of camera AND within image bounds
        valid_mask = (
            (depth > 0)
            & (u >= 0)
            & (u < self.image_width)
            & (v >= 0)
            & (v < self.image_height)
        )  # [B, N]

        return pixel_coords, valid_mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for spatial positions.

    Supports both 1D sequential positions and 2D BEV grid positions.
    For 2D positions, encodes x and y coordinates separately and concatenates
    the results.

    Args:
        d_model: Dimension of the encoding vectors.
        max_len: Maximum sequence length for 1D encoding.
    """

    def __init__(self, d_model: int = 256, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model

        # Precompute 1D sinusoidal encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward_1d(self, seq_len: int) -> Tensor:
        """Get 1D positional encoding for a sequence.

        Args:
            seq_len: Length of the sequence.

        Returns:
            Positional encoding [seq_len, d_model].
        """
        return self.pe[:seq_len]

    def forward_2d(self, height: int, width: int) -> Tensor:
        """Generate 2D positional encoding for a spatial grid.

        Encodes y positions using the first half of dimensions and x positions
        using the second half, then concatenates them.

        Args:
            height: Grid height.
            width: Grid width.

        Returns:
            2D positional encoding [height * width, d_model].
        """
        half_d = self.d_model // 2

        # Y-axis encoding
        y_pos = torch.arange(height, device=self.pe.device, dtype=torch.float).unsqueeze(1)
        div_term_y = torch.exp(
            torch.arange(0, half_d, 2, device=self.pe.device, dtype=torch.float)
            * (-math.log(10000.0) / half_d)
        )
        pe_y = torch.zeros(height, half_d, device=self.pe.device)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term_y)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term_y)

        # X-axis encoding
        x_pos = torch.arange(width, device=self.pe.device, dtype=torch.float).unsqueeze(1)
        div_term_x = torch.exp(
            torch.arange(0, half_d, 2, device=self.pe.device, dtype=torch.float)
            * (-math.log(10000.0) / half_d)
        )
        pe_x = torch.zeros(width, half_d, device=self.pe.device)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term_x)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term_x)

        # Combine: repeat y for each x, repeat x for each y
        pe_y_expanded = pe_y.unsqueeze(1).expand(height, width, half_d)  # [H, W, half_d]
        pe_x_expanded = pe_x.unsqueeze(0).expand(height, width, half_d)  # [H, W, half_d]

        pe_2d = torch.cat([pe_y_expanded, pe_x_expanded], dim=-1)  # [H, W, d_model]
        pe_2d = pe_2d.reshape(height * width, self.d_model)  # [H*W, d_model]

        return pe_2d

    def forward(self, x: Tensor, mode: str = "1d") -> Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor. For '1d' mode: [B, seq_len, d_model].
                For '2d' mode: [B, H*W, d_model] (caller must provide height/width
                via forward_2d directly).
            mode: Either '1d' or '2d'.

        Returns:
            Input tensor with positional encoding added.
        """
        if mode == "1d":
            seq_len = x.size(1)
            return x + self.forward_1d(seq_len).unsqueeze(0)
        else:
            raise ValueError(
                "For 2D positional encoding, use forward_2d() directly and add "
                "the result to your tensor."
            )


class SpatialContextEncoder(nn.Module):
    """Encodes radar spatial context properties as feature embeddings.

    Takes radar physical properties (range, azimuth angle, radial velocity,
    radar cross-section) and produces a context vector that captures the
    spatial and kinematic information of each radar detection.

    Args:
        d_model: Output embedding dimension.
        n_properties: Number of radar properties to encode (default: 4 for
            range, azimuth, velocity, RCS).
    """

    def __init__(self, d_model: int = 256, n_properties: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_properties = n_properties

        # Individual property encoders
        self.range_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, d_model),
        )
        self.azimuth_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, d_model),
        )
        self.velocity_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, d_model),
        )
        self.rcs_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, d_model),
        )

        # Fusion layer to combine all property embeddings
        self.fusion = nn.Sequential(
            nn.Linear(d_model * n_properties, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, radar_properties: Tensor) -> Tensor:
        """Encode radar spatial context.

        Args:
            radar_properties: Radar properties [B, N_points, 4] where the last
                dimension contains [range, azimuth, velocity, rcs].

        Returns:
            Context embeddings [B, N_points, d_model].
        """
        range_val = radar_properties[..., 0:1]   # [B, N, 1]
        azimuth = radar_properties[..., 1:2]     # [B, N, 1]
        velocity = radar_properties[..., 2:3]    # [B, N, 1]
        rcs = radar_properties[..., 3:4]         # [B, N, 1]

        range_emb = self.range_encoder(range_val)      # [B, N, d_model]
        azimuth_emb = self.azimuth_encoder(azimuth)    # [B, N, d_model]
        velocity_emb = self.velocity_encoder(velocity)  # [B, N, d_model]
        rcs_emb = self.rcs_encoder(rcs)                # [B, N, d_model]

        # Concatenate and fuse
        combined = torch.cat(
            [range_emb, azimuth_emb, velocity_emb, rcs_emb], dim=-1
        )  # [B, N, d_model * 4]
        context = self.fusion(combined)  # [B, N, d_model]

        return context


class CrossAttentionLayer(nn.Module):
    """Multi-head cross-attention layer for radar-camera fusion.

    Radar BEV features serve as queries that attend to camera features
    sampled at projected radar locations. Uses pre-LayerNorm and residual
    connections for stable training.

    Args:
        d_model: Feature dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Pre-LayerNorm
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_out.weight)
        nn.init.zeros_(self.w_q.bias)
        nn.init.zeros_(self.w_k.bias)
        nn.init.zeros_(self.w_v.bias)
        nn.init.zeros_(self.w_out.bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute multi-head cross-attention.

        Args:
            query: Query tensor (radar features) [B, N_q, d_model].
            key: Key tensor (camera features) [B, N_kv, d_model].
            value: Value tensor (camera features) [B, N_kv, d_model].
            attn_mask: Optional attention mask [B, N_q, N_kv] or [B, n_heads, N_q, N_kv].
                True/1 values indicate positions to mask (ignore).

        Returns:
            Output tensor [B, N_q, d_model].
        """
        residual = query
        B = query.size(0)

        # Pre-LayerNorm
        query = self.norm_q(query)
        key = self.norm_kv(key)
        value = self.norm_kv(value)

        # Project to multi-head space
        Q = self.w_q(query).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.w_k(key).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.w_v(value).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        # Q: [B, n_heads, N_q, d_head], K/V: [B, n_heads, N_kv, d_head]

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, n_heads, N_q, N_kv]

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, N_q, N_kv]
            attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)  # [B, n_heads, N_q, d_head]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # [B, N_q, d_model]

        out = self.w_out(out)
        out = self.dropout(out)

        # Residual connection
        return residual + out


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention, cross-attention, and FFN.

    Applies self-attention on radar/fused features, cross-attention to camera
    features, and a feed-forward network. Uses pre-LayerNorm and residual
    connections throughout.

    Args:
        d_model: Feature dimension.
        n_heads: Number of attention heads.
        d_ffn: Hidden dimension of the feed-forward network.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention
        self.norm_self_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_self_attn = nn.Dropout(dropout)

        # Cross-attention (radar queries attend to camera features)
        self.cross_attn = CrossAttentionLayer(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Feed-forward network
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize FFN weights."""
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the decoder layer.

        Args:
            tgt: Target (radar/fused) features [B, N_q, d_model].
            memory: Source (camera) features [B, N_kv, d_model].
            tgt_mask: Optional self-attention mask.
            memory_mask: Optional cross-attention mask [B, N_q, N_kv].

        Returns:
            Updated target features [B, N_q, d_model].
        """
        # Self-attention with pre-norm and residual
        residual = tgt
        tgt_norm = self.norm_self_attn(tgt)
        self_attn_out, _ = self.self_attn(
            tgt_norm, tgt_norm, tgt_norm,
            attn_mask=tgt_mask,
            need_weights=False,
        )
        tgt = residual + self.dropout_self_attn(self_attn_out)

        # Cross-attention (uses internal pre-norm and residual)
        tgt = self.cross_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
        )

        # Feed-forward with pre-norm and residual
        residual = tgt
        tgt_norm = self.norm_ffn(tgt)
        ffn_out = self.ffn(tgt_norm)
        tgt = residual + self.dropout_ffn(ffn_out)

        return tgt


class SpatioContextualFusionTransformer(nn.Module):
    """Spatio-Contextual Fusion Transformer for camera-radar 3D detection.

    This is the main fusion module that combines radar BEV features with
    multi-view camera features using a transformer decoder architecture.
    The radar BEV features are flattened into query tokens, camera features
    are sampled at projected radar locations via bilinear interpolation, and
    spatial context from radar properties enriches the representation.

    Architecture:
        1. Flatten radar BEV features to query tokens
        2. Project radar BEV grid centers to all camera views
        3. Bilinear sample camera features at projected locations
        4. Add spatial context encoding from radar properties
        5. Process through N transformer decoder layers
        6. Reshape back to BEV spatial dimensions

    Args:
        d_model: Feature dimension (default: 256).
        n_heads: Number of attention heads (default: 8).
        d_ffn: FFN hidden dimension (default: 1024).
        n_layers: Number of transformer decoder layers (default: 6).
        dropout: Dropout probability (default: 0.1).
        radar_channels: Number of input radar BEV feature channels.
        camera_channels: Number of input camera feature channels.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        n_layers: int = 6,
        dropout: float = 0.1,
        radar_channels: int = 256,
        camera_channels: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Input projections (if channel dims differ from d_model)
        self.radar_input_proj = (
            nn.Conv2d(radar_channels, d_model, kernel_size=1)
            if radar_channels != d_model
            else nn.Identity()
        )
        self.camera_input_proj = (
            nn.Conv2d(camera_channels, d_model, kernel_size=1)
            if camera_channels != d_model
            else nn.Identity()
        )

        # Positional encoding for BEV grid
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        # Spatial context encoder for radar properties
        self.spatial_context_encoder = SpatialContextEncoder(d_model=d_model)

        # Radar-to-image projection (image size set dynamically)
        self.projection = None  # Created dynamically based on image_shape

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ffn=d_ffn,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output LayerNorm
        self.output_norm = nn.LayerNorm(d_model)

        # Output projection back to original channel dim
        self.output_proj = (
            nn.Conv2d(d_model, radar_channels, kernel_size=1)
            if radar_channels != d_model
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolutional projection weights."""
        for module in [self.radar_input_proj, self.camera_input_proj, self.output_proj]:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _generate_bev_grid_points(
        self,
        H_bev: int,
        W_bev: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Generate 3D reference points for each BEV grid cell center.

        Assumes BEV grid represents a top-down view where grid indices map
        to normalized ego-frame coordinates. The z coordinate is set to 0
        (ground plane assumption for radar).

        Args:
            H_bev: BEV grid height.
            W_bev: BEV grid width.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Grid center points [H_bev * W_bev, 3] in ego frame.
        """
        # Create normalized grid coordinates [-1, 1]
        y_coords = torch.linspace(-1.0, 1.0, H_bev, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, W_bev, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Map to ego frame: x forward, y left, z up (ground plane z=0)
        # BEV x-axis -> ego x (forward), BEV y-axis -> ego y (left)
        points_x = grid_x.reshape(-1)  # [H*W]
        points_y = grid_y.reshape(-1)  # [H*W]
        points_z = torch.zeros_like(points_x)  # [H*W] ground plane

        return torch.stack([points_x, points_y, points_z], dim=-1)  # [H*W, 3]

    def _sample_camera_features(
        self,
        camera_features: Tensor,
        pixel_coords: Tensor,
        valid_mask: Tensor,
        image_shape: Tuple[int, int],
    ) -> Tensor:
        """Sample camera features at projected pixel locations using bilinear interpolation.

        Args:
            camera_features: Camera feature maps [B, N_cams, d_model, H_img, W_img].
            pixel_coords: Projected pixel coordinates [B, N_cams, N_points, 2].
            valid_mask: Validity mask [B, N_cams, N_points].
            image_shape: Original image size (H, W) for normalization.

        Returns:
            Sampled camera features [B, N_points, d_model] aggregated across cameras.
        """
        B, N_cams, C, H_feat, W_feat = camera_features.shape
        N_points = pixel_coords.size(2)
        H_img, W_img = image_shape

        # Normalize pixel coords to [-1, 1] for grid_sample
        # pixel_coords are in original image space, need to map to feature map space
        norm_coords = pixel_coords.clone()
        norm_coords[..., 0] = (norm_coords[..., 0] / W_img) * 2.0 - 1.0  # u -> [-1, 1]
        norm_coords[..., 1] = (norm_coords[..., 1] / H_img) * 2.0 - 1.0  # v -> [-1, 1]

        # Aggregate sampled features across all cameras
        aggregated = torch.zeros(
            B, N_points, C, device=camera_features.device, dtype=camera_features.dtype
        )
        count = torch.zeros(
            B, N_points, 1, device=camera_features.device, dtype=camera_features.dtype
        )

        for cam_idx in range(N_cams):
            cam_feats = camera_features[:, cam_idx]  # [B, C, H_feat, W_feat]
            cam_coords = norm_coords[:, cam_idx]  # [B, N_points, 2]
            cam_valid = valid_mask[:, cam_idx]  # [B, N_points]

            # Reshape coords for grid_sample: [B, 1, N_points, 2]
            grid = cam_coords.unsqueeze(1)  # [B, 1, N_points, 2]

            # Bilinear sample: output [B, C, 1, N_points]
            sampled = F.grid_sample(
                cam_feats,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled = sampled.squeeze(2).permute(0, 2, 1)  # [B, N_points, C]

            # Mask out invalid projections
            cam_valid_expanded = cam_valid.unsqueeze(-1).float()  # [B, N_points, 1]
            aggregated = aggregated + sampled * cam_valid_expanded
            count = count + cam_valid_expanded

        # Average across contributing cameras (avoid division by zero)
        count = count.clamp(min=1.0)
        aggregated = aggregated / count  # [B, N_points, C]

        return aggregated

    def forward(
        self,
        radar_bev_features: Tensor,
        camera_features: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        image_shape: Tuple[int, int],
        radar_properties: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the Spatio-Contextual Fusion Transformer.

        Args:
            radar_bev_features: Radar BEV feature map [B, C, H_bev, W_bev].
            camera_features: Multi-view camera features [B, N_cams, C, H_img, W_img].
            intrinsics: Camera intrinsic matrices [B, N_cams, 3, 3].
            extrinsics: Ego-to-camera extrinsic matrices [B, N_cams, 4, 4].
            image_shape: Original image dimensions as (H, W) for projection bounds.
            radar_properties: Optional radar properties [B, H_bev * W_bev, 4]
                containing [range, azimuth, velocity, rcs] per BEV cell.
                If None, spatial context encoding is skipped.

        Returns:
            Fused BEV features [B, C, H_bev, W_bev].
        """
        B, C_in, H_bev, W_bev = radar_bev_features.shape
        N_cams = camera_features.size(1)
        H_img, W_img = image_shape
        N_points = H_bev * W_bev

        # --- Input Projections ---
        # Project radar BEV features to d_model
        radar_feats = self.radar_input_proj(radar_bev_features)  # [B, d_model, H_bev, W_bev]

        # Project camera features to d_model
        # Reshape for conv2d: [B * N_cams, C, H, W]
        B_cam, N_c, C_cam, H_feat, W_feat = camera_features.shape
        cam_feats_flat = camera_features.view(B * N_cams, C_cam, H_feat, W_feat)
        cam_feats_proj = self.camera_input_proj(cam_feats_flat)  # [B*N_cams, d_model, H, W]
        cam_feats_proj = cam_feats_proj.view(B, N_cams, self.d_model, H_feat, W_feat)

        # --- Flatten radar BEV to query tokens ---
        radar_queries = radar_feats.flatten(2).permute(0, 2, 1)  # [B, H*W, d_model]

        # --- Add 2D positional encoding ---
        pos_enc = self.positional_encoding.forward_2d(H_bev, W_bev)  # [H*W, d_model]
        radar_queries = radar_queries + pos_enc.unsqueeze(0)  # [B, H*W, d_model]

        # --- Project BEV grid points to all camera views ---
        # Generate reference points for each BEV cell
        ref_points = self._generate_bev_grid_points(
            H_bev, W_bev,
            device=radar_bev_features.device,
            dtype=radar_bev_features.dtype,
        )  # [H*W, 3]
        ref_points_batch = ref_points.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 3]

        # Create projection module with correct image dimensions
        projector = RadarToImageProjection(
            image_height=H_img, image_width=W_img
        )

        # Project to each camera and collect pixel coordinates
        all_pixel_coords = []
        all_valid_masks = []

        for cam_idx in range(N_cams):
            cam_intrinsics = intrinsics[:, cam_idx]  # [B, 3, 3]
            cam_extrinsics = extrinsics[:, cam_idx]  # [B, 4, 4]

            pixel_coords, valid_mask = projector(
                ref_points_batch, cam_intrinsics, cam_extrinsics
            )  # [B, H*W, 2], [B, H*W]

            all_pixel_coords.append(pixel_coords)
            all_valid_masks.append(valid_mask)

        pixel_coords_all = torch.stack(all_pixel_coords, dim=1)  # [B, N_cams, H*W, 2]
        valid_masks_all = torch.stack(all_valid_masks, dim=1)  # [B, N_cams, H*W]

        # --- Sample camera features at projected locations ---
        camera_memory = self._sample_camera_features(
            cam_feats_proj, pixel_coords_all, valid_masks_all, image_shape
        )  # [B, H*W, d_model]

        # --- Add spatial context encoding ---
        if radar_properties is not None:
            spatial_context = self.spatial_context_encoder(radar_properties)  # [B, H*W, d_model]
            radar_queries = radar_queries + spatial_context

        # --- Transformer Decoder ---
        tgt = radar_queries  # [B, H*W, d_model]
        memory = camera_memory  # [B, H*W, d_model]

        for layer in self.decoder_layers:
            tgt = layer(tgt=tgt, memory=memory)

        # Output normalization
        tgt = self.output_norm(tgt)  # [B, H*W, d_model]

        # --- Reshape back to BEV spatial dimensions ---
        output = tgt.permute(0, 2, 1).view(B, self.d_model, H_bev, W_bev)  # [B, d_model, H, W]

        # Project back to original channel dimension
        output = self.output_proj(output)  # [B, C_in, H_bev, W_bev]

        return output


def build_fusion_transformer(
    d_model: int = 256,
    n_heads: int = 8,
    d_ffn: int = 1024,
    n_layers: int = 6,
    dropout: float = 0.1,
    radar_channels: int = 256,
    camera_channels: int = 256,
) -> SpatioContextualFusionTransformer:
    """Factory function to build the Spatio-Contextual Fusion Transformer.

    Args:
        d_model: Internal feature dimension.
        n_heads: Number of attention heads.
        d_ffn: FFN hidden dimension.
        n_layers: Number of transformer decoder layers.
        dropout: Dropout probability.
        radar_channels: Input radar BEV feature channels.
        camera_channels: Input camera feature channels.

    Returns:
        Configured SpatioContextualFusionTransformer instance.
    """
    return SpatioContextualFusionTransformer(
        d_model=d_model,
        n_heads=n_heads,
        d_ffn=d_ffn,
        n_layers=n_layers,
        dropout=dropout,
        radar_channels=radar_channels,
        camera_channels=camera_channels,
    )
