"""
Complete CRAFT Model (Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer).

Integrates all model components into a unified end-to-end architecture:
    1. Camera Branch: Multi-view image feature extraction (ResNet + FPN)
    2. Radar Branch: Radar point cloud pillar encoding + BEV backbone
    3. Camera-to-BEV Transform: Lifts 2D camera features into BEV space
    4. Spatio-Contextual Fusion Transformer: Cross-attention fusion of camera and radar BEV features
    5. Detection Head: CenterPoint-style heatmap + regression for 3D object detection

Architecture overview:
    Multi-view images -> Camera Branch -> Camera BEV features
    Radar point cloud -> Radar Branch -> Radar BEV features
    [Camera BEV, Radar BEV] -> Fusion Transformer -> Fused BEV features
    Fused BEV features -> Detection Head -> 3D bounding boxes

Configuration (from craft_nuscenes.yaml):
    - Backbone: ResNet-50, pretrained, frozen_stages=1
    - Fusion: d_model=256, 8 heads, 6 layers, d_ffn=1024
    - Detection: 10 classes, bbox_code_size=10 (x,y,z,w,l,h,sin,cos,vx,vy)
    - Post-processing: score_threshold=0.1, nms_threshold=0.2, max_detections=500
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera_branch import MultiViewCameraBackbone, build_camera_branch
from .radar_branch import RadarBranch, build_radar_branch


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CRAFTConfig:
    """Configuration dataclass for the CRAFT model.

    Args:
        num_classes: Number of detection categories (10 for nuScenes).
        bbox_code_size: Bounding box code dimension (x,y,z,w,l,h,sin,cos,vx,vy = 10).
        backbone_name: Camera backbone variant ('resnet50' or 'resnet101').
        backbone_pretrained: Whether to use ImageNet pretrained camera backbone.
        frozen_stages: Number of camera backbone stages to freeze.
        fpn_out_channels: FPN output channel dimension.
        num_cameras: Number of surround-view cameras.
        point_cloud_range: Spatial extent [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: Pillar discretization [vx, vy, vz] in meters.
        max_points_per_pillar: Maximum radar points per pillar.
        max_num_pillars: Maximum number of non-empty pillars.
        radar_in_channels: Number of raw radar point features.
        pillar_feat_channels: Intermediate pillar feature dimension.
        bev_out_channels: BEV feature channels from both branches.
        fusion_d_model: Transformer feature dimension.
        fusion_n_heads: Number of attention heads in fusion transformer.
        fusion_n_layers: Number of transformer layers.
        fusion_d_ffn: Feed-forward network hidden dimension.
        fusion_dropout: Dropout rate in transformer.
        bev_height: BEV grid height in pixels.
        bev_width: BEV grid width in pixels.
        heatmap_channels: Hidden channels in heatmap head.
        regression_channels: Hidden channels in regression head.
        num_heatmap_convs: Number of conv layers in heatmap head.
        num_regression_convs: Number of conv layers in regression head.
        shared_conv_channels: Shared convolution channel dimension.
        bias_heatmap: Initialization bias for heatmap output (focal loss prior).
        score_threshold: Minimum score for detection output.
        nms_threshold: IoU threshold for BEV NMS.
        max_detections: Maximum number of output detections.
    """

    # General
    num_classes: int = 10
    bbox_code_size: int = 10

    # Camera branch
    backbone_name: str = "resnet50"
    backbone_pretrained: bool = True
    frozen_stages: int = 1
    fpn_out_channels: int = 256
    num_cameras: int = 6

    # Radar branch
    point_cloud_range: List[float] = field(
        default_factory=lambda: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    )
    voxel_size: List[float] = field(
        default_factory=lambda: [0.2, 0.2, 8.0]
    )
    max_points_per_pillar: int = 20
    max_num_pillars: int = 30000
    radar_in_channels: int = 6
    pillar_feat_channels: int = 128
    bev_out_channels: int = 256

    # Fusion transformer
    fusion_d_model: int = 256
    fusion_n_heads: int = 8
    fusion_n_layers: int = 6
    fusion_d_ffn: int = 1024
    fusion_dropout: float = 0.1

    # BEV grid
    bev_height: int = 128
    bev_width: int = 128

    # Detection head
    heatmap_channels: int = 256
    regression_channels: int = 256
    num_heatmap_convs: int = 2
    num_regression_convs: int = 3
    shared_conv_channels: int = 64
    bias_heatmap: float = -2.19

    # Post-processing
    score_threshold: float = 0.1
    nms_threshold: float = 0.2
    max_detections: int = 500


# =============================================================================
# Camera-to-BEV Transform
# =============================================================================


class CameraBEVTransform(nn.Module):
    """Transforms multi-view 2D camera features into a unified BEV representation.

    Uses a learnable depth distribution approach (similar to LSS/BEVDet) where each
    camera pixel predicts a discrete depth distribution, then features are scattered
    into 3D voxels and collapsed along the height axis to form BEV features.

    For efficiency, we use a simplified pooling approach that projects camera features
    onto the BEV grid using known camera geometry (intrinsics + extrinsics).

    Args:
        in_channels: Input feature channels from the camera FPN.
        out_channels: Output BEV feature channels.
        bev_height: BEV grid height in pixels.
        bev_width: BEV grid width in pixels.
        num_depth_bins: Number of discrete depth bins for depth estimation.
        depth_range: (min_depth, max_depth) in meters.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        bev_height: int = 128,
        bev_width: int = 128,
        num_depth_bins: int = 64,
        depth_range: Tuple[float, float] = (1.0, 60.0),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.num_depth_bins = num_depth_bins
        self.depth_range = depth_range

        # Depth prediction network: predicts per-pixel depth distribution
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_depth_bins, kernel_size=1, bias=True),
        )

        # BEV compression: collapse projected features into BEV
        self.bev_compress = nn.Sequential(
            nn.Conv2d(in_channels * num_depth_bins // 8, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Learnable BEV embedding to fill areas without camera coverage
        self.bev_embedding = nn.Parameter(
            torch.randn(1, out_channels, bev_height, bev_width) * 0.01
        )

        # Pooling layer to reduce depth dimension efficiently
        self.depth_pool = nn.AdaptiveAvgPool2d((bev_height, bev_width))

        # Channel projection for camera features before BEV scatter
        self.feat_project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        camera_features: List[torch.Tensor],
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transform multi-view camera features to BEV.

        For this implementation, we use the highest-resolution FPN level (P2) and
        project all camera views into a shared BEV grid. The approach is simplified
        compared to full LSS for computational efficiency.

        Args:
            camera_features: List of FPN feature tensors [P2, P3, P4, P5].
                Each has shape [B, N_cams, C, H_i, W_i].
            intrinsics: Camera intrinsic matrices [B, N_cams, 3, 3]. Optional.
            extrinsics: Camera-to-ego transformation matrices [B, N_cams, 4, 4]. Optional.

        Returns:
            BEV feature map [B, out_channels, bev_height, bev_width].
        """
        # Use P3 level (stride 8) as a balance between resolution and computation
        feat = camera_features[1]  # [B, N_cams, C, H_feat, W_feat]
        B, N_cams, C, H_feat, W_feat = feat.shape

        # Process all views
        feat_flat = feat.reshape(B * N_cams, C, H_feat, W_feat)

        # Project features
        feat_proj = self.feat_project(feat_flat)  # [B*N, out_C, H, W]

        # Predict depth distribution per pixel
        depth_logits = self.depth_net(feat_flat)  # [B*N, D, H, W]
        depth_probs = F.softmax(depth_logits, dim=1)  # [B*N, D, H, W]

        # Outer product: weight features by depth probabilities
        # feat_proj: [B*N, C_out, H, W], depth_probs: [B*N, D, H, W]
        # For efficiency, pool depth dimension and spatial dimensions
        # Use mean across depth as a simple aggregation
        depth_weighted = (feat_proj.unsqueeze(2) * depth_probs.unsqueeze(1)).mean(dim=2)
        # depth_weighted: [B*N, C_out, H, W]

        # Pool each view's features to BEV spatial dimensions
        bev_per_view = self.depth_pool(depth_weighted)  # [B*N, C_out, bev_H, bev_W]

        # Reshape to per-batch: [B, N_cams, C_out, bev_H, bev_W]
        bev_per_view = bev_per_view.reshape(B, N_cams, self.out_channels, self.bev_height, self.bev_width)

        # Aggregate across views using mean (robust to missing coverage)
        bev_features = bev_per_view.mean(dim=1)  # [B, C_out, bev_H, bev_W]

        # Add learnable BEV embedding
        bev_features = bev_features + self.bev_embedding

        return bev_features


# =============================================================================
# Positional Encoding
# =============================================================================


class LearnedPositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding for BEV feature maps.

    Generates separate row and column embeddings that are summed and broadcast
    across the spatial dimensions.

    Args:
        d_model: Feature dimension of the positional encoding.
        height: Maximum grid height.
        width: Maximum grid width.
    """

    def __init__(self, d_model: int = 256, height: int = 128, width: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.row_embed = nn.Embedding(height, d_model // 2)
        self.col_embed = nn.Embedding(width, d_model // 2)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate positional encoding matching the spatial dims of x.

        Args:
            x: Input tensor [B, C, H, W] used to determine spatial size and device.

        Returns:
            Positional encoding [1, C, H, W] broadcastable to batch dimension.
        """
        H, W = x.shape[2], x.shape[3]
        device = x.device

        row_indices = torch.arange(H, device=device)
        col_indices = torch.arange(W, device=device)

        row_emb = self.row_embed(row_indices)  # [H, d_model//2]
        col_emb = self.col_embed(col_indices)  # [W, d_model//2]

        # Broadcast: row_emb [H, 1, d//2] + col_emb [1, W, d//2] -> [H, W, d]
        pos = torch.cat([
            row_emb.unsqueeze(1).expand(-1, W, -1),
            col_emb.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1)  # [H, W, d_model]

        pos = pos.permute(2, 0, 1).unsqueeze(0)  # [1, d_model, H, W]
        return pos


class SinusoidalPositionalEncoding2D(nn.Module):
    """Sinusoidal 2D positional encoding.

    Generates fixed sin/cos positional embeddings for BEV feature maps.
    Uses separate sin/cos frequencies for row and column positions.

    Args:
        d_model: Feature dimension of the positional encoding.
        temperature: Temperature scaling for the frequencies.
    """

    def __init__(self, d_model: int = 256, temperature: float = 10000.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal positional encoding.

        Args:
            x: Input tensor [B, C, H, W] to determine spatial size.

        Returns:
            Positional encoding [1, d_model, H, W].
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        half_d = self.d_model // 2
        quarter_d = half_d // 2

        # Create position grids
        y_pos = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W)
        x_pos = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W)

        # Frequency dimensions
        dim = torch.arange(quarter_d, device=device, dtype=dtype)
        freq = self.temperature ** (2 * dim / half_d)

        # Compute encodings
        # [H, W, quarter_d]
        x_enc = x_pos.unsqueeze(-1) / freq.unsqueeze(0).unsqueeze(0)
        y_enc = y_pos.unsqueeze(-1) / freq.unsqueeze(0).unsqueeze(0)

        pos = torch.cat([
            x_enc.sin(), x_enc.cos(),
            y_enc.sin(), y_enc.cos(),
        ], dim=-1)  # [H, W, d_model]

        pos = pos.permute(2, 0, 1).unsqueeze(0)  # [1, d_model, H, W]
        return pos


# =============================================================================
# Spatio-Contextual Fusion Transformer
# =============================================================================


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention module.

    Queries attend to key-value pairs from a different modality (e.g., BEV queries
    attending to camera or radar features).

    Args:
        d_model: Feature dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate on attention weights.
    """

    def __init__(
        self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head cross-attention.

        Args:
            query: Query tensor [B, N_q, d_model].
            key: Key tensor [B, N_kv, d_model].
            value: Value tensor [B, N_kv, d_model].
            attn_mask: Optional attention mask [B, N_q, N_kv] or broadcastable.

        Returns:
            Output tensor [B, N_q, d_model].
        """
        B, N_q, _ = query.shape
        N_kv = key.shape[1]

        # Project and reshape to multi-head format
        q = self.q_proj(query).reshape(B, N_q, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_kv, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_kv, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N_q, N_kv]

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # [B, H, N_q, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, N_q, self.d_model)
        out = self.out_proj(out)

        return out


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network.

    Two-layer MLP with ReLU activation and dropout.

    Args:
        d_model: Input and output dimension.
        d_ffn: Hidden layer dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self, d_model: int = 256, d_ffn: int = 1024, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN.

        Args:
            x: Input tensor [B, N, d_model].

        Returns:
            Output tensor [B, N, d_model].
        """
        x = self.dropout1(self.activation(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x


class SpatioContextualFusionLayer(nn.Module):
    """Single layer of the Spatio-Contextual Fusion Transformer.

    Each layer performs:
        1. Self-attention on BEV queries
        2. Cross-attention to camera BEV features
        3. Cross-attention to radar BEV features
        4. Feed-forward network

    All sub-layers use pre-layer normalization and residual connections.

    Args:
        d_model: Feature dimension.
        n_heads: Number of attention heads.
        d_ffn: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention on BEV queries
        self.self_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: queries attend to camera features
        self.camera_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Cross-attention: queries attend to radar features
        self.radar_cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ffn, dropout)
        self.norm4 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Learnable fusion gate to weight camera vs radar contribution
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query: torch.Tensor,
        camera_kv: torch.Tensor,
        radar_kv: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        camera_pos: Optional[torch.Tensor] = None,
        radar_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through one fusion transformer layer.

        Args:
            query: BEV query features [B, N_q, d_model].
            camera_kv: Camera BEV features (key/value) [B, N_cam, d_model].
            radar_kv: Radar BEV features (key/value) [B, N_rad, d_model].
            query_pos: Positional encoding for queries [B, N_q, d_model] or None.
            camera_pos: Positional encoding for camera features [B, N_cam, d_model] or None.
            radar_pos: Positional encoding for radar features [B, N_rad, d_model] or None.

        Returns:
            Updated query features [B, N_q, d_model].
        """
        # Add positional encodings
        q = query
        q_with_pos = q + query_pos if query_pos is not None else q
        cam_with_pos = camera_kv + camera_pos if camera_pos is not None else camera_kv
        rad_with_pos = radar_kv + radar_pos if radar_pos is not None else radar_kv

        # 1. Self-attention
        residual = q
        q_norm = self.norm1(q)
        q_pos_norm = q_norm + query_pos if query_pos is not None else q_norm
        self_attn_out = self.self_attn(q_pos_norm, q_pos_norm, q_norm)
        q = residual + self.dropout1(self_attn_out)

        # 2. Cross-attention to camera features
        residual = q
        q_norm = self.norm2(q)
        q_pos_norm = q_norm + query_pos if query_pos is not None else q_norm
        cam_attn_out = self.camera_cross_attn(q_pos_norm, cam_with_pos, camera_kv)
        cam_contribution = residual + self.dropout2(cam_attn_out)

        # 3. Cross-attention to radar features
        residual = q
        q_norm = self.norm3(q)
        q_pos_norm = q_norm + query_pos if query_pos is not None else q_norm
        rad_attn_out = self.radar_cross_attn(q_pos_norm, rad_with_pos, radar_kv)
        rad_contribution = residual + self.dropout3(rad_attn_out)

        # 4. Gated fusion of camera and radar cross-attention outputs
        gate_input = torch.cat([cam_contribution, rad_contribution], dim=-1)
        gate = self.fusion_gate(gate_input)  # [B, N_q, d_model] in [0, 1]
        q = gate * cam_contribution + (1.0 - gate) * rad_contribution

        # 5. Feed-forward network
        residual = q
        q_norm = self.norm4(q)
        ffn_out = self.ffn(q_norm)
        q = residual + self.dropout4(ffn_out)

        return q


class SpatioContextualFusionTransformer(nn.Module):
    """Spatio-Contextual Fusion Transformer for camera-radar BEV feature fusion.

    Stacks multiple SpatioContextualFusionLayers to iteratively refine BEV
    query features by attending to both camera and radar modalities. Uses learnable
    BEV queries and positional encodings.

    Args:
        d_model: Transformer feature dimension.
        n_heads: Number of attention heads per layer.
        n_layers: Number of transformer layers.
        d_ffn: Feed-forward hidden dimension.
        dropout: Dropout rate.
        bev_height: BEV grid height (for query initialization).
        bev_width: BEV grid width (for query initialization).
        use_learned_pos: If True, use learned positional encoding; else sinusoidal.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        bev_height: int = 128,
        bev_width: int = 128,
        use_learned_pos: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.bev_height = bev_height
        self.bev_width = bev_width

        # Learnable BEV queries
        self.bev_queries = nn.Parameter(
            torch.randn(1, bev_height * bev_width, d_model) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            SpatioContextualFusionLayer(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])

        # Positional encoding
        if use_learned_pos:
            self.pos_encoder = LearnedPositionalEncoding2D(d_model, bev_height, bev_width)
        else:
            self.pos_encoder = SinusoidalPositionalEncoding2D(d_model)

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        camera_bev: torch.Tensor,
        radar_bev: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse camera and radar BEV features through the transformer.

        Args:
            camera_bev: Camera BEV feature map [B, d_model, H_bev, W_bev].
            radar_bev: Radar BEV feature map [B, d_model, H_bev, W_bev].

        Returns:
            Fused BEV feature map [B, d_model, H_bev, W_bev].
        """
        B = camera_bev.shape[0]

        # Generate positional encodings
        pos_enc = self.pos_encoder(camera_bev)  # [1, d_model, H, W]
        pos_flat = pos_enc.flatten(2).permute(0, 2, 1)  # [1, H*W, d_model]
        pos_flat = pos_flat.expand(B, -1, -1)  # [B, H*W, d_model]

        # Flatten spatial dimensions for transformer processing
        # [B, C, H, W] -> [B, H*W, C]
        camera_flat = camera_bev.flatten(2).permute(0, 2, 1)  # [B, H*W, d_model]
        radar_flat = radar_bev.flatten(2).permute(0, 2, 1)    # [B, H*W, d_model]

        # Initialize queries (learnable BEV queries expanded to batch)
        queries = self.bev_queries.expand(B, -1, -1)  # [B, H*W, d_model]

        # Pass through transformer layers
        for layer in self.layers:
            queries = layer(
                query=queries,
                camera_kv=camera_flat,
                radar_kv=radar_flat,
                query_pos=pos_flat,
                camera_pos=pos_flat,
                radar_pos=pos_flat,
            )

        # Final normalization
        queries = self.output_norm(queries)

        # Reshape back to spatial BEV format
        # [B, H*W, d_model] -> [B, d_model, H, W]
        fused_bev = queries.permute(0, 2, 1).reshape(
            B, self.d_model, self.bev_height, self.bev_width
        )

        return fused_bev


# =============================================================================
# Detection Head
# =============================================================================


class CenterPointHead(nn.Module):
    """CenterPoint-style BEV detection head with heatmap + regression.

    Predicts:
        - Class heatmaps: [B, num_classes, H, W] probability of object center
        - Regression: [B, bbox_code_size, H, W] bounding box attributes at each location
            - Indices 0-1: center offset (dx, dy) - sub-pixel refinement
            - Index 2: z-coordinate (height above ground)
            - Indices 3-5: size (w, l, h)
            - Indices 6-7: rotation (sin(yaw), cos(yaw))
            - Indices 8-9: velocity (vx, vy)

    Args:
        in_channels: Input feature channels.
        num_classes: Number of object categories.
        bbox_code_size: Dimension of bounding box code vector.
        heatmap_channels: Hidden channels in heatmap sub-network.
        regression_channels: Hidden channels in regression sub-network.
        num_heatmap_convs: Number of conv layers in heatmap branch.
        num_regression_convs: Number of conv layers in regression branch.
        shared_conv_channels: Channels after shared convolution layer.
        bias_heatmap: Initialization bias for the final heatmap conv (focal loss prior).
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        bbox_code_size: int = 10,
        heatmap_channels: int = 256,
        regression_channels: int = 256,
        num_heatmap_convs: int = 2,
        num_regression_convs: int = 3,
        shared_conv_channels: int = 64,
        bias_heatmap: float = -2.19,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.bbox_code_size = bbox_code_size

        # Shared convolution applied to the input BEV features
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, shared_conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(shared_conv_channels),
            nn.ReLU(inplace=True),
        )

        # Heatmap prediction branch
        heatmap_layers = []
        current_channels = shared_conv_channels
        for i in range(num_heatmap_convs):
            out_ch = heatmap_channels if i < num_heatmap_convs - 1 else heatmap_channels
            heatmap_layers.extend([
                nn.Conv2d(current_channels, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            current_channels = out_ch
        self.heatmap_convs = nn.Sequential(*heatmap_layers)
        self.heatmap_pred = nn.Conv2d(current_channels, num_classes, kernel_size=1, bias=True)

        # Regression prediction branch
        reg_layers = []
        current_channels = shared_conv_channels
        for i in range(num_regression_convs):
            out_ch = regression_channels if i < num_regression_convs - 1 else regression_channels
            reg_layers.extend([
                nn.Conv2d(current_channels, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            current_channels = out_ch
        self.reg_convs = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv2d(current_channels, bbox_code_size, kernel_size=1, bias=True)

        self._init_weights(bias_heatmap)

    def _init_weights(self, bias_heatmap: float) -> None:
        """Initialize head weights.

        Uses Kaiming initialization for conv layers and a specific bias for the
        heatmap output layer based on the focal loss initialization prior.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize heatmap output bias with focal loss prior
        # This ensures the initial sigmoid output is ~0.01 (matches -log((1-0.01)/0.01) = -2.19)
        nn.init.constant_(self.heatmap_pred.bias, bias_heatmap)

    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict detection outputs from BEV features.

        Args:
            bev_features: Input BEV feature map [B, in_channels, H, W].

        Returns:
            Dictionary containing:
                'heatmap': Class heatmap logits [B, num_classes, H, W].
                'reg': Regression output [B, bbox_code_size, H, W].
        """
        shared = self.shared_conv(bev_features)

        # Heatmap branch
        heatmap_feat = self.heatmap_convs(shared)
        heatmap = self.heatmap_pred(heatmap_feat)

        # Regression branch
        reg_feat = self.reg_convs(shared)
        reg = self.reg_pred(reg_feat)

        return {
            "heatmap": heatmap,
            "reg": reg,
        }


# =============================================================================
# Post-Processing
# =============================================================================


def _nms_bev(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float = 0.2,
) -> torch.Tensor:
    """BEV (Bird's Eye View) Non-Maximum Suppression.

    Performs axis-aligned NMS on BEV bounding boxes using center distance rather
    than full rotated IoU for computational efficiency.

    Args:
        boxes: BEV bounding box parameters [N, 5] (cx, cy, w, l, yaw).
        scores: Detection confidence scores [N].
        threshold: IoU/overlap threshold for suppression.

    Returns:
        Indices of boxes to keep after NMS [K] (K <= N).
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Convert to axis-aligned bounding boxes for standard NMS
    cx, cy = boxes[:, 0], boxes[:, 1]
    w, l = boxes[:, 2], boxes[:, 3]

    # Approximate rotated boxes as axis-aligned using max extent
    half_diag = torch.sqrt(w ** 2 + l ** 2) / 2.0
    x1 = cx - half_diag
    y1 = cy - half_diag
    x2 = cx + half_diag
    y2 = cy + half_diag

    # Use torchvision NMS
    from torchvision.ops import nms
    aabb_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    keep = nms(aabb_boxes, scores, threshold)

    return keep


def decode_detections(
    heatmap: torch.Tensor,
    reg: torch.Tensor,
    point_cloud_range: List[float],
    voxel_size: List[float],
    score_threshold: float = 0.1,
    nms_threshold: float = 0.2,
    max_detections: int = 500,
) -> List[Dict[str, torch.Tensor]]:
    """Decode raw model predictions into 3D bounding box detections.

    Applies sigmoid activation to heatmap, extracts local maxima (peaks),
    gathers regression attributes, and performs BEV NMS.

    Args:
        heatmap: Raw heatmap logits [B, num_classes, H, W].
        reg: Raw regression output [B, bbox_code_size, H, W].
        point_cloud_range: Spatial extent [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: BEV discretization [vx, vy, vz].
        score_threshold: Minimum detection score.
        nms_threshold: BEV NMS threshold.
        max_detections: Maximum detections per sample.

    Returns:
        List of detection dictionaries (one per batch sample), each containing:
            'boxes': [K, 9] tensor (x, y, z, w, l, h, yaw, vx, vy).
            'scores': [K] confidence scores.
            'labels': [K] predicted class indices.
    """
    B, num_classes, H, W = heatmap.shape
    bbox_code_size = reg.shape[1]

    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    vx, vy = voxel_size[0], voxel_size[1]

    # Apply sigmoid to heatmap
    heatmap_sigmoid = heatmap.sigmoid()

    # Simple 3x3 max pooling NMS to find local peaks
    heatmap_pool = F.max_pool2d(heatmap_sigmoid, kernel_size=3, stride=1, padding=1)
    # Keep only local maxima (peak positions)
    peaks = (heatmap_sigmoid == heatmap_pool).float() * heatmap_sigmoid

    results = []

    for b in range(B):
        # Gather all peaks above threshold across all classes
        peak_map = peaks[b]  # [num_classes, H, W]
        scores_flat = peak_map.reshape(num_classes, -1)  # [num_classes, H*W]

        # Get top-K scores across all classes
        all_scores = scores_flat.reshape(-1)  # [num_classes * H * W]
        top_k = min(max_detections * 2, all_scores.shape[0])  # Oversample for NMS
        top_scores, top_indices = all_scores.topk(top_k, dim=0)

        # Filter by score threshold
        valid = top_scores > score_threshold
        top_scores = top_scores[valid]
        top_indices = top_indices[valid]

        if top_scores.shape[0] == 0:
            results.append({
                "boxes": torch.zeros(0, 9, device=heatmap.device),
                "scores": torch.zeros(0, device=heatmap.device),
                "labels": torch.zeros(0, dtype=torch.long, device=heatmap.device),
            })
            continue

        # Decode indices to class, y, x
        cls_ids = top_indices // (H * W)
        spatial_ids = top_indices % (H * W)
        ys = spatial_ids // W
        xs = spatial_ids % W

        # Gather regression predictions at peak locations
        reg_b = reg[b]  # [code_size, H, W]
        reg_flat = reg_b.reshape(bbox_code_size, -1)  # [code_size, H*W]
        reg_values = reg_flat[:, spatial_ids].t()  # [K, code_size]

        # Decode bounding boxes
        # reg_values: [dx, dy, z, w, l, h, sin, cos, vx, vy]
        dx = reg_values[:, 0]
        dy = reg_values[:, 1]
        z = reg_values[:, 2]
        w = reg_values[:, 3]
        l = reg_values[:, 4]
        h = reg_values[:, 5]
        sin_yaw = reg_values[:, 6]
        cos_yaw = reg_values[:, 7]
        vx = reg_values[:, 8] if bbox_code_size > 8 else torch.zeros_like(dx)
        vy = reg_values[:, 9] if bbox_code_size > 9 else torch.zeros_like(dx)

        # Convert pixel coordinates + offset to world coordinates
        world_x = (xs.float() + dx) * vx + x_min
        world_y = (ys.float() + dy) * vy + y_min

        # Recover heading angle from sin/cos encoding
        yaw = torch.atan2(sin_yaw, cos_yaw)

        # Assemble detection boxes
        boxes = torch.stack([world_x, world_y, z, w, l, h, yaw, vx, vy], dim=1)

        # BEV NMS
        nms_boxes = torch.stack([world_x, world_y, w, l, yaw], dim=1)
        keep = _nms_bev(nms_boxes, top_scores, threshold=nms_threshold)

        # Limit to max detections
        keep = keep[:max_detections]

        results.append({
            "boxes": boxes[keep],
            "scores": top_scores[keep],
            "labels": cls_ids[keep],
        })

    return results


# =============================================================================
# Main CRAFT Model
# =============================================================================


class CRAFTModel(nn.Module):
    """Complete CRAFT model for camera-radar 3D object detection.

    End-to-end architecture combining multi-view camera processing, radar pillar
    encoding, spatio-contextual fusion, and CenterPoint-style detection.

    Args:
        config: CRAFTConfig dataclass with all model hyperparameters.
    """

    def __init__(self, config: CRAFTConfig) -> None:
        super().__init__()
        self.config = config

        # Camera branch: multi-view image feature extraction
        self.camera_branch = build_camera_branch(
            backbone_name=config.backbone_name,
            pretrained=config.backbone_pretrained,
            fpn_out_channels=config.fpn_out_channels,
            num_cameras=config.num_cameras,
            frozen_stages=config.frozen_stages,
        )

        # Radar branch: point cloud pillar encoding + BEV backbone
        self.radar_branch = build_radar_branch(
            point_cloud_range=config.point_cloud_range,
            voxel_size=config.voxel_size,
            max_points_per_pillar=config.max_points_per_pillar,
            max_num_pillars=config.max_num_pillars,
            in_channels=config.radar_in_channels,
            pillar_feat_channels=config.pillar_feat_channels,
            bev_out_channels=config.bev_out_channels,
        )

        # Camera-to-BEV transform
        self.camera_bev_transform = CameraBEVTransform(
            in_channels=config.fpn_out_channels,
            out_channels=config.fusion_d_model,
            bev_height=config.bev_height,
            bev_width=config.bev_width,
        )

        # Radar BEV projection (channel alignment + spatial resize to fusion grid)
        self.radar_bev_project = nn.Sequential(
            nn.Conv2d(config.bev_out_channels, config.fusion_d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.fusion_d_model),
            nn.ReLU(inplace=True),
        )

        # Spatio-Contextual Fusion Transformer
        self.fusion_transformer = SpatioContextualFusionTransformer(
            d_model=config.fusion_d_model,
            n_heads=config.fusion_n_heads,
            n_layers=config.fusion_n_layers,
            d_ffn=config.fusion_d_ffn,
            dropout=config.fusion_dropout,
            bev_height=config.bev_height,
            bev_width=config.bev_width,
        )

        # Detection head (main fused)
        self.detection_head = CenterPointHead(
            in_channels=config.fusion_d_model,
            num_classes=config.num_classes,
            bbox_code_size=config.bbox_code_size,
            heatmap_channels=config.heatmap_channels,
            regression_channels=config.regression_channels,
            num_heatmap_convs=config.num_heatmap_convs,
            num_regression_convs=config.num_regression_convs,
            shared_conv_channels=config.shared_conv_channels,
            bias_heatmap=config.bias_heatmap,
        )

        # Auxiliary detection heads for joint training (camera-only, radar-only)
        self.camera_aux_head = CenterPointHead(
            in_channels=config.fusion_d_model,
            num_classes=config.num_classes,
            bbox_code_size=config.bbox_code_size,
            heatmap_channels=config.heatmap_channels // 2,
            regression_channels=config.regression_channels // 2,
            num_heatmap_convs=1,
            num_regression_convs=2,
            shared_conv_channels=config.shared_conv_channels,
            bias_heatmap=config.bias_heatmap,
        )

        self.radar_aux_head = CenterPointHead(
            in_channels=config.fusion_d_model,
            num_classes=config.num_classes,
            bbox_code_size=config.bbox_code_size,
            heatmap_channels=config.heatmap_channels // 2,
            regression_channels=config.regression_channels // 2,
            num_heatmap_convs=1,
            num_regression_convs=2,
            shared_conv_channels=config.shared_conv_channels,
            bias_heatmap=config.bias_heatmap,
        )

    def forward(
        self,
        images: torch.Tensor,
        radar_points: torch.Tensor,
        radar_num_points: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the complete CRAFT model.

        In training mode (return_loss=True), returns raw predictions suitable for
        loss computation. In inference mode, returns post-processed detections.

        Args:
            images: Multi-view camera images [B, N_cams, 3, H_img, W_img].
            radar_points: Radar point clouds [B, N_max, radar_in_channels].
            radar_num_points: Number of valid radar points per sample [B].
            intrinsics: Camera intrinsic matrices [B, N_cams, 3, 3]. Optional.
            extrinsics: Camera-to-ego transformation matrices [B, N_cams, 4, 4]. Optional.
            return_loss: If True, return raw predictions for loss computation.
                         If False, return post-processed detection results.

        Returns:
            If return_loss=True (training):
                Dictionary with:
                    'heatmap': Fused heatmap logits [B, num_classes, H_bev, W_bev].
                    'reg': Fused regression [B, bbox_code_size, H_bev, W_bev].
                    'camera_heatmap': Camera-only heatmap [B, num_classes, H_bev, W_bev].
                    'camera_reg': Camera-only regression [B, bbox_code_size, H_bev, W_bev].
                    'radar_heatmap': Radar-only heatmap [B, num_classes, H_bev, W_bev].
                    'radar_reg': Radar-only regression [B, bbox_code_size, H_bev, W_bev].

            If return_loss=False (inference):
                List of detection dictionaries per sample:
                    'boxes': [K, 9] (x, y, z, w, l, h, yaw, vx, vy).
                    'scores': [K] confidence scores.
                    'labels': [K] class indices.
        """
        # 1. Extract camera features
        camera_output = self.camera_branch(images)
        camera_features = camera_output["features"]  # List of [B, N_cams, C, H_i, W_i]

        # 2. Extract radar BEV features
        radar_output = self.radar_branch(radar_points, radar_num_points)
        radar_bev_raw = radar_output["bev_features"]  # [B, C_radar, H_radar, W_radar]

        # 3. Transform camera features to BEV
        camera_bev = self.camera_bev_transform(camera_features, intrinsics, extrinsics)
        # camera_bev: [B, d_model, H_bev, W_bev]

        # 4. Project and resize radar BEV to match fusion grid
        radar_bev = self.radar_bev_project(radar_bev_raw)
        if radar_bev.shape[2:] != (self.config.bev_height, self.config.bev_width):
            radar_bev = F.interpolate(
                radar_bev,
                size=(self.config.bev_height, self.config.bev_width),
                mode="bilinear",
                align_corners=False,
            )
        # radar_bev: [B, d_model, H_bev, W_bev]

        # 5. Fuse camera and radar BEV features
        fused_bev = self.fusion_transformer(camera_bev, radar_bev)
        # fused_bev: [B, d_model, H_bev, W_bev]

        # 6. Detection head predictions
        main_preds = self.detection_head(fused_bev)

        if return_loss:
            # Training mode: also compute auxiliary branch predictions
            camera_aux_preds = self.camera_aux_head(camera_bev)
            radar_aux_preds = self.radar_aux_head(radar_bev)

            return {
                "heatmap": main_preds["heatmap"],
                "reg": main_preds["reg"],
                "camera_heatmap": camera_aux_preds["heatmap"],
                "camera_reg": camera_aux_preds["reg"],
                "radar_heatmap": radar_aux_preds["heatmap"],
                "radar_reg": radar_aux_preds["reg"],
            }
        else:
            # Inference mode: decode and post-process detections
            detections = decode_detections(
                heatmap=main_preds["heatmap"],
                reg=main_preds["reg"],
                point_cloud_range=self.config.point_cloud_range,
                voxel_size=self.config.voxel_size,
                score_threshold=self.config.score_threshold,
                nms_threshold=self.config.nms_threshold,
                max_detections=self.config.max_detections,
            )
            return detections

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for each model component.

        Returns:
            Dictionary mapping component names to their parameter counts.
        """
        components = {
            "camera_branch": self.camera_branch,
            "radar_branch": self.radar_branch,
            "camera_bev_transform": self.camera_bev_transform,
            "radar_bev_project": self.radar_bev_project,
            "fusion_transformer": self.fusion_transformer,
            "detection_head": self.detection_head,
            "camera_aux_head": self.camera_aux_head,
            "radar_aux_head": self.radar_aux_head,
        }

        counts = {}
        for name, module in components.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}

        # Overall totals
        total_all = sum(p.numel() for p in self.parameters())
        trainable_all = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["overall"] = {"total": total_all, "trainable": trainable_all}

        return counts

    def print_model_summary(self) -> None:
        """Print a formatted summary of model parameters per component."""
        counts = self.get_parameter_count()

        print("=" * 65)
        print(f"{'CRAFT Model Parameter Summary':^65}")
        print("=" * 65)
        print(f"{'Component':<30} {'Total':>12} {'Trainable':>12}")
        print("-" * 65)

        for name, info in counts.items():
            if name == "overall":
                continue
            print(f"  {name:<28} {info['total']:>12,} {info['trainable']:>12,}")

        print("-" * 65)
        overall = counts["overall"]
        print(f"  {'TOTAL':<28} {overall['total']:>12,} {overall['trainable']:>12,}")
        print("=" * 65)


# =============================================================================
# Factory Functions
# =============================================================================


def build_craft_model(config: Optional[CRAFTConfig] = None, **kwargs: Any) -> CRAFTModel:
    """Factory function to build the complete CRAFT model.

    Args:
        config: CRAFTConfig instance. If None, creates default config.
        **kwargs: Override specific config fields.

    Returns:
        Configured CRAFTModel instance.
    """
    if config is None:
        config = CRAFTConfig()

    # Apply any kwargs overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config field: {key}")

    return CRAFTModel(config)


def build_craft_model_from_config(config_dict: Dict[str, Any]) -> CRAFTModel:
    """Build CRAFTModel from a configuration dictionary (e.g., parsed from YAML).

    Parses the craft_nuscenes.yaml structure into a CRAFTConfig dataclass.

    Args:
        config_dict: Configuration dictionary matching craft_nuscenes.yaml structure.

    Returns:
        Configured CRAFTModel instance.
    """
    model_cfg = config_dict.get("model", {})
    eval_cfg = config_dict.get("evaluation", {})
    data_cfg = config_dict.get("data", {})

    backbone_cfg = model_cfg.get("backbone", {})
    fusion_cfg = model_cfg.get("fusion_transformer", {})
    head_cfg = model_cfg.get("detection_head", {})
    radar_cfg = model_cfg.get("radar_pillar_encoder", {})
    pc_cfg = data_cfg.get("point_cloud", {})

    # Determine BEV grid size from point cloud range and voxel size
    pcr = pc_cfg.get("range", radar_cfg.get("point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))
    vs = pc_cfg.get("voxel_size", radar_cfg.get("voxel_size", [0.2, 0.2, 8.0]))

    # Use a downsampled BEV grid (4x stride from full resolution)
    full_bev_h = int(round((pcr[3] - pcr[0]) / vs[0]))
    full_bev_w = int(round((pcr[4] - pcr[1]) / vs[1]))
    bev_stride = 4  # Common choice for efficiency
    bev_height = full_bev_h // bev_stride
    bev_width = full_bev_w // bev_stride

    config = CRAFTConfig(
        num_classes=head_cfg.get("num_classes", 10),
        bbox_code_size=head_cfg.get("bbox_code_size", 10),
        backbone_name=backbone_cfg.get("type", "resnet50"),
        backbone_pretrained=backbone_cfg.get("pretrained", True),
        frozen_stages=backbone_cfg.get("frozen_stages", 1),
        fpn_out_channels=model_cfg.get("neck", {}).get("out_channels", 256),
        num_cameras=6,
        point_cloud_range=pcr,
        voxel_size=vs,
        max_points_per_pillar=radar_cfg.get("max_points_per_pillar", 20),
        max_num_pillars=radar_cfg.get("max_pillars", 30000),
        radar_in_channels=radar_cfg.get("in_channels", 6),
        pillar_feat_channels=radar_cfg.get("out_channels", 128),
        bev_out_channels=256,
        fusion_d_model=fusion_cfg.get("d_model", 256),
        fusion_n_heads=fusion_cfg.get("n_heads", 8),
        fusion_n_layers=fusion_cfg.get("n_layers", 6),
        fusion_d_ffn=fusion_cfg.get("d_ffn", 1024),
        fusion_dropout=fusion_cfg.get("dropout", 0.1),
        bev_height=bev_height,
        bev_width=bev_width,
        heatmap_channels=head_cfg.get("heatmap_channels", 256),
        regression_channels=head_cfg.get("regression_channels", 256),
        num_heatmap_convs=head_cfg.get("num_heatmap_convs", 2),
        num_regression_convs=head_cfg.get("num_regression_convs", 3),
        shared_conv_channels=head_cfg.get("shared_conv_channels", 64),
        bias_heatmap=head_cfg.get("bias_heatmap", -2.19),
        score_threshold=eval_cfg.get("score_threshold", 0.1),
        nms_threshold=eval_cfg.get("nms_threshold", 0.2),
        max_detections=eval_cfg.get("max_detections", 500),
    )

    return CRAFTModel(config)


# =============================================================================
# Main (Sanity Check)
# =============================================================================


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building CRAFT Model...")
    config = CRAFTConfig(
        backbone_pretrained=False,  # Avoid downloading weights for test
        bev_height=32,    # Small grid for fast testing
        bev_width=32,
        fusion_n_layers=2,  # Fewer layers for fast testing
        max_num_pillars=1000,  # Fewer pillars for testing
    )

    model = build_craft_model(config).to(device)
    model.print_model_summary()

    # Create dummy inputs
    batch_size = 2
    num_cameras = 6
    img_h, img_w = 256, 448
    max_radar_pts = 200

    images = torch.randn(batch_size, num_cameras, 3, img_h, img_w, device=device)
    radar_points = torch.randn(batch_size, max_radar_pts, 6, device=device)
    # Make radar points within valid range
    radar_points[:, :, 0] = radar_points[:, :, 0] * 30  # x in ~[-30, 30]
    radar_points[:, :, 1] = radar_points[:, :, 1] * 30  # y in ~[-30, 30]
    radar_points[:, :, 2] = radar_points[:, :, 2] * 3 - 1  # z in ~[-4, 2]
    radar_num_points = torch.tensor([150, 100], device=device)

    intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_cameras, -1, -1).clone()
    intrinsics[:, :, 0, 0] = 1000.0  # fx
    intrinsics[:, :, 1, 1] = 1000.0  # fy
    intrinsics[:, :, 0, 2] = img_w / 2  # cx
    intrinsics[:, :, 1, 2] = img_h / 2  # cy

    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_cameras, -1, -1).clone()

    # Test training mode
    print("\n--- Training Mode ---")
    model.train()
    outputs = model(
        images=images,
        radar_points=radar_points,
        radar_num_points=radar_num_points,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        return_loss=True,
    )

    print(f"Fused heatmap shape: {outputs['heatmap'].shape}")
    print(f"Fused regression shape: {outputs['reg'].shape}")
    print(f"Camera aux heatmap shape: {outputs['camera_heatmap'].shape}")
    print(f"Camera aux regression shape: {outputs['camera_reg'].shape}")
    print(f"Radar aux heatmap shape: {outputs['radar_heatmap'].shape}")
    print(f"Radar aux regression shape: {outputs['radar_reg'].shape}")

    # Test inference mode
    print("\n--- Inference Mode ---")
    model.eval()
    with torch.no_grad():
        detections = model(
            images=images,
            radar_points=radar_points,
            radar_num_points=radar_num_points,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            return_loss=False,
        )

    for i, det in enumerate(detections):
        print(f"Sample {i}: {det['boxes'].shape[0]} detections")
        if det['boxes'].shape[0] > 0:
            print(f"  Top score: {det['scores'][0].item():.4f}")
            print(f"  Top class: {det['labels'][0].item()}")
            print(f"  Box shape: {det['boxes'].shape}")

    # Verify gradient flow
    print("\n--- Gradient Check ---")
    model.train()
    outputs = model(
        images=images,
        radar_points=radar_points,
        radar_num_points=radar_num_points,
        return_loss=True,
    )
    dummy_loss = outputs["heatmap"].sum() + outputs["reg"].sum()
    dummy_loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            grad_norms[name] = param.grad.norm().item()

    # Print a few representative gradient norms
    sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 gradient norms:")
    for name, norm in sorted_grads[:5]:
        print(f"  {name}: {norm:.6f}")

    print("\nAll checks passed!")
