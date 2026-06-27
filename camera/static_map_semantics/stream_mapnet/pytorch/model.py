"""
StreamMapNet - PyTorch Implementation

A temporal BEV (Bird's Eye View) network for online HD map construction.
Takes multi-camera images and produces vectorized HD map element predictions
(lane dividers, pedestrian crossings, road boundaries) using streaming
temporal fusion of BEV features with ego-motion compensation.

Architecture:
  Multi-camera images -> ResNet-50 backbone -> FPN neck -> LSS BEV Transform
  -> Temporal Fusion (ego-motion warping + cross-attention) -> Transformer Decoder
  -> Map Element Heads (classification + polyline regression)

Reference:
  Yuan et al., "StreamMapNet: Streaming Mapping Network for Vectorized Online
  HD Map Construction", WACV 2024.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# =============================================================================
# Backbone: ResNet-50 + FPN
# =============================================================================


class ResNetBackbone(nn.Module):
    """
    ResNet-50 backbone for multi-scale feature extraction.

    Extracts features from stages C3, C4, C5 (strides 8, 16, 32) and
    feeds them into an FPN for multi-scale aggregation.
    """

    def __init__(
        self,
        pretrained: bool = True,
        frozen_stages: int = 1,
        out_indices: Tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Load pretrained ResNet-50
        if pretrained:
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            base = resnet50(weights=None)

        # Split into stages
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1  # C2, stride 4,  256 channels
        self.layer2 = base.layer2  # C3, stride 8,  512 channels
        self.layer3 = base.layer3  # C4, stride 16, 1024 channels
        self.layer4 = base.layer4  # C5, stride 32, 2048 channels

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze batch norm and parameters in early stages."""
        if self.frozen_stages >= 0:
            for param in self.stem.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1:
            for param in self.layer1.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train to keep frozen BN in eval mode."""
        super().train(mode)
        if mode:
            # Keep frozen stages in eval mode for BN
            if self.frozen_stages >= 0:
                self.stem.eval()
                for m in self.stem.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
            if self.frozen_stages >= 1:
                self.layer1.eval()
                for m in self.layer1.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B*N_cams, 3, H, W) flattened camera images

        Returns:
            List of feature maps at selected stages.
            Default out_indices=(1,2,3) returns [C3, C4, C5] with channels
            [512, 1024, 2048].
        """
        stages = []
        x = self.stem(x)
        x = self.layer1(x)
        stages.append(x)       # index 0: C2
        x = self.layer2(x)
        stages.append(x)       # index 1: C3
        x = self.layer3(x)
        stages.append(x)       # index 2: C4
        x = self.layer4(x)
        stages.append(x)       # index 3: C5

        return [stages[i] for i in self.out_indices]


class FPN(nn.Module):
    """
    Feature Pyramid Network.

    Combines multi-scale backbone features with top-down pathway and
    lateral connections.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_outs: int = 3,
    ):
        super().__init__()
        self.num_outs = num_outs

        # Lateral 1x1 convolutions
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
            )

        # Top-down 3x3 convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels:
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

        # Extra output levels via stride-2 convolution
        self.extra_convs = nn.ModuleList()
        for _ in range(num_outs - len(in_channels)):
            self.extra_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            inputs: List of backbone feature maps [C3, C4, C5]

        Returns:
            List of FPN feature maps, each with out_channels channels.
        """
        # Lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, inputs)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            h, w = laterals[i].shape[2:]
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=(h, w), mode="bilinear", align_corners=False
            )

        # Apply 3x3 conv
        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Extra output levels
        extra_in = outs[-1]
        for conv in self.extra_convs:
            extra_in = conv(F.relu(extra_in))
            outs.append(extra_in)

        return outs[: self.num_outs]


# =============================================================================
# BEV Transform: Lift-Splat-Shoot (LSS)
# =============================================================================


class DepthNet(nn.Module):
    """Predicts per-pixel discrete depth distribution."""

    def __init__(self, in_channels: int, mid_channels: int, num_depth_bins: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_depth_bins, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*N, C, H, W) image features

        Returns:
            depth_prob: (B*N, D, H, W) depth probability distribution
        """
        depth_logits = self.net(x)
        depth_prob = depth_logits.softmax(dim=1)
        return depth_prob


class BEVEncoder(nn.Module):
    """Refines raw BEV features after voxel pooling."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            layers.extend([
                nn.Conv2d(ch_in, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSSBEVTransform(nn.Module):
    """
    Lift-Splat-Shoot BEV transformation.

    Lifts 2D image features into 3D frustum using predicted depth distributions,
    then splats into a BEV plane via voxel pooling.
    """

    def __init__(
        self,
        in_channels: int = 256,
        bev_channels: int = 256,
        num_depth_bins: int = 59,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
        bev_x_range: Tuple[float, float] = (-30.0, 30.0),
        bev_y_range: Tuple[float, float] = (-15.0, 15.0),
        bev_resolution: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.bev_channels = bev_channels
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.bev_resolution = bev_resolution

        # BEV grid dimensions
        self.bev_w = int((bev_x_range[1] - bev_x_range[0]) / bev_resolution)
        self.bev_h = int((bev_y_range[1] - bev_y_range[0]) / bev_resolution)

        # Depth prediction network
        self.depth_net = DepthNet(in_channels, in_channels, num_depth_bins)

        # Feature reduction to BEV channels
        self.feat_reduce = nn.Conv2d(in_channels, bev_channels, kernel_size=1)

        # BEV encoder
        self.bev_encoder = BEVEncoder(bev_channels, bev_channels, num_layers=2)

        # Register depth bins as buffer
        depth_bins = torch.linspace(depth_min, depth_max, num_depth_bins)
        self.register_buffer("depth_bins", depth_bins)

    def _create_frustum(
        self, h_feat: int, w_feat: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create frustum grid in normalized image coordinates.

        Returns:
            frustum: (D, H_feat, W_feat, 3) with (norm_u, norm_v, depth)
        """
        D = self.num_depth_bins
        us = torch.linspace(0.0, 1.0, w_feat, device=device)
        vs = torch.linspace(0.0, 1.0, h_feat, device=device)
        grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")  # (H, W)

        # Expand depth: (D, H, W)
        grid_u = grid_u.unsqueeze(0).expand(D, -1, -1)
        grid_v = grid_v.unsqueeze(0).expand(D, -1, -1)
        depth_grid = self.depth_bins.view(D, 1, 1).expand(-1, h_feat, w_feat)

        frustum = torch.stack([grid_u, grid_v, depth_grid], dim=-1)  # (D, H, W, 3)
        return frustum

    def _frustum_to_ego(
        self,
        frustum: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """
        Transform frustum points from image space to ego frame.

        Args:
            frustum: (D, H_feat, W_feat, 3) with (norm_u, norm_v, depth)
            intrinsics: (B, N, 3, 3)
            extrinsics: (B, N, 4, 4) camera-to-ego

        Returns:
            pts_ego: (B, N, D, H_feat, W_feat, 3)
        """
        B, N = intrinsics.shape[:2]
        D, H_f, W_f, _ = frustum.shape

        # De-normalize to pixel coordinates
        pixel_u = frustum[..., 0] * img_w  # (D, H, W)
        pixel_v = frustum[..., 1] * img_h
        depth = frustum[..., 2]

        # Unproject: X = (u - cx) * d / fx, Y = (v - cy) * d / fy, Z = d
        fx = intrinsics[:, :, 0, 0].view(B, N, 1, 1, 1)
        fy = intrinsics[:, :, 1, 1].view(B, N, 1, 1, 1)
        cx = intrinsics[:, :, 0, 2].view(B, N, 1, 1, 1)
        cy = intrinsics[:, :, 1, 2].view(B, N, 1, 1, 1)

        pixel_u = pixel_u.view(1, 1, D, H_f, W_f)
        pixel_v = pixel_v.view(1, 1, D, H_f, W_f)
        depth = depth.view(1, 1, D, H_f, W_f)

        x_cam = (pixel_u - cx) * depth / fx
        y_cam = (pixel_v - cy) * depth / fy
        z_cam = depth

        pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (B, N, D, H, W, 3)

        # Transform to ego frame
        rot = extrinsics[:, :, :3, :3]    # (B, N, 3, 3)
        trans = extrinsics[:, :, :3, 3]   # (B, N, 3)

        pts_flat = pts_cam.reshape(B, N, -1, 3)  # (B, N, D*H*W, 3)
        # Rotate: (B, N, D*H*W, 3) @ (B, N, 3, 3)^T
        pts_ego = torch.einsum("bnpc,bnrc->bnpr", pts_flat, rot)
        pts_ego = pts_ego + trans.unsqueeze(2)  # (B, N, D*H*W, 3)
        pts_ego = pts_ego.reshape(B, N, D, H_f, W_f, 3)

        return pts_ego

    def _voxel_pool(
        self,
        features: torch.Tensor,
        depth_prob: torch.Tensor,
        pts_ego: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Pool lifted 3D features into BEV grid.

        Args:
            features: (B*N, C, H_feat, W_feat)
            depth_prob: (B*N, D, H_feat, W_feat)
            pts_ego: (B, N, D, H_feat, W_feat, 3)
            batch_size: B

        Returns:
            bev: (B, C, bev_h, bev_w)
        """
        B = batch_size
        N = pts_ego.shape[1]
        D, H_f, W_f = pts_ego.shape[2:5]
        C = features.shape[1]

        # Reshape features: (B, N, C, H, W)
        features = features.view(B, N, C, H_f, W_f)

        # Create lifted features: outer product of depth_prob and features
        # depth_prob: (B, N, D, H, W) -> (B, N, D, H, W, 1)
        depth_prob = depth_prob.view(B, N, D, H_f, W_f)
        # features: (B, N, C, H, W) -> (B, N, 1, H, W, C)
        feat_expanded = features.permute(0, 1, 3, 4, 2).unsqueeze(2)  # (B, N, 1, H, W, C)
        depth_expanded = depth_prob.unsqueeze(-1)  # (B, N, D, H, W, 1)
        lifted = depth_expanded * feat_expanded  # (B, N, D, H, W, C)

        # Compute BEV grid indices
        x_ego = pts_ego[..., 0]  # (B, N, D, H, W)
        y_ego = pts_ego[..., 1]

        bev_x_idx = ((x_ego - self.bev_x_range[0]) / self.bev_resolution).long()
        bev_y_idx = ((y_ego - self.bev_y_range[0]) / self.bev_resolution).long()

        # Valid mask
        valid = (
            (bev_x_idx >= 0) & (bev_x_idx < self.bev_w)
            & (bev_y_idx >= 0) & (bev_y_idx < self.bev_h)
        )

        # Initialize BEV tensor
        bev = torch.zeros(B, self.bev_h, self.bev_w, C, device=features.device)

        # Scatter add valid points into BEV grid
        for b in range(B):
            mask = valid[b]  # (N, D, H, W)
            bx = bev_x_idx[b][mask]  # (num_valid,)
            by = bev_y_idx[b][mask]
            feats = lifted[b][mask]   # (num_valid, C)

            # Flatten BEV index
            flat_idx = by * self.bev_w + bx  # (num_valid,)
            bev_flat = bev[b].view(-1, C)  # (bev_h * bev_w, C)
            bev_flat.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, C), feats)

        # Reshape to (B, C, bev_h, bev_w)
        bev = bev.permute(0, 3, 1, 2)
        return bev

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        img_h: int = 256,
        img_w: int = 704,
    ) -> torch.Tensor:
        """
        Args:
            features: (B*N, C, H_feat, W_feat) image features from FPN
            intrinsics: (B, N, 3, 3)
            extrinsics: (B, N, 4, 4)
            img_h: original image height
            img_w: original image width

        Returns:
            bev: (B, bev_channels, bev_h, bev_w) BEV feature map
        """
        B = intrinsics.shape[0]
        N = intrinsics.shape[1]
        H_f, W_f = features.shape[2:]

        # Predict depth distribution
        depth_prob = self.depth_net(features)  # (B*N, D, H, W)

        # Reduce feature channels
        feat_reduced = self.feat_reduce(features)  # (B*N, bev_C, H, W)

        # Create frustum and project to ego
        frustum = self._create_frustum(H_f, W_f, features.device)
        pts_ego = self._frustum_to_ego(frustum, intrinsics, extrinsics, img_h, img_w)

        # Voxel pooling
        bev = self._voxel_pool(feat_reduced, depth_prob, pts_ego, B)

        # Refine BEV features
        bev = self.bev_encoder(bev)

        return bev


# =============================================================================
# Temporal Fusion Module
# =============================================================================


class TemporalFusion(nn.Module):
    """
    Temporal fusion module for streaming BEV feature propagation.

    Warps previous hidden state to the current coordinate frame using
    ego-motion and fuses with current BEV features via cross-attention.
    """

    def __init__(
        self,
        bev_channels: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        bev_x_range: Tuple[float, float] = (-30.0, 30.0),
        bev_y_range: Tuple[float, float] = (-15.0, 15.0),
        bev_resolution: float = 0.3,
    ):
        super().__init__()
        self.bev_channels = bev_channels
        self.num_heads = num_heads
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.bev_resolution = bev_resolution
        self.bev_w = int((bev_x_range[1] - bev_x_range[0]) / bev_resolution)
        self.bev_h = int((bev_y_range[1] - bev_y_range[0]) / bev_resolution)

        # Cross-attention: current BEV queries attend to warped history
        self.query_proj = nn.Linear(bev_channels, bev_channels)
        self.key_proj = nn.Linear(bev_channels, bev_channels)
        self.value_proj = nn.Linear(bev_channels, bev_channels)
        self.out_proj = nn.Linear(bev_channels, bev_channels)

        self.norm1 = nn.LayerNorm(bev_channels)
        self.norm2 = nn.LayerNorm(bev_channels)
        self.dropout1 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(bev_channels, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, bev_channels),
            nn.Dropout(dropout),
        )

        # Temporal hidden state (managed externally by the model)
        self._prev_bev: Optional[torch.Tensor] = None

    def reset_state(self):
        """Reset temporal hidden state. Call at sequence boundaries."""
        self._prev_bev = None

    @property
    def has_history(self) -> bool:
        return self._prev_bev is not None

    def _warp_bev(
        self, prev_bev: torch.Tensor, ego_motion: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp previous BEV features to the current coordinate frame.

        Args:
            prev_bev: (B, C, H, W) BEV features from previous frame
            ego_motion: (B, 4, 4) transformation from previous to current frame

        Returns:
            warped: (B, C, H, W) warped BEV features in current coordinates
        """
        B, C, H, W = prev_bev.shape
        device = prev_bev.device

        # Create BEV coordinate grid in meters (current frame)
        xs = torch.linspace(
            self.bev_x_range[0] + self.bev_resolution / 2,
            self.bev_x_range[1] - self.bev_resolution / 2,
            W, device=device,
        )
        ys = torch.linspace(
            self.bev_y_range[0] + self.bev_resolution / 2,
            self.bev_y_range[1] - self.bev_resolution / 2,
            H, device=device,
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

        # Create homogeneous coordinates: (H*W, 4)
        ones = torch.ones_like(grid_x.reshape(-1))
        zeros = torch.zeros_like(ones)
        pts_current = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), zeros, ones], dim=-1
        )  # (H*W, 4)

        # Transform current grid points to previous frame
        # ego_motion: prev -> current, so we need inverse for current -> prev
        ego_inv = torch.inverse(ego_motion)  # (B, 4, 4)

        pts_current_batch = pts_current.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 4)
        pts_prev = torch.einsum("bij,bnj->bni", ego_inv, pts_current_batch)  # (B, H*W, 4)

        # Normalize to [-1, 1] for grid_sample
        norm_x = (pts_prev[:, :, 0] - self.bev_x_range[0]) / (
            self.bev_x_range[1] - self.bev_x_range[0]
        ) * 2.0 - 1.0
        norm_y = (pts_prev[:, :, 1] - self.bev_y_range[0]) / (
            self.bev_y_range[1] - self.bev_y_range[0]
        ) * 2.0 - 1.0

        # Reshape to (B, H, W, 2) for grid_sample
        sample_grid = torch.stack([norm_x, norm_y], dim=-1).view(B, H, W, 2)

        # Warp using grid_sample
        warped = F.grid_sample(
            prev_bev, sample_grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        return warped

    def _cross_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-head cross-attention.

        Args:
            query: (B, L_q, C)
            key: (B, L_k, C)
            value: (B, L_k, C)

        Returns:
            out: (B, L_q, C)
        """
        B, L_q, C = query.shape
        L_k = key.shape[1]
        head_dim = C // self.num_heads

        Q = self.query_proj(query).view(B, L_q, self.num_heads, head_dim).transpose(1, 2)
        K = self.key_proj(key).view(B, L_k, self.num_heads, head_dim).transpose(1, 2)
        V = self.value_proj(value).view(B, L_k, self.num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, heads, L_q, L_k)
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, V)  # (B, heads, L_q, head_dim)
        out = out.transpose(1, 2).reshape(B, L_q, C)
        out = self.out_proj(out)
        return out

    def forward(
        self,
        current_bev: torch.Tensor,
        ego_motion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse current BEV features with warped historical hidden state.

        Args:
            current_bev: (B, C, H, W) current frame BEV features
            ego_motion: (B, 4, 4) ego motion from previous to current frame.
                        Required if history exists.

        Returns:
            fused_bev: (B, C, H, W) temporally fused BEV features
        """
        if self._prev_bev is None:
            # First frame: no history, output current features and store
            self._prev_bev = current_bev.detach()
            return current_bev

        B, C, H, W = current_bev.shape

        # Warp previous hidden state to current frame
        if ego_motion is not None:
            warped = self._warp_bev(self._prev_bev, ego_motion)
        else:
            warped = self._prev_bev

        # Flatten to sequence for attention: (B, H*W, C)
        current_flat = current_bev.permute(0, 2, 3, 1).reshape(B, H * W, C)
        warped_flat = warped.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Cross-attention: current queries attend to warped history
        attn_out = self._cross_attention(current_flat, warped_flat, warped_flat)
        fused_flat = self.norm1(current_flat + self.dropout1(attn_out))

        # FFN with residual
        fused_flat = self.norm2(fused_flat + self.ffn(fused_flat))

        # Reshape back to spatial
        fused_bev = fused_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Update hidden state
        self._prev_bev = fused_bev.detach()

        return fused_bev


# =============================================================================
# Map Element Decoder (Transformer)
# =============================================================================


class MapDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention to BEV features
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, N_q, d_model) map element queries
            memory: (B, H*W, d_model) flattened BEV features

        Returns:
            queries: (B, N_q, d_model) updated queries
        """
        # Self-attention
        q = self.norm1(queries)
        q2, _ = self.self_attn(q, q, q)
        queries = queries + self.dropout1(q2)

        # Cross-attention to BEV
        q = self.norm2(queries)
        q2, _ = self.cross_attn(q, memory, memory)
        queries = queries + self.dropout2(q2)

        # FFN
        q = self.norm3(queries)
        queries = queries + self.dropout3(self.ffn(q))

        return queries


class MapDecoder(nn.Module):
    """
    Transformer decoder for vectorized map element prediction.

    Uses learnable queries to attend to BEV features and iteratively
    refine map element predictions.
    """

    def __init__(
        self,
        bev_channels: int = 256,
        num_queries: int = 150,
        num_decoder_layers: int = 6,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        num_classes: int = 3,
        num_points: int = 20,
        auxiliary_loss: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_points = num_points
        self.auxiliary_loss = auxiliary_loss

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_pos = nn.Embedding(num_queries, d_model)

        # BEV feature projection
        self.bev_proj = nn.Conv2d(bev_channels, d_model, kernel_size=1)

        # Learnable BEV positional encoding
        # Will be initialized properly in first forward pass
        self.bev_pos_embed = None
        self._bev_h = None
        self._bev_w = None

        # Decoder layers
        self.layers = nn.ModuleList([
            MapDecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Prediction heads (shared across layers for auxiliary loss)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes + 1),  # +1 for no-object
        )

        self.pts_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_points * 2),
        )

    def _get_bev_pos(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get or create BEV positional encoding."""
        if self.bev_pos_embed is None or self._bev_h != H or self._bev_w != W:
            self._bev_h = H
            self._bev_w = W
            self.bev_pos_embed = nn.Parameter(
                torch.randn(1, H * W, self.d_model, device=device) * 0.02
            )
        return self.bev_pos_embed

    def forward(
        self, bev_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_features: (B, C, H, W) BEV feature map

        Returns:
            dict with:
                'pred_logits': (B, N_queries, num_classes+1)
                'pred_points': (B, N_queries, num_points, 2)
                'aux_outputs': list of dicts from intermediate layers (if auxiliary_loss)
        """
        B, C, H, W = bev_features.shape

        # Project BEV features
        bev_proj = self.bev_proj(bev_features)  # (B, d_model, H, W)
        memory = bev_proj.flatten(2).permute(0, 2, 1)  # (B, H*W, d_model)

        # Add positional encoding
        bev_pos = self._get_bev_pos(H, W, bev_features.device)
        memory = memory + bev_pos

        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = queries + self.query_pos.weight.unsqueeze(0)

        # Apply decoder layers with optional auxiliary outputs
        aux_outputs = []
        for i, layer in enumerate(self.layers):
            queries = layer(queries, memory)

            if self.auxiliary_loss and i < self.num_decoder_layers - 1:
                # Intermediate predictions
                q_normed = self.final_norm(queries)
                aux_cls = self.cls_head(q_normed)
                aux_pts_raw = self.pts_head(q_normed)
                aux_pts = aux_pts_raw.view(B, self.num_queries, self.num_points, 2).sigmoid()
                aux_outputs.append({
                    "pred_logits": aux_cls,
                    "pred_points": aux_pts,
                })

        # Final predictions
        queries = self.final_norm(queries)
        pred_logits = self.cls_head(queries)  # (B, N_q, num_classes+1)
        pred_pts_raw = self.pts_head(queries)  # (B, N_q, num_points*2)
        pred_points = pred_pts_raw.view(B, self.num_queries, self.num_points, 2).sigmoid()

        outputs = {
            "pred_logits": pred_logits,
            "pred_points": pred_points,
        }
        if self.auxiliary_loss:
            outputs["aux_outputs"] = aux_outputs

        return outputs


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
            lane_1_left_pt0, ..., (num_lanes × 2 lines × points_per_line)
        [num_lane_queries, total_queries): Other line queries organized as
            line_0_pt0, ..., line_0_pt19,
            line_1_pt0, ..., (num_other_lines × points_per_line)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_lanes: int = 25,
        points_per_line: int = 20,
        num_other_lines: int = 0,
    ) -> None:
        """Initialize hierarchical lane positional embeddings.

        Args:
            embed_dim: Embedding dimension (must match decoder d_model).
            num_lanes: Number of lanes (each has left + right boundary).
            points_per_line: Number of points sampled per line.
            num_other_lines: Number of additional non-lane lines (e.g.,
                road boundaries, crosswalks).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_lanes = num_lanes
        self.points_per_line = points_per_line
        self.num_other_lines = num_other_lines

        self.num_lane_queries = num_lanes * 2 * points_per_line
        self.num_other_queries = num_other_lines * points_per_line
        self.total_queries = self.num_lane_queries + self.num_other_queries

        # Lane-level embedding: which lane (0..num_lanes-1) or other-line group
        self.lane_embedding = nn.Embedding(num_lanes + num_other_lines, embed_dim)

        # Line-type embedding: 0=left boundary, 1=right boundary, 2=other
        self.line_type_embedding = nn.Embedding(3, embed_dim)

        # Point-position embedding: ordinal position along the line (0..points-1)
        self.point_embedding = nn.Embedding(points_per_line, embed_dim)

        # Learnable content queries (one per structural slot)
        self.content_embedding = nn.Embedding(self.total_queries, embed_dim)

        self._init_weights()
        self._build_index_tables()

    def _init_weights(self) -> None:
        nn.init.normal_(self.lane_embedding.weight, std=0.02)
        nn.init.normal_(self.line_type_embedding.weight, std=0.02)
        nn.init.normal_(self.point_embedding.weight, std=0.02)
        nn.init.normal_(self.content_embedding.weight, std=0.02)

    def _build_index_tables(self) -> None:
        """Pre-compute index tensors for efficient lookup."""
        lane_ids = []
        line_type_ids = []
        point_ids = []

        # Lane queries: num_lanes × 2 lines × points_per_line
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

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute hierarchical positional and content embeddings.

        Returns:
            Tuple of:
                - pos_embed: (total_queries, embed_dim) positional embeddings
                - content_embed: (total_queries, embed_dim) content queries
        """
        pos_embed = (
            self.lane_embedding(self.lane_ids)
            + self.line_type_embedding(self.line_type_ids)
            + self.point_embedding(self.point_ids)
        )
        content_embed = self.content_embedding.weight
        return pos_embed, content_embed

    def get_lane_mask(self) -> torch.Tensor:
        """Return boolean mask identifying lane queries vs other-line queries.

        Returns:
            (total_queries,) boolean tensor, True for lane queries.
        """
        mask = torch.zeros(self.total_queries, dtype=torch.bool)
        mask[: self.num_lane_queries] = True
        return mask


class HierarchicalMapDecoder(nn.Module):
    """Map decoder with hierarchical lane-structured positional embeddings.

    Extends the standard MapDecoder by replacing flat query embeddings with
    hierarchical positional embeddings that encode the lane -> line -> point
    structure. This gives the transformer explicit knowledge of which query
    corresponds to which point on which boundary line of which lane.

    Output organization:
        - 25 lanes × 2 boundary lines × 20 points = 1000 lane queries
        - Additional non-lane polylines × 20 points each
    """

    def __init__(
        self,
        bev_channels: int = 256,
        num_decoder_layers: int = 6,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        num_lanes: int = 25,
        points_per_line: int = 20,
        num_other_lines: int = 0,
        auxiliary_loss: bool = True,
    ):
        """Initialize hierarchical map decoder.

        Args:
            bev_channels: Input BEV feature channels.
            num_decoder_layers: Number of transformer decoder layers.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            ffn_dim: FFN hidden dimension.
            dropout: Dropout rate.
            num_lanes: Number of lanes (each with left+right boundary).
            points_per_line: Points per line (default 20).
            num_other_lines: Non-lane polylines.
            auxiliary_loss: Whether to compute intermediate layer predictions.
        """
        super().__init__()
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.num_lanes = num_lanes
        self.points_per_line = points_per_line
        self.num_other_lines = num_other_lines
        self.auxiliary_loss = auxiliary_loss

        # Hierarchical positional embeddings
        self.hierarchical_pos = HierarchicalLanePositionalEmbedding(
            embed_dim=d_model,
            num_lanes=num_lanes,
            points_per_line=points_per_line,
            num_other_lines=num_other_lines,
        )

        self.total_queries = self.hierarchical_pos.total_queries
        self.num_total_lines = num_lanes * 2 + num_other_lines

        # BEV feature projection
        self.bev_proj = nn.Conv2d(bev_channels, d_model, kernel_size=1)

        # Learnable BEV positional encoding
        self.bev_pos_embed = None
        self._bev_h = None
        self._bev_w = None

        # Decoder layers
        self.layers = nn.ModuleList([
            MapDecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Per-point 2D coordinate prediction
        self.pts_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
        )

        # Per-line classification (lane exists or not)
        # Classes: lane_divider, road_boundary, crosswalk, no-object
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4),
        )

    def _get_bev_pos(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get or create BEV positional encoding."""
        if self.bev_pos_embed is None or self._bev_h != H or self._bev_w != W:
            self._bev_h = H
            self._bev_w = W
            self.bev_pos_embed = nn.Parameter(
                torch.randn(1, H * W, self.d_model, device=device) * 0.02
            )
        return self.bev_pos_embed

    def forward(
        self, bev_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through hierarchical map decoder.

        Args:
            bev_features: (B, C, H, W) BEV feature map.

        Returns:
            Dict with:
                'pred_points': (B, num_total_lines, points_per_line, 2)
                    predicted BEV coordinates per line
                'pred_logits': (B, num_total_lines, 4) per-line class scores
                'aux_outputs': list of intermediate predictions (if enabled)
        """
        B, C, H, W = bev_features.shape

        # Project BEV features
        bev_proj = self.bev_proj(bev_features)
        memory = bev_proj.flatten(2).permute(0, 2, 1)  # (B, H*W, d_model)

        # Add BEV positional encoding
        bev_pos = self._get_bev_pos(H, W, bev_features.device)
        memory = memory + bev_pos

        # Get hierarchical query embeddings
        query_pos, query_content = self.hierarchical_pos()

        # Expand for batch and combine
        queries = query_content.unsqueeze(0).expand(B, -1, -1)
        queries = queries + query_pos.unsqueeze(0)

        # Apply decoder layers
        aux_outputs = []
        for i, layer in enumerate(self.layers):
            queries = layer(queries, memory)

            if self.auxiliary_loss and i < self.num_decoder_layers - 1:
                q_normed = self.final_norm(queries)
                aux_pts, aux_cls = self._predict(q_normed, B)
                aux_outputs.append({
                    "pred_points": aux_pts,
                    "pred_logits": aux_cls,
                })

        # Final predictions
        queries = self.final_norm(queries)
        pred_points, pred_logits = self._predict(queries, B)

        outputs = {
            "pred_points": pred_points,
            "pred_logits": pred_logits,
        }
        if self.auxiliary_loss:
            outputs["aux_outputs"] = aux_outputs

        return outputs

    def _predict(
        self, queries: torch.Tensor, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute point coordinates and line classifications.

        Args:
            queries: (B, total_queries, d_model) decoder output.
            batch_size: Batch size.

        Returns:
            pred_points: (B, num_total_lines, points_per_line, 2)
            pred_logits: (B, num_total_lines, 4)
        """
        # Per-point coordinates
        raw_pts = self.pts_head(queries).sigmoid()  # (B, total_queries, 2)
        pred_points = raw_pts.view(
            batch_size, self.num_total_lines, self.points_per_line, 2
        )

        # Per-line classification: pool point features per line
        line_features = queries.view(
            batch_size, self.num_total_lines, self.points_per_line, self.d_model
        ).mean(dim=2)  # (B, num_lines, d_model)

        pred_logits = self.cls_head(line_features)  # (B, num_lines, 4)

        return pred_points, pred_logits


# =============================================================================
# StreamMapNet: Full Model
# =============================================================================


class StreamMapNet(nn.Module):
    """
    StreamMapNet: Streaming temporal BEV network for online HD map construction.

    Composes all submodules into a complete end-to-end pipeline:
      Backbone -> FPN -> LSS BEV -> Temporal Fusion -> Decoder -> Heads

    The model maintains a temporal hidden state across forward calls to enable
    streaming inference without re-processing historical frames.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Model configuration dictionary. Expected keys match
                    stream_mapnet_base.yaml structure.
        """
        super().__init__()
        self.config = config

        # Parse config
        backbone_cfg = config.get("model", {}).get("backbone", {})
        neck_cfg = config.get("model", {}).get("neck", {})
        bev_cfg = config.get("model", {}).get("bev_transform", {})
        temporal_cfg = config.get("model", {}).get("temporal_fusion", {})
        decoder_cfg = config.get("model", {}).get("map_decoder", {})
        data_cfg = config.get("data", {})

        # --- Backbone ---
        pretrained = backbone_cfg.get("pretrained", "torchvision://resnet50") != ""
        frozen_stages = backbone_cfg.get("frozen_stages", 1)
        out_indices = tuple(backbone_cfg.get("out_indices", [1, 2, 3]))
        self.backbone = ResNetBackbone(
            pretrained=pretrained,
            frozen_stages=frozen_stages,
            out_indices=out_indices,
        )

        # --- FPN Neck ---
        in_channels = neck_cfg.get("in_channels", [512, 1024, 2048])
        fpn_out_channels = neck_cfg.get("out_channels", 256)
        fpn_num_outs = neck_cfg.get("num_outs", 3)
        self.neck = FPN(
            in_channels=in_channels,
            out_channels=fpn_out_channels,
            num_outs=fpn_num_outs,
        )

        # --- BEV Transform (LSS) ---
        bev_grid = bev_cfg.get("bev_grid", {})
        depth_cfg = bev_cfg.get("depth_cfg", {})
        bev_x_range = tuple(data_cfg.get("bev_range", {}).get("x", [-30.0, 30.0]))
        bev_y_range = tuple(data_cfg.get("bev_range", {}).get("y", [-15.0, 15.0]))
        bev_resolution = bev_grid.get("resolution", 0.3)

        self.bev_transform = LSSBEVTransform(
            in_channels=fpn_out_channels,
            bev_channels=fpn_out_channels,
            num_depth_bins=depth_cfg.get("num_depth_bins", 59),
            depth_min=depth_cfg.get("min_depth", 1.0),
            depth_max=depth_cfg.get("max_depth", 60.0),
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
            bev_resolution=bev_resolution,
        )

        # --- Temporal Fusion ---
        attn_cfg = temporal_cfg.get("attention", {})
        ffn_cfg = temporal_cfg.get("ffn", {})
        self.temporal_fusion = TemporalFusion(
            bev_channels=fpn_out_channels,
            num_heads=attn_cfg.get("num_heads", 8),
            ffn_dim=ffn_cfg.get("hidden_dim", 512),
            dropout=attn_cfg.get("dropout", 0.1),
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
            bev_resolution=bev_resolution,
        )

        # --- Map Decoder ---
        num_classes = data_cfg.get("num_classes", 3)
        num_points = decoder_cfg.get("num_points_per_query", 20)
        num_queries = decoder_cfg.get("num_queries", 150)
        auxiliary_loss = config.get("loss", {}).get("auxiliary_loss", True)

        self.decoder = MapDecoder(
            bev_channels=fpn_out_channels,
            num_queries=num_queries,
            num_decoder_layers=decoder_cfg.get("num_decoder_layers", 6),
            d_model=decoder_cfg.get("hidden_dim", 256),
            num_heads=decoder_cfg.get("num_heads", 8),
            ffn_dim=decoder_cfg.get("ffn_dim", 512),
            dropout=decoder_cfg.get("dropout", 0.1),
            num_classes=num_classes,
            num_points=num_points,
            auxiliary_loss=auxiliary_loss,
        )

        # Store data config
        self.img_h, self.img_w = data_cfg.get("img_size", [256, 704])
        self.num_cameras = data_cfg.get("num_cameras", 6)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for non-pretrained layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reset_temporal_state(self):
        """
        Reset the temporal hidden state.

        Must be called at sequence boundaries (start of new driving scene)
        to prevent stale history from corrupting predictions.
        """
        self.temporal_fusion.reset_state()

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        ego_motion: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass of StreamMapNet.

        Args:
            images: (B, N_cams, 3, H, W) multi-camera images
            intrinsics: (B, N_cams, 3, 3) camera intrinsic matrices
            extrinsics: (B, N_cams, 4, 4) camera-to-ego extrinsic matrices
            ego_motion: (B, 4, 4) ego motion from previous to current frame.
                        None for the first frame in a sequence.

        Returns:
            dict with:
                'pred_logits': (B, N_queries, num_classes+1) classification logits
                'pred_points': (B, N_queries, num_points, 2) polyline point predictions
                'aux_outputs': list of intermediate layer predictions (training only)
        """
        B, N = images.shape[:2]

        # Step 1: Extract multi-scale features via backbone
        imgs_flat = images.flatten(0, 1)  # (B*N, 3, H, W)
        backbone_feats = self.backbone(imgs_flat)  # List of [C3, C4, C5]

        # Step 2: FPN for multi-scale aggregation
        fpn_feats = self.neck(backbone_feats)

        # Use primary FPN level (first) for BEV transform
        selected_feat = fpn_feats[0]  # (B*N, C, H/8, W/8)

        # Step 3: LSS BEV Transform
        bev_features = self.bev_transform(
            selected_feat, intrinsics, extrinsics,
            img_h=self.img_h, img_w=self.img_w,
        )  # (B, C, bev_h, bev_w)

        # Step 4: Temporal Fusion (streaming propagation)
        fused_bev = self.temporal_fusion(bev_features, ego_motion)

        # Step 5: Transformer Decoder + Prediction Heads
        outputs = self.decoder(fused_bev)

        return outputs

    @torch.no_grad()
    def inference(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        ego_motion: Optional[torch.Tensor] = None,
        score_threshold: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode forward pass with post-processing.

        Args:
            images: (B, N_cams, 3, H, W)
            intrinsics: (B, N_cams, 3, 3)
            extrinsics: (B, N_cams, 4, 4)
            ego_motion: (B, 4, 4) or None
            score_threshold: minimum confidence score for predictions

        Returns:
            dict with:
                'scores': (B, N_keep) confidence scores
                'labels': (B, N_keep) predicted class indices
                'points': (B, N_keep, num_points, 2) polyline points
        """
        self.eval()
        outputs = self.forward(images, intrinsics, extrinsics, ego_motion)

        # Post-process: apply softmax and threshold
        logits = outputs["pred_logits"]  # (B, N_q, num_classes+1)
        points = outputs["pred_points"]  # (B, N_q, num_points, 2)

        # Get class probabilities (exclude background class)
        probs = logits.softmax(dim=-1)  # (B, N_q, num_classes+1)
        # Score is max non-background probability
        scores, labels = probs[:, :, :-1].max(dim=-1)  # (B, N_q)

        # Filter by threshold
        results = []
        for b in range(images.shape[0]):
            mask = scores[b] > score_threshold
            results.append({
                "scores": scores[b][mask],
                "labels": labels[b][mask],
                "points": points[b][mask],
            })

        return results


# =============================================================================
# Factory function
# =============================================================================


def build_stream_mapnet(config: dict) -> StreamMapNet:
    """Build StreamMapNet model from configuration dictionary."""
    model = StreamMapNet(config)
    return model


# =============================================================================
# Main: Shape Verification
# =============================================================================

if __name__ == "__main__":
    import yaml

    print("=" * 70)
    print("StreamMapNet PyTorch - Shape Verification")
    print("=" * 70)

    # Minimal config for testing
    config = {
        "model": {
            "backbone": {
                "pretrained": "",
                "frozen_stages": 1,
                "out_indices": [1, 2, 3],
            },
            "neck": {
                "in_channels": [512, 1024, 2048],
                "out_channels": 256,
                "num_outs": 3,
            },
            "bev_transform": {
                "depth_cfg": {"min_depth": 1.0, "max_depth": 60.0, "num_depth_bins": 59},
                "bev_grid": {"x_size": 200, "y_size": 100, "resolution": 0.3},
            },
            "temporal_fusion": {
                "attention": {"num_heads": 8, "dropout": 0.1},
                "ffn": {"hidden_dim": 512},
            },
            "map_decoder": {
                "num_queries": 150,
                "num_decoder_layers": 6,
                "hidden_dim": 256,
                "num_heads": 8,
                "ffn_dim": 512,
                "dropout": 0.1,
                "num_points_per_query": 20,
            },
        },
        "data": {
            "img_size": [256, 704],
            "num_cameras": 6,
            "num_classes": 3,
            "bev_range": {"x": [-30.0, 30.0], "y": [-15.0, 15.0]},
        },
        "loss": {"auxiliary_loss": True},
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Build model (no pretrained weights for speed)
    model = build_stream_mapnet(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dummy inputs
    B, N = 2, 6
    H, W = 256, 704
    images = torch.randn(B, N, 3, H, W, device=device)
    intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    intrinsics[:, :, 0, 0] = 1260.0
    intrinsics[:, :, 1, 1] = 1260.0
    intrinsics[:, :, 0, 2] = W / 2
    intrinsics[:, :, 1, 2] = H / 2
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    ego_motion = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    ego_motion[:, 0, 3] = 0.5  # 0.5m forward

    print(f"\nInput shapes:")
    print(f"  images:     {images.shape}")
    print(f"  intrinsics: {intrinsics.shape}")
    print(f"  extrinsics: {extrinsics.shape}")
    print(f"  ego_motion: {ego_motion.shape}")

    # Forward pass 1: no temporal history
    model.eval()
    model.reset_temporal_state()
    print("\n--- Forward pass 1 (no history) ---")
    with torch.no_grad():
        out1 = model(images, intrinsics, extrinsics)
    print(f"  pred_logits: {out1['pred_logits'].shape}")
    print(f"  pred_points: {out1['pred_points'].shape}")
    assert out1["pred_logits"].shape == (B, 150, 4)
    assert out1["pred_points"].shape == (B, 150, 20, 2)

    # Forward pass 2: with temporal history
    print("\n--- Forward pass 2 (with history) ---")
    with torch.no_grad():
        out2 = model(images, intrinsics, extrinsics, ego_motion=ego_motion)
    print(f"  pred_logits: {out2['pred_logits'].shape}")
    print(f"  pred_points: {out2['pred_points'].shape}")
    assert out2["pred_logits"].shape == (B, 150, 4)
    assert out2["pred_points"].shape == (B, 150, 20, 2)

    # Verify points in [0, 1]
    assert out2["pred_points"].min() >= 0.0
    assert out2["pred_points"].max() <= 1.0
    print("  Points verified in [0, 1] range.")

    # Reset
    model.reset_temporal_state()
    assert not model.temporal_fusion.has_history
    print("\n  Temporal state reset verified.")

    print("\n" + "=" * 70)
    print("All shape checks PASSED.")
    print("=" * 70)
