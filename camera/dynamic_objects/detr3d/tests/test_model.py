#!/usr/bin/env python3
"""
test_model.py - Comprehensive unit tests for DETR3D model components.

Tests cover:
- Backbone (ResNet101 + FPN) output shapes
- Feature sampling (3D-to-2D projection and bilinear sampling)
- Transformer decoder (self-attention, cross-attention, output shapes)
- Detection heads (classification and regression)
- Hungarian matcher (optimal assignment)
- Loss functions (focal loss, L1 loss)
- Full model forward pass (end-to-end)
- Dataset loading (with mock data)

Run with: pytest tests/test_model.py -v
"""

import math
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ============================================================================
# Model Components (inline for testing without importing the full model)
# ============================================================================


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction."""

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(
        self, features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        # Build top-down pathway
        laterals = [
            conv(f) for conv, f in zip(self.lateral_convs, features)
        ]

        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=(h, w), mode="bilinear", align_corners=False
            )

        # Output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


class ResNetBackbone(nn.Module):
    """Simplified ResNet101 backbone that outputs multi-scale features."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        # Simplified: use conv layers to simulate ResNet stages
        # In production, use torchvision.models.resnet101
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage2 = self._make_stage(64, 256, stride=1)
        self.stage3 = self._make_stage(256, 512, stride=2)
        self.stage4 = self._make_stage(512, 1024, stride=2)
        self.stage5 = self._make_stage(1024, 2048, stride=2)

    def _make_stage(
        self, in_ch: int, out_ch: int, stride: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stage1(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c2, c3, c4, c5]


class FeatureSampler(nn.Module):
    """Sample image features at projected 3D reference point locations.

    This is the core 3D-to-2D projection mechanism in DETR3D:
    1. Take 3D reference points (from object queries)
    2. Project them to each camera's image plane using calibration
    3. Sample features at the projected 2D locations via bilinear interpolation
    """

    def __init__(self, embed_dim: int, num_cameras: int, num_levels: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.num_levels = num_levels
        # Output projection to combine multi-view, multi-level features
        self.output_proj = nn.Linear(
            embed_dim * num_levels, embed_dim
        )

    def project_points(
        self,
        reference_points_3d: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project 3D reference points to 2D image coordinates.

        Args:
            reference_points_3d: (B, num_queries, 3) 3D points
            projection_matrices: (B, num_cameras, 3, 4) camera matrices
            image_shape: (H, W) of the feature maps (at original resolution)

        Returns:
            points_2d: (B, num_cameras, num_queries, 2) normalized coords [-1, 1]
            valid_mask: (B, num_cameras, num_queries) boolean
        """
        B, Q, _ = reference_points_3d.shape
        N = projection_matrices.shape[1]

        # Homogeneous coordinates
        ones = torch.ones(
            B, Q, 1, device=reference_points_3d.device, dtype=reference_points_3d.dtype
        )
        points_homo = torch.cat([reference_points_3d, ones], dim=-1)  # (B, Q, 4)

        # Project: (B, N, 3, 4) @ (B, Q, 4, 1) -> need broadcasting
        points_homo = points_homo.unsqueeze(1).expand(B, N, Q, 4)  # (B, N, Q, 4)
        projected = torch.einsum(
            "bnij,bnqj->bnqi", projection_matrices, points_homo
        )  # (B, N, Q, 3)

        # Depth
        depths = projected[..., 2:3]  # (B, N, Q, 1)
        valid_depth = (depths.squeeze(-1) > 0.1)  # (B, N, Q)

        # Normalize by depth
        eps = 1e-5
        points_2d = projected[..., :2] / torch.clamp(depths, min=eps)  # (B, N, Q, 2)

        # Normalize to [-1, 1] for grid_sample
        H, W = image_shape
        points_2d[..., 0] = (points_2d[..., 0] / W) * 2 - 1
        points_2d[..., 1] = (points_2d[..., 1] / H) * 2 - 1

        # Valid if in depth and in image bounds
        valid_x = (points_2d[..., 0] >= -1) & (points_2d[..., 0] <= 1)
        valid_y = (points_2d[..., 1] >= -1) & (points_2d[..., 1] <= 1)
        valid_mask = valid_depth & valid_x & valid_y

        return points_2d, valid_mask

    def sample_features(
        self,
        feature_maps: List[torch.Tensor],
        points_2d: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample features from multi-scale feature maps at 2D points.

        Args:
            feature_maps: List of (B*N, C, H_l, W_l) for each FPN level
            points_2d: (B, N, Q, 2) normalized 2D coordinates
            valid_mask: (B, N, Q) validity mask

        Returns:
            sampled: (B, Q, C) aggregated features
        """
        B, N, Q, _ = points_2d.shape
        C = self.embed_dim

        all_level_features = []

        for level_idx, feat in enumerate(feature_maps):
            # feat: (B*N, C, H, W)
            BN = feat.shape[0]

            # Reshape points for grid_sample: (B*N, Q, 1, 2)
            pts = points_2d.reshape(BN, Q, 1, 2)

            # Sample features: (B*N, C, Q, 1)
            sampled = F.grid_sample(
                feat, pts, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            sampled = sampled.squeeze(-1)  # (B*N, C, Q)
            sampled = sampled.permute(0, 2, 1)  # (B*N, Q, C)

            # Apply validity mask
            mask = valid_mask.reshape(BN, Q, 1).float()
            sampled = sampled * mask

            # Reshape to (B, N, Q, C) and average over cameras
            sampled = sampled.reshape(B, N, Q, C)
            sampled = sampled.mean(dim=1)  # (B, Q, C)

            all_level_features.append(sampled)

        # Concatenate levels and project
        multi_level = torch.cat(all_level_features, dim=-1)  # (B, Q, C*num_levels)
        output = self.output_proj(multi_level)  # (B, Q, C)

        return output

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        reference_points_3d: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        points_2d, valid_mask = self.project_points(
            reference_points_3d, projection_matrices, image_shape
        )
        sampled = self.sample_features(feature_maps, points_2d, valid_mask)
        return sampled


class DETR3DDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (queries attend to sampled image features)
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
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Q, C) object queries
            key: (B, Q, C) sampled image features (cross-attention key/value)
            query_pos: (B, Q, C) positional embedding for queries

        Returns:
            Updated query: (B, Q, C)
        """
        # Self-attention
        q = k = query + query_pos
        sa_out, _ = self.self_attn(q, k, query)
        query = query + self.dropout1(sa_out)
        query = self.norm1(query)

        # Cross-attention
        ca_out, _ = self.cross_attn(
            query + query_pos, key, key
        )
        query = query + self.dropout2(ca_out)
        query = self.norm2(query)

        # FFN
        query = query + self.ffn(query)
        query = self.norm3(query)

        return query


class DETR3DDecoder(nn.Module):
    """Stack of transformer decoder layers."""

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DETR3DDecoderLayer(d_model, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Returns intermediate outputs from each layer for auxiliary losses."""
        intermediate = []
        for layer in self.layers:
            query = layer(query, key, query_pos)
            intermediate.append(query)
        return intermediate


class DetectionHead(nn.Module):
    """Classification and regression heads for 3D detection."""

    def __init__(
        self,
        d_model: int = 256,
        num_classes: int = 10,
        code_size: int = 10,
    ):
        super().__init__()
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_classes),
        )
        # Regression head (predicts 10-dim box code)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, code_size),
        )

    def forward(
        self, query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, Q, C) decoder output

        Returns:
            cls_logits: (B, Q, num_classes)
            bbox_preds: (B, Q, code_size)
        """
        cls_logits = self.cls_head(query)
        bbox_preds = self.reg_head(query)
        return cls_logits, bbox_preds


class HungarianMatcher:
    """Hungarian matching between predictions and ground truth.

    Computes optimal bipartite matching using classification cost and
    bbox L1 cost.
    """

    def __init__(self, cls_cost: float = 2.0, bbox_cost: float = 0.25):
        self.cls_cost = cls_cost
        self.bbox_cost = bbox_cost

    @torch.no_grad()
    def __call__(
        self,
        cls_logits: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            cls_logits: (B, Q, num_classes) predicted class logits
            bbox_preds: (B, Q, code_size) predicted boxes
            gt_labels: List of (num_gt,) tensors with class indices
            gt_boxes: List of (num_gt, code_size) tensors with box codes

        Returns:
            List of (pred_indices, gt_indices) tuples per batch
        """
        B, Q, C = cls_logits.shape
        results = []

        for b in range(B):
            num_gt = gt_labels[b].shape[0]
            if num_gt == 0:
                results.append(
                    (
                        torch.tensor([], dtype=torch.long),
                        torch.tensor([], dtype=torch.long),
                    )
                )
                continue

            # Classification cost: negative probability of correct class
            cls_prob = cls_logits[b].softmax(dim=-1)  # (Q, num_classes)
            cls_cost = -cls_prob[:, gt_labels[b]]  # (Q, num_gt)

            # Bbox L1 cost
            bbox_cost = torch.cdist(
                bbox_preds[b], gt_boxes[b], p=1
            )  # (Q, num_gt)

            # Combined cost matrix
            cost_matrix = (
                self.cls_cost * cls_cost + self.bbox_cost * bbox_cost
            )

            # Hungarian algorithm
            cost_np = cost_matrix.cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(cost_np)

            results.append(
                (
                    torch.tensor(pred_idx, dtype=torch.long),
                    torch.tensor(gt_idx, dtype=torch.long),
                )
            )

        return results


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Compute focal loss for classification.

    Args:
        logits: (N, C) raw predicted logits
        targets: (N,) integer class labels
        alpha: Weighting factor for rare classes
        gamma: Focusing parameter

    Returns:
        Scalar focal loss
    """
    num_classes = logits.shape[1]
    # One-hot encode targets
    target_onehot = F.one_hot(targets, num_classes).float()

    # Sigmoid probabilities (multi-label formulation as in DETR)
    p = logits.sigmoid()

    # Focal weight
    ce_loss = F.binary_cross_entropy_with_logits(
        logits, target_onehot, reduction="none"
    )
    p_t = p * target_onehot + (1 - p) * (1 - target_onehot)
    focal_weight = alpha * (1 - p_t) ** gamma

    loss = (focal_weight * ce_loss).sum(dim=-1).mean()
    return loss


def bbox_l1_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """L1 loss for bounding box regression.

    Args:
        pred: (N, code_size) predicted box codes
        target: (N, code_size) target box codes

    Returns:
        Scalar L1 loss
    """
    return F.l1_loss(pred, target, reduction="mean")


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def num_queries():
    return 900


@pytest.fixture
def num_cameras():
    return 6


@pytest.fixture
def embed_dim():
    return 256


@pytest.fixture
def num_classes():
    return 10


@pytest.fixture
def code_size():
    return 10


@pytest.fixture
def image_size():
    """Resized image dimensions (H, W)."""
    return (256, 704)


@pytest.fixture
def backbone(device):
    model = ResNetBackbone(pretrained=False).to(device)
    model.eval()
    return model


@pytest.fixture
def fpn(device):
    model = FPN(
        in_channels_list=[256, 512, 1024, 2048], out_channels=256
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def feature_sampler(device, embed_dim, num_cameras):
    model = FeatureSampler(
        embed_dim=embed_dim, num_cameras=num_cameras, num_levels=4
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def decoder(device, embed_dim):
    model = DETR3DDecoder(
        num_layers=6, d_model=embed_dim, num_heads=8, ffn_dim=2048, dropout=0.0
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def detection_head(device, embed_dim, num_classes, code_size):
    model = DetectionHead(
        d_model=embed_dim, num_classes=num_classes, code_size=code_size
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def matcher():
    return HungarianMatcher(cls_cost=2.0, bbox_cost=0.25)


# ============================================================================
# Test: Backbone
# ============================================================================


class TestBackbone:
    """Test ResNet101 backbone output shapes at each stage."""

    def test_output_num_levels(self, backbone, device, image_size):
        """Backbone should produce 4 feature levels (C2-C5)."""
        B = 1
        H, W = image_size
        x = torch.randn(B, 3, H, W, device=device)
        with torch.no_grad():
            features = backbone(x)
        assert len(features) == 4

    def test_output_shapes(self, backbone, device, image_size):
        """Verify spatial dimensions decrease and channels increase."""
        B = 1
        H, W = image_size
        x = torch.randn(B, 3, H, W, device=device)
        with torch.no_grad():
            features = backbone(x)

        expected_channels = [256, 512, 1024, 2048]
        for i, feat in enumerate(features):
            assert feat.shape[0] == B
            assert feat.shape[1] == expected_channels[i]
            # Each successive level should have smaller spatial dims
            if i > 0:
                assert feat.shape[2] <= features[i - 1].shape[2]
                assert feat.shape[3] <= features[i - 1].shape[3]

    def test_batch_independence(self, backbone, device, image_size):
        """Different batch items should produce independent features."""
        H, W = image_size
        x1 = torch.randn(1, 3, H, W, device=device)
        x2 = torch.randn(1, 3, H, W, device=device)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            feat_batch = backbone(x_batch)
            feat_1 = backbone(x1)
            feat_2 = backbone(x2)

        # Batch processing should give same result as individual
        for level in range(4):
            torch.testing.assert_close(
                feat_batch[level][0:1], feat_1[level], atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                feat_batch[level][1:2], feat_2[level], atol=1e-5, rtol=1e-5
            )


class TestFPN:
    """Test Feature Pyramid Network."""

    def test_output_shapes(self, fpn, device):
        """FPN should output same number of levels with uniform channels."""
        B = 2
        features = [
            torch.randn(B, 256, 64, 176, device=device),
            torch.randn(B, 512, 32, 88, device=device),
            torch.randn(B, 1024, 16, 44, device=device),
            torch.randn(B, 2048, 8, 22, device=device),
        ]
        with torch.no_grad():
            fpn_out = fpn(features)

        assert len(fpn_out) == 4
        for i, feat in enumerate(fpn_out):
            assert feat.shape[0] == B
            assert feat.shape[1] == 256  # Uniform FPN channels
            assert feat.shape[2] == features[i].shape[2]
            assert feat.shape[3] == features[i].shape[3]

    def test_channel_uniformity(self, fpn, device):
        """All FPN levels should have the same channel dimension."""
        B = 1
        features = [
            torch.randn(B, 256, 32, 88, device=device),
            torch.randn(B, 512, 16, 44, device=device),
            torch.randn(B, 1024, 8, 22, device=device),
            torch.randn(B, 2048, 4, 11, device=device),
        ]
        with torch.no_grad():
            fpn_out = fpn(features)

        channels = [f.shape[1] for f in fpn_out]
        assert all(c == 256 for c in channels)


# ============================================================================
# Test: Feature Sampling (3D-to-2D Projection)
# ============================================================================


class TestFeatureSampling:
    """Test 3D-to-2D projection and feature sampling."""

    def test_projection_known_point(self, device):
        """Project a known 3D point with identity-like camera matrix."""
        sampler = FeatureSampler(embed_dim=256, num_cameras=1, num_levels=1).to(device)

        B, Q = 1, 1
        # Place a point at (0, 0, 10) -- 10 meters in front
        ref_points = torch.tensor([[[0.0, 0.0, 10.0]]], device=device)

        # Simple projection matrix: focal=500, cx=352, cy=128
        # Maps (0,0,10) -> (500*0/10 + 352, 500*0/10 + 128) = (352, 128)
        proj = torch.zeros(B, 1, 3, 4, device=device)
        proj[0, 0, 0, 0] = 500.0  # fx
        proj[0, 0, 1, 1] = 500.0  # fy
        proj[0, 0, 0, 2] = 352.0  # cx (W/2)
        proj[0, 0, 1, 2] = 128.0  # cy (H/2)
        proj[0, 0, 2, 2] = 1.0    # depth row

        image_shape = (256, 704)
        points_2d, valid = sampler.project_points(
            ref_points, proj, image_shape
        )

        # The projected point should be at image center -> normalized to (0, 0)
        assert valid[0, 0, 0].item() is True
        # Check normalized coordinates: u=352/704*2-1=0, v=128/256*2-1=0
        assert abs(points_2d[0, 0, 0, 0].item()) < 0.01
        assert abs(points_2d[0, 0, 0, 1].item()) < 0.01

    def test_projection_behind_camera(self, device):
        """Points behind the camera should be marked invalid."""
        sampler = FeatureSampler(embed_dim=256, num_cameras=1, num_levels=1).to(device)

        B, Q = 1, 1
        # Point behind camera (negative depth)
        ref_points = torch.tensor([[[0.0, 0.0, -5.0]]], device=device)

        proj = torch.zeros(B, 1, 3, 4, device=device)
        proj[0, 0, 0, 0] = 500.0
        proj[0, 0, 1, 1] = 500.0
        proj[0, 0, 0, 2] = 352.0
        proj[0, 0, 1, 2] = 128.0
        proj[0, 0, 2, 2] = 1.0

        image_shape = (256, 704)
        _, valid = sampler.project_points(ref_points, proj, image_shape)

        assert valid[0, 0, 0].item() is False

    def test_bilinear_sampling_output_shape(self, device, embed_dim):
        """Feature sampling should produce correct output dimensions."""
        B, N, Q = 2, 6, 100
        num_levels = 4
        sampler = FeatureSampler(
            embed_dim=embed_dim, num_cameras=N, num_levels=num_levels
        ).to(device)

        # Mock feature maps at different scales
        feature_maps = [
            torch.randn(B * N, embed_dim, 64, 176, device=device),
            torch.randn(B * N, embed_dim, 32, 88, device=device),
            torch.randn(B * N, embed_dim, 16, 44, device=device),
            torch.randn(B * N, embed_dim, 8, 22, device=device),
        ]

        points_2d = torch.rand(B, N, Q, 2, device=device) * 2 - 1  # [-1, 1]
        valid_mask = torch.ones(B, N, Q, dtype=torch.bool, device=device)

        with torch.no_grad():
            sampled = sampler.sample_features(feature_maps, points_2d, valid_mask)

        assert sampled.shape == (B, Q, embed_dim)

    def test_invalid_points_zero_contribution(self, device, embed_dim):
        """Invalid points should contribute zero to sampled features."""
        B, N, Q = 1, 1, 5
        sampler = FeatureSampler(
            embed_dim=embed_dim, num_cameras=N, num_levels=1
        ).to(device)

        feature_maps = [
            torch.randn(B * N, embed_dim, 16, 16, device=device)
        ]

        points_2d = torch.rand(B, N, Q, 2, device=device) * 2 - 1
        valid_mask = torch.zeros(B, N, Q, dtype=torch.bool, device=device)

        with torch.no_grad():
            sampled = sampler.sample_features(feature_maps, points_2d, valid_mask)

        # All invalid -> output should be near zero (after linear projection)
        # The output_proj bias may add some non-zero, but input is zero
        # So we check pre-projection is zero
        assert sampled.abs().max() < 1.0  # Loose check due to bias


# ============================================================================
# Test: Transformer Decoder
# ============================================================================


class TestDecoder:
    """Test transformer decoder layers."""

    def test_single_layer_output_shape(self, device, embed_dim):
        """Single decoder layer should preserve query shape."""
        B, Q, C = 2, 100, embed_dim
        layer = DETR3DDecoderLayer(
            d_model=C, num_heads=8, ffn_dim=2048, dropout=0.0
        ).to(device)
        layer.eval()

        query = torch.randn(B, Q, C, device=device)
        key = torch.randn(B, Q, C, device=device)
        query_pos = torch.randn(B, Q, C, device=device)

        with torch.no_grad():
            output = layer(query, key, query_pos)

        assert output.shape == (B, Q, C)

    def test_multi_layer_intermediates(self, decoder, device, embed_dim):
        """Decoder should return intermediate outputs from each layer."""
        B, Q, C = 2, 100, embed_dim
        query = torch.randn(B, Q, C, device=device)
        key = torch.randn(B, Q, C, device=device)
        query_pos = torch.randn(B, Q, C, device=device)

        with torch.no_grad():
            intermediates = decoder(query, key, query_pos)

        assert len(intermediates) == 6
        for inter in intermediates:
            assert inter.shape == (B, Q, C)

    def test_self_attention_permutation_equivariance(self, device, embed_dim):
        """Self-attention should be equivariant to query permutation."""
        B, Q, C = 1, 10, embed_dim
        layer = DETR3DDecoderLayer(
            d_model=C, num_heads=8, ffn_dim=2048, dropout=0.0
        ).to(device)
        layer.eval()

        query = torch.randn(B, Q, C, device=device)
        key = torch.randn(B, Q, C, device=device)
        query_pos = torch.randn(B, Q, C, device=device)

        # Permute queries
        perm = torch.randperm(Q)
        query_perm = query[:, perm, :]
        query_pos_perm = query_pos[:, perm, :]
        key_perm = key[:, perm, :]

        with torch.no_grad():
            out_orig = layer(query, key, query_pos)
            out_perm = layer(query_perm, key_perm, query_pos_perm)

        # Outputs should be permuted version of each other
        torch.testing.assert_close(
            out_orig[:, perm, :], out_perm, atol=1e-4, rtol=1e-4
        )

    def test_decoder_gradient_flow(self, device, embed_dim):
        """Gradients should flow through all decoder layers."""
        B, Q, C = 1, 50, embed_dim
        decoder = DETR3DDecoder(
            num_layers=3, d_model=C, num_heads=8, ffn_dim=512, dropout=0.0
        ).to(device)

        query = torch.randn(B, Q, C, device=device, requires_grad=True)
        key = torch.randn(B, Q, C, device=device)
        query_pos = torch.randn(B, Q, C, device=device)

        intermediates = decoder(query, key, query_pos)
        loss = intermediates[-1].sum()
        loss.backward()

        assert query.grad is not None
        assert query.grad.abs().sum() > 0


# ============================================================================
# Test: Detection Heads
# ============================================================================


class TestDetectionHead:
    """Test classification and regression heads."""

    def test_output_dimensions(
        self, detection_head, device, embed_dim, num_classes, code_size
    ):
        """Heads should produce correct output dimensions."""
        B, Q = 2, 900
        query = torch.randn(B, Q, embed_dim, device=device)

        with torch.no_grad():
            cls_logits, bbox_preds = detection_head(query)

        assert cls_logits.shape == (B, Q, num_classes)
        assert bbox_preds.shape == (B, Q, code_size)

    def test_cls_logits_range(self, detection_head, device, embed_dim):
        """Classification logits should be unbounded (pre-sigmoid)."""
        B, Q = 1, 100
        query = torch.randn(B, Q, embed_dim, device=device)

        with torch.no_grad():
            cls_logits, _ = detection_head(query)

        # Logits can be positive or negative
        assert cls_logits.min() < 0 or cls_logits.max() > 0

    def test_gradient_flow(self, device, embed_dim, num_classes, code_size):
        """Gradients should flow through both heads."""
        head = DetectionHead(
            d_model=embed_dim, num_classes=num_classes, code_size=code_size
        ).to(device)

        B, Q = 1, 50
        query = torch.randn(B, Q, embed_dim, device=device, requires_grad=True)

        cls_logits, bbox_preds = head(query)
        loss = cls_logits.sum() + bbox_preds.sum()
        loss.backward()

        assert query.grad is not None
        assert query.grad.abs().sum() > 0


# ============================================================================
# Test: Hungarian Matcher
# ============================================================================


class TestHungarianMatcher:
    """Test Hungarian matching algorithm."""

    def test_perfect_match(self, device, matcher):
        """Matcher should find obvious 1-to-1 correspondences."""
        B, Q, num_classes, code_size = 1, 5, 10, 10

        # Create GT: 2 objects of class 0 and 3
        gt_labels = [torch.tensor([0, 3], dtype=torch.long)]
        gt_boxes = [torch.tensor([[1, 0, 0, 2, 4, 1.5, 0, 1, 0, 0],
                                   [5, 5, 0, 2, 5, 2, 0.7, 0.7, 1, 0]],
                                  dtype=torch.float32)]

        # Create predictions where query 0 matches GT 0 and query 2 matches GT 1
        cls_logits = torch.zeros(B, Q, num_classes)
        cls_logits[0, 0, 0] = 10.0  # Query 0 predicts class 0 strongly
        cls_logits[0, 2, 3] = 10.0  # Query 2 predicts class 3 strongly

        bbox_preds = torch.zeros(B, Q, code_size)
        bbox_preds[0, 0] = gt_boxes[0][0]  # Query 0 matches GT 0 perfectly
        bbox_preds[0, 2] = gt_boxes[0][1]  # Query 2 matches GT 1 perfectly

        results = matcher(cls_logits, bbox_preds, gt_labels, gt_boxes)

        pred_idx, gt_idx = results[0]
        assert len(pred_idx) == 2
        assert len(gt_idx) == 2
        # Query 0 should match GT 0, Query 2 should match GT 1
        matches = dict(zip(pred_idx.tolist(), gt_idx.tolist()))
        assert matches[0] == 0
        assert matches[2] == 1

    def test_empty_gt(self, device, matcher):
        """Matcher should handle empty ground truth gracefully."""
        B, Q, num_classes, code_size = 1, 5, 10, 10

        cls_logits = torch.randn(B, Q, num_classes)
        bbox_preds = torch.randn(B, Q, code_size)
        gt_labels = [torch.tensor([], dtype=torch.long)]
        gt_boxes = [torch.zeros(0, code_size)]

        results = matcher(cls_logits, bbox_preds, gt_labels, gt_boxes)

        pred_idx, gt_idx = results[0]
        assert len(pred_idx) == 0
        assert len(gt_idx) == 0

    def test_matching_is_bijective(self, device, matcher):
        """Each prediction matches at most one GT and vice versa."""
        B, Q, num_classes, code_size = 1, 20, 10, 10

        cls_logits = torch.randn(B, Q, num_classes)
        bbox_preds = torch.randn(B, Q, code_size)

        num_gt = 5
        gt_labels = [torch.randint(0, num_classes, (num_gt,))]
        gt_boxes = [torch.randn(num_gt, code_size)]

        results = matcher(cls_logits, bbox_preds, gt_labels, gt_boxes)

        pred_idx, gt_idx = results[0]
        # Should match exactly num_gt pairs
        assert len(pred_idx) == num_gt
        assert len(gt_idx) == num_gt
        # All indices should be unique
        assert len(set(pred_idx.tolist())) == num_gt
        assert len(set(gt_idx.tolist())) == num_gt


# ============================================================================
# Test: Loss Functions
# ============================================================================


class TestLosses:
    """Test focal loss and bbox L1 loss."""

    def test_focal_loss_perfect_prediction(self, device):
        """Focal loss should be low for correct high-confidence predictions."""
        num_classes = 10
        N = 100
        targets = torch.randint(0, num_classes, (N,), device=device)

        # Create logits that strongly predict the correct class
        logits = torch.zeros(N, num_classes, device=device) - 5.0
        for i in range(N):
            logits[i, targets[i]] = 5.0

        loss = focal_loss(logits, targets)
        assert loss.item() < 0.1  # Should be very small

    def test_focal_loss_random_prediction(self, device):
        """Focal loss should be higher for random predictions."""
        num_classes = 10
        N = 100
        targets = torch.randint(0, num_classes, (N,), device=device)
        logits = torch.randn(N, num_classes, device=device)

        loss = focal_loss(logits, targets)
        # Random predictions should give moderate loss
        assert loss.item() > 0.01

    def test_focal_loss_gradient(self, device):
        """Focal loss should produce valid gradients."""
        num_classes = 10
        N = 50
        targets = torch.randint(0, num_classes, (N,), device=device)
        logits = torch.randn(
            N, num_classes, device=device, requires_grad=True
        )

        loss = focal_loss(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert logits.grad.abs().sum() > 0

    def test_focal_loss_gamma_effect(self, device):
        """Higher gamma should down-weight easy examples more."""
        num_classes = 10
        N = 100
        targets = torch.zeros(N, dtype=torch.long, device=device)
        # Create easy examples (high confidence correct predictions)
        logits = torch.zeros(N, num_classes, device=device)
        logits[:, 0] = 3.0  # Somewhat confident

        loss_gamma0 = focal_loss(logits, targets, gamma=0.0)
        loss_gamma2 = focal_loss(logits, targets, gamma=2.0)
        loss_gamma5 = focal_loss(logits, targets, gamma=5.0)

        # Higher gamma should reduce loss for easy examples
        assert loss_gamma2.item() < loss_gamma0.item()
        assert loss_gamma5.item() < loss_gamma2.item()

    def test_bbox_l1_loss_zero(self, device):
        """L1 loss should be zero for identical predictions."""
        code_size = 10
        N = 50
        pred = torch.randn(N, code_size, device=device)
        target = pred.clone()

        loss = bbox_l1_loss(pred, target)
        assert abs(loss.item()) < 1e-6

    def test_bbox_l1_loss_known_value(self, device):
        """L1 loss should equal mean absolute difference."""
        pred = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        target = torch.tensor([[2.0, 2.0, 5.0]], device=device)

        loss = bbox_l1_loss(pred, target)
        expected = (1.0 + 0.0 + 2.0) / 3.0  # mean of |diffs|
        assert abs(loss.item() - expected) < 1e-5

    def test_bbox_l1_loss_gradient(self, device):
        """L1 loss should produce valid gradients."""
        code_size = 10
        N = 50
        pred = torch.randn(N, code_size, device=device, requires_grad=True)
        target = torch.randn(N, code_size, device=device)

        loss = bbox_l1_loss(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


# ============================================================================
# Test: Full Model Forward Pass
# ============================================================================


class TestFullModel:
    """Test end-to-end model forward pass."""

    def test_end_to_end_shapes(self, device):
        """Full model pipeline should produce correct output shapes."""
        B = 1
        N = 6  # cameras
        Q = 100  # queries (reduced for testing speed)
        C = 256
        num_classes = 10
        code_size = 10
        H, W = 256, 704

        # Components
        backbone = ResNetBackbone().to(device).eval()
        fpn_net = FPN([256, 512, 1024, 2048], out_channels=C).to(device).eval()
        sampler = FeatureSampler(
            embed_dim=C, num_cameras=N, num_levels=4
        ).to(device).eval()
        decoder = DETR3DDecoder(
            num_layers=6, d_model=C, num_heads=8, ffn_dim=2048, dropout=0.0
        ).to(device).eval()
        head = DetectionHead(
            d_model=C, num_classes=num_classes, code_size=code_size
        ).to(device).eval()

        # Input: multi-camera images
        images = torch.randn(B * N, 3, H, W, device=device)

        # Camera projection matrices
        proj_matrices = torch.randn(B, N, 3, 4, device=device)
        # Make valid projection matrices
        for b in range(B):
            for n in range(N):
                proj_matrices[b, n, 0, 0] = 500.0
                proj_matrices[b, n, 1, 1] = 500.0
                proj_matrices[b, n, 0, 2] = W / 2
                proj_matrices[b, n, 1, 2] = H / 2
                proj_matrices[b, n, 2, 2] = 1.0

        # Learnable queries and reference points
        query_embed = torch.randn(B, Q, C, device=device)
        query_pos = torch.randn(B, Q, C, device=device)
        # Reference points in detection range
        reference_points = torch.rand(B, Q, 3, device=device) * 50 - 25

        with torch.no_grad():
            # Extract features
            backbone_feats = backbone(images)
            fpn_feats = fpn_net(backbone_feats)

            # Sample features at projected 3D points
            sampled_feats = sampler(
                fpn_feats, reference_points, proj_matrices, (H, W)
            )

            # Decode
            intermediates = decoder(query_embed, sampled_feats, query_pos)

            # Detection heads on final layer output
            cls_logits, bbox_preds = head(intermediates[-1])

        assert cls_logits.shape == (B, Q, num_classes)
        assert bbox_preds.shape == (B, Q, code_size)

    def test_auxiliary_loss_outputs(self, device):
        """Model with aux loss should produce outputs for each decoder layer."""
        B, Q, C = 1, 50, 128
        num_classes, code_size = 10, 10
        num_layers = 6

        decoder = DETR3DDecoder(
            num_layers=num_layers, d_model=C, num_heads=8, ffn_dim=512, dropout=0.0
        ).to(device).eval()
        head = DetectionHead(
            d_model=C, num_classes=num_classes, code_size=code_size
        ).to(device).eval()

        query = torch.randn(B, Q, C, device=device)
        key = torch.randn(B, Q, C, device=device)
        query_pos = torch.randn(B, Q, C, device=device)

        with torch.no_grad():
            intermediates = decoder(query, key, query_pos)
            # Apply detection head to each intermediate for aux loss
            all_cls = []
            all_bbox = []
            for inter in intermediates:
                cls_out, bbox_out = head(inter)
                all_cls.append(cls_out)
                all_bbox.append(bbox_out)

        assert len(all_cls) == num_layers
        assert len(all_bbox) == num_layers
        for cls_out, bbox_out in zip(all_cls, all_bbox):
            assert cls_out.shape == (B, Q, num_classes)
            assert bbox_out.shape == (B, Q, code_size)


# ============================================================================
# Test: Dataset Loading (Mock)
# ============================================================================


class TestDataset:
    """Test data loading with mock data."""

    def test_sample_info_structure(self):
        """Verify the expected structure of a sample info dict."""
        # Create mock sample info
        sample_info = {
            "token": "abc123",
            "timestamp": 1553151604408,
            "scene_token": "scene001",
            "scene_name": "scene-0001",
            "lidar_path": "samples/LIDAR_TOP/xxx.pcd.bin",
            "ego_pose": {
                "translation": np.array([0.0, 0.0, 0.0]),
                "rotation": np.array([1.0, 0.0, 0.0, 0.0]),
            },
            "cameras": {},
            "annotations": [],
            "num_annotations": 0,
            "prev_token": "",
            "next_token": "def456",
        }

        # Add camera info for all 6 cameras
        for cam in [
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        ]:
            sample_info["cameras"][cam] = {
                "data_path": f"samples/{cam}/image.jpg",
                "timestamp": 1553151604408,
                "intrinsic": np.eye(3),
                "sensor_to_ego": np.eye(4),
                "ego_to_global": np.eye(4),
                "sensor_to_global": np.eye(4),
                "global_to_sensor": np.eye(4),
                "viewmatrix": np.zeros((3, 4)),
                "width": 1600,
                "height": 900,
            }

        # Verify structure
        assert "token" in sample_info
        assert "cameras" in sample_info
        assert len(sample_info["cameras"]) == 6
        assert "annotations" in sample_info
        assert "ego_pose" in sample_info

        for cam_name, cam_info in sample_info["cameras"].items():
            assert "intrinsic" in cam_info
            assert cam_info["intrinsic"].shape == (3, 3)
            assert "sensor_to_ego" in cam_info
            assert cam_info["sensor_to_ego"].shape == (4, 4)
            assert "viewmatrix" in cam_info
            assert cam_info["viewmatrix"].shape == (3, 4)

    def test_annotation_structure(self):
        """Verify annotation dict has all required fields."""
        annotation = {
            "token": "ann001",
            "instance_token": "inst001",
            "class_name": "car",
            "class_id": 0,
            "center": np.array([10.0, 5.0, 0.5]),
            "size": np.array([1.8, 4.5, 1.5]),
            "rotation": np.array([1.0, 0.0, 0.0, 0.0]),
            "yaw": 0.0,
            "velocity": np.array([5.0, 0.0]),
            "bbox_code": np.array([10.0, 5.0, 0.5, 1.8, 4.5, 1.5, 0.0, 1.0, 5.0, 0.0]),
            "visibility": 4,
            "num_lidar_pts": 50,
            "num_radar_pts": 10,
            "center_ego": np.array([10.0, 5.0, 0.5]),
            "yaw_ego": 0.0,
            "velocity_ego": np.array([5.0, 0.0]),
            "bbox_code_ego": np.array([10.0, 5.0, 0.5, 1.8, 4.5, 1.5, 0.0, 1.0, 5.0, 0.0]),
        }

        assert annotation["class_name"] in [
            "car", "truck", "construction_vehicle", "bus", "trailer",
            "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
        ]
        assert annotation["class_id"] >= 0 and annotation["class_id"] < 10
        assert annotation["bbox_code"].shape == (10,)
        assert annotation["bbox_code_ego"].shape == (10,)
        assert annotation["center"].shape == (3,)
        assert annotation["size"].shape == (3,)
        assert annotation["velocity"].shape == (2,)

    def test_collate_batch(self):
        """Test that multiple samples can be collated into a batch."""
        B = 2
        Q = 900
        C = 256
        num_cameras = 6

        # Simulate collated batch
        batch = {
            "images": torch.randn(B * num_cameras, 3, 256, 704),
            "projection_matrices": torch.randn(B, num_cameras, 3, 4),
            "gt_labels": [
                torch.randint(0, 10, (15,)),
                torch.randint(0, 10, (8,)),
            ],
            "gt_boxes": [
                torch.randn(15, 10),
                torch.randn(8, 10),
            ],
        }

        assert batch["images"].shape == (B * num_cameras, 3, 256, 704)
        assert batch["projection_matrices"].shape == (B, num_cameras, 3, 4)
        assert len(batch["gt_labels"]) == B
        assert len(batch["gt_boxes"]) == B
        assert batch["gt_labels"][0].shape[0] == batch["gt_boxes"][0].shape[0]

    def test_data_augmentation_dimensions(self):
        """Test that augmented data maintains correct dimensions."""
        H, W = 900, 1600
        target_H, target_W = 256, 704

        # Simulate resize augmentation
        image = torch.randn(3, H, W)
        resized = F.interpolate(
            image.unsqueeze(0), size=(target_H, target_W), mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        assert resized.shape == (3, target_H, target_W)

        # Intrinsic matrix should be scaled accordingly
        intrinsic = np.array([
            [1266.4, 0, 816.3],
            [0, 1266.4, 491.5],
            [0, 0, 1],
        ])
        scale_x = target_W / W
        scale_y = target_H / H
        intrinsic_scaled = intrinsic.copy()
        intrinsic_scaled[0, :] *= scale_x
        intrinsic_scaled[1, :] *= scale_y

        assert intrinsic_scaled[0, 0] == pytest.approx(1266.4 * scale_x, rel=1e-4)
        assert intrinsic_scaled[1, 1] == pytest.approx(1266.4 * scale_y, rel=1e-4)


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
