"""
Comprehensive tests for MapTR model components.

Run with:
    pytest tests/test_model.py -v
    pytest tests/test_model.py -v -k "test_backbone"

All tests use small dimensions for fast execution.
"""

import sys
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorch.backbone import ResNet50FPN, ResNet50, FPN, Bottleneck
from pytorch.gkt import GKT, GKTLayer, GeometryProjection, KernelAttention
from pytorch.map_decoder import MapDecoder, MapDecoderLayer, PositionalEncoding2D
from pytorch.heads import MapTRHead, ClassificationHead, PointRegressionHead
from pytorch.losses import (
    MapTRLoss, HungarianMatcher, PointSetLoss, DirectionLoss,
    PermutationLoss, chamfer_distance, focal_loss
)
from pytorch.model import MapTR, MapTRv2, build_model


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_config():
    """Small model configuration for fast testing."""
    return {
        "embed_dims": 32,
        "num_queries": 4,
        "num_points": 5,
        "num_classes": 3,
        "num_heads": 4,
        "num_decoder_layers": 2,
        "ffn_dims": 64,
        "bev_h": 10,
        "bev_w": 5,
        "num_cameras": 6,
        "img_h": 32,
        "img_w": 48,
        "batch_size": 2,
    }


@pytest.fixture
def random_images(small_config, device):
    """Random multi-camera images."""
    B = small_config["batch_size"]
    N = small_config["num_cameras"]
    H = small_config["img_h"]
    W = small_config["img_w"]
    return torch.randn(B, N, 3, H, W, device=device)


@pytest.fixture
def random_camera_params(small_config, device):
    """Random camera intrinsics and extrinsics."""
    B = small_config["batch_size"]
    N = small_config["num_cameras"]

    # Intrinsics: simple pinhole camera
    intrinsics = torch.zeros(B, N, 3, 3, device=device)
    intrinsics[:, :, 0, 0] = 400.0  # fx
    intrinsics[:, :, 1, 1] = 400.0  # fy
    intrinsics[:, :, 0, 2] = 24.0   # cx
    intrinsics[:, :, 1, 2] = 16.0   # cy
    intrinsics[:, :, 2, 2] = 1.0

    # Extrinsics: identity + small translation
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, 4, 4).clone()
    extrinsics[:, :, :3, 3] = torch.randn(B, N, 3, device=device) * 2

    return intrinsics, extrinsics


@pytest.fixture
def random_bev_features(small_config, device):
    """Random BEV feature map."""
    B = small_config["batch_size"]
    C = small_config["embed_dims"]
    H = small_config["bev_h"]
    W = small_config["bev_w"]
    return torch.randn(B, C, H, W, device=device)


# ============================================================================
# Backbone Tests
# ============================================================================

class TestBackbone:
    """Tests for ResNet-50 + FPN backbone."""

    def test_bottleneck_stride1(self, device):
        """Test Bottleneck block preserves spatial dimensions with stride=1."""
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
        )
        block = Bottleneck(64, 64, stride=1, downsample=downsample).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        out = block(x)
        assert out.shape == (2, 256, 16, 16)

    def test_bottleneck_stride2(self, device):
        """Test Bottleneck block with stride=2 halves spatial dims."""
        downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )
        block = Bottleneck(256, 128, stride=2, downsample=downsample).to(device)
        x = torch.randn(2, 256, 16, 16, device=device)
        out = block(x)
        assert out.shape == (2, 512, 8, 8)

    def test_resnet50_output_shapes(self, device):
        """Test ResNet-50 produces correct multi-scale feature shapes."""
        model = ResNet50(pretrained=False).to(device)
        x = torch.randn(2, 3, 64, 96, device=device)
        features = model(x)
        assert len(features) == 4
        assert features[0].shape[1] == 256    # C2: layer1
        assert features[1].shape[1] == 512    # C3: layer2
        assert features[2].shape[1] == 1024   # C4: layer3
        assert features[3].shape[1] == 2048   # C5: layer4
        # Check spatial dims relative to input
        assert features[0].shape[2] == 64 // 4
        assert features[0].shape[3] == 96 // 4

    def test_fpn_output_shapes(self, device):
        """Test FPN produces same-channel multi-scale features."""
        in_channels = [256, 512, 1024, 2048]
        out_channels = 64
        fpn = FPN(in_channels, out_channels, num_output_levels=4).to(device)

        features = [
            torch.randn(2, 256, 16, 24, device=device),
            torch.randn(2, 512, 8, 12, device=device),
            torch.randn(2, 1024, 4, 6, device=device),
            torch.randn(2, 2048, 2, 3, device=device),
        ]
        out = fpn(features)
        assert len(out) == 4
        for feat in out:
            assert feat.shape[1] == out_channels

    def test_resnet50fpn_multicam(self, device):
        """Test combined ResNet50FPN with multi-camera input."""
        model = ResNet50FPN(pretrained=False, fpn_out_channels=64, num_fpn_levels=4).to(device)
        # Input: [B, N_cams, 3, H, W]
        x = torch.randn(1, 6, 3, 64, 96, device=device)
        features = model(x)
        assert len(features) >= 4
        for feat in features:
            assert feat.shape[0] == 6  # B*N_cams = 1*6
            assert feat.shape[1] == 64

    def test_resnet50fpn_gradient_flow(self, device):
        """Test gradients flow through the backbone."""
        model = ResNet50FPN(pretrained=False, fpn_out_channels=32, num_fpn_levels=4).to(device)
        x = torch.randn(1, 2, 3, 32, 48, device=device, requires_grad=True)
        features = model(x)
        loss = sum(f.sum() for f in features)
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_resnet50fpn_cam_split(self, device):
        """Test forward_with_cam_split returns per-camera features."""
        model = ResNet50FPN(pretrained=False, fpn_out_channels=32, num_fpn_levels=4).to(device)
        B, N = 2, 3
        x = torch.randn(B, N, 3, 32, 48, device=device)
        per_cam_features = model.forward_with_cam_split(x)
        # per_cam_features is list (per level) of list (per camera)
        assert len(per_cam_features) >= 4
        for level_feats in per_cam_features:
            assert len(level_feats) == N
            for cam_feat in level_feats:
                assert cam_feat.shape[0] == B


# ============================================================================
# GKT Tests
# ============================================================================

class TestGKT:
    """Tests for Geometry-guided Kernel Transformer."""

    def test_geometry_projection_shapes(self, device, small_config, random_camera_params):
        """Test geometry projection produces valid coordinates and mask."""
        intrinsics, extrinsics = random_camera_params
        B = small_config["batch_size"]
        N = small_config["num_cameras"]
        bev_h = small_config["bev_h"]
        bev_w = small_config["bev_w"]
        num_z = 2

        proj = GeometryProjection(
            bev_h=bev_h,
            bev_w=bev_w,
            bev_x_range=(-30.0, 30.0),
            bev_y_range=(-15.0, 15.0),
            bev_z_range=(-5.0, 3.0),
            num_z_anchors=num_z,
        ).to(device)

        feat_h, feat_w = 8, 12
        coords, mask = proj(intrinsics, extrinsics, feat_h, feat_w)
        # coords: [B, N_cams, bev_h, bev_w, num_z, 2]
        assert coords.shape == (B, N, bev_h, bev_w, num_z, 2)
        # mask: [B, N_cams, bev_h, bev_w, num_z]
        assert mask.shape == (B, N, bev_h, bev_w, num_z)
        # Coordinates should be in [-1, 1] where valid
        assert coords[mask].abs().max() <= 1.0

    def test_kernel_attention_output(self, device, small_config):
        """Test KernelAttention produces correct output shape."""
        B = small_config["batch_size"]
        N = small_config["num_cameras"]
        Q = small_config["bev_h"] * small_config["bev_w"]
        C = small_config["embed_dims"]
        HW = 64  # spatial dims
        num_z = 2

        attn = KernelAttention(
            embed_dim=C,
            num_heads=small_config["num_heads"],
            num_points=4,
            num_z_anchors=num_z,
        ).to(device)

        query = torch.randn(B, Q, C, device=device)
        value = torch.randn(B, N, HW, C, device=device)
        ref_pts = torch.rand(B, N, Q, num_z, 2, device=device) * 2 - 1  # [-1, 1]
        valid_mask = torch.ones(B, N, Q, num_z, dtype=torch.bool, device=device)

        out = attn(query, value, ref_pts, valid_mask)
        assert out.shape == (B, Q, C)

    def test_gkt_output_shape(self, device, small_config, random_camera_params):
        """Test GKT produces correct BEV feature shape."""
        intrinsics, extrinsics = random_camera_params
        B = small_config["batch_size"]
        N = small_config["num_cameras"]
        bev_h = small_config["bev_h"]
        bev_w = small_config["bev_w"]
        C = small_config["embed_dims"]

        gkt = GKT(
            embed_dim=C,
            bev_h=bev_h,
            bev_w=bev_w,
            num_heads=small_config["num_heads"],
            num_points=4,
            num_z_anchors=2,
            num_layers=1,
            ffn_dim=C * 2,
            bev_x_range=(-30.0, 30.0),
            bev_y_range=(-15.0, 15.0),
            bev_z_range=(-5.0, 3.0),
            input_feat_channels=[C],
        ).to(device)

        feat_h = small_config["img_h"] // 4
        feat_w = small_config["img_w"] // 4
        multi_scale_feats = [
            torch.randn(B * N, C, feat_h, feat_w, device=device)
        ]

        bev_out = gkt(multi_scale_feats, intrinsics, extrinsics)
        assert bev_out.shape == (B, C, bev_h, bev_w)

    def test_gkt_gradient_flow(self, device, small_config, random_camera_params):
        """Test gradients flow through GKT."""
        intrinsics, extrinsics = random_camera_params
        B = small_config["batch_size"]
        N = small_config["num_cameras"]
        C = small_config["embed_dims"]

        gkt = GKT(
            embed_dim=C,
            bev_h=small_config["bev_h"],
            bev_w=small_config["bev_w"],
            num_heads=small_config["num_heads"],
            num_points=4,
            num_z_anchors=2,
            num_layers=1,
            ffn_dim=C * 2,
            input_feat_channels=[C],
        ).to(device)

        feat_h = small_config["img_h"] // 4
        feat_w = small_config["img_w"] // 4
        feats = [torch.randn(B * N, C, feat_h, feat_w,
                             device=device, requires_grad=True)]

        bev_out = gkt(feats, intrinsics, extrinsics)
        bev_out.sum().backward()
        assert feats[0].grad is not None
        assert feats[0].grad.abs().sum() > 0


# ============================================================================
# Map Decoder Tests
# ============================================================================

class TestMapDecoder:
    """Tests for transformer map decoder with hierarchical queries."""

    def test_positional_encoding_2d(self, device, small_config):
        """Test 2D positional encoding shape."""
        C = small_config["embed_dims"]
        pe = PositionalEncoding2D(embed_dims=C).to(device)
        H, W = small_config["bev_h"], small_config["bev_w"]
        encoding = pe((H, W), device)
        assert encoding.shape == (1, C, H, W)

    def test_decoder_layer(self, device, small_config):
        """Test single decoder layer."""
        C = small_config["embed_dims"]
        layer = MapDecoderLayer(
            embed_dims=C,
            num_heads=small_config["num_heads"],
            ffn_dims=small_config["ffn_dims"],
        ).to(device)

        B = small_config["batch_size"]
        num_queries = small_config["num_queries"]
        num_points = small_config["num_points"]
        Q = num_queries * num_points
        H, W = small_config["bev_h"], small_config["bev_w"]

        query = torch.randn(B, Q, C, device=device)
        query_pos = torch.randn(B, Q, C, device=device)
        memory = torch.randn(B, H * W, C, device=device)
        memory_pos = torch.randn(B, H * W, C, device=device)

        out = layer(query, query_pos, memory, memory_pos, num_queries, num_points)
        assert out.shape == (B, Q, C)

    def test_map_decoder_full(self, device, small_config, random_bev_features):
        """Test full map decoder with hierarchical queries."""
        C = small_config["embed_dims"]
        decoder = MapDecoder(
            embed_dims=C,
            num_heads=small_config["num_heads"],
            ffn_dims=small_config["ffn_dims"],
            num_layers=small_config["num_decoder_layers"],
            num_queries=small_config["num_queries"],
            num_points=small_config["num_points"],
            return_intermediate=True,
        ).to(device)

        intermediate_outputs, intermediate_ref_pts = decoder(random_bev_features)

        # Should return one output per decoder layer
        assert len(intermediate_outputs) == small_config["num_decoder_layers"]
        assert len(intermediate_ref_pts) == small_config["num_decoder_layers"]

        B = small_config["batch_size"]
        Q = small_config["num_queries"]
        P = small_config["num_points"]

        # Each output: [B, num_queries, num_points, embed_dims]
        for out in intermediate_outputs:
            assert out.shape == (B, Q, P, C)

        # Each ref_pts: [B, num_queries, num_points, 2]
        for ref in intermediate_ref_pts:
            assert ref.shape == (B, Q, P, 2)
            # Reference points should be in [0, 1] (sigmoid)
            assert ref.min() >= 0.0
            assert ref.max() <= 1.0

    def test_decoder_gradient_flow(self, device, small_config):
        """Test gradients flow through decoder."""
        C = small_config["embed_dims"]
        B = small_config["batch_size"]
        bev = torch.randn(B, C, small_config["bev_h"], small_config["bev_w"],
                          device=device, requires_grad=True)

        decoder = MapDecoder(
            embed_dims=C,
            num_heads=small_config["num_heads"],
            ffn_dims=small_config["ffn_dims"],
            num_layers=2,
            num_queries=small_config["num_queries"],
            num_points=small_config["num_points"],
            return_intermediate=True,
        ).to(device)

        outputs, ref_pts = decoder(bev)
        loss = outputs[-1].sum()
        loss.backward()
        assert bev.grad is not None
        assert bev.grad.abs().sum() > 0

    def test_decoder_decoupled_attention(self, device, small_config, random_bev_features):
        """Test decoder with decoupled self-attention mask."""
        C = small_config["embed_dims"]
        decoder = MapDecoder(
            embed_dims=C,
            num_heads=small_config["num_heads"],
            ffn_dims=small_config["ffn_dims"],
            num_layers=1,
            num_queries=small_config["num_queries"],
            num_points=small_config["num_points"],
            self_attn_mask_type="decoupled",
            return_intermediate=True,
        ).to(device)

        outputs, ref_pts = decoder(random_bev_features)
        assert len(outputs) == 1
        assert outputs[0].shape[1] == small_config["num_queries"]
        assert outputs[0].shape[2] == small_config["num_points"]


# ============================================================================
# Heads Tests
# ============================================================================

class TestHeads:
    """Tests for classification and regression heads."""

    def test_classification_head(self, device, small_config):
        """Test classification head output shape."""
        head = ClassificationHead(
            embed_dims=small_config["embed_dims"],
            num_classes=small_config["num_classes"],
        ).to(device)

        B = small_config["batch_size"]
        Q = small_config["num_queries"]
        features = torch.randn(B, Q, small_config["embed_dims"], device=device)

        logits = head(features)
        assert logits.shape == (B, Q, small_config["num_classes"])

    def test_point_regression_head(self, device, small_config):
        """Test point regression head output shape and range."""
        head = PointRegressionHead(
            embed_dims=small_config["embed_dims"],
            output_dims=2,
        ).to(device)

        B = small_config["batch_size"]
        Q = small_config["num_queries"]
        P = small_config["num_points"]
        features = torch.randn(B, Q, P, small_config["embed_dims"], device=device)

        points = head(features)
        assert points.shape == (B, Q, P, 2)
        # After sigmoid, values should be in [0, 1]
        assert points.min() >= 0.0
        assert points.max() <= 1.0

    def test_maptr_head_forward(self, device, small_config):
        """Test combined MapTR head with decoder outputs."""
        C = small_config["embed_dims"]
        Q = small_config["num_queries"]
        P = small_config["num_points"]
        B = small_config["batch_size"]
        num_layers = small_config["num_decoder_layers"]

        head = MapTRHead(
            embed_dims=C,
            num_classes=small_config["num_classes"],
            num_queries=Q,
            num_points=P,
            num_decoder_layers=num_layers,
            share_head_across_layers=True,
            use_iterative_refinement=True,
        ).to(device)

        # Simulate decoder outputs: list of [B, Q, P, C] per layer
        decoder_outputs = [
            torch.randn(B, Q, P, C, device=device)
            for _ in range(num_layers)
        ]
        # Reference points: list of [B, Q, P, 2] per layer
        reference_points = [
            torch.sigmoid(torch.randn(B, Q, P, 2, device=device))
            for _ in range(num_layers)
        ]

        result = head(decoder_outputs, reference_points)

        assert "cls_scores" in result
        assert "point_coords" in result
        assert len(result["cls_scores"]) == num_layers
        assert len(result["point_coords"]) == num_layers

        for cls in result["cls_scores"]:
            assert cls.shape == (B, Q, small_config["num_classes"])

        for pts in result["point_coords"]:
            assert pts.shape == (B, Q, P, 2)
            assert pts.min() >= 0.0
            assert pts.max() <= 1.0

    def test_maptr_head_predict(self, device, small_config):
        """Test MapTR head predict method for inference."""
        C = small_config["embed_dims"]
        Q = small_config["num_queries"]
        P = small_config["num_points"]
        B = small_config["batch_size"]

        head = MapTRHead(
            embed_dims=C,
            num_classes=small_config["num_classes"],
            num_queries=Q,
            num_points=P,
            num_decoder_layers=2,
        ).to(device)

        decoder_outputs = [
            torch.randn(B, Q, P, C, device=device) for _ in range(2)
        ]
        reference_points = [
            torch.sigmoid(torch.randn(B, Q, P, 2, device=device)) for _ in range(2)
        ]

        result = head.predict(decoder_outputs, reference_points, score_threshold=0.1)
        assert "scores" in result
        assert "labels" in result
        assert "points" in result
        assert "mask" in result
        assert result["scores"].shape == (B, Q)
        assert result["labels"].shape == (B, Q)
        assert result["points"].shape == (B, Q, P, 2)
        assert result["mask"].shape == (B, Q)


# ============================================================================
# Loss Tests
# ============================================================================

class TestLosses:
    """Tests for hierarchical matching and losses."""

    def test_chamfer_distance_identical(self, device):
        """Test Chamfer distance is zero for identical point sets."""
        pts = torch.rand(5, 2, device=device)
        dist = chamfer_distance(pts, pts, reduction="mean")
        assert dist.item() < 1e-5

    def test_chamfer_distance_known_value(self, device):
        """Test Chamfer distance with known values."""
        pts_a = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=device)
        pts_b = torch.tensor([[0.0, 1.0], [1.0, 1.0]], device=device)
        dist = chamfer_distance(pts_a, pts_b, reduction="mean")
        # Each point in A is distance 1.0 from nearest in B, and vice versa
        assert abs(dist.item() - 1.0) < 1e-5

    def test_chamfer_distance_batched(self, device):
        """Test batched Chamfer distance."""
        pts_a = torch.rand(3, 10, 2, device=device)
        pts_b = torch.rand(3, 8, 2, device=device)
        dist = chamfer_distance(pts_a, pts_b, reduction="none")
        assert dist.shape == (3,)
        assert (dist >= 0).all()

    def test_point_set_loss_zero(self, device):
        """Test PointSetLoss is zero for identical point sets."""
        loss_fn = PointSetLoss(loss_weight=1.0)
        pts = torch.rand(2, 3, 5, 2, device=device)  # [B, N_matched, N_pts, 2]
        loss = loss_fn(pts, pts)
        assert loss.item() < 1e-5

    def test_point_set_loss_positive(self, device):
        """Test PointSetLoss is positive for different point sets."""
        loss_fn = PointSetLoss(loss_weight=1.0)
        pts_a = torch.zeros(2, 3, 5, 2, device=device)
        pts_b = torch.ones(2, 3, 5, 2, device=device)
        loss = loss_fn(pts_a, pts_b)
        assert loss.item() > 0

    def test_direction_loss_same_direction(self, device):
        """Test direction loss is zero for same-direction polylines."""
        loss_fn = DirectionLoss(loss_weight=1.0)

        # Create polyline going left to right
        pts = torch.linspace(0, 1, 5).view(1, 1, 5, 1).to(device)
        pts = pts.expand(1, 1, 5, 2).clone()
        pts[:, :, :, 1] = 0.5  # y=0.5 constant

        loss = loss_fn(pts, pts)
        # Same direction, cosine similarity = 1, loss should be 0
        assert loss.item() < 1e-5

    def test_direction_loss_reversed(self, device):
        """Test direction loss penalizes reversed ordering."""
        loss_fn = DirectionLoss(loss_weight=1.0)

        pts_forward = torch.linspace(0, 1, 5).view(1, 1, 5, 1).to(device)
        pts_forward = pts_forward.expand(1, 1, 5, 2).clone()
        pts_forward[:, :, :, 1] = 0.5

        pts_reversed = pts_forward.flip(dims=[2])

        loss_same = loss_fn(pts_forward, pts_forward)
        loss_reversed = loss_fn(pts_forward, pts_reversed)

        # Reversed should have higher loss
        assert loss_reversed.item() > loss_same.item()

    def test_hungarian_matcher_indices(self, device, small_config):
        """Test Hungarian matcher returns valid indices."""
        Q = small_config["num_queries"]
        P = small_config["num_points"]
        C = small_config["num_classes"]
        B = small_config["batch_size"]

        matcher = HungarianMatcher(
            cost_class=2.0,
            cost_pts=5.0,
            cost_dir=0.005,
        )

        cls_scores = torch.randn(B, Q, C, device=device)
        pred_pts = torch.sigmoid(torch.randn(B, Q, P, 2, device=device))

        # GT: 2 instances per sample
        num_gt = 2
        gt_labels = torch.randint(0, C, (B, num_gt), device=device)
        gt_pts = torch.rand(B, num_gt, P, 2, device=device)
        gt_masks = torch.ones(B, num_gt, dtype=torch.bool, device=device)

        indices = matcher(cls_scores, pred_pts, gt_labels, gt_pts, gt_masks)

        assert len(indices) == B
        for pred_idx, gt_idx in indices:
            # Should match min(Q, num_gt) = 2 pairs
            assert len(pred_idx) == num_gt
            assert len(gt_idx) == num_gt
            # Indices should be valid
            assert (pred_idx < Q).all()
            assert (gt_idx < num_gt).all()
            # No duplicates
            assert len(set(pred_idx.tolist())) == len(pred_idx)
            assert len(set(gt_idx.tolist())) == len(gt_idx)

    def test_hungarian_matcher_empty_gt(self, device, small_config):
        """Test Hungarian matcher handles empty GT correctly."""
        Q = small_config["num_queries"]
        P = small_config["num_points"]
        C = small_config["num_classes"]

        matcher = HungarianMatcher()
        cls_scores = torch.randn(1, Q, C, device=device)
        pred_pts = torch.rand(1, Q, P, 2, device=device)
        gt_labels = torch.zeros(1, 0, dtype=torch.long, device=device)
        gt_pts = torch.zeros(1, 0, P, 2, device=device)
        gt_masks = torch.zeros(1, 0, dtype=torch.bool, device=device)

        indices = matcher(cls_scores, pred_pts, gt_labels, gt_pts, gt_masks)
        assert len(indices) == 1
        assert len(indices[0][0]) == 0
        assert len(indices[0][1]) == 0

    def test_permutation_loss_finds_shift(self, device):
        """Test permutation loss finds best cyclic shift."""
        perm_loss = PermutationLoss(try_reverse=True)

        # GT: simple ordered sequence
        gt = torch.tensor([[[0.0, 0.0], [0.25, 0.0], [0.5, 0.0],
                           [0.75, 0.0], [1.0, 0.0]]], device=device)

        # Pred: same but shifted by 2
        pred = torch.tensor([[[0.5, 0.0], [0.75, 0.0], [1.0, 0.0],
                             [0.0, 0.0], [0.25, 0.0]]], device=device)

        # Add batch dim: [B, N_matched, P, 2]
        gt = gt.unsqueeze(0)   # [1, 1, 5, 2]
        pred = pred.unsqueeze(0)  # [1, 1, 5, 2]

        loss, permuted_gt = perm_loss(pred, gt)
        # With optimal shift found, loss should be very small
        assert loss.item() < 0.1

    def test_maptr_loss_full(self, device, small_config):
        """Test full MapTR loss produces valid scalar outputs."""
        Q = small_config["num_queries"]
        P = small_config["num_points"]
        C = small_config["num_classes"]
        B = small_config["batch_size"]

        criterion = MapTRLoss(
            num_classes=C,
            num_points=P,
            cls_weight=2.0,
            pts_weight=5.0,
            dir_weight=0.005,
        )

        # Build predictions in the format MapTRHead produces
        predictions = {
            "cls_scores": [
                torch.randn(B, Q, C, device=device, requires_grad=True),
                torch.randn(B, Q, C, device=device, requires_grad=True),
            ],
            "point_coords": [
                torch.sigmoid(torch.randn(B, Q, P, 2, device=device, requires_grad=True)),
                torch.sigmoid(torch.randn(B, Q, P, 2, device=device, requires_grad=True)),
            ],
        }

        # GT
        num_gt = 2
        gt_labels = torch.randint(0, C, (B, num_gt), device=device)
        gt_pts = torch.rand(B, num_gt, P, 2, device=device)
        gt_masks = torch.ones(B, num_gt, dtype=torch.bool, device=device)

        loss_dict = criterion(predictions, gt_labels, gt_pts, gt_masks)

        assert isinstance(loss_dict, dict)
        assert "loss" in loss_dict
        assert "cls_loss" in loss_dict
        assert "pts_loss" in loss_dict
        assert "dir_loss" in loss_dict

        # All losses should be scalar
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                assert value.dim() == 0, f"{key} is not scalar: shape={value.shape}"
                assert not torch.isnan(value), f"{key} is NaN"
                assert not torch.isinf(value), f"{key} is Inf"

    def test_maptr_loss_gradient(self, device, small_config):
        """Test loss gradients flow to predictions."""
        Q = small_config["num_queries"]
        P = small_config["num_points"]
        C = small_config["num_classes"]
        B = small_config["batch_size"]

        criterion = MapTRLoss(num_classes=C, num_points=P)

        cls_scores = torch.randn(B, Q, C, device=device, requires_grad=True)
        pred_pts = torch.sigmoid(torch.randn(B, Q, P, 2, device=device, requires_grad=True))

        predictions = {
            "cls_scores": [cls_scores],
            "point_coords": [pred_pts],
        }

        gt_labels = torch.randint(0, C, (B, 2), device=device)
        gt_pts = torch.rand(B, 2, P, 2, device=device)
        gt_masks = torch.ones(B, 2, dtype=torch.bool, device=device)

        loss_dict = criterion(predictions, gt_labels, gt_pts, gt_masks)
        loss_dict["loss"].backward()

        assert cls_scores.grad is not None
        assert pred_pts.grad is not None

    def test_focal_loss_basic(self, device):
        """Test focal loss produces valid output."""
        pred = torch.randn(4, 3, device=device)
        target = torch.randint(0, 3, (4,), device=device)
        loss = focal_loss(pred, target, reduction="mean")
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)


# ============================================================================
# Full Model Tests
# ============================================================================

class TestFullModel:
    """Tests for the complete MapTR model."""

    def test_maptr_forward(self, device, small_config, random_images, random_camera_params):
        """Test end-to-end forward pass of MapTR."""
        intrinsics, extrinsics = random_camera_params
        B = small_config["batch_size"]
        Q = small_config["num_queries"]
        P = small_config["num_points"]
        C = small_config["num_classes"]

        model = MapTR(
            num_cameras=small_config["num_cameras"],
            num_classes=C,
            num_queries=Q,
            num_points=P,
            embed_dims=small_config["embed_dims"],
            backbone_pretrained=False,
            fpn_out_channels=small_config["embed_dims"],
            num_fpn_levels=4,
            bev_h=small_config["bev_h"],
            bev_w=small_config["bev_w"],
            gkt_num_heads=small_config["num_heads"],
            gkt_num_points=4,
            gkt_num_z_anchors=2,
            gkt_num_layers=1,
            decoder_num_heads=small_config["num_heads"],
            decoder_ffn_dims=small_config["ffn_dims"],
            decoder_num_layers=small_config["num_decoder_layers"],
        ).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(random_images, intrinsics, extrinsics)

        assert isinstance(outputs, dict)
        assert "cls_scores" in outputs
        assert "point_coords" in outputs
        assert "bev_features" in outputs

        # cls_scores is a list (one per decoder layer)
        assert len(outputs["cls_scores"]) == small_config["num_decoder_layers"]
        assert outputs["cls_scores"][-1].shape == (B, Q, C)

        # point_coords is a list (one per decoder layer)
        assert len(outputs["point_coords"]) == small_config["num_decoder_layers"]
        assert outputs["point_coords"][-1].shape == (B, Q, P, 2)

        # BEV features
        assert outputs["bev_features"].shape == (
            B, small_config["embed_dims"], small_config["bev_h"], small_config["bev_w"]
        )

    def test_maptr_inference(self, device, small_config, random_images, random_camera_params):
        """Test MapTR inference method with score thresholding."""
        intrinsics, extrinsics = random_camera_params

        model = MapTR(
            num_cameras=small_config["num_cameras"],
            num_classes=small_config["num_classes"],
            num_queries=small_config["num_queries"],
            num_points=small_config["num_points"],
            embed_dims=small_config["embed_dims"],
            backbone_pretrained=False,
            fpn_out_channels=small_config["embed_dims"],
            bev_h=small_config["bev_h"],
            bev_w=small_config["bev_w"],
            gkt_num_heads=small_config["num_heads"],
            gkt_num_points=4,
            gkt_num_z_anchors=2,
            gkt_num_layers=1,
            decoder_num_heads=small_config["num_heads"],
            decoder_ffn_dims=small_config["ffn_dims"],
            decoder_num_layers=small_config["num_decoder_layers"],
        ).to(device)

        model.eval()
        with torch.no_grad():
            results = model.inference(random_images, intrinsics, extrinsics, score_threshold=0.3)

        assert "scores" in results
        assert "labels" in results
        assert "points" in results
        assert "mask" in results

    def test_maptrv2_forward(self, device, small_config, random_images, random_camera_params):
        """Test MapTRv2 forward with auxiliary outputs."""
        intrinsics, extrinsics = random_camera_params
        B = small_config["batch_size"]

        model = MapTRv2(
            num_cameras=small_config["num_cameras"],
            num_classes=small_config["num_classes"],
            num_queries=small_config["num_queries"],
            num_points=small_config["num_points"],
            embed_dims=small_config["embed_dims"],
            backbone_pretrained=False,
            fpn_out_channels=small_config["embed_dims"],
            bev_h=small_config["bev_h"],
            bev_w=small_config["bev_w"],
            gkt_num_heads=small_config["num_heads"],
            gkt_num_points=4,
            gkt_num_z_anchors=2,
            gkt_num_layers=1,
            decoder_num_heads=small_config["num_heads"],
            decoder_ffn_dims=small_config["ffn_dims"],
            decoder_num_layers=small_config["num_decoder_layers"],
            use_decoupled_attn=True,
            one_to_many_num_groups=2,
            use_dense_bev_head=True,
        ).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(random_images, intrinsics, extrinsics)

        assert "cls_scores" in outputs
        assert "point_coords" in outputs
        # Dense BEV head should produce segmentation
        assert "dense_bev_seg" in outputs
        assert outputs["dense_bev_seg"].shape == (
            B, small_config["num_classes"], small_config["bev_h"], small_config["bev_w"]
        )

    def test_maptrv2_training_aux_outputs(self, device, small_config, random_images, random_camera_params):
        """Test MapTRv2 produces auxiliary outputs in train mode."""
        intrinsics, extrinsics = random_camera_params

        model = MapTRv2(
            num_cameras=small_config["num_cameras"],
            num_classes=small_config["num_classes"],
            num_queries=small_config["num_queries"],
            num_points=small_config["num_points"],
            embed_dims=small_config["embed_dims"],
            backbone_pretrained=False,
            fpn_out_channels=small_config["embed_dims"],
            bev_h=small_config["bev_h"],
            bev_w=small_config["bev_w"],
            gkt_num_heads=small_config["num_heads"],
            gkt_num_points=4,
            gkt_num_z_anchors=2,
            gkt_num_layers=1,
            decoder_num_heads=small_config["num_heads"],
            decoder_ffn_dims=small_config["ffn_dims"],
            decoder_num_layers=small_config["num_decoder_layers"],
            one_to_many_num_groups=2,
        ).to(device)

        model.train()
        outputs = model(random_images, intrinsics, extrinsics)

        # In training mode, should have auxiliary one-to-many outputs
        assert "aux_cls_scores" in outputs
        assert "aux_point_coords" in outputs

    def test_build_model_factory(self, device, small_config):
        """Test build_model factory function for MapTR."""
        model = build_model(
            model_type="MapTR",
            num_cameras=small_config["num_cameras"],
            num_classes=small_config["num_classes"],
            num_queries=small_config["num_queries"],
            num_points=small_config["num_points"],
            embed_dims=small_config["embed_dims"],
            backbone_pretrained=False,
            bev_h=small_config["bev_h"],
            bev_w=small_config["bev_w"],
            decoder_num_layers=small_config["num_decoder_layers"],
        )
        assert isinstance(model, MapTR)

    def test_build_model_maptrv2(self, device, small_config):
        """Test build_model factory function for MapTRv2."""
        model = build_model(
            model_type="MapTRv2",
            num_cameras=small_config["num_cameras"],
            num_classes=small_config["num_classes"],
            num_queries=small_config["num_queries"],
            num_points=small_config["num_points"],
            embed_dims=small_config["embed_dims"],
            backbone_pretrained=False,
            bev_h=small_config["bev_h"],
            bev_w=small_config["bev_w"],
            decoder_num_layers=small_config["num_decoder_layers"],
        )
        assert isinstance(model, MapTRv2)

    def test_build_model_invalid_type(self, small_config):
        """Test build_model raises ValueError for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model(model_type="InvalidModel")

    def test_model_parameter_count(self, device, small_config):
        """Test model has a reasonable number of parameters."""
        model = MapTR(
            num_cameras=small_config["num_cameras"],
            num_classes=small_config["num_classes"],
            num_queries=small_config["num_queries"],
            num_points=small_config["num_points"],
            embed_dims=small_config["embed_dims"],
            backbone_pretrained=False,
            bev_h=small_config["bev_h"],
            bev_w=small_config["bev_w"],
            gkt_num_layers=1,
            decoder_num_layers=small_config["num_decoder_layers"],
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        # Small config should have reasonable param count
        assert num_params > 1000
        assert num_params < 200_000_000


# ============================================================================
# Evaluation / Chamfer AP Tests
# ============================================================================

class TestEvaluation:
    """Tests for evaluation metrics."""

    def test_chamfer_distance_numpy_zero(self):
        """Test Chamfer distance is zero for identical points (numpy)."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        # Import from visualize_results since it has a numpy implementation
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
        ))
        from visualize_results import chamfer_distance as cd_np
        dist = cd_np(pts, pts)
        assert dist < 1e-6

    def test_chamfer_distance_numpy_known(self):
        """Test Chamfer distance with known values (numpy)."""
        pts_a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        pts_b = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
        ))
        from visualize_results import chamfer_distance as cd_np
        dist = cd_np(pts_a, pts_b)
        # All nearest distances = 1.0
        assert abs(dist - 1.0) < 1e-5

    def test_ap_computation_concept(self):
        """Test AP computation: perfect predictions should score high."""
        # When Chamfer distance = 0 (perfect match), all predictions
        # with score > threshold are true positives
        # With threshold=0.5m and distance=0, AP should be 1.0
        from pytorch.losses import chamfer_distance as cd_torch

        pred = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        gt = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        dist = cd_torch(pred, gt, reduction="mean")
        # Perfect match -> distance = 0 -> below any threshold -> TP
        assert dist.item() < 0.01


# ============================================================================
# Dataset Tests
# ============================================================================

class TestDataset:
    """Tests for dataset utilities with mock data."""

    def test_resample_polyline(self):
        """Test polyline resampling to fixed point count."""
        from scripts.prepare_data import resample_polyline

        points = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        resampled = resample_polyline(points, num_points=5)

        assert resampled.shape == (5, 2)
        expected_x = np.linspace(0, 10, 5)
        np.testing.assert_allclose(resampled[:, 0], expected_x, atol=1e-5)
        np.testing.assert_allclose(resampled[:, 1], 0, atol=1e-5)

    def test_resample_polyline_curve(self):
        """Test polyline resampling preserves endpoints on a curve."""
        from scripts.prepare_data import resample_polyline

        t = np.linspace(0, np.pi / 2, 50)
        points = np.column_stack([np.cos(t), np.sin(t)])
        resampled = resample_polyline(points, num_points=10)

        assert resampled.shape == (10, 2)
        np.testing.assert_allclose(resampled[0], [1, 0], atol=0.02)
        np.testing.assert_allclose(resampled[-1], [0, 1], atol=0.02)

    def test_resample_single_point(self):
        """Test resampling a single point repeats it."""
        from scripts.prepare_data import resample_polyline

        points = np.array([[5.0, 3.0]], dtype=np.float64)
        resampled = resample_polyline(points, num_points=10)
        assert resampled.shape == (10, 2)
        np.testing.assert_allclose(resampled, [[5.0, 3.0]] * 10, atol=1e-5)

    def test_clip_polyline_within_range(self):
        """Test clipping a polyline that's entirely within range."""
        from scripts.prepare_data import clip_polyline_to_range

        points = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]], dtype=np.float64)
        clipped = clip_polyline_to_range(points, (-30, 30), (-15, 15))
        assert clipped is not None
        assert len(clipped) == 3

    def test_clip_polyline_outside(self):
        """Test clipping returns None for polylines entirely outside."""
        from scripts.prepare_data import clip_polyline_to_range

        points = np.array([[100.0, 100.0], [200.0, 200.0]], dtype=np.float64)
        clipped = clip_polyline_to_range(points, (-30, 30), (-15, 15))
        assert clipped is None

    def test_clip_polyline_partial(self):
        """Test clipping keeps interior points within range."""
        from scripts.prepare_data import clip_polyline_to_range

        points = np.array([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0]], dtype=np.float64)
        clipped = clip_polyline_to_range(points, (-30, 30), (-15, 15))
        assert clipped is not None
        # Only the middle point is in range
        assert len(clipped) >= 1

    def test_quaternion_identity(self):
        """Test identity quaternion gives identity rotation."""
        from scripts.prepare_data import quaternion_to_rotation_matrix

        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_quaternion_90_degree_z(self):
        """Test 90-degree rotation around z-axis."""
        from scripts.prepare_data import quaternion_to_rotation_matrix

        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
        R = quaternion_to_rotation_matrix(q)
        # 90 degrees around z: x -> y, y -> -x
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_dataset_collate_fn(self, device):
        """Test dataset collate function handles variable-length GT."""
        from pytorch.dataset import NuScenesMapDataset

        # Create mock batch with different numbers of GT instances
        batch = [
            {
                "images": torch.randn(6, 3, 32, 48),
                "intrinsics": torch.eye(3).unsqueeze(0).expand(6, 3, 3),
                "extrinsics": torch.eye(4).unsqueeze(0).expand(6, 4, 4),
                "gt_labels": torch.tensor([0, 1, 2]),
                "gt_points": torch.rand(3, 5, 2),
                "sample_token": "token_a",
            },
            {
                "images": torch.randn(6, 3, 32, 48),
                "intrinsics": torch.eye(3).unsqueeze(0).expand(6, 3, 3),
                "extrinsics": torch.eye(4).unsqueeze(0).expand(6, 4, 4),
                "gt_labels": torch.tensor([1]),
                "gt_points": torch.rand(1, 5, 2),
                "sample_token": "token_b",
            },
        ]

        collated = NuScenesMapDataset.collate_fn(batch)

        assert collated["images"].shape == (2, 6, 3, 32, 48)
        assert collated["intrinsics"].shape == (2, 6, 3, 3)
        assert collated["extrinsics"].shape == (2, 6, 4, 4)
        # Max instances = 3 (from first sample)
        assert collated["gt_labels"].shape == (2, 3)
        assert collated["gt_points"].shape == (2, 3, 5, 2)
        assert collated["gt_masks"].shape == (2, 3)
        # First sample: all 3 valid
        assert collated["gt_masks"][0].sum() == 3
        # Second sample: only 1 valid
        assert collated["gt_masks"][1].sum() == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_train_step_simulation(self, device, small_config, random_images, random_camera_params):
        """Simulate a single training step: forward + loss + backward."""
        intrinsics, extrinsics = random_camera_params
        B = small_config["batch_size"]
        P = small_config["num_points"]
        Q = small_config["num_queries"]
        C = small_config["num_classes"]

        model = MapTR(
            num_cameras=small_config["num_cameras"],
            num_classes=C,
            num_queries=Q,
            num_points=P,
            embed_dims=small_config["embed_dims"],
            backbone_pretrained=False,
            fpn_out_channels=small_config["embed_dims"],
            bev_h=small_config["bev_h"],
            bev_w=small_config["bev_w"],
            gkt_num_heads=small_config["num_heads"],
            gkt_num_points=4,
            gkt_num_z_anchors=2,
            gkt_num_layers=1,
            decoder_num_heads=small_config["num_heads"],
            decoder_ffn_dims=small_config["ffn_dims"],
            decoder_num_layers=small_config["num_decoder_layers"],
        ).to(device)

        criterion = MapTRLoss(num_classes=C, num_points=P)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Forward
        model.train()
        outputs = model(random_images, intrinsics, extrinsics)

        # Create ground truth
        num_gt = 2
        gt_labels = torch.randint(0, C, (B, num_gt), device=device)
        gt_pts = torch.rand(B, num_gt, P, 2, device=device)
        gt_masks = torch.ones(B, num_gt, dtype=torch.bool, device=device)

        # Compute loss
        loss_dict = criterion(outputs, gt_labels, gt_pts, gt_masks)

        # Backward
        optimizer.zero_grad()
        loss_dict["loss"].backward()

        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

        # Optimizer step
        optimizer.step()

    def test_backbone_to_bev(self, device, small_config, random_images, random_camera_params):
        """Test backbone features can be transformed to BEV."""
        intrinsics, extrinsics = random_camera_params
        B = small_config["batch_size"]
        N = small_config["num_cameras"]
        C = small_config["embed_dims"]

        backbone = ResNet50FPN(
            pretrained=False, fpn_out_channels=C, num_fpn_levels=4
        ).to(device)

        gkt = GKT(
            embed_dim=C,
            bev_h=small_config["bev_h"],
            bev_w=small_config["bev_w"],
            num_heads=small_config["num_heads"],
            num_points=4,
            num_z_anchors=2,
            num_layers=1,
            ffn_dim=C * 2,
            input_feat_channels=[C],
        ).to(device)

        with torch.no_grad():
            multi_scale_feats = backbone(random_images)
            # Use only first level for GKT
            bev = gkt([multi_scale_feats[0]], intrinsics, extrinsics)

        assert bev.shape == (B, C, small_config["bev_h"], small_config["bev_w"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
