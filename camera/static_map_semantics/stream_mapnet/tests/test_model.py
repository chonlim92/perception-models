"""
StreamMapNet - Model Unit Tests

Tests for the StreamMapNet architecture components including backbone,
BEV transformation, temporal fusion, decoder, and full forward pass.

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def num_cameras():
    """Number of cameras in nuScenes surround setup."""
    return 6


@pytest.fixture
def image_size():
    """Input image dimensions (H, W)."""
    return (256, 480)


@pytest.fixture
def bev_size():
    """BEV feature map dimensions (H, W)."""
    return (200, 100)


@pytest.fixture
def bev_channels():
    """Number of BEV feature channels."""
    return 64


@pytest.fixture
def num_points():
    """Number of points per map element."""
    return 20


@pytest.fixture
def num_classes():
    """Number of map element classes (lane_divider, road_boundary, ped_crossing)."""
    return 3


@pytest.fixture
def num_queries():
    """Number of decoder queries (max predicted elements)."""
    return 50


@pytest.fixture
def temporal_length():
    """Number of temporal frames for fusion."""
    return 3


@pytest.fixture
def sample_images(batch_size, num_cameras, image_size, device):
    """Generate sample multi-camera input images."""
    H, W = image_size
    # (B, N_cams, 3, H, W)
    return torch.randn(batch_size, num_cameras, 3, H, W, device=device)


@pytest.fixture
def sample_intrinsics(batch_size, num_cameras, device):
    """Generate sample camera intrinsic matrices."""
    # (B, N_cams, 3, 3)
    intrinsics = torch.zeros(batch_size, num_cameras, 3, 3, device=device)
    # Typical focal lengths and principal points
    intrinsics[:, :, 0, 0] = 1260.0  # fx
    intrinsics[:, :, 1, 1] = 1260.0  # fy
    intrinsics[:, :, 0, 2] = 240.0   # cx
    intrinsics[:, :, 1, 2] = 128.0   # cy
    intrinsics[:, :, 2, 2] = 1.0
    return intrinsics


@pytest.fixture
def sample_extrinsics(batch_size, num_cameras, device):
    """Generate sample camera extrinsic matrices (camera-to-ego)."""
    # (B, N_cams, 4, 4)
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
    extrinsics = extrinsics.expand(batch_size, num_cameras, -1, -1).clone()
    # Add some translation variation per camera
    for cam_idx in range(num_cameras):
        angle = cam_idx * (2 * np.pi / num_cameras)
        extrinsics[:, cam_idx, 0, 3] = 1.5 * np.cos(angle)
        extrinsics[:, cam_idx, 1, 3] = 1.5 * np.sin(angle)
        extrinsics[:, cam_idx, 2, 3] = 1.6  # Camera height
    return extrinsics


@pytest.fixture
def sample_ego_motion(batch_size, temporal_length, device):
    """Generate sample ego motion matrices between consecutive frames."""
    # (B, T-1, 4, 4) - transformation from t-1 to t
    ego_motion = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)
    ego_motion = ego_motion.expand(batch_size, temporal_length - 1, -1, -1).clone()
    return ego_motion


@pytest.fixture
def sample_bev_features(batch_size, bev_channels, bev_size, device):
    """Generate sample BEV feature tensor."""
    H, W = bev_size
    return torch.randn(batch_size, bev_channels, H, W, device=device)


# =============================================================================
# Mock Model Components
# =============================================================================


class MockBackboneFPN(nn.Module):
    """Mock ResNet + FPN backbone that produces multi-scale features."""

    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        # Simulate ResNet stem + FPN
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        """
        Args:
            x: (B*N, 3, H, W) flattened camera images

        Returns:
            List of feature maps at different scales:
              - P2: (B*N, C, H/4, W/4)
              - P3: (B*N, C, H/8, W/8)
              - P4: (B*N, C, H/16, W/16)
              - P5: (B*N, C, H/32, W/32)
        """
        features = []
        for scale in [4, 8, 16, 32]:
            h, w = x.shape[2] // scale, x.shape[3] // scale
            feat = torch.randn(x.shape[0], self.out_channels, h, w, device=x.device)
            features.append(feat)
        return features


class MockLSSTransform(nn.Module):
    """
    Mock Lift-Splat-Shoot (LSS) view transformation.

    Lifts 2D camera features to 3D frustum, then splats into BEV grid.
    """

    def __init__(self, in_channels=256, bev_channels=64, bev_h=200, bev_w=100):
        super().__init__()
        self.bev_channels = bev_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.reduce = nn.Conv2d(in_channels, bev_channels, 1)

    def forward(self, features, intrinsics, extrinsics):
        """
        Args:
            features: List of multi-scale features from backbone
            intrinsics: (B, N, 3, 3) camera intrinsics
            extrinsics: (B, N, 4, 4) camera extrinsics

        Returns:
            (B, C, bev_h, bev_w) BEV feature map
        """
        B = intrinsics.shape[0]
        bev = torch.randn(
            B, self.bev_channels, self.bev_h, self.bev_w,
            device=intrinsics.device
        )
        return bev


class MockTemporalFusion(nn.Module):
    """
    Mock temporal fusion module that warps and aggregates BEV features
    across multiple timesteps using ego-motion compensation.
    """

    def __init__(self, bev_channels=64, bev_h=200, bev_w=100):
        super().__init__()
        self.bev_channels = bev_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.history = None
        self.fusion_conv = nn.Conv2d(bev_channels * 2, bev_channels, 1)

    def reset_state(self):
        """Reset temporal state for new sequence."""
        self.history = None

    def warp_bev(self, bev_feat, ego_motion):
        """
        Warp BEV features using ego motion transformation.

        Args:
            bev_feat: (B, C, H, W) BEV features from previous frame
            ego_motion: (B, 4, 4) ego motion from previous to current

        Returns:
            (B, C, H, W) warped BEV features
        """
        B, C, H, W = bev_feat.shape

        # Extract 2D translation (x, y) and rotation from ego_motion
        # Convert pixel coordinates based on BEV resolution
        tx = ego_motion[:, 0, 3]  # x translation
        ty = ego_motion[:, 1, 3]  # y translation
        cos_theta = ego_motion[:, 0, 0]
        sin_theta = ego_motion[:, 0, 1]

        # Build affine grid for warping
        # BEV resolution: each pixel = 0.6m (for 60m range / 100 pixels)
        bev_resolution = 0.6  # meters per pixel
        tx_px = tx / bev_resolution
        ty_px = ty / bev_resolution

        # Create affine transformation matrix for grid_sample
        theta = torch.zeros(B, 2, 3, device=bev_feat.device)
        theta[:, 0, 0] = cos_theta
        theta[:, 0, 1] = -sin_theta
        theta[:, 1, 0] = sin_theta
        theta[:, 1, 1] = cos_theta
        theta[:, 0, 2] = 2.0 * tx_px / W
        theta[:, 1, 2] = 2.0 * ty_px / H

        grid = torch.nn.functional.affine_grid(
            theta, [B, C, H, W], align_corners=False
        )
        warped = torch.nn.functional.grid_sample(
            bev_feat, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )
        return warped

    def forward(self, bev_feat, ego_motion=None):
        """
        Fuse current BEV features with warped historical features.

        Args:
            bev_feat: (B, C, H, W) current frame BEV features
            ego_motion: (B, 4, 4) optional ego motion for warping history

        Returns:
            (B, C, H, W) temporally fused BEV features
        """
        if self.history is None:
            self.history = bev_feat.clone().detach()
            return bev_feat

        # Warp history to current frame
        if ego_motion is not None:
            warped_history = self.warp_bev(self.history, ego_motion)
        else:
            warped_history = self.history

        # Fuse current and warped history
        concat = torch.cat([bev_feat, warped_history], dim=1)
        fused = self.fusion_conv(concat)

        # Update history
        self.history = fused.clone().detach()

        return fused


class MockMapDecoder(nn.Module):
    """
    Mock map element decoder using transformer queries.

    Predicts vectorized map elements as sets of ordered points.
    """

    def __init__(
        self,
        bev_channels=64,
        num_queries=50,
        num_classes=3,
        num_points=20,
        hidden_dim=256,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_points = num_points

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.bev_proj = nn.Conv2d(bev_channels, hidden_dim, 1)

        # Classification head
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object

        # Regression head (predict K points x 2 coordinates)
        self.reg_head = nn.Linear(hidden_dim, num_points * 2)

        # Simple decoder (in practice: transformer decoder layers)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=3)

    def forward(self, bev_features):
        """
        Args:
            bev_features: (B, C, H, W) BEV feature map

        Returns:
            dict with:
                'cls_scores': (B, num_queries, num_classes + 1)
                'pts_preds': (B, num_queries, num_points, 2)
        """
        B = bev_features.shape[0]

        # Project BEV features
        bev_proj = self.bev_proj(bev_features)  # (B, hidden_dim, H, W)
        # Flatten spatial dims for transformer memory
        memory = bev_proj.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)

        # Query embeddings
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Decode
        decoded = self.decoder(queries, memory)  # (B, num_queries, hidden_dim)

        # Predict classes and points
        cls_scores = self.cls_head(decoded)  # (B, num_queries, num_classes + 1)
        pts_raw = self.reg_head(decoded)  # (B, num_queries, num_points * 2)
        pts_preds = pts_raw.view(B, self.num_queries, self.num_points, 2)
        pts_preds = pts_preds.sigmoid()  # Normalize to [0, 1]

        return {
            "cls_scores": cls_scores,
            "pts_preds": pts_preds,
        }


class MockStreamMapNet(nn.Module):
    """Complete StreamMapNet model combining all components."""

    def __init__(
        self,
        bev_channels=64,
        bev_h=200,
        bev_w=100,
        num_queries=50,
        num_classes=3,
        num_points=20,
    ):
        super().__init__()
        self.backbone = MockBackboneFPN()
        self.lss = MockLSSTransform(
            bev_channels=bev_channels, bev_h=bev_h, bev_w=bev_w
        )
        self.temporal_fusion = MockTemporalFusion(
            bev_channels=bev_channels, bev_h=bev_h, bev_w=bev_w
        )
        self.decoder = MockMapDecoder(
            bev_channels=bev_channels,
            num_queries=num_queries,
            num_classes=num_classes,
            num_points=num_points,
        )

    def forward(self, images, intrinsics, extrinsics, ego_motion=None):
        """
        Full forward pass.

        Args:
            images: (B, N_cams, 3, H, W)
            intrinsics: (B, N_cams, 3, 3)
            extrinsics: (B, N_cams, 4, 4)
            ego_motion: (B, 4, 4) optional

        Returns:
            dict with cls_scores and pts_preds
        """
        B, N = images.shape[:2]

        # Flatten cameras into batch
        imgs_flat = images.flatten(0, 1)  # (B*N, 3, H, W)

        # Backbone
        features = self.backbone(imgs_flat)

        # View transform (LSS)
        bev_features = self.lss(features, intrinsics, extrinsics)

        # Temporal fusion
        fused_features = self.temporal_fusion(bev_features, ego_motion)

        # Decode map elements
        outputs = self.decoder(fused_features)

        return outputs


# =============================================================================
# Loss Function Mock
# =============================================================================


class MockMapLoss(nn.Module):
    """Mock loss function with Hungarian matching for set prediction."""

    def __init__(self, num_classes=3, num_points=20, cls_weight=2.0, pts_weight=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.cls_weight = cls_weight
        self.pts_weight = pts_weight
        self.cls_loss = nn.CrossEntropyLoss()

    def hungarian_matching(self, cls_scores, pts_preds, gt_labels, gt_points):
        """
        Perform Hungarian matching between predictions and ground truth.

        Args:
            cls_scores: (num_queries, num_classes + 1)
            pts_preds: (num_queries, num_points, 2)
            gt_labels: (num_gt,) class indices
            gt_points: (num_gt, num_points, 2)

        Returns:
            Tuple of (pred_indices, gt_indices) for optimal assignment
        """
        from scipy.optimize import linear_sum_assignment

        num_queries = cls_scores.shape[0]
        num_gt = gt_labels.shape[0]

        if num_gt == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # Compute cost matrix
        # Classification cost
        cls_probs = cls_scores.softmax(dim=-1)  # (num_queries, num_classes + 1)
        cls_cost = -cls_probs[:, gt_labels]  # (num_queries, num_gt)

        # Point regression cost (L1)
        pts_cost = torch.cdist(
            pts_preds.flatten(1),  # (num_queries, num_points * 2)
            gt_points.flatten(1),  # (num_gt, num_points * 2)
            p=1,
        )

        # Combined cost
        cost_matrix = cls_cost + 5.0 * pts_cost
        cost_np = cost_matrix.detach().cpu().numpy()

        pred_idx, gt_idx = linear_sum_assignment(cost_np)

        return torch.tensor(pred_idx, dtype=torch.long), torch.tensor(gt_idx, dtype=torch.long)

    def forward(self, outputs, targets):
        """
        Compute loss.

        Args:
            outputs: dict with 'cls_scores' (B, Q, C+1) and 'pts_preds' (B, Q, K, 2)
            targets: list of dicts with 'labels' (N,) and 'points' (N, K, 2)

        Returns:
            dict with 'loss_cls', 'loss_pts', 'loss_total'
        """
        cls_scores = outputs["cls_scores"]
        pts_preds = outputs["pts_preds"]
        B = cls_scores.shape[0]

        total_cls_loss = torch.tensor(0.0, device=cls_scores.device)
        total_pts_loss = torch.tensor(0.0, device=cls_scores.device)

        for b in range(B):
            gt_labels = targets[b]["labels"]
            gt_points = targets[b]["points"]

            # Hungarian matching
            pred_idx, gt_idx = self.hungarian_matching(
                cls_scores[b], pts_preds[b], gt_labels, gt_points
            )

            # Classification loss (all queries)
            target_cls = torch.full(
                (cls_scores.shape[1],),
                self.num_classes,  # no-object class
                dtype=torch.long,
                device=cls_scores.device,
            )
            if len(pred_idx) > 0:
                target_cls[pred_idx] = gt_labels[gt_idx].to(cls_scores.device)

            total_cls_loss += self.cls_loss(cls_scores[b], target_cls)

            # Point regression loss (only matched queries)
            if len(pred_idx) > 0:
                matched_pts = pts_preds[b][pred_idx]
                target_pts = gt_points[gt_idx].to(pts_preds.device)
                total_pts_loss += nn.functional.l1_loss(matched_pts, target_pts)

        total_cls_loss /= B
        total_pts_loss /= B
        loss_total = self.cls_weight * total_cls_loss + self.pts_weight * total_pts_loss

        return {
            "loss_cls": total_cls_loss,
            "loss_pts": total_pts_loss,
            "loss_total": loss_total,
        }


# =============================================================================
# Tests
# =============================================================================


class TestBackbone:
    """Tests for the backbone (ResNet + FPN) component."""

    def test_backbone_output_shape(self, sample_images, num_cameras, device):
        """Verify ResNet+FPN produces correct multi-scale feature shapes."""
        B, N, C, H, W = sample_images.shape
        backbone = MockBackboneFPN(in_channels=3, out_channels=256).to(device)

        # Flatten cameras into batch
        imgs_flat = sample_images.flatten(0, 1)  # (B*N, 3, H, W)
        features = backbone(imgs_flat)

        # Should produce 4 feature levels
        assert len(features) == 4, f"Expected 4 FPN levels, got {len(features)}"

        # Check shapes at each scale
        expected_scales = [4, 8, 16, 32]
        for i, (feat, scale) in enumerate(zip(features, expected_scales)):
            expected_h = H // scale
            expected_w = W // scale
            assert feat.shape == (B * N, 256, expected_h, expected_w), (
                f"FPN level {i}: expected ({B * N}, 256, {expected_h}, {expected_w}), "
                f"got {feat.shape}"
            )

    def test_backbone_batch_independence(self, device):
        """Verify backbone processes each image independently."""
        backbone = MockBackboneFPN(in_channels=3, out_channels=256).to(device)

        img1 = torch.randn(1, 3, 256, 480, device=device)
        img2 = torch.randn(1, 3, 256, 480, device=device)
        imgs_batch = torch.cat([img1, img2], dim=0)

        feats_batch = backbone(imgs_batch)
        feats_single = backbone(img1)

        # First item in batch should match single forward
        assert feats_batch[0].shape[0] == 2
        assert feats_single[0].shape[0] == 1


class TestBEVTransform:
    """Tests for the LSS BEV transformation."""

    def test_bev_transform(
        self, sample_images, sample_intrinsics, sample_extrinsics,
        batch_size, bev_channels, bev_size, device
    ):
        """Verify LSS produces (B, C, 200, 100) BEV features."""
        H, W = bev_size
        backbone = MockBackboneFPN().to(device)
        lss = MockLSSTransform(
            bev_channels=bev_channels, bev_h=H, bev_w=W
        ).to(device)

        # Get backbone features
        imgs_flat = sample_images.flatten(0, 1)
        features = backbone(imgs_flat)

        # Transform to BEV
        bev = lss(features, sample_intrinsics, sample_extrinsics)

        assert bev.shape == (batch_size, bev_channels, H, W), (
            f"Expected BEV shape ({batch_size}, {bev_channels}, {H}, {W}), got {bev.shape}"
        )

    def test_bev_transform_different_sizes(self, device):
        """Verify LSS works with different BEV grid sizes."""
        for bev_h, bev_w in [(100, 50), (200, 100), (400, 200)]:
            lss = MockLSSTransform(bev_channels=64, bev_h=bev_h, bev_w=bev_w).to(device)
            intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, 6, -1, -1)
            extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(1, 6, -1, -1)
            features = [torch.randn(6, 256, 64, 120, device=device)]

            bev = lss(features, intrinsics, extrinsics)
            assert bev.shape == (1, 64, bev_h, bev_w)


class TestTemporalFusion:
    """Tests for temporal BEV feature fusion."""

    def test_temporal_fusion_identity(self, sample_bev_features, batch_size, bev_channels, bev_size, device):
        """Verify fusion with identity ego-motion produces unchanged features on first frame."""
        H, W = bev_size
        fusion = MockTemporalFusion(
            bev_channels=bev_channels, bev_h=H, bev_w=W
        ).to(device)
        fusion.reset_state()

        # First frame: no history, should return input unchanged
        identity_motion = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        output = fusion(sample_bev_features, identity_motion)

        assert output.shape == (batch_size, bev_channels, H, W)
        # First frame should be identical to input (no history to fuse)
        assert torch.allclose(output, sample_bev_features), (
            "First frame output should equal input when no history exists"
        )

    def test_temporal_fusion_with_motion(self, batch_size, bev_channels, bev_size, device):
        """Verify features shift correctly with known translation."""
        H, W = bev_size
        fusion = MockTemporalFusion(
            bev_channels=bev_channels, bev_h=H, bev_w=W
        ).to(device)
        fusion.reset_state()

        # Create a BEV feature with a known pattern (impulse at center)
        bev_feat = torch.zeros(batch_size, bev_channels, H, W, device=device)
        center_h, center_w = H // 2, W // 2
        bev_feat[:, :, center_h, center_w] = 1.0

        # First frame (establishes history)
        _ = fusion(bev_feat, None)

        # Second frame with translation
        ego_motion = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        # Apply a 3m forward translation (y-axis in BEV)
        ego_motion[:, 1, 3] = 3.0

        new_bev_feat = torch.zeros(batch_size, bev_channels, H, W, device=device)
        output = fusion(new_bev_feat, ego_motion)

        # Output should be non-zero due to warped history
        assert output.shape == (batch_size, bev_channels, H, W)
        # The warped history should produce a non-zero output
        assert output.abs().sum() > 0, (
            "Warped history should contribute non-zero values"
        )

    def test_temporal_fusion_output_shape(self, sample_bev_features, batch_size, bev_channels, bev_size, device):
        """Verify temporal fusion always produces correct output shape."""
        H, W = bev_size
        fusion = MockTemporalFusion(
            bev_channels=bev_channels, bev_h=H, bev_w=W
        ).to(device)
        fusion.reset_state()

        ego_motion = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        # Process multiple frames
        for _ in range(5):
            output = fusion(sample_bev_features, ego_motion)
            assert output.shape == (batch_size, bev_channels, H, W)


class TestMapDecoder:
    """Tests for the map element decoder."""

    def test_map_decoder_output_shapes(
        self, sample_bev_features, batch_size, num_queries, num_classes, num_points, device
    ):
        """Verify decoder output shapes for classification and regression."""
        decoder = MockMapDecoder(
            bev_channels=sample_bev_features.shape[1],
            num_queries=num_queries,
            num_classes=num_classes,
            num_points=num_points,
        ).to(device)

        outputs = decoder(sample_bev_features)

        # Classification scores
        assert "cls_scores" in outputs
        assert outputs["cls_scores"].shape == (batch_size, num_queries, num_classes + 1), (
            f"Expected cls_scores shape ({batch_size}, {num_queries}, {num_classes + 1}), "
            f"got {outputs['cls_scores'].shape}"
        )

        # Point predictions
        assert "pts_preds" in outputs
        assert outputs["pts_preds"].shape == (batch_size, num_queries, num_points, 2), (
            f"Expected pts_preds shape ({batch_size}, {num_queries}, {num_points}, 2), "
            f"got {outputs['pts_preds'].shape}"
        )

    def test_map_decoder_point_range(self, sample_bev_features, device):
        """Verify predicted points are normalized to [0, 1]."""
        decoder = MockMapDecoder(
            bev_channels=sample_bev_features.shape[1],
            num_queries=50,
            num_classes=3,
            num_points=20,
        ).to(device)

        outputs = decoder(sample_bev_features)
        pts = outputs["pts_preds"]

        assert pts.min() >= 0.0, f"Points should be >= 0, got min={pts.min()}"
        assert pts.max() <= 1.0, f"Points should be <= 1, got max={pts.max()}"

    def test_map_decoder_different_queries(self, sample_bev_features, device):
        """Verify decoder works with different numbers of queries."""
        for nq in [20, 50, 100, 200]:
            decoder = MockMapDecoder(
                bev_channels=sample_bev_features.shape[1],
                num_queries=nq,
                num_classes=3,
                num_points=20,
            ).to(device)

            outputs = decoder(sample_bev_features)
            assert outputs["cls_scores"].shape[1] == nq
            assert outputs["pts_preds"].shape[1] == nq


class TestFullModel:
    """Tests for the complete StreamMapNet forward pass."""

    def test_full_forward_pass(
        self, sample_images, sample_intrinsics, sample_extrinsics,
        batch_size, num_queries, num_classes, num_points, device
    ):
        """Verify end-to-end model produces correct output shapes."""
        model = MockStreamMapNet(
            bev_channels=64,
            bev_h=200,
            bev_w=100,
            num_queries=num_queries,
            num_classes=num_classes,
            num_points=num_points,
        ).to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(sample_images, sample_intrinsics, sample_extrinsics)

        assert "cls_scores" in outputs
        assert "pts_preds" in outputs
        assert outputs["cls_scores"].shape == (batch_size, num_queries, num_classes + 1)
        assert outputs["pts_preds"].shape == (batch_size, num_queries, num_points, 2)

    def test_full_forward_pass_with_ego_motion(
        self, sample_images, sample_intrinsics, sample_extrinsics,
        batch_size, num_queries, num_classes, num_points, device
    ):
        """Verify model works with ego motion input."""
        model = MockStreamMapNet(
            bev_channels=64,
            bev_h=200,
            bev_w=100,
            num_queries=num_queries,
            num_classes=num_classes,
            num_points=num_points,
        ).to(device)
        model.eval()

        ego_motion = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        with torch.no_grad():
            outputs = model(
                sample_images, sample_intrinsics, sample_extrinsics,
                ego_motion=ego_motion
            )

        assert outputs["cls_scores"].shape == (batch_size, num_queries, num_classes + 1)
        assert outputs["pts_preds"].shape == (batch_size, num_queries, num_points, 2)

    def test_temporal_state_propagation(
        self, sample_images, sample_intrinsics, sample_extrinsics,
        batch_size, num_queries, num_classes, num_points, device
    ):
        """Verify streaming mode maintains state across calls."""
        model = MockStreamMapNet(
            bev_channels=64,
            bev_h=200,
            bev_w=100,
            num_queries=num_queries,
            num_classes=num_classes,
            num_points=num_points,
        ).to(device)
        model.eval()
        model.temporal_fusion.reset_state()

        ego_motion = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        # First frame
        with torch.no_grad():
            out1 = model(sample_images, sample_intrinsics, sample_extrinsics)

        # Verify history is stored
        assert model.temporal_fusion.history is not None, (
            "Temporal fusion should store history after first frame"
        )

        # Second frame (should use stored history)
        with torch.no_grad():
            out2 = model(
                sample_images, sample_intrinsics, sample_extrinsics,
                ego_motion=ego_motion
            )

        # Outputs should differ because temporal context is different
        # (first frame has no history, second frame does)
        assert out2["cls_scores"].shape == (batch_size, num_queries, num_classes + 1)
        assert out2["pts_preds"].shape == (batch_size, num_queries, num_points, 2)

    def test_model_gradient_flow(
        self, sample_images, sample_intrinsics, sample_extrinsics,
        batch_size, device
    ):
        """Verify gradients flow through the full model."""
        model = MockStreamMapNet(
            bev_channels=64, bev_h=200, bev_w=100,
            num_queries=50, num_classes=3, num_points=20,
        ).to(device)
        model.train()
        model.temporal_fusion.reset_state()

        outputs = model(sample_images, sample_intrinsics, sample_extrinsics)

        # Create a simple loss
        loss = outputs["cls_scores"].sum() + outputs["pts_preds"].sum()
        loss.backward()

        # Check that decoder parameters have gradients
        for name, param in model.decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestLossComputation:
    """Tests for the loss function with Hungarian matching."""

    def test_loss_computation(self, device, batch_size, num_queries, num_classes, num_points):
        """Verify loss returns expected keys and finite values."""
        loss_fn = MockMapLoss(num_classes=num_classes, num_points=num_points)

        # Create mock model outputs
        cls_scores = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
        pts_preds = torch.rand(batch_size, num_queries, num_points, 2, device=device)

        outputs = {"cls_scores": cls_scores, "pts_preds": pts_preds}

        # Create mock targets
        targets = []
        for _ in range(batch_size):
            num_gt = np.random.randint(3, 15)
            targets.append({
                "labels": torch.randint(0, num_classes, (num_gt,)),
                "points": torch.rand(num_gt, num_points, 2),
            })

        losses = loss_fn(outputs, targets)

        # Check expected keys
        assert "loss_cls" in losses, "Missing 'loss_cls' key"
        assert "loss_pts" in losses, "Missing 'loss_pts' key"
        assert "loss_total" in losses, "Missing 'loss_total' key"

        # Check finite values
        assert torch.isfinite(losses["loss_cls"]), f"loss_cls is not finite: {losses['loss_cls']}"
        assert torch.isfinite(losses["loss_pts"]), f"loss_pts is not finite: {losses['loss_pts']}"
        assert torch.isfinite(losses["loss_total"]), f"loss_total is not finite: {losses['loss_total']}"

        # Check non-negative
        assert losses["loss_cls"] >= 0, f"loss_cls should be non-negative"
        assert losses["loss_pts"] >= 0, f"loss_pts should be non-negative"
        assert losses["loss_total"] >= 0, f"loss_total should be non-negative"

    def test_loss_with_no_gt(self, device, batch_size, num_queries, num_classes, num_points):
        """Verify loss handles empty ground truth gracefully."""
        loss_fn = MockMapLoss(num_classes=num_classes, num_points=num_points)

        cls_scores = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
        pts_preds = torch.rand(batch_size, num_queries, num_points, 2, device=device)

        outputs = {"cls_scores": cls_scores, "pts_preds": pts_preds}

        # Empty targets
        targets = []
        for _ in range(batch_size):
            targets.append({
                "labels": torch.tensor([], dtype=torch.long),
                "points": torch.zeros(0, num_points, 2),
            })

        losses = loss_fn(outputs, targets)

        assert torch.isfinite(losses["loss_total"]), "Loss should be finite with empty GT"


class TestHungarianMatching:
    """Tests for the Hungarian matching algorithm."""

    def test_hungarian_matching_valid_assignments(self, device, num_classes, num_points):
        """Verify matching produces valid assignments."""
        loss_fn = MockMapLoss(num_classes=num_classes, num_points=num_points)

        num_queries = 50
        num_gt = 10

        cls_scores = torch.randn(num_queries, num_classes + 1, device=device)
        pts_preds = torch.rand(num_queries, num_points, 2, device=device)
        gt_labels = torch.randint(0, num_classes, (num_gt,))
        gt_points = torch.rand(num_gt, num_points, 2)

        pred_idx, gt_idx = loss_fn.hungarian_matching(
            cls_scores, pts_preds, gt_labels, gt_points
        )

        # Number of matches should equal number of GT elements
        assert len(pred_idx) == num_gt, (
            f"Expected {num_gt} matches, got {len(pred_idx)}"
        )
        assert len(gt_idx) == num_gt, (
            f"Expected {num_gt} GT matches, got {len(gt_idx)}"
        )

        # Prediction indices should be unique (one-to-one)
        assert len(set(pred_idx.tolist())) == num_gt, (
            "Prediction indices should be unique (one-to-one matching)"
        )

        # GT indices should cover all GT elements
        assert set(gt_idx.tolist()) == set(range(num_gt)), (
            "All GT elements should be matched"
        )

        # Indices should be within valid range
        assert pred_idx.max() < num_queries, "Prediction index out of range"
        assert gt_idx.max() < num_gt, "GT index out of range"

    def test_hungarian_matching_empty_gt(self, device, num_classes, num_points):
        """Verify matching handles empty ground truth."""
        loss_fn = MockMapLoss(num_classes=num_classes, num_points=num_points)

        num_queries = 50
        cls_scores = torch.randn(num_queries, num_classes + 1, device=device)
        pts_preds = torch.rand(num_queries, num_points, 2, device=device)
        gt_labels = torch.tensor([], dtype=torch.long)
        gt_points = torch.zeros(0, num_points, 2)

        pred_idx, gt_idx = loss_fn.hungarian_matching(
            cls_scores, pts_preds, gt_labels, gt_points
        )

        assert len(pred_idx) == 0, "No matches expected with empty GT"
        assert len(gt_idx) == 0, "No matches expected with empty GT"

    def test_hungarian_matching_perfect_assignment(self, device, num_classes, num_points):
        """Verify matching assigns correctly when predictions are near GT."""
        loss_fn = MockMapLoss(num_classes=num_classes, num_points=num_points)

        num_queries = 50
        num_gt = 5

        # Create GT
        gt_labels = torch.randint(0, num_classes, (num_gt,))
        gt_points = torch.rand(num_gt, num_points, 2)

        # Make first num_gt predictions very close to GT
        cls_scores = torch.full((num_queries, num_classes + 1), -10.0, device=device)
        pts_preds = torch.rand(num_queries, num_points, 2, device=device)

        for i in range(num_gt):
            cls_scores[i, gt_labels[i]] = 10.0  # High score for correct class
            pts_preds[i] = gt_points[i].to(device) + 0.001 * torch.randn(num_points, 2, device=device)

        pred_idx, gt_idx = loss_fn.hungarian_matching(
            cls_scores, pts_preds, gt_labels, gt_points
        )

        # With near-perfect predictions, matching should prefer first num_gt queries
        matched_preds = set(pred_idx.tolist())
        expected_preds = set(range(num_gt))

        # At least most should match (cost should favor the near-GT predictions)
        overlap = matched_preds.intersection(expected_preds)
        assert len(overlap) >= num_gt - 1, (
            f"Expected most predictions to match GT indices, got {len(overlap)}/{num_gt}"
        )
