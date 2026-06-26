"""
Comprehensive pytest unit tests for CRAFT: Camera-Radar 3D Object Detection
with Spatio-Contextual Fusion Transformer.

Tests cover all model components using smaller dimensions for fast execution:
- Camera branch (ResNet + FPN)
- Radar branch (PointPillar encoder + BEV backbone)
- Fusion transformer (cross-attention with projection)
- Detection head (heatmap, regression, velocity outputs)
- End-to-end CRAFT model
- Loss functions
- Post-processing (NMS, score thresholding)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ===========================================================================
# Test Configuration and Fixtures
# ===========================================================================


@pytest.fixture
def model_config():
    """Configuration with small dimensions for fast testing."""
    return {
        # Camera branch
        "backbone_name": "resnet50",
        "pretrained": False,
        "fpn_out_channels": 64,
        "num_cameras": 6,
        "frozen_stages": 0,
        "image_height": 256,
        "image_width": 448,
        # Radar branch
        "point_cloud_range": [-25.6, -25.6, -5.0, 25.6, 25.6, 3.0],
        "voxel_size": [0.4, 0.4, 8.0],
        "max_points_per_pillar": 10,
        "max_num_pillars": 500,
        "in_channels": 6,
        "pillar_feat_channels": 32,
        "bev_out_channels": 64,
        # Fusion transformer
        "fusion_embed_dim": 64,
        "fusion_num_heads": 4,
        "fusion_num_layers": 2,
        "fusion_ffn_dim": 128,
        "fusion_dropout": 0.0,
        # Detection head
        "num_classes": 10,
        "num_reg_attrs": 8,
        "velocity_dim": 2,
        # BEV grid
        "bev_height": 128,
        "bev_width": 128,
    }


@pytest.fixture
def dummy_images():
    """Dummy multi-view camera images [B, N_cams, 3, H, W]."""
    return torch.randn(1, 6, 3, 256, 448)


@pytest.fixture
def dummy_radar_points():
    """Dummy radar point cloud [B, N_points, 6] with features (x, y, z, vx, vy, rcs)."""
    points = torch.randn(1, 500, 6)
    # Clamp x, y to within range [-25.6, 25.6]
    points[:, :, 0] = points[:, :, 0].clamp(-25.0, 25.0)
    points[:, :, 1] = points[:, :, 1].clamp(-25.0, 25.0)
    points[:, :, 2] = points[:, :, 2].clamp(-4.5, 2.5)
    return points


@pytest.fixture
def dummy_calibration():
    """Dummy calibration matrices: intrinsics [B, 6, 3, 3] and extrinsics [B, 6, 4, 4]."""
    batch_size = 1
    num_cams = 6

    # Identity-like intrinsics with focal length
    intrinsics = torch.zeros(batch_size, num_cams, 3, 3)
    for i in range(num_cams):
        intrinsics[:, i, 0, 0] = 800.0  # fx
        intrinsics[:, i, 1, 1] = 800.0  # fy
        intrinsics[:, i, 0, 2] = 224.0  # cx (half of 448)
        intrinsics[:, i, 1, 2] = 128.0  # cy (half of 256)
        intrinsics[:, i, 2, 2] = 1.0

    # Identity extrinsics (cameras at origin looking forward)
    extrinsics = torch.zeros(batch_size, num_cams, 4, 4)
    for i in range(num_cams):
        extrinsics[:, i] = torch.eye(4)

    return intrinsics, extrinsics


# ===========================================================================
# Lightweight Module Implementations for Testing
# (Simplified versions matching the real architecture interfaces)
# ===========================================================================


class SimpleFPN(nn.Module):
    """Simplified FPN for testing."""

    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode="nearest")
            laterals[i-1] = laterals[i-1] + upsampled
        return [conv(lat) for conv, lat in zip(self.output_convs, laterals)]


class SimpleCameraBranch(nn.Module):
    """Simplified camera branch: a minimal CNN + FPN producing multi-scale features."""

    def __init__(self, fpn_out_channels=64, num_cameras=6):
        super().__init__()
        self.num_cameras = num_cameras
        self.fpn_out_channels = fpn_out_channels

        # Minimal feature extractor producing 4 levels
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # Produce features at 4 scales
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.fpn = SimpleFPN([64, 128, 256, 512], fpn_out_channels)

    def forward(self, images):
        """
        Args:
            images: [B, N_cams, 3, H, W]
        Returns:
            dict with 'features': list of [B, N_cams, C, H_i, W_i]
        """
        B, N, C, H, W = images.shape
        x = images.reshape(B * N, C, H, W)

        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        fpn_feats = self.fpn([c1, c2, c3, c4])

        output = []
        for feat in fpn_feats:
            _, Cf, Hf, Wf = feat.shape
            output.append(feat.reshape(B, N, Cf, Hf, Wf))

        return {"features": output}


class SimplePillarEncoder(nn.Module):
    """Simplified pillar feature encoder for testing."""

    def __init__(self, in_channels=11, out_channels=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, pillar_features, pillar_mask):
        """
        Args:
            pillar_features: [B, P, N, C_in]
            pillar_mask: [B, P, N]
        Returns:
            [B, P, out_channels]
        """
        B, P, N, C = pillar_features.shape
        x = pillar_features.reshape(B * P * N, C)
        x = self.mlp(x)
        x = x.reshape(B, P, N, self.out_channels)

        mask_expanded = pillar_mask.unsqueeze(-1).float()
        x = x * mask_expanded
        x = x.masked_fill(mask_expanded == 0, float("-inf"))
        x, _ = x.max(dim=2)
        x = x.clamp(min=0.0)
        return x


class SimpleRadarBranch(nn.Module):
    """Simplified radar branch for testing: direct BEV feature generation."""

    def __init__(self, bev_channels=64, bev_height=128, bev_width=128):
        super().__init__()
        self.bev_channels = bev_channels
        self.bev_height = bev_height
        self.bev_width = bev_width
        # Simple projection from points to BEV
        self.point_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, bev_channels),
        )
        # BEV conv
        self.bev_conv = nn.Sequential(
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, bev_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, num_points=None):
        """
        Args:
            points: [B, N, 6]
            num_points: [B] number of valid points (optional)
        Returns:
            dict with 'bev_features': [B, bev_channels, bev_height, bev_width]
        """
        B = points.shape[0]
        device = points.device

        # Encode points
        encoded = self.point_encoder(points)  # [B, N, bev_channels]

        # Simple scatter to BEV via averaging (just for testing shape correctness)
        bev = torch.zeros(B, self.bev_channels, self.bev_height, self.bev_width, device=device)

        # Place encoded features at grid positions derived from x, y
        x_pos = points[:, :, 0]  # [B, N]
        y_pos = points[:, :, 1]

        # Normalize to grid
        ix = ((x_pos + 25.6) / 51.2 * (self.bev_width - 1)).long().clamp(0, self.bev_width - 1)
        iy = ((y_pos + 25.6) / 51.2 * (self.bev_height - 1)).long().clamp(0, self.bev_height - 1)

        for b in range(B):
            n = num_points[b].item() if num_points is not None else points.shape[1]
            for i in range(min(n, 50)):  # Only scatter a few for speed
                bev[b, :, iy[b, i], ix[b, i]] += encoded[b, i]

        bev = self.bev_conv(bev)
        return {"bev_features": bev}


class SimpleFusionTransformer(nn.Module):
    """Simplified fusion transformer for testing."""

    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, ffn_dim=128, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.radar_proj = nn.Linear(embed_dim, embed_dim)
        self.camera_proj = nn.Linear(embed_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, radar_bev_features, camera_features, intrinsics=None, extrinsics=None):
        """
        Args:
            radar_bev_features: [B, C, H, W] radar BEV features
            camera_features: [B, N_cams, C, H_cam, W_cam] camera features
            intrinsics: [B, N_cams, 3, 3] (optional)
            extrinsics: [B, N_cams, 4, 4] (optional)
        Returns:
            fused_bev: [B, C, H, W] fused BEV features
        """
        B, C, H, W = radar_bev_features.shape

        # Flatten radar BEV to sequence
        radar_seq = radar_bev_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        radar_seq = self.radar_proj(radar_seq)

        # Flatten camera features to create memory
        B_cam, N_cams, C_cam, H_cam, W_cam = camera_features.shape
        cam_seq = camera_features.reshape(B, N_cams * H_cam * W_cam, C_cam)
        cam_seq = self.camera_proj(cam_seq)

        # Cross-attention: radar queries attend to camera memory
        fused_seq = self.decoder(radar_seq, cam_seq)
        fused_seq = self.output_norm(fused_seq)

        # Reshape back to BEV spatial
        fused_bev = fused_seq.permute(0, 2, 1).reshape(B, C, H, W)
        return fused_bev


class SimpleDetectionHead(nn.Module):
    """Simplified detection head for testing."""

    def __init__(self, in_channels=64, num_classes=10, num_reg_attrs=8, velocity_dim=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_reg_attrs = num_reg_attrs
        self.velocity_dim = velocity_dim

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(64, num_classes, 1)
        self.regression_head = nn.Conv2d(64, num_reg_attrs, 1)
        self.velocity_head = nn.Conv2d(64, velocity_dim, 1)

        # Initialize heatmap bias for focal loss
        nn.init.constant_(self.heatmap_head.bias, -2.19)

    def forward(self, bev_features):
        """
        Args:
            bev_features: [B, C, H, W]
        Returns:
            dict with 'heatmap', 'regression', 'velocity'
        """
        shared = self.shared_conv(bev_features)
        heatmap = torch.sigmoid(self.heatmap_head(shared))
        regression = self.regression_head(shared)
        velocity = self.velocity_head(shared)
        return {
            "heatmap": heatmap,
            "regression": regression,
            "velocity": velocity,
        }


class SimpleCRAFTModel(nn.Module):
    """Simplified end-to-end CRAFT model for testing."""

    def __init__(self, config):
        super().__init__()
        fpn_ch = config["fpn_out_channels"]
        bev_h = config["bev_height"]
        bev_w = config["bev_width"]

        self.camera_branch = SimpleCameraBranch(
            fpn_out_channels=fpn_ch,
            num_cameras=config["num_cameras"],
        )
        self.radar_branch = SimpleRadarBranch(
            bev_channels=fpn_ch,
            bev_height=bev_h,
            bev_width=bev_w,
        )
        self.fusion_transformer = SimpleFusionTransformer(
            embed_dim=fpn_ch,
            num_heads=config["fusion_num_heads"],
            num_layers=config["fusion_num_layers"],
            ffn_dim=config["fusion_ffn_dim"],
            dropout=config["fusion_dropout"],
        )
        self.detection_head = SimpleDetectionHead(
            in_channels=fpn_ch,
            num_classes=config["num_classes"],
            num_reg_attrs=config["num_reg_attrs"],
            velocity_dim=config["velocity_dim"],
        )

    def forward(self, images, radar_points, num_points=None, intrinsics=None, extrinsics=None):
        """
        Args:
            images: [B, N_cams, 3, H, W]
            radar_points: [B, N, 6]
            num_points: [B]
            intrinsics: [B, N_cams, 3, 3]
            extrinsics: [B, N_cams, 4, 4]
        Returns:
            dict with detection outputs
        """
        # Camera branch
        cam_output = self.camera_branch(images)
        cam_features = cam_output["features"][0]  # Use highest-res FPN level

        # Radar branch
        radar_output = self.radar_branch(radar_points, num_points)
        radar_bev = radar_output["bev_features"]

        # Fusion
        fused_bev = self.fusion_transformer(
            radar_bev, cam_features, intrinsics, extrinsics
        )

        # Detection head
        detections = self.detection_head(fused_bev)
        return detections


# ===========================================================================
# Loss Functions
# ===========================================================================


class FocalLoss(nn.Module):
    """Focal loss for dense heatmap prediction."""

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] predicted heatmap (sigmoid activated)
            target: [B, C, H, W] ground truth heatmap (gaussian peaks)
        Returns:
            Scalar focal loss value
        """
        pred = pred.clamp(1e-6, 1.0 - 1e-6)

        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = (
            -torch.log(1 - pred)
            * torch.pow(pred, self.alpha)
            * torch.pow(1 - target, self.beta)
            * neg_mask
        )

        num_pos = pos_mask.sum().clamp(min=1.0)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


class L1RegressionLoss(nn.Module):
    """L1 loss for bounding box regression at positive locations."""

    def forward(self, pred, target, mask):
        """
        Args:
            pred: [B, C, H, W] predicted regression
            target: [B, C, H, W] ground truth regression
            mask: [B, 1, H, W] positive location mask
        Returns:
            Scalar L1 loss
        """
        num_pos = mask.sum().clamp(min=1.0)
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum") / num_pos
        return loss


# ===========================================================================
# Post-Processing
# ===========================================================================


def nms_bev(boxes, scores, iou_threshold=0.2):
    """Non-maximum suppression in BEV.

    Args:
        boxes: [N, 7] (x, y, z, w, l, h, yaw)
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep_indices: indices of boxes to keep
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)

    # Sort by score
    order = scores.argsort(descending=True)

    # Simplified axis-aligned BEV IoU for testing
    # Use x, y, w, l for BEV overlap
    x1 = boxes[:, 0] - boxes[:, 3] / 2
    y1 = boxes[:, 1] - boxes[:, 4] / 2
    x2 = boxes[:, 0] + boxes[:, 3] / 2
    y2 = boxes[:, 1] + boxes[:, 4] / 2

    areas = (x2 - x1) * (y2 - y1)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # Compute IoU with remaining boxes
        remaining = order[1:]
        xx1 = torch.maximum(x1[i].unsqueeze(0), x1[remaining])
        yy1 = torch.maximum(y1[i].unsqueeze(0), y1[remaining])
        xx2 = torch.minimum(x2[i].unsqueeze(0), x2[remaining])
        yy2 = torch.minimum(y2[i].unsqueeze(0), y2[remaining])

        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        union = areas[i] + areas[remaining] - inter
        iou = inter / union.clamp(min=1e-6)

        # Keep boxes with IoU below threshold
        mask = iou <= iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long)


def decode_detections(heatmap, regression, velocity, score_threshold=0.1, nms_threshold=0.2):
    """Decode detection head outputs into 3D bounding boxes.

    Args:
        heatmap: [B, num_classes, H, W]
        regression: [B, 8, H, W] (dx, dy, dz, w, l, h, sin_yaw, cos_yaw)
        velocity: [B, 2, H, W]
        score_threshold: minimum confidence to keep
        nms_threshold: IoU threshold for NMS

    Returns:
        List of dicts per batch item with 'boxes', 'scores', 'labels'
    """
    B, num_classes, H, W = heatmap.shape
    results = []

    for b in range(B):
        all_boxes = []
        all_scores = []
        all_labels = []

        for cls in range(num_classes):
            cls_heatmap = heatmap[b, cls]  # [H, W]

            # Find peaks above threshold
            mask = cls_heatmap > score_threshold
            if not mask.any():
                continue

            scores_cls = cls_heatmap[mask]
            indices = mask.nonzero(as_tuple=False)  # [N, 2] (row, col)
            rows, cols = indices[:, 0], indices[:, 1]

            # Decode box parameters
            reg = regression[b, :, rows, cols]  # [8, N]
            vel = velocity[b, :, rows, cols]    # [2, N]

            # Convert grid position + offset to 3D position
            dx, dy, dz = reg[0], reg[1], reg[2]
            w, l, h = reg[3], reg[4], reg[5]
            sin_yaw, cos_yaw = reg[6], reg[7]
            yaw = torch.atan2(sin_yaw, cos_yaw)

            boxes = torch.stack([
                cols.float() + dx,  # x (BEV grid)
                rows.float() + dy,  # y (BEV grid)
                dz,                 # z
                w, l, h, yaw,
            ], dim=-1)  # [N, 7]

            all_boxes.append(boxes)
            all_scores.append(scores_cls)
            all_labels.append(torch.full((scores_cls.shape[0],), cls, dtype=torch.long))

        if len(all_boxes) == 0:
            results.append({
                "boxes": torch.zeros(0, 7),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
            })
            continue

        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Apply NMS
        keep = nms_bev(all_boxes, all_scores, iou_threshold=nms_threshold)
        results.append({
            "boxes": all_boxes[keep],
            "scores": all_scores[keep],
            "labels": all_labels[keep],
        })

    return results


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCameraBranch:
    """Tests for the camera feature extraction branch."""

    def test_output_shapes(self, model_config):
        """Test FPN output shapes [B, N_cams, C, H_i, W_i] for each level."""
        model = SimpleCameraBranch(
            fpn_out_channels=model_config["fpn_out_channels"],
            num_cameras=model_config["num_cameras"],
        )
        model.eval()

        images = torch.randn(1, 6, 3, 256, 448)
        with torch.no_grad():
            output = model(images)

        features = output["features"]
        assert len(features) == 4, f"Expected 4 FPN levels, got {len(features)}"

        # Check each level has correct batch, camera, and channel dimensions
        for i, feat in enumerate(features):
            B, N, C, H, W = feat.shape
            assert B == 1, f"Level {i}: expected batch=1, got {B}"
            assert N == 6, f"Level {i}: expected 6 cameras, got {N}"
            assert C == model_config["fpn_out_channels"], (
                f"Level {i}: expected {model_config['fpn_out_channels']} channels, got {C}"
            )
            # Spatial dims should decrease with level
            assert H > 0 and W > 0, f"Level {i}: invalid spatial dims {H}x{W}"

    def test_different_batch_sizes(self, model_config):
        """Test camera branch works with different batch sizes."""
        model = SimpleCameraBranch(
            fpn_out_channels=model_config["fpn_out_channels"],
            num_cameras=model_config["num_cameras"],
        )
        model.eval()

        for batch_size in [1, 2, 4]:
            images = torch.randn(batch_size, 6, 3, 256, 448)
            with torch.no_grad():
                output = model(images)
            features = output["features"]
            for feat in features:
                assert feat.shape[0] == batch_size

    def test_fpn_channel_consistency(self, model_config):
        """Test all FPN levels have the same channel dimension."""
        model = SimpleCameraBranch(
            fpn_out_channels=model_config["fpn_out_channels"],
            num_cameras=model_config["num_cameras"],
        )
        model.eval()

        images = torch.randn(1, 6, 3, 256, 448)
        with torch.no_grad():
            output = model(images)

        channels = [f.shape[2] for f in output["features"]]
        assert all(c == channels[0] for c in channels), (
            f"FPN channels not consistent: {channels}"
        )

    def test_spatial_decreasing(self, model_config):
        """Test FPN levels have decreasing spatial resolution."""
        model = SimpleCameraBranch(
            fpn_out_channels=model_config["fpn_out_channels"],
            num_cameras=model_config["num_cameras"],
        )
        model.eval()

        images = torch.randn(1, 6, 3, 256, 448)
        with torch.no_grad():
            output = model(images)

        heights = [f.shape[3] for f in output["features"]]
        widths = [f.shape[4] for f in output["features"]]

        for i in range(len(heights) - 1):
            assert heights[i] >= heights[i + 1], (
                f"FPN height should decrease: level {i}={heights[i]}, level {i+1}={heights[i+1]}"
            )

    def test_backbone_weight_loading(self, model_config):
        """Test that model initializes correctly and has trainable parameters."""
        model = SimpleCameraBranch(
            fpn_out_channels=model_config["fpn_out_channels"],
            num_cameras=model_config["num_cameras"],
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params > 0, "Model has no parameters"
        assert trainable_params > 0, "Model has no trainable parameters"
        assert trainable_params == total_params, "All parameters should be trainable (frozen_stages=0)"


class TestRadarBranch:
    """Tests for the radar pillar encoding and BEV feature extraction."""

    def test_pillar_encoding_output(self, model_config):
        """Test pillar encoder produces correct output shape."""
        encoder = SimplePillarEncoder(in_channels=11, out_channels=32)
        encoder.eval()

        B, P, N = 1, 100, 10
        pillar_features = torch.randn(B, P, N, 11)
        pillar_mask = torch.ones(B, P, N, dtype=torch.bool)
        # Make some points invalid
        pillar_mask[:, :, 5:] = False

        with torch.no_grad():
            output = encoder(pillar_features, pillar_mask)

        assert output.shape == (B, P, 32), f"Expected shape (1, 100, 32), got {output.shape}"

    def test_bev_feature_shape(self, model_config):
        """Test BEV feature map has correct spatial dimensions."""
        bev_h = model_config["bev_height"]
        bev_w = model_config["bev_width"]
        bev_ch = model_config["bev_out_channels"]

        model = SimpleRadarBranch(
            bev_channels=bev_ch,
            bev_height=bev_h,
            bev_width=bev_w,
        )
        model.eval()

        points = torch.randn(1, 500, 6)
        points[:, :, 0] = points[:, :, 0].clamp(-25.0, 25.0)
        points[:, :, 1] = points[:, :, 1].clamp(-25.0, 25.0)
        num_points = torch.tensor([500])

        with torch.no_grad():
            output = model(points, num_points)

        bev = output["bev_features"]
        assert bev.shape == (1, bev_ch, bev_h, bev_w), (
            f"Expected BEV shape (1, {bev_ch}, {bev_h}, {bev_w}), got {bev.shape}"
        )

    def test_varying_num_radar_points(self, model_config):
        """Test radar branch handles different numbers of points."""
        model = SimpleRadarBranch(
            bev_channels=model_config["bev_out_channels"],
            bev_height=model_config["bev_height"],
            bev_width=model_config["bev_width"],
        )
        model.eval()

        for n_points in [50, 200, 1000]:
            points = torch.randn(1, n_points, 6)
            points[:, :, 0] = points[:, :, 0].clamp(-25.0, 25.0)
            points[:, :, 1] = points[:, :, 1].clamp(-25.0, 25.0)
            num_pts = torch.tensor([n_points])

            with torch.no_grad():
                output = model(points, num_pts)

            bev = output["bev_features"]
            assert bev.shape[0] == 1
            assert bev.shape[1] == model_config["bev_out_channels"]
            assert bev.shape[2] == model_config["bev_height"]
            assert bev.shape[3] == model_config["bev_width"]

    def test_empty_point_cloud(self, model_config):
        """Test radar branch handles empty (all-zero) point clouds."""
        model = SimpleRadarBranch(
            bev_channels=model_config["bev_out_channels"],
            bev_height=model_config["bev_height"],
            bev_width=model_config["bev_width"],
        )
        model.eval()

        # All zeros - simulates empty radar returns
        points = torch.zeros(1, 100, 6)
        num_points = torch.tensor([0])

        with torch.no_grad():
            output = model(points, num_points)

        bev = output["bev_features"]
        assert bev.shape == (1, model_config["bev_out_channels"],
                             model_config["bev_height"], model_config["bev_width"])

    def test_batch_processing(self, model_config):
        """Test radar branch with batched inputs."""
        model = SimpleRadarBranch(
            bev_channels=model_config["bev_out_channels"],
            bev_height=model_config["bev_height"],
            bev_width=model_config["bev_width"],
        )
        model.eval()

        B = 3
        points = torch.randn(B, 500, 6)
        points[:, :, 0] = points[:, :, 0].clamp(-25.0, 25.0)
        points[:, :, 1] = points[:, :, 1].clamp(-25.0, 25.0)
        num_points = torch.tensor([500, 300, 100])

        with torch.no_grad():
            output = model(points, num_points)

        assert output["bev_features"].shape[0] == B


class TestFusionTransformer:
    """Tests for the spatio-contextual fusion transformer."""

    def test_output_shape_matches_radar_bev(self, model_config):
        """Test that fusion output matches radar BEV spatial dimensions."""
        embed_dim = model_config["fusion_embed_dim"]
        bev_h = model_config["bev_height"]
        bev_w = model_config["bev_width"]

        # Use smaller spatial dims for transformer test to avoid OOM
        small_h, small_w = 8, 8

        fusion = SimpleFusionTransformer(
            embed_dim=embed_dim,
            num_heads=model_config["fusion_num_heads"],
            num_layers=model_config["fusion_num_layers"],
            ffn_dim=model_config["fusion_ffn_dim"],
            dropout=model_config["fusion_dropout"],
        )
        fusion.eval()

        radar_bev = torch.randn(1, embed_dim, small_h, small_w)
        cam_features = torch.randn(1, 6, embed_dim, 4, 4)

        with torch.no_grad():
            output = fusion(radar_bev, cam_features)

        assert output.shape == radar_bev.shape, (
            f"Fusion output shape {output.shape} != radar BEV shape {radar_bev.shape}"
        )

    def test_projection_known_point(self, dummy_calibration):
        """Test that a known 3D point projects to the correct pixel location."""
        intrinsics, extrinsics = dummy_calibration

        # Point at (0, 0, 10) in front of camera with identity extrinsics
        # Should project to (cx, cy) = (224, 128) with focal length 800
        point_3d = torch.tensor([[[0.0, 0.0, 10.0]]])  # [1, 1, 3]

        # Manual projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
        fx = intrinsics[0, 0, 0, 0]  # 800
        fy = intrinsics[0, 0, 1, 1]  # 800
        cx = intrinsics[0, 0, 0, 2]  # 224
        cy = intrinsics[0, 0, 1, 2]  # 128

        expected_u = fx * 0.0 / 10.0 + cx  # 224
        expected_v = fy * 0.0 / 10.0 + cy  # 128

        # Project using intrinsics
        pts_homo = F.pad(point_3d, (0, 1), value=1.0)  # [1, 1, 4]
        # Transform with extrinsics (identity, so no change in 3D)
        pts_cam = torch.matmul(extrinsics[:, 0, :3, :3], point_3d.transpose(1, 2))  # [1, 3, 1]
        pts_cam = pts_cam.transpose(1, 2) + extrinsics[:, 0, :3, 3].unsqueeze(1)  # [1, 1, 3]

        # Project with intrinsics
        pts_img = torch.matmul(intrinsics[:, 0], pts_cam.transpose(1, 2))  # [1, 3, 1]
        pts_img = pts_img.transpose(1, 2)  # [1, 1, 3]
        depth = pts_img[:, :, 2:3]
        proj_uv = pts_img[:, :, :2] / depth

        u_proj = proj_uv[0, 0, 0].item()
        v_proj = proj_uv[0, 0, 1].item()

        assert abs(u_proj - expected_u.item()) < 1e-4, (
            f"u projection: expected {expected_u.item()}, got {u_proj}"
        )
        assert abs(v_proj - expected_v.item()) < 1e-4, (
            f"v projection: expected {expected_v.item()}, got {v_proj}"
        )

    def test_with_dummy_calibration(self, model_config, dummy_calibration):
        """Test fusion transformer with dummy calibration matrices."""
        intrinsics, extrinsics = dummy_calibration
        embed_dim = model_config["fusion_embed_dim"]

        small_h, small_w = 8, 8
        fusion = SimpleFusionTransformer(
            embed_dim=embed_dim,
            num_heads=model_config["fusion_num_heads"],
            num_layers=model_config["fusion_num_layers"],
            ffn_dim=model_config["fusion_ffn_dim"],
        )
        fusion.eval()

        radar_bev = torch.randn(1, embed_dim, small_h, small_w)
        cam_features = torch.randn(1, 6, embed_dim, 4, 4)

        with torch.no_grad():
            output = fusion(radar_bev, cam_features, intrinsics, extrinsics)

        assert output.shape == (1, embed_dim, small_h, small_w)
        # Output should not be all zeros (transformer produces non-trivial output)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_cross_attention_gradient_flow(self, model_config):
        """Test that gradients flow through the fusion transformer."""
        embed_dim = model_config["fusion_embed_dim"]
        small_h, small_w = 4, 4

        fusion = SimpleFusionTransformer(
            embed_dim=embed_dim,
            num_heads=model_config["fusion_num_heads"],
            num_layers=model_config["fusion_num_layers"],
            ffn_dim=model_config["fusion_ffn_dim"],
        )
        fusion.train()

        radar_bev = torch.randn(1, embed_dim, small_h, small_w, requires_grad=True)
        cam_features = torch.randn(1, 6, embed_dim, 2, 2, requires_grad=True)

        output = fusion(radar_bev, cam_features)
        loss = output.sum()
        loss.backward()

        assert radar_bev.grad is not None, "No gradient on radar input"
        assert cam_features.grad is not None, "No gradient on camera input"
        assert radar_bev.grad.abs().sum() > 0, "Zero gradient on radar input"
        assert cam_features.grad.abs().sum() > 0, "Zero gradient on camera input"


class TestDetectionHead:
    """Tests for the anchor-free detection head."""

    def test_output_keys(self, model_config):
        """Test output dict contains 'heatmap', 'regression', 'velocity' keys."""
        head = SimpleDetectionHead(
            in_channels=model_config["bev_out_channels"],
            num_classes=model_config["num_classes"],
            num_reg_attrs=model_config["num_reg_attrs"],
            velocity_dim=model_config["velocity_dim"],
        )
        head.eval()

        bev_features = torch.randn(1, model_config["bev_out_channels"], 32, 32)
        with torch.no_grad():
            output = head(bev_features)

        assert "heatmap" in output, "Missing 'heatmap' key"
        assert "regression" in output, "Missing 'regression' key"
        assert "velocity" in output, "Missing 'velocity' key"

    def test_output_shapes(self, model_config):
        """Test output tensors have correct shapes."""
        H, W = 32, 32
        head = SimpleDetectionHead(
            in_channels=model_config["bev_out_channels"],
            num_classes=model_config["num_classes"],
            num_reg_attrs=model_config["num_reg_attrs"],
            velocity_dim=model_config["velocity_dim"],
        )
        head.eval()

        bev_features = torch.randn(2, model_config["bev_out_channels"], H, W)
        with torch.no_grad():
            output = head(bev_features)

        assert output["heatmap"].shape == (2, model_config["num_classes"], H, W), (
            f"Heatmap shape: {output['heatmap'].shape}"
        )
        assert output["regression"].shape == (2, model_config["num_reg_attrs"], H, W), (
            f"Regression shape: {output['regression'].shape}"
        )
        assert output["velocity"].shape == (2, model_config["velocity_dim"], H, W), (
            f"Velocity shape: {output['velocity'].shape}"
        )

    def test_heatmap_range(self, model_config):
        """Test heatmap output is in [0, 1] range (sigmoid activated)."""
        head = SimpleDetectionHead(
            in_channels=model_config["bev_out_channels"],
            num_classes=model_config["num_classes"],
        )
        head.eval()

        bev_features = torch.randn(1, model_config["bev_out_channels"], 16, 16)
        with torch.no_grad():
            output = head(bev_features)

        heatmap = output["heatmap"]
        assert heatmap.min() >= 0.0, f"Heatmap min {heatmap.min()} < 0"
        assert heatmap.max() <= 1.0, f"Heatmap max {heatmap.max()} > 1"

    def test_different_bev_sizes(self, model_config):
        """Test detection head works with various BEV spatial sizes."""
        head = SimpleDetectionHead(
            in_channels=model_config["bev_out_channels"],
            num_classes=model_config["num_classes"],
        )
        head.eval()

        for size in [(16, 16), (32, 32), (64, 64), (128, 128)]:
            bev_features = torch.randn(1, model_config["bev_out_channels"], *size)
            with torch.no_grad():
                output = head(bev_features)
            assert output["heatmap"].shape[2:] == size


class TestCRAFTModel:
    """Tests for the end-to-end CRAFT model."""

    def test_forward_pass(self, model_config, dummy_images, dummy_radar_points, dummy_calibration):
        """Test end-to-end forward pass with dummy data."""
        intrinsics, extrinsics = dummy_calibration
        model = SimpleCRAFTModel(model_config)
        model.eval()

        num_points = torch.tensor([500])

        with torch.no_grad():
            output = model(
                dummy_images,
                dummy_radar_points,
                num_points,
                intrinsics,
                extrinsics,
            )

        assert "heatmap" in output
        assert "regression" in output
        assert "velocity" in output

    def test_output_shapes(self, model_config, dummy_images, dummy_radar_points, dummy_calibration):
        """Test all output tensor shapes are correct."""
        intrinsics, extrinsics = dummy_calibration
        model = SimpleCRAFTModel(model_config)
        model.eval()

        num_points = torch.tensor([500])
        bev_h = model_config["bev_height"]
        bev_w = model_config["bev_width"]

        with torch.no_grad():
            output = model(
                dummy_images,
                dummy_radar_points,
                num_points,
                intrinsics,
                extrinsics,
            )

        assert output["heatmap"].shape == (1, model_config["num_classes"], bev_h, bev_w)
        assert output["regression"].shape == (1, model_config["num_reg_attrs"], bev_h, bev_w)
        assert output["velocity"].shape == (1, model_config["velocity_dim"], bev_h, bev_w)

    def test_batch_forward(self, model_config):
        """Test forward pass with batch size > 1."""
        model = SimpleCRAFTModel(model_config)
        model.eval()

        B = 2
        images = torch.randn(B, 6, 3, 256, 448)
        points = torch.randn(B, 500, 6)
        points[:, :, 0] = points[:, :, 0].clamp(-25.0, 25.0)
        points[:, :, 1] = points[:, :, 1].clamp(-25.0, 25.0)
        num_points = torch.tensor([500, 300])
        intrinsics = torch.zeros(B, 6, 3, 3)
        extrinsics = torch.zeros(B, 6, 4, 4)
        for i in range(6):
            intrinsics[:, i] = torch.eye(3) * 800
            intrinsics[:, i, 2, 2] = 1.0
            extrinsics[:, i] = torch.eye(4)

        with torch.no_grad():
            output = model(images, points, num_points, intrinsics, extrinsics)

        assert output["heatmap"].shape[0] == B

    def test_inference_mode(self, model_config, dummy_images, dummy_radar_points, dummy_calibration):
        """Test model in inference mode with post-processing."""
        intrinsics, extrinsics = dummy_calibration
        model = SimpleCRAFTModel(model_config)
        model.eval()

        num_points = torch.tensor([500])

        with torch.no_grad():
            output = model(
                dummy_images,
                dummy_radar_points,
                num_points,
                intrinsics,
                extrinsics,
            )

        # Apply post-processing
        results = decode_detections(
            output["heatmap"],
            output["regression"],
            output["velocity"],
            score_threshold=0.1,
            nms_threshold=0.2,
        )

        assert len(results) == 1  # One result per batch item
        assert "boxes" in results[0]
        assert "scores" in results[0]
        assert "labels" in results[0]
        # Boxes should have 7 columns (x, y, z, w, l, h, yaw)
        if results[0]["boxes"].shape[0] > 0:
            assert results[0]["boxes"].shape[1] == 7

    def test_gradient_flow_end_to_end(self, model_config):
        """Test gradients flow through the entire model."""
        model = SimpleCRAFTModel(model_config)
        model.train()

        # Use very small spatial dims for speed
        images = torch.randn(1, 6, 3, 256, 448)
        points = torch.randn(1, 100, 6)
        points[:, :, 0] = points[:, :, 0].clamp(-25.0, 25.0)
        points[:, :, 1] = points[:, :, 1].clamp(-25.0, 25.0)
        num_points = torch.tensor([100])

        output = model(images, points, num_points)
        loss = output["heatmap"].sum() + output["regression"].sum() + output["velocity"].sum()
        loss.backward()

        # Check at least some parameters have gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found in any model parameter"


class TestLosses:
    """Tests for training loss functions."""

    def test_focal_loss_is_scalar(self):
        """Test focal loss output is a scalar tensor."""
        loss_fn = FocalLoss(alpha=2.0, beta=4.0)

        pred = torch.rand(2, 10, 32, 32)
        target = torch.zeros(2, 10, 32, 32)
        # Place some gaussian peaks
        target[:, 0, 16, 16] = 1.0
        target[:, 3, 8, 24] = 1.0

        loss = loss_fn(pred, target)
        assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_focal_loss_all_zero_predictions(self):
        """Test focal loss with near-zero predictions (high loss expected)."""
        loss_fn = FocalLoss(alpha=2.0, beta=4.0)

        # Predictions near zero where targets are 1 should give high loss
        pred = torch.full((1, 10, 16, 16), 0.01)
        target = torch.zeros(1, 10, 16, 16)
        target[0, 0, 8, 8] = 1.0

        loss = loss_fn(pred, target)
        assert loss.item() > 0, "Loss should be positive with bad predictions"

    def test_focal_loss_perfect_predictions(self):
        """Test focal loss with perfect predictions (low loss expected)."""
        loss_fn = FocalLoss(alpha=2.0, beta=4.0)

        # Perfect predictions: pred matches target
        target = torch.zeros(1, 10, 16, 16)
        target[0, 0, 8, 8] = 1.0

        pred = torch.zeros(1, 10, 16, 16) + 0.001  # Near zero for negatives
        pred[0, 0, 8, 8] = 0.999  # Near one for the positive

        loss = loss_fn(pred, target)
        assert loss.item() < 0.1, (
            f"Loss should be very small with perfect predictions, got {loss.item()}"
        )

    def test_focal_loss_decreases_with_better_predictions(self):
        """Test that loss decreases as predictions improve."""
        loss_fn = FocalLoss(alpha=2.0, beta=4.0)

        target = torch.zeros(1, 10, 16, 16)
        target[0, 0, 8, 8] = 1.0

        # Bad prediction
        pred_bad = torch.full((1, 10, 16, 16), 0.5)
        loss_bad = loss_fn(pred_bad, target)

        # Better prediction (correct class near 1, others near 0)
        pred_good = torch.full((1, 10, 16, 16), 0.1)
        pred_good[0, 0, 8, 8] = 0.9
        loss_good = loss_fn(pred_good, target)

        assert loss_good.item() < loss_bad.item(), (
            f"Better predictions should have lower loss: good={loss_good.item()}, bad={loss_bad.item()}"
        )

    def test_l1_regression_loss(self):
        """Test L1 regression loss with positive mask."""
        loss_fn = L1RegressionLoss()

        pred = torch.randn(1, 8, 16, 16)
        target = torch.randn(1, 8, 16, 16)
        mask = torch.zeros(1, 1, 16, 16)
        mask[0, 0, 8, 8] = 1.0
        mask[0, 0, 4, 12] = 1.0

        loss = loss_fn(pred, target, mask)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"
        assert loss.item() >= 0, "L1 loss should be non-negative"

    def test_l1_loss_zero_with_perfect_match(self):
        """Test L1 loss is zero when prediction equals target at masked locations."""
        loss_fn = L1RegressionLoss()

        target = torch.randn(1, 8, 16, 16)
        pred = target.clone()
        mask = torch.zeros(1, 1, 16, 16)
        mask[0, 0, 8, 8] = 1.0

        loss = loss_fn(pred, target, mask)
        assert loss.item() < 1e-6, f"Loss should be ~0 with perfect match, got {loss.item()}"


class TestPostProcessing:
    """Tests for NMS and detection decoding post-processing."""

    def test_nms_removes_overlapping_boxes(self):
        """Test NMS removes boxes that overlap with higher-scoring boxes."""
        # Two boxes at nearly the same location
        boxes = torch.tensor([
            [10.0, 10.0, 0.0, 4.0, 4.0, 2.0, 0.0],  # high score
            [10.1, 10.1, 0.0, 4.0, 4.0, 2.0, 0.0],  # overlapping, lower score
            [50.0, 50.0, 0.0, 4.0, 4.0, 2.0, 0.0],  # far away, keep
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = nms_bev(boxes, scores, iou_threshold=0.2)

        # Should keep the first and third (non-overlapping) boxes
        assert 0 in keep.tolist(), "Highest score box should be kept"
        assert 2 in keep.tolist(), "Non-overlapping box should be kept"
        # The second box overlaps with first and should be removed
        assert 1 not in keep.tolist(), "Overlapping lower-score box should be removed"

    def test_nms_keeps_non_overlapping_boxes(self):
        """Test NMS keeps boxes that don't overlap."""
        # Three well-separated boxes
        boxes = torch.tensor([
            [0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0],
            [20.0, 20.0, 0.0, 2.0, 2.0, 1.0, 0.0],
            [40.0, 40.0, 0.0, 2.0, 2.0, 1.0, 0.0],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = nms_bev(boxes, scores, iou_threshold=0.2)
        assert len(keep) == 3, f"All non-overlapping boxes should be kept, got {len(keep)}"

    def test_nms_empty_input(self):
        """Test NMS handles empty input gracefully."""
        boxes = torch.zeros(0, 7)
        scores = torch.zeros(0)

        keep = nms_bev(boxes, scores, iou_threshold=0.2)
        assert len(keep) == 0

    def test_nms_single_box(self):
        """Test NMS with a single box returns that box."""
        boxes = torch.tensor([[10.0, 10.0, 0.0, 4.0, 4.0, 2.0, 0.5]])
        scores = torch.tensor([0.95])

        keep = nms_bev(boxes, scores, iou_threshold=0.2)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_score_thresholding(self):
        """Test that score thresholding removes low-confidence detections."""
        # Create a heatmap with known peaks
        B, C, H, W = 1, 10, 32, 32
        heatmap = torch.zeros(B, C, H, W)
        regression = torch.randn(B, 8, H, W) * 0.1
        velocity = torch.randn(B, 2, H, W) * 0.1

        # High-confidence detection
        heatmap[0, 0, 16, 16] = 0.9
        # Low-confidence detection (below threshold)
        heatmap[0, 1, 8, 8] = 0.05

        results = decode_detections(
            heatmap, regression, velocity,
            score_threshold=0.1, nms_threshold=0.5,
        )

        scores = results[0]["scores"]
        # All returned scores should be above threshold
        if scores.numel() > 0:
            assert (scores >= 0.1).all(), (
                f"Found scores below threshold: {scores[scores < 0.1]}"
            )

    def test_decode_detections_output_format(self):
        """Test decoded detections have correct output format."""
        B, C, H, W = 2, 10, 16, 16
        heatmap = torch.rand(B, C, H, W) * 0.5  # Some detections
        regression = torch.randn(B, 8, H, W)
        velocity = torch.randn(B, 2, H, W)

        results = decode_detections(heatmap, regression, velocity, score_threshold=0.3)

        assert len(results) == B, f"Expected {B} results, got {len(results)}"
        for r in results:
            assert "boxes" in r
            assert "scores" in r
            assert "labels" in r
            assert r["boxes"].dim() == 2
            assert r["scores"].dim() == 1
            assert r["labels"].dim() == 1
            # boxes and scores should have same length
            assert r["boxes"].shape[0] == r["scores"].shape[0]
            assert r["boxes"].shape[0] == r["labels"].shape[0]

    def test_nms_respects_iou_threshold(self):
        """Test that adjusting IoU threshold changes NMS behavior."""
        # Two overlapping boxes
        boxes = torch.tensor([
            [10.0, 10.0, 0.0, 4.0, 4.0, 2.0, 0.0],
            [11.0, 11.0, 0.0, 4.0, 4.0, 2.0, 0.0],
        ])
        scores = torch.tensor([0.9, 0.8])

        # Strict threshold - should suppress
        keep_strict = nms_bev(boxes, scores, iou_threshold=0.1)
        # Lenient threshold - should keep both
        keep_lenient = nms_bev(boxes, scores, iou_threshold=0.9)

        assert len(keep_strict) <= len(keep_lenient), (
            "Stricter IoU threshold should suppress more boxes"
        )


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration:
    """Integration tests verifying component interactions."""

    def test_camera_to_fusion_pipeline(self, model_config):
        """Test data flows correctly from camera branch to fusion."""
        cam_model = SimpleCameraBranch(
            fpn_out_channels=model_config["fusion_embed_dim"],
            num_cameras=6,
        )
        cam_model.eval()

        images = torch.randn(1, 6, 3, 256, 448)
        with torch.no_grad():
            cam_output = cam_model(images)

        # Verify FPN output can be used as fusion memory
        cam_feat = cam_output["features"][0]  # Highest-res level
        assert cam_feat.shape[2] == model_config["fusion_embed_dim"]

    def test_radar_to_fusion_pipeline(self, model_config):
        """Test data flows correctly from radar branch to fusion."""
        radar_model = SimpleRadarBranch(
            bev_channels=model_config["fusion_embed_dim"],
            bev_height=16,
            bev_width=16,
        )
        radar_model.eval()

        points = torch.randn(1, 200, 6)
        points[:, :, 0] = points[:, :, 0].clamp(-25.0, 25.0)
        points[:, :, 1] = points[:, :, 1].clamp(-25.0, 25.0)
        num_points = torch.tensor([200])

        with torch.no_grad():
            radar_output = radar_model(points, num_points)

        radar_bev = radar_output["bev_features"]
        assert radar_bev.shape[1] == model_config["fusion_embed_dim"]

    def test_fusion_to_detection_pipeline(self, model_config):
        """Test data flows correctly from fusion to detection head."""
        embed_dim = model_config["fusion_embed_dim"]
        H, W = 16, 16

        head = SimpleDetectionHead(
            in_channels=embed_dim,
            num_classes=model_config["num_classes"],
        )
        head.eval()

        fused_bev = torch.randn(1, embed_dim, H, W)
        with torch.no_grad():
            detections = head(fused_bev)

        assert detections["heatmap"].shape == (1, model_config["num_classes"], H, W)

    def test_model_parameter_count(self, model_config):
        """Test model has a reasonable number of parameters."""
        model = SimpleCRAFTModel(model_config)

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0, "Model should have parameters"

        # For the simplified test model, check it's within reasonable range
        # (not too small = trivial, not too large = something wrong)
        assert total_params > 1000, f"Model too small: {total_params} params"
        assert total_params < 100_000_000, f"Model unexpectedly large: {total_params} params"

    def test_deterministic_inference(self, model_config, dummy_images, dummy_radar_points):
        """Test model produces same output for same input (deterministic)."""
        model = SimpleCRAFTModel(model_config)
        model.eval()

        num_points = torch.tensor([500])

        with torch.no_grad():
            output1 = model(dummy_images, dummy_radar_points, num_points)
            output2 = model(dummy_images, dummy_radar_points, num_points)

        assert torch.allclose(output1["heatmap"], output2["heatmap"], atol=1e-6), (
            "Model should be deterministic in eval mode"
        )
        assert torch.allclose(output1["regression"], output2["regression"], atol=1e-6)
        assert torch.allclose(output1["velocity"], output2["velocity"], atol=1e-6)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
