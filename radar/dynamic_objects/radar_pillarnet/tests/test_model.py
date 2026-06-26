"""
Comprehensive pytest unit tests for RadarPillarNet - a radar-based 3D object detection model.

This file includes minimal but realistic stub implementations of the model components
so that tests are self-contained and runnable without external model code.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Stub Model Implementations
# =============================================================================


class PillarEncoder(nn.Module):
    """PointNet-style pillar feature encoder with shared MLPs and max pooling."""

    def __init__(self, in_features: int, num_output_features: int = 64):
        super().__init__()
        self.in_features = in_features
        self.num_output_features = num_output_features

        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_output_features),
            nn.BatchNorm1d(num_output_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, pillars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pillars: (B, max_pillars, max_points, in_features)
        Returns:
            pillar_features: (B, max_pillars, num_output_features)
        """
        B, P, N, F_in = pillars.shape

        # Reshape for linear layers: (B*P*N, F_in)
        x = pillars.reshape(B * P * N, F_in)

        # Apply shared MLP with batch norm
        # For BN we need at least 2 elements, handle edge case
        if x.shape[0] > 1:
            x = self.mlp(x)
        else:
            # Skip batch norm for single element
            x = self.mlp[0](x)
            x = F.relu(x)
            x = self.mlp[3](x)
            x = F.relu(x)

        # Reshape back: (B, P, N, num_output_features)
        x = x.reshape(B, P, N, self.num_output_features)

        # Max pooling over points dimension: (B, P, num_output_features)
        x, _ = x.max(dim=2)

        return x


class PillarScatter(nn.Module):
    """Scatters pillar features to a BEV pseudo-image based on pillar coordinates."""

    def __init__(self, num_features: int, grid_h: int, grid_w: int):
        super().__init__()
        self.num_features = num_features
        self.grid_h = grid_h
        self.grid_w = grid_w

    def forward(
        self, pillar_features: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pillar_features: (B, max_pillars, C)
            coords: (B, max_pillars, 2) - (x_idx, y_idx) grid coordinates
        Returns:
            pseudo_image: (B, C, H, W)
        """
        B, P, C = pillar_features.shape
        device = pillar_features.device

        pseudo_image = torch.zeros(
            B, C, self.grid_h, self.grid_w, device=device, dtype=pillar_features.dtype
        )

        for b in range(B):
            for p in range(P):
                x_idx = int(coords[b, p, 0].item())
                y_idx = int(coords[b, p, 1].item())
                if 0 <= x_idx < self.grid_w and 0 <= y_idx < self.grid_h:
                    pseudo_image[b, :, y_idx, x_idx] = pillar_features[b, p]

        return pseudo_image


class Backbone(nn.Module):
    """Multi-scale 2D convolutional backbone for BEV feature extraction."""

    def __init__(self, in_channels: int = 64, layer_nums=(3, 5, 5)):
        super().__init__()
        self.in_channels = in_channels

        # Block 1: stride 1, output channels 64
        block1_layers = [
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        for _ in range(layer_nums[0] - 1):
            block1_layers.extend(
                [
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ]
            )
        self.block1 = nn.Sequential(*block1_layers)

        # Block 2: stride 2, output channels 128
        block2_layers = [
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        for _ in range(layer_nums[1] - 1):
            block2_layers.extend(
                [
                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                ]
            )
        self.block2 = nn.Sequential(*block2_layers)

        # Block 3: stride 2, output channels 256
        block3_layers = [
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        for _ in range(layer_nums[2] - 1):
            block3_layers.extend(
                [
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ]
            )
        self.block3 = nn.Sequential(*block3_layers)

        # Upsample blocks to bring features back to same resolution
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) pseudo-image
        Returns:
            multi_scale_features: list of feature maps at different scales
            fused: (B, 384, H/2, W/2) fused multi-scale feature map
        """
        x1 = self.block1(x)  # (B, 64, H/2, W/2)
        x2 = self.block2(x1)  # (B, 128, H/4, W/4)
        x3 = self.block3(x2)  # (B, 256, H/8, W/8)

        up1 = self.up1(x1)  # (B, 128, H/2, W/2)
        up2 = self.up2(x2)  # (B, 128, H/2, W/2)
        up3 = self.up3(x3)  # (B, 128, H/2, W/2)

        fused = torch.cat([up1, up2, up3], dim=1)  # (B, 384, H/2, W/2)

        return [x1, x2, x3], fused


class AnchorHead(nn.Module):
    """Detection head that predicts classification, regression, velocity, and direction."""

    def __init__(
        self,
        in_channels: int = 384,
        num_classes: int = 3,
        num_anchors: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.cls_head = nn.Conv2d(
            in_channels, num_anchors * num_classes, 1, bias=True
        )
        self.reg_head = nn.Conv2d(
            in_channels, num_anchors * 7, 1, bias=True
        )  # x, y, z, w, l, h, yaw
        self.vel_head = nn.Conv2d(
            in_channels, num_anchors * 2, 1, bias=True
        )  # vx, vy
        self.dir_head = nn.Conv2d(
            in_channels, num_anchors * 2, 1, bias=True
        )  # direction classification (2 bins)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, in_channels, H, W) fused backbone features
        Returns:
            dict with cls, reg, vel, dir predictions
        """
        cls_preds = self.cls_head(x)
        reg_preds = self.reg_head(x)
        vel_preds = self.vel_head(x)
        dir_preds = self.dir_head(x)

        return {
            "cls": cls_preds,
            "reg": reg_preds,
            "vel": vel_preds,
            "dir": dir_preds,
        }


class RadarPillarNet(nn.Module):
    """Full RadarPillarNet model combining all components."""

    def __init__(
        self,
        in_features: int = 7,
        num_output_features: int = 64,
        grid_h: int = 512,
        grid_w: int = 512,
        num_classes: int = 3,
        num_anchors: int = 2,
    ):
        super().__init__()
        self.pillar_encoder = PillarEncoder(in_features, num_output_features)
        self.pillar_scatter = PillarScatter(num_output_features, grid_h, grid_w)
        self.backbone = Backbone(in_channels=num_output_features)
        self.head = AnchorHead(
            in_channels=384, num_classes=num_classes, num_anchors=num_anchors
        )

    def forward(self, pillars: torch.Tensor, coords: torch.Tensor):
        """
        Args:
            pillars: (B, max_pillars, max_points, in_features)
            coords: (B, max_pillars, 2) grid coordinates for each pillar
        Returns:
            dict with detection outputs
        """
        pillar_features = self.pillar_encoder(pillars)
        pseudo_image = self.pillar_scatter(pillar_features, coords)
        multi_scale_features, fused = self.backbone(pseudo_image)
        predictions = self.head(fused)
        predictions["multi_scale_features"] = multi_scale_features
        return predictions


# =============================================================================
# Loss Computation
# =============================================================================


class RadarPillarNetLoss(nn.Module):
    """Combined loss for training RadarPillarNet."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.reg_loss_fn = nn.SmoothL1Loss(reduction="mean")
        self.dir_loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Args:
            predictions: dict with cls, reg, vel, dir tensors
            targets: dict with cls_targets, reg_targets, vel_targets, dir_targets
        Returns:
            dict with loss_cls, loss_reg, loss_vel, loss_dir, loss_total
        """
        loss_cls = self.cls_loss_fn(predictions["cls"], targets["cls_targets"])
        loss_reg = self.reg_loss_fn(predictions["reg"], targets["reg_targets"])
        loss_vel = self.reg_loss_fn(predictions["vel"], targets["vel_targets"])

        # Direction loss: reshape for cross entropy
        B, C, H, W = predictions["dir"].shape
        num_anchors = C // 2
        dir_preds = predictions["dir"].reshape(B, num_anchors, 2, H, W)
        dir_preds = dir_preds.permute(0, 1, 3, 4, 2).reshape(-1, 2)
        dir_targets = targets["dir_targets"].reshape(-1)
        loss_dir = self.dir_loss_fn(dir_preds, dir_targets)

        loss_total = loss_cls + loss_reg + loss_vel + 0.2 * loss_dir

        return {
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "loss_vel": loss_vel,
            "loss_dir": loss_dir,
            "loss_total": loss_total,
        }


# =============================================================================
# NMS Post-processing
# =============================================================================


def rotate_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Simplified NMS using axis-aligned IoU for testing purposes.

    Args:
        boxes: (N, 7) - x, y, z, w, l, h, yaw
        scores: (N,) confidence scores
    Returns:
        keep: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Use x, y, w, l for axis-aligned IoU approximation
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 3]
    l = boxes[:, 4]

    x1 = x - w / 2
    y1 = y - l / 2
    x2 = x + w / 2
    y2 = y + l / 2

    areas = w * l

    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break

        i = order[0].item()
        keep.append(i)

        xx1 = torch.max(x1[order[1:]], x1[i])
        yy1 = torch.max(y1[order[1:]], y1[i])
        xx2 = torch.min(x2[order[1:]], x2[i])
        yy2 = torch.min(y2[order[1:]], y2[i])

        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        union = areas[order[1:]] + areas[i] - inter
        iou = inter / (union + 1e-6)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def post_process(
    cls_preds: torch.Tensor,
    reg_preds: torch.Tensor,
    score_threshold: float = 0.3,
    nms_threshold: float = 0.5,
    max_detections: int = 100,
) -> dict:
    """
    Post-process model outputs to produce final detections.

    Args:
        cls_preds: (N, num_classes) class scores (logits)
        reg_preds: (N, 7) regression predictions
        score_threshold: minimum confidence to keep
        nms_threshold: IoU threshold for NMS
        max_detections: maximum number of output detections
    Returns:
        dict with boxes, scores, labels
    """
    scores = cls_preds.sigmoid()
    max_scores, labels = scores.max(dim=1)

    # Score threshold filter
    mask = max_scores > score_threshold
    filtered_boxes = reg_preds[mask]
    filtered_scores = max_scores[mask]
    filtered_labels = labels[mask]

    if filtered_boxes.shape[0] == 0:
        return {
            "boxes": torch.zeros(0, 7, device=cls_preds.device),
            "scores": torch.zeros(0, device=cls_preds.device),
            "labels": torch.zeros(0, dtype=torch.long, device=cls_preds.device),
        }

    # NMS
    keep = rotate_nms(filtered_boxes, filtered_scores, nms_threshold)

    # Max detections limit
    if keep.shape[0] > max_detections:
        keep = keep[:max_detections]

    return {
        "boxes": filtered_boxes[keep],
        "scores": filtered_scores[keep],
        "labels": filtered_labels[keep],
    }


# =============================================================================
# Multi-sweep Accumulation Utilities
# =============================================================================


def ego_motion_compensation(
    points: torch.Tensor, transform_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Apply ego-motion compensation to transform points from a past frame to current frame.

    Args:
        points: (N, 3) point coordinates (x, y, z)
        transform_matrix: (4, 4) transformation matrix from past to current
    Returns:
        transformed_points: (N, 3)
    """
    N = points.shape[0]
    # Homogeneous coordinates
    ones = torch.ones(N, 1, device=points.device, dtype=points.dtype)
    points_homo = torch.cat([points, ones], dim=1)  # (N, 4)
    transformed = (transform_matrix @ points_homo.T).T  # (N, 4)
    return transformed[:, :3]


def compute_time_lag_feature(
    sweep_indices: torch.Tensor, sweep_timestamps: torch.Tensor
) -> torch.Tensor:
    """
    Compute time lag feature for each point based on its sweep.

    Args:
        sweep_indices: (N,) index of sweep each point belongs to (0 = current)
        sweep_timestamps: (num_sweeps,) timestamp offset of each sweep (0.0 for current)
    Returns:
        time_lags: (N,) time lag in seconds for each point
    """
    return sweep_timestamps[sweep_indices]


# =============================================================================
# Data Augmentation Functions
# =============================================================================


def random_flip(
    points: torch.Tensor, boxes: torch.Tensor, axis: int = 0
) -> tuple:
    """
    Flip points and boxes along specified axis.

    Args:
        points: (N, 3+) point cloud
        boxes: (M, 7) boxes [x, y, z, w, l, h, yaw]
        axis: 0 for x-axis flip, 1 for y-axis flip
    Returns:
        flipped_points, flipped_boxes
    """
    flipped_points = points.clone()
    flipped_boxes = boxes.clone()

    flipped_points[:, axis] = -flipped_points[:, axis]
    flipped_boxes[:, axis] = -flipped_boxes[:, axis]

    # Adjust yaw angle
    if axis == 0:
        flipped_boxes[:, 6] = math.pi - flipped_boxes[:, 6]
    elif axis == 1:
        flipped_boxes[:, 6] = -flipped_boxes[:, 6]

    return flipped_points, flipped_boxes


def random_rotation(
    points: torch.Tensor, boxes: torch.Tensor, angle: float
) -> tuple:
    """
    Rotate points and boxes around z-axis by given angle.

    Args:
        points: (N, 3+) point cloud
        boxes: (M, 7) boxes [x, y, z, w, l, h, yaw]
        angle: rotation angle in radians
    Returns:
        rotated_points, rotated_boxes
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rot_matrix = torch.tensor(
        [[cos_a, -sin_a], [sin_a, cos_a]],
        device=points.device,
        dtype=points.dtype,
    )

    rotated_points = points.clone()
    rotated_points[:, :2] = (rot_matrix @ points[:, :2].T).T

    rotated_boxes = boxes.clone()
    rotated_boxes[:, :2] = (rot_matrix @ boxes[:, :2].T).T
    rotated_boxes[:, 6] = boxes[:, 6] + angle

    return rotated_points, rotated_boxes


def random_scaling(
    points: torch.Tensor, boxes: torch.Tensor, scale_factor: float
) -> tuple:
    """
    Scale points and boxes by given factor.

    Args:
        points: (N, 3+) point cloud
        boxes: (M, 7) boxes [x, y, z, w, l, h, yaw]
        scale_factor: scale multiplier
    Returns:
        scaled_points, scaled_boxes
    """
    scaled_points = points.clone()
    scaled_points[:, :3] = points[:, :3] * scale_factor

    scaled_boxes = boxes.clone()
    scaled_boxes[:, :3] = boxes[:, :3] * scale_factor  # center
    scaled_boxes[:, 3:6] = boxes[:, 3:6] * scale_factor  # size

    return scaled_points, scaled_boxes


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def max_pillars():
    return 100


@pytest.fixture
def max_points():
    return 32


@pytest.fixture
def in_features():
    return 7


@pytest.fixture
def num_output_features():
    return 64


@pytest.fixture
def grid_size():
    return (128, 128)  # (H, W) - small for fast testing


@pytest.fixture
def num_classes():
    return 3


@pytest.fixture
def num_anchors():
    return 2


@pytest.fixture
def pillar_encoder(in_features, num_output_features):
    model = PillarEncoder(in_features, num_output_features)
    model.eval()
    return model


@pytest.fixture
def pillar_scatter(num_output_features, grid_size):
    return PillarScatter(num_output_features, grid_size[0], grid_size[1])


@pytest.fixture
def backbone(num_output_features):
    model = Backbone(in_channels=num_output_features)
    model.eval()
    return model


@pytest.fixture
def anchor_head(num_classes, num_anchors):
    model = AnchorHead(in_channels=384, num_classes=num_classes, num_anchors=num_anchors)
    model.eval()
    return model


@pytest.fixture
def radar_pillarnet(in_features, num_output_features, grid_size, num_classes, num_anchors):
    model = RadarPillarNet(
        in_features=in_features,
        num_output_features=num_output_features,
        grid_h=grid_size[0],
        grid_w=grid_size[1],
        num_classes=num_classes,
        num_anchors=num_anchors,
    )
    model.eval()
    return model


@pytest.fixture
def sample_pillars(batch_size, max_pillars, max_points, in_features):
    """Generate random pillar input data."""
    return torch.randn(batch_size, max_pillars, max_points, in_features)


@pytest.fixture
def sample_coords(batch_size, max_pillars, grid_size):
    """Generate random pillar coordinates within grid bounds."""
    coords = torch.zeros(batch_size, max_pillars, 2, dtype=torch.long)
    coords[:, :, 0] = torch.randint(0, grid_size[1], (batch_size, max_pillars))
    coords[:, :, 1] = torch.randint(0, grid_size[0], (batch_size, max_pillars))
    return coords


# =============================================================================
# Tests: PillarEncoder
# =============================================================================


class TestPillarEncoder:
    """Tests for the PillarEncoder module."""

    def test_pillar_encoder_output_shape(
        self, pillar_encoder, sample_pillars, batch_size, max_pillars, num_output_features
    ):
        """Given input (batch, max_pillars, max_points, features),
        output should be (batch, max_pillars, num_output_features)."""
        output = pillar_encoder(sample_pillars)
        assert output.shape == (batch_size, max_pillars, num_output_features)

    def test_pillar_encoder_empty_pillars(
        self, pillar_encoder, batch_size, max_pillars, max_points, in_features, num_output_features
    ):
        """Pillars with all zeros should produce zero or near-zero output."""
        empty_pillars = torch.zeros(batch_size, max_pillars, max_points, in_features)
        output = pillar_encoder(empty_pillars)
        assert output.shape == (batch_size, max_pillars, num_output_features)
        # After ReLU and max-pool of zero input through linear layers with BN,
        # output should be near zero (BN centers the data)
        assert output.abs().max() < 1.0  # relaxed bound due to BN bias

    def test_pillar_encoder_single_point(
        self, pillar_encoder, batch_size, max_pillars, max_points, in_features, num_output_features
    ):
        """Pillar with only one non-zero point should still produce valid output."""
        pillars = torch.zeros(batch_size, max_pillars, max_points, in_features)
        # Set only the first point in each pillar to have data
        pillars[:, :, 0, :] = torch.randn(batch_size, max_pillars, in_features)

        output = pillar_encoder(pillars)
        assert output.shape == (batch_size, max_pillars, num_output_features)
        # Output should not be all zeros since we have valid data
        assert not torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    @pytest.mark.parametrize("batch", [1, 4, 8])
    def test_pillar_encoder_various_batch_sizes(
        self, in_features, num_output_features, max_pillars, max_points, batch
    ):
        """Encoder should work with various batch sizes."""
        encoder = PillarEncoder(in_features, num_output_features)
        encoder.eval()
        pillars = torch.randn(batch, max_pillars, max_points, in_features)
        output = encoder(pillars)
        assert output.shape == (batch, max_pillars, num_output_features)


# =============================================================================
# Tests: PillarScatter
# =============================================================================


class TestPillarScatter:
    """Tests for the PillarScatter module."""

    def test_pillar_scatter_output_shape(
        self, pillar_scatter, batch_size, max_pillars, num_output_features, grid_size
    ):
        """Should produce (batch, C, H, W) pseudo-image."""
        pillar_features = torch.randn(batch_size, max_pillars, num_output_features)
        coords = torch.zeros(batch_size, max_pillars, 2, dtype=torch.long)
        coords[:, :, 0] = torch.randint(0, grid_size[1], (batch_size, max_pillars))
        coords[:, :, 1] = torch.randint(0, grid_size[0], (batch_size, max_pillars))

        output = pillar_scatter(pillar_features, coords)
        assert output.shape == (batch_size, num_output_features, grid_size[0], grid_size[1])

    def test_pillar_scatter_correct_placement(
        self, num_output_features, grid_size
    ):
        """Features should be placed at correct spatial locations based on pillar coordinates."""
        scatter = PillarScatter(num_output_features, grid_size[0], grid_size[1])

        B = 1
        P = 3
        pillar_features = torch.zeros(B, P, num_output_features)
        # Set distinct feature vectors
        pillar_features[0, 0] = torch.ones(num_output_features) * 1.0
        pillar_features[0, 1] = torch.ones(num_output_features) * 2.0
        pillar_features[0, 2] = torch.ones(num_output_features) * 3.0

        coords = torch.zeros(B, P, 2, dtype=torch.long)
        coords[0, 0] = torch.tensor([10, 20])  # x=10, y=20
        coords[0, 1] = torch.tensor([30, 40])  # x=30, y=40
        coords[0, 2] = torch.tensor([50, 60])  # x=50, y=60

        output = scatter(pillar_features, coords)

        # Check that features were placed correctly
        assert torch.allclose(output[0, :, 20, 10], torch.ones(num_output_features) * 1.0)
        assert torch.allclose(output[0, :, 40, 30], torch.ones(num_output_features) * 2.0)
        assert torch.allclose(output[0, :, 60, 50], torch.ones(num_output_features) * 3.0)

        # Check that other locations remain zero
        assert output[0, :, 0, 0].sum() == 0.0
        assert output[0, :, 100, 100].sum() == 0.0

    def test_pillar_scatter_out_of_bounds_ignored(self, num_output_features, grid_size):
        """Coordinates outside grid bounds should be safely ignored."""
        scatter = PillarScatter(num_output_features, grid_size[0], grid_size[1])

        B, P = 1, 2
        pillar_features = torch.ones(B, P, num_output_features)
        coords = torch.zeros(B, P, 2, dtype=torch.long)
        coords[0, 0] = torch.tensor([grid_size[1] + 10, 5])  # out of bounds x
        coords[0, 1] = torch.tensor([5, grid_size[0] + 10])  # out of bounds y

        output = scatter(pillar_features, coords)
        # All values should be zero since both coords are out of bounds
        assert output.sum() == 0.0


# =============================================================================
# Tests: Backbone
# =============================================================================


class TestBackbone:
    """Tests for the multi-scale Backbone module."""

    def test_backbone_output_dimensions(self, backbone, batch_size, num_output_features, grid_size):
        """Check output spatial dimensions at each scale."""
        H, W = grid_size
        x = torch.randn(batch_size, num_output_features, H, W)
        multi_scale, fused = backbone(x)

        # Block 1: stride 2 -> H/2, W/2
        assert multi_scale[0].shape == (batch_size, 64, H // 2, W // 2)
        # Block 2: stride 2 again -> H/4, W/4
        assert multi_scale[1].shape == (batch_size, 128, H // 4, W // 4)
        # Block 3: stride 2 again -> H/8, W/8
        assert multi_scale[2].shape == (batch_size, 256, H // 8, W // 8)

    def test_backbone_multi_scale_features(self, backbone, batch_size, num_output_features, grid_size):
        """Verify correct number of feature maps and their channels."""
        H, W = grid_size
        x = torch.randn(batch_size, num_output_features, H, W)
        multi_scale, fused = backbone(x)

        # Should produce 3 scale levels
        assert len(multi_scale) == 3

        # Fused output should have 384 channels (128 * 3) at H/2, W/2 resolution
        assert fused.shape == (batch_size, 384, H // 2, W // 2)

    def test_backbone_gradient_flow(self, num_output_features, grid_size):
        """Gradients should flow through the backbone."""
        backbone = Backbone(in_channels=num_output_features)
        backbone.train()

        x = torch.randn(1, num_output_features, grid_size[0], grid_size[1], requires_grad=True)
        _, fused = backbone(x)
        loss = fused.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# =============================================================================
# Tests: AnchorHead
# =============================================================================


class TestAnchorHead:
    """Tests for the AnchorHead detection head."""

    def test_anchor_head_output_shape(
        self, anchor_head, batch_size, num_classes, num_anchors, grid_size
    ):
        """Should produce correct shapes for cls, reg, vel, dir predictions."""
        H, W = grid_size[0] // 2, grid_size[1] // 2  # After backbone downsampling
        x = torch.randn(batch_size, 384, H, W)
        output = anchor_head(x)

        assert output["cls"].shape == (batch_size, num_anchors * num_classes, H, W)
        assert output["reg"].shape == (batch_size, num_anchors * 7, H, W)
        assert output["vel"].shape == (batch_size, num_anchors * 2, H, W)
        assert output["dir"].shape == (batch_size, num_anchors * 2, H, W)

    def test_anchor_head_num_predictions(
        self, anchor_head, batch_size, num_classes, num_anchors, grid_size
    ):
        """Total predictions should match expected count."""
        H, W = grid_size[0] // 2, grid_size[1] // 2
        x = torch.randn(batch_size, 384, H, W)
        output = anchor_head(x)

        total_anchors_per_sample = num_anchors * H * W
        # Classification: one score per class per anchor
        total_cls_values = output["cls"].reshape(batch_size, -1).shape[1]
        assert total_cls_values == num_anchors * num_classes * H * W

        # Regression: 7 values per anchor
        total_reg_values = output["reg"].reshape(batch_size, -1).shape[1]
        assert total_reg_values == total_anchors_per_sample * 7

    @pytest.mark.parametrize("num_cls", [1, 3, 10])
    def test_anchor_head_various_num_classes(self, num_cls, num_anchors, grid_size):
        """Head should work with different numbers of classes."""
        head = AnchorHead(in_channels=384, num_classes=num_cls, num_anchors=num_anchors)
        head.eval()
        H, W = grid_size[0] // 2, grid_size[1] // 2
        x = torch.randn(1, 384, H, W)
        output = head(x)
        assert output["cls"].shape == (1, num_anchors * num_cls, H, W)


# =============================================================================
# Tests: Full Forward Pass
# =============================================================================


class TestForwardPass:
    """Tests for the full RadarPillarNet forward pass."""

    def test_forward_pass_end_to_end(
        self, radar_pillarnet, sample_pillars, sample_coords, batch_size, grid_size, num_anchors, num_classes
    ):
        """Synthetic batch through entire model, verify output dict keys and shapes."""
        output = radar_pillarnet(sample_pillars, sample_coords)

        expected_h = grid_size[0] // 2
        expected_w = grid_size[1] // 2

        assert "cls" in output
        assert "reg" in output
        assert "vel" in output
        assert "dir" in output
        assert "multi_scale_features" in output

        assert output["cls"].shape == (batch_size, num_anchors * num_classes, expected_h, expected_w)
        assert output["reg"].shape == (batch_size, num_anchors * 7, expected_h, expected_w)
        assert output["vel"].shape == (batch_size, num_anchors * 2, expected_h, expected_w)
        assert output["dir"].shape == (batch_size, num_anchors * 2, expected_h, expected_w)

    def test_forward_pass_batch_independence(
        self, in_features, num_output_features, grid_size, num_classes, num_anchors
    ):
        """Different batch elements should produce different outputs given different inputs."""
        model = RadarPillarNet(
            in_features=in_features,
            num_output_features=num_output_features,
            grid_h=grid_size[0],
            grid_w=grid_size[1],
            num_classes=num_classes,
            num_anchors=num_anchors,
        )
        model.eval()

        B, P, N = 2, 50, 16
        torch.manual_seed(42)
        pillars = torch.randn(B, P, N, in_features)
        # Make batch elements significantly different
        pillars[1] = pillars[1] * 5.0 + 3.0

        coords = torch.randint(0, min(grid_size), (B, P, 2))

        output = model(pillars, coords)

        # The two batch elements should produce different predictions
        cls_diff = (output["cls"][0] - output["cls"][1]).abs().sum()
        assert cls_diff > 0.0, "Batch elements should produce different outputs"

    def test_forward_pass_deterministic(
        self, radar_pillarnet, sample_pillars, sample_coords
    ):
        """Same input should produce same output in eval mode."""
        radar_pillarnet.eval()
        out1 = radar_pillarnet(sample_pillars, sample_coords)
        out2 = radar_pillarnet(sample_pillars, sample_coords)
        assert torch.allclose(out1["cls"], out2["cls"])
        assert torch.allclose(out1["reg"], out2["reg"])


# =============================================================================
# Tests: Loss Computation
# =============================================================================


class TestLossComputation:
    """Tests for RadarPillarNet loss computation."""

    @pytest.fixture
    def loss_fn(self, num_classes):
        return RadarPillarNetLoss(num_classes=num_classes)

    @pytest.fixture
    def predictions_and_targets(self, batch_size, num_anchors, num_classes, grid_size):
        """Generate matching predictions and target tensors."""
        H, W = grid_size[0] // 2, grid_size[1] // 2
        predictions = {
            "cls": torch.randn(batch_size, num_anchors * num_classes, H, W),
            "reg": torch.randn(batch_size, num_anchors * 7, H, W),
            "vel": torch.randn(batch_size, num_anchors * 2, H, W),
            "dir": torch.randn(batch_size, num_anchors * 2, H, W),
        }
        targets = {
            "cls_targets": torch.rand(batch_size, num_anchors * num_classes, H, W),
            "reg_targets": torch.randn(batch_size, num_anchors * 7, H, W),
            "vel_targets": torch.randn(batch_size, num_anchors * 2, H, W),
            "dir_targets": torch.randint(0, 2, (batch_size, num_anchors, H, W)),
        }
        return predictions, targets

    def test_loss_non_negative(self, loss_fn, predictions_and_targets):
        """All loss components should be >= 0."""
        predictions, targets = predictions_and_targets
        losses = loss_fn(predictions, targets)

        assert losses["loss_cls"] >= 0
        assert losses["loss_reg"] >= 0
        assert losses["loss_vel"] >= 0
        assert losses["loss_dir"] >= 0
        assert losses["loss_total"] >= 0

    def test_loss_gradient_flow(
        self, in_features, num_output_features, grid_size, num_classes, num_anchors
    ):
        """Gradients should flow to all parameters."""
        model = RadarPillarNet(
            in_features=in_features,
            num_output_features=num_output_features,
            grid_h=grid_size[0],
            grid_w=grid_size[1],
            num_classes=num_classes,
            num_anchors=num_anchors,
        )
        model.train()
        loss_fn = RadarPillarNetLoss(num_classes=num_classes)

        B, P, N = 2, 50, 16
        pillars = torch.randn(B, P, N, in_features)
        coords = torch.randint(0, min(grid_size), (B, P, 2))

        output = model(pillars, coords)

        H, W = grid_size[0] // 2, grid_size[1] // 2
        targets = {
            "cls_targets": torch.rand(B, num_anchors * num_classes, H, W),
            "reg_targets": torch.randn(B, num_anchors * 7, H, W),
            "vel_targets": torch.randn(B, num_anchors * 2, H, W),
            "dir_targets": torch.randint(0, 2, (B, num_anchors, H, W)),
        }

        losses = loss_fn(output, targets)
        losses["loss_total"].backward()

        # Check that gradients exist for all model parameters
        params_with_grad = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                params_with_grad += 1

        assert params_with_grad > 0, "No parameters received gradients"
        # At least 80% of parameters should have gradients
        assert params_with_grad / total_params > 0.8

    def test_loss_decreases_with_perfect_prediction(self, num_classes, num_anchors, grid_size):
        """Loss should be lower when predictions match targets."""
        loss_fn = RadarPillarNetLoss(num_classes=num_classes)
        B = 1
        H, W = grid_size[0] // 2, grid_size[1] // 2

        # Create targets
        cls_targets = torch.zeros(B, num_anchors * num_classes, H, W)
        reg_targets = torch.randn(B, num_anchors * 7, H, W)
        vel_targets = torch.randn(B, num_anchors * 2, H, W)
        dir_targets = torch.zeros(B, num_anchors, H, W, dtype=torch.long)

        targets = {
            "cls_targets": cls_targets,
            "reg_targets": reg_targets,
            "vel_targets": vel_targets,
            "dir_targets": dir_targets,
        }

        # Random predictions (bad)
        bad_predictions = {
            "cls": torch.randn(B, num_anchors * num_classes, H, W) * 5.0,
            "reg": torch.randn(B, num_anchors * 7, H, W) * 5.0,
            "vel": torch.randn(B, num_anchors * 2, H, W) * 5.0,
            "dir": torch.randn(B, num_anchors * 2, H, W),
        }

        # Near-perfect predictions (good)
        # For BCE: large negative logit -> sigmoid near 0, matching cls_targets=0
        good_predictions = {
            "cls": torch.ones(B, num_anchors * num_classes, H, W) * (-10.0),
            "reg": reg_targets.clone(),
            "vel": vel_targets.clone(),
            "dir": torch.zeros(B, num_anchors * 2, H, W),
        }
        # Set direction logits to strongly predict class 0
        good_predictions["dir"][:, ::2, :, :] = 10.0  # even indices (class 0 logit)
        good_predictions["dir"][:, 1::2, :, :] = -10.0  # odd indices (class 1 logit)

        bad_loss = loss_fn(bad_predictions, targets)["loss_total"]
        good_loss = loss_fn(good_predictions, targets)["loss_total"]

        assert good_loss < bad_loss, (
            f"Good predictions should have lower loss. Got good={good_loss.item():.4f}, "
            f"bad={bad_loss.item():.4f}"
        )


# =============================================================================
# Tests: NMS Post-processing
# =============================================================================


class TestNMSPostProcessing:
    """Tests for NMS and post-processing."""

    def test_nms_removes_overlapping(self):
        """Overlapping boxes with same class should be reduced."""
        # Create two highly overlapping boxes
        boxes = torch.tensor(
            [
                [5.0, 5.0, 0.0, 4.0, 4.0, 2.0, 0.0],  # Box at (5,5), size 4x4
                [5.1, 5.1, 0.0, 4.0, 4.0, 2.0, 0.0],  # Nearly same location
                [5.2, 5.2, 0.0, 4.0, 4.0, 2.0, 0.0],  # Nearly same location
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = rotate_nms(boxes, scores, iou_threshold=0.5)
        # Should keep only the highest scoring box since they all overlap heavily
        assert len(keep) == 1
        assert keep[0] == 0

    def test_nms_keeps_non_overlapping(self):
        """Non-overlapping boxes should all be kept."""
        boxes = torch.tensor(
            [
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],  # Far apart
                [50.0, 50.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [100.0, 100.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = rotate_nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 3

    def test_nms_score_threshold(self):
        """Low-score boxes should be filtered by post_process."""
        N = 10
        cls_preds = torch.randn(N, 3)
        # Make most scores very low (negative logits -> low sigmoid)
        cls_preds[:8] = -10.0
        # Make 2 boxes have high scores
        cls_preds[8] = 5.0
        cls_preds[9] = 5.0

        reg_preds = torch.randn(N, 7)
        # Space the high-score boxes far apart so NMS doesn't remove them
        reg_preds[8, :2] = torch.tensor([0.0, 0.0])
        reg_preds[9, :2] = torch.tensor([100.0, 100.0])
        reg_preds[:, 3:5] = 2.0  # give all boxes a reasonable size

        result = post_process(cls_preds, reg_preds, score_threshold=0.3)

        # Only the 2 high-score boxes should survive
        assert result["boxes"].shape[0] == 2
        assert (result["scores"] > 0.3).all()

    def test_nms_max_detections(self):
        """Output should not exceed max_detections limit."""
        N = 200
        # All boxes have high scores
        cls_preds = torch.ones(N, 3) * 5.0
        # All boxes are far apart (no NMS suppression)
        reg_preds = torch.zeros(N, 7)
        reg_preds[:, 0] = torch.arange(N).float() * 100  # spread along x
        reg_preds[:, 3] = 2.0  # width
        reg_preds[:, 4] = 2.0  # length

        max_det = 50
        result = post_process(cls_preds, reg_preds, score_threshold=0.1, max_detections=max_det)

        assert result["boxes"].shape[0] <= max_det

    def test_nms_empty_input(self):
        """Empty input should return empty results."""
        cls_preds = torch.zeros(0, 3)
        reg_preds = torch.zeros(0, 7)
        result = post_process(cls_preds, reg_preds)

        assert result["boxes"].shape[0] == 0
        assert result["scores"].shape[0] == 0
        assert result["labels"].shape[0] == 0

    def test_nms_single_box(self):
        """Single box should always be kept if score exceeds threshold."""
        cls_preds = torch.tensor([[5.0, -5.0, -5.0]])  # high score for class 0
        reg_preds = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 0.1]])

        result = post_process(cls_preds, reg_preds, score_threshold=0.3)
        assert result["boxes"].shape[0] == 1
        assert result["labels"][0] == 0


# =============================================================================
# Tests: Multi-sweep Accumulation
# =============================================================================


class TestMultiSweepAccumulation:
    """Tests for multi-sweep radar accumulation utilities."""

    def test_ego_motion_compensation(self):
        """Points should be correctly transformed between frames."""
        # Create a simple translation transform (move 5 meters in x)
        transform = torch.eye(4)
        transform[0, 3] = 5.0  # translate x by 5

        points = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
                [-1.0, -2.0, -3.0],
            ]
        )

        transformed = ego_motion_compensation(points, transform)

        expected = torch.tensor(
            [
                [6.0, 2.0, 3.0],
                [5.0, 0.0, 0.0],
                [4.0, -2.0, -3.0],
            ]
        )

        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_ego_motion_compensation_rotation(self):
        """90-degree rotation around z-axis should swap x and y."""
        transform = torch.eye(4)
        angle = math.pi / 2
        transform[0, 0] = math.cos(angle)
        transform[0, 1] = -math.sin(angle)
        transform[1, 0] = math.sin(angle)
        transform[1, 1] = math.cos(angle)

        points = torch.tensor([[1.0, 0.0, 0.0]])
        transformed = ego_motion_compensation(points, transform)

        # After 90-degree rotation: (1,0,0) -> (0,1,0)
        expected = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.allclose(transformed, expected, atol=1e-5)

    def test_ego_motion_compensation_identity(self):
        """Identity transform should not change points."""
        transform = torch.eye(4)
        points = torch.randn(100, 3)
        transformed = ego_motion_compensation(points, transform)
        assert torch.allclose(transformed, points, atol=1e-6)

    def test_time_lag_feature(self):
        """Time lag feature should correctly encode sweep timing."""
        # 5 sweeps with 50ms between each
        num_sweeps = 5
        sweep_timestamps = torch.tensor([0.0, -0.05, -0.10, -0.15, -0.20])

        # 10 points from different sweeps
        sweep_indices = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        time_lags = compute_time_lag_feature(sweep_indices, sweep_timestamps)

        assert time_lags.shape == (10,)
        # Current sweep points should have 0 time lag
        assert time_lags[0] == 0.0
        assert time_lags[1] == 0.0
        # Sweep 1 should have -0.05 time lag
        assert torch.isclose(time_lags[2], torch.tensor(-0.05))
        assert torch.isclose(time_lags[3], torch.tensor(-0.05))
        # Sweep 4 should have -0.20 time lag
        assert torch.isclose(time_lags[8], torch.tensor(-0.20))

    def test_time_lag_feature_single_sweep(self):
        """Single sweep should produce all-zero time lags."""
        sweep_timestamps = torch.tensor([0.0])
        sweep_indices = torch.zeros(50, dtype=torch.long)
        time_lags = compute_time_lag_feature(sweep_indices, sweep_timestamps)
        assert (time_lags == 0.0).all()


# =============================================================================
# Tests: Data Augmentation
# =============================================================================


class TestDataAugmentation:
    """Tests for data augmentation functions."""

    @pytest.fixture
    def sample_points(self):
        """Generate sample point cloud."""
        torch.manual_seed(123)
        return torch.randn(500, 3)

    @pytest.fixture
    def sample_boxes(self):
        """Generate sample 3D bounding boxes."""
        torch.manual_seed(456)
        boxes = torch.randn(10, 7)
        boxes[:, 3:6] = boxes[:, 3:6].abs() + 0.5  # sizes should be positive
        return boxes

    def test_random_flip_consistency(self, sample_points, sample_boxes):
        """Points and boxes should be flipped consistently along x-axis."""
        flipped_points, flipped_boxes = random_flip(sample_points, sample_boxes, axis=0)

        # X coordinates should be negated
        assert torch.allclose(flipped_points[:, 0], -sample_points[:, 0])
        assert torch.allclose(flipped_boxes[:, 0], -sample_boxes[:, 0])

        # Y and Z should remain unchanged
        assert torch.allclose(flipped_points[:, 1], sample_points[:, 1])
        assert torch.allclose(flipped_points[:, 2], sample_points[:, 2])
        assert torch.allclose(flipped_boxes[:, 1], sample_boxes[:, 1])
        assert torch.allclose(flipped_boxes[:, 2], sample_boxes[:, 2])

        # Yaw should be adjusted: pi - original
        expected_yaw = math.pi - sample_boxes[:, 6]
        assert torch.allclose(flipped_boxes[:, 6], expected_yaw)

    def test_random_flip_y_axis(self, sample_points, sample_boxes):
        """Points and boxes should be flipped consistently along y-axis."""
        flipped_points, flipped_boxes = random_flip(sample_points, sample_boxes, axis=1)

        # Y coordinates should be negated
        assert torch.allclose(flipped_points[:, 1], -sample_points[:, 1])
        assert torch.allclose(flipped_boxes[:, 1], -sample_boxes[:, 1])

        # X and Z should remain unchanged
        assert torch.allclose(flipped_points[:, 0], sample_points[:, 0])
        assert torch.allclose(flipped_boxes[:, 0], sample_boxes[:, 0])

        # Yaw for y-flip: negated
        expected_yaw = -sample_boxes[:, 6]
        assert torch.allclose(flipped_boxes[:, 6], expected_yaw)

    def test_random_flip_double_flip_identity(self, sample_points, sample_boxes):
        """Flipping twice should return to original."""
        flipped_points, flipped_boxes = random_flip(sample_points, sample_boxes, axis=0)
        restored_points, restored_boxes = random_flip(flipped_points, flipped_boxes, axis=0)

        assert torch.allclose(restored_points[:, :3], sample_points[:, :3], atol=1e-6)
        assert torch.allclose(restored_boxes[:, :3], sample_boxes[:, :3], atol=1e-6)

    def test_rotation_consistency(self, sample_points, sample_boxes):
        """Rotation should apply to both points and box centers/yaw."""
        angle = math.pi / 4  # 45 degrees
        rotated_points, rotated_boxes = random_rotation(sample_points, sample_boxes, angle)

        # Check point distances from origin are preserved (rotation preserves distance)
        original_dist = torch.norm(sample_points[:, :2], dim=1)
        rotated_dist = torch.norm(rotated_points[:, :2], dim=1)
        assert torch.allclose(original_dist, rotated_dist, atol=1e-5)

        # Check box center distances from origin are preserved
        original_box_dist = torch.norm(sample_boxes[:, :2], dim=1)
        rotated_box_dist = torch.norm(rotated_boxes[:, :2], dim=1)
        assert torch.allclose(original_box_dist, rotated_box_dist, atol=1e-5)

        # Yaw should be offset by the rotation angle
        expected_yaw = sample_boxes[:, 6] + angle
        assert torch.allclose(rotated_boxes[:, 6], expected_yaw, atol=1e-5)

        # Z coordinate should be unchanged
        assert torch.allclose(rotated_points[:, 2], sample_points[:, 2])
        assert torch.allclose(rotated_boxes[:, 2], sample_boxes[:, 2])

    def test_rotation_zero_angle(self, sample_points, sample_boxes):
        """Zero rotation should not change anything."""
        rotated_points, rotated_boxes = random_rotation(sample_points, sample_boxes, 0.0)
        assert torch.allclose(rotated_points, sample_points, atol=1e-6)
        assert torch.allclose(rotated_boxes, sample_boxes, atol=1e-6)

    @pytest.mark.parametrize("angle", [math.pi / 6, math.pi / 3, math.pi / 2, math.pi])
    def test_rotation_preserves_shape(self, sample_points, sample_boxes, angle):
        """Rotation should not change tensor shapes."""
        rotated_points, rotated_boxes = random_rotation(sample_points, sample_boxes, angle)
        assert rotated_points.shape == sample_points.shape
        assert rotated_boxes.shape == sample_boxes.shape

    def test_scaling_consistency(self, sample_points, sample_boxes):
        """Scaling should apply to points, box centers, and box sizes."""
        scale_factor = 2.0
        scaled_points, scaled_boxes = random_scaling(sample_points, sample_boxes, scale_factor)

        # Points should be scaled
        assert torch.allclose(scaled_points[:, :3], sample_points[:, :3] * scale_factor)

        # Box centers should be scaled
        assert torch.allclose(scaled_boxes[:, :3], sample_boxes[:, :3] * scale_factor)

        # Box sizes should be scaled
        assert torch.allclose(scaled_boxes[:, 3:6], sample_boxes[:, 3:6] * scale_factor)

        # Yaw should NOT be scaled
        assert torch.allclose(scaled_boxes[:, 6], sample_boxes[:, 6])

    def test_scaling_unit_factor(self, sample_points, sample_boxes):
        """Scale factor of 1.0 should not change anything."""
        scaled_points, scaled_boxes = random_scaling(sample_points, sample_boxes, 1.0)
        assert torch.allclose(scaled_points, sample_points)
        assert torch.allclose(scaled_boxes, sample_boxes)

    @pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_scaling_various_factors(self, sample_points, sample_boxes, scale):
        """Scaling should work correctly for various factors."""
        scaled_points, scaled_boxes = random_scaling(sample_points, sample_boxes, scale)

        # Verify spatial coordinates are scaled
        assert torch.allclose(scaled_points[:, 0], sample_points[:, 0] * scale)
        assert torch.allclose(scaled_boxes[:, 3], sample_boxes[:, 3] * scale)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_training_step_simulation(
        self, in_features, num_output_features, grid_size, num_classes, num_anchors
    ):
        """Simulate a complete training step with forward pass, loss, and backward."""
        model = RadarPillarNet(
            in_features=in_features,
            num_output_features=num_output_features,
            grid_h=grid_size[0],
            grid_w=grid_size[1],
            num_classes=num_classes,
            num_anchors=num_anchors,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = RadarPillarNetLoss(num_classes=num_classes)

        B, P, N = 2, 50, 16
        pillars = torch.randn(B, P, N, in_features)
        coords = torch.randint(0, min(grid_size), (B, P, 2))

        H, W = grid_size[0] // 2, grid_size[1] // 2
        targets = {
            "cls_targets": torch.rand(B, num_anchors * num_classes, H, W),
            "reg_targets": torch.randn(B, num_anchors * 7, H, W),
            "vel_targets": torch.randn(B, num_anchors * 2, H, W),
            "dir_targets": torch.randint(0, 2, (B, num_anchors, H, W)),
        }

        # Forward
        output = model(pillars, coords)

        # Loss
        losses = loss_fn(output, targets)

        # Backward
        optimizer.zero_grad()
        losses["loss_total"].backward()

        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

        # Step
        optimizer.step()

        # Verify parameters changed
        output2 = model(pillars, coords)
        # After optimizer step, predictions should differ
        assert not torch.allclose(output["cls"].detach(), output2["cls"].detach())

    def test_inference_pipeline(
        self, in_features, num_output_features, grid_size, num_classes, num_anchors
    ):
        """Test complete inference pipeline from input to final detections."""
        model = RadarPillarNet(
            in_features=in_features,
            num_output_features=num_output_features,
            grid_h=grid_size[0],
            grid_w=grid_size[1],
            num_classes=num_classes,
            num_anchors=num_anchors,
        )
        model.eval()

        B, P, N = 1, 50, 16
        pillars = torch.randn(B, P, N, in_features)
        coords = torch.randint(0, min(grid_size), (B, P, 2))

        with torch.no_grad():
            output = model(pillars, coords)

        # Reshape for post-processing (flatten spatial dims)
        H, W = output["cls"].shape[2], output["cls"].shape[3]
        cls_flat = output["cls"][0].reshape(num_anchors * num_classes, -1).T  # (H*W, A*C)
        # For simplicity, reshape cls to (H*W*A, C)
        cls_preds = output["cls"][0].reshape(num_anchors, num_classes, H, W)
        cls_preds = cls_preds.permute(0, 2, 3, 1).reshape(-1, num_classes)

        reg_preds = output["reg"][0].reshape(num_anchors, 7, H, W)
        reg_preds = reg_preds.permute(0, 2, 3, 1).reshape(-1, 7)

        result = post_process(
            cls_preds, reg_preds, score_threshold=0.1, max_detections=100
        )

        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result
        assert result["boxes"].shape[1] == 7 or result["boxes"].shape[0] == 0
        assert result["scores"].shape[0] == result["boxes"].shape[0]
        assert result["labels"].shape[0] == result["boxes"].shape[0]
