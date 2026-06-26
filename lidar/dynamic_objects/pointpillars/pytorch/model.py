"""
PointPillars: Fast Encoders for Object Detection from Point Clouds.

Complete PyTorch implementation of the PointPillars model for 3D object detection
from LiDAR point clouds. Combines PillarFeatureNet, scatter to BEV pseudo-image,
2D backbone, and detection head with anchor-based predictions.

Reference: Lang et al., "PointPillars: Fast Encoders for Object Detection from
Point Clouds", CVPR 2019.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .anchor_head import AnchorHead
from .backbone import BaseBEVBackbone
from .pillar_feature_net import PillarFeatureNet
from .scatter import PointPillarsScatter


@dataclass
class AnchorConfig:
    """Configuration for anchor generation per class."""

    sizes: List[List[float]]
    heights: List[float]
    rotations: List[float]
    matched_threshold: float = 0.6
    unmatched_threshold: float = 0.45


@dataclass
class PointPillarsConfig:
    """Full configuration for the PointPillars model."""

    # Point cloud voxelization parameters
    voxel_size: List[float] = field(default_factory=lambda: [0.16, 0.16, 4.0])
    point_cloud_range: List[float] = field(
        default_factory=lambda: [-39.68, 0.0, -3.0, 39.68, 69.12, 1.0]
    )
    max_points_per_pillar: int = 32
    max_pillars: int = 16000

    # Pillar Feature Net
    pillar_feat_channels: int = 64
    num_point_features: int = 4  # x, y, z, reflectance

    # Backbone config
    backbone_layer_nums: List[int] = field(default_factory=lambda: [3, 5, 5])
    backbone_layer_strides: List[int] = field(default_factory=lambda: [2, 2, 2])
    backbone_num_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    backbone_upsample_strides: List[int] = field(default_factory=lambda: [1, 2, 4])
    backbone_num_upsample_filters: List[int] = field(
        default_factory=lambda: [128, 128, 128]
    )

    # Detection head
    num_classes: int = 3
    class_names: List[str] = field(
        default_factory=lambda: ["Car", "Pedestrian", "Cyclist"]
    )
    num_dir_bins: int = 2
    use_direction_classifier: bool = True

    # Anchor configuration per class
    anchor_configs: List[Dict[str, Any]] = field(default_factory=list)

    # NMS parameters
    nms_pre_max_size: int = 1000
    nms_post_max_size: int = 300
    nms_iou_threshold: float = 0.5
    score_threshold: float = 0.1

    # Loss weights
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 2.0
    dir_loss_weight: float = 0.2

    def __post_init__(self) -> None:
        """Set default anchor configs if none provided."""
        if not self.anchor_configs:
            if self.num_classes == 3 and "Car" in self.class_names:
                # KITTI defaults
                self.anchor_configs = [
                    {
                        "sizes": [[3.9, 1.6, 1.56]],
                        "heights": [-1.78],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.6,
                        "unmatched_threshold": 0.45,
                    },
                    {
                        "sizes": [[0.8, 0.6, 1.73]],
                        "heights": [-0.6],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.5,
                        "unmatched_threshold": 0.35,
                    },
                    {
                        "sizes": [[1.76, 0.6, 1.73]],
                        "heights": [-0.6],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.5,
                        "unmatched_threshold": 0.35,
                    },
                ]
            elif self.num_classes == 10:
                # nuScenes defaults
                self.anchor_configs = [
                    {
                        "sizes": [[4.63, 1.97, 1.74]],
                        "heights": [-0.95],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.6,
                        "unmatched_threshold": 0.45,
                    },
                    {
                        "sizes": [[6.93, 2.51, 2.84]],
                        "heights": [-0.40],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.55,
                        "unmatched_threshold": 0.40,
                    },
                    {
                        "sizes": [[11.1, 2.95, 3.47]],
                        "heights": [-0.08],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.55,
                        "unmatched_threshold": 0.40,
                    },
                    {
                        "sizes": [[12.29, 2.90, 3.87]],
                        "heights": [0.12],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.5,
                        "unmatched_threshold": 0.35,
                    },
                    {
                        "sizes": [[6.37, 2.85, 3.19]],
                        "heights": [-0.25],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.5,
                        "unmatched_threshold": 0.35,
                    },
                    {
                        "sizes": [[0.73, 0.67, 1.77]],
                        "heights": [-0.98],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.6,
                        "unmatched_threshold": 0.40,
                    },
                    {
                        "sizes": [[2.11, 0.77, 1.47]],
                        "heights": [-1.03],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.5,
                        "unmatched_threshold": 0.30,
                    },
                    {
                        "sizes": [[1.70, 0.60, 1.28]],
                        "heights": [-1.03],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.5,
                        "unmatched_threshold": 0.30,
                    },
                    {
                        "sizes": [[0.41, 0.41, 1.07]],
                        "heights": [-1.28],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.4,
                        "unmatched_threshold": 0.25,
                    },
                    {
                        "sizes": [[2.49, 0.48, 0.98]],
                        "heights": [-1.33],
                        "rotations": [0, 1.5707963],
                        "matched_threshold": 0.5,
                        "unmatched_threshold": 0.35,
                    },
                ]

    @property
    def grid_size(self) -> np.ndarray:
        """Compute the BEV grid size from point cloud range and voxel size."""
        pc_range = np.array(self.point_cloud_range)
        voxel = np.array(self.voxel_size)
        grid = (pc_range[3:6] - pc_range[0:3]) / voxel
        return np.round(grid).astype(np.int64)

    @property
    def num_anchors_per_location(self) -> int:
        """Total number of anchors per spatial location."""
        total = 0
        for cfg in self.anchor_configs:
            num_sizes = len(cfg["sizes"])
            num_rotations = len(cfg["rotations"])
            total += num_sizes * num_rotations
        return total


def _kitti_default_config() -> PointPillarsConfig:
    """Return default PointPillars configuration for KITTI dataset."""
    return PointPillarsConfig(
        voxel_size=[0.16, 0.16, 4.0],
        point_cloud_range=[-39.68, 0.0, -3.0, 39.68, 69.12, 1.0],
        max_points_per_pillar=32,
        max_pillars=16000,
        pillar_feat_channels=64,
        num_point_features=4,
        backbone_layer_nums=[3, 5, 5],
        backbone_layer_strides=[2, 2, 2],
        backbone_num_filters=[64, 128, 256],
        backbone_upsample_strides=[1, 2, 4],
        backbone_num_upsample_filters=[128, 128, 128],
        num_classes=3,
        class_names=["Car", "Pedestrian", "Cyclist"],
        num_dir_bins=2,
        use_direction_classifier=True,
        nms_pre_max_size=1000,
        nms_post_max_size=300,
        nms_iou_threshold=0.5,
        score_threshold=0.1,
        cls_loss_weight=1.0,
        reg_loss_weight=2.0,
        dir_loss_weight=0.2,
    )


def _nuscenes_default_config() -> PointPillarsConfig:
    """Return default PointPillars configuration for nuScenes dataset."""
    return PointPillarsConfig(
        voxel_size=[0.2, 0.2, 8.0],
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_points_per_pillar=20,
        max_pillars=40000,
        pillar_feat_channels=64,
        num_point_features=5,  # x, y, z, reflectance, ring_index
        backbone_layer_nums=[3, 5, 5],
        backbone_layer_strides=[2, 2, 2],
        backbone_num_filters=[64, 128, 256],
        backbone_upsample_strides=[1, 2, 4],
        backbone_num_upsample_filters=[128, 128, 128],
        num_classes=10,
        class_names=[
            "car",
            "truck",
            "bus",
            "trailer",
            "construction_vehicle",
            "pedestrian",
            "motorcycle",
            "bicycle",
            "traffic_cone",
            "barrier",
        ],
        num_dir_bins=2,
        use_direction_classifier=True,
        nms_pre_max_size=1000,
        nms_post_max_size=500,
        nms_iou_threshold=0.2,
        score_threshold=0.1,
        cls_loss_weight=1.0,
        reg_loss_weight=2.0,
        dir_loss_weight=0.2,
    )


def _rotate_points_along_z(
    points: torch.Tensor, angle: torch.Tensor
) -> torch.Tensor:
    """Rotate points along the z-axis by given angles.

    Args:
        points: (N, 3+) tensor of point coordinates.
        angle: (N,) tensor of rotation angles in radians.

    Returns:
        Rotated points tensor with same shape as input.
    """
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    rotated = points.clone()
    rotated[:, 0] = points[:, 0] * cos_a - points[:, 1] * sin_a
    rotated[:, 1] = points[:, 0] * sin_a + points[:, 1] * cos_a
    return rotated


def _boxes3d_to_bev_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Convert 3D boxes to BEV corner representation.

    Args:
        boxes: (N, 7) tensor of boxes [x, y, z, dx, dy, dz, heading].

    Returns:
        (N, 4, 2) tensor of BEV corners.
    """
    centers = boxes[:, :2]
    dims = boxes[:, 3:5]
    angles = boxes[:, 6]

    # Half dimensions
    half_dx = dims[:, 0:1] / 2.0
    half_dy = dims[:, 1:2] / 2.0

    # Corners relative to center (before rotation)
    # Order: front-left, front-right, rear-right, rear-left
    corners = torch.stack(
        [
            torch.cat([half_dx, half_dy], dim=1),
            torch.cat([half_dx, -half_dy], dim=1),
            torch.cat([-half_dx, -half_dy], dim=1),
            torch.cat([-half_dx, half_dy], dim=1),
        ],
        dim=1,
    )  # (N, 4, 2)

    # Rotation matrix
    cos_a = torch.cos(angles).unsqueeze(1).unsqueeze(2)  # (N, 1, 1)
    sin_a = torch.sin(angles).unsqueeze(1).unsqueeze(2)  # (N, 1, 1)

    rot_matrix = torch.cat(
        [
            torch.cat([cos_a, -sin_a], dim=2),
            torch.cat([sin_a, cos_a], dim=2),
        ],
        dim=1,
    )  # (N, 2, 2)

    # Apply rotation: (N, 4, 2) @ (N, 2, 2)^T -> (N, 4, 2)
    rotated_corners = torch.bmm(corners, rot_matrix.transpose(1, 2))

    # Translate to world coordinates
    rotated_corners += centers.unsqueeze(1)

    return rotated_corners


def _nms_bev(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Perform NMS on BEV boxes using axis-aligned bounding box approximation.

    This implementation uses the axis-aligned bounding box of the rotated boxes
    for efficient NMS computation. For production use with rotated boxes, a
    CUDA-accelerated rotated NMS kernel is preferred.

    Args:
        boxes: (N, 7) tensor of 3D boxes [x, y, z, dx, dy, dz, heading].
        scores: (N,) tensor of confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Indices of kept boxes as a 1D tensor.
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Get BEV corners and compute axis-aligned bounding boxes
    corners = _boxes3d_to_bev_corners(boxes)  # (N, 4, 2)
    min_xy = corners.min(dim=1).values  # (N, 2)
    max_xy = corners.max(dim=1).values  # (N, 2)

    # Convert to [x1, y1, x2, y2] format for standard NMS
    bev_boxes = torch.cat([min_xy, max_xy], dim=1)  # (N, 4)

    # Sort by score (descending)
    order = scores.argsort(descending=True)
    bev_boxes = bev_boxes[order]

    # Greedy NMS
    keep: List[int] = []
    suppressed = torch.zeros(bev_boxes.shape[0], dtype=torch.bool, device=boxes.device)

    for i in range(bev_boxes.shape[0]):
        if suppressed[i]:
            continue
        keep.append(i)

        # Compute IoU with remaining boxes
        xx1 = torch.maximum(bev_boxes[i, 0], bev_boxes[i + 1 :, 0])
        yy1 = torch.maximum(bev_boxes[i, 1], bev_boxes[i + 1 :, 1])
        xx2 = torch.minimum(bev_boxes[i, 2], bev_boxes[i + 1 :, 2])
        yy2 = torch.minimum(bev_boxes[i, 3], bev_boxes[i + 1 :, 3])

        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        intersection = inter_w * inter_h

        area_i = (bev_boxes[i, 2] - bev_boxes[i, 0]) * (
            bev_boxes[i, 3] - bev_boxes[i, 1]
        )
        areas_rest = (bev_boxes[i + 1 :, 2] - bev_boxes[i + 1 :, 0]) * (
            bev_boxes[i + 1 :, 3] - bev_boxes[i + 1 :, 1]
        )
        union = area_i + areas_rest - intersection
        iou = intersection / (union + 1e-6)

        # Suppress boxes with high IoU
        suppress_mask = iou > iou_threshold
        suppressed[i + 1 :] |= suppress_mask

    keep_indices = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    # Map back to original indices
    return order[keep_indices]


def _decode_boxes(
    anchors: torch.Tensor,
    box_encodings: torch.Tensor,
) -> torch.Tensor:
    """Decode predicted box regression targets relative to anchors.

    Decoding follows the standard anchor-based encoding scheme:
        dx = (x_pred - x_a) / diag_a
        dy = (y_pred - y_a) / diag_a
        dz = (z_pred - z_a) / h_a
        dw = log(w_pred / w_a)
        dl = log(l_pred / l_a)
        dh = log(h_pred / h_a)
        dtheta = theta_pred - theta_a

    Args:
        anchors: (N, 7) anchor boxes [x, y, z, dx, dy, dz, heading].
        box_encodings: (N, 7) encoded box deltas.

    Returns:
        (N, 7) decoded boxes in the same format as anchors.
    """
    # Anchor dimensions
    xa, ya, za = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    dxa, dya, dza = anchors[:, 3], anchors[:, 4], anchors[:, 5]
    ra = anchors[:, 6]

    # Diagonal of the anchor base (used for x, y normalization)
    diagonal = torch.sqrt(dxa**2 + dya**2)

    # Encoded deltas
    xt, yt, zt = box_encodings[:, 0], box_encodings[:, 1], box_encodings[:, 2]
    dxt, dyt, dzt = box_encodings[:, 3], box_encodings[:, 4], box_encodings[:, 5]
    rt = box_encodings[:, 6]

    # Decode
    x_decoded = xt * diagonal + xa
    y_decoded = yt * diagonal + ya
    z_decoded = zt * dza + za
    dx_decoded = torch.exp(dxt) * dxa
    dy_decoded = torch.exp(dyt) * dya
    dz_decoded = torch.exp(dzt) * dza
    r_decoded = rt + ra

    decoded = torch.stack(
        [x_decoded, y_decoded, z_decoded, dx_decoded, dy_decoded, dz_decoded, r_decoded],
        dim=1,
    )
    return decoded


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in dense detection.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """Initialize FocalLoss.

        Args:
            alpha: Balancing factor for positive/negative examples.
            gamma: Focusing parameter that reduces loss for well-classified examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            predictions: (N, C) raw logits (before sigmoid).
            targets: (N, C) one-hot encoded targets.
            weights: (N,) optional per-sample weights.

        Returns:
            Scalar loss value.
        """
        pred_sigmoid = torch.sigmoid(predictions)
        # Binary cross-entropy component
        bce = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )
        # Focal modulating factor
        pt = torch.where(targets == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        loss = focal_weight * bce

        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        return loss.sum() / max(targets.sum(), 1.0)


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss (Huber loss) for box regression."""

    def __init__(self, beta: float = 1.0 / 9.0) -> None:
        """Initialize SmoothL1Loss.

        Args:
            beta: Transition point between L1 and L2 loss.
        """
        super().__init__()
        self.beta = beta

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute smooth L1 loss.

        Args:
            predictions: (N, 7) predicted box encodings.
            targets: (N, 7) target box encodings.
            weights: (N,) optional per-sample weights.

        Returns:
            Scalar loss value.
        """
        diff = torch.abs(predictions - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta,
            diff - 0.5 * self.beta,
        )

        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        # Normalize by number of positive anchors
        num_pos = max((weights > 0).sum().item(), 1.0) if weights is not None else max(loss.shape[0], 1)
        return loss.sum() / num_pos


class PointPillars(nn.Module):
    """PointPillars model for 3D object detection from LiDAR point clouds.

    Architecture:
        1. PillarFeatureNet: Encodes raw points within each pillar into a fixed
           feature vector using PointNet-style max pooling.
        2. PointPillarsScatter: Scatters pillar features back to a 2D BEV
           pseudo-image for efficient 2D convolution processing.
        3. BaseBEVBackbone: Multi-scale 2D CNN backbone with top-down feature
           aggregation (similar to FPN) operating on the BEV pseudo-image.
        4. AnchorHead: Dense prediction head that outputs per-anchor class scores,
           box regression, and direction classification.

    The model supports both training (returns loss dict) and inference
    (returns decoded predictions after NMS).
    """

    def __init__(self, config: PointPillarsConfig) -> None:
        """Initialize PointPillars model.

        Args:
            config: Model configuration dataclass containing all hyperparameters.
        """
        super().__init__()
        self.config = config
        self.register_buffer(
            "point_cloud_range",
            torch.tensor(config.point_cloud_range, dtype=torch.float32),
        )
        self.register_buffer(
            "voxel_size",
            torch.tensor(config.voxel_size, dtype=torch.float32),
        )

        grid_size = config.grid_size
        self._grid_size_x = int(grid_size[0])
        self._grid_size_y = int(grid_size[1])

        # Sub-modules
        self.pillar_feature_net = PillarFeatureNet(
            num_input_features=config.num_point_features,
            num_filters=[config.pillar_feat_channels],
            pillar_size=config.voxel_size,
            pc_range=config.point_cloud_range,
            max_points_per_pillar=config.max_points_per_pillar,
        )

        self.scatter = PointPillarsScatter(
            num_bev_features=config.pillar_feat_channels,
            grid_size_x=self._grid_size_x,
            grid_size_y=self._grid_size_y,
        )

        self.backbone = BaseBEVBackbone(
            input_channels=config.pillar_feat_channels,
            layer_nums=config.backbone_layer_nums,
            layer_strides=config.backbone_layer_strides,
            num_filters=config.backbone_num_filters,
            upsample_strides=config.backbone_upsample_strides,
            num_upsample_filters=config.backbone_num_upsample_filters,
        )

        # Compute the number of anchors per location across all classes
        num_anchors_per_location = config.num_anchors_per_location

        # The backbone output channels is sum of all upsample filter channels
        backbone_output_channels = sum(config.backbone_num_upsample_filters)

        self.anchor_head = AnchorHead(
            input_channels=backbone_output_channels,
            num_classes=config.num_classes,
            num_anchors_per_location=num_anchors_per_location,
            box_code_size=7,
            num_dir_bins=config.num_dir_bins,
            use_direction_classifier=config.use_direction_classifier,
        )

        # Loss functions
        self.cls_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.reg_loss_fn = SmoothL1Loss(beta=1.0 / 9.0)
        self.dir_loss_fn = nn.CrossEntropyLoss(reduction="none")

        # Generate anchors and register as buffer
        anchors = self._generate_anchors()
        self.register_buffer("anchors", anchors)

    def _generate_anchors(self) -> torch.Tensor:
        """Generate multi-class anchors across the BEV grid.

        Anchors are generated at each spatial location of the feature map for
        every class, with multiple sizes and rotations per class.

        Returns:
            Tensor of shape (num_anchors_total, 7) containing all anchors as
            [x, y, z, dx, dy, dz, heading].
        """
        config = self.config
        pc_range = np.array(config.point_cloud_range)
        voxel_size = np.array(config.voxel_size)

        # Feature map size (after first stride of backbone)
        feature_map_stride = config.backbone_layer_strides[0]
        feature_x = self._grid_size_x // feature_map_stride
        feature_y = self._grid_size_y // feature_map_stride

        # Anchor center offsets on the BEV grid
        x_offset = voxel_size[0] * feature_map_stride / 2.0
        y_offset = voxel_size[1] * feature_map_stride / 2.0

        x_centers = np.arange(0, feature_x) * voxel_size[0] * feature_map_stride + pc_range[0] + x_offset
        y_centers = np.arange(0, feature_y) * voxel_size[1] * feature_map_stride + pc_range[1] + y_offset

        # Create meshgrid of x, y centers
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        num_locations = xx.shape[0]
        all_anchors = []

        for anchor_cfg in config.anchor_configs:
            sizes = anchor_cfg["sizes"]
            heights = anchor_cfg["heights"]
            rotations = anchor_cfg["rotations"]

            for size_idx, size in enumerate(sizes):
                height = heights[size_idx] if size_idx < len(heights) else heights[0]
                for rotation in rotations:
                    # Create anchors at all locations for this size/rotation
                    anchors = np.zeros((num_locations, 7), dtype=np.float32)
                    anchors[:, 0] = xx  # x
                    anchors[:, 1] = yy  # y
                    anchors[:, 2] = height  # z
                    anchors[:, 3] = size[0]  # dx (length)
                    anchors[:, 4] = size[1]  # dy (width)
                    anchors[:, 5] = size[2]  # dz (height)
                    anchors[:, 6] = rotation  # heading
                    all_anchors.append(anchors)

        all_anchors_np = np.concatenate(all_anchors, axis=0)
        return torch.from_numpy(all_anchors_np)

    def forward(
        self,
        pillars: torch.Tensor,
        num_points_per_pillar: torch.Tensor,
        pillar_coords: torch.Tensor,
        batch_size: int,
        gt_boxes: Optional[torch.Tensor] = None,
        gt_classes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the PointPillars model.

        In training mode (self.training is True and gt_boxes is provided),
        returns a dict of losses. In eval mode, returns decoded predictions.

        Args:
            pillars: (total_pillars, max_points, num_features) padded points
                within each pillar across the batch.
            num_points_per_pillar: (total_pillars,) number of valid points in
                each pillar.
            pillar_coords: (total_pillars, 3) pillar coordinates as
                [batch_idx, grid_x, grid_y].
            batch_size: Number of samples in the batch.
            gt_boxes: (B, max_gt, 7) ground truth boxes for training.
                Each box is [x, y, z, dx, dy, dz, heading].
            gt_classes: (B, max_gt) integer class labels (1-indexed, 0 = padding).

        Returns:
            In training mode: Dict with keys 'cls_loss', 'reg_loss', 'dir_loss',
                and 'total_loss'.
            In eval mode: Dict with keys 'pred_boxes', 'pred_scores',
                'pred_labels' (lists of tensors, one per batch sample).
        """
        # 1. Pillar Feature Extraction
        pillar_features = self.pillar_feature_net(
            pillars, num_points_per_pillar, pillar_coords
        )  # (total_pillars, C)

        # 2. Scatter to BEV pseudo-image
        bev_map = self.scatter(
            pillar_features, pillar_coords, batch_size
        )  # (B, C, H, W)

        # 3. BEV Backbone
        spatial_features = self.backbone(bev_map)  # (B, C_out, H', W')

        # 4. Detection Head
        cls_preds, box_preds, dir_preds = self.anchor_head(spatial_features)
        # cls_preds: (B, num_anchors, num_classes)
        # box_preds: (B, num_anchors, 7)
        # dir_preds: (B, num_anchors, num_dir_bins)

        if self.training and gt_boxes is not None and gt_classes is not None:
            return self._compute_losses(
                cls_preds, box_preds, dir_preds, gt_boxes, gt_classes, batch_size
            )
        else:
            return self.post_processing(cls_preds, box_preds, dir_preds, batch_size)

    def _assign_targets(
        self,
        gt_boxes_single: torch.Tensor,
        gt_classes_single: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign ground truth targets to anchors for a single sample.

        Uses IoU-based matching with per-class thresholds.

        Args:
            gt_boxes_single: (M, 7) ground truth boxes (non-padded).
            gt_classes_single: (M,) class labels.

        Returns:
            cls_targets: (num_anchors,) integer class targets (0=background,
                1..C=foreground, -1=ignore).
            box_targets: (num_anchors, 7) regression targets for positive anchors.
            dir_targets: (num_anchors,) direction bin targets.
            pos_mask: (num_anchors,) boolean mask of positive anchors.
        """
        num_anchors = self.anchors.shape[0]
        device = self.anchors.device

        cls_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)
        box_targets = torch.zeros(num_anchors, 7, dtype=torch.float32, device=device)
        dir_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)
        pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)

        if gt_boxes_single.shape[0] == 0:
            return cls_targets, box_targets, dir_targets, pos_mask

        # Compute IoU between anchors and gt boxes using BEV AABB approximation
        anchor_corners = _boxes3d_to_bev_corners(self.anchors)  # (A, 4, 2)
        gt_corners = _boxes3d_to_bev_corners(gt_boxes_single)  # (G, 4, 2)

        # Axis-aligned bounding boxes for IoU
        anchor_min = anchor_corners.min(dim=1).values  # (A, 2)
        anchor_max = anchor_corners.max(dim=1).values  # (A, 2)
        gt_min = gt_corners.min(dim=1).values  # (G, 2)
        gt_max = gt_corners.max(dim=1).values  # (G, 2)

        # Compute pairwise IoU: (A, G)
        # Intersection
        inter_min = torch.maximum(
            anchor_min.unsqueeze(1), gt_min.unsqueeze(0)
        )  # (A, G, 2)
        inter_max = torch.minimum(
            anchor_max.unsqueeze(1), gt_max.unsqueeze(0)
        )  # (A, G, 2)
        inter_wh = (inter_max - inter_min).clamp(min=0)  # (A, G, 2)
        intersection = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # (A, G)

        # Areas
        anchor_area = (anchor_max[:, 0] - anchor_min[:, 0]) * (
            anchor_max[:, 1] - anchor_min[:, 1]
        )  # (A,)
        gt_area = (gt_max[:, 0] - gt_min[:, 0]) * (
            gt_max[:, 1] - gt_min[:, 1]
        )  # (G,)

        union = (
            anchor_area.unsqueeze(1) + gt_area.unsqueeze(0) - intersection
        )  # (A, G)
        iou_matrix = intersection / (union + 1e-6)  # (A, G)

        # Assign each anchor to its best-matching GT
        max_iou_per_anchor, matched_gt_idx = iou_matrix.max(dim=1)  # (A,), (A,)

        # Also ensure each GT has at least one anchor assigned
        max_iou_per_gt, best_anchor_per_gt = iou_matrix.max(dim=0)  # (G,), (G,)

        # Determine positive/negative using thresholds
        # Use the matched threshold of the first anchor config as default
        matched_threshold = self.config.anchor_configs[0]["matched_threshold"]
        unmatched_threshold = self.config.anchor_configs[0]["unmatched_threshold"]

        # Positive anchors: IoU >= matched_threshold
        positive_mask = max_iou_per_anchor >= matched_threshold
        # Force best anchor per GT to be positive
        positive_mask[best_anchor_per_gt] = True

        # Negative anchors: IoU < unmatched_threshold
        negative_mask = max_iou_per_anchor < unmatched_threshold

        # Ignore zone: between thresholds (neither positive nor negative)
        ignore_mask = ~positive_mask & ~negative_mask

        # Assign targets
        pos_mask = positive_mask
        cls_targets[positive_mask] = gt_classes_single[matched_gt_idx[positive_mask]]
        cls_targets[ignore_mask] = -1  # Ignore during loss computation

        # Encode box targets for positive anchors
        if positive_mask.any():
            matched_gt = gt_boxes_single[matched_gt_idx[positive_mask]]  # (P, 7)
            pos_anchors = self.anchors[positive_mask]  # (P, 7)
            box_targets[positive_mask] = self._encode_boxes(pos_anchors, matched_gt)

            # Direction targets: discretize heading into bins
            heading_diff = matched_gt[:, 6] - pos_anchors[:, 6]
            # Normalize to [0, 2*pi)
            heading_diff = heading_diff % (2 * np.pi)
            # Bin into num_dir_bins
            dir_targets[positive_mask] = (
                (heading_diff / (2 * np.pi / self.config.num_dir_bins))
                .long()
                .clamp(0, self.config.num_dir_bins - 1)
            )

        return cls_targets, box_targets, dir_targets, pos_mask

    def _encode_boxes(
        self, anchors: torch.Tensor, gt_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Encode ground truth boxes relative to anchors.

        Args:
            anchors: (N, 7) anchor boxes.
            gt_boxes: (N, 7) corresponding ground truth boxes.

        Returns:
            (N, 7) encoded regression targets.
        """
        xa, ya, za = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        dxa, dya, dza = anchors[:, 3], anchors[:, 4], anchors[:, 5]
        ra = anchors[:, 6]

        xg, yg, zg = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        dxg, dyg, dzg = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        rg = gt_boxes[:, 6]

        diagonal = torch.sqrt(dxa**2 + dya**2)

        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        rt = rg - ra

        return torch.stack([xt, yt, zt, dxt, dyt, dzt, rt], dim=1)

    def _compute_losses(
        self,
        cls_preds: torch.Tensor,
        box_preds: torch.Tensor,
        dir_preds: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses for classification, regression, and direction.

        Args:
            cls_preds: (B, num_anchors, num_classes) class logits.
            box_preds: (B, num_anchors, 7) box regression predictions.
            dir_preds: (B, num_anchors, num_dir_bins) direction logits.
            gt_boxes: (B, max_gt, 7) ground truth boxes.
            gt_classes: (B, max_gt) ground truth class labels.
            batch_size: Number of samples in batch.

        Returns:
            Dict containing 'cls_loss', 'reg_loss', 'dir_loss', 'total_loss'.
        """
        total_cls_loss = torch.tensor(0.0, device=cls_preds.device)
        total_reg_loss = torch.tensor(0.0, device=cls_preds.device)
        total_dir_loss = torch.tensor(0.0, device=cls_preds.device)

        for b in range(batch_size):
            # Filter out padded GT boxes (class == 0)
            valid_mask = gt_classes[b] > 0
            gt_boxes_b = gt_boxes[b][valid_mask]
            gt_classes_b = gt_classes[b][valid_mask]

            # Assign targets
            cls_targets, box_targets, dir_targets, pos_mask = self._assign_targets(
                gt_boxes_b, gt_classes_b
            )

            # Classification loss
            # Create one-hot targets
            num_anchors = cls_targets.shape[0]
            cls_targets_onehot = torch.zeros(
                num_anchors,
                self.config.num_classes,
                dtype=torch.float32,
                device=cls_preds.device,
            )
            foreground_mask = cls_targets > 0
            if foreground_mask.any():
                # Class labels are 1-indexed; convert to 0-indexed for one-hot
                cls_targets_onehot[foreground_mask, cls_targets[foreground_mask] - 1] = 1.0

            # Only compute loss for non-ignored anchors
            valid_anchors = cls_targets >= 0  # background (0) + foreground (>0)
            cls_weights = valid_anchors.float()

            batch_cls_loss = self.cls_loss_fn(
                cls_preds[b], cls_targets_onehot, cls_weights
            )
            total_cls_loss = total_cls_loss + batch_cls_loss

            # Regression loss (only for positive anchors)
            if pos_mask.any():
                batch_reg_loss = self.reg_loss_fn(
                    box_preds[b][pos_mask],
                    box_targets[pos_mask],
                    weights=None,
                )
                total_reg_loss = total_reg_loss + batch_reg_loss

                # Direction classification loss
                if self.config.use_direction_classifier:
                    dir_loss = self.dir_loss_fn(
                        dir_preds[b][pos_mask], dir_targets[pos_mask]
                    )
                    total_dir_loss = total_dir_loss + dir_loss.mean()

        # Average over batch
        total_cls_loss = total_cls_loss / batch_size
        total_reg_loss = total_reg_loss / batch_size
        total_dir_loss = total_dir_loss / batch_size

        # Weighted sum
        total_loss = (
            self.config.cls_loss_weight * total_cls_loss
            + self.config.reg_loss_weight * total_reg_loss
            + self.config.dir_loss_weight * total_dir_loss
        )

        return {
            "cls_loss": total_cls_loss,
            "reg_loss": total_reg_loss,
            "dir_loss": total_dir_loss,
            "total_loss": total_loss,
        }

    def post_processing(
        self,
        cls_preds: torch.Tensor,
        box_preds: torch.Tensor,
        dir_preds: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, List[torch.Tensor]]:
        """Decode predictions, apply score filtering, and run NMS.

        Args:
            cls_preds: (B, num_anchors, num_classes) class logits.
            box_preds: (B, num_anchors, 7) encoded box predictions.
            dir_preds: (B, num_anchors, num_dir_bins) direction logits.
            batch_size: Number of samples in the batch.

        Returns:
            Dict with:
                'pred_boxes': List of (K_i, 7) tensors of predicted boxes.
                'pred_scores': List of (K_i,) tensors of confidence scores.
                'pred_labels': List of (K_i,) tensors of class labels (1-indexed).
        """
        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        # Apply sigmoid to get class probabilities
        cls_scores = torch.sigmoid(cls_preds)  # (B, num_anchors, num_classes)

        for b in range(batch_size):
            # Decode boxes
            decoded_boxes = _decode_boxes(
                self.anchors, box_preds[b]
            )  # (num_anchors, 7)

            # Apply direction correction
            if self.config.use_direction_classifier:
                dir_labels = dir_preds[b].argmax(dim=-1)  # (num_anchors,)
                # Correct the heading based on direction bin
                dir_rot = (
                    dir_labels.float() * (2 * np.pi / self.config.num_dir_bins)
                )
                # Adjust heading to match predicted direction
                period = 2 * np.pi / self.config.num_dir_bins
                heading = decoded_boxes[:, 6]
                dir_offset = heading - self.anchors[:, 6]
                # Align to predicted direction bin
                dir_offset_corrected = (
                    dir_offset - dir_rot + period / 2
                ) % period - period / 2 + dir_rot
                decoded_boxes[:, 6] = self.anchors[:, 6] + dir_offset_corrected

            # Per-class NMS
            batch_boxes: List[torch.Tensor] = []
            batch_scores: List[torch.Tensor] = []
            batch_labels: List[torch.Tensor] = []

            for cls_idx in range(self.config.num_classes):
                class_scores = cls_scores[b, :, cls_idx]  # (num_anchors,)

                # Score filtering
                score_mask = class_scores > self.config.score_threshold
                if not score_mask.any():
                    continue

                filtered_scores = class_scores[score_mask]
                filtered_boxes = decoded_boxes[score_mask]

                # Keep top-K before NMS
                if filtered_scores.shape[0] > self.config.nms_pre_max_size:
                    topk_scores, topk_idx = filtered_scores.topk(
                        self.config.nms_pre_max_size
                    )
                    filtered_scores = topk_scores
                    filtered_boxes = filtered_boxes[topk_idx]

                # NMS
                keep_idx = _nms_bev(
                    filtered_boxes,
                    filtered_scores,
                    self.config.nms_iou_threshold,
                )

                # Post-NMS top-K
                if keep_idx.shape[0] > self.config.nms_post_max_size:
                    keep_idx = keep_idx[: self.config.nms_post_max_size]

                kept_boxes = filtered_boxes[keep_idx]
                kept_scores = filtered_scores[keep_idx]
                kept_labels = torch.full(
                    (keep_idx.shape[0],),
                    cls_idx + 1,  # 1-indexed class label
                    dtype=torch.long,
                    device=cls_preds.device,
                )

                batch_boxes.append(kept_boxes)
                batch_scores.append(kept_scores)
                batch_labels.append(kept_labels)

            if batch_boxes:
                all_boxes.append(torch.cat(batch_boxes, dim=0))
                all_scores.append(torch.cat(batch_scores, dim=0))
                all_labels.append(torch.cat(batch_labels, dim=0))
            else:
                all_boxes.append(
                    torch.zeros(0, 7, device=cls_preds.device)
                )
                all_scores.append(
                    torch.zeros(0, device=cls_preds.device)
                )
                all_labels.append(
                    torch.zeros(0, dtype=torch.long, device=cls_preds.device)
                )

        return {
            "pred_boxes": all_boxes,
            "pred_scores": all_scores,
            "pred_labels": all_labels,
        }

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "PointPillars":
        """Construct a PointPillars model from a YAML configuration file.

        The YAML file should have a top-level 'model' key containing all
        configuration parameters matching the PointPillarsConfig dataclass fields.

        Example YAML structure:
            model:
              voxel_size: [0.16, 0.16, 4.0]
              point_cloud_range: [-39.68, 0.0, -3.0, 39.68, 69.12, 1.0]
              max_points_per_pillar: 32
              max_pillars: 16000
              ...
            dataset: kitti  # or "nuscenes"

        If 'dataset' key is provided without a full 'model' block, uses the
        appropriate default config.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Initialized PointPillars model.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If required config fields are missing.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # Check for dataset shorthand
        dataset = raw_config.get("dataset", "").lower()
        if dataset == "nuscenes" and "model" not in raw_config:
            return cls(_nuscenes_default_config())
        elif dataset == "kitti" and "model" not in raw_config:
            return cls(_kitti_default_config())

        # Build config from model dict
        model_dict = raw_config.get("model", raw_config)

        # Start from the appropriate default based on dataset name
        if dataset == "nuscenes":
            config = _nuscenes_default_config()
        else:
            config = _kitti_default_config()

        # Override with values from YAML
        config_fields = {f.name for f in config.__dataclass_fields__.values()}
        for key, value in model_dict.items():
            if key in config_fields and value is not None:
                setattr(config, key, value)

        # Re-run post_init to set anchor configs if class names changed
        config.__post_init__()

        return cls(config)

    @staticmethod
    def kitti_model() -> "PointPillars":
        """Create a PointPillars model with default KITTI configuration.

        Returns:
            PointPillars model configured for KITTI (3 classes: Car, Pedestrian,
            Cyclist; front-view only point cloud range).
        """
        return PointPillars(_kitti_default_config())

    @staticmethod
    def nuscenes_model() -> "PointPillars":
        """Create a PointPillars model with default nuScenes configuration.

        Returns:
            PointPillars model configured for nuScenes (10 classes; 360-degree
            point cloud range).
        """
        return PointPillars(_nuscenes_default_config())
