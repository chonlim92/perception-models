"""
Detection head with anchor-based predictions for RadarPillarNet.

Implements a single-shot multi-class anchor-based detection head that predicts:
- 3D bounding boxes: (x, y, z, w, l, h, theta)
- Classification scores: per-class confidence
- Direction classification: binary head for heading disambiguation
- Velocity: (vx, vy) for dynamic object tracking

Anchor configuration follows nuScenes/automotive conventions:
- Car: [4.7, 2.1, 1.7] at rotations [0, pi/2]
- Truck: [10.0, 2.5, 3.2] at rotations [0, pi/2]
- Pedestrian: [0.7, 0.7, 1.8] at rotations [0, pi/2]
- Cyclist: [1.8, 0.8, 1.5] at rotations [0, pi/2]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class AnchorConfig:
    """Configuration for a single anchor class.

    Attributes:
        class_name: Name of the object class.
        sizes: List of [width, length, height] anchor sizes in meters.
        rotations: List of rotation angles in radians.
        anchor_z: Height of anchor center above ground.
        matched_threshold: IoU threshold for positive match.
        unmatched_threshold: IoU threshold below which is negative.
    """

    class_name: str
    sizes: List[List[float]]
    rotations: List[float] = field(default_factory=lambda: [0.0, np.pi / 2])
    anchor_z: float = -1.0
    matched_threshold: float = 0.6
    unmatched_threshold: float = 0.45


class AnchorGenerator:
    """Generates dense anchors for multi-class detection on BEV grid.

    Creates a set of 3D anchors at each spatial position in the feature map,
    with multiple sizes and rotations per class. Anchors are parameterized as
    (x, y, z, w, l, h, theta) in the ego-vehicle coordinate frame.
    """

    def __init__(
        self,
        anchor_configs: List[AnchorConfig],
        feature_map_size: Tuple[int, int],
        point_range: List[float],
    ) -> None:
        """Initialize anchor generator.

        Args:
            anchor_configs: List of AnchorConfig for each class.
            feature_map_size: (H, W) spatial size of the feature map.
            point_range: [x_min, y_min, z_min, x_max, y_max, z_max] detection range.
        """
        self.anchor_configs = anchor_configs
        self.feature_map_size = feature_map_size
        self.point_range = point_range
        self._anchors: Optional[torch.Tensor] = None
        self._anchors_per_class: Optional[List[torch.Tensor]] = None

    @property
    def num_classes(self) -> int:
        """Number of object classes."""
        return len(self.anchor_configs)

    @property
    def num_anchors_per_location(self) -> int:
        """Total number of anchors at each spatial position."""
        total = 0
        for cfg in self.anchor_configs:
            total += len(cfg.sizes) * len(cfg.rotations)
        return total

    def generate_anchors(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Generate all anchors for the feature map.

        Returns:
            (H, W, num_anchors_per_location, 7) tensor of anchors
            where each anchor is (x, y, z, w, l, h, theta).
        """
        if self._anchors is not None and self._anchors.device == device:
            return self._anchors

        h, w = self.feature_map_size
        x_min, y_min = self.point_range[0], self.point_range[1]
        x_max, y_max = self.point_range[3], self.point_range[4]

        # Compute anchor center positions (center of each grid cell)
        x_stride = (x_max - x_min) / w
        y_stride = (y_max - y_min) / h

        x_centers = torch.linspace(
            x_min + x_stride / 2, x_max - x_stride / 2, w, device=device
        )
        y_centers = torch.linspace(
            y_min + y_stride / 2, y_max - y_stride / 2, h, device=device
        )

        # Create meshgrid: (H, W)
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")  # (H, W) each

        all_anchors = []
        self._anchors_per_class = []

        for cfg in self.anchor_configs:
            class_anchors = []
            for size in cfg.sizes:
                for rotation in cfg.rotations:
                    # Create anchor at each position
                    # anchor: (x, y, z, w, l, h, theta)
                    anchor = torch.zeros(h, w, 7, device=device)
                    anchor[:, :, 0] = xx  # x center
                    anchor[:, :, 1] = yy  # y center
                    anchor[:, :, 2] = cfg.anchor_z  # z center
                    anchor[:, :, 3] = size[0]  # width
                    anchor[:, :, 4] = size[1]  # length
                    anchor[:, :, 5] = size[2]  # height
                    anchor[:, :, 6] = rotation  # theta

                    class_anchors.append(anchor)
                    all_anchors.append(anchor)

            # Stack class anchors: (H, W, num_anchors_this_class, 7)
            class_anchors_tensor = torch.stack(class_anchors, dim=2)
            self._anchors_per_class.append(class_anchors_tensor)

        # Stack all anchors: (H, W, total_anchors, 7)
        self._anchors = torch.stack(all_anchors, dim=2)
        return self._anchors

    def get_anchors_per_class(
        self, device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
        """Get anchors grouped by class.

        Returns:
            List of (H, W, anchors_per_class, 7) tensors, one per class.
        """
        if self._anchors_per_class is None:
            self.generate_anchors(device)
        return self._anchors_per_class


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Decode predicted box offsets relative to anchors.

    Uses the standard anchor-based encoding:
        dx = (x_pred - x_a) / diagonal_a
        dy = (y_pred - y_a) / diagonal_a
        dz = (z_pred - z_a) / h_a
        dw = log(w_pred / w_a)
        dl = log(l_pred / l_a)
        dh = log(h_pred / h_a)
        dtheta = theta_pred - theta_a

    Args:
        anchors: (..., 7) anchor boxes (x, y, z, w, l, h, theta).
        deltas: (..., 7) predicted offsets (dx, dy, dz, dw, dl, dh, dtheta).

    Returns:
        (..., 7) decoded boxes (x, y, z, w, l, h, theta).
    """
    # Extract anchor parameters
    xa, ya, za = anchors[..., 0], anchors[..., 1], anchors[..., 2]
    wa, la, ha = anchors[..., 3], anchors[..., 4], anchors[..., 5]
    theta_a = anchors[..., 6]

    # Diagonal of anchor base (for normalization of x, y offsets)
    diagonal = torch.sqrt(wa ** 2 + la ** 2)  # (...)

    # Extract deltas
    dx, dy, dz = deltas[..., 0], deltas[..., 1], deltas[..., 2]
    dw, dl, dh = deltas[..., 3], deltas[..., 4], deltas[..., 5]
    dtheta = deltas[..., 6]

    # Decode
    x_pred = dx * diagonal + xa
    y_pred = dy * diagonal + ya
    z_pred = dz * ha + za
    w_pred = torch.exp(dw) * wa
    l_pred = torch.exp(dl) * la
    h_pred = torch.exp(dh) * ha
    theta_pred = dtheta + theta_a

    # Stack decoded boxes
    decoded = torch.stack(
        [x_pred, y_pred, z_pred, w_pred, l_pred, h_pred, theta_pred], dim=-1
    )

    return decoded


def encode_boxes(boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Encode ground truth boxes as offsets from anchors (inverse of decode).

    Args:
        boxes: (..., 7) ground truth boxes (x, y, z, w, l, h, theta).
        anchors: (..., 7) anchor boxes (x, y, z, w, l, h, theta).

    Returns:
        (..., 7) encoded deltas.
    """
    xa, ya, za = anchors[..., 0], anchors[..., 1], anchors[..., 2]
    wa, la, ha = anchors[..., 3], anchors[..., 4], anchors[..., 5]
    theta_a = anchors[..., 6]

    xg, yg, zg = boxes[..., 0], boxes[..., 1], boxes[..., 2]
    wg, lg, hg = boxes[..., 3], boxes[..., 4], boxes[..., 5]
    theta_g = boxes[..., 6]

    diagonal = torch.sqrt(wa ** 2 + la ** 2)

    dx = (xg - xa) / diagonal
    dy = (yg - ya) / diagonal
    dz = (zg - za) / ha
    dw = torch.log(wg / wa.clamp(min=1e-7))
    dl = torch.log(lg / la.clamp(min=1e-7))
    dh = torch.log(hg / ha.clamp(min=1e-7))
    dtheta = theta_g - theta_a

    return torch.stack([dx, dy, dz, dw, dl, dh, dtheta], dim=-1)


def nms_rotated(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Axis-aligned NMS approximation for rotated boxes.

    Approximates rotated NMS by computing IoU on axis-aligned bounding boxes
    derived from the rotated box corners. For a full implementation, use
    torchvision.ops.nms or a custom rotated IoU kernel.

    Args:
        boxes: (N, 7) boxes (x, y, z, w, l, h, theta).
        scores: (N,) confidence scores.
        threshold: IoU threshold for suppression.

    Returns:
        (K,) indices of kept boxes, sorted by score (descending).
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Convert rotated boxes to axis-aligned bounding boxes for NMS
    x, y = boxes[:, 0], boxes[:, 1]
    w, l = boxes[:, 3], boxes[:, 4]
    theta = boxes[:, 6]

    # Compute corner offsets considering rotation
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Half-dimensions
    hw, hl = w / 2, l / 2

    # Four corners relative to center
    dx = torch.stack([hw, hw, -hw, -hw], dim=1)  # (N, 4)
    dy = torch.stack([hl, -hl, -hl, hl], dim=1)  # (N, 4)

    # Rotate corners
    corners_x = cos_t.unsqueeze(1) * dx - sin_t.unsqueeze(1) * dy + x.unsqueeze(1)
    corners_y = sin_t.unsqueeze(1) * dx + cos_t.unsqueeze(1) * dy + y.unsqueeze(1)

    # Axis-aligned bounding box
    x1 = corners_x.min(dim=1).values
    y1 = corners_y.min(dim=1).values
    x2 = corners_x.max(dim=1).values
    y2 = corners_y.max(dim=1).values

    # Standard NMS on AABB
    aabb = torch.stack([x1, y1, x2, y2], dim=1)  # (N, 4)

    # Sort by score
    order = scores.argsort(descending=True)
    aabb = aabb[order]

    keep = []
    suppressed = torch.zeros(aabb.shape[0], dtype=torch.bool, device=boxes.device)

    for i in range(aabb.shape[0]):
        if suppressed[i]:
            continue
        keep.append(order[i])

        # Compute IoU with remaining boxes
        ix1 = torch.max(aabb[i, 0], aabb[i + 1 :, 0])
        iy1 = torch.max(aabb[i, 1], aabb[i + 1 :, 1])
        ix2 = torch.min(aabb[i, 2], aabb[i + 1 :, 2])
        iy2 = torch.min(aabb[i, 3], aabb[i + 1 :, 3])

        inter_w = (ix2 - ix1).clamp(min=0)
        inter_h = (iy2 - iy1).clamp(min=0)
        intersection = inter_w * inter_h

        area_i = (aabb[i, 2] - aabb[i, 0]) * (aabb[i, 3] - aabb[i, 1])
        area_j = (aabb[i + 1 :, 2] - aabb[i + 1 :, 0]) * (
            aabb[i + 1 :, 3] - aabb[i + 1 :, 1]
        )
        union = area_i + area_j - intersection
        iou = intersection / union.clamp(min=1e-7)

        # Suppress overlapping boxes
        suppress_mask = iou > threshold
        suppressed[i + 1 :] |= suppress_mask

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class RadarAnchorHead(nn.Module):
    """Multi-class anchor-based detection head with velocity prediction.

    For each anchor at each spatial position, predicts:
    - Box regression: 7 values (dx, dy, dz, dw, dl, dh, dtheta)
    - Classification: num_classes scores
    - Direction: 2 values (binary classification for heading)
    - Velocity: 2 values (vx, vy in ego frame)

    The head uses separate Conv2d branches for each prediction type.
    """

    def __init__(
        self,
        in_channels: int = 384,
        num_classes: int = 4,
        anchor_configs: Optional[List[AnchorConfig]] = None,
        feature_map_size: Tuple[int, int] = (256, 256),
        point_range: Optional[List[float]] = None,
        nms_threshold: float = 0.2,
        score_threshold: float = 0.1,
        max_detections: int = 300,
    ) -> None:
        """Initialize detection head.

        Args:
            in_channels: Number of input feature channels from backbone (default 384).
            num_classes: Number of object classes (default 4: car, truck, ped, cyclist).
            anchor_configs: List of anchor configurations per class. Uses defaults if None.
            feature_map_size: (H, W) spatial size of the input feature map.
            point_range: Detection range [x_min, y_min, z_min, x_max, y_max, z_max].
            nms_threshold: IoU threshold for NMS post-processing.
            score_threshold: Minimum score to keep a detection.
            max_detections: Maximum number of detections after NMS.
        """
        super().__init__()

        self.num_classes = num_classes
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.feature_map_size = feature_map_size

        if point_range is None:
            point_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.point_range = point_range

        # Default anchor configs for automotive radar
        if anchor_configs is None:
            anchor_configs = [
                AnchorConfig(
                    class_name="car",
                    sizes=[[4.7, 2.1, 1.7]],
                    rotations=[0.0, np.pi / 2],
                    anchor_z=-1.0,
                    matched_threshold=0.6,
                    unmatched_threshold=0.45,
                ),
                AnchorConfig(
                    class_name="truck",
                    sizes=[[10.0, 2.5, 3.2]],
                    rotations=[0.0, np.pi / 2],
                    anchor_z=-0.5,
                    matched_threshold=0.55,
                    unmatched_threshold=0.4,
                ),
                AnchorConfig(
                    class_name="pedestrian",
                    sizes=[[0.7, 0.7, 1.8]],
                    rotations=[0.0, np.pi / 2],
                    anchor_z=-0.9,
                    matched_threshold=0.5,
                    unmatched_threshold=0.35,
                ),
                AnchorConfig(
                    class_name="cyclist",
                    sizes=[[1.8, 0.8, 1.5]],
                    rotations=[0.0, np.pi / 2],
                    anchor_z=-0.9,
                    matched_threshold=0.5,
                    unmatched_threshold=0.35,
                ),
            ]
        self.anchor_configs = anchor_configs

        # Anchor generator
        self.anchor_generator = AnchorGenerator(
            anchor_configs=anchor_configs,
            feature_map_size=feature_map_size,
            point_range=point_range,
        )

        # Total number of anchors per spatial location
        num_anchors_per_location = self.anchor_generator.num_anchors_per_location

        # Prediction heads (all use 1x1 conv for efficiency)
        # Box regression: 7 params per anchor
        self.conv_box = nn.Conv2d(
            in_channels,
            num_anchors_per_location * 7,
            kernel_size=1,
            bias=True,
        )

        # Classification: num_classes per anchor
        self.conv_cls = nn.Conv2d(
            in_channels,
            num_anchors_per_location * num_classes,
            kernel_size=1,
            bias=True,
        )

        # Direction classification: 2 bins per anchor (forward/backward)
        self.conv_dir = nn.Conv2d(
            in_channels,
            num_anchors_per_location * 2,
            kernel_size=1,
            bias=True,
        )

        # Velocity regression: 2 values (vx, vy) per anchor
        self.conv_vel = nn.Conv2d(
            in_channels,
            num_anchors_per_location * 2,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: predict boxes, classes, directions, and velocities.

        Args:
            x: (B, C, H, W) feature map from backbone (C=384 by default).

        Returns:
            Dict with keys:
                'cls_preds': (B, H*W*num_anchors, num_classes) classification logits
                'box_preds': (B, H*W*num_anchors, 7) box regression deltas
                'dir_preds': (B, H*W*num_anchors, 2) direction classification logits
                'vel_preds': (B, H*W*num_anchors, 2) velocity predictions (vx, vy)
        """
        batch_size = x.shape[0]
        h, w = x.shape[2], x.shape[3]
        num_anchors = self.anchor_generator.num_anchors_per_location

        # Classification prediction
        cls_preds = self.conv_cls(x)  # (B, num_anchors*num_classes, H, W)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # (B, H, W, A*C)
        cls_preds = cls_preds.reshape(
            batch_size, h * w * num_anchors, self.num_classes
        )  # (B, H*W*A, num_classes)

        # Box regression prediction
        box_preds = self.conv_box(x)  # (B, num_anchors*7, H, W)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # (B, H, W, A*7)
        box_preds = box_preds.reshape(
            batch_size, h * w * num_anchors, 7
        )  # (B, H*W*A, 7)

        # Direction prediction
        dir_preds = self.conv_dir(x)  # (B, num_anchors*2, H, W)
        dir_preds = dir_preds.permute(0, 2, 3, 1).contiguous()  # (B, H, W, A*2)
        dir_preds = dir_preds.reshape(
            batch_size, h * w * num_anchors, 2
        )  # (B, H*W*A, 2)

        # Velocity prediction
        vel_preds = self.conv_vel(x)  # (B, num_anchors*2, H, W)
        vel_preds = vel_preds.permute(0, 2, 3, 1).contiguous()  # (B, H, W, A*2)
        vel_preds = vel_preds.reshape(
            batch_size, h * w * num_anchors, 2
        )  # (B, H*W*A, 2)

        return {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds,
            "vel_preds": vel_preds,
        }

    @torch.no_grad()
    def predict(
        self,
        cls_preds: torch.Tensor,
        box_preds: torch.Tensor,
        dir_preds: torch.Tensor,
        vel_preds: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process predictions with NMS to produce final detections.

        Args:
            cls_preds: (B, N, num_classes) classification logits.
            box_preds: (B, N, 7) box regression deltas.
            dir_preds: (B, N, 2) direction classification logits.
            vel_preds: (B, N, 2) velocity predictions.

        Returns:
            List of dicts (one per batch), each containing:
                'boxes': (K, 7) decoded 3D boxes
                'scores': (K,) confidence scores
                'labels': (K,) class labels (0-indexed)
                'velocities': (K, 2) predicted velocities (vx, vy)
        """
        batch_size = cls_preds.shape[0]
        device = cls_preds.device

        # Generate anchors
        anchors = self.anchor_generator.generate_anchors(device)  # (H, W, A, 7)
        h, w = anchors.shape[0], anchors.shape[1]
        num_anchors_per_loc = anchors.shape[2]
        anchors_flat = anchors.reshape(-1, 7)  # (H*W*A, 7)

        results = []

        for b in range(batch_size):
            # Get predictions for this batch element
            cls_scores = torch.sigmoid(cls_preds[b])  # (N, num_classes)
            box_deltas = box_preds[b]  # (N, 7)
            dir_logits = dir_preds[b]  # (N, 2)
            vel = vel_preds[b]  # (N, 2)

            # Decode boxes
            decoded_boxes = decode_boxes(anchors_flat, box_deltas)  # (N, 7)

            # Apply direction classification to correct heading
            dir_labels = dir_logits.argmax(dim=-1)  # (N,)
            # Adjust theta based on direction (period = pi)
            # If dir_label == 1, rotate by pi
            theta_correction = dir_labels.float() * np.pi
            decoded_boxes[:, 6] = decoded_boxes[:, 6] + theta_correction
            # Normalize to [-pi, pi]
            decoded_boxes[:, 6] = torch.atan2(
                torch.sin(decoded_boxes[:, 6]),
                torch.cos(decoded_boxes[:, 6]),
            )

            # Multi-class NMS
            all_boxes = []
            all_scores = []
            all_labels = []
            all_vels = []

            for cls_idx in range(self.num_classes):
                cls_score = cls_scores[:, cls_idx]  # (N,)

                # Filter by score threshold
                score_mask = cls_score > self.score_threshold
                if score_mask.sum() == 0:
                    continue

                filtered_scores = cls_score[score_mask]
                filtered_boxes = decoded_boxes[score_mask]
                filtered_vels = vel[score_mask]

                # Apply NMS
                keep_idx = nms_rotated(
                    filtered_boxes,
                    filtered_scores,
                    self.nms_threshold,
                )

                if len(keep_idx) == 0:
                    continue

                all_boxes.append(filtered_boxes[keep_idx])
                all_scores.append(filtered_scores[keep_idx])
                all_labels.append(
                    torch.full(
                        (len(keep_idx),), cls_idx, dtype=torch.long, device=device
                    )
                )
                all_vels.append(filtered_vels[keep_idx])

            # Combine all classes
            if len(all_boxes) == 0:
                results.append(
                    {
                        "boxes": torch.zeros(0, 7, device=device),
                        "scores": torch.zeros(0, device=device),
                        "labels": torch.zeros(0, dtype=torch.long, device=device),
                        "velocities": torch.zeros(0, 2, device=device),
                    }
                )
            else:
                combined_boxes = torch.cat(all_boxes, dim=0)
                combined_scores = torch.cat(all_scores, dim=0)
                combined_labels = torch.cat(all_labels, dim=0)
                combined_vels = torch.cat(all_vels, dim=0)

                # Keep top-K detections by score
                if combined_scores.shape[0] > self.max_detections:
                    topk_idx = combined_scores.argsort(descending=True)[
                        : self.max_detections
                    ]
                    combined_boxes = combined_boxes[topk_idx]
                    combined_scores = combined_scores[topk_idx]
                    combined_labels = combined_labels[topk_idx]
                    combined_vels = combined_vels[topk_idx]

                results.append(
                    {
                        "boxes": combined_boxes,
                        "scores": combined_scores,
                        "labels": combined_labels,
                        "velocities": combined_vels,
                    }
                )

        return results
