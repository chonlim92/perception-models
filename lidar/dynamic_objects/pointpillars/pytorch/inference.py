"""
PointPillars Inference Engine for 3D Object Detection from LiDAR Point Clouds.

This module provides a complete inference pipeline for the PointPillars architecture,
including point cloud preprocessing (pillarization), model forward pass, post-processing
(NMS, score thresholding), and visualization (BEV and optional 3D).

Reference: Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds", CVPR 2019.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# -------------------------------------------------------------------------------------
# Model Architecture Components
# -------------------------------------------------------------------------------------


class PillarFeatureNet(nn.Module):
    """Encodes raw pillar point features into a fixed-size descriptor per pillar.

    Each point is augmented with offsets to the pillar center and to the mean of all
    points in the pillar. A simplified PointNet (shared MLP + max pool) produces a
    D-dimensional feature for every non-empty pillar.
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, pillar_features: torch.Tensor, num_points_per_pillar: torch.Tensor) -> torch.Tensor:
        """Forward pass through the pillar feature network.

        Args:
            pillar_features: (B, P, N, C) tensor of augmented point features per pillar.
            num_points_per_pillar: (B, P) tensor indicating valid point counts.

        Returns:
            (B, P, D) tensor of encoded pillar features after max-pooling.
        """
        batch_size, max_pillars, max_points, channels = pillar_features.shape

        x = pillar_features.view(-1, max_points, channels)
        x = self.linear(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)

        mask = torch.arange(max_points, device=pillar_features.device).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, max_pillars, max_points)
        valid_mask = mask < num_points_per_pillar.unsqueeze(-1)
        valid_mask = valid_mask.view(-1, max_points).unsqueeze(-1)

        x = x.masked_fill(~valid_mask, float("-inf"))
        x, _ = x.max(dim=1)
        x = x.view(batch_size, max_pillars, -1)
        x = torch.clamp(x, min=0.0)

        return x


class PillarScatter(nn.Module):
    """Scatters encoded pillar features back into a 2D pseudo-image (BEV grid)."""

    def __init__(self, num_features: int = 64, grid_size_x: int = 432, grid_size_y: int = 496):
        super().__init__()
        self.num_features = num_features
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y

    def forward(self, pillar_features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Scatter pillar features onto BEV canvas.

        Args:
            pillar_features: (B, P, D) encoded pillar features.
            coords: (B, P, 2) grid coordinates (x_idx, y_idx) for each pillar.

        Returns:
            (B, D, H, W) pseudo-image tensor.
        """
        batch_size = pillar_features.shape[0]
        canvas = torch.zeros(
            batch_size,
            self.num_features,
            self.grid_size_y,
            self.grid_size_x,
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )

        for b in range(batch_size):
            x_indices = coords[b, :, 0].long()
            y_indices = coords[b, :, 1].long()

            valid = (x_indices >= 0) & (x_indices < self.grid_size_x) & (y_indices >= 0) & (y_indices < self.grid_size_y)
            x_valid = x_indices[valid]
            y_valid = y_indices[valid]
            features_valid = pillar_features[b, valid, :]

            canvas[b, :, y_valid, x_valid] = features_valid.t()

        return canvas


class BackboneNetwork(nn.Module):
    """Multi-scale 2D convolutional backbone operating on the BEV pseudo-image.

    Produces multi-scale feature maps that are upsampled and concatenated for
    dense detection head input.
    """

    def __init__(self, in_channels: int = 64, layer_nums: Tuple[int, ...] = (3, 5, 5)):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        filter_nums = [64, 128, 256]
        upsample_strides = [1, 2, 4]

        current_channels = in_channels
        for i, (num_layers, num_filters) in enumerate(zip(layer_nums, filter_nums)):
            block = [
                nn.Conv2d(current_channels, num_filters, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            ]
            for _ in range(num_layers - 1):
                block.extend(
                    [
                        nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters),
                        nn.ReLU(inplace=True),
                    ]
                )
            self.blocks.append(nn.Sequential(*block))

            deblock = nn.Sequential(
                nn.ConvTranspose2d(num_filters, 128, upsample_strides[i], stride=upsample_strides[i], bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.deblocks.append(deblock)
            current_channels = num_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone.

        Args:
            x: (B, C, H, W) pseudo-image tensor.

        Returns:
            (B, C_out, H/2, W/2) concatenated multi-scale feature map.
        """
        ups = []
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))

        target_h = ups[0].shape[2]
        target_w = ups[0].shape[3]
        aligned_ups = []
        for u in ups:
            if u.shape[2] != target_h or u.shape[3] != target_w:
                u = F.interpolate(u, size=(target_h, target_w), mode="bilinear", align_corners=False)
            aligned_ups.append(u)

        return torch.cat(aligned_ups, dim=1)


class DetectionHead(nn.Module):
    """Single-shot detection head producing per-anchor box regression and classification."""

    def __init__(
        self,
        in_channels: int = 384,
        num_classes: int = 3,
        num_anchors_per_location: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location

        self.cls_head = nn.Conv2d(in_channels, num_anchors_per_location * num_classes, 1)
        self.box_head = nn.Conv2d(in_channels, num_anchors_per_location * 7, 1)
        self.dir_head = nn.Conv2d(in_channels, num_anchors_per_location * 2, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through detection head.

        Args:
            x: (B, C, H, W) feature map from backbone.

        Returns:
            Dictionary with 'cls_preds', 'box_preds', 'dir_preds' tensors.
        """
        cls_preds = self.cls_head(x)
        box_preds = self.box_head(x)
        dir_preds = self.dir_head(x)

        batch_size = x.shape[0]
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
        dir_preds = dir_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

        return {"cls_preds": cls_preds, "box_preds": box_preds, "dir_preds": dir_preds}


class PointPillarsModel(nn.Module):
    """Full PointPillars network combining pillar feature extraction, scatter, backbone, and detection head."""

    def __init__(self, config: Dict):
        super().__init__()
        num_classes = config.get("num_classes", 3)
        num_point_features = config.get("num_point_features", 9)
        pillar_feat_channels = config.get("pillar_feat_channels", 64)
        grid_size_x = config.get("grid_size_x", 432)
        grid_size_y = config.get("grid_size_y", 496)
        num_anchors = config.get("num_anchors_per_location", 2)

        self.pillar_feature_net = PillarFeatureNet(in_channels=num_point_features, out_channels=pillar_feat_channels)
        self.scatter = PillarScatter(num_features=pillar_feat_channels, grid_size_x=grid_size_x, grid_size_y=grid_size_y)
        self.backbone = BackboneNetwork(in_channels=pillar_feat_channels)
        self.detection_head = DetectionHead(
            in_channels=384,
            num_classes=num_classes,
            num_anchors_per_location=num_anchors,
        )

    def forward(
        self,
        pillar_features: torch.Tensor,
        num_points_per_pillar: torch.Tensor,
        coords: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            pillar_features: (B, P, N, C) augmented point features.
            num_points_per_pillar: (B, P) valid point counts per pillar.
            coords: (B, P, 2) pillar grid coordinates.

        Returns:
            Detection head outputs dict with cls_preds, box_preds, dir_preds.
        """
        encoded = self.pillar_feature_net(pillar_features, num_points_per_pillar)
        pseudo_image = self.scatter(encoded, coords)
        features = self.backbone(pseudo_image)
        predictions = self.detection_head(features)
        return predictions


# -------------------------------------------------------------------------------------
# Anchor Generation
# -------------------------------------------------------------------------------------


def generate_anchors(
    point_cloud_range: List[float],
    voxel_size: List[float],
    anchor_sizes: List[List[float]],
    anchor_rotations: List[float],
    anchor_heights: List[float],
) -> np.ndarray:
    """Generate dense anchor boxes over the BEV grid.

    Args:
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy, vz] in meters.
        anchor_sizes: List of [length, width, height] per class.
        anchor_rotations: List of rotation angles in radians.
        anchor_heights: List of anchor center z-heights.

    Returns:
        (H*W*num_anchors, 7) array of anchors [x, y, z, l, w, h, theta].
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

    feature_map_stride = 2
    x_stride = voxel_size[0] * feature_map_stride
    y_stride = voxel_size[1] * feature_map_stride

    x_centers = np.arange(x_min + x_stride / 2, x_max, x_stride, dtype=np.float32)
    y_centers = np.arange(y_min + y_stride / 2, y_max, y_stride, dtype=np.float32)

    xx, yy = np.meshgrid(x_centers, y_centers)
    xx = xx.flatten()
    yy = yy.flatten()
    num_locations = len(xx)

    all_anchors = []
    for size in anchor_sizes:
        for rot in anchor_rotations:
            for z_height in anchor_heights:
                anchors = np.zeros((num_locations, 7), dtype=np.float32)
                anchors[:, 0] = xx
                anchors[:, 1] = yy
                anchors[:, 2] = z_height
                anchors[:, 3] = size[0]
                anchors[:, 4] = size[1]
                anchors[:, 5] = size[2]
                anchors[:, 6] = rot
                all_anchors.append(anchors)

    return np.concatenate(all_anchors, axis=0)


# -------------------------------------------------------------------------------------
# Post-Processing: Decoding and NMS
# -------------------------------------------------------------------------------------


def decode_boxes(box_preds: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Decode predicted box residuals relative to anchors.

    Uses the standard residual encoding:
        dx = (x_pred - x_a) / diag_a, etc.
        dl = log(l_pred / l_a), etc.
        dtheta = theta_pred - theta_a

    Args:
        box_preds: (N, 7) predicted residuals [dx, dy, dz, dl, dw, dh, dtheta].
        anchors: (N, 7) anchor boxes [x, y, z, l, w, h, theta].

    Returns:
        (N, 7) decoded boxes in world coordinates.
    """
    diag = np.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    decoded = np.zeros_like(box_preds)
    decoded[:, 0] = box_preds[:, 0] * diag + anchors[:, 0]
    decoded[:, 1] = box_preds[:, 1] * diag + anchors[:, 1]
    decoded[:, 2] = box_preds[:, 2] * anchors[:, 5] + anchors[:, 2]
    decoded[:, 3] = np.exp(box_preds[:, 3]) * anchors[:, 3]
    decoded[:, 4] = np.exp(box_preds[:, 4]) * anchors[:, 4]
    decoded[:, 5] = np.exp(box_preds[:, 5]) * anchors[:, 5]
    decoded[:, 6] = box_preds[:, 6] + anchors[:, 6]

    return decoded


def rotated_iou_2d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute approximate IoU between two rotated 2D boxes using axis-aligned approximation.

    For speed, this uses the bounding-rectangle approximation. For higher accuracy in production,
    replace with Shapely or a dedicated rotated IoU CUDA kernel.

    Args:
        box_a: [x, y, l, w, theta] center-format box.
        box_b: [x, y, l, w, theta] center-format box.

    Returns:
        Approximate IoU value.
    """
    cos_a, sin_a = np.cos(box_a[4]), np.sin(box_a[4])
    cos_b, sin_b = np.cos(box_b[4]), np.sin(box_b[4])

    half_l_a, half_w_a = box_a[2] / 2, box_a[3] / 2
    half_l_b, half_w_b = box_b[2] / 2, box_b[3] / 2

    extent_x_a = abs(cos_a) * half_l_a + abs(sin_a) * half_w_a
    extent_y_a = abs(sin_a) * half_l_a + abs(cos_a) * half_w_a
    extent_x_b = abs(cos_b) * half_l_b + abs(sin_b) * half_w_b
    extent_y_b = abs(sin_b) * half_l_b + abs(cos_b) * half_w_b

    min_x_a, max_x_a = box_a[0] - extent_x_a, box_a[0] + extent_x_a
    min_y_a, max_y_a = box_a[1] - extent_y_a, box_a[1] + extent_y_a
    min_x_b, max_x_b = box_b[0] - extent_x_b, box_b[0] + extent_x_b
    min_y_b, max_y_b = box_b[1] - extent_y_b, box_b[1] + extent_y_b

    inter_x = max(0.0, min(max_x_a, max_x_b) - max(min_x_a, min_x_b))
    inter_y = max(0.0, min(max_y_a, max_y_b) - max(min_y_a, min_y_b))
    inter_area = inter_x * inter_y

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area

    if union_area < 1e-6:
        return 0.0
    return inter_area / union_area


def nms_rotated(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.1) -> np.ndarray:
    """Apply greedy NMS on rotated 3D bounding boxes projected to BEV.

    Args:
        boxes: (N, 7) decoded boxes [x, y, z, l, w, h, theta].
        scores: (N,) confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Array of indices to keep.
    """
    if len(scores) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    keep = []

    suppressed = np.zeros(len(scores), dtype=bool)

    for i in range(len(order)):
        idx = order[i]
        if suppressed[idx]:
            continue
        keep.append(idx)

        box_i = np.array([boxes[idx, 0], boxes[idx, 1], boxes[idx, 3], boxes[idx, 4], boxes[idx, 6]])

        for j in range(i + 1, len(order)):
            jdx = order[j]
            if suppressed[jdx]:
                continue
            box_j = np.array([boxes[jdx, 0], boxes[jdx, 1], boxes[jdx, 3], boxes[jdx, 4], boxes[jdx, 6]])
            iou = rotated_iou_2d(box_i, box_j)
            if iou > iou_threshold:
                suppressed[jdx] = True

    return np.array(keep, dtype=np.int64)


# -------------------------------------------------------------------------------------
# Pillarization (Point Cloud to Pillar Representation)
# -------------------------------------------------------------------------------------


def create_pillars(
    points: np.ndarray,
    point_cloud_range: List[float],
    voxel_size: List[float],
    max_pillars: int = 12000,
    max_points_per_pillar: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw point cloud into pillar representation.

    Each point is augmented with:
        - offset from pillar center (xc, yc, zc)
        - offset from mean of points in pillar (xp, yp)

    Args:
        points: (N, 4) array with columns [x, y, z, intensity].
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy, vz] in meters.
        max_pillars: Maximum number of pillars to generate.
        max_points_per_pillar: Maximum points sampled per pillar.

    Returns:
        pillar_features: (max_pillars, max_points_per_pillar, 9) augmented features.
        num_points_per_pillar: (max_pillars,) valid point count per pillar.
        coords: (max_pillars, 2) grid coordinates (x_idx, y_idx) for each pillar.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = voxel_size

    grid_x = int(np.round((x_max - x_min) / vx))
    grid_y = int(np.round((y_max - y_min) / vy))

    x_indices = np.floor((points[:, 0] - x_min) / vx).astype(np.int32)
    y_indices = np.floor((points[:, 1] - y_min) / vy).astype(np.int32)

    valid = (
        (x_indices >= 0)
        & (x_indices < grid_x)
        & (y_indices >= 0)
        & (y_indices < grid_y)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    points = points[valid]
    x_indices = x_indices[valid]
    y_indices = y_indices[valid]

    pillar_keys = y_indices * grid_x + x_indices

    unique_keys, inverse_indices = np.unique(pillar_keys, return_inverse=True)
    num_unique_pillars = min(len(unique_keys), max_pillars)

    if len(unique_keys) > max_pillars:
        selected_pillar_indices = np.random.choice(len(unique_keys), max_pillars, replace=False)
        selected_pillar_indices.sort()
    else:
        selected_pillar_indices = np.arange(len(unique_keys))

    pillar_features = np.zeros((max_pillars, max_points_per_pillar, 9), dtype=np.float32)
    num_points_arr = np.zeros(max_pillars, dtype=np.int32)
    coords = np.full((max_pillars, 2), -1, dtype=np.int32)

    for out_idx, pillar_idx in enumerate(selected_pillar_indices):
        key = unique_keys[pillar_idx]
        point_mask = inverse_indices == pillar_idx
        pillar_points = points[point_mask]

        num_pts = min(len(pillar_points), max_points_per_pillar)
        if len(pillar_points) > max_points_per_pillar:
            choice = np.random.choice(len(pillar_points), max_points_per_pillar, replace=False)
            pillar_points = pillar_points[choice]
            num_pts = max_points_per_pillar

        px = int(key % grid_x)
        py = int(key // grid_x)

        pillar_center_x = x_min + (px + 0.5) * vx
        pillar_center_y = y_min + (py + 0.5) * vy

        mean_x = pillar_points[:num_pts, 0].mean()
        mean_y = pillar_points[:num_pts, 1].mean()
        mean_z = pillar_points[:num_pts, 2].mean()

        features = np.zeros((num_pts, 9), dtype=np.float32)
        features[:, 0] = pillar_points[:num_pts, 0]
        features[:, 1] = pillar_points[:num_pts, 1]
        features[:, 2] = pillar_points[:num_pts, 2]
        features[:, 3] = pillar_points[:num_pts, 3]
        features[:, 4] = pillar_points[:num_pts, 0] - pillar_center_x
        features[:, 5] = pillar_points[:num_pts, 1] - pillar_center_y
        features[:, 6] = pillar_points[:num_pts, 0] - mean_x
        features[:, 7] = pillar_points[:num_pts, 1] - mean_y
        features[:, 8] = pillar_points[:num_pts, 2] - mean_z

        pillar_features[out_idx, :num_pts, :] = features
        num_points_arr[out_idx] = num_pts

        feature_stride = 2
        coords[out_idx, 0] = px // feature_stride
        coords[out_idx, 1] = py // feature_stride

    return pillar_features, num_points_arr, coords


# -------------------------------------------------------------------------------------
# Inference Class
# -------------------------------------------------------------------------------------


class PointPillarsInference:
    """End-to-end PointPillars inference engine.

    Loads a trained checkpoint, processes point cloud frames through pillarization,
    neural network forward pass, and post-processing (decoding + NMS) to produce
    3D bounding box detections.

    Attributes:
        config: Model and preprocessing configuration dictionary.
        device: Torch device (cuda or cpu).
        model: Loaded PointPillars model in eval mode.
        anchors: Pre-generated dense anchor boxes.
        score_threshold: Minimum confidence score for detections.
        nms_iou_threshold: IoU threshold for NMS suppression.
    """

    CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
    CLASS_COLORS = {
        "Car": (1.0, 0.0, 0.0),
        "Pedestrian": (0.0, 0.0, 1.0),
        "Cyclist": (0.0, 1.0, 0.0),
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        score_threshold: float = 0.3,
        nms_iou_threshold: float = 0.1,
    ):
        """Initialize PointPillars inference engine.

        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
            checkpoint_path: Path to model checkpoint (.pth). If None, uses random weights.
            device: Device string ('cuda' or 'cpu').
            score_threshold: Minimum score to keep a detection.
            nms_iou_threshold: IoU threshold for NMS.
        """
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self.point_cloud_range = self.config["point_cloud_range"]
        self.voxel_size = self.config["voxel_size"]
        self.max_pillars = self.config.get("max_pillars", 12000)
        self.max_points_per_pillar = self.config.get("max_points_per_pillar", 100)

        model_config = self.config.get("model", {})
        self.model = PointPillarsModel(model_config).to(self.device)
        self.model.eval()

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.anchors = generate_anchors(
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            anchor_sizes=self.config.get("anchor_sizes", [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]),
            anchor_rotations=self.config.get("anchor_rotations", [0, np.pi / 2]),
            anchor_heights=self.config.get("anchor_heights", [-1.78]),
        )

        print(f"[PointPillarsInference] Model loaded on {self.device}")
        print(f"[PointPillarsInference] Point cloud range: {self.point_cloud_range}")
        print(f"[PointPillarsInference] Number of anchors: {len(self.anchors)}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[PointPillarsInference] Total parameters: {total_params:,}")

    @staticmethod
    def _default_config() -> Dict:
        """Return default configuration for KITTI-like setup."""
        return {
            "point_cloud_range": [0.0, -39.68, -3.0, 69.12, 39.68, 1.0],
            "voxel_size": [0.16, 0.16, 4.0],
            "max_pillars": 12000,
            "max_points_per_pillar": 100,
            "anchor_sizes": [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]],
            "anchor_rotations": [0, 1.5707963267948966],
            "anchor_heights": [-1.78],
            "model": {
                "num_classes": 3,
                "num_point_features": 9,
                "pillar_feat_channels": 64,
                "grid_size_x": 432,
                "grid_size_y": 496,
                "num_anchors_per_location": 2,
            },
        }

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint file.

        Args:
            checkpoint_path: Path to .pth checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            cleaned_state_dict[new_key] = value

        self.model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"[PointPillarsInference] Checkpoint loaded from: {checkpoint_path}")

    def _load_point_cloud(self, input_data: Union[np.ndarray, str]) -> np.ndarray:
        """Load point cloud from numpy array or binary file.

        Args:
            input_data: Either an (N, 4) numpy array or path to a .bin file.

        Returns:
            (N, 4) numpy array of points [x, y, z, intensity].
        """
        if isinstance(input_data, np.ndarray):
            if input_data.ndim != 2 or input_data.shape[1] < 4:
                raise ValueError(f"Expected (N, 4+) array, got shape {input_data.shape}")
            return input_data[:, :4].astype(np.float32)

        file_path = Path(input_data)
        if not file_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")

        points = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)
        return points

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """Filter points to within the configured point cloud range.

        Args:
            points: (N, 4) raw point cloud.

        Returns:
            (M, 4) filtered points within range.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        )
        return points[mask]

    @torch.no_grad()
    def predict(self, point_cloud: Union[np.ndarray, str]) -> Dict[str, np.ndarray]:
        """Run inference on a single point cloud frame.

        Args:
            point_cloud: (N, 4) numpy array or path to .bin file.

        Returns:
            Dictionary with:
                'boxes': (M, 7) detected 3D bounding boxes [x, y, z, l, w, h, theta].
                'scores': (M,) confidence scores.
                'labels': (M,) integer class labels.
                'class_names': (M,) string class names.
                'inference_time_ms': float inference time in milliseconds.
        """
        t_start = time.perf_counter()

        points = self._load_point_cloud(point_cloud)
        points = self._filter_points(points)

        t_preprocess_start = time.perf_counter()
        pillar_features, num_points_per_pillar, coords = create_pillars(
            points,
            self.point_cloud_range,
            self.voxel_size,
            self.max_pillars,
            self.max_points_per_pillar,
        )

        pillar_features_t = torch.from_numpy(pillar_features).unsqueeze(0).to(self.device)
        num_points_t = torch.from_numpy(num_points_per_pillar).unsqueeze(0).to(self.device)
        coords_t = torch.from_numpy(coords).unsqueeze(0).to(self.device)
        t_preprocess_end = time.perf_counter()

        t_forward_start = time.perf_counter()
        outputs = self.model(pillar_features_t, num_points_t, coords_t)
        t_forward_end = time.perf_counter()

        t_postprocess_start = time.perf_counter()
        cls_preds = outputs["cls_preds"][0].cpu().numpy()
        box_preds = outputs["box_preds"][0].cpu().numpy()
        dir_preds = outputs["dir_preds"][0].cpu().numpy()

        cls_scores = 1.0 / (1.0 + np.exp(-cls_preds))

        num_anchors = len(self.anchors)
        num_preds = cls_scores.shape[0]

        if num_preds > num_anchors:
            cls_scores = cls_scores[:num_anchors]
            box_preds = box_preds[:num_anchors]
            dir_preds = dir_preds[:num_anchors]
        elif num_preds < num_anchors:
            anchors_used = self.anchors[:num_preds]
        else:
            anchors_used = self.anchors

        if num_preds > num_anchors:
            anchors_used = self.anchors
        elif num_preds < num_anchors:
            anchors_used = self.anchors[:num_preds]
        else:
            anchors_used = self.anchors

        max_scores = cls_scores.max(axis=1)
        max_labels = cls_scores.argmax(axis=1)

        score_mask = max_scores >= self.score_threshold
        filtered_scores = max_scores[score_mask]
        filtered_labels = max_labels[score_mask]
        filtered_box_preds = box_preds[score_mask]
        filtered_dir_preds = dir_preds[score_mask]
        filtered_anchors = anchors_used[score_mask]

        if len(filtered_scores) == 0:
            t_end = time.perf_counter()
            return {
                "boxes": np.zeros((0, 7), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int32),
                "class_names": [],
                "inference_time_ms": (t_end - t_start) * 1000.0,
            }

        decoded_boxes = decode_boxes(filtered_box_preds, filtered_anchors)

        dir_cls = filtered_dir_preds.argmax(axis=1)
        heading_adjustment = np.where(dir_cls == 0, 0.0, np.pi)
        decoded_boxes[:, 6] = decoded_boxes[:, 6] + heading_adjustment

        decoded_boxes[:, 6] = np.arctan2(np.sin(decoded_boxes[:, 6]), np.cos(decoded_boxes[:, 6]))

        all_boxes = []
        all_scores = []
        all_labels = []

        num_classes = self.config.get("model", {}).get("num_classes", 3)
        for cls_id in range(num_classes):
            cls_mask = filtered_labels == cls_id
            if not np.any(cls_mask):
                continue

            cls_boxes = decoded_boxes[cls_mask]
            cls_scores_filtered = filtered_scores[cls_mask]

            keep_indices = nms_rotated(cls_boxes, cls_scores_filtered, self.nms_iou_threshold)

            if len(keep_indices) > 0:
                all_boxes.append(cls_boxes[keep_indices])
                all_scores.append(cls_scores_filtered[keep_indices])
                all_labels.append(np.full(len(keep_indices), cls_id, dtype=np.int32))

        t_postprocess_end = time.perf_counter()
        t_end = time.perf_counter()

        if len(all_boxes) > 0:
            final_boxes = np.concatenate(all_boxes, axis=0)
            final_scores = np.concatenate(all_scores, axis=0)
            final_labels = np.concatenate(all_labels, axis=0)
        else:
            final_boxes = np.zeros((0, 7), dtype=np.float32)
            final_scores = np.zeros((0,), dtype=np.float32)
            final_labels = np.zeros((0,), dtype=np.int32)

        class_names = [self.CLASS_NAMES[lbl] if lbl < len(self.CLASS_NAMES) else f"Class_{lbl}" for lbl in final_labels]

        total_time_ms = (t_end - t_start) * 1000.0
        preprocess_ms = (t_preprocess_end - t_preprocess_start) * 1000.0
        forward_ms = (t_forward_end - t_forward_start) * 1000.0
        postprocess_ms = (t_postprocess_end - t_postprocess_start) * 1000.0

        print(
            f"  [Timing] Total: {total_time_ms:.1f}ms | "
            f"Preprocess: {preprocess_ms:.1f}ms | "
            f"Forward: {forward_ms:.1f}ms | "
            f"Postprocess: {postprocess_ms:.1f}ms | "
            f"FPS: {1000.0 / max(total_time_ms, 0.001):.1f}"
        )

        return {
            "boxes": final_boxes,
            "scores": final_scores,
            "labels": final_labels,
            "class_names": class_names,
            "inference_time_ms": total_time_ms,
        }

    def predict_sequence(self, file_list: List[str]) -> List[Dict[str, np.ndarray]]:
        """Run inference on a sequence of point cloud files.

        Args:
            file_list: List of paths to .bin point cloud files.

        Returns:
            List of prediction dictionaries (same format as predict()).
        """
        results = []
        total_time = 0.0
        num_frames = len(file_list)

        print(f"[PointPillarsInference] Processing sequence of {num_frames} frames...")

        for idx, file_path in enumerate(file_list):
            print(f"  Frame {idx + 1}/{num_frames}: {Path(file_path).name}")
            result = self.predict(file_path)
            results.append(result)
            total_time += result["inference_time_ms"]

        avg_time_ms = total_time / max(num_frames, 1)
        avg_fps = 1000.0 / max(avg_time_ms, 0.001)

        print(f"\n[Sequence Summary]")
        print(f"  Total frames: {num_frames}")
        print(f"  Total time: {total_time:.1f} ms")
        print(f"  Average per frame: {avg_time_ms:.1f} ms")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Target (60 Hz) {'ACHIEVED' if avg_fps >= 60 else 'NOT MET'}")

        return results

    def export_onnx(self, output_path: str = "pointpillars.onnx") -> None:
        """Export model to ONNX format for TensorRT optimization.

        Args:
            output_path: Destination path for the .onnx file.

        Note:
            After export, optimize with TensorRT:
                trtexec --onnx=pointpillars.onnx --saveEngine=pointpillars.trt --fp16
        """
        max_pillars = self.max_pillars
        max_points = self.max_points_per_pillar

        dummy_pillars = torch.randn(1, max_pillars, max_points, 9).to(self.device)
        dummy_num_points = torch.randint(0, max_points, (1, max_pillars)).to(self.device)
        dummy_coords = torch.randint(0, 200, (1, max_pillars, 2)).to(self.device)

        torch.onnx.export(
            self.model,
            (dummy_pillars, dummy_num_points, dummy_coords),
            output_path,
            opset_version=11,
            input_names=["pillar_features", "num_points_per_pillar", "coords"],
            output_names=["cls_preds", "box_preds", "dir_preds"],
            dynamic_axes={
                "pillar_features": {0: "batch"},
                "num_points_per_pillar": {0: "batch"},
                "coords": {0: "batch"},
            },
        )
        print(f"[PointPillarsInference] ONNX model exported to: {output_path}")
        print(f"  TensorRT optimization hint:")
        print(f"    trtexec --onnx={output_path} --saveEngine=pointpillars.trt --fp16")


# -------------------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------------------


def get_box_corners_bev(box: np.ndarray) -> np.ndarray:
    """Compute the 4 BEV corners of a rotated 3D box.

    Args:
        box: (7,) array [x, y, z, l, w, h, theta].

    Returns:
        (4, 2) array of corner coordinates in BEV (x, y).
    """
    x, y, z, length, width, height, theta = box

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    half_l = length / 2.0
    half_w = width / 2.0

    corners_local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ]
    )

    rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners_world = corners_local @ rotation_matrix.T + np.array([x, y])

    return corners_world


def visualize_bev(
    point_cloud: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    point_cloud_range: Optional[List[float]] = None,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 12),
    point_size: float = 0.3,
    dpi: int = 150,
) -> None:
    """Visualize point cloud and detections in Bird's Eye View.

    Draws the point cloud as a scatter plot and overlays detected 3D bounding boxes
    projected onto the BEV plane. Each class is color-coded. Heading direction is
    shown with an arrow from box center along the heading angle.

    Args:
        point_cloud: (N, 4) point cloud array [x, y, z, intensity].
        predictions: Detection dict from predict() with 'boxes', 'scores', 'labels'.
        save_path: If provided, save figure to this path. Otherwise, display interactively.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] for axis limits.
        class_names: List of class name strings.
        figsize: Matplotlib figure size.
        point_size: Scatter plot point size.
        dpi: Figure resolution.
    """
    if class_names is None:
        class_names = PointPillarsInference.CLASS_NAMES

    class_colors = {
        0: (1.0, 0.0, 0.0),
        1: (0.0, 0.0, 1.0),
        2: (0.0, 0.8, 0.0),
    }

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.set_facecolor("black")
    fig.set_facecolor("black")

    intensity = point_cloud[:, 3] if point_cloud.shape[1] > 3 else np.ones(len(point_cloud))
    intensity_normalized = np.clip(intensity / max(intensity.max(), 1e-5), 0.1, 1.0)

    ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        s=point_size,
        c=intensity_normalized,
        cmap="gray",
        alpha=0.6,
        edgecolors="none",
    )

    boxes = predictions.get("boxes", np.zeros((0, 7)))
    scores = predictions.get("scores", np.zeros((0,)))
    labels = predictions.get("labels", np.zeros((0,), dtype=np.int32))

    for i in range(len(boxes)):
        box = boxes[i]
        label = int(labels[i])
        score = scores[i]
        color = class_colors.get(label, (1.0, 1.0, 0.0))

        corners = get_box_corners_bev(box)
        polygon = plt.Polygon(corners, fill=False, edgecolor=color, linewidth=1.5, linestyle="-")
        ax.add_patch(polygon)

        front_center = (corners[0] + corners[1]) / 2.0
        ax.plot(
            [box[0], front_center[0]],
            [box[1], front_center[1]],
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
        )

        arrow_length = box[3] * 0.4
        arrow_dx = arrow_length * np.cos(box[6])
        arrow_dy = arrow_length * np.sin(box[6])
        ax.annotate(
            "",
            xy=(box[0] + arrow_dx, box[1] + arrow_dy),
            xytext=(box[0], box[1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

        label_text = f"{class_names[label] if label < len(class_names) else f'C{label}'}: {score:.2f}"
        ax.text(
            box[0],
            box[1] + box[4] / 2 + 0.5,
            label_text,
            color=color,
            fontsize=7,
            ha="center",
            va="bottom",
            weight="bold",
        )

    legend_patches = []
    for cls_id, cls_name in enumerate(class_names):
        color = class_colors.get(cls_id, (1.0, 1.0, 0.0))
        patch = mpatches.Patch(color=color, label=cls_name)
        legend_patches.append(patch)
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10, framealpha=0.7)

    if point_cloud_range is not None:
        ax.set_xlim(point_cloud_range[0], point_cloud_range[3])
        ax.set_ylim(point_cloud_range[1], point_cloud_range[4])
    else:
        margin = 5.0
        ax.set_xlim(point_cloud[:, 0].min() - margin, point_cloud[:, 0].max() + margin)
        ax.set_ylim(point_cloud[:, 1].min() - margin, point_cloud[:, 1].max() + margin)

    ax.set_xlabel("X (m)", color="white", fontsize=11)
    ax.set_ylabel("Y (m)", color="white", fontsize=11)
    ax.set_title("PointPillars BEV Detection", color="white", fontsize=14, weight="bold")
    ax.tick_params(colors="white")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, color="gray")

    num_detections = len(boxes)
    info_text = f"Detections: {num_detections}"
    if "inference_time_ms" in predictions:
        info_text += f" | {predictions['inference_time_ms']:.1f} ms"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  [Visualization] BEV saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_3d(
    point_cloud: np.ndarray,
    predictions: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    window_name: str = "PointPillars 3D Detection",
) -> None:
    """Visualize point cloud and 3D bounding boxes using Open3D.

    Falls back gracefully if Open3D is not installed.

    Args:
        point_cloud: (N, 4) point cloud [x, y, z, intensity].
        predictions: Detection dict with 'boxes', 'scores', 'labels'.
        class_names: Optional list of class name strings.
        window_name: Title for the visualization window.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[visualize_3d] Open3D not installed. Install with: pip install open3d")
        print("  Skipping 3D visualization.")
        return

    if class_names is None:
        class_names = PointPillarsInference.CLASS_NAMES

    class_colors_rgb = {
        0: [1.0, 0.0, 0.0],
        1: [0.0, 0.0, 1.0],
        2: [0.0, 0.8, 0.0],
    }

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    if point_cloud.shape[1] >= 4:
        intensity = point_cloud[:, 3]
        intensity_norm = np.clip(intensity / max(intensity.max(), 1e-5), 0.0, 1.0)
        colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    boxes = predictions.get("boxes", np.zeros((0, 7)))
    labels = predictions.get("labels", np.zeros((0,), dtype=np.int32))

    for i in range(len(boxes)):
        box = boxes[i]
        label = int(labels[i])
        color = class_colors_rgb.get(label, [1.0, 1.0, 0.0])

        x, y, z, length, width, height, theta = box
        center = np.array([x, y, z])

        bbox = o3d.geometry.OrientedBoundingBox()
        bbox.center = center
        bbox.extent = np.array([length, width, height])

        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        bbox.R = rotation_matrix
        bbox.color = color
        geometries.append(bbox)

        arrow_length = length * 0.7
        arrow_start = center.copy()
        arrow_end = center + rotation_matrix @ np.array([arrow_length, 0, 0])

        line_points = [arrow_start.tolist(), arrow_end.tolist()]
        line_indices = [[0, 1]]
        heading_line = o3d.geometry.LineSet()
        heading_line.points = o3d.utility.Vector3dVector(line_points)
        heading_line.lines = o3d.utility.Vector2iVector(line_indices)
        heading_line.colors = o3d.utility.Vector3dVector([color])
        geometries.append(heading_line)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    geometries.append(coord_frame)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    for geom in geometries:
        vis.add_geometry(geom)

    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, 1])
    view_ctl.set_lookat([35, 0, -1])
    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.15)

    vis.run()
    vis.destroy_window()


# -------------------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------------------


def main() -> None:
    """Main entry point for PointPillars inference from command line."""
    parser = argparse.ArgumentParser(
        description="PointPillars 3D Object Detection Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file. Uses default KITTI config if not specified.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single .bin file or directory containing .bin files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output_detections",
        help="Output directory for visualizations and results.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Minimum detection confidence score.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate BEV visualization images.",
    )
    parser.add_argument(
        "--visualize_3d",
        action="store_true",
        help="Show interactive 3D visualization (requires Open3D).",
    )
    parser.add_argument(
        "--export_onnx",
        type=str,
        default=None,
        help="If specified, export model to ONNX at this path and exit.",
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=0.1,
        help="IoU threshold for NMS.",
    )

    args = parser.parse_args()

    engine = PointPillarsInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
    )

    if args.export_onnx is not None:
        engine.export_onnx(args.export_onnx)
        return

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        file_list = [str(input_path)]
    elif input_path.is_dir():
        file_list = sorted([str(f) for f in input_path.glob("*.bin")])
        if not file_list:
            print(f"[Error] No .bin files found in: {input_path}")
            return
        print(f"[PointPillarsInference] Found {len(file_list)} point cloud files.")
    else:
        print(f"[Error] Input path does not exist: {input_path}")
        return

    if len(file_list) == 1:
        print(f"\n[Single Frame Inference]")
        result = engine.predict(file_list[0])
        print(f"  Detected {len(result['boxes'])} objects:")
        for i in range(len(result["boxes"])):
            box = result["boxes"][i]
            print(
                f"    {result['class_names'][i]}: score={result['scores'][i]:.3f} "
                f"pos=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}) "
                f"size=({box[3]:.1f}, {box[4]:.1f}, {box[5]:.1f}) "
                f"heading={np.degrees(box[6]):.1f} deg"
            )

        if args.visualize:
            points = np.fromfile(file_list[0], dtype=np.float32).reshape(-1, 4)
            save_name = output_path / f"{Path(file_list[0]).stem}_bev.png"
            visualize_bev(
                points,
                result,
                save_path=str(save_name),
                point_cloud_range=engine.point_cloud_range,
            )

        if args.visualize_3d:
            points = np.fromfile(file_list[0], dtype=np.float32).reshape(-1, 4)
            visualize_3d(points, result)
    else:
        print(f"\n[Sequence Inference]")
        results = engine.predict_sequence(file_list)

        if args.visualize:
            print(f"\n[Generating BEV Visualizations]")
            for idx, (file_path, result) in enumerate(zip(file_list, results)):
                points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
                save_name = output_path / f"{Path(file_path).stem}_bev.png"
                visualize_bev(
                    points,
                    result,
                    save_path=str(save_name),
                    point_cloud_range=engine.point_cloud_range,
                )

        total_detections = sum(len(r["boxes"]) for r in results)
        print(f"\n[Final Summary]")
        print(f"  Processed {len(results)} frames")
        print(f"  Total detections: {total_detections}")
        print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
