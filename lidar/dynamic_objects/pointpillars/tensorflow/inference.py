"""PointPillars TF2 Inference Script.

Production-quality inference pipeline for 3D object detection from LiDAR
point clouds. Supports single and batch processing, SavedModel or checkpoint
loading, and provides real-time performance measurement.

Output format: 3D bounding boxes (x, y, z, w, l, h, yaw, class_id, score)

Reference: Lang et al., "PointPillars: Fast Encoders for Object Detection
from Point Clouds", CVPR 2019.

Usage:
    # Single point cloud
    python inference.py --checkpoint ./output/checkpoints/ckpt-32 \
                        --input /data/kitti/velodyne/000001.bin

    # Batch processing
    python inference.py --saved_model ./output/saved_model \
                        --input_dir /data/kitti/velodyne/ \
                        --output_dir ./detections/ \
                        --batch_size 4

    # With custom thresholds
    python inference.py --checkpoint ./output/checkpoints/ckpt-32 \
                        --input /data/kitti/velodyne/000001.bin \
                        --score_threshold 0.5 \
                        --nms_iou_threshold 0.3
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class InferenceConfig:
    """Inference configuration parameters."""

    # Model
    checkpoint_path: Optional[str] = None
    saved_model_path: Optional[str] = None

    # Point cloud range and voxelization
    point_cloud_range: List[float] = field(
        default_factory=lambda: [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
    )
    voxel_size: List[float] = field(default_factory=lambda: [0.16, 0.16, 4.0])
    max_points_per_voxel: int = 32
    max_voxels: int = 40000

    # Model architecture
    num_classes: int = 3
    class_names: List[str] = field(
        default_factory=lambda: ["Car", "Pedestrian", "Cyclist"]
    )
    pillar_features: int = 64
    backbone_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    backbone_strides: List[int] = field(default_factory=lambda: [2, 2, 2])
    backbone_num_blocks: List[int] = field(default_factory=lambda: [3, 5, 5])

    # Anchor config
    anchor_sizes: List[List[float]] = field(
        default_factory=lambda: [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]
    )
    anchor_rotations: List[float] = field(
        default_factory=lambda: [0.0, 1.5707963]
    )
    anchor_z_centers: List[float] = field(
        default_factory=lambda: [-1.0, -0.6, -0.6]
    )

    # Detection post-processing
    score_threshold: float = 0.3
    nms_iou_threshold: float = 0.5
    max_detections_per_class: int = 100
    max_total_detections: int = 300


# =============================================================================
# Point Cloud I/O
# =============================================================================


def read_point_cloud_bin(file_path: str) -> np.ndarray:
    """Read a KITTI-format binary point cloud file.

    The binary file contains float32 values, with each point stored as
    4 consecutive floats: (x, y, z, intensity). Points are stored in
    row-major order with no header.

    Args:
        file_path: Path to the .bin file.

    Returns:
        Point cloud array of shape [N, 4] (x, y, z, intensity).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file size is not a multiple of 16 bytes.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size % 16 != 0:
        raise ValueError(
            f"Invalid .bin file: size ({file_size} bytes) is not a multiple of "
            f"16 (4 floats * 4 bytes each). File may be corrupted: {file_path}"
        )

    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points


def read_point_cloud_pcd(file_path: str) -> np.ndarray:
    """Read a PCD (Point Cloud Data) file, returning xyz and intensity.

    Supports ASCII and binary PCD formats. Extracts x, y, z, and intensity
    fields. If intensity is not present, it is set to zero.

    Args:
        file_path: Path to the .pcd file.

    Returns:
        Point cloud array of shape [N, 4] (x, y, z, intensity).
    """
    with open(file_path, "rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line.startswith("DATA"):
                break

        # Parse header
        fields: List[str] = []
        num_points = 0
        data_format = "ascii"

        for line in header_lines:
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("POINTS"):
                num_points = int(line.split()[1])
            elif line.startswith("DATA"):
                data_format = line.split()[1].lower()

        # Find x, y, z, intensity field indices
        x_idx = fields.index("x") if "x" in fields else 0
        y_idx = fields.index("y") if "y" in fields else 1
        z_idx = fields.index("z") if "z" in fields else 2
        intensity_idx = fields.index("intensity") if "intensity" in fields else -1

        if data_format == "ascii":
            points_list: List[List[float]] = []
            for _ in range(num_points):
                line = f.readline().decode("ascii", errors="replace").strip()
                values = line.split()
                x = float(values[x_idx])
                y = float(values[y_idx])
                z = float(values[z_idx])
                intensity = float(values[intensity_idx]) if intensity_idx >= 0 else 0.0
                points_list.append([x, y, z, intensity])
            points = np.array(points_list, dtype=np.float32)
        else:
            # Binary format: read remaining bytes
            raw_data = f.read()
            num_fields = len(fields)
            all_data = np.frombuffer(raw_data, dtype=np.float32).reshape(num_points, num_fields)
            x_arr = all_data[:, x_idx]
            y_arr = all_data[:, y_idx]
            z_arr = all_data[:, z_idx]
            intensity_arr = all_data[:, intensity_idx] if intensity_idx >= 0 else np.zeros(num_points, dtype=np.float32)
            points = np.stack([x_arr, y_arr, z_arr, intensity_arr], axis=-1)

    return points


def read_point_cloud(file_path: str) -> np.ndarray:
    """Read a point cloud file, auto-detecting format from extension.

    Supported formats: .bin (KITTI binary), .pcd (Point Cloud Data),
    .npy (NumPy array).

    Args:
        file_path: Path to the point cloud file.

    Returns:
        Point cloud array of shape [N, 4] (x, y, z, intensity).

    Raises:
        ValueError: If the file format is not supported.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".bin":
        return read_point_cloud_bin(file_path)
    elif ext == ".pcd":
        return read_point_cloud_pcd(file_path)
    elif ext == ".npy":
        data = np.load(file_path)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array from .npy file, got shape {data.shape}")
        if data.shape[1] < 4:
            # Pad with zero intensity if only xyz
            padded = np.zeros((data.shape[0], 4), dtype=np.float32)
            padded[:, :data.shape[1]] = data
            return padded
        return data[:, :4].astype(np.float32)
    else:
        raise ValueError(
            f"Unsupported point cloud format: '{ext}'. "
            f"Supported: .bin, .pcd, .npy"
        )


# =============================================================================
# Voxelization (Pillarization)
# =============================================================================


def voxelize_point_cloud(
    points: np.ndarray, config: InferenceConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw point cloud into the pillar representation for the network.

    Performs range filtering, grid binning, point subsampling within pillars,
    and feature augmentation (mean offsets and pillar center offsets).

    Args:
        points: Raw point cloud [N, 4] (x, y, z, intensity).
        config: Inference configuration with voxel parameters.

    Returns:
        Tuple of:
            pillars: [max_voxels, max_points_per_voxel, 9] pillar features.
            coords: [max_voxels, 2] grid coordinates (grid_x, grid_y).
            num_points_per_pillar: [max_voxels] valid point counts.
    """
    pc_range = np.array(config.point_cloud_range, dtype=np.float32)
    voxel_size = np.array(config.voxel_size, dtype=np.float32)
    max_points = config.max_points_per_voxel
    max_voxels = config.max_voxels

    # Range filtering
    mask = (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] < pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] < pc_range[4])
        & (points[:, 2] >= pc_range[2])
        & (points[:, 2] < pc_range[5])
    )
    points = points[mask]

    if points.shape[0] == 0:
        return (
            np.zeros((max_voxels, max_points, 9), dtype=np.float32),
            np.zeros((max_voxels, 2), dtype=np.int32),
            np.zeros(max_voxels, dtype=np.int32),
        )

    # Compute grid indices
    grid_idx_x = ((points[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int32)
    grid_idx_y = ((points[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int32)

    grid_size_x = int((pc_range[3] - pc_range[0]) / voxel_size[0])
    grid_size_y = int((pc_range[4] - pc_range[1]) / voxel_size[1])

    grid_idx_x = np.clip(grid_idx_x, 0, grid_size_x - 1)
    grid_idx_y = np.clip(grid_idx_y, 0, grid_size_y - 1)

    # Compute unique pillar IDs
    pillar_ids = grid_idx_y * grid_size_x + grid_idx_x
    unique_pillars, inverse_indices = np.unique(pillar_ids, return_inverse=True)

    # Subsample pillars if exceeding max
    if len(unique_pillars) > max_voxels:
        selected = np.random.choice(len(unique_pillars), max_voxels, replace=False)
        selected_set = set(selected.tolist())
        keep_mask = np.array(
            [inverse_indices[i] in selected_set for i in range(len(points))],
            dtype=np.bool_,
        )
        points = points[keep_mask]
        grid_idx_x = grid_idx_x[keep_mask]
        grid_idx_y = grid_idx_y[keep_mask]
        pillar_ids = pillar_ids[keep_mask]
        unique_pillars, inverse_indices = np.unique(pillar_ids, return_inverse=True)

    num_pillars = min(len(unique_pillars), max_voxels)

    # Initialize output arrays
    pillars = np.zeros((max_voxels, max_points, 9), dtype=np.float32)
    coords = np.zeros((max_voxels, 2), dtype=np.int32)
    num_points_per_pillar = np.zeros(max_voxels, dtype=np.int32)

    for i in range(num_pillars):
        point_mask = inverse_indices == i
        pillar_points = points[point_mask]

        n_pts = min(len(pillar_points), max_points)
        if len(pillar_points) > max_points:
            choice = np.random.choice(len(pillar_points), max_points, replace=False)
            pillar_points = pillar_points[choice]
            n_pts = max_points

        # Pillar center in world coordinates
        first_idx = np.where(point_mask)[0][0]
        center_x = grid_idx_x[first_idx] * voxel_size[0] + pc_range[0] + voxel_size[0] / 2.0
        center_y = grid_idx_y[first_idx] * voxel_size[1] + pc_range[1] + voxel_size[1] / 2.0

        # Mean of point coordinates within pillar
        mean_xyz = pillar_points[:n_pts, :3].mean(axis=0)

        # Build 9-dim features: x, y, z, intensity, xc, yc, zc, xp, yp
        features = np.zeros((n_pts, 9), dtype=np.float32)
        features[:, :4] = pillar_points[:n_pts, :4]
        features[:, 4] = pillar_points[:n_pts, 0] - mean_xyz[0]  # offset from mean x
        features[:, 5] = pillar_points[:n_pts, 1] - mean_xyz[1]  # offset from mean y
        features[:, 6] = pillar_points[:n_pts, 2] - mean_xyz[2]  # offset from mean z
        features[:, 7] = pillar_points[:n_pts, 0] - center_x     # offset from pillar center x
        features[:, 8] = pillar_points[:n_pts, 1] - center_y     # offset from pillar center y

        pillars[i, :n_pts, :] = features
        coords[i, 0] = grid_idx_x[first_idx]
        coords[i, 1] = grid_idx_y[first_idx]
        num_points_per_pillar[i] = n_pts

    return pillars, coords, num_points_per_pillar


# =============================================================================
# Anchor Generation
# =============================================================================


def generate_anchors(config: InferenceConfig) -> np.ndarray:
    """Generate anchor boxes matching the detection head grid.

    Creates anchors at each spatial location of the feature map for all
    class sizes and orientations.

    Args:
        config: Inference config with anchor and voxel parameters.

    Returns:
        Anchor array of shape [total_anchors, 7] (x, y, z, w, l, h, yaw).
    """
    pc_range = config.point_cloud_range
    voxel_size = config.voxel_size
    feature_stride = 2

    grid_x = int((pc_range[3] - pc_range[0]) / (voxel_size[0] * feature_stride))
    grid_y = int((pc_range[4] - pc_range[1]) / (voxel_size[1] * feature_stride))

    x_centers = np.linspace(
        pc_range[0] + voxel_size[0] * feature_stride * 0.5,
        pc_range[3] - voxel_size[0] * feature_stride * 0.5,
        grid_x,
    )
    y_centers = np.linspace(
        pc_range[1] + voxel_size[1] * feature_stride * 0.5,
        pc_range[4] - voxel_size[1] * feature_stride * 0.5,
        grid_y,
    )

    xx, yy = np.meshgrid(x_centers, y_centers)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    num_locations = len(xx)

    all_anchors: List[np.ndarray] = []
    for cls_idx, size in enumerate(config.anchor_sizes):
        z_center = config.anchor_z_centers[cls_idx]
        w, l, h = size
        for rotation in config.anchor_rotations:
            anchors = np.stack(
                [
                    xx,
                    yy,
                    np.full(num_locations, z_center),
                    np.full(num_locations, w),
                    np.full(num_locations, l),
                    np.full(num_locations, h),
                    np.full(num_locations, rotation),
                ],
                axis=-1,
            )
            all_anchors.append(anchors)

    return np.concatenate(all_anchors, axis=0).astype(np.float32)


# =============================================================================
# Box Decoding
# =============================================================================


def decode_boxes(
    reg_preds: np.ndarray, anchors: np.ndarray
) -> np.ndarray:
    """Decode bounding box regression predictions relative to anchors.

    Decoding follows the standard anchor-based encoding:
        x = dx * diag(a) + a_x
        y = dy * diag(a) + a_y
        z = dz * a_h + a_z
        w = exp(dw) * a_w
        l = exp(dl) * a_l
        h = exp(dh) * a_h
        yaw = dr + a_yaw

    Args:
        reg_preds: Regression predictions [N, 7] (dx, dy, dz, dw, dl, dh, dr).
        anchors: Anchor boxes [N, 7] (x, y, z, w, l, h, yaw).

    Returns:
        Decoded boxes [N, 7] in world coordinates.
    """
    anchor_diag = np.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    x = reg_preds[:, 0] * anchor_diag + anchors[:, 0]
    y = reg_preds[:, 1] * anchor_diag + anchors[:, 1]
    z = reg_preds[:, 2] * anchors[:, 5] + anchors[:, 2]
    w = np.exp(reg_preds[:, 3]) * anchors[:, 3]
    l_dim = np.exp(reg_preds[:, 4]) * anchors[:, 4]
    h = np.exp(reg_preds[:, 5]) * anchors[:, 5]
    yaw = reg_preds[:, 6] + anchors[:, 6]

    return np.stack([x, y, z, w, l_dim, h, yaw], axis=-1).astype(np.float32)


# =============================================================================
# NMS and Score Filtering
# =============================================================================


def nms_bev(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Non-maximum suppression using BEV axis-aligned box overlap.

    Uses a greedy algorithm that iteratively selects the highest-scoring
    box and removes all overlapping boxes above the IoU threshold.

    Args:
        boxes: 3D boxes [N, 7] (x, y, z, w, l, h, yaw).
        scores: Confidence scores [N].
        iou_threshold: IoU threshold for suppression.

    Returns:
        Indices of kept boxes, sorted by decreasing score.
    """
    if len(scores) == 0:
        return np.array([], dtype=np.int64)

    # Convert to BEV axis-aligned boxes [x1, y1, x2, y2]
    x1 = boxes[:, 0] - boxes[:, 3] / 2.0
    y1 = boxes[:, 1] - boxes[:, 4] / 2.0
    x2 = boxes[:, 0] + boxes[:, 3] / 2.0
    y2 = boxes[:, 1] + boxes[:, 4] / 2.0

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)

    keep: List[int] = []
    while order.size > 0:
        idx = order[0]
        keep.append(int(idx))

        if order.size == 1:
            break

        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = areas[idx] + areas[order[1:]] - inter_area
        iou = inter_area / np.maximum(union_area, 1e-7)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.int64)


def filter_and_nms(
    cls_preds: np.ndarray,
    reg_preds: np.ndarray,
    dir_preds: np.ndarray,
    anchors: np.ndarray,
    config: InferenceConfig,
) -> np.ndarray:
    """Apply score filtering, box decoding, direction correction, and class-wise NMS.

    This is the complete post-processing pipeline that converts raw network
    outputs into final 3D bounding box detections.

    Args:
        cls_preds: Classification logits [num_anchors, num_classes].
        reg_preds: Regression predictions [num_anchors, 7].
        dir_preds: Direction classification logits [num_anchors, 2].
        anchors: Anchor boxes [num_anchors, 7].
        config: Inference configuration.

    Returns:
        Detections array [K, 9] where each row is
        (x, y, z, w, l, h, yaw, class_id, score).
    """
    num_classes = config.num_classes

    # Sigmoid activation for class scores
    cls_scores = 1.0 / (1.0 + np.exp(-cls_preds))

    # Decode all boxes
    decoded_boxes = decode_boxes(reg_preds, anchors)

    # Apply direction classification to correct heading
    dir_labels = np.argmax(dir_preds, axis=-1)
    dir_offset = dir_labels.astype(np.float32) * np.pi
    decoded_boxes[:, 6] = decoded_boxes[:, 6] + dir_offset
    # Normalize yaw to [-pi, pi]
    decoded_boxes[:, 6] = np.arctan2(
        np.sin(decoded_boxes[:, 6]), np.cos(decoded_boxes[:, 6])
    )

    # Per-class NMS
    all_detections: List[np.ndarray] = []

    for cls_idx in range(num_classes):
        class_scores = cls_scores[:, cls_idx]

        # Score filtering
        score_mask = class_scores > config.score_threshold
        filtered_scores = class_scores[score_mask]
        filtered_boxes = decoded_boxes[score_mask]

        if len(filtered_scores) == 0:
            continue

        # NMS
        keep_indices = nms_bev(filtered_boxes, filtered_scores, config.nms_iou_threshold)

        if len(keep_indices) > config.max_detections_per_class:
            keep_indices = keep_indices[:config.max_detections_per_class]

        kept_boxes = filtered_boxes[keep_indices]
        kept_scores = filtered_scores[keep_indices]

        # Build detection rows: [x, y, z, w, l, h, yaw, class_id, score]
        num_kept = kept_boxes.shape[0]
        class_ids = np.full((num_kept, 1), cls_idx, dtype=np.float32)
        scores_col = kept_scores.reshape(-1, 1)
        detections = np.concatenate([kept_boxes, class_ids, scores_col], axis=-1)
        all_detections.append(detections)

    if len(all_detections) == 0:
        return np.zeros((0, 9), dtype=np.float32)

    all_detections_arr = np.concatenate(all_detections, axis=0)

    # Sort by score and limit total detections
    sorted_indices = np.argsort(-all_detections_arr[:, 8])
    if len(sorted_indices) > config.max_total_detections:
        sorted_indices = sorted_indices[:config.max_total_detections]

    return all_detections_arr[sorted_indices].astype(np.float32)


# =============================================================================
# Model Loading
# =============================================================================


def build_model_from_config(config: InferenceConfig) -> tf.keras.Model:
    """Build the PointPillars model architecture from inference config.

    Constructs the model by importing the training module's model class
    and configuring it with the inference parameters.

    Args:
        config: Inference configuration.

    Returns:
        Uninitialized PointPillarsModel.
    """
    from . import train as train_module

    train_config = train_module.TrainConfig(
        num_classes=config.num_classes,
        pillar_features=config.pillar_features,
        backbone_channels=config.backbone_channels,
        backbone_strides=config.backbone_strides,
        backbone_num_blocks=config.backbone_num_blocks,
    )

    train_config.anchors = []
    for cls_idx in range(config.num_classes):
        anchor_cfg = train_module.AnchorConfig(
            class_name=config.class_names[cls_idx],
            anchor_sizes=[config.anchor_sizes[cls_idx]],
            anchor_rotations=config.anchor_rotations,
            anchor_z_center=config.anchor_z_centers[cls_idx],
        )
        train_config.anchors.append(anchor_cfg)

    train_config.voxel = train_module.VoxelConfig(
        point_cloud_range=config.point_cloud_range,
        voxel_size=config.voxel_size,
        max_points_per_voxel=config.max_points_per_voxel,
        max_voxels=config.max_voxels,
    )

    model = train_module.PointPillarsModel(train_config, name="pointpillars")
    return model


def load_checkpoint(config: InferenceConfig) -> tf.keras.Model:
    """Load model weights from a TensorFlow checkpoint.

    Builds the model architecture, runs a dummy forward pass to create
    all variables, then restores weights from the checkpoint.

    Args:
        config: Inference config with checkpoint_path set.

    Returns:
        Model with restored weights.
    """
    model = build_model_from_config(config)

    # Initialize model weights with a dummy forward pass
    max_voxels = config.max_voxels
    max_points = config.max_points_per_voxel
    dummy_pillars = tf.zeros([1, max_voxels, max_points, 9], dtype=tf.float32)
    dummy_coords = tf.zeros([1, max_voxels, 2], dtype=tf.int32)
    dummy_num_points = tf.zeros([1, max_voxels], dtype=tf.int32)
    model(dummy_pillars, dummy_coords, dummy_num_points, training=False)

    # Restore weights
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(config.checkpoint_path)
    status.expect_partial()
    logger.info("Loaded checkpoint: %s", config.checkpoint_path)

    return model


def load_saved_model(config: InferenceConfig) -> Any:
    """Load a TF2 SavedModel for inference.

    SavedModels contain the complete graph and weights, enabling
    direct inference without rebuilding the architecture.

    Args:
        config: Inference config with saved_model_path set.

    Returns:
        Loaded SavedModel object.
    """
    model = tf.saved_model.load(config.saved_model_path)
    logger.info("Loaded SavedModel: %s", config.saved_model_path)
    return model


def load_model(config: InferenceConfig) -> Tuple[Any, bool]:
    """Load model from either checkpoint or SavedModel.

    Args:
        config: Inference configuration.

    Returns:
        Tuple of (model, is_saved_model_flag).

    Raises:
        ValueError: If neither checkpoint nor saved_model path is specified.
    """
    if config.saved_model_path and os.path.isdir(config.saved_model_path):
        return load_saved_model(config), True
    elif config.checkpoint_path:
        return load_checkpoint(config), False
    else:
        raise ValueError(
            "Must specify either --checkpoint or --saved_model for model loading."
        )


# =============================================================================
# Single Point Cloud Inference
# =============================================================================


def run_model_forward(
    model: Any,
    pillars: np.ndarray,
    coords: np.ndarray,
    num_points: np.ndarray,
    is_saved_model: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute a single forward pass through the model.

    Handles both checkpoint-based models (direct call) and SavedModels
    (may use serving signatures).

    Args:
        model: Loaded model object.
        pillars: Pillar features [max_voxels, max_points, 9].
        coords: Grid coordinates [max_voxels, 2].
        num_points: Point counts [max_voxels].
        is_saved_model: Whether model is a TF SavedModel.

    Returns:
        Tuple of (cls_preds, reg_preds, dir_preds), each [num_anchors, C].
    """
    pillars_tf = tf.constant(pillars[np.newaxis], dtype=tf.float32)
    coords_tf = tf.constant(coords[np.newaxis], dtype=tf.int32)
    num_points_tf = tf.constant(num_points[np.newaxis], dtype=tf.int32)

    if is_saved_model:
        infer_fn = model.signatures.get("serving_default", None)
        if infer_fn is not None:
            outputs = infer_fn(
                pillars=pillars_tf, coords=coords_tf, num_points=num_points_tf
            )
            cls_preds = outputs["cls_preds"].numpy()[0]
            reg_preds = outputs["reg_preds"].numpy()[0]
            dir_preds = outputs["dir_preds"].numpy()[0]
        else:
            result = model(pillars_tf, coords_tf, num_points_tf, training=False)
            cls_preds = result[0].numpy()[0]
            reg_preds = result[1].numpy()[0]
            dir_preds = result[2].numpy()[0]
    else:
        cls_preds_tf, reg_preds_tf, dir_preds_tf = model(
            pillars_tf, coords_tf, num_points_tf, training=False
        )
        cls_preds = cls_preds_tf.numpy()[0]
        reg_preds = reg_preds_tf.numpy()[0]
        dir_preds = dir_preds_tf.numpy()[0]

    return cls_preds, reg_preds, dir_preds


def process_single_point_cloud(
    model: Any,
    file_path: str,
    anchors: np.ndarray,
    config: InferenceConfig,
    is_saved_model: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Process a single point cloud file end-to-end.

    Pipeline: read -> voxelize -> forward pass -> post-process -> detections.

    Args:
        model: Loaded model.
        file_path: Path to the point cloud file.
        anchors: Pre-generated anchor boxes.
        config: Inference configuration.
        is_saved_model: Whether model is a SavedModel.

    Returns:
        Tuple of:
            detections: [K, 9] array (x, y, z, w, l, h, yaw, class_id, score).
            timing: Dict with timing breakdown in milliseconds.
    """
    timing: Dict[str, float] = {}

    # Read point cloud
    t0 = time.perf_counter()
    points = read_point_cloud(file_path)
    timing["read_ms"] = (time.perf_counter() - t0) * 1000.0

    # Voxelize
    t0 = time.perf_counter()
    pillars, coords, num_points = voxelize_point_cloud(points, config)
    timing["voxelize_ms"] = (time.perf_counter() - t0) * 1000.0

    # Forward pass
    t0 = time.perf_counter()
    cls_preds, reg_preds, dir_preds = run_model_forward(
        model, pillars, coords, num_points, is_saved_model
    )
    timing["forward_ms"] = (time.perf_counter() - t0) * 1000.0

    # Post-processing (NMS + score filtering)
    t0 = time.perf_counter()
    detections = filter_and_nms(cls_preds, reg_preds, dir_preds, anchors, config)
    timing["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    timing["total_ms"] = sum(timing.values())
    timing["num_points"] = float(points.shape[0])
    timing["num_detections"] = float(detections.shape[0])

    return detections, timing


# =============================================================================
# Batch Processing
# =============================================================================


def run_model_forward_batch(
    model: Any,
    pillars_batch: np.ndarray,
    coords_batch: np.ndarray,
    num_points_batch: np.ndarray,
    is_saved_model: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run batched model inference.

    Args:
        model: Loaded model.
        pillars_batch: Pillar features [B, max_voxels, max_points, 9].
        coords_batch: Grid coordinates [B, max_voxels, 2].
        num_points_batch: Point counts [B, max_voxels].
        is_saved_model: Whether model is a SavedModel.

    Returns:
        Tuple of (cls_preds, reg_preds, dir_preds), each [B, num_anchors, C].
    """
    pillars_tf = tf.constant(pillars_batch, dtype=tf.float32)
    coords_tf = tf.constant(coords_batch, dtype=tf.int32)
    num_points_tf = tf.constant(num_points_batch, dtype=tf.int32)

    if is_saved_model:
        infer_fn = model.signatures.get("serving_default", None)
        if infer_fn is not None:
            outputs = infer_fn(
                pillars=pillars_tf, coords=coords_tf, num_points=num_points_tf
            )
            cls_preds = outputs["cls_preds"].numpy()
            reg_preds = outputs["reg_preds"].numpy()
            dir_preds = outputs["dir_preds"].numpy()
        else:
            result = model(pillars_tf, coords_tf, num_points_tf, training=False)
            cls_preds = result[0].numpy()
            reg_preds = result[1].numpy()
            dir_preds = result[2].numpy()
    else:
        cls_preds_tf, reg_preds_tf, dir_preds_tf = model(
            pillars_tf, coords_tf, num_points_tf, training=False
        )
        cls_preds = cls_preds_tf.numpy()
        reg_preds = reg_preds_tf.numpy()
        dir_preds = dir_preds_tf.numpy()

    return cls_preds, reg_preds, dir_preds


def process_batch(
    model: Any,
    file_paths: List[str],
    anchors: np.ndarray,
    config: InferenceConfig,
    is_saved_model: bool = False,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """Process a batch of point cloud files with batched inference.

    Voxelizes each point cloud independently, then runs a single batched
    forward pass through the network for efficiency.

    Args:
        model: Loaded model.
        file_paths: List of point cloud file paths.
        anchors: Pre-generated anchor boxes.
        config: Inference configuration.
        is_saved_model: Whether model is a SavedModel.

    Returns:
        Tuple of:
            detections_list: List of detection arrays, one per file.
            timing: Timing breakdown for the batch.
    """
    batch_size = len(file_paths)
    max_voxels = config.max_voxels
    max_points = config.max_points_per_voxel
    timing: Dict[str, float] = {}

    # Voxelize all point clouds
    t0 = time.perf_counter()
    pillars_batch = np.zeros((batch_size, max_voxels, max_points, 9), dtype=np.float32)
    coords_batch = np.zeros((batch_size, max_voxels, 2), dtype=np.int32)
    num_points_batch = np.zeros((batch_size, max_voxels), dtype=np.int32)

    for i, fp in enumerate(file_paths):
        points = read_point_cloud(fp)
        pillars, coords, num_pts = voxelize_point_cloud(points, config)
        pillars_batch[i] = pillars
        coords_batch[i] = coords
        num_points_batch[i] = num_pts

    timing["voxelize_ms"] = (time.perf_counter() - t0) * 1000.0

    # Batched forward pass
    t0 = time.perf_counter()
    cls_preds, reg_preds, dir_preds = run_model_forward_batch(
        model, pillars_batch, coords_batch, num_points_batch, is_saved_model
    )
    timing["forward_ms"] = (time.perf_counter() - t0) * 1000.0

    # Post-process each sample independently
    t0 = time.perf_counter()
    detections_list: List[np.ndarray] = []
    for i in range(batch_size):
        detections = filter_and_nms(
            cls_preds[i], reg_preds[i], dir_preds[i], anchors, config
        )
        detections_list.append(detections)
    timing["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    timing["total_ms"] = sum(timing.values())
    timing["batch_size"] = float(batch_size)
    timing["per_sample_ms"] = timing["total_ms"] / max(batch_size, 1)

    return detections_list, timing


def process_directory(
    model: Any,
    input_dir: str,
    output_dir: str,
    anchors: np.ndarray,
    config: InferenceConfig,
    batch_size: int = 1,
    is_saved_model: bool = False,
) -> Dict[str, Any]:
    """Process all point cloud files in a directory.

    Supports batched inference for higher throughput. Saves detection results
    to the output directory as text files.

    Args:
        model: Loaded model.
        input_dir: Directory containing point cloud files.
        output_dir: Directory to save detection results.
        anchors: Pre-generated anchor boxes.
        config: Inference configuration.
        batch_size: Number of files to process in each batch.
        is_saved_model: Whether model is a SavedModel.

    Returns:
        Summary statistics dict.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all supported point cloud files
    supported_extensions = {".bin", ".pcd", ".npy"}
    all_files = sorted(
        [
            str(f)
            for f in input_path.iterdir()
            if f.suffix.lower() in supported_extensions
        ]
    )

    if len(all_files) == 0:
        logger.warning("No point cloud files found in %s", input_dir)
        return {"total_files": 0, "total_detections": 0}

    logger.info("Found %d point cloud files in %s", len(all_files), input_dir)

    total_detections = 0
    total_time_ms = 0.0
    all_timings: List[Dict[str, float]] = []

    # Process in batches
    for batch_start in range(0, len(all_files), batch_size):
        batch_files = all_files[batch_start:batch_start + batch_size]

        if batch_size > 1 and len(batch_files) > 1:
            detections_list, timing = process_batch(
                model, batch_files, anchors, config, is_saved_model
            )
        else:
            # Single-file processing for batch_size=1 or last incomplete batch
            detections_list = []
            timing_total = 0.0
            for fp in batch_files:
                dets, single_timing = process_single_point_cloud(
                    model, fp, anchors, config, is_saved_model
                )
                detections_list.append(dets)
                timing_total += single_timing["total_ms"]
            timing = {"total_ms": timing_total, "per_sample_ms": timing_total / len(batch_files)}

        all_timings.append(timing)
        total_time_ms += timing["total_ms"]

        # Save results
        for file_path, detections in zip(batch_files, detections_list):
            stem = Path(file_path).stem
            out_file = output_path / f"{stem}.txt"
            save_detections_txt(detections, str(out_file), config.class_names)
            total_detections += detections.shape[0]

        processed = min(batch_start + batch_size, len(all_files))
        if processed % max(1, len(all_files) // 10) < batch_size:
            avg_ms = total_time_ms / processed
            logger.info(
                "  Processed %d/%d files (avg %.1f ms/file, %.1f FPS)",
                processed,
                len(all_files),
                avg_ms,
                1000.0 / max(avg_ms, 1e-9),
            )

    summary = {
        "total_files": len(all_files),
        "total_detections": total_detections,
        "total_time_ms": total_time_ms,
        "avg_time_per_file_ms": total_time_ms / max(len(all_files), 1),
        "throughput_fps": 1000.0 * len(all_files) / max(total_time_ms, 1e-9),
        "output_dir": str(output_path.resolve()),
    }

    return summary


# =============================================================================
# Output Formatting
# =============================================================================


def save_detections_txt(
    detections: np.ndarray,
    output_path: str,
    class_names: List[str],
) -> None:
    """Save detection results to a text file.

    Each line contains: class_name x y z w l h yaw score

    Args:
        detections: Detection array [K, 9] (x, y, z, w, l, h, yaw, class_id, score).
        output_path: Path to output text file.
        class_names: List of class name strings for ID-to-name mapping.
    """
    with open(output_path, "w") as f:
        for det in detections:
            x, y, z, w, l, h, yaw, cls_id, score = det
            cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"class_{int(cls_id)}"
            f.write(
                f"{cls_name} {x:.4f} {y:.4f} {z:.4f} "
                f"{w:.4f} {l:.4f} {h:.4f} {yaw:.4f} {score:.4f}\n"
            )


def format_detections_table(
    detections: np.ndarray,
    class_names: List[str],
) -> str:
    """Format detections as a human-readable table string.

    Args:
        detections: Detection array [K, 9].
        class_names: List of class name strings.

    Returns:
        Formatted table string.
    """
    if detections.shape[0] == 0:
        return "No detections."

    lines: List[str] = []
    header = f"{'#':<4} {'Class':<12} {'Score':>6} {'X':>7} {'Y':>7} {'Z':>6} {'W':>5} {'L':>5} {'H':>5} {'Yaw':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, det in enumerate(detections):
        x, y, z, w, l, h, yaw, cls_id, score = det
        cls_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"cls_{int(cls_id)}"
        lines.append(
            f"{i:<4} {cls_name:<12} {score:>6.3f} {x:>7.2f} {y:>7.2f} {z:>6.2f} "
            f"{w:>5.2f} {l:>5.2f} {h:>5.2f} {yaw:>6.3f}"
        )

    return "\n".join(lines)


# =============================================================================
# Performance Measurement
# =============================================================================


def measure_inference_speed(
    model: Any,
    sample_file: str,
    anchors: np.ndarray,
    config: InferenceConfig,
    is_saved_model: bool = False,
    warmup: int = 5,
    iterations: int = 50,
) -> Dict[str, float]:
    """Measure end-to-end inference speed with warmup.

    Runs the full pipeline (voxelize + forward + postprocess) multiple times
    to produce reliable timing statistics.

    Args:
        model: Loaded model.
        sample_file: Path to a sample point cloud for benchmarking.
        anchors: Pre-generated anchors.
        config: Inference configuration.
        is_saved_model: Whether model is a SavedModel.
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.

    Returns:
        Timing statistics dict.
    """
    # Pre-load and voxelize once for consistent benchmarking
    points = read_point_cloud(sample_file)
    pillars, coords, num_points = voxelize_point_cloud(points, config)

    # Warmup
    logger.info("Warmup: %d iterations...", warmup)
    for _ in range(warmup):
        run_model_forward(model, pillars, coords, num_points, is_saved_model)

    # Benchmark: forward pass only
    logger.info("Benchmarking forward pass: %d iterations...", iterations)
    forward_times: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        run_model_forward(model, pillars, coords, num_points, is_saved_model)
        forward_times.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark: full pipeline (voxelize + forward + postprocess)
    logger.info("Benchmarking full pipeline: %d iterations...", iterations)
    full_times: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        pillars_i, coords_i, num_pts_i = voxelize_point_cloud(points, config)
        cls_p, reg_p, dir_p = run_model_forward(
            model, pillars_i, coords_i, num_pts_i, is_saved_model
        )
        filter_and_nms(cls_p, reg_p, dir_p, anchors, config)
        full_times.append((time.perf_counter() - t0) * 1000.0)

    forward_arr = np.array(forward_times)
    full_arr = np.array(full_times)

    return {
        "forward_mean_ms": float(np.mean(forward_arr)),
        "forward_std_ms": float(np.std(forward_arr)),
        "forward_p50_ms": float(np.percentile(forward_arr, 50)),
        "forward_p95_ms": float(np.percentile(forward_arr, 95)),
        "forward_fps": 1000.0 / float(np.mean(forward_arr)),
        "full_pipeline_mean_ms": float(np.mean(full_arr)),
        "full_pipeline_std_ms": float(np.std(full_arr)),
        "full_pipeline_p50_ms": float(np.percentile(full_arr, 50)),
        "full_pipeline_p95_ms": float(np.percentile(full_arr, 95)),
        "full_pipeline_fps": 1000.0 / float(np.mean(full_arr)),
        "num_input_points": int(points.shape[0]),
        "warmup_iterations": warmup,
        "benchmark_iterations": iterations,
    }


# =============================================================================
# CLI Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="PointPillars TF2 Inference - 3D Object Detection from LiDAR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model loading (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to TF2 checkpoint (e.g., ./output/checkpoints/ckpt-32).",
    )
    model_group.add_argument(
        "--saved_model",
        type=str,
        help="Path to TF2 SavedModel directory.",
    )

    # Input (single file or directory)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to a single point cloud file (.bin, .pcd, .npy).",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory of point cloud files for batch processing.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save detection results (used with --input_dir).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save detection results for single file (optional).",
    )

    # Detection parameters
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Minimum detection confidence score.",
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--max_detections",
        type=int,
        default=100,
        help="Maximum detections per class.",
    )

    # Batch processing
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for batch processing (with --input_dir).",
    )

    # Model architecture (defaults match train.py)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Number of detection classes.",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=["Car", "Pedestrian", "Cyclist"],
        help="Class names in order.",
    )

    # Performance
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference speed benchmark after processing.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations for benchmarking.",
    )
    parser.add_argument(
        "--benchmark_iterations",
        type=int,
        default=50,
        help="Number of timed iterations for benchmarking.",
    )

    # GPU
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU ID to use (e.g., '0'). Defaults to first available.",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed timing breakdown.",
    )

    return parser.parse_args()


def main() -> None:
    """Main inference entry point.

    Handles CLI argument parsing, model loading, single/batch inference
    execution, result output, and optional benchmarking.
    """
    args = parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # GPU configuration
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning("Could not set memory growth: %s", e)

    logger.info("TensorFlow version: %s", tf.__version__)
    logger.info("GPUs available: %d", len(gpus))

    # Build configuration
    config = InferenceConfig(
        checkpoint_path=args.checkpoint,
        saved_model_path=args.saved_model,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections_per_class=args.max_detections,
        num_classes=args.num_classes,
        class_names=args.class_names,
    )

    # Load model
    logger.info("Loading model...")
    model, is_saved_model = load_model(config)

    # Generate anchors
    anchors = generate_anchors(config)
    logger.info("Generated %d anchors", anchors.shape[0])

    # Execute inference
    if args.input:
        # Single file processing
        logger.info("Processing: %s", args.input)
        detections, timing = process_single_point_cloud(
            model, args.input, anchors, config, is_saved_model
        )

        # Print results
        logger.info("Detected %d objects:", detections.shape[0])
        table = format_detections_table(detections, config.class_names)
        print("\n" + table + "\n")

        # Print timing
        logger.info(
            "Timing - Read: %.1f ms | Voxelize: %.1f ms | "
            "Forward: %.1f ms | PostProcess: %.1f ms | Total: %.1f ms",
            timing["read_ms"],
            timing["voxelize_ms"],
            timing["forward_ms"],
            timing["postprocess_ms"],
            timing["total_ms"],
        )

        # Save to file if requested
        if args.output_file:
            save_detections_txt(detections, args.output_file, config.class_names)
            logger.info("Results saved to: %s", args.output_file)

        # Benchmark if requested
        if args.benchmark:
            logger.info("Running speed benchmark...")
            speed_results = measure_inference_speed(
                model, args.input, anchors, config, is_saved_model,
                warmup=args.warmup, iterations=args.benchmark_iterations,
            )
            logger.info("=== Performance Benchmark ===")
            logger.info("Forward pass:    %.1f FPS (%.2f +/- %.2f ms)",
                        speed_results["forward_fps"],
                        speed_results["forward_mean_ms"],
                        speed_results["forward_std_ms"])
            logger.info("Full pipeline:   %.1f FPS (%.2f +/- %.2f ms)",
                        speed_results["full_pipeline_fps"],
                        speed_results["full_pipeline_mean_ms"],
                        speed_results["full_pipeline_std_ms"])
            logger.info("P95 latency:     forward=%.2f ms, full=%.2f ms",
                        speed_results["forward_p95_ms"],
                        speed_results["full_pipeline_p95_ms"])

    elif args.input_dir:
        # Batch/directory processing
        out_dir = args.output_dir or str(Path(args.input_dir) / "detections")
        logger.info("Batch processing: %s -> %s (batch_size=%d)",
                    args.input_dir, out_dir, args.batch_size)

        summary = process_directory(
            model, args.input_dir, out_dir, anchors, config,
            batch_size=args.batch_size, is_saved_model=is_saved_model,
        )

        logger.info("=== Batch Processing Summary ===")
        logger.info("Files processed:     %d", summary["total_files"])
        logger.info("Total detections:    %d", summary["total_detections"])
        logger.info("Total time:          %.1f ms", summary["total_time_ms"])
        logger.info("Avg time per file:   %.1f ms", summary["avg_time_per_file_ms"])
        logger.info("Throughput:          %.1f FPS", summary["throughput_fps"])
        logger.info("Output directory:    %s", summary["output_dir"])

        # Benchmark if requested
        if args.benchmark:
            input_path = Path(args.input_dir)
            supported_extensions = {".bin", ".pcd", ".npy"}
            sample_files = [
                str(f) for f in input_path.iterdir()
                if f.suffix.lower() in supported_extensions
            ]
            if sample_files:
                logger.info("Running speed benchmark on first file...")
                speed_results = measure_inference_speed(
                    model, sample_files[0], anchors, config, is_saved_model,
                    warmup=args.warmup, iterations=args.benchmark_iterations,
                )
                logger.info("=== Performance Benchmark ===")
                logger.info("Forward pass:    %.1f FPS (%.2f +/- %.2f ms)",
                            speed_results["forward_fps"],
                            speed_results["forward_mean_ms"],
                            speed_results["forward_std_ms"])
                logger.info("Full pipeline:   %.1f FPS (%.2f +/- %.2f ms)",
                            speed_results["full_pipeline_fps"],
                            speed_results["full_pipeline_mean_ms"],
                            speed_results["full_pipeline_std_ms"])

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
