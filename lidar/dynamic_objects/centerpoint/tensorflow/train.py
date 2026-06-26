#!/usr/bin/env python3
"""
CenterPoint Training Script (TensorFlow 2)

Full-featured training pipeline for CenterPoint 3D LiDAR object detection with:
- Multi-GPU training via tf.distribute.MirroredStrategy
- Custom training loop with mixed precision (FP16)
- tf.data.Dataset pipeline with voxelization and Gaussian heatmap target generation
- Data augmentation: random flip, global rotation, global scaling, GT-sampling
- OneCycle-like learning rate schedule (cosine decay with linear warmup)
- @tf.function compiled training step with GradientTape
- TensorBoard logging for loss curves and metrics
- Checkpoint management (best and periodic)
- Command-line arguments for all training hyperparameters

Usage:
    python train.py --data_path /data/nuscenes/lidar_bins \
                    --ann_path /data/nuscenes/infos_train.pkl \
                    --batch_size 4 \
                    --epochs 20 \
                    --lr 0.001 \
                    --num_gpus 4 \
                    --output_dir ./work_dirs/centerpoint_tf

Reference: "Center-based 3D Object Detection and Tracking" (Yin et al., CVPR 2021)
"""

import argparse
import logging
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from model import (
    CenterPointModel,
    VOXEL_SIZE,
    POINT_CLOUD_RANGE,
    GRID_SIZE,
    NUSCENES_TASK_GROUPS,
    dynamic_voxelization,
    gaussian_focal_loss,
    reg_l1_loss,
    generate_gaussian_target,
    gaussian_radius,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("centerpoint.train")


# =============================================================================
# Constants
# =============================================================================

BEV_RESOLUTION = 180  # BEV feature map H and W (1440 / 8)
FEATURE_MAP_STRIDE = 8  # Downsampling factor from voxel grid to BEV feature map

# Loss weights (from config: height_weight is 2.0 per user spec)
LOSS_WEIGHTS = {
    "heatmap": 1.0,
    "offset": 2.0,
    "height": 2.0,
    "size": 0.2,
    "rotation": 1.0,
    "velocity": 0.2,
}

# nuScenes class name to index mapping (within each task group)
NUSCENES_CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# Mapping from class name to (task_group_idx, class_idx_within_group)
CLASS_TO_TASK_MAP = {}
for task_idx, group in enumerate(NUSCENES_TASK_GROUPS):
    for cls_idx, cls_name in enumerate(group):
        CLASS_TO_TASK_MAP[cls_name] = (task_idx, cls_idx)

# GT-sampling parameters
GT_SAMPLE_GROUPS = {
    "car": 15,
    "truck": 3,
    "construction_vehicle": 7,
    "bus": 4,
    "trailer": 6,
    "barrier": 10,
    "motorcycle": 6,
    "bicycle": 6,
    "pedestrian": 10,
    "traffic_cone": 10,
}

GT_MIN_POINTS = {cls: 5 for cls in NUSCENES_CLASS_NAMES}


# =============================================================================
# Learning Rate Schedule: OneCycle-like (Warmup + Cosine Decay)
# =============================================================================


class WarmupCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """OneCycle-like learning rate: linear warmup followed by cosine decay.

    Mimics OneCycleLR with pct_start=0.4, anneal_strategy='cos',
    div_factor=10, final_div_factor=100.

    Args:
        max_lr: Peak learning rate after warmup.
        total_steps: Total number of training steps.
        warmup_fraction: Fraction of total steps used for warmup (default: 0.4).
        div_factor: Divides max_lr for initial lr (start_lr = max_lr / div_factor).
        final_div_factor: Divides max_lr for final lr.
    """

    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        warmup_fraction: float = 0.4,
        div_factor: float = 10.0,
        final_div_factor: float = 100.0,
    ):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Phase 1: Linear warmup from initial_lr to max_lr
        warmup_progress = step / tf.maximum(warmup_steps, 1.0)
        warmup_lr = self.initial_lr + (self.max_lr - self.initial_lr) * warmup_progress

        # Phase 2: Cosine annealing from max_lr to final_lr
        decay_steps = total_steps - warmup_steps
        decay_progress = (step - warmup_steps) / tf.maximum(decay_steps, 1.0)
        decay_progress = tf.minimum(decay_progress, 1.0)
        cosine_lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (
            1.0 + tf.math.cos(math.pi * decay_progress)
        )

        lr = tf.where(step < warmup_steps, warmup_lr, cosine_lr)
        return lr

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
        }


# =============================================================================
# Data Augmentation (NumPy, executed via tf.py_function)
# =============================================================================


def augment_point_cloud(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_db: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    enable_gt_sampling: bool = True,
    enable_flip: bool = True,
    enable_rotation: bool = True,
    enable_scaling: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply data augmentation to point cloud and bounding boxes.

    Augmentations applied in order:
    1. GT-sampling: paste ground truth objects from database
    2. Random flip along X and/or Y axes
    3. Global rotation around Z axis
    4. Global scaling

    Args:
        points: (N, 5) point cloud [x, y, z, intensity, timestamp].
        gt_boxes: (M, 9) ground truth boxes [x, y, z, l, w, h, yaw, vx, vy].
        gt_labels: (M,) integer class labels.
        gt_db: Ground truth database for GT-sampling. Dict mapping class name
            to list of dicts with keys 'box', 'points', 'num_points'.
        enable_gt_sampling: Whether to apply GT-sampling augmentation.
        enable_flip: Whether to apply random flip.
        enable_rotation: Whether to apply random rotation.
        enable_scaling: Whether to apply random scaling.

    Returns:
        aug_points: Augmented point cloud.
        aug_boxes: Augmented ground truth boxes.
        aug_labels: Augmented labels (unchanged unless GT-sampling adds objects).
    """
    aug_points = points.copy()
    aug_boxes = gt_boxes.copy()
    aug_labels = gt_labels.copy()

    # 1. GT-Sampling: paste objects from database into the scene
    if enable_gt_sampling and gt_db is not None:
        aug_points, aug_boxes, aug_labels = _apply_gt_sampling(
            aug_points, aug_boxes, aug_labels, gt_db
        )

    # 2. Random Flip
    if enable_flip:
        aug_points, aug_boxes = _apply_random_flip(aug_points, aug_boxes)

    # 3. Global Rotation (around Z-axis)
    if enable_rotation:
        aug_points, aug_boxes = _apply_global_rotation(aug_points, aug_boxes)

    # 4. Global Scaling
    if enable_scaling:
        aug_points, aug_boxes = _apply_global_scaling(aug_points, aug_boxes)

    return aug_points, aug_boxes, aug_labels


def _apply_gt_sampling(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_db: Dict[str, List[Dict[str, Any]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paste ground truth objects from database into the scene.

    For each class, sample a specified number of objects from the database
    and place them into the current scene, ensuring no collision with existing boxes.

    Args:
        points: (N, 5) current scene points.
        gt_boxes: (M, 9) current ground truth boxes.
        gt_labels: (M,) current class labels.
        gt_db: Ground truth database.

    Returns:
        Updated points, boxes, and labels with sampled objects added.
    """
    sampled_points_list = []
    sampled_boxes_list = []
    sampled_labels_list = []

    for cls_name, num_samples in GT_SAMPLE_GROUPS.items():
        if cls_name not in gt_db or len(gt_db[cls_name]) == 0:
            continue

        # Determine class label index
        cls_idx = NUSCENES_CLASS_NAMES.index(cls_name) if cls_name in NUSCENES_CLASS_NAMES else -1
        if cls_idx < 0:
            continue

        # Count existing objects of this class
        existing_count = np.sum(gt_labels == cls_idx)
        samples_needed = max(0, num_samples - int(existing_count))
        if samples_needed == 0:
            continue

        # Sample from database
        db_entries = gt_db[cls_name]
        sample_indices = np.random.choice(
            len(db_entries), size=min(samples_needed, len(db_entries)), replace=False
        )

        for idx in sample_indices:
            entry = db_entries[idx]
            db_box = entry["box"]  # (9,)
            db_points = entry["points"]  # (K, 5)

            if db_points.shape[0] < GT_MIN_POINTS.get(cls_name, 5):
                continue

            # Check collision with existing boxes using BEV IoU approximation
            if gt_boxes.shape[0] > 0:
                distances = np.sqrt(
                    (gt_boxes[:, 0] - db_box[0]) ** 2
                    + (gt_boxes[:, 1] - db_box[1]) ** 2
                )
                min_dist = np.min(distances) if len(distances) > 0 else float("inf")
                # Skip if too close to an existing object
                if min_dist < 1.0:
                    continue

            sampled_points_list.append(db_points)
            sampled_boxes_list.append(db_box)
            sampled_labels_list.append(cls_idx)

    # Merge sampled objects into the scene
    if sampled_points_list:
        sampled_points = np.concatenate(sampled_points_list, axis=0)
        sampled_boxes = np.array(sampled_boxes_list, dtype=np.float32)
        sampled_labels = np.array(sampled_labels_list, dtype=np.int32)

        # Remove points inside sampled boxes from original scene
        points = _remove_points_in_boxes(points, sampled_boxes)

        # Add sampled points and annotations
        points = np.concatenate([points, sampled_points], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_boxes], axis=0)
        gt_labels = np.concatenate([gt_labels, sampled_labels], axis=0)

    return points, gt_boxes, gt_labels


def _remove_points_in_boxes(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Remove points that fall inside any of the given 3D bounding boxes.

    Uses an axis-aligned bounding box approximation for efficiency.

    Args:
        points: (N, 5) point cloud.
        boxes: (K, 9) bounding boxes [x, y, z, l, w, h, yaw, vx, vy].

    Returns:
        Filtered points with in-box points removed.
    """
    if boxes.shape[0] == 0:
        return points

    mask = np.ones(points.shape[0], dtype=bool)
    for box in boxes:
        cx, cy, cz, l, w, h = box[0], box[1], box[2], box[3], box[4], box[5]
        # Axis-aligned approximation (use max of l, w as radius)
        half_diag = np.sqrt(l ** 2 + w ** 2) / 2.0
        in_x = np.abs(points[:, 0] - cx) < half_diag
        in_y = np.abs(points[:, 1] - cy) < half_diag
        in_z = np.abs(points[:, 2] - cz) < h / 2.0
        in_box = in_x & in_y & in_z
        mask &= ~in_box

    return points[mask]


def _apply_random_flip(
    points: np.ndarray, gt_boxes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random flip along X-axis and/or Y-axis.

    Args:
        points: (N, 5) point cloud.
        gt_boxes: (M, 9) boxes [x, y, z, l, w, h, yaw, vx, vy].

    Returns:
        Flipped points and boxes.
    """
    # Flip along X-axis (negate Y)
    if np.random.random() < 0.5:
        points[:, 1] = -points[:, 1]
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]  # negate yaw
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]  # negate vy

    # Flip along Y-axis (negate X)
    if np.random.random() < 0.5:
        points[:, 0] = -points[:, 0]
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)  # adjust yaw
        # Normalize yaw to [-pi, pi]
        gt_boxes[:, 6] = np.arctan2(
            np.sin(gt_boxes[:, 6]), np.cos(gt_boxes[:, 6])
        )
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]  # negate vx

    return points, gt_boxes


def _apply_global_rotation(
    points: np.ndarray, gt_boxes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random global rotation around the Z-axis.

    Rotation range: [-pi/4, pi/4] (from config: [-0.7854, 0.7854]).

    Args:
        points: (N, 5) point cloud.
        gt_boxes: (M, 9) boxes [x, y, z, l, w, h, yaw, vx, vy].

    Returns:
        Rotated points and boxes.
    """
    angle = np.random.uniform(-np.pi / 4, np.pi / 4)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Rotate XY coordinates of points
    x = points[:, 0]
    y = points[:, 1]
    points[:, 0] = x * cos_a - y * sin_a
    points[:, 1] = x * sin_a + y * cos_a

    # Rotate box centers
    bx = gt_boxes[:, 0]
    by = gt_boxes[:, 1]
    gt_boxes[:, 0] = bx * cos_a - by * sin_a
    gt_boxes[:, 1] = bx * sin_a + by * cos_a

    # Rotate yaw
    gt_boxes[:, 6] += angle

    # Rotate velocity
    if gt_boxes.shape[1] > 7:
        vx = gt_boxes[:, 7]
        vy = gt_boxes[:, 8]
        gt_boxes[:, 7] = vx * cos_a - vy * sin_a
        gt_boxes[:, 8] = vx * sin_a + vy * cos_a

    return points, gt_boxes


def _apply_global_scaling(
    points: np.ndarray, gt_boxes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random global scaling to point cloud and boxes.

    Scale range: [0.95, 1.05] (from config).

    Args:
        points: (N, 5) point cloud.
        gt_boxes: (M, 9) boxes [x, y, z, l, w, h, yaw, vx, vy].

    Returns:
        Scaled points and boxes.
    """
    scale = np.random.uniform(0.95, 1.05)

    # Scale point positions
    points[:, :3] *= scale

    # Scale box centers and dimensions
    gt_boxes[:, 0:3] *= scale  # center x, y, z
    gt_boxes[:, 3:6] *= scale  # l, w, h

    # Scale velocity
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] *= scale

    return points, gt_boxes


# =============================================================================
# Target Generation: Gaussian Heatmaps and Regression Targets
# =============================================================================


def generate_targets(
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    feature_map_size: int = BEV_RESOLUTION,
    voxel_size: List[float] = None,
    point_cloud_range: List[float] = None,
    gaussian_overlap: float = 0.1,
    min_radius: int = 2,
) -> List[Dict[str, np.ndarray]]:
    """Generate training targets for all task groups.

    Creates Gaussian heatmaps at object center locations on the BEV feature map,
    along with regression targets (offset, height, size, rotation, velocity)
    at positive center positions.

    Args:
        gt_boxes: (M, 9) ground truth boxes [x, y, z, l, w, h, yaw, vx, vy].
        gt_labels: (M,) integer class labels (0-indexed, corresponding to
            NUSCENES_CLASS_NAMES ordering).
        feature_map_size: Height and width of the BEV feature map (180).
        voxel_size: [vx, vy, vz] voxel dimensions.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        gaussian_overlap: Overlap parameter for Gaussian radius computation.
        min_radius: Minimum radius for the Gaussian.

    Returns:
        targets: List of dicts (one per task group), each containing:
            'heatmap': (H, W, num_classes) Gaussian heatmap targets.
            'offset': (H, W, 2) sub-pixel center offset.
            'height': (H, W, 1) object center height.
            'size': (H, W, 3) log-scale dimensions.
            'rotation': (H, W, 2) sin/cos of yaw.
            'velocity': (H, W, 2) velocity (vx, vy).
            'mask': (H, W, 1) binary mask for regression targets.
    """
    if voxel_size is None:
        voxel_size = VOXEL_SIZE
    if point_cloud_range is None:
        point_cloud_range = POINT_CLOUD_RANGE

    H = feature_map_size
    W = feature_map_size

    # Compute the effective pixel size on the feature map
    # Feature map pixel = voxel_size * FEATURE_MAP_STRIDE
    pixel_size_x = voxel_size[0] * FEATURE_MAP_STRIDE
    pixel_size_y = voxel_size[1] * FEATURE_MAP_STRIDE

    targets = []

    for task_idx, group_classes in enumerate(NUSCENES_TASK_GROUPS):
        num_classes = len(group_classes)

        heatmap = np.zeros((H, W, num_classes), dtype=np.float32)
        offset_target = np.zeros((H, W, 2), dtype=np.float32)
        height_target = np.zeros((H, W, 1), dtype=np.float32)
        size_target = np.zeros((H, W, 3), dtype=np.float32)
        rotation_target = np.zeros((H, W, 2), dtype=np.float32)
        velocity_target = np.zeros((H, W, 2), dtype=np.float32)
        mask = np.zeros((H, W, 1), dtype=np.float32)

        for box_idx in range(gt_boxes.shape[0]):
            label = int(gt_labels[box_idx])
            if label < 0 or label >= len(NUSCENES_CLASS_NAMES):
                continue

            cls_name = NUSCENES_CLASS_NAMES[label]
            if cls_name not in CLASS_TO_TASK_MAP:
                continue

            box_task_idx, class_in_group = CLASS_TO_TASK_MAP[cls_name]
            if box_task_idx != task_idx:
                continue

            # Get box parameters
            x, y, z = gt_boxes[box_idx, 0], gt_boxes[box_idx, 1], gt_boxes[box_idx, 2]
            l, w, h = gt_boxes[box_idx, 3], gt_boxes[box_idx, 4], gt_boxes[box_idx, 5]
            yaw = gt_boxes[box_idx, 6]
            vx, vy = gt_boxes[box_idx, 7], gt_boxes[box_idx, 8]

            # Convert world coordinates to feature map pixel coordinates
            feat_x = (x - point_cloud_range[0]) / pixel_size_x
            feat_y = (y - point_cloud_range[1]) / pixel_size_y

            # Check if center is within the feature map
            if feat_x < 0 or feat_x >= W or feat_y < 0 or feat_y >= H:
                continue

            # Integer pixel location
            cx_int = int(feat_x)
            cy_int = int(feat_y)
            cx_int = min(max(cx_int, 0), W - 1)
            cy_int = min(max(cy_int, 0), H - 1)

            # Sub-pixel offset
            offset_x = feat_x - cx_int
            offset_y = feat_y - cy_int

            # Compute Gaussian radius from box size on feature map
            box_w_pixels = w / pixel_size_x
            box_l_pixels = l / pixel_size_y
            radius = gaussian_radius(
                (box_l_pixels, box_w_pixels), min_overlap=gaussian_overlap
            )
            radius = max(min_radius, radius)

            # Draw Gaussian on heatmap
            generate_gaussian_target(
                heatmap[:, :, class_in_group],
                center=(cx_int, cy_int),
                radius=radius,
            )

            # Set regression targets at center pixel
            offset_target[cy_int, cx_int, 0] = offset_x
            offset_target[cy_int, cx_int, 1] = offset_y
            height_target[cy_int, cx_int, 0] = z
            # Size in log-scale for numerical stability
            size_target[cy_int, cx_int, 0] = np.log(max(l, 1e-4))
            size_target[cy_int, cx_int, 1] = np.log(max(w, 1e-4))
            size_target[cy_int, cx_int, 2] = np.log(max(h, 1e-4))
            rotation_target[cy_int, cx_int, 0] = np.sin(yaw)
            rotation_target[cy_int, cx_int, 1] = np.cos(yaw)
            velocity_target[cy_int, cx_int, 0] = vx
            velocity_target[cy_int, cx_int, 1] = vy
            mask[cy_int, cx_int, 0] = 1.0

        targets.append({
            "heatmap": heatmap,
            "offset": offset_target,
            "height": height_target,
            "size": size_target,
            "rotation": rotation_target,
            "velocity": velocity_target,
            "mask": mask,
        })

    return targets


# =============================================================================
# Dataset Pipeline
# =============================================================================


class CenterPointDataPipeline:
    """tf.data.Dataset pipeline for CenterPoint training.

    Handles loading .bin point cloud files, augmentation, voxelization,
    and target generation using tf.py_function for NumPy operations.
    """

    def __init__(
        self,
        data_path: str,
        ann_path: str,
        gt_db_path: Optional[str] = None,
        is_training: bool = True,
        max_points: int = 300000,
        point_channels: int = 5,
        augmentation: bool = True,
    ):
        """Initialize the data pipeline.

        Args:
            data_path: Root directory containing .bin point cloud files.
            ann_path: Path to annotation pickle file.
            gt_db_path: Path to GT database pickle for GT-sampling.
            is_training: Whether this is a training dataset.
            max_points: Maximum number of points to load per scene.
            point_channels: Number of channels per point (x,y,z,intensity,timestamp).
            augmentation: Whether to apply data augmentation.
        """
        self.data_path = data_path
        self.ann_path = ann_path
        self.is_training = is_training
        self.max_points = max_points
        self.point_channels = point_channels
        self.augmentation = augmentation and is_training

        # Load annotations
        self.annotations = self._load_annotations(ann_path)
        logger.info(f"Loaded {len(self.annotations)} annotations from {ann_path}")

        # Load GT database for GT-sampling
        self.gt_db = None
        if gt_db_path and os.path.exists(gt_db_path) and self.augmentation:
            with open(gt_db_path, "rb") as f:
                self.gt_db = pickle.load(f)
            logger.info(f"Loaded GT database from {gt_db_path}")

    def _load_annotations(self, ann_path: str) -> List[Dict[str, Any]]:
        """Load annotation file.

        Expected format: list of dicts with:
            'lidar_path': str - relative or absolute path to .bin file
            'gt_boxes': np.ndarray (M, 9) - [x,y,z,l,w,h,yaw,vx,vy]
            'gt_names': list of str - class names
            'gt_labels': np.ndarray (M,) - integer class indices
        """
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(ann_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "infos" in data:
            return data["infos"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected annotation format in {ann_path}")

    def _load_point_cloud(self, lidar_path: str) -> np.ndarray:
        """Load a point cloud from a .bin file.

        Args:
            lidar_path: Path to .bin file containing float32 point data.

        Returns:
            points: (N, point_channels) numpy array.
        """
        if not os.path.isabs(lidar_path):
            lidar_path = os.path.join(self.data_path, lidar_path)

        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape(-1, self.point_channels)

        # Limit number of points
        if points.shape[0] > self.max_points:
            indices = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[indices]

        return points

    def _process_sample(self, idx: int) -> Tuple:
        """Process a single sample: load, augment, generate targets.

        Args:
            idx: Index into the annotations list.

        Returns:
            Tuple of (points, target_heatmaps, target_offsets, target_heights,
                      target_sizes, target_rotations, target_velocities, target_masks)
            where each target is stacked across task groups.
        """
        ann = self.annotations[idx]

        # Load point cloud
        lidar_path = ann.get("lidar_path", ann.get("path", ""))
        points = self._load_point_cloud(lidar_path)

        # Load ground truth
        gt_boxes = np.array(ann.get("gt_boxes", np.zeros((0, 9))), dtype=np.float32)

        # Get labels: either from 'gt_labels' directly or compute from 'gt_names'
        if "gt_labels" in ann:
            gt_labels = np.array(ann["gt_labels"], dtype=np.int32)
        elif "gt_names" in ann:
            gt_names = ann["gt_names"]
            gt_labels = np.array(
                [NUSCENES_CLASS_NAMES.index(n) if n in NUSCENES_CLASS_NAMES else -1 for n in gt_names],
                dtype=np.int32,
            )
        else:
            gt_labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)

        # Filter invalid labels
        valid_mask = gt_labels >= 0
        gt_boxes = gt_boxes[valid_mask]
        gt_labels = gt_labels[valid_mask]

        # Apply augmentation
        if self.augmentation:
            points, gt_boxes, gt_labels = augment_point_cloud(
                points, gt_boxes, gt_labels,
                gt_db=self.gt_db,
                enable_gt_sampling=(self.gt_db is not None),
                enable_flip=True,
                enable_rotation=True,
                enable_scaling=True,
            )

        # Filter points within point cloud range
        pc_range = np.array(POINT_CLOUD_RANGE, dtype=np.float32)
        in_range = (
            (points[:, 0] >= pc_range[0])
            & (points[:, 0] <= pc_range[3])
            & (points[:, 1] >= pc_range[1])
            & (points[:, 1] <= pc_range[4])
            & (points[:, 2] >= pc_range[2])
            & (points[:, 2] <= pc_range[5])
        )
        points = points[in_range]

        # Pad or truncate points to fixed size for batching
        padded_points = np.zeros((self.max_points, self.point_channels), dtype=np.float32)
        num_valid = min(points.shape[0], self.max_points)
        padded_points[:num_valid] = points[:num_valid]

        # Generate targets
        targets = generate_targets(gt_boxes, gt_labels)

        # Stack targets across task groups
        # Each target: (num_tasks, H, W, C)
        heatmaps = np.stack([t["heatmap"] for t in targets], axis=0)
        offsets = np.stack([t["offset"] for t in targets], axis=0)
        heights = np.stack([t["height"] for t in targets], axis=0)
        sizes = np.stack([t["size"] for t in targets], axis=0)
        rotations = np.stack([t["rotation"] for t in targets], axis=0)
        velocities = np.stack([t["velocity"] for t in targets], axis=0)
        masks = np.stack([t["mask"] for t in targets], axis=0)

        return (
            padded_points.astype(np.float32),
            np.int32(num_valid),
            heatmaps.astype(np.float32),
            offsets.astype(np.float32),
            heights.astype(np.float32),
            sizes.astype(np.float32),
            rotations.astype(np.float32),
            velocities.astype(np.float32),
            masks.astype(np.float32),
        )

    def build_dataset(self, batch_size: int, num_replicas: int = 1) -> tf.data.Dataset:
        """Build a tf.data.Dataset for training or validation.

        Uses tf.py_function to wrap the NumPy processing pipeline.

        Args:
            batch_size: Global batch size.
            num_replicas: Number of GPU replicas.

        Returns:
            Batched and prefetched tf.data.Dataset.
        """
        num_samples = len(self.annotations)
        num_tasks = len(NUSCENES_TASK_GROUPS)
        H = BEV_RESOLUTION
        W = BEV_RESOLUTION

        # Determine number of heatmap classes per task
        task_num_classes = [len(g) for g in NUSCENES_TASK_GROUPS]
        max_hm_classes = max(task_num_classes)

        def py_process_sample(idx_tensor):
            """tf.py_function wrapper for sample processing."""
            idx = idx_tensor.numpy()
            result = self._process_sample(idx)
            return result

        # Output types and shapes for tf.py_function
        output_types = (
            tf.float32,  # points: (max_points, 5)
            tf.int32,    # num_valid_points: scalar
            tf.float32,  # heatmaps: (num_tasks, H, W, max_classes)
            tf.float32,  # offsets: (num_tasks, H, W, 2)
            tf.float32,  # heights: (num_tasks, H, W, 1)
            tf.float32,  # sizes: (num_tasks, H, W, 3)
            tf.float32,  # rotations: (num_tasks, H, W, 2)
            tf.float32,  # velocities: (num_tasks, H, W, 2)
            tf.float32,  # masks: (num_tasks, H, W, 1)
        )

        # Create index dataset
        indices = tf.data.Dataset.range(num_samples)

        if self.is_training:
            indices = indices.shuffle(buffer_size=min(4096, num_samples), reshuffle_each_iteration=True)

        # Map using py_function
        def map_fn(idx):
            results = tf.py_function(
                func=py_process_sample,
                inp=[idx],
                Tout=output_types,
            )
            # Set shapes explicitly
            results[0].set_shape([self.max_points, self.point_channels])
            results[1].set_shape([])
            results[2].set_shape([num_tasks, H, W, None])
            results[3].set_shape([num_tasks, H, W, 2])
            results[4].set_shape([num_tasks, H, W, 1])
            results[5].set_shape([num_tasks, H, W, 3])
            results[6].set_shape([num_tasks, H, W, 2])
            results[7].set_shape([num_tasks, H, W, 2])
            results[8].set_shape([num_tasks, H, W, 1])
            return results

        dataset = indices.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

        per_replica_batch = max(1, batch_size // num_replicas)
        dataset = dataset.batch(per_replica_batch, drop_remainder=self.is_training)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    @property
    def num_samples(self) -> int:
        """Return total number of samples in the dataset."""
        return len(self.annotations)


# =============================================================================
# Trainer Class
# =============================================================================


class CenterPointTrainer:
    """Manages the full CenterPoint training loop.

    Handles multi-GPU distribution, mixed precision, checkpoint management,
    and TensorBoard logging.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Create output directory
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup distribution strategy
        self._setup_strategy()

        # Setup mixed precision
        self._setup_mixed_precision()

        # Build model within strategy scope
        with self.strategy.scope():
            self.model = CenterPointModel(
                point_channels=args.point_channels,
                use_pillar_backbone=args.use_pillar_backbone,
                backbone_channels=(16, 32, 64, 128),
                bev_output_channels=256,
                head_channels=64,
                task_groups=NUSCENES_TASK_GROUPS,
            )
            # Optimizer will be created after dataset is loaded (need total_steps)
            self.optimizer = None

        # Training state
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.best_loss = float("inf")

        # Setup checkpoint
        self._setup_checkpoints()

        # Setup TensorBoard
        self._setup_tensorboard()

        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model backbone: {'Pillar' if args.use_pillar_backbone else 'Sparse3D'}")

    def _setup_strategy(self):
        """Configure MirroredStrategy for multi-GPU training."""
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"Could not set memory growth for {gpu}: {e}")

            if self.args.num_gpus > 0:
                device_list = [f"/gpu:{i}" for i in range(min(self.args.num_gpus, len(gpus)))]
                self.strategy = tf.distribute.MirroredStrategy(devices=device_list)
            else:
                self.strategy = tf.distribute.MirroredStrategy()

            logger.info(
                f"Using MirroredStrategy with {self.strategy.num_replicas_in_sync} GPUs"
            )
        else:
            logger.warning("No GPUs found. Using default strategy (CPU).")
            self.strategy = tf.distribute.get_strategy()

        self.num_replicas = self.strategy.num_replicas_in_sync

    def _setup_mixed_precision(self):
        """Configure mixed precision training."""
        if self.args.fp16:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled: mixed_float16")
        else:
            logger.info("Mixed precision disabled: using float32")

    def _setup_optimizer(self, total_steps: int):
        """Create Adam optimizer with OneCycle LR schedule.

        Args:
            total_steps: Total number of training steps for LR scheduling.
        """
        with self.strategy.scope():
            self.lr_schedule = WarmupCosineDecaySchedule(
                max_lr=self.args.lr,
                total_steps=total_steps,
                warmup_fraction=self.args.warmup_fraction,
                div_factor=self.args.div_factor,
                final_div_factor=self.args.final_div_factor,
            )

            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.lr_schedule,
                beta_1=0.9,
                beta_2=0.99,
                epsilon=1e-8,
                clipnorm=self.args.grad_clip_norm,
            )

            if self.args.fp16:
                self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                    self.optimizer, dynamic=True
                )

        logger.info(
            f"Optimizer: Adam(max_lr={self.args.lr}, warmup={self.args.warmup_fraction}, "
            f"grad_clip={self.args.grad_clip_norm})"
        )

    def _setup_checkpoints(self):
        """Setup checkpoint saving."""
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            global_step=self.global_step,
        )

        self.ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint,
            ckpt_dir,
            max_to_keep=self.args.max_checkpoints,
        )

        # Resume from checkpoint if specified
        if self.args.resume:
            status = self.checkpoint.restore(self.args.resume)
            status.expect_partial()
            logger.info(f"Resumed from checkpoint: {self.args.resume}")
            logger.info(f"  Global step: {self.global_step.numpy()}")
        elif self.ckpt_manager.latest_checkpoint:
            status = self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            status.expect_partial()
            logger.info(f"Restored latest checkpoint: {self.ckpt_manager.latest_checkpoint}")

    def _setup_tensorboard(self):
        """Setup TensorBoard summary writer."""
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = tf.summary.create_file_writer(log_dir)
        logger.info(f"TensorBoard logs: {log_dir}")

    @tf.function
    def _train_step(
        self,
        points: tf.Tensor,
        num_valid_points: tf.Tensor,
        target_heatmaps: tf.Tensor,
        target_offsets: tf.Tensor,
        target_heights: tf.Tensor,
        target_sizes: tf.Tensor,
        target_rotations: tf.Tensor,
        target_velocities: tf.Tensor,
        target_masks: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Execute a single training step with GradientTape.

        Computes the CenterPoint multi-task loss:
            total = heatmap_loss * 1.0 + offset_loss * 2.0 + height_loss * 2.0
                  + size_loss * 0.2 + rot_loss * 1.0 + vel_loss * 0.2

        Args:
            points: (B, max_points, 5) padded point clouds.
            num_valid_points: (B,) number of valid points per sample.
            target_heatmaps: (B, num_tasks, H, W, C) Gaussian heatmap targets.
            target_offsets: (B, num_tasks, H, W, 2) offset targets.
            target_heights: (B, num_tasks, H, W, 1) height targets.
            target_sizes: (B, num_tasks, H, W, 3) size targets.
            target_rotations: (B, num_tasks, H, W, 2) rotation targets.
            target_velocities: (B, num_tasks, H, W, 2) velocity targets.
            target_masks: (B, num_tasks, H, W, 1) regression masks.

        Returns:
            total_loss: Scalar total loss.
            loss_dict: Dictionary of individual loss components.
        """
        batch_size = tf.shape(points)[0]

        with tf.GradientTape() as tape:
            # Process each sample in the batch individually
            # (CenterPoint voxelization is per-sample due to variable point counts)
            total_heatmap_loss = tf.constant(0.0, dtype=tf.float32)
            total_offset_loss = tf.constant(0.0, dtype=tf.float32)
            total_height_loss = tf.constant(0.0, dtype=tf.float32)
            total_size_loss = tf.constant(0.0, dtype=tf.float32)
            total_rot_loss = tf.constant(0.0, dtype=tf.float32)
            total_vel_loss = tf.constant(0.0, dtype=tf.float32)

            for b in tf.range(batch_size):
                # Extract valid points for this sample
                n_pts = num_valid_points[b]
                sample_points = points[b, :n_pts, :]

                # Forward pass through model
                predictions = self.model(sample_points, training=True)

                # Compute losses for each task group
                for task_idx in range(len(NUSCENES_TASK_GROUPS)):
                    pred = predictions[task_idx]

                    # Get targets for this task
                    tgt_heatmap = target_heatmaps[b, task_idx]  # (H, W, C)
                    tgt_offset = target_offsets[b, task_idx]    # (H, W, 2)
                    tgt_height = target_heights[b, task_idx]    # (H, W, 1)
                    tgt_size = target_sizes[b, task_idx]        # (H, W, 3)
                    tgt_rot = target_rotations[b, task_idx]     # (H, W, 2)
                    tgt_vel = target_velocities[b, task_idx]    # (H, W, 2)
                    tgt_mask = target_masks[b, task_idx]        # (H, W, 1)

                    # Add batch dimension for loss computation
                    pred_hm = pred["heatmap"]       # (1, H, W, C)
                    pred_off = pred["offset"]       # (1, H, W, 2)
                    pred_h = pred["height"]         # (1, H, W, 1)
                    pred_s = pred["size"]           # (1, H, W, 3)
                    pred_r = pred["rotation"]       # (1, H, W, 2)
                    pred_v = pred["velocity"]       # (1, H, W, 2)

                    tgt_heatmap = tf.expand_dims(tgt_heatmap, 0)
                    tgt_offset = tf.expand_dims(tgt_offset, 0)
                    tgt_height = tf.expand_dims(tgt_height, 0)
                    tgt_size = tf.expand_dims(tgt_size, 0)
                    tgt_rot = tf.expand_dims(tgt_rot, 0)
                    tgt_vel = tf.expand_dims(tgt_vel, 0)
                    tgt_mask = tf.expand_dims(tgt_mask, 0)

                    # Gaussian focal loss for heatmap
                    hm_loss = gaussian_focal_loss(pred_hm, tgt_heatmap)
                    total_heatmap_loss = total_heatmap_loss + hm_loss

                    # Regression L1 losses at positive locations
                    off_loss = reg_l1_loss(pred_off, tgt_offset, tgt_mask)
                    total_offset_loss = total_offset_loss + off_loss

                    h_loss = reg_l1_loss(pred_h, tgt_height, tgt_mask)
                    total_height_loss = total_height_loss + h_loss

                    s_loss = reg_l1_loss(pred_s, tgt_size, tgt_mask)
                    total_size_loss = total_size_loss + s_loss

                    r_loss = reg_l1_loss(pred_r, tgt_rot, tgt_mask)
                    total_rot_loss = total_rot_loss + r_loss

                    v_loss = reg_l1_loss(pred_v, tgt_vel, tgt_mask)
                    total_vel_loss = total_vel_loss + v_loss

            # Normalize by batch size
            batch_size_f = tf.cast(batch_size, tf.float32)
            total_heatmap_loss = total_heatmap_loss / batch_size_f
            total_offset_loss = total_offset_loss / batch_size_f
            total_height_loss = total_height_loss / batch_size_f
            total_size_loss = total_size_loss / batch_size_f
            total_rot_loss = total_rot_loss / batch_size_f
            total_vel_loss = total_vel_loss / batch_size_f

            # Weighted total loss
            total_loss = (
                LOSS_WEIGHTS["heatmap"] * total_heatmap_loss
                + LOSS_WEIGHTS["offset"] * total_offset_loss
                + LOSS_WEIGHTS["height"] * total_height_loss
                + LOSS_WEIGHTS["size"] * total_size_loss
                + LOSS_WEIGHTS["rotation"] * total_rot_loss
                + LOSS_WEIGHTS["velocity"] * total_vel_loss
            )

            # Scale loss for mixed precision
            if self.args.fp16:
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)
            else:
                scaled_loss = total_loss

        # Compute and apply gradients
        trainable_vars = self.model.trainable_variables
        if self.args.fp16:
            scaled_grads = tape.gradient(scaled_loss, trainable_vars)
            gradients = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            gradients = tape.gradient(total_loss, trainable_vars)

        # Clip gradients by global norm
        gradients, _ = tf.clip_by_global_norm(gradients, self.args.grad_clip_norm)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_dict = {
            "heatmap_loss": total_heatmap_loss,
            "offset_loss": total_offset_loss,
            "height_loss": total_height_loss,
            "size_loss": total_size_loss,
            "rotation_loss": total_rot_loss,
            "velocity_loss": total_vel_loss,
            "total_loss": total_loss,
        }

        return total_loss, loss_dict

    def train(self):
        """Execute the full training loop with epoch and step logging."""
        # Build datasets
        logger.info("Building training dataset...")
        train_pipeline = CenterPointDataPipeline(
            data_path=self.args.data_path,
            ann_path=self.args.ann_path,
            gt_db_path=self.args.gt_db_path,
            is_training=True,
            max_points=self.args.max_points,
            point_channels=self.args.point_channels,
            augmentation=True,
        )
        train_dataset = train_pipeline.build_dataset(
            batch_size=self.args.batch_size,
            num_replicas=self.num_replicas,
        )

        # Build validation dataset if path provided
        val_dataset = None
        if self.args.val_ann_path and os.path.exists(self.args.val_ann_path):
            logger.info("Building validation dataset...")
            val_pipeline = CenterPointDataPipeline(
                data_path=self.args.data_path,
                ann_path=self.args.val_ann_path,
                gt_db_path=None,
                is_training=False,
                max_points=self.args.max_points,
                point_channels=self.args.point_channels,
                augmentation=False,
            )
            val_dataset = val_pipeline.build_dataset(
                batch_size=self.args.batch_size,
                num_replicas=self.num_replicas,
            )

        # Calculate total steps and create optimizer
        num_samples = train_pipeline.num_samples
        per_replica_batch = max(1, self.args.batch_size // self.num_replicas)
        steps_per_epoch = max(1, num_samples // self.args.batch_size)
        total_steps = steps_per_epoch * self.args.epochs

        self._setup_optimizer(total_steps)

        # Add optimizer to checkpoint after creation
        self.checkpoint.optimizer = self.optimizer
        if self.ckpt_manager.latest_checkpoint:
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

        # Log training configuration
        logger.info("=" * 70)
        logger.info("CenterPoint Training Configuration")
        logger.info("=" * 70)
        logger.info(f"  Dataset samples: {num_samples}")
        logger.info(f"  Epochs: {self.args.epochs}")
        logger.info(f"  Batch size (global): {self.args.batch_size}")
        logger.info(f"  Batch size (per GPU): {per_replica_batch}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Max learning rate: {self.args.lr}")
        logger.info(f"  Warmup fraction: {self.args.warmup_fraction}")
        logger.info(f"  Gradient clip norm: {self.args.grad_clip_norm}")
        logger.info(f"  Mixed precision (FP16): {self.args.fp16}")
        logger.info(f"  Num GPUs: {self.num_replicas}")
        logger.info(f"  Voxel size: {VOXEL_SIZE}")
        logger.info(f"  Point cloud range: {POINT_CLOUD_RANGE}")
        logger.info(f"  Grid size: {GRID_SIZE}")
        logger.info(f"  BEV resolution: {BEV_RESOLUTION} x {BEV_RESOLUTION}")
        logger.info(f"  Task groups: {len(NUSCENES_TASK_GROUPS)}")
        logger.info(f"  Loss weights: {LOSS_WEIGHTS}")
        logger.info("=" * 70)

        # Distribute dataset
        if self.num_replicas > 1:
            dist_train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        else:
            dist_train_dataset = train_dataset

        # Training loop
        start_epoch = int(self.global_step.numpy()) // steps_per_epoch

        for epoch in range(start_epoch, self.args.epochs):
            epoch_start_time = time.time()
            epoch_losses = {
                "total_loss": 0.0,
                "heatmap_loss": 0.0,
                "offset_loss": 0.0,
                "height_loss": 0.0,
                "size_loss": 0.0,
                "rotation_loss": 0.0,
                "velocity_loss": 0.0,
            }
            epoch_steps = 0

            dataset_iter = iter(dist_train_dataset)

            for step_in_epoch in range(steps_per_epoch):
                step_start_time = time.time()

                try:
                    batch = next(dataset_iter)
                except StopIteration:
                    break

                # Unpack batch
                (
                    batch_points,
                    batch_num_valid,
                    batch_heatmaps,
                    batch_offsets,
                    batch_heights,
                    batch_sizes,
                    batch_rotations,
                    batch_velocities,
                    batch_masks,
                ) = batch

                # Execute training step
                loss, loss_dict = self._train_step(
                    batch_points,
                    batch_num_valid,
                    batch_heatmaps,
                    batch_offsets,
                    batch_heights,
                    batch_sizes,
                    batch_rotations,
                    batch_velocities,
                    batch_masks,
                )

                # Update global step
                self.global_step.assign_add(1)
                current_step = int(self.global_step.numpy())

                # Accumulate epoch metrics
                epoch_losses["total_loss"] += float(loss)
                for key in loss_dict:
                    if key in epoch_losses:
                        epoch_losses[key] += float(loss_dict[key])
                epoch_steps += 1

                # Get current learning rate
                current_lr = float(self.lr_schedule(tf.cast(current_step, tf.float32)))

                # Log to TensorBoard
                with self.tb_writer.as_default(step=current_step):
                    tf.summary.scalar("train/total_loss", loss)
                    tf.summary.scalar("train/learning_rate", current_lr)
                    for key, value in loss_dict.items():
                        if key != "total_loss":
                            tf.summary.scalar(f"train/{key}", value)

                # Step-level logging
                step_time = time.time() - step_start_time
                if current_step % self.args.log_interval == 0 or step_in_epoch == 0:
                    loss_str = (
                        f"total={float(loss):.4f} "
                        f"hm={float(loss_dict['heatmap_loss']):.4f} "
                        f"off={float(loss_dict['offset_loss']):.4f} "
                        f"ht={float(loss_dict['height_loss']):.4f} "
                        f"sz={float(loss_dict['size_loss']):.4f} "
                        f"rot={float(loss_dict['rotation_loss']):.4f} "
                        f"vel={float(loss_dict['velocity_loss']):.4f}"
                    )
                    logger.info(
                        f"Epoch [{epoch + 1}/{self.args.epochs}] "
                        f"Step [{step_in_epoch + 1}/{steps_per_epoch}] "
                        f"Global [{current_step}/{total_steps}] "
                        f"{loss_str} "
                        f"lr={current_lr:.2e} "
                        f"time={step_time:.2f}s"
                    )

            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_losses = {k: v / max(epoch_steps, 1) for k, v in epoch_losses.items()}

            logger.info(
                f"Epoch {epoch + 1}/{self.args.epochs} completed in {epoch_time:.1f}s. "
                f"Avg loss: {avg_losses['total_loss']:.4f} "
                f"(hm={avg_losses['heatmap_loss']:.4f} "
                f"off={avg_losses['offset_loss']:.4f} "
                f"ht={avg_losses['height_loss']:.4f} "
                f"sz={avg_losses['size_loss']:.4f} "
                f"rot={avg_losses['rotation_loss']:.4f} "
                f"vel={avg_losses['velocity_loss']:.4f})"
            )

            # Log epoch-level metrics to TensorBoard
            with self.tb_writer.as_default(step=epoch + 1):
                for key, value in avg_losses.items():
                    tf.summary.scalar(f"epoch/{key}", value)
                tf.summary.scalar("epoch/time_seconds", epoch_time)

            # Run validation
            if val_dataset is not None:
                val_loss = self._validate(val_dataset, epoch)
                if val_loss is not None:
                    with self.tb_writer.as_default(step=epoch + 1):
                        tf.summary.scalar("val/total_loss", val_loss)

                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        best_path = os.path.join(
                            self.output_dir, "checkpoints", "best_model"
                        )
                        self.checkpoint.save(file_prefix=best_path)
                        logger.info(
                            f"New best validation loss: {val_loss:.4f}. Saved best model."
                        )

            # Periodic checkpoint save
            if (epoch + 1) % self.args.save_interval == 0:
                save_path = self.ckpt_manager.save()
                logger.info(f"Checkpoint saved: {save_path}")

        # Final checkpoint
        final_path = self.ckpt_manager.save()
        logger.info(f"Training complete. Final checkpoint: {final_path}")
        logger.info(f"Best validation loss: {self.best_loss:.4f}")

        self.tb_writer.close()

    def _validate(
        self, val_dataset: tf.data.Dataset, epoch: int
    ) -> Optional[float]:
        """Run validation and compute average loss.

        Args:
            val_dataset: Validation tf.data.Dataset.
            epoch: Current epoch number.

        Returns:
            Average validation loss, or None if validation fails.
        """
        logger.info("Running validation...")
        val_losses = []
        max_val_steps = self.args.max_val_steps

        for step_idx, batch in enumerate(val_dataset):
            if step_idx >= max_val_steps:
                break

            (
                batch_points,
                batch_num_valid,
                batch_heatmaps,
                batch_offsets,
                batch_heights,
                batch_sizes,
                batch_rotations,
                batch_velocities,
                batch_masks,
            ) = batch

            batch_size = tf.shape(batch_points)[0]

            # Forward pass without gradients
            batch_loss = tf.constant(0.0, dtype=tf.float32)
            for b in range(batch_size):
                n_pts = batch_num_valid[b]
                sample_points = batch_points[b, :n_pts, :]

                predictions = self.model(sample_points, training=False)

                for task_idx in range(len(NUSCENES_TASK_GROUPS)):
                    pred = predictions[task_idx]
                    tgt_heatmap = tf.expand_dims(batch_heatmaps[b, task_idx], 0)
                    tgt_offset = tf.expand_dims(batch_offsets[b, task_idx], 0)
                    tgt_height = tf.expand_dims(batch_heights[b, task_idx], 0)
                    tgt_size = tf.expand_dims(batch_sizes[b, task_idx], 0)
                    tgt_rot = tf.expand_dims(batch_rotations[b, task_idx], 0)
                    tgt_vel = tf.expand_dims(batch_velocities[b, task_idx], 0)
                    tgt_mask = tf.expand_dims(batch_masks[b, task_idx], 0)

                    hm_loss = gaussian_focal_loss(pred["heatmap"], tgt_heatmap)
                    off_loss = reg_l1_loss(pred["offset"], tgt_offset, tgt_mask)
                    h_loss = reg_l1_loss(pred["height"], tgt_height, tgt_mask)
                    s_loss = reg_l1_loss(pred["size"], tgt_size, tgt_mask)
                    r_loss = reg_l1_loss(pred["rotation"], tgt_rot, tgt_mask)
                    v_loss = reg_l1_loss(pred["velocity"], tgt_vel, tgt_mask)

                    task_loss = (
                        LOSS_WEIGHTS["heatmap"] * hm_loss
                        + LOSS_WEIGHTS["offset"] * off_loss
                        + LOSS_WEIGHTS["height"] * h_loss
                        + LOSS_WEIGHTS["size"] * s_loss
                        + LOSS_WEIGHTS["rotation"] * r_loss
                        + LOSS_WEIGHTS["velocity"] * v_loss
                    )
                    batch_loss = batch_loss + task_loss

            batch_loss = batch_loss / tf.cast(batch_size, tf.float32)
            val_losses.append(float(batch_loss))

        if val_losses:
            avg_val_loss = float(np.mean(val_losses))
            logger.info(
                f"Validation: avg_loss={avg_val_loss:.4f} ({len(val_losses)} steps)"
            )
            return avg_val_loss

        return None


# =============================================================================
# Command-Line Arguments
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="CenterPoint 3D LiDAR Object Detection Training (TensorFlow 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root directory containing .bin point cloud files.",
    )
    parser.add_argument(
        "--ann_path",
        type=str,
        required=True,
        help="Path to training annotation pickle file.",
    )
    parser.add_argument(
        "--val_ann_path",
        type=str,
        default=None,
        help="Path to validation annotation pickle file.",
    )
    parser.add_argument(
        "--gt_db_path",
        type=str,
        default=None,
        help="Path to GT database pickle file for GT-sampling augmentation.",
    )

    # Model configuration
    parser.add_argument(
        "--point_channels",
        type=int,
        default=5,
        help="Number of input point channels (x, y, z, intensity, timestamp).",
    )
    parser.add_argument(
        "--use_pillar_backbone",
        action="store_true",
        help="Use PillarFeatureNet instead of Sparse3D backbone.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Global batch size (distributed across GPUs).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Maximum learning rate (peak of OneCycle schedule).",
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.4,
        help="Fraction of total steps for linear warmup (OneCycle pct_start).",
    )
    parser.add_argument(
        "--div_factor",
        type=float,
        default=10.0,
        help="Initial lr = max_lr / div_factor.",
    )
    parser.add_argument(
        "--final_div_factor",
        type=float,
        default=100.0,
        help="Final lr = max_lr / final_div_factor.",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=35.0,
        help="Maximum gradient norm for clipping.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=300000,
        help="Maximum number of points per scene.",
    )

    # Infrastructure
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of GPUs to use (0 = all available).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training (FP16).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./work_dirs/centerpoint_tf",
        help="Output directory for checkpoints, logs, and configs.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )

    # Logging and saving
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=200,
        help="Maximum number of validation steps per epoch.",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.lr <= 0:
        parser.error("--lr must be > 0")

    return args


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main training entry point."""
    args = parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Log system information
    logger.info("=" * 70)
    logger.info("CenterPoint 3D Object Detection Training (TensorFlow 2)")
    logger.info("=" * 70)
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    for i, gpu in enumerate(tf.config.list_physical_devices("GPU")):
        logger.info(f"  GPU {i}: {gpu.name}")

    logger.info(f"Arguments: {vars(args)}")

    # Create trainer and run training
    try:
        trainer = CenterPointTrainer(args)
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
