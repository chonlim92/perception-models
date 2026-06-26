"""PointPillars TF2 Training Script.

Production-quality training pipeline for PointPillars 3D object detection
on LiDAR point clouds. Supports multi-GPU training via MirroredStrategy,
custom training loop with focal loss, smooth L1, and direction classification loss.

Reference: Lang et al., "PointPillars: Fast Encoders for Object Detection
from Point Clouds", CVPR 2019.

Usage:
    python train.py --config configs/pointpillars_car.yaml
    python train.py --config configs/pointpillars_car.yaml --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AnchorConfig:
    """Anchor generation parameters for a single class."""

    class_name: str = "Car"
    anchor_sizes: List[List[float]] = field(
        default_factory=lambda: [[3.9, 1.6, 1.56]]
    )
    anchor_rotations: List[float] = field(default_factory=lambda: [0.0, 1.5707963])
    anchor_z_center: float = -1.0
    matched_threshold: float = 0.6
    unmatched_threshold: float = 0.45


@dataclass
class VoxelConfig:
    """Voxelization parameters."""

    point_cloud_range: List[float] = field(
        default_factory=lambda: [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
    )
    voxel_size: List[float] = field(default_factory=lambda: [0.16, 0.16, 4.0])
    max_points_per_voxel: int = 32
    max_voxels: int = 16000


@dataclass
class AugmentationConfig:
    """Data augmentation parameters."""

    random_flip_x: bool = True
    random_flip_y: bool = True
    rotation_range: List[float] = field(
        default_factory=lambda: [-0.78539816, 0.78539816]
    )
    scaling_range: List[float] = field(default_factory=lambda: [0.95, 1.05])
    gt_database_path: Optional[str] = None
    gt_database_max_samples: Dict[str, int] = field(
        default_factory=lambda: {"Car": 15, "Pedestrian": 10, "Cyclist": 10}
    )


@dataclass
class TrainConfig:
    """Full training configuration."""

    # Model
    num_classes: int = 3
    pillar_features: int = 64
    backbone_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    backbone_strides: List[int] = field(default_factory=lambda: [2, 2, 2])
    backbone_num_blocks: List[int] = field(default_factory=lambda: [3, 5, 5])

    # Training
    batch_size: int = 4
    epochs: int = 160
    learning_rate: float = 2.25e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    gradient_clip_norm: float = 10.0
    num_workers: int = 4

    # Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    cls_weight: float = 1.0
    reg_weight: float = 2.0
    dir_weight: float = 0.2
    pos_neg_ratio: float = 3.0

    # Data
    data_root: str = "/data/kitti"
    train_info_path: str = "kitti_infos_train.pkl"
    val_info_path: str = "kitti_infos_val.pkl"
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    anchors: List[AnchorConfig] = field(
        default_factory=lambda: [AnchorConfig()]
    )

    # Checkpointing
    output_dir: str = "./output"
    save_every_n_epochs: int = 5
    keep_max_checkpoints: int = 5

    # Logging
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 5


def load_config(yaml_path: str) -> TrainConfig:
    """Load training configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Populated TrainConfig dataclass.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    config = TrainConfig()

    # Flat fields
    for key in [
        "num_classes", "pillar_features", "backbone_channels",
        "backbone_strides", "backbone_num_blocks", "batch_size", "epochs",
        "learning_rate", "weight_decay", "warmup_epochs", "gradient_clip_norm",
        "num_workers", "focal_alpha", "focal_gamma", "cls_weight", "reg_weight",
        "dir_weight", "pos_neg_ratio", "data_root", "train_info_path",
        "val_info_path", "output_dir", "save_every_n_epochs",
        "keep_max_checkpoints", "log_every_n_steps", "eval_every_n_epochs",
    ]:
        if key in raw:
            setattr(config, key, raw[key])

    # Nested configs
    if "voxel" in raw:
        config.voxel = VoxelConfig(**raw["voxel"])

    if "augmentation" in raw:
        config.augmentation = AugmentationConfig(**raw["augmentation"])

    if "anchors" in raw:
        config.anchors = [AnchorConfig(**a) for a in raw["anchors"]]

    return config


# =============================================================================
# Learning Rate Schedule
# =============================================================================


class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """One-cycle learning rate policy with linear warmup and cosine annealing.

    Implements Smith's 1cycle policy: linear warmup from initial_lr to max_lr,
    then cosine decay to min_lr.

    Args:
        max_lr: Peak learning rate.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps (linear ramp).
        min_lr_factor: Factor of max_lr for the final learning rate.
    """

    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr_factor: float = 0.001,
    ) -> None:
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = max_lr * min_lr_factor
        self.initial_lr = max_lr * 0.1

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Linear warmup phase
        warmup_lr = self.initial_lr + (self.max_lr - self.initial_lr) * (
            step / tf.maximum(warmup_steps, 1.0)
        )

        # Cosine annealing phase
        decay_steps = total_steps - warmup_steps
        progress = (step - warmup_steps) / tf.maximum(decay_steps, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * progress))
        decay_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self) -> Dict[str, Any]:
        return {
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "initial_lr": self.initial_lr,
        }


# =============================================================================
# Loss Functions
# =============================================================================


def focal_loss(
    pred_cls: tf.Tensor,
    target_cls: tf.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> tf.Tensor:
    """Compute focal loss for classification.

    Focal loss addresses class imbalance by down-weighting easy examples
    and focusing training on hard negatives.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        pred_cls: Predicted class logits, shape [N, num_classes].
        target_cls: One-hot encoded targets, shape [N, num_classes].
        alpha: Balancing factor for positive/negative examples.
        gamma: Focusing parameter (higher = more focus on hard examples).

    Returns:
        Scalar focal loss averaged over positive samples.
    """
    pred_sigmoid = tf.sigmoid(pred_cls)
    # Numerically stable binary cross entropy per element
    bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_cls, logits=pred_cls
    )

    # p_t: probability of correct class
    p_t = target_cls * pred_sigmoid + (1.0 - target_cls) * (1.0 - pred_sigmoid)

    # Alpha weighting
    alpha_factor = target_cls * alpha + (1.0 - target_cls) * (1.0 - alpha)

    # Focal modulating factor
    modulating_factor = tf.pow(1.0 - p_t, gamma)

    loss = alpha_factor * modulating_factor * bce
    return tf.reduce_sum(loss)


def smooth_l1_loss(
    pred_reg: tf.Tensor,
    target_reg: tf.Tensor,
    sigma: float = 3.0,
) -> tf.Tensor:
    """Compute smooth L1 (Huber) loss for box regression.

    Args:
        pred_reg: Predicted regression values, shape [N, 7].
            Encodes (dx, dy, dz, dw, dl, dh, dtheta).
        target_reg: Target regression values, shape [N, 7].
        sigma: Transition point between L1 and L2 regimes.

    Returns:
        Scalar smooth L1 loss summed over all elements.
    """
    sigma_squared = sigma ** 2
    diff = pred_reg - target_reg
    abs_diff = tf.abs(diff)

    # Smooth L1: quadratic for |x| < 1/sigma^2, linear otherwise
    smooth_flag = tf.cast(abs_diff < (1.0 / sigma_squared), tf.float32)
    loss = smooth_flag * 0.5 * sigma_squared * tf.square(diff) + (
        1.0 - smooth_flag
    ) * (abs_diff - 0.5 / sigma_squared)

    return tf.reduce_sum(loss)


def direction_classification_loss(
    pred_dir: tf.Tensor,
    target_dir: tf.Tensor,
) -> tf.Tensor:
    """Compute binary cross-entropy loss for direction classification.

    Discretizes rotation into two bins (forward/backward) to resolve
    the heading ambiguity in box regression.

    Args:
        pred_dir: Predicted direction logits, shape [N, 2].
        target_dir: One-hot direction targets, shape [N, 2].

    Returns:
        Scalar direction loss summed over positive anchors.
    """
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_dir, logits=pred_dir
    )
    return tf.reduce_sum(loss)


# =============================================================================
# Anchor Generation and Target Assignment
# =============================================================================


class AnchorGenerator:
    """Generate 3D anchors for PointPillars detection head.

    Creates a grid of 3D anchor boxes at each spatial location of the
    bird's-eye-view feature map.

    Args:
        config: Training configuration containing anchor and voxel parameters.
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.anchors = self._generate_anchors()

    def _generate_anchors(self) -> np.ndarray:
        """Generate all anchors across the BEV feature map.

        Returns:
            Anchor array of shape [H, W, num_anchors_per_loc, 7]
            where each anchor is (x, y, z, w, l, h, rotation).
        """
        pc_range = self.config.voxel.point_cloud_range
        voxel_size = self.config.voxel.voxel_size

        # BEV grid dimensions (after backbone downsampling by factor 2)
        feature_stride = 2
        grid_x = int(
            (pc_range[3] - pc_range[0]) / (voxel_size[0] * feature_stride)
        )
        grid_y = int(
            (pc_range[4] - pc_range[1]) / (voxel_size[1] * feature_stride)
        )

        # Grid cell centers
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

        # Meshgrid for spatial locations
        xx, yy = np.meshgrid(x_centers, y_centers)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        num_locations = len(xx)

        all_anchors = []
        for anchor_cfg in self.config.anchors:
            for size in anchor_cfg.anchor_sizes:
                for rotation in anchor_cfg.anchor_rotations:
                    w, l, h = size
                    z = anchor_cfg.anchor_z_center
                    anchors = np.stack(
                        [
                            xx,
                            yy,
                            np.full(num_locations, z),
                            np.full(num_locations, w),
                            np.full(num_locations, l),
                            np.full(num_locations, h),
                            np.full(num_locations, rotation),
                        ],
                        axis=-1,
                    )
                    all_anchors.append(anchors)

        # Shape: [num_locations * num_anchor_types, 7]
        all_anchors = np.concatenate(all_anchors, axis=0).astype(np.float32)
        return all_anchors

    def get_anchors(self) -> np.ndarray:
        """Return pre-computed anchors.

        Returns:
            Array of shape [total_anchors, 7].
        """
        return self.anchors


def compute_iou_bev(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Compute axis-aligned BEV IoU between two sets of boxes.

    For efficiency, uses axis-aligned bounding box approximation
    (ignores rotation for IoU computation during anchor matching).

    Args:
        boxes_a: Array of shape [M, 7] (x, y, z, w, l, h, r).
        boxes_b: Array of shape [N, 7] (x, y, z, w, l, h, r).

    Returns:
        IoU matrix of shape [M, N].
    """
    # Extract BEV corners (x, y, w, l) ignoring rotation for AABB approx
    ax1 = boxes_a[:, 0] - boxes_a[:, 3] / 2.0
    ay1 = boxes_a[:, 1] - boxes_a[:, 4] / 2.0
    ax2 = boxes_a[:, 0] + boxes_a[:, 3] / 2.0
    ay2 = boxes_a[:, 1] + boxes_a[:, 4] / 2.0

    bx1 = boxes_b[:, 0] - boxes_b[:, 3] / 2.0
    by1 = boxes_b[:, 1] - boxes_b[:, 4] / 2.0
    bx2 = boxes_b[:, 0] + boxes_b[:, 3] / 2.0
    by2 = boxes_b[:, 1] + boxes_b[:, 4] / 2.0

    # Pairwise intersection
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    iou = inter_area / np.maximum(union_area, 1e-7)
    return iou


def encode_box_targets(
    anchors: np.ndarray, gt_boxes: np.ndarray
) -> np.ndarray:
    """Encode ground truth boxes as regression targets relative to anchors.

    Encoding: dx = (gt_x - a_x) / a_diag, dy = (gt_y - a_y) / a_diag,
    dz = (gt_z - a_z) / a_h, dw = log(gt_w / a_w), dl = log(gt_l / a_l),
    dh = log(gt_h / a_h), dr = gt_r - a_r.

    Args:
        anchors: Anchor boxes, shape [N, 7].
        gt_boxes: Matched GT boxes, shape [N, 7].

    Returns:
        Encoded targets, shape [N, 7].
    """
    anchor_diag = np.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    dx = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_diag
    dy = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_diag
    dz = (gt_boxes[:, 2] - anchors[:, 2]) / anchors[:, 5]
    dw = np.log(gt_boxes[:, 3] / np.maximum(anchors[:, 3], 1e-7))
    dl = np.log(gt_boxes[:, 4] / np.maximum(anchors[:, 4], 1e-7))
    dh = np.log(gt_boxes[:, 5] / np.maximum(anchors[:, 5], 1e-7))
    dr = gt_boxes[:, 6] - anchors[:, 6]

    return np.stack([dx, dy, dz, dw, dl, dh, dr], axis=-1).astype(np.float32)


def assign_anchor_targets(
    anchors: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    config: TrainConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign ground truth targets to anchors using IoU-based matching.

    Each anchor is assigned:
    - Positive if max IoU with any GT >= matched_threshold
    - Negative if max IoU with all GT < unmatched_threshold
    - Ignored (don't care) otherwise

    Args:
        anchors: All anchors, shape [num_anchors, 7].
        gt_boxes: Ground truth boxes, shape [num_gt, 7].
        gt_classes: Ground truth class labels, shape [num_gt].
        config: Training config with anchor thresholds.

    Returns:
        Tuple of:
            cls_targets: One-hot classification targets [num_anchors, num_classes].
            reg_targets: Regression targets [num_anchors, 7].
            dir_targets: Direction classification targets [num_anchors, 2].
            pos_mask: Boolean mask for positive anchors [num_anchors].
            neg_mask: Boolean mask for negative anchors [num_anchors].
    """
    num_anchors = anchors.shape[0]
    num_classes = config.num_classes

    cls_targets = np.zeros((num_anchors, num_classes), dtype=np.float32)
    reg_targets = np.zeros((num_anchors, 7), dtype=np.float32)
    dir_targets = np.zeros((num_anchors, 2), dtype=np.float32)
    pos_mask = np.zeros(num_anchors, dtype=np.bool_)
    neg_mask = np.zeros(num_anchors, dtype=np.bool_)

    if gt_boxes.shape[0] == 0:
        # No ground truth - all anchors are negative
        neg_mask[:] = True
        return cls_targets, reg_targets, dir_targets, pos_mask, neg_mask

    # Compute IoU between anchors and GT boxes
    iou_matrix = compute_iou_bev(anchors, gt_boxes)

    # For each anchor, find best matching GT
    max_iou_per_anchor = iou_matrix.max(axis=1)
    best_gt_per_anchor = iou_matrix.argmax(axis=1)

    # For each GT, ensure at least one anchor is matched (highest IoU)
    best_anchor_per_gt = iou_matrix.argmax(axis=0)

    # Apply thresholds using the first anchor config (primary class)
    matched_thresh = config.anchors[0].matched_threshold
    unmatched_thresh = config.anchors[0].unmatched_threshold

    # Positive anchors: IoU >= matched_threshold
    pos_mask = max_iou_per_anchor >= matched_thresh

    # Force-match best anchor for each GT to be positive
    for gt_idx in range(gt_boxes.shape[0]):
        pos_mask[best_anchor_per_gt[gt_idx]] = True

    # Negative anchors: IoU < unmatched_threshold
    neg_mask = max_iou_per_anchor < unmatched_thresh
    neg_mask[pos_mask] = False  # Positive overrides negative

    # Assign targets for positive anchors
    pos_indices = np.where(pos_mask)[0]
    matched_gt_indices = best_gt_per_anchor[pos_indices]
    matched_gt_boxes = gt_boxes[matched_gt_indices]
    matched_gt_classes = gt_classes[matched_gt_indices]

    # Classification targets (one-hot)
    for i, (pos_idx, cls_id) in enumerate(zip(pos_indices, matched_gt_classes)):
        cls_targets[pos_idx, int(cls_id)] = 1.0

    # Regression targets
    reg_targets[pos_indices] = encode_box_targets(
        anchors[pos_indices], matched_gt_boxes
    )

    # Direction targets: bin the rotation angle into [0, pi) and [pi, 2pi)
    gt_rotations = matched_gt_boxes[:, 6]
    # Normalize to [0, 2*pi)
    gt_rotations_norm = gt_rotations % (2.0 * np.pi)
    dir_bins = (gt_rotations_norm > np.pi).astype(np.int32)
    dir_targets[pos_indices, 0] = 1.0 - dir_bins.astype(np.float32)
    dir_targets[pos_indices, 1] = dir_bins.astype(np.float32)

    return cls_targets, reg_targets, dir_targets, pos_mask, neg_mask


# =============================================================================
# Data Pipeline
# =============================================================================


class GTDatabase:
    """Ground truth database for copy-paste augmentation.

    Stores pre-computed point cloud segments of individual objects from
    training set, enabling GT-sampling augmentation during training.

    Args:
        db_path: Path to the GT database directory containing .npy files.
        max_samples: Dictionary mapping class name to max samples per scene.
    """

    def __init__(
        self, db_path: str, max_samples: Dict[str, int]
    ) -> None:
        self.db_path = Path(db_path)
        self.max_samples = max_samples
        self.database: Dict[str, List[Dict[str, Any]]] = {}
        self._load_database()

    def _load_database(self) -> None:
        """Load the GT database index from disk."""
        db_info_path = self.db_path / "gt_database_info.pkl"
        if db_info_path.exists():
            import pickle
            with open(db_info_path, "rb") as f:
                self.database = pickle.load(f)
            logger.info(
                "Loaded GT database with classes: %s",
                {k: len(v) for k, v in self.database.items()},
            )
        else:
            logger.warning(
                "GT database info not found at %s. "
                "GT sampling augmentation will be disabled.",
                db_info_path,
            )

    def sample(
        self, existing_boxes: np.ndarray, existing_classes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample objects from GT database avoiding collisions.

        Args:
            existing_boxes: Current scene GT boxes, shape [N, 7].
            existing_classes: Current scene GT classes, shape [N].

        Returns:
            Tuple of (sampled_points, sampled_boxes, sampled_classes).
            sampled_points: shape [M, 4] (x, y, z, intensity).
            sampled_boxes: shape [K, 7].
            sampled_classes: shape [K].
        """
        sampled_points_all = []
        sampled_boxes_all = []
        sampled_classes_all = []

        for class_name, max_count in self.max_samples.items():
            if class_name not in self.database or len(self.database[class_name]) == 0:
                continue

            # How many of this class already exist
            current_count = np.sum(existing_classes == class_name) if len(existing_classes) > 0 else 0
            num_to_sample = max(0, max_count - int(current_count))
            if num_to_sample == 0:
                continue

            # Random sample from database
            db_entries = self.database[class_name]
            indices = np.random.choice(
                len(db_entries), size=min(num_to_sample, len(db_entries)), replace=False
            )

            for idx in indices:
                entry = db_entries[idx]
                box = entry["box3d"].copy()

                # Collision check with existing boxes using BEV IoU
                if existing_boxes.shape[0] > 0:
                    iou = compute_iou_bev(
                        box.reshape(1, -1), existing_boxes
                    )
                    if iou.max() > 0.05:
                        continue

                # Load points for this object
                points_path = self.db_path / entry["path"]
                if not points_path.exists():
                    continue
                points = np.load(str(points_path))

                sampled_points_all.append(points)
                sampled_boxes_all.append(box)
                sampled_classes_all.append(entry["class_id"])

                # Update existing boxes for subsequent collision checks
                existing_boxes = np.vstack([existing_boxes, box.reshape(1, -1)])

        if len(sampled_boxes_all) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        return (
            np.concatenate(sampled_points_all, axis=0).astype(np.float32),
            np.stack(sampled_boxes_all, axis=0).astype(np.float32),
            np.array(sampled_classes_all, dtype=np.int32),
        )


def augment_point_cloud(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    config: AugmentationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply geometric augmentations to point cloud and GT boxes.

    Augmentations applied in sequence: random flip X, random flip Y,
    global rotation, global scaling.

    Args:
        points: Point cloud, shape [N, 4+] (x, y, z, intensity, ...).
        gt_boxes: Ground truth boxes, shape [M, 7].
        config: Augmentation configuration.

    Returns:
        Tuple of (augmented_points, augmented_gt_boxes).
    """
    # Random flip along X axis
    if config.random_flip_x and np.random.random() > 0.5:
        points[:, 1] = -points[:, 1]
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]

    # Random flip along Y axis
    if config.random_flip_y and np.random.random() > 0.5:
        points[:, 0] = -points[:, 0]
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = np.pi - gt_boxes[:, 6]

    # Random global rotation
    rot_min, rot_max = config.rotation_range
    theta = np.random.uniform(rot_min, rot_max)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotate points
    rot_x = points[:, 0] * cos_t - points[:, 1] * sin_t
    rot_y = points[:, 0] * sin_t + points[:, 1] * cos_t
    points[:, 0] = rot_x
    points[:, 1] = rot_y

    # Rotate box centers
    box_rot_x = gt_boxes[:, 0] * cos_t - gt_boxes[:, 1] * sin_t
    box_rot_y = gt_boxes[:, 0] * sin_t + gt_boxes[:, 1] * cos_t
    gt_boxes[:, 0] = box_rot_x
    gt_boxes[:, 1] = box_rot_y
    gt_boxes[:, 6] += theta

    # Random global scaling
    scale_min, scale_max = config.scaling_range
    scale = np.random.uniform(scale_min, scale_max)
    points[:, :3] *= scale
    gt_boxes[:, :6] *= scale

    return points, gt_boxes


def voxelize_point_cloud(
    points: np.ndarray, config: VoxelConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert point cloud to pillar representation via voxelization.

    Points are binned into vertical columns (pillars) in the BEV grid.
    Each pillar stores up to max_points_per_voxel points with their
    features augmented by relative offsets from pillar center.

    Args:
        points: Raw point cloud, shape [N, 4] (x, y, z, intensity).
        config: Voxel configuration.

    Returns:
        Tuple of:
            pillars: Pillar features [max_voxels, max_points, 9].
                Features: (x, y, z, intensity, xc, yc, zc, xp, yp)
                where c=offset from mean, p=offset from pillar center.
            coords: Pillar grid coordinates [max_voxels, 2] (grid_x, grid_y).
            num_points_per_pillar: Points count [max_voxels].
    """
    pc_range = np.array(config.point_cloud_range)
    voxel_size = np.array(config.voxel_size)
    max_points = config.max_points_per_voxel
    max_voxels = config.max_voxels

    # Filter points within range
    mask = (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] < pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] < pc_range[4])
        & (points[:, 2] >= pc_range[2])
        & (points[:, 2] < pc_range[5])
    )
    points = points[mask]

    # Compute grid indices for each point
    grid_idx_x = ((points[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int32)
    grid_idx_y = ((points[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int32)

    # Unique pillar indices
    grid_size_x = int((pc_range[3] - pc_range[0]) / voxel_size[0])
    grid_size_y = int((pc_range[4] - pc_range[1]) / voxel_size[1])

    # Clip to valid range
    grid_idx_x = np.clip(grid_idx_x, 0, grid_size_x - 1)
    grid_idx_y = np.clip(grid_idx_y, 0, grid_size_y - 1)

    # Hash to unique pillar ID
    pillar_ids = grid_idx_y * grid_size_x + grid_idx_x

    # Find unique pillars and shuffle for random subsampling if needed
    unique_pillars, inverse_indices = np.unique(pillar_ids, return_inverse=True)

    if len(unique_pillars) > max_voxels:
        selected = np.random.choice(len(unique_pillars), max_voxels, replace=False)
        selected_set = set(selected)
        keep_mask = np.array([inverse_indices[i] in selected_set for i in range(len(points))])
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

        # Subsample if too many points
        n_pts = min(len(pillar_points), max_points)
        if len(pillar_points) > max_points:
            choice = np.random.choice(len(pillar_points), max_points, replace=False)
            pillar_points = pillar_points[choice]
            n_pts = max_points

        # Pillar center in world coordinates
        pillar_center_x = (
            grid_idx_x[np.where(point_mask)[0][0]] * voxel_size[0]
            + pc_range[0]
            + voxel_size[0] / 2.0
        )
        pillar_center_y = (
            grid_idx_y[np.where(point_mask)[0][0]] * voxel_size[1]
            + pc_range[1]
            + voxel_size[1] / 2.0
        )

        # Compute mean of points in pillar
        mean_xyz = pillar_points[:n_pts, :3].mean(axis=0)

        # Build 9-dim features: x, y, z, intensity, xc, yc, zc, xp, yp
        features = np.zeros((n_pts, 9), dtype=np.float32)
        features[:, :4] = pillar_points[:n_pts, :4]
        features[:, 4] = pillar_points[:n_pts, 0] - mean_xyz[0]
        features[:, 5] = pillar_points[:n_pts, 1] - mean_xyz[1]
        features[:, 6] = pillar_points[:n_pts, 2] - mean_xyz[2]
        features[:, 7] = pillar_points[:n_pts, 0] - pillar_center_x
        features[:, 8] = pillar_points[:n_pts, 1] - pillar_center_y

        pillars[i, :n_pts, :] = features
        coords[i, 0] = grid_idx_x[np.where(point_mask)[0][0]]
        coords[i, 1] = grid_idx_y[np.where(point_mask)[0][0]]
        num_points_per_pillar[i] = n_pts

    return pillars, coords, num_points_per_pillar


def load_sample(
    info: Dict[str, Any],
    config: TrainConfig,
    gt_db: Optional[GTDatabase],
    is_training: bool,
) -> Dict[str, np.ndarray]:
    """Load and preprocess a single training sample.

    Args:
        info: Sample metadata dict with keys 'point_cloud_path',
            'gt_boxes', 'gt_classes'.
        config: Training configuration.
        gt_db: Optional GT database for copy-paste augmentation.
        is_training: Whether in training mode (enables augmentation).

    Returns:
        Dictionary with keys:
            'pillars': [max_voxels, max_points, 9]
            'coords': [max_voxels, 2]
            'num_points': [max_voxels]
            'cls_targets': [num_anchors, num_classes]
            'reg_targets': [num_anchors, 7]
            'dir_targets': [num_anchors, 2]
            'pos_mask': [num_anchors]
            'neg_mask': [num_anchors]
    """
    # Load point cloud
    pc_path = info["point_cloud_path"]
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)

    gt_boxes = info["gt_boxes"].copy()
    gt_classes = info["gt_classes"].copy()

    if is_training:
        # GT database sampling (copy-paste augmentation)
        if gt_db is not None and gt_db.database:
            sampled_pts, sampled_boxes, sampled_classes = gt_db.sample(
                gt_boxes, gt_classes
            )
            if sampled_boxes.shape[0] > 0:
                points = np.concatenate([points, sampled_pts], axis=0)
                gt_boxes = np.concatenate([gt_boxes, sampled_boxes], axis=0)
                gt_classes = np.concatenate([gt_classes, sampled_classes], axis=0)

        # Geometric augmentation
        points, gt_boxes = augment_point_cloud(
            points, gt_boxes, config.augmentation
        )

    # Voxelization
    pillars, coords, num_points = voxelize_point_cloud(points, config.voxel)

    # Generate anchors and compute targets
    anchor_gen = AnchorGenerator(config)
    anchors = anchor_gen.get_anchors()

    cls_targets, reg_targets, dir_targets, pos_mask, neg_mask = (
        assign_anchor_targets(anchors, gt_boxes, gt_classes, config)
    )

    return {
        "pillars": pillars,
        "coords": coords,
        "num_points": num_points,
        "cls_targets": cls_targets,
        "reg_targets": reg_targets,
        "dir_targets": dir_targets,
        "pos_mask": pos_mask.astype(np.float32),
        "neg_mask": neg_mask.astype(np.float32),
    }


def create_dataset(
    info_path: str,
    config: TrainConfig,
    gt_db: Optional[GTDatabase],
    is_training: bool,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset for training or validation.

    Uses tf.py_function to wrap numpy-based preprocessing. Supports
    parallel loading, prefetching, and shuffling for training.

    Args:
        info_path: Path to pickled list of sample info dicts.
        config: Training configuration.
        gt_db: Optional GT database for augmentation.
        is_training: Whether this is a training dataset.

    Returns:
        tf.data.Dataset yielding batches of preprocessed samples.
    """
    import pickle

    with open(info_path, "rb") as f:
        sample_infos = pickle.load(f)

    logger.info("Loaded %d samples from %s", len(sample_infos), info_path)

    # Pre-compute anchor shape for output signature
    anchor_gen = AnchorGenerator(config)
    num_anchors = anchor_gen.get_anchors().shape[0]
    max_voxels = config.voxel.max_voxels
    max_points = config.voxel.max_points_per_voxel
    num_classes = config.num_classes

    output_signature = {
        "pillars": tf.TensorSpec([max_voxels, max_points, 9], tf.float32),
        "coords": tf.TensorSpec([max_voxels, 2], tf.int32),
        "num_points": tf.TensorSpec([max_voxels], tf.int32),
        "cls_targets": tf.TensorSpec([num_anchors, num_classes], tf.float32),
        "reg_targets": tf.TensorSpec([num_anchors, 7], tf.float32),
        "dir_targets": tf.TensorSpec([num_anchors, 2], tf.float32),
        "pos_mask": tf.TensorSpec([num_anchors], tf.float32),
        "neg_mask": tf.TensorSpec([num_anchors], tf.float32),
    }

    def generator():
        indices = np.arange(len(sample_infos))
        if is_training:
            np.random.shuffle(indices)
        for idx in indices:
            try:
                sample = load_sample(
                    sample_infos[idx], config, gt_db, is_training
                )
                yield sample
            except Exception as e:
                logger.warning("Failed to load sample %d: %s", idx, e)
                continue

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=output_signature
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(512, len(sample_infos)))

    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =============================================================================
# Model Definition
# =============================================================================


class PillarFeatureNet(tf.keras.layers.Layer):
    """Pillar Feature Network: encodes raw pillar features into a fixed representation.

    Applies a simplified PointNet (shared MLP + max pooling) to each pillar
    independently, producing a C-dimensional feature vector per pillar.

    Args:
        num_features: Output feature dimension per pillar.
    """

    def __init__(self, num_features: int = 64, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_features = num_features
        self.linear = tf.keras.layers.Dense(
            num_features, use_bias=False, name="linear"
        )
        self.bn = tf.keras.layers.BatchNormalization(name="bn")

    def call(
        self, pillars: tf.Tensor, num_points: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Forward pass through pillar feature network.

        Args:
            pillars: [B, max_voxels, max_points, 9] raw pillar features.
            num_points: [B, max_voxels] number of valid points per pillar.
            training: Whether in training mode (affects batch norm).

        Returns:
            Pillar features [B, max_voxels, num_features] after max pooling.
        """
        # pillars shape: [B, V, P, 9]
        x = self.linear(pillars)  # [B, V, P, C]

        # Reshape for batch norm: merge batch and voxel dims
        shape = tf.shape(x)
        B, V, P, C = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, [B * V * P, C])
        x = self.bn(x, training=training)
        x = tf.reshape(x, [B, V, P, self.num_features])

        x = tf.nn.relu(x)

        # Create mask for valid points
        # num_points: [B, V] -> mask: [B, V, P, 1]
        point_range = tf.range(P, dtype=tf.int32)
        mask = tf.cast(
            point_range[None, None, :] < num_points[:, :, None], tf.float32
        )
        mask = mask[:, :, :, None]  # [B, V, P, 1]

        # Apply mask and max pool over points dimension
        x = x * mask
        x = tf.reduce_max(x, axis=2)  # [B, V, C]

        return x


class PointPillarsScatter(tf.keras.layers.Layer):
    """Scatter pillar features into a pseudo-image (BEV feature map).

    Places the learned pillar features at their corresponding spatial
    locations to create a dense 2D representation suitable for 2D
    convolution-based backbone processing.

    Args:
        output_shape: Tuple of (height, width) for the BEV grid.
        num_features: Number of channels per pillar.
    """

    def __init__(
        self, output_shape: Tuple[int, int], num_features: int, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.output_h = output_shape[0]
        self.output_w = output_shape[1]
        self.num_features = num_features

    def call(self, pillar_features: tf.Tensor, coords: tf.Tensor) -> tf.Tensor:
        """Scatter pillar features to BEV pseudo-image.

        Args:
            pillar_features: [B, max_voxels, C] encoded pillar features.
            coords: [B, max_voxels, 2] grid coordinates (x, y).

        Returns:
            BEV pseudo-image [B, H, W, C].
        """
        B = tf.shape(pillar_features)[0]
        canvas = tf.zeros(
            [B, self.output_h, self.output_w, self.num_features],
            dtype=tf.float32,
        )

        # Batch indices
        batch_idx = tf.repeat(
            tf.range(B)[:, None], tf.shape(coords)[1], axis=1
        )  # [B, V]

        # Flatten batch for scatter
        batch_flat = tf.reshape(batch_idx, [-1])
        coords_flat = tf.reshape(coords, [-1, 2])
        features_flat = tf.reshape(pillar_features, [-1, self.num_features])

        # Build scatter indices [B*V, 3] -> (batch, y, x)
        indices = tf.stack(
            [batch_flat, coords_flat[:, 1], coords_flat[:, 0]], axis=1
        )

        canvas = tf.tensor_scatter_nd_update(canvas, indices, features_flat)
        return canvas


class BackboneBlock(tf.keras.layers.Layer):
    """A block of convolutional layers with batch normalization and ReLU.

    Args:
        num_filters: Number of output filters.
        num_layers: Number of convolutional layers in the block.
        stride: Stride for the first convolution (downsampling).
    """

    def __init__(
        self, num_filters: int, num_layers: int, stride: int = 1, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.convs = []
        self.bns = []

        for i in range(num_layers):
            s = stride if i == 0 else 1
            self.convs.append(
                tf.keras.layers.Conv2D(
                    num_filters,
                    kernel_size=3,
                    strides=s,
                    padding="same",
                    use_bias=False,
                    name=f"conv_{i}",
                )
            )
            self.bns.append(
                tf.keras.layers.BatchNormalization(name=f"bn_{i}")
            )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through backbone block.

        Args:
            x: Input feature map [B, H, W, C].
            training: Whether in training mode.

        Returns:
            Output feature map [B, H//stride, W//stride, num_filters].
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x, training=training)
            x = tf.nn.relu(x)
        return x


class Backbone(tf.keras.layers.Layer):
    """Multi-scale backbone network with top-down feature fusion.

    Processes the BEV pseudo-image through multiple resolution stages,
    then upsamples and concatenates features from all scales.

    Args:
        channels: List of channel counts per stage.
        strides: List of strides per stage.
        num_blocks: List of number of conv layers per stage.
    """

    def __init__(
        self,
        channels: List[int],
        strides: List[int],
        num_blocks: List[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.blocks = []
        self.deconvs = []

        in_channels = channels[0]
        for i, (c, s, n) in enumerate(zip(channels, strides, num_blocks)):
            self.blocks.append(
                BackboneBlock(c, n, stride=s, name=f"block_{i}")
            )
            self.deconvs.append(
                tf.keras.layers.Conv2DTranspose(
                    channels[0],
                    kernel_size=s * 2,
                    strides=int(np.prod(strides[: i + 1])),
                    padding="same",
                    use_bias=False,
                    name=f"deconv_{i}",
                )
            )

        self.deconv_bns = [
            tf.keras.layers.BatchNormalization(name=f"deconv_bn_{i}")
            for i in range(len(channels))
        ]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through multi-scale backbone.

        Args:
            x: BEV pseudo-image [B, H, W, C].
            training: Whether in training mode.

        Returns:
            Fused multi-scale features [B, H, W, C * num_stages].
        """
        ups = []
        for block, deconv, bn in zip(self.blocks, self.deconvs, self.deconv_bns):
            x = block(x, training=training)
            up = deconv(x)
            up = bn(up, training=training)
            up = tf.nn.relu(up)
            ups.append(up)

        # Concatenate upsampled features from all scales
        out = tf.concat(ups, axis=-1)
        return out


class DetectionHead(tf.keras.layers.Layer):
    """SSD-style detection head for classification, regression, and direction.

    Predicts per-anchor classification scores, box regression parameters,
    and direction classification using 1x1 convolutions.

    Args:
        num_classes: Number of object classes.
        num_anchors_per_location: Number of anchors at each spatial location.
    """

    def __init__(
        self,
        num_classes: int,
        num_anchors_per_location: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_location

        self.cls_conv = tf.keras.layers.Conv2D(
            num_anchors_per_location * num_classes,
            kernel_size=1,
            padding="same",
            name="cls_conv",
        )
        self.reg_conv = tf.keras.layers.Conv2D(
            num_anchors_per_location * 7,
            kernel_size=1,
            padding="same",
            name="reg_conv",
        )
        self.dir_conv = tf.keras.layers.Conv2D(
            num_anchors_per_location * 2,
            kernel_size=1,
            padding="same",
            name="dir_conv",
        )

    def call(
        self, x: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass through detection head.

        Args:
            x: Backbone features [B, H, W, C].

        Returns:
            Tuple of:
                cls_preds: [B, H*W*num_anchors, num_classes]
                reg_preds: [B, H*W*num_anchors, 7]
                dir_preds: [B, H*W*num_anchors, 2]
        """
        B = tf.shape(x)[0]

        cls_out = self.cls_conv(x)  # [B, H, W, A*C]
        reg_out = self.reg_conv(x)  # [B, H, W, A*7]
        dir_out = self.dir_conv(x)  # [B, H, W, A*2]

        # Reshape to [B, total_anchors, ...]
        cls_out = tf.reshape(cls_out, [B, -1, self.num_classes])
        reg_out = tf.reshape(reg_out, [B, -1, 7])
        dir_out = tf.reshape(dir_out, [B, -1, 2])

        return cls_out, reg_out, dir_out


class PointPillarsModel(tf.keras.Model):
    """Complete PointPillars model for 3D object detection from LiDAR.

    Architecture:
    1. Pillar Feature Net: PointNet per pillar -> fixed-size features
    2. Scatter: Map pillar features to BEV pseudo-image
    3. Backbone: Multi-scale 2D CNN with feature fusion
    4. Detection Head: Per-anchor predictions

    Args:
        config: Training configuration.
    """

    def __init__(self, config: TrainConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config

        pc_range = config.voxel.point_cloud_range
        voxel_size = config.voxel.voxel_size

        grid_x = int((pc_range[3] - pc_range[0]) / voxel_size[0])
        grid_y = int((pc_range[4] - pc_range[1]) / voxel_size[1])

        # Count anchors per location
        num_anchors_per_loc = sum(
            len(a.anchor_sizes) * len(a.anchor_rotations)
            for a in config.anchors
        )

        self.pfn = PillarFeatureNet(config.pillar_features, name="pfn")
        self.scatter = PointPillarsScatter(
            output_shape=(grid_y, grid_x),
            num_features=config.pillar_features,
            name="scatter",
        )
        self.backbone = Backbone(
            channels=config.backbone_channels,
            strides=config.backbone_strides,
            num_blocks=config.backbone_num_blocks,
            name="backbone",
        )
        self.head = DetectionHead(
            num_classes=config.num_classes,
            num_anchors_per_location=num_anchors_per_loc,
            name="head",
        )

    def call(
        self,
        pillars: tf.Tensor,
        coords: tf.Tensor,
        num_points: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass through the full PointPillars model.

        Args:
            pillars: [B, max_voxels, max_points, 9] raw pillar features.
            coords: [B, max_voxels, 2] pillar grid coordinates.
            num_points: [B, max_voxels] valid point counts per pillar.
            training: Whether in training mode.

        Returns:
            Tuple of (cls_preds, reg_preds, dir_preds).
        """
        # Encode pillars
        pillar_features = self.pfn(pillars, num_points, training=training)

        # Scatter to BEV
        bev = self.scatter(pillar_features, coords)

        # Backbone feature extraction
        features = self.backbone(bev, training=training)

        # Detection head
        cls_preds, reg_preds, dir_preds = self.head(features)

        return cls_preds, reg_preds, dir_preds


# =============================================================================
# Training Logic
# =============================================================================


def compute_losses(
    cls_preds: tf.Tensor,
    reg_preds: tf.Tensor,
    dir_preds: tf.Tensor,
    cls_targets: tf.Tensor,
    reg_targets: tf.Tensor,
    dir_targets: tf.Tensor,
    pos_mask: tf.Tensor,
    neg_mask: tf.Tensor,
    config: TrainConfig,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute all loss components with positive/negative sample balancing.

    Balances positive and negative samples using a fixed ratio to prevent
    the overwhelming number of negative anchors from dominating training.

    Args:
        cls_preds: [B, A, C] predicted class logits.
        reg_preds: [B, A, 7] predicted box regression.
        dir_preds: [B, A, 2] predicted direction logits.
        cls_targets: [B, A, C] one-hot class targets.
        reg_targets: [B, A, 7] regression targets.
        dir_targets: [B, A, 2] direction targets.
        pos_mask: [B, A] positive anchor mask.
        neg_mask: [B, A] negative anchor mask.
        config: Training configuration.

    Returns:
        Tuple of (total_loss, cls_loss, reg_loss, dir_loss).
    """
    batch_size = tf.shape(cls_preds)[0]
    batch_size_f = tf.cast(batch_size, tf.float32)

    # Count positives and balance negatives
    num_pos = tf.reduce_sum(pos_mask)
    num_neg_limit = tf.cast(
        tf.cast(num_pos, tf.float32) * config.pos_neg_ratio, tf.int32
    )
    num_neg_limit = tf.maximum(num_neg_limit, 1)

    # Hard negative mining: select top-scoring negatives
    # Flatten for per-sample processing
    cls_loss_all = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=cls_targets, logits=cls_preds
    )
    cls_loss_per_anchor = tf.reduce_sum(cls_loss_all, axis=-1)  # [B, A]

    # Mask negative losses and select top-k hardest
    neg_cls_loss = cls_loss_per_anchor * neg_mask
    neg_cls_loss_flat = tf.reshape(neg_cls_loss, [-1])

    # Get top-k negative losses
    total_neg = tf.shape(neg_cls_loss_flat)[0]
    k = tf.minimum(num_neg_limit, total_neg)
    _, top_neg_indices = tf.math.top_k(neg_cls_loss_flat, k=k)

    # Build balanced mask for negatives
    neg_mask_balanced = tf.zeros_like(neg_cls_loss_flat)
    neg_mask_balanced = tf.tensor_scatter_nd_update(
        neg_mask_balanced,
        top_neg_indices[:, None],
        tf.ones(k, dtype=tf.float32),
    )
    neg_mask_balanced = tf.reshape(neg_mask_balanced, tf.shape(neg_mask))

    # Classification loss: positives + balanced negatives
    combined_mask = pos_mask + neg_mask_balanced
    normalizer = tf.maximum(num_pos, 1.0)

    # Focal loss on positives and hard negatives
    cls_loss_masked = focal_loss(
        tf.boolean_mask(cls_preds, combined_mask > 0),
        tf.boolean_mask(cls_targets, combined_mask > 0),
        alpha=config.focal_alpha,
        gamma=config.focal_gamma,
    )
    cls_loss = cls_loss_masked / normalizer

    # Regression loss: only on positive anchors
    pos_reg_preds = tf.boolean_mask(reg_preds, pos_mask > 0)
    pos_reg_targets = tf.boolean_mask(reg_targets, pos_mask > 0)
    reg_loss = smooth_l1_loss(pos_reg_preds, pos_reg_targets) / normalizer

    # Direction loss: only on positive anchors
    pos_dir_preds = tf.boolean_mask(dir_preds, pos_mask > 0)
    pos_dir_targets = tf.boolean_mask(dir_targets, pos_mask > 0)
    dir_loss = direction_classification_loss(
        pos_dir_preds, pos_dir_targets
    ) / normalizer

    # Weighted total loss
    total_loss = (
        config.cls_weight * cls_loss
        + config.reg_weight * reg_loss
        + config.dir_weight * dir_loss
    )

    return total_loss, cls_loss, reg_loss, dir_loss


class Trainer:
    """Manages the full training lifecycle for PointPillars.

    Handles multi-GPU distribution, checkpoint management, TensorBoard
    logging, validation, and model saving.

    Args:
        config: Training configuration.
        strategy: TensorFlow distribution strategy.
    """

    def __init__(
        self, config: TrainConfig, strategy: tf.distribute.Strategy
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.global_step = 0
        self.best_val_metric = -float("inf")

        # Setup output directories
        self.output_dir = Path(config.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.tb_dir = self.output_dir / "tensorboard"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        # Build model and optimizer within strategy scope
        with self.strategy.scope():
            self.model = PointPillarsModel(config, name="pointpillars")

            # Compute total training steps for LR schedule
            # Estimated from dataset size (loaded later)
            self.steps_per_epoch = 1  # Will be updated after dataset creation
            total_steps = config.epochs * self.steps_per_epoch
            warmup_steps = config.warmup_epochs * self.steps_per_epoch

            self.lr_schedule = OneCycleLR(
                max_lr=config.learning_rate,
                total_steps=max(total_steps, 1),
                warmup_steps=max(warmup_steps, 1),
            )

            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.lr_schedule,
                weight_decay=config.weight_decay,
                clipnorm=config.gradient_clip_norm,
            )

            # Checkpoint
            self.checkpoint = tf.train.Checkpoint(
                model=self.model, optimizer=self.optimizer
            )
            self.ckpt_manager = tf.train.CheckpointManager(
                self.checkpoint,
                str(self.ckpt_dir),
                max_to_keep=config.keep_max_checkpoints,
            )

        # TensorBoard writer
        self.summary_writer = tf.summary.create_file_writer(str(self.tb_dir))

        # Restore from latest checkpoint if available
        if self.ckpt_manager.latest_checkpoint:
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            logger.info(
                "Restored from checkpoint: %s",
                self.ckpt_manager.latest_checkpoint,
            )

    def update_schedule(self, steps_per_epoch: int) -> None:
        """Update learning rate schedule with actual steps per epoch.

        Must be called after dataset creation when the true dataset
        size is known.

        Args:
            steps_per_epoch: Number of training steps per epoch.
        """
        self.steps_per_epoch = steps_per_epoch
        total_steps = self.config.epochs * steps_per_epoch
        warmup_steps = self.config.warmup_epochs * steps_per_epoch

        with self.strategy.scope():
            self.lr_schedule = OneCycleLR(
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )
            self.optimizer.learning_rate = self.lr_schedule

    @tf.function
    def _train_step(
        self, batch: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Execute a single distributed training step.

        Args:
            batch: Dictionary of batched tensors from the data pipeline.

        Returns:
            Dictionary of loss values for logging.
        """

        def step_fn(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            with tf.GradientTape() as tape:
                cls_preds, reg_preds, dir_preds = self.model(
                    batch["pillars"],
                    batch["coords"],
                    batch["num_points"],
                    training=True,
                )

                total_loss, cls_loss, reg_loss, dir_loss = compute_losses(
                    cls_preds,
                    reg_preds,
                    dir_preds,
                    batch["cls_targets"],
                    batch["reg_targets"],
                    batch["dir_targets"],
                    batch["pos_mask"],
                    batch["neg_mask"],
                    self.config,
                )

                # Scale loss for distributed training
                scaled_loss = total_loss / tf.cast(
                    self.strategy.num_replicas_in_sync, tf.float32
                )

            # Compute and apply gradients with clipping
            gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients, grad_norm = tf.clip_by_global_norm(
                gradients, self.config.gradient_clip_norm
            )
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            return {
                "total_loss": total_loss,
                "cls_loss": cls_loss,
                "reg_loss": reg_loss,
                "dir_loss": dir_loss,
                "grad_norm": grad_norm,
            }

        per_replica_losses = self.strategy.run(step_fn, args=(batch,))

        # Reduce across replicas
        reduced = {}
        for key, value in per_replica_losses.items():
            reduced[key] = self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, value, axis=None
            )
        return reduced

    @tf.function
    def _val_step(
        self, batch: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Execute a single distributed validation step.

        Args:
            batch: Dictionary of batched tensors.

        Returns:
            Dictionary of loss values.
        """

        def step_fn(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            cls_preds, reg_preds, dir_preds = self.model(
                batch["pillars"],
                batch["coords"],
                batch["num_points"],
                training=False,
            )

            total_loss, cls_loss, reg_loss, dir_loss = compute_losses(
                cls_preds,
                reg_preds,
                dir_preds,
                batch["cls_targets"],
                batch["reg_targets"],
                batch["dir_targets"],
                batch["pos_mask"],
                batch["neg_mask"],
                self.config,
            )

            return {
                "total_loss": total_loss,
                "cls_loss": cls_loss,
                "reg_loss": reg_loss,
                "dir_loss": dir_loss,
            }

        per_replica_losses = self.strategy.run(step_fn, args=(batch,))
        reduced = {}
        for key, value in per_replica_losses.items():
            reduced[key] = self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, value, axis=None
            )
        return reduced

    def validate(self, val_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Run full validation pass and compute average metrics.

        Args:
            val_dataset: Distributed validation dataset.

        Returns:
            Dictionary of average validation metrics.
        """
        val_metrics: Dict[str, List[float]] = {
            "total_loss": [],
            "cls_loss": [],
            "reg_loss": [],
            "dir_loss": [],
        }

        for batch in val_dataset:
            losses = self._val_step(batch)
            for key in val_metrics:
                val_metrics[key].append(float(losses[key].numpy()))

        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        return avg_metrics

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
    ) -> None:
        """Execute the full training loop.

        Iterates over epochs, performing training steps with logging,
        periodic validation, checkpoint saving, and best model tracking.

        Args:
            train_dataset: Distributed training dataset.
            val_dataset: Optional distributed validation dataset.
        """
        logger.info("Starting training for %d epochs", self.config.epochs)
        logger.info("Strategy: %s", self.strategy.__class__.__name__)
        logger.info(
            "Num replicas: %d", self.strategy.num_replicas_in_sync
        )

        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            epoch_losses: Dict[str, List[float]] = {
                "total_loss": [],
                "cls_loss": [],
                "reg_loss": [],
                "dir_loss": [],
                "grad_norm": [],
            }

            for step, batch in enumerate(train_dataset):
                losses = self._train_step(batch)
                self.global_step += 1

                for key in epoch_losses:
                    epoch_losses[key].append(float(losses[key].numpy()))

                # Periodic logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    current_lr = float(
                        self.lr_schedule(self.global_step)
                    )
                    avg_total = np.mean(epoch_losses["total_loss"][-50:])
                    avg_cls = np.mean(epoch_losses["cls_loss"][-50:])
                    avg_reg = np.mean(epoch_losses["reg_loss"][-50:])
                    avg_dir = np.mean(epoch_losses["dir_loss"][-50:])
                    avg_gnorm = np.mean(epoch_losses["grad_norm"][-50:])

                    logger.info(
                        "Epoch %d Step %d | "
                        "loss: %.4f cls: %.4f reg: %.4f dir: %.4f | "
                        "gnorm: %.2f lr: %.6f",
                        epoch + 1,
                        step + 1,
                        avg_total,
                        avg_cls,
                        avg_reg,
                        avg_dir,
                        avg_gnorm,
                        current_lr,
                    )

                    # TensorBoard logging
                    with self.summary_writer.as_default():
                        tf.summary.scalar(
                            "train/total_loss", avg_total, step=self.global_step
                        )
                        tf.summary.scalar(
                            "train/cls_loss", avg_cls, step=self.global_step
                        )
                        tf.summary.scalar(
                            "train/reg_loss", avg_reg, step=self.global_step
                        )
                        tf.summary.scalar(
                            "train/dir_loss", avg_dir, step=self.global_step
                        )
                        tf.summary.scalar(
                            "train/grad_norm", avg_gnorm, step=self.global_step
                        )
                        tf.summary.scalar(
                            "train/learning_rate",
                            current_lr,
                            step=self.global_step,
                        )

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = np.mean(epoch_losses["total_loss"])
            logger.info(
                "Epoch %d completed in %.1fs | avg loss: %.4f",
                epoch + 1,
                epoch_time,
                avg_epoch_loss,
            )

            with self.summary_writer.as_default():
                tf.summary.scalar(
                    "epoch/train_loss", avg_epoch_loss, step=epoch + 1
                )
                tf.summary.scalar(
                    "epoch/time_seconds", epoch_time, step=epoch + 1
                )

            # Validation
            if (
                val_dataset is not None
                and (epoch + 1) % self.config.eval_every_n_epochs == 0
            ):
                logger.info("Running validation...")
                val_metrics = self.validate(val_dataset)
                logger.info(
                    "Validation | loss: %.4f cls: %.4f reg: %.4f dir: %.4f",
                    val_metrics["total_loss"],
                    val_metrics["cls_loss"],
                    val_metrics["reg_loss"],
                    val_metrics["dir_loss"],
                )

                with self.summary_writer.as_default():
                    for key, value in val_metrics.items():
                        tf.summary.scalar(
                            f"val/{key}", value, step=epoch + 1
                        )

                # Save best model (lower total loss = better)
                # Using negative loss as "metric" since we track best as max
                val_metric = -val_metrics["total_loss"]
                if val_metric > self.best_val_metric:
                    self.best_val_metric = val_metric
                    best_path = self.output_dir / "best_model"
                    self.model.save_weights(str(best_path / "weights"))
                    logger.info(
                        "New best model saved (val_loss=%.4f) at %s",
                        val_metrics["total_loss"],
                        best_path,
                    )

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                ckpt_path = self.ckpt_manager.save()
                logger.info("Checkpoint saved: %s", ckpt_path)

        # Final save
        final_ckpt = self.ckpt_manager.save()
        logger.info("Training complete. Final checkpoint: %s", final_ckpt)
        self.summary_writer.close()


# =============================================================================
# Main Entry Point
# =============================================================================


def setup_logging(output_dir: str) -> None:
    """Configure logging to both console and file.

    Args:
        output_dir: Directory where log file will be written.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "train.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path)),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="PointPillars TF2 Training Script"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). "
        "If not specified, uses all available GPUs.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entry point.

    Sets up distribution strategy, loads data, builds model, and
    executes the training loop.
    """
    args = parse_args()

    # Load config
    config = load_config(args.config)
    setup_logging(config.output_dir)

    logger.info("Configuration loaded from: %s", args.config)
    logger.info("Output directory: %s", config.output_dir)

    # GPU setup
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info("Using GPUs: %s", args.gpus)

    # Setup distribution strategy
    gpus = tf.config.list_physical_devices("GPU")
    logger.info("Found %d GPU(s): %s", len(gpus), gpus)

    # Enable memory growth to avoid OOM
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning("Could not set memory growth for %s: %s", gpu, e)

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    elif len(gpus) == 1:
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    else:
        strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
        logger.warning("No GPUs found. Training on CPU.")

    logger.info(
        "Distribution strategy: %s with %d replicas",
        strategy.__class__.__name__,
        strategy.num_replicas_in_sync,
    )

    # Setup GT database for copy-paste augmentation
    gt_db = None
    if config.augmentation.gt_database_path:
        gt_db = GTDatabase(
            config.augmentation.gt_database_path,
            config.augmentation.gt_database_max_samples,
        )

    # Create datasets
    train_info_path = os.path.join(config.data_root, config.train_info_path)
    val_info_path = os.path.join(config.data_root, config.val_info_path)

    logger.info("Creating training dataset from: %s", train_info_path)
    train_dataset = create_dataset(
        train_info_path, config, gt_db, is_training=True
    )

    val_dataset = None
    if os.path.exists(val_info_path):
        logger.info("Creating validation dataset from: %s", val_info_path)
        val_dataset = create_dataset(
            val_info_path, config, gt_db=None, is_training=False
        )

    # Estimate steps per epoch
    import pickle

    with open(train_info_path, "rb") as f:
        train_infos = pickle.load(f)
    steps_per_epoch = len(train_infos) // (
        config.batch_size * strategy.num_replicas_in_sync
    )
    steps_per_epoch = max(steps_per_epoch, 1)
    logger.info("Estimated steps per epoch: %d", steps_per_epoch)

    # Distribute datasets
    dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dist_val_dataset = None
    if val_dataset is not None:
        dist_val_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # Build trainer
    trainer = Trainer(config, strategy)
    trainer.update_schedule(steps_per_epoch)

    # Resume from specific checkpoint if requested
    if args.resume:
        status = trainer.checkpoint.restore(args.resume)
        status.expect_partial()
        logger.info("Resumed from checkpoint: %s", args.resume)

    # Run training
    trainer.train(dist_train_dataset, dist_val_dataset)

    logger.info("Training finished successfully.")


if __name__ == "__main__":
    main()
