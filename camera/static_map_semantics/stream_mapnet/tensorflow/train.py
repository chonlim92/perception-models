"""
StreamMapNet - TensorFlow 2 Training Script

Multi-GPU training pipeline for StreamMapNet with temporal sequence handling,
Hungarian matching, focal loss, and cosine decay learning rate schedule.

Usage:
    # Training with synthetic data (for testing):
    python train.py --synthetic --epochs 5 --batch_size 2

    # Training with real data:
    python train.py --data_dir /path/to/nuscenes --epochs 24 --batch_size 4

    # Multi-GPU with mixed precision:
    python train.py --data_dir /path/to/data --mixed_precision --num_gpus 4
"""

import argparse
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.optimize import linear_sum_assignment

from model import StreamMapNet, DEFAULT_CONFIG


# =============================================================================
# Learning Rate Schedule
# =============================================================================

class WarmupCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay learning rate schedule."""

    def __init__(
        self,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ):
        super().__init__()
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.peak_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay
        decay_steps = total_steps - warmup_steps
        progress = (step - warmup_steps) / tf.maximum(decay_steps, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * progress))
        decay_lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay

        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


# =============================================================================
# Loss Functions
# =============================================================================

def focal_loss(
    logits: tf.Tensor,
    targets: tf.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> tf.Tensor:
    """
    Focal loss for classification.

    Args:
        logits: (B, N, C+1) raw logits from model (includes background class)
        targets: (B, N) integer class labels (0 = background, 1..C = classes)
        alpha: Balancing factor
        gamma: Focusing parameter

    Returns:
        Scalar focal loss averaged over batch and queries.
    """
    num_classes = logits.shape[-1]
    # One-hot encode targets
    targets_onehot = tf.one_hot(targets, depth=num_classes)  # (B, N, C+1)

    # Compute probabilities via softmax
    probs = tf.nn.softmax(logits, axis=-1)  # (B, N, C+1)

    # Focal loss per element
    ce = -targets_onehot * tf.math.log(tf.maximum(probs, 1e-8))
    p_t = tf.reduce_sum(probs * targets_onehot, axis=-1)  # (B, N)
    focal_weight = alpha * tf.pow(1.0 - p_t, gamma)  # (B, N)

    # Weighted cross-entropy
    loss_per_query = focal_weight * tf.reduce_sum(ce, axis=-1)  # (B, N)
    return tf.reduce_mean(loss_per_query)


def point_regression_loss(
    pred_points: tf.Tensor,
    gt_points: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """
    L1 regression loss on matched polyline points.

    Args:
        pred_points: (M, K, 2) predicted points for matched queries
        gt_points: (M, K, 2) ground truth points for matched targets
        mask: (M,) binary mask indicating valid matches

    Returns:
        Scalar L1 loss averaged over valid matches and points.
    """
    if tf.reduce_sum(mask) == 0:
        return tf.constant(0.0, dtype=pred_points.dtype)

    # L1 distance
    diff = tf.abs(pred_points - gt_points)  # (M, K, 2)
    l1 = tf.reduce_mean(diff, axis=[-2, -1])  # (M,)
    # Mask invalid matches
    l1 = l1 * mask
    return tf.reduce_sum(l1) / tf.maximum(tf.reduce_sum(mask), 1.0)


def direction_loss(
    pred_points: tf.Tensor,
    gt_points: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """
    Direction-aware loss to ensure consistent point ordering.
    Compares both forward and reverse point ordering and takes the minimum.

    Args:
        pred_points: (M, K, 2) predicted points
        gt_points: (M, K, 2) ground truth points
        mask: (M,) valid match mask

    Returns:
        Scalar direction loss.
    """
    if tf.reduce_sum(mask) == 0:
        return tf.constant(0.0, dtype=pred_points.dtype)

    # Forward direction loss
    forward_diff = tf.abs(pred_points - gt_points)
    forward_loss = tf.reduce_mean(forward_diff, axis=[-2, -1])  # (M,)

    # Reverse direction loss (reverse gt point order)
    gt_reversed = tf.reverse(gt_points, axis=[1])
    reverse_diff = tf.abs(pred_points - gt_reversed)
    reverse_loss = tf.reduce_mean(reverse_diff, axis=[-2, -1])  # (M,)

    # Take minimum of forward and reverse
    dir_loss = tf.minimum(forward_loss, reverse_loss)
    dir_loss = dir_loss * mask
    return tf.reduce_sum(dir_loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


# =============================================================================
# Hungarian Matching
# =============================================================================

def hungarian_match_single(
    pred_logits: np.ndarray,
    pred_points: np.ndarray,
    gt_classes: np.ndarray,
    gt_points: np.ndarray,
    num_gt: int,
    cls_weight: float = 2.0,
    pts_weight: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Hungarian matching for a single sample.

    Args:
        pred_logits: (N, C+1) prediction logits
        pred_points: (N, K, 2) predicted polyline points
        gt_classes: (M_max,) ground truth class labels (padded)
        gt_points: (M_max, K, 2) ground truth polyline points (padded)
        num_gt: actual number of ground truth elements
        cls_weight: weight for classification cost
        pts_weight: weight for point regression cost

    Returns:
        (pred_indices, gt_indices): matched index pairs
    """
    if num_gt == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    num_queries = pred_logits.shape[0]
    gt_classes_valid = gt_classes[:num_gt]
    gt_points_valid = gt_points[:num_gt]

    # Classification cost: negative probability of the correct class
    pred_probs = np.exp(pred_logits) / np.sum(np.exp(pred_logits), axis=-1, keepdims=True)
    # Cost for each (query, gt) pair
    cls_cost = -pred_probs[:, gt_classes_valid.astype(int)]  # (N, num_gt)

    # Point regression cost: L1 distance
    # pred_points: (N, K, 2), gt_points_valid: (num_gt, K, 2)
    pred_expanded = pred_points[:, np.newaxis, :, :]  # (N, 1, K, 2)
    gt_expanded = gt_points_valid[np.newaxis, :, :, :]  # (1, num_gt, K, 2)
    pts_cost = np.mean(np.abs(pred_expanded - gt_expanded), axis=(-2, -1))  # (N, num_gt)

    # Total cost matrix
    cost_matrix = cls_weight * cls_cost + pts_weight * pts_cost  # (N, num_gt)

    # Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    return pred_indices.astype(np.int64), gt_indices.astype(np.int64)


def hungarian_match_batch(
    pred_logits: tf.Tensor,
    pred_points: tf.Tensor,
    gt_classes: tf.Tensor,
    gt_points: tf.Tensor,
    num_gts: tf.Tensor,
    cls_weight: float = 2.0,
    pts_weight: float = 5.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Perform Hungarian matching for a batch using tf.numpy_function.

    Args:
        pred_logits: (B, N, C+1)
        pred_points: (B, N, K, 2)
        gt_classes: (B, M_max)
        gt_points: (B, M_max, K, 2)
        num_gts: (B,) number of valid GT elements per sample
        cls_weight: classification cost weight
        pts_weight: point cost weight

    Returns:
        matched_pred_indices: (B, N) indices (-1 for unmatched)
        matched_gt_indices: (B, N) indices (-1 for unmatched)
        match_mask: (B, N) binary mask of matched queries
    """
    batch_size = pred_logits.shape[0] or tf.shape(pred_logits)[0]
    num_queries = pred_logits.shape[1] or tf.shape(pred_logits)[1]

    def _match_numpy(pred_logits_np, pred_points_np, gt_classes_np, gt_points_np, num_gts_np):
        batch_size = pred_logits_np.shape[0]
        num_queries = pred_logits_np.shape[1]

        # Output arrays: for each sample, store matched class targets and point targets
        all_cls_targets = np.zeros((batch_size, num_queries), dtype=np.int32)  # 0 = background
        all_pts_targets = np.zeros(
            (batch_size, num_queries, pred_points_np.shape[2], 2), dtype=np.float32
        )
        all_mask = np.zeros((batch_size, num_queries), dtype=np.float32)

        for b in range(batch_size):
            n_gt = int(num_gts_np[b])
            pred_idx, gt_idx = hungarian_match_single(
                pred_logits_np[b],
                pred_points_np[b],
                gt_classes_np[b],
                gt_points_np[b],
                n_gt,
                cls_weight,
                pts_weight,
            )
            if len(pred_idx) > 0:
                all_cls_targets[b, pred_idx] = gt_classes_np[b, gt_idx].astype(np.int32)
                all_pts_targets[b, pred_idx] = gt_points_np[b, gt_idx]
                all_mask[b, pred_idx] = 1.0

        return all_cls_targets, all_pts_targets, all_mask

    cls_targets, pts_targets, mask = tf.numpy_function(
        _match_numpy,
        [pred_logits, pred_points, gt_classes, gt_points, num_gts],
        [tf.int32, tf.float32, tf.float32],
    )

    return cls_targets, pts_targets, mask


# =============================================================================
# Combined Loss with Hungarian Matching
# =============================================================================

def compute_loss(
    predictions: Dict[str, tf.Tensor],
    gt_classes: tf.Tensor,
    gt_points: tf.Tensor,
    num_gts: tf.Tensor,
    cls_weight: float = 2.0,
    pts_weight: float = 5.0,
    dir_weight: float = 0.5,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    match_cls_weight: float = 2.0,
    match_pts_weight: float = 5.0,
) -> Dict[str, tf.Tensor]:
    """
    Compute total StreamMapNet loss with Hungarian matching.

    Args:
        predictions: dict with 'logits' (B, N, C+1) and 'points' (B, N, K, 2)
        gt_classes: (B, M_max) ground truth class labels
        gt_points: (B, M_max, K, 2) ground truth polyline points
        num_gts: (B,) number of valid GT per sample
        cls_weight: weight for focal loss in total loss
        pts_weight: weight for point regression loss in total loss
        dir_weight: weight for direction loss in total loss
        focal_alpha: focal loss alpha parameter
        focal_gamma: focal loss gamma parameter
        match_cls_weight: classification weight in cost matrix
        match_pts_weight: point weight in cost matrix

    Returns:
        Dictionary of losses: total_loss, cls_loss, pts_loss, dir_loss
    """
    pred_logits = predictions["logits"]  # (B, N, C+1)
    pred_points = predictions["points"]  # (B, N, K, 2)

    # Hungarian matching
    cls_targets, pts_targets, match_mask = hungarian_match_batch(
        pred_logits, pred_points, gt_classes, gt_points, num_gts,
        cls_weight=match_cls_weight, pts_weight=match_pts_weight,
    )

    # Set shapes explicitly (lost through numpy_function)
    batch_size = tf.shape(pred_logits)[0]
    num_queries = pred_logits.shape[1] if pred_logits.shape[1] is not None else tf.shape(pred_logits)[1]
    num_pts = pred_points.shape[2] if pred_points.shape[2] is not None else tf.shape(pred_points)[2]
    cls_targets = tf.ensure_shape(cls_targets, [None, None])
    pts_targets = tf.ensure_shape(pts_targets, [None, None, None, 2])
    match_mask = tf.ensure_shape(match_mask, [None, None])

    # Classification loss (focal loss on all queries)
    cls_loss = focal_loss(pred_logits, cls_targets, alpha=focal_alpha, gamma=focal_gamma)

    # Point regression loss (only on matched queries)
    # Flatten batch and query dims for matched computation
    flat_pred = tf.reshape(pred_points, [-1, tf.shape(pred_points)[2], 2])
    flat_gt = tf.reshape(pts_targets, [-1, tf.shape(pts_targets)[2], 2])
    flat_mask = tf.reshape(match_mask, [-1])

    pts_loss = point_regression_loss(flat_pred, flat_gt, flat_mask)

    # Direction loss (only on matched queries)
    dir_loss = direction_loss(flat_pred, flat_gt, flat_mask)

    # Total weighted loss
    total_loss = cls_weight * cls_loss + pts_weight * pts_loss + dir_weight * dir_loss

    return {
        "total_loss": total_loss,
        "cls_loss": cls_loss,
        "pts_loss": pts_loss,
        "dir_loss": dir_loss,
    }


# =============================================================================
# Data Pipeline
# =============================================================================

def create_synthetic_dataset(
    batch_size: int,
    sequence_length: int,
    num_cameras: int = 6,
    image_height: int = 256,
    image_width: int = 704,
    num_queries: int = 100,
    num_classes: int = 3,
    num_points: int = 20,
    max_gt_elements: int = 30,
    num_samples: int = 200,
) -> tf.data.Dataset:
    """
    Create a synthetic dataset for testing the training pipeline.

    Returns a tf.data.Dataset yielding temporal sequences with:
        - images: (seq_len, num_cameras, H, W, 3)
        - intrinsics: (seq_len, num_cameras, 3, 3)
        - extrinsics: (seq_len, num_cameras, 4, 4)
        - ego_motion: (seq_len, 4, 4)
        - gt_classes: (seq_len, max_gt)
        - gt_points: (seq_len, max_gt, K, 2)
        - num_gts: (seq_len,)
    """

    def _generate_sample():
        for _ in range(num_samples):
            # Generate a temporal sequence
            images = np.random.rand(
                sequence_length, num_cameras, image_height, image_width, 3
            ).astype(np.float32)

            # Camera intrinsics (simple pinhole model)
            intrinsics = np.zeros(
                (sequence_length, num_cameras, 3, 3), dtype=np.float32
            )
            for t in range(sequence_length):
                for c in range(num_cameras):
                    fx = fy = 400.0 + np.random.randn() * 10
                    cx, cy = image_width / 2, image_height / 2
                    intrinsics[t, c] = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1],
                    ])

            # Camera extrinsics (identity with small perturbations)
            extrinsics = np.zeros(
                (sequence_length, num_cameras, 4, 4), dtype=np.float32
            )
            for t in range(sequence_length):
                for c in range(num_cameras):
                    extrinsics[t, c] = np.eye(4) + np.random.randn(4, 4) * 0.01
                    extrinsics[t, c, 3, :] = [0, 0, 0, 1]

            # Ego motion (small forward translation per frame)
            ego_motion = np.zeros((sequence_length, 4, 4), dtype=np.float32)
            for t in range(sequence_length):
                ego_motion[t] = np.eye(4)
                ego_motion[t, 0, 3] = 0.5 * (t + 1)  # forward motion

            # Ground truth map annotations
            n_gt = np.random.randint(5, max_gt_elements + 1)
            gt_classes = np.zeros((sequence_length, max_gt_elements), dtype=np.int32)
            gt_points = np.zeros(
                (sequence_length, max_gt_elements, num_points, 2), dtype=np.float32
            )
            num_gts = np.full((sequence_length,), n_gt, dtype=np.int32)

            for t in range(sequence_length):
                gt_classes[t, :n_gt] = np.random.randint(1, num_classes + 1, size=n_gt)
                # Generate polylines as sequences of points in BEV space [-1, 1]
                for g in range(n_gt):
                    start = np.random.rand(2) * 2 - 1
                    direction = np.random.randn(2) * 0.1
                    points = np.array([
                        start + direction * k for k in range(num_points)
                    ])
                    gt_points[t, g] = np.clip(points, -1, 1)

            yield (
                images.astype(np.float32),
                intrinsics.astype(np.float32),
                extrinsics.astype(np.float32),
                ego_motion.astype(np.float32),
                gt_classes.astype(np.int32),
                gt_points.astype(np.float32),
                num_gts.astype(np.int32),
            )

    output_signature = (
        tf.TensorSpec(shape=(sequence_length, num_cameras, image_height, image_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(sequence_length, num_cameras, 3, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(sequence_length, num_cameras, 4, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(sequence_length, 4, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(sequence_length, max_gt_elements), dtype=tf.int32),
        tf.TensorSpec(shape=(sequence_length, max_gt_elements, num_points, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(sequence_length,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        _generate_sample,
        output_signature=output_signature,
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# =============================================================================
# Data Augmentation
# =============================================================================

# ImageNet normalization constants
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def normalize_imagenet(images: tf.Tensor) -> tf.Tensor:
    """
    Normalize images with ImageNet mean and standard deviation.

    Args:
        images: (..., H, W, 3) float32 images in [0, 1]

    Returns:
        Normalized images.
    """
    return (images - IMAGENET_MEAN) / IMAGENET_STD


def random_horizontal_flip_bev(
    gt_points: tf.Tensor,
    images: tf.Tensor,
    probability: float = 0.5,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Random horizontal flip in BEV space.

    Args:
        gt_points: (..., K, 2) polyline points in normalized BEV coords
        images: (..., H, W, 3) camera images

    Returns:
        Flipped gt_points and images.
    """
    if tf.random.uniform([]) < probability:
        # Flip x-coordinate of BEV points
        x = gt_points[..., 0:1]
        y = gt_points[..., 1:2]
        gt_points = tf.concat([-x, y], axis=-1)
        # Flip images horizontally
        images = tf.reverse(images, axis=[-2])  # flip W dimension
    return gt_points, images


def random_rotate_bev(
    gt_points: tf.Tensor,
    max_angle_deg: float = 15.0,
) -> tf.Tensor:
    """
    Random rotation of BEV polyline points.

    Args:
        gt_points: (..., K, 2) polyline points
        max_angle_deg: maximum rotation angle in degrees

    Returns:
        Rotated gt_points.
    """
    angle_rad = tf.random.uniform(
        [], -max_angle_deg * math.pi / 180.0, max_angle_deg * math.pi / 180.0
    )
    cos_a = tf.cos(angle_rad)
    sin_a = tf.sin(angle_rad)

    x = gt_points[..., 0]
    y = gt_points[..., 1]

    new_x = cos_a * x - sin_a * y
    new_y = sin_a * x + cos_a * y

    return tf.stack([new_x, new_y], axis=-1)


def photometric_augmentation(images: tf.Tensor) -> tf.Tensor:
    """
    Apply photometric augmentations to camera images.

    Args:
        images: (..., H, W, 3) float32 images in [0, 1]

    Returns:
        Augmented images.
    """
    original_shape = tf.shape(images)
    # Flatten batch dims for image operations
    flat_images = tf.reshape(images, [-1, original_shape[-3], original_shape[-2], original_shape[-1]])

    # Random brightness
    flat_images = tf.image.random_brightness(flat_images, max_delta=0.2)
    # Random contrast
    flat_images = tf.image.random_contrast(flat_images, lower=0.8, upper=1.2)
    # Random saturation
    flat_images = tf.image.random_saturation(flat_images, lower=0.8, upper=1.2)
    # Clip to valid range
    flat_images = tf.clip_by_value(flat_images, 0.0, 1.0)

    return tf.reshape(flat_images, original_shape)


def augment_sequence(
    images: tf.Tensor,
    gt_points: tf.Tensor,
    training: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply augmentations to a full temporal sequence.

    Args:
        images: (T, Ncam, H, W, 3)
        gt_points: (T, M, K, 2)
        training: whether to apply augmentations

    Returns:
        Augmented images and gt_points.
    """
    if not training:
        # Always normalize with ImageNet stats
        images = normalize_imagenet(images)
        return images, gt_points

    # Photometric augmentation on images
    images = photometric_augmentation(images)

    # BEV augmentations (applied consistently across temporal sequence)
    gt_points = random_rotate_bev(gt_points, max_angle_deg=10.0)
    gt_points, images = random_horizontal_flip_bev(gt_points, images, probability=0.5)

    # Normalize with ImageNet stats (after color augmentation)
    images = normalize_imagenet(images)

    return images, gt_points


# =============================================================================
# Real Data Loading (NuScenes-like format)
# =============================================================================

def create_real_dataset(
    data_dir: str,
    split: str,
    batch_size: int,
    sequence_length: int,
    num_cameras: int = 6,
    image_height: int = 256,
    image_width: int = 704,
    num_points: int = 20,
    max_gt_elements: int = 30,
    augment: bool = True,
) -> tf.data.Dataset:
    """
    Create dataset from real data stored in TFRecord format.

    Expected TFRecord structure per example:
        - images: serialized float32 tensor (num_cameras, H, W, 3)
        - intrinsics: serialized float32 (num_cameras, 3, 3)
        - extrinsics: serialized float32 (num_cameras, 4, 4)
        - ego_motion: serialized float32 (4, 4)
        - gt_classes: serialized int32 (max_gt,)
        - gt_points: serialized float32 (max_gt, K, 2)
        - num_gt: int64 scalar

    Args:
        data_dir: path to data directory containing TFRecords
        split: 'train' or 'val'
        batch_size: batch size
        sequence_length: number of consecutive frames per sequence

    Returns:
        tf.data.Dataset of temporal sequences
    """
    tfrecord_pattern = os.path.join(data_dir, split, "*.tfrecord")
    filenames = tf.io.gfile.glob(tfrecord_pattern)

    if not filenames:
        raise FileNotFoundError(
            f"No TFRecord files found at {tfrecord_pattern}. "
            f"Use --synthetic for testing without data."
        )

    feature_description = {
        "images": tf.io.FixedLenFeature([], tf.string),
        "intrinsics": tf.io.FixedLenFeature([], tf.string),
        "extrinsics": tf.io.FixedLenFeature([], tf.string),
        "ego_motion": tf.io.FixedLenFeature([], tf.string),
        "gt_classes": tf.io.FixedLenFeature([], tf.string),
        "gt_points": tf.io.FixedLenFeature([], tf.string),
        "num_gt": tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_single_frame(serialized):
        example = tf.io.parse_single_example(serialized, feature_description)
        images = tf.io.parse_tensor(example["images"], out_type=tf.float32)
        images = tf.reshape(images, [num_cameras, image_height, image_width, 3])
        intrinsics = tf.io.parse_tensor(example["intrinsics"], out_type=tf.float32)
        intrinsics = tf.reshape(intrinsics, [num_cameras, 3, 3])
        extrinsics = tf.io.parse_tensor(example["extrinsics"], out_type=tf.float32)
        extrinsics = tf.reshape(extrinsics, [num_cameras, 4, 4])
        ego_motion = tf.io.parse_tensor(example["ego_motion"], out_type=tf.float32)
        ego_motion = tf.reshape(ego_motion, [4, 4])
        gt_classes = tf.io.parse_tensor(example["gt_classes"], out_type=tf.int32)
        gt_classes = tf.reshape(gt_classes, [max_gt_elements])
        gt_points = tf.io.parse_tensor(example["gt_points"], out_type=tf.float32)
        gt_points = tf.reshape(gt_points, [max_gt_elements, num_points, 2])
        num_gt = tf.cast(example["num_gt"], tf.int32)
        return images, intrinsics, extrinsics, ego_motion, gt_classes, gt_points, num_gt

    # Read raw dataset
    raw_dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=tf.data.AUTOTUNE
    )
    parsed_dataset = raw_dataset.map(_parse_single_frame, num_parallel_calls=tf.data.AUTOTUNE)

    # Window into temporal sequences
    windowed = parsed_dataset.window(sequence_length, shift=1, drop_remainder=True)

    def _flatten_window(*datasets):
        """Convert nested datasets from window() into stacked tensors."""
        results = []
        for ds in datasets:
            results.append(ds.batch(sequence_length))
        return tf.data.Dataset.zip(tuple(results))

    sequenced = windowed.flat_map(
        lambda *ds: tf.data.Dataset.zip(tuple(d.batch(sequence_length) for d in ds))
    )

    # Apply augmentation
    if augment:
        def _augment(images, intrinsics, extrinsics, ego_motion, gt_classes, gt_points, num_gts):
            images, gt_points = augment_sequence(images, gt_points, training=True)
            return images, intrinsics, extrinsics, ego_motion, gt_classes, gt_points, num_gts

        sequenced = sequenced.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = sequenced.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# =============================================================================
# Training Utilities
# =============================================================================

def compute_gradient_norm(gradients: List[tf.Tensor]) -> tf.Tensor:
    """Compute global gradient L2 norm."""
    norms = []
    for g in gradients:
        if g is not None:
            norms.append(tf.reduce_sum(tf.square(tf.cast(g, tf.float32))))
    if not norms:
        return tf.constant(0.0)
    return tf.sqrt(tf.add_n(norms))


# =============================================================================
# Training Step
# =============================================================================

def create_train_step(
    model: StreamMapNet,
    optimizer: tf.keras.optimizers.Optimizer,
    cls_weight: float,
    pts_weight: float,
    dir_weight: float,
    focal_alpha: float,
    focal_gamma: float,
    max_grad_norm: float,
    use_mixed_precision: bool,
):
    """
    Create a distributed training step function.

    Returns a function that processes one temporal frame within a sequence.
    """

    @tf.function
    def train_step_single_frame(
        images, intrinsics, extrinsics, ego_motion,
        gt_classes, gt_points, num_gts,
    ) -> Dict[str, tf.Tensor]:
        """
        Train on a single temporal frame.

        Args:
            images: (B, 6, H, W, 3)
            intrinsics: (B, 6, 3, 3)
            extrinsics: (B, 6, 4, 4)
            ego_motion: (B, 4, 4)
            gt_classes: (B, M_max)
            gt_points: (B, M_max, K, 2)
            num_gts: (B,)

        Returns:
            Dict of loss values.
        """
        with tf.GradientTape() as tape:
            predictions = model(
                {
                    "images": images,
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "ego_motion": ego_motion,
                },
                training=True,
            )

            losses = compute_loss(
                predictions=predictions,
                gt_classes=gt_classes,
                gt_points=gt_points,
                num_gts=num_gts,
                cls_weight=cls_weight,
                pts_weight=pts_weight,
                dir_weight=dir_weight,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )

            total_loss = losses["total_loss"]

            # Scale loss for mixed precision
            if use_mixed_precision:
                total_loss = optimizer.get_scaled_loss(total_loss)

        # Compute and apply gradients
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        if use_mixed_precision:
            gradients = optimizer.get_unscaled_gradients(gradients)

        # Gradient clipping
        if max_grad_norm > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)

        optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute gradient norm for logging
        grad_norm = compute_gradient_norm(gradients)
        losses["grad_norm"] = grad_norm

        return losses

    return train_step_single_frame


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    """Main training function."""

    # -------------------------------------------------------------------------
    # Setup distributed strategy
    # -------------------------------------------------------------------------
    if args.num_gpus > 1:
        devices = [f"/gpu:{i}" for i in range(args.num_gpus)]
        strategy = tf.distribute.MirroredStrategy(devices=devices)
    elif args.num_gpus == 1:
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    else:
        # CPU fallback
        strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])

    print(f"Number of replicas: {strategy.num_replicas_in_sync}")

    # -------------------------------------------------------------------------
    # Mixed precision setup
    # -------------------------------------------------------------------------
    use_mixed_precision = args.mixed_precision
    if use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision enabled: compute={policy.compute_dtype}, "
              f"variable={policy.variable_dtype}")

    # -------------------------------------------------------------------------
    # Model config
    # -------------------------------------------------------------------------
    model_config = dict(DEFAULT_CONFIG)
    model_config.update({
        "num_cameras": args.num_cameras,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "num_queries": args.num_queries,
        "num_classes": args.num_classes,
        "num_points": args.num_points,
        "temporal_queue_len": args.sequence_length,
        "decoder_layers": args.decoder_layers,
    })

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    if args.synthetic:
        print("Using synthetic dataset for testing...")
        train_dataset = create_synthetic_dataset(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_cameras=args.num_cameras,
            image_height=args.image_height,
            image_width=args.image_width,
            num_queries=args.num_queries,
            num_classes=args.num_classes,
            num_points=args.num_points,
            max_gt_elements=args.max_gt_elements,
            num_samples=args.num_train_samples,
        )
        val_dataset = create_synthetic_dataset(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_cameras=args.num_cameras,
            image_height=args.image_height,
            image_width=args.image_width,
            num_queries=args.num_queries,
            num_classes=args.num_classes,
            num_points=args.num_points,
            max_gt_elements=args.max_gt_elements,
            num_samples=args.num_val_samples,
        )
    else:
        train_dataset = create_real_dataset(
            data_dir=args.data_dir,
            split="train",
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_cameras=args.num_cameras,
            image_height=args.image_height,
            image_width=args.image_width,
            num_points=args.num_points,
            max_gt_elements=args.max_gt_elements,
            augment=True,
        )
        val_dataset = create_real_dataset(
            data_dir=args.data_dir,
            split="val",
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_cameras=args.num_cameras,
            image_height=args.image_height,
            image_width=args.image_width,
            num_points=args.num_points,
            max_gt_elements=args.max_gt_elements,
            augment=False,
        )

    # Distribute datasets
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # -------------------------------------------------------------------------
    # Compute total training steps for LR schedule
    # -------------------------------------------------------------------------
    steps_per_epoch = args.num_train_samples // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    # -------------------------------------------------------------------------
    # Build model, optimizer, and checkpoint within strategy scope
    # -------------------------------------------------------------------------
    with strategy.scope():
        # Model
        model = StreamMapNet(config=model_config)

        # Learning rate schedule
        lr_schedule = WarmupCosineDecaySchedule(
            peak_lr=args.learning_rate,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            total_steps=total_steps,
        )

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        )

        # Wrap optimizer for mixed precision loss scaling
        if use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        # Epoch counter variable (for checkpoint resume)
        epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="epoch")
        step_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
        best_val_loss = tf.Variable(float("inf"), trainable=False, dtype=tf.float32, name="best_val_loss")

        # Checkpoint
        checkpoint = tf.train.Checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch_var,
            global_step=step_var,
            best_val_loss=best_val_loss,
        )

    # -------------------------------------------------------------------------
    # Checkpoint manager and restoration
    # -------------------------------------------------------------------------
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=args.keep_checkpoints
    )

    # Resume from checkpoint if available
    if args.resume or ckpt_manager.latest_checkpoint:
        restore_path = args.resume if args.resume else ckpt_manager.latest_checkpoint
        if restore_path:
            status = checkpoint.restore(restore_path)
            # Allow partial restoration (model might have new layers)
            status.expect_partial()
            print(f"Restored from checkpoint: {restore_path}")
            print(f"  Resuming from epoch {epoch_var.numpy()}, "
                  f"global step {step_var.numpy()}")
    else:
        print("Starting training from scratch.")

    # -------------------------------------------------------------------------
    # TensorBoard setup
    # -------------------------------------------------------------------------
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # -------------------------------------------------------------------------
    # Create distributed training step
    # -------------------------------------------------------------------------
    train_step_fn = create_train_step(
        model=model,
        optimizer=optimizer,
        cls_weight=args.cls_weight,
        pts_weight=args.pts_weight,
        dir_weight=args.dir_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        max_grad_norm=args.max_grad_norm,
        use_mixed_precision=use_mixed_precision,
    )

    # -------------------------------------------------------------------------
    # Distributed step wrappers
    # -------------------------------------------------------------------------
    @tf.function
    def distributed_train_step(images, intrinsics, extrinsics, ego_motion,
                               gt_classes, gt_points, num_gts):
        per_replica_losses = strategy.run(
            train_step_fn,
            args=(images, intrinsics, extrinsics, ego_motion,
                  gt_classes, gt_points, num_gts),
        )
        # Reduce across replicas
        reduced = {}
        for key, val in per_replica_losses.items():
            reduced[key] = strategy.reduce(
                tf.distribute.ReduceOp.MEAN, val, axis=None
            )
        return reduced

    @tf.function
    def distributed_val_step(images, intrinsics, extrinsics, ego_motion,
                             gt_classes, gt_points, num_gts):
        def _val_forward(images, intrinsics, extrinsics, ego_motion,
                         gt_classes, gt_points, num_gts):
            predictions = model(
                {
                    "images": images,
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "ego_motion": ego_motion,
                },
                training=False,
            )
            losses = compute_loss(
                predictions=predictions,
                gt_classes=gt_classes,
                gt_points=gt_points,
                num_gts=num_gts,
                cls_weight=args.cls_weight,
                pts_weight=args.pts_weight,
                dir_weight=args.dir_weight,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
            )
            return losses

        per_replica_losses = strategy.run(
            _val_forward,
            args=(images, intrinsics, extrinsics, ego_motion,
                  gt_classes, gt_points, num_gts),
        )
        reduced = {}
        for key, val in per_replica_losses.items():
            reduced[key] = strategy.reduce(
                tf.distribute.ReduceOp.MEAN, val, axis=None
            )
        return reduced

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    start_epoch = int(epoch_var.numpy())
    global_step = int(step_var.numpy())

    print(f"\n{'='*70}")
    print(f"Starting StreamMapNet Training")
    print(f"{'='*70}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {strategy.num_replicas_in_sync} replicas")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Learning rate: {args.learning_rate} (peak), {args.min_lr} (min)")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Mixed precision: {use_mixed_precision}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        epoch_losses = {"total_loss": 0.0, "cls_loss": 0.0, "pts_loss": 0.0, "dir_loss": 0.0}
        num_batches = 0

        # --- Training ---
        for batch_data in train_dist_dataset:
            (images_seq, intrinsics_seq, extrinsics_seq, ego_motion_seq,
             gt_classes_seq, gt_points_seq, num_gts_seq) = batch_data

            # Reset temporal state at the start of each new sequence
            model.reset_temporal_state()

            # Process each frame in the temporal sequence
            seq_len = args.sequence_length
            for t in range(seq_len):
                # Slice temporal dimension
                # Each tensor has shape (B, T, ...) after batching
                images_t = images_seq[:, t]
                intrinsics_t = intrinsics_seq[:, t]
                extrinsics_t = extrinsics_seq[:, t]
                ego_motion_t = ego_motion_seq[:, t]
                gt_classes_t = gt_classes_seq[:, t]
                gt_points_t = gt_points_seq[:, t]
                num_gts_t = num_gts_seq[:, t]

                # Distributed train step
                step_losses = distributed_train_step(
                    images_t, intrinsics_t, extrinsics_t, ego_motion_t,
                    gt_classes_t, gt_points_t, num_gts_t,
                )

                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += float(step_losses[key].numpy())

                global_step += 1
                step_var.assign(global_step)

                # TensorBoard logging per step
                with summary_writer.as_default():
                    tf.summary.scalar("train/total_loss", step_losses["total_loss"], step=global_step)
                    tf.summary.scalar("train/cls_loss", step_losses["cls_loss"], step=global_step)
                    tf.summary.scalar("train/pts_loss", step_losses["pts_loss"], step=global_step)
                    tf.summary.scalar("train/dir_loss", step_losses["dir_loss"], step=global_step)
                    tf.summary.scalar("train/grad_norm", step_losses["grad_norm"], step=global_step)
                    current_lr = optimizer.learning_rate
                    if callable(current_lr):
                        current_lr = current_lr(optimizer.iterations)
                    tf.summary.scalar("train/learning_rate", current_lr, step=global_step)

            num_batches += 1

            # Print progress periodically
            if num_batches % args.log_interval == 0:
                avg_loss = epoch_losses["total_loss"] / (num_batches * seq_len)
                print(f"  Epoch {epoch+1}/{args.epochs} | "
                      f"Batch {num_batches} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Step: {global_step}")

        # Compute epoch averages
        total_steps_epoch = num_batches * seq_len if num_batches > 0 else 1
        for key in epoch_losses:
            epoch_losses[key] /= total_steps_epoch

        epoch_time = time.time() - epoch_start_time

        # --- Validation ---
        val_losses = {"total_loss": 0.0, "cls_loss": 0.0, "pts_loss": 0.0, "dir_loss": 0.0}
        val_batches = 0

        for batch_data in val_dist_dataset:
            (images_seq, intrinsics_seq, extrinsics_seq, ego_motion_seq,
             gt_classes_seq, gt_points_seq, num_gts_seq) = batch_data

            model.reset_temporal_state()

            for t in range(args.sequence_length):
                images_t = images_seq[:, t]
                intrinsics_t = intrinsics_seq[:, t]
                extrinsics_t = extrinsics_seq[:, t]
                ego_motion_t = ego_motion_seq[:, t]
                gt_classes_t = gt_classes_seq[:, t]
                gt_points_t = gt_points_seq[:, t]
                num_gts_t = num_gts_seq[:, t]

                step_losses = distributed_val_step(
                    images_t, intrinsics_t, extrinsics_t, ego_motion_t,
                    gt_classes_t, gt_points_t, num_gts_t,
                )

                for key in val_losses:
                    val_losses[key] += float(step_losses[key].numpy())

            val_batches += 1

        total_val_steps = val_batches * args.sequence_length if val_batches > 0 else 1
        for key in val_losses:
            val_losses[key] /= total_val_steps

        # Log validation metrics
        with summary_writer.as_default():
            tf.summary.scalar("val/total_loss", val_losses["total_loss"], step=global_step)
            tf.summary.scalar("val/cls_loss", val_losses["cls_loss"], step=global_step)
            tf.summary.scalar("val/pts_loss", val_losses["pts_loss"], step=global_step)
            tf.summary.scalar("val/dir_loss", val_losses["dir_loss"], step=global_step)

        # Print epoch summary
        print(f"\n  Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.1f}s")
        print(f"    Train Loss: {epoch_losses['total_loss']:.4f} "
              f"(cls={epoch_losses['cls_loss']:.4f}, "
              f"pts={epoch_losses['pts_loss']:.4f}, "
              f"dir={epoch_losses['dir_loss']:.4f})")
        print(f"    Val Loss:   {val_losses['total_loss']:.4f} "
              f"(cls={val_losses['cls_loss']:.4f}, "
              f"pts={val_losses['pts_loss']:.4f}, "
              f"dir={val_losses['dir_loss']:.4f})")

        # --- Checkpointing ---
        epoch_var.assign(epoch + 1)

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            save_path = ckpt_manager.save()
            print(f"    Checkpoint saved: {save_path}")

        # Save best model based on validation loss
        current_val_loss = val_losses["total_loss"]
        if current_val_loss < float(best_val_loss.numpy()):
            best_val_loss.assign(current_val_loss)
            best_ckpt_path = os.path.join(checkpoint_dir, "best")
            checkpoint.write(best_ckpt_path)
            print(f"    New best model saved (val_loss={current_val_loss:.4f})")

        print()

    # -------------------------------------------------------------------------
    # Final save
    # -------------------------------------------------------------------------
    final_path = ckpt_manager.save()
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    print(f"Best validation loss: {float(best_val_loss.numpy()):.4f}")
    print(f"Logs saved to: {log_dir}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="StreamMapNet Training Script (TensorFlow 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to dataset directory containing TFRecords")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for testing")
    parser.add_argument("--num_train_samples", type=int, default=200,
                        help="Number of synthetic training samples")
    parser.add_argument("--num_val_samples", type=int, default=50,
                        help="Number of synthetic validation samples")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory for checkpoints, logs, and outputs")

    # Model architecture
    parser.add_argument("--num_cameras", type=int, default=6,
                        help="Number of surround-view cameras")
    parser.add_argument("--image_height", type=int, default=256,
                        help="Input image height")
    parser.add_argument("--image_width", type=int, default=704,
                        help="Input image width")
    parser.add_argument("--num_queries", type=int, default=100,
                        help="Number of map element queries")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of map element classes")
    parser.add_argument("--num_points", type=int, default=20,
                        help="Number of points per polyline")
    parser.add_argument("--max_gt_elements", type=int, default=30,
                        help="Maximum ground truth elements per frame")
    parser.add_argument("--decoder_layers", type=int, default=6,
                        help="Number of transformer decoder layers")

    # Training
    parser.add_argument("--epochs", type=int, default=24,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-replica batch size")
    parser.add_argument("--sequence_length", type=int, default=3,
                        help="Number of temporal frames per sequence")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for distributed training")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate after cosine decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of linear warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--max_grad_norm", type=float, default=35.0,
                        help="Maximum gradient norm for clipping (0 to disable)")

    # Loss weights
    parser.add_argument("--cls_weight", type=float, default=2.0,
                        help="Classification loss weight")
    parser.add_argument("--pts_weight", type=float, default=5.0,
                        help="Point regression loss weight")
    parser.add_argument("--dir_weight", type=float, default=0.005,
                        help="Direction loss weight")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter")

    # Mixed precision
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision (float16) training")

    # Checkpointing and logging
    parser.add_argument("--save_every", type=int, default=2,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--keep_checkpoints", type=int, default=5,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Print training progress every N batches")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    args = parse_args()

    # Set memory growth for GPUs to avoid OOM
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if gpus:
        print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    else:
        print("No GPUs found. Training on CPU.")
        if args.num_gpus > 0:
            args.num_gpus = 0

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training
    train(args)
