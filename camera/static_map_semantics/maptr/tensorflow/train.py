"""
TensorFlow 2 training script for MapTR: Structured Modeling and Learning for
Online Vectorized HD Map Construction.

Trains the MapTR model on the nuScenes dataset for vectorized HD map element
prediction from multi-camera surround-view images.

Usage:
    python train.py --data_root /path/to/nuscenes --epochs 24 --batch_size 4
"""

import argparse
import math
import os
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from model import MapTRModel


# =============================================================================
# Constants
# =============================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_CAMERAS = 6
IMG_HEIGHT = 480
IMG_WIDTH = 800
NUM_CLASSES = 3  # ped_crossing, divider, boundary
NUM_QUERIES = 50
NUM_POINTS_PER_INSTANCE = 20
BEV_H = 200
BEV_W = 100
BEV_X_RANGE = 60.0  # meters
BEV_Y_RANGE = 30.0  # meters

CLASS_NAMES = ["ped_crossing", "divider", "boundary"]


# =============================================================================
# Data Pipeline
# =============================================================================


class NuScenesMapDataset:
    """tf.data.Dataset pipeline for nuScenes HD map training data.

    Each sample contains:
        - 6 camera images (480x800x3)
        - Camera intrinsics [6, 3, 3]
        - Camera extrinsics [6, 4, 4]
        - Annotations: class labels and polyline points normalized to [0, 1]
    """

    def __init__(
        self,
        data_root,
        split="train",
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_cameras=NUM_CAMERAS,
        max_instances=NUM_QUERIES,
        num_points=NUM_POINTS_PER_INSTANCE,
        augment=True,
    ):
        self.data_root = data_root
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        self.num_cameras = num_cameras
        self.max_instances = max_instances
        self.num_points = num_points
        self.augment = augment

        self.sample_tokens = self._load_sample_tokens()

    def _load_sample_tokens(self):
        """Load sample tokens from the split file."""
        split_file = os.path.join(self.data_root, "splits", f"{self.split}.txt")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                tokens = [line.strip() for line in f if line.strip()]
            return tokens
        # Fallback: scan annotation directory
        ann_dir = os.path.join(self.data_root, "annotations", self.split)
        if os.path.isdir(ann_dir):
            tokens = [
                f.replace(".npz", "")
                for f in sorted(os.listdir(ann_dir))
                if f.endswith(".npz")
            ]
            return tokens
        raise FileNotFoundError(
            f"Cannot find split file or annotations at {self.data_root}"
        )

    def __len__(self):
        return len(self.sample_tokens)

    def _load_sample(self, idx):
        """Load a single sample (called via tf.py_function)."""
        idx = idx.numpy()
        token = self.sample_tokens[idx]

        # Load images
        images = np.zeros(
            (self.num_cameras, self.img_height, self.img_width, 3), dtype=np.float32
        )
        for cam_idx in range(self.num_cameras):
            img_path = os.path.join(
                self.data_root, "images", self.split, token, f"cam_{cam_idx}.jpg"
            )
            if os.path.exists(img_path):
                img_raw = tf.io.read_file(img_path)
                img = tf.image.decode_jpeg(img_raw, channels=3)
                img = tf.image.resize(img, [self.img_height, self.img_width])
                images[cam_idx] = img.numpy() / 255.0
            else:
                images[cam_idx] = np.random.rand(
                    self.img_height, self.img_width, 3
                ).astype(np.float32)

        # Load annotations
        ann_path = os.path.join(
            self.data_root, "annotations", self.split, f"{token}.npz"
        )
        if os.path.exists(ann_path):
            ann_data = np.load(ann_path, allow_pickle=True)
            class_labels = ann_data.get("class_labels", np.array([], dtype=np.int32))
            polylines = ann_data.get("polylines", None)
            intrinsics = ann_data.get(
                "intrinsics",
                np.zeros((self.num_cameras, 3, 3), dtype=np.float32),
            )
            extrinsics = ann_data.get(
                "extrinsics",
                np.zeros((self.num_cameras, 4, 4), dtype=np.float32),
            )
        else:
            class_labels = np.array([], dtype=np.int32)
            polylines = None
            intrinsics = np.eye(3, dtype=np.float32)[None].repeat(
                self.num_cameras, axis=0
            )
            extrinsics = np.eye(4, dtype=np.float32)[None].repeat(
                self.num_cameras, axis=0
            )

        # Pad/truncate annotations to fixed size
        gt_labels = np.full(self.max_instances, -1, dtype=np.int32)
        gt_points = np.zeros(
            (self.max_instances, self.num_points, 2), dtype=np.float32
        )

        if class_labels is not None and len(class_labels) > 0:
            num_valid = min(len(class_labels), self.max_instances)
            gt_labels[:num_valid] = class_labels[:num_valid]

            if polylines is not None:
                for i in range(num_valid):
                    pts = polylines[i] if i < len(polylines) else np.zeros((0, 2))
                    pts = self._sample_or_pad_points(pts)
                    gt_points[i] = pts

        # Data augmentation
        if self.augment:
            images, gt_points, gt_labels = self._augment(
                images, gt_points, gt_labels
            )

        # Normalize images with ImageNet statistics
        images = (images - IMAGENET_MEAN) / IMAGENET_STD

        intrinsics = intrinsics.astype(np.float32)
        extrinsics = extrinsics.astype(np.float32)

        return (
            images.astype(np.float32),
            intrinsics,
            extrinsics,
            gt_labels.astype(np.int32),
            gt_points.astype(np.float32),
        )

    def _sample_or_pad_points(self, pts):
        """Resample or pad polyline points to a fixed number."""
        if len(pts) == 0:
            return np.zeros((self.num_points, 2), dtype=np.float32)

        if len(pts) == self.num_points:
            return pts.astype(np.float32)

        if len(pts) > self.num_points:
            # Uniformly subsample
            indices = np.linspace(0, len(pts) - 1, self.num_points, dtype=int)
            return pts[indices].astype(np.float32)

        # Interpolate to get exactly num_points
        # Parameterize by cumulative arc length
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
        total_length = cum_lengths[-1]

        if total_length < 1e-8:
            # Degenerate case: all points coincide
            return np.tile(pts[0:1], (self.num_points, 1)).astype(np.float32)

        # Sample uniformly along arc length
        target_lengths = np.linspace(0, total_length, self.num_points)
        sampled = np.zeros((self.num_points, 2), dtype=np.float32)
        for i, t in enumerate(target_lengths):
            seg_idx = np.searchsorted(cum_lengths, t, side="right") - 1
            seg_idx = np.clip(seg_idx, 0, len(pts) - 2)
            seg_len = seg_lengths[seg_idx]
            if seg_len < 1e-8:
                sampled[i] = pts[seg_idx]
            else:
                alpha = (t - cum_lengths[seg_idx]) / seg_len
                alpha = np.clip(alpha, 0.0, 1.0)
                sampled[i] = pts[seg_idx] * (1 - alpha) + pts[seg_idx + 1] * alpha

        return sampled

    def _augment(self, images, gt_points, gt_labels):
        """Apply data augmentation: photometric distortion and random flip."""
        # Random photometric distortion (per-camera)
        if np.random.rand() < 0.5:
            # Brightness
            delta = np.random.uniform(-0.1, 0.1)
            images = images + delta

        if np.random.rand() < 0.5:
            # Contrast
            factor = np.random.uniform(0.8, 1.2)
            mean_vals = images.mean(axis=(1, 2, 3), keepdims=True)
            images = (images - mean_vals) * factor + mean_vals

        if np.random.rand() < 0.5:
            # Saturation (approximate on RGB)
            gray = images.mean(axis=-1, keepdims=True)
            factor = np.random.uniform(0.8, 1.2)
            images = gray + (images - gray) * factor

        if np.random.rand() < 0.5:
            # Hue shift (approximate)
            shift = np.random.uniform(-0.05, 0.05)
            images = images + shift * np.array([[[[-1, 0.5, 0.5]]]])

        images = np.clip(images, 0.0, 1.0)

        # Random horizontal flip
        if np.random.rand() < 0.5:
            images = images[:, :, ::-1, :]  # Flip width axis
            # Flip x-coordinates of points (normalized [0,1])
            valid_mask = gt_labels >= 0
            gt_points[valid_mask, :, 0] = 1.0 - gt_points[valid_mask, :, 0]
            # Reverse point order for flipped instances
            gt_points[valid_mask] = gt_points[valid_mask][:, ::-1, :]

        return images, gt_points, gt_labels

    def build_dataset(self, batch_size, shuffle=True, num_parallel_calls=None):
        """Build a tf.data.Dataset from this loader."""
        if num_parallel_calls is None:
            num_parallel_calls = tf.data.AUTOTUNE

        num_samples = len(self.sample_tokens)
        indices = tf.data.Dataset.range(num_samples)

        if shuffle:
            indices = indices.shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)

        output_signature = (
            tf.TensorSpec(
                shape=(self.num_cameras, self.img_height, self.img_width, 3),
                dtype=tf.float32,
            ),
            tf.TensorSpec(shape=(self.num_cameras, 3, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(self.num_cameras, 4, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(self.max_instances,), dtype=tf.int32),
            tf.TensorSpec(
                shape=(self.max_instances, self.num_points, 2), dtype=tf.float32
            ),
        )

        dataset = indices.map(
            lambda idx: tf.py_function(
                self._load_sample,
                [idx],
                [tf.float32, tf.float32, tf.float32, tf.int32, tf.float32],
            ),
            num_parallel_calls=num_parallel_calls,
        )

        # Set shapes explicitly after py_function
        def set_shapes(images, intrinsics, extrinsics, labels, points):
            images.set_shape(
                [self.num_cameras, self.img_height, self.img_width, 3]
            )
            intrinsics.set_shape([self.num_cameras, 3, 3])
            extrinsics.set_shape([self.num_cameras, 4, 4])
            labels.set_shape([self.max_instances])
            points.set_shape([self.max_instances, self.num_points, 2])
            return images, intrinsics, extrinsics, labels, points

        dataset = dataset.map(set_shapes, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


# =============================================================================
# Loss Functions
# =============================================================================


def focal_loss(pred_logits, targets, gamma=2.0, alpha=0.25):
    """Compute focal loss for classification.

    Args:
        pred_logits: [B, N, num_classes] predicted class logits.
        targets: [B, N] integer class targets (-1 for no-object).

    Returns:
        Scalar focal loss value.
    """
    num_classes = pred_logits.shape[-1]
    # Convert targets to one-hot; -1 maps to background (all zeros)
    valid_mask = tf.cast(targets >= 0, tf.float32)  # [B, N]
    safe_targets = tf.maximum(targets, 0)
    one_hot = tf.one_hot(safe_targets, num_classes)  # [B, N, C]

    # Sigmoid focal loss
    prob = tf.sigmoid(pred_logits)
    ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=one_hot, logits=pred_logits
    )  # [B, N, C]

    p_t = one_hot * prob + (1.0 - one_hot) * (1.0 - prob)
    focal_weight = (1.0 - p_t) ** gamma

    alpha_t = one_hot * alpha + (1.0 - one_hot) * (1.0 - alpha)

    loss = alpha_t * focal_weight * ce_loss  # [B, N, C]
    loss = tf.reduce_sum(loss, axis=-1)  # [B, N]

    # Mask out invalid positions
    loss = loss * valid_mask
    num_valid = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    return tf.reduce_sum(loss) / num_valid


def points_loss(pred_points, gt_points, valid_mask):
    """Compute smooth L1 loss for predicted polyline points.

    Args:
        pred_points: [B, N, P, 2] predicted point coordinates.
        gt_points: [B, N, P, 2] ground truth point coordinates.
        valid_mask: [B, N] boolean mask for valid instances.

    Returns:
        Scalar smooth L1 loss.
    """
    diff = pred_points - gt_points  # [B, N, P, 2]
    abs_diff = tf.abs(diff)
    smooth_l1 = tf.where(abs_diff < 1.0, 0.5 * abs_diff ** 2, abs_diff - 0.5)
    loss_per_instance = tf.reduce_mean(smooth_l1, axis=[-2, -1])  # [B, N]

    mask = tf.cast(valid_mask, tf.float32)
    loss_per_instance = loss_per_instance * mask
    num_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(loss_per_instance) / num_valid


def direction_loss(pred_points, gt_points, valid_mask):
    """Compute direction cosine loss between edge vectors.

    Encourages predicted edge vectors (between consecutive points) to have
    the same direction as ground truth edge vectors.

    Args:
        pred_points: [B, N, P, 2] predicted point coordinates.
        gt_points: [B, N, P, 2] ground truth point coordinates.
        valid_mask: [B, N] boolean mask for valid instances.

    Returns:
        Scalar direction loss (1 - cosine similarity).
    """
    # Edge vectors: difference between consecutive points
    pred_edges = pred_points[:, :, 1:, :] - pred_points[:, :, :-1, :]  # [B, N, P-1, 2]
    gt_edges = gt_points[:, :, 1:, :] - gt_points[:, :, :-1, :]  # [B, N, P-1, 2]

    # Normalize edge vectors
    pred_norm = tf.maximum(
        tf.sqrt(tf.reduce_sum(pred_edges ** 2, axis=-1, keepdims=True)), 1e-6
    )
    gt_norm = tf.maximum(
        tf.sqrt(tf.reduce_sum(gt_edges ** 2, axis=-1, keepdims=True)), 1e-6
    )

    pred_dir = pred_edges / pred_norm
    gt_dir = gt_edges / gt_norm

    # Cosine similarity
    cos_sim = tf.reduce_sum(pred_dir * gt_dir, axis=-1)  # [B, N, P-1]
    dir_loss = 1.0 - cos_sim  # [B, N, P-1]

    # Average over points dimension
    dir_loss_per_instance = tf.reduce_mean(dir_loss, axis=-1)  # [B, N]

    mask = tf.cast(valid_mask, tf.float32)
    dir_loss_per_instance = dir_loss_per_instance * mask
    num_valid = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(dir_loss_per_instance) / num_valid


def hungarian_matching(pred_logits, pred_points, gt_labels, gt_points):
    """Perform Hungarian matching between predictions and ground truth.

    Uses scipy.optimize.linear_sum_assignment for optimal bipartite matching.

    Args:
        pred_logits: [B, N, num_classes] predicted class logits.
        pred_points: [B, N, P, 2] predicted points.
        gt_labels: [B, M] ground truth class labels (-1 for padding).
        gt_points: [B, M, P, 2] ground truth points.

    Returns:
        matched_pred_logits: [B, N, num_classes] reordered predictions.
        matched_pred_points: [B, N, P, 2] reordered predictions.
        matched_gt_labels: [B, N] matched ground truth labels.
        matched_gt_points: [B, N, P, 2] matched ground truth points.
        valid_mask: [B, N] boolean mask for matched valid instances.
    """
    batch_size = pred_logits.shape[0]
    num_queries = pred_logits.shape[1]
    num_classes = pred_logits.shape[2]
    num_points = pred_points.shape[2]

    pred_logits_np = pred_logits.numpy()
    pred_points_np = pred_points.numpy()
    gt_labels_np = gt_labels.numpy()
    gt_points_np = gt_points.numpy()

    matched_gt_labels = np.full((batch_size, num_queries), -1, dtype=np.int32)
    matched_gt_points = np.zeros(
        (batch_size, num_queries, num_points, 2), dtype=np.float32
    )
    valid_mask = np.zeros((batch_size, num_queries), dtype=np.float32)

    cost_class_weight = 2.0
    cost_pts_weight = 5.0
    cost_dir_weight = 0.005

    for b in range(batch_size):
        # Find valid ground truth instances
        gt_valid = gt_labels_np[b] >= 0
        num_gt = int(gt_valid.sum())

        if num_gt == 0:
            continue

        gt_idx = np.where(gt_valid)[0]
        b_gt_labels = gt_labels_np[b, gt_idx]  # [num_gt]
        b_gt_points = gt_points_np[b, gt_idx]  # [num_gt, P, 2]

        # Compute classification cost
        b_pred_logits = pred_logits_np[b]  # [N, C]
        pred_prob = 1.0 / (1.0 + np.exp(-b_pred_logits))  # sigmoid
        # Cost: negative probability of the target class
        cost_class = -pred_prob[:, b_gt_labels]  # [N, num_gt]

        # Compute points cost (L1 distance)
        b_pred_points = pred_points_np[b]  # [N, P, 2]
        # Pairwise L1 between each query and each GT
        pred_expand = b_pred_points[:, None, :, :]  # [N, 1, P, 2]
        gt_expand = b_gt_points[None, :, :, :]  # [1, num_gt, P, 2]
        cost_pts = np.abs(pred_expand - gt_expand).mean(axis=(-2, -1))  # [N, num_gt]

        # Compute direction cost
        pred_edges = b_pred_points[:, 1:, :] - b_pred_points[:, :-1, :]  # [N, P-1, 2]
        gt_edges = b_gt_points[:, 1:, :] - b_gt_points[:, :-1, :]  # [num_gt, P-1, 2]

        pred_edge_norm = np.maximum(
            np.sqrt((pred_edges ** 2).sum(axis=-1, keepdims=True)), 1e-6
        )
        gt_edge_norm = np.maximum(
            np.sqrt((gt_edges ** 2).sum(axis=-1, keepdims=True)), 1e-6
        )

        pred_dir = pred_edges / pred_edge_norm  # [N, P-1, 2]
        gt_dir = gt_edges / gt_edge_norm  # [num_gt, P-1, 2]

        # Pairwise direction cost
        pred_dir_expand = pred_dir[:, None, :, :]  # [N, 1, P-1, 2]
        gt_dir_expand = gt_dir[None, :, :, :]  # [1, num_gt, P-1, 2]
        cos_sim = (pred_dir_expand * gt_dir_expand).sum(axis=-1)  # [N, num_gt, P-1]
        cost_dir = (1.0 - cos_sim).mean(axis=-1)  # [N, num_gt]

        # Total cost matrix
        cost_matrix = (
            cost_class_weight * cost_class
            + cost_pts_weight * cost_pts
            + cost_dir_weight * cost_dir
        )

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Assign matched ground truth to prediction slots
        for pred_idx, gt_local_idx in zip(row_ind, col_ind):
            matched_gt_labels[b, pred_idx] = b_gt_labels[gt_local_idx]
            matched_gt_points[b, pred_idx] = b_gt_points[gt_local_idx]
            valid_mask[b, pred_idx] = 1.0

    matched_gt_labels = tf.constant(matched_gt_labels, dtype=tf.int32)
    matched_gt_points = tf.constant(matched_gt_points, dtype=tf.float32)
    valid_mask = tf.constant(valid_mask, dtype=tf.float32)

    return matched_gt_labels, matched_gt_points, valid_mask


def compute_total_loss(pred_logits, pred_points, gt_labels, gt_points):
    """Compute total training loss with Hungarian matching.

    Args:
        pred_logits: [B, N, num_classes] predicted class logits.
        pred_points: [B, N, P, 2] predicted polyline points.
        gt_labels: [B, M] ground truth class labels.
        gt_points: [B, M, P, 2] ground truth polyline points.

    Returns:
        total_loss: Scalar total loss.
        loss_dict: Dictionary of individual loss components.
    """
    # Hungarian matching (runs in numpy via py_function in graph mode)
    matched_gt_labels, matched_gt_points, valid_mask = tf.py_function(
        hungarian_matching,
        [pred_logits, pred_points, gt_labels, gt_points],
        [tf.int32, tf.float32, tf.float32],
    )

    # Restore shapes after py_function
    batch_size = tf.shape(pred_logits)[0]
    num_queries = pred_logits.shape[1] or tf.shape(pred_logits)[1]
    num_pts = pred_points.shape[2] or tf.shape(pred_points)[2]

    matched_gt_labels = tf.ensure_shape(matched_gt_labels, [None, NUM_QUERIES])
    matched_gt_points = tf.ensure_shape(
        matched_gt_points, [None, NUM_QUERIES, NUM_POINTS_PER_INSTANCE, 2]
    )
    valid_mask = tf.ensure_shape(valid_mask, [None, NUM_QUERIES])

    # Classification loss (focal loss)
    cls_weight = 2.0
    loss_cls = focal_loss(pred_logits, matched_gt_labels, gamma=2.0, alpha=0.25)

    # Points regression loss (smooth L1)
    pts_weight = 5.0
    loss_pts = points_loss(pred_points, matched_gt_points, valid_mask)

    # Direction loss
    dir_weight = 0.005
    loss_dir = direction_loss(pred_points, matched_gt_points, valid_mask)

    # Total loss
    total_loss = cls_weight * loss_cls + pts_weight * loss_pts + dir_weight * loss_dir

    loss_dict = {
        "loss_cls": loss_cls,
        "loss_pts": loss_pts,
        "loss_dir": loss_dir,
        "total_loss": total_loss,
    }

    return total_loss, loss_dict


# =============================================================================
# Learning Rate Schedule
# =============================================================================


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing learning rate schedule with linear warmup.

    Args:
        base_lr: Base learning rate after warmup.
        total_steps: Total number of training steps (T_max * steps_per_epoch).
        warmup_steps: Number of warmup steps.
        warmup_ratio: Starting learning rate ratio during warmup.
        eta_min: Minimum learning rate at the end of cosine decay.
    """

    def __init__(
        self,
        base_lr=6e-4,
        total_steps=1000,
        warmup_steps=500,
        warmup_ratio=0.001,
        eta_min=6e-6,
    ):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.eta_min = eta_min

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Linear warmup phase
        warmup_lr = self.base_lr * (
            self.warmup_ratio + (1.0 - self.warmup_ratio) * step / warmup_steps
        )

        # Cosine decay phase
        progress = (step - warmup_steps) / tf.maximum(
            total_steps - warmup_steps, 1.0
        )
        cosine_lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1.0 + tf.cos(math.pi * progress)
        )

        # Select based on current step
        lr = tf.where(step < warmup_steps, warmup_lr, cosine_lr)
        return lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "eta_min": self.eta_min,
        }


# =============================================================================
# Training Loop
# =============================================================================


def create_optimizer(model, lr_schedule, backbone_lr_mult=0.1, weight_decay=0.01):
    """Create AdamW optimizer with differential learning rates.

    Backbone parameters use a reduced learning rate (lr * backbone_lr_mult).

    Args:
        model: The MapTR model.
        lr_schedule: Learning rate schedule instance.
        backbone_lr_mult: Multiplier for backbone learning rate.
        weight_decay: Weight decay coefficient.

    Returns:
        Tuple of (backbone_optimizer, head_optimizer) for parameter groups.
    """
    # Main optimizer for non-backbone parameters
    main_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=35.0,
    )

    # Backbone optimizer with reduced learning rate
    backbone_schedule = CosineDecayWithWarmup(
        base_lr=lr_schedule.base_lr * backbone_lr_mult,
        total_steps=lr_schedule.total_steps,
        warmup_steps=lr_schedule.warmup_steps,
        warmup_ratio=lr_schedule.warmup_ratio,
        eta_min=lr_schedule.eta_min * backbone_lr_mult,
    )

    backbone_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=backbone_schedule,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=35.0,
    )

    return main_optimizer, backbone_optimizer


def split_variables(model):
    """Split model variables into backbone and head groups.

    Args:
        model: The MapTR model.

    Returns:
        Tuple of (backbone_vars, head_vars).
    """
    backbone_vars = []
    head_vars = []

    for var in model.trainable_variables:
        if "backbone" in var.name.lower():
            backbone_vars.append(var)
        else:
            head_vars.append(var)

    return backbone_vars, head_vars


def clip_gradients(gradients, max_norm=35.0):
    """Clip gradients by global norm.

    Args:
        gradients: List of gradient tensors.
        max_norm: Maximum gradient norm.

    Returns:
        Clipped gradients.
    """
    gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
    return gradients


def train_step(
    model,
    images,
    intrinsics,
    extrinsics,
    gt_labels,
    gt_points,
    main_optimizer,
    backbone_optimizer,
    backbone_vars,
    head_vars,
    loss_scale,
):
    """Execute a single training step.

    Args:
        model: The MapTR model.
        images: [B, 6, H, W, 3] input images.
        intrinsics: [B, 6, 3, 3] camera intrinsics.
        extrinsics: [B, 6, 4, 4] camera extrinsics.
        gt_labels: [B, M] ground truth labels.
        gt_points: [B, M, P, 2] ground truth points.
        main_optimizer: Optimizer for head parameters.
        backbone_optimizer: Optimizer for backbone parameters.
        backbone_vars: List of backbone trainable variables.
        head_vars: List of head trainable variables.
        loss_scale: Mixed precision loss scale factor.

    Returns:
        loss_dict: Dictionary of loss values.
    """
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        pred_logits, pred_points = model(
            images, intrinsics, extrinsics, training=True
        )

        # Compute loss
        total_loss, loss_dict = compute_total_loss(
            pred_logits, pred_points, gt_labels, gt_points
        )

        # Scale loss for mixed precision
        scaled_loss = total_loss * loss_scale

    # Compute and apply gradients for head variables
    if head_vars:
        head_grads = tape.gradient(scaled_loss, head_vars)
        # Unscale gradients
        head_grads = [g / loss_scale if g is not None else g for g in head_grads]
        # Filter out None gradients
        valid_head = [
            (g, v) for g, v in zip(head_grads, head_vars) if g is not None
        ]
        if valid_head:
            grads, vars_ = zip(*valid_head)
            grads = clip_gradients(list(grads), max_norm=35.0)
            main_optimizer.apply_gradients(zip(grads, vars_))

    # Compute and apply gradients for backbone variables
    if backbone_vars:
        backbone_grads = tape.gradient(scaled_loss, backbone_vars)
        # Unscale gradients
        backbone_grads = [
            g / loss_scale if g is not None else g for g in backbone_grads
        ]
        valid_backbone = [
            (g, v) for g, v in zip(backbone_grads, backbone_vars) if g is not None
        ]
        if valid_backbone:
            grads, vars_ = zip(*valid_backbone)
            grads = clip_gradients(list(grads), max_norm=35.0)
            backbone_optimizer.apply_gradients(zip(grads, vars_))

    del tape
    return loss_dict


@tf.function
def distributed_train_step(
    strategy,
    model,
    images,
    intrinsics,
    extrinsics,
    gt_labels,
    gt_points,
    main_optimizer,
    backbone_optimizer,
    backbone_vars,
    head_vars,
    loss_scale,
):
    """Distributed training step across multiple GPUs.

    Args:
        strategy: tf.distribute.Strategy instance.
        model: The MapTR model.
        images, intrinsics, extrinsics, gt_labels, gt_points: Batch data.
        main_optimizer: Head optimizer.
        backbone_optimizer: Backbone optimizer.
        backbone_vars: Backbone variables.
        head_vars: Head variables.
        loss_scale: Loss scale factor.

    Returns:
        Reduced loss dictionary.
    """
    per_replica_losses = strategy.run(
        train_step,
        args=(
            model,
            images,
            intrinsics,
            extrinsics,
            gt_labels,
            gt_points,
            main_optimizer,
            backbone_optimizer,
            backbone_vars,
            head_vars,
            loss_scale,
        ),
    )

    # Reduce losses across replicas
    reduced_losses = {}
    for key, value in per_replica_losses.items():
        reduced_losses[key] = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, value, axis=None
        )
    return reduced_losses


def evaluate(model, val_dataset, strategy, max_batches=50):
    """Run evaluation on the validation set.

    Args:
        model: The MapTR model.
        val_dataset: Validation tf.data.Dataset.
        strategy: Distribution strategy.
        max_batches: Maximum number of batches to evaluate.

    Returns:
        Dictionary of average evaluation metrics.
    """
    total_losses = {"loss_cls": 0.0, "loss_pts": 0.0, "loss_dir": 0.0, "total_loss": 0.0}
    num_batches = 0

    for batch in val_dataset:
        if num_batches >= max_batches:
            break

        images, intrinsics, extrinsics, gt_labels, gt_points = batch

        # Forward pass without gradient
        pred_logits, pred_points = model(
            images, intrinsics, extrinsics, training=False
        )

        _, loss_dict = compute_total_loss(
            pred_logits, pred_points, gt_labels, gt_points
        )

        for key in total_losses:
            total_losses[key] += float(loss_dict[key])
        num_batches += 1

    if num_batches > 0:
        for key in total_losses:
            total_losses[key] /= num_batches

    return total_losses


def train(args):
    """Main training function.

    Args:
        args: Parsed command-line arguments.
    """
    # Setup distribution strategy
    if args.num_gpus > 1:
        devices = [f"/gpu:{i}" for i in range(args.num_gpus)]
        strategy = tf.distribute.MirroredStrategy(devices=devices)
    else:
        strategy = tf.distribute.MirroredStrategy()

    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Mixed precision setup
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled: compute=float16, variables=float32")
        loss_scale = tf.constant(512.0, dtype=tf.float32)
    else:
        loss_scale = tf.constant(1.0, dtype=tf.float32)

    # Build datasets
    print("Building training dataset...")
    train_data = NuScenesMapDataset(
        data_root=args.data_root,
        split="train",
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_cameras=NUM_CAMERAS,
        max_instances=NUM_QUERIES,
        num_points=NUM_POINTS_PER_INSTANCE,
        augment=True,
    )

    print("Building validation dataset...")
    val_data = NuScenesMapDataset(
        data_root=args.data_root,
        split="val",
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_cameras=NUM_CAMERAS,
        max_instances=NUM_QUERIES,
        num_points=NUM_POINTS_PER_INSTANCE,
        augment=False,
    )

    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    steps_per_epoch = max(len(train_data) // global_batch_size, 1)
    total_steps = steps_per_epoch * args.epochs

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")

    # Build distributed datasets
    train_dataset = train_data.build_dataset(
        batch_size=global_batch_size, shuffle=True
    )
    val_dataset = val_data.build_dataset(
        batch_size=global_batch_size, shuffle=False
    )

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # Create model and optimizers within strategy scope
    with strategy.scope():
        model = MapTRModel(
            num_classes=NUM_CLASSES,
            num_queries=NUM_QUERIES,
            num_points=NUM_POINTS_PER_INSTANCE,
            bev_h=BEV_H,
            bev_w=BEV_W,
            bev_x_range=BEV_X_RANGE,
            bev_y_range=BEV_Y_RANGE,
            num_cameras=NUM_CAMERAS,
            img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH,
        )

        # Learning rate schedule
        lr_schedule = CosineDecayWithWarmup(
            base_lr=args.lr,
            total_steps=total_steps,
            warmup_steps=args.warmup_iters,
            warmup_ratio=args.warmup_ratio,
            eta_min=args.eta_min,
        )

        # Create optimizers
        main_optimizer, backbone_optimizer = create_optimizer(
            model,
            lr_schedule,
            backbone_lr_mult=args.backbone_lr_mult,
            weight_decay=args.weight_decay,
        )

        # Build model with dummy input to initialize variables
        dummy_images = tf.zeros(
            [1, NUM_CAMERAS, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32
        )
        dummy_intrinsics = tf.zeros([1, NUM_CAMERAS, 3, 3], dtype=tf.float32)
        dummy_extrinsics = tf.zeros([1, NUM_CAMERAS, 4, 4], dtype=tf.float32)
        _ = model(dummy_images, dummy_intrinsics, dummy_extrinsics, training=False)

        # Split variables into parameter groups
        backbone_vars, head_vars = split_variables(model)
        print(f"Backbone parameters: {len(backbone_vars)}")
        print(f"Head parameters: {len(head_vars)}")
        print(
            f"Total trainable parameters: "
            f"{sum(np.prod(v.shape) for v in model.trainable_variables):,}"
        )

        # Checkpoint management
        checkpoint = tf.train.Checkpoint(
            model=model,
            main_optimizer=main_optimizer,
            backbone_optimizer=backbone_optimizer,
            epoch=tf.Variable(0, dtype=tf.int64),
            global_step=tf.Variable(0, dtype=tf.int64),
        )

        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=args.checkpoint_dir,
            max_to_keep=args.max_checkpoints,
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        global_step = 0

        if args.resume:
            if os.path.isdir(args.resume):
                latest = tf.train.latest_checkpoint(args.resume)
                if latest:
                    checkpoint.restore(latest)
                    start_epoch = int(checkpoint.epoch.numpy())
                    global_step = int(checkpoint.global_step.numpy())
                    print(f"Resumed from checkpoint: {latest}")
                    print(f"  Epoch: {start_epoch}, Step: {global_step}")
            else:
                checkpoint.restore(args.resume)
                start_epoch = int(checkpoint.epoch.numpy())
                global_step = int(checkpoint.global_step.numpy())
                print(f"Resumed from checkpoint: {args.resume}")
        elif checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            start_epoch = int(checkpoint.epoch.numpy())
            global_step = int(checkpoint.global_step.numpy())
            print(
                f"Auto-resumed from latest checkpoint: "
                f"{checkpoint_manager.latest_checkpoint}"
            )

    # TensorBoard writer
    log_dir = os.path.join(args.checkpoint_dir, "logs")
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        epoch_losses = {
            "loss_cls": 0.0,
            "loss_pts": 0.0,
            "loss_dir": 0.0,
            "total_loss": 0.0,
        }
        step_in_epoch = 0

        for batch in train_dist_dataset:
            images, intrinsics, extrinsics, gt_labels, gt_points = batch

            # Training step
            loss_dict = distributed_train_step(
                strategy,
                model,
                images,
                intrinsics,
                extrinsics,
                gt_labels,
                gt_points,
                main_optimizer,
                backbone_optimizer,
                backbone_vars,
                head_vars,
                loss_scale,
            )

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += float(loss_dict[key])

            global_step += 1
            step_in_epoch += 1

            # Log to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar("train/total_loss", loss_dict["total_loss"], step=global_step)
                tf.summary.scalar("train/loss_cls", loss_dict["loss_cls"], step=global_step)
                tf.summary.scalar("train/loss_pts", loss_dict["loss_pts"], step=global_step)
                tf.summary.scalar("train/loss_dir", loss_dict["loss_dir"], step=global_step)
                current_lr = main_optimizer.learning_rate(global_step)
                tf.summary.scalar("train/learning_rate", current_lr, step=global_step)

            # Progress logging
            if global_step % args.log_interval == 0:
                avg_total = epoch_losses["total_loss"] / step_in_epoch
                current_lr_val = float(main_optimizer.learning_rate(global_step))
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Step [{step_in_epoch}/{steps_per_epoch}] "
                    f"Loss: {float(loss_dict['total_loss']):.4f} "
                    f"(cls: {float(loss_dict['loss_cls']):.4f}, "
                    f"pts: {float(loss_dict['loss_pts']):.4f}, "
                    f"dir: {float(loss_dict['loss_dir']):.4f}) "
                    f"LR: {current_lr_val:.2e}"
                )

        # End of epoch
        epoch_time = time.time() - epoch_start_time
        if step_in_epoch > 0:
            for key in epoch_losses:
                epoch_losses[key] /= step_in_epoch

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} completed in {epoch_time:.1f}s"
        )
        print(
            f"  Avg Loss: {epoch_losses['total_loss']:.4f} "
            f"(cls: {epoch_losses['loss_cls']:.4f}, "
            f"pts: {epoch_losses['loss_pts']:.4f}, "
            f"dir: {epoch_losses['loss_dir']:.4f})"
        )

        # Periodic evaluation
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            print("  Running validation...")
            val_losses = evaluate(model, val_dataset, strategy, max_batches=50)
            print(
                f"  Val Loss: {val_losses['total_loss']:.4f} "
                f"(cls: {val_losses['loss_cls']:.4f}, "
                f"pts: {val_losses['loss_pts']:.4f}, "
                f"dir: {val_losses['loss_dir']:.4f})"
            )

            # Log validation metrics
            with summary_writer.as_default():
                tf.summary.scalar(
                    "val/total_loss", val_losses["total_loss"], step=global_step
                )
                tf.summary.scalar(
                    "val/loss_cls", val_losses["loss_cls"], step=global_step
                )
                tf.summary.scalar(
                    "val/loss_pts", val_losses["loss_pts"], step=global_step
                )
                tf.summary.scalar(
                    "val/loss_dir", val_losses["loss_dir"], step=global_step
                )

            # Save best model
            if val_losses["total_loss"] < best_val_loss:
                best_val_loss = val_losses["total_loss"]
                best_path = os.path.join(args.checkpoint_dir, "best_model")
                model.save_weights(best_path)
                print(f"  New best model saved (val_loss: {best_val_loss:.4f})")

        # Save checkpoint
        checkpoint.epoch.assign(epoch + 1)
        checkpoint.global_step.assign(global_step)
        save_path = checkpoint_manager.save()
        print(f"  Checkpoint saved: {save_path}\n")

    # Final summary
    print("=" * 70)
    print("Training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final checkpoint: {checkpoint_manager.latest_checkpoint}")
    print(f"  TensorBoard logs: {log_dir}")
    print("=" * 70)


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MapTR model for vectorized HD map construction"
    )

    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the nuScenes dataset",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=24,
        help="Number of training epochs (default: 24)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=6e-4,
        help="Base learning rate (default: 6e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW (default: 0.01)",
    )
    parser.add_argument(
        "--backbone_lr_mult",
        type=float,
        default=0.1,
        help="Backbone learning rate multiplier (default: 0.1)",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=6e-6,
        help="Minimum learning rate for cosine annealing (default: 6e-6)",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=500,
        help="Number of warmup iterations (default: 500)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.001,
        help="Warmup starting ratio (default: 0.001)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=35.0,
        help="Maximum gradient norm for clipping (default: 35.0)",
    )

    # Hardware
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Enable mixed precision training (default: True)",
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Disable mixed precision training",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/maptr",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep (default: 5)",
    )

    # Logging
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Log every N steps (default: 50)",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=3,
        help="Evaluate every N epochs (default: 3)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Handle mixed precision flag
    if args.no_mixed_precision:
        args.mixed_precision = False

    return args


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Main entry point for MapTR training."""
    args = parse_args()

    # Set random seeds for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 70)
    print("MapTR Training Configuration")
    print("=" * 70)
    print(f"  Data root:          {args.data_root}")
    print(f"  Batch size/GPU:     {args.batch_size}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Weight decay:       {args.weight_decay}")
    print(f"  Backbone LR mult:   {args.backbone_lr_mult}")
    print(f"  Warmup iters:       {args.warmup_iters}")
    print(f"  Warmup ratio:       {args.warmup_ratio}")
    print(f"  Eta min:            {args.eta_min}")
    print(f"  Max grad norm:      {args.max_grad_norm}")
    print(f"  Num GPUs:           {args.num_gpus}")
    print(f"  Mixed precision:    {args.mixed_precision}")
    print(f"  Checkpoint dir:     {args.checkpoint_dir}")
    print(f"  Resume:             {args.resume}")
    print(f"  Seed:               {args.seed}")
    print(f"  Image size:         {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"  Num cameras:        {NUM_CAMERAS}")
    print(f"  Num classes:        {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")
    print(f"  Num queries:        {NUM_QUERIES}")
    print(f"  Points/instance:    {NUM_POINTS_PER_INSTANCE}")
    print(f"  BEV grid:           {BEV_H}x{BEV_W}")
    print(f"  BEV range:          {BEV_X_RANGE}m x {BEV_Y_RANGE}m")
    print("=" * 70 + "\n")

    # Start training
    train(args)


if __name__ == "__main__":
    main()
