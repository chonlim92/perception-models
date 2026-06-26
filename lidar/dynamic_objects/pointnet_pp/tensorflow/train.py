"""
PointNet++ Training Script for 3D Point Cloud Tasks.

Supports classification, detection, and segmentation tasks with:
- tf.data.Dataset pipeline with augmentation
- Cosine decay learning rate schedule
- Mixed precision training (optional)
- TensorBoard logging
- Checkpoint management with best model saving

Usage:
    python train.py --task classification --data_dir ./data --epochs 200
    python train.py --task detection --data_dir ./data --epochs 300 --mixed_precision
    python train.py --task segmentation --data_dir ./data --epochs 250 --num_points 4096
"""

import argparse
import os
import time
import glob

import numpy as np
import tensorflow as tf

from model import (
    PointNetPPClassification,
    PointNetPPDetection,
    PointNetPPSegmentation,
)


# ===========================================================================
# Argument Parsing
# ===========================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='PointNet++ Training Script for 3D Point Cloud Tasks'
    )

    parser.add_argument(
        '--task', type=str, required=True,
        choices=['classification', 'detection', 'segmentation'],
        help='Task type: classification, detection, or segmentation'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to the dataset directory containing .npy files'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Training batch size (default: 16)'
    )
    parser.add_argument(
        '--epochs', type=int, default=200,
        help='Number of training epochs (default: 200)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Initial learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--num_points', type=int, default=1024,
        help='Number of points per sample (default: 1024)'
    )
    parser.add_argument(
        '--num_classes', type=int, default=10,
        help='Number of output classes (default: 10)'
    )
    parser.add_argument(
        '--mixed_precision', action='store_true',
        help='Enable mixed precision training (float16)'
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, default='./checkpoints',
        help='Directory to save checkpoints (default: ./checkpoints)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./logs',
        help='Directory for TensorBoard logs (default: ./logs)'
    )

    return parser.parse_args()


# ===========================================================================
# Data Augmentation Functions
# ===========================================================================


def random_rotate_z(point_cloud):
    """Randomly rotate point cloud around the z-axis.

    Args:
        point_cloud: (N, 3) float32 tensor of point coordinates.

    Returns:
        Rotated point cloud with same shape.
    """
    angle = tf.random.uniform([], 0.0, 2.0 * np.pi)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    rotation_matrix = tf.stack([
        tf.stack([cos_a, -sin_a, 0.0]),
        tf.stack([sin_a,  cos_a, 0.0]),
        tf.stack([0.0,    0.0,   1.0]),
    ])  # (3, 3)

    rotated = tf.matmul(point_cloud, tf.transpose(rotation_matrix))
    return rotated


def random_scale(point_cloud):
    """Randomly scale point cloud uniformly between 0.8 and 1.2.

    Args:
        point_cloud: (N, 3) float32 tensor of point coordinates.

    Returns:
        Scaled point cloud with same shape.
    """
    scale = tf.random.uniform([], 0.8, 1.2)
    return point_cloud * scale


def random_jitter(point_cloud, sigma=0.01, clip=0.05):
    """Add random Gaussian jitter to point cloud coordinates.

    Args:
        point_cloud: (N, 3) float32 tensor of point coordinates.
        sigma: Standard deviation of Gaussian noise.
        clip: Maximum absolute value of noise.

    Returns:
        Jittered point cloud with same shape.
    """
    noise = tf.clip_by_value(
        tf.random.normal(tf.shape(point_cloud), stddev=sigma),
        -clip, clip
    )
    return point_cloud + noise


def random_point_dropout(point_cloud, max_dropout_ratio=0.875):
    """Randomly drop points by duplicating remaining points.

    With some probability, a fraction of points are replaced by copies of
    other points (effectively dropping them while maintaining tensor shape).

    Args:
        point_cloud: (N, 3) float32 tensor of point coordinates.
        max_dropout_ratio: Maximum fraction of points to drop.

    Returns:
        Point cloud with same shape, some points duplicated.
    """
    dropout_ratio = tf.random.uniform([], 0.0, max_dropout_ratio)
    num_points = tf.shape(point_cloud)[0]
    num_drop = tf.cast(
        tf.cast(num_points, tf.float32) * dropout_ratio, tf.int32
    )

    # Generate random mask for which points to drop
    random_vals = tf.random.uniform([num_points])
    _, drop_indices = tf.math.top_k(random_vals, k=num_drop)

    # Replace dropped points with the first point
    first_point = point_cloud[0:1]  # (1, 3)
    first_point_tiled = tf.tile(first_point, [num_drop, 1])  # (num_drop, 3)

    # Use tensor_scatter_nd_update to replace
    drop_indices_2d = tf.expand_dims(drop_indices, axis=1)
    point_cloud = tf.tensor_scatter_nd_update(
        point_cloud, drop_indices_2d, first_point_tiled
    )

    return point_cloud


def augment_point_cloud(point_cloud):
    """Apply all augmentation transforms to a point cloud.

    Args:
        point_cloud: (N, 3) float32 tensor of point coordinates.

    Returns:
        Augmented point cloud with same shape.
    """
    point_cloud = random_rotate_z(point_cloud)
    point_cloud = random_scale(point_cloud)
    point_cloud = random_jitter(point_cloud)
    point_cloud = random_point_dropout(point_cloud)
    return point_cloud


# ===========================================================================
# Dataset Pipeline
# ===========================================================================


def load_classification_dataset(data_dir, num_points, batch_size, split='train'):
    """Load classification dataset from .npy files.

    Expected directory structure:
        data_dir/
            train/
                points/   - .npy files with shape (N, 3+)
                labels/   - .npy files with scalar labels
            val/
                points/
                labels/

    Args:
        data_dir: Root directory of the dataset.
        num_points: Number of points to sample per cloud.
        batch_size: Batch size.
        split: 'train' or 'val'.

    Returns:
        tf.data.Dataset yielding (point_cloud, label) tuples.
    """
    points_dir = os.path.join(data_dir, split, 'points')
    labels_dir = os.path.join(data_dir, split, 'labels')

    point_files = sorted(glob.glob(os.path.join(points_dir, '*.npy')))
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.npy')))

    assert len(point_files) == len(label_files), (
        f"Mismatch: {len(point_files)} point files vs {len(label_files)} label files"
    )
    assert len(point_files) > 0, f"No .npy files found in {points_dir}"

    def generator():
        for pf, lf in zip(point_files, label_files):
            points = np.load(pf).astype(np.float32)
            label = np.load(lf).astype(np.int32)

            # Take only xyz coordinates (first 3 columns)
            if points.shape[-1] > 3:
                points = points[:, :3]

            # Sample or pad to num_points
            n = points.shape[0]
            if n >= num_points:
                indices = np.random.choice(n, num_points, replace=False)
                points = points[indices]
            else:
                # Pad by repeating points
                pad_indices = np.random.choice(n, num_points - n, replace=True)
                points = np.concatenate([points, points[pad_indices]], axis=0)

            yield points, label.item() if label.ndim > 0 else int(label)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(num_points, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )

    if split == 'train':
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(
            lambda pts, lbl: (augment_point_cloud(pts), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size, drop_remainder=(split == 'train'))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_detection_dataset(data_dir, num_points, batch_size, split='train'):
    """Load detection dataset from .npy files.

    Expected directory structure:
        data_dir/
            train/
                points/   - .npy files with shape (N, 3+)
                boxes/    - .npy files with shape (M, 7) [x,y,z,w,h,l,yaw]
                box_labels/ - .npy files with shape (M,) class labels per box
            val/
                points/
                boxes/
                box_labels/

    For training we produce fixed-size outputs by padding/truncating to
    a maximum number of boxes (128).

    Args:
        data_dir: Root directory of the dataset.
        num_points: Number of points to sample per cloud.
        batch_size: Batch size.
        split: 'train' or 'val'.

    Returns:
        tf.data.Dataset yielding (point_cloud, boxes, box_labels, num_valid_boxes).
    """
    points_dir = os.path.join(data_dir, split, 'points')
    boxes_dir = os.path.join(data_dir, split, 'boxes')
    box_labels_dir = os.path.join(data_dir, split, 'box_labels')

    point_files = sorted(glob.glob(os.path.join(points_dir, '*.npy')))
    box_files = sorted(glob.glob(os.path.join(boxes_dir, '*.npy')))
    label_files = sorted(glob.glob(os.path.join(box_labels_dir, '*.npy')))

    assert len(point_files) == len(box_files) == len(label_files), (
        f"File count mismatch in {data_dir}/{split}"
    )
    assert len(point_files) > 0, f"No .npy files found in {points_dir}"

    max_boxes = 128

    def generator():
        for pf, bf, lf in zip(point_files, box_files, label_files):
            points = np.load(pf).astype(np.float32)
            boxes = np.load(bf).astype(np.float32)
            box_labels = np.load(lf).astype(np.int32)

            if points.shape[-1] > 3:
                points = points[:, :3]

            # Sample/pad points
            n = points.shape[0]
            if n >= num_points:
                indices = np.random.choice(n, num_points, replace=False)
                points = points[indices]
            else:
                pad_indices = np.random.choice(n, num_points - n, replace=True)
                points = np.concatenate([points, points[pad_indices]], axis=0)

            # Pad/truncate boxes to max_boxes
            num_valid = min(boxes.shape[0], max_boxes)
            padded_boxes = np.zeros((max_boxes, 7), dtype=np.float32)
            padded_labels = np.zeros((max_boxes,), dtype=np.int32)
            padded_boxes[:num_valid] = boxes[:num_valid]
            padded_labels[:num_valid] = box_labels[:num_valid]

            yield points, padded_boxes, padded_labels, np.int32(num_valid)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(num_points, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(max_boxes, 7), dtype=tf.float32),
            tf.TensorSpec(shape=(max_boxes,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )

    if split == 'train':
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.map(
            lambda pts, bxs, lbls, nv: (augment_point_cloud(pts), bxs, lbls, nv),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size, drop_remainder=(split == 'train'))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_segmentation_dataset(data_dir, num_points, batch_size, split='train'):
    """Load segmentation dataset from .npy files.

    Expected directory structure:
        data_dir/
            train/
                points/   - .npy files with shape (N, 3+)
                labels/   - .npy files with shape (N,) per-point labels
            val/
                points/
                labels/

    Args:
        data_dir: Root directory of the dataset.
        num_points: Number of points to sample per cloud.
        batch_size: Batch size.
        split: 'train' or 'val'.

    Returns:
        tf.data.Dataset yielding (point_cloud, per_point_labels) tuples.
    """
    points_dir = os.path.join(data_dir, split, 'points')
    labels_dir = os.path.join(data_dir, split, 'labels')

    point_files = sorted(glob.glob(os.path.join(points_dir, '*.npy')))
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.npy')))

    assert len(point_files) == len(label_files), (
        f"Mismatch: {len(point_files)} point files vs {len(label_files)} label files"
    )
    assert len(point_files) > 0, f"No .npy files found in {points_dir}"

    def generator():
        for pf, lf in zip(point_files, label_files):
            points = np.load(pf).astype(np.float32)
            labels = np.load(lf).astype(np.int32)

            if points.shape[-1] > 3:
                points = points[:, :3]

            # Sample or pad points and corresponding labels
            n = points.shape[0]
            if n >= num_points:
                indices = np.random.choice(n, num_points, replace=False)
                points = points[indices]
                labels = labels[indices]
            else:
                pad_indices = np.random.choice(n, num_points - n, replace=True)
                points = np.concatenate([points, points[pad_indices]], axis=0)
                labels = np.concatenate([labels, labels[pad_indices]], axis=0)

            yield points, labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(num_points, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_points,), dtype=tf.int32),
        )
    )

    if split == 'train':
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(
            lambda pts, lbls: (augment_point_cloud(pts), lbls),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size, drop_remainder=(split == 'train'))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ===========================================================================
# Loss Functions
# ===========================================================================


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance in detection.

    Args:
        y_true: (B, num_proposals) integer class labels.
        y_pred: (B, num_proposals, num_classes) logits.
        alpha: Weighting factor for rare classes.
        gamma: Focusing parameter to down-weight easy examples.

    Returns:
        Scalar mean focal loss.
    """
    num_classes = tf.shape(y_pred)[-1]
    y_true_onehot = tf.one_hot(y_true, depth=num_classes)  # (B, P, C)

    probs = tf.nn.softmax(y_pred, axis=-1)
    # Clip for numerical stability
    probs = tf.clip_by_value(probs, 1e-7, 1.0 - 1e-7)

    # Cross-entropy per class
    ce = -y_true_onehot * tf.math.log(probs)

    # Focal weight
    pt = tf.reduce_sum(probs * y_true_onehot, axis=-1)  # (B, P)
    focal_weight = alpha * tf.pow(1.0 - pt, gamma)  # (B, P)

    # Weighted cross-entropy
    loss = tf.reduce_sum(ce, axis=-1) * focal_weight  # (B, P)

    return tf.reduce_mean(loss)


def smooth_l1_loss(y_true, y_pred, sigma=1.0):
    """Smooth L1 (Huber-like) loss for bounding box regression.

    Args:
        y_true: (B, num_proposals, 7) ground truth boxes.
        y_pred: (B, num_proposals, 7) predicted boxes.
        sigma: Transition point between L1 and L2 regions.

    Returns:
        Scalar mean smooth L1 loss.
    """
    sigma_sq = sigma ** 2
    diff = y_pred - y_true
    abs_diff = tf.abs(diff)

    smooth_l1 = tf.where(
        abs_diff < 1.0 / sigma_sq,
        0.5 * sigma_sq * tf.square(diff),
        abs_diff - 0.5 / sigma_sq
    )

    return tf.reduce_mean(smooth_l1)


def detection_loss(pred_boxes, pred_cls, gt_boxes, gt_labels, num_valid,
                   box_weight=1.0, cls_weight=1.0):
    """Combined detection loss: smooth L1 for boxes + focal loss for classes.

    Only computes loss over valid (non-padded) ground truth boxes.

    Args:
        pred_boxes: (B, num_proposals, 7) predicted boxes.
        pred_cls: (B, num_proposals, num_classes) predicted class logits.
        gt_boxes: (B, max_boxes, 7) ground truth boxes (padded).
        gt_labels: (B, max_boxes) ground truth class labels (padded).
        num_valid: (B,) number of valid boxes per sample.
        box_weight: Weight for box regression loss.
        cls_weight: Weight for classification loss.

    Returns:
        total_loss: Scalar combined loss.
        box_loss: Scalar box regression loss.
        cls_loss: Scalar classification loss.
    """
    # Use the number of proposals from predictions for loss computation
    num_proposals = tf.shape(pred_boxes)[1]

    # Truncate ground truth to match proposal count for loss computation
    gt_boxes_trunc = gt_boxes[:, :num_proposals, :]
    gt_labels_trunc = gt_labels[:, :num_proposals]

    # Create mask for valid boxes
    batch_size = tf.shape(num_valid)[0]
    proposal_indices = tf.range(num_proposals, dtype=tf.int32)
    proposal_indices = tf.tile(
        tf.expand_dims(proposal_indices, 0), [batch_size, 1]
    )  # (B, num_proposals)
    valid_mask = tf.cast(
        proposal_indices < tf.expand_dims(num_valid, 1), tf.float32
    )  # (B, num_proposals)

    # Box regression loss (masked)
    box_diff = pred_boxes - gt_boxes_trunc  # (B, P, 7)
    abs_diff = tf.abs(box_diff)
    smooth_l1 = tf.where(
        abs_diff < 1.0,
        0.5 * tf.square(box_diff),
        abs_diff - 0.5
    )
    # Mean over box dims, masked over proposals
    smooth_l1_per_proposal = tf.reduce_mean(smooth_l1, axis=-1)  # (B, P)
    masked_box_loss = smooth_l1_per_proposal * valid_mask
    num_valid_total = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    box_loss = tf.reduce_sum(masked_box_loss) / num_valid_total

    # Classification focal loss
    cls_loss = focal_loss(gt_labels_trunc, pred_cls)

    total_loss = box_weight * box_loss + cls_weight * cls_loss

    return total_loss, box_loss, cls_loss


# ===========================================================================
# Training and Validation Steps
# ===========================================================================


def build_train_step(model, optimizer, task, num_classes):
    """Build the tf.function-decorated training step.

    Args:
        model: The PointNet++ model instance.
        optimizer: The optimizer instance.
        task: One of 'classification', 'detection', 'segmentation'.
        num_classes: Number of classes.

    Returns:
        train_step function.
    """
    if task == 'classification':
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )

        @tf.function
        def train_step(point_clouds, labels):
            with tf.GradientTape() as tape:
                logits = model(point_clouds, training=True)
                loss = loss_fn(labels, logits)
                # Scale loss for mixed precision
                scaled_loss = optimizer.get_scaled_loss(loss) if hasattr(
                    optimizer, 'get_scaled_loss'
                ) else loss

            if hasattr(optimizer, 'get_scaled_loss'):
                scaled_grads = tape.gradient(
                    scaled_loss, model.trainable_variables
                )
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )

            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(predictions, labels), tf.float32)
            )

            return loss, accuracy

    elif task == 'detection':

        @tf.function
        def train_step(point_clouds, gt_boxes, gt_labels, num_valid):
            with tf.GradientTape() as tape:
                pred_boxes, pred_cls = model(point_clouds, training=True)
                total_loss, box_loss, cls_loss = detection_loss(
                    pred_boxes, pred_cls, gt_boxes, gt_labels, num_valid
                )
                scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(
                    optimizer, 'get_scaled_loss'
                ) else total_loss

            if hasattr(optimizer, 'get_scaled_loss'):
                scaled_grads = tape.gradient(
                    scaled_loss, model.trainable_variables
                )
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(total_loss, model.trainable_variables)

            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )

            return total_loss, box_loss, cls_loss

    elif task == 'segmentation':
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )

        @tf.function
        def train_step(point_clouds, labels):
            with tf.GradientTape() as tape:
                logits = model(point_clouds, training=True)
                # logits: (B, N, C), labels: (B, N)
                loss = loss_fn(labels, logits)
                scaled_loss = optimizer.get_scaled_loss(loss) if hasattr(
                    optimizer, 'get_scaled_loss'
                ) else loss

            if hasattr(optimizer, 'get_scaled_loss'):
                scaled_grads = tape.gradient(
                    scaled_loss, model.trainable_variables
                )
                grads = optimizer.get_unscaled_gradients(scaled_grads)
            else:
                grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )

            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(predictions, labels), tf.float32)
            )

            return loss, accuracy

    return train_step


def build_val_step(model, task, num_classes):
    """Build the tf.function-decorated validation step.

    Args:
        model: The PointNet++ model instance.
        task: One of 'classification', 'detection', 'segmentation'.
        num_classes: Number of classes.

    Returns:
        val_step function.
    """
    if task == 'classification':
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )

        @tf.function
        def val_step(point_clouds, labels):
            logits = model(point_clouds, training=False)
            loss = loss_fn(labels, logits)

            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(predictions, labels), tf.float32)
            )

            return loss, accuracy

    elif task == 'detection':

        @tf.function
        def val_step(point_clouds, gt_boxes, gt_labels, num_valid):
            pred_boxes, pred_cls = model(point_clouds, training=False)
            total_loss, box_loss, cls_loss = detection_loss(
                pred_boxes, pred_cls, gt_boxes, gt_labels, num_valid
            )
            return total_loss, box_loss, cls_loss

    elif task == 'segmentation':
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )

        @tf.function
        def val_step(point_clouds, labels):
            logits = model(point_clouds, training=False)
            loss = loss_fn(labels, logits)

            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(predictions, labels), tf.float32)
            )

            return loss, accuracy

    return val_step


# ===========================================================================
# Main Training Loop
# ===========================================================================


def main():
    """Main training entry point."""
    args = parse_args()

    # -----------------------------------------------------------------------
    # Mixed precision setup
    # -----------------------------------------------------------------------
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("[INFO] Mixed precision training enabled (mixed_float16)")

    # -----------------------------------------------------------------------
    # Create directories
    # -----------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    if args.task == 'classification':
        model = PointNetPPClassification(num_classes=args.num_classes)
        print(f"[INFO] Created PointNet++ Classification model "
              f"(num_classes={args.num_classes})")
    elif args.task == 'detection':
        model = PointNetPPDetection(
            num_classes=args.num_classes, num_proposals=128
        )
        print(f"[INFO] Created PointNet++ Detection model "
              f"(num_classes={args.num_classes}, num_proposals=128)")
    elif args.task == 'segmentation':
        model = PointNetPPSegmentation(num_classes=args.num_classes)
        print(f"[INFO] Created PointNet++ Segmentation model "
              f"(num_classes={args.num_classes})")

    # -----------------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------------
    if args.task == 'classification':
        train_ds = load_classification_dataset(
            args.data_dir, args.num_points, args.batch_size, split='train'
        )
        val_ds = load_classification_dataset(
            args.data_dir, args.num_points, args.batch_size, split='val'
        )
    elif args.task == 'detection':
        train_ds = load_detection_dataset(
            args.data_dir, args.num_points, args.batch_size, split='train'
        )
        val_ds = load_detection_dataset(
            args.data_dir, args.num_points, args.batch_size, split='val'
        )
    elif args.task == 'segmentation':
        train_ds = load_segmentation_dataset(
            args.data_dir, args.num_points, args.batch_size, split='train'
        )
        val_ds = load_segmentation_dataset(
            args.data_dir, args.num_points, args.batch_size, split='val'
        )

    # -----------------------------------------------------------------------
    # Compute training steps for LR schedule
    # -----------------------------------------------------------------------
    # Estimate steps per epoch from dataset (count batches in one pass)
    steps_per_epoch = 0
    for _ in train_ds:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * args.epochs
    print(f"[INFO] Steps per epoch: {steps_per_epoch}, "
          f"Total steps: {total_steps}")

    # -----------------------------------------------------------------------
    # Learning rate schedule and optimizer
    # -----------------------------------------------------------------------
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=1e-6  # Minimum learning rate ratio
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    if args.mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # -----------------------------------------------------------------------
    # Checkpoint management
    # -----------------------------------------------------------------------
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=args.checkpoint_dir,
        max_to_keep=5
    )

    # Restore from latest checkpoint if available
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"[INFO] Restored from checkpoint: "
              f"{checkpoint_manager.latest_checkpoint}")

    # Best model directory
    best_model_dir = os.path.join(args.checkpoint_dir, 'best')
    os.makedirs(best_model_dir, exist_ok=True)
    best_checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    best_checkpoint_manager = tf.train.CheckpointManager(
        best_checkpoint,
        directory=best_model_dir,
        max_to_keep=1
    )

    # -----------------------------------------------------------------------
    # TensorBoard writer
    # -----------------------------------------------------------------------
    train_log_dir = os.path.join(args.log_dir, 'train')
    val_log_dir = os.path.join(args.log_dir, 'val')
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)

    # -----------------------------------------------------------------------
    # Build train/val step functions
    # -----------------------------------------------------------------------
    train_step = build_train_step(model, optimizer, args.task, args.num_classes)
    val_step = build_val_step(model, args.task, args.num_classes)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_metric = float('inf') if args.task == 'detection' else 0.0
    global_step = 0

    print(f"\n[INFO] Starting training for {args.epochs} epochs...")
    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Num points: {args.num_points}")
    print(f"[INFO] Learning rate: {args.lr}")
    print(f"[INFO] Checkpoint dir: {args.checkpoint_dir}")
    print(f"[INFO] Log dir: {args.log_dir}")
    print("=" * 70)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # -------------------------------------------------------------------
        # Training
        # -------------------------------------------------------------------
        train_loss_accum = 0.0
        train_metric_accum = 0.0
        train_steps = 0

        if args.task == 'classification':
            for batch in train_ds:
                point_clouds, labels = batch
                loss, accuracy = train_step(point_clouds, labels)
                train_loss_accum += loss.numpy()
                train_metric_accum += accuracy.numpy()
                train_steps += 1
                global_step += 1

        elif args.task == 'detection':
            for batch in train_ds:
                point_clouds, gt_boxes, gt_labels, num_valid = batch
                total_loss, box_loss, cls_loss = train_step(
                    point_clouds, gt_boxes, gt_labels, num_valid
                )
                train_loss_accum += total_loss.numpy()
                train_metric_accum += box_loss.numpy()
                train_steps += 1
                global_step += 1

        elif args.task == 'segmentation':
            for batch in train_ds:
                point_clouds, labels = batch
                loss, accuracy = train_step(point_clouds, labels)
                train_loss_accum += loss.numpy()
                train_metric_accum += accuracy.numpy()
                train_steps += 1
                global_step += 1

        avg_train_loss = train_loss_accum / max(train_steps, 1)
        avg_train_metric = train_metric_accum / max(train_steps, 1)

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------
        val_loss_accum = 0.0
        val_metric_accum = 0.0
        val_steps = 0

        if args.task == 'classification':
            for batch in val_ds:
                point_clouds, labels = batch
                loss, accuracy = val_step(point_clouds, labels)
                val_loss_accum += loss.numpy()
                val_metric_accum += accuracy.numpy()
                val_steps += 1

        elif args.task == 'detection':
            for batch in val_ds:
                point_clouds, gt_boxes, gt_labels, num_valid = batch
                total_loss, box_loss, cls_loss = val_step(
                    point_clouds, gt_boxes, gt_labels, num_valid
                )
                val_loss_accum += total_loss.numpy()
                val_metric_accum += box_loss.numpy()
                val_steps += 1

        elif args.task == 'segmentation':
            for batch in val_ds:
                point_clouds, labels = batch
                loss, accuracy = val_step(point_clouds, labels)
                val_loss_accum += loss.numpy()
                val_metric_accum += accuracy.numpy()
                val_steps += 1

        avg_val_loss = val_loss_accum / max(val_steps, 1)
        avg_val_metric = val_metric_accum / max(val_steps, 1)

        epoch_time = time.time() - epoch_start_time

        # -------------------------------------------------------------------
        # TensorBoard logging
        # -------------------------------------------------------------------
        with train_writer.as_default():
            tf.summary.scalar('loss', avg_train_loss, step=epoch)
            if args.task in ('classification', 'segmentation'):
                tf.summary.scalar('accuracy', avg_train_metric, step=epoch)
            elif args.task == 'detection':
                tf.summary.scalar('box_loss', avg_train_metric, step=epoch)
            tf.summary.scalar(
                'learning_rate',
                lr_schedule(global_step),
                step=epoch
            )

        with val_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=epoch)
            if args.task in ('classification', 'segmentation'):
                tf.summary.scalar('accuracy', avg_val_metric, step=epoch)
            elif args.task == 'detection':
                tf.summary.scalar('box_loss', avg_val_metric, step=epoch)

        # -------------------------------------------------------------------
        # Progress output
        # -------------------------------------------------------------------
        if args.task in ('classification', 'segmentation'):
            print(
                f"Epoch {epoch + 1:4d}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Acc: {avg_train_metric:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {avg_val_metric:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
        elif args.task == 'detection':
            print(
                f"Epoch {epoch + 1:4d}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Box Loss: {avg_train_metric:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Box Loss: {avg_val_metric:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

        # -------------------------------------------------------------------
        # Save checkpoint every epoch
        # -------------------------------------------------------------------
        checkpoint_manager.save()

        # -------------------------------------------------------------------
        # Save best model
        # -------------------------------------------------------------------
        if args.task in ('classification', 'segmentation'):
            # Higher accuracy is better
            if avg_val_metric > best_val_metric:
                best_val_metric = avg_val_metric
                best_checkpoint_manager.save()
                print(f"  -> New best model saved "
                      f"(val accuracy: {best_val_metric:.4f})")
        elif args.task == 'detection':
            # Lower loss is better
            if avg_val_loss < best_val_metric:
                best_val_metric = avg_val_loss
                best_checkpoint_manager.save()
                print(f"  -> New best model saved "
                      f"(val loss: {best_val_metric:.4f})")

    # -----------------------------------------------------------------------
    # Training complete
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("[INFO] Training complete!")
    if args.task in ('classification', 'segmentation'):
        print(f"[INFO] Best validation accuracy: {best_val_metric:.4f}")
    elif args.task == 'detection':
        print(f"[INFO] Best validation loss: {best_val_metric:.4f}")
    print(f"[INFO] Checkpoints saved to: {args.checkpoint_dir}")
    print(f"[INFO] Best model saved to: {best_model_dir}")
    print(f"[INFO] TensorBoard logs at: {args.log_dir}")
    print(f"[INFO] Run: tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()
