"""Cylinder3D TensorFlow 2 training script.

Complete training pipeline for Cylinder3D on SemanticKITTI dataset with:
- YAML-based configuration
- tf.data.Dataset pipeline with augmentations
- Custom training loop with tf.GradientTape
- Mixed precision training
- Multi-GPU support via tf.distribute.MirroredStrategy
- Cosine decay learning rate schedule
- TensorBoard logging, checkpointing, early stopping
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import yaml
import tensorflow as tf

from model import (
    Cylinder3DModel,
    CombinedLoss,
    SEMANTICKITTI_NUM_CLASSES,
    DEFAULT_GRID_SIZE,
)


# SemanticKITTI label mapping (raw label -> training id)
SEMANTICKITTI_LABEL_MAP = {
    0: 0,    # unlabeled
    1: 0,    # outlier
    10: 1,   # car
    11: 2,   # bicycle
    13: 3,   # bus
    15: 4,   # motorcycle
    16: 5,   # on-rails
    18: 6,   # truck
    20: 7,   # other-vehicle
    30: 8,   # person
    31: 9,   # bicyclist
    32: 10,  # motorcyclist
    40: 11,  # road
    44: 12,  # parking
    48: 13,  # sidewalk
    49: 14,  # other-ground
    50: 15,  # building
    51: 16,  # fence
    52: 0,   # other-structure
    60: 17,  # lane-marking
    70: 18,  # vegetation
    71: 19,  # trunk
    72: 18,  # terrain -> vegetation
    80: 19,  # pole -> trunk
    81: 19,  # traffic-sign -> trunk
    99: 0,   # other-object
    252: 1,  # moving-car -> car
    253: 7,  # moving-other-vehicle
    254: 6,  # moving-truck -> truck
    255: 8,  # moving-person -> person
    256: 9,  # moving-bicyclist -> bicyclist
    257: 10, # moving-motorcyclist -> motorcyclist
}

# Default class weights (inverse frequency, SemanticKITTI)
DEFAULT_CLASS_WEIGHTS = [
    0.0,    # unlabeled (ignored)
    1.0,    # car
    5.0,    # bicycle
    4.0,    # bus
    5.0,    # motorcycle
    5.0,    # on-rails
    3.0,    # truck
    4.0,    # other-vehicle
    5.0,    # person
    5.0,    # bicyclist
    5.0,    # motorcyclist
    0.3,    # road
    1.0,    # parking
    0.5,    # sidewalk
    2.0,    # other-ground
    0.8,    # building
    2.0,    # fence
    2.0,    # lane-marking
    0.5,    # vegetation
    2.0,    # trunk
]

# Training sequences for SemanticKITTI
TRAIN_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
VAL_SEQUENCES = ["08"]


def get_default_config():
    """Return default training configuration."""
    return {
        "model": {
            "num_classes": SEMANTICKITTI_NUM_CLASSES,
            "grid_size": DEFAULT_GRID_SIZE,
            "rho_range": [0.0, 50.0],
            "theta_range": [-3.14159, 3.14159],
            "z_range": [-3.0, 1.0],
        },
        "data": {
            "dataset_path": "/data/semantickitti/dataset/sequences",
            "num_points": 100000,
            "batch_size": 2,
            "num_workers": 4,
            "augmentation": {
                "rotation": True,
                "flip": True,
                "scale": True,
                "scale_range": [0.95, 1.05],
                "rotation_range": [-0.3927, 0.3927],
            },
        },
        "training": {
            "epochs": 40,
            "initial_lr": 0.001,
            "min_lr": 1e-6,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 10.0,
            "mixed_precision": True,
            "ce_weight": 1.0,
            "lovasz_weight": 1.5,
            "class_weights": DEFAULT_CLASS_WEIGHTS,
            "warmup_epochs": 2,
        },
        "checkpoint": {
            "save_dir": "./checkpoints",
            "save_freq": 1,
            "keep_top_k": 3,
        },
        "logging": {
            "log_dir": "./logs",
            "log_freq": 50,
        },
    }


def load_config(config_path=None):
    """Load configuration from YAML file, with defaults for missing keys."""
    config = get_default_config()
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            _deep_update(config, user_config)
    return config


def _deep_update(base, update):
    """Recursively update nested dict."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def remap_labels(labels):
    """Remap raw SemanticKITTI labels to training IDs.

    Args:
        labels: numpy array of raw label IDs

    Returns:
        Remapped labels as numpy array
    """
    remapped = np.zeros_like(labels, dtype=np.int32)
    for raw_id, train_id in SEMANTICKITTI_LABEL_MAP.items():
        remapped[labels == raw_id] = train_id
    return remapped


def load_point_cloud(bin_path, label_path, num_points):
    """Load a point cloud and its labels from SemanticKITTI format.

    Args:
        bin_path: path to .bin file (float32 x,y,z,intensity)
        label_path: path to .label file (uint32, lower 16 bits = sem label)
        num_points: target number of points (pad or subsample)

    Returns:
        points: [num_points, 4] float32
        labels: [num_points] int32
    """
    bin_path = bin_path.numpy().decode("utf-8") if hasattr(bin_path, "numpy") else str(bin_path)
    label_path = label_path.numpy().decode("utf-8") if hasattr(label_path, "numpy") else str(label_path)
    num_points = int(num_points.numpy()) if hasattr(num_points, "numpy") else int(num_points)

    # Load points
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    # Load labels
    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    sem_labels = (raw_labels & 0xFFFF).astype(np.int32)
    sem_labels = remap_labels(sem_labels)

    n = points.shape[0]

    if n >= num_points:
        # Random subsample
        indices = np.random.choice(n, num_points, replace=False)
        points = points[indices]
        sem_labels = sem_labels[indices]
    else:
        # Pad with zeros
        pad_points = np.zeros((num_points - n, 4), dtype=np.float32)
        pad_labels = np.zeros(num_points - n, dtype=np.int32)
        points = np.concatenate([points, pad_points], axis=0)
        sem_labels = np.concatenate([sem_labels, pad_labels], axis=0)

    return points.astype(np.float32), sem_labels.astype(np.int32)


def augment_point_cloud(points, labels, config):
    """Apply data augmentation to point cloud.

    Args:
        points: [N, 4] point cloud
        labels: [N] labels
        config: augmentation config dict

    Returns:
        Augmented points and labels
    """
    aug_config = config["data"]["augmentation"]

    # Random rotation around z-axis
    if aug_config.get("rotation", False):
        rot_range = aug_config.get("rotation_range", [-0.3927, 0.3927])
        angle = tf.random.uniform([], rot_range[0], rot_range[1])
        cos_a = tf.cos(angle)
        sin_a = tf.sin(angle)
        rot_matrix = tf.stack(
            [
                tf.stack([cos_a, -sin_a, 0.0]),
                tf.stack([sin_a, cos_a, 0.0]),
                tf.stack([0.0, 0.0, 1.0]),
            ]
        )  # [3, 3]
        xyz = points[:, :3]
        rotated = tf.matmul(xyz, rot_matrix, transpose_b=True)
        points = tf.concat([rotated, points[:, 3:]], axis=-1)

    # Random flip along x or y axis
    if aug_config.get("flip", False):
        if tf.random.uniform([]) > 0.5:
            # Flip x
            points = tf.concat([-points[:, :1], points[:, 1:]], axis=-1)
        if tf.random.uniform([]) > 0.5:
            # Flip y
            points = tf.concat(
                [points[:, :1], -points[:, 1:2], points[:, 2:]], axis=-1
            )

    # Random scale
    if aug_config.get("scale", False):
        scale_range = aug_config.get("scale_range", [0.95, 1.05])
        scale = tf.random.uniform([], scale_range[0], scale_range[1])
        xyz_scaled = points[:, :3] * scale
        points = tf.concat([xyz_scaled, points[:, 3:]], axis=-1)

    return points, labels


def create_dataset(config, sequences, is_training=True):
    """Create a tf.data.Dataset for SemanticKITTI sequences.

    Args:
        config: configuration dict
        sequences: list of sequence strings (e.g., ["00", "01", ...])
        is_training: whether to apply augmentation and shuffling

    Returns:
        tf.data.Dataset yielding (points, labels) batches
    """
    dataset_path = config["data"]["dataset_path"]
    num_points = config["data"]["num_points"]
    batch_size = config["data"]["batch_size"]

    # Collect all bin/label file pairs
    bin_files = []
    label_files = []

    for seq in sequences:
        bin_dir = os.path.join(dataset_path, seq, "velodyne")
        label_dir = os.path.join(dataset_path, seq, "labels")

        if not os.path.isdir(bin_dir):
            print(f"Warning: {bin_dir} not found, skipping sequence {seq}")
            continue

        frames = sorted(os.listdir(bin_dir))
        for frame in frames:
            if frame.endswith(".bin"):
                bin_files.append(os.path.join(bin_dir, frame))
                label_file = frame.replace(".bin", ".label")
                label_files.append(os.path.join(label_dir, label_file))

    print(f"Found {len(bin_files)} frames in sequences {sequences}")

    if len(bin_files) == 0:
        raise ValueError(f"No data found in {dataset_path} for sequences {sequences}")

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((bin_files, label_files))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(bin_files), reshuffle_each_iteration=True)

    # Load point clouds using py_function
    def _load_fn(bin_path, label_path):
        points, labels = tf.py_function(
            load_point_cloud,
            [bin_path, label_path, num_points],
            [tf.float32, tf.int32],
        )
        points.set_shape([num_points, 4])
        labels.set_shape([num_points])
        return points, labels

    dataset = dataset.map(_load_fn, num_parallel_calls=config["data"]["num_workers"])

    # Apply augmentation
    if is_training:
        def _augment_fn(points, labels):
            points, labels = augment_point_cloud(points, labels, config)
            return points, labels

        dataset = dataset.map(_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_lr_schedule(config, steps_per_epoch):
    """Create cosine decay learning rate schedule with warmup.

    Args:
        config: training config
        steps_per_epoch: number of training steps per epoch

    Returns:
        tf.keras.optimizers.schedules.LearningRateSchedule
    """
    initial_lr = config["training"]["initial_lr"]
    min_lr = config["training"]["min_lr"]
    epochs = config["training"]["epochs"]
    warmup_epochs = config["training"]["warmup_epochs"]

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self):
            super().__init__()
            self.initial_lr = initial_lr
            self.min_lr = min_lr
            self.warmup_steps = warmup_steps
            self.total_steps = total_steps

        def __call__(self, step):
            step = tf.cast(step, tf.float32)

            # Linear warmup
            warmup_lr = self.initial_lr * (step / tf.maximum(tf.cast(self.warmup_steps, tf.float32), 1.0))

            # Cosine decay after warmup
            decay_steps = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
            progress = (step - tf.cast(self.warmup_steps, tf.float32)) / tf.maximum(decay_steps, 1.0)
            progress = tf.clip_by_value(progress, 0.0, 1.0)
            cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1.0 + tf.cos(np.pi * progress)
            )

            return tf.where(step < tf.cast(self.warmup_steps, tf.float32), warmup_lr, cosine_lr)

        def get_config(self):
            return {
                "initial_lr": self.initial_lr,
                "min_lr": self.min_lr,
                "warmup_steps": int(self.warmup_steps),
                "total_steps": int(self.total_steps),
            }

    return WarmupCosineDecay()


def compute_iou(confusion_matrix):
    """Compute per-class IoU from confusion matrix.

    Args:
        confusion_matrix: [num_classes, num_classes] numpy array

    Returns:
        per_class_iou: [num_classes] array
        mean_iou: scalar (excluding class 0 / unlabeled)
    """
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp

    denominator = tp + fp + fn
    iou = np.where(denominator > 0, tp / denominator, 0.0)

    # Mean IoU excluding unlabeled (class 0)
    valid = denominator[1:] > 0
    mean_iou = np.mean(iou[1:][valid]) if valid.any() else 0.0

    return iou, mean_iou


@tf.function
def train_step(model, optimizer, loss_fn, points, labels, gradient_clip_norm):
    """Single training step with gradient tape.

    Args:
        model: Cylinder3DModel instance
        optimizer: tf.keras.optimizers.Optimizer
        loss_fn: CombinedLoss instance
        points: [B, N, 4] input points
        labels: [B, N] ground truth labels
        gradient_clip_norm: float, max gradient norm

    Returns:
        loss: scalar loss value
        point_logits: [B, N, C] predictions
    """
    with tf.GradientTape() as tape:
        point_logits, voxel_logits = model(points, training=True)
        loss = loss_fn(labels, point_logits)

        # Scale loss for mixed precision
        scaled_loss = optimizer.get_scaled_loss(loss) if hasattr(optimizer, "get_scaled_loss") else loss

    # Compute and apply gradients
    if hasattr(optimizer, "get_scaled_loss"):
        scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads = optimizer.get_unscaled_gradients(scaled_grads)
    else:
        grads = tape.gradient(loss, model.trainable_variables)

    # Clip gradients
    grads, grad_norm = tf.clip_by_global_norm(grads, gradient_clip_norm)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, point_logits


@tf.function
def val_step(model, loss_fn, points, labels):
    """Single validation step.

    Args:
        model: Cylinder3DModel instance
        loss_fn: CombinedLoss instance
        points: [B, N, 4] input points
        labels: [B, N] ground truth labels

    Returns:
        loss: scalar loss value
        predictions: [B, N] predicted class indices
    """
    point_logits, _ = model(points, training=False)
    loss = loss_fn(labels, point_logits)
    predictions = tf.argmax(point_logits, axis=-1, output_type=tf.int32)
    return loss, predictions


def train(config, args):
    """Main training function.

    Args:
        config: configuration dict
        args: argparse namespace with GPU settings
    """
    # Setup GPU
    gpus = tf.config.list_physical_devices("GPU")
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        visible_gpus = [gpus[i] for i in gpu_ids if i < len(gpus)]
        tf.config.set_visible_devices(visible_gpus, "GPU")
        gpus = visible_gpus

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"Using {len(gpus)} GPU(s): {[g.name for g in gpus]}")

    # Mixed precision
    if config["training"]["mixed_precision"] and len(gpus) > 0:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled (float16)")

    # Multi-GPU strategy
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Multi-GPU training with MirroredStrategy ({strategy.num_replicas_in_sync} replicas)")
    else:
        strategy = tf.distribute.get_strategy()  # Default (single device)

    # Create datasets
    print("Loading training data...")
    train_dataset = create_dataset(config, TRAIN_SEQUENCES, is_training=True)
    print("Loading validation data...")
    val_dataset = create_dataset(config, VAL_SEQUENCES, is_training=False)

    # Estimate steps per epoch
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch else 19130 // config["data"]["batch_size"]
    val_steps = args.val_steps if args.val_steps else 4071 // config["data"]["batch_size"]

    print(f"Steps per epoch: {steps_per_epoch}, Val steps: {val_steps}")

    # Create model and optimizer within strategy scope
    with strategy.scope():
        model = Cylinder3DModel(
            num_classes=config["model"]["num_classes"],
            grid_size=config["model"]["grid_size"],
            rho_range=config["model"].get("rho_range"),
            theta_range=config["model"].get("theta_range"),
            z_range=config["model"].get("z_range"),
        )

        # Learning rate schedule
        lr_schedule = create_lr_schedule(config, steps_per_epoch)

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        )

        # Wrap optimizer for mixed precision
        if config["training"]["mixed_precision"] and len(gpus) > 0:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        # Loss function
        loss_fn = CombinedLoss(
            num_classes=config["model"]["num_classes"],
            ignore_index=0,
            ce_weight=config["training"]["ce_weight"],
            lovasz_weight=config["training"]["lovasz_weight"],
            class_weights=config["training"].get("class_weights"),
        )

    # Directories
    checkpoint_dir = config["checkpoint"]["save_dir"]
    log_dir = config["logging"]["log_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard writer
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Checkpoint manager
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint,
        checkpoint_dir,
        max_to_keep=config["checkpoint"]["keep_top_k"],
    )

    # Restore if exists
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print(f"Restored from {ckpt_manager.latest_checkpoint}")

    # Training loop
    gradient_clip_norm = config["training"]["gradient_clip_norm"]
    log_freq = config["logging"]["log_freq"]
    best_miou = 0.0
    patience = 10
    patience_counter = 0
    num_classes = config["model"]["num_classes"]
    epochs = config["training"]["epochs"]

    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 80)

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        # Training
        for step, (points, labels) in enumerate(train_dataset):
            if step >= steps_per_epoch:
                break

            loss, point_logits = train_step(
                model, optimizer, loss_fn, points, labels, gradient_clip_norm
            )

            epoch_loss += loss.numpy()
            epoch_steps += 1

            global_step = epoch * steps_per_epoch + step

            # Log training metrics
            if step % log_freq == 0:
                current_lr = optimizer.learning_rate
                if callable(current_lr):
                    current_lr = current_lr(global_step)
                elif hasattr(current_lr, "numpy"):
                    current_lr = current_lr.numpy()

                with summary_writer.as_default():
                    tf.summary.scalar("train/loss", loss, step=global_step)
                    tf.summary.scalar("train/learning_rate", current_lr, step=global_step)

                print(
                    f"  Epoch {epoch+1}/{epochs} | Step {step}/{steps_per_epoch} | "
                    f"Loss: {loss.numpy():.4f} | LR: {current_lr:.6f}"
                )

        avg_train_loss = epoch_loss / max(epoch_steps, 1)

        # Validation
        val_loss_total = 0.0
        val_steps_done = 0
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        print(f"\n  Validating...")
        for step, (points, labels) in enumerate(val_dataset):
            if step >= val_steps:
                break

            loss, predictions = val_step(model, loss_fn, points, labels)
            val_loss_total += loss.numpy()
            val_steps_done += 1

            # Update confusion matrix
            preds_np = predictions.numpy().flatten()
            labels_np = labels.numpy().flatten()

            # Only count valid labels
            valid = labels_np != 0
            preds_valid = preds_np[valid]
            labels_valid = labels_np[valid]

            for pred, gt in zip(preds_valid, labels_valid):
                if 0 <= pred < num_classes and 0 <= gt < num_classes:
                    confusion_matrix[gt, pred] += 1

        avg_val_loss = val_loss_total / max(val_steps_done, 1)
        per_class_iou, mean_iou = compute_iou(confusion_matrix)

        epoch_time = time.time() - epoch_start

        # Log validation metrics
        with summary_writer.as_default():
            tf.summary.scalar("val/loss", avg_val_loss, step=(epoch + 1) * steps_per_epoch)
            tf.summary.scalar("val/mean_iou", mean_iou, step=(epoch + 1) * steps_per_epoch)
            for c in range(1, num_classes):
                tf.summary.scalar(f"val/iou_class_{c}", per_class_iou[c], step=(epoch + 1) * steps_per_epoch)

        print(f"\n  Epoch {epoch+1}/{epochs} Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss: {avg_val_loss:.4f}")
        print(f"    Mean IoU: {mean_iou:.4f}")
        print(f"    Time: {epoch_time:.1f}s")
        print("-" * 80)

        # Checkpointing
        if mean_iou > best_miou:
            best_miou = mean_iou
            patience_counter = 0
            ckpt_path = ckpt_manager.save()
            print(f"    New best mIoU! Saved checkpoint: {ckpt_path}")

            # Save best model weights separately
            best_weights_path = os.path.join(checkpoint_dir, "best_model")
            model.save_weights(best_weights_path)
        else:
            patience_counter += 1
            if epoch % config["checkpoint"]["save_freq"] == 0:
                ckpt_path = ckpt_manager.save()
                print(f"    Saved periodic checkpoint: {ckpt_path}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping after {patience} epochs without improvement.")
            break

    print("\n" + "=" * 80)
    print(f"Training complete! Best mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"TensorBoard logs: {log_dir}")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model")
    model.save_weights(final_path)
    print(f"Final model saved to: {final_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Cylinder3D on SemanticKITTI (TensorFlow 2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1')",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Override steps per epoch (for debugging)",
    )
    parser.add_argument(
        "--val_steps",
        type=int,
        default=None,
        help="Override validation steps (for debugging)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    """Entry point for training."""
    args = parse_args()

    # Set random seeds
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Override checkpoint dir with resume path if provided
    if args.resume:
        config["checkpoint"]["save_dir"] = os.path.dirname(args.resume)

    print("Cylinder3D TensorFlow 2 Training")
    print("=" * 80)
    print(f"Config: {args.config or 'default'}")
    print(f"GPUs: {args.gpus or 'all available'}")
    print(f"Grid size: {config['model']['grid_size']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Initial LR: {config['training']['initial_lr']}")
    print(f"Mixed precision: {config['training']['mixed_precision']}")
    print("=" * 80)

    train(config, args)


if __name__ == "__main__":
    main()
