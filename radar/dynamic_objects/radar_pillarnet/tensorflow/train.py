"""
RadarPillarNet training script.

Uses TF2 custom training loop with:
- tf.data.Dataset pipeline for nuScenes radar data
- Custom training loop with tf.GradientTape
- Focal loss + Smooth L1 + velocity loss + direction loss
- Adam optimizer with one-cycle learning rate schedule
- Mixed precision (tf.keras.mixed_precision)
- Multi-GPU with tf.distribute.MirroredStrategy
- TensorBoard logging
- Checkpoint management with best-model tracking
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from model import RadarPillarNet, DEFAULT_CONFIG, build_radar_pillarnet


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

TRAIN_CONFIG: Dict[str, Any] = {
    # Data
    "data_root": "/data/nuscenes",
    "train_split": "train",
    "val_split": "val",
    "batch_size": 8,
    "num_workers": 8,
    "prefetch_buffer": 4,
    # Optimization
    "epochs": 20,
    "base_lr": 1e-3,
    "max_lr": 3e-3,
    "min_lr": 1e-6,
    "weight_decay": 0.01,
    "warmup_fraction": 0.3,
    "grad_clip_norm": 35.0,
    # Loss weights
    "cls_loss_weight": 1.0,
    "box_loss_weight": 2.0,
    "vel_loss_weight": 0.2,
    "dir_loss_weight": 0.2,
    # Focal loss params
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    # Smooth L1
    "smooth_l1_beta": 1.0 / 9.0,
    # Logging
    "log_dir": "./logs/radar_pillarnet",
    "checkpoint_dir": "./checkpoints/radar_pillarnet",
    "log_interval": 50,
    "val_interval": 1,
    "save_interval": 1,
    # Mixed precision
    "mixed_precision": True,
    # Multi-GPU
    "multi_gpu": True,
}


# ---------------------------------------------------------------------------
# nuScenes Radar Dataset Pipeline
# ---------------------------------------------------------------------------


class NuScenesRadarDataLoader:
    """tf.data.Dataset pipeline for loading nuScenes radar pillar data."""

    def __init__(
        self, config: Dict[str, Any], model_config: Dict[str, Any], split: str = "train"
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.split = split
        self.data_root = Path(config["data_root"])
        self.samples = self._load_sample_list()

    def _load_sample_list(self) -> List[Dict[str, Any]]:
        """Load sample metadata from nuScenes info files."""
        info_path = self.data_root / f"nuscenes_infos_{self.split}.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                infos = json.load(f)
            return infos.get("infos", [])
        # Fallback: generate dummy sample list for testing
        return [{"token": f"sample_{i}"} for i in range(1000)]

    def _parse_sample(self, sample_idx: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Parse a single sample into model inputs and targets."""
        idx = sample_idx.numpy()
        sample_info = self.samples[idx % len(self.samples)]

        max_pillars = self.model_config["max_pillars"]
        max_pts = self.model_config["max_points_per_pillar"]
        grid_x = self.model_config["grid_x"]
        grid_y = self.model_config["grid_y"]
        num_classes = self.model_config["num_classes"]
        num_anchors = self.model_config["num_anchors_per_location"]

        token = sample_info.get("token", "dummy")

        # Load radar pillar data
        radar_path = self.data_root / "radar_pillars" / f"{token}.npz"
        if radar_path.exists():
            radar_data = np.load(str(radar_path))
            pillar_features = np.zeros((max_pillars, max_pts, 9), dtype=np.float32)
            pillar_mask = np.zeros((max_pillars, max_pts), dtype=np.float32)
            pillar_coords = np.zeros((max_pillars, 2), dtype=np.int32)
            n = min(radar_data["features"].shape[0], max_pillars)
            pillar_features[:n] = radar_data["features"][:n]
            pillar_mask[:n] = radar_data["mask"][:n].astype(np.float32)
            pillar_coords[:n] = radar_data["coords"][:n]
        else:
            # Synthetic radar data for pipeline testing
            n_active = np.random.randint(200, max_pillars)
            pillar_features = np.zeros((max_pillars, max_pts, 9), dtype=np.float32)
            pillar_features[:n_active] = np.random.randn(n_active, max_pts, 9).astype(np.float32) * 0.1
            pillar_mask = np.zeros((max_pillars, max_pts), dtype=np.float32)
            n_points = np.random.randint(1, max_pts, size=n_active)
            for i in range(n_active):
                pillar_mask[i, :n_points[i]] = 1.0
            pillar_coords = np.zeros((max_pillars, 2), dtype=np.int32)
            pillar_coords[:n_active, 0] = np.random.randint(0, grid_x, n_active)
            pillar_coords[:n_active, 1] = np.random.randint(0, grid_y, n_active)

        # Compute feature map spatial size (after backbone downsampling)
        # First stride is 2, so feature map is grid_size / 2
        feat_h = grid_y // 2
        feat_w = grid_x // 2

        # Load ground truth targets
        # Target shapes: (feat_h, feat_w, num_anchors, ...)
        gt_path = self.data_root / "radar_targets" / f"{token}.npz"
        if gt_path.exists():
            gt_data = np.load(str(gt_path))
            cls_targets = gt_data["cls_targets"]       # (H, W, A)
            box_targets = gt_data["box_targets"]       # (H, W, A, 7)
            vel_targets = gt_data["vel_targets"]       # (H, W, A, 2)
            dir_targets = gt_data["dir_targets"]       # (H, W, A)
            reg_mask = gt_data["reg_mask"]             # (H, W, A) - 1 for positive anchors
        else:
            # Synthetic targets
            cls_targets = np.zeros((feat_h, feat_w, num_anchors), dtype=np.int32)
            box_targets = np.zeros((feat_h, feat_w, num_anchors, 7), dtype=np.float32)
            vel_targets = np.zeros((feat_h, feat_w, num_anchors, 2), dtype=np.float32)
            dir_targets = np.zeros((feat_h, feat_w, num_anchors), dtype=np.int32)
            reg_mask = np.zeros((feat_h, feat_w, num_anchors), dtype=np.float32)

            # Add a few synthetic positive anchors
            n_pos = np.random.randint(5, 30)
            for _ in range(n_pos):
                yi = np.random.randint(0, feat_h)
                xi = np.random.randint(0, feat_w)
                ai = np.random.randint(0, num_anchors)
                cls_targets[yi, xi, ai] = np.random.randint(1, num_classes + 1)
                box_targets[yi, xi, ai] = np.random.randn(7).astype(np.float32) * 0.1
                vel_targets[yi, xi, ai] = np.random.randn(2).astype(np.float32) * 2.0
                dir_targets[yi, xi, ai] = np.random.randint(0, 2)
                reg_mask[yi, xi, ai] = 1.0

        return {
            "pillar_features": pillar_features.astype(np.float32),
            "pillar_mask": pillar_mask.astype(np.float32),
            "pillar_coords": pillar_coords.astype(np.int32),
            "cls_targets": cls_targets.astype(np.int32),
            "box_targets": box_targets.astype(np.float32),
            "vel_targets": vel_targets.astype(np.float32),
            "dir_targets": dir_targets.astype(np.int32),
            "reg_mask": reg_mask.astype(np.float32),
        }

    def build_dataset(self) -> tf.data.Dataset:
        """Build tf.data.Dataset with batching and prefetching."""
        num_samples = len(self.samples)
        indices = tf.data.Dataset.range(num_samples)

        if self.split == "train":
            indices = indices.shuffle(buffer_size=min(num_samples, 10000), reshuffle_each_iteration=True)

        m_cfg = self.model_config
        max_pillars = m_cfg["max_pillars"]
        max_pts = m_cfg["max_points_per_pillar"]
        grid_x = m_cfg["grid_x"]
        grid_y = m_cfg["grid_y"]
        feat_h = grid_y // 2
        feat_w = grid_x // 2
        num_anchors = m_cfg["num_anchors_per_location"]

        output_signature = {
            "pillar_features": tf.TensorSpec(shape=(max_pillars, max_pts, 9), dtype=tf.float32),
            "pillar_mask": tf.TensorSpec(shape=(max_pillars, max_pts), dtype=tf.float32),
            "pillar_coords": tf.TensorSpec(shape=(max_pillars, 2), dtype=tf.int32),
            "cls_targets": tf.TensorSpec(shape=(feat_h, feat_w, num_anchors), dtype=tf.int32),
            "box_targets": tf.TensorSpec(shape=(feat_h, feat_w, num_anchors, 7), dtype=tf.float32),
            "vel_targets": tf.TensorSpec(shape=(feat_h, feat_w, num_anchors, 2), dtype=tf.float32),
            "dir_targets": tf.TensorSpec(shape=(feat_h, feat_w, num_anchors), dtype=tf.int32),
            "reg_mask": tf.TensorSpec(shape=(feat_h, feat_w, num_anchors), dtype=tf.float32),
        }

        dataset = indices.map(
            lambda idx: tf.py_function(
                func=self._parse_sample,
                inp=[idx],
                Tout={k: v.dtype for k, v in output_signature.items()},
            ),
            num_parallel_calls=self.config.get("num_workers", 8),
        )

        def set_shapes(sample: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            sample["pillar_features"].set_shape((max_pillars, max_pts, 9))
            sample["pillar_mask"].set_shape((max_pillars, max_pts))
            sample["pillar_coords"].set_shape((max_pillars, 2))
            sample["cls_targets"].set_shape((feat_h, feat_w, num_anchors))
            sample["box_targets"].set_shape((feat_h, feat_w, num_anchors, 7))
            sample["vel_targets"].set_shape((feat_h, feat_w, num_anchors, 2))
            sample["dir_targets"].set_shape((feat_h, feat_w, num_anchors))
            sample["reg_mask"].set_shape((feat_h, feat_w, num_anchors))
            return sample

        dataset = dataset.map(set_shapes)
        dataset = dataset.batch(self.config["batch_size"], drop_remainder=True)
        dataset = dataset.prefetch(self.config.get("prefetch_buffer", 4))
        return dataset


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------


def focal_loss(
    cls_preds: tf.Tensor,
    cls_targets: tf.Tensor,
    num_classes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> tf.Tensor:
    """
    Focal loss for classification.

    Args:
        cls_preds: (B, H, W, A, num_classes) raw logits
        cls_targets: (B, H, W, A) integer class labels (0 = background)
        num_classes: number of foreground classes
        alpha: weighting factor
        gamma: focusing parameter
    Returns:
        Scalar focal loss
    """
    # Create one-hot targets (background class 0 -> all zeros)
    # cls_targets has values 0..num_classes where 0 = background
    # For focal loss, we treat it as multi-class with background explicit
    one_hot = tf.one_hot(cls_targets, depth=num_classes + 1)  # (B, H, W, A, C+1)
    # Remove background class (index 0) since we only predict foreground
    one_hot = one_hot[:, :, :, :, 1:]  # (B, H, W, A, C)

    # Sigmoid focal loss
    pred_sigmoid = tf.sigmoid(cls_preds)
    pred_sigmoid = tf.clip_by_value(pred_sigmoid, 1e-6, 1.0 - 1e-6)

    # Focal weight
    pt = one_hot * pred_sigmoid + (1.0 - one_hot) * (1.0 - pred_sigmoid)
    focal_weight = tf.pow(1.0 - pt, gamma)

    # Binary cross entropy
    bce = -(
        one_hot * tf.math.log(pred_sigmoid)
        + (1.0 - one_hot) * tf.math.log(1.0 - pred_sigmoid)
    )

    # Alpha weighting
    alpha_weight = one_hot * alpha + (1.0 - one_hot) * (1.0 - alpha)

    loss = focal_weight * alpha_weight * bce

    # Normalize by number of positive anchors
    num_pos = tf.maximum(tf.reduce_sum(tf.cast(cls_targets > 0, tf.float32)), 1.0)
    loss = tf.reduce_sum(loss) / num_pos
    return loss


def smooth_l1_loss(
    pred: tf.Tensor,
    target: tf.Tensor,
    mask: tf.Tensor,
    beta: float = 1.0 / 9.0,
) -> tf.Tensor:
    """
    Smooth L1 (Huber) loss for box regression, masked to positive anchors.

    Args:
        pred: (B, H, W, A, D) predictions
        target: (B, H, W, A, D) targets
        mask: (B, H, W, A) positive anchor mask
        beta: transition point between L1 and L2
    Returns:
        Scalar loss
    """
    mask_expanded = tf.expand_dims(mask, axis=-1)  # (B, H, W, A, 1)
    diff = tf.abs(pred - target) * mask_expanded

    # Smooth L1: |x| < beta -> 0.5*x^2/beta, else |x| - 0.5*beta
    smooth_l1 = tf.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta,
    )

    num_pos = tf.maximum(tf.reduce_sum(mask), 1.0)
    loss = tf.reduce_sum(smooth_l1) / num_pos
    return loss


def velocity_loss(
    vel_preds: tf.Tensor,
    vel_targets: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """
    L1 loss for velocity regression on positive anchors.

    Args:
        vel_preds: (B, H, W, A, 2)
        vel_targets: (B, H, W, A, 2)
        mask: (B, H, W, A) positive anchor mask
    Returns:
        Scalar loss
    """
    mask_expanded = tf.expand_dims(mask, axis=-1)
    diff = tf.abs(vel_preds - vel_targets) * mask_expanded
    num_pos = tf.maximum(tf.reduce_sum(mask), 1.0)
    loss = tf.reduce_sum(diff) / num_pos
    return loss


def direction_loss(
    dir_preds: tf.Tensor,
    dir_targets: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """
    Cross-entropy loss for direction classification on positive anchors.

    Args:
        dir_preds: (B, H, W, A, 2) logits for direction bins
        dir_targets: (B, H, W, A) integer direction labels (0 or 1)
        mask: (B, H, W, A) positive anchor mask
    Returns:
        Scalar loss
    """
    # Flatten
    dir_preds_flat = tf.reshape(dir_preds, [-1, 2])
    dir_targets_flat = tf.reshape(dir_targets, [-1])
    mask_flat = tf.reshape(mask, [-1])

    # Cross-entropy
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=dir_targets_flat,
        logits=dir_preds_flat,
    )

    # Mask
    ce_masked = ce * mask_flat
    num_pos = tf.maximum(tf.reduce_sum(mask_flat), 1.0)
    loss = tf.reduce_sum(ce_masked) / num_pos
    return loss


def compute_total_loss(
    predictions: Dict[str, tf.Tensor],
    targets: Dict[str, tf.Tensor],
    config: Dict[str, Any],
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """
    Compute combined multi-task training loss.

    Args:
        predictions: model output dict
        targets: ground truth dict
        config: training config with loss weights
    Returns:
        (total_loss, loss_dict) where loss_dict has individual components
    """
    cls_preds = predictions["cls_preds"]   # (B, H, W, A, C)
    box_preds = predictions["box_preds"]   # (B, H, W, A, 7)
    vel_preds = predictions["vel_preds"]   # (B, H, W, A, 2)
    dir_preds = predictions["dir_preds"]   # (B, H, W, A, 2)

    cls_targets = targets["cls_targets"]   # (B, H, W, A)
    box_targets = targets["box_targets"]   # (B, H, W, A, 7)
    vel_targets = targets["vel_targets"]   # (B, H, W, A, 2)
    dir_targets = targets["dir_targets"]   # (B, H, W, A)
    reg_mask = targets["reg_mask"]         # (B, H, W, A)

    num_classes = cls_preds.shape[-1] if cls_preds.shape[-1] is not None else config.get("num_classes", 10)

    # Classification: focal loss
    cls_loss = focal_loss(
        cls_preds, cls_targets, num_classes,
        alpha=config.get("focal_alpha", 0.25),
        gamma=config.get("focal_gamma", 2.0),
    )

    # Box regression: smooth L1
    box_loss = smooth_l1_loss(
        box_preds, box_targets, reg_mask,
        beta=config.get("smooth_l1_beta", 1.0 / 9.0),
    )

    # Velocity: L1
    vel_loss = velocity_loss(vel_preds, vel_targets, reg_mask)

    # Direction classification
    dir_loss = direction_loss(dir_preds, dir_targets, reg_mask)

    # Weighted sum
    total_loss = (
        config.get("cls_loss_weight", 1.0) * cls_loss
        + config.get("box_loss_weight", 2.0) * box_loss
        + config.get("vel_loss_weight", 0.2) * vel_loss
        + config.get("dir_loss_weight", 0.2) * dir_loss
    )

    loss_dict = {
        "total_loss": total_loss,
        "cls_loss": cls_loss,
        "box_loss": box_loss,
        "vel_loss": vel_loss,
        "dir_loss": dir_loss,
    }
    return total_loss, loss_dict


# ---------------------------------------------------------------------------
# One-Cycle Learning Rate Schedule
# ---------------------------------------------------------------------------


class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    One-cycle learning rate policy (Smith, 2018).

    Ramps LR from base_lr to max_lr during warmup phase, then cosine
    anneals down to min_lr for the remainder of training.
    """

    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        min_lr: float,
        total_steps: int,
        warmup_fraction: float = 0.3,
    ) -> None:
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.decay_steps = total_steps - self.warmup_steps

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Phase 1: linear warmup from base_lr to max_lr
        warmup_progress = step / tf.maximum(warmup_steps, 1.0)
        warmup_lr = self.base_lr + (self.max_lr - self.base_lr) * warmup_progress

        # Phase 2: cosine anneal from max_lr to min_lr
        decay_progress = (step - warmup_steps) / tf.maximum(decay_steps, 1.0)
        decay_progress = tf.clip_by_value(decay_progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * decay_progress)
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self) -> Dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "total_steps": self.total_steps,
            "warmup_fraction": self.warmup_steps / max(self.total_steps, 1),
        }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train(
    train_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Main training function with custom training loop.

    Args:
        train_config: training hyperparameters (overrides TRAIN_CONFIG)
        model_config: model architecture config (overrides DEFAULT_CONFIG)
    """
    cfg = {**TRAIN_CONFIG, **(train_config or {})}
    m_cfg = {**DEFAULT_CONFIG, **(model_config or {})}

    # Setup mixed precision
    if cfg["mixed_precision"]:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("[INFO] Mixed precision enabled: mixed_float16")

    # Setup distribution strategy
    if cfg["multi_gpu"] and len(tf.config.list_physical_devices("GPU")) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"[INFO] MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
    else:
        strategy = tf.distribute.get_strategy()
        print("[INFO] Single device training")

    # Build datasets
    print("[INFO] Building datasets...")
    train_loader = NuScenesRadarDataLoader(cfg, m_cfg, split=cfg["train_split"])
    val_loader = NuScenesRadarDataLoader(cfg, m_cfg, split=cfg["val_split"])
    train_dataset = train_loader.build_dataset()
    val_dataset = val_loader.build_dataset()

    # Distribute datasets
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # Build model within strategy scope
    with strategy.scope():
        model = build_radar_pillarnet(config=m_cfg)

        # Warm up model by running a dummy forward pass
        dummy_inputs = {
            "pillar_features": tf.zeros([1, m_cfg["max_pillars"], m_cfg["max_points_per_pillar"], 9]),
            "pillar_mask": tf.zeros([1, m_cfg["max_pillars"], m_cfg["max_points_per_pillar"]]),
            "pillar_coords": tf.zeros([1, m_cfg["max_pillars"], 2], dtype=tf.int32),
        }
        _ = model(dummy_inputs, training=False)
        print(f"[INFO] Model built: {model.count_params():,} parameters")

        # LR schedule (one-cycle)
        steps_per_epoch = max(len(train_loader.samples) // cfg["batch_size"], 1)
        total_steps = steps_per_epoch * cfg["epochs"]

        lr_schedule = OneCycleLR(
            base_lr=cfg["base_lr"],
            max_lr=cfg["max_lr"],
            min_lr=cfg["min_lr"],
            total_steps=total_steps,
            warmup_fraction=cfg["warmup_fraction"],
        )

        # Adam optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=cfg["grad_clip_norm"],
        )

    # TensorBoard
    log_dir = os.path.join(cfg["log_dir"], time.strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.create_file_writer(log_dir)
    print(f"[INFO] TensorBoard logs: {log_dir}")

    # Checkpoint manager
    checkpoint_dir = cfg["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=5
    )

    # Restore from checkpoint if exists
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print(f"[INFO] Restored from checkpoint: {ckpt_manager.latest_checkpoint}")

    # Training step function
    @tf.function
    def train_step(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Single training step."""
        model_inputs = {
            "pillar_features": inputs["pillar_features"],
            "pillar_mask": inputs["pillar_mask"],
            "pillar_coords": inputs["pillar_coords"],
        }
        targets = {
            "cls_targets": inputs["cls_targets"],
            "box_targets": inputs["box_targets"],
            "vel_targets": inputs["vel_targets"],
            "dir_targets": inputs["dir_targets"],
            "reg_mask": inputs["reg_mask"],
        }

        with tf.GradientTape() as tape:
            predictions = model(model_inputs, training=True)
            total_loss, loss_dict = compute_total_loss(predictions, targets, cfg)

            # Scale for mixed precision
            if cfg["mixed_precision"]:
                scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, "get_scaled_loss") else total_loss
            else:
                scaled_loss = total_loss

        # Compute and apply gradients
        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        if cfg["mixed_precision"] and hasattr(optimizer, "get_unscaled_gradients"):
            gradients = optimizer.get_unscaled_gradients(gradients)

        # Filter out None gradients (can happen with unused variables)
        grads_and_vars = [
            (g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None
        ]
        optimizer.apply_gradients(grads_and_vars)

        return loss_dict

    @tf.function
    def distributed_train_step(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Distributed training step."""
        per_replica_losses = strategy.run(train_step, args=(inputs,))
        reduced = {}
        for key, val in per_replica_losses.items():
            reduced[key] = strategy.reduce(tf.distribute.ReduceOp.MEAN, val, axis=None)
        return reduced

    # Validation step
    @tf.function
    def val_step(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Single validation step (no gradients)."""
        model_inputs = {
            "pillar_features": inputs["pillar_features"],
            "pillar_mask": inputs["pillar_mask"],
            "pillar_coords": inputs["pillar_coords"],
        }
        targets = {
            "cls_targets": inputs["cls_targets"],
            "box_targets": inputs["box_targets"],
            "vel_targets": inputs["vel_targets"],
            "dir_targets": inputs["dir_targets"],
            "reg_mask": inputs["reg_mask"],
        }
        predictions = model(model_inputs, training=False)
        _, loss_dict = compute_total_loss(predictions, targets, cfg)
        return loss_dict

    @tf.function
    def distributed_val_step(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        per_replica_losses = strategy.run(val_step, args=(inputs,))
        reduced = {}
        for key, val in per_replica_losses.items():
            reduced[key] = strategy.reduce(tf.distribute.ReduceOp.MEAN, val, axis=None)
        return reduced

    # ---------------------------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------------------------
    global_step = 0
    best_val_loss = float("inf")

    print(f"\n[INFO] Starting training for {cfg['epochs']} epochs")
    print(f"[INFO] Steps per epoch: {steps_per_epoch}")
    print(f"[INFO] Total steps: {total_steps}")
    print(f"[INFO] Batch size: {cfg['batch_size']}")

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        epoch_losses: Dict[str, List[float]] = {
            "total_loss": [], "cls_loss": [], "box_loss": [],
            "vel_loss": [], "dir_loss": [],
        }

        # Training
        for step, batch in enumerate(train_dist_dataset):
            step_start = time.time()
            loss_dict = distributed_train_step(batch)
            global_step += 1

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key].append(float(loss_dict[key]))

            # Logging
            if global_step % cfg["log_interval"] == 0:
                step_time = time.time() - step_start
                current_lr = float(lr_schedule(global_step))

                print(
                    f"  Epoch {epoch+1}/{cfg['epochs']} | "
                    f"Step {step+1} | "
                    f"Loss: {float(loss_dict['total_loss']):.4f} | "
                    f"Cls: {float(loss_dict['cls_loss']):.4f} | "
                    f"Box: {float(loss_dict['box_loss']):.4f} | "
                    f"Vel: {float(loss_dict['vel_loss']):.4f} | "
                    f"Dir: {float(loss_dict['dir_loss']):.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {step_time:.2f}s"
                )

                with summary_writer.as_default():
                    tf.summary.scalar("train/total_loss", loss_dict["total_loss"], step=global_step)
                    tf.summary.scalar("train/cls_loss", loss_dict["cls_loss"], step=global_step)
                    tf.summary.scalar("train/box_loss", loss_dict["box_loss"], step=global_step)
                    tf.summary.scalar("train/vel_loss", loss_dict["vel_loss"], step=global_step)
                    tf.summary.scalar("train/dir_loss", loss_dict["dir_loss"], step=global_step)
                    tf.summary.scalar("train/learning_rate", current_lr, step=global_step)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items() if v}
        print(
            f"\n[Epoch {epoch+1}/{cfg['epochs']}] "
            f"Avg Loss: {avg_losses.get('total_loss', 0):.4f} | "
            f"Cls: {avg_losses.get('cls_loss', 0):.4f} | "
            f"Box: {avg_losses.get('box_loss', 0):.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Validation
        if (epoch + 1) % cfg["val_interval"] == 0:
            print("[INFO] Running validation...")
            val_losses: Dict[str, List[float]] = {
                "total_loss": [], "cls_loss": [], "box_loss": [],
                "vel_loss": [], "dir_loss": [],
            }

            for val_batch in val_dist_dataset:
                val_loss_dict = distributed_val_step(val_batch)
                for key in val_losses:
                    val_losses[key].append(float(val_loss_dict[key]))

            avg_val_losses = {k: np.mean(v) for k, v in val_losses.items() if v}
            val_total = avg_val_losses.get("total_loss", float("inf"))

            print(
                f"  Val Loss: {val_total:.4f} | "
                f"Val Cls: {avg_val_losses.get('cls_loss', 0):.4f} | "
                f"Val Box: {avg_val_losses.get('box_loss', 0):.4f} | "
                f"Val Vel: {avg_val_losses.get('vel_loss', 0):.4f}"
            )

            with summary_writer.as_default():
                for key, val in avg_val_losses.items():
                    tf.summary.scalar(f"val/{key}", val, step=global_step)

            if val_total < best_val_loss:
                best_val_loss = val_total
                model.save_weights(os.path.join(checkpoint_dir, "best_model.weights.h5"))
                print(f"  [BEST] New best val loss: {best_val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % cfg["save_interval"] == 0:
            save_path = ckpt_manager.save()
            print(f"  Checkpoint saved: {save_path}")

    # Final save
    print("\n[INFO] Training complete!")
    model.save_weights(os.path.join(checkpoint_dir, "final_model.weights.h5"))

    # Export as SavedModel
    export_path = os.path.join(checkpoint_dir, "saved_model")
    print(f"[INFO] Exporting SavedModel to: {export_path}")

    @tf.function(input_signature=[{
        "pillar_features": tf.TensorSpec([None, m_cfg["max_pillars"], m_cfg["max_points_per_pillar"], 9], tf.float32),
        "pillar_mask": tf.TensorSpec([None, m_cfg["max_pillars"], m_cfg["max_points_per_pillar"]], tf.float32),
        "pillar_coords": tf.TensorSpec([None, m_cfg["max_pillars"], 2], tf.int32),
    }])
    def serve(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return model(inputs, training=False)

    tf.saved_model.save(model, export_path, signatures={"serving_default": serve})
    print("[INFO] SavedModel exported successfully")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RadarPillarNet model")
    parser.add_argument("--data-root", type=str, default="/data/nuscenes", help="nuScenes data root")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--base-lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--max-lr", type=float, default=3e-3, help="Max learning rate (one-cycle)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--log-dir", type=str, default="./logs/radar_pillarnet", help="TensorBoard log dir")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/radar_pillarnet", help="Checkpoint dir")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--single-gpu", action="store_true", help="Force single GPU training")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--grad-clip", type=float, default=35.0, help="Gradient clip norm")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override config from args
    overrides: Dict[str, Any] = {
        "data_root": args.data_root,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "base_lr": args.base_lr,
        "max_lr": args.max_lr,
        "weight_decay": args.weight_decay,
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "mixed_precision": not args.no_mixed_precision,
        "multi_gpu": not args.single_gpu,
        "grad_clip_norm": args.grad_clip,
    }

    # Load external config if provided
    model_overrides: Optional[Dict[str, Any]] = None
    if args.config:
        with open(args.config, "r") as f:
            ext_config = json.load(f)
        if "train" in ext_config:
            overrides.update(ext_config["train"])
        if "model" in ext_config:
            model_overrides = ext_config["model"]

    # Handle resume
    if args.resume:
        overrides["checkpoint_dir"] = args.resume

    train(train_config=overrides, model_config=model_overrides)
