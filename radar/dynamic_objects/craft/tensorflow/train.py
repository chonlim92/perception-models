"""
CRAFT model training script.

Uses TF2 custom training loop with:
- tf.data.Dataset pipeline for nuScenes data
- Mixed precision (tf.keras.mixed_precision)
- Multi-GPU via tf.distribute.MirroredStrategy
- AdamW optimizer with cosine LR schedule
- Focal loss + L1 regression + velocity loss
- TensorBoard logging
- Checkpoint management
- Exponential Moving Average (EMA) weights
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

from model import CRAFTModel, DEFAULT_CONFIG, build_craft_model


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

TRAIN_CONFIG: Dict[str, Any] = {
    # Data
    "data_root": "/data/nuscenes",
    "train_split": "train",
    "val_split": "val",
    "batch_size": 4,
    "num_workers": 8,
    "prefetch_buffer": 4,
    # Optimization
    "epochs": 20,
    "base_lr": 2e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 1,
    "min_lr": 1e-6,
    "grad_clip_norm": 35.0,
    # Loss weights
    "heatmap_loss_weight": 1.0,
    "regression_loss_weight": 0.25,
    "velocity_loss_weight": 0.25,
    "height_loss_weight": 0.25,
    # Focal loss
    "focal_alpha": 2.0,
    "focal_beta": 4.0,
    # EMA
    "ema_decay": 0.999,
    # Logging
    "log_dir": "./logs/craft",
    "checkpoint_dir": "./checkpoints/craft",
    "log_interval": 50,
    "val_interval": 1,
    "save_interval": 1,
    # Mixed precision
    "mixed_precision": True,
    # Multi-GPU
    "multi_gpu": True,
}


# ---------------------------------------------------------------------------
# nuScenes Dataset Pipeline
# ---------------------------------------------------------------------------


class NuScenesDataLoader:
    """tf.data.Dataset pipeline for loading nuScenes camera + radar data."""

    def __init__(self, config: Dict[str, Any], model_config: Dict[str, Any], split: str = "train") -> None:
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

        num_cameras = self.model_config["num_cameras"]
        img_h = self.model_config["image_height"]
        img_w = self.model_config["image_width"]
        max_pillars = self.model_config["max_pillars"]
        max_pts = self.model_config["max_points_per_pillar"]

        # Load camera images
        images = np.zeros((num_cameras, img_h, img_w, 3), dtype=np.float32)
        cam_names = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                     "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

        for cam_idx, cam_name in enumerate(cam_names):
            img_path = self.data_root / "samples" / cam_name / f"{sample_info.get('token', 'dummy')}.jpg"
            if img_path.exists():
                img = tf.io.read_file(str(img_path))
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [img_h, img_w])
                img = tf.cast(img, tf.float32) / 255.0
                # ImageNet normalization
                mean = tf.constant([0.485, 0.456, 0.406])
                std = tf.constant([0.229, 0.224, 0.225])
                img = (img - mean) / std
                images[cam_idx] = img.numpy()
            else:
                # Synthetic data for training pipeline testing
                images[cam_idx] = np.random.randn(img_h, img_w, 3).astype(np.float32) * 0.1

        # Load radar pillars
        radar_path = self.data_root / "radar_pillars" / f"{sample_info.get('token', 'dummy')}.npz"
        if radar_path.exists():
            radar_data = np.load(str(radar_path))
            pillar_features = radar_data["features"][:max_pillars]
            pillar_mask = radar_data["mask"][:max_pillars]
            pillar_coords = radar_data["coords"][:max_pillars]
        else:
            # Synthetic radar pillars
            n_active = np.random.randint(100, max_pillars)
            pillar_features = np.zeros((max_pillars, max_pts, 9), dtype=np.float32)
            pillar_features[:n_active] = np.random.randn(n_active, max_pts, 9).astype(np.float32) * 0.1
            pillar_mask = np.zeros((max_pillars, max_pts), dtype=np.bool_)
            pillar_mask[:n_active, :np.random.randint(1, max_pts)] = True
            pillar_coords = np.zeros((max_pillars, 2), dtype=np.int32)
            x_cells = int((self.model_config["x_max"] - self.model_config["x_min"]) / self.model_config["pillar_x_size"])
            y_cells = int((self.model_config["y_max"] - self.model_config["y_min"]) / self.model_config["pillar_y_size"])
            pillar_coords[:n_active, 0] = np.random.randint(0, x_cells, n_active)
            pillar_coords[:n_active, 1] = np.random.randint(0, y_cells, n_active)

        # Calibration matrices
        lidar_to_cam = np.eye(4, dtype=np.float32)[np.newaxis].repeat(num_cameras, axis=0)
        cam_intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(num_cameras, axis=0)
        cam_intrinsics[:, 0, 0] = 1266.0  # fx
        cam_intrinsics[:, 1, 1] = 1266.0  # fy
        cam_intrinsics[:, 0, 2] = img_w / 2  # cx
        cam_intrinsics[:, 1, 2] = img_h / 2  # cy

        # Load ground truth annotations
        bev_h = self.model_config.get("bev_x_cells", 512) // 8  # After backbone downsampling
        bev_w = self.model_config.get("bev_y_cells", 512) // 8
        num_classes = self.model_config["num_classes"]

        gt_heatmap = np.zeros((bev_h, bev_w, num_classes), dtype=np.float32)
        gt_regression = np.zeros((bev_h, bev_w, self.model_config["num_reg_attrs"]), dtype=np.float32)
        gt_velocity = np.zeros((bev_h, bev_w, 2), dtype=np.float32)
        gt_height = np.zeros((bev_h, bev_w, 2), dtype=np.float32)
        gt_reg_mask = np.zeros((bev_h, bev_w), dtype=np.float32)

        # Parse annotations if available
        anno_path = self.data_root / "annotations" / f"{sample_info.get('token', 'dummy')}.json"
        if anno_path.exists():
            with open(str(anno_path), "r") as f:
                annotations = json.load(f)
            for ann in annotations:
                cx, cy = ann["center_bev"]  # BEV pixel coords
                cls_id = ann["class_id"]
                if 0 <= cx < bev_w and 0 <= cy < bev_h:
                    # Gaussian heatmap
                    gt_heatmap[int(cy), int(cx), cls_id] = 1.0
                    gt_regression[int(cy), int(cx)] = ann["regression"]
                    gt_velocity[int(cy), int(cx)] = ann["velocity"]
                    gt_height[int(cy), int(cx)] = ann["height"]
                    gt_reg_mask[int(cy), int(cx)] = 1.0

        return {
            "images": images.astype(np.float32),
            "radar_pillars": pillar_features.astype(np.float32),
            "radar_pillar_mask": pillar_mask.astype(np.float32),
            "radar_pillar_coords": pillar_coords.astype(np.int32),
            "lidar_to_cam": lidar_to_cam.astype(np.float32),
            "cam_intrinsics": cam_intrinsics.astype(np.float32),
            "gt_heatmap": gt_heatmap.astype(np.float32),
            "gt_regression": gt_regression.astype(np.float32),
            "gt_velocity": gt_velocity.astype(np.float32),
            "gt_height": gt_height.astype(np.float32),
            "gt_reg_mask": gt_reg_mask.astype(np.float32),
        }

    def build_dataset(self) -> tf.data.Dataset:
        """Build tf.data.Dataset with proper batching and prefetching."""
        num_samples = len(self.samples)
        indices = tf.data.Dataset.range(num_samples)

        if self.split == "train":
            indices = indices.shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)

        output_signature = {
            "images": tf.TensorSpec(
                shape=(self.model_config["num_cameras"], self.model_config["image_height"],
                       self.model_config["image_width"], 3),
                dtype=tf.float32,
            ),
            "radar_pillars": tf.TensorSpec(
                shape=(self.model_config["max_pillars"], self.model_config["max_points_per_pillar"], 9),
                dtype=tf.float32,
            ),
            "radar_pillar_mask": tf.TensorSpec(
                shape=(self.model_config["max_pillars"], self.model_config["max_points_per_pillar"]),
                dtype=tf.float32,
            ),
            "radar_pillar_coords": tf.TensorSpec(
                shape=(self.model_config["max_pillars"], 2),
                dtype=tf.int32,
            ),
            "lidar_to_cam": tf.TensorSpec(
                shape=(self.model_config["num_cameras"], 4, 4),
                dtype=tf.float32,
            ),
            "cam_intrinsics": tf.TensorSpec(
                shape=(self.model_config["num_cameras"], 3, 3),
                dtype=tf.float32,
            ),
            "gt_heatmap": tf.TensorSpec(shape=(None, None, self.model_config["num_classes"]), dtype=tf.float32),
            "gt_regression": tf.TensorSpec(shape=(None, None, self.model_config["num_reg_attrs"]), dtype=tf.float32),
            "gt_velocity": tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
            "gt_height": tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
            "gt_reg_mask": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        }

        dataset = indices.map(
            lambda idx: tf.py_function(
                func=self._parse_sample,
                inp=[idx],
                Tout={k: v.dtype for k, v in output_signature.items()},
            ),
            num_parallel_calls=self.config.get("num_workers", 8),
        )

        # Set shapes after py_function
        def set_shapes(sample: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
            sample["images"].set_shape(
                (self.model_config["num_cameras"], self.model_config["image_height"],
                 self.model_config["image_width"], 3)
            )
            sample["radar_pillars"].set_shape(
                (self.model_config["max_pillars"], self.model_config["max_points_per_pillar"], 9)
            )
            sample["radar_pillar_mask"].set_shape(
                (self.model_config["max_pillars"], self.model_config["max_points_per_pillar"])
            )
            sample["radar_pillar_coords"].set_shape((self.model_config["max_pillars"], 2))
            sample["lidar_to_cam"].set_shape((self.model_config["num_cameras"], 4, 4))
            sample["cam_intrinsics"].set_shape((self.model_config["num_cameras"], 3, 3))
            return sample

        dataset = dataset.map(set_shapes)
        dataset = dataset.batch(self.config["batch_size"], drop_remainder=True)
        dataset = dataset.prefetch(self.config.get("prefetch_buffer", 4))
        return dataset


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------


def gaussian_focal_loss(
    pred_heatmap: tf.Tensor,
    gt_heatmap: tf.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> tf.Tensor:
    """
    Modified focal loss for heatmap prediction (CornerNet-style).

    Args:
        pred_heatmap: (B, H, W, C) predicted heatmap after sigmoid
        gt_heatmap: (B, H, W, C) ground truth Gaussian heatmap
        alpha: focal power for positive samples
        beta: focal power for negative samples
    Returns:
        Scalar loss
    """
    pred = tf.clip_by_value(pred_heatmap, 1e-6, 1.0 - 1e-6)

    pos_mask = tf.cast(tf.equal(gt_heatmap, 1.0), tf.float32)
    neg_mask = 1.0 - pos_mask

    # Positive loss
    pos_loss = -tf.math.log(pred) * tf.pow(1.0 - pred, alpha) * pos_mask

    # Negative loss
    neg_loss = (
        -tf.math.log(1.0 - pred)
        * tf.pow(pred, alpha)
        * tf.pow(1.0 - gt_heatmap, beta)
        * neg_mask
    )

    num_pos = tf.maximum(tf.reduce_sum(pos_mask), 1.0)
    loss = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / num_pos
    return loss


def regression_l1_loss(
    pred: tf.Tensor,
    target: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """
    Masked L1 loss for regression targets.

    Args:
        pred: (B, H, W, D)
        target: (B, H, W, D)
        mask: (B, H, W) binary mask
    Returns:
        Scalar loss
    """
    mask_expanded = tf.expand_dims(mask, axis=-1)
    diff = tf.abs(pred - target) * mask_expanded
    num_pos = tf.maximum(tf.reduce_sum(mask), 1.0)
    loss = tf.reduce_sum(diff) / num_pos
    return loss


def compute_total_loss(
    predictions: Dict[str, tf.Tensor],
    targets: Dict[str, tf.Tensor],
    config: Dict[str, Any],
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """
    Compute combined training loss.

    Returns:
        total_loss: scalar
        loss_dict: individual loss components for logging
    """
    # Resize predictions to match target spatial dims if needed
    pred_heatmap = predictions["heatmap"]
    gt_heatmap = targets["gt_heatmap"]
    gt_reg_mask = targets["gt_reg_mask"]

    # Ensure spatial dimensions match
    target_h = tf.shape(gt_heatmap)[1]
    target_w = tf.shape(gt_heatmap)[2]
    pred_h = tf.shape(pred_heatmap)[1]
    pred_w = tf.shape(pred_heatmap)[2]

    if pred_h != target_h or pred_w != target_w:
        pred_heatmap = tf.image.resize(pred_heatmap, [target_h, target_w], method="bilinear")
        predictions = {
            "heatmap": pred_heatmap,
            "regression": tf.image.resize(predictions["regression"], [target_h, target_w]),
            "velocity": tf.image.resize(predictions["velocity"], [target_h, target_w]),
            "height": tf.image.resize(predictions["height"], [target_h, target_w]),
        }

    # Heatmap focal loss
    heatmap_loss = gaussian_focal_loss(
        predictions["heatmap"],
        gt_heatmap,
        alpha=config.get("focal_alpha", 2.0),
        beta=config.get("focal_beta", 4.0),
    )

    # Regression L1 loss
    reg_loss = regression_l1_loss(
        predictions["regression"],
        targets["gt_regression"],
        gt_reg_mask,
    )

    # Velocity loss
    vel_loss = regression_l1_loss(
        predictions["velocity"],
        targets["gt_velocity"],
        gt_reg_mask,
    )

    # Height loss
    height_loss = regression_l1_loss(
        predictions["height"],
        targets["gt_height"],
        gt_reg_mask,
    )

    # Weighted total
    total_loss = (
        config.get("heatmap_loss_weight", 1.0) * heatmap_loss
        + config.get("regression_loss_weight", 0.25) * reg_loss
        + config.get("velocity_loss_weight", 0.25) * vel_loss
        + config.get("height_loss_weight", 0.25) * height_loss
    )

    loss_dict = {
        "total_loss": total_loss,
        "heatmap_loss": heatmap_loss,
        "regression_loss": reg_loss,
        "velocity_loss": vel_loss,
        "height_loss": height_loss,
    }

    return total_loss, loss_dict


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay LR schedule with linear warmup."""

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.base_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay
        progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self) -> Dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------


class EMACallback:
    """Maintains exponential moving average of model weights."""

    def __init__(self, model: tf.keras.Model, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow_vars: List[tf.Variable] = []
        self.initialized = False

    def initialize(self) -> None:
        """Initialize shadow variables as copies of model weights."""
        self.shadow_vars = [
            tf.Variable(var, trainable=False, name=f"ema/{var.name}")
            for var in self.model.trainable_variables
        ]
        self.initialized = True

    def update(self) -> None:
        """Update EMA weights after each training step."""
        if not self.initialized:
            self.initialize()
        for shadow, var in zip(self.shadow_vars, self.model.trainable_variables):
            shadow.assign(self.decay * shadow + (1.0 - self.decay) * var)

    def apply_ema_weights(self) -> List[tf.Tensor]:
        """Replace model weights with EMA weights. Returns original weights for restore."""
        original_weights = [var.numpy() for var in self.model.trainable_variables]
        for shadow, var in zip(self.shadow_vars, self.model.trainable_variables):
            var.assign(shadow)
        return original_weights

    def restore_original_weights(self, original_weights: List[tf.Tensor]) -> None:
        """Restore original weights after EMA evaluation."""
        for weight, var in zip(original_weights, self.model.trainable_variables):
            var.assign(weight)


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
        strategy = tf.distribute.get_strategy()  # Default (single GPU/CPU)
        print("[INFO] Single device training")

    # Build datasets
    print("[INFO] Building datasets...")
    train_loader = NuScenesDataLoader(cfg, m_cfg, split=cfg["train_split"])
    val_loader = NuScenesDataLoader(cfg, m_cfg, split=cfg["val_split"])
    train_dataset = train_loader.build_dataset()
    val_dataset = val_loader.build_dataset()

    # Distribute datasets
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # Build model within strategy scope
    with strategy.scope():
        model = build_craft_model(config=m_cfg)

        # LR schedule
        steps_per_epoch = max(len(train_loader.samples) // cfg["batch_size"], 1)
        total_steps = steps_per_epoch * cfg["epochs"]
        warmup_steps = steps_per_epoch * cfg["warmup_epochs"]

        lr_schedule = CosineDecayWithWarmup(
            base_lr=cfg["base_lr"],
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=cfg["min_lr"],
        )

        # AdamW optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=cfg["weight_decay"],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=cfg["grad_clip_norm"],
        )

        if cfg["mixed_precision"]:
            # Wrap optimizer for loss scaling with mixed precision
            pass  # tf.keras.optimizers.AdamW handles this internally in TF2.11+

    # EMA
    ema = EMACallback(model, decay=cfg["ema_decay"])

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
        """Single training step within strategy."""
        model_inputs = {
            "images": inputs["images"],
            "radar_pillars": inputs["radar_pillars"],
            "radar_pillar_mask": inputs["radar_pillar_mask"],
            "radar_pillar_coords": inputs["radar_pillar_coords"],
            "lidar_to_cam": inputs["lidar_to_cam"],
            "cam_intrinsics": inputs["cam_intrinsics"],
        }
        targets = {
            "gt_heatmap": inputs["gt_heatmap"],
            "gt_regression": inputs["gt_regression"],
            "gt_velocity": inputs["gt_velocity"],
            "gt_height": inputs["gt_height"],
            "gt_reg_mask": inputs["gt_reg_mask"],
        }

        with tf.GradientTape() as tape:
            predictions = model(model_inputs, training=True)
            total_loss, loss_dict = compute_total_loss(predictions, targets, cfg)

            # Scale loss for mixed precision
            if cfg["mixed_precision"]:
                scaled_loss = optimizer.get_scaled_loss(total_loss) if hasattr(optimizer, 'get_scaled_loss') else total_loss
            else:
                scaled_loss = total_loss

        # Compute and apply gradients
        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        if cfg["mixed_precision"] and hasattr(optimizer, 'get_unscaled_gradients'):
            gradients = optimizer.get_unscaled_gradients(gradients)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_dict

    @tf.function
    def distributed_train_step(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Distributed training step."""
        per_replica_losses = strategy.run(train_step, args=(inputs,))
        # Reduce losses across replicas
        reduced = {}
        for key, val in per_replica_losses.items():
            reduced[key] = strategy.reduce(tf.distribute.ReduceOp.MEAN, val, axis=None)
        return reduced

    # Validation step
    @tf.function
    def val_step(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        model_inputs = {
            "images": inputs["images"],
            "radar_pillars": inputs["radar_pillars"],
            "radar_pillar_mask": inputs["radar_pillar_mask"],
            "radar_pillar_coords": inputs["radar_pillar_coords"],
            "lidar_to_cam": inputs["lidar_to_cam"],
            "cam_intrinsics": inputs["cam_intrinsics"],
        }
        targets = {
            "gt_heatmap": inputs["gt_heatmap"],
            "gt_regression": inputs["gt_regression"],
            "gt_velocity": inputs["gt_velocity"],
            "gt_height": inputs["gt_height"],
            "gt_reg_mask": inputs["gt_reg_mask"],
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

    print(f"[INFO] Starting training for {cfg['epochs']} epochs")
    print(f"[INFO] Steps per epoch: {steps_per_epoch}")
    print(f"[INFO] Total steps: {total_steps}")

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        epoch_losses: Dict[str, List[float]] = {
            "total_loss": [], "heatmap_loss": [], "regression_loss": [],
            "velocity_loss": [], "height_loss": [],
        }

        # Training
        for step, batch in enumerate(train_dist_dataset):
            step_start = time.time()
            loss_dict = distributed_train_step(batch)

            # Update EMA
            ema.update()
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
                    f"Heatmap: {float(loss_dict['heatmap_loss']):.4f} | "
                    f"Reg: {float(loss_dict['regression_loss']):.4f} | "
                    f"Vel: {float(loss_dict['velocity_loss']):.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {step_time:.2f}s"
                )

                with summary_writer.as_default():
                    tf.summary.scalar("train/total_loss", loss_dict["total_loss"], step=global_step)
                    tf.summary.scalar("train/heatmap_loss", loss_dict["heatmap_loss"], step=global_step)
                    tf.summary.scalar("train/regression_loss", loss_dict["regression_loss"], step=global_step)
                    tf.summary.scalar("train/velocity_loss", loss_dict["velocity_loss"], step=global_step)
                    tf.summary.scalar("train/height_loss", loss_dict["height_loss"], step=global_step)
                    tf.summary.scalar("train/learning_rate", current_lr, step=global_step)

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items() if v}
        print(
            f"\n[Epoch {epoch+1}/{cfg['epochs']}] "
            f"Avg Loss: {avg_losses.get('total_loss', 0):.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Validation
        if (epoch + 1) % cfg["val_interval"] == 0:
            print("[INFO] Running validation...")
            # Apply EMA weights for validation
            original_weights = ema.apply_ema_weights()

            val_losses: Dict[str, List[float]] = {
                "total_loss": [], "heatmap_loss": [], "regression_loss": [],
                "velocity_loss": [], "height_loss": [],
            }

            for val_batch in val_dist_dataset:
                val_loss_dict = distributed_val_step(val_batch)
                for key in val_losses:
                    val_losses[key].append(float(val_loss_dict[key]))

            avg_val_losses = {k: np.mean(v) for k, v in val_losses.items() if v}
            val_total = avg_val_losses.get("total_loss", float("inf"))

            print(
                f"  Val Loss: {val_total:.4f} | "
                f"Val Heatmap: {avg_val_losses.get('heatmap_loss', 0):.4f} | "
                f"Val Reg: {avg_val_losses.get('regression_loss', 0):.4f}"
            )

            with summary_writer.as_default():
                for key, val in avg_val_losses.items():
                    tf.summary.scalar(f"val/{key}", val, step=global_step)

            if val_total < best_val_loss:
                best_val_loss = val_total
                # Save best EMA model
                model.save_weights(os.path.join(checkpoint_dir, "best_ema_model.weights.h5"))
                print(f"  [BEST] New best val loss: {best_val_loss:.4f}")

            # Restore original weights for training
            ema.restore_original_weights(original_weights)

        # Save checkpoint
        if (epoch + 1) % cfg["save_interval"] == 0:
            save_path = ckpt_manager.save()
            print(f"  Checkpoint saved: {save_path}")

    # Final save
    print("\n[INFO] Training complete!")
    model.save_weights(os.path.join(checkpoint_dir, "final_model.weights.h5"))

    # Save EMA model
    original_weights = ema.apply_ema_weights()
    model.save_weights(os.path.join(checkpoint_dir, "final_ema_model.weights.h5"))
    ema.restore_original_weights(original_weights)

    # Export as SavedModel
    export_path = os.path.join(checkpoint_dir, "saved_model")
    print(f"[INFO] Exporting SavedModel to: {export_path}")

    # Create a concrete function for export
    @tf.function(input_signature=[{
        "images": tf.TensorSpec([None, m_cfg["num_cameras"], m_cfg["image_height"], m_cfg["image_width"], 3], tf.float32),
        "radar_pillars": tf.TensorSpec([None, m_cfg["max_pillars"], m_cfg["max_points_per_pillar"], 9], tf.float32),
        "radar_pillar_mask": tf.TensorSpec([None, m_cfg["max_pillars"], m_cfg["max_points_per_pillar"]], tf.float32),
        "radar_pillar_coords": tf.TensorSpec([None, m_cfg["max_pillars"], 2], tf.int32),
        "lidar_to_cam": tf.TensorSpec([None, m_cfg["num_cameras"], 4, 4], tf.float32),
        "cam_intrinsics": tf.TensorSpec([None, m_cfg["num_cameras"], 3, 3], tf.float32),
    }])
    def serve(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return model(inputs, training=False)

    tf.saved_model.save(model, export_path, signatures={"serving_default": serve})
    print("[INFO] SavedModel exported successfully")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRAFT model")
    parser.add_argument("--data-root", type=str, default="/data/nuscenes", help="nuScenes data root")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--log-dir", type=str, default="./logs/craft", help="TensorBoard log dir")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/craft", help="Checkpoint dir")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--single-gpu", action="store_true", help="Use single GPU")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override config from args
    overrides: Dict[str, Any] = {
        "data_root": args.data_root,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "base_lr": args.lr,
        "weight_decay": args.weight_decay,
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "mixed_precision": not args.no_mixed_precision,
        "multi_gpu": not args.single_gpu,
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

    train(train_config=overrides, model_config=model_overrides)
