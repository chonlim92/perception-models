"""
TensorFlow 2 training script for Radar Occupancy models.

Implements:
- FocalLoss for occupancy prediction (binary, ignoring unknown cells)
- SemanticLoss (weighted cross-entropy, ignoring unknown cells)
- Custom training loop with tf.GradientTape and mixed precision
- Cosine learning rate schedule with linear warmup
- Checkpoint saving (best mIoU + latest)
- TensorBoard logging
- IoU metrics computation
"""

import argparse
import os
import time
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from model import PillarOccNet, TemporalPillarOccNet, build_model


# =============================================================================
# Learning Rate Schedule
# =============================================================================

class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine learning rate decay with linear warmup."""

    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-6):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Linear warmup phase
        warmup_lr = self.base_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay phase
        progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * progress))
        decay_lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

        # Select based on whether we are in warmup or decay
        lr = tf.where(step < warmup_steps, warmup_lr, decay_lr)
        return lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


# =============================================================================
# Loss Functions
# =============================================================================

class FocalLoss:
    """
    Focal loss for binary occupancy prediction.

    Targets: 0=free, 1=occupied, 2=unknown (ignored).
    Predictions: logits of shape (B, H, W, 1).
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, pred_logits, target):
        """
        Args:
            pred_logits: (B, H, W, 1) raw logits for occupancy.
            target: (B, H, W) integer tensor with values 0, 1, 2.

        Returns:
            Scalar mean focal loss over valid (non-unknown) cells.
        """
        pred_logits = tf.squeeze(pred_logits, axis=-1)  # (B, H, W)

        # Create mask for valid cells (not unknown=2)
        valid_mask = tf.not_equal(target, 2)
        valid_mask_float = tf.cast(valid_mask, tf.float32)

        # Binary target: occupied=1, free=0
        binary_target = tf.cast(tf.equal(target, 1), tf.float32)

        # Compute sigmoid probabilities
        pred_prob = tf.sigmoid(pred_logits)

        # Compute binary cross-entropy (element-wise, no reduction)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=binary_target, logits=pred_logits
        )

        # Compute p_t for focal weighting
        p_t = binary_target * pred_prob + (1.0 - binary_target) * (1.0 - pred_prob)

        # Alpha weighting
        alpha_t = binary_target * self.alpha + (1.0 - binary_target) * (1.0 - self.alpha)

        # Focal weight
        focal_weight = alpha_t * tf.pow(1.0 - p_t, self.gamma)

        # Focal loss per element
        focal_loss = focal_weight * bce

        # Apply valid mask and compute mean over valid cells
        focal_loss = focal_loss * valid_mask_float
        num_valid = tf.maximum(tf.reduce_sum(valid_mask_float), 1.0)
        loss = tf.reduce_sum(focal_loss) / num_valid

        return loss


class SemanticLoss:
    """
    Weighted cross-entropy loss for semantic class prediction.

    Ignores cells with target == ignore_index (unknown).
    """

    def __init__(self, num_classes, class_weights=None, ignore_index=2):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        else:
            self.class_weights = None

    def __call__(self, pred_logits, target):
        """
        Args:
            pred_logits: (B, H, W, K) raw logits for K semantic classes.
            target: (B, H, W) integer class indices.

        Returns:
            Scalar mean weighted cross-entropy loss over valid cells.
        """
        # Create valid mask (exclude ignore_index)
        valid_mask = tf.not_equal(target, self.ignore_index)
        valid_mask_float = tf.cast(valid_mask, tf.float32)

        # Clamp target to valid range for gather (replace ignore_index with 0 temporarily)
        safe_target = tf.where(valid_mask, target, tf.zeros_like(target))

        # Compute cross-entropy
        # sparse_softmax_cross_entropy_with_logits expects (batch, classes) input
        # We need to reshape for the computation
        shape = tf.shape(pred_logits)
        B, H, W, K = shape[0], shape[1], shape[2], shape[3]

        # Flatten spatial dims
        pred_flat = tf.reshape(pred_logits, [-1, K])  # (B*H*W, K)
        target_flat = tf.reshape(safe_target, [-1])  # (B*H*W,)
        valid_flat = tf.reshape(valid_mask_float, [-1])  # (B*H*W,)

        # Compute per-element cross-entropy
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_flat, logits=pred_flat
        )

        # Apply class weights if provided
        if self.class_weights is not None:
            sample_weights = tf.gather(self.class_weights, target_flat)
            ce_loss = ce_loss * sample_weights

        # Apply valid mask
        ce_loss = ce_loss * valid_flat
        num_valid = tf.maximum(tf.reduce_sum(valid_flat), 1.0)
        loss = tf.reduce_sum(ce_loss) / num_valid

        return loss


# =============================================================================
# Metrics
# =============================================================================

def compute_iou(pred_logits, target):
    """
    Compute IoU for occupied and free classes.

    Args:
        pred_logits: (B, H, W, 1) raw logits for occupancy.
        target: (B, H, W) integer tensor with values 0=free, 1=occupied, 2=unknown.

    Returns:
        Tuple (occupied_iou, free_iou) as float scalars.
    """
    pred_logits = tf.squeeze(pred_logits, axis=-1)  # (B, H, W)

    # Valid mask (exclude unknown=2)
    valid_mask = tf.not_equal(target, 2)

    # Predicted classes (threshold at 0)
    pred_class = tf.cast(pred_logits > 0.0, tf.int32)

    # Ground truth binary
    gt_class = tf.cast(tf.equal(target, 1), tf.int32)

    # Apply valid mask
    pred_valid = tf.where(valid_mask, pred_class, -1 * tf.ones_like(pred_class))
    gt_valid = tf.where(valid_mask, gt_class, -1 * tf.ones_like(gt_class))

    # Occupied IoU: pred=1 and gt=1
    pred_occ = tf.equal(pred_valid, 1)
    gt_occ = tf.equal(gt_valid, 1)
    intersection_occ = tf.reduce_sum(tf.cast(tf.logical_and(pred_occ, gt_occ), tf.float32))
    union_occ = tf.reduce_sum(tf.cast(tf.logical_or(pred_occ, gt_occ), tf.float32))
    occupied_iou = intersection_occ / tf.maximum(union_occ, 1.0)

    # Free IoU: pred=0 and gt=0 (within valid mask)
    pred_free = tf.logical_and(tf.equal(pred_valid, 0), valid_mask)
    gt_free = tf.logical_and(tf.equal(gt_valid, 0), valid_mask)
    intersection_free = tf.reduce_sum(tf.cast(tf.logical_and(pred_free, gt_free), tf.float32))
    union_free = tf.reduce_sum(tf.cast(tf.logical_or(pred_free, gt_free), tf.float32))
    free_iou = intersection_free / tf.maximum(union_free, 1.0)

    return occupied_iou, free_iou


# =============================================================================
# Dataset
# =============================================================================

class RadarOccupancyDataset:
    """
    Placeholder radar occupancy dataset that yields training samples.

    Each sample is a dictionary with:
        - pillar_features: (max_pillars, max_points_per_pillar, feature_dim)
        - pillar_indices: (max_pillars, 2) grid indices for each pillar
        - num_pillars: scalar, number of valid pillars
        - occupancy_gt: (H, W) ground truth occupancy map (0=free, 1=occupied, 2=unknown)
        - semantic_gt: (H, W) ground truth semantic labels (optional)
    """

    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.grid_size = config["grid"]["grid_size"]  # [H, W]
        self.max_pillars = config.get("model", {}).get("pillar", {}).get("max_pillars", 10000)
        self.max_points_per_pillar = config.get("model", {}).get("pillar", {}).get("max_points", 32)
        self.feature_dim = config.get("model", {}).get("pillar", {}).get("feature_dim", 7)
        self.num_semantic_classes = config.get("model", {}).get("heads", {}).get("num_semantic_classes", 5)

        # Determine dataset size based on split
        if split == "train":
            self.num_samples = 2000
        else:
            self.num_samples = 500

    def __len__(self):
        return self.num_samples

    def generator(self):
        """Generator that yields individual samples."""
        rng = np.random.default_rng(seed=42 if self.split == "val" else None)
        H, W = self.grid_size

        for _ in range(self.num_samples):
            num_pillars = rng.integers(100, self.max_pillars)

            pillar_features = rng.standard_normal(
                (self.max_pillars, self.max_points_per_pillar, self.feature_dim)
            ).astype(np.float32)

            pillar_indices = np.zeros((self.max_pillars, 2), dtype=np.int32)
            pillar_indices[:num_pillars, 0] = rng.integers(0, H, size=num_pillars)
            pillar_indices[:num_pillars, 1] = rng.integers(0, W, size=num_pillars)

            # Occupancy ground truth: mostly free, some occupied, some unknown
            occupancy_gt = np.zeros((H, W), dtype=np.int32)
            num_occupied = rng.integers(50, 500)
            occ_rows = rng.integers(0, H, size=num_occupied)
            occ_cols = rng.integers(0, W, size=num_occupied)
            occupancy_gt[occ_rows, occ_cols] = 1
            # Mark some cells as unknown
            num_unknown = rng.integers(100, 1000)
            unk_rows = rng.integers(0, H, size=num_unknown)
            unk_cols = rng.integers(0, W, size=num_unknown)
            occupancy_gt[unk_rows, unk_cols] = 2

            # Semantic ground truth
            semantic_gt = rng.integers(0, self.num_semantic_classes, size=(H, W)).astype(np.int32)
            # Mark unknown cells in semantic as well
            semantic_gt[unk_rows, unk_cols] = 2  # use same ignore index

            yield {
                "pillar_features": pillar_features,
                "pillar_indices": pillar_indices,
                "num_pillars": np.int32(num_pillars),
                "occupancy_gt": occupancy_gt,
                "semantic_gt": semantic_gt,
            }

    def create_tf_dataset(self, batch_size, shuffle=True):
        """Create a tf.data.Dataset from this dataset."""
        H, W = self.grid_size

        output_signature = {
            "pillar_features": tf.TensorSpec(
                shape=(self.max_pillars, self.max_points_per_pillar, self.feature_dim),
                dtype=tf.float32,
            ),
            "pillar_indices": tf.TensorSpec(
                shape=(self.max_pillars, 2), dtype=tf.int32
            ),
            "num_pillars": tf.TensorSpec(shape=(), dtype=tf.int32),
            "occupancy_gt": tf.TensorSpec(shape=(H, W), dtype=tf.int32),
            "semantic_gt": tf.TensorSpec(shape=(H, W), dtype=tf.int32),
        }

        dataset = tf.data.Dataset.from_generator(
            self.generator, output_signature=output_signature
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, self.num_samples))

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


# =============================================================================
# Training Step
# =============================================================================

@tf.function
def train_step(model, batch, focal_loss_fn, semantic_loss_fn,
               optimizer, occ_weight, sem_weight, max_grad_norm):
    """
    Single training step with gradient tape, mixed precision, and gradient clipping.

    Args:
        model: The occupancy prediction model.
        batch: Dictionary of input tensors.
        focal_loss_fn: FocalLoss instance.
        semantic_loss_fn: SemanticLoss instance.
        optimizer: The optimizer (with loss scale for mixed precision).
        occ_weight: Weight for occupancy loss.
        sem_weight: Weight for semantic loss.
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Dictionary of loss values and metrics.
    """
    pillar_features = batch["pillar_features"]
    pillar_indices = batch["pillar_indices"]
    num_pillars = batch["num_pillars"]
    occupancy_gt = batch["occupancy_gt"]
    semantic_gt = batch["semantic_gt"]

    with tf.GradientTape() as tape:
        # Forward pass
        outputs = model(
            {"pillar_features": pillar_features,
             "pillar_indices": pillar_indices,
             "num_pillars": num_pillars},
            training=True,
        )

        occ_logits = outputs["occupancy"]  # (B, H, W, 1)
        sem_logits = outputs["semantic"]   # (B, H, W, K)

        # Compute losses
        occ_loss = focal_loss_fn(occ_logits, occupancy_gt)
        sem_loss = semantic_loss_fn(sem_logits, semantic_gt)
        total_loss = occ_weight * occ_loss + sem_weight * sem_loss

        # Scale loss for mixed precision
        scaled_loss = optimizer.get_scaled_loss(total_loss)

    # Compute and unscale gradients
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)

    # Clip gradients by global norm
    gradients, grad_norm = tf.clip_by_global_norm(gradients, max_grad_norm)

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Compute IoU metrics
    occupied_iou, free_iou = compute_iou(occ_logits, occupancy_gt)
    miou = (occupied_iou + free_iou) / 2.0

    return {
        "total_loss": total_loss,
        "occ_loss": occ_loss,
        "sem_loss": sem_loss,
        "occupied_iou": occupied_iou,
        "free_iou": free_iou,
        "miou": miou,
        "grad_norm": grad_norm,
    }


@tf.function
def val_step(model, batch, focal_loss_fn, semantic_loss_fn, occ_weight, sem_weight):
    """
    Single validation step (no gradient computation).

    Args:
        model: The occupancy prediction model.
        batch: Dictionary of input tensors.
        focal_loss_fn: FocalLoss instance.
        semantic_loss_fn: SemanticLoss instance.
        occ_weight: Weight for occupancy loss.
        sem_weight: Weight for semantic loss.

    Returns:
        Dictionary of loss values and metrics.
    """
    pillar_features = batch["pillar_features"]
    pillar_indices = batch["pillar_indices"]
    num_pillars = batch["num_pillars"]
    occupancy_gt = batch["occupancy_gt"]
    semantic_gt = batch["semantic_gt"]

    # Forward pass (no training)
    outputs = model(
        {"pillar_features": pillar_features,
         "pillar_indices": pillar_indices,
         "num_pillars": num_pillars},
        training=False,
    )

    occ_logits = outputs["occupancy"]  # (B, H, W, 1)
    sem_logits = outputs["semantic"]   # (B, H, W, K)

    # Compute losses
    occ_loss = focal_loss_fn(occ_logits, occupancy_gt)
    sem_loss = semantic_loss_fn(sem_logits, semantic_gt)
    total_loss = occ_weight * occ_loss + sem_weight * sem_loss

    # Compute IoU metrics
    occupied_iou, free_iou = compute_iou(occ_logits, occupancy_gt)
    miou = (occupied_iou + free_iou) / 2.0

    return {
        "total_loss": total_loss,
        "occ_loss": occ_loss,
        "sem_loss": sem_loss,
        "occupied_iou": occupied_iou,
        "free_iou": free_iou,
        "miou": miou,
    }


# =============================================================================
# Training Loop
# =============================================================================

def run_training(config, resume_dir=None, output_dir="./output"):
    """
    Main training loop.

    Args:
        config: Configuration dictionary loaded from YAML.
        resume_dir: Path to checkpoint directory for resuming training.
        output_dir: Directory to save checkpoints and logs.
    """
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Mixed precision
    use_mixed_precision = config.get("hardware", {}).get("mixed_precision", True)
    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision (float16) enabled.")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        print("[INFO] Using float32 precision.")

    # -------------------------------------------------------------------------
    # Training parameters
    # -------------------------------------------------------------------------
    training_cfg = config["training"]
    batch_size = training_cfg.get("batch_size", 8)
    num_epochs = training_cfg.get("num_epochs", 50)
    base_lr = training_cfg["optimizer"].get("lr", 1e-3)
    weight_decay = training_cfg["optimizer"].get("weight_decay", 1e-4)
    betas = training_cfg["optimizer"].get("betas", [0.9, 0.999])
    warmup_epochs = training_cfg["scheduler"].get("warmup_epochs", 5)
    min_lr = training_cfg["scheduler"].get("min_lr", 1e-6)
    occ_weight = training_cfg["loss"].get("occupancy_weight", 1.0)
    sem_weight = training_cfg["loss"].get("semantic_weight", 0.5)
    focal_alpha = training_cfg["loss"].get("focal_alpha", 0.75)
    focal_gamma = training_cfg["loss"].get("focal_gamma", 2.0)
    class_weights = training_cfg["loss"].get("class_weights", None)
    max_grad_norm = 10.0

    num_semantic_classes = config.get("model", {}).get("heads", {}).get("num_semantic_classes", 5)

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    print("[INFO] Creating datasets...")
    train_dataset_obj = RadarOccupancyDataset(config, split="train")
    val_dataset_obj = RadarOccupancyDataset(config, split="val")

    train_dataset = train_dataset_obj.create_tf_dataset(batch_size, shuffle=True)
    val_dataset = val_dataset_obj.create_tf_dataset(batch_size, shuffle=False)

    steps_per_epoch = len(train_dataset_obj) // batch_size
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    print(f"[INFO] Training samples: {len(train_dataset_obj)}, "
          f"Validation samples: {len(val_dataset_obj)}")
    print(f"[INFO] Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    print("[INFO] Building model...")
    model = build_model(config)
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # -------------------------------------------------------------------------
    # Optimizer and Schedule
    # -------------------------------------------------------------------------
    lr_schedule = CosineDecayWithWarmup(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr,
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        beta_1=betas[0],
        beta_2=betas[1],
        clipnorm=None,  # We clip manually in the train step
    )

    # Wrap optimizer for mixed precision loss scaling
    if use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # -------------------------------------------------------------------------
    # Loss Functions
    # -------------------------------------------------------------------------
    focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    semantic_loss_fn = SemanticLoss(
        num_classes=num_semantic_classes,
        class_weights=class_weights,
        ignore_index=2,
    )

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=3
    )

    start_epoch = 0
    global_step = 0
    best_miou = 0.0

    if resume_dir is not None:
        latest_ckpt = tf.train.latest_checkpoint(resume_dir)
        if latest_ckpt:
            status = checkpoint.restore(latest_ckpt)
            status.expect_partial()
            # Try to recover epoch from checkpoint filename
            ckpt_name = os.path.basename(latest_ckpt)
            print(f"[INFO] Resumed from checkpoint: {latest_ckpt}")
            # Attempt to load metadata
            meta_path = os.path.join(resume_dir, "training_meta.yaml")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = yaml.safe_load(f)
                start_epoch = meta.get("epoch", 0)
                global_step = meta.get("global_step", 0)
                best_miou = meta.get("best_miou", 0.0)
                print(f"[INFO] Resuming from epoch {start_epoch}, "
                      f"step {global_step}, best mIoU {best_miou:.4f}")
        else:
            print(f"[WARNING] No checkpoint found in {resume_dir}, training from scratch.")

    # -------------------------------------------------------------------------
    # TensorBoard
    # -------------------------------------------------------------------------
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "train")
    )
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(log_dir, "val")
    )

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print(f"\n[INFO] Starting training for {num_epochs} epochs "
          f"(from epoch {start_epoch})...\n")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # --- Train Phase ---
        train_metrics = {
            "total_loss": 0.0,
            "occ_loss": 0.0,
            "sem_loss": 0.0,
            "occupied_iou": 0.0,
            "free_iou": 0.0,
            "miou": 0.0,
        }
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_dataset):
            step_metrics = train_step(
                model, batch, focal_loss_fn, semantic_loss_fn,
                optimizer, occ_weight, sem_weight, max_grad_norm,
            )

            # Accumulate metrics
            for key in train_metrics:
                train_metrics[key] += float(step_metrics[key])
            num_train_batches += 1
            global_step += 1

            # Log every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = train_metrics["total_loss"] / num_train_batches
                avg_miou = train_metrics["miou"] / num_train_batches
                current_lr = float(lr_schedule(global_step))
                grad_norm_val = float(step_metrics["grad_norm"])

                print(
                    f"  Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx+1}/{steps_per_epoch}] "
                    f"Loss: {avg_loss:.4f} "
                    f"mIoU: {avg_miou:.4f} "
                    f"LR: {current_lr:.2e} "
                    f"GradNorm: {grad_norm_val:.3f}"
                )

                # TensorBoard train logging
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss/total", avg_loss, step=global_step)
                    tf.summary.scalar(
                        "loss/occupancy",
                        train_metrics["occ_loss"] / num_train_batches,
                        step=global_step,
                    )
                    tf.summary.scalar(
                        "loss/semantic",
                        train_metrics["sem_loss"] / num_train_batches,
                        step=global_step,
                    )
                    tf.summary.scalar("metrics/miou", avg_miou, step=global_step)
                    tf.summary.scalar(
                        "metrics/occupied_iou",
                        train_metrics["occupied_iou"] / num_train_batches,
                        step=global_step,
                    )
                    tf.summary.scalar(
                        "metrics/free_iou",
                        train_metrics["free_iou"] / num_train_batches,
                        step=global_step,
                    )
                    tf.summary.scalar("lr", current_lr, step=global_step)
                    tf.summary.scalar("grad_norm", grad_norm_val, step=global_step)

        # Epoch train averages
        if num_train_batches > 0:
            for key in train_metrics:
                train_metrics[key] /= num_train_batches

        epoch_time = time.time() - epoch_start_time
        print(
            f"\n  [Train] Epoch {epoch+1}/{num_epochs} "
            f"({epoch_time:.1f}s) "
            f"Loss: {train_metrics['total_loss']:.4f} "
            f"OccLoss: {train_metrics['occ_loss']:.4f} "
            f"SemLoss: {train_metrics['sem_loss']:.4f} "
            f"mIoU: {train_metrics['miou']:.4f} "
            f"OccIoU: {train_metrics['occupied_iou']:.4f} "
            f"FreeIoU: {train_metrics['free_iou']:.4f}"
        )

        # --- Validation Phase ---
        val_metrics = {
            "total_loss": 0.0,
            "occ_loss": 0.0,
            "sem_loss": 0.0,
            "occupied_iou": 0.0,
            "free_iou": 0.0,
            "miou": 0.0,
        }
        num_val_batches = 0

        for batch in val_dataset:
            step_metrics = val_step(
                model, batch, focal_loss_fn, semantic_loss_fn,
                occ_weight, sem_weight,
            )

            for key in val_metrics:
                val_metrics[key] += float(step_metrics[key])
            num_val_batches += 1

        if num_val_batches > 0:
            for key in val_metrics:
                val_metrics[key] /= num_val_batches

        print(
            f"  [Val]   Epoch {epoch+1}/{num_epochs} "
            f"Loss: {val_metrics['total_loss']:.4f} "
            f"OccLoss: {val_metrics['occ_loss']:.4f} "
            f"SemLoss: {val_metrics['sem_loss']:.4f} "
            f"mIoU: {val_metrics['miou']:.4f} "
            f"OccIoU: {val_metrics['occupied_iou']:.4f} "
            f"FreeIoU: {val_metrics['free_iou']:.4f}\n"
        )

        # TensorBoard val logging
        with val_summary_writer.as_default():
            tf.summary.scalar("loss/total", val_metrics["total_loss"], step=global_step)
            tf.summary.scalar("loss/occupancy", val_metrics["occ_loss"], step=global_step)
            tf.summary.scalar("loss/semantic", val_metrics["sem_loss"], step=global_step)
            tf.summary.scalar("metrics/miou", val_metrics["miou"], step=global_step)
            tf.summary.scalar("metrics/occupied_iou", val_metrics["occupied_iou"], step=global_step)
            tf.summary.scalar("metrics/free_iou", val_metrics["free_iou"], step=global_step)

        # --- Checkpoint Saving ---
        # Save latest
        checkpoint_manager.save()

        # Save training metadata
        meta = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_miou": best_miou,
            "val_miou": float(val_metrics["miou"]),
            "val_loss": float(val_metrics["total_loss"]),
        }
        meta_path = os.path.join(checkpoint_dir, "training_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

        # Save best model based on mIoU
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            best_ckpt_dir = os.path.join(output_dir, "best_checkpoint")
            os.makedirs(best_ckpt_dir, exist_ok=True)

            best_checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
            best_ckpt_manager = tf.train.CheckpointManager(
                best_checkpoint, best_ckpt_dir, max_to_keep=1
            )
            best_ckpt_manager.save()

            # Save best metadata
            best_meta = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_miou": float(best_miou),
                "val_loss": float(val_metrics["total_loss"]),
            }
            best_meta_path = os.path.join(best_ckpt_dir, "training_meta.yaml")
            with open(best_meta_path, "w") as f:
                yaml.dump(best_meta, f, default_flow_style=False)

            print(f"  [BEST] New best mIoU: {best_miou:.4f} at epoch {epoch+1}\n")

    # -------------------------------------------------------------------------
    # Training Complete
    # -------------------------------------------------------------------------
    print("=" * 60)
    print(f"Training complete. Best mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Best checkpoint saved to: {os.path.join(output_dir, 'best_checkpoint')}")
    print(f"TensorBoard logs: {log_dir}")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="TF2 Training Script for Radar Occupancy Models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save checkpoints, logs, and outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  Radar Occupancy Model - TensorFlow 2 Training")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output_dir}")
    print(f"  Resume: {args.resume or 'None (training from scratch)'}")
    print(f"  TF version: {tf.__version__}")
    print(f"  GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    print("=" * 60)

    # Set memory growth for GPUs to avoid OOM
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[WARNING] Could not set memory growth for {gpu}: {e}")

    # Run training
    run_training(
        config=config,
        resume_dir=args.resume,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
