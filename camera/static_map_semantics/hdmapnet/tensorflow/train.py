"""
HDMapNet TensorFlow 2 Training Script.

Trains HDMapNet for BEV semantic map prediction from multi-camera surround images.
Supports multi-GPU training via tf.distribute.MirroredStrategy, cosine LR with warmup,
multi-task loss (semantic + instance embedding + direction), TensorBoard logging, and
checkpoint management.

Usage:
    python train.py --data_dir /path/to/npz --output_dir /path/to/output --batch_size 4
"""

import argparse
import glob
import math
import os
import time

import numpy as np
import tensorflow as tf

from model import HDMapNet


# =============================================================================
# Data Pipeline
# =============================================================================

def parse_npz_file(file_path):
    """Load a single .npz file and return tensors."""

    def _load_npz(path_bytes):
        path_str = path_bytes.numpy().decode("utf-8")
        data = np.load(path_str)
        images = data["images"].astype(np.float32)            # [6, 128, 352, 3]
        extrinsics = data["extrinsics"].astype(np.float32)    # [6, 4, 4]
        intrinsics = data["intrinsics"].astype(np.float32)    # [6, 3, 3]
        semantic_masks = data["semantic_masks"].astype(np.float32)  # [200, 200, 3]
        instance_masks = data["instance_masks"].astype(np.int32)   # [200, 200]
        direction_masks = data["direction_masks"].astype(np.float32)  # [200, 200, 2]
        return images, extrinsics, intrinsics, semantic_masks, instance_masks, direction_masks

    images, extrinsics, intrinsics, semantic_masks, instance_masks, direction_masks = tf.py_function(
        _load_npz,
        [file_path],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32],
    )

    images.set_shape([6, 128, 352, 3])
    extrinsics.set_shape([6, 4, 4])
    intrinsics.set_shape([6, 3, 3])
    semantic_masks.set_shape([200, 200, 3])
    instance_masks.set_shape([200, 200])
    direction_masks.set_shape([200, 200, 2])

    return images, extrinsics, intrinsics, semantic_masks, instance_masks, direction_masks


def augment_horizontal_flip(images, extrinsics, intrinsics, semantic_masks, instance_masks, direction_masks):
    """Random horizontal flip applied consistently to images and BEV targets."""

    do_flip = tf.random.uniform([]) > 0.5

    def flip_fn():
        # Flip each camera image along width axis
        flipped_images = tf.reverse(images, axis=[2])  # [6, H, W, 3] flip W

        # Flip BEV targets along the x-axis (axis=1 for width in [200,200,...])
        flipped_semantic = tf.reverse(semantic_masks, axis=[1])
        flipped_instance = tf.reverse(instance_masks, axis=[1])

        # For direction masks, flip along x and negate the x-component
        flipped_direction = tf.reverse(direction_masks, axis=[1])
        # Negate the x component (index 0) of the direction vector
        flip_multiplier = tf.constant([-1.0, 1.0], dtype=tf.float32)
        flipped_direction = flipped_direction * flip_multiplier

        # Adjust intrinsics: cx = W - cx for each camera
        # intrinsics shape: [6, 3, 3]
        flipped_intrinsics = tf.identity(intrinsics)
        # cx is at [i, 0, 2]; image width is 352
        cx_update = 352.0 - intrinsics[:, 0, 2]  # [6]
        # Build updated intrinsics
        row0 = tf.stack([intrinsics[:, 0, 0], intrinsics[:, 0, 1], cx_update], axis=1)  # [6, 3]
        row1 = intrinsics[:, 1, :]  # [6, 3]
        row2 = intrinsics[:, 2, :]  # [6, 3]
        flipped_intrinsics = tf.stack([row0, row1, row2], axis=1)  # [6, 3, 3]

        return flipped_images, extrinsics, flipped_intrinsics, flipped_semantic, flipped_instance, flipped_direction

    def no_flip_fn():
        return images, extrinsics, intrinsics, semantic_masks, instance_masks, direction_masks

    return tf.cond(do_flip, flip_fn, no_flip_fn)


def build_dataset(data_dir, batch_size, is_training=True):
    """Build a tf.data.Dataset pipeline from .npz files."""

    file_pattern = os.path.join(data_dir, "*.npz")
    file_list = sorted(glob.glob(file_pattern))

    if not file_list:
        raise ValueError(f"No .npz files found in {data_dir}")

    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(file_list), reshuffle_each_iteration=True)

    dataset = dataset.map(parse_npz_file, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(
            lambda imgs, ext, intr, sem, inst, dir_m: augment_horizontal_flip(
                imgs, ext, intr, sem, inst, dir_m
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =============================================================================
# Loss Functions
# =============================================================================

def compute_semantic_loss(pred_semantic, gt_semantic, class_weights=None):
    """
    Binary cross-entropy loss for semantic segmentation with class weighting.

    Args:
        pred_semantic: [B, 200, 200, 3] logits (pre-sigmoid)
        gt_semantic: [B, 200, 200, 3] binary targets
        class_weights: [3] per-class weights for imbalance handling

    Returns:
        Scalar loss
    """
    if class_weights is None:
        # Default weights: roads are common, dividers/crossings are rare
        class_weights = tf.constant([1.0, 2.0, 2.0], dtype=tf.float32)

    # Compute binary cross entropy per pixel per class
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_semantic, logits=pred_semantic)
    # bce shape: [B, 200, 200, 3]

    # Apply class weights: broadcast [3] over [B, 200, 200, 3]
    weighted_bce = bce * tf.reshape(class_weights, [1, 1, 1, 3])

    return tf.reduce_mean(weighted_bce)


def compute_instance_loss(pred_embedding, gt_instance, gt_semantic, delta_v=0.5, delta_d=3.0):
    """
    Discriminative loss for instance embedding (push-pull loss).

    For each sample in the batch:
      - Pull loss: embeddings of the same instance are pulled toward their mean
      - Push loss: means of different instances are pushed apart

    Args:
        pred_embedding: [B, 200, 200, E] instance embedding predictions
        gt_instance: [B, 200, 200] integer instance IDs (0 = background)
        gt_semantic: [B, 200, 200, 3] semantic masks (used to identify valid pixels)
        delta_v: margin for pull loss
        delta_d: margin for push loss

    Returns:
        Scalar loss
    """
    batch_size = tf.shape(pred_embedding)[0]
    embedding_dim = tf.shape(pred_embedding)[3]

    total_pull_loss = 0.0
    total_push_loss = 0.0
    valid_samples = 0.0

    for b in tf.range(batch_size):
        embedding = pred_embedding[b]  # [200, 200, E]
        instances = gt_instance[b]     # [200, 200]

        # Valid mask: any semantic class is active
        valid_mask = tf.reduce_any(tf.cast(gt_semantic[b], tf.bool), axis=-1)  # [200, 200]
        # Only consider instances on valid pixels
        masked_instances = tf.where(valid_mask, instances, tf.zeros_like(instances))

        # Get unique instance IDs (excluding 0 = background)
        unique_ids, _ = tf.unique(tf.reshape(masked_instances, [-1]))
        unique_ids = tf.boolean_mask(unique_ids, unique_ids > 0)

        num_instances = tf.shape(unique_ids)[0]

        if num_instances <= 0:
            continue

        # Compute pull loss
        means = []
        pull_loss = 0.0

        for i in tf.range(num_instances):
            inst_id = unique_ids[i]
            mask = tf.equal(masked_instances, inst_id)  # [200, 200]
            mask_float = tf.cast(mask, tf.float32)
            num_pixels = tf.reduce_sum(mask_float)

            if num_pixels < 1.0:
                continue

            # Get embeddings for this instance
            mask_3d = tf.expand_dims(mask_float, -1)  # [200, 200, 1]
            inst_embeddings = embedding * mask_3d  # [200, 200, E]
            mean_embedding = tf.reduce_sum(inst_embeddings, axis=[0, 1]) / num_pixels  # [E]
            means.append(mean_embedding)

            # Pull: distance of each pixel to its mean
            diff = embedding - tf.reshape(mean_embedding, [1, 1, embedding_dim])  # [200, 200, E]
            dist = tf.norm(diff, axis=-1)  # [200, 200]
            hinged = tf.maximum(dist - delta_v, 0.0)
            pull_loss += tf.reduce_sum(hinged * mask_float) / num_pixels

        num_instances_f = tf.cast(num_instances, tf.float32)
        pull_loss = pull_loss / tf.maximum(num_instances_f, 1.0)

        # Compute push loss
        push_loss = 0.0
        if len(means) > 1:
            means_tensor = tf.stack(means, axis=0)  # [N, E]
            n = tf.shape(means_tensor)[0]
            count = 0.0
            for i in tf.range(n):
                for j in tf.range(i + 1, n):
                    dist = tf.norm(means_tensor[i] - means_tensor[j])
                    hinged = tf.maximum(delta_d - dist, 0.0)
                    push_loss += hinged
                    count += 1.0
            push_loss = push_loss / tf.maximum(count, 1.0)

        total_pull_loss += pull_loss
        total_push_loss += push_loss
        valid_samples += 1.0

    total_pull_loss = total_pull_loss / tf.maximum(valid_samples, 1.0)
    total_push_loss = total_push_loss / tf.maximum(valid_samples, 1.0)

    return total_pull_loss + total_push_loss


def compute_direction_loss(pred_direction, gt_direction, gt_semantic):
    """
    L1 loss for direction prediction, only on valid pixels where semantic mask > 0.

    Args:
        pred_direction: [B, 200, 200, 2] predicted direction vectors
        gt_direction: [B, 200, 200, 2] ground truth direction vectors
        gt_semantic: [B, 200, 200, 3] semantic masks

    Returns:
        Scalar loss
    """
    # Valid mask: any semantic class has value > 0
    valid_mask = tf.reduce_any(gt_semantic > 0.0, axis=-1)  # [B, 200, 200]
    valid_mask_float = tf.cast(valid_mask, tf.float32)  # [B, 200, 200]

    # Compute L1 difference
    l1_diff = tf.abs(pred_direction - gt_direction)  # [B, 200, 200, 2]
    l1_per_pixel = tf.reduce_sum(l1_diff, axis=-1)   # [B, 200, 200]

    # Apply mask
    masked_l1 = l1_per_pixel * valid_mask_float

    num_valid = tf.reduce_sum(valid_mask_float)
    loss = tf.reduce_sum(masked_l1) / tf.maximum(num_valid, 1.0)

    return loss


# =============================================================================
# Learning Rate Schedule
# =============================================================================

class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay learning rate schedule with linear warmup."""

    def __init__(self, base_lr, total_steps, warmup_steps=500):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.base_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay after warmup
        progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_lr = self.base_lr * 0.5 * (1.0 + tf.cos(math.pi * progress))

        # Select based on step
        lr = tf.where(step < warmup_steps, warmup_lr, cosine_lr)
        return lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


# =============================================================================
# TensorBoard Logging Utilities
# =============================================================================

def log_predictions(writer, step, pred_semantic, gt_semantic, max_images=2):
    """Log sample prediction visualizations to TensorBoard."""
    with writer.as_default():
        # Sigmoid to get probabilities
        pred_prob = tf.sigmoid(pred_semantic[:max_images])  # [N, 200, 200, 3]
        gt_vis = gt_semantic[:max_images]  # [N, 200, 200, 3]

        # Stack prediction and ground truth side by side for comparison
        # Predictions thresholded at 0.5
        pred_binary = tf.cast(pred_prob > 0.5, tf.float32)

        # Create comparison image: top=GT, bottom=pred
        comparison = tf.concat([gt_vis, pred_binary], axis=1)  # [N, 400, 200, 3]

        tf.summary.image("semantic_predictions_vs_gt", comparison, step=step, max_outputs=max_images)


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    """Main training function."""

    # -------------------------------------------------------------------------
    # Strategy setup
    # -------------------------------------------------------------------------
    if args.num_gpus > 1:
        devices = [f"/gpu:{i}" for i in range(args.num_gpus)]
        strategy = tf.distribute.MirroredStrategy(devices=devices)
    elif args.num_gpus == 1:
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    else:
        strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])

    print(f"Number of replicas: {strategy.num_replicas_in_sync}")

    global_batch_size = args.batch_size * strategy.num_replicas_in_sync

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    train_data_dir = os.path.join(args.data_dir, "train") if os.path.isdir(
        os.path.join(args.data_dir, "train")
    ) else args.data_dir

    val_data_dir = os.path.join(args.data_dir, "val") if os.path.isdir(
        os.path.join(args.data_dir, "val")
    ) else None

    train_dataset = build_dataset(train_data_dir, global_batch_size, is_training=True)

    # Count files for step calculation
    train_files = glob.glob(os.path.join(train_data_dir, "*.npz"))
    steps_per_epoch = len(train_files) // global_batch_size
    total_steps = steps_per_epoch * args.epochs

    print(f"Training files: {len(train_files)}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")

    if val_data_dir:
        val_dataset = build_dataset(val_data_dir, global_batch_size, is_training=False)
    else:
        val_dataset = None

    # Distribute datasets
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    if val_dataset is not None:
        val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # -------------------------------------------------------------------------
    # Model, Optimizer, Checkpoint
    # -------------------------------------------------------------------------
    with strategy.scope():
        model = HDMapNet(view_transform=args.view_transform)

        lr_schedule = CosineDecayWithWarmup(
            base_lr=args.lr,
            total_steps=total_steps,
            warmup_steps=500,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Checkpoint
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=5
        )

        # Restore if available
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"Restored from {checkpoint_manager.latest_checkpoint}")
        else:
            print("Starting training from scratch.")

    # -------------------------------------------------------------------------
    # TensorBoard
    # -------------------------------------------------------------------------
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # -------------------------------------------------------------------------
    # Loss weights
    # -------------------------------------------------------------------------
    semantic_weight = 1.0
    instance_weight = 1.0
    direction_weight = 1.0

    # Class weights for semantic loss (handle class imbalance)
    semantic_class_weights = tf.constant([1.0, 5.0, 5.0], dtype=tf.float32)

    # -------------------------------------------------------------------------
    # Training Step
    # -------------------------------------------------------------------------
    @tf.function
    def train_step(inputs):
        images, extrinsics, intrinsics, semantic_masks, instance_masks, direction_masks = inputs

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(
                images, extrinsics, intrinsics, training=True
            )
            # predictions is a dict with keys: 'semantic', 'instance', 'direction'
            pred_semantic = predictions["semantic"]      # [B, 200, 200, 3]
            pred_instance = predictions["instance"]      # [B, 200, 200, E]
            pred_direction = predictions["direction"]    # [B, 200, 200, 2]

            # Compute losses
            sem_loss = compute_semantic_loss(
                pred_semantic, semantic_masks, class_weights=semantic_class_weights
            )
            inst_loss = compute_instance_loss(
                pred_instance, instance_masks, semantic_masks, delta_v=0.5, delta_d=3.0
            )
            dir_loss = compute_direction_loss(
                pred_direction, direction_masks, semantic_masks
            )

            # Total loss
            total_loss = (
                semantic_weight * sem_loss
                + instance_weight * inst_loss
                + direction_weight * dir_loss
            )

            # Scale loss for distributed training
            scaled_loss = total_loss / tf.cast(strategy.num_replicas_in_sync, tf.float32)

        # Compute and apply gradients
        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        # Clip gradients to prevent explosion
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 35.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return total_loss, sem_loss, inst_loss, dir_loss, pred_semantic

    @tf.function
    def distributed_train_step(inputs):
        per_replica_results = strategy.run(train_step, args=(inputs,))
        total_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[0], axis=None)
        sem_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[1], axis=None)
        inst_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[2], axis=None)
        dir_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[3], axis=None)
        pred_semantic = per_replica_results[4]
        return total_loss, sem_loss, inst_loss, dir_loss, pred_semantic

    # -------------------------------------------------------------------------
    # Validation Step
    # -------------------------------------------------------------------------
    @tf.function
    def val_step(inputs):
        images, extrinsics, intrinsics, semantic_masks, instance_masks, direction_masks = inputs

        predictions = model(images, extrinsics, intrinsics, training=False)
        pred_semantic = predictions["semantic"]
        pred_instance = predictions["instance"]
        pred_direction = predictions["direction"]

        sem_loss = compute_semantic_loss(
            pred_semantic, semantic_masks, class_weights=semantic_class_weights
        )
        inst_loss = compute_instance_loss(
            pred_instance, instance_masks, semantic_masks, delta_v=0.5, delta_d=3.0
        )
        dir_loss = compute_direction_loss(
            pred_direction, direction_masks, semantic_masks
        )

        total_loss = (
            semantic_weight * sem_loss
            + instance_weight * inst_loss
            + direction_weight * dir_loss
        )
        return total_loss, sem_loss, inst_loss, dir_loss

    @tf.function
    def distributed_val_step(inputs):
        per_replica_results = strategy.run(val_step, args=(inputs,))
        total_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[0], axis=None)
        sem_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[1], axis=None)
        inst_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[2], axis=None)
        dir_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[3], axis=None)
        return total_loss, sem_loss, inst_loss, dir_loss

    # -------------------------------------------------------------------------
    # Main Training Loop
    # -------------------------------------------------------------------------
    global_step = optimizer.iterations.numpy()
    log_interval = 50
    vis_interval = 500

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Global batch size: {global_batch_size}")
    print(f"Base learning rate: {args.lr}")
    print(f"View transform: {args.view_transform}")
    print("=" * 70)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_sem_loss = 0.0
        epoch_inst_loss = 0.0
        epoch_dir_loss = 0.0
        num_batches = 0

        for batch_data in train_dist_dataset:
            total_loss, sem_loss, inst_loss, dir_loss, pred_semantic = distributed_train_step(
                batch_data
            )

            global_step = optimizer.iterations.numpy()
            epoch_loss += total_loss.numpy()
            epoch_sem_loss += sem_loss.numpy()
            epoch_inst_loss += inst_loss.numpy()
            epoch_dir_loss += dir_loss.numpy()
            num_batches += 1

            # Logging
            if global_step % log_interval == 0:
                current_lr = lr_schedule(tf.cast(global_step, tf.float32)).numpy()
                with summary_writer.as_default():
                    tf.summary.scalar("train/total_loss", total_loss, step=global_step)
                    tf.summary.scalar("train/semantic_loss", sem_loss, step=global_step)
                    tf.summary.scalar("train/instance_loss", inst_loss, step=global_step)
                    tf.summary.scalar("train/direction_loss", dir_loss, step=global_step)
                    tf.summary.scalar("train/learning_rate", current_lr, step=global_step)

                print(
                    f"  [Step {global_step}] "
                    f"loss={total_loss.numpy():.4f} "
                    f"sem={sem_loss.numpy():.4f} "
                    f"inst={inst_loss.numpy():.4f} "
                    f"dir={dir_loss.numpy():.4f} "
                    f"lr={current_lr:.6f}"
                )

            # Visualization logging
            if global_step % vis_interval == 0:
                # Get the first replica's predictions for visualization
                if isinstance(pred_semantic, tf.distribute.DistributedValues):
                    vis_pred = pred_semantic.values[0]
                else:
                    vis_pred = pred_semantic

                # Get corresponding ground truth
                if isinstance(batch_data[3], tf.distribute.DistributedValues):
                    vis_gt = batch_data[3].values[0]
                else:
                    vis_gt = batch_data[3]

                log_predictions(summary_writer, global_step, vis_pred, vis_gt)

        # End of epoch stats
        epoch_duration = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_sem = epoch_sem_loss / max(num_batches, 1)
        avg_inst = epoch_inst_loss / max(num_batches, 1)
        avg_dir = epoch_dir_loss / max(num_batches, 1)

        print(f"\nEpoch {epoch + 1}/{args.epochs} completed in {epoch_duration:.1f}s")
        print(
            f"  Avg loss: {avg_loss:.4f} | sem: {avg_sem:.4f} | "
            f"inst: {avg_inst:.4f} | dir: {avg_dir:.4f}"
        )

        # TensorBoard epoch summaries
        with summary_writer.as_default():
            tf.summary.scalar("epoch/total_loss", avg_loss, step=epoch)
            tf.summary.scalar("epoch/semantic_loss", avg_sem, step=epoch)
            tf.summary.scalar("epoch/instance_loss", avg_inst, step=epoch)
            tf.summary.scalar("epoch/direction_loss", avg_dir, step=epoch)

        # Validation
        if val_dataset is not None:
            val_total = 0.0
            val_sem = 0.0
            val_inst = 0.0
            val_dir = 0.0
            val_batches = 0

            for val_batch in val_dist_dataset:
                vt, vs, vi, vd = distributed_val_step(val_batch)
                val_total += vt.numpy()
                val_sem += vs.numpy()
                val_inst += vi.numpy()
                val_dir += vd.numpy()
                val_batches += 1

            if val_batches > 0:
                val_avg = val_total / val_batches
                val_sem_avg = val_sem / val_batches
                val_inst_avg = val_inst / val_batches
                val_dir_avg = val_dir / val_batches

                print(
                    f"  Val loss: {val_avg:.4f} | sem: {val_sem_avg:.4f} | "
                    f"inst: {val_inst_avg:.4f} | dir: {val_dir_avg:.4f}"
                )

                with summary_writer.as_default():
                    tf.summary.scalar("val/total_loss", val_avg, step=epoch)
                    tf.summary.scalar("val/semantic_loss", val_sem_avg, step=epoch)
                    tf.summary.scalar("val/instance_loss", val_inst_avg, step=epoch)
                    tf.summary.scalar("val/direction_loss", val_dir_avg, step=epoch)

        # Save checkpoint
        save_path = checkpoint_manager.save()
        print(f"  Checkpoint saved: {save_path}")
        print("=" * 70)

    # Final save
    final_model_dir = os.path.join(args.output_dir, "saved_model")
    os.makedirs(final_model_dir, exist_ok=True)
    checkpoint_manager.save()
    print(f"\nTraining complete. Final checkpoint saved to {checkpoint_dir}")
    print(f"TensorBoard logs: {log_dir}")


# =============================================================================
# Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HDMapNet for BEV semantic map prediction."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to preprocessed .npz data directory. "
             "If it contains train/ and val/ subdirectories, they will be used automatically.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path for checkpoints, logs, and saved model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-replica batch size (default: 4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Base learning rate (default: 2e-4).",
    )
    parser.add_argument(
        "--view_transform",
        type=str,
        default="lss",
        choices=["ipm", "lss"],
        help="View transformation method: 'ipm' or 'lss' (default: 'lss').",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training (default: 1). Set 0 for CPU.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print configuration
    print("HDMapNet Training Configuration")
    print("=" * 70)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 70)

    train(args)
