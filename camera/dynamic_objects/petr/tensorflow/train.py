"""
Training script for PETR/StreamPETR using TensorFlow 2.

Features:
  - Config loading from YAML
  - Multi-GPU training via tf.distribute.MirroredStrategy
  - Custom training loop with tf.GradientTape
  - Mixed precision support (float16 compute, float32 params)
  - Cosine learning rate schedule with linear warmup
  - Checkpoint management
  - TensorBoard logging
  - tf.data pipeline with augmentations
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import tensorflow as tf

from model import PETR, PETRLoss, build_petr_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train PETR/StreamPETR model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay learning rate schedule with linear warmup."""

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.base_lr * (step / tf.maximum(warmup_steps, 1.0))

        progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + tf.cos(np.pi * progress)
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


def create_dataset(
    data_info_path: str,
    data_root: str,
    batch_size: int,
    image_size: tuple = (900, 1600),
    augment: bool = True,
    temporal: bool = False,
) -> tf.data.Dataset:
    """
    Create tf.data dataset from prepared info files.

    The info file is expected to be a pickle containing a list of dicts with:
      - 'img_paths': list of 6 image paths
      - 'intrinsics': (6, 3, 3) array
      - 'extrinsics': (6, 4, 4) array
      - 'gt_labels': (N,) array of class indices
      - 'gt_bboxes': (N, 10) array of bbox params
      - 'ego_motion': (4, 4) array (if temporal)
      - 'prev_idx': int index to previous frame (if temporal)
    """
    import pickle

    with open(data_info_path, "rb") as f:
        data_infos = pickle.load(f)

    num_cameras = 6
    max_gt = 300

    def generator():
        indices = np.arange(len(data_infos))
        np.random.shuffle(indices)
        for idx in indices:
            info = data_infos[idx]

            images = []
            for cam_path in info["img_paths"]:
                full_path = os.path.join(data_root, cam_path)
                img_raw = tf.io.read_file(full_path)
                img = tf.io.decode_jpeg(img_raw, channels=3)
                img = tf.image.resize(img, image_size)
                img = tf.cast(img, tf.float32) / 255.0
                images.append(img)
            images = tf.stack(images, axis=0)

            intrinsics = tf.constant(info["intrinsics"], dtype=tf.float32)
            extrinsics = tf.constant(info["extrinsics"], dtype=tf.float32)

            gt_labels_raw = info["gt_labels"]
            gt_bboxes_raw = info["gt_bboxes"]
            num_gt = len(gt_labels_raw)

            gt_labels = np.full(max_gt, -1, dtype=np.int32)
            gt_bboxes = np.zeros((max_gt, 10), dtype=np.float32)
            gt_labels[:num_gt] = gt_labels_raw
            gt_bboxes[:num_gt] = gt_bboxes_raw

            sample = {
                "images": images,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "gt_labels": tf.constant(gt_labels, dtype=tf.int32),
                "gt_bboxes": tf.constant(gt_bboxes, dtype=tf.float32),
            }

            if temporal and "ego_motion" in info:
                sample["ego_motion"] = tf.constant(info["ego_motion"], dtype=tf.float32)

            yield sample

    output_signature = {
        "images": tf.TensorSpec(shape=(num_cameras, image_size[0], image_size[1], 3), dtype=tf.float32),
        "intrinsics": tf.TensorSpec(shape=(num_cameras, 3, 3), dtype=tf.float32),
        "extrinsics": tf.TensorSpec(shape=(num_cameras, 4, 4), dtype=tf.float32),
        "gt_labels": tf.TensorSpec(shape=(max_gt,), dtype=tf.int32),
        "gt_bboxes": tf.TensorSpec(shape=(max_gt, 10), dtype=tf.float32),
    }

    if temporal:
        output_signature["ego_motion"] = tf.TensorSpec(shape=(4, 4), dtype=tf.float32)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if augment:
        dataset = dataset.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def augment_sample(sample: dict) -> dict:
    """Apply data augmentation to a single sample."""
    images = sample["images"]

    images = tf.image.random_brightness(images, max_delta=0.2)
    images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
    images = tf.clip_by_value(images, 0.0, 1.0)

    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    images = (images - mean) / std

    sample["images"] = images
    return sample


def train(config: dict, args):
    """Main training function."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.makedirs(args.output_dir, exist_ok=True)

    train_config = config["training"]
    model_config = config["model"]
    data_config = config["data"]

    if train_config.get("mixed_precision", True):
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled: compute=float16, params=float32")

    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    global_batch_size = train_config["batch_size"] * strategy.num_replicas_in_sync
    num_epochs = train_config["num_epochs"]
    base_lr = train_config["learning_rate"]
    weight_decay = train_config.get("weight_decay", 0.01)
    grad_clip_norm = train_config.get("grad_clip_norm", 35.0)
    warmup_epochs = train_config.get("warmup_epochs", 1)

    train_dataset = create_dataset(
        data_info_path=data_config["train_info_path"],
        data_root=data_config["data_root"],
        batch_size=global_batch_size,
        image_size=tuple(data_config.get("image_size", [900, 1600])),
        augment=True,
        temporal=model_config.get("temporal", False),
    )

    steps_per_epoch = data_config.get("train_samples", 28130) // global_batch_size
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    with strategy.scope():
        model = build_petr_model(model_config)
        loss_fn = PETRLoss(
            num_classes=model_config.get("num_classes", 10),
            cls_weight=train_config.get("cls_weight", 2.0),
            bbox_weight=train_config.get("bbox_weight", 5.0),
        )

        lr_schedule = CosineWarmupSchedule(
            base_lr=base_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=train_config.get("min_lr", 1e-6),
        )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        )

        if train_config.get("mixed_precision", True):
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=os.path.join(args.output_dir, "checkpoints"),
            max_to_keep=5,
        )

        if args.resume:
            checkpoint.restore(args.resume)
            print(f"Resumed from checkpoint: {args.resume}")
        elif ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint)
            print(f"Restored from latest checkpoint: {ckpt_manager.latest_checkpoint}")

    log_dir = os.path.join(args.output_dir, "logs")
    summary_writer = tf.summary.create_file_writer(log_dir)

    @tf.function
    def train_step(batch):
        """Single training step within strategy scope."""
        images = batch["images"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        gt_labels = batch["gt_labels"]
        gt_bboxes = batch["gt_bboxes"]
        ego_motion = batch.get("ego_motion", None)

        with tf.GradientTape() as tape:
            outputs = model(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                ego_motion=ego_motion,
                training=True,
            )
            losses = loss_fn(outputs, gt_labels, gt_bboxes)
            total_loss = losses["total_loss"]

            if train_config.get("mixed_precision", True):
                scaled_loss = optimizer.get_scaled_loss(total_loss)

        if train_config.get("mixed_precision", True):
            scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
            gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(total_loss, model.trainable_variables)

        gradients, grad_norm = tf.clip_by_global_norm(gradients, grad_clip_norm)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return losses, grad_norm

    @tf.function
    def distributed_train_step(batch):
        """Distributed training step across replicas."""
        per_replica_losses, per_replica_grad_norm = strategy.run(train_step, args=(batch,))
        reduced_losses = {
            k: strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis=None)
            for k, v in per_replica_losses.items()
        }
        mean_grad_norm = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_grad_norm, axis=None
        )
        return reduced_losses, mean_grad_norm

    print("=" * 60)
    print("Starting training")
    print(f"  Model: {'StreamPETR' if model_config.get('temporal', False) else 'PETR'}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Global batch size: {global_batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Base LR: {base_lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print("=" * 60)

    global_step = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_losses = {"total_loss": 0.0, "cls_loss": 0.0, "bbox_loss": 0.0}
        num_batches = 0

        for batch in dist_dataset:
            losses, grad_norm = distributed_train_step(batch)

            for k in epoch_losses:
                epoch_losses[k] += losses[k].numpy()
            num_batches += 1
            global_step += 1

            if global_step % train_config.get("log_interval", 50) == 0:
                current_lr = lr_schedule(tf.cast(global_step, tf.float32)).numpy()
                print(
                    f"  [Step {global_step}/{total_steps}] "
                    f"loss={losses['total_loss'].numpy():.4f} "
                    f"cls={losses['cls_loss'].numpy():.4f} "
                    f"bbox={losses['bbox_loss'].numpy():.4f} "
                    f"grad_norm={grad_norm.numpy():.2f} "
                    f"lr={current_lr:.2e}"
                )

                with summary_writer.as_default():
                    tf.summary.scalar("train/total_loss", losses["total_loss"], step=global_step)
                    tf.summary.scalar("train/cls_loss", losses["cls_loss"], step=global_step)
                    tf.summary.scalar("train/bbox_loss", losses["bbox_loss"], step=global_step)
                    tf.summary.scalar("train/grad_norm", grad_norm, step=global_step)
                    tf.summary.scalar("train/lr", current_lr, step=global_step)

            if global_step >= total_steps:
                break

        epoch_time = time.time() - epoch_start
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}

        print(
            f"\nEpoch {epoch + 1}/{num_epochs} completed in {epoch_time:.1f}s | "
            f"avg_loss={avg_losses['total_loss']:.4f} "
            f"avg_cls={avg_losses['cls_loss']:.4f} "
            f"avg_bbox={avg_losses['bbox_loss']:.4f}"
        )

        ckpt_path = ckpt_manager.save()
        print(f"  Checkpoint saved: {ckpt_path}")

        with summary_writer.as_default():
            tf.summary.scalar("epoch/total_loss", avg_losses["total_loss"], step=epoch)
            tf.summary.scalar("epoch/cls_loss", avg_losses["cls_loss"], step=epoch)
            tf.summary.scalar("epoch/bbox_loss", avg_losses["bbox_loss"], step=epoch)

    model.save(os.path.join(args.output_dir, "saved_model"), save_format="tf")
    print(f"\nTraining complete. Model saved to {args.output_dir}/saved_model")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, args)
