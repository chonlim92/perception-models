# [IMPLEMENTED BY CLAUDE - was missing]
"""
TensorFlow 2 evaluation script for Radar Occupancy models.

Computes occupancy IoU (occupied/free), semantic per-class IoU, and mIoU
over a held-out evaluation dataset. Results can be printed to stdout or
saved to a file.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import yaml

from model import build_model


# =============================================================================
# Metrics
# =============================================================================

def compute_iou(pred_logits, target):
    """
    Compute IoU for occupied and free classes (binary occupancy).

    Args:
        pred_logits: (B, H, W, 1) raw logits for occupancy prediction.
        target: (B, H, W) integer tensor with values 0=free, 1=occupied, 2=unknown.

    Returns:
        Tuple (occupied_iou, free_iou) as float scalars, ignoring unknown cells.
    """
    pred_logits = tf.squeeze(pred_logits, axis=-1)  # (B, H, W)

    # Valid mask: exclude unknown=2
    valid_mask = tf.not_equal(target, 2)

    # Predicted classes (threshold at 0 for logits)
    pred_class = tf.cast(pred_logits > 0.0, tf.int32)

    # Ground truth binary: occupied=1, free=0
    gt_class = tf.cast(tf.equal(target, 1), tf.int32)

    # Apply valid mask (set invalid cells to -1 so they never match)
    pred_valid = tf.where(valid_mask, pred_class, -1 * tf.ones_like(pred_class))
    gt_valid = tf.where(valid_mask, gt_class, -1 * tf.ones_like(gt_class))

    # Occupied IoU
    pred_occ = tf.equal(pred_valid, 1)
    gt_occ = tf.equal(gt_valid, 1)
    intersection_occ = tf.reduce_sum(tf.cast(tf.logical_and(pred_occ, gt_occ), tf.float32))
    union_occ = tf.reduce_sum(tf.cast(tf.logical_or(pred_occ, gt_occ), tf.float32))
    occupied_iou = intersection_occ / tf.maximum(union_occ, 1.0)

    # Free IoU
    pred_free = tf.logical_and(tf.equal(pred_valid, 0), valid_mask)
    gt_free = tf.logical_and(tf.equal(gt_valid, 0), valid_mask)
    intersection_free = tf.reduce_sum(tf.cast(tf.logical_and(pred_free, gt_free), tf.float32))
    union_free = tf.reduce_sum(tf.cast(tf.logical_or(pred_free, gt_free), tf.float32))
    free_iou = intersection_free / tf.maximum(union_free, 1.0)

    return occupied_iou, free_iou


def compute_semantic_iou(pred_logits, target, num_classes, ignore_index=2):
    """
    Compute per-class IoU and mean IoU for semantic segmentation.

    Args:
        pred_logits: (B, H, W, K) raw logits for K semantic classes.
        target: (B, H, W) integer class indices.
        num_classes: Number of semantic classes (K).
        ignore_index: Class index to ignore in evaluation (default=2 for unknown).

    Returns:
        Tuple (per_class_iou, miou):
            - per_class_iou: numpy array of shape (num_classes,) with IoU for each class.
            - miou: scalar mean IoU over classes that have at least one sample.
    """
    # Predicted class: argmax over channel dimension
    pred_class = tf.argmax(pred_logits, axis=-1)  # (B, H, W), dtype int64
    pred_class = tf.cast(pred_class, tf.int32)

    # Valid mask: exclude ignore_index
    valid_mask = tf.not_equal(target, ignore_index)

    # Flatten
    pred_flat = tf.reshape(pred_class, [-1])
    target_flat = tf.reshape(target, [-1])
    valid_flat = tf.reshape(valid_mask, [-1])

    # Keep only valid cells
    valid_indices = tf.where(valid_flat)
    pred_valid = tf.gather(pred_flat, valid_indices[:, 0])
    target_valid = tf.gather(target_flat, valid_indices[:, 0])

    # Build confusion matrix
    confusion = tf.math.confusion_matrix(
        target_valid, pred_valid, num_classes=num_classes, dtype=tf.float64
    )

    # Compute per-class IoU from confusion matrix
    # IoU_c = TP_c / (TP_c + FP_c + FN_c)
    # TP_c = confusion[c, c]
    # FP_c = sum of column c minus TP_c
    # FN_c = sum of row c minus TP_c
    confusion = tf.cast(confusion, tf.float64)
    tp = tf.linalg.diag_part(confusion)
    fp = tf.reduce_sum(confusion, axis=0) - tp  # column sum - diagonal
    fn = tf.reduce_sum(confusion, axis=1) - tp  # row sum - diagonal

    denominator = tp + fp + fn
    per_class_iou = tp / tf.maximum(denominator, 1.0)

    # Convert to numpy
    per_class_iou_np = per_class_iou.numpy().astype(np.float64)

    # Mean IoU: only over classes that appear in ground truth (denominator > 0)
    denominator_np = denominator.numpy()
    valid_classes = denominator_np > 0
    if valid_classes.any():
        miou = float(np.mean(per_class_iou_np[valid_classes]))
    else:
        miou = 0.0

    return per_class_iou_np, miou


# =============================================================================
# Evaluation Loop
# =============================================================================

def evaluate(model, dataset, num_classes):
    """
    Run evaluation over the full dataset.

    Iterates over all batches, accumulates predictions, and computes
    final IoU metrics for both occupancy and semantic segmentation.

    Args:
        model: The trained radar occupancy model.
        dataset: A tf.data.Dataset yielding batches of input dictionaries.
        num_classes: Number of semantic classes.

    Returns:
        Dictionary with keys:
            - occupied_iou: float, IoU for occupied cells
            - free_iou: float, IoU for free cells
            - miou: float, mean of occupied_iou and free_iou
            - semantic_per_class_iou: numpy array of per-class IoU
            - semantic_miou: float, mean IoU over valid semantic classes
    """
    # Accumulators for occupancy confusion (binary: pred x gt, 2x2)
    occ_confusion = np.zeros((2, 2), dtype=np.float64)

    # Accumulator for semantic confusion matrix
    sem_confusion = np.zeros((num_classes, num_classes), dtype=np.float64)

    num_batches = 0

    for batch in dataset:
        pillar_features = batch["pillar_features"]
        pillar_indices = batch["pillar_indices"]
        num_pillars = batch["num_pillars"]
        occupancy_gt = batch["occupancy_gt"]
        semantic_gt = batch["semantic_gt"]

        # Forward pass (inference mode)
        outputs = model(
            {
                "pillar_features": pillar_features,
                "pillar_indices": pillar_indices,
                "num_pillars": num_pillars,
            },
            training=False,
        )

        occ_logits = outputs["occupancy"]  # (B, H, W, 1)
        sem_logits = outputs["semantic"]   # (B, H, W, K)

        # --- Occupancy confusion matrix accumulation ---
        occ_pred = tf.squeeze(occ_logits, axis=-1)  # (B, H, W)
        occ_pred_class = tf.cast(occ_pred > 0.0, tf.int32)  # 0=free, 1=occupied

        valid_mask_occ = tf.not_equal(occupancy_gt, 2)
        gt_binary = tf.cast(tf.equal(occupancy_gt, 1), tf.int32)

        # Flatten valid cells
        valid_flat_occ = tf.reshape(valid_mask_occ, [-1])
        pred_flat_occ = tf.reshape(occ_pred_class, [-1])
        gt_flat_occ = tf.reshape(gt_binary, [-1])

        valid_indices_occ = tf.where(valid_flat_occ)
        pred_valid_occ = tf.gather(pred_flat_occ, valid_indices_occ[:, 0])
        gt_valid_occ = tf.gather(gt_flat_occ, valid_indices_occ[:, 0])

        batch_occ_cm = tf.math.confusion_matrix(
            gt_valid_occ, pred_valid_occ, num_classes=2, dtype=tf.float64
        )
        occ_confusion += batch_occ_cm.numpy()

        # --- Semantic confusion matrix accumulation ---
        sem_pred_class = tf.cast(tf.argmax(sem_logits, axis=-1), tf.int32)  # (B, H, W)
        valid_mask_sem = tf.not_equal(semantic_gt, 2)

        valid_flat_sem = tf.reshape(valid_mask_sem, [-1])
        pred_flat_sem = tf.reshape(sem_pred_class, [-1])
        gt_flat_sem = tf.reshape(semantic_gt, [-1])

        valid_indices_sem = tf.where(valid_flat_sem)
        pred_valid_sem = tf.gather(pred_flat_sem, valid_indices_sem[:, 0])
        gt_valid_sem = tf.gather(gt_flat_sem, valid_indices_sem[:, 0])

        batch_sem_cm = tf.math.confusion_matrix(
            gt_valid_sem, pred_valid_sem, num_classes=num_classes, dtype=tf.float64
        )
        sem_confusion += batch_sem_cm.numpy()

        num_batches += 1

    # --- Compute occupancy IoU from accumulated confusion matrix ---
    # occ_confusion[i, j] = count where gt=i, pred=j
    # Class 0 = free, Class 1 = occupied
    tp_occ = np.diag(occ_confusion)
    fp_occ = np.sum(occ_confusion, axis=0) - tp_occ
    fn_occ = np.sum(occ_confusion, axis=1) - tp_occ

    denom_occ = tp_occ + fp_occ + fn_occ
    iou_occ = np.where(denom_occ > 0, tp_occ / denom_occ, 0.0)

    free_iou = float(iou_occ[0])
    occupied_iou = float(iou_occ[1])
    miou = (occupied_iou + free_iou) / 2.0

    # --- Compute semantic IoU from accumulated confusion matrix ---
    tp_sem = np.diag(sem_confusion)
    fp_sem = np.sum(sem_confusion, axis=0) - tp_sem
    fn_sem = np.sum(sem_confusion, axis=1) - tp_sem

    denom_sem = tp_sem + fp_sem + fn_sem
    semantic_per_class_iou = np.where(denom_sem > 0, tp_sem / denom_sem, 0.0)

    valid_sem_classes = denom_sem > 0
    if valid_sem_classes.any():
        semantic_miou = float(np.mean(semantic_per_class_iou[valid_sem_classes]))
    else:
        semantic_miou = 0.0

    results = {
        "occupied_iou": occupied_iou,
        "free_iou": free_iou,
        "miou": miou,
        "semantic_per_class_iou": semantic_per_class_iou,
        "semantic_miou": semantic_miou,
    }

    print(f"\n[INFO] Evaluation complete. Processed {num_batches} batches.")
    return results


# =============================================================================
# Results Formatting
# =============================================================================

def format_results(results, class_names=None):
    """
    Pretty-print evaluation results as a formatted table.

    Args:
        results: Dictionary returned by evaluate().
        class_names: Optional list of semantic class names. If None, uses
                     generic "Class 0", "Class 1", etc.

    Returns:
        Formatted string of the results table.
    """
    lines = []
    separator = "=" * 60

    lines.append(separator)
    lines.append("  Radar Occupancy Model - Evaluation Results")
    lines.append(separator)

    # Occupancy metrics
    lines.append("")
    lines.append("  Occupancy Metrics (Binary)")
    lines.append("  " + "-" * 40)
    lines.append(f"    Occupied IoU:    {results['occupied_iou']:.4f}")
    lines.append(f"    Free IoU:        {results['free_iou']:.4f}")
    lines.append(f"    Mean IoU:        {results['miou']:.4f}")

    # Semantic metrics
    lines.append("")
    lines.append("  Semantic Segmentation Metrics")
    lines.append("  " + "-" * 40)

    per_class_iou = results["semantic_per_class_iou"]
    num_classes = len(per_class_iou)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Ensure class_names matches num_classes
    while len(class_names) < num_classes:
        class_names.append(f"Class {len(class_names)}")

    # Table header
    lines.append(f"    {'Class':<20} {'IoU':>10}")
    lines.append(f"    {'-----':<20} {'---':>10}")

    for i in range(num_classes):
        iou_val = per_class_iou[i]
        name = class_names[i]
        lines.append(f"    {name:<20} {iou_val:>10.4f}")

    lines.append(f"    {'-----':<20} {'---':>10}")
    lines.append(f"    {'Semantic mIoU':<20} {results['semantic_miou']:>10.4f}")

    lines.append("")
    lines.append(separator)

    output = "\n".join(lines)
    print(output)
    return output


# =============================================================================
# Dataset (reuses structure from train.py)
# =============================================================================

class RadarOccupancyDataset:
    """
    Placeholder radar occupancy dataset for evaluation.

    Each sample is a dictionary with:
        - pillar_features: (max_pillars, max_points_per_pillar, feature_dim)
        - pillar_indices: (max_pillars, 2) grid indices for each pillar
        - num_pillars: scalar, number of valid pillars
        - occupancy_gt: (H, W) ground truth occupancy map (0=free, 1=occupied, 2=unknown)
        - semantic_gt: (H, W) ground truth semantic labels
    """

    def __init__(self, config, data_dir=None, split="val"):
        self.config = config
        self.data_dir = data_dir
        self.split = split
        self.grid_size = config["grid"]["grid_size"]  # [H, W]
        self.max_pillars = config.get("model", {}).get("pillar", {}).get("max_pillars", 10000)
        self.max_points_per_pillar = config.get("model", {}).get("pillar", {}).get("max_points", 32)
        self.feature_dim = config.get("model", {}).get("pillar", {}).get("feature_dim", 7)
        self.num_semantic_classes = config.get("model", {}).get("heads", {}).get("num_semantic_classes", 5)
        self.num_samples = 500

    def __len__(self):
        return self.num_samples

    def generator(self):
        """Generator that yields individual evaluation samples."""
        rng = np.random.default_rng(seed=123)
        H, W = self.grid_size

        for _ in range(self.num_samples):
            num_pillars = rng.integers(100, self.max_pillars)

            pillar_features = rng.standard_normal(
                (self.max_pillars, self.max_points_per_pillar, self.feature_dim)
            ).astype(np.float32)

            pillar_indices = np.zeros((self.max_pillars, 2), dtype=np.int32)
            pillar_indices[:num_pillars, 0] = rng.integers(0, H, size=num_pillars)
            pillar_indices[:num_pillars, 1] = rng.integers(0, W, size=num_pillars)

            # Occupancy ground truth
            occupancy_gt = np.zeros((H, W), dtype=np.int32)
            num_occupied = rng.integers(50, 500)
            occ_rows = rng.integers(0, H, size=num_occupied)
            occ_cols = rng.integers(0, W, size=num_occupied)
            occupancy_gt[occ_rows, occ_cols] = 1
            # Unknown cells
            num_unknown = rng.integers(100, 1000)
            unk_rows = rng.integers(0, H, size=num_unknown)
            unk_cols = rng.integers(0, W, size=num_unknown)
            occupancy_gt[unk_rows, unk_cols] = 2

            # Semantic ground truth
            semantic_gt = rng.integers(
                0, self.num_semantic_classes, size=(H, W)
            ).astype(np.int32)
            semantic_gt[unk_rows, unk_cols] = 2  # same ignore index

            yield {
                "pillar_features": pillar_features,
                "pillar_indices": pillar_indices,
                "num_pillars": np.int32(num_pillars),
                "occupancy_gt": occupancy_gt,
                "semantic_gt": semantic_gt,
            }

    def create_tf_dataset(self, batch_size):
        """Create a tf.data.Dataset for evaluation (no shuffle)."""
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

        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="TF2 Evaluation Script for Radar Occupancy Models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory containing the saved model.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to evaluation data directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path to save evaluation results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  Radar Occupancy Model - TensorFlow 2 Evaluation")
    print("=" * 60)
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data dir:   {args.data_dir or '(placeholder dataset)'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output:     {args.output or '(stdout only)'}")
    print(f"  TF version: {tf.__version__}")
    print(f"  GPUs:       {len(tf.config.list_physical_devices('GPU'))}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # GPU memory growth
    # -------------------------------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[WARNING] Could not set memory growth for {gpu}: {e}")

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    print("\n[INFO] Building model...")
    model = build_model(config)
    model.summary(print_fn=lambda x: print(f"  {x}"))

    # -------------------------------------------------------------------------
    # Restore checkpoint
    # -------------------------------------------------------------------------
    print(f"\n[INFO] Restoring checkpoint from: {args.checkpoint}")
    checkpoint = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(args.checkpoint)

    if latest_ckpt is None:
        print(f"[ERROR] No checkpoint found in {args.checkpoint}")
        print("  Please provide a valid checkpoint directory.")
        return

    status = checkpoint.restore(latest_ckpt)
    status.expect_partial()
    print(f"[INFO] Restored checkpoint: {latest_ckpt}")

    # Load checkpoint metadata if available
    meta_path = os.path.join(args.checkpoint, "training_meta.yaml")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
        print(f"[INFO] Checkpoint metadata: epoch={meta.get('epoch', '?')}, "
              f"best_miou={meta.get('best_miou', '?')}")

    # -------------------------------------------------------------------------
    # Create evaluation dataset
    # -------------------------------------------------------------------------
    print("\n[INFO] Creating evaluation dataset...")
    num_semantic_classes = config.get("model", {}).get("heads", {}).get(
        "num_semantic_classes", 5
    )

    eval_dataset_obj = RadarOccupancyDataset(
        config, data_dir=args.data_dir, split="val"
    )
    eval_dataset = eval_dataset_obj.create_tf_dataset(batch_size=args.batch_size)
    print(f"[INFO] Evaluation samples: {len(eval_dataset_obj)}")

    # -------------------------------------------------------------------------
    # Run evaluation
    # -------------------------------------------------------------------------
    print("\n[INFO] Running evaluation...")
    results = evaluate(model, eval_dataset, num_classes=num_semantic_classes)

    # -------------------------------------------------------------------------
    # Format and display results
    # -------------------------------------------------------------------------
    class_names = config.get("data", {}).get("class_names", None)
    output_text = format_results(results, class_names=class_names)

    # -------------------------------------------------------------------------
    # Save results to file if requested
    # -------------------------------------------------------------------------
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, "w") as f:
            f.write(output_text)
            f.write("\n\n# Raw metrics (YAML format)\n")
            # Write machine-readable metrics
            raw_metrics = {
                "occupied_iou": float(results["occupied_iou"]),
                "free_iou": float(results["free_iou"]),
                "miou": float(results["miou"]),
                "semantic_miou": float(results["semantic_miou"]),
                "semantic_per_class_iou": [
                    float(v) for v in results["semantic_per_class_iou"]
                ],
            }
            yaml.dump(raw_metrics, f, default_flow_style=False)

        print(f"\n[INFO] Results saved to: {args.output}")

    print("\n[INFO] Evaluation finished.")


if __name__ == "__main__":
    main()
