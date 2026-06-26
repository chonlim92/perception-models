"""Cylinder3D TensorFlow 2 evaluation script.

Evaluates a trained Cylinder3D model on the SemanticKITTI validation set:
- Loads model from checkpoint
- Iterates over the validation set
- Computes per-class IoU using a confusion matrix
- Prints a formatted results table
- Optionally saves predictions in SemanticKITTI submission format
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import yaml
import tensorflow as tf

from model import Cylinder3DModel, SEMANTICKITTI_NUM_CLASSES, DEFAULT_GRID_SIZE
from train import (
    load_config,
    load_point_cloud,
    remap_labels,
    VAL_SEQUENCES,
    SEMANTICKITTI_LABEL_MAP,
)


# SemanticKITTI class names for display
SEMANTICKITTI_CLASS_NAMES = [
    "unlabeled",     # 0
    "car",           # 1
    "bicycle",       # 2
    "bus",           # 3
    "motorcycle",    # 4
    "on-rails",     # 5
    "truck",         # 6
    "other-vehicle", # 7
    "person",        # 8
    "bicyclist",     # 9
    "motorcyclist",  # 10
    "road",          # 11
    "parking",       # 12
    "sidewalk",      # 13
    "other-ground",  # 14
    "building",      # 15
    "fence",         # 16
    "lane-marking",  # 17
    "vegetation",    # 18
    "trunk",         # 19
]

# Inverse label map: training ID -> raw label ID (for submission)
INVERSE_LABEL_MAP = {}
for raw_id, train_id in SEMANTICKITTI_LABEL_MAP.items():
    if train_id not in INVERSE_LABEL_MAP:
        INVERSE_LABEL_MAP[train_id] = raw_id


def compute_confusion_matrix(predictions, labels, num_classes, ignore_index=0):
    """Compute confusion matrix from predictions and labels.

    Args:
        predictions: numpy array of predicted class indices
        labels: numpy array of ground truth class indices
        num_classes: int
        ignore_index: class to ignore (e.g., unlabeled)

    Returns:
        confusion_matrix: [num_classes, num_classes] numpy array
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Filter ignored indices
    valid_mask = labels != ignore_index
    pred_valid = predictions[valid_mask]
    label_valid = labels[valid_mask]

    # Clip to valid range
    pred_valid = np.clip(pred_valid, 0, num_classes - 1)
    label_valid = np.clip(label_valid, 0, num_classes - 1)

    # Populate confusion matrix
    for gt, pred in zip(label_valid, pred_valid):
        confusion[gt, pred] += 1

    return confusion


def compute_metrics(confusion_matrix):
    """Compute per-class IoU, precision, recall from confusion matrix.

    Args:
        confusion_matrix: [num_classes, num_classes] numpy array

    Returns:
        dict with per_class_iou, per_class_precision, per_class_recall, mean_iou
    """
    num_classes = confusion_matrix.shape[0]
    tp = np.diag(confusion_matrix).astype(np.float64)
    fp = confusion_matrix.sum(axis=0).astype(np.float64) - tp
    fn = confusion_matrix.sum(axis=1).astype(np.float64) - tp

    # IoU
    denominator = tp + fp + fn
    iou = np.where(denominator > 0, tp / denominator, 0.0)

    # Precision
    prec_denom = tp + fp
    precision = np.where(prec_denom > 0, tp / prec_denom, 0.0)

    # Recall
    rec_denom = tp + fn
    recall = np.where(rec_denom > 0, tp / rec_denom, 0.0)

    # Mean IoU (excluding class 0 = unlabeled)
    valid_classes = denominator[1:] > 0
    mean_iou = np.mean(iou[1:][valid_classes]) if valid_classes.any() else 0.0

    return {
        "per_class_iou": iou,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "mean_iou": mean_iou,
    }


def print_results_table(metrics, class_names):
    """Print a formatted table of evaluation results.

    Args:
        metrics: dict from compute_metrics
        class_names: list of class name strings
    """
    iou = metrics["per_class_iou"]
    precision = metrics["per_class_precision"]
    recall = metrics["per_class_recall"]
    mean_iou = metrics["mean_iou"]

    print("\n" + "=" * 75)
    print("EVALUATION RESULTS - Cylinder3D (TensorFlow 2)")
    print("=" * 75)
    print(f"{'Class':<16} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'Support':>10}")
    print("-" * 75)

    for i in range(1, len(class_names)):
        print(
            f"{class_names[i]:<16} {iou[i]*100:>7.2f}% {precision[i]*100:>9.2f}% "
            f"{recall[i]*100:>7.2f}%"
        )

    print("-" * 75)
    print(f"{'MEAN IoU':<16} {mean_iou*100:>7.2f}%")
    print("=" * 75)


def save_submission(predictions_list, output_dir, sequence, frame_names):
    """Save predictions in SemanticKITTI submission format.

    Saves .label files with remapped labels suitable for the competition server.

    Args:
        predictions_list: list of numpy arrays with per-frame predictions
        output_dir: base output directory
        sequence: sequence string (e.g., "08")
        frame_names: list of frame file names
    """
    pred_dir = os.path.join(output_dir, "sequences", sequence, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for preds, frame_name in zip(predictions_list, frame_names):
        # Map training IDs back to raw label IDs
        raw_preds = np.zeros_like(preds, dtype=np.uint32)
        for train_id, raw_id in INVERSE_LABEL_MAP.items():
            raw_preds[preds == train_id] = raw_id

        # Save as .label file
        label_file = frame_name.replace(".bin", ".label")
        output_path = os.path.join(pred_dir, label_file)
        raw_preds.astype(np.uint32).tofile(output_path)

    print(f"Saved {len(predictions_list)} prediction files to {pred_dir}")


def evaluate(config, args):
    """Run evaluation on the validation set.

    Args:
        config: configuration dict
        args: argparse namespace
    """
    num_classes = config["model"]["num_classes"]
    num_points = config["data"]["num_points"]
    dataset_path = config["data"]["dataset_path"]

    # Build model
    print("Building model...")
    model = Cylinder3DModel(
        num_classes=num_classes,
        grid_size=config["model"]["grid_size"],
        rho_range=config["model"].get("rho_range"),
        theta_range=config["model"].get("theta_range"),
        z_range=config["model"].get("z_range"),
    )

    # Build model with a dummy input to initialize weights
    dummy_input = tf.zeros([1, num_points, 4], dtype=tf.float32)
    _ = model(dummy_input, training=False)

    # Load checkpoint
    if args.checkpoint:
        if args.checkpoint.endswith(".index") or os.path.isdir(args.checkpoint):
            checkpoint_path = args.checkpoint.replace(".index", "")
        else:
            checkpoint_path = args.checkpoint

        # Try loading as weights first
        try:
            model.load_weights(checkpoint_path)
            print(f"Loaded weights from: {checkpoint_path}")
        except (ValueError, tf.errors.NotFoundError):
            # Try as tf.train.Checkpoint
            checkpoint = tf.train.Checkpoint(model=model)
            status = checkpoint.restore(checkpoint_path)
            status.expect_partial()
            print(f"Restored checkpoint: {checkpoint_path}")
    else:
        # Try to find latest checkpoint in default dir
        ckpt_dir = config["checkpoint"]["save_dir"]
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest:
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(latest).expect_partial()
            print(f"Restored latest checkpoint: {latest}")
        else:
            print("WARNING: No checkpoint found! Evaluating with random weights.")

    print(f"Model parameters: {model.count_params():,}")

    # Collect validation frames
    sequences = args.sequences.split(",") if args.sequences else VAL_SEQUENCES
    print(f"Evaluating on sequences: {sequences}")

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_frames = 0
    total_time = 0.0

    all_predictions = []
    all_frame_names = []

    for seq in sequences:
        bin_dir = os.path.join(dataset_path, seq, "velodyne")
        label_dir = os.path.join(dataset_path, seq, "labels")

        if not os.path.isdir(bin_dir):
            print(f"Warning: {bin_dir} not found, skipping")
            continue

        frames = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
        print(f"\nSequence {seq}: {len(frames)} frames")

        seq_predictions = []

        for i, frame in enumerate(frames):
            bin_path = os.path.join(bin_dir, frame)
            label_path = os.path.join(label_dir, frame.replace(".bin", ".label"))

            # Load point cloud
            points_np, labels_np = load_point_cloud(
                tf.constant(bin_path),
                tf.constant(label_path),
                tf.constant(num_points),
            )

            # Inference
            points_tensor = tf.expand_dims(tf.constant(points_np), axis=0)  # [1, N, 4]

            start_time = time.time()
            point_logits, _ = model(points_tensor, training=False)
            predictions = tf.argmax(point_logits, axis=-1, output_type=tf.int32)
            inference_time = time.time() - start_time

            total_time += inference_time

            # Convert to numpy
            preds_np = predictions.numpy().squeeze()  # [N]

            # Update confusion matrix
            confusion_matrix += compute_confusion_matrix(
                preds_np, labels_np, num_classes, ignore_index=0
            )

            total_frames += 1

            if args.save_predictions:
                seq_predictions.append(preds_np)
                all_frame_names.append(frame)

            # Print progress
            if (i + 1) % 100 == 0 or i == len(frames) - 1:
                current_metrics = compute_metrics(confusion_matrix)
                print(
                    f"  [{i+1}/{len(frames)}] "
                    f"mIoU: {current_metrics['mean_iou']*100:.2f}% | "
                    f"Time: {inference_time*1000:.1f}ms"
                )

        if args.save_predictions:
            all_predictions.extend(seq_predictions)

    # Final metrics
    metrics = compute_metrics(confusion_matrix)
    print_results_table(metrics, SEMANTICKITTI_CLASS_NAMES)

    # Timing stats
    avg_time = (total_time / total_frames * 1000) if total_frames > 0 else 0
    fps = total_frames / total_time if total_time > 0 else 0
    print(f"\nTiming: {avg_time:.1f} ms/frame, {fps:.1f} FPS ({total_frames} frames)")

    # Save predictions in submission format
    if args.save_predictions and args.output_dir:
        print(f"\nSaving predictions to {args.output_dir}...")
        for seq in sequences:
            seq_frames = [f for f in all_frame_names]
            seq_preds = all_predictions[: len(seq_frames)]
            save_submission(seq_preds, args.output_dir, seq, seq_frames)

    # Save results to file
    if args.output_dir:
        results_path = os.path.join(args.output_dir, "eval_results.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(results_path, "w") as f:
            f.write("Cylinder3D TensorFlow 2 - Evaluation Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Mean IoU: {metrics['mean_iou']*100:.2f}%\n")
            f.write(f"Frames: {total_frames}\n")
            f.write(f"Avg inference time: {avg_time:.1f} ms\n\n")
            f.write(f"{'Class':<16} {'IoU':>8}\n")
            f.write("-" * 30 + "\n")
            for i in range(1, num_classes):
                f.write(
                    f"{SEMANTICKITTI_CLASS_NAMES[i]:<16} "
                    f"{metrics['per_class_iou'][i]*100:>7.2f}%\n"
                )
        print(f"Results saved to: {results_path}")

    return metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Cylinder3D on SemanticKITTI (TensorFlow 2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint or weights file",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="Comma-separated sequence IDs to evaluate (default: '08')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_output",
        help="Directory to save evaluation results and predictions",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions in SemanticKITTI submission format",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU ID to use for evaluation",
    )
    return parser.parse_args()


def main():
    """Entry point for evaluation."""
    args = parse_args()

    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    if args.gpus and gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        visible = [gpus[i] for i in gpu_ids if i < len(gpus)]
        tf.config.set_visible_devices(visible, "GPU")
        for gpu in visible:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Load config
    config = load_config(args.config)

    print("Cylinder3D TensorFlow 2 Evaluation")
    print("=" * 60)
    print(f"Config: {args.config or 'default'}")
    print(f"Checkpoint: {args.checkpoint or 'latest'}")
    print(f"Grid size: {config['model']['grid_size']}")
    print(f"Num points: {config['data']['num_points']}")
    print("=" * 60)

    evaluate(config, args)


if __name__ == "__main__":
    main()
