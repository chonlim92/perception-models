"""
TensorFlow 2 evaluation script for MapTR model.

Evaluates vectorized HD map construction from multi-camera images on nuScenes dataset.
Computes Average Precision (AP) at multiple Chamfer distance thresholds for
ped_crossing, divider, and boundary map element classes.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from .model import MapTRModel

# ==============================================================================
# Evaluation Configuration
# ==============================================================================

MAP_CLASSES = ["ped_crossing", "divider", "boundary"]
NUM_CLASSES = len(MAP_CLASSES)

AP_THRESHOLDS = [0.5, 1.0, 1.5]  # Chamfer distance thresholds in meters

NUM_QUERIES = 50
NUM_POINTS_PER_POLYLINE = 20

BEV_X_RANGE = (-30.0, 30.0)  # meters
BEV_Y_RANGE = (-15.0, 15.0)  # meters
BEV_WIDTH = BEV_X_RANGE[1] - BEV_X_RANGE[0]   # 60m
BEV_HEIGHT = BEV_Y_RANGE[1] - BEV_Y_RANGE[0]  # 30m

NUM_CAMERAS = 6
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 800

DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_BATCH_SIZE = 4


# ==============================================================================
# Chamfer Distance
# ==============================================================================

def chamfer_distance(pred_points: tf.Tensor, gt_points: tf.Tensor) -> tf.Tensor:
    """Compute bidirectional Chamfer distance between two point sets.

    Args:
        pred_points: Predicted points with shape (N, 2) in meters.
        gt_points: Ground truth points with shape (M, 2) in meters.

    Returns:
        Scalar tensor representing the mean bidirectional Chamfer distance in meters.
    """
    # pred_points: (N, 2), gt_points: (M, 2)
    # Expand dimensions for broadcasting: (N, 1, 2) and (1, M, 2)
    pred_expanded = tf.expand_dims(pred_points, axis=1)  # (N, 1, 2)
    gt_expanded = tf.expand_dims(gt_points, axis=0)      # (1, M, 2)

    # Compute pairwise squared distances: (N, M)
    pairwise_distances = tf.reduce_sum(
        tf.square(pred_expanded - gt_expanded), axis=-1
    )

    # For each predicted point, find the nearest ground truth point
    min_dist_pred_to_gt = tf.reduce_min(pairwise_distances, axis=1)  # (N,)

    # For each ground truth point, find the nearest predicted point
    min_dist_gt_to_pred = tf.reduce_min(pairwise_distances, axis=0)  # (M,)

    # Mean of both directions (using sqrt for actual distances)
    chamfer_pred_to_gt = tf.reduce_mean(tf.sqrt(min_dist_pred_to_gt + 1e-8))
    chamfer_gt_to_pred = tf.reduce_mean(tf.sqrt(min_dist_gt_to_pred + 1e-8))

    # Bidirectional Chamfer distance
    chamfer_dist = (chamfer_pred_to_gt + chamfer_gt_to_pred) / 2.0

    return chamfer_dist


# ==============================================================================
# Average Precision Computation
# ==============================================================================

def compute_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_id: int,
    distance_threshold: float,
) -> float:
    """Compute Average Precision for a single class at a given distance threshold.

    Uses Chamfer distance to match predictions to ground truth instances.
    Predictions are sorted by confidence and matched greedily.

    Args:
        predictions: List of prediction dicts, each containing:
            - 'points': np.ndarray of shape (num_points, 2) in meters
            - 'class_id': int
            - 'confidence': float
            - 'sample_idx': int (identifies the sample/frame)
        ground_truths: List of ground truth dicts, each containing:
            - 'points': np.ndarray of shape (num_points, 2) in meters
            - 'class_id': int
            - 'sample_idx': int
        class_id: The class to evaluate.
        distance_threshold: Chamfer distance threshold for a true positive match.

    Returns:
        AP value as a float between 0 and 1.
    """
    # Filter predictions and ground truths for this class
    class_preds = [p for p in predictions if p["class_id"] == class_id]
    class_gts = [g for g in ground_truths if g["class_id"] == class_id]

    if len(class_gts) == 0:
        return 0.0

    # Sort predictions by confidence (descending)
    class_preds = sorted(class_preds, key=lambda x: x["confidence"], reverse=True)

    # Build a mapping from sample_idx to ground truth instances
    gt_by_sample: Dict[int, List[Dict]] = {}
    for gt in class_gts:
        sample_idx = gt["sample_idx"]
        if sample_idx not in gt_by_sample:
            gt_by_sample[sample_idx] = []
        gt_by_sample[sample_idx].append({"points": gt["points"], "matched": False})

    total_gt = len(class_gts)
    tp_list = []
    fp_list = []

    for pred in class_preds:
        sample_idx = pred["sample_idx"]
        pred_points = pred["points"]

        if sample_idx not in gt_by_sample:
            # No ground truth for this sample, false positive
            tp_list.append(0)
            fp_list.append(1)
            continue

        sample_gts = gt_by_sample[sample_idx]

        # Find the best matching (unmatched) ground truth
        best_dist = float("inf")
        best_gt_idx = -1

        for gt_idx, gt_item in enumerate(sample_gts):
            if gt_item["matched"]:
                continue

            # Compute Chamfer distance
            dist = chamfer_distance(
                tf.constant(pred_points, dtype=tf.float32),
                tf.constant(gt_item["points"], dtype=tf.float32),
            ).numpy()

            if dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx

        if best_dist <= distance_threshold and best_gt_idx >= 0:
            # True positive
            sample_gts[best_gt_idx]["matched"] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            # False positive
            tp_list.append(0)
            fp_list.append(1)

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp_list).astype(np.float64)
    fp_cumsum = np.cumsum(fp_list).astype(np.float64)

    recall = tp_cumsum / total_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Append sentinel values for the PR curve
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Make precision monotonically decreasing (from right to left)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute AP as area under the precision-recall curve (11-point interpolation)
    ap = 0.0
    recall_points = np.linspace(0.0, 1.0, 11)
    for r in recall_points:
        # Find precision at recall >= r
        mask = recall >= r
        if mask.any():
            ap += np.max(precision[mask])

    ap /= 11.0
    return float(ap)


# ==============================================================================
# MapTR Evaluator
# ==============================================================================

class MapTREvaluator:
    """Accumulates predictions and ground truth across batches and computes metrics."""

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        """Initialize the evaluator.

        Args:
            confidence_threshold: Minimum confidence to consider a prediction.
        """
        self.confidence_threshold = confidence_threshold
        self.predictions: List[Dict] = []
        self.ground_truths: List[Dict] = []
        self.sample_counter = 0

    def reset(self):
        """Reset accumulated predictions and ground truths."""
        self.predictions = []
        self.ground_truths = []
        self.sample_counter = 0

    def update(
        self,
        pred_points: np.ndarray,
        pred_scores: np.ndarray,
        pred_classes: np.ndarray,
        gt_points: np.ndarray,
        gt_classes: np.ndarray,
        batch_size: int,
    ):
        """Accumulate a batch of predictions and ground truths.

        Args:
            pred_points: Predicted polyline points, shape (B, num_queries, num_points, 2)
                         in normalized [0,1] coordinates.
            pred_scores: Confidence scores, shape (B, num_queries).
            pred_classes: Predicted class indices, shape (B, num_queries).
            gt_points: Ground truth polyline points, shape (B, max_gt, num_points, 2)
                       in normalized [0,1] coordinates.
            gt_classes: Ground truth class indices, shape (B, max_gt).
                        Use -1 for padding.
            batch_size: Number of valid samples in this batch.
        """
        for b in range(batch_size):
            sample_idx = self.sample_counter
            self.sample_counter += 1

            # Process predictions
            for q in range(pred_points.shape[1]):
                confidence = float(pred_scores[b, q])
                if confidence < self.confidence_threshold:
                    continue

                class_id = int(pred_classes[b, q])
                if class_id < 0 or class_id >= NUM_CLASSES:
                    continue

                # Denormalize from [0,1] to real-world meters
                points = pred_points[b, q].copy()  # (num_points, 2)
                points[:, 0] = points[:, 0] * BEV_WIDTH + BEV_X_RANGE[0]
                points[:, 1] = points[:, 1] * BEV_HEIGHT + BEV_Y_RANGE[0]

                self.predictions.append({
                    "points": points,
                    "class_id": class_id,
                    "confidence": confidence,
                    "sample_idx": sample_idx,
                })

            # Process ground truths
            for g in range(gt_points.shape[1]):
                class_id = int(gt_classes[b, g])
                if class_id < 0:
                    # Padding entry
                    continue

                # Denormalize from [0,1] to real-world meters
                points = gt_points[b, g].copy()  # (num_points, 2)
                points[:, 0] = points[:, 0] * BEV_WIDTH + BEV_X_RANGE[0]
                points[:, 1] = points[:, 1] * BEV_HEIGHT + BEV_Y_RANGE[0]

                self.ground_truths.append({
                    "points": points,
                    "class_id": class_id,
                    "sample_idx": sample_idx,
                })

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute per-class AP at each distance threshold.

        Returns:
            Nested dict: {class_name: {f"AP@{threshold}m": ap_value, ...}, ...}
            Also includes a "mean" entry with mAP across classes.
        """
        results: Dict[str, Dict[str, float]] = {}

        for class_id, class_name in enumerate(MAP_CLASSES):
            results[class_name] = {}
            for threshold in AP_THRESHOLDS:
                ap = compute_ap(
                    self.predictions,
                    self.ground_truths,
                    class_id=class_id,
                    distance_threshold=threshold,
                )
                results[class_name][f"AP@{threshold}m"] = ap

        # Compute mean AP across classes at each threshold
        results["mean"] = {}
        for threshold in AP_THRESHOLDS:
            key = f"AP@{threshold}m"
            mean_ap = np.mean([results[c][key] for c in MAP_CLASSES])
            results["mean"][key] = float(mean_ap)

        # Overall mAP (mean across all thresholds and classes)
        all_aps = []
        for class_name in MAP_CLASSES:
            for threshold in AP_THRESHOLDS:
                all_aps.append(results[class_name][f"AP@{threshold}m"])
        results["mean"]["mAP"] = float(np.mean(all_aps))

        return results

    def compute_chamfer_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute mean and median Chamfer distances per class.

        For each class, matches each prediction to its nearest ground truth
        (within the same sample) and reports distance statistics.

        Returns:
            Dict: {class_name: {"mean_chamfer": float, "median_chamfer": float}, ...}
        """
        stats: Dict[str, Dict[str, float]] = {}

        for class_id, class_name in enumerate(MAP_CLASSES):
            class_preds = [p for p in self.predictions if p["class_id"] == class_id]
            class_gts = [g for g in self.ground_truths if g["class_id"] == class_id]

            if len(class_preds) == 0 or len(class_gts) == 0:
                stats[class_name] = {"mean_chamfer": float("inf"), "median_chamfer": float("inf")}
                continue

            # Build GT lookup by sample
            gt_by_sample: Dict[int, List[np.ndarray]] = {}
            for gt in class_gts:
                sample_idx = gt["sample_idx"]
                if sample_idx not in gt_by_sample:
                    gt_by_sample[sample_idx] = []
                gt_by_sample[sample_idx].append(gt["points"])

            distances = []
            for pred in class_preds:
                sample_idx = pred["sample_idx"]
                if sample_idx not in gt_by_sample:
                    continue

                # Find minimum Chamfer distance to any GT in this sample
                min_dist = float("inf")
                for gt_pts in gt_by_sample[sample_idx]:
                    dist = chamfer_distance(
                        tf.constant(pred["points"], dtype=tf.float32),
                        tf.constant(gt_pts, dtype=tf.float32),
                    ).numpy()
                    min_dist = min(min_dist, dist)

                if min_dist < float("inf"):
                    distances.append(min_dist)

            if len(distances) > 0:
                stats[class_name] = {
                    "mean_chamfer": float(np.mean(distances)),
                    "median_chamfer": float(np.median(distances)),
                }
            else:
                stats[class_name] = {"mean_chamfer": float("inf"), "median_chamfer": float("inf")}

        return stats

    def format_results(self) -> str:
        """Format evaluation results as a readable string table.

        Returns:
            Formatted results string with per-class AP and mAP.
        """
        metrics = self.compute_metrics()
        chamfer_stats = self.compute_chamfer_stats()

        lines = []
        lines.append("=" * 72)
        lines.append("MapTR Evaluation Results")
        lines.append("=" * 72)
        lines.append("")

        # AP table header
        header = f"{'Class':<15}"
        for threshold in AP_THRESHOLDS:
            header += f"{'AP@' + str(threshold) + 'm':<12}"
        header += f"{'Mean CD':<12}{'Median CD':<12}"
        lines.append(header)
        lines.append("-" * 72)

        # Per-class rows
        for class_name in MAP_CLASSES:
            row = f"{class_name:<15}"
            for threshold in AP_THRESHOLDS:
                ap_val = metrics[class_name][f"AP@{threshold}m"]
                row += f"{ap_val:<12.4f}"
            if class_name in chamfer_stats:
                row += f"{chamfer_stats[class_name]['mean_chamfer']:<12.4f}"
                row += f"{chamfer_stats[class_name]['median_chamfer']:<12.4f}"
            else:
                row += f"{'N/A':<12}{'N/A':<12}"
            lines.append(row)

        lines.append("-" * 72)

        # Mean row
        mean_row = f"{'Mean':<15}"
        for threshold in AP_THRESHOLDS:
            mean_ap = metrics["mean"][f"AP@{threshold}m"]
            mean_row += f"{mean_ap:<12.4f}"
        lines.append(mean_row)

        lines.append("")
        lines.append(f"Overall mAP: {metrics['mean']['mAP']:.4f}")
        lines.append("")
        lines.append(f"Total predictions: {len(self.predictions)}")
        lines.append(f"Total ground truths: {len(self.ground_truths)}")
        lines.append(f"Total samples evaluated: {self.sample_counter}")
        lines.append("=" * 72)

        return "\n".join(lines)


# ==============================================================================
# Validation Dataset
# ==============================================================================

def create_validation_dataset(
    data_root: str,
    batch_size: int,
) -> tf.data.Dataset:
    """Create validation dataset from nuScenes data.

    Loads multi-camera images and HD map annotations without augmentation.
    The dataset yields batches of (images, gt_points, gt_classes).

    Args:
        data_root: Path to the nuScenes dataset root directory.
        batch_size: Batch size for evaluation.

    Returns:
        tf.data.Dataset yielding (images, gt_points, gt_classes) tuples where:
            - images: (B, num_cameras, H, W, 3) float32 normalized to [0, 1]
            - gt_points: (B, max_gt, num_points, 2) float32 in [0, 1]
            - gt_classes: (B, max_gt) int32, padded with -1
    """
    annotations_path = os.path.join(data_root, "annotations", "val_maptr.json")

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    samples = annotations["samples"]

    def generator():
        for sample in samples:
            # Load multi-camera images
            images = []
            for cam_idx in range(NUM_CAMERAS):
                cam_path = os.path.join(data_root, sample["camera_paths"][cam_idx])
                img = tf.io.read_file(cam_path)
                img = tf.io.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
                img = tf.cast(img, tf.float32) / 255.0
                images.append(img)

            images = tf.stack(images, axis=0)  # (6, 480, 800, 3)

            # Load ground truth polylines
            gt_polylines = sample["polylines"]
            max_gt = len(gt_polylines)

            gt_points_list = []
            gt_classes_list = []

            for polyline in gt_polylines:
                points = np.array(polyline["points"], dtype=np.float32)  # (num_points, 2)
                # Normalize to [0, 1]
                points[:, 0] = (points[:, 0] - BEV_X_RANGE[0]) / BEV_WIDTH
                points[:, 1] = (points[:, 1] - BEV_Y_RANGE[0]) / BEV_HEIGHT
                # Resample to fixed number of points
                if len(points) != NUM_POINTS_PER_POLYLINE:
                    indices = np.linspace(0, len(points) - 1, NUM_POINTS_PER_POLYLINE)
                    x_interp = np.interp(indices, np.arange(len(points)), points[:, 0])
                    y_interp = np.interp(indices, np.arange(len(points)), points[:, 1])
                    points = np.stack([x_interp, y_interp], axis=-1).astype(np.float32)

                gt_points_list.append(points)
                gt_classes_list.append(polyline["class_id"])

            gt_points = np.array(gt_points_list, dtype=np.float32)   # (max_gt, 20, 2)
            gt_classes = np.array(gt_classes_list, dtype=np.int32)   # (max_gt,)

            yield images, gt_points, gt_classes

    # Determine max number of GT instances for padding
    max_gt_instances = max(len(s["polylines"]) for s in samples)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(NUM_CAMERAS, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_POINTS_PER_POLYLINE, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )

    # Pad ground truth to fixed size for batching
    def pad_gt(images, gt_points, gt_classes):
        num_gt = tf.shape(gt_points)[0]
        pad_size = max_gt_instances - num_gt

        gt_points_padded = tf.pad(
            gt_points,
            [[0, pad_size], [0, 0], [0, 0]],
            constant_values=0.0,
        )
        gt_classes_padded = tf.pad(
            gt_classes,
            [[0, pad_size]],
            constant_values=-1,
        )

        gt_points_padded = tf.ensure_shape(
            gt_points_padded, [max_gt_instances, NUM_POINTS_PER_POLYLINE, 2]
        )
        gt_classes_padded = tf.ensure_shape(gt_classes_padded, [max_gt_instances])

        return images, gt_points_padded, gt_classes_padded

    dataset = dataset.map(pad_gt, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ==============================================================================
# Evaluation Main Logic
# ==============================================================================

def evaluate(
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str = "gpu",
) -> Dict:
    """Run full evaluation of a MapTR model on the nuScenes validation set.

    Loads the model from a checkpoint, runs inference on the entire validation set,
    applies confidence filtering, computes AP metrics at multiple Chamfer distance
    thresholds, and saves results.

    Args:
        checkpoint_path: Path to the TensorFlow checkpoint directory or prefix.
        data_root: Path to the nuScenes dataset root.
        output_dir: Directory to save evaluation results JSON.
        batch_size: Batch size for inference.
        confidence_threshold: Minimum confidence score to keep a prediction.
        device: Device to use ('gpu' or 'cpu').

    Returns:
        Dictionary containing all computed metrics.
    """
    # Configure device
    if device.lower() == "cpu":
        tf.config.set_visible_devices([], "GPU")
        print("Running evaluation on CPU.")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Running evaluation on {len(gpus)} GPU(s).")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                print("Falling back to CPU.")
                tf.config.set_visible_devices([], "GPU")
        else:
            print("No GPU available. Running evaluation on CPU.")

    # Create model
    print("Building MapTR model...")
    model = MapTRModel(
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        num_points_per_polyline=NUM_POINTS_PER_POLYLINE,
        bev_x_range=BEV_X_RANGE,
        bev_y_range=BEV_Y_RANGE,
        num_cameras=NUM_CAMERAS,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
    )

    # Build model by calling it with dummy input
    dummy_input = tf.zeros(
        (1, NUM_CAMERAS, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.float32
    )
    _ = model(dummy_input, training=False)

    # Restore checkpoint
    print(f"Restoring checkpoint from: {checkpoint_path}")
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)
                                if os.path.isdir(checkpoint_path)
                                else checkpoint_path)
    status.expect_partial()
    print("Checkpoint restored successfully.")

    # Create validation dataset
    print(f"Loading validation data from: {data_root}")
    val_dataset = create_validation_dataset(data_root, batch_size)

    # Initialize evaluator
    evaluator = MapTREvaluator(confidence_threshold=confidence_threshold)

    # Run inference
    print("Running inference on validation set...")
    total_samples = 0
    start_time = time.time()

    for batch_idx, (images, gt_points, gt_classes) in enumerate(val_dataset):
        current_batch_size = tf.shape(images)[0].numpy()

        # Forward pass
        outputs = model(images, training=False)

        # Extract predictions
        # Expected model output format:
        #   outputs["pred_points"]: (B, num_queries, num_points, 2) in [0, 1]
        #   outputs["pred_logits"]: (B, num_queries, num_classes)
        pred_points = outputs["pred_points"].numpy()
        pred_logits = outputs["pred_logits"].numpy()

        # Convert logits to class predictions and confidence scores
        pred_probs = tf.nn.softmax(
            tf.constant(pred_logits, dtype=tf.float32), axis=-1
        ).numpy()
        pred_classes = np.argmax(pred_probs, axis=-1)        # (B, num_queries)
        pred_scores = np.max(pred_probs, axis=-1)            # (B, num_queries)

        # Accumulate results
        evaluator.update(
            pred_points=pred_points,
            pred_scores=pred_scores,
            pred_classes=pred_classes,
            gt_points=gt_points.numpy(),
            gt_classes=gt_classes.numpy(),
            batch_size=current_batch_size,
        )

        total_samples += current_batch_size
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed
            print(
                f"  Batch {batch_idx + 1}: "
                f"{total_samples} samples processed "
                f"({samples_per_sec:.1f} samples/sec)"
            )

    elapsed_total = time.time() - start_time
    print(
        f"\nInference complete. {total_samples} samples in {elapsed_total:.1f}s "
        f"({total_samples / elapsed_total:.1f} samples/sec)"
    )

    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics = evaluator.compute_metrics()
    chamfer_stats = evaluator.compute_chamfer_stats()

    # Print formatted results
    results_str = evaluator.format_results()
    print("\n" + results_str)

    # Prepare output
    output_metrics = {
        "metrics": metrics,
        "chamfer_stats": chamfer_stats,
        "config": {
            "checkpoint_path": checkpoint_path,
            "data_root": data_root,
            "batch_size": batch_size,
            "confidence_threshold": confidence_threshold,
            "num_samples": total_samples,
            "inference_time_sec": elapsed_total,
            "ap_thresholds_m": AP_THRESHOLDS,
            "map_classes": MAP_CLASSES,
            "bev_range": {
                "x": list(BEV_X_RANGE),
                "y": list(BEV_Y_RANGE),
            },
            "num_queries": NUM_QUERIES,
            "num_points_per_polyline": NUM_POINTS_PER_POLYLINE,
        },
    }

    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(output_metrics, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output_metrics


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MapTR evaluation.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate MapTR model for HD map construction on nuScenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory or file prefix.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the nuScenes dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for evaluation inference.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence score to consider a prediction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device to run evaluation on.",
    )

    return parser.parse_args()


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main entry point for MapTR evaluation."""
    args = parse_args()

    print("=" * 72)
    print("MapTR Evaluation - Vectorized HD Map Construction")
    print("=" * 72)
    print(f"  Checkpoint:   {args.checkpoint_path}")
    print(f"  Data root:    {args.data_root}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Confidence:   {args.confidence_threshold}")
    print(f"  Device:       {args.device}")
    print(f"  Map classes:  {MAP_CLASSES}")
    print(f"  AP thresholds: {AP_THRESHOLDS} meters")
    print(f"  BEV range:    X={BEV_X_RANGE}, Y={BEV_Y_RANGE}")
    print(f"  Queries:      {NUM_QUERIES}")
    print(f"  Points/poly:  {NUM_POINTS_PER_POLYLINE}")
    print("=" * 72)
    print()

    evaluate(
        checkpoint_path=args.checkpoint_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
