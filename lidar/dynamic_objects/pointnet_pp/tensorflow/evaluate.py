#!/usr/bin/env python3
"""
PointNet++ Evaluation Script for LiDAR Dynamic Object Perception.

Supports three tasks:
  - classification: per-class precision/recall/F1, overall accuracy
  - detection: 3D Average Precision at IoU 0.5 and 0.7 (oriented bounding boxes)
  - segmentation: mean IoU, per-class IoU, overall accuracy

Usage:
    python evaluate.py --task detection --model_path ./checkpoints/best \
                       --data_dir ./data/test --batch_size 16 --num_points 4096 \
                       --num_classes 4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a PointNet++ model on classification, detection, or segmentation."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "detection", "segmentation"],
        help="Evaluation task: classification, detection, or segmentation.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model directory (SavedModel) or checkpoint prefix.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to test data directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=4096,
        help="Number of points per sample.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,
        help="Number of object classes.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results JSON.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load a PointNet++ model from either SavedModel format or a checkpoint.

    Tries SavedModel first; falls back to checkpoint-based loading if the
    SavedModel signature is not found.
    """
    model_path = str(Path(model_path).resolve())

    # Attempt 1: TF SavedModel directory
    saved_model_pb = os.path.join(model_path, "saved_model.pb")
    if os.path.isdir(model_path) and os.path.isfile(saved_model_pb):
        print(f"[INFO] Loading SavedModel from: {model_path}")
        model = tf.saved_model.load(model_path)
        return model, "saved_model"

    # Attempt 2: Keras .keras / .h5 file
    if os.path.isfile(model_path) and (
        model_path.endswith(".keras") or model_path.endswith(".h5")
    ):
        print(f"[INFO] Loading Keras model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        return model, "keras"

    # Attempt 3: Checkpoint (prefix-based)
    checkpoint_index = model_path + ".index"
    if os.path.isfile(checkpoint_index) or os.path.isfile(model_path + ".ckpt.index"):
        print(f"[INFO] Loading model weights from checkpoint: {model_path}")
        # For checkpoint loading we need the model architecture to be rebuilt.
        # We look for a keras model definition saved alongside the checkpoint.
        model_json_path = os.path.join(os.path.dirname(model_path), "model_config.json")
        if os.path.isfile(model_json_path):
            with open(model_json_path, "r") as f:
                model_config = json.load(f)
            model = tf.keras.models.model_from_json(json.dumps(model_config))
            checkpoint = tf.train.Checkpoint(model=model)
            status = checkpoint.restore(model_path)
            status.expect_partial()
            return model, "checkpoint"
        else:
            # Fallback: try tf.train.Checkpoint with generic restore
            checkpoint = tf.train.Checkpoint()
            status = checkpoint.restore(model_path)
            status.expect_partial()
            print(
                "[WARN] Loaded checkpoint without model architecture. "
                "Ensure model_config.json exists next to the checkpoint."
            )
            return checkpoint, "checkpoint_raw"

    raise FileNotFoundError(
        f"Could not find a valid model at '{model_path}'. "
        "Provide a SavedModel directory, a .keras/.h5 file, or a checkpoint prefix."
    )


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------

def load_classification_data(data_dir: str, num_points: int, batch_size: int):
    """Load point cloud classification data.

    Expected directory structure:
        data_dir/
            points/     - .npy files of shape (N, 3+) per sample
            labels/     - .npy files of shape () or (1,) per sample (class index)

    Returns a tf.data.Dataset yielding (points, labels) batches.
    """
    points_dir = os.path.join(data_dir, "points")
    labels_dir = os.path.join(data_dir, "labels")

    point_files = sorted(
        [f for f in os.listdir(points_dir) if f.endswith(".npy")]
    )
    label_files = sorted(
        [f for f in os.listdir(labels_dir) if f.endswith(".npy")]
    )

    assert len(point_files) == len(label_files), (
        f"Mismatch: {len(point_files)} point files vs {len(label_files)} label files."
    )

    all_points = []
    all_labels = []

    for pf, lf in zip(point_files, label_files):
        pts = np.load(os.path.join(points_dir, pf)).astype(np.float32)
        # Subsample or pad to num_points
        pts = _normalize_point_count(pts, num_points)
        lbl = int(np.load(os.path.join(labels_dir, lf)).flatten()[0])
        all_points.append(pts)
        all_labels.append(lbl)

    all_points = np.array(all_points, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((all_points, all_labels))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, len(point_files)


def load_detection_data(data_dir: str, num_points: int, batch_size: int):
    """Load point cloud detection data.

    Expected directory structure:
        data_dir/
            points/         - .npy files of shape (N, 3+)
            annotations/    - .npy files of shape (M, 8) per sample
                              Each row: [x, y, z, dx, dy, dz, yaw, class_id]
                              (center_x, center_y, center_z, length, width, height, yaw, class)

    Returns lists of (points_batch, annotations_list) where annotations_list
    is a list of arrays (variable number of boxes per sample).
    """
    points_dir = os.path.join(data_dir, "points")
    annot_dir = os.path.join(data_dir, "annotations")

    point_files = sorted(
        [f for f in os.listdir(points_dir) if f.endswith(".npy")]
    )
    annot_files = sorted(
        [f for f in os.listdir(annot_dir) if f.endswith(".npy")]
    )

    assert len(point_files) == len(annot_files), (
        f"Mismatch: {len(point_files)} point files vs {len(annot_files)} annotation files."
    )

    all_points = []
    all_annotations = []

    for pf, af in zip(point_files, annot_files):
        pts = np.load(os.path.join(points_dir, pf)).astype(np.float32)
        pts = _normalize_point_count(pts, num_points)
        annot = np.load(os.path.join(annot_dir, af)).astype(np.float32)
        all_points.append(pts)
        all_annotations.append(annot)

    return all_points, all_annotations, len(point_files)


def load_segmentation_data(data_dir: str, num_points: int, batch_size: int):
    """Load point cloud segmentation data.

    Expected directory structure:
        data_dir/
            points/     - .npy files of shape (N, 3+)
            labels/     - .npy files of shape (N,) per-point class indices

    Returns a tf.data.Dataset yielding (points, labels) batches.
    """
    points_dir = os.path.join(data_dir, "points")
    labels_dir = os.path.join(data_dir, "labels")

    point_files = sorted(
        [f for f in os.listdir(points_dir) if f.endswith(".npy")]
    )
    label_files = sorted(
        [f for f in os.listdir(labels_dir) if f.endswith(".npy")]
    )

    assert len(point_files) == len(label_files), (
        f"Mismatch: {len(point_files)} point files vs {len(label_files)} label files."
    )

    all_points = []
    all_labels = []

    for pf, lf in zip(point_files, label_files):
        pts = np.load(os.path.join(points_dir, pf)).astype(np.float32)
        lbl = np.load(os.path.join(labels_dir, lf)).astype(np.int32).flatten()
        pts, lbl = _normalize_point_count_with_labels(pts, lbl, num_points)
        all_points.append(pts)
        all_labels.append(lbl)

    all_points = np.array(all_points, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((all_points, all_labels))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, len(point_files)


def _normalize_point_count(points: np.ndarray, num_points: int) -> np.ndarray:
    """Subsample or pad point cloud to exactly num_points rows."""
    n = points.shape[0]
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
        return points[indices]
    else:
        pad_indices = np.random.choice(n, num_points - n, replace=True)
        return np.concatenate([points, points[pad_indices]], axis=0)


def _normalize_point_count_with_labels(
    points: np.ndarray, labels: np.ndarray, num_points: int
):
    """Subsample or pad both points and per-point labels."""
    n = points.shape[0]
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
        return points[indices], labels[indices]
    else:
        pad_indices = np.random.choice(n, num_points - n, replace=True)
        pts = np.concatenate([points, points[pad_indices]], axis=0)
        lbl = np.concatenate([labels, labels[pad_indices]], axis=0)
        return pts, lbl


# ---------------------------------------------------------------------------
# Inference Helpers
# ---------------------------------------------------------------------------

def run_inference(model, model_type: str, inputs: tf.Tensor):
    """Run forward pass through the model, handling different model formats."""
    if model_type == "saved_model":
        # Try the default serving signature
        if hasattr(model, "signatures"):
            serve_fn = model.signatures.get("serving_default", None)
            if serve_fn is not None:
                output = serve_fn(inputs)
                # Return the first output tensor
                key = list(output.keys())[0]
                return output[key]
        # Fallback: call model directly
        return model(inputs)
    elif model_type in ("keras", "checkpoint"):
        return model(inputs, training=False)
    else:
        # checkpoint_raw - try __call__
        return model(inputs, training=False)


# ---------------------------------------------------------------------------
# 3D Bounding Box IoU Computation (Oriented Boxes)
# ---------------------------------------------------------------------------

def corners_from_box(box: np.ndarray) -> np.ndarray:
    """Compute the 8 corners of an oriented 3D bounding box.

    Args:
        box: array of shape (7,) -> [x, y, z, dx, dy, dz, yaw]
             (center_x, center_y, center_z, length, width, height, heading)

    Returns:
        corners: (8, 3) array of corner coordinates.
    """
    x, y, z, dx, dy, dz, yaw = box
    # Half extents
    hdx, hdy, hdz = dx / 2.0, dy / 2.0, dz / 2.0

    # 8 corners in local frame (before rotation)
    # Order: bottom-face (z-), then top-face (z+)
    local_corners = np.array([
        [-hdx, -hdy, -hdz],
        [+hdx, -hdy, -hdz],
        [+hdx, +hdy, -hdz],
        [-hdx, +hdy, -hdz],
        [-hdx, -hdy, +hdz],
        [+hdx, -hdy, +hdz],
        [+hdx, +hdy, +hdz],
        [-hdx, +hdy, +hdz],
    ])

    # Rotation matrix around Z-axis (yaw)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Rotate and translate
    corners = local_corners @ rotation.T + np.array([x, y, z])
    return corners


def polygon_area(vertices: np.ndarray) -> float:
    """Compute area of a convex polygon given ordered 2D vertices using the shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Compute the convex hull of a set of 2D points using Andrew's monotone chain.

    Args:
        points: (N, 2) array of 2D points.

    Returns:
        hull: (M, 2) array of hull vertices in counter-clockwise order.
    """
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull)


def polygon_clip_sutherland_hodgman(
    subject: np.ndarray, clip: np.ndarray
) -> np.ndarray:
    """Clip a convex polygon (subject) by another convex polygon (clip) using
    the Sutherland-Hodgman algorithm.

    Args:
        subject: (N, 2) vertices of the polygon to be clipped.
        clip: (M, 2) vertices of the clipping polygon.

    Returns:
        result: (K, 2) vertices of the clipped polygon (may be empty).
    """
    def inside(p, edge_start, edge_end):
        return (
            (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1])
            - (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])
        ) >= 0

    def line_intersection(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p1  # Degenerate; return one of the points
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return np.array([x, y])

    output = list(subject)
    if len(output) == 0:
        return np.array([]).reshape(0, 2)

    for i in range(len(clip)):
        if len(output) == 0:
            return np.array([]).reshape(0, 2)
        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]

        input_list = output
        output = []

        for j in range(len(input_list)):
            current = input_list[j]
            previous = input_list[j - 1]

            if inside(current, edge_start, edge_end):
                if not inside(previous, edge_start, edge_end):
                    intersection = line_intersection(
                        previous, current, edge_start, edge_end
                    )
                    output.append(intersection)
                output.append(current)
            elif inside(previous, edge_start, edge_end):
                intersection = line_intersection(
                    previous, current, edge_start, edge_end
                )
                output.append(intersection)

    if len(output) == 0:
        return np.array([]).reshape(0, 2)
    return np.array(output)


def iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute the 3D Intersection-over-Union of two oriented bounding boxes.

    Each box is [x, y, z, dx, dy, dz, yaw].

    The approach:
      1. Project boxes onto the XY plane and compute 2D intersection polygon area
         using Sutherland-Hodgman clipping.
      2. Compute the overlap along the Z axis.
      3. IoU = intersection_volume / union_volume.
    """
    corners_a = corners_from_box(box_a)  # (8, 3)
    corners_b = corners_from_box(box_b)  # (8, 3)

    # Bottom 4 corners for BEV polygon (indices 0..3 are the bottom face)
    poly_a = corners_a[:4, :2]  # (4, 2)
    poly_b = corners_b[:4, :2]  # (4, 2)

    # Order polygons counter-clockwise via convex hull
    poly_a = convex_hull_2d(poly_a)
    poly_b = convex_hull_2d(poly_b)

    # Clip poly_a by poly_b to get intersection polygon
    intersection_polygon = polygon_clip_sutherland_hodgman(poly_a, poly_b)

    if len(intersection_polygon) < 3:
        return 0.0

    inter_area = polygon_area(intersection_polygon)

    # Z-axis overlap
    z_min_a = corners_a[:, 2].min()
    z_max_a = corners_a[:, 2].max()
    z_min_b = corners_b[:, 2].min()
    z_max_b = corners_b[:, 2].max()

    z_overlap = max(0.0, min(z_max_a, z_max_b) - max(z_min_a, z_min_b))

    inter_volume = inter_area * z_overlap

    vol_a = box_a[3] * box_a[4] * box_a[5]
    vol_b = box_b[3] * box_b[4] * box_b[5]

    union_volume = vol_a + vol_b - inter_volume

    if union_volume <= 0:
        return 0.0

    return inter_volume / union_volume


# ---------------------------------------------------------------------------
# 3D Average Precision (AP) Computation
# ---------------------------------------------------------------------------

def compute_ap_11_point(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute Average Precision using the 11-point interpolation method.

    For each of the 11 recall thresholds [0.0, 0.1, ..., 1.0], take the
    maximum precision at recall >= threshold. AP is the mean of these values.
    """
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()
        else:
            ap += 0.0
    ap /= 11.0
    return ap


def compute_3d_ap(
    predictions: list,
    ground_truths: list,
    iou_threshold: float,
    class_id: int,
) -> float:
    """Compute 3D Average Precision for a single class at a given IoU threshold.

    Args:
        predictions: list of dicts per sample:
            {
                "boxes": np.ndarray (K, 7),  # predicted boxes [x,y,z,dx,dy,dz,yaw]
                "scores": np.ndarray (K,),   # confidence scores
                "classes": np.ndarray (K,),  # predicted class IDs
            }
        ground_truths: list of np.ndarray per sample, shape (M, 8)
            Each row: [x, y, z, dx, dy, dz, yaw, class_id]
        iou_threshold: IoU threshold for a true positive match.
        class_id: The class to evaluate.

    Returns:
        AP value (float).
    """
    # Gather all predictions for this class across all samples, with sample index
    all_preds = []  # (score, sample_idx, box_idx)
    for sample_idx, pred in enumerate(predictions):
        class_mask = pred["classes"] == class_id
        boxes = pred["boxes"][class_mask]
        scores = pred["scores"][class_mask]
        for i in range(len(scores)):
            all_preds.append((scores[i], sample_idx, boxes[i]))

    # Sort by confidence descending
    all_preds.sort(key=lambda x: x[0], reverse=True)

    # Count total ground truth boxes for this class
    num_gt = 0
    gt_matched = []  # per-sample list of matched flags
    for sample_idx, gt in enumerate(ground_truths):
        gt_class_mask = gt[:, 7].astype(int) == class_id
        n = gt_class_mask.sum()
        num_gt += n
        gt_matched.append(np.zeros(int(n), dtype=bool))

    if num_gt == 0:
        return 0.0

    # Build per-sample GT boxes for this class
    gt_boxes_per_sample = []
    for gt in ground_truths:
        gt_class_mask = gt[:, 7].astype(int) == class_id
        gt_boxes_per_sample.append(gt[gt_class_mask, :7])

    # Compute precision/recall
    tp = np.zeros(len(all_preds), dtype=np.float32)
    fp = np.zeros(len(all_preds), dtype=np.float32)

    for pred_idx, (score, sample_idx, pred_box) in enumerate(all_preds):
        gt_boxes = gt_boxes_per_sample[sample_idx]
        matched_flags = gt_matched[sample_idx]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx in range(len(gt_boxes)):
            if matched_flags[gt_idx]:
                continue
            iou_val = iou_3d(pred_box, gt_boxes[gt_idx])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1.0
            matched_flags[best_gt_idx] = True
        else:
            fp[pred_idx] = 1.0

    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / num_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = compute_ap_11_point(precision, recall)
    return ap


# ---------------------------------------------------------------------------
# Classification Evaluation
# ---------------------------------------------------------------------------

def evaluate_classification(model, model_type, data_dir, num_points, batch_size, num_classes):
    """Run classification evaluation and return metrics dict."""
    print("[INFO] Loading classification test data...")
    dataset, num_samples = load_classification_data(data_dir, num_points, batch_size)
    print(f"[INFO] Loaded {num_samples} samples.")

    all_preds = []
    all_labels = []

    print("[INFO] Running inference...")
    for batch_points, batch_labels in dataset:
        logits = run_inference(model, model_type, batch_points)
        preds = tf.argmax(logits, axis=-1).numpy()
        all_preds.append(preds)
        all_labels.append(batch_labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Overall accuracy
    accuracy = np.mean(all_preds == all_labels)

    # Per-class metrics
    per_class = {}
    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[f"class_{c}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(all_labels == c)),
        }

    results = {
        "task": "classification",
        "num_samples": int(num_samples),
        "overall_accuracy": float(accuracy),
        "per_class": per_class,
    }
    return results


# ---------------------------------------------------------------------------
# Detection Evaluation
# ---------------------------------------------------------------------------

def evaluate_detection(model, model_type, data_dir, num_points, batch_size, num_classes):
    """Run detection evaluation and return metrics dict."""
    print("[INFO] Loading detection test data...")
    all_points, all_annotations, num_samples = load_detection_data(
        data_dir, num_points, batch_size
    )
    print(f"[INFO] Loaded {num_samples} samples.")

    print("[INFO] Running inference...")
    predictions = []

    # Process in batches
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_pts = np.array(all_points[start_idx:end_idx], dtype=np.float32)
        batch_tensor = tf.constant(batch_pts)

        output = run_inference(model, model_type, batch_tensor)

        # Expected output format from detection model:
        # A dict or tuple with keys/components: boxes (B, K, 7), scores (B, K), classes (B, K)
        if isinstance(output, dict):
            boxes = output.get("boxes", output.get("pred_boxes"))
            scores = output.get("scores", output.get("pred_scores"))
            classes = output.get("classes", output.get("pred_classes"))
            if hasattr(boxes, "numpy"):
                boxes = boxes.numpy()
            if hasattr(scores, "numpy"):
                scores = scores.numpy()
            if hasattr(classes, "numpy"):
                classes = classes.numpy()
        elif isinstance(output, (list, tuple)):
            boxes = output[0].numpy() if hasattr(output[0], "numpy") else np.array(output[0])
            scores = output[1].numpy() if hasattr(output[1], "numpy") else np.array(output[1])
            classes = output[2].numpy() if hasattr(output[2], "numpy") else np.array(output[2])
        else:
            # Single tensor output - attempt to parse
            out_np = output.numpy() if hasattr(output, "numpy") else np.array(output)
            # Assume last dim: 7 (box) + 1 (score) + 1 (class) = 9
            boxes = out_np[..., :7]
            scores = out_np[..., 7]
            classes = out_np[..., 8].astype(int)

        for i in range(boxes.shape[0]):
            # Filter out padding (score > 0)
            valid_mask = scores[i] > 0.0
            predictions.append({
                "boxes": boxes[i][valid_mask],
                "scores": scores[i][valid_mask],
                "classes": classes[i][valid_mask].astype(int),
            })

    # Compute AP for each class at IoU 0.5 and 0.7
    iou_thresholds = [0.5, 0.7]
    per_class = {}
    mean_ap = {0.5: [], 0.7: []}

    for c in range(num_classes):
        class_results = {}
        for iou_thresh in iou_thresholds:
            ap = compute_3d_ap(predictions, all_annotations, iou_thresh, c)
            class_results[f"AP@{iou_thresh}"] = float(ap)
            mean_ap[iou_thresh].append(ap)
        per_class[f"class_{c}"] = class_results

    results = {
        "task": "detection",
        "num_samples": int(num_samples),
        "mAP@0.5": float(np.mean(mean_ap[0.5])) if mean_ap[0.5] else 0.0,
        "mAP@0.7": float(np.mean(mean_ap[0.7])) if mean_ap[0.7] else 0.0,
        "per_class": per_class,
    }
    return results


# ---------------------------------------------------------------------------
# Segmentation Evaluation
# ---------------------------------------------------------------------------

def evaluate_segmentation(model, model_type, data_dir, num_points, batch_size, num_classes):
    """Run segmentation evaluation and return metrics dict."""
    print("[INFO] Loading segmentation test data...")
    dataset, num_samples = load_segmentation_data(data_dir, num_points, batch_size)
    print(f"[INFO] Loaded {num_samples} samples.")

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    print("[INFO] Running inference...")
    for batch_points, batch_labels in dataset:
        logits = run_inference(model, model_type, batch_points)
        # logits shape: (B, num_points, num_classes)
        preds = tf.argmax(logits, axis=-1).numpy()  # (B, num_points)
        labels = batch_labels.numpy()  # (B, num_points)

        for i in range(preds.shape[0]):
            for c_true in range(num_classes):
                for c_pred in range(num_classes):
                    confusion[c_true, c_pred] += np.sum(
                        (labels[i] == c_true) & (preds[i] == c_pred)
                    )

    # Compute per-class IoU
    per_class_iou = {}
    iou_values = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = tp + fp + fn
        iou = float(tp) / float(denom) if denom > 0 else 0.0
        per_class_iou[f"class_{c}"] = {
            "iou": float(iou),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
        iou_values.append(iou)

    mean_iou = float(np.mean(iou_values))

    # Overall pixel accuracy
    overall_accuracy = float(np.trace(confusion)) / float(confusion.sum()) if confusion.sum() > 0 else 0.0

    results = {
        "task": "segmentation",
        "num_samples": int(num_samples),
        "mean_iou": mean_iou,
        "overall_accuracy": overall_accuracy,
        "per_class": per_class_iou,
    }
    return results


# ---------------------------------------------------------------------------
# Results Formatting
# ---------------------------------------------------------------------------

def print_classification_results(results: dict):
    """Print classification results as a formatted table."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Total samples:     {results['num_samples']}")
    print(f"  Overall Accuracy:  {results['overall_accuracy']:.4f}")
    print("-" * 70)
    print(f"  {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for class_name, metrics in results["per_class"].items():
        print(
            f"  {class_name:<12} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1']:<12.4f} "
            f"{metrics['support']:<10}"
        )
    print("=" * 70)


def print_detection_results(results: dict):
    """Print detection results as a formatted table."""
    print("\n" + "=" * 70)
    print("3D DETECTION EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Total samples:  {results['num_samples']}")
    print(f"  mAP@0.5:       {results['mAP@0.5']:.4f}")
    print(f"  mAP@0.7:       {results['mAP@0.7']:.4f}")
    print("-" * 70)
    print(f"  {'Class':<12} {'AP@0.5':<12} {'AP@0.7':<12}")
    print("-" * 70)
    for class_name, metrics in results["per_class"].items():
        print(
            f"  {class_name:<12} "
            f"{metrics['AP@0.5']:<12.4f} "
            f"{metrics['AP@0.7']:<12.4f}"
        )
    print("=" * 70)


def print_segmentation_results(results: dict):
    """Print segmentation results as a formatted table."""
    print("\n" + "=" * 70)
    print("SEGMENTATION EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Total samples:      {results['num_samples']}")
    print(f"  Mean IoU:           {results['mean_iou']:.4f}")
    print(f"  Overall Accuracy:   {results['overall_accuracy']:.4f}")
    print("-" * 70)
    print(f"  {'Class':<12} {'IoU':<12} {'TP':<10} {'FP':<10} {'FN':<10}")
    print("-" * 70)
    for class_name, metrics in results["per_class"].items():
        print(
            f"  {class_name:<12} "
            f"{metrics['iou']:<12.4f} "
            f"{metrics['tp']:<10} "
            f"{metrics['fp']:<10} "
            f"{metrics['fn']:<10}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"[INFO] Task:        {args.task}")
    print(f"[INFO] Model path:  {args.model_path}")
    print(f"[INFO] Data dir:    {args.data_dir}")
    print(f"[INFO] Batch size:  {args.batch_size}")
    print(f"[INFO] Num points:  {args.num_points}")
    print(f"[INFO] Num classes: {args.num_classes}")
    print()

    # Load model
    print("[INFO] Loading model...")
    model, model_type = load_model(args.model_path)
    print(f"[INFO] Model loaded successfully (format: {model_type}).")

    # Run evaluation
    if args.task == "classification":
        results = evaluate_classification(
            model, model_type, args.data_dir,
            args.num_points, args.batch_size, args.num_classes,
        )
        print_classification_results(results)

    elif args.task == "detection":
        results = evaluate_detection(
            model, model_type, args.data_dir,
            args.num_points, args.batch_size, args.num_classes,
        )
        print_detection_results(results)

    elif args.task == "segmentation":
        results = evaluate_segmentation(
            model, model_type, args.data_dir,
            args.num_points, args.batch_size, args.num_classes,
        )
        print_segmentation_results(results)

    else:
        print(f"[ERROR] Unknown task: {args.task}")
        sys.exit(1)

    # Save results to JSON
    output_path = args.output_json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
