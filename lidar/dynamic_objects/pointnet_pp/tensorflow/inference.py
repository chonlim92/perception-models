"""
PointNet++ TensorFlow 2 Inference Script

Supports classification, 3D object detection, and semantic segmentation
on point cloud data (KITTI .bin or NumPy .npy format).

Usage:
    python inference.py --model_path ./saved_model --input cloud.bin --task detection
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PointNet++ inference on point cloud data (TensorFlow 2)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the TensorFlow SavedModel directory"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to point cloud file (.bin KITTI format or .npy)"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["classification", "detection", "segmentation"],
        help="Inference task: classification, detection, or segmentation"
    )
    parser.add_argument(
        "--num_points", type=int, default=16384,
        help="Number of points to sample from the point cloud (default: 16384)"
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.5,
        help="Confidence threshold for detection task (default: 0.5)"
    )
    parser.add_argument(
        "--nms_threshold", type=float, default=0.3,
        help="NMS IoU threshold for detection task (default: 0.3)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (detection: .txt, segmentation: .npy)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Point cloud loading and preprocessing
# ---------------------------------------------------------------------------

def load_point_cloud(file_path: str) -> np.ndarray:
    """Load point cloud from .bin (KITTI format) or .npy file.

    KITTI .bin format: N x 4 float32 array (x, y, z, reflectance).
    .npy format: expects at least N x 3 (x, y, z), optional extra channels.

    Returns:
        np.ndarray of shape (N, C) where C >= 3.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")

    if path.suffix == ".bin":
        points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    elif path.suffix == ".npy":
        points = np.load(str(path)).astype(np.float32)
        if points.ndim == 1:
            raise ValueError("Expected at least a 2D array from .npy file")
        if points.shape[1] < 3:
            raise ValueError(
                f"Point cloud must have at least 3 columns (x,y,z), got {points.shape[1]}"
            )
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .bin or .npy")

    return points


def preprocess_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """Subsample or pad point cloud to a fixed number of points, then normalize.

    Args:
        points: (N, C) raw point cloud array.
        num_points: Target number of points.

    Returns:
        Normalized point cloud of shape (num_points, C).
    """
    n = points.shape[0]

    if n == 0:
        raise ValueError("Point cloud is empty (0 points)")

    if n >= num_points:
        # Random subsample without replacement
        indices = np.random.choice(n, num_points, replace=False)
        points = points[indices]
    else:
        # Pad by repeating random points
        pad_indices = np.random.choice(n, num_points - n, replace=True)
        points = np.concatenate([points, points[pad_indices]], axis=0)

    # Normalize XYZ coordinates to zero-mean, unit-scale
    xyz = points[:, :3]
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid
    max_dist = np.max(np.linalg.norm(xyz_centered, axis=1))
    if max_dist > 0:
        xyz_centered = xyz_centered / max_dist

    points[:, :3] = xyz_centered
    return points


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load a TensorFlow SavedModel.

    Args:
        model_path: Path to the SavedModel directory.

    Returns:
        Loaded model with a callable inference function.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    model = tf.saved_model.load(model_path)

    # Try to get the default serving signature
    if hasattr(model, "signatures") and "serving_default" in model.signatures:
        infer_fn = model.signatures["serving_default"]
    elif hasattr(model, "__call__"):
        infer_fn = model
    else:
        raise RuntimeError(
            "Model has no 'serving_default' signature and is not directly callable. "
            "Ensure the SavedModel was exported with a valid signature."
        )

    return model, infer_fn


# ---------------------------------------------------------------------------
# 3D NMS (Bird's Eye View IoU)
# ---------------------------------------------------------------------------

def compute_bev_corners(box: np.ndarray) -> np.ndarray:
    """Compute the 4 BEV corners of a 3D bounding box.

    Args:
        box: array of [x, y, z, w, h, l, yaw] where
             x, y, z = center; w = width (along x); h = height (along z);
             l = length (along y); yaw = rotation around z-axis.

    Returns:
        (4, 2) array of BEV corner coordinates (x, y).
    """
    x, y, _, w, _, l, yaw = box
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Half extents
    hw = w / 2.0
    hl = l / 2.0

    # Corners in local frame (BEV: x-y plane)
    corners_local = np.array([
        [ hw,  hl],
        [-hw,  hl],
        [-hw, -hl],
        [ hw, -hl],
    ])

    # Rotation matrix
    rot = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw],
    ])

    corners_world = (rot @ corners_local.T).T + np.array([x, y])
    return corners_world


def polygon_area(vertices: np.ndarray) -> float:
    """Compute area of a polygon using the shoelace formula."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def polygon_clip_sutherland_hodgman(
    subject: np.ndarray, clip: np.ndarray
) -> np.ndarray:
    """Clip a polygon (subject) by a convex polygon (clip) using Sutherland-Hodgman.

    Args:
        subject: (M, 2) polygon vertices.
        clip: (N, 2) convex clipping polygon vertices.

    Returns:
        Clipped polygon vertices as (K, 2) array (may be empty).
    """
    def inside(p, edge_start, edge_end):
        return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - \
               (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0]) >= 0

    def intersection(p1, p2, edge_start, edge_end):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = edge_start
        x4, y4 = edge_end
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return np.array([ix, iy])

    output = list(subject)

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
                    output.append(intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif inside(previous, edge_start, edge_end):
                output.append(intersection(previous, current, edge_start, edge_end))

    if len(output) == 0:
        return np.array([]).reshape(0, 2)
    return np.array(output)


def bev_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute BEV IoU between two 3D bounding boxes.

    Each box: [x, y, z, w, h, l, yaw]
    """
    corners_a = compute_bev_corners(box_a)
    corners_b = compute_bev_corners(box_b)

    # Compute intersection polygon
    intersection_poly = polygon_clip_sutherland_hodgman(corners_a, corners_b)

    if len(intersection_poly) < 3:
        return 0.0

    inter_area = polygon_area(intersection_poly)
    area_a = box_a[3] * box_a[5]  # w * l
    area_b = box_b[3] * box_b[5]  # w * l
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def nms_3d_bev(
    detections: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float
) -> np.ndarray:
    """Perform 3D NMS using BEV IoU.

    Args:
        detections: (N, 7) array of [x, y, z, w, h, l, yaw].
        scores: (N,) confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Indices of detections to keep.
    """
    if len(detections) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    keep = []

    suppressed = np.zeros(len(detections), dtype=bool)

    for i in range(len(order)):
        idx = order[i]
        if suppressed[idx]:
            continue
        keep.append(idx)

        for j in range(i + 1, len(order)):
            jdx = order[j]
            if suppressed[jdx]:
                continue
            iou = bev_iou(detections[idx], detections[jdx])
            if iou > iou_threshold:
                suppressed[jdx] = True

    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Post-processing per task
# ---------------------------------------------------------------------------

def postprocess_classification(output_tensor: np.ndarray) -> tuple:
    """Post-process classification output.

    Args:
        output_tensor: (1, num_classes) logits or probabilities.

    Returns:
        (predicted_class_index, confidence)
    """
    if output_tensor.ndim == 1:
        probs = output_tensor
    else:
        probs = output_tensor[0]

    # Apply softmax if outputs look like logits (contain negatives or > 1)
    if np.any(probs < 0) or np.any(probs > 1.0 + 1e-6):
        exp_probs = np.exp(probs - np.max(probs))
        probs = exp_probs / np.sum(exp_probs)

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    return predicted_class, confidence


def postprocess_detection(
    output_dict: dict,
    conf_threshold: float,
    nms_threshold: float
) -> list:
    """Post-process detection output.

    Expected output_dict keys:
        - 'boxes' or 'pred_boxes': (1, N, 7) [x, y, z, w, h, l, yaw]
        - 'scores' or 'pred_scores': (1, N)
        - 'classes' or 'pred_classes': (1, N)

    Returns:
        List of dicts with keys: class_id, confidence, x, y, z, w, h, l, yaw
    """
    # Extract arrays from output dict with flexible key naming
    boxes = None
    scores = None
    classes = None

    for key in ["boxes", "pred_boxes", "detection_boxes", "output_boxes"]:
        if key in output_dict:
            boxes = np.array(output_dict[key])
            break

    for key in ["scores", "pred_scores", "detection_scores", "output_scores"]:
        if key in output_dict:
            scores = np.array(output_dict[key])
            break

    for key in ["classes", "pred_classes", "detection_classes", "output_classes"]:
        if key in output_dict:
            classes = np.array(output_dict[key])
            break

    if boxes is None or scores is None:
        raise RuntimeError(
            "Detection model output must contain 'boxes'/'pred_boxes' and "
            "'scores'/'pred_scores' keys. "
            f"Available keys: {list(output_dict.keys())}"
        )

    # Remove batch dimension if present
    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores.ndim == 2:
        scores = scores[0]
    if classes is not None and classes.ndim == 2:
        classes = classes[0]

    # Filter by confidence threshold
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    if classes is not None:
        classes = classes[mask]
    else:
        classes = np.zeros(len(scores), dtype=np.int32)

    if len(boxes) == 0:
        return []

    # Apply 3D NMS using BEV IoU
    keep_indices = nms_3d_bev(boxes, scores, nms_threshold)

    results = []
    for idx in keep_indices:
        box = boxes[idx]
        results.append({
            "class_id": int(classes[idx]),
            "confidence": float(scores[idx]),
            "x": float(box[0]),
            "y": float(box[1]),
            "z": float(box[2]),
            "w": float(box[3]),
            "h": float(box[4]),
            "l": float(box[5]),
            "yaw": float(box[6]),
        })

    return results


def postprocess_segmentation(output_tensor: np.ndarray) -> np.ndarray:
    """Post-process segmentation output.

    Args:
        output_tensor: (1, N, num_classes) logits or (1, N) class indices.

    Returns:
        (N,) per-point class labels.
    """
    if output_tensor.ndim == 3:
        # (1, N, num_classes) -> argmax over classes
        labels = np.argmax(output_tensor[0], axis=-1)
    elif output_tensor.ndim == 2:
        labels = output_tensor[0].astype(np.int32)
    elif output_tensor.ndim == 1:
        labels = output_tensor.astype(np.int32)
    else:
        raise ValueError(
            f"Unexpected segmentation output shape: {output_tensor.shape}"
        )

    return labels


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_inference(infer_fn, points: np.ndarray, task: str) -> dict:
    """Run model inference.

    Args:
        infer_fn: Callable model or signature function.
        points: Preprocessed point cloud (num_points, C).
        task: One of classification, detection, segmentation.

    Returns:
        Raw model output as dict or tensor.
    """
    # Add batch dimension: (1, num_points, C)
    input_tensor = tf.constant(points[np.newaxis, ...], dtype=tf.float32)

    # Time the inference
    start_time = time.perf_counter()

    # Handle different callable signatures
    if hasattr(infer_fn, "structured_input_signature"):
        # This is a concrete function from signatures
        # Determine input key name
        input_keys = list(infer_fn.structured_input_signature[1].keys()) if \
            infer_fn.structured_input_signature[1] else []
        if input_keys:
            input_key = input_keys[0]
            output = infer_fn(**{input_key: input_tensor})
        else:
            output = infer_fn(input_tensor)
    else:
        output = infer_fn(input_tensor)

    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000.0

    # Convert output to numpy dict
    if isinstance(output, dict):
        output_np = {k: v.numpy() if hasattr(v, "numpy") else np.array(v)
                     for k, v in output.items()}
    elif isinstance(output, (list, tuple)):
        output_np = {f"output_{i}": v.numpy() if hasattr(v, "numpy") else np.array(v)
                     for i, v in enumerate(output)}
    else:
        output_np = output.numpy() if hasattr(output, "numpy") else np.array(output)

    return output_np, inference_time_ms


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_classification_output(class_id: int, confidence: float) -> str:
    lines = [
        "=== Classification Result ===",
        f"  Predicted class: {class_id}",
        f"  Confidence:      {confidence:.4f}",
    ]
    return "\n".join(lines)


def format_detection_output(detections: list) -> str:
    lines = [f"=== Detection Results ({len(detections)} objects) ==="]
    for i, det in enumerate(detections):
        lines.append(
            f"  [{i:3d}] class={det['class_id']:2d}  "
            f"conf={det['confidence']:.3f}  "
            f"pos=({det['x']:.2f}, {det['y']:.2f}, {det['z']:.2f})  "
            f"size=({det['w']:.2f}, {det['h']:.2f}, {det['l']:.2f})  "
            f"yaw={det['yaw']:.3f}"
        )
    return "\n".join(lines)


def save_detection_output(detections: list, output_path: str):
    """Save detections to a text file in KITTI-like format."""
    with open(output_path, "w") as f:
        f.write("# class_id confidence x y z w h l yaw\n")
        for det in detections:
            f.write(
                f"{det['class_id']} {det['confidence']:.4f} "
                f"{det['x']:.4f} {det['y']:.4f} {det['z']:.4f} "
                f"{det['w']:.4f} {det['h']:.4f} {det['l']:.4f} "
                f"{det['yaw']:.4f}\n"
            )


def save_segmentation_output(labels: np.ndarray, output_path: str):
    """Save per-point segmentation labels as .npy."""
    np.save(output_path, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"PointNet++ TensorFlow 2 Inference")
    print(f"{'=' * 50}")
    print(f"  Task:           {args.task}")
    print(f"  Model:          {args.model_path}")
    print(f"  Input:          {args.input}")
    print(f"  Num points:     {args.num_points}")
    if args.task == "detection":
        print(f"  Conf threshold: {args.conf_threshold}")
        print(f"  NMS threshold:  {args.nms_threshold}")
    if args.output:
        print(f"  Output:         {args.output}")
    print(f"{'=' * 50}")

    # Load point cloud
    print("\n[1/4] Loading point cloud...")
    load_start = time.perf_counter()
    raw_points = load_point_cloud(args.input)
    load_time = (time.perf_counter() - load_start) * 1000.0
    print(f"  Loaded {raw_points.shape[0]} points with {raw_points.shape[1]} channels "
          f"({load_time:.1f} ms)")

    # Preprocess
    print("\n[2/4] Preprocessing...")
    preprocess_start = time.perf_counter()
    points = preprocess_points(raw_points, args.num_points)
    preprocess_time = (time.perf_counter() - preprocess_start) * 1000.0
    print(f"  Resampled to {points.shape[0]} points, normalized ({preprocess_time:.1f} ms)")

    # Load model
    print("\n[3/4] Loading model...")
    model_load_start = time.perf_counter()
    model, infer_fn = load_model(args.model_path)
    model_load_time = (time.perf_counter() - model_load_start) * 1000.0
    print(f"  Model loaded ({model_load_time:.1f} ms)")

    # Run inference
    print("\n[4/4] Running inference...")
    output, inference_time_ms = run_inference(infer_fn, points, args.task)

    # Post-process based on task
    print(f"\n{'=' * 50}")

    if args.task == "classification":
        if isinstance(output, dict):
            # Take first output tensor from dict
            out_key = list(output.keys())[0]
            out_tensor = output[out_key]
        else:
            out_tensor = output

        class_id, confidence = postprocess_classification(out_tensor)
        result_str = format_classification_output(class_id, confidence)
        print(result_str)

        if args.output:
            with open(args.output, "w") as f:
                f.write(f"{class_id},{confidence:.6f}\n")
            print(f"\n  Result saved to: {args.output}")

    elif args.task == "detection":
        if isinstance(output, dict):
            detections = postprocess_detection(
                output, args.conf_threshold, args.nms_threshold
            )
        else:
            raise RuntimeError(
                "Detection model must return a dictionary with 'boxes' and 'scores' keys."
            )

        result_str = format_detection_output(detections)
        print(result_str)

        if args.output:
            save_detection_output(detections, args.output)
            print(f"\n  Detections saved to: {args.output}")

    elif args.task == "segmentation":
        if isinstance(output, dict):
            out_key = list(output.keys())[0]
            out_tensor = output[out_key]
        else:
            out_tensor = output

        labels = postprocess_segmentation(out_tensor)
        unique_classes = np.unique(labels)
        print(f"=== Segmentation Result ===")
        print(f"  Points segmented: {len(labels)}")
        print(f"  Unique classes:   {len(unique_classes)} {unique_classes.tolist()}")

        # Class distribution
        for cls_id in unique_classes:
            count = np.sum(labels == cls_id)
            pct = 100.0 * count / len(labels)
            print(f"    Class {cls_id:3d}: {count:7d} points ({pct:.1f}%)")

        output_path = args.output if args.output else "segmentation_labels.npy"
        save_segmentation_output(labels, output_path)
        print(f"\n  Labels saved to: {output_path}")

    # Performance summary
    total_time = load_time + preprocess_time + model_load_time + inference_time_ms
    print(f"\n{'=' * 50}")
    print(f"  Performance Summary")
    print(f"{'=' * 50}")
    print(f"  Point cloud loading:  {load_time:8.1f} ms")
    print(f"  Preprocessing:        {preprocess_time:8.1f} ms")
    print(f"  Model loading:        {model_load_time:8.1f} ms")
    print(f"  Inference:            {inference_time_ms:8.1f} ms")
    print(f"  {'─' * 36}")
    print(f"  Total:                {total_time:8.1f} ms")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
