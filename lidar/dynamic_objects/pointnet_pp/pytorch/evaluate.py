"""
Evaluation script for PointNet++ 3D object detection.

Implements KITTI-style 3D Average Precision (AP) evaluation:
- 3D IoU computation (BEV IoU * height overlap)
- 40-point interpolation for AP
- Per-class metrics at standard IoU thresholds
"""

import math
import numpy as np
from typing import List, Dict, Tuple

import torch


def compute_bev_iou(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    Compute Bird's Eye View (BEV) IoU between two sets of boxes.

    Each box is represented as (x, y, w, l, yaw) where:
    - x, y: center coordinates
    - w, l: width and length
    - yaw: rotation angle around Z-axis

    Uses a simplified axis-aligned approximation for rotated IoU.
    For exact rotated IoU, a polygon intersection would be needed.

    Args:
        boxes_a: (M, 5) array of boxes [x, y, w, l, yaw]
        boxes_b: (N, 5) array of boxes [x, y, w, l, yaw]

    Returns:
        IoU matrix of shape (M, N)
    """
    M = boxes_a.shape[0]
    N = boxes_b.shape[0]

    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)

    iou_matrix = np.zeros((M, N), dtype=np.float32)

    for i in range(M):
        xa, ya, wa, la, yaw_a = boxes_a[i]
        corners_a = _get_rotated_corners_2d(xa, ya, wa, la, yaw_a)

        for j in range(N):
            xb, yb, wb, lb, yaw_b = boxes_b[j]
            corners_b = _get_rotated_corners_2d(xb, yb, wb, lb, yaw_b)

            # Compute intersection area using Sutherland-Hodgman polygon clipping
            intersection_area = _polygon_intersection_area(corners_a, corners_b)
            area_a = wa * la
            area_b = wb * lb
            union = area_a + area_b - intersection_area

            if union > 0:
                iou_matrix[i, j] = intersection_area / union

    return iou_matrix


def _get_rotated_corners_2d(
    cx: float, cy: float, w: float, l: float, yaw: float
) -> np.ndarray:
    """Get 4 BEV corners of a rotated box."""
    cos_a = np.cos(yaw)
    sin_a = np.sin(yaw)

    hw, hl = w / 2.0, l / 2.0

    # Local corners (before rotation)
    corners_local = np.array([
        [-hw, -hl],
        [hw, -hl],
        [hw, hl],
        [-hw, hl],
    ], dtype=np.float64)

    # Rotation matrix
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

    # Rotate and translate
    corners = corners_local @ R.T + np.array([cx, cy])
    return corners


def _polygon_intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """
    Compute intersection area of two convex polygons using
    Sutherland-Hodgman clipping algorithm.

    Args:
        poly_a: (4, 2) polygon vertices
        poly_b: (4, 2) polygon vertices

    Returns:
        Intersection area
    """
    # Clip poly_a by each edge of poly_b
    output = list(poly_a)

    for i in range(len(poly_b)):
        if len(output) == 0:
            return 0.0

        edge_start = poly_b[i]
        edge_end = poly_b[(i + 1) % len(poly_b)]

        input_list = output
        output = []

        for j in range(len(input_list)):
            current = input_list[j]
            prev = input_list[j - 1]

            current_inside = _is_left(edge_start, edge_end, current)
            prev_inside = _is_left(edge_start, edge_end, prev)

            if current_inside:
                if not prev_inside:
                    intersection = _line_intersection(
                        edge_start, edge_end, prev, current
                    )
                    if intersection is not None:
                        output.append(intersection)
                output.append(current)
            elif prev_inside:
                intersection = _line_intersection(
                    edge_start, edge_end, prev, current
                )
                if intersection is not None:
                    output.append(intersection)

    if len(output) < 3:
        return 0.0

    return _polygon_area(np.array(output))


def _is_left(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
    """Check if point p is to the left of line segment a->b."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0


def _line_intersection(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> np.ndarray:
    """Find intersection of line p1-p2 and p3-p4."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return np.array([x, y])


def _polygon_area(vertices: np.ndarray) -> float:
    """Compute area of a polygon given vertices using the shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def compute_height_overlap(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    Compute height overlap ratio between two sets of 3D boxes.

    Args:
        boxes_a: (M, 2) array with [z_center, height] for each box
        boxes_b: (N, 2) array with [z_center, height] for each box

    Returns:
        Height overlap matrix of shape (M, N), normalized by min height
    """
    M = boxes_a.shape[0]
    N = boxes_b.shape[0]

    overlap = np.zeros((M, N), dtype=np.float32)

    for i in range(M):
        z_a = boxes_a[i, 0]
        h_a = boxes_a[i, 1]
        top_a = z_a + h_a / 2.0
        bot_a = z_a - h_a / 2.0

        for j in range(N):
            z_b = boxes_b[j, 0]
            h_b = boxes_b[j, 1]
            top_b = z_b + h_b / 2.0
            bot_b = z_b - h_b / 2.0

            inter_top = min(top_a, top_b)
            inter_bot = max(bot_a, bot_b)
            inter_h = max(0.0, inter_top - inter_bot)

            union_h = max(top_a, top_b) - min(bot_a, bot_b)
            if union_h > 0:
                overlap[i, j] = inter_h / union_h

    return overlap


def compute_iou_3d(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
) -> np.ndarray:
    """
    Compute 3D IoU between predicted and ground truth boxes.

    3D IoU = BEV IoU * height_overlap_ratio (approximation)

    For exact 3D IoU, one would need to compute volume intersection,
    but BEV IoU * height overlap is the standard KITTI approximation.

    Args:
        pred_boxes: (M, 7) array [x, y, z, w, l, h, yaw]
        gt_boxes: (N, 7) array [x, y, z, w, l, h, yaw]

    Returns:
        3D IoU matrix of shape (M, N)
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    # BEV IoU: use (x, y, w, l, yaw)
    bev_a = pred_boxes[:, [0, 1, 3, 4, 6]]  # x, y, w, l, yaw
    bev_b = gt_boxes[:, [0, 1, 3, 4, 6]]
    bev_iou = compute_bev_iou(bev_a, bev_b)

    # Height overlap: use (z, h)
    height_a = pred_boxes[:, [2, 5]]  # z, h
    height_b = gt_boxes[:, [2, 5]]
    h_overlap = compute_height_overlap(height_a, height_b)

    # 3D IoU approximation
    iou_3d = bev_iou * h_overlap

    return iou_3d


def compute_ap_40(
    recall: np.ndarray,
    precision: np.ndarray,
) -> float:
    """
    Compute Average Precision using 40-point interpolation (KITTI style).

    Args:
        recall: Array of recall values (sorted ascending)
        precision: Array of precision values corresponding to recalls

    Returns:
        AP value
    """
    # 40 recall points evenly spaced from 0 to 1
    recall_thresholds = np.linspace(0.0, 1.0, 41)

    ap = 0.0
    for t in recall_thresholds:
        # Find precision at recall >= t
        mask = recall >= t
        if mask.any():
            p = precision[mask].max()
        else:
            p = 0.0
        ap += p

    ap /= 41.0
    return ap


def evaluate_detections(
    predictions: List[Dict],
    ground_truths: List[Dict],
    classes: List[str],
    iou_thresholds: Dict[str, float] = None,
) -> Dict:
    """
    Evaluate 3D object detection results using KITTI-style metrics.

    Args:
        predictions: List of per-sample predictions, each dict containing:
            'boxes': (M, 7) numpy array [x, y, z, w, l, h, yaw]
            'scores': (M,) confidence scores
            'labels': (M,) class indices (1-indexed)
        ground_truths: List of per-sample ground truths, each dict containing:
            'boxes': (N, 7) numpy array
            'labels': (N,) class indices (1-indexed)
        classes: List of class names (1-indexed mapping)
        iou_thresholds: Per-class IoU thresholds for positive match
                        Default: 0.7 for Car, 0.5 for Pedestrian/Cyclist

    Returns:
        Dictionary with per-class AP and summary metrics
    """
    if iou_thresholds is None:
        iou_thresholds = {
            "Car": 0.7,
            "Pedestrian": 0.5,
            "Cyclist": 0.5,
        }
        # Default threshold for unlisted classes
        default_iou = 0.5
    else:
        default_iou = 0.5

    results = {}
    all_aps = []

    for cls_idx, cls_name in enumerate(classes, start=1):
        iou_thresh = iou_thresholds.get(cls_name, default_iou)

        # Gather all predictions and GTs for this class
        all_pred_scores = []
        all_pred_tp = []
        total_gt = 0

        for sample_idx in range(len(predictions)):
            pred = predictions[sample_idx]
            gt = ground_truths[sample_idx]

            # Filter predictions and GTs for this class
            pred_mask = pred["labels"] == cls_idx
            gt_mask = gt["labels"] == cls_idx

            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]
            gt_boxes = gt["boxes"][gt_mask]

            num_gt = gt_boxes.shape[0]
            total_gt += num_gt

            if pred_boxes.shape[0] == 0:
                continue

            # Sort predictions by score (descending)
            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]

            # Compute 3D IoU
            if num_gt > 0:
                iou_matrix = compute_iou_3d(pred_boxes, gt_boxes)
            else:
                iou_matrix = np.zeros(
                    (pred_boxes.shape[0], 0), dtype=np.float32
                )

            # Greedy matching
            gt_matched = np.zeros(num_gt, dtype=bool)
            for i in range(pred_boxes.shape[0]):
                all_pred_scores.append(pred_scores[i])

                if num_gt == 0:
                    all_pred_tp.append(False)
                    continue

                # Find best matching GT
                best_gt_idx = np.argmax(iou_matrix[i])
                best_iou = iou_matrix[i, best_gt_idx]

                if best_iou >= iou_thresh and not gt_matched[best_gt_idx]:
                    all_pred_tp.append(True)
                    gt_matched[best_gt_idx] = True
                else:
                    all_pred_tp.append(False)

        # Compute precision-recall curve
        if total_gt == 0:
            results[cls_name] = {"AP": 0.0, "num_gt": 0, "num_pred": len(all_pred_scores)}
            continue

        all_pred_scores = np.array(all_pred_scores)
        all_pred_tp = np.array(all_pred_tp)

        # Sort by score descending
        sort_idx = np.argsort(-all_pred_scores)
        all_pred_tp = all_pred_tp[sort_idx]

        # Cumulative TP and FP
        tp_cumsum = np.cumsum(all_pred_tp)
        fp_cumsum = np.cumsum(~all_pred_tp)

        recall = tp_cumsum / total_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Compute AP with 40-point interpolation
        ap = compute_ap_40(recall, precision)

        results[cls_name] = {
            "AP": ap,
            "num_gt": total_gt,
            "num_pred": len(all_pred_scores),
            "max_recall": recall[-1] if len(recall) > 0 else 0.0,
        }
        all_aps.append(ap)

    # Summary metrics
    results["mAP"] = np.mean(all_aps) if all_aps else 0.0

    return results


def generate_evaluation_report(results: Dict, classes: List[str]) -> str:
    """
    Generate a formatted evaluation report string.

    Args:
        results: Results dictionary from evaluate_detections
        classes: List of class names

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("3D Object Detection Evaluation Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'Class':<20} {'AP':>8} {'#GT':>8} {'#Pred':>8} {'Recall':>8}")
    lines.append("-" * 60)

    for cls_name in classes:
        if cls_name in results:
            r = results[cls_name]
            ap = r["AP"] * 100
            num_gt = r["num_gt"]
            num_pred = r["num_pred"]
            recall = r.get("max_recall", 0.0) * 100
            lines.append(
                f"{cls_name:<20} {ap:>7.2f}% {num_gt:>8d} {num_pred:>8d} {recall:>7.2f}%"
            )

    lines.append("-" * 60)
    lines.append(f"{'mAP':<20} {results['mAP'] * 100:>7.2f}%")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """Main evaluation entry point (standalone usage)."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate PointNet++ detections")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction .txt files")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth .txt files")
    parser.add_argument("--classes", type=str, nargs="+",
                        default=["Car", "Pedestrian", "Cyclist"])
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for report")
    args = parser.parse_args()

    from pathlib import Path

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    # Load predictions and ground truths
    pred_files = sorted(pred_dir.glob("*.txt"))

    predictions = []
    ground_truths = []

    class_to_idx = {cls: i + 1 for i, cls in enumerate(args.classes)}

    for pred_file in pred_files:
        sample_id = pred_file.stem
        gt_file = gt_dir / f"{sample_id}.txt"

        # Parse prediction file: class_name score x y z w l h yaw
        pred_boxes = []
        pred_scores = []
        pred_labels = []

        with open(pred_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls_name = parts[0]
                if cls_name not in class_to_idx:
                    continue
                score = float(parts[1])
                box = [float(x) for x in parts[2:9]]
                pred_boxes.append(box)
                pred_scores.append(score)
                pred_labels.append(class_to_idx[cls_name])

        predictions.append({
            "boxes": np.array(pred_boxes, dtype=np.float32).reshape(-1, 7),
            "scores": np.array(pred_scores, dtype=np.float32),
            "labels": np.array(pred_labels, dtype=np.int64),
        })

        # Parse GT file: class_name x y z w l h yaw
        gt_boxes = []
        gt_labels = []

        if gt_file.exists():
            with open(gt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 8:
                        continue
                    cls_name = parts[0]
                    if cls_name not in class_to_idx:
                        continue
                    box = [float(x) for x in parts[1:8]]
                    gt_boxes.append(box)
                    gt_labels.append(class_to_idx[cls_name])

        ground_truths.append({
            "boxes": np.array(gt_boxes, dtype=np.float32).reshape(-1, 7),
            "labels": np.array(gt_labels, dtype=np.int64),
        })

    # Evaluate
    results = evaluate_detections(predictions, ground_truths, args.classes)
    report = generate_evaluation_report(results, args.classes)

    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
