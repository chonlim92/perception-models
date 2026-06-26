"""
PointPillars Evaluation Script for KITTI and nuScenes Benchmarks.

This module provides comprehensive evaluation utilities for 3D object detection
models using both KITTI-style (IoU-based) and nuScenes-style (distance-based)
metrics. It includes oriented 3D/BEV IoU computation, NMS, and full evaluation
pipelines with FPS measurement.

Usage:
    python evaluate.py --config config.yaml --checkpoint model.pth --dataset kitti
"""

import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =============================================================================
# IoU Computation Utilities
# =============================================================================


def rotation_matrix_z(angle: torch.Tensor) -> torch.Tensor:
    """Create 2D rotation matrices for given angles (rotation around Z axis).

    Args:
        angle: Tensor of shape (N,) with rotation angles in radians.

    Returns:
        Rotation matrices of shape (N, 2, 2).
    """
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rot = torch.zeros(angle.shape[0], 2, 2, device=angle.device, dtype=angle.dtype)
    rot[:, 0, 0] = cos_a
    rot[:, 0, 1] = -sin_a
    rot[:, 1, 0] = sin_a
    rot[:, 1, 1] = cos_a
    return rot


def get_corners_2d(boxes: torch.Tensor) -> torch.Tensor:
    """Compute 2D BEV corners of oriented bounding boxes.

    Args:
        boxes: Tensor of shape (N, 7) with columns [x, y, z, w, l, h, yaw].
               w = width (along x), l = length (along y), h = height (along z).

    Returns:
        Corners of shape (N, 4, 2) in BEV (x-y plane).
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 3]
    l = boxes[:, 4]
    yaw = boxes[:, 6]

    half_w = w / 2.0
    half_l = l / 2.0

    corners_local = torch.zeros(boxes.shape[0], 4, 2, device=boxes.device, dtype=boxes.dtype)
    corners_local[:, 0, 0] = -half_w
    corners_local[:, 0, 1] = -half_l
    corners_local[:, 1, 0] = half_w
    corners_local[:, 1, 1] = -half_l
    corners_local[:, 2, 0] = half_w
    corners_local[:, 2, 1] = half_l
    corners_local[:, 3, 0] = -half_w
    corners_local[:, 3, 1] = half_l

    rot = rotation_matrix_z(yaw)
    corners_rotated = torch.bmm(corners_local, rot.transpose(1, 2))

    center = torch.stack([x, y], dim=1).unsqueeze(1)
    corners_world = corners_rotated + center

    return corners_world


def polygon_area(vertices: torch.Tensor) -> torch.Tensor:
    """Compute area of convex polygons using the shoelace formula.

    Args:
        vertices: Tensor of shape (N, M, 2) representing polygon vertices in order.

    Returns:
        Areas of shape (N,).
    """
    n_verts = vertices.shape[1]
    area = torch.zeros(vertices.shape[0], device=vertices.device, dtype=vertices.dtype)
    for i in range(n_verts):
        j = (i + 1) % n_verts
        area += vertices[:, i, 0] * vertices[:, j, 1]
        area -= vertices[:, j, 0] * vertices[:, i, 1]
    return torch.abs(area) / 2.0


def sutherland_hodgman_clip(
    subject_polygon: np.ndarray, clip_polygon: np.ndarray
) -> np.ndarray:
    """Clip a polygon against another polygon using Sutherland-Hodgman algorithm.

    Args:
        subject_polygon: Array of shape (M, 2) - the polygon to be clipped.
        clip_polygon: Array of shape (K, 2) - the clipping polygon.

    Returns:
        Clipped polygon vertices as array of shape (P, 2), may be empty.
    """
    output_list = list(subject_polygon)

    if len(output_list) == 0:
        return np.array([]).reshape(0, 2)

    for i in range(len(clip_polygon)):
        if len(output_list) == 0:
            return np.array([]).reshape(0, 2)

        input_list = output_list
        output_list = []

        edge_start = clip_polygon[i]
        edge_end = clip_polygon[(i + 1) % len(clip_polygon)]
        edge_vec = edge_end - edge_start

        for j in range(len(input_list)):
            current = input_list[j]
            previous = input_list[j - 1]

            curr_inside = np.cross(edge_vec, current - edge_start) >= 0
            prev_inside = np.cross(edge_vec, previous - edge_start) >= 0

            if curr_inside:
                if not prev_inside:
                    intersection = _line_intersection(previous, current, edge_start, edge_end)
                    if intersection is not None:
                        output_list.append(intersection)
                output_list.append(current)
            elif prev_inside:
                intersection = _line_intersection(previous, current, edge_start, edge_end)
                if intersection is not None:
                    output_list.append(intersection)

    if len(output_list) == 0:
        return np.array([]).reshape(0, 2)
    return np.array(output_list)


def _line_intersection(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> Optional[np.ndarray]:
    """Compute the intersection point of two line segments.

    Args:
        p1, p2: Endpoints of the first segment.
        p3, p4: Endpoints of the second segment.

    Returns:
        Intersection point as array of shape (2,), or None if parallel.
    """
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


def compute_bev_iou_single(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute BEV IoU between two oriented bounding boxes.

    Args:
        box_a: Array of shape (7,) [x, y, z, w, l, h, yaw].
        box_b: Array of shape (7,) [x, y, z, w, l, h, yaw].

    Returns:
        BEV IoU value between 0 and 1.
    """
    corners_a = _get_single_corners_2d(box_a)
    corners_b = _get_single_corners_2d(box_b)

    intersection_poly = sutherland_hodgman_clip(corners_a, corners_b)

    if intersection_poly.shape[0] < 3:
        return 0.0

    inter_area = _polygon_area_np(intersection_poly)
    area_a = box_a[3] * box_a[4]
    area_b = box_b[3] * box_b[4]
    union_area = area_a + area_b - inter_area

    if union_area < 1e-10:
        return 0.0

    return float(inter_area / union_area)


def _get_single_corners_2d(box: np.ndarray) -> np.ndarray:
    """Get 2D BEV corners for a single box.

    Args:
        box: Array of shape (7,) [x, y, z, w, l, h, yaw].

    Returns:
        Corners of shape (4, 2).
    """
    x, y = box[0], box[1]
    w, l = box[3], box[4]
    yaw = box[6]

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    half_w = w / 2.0
    half_l = l / 2.0

    corners_local = np.array([
        [-half_w, -half_l],
        [half_w, -half_l],
        [half_w, half_l],
        [-half_w, half_l],
    ])

    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    corners_world = corners_local @ rot.T + np.array([x, y])
    return corners_world


def _polygon_area_np(vertices: np.ndarray) -> float:
    """Compute polygon area using the shoelace formula (numpy).

    Args:
        vertices: Array of shape (N, 2) polygon vertices in order.

    Returns:
        Polygon area.
    """
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def compute_3d_iou_single(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute 3D IoU between two oriented bounding boxes.

    Uses BEV intersection polygon and height overlap for volumetric IoU.

    Args:
        box_a: Array of shape (7,) [x, y, z, w, l, h, yaw]. z is center height.
        box_b: Array of shape (7,) [x, y, z, w, l, h, yaw]. z is center height.

    Returns:
        3D IoU value between 0 and 1.
    """
    corners_a = _get_single_corners_2d(box_a)
    corners_b = _get_single_corners_2d(box_b)

    intersection_poly = sutherland_hodgman_clip(corners_a, corners_b)
    if intersection_poly.shape[0] < 3:
        return 0.0

    inter_area_2d = _polygon_area_np(intersection_poly)

    z_a_min = box_a[2] - box_a[5] / 2.0
    z_a_max = box_a[2] + box_a[5] / 2.0
    z_b_min = box_b[2] - box_b[5] / 2.0
    z_b_max = box_b[2] + box_b[5] / 2.0

    z_overlap = max(0.0, min(z_a_max, z_b_max) - max(z_a_min, z_b_min))

    inter_volume = inter_area_2d * z_overlap

    vol_a = box_a[3] * box_a[4] * box_a[5]
    vol_b = box_b[3] * box_b[4] * box_b[5]
    union_volume = vol_a + vol_b - inter_volume

    if union_volume < 1e-10:
        return 0.0

    return float(inter_volume / union_volume)


def compute_iou_matrix(
    boxes_a: np.ndarray, boxes_b: np.ndarray, mode: str = "3d"
) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of boxes.

    Args:
        boxes_a: Array of shape (N, 7) [x, y, z, w, l, h, yaw].
        boxes_b: Array of shape (M, 7) [x, y, z, w, l, h, yaw].
        mode: Either '3d' for volumetric IoU or 'bev' for bird's-eye view IoU.

    Returns:
        IoU matrix of shape (N, M).
    """
    n = boxes_a.shape[0]
    m = boxes_b.shape[0]
    iou_matrix = np.zeros((n, m), dtype=np.float64)

    iou_fn = compute_3d_iou_single if mode == "3d" else compute_bev_iou_single

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = iou_fn(boxes_a[i], boxes_b[j])

    return iou_matrix


# =============================================================================
# KITTI Evaluation Metrics
# =============================================================================


KITTI_DIFFICULTY_PARAMS = {
    "easy": {"min_height": 40.0, "max_occlusion": 0, "max_truncation": 0.15},
    "moderate": {"min_height": 25.0, "max_occlusion": 1, "max_truncation": 0.30},
    "hard": {"min_height": 25.0, "max_occlusion": 2, "max_truncation": 0.50},
}


def filter_by_difficulty(
    annotations: Dict,
    difficulty: str,
) -> np.ndarray:
    """Filter annotations based on KITTI difficulty level.

    Args:
        annotations: Dictionary with keys 'bbox_height', 'occlusion', 'truncation'
                     each being arrays of shape (N,).
        difficulty: One of 'easy', 'moderate', 'hard'.

    Returns:
        Boolean mask array of shape (N,) indicating valid annotations.
    """
    params = KITTI_DIFFICULTY_PARAMS[difficulty]
    bbox_height = annotations["bbox_height"]
    occlusion = annotations["occlusion"]
    truncation = annotations["truncation"]

    mask = (
        (bbox_height >= params["min_height"])
        & (occlusion <= params["max_occlusion"])
        & (truncation <= params["max_truncation"])
    )
    return mask


def compute_ap_r40(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute Average Precision using 40-point interpolation (R40).

    The R40 method samples recall at 40 evenly-spaced points and takes the
    maximum precision at or above each recall threshold.

    Args:
        precision: Array of precision values sorted by decreasing confidence.
        recall: Array of recall values sorted by decreasing confidence.

    Returns:
        AP value computed with 40-point interpolation.
    """
    recall_thresholds = np.linspace(0.0, 1.0, 40)
    ap = 0.0

    for t in recall_thresholds:
        prec_at_recall = precision[recall >= t]
        if prec_at_recall.size > 0:
            ap += np.max(prec_at_recall)

    ap /= 40.0
    return float(ap)


def compute_kitti_metrics(
    gt_annos: List[Dict],
    pred_annos: List[Dict],
    class_names: List[str],
    iou_thresholds: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Compute KITTI-style 3D AP metrics for all classes and difficulty levels.

    Args:
        gt_annos: List of ground truth annotation dicts per frame. Each dict has:
            - 'name': np.ndarray of shape (N,) with class names
            - 'boxes_3d': np.ndarray of shape (N, 7) [x, y, z, w, l, h, yaw]
            - 'bbox_height': np.ndarray of shape (N,) 2D bbox height in pixels
            - 'occlusion': np.ndarray of shape (N,) occlusion level (0, 1, 2, 3)
            - 'truncation': np.ndarray of shape (N,) truncation ratio [0, 1]
            - 'score': np.ndarray of shape (N,) (optional, for predictions)
        pred_annos: List of prediction annotation dicts per frame. Each dict has:
            - 'name': np.ndarray of shape (M,) with class names
            - 'boxes_3d': np.ndarray of shape (M, 7) [x, y, z, w, l, h, yaw]
            - 'score': np.ndarray of shape (M,) confidence scores
        class_names: List of class names to evaluate (e.g., ['Car', 'Pedestrian', 'Cyclist']).
        iou_thresholds: Dict mapping class name to IoU threshold
                        (e.g., {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}).

    Returns:
        Nested dict: results[class_name][difficulty] = AP value.
    """
    results: Dict[str, Dict[str, float]] = {}
    difficulties = ["easy", "moderate", "hard"]

    for cls_name in class_names:
        results[cls_name] = {}
        iou_thresh = iou_thresholds[cls_name]

        for difficulty in difficulties:
            all_scores: List[float] = []
            all_tp_fp: List[int] = []
            total_gt = 0

            for frame_idx in range(len(gt_annos)):
                gt = gt_annos[frame_idx]
                pred = pred_annos[frame_idx]

                gt_cls_mask = gt["name"] == cls_name
                gt_boxes_cls = gt["boxes_3d"][gt_cls_mask]

                gt_frame_annotations = {
                    "bbox_height": gt["bbox_height"][gt_cls_mask],
                    "occlusion": gt["occlusion"][gt_cls_mask],
                    "truncation": gt["truncation"][gt_cls_mask],
                }
                diff_mask = filter_by_difficulty(gt_frame_annotations, difficulty)
                gt_boxes_valid = gt_boxes_cls[diff_mask]
                num_gt = gt_boxes_valid.shape[0]
                total_gt += num_gt

                pred_cls_mask = pred["name"] == cls_name
                pred_boxes_cls = pred["boxes_3d"][pred_cls_mask]
                pred_scores_cls = pred["score"][pred_cls_mask]

                if pred_boxes_cls.shape[0] == 0:
                    continue

                if num_gt == 0:
                    for s in pred_scores_cls:
                        all_scores.append(float(s))
                        all_tp_fp.append(0)
                    continue

                iou_matrix = compute_iou_matrix(pred_boxes_cls, gt_boxes_valid, mode="3d")

                sort_indices = np.argsort(-pred_scores_cls)
                gt_matched = np.zeros(num_gt, dtype=bool)

                for pred_idx in sort_indices:
                    all_scores.append(float(pred_scores_cls[pred_idx]))

                    ious_for_pred = iou_matrix[pred_idx]
                    best_gt_idx = np.argmax(ious_for_pred)
                    best_iou = ious_for_pred[best_gt_idx]

                    if best_iou >= iou_thresh and not gt_matched[best_gt_idx]:
                        all_tp_fp.append(1)
                        gt_matched[best_gt_idx] = True
                    else:
                        all_tp_fp.append(0)

            if total_gt == 0:
                results[cls_name][difficulty] = 0.0
                continue

            sorted_indices = np.argsort(-np.array(all_scores))
            tp_fp_sorted = np.array(all_tp_fp)[sorted_indices]

            tp_cumsum = np.cumsum(tp_fp_sorted)
            fp_cumsum = np.cumsum(1 - tp_fp_sorted)

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / float(total_gt)

            ap = compute_ap_r40(precision, recall)
            results[cls_name][difficulty] = ap

    return results


# =============================================================================
# nuScenes Evaluation Metrics
# =============================================================================


NUSCENES_DISTANCE_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]


def compute_center_distance(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Compute pairwise center distance matrix between boxes.

    Args:
        boxes_a: Array of shape (N, 7) [x, y, z, w, l, h, yaw].
        boxes_b: Array of shape (M, 7) [x, y, z, w, l, h, yaw].

    Returns:
        Distance matrix of shape (N, M).
    """
    centers_a = boxes_a[:, :2]
    centers_b = boxes_b[:, :2]

    diff = centers_a[:, np.newaxis, :] - centers_b[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist_matrix


def compute_ate(matched_pred: np.ndarray, matched_gt: np.ndarray) -> float:
    """Compute Average Translation Error (ATE).

    Args:
        matched_pred: Array of shape (K, 7) matched prediction boxes.
        matched_gt: Array of shape (K, 7) matched ground truth boxes.

    Returns:
        Mean Euclidean center distance (2D) in meters.
    """
    if matched_pred.shape[0] == 0:
        return 1.0
    diffs = matched_pred[:, :2] - matched_gt[:, :2]
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.mean(distances))


def compute_ase(matched_pred: np.ndarray, matched_gt: np.ndarray) -> float:
    """Compute Average Scale Error (ASE).

    Measures 1 - IoU_3D for matched pairs (only scale, not translation/rotation).
    Approximated here as 1 - volume_intersection / volume_union using axis-aligned
    dimension comparison.

    Args:
        matched_pred: Array of shape (K, 7) matched prediction boxes.
        matched_gt: Array of shape (K, 7) matched ground truth boxes.

    Returns:
        Mean scale error in [0, 1].
    """
    if matched_pred.shape[0] == 0:
        return 1.0

    pred_dims = matched_pred[:, 3:6]
    gt_dims = matched_gt[:, 3:6]

    pred_vol = np.prod(pred_dims, axis=1)
    gt_vol = np.prod(gt_dims, axis=1)

    min_dims = np.minimum(pred_dims, gt_dims)
    intersection_vol = np.prod(min_dims, axis=1)

    union_vol = pred_vol + gt_vol - intersection_vol
    iou_approx = np.where(union_vol > 1e-10, intersection_vol / union_vol, 0.0)

    ase = 1.0 - iou_approx
    return float(np.mean(ase))


def compute_aoe(matched_pred: np.ndarray, matched_gt: np.ndarray) -> float:
    """Compute Average Orientation Error (AOE).

    Smallest angle difference between predicted and ground truth yaw.

    Args:
        matched_pred: Array of shape (K, 7) matched prediction boxes.
        matched_gt: Array of shape (K, 7) matched ground truth boxes.

    Returns:
        Mean absolute orientation error in radians, capped at pi.
    """
    if matched_pred.shape[0] == 0:
        return np.pi

    pred_yaw = matched_pred[:, 6]
    gt_yaw = matched_gt[:, 6]

    diff = pred_yaw - gt_yaw
    diff = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    return float(np.mean(diff))


def compute_ave(
    matched_pred_vels: np.ndarray, matched_gt_vels: np.ndarray
) -> float:
    """Compute Average Velocity Error (AVE).

    Args:
        matched_pred_vels: Array of shape (K, 2) predicted velocities [vx, vy].
        matched_gt_vels: Array of shape (K, 2) ground truth velocities [vx, vy].

    Returns:
        Mean L2 velocity error in m/s.
    """
    if matched_pred_vels.shape[0] == 0:
        return 1.0

    diff = matched_pred_vels - matched_gt_vels
    errors = np.sqrt(np.sum(diff ** 2, axis=1))
    return float(np.mean(errors))


def compute_aae(
    matched_pred_attrs: np.ndarray, matched_gt_attrs: np.ndarray
) -> float:
    """Compute Average Attribute Error (AAE).

    Args:
        matched_pred_attrs: Array of shape (K,) predicted attribute indices.
        matched_gt_attrs: Array of shape (K,) ground truth attribute indices.

    Returns:
        Mean attribute classification error (1 - accuracy).
    """
    if matched_pred_attrs.shape[0] == 0:
        return 1.0

    correct = (matched_pred_attrs == matched_gt_attrs).astype(np.float64)
    accuracy = np.mean(correct)
    return float(1.0 - accuracy)


def compute_nuscenes_ap_single(
    gt_annos: List[Dict],
    pred_annos: List[Dict],
    cls_name: str,
    dist_thresh: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute AP and collect TP matches for a single class and distance threshold.

    Args:
        gt_annos: List of ground truth annotation dicts per frame.
        pred_annos: List of prediction annotation dicts per frame.
        cls_name: Class name to evaluate.
        dist_thresh: Maximum center distance for a match.

    Returns:
        Tuple of (AP, matched_pred_boxes, matched_gt_boxes, matched_indices_info).
        matched_pred_boxes and matched_gt_boxes are concatenated matched pairs.
    """
    all_scores: List[float] = []
    all_tp_fp: List[int] = []
    total_gt = 0
    matched_preds_list: List[np.ndarray] = []
    matched_gts_list: List[np.ndarray] = []

    for frame_idx in range(len(gt_annos)):
        gt = gt_annos[frame_idx]
        pred = pred_annos[frame_idx]

        gt_cls_mask = gt["name"] == cls_name
        gt_boxes_cls = gt["boxes_3d"][gt_cls_mask]
        num_gt = gt_boxes_cls.shape[0]
        total_gt += num_gt

        pred_cls_mask = pred["name"] == cls_name
        pred_boxes_cls = pred["boxes_3d"][pred_cls_mask]
        pred_scores_cls = pred["score"][pred_cls_mask]

        if pred_boxes_cls.shape[0] == 0:
            continue

        if num_gt == 0:
            for s in pred_scores_cls:
                all_scores.append(float(s))
                all_tp_fp.append(0)
            continue

        dist_matrix = compute_center_distance(pred_boxes_cls, gt_boxes_cls)

        sort_indices = np.argsort(-pred_scores_cls)
        gt_matched = np.zeros(num_gt, dtype=bool)

        for pred_idx in sort_indices:
            all_scores.append(float(pred_scores_cls[pred_idx]))
            dists_for_pred = dist_matrix[pred_idx]
            best_gt_idx = np.argmin(dists_for_pred)
            best_dist = dists_for_pred[best_gt_idx]

            if best_dist <= dist_thresh and not gt_matched[best_gt_idx]:
                all_tp_fp.append(1)
                gt_matched[best_gt_idx] = True
                matched_preds_list.append(pred_boxes_cls[pred_idx:pred_idx + 1])
                matched_gts_list.append(gt_boxes_cls[best_gt_idx:best_gt_idx + 1])
            else:
                all_tp_fp.append(0)

    if total_gt == 0:
        empty = np.zeros((0, 7), dtype=np.float64)
        return 0.0, empty, empty, np.array([])

    if len(all_scores) == 0:
        empty = np.zeros((0, 7), dtype=np.float64)
        return 0.0, empty, empty, np.array([])

    sorted_indices = np.argsort(-np.array(all_scores))
    tp_fp_sorted = np.array(all_tp_fp)[sorted_indices]

    tp_cumsum = np.cumsum(tp_fp_sorted)
    fp_cumsum = np.cumsum(1 - tp_fp_sorted)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / float(total_gt)

    ap = compute_ap_r40(precision, recall)

    if matched_preds_list:
        matched_preds = np.concatenate(matched_preds_list, axis=0)
        matched_gts = np.concatenate(matched_gts_list, axis=0)
    else:
        matched_preds = np.zeros((0, 7), dtype=np.float64)
        matched_gts = np.zeros((0, 7), dtype=np.float64)

    return ap, matched_preds, matched_gts, np.array([])


def compute_nuscenes_metrics(
    gt_annos: List[Dict],
    pred_annos: List[Dict],
    class_names: List[str],
) -> Dict[str, float]:
    """Compute nuScenes-style detection metrics (mAP + NDS).

    Args:
        gt_annos: List of ground truth annotation dicts per frame. Each dict has:
            - 'name': np.ndarray of shape (N,) with class names
            - 'boxes_3d': np.ndarray of shape (N, 7) [x, y, z, w, l, h, yaw]
            - 'velocity': np.ndarray of shape (N, 2) [vx, vy] (optional)
            - 'attribute': np.ndarray of shape (N,) attribute indices (optional)
        pred_annos: List of prediction annotation dicts per frame. Each dict has:
            - 'name': np.ndarray of shape (M,) with class names
            - 'boxes_3d': np.ndarray of shape (M, 7) [x, y, z, w, l, h, yaw]
            - 'score': np.ndarray of shape (M,) confidence scores
            - 'velocity': np.ndarray of shape (M, 2) [vx, vy] (optional)
            - 'attribute': np.ndarray of shape (M,) attribute indices (optional)
        class_names: List of class names to evaluate.

    Returns:
        Dictionary with keys:
            - 'mAP': mean Average Precision across all classes and distance thresholds
            - 'NDS': nuScenes Detection Score
            - 'mATE': mean Average Translation Error
            - 'mASE': mean Average Scale Error
            - 'mAOE': mean Average Orientation Error
            - 'mAVE': mean Average Velocity Error
            - 'mAAE': mean Average Attribute Error
            - per-class AP values: '{class_name}_AP'
    """
    all_aps: List[float] = []
    all_ate: List[float] = []
    all_ase: List[float] = []
    all_aoe: List[float] = []
    all_ave: List[float] = []
    all_aae: List[float] = []

    results: Dict[str, float] = {}

    for cls_name in class_names:
        cls_aps: List[float] = []
        cls_matched_preds: List[np.ndarray] = []
        cls_matched_gts: List[np.ndarray] = []

        for dist_thresh in NUSCENES_DISTANCE_THRESHOLDS:
            ap, matched_preds, matched_gts, _ = compute_nuscenes_ap_single(
                gt_annos, pred_annos, cls_name, dist_thresh
            )
            cls_aps.append(ap)
            if dist_thresh == 2.0:
                cls_matched_preds.append(matched_preds)
                cls_matched_gts.append(matched_gts)

        cls_mean_ap = float(np.mean(cls_aps))
        all_aps.append(cls_mean_ap)
        results[f"{cls_name}_AP"] = cls_mean_ap

        if cls_matched_preds and cls_matched_preds[0].shape[0] > 0:
            mp = np.concatenate(cls_matched_preds, axis=0)
            mg = np.concatenate(cls_matched_gts, axis=0)
        else:
            mp = np.zeros((0, 7), dtype=np.float64)
            mg = np.zeros((0, 7), dtype=np.float64)

        ate = compute_ate(mp, mg)
        ase = compute_ase(mp, mg)
        aoe = compute_aoe(mp, mg)
        all_ate.append(ate)
        all_ase.append(ase)
        all_aoe.append(aoe)

        has_velocity = (
            "velocity" in gt_annos[0] and "velocity" in pred_annos[0]
        ) if len(gt_annos) > 0 else False

        if has_velocity and mp.shape[0] > 0:
            pred_vels_list: List[np.ndarray] = []
            gt_vels_list: List[np.ndarray] = []
            for frame_idx in range(len(gt_annos)):
                gt = gt_annos[frame_idx]
                pred = pred_annos[frame_idx]
                gt_cls_mask = gt["name"] == cls_name
                pred_cls_mask = pred["name"] == cls_name
                if "velocity" in gt and "velocity" in pred:
                    gt_vels_frame = gt["velocity"][gt_cls_mask]
                    pred_vels_frame = pred["velocity"][pred_cls_mask]
                    if pred_vels_frame.shape[0] > 0 and gt_vels_frame.shape[0] > 0:
                        min_len = min(pred_vels_frame.shape[0], gt_vels_frame.shape[0])
                        pred_vels_list.append(pred_vels_frame[:min_len])
                        gt_vels_list.append(gt_vels_frame[:min_len])

            if pred_vels_list:
                pred_vels = np.concatenate(pred_vels_list, axis=0)
                gt_vels = np.concatenate(gt_vels_list, axis=0)
                ave = compute_ave(pred_vels, gt_vels)
            else:
                ave = 1.0
        else:
            ave = 1.0
        all_ave.append(ave)

        has_attribute = (
            "attribute" in gt_annos[0] and "attribute" in pred_annos[0]
        ) if len(gt_annos) > 0 else False

        if has_attribute and mp.shape[0] > 0:
            pred_attrs_list: List[np.ndarray] = []
            gt_attrs_list: List[np.ndarray] = []
            for frame_idx in range(len(gt_annos)):
                gt = gt_annos[frame_idx]
                pred = pred_annos[frame_idx]
                gt_cls_mask = gt["name"] == cls_name
                pred_cls_mask = pred["name"] == cls_name
                if "attribute" in gt and "attribute" in pred:
                    gt_attrs_frame = gt["attribute"][gt_cls_mask]
                    pred_attrs_frame = pred["attribute"][pred_cls_mask]
                    if pred_attrs_frame.shape[0] > 0 and gt_attrs_frame.shape[0] > 0:
                        min_len = min(pred_attrs_frame.shape[0], gt_attrs_frame.shape[0])
                        pred_attrs_list.append(pred_attrs_frame[:min_len])
                        gt_attrs_list.append(gt_attrs_frame[:min_len])

            if pred_attrs_list:
                pred_attrs = np.concatenate(pred_attrs_list, axis=0)
                gt_attrs = np.concatenate(gt_attrs_list, axis=0)
                aae = compute_aae(pred_attrs, gt_attrs)
            else:
                aae = 1.0
        else:
            aae = 1.0
        all_aae.append(aae)

    mAP = float(np.mean(all_aps)) if all_aps else 0.0
    mATE = float(np.mean(all_ate)) if all_ate else 1.0
    mASE = float(np.mean(all_ase)) if all_ase else 1.0
    mAOE = float(np.mean(all_aoe)) if all_aoe else 1.0
    mAVE = float(np.mean(all_ave)) if all_ave else 1.0
    mAAE = float(np.mean(all_aae)) if all_aae else 1.0

    tp_errors = {
        "mATE": min(mATE, 1.0),
        "mASE": min(mASE, 1.0),
        "mAOE": min(mAOE, 1.0),
        "mAVE": min(mAVE, 1.0),
        "mAAE": min(mAAE, 1.0),
    }
    mean_tp_error = float(np.mean(list(tp_errors.values())))
    nds = (1.0 / 10.0) * (5.0 * mAP + 5.0 * (1.0 - mean_tp_error))

    results["mAP"] = mAP
    results["NDS"] = nds
    results["mATE"] = mATE
    results["mASE"] = mASE
    results["mAOE"] = mAOE
    results["mAVE"] = mAVE
    results["mAAE"] = mAAE

    return results


# =============================================================================
# NMS and Post-Processing
# =============================================================================


def nms_bev(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Non-maximum suppression in BEV for oriented 3D bounding boxes.

    Args:
        boxes: Array of shape (N, 7) [x, y, z, w, l, h, yaw].
        scores: Array of shape (N,) confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Array of kept indices after NMS.
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    keep: List[int] = []

    suppressed = np.zeros(boxes.shape[0], dtype=bool)

    for i in range(len(order)):
        idx = order[i]
        if suppressed[idx]:
            continue
        keep.append(idx)

        for j in range(i + 1, len(order)):
            jdx = order[j]
            if suppressed[jdx]:
                continue

            iou = compute_bev_iou_single(boxes[idx], boxes[jdx])
            if iou >= iou_threshold:
                suppressed[jdx] = True

    return np.array(keep, dtype=np.int64)


def apply_nms_and_filter(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    nms_iou_threshold: float = 0.1,
    score_threshold: float = 0.1,
    max_detections: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply score filtering and class-wise NMS to raw predictions.

    Args:
        boxes: Array of shape (N, 7) predicted boxes.
        scores: Array of shape (N,) predicted scores.
        labels: Array of shape (N,) predicted class indices.
        nms_iou_threshold: IoU threshold for NMS.
        score_threshold: Minimum score to keep.
        max_detections: Maximum number of detections to return.

    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_labels).
    """
    score_mask = scores >= score_threshold
    boxes = boxes[score_mask]
    scores = scores[score_mask]
    labels = labels[score_mask]

    if boxes.shape[0] == 0:
        return (
            np.zeros((0, 7), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )

    unique_labels = np.unique(labels)
    keep_indices: List[int] = []
    global_indices = np.arange(boxes.shape[0])

    for cls_id in unique_labels:
        cls_mask = labels == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_global_indices = global_indices[cls_mask]

        nms_keep = nms_bev(cls_boxes, cls_scores, nms_iou_threshold)
        keep_indices.extend(cls_global_indices[nms_keep].tolist())

    keep_indices = np.array(keep_indices, dtype=np.int64)
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    if boxes.shape[0] > max_detections:
        top_k = np.argsort(-scores)[:max_detections]
        boxes = boxes[top_k]
        scores = scores[top_k]
        labels = labels[top_k]

    return boxes, scores, labels


# =============================================================================
# Model Loading and Inference
# =============================================================================


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: Dict) -> torch.nn.Module:
    """Build the PointPillars model from configuration.

    Attempts to import a local model module. Falls back to constructing
    a minimal model definition if the module is not available.

    Args:
        config: Configuration dictionary with 'model' key.

    Returns:
        PyTorch model (not yet loaded with weights).
    """
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "PointPillars")

    try:
        from model import PointPillars as ModelClass
        model = ModelClass(**model_cfg.get("params", {}))
    except ImportError:
        try:
            from pointpillars_model import PointPillars as ModelClass
            model = ModelClass(**model_cfg.get("params", {}))
        except ImportError:
            raise ImportError(
                f"Cannot import model '{model_name}'. Ensure model.py or "
                f"pointpillars_model.py is in the module path."
            )

    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model weights from a checkpoint file.

    Args:
        model: The model to load weights into.
        checkpoint_path: Path to the .pth or .pt checkpoint.
        device: Device to load the checkpoint onto.

    Returns:
        Model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    return model


def build_dataloader(config: Dict, batch_size: int, num_workers: int) -> DataLoader:
    """Build the validation dataloader from configuration.

    Args:
        config: Configuration dictionary with 'dataset' key.
        batch_size: Batch size for evaluation.
        num_workers: Number of data loading workers.

    Returns:
        PyTorch DataLoader for the validation set.
    """
    dataset_cfg = config.get("dataset", {})
    dataset_type = dataset_cfg.get("type", "kitti")
    data_root = dataset_cfg.get("data_root", "./data")
    val_split = dataset_cfg.get("val_split", "val")

    try:
        if dataset_type == "kitti":
            from dataset import KITTIDataset as DatasetClass
        else:
            from dataset import NuScenesDataset as DatasetClass

        dataset = DatasetClass(
            root=data_root,
            split=val_split,
            **dataset_cfg.get("params", {}),
        )
    except ImportError:
        try:
            if dataset_type == "kitti":
                from kitti_dataset import KITTIDataset as DatasetClass
            else:
                from nuscenes_dataset import NuScenesDataset as DatasetClass

            dataset = DatasetClass(
                root=data_root,
                split=val_split,
                **dataset_cfg.get("params", {}),
            )
        except ImportError:
            raise ImportError(
                f"Cannot import dataset for type '{dataset_type}'. "
                f"Ensure the appropriate dataset module is available."
            )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=getattr(dataset, "collate_fn", None),
    )
    return dataloader


def decode_predictions(
    output: Dict[str, torch.Tensor],
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw model output into boxes, scores, and labels.

    Handles common output formats from PointPillars-style models.

    Args:
        output: Dictionary of model outputs. Expected keys vary by model but
                commonly include 'box_preds', 'cls_preds', 'dir_cls_preds'.
        config: Configuration dictionary for decoding parameters.

    Returns:
        Tuple of (boxes, scores, labels) as numpy arrays.
        boxes: shape (N, 7), scores: shape (N,), labels: shape (N,).
    """
    if "boxes" in output and "scores" in output and "labels" in output:
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        return boxes, scores, labels

    if "box_preds" in output and "cls_preds" in output:
        box_preds = output["box_preds"].detach().cpu()
        cls_preds = output["cls_preds"].detach().cpu()

        if cls_preds.dim() == 3:
            cls_preds = cls_preds.view(-1, cls_preds.shape[-1])
            box_preds = box_preds.view(-1, box_preds.shape[-1])

        cls_scores = torch.sigmoid(cls_preds)
        max_scores, max_labels = cls_scores.max(dim=-1)

        boxes = box_preds.numpy()
        scores = max_scores.numpy()
        labels = max_labels.numpy()
        return boxes, scores, labels

    if "pred_boxes" in output:
        boxes = output["pred_boxes"].detach().cpu().numpy()
        scores = output["pred_scores"].detach().cpu().numpy()
        labels = output["pred_labels"].detach().cpu().numpy()
        return boxes, scores, labels

    raise ValueError(
        f"Unrecognized model output format. Available keys: {list(output.keys())}"
    )


# =============================================================================
# Results Formatting
# =============================================================================


def print_kitti_results(results: Dict[str, Dict[str, float]]) -> None:
    """Print KITTI evaluation results in a formatted table.

    Args:
        results: Nested dict from compute_kitti_metrics.
    """
    header = f"{'Class':<15} {'Easy':>10} {'Moderate':>10} {'Hard':>10}"
    separator = "-" * 50
    print("\n" + separator)
    print("KITTI 3D AP (R40) Results")
    print(separator)
    print(header)
    print(separator)

    map_values: List[float] = []
    for cls_name, difficulties in results.items():
        easy = difficulties.get("easy", 0.0) * 100
        moderate = difficulties.get("moderate", 0.0) * 100
        hard = difficulties.get("hard", 0.0) * 100
        print(f"{cls_name:<15} {easy:>9.2f}% {moderate:>9.2f}% {hard:>9.2f}%")
        map_values.append(moderate)

    print(separator)
    overall_map = float(np.mean(map_values)) if map_values else 0.0
    print(f"{'mAP (Mod.)':<15} {overall_map:>9.2f}%")
    print(separator + "\n")


def print_nuscenes_results(results: Dict[str, float], class_names: List[str]) -> None:
    """Print nuScenes evaluation results in a formatted table.

    Args:
        results: Dictionary from compute_nuscenes_metrics.
        class_names: List of class names evaluated.
    """
    separator = "-" * 55
    print("\n" + separator)
    print("nuScenes Detection Metrics")
    print(separator)

    print(f"{'Metric':<15} {'Value':>10}")
    print(separator)
    print(f"{'mAP':<15} {results['mAP'] * 100:>9.2f}%")
    print(f"{'NDS':<15} {results['NDS'] * 100:>9.2f}%")
    print(separator)
    print(f"{'mATE (m)':<15} {results['mATE']:>10.4f}")
    print(f"{'mASE':<15} {results['mASE']:>10.4f}")
    print(f"{'mAOE (rad)':<15} {results['mAOE']:>10.4f}")
    print(f"{'mAVE (m/s)':<15} {results['mAVE']:>10.4f}")
    print(f"{'mAAE':<15} {results['mAAE']:>10.4f}")
    print(separator)

    print("\nPer-class AP:")
    print(f"{'Class':<20} {'AP':>10}")
    print("-" * 35)
    for cls_name in class_names:
        ap_key = f"{cls_name}_AP"
        ap_val = results.get(ap_key, 0.0) * 100
        print(f"{cls_name:<20} {ap_val:>9.2f}%")
    print("-" * 35 + "\n")


# =============================================================================
# Main Evaluation Function
# =============================================================================


def evaluate(
    config_path: str,
    checkpoint_path: str,
    dataset_type: str = "kitti",
    batch_size: int = 4,
    num_workers: int = 4,
) -> Dict:
    """Main evaluation function: load model, run inference, compute metrics.

    Args:
        config_path: Path to the YAML configuration file.
        checkpoint_path: Path to the model checkpoint.
        dataset_type: Either 'kitti' or 'nuscenes'.
        batch_size: Batch size for the dataloader.
        num_workers: Number of dataloader workers.

    Returns:
        Dictionary containing all computed metrics.
    """
    config = load_config(config_path)

    if dataset_type == "kitti":
        config.setdefault("dataset", {})["type"] = "kitti"
    else:
        config.setdefault("dataset", {})["type"] = "nuscenes"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building model...")
    model = build_model(config)
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    print("Building dataloader...")
    dataloader = build_dataloader(config, batch_size, num_workers)

    class_names = config.get("class_names", ["Car", "Pedestrian", "Cyclist"])

    if dataset_type == "kitti":
        iou_thresholds = config.get("iou_thresholds", {
            "Car": 0.7,
            "Pedestrian": 0.5,
            "Cyclist": 0.5,
        })
    nms_iou_threshold = config.get("nms_iou_threshold", 0.1)
    score_threshold = config.get("score_threshold", 0.1)
    max_detections = config.get("max_detections", 500)

    gt_annos: List[Dict] = []
    pred_annos: List[Dict] = []

    total_inference_time = 0.0
    total_samples = 0
    total_batches = 0

    print(f"Running inference on {len(dataloader)} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                input_data = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                    if k != "gt_annos"
                }
                batch_gt_annos = batch.get("gt_annos", [])
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    input_data = batch[0]
                    batch_gt_annos = batch[1] if len(batch) > 1 else []
                    if isinstance(input_data, torch.Tensor):
                        input_data = input_data.to(device)
                    elif isinstance(input_data, dict):
                        input_data = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in input_data.items()
                        }
                else:
                    input_data = batch[0]
                    batch_gt_annos = []
                    if isinstance(input_data, torch.Tensor):
                        input_data = input_data.to(device)
            else:
                input_data = batch.to(device)
                batch_gt_annos = []

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            if isinstance(input_data, dict):
                output = model(**input_data)
            else:
                output = model(input_data)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            batch_time = end_time - start_time
            total_inference_time += batch_time

            if isinstance(output, (list, tuple)):
                batch_outputs = output
            elif isinstance(output, dict):
                if "batch_boxes" in output:
                    num_in_batch = len(output["batch_boxes"])
                    batch_outputs = [
                        {
                            "boxes": output["batch_boxes"][i],
                            "scores": output["batch_scores"][i],
                            "labels": output["batch_labels"][i],
                        }
                        for i in range(num_in_batch)
                    ]
                else:
                    batch_outputs = [output]
            else:
                batch_outputs = [output]

            for sample_idx, sample_output in enumerate(batch_outputs):
                if isinstance(sample_output, dict):
                    boxes, scores, labels = decode_predictions(sample_output, config)
                else:
                    boxes = sample_output[0].detach().cpu().numpy() if isinstance(sample_output[0], torch.Tensor) else sample_output[0]
                    scores = sample_output[1].detach().cpu().numpy() if isinstance(sample_output[1], torch.Tensor) else sample_output[1]
                    labels = sample_output[2].detach().cpu().numpy() if isinstance(sample_output[2], torch.Tensor) else sample_output[2]

                boxes, scores, labels = apply_nms_and_filter(
                    boxes, scores, labels,
                    nms_iou_threshold=nms_iou_threshold,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                )

                label_to_name = {i: name for i, name in enumerate(class_names)}
                pred_names = np.array([label_to_name.get(int(l), "Unknown") for l in labels])

                pred_anno = {
                    "name": pred_names,
                    "boxes_3d": boxes,
                    "score": scores,
                }
                pred_annos.append(pred_anno)

                if isinstance(batch_gt_annos, list) and sample_idx < len(batch_gt_annos):
                    gt_annos.append(batch_gt_annos[sample_idx])
                elif isinstance(batch_gt_annos, dict):
                    gt_anno_single = {}
                    for key, val in batch_gt_annos.items():
                        if isinstance(val, (list, np.ndarray)) and len(val) > sample_idx:
                            gt_anno_single[key] = val[sample_idx]
                        else:
                            gt_anno_single[key] = val
                    gt_annos.append(gt_anno_single)

                total_samples += 1

            total_batches += 1

            if (batch_idx + 1) % 50 == 0:
                elapsed_fps = total_samples / total_inference_time if total_inference_time > 0 else 0
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                      f"FPS: {elapsed_fps:.1f}")

    avg_fps = total_samples / total_inference_time if total_inference_time > 0 else 0.0
    avg_latency_ms = (total_inference_time / total_samples * 1000) if total_samples > 0 else 0.0

    print(f"\nInference complete: {total_samples} samples in {total_inference_time:.2f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average latency: {avg_latency_ms:.2f} ms/sample")

    if dataset_type == "kitti":
        print("\nComputing KITTI metrics...")
        metrics = compute_kitti_metrics(gt_annos, pred_annos, class_names, iou_thresholds)
        print_kitti_results(metrics)
        final_results = {
            "dataset": "kitti",
            "metrics": metrics,
            "fps": avg_fps,
            "latency_ms": avg_latency_ms,
            "total_samples": total_samples,
        }
    else:
        print("\nComputing nuScenes metrics...")
        metrics = compute_nuscenes_metrics(gt_annos, pred_annos, class_names)
        print_nuscenes_results(metrics, class_names)
        final_results = {
            "dataset": "nuscenes",
            "metrics": metrics,
            "fps": avg_fps,
            "latency_ms": avg_latency_ms,
            "total_samples": total_samples,
        }

    return final_results


# =============================================================================
# CLI Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="PointPillars 3D Object Detection Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth or .pt).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti",
        choices=["kitti", "nuscenes"],
        help="Dataset type for evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the evaluation script."""
    args = parse_args()

    print("=" * 60)
    print("PointPillars Evaluation")
    print("=" * 60)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers:    {args.num_workers}")
    print("=" * 60)

    results = evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\nEvaluation complete.")
    print(f"Results summary - FPS: {results['fps']:.1f}, "
          f"Latency: {results['latency_ms']:.2f} ms/sample")


if __name__ == "__main__":
    main()
