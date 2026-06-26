"""PointPillars TF2 Evaluation Script.

Production-quality evaluation pipeline computing KITTI-style 3D AP metrics and
nuScenes-style mAP/NDS. Supports checkpoint or SavedModel loading, FPS
measurement with proper warmup, and JSON result export.

Reference: Lang et al., "PointPillars: Fast Encoders for Object Detection
from Point Clouds", CVPR 2019.

Usage:
    python evaluate.py --checkpoint ./output/checkpoints/ckpt-32 \
                       --data_root /data/kitti \
                       --val_info kitti_infos_val.pkl \
                       --output_json results.json

    python evaluate.py --saved_model ./output/saved_model \
                       --data_root /data/nuscenes \
                       --val_info nuscenes_infos_val.pkl \
                       --metrics nuscenes \
                       --output_json results_nusc.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvalConfig:
    """Evaluation configuration parameters."""

    # Model
    checkpoint_path: Optional[str] = None
    saved_model_path: Optional[str] = None
    config_path: Optional[str] = None

    # Data
    data_root: str = "/data/kitti"
    val_info_path: str = "kitti_infos_val.pkl"

    # Voxelization
    point_cloud_range: List[float] = field(
        default_factory=lambda: [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
    )
    voxel_size: List[float] = field(default_factory=lambda: [0.16, 0.16, 4.0])
    max_points_per_voxel: int = 32
    max_voxels: int = 40000

    # Model params
    num_classes: int = 3
    class_names: List[str] = field(
        default_factory=lambda: ["Car", "Pedestrian", "Cyclist"]
    )
    pillar_features: int = 64
    backbone_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    backbone_strides: List[int] = field(default_factory=lambda: [2, 2, 2])
    backbone_num_blocks: List[int] = field(default_factory=lambda: [3, 5, 5])

    # Anchor config
    anchor_sizes: List[List[float]] = field(
        default_factory=lambda: [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]
    )
    anchor_rotations: List[float] = field(
        default_factory=lambda: [0.0, 1.5707963]
    )
    anchor_z_centers: List[float] = field(
        default_factory=lambda: [-1.0, -0.6, -0.6]
    )

    # Detection thresholds
    score_threshold: float = 0.1
    nms_iou_threshold: float = 0.5
    max_detections_per_class: int = 100

    # FPS measurement
    warmup_iterations: int = 10
    fps_iterations: int = 100

    # Metrics
    metrics_type: str = "kitti"  # "kitti" or "nuscenes" or "both"

    # Output
    output_json: str = "evaluation_results.json"


# =============================================================================
# KITTI 3D AP Metrics
# =============================================================================


@dataclass
class KITTIDifficulty:
    """KITTI difficulty level definitions based on bbox height, occlusion, truncation."""

    name: str
    min_bbox_height: float
    max_occlusion: int
    max_truncation: float


KITTI_DIFFICULTIES: List[KITTIDifficulty] = [
    KITTIDifficulty(name="easy", min_bbox_height=40.0, max_occlusion=0, max_truncation=0.15),
    KITTIDifficulty(name="moderate", min_bbox_height=25.0, max_occlusion=1, max_truncation=0.30),
    KITTIDifficulty(name="hard", min_bbox_height=25.0, max_occlusion=2, max_truncation=0.50),
]

KITTI_IOU_THRESHOLDS: Dict[str, float] = {
    "Car": 0.7,
    "Pedestrian": 0.5,
    "Cyclist": 0.5,
}


def compute_iou_3d(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Compute 3D IoU between two sets of axis-aligned bounding boxes.

    Boxes are parameterized as (x, y, z, w, l, h, yaw). The IoU computation
    uses the BEV (bird's eye view) overlap combined with height overlap for
    the 3D intersection volume.

    Args:
        boxes_a: Array of shape [M, 7] (x, y, z, w, l, h, yaw).
        boxes_b: Array of shape [N, 7] (x, y, z, w, l, h, yaw).

    Returns:
        IoU matrix of shape [M, N].
    """
    m = boxes_a.shape[0]
    n = boxes_b.shape[0]

    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float32)

    # BEV overlap using axis-aligned approximation for efficiency
    ax1 = boxes_a[:, 0] - boxes_a[:, 3] / 2.0
    ay1 = boxes_a[:, 1] - boxes_a[:, 4] / 2.0
    ax2 = boxes_a[:, 0] + boxes_a[:, 3] / 2.0
    ay2 = boxes_a[:, 1] + boxes_a[:, 4] / 2.0

    bx1 = boxes_b[:, 0] - boxes_b[:, 3] / 2.0
    by1 = boxes_b[:, 1] - boxes_b[:, 4] / 2.0
    bx2 = boxes_b[:, 0] + boxes_b[:, 3] / 2.0
    by2 = boxes_b[:, 1] + boxes_b[:, 4] / 2.0

    # BEV intersection
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_l = np.maximum(0.0, inter_y2 - inter_y1)
    bev_inter_area = inter_w * inter_l

    # Height overlap for 3D intersection
    az_bottom = boxes_a[:, 2] - boxes_a[:, 5] / 2.0
    az_top = boxes_a[:, 2] + boxes_a[:, 5] / 2.0
    bz_bottom = boxes_b[:, 2] - boxes_b[:, 5] / 2.0
    bz_top = boxes_b[:, 2] + boxes_b[:, 5] / 2.0

    z_inter_bottom = np.maximum(az_bottom[:, None], bz_bottom[None, :])
    z_inter_top = np.minimum(az_top[:, None], bz_top[None, :])
    z_inter = np.maximum(0.0, z_inter_top - z_inter_bottom)

    # 3D intersection volume
    inter_volume = bev_inter_area * z_inter

    # 3D volumes
    vol_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    vol_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]

    # Union volume
    union_volume = vol_a[:, None] + vol_b[None, :] - inter_volume

    iou = inter_volume / np.maximum(union_volume, 1e-7)
    return iou.astype(np.float32)


def filter_by_difficulty(
    gt_info: Dict[str, Any],
    difficulty: KITTIDifficulty,
) -> np.ndarray:
    """Filter ground truth annotations by KITTI difficulty level.

    Args:
        gt_info: Dictionary with keys 'bbox_heights', 'occlusions', 'truncations',
            each an array of shape [num_gt].
        difficulty: KITTI difficulty level specification.

    Returns:
        Boolean mask of shape [num_gt] indicating which GTs pass the filter.
    """
    bbox_heights = gt_info.get("bbox_heights", np.ones(gt_info["gt_boxes"].shape[0]) * 50.0)
    occlusions = gt_info.get("occlusions", np.zeros(gt_info["gt_boxes"].shape[0], dtype=np.int32))
    truncations = gt_info.get("truncations", np.zeros(gt_info["gt_boxes"].shape[0], dtype=np.float32))

    height_mask = bbox_heights >= difficulty.min_bbox_height
    occlusion_mask = occlusions <= difficulty.max_occlusion
    truncation_mask = truncations <= difficulty.max_truncation

    valid_mask = height_mask & occlusion_mask & truncation_mask
    return valid_mask


def compute_ap_single_class(
    precision: np.ndarray,
    recall: np.ndarray,
    num_recall_positions: int = 40,
) -> float:
    """Compute Average Precision using KITTI's 40-point interpolation.

    The AP is computed by sampling precision at 40 equally spaced recall
    points and averaging (KITTI's official protocol as of 2019).

    Args:
        precision: Precision values at each detection threshold.
        recall: Recall values at each detection threshold.
        num_recall_positions: Number of recall positions to sample (40 for KITTI).

    Returns:
        Average precision value.
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    recall_thresholds = np.linspace(0.0, 1.0, num_recall_positions + 1)[1:]
    precisions_at_recall = np.zeros(num_recall_positions, dtype=np.float64)

    for i, r_thresh in enumerate(recall_thresholds):
        # Find all precision values where recall >= threshold
        valid_mask = recall >= r_thresh
        if np.any(valid_mask):
            precisions_at_recall[i] = np.max(precision[valid_mask])
        else:
            precisions_at_recall[i] = 0.0

    ap = np.mean(precisions_at_recall)
    return float(ap)


def compute_precision_recall(
    det_scores: np.ndarray,
    det_matched: np.ndarray,
    num_gt: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute precision-recall curve from detection matches.

    Detections are sorted by decreasing confidence, then TP/FP is accumulated
    to produce the P-R curve.

    Args:
        det_scores: Confidence scores for each detection.
        det_matched: Boolean array indicating if each detection is a true positive.
        num_gt: Total number of ground truth objects.

    Returns:
        Tuple of (precision_array, recall_array) both of shape [num_detections].
    """
    if num_gt == 0 or len(det_scores) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Sort detections by score (descending)
    sorted_indices = np.argsort(-det_scores)
    det_matched_sorted = det_matched[sorted_indices]

    # Accumulate TP and FP
    tp_cumsum = np.cumsum(det_matched_sorted.astype(np.float64))
    fp_cumsum = np.cumsum((~det_matched_sorted).astype(np.float64))

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / float(num_gt)

    return precision, recall


def evaluate_kitti_3d_ap(
    all_predictions: List[Dict[str, np.ndarray]],
    all_ground_truths: List[Dict[str, Any]],
    class_names: List[str],
    iou_thresholds: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Compute KITTI-style 3D Average Precision for all classes and difficulties.

    For each class and difficulty level, matches detections to ground truths
    using 3D IoU and computes the 40-point interpolated AP.

    Args:
        all_predictions: List of per-sample prediction dicts with keys:
            'boxes': [N, 7], 'scores': [N], 'classes': [N] (int indices).
        all_ground_truths: List of per-sample GT dicts with keys:
            'gt_boxes': [M, 7], 'gt_classes': [M] (int indices),
            'bbox_heights': [M], 'occlusions': [M], 'truncations': [M].
        class_names: List of class name strings.
        iou_thresholds: Dict mapping class name to IoU threshold.

    Returns:
        Nested dict: results[class_name][difficulty_name] = AP value.
    """
    results: Dict[str, Dict[str, float]] = {}

    for cls_idx, cls_name in enumerate(class_names):
        iou_thresh = iou_thresholds.get(cls_name, 0.5)
        results[cls_name] = {}

        for difficulty in KITTI_DIFFICULTIES:
            all_det_scores: List[float] = []
            all_det_matched: List[bool] = []
            total_num_gt = 0

            for pred, gt_info in zip(all_predictions, all_ground_truths):
                # Filter ground truths for this class and difficulty
                gt_boxes = gt_info["gt_boxes"]
                gt_classes = gt_info["gt_classes"]

                gt_class_mask = gt_classes == cls_idx
                diff_mask = filter_by_difficulty(gt_info, difficulty)
                valid_gt_mask = gt_class_mask & diff_mask

                gt_boxes_filtered = gt_boxes[valid_gt_mask]
                num_gt_this_sample = gt_boxes_filtered.shape[0]
                total_num_gt += num_gt_this_sample

                # Filter predictions for this class
                pred_boxes = pred["boxes"]
                pred_scores = pred["scores"]
                pred_classes = pred["classes"]

                pred_class_mask = pred_classes == cls_idx
                pred_boxes_cls = pred_boxes[pred_class_mask]
                pred_scores_cls = pred_scores[pred_class_mask]

                if pred_boxes_cls.shape[0] == 0:
                    continue

                if num_gt_this_sample == 0:
                    # All detections are false positives
                    for score in pred_scores_cls:
                        all_det_scores.append(float(score))
                        all_det_matched.append(False)
                    continue

                # Compute 3D IoU between predictions and ground truths
                iou_matrix = compute_iou_3d(pred_boxes_cls, gt_boxes_filtered)

                # Greedy matching: for each detection (sorted by score), find best GT
                sorted_det_indices = np.argsort(-pred_scores_cls)
                gt_matched = np.zeros(num_gt_this_sample, dtype=np.bool_)

                for det_idx in sorted_det_indices:
                    score = pred_scores_cls[det_idx]
                    all_det_scores.append(float(score))

                    # Find best matching GT for this detection
                    ious_for_det = iou_matrix[det_idx]
                    best_gt_idx = np.argmax(ious_for_det)
                    best_iou = ious_for_det[best_gt_idx]

                    if best_iou >= iou_thresh and not gt_matched[best_gt_idx]:
                        all_det_matched.append(True)
                        gt_matched[best_gt_idx] = True
                    else:
                        all_det_matched.append(False)

            # Compute AP for this class and difficulty
            det_scores_arr = np.array(all_det_scores, dtype=np.float64)
            det_matched_arr = np.array(all_det_matched, dtype=np.bool_)

            precision, recall = compute_precision_recall(
                det_scores_arr, det_matched_arr, total_num_gt
            )
            ap = compute_ap_single_class(precision, recall, num_recall_positions=40)
            results[cls_name][difficulty.name] = ap

    return results


# =============================================================================
# nuScenes-style mAP and NDS Metrics
# =============================================================================


NUSCENES_IOU_THRESHOLDS: List[float] = [0.5, 1.0, 2.0, 4.0]
NUSCENES_DISTANCE_THRESHOLDS: Dict[str, List[float]] = {
    "Car": [0.5, 1.0, 2.0, 4.0],
    "Pedestrian": [0.5, 1.0, 2.0, 4.0],
    "Cyclist": [0.5, 1.0, 2.0, 4.0],
}


def compute_center_distance(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Compute Euclidean center distance between two sets of boxes in BEV.

    Args:
        boxes_a: Array of shape [M, 7] (x, y, z, w, l, h, yaw).
        boxes_b: Array of shape [N, 7] (x, y, z, w, l, h, yaw).

    Returns:
        Distance matrix of shape [M, N].
    """
    diff_x = boxes_a[:, 0:1] - boxes_b[:, 0:1].T
    diff_y = boxes_a[:, 1:2] - boxes_b[:, 1:2].T
    dist = np.sqrt(diff_x ** 2 + diff_y ** 2)
    return dist.astype(np.float32)


def compute_translation_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute translation error (ATE) as 2D Euclidean center distance.

    Args:
        pred_box: Predicted box [7] (x, y, z, w, l, h, yaw).
        gt_box: Ground truth box [7].

    Returns:
        Translation error in meters.
    """
    dx = pred_box[0] - gt_box[0]
    dy = pred_box[1] - gt_box[1]
    return float(np.sqrt(dx ** 2 + dy ** 2))


def compute_scale_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute scale error (ASE) as 1 - IOU3D of axis-aligned boxes.

    The scale error measures the mismatch in dimensions between
    the predicted and ground truth bounding box volumes.

    Args:
        pred_box: Predicted box [7] (x, y, z, w, l, h, yaw).
        gt_box: Ground truth box [7].

    Returns:
        Scale error in range [0, 1].
    """
    pred_vol = pred_box[3] * pred_box[4] * pred_box[5]
    gt_vol = gt_box[3] * gt_box[4] * gt_box[5]

    # Compute intersection volume (using min of dimensions)
    inter_w = min(pred_box[3], gt_box[3])
    inter_l = min(pred_box[4], gt_box[4])
    inter_h = min(pred_box[5], gt_box[5])
    inter_vol = inter_w * inter_l * inter_h

    union_vol = pred_vol + gt_vol - inter_vol
    iou = inter_vol / max(union_vol, 1e-7)
    return float(1.0 - iou)


def compute_orientation_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute orientation error (AOE) as the smallest angle difference.

    Uses the absolute angular difference between predicted and ground truth
    yaw angles, normalized to [0, pi].

    Args:
        pred_box: Predicted box [7] (x, y, z, w, l, h, yaw).
        gt_box: Ground truth box [7].

    Returns:
        Orientation error in radians [0, pi].
    """
    angle_diff = pred_box[6] - gt_box[6]
    # Normalize to [-pi, pi]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    return float(np.abs(angle_diff))


def compute_velocity_error(
    pred_velocity: Optional[np.ndarray],
    gt_velocity: Optional[np.ndarray],
) -> float:
    """Compute velocity error (AVE) as L2 distance of velocity vectors.

    Args:
        pred_velocity: Predicted velocity [2] (vx, vy) or None.
        gt_velocity: Ground truth velocity [2] (vx, vy) or None.

    Returns:
        Velocity error in m/s. Returns 1.0 if either velocity is unavailable.
    """
    if pred_velocity is None or gt_velocity is None:
        return 1.0
    diff = pred_velocity - gt_velocity
    return float(np.sqrt(np.sum(diff ** 2)))


def compute_attribute_error(
    pred_attribute: Optional[int],
    gt_attribute: Optional[int],
) -> float:
    """Compute attribute error (AAE) as 1 - accuracy.

    Attributes represent discrete object states (e.g., parked, moving, stopped).

    Args:
        pred_attribute: Predicted attribute index or None.
        gt_attribute: Ground truth attribute index or None.

    Returns:
        Attribute error: 0.0 if correct, 1.0 otherwise.
    """
    if pred_attribute is None or gt_attribute is None:
        return 1.0
    return 0.0 if pred_attribute == gt_attribute else 1.0


def compute_nuscenes_ap_single(
    det_scores: np.ndarray,
    det_matched: np.ndarray,
    num_gt: int,
    num_recall_positions: int = 100,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute AP using nuScenes 101-point interpolation.

    Args:
        det_scores: Confidence scores for detections.
        det_matched: Boolean array of true positive matches.
        num_gt: Total number of ground truth objects.
        num_recall_positions: Number of recall sample points (100 for nuScenes).

    Returns:
        Tuple of (AP, precision_array, recall_array).
    """
    if num_gt == 0 or len(det_scores) == 0:
        return 0.0, np.array([]), np.array([])

    # Sort by descending score
    sorted_indices = np.argsort(-det_scores)
    det_matched_sorted = det_matched[sorted_indices]

    tp_cumsum = np.cumsum(det_matched_sorted.astype(np.float64))
    fp_cumsum = np.cumsum((~det_matched_sorted).astype(np.float64))

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / float(num_gt)

    # Sample at equally-spaced recall values
    recall_thresholds = np.linspace(0.0, 1.0, num_recall_positions + 1)[1:]
    precisions_at_recall = np.zeros(num_recall_positions, dtype=np.float64)

    for i, r_thresh in enumerate(recall_thresholds):
        valid_mask = recall >= r_thresh
        if np.any(valid_mask):
            precisions_at_recall[i] = np.max(precision[valid_mask])

    ap = float(np.mean(precisions_at_recall))
    return ap, precision, recall


def evaluate_nuscenes_metrics(
    all_predictions: List[Dict[str, Any]],
    all_ground_truths: List[Dict[str, Any]],
    class_names: List[str],
    distance_thresholds: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Compute nuScenes-style mAP and NDS metrics.

    Evaluates detections using center-distance matching (instead of IoU)
    at multiple distance thresholds, then computes mean AP and True Positive
    error metrics (ATE, ASE, AOE, AVE, AAE) to produce the NDS.

    Args:
        all_predictions: Per-sample predictions with keys:
            'boxes': [N, 7], 'scores': [N], 'classes': [N],
            'velocities': Optional [N, 2], 'attributes': Optional [N].
        all_ground_truths: Per-sample GTs with keys:
            'gt_boxes': [M, 7], 'gt_classes': [M],
            'gt_velocities': Optional [M, 2], 'gt_attributes': Optional [M].
        class_names: List of class name strings.
        distance_thresholds: Dict mapping class name to list of distance thresholds.

    Returns:
        Dictionary with 'mAP', 'NDS', per-class AP, and TP error metrics.
    """
    results: Dict[str, Any] = {
        "per_class_ap": {},
        "per_class_errors": {},
        "mean_ap_per_threshold": {},
    }

    all_class_aps: List[float] = []
    all_translation_errors: List[float] = []
    all_scale_errors: List[float] = []
    all_orientation_errors: List[float] = []
    all_velocity_errors: List[float] = []
    all_attribute_errors: List[float] = []

    for cls_idx, cls_name in enumerate(class_names):
        thresholds = distance_thresholds.get(cls_name, [2.0])
        class_aps: List[float] = []

        cls_translation_errors: List[float] = []
        cls_scale_errors: List[float] = []
        cls_orientation_errors: List[float] = []
        cls_velocity_errors: List[float] = []
        cls_attribute_errors: List[float] = []

        for dist_thresh in thresholds:
            all_det_scores: List[float] = []
            all_det_matched: List[bool] = []
            total_num_gt = 0

            for pred, gt_info in zip(all_predictions, all_ground_truths):
                gt_boxes = gt_info["gt_boxes"]
                gt_classes = gt_info["gt_classes"]
                gt_class_mask = gt_classes == cls_idx
                gt_boxes_cls = gt_boxes[gt_class_mask]
                num_gt_this = gt_boxes_cls.shape[0]
                total_num_gt += num_gt_this

                pred_boxes = pred["boxes"]
                pred_scores = pred["scores"]
                pred_classes = pred["classes"]
                pred_class_mask = pred_classes == cls_idx
                pred_boxes_cls = pred_boxes[pred_class_mask]
                pred_scores_cls = pred_scores[pred_class_mask]

                if pred_boxes_cls.shape[0] == 0:
                    continue

                if num_gt_this == 0:
                    for s in pred_scores_cls:
                        all_det_scores.append(float(s))
                        all_det_matched.append(False)
                    continue

                # Center distance matching
                dist_matrix = compute_center_distance(pred_boxes_cls, gt_boxes_cls)

                # Match detections to GTs (greedy, by score)
                sorted_det_indices = np.argsort(-pred_scores_cls)
                gt_matched_flags = np.zeros(num_gt_this, dtype=np.bool_)

                for det_idx in sorted_det_indices:
                    all_det_scores.append(float(pred_scores_cls[det_idx]))
                    dists_for_det = dist_matrix[det_idx]
                    best_gt_idx = np.argmin(dists_for_det)
                    best_dist = dists_for_det[best_gt_idx]

                    if best_dist <= dist_thresh and not gt_matched_flags[best_gt_idx]:
                        all_det_matched.append(True)
                        gt_matched_flags[best_gt_idx] = True

                        # Compute TP errors (only for matched pairs at this threshold)
                        pred_box = pred_boxes_cls[det_idx]
                        gt_box = gt_boxes_cls[best_gt_idx]
                        cls_translation_errors.append(
                            compute_translation_error(pred_box, gt_box)
                        )
                        cls_scale_errors.append(
                            compute_scale_error(pred_box, gt_box)
                        )
                        cls_orientation_errors.append(
                            compute_orientation_error(pred_box, gt_box)
                        )

                        # Velocity error
                        pred_vel = pred.get("velocities")
                        gt_vel = gt_info.get("gt_velocities")
                        pred_vel_i = pred_vel[pred_class_mask][det_idx] if pred_vel is not None else None
                        gt_vel_i = gt_vel[gt_class_mask][best_gt_idx] if gt_vel is not None else None
                        cls_velocity_errors.append(
                            compute_velocity_error(pred_vel_i, gt_vel_i)
                        )

                        # Attribute error
                        pred_attr = pred.get("attributes")
                        gt_attr = gt_info.get("gt_attributes")
                        pred_attr_i = int(pred_attr[pred_class_mask][det_idx]) if pred_attr is not None else None
                        gt_attr_i = int(gt_attr[gt_class_mask][best_gt_idx]) if gt_attr is not None else None
                        cls_attribute_errors.append(
                            compute_attribute_error(pred_attr_i, gt_attr_i)
                        )
                    else:
                        all_det_matched.append(False)

            # Compute AP for this threshold
            det_scores_arr = np.array(all_det_scores, dtype=np.float64)
            det_matched_arr = np.array(all_det_matched, dtype=np.bool_)
            ap, _, _ = compute_nuscenes_ap_single(det_scores_arr, det_matched_arr, total_num_gt)
            class_aps.append(ap)

        # Average AP across thresholds for this class
        mean_class_ap = float(np.mean(class_aps)) if class_aps else 0.0
        results["per_class_ap"][cls_name] = mean_class_ap
        all_class_aps.append(mean_class_ap)

        # Average TP errors for this class
        mean_ate = float(np.mean(cls_translation_errors)) if cls_translation_errors else 1.0
        mean_ase = float(np.mean(cls_scale_errors)) if cls_scale_errors else 1.0
        mean_aoe = float(np.mean(cls_orientation_errors)) if cls_orientation_errors else 1.0
        mean_ave = float(np.mean(cls_velocity_errors)) if cls_velocity_errors else 1.0
        mean_aae = float(np.mean(cls_attribute_errors)) if cls_attribute_errors else 1.0

        results["per_class_errors"][cls_name] = {
            "ATE": mean_ate,
            "ASE": mean_ase,
            "AOE": mean_aoe,
            "AVE": mean_ave,
            "AAE": mean_aae,
        }

        all_translation_errors.append(mean_ate)
        all_scale_errors.append(mean_ase)
        all_orientation_errors.append(mean_aoe)
        all_velocity_errors.append(mean_ave)
        all_attribute_errors.append(mean_aae)

    # Compute mAP (mean over classes)
    mAP = float(np.mean(all_class_aps)) if all_class_aps else 0.0
    results["mAP"] = mAP

    # Compute NDS (nuScenes Detection Score)
    # NDS = 1/10 * [5*mAP + sum(1 - min(1, TP_err)) for each of 5 TP metrics]
    mean_ate_all = float(np.mean(all_translation_errors)) if all_translation_errors else 1.0
    mean_ase_all = float(np.mean(all_scale_errors)) if all_scale_errors else 1.0
    mean_aoe_all = float(np.mean(all_orientation_errors)) if all_orientation_errors else 1.0
    mean_ave_all = float(np.mean(all_velocity_errors)) if all_velocity_errors else 1.0
    mean_aae_all = float(np.mean(all_attribute_errors)) if all_attribute_errors else 1.0

    tp_scores = [
        max(0.0, 1.0 - min(1.0, mean_ate_all)),
        max(0.0, 1.0 - min(1.0, mean_ase_all)),
        max(0.0, 1.0 - min(1.0, mean_aoe_all)),
        max(0.0, 1.0 - min(1.0, mean_ave_all)),
        max(0.0, 1.0 - min(1.0, mean_aae_all)),
    ]

    nds = (5.0 * mAP + sum(tp_scores)) / 10.0
    results["NDS"] = float(nds)

    results["mean_errors"] = {
        "mATE": mean_ate_all,
        "mASE": mean_ase_all,
        "mAOE": mean_aoe_all,
        "mAVE": mean_ave_all,
        "mAAE": mean_aae_all,
    }

    return results


# =============================================================================
# Voxelization for Evaluation
# =============================================================================


def voxelize_point_cloud(
    points: np.ndarray, config: EvalConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert point cloud to pillar representation.

    Args:
        points: Raw point cloud [N, 4] (x, y, z, intensity).
        config: Evaluation configuration with voxel params.

    Returns:
        Tuple of (pillars, coords, num_points_per_pillar).
    """
    pc_range = np.array(config.point_cloud_range)
    voxel_size = np.array(config.voxel_size)
    max_points = config.max_points_per_voxel
    max_voxels = config.max_voxels

    # Filter points within range
    mask = (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] < pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] < pc_range[4])
        & (points[:, 2] >= pc_range[2])
        & (points[:, 2] < pc_range[5])
    )
    points = points[mask]

    grid_idx_x = ((points[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int32)
    grid_idx_y = ((points[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int32)

    grid_size_x = int((pc_range[3] - pc_range[0]) / voxel_size[0])
    grid_size_y = int((pc_range[4] - pc_range[1]) / voxel_size[1])

    grid_idx_x = np.clip(grid_idx_x, 0, grid_size_x - 1)
    grid_idx_y = np.clip(grid_idx_y, 0, grid_size_y - 1)

    pillar_ids = grid_idx_y * grid_size_x + grid_idx_x
    unique_pillars, inverse_indices = np.unique(pillar_ids, return_inverse=True)

    if len(unique_pillars) > max_voxels:
        selected = np.random.choice(len(unique_pillars), max_voxels, replace=False)
        selected_set = set(selected.tolist())
        keep_mask = np.array(
            [inverse_indices[i] in selected_set for i in range(len(points))],
            dtype=np.bool_,
        )
        points = points[keep_mask]
        grid_idx_x = grid_idx_x[keep_mask]
        grid_idx_y = grid_idx_y[keep_mask]
        pillar_ids = pillar_ids[keep_mask]
        unique_pillars, inverse_indices = np.unique(pillar_ids, return_inverse=True)

    num_pillars = min(len(unique_pillars), max_voxels)

    pillars = np.zeros((max_voxels, max_points, 9), dtype=np.float32)
    coords = np.zeros((max_voxels, 2), dtype=np.int32)
    num_points_per_pillar = np.zeros(max_voxels, dtype=np.int32)

    for i in range(num_pillars):
        point_mask = inverse_indices == i
        pillar_points = points[point_mask]

        n_pts = min(len(pillar_points), max_points)
        if len(pillar_points) > max_points:
            choice = np.random.choice(len(pillar_points), max_points, replace=False)
            pillar_points = pillar_points[choice]
            n_pts = max_points

        first_pt_idx = np.where(point_mask)[0][0]
        cx = grid_idx_x[first_pt_idx] * voxel_size[0] + pc_range[0] + voxel_size[0] / 2.0
        cy = grid_idx_y[first_pt_idx] * voxel_size[1] + pc_range[1] + voxel_size[1] / 2.0

        mean_xyz = pillar_points[:n_pts, :3].mean(axis=0)

        features = np.zeros((n_pts, 9), dtype=np.float32)
        features[:, :4] = pillar_points[:n_pts, :4]
        features[:, 4] = pillar_points[:n_pts, 0] - mean_xyz[0]
        features[:, 5] = pillar_points[:n_pts, 1] - mean_xyz[1]
        features[:, 6] = pillar_points[:n_pts, 2] - mean_xyz[2]
        features[:, 7] = pillar_points[:n_pts, 0] - cx
        features[:, 8] = pillar_points[:n_pts, 1] - cy

        pillars[i, :n_pts, :] = features
        coords[i, 0] = grid_idx_x[first_pt_idx]
        coords[i, 1] = grid_idx_y[first_pt_idx]
        num_points_per_pillar[i] = n_pts

    return pillars, coords, num_points_per_pillar


# =============================================================================
# Anchor Generation
# =============================================================================


def generate_anchors(config: EvalConfig) -> np.ndarray:
    """Generate anchor boxes for the detection head.

    Args:
        config: Evaluation config with anchor parameters.

    Returns:
        Anchor array of shape [total_anchors, 7].
    """
    pc_range = config.point_cloud_range
    voxel_size = config.voxel_size
    feature_stride = 2

    grid_x = int((pc_range[3] - pc_range[0]) / (voxel_size[0] * feature_stride))
    grid_y = int((pc_range[4] - pc_range[1]) / (voxel_size[1] * feature_stride))

    x_centers = np.linspace(
        pc_range[0] + voxel_size[0] * feature_stride * 0.5,
        pc_range[3] - voxel_size[0] * feature_stride * 0.5,
        grid_x,
    )
    y_centers = np.linspace(
        pc_range[1] + voxel_size[1] * feature_stride * 0.5,
        pc_range[4] - voxel_size[1] * feature_stride * 0.5,
        grid_y,
    )

    xx, yy = np.meshgrid(x_centers, y_centers)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    num_locations = len(xx)

    all_anchors = []
    for cls_idx, size in enumerate(config.anchor_sizes):
        z_center = config.anchor_z_centers[cls_idx]
        w, l, h = size
        for rotation in config.anchor_rotations:
            anchors = np.stack(
                [
                    xx,
                    yy,
                    np.full(num_locations, z_center),
                    np.full(num_locations, w),
                    np.full(num_locations, l),
                    np.full(num_locations, h),
                    np.full(num_locations, rotation),
                ],
                axis=-1,
            )
            all_anchors.append(anchors)

    return np.concatenate(all_anchors, axis=0).astype(np.float32)


# =============================================================================
# Box Decoding and NMS
# =============================================================================


def decode_boxes(
    reg_preds: np.ndarray, anchors: np.ndarray
) -> np.ndarray:
    """Decode box regression predictions relative to anchors.

    Args:
        reg_preds: Regression predictions [N, 7].
        anchors: Anchor boxes [N, 7].

    Returns:
        Decoded boxes [N, 7] in world coordinates.
    """
    anchor_diag = np.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    x = reg_preds[:, 0] * anchor_diag + anchors[:, 0]
    y = reg_preds[:, 1] * anchor_diag + anchors[:, 1]
    z = reg_preds[:, 2] * anchors[:, 5] + anchors[:, 2]
    w = np.exp(reg_preds[:, 3]) * anchors[:, 3]
    l_dim = np.exp(reg_preds[:, 4]) * anchors[:, 4]
    h = np.exp(reg_preds[:, 5]) * anchors[:, 5]
    yaw = reg_preds[:, 6] + anchors[:, 6]

    decoded = np.stack([x, y, z, w, l_dim, h, yaw], axis=-1)
    return decoded.astype(np.float32)


def nms_bev(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Non-maximum suppression in BEV (bird's eye view).

    Args:
        boxes: Boxes [N, 7] (x, y, z, w, l, h, yaw).
        scores: Confidence scores [N].
        iou_threshold: IoU threshold for suppression.

    Returns:
        Indices of boxes to keep, sorted by score.
    """
    if len(scores) == 0:
        return np.array([], dtype=np.int64)

    # Convert to axis-aligned BEV rectangles: [x1, y1, x2, y2]
    x1 = boxes[:, 0] - boxes[:, 3] / 2.0
    y1 = boxes[:, 1] - boxes[:, 4] / 2.0
    x2 = boxes[:, 0] + boxes[:, 3] / 2.0
    y2 = boxes[:, 1] + boxes[:, 4] / 2.0

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)

    keep: List[int] = []
    while order.size > 0:
        idx = order[0]
        keep.append(int(idx))

        # Compute IoU of this box with all remaining
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = areas[idx] + areas[order[1:]] - inter_area
        iou = inter_area / np.maximum(union_area, 1e-7)

        # Keep boxes with IoU below threshold
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.int64)


def post_process_predictions(
    cls_preds: np.ndarray,
    reg_preds: np.ndarray,
    dir_preds: np.ndarray,
    anchors: np.ndarray,
    config: EvalConfig,
) -> Dict[str, np.ndarray]:
    """Post-process raw network outputs into final detections.

    Applies sigmoid to class scores, decodes boxes, applies direction
    correction, filters by score, and runs per-class NMS.

    Args:
        cls_preds: Raw class logits [num_anchors, num_classes].
        reg_preds: Raw regression predictions [num_anchors, 7].
        dir_preds: Raw direction logits [num_anchors, 2].
        anchors: Anchor boxes [num_anchors, 7].
        config: Evaluation configuration.

    Returns:
        Dict with 'boxes' [K, 7], 'scores' [K], 'classes' [K].
    """
    num_classes = config.num_classes

    # Sigmoid for scores
    cls_scores = 1.0 / (1.0 + np.exp(-cls_preds))

    # Decode boxes
    decoded_boxes = decode_boxes(reg_preds, anchors)

    # Direction correction
    dir_labels = np.argmax(dir_preds, axis=-1)
    dir_offset = dir_labels.astype(np.float32) * np.pi
    decoded_boxes[:, 6] = decoded_boxes[:, 6] + dir_offset
    # Normalize yaw to [-pi, pi]
    decoded_boxes[:, 6] = np.arctan2(
        np.sin(decoded_boxes[:, 6]), np.cos(decoded_boxes[:, 6])
    )

    # Per-class NMS
    final_boxes: List[np.ndarray] = []
    final_scores: List[np.ndarray] = []
    final_classes: List[np.ndarray] = []

    for cls_idx in range(num_classes):
        class_scores = cls_scores[:, cls_idx]
        score_mask = class_scores > config.score_threshold
        filtered_scores = class_scores[score_mask]
        filtered_boxes = decoded_boxes[score_mask]

        if len(filtered_scores) == 0:
            continue

        keep_indices = nms_bev(filtered_boxes, filtered_scores, config.nms_iou_threshold)

        if len(keep_indices) > config.max_detections_per_class:
            keep_indices = keep_indices[:config.max_detections_per_class]

        final_boxes.append(filtered_boxes[keep_indices])
        final_scores.append(filtered_scores[keep_indices])
        final_classes.append(np.full(len(keep_indices), cls_idx, dtype=np.int32))

    if len(final_boxes) == 0:
        return {
            "boxes": np.zeros((0, 7), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int32),
        }

    return {
        "boxes": np.concatenate(final_boxes, axis=0),
        "scores": np.concatenate(final_scores, axis=0),
        "classes": np.concatenate(final_classes, axis=0),
    }


# =============================================================================
# Model Loading
# =============================================================================


def build_model(config: EvalConfig) -> tf.keras.Model:
    """Build the PointPillars model architecture matching the training config.

    This imports and constructs the model from train.py using the eval config
    parameters, providing a model that can load weights from a checkpoint.

    Args:
        config: Evaluation configuration.

    Returns:
        Uninitialized PointPillarsModel ready for weight loading.
    """
    # Import from the training module in the same package
    from . import train as train_module

    train_config = train_module.TrainConfig(
        num_classes=config.num_classes,
        pillar_features=config.pillar_features,
        backbone_channels=config.backbone_channels,
        backbone_strides=config.backbone_strides,
        backbone_num_blocks=config.backbone_num_blocks,
    )

    # Configure anchors to match
    train_config.anchors = []
    for cls_idx in range(config.num_classes):
        anchor_cfg = train_module.AnchorConfig(
            class_name=config.class_names[cls_idx],
            anchor_sizes=[config.anchor_sizes[cls_idx]],
            anchor_rotations=config.anchor_rotations,
            anchor_z_center=config.anchor_z_centers[cls_idx],
        )
        train_config.anchors.append(anchor_cfg)

    train_config.voxel = train_module.VoxelConfig(
        point_cloud_range=config.point_cloud_range,
        voxel_size=config.voxel_size,
        max_points_per_voxel=config.max_points_per_voxel,
        max_voxels=config.max_voxels,
    )

    model = train_module.PointPillarsModel(train_config, name="pointpillars")
    return model


def load_model_from_checkpoint(
    config: EvalConfig,
) -> tf.keras.Model:
    """Load model from a TF2 checkpoint.

    Builds the model architecture, creates a dummy forward pass to initialize
    weights, then restores from the checkpoint file.

    Args:
        config: Evaluation config with checkpoint_path set.

    Returns:
        Model with restored weights.
    """
    model = build_model(config)

    # Build model with dummy input to initialize variables
    max_voxels = config.max_voxels
    max_points = config.max_points_per_voxel
    dummy_pillars = tf.zeros([1, max_voxels, max_points, 9])
    dummy_coords = tf.zeros([1, max_voxels, 2], dtype=tf.int32)
    dummy_num_points = tf.zeros([1, max_voxels], dtype=tf.int32)
    model(dummy_pillars, dummy_coords, dummy_num_points, training=False)

    # Restore from checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(config.checkpoint_path)
    status.expect_partial()
    logger.info("Model restored from checkpoint: %s", config.checkpoint_path)

    return model


def load_model_from_saved_model(
    config: EvalConfig,
) -> tf.keras.Model:
    """Load model from a TF2 SavedModel directory.

    Args:
        config: Evaluation config with saved_model_path set.

    Returns:
        Loaded SavedModel.
    """
    model = tf.saved_model.load(config.saved_model_path)
    logger.info("Model loaded from SavedModel: %s", config.saved_model_path)
    return model


# =============================================================================
# Inference Runner
# =============================================================================


def run_inference_single(
    model: tf.keras.Model,
    pillars: np.ndarray,
    coords: np.ndarray,
    num_points: np.ndarray,
    is_saved_model: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on a single sample.

    Args:
        model: Loaded model (checkpoint-based or SavedModel).
        pillars: Voxelized pillars [max_voxels, max_points, 9].
        coords: Grid coordinates [max_voxels, 2].
        num_points: Points per pillar [max_voxels].
        is_saved_model: Whether model is a SavedModel (different call signature).

    Returns:
        Tuple of (cls_preds, reg_preds, dir_preds) each as numpy arrays.
    """
    pillars_tf = tf.constant(pillars[np.newaxis], dtype=tf.float32)
    coords_tf = tf.constant(coords[np.newaxis], dtype=tf.int32)
    num_points_tf = tf.constant(num_points[np.newaxis], dtype=tf.int32)

    if is_saved_model:
        # SavedModel may use a different signature
        infer_fn = model.signatures.get("serving_default", None)
        if infer_fn is not None:
            outputs = infer_fn(
                pillars=pillars_tf, coords=coords_tf, num_points=num_points_tf
            )
            cls_preds = outputs["cls_preds"].numpy()[0]
            reg_preds = outputs["reg_preds"].numpy()[0]
            dir_preds = outputs["dir_preds"].numpy()[0]
        else:
            result = model(pillars_tf, coords_tf, num_points_tf, training=False)
            cls_preds = result[0].numpy()[0]
            reg_preds = result[1].numpy()[0]
            dir_preds = result[2].numpy()[0]
    else:
        cls_preds_tf, reg_preds_tf, dir_preds_tf = model(
            pillars_tf, coords_tf, num_points_tf, training=False
        )
        cls_preds = cls_preds_tf.numpy()[0]
        reg_preds = reg_preds_tf.numpy()[0]
        dir_preds = dir_preds_tf.numpy()[0]

    return cls_preds, reg_preds, dir_preds


# =============================================================================
# FPS Measurement
# =============================================================================


def measure_fps(
    model: tf.keras.Model,
    sample_pillars: np.ndarray,
    sample_coords: np.ndarray,
    sample_num_points: np.ndarray,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    is_saved_model: bool = False,
) -> Dict[str, float]:
    """Measure inference FPS with proper warmup and statistical reporting.

    Performs warmup iterations to allow GPU kernels to be compiled and cached,
    then measures latency over multiple iterations to compute mean, std, and FPS.

    Args:
        model: Loaded model.
        sample_pillars: Sample pillar data for benchmarking.
        sample_coords: Sample coordinates.
        sample_num_points: Sample point counts.
        warmup_iterations: Number of warmup forward passes.
        benchmark_iterations: Number of timed forward passes.
        is_saved_model: Whether model is a SavedModel.

    Returns:
        Dict with 'mean_latency_ms', 'std_latency_ms', 'fps',
        'min_latency_ms', 'max_latency_ms', 'p50_latency_ms', 'p95_latency_ms'.
    """
    logger.info("Running %d warmup iterations...", warmup_iterations)
    for _ in range(warmup_iterations):
        run_inference_single(
            model, sample_pillars, sample_coords, sample_num_points, is_saved_model
        )

    logger.info("Running %d benchmark iterations...", benchmark_iterations)
    latencies: List[float] = []

    for _ in range(benchmark_iterations):
        start_time = time.perf_counter()
        run_inference_single(
            model, sample_pillars, sample_coords, sample_num_points, is_saved_model
        )
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000.0)  # ms

    latencies_arr = np.array(latencies, dtype=np.float64)
    mean_latency = float(np.mean(latencies_arr))
    std_latency = float(np.std(latencies_arr))
    min_latency = float(np.min(latencies_arr))
    max_latency = float(np.max(latencies_arr))
    p50_latency = float(np.percentile(latencies_arr, 50))
    p95_latency = float(np.percentile(latencies_arr, 95))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    return {
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "fps": fps,
        "warmup_iterations": warmup_iterations,
        "benchmark_iterations": benchmark_iterations,
    }


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================


def run_evaluation(config: EvalConfig) -> Dict[str, Any]:
    """Execute the full evaluation pipeline.

    Steps:
    1. Load model from checkpoint or SavedModel
    2. Generate anchors
    3. Run inference on all validation samples
    4. Post-process predictions (decode + NMS)
    5. Compute KITTI and/or nuScenes metrics
    6. Measure FPS
    7. Save results to JSON

    Args:
        config: Evaluation configuration.

    Returns:
        Complete results dictionary.
    """
    results: Dict[str, Any] = {
        "config": {
            "metrics_type": config.metrics_type,
            "score_threshold": config.score_threshold,
            "nms_iou_threshold": config.nms_iou_threshold,
            "num_classes": config.num_classes,
            "class_names": config.class_names,
        },
    }

    # Load model
    is_saved_model = False
    if config.saved_model_path and os.path.isdir(config.saved_model_path):
        model = load_model_from_saved_model(config)
        is_saved_model = True
    elif config.checkpoint_path:
        model = load_model_from_checkpoint(config)
    else:
        raise ValueError(
            "Must specify either --checkpoint or --saved_model for model loading."
        )

    # Generate anchors
    anchors = generate_anchors(config)
    logger.info("Generated %d anchors", anchors.shape[0])

    # Load validation info
    val_info_path = os.path.join(config.data_root, config.val_info_path)
    with open(val_info_path, "rb") as f:
        val_infos = pickle.load(f)
    logger.info("Loaded %d validation samples from %s", len(val_infos), val_info_path)

    # Run inference on all samples
    all_predictions: List[Dict[str, np.ndarray]] = []
    all_ground_truths: List[Dict[str, Any]] = []
    inference_times: List[float] = []

    logger.info("Running inference on %d samples...", len(val_infos))
    for sample_idx, info in enumerate(val_infos):
        # Load point cloud
        pc_path = info["point_cloud_path"]
        points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)

        # Voxelize
        pillars, coords, num_points = voxelize_point_cloud(points, config)

        # Run inference
        start_time = time.perf_counter()
        cls_preds, reg_preds, dir_preds = run_inference_single(
            model, pillars, coords, num_points, is_saved_model
        )
        inference_time = time.perf_counter() - start_time
        inference_times.append(inference_time)

        # Post-process
        detections = post_process_predictions(
            cls_preds, reg_preds, dir_preds, anchors, config
        )
        all_predictions.append(detections)

        # Collect ground truth
        gt_info = {
            "gt_boxes": info.get("gt_boxes", np.zeros((0, 7), dtype=np.float32)),
            "gt_classes": info.get("gt_classes", np.zeros((0,), dtype=np.int32)),
            "bbox_heights": info.get("bbox_heights", np.array([])),
            "occlusions": info.get("occlusions", np.array([])),
            "truncations": info.get("truncations", np.array([])),
        }
        # Optional nuScenes fields
        if "gt_velocities" in info:
            gt_info["gt_velocities"] = info["gt_velocities"]
        if "gt_attributes" in info:
            gt_info["gt_attributes"] = info["gt_attributes"]

        all_ground_truths.append(gt_info)

        if (sample_idx + 1) % 100 == 0:
            avg_time = np.mean(inference_times[-100:]) * 1000.0
            logger.info(
                "  Processed %d/%d samples (avg %.1f ms/sample)",
                sample_idx + 1,
                len(val_infos),
                avg_time,
            )

    # Compute metrics
    logger.info("Computing metrics...")

    if config.metrics_type in ("kitti", "both"):
        kitti_results = evaluate_kitti_3d_ap(
            all_predictions,
            all_ground_truths,
            config.class_names,
            KITTI_IOU_THRESHOLDS,
        )
        results["kitti_3d_ap"] = kitti_results

        # Compute mAP across all classes and difficulties
        all_aps: List[float] = []
        for cls_name in config.class_names:
            for diff_name in ["easy", "moderate", "hard"]:
                ap_val = kitti_results.get(cls_name, {}).get(diff_name, 0.0)
                all_aps.append(ap_val)
        results["kitti_mAP"] = float(np.mean(all_aps)) if all_aps else 0.0

        # Log results
        logger.info("=== KITTI 3D AP Results ===")
        header = f"{'Class':<15} {'Easy':>8} {'Moderate':>10} {'Hard':>8}"
        logger.info(header)
        logger.info("-" * len(header))
        for cls_name in config.class_names:
            easy = kitti_results.get(cls_name, {}).get("easy", 0.0) * 100
            mod = kitti_results.get(cls_name, {}).get("moderate", 0.0) * 100
            hard = kitti_results.get(cls_name, {}).get("hard", 0.0) * 100
            logger.info(f"{cls_name:<15} {easy:>7.2f}% {mod:>9.2f}% {hard:>7.2f}%")
        logger.info(f"{'mAP':<15} {results['kitti_mAP'] * 100:>7.2f}%")

    if config.metrics_type in ("nuscenes", "both"):
        nuscenes_results = evaluate_nuscenes_metrics(
            all_predictions,
            all_ground_truths,
            config.class_names,
            NUSCENES_DISTANCE_THRESHOLDS,
        )
        results["nuscenes"] = nuscenes_results

        # Log results
        logger.info("=== nuScenes Metrics ===")
        logger.info("mAP:  %.4f", nuscenes_results["mAP"])
        logger.info("NDS:  %.4f", nuscenes_results["NDS"])
        logger.info("--- Per-class AP ---")
        for cls_name in config.class_names:
            ap = nuscenes_results["per_class_ap"].get(cls_name, 0.0)
            logger.info("  %s: %.4f", cls_name, ap)
        logger.info("--- Mean TP Errors ---")
        for metric, value in nuscenes_results["mean_errors"].items():
            logger.info("  %s: %.4f", metric, value)

    # FPS measurement
    logger.info("Measuring inference speed...")
    if len(val_infos) > 0:
        # Use first sample for FPS benchmark
        sample_pc_path = val_infos[0]["point_cloud_path"]
        sample_points = np.fromfile(sample_pc_path, dtype=np.float32).reshape(-1, 4)
        sample_pillars, sample_coords, sample_num_pts = voxelize_point_cloud(
            sample_points, config
        )

        fps_results = measure_fps(
            model,
            sample_pillars,
            sample_coords,
            sample_num_pts,
            warmup_iterations=config.warmup_iterations,
            benchmark_iterations=config.fps_iterations,
            is_saved_model=is_saved_model,
        )
        results["performance"] = fps_results

        logger.info("=== Performance ===")
        logger.info("FPS:            %.1f", fps_results["fps"])
        logger.info("Mean latency:   %.2f ms", fps_results["mean_latency_ms"])
        logger.info("Std latency:    %.2f ms", fps_results["std_latency_ms"])
        logger.info("P50 latency:    %.2f ms", fps_results["p50_latency_ms"])
        logger.info("P95 latency:    %.2f ms", fps_results["p95_latency_ms"])
    else:
        results["performance"] = {"fps": 0.0, "error": "No validation samples found"}

    # Overall inference time statistics
    if inference_times:
        results["inference_stats"] = {
            "total_samples": len(inference_times),
            "total_time_s": float(np.sum(inference_times)),
            "mean_time_ms": float(np.mean(inference_times)) * 1000.0,
            "throughput_fps": len(inference_times) / max(float(np.sum(inference_times)), 1e-9),
        }

    return results


# =============================================================================
# CLI Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="PointPillars TF2 Evaluation - KITTI 3D AP & nuScenes mAP/NDS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model loading
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to TF2 checkpoint (e.g., ./output/checkpoints/ckpt-32).",
    )
    model_group.add_argument(
        "--saved_model",
        type=str,
        help="Path to SavedModel directory.",
    )

    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/kitti",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--val_info",
        type=str,
        default="kitti_infos_val.pkl",
        help="Validation info pickle file (relative to data_root).",
    )

    # Metrics
    parser.add_argument(
        "--metrics",
        type=str,
        choices=["kitti", "nuscenes", "both"],
        default="kitti",
        help="Which metric suite to compute.",
    )

    # Detection parameters
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.1,
        help="Minimum detection score threshold.",
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--max_detections",
        type=int,
        default=100,
        help="Maximum detections per class.",
    )

    # Model architecture
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Number of detection classes.",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=["Car", "Pedestrian", "Cyclist"],
        help="Class names.",
    )

    # FPS
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Number of warmup iterations for FPS measurement.",
    )
    parser.add_argument(
        "--fps_iterations",
        type=int,
        default=100,
        help="Number of iterations for FPS measurement.",
    )

    # Output
    parser.add_argument(
        "--output_json",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results as JSON.",
    )

    # GPU
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU ID to use (e.g., '0'). Default uses first available.",
    )

    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point.

    Parses arguments, configures GPU and logging, runs evaluation,
    prints summary, and saves results to JSON.
    """
    args = parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # GPU setup
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning("Could not set memory growth: %s", e)

    logger.info("TensorFlow version: %s", tf.__version__)
    logger.info("GPUs available: %d", len(gpus))

    # Build eval config from args
    config = EvalConfig(
        checkpoint_path=args.checkpoint,
        saved_model_path=args.saved_model,
        data_root=args.data_root,
        val_info_path=args.val_info,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections_per_class=args.max_detections,
        num_classes=args.num_classes,
        class_names=args.class_names,
        warmup_iterations=args.warmup_iterations,
        fps_iterations=args.fps_iterations,
        metrics_type=args.metrics,
        output_json=args.output_json,
    )

    # Run evaluation
    results = run_evaluation(config)

    # Save results to JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else None)

    logger.info("Results saved to: %s", output_path.resolve())
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
