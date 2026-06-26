"""
CenterPoint 3D LiDAR Object Detection - TensorFlow 2 Evaluation Script
========================================================================
Evaluates a trained CenterPoint model on the nuScenes validation set.
Computes mAP (center-distance matching) and NDS (nuScenes Detection Score).

Reference: "Center-based 3D Object Detection and Tracking" (Yin et al., CVPR 2021)
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ==============================================================================
# Configuration Constants
# ==============================================================================

VOXEL_SIZE = [0.075, 0.075, 0.2]
POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
BEV_RESOLUTION = [180, 180]
BEV_PIXEL_SIZE = 0.6  # meters per pixel

NUSCENES_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

NUSCENES_CLASS_GROUPS = [
    ["car"],
    ["truck", "construction_vehicle"],
    ["bus", "trailer"],
    ["barrier"],
    ["motorcycle", "bicycle"],
    ["pedestrian", "traffic_cone"],
]

# nuScenes mAP distance thresholds in meters
DISTANCE_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]

# Per-class distance thresholds for NMS (circle NMS radii)
CLASS_NMS_RADII = {
    "car": 4.0,
    "truck": 6.0,
    "construction_vehicle": 6.0,
    "bus": 8.0,
    "trailer": 8.0,
    "barrier": 2.0,
    "motorcycle": 2.5,
    "bicycle": 2.5,
    "pedestrian": 2.0,
    "traffic_cone": 1.5,
}

# Velocity and attribute defaults
DEFAULT_VELOCITY = [0.0, 0.0]
DEFAULT_ATTRIBUTE = 0


# ==============================================================================
# Voxelization
# ==============================================================================


def create_voxel_grid(
    points: np.ndarray,
    voxel_size: List[float],
    point_cloud_range: List[float],
    max_points_per_voxel: int = 10,
    max_voxels: int = 60000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw point cloud to voxel representation.

    Args:
        points: (N, 5) array with columns [x, y, z, intensity, timestamp].
        voxel_size: [vx, vy, vz] voxel dimensions in meters.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        max_points_per_voxel: Maximum number of points per voxel.
        max_voxels: Maximum total number of non-empty voxels.

    Returns:
        voxels: (M, max_points_per_voxel, C) padded point features per voxel.
        coordinates: (M, 3) voxel grid indices [z_idx, y_idx, x_idx].
        num_points_per_voxel: (M,) actual point count per voxel.
    """
    pcr = point_cloud_range
    vs = voxel_size

    grid_size = np.array(
        [
            int(round((pcr[3] - pcr[0]) / vs[0])),
            int(round((pcr[4] - pcr[1]) / vs[1])),
            int(round((pcr[5] - pcr[2]) / vs[2])),
        ],
        dtype=np.int32,
    )

    # Filter points outside range
    mask = (
        (points[:, 0] >= pcr[0])
        & (points[:, 0] < pcr[3])
        & (points[:, 1] >= pcr[1])
        & (points[:, 1] < pcr[4])
        & (points[:, 2] >= pcr[2])
        & (points[:, 2] < pcr[5])
    )
    points = points[mask]

    # Compute voxel indices
    voxel_indices = np.floor(
        (points[:, :3] - np.array(pcr[:3])) / np.array(vs)
    ).astype(np.int32)

    # Clip to grid boundaries
    voxel_indices[:, 0] = np.clip(voxel_indices[:, 0], 0, grid_size[0] - 1)
    voxel_indices[:, 1] = np.clip(voxel_indices[:, 1], 0, grid_size[1] - 1)
    voxel_indices[:, 2] = np.clip(voxel_indices[:, 2], 0, grid_size[2] - 1)

    # Hash voxel coordinates for grouping
    voxel_hash = (
        voxel_indices[:, 2] * grid_size[1] * grid_size[0]
        + voxel_indices[:, 1] * grid_size[0]
        + voxel_indices[:, 0]
    )

    # Get unique voxels
    unique_hashes, inverse_indices = np.unique(voxel_hash, return_inverse=True)
    num_voxels = min(len(unique_hashes), max_voxels)

    num_features = points.shape[1]
    voxels = np.zeros((num_voxels, max_points_per_voxel, num_features), dtype=np.float32)
    coordinates = np.zeros((num_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(num_voxels, dtype=np.int32)

    for i in range(num_voxels):
        voxel_mask = inverse_indices == i
        pts_in_voxel = points[voxel_mask]
        n_pts = min(len(pts_in_voxel), max_points_per_voxel)

        voxels[i, :n_pts, :] = pts_in_voxel[:n_pts]
        num_points_per_voxel[i] = n_pts

        # Store coordinate as [z, y, x] for sparse conv format
        idx = voxel_indices[voxel_mask][0]
        coordinates[i] = [idx[2], idx[1], idx[0]]

    return voxels, coordinates, num_points_per_voxel


# ==============================================================================
# Circle NMS (nuScenes-style)
# ==============================================================================


def circle_nms(
    detections: np.ndarray, scores: np.ndarray, radius: float
) -> np.ndarray:
    """
    Apply circle-based NMS for nuScenes evaluation.

    Unlike axis-aligned box NMS, circle NMS suppresses detections whose
    2D center distance is within a given radius.

    Args:
        detections: (N, 2) array of 2D center positions [x, y].
        scores: (N,) confidence scores.
        radius: Suppression radius in meters.

    Returns:
        keep_indices: Array of indices to keep.
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

        # Compute distances from current detection to all remaining
        dists = np.sqrt(
            np.sum((detections[order[i + 1 :]] - detections[idx]) ** 2, axis=1)
        )
        suppress_mask = dists < radius
        suppressed[order[i + 1 :][suppress_mask]] = True

    return np.array(keep, dtype=np.int64)


# ==============================================================================
# Post-Processing: Decode CenterPoint Outputs
# ==============================================================================


def decode_centerpoint_output(
    heatmap: np.ndarray,
    offset: np.ndarray,
    height: np.ndarray,
    dim: np.ndarray,
    rotation: np.ndarray,
    velocity: Optional[np.ndarray],
    score_threshold: float,
    nms_radius_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Decode CenterPoint dense prediction maps into 3D bounding boxes.

    Args:
        heatmap: (H, W, num_classes) class heatmap after sigmoid.
        offset: (H, W, 2) sub-pixel center offset [dx, dy].
        height: (H, W, 1) center height above ground.
        dim: (H, W, 3) box dimensions [w, l, h] in meters.
        rotation: (H, W, 2) rotation as [sin, cos].
        velocity: (H, W, 2) velocity [vx, vy] or None.
        score_threshold: Minimum confidence score.
        nms_radius_override: If set, use this radius for all classes.

    Returns:
        List of detection dicts with keys:
            center_2d, center_3d, dimensions, rotation_angle,
            score, class_id, class_name, velocity
    """
    H, W, num_classes = heatmap.shape
    detections = []

    for cls_id in range(num_classes):
        cls_heatmap = heatmap[:, :, cls_id]

        # Find local peaks (3x3 max pooling then compare)
        padded = np.pad(cls_heatmap, 1, mode="constant", constant_values=0)
        local_max = np.zeros_like(cls_heatmap)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                local_max = np.maximum(
                    local_max, padded[1 + dy : H + 1 + dy, 1 + dx : W + 1 + dx]
                )
        peak_mask = (cls_heatmap == local_max) & (cls_heatmap >= score_threshold)

        ys, xs = np.where(peak_mask)
        scores = cls_heatmap[ys, xs]

        cls_name = NUSCENES_CLASSES[cls_id] if cls_id < len(NUSCENES_CLASSES) else f"class_{cls_id}"
        nms_radius = nms_radius_override if nms_radius_override else CLASS_NMS_RADII.get(cls_name, 4.0)

        if len(scores) == 0:
            continue

        # Decode centers in BEV coordinates
        center_x = (xs.astype(np.float32) + offset[ys, xs, 0]) * BEV_PIXEL_SIZE + POINT_CLOUD_RANGE[0]
        center_y = (ys.astype(np.float32) + offset[ys, xs, 1]) * BEV_PIXEL_SIZE + POINT_CLOUD_RANGE[1]
        center_z = height[ys, xs, 0]

        # Decode dimensions
        w = dim[ys, xs, 0]
        l = dim[ys, xs, 1]
        h = dim[ys, xs, 2]

        # Decode rotation
        sin_rot = rotation[ys, xs, 0]
        cos_rot = rotation[ys, xs, 1]
        rot_angle = np.arctan2(sin_rot, cos_rot)

        # Decode velocity
        if velocity is not None:
            vx = velocity[ys, xs, 0]
            vy = velocity[ys, xs, 1]
        else:
            vx = np.zeros_like(scores)
            vy = np.zeros_like(scores)

        # Apply circle NMS
        centers_2d = np.stack([center_x, center_y], axis=1)
        keep = circle_nms(centers_2d, scores, nms_radius)

        for k in keep:
            detections.append(
                {
                    "center_2d": [float(center_x[k]), float(center_y[k])],
                    "center_3d": [float(center_x[k]), float(center_y[k]), float(center_z[k])],
                    "dimensions": [float(w[k]), float(l[k]), float(h[k])],
                    "rotation_angle": float(rot_angle[k]),
                    "score": float(scores[k]),
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "velocity": [float(vx[k]), float(vy[k])],
                }
            )

    return detections


# ==============================================================================
# nuScenes Evaluation Metrics
# ==============================================================================


def compute_center_distance_matrix(
    pred_centers: np.ndarray, gt_centers: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise 2D center distance matrix.

    Args:
        pred_centers: (M, 2) predicted centers.
        gt_centers: (N, 2) ground truth centers.

    Returns:
        dist_matrix: (M, N) distance matrix.
    """
    diff = pred_centers[:, None, :] - gt_centers[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def compute_ap_at_threshold(
    pred_scores: np.ndarray,
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    distance_threshold: float,
) -> float:
    """
    Compute Average Precision at a single distance threshold.

    Uses Hungarian matching: predictions are sorted by score descending,
    and matched greedily to ground truth within the distance threshold.

    Args:
        pred_scores: (M,) prediction confidence scores.
        pred_centers: (M, 2) predicted 2D centers.
        gt_centers: (N, 2) ground truth 2D centers.
        distance_threshold: Maximum center distance for a valid match.

    Returns:
        AP value (float between 0 and 1).
    """
    if len(gt_centers) == 0:
        return 1.0 if len(pred_centers) == 0 else 0.0
    if len(pred_centers) == 0:
        return 0.0

    num_gt = len(gt_centers)
    num_pred = len(pred_centers)

    # Sort predictions by descending score
    sorted_indices = np.argsort(-pred_scores)
    pred_centers_sorted = pred_centers[sorted_indices]

    gt_matched = np.zeros(num_gt, dtype=bool)
    tp = np.zeros(num_pred, dtype=bool)

    for i in range(num_pred):
        distances = np.sqrt(
            np.sum((gt_centers - pred_centers_sorted[i]) ** 2, axis=1)
        )
        # Mask already matched GT
        distances[gt_matched] = np.inf

        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist <= distance_threshold:
            tp[i] = True
            gt_matched[min_dist_idx] = True

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp).astype(np.float64)
    fp_cumsum = np.cumsum(~tp).astype(np.float64)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gt

    # Append sentinel values for AP computation
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute AP as area under the precision-recall curve (11-point interpolation)
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in recall_thresholds:
        prec_at_recall = precision[recall >= t]
        if len(prec_at_recall) > 0:
            ap += np.max(prec_at_recall)
    ap /= len(recall_thresholds)

    return float(ap)


def compute_translation_error(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    pred_scores: np.ndarray,
    distance_threshold: float = 2.0,
) -> float:
    """
    Compute mean Average Translation Error (mATE).

    ATE is the Euclidean center distance (2D) for matched detections.

    Args:
        pred_centers: (M, 3) predicted 3D centers.
        gt_centers: (N, 3) ground truth 3D centers.
        pred_scores: (M,) scores for sorting.
        distance_threshold: Max 2D distance for matching.

    Returns:
        Mean translation error for matched pairs, or 1.0 if no matches.
    """
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return 1.0

    sorted_indices = np.argsort(-pred_scores)
    pred_sorted = pred_centers[sorted_indices]

    gt_matched = np.zeros(len(gt_centers), dtype=bool)
    errors = []

    for i in range(len(pred_sorted)):
        dists_2d = np.sqrt(
            np.sum((gt_centers[:, :2] - pred_sorted[i, :2]) ** 2, axis=1)
        )
        dists_2d[gt_matched] = np.inf
        min_idx = np.argmin(dists_2d)

        if dists_2d[min_idx] <= distance_threshold:
            # 3D Euclidean distance as translation error
            error_3d = np.sqrt(np.sum((gt_centers[min_idx] - pred_sorted[i]) ** 2))
            errors.append(error_3d)
            gt_matched[min_idx] = True

    return float(np.mean(errors)) if errors else 1.0


def compute_scale_error(
    pred_dims: np.ndarray,
    gt_dims: np.ndarray,
    pred_centers_2d: np.ndarray,
    gt_centers_2d: np.ndarray,
    pred_scores: np.ndarray,
    distance_threshold: float = 2.0,
) -> float:
    """
    Compute mean Average Scale Error (mASE).

    ASE = 1 - IoU(3D) for matched detections (using axis-aligned volume overlap
    approximation as 1 - min(vol_pred, vol_gt) / max(vol_pred, vol_gt)).

    Args:
        pred_dims: (M, 3) predicted dimensions [w, l, h].
        gt_dims: (N, 3) ground truth dimensions [w, l, h].
        pred_centers_2d: (M, 2) predicted 2D centers for matching.
        gt_centers_2d: (N, 2) ground truth 2D centers for matching.
        pred_scores: (M,) scores for sorting.
        distance_threshold: Max 2D distance for matching.

    Returns:
        Mean scale error, or 1.0 if no matches.
    """
    if len(pred_dims) == 0 or len(gt_dims) == 0:
        return 1.0

    sorted_indices = np.argsort(-pred_scores)
    pred_dims_sorted = pred_dims[sorted_indices]
    pred_centers_sorted = pred_centers_2d[sorted_indices]

    gt_matched = np.zeros(len(gt_centers_2d), dtype=bool)
    errors = []

    for i in range(len(pred_dims_sorted)):
        dists = np.sqrt(
            np.sum((gt_centers_2d - pred_centers_sorted[i]) ** 2, axis=1)
        )
        dists[gt_matched] = np.inf
        min_idx = np.argmin(dists)

        if dists[min_idx] <= distance_threshold:
            # Scale error: 1 - IoU (approximated by volume ratio)
            vol_pred = np.prod(np.maximum(pred_dims_sorted[i], 1e-6))
            vol_gt = np.prod(np.maximum(gt_dims[min_idx], 1e-6))
            iou_approx = min(vol_pred, vol_gt) / max(vol_pred, vol_gt)
            errors.append(1.0 - iou_approx)
            gt_matched[min_idx] = True

    return float(np.mean(errors)) if errors else 1.0


def compute_orientation_error(
    pred_angles: np.ndarray,
    gt_angles: np.ndarray,
    pred_centers_2d: np.ndarray,
    gt_centers_2d: np.ndarray,
    pred_scores: np.ndarray,
    distance_threshold: float = 2.0,
) -> float:
    """
    Compute mean Average Orientation Error (mAOE).

    AOE is the smallest yaw angle difference between prediction and GT.

    Args:
        pred_angles: (M,) predicted yaw angles in radians.
        gt_angles: (N,) ground truth yaw angles in radians.
        pred_centers_2d: (M, 2) predicted 2D centers for matching.
        gt_centers_2d: (N, 2) ground truth 2D centers for matching.
        pred_scores: (M,) scores for sorting.
        distance_threshold: Max 2D distance for matching.

    Returns:
        Mean orientation error in radians, or pi if no matches.
    """
    if len(pred_angles) == 0 or len(gt_angles) == 0:
        return float(np.pi)

    sorted_indices = np.argsort(-pred_scores)
    pred_angles_sorted = pred_angles[sorted_indices]
    pred_centers_sorted = pred_centers_2d[sorted_indices]

    gt_matched = np.zeros(len(gt_centers_2d), dtype=bool)
    errors = []

    for i in range(len(pred_angles_sorted)):
        dists = np.sqrt(
            np.sum((gt_centers_2d - pred_centers_sorted[i]) ** 2, axis=1)
        )
        dists[gt_matched] = np.inf
        min_idx = np.argmin(dists)

        if dists[min_idx] <= distance_threshold:
            # Orientation error: smallest angle difference
            diff = np.abs(pred_angles_sorted[i] - gt_angles[min_idx])
            diff = np.minimum(diff, 2 * np.pi - diff)
            # For objects with 180-degree symmetry (e.g., barrier), use pi symmetry
            diff = np.minimum(diff, np.pi - diff) if diff > np.pi / 2 else diff
            errors.append(diff)
            gt_matched[min_idx] = True

    return float(np.mean(errors)) if errors else float(np.pi)


def compute_velocity_error(
    pred_velocities: np.ndarray,
    gt_velocities: np.ndarray,
    pred_centers_2d: np.ndarray,
    gt_centers_2d: np.ndarray,
    pred_scores: np.ndarray,
    distance_threshold: float = 2.0,
) -> float:
    """
    Compute mean Average Velocity Error (mAVE).

    AVE is the L2 norm of the velocity difference for matched detections.

    Args:
        pred_velocities: (M, 2) predicted velocities [vx, vy].
        gt_velocities: (N, 2) ground truth velocities [vx, vy].
        pred_centers_2d: (M, 2) predicted 2D centers for matching.
        gt_centers_2d: (N, 2) ground truth 2D centers for matching.
        pred_scores: (M,) scores for sorting.
        distance_threshold: Max 2D distance for matching.

    Returns:
        Mean velocity error, or 1.0 if no matches.
    """
    if len(pred_velocities) == 0 or len(gt_velocities) == 0:
        return 1.0

    sorted_indices = np.argsort(-pred_scores)
    pred_vel_sorted = pred_velocities[sorted_indices]
    pred_centers_sorted = pred_centers_2d[sorted_indices]

    gt_matched = np.zeros(len(gt_centers_2d), dtype=bool)
    errors = []

    for i in range(len(pred_vel_sorted)):
        dists = np.sqrt(
            np.sum((gt_centers_2d - pred_centers_sorted[i]) ** 2, axis=1)
        )
        dists[gt_matched] = np.inf
        min_idx = np.argmin(dists)

        if dists[min_idx] <= distance_threshold:
            vel_error = np.sqrt(
                np.sum((pred_vel_sorted[i] - gt_velocities[min_idx]) ** 2)
            )
            errors.append(vel_error)
            gt_matched[min_idx] = True

    return float(np.mean(errors)) if errors else 1.0


def compute_attribute_error(
    pred_attributes: np.ndarray,
    gt_attributes: np.ndarray,
    pred_centers_2d: np.ndarray,
    gt_centers_2d: np.ndarray,
    pred_scores: np.ndarray,
    distance_threshold: float = 2.0,
) -> float:
    """
    Compute mean Average Attribute Error (mAAE).

    AAE is 1 - accuracy of attribute classification for matched detections.

    Args:
        pred_attributes: (M,) predicted attribute indices.
        gt_attributes: (N,) ground truth attribute indices.
        pred_centers_2d: (M, 2) predicted 2D centers for matching.
        gt_centers_2d: (N, 2) ground truth 2D centers for matching.
        pred_scores: (M,) scores for sorting.
        distance_threshold: Max 2D distance for matching.

    Returns:
        Mean attribute error (1 - accuracy), or 1.0 if no matches.
    """
    if len(pred_attributes) == 0 or len(gt_attributes) == 0:
        return 1.0

    sorted_indices = np.argsort(-pred_scores)
    pred_attr_sorted = pred_attributes[sorted_indices]
    pred_centers_sorted = pred_centers_2d[sorted_indices]

    gt_matched = np.zeros(len(gt_centers_2d), dtype=bool)
    correct = 0
    total = 0

    for i in range(len(pred_attr_sorted)):
        dists = np.sqrt(
            np.sum((gt_centers_2d - pred_centers_sorted[i]) ** 2, axis=1)
        )
        dists[gt_matched] = np.inf
        min_idx = np.argmin(dists)

        if dists[min_idx] <= distance_threshold:
            if pred_attr_sorted[i] == gt_attributes[min_idx]:
                correct += 1
            total += 1
            gt_matched[min_idx] = True

    if total == 0:
        return 1.0
    return 1.0 - (correct / total)


def compute_nds(mAP: float, mtp_metrics: Dict[str, float]) -> float:
    """
    Compute nuScenes Detection Score (NDS).

    NDS = (5 * mAP + sum(max(1 - mTP, 0) for mTP in [mATE, mASE, mAOE, mAVE, mAAE])) / 10

    The mTP values are errors, so we convert to "true positive metrics" as (1 - error)
    clamped to [0, 1].

    Args:
        mAP: Mean Average Precision.
        mtp_metrics: Dict with keys 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE'.

    Returns:
        NDS value.
    """
    tp_scores = sum(
        max(1.0 - v, 0.0) for v in mtp_metrics.values()
    )
    nds = (5.0 * mAP + tp_scores) / 10.0
    return nds


# ==============================================================================
# Data Loading
# ==============================================================================


def load_ground_truth(data_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load ground truth annotations from nuScenes-format JSON or directory.

    Expected structure (JSON):
    {
        "sample_token_1": [
            {
                "center": [x, y, z],
                "dimensions": [w, l, h],
                "rotation": angle_rad,
                "class_name": "car",
                "velocity": [vx, vy],
                "attribute": 0
            },
            ...
        ],
        ...
    }

    Or as a directory of per-sample .json files.

    Args:
        data_path: Path to ground truth annotations.

    Returns:
        Dict mapping sample tokens to lists of GT annotations.
    """
    gt_path = Path(data_path)

    if gt_path.is_file() and gt_path.suffix == ".json":
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        return gt_data

    # Try loading from a directory structure
    gt_dir = gt_path / "gt_annotations"
    if not gt_dir.exists():
        gt_dir = gt_path / "annotations"
    if not gt_dir.exists():
        gt_dir = gt_path

    gt_data = {}
    json_files = sorted(gt_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No ground truth JSON files found in {gt_dir}. "
            "Expected either a single JSON file or directory of per-sample JSONs."
        )

    for jf in json_files:
        sample_token = jf.stem
        with open(jf, "r") as f:
            gt_data[sample_token] = json.load(f)

    return gt_data


def load_point_cloud(file_path: str) -> np.ndarray:
    """
    Load a point cloud from binary (.bin) or numpy (.npy) file.

    Args:
        file_path: Path to point cloud file.

    Returns:
        points: (N, 5) array [x, y, z, intensity, timestamp].
                If fewer features exist, pads with zeros.
    """
    path = Path(file_path)
    if path.suffix == ".bin":
        points = np.fromfile(str(path), dtype=np.float32)
        # nuScenes uses 5 features per point; KITTI uses 4
        if len(points) % 5 == 0:
            points = points.reshape(-1, 5)
        elif len(points) % 4 == 0:
            points = points.reshape(-1, 4)
            # Pad with zero timestamp
            points = np.hstack([points, np.zeros((len(points), 1), dtype=np.float32)])
        else:
            raise ValueError(
                f"Cannot reshape point cloud of size {len(points)} into (N, 4) or (N, 5)"
            )
    elif path.suffix == ".npy":
        points = np.load(str(path)).astype(np.float32)
        if points.ndim == 1:
            if len(points) % 5 == 0:
                points = points.reshape(-1, 5)
            else:
                points = points.reshape(-1, 4)
                points = np.hstack([points, np.zeros((len(points), 1), dtype=np.float32)])
        if points.shape[1] < 5:
            pad = np.zeros((len(points), 5 - points.shape[1]), dtype=np.float32)
            points = np.hstack([points, pad])
    else:
        raise ValueError(f"Unsupported point cloud format: {path.suffix}")

    return points


def get_sample_list(data_path: str) -> List[Dict[str, str]]:
    """
    Get list of samples to evaluate.

    Looks for an info file (val_infos.json or similar) listing samples,
    or falls back to discovering point cloud files in the data directory.

    Args:
        data_path: Path to dataset root.

    Returns:
        List of dicts with 'token' and 'lidar_path' keys.
    """
    data_dir = Path(data_path)

    # Try standard info files
    info_candidates = [
        data_dir / "val_infos.json",
        data_dir / "infos_val.json",
        data_dir / "nuscenes_infos_val.json",
        data_dir / "val.json",
    ]

    for info_file in info_candidates:
        if info_file.exists():
            with open(info_file, "r") as f:
                infos = json.load(f)
            samples = []
            if isinstance(infos, list):
                for info in infos:
                    token = info.get("token", info.get("sample_token", ""))
                    lidar_path = info.get(
                        "lidar_path",
                        info.get("pts_filename", info.get("point_cloud_path", "")),
                    )
                    if not os.path.isabs(lidar_path):
                        lidar_path = str(data_dir / lidar_path)
                    samples.append({"token": token, "lidar_path": lidar_path})
            elif isinstance(infos, dict) and "infos" in infos:
                for info in infos["infos"]:
                    token = info.get("token", info.get("sample_token", ""))
                    lidar_path = info.get(
                        "lidar_path",
                        info.get("pts_filename", info.get("point_cloud_path", "")),
                    )
                    if not os.path.isabs(lidar_path):
                        lidar_path = str(data_dir / lidar_path)
                    samples.append({"token": token, "lidar_path": lidar_path})
            return samples

    # Fallback: discover point cloud files
    lidar_dirs = [
        data_dir / "samples" / "LIDAR_TOP",
        data_dir / "lidar",
        data_dir / "velodyne",
        data_dir,
    ]

    for lidar_dir in lidar_dirs:
        if lidar_dir.exists():
            bin_files = sorted(lidar_dir.glob("*.bin")) + sorted(lidar_dir.glob("*.npy"))
            if bin_files:
                samples = []
                for bf in bin_files:
                    samples.append({"token": bf.stem, "lidar_path": str(bf)})
                return samples

    raise FileNotFoundError(
        f"No validation samples found in {data_path}. "
        "Provide a val_infos.json or a directory with .bin/.npy point clouds."
    )


# ==============================================================================
# Model Loading and Inference
# ==============================================================================


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained CenterPoint model from SavedModel or checkpoint.

    Args:
        model_path: Path to SavedModel directory or checkpoint prefix.

    Returns:
        Loaded TensorFlow model ready for inference.
    """
    model_dir = Path(model_path)

    # Try SavedModel format first
    if (model_dir / "saved_model.pb").exists() or model_dir.suffix == "":
        try:
            model = tf.saved_model.load(str(model_dir))
            print(f"[INFO] Loaded SavedModel from {model_dir}")
            return model
        except Exception as e:
            print(f"[WARN] SavedModel load failed: {e}")

    # Try Keras format
    keras_candidates = [
        model_dir,
        model_dir.with_suffix(".keras"),
        model_dir.with_suffix(".h5"),
    ]
    for kp in keras_candidates:
        if kp.exists():
            try:
                model = tf.keras.models.load_model(str(kp), compile=False)
                print(f"[INFO] Loaded Keras model from {kp}")
                return model
            except Exception as e:
                print(f"[WARN] Keras model load failed from {kp}: {e}")

    # Try checkpoint format
    checkpoint_dir = model_dir if model_dir.is_dir() else model_dir.parent
    ckpt = tf.train.latest_checkpoint(str(checkpoint_dir))
    if ckpt is not None:
        # Build model architecture and restore weights
        model = build_centerpoint_model()
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(ckpt)
        status.expect_partial()
        print(f"[INFO] Restored checkpoint from {ckpt}")
        return model

    raise FileNotFoundError(
        f"Could not load model from {model_path}. "
        "Expected SavedModel directory, .keras/.h5 file, or checkpoint."
    )


def build_centerpoint_model() -> tf.keras.Model:
    """
    Build the CenterPoint model architecture for checkpoint restoration.

    This creates the full architecture matching the training configuration:
    - Pillar Feature Net (simplified Mean VFE)
    - Sparse 2D backbone with BEV feature aggregation
    - Detection head with multi-task outputs

    Returns:
        CenterPoint Keras model.
    """
    num_classes = len(NUSCENES_CLASSES)
    bev_h, bev_w = BEV_RESOLUTION

    # Input: BEV feature map (from voxelization + pillar encoding)
    bev_input = tf.keras.Input(shape=(bev_h, bev_w, 64), name="bev_features")

    # Backbone: series of conv blocks
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(bev_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for _ in range(5):
        x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Shared feature map
    shared = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    shared = tf.keras.layers.BatchNormalization()(shared)
    shared = tf.keras.layers.ReLU()(shared)

    # Detection heads
    heatmap = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(shared)
    heatmap = tf.keras.layers.BatchNormalization()(heatmap)
    heatmap = tf.keras.layers.ReLU()(heatmap)
    heatmap = tf.keras.layers.Conv2D(num_classes, 1, padding="same", name="heatmap")(heatmap)

    offset = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(shared)
    offset = tf.keras.layers.BatchNormalization()(offset)
    offset = tf.keras.layers.ReLU()(offset)
    offset = tf.keras.layers.Conv2D(2, 1, padding="same", name="offset")(offset)

    height = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(shared)
    height = tf.keras.layers.BatchNormalization()(height)
    height = tf.keras.layers.ReLU()(height)
    height = tf.keras.layers.Conv2D(1, 1, padding="same", name="height")(height)

    dim = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(shared)
    dim = tf.keras.layers.BatchNormalization()(dim)
    dim = tf.keras.layers.ReLU()(dim)
    dim = tf.keras.layers.Conv2D(3, 1, padding="same", name="dim")(dim)

    rotation = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(shared)
    rotation = tf.keras.layers.BatchNormalization()(rotation)
    rotation = tf.keras.layers.ReLU()(rotation)
    rotation = tf.keras.layers.Conv2D(2, 1, padding="same", name="rotation")(rotation)

    velocity = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(shared)
    velocity = tf.keras.layers.BatchNormalization()(velocity)
    velocity = tf.keras.layers.ReLU()(velocity)
    velocity = tf.keras.layers.Conv2D(2, 1, padding="same", name="velocity")(velocity)

    model = tf.keras.Model(
        inputs=bev_input,
        outputs={
            "heatmap": heatmap,
            "offset": offset,
            "height": height,
            "dim": dim,
            "rotation": rotation,
            "velocity": velocity,
        },
        name="centerpoint",
    )

    return model


def pillar_encode(
    voxels: np.ndarray,
    coordinates: np.ndarray,
    num_points_per_voxel: np.ndarray,
) -> np.ndarray:
    """
    Encode voxels into a BEV pseudo-image using pillar mean encoding.

    Computes per-pillar mean features and scatters them onto the BEV grid.

    Args:
        voxels: (M, max_points, C) point features per voxel.
        coordinates: (M, 3) voxel indices [z, y, x].
        num_points_per_voxel: (M,) point counts.

    Returns:
        bev_map: (1, BEV_H, BEV_W, 64) pseudo-image feature tensor.
    """
    bev_h, bev_w = BEV_RESOLUTION
    num_features = voxels.shape[2]

    # Compute mean features per pillar (collapse z axis)
    pillar_features = np.zeros((len(voxels), num_features), dtype=np.float32)
    for i in range(len(voxels)):
        n = num_points_per_voxel[i]
        if n > 0:
            pillar_features[i] = voxels[i, :n, :].mean(axis=0)

    # Zero-pad to 64 features if needed
    if num_features < 64:
        pad = np.zeros((len(pillar_features), 64 - num_features), dtype=np.float32)
        pillar_features = np.concatenate([pillar_features, pad], axis=1)
    elif num_features > 64:
        pillar_features = pillar_features[:, :64]

    # Scatter to BEV grid (collapse z, use x and y indices)
    # coordinates are [z, y, x] format
    bev_map = np.zeros((bev_h, bev_w, 64), dtype=np.float32)

    # Map voxel coordinates to BEV pixels
    # BEV pixel = voxel_xy_index * voxel_size / bev_pixel_size
    voxel_x = coordinates[:, 2]  # x index in voxel grid
    voxel_y = coordinates[:, 1]  # y index in voxel grid

    # Convert voxel indices to BEV pixel indices
    grid_size_x = int(round((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]))
    grid_size_y = int(round((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]))

    bev_x = (voxel_x.astype(np.float32) * VOXEL_SIZE[0] / BEV_PIXEL_SIZE).astype(np.int32)
    bev_y = (voxel_y.astype(np.float32) * VOXEL_SIZE[1] / BEV_PIXEL_SIZE).astype(np.int32)

    # Clip to BEV bounds
    bev_x = np.clip(bev_x, 0, bev_w - 1)
    bev_y = np.clip(bev_y, 0, bev_h - 1)

    # Scatter (last write wins for overlapping pillars)
    for i in range(len(pillar_features)):
        bev_map[bev_y[i], bev_x[i]] = pillar_features[i]

    return bev_map[np.newaxis, ...]  # Add batch dimension


def run_inference(
    model: Any,
    bev_features: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Run model inference on BEV features.

    Handles both SavedModel (with __call__ or signatures) and Keras model.

    Args:
        model: Loaded TF model.
        bev_features: (1, H, W, C) BEV feature tensor.

    Returns:
        Dict with prediction maps: heatmap, offset, height, dim, rotation, velocity.
    """
    input_tensor = tf.constant(bev_features, dtype=tf.float32)

    # Try Keras model predict
    if isinstance(model, tf.keras.Model):
        outputs = model(input_tensor, training=False)
        if isinstance(outputs, dict):
            return {k: v.numpy() for k, v in outputs.items()}
        else:
            # Assume ordered outputs
            keys = ["heatmap", "offset", "height", "dim", "rotation", "velocity"]
            if isinstance(outputs, (list, tuple)):
                return {k: v.numpy() for k, v in zip(keys, outputs)}
            return {"heatmap": outputs.numpy()}

    # Try SavedModel signatures
    if hasattr(model, "signatures"):
        serve_fn = model.signatures.get("serving_default", None)
        if serve_fn is not None:
            result = serve_fn(input_tensor)
            return {k: v.numpy() for k, v in result.items()}

    # Try direct call
    if callable(model):
        result = model(input_tensor)
        if isinstance(result, dict):
            return {k: v.numpy() for k, v in result.items()}
        elif isinstance(result, (list, tuple)):
            keys = ["heatmap", "offset", "height", "dim", "rotation", "velocity"]
            return {k: v.numpy() for k, v in zip(keys, result)}
        return {"heatmap": result.numpy()}

    raise RuntimeError("Could not run inference: model format not recognized.")


# ==============================================================================
# Main Evaluation Loop
# ==============================================================================


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Main evaluation function.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Results dictionary containing mAP, NDS, per-class metrics.
    """
    print("=" * 70)
    print("CenterPoint TF2 Evaluation - nuScenes Validation")
    print("=" * 70)
    print(f"  Model path:       {args.model_path}")
    print(f"  Data path:        {args.data_path}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Score threshold:  {args.score_threshold}")
    print(f"  NMS radius:       {args.nms_radius if args.nms_radius else 'per-class default'}")
    print(f"  Max detections:   {args.max_detections}")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    model = load_model(args.model_path)

    # Load ground truth and sample list
    print("[2/4] Loading validation data...")
    gt_data = load_ground_truth(args.gt_path if args.gt_path else args.data_path)
    samples = get_sample_list(args.data_path)
    print(f"  Found {len(samples)} validation samples")
    print(f"  Found {len(gt_data)} ground truth annotations")

    # Filter to samples with GT
    if gt_data:
        available_tokens = set(gt_data.keys())
        samples = [s for s in samples if s["token"] in available_tokens]
        print(f"  Matched {len(samples)} samples with ground truth")

    if not samples:
        raise ValueError("No valid samples found for evaluation.")

    # Run inference
    print("[3/4] Running inference...")
    all_predictions = {}
    total_inference_time = 0.0
    total_preprocess_time = 0.0

    for i in tqdm(range(0, len(samples), args.batch_size), desc="Evaluating"):
        batch_samples = samples[i : i + args.batch_size]

        for sample in batch_samples:
            token = sample["token"]
            lidar_path = sample["lidar_path"]

            # Load and voxelize point cloud
            t0 = time.time()
            points = load_point_cloud(lidar_path)
            voxels, coordinates, num_points = create_voxel_grid(
                points,
                voxel_size=VOXEL_SIZE,
                point_cloud_range=POINT_CLOUD_RANGE,
                max_points_per_voxel=args.max_points_per_voxel,
                max_voxels=args.max_voxels,
            )

            # Encode to BEV
            bev_features = pillar_encode(voxels, coordinates, num_points)
            t1 = time.time()
            total_preprocess_time += t1 - t0

            # Model inference
            t2 = time.time()
            outputs = run_inference(model, bev_features)
            t3 = time.time()
            total_inference_time += t3 - t2

            # Decode predictions
            heatmap = outputs["heatmap"][0]  # Remove batch dim
            heatmap = 1.0 / (1.0 + np.exp(-heatmap))  # Sigmoid activation

            offset = outputs["offset"][0]
            height_map = outputs["height"][0]
            dim_map = outputs["dim"][0]
            dim_map = np.exp(dim_map)  # Exponential activation for dimensions
            rotation_map = outputs["rotation"][0]
            velocity_map = outputs.get("velocity", None)
            if velocity_map is not None:
                velocity_map = velocity_map[0]

            detections = decode_centerpoint_output(
                heatmap=heatmap,
                offset=offset,
                height=height_map,
                dim=dim_map,
                rotation=rotation_map,
                velocity=velocity_map,
                score_threshold=args.score_threshold,
                nms_radius_override=args.nms_radius,
            )

            # Limit max detections per sample
            if len(detections) > args.max_detections:
                detections = sorted(detections, key=lambda d: d["score"], reverse=True)
                detections = detections[: args.max_detections]

            all_predictions[token] = detections

    num_samples = len(samples)
    avg_inference = total_inference_time / max(num_samples, 1) * 1000
    avg_preprocess = total_preprocess_time / max(num_samples, 1) * 1000
    fps = num_samples / max(total_inference_time, 1e-6)

    print(f"\n  Preprocessing: {avg_preprocess:.1f} ms/sample")
    print(f"  Inference:     {avg_inference:.1f} ms/sample")
    print(f"  Throughput:    {fps:.1f} FPS (inference only)")
    print(f"  Total time:    {total_inference_time + total_preprocess_time:.1f} s")

    # Compute metrics
    print("\n[4/4] Computing nuScenes metrics...")
    results = compute_nuscenes_metrics(all_predictions, gt_data)

    return results


def compute_nuscenes_metrics(
    predictions: Dict[str, List[Dict[str, Any]]],
    ground_truth: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Compute full nuScenes evaluation metrics.

    Args:
        predictions: Dict mapping sample tokens to list of prediction dicts.
        ground_truth: Dict mapping sample tokens to list of GT annotation dicts.

    Returns:
        Results dictionary with mAP, NDS, per-class AP, and TP metrics.
    """
    # Organize predictions and GT by class
    per_class_preds = defaultdict(lambda: defaultdict(list))
    per_class_gt = defaultdict(lambda: defaultdict(list))

    for token, preds in predictions.items():
        for pred in preds:
            cls_name = pred["class_name"]
            per_class_preds[cls_name][token].append(pred)

    for token, gts in ground_truth.items():
        if isinstance(gts, list):
            for gt in gts:
                cls_name = gt.get("class_name", gt.get("detection_name", "unknown"))
                per_class_gt[cls_name][token].append(gt)
        elif isinstance(gts, dict):
            for cls_name, cls_gts in gts.items():
                if isinstance(cls_gts, list):
                    per_class_gt[cls_name][token] = cls_gts

    # Compute per-class AP at each distance threshold
    class_aps = {}
    class_tp_metrics = {}
    all_tokens = set(predictions.keys()) | set(ground_truth.keys())

    for cls_name in NUSCENES_CLASSES:
        aps_per_threshold = []

        # Gather all predictions and GT for this class across all samples
        all_pred_centers = []
        all_pred_scores = []
        all_pred_centers_3d = []
        all_pred_dims = []
        all_pred_angles = []
        all_pred_velocities = []
        all_pred_attributes = []

        all_gt_centers = []
        all_gt_centers_3d = []
        all_gt_dims = []
        all_gt_angles = []
        all_gt_velocities = []
        all_gt_attributes = []

        for token in all_tokens:
            # Predictions
            preds = per_class_preds[cls_name].get(token, [])
            for p in preds:
                all_pred_centers.append(p["center_2d"])
                all_pred_scores.append(p["score"])
                all_pred_centers_3d.append(p["center_3d"])
                all_pred_dims.append(p["dimensions"])
                all_pred_angles.append(p["rotation_angle"])
                all_pred_velocities.append(p.get("velocity", DEFAULT_VELOCITY))
                all_pred_attributes.append(p.get("attribute", DEFAULT_ATTRIBUTE))

            # Ground truth
            gts = per_class_gt[cls_name].get(token, [])
            for g in gts:
                center = g.get("center", g.get("translation", [0, 0, 0]))
                all_gt_centers.append(center[:2])
                all_gt_centers_3d.append(center[:3])
                all_gt_dims.append(g.get("dimensions", g.get("size", [1, 1, 1])))
                all_gt_angles.append(g.get("rotation", g.get("yaw", 0.0)))
                all_gt_velocities.append(g.get("velocity", DEFAULT_VELOCITY))
                all_gt_attributes.append(g.get("attribute", DEFAULT_ATTRIBUTE))

        pred_centers = np.array(all_pred_centers, dtype=np.float64) if all_pred_centers else np.zeros((0, 2))
        pred_scores = np.array(all_pred_scores, dtype=np.float64) if all_pred_scores else np.zeros(0)
        gt_centers = np.array(all_gt_centers, dtype=np.float64) if all_gt_centers else np.zeros((0, 2))

        pred_centers_3d = np.array(all_pred_centers_3d, dtype=np.float64) if all_pred_centers_3d else np.zeros((0, 3))
        gt_centers_3d = np.array(all_gt_centers_3d, dtype=np.float64) if all_gt_centers_3d else np.zeros((0, 3))

        pred_dims = np.array(all_pred_dims, dtype=np.float64) if all_pred_dims else np.zeros((0, 3))
        gt_dims = np.array(all_gt_dims, dtype=np.float64) if all_gt_dims else np.zeros((0, 3))

        pred_angles = np.array(all_pred_angles, dtype=np.float64) if all_pred_angles else np.zeros(0)
        gt_angles = np.array(all_gt_angles, dtype=np.float64) if all_gt_angles else np.zeros(0)

        pred_velocities = np.array(all_pred_velocities, dtype=np.float64) if all_pred_velocities else np.zeros((0, 2))
        gt_velocities = np.array(all_gt_velocities, dtype=np.float64) if all_gt_velocities else np.zeros((0, 2))

        pred_attributes = np.array(all_pred_attributes, dtype=np.int32) if all_pred_attributes else np.zeros(0, dtype=np.int32)
        gt_attributes = np.array(all_gt_attributes, dtype=np.int32) if all_gt_attributes else np.zeros(0, dtype=np.int32)

        # AP at each distance threshold
        for dist_thresh in DISTANCE_THRESHOLDS:
            ap = compute_ap_at_threshold(pred_scores, pred_centers, gt_centers, dist_thresh)
            aps_per_threshold.append(ap)

        class_aps[cls_name] = {
            "ap_per_threshold": dict(zip([str(d) for d in DISTANCE_THRESHOLDS], aps_per_threshold)),
            "mean_ap": float(np.mean(aps_per_threshold)),
            "num_predictions": len(pred_scores),
            "num_ground_truth": len(gt_centers),
        }

        # TP metrics (computed at 2.0m matching threshold per nuScenes convention)
        tp_match_dist = 2.0
        mate = compute_translation_error(pred_centers_3d, gt_centers_3d, pred_scores, tp_match_dist)
        mase = compute_scale_error(pred_dims, gt_dims, pred_centers, gt_centers, pred_scores, tp_match_dist)
        maoe = compute_orientation_error(pred_angles, gt_angles, pred_centers, gt_centers, pred_scores, tp_match_dist)
        mave = compute_velocity_error(pred_velocities, gt_velocities, pred_centers, gt_centers, pred_scores, tp_match_dist)
        maae = compute_attribute_error(pred_attributes, gt_attributes, pred_centers, gt_centers, pred_scores, tp_match_dist)

        class_tp_metrics[cls_name] = {
            "mATE": mate,
            "mASE": mase,
            "mAOE": maoe,
            "mAVE": mave,
            "mAAE": maae,
        }

    # Compute overall mAP
    per_class_mean_aps = [class_aps[cls]["mean_ap"] for cls in NUSCENES_CLASSES]
    overall_mAP = float(np.mean(per_class_mean_aps))

    # Compute overall TP metrics (mean across classes)
    overall_tp = {
        "mATE": float(np.mean([class_tp_metrics[cls]["mATE"] for cls in NUSCENES_CLASSES])),
        "mASE": float(np.mean([class_tp_metrics[cls]["mASE"] for cls in NUSCENES_CLASSES])),
        "mAOE": float(np.mean([class_tp_metrics[cls]["mAOE"] for cls in NUSCENES_CLASSES])),
        "mAVE": float(np.mean([class_tp_metrics[cls]["mAVE"] for cls in NUSCENES_CLASSES])),
        "mAAE": float(np.mean([class_tp_metrics[cls]["mAAE"] for cls in NUSCENES_CLASSES])),
    }

    # Compute NDS
    nds = compute_nds(overall_mAP, overall_tp)

    results = {
        "mAP": overall_mAP,
        "NDS": nds,
        "mTP_metrics": overall_tp,
        "per_class_ap": class_aps,
        "per_class_tp_metrics": class_tp_metrics,
        "config": {
            "voxel_size": VOXEL_SIZE,
            "point_cloud_range": POINT_CLOUD_RANGE,
            "bev_resolution": BEV_RESOLUTION,
            "distance_thresholds": DISTANCE_THRESHOLDS,
            "num_classes": len(NUSCENES_CLASSES),
            "classes": NUSCENES_CLASSES,
            "class_groups": NUSCENES_CLASS_GROUPS,
        },
    }

    # Print results table
    print_results_table(results)

    return results


def print_results_table(results: Dict[str, Any]) -> None:
    """Print formatted evaluation results table."""
    print("\n" + "=" * 90)
    print("EVALUATION RESULTS")
    print("=" * 90)

    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 30)
    print(f"{'mAP':<20} {results['mAP']:>10.4f}")
    print(f"{'NDS':<20} {results['NDS']:>10.4f}")
    print(f"{'mATE':<20} {results['mTP_metrics']['mATE']:>10.4f}")
    print(f"{'mASE':<20} {results['mTP_metrics']['mASE']:>10.4f}")
    print(f"{'mAOE':<20} {results['mTP_metrics']['mAOE']:>10.4f}")
    print(f"{'mAVE':<20} {results['mTP_metrics']['mAVE']:>10.4f}")
    print(f"{'mAAE':<20} {results['mTP_metrics']['mAAE']:>10.4f}")

    print(f"\n\n{'Per-Class Results':^90}")
    print("-" * 90)
    header = f"{'Class':<25} {'AP':>8} {'ATE':>8} {'ASE':>8} {'AOE':>8} {'AVE':>8} {'AAE':>8} {'#Pred':>7} {'#GT':>7}"
    print(header)
    print("-" * 90)

    for cls_name in NUSCENES_CLASSES:
        ap_info = results["per_class_ap"][cls_name]
        tp_info = results["per_class_tp_metrics"][cls_name]
        print(
            f"{cls_name:<25} "
            f"{ap_info['mean_ap']:>8.4f} "
            f"{tp_info['mATE']:>8.4f} "
            f"{tp_info['mASE']:>8.4f} "
            f"{tp_info['mAOE']:>8.4f} "
            f"{tp_info['mAVE']:>8.4f} "
            f"{tp_info['mAAE']:>8.4f} "
            f"{ap_info['num_predictions']:>7d} "
            f"{ap_info['num_ground_truth']:>7d}"
        )

    print("-" * 90)
    print(
        f"{'MEAN':<25} "
        f"{results['mAP']:>8.4f} "
        f"{results['mTP_metrics']['mATE']:>8.4f} "
        f"{results['mTP_metrics']['mASE']:>8.4f} "
        f"{results['mTP_metrics']['mAOE']:>8.4f} "
        f"{results['mTP_metrics']['mAVE']:>8.4f} "
        f"{results['mTP_metrics']['mAAE']:>8.4f}"
    )
    print("=" * 90)

    # Per-threshold AP breakdown
    print(f"\n\n{'AP at Each Distance Threshold':^90}")
    print("-" * 90)
    thresh_header = f"{'Class':<25}"
    for t in DISTANCE_THRESHOLDS:
        thresh_header += f" {'AP@' + str(t) + 'm':>10}"
    thresh_header += f" {'Mean AP':>10}"
    print(thresh_header)
    print("-" * 90)

    for cls_name in NUSCENES_CLASSES:
        row = f"{cls_name:<25}"
        for t in DISTANCE_THRESHOLDS:
            ap_val = results["per_class_ap"][cls_name]["ap_per_threshold"][str(t)]
            row += f" {ap_val:>10.4f}"
        row += f" {results['per_class_ap'][cls_name]['mean_ap']:>10.4f}"
        print(row)

    print("=" * 90)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[INFO] Results saved to: {output_path}")


# ==============================================================================
# Command-Line Interface
# ==============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CenterPoint TF2 Evaluation on nuScenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (SavedModel dir, .keras, .h5, or checkpoint prefix)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to validation data (directory with point clouds and info file)",
    )

    # Optional: separate GT path
    parser.add_argument(
        "--gt-path",
        type=str,
        default=None,
        help="Path to ground truth annotations (JSON file or directory). "
        "Defaults to data-path if not specified.",
    )

    # Inference settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (samples processed per iteration)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="Minimum confidence score for detections",
    )
    parser.add_argument(
        "--nms-radius",
        type=float,
        default=None,
        help="Circle NMS radius override (meters). If not set, uses per-class defaults.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=500,
        help="Maximum number of detections per sample",
    )

    # Voxelization settings
    parser.add_argument(
        "--max-points-per-voxel",
        type=int,
        default=10,
        help="Maximum number of points per voxel",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=60000,
        help="Maximum number of non-empty voxels",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output JSON file path for results",
    )

    # Device settings
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index (-1 for CPU)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision (float16) for faster inference",
    )

    return parser.parse_args()


def configure_gpu(gpu_id: int, mixed_precision: bool) -> None:
    """Configure GPU settings."""
    if gpu_id < 0:
        tf.config.set_visible_devices([], "GPU")
        print("[INFO] Running on CPU")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            if gpu_id < len(gpus):
                tf.config.set_visible_devices(gpus[gpu_id], "GPU")
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                print(f"[INFO] Using GPU {gpu_id}: {gpus[gpu_id].name}")
            else:
                print(f"[WARN] GPU {gpu_id} not found. Using GPU 0.")
                tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(f"[WARN] GPU configuration error: {e}")
    else:
        print("[INFO] No GPUs detected. Running on CPU.")

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision (float16) enabled")


# ==============================================================================
# Entry Point
# ==============================================================================


def main() -> None:
    """Main entry point for evaluation."""
    args = parse_args()

    # Configure GPU
    configure_gpu(args.gpu, args.mixed_precision)

    # Run evaluation
    start_time = time.time()
    results = evaluate(args)
    elapsed = time.time() - start_time

    # Add timing info
    results["timing"] = {
        "total_evaluation_time_seconds": elapsed,
        "total_evaluation_time_formatted": f"{elapsed / 60:.1f} minutes",
    }

    # Save results
    save_results(results, args.output)

    print(f"\n[DONE] Evaluation completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Final mAP: {results['mAP']:.4f}")
    print(f"  Final NDS: {results['NDS']:.4f}")


if __name__ == "__main__":
    main()
