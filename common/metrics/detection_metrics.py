"""
3D Object Detection Metrics for Autonomous Driving (nuScenes-style).

This module implements the nuScenes detection evaluation metrics including:
- mAP (mean Average Precision) using center-distance matching
- NDS (nuScenes Detection Score)
- ATE (Average Translation Error)
- ASE (Average Scale Error)
- AOE (Average Orientation Error)
- AVE (Average Velocity Error)
- AAE (Average Attribute Error)

Reference:
    Caesar et al., "nuScenes: A multimodal dataset for autonomous driving", CVPR 2020.

The nuScenes evaluation uses center-distance based matching (BEV 2D Euclidean distance)
rather than IoU-based matching. Predictions are matched greedily to ground truths
sorted by descending confidence score.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


# Default nuScenes distance thresholds per class (in meters)
NUSCENES_DIST_THRESHOLDS: Dict[str, List[float]] = {
    "car": [0.5, 1.0, 2.0, 4.0],
    "truck": [0.5, 1.0, 2.0, 4.0],
    "bus": [0.5, 1.0, 2.0, 4.0],
    "trailer": [0.5, 1.0, 2.0, 4.0],
    "construction_vehicle": [0.5, 1.0, 2.0, 4.0],
    "pedestrian": [0.5, 1.0, 2.0, 4.0],
    "motorcycle": [0.5, 1.0, 2.0, 4.0],
    "bicycle": [0.5, 1.0, 2.0, 4.0],
    "traffic_cone": [0.5, 1.0, 2.0, 4.0],
    "barrier": [0.5, 1.0, 2.0, 4.0],
}

# NDS weights for TP error metrics
NDS_TP_WEIGHTS: Dict[str, float] = {
    "ATE": 1.0,
    "ASE": 1.0,
    "AOE": 1.0,
    "AVE": 1.0,
    "AAE": 1.0,
}

# Maximum values for TP error normalization in NDS computation
TP_ERROR_MAX: Dict[str, float] = {
    "ATE": 1.0,    # 1 meter
    "ASE": 1.0,    # dimensionless [0, 1]
    "AOE": 1.0,    # normalized, max is 1.0 (from pi / pi)
    "AVE": 1.0,    # 1 m/s
    "AAE": 1.0,    # dimensionless [0, 1]
}


def _normalize_angle(angle: np.ndarray) -> np.ndarray:
    """Normalize angles to [-pi, pi].

    Args:
        angle: Array of angles in radians.

    Returns:
        Normalized angles in [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _bev_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute BEV (Bird's Eye View) 2D Euclidean distance between box centers.

    Args:
        boxes_a: (N, 7) array [x, y, z, w, l, h, yaw].
        boxes_b: (M, 7) array [x, y, z, w, l, h, yaw].

    Returns:
        (N, M) distance matrix.
    """
    centers_a = boxes_a[:, :2]  # (N, 2) - x, y only
    centers_b = boxes_b[:, :2]  # (M, 2)
    # Compute pairwise distances
    diff = centers_a[:, np.newaxis, :] - centers_b[np.newaxis, :, :]  # (N, M, 2)
    return np.linalg.norm(diff, axis=-1)  # (N, M)


def _compute_3d_iou_aligned(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute 3D IoU between two boxes after aligning centers and orientations.

    For ASE computation: boxes are axis-aligned (centered at origin, zero yaw)
    and we compute the volumetric IoU.

    Args:
        box_a: (7,) array [x, y, z, w, l, h, yaw] - prediction box.
        box_b: (7,) array [x, y, z, w, l, h, yaw] - ground truth box.

    Returns:
        3D IoU value in [0, 1].
    """
    # After alignment, both boxes are centered at origin with zero yaw.
    # Just compute axis-aligned 3D IoU using the dimensions.
    w_a, l_a, h_a = box_a[3], box_a[4], box_a[5]
    w_b, l_b, h_b = box_b[3], box_b[4], box_b[5]

    # Axis-aligned boxes centered at origin:
    # box_a spans [-w_a/2, w_a/2] x [-l_a/2, l_a/2] x [-h_a/2, h_a/2]
    # box_b spans [-w_b/2, w_b/2] x [-l_b/2, l_b/2] x [-h_b/2, h_b/2]
    overlap_w = max(0.0, min(w_a / 2, w_b / 2) - max(-w_a / 2, -w_b / 2))
    overlap_l = max(0.0, min(l_a / 2, l_b / 2) - max(-l_a / 2, -l_b / 2))
    overlap_h = max(0.0, min(h_a / 2, h_b / 2) - max(-h_a / 2, -h_b / 2))

    # Simplification: overlap in each axis is just min of the two half-extents * 2
    overlap_w = min(w_a, w_b)
    overlap_l = min(l_a, l_b)
    overlap_h = min(h_a, h_b)

    intersection = overlap_w * overlap_l * overlap_h
    vol_a = w_a * l_a * h_a
    vol_b = w_b * l_b * h_b
    union = vol_a + vol_b - intersection

    if union <= 0:
        return 0.0
    return float(intersection / union)


def _greedy_match(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    pred_scores: np.ndarray,
    dist_threshold: float,
) -> List[Tuple[int, int]]:
    """Greedy matching of predictions to ground truths by center distance.

    Predictions are processed in order of decreasing confidence score.
    Each prediction is matched to the closest unmatched ground truth
    within the distance threshold.

    Args:
        pred_boxes: (N, 7) predicted boxes.
        gt_boxes: (M, 7) ground truth boxes.
        pred_scores: (N,) confidence scores for predictions.
        dist_threshold: Maximum BEV distance for a valid match (meters).

    Returns:
        List of (pred_idx, gt_idx) matched pairs.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    # Compute distance matrix
    dist_matrix = _bev_distance(pred_boxes, gt_boxes)  # (N, M)

    # Sort predictions by confidence (descending)
    sorted_pred_indices = np.argsort(-pred_scores)

    matched_gt = set()
    matches = []

    for pred_idx in sorted_pred_indices:
        # Find closest unmatched GT within threshold
        distances = dist_matrix[pred_idx]
        # Mask already matched GTs
        valid_mask = np.ones(len(gt_boxes), dtype=bool)
        for gt_idx in matched_gt:
            valid_mask[gt_idx] = False

        masked_distances = np.where(valid_mask, distances, np.inf)
        min_dist_idx = np.argmin(masked_distances)
        min_dist = masked_distances[min_dist_idx]

        if min_dist <= dist_threshold:
            matches.append((int(pred_idx), int(min_dist_idx)))
            matched_gt.add(int(min_dist_idx))

    return matches


def _hungarian_match(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    dist_threshold: float,
) -> List[Tuple[int, int]]:
    """Hungarian (optimal) matching of predictions to ground truths.

    Uses scipy's linear_sum_assignment for optimal bipartite matching
    based on BEV center distances.

    Args:
        pred_boxes: (N, 7) predicted boxes.
        gt_boxes: (M, 7) ground truth boxes.
        dist_threshold: Maximum BEV distance for a valid match (meters).

    Returns:
        List of (pred_idx, gt_idx) matched pairs within the distance threshold.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    dist_matrix = _bev_distance(pred_boxes, gt_boxes)  # (N, M)

    # Replace distances above threshold with a large value
    large_val = 1e6
    cost_matrix = np.where(dist_matrix <= dist_threshold, dist_matrix, large_val)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] < large_val:
            matches.append((int(r), int(c)))

    return matches


def compute_translation_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute Average Translation Error (ATE).

    ATE is the Euclidean distance between the 3D centers of prediction and
    ground truth boxes.

    Args:
        pred_box: (7,) array [x, y, z, w, l, h, yaw].
        gt_box: (7,) array [x, y, z, w, l, h, yaw].

    Returns:
        Translation error in meters (Euclidean distance of 3D centers).
    """
    return float(np.linalg.norm(pred_box[:3] - gt_box[:3]))


def compute_scale_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute Average Scale Error (ASE).

    ASE = 1 - IoU(pred, gt) after aligning centers and orientations.
    Both boxes are placed at the origin with zero yaw, then 3D IoU is computed.

    Args:
        pred_box: (7,) array [x, y, z, w, l, h, yaw].
        gt_box: (7,) array [x, y, z, w, l, h, yaw].

    Returns:
        Scale error in [0, 1]. Lower is better.
    """
    iou = _compute_3d_iou_aligned(pred_box, gt_box)
    return 1.0 - iou


def compute_orientation_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute Average Orientation Error (AOE).

    AOE is the smallest yaw angle difference between prediction and ground truth.
    The difference is computed modulo pi to handle directional ambiguity for
    symmetric objects (e.g., cars look the same from front/back in some cases).
    However, for nuScenes the full yaw difference is used.

    Args:
        pred_box: (7,) array [x, y, z, w, l, h, yaw].
        gt_box: (7,) array [x, y, z, w, l, h, yaw].

    Returns:
        Orientation error in radians [0, pi].
    """
    yaw_pred = pred_box[6]
    yaw_gt = gt_box[6]
    diff = _normalize_angle(yaw_pred - yaw_gt)
    return float(np.abs(diff))


def compute_velocity_error(
    pred_velocity: np.ndarray, gt_velocity: np.ndarray
) -> float:
    """Compute Average Velocity Error (AVE).

    AVE is the L2 norm of the velocity difference in the BEV plane.

    Args:
        pred_velocity: (2,) predicted velocity [vx, vy] in m/s.
        gt_velocity: (2,) ground truth velocity [vx, vy] in m/s.

    Returns:
        Velocity error in m/s.
    """
    return float(np.linalg.norm(pred_velocity - gt_velocity))


def compute_attribute_error(pred_attr: int, gt_attr: int) -> float:
    """Compute Average Attribute Error (AAE).

    AAE = 1 - (attribute classification accuracy).
    For a single pair, it is 0 if attributes match, 1 otherwise.

    Args:
        pred_attr: Predicted attribute class index.
        gt_attr: Ground truth attribute class index.

    Returns:
        Attribute error: 0.0 if match, 1.0 if mismatch.
    """
    return 0.0 if pred_attr == gt_attr else 1.0


def _compute_ap(
    recalls: np.ndarray, precisions: np.ndarray, min_recall: float = 0.1, min_precision: float = 0.1
) -> float:
    """Compute Average Precision from precision-recall curve (nuScenes style).

    In nuScenes, AP is computed as the normalized area under the PR curve
    above min_recall and min_precision thresholds. The PR curve is interpolated
    using the maximum precision at each recall level (right-to-left envelope).

    Args:
        recalls: (K,) sorted recall values (ascending).
        precisions: (K,) precision values corresponding to each recall level.
        min_recall: Minimum recall threshold. Points below this are ignored.
        min_precision: Minimum precision threshold. Precision below this counts as 0.

    Returns:
        Average Precision value in [0, 1].
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    # Filter to points above min_recall
    valid = recalls >= min_recall
    if not np.any(valid):
        return 0.0

    recalls_valid = recalls[valid]
    precisions_valid = precisions[valid]

    # Apply min_precision threshold
    precisions_valid = np.where(precisions_valid >= min_precision, precisions_valid, 0.0)

    # nuScenes-style: interpolate precision at 101 recall levels
    recall_interp = np.linspace(min_recall, 1.0, 101)
    precision_interp = np.zeros_like(recall_interp)

    for i, r in enumerate(recall_interp):
        # Maximum precision at recalls >= r
        mask = recalls_valid >= r
        if np.any(mask):
            precision_interp[i] = np.max(precisions_valid[mask])
        else:
            precision_interp[i] = 0.0

    # AP is the mean of interpolated precisions normalized by (1 - min_recall)
    ap = float(np.mean(precision_interp))
    return ap


def _accumulate_single_class(
    pred_data: List[Dict[str, Any]],
    gt_data: List[Dict[str, Any]],
    dist_threshold: float,
    matching: str = "greedy",
) -> Tuple[np.ndarray, np.ndarray, int, List[Tuple[int, int, int]]]:
    """Accumulate true positives and false positives for a single class across all frames.

    Args:
        pred_data: List of per-frame prediction dicts, each containing:
            - 'boxes': (N, 7) boxes
            - 'scores': (N,) confidence scores
        gt_data: List of per-frame ground truth dicts, each containing:
            - 'boxes': (M, 7) boxes
        dist_threshold: Distance threshold for matching.
        matching: Matching algorithm ('greedy' or 'hungarian').

    Returns:
        Tuple of:
            - recalls: (K,) recall values at each detection.
            - precisions: (K,) precision values at each detection.
            - total_gt: Total number of ground truth instances.
            - all_matches: List of (frame_idx, pred_idx, gt_idx) for TP matches.
    """
    assert len(pred_data) == len(gt_data), "pred_data and gt_data must have same length"

    # Collect all predictions with frame indices
    all_preds = []  # (score, frame_idx, local_pred_idx)
    total_gt = 0

    for frame_idx in range(len(pred_data)):
        pred = pred_data[frame_idx]
        gt = gt_data[frame_idx]

        n_pred = len(pred["boxes"]) if len(pred["boxes"]) > 0 else 0
        n_gt = len(gt["boxes"]) if len(gt["boxes"]) > 0 else 0
        total_gt += n_gt

        for pred_idx in range(n_pred):
            all_preds.append((pred["scores"][pred_idx], frame_idx, pred_idx))

    if total_gt == 0:
        return np.array([]), np.array([]), 0, []

    # Sort all predictions by score (descending) across all frames
    all_preds.sort(key=lambda x: -x[0])

    # Perform matching per frame
    # Pre-compute matches for each frame and threshold
    frame_matches: Dict[int, List[Tuple[int, int]]] = {}
    for frame_idx in range(len(pred_data)):
        pred = pred_data[frame_idx]
        gt = gt_data[frame_idx]

        if len(pred["boxes"]) == 0 or len(gt["boxes"]) == 0:
            frame_matches[frame_idx] = []
            continue

        pred_boxes = np.array(pred["boxes"])
        gt_boxes = np.array(gt["boxes"])
        pred_scores = np.array(pred["scores"])

        if matching == "greedy":
            matches = _greedy_match(pred_boxes, gt_boxes, pred_scores, dist_threshold)
        else:
            matches = _hungarian_match(pred_boxes, gt_boxes, dist_threshold)

        frame_matches[frame_idx] = matches

    # Build TP/FP arrays in order of decreasing confidence
    tp = np.zeros(len(all_preds), dtype=np.int32)
    fp = np.zeros(len(all_preds), dtype=np.int32)
    all_tp_matches = []  # (frame_idx, pred_idx, gt_idx)

    # Create lookup for matches per frame: pred_idx -> gt_idx
    frame_match_lookup: Dict[int, Dict[int, int]] = {}
    for frame_idx, matches in frame_matches.items():
        frame_match_lookup[frame_idx] = {p: g for p, g in matches}

    for i, (score, frame_idx, pred_idx) in enumerate(all_preds):
        match_dict = frame_match_lookup.get(frame_idx, {})
        if pred_idx in match_dict:
            tp[i] = 1
            all_tp_matches.append((frame_idx, pred_idx, match_dict[pred_idx]))
        else:
            fp[i] = 1

    # Compute cumulative TP and FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    # Compute precision and recall
    precisions = cum_tp / (cum_tp + cum_fp)
    recalls = cum_tp / total_gt

    return recalls, precisions, total_gt, all_tp_matches


def compute_ap_per_class(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    class_id: int,
    dist_thresholds: List[float],
    matching: str = "greedy",
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> Dict[str, Any]:
    """Compute Average Precision for a single class at multiple distance thresholds.

    Args:
        predictions: List of per-frame prediction dicts with keys:
            - 'boxes': (N, 7) array [x, y, z, w, l, h, yaw]
            - 'scores': (N,) confidence scores
            - 'labels': (N,) class labels (int)
            - 'velocities': (N, 2) optional velocities [vx, vy]
            - 'attributes': (N,) optional attribute indices
        ground_truths: List of per-frame ground truth dicts with same keys
            (scores not required for GT).
        class_id: The class ID to evaluate.
        dist_thresholds: List of distance thresholds for matching.
        matching: 'greedy' (default, nuScenes-style) or 'hungarian'.
        min_recall: Minimum recall for AP computation.
        min_precision: Minimum precision for AP computation.

    Returns:
        Dictionary with:
            - 'ap_per_threshold': Dict[float, float] AP at each threshold.
            - 'mean_ap': float, mean AP across thresholds.
    """
    # Filter predictions and ground truths by class
    pred_data = []
    gt_data = []

    for frame_idx in range(len(predictions)):
        pred = predictions[frame_idx]
        gt = ground_truths[frame_idx]

        # Filter predictions for this class
        pred_labels = np.array(pred.get("labels", []))
        pred_boxes = np.array(pred.get("boxes", [])).reshape(-1, 7)
        pred_scores = np.array(pred.get("scores", []))

        class_mask_pred = pred_labels == class_id
        if np.any(class_mask_pred):
            pred_data.append({
                "boxes": pred_boxes[class_mask_pred],
                "scores": pred_scores[class_mask_pred],
            })
        else:
            pred_data.append({"boxes": np.empty((0, 7)), "scores": np.array([])})

        # Filter GT for this class
        gt_labels = np.array(gt.get("labels", []))
        gt_boxes = np.array(gt.get("boxes", [])).reshape(-1, 7)

        class_mask_gt = gt_labels == class_id
        if np.any(class_mask_gt):
            gt_data.append({"boxes": gt_boxes[class_mask_gt]})
        else:
            gt_data.append({"boxes": np.empty((0, 7))})

    ap_per_threshold = {}
    for thresh in dist_thresholds:
        recalls, precisions, total_gt, _ = _accumulate_single_class(
            pred_data, gt_data, thresh, matching
        )
        if total_gt == 0:
            ap_per_threshold[thresh] = 0.0
        else:
            ap_per_threshold[thresh] = _compute_ap(recalls, precisions, min_recall, min_precision)

    mean_ap = float(np.mean(list(ap_per_threshold.values()))) if ap_per_threshold else 0.0

    return {
        "ap_per_threshold": ap_per_threshold,
        "mean_ap": mean_ap,
    }


def compute_tp_errors(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    class_id: int,
    dist_threshold: float = 2.0,
    matching: str = "greedy",
) -> Dict[str, float]:
    """Compute True Positive error metrics for a single class.

    For each matched (TP) pair at the given distance threshold, compute:
    - ATE: Average Translation Error
    - ASE: Average Scale Error
    - AOE: Average Orientation Error
    - AVE: Average Velocity Error
    - AAE: Average Attribute Error

    Args:
        predictions: List of per-frame prediction dicts.
        ground_truths: List of per-frame ground truth dicts.
        class_id: The class ID to evaluate.
        dist_threshold: Distance threshold for matching (default 2.0m).
        matching: 'greedy' or 'hungarian'.

    Returns:
        Dictionary with keys 'ATE', 'ASE', 'AOE', 'AVE', 'AAE' -> float values.
        Returns NaN for metrics that cannot be computed (no matches or missing data).
    """
    ate_errors = []
    ase_errors = []
    aoe_errors = []
    ave_errors = []
    aae_errors = []

    for frame_idx in range(len(predictions)):
        pred = predictions[frame_idx]
        gt = ground_truths[frame_idx]

        # Filter by class
        pred_labels = np.array(pred.get("labels", []))
        pred_boxes = np.array(pred.get("boxes", [])).reshape(-1, 7)
        pred_scores = np.array(pred.get("scores", []))
        pred_velocities = np.array(pred.get("velocities", [])).reshape(-1, 2) if "velocities" in pred and len(pred["velocities"]) > 0 else None
        pred_attributes = np.array(pred.get("attributes", [])) if "attributes" in pred and len(pred["attributes"]) > 0 else None

        gt_labels = np.array(gt.get("labels", []))
        gt_boxes = np.array(gt.get("boxes", [])).reshape(-1, 7)
        gt_velocities = np.array(gt.get("velocities", [])).reshape(-1, 2) if "velocities" in gt and len(gt["velocities"]) > 0 else None
        gt_attributes = np.array(gt.get("attributes", [])) if "attributes" in gt and len(gt["attributes"]) > 0 else None

        # Class filtering
        pred_mask = pred_labels == class_id
        gt_mask = gt_labels == class_id

        if not np.any(pred_mask) or not np.any(gt_mask):
            continue

        cls_pred_boxes = pred_boxes[pred_mask]
        cls_pred_scores = pred_scores[pred_mask]
        cls_gt_boxes = gt_boxes[gt_mask]

        # Get corresponding velocities and attributes
        cls_pred_vel = pred_velocities[pred_mask] if pred_velocities is not None and len(pred_velocities) == len(pred_labels) else None
        cls_gt_vel = gt_velocities[gt_mask] if gt_velocities is not None and len(gt_velocities) == len(gt_labels) else None
        cls_pred_attr = pred_attributes[pred_mask] if pred_attributes is not None and len(pred_attributes) == len(pred_labels) else None
        cls_gt_attr = gt_attributes[gt_mask] if gt_attributes is not None and len(gt_attributes) == len(gt_labels) else None

        # Match predictions to ground truths
        if matching == "greedy":
            matches = _greedy_match(cls_pred_boxes, cls_gt_boxes, cls_pred_scores, dist_threshold)
        else:
            matches = _hungarian_match(cls_pred_boxes, cls_gt_boxes, dist_threshold)

        for pred_idx, gt_idx in matches:
            p_box = cls_pred_boxes[pred_idx]
            g_box = cls_gt_boxes[gt_idx]

            ate_errors.append(compute_translation_error(p_box, g_box))
            ase_errors.append(compute_scale_error(p_box, g_box))
            aoe_errors.append(compute_orientation_error(p_box, g_box))

            if cls_pred_vel is not None and cls_gt_vel is not None:
                ave_errors.append(compute_velocity_error(cls_pred_vel[pred_idx], cls_gt_vel[gt_idx]))

            if cls_pred_attr is not None and cls_gt_attr is not None:
                aae_errors.append(compute_attribute_error(int(cls_pred_attr[pred_idx]), int(cls_gt_attr[gt_idx])))

    results = {
        "ATE": float(np.mean(ate_errors)) if ate_errors else float("nan"),
        "ASE": float(np.mean(ase_errors)) if ase_errors else float("nan"),
        "AOE": float(np.mean(aoe_errors)) if aoe_errors else float("nan"),
        "AVE": float(np.mean(ave_errors)) if ave_errors else float("nan"),
        "AAE": float(np.mean(aae_errors)) if aae_errors else float("nan"),
    }

    return results


def compute_map(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    class_ids: Optional[List[int]] = None,
    class_names: Optional[Dict[int, str]] = None,
    dist_thresholds: Optional[Dict[str, List[float]]] = None,
    matching: str = "greedy",
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> Dict[str, Any]:
    """Compute mean Average Precision (mAP) across all classes and distance thresholds.

    Args:
        predictions: List of per-frame prediction dicts with keys:
            - 'boxes': (N, 7) array [x, y, z, w, l, h, yaw]
            - 'scores': (N,) confidence scores
            - 'labels': (N,) integer class labels
            - 'velocities': (N, 2) optional [vx, vy]
            - 'attributes': (N,) optional attribute indices
        ground_truths: List of per-frame ground truth dicts with same keys.
        class_ids: List of class IDs to evaluate. If None, inferred from GT.
        class_names: Optional mapping from class_id -> class name string.
        dist_thresholds: Dict of class_name -> list of distance thresholds.
            If None, uses NUSCENES_DIST_THRESHOLDS. If class_names is not provided,
            all classes use [0.5, 1.0, 2.0, 4.0].
        matching: 'greedy' (default) or 'hungarian'.
        min_recall: Minimum recall threshold for AP computation.
        min_precision: Minimum precision threshold for AP computation.

    Returns:
        Dictionary with:
            - 'mAP': float, overall mean AP.
            - 'per_class': Dict[int, Dict] with per-class results including
              'ap_per_threshold' and 'mean_ap'.
    """
    # Infer class IDs from ground truths if not provided
    if class_ids is None:
        all_labels = set()
        for gt in ground_truths:
            labels = np.array(gt.get("labels", []))
            if len(labels) > 0:
                all_labels.update(labels.tolist())
        class_ids = sorted(all_labels)

    if not class_ids:
        return {"mAP": 0.0, "per_class": {}}

    # Determine distance thresholds per class
    default_thresholds = [0.5, 1.0, 2.0, 4.0]

    per_class_results = {}
    class_aps = []

    for cls_id in class_ids:
        # Get thresholds for this class
        if dist_thresholds is not None and class_names is not None and cls_id in class_names:
            cls_name = class_names[cls_id]
            thresholds = dist_thresholds.get(cls_name, default_thresholds)
        elif dist_thresholds is not None and class_names is None:
            # Use default thresholds for all classes
            thresholds = default_thresholds
        else:
            thresholds = default_thresholds

        result = compute_ap_per_class(
            predictions, ground_truths, cls_id, thresholds, matching, min_recall, min_precision
        )
        per_class_results[cls_id] = result
        class_aps.append(result["mean_ap"])

    mAP = float(np.mean(class_aps)) if class_aps else 0.0

    return {
        "mAP": mAP,
        "per_class": per_class_results,
    }


def compute_nds(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    class_ids: Optional[List[int]] = None,
    class_names: Optional[Dict[int, str]] = None,
    dist_thresholds: Optional[Dict[str, List[float]]] = None,
    tp_dist_threshold: float = 2.0,
    matching: str = "greedy",
    tp_weights: Optional[Dict[str, float]] = None,
    tp_error_max: Optional[Dict[str, float]] = None,
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> Dict[str, Any]:
    """Compute the nuScenes Detection Score (NDS).

    NDS = (1/10) * [5 * mAP + sum(max(1 - TP_err/TP_max, 0) for each TP metric)]

    This is a weighted combination of mAP and the five TP error metrics
    (ATE, ASE, AOE, AVE, AAE), where each TP error is normalized and
    converted to a score (higher is better).

    Args:
        predictions: List of per-frame prediction dicts.
        ground_truths: List of per-frame ground truth dicts.
        class_ids: List of class IDs to evaluate.
        class_names: Optional mapping from class_id -> class name.
        dist_thresholds: Distance thresholds per class for mAP.
        tp_dist_threshold: Distance threshold for TP error computation.
        matching: 'greedy' or 'hungarian'.
        tp_weights: Weights for each TP error metric. Default: all 1.0.
        tp_error_max: Maximum error values for normalization.
        min_recall: Minimum recall for AP computation.
        min_precision: Minimum precision for AP computation.

    Returns:
        Dictionary with:
            - 'NDS': float, the detection score [0, 1].
            - 'mAP': float, mean AP.
            - 'tp_errors': Dict[str, float], per-metric mean TP errors.
            - 'tp_scores': Dict[str, float], per-metric normalized scores.
            - 'per_class_tp_errors': Dict[int, Dict[str, float]], per-class TP errors.
            - 'per_class_ap': Dict[int, Dict], per-class AP results.
    """
    if tp_weights is None:
        tp_weights = NDS_TP_WEIGHTS.copy()
    if tp_error_max is None:
        tp_error_max = TP_ERROR_MAX.copy()

    # Compute mAP
    map_result = compute_map(
        predictions, ground_truths, class_ids, class_names,
        dist_thresholds, matching, min_recall, min_precision
    )
    mAP = map_result["mAP"]
    used_class_ids = list(map_result["per_class"].keys())

    # Compute TP errors per class
    per_class_tp_errors: Dict[int, Dict[str, float]] = {}
    for cls_id in used_class_ids:
        tp_errors = compute_tp_errors(
            predictions, ground_truths, cls_id, tp_dist_threshold, matching
        )
        per_class_tp_errors[cls_id] = tp_errors

    # Average TP errors across classes (ignoring NaN)
    mean_tp_errors: Dict[str, float] = {}
    for metric in ["ATE", "ASE", "AOE", "AVE", "AAE"]:
        values = [
            per_class_tp_errors[cls_id][metric]
            for cls_id in used_class_ids
            if not np.isnan(per_class_tp_errors[cls_id][metric])
        ]
        mean_tp_errors[metric] = float(np.mean(values)) if values else float("nan")

    # Compute TP scores (1 - error/max, clipped to [0, 1])
    tp_scores: Dict[str, float] = {}
    for metric in ["ATE", "ASE", "AOE", "AVE", "AAE"]:
        if np.isnan(mean_tp_errors[metric]):
            tp_scores[metric] = 0.0
        else:
            score = max(0.0, 1.0 - mean_tp_errors[metric] / tp_error_max[metric])
            tp_scores[metric] = score

    # Compute NDS
    # NDS = (1/10) * [5 * mAP + sum(tp_scores)]
    # With 5 TP metrics each weighted 1.0, total weight = 5 (mAP) + 5 (TP) = 10
    total_tp_weight = sum(tp_weights.values())
    map_weight = 5.0  # nuScenes uses weight 5 for mAP

    weighted_tp_sum = sum(
        tp_weights[metric] * tp_scores[metric] for metric in tp_scores
    )

    nds = (map_weight * mAP + weighted_tp_sum) / (map_weight + total_tp_weight)

    return {
        "NDS": float(nds),
        "mAP": mAP,
        "tp_errors": mean_tp_errors,
        "tp_scores": tp_scores,
        "per_class_tp_errors": per_class_tp_errors,
        "per_class_ap": map_result["per_class"],
    }


def evaluate_detection(
    predictions: Union[List[Dict[str, Any]], Dict[str, Any]],
    ground_truths: Union[List[Dict[str, Any]], Dict[str, Any]],
    class_ids: Optional[List[int]] = None,
    class_names: Optional[Dict[int, str]] = None,
    dist_thresholds: Optional[Dict[str, List[float]]] = None,
    tp_dist_threshold: float = 2.0,
    matching: str = "greedy",
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> Dict[str, Any]:
    """Full nuScenes-style 3D detection evaluation.

    This is the main entry point for computing all detection metrics.
    Supports both batched (list of frames) and single-sample inputs.

    Args:
        predictions: Single prediction dict or list of per-frame prediction dicts.
            Each dict should have:
            - 'boxes': (N, 7) np.ndarray [x, y, z, w, l, h, yaw]
            - 'scores': (N,) np.ndarray, confidence scores in [0, 1]
            - 'labels': (N,) np.ndarray of int, class labels
            - 'velocities': (N, 2) np.ndarray [vx, vy] (optional)
            - 'attributes': (N,) np.ndarray of int (optional)
        ground_truths: Single GT dict or list of per-frame GT dicts.
            Each dict should have:
            - 'boxes': (N, 7) np.ndarray [x, y, z, w, l, h, yaw]
            - 'labels': (N,) np.ndarray of int, class labels
            - 'velocities': (N, 2) np.ndarray [vx, vy] (optional)
            - 'attributes': (N,) np.ndarray of int (optional)
        class_ids: List of class IDs to evaluate. If None, auto-detected from GT.
        class_names: Optional dict mapping class_id -> string name.
        dist_thresholds: Dict of class_name -> list of BEV distance thresholds
            for matching (meters). If None, all classes use [0.5, 1.0, 2.0, 4.0].
        tp_dist_threshold: Distance threshold for TP error computation (default 2.0m).
        matching: Matching strategy: 'greedy' (nuScenes default) or 'hungarian'.
        min_recall: Minimum recall threshold for AP integration.
        min_precision: Minimum precision threshold for AP integration.

    Returns:
        Dictionary containing all metrics:
            - 'NDS': nuScenes Detection Score [0, 1].
            - 'mAP': mean Average Precision [0, 1].
            - 'tp_errors': Dict with mean ATE, ASE, AOE, AVE, AAE.
            - 'tp_scores': Dict with normalized TP scores.
            - 'per_class_tp_errors': Dict[class_id, Dict[metric, value]].
            - 'per_class_ap': Dict[class_id, Dict] with AP per threshold.

    Example:
        >>> predictions = [{
        ...     'boxes': np.array([[10.0, 20.0, 1.0, 2.0, 4.5, 1.5, 0.1]]),
        ...     'scores': np.array([0.9]),
        ...     'labels': np.array([0]),
        ...     'velocities': np.array([[5.0, 0.0]]),
        ...     'attributes': np.array([1]),
        ... }]
        >>> ground_truths = [{
        ...     'boxes': np.array([[10.2, 20.1, 1.0, 2.0, 4.5, 1.5, 0.1]]),
        ...     'labels': np.array([0]),
        ...     'velocities': np.array([[5.1, 0.0]]),
        ...     'attributes': np.array([1]),
        ... }]
        >>> results = evaluate_detection(predictions, ground_truths)
        >>> print(f"NDS: {results['NDS']:.4f}, mAP: {results['mAP']:.4f}")
    """
    # Handle single-sample input
    if isinstance(predictions, dict):
        predictions = [predictions]
    if isinstance(ground_truths, dict):
        ground_truths = [ground_truths]

    assert len(predictions) == len(ground_truths), (
        f"Number of prediction frames ({len(predictions)}) must match "
        f"ground truth frames ({len(ground_truths)})"
    )

    # Validate and normalize inputs
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Ensure numpy arrays
        if "boxes" not in pred or len(pred["boxes"]) == 0:
            pred["boxes"] = np.empty((0, 7))
            pred["scores"] = np.array([])
            pred["labels"] = np.array([], dtype=np.int64)
        else:
            pred["boxes"] = np.asarray(pred["boxes"], dtype=np.float64)
            pred["scores"] = np.asarray(pred["scores"], dtype=np.float64)
            pred["labels"] = np.asarray(pred["labels"], dtype=np.int64)

        if "boxes" not in gt or len(gt["boxes"]) == 0:
            gt["boxes"] = np.empty((0, 7))
            gt["labels"] = np.array([], dtype=np.int64)
        else:
            gt["boxes"] = np.asarray(gt["boxes"], dtype=np.float64)
            gt["labels"] = np.asarray(gt["labels"], dtype=np.int64)

        # Optional fields
        if "velocities" in pred and pred["velocities"] is not None and len(pred["velocities"]) > 0:
            pred["velocities"] = np.asarray(pred["velocities"], dtype=np.float64)
        if "velocities" in gt and gt["velocities"] is not None and len(gt["velocities"]) > 0:
            gt["velocities"] = np.asarray(gt["velocities"], dtype=np.float64)
        if "attributes" in pred and pred["attributes"] is not None and len(pred["attributes"]) > 0:
            pred["attributes"] = np.asarray(pred["attributes"], dtype=np.int64)
        if "attributes" in gt and gt["attributes"] is not None and len(gt["attributes"]) > 0:
            gt["attributes"] = np.asarray(gt["attributes"], dtype=np.int64)

    # Run full evaluation
    results = compute_nds(
        predictions,
        ground_truths,
        class_ids=class_ids,
        class_names=class_names,
        dist_thresholds=dist_thresholds,
        tp_dist_threshold=tp_dist_threshold,
        matching=matching,
        min_recall=min_recall,
        min_precision=min_precision,
    )

    return results


def compute_precision_recall_curve(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    class_id: int,
    dist_threshold: float = 2.0,
    matching: str = "greedy",
) -> Dict[str, np.ndarray]:
    """Compute the full precision-recall curve for a single class.

    Useful for visualization and debugging.

    Args:
        predictions: List of per-frame prediction dicts.
        ground_truths: List of per-frame ground truth dicts.
        class_id: Class ID to compute PR curve for.
        dist_threshold: Distance threshold for matching.
        matching: 'greedy' or 'hungarian'.

    Returns:
        Dictionary with:
            - 'recall': (K,) recall values.
            - 'precision': (K,) precision values.
            - 'scores': (K,) confidence thresholds.
            - 'n_gt': int, total number of GT instances.
            - 'n_pred': int, total number of predictions.
    """
    # Filter by class
    pred_data = []
    gt_data = []
    all_scores = []

    for frame_idx in range(len(predictions)):
        pred = predictions[frame_idx]
        gt = ground_truths[frame_idx]

        pred_labels = np.array(pred.get("labels", []))
        pred_boxes = np.array(pred.get("boxes", [])).reshape(-1, 7)
        pred_scores = np.array(pred.get("scores", []))

        gt_labels = np.array(gt.get("labels", []))
        gt_boxes = np.array(gt.get("boxes", [])).reshape(-1, 7)

        class_mask_pred = pred_labels == class_id
        class_mask_gt = gt_labels == class_id

        if np.any(class_mask_pred):
            filtered_scores = pred_scores[class_mask_pred]
            pred_data.append({
                "boxes": pred_boxes[class_mask_pred],
                "scores": filtered_scores,
            })
            all_scores.extend(filtered_scores.tolist())
        else:
            pred_data.append({"boxes": np.empty((0, 7)), "scores": np.array([])})

        if np.any(class_mask_gt):
            gt_data.append({"boxes": gt_boxes[class_mask_gt]})
        else:
            gt_data.append({"boxes": np.empty((0, 7))})

    recalls, precisions, total_gt, _ = _accumulate_single_class(
        pred_data, gt_data, dist_threshold, matching
    )

    # Sort scores for the curve
    all_scores_sorted = sorted(all_scores, reverse=True)
    score_thresholds = np.array(all_scores_sorted) if all_scores_sorted else np.array([])

    return {
        "recall": recalls,
        "precision": precisions,
        "scores": score_thresholds,
        "n_gt": total_gt,
        "n_pred": len(all_scores),
    }
