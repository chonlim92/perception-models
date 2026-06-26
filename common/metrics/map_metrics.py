"""Vectorized HD Map metrics for autonomous driving perception.

Implements evaluation metrics for vectorized map element predictions in the style
of MapTR / VectorMapNet. Supports Chamfer distance, discrete Frechet distance,
and Average Precision at multiple Chamfer distance thresholds.

Polylines are represented as Nx2 or Nx3 numpy arrays (sequences of 2D/3D points).
Predictions are lists of dicts with keys: 'points', 'score', 'label'.
Ground truths are lists of dicts with keys: 'points', 'label'.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CLASSES: List[str] = ["lane_divider", "road_boundary", "pedestrian_crossing"]
DEFAULT_THRESHOLDS: List[float] = [0.5, 1.0, 1.5]

# Type aliases
Polyline = np.ndarray  # shape (N, 2) or (N, 3)
Prediction = Dict[str, Union[Polyline, float, str]]
GroundTruth = Dict[str, Union[Polyline, str]]


# ---------------------------------------------------------------------------
# Core distance metrics
# ---------------------------------------------------------------------------


def chamfer_distance(pred: Polyline, gt: Polyline) -> float:
    """Compute the symmetric Chamfer distance between two polylines.

    For each point in *pred*, find the nearest point in *gt* (forward direction),
    and vice versa (backward direction). The Chamfer distance is the average of
    the mean forward and mean backward minimum distances.

    Parameters
    ----------
    pred : np.ndarray, shape (M, D)
        Predicted polyline with M points in D dimensions.
    gt : np.ndarray, shape (N, D)
        Ground-truth polyline with N points in D dimensions.

    Returns
    -------
    float
        Symmetric Chamfer distance (in the same units as point coordinates, e.g. meters).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return float("inf")

    # Pairwise Euclidean distance matrix: shape (M, N)
    dist_matrix = cdist(pred, gt, metric="euclidean")

    # Forward: for each predicted point, min distance to any GT point
    forward = dist_matrix.min(axis=1).mean()

    # Backward: for each GT point, min distance to any predicted point
    backward = dist_matrix.min(axis=0).mean()

    return float((forward + backward) / 2.0)


def frechet_distance(curve_a: Polyline, curve_b: Polyline) -> float:
    """Compute the discrete Frechet distance between two curves.

    Uses dynamic programming to compute the exact discrete Frechet distance,
    which measures the similarity between two curves while considering the
    ordering of points along each curve.

    Parameters
    ----------
    curve_a : np.ndarray, shape (M, D)
        First polyline with M points.
    curve_b : np.ndarray, shape (N, D)
        Second polyline with N points.

    Returns
    -------
    float
        Discrete Frechet distance between the two curves.
    """
    curve_a = np.asarray(curve_a, dtype=np.float64)
    curve_b = np.asarray(curve_b, dtype=np.float64)

    m = curve_a.shape[0]
    n = curve_b.shape[0]

    if m == 0 or n == 0:
        return float("inf")

    # Pairwise distance matrix
    dist_matrix = cdist(curve_a, curve_b, metric="euclidean")

    # DP table: dp[i, j] = discrete Frechet distance considering curve_a[:i+1]
    # and curve_b[:j+1]
    dp = np.full((m, n), -1.0, dtype=np.float64)

    dp[0, 0] = dist_matrix[0, 0]

    # Fill first column
    for i in range(1, m):
        dp[i, 0] = max(dp[i - 1, 0], dist_matrix[i, 0])

    # Fill first row
    for j in range(1, n):
        dp[0, j] = max(dp[0, j - 1], dist_matrix[0, j])

    # Fill the rest of the table
    for i in range(1, m):
        for j in range(1, n):
            prev_min = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            dp[i, j] = max(prev_min, dist_matrix[i, j])

    return float(dp[m - 1, n - 1])


# ---------------------------------------------------------------------------
# Matching and AP computation
# ---------------------------------------------------------------------------


def _greedy_match(
    predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    threshold: float,
) -> Tuple[List[bool], int]:
    """Greedily match predictions to ground truths using Chamfer distance.

    Predictions must be pre-sorted by descending confidence score.
    Each ground-truth element can only be matched once.

    Parameters
    ----------
    predictions : list of dict
        Predictions sorted by descending score, each with 'points' key.
    ground_truths : list of dict
        Ground-truth elements, each with 'points' key.
    threshold : float
        Maximum Chamfer distance (meters) for a valid match.

    Returns
    -------
    tp_flags : list of bool
        For each prediction (in order), whether it is a true positive.
    n_gt : int
        Total number of ground-truth elements.
    """
    n_gt = len(ground_truths)
    matched_gt = [False] * n_gt
    tp_flags: List[bool] = []

    for pred in predictions:
        pred_points = pred["points"]
        best_dist = float("inf")
        best_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if matched_gt[gt_idx]:
                continue
            dist = chamfer_distance(pred_points, gt["points"])
            if dist < best_dist:
                best_dist = dist
                best_idx = gt_idx

        if best_dist <= threshold and best_idx >= 0:
            tp_flags.append(True)
            matched_gt[best_idx] = True
        else:
            tp_flags.append(False)

    return tp_flags, n_gt


def _compute_ap(tp_flags: List[bool], n_gt: int) -> float:
    """Compute Average Precision from true-positive flags using all-point interpolation.

    Parameters
    ----------
    tp_flags : list of bool
        Ordered list indicating whether each prediction is a true positive.
    n_gt : int
        Total number of ground-truth elements for recall computation.

    Returns
    -------
    float
        Average Precision value in [0, 1].
    """
    if n_gt == 0:
        return 0.0 if len(tp_flags) > 0 else 1.0

    tp_cumsum = np.cumsum(tp_flags).astype(np.float64)
    fp_cumsum = np.cumsum([not tp for tp in tp_flags]).astype(np.float64)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_gt

    # All-point interpolation: for each recall level, precision is the maximum
    # precision at any recall >= that level.
    # Prepend (recall=0, precision=1) and append (recall=1, precision=0) sentinel values
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    # Make precision monotonically decreasing (from right to left)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Find points where recall changes
    recall_diff = np.diff(recall)
    ap = float(np.sum(recall_diff * precision[1:]))

    return ap


def compute_ap_per_class(
    predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    threshold: float,
    classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute per-class AP at a given Chamfer distance threshold.

    Parameters
    ----------
    predictions : list of dict
        Each dict has keys 'points' (Polyline), 'score' (float), 'label' (str).
    ground_truths : list of dict
        Each dict has keys 'points' (Polyline), 'label' (str).
    threshold : float
        Chamfer distance threshold in meters for matching.
    classes : list of str, optional
        Class names to evaluate. Defaults to DEFAULT_CLASSES.

    Returns
    -------
    dict
        Mapping from class name to AP value.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    results: Dict[str, float] = {}

    for cls in classes:
        # Filter predictions and GTs for this class
        cls_preds = [p for p in predictions if p["label"] == cls]
        cls_gts = [g for g in ground_truths if g["label"] == cls]

        # Sort predictions by score (descending)
        cls_preds = sorted(cls_preds, key=lambda x: x["score"], reverse=True)

        tp_flags, n_gt = _greedy_match(cls_preds, cls_gts, threshold)
        results[cls] = _compute_ap(tp_flags, n_gt)

    return results


# ---------------------------------------------------------------------------
# Single-frame evaluation
# ---------------------------------------------------------------------------


def evaluate_frame(
    predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    thresholds: Optional[List[float]] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate all vectorized map metrics for a single frame.

    Computes per-class AP at each Chamfer distance threshold, the mean AP
    across classes and thresholds, and mean Chamfer and Frechet distances
    for matched pairs.

    Parameters
    ----------
    predictions : list of dict
        Predictions with 'points', 'score', 'label' keys.
    ground_truths : list of dict
        Ground truths with 'points', 'label' keys.
    thresholds : list of float, optional
        Chamfer distance thresholds (meters). Defaults to [0.5, 1.0, 1.5].
    classes : list of str, optional
        Class names to evaluate. Defaults to DEFAULT_CLASSES.

    Returns
    -------
    dict
        Metric name -> value. Includes:
        - 'AP/{class}@{threshold}': per-class AP at each threshold
        - 'mAP@{threshold}': mean AP across classes at each threshold
        - 'mAP': mean AP across all classes and thresholds
        - 'chamfer/mean': mean Chamfer distance over all prediction-GT pairs
        - 'frechet/mean': mean Frechet distance over all prediction-GT pairs
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if classes is None:
        classes = DEFAULT_CLASSES

    metrics: Dict[str, float] = {}
    all_aps: List[float] = []

    # Per-class AP at each threshold
    for thresh in thresholds:
        class_aps = compute_ap_per_class(predictions, ground_truths, thresh, classes)
        threshold_aps: List[float] = []
        for cls, ap in class_aps.items():
            key = f"AP/{cls}@{thresh:.1f}"
            metrics[key] = ap
            threshold_aps.append(ap)
            all_aps.append(ap)

        metrics[f"mAP@{thresh:.1f}"] = float(np.mean(threshold_aps)) if threshold_aps else 0.0

    metrics["mAP"] = float(np.mean(all_aps)) if all_aps else 0.0

    # Compute mean Chamfer and Frechet distances for closest matched pairs
    chamfer_dists: List[float] = []
    frechet_dists: List[float] = []

    for cls in classes:
        cls_preds = sorted(
            [p for p in predictions if p["label"] == cls],
            key=lambda x: x["score"],
            reverse=True,
        )
        cls_gts = [g for g in ground_truths if g["label"] == cls]

        # Match using greedy assignment at the loosest threshold
        matched_gt_flags = [False] * len(cls_gts)
        for pred in cls_preds:
            best_dist = float("inf")
            best_idx = -1
            for gt_idx, gt in enumerate(cls_gts):
                if matched_gt_flags[gt_idx]:
                    continue
                dist = chamfer_distance(pred["points"], gt["points"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = gt_idx

            if best_idx >= 0 and best_dist < float("inf"):
                matched_gt_flags[best_idx] = True
                chamfer_dists.append(best_dist)
                frechet_dists.append(
                    frechet_distance(pred["points"], cls_gts[best_idx]["points"])
                )

    metrics["chamfer/mean"] = float(np.mean(chamfer_dists)) if chamfer_dists else 0.0
    metrics["frechet/mean"] = float(np.mean(frechet_dists)) if frechet_dists else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Batched evaluation (multiple frames)
# ---------------------------------------------------------------------------


def evaluate_batch(
    batch_predictions: List[List[Prediction]],
    batch_ground_truths: List[List[GroundTruth]],
    thresholds: Optional[List[float]] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate vectorized map metrics over a batch of frames.

    Aggregates predictions and ground truths across all frames before computing
    AP (i.e. treats the batch as a single evaluation set), which is the standard
    approach for detection AP computation.

    Parameters
    ----------
    batch_predictions : list of list of dict
        Outer list = frames, inner list = predictions per frame.
    batch_ground_truths : list of list of dict
        Outer list = frames, inner list = ground truths per frame.
    thresholds : list of float, optional
        Chamfer distance thresholds (meters). Defaults to [0.5, 1.0, 1.5].
    classes : list of str, optional
        Class names. Defaults to DEFAULT_CLASSES.

    Returns
    -------
    dict
        Aggregated metrics (same keys as evaluate_frame).
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if classes is None:
        classes = DEFAULT_CLASSES

    if len(batch_predictions) != len(batch_ground_truths):
        raise ValueError(
            f"Number of prediction frames ({len(batch_predictions)}) must match "
            f"number of ground-truth frames ({len(batch_ground_truths)})."
        )

    # Aggregate all predictions and GTs, tagging each with a frame index
    # to prevent cross-frame matching.
    # We achieve this by adding a unique frame-level offset to a dummy coordinate
    # dimension -- but the simpler correct approach for AP is:
    # accumulate per-frame TP/FP decisions in score-sorted order.

    metrics: Dict[str, float] = {}
    all_aps: List[float] = []

    for thresh in thresholds:
        threshold_aps: List[float] = []

        for cls in classes:
            # Gather all scored predictions for this class across frames
            all_preds: List[Tuple[float, int, int]] = []  # (score, frame_idx, pred_idx_in_frame)
            all_gts_per_frame: List[List[GroundTruth]] = []

            for frame_idx, (preds, gts) in enumerate(
                zip(batch_predictions, batch_ground_truths)
            ):
                cls_preds_in_frame = [
                    (i, p) for i, p in enumerate(preds) if p["label"] == cls
                ]
                cls_gts_in_frame = [g for g in gts if g["label"] == cls]

                for local_idx, pred in cls_preds_in_frame:
                    all_preds.append((pred["score"], frame_idx, local_idx))

                all_gts_per_frame.append(cls_gts_in_frame)

            # Sort all predictions by score descending
            all_preds.sort(key=lambda x: x[0], reverse=True)

            # Track which GT elements have been matched per frame
            matched_per_frame: List[List[bool]] = [
                [False] * len(gts) for gts in all_gts_per_frame
            ]

            tp_flags: List[bool] = []
            total_gt = sum(len(gts) for gts in all_gts_per_frame)

            for score, frame_idx, pred_local_idx in all_preds:
                pred = batch_predictions[frame_idx][pred_local_idx]
                pred_points = pred["points"]
                frame_gts = all_gts_per_frame[frame_idx]

                best_dist = float("inf")
                best_gt_idx = -1

                for gt_idx, gt in enumerate(frame_gts):
                    if matched_per_frame[frame_idx][gt_idx]:
                        continue
                    dist = chamfer_distance(pred_points, gt["points"])
                    if dist < best_dist:
                        best_dist = dist
                        best_gt_idx = gt_idx

                if best_dist <= thresh and best_gt_idx >= 0:
                    tp_flags.append(True)
                    matched_per_frame[frame_idx][best_gt_idx] = True
                else:
                    tp_flags.append(False)

            ap = _compute_ap(tp_flags, total_gt)
            metrics[f"AP/{cls}@{thresh:.1f}"] = ap
            threshold_aps.append(ap)
            all_aps.append(ap)

        metrics[f"mAP@{thresh:.1f}"] = float(np.mean(threshold_aps)) if threshold_aps else 0.0

    metrics["mAP"] = float(np.mean(all_aps)) if all_aps else 0.0

    # Compute mean Chamfer and Frechet distances across all frames
    chamfer_dists: List[float] = []
    frechet_dists: List[float] = []

    for preds, gts in zip(batch_predictions, batch_ground_truths):
        for cls in classes:
            cls_preds = sorted(
                [p for p in preds if p["label"] == cls],
                key=lambda x: x["score"],
                reverse=True,
            )
            cls_gts = [g for g in gts if g["label"] == cls]
            matched_flags = [False] * len(cls_gts)

            for pred in cls_preds:
                best_dist = float("inf")
                best_idx = -1
                for gt_idx, gt in enumerate(cls_gts):
                    if matched_flags[gt_idx]:
                        continue
                    dist = chamfer_distance(pred["points"], gt["points"])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = gt_idx

                if best_idx >= 0 and best_dist < float("inf"):
                    matched_flags[best_idx] = True
                    chamfer_dists.append(best_dist)
                    frechet_dists.append(
                        frechet_distance(pred["points"], cls_gts[best_idx]["points"])
                    )

    metrics["chamfer/mean"] = float(np.mean(chamfer_dists)) if chamfer_dists else 0.0
    metrics["frechet/mean"] = float(np.mean(frechet_dists)) if frechet_dists else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def compute_chamfer_matrix(
    polylines_a: Sequence[Polyline],
    polylines_b: Sequence[Polyline],
) -> np.ndarray:
    """Compute pairwise Chamfer distance matrix between two sets of polylines.

    Parameters
    ----------
    polylines_a : sequence of np.ndarray
        First set of polylines.
    polylines_b : sequence of np.ndarray
        Second set of polylines.

    Returns
    -------
    np.ndarray, shape (len(polylines_a), len(polylines_b))
        Pairwise Chamfer distance matrix.
    """
    m = len(polylines_a)
    n = len(polylines_b)
    matrix = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        for j in range(n):
            matrix[i, j] = chamfer_distance(polylines_a[i], polylines_b[j])

    return matrix


def hungarian_match(
    predictions: List[Prediction],
    ground_truths: List[GroundTruth],
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Match predictions to ground truths using the Hungarian algorithm.

    Uses scipy's linear_sum_assignment for optimal matching based on Chamfer
    distance, then filters matches exceeding the threshold.

    Parameters
    ----------
    predictions : list of dict
        Predictions with 'points' key.
    ground_truths : list of dict
        Ground truths with 'points' key.
    threshold : float
        Maximum Chamfer distance for a valid match.

    Returns
    -------
    list of (pred_idx, gt_idx, distance)
        Valid matched pairs with their Chamfer distances.
    """
    from scipy.optimize import linear_sum_assignment

    if not predictions or not ground_truths:
        return []

    pred_polylines = [p["points"] for p in predictions]
    gt_polylines = [g["points"] for g in ground_truths]

    cost_matrix = compute_chamfer_matrix(pred_polylines, gt_polylines)

    # Hungarian algorithm (minimization)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int, float]] = []
    for row, col in zip(row_indices, col_indices):
        dist = cost_matrix[row, col]
        if dist <= threshold:
            matches.append((int(row), int(col), float(dist)))

    return matches


def evaluate(
    predictions: Union[List[Prediction], List[List[Prediction]]],
    ground_truths: Union[List[GroundTruth], List[List[GroundTruth]]],
    thresholds: Optional[List[float]] = None,
    classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Unified evaluation entry point supporting both single-frame and batch inputs.

    Automatically detects whether the input is a single frame (list of dicts) or
    a batch (list of lists of dicts) and dispatches accordingly.

    Parameters
    ----------
    predictions : list of dict or list of list of dict
        Single frame: list of prediction dicts.
        Batch: list of frames, each containing a list of prediction dicts.
    ground_truths : list of dict or list of list of dict
        Single frame: list of ground-truth dicts.
        Batch: list of frames, each containing a list of ground-truth dicts.
    thresholds : list of float, optional
        Chamfer distance thresholds. Defaults to [0.5, 1.0, 1.5].
    classes : list of str, optional
        Class names. Defaults to DEFAULT_CLASSES.

    Returns
    -------
    dict
        Metric name -> value.
    """
    # Detect batched vs single-frame input
    is_batch = False
    if predictions and isinstance(predictions[0], list):
        is_batch = True
    elif not predictions and ground_truths and isinstance(ground_truths[0], list):
        is_batch = True

    if is_batch:
        return evaluate_batch(predictions, ground_truths, thresholds, classes)  # type: ignore[arg-type]
    else:
        return evaluate_frame(predictions, ground_truths, thresholds, classes)  # type: ignore[arg-type]
