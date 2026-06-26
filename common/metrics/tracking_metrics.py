"""Multi-Object Tracking (MOT) metrics for autonomous driving perception.

This module implements standard MOT metrics used in autonomous driving benchmarks
(nuScenes, KITTI, etc.) including AMOTA, AMOTP, IDF1, MOTA, ID switches,
track fragmentation, and mostly-tracked/mostly-lost ratios.

Matching is performed using center-distance (BEV Euclidean) with Hungarian assignment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FramePredictions = Dict[str, Any]
"""Expected keys: 'boxes' (N, 7) or (N, 4), 'scores' (N,), 'track_ids' (N,), 'labels' (N,)."""

FrameGroundTruths = Dict[str, Any]
"""Expected keys: 'boxes' (M, 7) or (M, 4), 'track_ids' (M,), 'labels' (M,)."""


# ---------------------------------------------------------------------------
# Utility: center extraction
# ---------------------------------------------------------------------------

def _get_centers(boxes: np.ndarray) -> np.ndarray:
    """Extract BEV (x, y) centers from boxes.

    Supports:
        - (N, 7): [x, y, z, w, l, h, yaw] -> center is (x, y)
        - (N, 4): [x1, y1, x2, y2] -> center is midpoint
        - (N, 2): already centers

    Parameters
    ----------
    boxes : np.ndarray
        Array of shape (N, D) where D in {2, 4, 7}.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with (x, y) centers.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    dim = boxes.shape[1]
    if dim == 7:
        return boxes[:, :2]
    elif dim == 4:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        return np.stack([cx, cy], axis=1)
    elif dim == 2:
        return boxes
    else:
        raise ValueError(
            f"Unsupported box dimension {dim}. Expected 2, 4, or 7."
        )


# ---------------------------------------------------------------------------
# Utility: pairwise distance and matching
# ---------------------------------------------------------------------------

def _pairwise_distances(centers_a: np.ndarray, centers_b: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between two sets of 2D centers.

    Parameters
    ----------
    centers_a : np.ndarray, shape (M, 2)
    centers_b : np.ndarray, shape (N, 2)

    Returns
    -------
    np.ndarray
        Distance matrix of shape (M, N).
    """
    diff = centers_a[:, np.newaxis, :] - centers_b[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def _match_frame(
    gt_centers: np.ndarray,
    pred_centers: np.ndarray,
    distance_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match ground truths to predictions using Hungarian algorithm.

    Parameters
    ----------
    gt_centers : np.ndarray, shape (M, 2)
    pred_centers : np.ndarray, shape (N, 2)
    distance_threshold : float
        Maximum allowable distance for a valid match.

    Returns
    -------
    matches : list of (gt_idx, pred_idx) tuples
    unmatched_gt : list of gt indices with no match
    unmatched_pred : list of pred indices with no match
    """
    num_gt = gt_centers.shape[0]
    num_pred = pred_centers.shape[0]

    if num_gt == 0 and num_pred == 0:
        return [], [], []
    if num_gt == 0:
        return [], [], list(range(num_pred))
    if num_pred == 0:
        return [], list(range(num_gt)), []

    cost_matrix = _pairwise_distances(gt_centers, pred_centers)

    # Hungarian assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    unmatched_gt = set(range(num_gt))
    unmatched_pred = set(range(num_pred))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= distance_threshold:
            matches.append((r, c))
            unmatched_gt.discard(r)
            unmatched_pred.discard(c)

    return matches, sorted(unmatched_gt), sorted(unmatched_pred)


# ---------------------------------------------------------------------------
# Core per-sequence accumulation
# ---------------------------------------------------------------------------

class _TrackingAccumulator:
    """Accumulates frame-by-frame tracking results for a single sequence.

    This is an internal helper that computes raw counts needed for all metrics.
    """

    def __init__(self, distance_threshold: float = 2.0) -> None:
        self.distance_threshold = distance_threshold

        # Per-frame accumulators
        self.num_gt_per_frame: List[int] = []
        self.num_pred_per_frame: List[int] = []
        self.num_matches_per_frame: List[int] = []
        self.num_fp_per_frame: List[int] = []
        self.num_fn_per_frame: List[int] = []
        self.num_id_switches_per_frame: List[int] = []
        self.match_distances_per_frame: List[List[float]] = []

        # Track-level information
        # gt_track_id -> list of (frame_idx, matched_pred_track_id or None)
        self.gt_track_history: Dict[int, List[Tuple[int, Optional[int]]]] = {}

        # Previous frame matching: gt_track_id -> last matched pred_track_id
        self._prev_match: Dict[int, int] = {}

    def process_frame(
        self,
        frame_idx: int,
        gt_boxes: np.ndarray,
        gt_track_ids: np.ndarray,
        gt_labels: np.ndarray,
        pred_boxes: np.ndarray,
        pred_track_ids: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: Optional[np.ndarray] = None,
        score_threshold: float = 0.0,
    ) -> None:
        """Process a single frame and accumulate statistics.

        Parameters
        ----------
        frame_idx : int
            Index of this frame in the sequence.
        gt_boxes : np.ndarray
            Ground truth boxes, shape (M, D).
        gt_track_ids : np.ndarray
            GT track IDs, shape (M,).
        gt_labels : np.ndarray
            GT class labels, shape (M,).
        pred_boxes : np.ndarray
            Predicted boxes, shape (N, D).
        pred_track_ids : np.ndarray
            Predicted track IDs, shape (N,).
        pred_labels : np.ndarray
            Predicted class labels, shape (N,).
        pred_scores : np.ndarray or None
            Confidence scores, shape (N,). Used for score thresholding.
        score_threshold : float
            Minimum score to keep a prediction.
        """
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
        gt_track_ids = np.asarray(gt_track_ids, dtype=np.int64)
        gt_labels = np.asarray(gt_labels, dtype=np.int64)
        pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
        pred_track_ids = np.asarray(pred_track_ids, dtype=np.int64)
        pred_labels = np.asarray(pred_labels, dtype=np.int64)

        # Filter predictions by score
        if pred_scores is not None:
            pred_scores = np.asarray(pred_scores, dtype=np.float64)
            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_track_ids = pred_track_ids[keep]
            pred_labels = pred_labels[keep]

        # Reshape if needed
        if gt_boxes.ndim == 1 and gt_boxes.size > 0:
            gt_boxes = gt_boxes.reshape(1, -1)
        if pred_boxes.ndim == 1 and pred_boxes.size > 0:
            pred_boxes = pred_boxes.reshape(1, -1)
        if gt_boxes.size == 0:
            gt_boxes = gt_boxes.reshape(0, 2)
        if pred_boxes.size == 0:
            pred_boxes = pred_boxes.reshape(0, 2)

        gt_centers = _get_centers(gt_boxes)
        pred_centers = _get_centers(pred_boxes)

        num_gt = gt_centers.shape[0]
        num_pred = pred_centers.shape[0]

        matches, unmatched_gt_idxs, unmatched_pred_idxs = _match_frame(
            gt_centers, pred_centers, self.distance_threshold
        )

        # Count ID switches
        id_switches = 0
        current_match: Dict[int, int] = {}
        match_dists: List[float] = []

        for gt_idx, pred_idx in matches:
            gt_tid = int(gt_track_ids[gt_idx])
            pred_tid = int(pred_track_ids[pred_idx])
            current_match[gt_tid] = pred_tid

            # Compute match distance
            dist = np.linalg.norm(gt_centers[gt_idx] - pred_centers[pred_idx])
            match_dists.append(float(dist))

            # Check for ID switch
            if gt_tid in self._prev_match:
                if self._prev_match[gt_tid] != pred_tid:
                    id_switches += 1

            # Record GT track history
            if gt_tid not in self.gt_track_history:
                self.gt_track_history[gt_tid] = []
            self.gt_track_history[gt_tid].append((frame_idx, pred_tid))

        # Record unmatched GTs in track history
        for gt_idx in unmatched_gt_idxs:
            gt_tid = int(gt_track_ids[gt_idx])
            if gt_tid not in self.gt_track_history:
                self.gt_track_history[gt_tid] = []
            self.gt_track_history[gt_tid].append((frame_idx, None))

        # Update previous match state
        self._prev_match = current_match

        num_matches = len(matches)
        num_fp = len(unmatched_pred_idxs)
        num_fn = len(unmatched_gt_idxs)

        self.num_gt_per_frame.append(num_gt)
        self.num_pred_per_frame.append(num_pred)
        self.num_matches_per_frame.append(num_matches)
        self.num_fp_per_frame.append(num_fp)
        self.num_fn_per_frame.append(num_fn)
        self.num_id_switches_per_frame.append(id_switches)
        self.match_distances_per_frame.append(match_dists)

    @property
    def total_gt(self) -> int:
        """Total number of GT objects across all frames."""
        return int(np.sum(self.num_gt_per_frame))

    @property
    def total_fp(self) -> int:
        return int(np.sum(self.num_fp_per_frame))

    @property
    def total_fn(self) -> int:
        return int(np.sum(self.num_fn_per_frame))

    @property
    def total_id_switches(self) -> int:
        return int(np.sum(self.num_id_switches_per_frame))

    @property
    def total_matches(self) -> int:
        return int(np.sum(self.num_matches_per_frame))


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_mota(accumulator: _TrackingAccumulator) -> float:
    """Compute Multi-Object Tracking Accuracy (MOTA).

    MOTA = 1 - (FN + FP + ID_switches) / total_GT

    Parameters
    ----------
    accumulator : _TrackingAccumulator
        Accumulated tracking results.

    Returns
    -------
    float
        MOTA value. Can be negative if errors exceed GT count.
    """
    total_gt = accumulator.total_gt
    if total_gt == 0:
        return 0.0
    errors = accumulator.total_fn + accumulator.total_fp + accumulator.total_id_switches
    return 1.0 - errors / total_gt


def compute_motp(accumulator: _TrackingAccumulator) -> float:
    """Compute Multi-Object Tracking Precision (MOTP).

    MOTP = sum(match_distances) / total_matches

    Lower is better. Returns 0 if no matches exist.

    Parameters
    ----------
    accumulator : _TrackingAccumulator
        Accumulated tracking results.

    Returns
    -------
    float
        Mean position error for matched pairs.
    """
    all_dists = []
    for dists in accumulator.match_distances_per_frame:
        all_dists.extend(dists)
    if len(all_dists) == 0:
        return 0.0
    return float(np.mean(all_dists))


def compute_id_switches(accumulator: _TrackingAccumulator) -> int:
    """Return total number of ID switches.

    An ID switch occurs when a GT object is matched to a different predicted
    track ID than in the previous frame.

    Parameters
    ----------
    accumulator : _TrackingAccumulator
        Accumulated tracking results.

    Returns
    -------
    int
        Total ID switch count.
    """
    return accumulator.total_id_switches


def compute_fragmentations(accumulator: _TrackingAccumulator) -> int:
    """Compute track fragmentation count.

    A fragmentation occurs when a GT track that was being tracked (matched)
    becomes untracked (unmatched) and then is matched again later. Each such
    interruption counts as one fragmentation.

    Parameters
    ----------
    accumulator : _TrackingAccumulator
        Accumulated tracking results.

    Returns
    -------
    int
        Total fragmentation count across all GT tracks.
    """
    total_frags = 0
    for gt_tid, history in accumulator.gt_track_history.items():
        # history is a list of (frame_idx, matched_pred_tid_or_None)
        # Sort by frame index to ensure correct temporal ordering
        history_sorted = sorted(history, key=lambda x: x[0])
        was_tracked = False
        for _, pred_tid in history_sorted:
            if pred_tid is not None:
                if was_tracked is False and was_tracked is not None:
                    # First time being tracked or resuming after gap
                    pass
                was_tracked = True
            else:
                if was_tracked:
                    was_tracked = False
        # Count transitions from tracked -> untracked -> tracked
        # Re-do with explicit fragmentation counting
        pass

    # Recompute properly
    total_frags = 0
    for gt_tid, history in accumulator.gt_track_history.items():
        history_sorted = sorted(history, key=lambda x: x[0])
        frags = 0
        prev_state: Optional[bool] = None  # None = no prior state, True = tracked, False = untracked
        for _, pred_tid in history_sorted:
            is_tracked = pred_tid is not None
            if prev_state is True and not is_tracked:
                # Transition from tracked to untracked - potential fragmentation
                pass
            elif prev_state is False and is_tracked:
                # Transition from untracked back to tracked - this is a fragmentation
                frags += 1
            prev_state = is_tracked
        total_frags += frags

    return total_frags


def compute_mt_ml(
    accumulator: _TrackingAccumulator,
    mt_threshold: float = 0.8,
    ml_threshold: float = 0.2,
) -> Tuple[float, float, int, int, int]:
    """Compute Mostly Tracked (MT) and Mostly Lost (ML) ratios.

    A GT track is:
    - Mostly Tracked (MT) if it is matched in >= mt_threshold of its lifespan frames
    - Mostly Lost (ML) if it is matched in <= ml_threshold of its lifespan frames
    - Partially Tracked (PT) otherwise

    Parameters
    ----------
    accumulator : _TrackingAccumulator
        Accumulated tracking results.
    mt_threshold : float
        Fraction threshold for mostly tracked (default 0.8).
    ml_threshold : float
        Fraction threshold for mostly lost (default 0.2).

    Returns
    -------
    mt_ratio : float
        Fraction of GT tracks that are mostly tracked.
    ml_ratio : float
        Fraction of GT tracks that are mostly lost.
    num_mt : int
        Count of mostly tracked GT tracks.
    num_ml : int
        Count of mostly lost GT tracks.
    num_total : int
        Total number of unique GT tracks.
    """
    if not accumulator.gt_track_history:
        return 0.0, 0.0, 0, 0, 0

    num_mt = 0
    num_ml = 0
    num_total = len(accumulator.gt_track_history)

    for gt_tid, history in accumulator.gt_track_history.items():
        total_frames = len(history)
        if total_frames == 0:
            num_ml += 1
            continue
        tracked_frames = sum(1 for _, pred_tid in history if pred_tid is not None)
        tracked_ratio = tracked_frames / total_frames

        if tracked_ratio >= mt_threshold:
            num_mt += 1
        elif tracked_ratio <= ml_threshold:
            num_ml += 1

    mt_ratio = num_mt / num_total if num_total > 0 else 0.0
    ml_ratio = num_ml / num_total if num_total > 0 else 0.0

    return mt_ratio, ml_ratio, num_mt, num_ml, num_total


def compute_idf1(accumulator: _TrackingAccumulator) -> float:
    """Compute ID F1 score (IDF1).

    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

    Where:
    - IDTP: number of correctly identified detections (matched with consistent ID)
    - IDFP: false positive identifications
    - IDFN: false negative identifications

    We compute IDTP as the total number of matches across all frames, IDFP as
    total false positives, and IDFN as total false negatives. This corresponds
    to the global minimum-cost ID assignment interpretation.

    Parameters
    ----------
    accumulator : _TrackingAccumulator
        Accumulated tracking results.

    Returns
    -------
    float
        IDF1 score in [0, 1].
    """
    # For a proper IDF1 we need to find the best global ID assignment.
    # We use the approach: for each unique GT track, find the predicted track ID
    # that matches it most often, then compute true positives under that assignment.

    # Build co-occurrence matrix: gt_track_id -> {pred_track_id -> count}
    gt_pred_cooccurrence: Dict[int, Dict[int, int]] = {}
    for gt_tid, history in accumulator.gt_track_history.items():
        gt_pred_cooccurrence[gt_tid] = {}
        for _, pred_tid in history:
            if pred_tid is not None:
                gt_pred_cooccurrence[gt_tid][pred_tid] = (
                    gt_pred_cooccurrence[gt_tid].get(pred_tid, 0) + 1
                )

    # Get all unique GT and pred track IDs
    all_gt_tids = list(gt_pred_cooccurrence.keys())
    all_pred_tids_set: set = set()
    for counts in gt_pred_cooccurrence.values():
        all_pred_tids_set.update(counts.keys())
    all_pred_tids = list(all_pred_tids_set)

    if not all_gt_tids or not all_pred_tids:
        # No matches at all
        total_gt = accumulator.total_gt
        total_pred = sum(accumulator.num_pred_per_frame)
        if total_gt == 0 and total_pred == 0:
            return 1.0
        return 0.0

    # Build cost matrix for Hungarian assignment (maximize matches -> minimize negative)
    gt_id_to_idx = {tid: i for i, tid in enumerate(all_gt_tids)}
    pred_id_to_idx = {tid: i for i, tid in enumerate(all_pred_tids)}

    num_gt_ids = len(all_gt_tids)
    num_pred_ids = len(all_pred_tids)
    cost_matrix = np.zeros((num_gt_ids, num_pred_ids), dtype=np.float64)

    for gt_tid, pred_counts in gt_pred_cooccurrence.items():
        gi = gt_id_to_idx[gt_tid]
        for pred_tid, count in pred_counts.items():
            pi = pred_id_to_idx[pred_tid]
            cost_matrix[gi, pi] = count

    # Hungarian assignment to maximize total correct ID associations
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    idtp = 0
    for r, c in zip(row_ind, col_ind):
        idtp += int(cost_matrix[r, c])

    # Total GT appearances and total pred appearances
    total_gt_appearances = accumulator.total_gt
    total_pred_appearances = sum(accumulator.num_pred_per_frame)

    idfn = total_gt_appearances - idtp
    idfp = total_pred_appearances - idtp

    denominator = 2 * idtp + idfp + idfn
    if denominator == 0:
        return 1.0
    return 2 * idtp / denominator


# ---------------------------------------------------------------------------
# AMOTA / AMOTP (nuScenes style)
# ---------------------------------------------------------------------------

def _compute_mota_at_threshold(
    frames: List[Tuple[FrameGroundTruths, FramePredictions]],
    score_threshold: float,
    distance_threshold: float,
) -> Tuple[float, float]:
    """Compute MOTA and MOTP at a specific score threshold.

    Parameters
    ----------
    frames : list of (gt_dict, pred_dict) tuples
    score_threshold : float
        Minimum confidence to retain a prediction.
    distance_threshold : float
        Maximum distance for valid match.

    Returns
    -------
    mota : float
    motp : float
    """
    acc = _TrackingAccumulator(distance_threshold=distance_threshold)
    for frame_idx, (gt, pred) in enumerate(frames):
        gt_boxes = np.asarray(gt.get("boxes", np.empty((0, 2))), dtype=np.float64)
        gt_track_ids = np.asarray(gt.get("track_ids", []), dtype=np.int64)
        gt_labels = np.asarray(gt.get("labels", []), dtype=np.int64)

        pred_boxes = np.asarray(pred.get("boxes", np.empty((0, 2))), dtype=np.float64)
        pred_track_ids = np.asarray(pred.get("track_ids", []), dtype=np.int64)
        pred_labels = np.asarray(pred.get("labels", []), dtype=np.int64)
        pred_scores = np.asarray(pred.get("scores", []), dtype=np.float64)

        acc.process_frame(
            frame_idx=frame_idx,
            gt_boxes=gt_boxes,
            gt_track_ids=gt_track_ids,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_track_ids=pred_track_ids,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            score_threshold=score_threshold,
        )

    mota = compute_mota(acc)
    motp = compute_motp(acc)
    return mota, motp


def compute_amota_amotp(
    frames: List[Tuple[FrameGroundTruths, FramePredictions]],
    distance_threshold: float = 2.0,
    num_thresholds: int = 40,
) -> Tuple[float, float]:
    """Compute AMOTA and AMOTP (nuScenes-style).

    AMOTA averages MOTA computed at multiple recall thresholds (score thresholds)
    linearly spaced between 0 and 1 (exclusive of 1). This penalizes trackers
    that only perform well at a single operating point.

    AMOTP averages MOTP (mean position error) across the same thresholds,
    considering only thresholds where there are matches.

    Parameters
    ----------
    frames : list of (gt_dict, pred_dict) tuples
        Each tuple contains ground truth and prediction dicts for one frame.
    distance_threshold : float
        Maximum BEV distance for valid match (default 2.0 meters).
    num_thresholds : int
        Number of recall thresholds (default 40, nuScenes standard).

    Returns
    -------
    amota : float
        Average MOTA across thresholds. Clamped to [0, 1] per threshold before
        averaging (following nuScenes convention).
    amotp : float
        Average MOTP across thresholds where matches exist.
    """
    # Linearly spaced score thresholds from near-0 to near-1
    # nuScenes uses recall thresholds; we approximate by varying score threshold
    score_thresholds = np.linspace(0.0, 1.0, num_thresholds, endpoint=False)
    # Skip 0.0 to avoid including all predictions regardless of confidence
    score_thresholds = np.linspace(1.0 / num_thresholds, 1.0, num_thresholds, endpoint=False)

    mota_values = []
    motp_values = []

    for thresh in score_thresholds:
        mota, motp = _compute_mota_at_threshold(frames, thresh, distance_threshold)
        # Clamp MOTA to [0, 1] before averaging (nuScenes convention)
        mota_clamped = max(0.0, min(1.0, mota))
        mota_values.append(mota_clamped)
        if motp > 0:
            motp_values.append(motp)

    amota = float(np.mean(mota_values)) if mota_values else 0.0
    amotp = float(np.mean(motp_values)) if motp_values else 0.0

    return amota, amotp


# ---------------------------------------------------------------------------
# Single-sequence evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    predictions: Sequence[FramePredictions],
    ground_truths: Sequence[FrameGroundTruths],
    distance_threshold: float = 2.0,
    num_amota_thresholds: int = 40,
    score_threshold: float = 0.0,
) -> Dict[str, float]:
    """Evaluate all MOT metrics for a single tracking sequence.

    Parameters
    ----------
    predictions : sequence of prediction dicts
        Each dict has keys: 'boxes', 'scores', 'track_ids', 'labels'.
    ground_truths : sequence of ground truth dicts
        Each dict has keys: 'boxes', 'track_ids', 'labels'.
    distance_threshold : float
        Maximum BEV Euclidean distance for matching (default 2.0 m).
    num_amota_thresholds : int
        Number of thresholds for AMOTA/AMOTP computation (default 40).
    score_threshold : float
        Minimum score to retain predictions for standard metrics (default 0.0,
        meaning all predictions are kept). AMOTA/AMOTP sweep their own thresholds.

    Returns
    -------
    dict
        Dictionary of metric name -> value:
        - 'mota': MOTA
        - 'motp': MOTP (mean position error, lower is better)
        - 'idf1': IDF1 score
        - 'id_switches': number of ID switches
        - 'fragmentations': number of track fragmentations
        - 'mt_ratio': mostly tracked ratio
        - 'ml_ratio': mostly lost ratio
        - 'num_mt': number of mostly tracked GT tracks
        - 'num_ml': number of mostly lost GT tracks
        - 'num_gt_tracks': total unique GT tracks
        - 'amota': AMOTA
        - 'amotp': AMOTP
        - 'total_gt': total GT object appearances
        - 'total_fp': total false positives
        - 'total_fn': total false negatives
    """
    assert len(predictions) == len(ground_truths), (
        f"Number of prediction frames ({len(predictions)}) must match "
        f"ground truth frames ({len(ground_truths)})"
    )

    # Build accumulator for standard metrics
    acc = _TrackingAccumulator(distance_threshold=distance_threshold)
    for frame_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        gt_boxes = np.asarray(gt.get("boxes", np.empty((0, 2))), dtype=np.float64)
        gt_track_ids = np.asarray(gt.get("track_ids", []), dtype=np.int64)
        gt_labels = np.asarray(gt.get("labels", []), dtype=np.int64)

        pred_boxes = np.asarray(pred.get("boxes", np.empty((0, 2))), dtype=np.float64)
        pred_track_ids = np.asarray(pred.get("track_ids", []), dtype=np.int64)
        pred_labels = np.asarray(pred.get("labels", []), dtype=np.int64)
        pred_scores = np.asarray(pred.get("scores", []), dtype=np.float64)

        acc.process_frame(
            frame_idx=frame_idx,
            gt_boxes=gt_boxes,
            gt_track_ids=gt_track_ids,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes,
            pred_track_ids=pred_track_ids,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            score_threshold=score_threshold,
        )

    # Compute metrics
    mota = compute_mota(acc)
    motp = compute_motp(acc)
    idf1 = compute_idf1(acc)
    id_sw = compute_id_switches(acc)
    frags = compute_fragmentations(acc)
    mt_ratio, ml_ratio, num_mt, num_ml, num_total = compute_mt_ml(acc)

    # AMOTA/AMOTP
    frames_paired = list(zip(ground_truths, predictions))
    amota, amotp = compute_amota_amotp(
        frames_paired,
        distance_threshold=distance_threshold,
        num_thresholds=num_amota_thresholds,
    )

    return {
        "mota": mota,
        "motp": motp,
        "idf1": idf1,
        "id_switches": id_sw,
        "fragmentations": frags,
        "mt_ratio": mt_ratio,
        "ml_ratio": ml_ratio,
        "num_mt": num_mt,
        "num_ml": num_ml,
        "num_gt_tracks": num_total,
        "amota": amota,
        "amotp": amotp,
        "total_gt": acc.total_gt,
        "total_fp": acc.total_fp,
        "total_fn": acc.total_fn,
    }


# ---------------------------------------------------------------------------
# Batched (multi-sequence) evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    batch_predictions: Sequence[Sequence[FramePredictions]],
    batch_ground_truths: Sequence[Sequence[FrameGroundTruths]],
    distance_threshold: float = 2.0,
    num_amota_thresholds: int = 40,
    score_threshold: float = 0.0,
) -> Dict[str, float]:
    """Evaluate MOT metrics across multiple sequences (batched).

    Aggregates results by summing raw counts across sequences before computing
    ratios, yielding a single set of metrics reflecting overall performance.

    Parameters
    ----------
    batch_predictions : sequence of sequences of prediction dicts
        Outer sequence is over sequences; inner is over frames.
    batch_ground_truths : sequence of sequences of ground truth dicts
        Same structure as batch_predictions.
    distance_threshold : float
        Maximum BEV distance for matching (default 2.0 m).
    num_amota_thresholds : int
        Number of thresholds for AMOTA/AMOTP (default 40).
    score_threshold : float
        Minimum prediction score for standard metrics (default 0.0).

    Returns
    -------
    dict
        Aggregated metric dictionary (same keys as evaluate_sequence) plus:
        - 'per_sequence': list of per-sequence metric dicts
    """
    assert len(batch_predictions) == len(batch_ground_truths), (
        f"Number of prediction sequences ({len(batch_predictions)}) must match "
        f"ground truth sequences ({len(batch_ground_truths)})"
    )

    # Aggregate accumulator across all sequences
    global_acc = _TrackingAccumulator(distance_threshold=distance_threshold)
    all_frames_paired: List[Tuple[FrameGroundTruths, FramePredictions]] = []
    per_sequence_results: List[Dict[str, float]] = []

    frame_offset = 0
    for seq_idx, (preds, gts) in enumerate(zip(batch_predictions, batch_ground_truths)):
        assert len(preds) == len(gts), (
            f"Sequence {seq_idx}: prediction frame count ({len(preds)}) != "
            f"GT frame count ({len(gts)})"
        )

        # Per-sequence evaluation
        seq_result = evaluate_sequence(
            preds, gts,
            distance_threshold=distance_threshold,
            num_amota_thresholds=num_amota_thresholds,
            score_threshold=score_threshold,
        )
        per_sequence_results.append(seq_result)

        # Accumulate globally (with offset frame index to avoid collisions
        # in track history across sequences - use unique GT track IDs with seq prefix)
        for frame_idx, (pred, gt) in enumerate(zip(preds, gts)):
            gt_boxes = np.asarray(gt.get("boxes", np.empty((0, 2))), dtype=np.float64)
            gt_track_ids = np.asarray(gt.get("track_ids", []), dtype=np.int64)
            gt_labels = np.asarray(gt.get("labels", []), dtype=np.int64)

            pred_boxes = np.asarray(pred.get("boxes", np.empty((0, 2))), dtype=np.float64)
            pred_track_ids = np.asarray(pred.get("track_ids", []), dtype=np.int64)
            pred_labels = np.asarray(pred.get("labels", []), dtype=np.int64)
            pred_scores = np.asarray(pred.get("scores", []), dtype=np.float64)

            # Offset track IDs to avoid inter-sequence collisions
            seq_id_offset = seq_idx * 1_000_000
            gt_track_ids_offset = gt_track_ids + seq_id_offset
            pred_track_ids_offset = pred_track_ids + seq_id_offset

            global_acc.process_frame(
                frame_idx=frame_offset + frame_idx,
                gt_boxes=gt_boxes,
                gt_track_ids=gt_track_ids_offset,
                gt_labels=gt_labels,
                pred_boxes=pred_boxes,
                pred_track_ids=pred_track_ids_offset,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                score_threshold=score_threshold,
            )

            all_frames_paired.append((gt, pred))

        frame_offset += len(preds)

    # Compute global metrics
    mota = compute_mota(global_acc)
    motp = compute_motp(global_acc)
    idf1 = compute_idf1(global_acc)
    id_sw = compute_id_switches(global_acc)
    frags = compute_fragmentations(global_acc)
    mt_ratio, ml_ratio, num_mt, num_ml, num_total = compute_mt_ml(global_acc)

    # AMOTA/AMOTP over all frames
    amota, amotp = compute_amota_amotp(
        all_frames_paired,
        distance_threshold=distance_threshold,
        num_thresholds=num_amota_thresholds,
    )

    return {
        "mota": mota,
        "motp": motp,
        "idf1": idf1,
        "id_switches": id_sw,
        "fragmentations": frags,
        "mt_ratio": mt_ratio,
        "ml_ratio": ml_ratio,
        "num_mt": num_mt,
        "num_ml": num_ml,
        "num_gt_tracks": num_total,
        "amota": amota,
        "amotp": amotp,
        "total_gt": global_acc.total_gt,
        "total_fp": global_acc.total_fp,
        "total_fn": global_acc.total_fn,
        "per_sequence": per_sequence_results,
    }
