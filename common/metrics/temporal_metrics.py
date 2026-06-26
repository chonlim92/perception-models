"""Temporal Consistency Metrics for Autonomous Driving Perception.

This module provides metrics that evaluate the temporal consistency and stability
of perception model predictions across sequential frames. These metrics are
critical for autonomous driving where smooth, physically-plausible predictions
are required for safe planning and control.

Metrics implemented:
    - Map Consistency: Measures prediction stability of HD map elements between
      consecutive frames after ego-motion compensation.
    - Streaming AP: Latency-aware Average Precision that accounts for processing
      delay in real-time perception systems.
    - Temporal Smoothness: Quantifies jitter/instability in tracked object
      predictions over time via acceleration analysis.
    - Velocity Consistency: Measures agreement between estimated velocity fields
      and finite-difference velocities from position changes.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d


# Type aliases for clarity
TransformMatrix = NDArray[np.float64]  # (4, 4)
BBox3D = NDArray[np.float64]  # (N, 7) or (N, 9) - x,y,z,w,l,h,yaw[,vx,vy]
Polyline = NDArray[np.float64]  # (M, 2) or (M, 3)


def _chamfer_distance(points_a: NDArray, points_b: NDArray) -> float:
    """Compute symmetric Chamfer distance between two point sets.

    Args:
        points_a: First point set, shape (N, D).
        points_b: Second point set, shape (M, D).

    Returns:
        Mean symmetric Chamfer distance (average of both directed distances).
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")

    dist_matrix = cdist(points_a, points_b, metric="euclidean")
    # Directed distance A -> B
    min_a_to_b = np.min(dist_matrix, axis=1)
    # Directed distance B -> A
    min_b_to_a = np.min(dist_matrix, axis=0)

    chamfer = 0.5 * (np.mean(min_a_to_b) + np.mean(min_b_to_a))
    return float(chamfer)


def _polygon_iou_2d(poly_a: NDArray, poly_b: NDArray) -> float:
    """Compute 2D IoU between two polygons using a rasterization approach.

    Uses a grid-based approximation for polygon IoU computation that avoids
    the dependency on Shapely while remaining reasonably accurate.

    Args:
        poly_a: Polygon vertices, shape (N, 2).
        poly_b: Polygon vertices, shape (M, 2).

    Returns:
        Intersection over Union in [0, 1].
    """
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0

    # Determine bounding box of both polygons combined
    all_pts = np.vstack([poly_a[:, :2], poly_b[:, :2]])
    min_xy = np.min(all_pts, axis=0)
    max_xy = np.max(all_pts, axis=0)
    extent = max_xy - min_xy

    if np.any(extent < 1e-8):
        return 0.0

    # Rasterize at ~100 points per largest dimension
    resolution = max(extent) / 100.0
    grid_x = np.arange(min_xy[0], max_xy[0], resolution)
    grid_y = np.arange(min_xy[1], max_xy[1], resolution)

    if len(grid_x) == 0 or len(grid_y) == 0:
        return 0.0

    xx, yy = np.meshgrid(grid_x, grid_y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    def _point_in_polygon(pts: NDArray, polygon: NDArray) -> NDArray:
        """Ray-casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = np.zeros(len(pts), dtype=bool)
        x, y = pts[:, 0], pts[:, 1]

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i, 0], polygon[i, 1]
            xj, yj = polygon[j, 0], polygon[j, 1]

            # Check if ray from point crosses edge
            cond1 = (yi > y) != (yj > y)
            slope = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            cond2 = x < slope

            cross = cond1 & cond2
            inside = inside ^ cross
            j = i

        return inside

    mask_a = _point_in_polygon(points, poly_a[:, :2])
    mask_b = _point_in_polygon(points, poly_b[:, :2])

    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)

    if union == 0:
        return 0.0

    return float(intersection / union)


def _transform_points(points: NDArray, transform: TransformMatrix) -> NDArray:
    """Apply a 4x4 transformation matrix to a set of 2D or 3D points.

    Args:
        points: Point array of shape (N, 2) or (N, 3).
        transform: 4x4 homogeneous transformation matrix.

    Returns:
        Transformed points with same shape as input.
    """
    ndim = points.shape[1]
    n_points = points.shape[0]

    if ndim == 2:
        # Extend to 3D with z=0
        pts_3d = np.hstack([points, np.zeros((n_points, 1))])
    else:
        pts_3d = points

    # Convert to homogeneous coordinates
    pts_homo = np.hstack([pts_3d, np.ones((n_points, 1))])  # (N, 4)
    transformed = (transform @ pts_homo.T).T  # (N, 4)

    if ndim == 2:
        return transformed[:, :2]
    return transformed[:, :3]


def _compute_ap_at_threshold(
    scores: NDArray,
    matches: NDArray,
    n_gt: int,
) -> float:
    """Compute Average Precision given sorted scores and match indicators.

    Args:
        scores: Confidence scores sorted in descending order, shape (N,).
        matches: Boolean array indicating true positive matches, shape (N,).
        n_gt: Total number of ground truth instances.

    Returns:
        Average Precision value in [0, 1].
    """
    if n_gt == 0:
        return 0.0

    tp_cumsum = np.cumsum(matches).astype(np.float64)
    fp_cumsum = np.cumsum(~matches).astype(np.float64)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_gt

    # Prepend sentinel values
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    # Make precision monotonically decreasing (envelope)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute area under PR curve (11-point interpolation)
    recall_thresholds = np.linspace(0, 1, 11)
    ap = 0.0
    for r_thresh in recall_thresholds:
        prec_at_recall = precision[recall >= r_thresh]
        if len(prec_at_recall) > 0:
            ap += np.max(prec_at_recall)
    ap /= 11.0

    return float(ap)


# =============================================================================
# Map Consistency Metric
# =============================================================================


def compute_map_consistency(
    frames: List[Dict[str, Any]],
    ego_transforms: List[TransformMatrix],
    distance_metric: str = "chamfer",
    match_threshold: float = 5.0,
) -> Dict[str, float]:
    """Compute map prediction consistency across consecutive frames.

    Measures how stable HD map element predictions are between consecutive frames
    after compensating for ego-vehicle motion. High consistency indicates that the
    model produces stable map predictions regardless of ego-motion.

    Args:
        frames: List of frame dictionaries. Each frame must contain:
            - "map_elements": List of NDArrays, each of shape (M_i, 2) or (M_i, 3)
              representing polylines or polygon vertices.
            - "map_labels": List of int labels for each map element.
            - "timestamp": float, frame timestamp in seconds.
        ego_transforms: List of 4x4 transformation matrices of length len(frames)-1.
            ego_transforms[i] transforms points from frame i coordinate system to
            frame i+1 coordinate system.
        distance_metric: One of "chamfer" or "iou". Metric used to compare
            map elements between frames.
        match_threshold: Maximum distance (Chamfer) or minimum IoU to consider
            two map elements as matching across frames.

    Returns:
        Dictionary with:
            - "map_consistency_mean": Mean consistency score across all frame pairs.
            - "map_consistency_std": Standard deviation of consistency scores.
            - "map_consistency_per_frame": List of per-frame-pair consistency scores.
            - "map_match_rate": Fraction of elements successfully matched.
    """
    if len(frames) < 2:
        return {
            "map_consistency_mean": 1.0,
            "map_consistency_std": 0.0,
            "map_consistency_per_frame": [],
            "map_match_rate": 1.0,
        }

    assert len(ego_transforms) == len(frames) - 1, (
        f"Expected {len(frames) - 1} ego transforms, got {len(ego_transforms)}"
    )

    frame_consistencies: List[float] = []
    total_elements = 0
    total_matched = 0

    for i in range(len(frames) - 1):
        frame_t = frames[i]
        frame_t1 = frames[i + 1]
        transform = np.asarray(ego_transforms[i], dtype=np.float64)

        elements_t = frame_t.get("map_elements", [])
        labels_t = frame_t.get("map_labels", [])
        elements_t1 = frame_t1.get("map_elements", [])
        labels_t1 = frame_t1.get("map_labels", [])

        if len(elements_t) == 0 or len(elements_t1) == 0:
            continue

        # Transform frame t elements to frame t+1 coordinate system
        transformed_elements_t = []
        for elem in elements_t:
            elem = np.asarray(elem, dtype=np.float64)
            transformed_elements_t.append(_transform_points(elem, transform))

        # Build cost matrix between transformed frame t elements and frame t+1 elements
        n_t = len(transformed_elements_t)
        n_t1 = len(elements_t1)
        cost_matrix = np.full((n_t, n_t1), fill_value=1e6)

        for idx_a in range(n_t):
            for idx_b in range(n_t1):
                # Only match elements with same label if labels provided
                if labels_t and labels_t1:
                    if labels_t[idx_a] != labels_t1[idx_b]:
                        continue

                elem_a = transformed_elements_t[idx_a]
                elem_b = np.asarray(elements_t1[idx_b], dtype=np.float64)

                if distance_metric == "chamfer":
                    dist = _chamfer_distance(elem_a, elem_b)
                    cost_matrix[idx_a, idx_b] = dist
                elif distance_metric == "iou":
                    iou = _polygon_iou_2d(elem_a, elem_b)
                    # Convert IoU to cost (higher IoU = lower cost)
                    cost_matrix[idx_a, idx_b] = 1.0 - iou
                else:
                    raise ValueError(
                        f"Unknown distance_metric: {distance_metric}. "
                        f"Use 'chamfer' or 'iou'."
                    )

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Compute consistency for matched pairs
        pair_scores: List[float] = []
        matched_count = 0

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if distance_metric == "chamfer":
                if cost < match_threshold:
                    # Convert distance to consistency score [0, 1]
                    consistency = max(0.0, 1.0 - cost / match_threshold)
                    pair_scores.append(consistency)
                    matched_count += 1
            elif distance_metric == "iou":
                iou_val = 1.0 - cost
                if iou_val >= match_threshold:
                    pair_scores.append(iou_val)
                    matched_count += 1

        total_elements += max(n_t, n_t1)
        total_matched += matched_count

        if pair_scores:
            frame_consistencies.append(float(np.mean(pair_scores)))
        else:
            frame_consistencies.append(0.0)

    if not frame_consistencies:
        return {
            "map_consistency_mean": 0.0,
            "map_consistency_std": 0.0,
            "map_consistency_per_frame": [],
            "map_match_rate": 0.0,
        }

    return {
        "map_consistency_mean": float(np.mean(frame_consistencies)),
        "map_consistency_std": float(np.std(frame_consistencies)),
        "map_consistency_per_frame": frame_consistencies,
        "map_match_rate": float(total_matched / max(total_elements, 1)),
    }


# =============================================================================
# Streaming AP Metric
# =============================================================================


def compute_streaming_ap(
    frames: List[Dict[str, Any]],
    latency_ms: float = 100.0,
    iou_thresholds: Optional[List[float]] = None,
    distance_threshold: float = 2.0,
    match_by: str = "center_distance",
) -> Dict[str, float]:
    """Compute Streaming Average Precision accounting for processing latency.

    In real-time autonomous driving, predictions are evaluated against the ground
    truth state at the time predictions arrive (not at capture time). This metric
    shifts the ground truth forward in time by the processing latency to simulate
    the actual conditions under which predictions are used.

    Reference: Li et al., "Streaming Perception" (ECCV 2020).

    Args:
        frames: List of frame dictionaries. Each frame must contain:
            - "timestamp": float, frame capture timestamp in seconds.
            - "predictions": Dict with:
                - "boxes": NDArray of shape (N, 7) [x,y,z,w,l,h,yaw].
                - "scores": NDArray of shape (N,).
                - "labels": NDArray of shape (N,) integer class labels.
            - "ground_truth": Dict with:
                - "boxes": NDArray of shape (M, 7) [x,y,z,w,l,h,yaw].
                - "labels": NDArray of shape (M,) integer class labels.
        latency_ms: Processing latency in milliseconds. Predictions from frame t
            are evaluated against GT interpolated at time t + latency.
        iou_thresholds: List of IoU thresholds for AP computation. If None,
            uses center distance matching instead.
        distance_threshold: Maximum center distance for a true positive match
            (used when match_by="center_distance").
        match_by: Matching strategy - "center_distance" or "iou".

    Returns:
        Dictionary with:
            - "streaming_ap": Overall streaming AP across all classes.
            - "streaming_ap_per_class": Dict mapping class_id to AP.
            - "latency_ms": The latency value used.
            - "ap_drop_vs_offline": Difference between offline AP and streaming AP.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.7]

    latency_s = latency_ms / 1000.0

    # Extract timestamps and GT boxes for interpolation
    timestamps = np.array([f["timestamp"] for f in frames], dtype=np.float64)

    # Collect all class labels present
    all_classes = set()
    for f in frames:
        gt = f.get("ground_truth", {})
        gt_labels = gt.get("labels", np.array([]))
        if len(gt_labels) > 0:
            all_classes.update(gt_labels.tolist())
        pred = f.get("predictions", {})
        pred_labels = pred.get("labels", np.array([]))
        if len(pred_labels) > 0:
            all_classes.update(pred_labels.tolist())

    if not all_classes:
        return {
            "streaming_ap": 0.0,
            "streaming_ap_per_class": {},
            "latency_ms": latency_ms,
            "ap_drop_vs_offline": 0.0,
        }

    def _interpolate_gt_boxes(
        target_time: float, class_id: int
    ) -> Tuple[NDArray, int]:
        """Interpolate GT boxes at a target timestamp for a specific class.

        Uses linear interpolation between the two nearest frames.
        Returns interpolated boxes and count.
        """
        # Find bracketing frames
        if target_time <= timestamps[0]:
            gt = frames[0].get("ground_truth", {})
            gt_boxes = np.asarray(gt.get("boxes", np.empty((0, 7))))
            gt_labels = np.asarray(gt.get("labels", np.array([])))
            if len(gt_boxes) == 0:
                return np.empty((0, 7)), 0
            mask = gt_labels == class_id
            return gt_boxes[mask], int(np.sum(mask))

        if target_time >= timestamps[-1]:
            gt = frames[-1].get("ground_truth", {})
            gt_boxes = np.asarray(gt.get("boxes", np.empty((0, 7))))
            gt_labels = np.asarray(gt.get("labels", np.array([])))
            if len(gt_boxes) == 0:
                return np.empty((0, 7)), 0
            mask = gt_labels == class_id
            return gt_boxes[mask], int(np.sum(mask))

        # Find nearest frame index (snap to closest future frame)
        idx = np.searchsorted(timestamps, target_time, side="right")
        idx = min(idx, len(frames) - 1)

        # Use the frame at or just after target time
        gt = frames[idx].get("ground_truth", {})
        gt_boxes = np.asarray(gt.get("boxes", np.empty((0, 7))))
        gt_labels = np.asarray(gt.get("labels", np.array([])))
        if len(gt_boxes) == 0:
            return np.empty((0, 7)), 0
        mask = gt_labels == class_id
        return gt_boxes[mask], int(np.sum(mask))

    def _match_predictions_to_gt(
        pred_boxes: NDArray, gt_boxes: NDArray, threshold: float
    ) -> NDArray:
        """Match predictions to GT and return boolean match array."""
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        if n_pred == 0:
            return np.array([], dtype=bool)
        if n_gt == 0:
            return np.zeros(n_pred, dtype=bool)

        if match_by == "center_distance":
            pred_centers = pred_boxes[:, :3]
            gt_centers = gt_boxes[:, :3]
            dist_mat = cdist(pred_centers, gt_centers, metric="euclidean")
            matches = np.zeros(n_pred, dtype=bool)
            gt_matched = np.zeros(n_gt, dtype=bool)

            # Greedy matching by distance (predictions sorted by score externally)
            for p_idx in range(n_pred):
                min_dist_idx = np.argmin(dist_mat[p_idx])
                if (
                    dist_mat[p_idx, min_dist_idx] < threshold
                    and not gt_matched[min_dist_idx]
                ):
                    matches[p_idx] = True
                    gt_matched[min_dist_idx] = True
                    # Prevent re-matching
                    dist_mat[:, min_dist_idx] = 1e9

            return matches
        elif match_by == "iou":
            # 2D BEV IoU using box corners
            matches = np.zeros(n_pred, dtype=bool)
            gt_matched = np.zeros(n_gt, dtype=bool)

            # Simplified axis-aligned BEV IoU
            for p_idx in range(n_pred):
                best_iou = 0.0
                best_gt = -1
                for g_idx in range(n_gt):
                    if gt_matched[g_idx]:
                        continue
                    iou = _box_bev_iou(pred_boxes[p_idx], gt_boxes[g_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g_idx
                if best_iou >= threshold and best_gt >= 0:
                    matches[p_idx] = True
                    gt_matched[best_gt] = True

            return matches
        else:
            raise ValueError(f"Unknown match_by: {match_by}")

    def _box_bev_iou(box_a: NDArray, box_b: NDArray) -> float:
        """Compute BEV IoU between two boxes (simplified axis-aligned)."""
        # box format: [x, y, z, w, l, h, yaw]
        # For simplicity, use axis-aligned bounding box in BEV
        xa, ya, wa, la = box_a[0], box_a[1], box_a[3], box_a[4]
        xb, yb, wb, lb = box_b[0], box_b[1], box_b[3], box_b[4]

        # Half extents (approximate, ignoring yaw for this basic version)
        ha_w, ha_l = wa / 2, la / 2
        hb_w, hb_l = wb / 2, lb / 2

        # AABB overlap
        x_overlap = max(
            0, min(xa + ha_w, xb + hb_w) - max(xa - ha_w, xb - hb_w)
        )
        y_overlap = max(
            0, min(ya + ha_l, yb + hb_l) - max(ya - ha_l, yb - hb_l)
        )

        intersection = x_overlap * y_overlap
        area_a = wa * la
        area_b = wb * lb
        union = area_a + area_b - intersection

        if union < 1e-8:
            return 0.0
        return intersection / union

    # Compute streaming AP per class
    streaming_ap_per_class: Dict[int, float] = {}
    offline_ap_per_class: Dict[int, float] = {}

    for class_id in sorted(all_classes):
        # Gather all predictions and streaming GT for this class
        all_scores: List[float] = []
        all_matches_streaming: List[bool] = []
        all_matches_offline: List[bool] = []
        total_gt_streaming = 0
        total_gt_offline = 0

        for frame_idx, frame in enumerate(frames):
            pred = frame.get("predictions", {})
            pred_boxes = np.asarray(pred.get("boxes", np.empty((0, 7))))
            pred_scores = np.asarray(pred.get("scores", np.array([])))
            pred_labels = np.asarray(pred.get("labels", np.array([])))

            if len(pred_boxes) == 0:
                # Still count GT
                target_time = frame["timestamp"] + latency_s
                _, n_gt_stream = _interpolate_gt_boxes(target_time, class_id)
                total_gt_streaming += n_gt_stream

                gt = frame.get("ground_truth", {})
                gt_labels = np.asarray(gt.get("labels", np.array([])))
                total_gt_offline += int(np.sum(gt_labels == class_id))
                continue

            # Filter to this class
            class_mask = pred_labels == class_id
            if not np.any(class_mask):
                target_time = frame["timestamp"] + latency_s
                _, n_gt_stream = _interpolate_gt_boxes(target_time, class_id)
                total_gt_streaming += n_gt_stream

                gt = frame.get("ground_truth", {})
                gt_labels = np.asarray(gt.get("labels", np.array([])))
                total_gt_offline += int(np.sum(gt_labels == class_id))
                continue

            c_pred_boxes = pred_boxes[class_mask]
            c_pred_scores = pred_scores[class_mask]

            # Sort by score descending
            sort_idx = np.argsort(-c_pred_scores)
            c_pred_boxes = c_pred_boxes[sort_idx]
            c_pred_scores = c_pred_scores[sort_idx]

            # Streaming evaluation: match against GT at t + latency
            target_time = frame["timestamp"] + latency_s
            gt_boxes_stream, n_gt_stream = _interpolate_gt_boxes(
                target_time, class_id
            )
            total_gt_streaming += n_gt_stream

            threshold = (
                distance_threshold if match_by == "center_distance" else 0.5
            )
            matches_stream = _match_predictions_to_gt(
                c_pred_boxes, gt_boxes_stream, threshold
            )

            # Offline evaluation: match against GT at frame capture time
            gt = frame.get("ground_truth", {})
            gt_boxes_offline = np.asarray(gt.get("boxes", np.empty((0, 7))))
            gt_labels_offline = np.asarray(gt.get("labels", np.array([])))
            if len(gt_boxes_offline) > 0:
                offline_mask = gt_labels_offline == class_id
                gt_boxes_class = gt_boxes_offline[offline_mask]
                total_gt_offline += int(np.sum(offline_mask))
            else:
                gt_boxes_class = np.empty((0, 7))

            matches_offline = _match_predictions_to_gt(
                c_pred_boxes, gt_boxes_class, threshold
            )

            all_scores.extend(c_pred_scores.tolist())
            all_matches_streaming.extend(matches_stream.tolist())
            all_matches_offline.extend(matches_offline.tolist())

        # Compute AP from collected predictions
        if len(all_scores) == 0:
            streaming_ap_per_class[class_id] = 0.0
            offline_ap_per_class[class_id] = 0.0
            continue

        # Sort all predictions globally by score
        global_sort = np.argsort(-np.array(all_scores))
        sorted_matches_stream = np.array(all_matches_streaming)[global_sort]
        sorted_matches_offline = np.array(all_matches_offline)[global_sort]
        sorted_scores = np.array(all_scores)[global_sort]

        streaming_ap_per_class[class_id] = _compute_ap_at_threshold(
            sorted_scores, sorted_matches_stream, total_gt_streaming
        )
        offline_ap_per_class[class_id] = _compute_ap_at_threshold(
            sorted_scores, sorted_matches_offline, total_gt_offline
        )

    # Compute mean AP across classes
    if streaming_ap_per_class:
        streaming_map = float(np.mean(list(streaming_ap_per_class.values())))
        offline_map = float(np.mean(list(offline_ap_per_class.values())))
    else:
        streaming_map = 0.0
        offline_map = 0.0

    return {
        "streaming_ap": streaming_map,
        "streaming_ap_per_class": streaming_ap_per_class,
        "latency_ms": latency_ms,
        "ap_drop_vs_offline": offline_map - streaming_map,
    }


# =============================================================================
# Temporal Smoothness Metric
# =============================================================================


def compute_temporal_smoothness(
    frames: List[Dict[str, Any]],
    min_track_length: int = 3,
) -> Dict[str, float]:
    """Compute temporal smoothness of tracked object predictions.

    Measures the jitter in predictions by computing the second derivative
    (acceleration) of object positions over time. High jitter indicates
    unstable tracking that would be problematic for downstream planning.

    Args:
        frames: List of frame dictionaries. Each frame must contain:
            - "timestamp": float, frame timestamp in seconds.
            - "predictions": Dict with:
                - "boxes": NDArray of shape (N, 7) [x,y,z,w,l,h,yaw].
                - "track_ids": NDArray of shape (N,) integer track IDs.
                - "scores": NDArray of shape (N,) (optional).
        min_track_length: Minimum number of frames a track must span to be
            included in smoothness computation. Must be >= 3 for second
            derivative computation.

    Returns:
        Dictionary with:
            - "mean_position_jitter": Mean acceleration magnitude across all
              tracks and time steps (m/s^2).
            - "max_position_jitter": Maximum acceleration magnitude observed.
            - "mean_size_jitter": Mean second derivative of bounding box
              dimensions (w, l, h).
            - "max_size_jitter": Maximum size jitter observed.
            - "mean_heading_jitter": Mean second derivative of heading angle
              (rad/s^2).
            - "num_tracks_evaluated": Number of tracks meeting min length.
            - "jitter_per_track": Dict mapping track_id to mean jitter value.
    """
    min_track_length = max(min_track_length, 3)

    # Collect tracks: track_id -> list of (timestamp, box)
    tracks: Dict[int, List[Tuple[float, NDArray]]] = {}

    for frame in frames:
        timestamp = frame["timestamp"]
        pred = frame.get("predictions", {})
        boxes = np.asarray(pred.get("boxes", np.empty((0, 7))))
        track_ids = np.asarray(pred.get("track_ids", np.array([])))

        if len(boxes) == 0 or len(track_ids) == 0:
            continue

        for i, tid in enumerate(track_ids):
            tid_int = int(tid)
            if tid_int not in tracks:
                tracks[tid_int] = []
            tracks[tid_int].append((timestamp, boxes[i]))

    # Filter tracks by minimum length
    valid_tracks = {
        tid: sorted(data, key=lambda x: x[0])
        for tid, data in tracks.items()
        if len(data) >= min_track_length
    }

    if not valid_tracks:
        return {
            "mean_position_jitter": 0.0,
            "max_position_jitter": 0.0,
            "mean_size_jitter": 0.0,
            "max_size_jitter": 0.0,
            "mean_heading_jitter": 0.0,
            "num_tracks_evaluated": 0,
            "jitter_per_track": {},
        }

    all_pos_jitters: List[float] = []
    all_size_jitters: List[float] = []
    all_heading_jitters: List[float] = []
    jitter_per_track: Dict[int, float] = {}

    for tid, track_data in valid_tracks.items():
        timestamps_track = np.array([t for t, _ in track_data])
        boxes_track = np.array([b for _, b in track_data])

        # Positions: x, y, z
        positions = boxes_track[:, :3]  # (T, 3)
        # Sizes: w, l, h
        sizes = boxes_track[:, 3:6]  # (T, 3)
        # Heading: yaw
        headings = boxes_track[:, 6]  # (T,)

        # Time deltas
        dt = np.diff(timestamps_track)

        # Avoid division by zero for duplicate timestamps
        dt = np.maximum(dt, 1e-6)

        # First derivative (velocity)
        velocity = np.diff(positions, axis=0) / dt[:, np.newaxis]  # (T-1, 3)

        # Second derivative (acceleration / jitter)
        if len(velocity) >= 2:
            dt2 = (dt[:-1] + dt[1:]) / 2.0  # midpoint time deltas
            acceleration = (
                np.diff(velocity, axis=0) / dt2[:, np.newaxis]
            )  # (T-2, 3)
            pos_jitter = np.linalg.norm(acceleration, axis=1)  # (T-2,)
            all_pos_jitters.extend(pos_jitter.tolist())
        else:
            pos_jitter = np.array([0.0])

        # Size jitter
        size_velocity = np.diff(sizes, axis=0) / dt[:, np.newaxis]
        if len(size_velocity) >= 2:
            dt2 = (dt[:-1] + dt[1:]) / 2.0
            size_accel = np.diff(size_velocity, axis=0) / dt2[:, np.newaxis]
            size_jitter = np.linalg.norm(size_accel, axis=1)
            all_size_jitters.extend(size_jitter.tolist())
        else:
            size_jitter = np.array([0.0])

        # Heading jitter (handle angle wrapping)
        heading_diff = np.diff(headings)
        # Wrap to [-pi, pi]
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
        heading_vel = heading_diff / dt
        if len(heading_vel) >= 2:
            dt2 = (dt[:-1] + dt[1:]) / 2.0
            heading_accel = np.diff(heading_vel) / dt2
            all_heading_jitters.extend(np.abs(heading_accel).tolist())
        else:
            heading_accel = np.array([0.0])

        # Per-track mean jitter
        jitter_per_track[tid] = float(np.mean(pos_jitter))

    results = {
        "mean_position_jitter": float(np.mean(all_pos_jitters))
        if all_pos_jitters
        else 0.0,
        "max_position_jitter": float(np.max(all_pos_jitters))
        if all_pos_jitters
        else 0.0,
        "mean_size_jitter": float(np.mean(all_size_jitters))
        if all_size_jitters
        else 0.0,
        "max_size_jitter": float(np.max(all_size_jitters))
        if all_size_jitters
        else 0.0,
        "mean_heading_jitter": float(np.mean(all_heading_jitters))
        if all_heading_jitters
        else 0.0,
        "num_tracks_evaluated": len(valid_tracks),
        "jitter_per_track": jitter_per_track,
    }

    return results


# =============================================================================
# Velocity Consistency Metric
# =============================================================================


def compute_velocity_consistency(
    frames: List[Dict[str, Any]],
    min_track_length: int = 2,
    velocity_keys: Tuple[str, ...] = ("velocities",),
) -> Dict[str, float]:
    """Compute velocity prediction consistency for tracked objects.

    Compares the model's estimated velocity (from a velocity head or velocity
    field) with the finite-difference velocity computed from consecutive position
    observations. High consistency indicates the velocity predictions are
    physically plausible.

    Args:
        frames: List of frame dictionaries. Each frame must contain:
            - "timestamp": float, frame timestamp in seconds.
            - "predictions": Dict with:
                - "boxes": NDArray of shape (N, 7) [x,y,z,w,l,h,yaw].
                - "track_ids": NDArray of shape (N,) integer track IDs.
                - "velocities": NDArray of shape (N, 2) or (N, 3) estimated
                  velocities [vx, vy] or [vx, vy, vz].
        min_track_length: Minimum number of frames for a track to be evaluated.
        velocity_keys: Tuple of possible keys to look for velocity data in
            predictions dict.

    Returns:
        Dictionary with:
            - "velocity_mae": Mean absolute error between estimated and
              finite-difference velocities (m/s).
            - "velocity_rmse": Root mean squared error (m/s).
            - "velocity_correlation": Pearson correlation coefficient between
              estimated and finite-difference velocity magnitudes.
            - "velocity_direction_error_deg": Mean angular error between
              estimated and finite-difference velocity directions (degrees).
            - "num_tracks_evaluated": Number of tracks used.
            - "num_velocity_samples": Total number of velocity comparisons made.
    """
    # Collect tracks: track_id -> list of (timestamp, box, velocity)
    tracks: Dict[int, List[Tuple[float, NDArray, Optional[NDArray]]]] = {}

    for frame in frames:
        timestamp = frame["timestamp"]
        pred = frame.get("predictions", {})
        boxes = np.asarray(pred.get("boxes", np.empty((0, 7))))
        track_ids = np.asarray(pred.get("track_ids", np.array([])))

        # Try to find velocity data
        velocities = None
        for key in velocity_keys:
            if key in pred:
                velocities = np.asarray(pred[key])
                break

        if len(boxes) == 0 or len(track_ids) == 0:
            continue

        for i, tid in enumerate(track_ids):
            tid_int = int(tid)
            if tid_int not in tracks:
                tracks[tid_int] = []

            vel = velocities[i] if velocities is not None and i < len(velocities) else None
            tracks[tid_int].append((timestamp, boxes[i], vel))

    # Filter tracks
    valid_tracks = {
        tid: sorted(data, key=lambda x: x[0])
        for tid, data in tracks.items()
        if len(data) >= min_track_length
    }

    if not valid_tracks:
        return {
            "velocity_mae": 0.0,
            "velocity_rmse": 0.0,
            "velocity_correlation": 0.0,
            "velocity_direction_error_deg": 0.0,
            "num_tracks_evaluated": 0,
            "num_velocity_samples": 0,
        }

    estimated_speeds: List[float] = []
    fd_speeds: List[float] = []
    velocity_errors: List[float] = []
    direction_errors: List[float] = []

    for tid, track_data in valid_tracks.items():
        for i in range(len(track_data) - 1):
            t0, box0, vel0 = track_data[i]
            t1, box1, vel1 = track_data[i + 1]

            dt = t1 - t0
            if dt < 1e-6:
                continue

            # Finite-difference velocity from positions
            pos0 = box0[:3]
            pos1 = box1[:3]
            fd_velocity = (pos1 - pos0) / dt  # (3,) or derived

            # Get estimated velocity (use midpoint or frame 0's estimate)
            est_vel = vel0
            if est_vel is None:
                # If no velocity at t0, try t1
                est_vel = vel1
            if est_vel is None:
                continue

            # Ensure both are same dimensionality
            est_vel = np.asarray(est_vel, dtype=np.float64)
            if len(est_vel) == 2:
                # 2D velocity: compare in xy plane
                fd_vel_2d = fd_velocity[:2]
                error = np.linalg.norm(est_vel - fd_vel_2d)
                est_speed = np.linalg.norm(est_vel)
                fd_speed = np.linalg.norm(fd_vel_2d)

                # Direction error
                if est_speed > 0.1 and fd_speed > 0.1:
                    cos_angle = np.clip(
                        np.dot(est_vel, fd_vel_2d) / (est_speed * fd_speed),
                        -1.0,
                        1.0,
                    )
                    angle_error = np.arccos(cos_angle)
                    direction_errors.append(float(np.degrees(angle_error)))
            elif len(est_vel) == 3:
                # 3D velocity
                error = np.linalg.norm(est_vel - fd_velocity)
                est_speed = np.linalg.norm(est_vel)
                fd_speed = np.linalg.norm(fd_velocity)

                if est_speed > 0.1 and fd_speed > 0.1:
                    cos_angle = np.clip(
                        np.dot(est_vel, fd_velocity) / (est_speed * fd_speed),
                        -1.0,
                        1.0,
                    )
                    angle_error = np.arccos(cos_angle)
                    direction_errors.append(float(np.degrees(angle_error)))
            else:
                continue

            velocity_errors.append(float(error))
            estimated_speeds.append(float(est_speed))
            fd_speeds.append(float(fd_speed))

    if not velocity_errors:
        return {
            "velocity_mae": 0.0,
            "velocity_rmse": 0.0,
            "velocity_correlation": 0.0,
            "velocity_direction_error_deg": 0.0,
            "num_tracks_evaluated": len(valid_tracks),
            "num_velocity_samples": 0,
        }

    errors_arr = np.array(velocity_errors)
    mae = float(np.mean(errors_arr))
    rmse = float(np.sqrt(np.mean(errors_arr**2)))

    # Pearson correlation between estimated and FD speed magnitudes
    est_arr = np.array(estimated_speeds)
    fd_arr = np.array(fd_speeds)

    if len(est_arr) > 1 and np.std(est_arr) > 1e-8 and np.std(fd_arr) > 1e-8:
        correlation = float(np.corrcoef(est_arr, fd_arr)[0, 1])
    else:
        correlation = 0.0

    mean_direction_error = (
        float(np.mean(direction_errors)) if direction_errors else 0.0
    )

    return {
        "velocity_mae": mae,
        "velocity_rmse": rmse,
        "velocity_correlation": correlation,
        "velocity_direction_error_deg": mean_direction_error,
        "num_tracks_evaluated": len(valid_tracks),
        "num_velocity_samples": len(velocity_errors),
    }


# =============================================================================
# Batch Processing Interface
# =============================================================================


def compute_all_temporal_metrics(
    frames: List[Dict[str, Any]],
    ego_transforms: Optional[List[TransformMatrix]] = None,
    latency_ms: float = 100.0,
    map_distance_metric: str = "chamfer",
    map_match_threshold: float = 5.0,
    min_track_length: int = 3,
    distance_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Compute all temporal consistency metrics for a single sequence.

    Convenience function that runs all four temporal metrics and returns
    combined results.

    Args:
        frames: List of frame dictionaries containing predictions, ground truth,
            timestamps, track IDs, velocities, and map elements as needed by
            individual metrics.
        ego_transforms: List of 4x4 transformation matrices (frame t -> t+1).
            Required for map consistency; skipped if None.
        latency_ms: Processing latency for streaming AP (milliseconds).
        map_distance_metric: Distance metric for map consistency ("chamfer"/"iou").
        map_match_threshold: Matching threshold for map consistency.
        min_track_length: Minimum track length for smoothness/velocity metrics.
        distance_threshold: Center distance threshold for streaming AP matching.

    Returns:
        Combined dictionary with all metric results under namespaced keys:
            - "map_consistency/...": Map consistency results.
            - "streaming_ap/...": Streaming AP results.
            - "temporal_smoothness/...": Smoothness results.
            - "velocity_consistency/...": Velocity consistency results.
    """
    results: Dict[str, Any] = {}

    # Map consistency (requires ego transforms and map elements)
    has_map_elements = any(
        "map_elements" in f and len(f.get("map_elements", [])) > 0
        for f in frames
    )
    if ego_transforms is not None and has_map_elements:
        map_results = compute_map_consistency(
            frames=frames,
            ego_transforms=ego_transforms,
            distance_metric=map_distance_metric,
            match_threshold=map_match_threshold,
        )
        for k, v in map_results.items():
            results[f"map_consistency/{k}"] = v

    # Streaming AP (requires predictions and ground truth)
    has_predictions = any(
        "predictions" in f
        and len(f["predictions"].get("boxes", [])) > 0
        for f in frames
    )
    has_gt = any(
        "ground_truth" in f
        and len(f["ground_truth"].get("boxes", [])) > 0
        for f in frames
    )
    if has_predictions and has_gt:
        streaming_results = compute_streaming_ap(
            frames=frames,
            latency_ms=latency_ms,
            distance_threshold=distance_threshold,
        )
        for k, v in streaming_results.items():
            results[f"streaming_ap/{k}"] = v

    # Temporal smoothness (requires track IDs)
    has_tracks = any(
        "predictions" in f
        and len(f["predictions"].get("track_ids", [])) > 0
        for f in frames
    )
    if has_tracks:
        smoothness_results = compute_temporal_smoothness(
            frames=frames,
            min_track_length=min_track_length,
        )
        for k, v in smoothness_results.items():
            results[f"temporal_smoothness/{k}"] = v

    # Velocity consistency (requires velocities and track IDs)
    has_velocities = any(
        "predictions" in f and "velocities" in f.get("predictions", {})
        for f in frames
    )
    if has_tracks and has_velocities:
        velocity_results = compute_velocity_consistency(
            frames=frames,
            min_track_length=max(2, min_track_length),
        )
        for k, v in velocity_results.items():
            results[f"velocity_consistency/{k}"] = v

    return results


def compute_all_temporal_metrics_batched(
    sequences: List[List[Dict[str, Any]]],
    ego_transforms_batch: Optional[List[List[TransformMatrix]]] = None,
    latency_ms: float = 100.0,
    map_distance_metric: str = "chamfer",
    map_match_threshold: float = 5.0,
    min_track_length: int = 3,
    distance_threshold: float = 2.0,
    aggregate: bool = True,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute temporal metrics over a batch of sequences.

    Args:
        sequences: List of sequences, where each sequence is a list of frame
            dictionaries.
        ego_transforms_batch: List of ego transform lists, one per sequence.
            If None, map consistency is skipped for all sequences.
        latency_ms: Processing latency for streaming AP.
        map_distance_metric: Distance metric for map consistency.
        map_match_threshold: Matching threshold for map consistency.
        min_track_length: Minimum track length for temporal metrics.
        distance_threshold: Center distance threshold for streaming AP.
        aggregate: If True, return mean metrics across all sequences.
            If False, return list of per-sequence results.

    Returns:
        If aggregate=True: Dictionary with mean values for each metric.
        If aggregate=False: List of per-sequence result dictionaries.
    """
    all_results: List[Dict[str, Any]] = []

    for seq_idx, frames in enumerate(sequences):
        ego_transforms = None
        if ego_transforms_batch is not None and seq_idx < len(ego_transforms_batch):
            ego_transforms = ego_transforms_batch[seq_idx]

        seq_results = compute_all_temporal_metrics(
            frames=frames,
            ego_transforms=ego_transforms,
            latency_ms=latency_ms,
            map_distance_metric=map_distance_metric,
            map_match_threshold=map_match_threshold,
            min_track_length=min_track_length,
            distance_threshold=distance_threshold,
        )
        all_results.append(seq_results)

    if not aggregate:
        return all_results

    # Aggregate: compute mean for scalar metrics across sequences
    if not all_results:
        return {}

    aggregated: Dict[str, Any] = {}
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())

    for key in sorted(all_keys):
        values = []
        for r in all_results:
            if key in r:
                val = r[key]
                if isinstance(val, (int, float)):
                    values.append(val)

        if values:
            aggregated[key] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

    aggregated["num_sequences"] = len(sequences)
    return aggregated
