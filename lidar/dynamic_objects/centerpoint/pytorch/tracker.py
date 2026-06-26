"""Greedy center-distance tracking module for CenterPoint 3D object detector.

This module implements a simple yet effective multi-object tracker that associates
detections across frames using greedy nearest-neighbor matching on BEV (bird's eye
view) center distances. It is designed to operate frame-by-frame on the output of
CenterPoint's detection head.

Reference:
    Yin, Zhou, and Krahenbuhl. "Center-based 3D Object Detection and Tracking."
    CVPR 2021.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Track:
    """Represents a single tracked object across frames.

    Attributes:
        track_id: Unique identifier for this track.
        state: Current track state, either 'active' or 'lost'.
        age: Number of frames since the track was first created.
        last_box: Most recent bounding box as (x, y, z, w, h, l, yaw, vx, vy).
        score: Detection confidence score from the most recent matched detection.
        class_id: Predicted class identifier.
        hits: Total number of frames in which this track was matched to a detection.
        time_since_update: Number of consecutive frames without a matched detection.
    """

    track_id: int
    state: str  # 'active' or 'lost'
    age: int
    last_box: np.ndarray  # shape (9,): x, y, z, w, h, l, yaw, vx, vy
    score: float
    class_id: int
    hits: int
    time_since_update: int

    def __post_init__(self) -> None:
        self.last_box = np.asarray(self.last_box, dtype=np.float64)


class CenterPointTracker:
    """Frame-by-frame greedy center-distance tracker for CenterPoint detections.

    The tracker maintains a list of active tracks and performs the following steps
    each frame:
        1. predict() -- extrapolate track positions forward using stored velocities.
        2. update(detections) -- match incoming detections to existing tracks via
           greedy nearest-neighbor on BEV center distance, then manage track
           lifecycle (create, update, lose).
        3. get_active_tracks() -- retrieve tracks satisfying activity criteria.

    Args:
        max_age: Maximum number of consecutive frames a track can go unmatched
            before being removed.
        min_hits: Minimum number of total hits required for a track to be reported
            as active.
        distance_threshold: Maximum BEV center distance (in meters) for a valid
            detection-to-track match.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 1,
        distance_threshold: float = 2.0,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold

        self._tracks: List[Track] = []
        self._next_id: int = 0
        self._dt: float = 1.0  # assume unit time step between frames

    @property
    def tracks(self) -> List[Track]:
        """Return the internal list of all tracks (active and lost)."""
        return self._tracks

    def predict(self) -> None:
        """Extrapolate all track positions forward using stored velocities.

        For each track with state 'active' or 'lost', the center position (x, y) is
        updated as:
            x += vx * dt
            y += vy * dt

        This should be called once per frame before update().
        """
        for track in self._tracks:
            vx = track.last_box[7]
            vy = track.last_box[8]
            track.last_box[0] += vx * self._dt
            track.last_box[1] += vy * self._dt
            track.age += 1
            track.time_since_update += 1

    def update(self, detections: List[np.ndarray], scores: Optional[List[float]] = None,
               class_ids: Optional[List[int]] = None) -> None:
        """Match detections to tracks and manage track lifecycle.

        Performs greedy matching: compute a full distance matrix between predicted
        track centers and detection centers, then iteratively assign the closest
        pair that falls within the distance threshold.

        Args:
            detections: List of detection boxes, each a numpy array of shape (9,)
                with layout (x, y, z, w, h, l, yaw, vx, vy).
            scores: Optional list of detection confidence scores. Defaults to 1.0
                for all detections if not provided.
            class_ids: Optional list of class identifiers for each detection.
                Defaults to 0 for all detections if not provided.
        """
        num_dets = len(detections)
        num_tracks = len(self._tracks)

        if scores is None:
            scores = [1.0] * num_dets
        if class_ids is None:
            class_ids = [0] * num_dets

        # Convert detections to numpy array for vectorized distance computation
        if num_dets == 0:
            # No detections: all tracks become unmatched
            for track in self._tracks:
                if track.time_since_update > self.max_age:
                    track.state = "lost"
            self._remove_lost_tracks()
            return

        det_boxes = np.array([np.asarray(d, dtype=np.float64) for d in detections])
        det_centers = det_boxes[:, :2]  # shape (num_dets, 2)

        if num_tracks == 0:
            # No existing tracks: create a new track for each detection
            for i in range(num_dets):
                self._create_track(det_boxes[i], scores[i], class_ids[i])
            return

        # Compute BEV center distance matrix: shape (num_tracks, num_dets)
        track_centers = np.array([t.last_box[:2] for t in self._tracks])  # (num_tracks, 2)
        distance_matrix = self._compute_distance_matrix(track_centers, det_centers)

        # Greedy matching: iteratively pick the smallest distance below threshold
        matched_tracks, matched_dets = self._greedy_match(distance_matrix)

        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            track = self._tracks[track_idx]
            track.last_box = det_boxes[det_idx].copy()
            track.score = scores[det_idx]
            track.class_id = class_ids[det_idx]
            track.hits += 1
            track.time_since_update = 0
            track.state = "active"

        # Create new tracks for unmatched detections
        unmatched_det_indices = set(range(num_dets)) - set(matched_dets)
        for det_idx in unmatched_det_indices:
            self._create_track(det_boxes[det_idx], scores[det_idx], class_ids[det_idx])

        # Handle unmatched tracks
        unmatched_track_indices = set(range(num_tracks)) - set(matched_tracks)
        for track_idx in unmatched_track_indices:
            track = self._tracks[track_idx]
            if track.time_since_update > self.max_age:
                track.state = "lost"

        self._remove_lost_tracks()

    def get_active_tracks(self) -> List[Track]:
        """Return tracks meeting the activity criteria.

        A track is considered active if:
            - It has accumulated at least `min_hits` matched detections.
            - Its time since last update does not exceed `max_age`.

        Returns:
            List of Track objects that satisfy the activity criteria.
        """
        active: List[Track] = []
        for track in self._tracks:
            if track.hits >= self.min_hits and track.time_since_update <= self.max_age:
                active.append(track)
        return active

    def reset(self) -> None:
        """Clear all tracks and reset the ID counter."""
        self._tracks.clear()
        self._next_id = 0

    def _compute_distance_matrix(
        self, track_centers: np.ndarray, det_centers: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise Euclidean distances between track and detection centers.

        Args:
            track_centers: Array of shape (N, 2) with track BEV positions.
            det_centers: Array of shape (M, 2) with detection BEV positions.

        Returns:
            Distance matrix of shape (N, M).
        """
        # Efficient broadcasting: (N, 1, 2) - (1, M, 2) -> (N, M, 2) -> norm -> (N, M)
        diff = track_centers[:, np.newaxis, :] - det_centers[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return distances

    def _greedy_match(
        self, distance_matrix: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Perform greedy nearest-neighbor matching on the distance matrix.

        Iteratively selects the globally smallest distance entry that is below the
        threshold and whose row (track) and column (detection) have not yet been
        assigned. Continues until no valid assignments remain.

        Args:
            distance_matrix: Array of shape (num_tracks, num_dets) with pairwise
                BEV center distances.

        Returns:
            Tuple of (matched_track_indices, matched_detection_indices), where
            corresponding entries form matched pairs.
        """
        num_tracks, num_dets = distance_matrix.shape
        matched_tracks: List[int] = []
        matched_dets: List[int] = []

        # Flatten and sort distances
        flat_indices = np.argsort(distance_matrix, axis=None)

        used_tracks: set = set()
        used_dets: set = set()

        for flat_idx in flat_indices:
            track_idx = int(flat_idx // num_dets)
            det_idx = int(flat_idx % num_dets)

            if distance_matrix[track_idx, det_idx] > self.distance_threshold:
                # All remaining entries exceed threshold (sorted order)
                break

            if track_idx in used_tracks or det_idx in used_dets:
                continue

            matched_tracks.append(track_idx)
            matched_dets.append(det_idx)
            used_tracks.add(track_idx)
            used_dets.add(det_idx)

            # Early exit if all tracks or all detections are matched
            if len(used_tracks) == num_tracks or len(used_dets) == num_dets:
                break

        return matched_tracks, matched_dets

    def _create_track(self, box: np.ndarray, score: float, class_id: int) -> Track:
        """Instantiate a new track from a detection.

        Args:
            box: Detection bounding box of shape (9,).
            score: Detection confidence score.
            class_id: Predicted class identifier.

        Returns:
            The newly created Track object.
        """
        track = Track(
            track_id=self._next_id,
            state="active",
            age=1,
            last_box=box.copy(),
            score=score,
            class_id=class_id,
            hits=1,
            time_since_update=0,
        )
        self._tracks.append(track)
        self._next_id += 1
        return track

    def _remove_lost_tracks(self) -> None:
        """Remove tracks that have been marked as 'lost'."""
        self._tracks = [t for t in self._tracks if t.state != "lost"]
