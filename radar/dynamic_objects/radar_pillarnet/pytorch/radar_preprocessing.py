"""
Multi-sweep radar accumulation with ego-motion compensation.

Radar point clouds are significantly sparser than LiDAR (~100-500 points per sweep
vs ~100k for LiDAR). To compensate, multiple sweeps are accumulated and transformed
to the current ego-vehicle frame using SE(3) ego-motion matrices.

Each radar point has features: [x, y, z, RCS, vr_compensated, vr_raw]
where:
    - x, y, z: 3D position in sensor frame
    - RCS: Radar Cross Section (dBsm), indicates target reflectivity
    - vr_compensated: Radial velocity with ego-motion removed
    - vr_raw: Raw measured radial velocity

After accumulation, a time delta feature (dt) is appended indicating how old
each point is relative to the current timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RadarSweep:
    """Container for a single radar sweep.

    Attributes:
        points: (N, 6) array [x, y, z, RCS, vr_compensated, vr_raw]
        timestamp: Sweep timestamp in seconds (float64)
        ego_pose: (4, 4) SE(3) transformation from this sweep's frame to global frame
    """

    points: np.ndarray  # (N, 6)
    timestamp: float
    ego_pose: np.ndarray  # (4, 4)


@dataclass
class FilterConfig:
    """Configuration for radar clutter filtering.

    Attributes:
        min_rcs: Minimum RCS threshold in dBsm (filter weak returns)
        max_rcs: Maximum RCS threshold in dBsm (filter unrealistic returns)
        min_velocity: Minimum absolute radial velocity for dynamic classification
        max_velocity: Maximum absolute radial velocity (filter noise)
        velocity_consistency_threshold: Maximum allowed difference between
            compensated and expected velocity for consistency check
        dynamic_property_flags: Set of integer flags indicating valid dynamic states.
            Common convention: 0=moving, 1=stationary, 2=oncoming, 3=crossing_left,
            4=crossing_right, 5=unknown, 6=stopped, 7=invalid
        invalid_flags: Set of flags to reject (e.g., {7} for invalid)
        min_distance: Minimum distance from sensor to keep (filter near-field clutter)
        max_distance: Maximum detection range
    """

    min_rcs: float = -5.0
    max_rcs: float = 60.0
    min_velocity: float = 0.0
    max_velocity: float = 100.0
    velocity_consistency_threshold: float = 5.0
    dynamic_property_flags: Optional[set] = None
    invalid_flags: set = field(default_factory=lambda: {7})
    min_distance: float = 1.0
    max_distance: float = 100.0


def compensate_ego_motion(
    points: np.ndarray, transform_matrix: np.ndarray
) -> np.ndarray:
    """Apply SE(3) transformation to point coordinates and velocities.

    Transforms points from a past frame to the current ego-vehicle frame.
    Both positions and velocity vectors are transformed (velocities use
    only the rotation component since they are direction vectors).

    Args:
        points: (N, 6) array with columns [x, y, z, RCS, vr_compensated, vr_raw].
            Positions (x, y, z) are transformed with full SE(3).
            Velocities remain scalar (radial) so only position is rotated.
        transform_matrix: (4, 4) SE(3) matrix transforming from source frame
            to target frame. Must be a valid rigid-body transform (det(R)=1).

    Returns:
        (N, 6) array with transformed positions. RCS and velocities preserved
        as-is since radial velocity is recomputed during detection.
    """
    if points.shape[0] == 0:
        return points.copy()

    n_points = points.shape[0]

    # Extract rotation and translation
    rotation = transform_matrix[:3, :3]  # (3, 3)
    translation = transform_matrix[:3, 3]  # (3,)

    # Transform positions: p_target = R @ p_source + t
    positions = points[:, :3]  # (N, 3)
    transformed_positions = (rotation @ positions.T).T + translation  # (N, 3)

    # Construct output: keep RCS and velocities unchanged
    # Radial velocity is sensor-relative and will be recomputed if needed
    result = points.copy()
    result[:, :3] = transformed_positions

    return result


def accumulate_sweeps(
    current_sweep: RadarSweep,
    history_sweeps: List[RadarSweep],
    ego_poses: Optional[List[np.ndarray]] = None,
    num_sweeps: int = 6,
) -> np.ndarray:
    """Full multi-sweep accumulation pipeline.

    Accumulates the current sweep and up to (num_sweeps - 1) historical sweeps,
    transforming all past points into the current ego-vehicle frame. Appends
    a time delta feature to each point.

    Args:
        current_sweep: The most recent radar sweep.
        history_sweeps: List of past sweeps, ordered from most recent to oldest.
        ego_poses: Optional list of 4x4 ego-pose matrices for each history sweep.
            If None, uses ego_pose from each RadarSweep object.
        num_sweeps: Total number of sweeps to accumulate (including current).
            Default is 6 (current + 5 historical).

    Returns:
        (M, 7) array with columns [x, y, z, RCS, vr_compensated, vr_raw, dt]
        where dt is the time elapsed since the point was measured (0 for current).
        All points are in the current ego-vehicle frame.
    """
    # Current sweep gets dt = 0
    current_points = current_sweep.points.copy()  # (N_curr, 6)
    n_current = current_points.shape[0]

    # Add time delta column (0 for current)
    current_with_dt = np.column_stack(
        [current_points, np.zeros(n_current, dtype=np.float32)]
    )  # (N_curr, 7)

    accumulated = [current_with_dt]

    # Compute inverse of current ego pose (to transform from global to current frame)
    current_ego_inv = np.linalg.inv(current_sweep.ego_pose)  # (4, 4)

    # Accumulate historical sweeps
    n_history = min(num_sweeps - 1, len(history_sweeps))
    for i in range(n_history):
        hist_sweep = history_sweeps[i]

        if hist_sweep.points.shape[0] == 0:
            continue

        # Get ego pose for this historical sweep
        if ego_poses is not None and i < len(ego_poses):
            hist_ego_pose = ego_poses[i]
        else:
            hist_ego_pose = hist_sweep.ego_pose

        # Transform: past_global = hist_ego_pose @ past_local
        # Then: past_in_current = current_ego_inv @ past_global
        # Combined: transform = current_ego_inv @ hist_ego_pose
        transform = current_ego_inv @ hist_ego_pose  # (4, 4)

        # Apply ego-motion compensation
        transformed_points = compensate_ego_motion(hist_sweep.points, transform)

        # Compute time delta
        dt = current_sweep.timestamp - hist_sweep.timestamp  # positive value
        n_hist = transformed_points.shape[0]

        # Append time delta
        hist_with_dt = np.column_stack(
            [transformed_points, np.full(n_hist, dt, dtype=np.float32)]
        )  # (N_hist, 7)

        accumulated.append(hist_with_dt)

    # Concatenate all sweeps
    if len(accumulated) == 1:
        return accumulated[0]

    return np.concatenate(accumulated, axis=0)  # (M, 7)


class RadarClutterFilter:
    """Filters radar points based on dynamic properties, RCS, and velocity.

    Radar sensors produce significant clutter from:
    - Ground reflections (low RCS, near-field)
    - Multi-path propagation (inconsistent velocity)
    - Static infrastructure tagged as invalid
    - Noise (very low RCS or extreme velocities)

    This filter applies cascaded criteria to remove clutter while preserving
    true detections from vehicles, pedestrians, and cyclists.
    """

    def __init__(self, config: Optional[FilterConfig] = None) -> None:
        """Initialize clutter filter with given configuration.

        Args:
            config: FilterConfig dataclass with threshold parameters.
                Uses defaults if None.
        """
        self.config = config if config is not None else FilterConfig()

    def filter_by_rcs(self, points: np.ndarray) -> np.ndarray:
        """Filter points by Radar Cross Section thresholds.

        Args:
            points: (N, D) array where column 3 is RCS in dBsm.

        Returns:
            Boolean mask of shape (N,) where True means the point passes.
        """
        rcs = points[:, 3]
        mask = (rcs >= self.config.min_rcs) & (rcs <= self.config.max_rcs)
        return mask

    def filter_by_velocity(self, points: np.ndarray) -> np.ndarray:
        """Filter points by velocity magnitude thresholds.

        Points with absolute radial velocity outside [min_velocity, max_velocity]
        are rejected. Note: min_velocity=0 keeps all static and dynamic points.

        Args:
            points: (N, D) array where column 4 is compensated radial velocity.

        Returns:
            Boolean mask of shape (N,) where True means the point passes.
        """
        vr = np.abs(points[:, 4])
        mask = (vr >= self.config.min_velocity) & (vr <= self.config.max_velocity)
        return mask

    def filter_by_velocity_consistency(self, points: np.ndarray) -> np.ndarray:
        """Filter points with inconsistent velocity measurements.

        Compares compensated velocity (column 4) with raw velocity (column 5).
        Large discrepancies indicate multi-path or processing errors.

        Args:
            points: (N, D) array where columns 4,5 are vr_compensated and vr_raw.

        Returns:
            Boolean mask of shape (N,) where True means the point passes.
        """
        vr_comp = points[:, 4]
        vr_raw = points[:, 5]
        diff = np.abs(vr_comp - vr_raw)
        mask = diff <= self.config.velocity_consistency_threshold
        return mask

    def filter_by_distance(self, points: np.ndarray) -> np.ndarray:
        """Filter points by distance from sensor.

        Removes near-field clutter and points beyond maximum detection range.

        Args:
            points: (N, D) array where columns 0,1,2 are x,y,z coordinates.

        Returns:
            Boolean mask of shape (N,) where True means the point passes.
        """
        distances = np.sqrt(
            points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2
        )
        mask = (distances >= self.config.min_distance) & (
            distances <= self.config.max_distance
        )
        return mask

    def filter_by_dynamic_property(
        self, points: np.ndarray, dynamic_flags: np.ndarray
    ) -> np.ndarray:
        """Filter points by dynamic property classification flags.

        Args:
            points: (N, D) array of radar points (unused, kept for API consistency).
            dynamic_flags: (N,) integer array of dynamic property flags per point.

        Returns:
            Boolean mask of shape (N,) where True means the point passes.
        """
        mask = np.ones(len(dynamic_flags), dtype=bool)

        # Reject invalid flags
        for flag in self.config.invalid_flags:
            mask &= dynamic_flags != flag

        # If specific valid flags are defined, only keep those
        if self.config.dynamic_property_flags is not None:
            valid_mask = np.zeros(len(dynamic_flags), dtype=bool)
            for flag in self.config.dynamic_property_flags:
                valid_mask |= dynamic_flags == flag
            mask &= valid_mask

        return mask

    def __call__(
        self,
        points: np.ndarray,
        dynamic_flags: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply all filters sequentially and return filtered points.

        Args:
            points: (N, D) array with D >= 6, columns [x, y, z, RCS, vr_comp, vr_raw, ...].
            dynamic_flags: Optional (N,) integer array of dynamic property flags.

        Returns:
            (M, D) array of filtered points where M <= N.
        """
        if points.shape[0] == 0:
            return points.copy()

        # Start with all points valid
        mask = np.ones(points.shape[0], dtype=bool)

        # Apply cascaded filters
        mask &= self.filter_by_rcs(points)
        mask &= self.filter_by_velocity(points)
        mask &= self.filter_by_velocity_consistency(points)
        mask &= self.filter_by_distance(points)

        # Apply dynamic property filter if flags provided
        if dynamic_flags is not None:
            mask &= self.filter_by_dynamic_property(points, dynamic_flags)

        return points[mask]


class RadarMultiSweepAccumulator:
    """Accumulates multiple radar sweeps with ego-motion compensation.

    Maintains a buffer of historical sweeps and produces accumulated point
    clouds aligned to the current ego-vehicle frame. Handles variable sweep
    rates and missing data gracefully.

    Typical usage:
        accumulator = RadarMultiSweepAccumulator(num_sweeps=6, max_age=0.5)
        for sweep in radar_stream:
            accumulated = accumulator.add_sweep(sweep)
            # accumulated is (M, 7) with [x, y, z, RCS, vr, vr_raw, dt]
    """

    def __init__(
        self,
        num_sweeps: int = 6,
        max_age: float = 0.5,
        clutter_filter: Optional[RadarClutterFilter] = None,
    ) -> None:
        """Initialize multi-sweep accumulator.

        Args:
            num_sweeps: Number of sweeps to accumulate (including current).
            max_age: Maximum age in seconds for a sweep to be included.
                Sweeps older than this are discarded from the buffer.
            clutter_filter: Optional filter applied to each sweep before accumulation.
        """
        self.num_sweeps = num_sweeps
        self.max_age = max_age
        self.clutter_filter = clutter_filter
        self._history: List[RadarSweep] = []

    def reset(self) -> None:
        """Clear the sweep history buffer."""
        self._history.clear()

    def _prune_history(self, current_timestamp: float) -> None:
        """Remove sweeps older than max_age from buffer.

        Args:
            current_timestamp: Reference timestamp for age computation.
        """
        self._history = [
            s
            for s in self._history
            if (current_timestamp - s.timestamp) <= self.max_age
        ]

    def add_sweep(
        self,
        sweep: RadarSweep,
        dynamic_flags: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Add a new sweep and return accumulated point cloud.

        Args:
            sweep: New radar sweep to add.
            dynamic_flags: Optional dynamic property flags for clutter filtering.

        Returns:
            (M, 7) accumulated point cloud [x, y, z, RCS, vr, vr_raw, dt]
            in the current ego-vehicle frame.
        """
        # Apply clutter filter if configured
        if self.clutter_filter is not None:
            filtered_points = self.clutter_filter(sweep.points, dynamic_flags)
            sweep = RadarSweep(
                points=filtered_points,
                timestamp=sweep.timestamp,
                ego_pose=sweep.ego_pose,
            )

        # Prune old sweeps
        self._prune_history(sweep.timestamp)

        # Accumulate
        accumulated = accumulate_sweeps(
            current_sweep=sweep,
            history_sweeps=self._history,
            num_sweeps=self.num_sweeps,
        )

        # Add current sweep to history for next iteration
        self._history.insert(0, sweep)

        # Keep buffer bounded
        if len(self._history) > self.num_sweeps:
            self._history = self._history[: self.num_sweeps]

        return accumulated

    def get_accumulated_static(
        self,
        current_sweep: RadarSweep,
        history_sweeps: List[RadarSweep],
        ego_poses: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Static method-like interface for batch processing (no internal state).

        Useful for dataset preprocessing where sweeps are loaded from disk.

        Args:
            current_sweep: The current radar sweep.
            history_sweeps: List of historical sweeps.
            ego_poses: Optional list of ego-pose matrices for history sweeps.

        Returns:
            (M, 7) accumulated point cloud.
        """
        return accumulate_sweeps(
            current_sweep=current_sweep,
            history_sweeps=history_sweeps,
            ego_poses=ego_poses,
            num_sweeps=self.num_sweeps,
        )
