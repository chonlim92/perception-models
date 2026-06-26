"""
Temporal event detection for autonomous driving scenario annotation.

Detects events in temporal sequences of tracked objects and ego vehicle states.
Supports detection of cut-in maneuvers, hard braking, lane changes, and
near-miss situations using sliding window analysis over time-series data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .object_detector import EgoState


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Track:
    """A tracked object over multiple timesteps.

    Attributes:
        track_id: Unique identifier for this track.
        timestamps: List of timestamps (seconds) for each observation.
        positions: Tx3 array of [x, y, z] positions in ego frame over time.
        velocities: Tx3 array of [vx, vy, vz] velocities in ego frame over time.
        class_name: Semantic class of the tracked object (e.g., "car", "truck").
    """

    track_id: int
    timestamps: List[float]
    positions: np.ndarray  # shape (T, 3)
    velocities: np.ndarray  # shape (T, 3)
    class_name: str


@dataclass
class Event:
    """A detected temporal event.

    Attributes:
        event_type: Type of event (e.g., "cut_in", "hard_braking",
                    "lane_change", "near_miss").
        timestamp: When the event started (seconds).
        duration: Duration of the event (seconds).
        confidence: Confidence score in [0, 1].
        metadata: Additional information about the event (e.g., track_id,
                  severity, min_ttc).
    """

    event_type: str
    timestamp: float
    duration: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TemporalEventDetector
# ---------------------------------------------------------------------------


class TemporalEventDetector:
    """Detects events in temporal sequences of tracks and ego states.

    Performs sliding-window analysis on time-series data to identify
    driving events relevant for scenario annotation. All detection methods
    return lists of Event instances with timestamps, durations, and
    confidence scores.

    Args:
        cut_in_lateral_velocity_threshold: Minimum lateral velocity (m/s)
            toward ego lane to consider a cut-in. Default 0.5.
        cut_in_proximity_threshold: Maximum longitudinal distance (m) for
            cut-in consideration. Default 20.0.
        hard_braking_threshold: Longitudinal acceleration threshold (m/s^2,
            negative) below which hard braking is detected. Default -4.0.
        lane_change_threshold: Lateral displacement threshold (m) to detect
            a lane change. Default 1.85 (approximately half lane width).
        lane_change_time_window: Time window (seconds) within which the
            lateral displacement must occur. Default 3.0.
        ttc_threshold: Time-to-collision threshold (seconds) below which
            a near-miss is flagged. Default 2.0.
    """

    def __init__(
        self,
        cut_in_lateral_velocity_threshold: float = 0.5,
        cut_in_proximity_threshold: float = 20.0,
        hard_braking_threshold: float = -4.0,
        lane_change_threshold: float = 1.85,
        lane_change_time_window: float = 3.0,
        ttc_threshold: float = 2.0,
    ) -> None:
        self.cut_in_lateral_velocity_threshold = cut_in_lateral_velocity_threshold
        self.cut_in_proximity_threshold = cut_in_proximity_threshold
        self.hard_braking_threshold = hard_braking_threshold
        self.lane_change_threshold = lane_change_threshold
        self.lane_change_time_window = lane_change_time_window
        self.ttc_threshold = ttc_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_cut_in(
        self, tracks: List[Track], ego_state: EgoState
    ) -> List[Event]:
        """Detect cut-in events from tracked objects relative to ego.

        A cut-in is detected when a tracked object:
          1. Has lateral position moving toward ego lane center (y=0).
          2. Has lateral velocity directed toward the ego lane exceeding
             the configured threshold.
          3. Is within longitudinal proximity threshold of the ego vehicle.

        Confidence is computed based on aggressiveness (higher lateral
        velocity yields higher confidence).

        Args:
            tracks: List of tracked objects with temporal history.
            ego_state: Current ego vehicle state.

        Returns:
            List of detected cut-in Event instances.
        """
        events: List[Event] = []

        for track in tracks:
            if len(track.timestamps) < 2:
                continue

            positions = track.positions  # (T, 3)
            velocities = track.velocities  # (T, 3)
            timestamps = track.timestamps

            # Analyze each timestep for cut-in behavior
            event_start: float | None = None
            event_start_idx: int | None = None
            max_lat_vel: float = 0.0

            for i in range(len(timestamps)):
                x_pos = float(positions[i, 0])  # longitudinal
                y_pos = float(positions[i, 1])  # lateral
                vy = float(velocities[i, 1])  # lateral velocity

                # Check longitudinal proximity
                if x_pos <= 0 or x_pos > self.cut_in_proximity_threshold:
                    # End any ongoing event
                    if event_start is not None:
                        duration = timestamps[i - 1] - event_start
                        confidence = min(
                            1.0,
                            max_lat_vel
                            / (self.cut_in_lateral_velocity_threshold * 3.0),
                        )
                        confidence = max(0.3, confidence)
                        events.append(
                            Event(
                                event_type="cut_in",
                                timestamp=event_start,
                                duration=max(duration, 0.0),
                                confidence=confidence,
                                metadata={
                                    "track_id": track.track_id,
                                    "class_name": track.class_name,
                                    "max_lateral_velocity": max_lat_vel,
                                },
                            )
                        )
                        event_start = None
                        event_start_idx = None
                        max_lat_vel = 0.0
                    continue

                # Determine if lateral velocity is toward ego lane (y=0)
                # If object is to the left (y > 0), moving right (vy < 0) is toward ego
                # If object is to the right (y < 0), moving left (vy > 0) is toward ego
                moving_toward_ego = (y_pos > 0 and vy < 0) or (
                    y_pos < 0 and vy > 0
                )

                # Check lateral velocity magnitude
                lat_vel_magnitude = abs(vy)
                exceeds_threshold = (
                    lat_vel_magnitude >= self.cut_in_lateral_velocity_threshold
                )

                # Check that |y| is decreasing (object approaching ego lane)
                y_decreasing = False
                if i > 0:
                    prev_y = abs(float(positions[i - 1, 1]))
                    curr_y = abs(y_pos)
                    y_decreasing = curr_y < prev_y

                if moving_toward_ego and exceeds_threshold and y_decreasing:
                    if event_start is None:
                        event_start = timestamps[i]
                        event_start_idx = i
                    max_lat_vel = max(max_lat_vel, lat_vel_magnitude)
                else:
                    # End ongoing event if conditions no longer met
                    if event_start is not None:
                        duration = timestamps[i - 1] - event_start
                        confidence = min(
                            1.0,
                            max_lat_vel
                            / (self.cut_in_lateral_velocity_threshold * 3.0),
                        )
                        confidence = max(0.3, confidence)
                        events.append(
                            Event(
                                event_type="cut_in",
                                timestamp=event_start,
                                duration=max(duration, 0.0),
                                confidence=confidence,
                                metadata={
                                    "track_id": track.track_id,
                                    "class_name": track.class_name,
                                    "max_lateral_velocity": max_lat_vel,
                                },
                            )
                        )
                        event_start = None
                        event_start_idx = None
                        max_lat_vel = 0.0

            # Close any open event at end of track
            if event_start is not None:
                duration = timestamps[-1] - event_start
                confidence = min(
                    1.0,
                    max_lat_vel / (self.cut_in_lateral_velocity_threshold * 3.0),
                )
                confidence = max(0.3, confidence)
                events.append(
                    Event(
                        event_type="cut_in",
                        timestamp=event_start,
                        duration=max(duration, 0.0),
                        confidence=confidence,
                        metadata={
                            "track_id": track.track_id,
                            "class_name": track.class_name,
                            "max_lateral_velocity": max_lat_vel,
                        },
                    )
                )

        return events

    def detect_hard_braking(self, ego_states: List[EgoState]) -> List[Event]:
        """Detect hard braking events from ego vehicle state history.

        Hard braking is detected when the longitudinal acceleration
        (acceleration[0]) drops below the configured threshold. The event
        duration covers the contiguous interval where deceleration exceeds
        the threshold.

        Confidence is computed as min(1.0, |decel| / (2 * |threshold|)).

        Args:
            ego_states: Ordered list of ego vehicle states over time.

        Returns:
            List of detected hard braking Event instances.
        """
        events: List[Event] = []

        if len(ego_states) < 2:
            return events

        event_start: float | None = None
        max_decel: float = 0.0

        for i, state in enumerate(ego_states):
            ax = float(state.acceleration[0])  # longitudinal acceleration

            if ax < self.hard_braking_threshold:
                if event_start is None:
                    event_start = state.timestamp
                max_decel = min(max_decel, ax)
            else:
                # End event if we were in a braking phase
                if event_start is not None:
                    # Duration from start to last braking state
                    event_end = ego_states[i - 1].timestamp
                    duration = event_end - event_start
                    # Confidence based on severity relative to threshold
                    confidence = min(
                        1.0,
                        abs(max_decel) / (2.0 * abs(self.hard_braking_threshold)),
                    )
                    events.append(
                        Event(
                            event_type="hard_braking",
                            timestamp=event_start,
                            duration=max(duration, 0.0),
                            confidence=confidence,
                            metadata={
                                "max_deceleration": max_decel,
                                "severity": (
                                    "extreme"
                                    if max_decel < 2.0 * self.hard_braking_threshold
                                    else "severe"
                                ),
                            },
                        )
                    )
                    event_start = None
                    max_decel = 0.0

        # Close any open event at end of sequence
        if event_start is not None:
            event_end = ego_states[-1].timestamp
            duration = event_end - event_start
            confidence = min(
                1.0,
                abs(max_decel) / (2.0 * abs(self.hard_braking_threshold)),
            )
            events.append(
                Event(
                    event_type="hard_braking",
                    timestamp=event_start,
                    duration=max(duration, 0.0),
                    confidence=confidence,
                    metadata={
                        "max_deceleration": max_decel,
                        "severity": (
                            "extreme"
                            if max_decel < 2.0 * self.hard_braking_threshold
                            else "severe"
                        ),
                    },
                )
            )

        return events

    def detect_lane_change(self, ego_states: List[EgoState]) -> List[Event]:
        """Detect lane change events from ego vehicle state history.

        A lane change is detected when the ego lateral displacement exceeds
        the configured threshold within the configured time window. Uses a
        sliding window approach over the position history.

        Confidence is based on smoothness of the lane change: a smoother
        lateral trajectory (lower variance in lateral velocity) yields
        higher confidence.

        Args:
            ego_states: Ordered list of ego vehicle states over time.

        Returns:
            List of detected lane change Event instances.
        """
        events: List[Event] = []

        if len(ego_states) < 3:
            return events

        timestamps = np.array([s.timestamp for s in ego_states])
        # Extract lateral positions (y component of ego position)
        lateral_positions = np.array([float(s.position[1]) for s in ego_states])

        # Sliding window approach
        n = len(ego_states)
        i = 0
        detected_intervals: List[tuple] = []  # (start_idx, end_idx)

        while i < n:
            # Find the window end index based on time window
            j = i
            while (
                j < n
                and (timestamps[j] - timestamps[i]) <= self.lane_change_time_window
            ):
                j += 1
            j = min(j, n - 1)

            # Compute lateral displacement within this window
            if j > i:
                displacement = abs(lateral_positions[j] - lateral_positions[i])

                if displacement >= self.lane_change_threshold:
                    # Check that this interval doesn't overlap with already detected ones
                    overlaps = False
                    for start_idx, end_idx in detected_intervals:
                        if i < end_idx and j > start_idx:
                            overlaps = True
                            break

                    if not overlaps:
                        # Compute smoothness: lower variance in lateral velocity
                        # relative to mean lateral velocity => smoother
                        window_lat_positions = lateral_positions[i : j + 1]
                        window_timestamps = timestamps[i : j + 1]
                        dt_arr = np.diff(window_timestamps)

                        if len(dt_arr) > 0 and np.all(dt_arr > 0):
                            lat_velocities = (
                                np.diff(window_lat_positions) / dt_arr
                            )
                            mean_lat_vel = np.mean(np.abs(lat_velocities))
                            std_lat_vel = np.std(lat_velocities)

                            # Smoothness metric: 1 - normalized std
                            if mean_lat_vel > 1e-6:
                                smoothness = 1.0 - min(
                                    1.0, std_lat_vel / (mean_lat_vel + 1e-6)
                                )
                            else:
                                smoothness = 0.5

                            confidence = float(
                                np.clip(0.5 + 0.5 * smoothness, 0.4, 1.0)
                            )
                        else:
                            confidence = 0.5

                        duration = float(timestamps[j] - timestamps[i])
                        direction = (
                            "left"
                            if lateral_positions[j] > lateral_positions[i]
                            else "right"
                        )

                        events.append(
                            Event(
                                event_type="lane_change",
                                timestamp=float(timestamps[i]),
                                duration=duration,
                                confidence=confidence,
                                metadata={
                                    "lateral_displacement": float(displacement),
                                    "direction": direction,
                                },
                            )
                        )
                        detected_intervals.append((i, j))
                        # Jump past this detected lane change
                        i = j + 1
                        continue

            i += 1

        return events

    def detect_near_miss(
        self, tracks: List[Track], ego_states: List[EgoState]
    ) -> List[Event]:
        """Detect near-miss events between ego and tracked objects.

        A near-miss is detected when the time-to-collision (TTC) between
        the ego vehicle and a tracked object falls below the configured
        threshold while remaining positive (objects are closing).

        TTC is computed as: distance / closing_speed, where closing_speed
        is the relative velocity projected along the vector from ego to
        the tracked object.

        Confidence is computed as 1.0 - (min_ttc / ttc_threshold).

        Args:
            tracks: List of tracked objects with temporal history.
            ego_states: Ordered list of ego vehicle states (must be
                temporally aligned with tracks where timestamps match).

        Returns:
            List of detected near-miss Event instances.
        """
        events: List[Event] = []

        if not ego_states or not tracks:
            return events

        # Build a lookup of ego states by timestamp for efficient matching
        ego_by_time: Dict[float, EgoState] = {
            s.timestamp: s for s in ego_states
        }
        ego_timestamps = sorted(ego_by_time.keys())

        for track in tracks:
            if len(track.timestamps) < 2:
                continue

            event_start: float | None = None
            min_ttc: float = float("inf")
            min_ttc_timestamp: float = 0.0

            for i, t in enumerate(track.timestamps):
                # Find the closest ego state by timestamp
                ego_state = self._find_closest_ego_state(
                    t, ego_timestamps, ego_by_time
                )
                if ego_state is None:
                    continue

                # Object position relative to ego (already in ego frame)
                obj_pos = track.positions[i]  # [x, y, z]
                obj_vel = track.velocities[i]  # [vx, vy, vz]

                # Distance from ego (at origin in ego frame) to object
                distance = float(np.linalg.norm(obj_pos))
                if distance < 1e-3:
                    # Essentially at the same point, skip to avoid div by zero
                    continue

                # Unit vector from ego to object
                direction = obj_pos / distance

                # Ego velocity in ego frame
                ego_vel = ego_state.velocity

                # Relative velocity: object velocity minus ego velocity
                # (both in ego frame)
                rel_vel = obj_vel - ego_vel

                # Closing speed: negative component of relative velocity
                # along the direction from ego to object
                # (negative means objects are approaching each other)
                closing_speed = -float(np.dot(rel_vel, direction))

                if closing_speed <= 0:
                    # Objects are not closing; end any ongoing event
                    if event_start is not None:
                        self._finalize_near_miss_event(
                            events,
                            event_start,
                            track.timestamps[i - 1],
                            min_ttc,
                            min_ttc_timestamp,
                            track,
                        )
                        event_start = None
                        min_ttc = float("inf")
                    continue

                # Compute TTC
                ttc = distance / closing_speed

                if 0 < ttc < self.ttc_threshold:
                    if event_start is None:
                        event_start = t
                    if ttc < min_ttc:
                        min_ttc = ttc
                        min_ttc_timestamp = t
                else:
                    # TTC above threshold; end any ongoing event
                    if event_start is not None:
                        self._finalize_near_miss_event(
                            events,
                            event_start,
                            track.timestamps[i - 1],
                            min_ttc,
                            min_ttc_timestamp,
                            track,
                        )
                        event_start = None
                        min_ttc = float("inf")

            # Close any open event at end of track
            if event_start is not None:
                self._finalize_near_miss_event(
                    events,
                    event_start,
                    track.timestamps[-1],
                    min_ttc,
                    min_ttc_timestamp,
                    track,
                )

        return events

    def detect_all_events(
        self, tracks: List[Track], ego_states: List[EgoState]
    ) -> List[Event]:
        """Run all event detectors and return merged, time-sorted results.

        Executes cut-in, hard braking, lane change, and near-miss detection,
        then merges all events into a single list sorted by timestamp.

        Args:
            tracks: List of tracked objects with temporal history.
            ego_states: Ordered list of ego vehicle states over time.

        Returns:
            Time-sorted list of all detected Event instances.
        """
        all_events: List[Event] = []

        # Run each detector
        # For cut-in, use the last ego state as reference (current state)
        if ego_states:
            current_ego = ego_states[-1]
            all_events.extend(self.detect_cut_in(tracks, current_ego))

        all_events.extend(self.detect_hard_braking(ego_states))
        all_events.extend(self.detect_lane_change(ego_states))
        all_events.extend(self.detect_near_miss(tracks, ego_states))

        # Sort by timestamp
        all_events.sort(key=lambda e: e.timestamp)

        return all_events

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_closest_ego_state(
        self,
        target_time: float,
        sorted_timestamps: List[float],
        ego_by_time: Dict[float, EgoState],
    ) -> EgoState | None:
        """Find the ego state closest in time to the target timestamp.

        Uses binary search for efficient lookup. Returns None if no ego
        state is available within 0.5 seconds of the target time.

        Args:
            target_time: Target timestamp to find closest ego state for.
            sorted_timestamps: Sorted list of available ego timestamps.
            ego_by_time: Mapping from timestamp to EgoState.

        Returns:
            Closest EgoState or None if too far in time.
        """
        if not sorted_timestamps:
            return None

        # Binary search for closest timestamp
        idx = np.searchsorted(sorted_timestamps, target_time)

        candidates: List[float] = []
        if idx < len(sorted_timestamps):
            candidates.append(sorted_timestamps[idx])
        if idx > 0:
            candidates.append(sorted_timestamps[idx - 1])

        if not candidates:
            return None

        closest = min(candidates, key=lambda t: abs(t - target_time))

        # Only return if within 0.5 second tolerance
        if abs(closest - target_time) > 0.5:
            return None

        return ego_by_time[closest]

    def _finalize_near_miss_event(
        self,
        events: List[Event],
        event_start: float,
        event_end: float,
        min_ttc: float,
        min_ttc_timestamp: float,
        track: Track,
    ) -> None:
        """Create and append a near-miss event to the events list.

        Args:
            events: List to append the event to.
            event_start: Start timestamp of the event.
            event_end: End timestamp of the event.
            min_ttc: Minimum TTC observed during the event.
            min_ttc_timestamp: Timestamp at which minimum TTC occurred.
            track: The track involved in the near-miss.
        """
        duration = max(event_end - event_start, 0.0)
        confidence = float(np.clip(1.0 - (min_ttc / self.ttc_threshold), 0.1, 1.0))

        events.append(
            Event(
                event_type="near_miss",
                timestamp=event_start,
                duration=duration,
                confidence=confidence,
                metadata={
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "min_ttc": min_ttc,
                    "min_ttc_timestamp": min_ttc_timestamp,
                    "severity": (
                        "critical" if min_ttc < 0.5 else
                        "high" if min_ttc < 1.0 else
                        "moderate"
                    ),
                },
            )
        )
