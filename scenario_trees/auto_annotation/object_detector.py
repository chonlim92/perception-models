"""
Object detection scenario tagger for autonomous driving perception pipelines.

Processes detection model outputs (bounding boxes, classes, tracks) and generates
scenario tags based on detected object behaviors such as cut-in, lane change,
sudden braking, and jaywalking.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..taxonomy.scenario_schema import ScenarioTag


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DetectedObject:
    """A single detected object from the perception pipeline.

    Attributes:
        class_name: Semantic class (e.g., "car", "pedestrian", "cyclist", "truck").
        bbox: 2-D bounding box as [x1, y1, x2, y2] in image coordinates.
        track_id: Persistent track identifier across frames (None if untracked).
        velocity: Velocity vector [vx, vy, vz] in the ego frame (m/s).
                  vx = longitudinal (positive forward), vy = lateral (positive left),
                  vz = vertical.
        position_3d: 3-D position [x, y, z] in the ego frame (meters).
        confidence: Detection confidence score in [0, 1].
    """

    class_name: str
    bbox: np.ndarray  # shape (4,) -> [x1, y1, x2, y2]
    track_id: Optional[int] = None
    velocity: Optional[np.ndarray] = None  # shape (3,) -> [vx, vy, vz]
    position_3d: Optional[np.ndarray] = None  # shape (3,) -> [x, y, z]
    confidence: float = 1.0


@dataclass
class EgoState:
    """Ego vehicle state at the current timestep.

    Attributes:
        velocity: Ego velocity [vx, vy, vz] in the ego frame (m/s).
        acceleration: Ego acceleration [ax, ay, az] in the ego frame (m/s^2).
        position: Ego global position [x, y, z] (meters).
        heading: Ego heading in radians (0 = forward along x-axis).
        lane_width: Nominal lane width (meters). Default 3.7 m.
        timestamp: Timestamp in seconds for temporal analysis. Default 0.0.
        lane_id: Current lane identifier (integer). Default 0.
    """

    velocity: np.ndarray  # shape (3,)
    acceleration: np.ndarray  # shape (3,)
    position: np.ndarray  # shape (3,)
    heading: float  # radians
    lane_width: float = 3.7
    timestamp: float = 0.0
    lane_id: int = 0


# ---------------------------------------------------------------------------
# Node ID constants (semantic PEGASUS Layer 4 style)
# ---------------------------------------------------------------------------

# Behavior tags
NODE_ID_CUT_IN = "L4.behavior.cut_in"
NODE_ID_LANE_CHANGE = "L4.behavior.lane_change"
NODE_ID_SUDDEN_BRAKING = "L4.behavior.sudden_braking"
NODE_ID_JAYWALKING = "L4.behavior.jaywalking"

# Object presence tags
NODE_ID_PEDESTRIAN_PRESENCE = "L4.pedestrian.adult"
NODE_ID_CYCLIST_PRESENCE = "L4.vehicle.bicycle"
NODE_ID_TRUCK_PRESENCE = "L4.vehicle.truck"
NODE_ID_CAR_PRESENCE = "L4.vehicle.car"
NODE_ID_MOTORCYCLE_PRESENCE = "L4.vehicle.motorcycle"
NODE_ID_BUS_PRESENCE = "L4.vehicle.bus"
NODE_ID_EMERGENCY_PRESENCE = "L4.vehicle.emergency"

# Mapping from class names to presence node IDs
_PRESENCE_NODE_MAP: Dict[str, str] = {
    "pedestrian": NODE_ID_PEDESTRIAN_PRESENCE,
    "cyclist": NODE_ID_CYCLIST_PRESENCE,
    "bicycle": NODE_ID_CYCLIST_PRESENCE,
    "truck": NODE_ID_TRUCK_PRESENCE,
    "car": NODE_ID_CAR_PRESENCE,
    "motorcycle": NODE_ID_MOTORCYCLE_PRESENCE,
    "bus": NODE_ID_BUS_PRESENCE,
    "emergency": NODE_ID_EMERGENCY_PRESENCE,
}


# ---------------------------------------------------------------------------
# Track history entry
# ---------------------------------------------------------------------------


@dataclass
class _TrackHistoryEntry:
    """Internal record for a single frame of a tracked object."""

    position_3d: np.ndarray
    velocity: np.ndarray
    class_name: str
    timestamp: int  # frame index


# ---------------------------------------------------------------------------
# ObjectScenarioTagger
# ---------------------------------------------------------------------------


class ObjectScenarioTagger:
    """Generates scenario tags from object detection outputs.

    The tagger maintains internal track histories to enable temporal behavior
    analysis (e.g., lateral displacement for lane-change detection).

    Args:
        cut_in_lateral_velocity_threshold: Minimum absolute lateral velocity
            (m/s) toward the ego lane to trigger a cut-in tag.
        cut_in_distance_threshold: Maximum longitudinal distance (m) for
            an object to be considered a cut-in threat.
        lane_change_displacement_threshold: Minimum lateral displacement (m)
            across tracked history to flag a lane change. Default is
            approximately half a lane width (1.85 m).
        braking_decel_threshold: Longitudinal deceleration threshold (m/s^2,
            negative value) below which sudden braking is flagged.
        history_length: Number of frames to retain in track history.
    """

    def __init__(
        self,
        cut_in_lateral_velocity_threshold: float = 0.5,
        cut_in_distance_threshold: float = 15.0,
        lane_change_displacement_threshold: float = 1.85,
        braking_decel_threshold: float = -4.0,
        history_length: int = 30,
    ) -> None:
        self.cut_in_lateral_velocity_threshold = cut_in_lateral_velocity_threshold
        self.cut_in_distance_threshold = cut_in_distance_threshold
        self.lane_change_displacement_threshold = lane_change_displacement_threshold
        self.braking_decel_threshold = braking_decel_threshold
        self.history_length = history_length

        # track_id -> list of _TrackHistoryEntry (most recent last)
        self._track_history: Dict[int, List[_TrackHistoryEntry]] = defaultdict(list)
        self._frame_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_object_scenarios(
        self,
        detections: List[DetectedObject],
        ego_state: EgoState,
    ) -> List[ScenarioTag]:
        """Analyze detections and generate scenario tags.

        This is the main entry point. It updates internal track history,
        counts objects by class, and detects the following behaviors:
          - Cut-in
          - Lane change
          - Sudden braking
          - Jaywalking (pedestrian on road outside crosswalk)

        Args:
            detections: List of detected objects for the current frame.
            ego_state: Current ego vehicle state.

        Returns:
            List of ScenarioTag instances describing the detected scenarios.
        """
        self.update_tracks(detections)
        tags: List[ScenarioTag] = []

        # --- Object presence tags ---
        class_counts: Dict[str, int] = defaultdict(int)
        for det in detections:
            class_counts[det.class_name.lower()] += 1

        for class_name, count in class_counts.items():
            if class_name in _PRESENCE_NODE_MAP and count > 0:
                # Confidence based on average detection confidence for this class
                class_detections = [
                    d for d in detections if d.class_name.lower() == class_name
                ]
                avg_conf = float(
                    np.mean([d.confidence for d in class_detections])
                )
                tags.append(
                    ScenarioTag(
                        node_id=_PRESENCE_NODE_MAP[class_name],
                        confidence=min(avg_conf, 1.0),
                        source="auto",
                    )
                )

        # --- Behavior detection ---
        for det in detections:
            behavior_tags = self._detect_behaviors(det, ego_state)
            tags.extend(behavior_tags)

        return tags

    def update_tracks(self, detections: List[DetectedObject]) -> None:
        """Update internal track history with current detections.

        Only objects with a valid track_id and position_3d are tracked.

        Args:
            detections: List of detected objects for the current frame.
        """
        self._frame_counter += 1
        for det in detections:
            if det.track_id is None or det.position_3d is None:
                continue
            if det.velocity is None:
                continue

            entry = _TrackHistoryEntry(
                position_3d=det.position_3d.copy(),
                velocity=det.velocity.copy(),
                class_name=det.class_name,
                timestamp=self._frame_counter,
            )
            history = self._track_history[det.track_id]
            history.append(entry)

            # Trim history to keep bounded memory
            if len(history) > self.history_length:
                self._track_history[det.track_id] = history[-self.history_length :]

    # ------------------------------------------------------------------
    # Private behavior detectors
    # ------------------------------------------------------------------

    def _detect_behaviors(
        self,
        det: DetectedObject,
        ego_state: EgoState,
    ) -> List[ScenarioTag]:
        """Run all behavior detectors on a single detection.

        Args:
            det: A single detected object.
            ego_state: Current ego vehicle state.

        Returns:
            List of behavior scenario tags (may be empty).
        """
        tags: List[ScenarioTag] = []

        # Cut-in detection
        cut_in_tag = self._detect_cut_in(det, ego_state)
        if cut_in_tag is not None:
            tags.append(cut_in_tag)

        # Lane change detection (requires track history)
        lane_change_tag = self._detect_lane_change(det, ego_state)
        if lane_change_tag is not None:
            tags.append(lane_change_tag)

        # Sudden braking detection
        braking_tag = self._detect_sudden_braking(det, ego_state)
        if braking_tag is not None:
            tags.append(braking_tag)

        # Jaywalking detection
        jaywalking_tag = self._detect_jaywalking(det, ego_state)
        if jaywalking_tag is not None:
            tags.append(jaywalking_tag)

        return tags

    def _detect_cut_in(
        self,
        det: DetectedObject,
        ego_state: EgoState,
    ) -> Optional[ScenarioTag]:
        """Detect cut-in: object has lateral velocity toward ego lane AND is close.

        A cut-in is identified when:
          1. The object's absolute lateral velocity exceeds the threshold.
          2. The lateral velocity direction is toward the ego lane center (y=0).
          3. The object is within the longitudinal distance threshold.

        Args:
            det: Detected object to evaluate.
            ego_state: Current ego vehicle state.

        Returns:
            ScenarioTag if cut-in detected, else None.
        """
        if det.velocity is None or det.position_3d is None:
            return None

        # Lateral velocity (vy in ego frame); positive = moving left
        vy = float(det.velocity[1])
        # Lateral position of object
        lateral_pos = float(det.position_3d[1])
        # Longitudinal distance
        longitudinal_dist = float(det.position_3d[0])

        # Object must be ahead and within distance threshold
        if longitudinal_dist <= 0 or longitudinal_dist > self.cut_in_distance_threshold:
            return None

        # Check if lateral velocity is toward ego lane center (y=0)
        # If object is to the right (lateral_pos < 0), it must be moving left (vy > 0)
        # If object is to the left (lateral_pos > 0), it must be moving right (vy < 0)
        moving_toward_ego = (lateral_pos < 0 and vy > 0) or (
            lateral_pos > 0 and vy < 0
        )

        if not moving_toward_ego:
            return None

        # Check if lateral velocity magnitude exceeds threshold
        if abs(vy) < self.cut_in_lateral_velocity_threshold:
            return None

        # Object must currently be outside ego lane (half lane width)
        half_lane = ego_state.lane_width / 2.0
        if abs(lateral_pos) < half_lane:
            # Already in ego lane, not a cut-in (may already be a lead vehicle)
            return None

        # Confidence: higher when closer and faster lateral movement
        dist_factor = 1.0 - (longitudinal_dist / self.cut_in_distance_threshold)
        vel_factor = min(abs(vy) / (self.cut_in_lateral_velocity_threshold * 3.0), 1.0)
        confidence = float(np.clip(dist_factor * 0.5 + vel_factor * 0.5, 0.3, 1.0))

        return ScenarioTag(
            node_id=NODE_ID_CUT_IN,
            confidence=confidence * det.confidence,
            source="auto",
        )

    def _detect_lane_change(
        self,
        det: DetectedObject,
        ego_state: EgoState,
    ) -> Optional[ScenarioTag]:
        """Detect lane change: object's lateral displacement exceeds threshold.

        Requires sufficient track history to measure lateral displacement over
        time. A lane change is flagged when the total lateral displacement
        from the earliest tracked position exceeds the configured threshold.

        Args:
            det: Detected object to evaluate.
            ego_state: Current ego vehicle state.

        Returns:
            ScenarioTag if lane change detected, else None.
        """
        if det.track_id is None or det.position_3d is None:
            return None

        history = self._track_history.get(det.track_id)
        if history is None or len(history) < 5:
            # Need at least 5 frames of history for reliable detection
            return None

        # Compute lateral displacement between earliest and latest positions
        earliest_pos = history[0].position_3d
        latest_pos = history[-1].position_3d

        lateral_displacement = abs(float(latest_pos[1]) - float(earliest_pos[1]))

        if lateral_displacement < self.lane_change_displacement_threshold:
            return None

        # Confidence based on how far the displacement exceeds the threshold
        excess_ratio = lateral_displacement / self.lane_change_displacement_threshold
        confidence = float(np.clip(min(excess_ratio, 2.0) / 2.0, 0.5, 1.0))

        return ScenarioTag(
            node_id=NODE_ID_LANE_CHANGE,
            confidence=confidence * det.confidence,
            source="auto",
        )

    def _detect_sudden_braking(
        self,
        det: DetectedObject,
        ego_state: EgoState,
    ) -> Optional[ScenarioTag]:
        """Detect sudden braking: object decelerates harder than threshold.

        Uses longitudinal velocity history to estimate deceleration. If the
        change in longitudinal velocity between consecutive frames indicates
        deceleration exceeding the threshold, the tag is generated.

        Args:
            det: Detected object to evaluate.
            ego_state: Current ego vehicle state.

        Returns:
            ScenarioTag if sudden braking detected, else None.
        """
        if det.track_id is None or det.velocity is None:
            return None

        history = self._track_history.get(det.track_id)
        if history is None or len(history) < 2:
            return None

        # Estimate longitudinal deceleration from last two frames
        # vx is longitudinal velocity in ego frame
        current_vx = float(history[-1].velocity[0])
        previous_vx = float(history[-2].velocity[0])

        # Deceleration estimate (negative means braking)
        # Assuming ~10 Hz frame rate => dt ~ 0.1s
        # We use velocity difference as a proxy for deceleration
        # For more accuracy, the caller should provide dt, but we estimate
        decel_estimate = current_vx - previous_vx  # negative if decelerating

        # Also check using a sliding window for robustness
        if len(history) >= 3:
            # Average deceleration over last 3 frames
            vx_values = [float(h.velocity[0]) for h in history[-3:]]
            decel_sliding = (vx_values[-1] - vx_values[0]) / 2.0
            # Use the more aggressive deceleration estimate
            decel_estimate = min(decel_estimate, decel_sliding)

        if decel_estimate >= self.braking_decel_threshold:
            # Not braking hard enough
            return None

        # Object must be ahead of ego
        if det.position_3d is not None and float(det.position_3d[0]) <= 0:
            return None

        # Confidence: stronger deceleration -> higher confidence
        decel_ratio = abs(decel_estimate) / abs(self.braking_decel_threshold)
        confidence = float(np.clip(decel_ratio * 0.8, 0.4, 1.0))

        return ScenarioTag(
            node_id=NODE_ID_SUDDEN_BRAKING,
            confidence=confidence * det.confidence,
            source="auto",
        )

    def _detect_jaywalking(
        self,
        det: DetectedObject,
        ego_state: EgoState,
    ) -> Optional[ScenarioTag]:
        """Detect jaywalking: pedestrian on road outside a crosswalk.

        Heuristic: a pedestrian is considered jaywalking if they are:
          1. Classified as "pedestrian"
          2. Located within the road area (lateral position within ~1.5 lane
             widths of ego, i.e., on the roadway)
          3. Moving laterally (crossing the road)

        Note: Without crosswalk map data, this uses a simplified heuristic.
        In production, crosswalk geometry from HD maps should be checked.

        Args:
            det: Detected object to evaluate.
            ego_state: Current ego vehicle state.

        Returns:
            ScenarioTag if jaywalking detected, else None.
        """
        if det.class_name.lower() != "pedestrian":
            return None

        if det.position_3d is None or det.velocity is None:
            return None

        lateral_pos = float(det.position_3d[1])
        longitudinal_pos = float(det.position_3d[0])
        lateral_velocity = abs(float(det.velocity[1]))

        # Pedestrian must be on or near the road
        # Heuristic: within 1.5 lane widths laterally (covers ego lane + adjacent)
        road_half_width = ego_state.lane_width * 1.5
        if abs(lateral_pos) > road_half_width:
            return None

        # Pedestrian must be roughly ahead (in sensor field of view)
        if longitudinal_pos < 0 or longitudinal_pos > 50.0:
            return None

        # Pedestrian should have lateral movement (crossing behavior)
        min_crossing_velocity = 0.3  # m/s - typical slow walk speed
        if lateral_velocity < min_crossing_velocity:
            return None

        # Confidence based on how centrally located and how fast crossing
        centrality = 1.0 - (abs(lateral_pos) / road_half_width)
        crossing_speed_factor = min(lateral_velocity / 1.5, 1.0)
        confidence = float(
            np.clip(centrality * 0.4 + crossing_speed_factor * 0.6, 0.3, 1.0)
        )

        return ScenarioTag(
            node_id=NODE_ID_JAYWALKING,
            confidence=confidence * det.confidence,
            source="auto",
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all track history and reset frame counter."""
        self._track_history.clear()
        self._frame_counter = 0

    @property
    def active_tracks(self) -> int:
        """Return the number of currently tracked objects."""
        return len(self._track_history)
