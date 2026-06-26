"""
Road topology extraction from vectorized map data.

Extracts lane structure, intersection type, and road geometry
from polyline-based map representations, mapping results to
PEGASUS scenario tree tags.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

from ..taxonomy.scenario_schema import ScenarioTag


class RoadTopologyExtractor:
    """Extracts road topology features from vectorized lane and centerline data."""

    # PEGASUS Layer 1 node IDs (semantic format)
    _ROAD_TYPE_NODES = {
        "highway": "L1.highway",
        "urban": "L1.urban",
        "rural": "L1.rural",
    }
    _GEOMETRY_NODES = {
        "straight": "L1.geometry.straight",
        "curve": "L1.geometry.curve",
        "intersection": "L1.intersection.crossroads",
    }
    _GRADE_NODES = {
        "flat": "L1.geometry.straight",
        "uphill": "L1.geometry.hill",
        "downhill": "L1.geometry.hill",
    }

    # Thresholds
    INTERSECTION_RADIUS: float = 50.0  # meters
    INTERSECTION_ANGLE_THRESHOLD: float = 45.0  # degrees
    HIGHWAY_MIN_LANES: int = 3
    HIGHWAY_MIN_WIDTH: float = 3.5  # meters per lane
    CURVATURE_STRAIGHT_THRESHOLD: float = 0.005  # 1/m; below this is considered straight
    GRADE_FLAT_THRESHOLD: float = 0.02  # 2% grade

    def extract_from_lanes(self, lanes: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract road structure information from lane polylines.

        Args:
            lanes: List of Nx2 or Nx3 arrays, each representing a lane polyline.

        Returns:
            Dict with keys: lane_count, avg_lane_width, road_type, total_road_width.
        """
        lane_count = len(lanes)

        if lane_count == 0:
            return {
                "lane_count": 0,
                "avg_lane_width": 0.0,
                "road_type": "unknown",
                "total_road_width": 0.0,
            }

        # Compute average lane width from distances between adjacent lanes
        lane_widths: List[float] = []
        for i in range(lane_count - 1):
            width = self._compute_lane_separation(lanes[i], lanes[i + 1])
            lane_widths.append(width)

        if lane_widths:
            avg_lane_width = float(np.mean(lane_widths))
        else:
            # Single lane: estimate width from polyline spread (fallback)
            avg_lane_width = 3.5  # default assumption

        total_road_width = avg_lane_width * lane_count

        # Determine road type
        road_type = self._classify_road_type(lane_count, avg_lane_width)

        return {
            "lane_count": lane_count,
            "avg_lane_width": avg_lane_width,
            "road_type": road_type,
            "total_road_width": total_road_width,
        }

    def detect_intersection(
        self, lanes: List[np.ndarray], ego_position: np.ndarray
    ) -> Optional[str]:
        """
        Detect if the ego vehicle is near an intersection and classify its type.

        Args:
            lanes: List of Nx2 or Nx3 arrays representing lane polylines.
            ego_position: [x, y] or [x, y, z] position of the ego vehicle.

        Returns:
            Intersection type string ("T-junction", "crossroad", "roundabout",
            "merge") or None if not near an intersection.
        """
        if len(lanes) < 2:
            return None

        ego_xy = np.asarray(ego_position[:2], dtype=np.float64)

        # Collect tangent directions of lanes near ego
        tangent_directions: List[float] = []  # angles in degrees
        nearby_lane_indices: List[int] = []

        for idx, lane in enumerate(lanes):
            lane_xy = np.asarray(lane[:, :2], dtype=np.float64)

            # Find closest point on this lane to ego
            distances = np.linalg.norm(lane_xy - ego_xy, axis=1)
            min_dist = float(np.min(distances))

            if min_dist > self.INTERSECTION_RADIUS:
                continue

            nearby_lane_indices.append(idx)

            # Compute tangent at closest point
            closest_idx = int(np.argmin(distances))
            tangent_angle = self._compute_tangent_angle(lane_xy, closest_idx)
            tangent_directions.append(tangent_angle)

        if len(tangent_directions) < 2:
            return None

        # Check for divergence/convergence: compute pairwise angle differences
        distinct_directions = self._cluster_directions(tangent_directions)
        num_branches = len(distinct_directions)

        if num_branches < 2:
            return None

        # Check for roundabout: detect circular curvature in any nearby lane
        for idx in nearby_lane_indices:
            lane_xy = np.asarray(lanes[idx][:, :2], dtype=np.float64)
            if self._detect_circular_curvature(lane_xy, ego_xy):
                return "roundabout"

        # Check if lanes converge (merge) vs diverge
        if self._detect_merge(lanes, nearby_lane_indices, ego_xy):
            return "merge"

        # Classify by branch count
        if num_branches == 3:
            return "T-junction"
        elif num_branches >= 4:
            return "crossroad"

        # Two distinct directions that differ by > 45 degrees means intersection
        if num_branches == 2:
            angle_diff = abs(distinct_directions[1] - distinct_directions[0])
            if angle_diff > self.INTERSECTION_ANGLE_THRESHOLD:
                return "T-junction"

        return None

    def compute_road_geometry(self, centerline: np.ndarray) -> Dict[str, float]:
        """
        Compute geometric properties of a road centerline.

        Args:
            centerline: Nx2 or Nx3 array of points along the road center.

        Returns:
            Dict with keys: max_curvature, mean_curvature, max_grade, mean_grade,
            total_heading_change, road_length.
        """
        centerline = np.asarray(centerline, dtype=np.float64)
        n_points = centerline.shape[0]

        if n_points < 2:
            return {
                "max_curvature": 0.0,
                "mean_curvature": 0.0,
                "max_grade": 0.0,
                "mean_grade": 0.0,
                "total_heading_change": 0.0,
                "road_length": 0.0,
            }

        # Extract 2D coordinates for curvature/heading
        xy = centerline[:, :2]

        # Compute segment lengths (arc length increments)
        diffs = np.diff(xy, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        road_length = float(np.sum(segment_lengths))

        # Avoid division by zero for degenerate segments
        segment_lengths_safe = np.where(
            segment_lengths > 1e-10, segment_lengths, 1e-10
        )

        # Compute curvature using the discrete cross-product formula
        # kappa_i = |dx' * dy'' - dy' * dx''| / (dx'^2 + dy'^2)^(3/2)
        curvatures = self._compute_discrete_curvature(xy, segment_lengths_safe)

        max_curvature = float(np.max(curvatures)) if len(curvatures) > 0 else 0.0
        mean_curvature = float(np.mean(curvatures)) if len(curvatures) > 0 else 0.0

        # Compute grade (slope) if 3D
        max_grade = 0.0
        mean_grade = 0.0
        if centerline.shape[1] >= 3:
            dz = np.diff(centerline[:, 2])
            grades = dz / segment_lengths_safe
            max_grade = float(np.max(np.abs(grades)))
            mean_grade = float(np.mean(grades))

        # Compute total heading change
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        heading_diffs = np.diff(headings)
        # Normalize to [-pi, pi]
        heading_diffs = (heading_diffs + np.pi) % (2 * np.pi) - np.pi
        total_heading_change = float(np.sum(np.abs(heading_diffs)))

        return {
            "max_curvature": max_curvature,
            "mean_curvature": mean_curvature,
            "max_grade": max_grade,
            "mean_grade": mean_grade,
            "total_heading_change": total_heading_change,
            "road_length": road_length,
        }

    def extract_topology(
        self,
        lanes: List[np.ndarray],
        ego_position: np.ndarray,
        centerline: np.ndarray,
    ) -> List[ScenarioTag]:
        """
        Perform complete road topology extraction and return scenario tags.

        Args:
            lanes: List of Nx2 or Nx3 arrays representing lane polylines.
            ego_position: [x, y] or [x, y, z] ego vehicle position.
            centerline: Nx2 or Nx3 array of points along the road center.

        Returns:
            List of ScenarioTag objects with PEGASUS node IDs.
        """
        tags: List[ScenarioTag] = []

        # --- Road type from lane structure ---
        lane_info = self.extract_from_lanes(lanes)
        road_type = lane_info["road_type"]

        if road_type == "highway":
            tags.append(
                ScenarioTag(
                    node_id=self._ROAD_TYPE_NODES["highway"],
                    confidence=self._road_type_confidence(lane_info),
                    source="auto",
                )
            )
        elif road_type == "arterial":
            # Arterial maps to urban
            tags.append(
                ScenarioTag(
                    node_id=self._ROAD_TYPE_NODES["urban"],
                    confidence=self._road_type_confidence(lane_info),
                    source="auto",
                )
            )
        elif road_type == "residential":
            # Residential maps to rural
            tags.append(
                ScenarioTag(
                    node_id=self._ROAD_TYPE_NODES["rural"],
                    confidence=self._road_type_confidence(lane_info),
                    source="auto",
                )
            )

        # --- Geometry classification ---
        geometry = self.compute_road_geometry(centerline)

        # Intersection detection
        intersection_type = self.detect_intersection(lanes, ego_position)
        if intersection_type is not None:
            confidence = min(1.0, 0.7 + 0.1 * lane_info["lane_count"])
            tags.append(
                ScenarioTag(
                    node_id=self._GEOMETRY_NODES["intersection"],
                    confidence=confidence,
                    source="auto",
                )
            )
        elif geometry["max_curvature"] > self.CURVATURE_STRAIGHT_THRESHOLD:
            # Curve classification: confidence scales with curvature magnitude
            curve_confidence = min(
                1.0, geometry["max_curvature"] / (self.CURVATURE_STRAIGHT_THRESHOLD * 10)
            )
            tags.append(
                ScenarioTag(
                    node_id=self._GEOMETRY_NODES["curve"],
                    confidence=max(0.5, curve_confidence),
                    source="auto",
                )
            )
        else:
            # Straight road: high confidence when curvature is very low
            straight_confidence = max(
                0.5,
                1.0 - geometry["mean_curvature"] / self.CURVATURE_STRAIGHT_THRESHOLD,
            )
            tags.append(
                ScenarioTag(
                    node_id=self._GEOMETRY_NODES["straight"],
                    confidence=min(1.0, straight_confidence),
                    source="auto",
                )
            )

        # --- Grade classification ---
        if centerline.shape[1] >= 3:
            if geometry["max_grade"] < self.GRADE_FLAT_THRESHOLD:
                grade_confidence = 1.0 - geometry["max_grade"] / self.GRADE_FLAT_THRESHOLD
                tags.append(
                    ScenarioTag(
                        node_id=self._GRADE_NODES["flat"],
                        confidence=max(0.5, min(1.0, grade_confidence)),
                        source="auto",
                    )
                )
            elif geometry["mean_grade"] > 0:
                # Uphill: confidence based on grade magnitude
                grade_confidence = min(1.0, geometry["mean_grade"] / 0.1)
                tags.append(
                    ScenarioTag(
                        node_id=self._GRADE_NODES["uphill"],
                        confidence=max(0.5, grade_confidence),
                        source="auto",
                    )
                )
            else:
                # Downhill
                grade_confidence = min(1.0, abs(geometry["mean_grade"]) / 0.1)
                tags.append(
                    ScenarioTag(
                        node_id=self._GRADE_NODES["downhill"],
                        confidence=max(0.5, grade_confidence),
                        source="auto",
                    )
                )
        else:
            # 2D data: assume flat with moderate confidence
            tags.append(
                ScenarioTag(
                    node_id=self._GRADE_NODES["flat"],
                    confidence=0.6,
                    source="auto",
                )
            )

        return tags

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _compute_lane_separation(
        self, lane_a: np.ndarray, lane_b: np.ndarray
    ) -> float:
        """Compute average lateral distance between two adjacent lane polylines."""
        a_xy = np.asarray(lane_a[:, :2], dtype=np.float64)
        b_xy = np.asarray(lane_b[:, :2], dtype=np.float64)

        # For each point in lane_a, find minimum distance to lane_b
        dist_matrix = cdist(a_xy, b_xy)
        min_dists_a_to_b = np.min(dist_matrix, axis=1)

        return float(np.mean(min_dists_a_to_b))

    def _classify_road_type(self, lane_count: int, avg_lane_width: float) -> str:
        """Classify road type based on lane count and width."""
        if lane_count >= self.HIGHWAY_MIN_LANES and avg_lane_width > self.HIGHWAY_MIN_WIDTH:
            return "highway"
        elif 2 <= lane_count <= 3:
            return "arterial"
        elif lane_count <= 2 and avg_lane_width <= self.HIGHWAY_MIN_WIDTH:
            return "residential"
        else:
            return "arterial"  # default fallback

    def _compute_tangent_angle(self, lane_xy: np.ndarray, idx: int) -> float:
        """
        Compute tangent direction (in degrees) at a given index along a polyline.
        Uses central differences where possible.
        """
        n = len(lane_xy)
        if n < 2:
            return 0.0

        if idx == 0:
            dx = lane_xy[1, 0] - lane_xy[0, 0]
            dy = lane_xy[1, 1] - lane_xy[0, 1]
        elif idx == n - 1:
            dx = lane_xy[-1, 0] - lane_xy[-2, 0]
            dy = lane_xy[-1, 1] - lane_xy[-2, 1]
        else:
            dx = lane_xy[idx + 1, 0] - lane_xy[idx - 1, 0]
            dy = lane_xy[idx + 1, 1] - lane_xy[idx - 1, 1]

        angle_rad = np.arctan2(dy, dx)
        return float(np.degrees(angle_rad))

    def _cluster_directions(
        self, angles_deg: List[float], threshold: float = 45.0
    ) -> List[float]:
        """
        Cluster a list of angles (degrees) into distinct directions.

        Angles within `threshold` degrees of each other are merged.
        Returns representative angles for each cluster.
        """
        if not angles_deg:
            return []

        # Normalize angles to [0, 360)
        normalized = [(a % 360.0) for a in angles_deg]
        normalized.sort()

        clusters: List[List[float]] = [[normalized[0]]]

        for angle in normalized[1:]:
            # Check angular distance to current cluster representative
            merged = False
            for cluster in clusters:
                rep = np.mean(cluster)
                diff = abs(angle - rep)
                # Handle wraparound
                diff = min(diff, 360.0 - diff)
                if diff < threshold:
                    cluster.append(angle)
                    merged = True
                    break
            if not merged:
                clusters.append([angle])

        # Also check if first and last clusters should merge (wraparound)
        if len(clusters) > 1:
            rep_first = np.mean(clusters[0])
            rep_last = np.mean(clusters[-1])
            diff = abs(rep_first - rep_last)
            diff = min(diff, 360.0 - diff)
            if diff < threshold:
                clusters[0].extend(clusters[-1])
                clusters.pop()

        return [float(np.mean(c)) for c in clusters]

    def _detect_circular_curvature(
        self, lane_xy: np.ndarray, ego_xy: np.ndarray
    ) -> bool:
        """
        Detect if a lane segment near ego exhibits circular (roundabout) curvature.

        A roundabout segment has consistently high curvature with the same sign,
        forming an arc.
        """
        # Extract portion within intersection radius
        distances = np.linalg.norm(lane_xy - ego_xy, axis=1)
        mask = distances < self.INTERSECTION_RADIUS
        nearby_points = lane_xy[mask]

        if len(nearby_points) < 5:
            return False

        # Compute curvature sign consistency
        diffs = np.diff(nearby_points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        segment_lengths_safe = np.where(segment_lengths > 1e-10, segment_lengths, 1e-10)

        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        heading_diffs = np.diff(headings)
        heading_diffs = (heading_diffs + np.pi) % (2 * np.pi) - np.pi

        if len(heading_diffs) < 3:
            return False

        # Roundabout: consistent turning direction and high total turn
        signs = np.sign(heading_diffs)
        sign_consistency = abs(float(np.mean(signs)))
        total_turn = abs(float(np.sum(heading_diffs)))

        # Consistent sign (>0.8) and total turn > 90 degrees suggests roundabout
        return sign_consistency > 0.8 and total_turn > np.pi / 2

    def _detect_merge(
        self,
        lanes: List[np.ndarray],
        nearby_indices: List[int],
        ego_xy: np.ndarray,
    ) -> bool:
        """
        Detect if nearby lanes are converging (merge) ahead of ego.

        Checks if the distance between lanes decreases in the forward direction.
        """
        if len(nearby_indices) < 2:
            return False

        # Compare distances between pairs of nearby lanes at different
        # longitudinal positions relative to ego
        for i in range(len(nearby_indices) - 1):
            lane_a = np.asarray(lanes[nearby_indices[i]][:, :2], dtype=np.float64)
            lane_b = np.asarray(lanes[nearby_indices[i + 1]][:, :2], dtype=np.float64)

            # Find points ahead of ego (using simple forward criterion)
            # Use the lane tangent at ego to define "ahead"
            dists_a = np.linalg.norm(lane_a - ego_xy, axis=1)
            closest_a = int(np.argmin(dists_a))

            dists_b = np.linalg.norm(lane_b - ego_xy, axis=1)
            closest_b = int(np.argmin(dists_b))

            # Look at points after closest point (ahead of ego)
            ahead_a = lane_a[closest_a:]
            ahead_b = lane_b[closest_b:]

            if len(ahead_a) < 3 or len(ahead_b) < 3:
                continue

            # Compute inter-lane distance at start vs end of ahead segment
            n_sample = min(len(ahead_a), len(ahead_b))
            start_dist = float(np.linalg.norm(ahead_a[0] - ahead_b[0]))
            end_dist = float(
                np.linalg.norm(ahead_a[n_sample - 1] - ahead_b[n_sample - 1])
            )

            # Lanes converge if end distance is significantly less than start
            if end_dist < start_dist * 0.5 and start_dist > 1.0:
                return True

        return False

    def _compute_discrete_curvature(
        self, xy: np.ndarray, segment_lengths: np.ndarray
    ) -> np.ndarray:
        """
        Compute discrete curvature at interior points using the cross-product formula.

        For three consecutive points P_{i-1}, P_i, P_{i+1}:
            kappa_i = 2 * |cross(P_i - P_{i-1}, P_{i+1} - P_i)| /
                      (|P_i - P_{i-1}| * |P_{i+1} - P_i| * |P_{i+1} - P_{i-1}|)
        """
        n = len(xy)
        if n < 3:
            return np.array([], dtype=np.float64)

        curvatures = np.zeros(n - 2, dtype=np.float64)

        for i in range(1, n - 1):
            v1 = xy[i] - xy[i - 1]  # vector from P_{i-1} to P_i
            v2 = xy[i + 1] - xy[i]  # vector from P_i to P_{i+1}

            # 2D cross product magnitude
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])

            len_v1 = np.linalg.norm(v1)
            len_v2 = np.linalg.norm(v2)
            len_v3 = np.linalg.norm(xy[i + 1] - xy[i - 1])

            denom = len_v1 * len_v2 * len_v3
            if denom > 1e-12:
                curvatures[i - 1] = 2.0 * cross / denom

        return curvatures

    def _road_type_confidence(self, lane_info: Dict[str, Any]) -> float:
        """Compute confidence for road type classification."""
        road_type = lane_info["road_type"]
        lane_count = lane_info["lane_count"]
        avg_width = lane_info["avg_lane_width"]

        if road_type == "highway":
            # More lanes and wider = higher confidence
            lane_factor = min(1.0, lane_count / 5.0)
            width_factor = min(1.0, avg_width / 4.0)
            return max(0.6, min(1.0, 0.5 * lane_factor + 0.5 * width_factor))
        elif road_type == "arterial":
            # Moderate confidence for typical arterial
            return 0.7
        elif road_type == "residential":
            # Narrower and fewer lanes = higher confidence
            narrowness = max(0.0, 1.0 - avg_width / 4.0)
            return max(0.5, min(1.0, 0.6 + 0.4 * narrowness))
        else:
            return 0.5
