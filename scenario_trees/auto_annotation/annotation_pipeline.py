"""
End-to-end annotation pipeline that orchestrates all auto-annotation classifiers.

Processes recording directories or individual frames to produce structured
ScenarioAnnotation objects containing scenario tags from scene classification,
weather detection, object behavior analysis, road topology extraction, and
temporal event detection.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..taxonomy.scenario_schema import ScenarioAnnotation, ScenarioTag
from .object_detector import DetectedObject, EgoState, ObjectScenarioTagger
from .scene_classifier import CLIPSceneClassifier
from .weather_classifier import WeatherClassifier
from .road_topology_extractor import RoadTopologyExtractor
from .temporal_event_detector import TemporalEventDetector, Track, Event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FrameData dataclass
# ---------------------------------------------------------------------------


@dataclass
class FrameData:
    """Container for all sensor data associated with a single frame.

    Attributes:
        image: HWC uint8 camera image, may be None if no camera data available.
        points: Nx3 or Nx4 lidar point cloud (x, y, z [, intensity]),
            may be None if no lidar data available.
        detections: Object detections for this frame.
        ego_state: Current ego vehicle state.
        timestamp: Frame timestamp in seconds.
        radar_data: Optional radar point cloud or detection array.
        lanes: Optional list of lane polylines from HD map (each Nx2 array).
        centerline: Optional road centerline as Nx2 array.
    """

    image: Optional[np.ndarray]  # HWC uint8
    points: Optional[np.ndarray]  # Nx3 or Nx4
    detections: List[DetectedObject]
    ego_state: EgoState
    timestamp: float
    radar_data: Optional[np.ndarray] = None
    lanes: Optional[List[np.ndarray]] = None
    centerline: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Node ID mapping constants
# ---------------------------------------------------------------------------

# Scene type -> road type tags (semantic L1.x IDs)
_ROAD_TYPE_MAP: Dict[str, str] = {
    "highway": "L1.highway",
    "motorway": "L1.highway",
    "urban": "L1.urban",
    "city": "L1.urban",
    "rural": "L1.rural",
    "country_road": "L1.rural",
    "intersection": "L1.intersection.crossroads",
    "roundabout": "L1.intersection.roundabout",
    "parking": "L1.urban",
    "tunnel": "L1.geometry.tunnel",
    "bridge": "L1.geometry.bridge",
}

# Weather -> L5.weather.x tags
_WEATHER_TAG_MAP: Dict[str, str] = {
    "clear": "L5.weather.clear",
    "rain": "L5.weather.rain",
    "heavy_rain": "L5.weather.heavy_rain",
    "snow": "L5.weather.snow",
    "fog": "L5.weather.fog",
    "overcast": "L5.weather.clear",
    "hail": "L5.weather.hail",
}

# Time of day -> L5.lighting.x tags
_TIME_OF_DAY_MAP: Dict[str, str] = {
    "day": "L5.lighting.daylight",
    "daylight": "L5.lighting.daylight",
    "night": "L5.lighting.night",
    "dawn": "L5.lighting.dawn",
    "dusk": "L5.lighting.dusk",
    "twilight": "L5.lighting.dusk",
}

# Temporal events -> tags
_TEMPORAL_EVENT_MAP: Dict[str, str] = {
    "cut_in": "L4.behavior.cut_in",
    "lane_change": "L4.behavior.lane_change",
    "hard_braking": "L4.behavior.sudden_braking",
    "emergency_brake": "L4.behavior.sudden_braking",
}


# ---------------------------------------------------------------------------
# AnnotationPipeline
# ---------------------------------------------------------------------------


class AnnotationPipeline:
    """End-to-end pipeline that orchestrates all scenario classifiers.

    Combines scene classification (CLIP), weather analysis, object behavior
    tagging, road topology extraction, and temporal event detection into a
    unified annotation workflow.

    Each sub-classifier is lazy-initialized on first use to minimize startup
    cost and memory when only a subset of classifiers is needed.

    Args:
        enable_clip: Whether to use CLIP scene classification.
        enable_weather: Whether to use weather classification.
        enable_topology: Whether to use road topology extraction.
        enable_objects: Whether to use object scenario tagging.
        enable_temporal: Whether to use temporal event detection.
        clip_model_name: CLIP model architecture name.
        clip_pretrained: CLIP pretrained weights identifier.
        device: Compute device (e.g., 'cuda', 'cpu'). None for auto-detect.
        use_clip: Alias for enable_clip (for backward compatibility).
        confidence_threshold: Minimum confidence for tags to be included in output.
    """

    def __init__(
        self,
        enable_clip: bool = True,
        enable_weather: bool = True,
        enable_topology: bool = True,
        enable_objects: bool = True,
        enable_temporal: bool = True,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        *,
        use_clip: Optional[bool] = None,
        confidence_threshold: float = 0.0,
    ) -> None:
        # Handle backward-compat alias
        if use_clip is not None:
            self.enable_clip = use_clip
        else:
            self.enable_clip = enable_clip

        self.enable_weather = enable_weather
        self.enable_topology = enable_topology
        self.enable_objects = enable_objects
        self.enable_temporal = enable_temporal
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Lazy-initialized classifiers (None until first use)
        self._scene_classifier: Optional[CLIPSceneClassifier] = None
        self._weather_classifier: Optional[WeatherClassifier] = None
        self._object_tagger: Optional[ObjectScenarioTagger] = None
        self._topology_extractor: Optional[RoadTopologyExtractor] = None
        self._temporal_detector: Optional[TemporalEventDetector] = None

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _get_scene_classifier(self) -> CLIPSceneClassifier:
        """Lazy-initialize and return the CLIP scene classifier."""
        if self._scene_classifier is None:
            logger.info(
                "Initializing CLIPSceneClassifier (model=%s, pretrained=%s)",
                self.clip_model_name,
                self.clip_pretrained,
            )
            self._scene_classifier = CLIPSceneClassifier(
                model_name=self.clip_model_name,
                pretrained=self.clip_pretrained,
                device=self.device,
            )
        return self._scene_classifier

    def _get_weather_classifier(self) -> WeatherClassifier:
        """Lazy-initialize and return the weather classifier."""
        if self._weather_classifier is None:
            logger.info("Initializing WeatherClassifier")
            self._weather_classifier = WeatherClassifier()
        return self._weather_classifier

    def _get_object_tagger(self) -> ObjectScenarioTagger:
        """Lazy-initialize and return the object scenario tagger."""
        if self._object_tagger is None:
            logger.info("Initializing ObjectScenarioTagger")
            self._object_tagger = ObjectScenarioTagger()
        return self._object_tagger

    def _get_topology_extractor(self) -> RoadTopologyExtractor:
        """Lazy-initialize and return the road topology extractor."""
        if self._topology_extractor is None:
            logger.info("Initializing RoadTopologyExtractor")
            self._topology_extractor = RoadTopologyExtractor()
        return self._topology_extractor

    def _get_temporal_detector(self) -> TemporalEventDetector:
        """Lazy-initialize and return the temporal event detector."""
        if self._temporal_detector is None:
            logger.info("Initializing TemporalEventDetector")
            self._temporal_detector = TemporalEventDetector()
        return self._temporal_detector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame_data: FrameData) -> List[ScenarioTag]:
        """Process a single frame and return scenario tags.

        Runs all enabled classifiers on the frame data and aggregates the
        resulting tags, deduplicating by node_id (keeping highest confidence).

        Args:
            frame_data: Complete sensor data for one frame.

        Returns:
            List of deduplicated ScenarioTag instances.
        """
        all_tags: List[ScenarioTag] = []

        # --- Scene classification (CLIP) ---
        if self.enable_clip and frame_data.image is not None:
            try:
                classifier = self._get_scene_classifier()
                scene_result = classifier.classify_scene(frame_data.image)
                scene_tags = self._map_scene_to_tags(scene_result)
                all_tags.extend(scene_tags)
            except Exception as e:
                logger.warning("Scene classification failed: %s", e)

        # --- Weather classification ---
        if self.enable_weather:
            try:
                weather_clf = self._get_weather_classifier()
                weather_result: Optional[Dict[str, float]] = None

                # Try camera-based weather classification
                if frame_data.image is not None:
                    camera_weather = weather_clf.classify_from_camera(frame_data.image)
                else:
                    camera_weather = None

                # Try lidar-based weather classification
                if frame_data.points is not None:
                    lidar_weather = weather_clf.classify_from_lidar(frame_data.points)
                else:
                    lidar_weather = None

                # Try radar-based weather classification
                if frame_data.radar_data is not None:
                    radar_weather = weather_clf.classify_from_radar(frame_data.radar_data)
                else:
                    radar_weather = None

                # Fuse available classifications
                if any(r is not None for r in (camera_weather, lidar_weather, radar_weather)):
                    weather_result = weather_clf.fuse_classifications(
                        camera_weather, lidar_weather, radar_weather
                    )

                if weather_result is not None:
                    weather_tags = self._map_weather_to_tags(weather_result)
                    all_tags.extend(weather_tags)
            except Exception as e:
                logger.warning("Weather classification failed: %s", e)

        # --- Object scenario tagging ---
        if self.enable_objects and frame_data.detections:
            try:
                tagger = self._get_object_tagger()
                object_tags = tagger.tag_object_scenarios(
                    frame_data.detections, frame_data.ego_state
                )
                all_tags.extend(object_tags)
            except Exception as e:
                logger.warning("Object scenario tagging failed: %s", e)

        # --- Road topology extraction ---
        if self.enable_topology and frame_data.lanes is not None:
            try:
                extractor = self._get_topology_extractor()
                if frame_data.centerline is not None:
                    # Use the full extract_topology method when centerline is available
                    ego_pos = frame_data.ego_state.position[:2]
                    topology_tags = extractor.extract_topology(
                        frame_data.lanes, ego_pos, frame_data.centerline
                    )
                else:
                    # Fallback to simpler lane-based extraction
                    topology_tags = self._extract_topology_tags(
                        extractor, frame_data.lanes, frame_data.centerline
                    )
                all_tags.extend(topology_tags)
            except Exception as e:
                logger.warning("Road topology extraction failed: %s", e)

        # --- Filter by confidence threshold and deduplicate ---
        filtered_tags = [
            tag for tag in all_tags if tag.confidence >= self.confidence_threshold
        ]
        return self._aggregate_tags(filtered_tags)

    def process_recording(self, recording_path: str) -> ScenarioAnnotation:
        """Process an entire recording directory and return a full annotation.

        Loads frames sequentially from the recording directory, runs per-frame
        classification, then performs temporal event detection across the full
        sequence.

        Expected directory structure:
            recording_path/
                images/         - Camera images (PNG/JPG), sorted by name
                pointclouds/    - LiDAR point clouds (NPY/BIN), sorted by name
                detections/     - Detection JSON files, sorted by name
                ego_states/     - Ego state JSON files, sorted by name

        Args:
            recording_path: Path to the recording directory.

        Returns:
            ScenarioAnnotation with aggregated tags and metadata.
        """
        recording_dir = Path(recording_path)
        recording_id = recording_dir.name

        logger.info("Processing recording: %s", recording_id)

        # Determine frame count from available data
        frame_count = self._count_frames(recording_dir)
        if frame_count == 0:
            logger.warning("No frames found in recording: %s", recording_path)
            return ScenarioAnnotation(
                recording_id=recording_id,
                tags=[],
                metadata={"frame_count": 0, "processing_timestamp": datetime.now(timezone.utc).isoformat()},
            )

        # Process frames
        all_frame_tags: List[ScenarioTag] = []
        all_ego_states: List[EgoState] = []
        all_tracks: Dict[int, List[Any]] = {}

        for frame_idx in range(frame_count):
            frame_data = self._load_frame(recording_path, frame_idx)
            if frame_data is None:
                logger.debug("Skipping frame %d (could not load)", frame_idx)
                continue

            # Run per-frame classification
            frame_tags = self.process_frame(frame_data)
            all_frame_tags.extend(frame_tags)

            # Accumulate ego states for temporal analysis
            all_ego_states.append(frame_data.ego_state)

            # Accumulate track data for temporal event detection
            for det in frame_data.detections:
                if det.track_id is not None and det.position_3d is not None:
                    if det.track_id not in all_tracks:
                        all_tracks[det.track_id] = []
                    all_tracks[det.track_id].append({
                        "position": det.position_3d.copy(),
                        "velocity": det.velocity.copy() if det.velocity is not None else np.zeros(3),
                        "timestamp": frame_data.timestamp,
                        "class_name": det.class_name,
                    })

        # --- Temporal event detection ---
        if self.enable_temporal and len(all_ego_states) > 1:
            try:
                temporal_detector = self._get_temporal_detector()

                # Detect hard braking events
                braking_events = temporal_detector.detect_hard_braking(all_ego_states)
                for event in braking_events:
                    event_node_id = _TEMPORAL_EVENT_MAP.get(
                        event.event_type, "L4.behavior.sudden_braking"
                    )
                    all_frame_tags.append(
                        ScenarioTag(
                            node_id=event_node_id,
                            confidence=event.confidence if hasattr(event, "confidence") else 0.8,
                            source="auto",
                        )
                    )

                # Detect lane change events
                lane_change_events = temporal_detector.detect_lane_change(all_ego_states)
                for event in lane_change_events:
                    event_node_id = _TEMPORAL_EVENT_MAP.get(
                        event.event_type, "L4.behavior.lane_change"
                    )
                    all_frame_tags.append(
                        ScenarioTag(
                            node_id=event_node_id,
                            confidence=event.confidence if hasattr(event, "confidence") else 0.8,
                            source="auto",
                        )
                    )

                # Build Track objects for cut-in and near-miss detection
                tracks: List[Track] = []
                for track_id, track_points in all_tracks.items():
                    if len(track_points) >= 2:
                        timestamps = [tp["timestamp"] for tp in track_points]
                        positions = np.array(
                            [tp["position"] for tp in track_points], dtype=np.float64
                        )
                        velocities = np.array(
                            [tp["velocity"] for tp in track_points], dtype=np.float64
                        )
                        tracks.append(Track(
                            track_id=track_id,
                            timestamps=timestamps,
                            positions=positions,
                            velocities=velocities,
                            class_name=track_points[0]["class_name"],
                        ))

                if tracks and all_ego_states:
                    # Detect cut-in events
                    cut_in_events = temporal_detector.detect_cut_in(
                        tracks, all_ego_states[-1]
                    )
                    for event in cut_in_events:
                        all_frame_tags.append(
                            ScenarioTag(
                                node_id="L4.behavior.cut_in",
                                confidence=event.confidence,
                                source="auto",
                            )
                        )

                    # Detect near-miss events
                    near_miss_events = temporal_detector.detect_near_miss(
                        tracks, all_ego_states
                    )
                    for event in near_miss_events:
                        all_frame_tags.append(
                            ScenarioTag(
                                node_id="L4.behavior.sudden_braking",
                                confidence=event.confidence,
                                source="auto",
                            )
                        )

            except Exception as e:
                logger.warning("Temporal event detection failed: %s", e)

        # Aggregate all tags across frames (max confidence per node_id)
        aggregated_tags = self._aggregate_tags(all_frame_tags)

        # Build annotation
        annotation = ScenarioAnnotation(
            recording_id=recording_id,
            tags=aggregated_tags,
            metadata={
                "frame_count": frame_count,
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "ego_state_count": len(all_ego_states),
                "track_count": len(all_tracks),
            },
        )

        logger.info(
            "Recording %s processed: %d tags from %d frames",
            recording_id,
            len(aggregated_tags),
            frame_count,
        )
        return annotation

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _load_frame(self, recording_path: str, frame_idx: int) -> Optional[FrameData]:
        """Load a single frame's data from the recording directory.

        Looks for:
          - images/{frame_idx:06d}.png or .jpg
          - pointclouds/{frame_idx:06d}.npy or .bin
          - detections/{frame_idx:06d}.json
          - ego_states/{frame_idx:06d}.json

        Args:
            recording_path: Path to the recording directory.
            frame_idx: Zero-based frame index.

        Returns:
            FrameData if at least ego_state could be loaded, None otherwise.
        """
        rec_dir = Path(recording_path)
        frame_name = f"{frame_idx:06d}"

        # Load image
        image: Optional[np.ndarray] = None
        images_dir = rec_dir / "images"
        if images_dir.exists():
            for ext in (".png", ".jpg", ".jpeg"):
                img_path = images_dir / f"{frame_name}{ext}"
                if img_path.exists():
                    try:
                        # Use numpy to load image if possible, fallback to raw bytes
                        import cv2  # type: ignore
                        image = cv2.imread(str(img_path))
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    except ImportError:
                        # Fallback: try PIL
                        try:
                            from PIL import Image  # type: ignore
                            pil_img = Image.open(str(img_path)).convert("RGB")
                            image = np.array(pil_img, dtype=np.uint8)
                        except ImportError:
                            logger.debug(
                                "Neither cv2 nor PIL available; skipping image %s",
                                img_path,
                            )
                    break

        # Load point cloud
        points: Optional[np.ndarray] = None
        pc_dir = rec_dir / "pointclouds"
        if pc_dir.exists():
            npy_path = pc_dir / f"{frame_name}.npy"
            bin_path = pc_dir / f"{frame_name}.bin"
            if npy_path.exists():
                try:
                    points = np.load(str(npy_path))
                except Exception as e:
                    logger.debug("Failed to load point cloud %s: %s", npy_path, e)
            elif bin_path.exists():
                try:
                    raw = np.fromfile(str(bin_path), dtype=np.float32)
                    # Assume 4 columns (x, y, z, intensity)
                    if raw.size % 4 == 0:
                        points = raw.reshape(-1, 4)
                    elif raw.size % 3 == 0:
                        points = raw.reshape(-1, 3)
                    else:
                        logger.debug(
                            "Point cloud %s has unexpected size %d", bin_path, raw.size
                        )
                except Exception as e:
                    logger.debug("Failed to load point cloud %s: %s", bin_path, e)

        # Load detections
        detections: List[DetectedObject] = []
        det_dir = rec_dir / "detections"
        if det_dir.exists():
            det_path = det_dir / f"{frame_name}.json"
            if det_path.exists():
                try:
                    with open(str(det_path), "r", encoding="utf-8") as f:
                        det_data = json.load(f)
                    detections = self._parse_detections(det_data)
                except Exception as e:
                    logger.debug("Failed to load detections %s: %s", det_path, e)

        # Load ego state
        ego_state: Optional[EgoState] = None
        ego_dir = rec_dir / "ego_states"
        if ego_dir.exists():
            ego_path = ego_dir / f"{frame_name}.json"
            if ego_path.exists():
                try:
                    with open(str(ego_path), "r", encoding="utf-8") as f:
                        ego_data = json.load(f)
                    ego_state = self._parse_ego_state(ego_data)
                except Exception as e:
                    logger.debug("Failed to load ego state %s: %s", ego_path, e)

        if ego_state is None:
            # Cannot process frame without ego state
            return None

        # Determine timestamp
        timestamp = ego_data.get("timestamp", float(frame_idx) * 0.1) if ego_state else float(frame_idx) * 0.1

        return FrameData(
            image=image,
            points=points,
            detections=detections,
            ego_state=ego_state,
            timestamp=timestamp,
        )

    def _map_scene_to_tags(self, scene_result: Dict[str, Dict[str, float]]) -> List[ScenarioTag]:
        """Convert CLIP scene classification output to ScenarioTag objects.

        CLIP classifier returns a dict with keys like 'road_type', 'weather',
        'time_of_day', each mapping to a dict of {label: confidence}.

        Args:
            scene_result: CLIP classification output.

        Returns:
            List of ScenarioTag instances for recognized scene attributes.
        """
        tags: List[ScenarioTag] = []

        # Road type classification -> L1.x
        road_types = scene_result.get("road_type", {})
        for label, confidence in road_types.items():
            node_id = _ROAD_TYPE_MAP.get(label.lower())
            if node_id is not None and confidence > 0.0:
                tags.append(
                    ScenarioTag(
                        node_id=node_id,
                        confidence=float(np.clip(confidence, 0.0, 1.0)),
                        source="model",
                    )
                )

        # Weather from scene -> L5.1.x
        weather = scene_result.get("weather", {})
        for label, confidence in weather.items():
            node_id = _WEATHER_TAG_MAP.get(label.lower())
            if node_id is not None and confidence > 0.0:
                tags.append(
                    ScenarioTag(
                        node_id=node_id,
                        confidence=float(np.clip(confidence, 0.0, 1.0)),
                        source="model",
                    )
                )

        # Time of day -> L5.2.x
        time_of_day = scene_result.get("time_of_day", {})
        for label, confidence in time_of_day.items():
            node_id = _TIME_OF_DAY_MAP.get(label.lower())
            if node_id is not None and confidence > 0.0:
                tags.append(
                    ScenarioTag(
                        node_id=node_id,
                        confidence=float(np.clip(confidence, 0.0, 1.0)),
                        source="model",
                    )
                )

        return tags

    def _map_weather_to_tags(self, weather_result: Dict[str, float]) -> List[ScenarioTag]:
        """Convert weather classification output to ScenarioTag objects.

        Args:
            weather_result: Dict mapping weather condition to probability.

        Returns:
            List of ScenarioTag instances for detected weather conditions.
        """
        tags: List[ScenarioTag] = []
        for label, confidence in weather_result.items():
            node_id = _WEATHER_TAG_MAP.get(label.lower())
            if node_id is not None and confidence > 0.0:
                tags.append(
                    ScenarioTag(
                        node_id=node_id,
                        confidence=float(np.clip(confidence, 0.0, 1.0)),
                        source="auto",
                    )
                )
        return tags

    def _aggregate_tags(self, all_tags: List[ScenarioTag]) -> List[ScenarioTag]:
        """Deduplicate tags by node_id, keeping the highest confidence for each.

        Args:
            all_tags: List of tags, possibly with duplicate node_ids.

        Returns:
            Deduplicated list with one tag per node_id (max confidence).
        """
        if not all_tags:
            return []

        best_tags: Dict[str, ScenarioTag] = {}
        for tag in all_tags:
            existing = best_tags.get(tag.node_id)
            if existing is None or tag.confidence > existing.confidence:
                best_tags[tag.node_id] = tag

        return list(best_tags.values())

    def _extract_topology_tags(
        self,
        extractor: RoadTopologyExtractor,
        lanes: List[np.ndarray],
        centerline: Optional[np.ndarray],
    ) -> List[ScenarioTag]:
        """Extract road topology tags from lane data.

        Args:
            extractor: RoadTopologyExtractor instance.
            lanes: List of lane polylines.
            centerline: Optional road centerline.

        Returns:
            List of ScenarioTag for road topology features.
        """
        tags: List[ScenarioTag] = []

        # Extract lane information
        lane_result = extractor.extract_from_lanes(lanes)
        lane_count = lane_result.get("lane_count", 0)

        # Multi-lane highway heuristic
        if lane_count >= 3:
            tags.append(
                ScenarioTag(
                    node_id="L1.highway",
                    confidence=min(0.5 + 0.1 * lane_count, 0.9),
                    source="auto",
                )
            )
        elif lane_count == 2:
            tags.append(
                ScenarioTag(
                    node_id="L1.urban",
                    confidence=0.5,
                    source="auto",
                )
            )
        elif lane_count == 1:
            tags.append(
                ScenarioTag(
                    node_id="L1.rural",
                    confidence=0.5,
                    source="auto",
                )
            )

        # Road geometry from centerline
        if centerline is not None and len(centerline) >= 3:
            geometry = extractor.compute_road_geometry(centerline)
            mean_curvature = geometry.get("mean_curvature", 0.0)

            # High curvature suggests curved road / intersection / roundabout
            if mean_curvature > 0.05:
                tags.append(
                    ScenarioTag(
                        node_id="L1.intersection.roundabout",
                        confidence=float(np.clip(mean_curvature * 5.0, 0.3, 0.9)),
                        source="auto",
                    )
                )
            elif mean_curvature > 0.02:
                tags.append(
                    ScenarioTag(
                        node_id="L1.geometry.curve",
                        confidence=float(np.clip(mean_curvature * 10.0, 0.3, 0.8)),
                        source="auto",
                    )
                )

        return tags

    def _count_frames(self, recording_dir: Path) -> int:
        """Count available frames in a recording directory.

        Uses the ego_states directory as the primary source of frame count,
        falling back to images or pointclouds directories.

        Args:
            recording_dir: Path to the recording directory.

        Returns:
            Number of frames available.
        """
        # Try ego_states first (required for processing)
        ego_dir = recording_dir / "ego_states"
        if ego_dir.exists():
            count = len([
                f for f in os.listdir(str(ego_dir))
                if f.endswith(".json")
            ])
            if count > 0:
                return count

        # Fallback to images
        images_dir = recording_dir / "images"
        if images_dir.exists():
            count = len([
                f for f in os.listdir(str(images_dir))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            if count > 0:
                return count

        # Fallback to pointclouds
        pc_dir = recording_dir / "pointclouds"
        if pc_dir.exists():
            count = len([
                f for f in os.listdir(str(pc_dir))
                if f.endswith((".npy", ".bin"))
            ])
            if count > 0:
                return count

        return 0

    def _parse_detections(self, det_data: Any) -> List[DetectedObject]:
        """Parse detection JSON data into DetectedObject instances.

        Expected JSON structure:
            {
                "detections": [
                    {
                        "class_name": "car",
                        "bbox": [x1, y1, x2, y2],
                        "track_id": 1,
                        "velocity": [vx, vy, vz],
                        "position_3d": [x, y, z],
                        "confidence": 0.95
                    },
                    ...
                ]
            }

        Args:
            det_data: Parsed JSON data (dict or list).

        Returns:
            List of DetectedObject instances.
        """
        detections: List[DetectedObject] = []

        # Handle both formats: list directly or dict with "detections" key
        if isinstance(det_data, dict):
            items = det_data.get("detections", det_data.get("objects", []))
        elif isinstance(det_data, list):
            items = det_data
        else:
            return detections

        for item in items:
            try:
                class_name = item.get("class_name", item.get("class", "unknown"))
                bbox = np.array(item.get("bbox", [0, 0, 0, 0]), dtype=np.float32)

                track_id = item.get("track_id")
                if track_id is not None:
                    track_id = int(track_id)

                velocity = None
                if "velocity" in item and item["velocity"] is not None:
                    velocity = np.array(item["velocity"], dtype=np.float32)

                position_3d = None
                if "position_3d" in item and item["position_3d"] is not None:
                    position_3d = np.array(item["position_3d"], dtype=np.float32)

                confidence = float(item.get("confidence", 1.0))

                detections.append(DetectedObject(
                    class_name=class_name,
                    bbox=bbox,
                    track_id=track_id,
                    velocity=velocity,
                    position_3d=position_3d,
                    confidence=confidence,
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.debug("Failed to parse detection: %s", e)
                continue

        return detections

    def _parse_ego_state(self, ego_data: Dict[str, Any]) -> EgoState:
        """Parse ego state JSON data into an EgoState instance.

        Expected JSON structure:
            {
                "position": [x, y, z],
                "velocity": [vx, vy, vz],
                "acceleration": [ax, ay, az],
                "heading": 0.0,
                "lane_width": 3.7,
                "timestamp": 0.0
            }

        Args:
            ego_data: Parsed JSON dict for ego state.

        Returns:
            EgoState instance.
        """
        position = np.array(
            ego_data.get("position", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        velocity = np.array(
            ego_data.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        acceleration = np.array(
            ego_data.get("acceleration", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        heading = float(ego_data.get("heading", 0.0))
        lane_width = float(ego_data.get("lane_width", 3.7))

        return EgoState(
            velocity=velocity,
            acceleration=acceleration,
            position=position,
            heading=heading,
            lane_width=lane_width,
        )
