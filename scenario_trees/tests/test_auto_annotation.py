"""Tests for the auto_annotation module."""

import pytest
import numpy as np

from scenario_trees.auto_annotation.object_detector import (
    ObjectScenarioTagger,
    DetectedObject,
    EgoState,
)
from scenario_trees.auto_annotation.weather_classifier import WeatherClassifier
from scenario_trees.auto_annotation.road_topology_extractor import RoadTopologyExtractor
from scenario_trees.auto_annotation.temporal_event_detector import (
    TemporalEventDetector,
    Track,
    Event,
)
from scenario_trees.auto_annotation.annotation_pipeline import (
    AnnotationPipeline,
    FrameData,
)


class TestObjectDetector:
    """Test object-based scenario tagging."""

    def setup_method(self):
        """Create tagger instance."""
        self.tagger = ObjectScenarioTagger()

    def test_basic_object_counting(self):
        """Should tag scenarios based on object counts."""
        detections = [
            DetectedObject(
                class_name="car",
                bbox=np.array([100, 200, 150, 250]),
                track_id=1,
                velocity=np.array([10.0, 0.0, 0.0]),
                position_3d=np.array([20.0, 0.0, 0.0]),
            ),
            DetectedObject(
                class_name="car",
                bbox=np.array([200, 200, 260, 260]),
                track_id=2,
                velocity=np.array([12.0, 0.0, 0.0]),
                position_3d=np.array([30.0, 2.0, 0.0]),
            ),
            DetectedObject(
                class_name="pedestrian",
                bbox=np.array([50, 300, 80, 380]),
                track_id=3,
                velocity=np.array([0.0, 1.0, 0.0]),
                position_3d=np.array([5.0, 4.0, 0.0]),
            ),
        ]
        ego = EgoState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([15.0, 0.0, 0.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            heading=0.0,
            timestamp=0.0,
            lane_id=1,
        )

        tags = self.tagger.tag_object_scenarios(detections, ego)
        assert len(tags) > 0

        # Should detect cars and pedestrians
        tag_ids = [t.node_id for t in tags]
        assert any("car" in tid for tid in tag_ids)
        assert any("pedestrian" in tid for tid in tag_ids)

    def test_cut_in_detection(self):
        """Should detect cut-in behavior."""
        # Object moving laterally toward ego lane
        detections = [
            DetectedObject(
                class_name="car",
                bbox=np.array([200, 200, 280, 260]),
                track_id=1,
                velocity=np.array([10.0, -2.5, 0.0]),  # Moving toward ego lane
                position_3d=np.array([15.0, 3.0, 0.0]),  # Close and in adjacent lane
            ),
        ]
        ego = EgoState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([15.0, 0.0, 0.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            heading=0.0,
            timestamp=0.0,
            lane_id=1,
        )

        tags = self.tagger.tag_object_scenarios(detections, ego)
        tag_ids = [t.node_id for t in tags]
        assert any("cut" in tid.lower() for tid in tag_ids)

    def test_empty_detections(self):
        """Should handle empty detection list."""
        ego = EgoState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([15.0, 0.0, 0.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            heading=0.0,
            timestamp=0.0,
            lane_id=1,
        )
        tags = self.tagger.tag_object_scenarios([], ego)
        assert isinstance(tags, list)


class TestWeatherClassifier:
    """Test weather classification from sensor data."""

    def setup_method(self):
        """Create classifier instance."""
        self.classifier = WeatherClassifier()

    def test_classify_from_camera_bright(self):
        """Bright image should suggest clear weather."""
        # Bright image (high mean intensity)
        image = np.random.randint(150, 255, (480, 640, 3), dtype=np.uint8)
        result = self.classifier.classify_from_camera(image)
        assert isinstance(result, dict)
        assert "clear" in result or "overcast" in result
        # All probabilities should sum roughly to 1
        total = sum(result.values())
        assert 0.9 <= total <= 1.1

    def test_classify_from_camera_dark(self):
        """Dark image should suggest night or fog."""
        image = np.random.randint(0, 40, (480, 640, 3), dtype=np.uint8)
        result = self.classifier.classify_from_camera(image)
        assert isinstance(result, dict)

    def test_classify_from_lidar_normal(self):
        """Normal point cloud should suggest clear weather."""
        # Dense point cloud (clear weather)
        points = np.random.randn(60000, 4).astype(np.float32)
        points[:, :3] *= 50
        result = self.classifier.classify_from_lidar(points)
        assert isinstance(result, dict)
        assert "clear" in result

    def test_classify_from_lidar_sparse(self):
        """Sparse point cloud should suggest degraded weather."""
        # Very sparse (fog/rain reduces range)
        points = np.random.randn(5000, 4).astype(np.float32)
        points[:, :3] *= 15  # Short range only
        result = self.classifier.classify_from_lidar(points)
        assert isinstance(result, dict)

    def test_fusion(self):
        """Sensor fusion should produce valid combined result."""
        camera_result = {"clear": 0.7, "rain": 0.2, "fog": 0.1}
        lidar_result = {"clear": 0.8, "rain": 0.1, "fog": 0.1}
        radar_result = {"clear": 0.6, "rain": 0.3, "fog": 0.1}

        fused = self.classifier.fuse_classifications(camera_result, lidar_result, radar_result)
        assert isinstance(fused, dict)
        assert "clear" in fused
        total = sum(fused.values())
        assert 0.9 <= total <= 1.1


class TestRoadTopologyExtractor:
    """Test road topology extraction."""

    def setup_method(self):
        """Create extractor instance."""
        self.extractor = RoadTopologyExtractor()

    def test_extract_lane_count(self):
        """Should correctly count lanes."""
        # 3 parallel lanes
        lanes = [
            np.array([[0, -3.5], [50, -3.5], [100, -3.5]]),
            np.array([[0, 0], [50, 0], [100, 0]]),
            np.array([[0, 3.5], [50, 3.5], [100, 3.5]]),
        ]
        result = self.extractor.extract_from_lanes(lanes)
        assert "lane_count" in result
        assert result["lane_count"] == 3

    def test_compute_road_geometry_straight(self):
        """Straight road should have low curvature."""
        centerline = np.array([[i, 0.0] for i in range(100)], dtype=np.float64)
        geometry = self.extractor.compute_road_geometry(centerline)
        assert "mean_curvature" in geometry
        assert geometry["mean_curvature"] < 0.01

    def test_compute_road_geometry_curved(self):
        """Curved road should have higher curvature."""
        t = np.linspace(0, np.pi / 2, 100)
        centerline = np.column_stack([np.cos(t) * 50, np.sin(t) * 50])
        geometry = self.extractor.compute_road_geometry(centerline)
        assert geometry["mean_curvature"] > 0.01


class TestTemporalEventDetector:
    """Test temporal event detection."""

    def setup_method(self):
        """Create detector instance."""
        self.detector = TemporalEventDetector()

    def test_detect_hard_braking(self):
        """Should detect hard braking events."""
        # Ego decelerating sharply
        ego_states = []
        for i in range(50):
            t = i * 0.1  # 10 Hz
            if i < 20:
                vel = 20.0
                acc = 0.0
            else:
                vel = max(0, 20.0 - 5.0 * (i - 20) * 0.1)
                acc = -5.0  # Hard braking
            ego_states.append(EgoState(
                position=np.array([vel * t, 0.0, 0.0]),
                velocity=np.array([vel, 0.0, 0.0]),
                acceleration=np.array([acc, 0.0, 0.0]),
                heading=0.0,
                timestamp=t,
                lane_id=1,
            ))

        events = self.detector.detect_hard_braking(ego_states)
        assert len(events) > 0
        assert all(e.event_type == "hard_braking" for e in events)

    def test_detect_lane_change(self):
        """Should detect ego lane change."""
        ego_states = []
        for i in range(60):
            t = i * 0.1
            if i < 20:
                y = 0.0
            elif i < 40:
                y = 3.5 * (i - 20) / 20.0  # Moving laterally
            else:
                y = 3.5
            ego_states.append(EgoState(
                position=np.array([15.0 * t, y, 0.0]),
                velocity=np.array([15.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                heading=0.0,
                timestamp=t,
                lane_id=1 if y < 1.75 else 2,
            ))

        events = self.detector.detect_lane_change(ego_states)
        assert len(events) > 0
        assert all(e.event_type == "lane_change" for e in events)

    def test_no_events_in_steady_driving(self):
        """Steady driving should produce no events."""
        ego_states = []
        for i in range(50):
            t = i * 0.1
            ego_states.append(EgoState(
                position=np.array([15.0 * t, 0.0, 0.0]),
                velocity=np.array([15.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                heading=0.0,
                timestamp=t,
                lane_id=1,
            ))

        braking_events = self.detector.detect_hard_braking(ego_states)
        lane_change_events = self.detector.detect_lane_change(ego_states)
        assert len(braking_events) == 0
        assert len(lane_change_events) == 0


class TestAnnotationPipeline:
    """Test the end-to-end annotation pipeline."""

    def setup_method(self):
        """Create pipeline instance."""
        self.pipeline = AnnotationPipeline(use_clip=False, confidence_threshold=0.3)

    def test_process_frame(self):
        """Should process a single frame and return tags."""
        image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        points = np.random.randn(30000, 4).astype(np.float32)
        points[:, :3] *= 40

        detections = [
            DetectedObject(
                class_name="car",
                bbox=np.array([200, 200, 280, 260]),
                track_id=1,
                velocity=np.array([10.0, 0.0, 0.0]),
                position_3d=np.array([20.0, 0.0, 0.0]),
            ),
        ]
        ego = EgoState(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([15.0, 0.0, 0.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            heading=0.0,
            timestamp=0.0,
            lane_id=1,
        )

        frame = FrameData(
            image=image,
            points=points,
            detections=detections,
            ego_state=ego,
            timestamp=0.0,
        )

        tags = self.pipeline.process_frame(frame)
        assert isinstance(tags, list)
        assert len(tags) > 0
        # All tags should have valid structure
        for tag in tags:
            assert hasattr(tag, "node_id")
            assert hasattr(tag, "confidence")
            assert 0 <= tag.confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
