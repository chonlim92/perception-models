"""Auto-annotation module for Functional Scenario Trees.

Provides automated scene understanding and annotation capabilities for
autonomous driving scenarios. This module integrates computer vision models
and rule-based classifiers to extract structured scenario information from
sensor data, including scene classification, object detection with scenario
tagging, weather classification, road topology extraction, temporal event
detection, and a unified annotation pipeline.
"""

from .scene_classifier import CLIPSceneClassifier
from .object_detector import ObjectScenarioTagger, DetectedObject, EgoState
from .weather_classifier import WeatherClassifier
from .road_topology_extractor import RoadTopologyExtractor
from .temporal_event_detector import TemporalEventDetector, Track, Event
from .annotation_pipeline import AnnotationPipeline, FrameData

__all__ = [
    "CLIPSceneClassifier",
    "ObjectScenarioTagger",
    "DetectedObject",
    "EgoState",
    "WeatherClassifier",
    "RoadTopologyExtractor",
    "TemporalEventDetector",
    "Track",
    "Event",
    "AnnotationPipeline",
    "FrameData",
]
