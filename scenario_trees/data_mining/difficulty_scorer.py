"""
Difficulty scoring for autonomous driving recordings.

Scores recordings from 0 (easy) to 1 (hard) based on multiple factors:
dynamic objects, weather, lighting, object behaviors, and road complexity.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..taxonomy.scenario_schema import ScenarioAnnotation, ScenarioTag


# Weather severity mappings (semantic node_id -> score)
_WEATHER_SCORES: Dict[str, float] = {
    "L5.weather.clear": 0.0,
    "L5.weather.rain": 0.3,
    "L5.weather.heavy_rain": 0.6,
    "L5.weather.snow": 0.8,
    "L5.weather.fog": 0.7,
    "L5.weather.hail": 0.7,
}

# Lighting condition mappings
_LIGHTING_SCORES: Dict[str, float] = {
    "L5.lighting.daylight": 0.0,
    "L5.lighting.dawn": 0.3,
    "L5.lighting.dusk": 0.3,
    "L5.lighting.night": 0.6,
    "L5.lighting.tunnel_dark": 0.5,
}

# Object behavior difficulty scores
_BEHAVIOR_SCORES: Dict[str, float] = {
    "L4.behavior.cut_in": 0.5,
    "L4.behavior.lane_change": 0.3,
    "L4.behavior.jaywalking": 0.7,
    "L4.behavior.sudden_braking": 0.6,
    "L4.behavior.u_turn": 0.4,
    "L4.behavior.overtaking": 0.4,
}

# Road complexity scores
_ROAD_COMPLEXITY_SCORES: Dict[str, float] = {
    "L1.intersection.t_junction": 0.4,
    "L1.intersection.crossroads": 0.4,
    "L1.intersection.roundabout": 0.5,
    "L3.construction.lane_closure": 0.5,
    "L3.construction.detour": 0.5,
    "L3.construction.speed_reduction": 0.3,
    "L3.closure.full": 0.6,
    "L3.closure.partial": 0.5,
    "L3.event.accident": 0.7,
}

# Dynamic object type indicators (Layer 4 vehicles and pedestrians)
_DYNAMIC_OBJECT_PREFIXES = ("L4.vehicle.", "L4.pedestrian.")


class DifficultyScorer:
    """
    Score recordings by driving difficulty on a 0-to-1 scale.

    Combines multiple difficulty factors with configurable weights:
    - Dynamic object count
    - Weather severity
    - Lighting conditions
    - Object behaviors (cut-in, jaywalking, etc.)
    - Road complexity (intersections, construction zones)

    Parameters
    ----------
    weights : dict or None
        Custom weights for each factor. Keys: 'objects', 'weather',
        'lighting', 'behaviors', 'road_complexity'. Default: equal weighting.
    max_objects : int
        Number of dynamic objects that saturates the object count score.
        Default 15 (i.e., 15+ objects -> score 1.0 for that factor).
    """

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        max_objects: int = 15,
    ) -> None:
        self.weights = weights or {
            "objects": 0.20,
            "weather": 0.20,
            "lighting": 0.15,
            "behaviors": 0.25,
            "road_complexity": 0.20,
        }
        self.max_objects = max_objects

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def score_recording(self, annotation: ScenarioAnnotation) -> float:
        """
        Score a single recording's difficulty from 0 (easy) to 1 (hard).

        Parameters
        ----------
        annotation : ScenarioAnnotation
            The scenario annotation for this recording.

        Returns
        -------
        float
            Difficulty score between 0.0 and 1.0.
        """
        node_ids = {tag.node_id for tag in annotation.tags}

        # Factor 1: Dynamic object count
        object_score = self._score_objects(node_ids)

        # Factor 2: Weather severity
        weather_score = self._score_weather(node_ids)

        # Factor 3: Lighting conditions
        lighting_score = self._score_lighting(node_ids)

        # Factor 4: Object behaviors
        behavior_score = self._score_behaviors(node_ids)

        # Factor 5: Road complexity
        road_score = self._score_road_complexity(node_ids)

        # Weighted combination
        total = (
            self.weights["objects"] * object_score
            + self.weights["weather"] * weather_score
            + self.weights["lighting"] * lighting_score
            + self.weights["behaviors"] * behavior_score
            + self.weights["road_complexity"] * road_score
        )

        return min(1.0, max(0.0, total))

    def _score_objects(self, node_ids: set) -> float:
        """Score based on number of dynamic object types present."""
        count = sum(
            1 for nid in node_ids
            if any(nid.startswith(prefix) for prefix in _DYNAMIC_OBJECT_PREFIXES)
        )
        # Also check metadata for actual object counts if available
        # For now, use type count as proxy, scaled to max_objects
        return min(1.0, count / self.max_objects * 3.0)

    def _score_weather(self, node_ids: set) -> float:
        """Score based on worst weather condition present."""
        scores = [
            _WEATHER_SCORES[nid]
            for nid in node_ids
            if nid in _WEATHER_SCORES
        ]
        return max(scores) if scores else 0.0

    def _score_lighting(self, node_ids: set) -> float:
        """Score based on worst lighting condition present."""
        scores = [
            _LIGHTING_SCORES[nid]
            for nid in node_ids
            if nid in _LIGHTING_SCORES
        ]
        return max(scores) if scores else 0.0

    def _score_behaviors(self, node_ids: set) -> float:
        """
        Score based on challenging behaviors present.

        Uses max + average to reward both having the hardest behavior
        and having multiple challenging behaviors simultaneously.
        """
        scores = [
            _BEHAVIOR_SCORES[nid]
            for nid in node_ids
            if nid in _BEHAVIOR_SCORES
        ]
        if not scores:
            return 0.0

        # Combine max with density: max captures worst case, mean captures volume
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)
        # Bonus for multiple behaviors (up to 0.2 extra)
        multi_bonus = min(0.2, (len(scores) - 1) * 0.1)

        return min(1.0, 0.6 * max_score + 0.3 * mean_score + multi_bonus)

    def _score_road_complexity(self, node_ids: set) -> float:
        """Score based on road infrastructure complexity."""
        scores = [
            _ROAD_COMPLEXITY_SCORES[nid]
            for nid in node_ids
            if nid in _ROAD_COMPLEXITY_SCORES
        ]
        if not scores:
            return 0.0

        # Take the maximum complexity factor, with small bonus for multiple factors
        max_score = max(scores)
        multi_bonus = min(0.15, (len(scores) - 1) * 0.1)
        return min(1.0, max_score + multi_bonus)

    def score_batch(
        self, annotations: List[ScenarioAnnotation]
    ) -> List[Tuple[str, float]]:
        """
        Score multiple recordings and return (recording_id, score) pairs.

        Parameters
        ----------
        annotations : list of ScenarioAnnotation
            List of annotations to score.

        Returns
        -------
        list of (str, float)
            Tuples of (recording_id, difficulty_score) sorted by score descending.
        """
        results = [
            (ann.recording_id, self.score_recording(ann))
            for ann in annotations
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_difficulty_distribution(
        self, annotations: List[ScenarioAnnotation]
    ) -> Dict[str, int]:
        """
        Bucket recordings into difficulty categories.

        Categories:
        - easy: score < 0.25
        - medium: 0.25 <= score < 0.50
        - hard: 0.50 <= score < 0.75
        - extreme: score >= 0.75

        Parameters
        ----------
        annotations : list of ScenarioAnnotation
            List of annotations to analyze.

        Returns
        -------
        dict
            Mapping from category name to count of recordings.
        """
        distribution: Dict[str, int] = {
            "easy": 0,
            "medium": 0,
            "hard": 0,
            "extreme": 0,
        }

        for ann in annotations:
            score = self.score_recording(ann)
            if score < 0.25:
                distribution["easy"] += 1
            elif score < 0.50:
                distribution["medium"] += 1
            elif score < 0.75:
                distribution["hard"] += 1
            else:
                distribution["extreme"] += 1

        return distribution

    def select_hard_examples(
        self,
        annotations: List[ScenarioAnnotation],
        top_k: int = 50,
    ) -> List[str]:
        """
        Select the hardest recordings for active learning or focused testing.

        Parameters
        ----------
        annotations : list of ScenarioAnnotation
            All available annotations.
        top_k : int
            Number of hard examples to select.

        Returns
        -------
        list of str
            Recording IDs of the top_k hardest recordings, sorted by
            difficulty (hardest first).
        """
        scored = self.score_batch(annotations)
        return [recording_id for recording_id, _ in scored[:top_k]]
