"""Tests for the scenario_manager query engine."""

import pytest
import numpy as np

from scenario_trees.scenario_manager.database import ScenarioDatabase
from scenario_trees.scenario_manager.query_engine import ScenarioQueryEngine
from scenario_trees.scenario_manager.split_generator import SplitGenerator
from scenario_trees.taxonomy.scenario_schema import ScenarioTag, ScenarioQuery


@pytest.fixture
def populated_db():
    """Create and populate an in-memory database for testing."""
    db = ScenarioDatabase(":memory:")

    # Add recordings with known tags
    recordings = [
        ("rec_001", "/data/rec_001.bag", "2024-01-01T10:00:00", 60.0, "Munich"),
        ("rec_002", "/data/rec_002.bag", "2024-01-02T14:00:00", 90.0, "Munich"),
        ("rec_003", "/data/rec_003.bag", "2024-01-03T22:00:00", 45.0, "Berlin"),
        ("rec_004", "/data/rec_004.bag", "2024-02-01T08:00:00", 120.0, "Berlin"),
        ("rec_005", "/data/rec_005.bag", "2024-02-15T16:00:00", 75.0, "Stuttgart"),
        ("rec_006", "/data/rec_006.bag", "2024-03-01T12:00:00", 55.0, "Stuttgart"),
        ("rec_007", "/data/rec_007.bag", "2024-03-10T06:00:00", 80.0, "Hamburg"),
        ("rec_008", "/data/rec_008.bag", "2024-03-20T20:00:00", 100.0, "Hamburg"),
    ]

    for rec_id, path, ts, dur, loc in recordings:
        db.add_recording(rec_id, path, ts, dur, loc)

    # Add tags
    tag_assignments = {
        "rec_001": [("L1.highway", 0.95), ("L5.weather.clear", 0.9), ("L5.lighting.daylight", 0.92)],
        "rec_002": [("L1.urban", 0.88), ("L5.weather.rain", 0.75), ("L5.lighting.daylight", 0.85)],
        "rec_003": [("L1.highway", 0.9), ("L5.weather.clear", 0.85), ("L5.lighting.night", 0.88)],
        "rec_004": [("L1.urban", 0.82), ("L5.weather.fog", 0.7), ("L5.lighting.dawn", 0.8), ("L4.pedestrian.adult", 0.75)],
        "rec_005": [("L1.rural", 0.9), ("L5.weather.clear", 0.95), ("L5.lighting.daylight", 0.9), ("L4.vehicle.car", 0.85)],
        "rec_006": [("L1.intersection.crossroads", 0.85), ("L5.weather.rain", 0.8), ("L5.lighting.daylight", 0.87), ("L4.pedestrian.adult", 0.9)],
        "rec_007": [("L1.highway", 0.92), ("L5.weather.snow", 0.65), ("L5.lighting.dawn", 0.75)],
        "rec_008": [("L1.urban", 0.87), ("L5.weather.rain", 0.82), ("L5.lighting.night", 0.9), ("L4.behavior.cut_in", 0.7)],
    }

    for rec_id, tag_list in tag_assignments.items():
        tags = [ScenarioTag(node_id=nid, confidence=conf, source="auto") for nid, conf in tag_list]
        db.add_tags(rec_id, tags)

    return db


class TestScenarioDatabase:
    """Test database operations."""

    def test_create_database(self):
        """Should create database without errors."""
        db = ScenarioDatabase(":memory:")
        assert db is not None

    def test_add_and_get_recording(self, populated_db):
        """Should store and retrieve recordings."""
        rec = populated_db.get_recording("rec_001")
        assert rec is not None
        assert rec["path"] == "/data/rec_001.bag"
        assert rec["location"] == "Munich"

    def test_get_nonexistent_recording(self, populated_db):
        """Should return None for non-existent recording."""
        rec = populated_db.get_recording("nonexistent")
        assert rec is None

    def test_get_tags_for_recording(self, populated_db):
        """Should retrieve tags for a recording."""
        tags = populated_db.get_tags_for_recording("rec_001")
        assert len(tags) == 3
        node_ids = [t["node_id"] for t in tags]
        assert "L1.highway" in node_ids

    def test_get_recordings_with_tag(self, populated_db):
        """Should find recordings with a specific tag."""
        highway_recs = populated_db.get_recordings_with_tag("L1.highway", min_confidence=0.8)
        assert "rec_001" in highway_recs
        assert "rec_003" in highway_recs
        assert "rec_007" in highway_recs
        assert "rec_002" not in highway_recs  # urban, not highway

    def test_get_all_recordings(self, populated_db):
        """Should return all recordings."""
        all_recs = populated_db.get_all_recordings()
        assert len(all_recs) == 8

    def test_statistics(self, populated_db):
        """Should compute valid statistics."""
        stats = populated_db.get_statistics()
        assert "total_recordings" in stats
        assert stats["total_recordings"] == 8
        assert "tag_counts" in stats

    def test_search(self, populated_db):
        """Should search recordings by text."""
        results = populated_db.search("Munich")
        assert len(results) == 2
        rec_ids = [r["id"] for r in results]
        assert "rec_001" in rec_ids
        assert "rec_002" in rec_ids


class TestQueryEngine:
    """Test the query engine."""

    def test_simple_query(self, populated_db):
        """Should find recordings matching required tags."""
        engine = ScenarioQueryEngine(populated_db)
        query = ScenarioQuery(
            required_tags=["L1.highway"],
            min_confidence=0.8,
        )
        results = engine.query(query)
        assert "rec_001" in results
        assert "rec_003" in results
        assert "rec_007" in results

    def test_multi_tag_query(self, populated_db):
        """Should find recordings matching ALL required tags."""
        engine = ScenarioQueryEngine(populated_db)
        query = ScenarioQuery(
            required_tags=["L5.weather.rain", "L5.lighting.daylight"],
            min_confidence=0.5,
        )
        results = engine.query(query)
        assert "rec_002" in results
        assert "rec_006" in results
        # rec_008 has rain but night, not daylight
        assert "rec_008" not in results

    def test_excluded_tags(self, populated_db):
        """Should exclude recordings with excluded tags."""
        engine = ScenarioQueryEngine(populated_db)
        query = ScenarioQuery(
            required_tags=["L5.weather.rain"],
            excluded_tags=["L5.lighting.night"],
            min_confidence=0.5,
        )
        results = engine.query(query)
        assert "rec_002" in results
        assert "rec_006" in results
        assert "rec_008" not in results  # Has night lighting

    def test_confidence_threshold(self, populated_db):
        """Should respect confidence threshold."""
        engine = ScenarioQueryEngine(populated_db)
        # Snow in rec_007 has confidence 0.65
        query_low = ScenarioQuery(required_tags=["L5.weather.snow"], min_confidence=0.5)
        query_high = ScenarioQuery(required_tags=["L5.weather.snow"], min_confidence=0.9)

        results_low = engine.query(query_low)
        results_high = engine.query(query_high)

        assert "rec_007" in results_low
        assert "rec_007" not in results_high

    def test_composite_query_and(self, populated_db):
        """Should parse AND expressions."""
        engine = ScenarioQueryEngine(populated_db)
        results = engine.query_composite("L1.highway AND L5.weather.clear")
        assert "rec_001" in results
        assert "rec_003" in results

    def test_composite_query_or(self, populated_db):
        """Should parse OR expressions."""
        engine = ScenarioQueryEngine(populated_db)
        results = engine.query_composite("L5.weather.snow OR L5.weather.fog")
        assert "rec_007" in results  # snow
        assert "rec_004" in results  # fog

    def test_composite_query_not(self, populated_db):
        """Should parse NOT expressions."""
        engine = ScenarioQueryEngine(populated_db)
        results = engine.query_composite("L1.highway AND NOT L5.lighting.night")
        assert "rec_001" in results  # highway + daylight
        assert "rec_003" not in results  # highway + night

    def test_count_by_attribute(self, populated_db):
        """Should count recordings by attribute."""
        engine = ScenarioQueryEngine(populated_db)
        counts = engine.count_by_attribute("node_id")
        assert isinstance(counts, dict)
        assert sum(counts.values()) > 0
        # Should include our known tags
        assert "L1.highway" in counts


class TestSplitGenerator:
    """Test split generation."""

    def test_basic_split(self, populated_db):
        """Should generate non-overlapping splits."""
        generator = SplitGenerator()
        split = generator.generate_balanced_split(
            database=populated_db,
            ratios={"train": 0.6, "val": 0.2, "test": 0.2},
            seed=42,
        )

        assert "train" in split
        assert "val" in split
        assert "test" in split

        # Check no overlap
        all_ids = []
        for ids in split.values():
            all_ids.extend(ids)
        assert len(all_ids) == len(set(all_ids))

        # Check all recordings assigned
        assert len(all_ids) == 8

    def test_split_ratios(self, populated_db):
        """Should approximately respect requested ratios."""
        generator = SplitGenerator()
        split = generator.generate_balanced_split(
            database=populated_db,
            ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            seed=42,
        )

        total = sum(len(v) for v in split.values())
        # With 8 recordings, ratios won't be exact but train should be largest
        assert len(split["train"]) >= len(split["val"])
        assert len(split["train"]) >= len(split["test"])

    def test_validate_split(self, populated_db):
        """Should validate split properties."""
        generator = SplitGenerator()
        split = generator.generate_balanced_split(
            database=populated_db,
            ratios={"train": 0.6, "val": 0.2, "test": 0.2},
            seed=42,
        )
        validation = generator.validate_split(split, populated_db)
        assert validation["no_overlap"] is True
        assert validation["full_coverage"] is True

    def test_assign_splits(self, populated_db):
        """Should store split assignments in database."""
        generator = SplitGenerator()
        split = generator.generate_balanced_split(
            database=populated_db,
            ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            seed=42,
        )
        generator.assign_splits(populated_db, split, version="v1.0")

        # Verify assignments were stored
        stats = populated_db.get_statistics()
        assert "splits" in stats or "split_counts" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
