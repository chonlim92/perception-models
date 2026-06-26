"""
Example: Generate balanced train/val/test splits.

This example demonstrates how to create scenario-balanced dataset splits
that ensure proportional representation of all scenario attributes.
"""

import numpy as np
from typing import List

from scenario_trees.scenario_manager.database import ScenarioDatabase
from scenario_trees.scenario_manager.split_generator import SplitGenerator
from scenario_trees.scenario_manager.dashboard import ScenarioDashboard
from scenario_trees.scenario_manager.query_engine import ScenarioQueryEngine
from scenario_trees.taxonomy.scenario_schema import ScenarioAnnotation, ScenarioTag


def populate_database(db: ScenarioDatabase, n_recordings: int = 300):
    """Populate database with synthetic recordings."""
    rng = np.random.default_rng(42)

    road_types = ["L1.highway", "L1.urban", "L1.rural", "L1.intersection.crossroads"]
    weather_types = ["L5.weather.clear", "L5.weather.rain", "L5.weather.fog", "L5.weather.snow"]
    lighting_types = ["L5.lighting.daylight", "L5.lighting.night", "L5.lighting.dawn"]
    object_types = ["L4.vehicle.car", "L4.vehicle.truck", "L4.pedestrian.adult", "L4.cyclist"]

    locations = ["Munich", "Stuttgart", "Berlin", "Hamburg", "Frankfurt"]

    for i in range(n_recordings):
        recording_id = f"rec_{i:04d}"
        location = locations[i % len(locations)]

        # Add recording
        db.add_recording(
            recording_id=recording_id,
            path=f"/data/recordings/{location.lower()}/{recording_id}.bag",
            timestamp=f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00",
            duration=rng.uniform(30, 300),
            location=location,
        )

        # Generate tags with realistic distribution
        tags = []

        # Road type (pick 1)
        road_weights = [0.4, 0.35, 0.15, 0.1]
        road = rng.choice(road_types, p=road_weights)
        tags.append(ScenarioTag(node_id=road, confidence=rng.uniform(0.7, 0.99), source="auto"))

        # Weather (pick 1, biased toward clear)
        weather_weights = [0.6, 0.2, 0.1, 0.1]
        weather = rng.choice(weather_types, p=weather_weights)
        tags.append(ScenarioTag(node_id=weather, confidence=rng.uniform(0.6, 0.95), source="auto"))

        # Lighting (pick 1)
        lighting_weights = [0.6, 0.25, 0.15]
        lighting = rng.choice(lighting_types, p=lighting_weights)
        tags.append(ScenarioTag(node_id=lighting, confidence=rng.uniform(0.7, 0.95), source="auto"))

        # Objects (pick 1-3)
        n_objects = rng.integers(1, 4)
        objects = rng.choice(object_types, size=n_objects, replace=False)
        for obj in objects:
            tags.append(ScenarioTag(node_id=obj, confidence=rng.uniform(0.5, 0.95), source="auto"))

        # Add tags to database
        db.add_tags(recording_id, tags)

    return n_recordings


def main():
    """Demonstrate balanced split generation."""
    print("=" * 70)
    print("Functional Scenario Trees - Balanced Split Generation Example")
    print("=" * 70)

    # 1. Create and populate database (in-memory for demo)
    print("\n--- Setting Up Database ---\n")
    db = ScenarioDatabase(":memory:")
    n_recordings = populate_database(db)
    print(f"Populated database with {n_recordings} recordings")

    # 2. Show initial statistics
    print("\n--- Dataset Statistics ---\n")
    dashboard = ScenarioDashboard(db)
    print(dashboard.render_statistics())

    # 3. Generate balanced split
    print("\n--- Generating Balanced Split ---\n")
    generator = SplitGenerator()

    split = generator.generate_balanced_split(
        database=db,
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        seed=42,
    )

    print(f"Split sizes:")
    for split_name, rec_ids in split.items():
        print(f"  {split_name:6s}: {len(rec_ids)} recordings")

    # 4. Validate the split
    print("\n--- Split Validation ---\n")
    validation = generator.validate_split(split, db)
    print(f"Overlap check: {'PASS' if validation.get('no_overlap', False) else 'FAIL'}")
    print(f"Coverage check: {'PASS' if validation.get('full_coverage', False) else 'FAIL'}")
    if "balance_scores" in validation:
        print(f"Balance scores by attribute:")
        for attr, score in list(validation["balance_scores"].items())[:5]:
            print(f"  {attr:30s}: {score:.3f}")

    # 5. Assign splits to database
    print("\n--- Assigning Splits ---\n")
    generator.assign_splits(db, split, version="v1.0")
    print("Splits assigned to database with version 'v1.0'")

    # 6. Show split composition
    print("\n--- Split Composition ---\n")
    print(dashboard.render_split_composition("v1.0"))

    # 7. Query within a split
    print("\n--- Querying Within Splits ---\n")
    engine = ScenarioQueryEngine(db)

    # Find rainy night recordings in test set
    from scenario_trees.taxonomy.scenario_schema import ScenarioQuery
    query = ScenarioQuery(
        required_tags=["L5.weather.rain", "L5.lighting.night"],
        min_confidence=0.5,
    )
    results = engine.query(query)
    test_results = [r for r in results if r in split.get("test", [])]
    print(f"Rainy night recordings in test set: {len(test_results)}")
    for rec_id in test_results[:5]:
        print(f"  - {rec_id}")

    print("\n" + "=" * 70)
    print("Balanced split generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
