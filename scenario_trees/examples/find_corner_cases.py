"""
Example: Discover rare/corner-case scenarios in the dataset.

This example demonstrates how to use the data mining module to find
unusual scenarios that may be underrepresented in training data.
"""

import numpy as np
from typing import List

from scenario_trees.data_mining.embedding_store import EmbeddingStore
from scenario_trees.data_mining.novelty_detector import NoveltyDetector
from scenario_trees.data_mining.coverage_analyzer import CoverageAnalyzer
from scenario_trees.data_mining.difficulty_scorer import DifficultyScorer
from scenario_trees.taxonomy.scenario_schema import ScenarioAnnotation, ScenarioTag


def generate_synthetic_dataset(n_recordings: int = 200) -> List[ScenarioAnnotation]:
    """Generate a synthetic dataset with known rare scenarios."""
    annotations = []
    rng = np.random.default_rng(42)

    # Common scenarios (80% of data)
    common_tags_pool = [
        ("L1.highway", 0.9),
        ("L1.urban", 0.85),
        ("L4.vehicle.car", 0.95),
        ("L5.weather.clear", 0.9),
        ("L5.lighting.daylight", 0.92),
        ("L5.surface.dry", 0.88),
    ]

    # Rare scenarios (should be flagged as corner cases)
    rare_tags_pool = [
        ("L5.weather.fog", 0.7),
        ("L5.weather.snow", 0.65),
        ("L5.lighting.night", 0.8),
        ("L4.behavior.jaywalking", 0.6),
        ("L4.pedestrian.child", 0.75),
        ("L3.construction.lane_closure", 0.7),
        ("L6.sensor.lidar_blocked", 0.5),
    ]

    for i in range(n_recordings):
        tags = []

        if i < int(n_recordings * 0.8):
            # Common scenario: pick 3-5 common tags
            n_tags = rng.integers(3, 6)
            selected = rng.choice(len(common_tags_pool), size=min(n_tags, len(common_tags_pool)), replace=False)
            for idx in selected:
                node_id, conf = common_tags_pool[idx]
                tags.append(ScenarioTag(
                    node_id=node_id,
                    confidence=conf + rng.normal(0, 0.05),
                    source="auto",
                ))
        else:
            # Rare scenario: pick 2-3 rare tags + 1-2 common
            n_rare = rng.integers(2, 4)
            n_common = rng.integers(1, 3)
            rare_selected = rng.choice(len(rare_tags_pool), size=min(n_rare, len(rare_tags_pool)), replace=False)
            common_selected = rng.choice(len(common_tags_pool), size=min(n_common, len(common_tags_pool)), replace=False)

            for idx in rare_selected:
                node_id, conf = rare_tags_pool[idx]
                tags.append(ScenarioTag(
                    node_id=node_id,
                    confidence=conf + rng.normal(0, 0.05),
                    source="auto",
                ))
            for idx in common_selected:
                node_id, conf = common_tags_pool[idx]
                tags.append(ScenarioTag(
                    node_id=node_id,
                    confidence=conf + rng.normal(0, 0.05),
                    source="auto",
                ))

        annotations.append(ScenarioAnnotation(
            recording_id=f"recording_{i:04d}",
            timestamp=f"2024-03-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00",
            tags=tags,
            metadata={"synthetic": True, "index": i},
        ))

    return annotations


def main():
    """Demonstrate corner case discovery."""
    print("=" * 70)
    print("Functional Scenario Trees - Corner Case Discovery Example")
    print("=" * 70)

    # 1. Generate synthetic dataset
    print("\n--- Generating Synthetic Dataset ---\n")
    annotations = generate_synthetic_dataset(200)
    print(f"Generated {len(annotations)} synthetic recordings")

    # 2. Create embeddings (synthetic for demo)
    print("\n--- Building Embedding Store ---\n")
    rng = np.random.default_rng(42)
    embedding_dim = 512
    store = EmbeddingStore(embedding_dim=embedding_dim)

    for ann in annotations:
        # Create a pseudo-embedding from tags (in practice, use CLIP embeddings)
        embedding = rng.normal(0, 1, embedding_dim).astype(np.float32)
        # Make rare scenarios have distinctive embeddings
        if any("fog" in t.node_id or "snow" in t.node_id for t in ann.tags):
            embedding[:50] += 3.0  # Shift rare embeddings
        store.add(ann.recording_id, embedding, {"n_tags": len(ann.tags)})

    print(f"Stored {len(store.recording_ids)} embeddings (dim={embedding_dim})")

    # 3. Novelty detection
    print("\n--- Running Novelty Detection ---\n")
    detector = NoveltyDetector()

    all_embeddings = store.get_all_embeddings()
    tag_dicts = [{"tags": [t.node_id for t in ann.tags]} for ann in annotations]
    detector.fit(all_embeddings, tag_dicts)

    # Score each recording
    novelty_scores = []
    for i, ann in enumerate(annotations):
        score = detector.score_novelty(all_embeddings[i], tag_dicts[i])
        novelty_scores.append((ann.recording_id, score))

    novelty_scores.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 most novel/unusual recordings:")
    for rec_id, score in novelty_scores[:10]:
        ann = next(a for a in annotations if a.recording_id == rec_id)
        tag_names = [t.node_id for t in ann.tags]
        print(f"  {rec_id}: novelty={score:.3f} | tags={tag_names}")

    # 4. Coverage analysis
    print("\n--- Coverage Analysis ---\n")
    analyzer = CoverageAnalyzer()

    layer_coverage = analyzer.compute_layer_coverage(annotations, layer=5)
    print("Layer 5 (Environment) coverage:")
    for node_id, count in sorted(layer_coverage.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * min(count, 50)
        print(f"  {node_id:30s}: {count:4d} {bar}")

    # 5. Find gaps
    print("\n--- Coverage Gaps ---\n")
    gaps = analyzer.find_gaps(annotations, required_attributes=["L5.weather", "L5.lighting"])
    if gaps:
        print(f"Found {len(gaps)} coverage gaps:")
        for gap in gaps[:5]:
            print(f"  - Missing: {gap.missing_combination} (priority: {gap.priority})")
    else:
        print("  No gaps found in the specified attributes.")

    # 6. Difficulty scoring
    print("\n--- Difficulty Scoring ---\n")
    scorer = DifficultyScorer()

    difficulty_scores = scorer.score_batch(annotations)
    difficulty_scores.sort(key=lambda x: x[1], reverse=True)

    print("Top 10 most difficult recordings:")
    for rec_id, score in difficulty_scores[:10]:
        ann = next(a for a in annotations if a.recording_id == rec_id)
        tag_names = [t.node_id for t in ann.tags]
        print(f"  {rec_id}: difficulty={score:.3f} | tags={tag_names}")

    # Distribution
    distribution = scorer.get_difficulty_distribution(annotations)
    print("\nDifficulty distribution:")
    for bucket, count in distribution.items():
        bar = "#" * count
        print(f"  {bucket:10s}: {count:4d} {bar}")

    print("\n" + "=" * 70)
    print("Corner case discovery complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
