# Functional Scenario Trees — Automated Data Management for Autonomous Driving

## Overview

A comprehensive scenario-based data management system for autonomous driving recordings. It automatically classifies, tags, mines, and organizes driving scenarios using a 6-layer hierarchical taxonomy derived from PEGASUS/ASAM standards.

**Key capabilities:**
- **Automated scenario tagging** — CLIP-based scene classification, weather detection, behavior analysis
- **Corner case discovery** — Find rare/unusual scenarios via novelty detection and clustering
- **Coverage analysis** — Identify gaps in your dataset (e.g., "no night + rain + pedestrian scenarios")
- **Balanced splits** — Generate train/val/test splits that cover all scenario combinations
- **Query interface** — Search recordings by complex scenario criteria

## Architecture

```
Recording (cameras + LiDAR + radar + ego)
           ↓
┌──────────────────────────────────────┐
│  Auto-Annotation Pipeline             │
│  ├─ CLIP Scene Classifier            │ → road type, setting
│  ├─ Weather/Lighting Classifier      │ → weather, time of day
│  ├─ Object Behavior Detector         │ → cut-ins, lane changes
│  ├─ Road Topology Extractor          │ → lanes, intersections
│  └─ Temporal Event Detector          │ → hard braking, near-miss
└──────────────────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│  Scenario Tag Vector                  │
│  [highway, 3-lane, rain, night,      │
│   2-vehicles, 1-pedestrian, cut-in]  │
└──────────────────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│  Scenario Manager (Database)          │
│  ├─ Query Engine                     │ → Find recordings by criteria
│  ├─ Coverage Analyzer                │ → What's missing?
│  ├─ Split Generator                  │ → Balanced train/val/test
│  └─ Data Mining                      │ → Corner cases, novelty
└──────────────────────────────────────┘
```

## 6-Layer Scenario Taxonomy

| Layer | Category | Examples |
|-------|----------|----------|
| L1 | Road Topology | highway, urban, rural, intersection, roundabout, lane count |
| L2 | Traffic Infrastructure | traffic lights, signs, markings, barriers, guardrails |
| L3 | Temporary Modifications | construction zones, temp signs, road closures |
| L4 | Dynamic Objects | vehicles (car/truck/bus), pedestrians, cyclists, behaviors |
| L5 | Environment | weather (rain/snow/fog), lighting (day/night/dusk), road surface |
| L6 | Digital Information | sensor degradation, map accuracy, V2X signals |

## Quick Start

### 1. Install

```bash
cd scenario_trees
pip install -r requirements.txt
```

### 2. Auto-Tag a Recording

```python
from scenario_trees.auto_annotation.annotation_pipeline import AnnotationPipeline

pipeline = AnnotationPipeline()
tags = pipeline.annotate_recording(
    recording_path="/path/to/nuscenes/scene-0001",
    camera_images=True,
    lidar_points=True,
)
print(tags)
# ScenarioTags(road_type='urban', lanes=2, weather='clear', 
#              lighting='day', objects={'car': 3, 'pedestrian': 1},
#              behaviors=['cut_in'], ...)
```

### 3. Find Corner Cases

```python
from scenario_trees.data_mining.novelty_detector import NoveltyDetector

detector = NoveltyDetector()
detector.fit(all_recordings_embeddings)
corner_cases = detector.find_novel(threshold=0.8)
print(f"Found {len(corner_cases)} unusual scenarios")
```

### 4. Coverage Analysis

```python
from scenario_trees.data_mining.coverage_analyzer import CoverageAnalyzer

analyzer = CoverageAnalyzer(database)
gaps = analyzer.find_gaps(
    dimensions=["weather", "lighting", "road_type", "has_pedestrian"]
)
for gap in gaps:
    print(f"MISSING: {gap}")
# MISSING: weather=rain, lighting=night, road_type=highway, has_pedestrian=True
```

### 5. Generate Balanced Split

```python
from scenario_trees.scenario_manager.split_generator import SplitGenerator

generator = SplitGenerator(database)
splits = generator.generate(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    balance_on=["weather", "road_type", "lighting"],
)
```

### 6. Query Recordings

```python
from scenario_trees.scenario_manager.query_engine import QueryEngine

engine = QueryEngine(database)
results = engine.query(
    weather="rain",
    lighting="night",
    min_objects=3,
    has_behavior="cut_in",
)
print(f"Found {len(results)} matching recordings")
```

## Module Details

### auto_annotation/
| File | Description |
|------|-------------|
| `scene_classifier.py` | CLIP-based road type and setting classification |
| `weather_classifier.py` | Multi-modal weather detection (camera histograms, LiDAR density, radar clutter) |
| `object_detector.py` | Object counting, behavior analysis from trajectories |
| `road_topology_extractor.py` | Lane count, intersection detection from map predictions |
| `temporal_event_detector.py` | Cut-in, hard brake, lane change, near-miss detection |
| `annotation_pipeline.py` | End-to-end pipeline combining all classifiers |

### data_mining/
| File | Description |
|------|-------------|
| `embedding_store.py` | FAISS/numpy-based embedding storage and similarity search |
| `novelty_detector.py` | Isolation Forest + embedding density for rare scenario discovery |
| `coverage_analyzer.py` | Cross-tabulation of scenario attributes, gap identification |
| `difficulty_scorer.py` | Scoring recordings by complexity for curriculum learning |
| `cluster_analysis.py` | HDBSCAN/K-means clustering with auto-labeling |

### scenario_manager/
| File | Description |
|------|-------------|
| `database.py` | SQLAlchemy-based schema for recordings, tags, splits, results |
| `query_engine.py` | Composite queries (AND/OR/NOT) on scenario attributes |
| `split_generator.py` | Stratified splitting balanced across scenario dimensions |
| `dashboard.py` | Text-based statistics dashboard |
| `export.py` | Export filtered subsets as file lists for training |

### taxonomy/
| File | Description |
|------|-------------|
| `scenario_tree.py` | Full 6-layer tree with 100+ nodes, programmatic construction |
| `scenario_schema.py` | Pydantic models for validation and serialization |
| `tree_visualization.py` | Text, HTML, and Graphviz tree rendering |

## Use Cases

1. **Training data selection** — Select balanced subset of scenarios for model training
2. **Evaluation coverage** — Ensure test set covers all critical scenario combinations
3. **Failure analysis** — Correlate model failures with scenario attributes
4. **Data collection planning** — Identify which scenarios need more recordings
5. **Regression testing** — Tag which scenarios are affected by model changes
6. **Safety validation** — Verify coverage of safety-critical scenarios (SOTIF)

## Dependencies

- `transformers` — CLIP model for scene embedding
- `sentence-transformers` — Efficient embedding computation
- `scikit-learn` — Clustering, novelty detection
- `hdbscan` — Density-based clustering
- `sqlalchemy` — Database ORM
- `pydantic` — Data validation
- `numpy`, `scipy` — Numerical operations
