# Perception Models Repository — Design Specification

**Date:** 2026-06-25
**Author:** Staff AI Engineer (Autonomous Driving Perception)
**Status:** Approved

## 1. Purpose

A comprehensive, runnable repository of state-of-the-art deep learning perception models for autonomous driving. Covers camera, LiDAR, and radar modalities for both static map semantics (lane lines, road boundaries, pedestrian crossings) and dynamic object detection/tracking (vehicles, pedestrians, cyclists).

Additionally includes a Functional Scenario Trees system for automated annotation, data mining, and scenario-based test management.

## 2. Target Users

- Staff/Senior AI Engineers ramping up on perception algorithms
- Teams evaluating model architectures for production deployment
- Researchers benchmarking new approaches against established baselines

## 3. Architecture Overview

### 3.1 Repository Structure

```
perception-models/
├── common/                            # Shared utilities across all models
│   ├── datasets/                      # nuScenes, Waymo, KITTI data loaders
│   ├── metrics/                       # mAP, NDS, mIoU, Chamfer, AMOTA
│   ├── visualization/                 # BEV plots, 3D bbox, point clouds
│   ├── transforms/                    # Augmentations, coordinate systems
│   └── registry.py                    # Model/dataset/metric registry
│
├── camera/
│   ├── static_map_semantics/
│   │   ├── stream_mapnet/             # Temporal HD map construction
│   │   ├── maptr/                     # MapTR vectorized map prediction
│   │   └── hdmapnet/                  # Semantic map from surround cameras
│   └── dynamic_objects/
│       ├── bevformer/                 # BEV-based 3D detection (deformable attn)
│       ├── detr3d/                    # Set-prediction 3D detection
│       └── petr/                      # Position-embedding 3D detection
│
├── lidar/
│   ├── static_map_semantics/
│   │   ├── cylinder3d/               # Cylindrical point cloud segmentation
│   │   └── rangenet_pp/              # Range-image LiDAR segmentation
│   └── dynamic_objects/
│       ├── pointnet_pp/              # Hierarchical point set learning
│       ├── centerpoint/              # Center-based 3D detection + tracking
│       └── pointpillars/             # Fast pillar-based detection
│
├── radar/
│   ├── static_map_semantics/
│   │   └── radar_occupancy/          # Occupancy grid from radar
│   └── dynamic_objects/
│       ├── radar_pillarnet/          # Pillar-based radar 3D detection
│       └── craft/                    # Camera-Radar Fusion Transformer
│
└── scenario_trees/                    # Functional Scenario Trees
    ├── taxonomy/                      # 6-layer scenario classification
    ├── auto_annotation/               # CLIP + rule-based auto-tagging
    ├── data_mining/                   # Corner case discovery
    └── scenario_manager/              # Query/filter interface
```

### 3.2 Each Model Contains

```
<model_name>/
├── README.md                    # Complete guide (what, why, how)
├── docs/
│   ├── research_summary.md      # Paper analysis, key contributions
│   ├── data_collection.md       # Dataset requirements + download
│   ├── annotation_guide.md      # Label formats, tools
│   ├── model_architecture.md    # Layer-by-layer architecture
│   ├── training_guide.md        # Step-by-step training
│   └── evaluation_guide.md      # Metrics, benchmarks, temporal
├── configs/
│   └── *.yaml                   # Training/eval configurations
├── pytorch/
│   ├── model.py                 # Complete model definition
│   ├── backbone.py              # Feature extraction network
│   ├── heads.py                 # Task-specific heads
│   ├── losses.py                # Loss functions
│   ├── dataset.py               # Data pipeline
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation with all metrics
│   └── inference.py             # Single-sample demo
├── tensorflow/
│   ├── model.py                 # TF2/Keras equivalent
│   ├── train.py                 # Training
│   ├── evaluate.py              # Evaluation
│   └── inference.py             # Inference demo
├── scripts/
│   ├── download_data.sh         # Data download automation
│   ├── prepare_data.py          # Preprocessing
│   └── visualize_results.py     # Visualization
└── tests/
    └── test_model.py            # Forward-pass + metric tests
```

## 4. Models Selection Rationale

### 4.1 Camera — Static Map Semantics

| Model | Paper | Key Innovation | Year |
|-------|-------|----------------|------|
| StreamMapNet | "Streaming HD Map from Multi-sensor" | Temporal fusion across frames for stable map prediction | 2023 |
| MapTR/v2 | "MapTR: Structured Modeling for Online Vectorized HD Map" | Unified permutation-equivalent modeling for map elements | 2023 |
| HDMapNet | "HDMapNet: An Online HD Map Construction and Evaluation Framework" | First end-to-end surround-view to BEV semantic map | 2022 |

### 4.2 Camera — Dynamic Objects

| Model | Paper | Key Innovation | Year |
|-------|-------|----------------|------|
| BEVFormer | "BEVFormer: Learning BEV Representation via Spatiotemporal Transformers" | Deformable attention for BEV feature generation + temporal self-attention | 2022 |
| DETR3D | "DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries" | Back-projection of 3D reference points to 2D features | 2021 |
| PETR/StreamPETR | "PETR: Position Embedding Transformation for Multi-View 3D Object Detection" | 3D position-aware feature encoding, streaming temporal | 2022-2023 |

### 4.3 LiDAR — Static Map Semantics

| Model | Paper | Key Innovation | Year |
|-------|-------|----------------|------|
| Cylinder3D | "Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation" | Cylindrical partition preserving point distribution | 2021 |
| RangeNet++ | "RangeNet++: Fast and Accurate LiDAR Semantic Segmentation" | Range-image 2D CNN + KNN post-processing | 2019 |

### 4.4 LiDAR — Dynamic Objects

| Model | Paper | Key Innovation | Year |
|-------|-------|----------------|------|
| PointNet++ | "PointNet++: Deep Hierarchical Feature Learning on Point Sets" | Multi-scale grouping, hierarchical point learning | 2017 |
| CenterPoint | "Center-based 3D Object Detection and Tracking" | Center heatmap + two-stage refinement + tracking | 2021 |
| PointPillars | "PointPillars: Fast Encoders for Object Detection from Point Clouds" | Pillar-based encoding for real-time inference | 2019 |

### 4.5 Radar

| Model | Paper | Key Innovation | Year |
|-------|-------|----------------|------|
| RadarPillarNet | Pillar-based radar 3D detection | Adapted PointPillars for sparse radar returns | 2022 |
| CRAFT | "CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer" | Cross-attention radar-camera fusion | 2023 |

## 5. Functional Scenario Trees

### 5.1 6-Layer Taxonomy (PEGASUS/ASAM-derived)

1. **Road Topology** — road type, lanes, intersections, merges
2. **Traffic Infrastructure** — signals, signs, markings, barriers
3. **Temporary Modifications** — construction zones, temporary signs
4. **Dynamic Objects** — vehicles, VRUs, animals, behaviors
5. **Environment** — weather, lighting, visibility, road surface
6. **Digital Information** — V2X signals, map quality, sensor degradation

### 5.2 Automated Annotation Pipeline

- CLIP-based scene embedding for semantic similarity
- Rule-based classifiers for weather/lighting from camera metadata
- Object co-occurrence patterns for scenario complexity scoring
- Temporal event detection (cut-ins, lane changes, near-misses)

## 6. Datasets

| Dataset | Sensors | Annotations | Size |
|---------|---------|-------------|------|
| nuScenes | 6 cam + 1 LiDAR + 5 radar | 3D boxes, maps, attributes | 1000 scenes |
| nuScenes Map | Same | Vectorized HD map layers | 1000 scenes |
| KITTI | 2 cam + 1 LiDAR | 3D boxes, 2D boxes | 7481 training |
| Waymo Open | 5 cam + 5 LiDAR | 3D boxes, segmentation | 1150 scenes |

## 7. Metrics

### Detection
- **mAP** — mean Average Precision (IoU thresholds)
- **NDS** — nuScenes Detection Score (composite)
- **ATE/ASE/AOE/AVE/AAE** — translation/scale/orientation/velocity/attribute errors

### Segmentation
- **mIoU** — mean Intersection over Union
- **Per-class IoU** — per semantic class

### Map Construction
- **Chamfer Distance** — point-set distance for vectorized maps
- **AP** — per-element (lane divider, boundary, crossing)

### Temporal
- **AMOTA/AMOTP** — tracking accuracy/precision
- **Map Consistency** — IoU across consecutive frames
- **Streaming AP** — latency-aware detection metric

## 8. Technology Stack

- **Python 3.10+**
- **PyTorch 2.x** (primary), **TensorFlow 2.x** (secondary)
- **CUDA 11.8+** / **cuDNN 8.6+**
- **nuScenes-devkit**, **waymo-open-dataset**
- **Open3D**, **PyTorch3D** for 3D operations
- **TensorBoard**, **Weights & Biases** for logging
- **Hydra/YAML** for configuration
- **pytest** for testing

## 9. Implementation Order

1. `common/` — datasets, metrics, transforms, visualization
2. `camera/dynamic_objects/bevformer/` — flagship model
3. `camera/static_map_semantics/stream_mapnet/` — temporal map
4. `lidar/dynamic_objects/pointnet_pp/` — foundational LiDAR
5. `lidar/dynamic_objects/centerpoint/` — production-grade detection
6. `lidar/dynamic_objects/pointpillars/` — real-time baseline
7. `lidar/static_map_semantics/cylinder3d/` — semantic segmentation
8. `camera/dynamic_objects/detr3d/` — alternative to BEVFormer
9. `camera/dynamic_objects/petr/` — StreamPETR temporal
10. `camera/static_map_semantics/maptr/` — vectorized maps
11. `camera/static_map_semantics/hdmapnet/` — semantic BEV maps
12. `lidar/static_map_semantics/rangenet_pp/` — range-based
13. `radar/dynamic_objects/radar_pillarnet/` — radar detection
14. `radar/dynamic_objects/craft/` — radar-camera fusion
15. `radar/static_map_semantics/radar_occupancy/` — radar mapping
16. `scenario_trees/` — full scenario management system

## 10. Success Criteria

- Every model runs with `python train.py --config <config>.yaml` on nuScenes mini split
- Every model includes pre-trained weight download instructions
- Documentation is sufficient for a new engineer to reproduce results independently
- Evaluation scripts reproduce published paper metrics (within 1-2% tolerance)
- Temporal models demonstrate frame-to-frame consistency metrics
