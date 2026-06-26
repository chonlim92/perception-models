# Autonomous Driving Perception Models

A comprehensive, production-quality repository of state-of-the-art deep learning perception models for autonomous driving. Each model includes full research documentation, PyTorch and TensorFlow implementations, training scripts, evaluation pipelines, and step-by-step guides.

## Repository Structure

```
perception-models/
├── common/                         # Shared utilities (datasets, metrics, viz)
├── camera/                         # Camera-based perception
│   ├── static_map_semantics/       # HD map construction from cameras
│   │   ├── stream_mapnet/          # StreamMapNet (temporal vectorized mapping)
│   │   ├── maptr/                  # MapTR/MapTRv2 (permutation-equiv mapping)
│   │   └── hdmapnet/              # HDMapNet (semantic BEV map)
│   └── dynamic_objects/            # 3D object detection from cameras
│       ├── bevformer/              # BEVFormer (deformable BEV attention)
│       ├── detr3d/                 # DETR3D (3D-to-2D query projection)
│       └── petr/                   # PETR/StreamPETR (3D position embedding)
├── lidar/                          # LiDAR-based perception
│   ├── static_map_semantics/       # Semantic segmentation
│   │   ├── cylinder3d/             # Cylinder3D (cylindrical partition)
│   │   └── rangenet_pp/            # RangeNet++ (range image CNN)
│   └── dynamic_objects/            # 3D detection and tracking
│       ├── pointnet_pp/            # PointNet++ (hierarchical point learning)
│       ├── centerpoint/            # CenterPoint (center heatmap + tracking)
│       └── pointpillars/           # PointPillars (fast pillar encoding)
├── radar/                          # Radar-based perception
│   ├── static_map_semantics/
│   │   └── radar_occupancy/        # Radar occupancy grid mapping
│   └── dynamic_objects/
│       ├── radar_pillarnet/        # Pillar-based radar detection
│       └── craft/                  # Camera-Radar Fusion Transformer
└── scenario_trees/                 # Functional Scenario Management
    ├── taxonomy/                   # 6-layer scenario classification
    ├── auto_annotation/            # Automated scenario tagging
    ├── data_mining/                # Corner case discovery
    └── scenario_manager/           # Query and data management
```

## Quick Start

### 1. Installation

```bash
# Clone and install
cd perception-models
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### 2. Download Data (nuScenes Mini for quick experiments)

```bash
# Download nuScenes mini split (≈4GB)
cd camera/dynamic_objects/bevformer/scripts
bash download_data.sh --split mini

# Or set your data path
export NUSCENES_ROOT=/path/to/nuscenes
```

### 3. Train a Model

```bash
# Train BEVFormer on nuScenes
cd camera/dynamic_objects/bevformer
python pytorch/train.py --config configs/bevformer_base.yaml

# Train StreamMapNet with temporal fusion
cd camera/static_map_semantics/stream_mapnet
python pytorch/train.py --config configs/stream_mapnet_nuscenes.yaml

# Train CenterPoint with tracking
cd lidar/dynamic_objects/centerpoint
python pytorch/train.py --config configs/centerpoint_voxel.yaml
```

### 4. Evaluate

```bash
# Evaluate with full metrics (mAP, NDS, temporal)
python pytorch/evaluate.py --config configs/bevformer_base.yaml --checkpoint best.pth
```

### 5. Use Scenario Trees

```bash
# Auto-tag a recording
cd scenario_trees
python examples/tag_recording.py --recording /path/to/recording

# Find corner cases
python examples/find_corner_cases.py --dataset nuscenes --min-novelty 0.8
```

## Models Overview

### Camera — Static Map Semantics

| Model | Key Innovation | Temporal | mAP (nuScenes) |
|-------|---------------|----------|-----------------|
| **StreamMapNet** | Streaming temporal BEV fusion for stable map prediction | Yes (multi-frame) | 62.3 |
| **MapTR/v2** | Permutation-equivalent set prediction for vectorized maps | No (single frame) | 58.7 |
| **HDMapNet** | First end-to-end camera→BEV semantic map framework | No | 38.5 |

### Camera — Dynamic Objects (3D Detection)

| Model | Key Innovation | Temporal | mAP / NDS |
|-------|---------------|----------|-----------|
| **BEVFormer** | Deformable attention for BEV + temporal self-attention | Yes | 51.7 / 59.6 |
| **DETR3D** | 3D reference point projection to 2D (no explicit BEV) | No | 34.9 / 42.2 |
| **PETR/StreamPETR** | 3D position embedding + object-centric temporal | Yes (StreamPETR) | 50.4 / 59.2 |

### LiDAR — Dynamic Objects (3D Detection + Tracking)

| Model | Key Innovation | Tracking | mAP / AMOTA |
|-------|---------------|----------|-------------|
| **CenterPoint** | Center heatmap + velocity-based tracking | Yes | 60.3 / 63.8 |
| **PointPillars** | Fast pillar encoding (62 Hz) | No | 40.1 / — |
| **PointNet++** | Hierarchical point set learning (foundational) | No | — / — |

### LiDAR — Semantic Segmentation

| Model | Key Innovation | Speed | mIoU (SemanticKITTI) |
|-------|---------------|-------|---------------------|
| **Cylinder3D** | Cylindrical partition + asymmetric 3D convolutions | 10 Hz | 67.8 |
| **RangeNet++** | Range image 2D CNN + KNN post-processing | 50 Hz | 52.2 |

### Radar

| Model | Key Innovation | Fusion | mAP (nuScenes) |
|-------|---------------|--------|-----------------|
| **CRAFT** | Camera-Radar spatio-contextual fusion transformer | Camera+Radar | 41.1 |
| **RadarPillarNet** | PointPillars adapted for sparse radar | Radar-only | 28.3 |

## Key Features

- **Temporal Modeling**: StreamMapNet, BEVFormer, StreamPETR, CenterPoint all use temporal fusion across frames
- **Deformable Attention**: BEVFormer's spatial cross-attention uses deformable attention for efficient BEV feature generation
- **Object Tracking**: CenterPoint includes greedy center-distance tracking with velocity prediction
- **Multi-Sensor Fusion**: CRAFT demonstrates camera-radar fusion via cross-attention transformers
- **Scenario Management**: Automated tagging, corner case mining, and balanced dataset splitting

## Datasets Supported

| Dataset | Sensors | Tasks | Download |
|---------|---------|-------|----------|
| **nuScenes** | 6 cam + 1 LiDAR + 5 radar | Detection, Tracking, Maps | [nuscenes.org](https://www.nuscenes.org/) |
| **nuScenes Map** | Same + HD map | Map construction | Same as above |
| **KITTI** | 2 cam + 1 LiDAR | Detection, Segmentation | [cvlibs.net](http://www.cvlibs.net/datasets/kitti/) |
| **SemanticKITTI** | 1 LiDAR | Semantic segmentation | [semantic-kitti.org](http://www.semantic-kitti.org/) |
| **Waymo Open** | 5 cam + 5 LiDAR | Detection, Tracking | [waymo.com/open](https://waymo.com/open/) |

## Hardware Requirements

| Model | Training GPU | Training Time (nuScenes) | Inference FPS |
|-------|-------------|-------------------------|---------------|
| BEVFormer | 8× A100 (40GB) | ~48h | 4.2 |
| StreamMapNet | 8× A100 (40GB) | ~36h | 12.5 |
| CenterPoint | 4× V100 (32GB) | ~20h | 11.0 |
| PointPillars | 2× V100 (32GB) | ~8h | 62.0 |
| RangeNet++ | 1× V100 (32GB) | ~12h | 50.0 |

## Metrics Reference

### Detection Metrics
- **mAP**: mean Average Precision across classes and IoU thresholds
- **NDS**: nuScenes Detection Score (weighted combination of mAP + errors)
- **ATE/ASE/AOE/AVE/AAE**: Translation/Scale/Orientation/Velocity/Attribute errors

### Segmentation Metrics
- **mIoU**: mean Intersection over Union across all classes

### Map Metrics
- **Chamfer Distance**: Average closest-point distance between predicted and GT polylines
- **AP**: Average Precision for vectorized map elements at distance thresholds

### Tracking Metrics
- **AMOTA**: Average Multi-Object Tracking Accuracy (over all recall thresholds)
- **AMOTP**: Average Multi-Object Tracking Precision
- **IDS**: Identity Switches (track fragmentation)

### Temporal Metrics
- **Map Consistency**: IoU of predicted maps between consecutive frames
- **Streaming AP**: Latency-aware detection metric accounting for computation time

## Contributing

Each model follows a consistent structure. To add a new model:

1. Create the directory under the appropriate `{sensor}/{task}/` path
2. Follow the template: `docs/`, `configs/`, `pytorch/`, `tensorflow/`, `scripts/`, `tests/`
3. Implement the full pipeline: data → model → train → evaluate → infer
4. Document everything in the `docs/` folder and `README.md`

## License

This repository is for research and educational purposes. Individual model implementations may reference papers with their own licenses. Check each model's README for specific paper citations and license information.
