# Autonomous Driving Perception Models

A comprehensive, production-quality repository of state-of-the-art deep learning perception models for autonomous driving. Each model includes full research documentation, PyTorch and TensorFlow implementations, training scripts, evaluation pipelines, and step-by-step guides.

---

## Table of Contents

1. [Introduction to Autonomous Driving Perception](#introduction-to-autonomous-driving-perception)
2. [Sensor Fundamentals](#sensor-fundamentals)
3. [Core Concepts You Need to Know](#core-concepts-you-need-to-know)
4. [Repository Structure](#repository-structure)
5. [Models Overview](#models-overview)
6. [Learning Path (Recommended Order)](#learning-path-recommended-order)
7. [Quick Start](#quick-start)
8. [Datasets Supported](#datasets-supported)
9. [Hardware Requirements](#hardware-requirements)
10. [Metrics Reference](#metrics-reference)
11. [Contributing](#contributing)

---

## Introduction to Autonomous Driving Perception

### What is Perception?

Perception is the "eyes and ears" of an autonomous vehicle. It answers three fundamental questions:

1. **Where am I?** (Localization relative to the HD map)
2. **What is around me?** (Object detection: cars, pedestrians, cyclists, etc.)
3. **What does the road look like?** (Lane lines, road boundaries, drivable area)

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS DRIVING STACK                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│    ┌───────────┐    ┌──────────────┐    ┌───────────────┐   │
│    │ PERCEPTION│───→│   PREDICTION │───→│   PLANNING    │   │
│    │           │    │              │    │               │   │
│    │ "What is  │    │ "Where will  │    │ "What should  │   │
│    │  there?"  │    │  they go?"   │    │  I do?"       │   │
│    └───────────┘    └──────────────┘    └───────────────┘   │
│          ↑                                      │           │
│    ┌───────────┐                         ┌──────────────┐   │
│    │  SENSORS  │                         │   CONTROL    │   │
│    │ cam/lidar │                         │ steer/brake  │   │
│    │  /radar   │                         │  /throttle   │   │
│    └───────────┘                         └──────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Why is Perception Hard?

- **3D world from 2D images**: Cameras give us flat pictures, but we need to understand the 3D world
- **Sparse data**: LiDAR gives only ~100k points per frame (vs millions of image pixels)
- **Sensor noise**: Radar has multipath reflections, cameras struggle in darkness/glare
- **Real-time requirements**: Must process at 10-20+ Hz for safe driving
- **Long-tail edge cases**: Construction zones, unusual vehicles, adverse weather
- **Temporal reasoning**: Objects move; we need to track them across time

### Two Main Perception Tasks in This Repository

| Task | What It Does | Example Models |
|------|-------------|----------------|
| **Static Map Semantics** | Predicts the road structure (lanes, boundaries, crossings) | StreamMapNet, MapTR, HDMapNet, Cylinder3D, RangeNet++ |
| **Dynamic Object Detection** | Finds and tracks moving objects (cars, pedestrians, cyclists) | BEVFormer, CenterPoint, DETR3D, PETR, PointPillars |

---

## Sensor Fundamentals

### The Three Sensor Modalities

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAMERA (6x surround-view)                      │
│                                                                   │
│  Strengths: Dense texture, color, semantics, cheap               │
│  Weaknesses: No direct depth, affected by lighting               │
│  Output: 2D images (e.g., 1600×900 pixels × 6 cameras)          │
│  Range: Up to ~250m (degrades with distance)                     │
│  Rate: 12 Hz (nuScenes)                                         │
├─────────────────────────────────────────────────────────────────┤
│                    LiDAR (1x 360° spinning)                       │
│                                                                   │
│  Strengths: Precise 3D geometry, range accuracy, all lighting    │
│  Weaknesses: Sparse, expensive, affected by rain/fog             │
│  Output: Point cloud (~34,000 points per frame, x/y/z/intensity) │
│  Range: Up to ~80m (dense) / 200m (sparse)                      │
│  Rate: 20 Hz (nuScenes)                                         │
├─────────────────────────────────────────────────────────────────┤
│                    RADAR (5x surround-view)                       │
│                                                                   │
│  Strengths: Direct velocity (Doppler), works in ALL weather      │
│  Weaknesses: Very sparse (~100-300 pts), low resolution          │
│  Output: Sparse points (x/y/vx/vy/rcs per point)                │
│  Range: Up to ~250m                                              │
│  Rate: 13 Hz (nuScenes)                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Sensor Placement on a Typical Autonomous Vehicle (nuScenes Setup)

```
                    CAM_FRONT
                       │
              CAM_FL ──┼── CAM_FR
                 /     │     \
    RADAR_FL ── /   LIDAR_TOP  \ ── RADAR_FR
               │   ┌───────┐   │
               │   │       │   │
    RADAR_FL ──│   │  CAR  │   │── RADAR_FR
               │   │       │   │
               │   └───────┘   │
              CAM_BL ──┼── CAM_BR
                       │
                   CAM_BACK
                       │
               RADAR_BACK
```

### Why Use Multiple Sensors?

Each sensor has different failure modes. Combining them provides **redundancy** and **complementary information**:

| Condition | Camera | LiDAR | Radar |
|-----------|--------|-------|-------|
| Bright sunlight | Poor (glare) | Good | Good |
| Night/darkness | Poor | Good | Good |
| Heavy rain | Moderate | Poor | Good |
| Fog | Poor | Poor | Good |
| Range accuracy | Poor (no depth) | Excellent | Good |
| Velocity | Computed (slow) | Computed | Direct (Doppler) |
| Semantic understanding | Excellent | Poor | Poor |
| Cost | Low ($50-200) | High ($1k-75k) | Medium ($100-500) |

---

## Core Concepts You Need to Know

### Bird's Eye View (BEV) Representation

**BEV is THE most important concept in modern autonomous driving perception.**

Instead of reasoning about the world from each camera's perspective, we transform everything to a "top-down" view — like looking at the car from a helicopter. This makes it easy to:
- Measure distances between objects
- Plan driving paths
- Fuse information from all sensors (they all project to the same BEV plane)

```
Camera View (2D perspective)         BEV (top-down, metric space)
                                    
   ┌────────────────────┐           ┌────────────────────┐
   │    sky              │           │        ↑ 50m       │
   │   ┌──┐  ┌──┐      │           │    ┌──┐            │
   │   │  │  │  │      │   ───→    │    │  │  ←car      │
   │   └──┘  └──┘      │           │         ┌──┐       │
   │  ══════════════    │           │    ═══  │  │       │
   │   road surface     │           │    road └──┘       │
   └────────────────────┘           │    EGO↓            │
                                    └────────────────────┘
                                       -50m ← → +50m
```

**Key BEV generation methods** (all implemented in this repo):

| Method | How It Works | Models Using It |
|--------|-------------|-----------------|
| **LSS** (Lift-Splat-Shoot) | Predict depth for each pixel, create 3D frustum, project to BEV | StreamMapNet, HDMapNet |
| **Deformable Attention** | BEV queries attend to image features via 3D-to-2D projection | BEVFormer |
| **3D Position Embedding** | Encode 3D coordinates into 2D features (implicit BEV) | PETR |
| **GKT** (Geometry-guided Kernel Transformer) | Use camera geometry to guide cross-attention | MapTR |
| **Voxelization + Collapse** | Create 3D voxel grid from point cloud, compress height | CenterPoint |
| **Pillar Encoding** | Point cloud → vertical columns → PointNet → BEV image | PointPillars |

### Temporal Fusion (Using History)

Self-driving cars process video, not single frames. Using past observations improves perception:

- **More coverage**: The car moves, revealing previously occluded areas
- **Velocity estimation**: Track objects over time to estimate speed
- **Stability**: Reduce flickering/inconsistency in predictions
- **Disambiguation**: See an object from multiple angles as you drive past

```
Time t-2          Time t-1          Time t (current)
┌─────────┐       ┌─────────┐       ┌─────────┐
│  BEV_2  │ ─ego─→│  BEV_1  │ ─ego─→│  BEV_0  │
│(warped) │ motion│(warped) │ motion│(current)│
└────┬────┘       └────┬────┘       └────┬────┘
     └──────────────────┴──────────────────┘
                        │
                  ┌─────┴─────┐
                  │  TEMPORAL  │
                  │   FUSION   │
                  │ (attention │
                  │  or concat)│
                  └─────┬─────┘
                        ↓
                  Enhanced BEV
                  (uses ALL 3 frames)
```

**Key temporal methods** in this repo:
- **BEV Warping** (StreamMapNet, BEVFormer): Transform past BEV to current frame using ego-motion
- **Query Propagation** (StreamPETR): Pass object queries from frame to frame
- **Velocity Tracking** (CenterPoint): Predict velocity, then match objects across frames

### Attention Mechanisms in Perception

Modern perception models heavily use **attention** (from the Transformer architecture):

1. **Standard Attention**: Query attends to all keys — O(N²) cost
2. **Deformable Attention**: Query only attends to a few learned offset locations — O(N·K) with K<<N
3. **Cross-Attention**: One modality attends to another (e.g., BEV queries attend to image features)

BEVFormer's **Spatial Cross-Attention** (the key innovation):
```
BEV Query (at position x,y)
     │
     ↓ generate 3D reference points at multiple heights
     │
     ↓ project to each camera using intrinsics/extrinsics
     │
     ↓ sample image features at projected locations (deformable)
     │
     ↓ aggregate with learned attention weights
     │
     ↓ updated BEV feature at (x,y)
```

### Coordinate Systems

Understanding coordinate transformations is critical:

```
Sensor Frame (camera/LiDAR)     Ego Frame (car body)        World Frame (global)
  x: right                       x: forward                  x: East
  y: down (camera)               y: left                     y: North
  z: forward (camera)            z: up                       z: up
  
  Sensor → Ego: calibration (fixed per sensor)
  Ego → World: ego_pose (changes every frame as car moves)
```

**Transformation chain**: Sensor → Ego → World → BEV

---

## Repository Structure

```
perception-models/
├── README.md                       # This file (you are here)
├── requirements.txt                # All Python dependencies
├── setup.py                        # pip install -e . (installable package)
├── implementation_status.md        # Current progress tracker
│
├── common/                         # Shared utilities across all models
│   ├── datasets/                   # Data loaders (nuScenes, KITTI)
│   ├── metrics/                    # All evaluation metrics
│   ├── transforms/                 # Augmentations, coordinate systems
│   └── visualization/              # BEV plots, point cloud viz, image overlays
│
├── camera/                         # Camera-based perception
│   ├── static_map_semantics/       # HD map construction from cameras
│   │   ├── stream_mapnet/          # StreamMapNet (temporal vectorized mapping)
│   │   ├── maptr/                  # MapTR/MapTRv2 (permutation-equiv mapping)
│   │   └── hdmapnet/              # HDMapNet (semantic BEV map)
│   └── dynamic_objects/            # 3D object detection from cameras
│       ├── bevformer/              # BEVFormer (deformable BEV attention)
│       ├── detr3d/                 # DETR3D (3D-to-2D query projection)
│       └── petr/                   # PETR/StreamPETR (3D position embedding)
│
├── lidar/                          # LiDAR-based perception
│   ├── static_map_semantics/       # Semantic segmentation
│   │   ├── cylinder3d/             # Cylinder3D (cylindrical partition)
│   │   └── rangenet_pp/            # RangeNet++ (range image CNN)
│   └── dynamic_objects/            # 3D detection and tracking
│       ├── pointnet_pp/            # PointNet++ (hierarchical point learning)
│       ├── centerpoint/            # CenterPoint (center heatmap + tracking)
│       └── pointpillars/           # PointPillars (fast pillar encoding)
│
├── radar/                          # Radar-based perception
│   ├── static_map_semantics/
│   │   └── radar_occupancy/        # Radar occupancy grid mapping
│   └── dynamic_objects/
│       ├── radar_pillarnet/        # Pillar-based radar detection
│       └── craft/                  # Camera-Radar Fusion Transformer
│
└── scenario_trees/                 # Functional Scenario Management
    ├── taxonomy/                   # 6-layer scenario classification
    ├── auto_annotation/            # Automated scenario tagging (CLIP)
    ├── data_mining/                # Corner case discovery (Isolation Forest)
    └── scenario_manager/           # Query engine, database, splits
```

### Per-Model Directory Structure (Consistent Across All Models)

```
model_name/
├── README.md                # Quick start + architecture overview
├── docs/
│   ├── research_summary.md     # Deep teaching doc (theory + intuition)
│   ├── model_architecture.md   # Full architecture with tensor shapes
│   ├── training_guide.md       # Step-by-step training tutorial
│   ├── evaluation_guide.md     # Metrics explained + how to evaluate
│   └── data_collection.md      # Dataset format + preprocessing
├── configs/
│   └── *.yaml                  # Hyperparameter configs
├── pytorch/
│   ├── model.py                # Model definition
│   ├── dataset.py              # Data loading + preprocessing
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation pipeline
│   └── inference.py            # Single-sample inference + viz
├── tensorflow/
│   └── (same structure)        # TF2/Keras implementation
├── scripts/
│   ├── download_data.sh        # Dataset download
│   ├── prepare_data.sh         # Preprocessing
│   └── visualize.py            # Visualization utilities
└── tests/
    └── test_*.py               # Unit tests
```

---

## Models Overview

### Camera — Static Map Semantics (HD Map Construction)

These models predict the road structure (lane lines, road boundaries, pedestrian crossings) from camera images, typically outputting a Bird's Eye View map.

| Model | Key Innovation | Temporal | Representation | mAP (nuScenes) |
|-------|---------------|----------|----------------|-----------------|
| **StreamMapNet** | Streaming temporal BEV fusion for stable map prediction | Yes (multi-frame) | Vectorized (polylines) | 62.3 |
| **MapTR/v2** | Permutation-equivalent set prediction + hierarchical matching | No (single frame) | Vectorized (polylines) | 58.7 |
| **HDMapNet** | First end-to-end camera→BEV semantic map framework | No | Rasterized (segmentation) | 38.5 |

**Evolution**: HDMapNet (rasterized) → MapTR (vectorized, single-frame) → StreamMapNet (vectorized, temporal)

### Camera — Dynamic Objects (3D Detection from Images)

These models detect 3D bounding boxes (position, size, orientation, velocity) of objects from multi-camera images — the most challenging camera perception task.

| Model | Key Innovation | Temporal | mAP / NDS |
|-------|---------------|----------|-----------|
| **BEVFormer** | Deformable attention for BEV generation + temporal self-attention | Yes | 51.7 / 59.6 |
| **StreamPETR** | Object-centric temporal propagation via query passing | Yes | 50.4 / 59.2 |
| **PETR** | 3D position embedding (encode 3D coords into 2D features) | No | 38.3 / 44.2 |
| **DETR3D** | 3D reference point → project to 2D → sample features | No | 34.9 / 42.2 |

**Evolution**: DETR3D (project & sample) → PETR (position embedding) → BEVFormer (deformable BEV) → StreamPETR (temporal queries)

### LiDAR — Dynamic Objects (3D Detection + Tracking)

These models detect and track objects from LiDAR point clouds — generally more accurate than camera-based methods due to direct 3D measurement.

| Model | Key Innovation | Tracking | Speed | mAP / AMOTA |
|-------|---------------|----------|-------|-------------|
| **CenterPoint** | Center heatmap + velocity-based tracking | Yes | 11 Hz | 60.3 / 63.8 |
| **PointPillars** | Fast pillar encoding → 2D CNN (62 Hz real-time) | No | 62 Hz | 40.1 / — |
| **PointNet++** | Hierarchical point set learning (foundational architecture) | No | ~5 Hz | — / — |

**Evolution**: PointNet++ (foundational) → PointPillars (fast via pillars) → CenterPoint (center-based + tracking)

### LiDAR — Semantic Segmentation

These models assign a semantic class label (road, sidewalk, car, vegetation, etc.) to every LiDAR point.

| Model | Key Innovation | Speed | mIoU (SemanticKITTI) |
|-------|---------------|-------|---------------------|
| **Cylinder3D** | Cylindrical partition + asymmetric 3D convolutions | 10 Hz | 67.8 |
| **RangeNet++** | Range image 2D CNN + KNN post-processing | 50 Hz | 52.2 |

**Trade-off**: Cylinder3D is more accurate (cylindrical coords match LiDAR distribution), RangeNet++ is faster (uses standard 2D CNN).

### Radar — Detection and Fusion

Radar-based perception is the most challenging due to extreme sparsity, but radar is the only sensor that works reliably in ALL weather conditions and provides direct velocity measurements.

| Model | Key Innovation | Modalities | mAP (nuScenes) |
|-------|---------------|------------|-----------------|
| **CRAFT** | Camera-Radar spatio-contextual fusion transformer | Camera+Radar | 41.1 |
| **RadarPillarNet** | PointPillars adapted for sparse radar (larger pillars, velocity features) | Radar-only | 28.3 |
| **Radar Occupancy** | Bayesian/neural occupancy grid from radar (classical + learned) | Radar-only | IoU: 62.4 |

---

## Learning Path (Recommended Order)

If you're new to autonomous driving perception, follow this order:

### Phase 1: Foundations (Week 1-2)

1. **PointNet++** — Learn how neural networks process raw 3D point clouds
   - Key concepts: FPS, ball query, set abstraction, permutation invariance
   - Why: Foundation for ALL point cloud methods

2. **PointPillars** — Learn the fastest approach to 3D detection
   - Key concepts: Pillar encoding, scatter to pseudo-image, anchor-based detection
   - Why: Simple architecture, introduces BEV concept

3. **HDMapNet** — Learn camera-to-BEV transformation
   - Key concepts: IPM, LSS (Lift-Splat-Shoot), BEV segmentation
   - Why: Introduces the camera→BEV pipeline that all camera models use

### Phase 2: Core Techniques (Week 3-4)

4. **BEVFormer** — Learn deformable attention for BEV
   - Key concepts: Deformable attention, spatial cross-attention, temporal self-attention
   - Why: State-of-the-art technique, combines many important ideas

5. **CenterPoint** — Learn center-based detection + tracking
   - Key concepts: Sparse 3D convolutions, center heatmaps, velocity-based tracking
   - Why: Top-performing LiDAR detector with integrated tracking

6. **MapTR** — Learn vectorized map prediction with set matching
   - Key concepts: Permutation equivalence, Hungarian matching, GKT
   - Why: Modern map prediction approach, elegant formulation

### Phase 3: Advanced Topics (Week 5-6)

7. **StreamMapNet** — Learn temporal fusion for mapping
   - Key concepts: BEV warping, ego-motion alignment, temporal attention
   - Why: Shows how temporal modeling dramatically improves results

8. **PETR/StreamPETR** — Learn position embedding approach + object-centric temporal
   - Key concepts: 3D PE, query propagation, motion-aware layer norm
   - Why: Alternative to BEV, more memory-efficient temporal modeling

9. **CRAFT** — Learn sensor fusion
   - Key concepts: Cross-attention fusion, multi-modal learning
   - Why: Real production systems use multiple sensors

### Phase 4: Specialization (Week 7-8)

10. **Cylinder3D** — Cylindrical coordinates for LiDAR segmentation
11. **RangeNet++** — Range image approach (fastest segmentation)
12. **Radar Occupancy** — Classical + neural occupancy mapping
13. **Scenario Trees** — Data management and corner case mining

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/chonlim92/perception-models.git
cd perception-models

# Option A: Install as package (recommended)
pip install -e .

# Option B: Install dependencies directly
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch >= 1.12 (GPU support required for training)
- TensorFlow >= 2.10 (for TF implementations)
- nuscenes-devkit (dataset API)
- open3d (point cloud visualization)
- einops (tensor reshaping)
- timm (pretrained image backbones)

### 2. Download Data

```bash
# Option A: nuScenes Mini (4GB, for quick experiments)
cd camera/dynamic_objects/bevformer/scripts
bash download_data.sh --split mini

# Option B: Full nuScenes (dozens of GB, for real training)
# Register at nuscenes.org, download, set path:
export NUSCENES_ROOT=/path/to/nuscenes

# Option C: SemanticKITTI (for LiDAR segmentation)
cd lidar/static_map_semantics/cylinder3d/scripts
bash download_data.sh
```

### 3. Train a Model

```bash
# Train BEVFormer (camera 3D detection with temporal fusion)
cd camera/dynamic_objects/bevformer
python pytorch/train.py --config configs/bevformer_base.yaml

# Train StreamMapNet (temporal HD map construction)
cd camera/static_map_semantics/stream_mapnet
python pytorch/train.py --config configs/stream_mapnet_nuscenes.yaml

# Train CenterPoint (LiDAR detection + tracking)
cd lidar/dynamic_objects/centerpoint
python pytorch/train.py --config configs/centerpoint_voxel.yaml

# Train PointPillars (fastest LiDAR detector, real-time)
cd lidar/dynamic_objects/pointpillars
python pytorch/train.py --config configs/pointpillars_nuscenes.yaml
```

### 4. Evaluate

```bash
# Full evaluation with all metrics
python pytorch/evaluate.py --config configs/bevformer_base.yaml --checkpoint outputs/best.pth

# Quick evaluation on mini split
python pytorch/evaluate.py --config configs/bevformer_base.yaml --checkpoint best.pth --split mini_val
```

### 5. Inference (Visualization)

```bash
# Visualize detections on a single sample
python pytorch/inference.py --config configs/bevformer_base.yaml --checkpoint best.pth --sample-idx 0
```

### 6. Use Scenario Trees

```bash
# Auto-tag a recording with scenario metadata
cd scenario_trees
python examples/tag_recording.py --recording /path/to/recording

# Find corner cases (rare/unusual scenarios)
python examples/find_corner_cases.py --dataset nuscenes --min-novelty 0.8

# Generate balanced training split
python examples/generate_split.py --coverage-target 0.9
```

---

## Datasets Supported

| Dataset | Sensors | Scenes | Tasks | Size | Download |
|---------|---------|--------|-------|------|----------|
| **nuScenes** | 6 cam + 1 LiDAR + 5 radar | 1000 (850 train/val) | Detection, Tracking, Maps | ~300GB | [nuscenes.org](https://www.nuscenes.org/) |
| **nuScenes Mini** | Same | 10 scenes | Same (for prototyping) | ~4GB | Same site |
| **KITTI** | 2 cam + 1 LiDAR | 7481 training | Detection, Segmentation | ~12GB | [cvlibs.net](http://www.cvlibs.net/datasets/kitti/) |
| **SemanticKITTI** | 1 LiDAR | 22 sequences | Semantic segmentation | ~80GB | [semantic-kitti.org](http://www.semantic-kitti.org/) |
| **Waymo Open** | 5 cam + 5 LiDAR | 1150 segments | Detection, Tracking | ~1.5TB | [waymo.com/open](https://waymo.com/open/) |

### nuScenes — The Primary Dataset

nuScenes is the most commonly used dataset in this repository. Key facts:

- **Location**: Boston and Singapore (diverse driving conditions)
- **Annotations**: Full 3D boxes with tracking IDs, 23 object classes, HD map layers
- **Sensor setup**: 360° coverage with 6 cameras, 32-beam LiDAR, 5 radar sensors
- **Temporal**: 2 Hz keyframes (annotated), 12 Hz intermediate frames
- **Map**: Vectorized HD map with lane lines, road boundaries, walkways, etc.

---

## Hardware Requirements

| Model | Min GPU (Training) | Recommended | Training Time (nuScenes full) | Inference FPS |
|-------|-------------------|-------------|-------------------------------|---------------|
| BEVFormer | 4× V100 (32GB) | 8× A100 (40GB) | ~48h | 4.2 |
| StreamMapNet | 4× V100 (32GB) | 8× A100 (40GB) | ~36h | 12.5 |
| MapTR | 4× V100 (32GB) | 8× A100 (40GB) | ~24h | 15.8 |
| CenterPoint | 2× V100 (32GB) | 4× V100 (32GB) | ~20h | 11.0 |
| PointPillars | 1× V100 (32GB) | 2× V100 (32GB) | ~8h | 62.0 |
| PointNet++ | 1× V100 (32GB) | 2× V100 (32GB) | ~12h | ~5 |
| Cylinder3D | 1× V100 (32GB) | 4× V100 (32GB) | ~24h | 10.0 |
| RangeNet++ | 1× V100 (16GB) | 1× V100 (32GB) | ~12h | 50.0 |
| CRAFT | 4× V100 (32GB) | 8× A100 (40GB) | ~30h | 8.0 |
| RadarPillarNet | 1× V100 (16GB) | 2× V100 (32GB) | ~6h | 45.0 |

**For prototyping**: Use nuScenes Mini split + single GPU. Most models train in <1h on mini.

---

## Metrics Reference

### Detection Metrics (3D Object Detection)

| Metric | What It Measures | Formula/Description | Good Value |
|--------|-----------------|---------------------|------------|
| **mAP** | Detection accuracy | Mean of per-class AP (center-distance matching at 0.5/1/2/4m) | >50% |
| **NDS** | Overall detection quality | 0.5×mAP + 0.1×(5 - sum of TP errors) | >55% |
| **ATE** | Position error | Euclidean distance between predicted and GT center | <0.5m |
| **ASE** | Size error | 1 - 3D IoU (after center alignment) | <0.2 |
| **AOE** | Orientation error | Yaw angle difference (smallest arc) | <0.3 rad |
| **AVE** | Velocity error | L2 distance between velocity vectors | <0.5 m/s |
| **AAE** | Attribute error | 1 - accuracy of attribute classification | <0.2 |

### Segmentation Metrics (Semantic Segmentation)

| Metric | What It Measures | Formula | Good Value |
|--------|-----------------|---------|------------|
| **mIoU** | Per-class overlap | Mean of (TP / (TP + FP + FN)) per class | >65% |
| **Overall Accuracy** | Pixel-level correctness | Total correct / Total pixels | >90% |

### Map Metrics (HD Map Construction)

| Metric | What It Measures | Description | Good Value |
|--------|-----------------|-------------|------------|
| **Chamfer Distance** | Polyline accuracy | Average closest-point distance (meters) | <1.0m |
| **AP@0.5/1.0/1.5m** | Detection + accuracy | True positive if Chamfer < threshold | >50% |
| **mAP** | Mean across elements | Mean of AP over lane/boundary/crossing | >55% |

### Tracking Metrics (Multi-Object Tracking)

| Metric | What It Measures | Description | Good Value |
|--------|-----------------|-------------|------------|
| **AMOTA** | Overall tracking accuracy | Recall-normalized MOTA averaged over thresholds | >60% |
| **AMOTP** | Tracking precision | Average position error of true positive tracks | <1.0m |
| **IDS** | Identity switches | Times a track gets reassigned to a different object | <500 |
| **Frag** | Track fragmentation | Times a track is lost then re-found | <500 |

### Temporal Metrics

| Metric | What It Measures | Description |
|--------|-----------------|-------------|
| **Map Consistency** | Stability across frames | IoU between consecutive frame predictions |
| **Streaming AP** | Latency-aware accuracy | AP that penalizes slow inference |

---

## Key Technical Features Implemented

### Temporal Modeling
- BEVFormer: Temporal self-attention on BEV features with ego-motion alignment
- StreamMapNet: Streaming BEV propagation via ego-motion warping + temporal attention
- StreamPETR: Object-centric query propagation across frames
- CenterPoint: Velocity-based greedy center-distance tracking

### Attention Mechanisms
- Multi-scale deformable attention (BEVFormer spatial cross-attention)
- Standard cross-attention with 3D PE (PETR)
- 3D-to-2D projected feature sampling (DETR3D)
- Cross-modal fusion attention (CRAFT camera-radar)

### BEV Generation Methods
- LSS (Lift-Splat-Shoot): Learned depth + voxel pooling
- IPM (Inverse Perspective Mapping): Homography under flat-ground assumption
- GKT (Geometry-guided Kernel Transformer): Camera-geometry-guided cross-attention
- Voxelization + BEV Collapse (CenterPoint)
- Pillar Encoding + Scatter (PointPillars, RadarPillarNet)

### Point Cloud Processing
- Farthest Point Sampling (FPS)
- Ball Query / KNN Grouping
- Set Abstraction layers (PointNet++)
- Cylindrical Voxelization (Cylinder3D)
- Spherical Projection to Range Image (RangeNet++)
- Dynamic Voxelization and 3D Sparse Convolutions (CenterPoint)

---

## Contributing

Each model follows a consistent structure. To add a new model:

1. Create the directory under the appropriate `{sensor}/{task}/` path
2. Follow the template: `docs/`, `configs/`, `pytorch/`, `tensorflow/`, `scripts/`, `tests/`
3. Implement the full pipeline: data → model → train → evaluate → infer
4. Document everything in the `docs/` folder and `README.md`
5. Add comprehensive teaching content (see existing models for examples)

---

## Glossary

| Term | Meaning |
|------|---------|
| **BEV** | Bird's Eye View — top-down 2D representation of the 3D world |
| **Ego** | The self-driving vehicle itself (ego vehicle) |
| **Ego-motion** | How the ego vehicle moved between frames (rotation + translation) |
| **FPN** | Feature Pyramid Network — multi-scale feature extraction |
| **GT** | Ground Truth — the correct labels for training/evaluation |
| **Hungarian Matching** | Optimal bipartite assignment between predictions and GT |
| **Intrinsics** | Camera internal parameters (focal length, principal point) |
| **Extrinsics** | Camera pose relative to the vehicle (rotation + translation) |
| **Keyframe** | An annotated frame in a temporal sequence (2 Hz in nuScenes) |
| **LSS** | Lift-Splat-Shoot — depth-based camera-to-BEV projection |
| **NMS** | Non-Maximum Suppression — removing duplicate detections |
| **Pillar** | A vertical column in a 3D grid (x-y cell, all heights) |
| **Voxel** | A 3D grid cell (x-y-z) |
| **Sweep** | A single sensor capture (one rotation of LiDAR, one camera shot) |

---

## Citation

If you use any model in this repository for research, please cite the original papers:

```bibtex
@inproceedings{li2022bevformer,
  title={BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  booktitle={ECCV},
  year={2022}
}

@inproceedings{yuan2024streammapnet,
  title={StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction},
  author={Yuan, Tianyuan and Liu, Yicheng and Wang, Yue and Wang, Yilun and Zhao, Hang},
  booktitle={WACV},
  year={2024}
}

@inproceedings{yin2021centerpoint,
  title={Center-based 3D Object Detection and Tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Krahenbuhl, Philipp},
  booktitle={CVPR},
  year={2021}
}
```

See each model's README for its specific citation.

---

## License

This repository is for research and educational purposes. Individual model implementations reference papers with their own licenses. Check each model's README for specific paper citations and license details.
