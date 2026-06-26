# Perception Models — Implementation Status

## Resume This Session

```bash
claude --resume <SESSION_ID>
```

> Replace `<SESSION_ID>` with the session ID from your terminal prompt or `claude sessions list`.

---

## Overall Status: ~90% Complete

| Component | Status | Files | Notes |
|-----------|--------|-------|-------|
| **Repository Structure** | DONE | setup.py, requirements.txt, README.md | Installable package |
| **Common Utilities** | DONE | 22 files | Datasets, metrics, transforms, visualization |

---

## Camera / Static Map Semantics

| Model | PyTorch | TensorFlow | Docs | Configs | Scripts | Tests | Status |
|-------|---------|------------|------|---------|---------|-------|--------|
| **StreamMapNet** | DONE | DONE | DONE (6) | DONE (2) | DONE (3) | DONE | COMPLETE |
| **MapTR/v2** | DONE | DONE | DONE (6) | DONE (2) | DONE (3) | DONE | COMPLETE |
| **HDMapNet** | DONE | DONE | DONE (6) | DONE (2) | DONE (3) | DONE | COMPLETE |

## Camera / Dynamic Objects

| Model | PyTorch | TensorFlow | Docs | Configs | Scripts | Tests | Status |
|-------|---------|------------|------|---------|---------|-------|--------|
| **BEVFormer** | DONE | DONE | DONE (6) | DONE (2) | DONE (3) | DONE | COMPLETE |
| **DETR3D** | DONE | DONE | DONE (6) | DONE (1) | DONE (3) | DONE | COMPLETE |
| **PETR/StreamPETR** | DONE | DONE | DONE (6) | DONE (3) | DONE (3) | DONE | COMPLETE |

## LiDAR / Static Map Semantics

| Model | PyTorch | TensorFlow | Docs | Configs | Scripts | Tests | Status |
|-------|---------|------------|------|---------|---------|-------|--------|
| **Cylinder3D** | DONE | DONE | DONE (6) | DONE (2) | DONE (3) | DONE | COMPLETE |
| **RangeNet++** | DONE | DONE | DONE (6) | DONE (2) | DONE (3) | DONE | COMPLETE |

## LiDAR / Dynamic Objects

| Model | PyTorch | TensorFlow | Docs | Configs | Scripts | Tests | Status |
|-------|---------|------------|------|---------|---------|-------|--------|
| **PointNet++** | DONE | DONE | DONE (6) | DONE (3) | DONE (3) | DONE | COMPLETE |
| **CenterPoint** | DONE | DONE | DONE (6) | DONE (2) | DONE (3) | DONE | COMPLETE |
| **PointPillars** | DONE | DONE | DONE (6) | DONE (2) | DONE (3) | DONE | COMPLETE |

## Radar

| Model | PyTorch | TensorFlow | Docs | Configs | Scripts | Tests | Status |
|-------|---------|------------|------|---------|---------|-------|--------|
| **RadarPillarNet** | DONE | DONE | DONE (6) | DONE (1) | DONE (3) | DONE | COMPLETE |
| **CRAFT** | DONE | DONE | DONE (6) | DONE (1) | DONE (3) | DONE | COMPLETE |
| **Radar Occupancy** | DONE | IN PROGRESS | DONE (3) | DONE (1) | — | — | ~80% |

## Scenario Trees

| Component | Status | Files | Notes |
|-----------|--------|-------|-------|
| **Taxonomy** | DONE | 3 | 6-layer tree, schema, visualization |
| **Auto Annotation** | DONE | 6 | CLIP classifier, weather, objects, temporal events |
| **Data Mining** | DONE | 5 | Novelty, coverage, clustering, difficulty, embeddings |
| **Scenario Manager** | DONE | 5 | DB, query engine, splits, dashboard, export |
| **Examples** | DONE | 3 | tag_recording, find_corner_cases, generate_split |
| **Tests** | DONE | 3 | taxonomy, auto_annotation, query_engine |

---

## Key Technical Features Implemented

### Temporal Modeling
- [x] BEVFormer: temporal self-attention on BEV features with ego-motion alignment
- [x] StreamMapNet: streaming BEV propagation via ego-motion warping + temporal attention
- [x] StreamPETR: object-centric query propagation across frames
- [x] CenterPoint: velocity-based greedy center-distance tracking
- [x] Radar Occupancy: temporal BEV accumulation with warp

### Attention Mechanisms
- [x] Multi-scale deformable attention (BEVFormer spatial cross-attention)
- [x] Standard cross-attention with 3D PE (PETR)
- [x] 3D-to-2D projected feature sampling (DETR3D)
- [x] Cross-modal fusion attention (CRAFT camera-radar)

### BEV Transformations
- [x] LSS (Lift-Splat-Shoot) depth-based view transform
- [x] IPM (Inverse Perspective Mapping) homography-based
- [x] GKT (Geometry-guided Kernel Transformer)
- [x] Voxelization + BEV collapse (CenterPoint)
- [x] Pillar encoding + scatter (PointPillars, RadarPillarNet)

### Point Cloud Processing
- [x] Farthest Point Sampling (FPS)
- [x] Ball query / KNN grouping
- [x] Set Abstraction layers (PointNet++)
- [x] Cylindrical voxelization (Cylinder3D)
- [x] Spherical projection to range image (RangeNet++)
- [x] Dynamic voxelization (CenterPoint)
- [x] 3D sparse convolutions via gather/scatter (CenterPoint)

### Metrics
- [x] mAP (center-distance matching, nuScenes style)
- [x] NDS (nuScenes Detection Score)
- [x] ATE/ASE/AOE/AVE/AAE (per-attribute errors)
- [x] mIoU (semantic segmentation)
- [x] Chamfer distance + AP (vectorized maps)
- [x] AMOTA/AMOTP (multi-object tracking)
- [x] Temporal consistency (map stability across frames)
- [x] Streaming AP (latency-aware)

---

## Remaining Work (for next session)

1. **Radar Occupancy TF model** — TensorFlow implementation (agent was in progress)
2. **Additional docs** — training/evaluation guides for radar_occupancy
3. **Integration tests** — End-to-end tests using nuScenes mini split
4. **Pre-trained weight download scripts** — Links to model zoo checkpoints
5. **Docker/environment setup** — Reproducible training environment
6. **Benchmarking scripts** — Standardized FPS/latency measurement

---

## How to Use

```bash
# Install
cd perception-models
pip install -e .

# Train any model
cd camera/dynamic_objects/bevformer
python pytorch/train.py --config configs/bevformer_base.yaml

# Evaluate
python pytorch/evaluate.py --config configs/bevformer_base.yaml --checkpoint best.pth

# Inference on single sample
python pytorch/inference.py --config configs/bevformer_base.yaml --checkpoint best.pth

# Scenario tagging
cd scenario_trees
python examples/tag_recording.py --recording /path/to/scene
```

---

*Last updated: 2026-06-26*
*Session: Claude Code autonomous implementation*
