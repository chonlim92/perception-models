# Perception Models — Implementation Status

## Resume This Session

```bash
claude --resume <SESSION_ID>
```

> Replace `<SESSION_ID>` with the session ID from your terminal prompt or `claude sessions list`.

---

## Overall Status: ~98% Complete

| Component | Status | Files | Notes |
|-----------|--------|-------|-------|
| **Repository Structure** | DONE | setup.py, requirements.txt, README.md | Installable package |
| **Common Utilities** | DONE | 22 files | Datasets, metrics, transforms, visualization |
| **FST Backend API** | DONE | 5 files | FastAPI + SQLite + Analysis Engine |
| **FST React Frontend** | DONE | 14 files | React + TypeScript + ReactFlow |

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
| **Radar Occupancy** | DONE | DONE | DONE (6) | DONE (1) | DONE (3) | DONE | COMPLETE |

## Scenario Trees

| Component | Status | Files | Notes |
|-----------|--------|-------|-------|
| **Taxonomy** | DONE | 3 | 6-layer tree, schema, visualization |
| **Auto Annotation** | DONE | 6 | CLIP classifier, weather, objects, temporal events |
| **Data Mining** | DONE | 5 | Novelty, coverage, clustering, difficulty, embeddings |
| **Scenario Manager** | DONE | 5 | DB, query engine, splits, dashboard, export |
| **Examples** | DONE | 3 | tag_recording, find_corner_cases, generate_split |
| **Tests** | DONE | 3 | taxonomy, auto_annotation, query_engine |

## FST Interactive System (NEW)

| Component | Status | Files | Notes |
|-----------|--------|-------|-------|
| **Backend API** | DONE | 5 | FastAPI, versioning, analysis engine |
| **React Frontend** | DONE | 14 | ReactFlow tree, metrics dashboard, suggestions |
| **Documentation** | DONE | 3 | README, API docs, design spec |

---

## Key Technical Features Implemented

### Temporal Modeling
- [x] BEVFormer: temporal self-attention on BEV features with ego-motion alignment
- [x] StreamMapNet: streaming BEV propagation via ego-motion warping + temporal attention
- [x] StreamPETR: object-centric query propagation across frames
- [x] CenterPoint: velocity-based greedy center-distance tracking
- [x] Radar Occupancy: temporal BEV accumulation with warp (concat_conv/attention/GRU fusion)

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
- [x] Pillar encoding + scatter (PointPillars, RadarPillarNet, Radar Occupancy)

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
- [x] Occupied/Free IoU (radar occupancy)

### FST System Features
- [x] Interactive tree visualization (ReactFlow)
- [x] Tree versioning with semantic version bumps
- [x] Per-node KPI monitoring (pass/warn/fail)
- [x] Recording attachment and management
- [x] Root cause analysis with statistical pattern mining
- [x] Automated node splitting suggestions
- [x] Human approval workflow for tree modifications
- [x] Evaluation script management

---

## Audit Results (2026-06-27)

### Camera Models: ALL COMPLETE
All 6 camera models (BEVFormer, DETR3D, PETR, HDMapNet, MapTR, StreamMapNet) have complete implementations in both PyTorch and TensorFlow. No stubs, no placeholders found.

### LiDAR Models: ALL COMPLETE
All 5 LiDAR models (CenterPoint, PointNet++, PointPillars, Cylinder3D, RangeNet++) have complete implementations. No missing components.

### Radar Models: COMPLETED (was ~80%)
- CRAFT and RadarPillarNet were fully complete
- Radar Occupancy had gaps that were filled:
  - Added: pytorch/losses.py, tensorflow/model.py, tensorflow/evaluate.py, tensorflow/inference.py
  - Added: tests/test_model.py, scripts/ directory (3 files)
  - Added: docs/annotation_guide.md, docs/evaluation_guide.md, docs/training_guide.md
  - All new code tagged with `# [IMPLEMENTED BY CLAUDE - was missing]`

---

## How to Use

```bash
# Install
cd perception-models
pip install -e .

# Train any model (PyTorch)
cd camera/dynamic_objects/bevformer
python pytorch/train.py --config configs/bevformer_base.yaml

# Train (TensorFlow)
python tensorflow/train.py --config configs/bevformer_base.yaml

# Evaluate
python pytorch/evaluate.py --config configs/bevformer_base.yaml --checkpoint best.pth

# Inference on single sample
python pytorch/inference.py --config configs/bevformer_base.yaml --checkpoint best.pth

# Scenario tagging
cd scenario_trees
python examples/tag_recording.py --recording /path/to/scene

# FST Interactive System
# Terminal 1: Backend
uvicorn scenario_trees.api.app:app --reload --port 8000

# Terminal 2: Frontend
cd fst-frontend && npm install && npm run dev
# Open http://localhost:3000
```

---

## Changes Made in This Session (2026-06-27)

### New Files Created
1. `scenario_trees/api/__init__.py` - API package
2. `scenario_trees/api/app.py` - FastAPI application
3. `scenario_trees/api/models.py` - Pydantic schemas
4. `scenario_trees/api/database.py` - Extended DB for versioning/analysis
5. `scenario_trees/api/analysis_engine.py` - Root cause analysis
6. `scenario_trees/api/README.md` - API documentation
7. `fst-frontend/` - Complete React application (14 source files)
8. `radar/static_map_semantics/radar_occupancy/` - Missing components (10 files)
9. `docs/superpowers/specs/2026-06-27-fst-frontend-and-model-audit-design.md` - Design spec

### Tagging Convention
All code implemented to fill gaps is tagged:
```python
# [IMPLEMENTED BY CLAUDE - was missing]
```

---

*Last updated: 2026-06-27*
*Session: Claude Code autonomous implementation (model audit + FST system)*
