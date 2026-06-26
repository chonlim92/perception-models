# Evaluation Guide - PETR / PETRv2 / StreamPETR

## Overview

This guide covers the evaluation protocol, metrics, and performance benchmarks for PETR family models on the nuScenes detection benchmark. It includes standard evaluation, streaming evaluation (for StreamPETR), speed benchmarks, and ablation studies.

---

## nuScenes Detection Metrics

### Primary Metrics

The nuScenes benchmark uses two primary metrics:

#### 1. Mean Average Precision (mAP)

Unlike traditional 2D mAP which uses IoU for matching, nuScenes uses **center distance** in BEV:

```
Matching criteria: 2D Euclidean distance between predicted and GT box centers
                   projected onto the ground plane (BEV)

Distance thresholds: {0.5m, 1.0m, 2.0m, 4.0m}

mAP = mean over 10 classes of (mean AP over 4 distance thresholds)
    = (1/10) * sum_c [ (1/4) * sum_d [ AP(class=c, dist=d) ] ]
```

AP computation per class per threshold:
1. Sort all predictions by confidence score (descending)
2. Match predictions to ground truth using center distance < threshold
3. Compute precision-recall curve
4. Compute area under precision-recall curve (AUC)

#### 2. nuScenes Detection Score (NDS)

A composite metric that weighs both detection quality and localization accuracy:

```
NDS = (1/10) * [5 * mAP + sum(TP_metrics)]

Where TP_metrics = {1-min(1,mATE), 1-min(1,mASE), 1-min(1,mAOE), 
                    1-min(1,mAVE), 1-min(1,mAAE)}

Range: [0, 1], higher is better
```

### True Positive (TP) Metrics

Computed over true positive detections (matched at 2m center distance threshold):

| Metric | Full Name | Measures | Unit | Perfect Score |
|--------|-----------|----------|------|---------------|
| mATE | Mean Average Translation Error | Position accuracy | meters | 0.0 |
| mASE | Mean Average Scale Error | Size accuracy | 1 - IoU | 0.0 |
| mAOE | Mean Average Orientation Error | Heading accuracy | radians | 0.0 |
| mAVE | Mean Average Velocity Error | Velocity accuracy | m/s | 0.0 |
| mAAE | Mean Average Attribute Error | Attribute accuracy | 1 - acc | 0.0 |

#### Metric Computation Details

**mATE** (Translation Error):
```
ATE(pred, gt) = sqrt((cx_pred - cx_gt)^2 + (cy_pred - cy_gt)^2 + (cz_pred - cz_gt)^2)
mATE = mean over all TP detections of ATE
```

**mASE** (Scale Error):
```
ASE(pred, gt) = 1 - IoU_3D(pred_box_aligned, gt_box_aligned)
# Boxes are aligned (centered + oriented) before computing 3D IoU
# This isolates scale error from position/rotation error
mASE = mean over all TP detections of ASE
```

**mAOE** (Orientation Error):
```
AOE(pred, gt) = min(|yaw_pred - yaw_gt|, 2*pi - |yaw_pred - yaw_gt|)
# Symmetric objects (barrier, traffic_cone) are excluded from mAOE
mAOE = mean over applicable TP detections of AOE
```

**mAVE** (Velocity Error):
```
AVE(pred, gt) = sqrt((vx_pred - vx_gt)^2 + (vy_pred - vy_gt)^2)
# Static classes (barrier, traffic_cone) are excluded from mAVE
mAVE = mean over applicable TP detections of AVE
```

**mAAE** (Attribute Error):
```
# Attributes: pedestrian state (sitting/standing/lying), vehicle state (moving/parked/stopped)
AAE(pred, gt) = 1 if attribute_pred != attribute_gt else 0
mAAE = mean over applicable TP detections of AAE
```

---

## Evaluation Commands

### Standard Evaluation

```bash
# Evaluate a trained model
python tools/test.py \
    configs/petr_r50_nuscenes.yaml \
    work_dirs/petr_r50/epoch_24.pth \
    --eval mAP

# Evaluate with all metrics
python tools/test.py \
    configs/petr_r50_nuscenes.yaml \
    work_dirs/petr_r50/epoch_24.pth \
    --eval mAP NDS
```

### Streaming Evaluation (StreamPETR)

```bash
# Evaluate in streaming mode (sequential frame processing)
python tools/test_stream.py \
    configs/stream_petr_r50_nuscenes.yaml \
    work_dirs/stream_petr_r50/epoch_24.pth \
    --eval mAP NDS \
    --streaming
```

Streaming evaluation processes frames sequentially within each scene, propagating queries as in deployment. This gives a more realistic performance estimate.

### Test-Time Augmentation (TTA)

```bash
# Evaluate with horizontal flip TTA
python tools/test.py \
    configs/petr_r50_nuscenes.yaml \
    work_dirs/petr_r50/epoch_24.pth \
    --eval mAP \
    --tta flip
```

---

## Streaming Evaluation Protocol

### Why Streaming Evaluation Matters

Standard evaluation processes each frame independently. For temporal models (PETRv2, StreamPETR), this underestimates real performance because:
1. Temporal models need sequential frame access
2. Query propagation state must be maintained across frames
3. First frame of a scene has no temporal context

### Protocol

```
For each scene in val set:
  1. Reset temporal state (clear propagated queries / feature memory)
  2. Process frames in chronological order:
     frame_0: detect with random queries (no temporal context)
     frame_1: detect with propagated queries from frame_0
     frame_2: detect with propagated queries from frame_1
     ...
  3. Collect detections from ALL frames (including first)
  4. Evaluate using nuScenes metrics

Key differences from standard evaluation:
  - First frame of each scene has degraded performance (no temporal context)
  - Performance improves after 2-3 frames as queries stabilize
  - This matches real deployment where the system starts "cold"
```

### Warm-up Effect

```
Typical StreamPETR performance by frame position in scene:

Frame Position | mAP  | NDS  | Notes
─────────────────────────────────────────────
Frame 0        | 31.5 | 38.2 | No temporal (same as PETR)
Frame 1        | 35.8 | 42.1 | First propagation
Frame 2        | 37.5 | 44.0 | Stabilizing
Frame 3+       | 38.4 | 44.9 | Steady-state performance
─────────────────────────────────────────────
Overall (all)  | 37.8 | 44.3 | Averaged across all frames
```

---

## Performance Benchmarks

### PETR Family Comparison (nuScenes val, ResNet-50)

| Model | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE | FPS |
|-------|-----|-----|------|------|------|------|------|-----|
| PETR | 31.3 | 38.1 | 0.768 | 0.278 | 0.564 | 0.923 | 0.225 | ~10 |
| PETRv2 | 34.6 | 42.1 | 0.700 | 0.274 | 0.490 | 0.413 | 0.200 | ~8 |
| StreamPETR | 38.4 | 44.9 | 0.660 | 0.270 | 0.440 | 0.370 | 0.195 | ~30 |

### Comparison with Other Methods (nuScenes val)

| Model | Backbone | mAP | NDS | FPS | Memory |
|-------|----------|-----|-----|-----|--------|
| DETR3D | ResNet-101 | 34.9 | 42.2 | ~12 | ~8 GB |
| BEVFormer-S | ResNet-50 | 37.5 | 44.8 | ~4 | ~12 GB |
| BEVFormer-B | ResNet-101-DCN | 41.6 | 51.7 | ~2 | ~18 GB |
| PETR | ResNet-50 | 31.3 | 38.1 | ~10 | ~6 GB |
| PETRv2 | ResNet-50 | 34.6 | 42.1 | ~8 | ~10 GB |
| StreamPETR | ResNet-50 | 38.4 | 44.9 | ~30 | ~8 GB |
| StreamPETR | VoVNet-99 | 45.0 | 55.0 | ~15 | ~16 GB |

### nuScenes Test Leaderboard Results

| Model | Backbone | mAP | NDS | Submission |
|-------|----------|-----|-----|-----------|
| StreamPETR | ViT-L (EVA02) | 55.2 | 63.6 | ICCV 2023 |
| StreamPETR | VoVNet-99 | 45.0 | 55.0 | ICCV 2023 |
| PETRv2 | VoVNet-99 | 42.1 | 52.4 | ICCV 2023 |
| PETR | VoVNet-99 | 37.8 | 44.2 | ECCV 2022 |

---

## FPS Benchmarks

### Measurement Protocol

```
Hardware: Single NVIDIA A100 40GB GPU
Software: PyTorch 1.12, CUDA 11.6, cuDNN 8.4
Settings: FP16 inference, batch_size=1
Input: 6 cameras x [900, 1600] resolution
Warmup: 50 iterations (excluded from timing)
Measured: 200 iterations average
```

### Detailed Timing Breakdown

#### PETR (ResNet-50) - ~10 FPS

| Stage | Time (ms) | % of Total |
|-------|----------|-----------|
| Data loading + preprocessing | 5 | 5% |
| Backbone (ResNet-50) | 15 | 16% |
| FPN | 3 | 3% |
| 3D PE generation | 5 | 5% |
| Decoder cross-attention (6 layers) | 60 | 63% |
| Detection head | 2 | 2% |
| Post-processing (NMS) | 5 | 5% |
| **Total** | **95** | **100%** |

**Bottleneck**: Global cross-attention over 178K tokens is O(900 * 178K) per layer.

#### StreamPETR (ResNet-50) - ~30 FPS

| Stage | Time (ms) | % of Total |
|-------|----------|-----------|
| Data loading + preprocessing | 5 | 15% |
| Backbone (ResNet-50) | 15 | 45% |
| FPN | 3 | 9% |
| 3D PE generation | 3 | 9% |
| Query propagation + ego comp | 0.5 | 2% |
| Decoder (6 layers, optimized) | 5 | 15% |
| Detection head | 1 | 3% |
| Post-processing | 1 | 3% |
| **Total** | **33** | **100%** |

**Why StreamPETR is faster**:
1. Propagated queries have good initial positions -> attention converges faster
2. Optimized implementation with flash attention
3. Smaller effective attention due to query quality (less wasted computation)
4. FP16 throughout with no precision-critical operations

---

## Ablation Studies

### PETR: 3D Position Embedding Ablations

| Configuration | mAP | NDS | Change |
|--------------|-----|-----|--------|
| Full PETR (baseline) | 31.3 | 38.1 | - |
| Without 3D PE (2D PE only) | 22.1 | 28.5 | -9.2 mAP |
| Without depth discretization (single depth) | 26.8 | 33.2 | -4.5 mAP |
| Depth bins: 32 (instead of 64) | 30.5 | 37.3 | -0.8 mAP |
| Depth bins: 128 | 31.5 | 38.3 | +0.2 mAP |
| MLP layers: 1 (instead of 2) | 29.8 | 36.5 | -1.5 mAP |
| MLP layers: 3 | 31.2 | 38.0 | -0.1 mAP |
| Without coordinate normalization | 28.9 | 35.1 | -2.4 mAP |

**Key findings**:
- 3D PE is essential (removing it causes -9.2 mAP drop)
- 64 depth bins is a good tradeoff (32 is too coarse, 128 adds minimal benefit)
- 2-layer MLP is sufficient
- Coordinate normalization is important for training stability

### PETRv2: Temporal Ablations

| Configuration | mAP | NDS | mAVE |
|--------------|-----|-----|------|
| PETR (no temporal) | 31.3 | 38.1 | 0.923 |
| + Temporal features (no alignment) | 32.5 | 39.8 | 0.680 |
| + Ego-motion compensation | 34.2 | 41.6 | 0.430 |
| + Feature alignment | 34.6 | 42.1 | 0.413 |
| + 2D PE addition | 35.0 | 42.5 | 0.405 |
| + LID depth discretization | 35.3 | 42.8 | 0.400 |

**Key findings**:
- Ego-motion compensation is crucial (+1.7 mAP over naive temporal)
- Velocity estimation improves dramatically with temporal (0.92 -> 0.41 mAVE)
- 2D PE and LID provide incremental improvements

### StreamPETR: Query Propagation Ablations

| Configuration | mAP | NDS | FPS |
|--------------|-----|-----|-----|
| PETR (no temporal) | 31.3 | 38.1 | 10 |
| + Query propagation (128 queries) | 35.8 | 42.5 | 28 |
| + Query propagation (256 queries) | 37.5 | 44.0 | 30 |
| + Query propagation (512 queries) | 37.8 | 44.2 | 25 |
| + Motion-aware LayerNorm | 38.4 | 44.9 | 30 |
| + Velocity prediction for position | 38.6 | 45.1 | 29 |
| + Memory buffer (512) | 38.8 | 45.3 | 29 |

**Key findings**:
- 256 propagated queries is the sweet spot (more queries = diminishing returns + slower)
- Motion-aware LayerNorm provides ~1 mAP improvement at negligible cost
- Velocity-based position prediction helps moderately
- Memory buffer helps with occluded objects

### StreamPETR: Selection Strategy Ablations

| Selection Strategy | mAP | NDS |
|-------------------|-----|-----|
| Random selection | 34.2 | 40.8 |
| Top-K by confidence | 38.4 | 44.9 |
| NMS then top-K | 38.1 | 44.6 |
| Diversity-aware (FPS) | 37.9 | 44.5 |

**Key findings**:
- Simple top-K by confidence works best
- NMS slightly hurts because it removes nearby queries that could track different objects
- Random selection is much worse (confirms that query quality matters)

---

## Per-Class Performance Analysis

### StreamPETR (R50) Per-Class AP

| Class | AP@0.5m | AP@1m | AP@2m | AP@4m | Mean AP |
|-------|---------|-------|-------|-------|---------|
| car | 35.2 | 55.8 | 68.4 | 72.1 | 57.9 |
| truck | 18.5 | 35.2 | 48.3 | 54.8 | 39.2 |
| construction_vehicle | 4.2 | 12.8 | 22.5 | 29.1 | 17.2 |
| bus | 20.1 | 40.5 | 55.8 | 63.2 | 44.9 |
| trailer | 8.5 | 22.1 | 38.4 | 48.2 | 29.3 |
| barrier | 28.5 | 48.2 | 58.9 | 62.1 | 49.4 |
| motorcycle | 15.8 | 30.2 | 40.5 | 45.8 | 33.1 |
| bicycle | 10.2 | 22.5 | 32.8 | 38.5 | 26.0 |
| pedestrian | 25.8 | 42.5 | 52.1 | 56.8 | 44.3 |
| traffic_cone | 28.2 | 48.5 | 55.2 | 58.1 | 47.5 |

**Observations**:
- Cars have highest AP (most training data, consistent appearance)
- Construction vehicles are hardest (rare, diverse appearance)
- Small objects (bicycle, motorcycle) struggle at tight thresholds
- Performance improves significantly from 0.5m to 4m threshold

---

## Evaluation Tips

### 1. Reproducing Paper Results

```bash
# Ensure deterministic evaluation
export CUBLAS_WORKSPACE_CONFIG=:16:8
python tools/test.py \
    configs/stream_petr_r50_nuscenes.yaml \
    work_dirs/stream_petr_r50/epoch_24.pth \
    --eval mAP NDS \
    --deterministic
```

### 2. Visualizing Detections

```bash
# Generate visualization of predictions
python tools/visualize.py \
    configs/stream_petr_r50_nuscenes.yaml \
    work_dirs/stream_petr_r50/epoch_24.pth \
    --show-bev \
    --show-camera \
    --score-threshold 0.3 \
    --out-dir visualizations/
```

### 3. Analyzing Failure Cases

Common failure modes:
- **Far-range objects (>50m)**: Low feature resolution, few depth bins
- **Heavily occluded objects**: Limited visible features to attend to
- **Fast-moving objects**: Velocity prediction lag for StreamPETR
- **Night/rain scenes**: Domain gap from training data
- **Camera transition zones**: Objects at boundaries between cameras

### 4. Test Set Submission

```bash
# Generate test predictions for nuScenes submission
python tools/test.py \
    configs/stream_petr_r50_nuscenes.yaml \
    work_dirs/stream_petr_r50/epoch_24.pth \
    --format-only \
    --eval-options jsonfile_prefix=results/stream_petr_r50_test

# Upload results/stream_petr_r50_test_results_nusc.json to eval.ai
```

---

## Model Comparison Decision Guide

| Priority | Recommended Model | Reason |
|----------|------------------|--------|
| Highest accuracy | StreamPETR (large backbone) | Best mAP/NDS |
| Real-time (>20 FPS) | StreamPETR (R50) | ~30 FPS with strong accuracy |
| Low memory (<8 GB) | PETR (R50) | Minimal memory footprint |
| Multi-task (det + seg) | PETRv2 | Built-in multi-task framework |
| Simple deployment | PETR | No temporal state management |
| Object tracking | StreamPETR | Natural tracking from query identity |
| Velocity estimation | StreamPETR or PETRv2 | Temporal models excel at velocity |
