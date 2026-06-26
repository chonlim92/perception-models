# Evaluation Guide: RadarPillarNet

## 1. Overview

This document describes the evaluation methodology, metrics, and expected results for
the RadarPillarNet model on the nuScenes 3D object detection benchmark. It covers the
official nuScenes metrics, distance-stratified evaluation, comparison with baselines,
and ablation study results.

## 2. nuScenes Detection Metrics

### 2.1 Primary Metrics

The nuScenes detection benchmark uses a comprehensive set of metrics:

| Metric | Full Name | Description | Unit | Better |
|--------|-----------|-------------|------|--------|
| mAP | Mean Average Precision | Detection quality across BEV distance thresholds | % | Higher |
| NDS | nuScenes Detection Score | Weighted combination of mAP and TP metrics | % | Higher |
| mATE | Mean Average Translation Error | Euclidean distance (2D center) | meters | Lower |
| mASE | Mean Average Scale Error | 1 - IoU (after alignment) | 0-1 | Lower |
| mAOE | Mean Average Orientation Error | Smallest yaw angle difference | radians | Lower |
| mAVE | Mean Average Velocity Error | L2 velocity error | m/s | Lower |
| mAAE | Mean Average Attribute Error | 1 - attribute accuracy | 0-1 | Lower |

### 2.2 NDS Calculation

The nuScenes Detection Score combines all metrics:

```
NDS = (1/10) * [5 * mAP + sum(1 - min(1, metric_i)) for metric_i in TP_metrics]

Where TP_metrics = [mATE, mASE, mAOE, mAVE, mAAE]
Each TP metric is capped at 1.0 before computing (1 - metric)
```

### 2.3 mAP Calculation

Average Precision is computed differently from KITTI:

- Match threshold is based on BEV center distance (not IoU)
- Distance thresholds: {0.5, 1.0, 2.0, 4.0} meters
- mAP is the mean across all classes and all distance thresholds

```
AP_class = mean([AP@0.5m, AP@1.0m, AP@2.0m, AP@4.0m])
mAP = mean([AP_class for class in all_classes])
```

## 3. Running Evaluation

### 3.1 Evaluation Command

```bash
# Standard evaluation on validation set
python scripts/evaluate.py \
    --config configs/radar_pillarnet_nuscenes.yaml \
    --checkpoint checkpoints/radar_pillarnet_epoch80.pth \
    --dataroot /data/nuscenes \
    --version v1.0-trainval \
    --eval_set val \
    --output_dir results/

# Generate submission file for test server
python scripts/evaluate.py \
    --config configs/radar_pillarnet_nuscenes.yaml \
    --checkpoint checkpoints/radar_pillarnet_epoch80.pth \
    --dataroot /data/nuscenes \
    --version v1.0-test \
    --eval_set test \
    --output_dir results/submission/ \
    --format submission
```

### 3.2 Output Format

The evaluation script produces:

```
results/
├── detection_results.json    # nuScenes format predictions
├── metrics_summary.json      # Overall metrics
├── per_class_metrics.json    # Per-class breakdown
├── distance_metrics.json     # Distance-stratified results
└── visualizations/           # Optional visualization outputs
```

## 4. Expected Results

### 4.1 Overall Performance (nuScenes val)

| Model | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
|-------|-----|-----|------|------|------|------|------|
| RadarPillarNet (6 sweeps) | 23.4 | 35.8 | 0.72 | 0.28 | 0.58 | 0.89 | 0.21 |
| RadarPillarNet (10 sweeps) | 25.1 | 37.2 | 0.69 | 0.27 | 0.55 | 0.82 | 0.20 |

### 4.2 Per-Class Results (6 sweeps)

| Class | AP | ATE | ASE | AOE | AVE | AAE |
|-------|-----|------|------|------|------|------|
| car | 42.1 | 0.52 | 0.17 | 0.34 | 0.78 | 0.15 |
| truck | 22.8 | 0.68 | 0.23 | 0.42 | 1.02 | 0.19 |
| bus | 28.5 | 0.71 | 0.21 | 0.28 | 1.31 | 0.24 |
| trailer | 12.3 | 1.05 | 0.25 | 0.68 | 0.54 | 0.18 |
| construction_vehicle | 8.7 | 1.12 | 0.48 | 1.21 | 0.13 | 0.35 |
| pedestrian | 20.5 | 0.78 | 0.30 | 1.02 | 0.85 | 0.28 |
| motorcycle | 18.2 | 0.72 | 0.26 | 0.72 | 1.45 | 0.15 |
| bicycle | 9.8 | 0.68 | 0.28 | 0.88 | 0.42 | 0.12 |
| traffic_cone | 6.2 | 0.61 | 0.35 | N/A | N/A | N/A |
| barrier | 14.8 | 0.59 | 0.29 | 0.15 | N/A | N/A |

### 4.3 Distance-Stratified Results

| Distance Range | mAP | NDS | Avg Points per Object |
|---------------|-----|-----|----------------------|
| 0-30 m | 35.2 | 45.1 | 8.5 |
| 30-50 m | 19.8 | 32.4 | 3.2 |
| 50-70 m | 10.1 | 24.6 | 1.4 |
| 70+ m | 4.2 | 18.3 | 0.6 |

## 5. Comparison with Baselines

### 5.1 Radar Methods

| Method | Modality | mAP | NDS |
|--------|----------|-----|-----|
| PointPillars (radar, vanilla) | Radar (1 sweep) | 8.2 | 18.6 |
| PointPillars (radar, 6 sweeps) | Radar (6 sweeps) | 18.9 | 30.2 |
| **RadarPillarNet** | Radar (6 sweeps) | **23.4** | **35.8** |
| RPFA-Net | Radar (6 sweeps) | 24.8 | 36.5 |

### 5.2 LiDAR Methods (Upper Bound Reference)

| Method | Modality | mAP | NDS |
|--------|----------|-----|-----|
| PointPillars | LiDAR | 40.1 | 55.0 |
| SECOND | LiDAR | 44.8 | 58.3 |
| CenterPoint | LiDAR | 56.4 | 64.8 |
| TransFusion-L | LiDAR | 65.5 | 70.2 |

### 5.3 Performance Gap Analysis

The radar-LiDAR performance gap (~30 NDS points) is primarily due to:

1. **Sparsity:** 100x fewer points limits shape recovery and small object detection
2. **Elevation:** Limited vertical information affects height and size estimation (mASE)
3. **Angular resolution:** Coarse azimuth causes poor localization (mATE)
4. **Ghost detections:** False positives from multipath reduce precision

## 6. Ablation Studies

### 6.1 Multi-Sweep Impact

| Sweeps | mAP | NDS | Delta NDS |
|--------|-----|-----|-----------|
| 1 | 8.2 | 18.6 | baseline |
| 3 | 16.5 | 27.4 | +8.8 |
| 6 | 23.4 | 35.8 | +17.2 |
| 10 | 25.1 | 37.2 | +18.6 |
| 15 | 24.8 | 36.9 | +18.3 |

Observations:
- Massive improvement from 1 to 6 sweeps (+17.2 NDS)
- Diminishing returns beyond 10 sweeps
- Slight degradation at 15 sweeps due to temporal smearing of moving objects

### 6.2 Feature Ablation

| Features Used | mAP | NDS | Delta |
|--------------|-----|-----|-------|
| x, y, z only | 14.2 | 24.1 | baseline |
| + RCS | 16.8 | 27.5 | +3.4 |
| + vx, vy | 21.2 | 33.6 | +9.5 |
| + dt | 22.5 | 34.9 | +10.8 |
| + xc, yc (full) | 23.4 | 35.8 | +11.7 |

Key insight: Velocity features provide the single largest improvement (+6.1 NDS over
position+RCS), confirming the importance of radar's Doppler measurements.

### 6.3 Augmentation Ablation

| Configuration | mAP | NDS |
|--------------|-----|-----|
| No augmentation | 15.8 | 26.2 |
| + Random flip | 17.9 | 29.1 |
| + Global rotation | 19.5 | 31.4 |
| + Global scaling | 20.1 | 32.0 |
| + GT-sampling | 23.4 | 35.8 |

GT-sampling contributes +3.8 NDS, making it the single most impactful augmentation
strategy. This is because it directly addresses radar's sparsity by ensuring more
diverse training examples with sufficient point density.

### 6.4 Pillar Resolution

| Resolution (m) | Grid Size | mAP | NDS | Inference (ms) |
|---------------|-----------|-----|-----|----------------|
| 0.10 | 1024x1024 | 24.1 | 36.2 | 38 |
| 0.15 | 683x683 | 23.8 | 36.0 | 22 |
| 0.20 | 512x512 | 23.4 | 35.8 | 15 |
| 0.30 | 341x341 | 22.1 | 34.5 | 10 |
| 0.40 | 256x256 | 20.5 | 32.8 | 7 |

The 0.20m resolution provides the best trade-off between accuracy and speed for radar.
Finer resolutions show minimal improvement since most pillars are empty anyway.

## 7. Visualization

### 7.1 Generating Visualizations

```bash
# BEV visualization with detections
python scripts/visualize.py \
    --config configs/radar_pillarnet_nuscenes.yaml \
    --checkpoint checkpoints/radar_pillarnet_epoch80.pth \
    --dataroot /data/nuscenes \
    --sample_token <token> \
    --output_dir visualizations/ \
    --show_gt \
    --show_radar_points

# Batch visualization
python scripts/visualize.py \
    --config configs/radar_pillarnet_nuscenes.yaml \
    --checkpoint checkpoints/radar_pillarnet_epoch80.pth \
    --dataroot /data/nuscenes \
    --num_samples 50 \
    --output_dir visualizations/batch/
```

### 7.2 Visualization Color Coding

```
Ground truth boxes: Green
Predicted boxes:    Red (score > 0.3), Orange (0.1 < score < 0.3)
Radar points:       Blue (current sweep), Cyan (historical sweeps)
Velocity vectors:   Yellow arrows (ground truth), Magenta arrows (predicted)
```

## 8. Error Analysis

### 8.1 Common Failure Modes

1. **Missed pedestrians:** Low RCS and few radar returns make pedestrians the hardest class
2. **Size estimation errors:** Without LiDAR's shape information, size regression relies on priors
3. **Ghost detections near guardrails:** Multipath creates systematic false positives
4. **Heading ambiguity:** Radar cannot distinguish front from back of symmetric objects
5. **Merging close objects:** Coarse angular resolution merges nearby targets

### 8.2 Improvement Strategies

- **For mAP:** Focus on recall at easy thresholds (2m, 4m) through better GT-sampling
- **For mATE:** Improve pillar resolution or add spatial attention mechanisms
- **For mAOE:** Leverage velocity direction as a heading cue
- **For mAVE:** Increase velocity loss weight, verify ego-motion compensation
- **For NDS:** Multi-sweep count and velocity features have the highest impact
