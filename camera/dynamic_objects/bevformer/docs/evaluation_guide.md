# BEVFormer: Evaluation Guide

## Evaluation Metrics, Results, and Benchmarks

This document provides comprehensive information about evaluating BEVFormer using the nuScenes detection metrics, interpreting results, and comparing to other methods.

---

## 1. nuScenes Detection Metrics

### 1.1 Overview

The nuScenes detection benchmark uses two primary metrics:

| Metric | Full Name | Description |
|--------|-----------|-------------|
| **NDS** | nuScenes Detection Score | Holistic score combining mAP + TP metrics |
| **mAP** | mean Average Precision | Detection accuracy across distance thresholds |

**NDS Formula:**
```
NDS = (1/10) * [5 * mAP + sum(1 - min(1, TP_metric)) for each TP metric]
    = (1/10) * [5 * mAP + (1 - mATE) + (1 - mASE) + (1 - mAOE) + (1 - mAVE) + (1 - mAAE)]
```

Where TP metrics are capped at 1.0 (no bonus for being better than perfect).

### 1.2 Mean Average Precision (mAP)

#### Distance-Based Matching

Unlike 2D detection (IoU-based matching), nuScenes uses **center distance** for matching predictions to ground truth:

| Distance Threshold | Description |
|-------------------|-------------|
| 0.5m | Very strict (primarily affects small objects) |
| 1.0m | Strict |
| 2.0m | Moderate |
| 4.0m | Lenient |

#### mAP Computation

```
For each class c:
    For each distance threshold d in [0.5, 1.0, 2.0, 4.0]:
        1. Match predictions to GT by center distance < d
        2. Compute precision-recall curve
        3. Compute AP (area under interpolated P-R curve)
    AP_c = mean(AP_d for d in thresholds)

mAP = mean(AP_c for c in classes)
```

#### Key Differences from KITTI/Waymo

- **Center distance** instead of IoU for matching
- **Multiple thresholds** averaged (not single IoU threshold)
- **No orientation** requirement for matching (orientation evaluated separately)
- **Global frame** distances (not relative to ego)

### 1.3 True Positive (TP) Metrics

For all true positive detections (correctly matched), nuScenes evaluates quality:

| Metric | Full Name | Unit | Range | What It Measures |
|--------|-----------|------|-------|------------------|
| **mATE** | mean Average Translation Error | meters | [0, inf) | Center position accuracy |
| **mASE** | mean Average Scale Error | ratio | [0, 1] | Size accuracy (1 - IoU of aligned boxes) |
| **mAOE** | mean Average Orientation Error | radians | [0, pi] | Heading angle accuracy |
| **mAVE** | mean Average Velocity Error | m/s | [0, inf) | Velocity estimation accuracy |
| **mAAE** | mean Average Attribute Error | ratio | [0, 1] | Attribute classification accuracy |

#### Detailed TP Metric Definitions

**mATE (Translation Error):**
```
ATE = Euclidean distance between predicted and GT center (2D, x-y plane)
    = sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)

mATE = mean(ATE) over all true positive matches, across all classes
```

**mASE (Scale Error):**
```
ASE = 1 - IoU(pred_box_aligned, gt_box_aligned)
    where boxes are aligned at center and orientation (only size differs)

mASE = mean(ASE) over all true positive matches
```

**mAOE (Orientation Error):**
```
AOE = |angle_diff(yaw_pred, yaw_gt)|
    = min(|yaw_pred - yaw_gt|, 2*pi - |yaw_pred - yaw_gt|)

Note: For symmetric objects (barriers, traffic cones), orientation
is measured modulo pi (180-degree ambiguity allowed)

mAOE = mean(AOE) over all true positive matches
```

**mAVE (Velocity Error):**
```
AVE = L2 norm of velocity difference
    = sqrt((vx_pred - vx_gt)^2 + (vy_pred - vy_gt)^2)

mAVE = mean(AVE) over all true positive matches
Note: Only computed for classes that can move (vehicles, pedestrians, cyclists)
      Static classes (barrier, traffic_cone) excluded
```

**mAAE (Attribute Error):**
```
AAE = 1 - accuracy(attribute_pred, attribute_gt)
    = fraction of TPs with incorrect attribute prediction

mAAE = mean(AAE) over all applicable classes
```

---

## 2. Running Evaluation

### 2.1 Evaluation Command

```bash
# Standard evaluation on validation set
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    8 \
    --eval bbox

# Single-GPU evaluation
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --eval bbox \
    --gpu-ids 0

# Evaluation with visualization
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --eval bbox \
    --show-dir work_dirs/bevformer_base/visualizations/
```

### 2.2 Evaluation Output Format

```
---------- nuScenes Detection Evaluation ----------
mAP: 0.4163
mATE: 0.6728
mASE: 0.2731
mAOE: 0.3718
mAVE: 0.3944
mAAE: 0.1981
NDS: 0.5170

---------- Per-Class Results ----------
         Class    AP    ATE    ASE    AOE    AVE    AAE
           car 0.594  0.462  0.154  0.081  0.359  0.177
         truck 0.388  0.692  0.207  0.096  0.348  0.198
           bus 0.445  0.723  0.197  0.051  0.846  0.245
       trailer 0.205  1.040  0.243  0.557  0.232  0.092
construction_v 0.091  1.058  0.481  1.125  0.121  0.362
    pedestrian 0.449  0.704  0.295  0.592  0.432  0.216
    motorcycle 0.393  0.616  0.261  0.449  0.601  0.003
       bicycle 0.331  0.607  0.270  0.661  0.255  0.009
       barrier 0.534  0.557  0.284  0.136  nan    nan
  traffic_cone 0.532  0.404  0.342  nan    nan    nan
```

### 2.3 Generating Submission File

```bash
# For official leaderboard submission
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --format-only \
    --eval-options jsonfile_prefix=results/bevformer_submit

# Creates: results/bevformer_submit/results_nusc.json
# Submit this file to https://eval.ai/web/challenges/challenge-page/356/
```

---

## 3. BEVFormer Results

### 3.1 Main Results (nuScenes Validation Set)

| Model | Backbone | Epochs | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|-------|----------|--------|-----|-----|------|------|------|------|------|
| BEVFormer-Small | R101 | 24 | 0.462 | 0.349 | 0.725 | 0.279 | 0.407 | 0.521 | 0.209 |
| BEVFormer-Base | R101-DCN | 24 | 0.517 | 0.416 | 0.673 | 0.274 | 0.372 | 0.394 | 0.198 |

### 3.2 Main Results (nuScenes Test Set)

| Model | Backbone | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|-------|----------|-----|-----|------|------|------|------|------|
| BEVFormer-Small | R101 | 0.478 | 0.370 | 0.698 | 0.281 | 0.423 | 0.497 | 0.208 |
| BEVFormer-Base | R101-DCN | 0.569 | 0.481 | 0.582 | 0.256 | 0.375 | 0.378 | 0.126 |
| BEVFormer-Large | V2-99 | 0.592 | 0.517 | 0.549 | 0.253 | 0.358 | 0.322 | 0.118 |

### 3.3 Per-Class AP (BEVFormer-Base, Val Set)

| Class | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m | AP (mean) |
|-------|---------|---------|---------|---------|-----------|
| car | 0.321 | 0.564 | 0.726 | 0.765 | 0.594 |
| truck | 0.143 | 0.335 | 0.509 | 0.564 | 0.388 |
| bus | 0.171 | 0.404 | 0.580 | 0.626 | 0.445 |
| trailer | 0.032 | 0.125 | 0.293 | 0.369 | 0.205 |
| construction_vehicle | 0.012 | 0.054 | 0.130 | 0.169 | 0.091 |
| pedestrian | 0.201 | 0.425 | 0.571 | 0.600 | 0.449 |
| motorcycle | 0.177 | 0.372 | 0.495 | 0.527 | 0.393 |
| bicycle | 0.149 | 0.308 | 0.422 | 0.447 | 0.331 |
| barrier | 0.254 | 0.499 | 0.656 | 0.726 | 0.534 |
| traffic_cone | 0.289 | 0.521 | 0.651 | 0.669 | 0.532 |

### 3.4 Distance-Binned Performance

| Distance | mAP | mATE | Notes |
|----------|-----|------|-------|
| 0-20m | 0.58 | 0.38 | Best performance (high resolution) |
| 20-40m | 0.44 | 0.62 | Good performance |
| 40-60m | 0.31 | 0.89 | Degraded (lower resolution) |
| 60-80m | 0.18 | 1.21 | Significantly degraded |
| 80-100m | 0.09 | 1.65 | Poor (very few pixels) |

---

## 4. Ablation Studies

### 4.1 Effect of Temporal Frames

| Temporal Frames | NDS | mAP | mAVE | Delta NDS |
|-----------------|-----|-----|------|-----------|
| 1 (no temporal) | 0.492 | 0.390 | 0.842 | baseline |
| 2 (current + 1 prev) | 0.505 | 0.403 | 0.468 | +1.3 |
| 3 | 0.512 | 0.410 | 0.412 | +2.0 |
| 4 (default) | 0.517 | 0.416 | 0.394 | +2.5 |
| 8 | 0.519 | 0.418 | 0.381 | +2.7 |

**Key insight:** Temporal fusion most dramatically improves velocity estimation (mAVE drops from 0.842 to 0.394). Diminishing returns beyond 4 frames.

### 4.2 BEV Resolution

| BEV Size | Resolution | NDS | mAP | Memory | Speed |
|----------|-----------|-----|-----|--------|-------|
| 50 x 50 | 2.048 m/cell | 0.451 | 0.342 | 6 GB | 15 FPS |
| 100 x 100 | 1.024 m/cell | 0.489 | 0.383 | 10 GB | 12 FPS |
| 150 x 150 | 0.683 m/cell | 0.507 | 0.405 | 14 GB | 10 FPS |
| 200 x 200 | 0.512 m/cell | 0.517 | 0.416 | 18 GB | 9 FPS |
| 300 x 300 | 0.341 m/cell | 0.521 | 0.420 | 32 GB | 5 FPS |

**Key insight:** 200x200 provides the best accuracy/efficiency trade-off. Higher resolutions give marginal gains with significant memory and speed costs.

### 4.3 Number of Encoder Layers

| Encoder Layers | NDS | mAP | Parameters | Speed |
|----------------|-----|-----|-----------|-------|
| 1 | 0.467 | 0.357 | 65.0M | 14 FPS |
| 2 | 0.485 | 0.380 | 66.0M | 13 FPS |
| 3 | 0.498 | 0.396 | 67.0M | 11 FPS |
| 4 | 0.508 | 0.408 | 68.0M | 10 FPS |
| 6 (default) | 0.517 | 0.416 | 70.0M | 9 FPS |
| 8 | 0.519 | 0.418 | 72.0M | 7 FPS |

**Key insight:** Performance saturates around 6 layers. The encoder is the most computationally expensive component.

### 4.4 Number of Reference Points (Heights)

| N_ref (heights) | NDS | mAP | Description |
|-----------------|-----|-----|-------------|
| 1 | 0.487 | 0.381 | Single height (ground plane) |
| 2 | 0.502 | 0.400 | Two heights |
| 4 (default) | 0.517 | 0.416 | Four heights |
| 8 | 0.518 | 0.417 | Eight heights (diminishing returns) |

### 4.5 Backbone Comparison

| Backbone | Pretrain | NDS | mAP | Backbone Params | FLOPs |
|----------|----------|-----|-----|-----------------|-------|
| ResNet-50 | ImageNet | 0.462 | 0.349 | 25.6M | 4.1G |
| ResNet-101 | ImageNet | 0.478 | 0.370 | 44.5M | 7.8G |
| ResNet-101-DCN | FCOS3D | 0.517 | 0.416 | 44.5M | 8.2G |
| V2-99 | DD3D | 0.535 | 0.440 | 52.7M | 14.3G |

**Key insight:** Pretrained weights (FCOS3D or DD3D) significantly impact final performance (+3-5 NDS over ImageNet pretraining).

### 4.6 Effect of Key Components

| Configuration | NDS | mAP | Delta |
|--------------|-----|-----|-------|
| Full BEVFormer-Base | 0.517 | 0.416 | - |
| Without temporal self-attention | 0.492 | 0.390 | -2.5 |
| Without deformable attention (use full attention) | 0.509 | 0.410 | -0.8 |
| Without multi-scale features (single scale) | 0.498 | 0.393 | -1.9 |
| Without grid mask augmentation | 0.507 | 0.404 | -1.0 |
| Without CBGS | 0.503 | 0.398 | -1.4 |
| Without can_bus (ego-motion) | 0.494 | 0.391 | -2.3 |

---

## 5. Temporal Consistency Metrics

### 5.1 Tracking-Based Evaluation

While BEVFormer is primarily a detection model, temporal consistency can be evaluated:

```bash
# Run tracking on BEVFormer detections
python tools/track.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --eval track
```

### 5.2 Consistency Metrics

| Metric | Description | BEVFormer-Base |
|--------|-------------|----------------|
| AMOTA | Average Multi-Object Tracking Accuracy | 0.412 |
| AMOTP | Average Multi-Object Tracking Precision | 1.132 |
| ID Switches | Number of identity switches | 842 |
| Fragmentation | Track fragmentation count | 1,247 |
| Velocity Consistency | Std of velocity across frames | 0.52 m/s |

### 5.3 Temporal vs. Single-Frame Detection

| Metric | Single Frame | With Temporal | Improvement |
|--------|-------------|---------------|-------------|
| False Positives / frame | 4.2 | 3.1 | -26% |
| False Negatives / frame | 5.8 | 4.9 | -16% |
| Box Jitter (mATE std) | 0.31m | 0.19m | -39% |
| Velocity Error | 0.842 m/s | 0.394 m/s | -53% |
| Heading Jitter | 0.12 rad | 0.08 rad | -33% |

---

## 6. Benchmark Comparison

### 6.1 Camera-Only Methods (nuScenes Test Set)

| Method | Year | Backbone | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|--------|------|----------|-----|-----|------|------|------|------|------|
| FCOS3D | 2021 | R101 | 0.428 | 0.358 | 0.690 | 0.249 | 0.452 | 1.434 | 0.124 |
| DETR3D | 2022 | V2-99 | 0.479 | 0.412 | 0.641 | 0.255 | 0.394 | 0.845 | 0.133 |
| PETR | 2022 | V2-99 | 0.504 | 0.441 | 0.593 | 0.249 | 0.383 | 0.808 | 0.132 |
| BEVDet | 2022 | Swin-B | 0.488 | 0.396 | 0.556 | 0.239 | 0.414 | 0.819 | 0.140 |
| BEVDet4D | 2022 | Swin-B | 0.515 | 0.421 | 0.517 | 0.241 | 0.386 | 0.556 | 0.138 |
| **BEVFormer** | **2022** | **R101-DCN** | **0.569** | **0.481** | **0.582** | **0.256** | **0.375** | **0.378** | **0.126** |
| **BEVFormer** | **2022** | **V2-99** | **0.592** | **0.517** | **0.549** | **0.253** | **0.358** | **0.322** | **0.118** |
| PolarFormer | 2023 | V2-99 | 0.572 | 0.493 | 0.556 | 0.256 | 0.364 | 0.440 | 0.127 |
| SOLOFusion | 2023 | R101-DCN | 0.582 | 0.483 | 0.503 | 0.264 | 0.381 | 0.246 | 0.207 |
| StreamPETR | 2023 | V2-99 | 0.592 | 0.504 | 0.540 | 0.247 | 0.370 | 0.283 | 0.120 |

### 6.2 LiDAR-Based Methods (for reference)

| Method | Year | NDS | mAP | Gap to BEVFormer |
|--------|------|-----|-----|------------------|
| CenterPoint | 2021 | 0.673 | 0.603 | +8.1 NDS |
| TransFusion-L | 2022 | 0.702 | 0.652 | +11.0 NDS |
| LargeKernel3D | 2023 | 0.714 | 0.657 | +12.2 NDS |

### 6.3 Multi-Modal Methods (Camera + LiDAR)

| Method | Year | NDS | mAP | Sensors |
|--------|------|-----|-----|---------|
| BEVFusion | 2022 | 0.714 | 0.685 | Camera + LiDAR |
| TransFusion | 2022 | 0.718 | 0.682 | Camera + LiDAR |
| DeepInteraction | 2023 | 0.726 | 0.697 | Camera + LiDAR |

### 6.4 Key Observations

1. **BEVFormer leads camera-only methods** at time of publication, especially in velocity estimation (mAVE)
2. **Gap to LiDAR** remains significant (~10 NDS) but BEVFormer represents a major step in closing it
3. **Temporal fusion** is the key differentiator vs. single-frame methods (DETR3D, PETR)
4. **Subsequent methods** (StreamPETR, SOLOFusion) match or slightly exceed BEVFormer, often building on its insights

---

## 7. Per-Class Analysis

### 7.1 Strengths

| Class | BEVFormer AP | Why BEVFormer Works Well |
|-------|-------------|--------------------------|
| car | 0.594 | Large, abundant, consistent appearance |
| barrier | 0.534 | Static, regular shape, high contrast |
| traffic_cone | 0.532 | Distinctive color, predictable size |
| pedestrian | 0.449 | Temporal helps with velocity, common class |

### 7.2 Weaknesses

| Class | BEVFormer AP | Why It Struggles |
|-------|-------------|------------------|
| construction_vehicle | 0.091 | Rare, highly variable appearance |
| trailer | 0.205 | Large, often partially occluded |
| bicycle | 0.331 | Small, rare, fast-moving |
| motorcycle | 0.393 | Similar to bicycle, higher speed |

### 7.3 Failure Mode Analysis

| Failure Mode | Affected Classes | Frequency | Potential Fix |
|--------------|-----------------|-----------|---------------|
| Distance > 60m | All | 20% of FN | Higher resolution BEV at distance |
| Heavy occlusion | truck, trailer | 15% of FN | Longer temporal window |
| Night/low light | pedestrian, bicycle | 10% of FN | Better augmentation, night pretraining |
| Similar objects | motorcycle/bicycle | 8% of FP | Attribute-aware detection |
| Calibration drift | All | 5% of errors | Online calibration refinement |

---

## 8. Evaluation Best Practices

### 8.1 Fair Comparison Checklist

When comparing BEVFormer to other methods, ensure:

- [ ] Same dataset split (v1.0-trainval, train/val division)
- [ ] Same input resolution (900x1600 unless explicitly noted)
- [ ] Same backbone pretraining (FCOS3D pretrained vs. ImageNet only)
- [ ] Same training epochs (24)
- [ ] Same data augmentation (grid mask, photometric distortion)
- [ ] Test-time augmentation noted if used (flip, multi-scale)
- [ ] Single model vs. ensemble clearly stated

### 8.2 Statistical Significance

- Single-run results can vary by ±0.5 NDS due to random initialization
- For publishable comparisons, report mean ± std over 3 runs
- Differences < 1.0 NDS may not be statistically significant

### 8.3 Efficiency Metrics

When reporting results, also consider:

| Metric | BEVFormer-Base | Description |
|--------|---------------|-------------|
| FPS | 9.4 | Frames per second (A100) |
| Latency | 106 ms | End-to-end inference time |
| GPU Memory | 18 GB | Training memory (single GPU) |
| Parameters | 70M | Total model parameters |
| FLOPs | ~200G | Per-sample computation |
| Training time | 28h (8x A100) | Total training duration |

### 8.4 Visualization for Debugging

```bash
# Visualize predictions vs. ground truth
python tools/visualize.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --show-dir visualizations/ \
    --show-bev \
    --show-cameras \
    --score-thr 0.3

# Visualize BEV feature maps
python tools/visualize_bev.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --save-dir bev_vis/
```

---

## 9. Evaluation Configuration Reference

### 9.1 Detection Evaluation Config

```python
# Standard detection evaluation settings
eval_config = dict(
    # Detection settings
    class_range={
        'car': 50,                    # meters (max range for evaluation)
        'truck': 50,
        'bus': 50,
        'trailer': 50,
        'construction_vehicle': 50,
        'pedestrian': 40,
        'motorcycle': 40,
        'bicycle': 40,
        'barrier': 30,
        'traffic_cone': 30,
    },
    # Distance thresholds for matching
    dist_fcn='center_distance',
    dist_ths=[0.5, 1.0, 2.0, 4.0],
    dist_th_tp=2.0,
    
    # Minimum annotation criteria
    min_recall=0.1,
    min_precision=0.1,
    max_boxes_per_sample=500,
    
    # Velocity evaluation
    max_velocity_error=10.0,  # Cap velocity error at 10 m/s
)
```

### 9.2 Post-Processing Parameters

```python
# Test-time post-processing
test_cfg = dict(
    pts=dict(
        score_threshold=0.0,      # Keep all predictions for evaluation
        max_per_sample=300,       # Maximum predictions per frame
        nms_type=None,            # No NMS (DETR-style)
        # Alternative with NMS:
        # nms_type='circle',
        # nms_thr=[4.0, ...],    # Per-class NMS thresholds
    )
)
```

---

## 10. Reproducing Published Results

### 10.1 Exact Reproduction Steps

```bash
# Step 1: Environment (exact versions)
pip install torch==2.0.1+cu118
pip install mmcv-full==1.7.1
pip install mmdet==2.28.2
pip install mmdet3d==1.0.0rc6

# Step 2: Pretrained backbone
mkdir ckpts
wget -O ckpts/r101_dcn_fcos3d_pretrain.pth \
    <pretrained_model_url>

# Step 3: Data preparation (see data preparation section)
python tools/create_data.py nuscenes ...

# Step 4: Training
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_base.py 8

# Step 5: Evaluation
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth 8 --eval bbox
```

### 10.2 Expected Validation Results (with tolerance)

| Metric | Expected | Acceptable Range |
|--------|----------|-----------------|
| NDS | 0.517 | 0.512 - 0.522 |
| mAP | 0.416 | 0.411 - 0.421 |
| mATE | 0.673 | 0.660 - 0.690 |
| mASE | 0.274 | 0.268 - 0.280 |
| mAOE | 0.372 | 0.360 - 0.385 |
| mAVE | 0.394 | 0.380 - 0.410 |
| mAAE | 0.198 | 0.185 - 0.210 |

### 10.3 Common Reproduction Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong backbone pretrain | NDS ~0.48 instead of ~0.52 | Use FCOS3D pretrained weights |
| Missing can_bus data | mAVE ~0.8 (poor velocity) | Download and extract can_bus |
| Wrong mmcv version | Deformable attention errors | Use mmcv-full==1.7.1 exactly |
| Queue length mismatch | No temporal benefit | Ensure queue_length=4 in config |
| Seed difference | ±0.5 NDS variation | Run 3 seeds, report mean |
