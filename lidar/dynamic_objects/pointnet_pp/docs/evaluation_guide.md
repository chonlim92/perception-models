# PointNet++ Evaluation Guide

## 1. Overview

This guide covers the evaluation metrics, protocols, and benchmarking procedures for PointNet++ models applied to 3D object detection, semantic segmentation, and classification tasks in autonomous driving.

---

## 2. 3D Object Detection Metrics

### 2.1 Average Precision (AP) — KITTI Protocol

KITTI uses Average Precision computed from the precision-recall curve for 3D object detection.

#### IoU Thresholds

| Class | BEV AP IoU | 3D AP IoU |
|-------|-----------|-----------|
| Car | 0.7 | 0.7 |
| Pedestrian | 0.5 | 0.5 |
| Cyclist | 0.5 | 0.5 |
| Van | 0.7 | 0.7 |
| Truck | 0.7 | 0.7 |

#### KITTI Difficulty Levels (for evaluation filtering)

| Difficulty | Min BBox Height | Max Occlusion | Max Truncation |
|------------|-----------------|---------------|----------------|
| Easy | 40 px | Fully visible (0) | 15% |
| Moderate | 25 px | Partly (0,1) | 30% |
| Hard | 25 px | Largely (0,1,2) | 50% |

#### AP Calculation (KITTI 40-point interpolation, post-2019)

```python
import numpy as np

def compute_ap_kitti(recall, precision, num_recall_points=40):
    """Compute Average Precision using KITTI's 40-point interpolation.
    
    Note: Pre-2019 KITTI used 11-point interpolation. Current benchmark uses 40 points.
    
    Args:
        recall: (N,) sorted recall values
        precision: (N,) corresponding precision values
        num_recall_points: 40 for current KITTI, 11 for legacy
    
    Returns:
        ap: float, average precision
    """
    # Sample recall at uniform points
    recall_points = np.linspace(0, 1, num_recall_points + 1)
    
    # For each recall threshold, find the maximum precision at recall >= threshold
    precisions_at_recall = []
    for r in recall_points:
        # Precision at recall >= r (right-envelope)
        mask = recall >= r
        if mask.any():
            precisions_at_recall.append(precision[mask].max())
        else:
            precisions_at_recall.append(0.0)
    
    ap = np.mean(precisions_at_recall)
    return ap

def evaluate_detections_kitti(predictions, ground_truths, iou_threshold=0.7, 
                               difficulty='moderate'):
    """Full KITTI-style 3D detection evaluation.
    
    Args:
        predictions: list of dicts with 'boxes_3d', 'scores', 'labels'
        ground_truths: list of dicts with 'boxes_3d', 'labels', 'difficulty'
        iou_threshold: IoU threshold for matching
        difficulty: 'easy', 'moderate', or 'hard'
    
    Returns:
        ap_dict: {class_name: AP} dictionary
    """
    # 1. Filter ground truths by difficulty
    # 2. Sort predictions by confidence score (descending)
    # 3. Match predictions to ground truths using IoU
    # 4. Compute precision-recall curve
    # 5. Compute AP using 40-point interpolation
    
    all_scores = []
    all_matches = []  # True positive or false positive
    total_gt = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Filter GT by difficulty
        valid_gt = filter_by_difficulty(gt, difficulty)
        total_gt += len(valid_gt)
        
        # Sort predictions by score
        sorted_idx = np.argsort(-pred['scores'])
        
        matched_gt = set()
        for idx in sorted_idx:
            pred_box = pred['boxes_3d'][idx]
            all_scores.append(pred['scores'][idx])
            
            # Find best matching GT
            best_iou = 0
            best_gt_idx = -1
            for gi, gt_box in enumerate(valid_gt['boxes_3d']):
                if gi in matched_gt:
                    continue
                iou = compute_3d_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                all_matches.append(1)  # True positive
                matched_gt.add(best_gt_idx)
            else:
                all_matches.append(0)  # False positive
    
    # Sort by score globally
    sorted_idx = np.argsort(-np.array(all_scores))
    all_matches = np.array(all_matches)[sorted_idx]
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(all_matches)
    fp_cumsum = np.cumsum(1 - all_matches)
    
    recall = tp_cumsum / max(total_gt, 1)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    ap = compute_ap_kitti(recall, precision)
    return ap
```

### 2.2 nuScenes Detection Score (NDS)

nuScenes uses a composite metric combining mAP with additional true positive metrics:

```
NDS = (5 × mAP + Σ(1 - min(1, TP_metric))) / 10

where TP metrics are:
  - ATE: Average Translation Error (meters)
  - ASE: Average Scale Error (1 - IoU after alignment)
  - AOE: Average Orientation Error (radians)
  - AVE: Average Velocity Error (m/s)
  - AAE: Average Attribute Error (1 - attribute_accuracy)
```

#### nuScenes mAP Calculation

```python
def compute_nuscenes_mAP(predictions, ground_truths, dist_thresholds=[0.5, 1.0, 2.0, 4.0]):
    """Compute nuScenes-style mAP using center distance matching.
    
    Key difference from KITTI: 
    - Uses center distance instead of 3D IoU for matching
    - Averages AP over multiple distance thresholds
    - 10 detection classes
    
    Args:
        dist_thresholds: matching thresholds in meters
    """
    # nuScenes detection classes
    classes = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
               'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
    
    ap_per_class = {}
    for cls in classes:
        ap_at_thresholds = []
        for dist_thresh in dist_thresholds:
            # Match by center distance (BEV) < dist_thresh
            ap = compute_ap_by_distance(predictions, ground_truths, 
                                         cls, dist_thresh)
            ap_at_thresholds.append(ap)
        ap_per_class[cls] = np.mean(ap_at_thresholds)
    
    mAP = np.mean(list(ap_per_class.values()))
    return mAP, ap_per_class
```

#### True Positive Metrics (nuScenes)

```python
def compute_tp_metrics(matched_predictions, matched_ground_truths):
    """Compute nuScenes true positive metrics for matched detections."""
    
    metrics = {}
    
    # ATE: Average Translation Error (Euclidean distance in BEV)
    translation_errors = np.linalg.norm(
        matched_predictions['centers'][:, :2] - matched_ground_truths['centers'][:, :2],
        axis=1
    )
    metrics['ATE'] = np.mean(translation_errors)  # Target: < 0.5m
    
    # ASE: Average Scale Error (1 - 3D IoU after center alignment)
    scale_errors = []
    for pred, gt in zip(matched_predictions['sizes'], matched_ground_truths['sizes']):
        # Compute IoU with boxes aligned at same center
        iou = aligned_box_iou(pred, gt)
        scale_errors.append(1 - iou)
    metrics['ASE'] = np.mean(scale_errors)  # Target: < 0.2
    
    # AOE: Average Orientation Error (smallest angle between headings)
    orientation_errors = np.abs(
        angle_diff(matched_predictions['yaws'], matched_ground_truths['yaws'])
    )
    metrics['AOE'] = np.mean(orientation_errors)  # Target: < 0.3 rad
    
    # AVE: Average Velocity Error (L2 norm of velocity difference)
    velocity_errors = np.linalg.norm(
        matched_predictions['velocities'] - matched_ground_truths['velocities'],
        axis=1
    )
    metrics['AVE'] = np.mean(velocity_errors)  # Target: < 0.5 m/s
    
    # AAE: Average Attribute Error (1 - attribute accuracy)
    attribute_correct = (matched_predictions['attributes'] == 
                         matched_ground_truths['attributes'])
    metrics['AAE'] = 1 - np.mean(attribute_correct)  # Target: < 0.2
    
    return metrics

def compute_nds(mAP, tp_metrics):
    """Compute nuScenes Detection Score."""
    tp_scores = sum(max(1 - v, 0) for v in tp_metrics.values())
    nds = (5 * mAP + tp_scores) / 10
    return nds
```

---

## 3. Semantic Segmentation Metrics

### 3.1 Mean Intersection over Union (mIoU)

```python
def compute_miou(predictions, ground_truths, num_classes=19, ignore_index=0):
    """Compute mean IoU for semantic segmentation.
    
    Args:
        predictions: list of (N_i,) predicted class labels
        ground_truths: list of (N_i,) ground truth class labels
        num_classes: total number of semantic classes
        ignore_index: class index to ignore (typically 'unlabeled')
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for pred, gt in zip(predictions, ground_truths):
        valid_mask = gt != ignore_index
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        
        for p, g in zip(pred_valid, gt_valid):
            confusion_matrix[g, p] += 1
    
    # IoU per class
    intersection = np.diag(confusion_matrix)
    union = (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) 
             - intersection)
    
    # Avoid division by zero for classes not present
    valid_classes = union > 0
    iou_per_class = np.zeros(num_classes)
    iou_per_class[valid_classes] = intersection[valid_classes] / union[valid_classes]
    
    # Mean over valid classes (excluding ignore_index)
    eval_classes = valid_classes.copy()
    eval_classes[ignore_index] = False
    mIoU = iou_per_class[eval_classes].mean()
    
    return mIoU, iou_per_class
```

### 3.2 SemanticKITTI Classes (19 evaluation classes)

| ID | Class | Category |
|----|-------|----------|
| 1 | car | vehicle |
| 2 | bicycle | vehicle |
| 3 | motorcycle | vehicle |
| 4 | truck | vehicle |
| 5 | other-vehicle | vehicle |
| 6 | person | human |
| 7 | bicyclist | human |
| 8 | motorcyclist | human |
| 9 | road | ground |
| 10 | parking | ground |
| 11 | sidewalk | ground |
| 12 | other-ground | ground |
| 13 | building | structure |
| 14 | fence | structure |
| 15 | vegetation | nature |
| 16 | trunk | nature |
| 17 | terrain | nature |
| 18 | pole | object |
| 19 | traffic-sign | object |

### 3.3 Per-Class Performance Expectations

| Class | Typical IoU (PointNet++) | Challenge |
|-------|--------------------------|-----------|
| car | 85-92% | Well-represented, distinctive shape |
| road | 88-94% | Large surface area, many points |
| building | 82-90% | Large structures, well-defined |
| vegetation | 78-86% | Varied shapes, seasonal changes |
| person | 45-60% | Few points at distance, varied pose |
| bicyclist | 40-55% | Rare, similar to pedestrian |
| motorcycle | 35-50% | Rare, varied appearance |
| traffic-sign | 30-45% | Very small, few points |

---

## 4. Classification Metrics

### 4.1 Overall Accuracy and Per-Class Accuracy

```python
def evaluate_classification(predictions, ground_truths, num_classes):
    """Compute classification metrics.
    
    Returns:
        overall_accuracy: fraction of correct predictions
        mean_class_accuracy: average per-class accuracy
        per_class_accuracy: accuracy for each class
    """
    correct = (predictions == ground_truths)
    overall_accuracy = correct.mean()
    
    per_class_accuracy = np.zeros(num_classes)
    for c in range(num_classes):
        class_mask = ground_truths == c
        if class_mask.sum() > 0:
            per_class_accuracy[c] = correct[class_mask].mean()
    
    mean_class_accuracy = per_class_accuracy[per_class_accuracy > 0].mean()
    
    return overall_accuracy, mean_class_accuracy, per_class_accuracy
```

### 4.2 Expected Results (ModelNet40)

| Method | OA (%) | mAcc (%) |
|--------|--------|----------|
| PointNet | 89.2 | 86.0 |
| PointNet++ SSG | 90.7 | - |
| PointNet++ MSG | 91.9 | - |
| PointNet++ MSG + DP | 91.9 | - |
| DGCNN | 92.9 | 90.2 |
| Point Transformer | 93.7 | 90.6 |

---

## 5. Inference Speed Benchmarks

### 5.1 Measurement Protocol

```python
import time
import torch

def benchmark_inference(model, input_shape, num_warmup=50, num_runs=200, device='cuda'):
    """Measure inference latency.
    
    Args:
        model: trained model in eval mode
        input_shape: (B, N, C) input tensor shape
        num_warmup: warmup iterations (not timed)
        num_runs: timed iterations
    """
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup (CUDA kernel compilation, memory allocation)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_event.record()
            _ = model(dummy_input)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
    
    latencies = np.array(latencies)
    
    return {
        'mean_ms': latencies.mean(),
        'std_ms': latencies.std(),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'fps': 1000.0 / latencies.mean(),
    }
```

### 5.2 Latency Breakdown by Component

| Component | 16K points | 32K points | 64K points |
|-----------|-----------|-----------|-----------|
| FPS (SA1) | 8ms | 15ms | 28ms |
| Ball Query (SA1) | 5ms | 9ms | 16ms |
| MLP (SA1) | 3ms | 5ms | 8ms |
| FPS (SA2) | 3ms | 5ms | 8ms |
| Ball Query (SA2) | 2ms | 3ms | 5ms |
| MLP (SA2) | 2ms | 3ms | 4ms |
| Deeper layers | 5ms | 7ms | 10ms |
| Detection Head | 4ms | 5ms | 6ms |
| NMS | 2ms | 2ms | 3ms |
| **Total** | **34ms** | **54ms** | **88ms** |

### 5.3 Hardware Comparison

| Model Config | RTX 3090 | A100 | RTX 4090 | Orin (edge) |
|-------------|----------|------|----------|-------------|
| Cls (1024 pts) | 2.5ms | 1.8ms | 2.0ms | 12ms |
| Seg (8192 pts) | 38ms | 25ms | 30ms | 180ms |
| Det (16384 pts) | 55ms | 35ms | 42ms | 280ms |
| Det (16384, TensorRT) | 32ms | 20ms | 24ms | 150ms |

### 5.4 Real-Time Feasibility for Autonomous Driving

```
Requirement: LiDAR operates at 10 Hz → 100ms budget per frame

Budget allocation:
  - Preprocessing (filtering, FPS): 10ms
  - PointNet++ backbone: 35-55ms
  - Detection head + NMS: 10ms
  - Post-processing (tracking input): 5ms
  - Buffer for jitter: 10-20ms
  ─────────────────────────────────
  Total: 70-100ms ✓ (marginal for single-stage)

For two-stage detectors: ~120-150ms → requires optimization or 5Hz operation
```

---

## 6. Per-Class Evaluation Breakdown

### 6.1 KITTI 3D AP Results (Moderate Difficulty)

| Method | Car | Pedestrian | Cyclist |
|--------|-----|------------|---------|
| PointNet++ backbone (basic) | ~72% | ~48% | ~55% |
| PointRCNN | 75.64% | 54.68% | 62.68% |
| STD | 79.80% | 53.29% | 62.17% |
| PV-RCNN | 83.61% | 57.90% | 70.47% |
| Voxel R-CNN | 84.52% | - | - |
| CenterPoint | 84.6% | 58.2% | 71.3% |

### 6.2 nuScenes Detection Leaderboard Context

| Method | mAP | NDS | Latency |
|--------|-----|-----|---------|
| PointPillars | 30.5% | 45.3% | 23ms |
| CenterPoint (voxel) | 58.0% | 65.5% | 60ms |
| TransFusion-L | 65.5% | 70.2% | 68ms |
| BEVFusion | 68.5% | 71.4% | 82ms |
| PointNet++ based (custom) | ~45% | ~55% | 90ms |

### 6.3 Distance-Based Performance Degradation

Performance drops significantly with distance due to point sparsity:

| Distance Range | Points (Car) | 3D AP (Car) | AP Drop |
|---------------|--------------|-------------|---------|
| 0-20m | 500-2000 | 88% | baseline |
| 20-40m | 50-500 | 72% | -16% |
| 40-60m | 10-50 | 45% | -43% |
| 60-80m | 2-10 | 15% | -73% |

---

## 7. Evaluation Protocol

### 7.1 KITTI Evaluation Split

```
Official split (used for benchmark submission):
  Training: 7481 samples (indices in train.txt)
  Testing: 7518 samples (no public labels)

Common research split (Chen et al.):
  Train: 3712 samples
  Val: 3769 samples
  
This is the split used for ablation studies and development.
```

### 7.2 nuScenes Evaluation Split

```
v1.0-trainval:
  Train: 700 scenes, 28130 keyframes
  Val: 150 scenes, 6019 keyframes

v1.0-test:
  Test: 150 scenes, 6008 keyframes (labels withheld)
```

### 7.3 Cross-Validation for Small Datasets

```python
from sklearn.model_selection import KFold

def cross_validate(dataset, model_fn, config, n_folds=5):
    """K-fold cross-validation for point cloud models."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        model = model_fn(config)
        train(model, train_subset, config)
        result = evaluate(model, val_subset)
        fold_results.append(result)
    
    mean_result = np.mean(fold_results)
    std_result = np.std(fold_results)
    return mean_result, std_result
```

---

## 8. Ablation Study Framework

### 8.1 Key Ablations to Report

| Ablation | What it tests | Expected finding |
|----------|--------------|------------------|
| SSG vs MSG | Multi-scale grouping benefit | MSG +1-2% AP for varied density |
| Number of SA layers | Depth vs. efficiency | 3-4 layers optimal |
| FPS vs random sampling | Sampling quality | FPS +2-3% accuracy |
| Ball query vs KNN | Grouping strategy | Ball query better for outdoor |
| Radius values | Neighborhood size | Larger radii needed at later layers |
| nsample | Points per group | 32-64 is sweet spot |
| With/without normals | Feature importance | Normals help indoor (+1-2%) |
| Augmentation ablation | Each augmentation's contribution | GT-aug largest impact for detection |

### 8.2 Reporting Template

```
Table X: Ablation study on [component] (KITTI val, Car class, Moderate)

| Variant | 3D AP@0.7 | BEV AP@0.7 | Latency (ms) | Δ AP |
|---------|-----------|-----------|-------------|------|
| Baseline | 75.6 | 82.3 | 55 | - |
| + MSG | 76.8 | 83.1 | 72 | +1.2 |
| + larger radius | 77.3 | 83.5 | 73 | +1.7 |
| - augmentation | 72.1 | 79.4 | 55 | -3.5 |
```

---

## 9. Visualization for Evaluation

### 9.1 Qualitative Evaluation Checklist

For each model, visualize and inspect:

1. **Detection results in BEV (Bird's Eye View):**
   - Ground truth boxes (green)
   - Predicted boxes with confidence > 0.3 (blue)
   - False positives (red)
   - Missed objects (yellow)

2. **3D view with point cloud:**
   - Points colored by class (segmentation)
   - Box orientation arrows
   - Confidence scores displayed

3. **Failure cases analysis:**
   - Objects at maximum range
   - Heavily occluded objects
   - Objects on the boundary of the detection region
   - Small objects (pedestrians at distance)

### 9.2 Confidence Score Distribution

```python
def analyze_score_distribution(predictions, ground_truths, iou_threshold=0.7):
    """Analyze confidence score distribution for TP and FP."""
    tp_scores = []
    fp_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        for i, score in enumerate(pred['scores']):
            is_tp = any(
                compute_3d_iou(pred['boxes_3d'][i], gt_box) >= iou_threshold
                for gt_box in gt['boxes_3d']
            )
            if is_tp:
                tp_scores.append(score)
            else:
                fp_scores.append(score)
    
    return {
        'tp_mean': np.mean(tp_scores),
        'tp_median': np.median(tp_scores),
        'fp_mean': np.mean(fp_scores),
        'fp_median': np.median(fp_scores),
        'optimal_threshold': find_optimal_threshold(tp_scores, fp_scores)
    }
```

---

## 10. Reproducing Published Results

### 10.1 Common Pitfalls

1. **KITTI AP version:** Ensure using 40-point interpolation (post-2019), not 11-point
2. **Evaluation split:** Many papers use the Chen et al. train/val split, not full training set
3. **Point sampling:** Results vary with the random seed for FPS initialization
4. **NMS implementation:** Rotated NMS vs axis-aligned NMS gives different results
5. **Multi-sweep:** nuScenes results depend on number of aggregated sweeps (typically 10)
6. **Test-time augmentation:** Some results include TTA (flip + rotation voting)

### 10.2 Reproducibility Checklist

```
□ Random seed fixed (torch, numpy, CUDA)
□ Evaluation split matches reported split
□ IoU computation uses rotated boxes (not AABB)
□ AP calculation uses correct interpolation method
□ NMS threshold matches paper
□ Score threshold for reporting matches
□ Point sampling strategy matches (FPS vs random)
□ Number of points matches reported npoints
□ Augmentation disabled during evaluation
□ Model checkpoint is from best validation epoch (not last)
```
