# MapTR: Evaluation Guide

## Overview

MapTR evaluation measures the quality of predicted vectorized map elements against ground truth using Chamfer-distance-based Average Precision (AP). This metric captures both the detection accuracy (are the right elements found?) and the geometric precision (are the predicted point positions correct?).

---

## Primary Metric: Chamfer-Distance Based AP

### Concept

Unlike object detection which uses IoU (Intersection over Union) for matching, vectorized map evaluation uses **Chamfer distance** as the geometric similarity measure between predicted and ground truth elements. A prediction is considered a "true positive" if its Chamfer distance to a matched ground truth element is below a specified threshold.

### Chamfer Distance Definition

For a predicted point set P = {p_1, ..., p_N} and ground truth point set Q = {q_1, ..., q_N}:

```python
def chamfer_distance(P, Q):
    """
    Symmetric Chamfer distance between two point sets.
    
    P: (N, 2) - predicted points
    Q: (N, 2) - ground truth points
    
    Returns: scalar distance in meters
    """
    # For each point in P, find nearest point in Q
    dist_P_to_Q = torch.cdist(P, Q, p=2)  # (N, N) pairwise L2 distances
    min_P_to_Q = dist_P_to_Q.min(dim=1)[0]  # (N,) nearest distance for each P point
    
    # For each point in Q, find nearest point in P
    min_Q_to_P = dist_P_to_Q.min(dim=0)[0]  # (N,) nearest distance for each Q point
    
    # Symmetric Chamfer distance
    chamfer = (min_P_to_Q.mean() + min_Q_to_P.mean()) / 2
    
    return chamfer  # In meters
```

### Why Chamfer Distance?

| Property | Chamfer Distance | IoU |
|----------|-----------------|-----|
| Applicable to | Point sets, polylines | Boxes, masks |
| Handles open curves | Yes | No (requires closed regions) |
| Permutation invariant | Yes (nearest-neighbor) | N/A |
| Metric unit | Meters (interpretable) | Dimensionless ratio |
| Threshold meaning | "Within X meters" | "X% overlap" |

---

## AP Computation

### Step 1: Per-Frame Matching

For each frame, match predicted elements to ground truth elements:

```python
def match_predictions_to_gt(predictions, ground_truth, category):
    """
    Match predictions to GT for a single frame and category.
    Uses greedy matching based on Chamfer distance.
    """
    # Filter by category
    preds = [p for p in predictions if p['category'] == category]
    gts = [g for g in ground_truth if g['category'] == category]
    
    # Sort predictions by confidence score (descending)
    preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)
    
    # Compute pairwise Chamfer distances
    num_preds = len(preds)
    num_gts = len(gts)
    distance_matrix = np.zeros((num_preds, num_gts))
    
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            distance_matrix[i, j] = chamfer_distance(
                pred['points'], gt['points']
            )
    
    return preds, gts, distance_matrix
```

### Step 2: True Positive / False Positive Assignment

```python
def assign_tp_fp(preds, gts, distance_matrix, threshold):
    """
    Assign TP/FP labels to predictions at a given Chamfer distance threshold.
    
    threshold: distance in meters (e.g., 0.5, 1.0, 1.5)
    """
    num_preds = len(preds)
    num_gts = len(gts)
    
    tp = np.zeros(num_preds)
    fp = np.zeros(num_preds)
    gt_matched = np.zeros(num_gts, dtype=bool)
    
    # Process predictions in order of decreasing confidence
    for i in range(num_preds):
        # Find best matching GT (minimum Chamfer distance)
        if num_gts == 0:
            fp[i] = 1
            continue
        
        min_dist_idx = np.argmin(distance_matrix[i])
        min_dist = distance_matrix[i, min_dist_idx]
        
        if min_dist < threshold and not gt_matched[min_dist_idx]:
            # True positive: distance below threshold and GT not already matched
            tp[i] = 1
            gt_matched[min_dist_idx] = True
        else:
            # False positive: distance above threshold or GT already matched
            fp[i] = 1
    
    return tp, fp
```

### Step 3: Precision-Recall Curve

```python
def compute_precision_recall(all_tp, all_fp, all_confidences, total_gt):
    """
    Compute precision-recall curve across all frames.
    
    all_tp: concatenated TP flags from all frames
    all_fp: concatenated FP flags from all frames
    all_confidences: concatenated confidence scores
    total_gt: total number of GT elements across all frames
    """
    # Sort by confidence (descending)
    sorted_indices = np.argsort(-all_confidences)
    tp_sorted = all_tp[sorted_indices]
    fp_sorted = all_fp[sorted_indices]
    
    # Cumulative sums
    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(fp_sorted)
    
    # Precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / total_gt
    
    return precision, recall
```

### Step 4: AP Calculation (11-Point Interpolation)

```python
def compute_ap(precision, recall):
    """
    Compute Average Precision using 11-point interpolation.
    """
    ap = 0.0
    recall_thresholds = np.linspace(0, 1, 11)  # [0, 0.1, 0.2, ..., 1.0]
    
    for r_thresh in recall_thresholds:
        # Maximum precision at recall >= r_thresh
        precisions_at_recall = precision[recall >= r_thresh]
        if len(precisions_at_recall) == 0:
            p = 0.0
        else:
            p = precisions_at_recall.max()
        ap += p
    
    ap /= 11.0
    return ap
```

Alternative: All-point interpolation (used in some implementations):

```python
def compute_ap_all_points(precision, recall):
    """
    Compute AP using all-point interpolation (more accurate).
    """
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum (Delta recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
```

---

## Distance Thresholds

### Standard Thresholds

Evaluation is performed at three Chamfer distance thresholds:

| Threshold | Interpretation | Strictness |
|-----------|---------------|-----------|
| 0.5 m | Within half a meter | Strict (high precision required) |
| 1.0 m | Within one meter | Moderate |
| 1.5 m | Within 1.5 meters | Lenient |

### Per-Threshold AP

```python
thresholds = [0.5, 1.0, 1.5]  # meters

for category in ["ped_crossing", "divider", "boundary"]:
    for threshold in thresholds:
        ap = compute_ap_at_threshold(predictions, ground_truth, category, threshold)
        print(f"AP@{threshold}m ({category}): {ap:.3f}")
```

---

## Per-Category Evaluation

### Category Definitions

| Category ID | Name | Geometry | Typical Difficulty |
|-------------|------|----------|-------------------|
| 0 | Pedestrian Crossing | Polygon | Medium (large, regular shape) |
| 1 | Lane Divider | Polyline | High (many instances, varying curvature) |
| 2 | Road Boundary | Polyline | Medium (fewer instances, longer curves) |

### Per-Category Metrics

```python
results = {}
for category in ["ped_crossing", "divider", "boundary"]:
    results[category] = {}
    for threshold in [0.5, 1.0, 1.5]:
        ap = compute_ap_at_threshold(predictions, ground_truth, category, threshold)
        results[category][f"AP@{threshold}"] = ap
    
    # Average across thresholds for this category
    results[category]["AP_avg"] = np.mean([
        results[category]["AP@0.5"],
        results[category]["AP@1.0"],
        results[category]["AP@1.5"]
    ])
```

---

## mAP Computation

### Final mAP Score

The mean Average Precision (mAP) is computed as the mean across all categories and all thresholds:

```python
def compute_mAP(results):
    """
    mAP = mean over (categories x thresholds)
    """
    all_aps = []
    for category in ["ped_crossing", "divider", "boundary"]:
        for threshold in [0.5, 1.0, 1.5]:
            all_aps.append(results[category][f"AP@{threshold}"])
    
    mAP = np.mean(all_aps)  # Mean of 3 categories x 3 thresholds = 9 AP values
    return mAP
```

### Results Table Format

```
+-------------------+--------+--------+--------+--------+
| Category          | AP@0.5 | AP@1.0 | AP@1.5 | AP_avg |
+-------------------+--------+--------+--------+--------+
| Ped Crossing      | 38.7   | 55.2   | 61.4   | 51.8   |
| Lane Divider      | 42.1   | 58.9   | 65.3   | 55.4   |
| Road Boundary     | 45.6   | 61.7   | 67.8   | 58.4   |
+-------------------+--------+--------+--------+--------+
| mAP               |  42.1  |  58.6  |  64.8  | 50.3   |
+-------------------+--------+--------+--------+--------+
```

---

## Comparison with Baselines

### nuScenes Benchmark (ResNet-50 backbone)

| Method | Epochs | Ped Cross | Divider | Boundary | mAP | FPS |
|--------|--------|-----------|---------|----------|-----|-----|
| HDMapNet | 30 | 13.1 | 23.4 | 28.6 | 21.7 | 3.2 |
| VectorMapNet | 110 | 27.5 | 38.3 | 42.5 | 36.1 | 2.9 |
| MapTR | 24 | 35.2 | 46.8 | 47.6 | 43.2 | 25.1 |
| MapTR | 110 | 38.1 | 50.2 | 50.6 | 46.3 | 25.1 |
| MapTRv2 | 24 | 39.5 | 49.8 | 50.8 | 46.7 | 21.8 |
| MapTRv2 | 110 | 43.2 | 54.1 | 53.6 | 50.3 | 21.8 |

### Key Takeaways

1. **MapTR vs HDMapNet**: +21.5 mAP improvement while being 8x faster (vectorized vs rasterized)
2. **MapTR vs VectorMapNet**: +7.1 mAP improvement while being 9x faster (parallel vs autoregressive)
3. **MapTRv2 vs MapTR**: +3.5 mAP at same training cost (24ep), enabled by one-to-many matching
4. **Extended training**: +4 mAP from 24 to 110 epochs (diminishing returns after 80 epochs)

---

## FPS Measurement Methodology

### Standard Protocol

```python
def measure_fps(model, dataloader, device, num_warmup=50, num_test=500):
    """
    Measure inference FPS following standard protocol.
    """
    model.eval()
    
    # Warm-up (GPU initialization, CUDA graph compilation)
    for i, batch in enumerate(dataloader):
        if i >= num_warmup:
            break
        with torch.no_grad():
            _ = model(batch.to(device))
    
    # Timed inference
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_test:
            break
        with torch.no_grad():
            _ = model(batch.to(device))
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    fps = num_test / elapsed
    latency_ms = elapsed / num_test * 1000
    
    return fps, latency_ms
```

### Measurement Conditions

| Parameter | Standard Setting |
|-----------|-----------------|
| GPU | NVIDIA RTX 3090 (24 GB) |
| Batch size | 1 (single frame) |
| Input resolution | 800 x 480 per camera |
| Precision | FP32 (or FP16 if specified) |
| Warm-up iterations | 50 |
| Test iterations | 500 |
| Include data loading | No (GPU time only) |
| Include post-processing | Yes (included in model forward) |
| CUDA synchronization | Yes (accurate GPU timing) |

### What Is Included in FPS

| Component | Included | Notes |
|-----------|----------|-------|
| Image backbone | Yes | Feature extraction |
| BEV encoder | Yes | Perspective-to-BEV |
| Map decoder | Yes | Query refinement |
| Prediction heads | Yes | Classification + regression |
| NMS / filtering | Yes | Confidence threshold |
| Data loading | No | I/O dependent |
| Visualization | No | Not part of inference |
| Point coordinate denormalization | Yes | Trivial cost |

### FPS vs Backbone Comparison

| Configuration | Params | FLOPs | FPS | Latency |
|--------------|--------|-------|-----|---------|
| MapTR + R50 | 50.2M | 46.6G | 25.1 | 39.8 ms |
| MapTR + R101 | 69.1M | 73.4G | 18.7 | 53.5 ms |
| MapTRv2 + R50 | 52.6M | 49.7G | 21.8 | 45.9 ms |
| MapTRv2 + VoV99 | 62.1M | 61.3G | 14.1 | 70.9 ms |

---

## Evaluation Script Usage

### Running Evaluation

```bash
# Evaluate a trained model on the validation set
python tools/eval.py \
    --config configs/maptr_r50_24ep.py \
    --checkpoint work_dirs/maptr_r50_24ep/epoch_24.pth \
    --eval chamfer \
    --gpu-ids 0

# Evaluate with specific thresholds
python tools/eval.py \
    --config configs/maptr_r50_24ep.py \
    --checkpoint work_dirs/maptr_r50_24ep/epoch_24.pth \
    --eval chamfer \
    --thresholds 0.5 1.0 1.5 \
    --out results/maptr_r50_24ep_eval.json
```

### Evaluation Output Format

```json
{
    "mAP": 0.503,
    "per_category": {
        "ped_crossing": {
            "AP@0.5": 0.387,
            "AP@1.0": 0.552,
            "AP@1.5": 0.614,
            "AP_avg": 0.518,
            "num_gt": 4200,
            "num_pred": 3850
        },
        "divider": {
            "AP@0.5": 0.421,
            "AP@1.0": 0.589,
            "AP@1.5": 0.653,
            "AP_avg": 0.554,
            "num_gt": 24800,
            "num_pred": 23100
        },
        "boundary": {
            "AP@0.5": 0.456,
            "AP@1.0": 0.617,
            "AP@1.5": 0.678,
            "AP_avg": 0.584,
            "num_gt": 11400,
            "num_pred": 10900
        }
    },
    "per_threshold": {
        "AP@0.5": 0.421,
        "AP@1.0": 0.586,
        "AP@1.5": 0.648
    },
    "inference_fps": 21.8,
    "inference_latency_ms": 45.9
}
```

---

## Visualization for Qualitative Evaluation

### Predicted Map Visualization

```python
def visualize_predictions(predictions, ground_truth, perception_range, save_path):
    """
    Visualize predicted map elements overlaid on BEV for qualitative evaluation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: Ground truth
    ax = axes[0]
    ax.set_title("Ground Truth")
    for gt in ground_truth:
        color = category_colors[gt['category']]
        pts = gt['points']
        if gt['type'] == 'polygon':
            polygon = plt.Polygon(pts, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(polygon)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2)
    
    # Center: Predictions
    ax = axes[1]
    ax.set_title("Predictions")
    for pred in predictions:
        color = category_colors[pred['category']]
        pts = pred['points']
        alpha = pred['confidence']
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, alpha=alpha)
    
    # Right: Overlay (TP=green, FP=red, FN=blue)
    ax = axes[2]
    ax.set_title("Matching (TP/FP/FN)")
    # ... matching visualization ...
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
```

---

## Advanced Evaluation Aspects

### Evaluation by Scene Difficulty

| Scene Type | Typical mAP | Challenge |
|-----------|-------------|-----------|
| Highway | 55-60 | Long straight elements, fewer instances |
| Urban intersection | 35-40 | Many elements, occlusions, complex topology |
| Parking lot | 40-45 | Dense crossings, short boundaries |
| Night/rain | 30-35 | Degraded image quality |

### Evaluation by Distance from Ego

| Distance Band | Elements | AP Drop vs Close |
|--------------|----------|-----------------|
| 0-10m | Nearby | Baseline (best) |
| 10-20m | Medium | -3 to -5 mAP |
| 20-30m | Far | -8 to -12 mAP |

### Evaluation by Element Length

| Length Range | Difficulty | Notes |
|-------------|-----------|-------|
| < 5m | Higher (partial visibility) | Often clipped at range boundary |
| 5-20m | Standard | Majority of elements |
| > 20m | Lower (more context) | Easier to detect, harder to regress precisely |

---

## Reproducing Published Results

### Checklist

- [ ] Use official nuScenes val split (150 scenes, scene tokens must match)
- [ ] Perception range: [-30, 30] x [-15, 15] meters
- [ ] N_pts = 20 points per element
- [ ] Chamfer distance thresholds: {0.5, 1.0, 1.5} meters
- [ ] Three categories: pedestrian crossing, lane divider, road boundary
- [ ] Confidence threshold: 0.0 (include all predictions for AP curve)
- [ ] No test-time augmentation unless specified
- [ ] Single-frame input (no temporal aggregation) for fair comparison
- [ ] Report FPS on RTX 3090 with batch size 1

### Common Pitfalls

| Issue | Effect on mAP | Solution |
|-------|--------------|----------|
| Wrong perception range | ±5 mAP | Verify x/y ranges match config |
| Wrong coordinate normalization | Catastrophic | Ensure [0,1] normalization consistent |
| Missing map elements at boundary | -1 to -2 mAP | Proper clipping implementation |
| Different N_pts | ±2 mAP | Use N_pts=20 |
| Including test-time augmentation | +1 to +2 mAP | Report with and without TTA |
| Wrong Chamfer distance (ordered vs unordered) | ±3 mAP | Use unordered (nearest-neighbor) |
