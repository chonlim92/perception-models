# StreamMapNet: Evaluation Guide

## Overview

StreamMapNet evaluation measures how accurately predicted vectorized map elements match ground truth annotations. The primary metrics are based on Chamfer distance and Average Precision (AP) at multiple distance thresholds.

---

## Evaluation Metrics

### 1. Chamfer Distance

The Chamfer distance measures the similarity between two point sets (predicted polyline vs. GT polyline). It is computed bidirectionally:

```python
import numpy as np
from scipy.spatial.distance import cdist

def chamfer_distance(pred_points, gt_points):
    """
    Compute symmetric Chamfer distance between two polylines.
    
    Args:
        pred_points: (K, 2) predicted polyline points
        gt_points: (K, 2) ground truth polyline points
    
    Returns:
        chamfer_dist: scalar, average bidirectional distance in meters
    """
    # Pairwise distance matrix
    dist_matrix = cdist(pred_points, gt_points)  # (K, K)
    
    # Forward: for each predicted point, distance to nearest GT point
    forward_dist = dist_matrix.min(axis=1).mean()  # mean of min distances
    
    # Backward: for each GT point, distance to nearest predicted point
    backward_dist = dist_matrix.min(axis=0).mean()
    
    # Symmetric Chamfer distance
    chamfer_dist = (forward_dist + backward_dist) / 2.0
    
    return chamfer_dist
```

### Why Chamfer Distance?

- **Handles different point densities:** Even though both pred and GT have K points, they may not be aligned index-wise
- **Robust to small shifts:** Tolerant of slight positional offsets that don't affect semantic correctness
- **Threshold-friendly:** Easy to threshold for AP computation (element is "correct" if Chamfer < threshold)

### Direction-Aware Chamfer Distance

Since polylines can be traversed in either direction, the evaluation considers both:

```python
def direction_aware_chamfer(pred_points, gt_points):
    """
    Compute Chamfer distance considering both polyline directions.
    Returns the minimum of forward and reversed Chamfer distances.
    """
    # Forward direction
    cd_forward = chamfer_distance(pred_points, gt_points)
    
    # Reversed direction
    gt_reversed = gt_points[::-1]
    cd_reverse = chamfer_distance(pred_points, gt_reversed)
    
    return min(cd_forward, cd_reverse)
```

---

### 2. Average Precision (AP) at Distance Thresholds

AP is computed similarly to object detection AP, but using Chamfer distance instead of IoU:

#### Thresholds

| Threshold | Description | Interpretation |
|-----------|-------------|----------------|
| 0.5 m | Strict | Sub-lane-width accuracy |
| 1.0 m | Moderate | Approximately lane-width accuracy |
| 1.5 m | Lenient | Within-road accuracy |

#### AP Computation Pipeline

```python
def compute_ap_per_class(predictions, ground_truths, threshold, num_recall_points=40):
    """
    Compute AP for a single class at a given Chamfer distance threshold.
    
    Args:
        predictions: list of dicts with 'points' (K,2), 'score' (float), 'frame_id'
        ground_truths: list of dicts with 'points' (K,2), 'frame_id'
        threshold: Chamfer distance threshold in meters
        num_recall_points: number of recall levels for AP interpolation
    
    Returns:
        ap: Average Precision value
    """
    # Sort predictions by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Group GT by frame
    gt_by_frame = defaultdict(list)
    for gt in ground_truths:
        gt_by_frame[gt['frame_id']].append(gt)
    
    # Track which GTs have been matched
    gt_matched = defaultdict(lambda: [False] * 100)
    
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    for pred_idx, pred in enumerate(predictions):
        frame_id = pred['frame_id']
        frame_gts = gt_by_frame[frame_id]
        
        if len(frame_gts) == 0:
            fp[pred_idx] = 1
            continue
        
        # Compute Chamfer distance to all GTs in this frame
        min_dist = float('inf')
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(frame_gts):
            if gt_matched[frame_id][gt_idx]:
                continue  # Already matched
            
            dist = direction_aware_chamfer(pred['points'], gt['points'])
            if dist < min_dist:
                min_dist = dist
                best_gt_idx = gt_idx
        
        # Check if match is within threshold
        if min_dist <= threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[frame_id][best_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    total_gt = len(ground_truths)
    recall = tp_cumsum / total_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Interpolated AP (similar to PASCAL VOC 11-point or COCO-style)
    recall_levels = np.linspace(0, 1, num_recall_points)
    ap = 0
    for r_level in recall_levels:
        precisions_at_recall = precision[recall >= r_level]
        if len(precisions_at_recall) > 0:
            ap += precisions_at_recall.max()
    ap /= num_recall_points
    
    return ap
```

---

### 3. Mean Average Precision (mAP)

mAP is computed by averaging AP across all classes and thresholds:

```python
def compute_mAP(predictions_by_class, ground_truths_by_class):
    """
    Compute mean AP across classes and thresholds.
    
    Returns:
        mAP: float, primary evaluation metric
        detailed_results: dict with per-class, per-threshold AP
    """
    thresholds = [0.5, 1.0, 1.5]
    classes = ['lane_divider', 'road_boundary', 'ped_crossing']
    
    results = {}
    all_aps = []
    
    for cls_name in classes:
        results[cls_name] = {}
        preds = predictions_by_class[cls_name]
        gts = ground_truths_by_class[cls_name]
        
        for thresh in thresholds:
            ap = compute_ap_per_class(preds, gts, threshold=thresh)
            results[cls_name][f'AP_{thresh}'] = ap
            all_aps.append(ap)
        
        # Per-class mean AP (averaged over thresholds)
        results[cls_name]['AP'] = np.mean([
            results[cls_name][f'AP_{t}'] for t in thresholds
        ])
    
    # Overall mAP
    mAP = np.mean(all_aps)
    
    return mAP, results
```

### Result Format

```
+------------------+--------+--------+--------+--------+
| Category         | AP@0.5 | AP@1.0 | AP@1.5 |   AP   |
+------------------+--------+--------+--------+--------+
| Lane Divider     |  41.2  |  58.3  |  69.4  |  56.3  |
| Road Boundary    |  38.7  |  57.1  |  71.6  |  55.8  |
| Ped Crossing     |  33.8  |  51.4  |  65.1  |  50.1  |
+------------------+--------+--------+--------+--------+
| mAP              |  37.9  |  55.6  |  68.7  |  54.1  |
+------------------+--------+--------+--------+--------+
```

---

### 4. Per-Element Metrics

#### Lane Divider Metrics

| Metric | Description |
|--------|-------------|
| AP@0.5/1.0/1.5 | Standard AP at distance thresholds |
| Precision | Fraction of predictions that match a GT |
| Recall | Fraction of GTs that are detected |
| Mean Chamfer | Average Chamfer distance for true positives |

#### Road Boundary Metrics

Same as lane divider, but road boundaries tend to be longer and more continuous, making them easier to detect but harder to localize precisely.

#### Pedestrian Crossing Metrics

Pedestrian crossings are polygons (closed polylines). The Chamfer distance is computed on the closed polygon boundary. These elements are typically fewer in number and more localized.

---

### 5. Temporal Consistency Metrics

Beyond per-frame accuracy, StreamMapNet's streaming design is evaluated for temporal stability:

#### Map Element Stability Score

Measures how consistent predictions are across consecutive frames for the same physical map element:

```python
def temporal_consistency_score(predictions_sequence):
    """
    Compute temporal consistency across a sequence of frames.
    
    For each predicted element at frame t, find its correspondence at frame t+1
    (after ego-motion compensation) and measure the positional deviation.
    
    Args:
        predictions_sequence: list of per-frame predictions, 
                             each containing 'points', 'labels', 'scores'
    
    Returns:
        consistency_score: average deviation in meters (lower is better)
    """
    deviations = []
    
    for t in range(len(predictions_sequence) - 1):
        preds_t = predictions_sequence[t]
        preds_t1 = predictions_sequence[t + 1]
        ego_motion = get_ego_motion(t, t + 1)  # Transform t -> t+1
        
        for pred in preds_t['elements']:
            # Transform prediction from frame t to frame t+1 coordinates
            pred_warped = transform_points(pred['points'], ego_motion)
            
            # Find closest prediction in frame t+1 (same class)
            min_dist = float('inf')
            for pred_next in preds_t1['elements']:
                if pred_next['label'] != pred['label']:
                    continue
                dist = chamfer_distance(pred_warped, pred_next['points'])
                min_dist = min(min_dist, dist)
            
            if min_dist < 3.0:  # Only count if element persists
                deviations.append(min_dist)
    
    return np.mean(deviations) if deviations else 0.0
```

#### Flicker Rate

Measures how often map elements appear and disappear between consecutive frames:

```python
def flicker_rate(predictions_sequence, chamfer_threshold=2.0):
    """
    Compute the fraction of elements that disappear between consecutive frames.
    A "flicker" occurs when an element exists at time t but has no correspondence
    at time t+1 (or vice versa), despite being within the perception range.
    
    Returns:
        rate: fraction of element-frame pairs that flicker (lower is better)
    """
    total_elements = 0
    flickered = 0
    
    for t in range(len(predictions_sequence) - 1):
        preds_t = predictions_sequence[t]
        preds_t1 = predictions_sequence[t + 1]
        ego_motion = get_ego_motion(t, t + 1)
        
        for pred in preds_t['elements']:
            pred_warped = transform_points(pred['points'], ego_motion)
            
            # Check if element is still in perception range at t+1
            if not is_in_range(pred_warped):
                continue
            
            total_elements += 1
            
            # Check if a corresponding element exists at t+1
            found = False
            for pred_next in preds_t1['elements']:
                if pred_next['label'] != pred['label']:
                    continue
                if chamfer_distance(pred_warped, pred_next['points']) < chamfer_threshold:
                    found = True
                    break
            
            if not found:
                flickered += 1
    
    return flickered / max(total_elements, 1)
```

#### Temporal Metrics Comparison

| Method | Consistency (m) | Flicker Rate |
|--------|----------------|--------------|
| MapTR (single-frame) | 1.42 | 18.3% |
| StreamMapNet (no temporal) | 1.42 | 18.3% |
| StreamMapNet (temporal) | 0.67 | 7.2% |

---

## Evaluation Protocol

### Matching Predictions to Ground Truth

The evaluation uses greedy matching (not Hungarian) for the AP metric, following the standard object detection protocol:

1. **Sort predictions** by confidence score (descending)
2. **For each prediction** (in confidence order):
   - Compute Chamfer distance to all unmatched GT elements of the same class in the same frame
   - If minimum distance < threshold, mark as True Positive (TP) and mark that GT as matched
   - Otherwise, mark as False Positive (FP)
3. **Unmatched GT elements** count as False Negatives (FN)
4. **Compute precision-recall curve** from the cumulative TP/FP counts

### Important Protocol Details

| Aspect | Protocol |
|--------|----------|
| Matching scope | Per-frame (within same frame only) |
| Matching strategy | Greedy (highest confidence first) |
| Direction handling | Consider both polyline directions, take min Chamfer |
| Perception range | [-30m, 30m] x [-15m, 15m] (only evaluate within range) |
| Confidence threshold | None (use all predictions for P-R curve) |
| GT filtering | Only GT elements with length >= 2m |
| Point count (K) | Same for pred and GT (20 points) |

### Coordinate System for Evaluation

All evaluation is done in the **ego-vehicle coordinate frame**:
- Origin at ego-vehicle center
- X-axis pointing forward
- Y-axis pointing left
- Units in meters

Predictions and GT are both in normalized [0, 1] coordinates during model output, then denormalized to meters for Chamfer distance computation:

```python
def denormalize_points(points_norm, perception_range):
    """
    Convert normalized [0, 1] coordinates to metric coordinates.
    
    Args:
        points_norm: (K, 2) in [0, 1]
        perception_range: [x_min, y_min, x_max, y_max]
    
    Returns:
        points_metric: (K, 2) in meters
    """
    x_min, y_min, x_max, y_max = perception_range
    points_metric = np.zeros_like(points_norm)
    points_metric[:, 0] = points_norm[:, 0] * (x_max - x_min) + x_min
    points_metric[:, 1] = points_norm[:, 1] * (y_max - y_min) + y_min
    return points_metric
```

---

## Running Evaluation

### Single-GPU Evaluation

```bash
python tools/test.py \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    work_dirs/streammapnet_r50/epoch_24.pth \
    --eval chamfer \
    --gpu-ids 0
```

### Multi-GPU Evaluation

```bash
bash tools/dist_test.sh \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    work_dirs/streammapnet_r50/epoch_24.pth \
    4 \
    --eval chamfer
```

### Evaluation with Temporal Consistency Metrics

```bash
python tools/test.py \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    work_dirs/streammapnet_r50/epoch_24.pth \
    --eval chamfer temporal \
    --eval-options sequence_mode=True
```

### Evaluation Options

| Flag | Description |
|------|-------------|
| `--eval chamfer` | Standard Chamfer-based AP evaluation |
| `--eval temporal` | Include temporal consistency metrics |
| `--show-dir vis_results/` | Save visualization of predictions |
| `--eval-options threshold=1.0` | Custom AP threshold |
| `--format-only` | Save predictions without evaluation |

---

## Visualization

### Per-Frame Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_map_predictions(pred_elements, gt_elements, perception_range,
                               save_path=None):
    """
    Visualize predicted and GT map elements in BEV.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    class_colors = {
        0: 'orange',    # Lane divider
        1: 'green',     # Road boundary
        2: 'blue',      # Pedestrian crossing
    }
    class_names = ['Lane Divider', 'Road Boundary', 'Ped Crossing']
    
    # GT
    ax = axes[0]
    ax.set_title('Ground Truth')
    for elem in gt_elements:
        pts = elem['points']  # (K, 2) in meters
        color = class_colors[elem['label']]
        ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=2)
    
    ax.set_xlim(perception_range[1], perception_range[3])
    ax.set_ylim(perception_range[0], perception_range[2])
    ax.set_aspect('equal')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    
    # Predictions
    ax = axes[1]
    ax.set_title('Predictions')
    for elem in pred_elements:
        pts = elem['points']
        color = class_colors[elem['label']]
        alpha = min(1.0, elem['score'] + 0.3)
        ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=2, alpha=alpha)
    
    ax.set_xlim(perception_range[1], perception_range[3])
    ax.set_ylim(perception_range[0], perception_range[2])
    ax.set_aspect('equal')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    
    # Legend
    legend_elements = [plt.Line2D([0], [0], color=c, label=n) 
                       for c, n in zip(class_colors.values(), class_names)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Temporal Sequence Visualization

```bash
# Generate video of predictions across a sequence
python tools/visualize_sequence.py \
    --config configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    --checkpoint work_dirs/streammapnet_r50/epoch_24.pth \
    --scene-token <scene_token> \
    --output-dir vis_results/sequence/ \
    --format video
```

---

## Benchmark Results Reference

### nuScenes val set (24 epochs, ResNet-50)

| Method | Divider | Boundary | Ped Cross | mAP |
|--------|---------|----------|-----------|-----|
| HDMapNet | 18.5 | 37.6 | 14.1 | 23.4 |
| VectorMapNet | 36.2 | 43.5 | 28.5 | 36.1 |
| MapTR | 51.5 | 53.1 | 46.3 | 50.3 |
| MapTRv2 | 55.7 | 57.4 | 49.2 | 54.1 |
| StreamMapNet | 56.3 | 55.8 | 50.1 | 54.1 |

### Argoverse 2 val set

| Method | Divider | Boundary | Ped Cross | mAP |
|--------|---------|----------|-----------|-----|
| MapTR | 58.7 | 60.3 | 52.1 | 57.0 |
| StreamMapNet | 62.4 | 63.1 | 56.8 | 60.8 |

---

## Common Evaluation Pitfalls

### 1. Inconsistent Perception Range

Ensure the same perception range is used for both training and evaluation. Mismatched ranges lead to unfair comparisons.

### 2. Direction Ambiguity

Always evaluate with direction-aware Chamfer distance. Without it, a correctly detected polyline with reversed point order would be penalized heavily.

### 3. Point Count Mismatch

If the model outputs a different number of points than the GT, resample both to the same K before computing Chamfer distance. The standard is K=20.

### 4. Confidence Calibration

AP is sensitive to confidence score ordering. Poor calibration (e.g., all scores clustered near 0.5) can hurt AP even with good detection quality. Use the raw classification logits (after sigmoid) as confidence scores.

### 5. Scene Boundary Handling

When evaluating temporal consistency, exclude the first frame of each scene (where no temporal context is available) from temporal metrics but include it in per-frame mAP.

### 6. Evaluation on v1.0-mini

The mini split is too small for reliable evaluation (only 10 scenes). Use it only for sanity checks, not for reporting metrics.
