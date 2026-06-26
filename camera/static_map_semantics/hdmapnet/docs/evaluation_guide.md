# HDMapNet Evaluation Guide

## 1. Overview of Evaluation for HD Map Construction

HD map construction from onboard sensors poses a unique evaluation challenge. Unlike
standard semantic segmentation (where per-pixel IoU suffices), HD maps consist of
**vectorized geometric primitives** -- polylines representing lane dividers, road
boundaries, and pedestrian crossings. These elements have thin, elongated spatial
extent; a prediction shifted by just one pixel can drop IoU to near zero while
remaining geometrically accurate.

### Why Standard IoU Is Insufficient

```
Ground Truth:       ─────────────────────
Prediction (1px off): ─────────────────────

Pixel-wise IoU ≈ 0  (no overlap on a 1-px-wide line)
Geometric error  ≈ 0.1 m  (perfectly acceptable for planning)
```

Standard IoU penalizes spatial misalignment disproportionately for thin structures.
A prediction that captures the correct topology and is within centimeters of the
ground truth can receive a near-zero IoU score.

### HDMapNet's Dual Evaluation Protocol

HDMapNet addresses this by introducing **two complementary evaluation protocols**:

| Protocol | What It Measures | Representation |
|----------|-----------------|----------------|
| Rasterized (Semantic) | Dense BEV segmentation quality | Binary grid per class |
| Vectorized (Geometric) | Structural fidelity of extracted polylines | Ordered point sequences |

The rasterized protocol evaluates the neural network output directly. The vectorized
protocol evaluates the final map representation after post-processing, measuring
what actually matters for downstream planning and localization.

---

## 2. Rasterized Metrics (Semantic Evaluation)

The rasterized evaluation treats BEV map prediction as a multi-class semantic
segmentation problem on a discretized grid.

### Grid Specification

- **Resolution**: 0.15 m/pixel (default)
- **Spatial extent**: 60 m x 30 m around the ego vehicle ([-30, 30] x [-15, 15] meters)
- **Grid size**: 400 x 200 pixels

### Classes Evaluated

| Class ID | Class Name | Description |
|----------|-----------|-------------|
| 0 | Lane Divider | Dashed and solid lane markings |
| 1 | Pedestrian Crossing | Crosswalk regions |
| 2 | Road Boundary | Curbs and road edges |

### IoU Computation

For each class `c`, the model outputs a probability map `P_c` of shape `(H, W)`.
Evaluation proceeds as:

```python
import numpy as np

def compute_iou(pred_prob: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute IoU for a single class on BEV grid.
    
    Args:
        pred_prob: (H, W) predicted probability map, values in [0, 1]
        gt_mask:   (H, W) ground truth binary mask
        threshold: binarization threshold for predictions
    
    Returns:
        IoU score for this class
    """
    pred_binary = (pred_prob >= threshold).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_mask).sum()
    union = np.logical_or(pred_binary, gt_mask).sum()
    
    if union == 0:
        return float('nan')  # No GT and no prediction for this class
    
    return intersection / union
```

### Aggregation

```python
def compute_mean_iou(per_class_ious: dict) -> float:
    """Compute mean IoU across classes, ignoring NaN entries."""
    valid = [v for v in per_class_ious.values() if not np.isnan(v)]
    return np.mean(valid)
```

Per-sample IoU is computed, then averaged across all samples in the validation set.
The final reported metrics are:

- **Divider IoU**: IoU for lane divider class
- **Crossing IoU**: IoU for pedestrian crossing class
- **Boundary IoU**: IoU for road boundary class
- **mIoU**: Arithmetic mean of the three per-class IoU values

---

## 3. Vectorized Metrics (The HDMapNet Contribution)

The vectorized evaluation protocol is HDMapNet's primary methodological contribution
to the field. It evaluates predicted polylines against ground-truth polylines using
geometric distance measures.

### 3.1 Chamfer Distance (CD)

The Chamfer Distance measures the average closest-point distance between two point
sets. For two polylines resampled to point sets `A` and `B`:

```
CD(A, B) = (1/|A|) * sum_{a in A} min_{b in B} ||a - b||_2
         + (1/|B|) * sum_{b in B} min_{a in A} ||b - a||_2
```

In practice, the **symmetric** Chamfer Distance is used (average of both
directions):

```python
import numpy as np
from scipy.spatial.distance import cdist

def chamfer_distance(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    """
    Compute symmetric Chamfer Distance between two polylines.
    
    Args:
        pred_points: (N, 2) array of predicted polyline points
        gt_points:   (M, 2) array of ground truth polyline points
    
    Returns:
        Symmetric Chamfer Distance in meters
    """
    # Pairwise distance matrix: (N, M)
    dist_matrix = cdist(pred_points, gt_points, metric='euclidean')
    
    # For each predicted point, find closest GT point
    pred_to_gt = dist_matrix.min(axis=1).mean()
    
    # For each GT point, find closest predicted point
    gt_to_pred = dist_matrix.min(axis=0).mean()
    
    return (pred_to_gt + gt_to_pred) / 2.0
```

### 3.2 Average Precision with Distance Thresholds

HDMapNet defines AP using Chamfer Distance as the matching criterion instead of
IoU overlap:

**Algorithm:**

1. For each predicted polyline `p_i`, compute CD to every GT polyline of the same
   class.
2. Find the closest GT polyline `g_j` (minimum CD).
3. If `CD(p_i, g_j) < threshold_tau` and `g_j` has not been matched yet, mark
   `p_i` as a **True Positive (TP)**.
4. Otherwise, mark `p_i` as a **False Positive (FP)**.
5. GT polylines that remain unmatched are **False Negatives (FN)**.
6. Sort predictions by confidence score, compute precision-recall curve, then
   compute AP as area under the curve.

### Distance Thresholds

| Threshold | Interpretation |
|-----------|---------------|
| 0.5 m | Strict -- suitable for lane-level localization |
| 1.0 m | Moderate -- acceptable for most planning tasks |
| 1.5 m | Relaxed -- captures topological correctness |

### AP Computation

```python
import numpy as np
from typing import List, Tuple

def compute_ap(
    predictions: List[Tuple[np.ndarray, float]],  # (polyline_points, confidence)
    ground_truths: List[np.ndarray],               # list of GT polyline points
    threshold: float = 1.0                         # CD threshold in meters
) -> float:
    """
    Compute Average Precision for a single class at a given CD threshold.
    
    Args:
        predictions: List of (points, confidence) tuples, sorted by confidence desc
        ground_truths: List of GT polyline point arrays
        threshold: Chamfer Distance threshold for TP/FP determination
    
    Returns:
        AP value (area under precision-recall curve)
    """
    # Sort predictions by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    n_gt = len(ground_truths)
    if n_gt == 0:
        return 0.0
    
    matched_gt = set()
    tp_list = []
    fp_list = []
    
    for pred_points, conf in predictions:
        min_cd = float('inf')
        best_gt_idx = -1
        
        for gt_idx, gt_points in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            cd = chamfer_distance(pred_points, gt_points)
            if cd < min_cd:
                min_cd = cd
                best_gt_idx = gt_idx
        
        if min_cd < threshold and best_gt_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp_list.append(0)
            fp_list.append(1)
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    
    recall = tp_cumsum / n_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Compute AP using 11-point interpolation (PASCAL VOC style)
    ap = 0.0
    for r_threshold in np.linspace(0, 1, 11):
        precisions_at_recall = precision[recall >= r_threshold]
        if len(precisions_at_recall) == 0:
            p = 0.0
        else:
            p = precisions_at_recall.max()
        ap += p / 11.0
    
    return ap
```

### Reported Metrics

- **AP@0.5**: Average Precision at 0.5 m threshold
- **AP@1.0**: Average Precision at 1.0 m threshold
- **AP@1.5**: Average Precision at 1.5 m threshold
- **mAP**: Mean over all three thresholds: `(AP@0.5 + AP@1.0 + AP@1.5) / 3`

---

## 4. Post-Processing for Vectorized Evaluation

Converting dense BEV semantic predictions into vectorized polylines requires a
multi-stage post-processing pipeline. This is critical because the vectorized
metrics cannot be applied directly to the rasterized output.

### Pipeline Diagram

```
BEV Semantic Map (H x W per class)
        │
        ▼
┌─────────────────────┐
│  Threshold at 0.5   │  → Binary mask per class
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Skeletonization   │  → Thin binary structures to 1-px width
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Connected Components │  → Separate individual instances
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Polyline Tracing   │  → Convert skeleton pixels to ordered points
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Douglas-Peucker      │  → Simplify polylines (reduce point count)
└─────────────────────┘
        │
        ▼
Vectorized Polylines (list of Nx2 arrays)
```

### Step-by-Step Implementation

#### 4.1 Semantic Map Thresholding

```python
import numpy as np

def threshold_semantic_map(pred_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarize the predicted probability map."""
    return (pred_map >= threshold).astype(np.uint8)
```

#### 4.2 Skeletonization

Thin structures (1-3 pixels wide in the BEV grid) are reduced to single-pixel
skeletons to enable clean polyline extraction:

```python
from skimage.morphology import skeletonize

def extract_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Reduce binary mask to 1-pixel-wide skeleton.
    Uses Zhang-Suen thinning algorithm.
    """
    return skeletonize(binary_mask).astype(np.uint8)
```

#### 4.3 Connected Component Extraction

Each connected component corresponds to one map element instance:

```python
import cv2

def extract_components(skeleton: np.ndarray, min_pixels: int = 10):
    """
    Extract connected components from skeleton.
    Filter out small components (noise).
    
    Args:
        skeleton: (H, W) binary skeleton image
        min_pixels: minimum component size to keep
    
    Returns:
        List of binary masks, one per component
    """
    num_labels, labels = cv2.connectedComponents(skeleton)
    
    components = []
    for label_id in range(1, num_labels):  # Skip background (0)
        component_mask = (labels == label_id).astype(np.uint8)
        if component_mask.sum() >= min_pixels:
            components.append(component_mask)
    
    return components
```

#### 4.4 Polyline Tracing

Convert each skeleton component into an ordered sequence of points:

```python
import numpy as np

def trace_polyline(component_mask: np.ndarray) -> np.ndarray:
    """
    Trace skeleton pixels into an ordered polyline.
    Uses endpoint detection and greedy neighbor walking.
    
    Returns:
        (N, 2) array of ordered points in pixel coordinates
    """
    # Find all skeleton pixel coordinates
    ys, xs = np.where(component_mask > 0)
    points = np.column_stack([xs, ys])
    
    if len(points) < 2:
        return points
    
    # Find endpoints (pixels with exactly 1 neighbor in 8-connectivity)
    # Start tracing from an endpoint if available
    # ... (greedy neighbor walking algorithm)
    
    return ordered_points
```

#### 4.5 Douglas-Peucker Simplification

Reduce point count while preserving geometric shape:

```python
from shapely.geometry import LineString

def simplify_polyline(
    points: np.ndarray,
    epsilon: float = 0.3  # tolerance in meters
) -> np.ndarray:
    """
    Apply Douglas-Peucker simplification.
    
    Args:
        points: (N, 2) polyline points in metric coordinates
        epsilon: maximum perpendicular distance tolerance
    
    Returns:
        Simplified polyline as (M, 2) array where M <= N
    """
    line = LineString(points)
    simplified = line.simplify(epsilon, preserve_topology=True)
    return np.array(simplified.coords)
```

#### 4.6 Coordinate Conversion

Points must be converted from pixel coordinates to metric coordinates for CD
computation:

```python
def pixel_to_metric(
    points_px: np.ndarray,
    resolution: float = 0.15,     # meters per pixel
    origin: tuple = (-30.0, -15.0) # (x_min, y_min) in meters
) -> np.ndarray:
    """Convert pixel coordinates to ego-frame metric coordinates."""
    points_m = points_px * resolution + np.array(origin)
    return points_m
```

### 4.7 Matching Predicted Polylines to GT

Before computing AP, predictions and ground truths must be associated per class:

```python
def match_predictions_to_gt(
    pred_polylines: List[Tuple[np.ndarray, float]],  # (points, confidence)
    gt_polylines: List[np.ndarray],
    threshold: float
) -> Tuple[int, int, int]:
    """
    Match predictions to GT using greedy assignment by confidence.
    Returns (TP, FP, FN) counts.
    """
    # Sort by confidence descending
    preds_sorted = sorted(pred_polylines, key=lambda x: x[1], reverse=True)
    matched = set()
    tp, fp = 0, 0
    
    for pred_pts, conf in preds_sorted:
        best_cd = float('inf')
        best_idx = -1
        for i, gt_pts in enumerate(gt_polylines):
            if i in matched:
                continue
            cd = chamfer_distance(pred_pts, gt_pts)
            if cd < best_cd:
                best_cd = cd
                best_idx = i
        
        if best_cd < threshold and best_idx >= 0:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1
    
    fn = len(gt_polylines) - len(matched)
    return tp, fp, fn
```

---

## 5. Evaluation Pipeline (Step by Step)

The full evaluation pipeline proceeds as follows:

```
┌──────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Load val split (nuScenes mini/full)                      │
│  2. Run model inference on each sample                       │
│  3. For each sample:                                         │
│     a. Collect BEV semantic predictions                      │
│     b. Compute rasterized IoU vs GT raster                   │
│     c. Apply post-processing → extract polylines             │
│     d. Compute CD to GT polylines                            │
│     e. Determine TP/FP at each threshold                     │
│  4. Aggregate rasterized IoU across dataset                  │
│  5. Compute AP at {0.5, 1.0, 1.5} m across dataset          │
│  6. Report final metrics                                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Step 1: Run Model Inference

```bash
# Run inference on nuScenes validation set
python evaluate.py \
    --config configs/hdmapnet_nusc.yaml \
    --checkpoint checkpoints/hdmapnet_lss_epoch24.pth \
    --split val \
    --output-dir results/val_predictions/ \
    --batch-size 4 \
    --gpus 1
```

### Step 2: Generate BEV Semantic Predictions

The model outputs per-class probability maps for each sample:

```python
# Output format per sample
{
    "sample_token": "abc123...",
    "predictions": {
        "divider": np.ndarray,       # (200, 400) float32, range [0, 1]
        "crossing": np.ndarray,      # (200, 400) float32, range [0, 1]
        "boundary": np.ndarray,      # (200, 400) float32, range [0, 1]
    }
}
```

### Step 3: Apply Post-Processing

```bash
# Generate vectorized polylines from BEV predictions
python postprocess.py \
    --predictions results/val_predictions/ \
    --output results/val_polylines/ \
    --threshold 0.5 \
    --min-component-size 10 \
    --simplification-epsilon 0.3
```

### Step 4: Compute Rasterized Metrics

```bash
# Evaluate rasterized IoU
python eval_rasterized.py \
    --predictions results/val_predictions/ \
    --gt-dir data/nuscenes/gt_raster/ \
    --output results/rasterized_metrics.json
```

Expected output:

```json
{
    "divider_iou": 0.406,
    "crossing_iou": 0.187,
    "boundary_iou": 0.395,
    "mean_iou": 0.329
}
```

### Step 5: Compute Vectorized Metrics

```bash
# Evaluate vectorized AP at multiple thresholds
python eval_vectorized.py \
    --predictions results/val_polylines/ \
    --gt-dir data/nuscenes/gt_vectors/ \
    --thresholds 0.5 1.0 1.5 \
    --output results/vectorized_metrics.json
```

Expected output:

```json
{
    "divider": {"ap_0.5": 0.098, "ap_1.0": 0.162, "ap_1.5": 0.241},
    "crossing": {"ap_0.5": 0.045, "ap_1.0": 0.087, "ap_1.5": 0.132},
    "boundary": {"ap_0.5": 0.112, "ap_1.0": 0.188, "ap_1.5": 0.279},
    "mean": {"ap_0.5": 0.085, "ap_1.0": 0.146, "ap_1.5": 0.217},
    "mAP": 0.149
}
```

### Step 6: Aggregate and Report

```bash
# Generate full evaluation report
python report_metrics.py \
    --rasterized results/rasterized_metrics.json \
    --vectorized results/vectorized_metrics.json \
    --output results/evaluation_report.txt
```

---

## 6. Baseline Results on nuScenes Val

The following results are reported on the nuScenes validation split using the
standard 60 m x 30 m perception range at 0.15 m resolution.

### Rasterized Metrics (IoU)

| Method | Divider IoU | Crossing IoU | Boundary IoU | mIoU |
|--------|-------------|--------------|--------------|------|
| HDMapNet-IPM | 38.7 | 17.2 | 39.3 | 31.7 |
| HDMapNet-LSS | 40.6 | 18.7 | 39.5 | 32.9 |
| HDMapNet-Surround | 21.7 | 5.7 | 37.6 | 21.7 |

### Vectorized Metrics (AP)

| Method | AP@0.5 | AP@1.0 | AP@1.5 | mAP |
|--------|--------|--------|--------|-----|
| HDMapNet-IPM | 8.2 | 14.3 | 21.7 | 14.7 |
| HDMapNet-LSS | 9.8 | 16.2 | 24.1 | 16.7 |
| HDMapNet-Surround | - | - | - | - |

### Method Variants Explained

| Variant | View Transform | Input |
|---------|---------------|-------|
| HDMapNet-IPM | Inverse Perspective Mapping | Front camera only |
| HDMapNet-LSS | Lift-Splat-Shoot (learned depth) | Surround cameras (6x) |
| HDMapNet-Surround | Direct surround projection | Surround cameras (6x) |

**Key observations:**

- HDMapNet-LSS achieves the best results across both rasterized and vectorized
  metrics, validating the importance of learned depth for view transformation.
- Pedestrian crossing consistently has the lowest scores due to its irregular
  geometry and sparse annotations.
- The gap between rasterized mIoU (32.9) and vectorized mAP (16.7) highlights how
  post-processing degrades the final map quality.

---

## 7. Running Evaluation

### Configuration File

```yaml
# configs/eval_config.yaml
evaluation:
  split: val
  dataset: nuscenes
  version: v1.0-trainval
  
  # Spatial parameters
  xbound: [-30.0, 30.0, 0.15]   # [min, max, resolution] in meters
  ybound: [-15.0, 15.0, 0.15]
  
  # Rasterized evaluation
  rasterized:
    threshold: 0.5
    classes: [divider, crossing, boundary]
  
  # Vectorized evaluation
  vectorized:
    cd_thresholds: [0.5, 1.0, 1.5]
    min_confidence: 0.1
    
  # Post-processing
  postprocess:
    skeleton_method: zhang        # 'zhang' or 'lee'
    min_component_pixels: 10
    simplification_epsilon: 0.3   # Douglas-Peucker tolerance (meters)
    resample_interval: 0.5        # Point spacing for CD computation (meters)
    
  # Output
  output_dir: results/
  save_visualizations: true
  vis_samples: 50                 # Number of samples to visualize
```

### Full Evaluation Command

```bash
# Complete evaluation pipeline (rasterized + vectorized)
python tools/evaluate.py \
    --config configs/eval_config.yaml \
    --checkpoint checkpoints/hdmapnet_lss_epoch24.pth \
    --gpus 0 \
    --workers 8
```

### Evaluating Only Rasterized Metrics

```bash
python tools/evaluate.py \
    --config configs/eval_config.yaml \
    --checkpoint checkpoints/hdmapnet_lss_epoch24.pth \
    --eval-mode rasterized
```

### Evaluating Only Vectorized Metrics (from saved predictions)

```bash
python tools/evaluate.py \
    --config configs/eval_config.yaml \
    --predictions-dir results/val_predictions/ \
    --eval-mode vectorized
```

### Visualization of Results

```bash
# Generate side-by-side visualizations (prediction vs GT)
python tools/visualize.py \
    --config configs/eval_config.yaml \
    --predictions results/val_predictions/ \
    --gt-dir data/nuscenes/gt_vectors/ \
    --output vis_output/ \
    --num-samples 20
```

Visualization output includes:

```
vis_output/
├── sample_001_raster.png      # BEV prediction overlaid on GT raster
├── sample_001_vector.png      # Predicted polylines vs GT polylines
├── sample_001_camera.png      # Input camera views with projected map
└── ...
```

### Visualization Code Snippet

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction_vs_gt(
    pred_polylines: list,
    gt_polylines: list,
    extent: tuple = (-30, 30, -15, 15),
    save_path: str = None
):
    """
    Plot predicted and GT polylines in BEV.
    
    Args:
        pred_polylines: List of (N, 2) arrays (predicted)
        gt_polylines: List of (M, 2) arrays (ground truth)
        extent: (x_min, x_max, y_min, y_max) in meters
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot GT polylines (green)
    for gt in gt_polylines:
        ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=2, label='GT')
    
    # Plot predicted polylines (red)
    for pred in pred_polylines:
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=1.5, label='Pred')
    
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('BEV Map: Prediction vs Ground Truth')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

---

## 8. Common Evaluation Pitfalls

### 8.1 Post-Processing Sensitivity

The vectorized AP is highly sensitive to post-processing hyperparameters:

| Parameter | Effect of Increasing |
|-----------|---------------------|
| `threshold` (binarization) | Fewer but higher-confidence predictions |
| `min_component_pixels` | Filters more small fragments (reduces FP, may increase FN) |
| `simplification_epsilon` | Coarser polylines (faster CD, but loses detail) |
| `resample_interval` | Fewer points for CD (faster but less precise) |

**Recommendation:** Always report post-processing parameters alongside vectorized
metrics. Small changes (e.g., threshold from 0.4 to 0.6) can shift AP by 2-5 points.

```python
# Example: Sensitivity sweep
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for t in thresholds:
    ap = evaluate_vectorized(predictions, gt, binarization_threshold=t)
    print(f"threshold={t:.1f} -> AP@1.0={ap:.3f}")

# Typical output:
# threshold=0.3 -> AP@1.0=0.121  (too many FP fragments)
# threshold=0.4 -> AP@1.0=0.148
# threshold=0.5 -> AP@1.0=0.162  (default)
# threshold=0.6 -> AP@1.0=0.155
# threshold=0.7 -> AP@1.0=0.131  (too many FN, predictions too sparse)
```

### 8.2 Coordinate System Alignment

HD map evaluation requires precise coordinate system handling:

```
                    +X (forward)
                     │
                     │
         +Y ────────┼──────── -Y
        (left)       │        (right)
                     │
                    -X (backward)
```

**Common mistakes:**

- Confusing **ego frame** (moves with the vehicle) and **global frame** (fixed
  world coordinates). Evaluation is always in **ego frame** per sample.
- Swapping X/Y axes between image coordinates (row, col) and BEV coordinates
  (forward, lateral).
- Off-by-one errors when converting between pixel indices and metric coordinates.

```python
# WRONG: Treating (row, col) as (x, y)
points_metric = pixel_coords * resolution  # Axes are swapped!

# CORRECT: Convert (row, col) to (x_forward, y_lateral)
x_metric = (col_idx * resolution) + x_min   # cols correspond to lateral
y_metric = (row_idx * resolution) + y_min   # rows correspond to forward
# Check your specific convention! HDMapNet uses x=lateral, y=forward in some code.
```

### 8.3 Handling Missing Annotations

nuScenes HD map annotations are not exhaustive. Some map elements may be missing
from the ground truth:

- Newly constructed roads or temporary markings
- Occluded or worn-out lane markings
- Regions outside the annotated map extent

**Strategy:** Only evaluate within the annotated spatial extent. If a predicted
polyline extends beyond the GT coverage area, clip it to the evaluation boundary
before computing CD.

```python
from shapely.geometry import LineString, box

def clip_to_bounds(polyline: np.ndarray, bounds: tuple) -> np.ndarray:
    """Clip polyline to evaluation boundary."""
    x_min, x_max, y_min, y_max = bounds
    boundary = box(x_min, y_min, x_max, y_max)
    line = LineString(polyline)
    clipped = line.intersection(boundary)
    if clipped.is_empty:
        return np.array([]).reshape(0, 2)
    return np.array(clipped.coords)
```

### 8.4 Fair Comparison with End-to-End Vectorized Methods

Recent methods like **MapTR**, **VectorMapNet**, and **MapTRv2** directly predict
vectorized map elements without requiring post-processing. This creates an apples-
to-oranges comparison problem:

| Aspect | HDMapNet | End-to-End (MapTR) |
|--------|----------|-------------------|
| Output | Raster BEV + post-processing | Direct polyline queries |
| Post-processing needed | Yes (skeletonization, tracing) | No |
| Confidence scores | From semantic probability | From detection head |
| Instance separation | Connected components | Learned instance queries |

**Fair comparison guidelines:**

1. Use the **same evaluation code** (preferably the one from MapTR or a shared
   benchmark like OpenLaneV2).
2. Report post-processing parameters when evaluating HDMapNet.
3. Acknowledge that post-processing is a bottleneck: HDMapNet's neural network may
   produce excellent rasterized predictions that are degraded by imperfect
   vectorization.
4. When comparing, always include the rasterized IoU alongside vectorized AP to
   separate model quality from post-processing quality.

### 8.5 Polyline Resampling for CD Computation

CD is sensitive to point density. A polyline with 100 points will dominate the CD
over a polyline with 10 points. Always resample to uniform spacing before computing
CD:

```python
from shapely.geometry import LineString

def resample_polyline(points: np.ndarray, interval: float = 0.5) -> np.ndarray:
    """
    Resample polyline to uniform point spacing.
    
    Args:
        points: (N, 2) polyline
        interval: target spacing between points in meters
    
    Returns:
        Resampled polyline with uniform spacing
    """
    line = LineString(points)
    num_points = max(int(line.length / interval), 2)
    distances = np.linspace(0, line.length, num_points)
    resampled = [line.interpolate(d) for d in distances]
    return np.array([(p.x, p.y) for p in resampled])
```

### 8.6 Evaluation Speed Considerations

For large validation sets (6019 samples in nuScenes val), evaluation can be slow:

| Component | Approximate Time |
|-----------|-----------------|
| Model inference (6019 samples) | ~30 min (1 GPU, batch=4) |
| Post-processing | ~5 min |
| Rasterized IoU computation | ~1 min |
| Vectorized AP computation | ~10-20 min (CD is O(N*M) per pair) |

**Optimization tips:**

- Use `scipy.spatial.cKDTree` instead of brute-force `cdist` for CD computation.
- Parallelize per-sample evaluation with multiprocessing.
- Cache post-processed polylines to disk for repeated AP threshold sweeps.

```python
from scipy.spatial import cKDTree

def chamfer_distance_fast(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """Fast Chamfer Distance using KD-Trees."""
    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    
    dist_pred_to_gt, _ = tree_gt.query(pred_pts)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts)
    
    return (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2.0
```

---

## Summary

The HDMapNet evaluation framework provides a principled way to measure both
rasterized segmentation quality and vectorized geometric accuracy of predicted HD
maps. The key takeaways for practitioners:

1. **Always report both** rasterized IoU and vectorized AP -- they measure different
   aspects of performance.
2. **Post-processing is a critical variable** -- document all hyperparameters when
   reporting vectorized metrics.
3. **Chamfer Distance with AP** is the standard for vectorized map evaluation,
   adopted by subsequent works (MapTR, VectorMapNet, StreamMapNet).
4. **Coordinate systems and resampling** are common sources of bugs -- validate
   with simple synthetic examples before running on the full dataset.
5. **Use KD-trees** for efficient CD computation at scale.
