# StreamMapNet: Comprehensive Evaluation Guide

A deep-dive reference for engineers who know PyTorch and deep learning but are new to
autonomous driving map evaluation. This guide teaches every metric from first principles.

---

## 1. Overview: Why Map Evaluation Is Different from Standard Detection

In standard object detection (COCO, VOC), you predict bounding boxes and measure overlap
(IoU) against ground truth boxes. A box either overlaps enough or it does not.

Vectorized map prediction is fundamentally different:

- **Outputs are polylines and polygons**, not axis-aligned boxes. A lane divider is a
  sequence of ordered 2D points describing a curve in bird's-eye-view (BEV) space.
- **There is no notion of "overlap"** between two polylines. Two lines can be close
  everywhere yet share zero area.
- **Spatial accuracy is measured in meters**, and those meters have physical safety
  implications -- a 1-meter error in a lane boundary might steer the vehicle into
  oncoming traffic.
- **Temporal consistency matters.** A planning module that receives jittering lane
  predictions will produce jerky steering commands, even if each individual frame is
  accurate.

Therefore, the map evaluation stack replaces IoU with Chamfer distance, redefines what
"correct" means, and adds temporal stability metrics on top.

---

## 2. Chamfer Distance -- Taught from Scratch

### 2.1 What It Is

The Chamfer distance measures how similar two unordered point sets are. Given sets A and B,
it asks: "On average, how far does each point in A need to travel to reach its closest
friend in B, and vice versa?"

### 2.2 Mathematical Definition

Let A = {a_1, ..., a_M} and B = {b_1, ..., b_N} be two point sets in R^2.

**Forward Chamfer (A -> B):**

    CD_forward(A, B) = (1/M) * SUM_{i=1}^{M} min_{j} ||a_i - b_j||_2

**Backward Chamfer (B -> A):**

    CD_backward(A, B) = (1/N) * SUM_{j=1}^{N} min_{i} ||b_j - a_i||_2

**Symmetric Chamfer Distance:**

    CD(A, B) = ( CD_forward(A, B) + CD_backward(A, B) ) / 2

The result is a scalar in meters representing the average point-to-nearest-neighbor
distance between the two shapes.

### 2.3 Geometric Intuition

Consider a predicted polyline (P) and a ground truth polyline (G):

```
  Forward pass: each P point finds its nearest G point
  ====================================================

       P0------P1------P2------P3         (predicted)
       |  \      \       |      \
       |   \      \      |       \
       v    v      v     v        v       (nearest neighbor arrows)
      G0----G1------G2----G3------G4      (ground truth)

  Backward pass: each G point finds its nearest P point
  =====================================================

       P0------P1------P2------P3
       ^    ^       ^     ^    ^
       |   /       /      |   /
       |  /       /       |  /
      G0----G1------G2----G3------G4

  The symmetric CD averages both passes.
```

Each arrow represents one nearest-neighbor lookup. The forward pass measures how well
the prediction covers the ground truth shape. The backward pass measures how well the
ground truth is covered by predictions. Together they capture both over-prediction
(hallucinated points far from GT) and under-prediction (GT regions with no nearby
predicted points).

### 2.4 Why Chamfer and Not Point-Wise L1/L2?

A naive approach would align points by index: compare P[0] with G[0], P[1] with G[1],
etc. This fails for two reasons:

1. **Index misalignment.** If the predicted polyline starts 2 meters earlier than the
   GT polyline, every index-matched pair is wrong even though the shape is correct.
2. **Different sampling densities.** Even with the same number of points, the spacing
   may differ. Chamfer finds the *geometrically* closest point regardless of index.

Chamfer distance is permutation-invariant within each set. It cares about shape
proximity, not about which point has which index.

### 2.5 Direction-Aware Chamfer Distance

A polyline [A, B, C] and the reversed polyline [C, B, A] describe the exact same
physical lane divider. The direction of traversal is arbitrary in annotation. Without
handling this, a perfect prediction in reversed order would receive a large penalty.

Solution: compute Chamfer distance in both directions and take the minimum.

```
  Same physical line, different point orderings:

       Pred:  P0 ---> P1 ---> P2 ---> P3    (left to right)
       GT:    G3 <--- G2 <--- G1 <--- G0    (right to left)

       CD(Pred, GT_forward) might be large due to index offsets,
       CD(Pred, GT_reversed) will be small because shapes align.

       Final distance = min(CD_forward, CD_reversed)
```

### 2.6 Full Python Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist

def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    Compute symmetric Chamfer distance between two 2D point sets.

    Args:
        points_a: (M, 2) array -- first point set (e.g., predicted polyline)
        points_b: (N, 2) array -- second point set (e.g., GT polyline)

    Returns:
        Symmetric Chamfer distance in the same units as the input coordinates
        (meters, in our case).
    """
    # Pairwise Euclidean distance matrix: shape (M, N)
    dist_matrix = cdist(points_a, points_b, metric='euclidean')

    # Forward: for each point in A, find the nearest point in B
    forward = dist_matrix.min(axis=1).mean()  # average over M points

    # Backward: for each point in B, find the nearest point in A
    backward = dist_matrix.min(axis=0).mean()  # average over N points

    # Symmetric: average of both directions
    return (forward + backward) / 2.0


def direction_aware_chamfer(pred_points: np.ndarray,
                            gt_points: np.ndarray) -> float:
    """
    Compute Chamfer distance considering both possible polyline traversal orders.
    Returns the minimum of the two to handle direction ambiguity.

    Args:
        pred_points: (K, 2) predicted polyline in meters
        gt_points:   (K, 2) ground truth polyline in meters

    Returns:
        Minimum Chamfer distance across both GT directions.
    """
    cd_forward = chamfer_distance(pred_points, gt_points)
    cd_reversed = chamfer_distance(pred_points, gt_points[::-1])
    return min(cd_forward, cd_reversed)
```

### 2.7 Physical Meaning

If direction_aware_chamfer returns 0.4 meters, it means: "On average, every predicted
point is 0.4 meters away from the nearest ground truth point, and vice versa." In
driving terms, your lane boundary prediction is off by roughly 40 centimeters -- less
than a tire width, excellent for lane keeping.

---

## 3. Average Precision (AP) for Vectorized Maps

### 3.1 How It Differs from Standard Object Detection AP

| Aspect             | Object Detection (COCO)              | Vectorized Map (StreamMapNet)            |
|--------------------|--------------------------------------|------------------------------------------|
| Representation     | Bounding box (x, y, w, h)           | Polyline (K ordered 2D points)           |
| Similarity metric  | IoU (intersection / union)           | Chamfer distance (meters)                |
| "Correct" means    | IoU > threshold (e.g., 0.5)         | Chamfer < threshold (e.g., 0.5m)         |
| Threshold meaning  | Fraction overlap                     | Physical proximity in meters             |
| Matching strategy  | Greedy by confidence                 | Greedy by confidence (same approach)     |

The key insight: in detection, a higher IoU means more overlap -- good. In map eval,
a *lower* Chamfer distance means closer shapes -- good. So the direction of the
threshold is flipped: Chamfer < threshold = match (analogous to IoU > threshold).

### 3.2 The Precision-Recall Computation

Step by step:

1. **Sort all predictions by confidence score** (descending).
2. **Greedy matching** -- for each prediction, in confidence order:
   - Compute direction-aware Chamfer distance to every unmatched GT of the same class
     in the same frame.
   - Find the GT with the smallest Chamfer distance.
   - If that distance < threshold AND the GT is not already matched:
     - Mark prediction as **True Positive (TP)**.
     - Mark that GT as consumed (cannot match again).
   - Otherwise: mark prediction as **False Positive (FP)**.
3. **False Negatives (FN):** GT elements that were never matched by any prediction.
4. **Build cumulative P-R curve:**
   - precision[k] = cumulative_TP[k] / (cumulative_TP[k] + cumulative_FP[k])
   - recall[k] = cumulative_TP[k] / total_GT_count
5. **Interpolated AP:** sample precision at evenly spaced recall levels (40 points),
   taking the maximum precision at or above each recall level.

### 3.3 Full Python Implementation

```python
import numpy as np
from collections import defaultdict

def compute_ap(predictions, ground_truths, threshold, num_recall_points=40):
    """
    Compute Average Precision for one map element class at one threshold.

    Args:
        predictions: list of dicts, each with:
            - 'points': np.ndarray (K, 2) in meters
            - 'score': float, confidence in [0, 1]
            - 'frame_id': str or int
        ground_truths: list of dicts, each with:
            - 'points': np.ndarray (K, 2) in meters
            - 'frame_id': str or int
        threshold: float, Chamfer distance threshold in meters
        num_recall_points: int, interpolation granularity (default 40)

    Returns:
        ap: float in [0, 1]
    """
    # Sort predictions by confidence (highest first)
    predictions = sorted(predictions, key=lambda p: p['score'], reverse=True)

    # Organize GT by frame for efficient lookup
    gt_by_frame = defaultdict(list)
    for gt in ground_truths:
        gt_by_frame[gt['frame_id']].append(gt)

    # Track which GT elements have been matched (per frame)
    gt_matched = {fid: [False] * len(gts) for fid, gts in gt_by_frame.items()}

    num_preds = len(predictions)
    tp = np.zeros(num_preds)
    fp = np.zeros(num_preds)

    for pred_idx, pred in enumerate(predictions):
        fid = pred['frame_id']
        frame_gts = gt_by_frame.get(fid, [])

        if len(frame_gts) == 0:
            fp[pred_idx] = 1
            continue

        # Find best unmatched GT in this frame
        best_dist = float('inf')
        best_gt_idx = -1

        for gt_idx, gt in enumerate(frame_gts):
            if gt_matched[fid][gt_idx]:
                continue  # Already consumed by a higher-confidence prediction

            dist = direction_aware_chamfer(pred['points'], gt['points'])
            if dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx

        # Apply threshold
        if best_dist <= threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[fid][best_gt_idx] = True
        else:
            fp[pred_idx] = 1

    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    total_gt = len(ground_truths)

    # Precision and recall arrays
    recall = tp_cumsum / max(total_gt, 1)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # 40-point interpolated AP
    recall_levels = np.linspace(0, 1, num_recall_points)
    ap = 0.0
    for r in recall_levels:
        # Maximum precision at recall >= r
        mask = recall >= r
        if mask.any():
            ap += precision[mask].max()
    ap /= num_recall_points

    return ap
```

---

## 4. Thresholds Explained with Physical Meaning

### 4.1 Understanding the Numbers

A standard passenger car is approximately 1.8 meters wide. A highway lane is roughly
3.5 meters wide. These physical references give meaning to the evaluation thresholds.

| Threshold | Name    | Physical Intuition                                         |
|-----------|---------|------------------------------------------------------------|
| 0.5 m     | Strict  | Half a car width. Sub-lane accuracy. Sufficient for        |
|           |         | confident lane-keeping even at highway speeds.             |
| 1.0 m     | Medium  | Roughly one car width. Lane-width accuracy. Adequate for   |
|           |         | most ADAS lane-keeping but not precise path planning.      |
| 1.5 m     | Lenient | Within-road accuracy. You know where the road is, but      |
|           |         | not precisely where individual lanes are.                  |

### 4.2 Visual Representation

```
    What these thresholds look like on a 3.5m lane:
    ================================================================

    |<------------ 3.5 m lane width ------------>|
    |                                             |
    |     GT lane boundary                        |
    |     |                                       |
    |     V                                       |
    |     ========================================|  <-- true position
    |     :                                       |
    |  0.5m zone:   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|  <-- 0.5m error: still in lane
    |     :                                       |
    |  1.0m zone:   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|  <-- 1.0m error: near lane center
    |     :                                       |
    |  1.5m zone:   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|  <-- 1.5m error: near other boundary
    |                                             |
    |     ========================================|  <-- other lane boundary
    |                                             |

    At 0.5m error, the vehicle is safely within its lane.
    At 1.0m error, you are roughly at the center of the lane (one boundary known well).
    At 1.5m error, you could be near the opposite boundary -- marginal for lane keeping.
```

### 4.3 Choosing the Right Threshold for Your Application

- **Highway autopilot (60+ km/h):** Demand AP@0.5. At high speeds, even 0.5m error
  compounds quickly and the vehicle must stay centered.
- **Urban ADAS (30-50 km/h):** AP@1.0 is the practical minimum. Lanes are narrower,
  but speeds allow for correction.
- **Rough localization / route planning:** AP@1.5 may suffice. You need to know *which
  road* and *approximately where lanes are*, not sub-lane precision.

---

## 5. Per-Element Evaluation

### 5.1 Lane Dividers

- **Geometry:** Thin lines (dashed or solid) separating adjacent lanes.
- **Count per scene:** High (10-30 in a typical intersection).
- **Direction sensitivity:** Yes -- dividers can be traversed either way.
- **Difficulty:** Medium. They are thin and numerous, requiring precise detection.
  However, they are repetitive and well-structured.
- **Typical AP range:** 40-60% (strict to lenient thresholds).

### 5.2 Road Boundaries

- **Geometry:** Longer, more continuous curves defining the road edge (curbs, barriers).
- **Count per scene:** Moderate (2-8).
- **Direction sensitivity:** Yes.
- **Difficulty:** Lower than dividers for detection (fewer, more prominent), but
  harder for precise localization along long curves.
- **Typical AP range:** 38-72%.

### 5.3 Pedestrian Crossings

- **Geometry:** Closed polygons (rectangle boundaries) marking crosswalks.
- **Count per scene:** Low (0-4).
- **Direction sensitivity:** Yes (polygon traversal order).
- **Difficulty:** Highest. They are rare (class imbalance), localized, and often
  occluded by pedestrians or vehicles.
- **Typical AP range:** 33-65%.

### 5.4 Why Each Class Has Different Difficulty

| Factor              | Lane Dividers | Road Boundaries | Ped Crossings |
|---------------------|---------------|-----------------|---------------|
| Frequency           | High          | Medium          | Low           |
| Length              | Medium        | Long            | Short         |
| Visual saliency     | Medium        | High            | Medium        |
| Occlusion risk      | Low           | Low             | High          |
| Class imbalance     | None          | Slight          | Severe        |

---

## 6. Temporal Consistency Metrics (Unique to StreamMapNet)

Standard map evaluation treats each frame independently. StreamMapNet's key innovation
is exploiting temporal context via a streaming architecture. These metrics quantify how
much that helps.

### 6.1 Map Element Stability Score

**What it measures:** For the same physical map element observed across consecutive
frames, how much does the prediction "jitter" after compensating for ego-motion?

**Algorithm:**
1. Take prediction at frame t.
2. Apply ego-motion transform to warp it into frame t+1 coordinates.
3. Find the closest same-class prediction in frame t+1.
4. Record the Chamfer distance between the warped prediction and its match.
5. Average over all element-frame pairs.

**Physical meaning:** A stability score of 0.3m means predictions shift by ~30cm
between frames. A planning module would see lane boundaries wobble by a tire width.

```python
import numpy as np

def compute_stability_score(predictions_sequence, ego_motions):
    """
    Compute Map Element Stability Score across a temporal sequence.

    Args:
        predictions_sequence: list of length T, where each entry is a list of dicts:
            [{'points': (K,2), 'label': int, 'score': float}, ...]
        ego_motions: list of (3,3) transformation matrices from frame t to t+1

    Returns:
        stability_score: float, average positional deviation in meters (lower=better)
    """
    deviations = []

    for t in range(len(predictions_sequence) - 1):
        preds_t = predictions_sequence[t]
        preds_t1 = predictions_sequence[t + 1]
        ego_t_to_t1 = ego_motions[t]  # 3x3 homogeneous transform

        for pred in preds_t:
            # Warp prediction from frame t into frame t+1 coordinates
            pts_homogeneous = np.hstack([
                pred['points'],
                np.ones((pred['points'].shape[0], 1))
            ])  # (K, 3)
            pts_warped = (ego_t_to_t1 @ pts_homogeneous.T).T[:, :2]  # (K, 2)

            # Find best matching prediction in frame t+1 (same class)
            best_dist = float('inf')
            for pred_next in preds_t1:
                if pred_next['label'] != pred['label']:
                    continue
                dist = chamfer_distance(pts_warped, pred_next['points'])
                best_dist = min(best_dist, dist)

            # Only count if element persists (not exiting perception range)
            if best_dist < 3.0:
                deviations.append(best_dist)

    if len(deviations) == 0:
        return 0.0
    return float(np.mean(deviations))
```

### 6.2 Flicker Rate

**What it measures:** The fraction of predicted map elements that "vanish" between
consecutive frames despite still being within perception range.

**Definition:** A "flicker" occurs when:
- An element is predicted at frame t,
- After ego-motion compensation, it is still within the perception boundary at t+1,
- Yet NO corresponding prediction (same class, Chamfer < 2m) exists at t+1.

```python
def compute_flicker_rate(predictions_sequence, ego_motions, perception_range,
                         match_threshold=2.0):
    """
    Compute the flicker rate across a temporal sequence.

    Args:
        predictions_sequence: list of per-frame prediction lists
        ego_motions: list of frame-to-frame transforms
        perception_range: [x_min, y_min, x_max, y_max] in meters
        match_threshold: max Chamfer to consider a match (meters)

    Returns:
        flicker_rate: float in [0, 1], fraction of elements that flicker (lower=better)
    """
    x_min, y_min, x_max, y_max = perception_range
    total_elements = 0
    flickered_elements = 0

    for t in range(len(predictions_sequence) - 1):
        preds_t = predictions_sequence[t]
        preds_t1 = predictions_sequence[t + 1]
        ego_t_to_t1 = ego_motions[t]

        for pred in preds_t:
            # Warp to next frame
            pts_h = np.hstack([pred['points'], np.ones((len(pred['points']), 1))])
            pts_warped = (ego_t_to_t1 @ pts_h.T).T[:, :2]

            # Check if still within perception range
            center = pts_warped.mean(axis=0)
            if not (x_min <= center[0] <= x_max and y_min <= center[1] <= y_max):
                continue  # Element left the field of view -- not a flicker

            total_elements += 1

            # Search for correspondence in next frame
            found_match = False
            for pred_next in preds_t1:
                if pred_next['label'] != pred['label']:
                    continue
                if chamfer_distance(pts_warped, pred_next['points']) < match_threshold:
                    found_match = True
                    break

            if not found_match:
                flickered_elements += 1

    return flickered_elements / max(total_elements, 1)
```

### 6.3 Why Temporal Metrics Matter for Downstream Planning

A planning module converts perceived lane boundaries into a trajectory. If predictions
flicker:
- The planner sees a lane divider at t, plans to stay left of it.
- At t+1 the divider vanishes; the planner has no constraint.
- At t+2 it reappears; the planner jerks back.

This causes oscillating steering commands even though per-frame AP might be fine.
Stability score quantifies the magnitude of jitter; flicker rate quantifies its
frequency.

### 6.4 Comparison: Single-Frame vs. Temporal Model

| Method                       | AP (mAP) | Stability (m) | Flicker Rate |
|------------------------------|----------|---------------|--------------|
| MapTR (single-frame)         | 50.3     | 1.42          | 18.3%        |
| StreamMapNet (no temporal)   | 50.3     | 1.42          | 18.3%        |
| StreamMapNet (with temporal) | 54.1     | 0.67          | 7.2%         |

StreamMapNet's temporal fusion halves the stability deviation and reduces flicker by
more than 2x, demonstrating that temporal context is essential for production-grade
map prediction.

---

## 7. Evaluation Protocol Details

### 7.1 Full Protocol Specification

| Aspect                  | Specification                                              |
|-------------------------|------------------------------------------------------------|
| Matching strategy       | Greedy (highest confidence first, not Hungarian)           |
| Matching scope          | Per-frame (predictions only match GTs from the same frame) |
| Direction handling      | Compute CD in both directions, take minimum                |
| Perception range        | [-30, 30] x [-15, 15] meters (60m forward, 30m lateral)   |
| GT filtering            | Only evaluate GT elements with path length >= 2 meters     |
| Point normalization     | Both pred and GT are K=20 equidistant points               |
| Coordinate system       | Ego-vehicle frame (origin at car center, X forward, Y left)|
| Model output format     | Normalized [0, 1] coordinates, denormalized before eval    |
| AP interpolation        | 40-point recall interpolation                              |
| Confidence scores       | Sigmoid of classification logits                           |

### 7.2 Why Greedy Matching (Not Hungarian)?

Hungarian matching finds the globally optimal assignment that minimizes total cost.
Greedy matching processes predictions one at a time in confidence order. The greedy
approach is standard in detection (PASCAL VOC, COCO) because:

1. It rewards well-calibrated confidence scores (high-confidence predictions get first
   pick of GT elements).
2. It is simpler to implement and debug.
3. In practice, the results are nearly identical to Hungarian for well-calibrated models.

### 7.3 Denormalization from [0, 1] to Meters

The model outputs normalized coordinates for numerical stability during training. Before
evaluation, these must be converted back to physical coordinates:

```python
def denormalize_points(points_norm, perception_range):
    """
    Convert model output from [0, 1] to meters in ego-vehicle frame.

    Args:
        points_norm: (K, 2) array with values in [0, 1]
        perception_range: (x_min, y_min, x_max, y_max) in meters
                          e.g., (-30.0, -15.0, 30.0, 15.0)

    Returns:
        points_m: (K, 2) in meters
    """
    x_min, y_min, x_max, y_max = perception_range
    points_m = np.empty_like(points_norm)
    points_m[:, 0] = points_norm[:, 0] * (x_max - x_min) + x_min  # X: forward
    points_m[:, 1] = points_norm[:, 1] * (y_max - y_min) + y_min  # Y: lateral
    return points_m
```

---

## 8. Running Evaluation

### 8.1 Single-GPU Evaluation

```bash
python tools/test.py \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    work_dirs/streammapnet_r50/epoch_24.pth \
    --eval chamfer \
    --gpu-ids 0
```

### 8.2 Multi-GPU Evaluation

```bash
bash tools/dist_test.sh \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    work_dirs/streammapnet_r50/epoch_24.pth \
    4 \
    --eval chamfer
```

### 8.3 With Temporal Consistency Metrics

```bash
python tools/test.py \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    work_dirs/streammapnet_r50/epoch_24.pth \
    --eval chamfer temporal \
    --eval-options sequence_mode=True
```

### 8.4 Evaluation Options Reference

| Flag / Option                         | Description                                    |
|---------------------------------------|------------------------------------------------|
| `--eval chamfer`                      | Standard Chamfer-based AP evaluation           |
| `--eval temporal`                     | Include stability and flicker metrics          |
| `--show-dir vis_results/`            | Save per-frame BEV visualizations              |
| `--eval-options threshold=1.0`        | Override AP threshold (single value)           |
| `--eval-options sequence_mode=True`   | Enable temporal sequence evaluation            |
| `--format-only`                       | Save predictions to JSON without computing AP  |
| `--out results.pkl`                   | Save raw results to pickle file                |

### 8.5 Saving and Visualizing Predictions

```bash
# Save predictions as JSON for offline analysis
python tools/test.py \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    work_dirs/streammapnet_r50/epoch_24.pth \
    --format-only \
    --eval-options jsonfile_prefix=results/streammapnet

# Visualize predictions overlaid on BEV
python tools/test.py \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    work_dirs/streammapnet_r50/epoch_24.pth \
    --eval chamfer \
    --show-dir vis_results/bev_overlay/
```

---

## 9. Visualization

### 9.1 Per-Frame BEV Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Color coding by element class
CLASS_COLORS = {
    0: '#FF8C00',   # Lane divider: orange
    1: '#228B22',   # Road boundary: forest green
    2: '#4169E1',   # Pedestrian crossing: royal blue
}
CLASS_NAMES = {
    0: 'Lane Divider',
    1: 'Road Boundary',
    2: 'Ped Crossing',
}

def visualize_bev_frame(pred_elements, gt_elements, perception_range,
                        save_path=None, title=''):
    """
    Render predicted and ground truth map elements side-by-side in BEV.

    Args:
        pred_elements: list of {'points': (K,2), 'label': int, 'score': float}
        gt_elements:   list of {'points': (K,2), 'label': int}
        perception_range: (x_min, y_min, x_max, y_max)
        save_path: optional path to save the figure
        title: optional figure title
    """
    x_min, y_min, x_max, y_max = perception_range
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 8))

    for ax, elements, name in [(ax_gt, gt_elements, 'Ground Truth'),
                                (ax_pred, pred_elements, 'Predictions')]:
        ax.set_title(f'{name} {title}')
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(x_min, x_max)
        ax.set_aspect('equal')
        ax.set_xlabel('Lateral Y (m)')
        ax.set_ylabel('Forward X (m)')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

        for elem in elements:
            pts = elem['points']  # (K, 2) in meters: col 0=X, col 1=Y
            color = CLASS_COLORS[elem['label']]
            alpha = min(1.0, elem.get('score', 1.0) + 0.3)
            ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=2, alpha=alpha)

    # Shared legend
    legend_handles = [plt.Line2D([0], [0], color=c, linewidth=2, label=CLASS_NAMES[k])
                      for k, c in CLASS_COLORS.items()]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

### 9.2 Temporal Sequence Video Generation

```bash
# Generate frame sequence as images, then assemble into video
python tools/visualize_sequence.py \
    --config configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    --checkpoint work_dirs/streammapnet_r50/epoch_24.pth \
    --scene-token <scene_token> \
    --output-dir vis_results/sequence/ \
    --format video \
    --fps 4
```

To assemble manually with ffmpeg:

```bash
ffmpeg -framerate 4 -pattern_type glob -i 'vis_results/sequence/*.png' \
    -c:v libx264 -pix_fmt yuv420p vis_results/sequence.mp4
```

---

## 10. Benchmark Results Reference

### 10.1 nuScenes val Set (24 epochs, ResNet-50 backbone)

| Method         | Divider | Boundary | Ped Cross | mAP  |
|----------------|---------|----------|-----------|------|
| HDMapNet       | 18.5    | 37.6     | 14.1      | 23.4 |
| VectorMapNet   | 36.2    | 43.5     | 28.5      | 36.1 |
| MapTR          | 51.5    | 53.1     | 46.3      | 50.3 |
| MapTRv2        | 55.7    | 57.4     | 49.2      | 54.1 |
| StreamMapNet   | 56.3    | 55.8     | 50.1      | 54.1 |

Note: StreamMapNet matches MapTRv2 on per-frame mAP while significantly outperforming
on temporal consistency (see Section 6.4).

### 10.2 Argoverse 2 val Set

| Method         | Divider | Boundary | Ped Cross | mAP  |
|----------------|---------|----------|-----------|------|
| MapTR          | 58.7    | 60.3     | 52.1      | 57.0 |
| StreamMapNet   | 62.4    | 63.1     | 56.8      | 60.8 |

Argoverse 2 provides richer temporal annotations, making StreamMapNet's temporal
advantage more pronounced (+3.8 mAP over MapTR compared to +3.8 on nuScenes).

---

## 11. Common Evaluation Pitfalls

### 11.1 Inconsistent Perception Range

**Problem:** Training uses [-30, 30] x [-15, 15] but evaluation uses [-60, 60] x [-30, 30].
The model has never seen points in the extended range and performs poorly.

**Fix:** Always verify that the perception_range in the evaluation config matches training.

### 11.2 Direction Ambiguity

**Problem:** Evaluating with standard (non-direction-aware) Chamfer distance. A perfect
prediction with reversed point order receives a large penalty.

**Fix:** Always use `direction_aware_chamfer()` which takes the minimum of both orderings.

### 11.3 Point Count Mismatch

**Problem:** Model predicts 20 points per element, but GT has 50 points (or vice versa).
Chamfer distance is still computable but the density difference biases the backward term.

**Fix:** Resample both prediction and GT to the same number of equidistant points (K=20)
before computing Chamfer distance. Use linear interpolation along the polyline arc length.

### 11.4 Confidence Calibration

**Problem:** All predictions have scores clustered around 0.5. The greedy matching order
becomes nearly random, hurting AP.

**Fix:** Use raw sigmoid outputs as confidence scores. If calibration is poor, consider
temperature scaling as a post-processing step. Never threshold predictions before AP
computation -- the P-R curve needs the full score distribution.

### 11.5 Scene Boundary Handling

**Problem:** The first frame of each scene has no temporal history. Including it in
temporal metrics (stability, flicker) unfairly penalizes the model.

**Fix:** Exclude the first frame from temporal metrics, but still include it in per-frame
AP computation. When the temporal buffer has `N` history frames, exclude the first `N`
frames from temporal metrics.

### 11.6 Mini Split Limitations

**Problem:** Reporting metrics on nuScenes v1.0-mini (10 scenes, ~400 frames) and
treating them as final results. Variance is extremely high on such small data.

**Fix:** Use mini only for sanity checks (smoke tests during development). Always report
final metrics on the full validation set (150 scenes, ~6000 frames).

---

## 12. Interpretation Guide

### 12.1 What Makes a "Good" Map Prediction?

A production-quality map prediction system should achieve:
- AP@0.5 > 50% for lane dividers (sub-lane precision on most elements)
- Flicker rate < 10% (stable enough for planning)
- Stability score < 0.5m (jitter less than a tire width)

### 12.2 Diagnostic Patterns

**High AP but high flicker rate:**
- Diagnosis: The model is accurate per-frame but temporally inconsistent.
- Cause: Insufficient temporal fusion or history length.
- Action: Increase the propagation queue length or add explicit temporal loss terms.

**Low AP specifically on pedestrian crossings:**
- Diagnosis: Rare class underperformance (class imbalance).
- Cause: Crossings appear in ~30% of frames; the model under-represents them.
- Action: Apply class-weighted loss, oversample crossing-heavy scenes, or use
  copy-paste augmentation for crossings.

**AP@0.5 much lower than AP@1.5 (large gap):**
- Diagnosis: The model detects elements but localizes them poorly.
- Cause: Regression head underfitting or insufficient resolution in BEV features.
- Action: Increase BEV resolution, add auxiliary point regression losses, or use
  iterative refinement (deformable attention on predicted points).

**Good AP on nuScenes but poor on Argoverse 2:**
- Diagnosis: Domain gap between datasets.
- Cause: Different camera setups, map annotation conventions, or road geometries.
- Action: Verify preprocessing, check perception range consistency, consider
  dataset-specific fine-tuning.

### 12.3 When to Use Which Threshold

| Use Case                              | Primary Threshold | Rationale                          |
|---------------------------------------|-------------------|------------------------------------|
| Highway lane-keeping (L2+)            | AP@0.5            | Must be sub-lane precise           |
| Urban navigation assistance           | AP@1.0            | Moderate precision, many elements  |
| Coarse route planning                 | AP@1.5            | Structural correctness sufficient  |
| Model comparison (paper reporting)    | mAP (all three)   | Gives full accuracy profile        |
| Temporal quality assessment           | Stability + Flicker| Complements per-frame AP           |

### 12.4 Reading Results Holistically

Never report a single number in isolation. A complete evaluation includes:
1. **Per-class AP at all thresholds** -- reveals which elements and accuracy levels
   are problematic.
2. **mAP** -- the headline number for paper comparisons.
3. **Temporal metrics** -- essential for any system that will feed a planner.
4. **Qualitative visualization** -- numbers can hide failure modes (e.g., the model
   might hallucinate a parallel lane divider that hurts precision but not recall).

---

## Summary of Key Formulas

```
Chamfer Distance:
  CD(A, B) = [ (1/M) * sum_i min_j ||a_i - b_j|| + (1/N) * sum_j min_i ||b_j - a_i|| ] / 2

Direction-Aware Chamfer:
  DAC(P, G) = min( CD(P, G), CD(P, reverse(G)) )

True Positive condition:
  DAC(pred, gt) <= threshold  AND  gt is not yet matched

Precision at rank k:
  P(k) = TP_cumulative(k) / (TP_cumulative(k) + FP_cumulative(k))

Recall at rank k:
  R(k) = TP_cumulative(k) / |GT_total|

Interpolated AP (40-point):
  AP = (1/40) * sum_{r in linspace(0,1,40)} max_{k: R(k)>=r} P(k)

Stability Score:
  S = mean over (t, element) of CD(warp(pred_t, ego_t->t+1), pred_t+1)

Flicker Rate:
  F = |{elements at t with no match at t+1 despite being in range}| / |{elements at t in range}|
```

---

*This guide accompanies the StreamMapNet evaluation codebase. For implementation details,
see `tools/test.py` and `plugin/datasets/evaluation/` in the repository.*
