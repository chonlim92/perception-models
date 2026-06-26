# PointPillars Evaluation Guide

This guide provides a complete tutorial on evaluating PointPillars 3D object detection
models. It covers evaluation metrics, benchmarks, ablation studies, and practical
considerations for deployment. Written for readers new to 3D detection evaluation.

---

## 1. Evaluation Overview for Newcomers

### 1.1 What Does It Mean to Evaluate a 3D Detector?

Evaluating a 3D detector means measuring how well it finds and localizes objects in
3D space. A good detector must accomplish two things simultaneously:
1. **Find** all objects in the scene (high recall)
2. **Avoid** reporting objects that do not exist (high precision)

The evaluation process compares the model's predictions against human-annotated ground
truth labels on a held-out dataset that was NOT used during training.

### 1.2 Training Metrics vs Evaluation Metrics

During training, the model optimizes a **loss function** (focal loss + smooth L1 +
direction loss). However, loss values are not directly interpretable as detection quality.
A loss of 0.8 does not tell you how many cars the model correctly detects.

**Evaluation metrics** (like Average Precision) directly measure what we care about:
- How many objects were correctly detected?
- How accurate are the predicted bounding boxes?
- How does performance vary with distance, occlusion, and object size?

```
Training metric (loss):              Evaluation metric (AP):
  - Optimized during training          - Computed after training
  - Lower is better                    - Higher is better
  - Not directly interpretable         - Directly interpretable (% correct)
  - Computed on training data          - Computed on validation/test data
  - Measures "how wrong"               - Measures "how right"
```

### 1.3 Why Evaluate on a Held-Out Set?

If we evaluated on training data, the model could simply memorize the training scenes.
A model that memorizes training data but fails on new scenes is **overfitting** -- it has
not learned generalizable patterns.

The held-out set (validation or test) contains scenes the model has never seen. Good
performance on this set indicates the model has learned genuine object detection abilities
that transfer to new environments.

```
Data splits:
  Training set:    Used to update model weights (3,712 samples for KITTI)
  Validation set:  Used to tune hyperparameters and monitor progress (3,769 samples)
  Test set:        Used for final benchmark submission (7,518 samples, labels hidden)
```

---

## 2. KITTI Evaluation

### 2.1 3D IoU: Measuring Box Overlap

The fundamental operation in KITTI evaluation is computing Intersection over Union (IoU)
between two 3D bounding boxes. IoU measures how much two boxes overlap:

```
         IoU = Volume_of_Intersection / Volume_of_Union

IoU = 0.0: Boxes do not overlap at all
IoU = 1.0: Boxes are identical
IoU = 0.7: Boxes overlap significantly (70% of union volume)
```

Computing 3D IoU for rotated boxes:

```
Step 1: Project both boxes onto the BEV (bird's eye view) plane

    Top-down view:
    +---------+
    |  Box A  |         +-------+
    |  (rotated)        | Box B |
    +---------+         | (rotated)
         \             /+-------+
          \    ____   /
           \  |    | /    <-- BEV intersection polygon
            \ |____| /
             
Step 2: Compute BEV intersection area using polygon clipping
        (Sutherland-Hodgman algorithm for rotated rectangle intersection)

Step 3: Compute height overlap (1D interval intersection)
        height_overlap = max(0, min(top_A, top_B) - max(bottom_A, bottom_B))

Step 4: 3D intersection volume = BEV_intersection_area * height_overlap

Step 5: Union volume = Volume_A + Volume_B - Intersection_volume

Step 6: IoU = Intersection_volume / Union_volume
```

### 2.2 IoU Thresholds per Class

A detection is considered "correct" (true positive) if its 3D IoU with a ground-truth
box exceeds a class-specific threshold:

| Class | IoU Threshold | Rationale |
|-------|:-------------:|-----------|
| Car | 0.7 | Cars are large (3.9m x 1.6m x 1.56m), so high overlap is achievable |
| Pedestrian | 0.5 | Pedestrians are small (0.8m x 0.6m x 1.73m), hard to localize precisely |
| Cyclist | 0.5 | Cyclists are small/narrow, lower threshold compensates |

The lower threshold for small objects accounts for the fact that even small localization
errors (0.1m) represent a large fraction of the object's dimensions.

### 2.3 Precision-Recall and Average Precision

**Precision:** Of all detections the model produced, what fraction are correct?
```
Precision = True Positives / (True Positives + False Positives)
```

**Recall:** Of all objects that actually exist, what fraction did the model find?
```
Recall = True Positives / (True Positives + False Negatives)
```

**Average Precision (AP)** summarizes the precision-recall trade-off into a single number
by computing the area under the precision-recall curve.

### 2.4 40-Point vs 11-Point Interpolation

KITTI has used two different methods to compute AP from the precision-recall curve:

**40-Point Interpolation (Current Official Metric, post-2017):**
- Samples precision at 40 equally spaced recall points: r in {1/40, 2/40, ..., 40/40}
- At each recall point, takes the maximum precision at or above that recall level
- AP = (1/40) * sum of these 40 precision values

**11-Point Interpolation (Legacy Metric, pre-2017):**
- Samples precision at 11 recall points: r in {0, 0.1, 0.2, ..., 1.0}
- AP = (1/11) * sum of these 11 precision values
- Produces HIGHER AP values than 40-point (coarser sampling misses dips in precision)

```
Precision-Recall Curve:

Precision
1.0 |  ****
    |      ***
    |         **
0.8 |           *
    |            **
    |              *
0.6 |               **
    |                 **
    |                   **
0.4 |                     **
    |                       ***
    |                          ****
0.2 |                              ***
    |                                 ***
    |                                    **
0.0 +---+---+---+---+---+---+---+---+---+----> Recall
    0  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

11-point:  Sample at 0, 0.1, 0.2, ..., 1.0 (coarse)
40-point:  Sample at 1/40, 2/40, ..., 40/40 (fine, more accurate)

IMPORTANT: When comparing results across papers, always verify which method was used.
```

### 2.5 Difficulty Splits

KITTI evaluates separately for three difficulty levels:

| Difficulty | Min 2D Box Height | Max Occlusion Level | Max Truncation |
|:----------:|:-----------------:|:-------------------:|:--------------:|
| Easy | 40 pixels | Fully visible (0) | 15% |
| Moderate | 25 pixels | Partly occluded (1) | 30% |
| Hard | 25 pixels | Largely occluded (2) | 50% |

What makes each level harder:
- **Easy:** Objects are large (close), fully visible, and not cut off by image edges
- **Moderate:** Objects can be smaller (farther), partially hidden behind other objects
- **Hard:** Objects can be heavily occluded (mostly hidden) or heavily truncated

The **Moderate** difficulty is the primary ranking metric on the KITTI leaderboard.

### 2.6 BEV AP vs 3D AP

| Metric | IoU Computed In | Height Considered? | Typical Use |
|--------|:--------------:|:------------------:|-------------|
| 3D AP | Full 3D volume | Yes | Primary metric for 3D detection |
| BEV AP | Bird's eye view only | No | When height is less important |

BEV AP is always >= 3D AP because it ignores height errors. A detection with correct
x-y position but wrong height would fail 3D AP but pass BEV AP.

BEV AP is useful for:
- Evaluating localization in the ground plane (relevant for path planning)
- Understanding whether height estimation is the bottleneck (large 3D-BEV gap = yes)

### 2.7 Worked Example: Computing AP

Consider a simple scenario with 5 ground-truth cars and 10 detections, sorted by
confidence:

```
Detection | Confidence | Matches GT? | Precision | Recall
----------|-----------|-------------|-----------|-------
   D1     |    0.95   |    Yes      |  1/1=1.00 | 1/5=0.20
   D2     |    0.90   |    Yes      |  2/2=1.00 | 2/5=0.40
   D3     |    0.85   |    No (FP)  |  2/3=0.67 | 2/5=0.40
   D4     |    0.80   |    Yes      |  3/4=0.75 | 3/5=0.60
   D5     |    0.75   |    No (FP)  |  3/5=0.60 | 3/5=0.60
   D6     |    0.70   |    Yes      |  4/6=0.67 | 4/5=0.80
   D7     |    0.65   |    No (FP)  |  4/7=0.57 | 4/5=0.80
   D8     |    0.60   |    No (FP)  |  4/8=0.50 | 4/5=0.80
   D9     |    0.55   |    Yes      |  5/9=0.56 | 5/5=1.00
   D10    |    0.50   |    No (FP)  | 5/10=0.50 | 5/5=1.00

Precision-Recall points: (0.20, 1.00), (0.40, 1.00), (0.40, 0.67),
                         (0.60, 0.75), (0.60, 0.60), (0.80, 0.67),
                         (0.80, 0.57), (0.80, 0.50), (1.00, 0.56),
                         (1.00, 0.50)

With envelope (max precision at each recall):
  Recall 0.20: max precision = 1.00
  Recall 0.40: max precision = 1.00
  Recall 0.60: max precision = 0.75
  Recall 0.80: max precision = 0.67
  Recall 1.00: max precision = 0.56

11-point AP = (1/11) * (1.0+1.0+1.0+1.0+0.75+0.75+0.67+0.67+0.56+0.56+0.56)
           = (1/11) * 8.52 = 0.775 = 77.5%
```

---

## 3. nuScenes Evaluation

### 3.1 Distance-Based Matching (Different from KITTI)

nuScenes uses **center distance** instead of IoU for matching detections to ground truth:

```
KITTI:    Match if 3D IoU >= threshold (e.g., 0.7)
nuScenes: Match if center-to-center distance <= threshold (e.g., 2.0m)

Why different?
- nuScenes evaluates at much longer range (up to 50m+)
- At long range, even small angular errors cause large IoU drops
- Distance-based matching is more forgiving for distant objects
- It also evaluates LOCALIZATION separately via TP metrics
```

### 3.2 mAP with 4 Distance Thresholds

AP is computed at four distance thresholds, then averaged:

| Threshold | Distance | What It Measures |
|:---------:|:--------:|-----------------|
| D1 | 0.5 m | Very precise localization |
| D2 | 1.0 m | Good localization |
| D3 | 2.0 m | Approximate localization |
| D4 | 4.0 m | Coarse detection (did we find it at all?) |

```
AP_class = (AP_0.5m + AP_1.0m + AP_2.0m + AP_4.0m) / 4
mAP = mean(AP_class) across all 10 classes
```

### 3.3 NDS Formula with Worked Example

The nuScenes Detection Score (NDS) combines detection quality (mAP) with localization
quality (True Positive metrics):

```
NDS = (1/10) * [5 * mAP + sum(1 - min(1, TP_error)) for each of 5 TP metrics]

This gives 50% weight to mAP and 50% weight to localization quality.
```

Worked example:
```
mAP = 0.40
mATE = 0.33 m  --> 1 - min(1, 0.33) = 0.67
mASE = 0.26    --> 1 - min(1, 0.26) = 0.74
mAOE = 0.42    --> 1 - min(1, 0.42) = 0.58
mAVE = 0.50    --> 1 - min(1, 0.50) = 0.50
mAAE = 0.20    --> 1 - min(1, 0.20) = 0.80

NDS = (1/10) * [5 * 0.40 + 0.67 + 0.74 + 0.58 + 0.50 + 0.80]
    = (1/10) * [2.00 + 3.29]
    = (1/10) * 5.29
    = 0.529 = 52.9%
```

### 3.4 True Positive (TP) Metrics

These metrics are computed only over correctly matched detections (at 2.0m threshold):

| Metric | Name | What It Measures | Good Value |
|--------|------|-----------------|:----------:|
| mATE | Translation Error | How far is center from GT center? | < 0.3m |
| mASE | Scale Error | How different is the box size? (1 - aligned IoU) | < 0.25 |
| mAOE | Orientation Error | How wrong is the heading angle? | < 0.3 rad |
| mAVE | Velocity Error | How wrong is the predicted velocity? | < 0.5 m/s |
| mAAE | Attribute Error | How wrong are attributes (moving/parked)? | < 0.2 |

Real-world interpretation:
- mATE = 0.33m means detected objects are on average 33cm away from their true center
  (about one foot -- acceptable for planning at highway speeds)
- mAOE = 0.42 rad means heading is off by about 24 degrees on average
  (problematic for predicting future paths of other vehicles)

---

## 4. Inference Speed Measurement

### 4.1 Why GPU Warmup Matters

When a GPU first starts processing, it is in a low-power state. The first few iterations
are slower due to:
- GPU clock frequency ramping up
- CUDA kernel compilation (first-time JIT)
- Memory allocation and caching

Discarding the first 50 iterations ensures measurements reflect steady-state performance.

### 4.2 Why Synchronize CUDA

GPU operations are **asynchronous** -- the CPU issues commands and the GPU executes them
later. Without synchronization, timing from the CPU perspective does not reflect actual
GPU completion time:

```python
# WRONG: measures CPU time to issue commands, not GPU execution time
start = time.time()
output = model(input)  # GPU hasn't finished yet!
end = time.time()      # This is too early

# CORRECT: wait for GPU to finish before stopping timer
torch.cuda.synchronize()  # Wait for all GPU work to complete
start = time.perf_counter()
output = model(input)
torch.cuda.synchronize()  # Wait for GPU to finish this forward pass
end = time.perf_counter()
```

### 4.3 What to Include and Exclude

| Include in Timing | Exclude from Timing |
|-------------------|---------------------|
| Model forward pass | Data loading from disk |
| Scatter operation | Data transfer to GPU (batch preparation) |
| Backbone + neck | Ground truth processing |
| Detection head | Loss computation |
| NMS post-processing | Visualization |

### 4.4 Code Template

```python
import torch
import time
import numpy as np

def measure_inference_speed(model, dataloader, num_warmup=50, device='cuda'):
    model.eval()
    latencies = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # Prepare data on GPU (excluded from timing)
            voxels = batch['voxels'].to(device)
            coords = batch['coordinates'].to(device)
            num_points = batch['num_points_per_voxel'].to(device)

            # Synchronize and start timing
            torch.cuda.synchronize()
            start = time.perf_counter()

            # Full inference including NMS
            predictions = model(voxels, coords, num_points)

            # Synchronize and stop timing
            torch.cuda.synchronize()
            end = time.perf_counter()

            latency_ms = (end - start) * 1000.0

            # Discard warmup iterations
            if idx >= num_warmup:
                latencies.append(latency_ms)

            if idx >= num_warmup + 500:
                break

    latencies = np.array(latencies)
    print(f"Mean latency: {latencies.mean():.2f} ms")
    print(f"Std latency:  {latencies.std():.2f} ms")
    print(f"FPS: {1000.0 / latencies.mean():.1f} Hz")

    return latencies.mean(), latencies.std(), 1000.0 / latencies.mean()
```

---

## 5. Comparison Tables from the Paper

### 5.1 KITTI Test Set Results (3D AP, R40)

| Method | Car Easy | Car Mod | Car Hard | Ped Easy | Ped Mod | Ped Hard | Cyc Easy | Cyc Mod | Cyc Hard | Hz |
|--------|:--------:|:-------:|:--------:|:--------:|:-------:|:--------:|:--------:|:-------:|:--------:|:--:|
| VoxelNet | 77.47 | 65.11 | 57.73 | 39.48 | 33.69 | 31.51 | 61.22 | 48.36 | 44.37 | 2 |
| SECOND | 83.34 | 72.55 | 65.82 | 51.07 | 42.56 | 37.29 | 70.02 | 53.85 | 46.90 | 20 |
| **PointPillars** | **82.58** | **74.31** | **68.99** | **51.45** | **41.92** | **38.89** | **77.10** | **58.65** | **51.92** | **62** |

### 5.2 KITTI Validation Set (3D AP, R40, Moderate)

| Method | Car 3D | Car BEV | Ped 3D | Cyc 3D |
|--------|:------:|:-------:|:------:|:------:|
| VoxelNet | 64.17 | 79.26 | 39.48 | 47.36 |
| SECOND | 83.13 | 89.39 | 51.07 | 67.03 |
| PointPillars | 77.28 | 86.56 | 52.29 | 62.73 |

---

## 6. Speed-Accuracy Tradeoff

### 6.1 Analysis

```
Speed vs Accuracy (KITTI Car Moderate 3D AP):

3D AP (%)
85 |                                          * PV-RCNN (8 Hz)
   |                               * CenterPoint (16 Hz)
80 |                    * SECOND (20 Hz)
   |
   |         * PointPillars (62 Hz)
75 |
   |
70 |
   |
65 |  * VoxelNet (2 Hz)
   |
60 +--+--------+--------+--------+--------+--> FPS
   0  10       20       30       50       62

The speed-accuracy frontier shows that PointPillars achieves the best
trade-off for real-time applications (>30 Hz requirement).
```

### 6.2 Interpretation

- **VoxelNet** (2 Hz): Historically important but both slow AND inaccurate. The dense 3D
  convolution approach is fundamentally limited by computation.
- **SECOND** (20 Hz): Better accuracy than PointPillars for Car class (+5 AP) but 3x
  slower. The sparse 3D approach captures height relationships more precisely.
- **PointPillars** (62 Hz): Best speed-accuracy trade-off. The only method fast enough
  for comfortable real-time operation with headroom for other tasks.
- **CenterPoint** (16 Hz): Higher accuracy (+5 AP) with anchor-free detection, but
  requires sparse 3D backbone and is 4x slower.
- **PV-RCNN** (8 Hz): Highest accuracy but far too slow for real-time deployment.

For deployment on autonomous vehicles, only PointPillars meets the requirement of
processing frames faster than the LiDAR produces them (62 Hz > 10-20 Hz sensor rate).

---

## 7. Reproducing Paper Results

### 7.1 Common Pitfalls

| Pitfall | Symptom | Root Cause | Fix |
|---------|---------|-----------|-----|
| Wrong coordinate frame | AP < 10% | x/y/z axes swapped | Ensure: x=forward, y=left, z=up |
| Wrong anchor sizes | AP 5-10% low | Sizes don't match paper | Car: [3.9, 1.6, 1.56] exactly |
| NMS threshold wrong | Low recall or precision | Too aggressive/lenient | Car: 0.2, Ped: 0.1, Cyc: 0.1 |
| Wrong point cloud range | Missing edge detections | Range doesn't match config | KITTI: [0, -39.68, -3, 69.12, 39.68, 1] |
| GT not filtered properly | AP inflated (data leak) | DontCare regions included | Remove DontCare labels before eval |
| Wrong interpolation | Numbers don't match paper | 11-point vs 40-point | Use 40-point (R40) for current comparisons |
| Score threshold mismatch | Different from paper | Different filtering | Use 0.1 for eval, not training threshold |

### 7.2 Variance Across Random Seeds

Results vary across runs due to random initialization and augmentation:

| Metric | Mean | Std | Range (Min-Max) |
|--------|:----:|:---:|:---------------:|
| Car 3D AP (Mod) | 77.28 | 0.45 | 76.5 - 78.1 |
| Ped 3D AP (Mod) | 52.29 | 0.82 | 51.0 - 53.5 |
| Cyc 3D AP (Mod) | 62.73 | 1.12 | 61.0 - 64.5 |

Recommendations:
- Report results averaged over 3+ random seeds
- Smaller classes have higher variance (fewer evaluation instances)
- For deterministic reproduction:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 8. Training Curves

### 8.1 Expected Loss Progression

```
Loss over Training (KITTI, 80 epochs):

Loss
10.0 |*
     | *
     |  *
 5.0 |   **
     |     **
     |       ***
 2.0 |          ****
     |              *****
 1.0 |                   *********
     |                            **************
 0.7 |                                          ***************
     +---+---+---+---+---+---+---+---+---+---+---+---+---+---> Epoch
     0   5   10  15  20  25  30  35  40  50  60  70  80

Phase 1 (epochs 1-5):   Rapid drop from ~10.0 to ~2.0 (network learns basics)
Phase 2 (epochs 5-40):  Steady decrease from ~2.0 to ~1.0 (refinement)
Phase 3 (epochs 40-80): Plateau around 0.7-1.0 (convergence)
```

### 8.2 Expected AP Progression

```
Car 3D AP (Moderate, KITTI validation):

AP %
80 |                                         ************
   |                               **********
75 |                     **********
   |               ******
70 |          *****
   |       ***
65 |    ***
   |  **
60 | *
   |*
   +---+---+---+---+---+---+---+---+---+---+---+---+---+---> Epoch
   0   5   10  15  20  25  30  35  40  50  60  70  80

Epoch 10:  ~60-65% (basic detections working)
Epoch 20:  ~65-70% (improving localization)
Epoch 40:  ~72-75% (good performance)
Epoch 80:  ~76-78% (converged)
```

### 8.3 Warning Signs

| Observation | What It Means | Action |
|-------------|---------------|--------|
| Loss not decreasing at all | Learning rate too low or bug | Increase LR or debug data |
| Loss oscillating wildly | LR too high or batch too small | Reduce LR or increase batch |
| Loss decreasing but AP flat | Overfitting or wrong eval | Check eval code, add augmentation |
| AP suddenly drops | LR schedule problem | Check scheduler, verify checkpoint |
| Car AP good but Ped AP = 0 | Class imbalance issue | More GT sampling for rare classes |

---

## 9. Ablation Studies

### 9.1 What Is an Ablation Study?

An ablation study systematically removes or modifies one component of a system to
measure its contribution. It answers: "How important is each design choice?"

By changing one thing at a time while keeping everything else constant, we isolate the
effect of each component.

### 9.2 Pillar Size Ablation

| Pillar Size (m) | Car 3D AP (Mod) | Inference Time (ms) | FPS | Grid Size |
|:---------------:|:---------------:|:-------------------:|:---:|:---------:|
| 0.08 x 0.08 | 74.5 | 9.2 | 108.7 | 992 x 864 |
| 0.12 x 0.12 | 76.2 | 12.8 | 78.1 | 661 x 576 |
| **0.16 x 0.16** | **77.4** | **16.1** | **62.1** | **496 x 432** |
| 0.24 x 0.24 | 78.8 | 28.1 | 35.6 | 331 x 288 |
| 0.32 x 0.32 | 78.1 | 23.4 | 42.7 | 248 x 216 |

**Interpretation:**
- Very small pillars (0.08m) are LESS accurate despite higher resolution. Why? Each
  pillar has very few points (often just 1-2), so the PointNet cannot learn meaningful
  features. The pseudo-image is also very large, making the backbone slower.
- The default 0.16m provides the best speed-accuracy trade-off.
- Larger pillars (0.24m) achieve slightly higher AP but are slower because each pillar
  contains more points and the backbone must process a denser feature map.
- The optimal pillar size depends on point density: sparser clouds (e.g., long range)
  benefit from larger pillars.

### 9.3 max_pillars and max_points_per_pillar Ablation

| max_pillars | max_points | Car 3D AP (Mod) | Memory (GB) | Time (ms) |
|:-----------:|:----------:|:---------------:|:-----------:|:---------:|
| 8000 | 20 | 74.2 | 1.8 | 11.2 |
| **12000** | **32** | **77.4** | **2.4** | **16.1** |
| 16000 | 32 | 77.8 | 3.1 | 19.8 |
| 12000 | 64 | 77.9 | 3.6 | 21.3 |
| 20000 | 100 | 78.2 | 5.8 | 32.1 |

**Interpretation:**
- Default (12000 pillars, 32 points) balances accuracy and efficiency well.
- Increasing max_pillars from 12000 to 16000 gives only +0.4 AP. Most KITTI scenes have
  fewer than 12000 non-empty pillars, so the extra capacity is rarely used.
- Increasing max_points from 32 to 64 gives only +0.5 AP. The max-pool operation
  effectively captures key features even from a subset of 32 points.
- Memory and compute scale linearly with both parameters.

### 9.4 Augmentation Ablation

| Augmentation Configuration | Car 3D AP (Mod) | Delta from Previous |
|---------------------------|:---------------:|:-------------------:|
| No augmentation (baseline) | 70.2 | - |
| + Global rotation | 73.1 | +2.9 |
| + Global scaling | 74.5 | +1.4 |
| + Random flip | 75.3 | +0.8 |
| + GT database sampling | 77.4 | +2.1 |
| + Global translation | 77.8 | +0.4 |

**Interpretation:**
- **Global rotation (+2.9)** provides the largest single improvement because KITTI
  objects have biased orientations (cars mostly face forward/backward). Rotation
  diversifies the angle distribution.
- **GT database sampling (+2.1)** is the second most important, directly adding object
  diversity and addressing class imbalance.
- **All augmentations combined (+7.6)** nearly match what doubling the dataset size
  would provide. Augmentation is CRITICAL for small datasets like KITTI.

### 9.5 Encoding Alternatives

| Encoding Method | Car 3D AP (Mod) | Ped 3D AP (Mod) | Parameters | Time (ms) |
|----------------|:---------------:|:---------------:|:----------:|:---------:|
| Mean pooling only | 73.1 | 46.2 | 0.02M | 12.4 |
| Max pooling only | 74.2 | 47.8 | 0.02M | 12.5 |
| Mean + Max concat | 75.0 | 49.1 | 0.04M | 13.1 |
| Linear + BN + ReLU + Max | 76.1 | 50.5 | 0.12M | 14.2 |
| **PointNet (Linear+BN+ReLU+Max)** | **77.4** | **52.3** | **0.48M** | **16.1** |
| PointNet (2 layers) | 77.8 | 52.7 | 0.96M | 19.4 |

**Interpretation:**
- Simple pooling (mean or max) without a learned layer loses 3-4 AP. The raw 9 features
  are not in a good space for pooling -- the learned linear layer projects them into a
  space where max pooling is more meaningful.
- The learned single-layer PointNet provides the best trade-off.
- A deeper PointNet (2 layers) adds only +0.4 AP at significant speed cost.
- The impact is especially large for small objects (Ped: 46.2 vs 52.3) because they have
  fewer points, making the quality of the encoding more critical.

### 9.6 Feature Dimension Ablation

| Feature Dim (C) | Car 3D AP (Mod) | Params (Encoder) | Time (ms) |
|:---------------:|:---------------:|:----------------:|:---------:|
| 32 | 75.8 | 0.12M | 13.8 |
| **64** | **77.4** | **0.48M** | **16.1** |
| 128 | 77.7 | 1.92M | 21.4 |
| 256 | 77.5 | 7.68M | 33.2 |

**Interpretation:**
- C=64 is optimal. Doubling to C=128 gives only +0.3 AP at 33% speed cost.
- C=256 actually DECREASES performance slightly -- overfitting with too many parameters.
- The pillar feature dimension is not the bottleneck. The backbone (which processes a
  64-channel pseudo-image regardless of C) dominates compute.

---

## 10. Evaluation for Deployment

### 10.1 Real-World Considerations

When evaluating for production deployment on an autonomous vehicle, additional factors
matter beyond benchmark AP:

**Latency consistency:** Not just average speed but worst-case latency matters. A model
that averages 62 Hz but occasionally takes 50ms (20 Hz) could miss critical detections.

```
Measure: p50, p95, p99 latencies
  p50 = 16.1 ms (typical)
  p95 = 18.3 ms (most cases)
  p99 = 22.1 ms (worst case excl. outliers)

For safety: p99 must still be < sensor period (100ms for 10 Hz LiDAR)
```

**Range-stratified evaluation:** Performance varies dramatically with distance:

| Range | Car 3D AP | Points per Car | Difficulty |
|:-----:|:---------:|:--------------:|:----------:|
| 0-20m | 92.1 | 200-500 | Easy (many points) |
| 20-40m | 81.3 | 50-200 | Medium |
| 40-60m | 63.7 | 20-50 | Hard (few points) |
| 60-80m | 41.2 | 5-20 | Very hard |

**Weather degradation:** LiDAR performance degrades in rain, fog, and snow due to
scattering of laser pulses. Evaluation should include adverse weather if available.

**Edge cases:** Unusual objects (construction equipment, animals, debris) are not in
standard benchmarks. Production systems need separate evaluation for these scenarios.

### 10.2 Deployment Optimization Evaluation

When optimizing for deployment (TensorRT, INT8 quantization), evaluate:

| Optimization | FPS Gain | AP Change | Acceptable? |
|--------------|:--------:|:---------:|:-----------:|
| FP32 (baseline) | 62 Hz | 0.0 | Baseline |
| FP16 | 95 Hz | -0.1 AP | Yes (negligible loss) |
| INT8 | 140 Hz | -0.8 AP | Usually yes |
| INT8 + pruned backbone | 200 Hz | -2.1 AP | Depends on requirements |

Always re-evaluate after optimization -- quantization can cause unexpected accuracy drops
on specific classes or scenarios.

### 10.3 Failure Mode Analysis for Deployment

Beyond aggregate metrics, deployment evaluation should analyze specific failure modes:

| Failure Mode | How to Detect | Risk Level | Mitigation |
|--------------|---------------|:----------:|------------|
| Missed pedestrian at close range | Range-stratified recall for Ped class | Critical | Lower score threshold for Ped |
| False positive in empty space | Precision analysis in no-object regions | Medium | Raise score threshold |
| Wrong heading | mAOE > 0.5 rad on specific classes | High | Improve direction head |
| Merged detections | Cases where 2 GT map to 1 pred | High | Lower NMS threshold |
| Ghost detections from reflections | FP in specific locations (bridges, guardrails) | Medium | Train with hard negatives |

### 10.4 Evaluation Pipeline for Continuous Integration

For production systems, automated evaluation should run on every model update:

```python
def ci_evaluation_pipeline(model_checkpoint, eval_dataset):
    """Run complete evaluation suite for CI/CD."""

    # 1. Standard metrics
    results = evaluate_standard(model_checkpoint, eval_dataset)
    assert results['Car_3D_AP_Moderate'] > 75.0, "Car AP regression"
    assert results['Ped_3D_AP_Moderate'] > 48.0, "Ped AP regression"

    # 2. Speed test
    latency = measure_speed(model_checkpoint)
    assert latency.p99 < 25.0, "Latency p99 regression"
    assert latency.mean < 18.0, "Mean latency regression"

    # 3. Range-stratified
    range_results = evaluate_by_range(model_checkpoint, eval_dataset)
    assert range_results['Car_0-20m'] > 88.0, "Close-range regression"
    assert range_results['Ped_0-30m'] > 55.0, "Ped close-range regression"

    # 4. Corner cases
    corner_results = evaluate_corner_cases(model_checkpoint)
    assert corner_results['occluded_pedestrian_recall'] > 0.40

    return results
```

### 10.5 Evaluation Metrics Glossary

| Term | Definition |
|------|-----------|
| AP | Average Precision: area under precision-recall curve |
| mAP | Mean AP across all classes |
| R40 | 40-point recall interpolation (KITTI standard since 2017) |
| R11 | 11-point recall interpolation (legacy KITTI metric) |
| IoU | Intersection over Union: overlap between predicted and GT boxes |
| BEV | Bird's Eye View: top-down 2D projection |
| NDS | nuScenes Detection Score: combined mAP + TP metrics |
| TP | True Positive: correct detection |
| FP | False Positive: incorrect detection (no matching GT) |
| FN | False Negative: missed GT object (not detected) |
| NMS | Non-Maximum Suppression: removes duplicate detections |
| mATE | Mean Average Translation Error (nuScenes) |
| mASE | Mean Average Scale Error (nuScenes) |
| mAOE | Mean Average Orientation Error (nuScenes) |
| mAVE | Mean Average Velocity Error (nuScenes) |
| mAAE | Mean Average Attribute Error (nuScenes) |

### 10.6 Per-Class Performance Analysis

Understanding per-class performance helps identify specific weaknesses:

```
KITTI Per-Class Analysis (PointPillars):

Car (IoU threshold 0.7):
  - Best performing class (most training data, largest objects)
  - Main failure: long-range vehicles (>60m) with few points
  - BEV AP >> 3D AP gap is small (1-2%), indicating good height estimation

Pedestrian (IoU threshold 0.5):
  - Challenging: small objects, few LiDAR points (5-50 per pedestrian)
  - Main failure: occluded pedestrians (partially behind vehicles)
  - Significant BEV-3D gap (5-8%) indicating height estimation difficulty
  - High variance across seeds (+/- 0.82 AP)

Cyclist (IoU threshold 0.5):
  - Most variable class (fewest training instances: ~1600)
  - Main failure: confusion with pedestrians (similar height/width)
  - Benefits most from GT database sampling augmentation
  - Highest variance across seeds (+/- 1.12 AP)
```

### 10.7 Comparison with Human Performance

For context, trained human annotators achieve approximately:
- Car 3D annotation consistency: ~95% IoU between annotators (boxes are well-defined)
- Pedestrian 3D annotation consistency: ~80% IoU (ambiguous in sparse regions)
- Heading annotation consistency: +/- 5 degrees for cars, +/- 15 degrees for pedestrians

PointPillars performance (78% AP for Car at IoU 0.7) is not directly comparable to human
annotation consistency, but it indicates that the model correctly identifies and localizes
the majority of objects to within the IoU threshold.

---

## 11. Evaluation Checklist

### 11.1 Before Running Evaluation

```
[ ] Model checkpoint is from the BEST epoch (by validation AP), not the last epoch
[ ] Evaluation uses the validation set (NOT training set)
[ ] Score threshold matches the intended use (0.1 for benchmark, higher for deployment)
[ ] NMS thresholds match the paper's settings
[ ] Point cloud range matches training range exactly
[ ] Coordinate frame is consistent between predictions and ground truth
[ ] DontCare regions are properly handled (excluded from evaluation)
[ ] Difficulty filtering is applied correctly
```

### 11.2 After Running Evaluation

```
[ ] Results are within expected range (compare to paper +/- 1-2 AP)
[ ] Per-class results are reported (not just aggregate)
[ ] Hardware and software versions are documented
[ ] Random seed is recorded for reproducibility
[ ] Both BEV and 3D AP are reported
[ ] Inference speed is measured separately from evaluation
```

---

## Summary of Best Practices

1. **Always report the evaluation protocol**: 40-point vs 11-point for KITTI
2. **Use consistent hardware**: Report GPU model, CUDA version, batch size
3. **Warm up before timing**: Discard first 50 iterations
4. **Control for randomness**: Average over multiple seeds (3+ recommended)
5. **Use official evaluation tools**: Minor implementation differences cause 1-2% discrepancy
6. **Compare fairly**: Match training schedules, augmentation, and data splits
7. **Report range-stratified results** for deployment evaluation
8. **Measure latency percentiles** (p50, p95, p99) not just mean
9. **Analyze per-class performance** to identify specific weaknesses
10. **Test deployment optimizations** (FP16, INT8) with full re-evaluation

---

## Section 12: Official Evaluation Tool Usage

### 12.1 KITTI Evaluation Tool Setup

The official KITTI evaluation tool is a C++ program that must be compiled locally.
It computes AP using the 40-point interpolation method (post-2017) or the legacy
11-point method.

```
# Step 1: Download the evaluation code from KITTI benchmark
# The evaluation code is distributed as part of the devkit

# Step 2: Compile the evaluation binary
cd kitti_native_eval/
g++ -O2 -o evaluate_object evaluate_object.cpp -lm

# Step 3: Prepare predictions in KITTI format
# Each prediction file: one line per detection
# Format: type truncation occlusion alpha x1 y1 x2 y2 h w l x y z ry score
# Example line:
# Car 0.0 0 -1.57 614.24 181.78 727.31 284.77 1.50 1.62 3.88 1.84 1.47 8.41 -1.56 0.92

# Step 4: Run evaluation
./evaluate_object /path/to/predictions /path/to/groundtruth
```

Important notes on format:
- Predictions MUST be in camera coordinates (not LiDAR coordinates)
- The score field is critical: evaluation ranks detections by score for AP computation
- One file per frame, named by frame index (e.g., 000001.txt)
- Empty files mean zero detections for that frame (this is valid)

### 12.2 nuScenes Evaluation Tool Setup

The nuScenes devkit provides a Python-based evaluation tool with rich metrics.

```python
# Install the nuScenes devkit
pip install nuscenes-devkit

# Run evaluation from Python
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

# Load the nuScenes dataset
nusc = NuScenes(version='v1.0-trainval', dataroot='/data/nuscenes')

# Configure evaluation
cfg = config_factory('detection_cvpr_2019')

# Run evaluation
nusc_eval = DetectionEval(
    nusc,
    config=cfg,
    result_path='/path/to/results.json',
    eval_set='val',
    output_dir='/path/to/output/',
    verbose=True
)
metrics, metric_data_list = nusc_eval.main(render_curves=True)
```

The results.json format requires:
- Each detection: translation (x,y,z), size (w,l,h), rotation (quaternion),
  detection_name, detection_score, velocity (vx, vy), attribute_name
- All coordinates in the global frame
- Quaternion rotation (not Euler angles)

### 12.3 Common Format Conversion Errors

| Error | Symptom | Fix |
|-------|---------|-----|
| Wrong coordinate frame | AP drops to near 0% | Apply calibration matrices |
| Swapped width/length | AP drops 5-15% | Check dimension ordering convention |
| Radians vs degrees | Orientation AP degrades | KITTI uses radians |
| Missing score field | Evaluation crashes | Add confidence score to each detection |
| Wrong quaternion convention | nuScenes AP drops | Use scipy Rotation for conversion |
| Frame ID mismatch | Missing evaluations | Verify filename/token mapping |

---

## Section 13: Multi-Class Evaluation Analysis

### 13.1 Understanding Per-Class Performance Gaps

PointPillars typically shows uneven performance across object classes. Understanding
why helps guide improvements:

```
Typical KITTI Results Breakdown:
+----------+--------+----------+------+----------------------------+
| Class    | Easy   | Moderate | Hard | Why This Performance?      |
+----------+--------+----------+------+----------------------------+
| Car      | 87.4%  | 77.3%    | 74.9%| Many training examples,    |
|          |        |          |      | consistent size, high      |
|          |        |          |      | reflectivity               |
+----------+--------+----------+------+----------------------------+
| Cyclist  | 82.1%  | 63.5%    | 59.2%| Fewer examples, variable   |
|          |        |          |      | pose, thin profile from    |
|          |        |          |      | some angles                |
+----------+--------+----------+------+----------------------------+
| Pedestrian| 57.8% | 52.1%    | 47.9%| Fewest LiDAR points,       |
|          |        |          |      | highly variable appearance,|
|          |        |          |      | small footprint in BEV     |
+----------+--------+----------+------+----------------------------+
```

### 13.2 Why Pedestrians Are Hardest

A pedestrian at 30m returns approximately 5-15 LiDAR points. At 50m, this drops
to 2-5 points. The pillar encoding must extract meaningful features from these
sparse inputs:

```
Points returned vs range (typical 64-beam LiDAR):

  Points
  200 |****
      |****
  150 |*****
      |*****
  100 |*******
      |*******          Car
   50 |**********
      |*************
   20 |...                   Pedestrian
   10 |......
    5 |..........
      +---+---+---+---+----> Range (m)
      10  20  30  40  50
```

Strategies to improve pedestrian detection:
1. Use smaller pillar sizes (0.1m x 0.1m) in near range
2. Increase max_points_per_pillar for dense nearby regions
3. Apply heavier GT database sampling for pedestrians (sample_ratio: 15-20)
4. Consider multi-scale pillar approaches with finer grid for small objects

### 13.3 Cross-Dataset Class Mapping

When comparing results across datasets, class definitions differ:

```
KITTI classes     nuScenes classes          Mapping notes
-----------       ----------------          -------------
Car           --> car                       Direct mapping
              --> construction_vehicle      No KITTI equivalent
              --> bus                       No KITTI equivalent
              --> trailer                   No KITTI equivalent
              --> truck                     Partially "Van" in KITTI
Pedestrian    --> pedestrian                Direct mapping
Cyclist       --> bicycle + motorcycle      nuScenes separates these
              --> barrier                   No KITTI equivalent
              --> traffic_cone              No KITTI equivalent
```

This means nuScenes mAP and KITTI AP are NOT directly comparable, even ignoring
the different matching criteria (distance vs IoU).

---

## Section 14: Confidence Calibration Analysis

### 14.1 What Is Confidence Calibration?

A well-calibrated model means: when it predicts 90% confidence, approximately 90%
of those predictions are correct. Most object detectors, including PointPillars,
are poorly calibrated out of the box.

### 14.2 Measuring Calibration

Compute the Expected Calibration Error (ECE):

```
ECE = sum over bins: (|bin_size| / N) * |accuracy(bin) - confidence(bin)|

Procedure:
1. Sort all detections by confidence score
2. Divide into B bins (typically 10-15)
3. For each bin, compute:
   - Average confidence (mean of scores in bin)
   - Accuracy (fraction of TPs at IoU > threshold)
4. ECE is the weighted average of |accuracy - confidence|
```

Typical PointPillars calibration (before calibration):
```
Reliability Diagram:

  Accuracy
  1.0 |                              /
      |                           ../
  0.8 |                        ../
      |                     ../      Ideal (diagonal)
  0.6 |                ../
      |            ../
  0.4 |        .x'      x = actual model
      |     .x'
  0.2 | .x'
      |x'
  0.0 +---+---+---+---+---+----> Confidence
      0   0.2 0.4 0.6 0.8 1.0

Model is overconfident: actual accuracy < predicted confidence
```

### 14.3 Post-Hoc Calibration Methods

Temperature scaling is the simplest calibration fix:

```
calibrated_score = sigmoid(logit(raw_score) / T)

where T > 1 reduces overconfidence
      T < 1 increases confidence (rare for detectors)
      T is optimized on the validation set using NLL
```

Impact on downstream systems:
- Planning modules often threshold at specific confidences
- A miscalibrated model may pass too many false positives to tracking
- Calibration does NOT change AP (rank ordering is preserved)
- Calibration DOES improve decision-making in the full autonomy stack
