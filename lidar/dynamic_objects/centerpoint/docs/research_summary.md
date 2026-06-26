# Research Summary: Center-based 3D Object Detection and Tracking

## Paper Reference

**Title:** Center-based 3D Object Detection and Tracking  
**Authors:** Tianwei Yin, Xingyi Zhou, Philipp Krahenbuhl  
**Venue:** CVPR 2021  
**arXiv:** 2006.11275  

---

## Core Contribution

CenterPoint introduces a center-based representation for 3D object detection and tracking from LiDAR point clouds. Instead of relying on predefined anchor boxes (as in PointPillars, SECOND, PartA2), CenterPoint represents objects as points in a bird's-eye view (BEV) and detects them via keypoint estimation. This paradigm simplifies the detection pipeline, eliminates anchor hyperparameter tuning, and naturally extends to multi-object tracking.

---

## What is Anchor-Based Detection?

### The Anchor Concept

If you come from 2D object detection (Faster R-CNN, SSD, RetinaNet), you already know anchors: they are pre-defined reference boxes tiled densely across the feature map. The network does not predict boxes from scratch; instead, it predicts *adjustments* (residuals) to these reference boxes. The key assumption is that one of the many anchors will be "close enough" to the ground truth that predicting a small residual is easier than predicting the full box parameters.

In 3D LiDAR detection, anchors work the same way but in bird's-eye view (BEV):

```
  BEV Feature Map (e.g., 200 x 200 cells)
  +-------+-------+-------+-------+-------+
  |  A A  |  A A  |  A A  |  A A  |  A A  |
  |  A A  |  A A  |  A A  |  A A  |  A A  |
  +-------+-------+-------+-------+-------+
  |  A A  |  A A  |  A A  |  A A  |  A A  |
  |  A A  |  A A  |  A A  |  A A  |  A A  |
  +-------+-------+-------+-------+-------+
  |  A A  |  A A  |  A A  |  A A  |  A A  |
  |  A A  |  A A  |  A A  |  A A  |  A A  |
  +-------+-------+-------+-------+-------+

  Each cell has N anchors (e.g., N=2 per class for 0° and 90° orientations)
  Each anchor: (x, y, z, w, l, h, theta) — a full 3D box proposal
```

At every spatial location in the BEV feature map, the detector places N anchor boxes per class. Each anchor has a predefined width, length, height, and orientation. The network then classifies each anchor as foreground/background and regresses residuals to refine the box.

### How PointPillars and SECOND Use Anchors

**PointPillars** and **SECOND** both follow this recipe:

1. Define class-specific anchor dimensions from dataset statistics:
   - Car: (w=1.93, l=4.73, h=1.73) meters
   - Truck: (w=2.51, l=10.13, h=3.22) meters
   - Pedestrian: (w=0.73, l=0.67, h=1.77) meters
   - Cyclist: (w=0.60, l=1.70, h=1.28) meters
   - Bus: (w=2.94, l=12.34, h=3.47) meters

2. At each BEV cell, place 2 anchors per class (heading = 0° and heading = 90°).

3. For a feature map of size 200x200 with 5 classes and 2 orientations:
   - Total anchors = 200 x 200 x 5 x 2 = **400,000 anchors**

4. For each anchor, the network predicts:
   - Classification score (foreground vs background)
   - Regression residuals: (dx, dy, dz, dw, dl, dh, d_theta)
   - Direction classification (to resolve 180° ambiguity)

5. After inference, apply class-specific NMS to remove duplicate detections.

### Problems with the Anchor-Based Approach

**Problem 1: Anchor size hyperparameters must be manually defined per class**

Every new class or dataset requires recomputing anchor statistics. If your anchors are poorly sized, matching fails and the network cannot learn. For a dataset with 10+ classes spanning motorcycles to articulated trucks, this becomes tedious and error-prone.

**Problem 2: Massive number of anchors creates computational burden**

With 400,000 anchors, the classification head alone produces 400K scores. The regression head produces 400K x 7 = 2.8M values. Most anchors are negative (background), leading to extreme class imbalance that requires focal loss or hard negative mining.

**Problem 3: IoU-based matching in 3D is expensive and complex**

To assign ground-truth boxes to anchors during training, you must compute the IoU (Intersection over Union) between every anchor and every ground-truth box. In 3D with rotation, this requires computing rotated IoU — a significantly more expensive operation than axis-aligned 2D IoU. The Shapely polygon intersection or custom CUDA kernels are needed.

**Problem 4: Orientation regression is discontinuous**

The yaw angle wraps around: 0° and 360° represent the same orientation, but their L1 distance is 360°. Even with sin/cos decomposition, the direction ambiguity (a car facing 0° looks identical from above to one facing 180°) requires an additional direction classification head.

**Problem 5: NMS is required and has sensitive hyperparameters**

Because many anchors overlap with the same object, the network produces many high-confidence detections for a single object. Non-Maximum Suppression removes duplicates, but its IoU threshold is class-specific and dataset-specific. Too aggressive → miss nearby objects. Too lenient → duplicate detections.

---

## Center-Based Paradigm: The Key Insight

### Objects as Points

CenterPoint's fundamental insight: **an object can be represented by a single point — its center in BEV space**. This reduces detection to keypoint estimation, a well-understood problem in 2D vision (think: pose estimation, CenterNet for 2D detection).

```
  Ground Truth Boxes              Center Points             Gaussian Heatmap
  +------------------+            
  |    +--------+    |                 *                     . . . . . . . .
  |    |  Car   |    |                                      . . . . . . . .
  |    |   *    |    |            *         *               . .***. . . . .
  |    +--------+    |                                      . .*X*. . . . .
  |         +----+   |                  *                   . .***.*X*.. .
  |         |Ped |   |                                      . . . .***. . .
  |         | *  |   |                                      . . .**X**. . .
  |         +----+   |                                      . . .***. . . .
  +------------------+                                      . . . . . . . .

  Step 1: Extract centers    Step 2: Render Gaussians    Step 3: Find peaks
  from GT bounding boxes     at each center location     via 3x3 max pooling
```

### Why This is Rotation-Invariant

The center of a car does not move when the car rotates. Whether a vehicle faces north, south, east, or west, its BEV center coordinates (x, y) remain the same. This means the detection head (heatmap peak finding) is completely decoupled from orientation — orientation is predicted by a separate regression head only *after* the center is detected.

In contrast, anchor-based methods couple detection with orientation: the anchor at 0° might match a car facing 0° but fail to match the same car facing 45°, requiring more anchors to cover orientation diversity.

### Why NMS-Free (Or Nearly So)

In a well-trained heatmap, each object produces exactly one Gaussian peak. Peaks are isolated by construction — two objects whose centers are close produce slightly overlapping Gaussians, but each has its own local maximum. A simple 3x3 max pooling operation extracts these peaks:

```
  Heatmap values (zoomed in):

     0.1  0.2  0.3  0.2  0.1
     0.2  0.5  0.7  0.5  0.2
     0.3  0.7  1.0  0.7  0.3    <-- center pixel is the local maximum
     0.2  0.5  0.7  0.5  0.2
     0.1  0.2  0.3  0.2  0.1

  After 3x3 max pooling, only the center pixel (1.0) equals
  its max-pooled value. All other pixels are suppressed.
```

This eliminates the need for expensive, threshold-sensitive NMS. In practice, CenterPoint applies a very light NMS only for the optional two-stage refinement (and even that can be omitted with negligible mAP change).

### From Peaks to Detections

Once peaks are extracted (top-K scoring peaks, e.g., K=500), the detection is assembled by reading regression head outputs at each peak location:

1. Peak location gives (x_grid, y_grid) — the quantized BEV position.
2. Sub-voxel offset head gives (dx, dy) — corrects for quantization.
3. Height head gives z_center.
4. Size head gives (w, l, h).
5. Rotation head gives (sin_theta, cos_theta).
6. Velocity head gives (vx, vy).

Final detection: (x_grid * stride + dx, y_grid * stride + dy, z_center, w, l, h, atan2(sin_theta, cos_theta), vx, vy, confidence)

---

## Pipeline Overview

```
Raw Point Cloud (e.g., 300,000 points from one LiDAR sweep)
    |
    v
Voxelization (divide 3D space into regular grid cells)
    |
    v
3D Sparse Convolutional Backbone (extract volumetric features, preserve sparsity)
    |
    v
BEV Feature Map (collapse Z-axis: 3D volume -> 2D map)
    |
    v
2D Backbone + Neck (refine BEV features with ResNet blocks + FPN deconv)
    |
    v
+-----------------------------------------------------+
|              Detection Heads (shared backbone)        |
+-----------------------------------------------------+
| Center Heatmap  | Offset | Height | Size | Rot | Vel |
+-----------------------------------------------------+
    |
    v
Peak Extraction (3x3 max pool + top-K selection)
    |
    v
[Optional] Two-Stage Refinement (point feature extraction + MLP)
    |
    v
Tracking (greedy center-distance matching with velocity prediction)
    |
    v
Final Output: 3D Bounding Boxes + Track IDs + Velocities
```

---

## Voxelization

The input point cloud is discretized into a regular 3D grid of voxels. Each voxel aggregates the points falling within it:

- **nuScenes:** voxel size = [0.075, 0.075, 0.2] m, point cloud range = [-54, -54, -5, 54, 54, 3] m
- **Waymo:** voxel size = [0.1, 0.1, 0.15] m, point cloud range = [-75.2, -75.2, -2, 75.2, 75.2, 4] m

Within each voxel, points are encoded using mean VFE (Voxel Feature Encoding) or simple mean pooling of point coordinates and features (x, y, z, intensity, time_lag).

### Voxel Grid Dimensions

For nuScenes with voxel size [0.075, 0.075, 0.2] and range [-54, -54, -5, 54, 54, 3]:
- X dimension: (54 - (-54)) / 0.075 = 1440 voxels
- Y dimension: (54 - (-54)) / 0.075 = 1440 voxels
- Z dimension: (3 - (-5)) / 0.2 = 40 voxels
- Total grid: 1440 x 1440 x 40 = ~83 million cells

But only ~1-5% of cells are occupied (LiDAR is sparse), which is why sparse convolutions are essential.

---

## 3D Sparse Convolutional Backbone

The voxelized representation is processed by a 3D sparse convolutional network:

- **Architecture:** 4 stages with increasing channel dimensions (e.g., 16 -> 32 -> 64 -> 128).
- **Operations:** Combination of Submanifold Sparse Convolution (SubMConv3d) that preserves sparsity patterns, and regular Sparse Convolution (SparseConv3d) with stride > 1 for downsampling.
- **Strides:** [1, 2, 4, 8] across the 4 stages (spatial dimensions halved at each strided stage).
- **Sparse tensors:** Only active (non-empty) voxels participate in computation, making the network efficient despite the large 3D volume.

The output is a sparse 3D feature volume at 8x downsampled resolution.

### Submanifold vs Regular Sparse Convolution

**Submanifold Sparse Conv:** Output is active only where the input was active. No new active sites are created. This preserves the sparsity pattern exactly.

**Regular Sparse Conv (stride=2):** Output is active where the kernel overlaps any active input. This "dilates" the active set slightly and reduces resolution. Used for downsampling.

```
  Submanifold (preserves sparsity):     Regular with stride 2 (dilates + downsamples):
  
  Input:    . X . .                      Input:    . X . .
            . X X .                                . X X .
            . . X .                                . . X .
            . . . .                                . . . .

  Output:   . X . .                      Output:   X X
            . X X .   (same pattern)              X X  (lower res, denser)
            . . X .
            . . . .
```

---

## BEV Feature Extraction

The 3D sparse feature volume is collapsed along the Z (height) axis to produce a dense 2D bird's-eye view feature map:

- Reshape: (X, Y, Z, C) -> (X, Y, Z*C) by concatenating features along the height dimension.
- This produces a dense BEV feature map of shape (H/8, W/8, C') where C' = Z_bins * C_last_stage.

A 2D convolutional backbone (ResNet-style) then refines the BEV features:

- **Two stages:** stride-1 and stride-2 blocks.
- **Feature Pyramid / Deconvolution:** Upsample lower-resolution features and concatenate with higher-resolution features to produce a multi-scale BEV representation.
- Final BEV feature map resolution: typically H/8 x W/8 (same as backbone output).

---

## Center Heatmap Prediction

Objects are detected as peaks in class-specific heatmaps:

- **Heatmap output:** One channel per class (or class group), spatial resolution = BEV feature map resolution.
- **Ground truth generation:** For each object, render a 2D Gaussian centered at the object's BEV center. The Gaussian radius is determined by the object's BEV footprint (size-adaptive).
- **Loss:** Gaussian Focal Loss (modified focal loss that accounts for the Gaussian ground truth distribution), which reduces penalty for predictions near but not exactly at the center.
- **Peak extraction:** At inference, apply a 3x3 max pooling to find local maxima, then take the top-K peaks (e.g., K=500) as candidate detections.

---

## Gaussian Focal Loss Explained

### Why Standard Focal Loss Fails for Heatmaps

Standard focal loss (from RetinaNet) is designed for binary classification targets: each pixel is either 0 (background) or 1 (foreground). The formula is:

```
FL(p, y) = -alpha * (1-p)^gamma * log(p)       if y = 1
           -(1-alpha) * p^gamma * log(1-p)      if y = 0
```

But heatmap targets are NOT binary. They are continuous Gaussians:

```
  Ground truth heatmap (1D slice through a car center):

  y:  0.0  0.01  0.1  0.5  1.0  0.5  0.1  0.01  0.0
                              ^
                           center
```

If you naively apply standard focal loss with y=0 at locations where the Gaussian is 0.5, you heavily penalize predictions near the center as false positives. This discourages the network from producing the smooth, high-confidence peaks it should.

### The Gaussian Focal Loss Formula

CenterPoint uses a modified focal loss (from CornerNet/CenterNet) that accounts for the Gaussian ground truth:

```
L_heatmap = -1/N * sum over all pixels:

  If y_xy = 1 (at the exact center):
      (1 - p_hat)^alpha * log(p_hat)

  If y_xy < 1 (near or far from center):
      (1 - y_xy)^beta * (p_hat)^alpha * log(1 - p_hat)
```

Where:
- `y_xy` = ground truth heatmap value at pixel (x, y), ranging from 0 to 1
- `p_hat` = predicted heatmap value (after sigmoid)
- `alpha` = 2 (focusing parameter, same role as in standard focal loss)
- `beta` = 4 (controls the penalty reduction near the Gaussian peak)
- `N` = number of objects (normalizer)

### How Each Term Works

**At the exact center (y_xy = 1):**
- Standard focal loss for positive samples: `(1 - p_hat)^alpha * log(p_hat)`
- Penalizes under-confident predictions. If the network predicts 0.9 instead of 1.0, the loss is small. If it predicts 0.1, the loss is large.

**Near the center (e.g., y_xy = 0.7):**
- The `(1 - y_xy)^beta` term = `(1 - 0.7)^4` = `0.3^4` = 0.0081
- This *dramatically* reduces the penalty for false-positive predictions near the center.
- The network is barely penalized for predicting high confidence at locations close to an object center.

**Far from any center (y_xy = 0.0):**
- The `(1 - y_xy)^beta` term = `(1 - 0.0)^4` = 1.0
- Full penalty applied — this is a true negative location.

### Why the Gaussian Radius Matters

The Gaussian radius determines how "wide" the target blob is. It is computed from the object's BEV size:

```python
# Simplified radius computation
radius = gaussian_radius(object_height_pixels, object_width_pixels, min_overlap=0.1)
# For a car ~4.7m long at 0.075m/px resolution: ~62 pixels long in BEV
# Radius is typically 3-6 pixels depending on implementation
```

A larger radius means:
- More pixels get non-zero target values (softer supervision near center)
- The network is less penalized for slightly offset peak predictions
- Good for large objects (trucks, buses) where center localization is harder

A smaller radius means:
- Tighter supervision, higher localization precision required
- Good for small objects (pedestrians, cones) where precision matters

---

## Regression Heads

At each detected center location, separate regression heads predict:

| Head | Output | Description |
|------|--------|-------------|
| Sub-voxel offset | (dx, dy) | Compensates for quantization error from BEV discretization |
| Height | z_center | Absolute height of object center above ground |
| Size | (log_w, log_l, log_h) | Log-normalized 3D dimensions |
| Rotation | (sin_theta, cos_theta) | Yaw angle decomposed to avoid discontinuity |
| Velocity | (vx, vy) | BEV velocity for tracking (nuScenes) |

All regression heads share the BEV backbone features and use lightweight convolutional layers (typically 2 conv layers with 64 channels + final 1x1 projection).

### Why sin/cos for Rotation Instead of Raw Angle

Raw angle regression suffers from the wraparound discontinuity:
- 359° and 1° are functionally close (2° apart) but numerically far (358 units in L1)
- The loss landscape has a cliff at the 0°/360° boundary

By predicting (sin_theta, cos_theta), the representation is continuous everywhere:
- sin(359°) = -0.017, sin(1°) = 0.017 → smooth, close values
- The network can recover the angle: theta = atan2(sin_theta, cos_theta)

However, there is still a 180° ambiguity (a car facing 0° looks the same from above as one facing 180° for a symmetric object). CenterPoint handles this with a direction classification head that predicts which 180° half the heading falls into.

---

## Two-Stage Refinement: Step-by-Step

CenterPoint optionally employs a second stage to refine first-stage detections. This section explains the full mechanism.

### Why a Second Stage?

The first-stage detection reads features from a *single point* — the predicted center. For small objects (pedestrians), this captures enough context. But for large objects (trucks: 10m+, buses: 12m+), the center feature may miss information about the object's extent, shape, and boundaries. The object might be partially occluded, and the center feature alone cannot capture what is visible at the edges.

### The Five Sampling Points

For each first-stage detection, the second stage samples BEV features at 5 locations:

```
  Top view of a detected box:

       Face Center (front)
              |
    +---------*---------+
    |                   |
    |  Face    *    Face|
    |  Center  |  Center|
    *  (left)  |  (right)*
    |          |        |
    |          |        |
    +---------*---------+
              |
       Face Center (back)
              
              * = Center point

  5 sampling points:
  1. Box center (x, y)
  2. Front face center (x + l/2 * cos_theta, y + l/2 * sin_theta)
  3. Back face center (x - l/2 * cos_theta, y - l/2 * sin_theta)
  4. Left face center (x - w/2 * sin_theta, y + w/2 * cos_theta)
  5. Right face center (x + w/2 * sin_theta, y - w/2 * cos_theta)
```

### Feature Extraction and Refinement

```
Step 1: For each of the 5 points, bilinearly interpolate BEV features
        feature_center = bilinear_sample(bev_map, x_center, y_center)  # shape: [C]
        feature_front  = bilinear_sample(bev_map, x_front, y_front)    # shape: [C]
        feature_back   = bilinear_sample(bev_map, x_back, y_back)      # shape: [C]
        feature_left   = bilinear_sample(bev_map, x_left, y_left)      # shape: [C]
        feature_right  = bilinear_sample(bev_map, x_right, y_right)    # shape: [C]

Step 2: Concatenate all features
        combined = concat([feature_center, feature_front, feature_back,
                          feature_left, feature_right])  # shape: [5*C]

Step 3: MLP refinement
        hidden = ReLU(Linear(5*C -> 256))
        confidence_delta = Linear(256 -> 1)   # adjust detection confidence
        box_delta = Linear(256 -> 7)          # refine (dx, dy, dz, dw, dl, dh, d_theta)

Step 4: Apply refinements
        final_confidence = stage1_confidence + confidence_delta
        final_box = stage1_box + box_delta
```

### Improvement from Two-Stage

On nuScenes:
- Stage 1 only: 56.4 mAP
- Stage 1 + Stage 2: 58.0 mAP (+1.6)
- Largest gains on trucks (+3.2 mAP) and construction vehicles (+2.8 mAP) — large objects where edge features help most.

---

## Velocity Prediction and Greedy Tracking

### Multi-Sweep Input

The nuScenes LiDAR operates at 20 Hz, but keyframes are annotated at 2 Hz. CenterPoint uses multi-sweep input: the last 10 LiDAR sweeps (0.5 seconds of data) are aggregated into a single point cloud. Each point carries an additional feature: **time_lag** — how many seconds ago it was captured.

```
  Multi-sweep point cloud (top view):

  Sweep t=0.0s (current):    **** (car position now)
  Sweep t=0.05s:             ****  (car position 50ms ago, shifted slightly)
  Sweep t=0.10s:              ****
  Sweep t=0.15s:               ****
  ...
  Sweep t=0.45s:                       **** (car position 450ms ago)

  The temporal displacement of points encodes velocity information.
  A stationary car: all sweeps align perfectly.
  A moving car: sweeps form a "trail" in the direction of motion.
```

The network learns to extract velocity from this temporal point pattern. The velocity prediction head outputs (vx, vy) in meters per second, trained with L1 loss against ground-truth velocities.

### Tracking Algorithm: Full Pseudocode

```
Algorithm: CenterPoint Greedy Tracking

Initialize:
    active_tracks = []
    next_track_id = 0
    max_age = 3          # frames without match before deletion
    match_threshold = 2.0  # meters; max distance for valid match

For each new frame at time t:
    # Step 1: Run detection
    detections = CenterPoint(point_cloud_t)
    # Each detection has: (x, y, z, w, l, h, theta, vx, vy, score, class)

    # Step 2: Predict where existing tracks should be now
    for each track in active_tracks:
        dt = t - track.last_update_time
        track.predicted_x = track.x + track.vx * dt
        track.predicted_y = track.y + track.vy * dt

    # Step 3: Compute cost matrix (L2 distance in BEV)
    cost_matrix = zeros(len(detections), len(active_tracks))
    for i, det in enumerate(detections):
        for j, track in enumerate(active_tracks):
            cost_matrix[i][j] = sqrt(
                (det.x - track.predicted_x)^2 +
                (det.y - track.predicted_y)^2
            )

    # Step 4: Greedy matching (NOT Hungarian — simpler and sufficient)
    matched_dets = set()
    matched_tracks = set()
    
    # Sort all pairs by cost (ascending)
    pairs = sorted([(cost_matrix[i][j], i, j) 
                    for i in range(len(detections))
                    for j in range(len(active_tracks))])
    
    for cost, det_idx, track_idx in pairs:
        if cost > match_threshold:
            break  # all remaining pairs are too far
        if det_idx in matched_dets or track_idx in matched_tracks:
            continue  # already assigned
        # Match this pair
        matched_dets.add(det_idx)
        matched_tracks.add(track_idx)
        # Update track with detection
        active_tracks[track_idx].x = detections[det_idx].x
        active_tracks[track_idx].y = detections[det_idx].y
        active_tracks[track_idx].vx = detections[det_idx].vx
        active_tracks[track_idx].vy = detections[det_idx].vy
        active_tracks[track_idx].age = 0
        active_tracks[track_idx].last_update_time = t

    # Step 5: Create new tracks for unmatched detections
    for i in range(len(detections)):
        if i not in matched_dets:
            new_track = Track(
                id = next_track_id,
                x = detections[i].x,
                y = detections[i].y,
                vx = detections[i].vx,
                vy = detections[i].vy,
                age = 0,
                last_update_time = t
            )
            active_tracks.append(new_track)
            next_track_id += 1

    # Step 6: Age and delete stale tracks
    for j in range(len(active_tracks)):
        if j not in matched_tracks:
            active_tracks[j].age += 1
    active_tracks = [t for t in active_tracks if t.age <= max_age]
```

### Why Greedy Matching Works (Not Hungarian)

The Hungarian algorithm finds the globally optimal assignment (minimum total cost). It has O(n^3) complexity. CenterPoint uses simple greedy matching instead, which is O(n^2 log n) for the sort + O(n) for the assignment.

**Why this is sufficient:**

1. Velocity prediction makes associations nearly unambiguous. If each track's predicted position is within 0.5m of the correct detection, assignments are obvious — no global optimization needed.

2. In autonomous driving, objects rarely have crossing trajectories when viewed in consecutive frames (50ms apart). The typical displacement at 60 km/h is only ~0.83m per frame.

3. Empirically, switching from greedy to Hungarian matching improves AMOTA by only 0.1-0.2 points on nuScenes — not worth the complexity.

### When Greedy Tracking Fails

- Very dense traffic with many objects at similar velocities (parking lots)
- Objects with failed velocity predictions (newly appeared objects have no temporal signal)
- Extremely fast-moving objects that travel > 2m between frames

For these edge cases, a Kalman filter or Hungarian matching would help, but CenterPoint shows that for standard highway/urban driving, greedy matching with velocity prediction is sufficient.

---

## Comparison with PointPillars, SECOND, 3DSSD, and Others

| Method | Year | Backbone | Detection Style | Two-Stage | Tracking | NMS Required | NDS (nuScenes) | mAP (nuScenes) | Latency (ms) |
|--------|------|----------|-----------------|-----------|----------|--------------|----------------|-----------------|-------------|
| PointPillars | 2019 | Pillar + 2D Conv | Anchor-based | No | No | Yes | 45.3 | 30.5 | ~25 |
| SECOND | 2018 | Voxel + Sparse Conv | Anchor-based | No | No | Yes | 53.5 | 42.1 | ~40 |
| PartA2 | 2020 | Voxel + Point | Anchor-based | Yes | No | Yes | - | - | ~80 |
| 3DSSD | 2020 | Point-based (SA+FP) | Anchor-free (vote) | No | No | Yes | 56.4 | 42.6 | ~38 |
| CenterPoint | 2021 | Voxel + Sparse Conv | Center-based | Optional | Yes | No* | 67.3 | 60.3 | ~45 |

*CenterPoint uses 3x3 max pooling instead of traditional NMS.

### Detailed Comparison

**PointPillars vs CenterPoint:**
- PointPillars is faster (pillar encoding avoids 3D conv) but less accurate.
- PointPillars cannot capture fine-grained height information (single pillar per vertical column).
- CenterPoint with pillar backbone (CenterPoint-Pillar) achieves 58.0 mAP — a +27.5 mAP improvement over PointPillars with similar speed.

**SECOND vs CenterPoint:**
- Same backbone concept (voxel + sparse 3D conv), but different detection heads.
- SECOND uses 200K+ anchors with class-specific NMS.
- CenterPoint produces ~500 detections directly from heatmap peaks.
- The detection head difference alone gives +18 mAP improvement.

**3DSSD vs CenterPoint:**
- 3DSSD uses raw points (no voxelization) — more accurate per-point features but harder to scale.
- 3DSSD is anchor-free but uses farthest point sampling + voting — different paradigm.
- CenterPoint is significantly more accurate (+17.7 mAP) and includes tracking.

**PartA2 vs CenterPoint:**
- PartA2 pioneered two-stage refinement in 3D detection.
- CenterPoint's two-stage is simpler (5-point feature + MLP vs. RoI-aware feature aggregation).
- CenterPoint achieves better accuracy with less complexity.

---

## Practical Implementation Notes

### Voxel Size Selection Tradeoffs

| Voxel Size | BEV Resolution | Memory | Accuracy | Speed |
|-----------|----------------|--------|----------|-------|
| 0.05m | 2160 x 2160 | Very high | Best | Slow |
| 0.075m | 1440 x 1440 | High | Very good | Medium |
| 0.1m | 1080 x 1080 | Medium | Good | Fast |
| 0.2m | 540 x 540 | Low | Moderate | Very fast |

Rules of thumb:
- **0.075m** is the standard for nuScenes (competition settings).
- **0.1m** is used for Waymo (larger range, need to keep memory manageable).
- Smaller voxels help small objects (pedestrians, cyclists) more than large objects.
- Halving voxel size quadruples memory and roughly doubles compute.

### How to Tune Heatmap Gaussian Radius

The Gaussian radius is critical for training stability and accuracy:

```python
def gaussian_radius(height, width, min_overlap=0.1):
    """
    Compute Gaussian radius such that a box centered at distance=radius
    from the GT center still has IoU >= min_overlap with the GT box.
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = sqrt(b1**2 - 4*a1*c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = sqrt(b2**2 - 4*a2*c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = sqrt(b3**2 - 4*a3*c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)
```

**Tuning guidelines:**
- `min_overlap=0.1` is the default and works well for most settings.
- If you see many missed small objects, try increasing the radius (lower min_overlap, e.g., 0.05).
- If you see poor localization for large objects, the radius might be too large — predictions get rewarded even when off-center.
- A common sanity check: visualize GT heatmaps. Each object should have a clearly distinct, non-overlapping peak.

### Training Schedule

The standard CenterPoint training recipe:

```
Optimizer: AdamW
Learning rate: 1e-3 (with OneCycleLR scheduler)
Weight decay: 0.01
Batch size: 4 per GPU (8 GPUs total = 32 effective batch)
Epochs: 20
LR schedule:
  - Warm up linearly for 1 epoch (1e-5 -> 1e-3)
  - Cosine anneal from epoch 1 to epoch 20 (1e-3 -> 1e-5)
  - "Fade" final 5 epochs with reduced data augmentation

Data augmentation:
  - Random global rotation: [-pi/4, pi/4]
  - Random global scaling: [0.95, 1.05]
  - Random flip (X-axis)
  - GT-AUG (copy-paste ground truth boxes from database)
  - Disable GT-AUG in final 5 epochs ("fade")

Loss weights:
  - Heatmap (Gaussian Focal Loss): 1.0
  - Regression (L1): 0.25 per head
  - Two-stage (if used): 1.0
```

**Why "fade" the augmentation:**
GT-AUG (pasting boxes from other scenes) creates unrealistic object placements. Training with it improves diversity but the network slightly overfits to impossible configurations. Disabling it for the last 5 epochs lets the network "calibrate" to realistic scenes, improving calibration and validation mAP by ~0.5-1.0 points.

### Common Failure Modes

**1. Small objects at long range (>50m)**
- Problem: At 50m, a pedestrian occupies only 1-2 voxels. The signal is too weak.
- Mitigation: Multi-resolution voxelization, finer voxels near ego, or range-stratified detection heads.
- CenterPoint's performance drops ~15-20 mAP for pedestrians beyond 50m.

**2. Heavily occluded objects**
- Problem: If <10% of an object's surface returns LiDAR points, the features are noisy.
- Mitigation: Multi-sweep helps (more chances to see the object), but fundamental LiDAR limitation.
- Two-stage helps here because face-center features can capture visible portions.

**3. Objects on the ground truth boundary (at max range)**
- Problem: Partial objects at detection range boundary get inconsistent supervision.
- Mitigation: Ignore objects with <5 LiDAR points during training.

**4. Similar adjacent objects (row of parked cars)**
- Problem: Heatmap peaks can merge if objects are separated by < 2 BEV pixels.
- Mitigation: Smaller voxel size, or use a higher-resolution detection head.
- At 0.075m voxel and 8x backbone stride, BEV pixel = 0.6m. Two parked cars 0.3m apart may merge.

**5. Objects with unusual aspect ratios (articulated trucks, long trailers)**
- Problem: The center point is far from both ends; single center feature misses endpoints.
- Mitigation: Two-stage refinement is essential for these objects.

---

## Key Results (from paper)

### nuScenes Test Set

| Model | NDS | mAP | AMOTA |
|-------|-----|-----|-------|
| CenterPoint (voxel) | 67.3 | 60.3 | 63.8 |
| CenterPoint (pillar) | 65.5 | 58.0 | - |

### Waymo Open Dataset (val)

| Model | Vehicle APH (L2) | Pedestrian APH (L2) | Cyclist APH (L2) |
|-------|-------------------|----------------------|-------------------|
| CenterPoint | 66.2 | 62.6 | 65.0 |

### Ablation: Detection Components

| Configuration | mAP | Delta |
|---------------|-----|-------|
| Baseline (anchor-based, single sweep) | 48.3 | - |
| + Center-based head | 52.1 | +3.8 |
| + Multi-sweep (10 sweeps) | 55.2 | +3.1 |
| + Two-stage refinement | 56.4 | +1.2 |
| + Larger backbone (double channels) | 58.0 | +1.6 |
| + Test-time augmentation | 60.3 | +2.3 |

### Ablation: Tracking Components

| Configuration | AMOTA | Delta |
|---------------|-------|-------|
| Hungarian matching (no velocity) | 55.2 | - |
| Greedy matching (no velocity) | 54.8 | -0.4 |
| Greedy matching + velocity prediction | 63.8 | +9.0 |

The velocity prediction contributes far more (+9.0) than the matching algorithm choice (-0.4). This validates the greedy approach.

---

## Significance and Impact

1. **Unified detection and tracking:** First method to achieve competitive performance on both tasks with a single model.
2. **Simplicity:** Removes anchor hyperparameters, complex NMS tuning, and expensive tracking modules.
3. **Generality:** Works with both voxel-based and pillar-based representations.
4. **Foundation for future work:** CenterPoint's center-based paradigm influenced subsequent methods like CenterFormer, TransFusion, BEVFusion, and LargeKernel3D.

### Downstream Influence

- **TransFusion (2022):** Replaces the 2D BEV backbone with transformers, using CenterPoint's heatmap queries as initialization for cross-attention.
- **BEVFusion (2022):** Fuses LiDAR BEV features with camera BEV features, using CenterPoint as the LiDAR branch.
- **LargeKernel3D (2023):** Replaces 3x3 sparse convolutions with large kernels (7x7, 11x11), keeping the CenterPoint detection head.
- **CenterFormer (2022):** Adds deformable attention in BEV, again using center-based detection.

The CenterPoint detection head (heatmap + regression) has become the de facto standard for LiDAR-based 3D detection, even in methods that change every other component.

---

## Summary for the PyTorch Practitioner

If you are implementing CenterPoint from scratch, here is the dependency chain:

```
1. Data pipeline:
   - Load point clouds (LidarPointCloud from nuscenes-devkit)
   - Multi-sweep aggregation (concatenate 10 sweeps with time_lag feature)
   - Voxelization (hard voxelization with max 20 points per voxel)

2. Model:
   - VoxelNet or MeanVFE: per-voxel feature encoding
   - SpMiddle3D: 3D sparse conv backbone (use spconv or torchsparse)
   - RPNV2 or SECONDFPN: 2D BEV backbone + neck
   - CenterHead: heatmap + regression heads (6 parallel conv branches)

3. Loss:
   - Gaussian focal loss for heatmap (custom implementation required)
   - L1 loss for all regression targets
   - Weighted sum: L_total = L_heatmap + 0.25 * (L_offset + L_height + L_size + L_rot + L_vel)

4. Inference:
   - Sigmoid on heatmap → 3x3 max pool → top-K peaks
   - Gather regression outputs at peak locations
   - Optional: two-stage refinement
   - Optional: greedy tracking across frames

5. Key libraries:
   - spconv (or spconv2.x) for sparse convolutions
   - nuscenes-devkit for data loading
   - pyquaternion for rotation handling
   - numba for CUDA-accelerated voxelization
```

---
