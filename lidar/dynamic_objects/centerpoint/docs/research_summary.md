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

## Center-Based Paradigm vs. Anchor-Based Methods

### Anchor-Based Approaches (PointPillars, SECOND)

- Pre-define a dense set of 3D anchor boxes at each BEV location with fixed sizes, orientations, and aspect ratios per class.
- Classification: predict whether each anchor contains an object.
- Regression: predict residuals (dx, dy, dz, dw, dl, dh, d_theta) relative to anchor dimensions.
- **Drawbacks:**
  - Large number of anchors (often 200k+) leading to heavy computation in the detection head.
  - Class-specific anchor dimensions require careful tuning per dataset.
  - IoU-based matching is complex in 3D due to rotation.
  - Non-maximum suppression (NMS) required as post-processing.

### Center-Based Approach (CenterPoint)

- Represent each object by its center point in BEV space.
- Predict a class-specific heatmap where each object appears as a Gaussian peak at its center location.
- Regression heads predict object properties (size, height, rotation, velocity) directly at detected center locations.
- **Advantages:**
  - Anchor-free: no need to define class-specific anchor dimensions.
  - Rotation-invariant detection: the center point is invariant to object orientation.
  - NMS-free (or minimal NMS): peaks in the heatmap are already local maxima.
  - Naturally extends to tracking via center displacement across frames.

---

## Pipeline Overview

```
Raw Point Cloud
    |
    v
Voxelization (divide space into 3D grid)
    |
    v
3D Sparse Convolutional Backbone (extract volumetric features)
    |
    v
BEV Feature Map (collapse Z-axis)
    |
    v
2D Backbone + Neck (refine BEV features, multi-scale)
    |
    v
Center Heatmap Head (detect object centers as Gaussian peaks)
    |
    v
Regression Heads (offset, height, size, rotation, velocity)
    |
    v
[Optional] Two-Stage Refinement (extract point features, MLP)
    |
    v
Tracking (greedy center-distance matching with velocity prediction)
```

---

## Voxelization

The input point cloud is discretized into a regular 3D grid of voxels. Each voxel aggregates the points falling within it:

- **nuScenes:** voxel size = [0.075, 0.075, 0.2] m, point cloud range = [-54, -54, -5, 54, 54, 3] m
- **Waymo:** voxel size = [0.1, 0.1, 0.15] m, point cloud range = [-75.2, -75.2, -2, 75.2, 75.2, 4] m

Within each voxel, points are encoded using mean VFE (Voxel Feature Encoding) or simple mean pooling of point coordinates and features (x, y, z, intensity, time_lag).

---

## 3D Sparse Convolutional Backbone

The voxelized representation is processed by a 3D sparse convolutional network:

- **Architecture:** 4 stages with increasing channel dimensions (e.g., 16 -> 32 -> 64 -> 128).
- **Operations:** Combination of Submanifold Sparse Convolution (SubMConv3d) that preserves sparsity patterns, and regular Sparse Convolution (SparseConv3d) with stride > 1 for downsampling.
- **Strides:** [1, 2, 4, 8] across the 4 stages (spatial dimensions halved at each strided stage).
- **Sparse tensors:** Only active (non-empty) voxels participate in computation, making the network efficient despite the large 3D volume.

The output is a sparse 3D feature volume at 8x downsampled resolution.

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

## Regression Heads

At each detected center location, separate regression heads predict:

| Head | Output | Description |
|------|--------|-------------|
| Sub-voxel offset | (dx, dy) | Compensates for quantization error from BEV discretization |
| Height | z_center | Absolute height of object center above ground |
| Size | (log_w, log_l, log_h) | Log-normalized 3D dimensions |
| Rotation | (sin_θ, cos_θ) | Yaw angle decomposed to avoid discontinuity |
| Velocity | (vx, vy) | BEV velocity for tracking (nuScenes) |

All regression heads share the BEV backbone features and use lightweight convolutional layers (typically 2 conv layers with 64 channels + final 1x1 projection).

---

## Two-Stage Refinement

CenterPoint optionally employs a second stage to refine first-stage detections:

1. **Feature extraction:** For each detected center, extract BEV features using bilinear interpolation at the predicted center location (and optionally at face centers of the predicted box).
2. **Point feature aggregation:** Concatenate features from the center point and face-center points.
3. **MLP refinement:** A small multi-layer perceptron predicts confidence score adjustments and box regression refinements (residuals to the first-stage predictions).
4. **Benefit:** The second stage improves localization accuracy, especially for large objects (trucks, buses) where the center feature alone may not capture the full extent.

---

## Integrated Tracking

CenterPoint introduces a simple yet effective tracking approach:

### Method: Greedy Closest-Point Matching

1. **Velocity prediction:** The detection head predicts per-object velocity (vx, vy) from multi-sweep input.
2. **Position extrapolation:** For each tracked object from the previous frame, predict its current position using: `predicted_center = previous_center + velocity * dt`
3. **Association:** Compute pairwise L2 distances between current detections and velocity-extrapolated previous tracks.
4. **Greedy matching:** Assign detections to tracks greedily (closest first) with a maximum distance threshold.
5. **Track management:**
   - Unmatched detections -> new tracks
   - Unmatched tracks -> increment age counter; delete if age > max_age
   - Matched pairs -> update track state

### Key Properties

- No appearance features or learned Re-ID required.
- No Hungarian algorithm needed (greedy matching suffices due to velocity prediction reducing ambiguity).
- Real-time performance with minimal computational overhead.
- Achieves state-of-the-art tracking performance on nuScenes.

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

---

## Comparison with Related Methods

| Method | Representation | Anchor-free | Tracking | NMS-free |
|--------|---------------|-------------|----------|----------|
| SECOND | Voxel + Sparse Conv | No | No | No |
| PointPillars | Pillar + 2D Conv | No | No | No |
| PartA2 | Voxel + Point | No | No | No |
| 3DSSD | Point-based | Yes | No | No |
| CenterPoint | Voxel + Center | Yes | Yes | Yes* |

*CenterPoint uses max-pooling for peak extraction instead of traditional NMS, though a light NMS can optionally be applied.

---

## Significance and Impact

1. **Unified detection and tracking:** First method to achieve competitive performance on both tasks with a single model.
2. **Simplicity:** Removes anchor hyperparameters, complex NMS tuning, and expensive tracking modules.
3. **Generality:** Works with both voxel-based and pillar-based representations.
4. **Foundation for future work:** CenterPoint's center-based paradigm influenced subsequent methods like CenterFormer, TransFusion, and BEVFusion.
