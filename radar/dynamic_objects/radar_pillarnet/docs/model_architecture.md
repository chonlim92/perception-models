# Model Architecture: RadarPillarNet

## 1. Overview

RadarPillarNet is a pillar-based 3D object detection network designed specifically for
automotive radar point clouds. The architecture processes sparse, multi-sweep radar data
and outputs 3D bounding boxes with velocity estimates. It extends the PointPillars
framework with radar-specific feature engineering and velocity regression.

## 2. Architecture Diagram

```
                          RadarPillarNet Architecture
 ============================================================================

 INPUT: Accumulated Radar Point Cloud (N x 9 features)
         [x, y, z, rcs, vx_comp, vy_comp, dt, xc, yc]
                              |
                              v
 +------------------------------------------------------------------+
 |                    PILLAR GENERATION                              |
 |  Grid: 512 x 512 pillars (0.2m x 0.2m resolution)              |
 |  Range: [-51.2, 51.2] x [-51.2, 51.2] x [-5.0, 3.0] m         |
 |  Max points per pillar: 32                                       |
 |  Max pillars: 16,000                                             |
 +------------------------------------------------------------------+
                              |
                              v
 +------------------------------------------------------------------+
 |                   PILLAR FEATURE NET (PointNet)                   |
 |  Input: (P, N, 9) -> Linear(9, 64) -> BN -> ReLU               |
 |  MaxPool over N points: (P, N, 64) -> (P, 64)                  |
 |  Output: P pillar features of dimension 64                       |
 +------------------------------------------------------------------+
                              |
                              v
 +------------------------------------------------------------------+
 |                   SCATTER TO PSEUDO-IMAGE                         |
 |  Place pillar features at grid coordinates                       |
 |  Output: (1, 64, 512, 512) BEV pseudo-image                    |
 +------------------------------------------------------------------+
                              |
                              v
 +------------------------------------------------------------------+
 |                    2D BACKBONE (SECOND-like)                      |
 |                                                                  |
 |  Block 1: stride=2, 3x Conv(64, 64, 3x3, s=2/1/1) + BN + ReLU |
 |  Block 2: stride=2, 5x Conv(64, 128, 3x3, s=2/1/1) + BN + ReLU|
 |  Block 3: stride=2, 5x Conv(128, 256, 3x3, s=2/1/1) + BN + ReLU|
 |                                                                  |
 |  Feature maps: (64, 256, 256), (128, 128, 128), (256, 64, 64)  |
 +------------------------------------------------------------------+
                              |
                              v
 +------------------------------------------------------------------+
 |              FEATURE PYRAMID NETWORK (FPN) / Neck                |
 |                                                                  |
 |  Upsample Block 1: ConvT(64, 128, 1x1, s=1)  -> (128, 256, 256)|
 |  Upsample Block 2: ConvT(128, 128, 2x2, s=2) -> (128, 256, 256)|
 |  Upsample Block 3: ConvT(256, 128, 4x4, s=4) -> (128, 256, 256)|
 |                                                                  |
 |  Concatenate: (384, 256, 256)                                   |
 +------------------------------------------------------------------+
                              |
                              v
 +------------------------------------------------------------------+
 |                    DETECTION HEAD                                 |
 |                                                                  |
 |  Shared Conv: Conv(384, 384, 3x3) + BN + ReLU                  |
 |                                                                  |
 |  Branch 1 - Classification: Conv(384, num_anchors * num_classes)|
 |  Branch 2 - Box Regression:  Conv(384, num_anchors * 7)         |
 |             [dx, dy, dz, dw, dl, dh, dyaw]                     |
 |  Branch 3 - Direction:       Conv(384, num_anchors * 2)         |
 |  Branch 4 - Velocity:        Conv(384, num_anchors * 2)         |
 |             [vx, vy]                                            |
 +------------------------------------------------------------------+
                              |
                              v
 OUTPUT: Detected 3D Bounding Boxes with Velocity
         [x, y, z, w, l, h, yaw, vx, vy, score, class]
```

## 3. Detailed Component Description

### 3.1 Multi-Sweep Accumulation with Ego-Motion Compensation

The input pipeline accumulates radar points from multiple sweeps:

```
Input Processing:
  1. Load current sweep from all 5 radar sensors
  2. Load previous (n_sweeps - 1) sweeps
  3. For each historical sweep:
     - Transform points to current ego frame using calibration + ego-pose
     - Compute relative timestamp: dt = (t_current - t_sweep) / max_time_lag
  4. Concatenate all points: N_total = sum(N_sweep_i)
  5. Apply range filter: keep points within detection grid

Typical dimensions:
  - Single sweep: ~200 points
  - After 6-sweep accumulation: ~1,200 points
  - After range filtering: ~1,000 points
```

### 3.2 Pillar Feature Encoding

The pillar encoding stage converts the irregular point cloud into a structured grid:

```
Pillar Generation Parameters:
  - Grid resolution: 0.2m x 0.2m (x, y)
  - Grid size: 512 x 512 pillars
  - Spatial range: x in [-51.2, 51.2], y in [-51.2, 51.2], z in [-5.0, 3.0]
  - Max points per pillar (N): 32
  - Max non-empty pillars (P): 16,000

Per-Point Features (9 dimensions):
  1. x      - Absolute X position in ego frame
  2. y      - Absolute Y position in ego frame
  3. z      - Absolute Z position in ego frame
  4. rcs    - Radar cross section (dBsm)
  5. vx_comp - Compensated velocity X (m/s)
  6. vy_comp - Compensated velocity Y (m/s)
  7. dt     - Relative timestamp [0, 1]
  8. xc     - X offset from pillar center (x - x_pillar_center)
  9. yc     - Y offset from pillar center (y - y_pillar_center)
```

### 3.3 PointNet Encoder

The simplified PointNet processes each pillar independently:

```
Architecture:
  Input:  (P, N, 9)    # P pillars, N points per pillar, 9 features
  Linear: (P, N, 9) -> (P, N, 64)
  BatchNorm1d: applied on feature dimension
  ReLU: activation
  MaxPool: over N dimension -> (P, 64)
  Output: (P, 64)      # One 64-dim feature vector per pillar

Notes:
  - Empty point slots (padding) are masked before max pooling
  - Single linear layer (unlike full PointNet with multiple layers)
  - This simplification works well for radar due to low point count per pillar
```

### 3.4 Scatter Operation

The scatter operation places pillar features onto the 2D grid:

```
Input:  (P, 64) pillar features + (P, 2) pillar grid indices
Output: (1, 64, 512, 512) pseudo-image

Operation:
  pseudo_image = zeros(1, 64, 512, 512)
  for each pillar i with grid index (xi, yi):
      pseudo_image[0, :, xi, yi] = pillar_features[i]

Note: This operation is implemented efficiently using scatter_nd/index_put
```

### 3.5 2D Backbone Network

The backbone extracts multi-scale features from the BEV pseudo-image:

```
Block 1 (stride 2):
  Conv2d(64, 64, 3x3, stride=2, pad=1) + BN + ReLU    # Downsample
  Conv2d(64, 64, 3x3, stride=1, pad=1) + BN + ReLU    # Process
  Conv2d(64, 64, 3x3, stride=1, pad=1) + BN + ReLU    # Process
  Output: (64, 256, 256)

Block 2 (stride 4):
  Conv2d(64, 128, 3x3, stride=2, pad=1) + BN + ReLU   # Downsample
  Conv2d(128, 128, 3x3, stride=1, pad=1) + BN + ReLU  # Process
  Conv2d(128, 128, 3x3, stride=1, pad=1) + BN + ReLU  # Process
  Conv2d(128, 128, 3x3, stride=1, pad=1) + BN + ReLU  # Process
  Conv2d(128, 128, 3x3, stride=1, pad=1) + BN + ReLU  # Process
  Output: (128, 128, 128)

Block 3 (stride 8):
  Conv2d(128, 256, 3x3, stride=2, pad=1) + BN + ReLU  # Downsample
  Conv2d(256, 256, 3x3, stride=1, pad=1) + BN + ReLU  # Process
  Conv2d(256, 256, 3x3, stride=1, pad=1) + BN + ReLU  # Process
  Conv2d(256, 256, 3x3, stride=1, pad=1) + BN + ReLU  # Process
  Conv2d(256, 256, 3x3, stride=1, pad=1) + BN + ReLU  # Process
  Output: (256, 64, 64)
```

### 3.6 Feature Pyramid Network (Neck)

The FPN upsamples and concatenates multi-scale features:

```
Upsample Block 1 (from Block 1 output):
  ConvTranspose2d(64, 128, 1x1, stride=1) + BN + ReLU
  Output: (128, 256, 256)

Upsample Block 2 (from Block 2 output):
  ConvTranspose2d(128, 128, 2x2, stride=2) + BN + ReLU
  Output: (128, 256, 256)

Upsample Block 3 (from Block 3 output):
  ConvTranspose2d(256, 128, 4x4, stride=4) + BN + ReLU
  Output: (128, 256, 256)

Concatenation:
  cat([block1_up, block2_up, block3_up], dim=1)
  Output: (384, 256, 256)
```

### 3.7 Anchor-Based Detection Head

The detection head predicts 3D bounding boxes using pre-defined anchors:

```
Anchor Configuration:
  Classes: [car, truck, bus, trailer, construction, pedestrian, motorcycle, bicycle,
            traffic_cone, barrier]
  Per-class anchors: 2 orientations (0 and pi/2 radians)
  Anchor sizes: class-specific (e.g., car: 4.6 x 1.9 x 1.7 m)
  Total anchors per location: 2 * num_classes = 20

Head Architecture:
  Shared: Conv2d(384, 384, 3x3, pad=1) + BN + ReLU

  Classification Branch:
    Conv2d(384, num_anchors * num_classes, 1x1)
    Output: (20 * 10, 256, 256) -> focal loss

  Box Regression Branch:
    Conv2d(384, num_anchors * 7, 1x1)
    Output: (20 * 7, 256, 256) -> smooth L1 loss
    Encodes: [dx, dy, dz, log(dw), log(dl), log(dh), sin(dyaw)]

  Direction Branch:
    Conv2d(384, num_anchors * 2, 1x1)
    Output: (20 * 2, 256, 256) -> cross-entropy loss
    Resolves heading ambiguity (forward vs. backward)

  Velocity Branch:
    Conv2d(384, num_anchors * 2, 1x1)
    Output: (20 * 2, 256, 256) -> smooth L1 loss
    Regresses: [vx, vy] in m/s
```

### 3.8 Box Encoding and Decoding

```
Encoding (target generation):
  dx = (x_gt - x_anchor) / diagonal_anchor
  dy = (y_gt - y_anchor) / diagonal_anchor
  dz = (z_gt - z_anchor) / h_anchor
  dw = log(w_gt / w_anchor)
  dl = log(l_gt / l_anchor)
  dh = log(h_gt / h_anchor)
  dyaw = sin(yaw_gt - yaw_anchor)

Decoding (inference):
  x = dx * diagonal_anchor + x_anchor
  y = dy * diagonal_anchor + y_anchor
  z = dz * h_anchor + z_anchor
  w = exp(dw) * w_anchor
  l = exp(dl) * l_anchor
  h = exp(dh) * h_anchor
  yaw = yaw_anchor + arcsin(dyaw)  # combined with direction classification
```

## 4. Model Dimensions Summary

| Component | Input Shape | Output Shape | Parameters |
|-----------|-------------|--------------|------------|
| Pillar PointNet | (P, 32, 9) | (P, 64) | ~640 |
| Scatter | (P, 64) | (1, 64, 512, 512) | 0 |
| Backbone Block 1 | (1, 64, 512, 512) | (1, 64, 256, 256) | ~111K |
| Backbone Block 2 | (1, 64, 256, 256) | (1, 128, 128, 128) | ~590K |
| Backbone Block 3 | (1, 128, 128, 128) | (1, 256, 64, 64) | ~2.36M |
| FPN Neck | Multi-scale | (1, 384, 256, 256) | ~280K |
| Detection Head | (1, 384, 256, 256) | Predictions | ~1.2M |
| **Total** | - | - | **~4.6M** |

## 5. Inference Pipeline

```
1. Load and preprocess radar point cloud (6-sweep accumulation)
2. Generate pillars and extract features
3. Apply PointNet encoder
4. Scatter to pseudo-image
5. Forward through backbone and FPN
6. Generate predictions from detection head
7. Apply NMS (IoU threshold = 0.2 for cars, 0.2 for other classes)
8. Score thresholding (score > 0.1)
9. Output final detections

Inference time (single NVIDIA V100):
  - Pillar generation + encoding: ~3 ms
  - Backbone + FPN: ~8 ms
  - Detection head + NMS: ~4 ms
  - Total: ~15 ms (~67 FPS)
```

## 6. Comparison with Standard PointPillars

| Feature | Standard PointPillars (LiDAR) | RadarPillarNet (Radar) |
|---------|-------------------------------|------------------------|
| Input features | 4 (x, y, z, intensity) | 9 (x, y, z, rcs, vx, vy, dt, xc, yc) |
| Typical input points | 30,000+ | ~1,200 |
| Max pillars | 12,000 | 16,000 |
| Points per pillar | 100 | 32 |
| Pillar resolution | 0.16m | 0.20m |
| Velocity head | No | Yes |
| Multi-sweep input | Optional | Required |
| Backbone channels | [64, 128, 256] | [64, 128, 256] |
