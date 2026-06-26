# PointPillars Model Architecture

This document provides a detailed, layer-by-layer walkthrough of the PointPillars
architecture for 3D object detection from LiDAR point clouds. It is written for readers
who are new to autonomous driving perception and assumes only basic familiarity with
neural networks (what a convolution is, what a linear layer does).

---

## 1. Prerequisites

Before diving into the architecture, this section defines the key concepts you need.

### 1.1 What Is a Point Cloud?

A point cloud is a collection of 3D points produced by a LiDAR sensor. Each point
has coordinates (x, y, z) in meters and an intensity value (how strongly the surface
reflected the laser). A typical automotive LiDAR produces 100,000-300,000 points per
scan at 10-20 Hz.

Key properties:
- **Unordered:** There is no inherent sequence; any permutation is the same cloud.
- **Irregular:** Points do not lie on a regular grid -- they cluster on surfaces.
- **Variable size:** Each scan has a different number of points.
- **Sparse in 3D:** Points exist only on visible surfaces; most of 3D space is empty.

### 1.2 What Is Bird's Eye View (BEV)?

Bird's Eye View is a top-down representation of the scene, looking straight down from
above. In BEV, each location (x, y) corresponds to a physical position on the ground
plane. The height (z) axis is either compressed or encoded within the features at each
(x, y) location.

```
Side View:                          Bird's Eye View:
                                    (looking straight down)
z ^
  |     [Car]                       +---+
  |   __|___|__                     |Car|  <-- x-y footprint only
  |  /  ground  \                   +---+
  +-------------- > x

BEV discards explicit height but preserves the spatial layout needed for driving.
```

BEV is the natural representation for autonomous driving because path planning operates
in 2D (the vehicle moves on the ground plane). Detecting objects in BEV directly tells
the planner where obstacles are.

### 1.3 What Is an Anchor-Based Detector?

An anchor-based detector places predefined bounding box templates ("anchors") at every
spatial location of the feature map. The network then:
1. **Classifies** each anchor: does it contain an object, and what class?
2. **Regresses** offsets: how much does the true box differ from the anchor?

This approach provides a strong geometric prior -- the network starts from a reasonable
box shape and only needs to make small adjustments, rather than predicting boxes from
scratch. PointPillars uses this anchor-based approach with predefined sizes for each
object class (Car, Pedestrian, Cyclist).

---

## 2. Architecture Overview

### 2.1 High-Level Data Flow

```
+-------------+     +-----------------+     +------------------+     +-----------+
| Point Cloud | --> | Pillar Creation | --> | Pillar Feature   | --> | Scatter   |
| (N, 4)      |     | (Discretize to  |     | Net (PointNet)   |     | to 2D     |
|             |     |  x-y grid)      |     | (MLP + MaxPool)  |     | Grid      |
+-------------+     +-----------------+     +------------------+     +-----------+
                                                                          |
                                                                          v
+-------------+     +---------+     +---------------+     +---------------------------+
| Detection   | <-- | Neck    | <-- | 2D Backbone   | <-- | Pseudo-Image              |
| Head (SSD)  |     | (FPN)   |     | (3 Conv Blocks)|    | (B, 64, 496, 432)        |
+-------------+     +---------+     +---------------+     +---------------------------+
      |
      v
+-------------------+     +---------+     +-------------------+
| Raw Predictions   | --> | Decode  | --> | Final 3D Boxes    |
| (cls, box, dir)   |     | + NMS   |     | (x,y,z,w,l,h,yaw)|
+-------------------+     +---------+     +-------------------+
```

### 2.2 End-to-End Pipeline Summary

```
Point Cloud (N x 4: x, y, z, intensity)
    |
    v
[Pillar Creation] -- assign each point to an x-y grid cell (pillar)
    |
    v
(B, max_pillars, max_points, 9) -- zero-padded tensor with augmented features
    |
    v
[Pillar Feature Net] -- Linear(9,64) + BN + ReLU + MaxPool across points
    |
    v
(B, max_pillars, 64) -- one 64-dim feature vector per non-empty pillar
    |
    v
[Scatter] -- place each pillar feature at its (x,y) grid position
    |
    v
(B, 64, 496, 432) -- dense 2D pseudo-image (like a 64-channel "photograph")
    |
    v
[2D Backbone] -- three conv blocks with stride-2 downsampling
    |
    v
(B, 64, 248, 216), (B, 128, 124, 108), (B, 256, 62, 54) -- multi-scale features
    |
    v
[Neck / FPN] -- upsample all to same resolution, concatenate
    |
    v
(B, 384, 248, 216) -- fused multi-scale feature map
    |
    v
[Detection Head] -- 1x1 convolutions for classification, box regression, direction
    |
    v
cls: (B, 18, 248, 216), box: (B, 42, 248, 216), dir: (B, 12, 248, 216)
    |
    v
[Post-Processing] -- score threshold, box decode, direction fix, rotated NMS
    |
    v
List of 3D detections: (x, y, z, w, l, h, yaw, class, score)
```

---

## 3. Pillar Creation

### 3.1 Defining the x-y Grid

The first step is to define a 2D grid over the x-y plane that covers the detection range.
For the KITTI dataset, the standard configuration is:

```
Physical detection range:
  x: [0.0, 69.12] meters     (forward direction from the ego vehicle)
  y: [-39.68, 39.68] meters  (left-right)
  z: [-3.0, 1.0] meters      (below to above the sensor)

Pillar size:
  dx = 0.16 meters (in x)
  dy = 0.16 meters (in y)
  dz = 4.0 meters  (full height -- this is the "infinite" pillar height)

Grid dimensions:
  W = (69.12 - 0.0) / 0.16 = 432 pillars in x
  H = (39.68 - (-39.68)) / 0.16 = 496 pillars in y
  Total grid: 496 x 432 = 214,272 possible pillar locations
```

### 3.2 Point Assignment

Each LiDAR point is assigned to a pillar based solely on its x and y coordinates:

```python
# For a point with coordinates (px, py, pz):
pillar_x_index = floor((px - x_min) / pillar_size_x)  # which column
pillar_y_index = floor((py - y_min) / pillar_size_y)  # which row

# Points outside the detection range are discarded
if pillar_x_index < 0 or pillar_x_index >= W: discard
if pillar_y_index < 0 or pillar_y_index >= H: discard
```

The z-coordinate does NOT affect pillar assignment -- all points in the same (x, y) cell
go to the same pillar regardless of their height.

### 3.3 Random Sampling and Zero-Padding

To create a fixed-size tensor for GPU processing, two constraints are applied:

**Maximum points per pillar (N = 32 or 100):** If a pillar contains more than N points,
N points are randomly sampled. If fewer, the pillar is zero-padded to N points.

**Maximum number of non-empty pillars (P = 12,000 or 16,000):** If more than P pillars
contain at least one point, P pillars are randomly selected. In practice, a typical
KITTI scan has 6,000-10,000 non-empty pillars, so this limit rarely triggers.

```
Top-down view of pillar grid (simplified 8x8 for illustration):

    y
    ^
    |  +--+--+--+--+--+--+--+--+
    |  |  | 3|  |  |  |  |  |  |   Numbers show point count per pillar
    |  +--+--+--+--+--+--+--+--+
    |  |  |  |  | 7|12|  |  |  |   Pillar (3,4) has 7 points
    |  +--+--+--+--+--+--+--+--+   Pillar (4,4) has 12 points
    |  |  |  |  |  | 5|  |  |  |     -> if max_points=10, randomly sample 10
    |  +--+--+--+--+--+--+--+--+
    |  |  |  |45|22| 1|  |  |  |   Pillar (2,5) has 45 points (dense, near sensor)
    |  +--+--+--+--+--+--+--+--+     -> if max_points=32, randomly sample 32
    |  |  |  |  |  |  |  |  |  |
    |  +--+--+--+--+--+--+--+--+   Empty pillars: not processed at all
    |  |  |  |  |  |  |  | 2|  |   Pillar (6,2) has 2 points
    |  +--+--+--+--+--+--+--+--+     -> zero-pad to max_points
    +-------------------------------> x

After processing:
  - Non-empty pillars are collected (say K = 8 in this example)
  - Each pillar becomes a (max_points, 9) tensor
  - Result: (K, max_points, 9) -- or (P, max_points, 9) with zero-padding for K < P
```

### 3.4 Why Random Sampling?

Random sampling serves two purposes:
1. **Fixed tensor size:** GPUs work best with fixed-size tensors; variable sizes require
   complex padding/masking logic.
2. **Regularization:** Different random samples across training epochs expose the network
   to different subsets of points in dense pillars, acting as a form of data augmentation.

---

## 4. Point Feature Augmentation

### 4.1 The 9 Input Features

Before encoding, each point is augmented from 4 raw features to 9 features. This
augmentation provides the network with both absolute position and relative position
information:

| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| x | raw | Absolute x-coordinate in LiDAR frame (meters) |
| y | raw | Absolute y-coordinate in LiDAR frame (meters) |
| z | raw | Absolute z-coordinate (height, meters) |
| intensity | raw | Reflectance of the surface (0 to 1, or 0 to 255) |
| xc | x - x_mean | Offset from arithmetic mean x of all points in this pillar |
| yc | y - y_mean | Offset from arithmetic mean y of all points in this pillar |
| zc | z - z_mean | Offset from arithmetic mean z of all points in this pillar |
| xp | x - x_center | Offset from the geometric center of the pillar cell in x |
| yp | y - y_center | Offset from the geometric center of the pillar cell in y |

### 4.2 Understanding Each Feature Group

**Raw features (x, y, z, intensity):**
These give the absolute position and surface property of each point. The network can
learn that points at z = -1.5m are likely ground, while points at z = 0m to z = 1.5m
are likely vehicle surfaces.

**Pillar mean offsets (xc, yc, zc):**
These capture the distribution of points WITHIN the pillar relative to their centroid.
For a car, points near the top of the car have positive zc (above the mean), while
points on the lower body have negative zc. This helps the network understand the local
3D shape without knowing absolute coordinates.

```
Example: Two pillars with the same number of points but different height distributions

Pillar A (flat ground):         Pillar B (car roof + ground):
z  |                            z  |   . . .  (roof points, zc > 0)
   |                               |
   | . . . . . . (all at z=-1.5)   |
   |  zc = 0 for all points        |   . . .  (ground points, zc < 0)
                                    |
The zc feature distinguishes these two cases even though both pillars
have similar point counts.
```

**Pillar center offsets (xp, yp):**
These indicate where within the 0.16m x 0.16m pillar cell each point falls. A point
exactly at the cell center has xp = yp = 0. A point at the left edge has yp close to
-0.08m. This fine-grained position within the cell preserves sub-pillar spatial detail
that would otherwise be lost by the discretization.

### 4.3 Why These Features Help

The combination of absolute and relative features serves complementary purposes:

- **Absolute (x, y, z):** Allows the network to learn position-dependent patterns
  (e.g., the ground is always at a certain z, objects at far range are always sparse).
- **Mean offsets (xc, yc, zc):** Provides translation-invariant local shape information
  (the internal structure of the pillar's point distribution).
- **Center offsets (xp, yp):** Preserves fine spatial detail within each cell, compensating
  for the quantization introduced by the pillar grid.

Together, these 9 features give the PointNet encoder everything it needs to produce a
rich pillar-level representation from the raw point measurements.

---

## 5. PointNet per Pillar

### 5.1 Background: What Is PointNet?

PointNet (Qi et al., CVPR 2017) is a neural network architecture designed specifically
for processing unordered point sets. Its key insight is that a symmetric function
(like max pooling) applied to per-point features produces an output that is invariant
to point ordering.

In PointPillars, a simplified version of PointNet is applied independently to each
pillar. "Simplified" means: a single shared linear layer (instead of multiple), followed
by batch normalization, ReLU activation, and max pooling.

### 5.2 Shared MLP (Linear Layer)

The term "shared MLP" means the same linear transformation is applied to every point
independently (no interaction between points at this stage):

```
For each point in the pillar:
  output_features = Linear(input_features)

  input_features:  9-dimensional vector (x, y, z, i, xc, yc, zc, xp, yp)
  output_features: 64-dimensional vector

The weight matrix W is shape (9, 64) and bias b is shape (64,).
The SAME W and b are used for every point in every pillar.
```

This is equivalent to a 1D convolution with kernel size 1 applied across the points
dimension. It transforms each point's 9 features into a richer 64-dimensional
representation, projecting from the input feature space into a learned embedding space
where max pooling can effectively aggregate information.

### 5.3 Batch Normalization and ReLU

After the linear layer:
- **BatchNorm** normalizes the 64-dimensional features across the batch, stabilizing
  training and enabling higher learning rates.
- **ReLU** (Rectified Linear Unit) introduces non-linearity: negative values become zero,
  positive values pass through unchanged. This allows the network to learn non-linear
  relationships between point features.

### 5.4 Max Pooling Across Points

The critical step that makes PointNet work for unordered sets:

```
Input to max pool: (max_points, 64) -- one 64-dim vector per point
                                        within this pillar

Max pool operation: For each of the 64 feature dimensions,
                    take the MAXIMUM value across all points.

Output: (64,) -- a single 64-dim vector representing the entire pillar
```

Why max pooling works:
1. **Order invariance:** max(a, b, c) = max(c, a, b). The output does not depend on
   the order of points, which is essential since point clouds are unordered.
2. **Variable size handling:** Max pooling works regardless of how many points
   contribute (whether 1 point or 100 points, the output is always 64-dimensional).
3. **Information selection:** For each feature dimension, the max operation selects the
   most "activated" point -- effectively learning to attend to the most informative
   point in each pillar for each feature channel.

### 5.5 Tensor Shapes at Each Step

```
Stage                              | Shape                     | Explanation
-----------------------------------|---------------------------|---------------------------
Input point cloud                  | (N, 4)                   | N varies per scan
After pillarization + augmentation | (B, P, N_pts, 9)         | B=batch, P=max_pillars
                                   |                           | N_pts=max_points_per_pillar
After Linear(9, 64)                | (B, P, N_pts, 64)        | Each point: 9 -> 64 features
After BatchNorm                    | (B, P, N_pts, 64)        | Same shape, normalized
After ReLU                         | (B, P, N_pts, 64)        | Same shape, negatives zeroed
After MaxPool(dim=2)               | (B, P, 64)              | One vector per pillar
```

With typical KITTI values: B=4, P=12000, N_pts=32
- Input to PointNet: (4, 12000, 32, 9)
- Output of PointNet: (4, 12000, 64)

### 5.6 Why Not a Deeper PointNet?

The original PointPillars paper found that a single linear layer before max pooling
provides the best speed-accuracy trade-off. Adding a second layer improves accuracy by
only ~0.4 AP while increasing latency by ~3ms. The pillar feature network is not the
computational bottleneck (it takes only 0.5ms), but keeping it simple reduces overall
model complexity and parameter count.

---

## 6. Scatter Operation

### 6.1 Purpose: Bridge from Sparse to Dense

After the PointNet, we have feature vectors for each non-empty pillar, but they are
stored in a flat list with associated (x, y) coordinates. The 2D CNN backbone requires
a dense, regularly-structured 2D tensor (like an image). The scatter operation bridges
this gap.

### 6.2 How It Works

Each pillar's 64-dimensional feature vector is placed at its corresponding position
in a 2D grid:

```
Pillar features: (B, P, 64)          Pillar coordinates: (B, P, 2)
  |                                    |
  | For each pillar i:                 |
  |   feature_vector = pillar_features[i]   (64-dim)
  |   (xi, yi) = pillar_coords[i]          (grid indices)
  |   pseudo_image[:, :, xi, yi] = feature_vector
  |
  v
Pseudo-Image: (B, 64, H, W) = (B, 64, 496, 432)
```

### 6.3 Implementation

```python
# Scatter operation (simplified)
pseudo_image = torch.zeros(B, 64, 496, 432, device=device)

for batch_idx in range(B):
    for pillar_idx in range(num_valid_pillars[batch_idx]):
        x_idx = pillar_coords[batch_idx, pillar_idx, 0]
        y_idx = pillar_coords[batch_idx, pillar_idx, 1]
        pseudo_image[batch_idx, :, x_idx, y_idx] = \
            pillar_features[batch_idx, pillar_idx, :]
```

In practice, this is implemented as a single scatter operation (not a loop) using
tensor indexing for GPU efficiency.

### 6.4 ASCII Diagram

```
Before Scatter:                       After Scatter:
(List of pillar features)             (Dense 2D pseudo-image)

Pillar 0: [f0_0, f0_1, ..., f0_63]      +--+--+--+--+--+--+--+--+
  at grid position (2, 3)                |  |  |  |  |  |  |  |  |
                                         +--+--+--+--+--+--+--+--+
Pillar 1: [f1_0, f1_1, ..., f1_63]      |  |  |  |P0|  |  |  |  |  <- (2,3)
  at grid position (5, 1)                +--+--+--+--+--+--+--+--+
                                         |  |P1|  |  |  |  |  |  |  <- (5,1)
Pillar 2: [f2_0, f2_1, ..., f2_63]      +--+--+--+--+--+--+--+--+
  at grid position (4, 6)                |  |  |  |  |  |  |P2|  |  <- (4,6)
                                         +--+--+--+--+--+--+--+--+
...                                      |  |  |  |  |  |  |  |  |
                                         +--+--+--+--+--+--+--+--+

Empty cells contain all zeros (64 zeros).
The pseudo-image is SPARSE -- most cells are zero.
But the 2D backbone handles this transparently.
```

### 6.5 Key Properties

1. **No learnable parameters.** Scatter is a purely geometric operation -- it simply
   places data at the correct coordinates. There is nothing to train.
2. **Empty cells remain zero.** Cells with no corresponding pillar feature contain
   all zeros. Since most of the 496x432 grid is empty (only ~12,000 of 214,272 cells
   are occupied), the pseudo-image is very sparse.
3. **Reversible mapping.** The mapping from physical coordinates to grid indices is
   deterministic and invertible: given a grid index, you can compute the physical
   (x, y) location it represents.
4. **Efficient on GPU.** Despite the sparsity, the dense tensor format allows the
   subsequent 2D convolutions to use highly optimized CUDA kernels without any special
   sparse processing logic.

---

## 7. 2D CNN Backbone

### 7.1 Purpose

The backbone extracts hierarchical features from the pseudo-image at multiple spatial
scales. Each successive block doubles the receptive field while compressing spatial
dimensions, allowing the network to detect objects of different sizes.

### 7.2 Why Multi-Scale Matters

In the BEV pseudo-image:
- A **car** occupies approximately 3.9m / 0.16m = ~24 pixels in length
- A **pedestrian** occupies approximately 0.8m / 0.16m = ~5 pixels in length

Detecting both requires features at different spatial granularities:
- Fine-scale features capture small objects (pedestrians, cyclists)
- Coarse-scale features capture large objects (cars, trucks) and context

### 7.3 Architecture: Three Blocks

```
Input: Pseudo-Image (B, 64, 496, 432)
         |
         v
+--------------------------------------------+
| BLOCK 1                                    |
|   Conv2d(64, 64, k=3, s=2, p=1) + BN + ReLU  (downsample 2x)
|   Conv2d(64, 64, k=3, s=1, p=1) + BN + ReLU  (refine)
|   Conv2d(64, 64, k=3, s=1, p=1) + BN + ReLU  (refine)
|   Conv2d(64, 64, k=3, s=1, p=1) + BN + ReLU  (refine)
+--------------------------------------------+
         |
Output: (B, 64, 248, 216)   [spatial dimensions halved]
         |
         v
+--------------------------------------------+
| BLOCK 2                                    |
|   Conv2d(64, 128, k=3, s=2, p=1) + BN + ReLU  (downsample 2x)
|   Conv2d(128, 128, k=3, s=1, p=1) + BN + ReLU  (refine)
|   Conv2d(128, 128, k=3, s=1, p=1) + BN + ReLU  (refine)
|   Conv2d(128, 128, k=3, s=1, p=1) + BN + ReLU  (refine)
|   Conv2d(128, 128, k=3, s=1, p=1) + BN + ReLU  (refine)
|   Conv2d(128, 128, k=3, s=1, p=1) + BN + ReLU  (refine)
+--------------------------------------------+
         |
Output: (B, 128, 124, 108)  [spatial dimensions halved again]
         |
         v
+--------------------------------------------+
| BLOCK 3                                    |
|   Conv2d(128, 256, k=3, s=2, p=1) + BN + ReLU  (downsample 2x)
|   Conv2d(256, 256, k=3, s=1, p=1) + BN + ReLU  (refine)
|   Conv2d(256, 256, k=3, s=1, p=1) + BN + ReLU  (refine)
|   Conv2d(256, 256, k=3, s=1, p=1) + BN + ReLU  (refine)
|   Conv2d(256, 256, k=3, s=1, p=1) + BN + ReLU  (refine)
|   Conv2d(256, 256, k=3, s=1, p=1) + BN + ReLU  (refine)
+--------------------------------------------+
         |
Output: (B, 256, 62, 54)    [spatial dimensions halved again]
```

### 7.4 Block Structure Details

| Block | Input Shape | Output Shape | Channels | Stride-1 Layers | Total Convs |
|-------|-------------|--------------|----------|-----------------|-------------|
| 1 | (B, 64, 496, 432) | (B, 64, 248, 216) | 64 | 3 | 4 |
| 2 | (B, 64, 248, 216) | (B, 128, 124, 108) | 128 | 5 | 6 |
| 3 | (B, 128, 124, 108) | (B, 256, 62, 54) | 256 | 5 | 6 |

Each block begins with a **stride-2 convolution** that halves spatial dimensions and
(optionally) increases the channel count. This is followed by several **stride-1
convolutions** that maintain spatial dimensions while refining features.

### 7.5 Receptive Field Growth

After each block, the effective receptive field doubles:

| After Block | Feature Map Resolution | Physical Resolution per Cell | Receptive Field |
|-------------|----------------------|------------------------------|-----------------|
| Input | 496 x 432 | 0.16m | - |
| Block 1 | 248 x 216 | 0.32m | ~1.3m |
| Block 2 | 124 x 108 | 0.64m | ~3.8m |
| Block 3 | 62 x 54 | 1.28m | ~9.5m |

Block 3 features have a receptive field of ~9.5m, which is larger than most cars (~4m),
meaning each Block 3 cell "sees" an entire car. Block 1 features have finer resolution
better suited for small objects.

---

## 8. Neck / Feature Pyramid Network (FPN)

### 8.1 Purpose

The backbone produces three feature maps at different scales. The neck brings them all
to the same spatial resolution and concatenates them, creating a single feature map that
contains both fine-grained and coarse-grained information.

### 8.2 Upsampling with Transposed Convolutions

Each backbone output is upsampled to the resolution of Block 1 output (248 x 216)
using transposed convolutions (also called deconvolutions):

```
Block 1 output:  (B, 64, 248, 216)   ---> ConvTranspose2d(64, 128, k=1, s=1)
                                            --> (B, 128, 248, 216)   [no spatial change]

Block 2 output:  (B, 128, 124, 108)  ---> ConvTranspose2d(128, 128, k=4, s=2, p=1)
                                            --> (B, 128, 248, 216)   [2x upsample]

Block 3 output:  (B, 256, 62, 54)    ---> ConvTranspose2d(256, 128, k=8, s=4, p=2)
                                            --> (B, 128, 248, 216)   [4x upsample]
```

Each transposed convolution is followed by BatchNorm and ReLU.

### 8.3 Concatenation

All three upsampled feature maps are concatenated along the channel dimension:

```
Upsampled Block 1: (B, 128, 248, 216)
Upsampled Block 2: (B, 128, 248, 216)
Upsampled Block 3: (B, 128, 248, 216)
         |
         | Concatenate along channel dimension (dim=1)
         v
Fused output: (B, 384, 248, 216)    [128 + 128 + 128 = 384 channels]
```

### 8.4 Why Transposed Convolution for Upsampling?

Transposed convolutions (deconvolutions) are preferred over simpler upsampling methods
(bilinear interpolation) because they are **learnable**. The network can learn how to
best reconstruct fine-grained spatial information from coarse features, rather than
using a fixed interpolation formula.

### 8.5 Why Multi-Scale Fusion Matters

Without the FPN, the detection head would only have Block 3 features (62 x 54 resolution,
1.28m per cell). A pedestrian occupying only 0.8m would be less than one cell wide --
impossible to localize accurately. By fusing with Block 1 features (0.32m per cell), the
pedestrian spans ~2.5 cells, which is sufficient for accurate detection.

Conversely, detecting large trucks benefits from the large receptive field of Block 3,
which captures the full extent of the object in context.

---

## 9. Anchor-Based Detection Head

### 9.1 Anchor Definition

Anchors are predefined 3D bounding boxes placed at every spatial location of the fused
feature map. Their sizes are determined by dataset statistics (the average dimensions
of each object class in the training set):

| Class | Length (m) | Width (m) | Height (m) | Z-center (m) |
|-------|:----------:|:---------:|:----------:|:------------:|
| Car | 3.9 | 1.6 | 1.56 | -1.0 |
| Pedestrian | 0.8 | 0.6 | 1.73 | -0.6 |
| Cyclist | 1.76 | 0.6 | 1.73 | -0.6 |

At each of the 248 x 216 = 53,568 spatial locations, 6 anchors are placed:
- 3 classes x 2 rotations (0 degrees and 90 degrees) = 6 anchors per location
- Total anchors: 53,568 x 6 = 321,408 anchors per sample

### 9.2 Per-Anchor Predictions

For each of the 321,408 anchors, the detection head predicts three things:

| Output | Dimensions per Anchor | Description |
|--------|:---------------------:|-------------|
| Classification | 3 | Score for each class (Car, Ped, Cyc) |
| Box Regression | 7 | Residuals: (dx, dy, dz, dw, dl, dh, d_theta) |
| Direction | 2 | Binary classification: heading in [0,pi) or [pi,2pi) |

### 9.3 Implementation as 1x1 Convolutions

The head uses 1x1 convolutions applied to the fused feature map:

```python
# Classification branch
cls_conv = Conv2d(384, num_anchors * num_classes, kernel_size=1)
# = Conv2d(384, 6 * 3, 1) = Conv2d(384, 18, 1)
# Output: (B, 18, 248, 216)

# Box regression branch
box_conv = Conv2d(384, num_anchors * 7, kernel_size=1)
# = Conv2d(384, 6 * 7, 1) = Conv2d(384, 42, 1)
# Output: (B, 42, 248, 216)

# Direction classification branch
dir_conv = Conv2d(384, num_anchors * 2, kernel_size=1)
# = Conv2d(384, 6 * 2, 1) = Conv2d(384, 12, 1)
# Output: (B, 12, 248, 216)
```

### 9.4 IoU-Based Target Assignment (Training Only)

During training, each anchor must be labeled as positive, negative, or ignored:

1. Compute BEV IoU between every anchor and every ground-truth box
2. Assign labels:
   - **Positive:** IoU >= positive threshold (0.6 for Car, 0.5 for Ped/Cyc)
   - **Negative:** IoU < negative threshold (0.45 for Car, 0.35 for Ped/Cyc)
   - **Ignored:** IoU between thresholds (excluded from loss)
3. Additionally: the highest-IoU anchor for each GT box is always positive

```
Anchor-GT Assignment Example:

GT Box (Car): located at (30.0, 5.2) meters, heading 10 degrees

Nearby anchors:
  Anchor A: Car template, 0-deg rotation, IoU=0.72 --> POSITIVE (>0.6)
  Anchor B: Car template, 90-deg rotation, IoU=0.15 --> NEGATIVE (<0.45)
  Anchor C: Ped template, 0-deg rotation, IoU=0.08 --> NEGATIVE (<0.35)
  Anchor D: Car template, 0-deg rotation, IoU=0.52 --> IGNORED (between thresholds)
```

### 9.5 Loss Functions

The total training loss combines three components:

```
Total Loss = (1/N_pos) * [L_cls + beta_reg * L_reg + beta_dir * L_dir]

where N_pos = number of positive anchors

L_cls = Focal Loss (classification)
      = -alpha * (1 - p_t)^gamma * log(p_t)
      where alpha = 0.25, gamma = 2.0

      Purpose: Handles extreme class imbalance (vast majority of anchors
      are negative). Down-weights easy negatives, focuses on hard examples.

L_reg = Smooth L1 Loss (box regression, positive anchors only)
      = 0.5 * x^2           if |x| < 1
        |x| - 0.5           otherwise

      Purpose: Robust to outliers (large errors contribute linearly,
      not quadratically), stable gradients for small errors.

L_dir = Binary Cross-Entropy (direction classification, positive anchors only)
      = -(t * log(p) + (1-t) * log(1-p))

      Purpose: Resolves the 180-degree heading ambiguity.
```

### 9.6 Direction Classification: Resolving Heading Ambiguity

The box regression head predicts the heading angle (yaw) as a residual relative to the
anchor's rotation. However, a fundamental ambiguity exists: a box rotated by 0 degrees
and the same box rotated by 180 degrees have identical BEV IoU with many anchors. The
regression head alone cannot distinguish between these two orientations.

The direction classification head provides a binary signal:
- Bin 0: heading is in [0, pi) -- pointing "forward"
- Bin 1: heading is in [pi, 2*pi) -- pointing "backward"

During inference, if the direction classification disagrees with the regressed angle,
the heading is flipped by pi radians.

```
Heading Ambiguity Example:

    A car heading East (0 degrees):      A car heading West (180 degrees):
    +--------->                           <---------+
    |  Car    |                           |  Car    |
    +---------+                           +---------+

    Both have the SAME BEV rectangle!
    Only the direction classification distinguishes them.
```

---

## 10. NMS Post-Processing

### 10.1 Score Thresholding

First, raw classification logits are converted to probabilities via sigmoid, and
low-confidence predictions are removed:

```python
scores = sigmoid(cls_logits)  # Convert to [0, 1]
mask = scores > score_threshold  # e.g., threshold = 0.1
# Keep only predictions above threshold
# Typically retains top-K (e.g., 500) per class for efficiency
```

### 10.2 Box Decoding from Residuals

Predicted residuals are converted to absolute box parameters using anchor geometry:

```
Given: anchor parameters (x_a, y_a, z_a, w_a, l_a, h_a, theta_a)
       predicted residuals (dx, dy, dz, dw, dl, dh, d_theta)
       diagonal d_a = sqrt(l_a^2 + w_a^2)

Decoded box:
  x     = dx * d_a + x_a           (position normalized by anchor diagonal)
  y     = dy * d_a + y_a           (position normalized by anchor diagonal)
  z     = dz * h_a + z_a           (height normalized by anchor height)
  w     = exp(dw) * w_a            (exponential ensures positive width)
  l     = exp(dl) * l_a            (exponential ensures positive length)
  h     = exp(dh) * h_a            (exponential ensures positive height)
  theta = d_theta + theta_a        (additive angle residual)
```

The diagonal normalization for x and y ensures that position residuals are scale-invariant
(a residual of 1.0 means "one diagonal length away from the anchor center"). The
exponential encoding for dimensions ensures they remain positive.

### 10.3 Rotated IoU in BEV

For NMS, IoU is computed between oriented (rotated) rectangles in the bird's eye view:

```
Two rotated boxes in BEV:

       +-------+
      /       /        Box A (rotated by angle theta_A)
     /       /
    +-------+
                  +------+
                 /      /   Box B (rotated by angle theta_B)
                /      /
               +------+

IoU = Intersection_Area / Union_Area

Computing intersection of two rotated rectangles:
1. Find the vertices of both rectangles (4 corners each)
2. Use polygon clipping (Sutherland-Hodgman algorithm) to find intersection polygon
3. Compute area of intersection polygon
4. Union = Area_A + Area_B - Intersection
5. IoU = Intersection / Union
```

### 10.4 Per-Class NMS

NMS is applied independently for each class:

```python
for each class c:
    # Get all detections of class c, sorted by score (descending)
    dets = get_detections(class=c, sorted_by_score=True)

    keep = []
    while dets is not empty:
        # Keep the highest-scoring detection
        best = dets[0]
        keep.append(best)

        # Remove all detections that overlap too much with 'best'
        ious = rotated_bev_iou(best, dets[1:])
        dets = dets[1:][ious < nms_threshold]

    results[c] = keep
```

Typical NMS IoU thresholds:
- Car: 0.2 (strict, because cars are large and well-separated)
- Pedestrian: 0.1 (very strict, pedestrians can be close together)
- Cyclist: 0.1

---

## 11. Complete Tensor Flow Table

| Stage | Tensor Shape | Notes |
|-------|-------------|-------|
| Raw point cloud | (N, 4) | N varies per scan (~100K) |
| After pillarization + augmentation | (B, 12000, 32, 9) | Fixed size, zero-padded |
| After Linear(9, 64) | (B, 12000, 32, 64) | Shared weights across all points |
| After BatchNorm + ReLU | (B, 12000, 32, 64) | Normalized, non-linear |
| After MaxPool (dim=2) | (B, 12000, 64) | One vector per pillar |
| After Scatter | (B, 64, 496, 432) | Dense BEV pseudo-image |
| After Backbone Block 1 | (B, 64, 248, 216) | 2x downsample, same channels |
| After Backbone Block 2 | (B, 128, 124, 108) | 4x downsample total |
| After Backbone Block 3 | (B, 256, 62, 54) | 8x downsample total |
| After Neck upsample 1 | (B, 128, 248, 216) | From Block 1, stride-1 deconv |
| After Neck upsample 2 | (B, 128, 248, 216) | From Block 2, stride-2 deconv |
| After Neck upsample 3 | (B, 128, 248, 216) | From Block 3, stride-4 deconv |
| After Neck concatenation | (B, 384, 248, 216) | 128*3 = 384 channels |
| Detection Head - cls | (B, 18, 248, 216) | 6 anchors * 3 classes |
| Detection Head - reg | (B, 42, 248, 216) | 6 anchors * 7 box params |
| Detection Head - dir | (B, 12, 248, 216) | 6 anchors * 2 direction bins |
| After score threshold | (B, K, 9) | K = variable, per-class filtering |
| After NMS | List[(M, 9)] | M detections: 7 box + class + score |

---

## 12. Computational Analysis

### 12.1 Where Is the Bottleneck?

| Component | Latency (ms) | % of Total | FLOPs (approximate) |
|-----------|:------------:|:----------:|:-------------------:|
| Pillar Feature Net | 0.5 | 10% | ~0.1 GFLOPs |
| Scatter | 0.1 | 2% | Negligible (index ops) |
| 2D Backbone | 3.2 | 64% | ~12 GFLOPs |
| Neck (FPN) | 0.5 | 10% | ~2 GFLOPs |
| Detection Head | 0.7 | 14% | ~1 GFLOPs |
| NMS | 0.5 | 10% | CPU-bound |
| **Total** | **~5.5** | **100%** | **~15 GFLOPs** |

The **2D backbone dominates** (64% of inference time). This is because:
1. It processes the full 248x216 feature map through 16 convolutional layers
2. Each layer involves millions of multiply-accumulate operations
3. The feature map is dense (even though many cells are zero, the convolution still
   touches every cell)

### 12.2 Why the Encoder Is NOT the Bottleneck

Despite processing up to 12,000 pillars with 32 points each, the Pillar Feature Net
takes only 0.5ms because:
- The operation is a single matrix multiplication: (12000*32, 9) x (9, 64)
- Followed by a simple max reduction along one dimension
- These are embarrassingly parallel operations that GPUs execute very efficiently
- The total parameter count is tiny: 9*64 + 64 = 640 parameters

### 12.3 Optimization Opportunities

1. **Backbone pruning:** Since the backbone dominates, reducing its width (fewer channels)
   or depth (fewer layers) gives the most speedup per accuracy point sacrificed.
2. **TensorRT optimization:** The entire model uses standard operations, making it fully
   compatible with TensorRT for inference optimization (INT8 quantization can provide
   an additional 2-3x speedup).
3. **Sparse backbone:** Since the pseudo-image is ~95% zeros, a sparse 2D backbone could
   theoretically skip computations on zero-valued regions. In practice, the overhead of
   sparse indexing often negates the savings for 2D operations.
4. **Smaller input resolution:** Using larger pillars (0.32m instead of 0.16m) reduces
   the pseudo-image to 248x216, quartering the backbone computation at some accuracy cost.

### 12.4 Memory Footprint

| Component | Memory (per sample) | Notes |
|-----------|:-------------------:|-------|
| Input tensor | 12000 * 32 * 9 * 4 bytes = ~14 MB | Float32 |
| Pseudo-image | 64 * 496 * 432 * 4 bytes = ~55 MB | Float32, mostly zeros |
| Backbone features | ~45 MB total | Three feature maps |
| Neck output | 384 * 248 * 216 * 4 bytes = ~80 MB | Float32 |
| Head outputs | ~30 MB total | cls + reg + dir |
| **Peak memory** | **~200-250 MB per sample** | During forward pass |

With batch size 4 and gradients (training), total GPU memory usage is approximately
4-6 GB, fitting comfortably on modern GPUs (16-24 GB).

---

## References

- Lang, A. H., et al. "PointPillars: Fast Encoders for Object Detection from Point Clouds." CVPR 2019.
- Qi, C. R., et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation." CVPR 2017.
- Implementation references: OpenPCDet, MMDetection3D
