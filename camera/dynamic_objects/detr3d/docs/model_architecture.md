# Model Architecture: DETR3D

A tutorial-style guide to DETR3D's architecture for readers new to autonomous driving
perception and transformer-based 3D object detection.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Backbone: ResNet-101 + FPN](#backbone-resnet-101--fpn)
4. [3D Object Queries and Reference Points](#3d-object-queries-and-reference-points)
5. [3D-to-2D Projection and Feature Sampling](#3d-to-2d-projection-and-feature-sampling)
6. [Transformer Decoder](#transformer-decoder)
7. [Detection Head](#detection-head)
8. [Inference Pipeline](#inference-pipeline)

---

## Prerequisites

Before diving into DETR3D, you should be comfortable with two foundational topics:
camera projection and transformer decoders. This section provides the minimum
background needed.

### Camera Projection Basics

A camera converts 3D world points into 2D image pixels through a two-stage
process: an extrinsic transformation (world-to-camera) followed by an intrinsic
projection (camera-to-pixel).

**Extrinsic Matrix [R|t] (4x4):**

The extrinsic matrix describes where the camera is located and how it is oriented
in the world. It consists of a 3x3 rotation matrix R and a 3x1 translation vector t.
Together, they form a 4x4 homogeneous transformation matrix:

```
            [ r11  r12  r13  tx ]
[R|t] =    [ r21  r22  r23  ty ]
            [ r31  r32  r33  tz ]
            [  0    0    0    1 ]
```

This matrix transforms a point from the world (or ego-vehicle) coordinate system
into the camera's local coordinate system. In the camera frame:
- Z-axis points forward (out of the lens)
- X-axis points right
- Y-axis points down

**Intrinsic Matrix K (3x3):**

The intrinsic matrix encodes the camera's internal optical properties:

```
        [ fx   0   cx ]
K =     [  0  fy   cy ]
        [  0   0    1 ]
```

Where:
- fx, fy: focal lengths in pixels (how strongly the lens converges light)
- cx, cy: principal point (where the optical axis hits the image sensor, usually
  near the image center)

**Full Projection Pipeline:**

Given a 3D point P_world = [X, Y, Z, 1]^T in world coordinates:

```
Step 1:  P_cam = [R|t] * P_world         --> [X_c, Y_c, Z_c]^T in camera frame
Step 2:  p_img = K * P_cam[:3]           --> [u*Z_c, v*Z_c, Z_c]^T
Step 3:  pixel = [u*Z_c/Z_c, v*Z_c/Z_c] --> [u, v] in pixels
```

A point is visible only if:
- Z_c > 0 (point is in front of the camera, not behind it)
- 0 <= u < image_width and 0 <= v < image_height (within the image frame)

**Numerical Example:**

Suppose we have a point at P_world = [10, 5, -1, 1]^T (meters, in ego frame), a
camera with fx=800, fy=800, cx=640, cy=360, and after applying the extrinsic
transform we get P_cam = [2.5, 1.0, 15.0]^T. Then:

```
p_img = K * [2.5, 1.0, 15.0]^T
      = [800*2.5 + 640*15.0, 800*1.0 + 360*15.0, 15.0]^T
      = [2000 + 9600, 800 + 5400, 15.0]^T
      = [11600, 6200, 15.0]^T

pixel  = [11600/15, 6200/15] = [773, 413]
```

This point projects to pixel (773, 413) -- visible if the image is at least 774
pixels wide and 414 pixels tall.

### Transformer Decoder Basics

The Transformer decoder (from "Attention is All You Need", Vaswani et al. 2017)
is an architecture that refines a set of query vectors by attending to encoded
features. In object detection, each query represents a potential object.

**Key concepts:**

1. **Queries:** Learnable vectors that "ask questions" of the feature maps.
   Each query eventually becomes one detection (or a no-object prediction).

2. **Self-attention:** Queries attend to each other. This allows the model to
   reason about relationships -- for example, suppressing duplicate detections
   of the same object.

3. **Cross-attention:** Queries attend to encoded features (in standard DETR,
   these are flattened image features). This is where queries gather visual
   evidence. In DETR3D, cross-attention is replaced by geometric feature
   sampling (the core innovation).

4. **Feed-forward network (FFN):** A per-query MLP that provides non-linear
   feature transformation after attention.

5. **Iterative refinement:** The decoder has multiple layers. Each layer refines
   the queries, making predictions progressively more accurate. This is the
   "coarse to fine" principle.

**Standard Decoder Layer (simplified):**

```
query = query + SelfAttention(query, query, query)
query = query + CrossAttention(query, encoder_features, encoder_features)
query = query + FFN(query)
```

In DETR3D, the cross-attention line is replaced by geometry-guided feature sampling,
which we will explain in detail later.

---

## Architecture Overview

DETR3D is a transformer-based 3D object detection framework that operates on
multi-view camera images (typically 6 cameras covering a full 360-degree view
around the ego vehicle). The key insight is that instead of building an explicit
3D feature volume (which is expensive), DETR3D uses camera geometry to project
3D query locations back into 2D images and sample features at those locations.

### High-Level Data Flow

```
                        6 Camera Images
                    (FRONT, FRONT_LEFT, FRONT_RIGHT,
                     BACK, BACK_LEFT, BACK_RIGHT)
                              |
                              v
     +------------------------------------------------+
     |         BACKBONE: ResNet-101 + FPN              |
     |         (shared weights across all 6 views)     |
     |                                                 |
     |   Input: 6 x (3, H, W) images                  |
     |   Output: 6 x 4 feature maps at scales P2-P5   |
     |           each with 256 channels                |
     +------------------------------------------------+
                              |
                              v
              Feature Maps: 6 cameras x 4 FPN levels
              P2: (256, H/4,  W/4 )  -- highest resolution
              P3: (256, H/8,  W/8 )
              P4: (256, H/16, W/16)
              P5: (256, H/32, W/32)  -- lowest resolution
                              |
                              |       +---------------------------+
                              |       | 900 Learnable Queries     |
                              |       | (randomly initialized)    |
                              |       |                           |
                              |       | query_embed: (900, 256)   |
                              |       |          |                |
                              |       |          v                |
                              |       | ref_pts = sigmoid(MLP(q)) |
                              |       | ref_pts: (900, 3)         |
                              |       +---------------------------+
                              |                   |
                              v                   v
     +----------------------------------------------------+
     |           TRANSFORMER DECODER (6 layers)            |
     |                                                     |
     |  For each layer:                                    |
     |  +----------------------------------------------+  |
     |  | 1. Self-Attention                            |  |
     |  |    Queries attend to each other              |  |
     |  |    (900 x 900 attention matrix)              |  |
     |  +----------------------------------------------+  |
     |  +----------------------------------------------+  |
     |  | 2. 3D-to-2D Projection + Feature Sampling    |  |
     |  |    (REPLACES standard cross-attention)       |  |
     |  |                                              |  |
     |  |    ref_pts --[project]--> 2D pixel coords    |  |
     |  |    per camera                                |  |
     |  |                                              |  |
     |  |    Uses: Camera intrinsics K (3x3)           |  |
     |  |          Camera extrinsics [R|t] (4x4)       |  |
     |  |                                              |  |
     |  |    F.grid_sample(features, projected_coords) |  |
     |  +----------------------------------------------+  |
     |  +----------------------------------------------+  |
     |  | 3. Feed-Forward Network (FFN)                |  |
     |  |    2-layer MLP: 256 -> 512 -> 256            |  |
     |  +----------------------------------------------+  |
     |  +----------------------------------------------+  |
     |  | 4. Reference Point Refinement                |  |
     |  |    ref_pts += MLP(query)                     |  |
     |  |    (coarse-to-fine localization)             |  |
     |  +----------------------------------------------+  |
     |  +----------------------------------------------+  |
     |  | 5. Auxiliary Detection Head (for training)    |  |
     |  |    class_scores, bbox_preds per layer        |  |
     |  +----------------------------------------------+  |
     |                                                     |
     +----------------------------------------------------+
                              |
                              v
     +------------------------------------------------+
     |            DETECTION HEAD                        |
     |                                                 |
     |   Classification: 10 classes + background       |
     |   Regression: [cx,cy,cz,w,l,h,sin,cos,vx,vy]  |
     |   Attributes: activity state (optional)         |
     |                                                 |
     |   Output: 900 predictions, filtered by score    |
     +------------------------------------------------+
                              |
                              v
                    Final 3D Detections
              (class, box, velocity, attributes)
```

### Why This Architecture?

Traditional approaches to multi-view 3D detection either:
1. Lift 2D features into a 3D volume (expensive, O(H*W*D*C) memory), or
2. Detect in 2D per-camera and then fuse in 3D (loses multi-view consistency).

DETR3D takes a third approach: it operates in 3D query space but retrieves 2D
evidence on-demand via geometric projection. This gives the model 3D reasoning
ability without the memory cost of a full voxel grid.

---

## Backbone: ResNet-101 + FPN

The backbone extracts rich visual features from each camera image independently.
It consists of two parts: a feature extractor (ResNet-101) and a multi-scale
feature aggregator (Feature Pyramid Network, FPN).

### ResNet-101: The Feature Extractor

ResNet-101 is a 101-layer deep convolutional network pre-trained on ImageNet
(1.2M images, 1000 classes). It learns to extract increasingly abstract visual
features at each stage.

**Architecture stages:**

```
Input Image: (3, 900, 1600)   [channels, height, width]
        |
        v
+------------------+
| Stage 1 (C1)    |  7x7 conv, stride 2, then max pool
| 64 channels     |  Output: (64, 225, 400) -- stride 4
| NOT used in FPN |
+------------------+
        |
        v
+------------------+
| Stage 2 (C2)    |  3 bottleneck blocks
| 256 channels    |  Output: (256, 225, 400) -- stride 4
+------------------+
        |
        v
+------------------+
| Stage 3 (C3)    |  4 bottleneck blocks
| 512 channels    |  Output: (512, 113, 200) -- stride 8
+------------------+
        |
        v
+------------------+
| Stage 4 (C4)    |  23 bottleneck blocks
| 1024 channels   |  Output: (1024, 57, 100) -- stride 16
+------------------+
        |
        v
+------------------+
| Stage 5 (C5)    |  3 bottleneck blocks
| 2048 channels   |  Output: (2048, 29, 50) -- stride 32
+------------------+
```

**Why ResNet-101?**

- Deep enough to capture complex visual patterns (vehicles, pedestrians at
  varying distances and orientations)
- Skip connections prevent gradient vanishing, enabling stable training
- Pre-trained weights provide a strong initialization (critical for perception
  tasks with limited labeled data)
- Well-validated in autonomous driving literature

**Bottleneck Block (the building unit):**

```
Input (C channels)
    |
    +--> 1x1 conv (C -> C/4) --> BatchNorm --> ReLU
    |                                            |
    |         3x3 conv (C/4 -> C/4) --> BatchNorm --> ReLU
    |                                                   |
    |              1x1 conv (C/4 -> C) --> BatchNorm
    |                                          |
    +------------------------------------------+  (skip connection)
                       |
                      ReLU
                       |
                    Output (C channels)
```

### Feature Pyramid Network (FPN): Multi-Scale Features

Objects in autonomous driving vary enormously in apparent size. A pedestrian at
5 meters fills hundreds of pixels; a vehicle at 50 meters might be only 20 pixels
wide. FPN addresses this by combining high-resolution (detailed) features with
low-resolution (semantic) features.

**FPN Architecture:**

```
          C5 (2048, 29, 50)
              |
              v
         1x1 conv --> P5 (256, 29, 50)     stride 32
              |
         upsample 2x
              |
              v
          C4 (1024, 57, 100)
              |
         1x1 conv --> + --> 3x3 conv --> P4 (256, 57, 100)    stride 16
              |
         upsample 2x
              |
              v
          C3 (512, 113, 200)
              |
         1x1 conv --> + --> 3x3 conv --> P3 (256, 113, 200)   stride 8
              |
         upsample 2x
              |
              v
          C2 (256, 225, 400)
              |
         1x1 conv --> + --> 3x3 conv --> P2 (256, 225, 400)   stride 4
```

**Key design choices:**
- All FPN levels output 256 channels (uniform channel dimension simplifies
  downstream processing)
- 4 FPN levels (P2 through P5) provide 4x coverage of spatial scales
- Lateral connections (the `+` operations) merge bottom-up detail with
  top-down semantics
- 3x3 convolutions after merging reduce aliasing artifacts from upsampling

**Which level detects what:**
- P2 (stride 4): Nearby small objects -- pedestrians at close range
- P3 (stride 8): Medium objects -- cyclists, nearby cars
- P4 (stride 16): Standard objects -- cars at typical distances (primary level)
- P5 (stride 32): Distant or large objects -- trucks, buses, far-away vehicles

### Weight Sharing Across Cameras

A critical design decision: the backbone and FPN use **shared weights** across
all 6 camera views. This means the same network processes every camera image.

**Why share weights?**

1. **Parameter efficiency:** 6 separate backbones would use 6x the parameters
   (~330M instead of ~55M). Most of these parameters would learn redundant
   visual features.

2. **Computational efficiency:** With shared weights, we can batch all 6 images
   and process them in a single forward pass through the backbone.

3. **Generalization:** A shared backbone learns camera-agnostic features. It
   does not overfit to the specific viewpoint of each camera. Since cameras can
   be recalibrated or repositioned, viewpoint-agnostic features are more robust.

4. **Transfer learning:** Pre-trained ImageNet weights apply equally to all views.

**What shared weights do NOT handle:**

The backbone does not perform any cross-view reasoning. It extracts features from
each image independently. Cross-view fusion happens later, in the transformer
decoder, where projected features from multiple cameras are aggregated.

### Optional Backbone Enhancements

- **DCNv2 (Deformable Convolutions):** Replacing standard convolutions in Stage 5
  with deformable convolutions allows the network to adaptively adjust receptive
  fields. This helps with objects at non-standard aspect ratios and improves
  detection by ~1-2 NDS points.

- **VoVNet-99:** An alternative backbone with one-shot aggregation that provides
  better efficiency-accuracy trade-off for some configurations.

- **Pre-training on FCOS3D:** Initializing from a monocular 3D detector (FCOS3D)
  pre-trained on nuScenes provides an additional ~1 NDS boost over ImageNet-only
  initialization.

---

## 3D Object Queries and Reference Points

In DETR3D, object queries are the mechanism by which the model "asks" the image
features: "Is there an object here?" Each query learns to specialize in detecting
certain types of objects at certain spatial configurations.

### Query Embeddings

- **Count:** 900 queries (chosen to exceed the maximum number of objects expected
  in any single driving scene; nuScenes scenes typically have 20-80 annotated
  objects, so 900 provides ample capacity)
- **Dimension:** 256 (matching the FPN feature dimension)
- **Initialization:** Randomly initialized as learnable parameters (nn.Embedding)
- **Role:** Each query embedding encodes a hypothesis about an object's existence,
  class, and spatial configuration

### How Reference Points Are Initialized

Each query embedding is mapped to a 3D reference point via a small MLP:

```
query_embedding: (900, 256)
        |
        v
+-------------------+
| Linear(256, 256)  |
| ReLU              |
| Linear(256, 256)  |
| ReLU              |
| Linear(256, 3)    |
+-------------------+
        |
        v
   sigmoid(output)  --> ref_point_normalized: (900, 3)  in [0, 1]^3
```

The sigmoid activation ensures reference points are in the normalized range [0, 1].
This normalized space is then mapped to physical coordinates:

```
ref_point_physical.x = ref_point_normalized.x * (x_max - x_min) + x_min
ref_point_physical.y = ref_point_normalized.y * (y_max - y_min) + y_min
ref_point_physical.z = ref_point_normalized.z * (z_max - z_min) + z_min
```

With our detection ranges:
- X: [-51.2m, 51.2m] --> range = 102.4m
- Y: [-51.2m, 51.2m] --> range = 102.4m
- Z: [-5.0m, 3.0m]   --> range = 8.0m

**Example:** A normalized reference point at (0.6, 0.5, 0.625) maps to:
```
x = 0.6 * 102.4 - 51.2 =  10.24 m
y = 0.5 * 102.4 - 51.2 =   0.0  m
z = 0.625 * 8.0 - 5.0  =   0.0  m
```
This corresponds to a point 10.24 meters ahead of the ego vehicle, centered
laterally, at ground level.

### What Reference Points Represent Physically

A reference point is the model's current best guess for where an object's center
is located in 3D space. Think of it as a "search pointer":

```
         TOP VIEW (bird's eye)
         Y-axis (lateral)
         ^
         |
    -51.2|. . . . . . . . . . . .
         |                        .
         |    *  <-- ref point    .
         |    (a query's guess    .
         |     for object center) .
    -----+-------------------------> X-axis (forward)
         |                        .    (longitudinal)
         |                        .
         |                        .
   -51.2 |. . . . . . . . . . . .
                              51.2

         Detection range: 102.4m x 102.4m
```

During training, the model learns to:
1. Place reference points near actual object locations
2. Refine them through decoder layers (coarse to fine)
3. Predict residual offsets from reference points to exact object centers

### Query Semantics and Specialization

After training, queries develop implicit specializations:
- Some queries learn to detect cars (their reference points cluster at road level)
- Some specialize in pedestrians (reference points near sidewalk regions)
- Some handle distant objects (reference points at the edge of detection range)
- Some remain "idle" (always predict background/no-object)

This specialization emerges naturally from end-to-end training with Hungarian
matching (bipartite assignment between predictions and ground truth). No explicit
anchoring or spatial priors are imposed.

**Key difference from anchor-based methods:** In anchor-based detectors, each
spatial location has fixed anchor boxes. In DETR3D, queries are free to detect
objects anywhere in the 3D space -- their spatial preference is learned, not
hard-coded.

---

## 3D-to-2D Projection and Feature Sampling

This is the core innovation of DETR3D. Instead of standard cross-attention
(which would require flattening all multi-view features into a single sequence,
costing O(N_queries * 6 * H * W) computation), DETR3D uses known camera geometry
to project each 3D reference point to specific 2D pixel locations and samples
features only at those locations.

### The Projection Pipeline Step by Step

Given a 3D reference point in ego-vehicle coordinates, we need to find where it
appears in each camera's image.

**Step 1: Ego-to-Camera Coordinate Transformation**

The ego-vehicle frame uses a coordinate system centered on the vehicle. Each
camera has its own coordinate system defined by its mounting position and
orientation. The extrinsic matrix T_cam_from_ego (4x4) converts between them:

```
P_cam = T_cam_from_ego * P_ego

Where:
  P_ego = [x_ego, y_ego, z_ego, 1]^T    (homogeneous coordinates)
  T_cam_from_ego = [R | t]               (3x3 rotation, 3x1 translation)
                   [0   1]               (bottom row for homogeneous form)

Result:
  P_cam = [x_cam, y_cam, z_cam, 1]^T    (point in camera frame)
```

**Step 2: Camera-to-Pixel Projection**

Using the camera's intrinsic matrix K, we project the 3D camera-frame point
onto the 2D image plane:

```
p_homogeneous = K * P_cam[:3]

         [ fx   0   cx ]   [ x_cam ]   [ fx*x_cam + cx*z_cam ]
p_hom =  [  0  fy   cy ] * [ y_cam ] = [ fy*y_cam + cy*z_cam ]
         [  0   0    1 ]   [ z_cam ]   [        z_cam         ]

Pixel coordinates:
  u = (fx*x_cam + cx*z_cam) / z_cam = fx*(x_cam/z_cam) + cx
  v = (fy*y_cam + cy*z_cam) / z_cam = fy*(y_cam/z_cam) + cy
  depth = z_cam
```

**Step 3: Normalize to [0, 1] Range**

For use with F.grid_sample, pixel coordinates are normalized:

```
u_norm = u / image_width    # in [0, 1] if visible
v_norm = v / image_height   # in [0, 1] if visible
```

### Worked Example with Real Numbers

Consider a reference point at P_ego = (10, 5, -1, 1) in the ego frame
(10m forward, 5m to the left, 1m below ego origin).

**Camera: FRONT camera**
- Extrinsic: Looking forward along ego X-axis, mounted at height 1.5m
- Intrinsic: fx=1260, fy=1260, cx=800, cy=450 (1600x900 image)

After extrinsic transformation (simplified -- real rotation handles axis flips):
```
P_cam_front = [5.0, 2.5, 10.0]^T
(x_cam = lateral offset, y_cam = vertical offset, z_cam = depth = 10m)
```

Projection:
```
u = 1260 * (5.0 / 10.0) + 800 = 630 + 800 = 1430
v = 1260 * (2.5 / 10.0) + 450 = 315 + 450 = 765

depth = 10.0 > 0  (in front of camera)
u_norm = 1430 / 1600 = 0.894  (within [0,1])
v_norm = 765 / 900  = 0.850   (within [0,1])
```

Result: This reference point projects to pixel (1430, 765) in the FRONT camera
and IS visible (depth > 0, coordinates within image bounds).

**Camera: BACK camera**
- Looking backward (opposite to ego X-axis)

After extrinsic transformation:
```
P_cam_back = [-5.0, 2.5, -10.0]^T
depth = -10.0 < 0  --> INVISIBLE (point is behind the back camera)
```

The point is behind the back camera, so it is marked as not visible.

### Visibility Check

For each reference point projected to each camera, we check three conditions:

```python
visible = (depth > 0) and (0 <= u_norm <= 1) and (0 <= v_norm <= 1)
```

```
Visibility Decision Tree:

    depth > 0?
     /      \
   NO       YES
   |          \
INVISIBLE    u_norm in [0,1]?
              /        \
            NO         YES
            |            \
         INVISIBLE     v_norm in [0,1]?
                        /        \
                      NO         YES
                      |            \
                   INVISIBLE     VISIBLE
                                   |
                              Sample features here
```

### Multi-Scale Sampling from FPN Levels

Once we have normalized 2D coordinates (u_norm, v_norm) for a visible camera,
we sample features from all 4 FPN levels at that location:

```
For reference point q projected to (u_norm, v_norm) in camera c:

    sampled_P2 = grid_sample(P2_features[c], (u_norm, v_norm))  # (256,)
    sampled_P3 = grid_sample(P3_features[c], (u_norm, v_norm))  # (256,)
    sampled_P4 = grid_sample(P4_features[c], (u_norm, v_norm))  # (256,)
    sampled_P5 = grid_sample(P5_features[c], (u_norm, v_norm))  # (256,)

    aggregated = sampled_P2 + sampled_P3 + sampled_P4 + sampled_P5  # (256,)
```

The same normalized (u_norm, v_norm) works across all FPN levels because
grid_sample interprets coordinates as fractions of the feature map dimensions.
A point at (0.5, 0.5) normalized is always the center, regardless of the spatial
resolution.

**Bilinear interpolation:** Since projected coordinates rarely land exactly on a
feature map grid cell, F.grid_sample uses bilinear interpolation to compute a
weighted average of the 4 nearest grid cells:

```
     (floor_x, floor_y)-----(ceil_x, floor_y)
           |         *p          |
           |      (u_frac,       |
           |       v_frac)       |
     (floor_x, ceil_y)------(ceil_x, ceil_y)

value = (1-u_frac)*(1-v_frac)*top_left
      + u_frac*(1-v_frac)*top_right
      + (1-u_frac)*v_frac*bottom_left
      + u_frac*v_frac*bottom_right
```

### Multi-Camera Aggregation

A reference point may be visible in multiple cameras simultaneously (especially
near camera overlap regions). DETR3D aggregates features from all cameras where
the point is visible:

```
Algorithm: Multi-camera feature aggregation for query q

visible_cameras = []
features_per_camera = []

For each camera c in {FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT}:
    (u, v, depth) = project(ref_point_q, extrinsics[c], intrinsics[c])
    if depth > 0 and 0 <= u <= 1 and 0 <= v <= 1:
        visible_cameras.append(c)
        feat = sum(grid_sample(FPN_level[c], (u, v)) for all FPN levels)
        features_per_camera.append(feat)

if len(visible_cameras) > 0:
    aggregated_feature = mean(features_per_camera)  # element-wise mean
else:
    aggregated_feature = zeros(256)  # zero vector
```

### What Happens When a Point Is Not Visible in Any Camera

If a reference point is not visible in any camera (all projections fail the
visibility check), the sampled feature is a zero vector. This means:
- The query receives no visual evidence for this position
- The decoder must rely solely on self-attention context and its learned embedding
- In practice, such queries typically predict "no object" (background class)

This can happen for:
- Reference points at extreme positions near the detection boundary
- Points in occluded regions or camera blind spots
- Points very close to the ego vehicle (below all cameras)

### Feature Sampling Code (Detailed)

```python
def feature_sampling(mlvl_feats, reference_points, pc_range, lidar2img):
    """
    Core feature sampling operation of DETR3D.

    Args:
        mlvl_feats: List of 4 tensors, each (B, num_cams, C, H_i, W_i)
                    representing FPN levels P2-P5
        reference_points: (B, num_queries, 3) in normalized [0,1] space
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
                  = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        lidar2img: (B, num_cams, 4, 4) combined extrinsic+intrinsic matrix

    Returns:
        sampled_feats: (B, num_queries, C) aggregated features
    """
    B, num_queries, _ = reference_points.shape
    num_cams = lidar2img.shape[1]

    # 1. Denormalize reference points to physical coordinates
    ref_3d = reference_points.clone()
    ref_3d[..., 0] = ref_3d[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    ref_3d[..., 1] = ref_3d[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    ref_3d[..., 2] = ref_3d[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    # 2. Convert to homogeneous coordinates: (B, num_queries, 4)
    ref_3d_homo = torch.cat([ref_3d, torch.ones_like(ref_3d[..., :1])], dim=-1)

    # 3. Project to all cameras: (B, num_cams, num_queries, 4)
    #    lidar2img: (B, num_cams, 4, 4) @ ref_3d_homo: (B, 1, num_queries, 4, 1)
    ref_pts_cam = torch.einsum('bcij,bqj->bcqi', lidar2img, ref_3d_homo)

    # 4. Perspective division
    depth = ref_pts_cam[..., 2:3]  # (B, num_cams, num_queries, 1)
    eps = 1e-5
    ref_pts_2d = ref_pts_cam[..., :2] / torch.clamp(depth, min=eps)

    # 5. Normalize to [0, 1]
    # (Assuming lidar2img already incorporates image dimensions normalization,
    #  or we divide by image_width and image_height here)
    ref_pts_2d[..., 0] /= img_width
    ref_pts_2d[..., 1] /= img_height

    # 6. Compute visibility mask
    mask = (depth[..., 0] > eps) & \
           (ref_pts_2d[..., 0] > 0) & (ref_pts_2d[..., 0] < 1) & \
           (ref_pts_2d[..., 1] > 0) & (ref_pts_2d[..., 1] < 1)
    # mask: (B, num_cams, num_queries)

    # 7. Convert to grid_sample format: [0,1] -> [-1,1]
    grid = ref_pts_2d * 2 - 1  # (B, num_cams, num_queries, 2)
    grid = grid.unsqueeze(2)   # (B, num_cams, num_queries, 1, 2) for grid_sample

    # 8. Sample from each FPN level and aggregate
    sampled_feats = torch.zeros(B, num_cams, num_queries, C)
    for lvl_feat in mlvl_feats:
        # lvl_feat: (B, num_cams, C, H_l, W_l)
        for cam in range(num_cams):
            feat_cam = lvl_feat[:, cam]  # (B, C, H_l, W_l)
            grid_cam = grid[:, cam]      # (B, num_queries, 1, 2)
            sampled = F.grid_sample(feat_cam, grid_cam,
                                    mode='bilinear',
                                    padding_mode='zeros',
                                    align_corners=False)
            # sampled: (B, C, num_queries, 1) -> squeeze -> (B, C, num_queries)
            sampled_feats[:, cam] += sampled.squeeze(-1).permute(0, 2, 1)

    # 9. Apply visibility mask and aggregate across cameras
    mask_float = mask.float().unsqueeze(-1)  # (B, num_cams, num_queries, 1)
    sampled_feats = sampled_feats * mask_float
    num_visible = mask_float.sum(dim=1).clamp(min=1)  # avoid division by zero
    output = sampled_feats.sum(dim=1) / num_visible  # (B, num_queries, C)

    return output  # (B, 900, 256)
```

---

## Transformer Decoder

The transformer decoder iteratively refines query embeddings and reference points
across 6 layers. Each layer improves the quality of predictions by gathering
better features and reasoning about inter-object relationships.

### Decoder Configuration

| Parameter                | Value |
|--------------------------|-------|
| Number of layers         | 6     |
| Hidden dimension         | 256   |
| Number of attention heads| 8     |
| Head dimension           | 32    |
| FFN intermediate dim     | 512   |
| Dropout                  | 0.1   |
| Activation in FFN        | ReLU  |
| Layer normalization      | Pre-norm (before each sub-layer) |

### Tensor Shapes Through the Decoder

```
Input tensors:
  query_embed:     (B, 900, 256)   -- learnable query embeddings
  ref_points:      (B, 900, 3)     -- initial 3D reference points (normalized)
  mlvl_feats:      list of 4 tensors, each (B, 6, 256, H_l, W_l)

Intermediate tensors at each layer:
  self_attn_Q:     (B, 900, 256)   --> split into 8 heads: (B, 8, 900, 32)
  self_attn_K:     (B, 900, 256)   --> split into 8 heads: (B, 8, 900, 32)
  self_attn_V:     (B, 900, 256)   --> split into 8 heads: (B, 8, 900, 32)
  attn_weights:    (B, 8, 900, 900)  -- 900x900 attention matrix per head
  sampled_feats:   (B, 900, 256)   -- from projection + sampling
  ffn_hidden:      (B, 900, 512)   -- FFN intermediate
  ref_delta:       (B, 900, 3)     -- reference point update

Output tensors:
  query_output:    (B, 900, 256)   -- refined queries
  ref_points_out:  (B, 900, 3)     -- refined reference points
  cls_scores:      (B, 900, 11)    -- per-layer classification (auxiliary)
  bbox_preds:      (B, 900, 10)    -- per-layer regression (auxiliary)
```

### Sub-Layer 1: Self-Attention (Query-to-Query Interactions)

Multi-head self-attention allows queries to communicate with each other. This
serves several purposes:

1. **Duplicate suppression:** If two queries are attending to the same object,
   self-attention allows one to "back off" and search elsewhere.

2. **Spatial reasoning:** Queries can learn that cars on a highway tend to be
   evenly spaced, or that pedestrians cluster near crosswalks.

3. **Contextual hints:** Knowing that a traffic light is present (from one query)
   may help another query better classify a waiting vehicle.

**Computation:**

```
# Pre-norm
query_normed = LayerNorm(query)

# Add positional encoding (derived from reference points)
pos_embed = positional_encoding(ref_points)  # (B, 900, 256)
Q = query_normed + pos_embed
K = query_normed + pos_embed
V = query_normed

# Multi-head attention
Q = reshape(Linear_Q(Q), (B, 8, 900, 32))    # 8 heads, dim 32 each
K = reshape(Linear_K(K), (B, 8, 900, 32))
V = reshape(Linear_V(V), (B, 8, 900, 32))

attn_scores = (Q @ K^T) / sqrt(32)            # (B, 8, 900, 900)
attn_weights = softmax(attn_scores, dim=-1)    # (B, 8, 900, 900)
attn_output = attn_weights @ V                 # (B, 8, 900, 32)

attn_output = reshape(attn_output, (B, 900, 256))  # concatenate heads
attn_output = Linear_out(attn_output)               # (B, 900, 256)

# Residual connection
query = query + Dropout(attn_output)
```

**Complexity:** O(900^2 * 256) = O(207M) multiply-adds. With 900 queries, this
is manageable (much less than O(N*H*W) cross-attention over dense features).

### Sub-Layer 2: Feature Sampling (Replaces Cross-Attention)

In standard DETR, cross-attention computes attention over all spatial positions
in the encoder output. With 6 cameras and multi-scale features, this would be:
- 6 cameras * (225*400 + 113*200 + 57*100 + 29*50) = 6 * 114,100 = 684,600 keys

Computing attention over 684,600 keys for each of 900 queries would be extremely
expensive: O(900 * 684,600 * 256) -- hundreds of billions of operations.

DETR3D replaces this with geometry-guided sampling:

```
# 1. Project reference points to all cameras
ref_pts_2d = project_to_all_cameras(ref_points, camera_matrices)
# ref_pts_2d: (B, 6, 900, 2) -- 2D coords per camera

# 2. Check visibility
vis_mask = compute_visibility(ref_pts_2d, depths)
# vis_mask: (B, 6, 900) -- boolean

# 3. Sample features at projected locations
sampled = sample_from_fpn(mlvl_feats, ref_pts_2d, vis_mask)
# sampled: (B, 900, 256)

# 4. Project sampled features (acts as "value" projection)
cross_output = Linear(sampled)  # (B, 900, 256)

# 5. Residual connection
query = query + Dropout(cross_output)
```

**Key insight:** This replaces a learned attention mechanism (where the model
learns WHERE to look) with a geometric one (where physics TELLS us where to look).
The camera calibration provides exact 3D-to-2D correspondence, eliminating the
need to learn it.

**Computational advantage:**
- Standard cross-attention: O(900 * 684,600) = O(616M) attention computations
- DETR3D sampling: O(900 * 6 * 4) = O(21,600) sampling operations
- Speedup: ~28,500x fewer attention computations (though each sampling operation
  includes the grid_sample cost)

**Important: No learned attention weights.** Unlike standard cross-attention where
attention weights are computed via softmax(Q*K^T/sqrt(d)), the feature sampling
in DETR3D is deterministic given the geometry. The model cannot "choose" to look
elsewhere -- it always looks exactly where the reference point projects to. This
makes the model dependent on accurate reference points (hence the refinement
mechanism).

### Sub-Layer 3: Feed-Forward Network (FFN)

The FFN provides per-query non-linear feature transformation:

```
# Pre-norm
query_normed = LayerNorm(query)

# Two-layer MLP
hidden = Linear(query_normed)    # (B, 900, 256) -> (B, 900, 512)
hidden = ReLU(hidden)
hidden = Dropout(hidden)
output = Linear(hidden)          # (B, 900, 512) -> (B, 900, 256)
output = Dropout(output)

# Residual connection
query = query + output
```

The expansion ratio is 2x (256 -> 512 -> 256), which is modest compared to
standard transformers (typically 4x). This keeps the decoder lightweight.

### Sub-Layer 4: Reference Point Refinement

After each decoder layer, the reference points are updated to better localize
the object center. This enables coarse-to-fine detection:

```
# Predict position offset from current query
delta = MLP_refine(query)   # (B, 900, 3)
# MLP_refine: Linear(256, 256) -> ReLU -> Linear(256, 3)

# Update reference points (in normalized space)
ref_points_new = sigmoid(inverse_sigmoid(ref_points) + delta)
# inverse_sigmoid ensures we can add in logit space before re-normalizing
```

**Why inverse_sigmoid + sigmoid?**

Reference points live in [0, 1] space (after sigmoid). To add a refinement delta,
we first undo the sigmoid (mapping back to unbounded logit space), add the delta
there, and then re-apply sigmoid. This prevents reference points from leaving
the valid [0, 1] range.

```
Mathematically:
  ref_logit = log(ref / (1 - ref))     # inverse sigmoid
  ref_logit_new = ref_logit + delta
  ref_new = sigmoid(ref_logit_new)      # back to [0, 1]
```

**Coarse-to-fine behavior:**

```
Layer 1: ref_point at (0.55, 0.48, 0.60) -- rough guess
         Sampled features give some signal about object location
         Refinement: delta = (+0.02, -0.01, +0.005)

Layer 2: ref_point at (0.57, 0.47, 0.605) -- closer to true center
         Better features sampled from closer-to-correct location
         Refinement: delta = (+0.005, +0.002, -0.001)

Layer 3: ref_point at (0.575, 0.472, 0.604) -- converging
         ...and so on through layers 4, 5, 6

Layer 6: ref_point at (0.578, 0.473, 0.603) -- final position
         This maps to physical coordinates:
         x = 0.578 * 102.4 - 51.2 = 7.99 m
         y = 0.473 * 102.4 - 51.2 = -2.75 m
         z = 0.603 * 8.0 - 5.0    = -0.18 m
```

### Sub-Layer 5: Auxiliary Predictions

At each decoder layer, a detection head (shared across layers) produces
predictions. During training, all 6 layers' predictions are supervised:

```
For layer l in [1, 2, 3, 4, 5, 6]:
    cls_scores_l = classification_head(query_l)   # (B, 900, 11)
    bbox_preds_l = regression_head(query_l, ref_points_l)  # (B, 900, 10)

Loss = sum(loss_l for l in layers) / num_layers
```

Auxiliary losses serve multiple purposes:
- Combat vanishing gradients (direct supervision at every layer)
- Encourage progressive refinement (each layer must improve)
- Stabilize training (the model cannot rely solely on the last layer)

During inference, only the final layer's predictions are used.

### Complete Decoder Forward Pass

```
def decoder_forward(query_embed, ref_points, mlvl_feats, camera_matrices):
    """
    Complete decoder with 6 layers.

    Tensor shapes annotated at each step.
    """
    query = query_embed              # (B, 900, 256)
    all_cls_scores = []
    all_bbox_preds = []

    for layer_idx in range(6):
        # --- Self-Attention ---
        # query: (B, 900, 256) -> self_attn -> (B, 900, 256)
        query = self_attention_sublayer(query, ref_points)

        # --- Feature Sampling (replaces Cross-Attention) ---
        # Project ref_points (B, 900, 3) to all cameras
        # Sample features: (B, 900, 256)
        sampled = feature_sampling(mlvl_feats, ref_points, camera_matrices)
        query = query + linear_projection(sampled)  # (B, 900, 256)

        # --- FFN ---
        # query: (B, 900, 256) -> FFN -> (B, 900, 256)
        query = ffn_sublayer(query)

        # --- Reference Point Refinement ---
        # ref_points: (B, 900, 3) + delta -> (B, 900, 3)
        delta = refine_mlp(query)  # (B, 900, 3)
        ref_points = sigmoid(inverse_sigmoid(ref_points) + delta)

        # --- Auxiliary Predictions ---
        cls = classification_head(query)           # (B, 900, 11)
        bbox = regression_head(query, ref_points)  # (B, 900, 10)
        all_cls_scores.append(cls)
        all_bbox_preds.append(bbox)

    return query, ref_points, all_cls_scores, all_bbox_preds
```

---

## Detection Head

The detection head converts refined query features into actual object predictions.
It is applied at every decoder layer (with shared weights) for auxiliary supervision.

### Classification Head

Predicts which of the 10 nuScenes object classes (or background) each query
represents.

```
Architecture:
  Input: query feature (B, 900, 256)
      |
      v
  Linear(256, 256) + ReLU + LayerNorm
      |
      v
  Linear(256, 11)
      |
      v
  Output: class logits (B, 900, 11)

Classes (10 + background):
  0: car
  1: truck
  2: construction_vehicle
  3: bus
  4: trailer
  5: barrier
  6: motorcycle
  7: bicycle
  8: pedestrian
  9: traffic_cone
  10: background (no object)
```

During training: focal loss is applied to the raw logits (no activation needed).
During inference: sigmoid is applied to get per-class probabilities.

### 3D Bounding Box Regression Head

Predicts a 10-dimensional vector encoding the full 3D bounding box and velocity:

```
Architecture:
  Input: query feature (B, 900, 256)
      |
      v
  Linear(256, 256) + ReLU + LayerNorm
      |
      v
  Linear(256, 10)
      |
      v
  Output: raw predictions (B, 900, 10)

Output vector breakdown:
  [0:3]  delta_cx, delta_cy, delta_cz  -- offset from reference point to box center
  [3:6]  log(w), log(l), log(h)        -- log-scale dimensions
  [6:8]  sin(yaw), cos(yaw)            -- heading angle as sine/cosine
  [8:10] vx, vy                        -- velocity in X and Y (m/s)
```

**Computing final box parameters:**

```python
# Center position: reference point + predicted offset
center_x = ref_point_x + delta_cx  # already in physical meters
center_y = ref_point_y + delta_cy
center_z = ref_point_z + delta_cz

# Dimensions: exponentiate log predictions
width  = exp(log_w)   # in meters
length = exp(log_l)   # in meters
height = exp(log_h)   # in meters

# Heading: atan2 recovers the angle
yaw = atan2(sin_yaw, cos_yaw)  # in radians

# Velocity: direct prediction
vel_x = vx  # m/s in ego frame
vel_y = vy  # m/s in ego frame
```

**Why sin/cos for heading instead of raw angle?**

Raw angles have a discontinuity at +/- pi (wrapping). Using sin/cos
representation avoids this discontinuity, making regression smoother. The
network can predict any heading without encountering boundary effects.

**Why log-scale for dimensions?**

Vehicle dimensions vary by orders of magnitude (a traffic cone is ~0.5m tall,
a bus is ~4m tall). Log-scale predictions ensure that relative errors are
penalized equally across all sizes.

### Attribute Prediction Head (Optional)

Some nuScenes classes have associated attributes (e.g., vehicle moving/stopped,
pedestrian sitting/standing):

```
Architecture:
  Input: query feature (B, 900, 256)
      |
      v
  Linear(256, 8)
      |
      v
  Output: attribute logits (B, 900, 8)

Attribute classes:
  0: vehicle.moving
  1: vehicle.parked
  2: vehicle.stopped
  3: pedestrian.moving
  4: pedestrian.sitting_lying_down
  5: pedestrian.standing
  6: cycle.with_rider
  7: cycle.without_rider
```

Attributes are class-conditional: only certain attributes are valid for each
object class. Invalid attribute predictions are masked during evaluation.

### Output Format Summary

For each frame, the model outputs predictions for all 900 queries:

```python
predictions = {
    'cls_scores': tensor(B, 900, 11),     # class logits (no activation)
    'bbox_preds': tensor(B, 900, 10),     # [cx,cy,cz,w,l,h,sin,cos,vx,vy]
    'attr_preds': tensor(B, 900, 8),      # attribute logits (optional)
}

# With auxiliary outputs from all 6 decoder layers:
all_predictions = {
    'cls_scores': tensor(6, B, 900, 11),  # per-layer class logits
    'bbox_preds': tensor(6, B, 900, 10),  # per-layer regressions
}
```

---

## Inference Pipeline

This section describes the complete forward pass during deployment, from raw
images to final 3D detections.

### Step-by-Step Inference

```
Step 1: Load Inputs
+-------------------------------------------------------------------+
| - 6 camera images: each (3, 900, 1600) in RGB                    |
| - 6 intrinsic matrices K: each (3, 3)                            |
| - 6 extrinsic matrices [R|t]: each (4, 4)                        |
| - Combined lidar2img: (6, 4, 4) = K @ [R|t]                      |
+-------------------------------------------------------------------+
                              |
                              v
Step 2: Backbone Feature Extraction
+-------------------------------------------------------------------+
| - Stack 6 images into batch: (6, 3, 900, 1600)                   |
| - Forward through ResNet-101:                                     |
|     C2: (6, 256, 225, 400)                                        |
|     C3: (6, 512, 113, 200)                                        |
|     C4: (6, 1024, 57, 100)                                        |
|     C5: (6, 2048, 29, 50)                                         |
| - Forward through FPN:                                            |
|     P2: (6, 256, 225, 400)                                        |
|     P3: (6, 256, 113, 200)                                        |
|     P4: (6, 256, 57, 100)                                         |
|     P5: (6, 256, 29, 50)                                          |
+-------------------------------------------------------------------+
                              |
                              v
Step 3: Initialize Queries and Reference Points
+-------------------------------------------------------------------+
| - Load query_embed: (900, 256) from learned parameters            |
| - Predict reference points: sigmoid(MLP(query_embed))             |
|   ref_points: (900, 3) in [0, 1]^3                               |
| - Expand for batch: (B, 900, 256) and (B, 900, 3)                |
+-------------------------------------------------------------------+
                              |
                              v
Step 4: Transformer Decoder (6 layers)
+-------------------------------------------------------------------+
| For each layer:                                                   |
|   1. Self-attention among 900 queries                             |
|   2. Project ref_points to 6 cameras, sample features             |
|   3. FFN transformation                                           |
|   4. Refine reference points                                      |
|                                                                   |
| Output: query_feat (B, 900, 256), ref_points (B, 900, 3)         |
+-------------------------------------------------------------------+
                              |
                              v
Step 5: Detection Head (final layer only)
+-------------------------------------------------------------------+
| - cls_scores = classification_head(query_feat)  # (B, 900, 11)   |
| - bbox_preds = regression_head(query_feat, ref_points) # (B,900,10)|
| - attr_preds = attribute_head(query_feat)       # (B, 900, 8)    |
+-------------------------------------------------------------------+
                              |
                              v
Step 6: Post-Processing
+-------------------------------------------------------------------+
| 1. Apply sigmoid to cls_scores -> probabilities                   |
| 2. Take max class probability per query (exclude background)      |
| 3. Filter by confidence threshold (e.g., score > 0.1)            |
| 4. Decode bbox_preds:                                             |
|    - Add offsets to reference points for centers                  |
|    - Exp for dimensions                                           |
|    - atan2 for heading                                            |
| 5. Convert from normalized to physical coordinates                |
| 6. NO NMS required (set-based formulation)                        |
|                                                                   |
| Output: List of detections, each with:                            |
|   - class label and confidence score                              |
|   - 3D center (x, y, z) in ego frame                             |
|   - dimensions (w, l, h) in meters                                |
|   - heading angle (yaw) in radians                                |
|   - velocity (vx, vy) in m/s                                     |
|   - attribute prediction                                          |
+-------------------------------------------------------------------+
```

### Why No NMS?

Non-Maximum Suppression (NMS) is a standard post-processing step in most
detectors that removes duplicate boxes. DETR3D does NOT need NMS because:

1. **Hungarian matching during training:** Each ground-truth object is assigned
   to exactly one query. This trains each query to predict at most one object.

2. **Self-attention suppression:** Queries communicate via self-attention and
   learn to avoid predicting the same object.

3. **Set-based loss:** The bipartite matching loss inherently penalizes
   duplicate predictions.

This NMS-free design eliminates a hyper-parameter (NMS threshold) and avoids
the failure mode where NMS incorrectly suppresses nearby but distinct objects
(e.g., two pedestrians walking side-by-side).

### Inference Timing

**Hardware:** Single NVIDIA V100 (32 GB, Tensor Cores)

```
+----------------------------+----------+---------+
| Stage                      | Time (ms)| % Total |
+----------------------------+----------+---------+
| Data loading + preprocess  |    10    |    6%   |
| Backbone (ResNet-101)      |    60    |   36%   |
| FPN                        |    15    |    9%   |
| Decoder (6 layers)         |    55    |   33%   |
|   - Self-attention         |    (15)  |         |
|   - Projection + sampling  |    (30)  |         |
|   - FFN + refinement       |    (10)  |         |
| Detection head             |     5    |    3%   |
| Post-processing            |     5    |    3%   |
| Total (overhead included)  |   ~165   |  100%   |
+----------------------------+----------+---------+

Throughput: ~6-8 FPS (frames per second)
Latency: ~125-165 ms per frame
```

**Memory Usage:**

```
+----------------------------+----------+
| Component                  | Memory   |
+----------------------------+----------+
| Model parameters           |  ~250 MB |
| Feature maps (6 cams, FPN) | ~3.5 GB  |
| Decoder activations        |  ~500 MB |
| Workspace / overhead       | ~3.5 GB  |
| Total (inference, BS=1)    |  ~8 GB   |
+----------------------------+----------+
```

### Latency Breakdown Analysis

The backbone dominates inference time (~45% combining ResNet + FPN). This is
because processing 6 high-resolution images (900x1600 each) through a 101-layer
network is inherently compute-intensive.

**Potential optimizations:**
- Smaller backbone (ResNet-50): saves ~25% backbone time, loses ~1-2 NDS
- Lower input resolution (600x1067): saves ~40% backbone time, loses ~2-3 NDS
- TensorRT optimization: 1.5-2x overall speedup
- INT8 quantization: additional 1.3-1.5x speedup (with calibration)
- Temporal feature caching: reuse backbone features from previous frame for
  static scene elements

### Comparison with Other Approaches

```
+----------------+--------+--------+---------+--------+
| Method         | FPS    | Memory | NDS     | mAP    |
+----------------+--------+--------+---------+--------+
| DETR3D         | 6-8    | 8 GB   | 42.2    | 34.9   |
| BEVFormer      | 3-4    | 12 GB  | 51.7    | 41.6   |
| PETR           | 8-10   | 7 GB   | 38.1    | 31.3   |
| BEVDet         | 5-6    | 10 GB  | 39.2    | 31.2   |
+----------------+--------+--------+---------+--------+
(nuScenes val set, ResNet-101 backbone, no test-time augmentation)
```

DETR3D offers a good balance between speed and accuracy. It is faster than
BEV-based methods (BEVFormer, BEVDet) because it avoids constructing an explicit
3D feature volume, while being more accurate than purely query-based methods
(PETR) because it uses precise geometric projection rather than learned positional
encodings for 3D-to-2D correspondence.

---

## Model Dimensions Summary

| Component | Parameter | Value |
|-----------|-----------|-------|
| Input image size | H x W | 900 x 1600 |
| Number of cameras | - | 6 |
| Backbone | - | ResNet-101 (pre-trained ImageNet) |
| FPN output channels | C | 256 |
| FPN levels | - | 4 (P2, P3, P4, P5) |
| Number of queries | N_q | 900 |
| Query dimension | d | 256 |
| Decoder layers | L | 6 |
| Attention heads | H | 8 |
| Head dimension | d_h | 32 (= 256/8) |
| FFN hidden dim | d_ff | 512 |
| FFN expansion ratio | - | 2x |
| Detection range (X) | - | [-51.2m, 51.2m] |
| Detection range (Y) | - | [-51.2m, 51.2m] |
| Detection range (Z) | - | [-5.0m, 3.0m] |
| Output classes | - | 10 + 1 (background) |
| Bbox parameters | - | 10 (cx,cy,cz,w,l,h,sin,cos,vx,vy) |
| Backbone parameters | - | ~55M |
| Decoder + heads parameters | - | ~10M |
| Total parameters | - | ~65M |
| Inference FPS (V100) | - | 6-8 |
| Inference memory (V100) | - | ~8 GB |

---

## Key Takeaways

1. **Geometry over attention:** DETR3D's core insight is that camera calibration
   provides exact 3D-to-2D correspondence, eliminating the need for learned
   cross-attention over dense feature maps.

2. **Query-based set prediction:** 900 learnable queries with Hungarian matching
   produce duplicate-free predictions without NMS.

3. **Iterative refinement:** 6 decoder layers progressively improve reference
   point accuracy and feature quality (coarse to fine).

4. **Multi-view aggregation:** Features from all cameras where a point is visible
   are averaged, providing multi-view consistency without explicit 3D volumes.

5. **Lightweight decoder:** The expensive part is the backbone, not the decoder.
   The geometric sampling approach makes the decoder O(N_queries * N_cameras)
   instead of O(N_queries * total_spatial_positions).

6. **End-to-end training:** No hand-crafted components (no anchors, no NMS, no
   depth estimation). Everything is learned through backpropagation.
