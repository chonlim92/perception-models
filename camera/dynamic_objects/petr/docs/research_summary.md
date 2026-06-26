# PETR / PETRv2 / StreamPETR - Research Summary

## A Comprehensive Guide to 3D Position Embedding for Camera-Based 3D Object Detection

This document teaches the PETR family of models from first principles. It is intended
for someone new to autonomous driving perception who wants to understand not just *what*
these models do, but *why* each design decision was made.

---

## 1. Why 3D Detection from Cameras is Hard

### The Fundamental Problem

An autonomous vehicle needs to know the 3D position, size, and orientation of every
object around it -- cars, pedestrians, cyclists, trucks. The output for each detected
object is a **3D bounding box**: a center position (x, y, z) in meters, dimensions
(width, height, length), heading angle, and velocity.

However, cameras produce **2D images**. A camera projects the 3D world onto a flat
sensor through perspective projection. This process is inherently lossy: it discards
depth information.

### Depth Ambiguity

Consider a single pixel in a camera image showing a car bumper. That pixel could
correspond to:
- A small car 10 meters away
- A medium car 20 meters away
- A large truck 50 meters away

All three scenarios project to the same pixel location. Without depth, we cannot tell
which interpretation is correct. This is the **depth ambiguity** problem.

```
                        Camera
                          |
                          v
     Far (50m)    --------+--------    Large truck
                       /     \
     Mid (20m)    ----+-------+----    Medium car
                     / \     / \
     Near (10m)  --+-----+-----+--     Small car
                   |     |     |
                   v     v     v
              Same pixels on image sensor
```

### Why Monocular Depth is Unreliable

Humans estimate depth using many cues: object size priors, texture gradients, occlusion,
perspective convergence. Neural networks can learn these cues too, but they remain
**ambiguous and error-prone**, especially for:
- Unusual object sizes (a child vs. an adult at the same distance)
- Featureless regions (blank walls, uniform road surfaces)
- Adverse weather (fog, rain blur distance cues)

The core challenge for any camera-based 3D detection system is: **how do we recover the
lost depth dimension?** Different paradigms answer this question differently.

---

## 2. The Three Paradigms for Camera-Based 3D Detection

There are three dominant approaches to bridging the 2D-to-3D gap. Understanding all
three clarifies what makes PETR distinctive.

### 2.1 Lift-Splat-Shoot (LSS) / BEVDet: Explicit Depth Prediction

**Core idea**: Predict depth for every pixel, "lift" 2D features into 3D space, then
compress (splat) them into a Bird's-Eye-View (BEV) grid.

**Step by step**:
1. Extract 2D image features with a backbone (ResNet, Swin, etc.)
2. For each pixel, predict a **depth distribution** over D bins (e.g., 59 bins from
   1m to 60m). This produces a D-dimensional probability vector per pixel.
3. Multiply the feature at each pixel by its depth distribution, creating D copies of
   the feature, each weighted by the probability of that depth.
4. Place these weighted features into their corresponding 3D voxel positions. This
   "lifts" 2D features into a 3D volume.
5. Collapse the vertical (Z) axis to produce a 2D BEV feature map (e.g., 200x200 grid
   covering 100m x 100m around the vehicle).
6. Run a detection head (or segmentation head) on the BEV features.

```
  Image Features        Depth Distribution        3D Voxels            BEV Grid
  [H x W x C]    x     [H x W x D]        ->   [X x Y x Z x C]  ->  [X x Y x C]
                                                    "Lift"              "Splat"
```

**Strengths**:
- Produces an explicit BEV representation useful for many tasks (detection, segmentation,
  motion planning)
- Geometric reasoning is baked in through the lift operation

**Weaknesses**:
- Depth prediction is the bottleneck: errors in depth corrupt the entire 3D volume
- Memory-intensive: the intermediate 3D voxel grid is large
- Slow due to the splat operation (scatter operation on GPU)

**Representative methods**: LSS (2020), BEVDet (2022), BEVDepth (2022)

---

### 2.2 DETR3D: Project 3D Reference Points Down onto 2D Images

**Core idea**: Start with learnable 3D reference points (one per object query), project
them onto 2D images using known camera parameters, and sample features at those 2D
locations.

**Step by step**:
1. Extract 2D image features from all cameras.
2. Initialize N object queries (e.g., 900), each with a learnable 3D reference point
   (x, y, z) in ego-vehicle coordinates.
3. For each query's 3D reference point, project it onto each camera image using the
   camera calibration matrices. This gives a 2D pixel coordinate per camera.
4. Sample the 2D feature at that pixel location (using bilinear interpolation).
5. Aggregate sampled features across cameras and feed them to the transformer decoder.
6. The decoder refines the reference points and predicts bounding boxes.

```
    3D Reference Point (x, y, z)
             |
             | Project using camera matrices
             v
    2D pixel (u, v) on Camera 3
             |
             | Sample feature at (u, v)
             v
    Feature vector for this query
```

**Strengths**:
- No explicit depth prediction needed
- No expensive 3D voxel grid
- Elegant and simple geometry

**Weaknesses**:
- Projection happens at every decoder layer (6 layers), adding computational cost
- A query only attends to its projected location -- it cannot "look around" for context
- Projection requires accurate camera calibration; errors shift sampling locations
- Gradient flow through the projection operation can be unstable

**Representative methods**: DETR3D (2021), PETR (2022) improves upon this

---

### 2.3 PETR: Encode 3D Information INTO 2D Features

**Core idea**: Instead of projecting 3D points onto images at runtime, pre-encode the
3D position information directly into the 2D features. Then standard cross-attention
naturally performs 3D-aware reasoning without any explicit geometric operations.

This is the PETR paradigm. The key insight is:

> If every image feature already "knows" its potential 3D location, then attention
> between queries and features is inherently 3D-aware.

Think of it like labeling every book in a library with its GPS coordinates. A librarian
looking for "the book at latitude 48.8566, longitude 2.3522" can simply read the labels
-- no map consultation needed.

The rest of this document explains this paradigm in detail.

---

## 3. PETR's 3D Position Embedding Explained from Scratch

### 3.1 The Camera Frustum: What a Camera Can See

A **frustum** is the 3D volume visible to a camera. It looks like a truncated pyramid:
narrow near the camera, widening with distance.

```
  Side view of a camera frustum:
  
  Camera ----+
             |\
             | \
             |  \        Far plane (e.g., 61m)
             |   +-------------------------------+
             |  /                                 |
             | /          FRUSTUM VOLUME          |
             |/           (what camera sees)      |
             +-------------------------------+    |
             |\                              |    |
             | \         Near plane (1m)     |    |
             |  +---+                        |    |
             | /    |                        |    |
             |/     |                        +----+
  Camera ----+      |
                    |
                    v
              Image plane
```

More precisely: for each pixel (u, v) in the image, there is a **ray** extending from
the camera center through that pixel out into the world. Any object along that ray
projects to the same pixel. The frustum is the union of all such rays, bounded by a
near plane and a far plane.

```
  Top-down view of rays from a single camera:
  
                    Camera center
                         *
                        /|\
                       / | \
                      /  |  \
                     /   |   \
                    /    |    \
                   / ray | ray \
                  /      |      \
                 /       |       \
                /________|________\
               |                   |
               |   visible area    |
               |   (frustum)       |
               |___________________|
                                    
               1m                 61m
               (near)            (far)
```

### 3.2 Discretizing the Frustum into 3D Points

PETR creates a grid of 3D points within each camera's frustum. For each pixel in the
**feature map** (not the original image -- typically 1/32 of the original resolution),
it creates multiple points along the depth ray.

Specifically:
- Let the feature map have dimensions H_f x W_f (e.g., 16 x 44 for a 512 x 1408 input
  image with 1/32 downsampling).
- Choose D depth bins (e.g., D = 64, uniformly spaced from 1m to 61.2m, so each bin
  covers ~0.94m).
- For each pixel (u, v) in the feature map and each depth d_k, create a 3D point.

This gives us H_f x W_f x D points per camera. For 6 cameras with 16x44 feature maps
and 64 depth bins, that is 6 x 16 x 44 x 64 = 270,336 points total.

```
  A single pixel's depth samples:
  
  Pixel (u, v) --->  d1=1m   d2=1.94m   d3=2.88m  ...  d64=61.2m
                      *         *           *                *
                      |---------|-----------|----...---------|
                      
  Each * is a 3D point in camera coordinates:
  Point_k = (u * d_k, v * d_k, d_k)  [in normalized camera coords]
```

### 3.3 From Pixels to 3D Camera Coordinates

To convert a pixel (u, v) at depth d into 3D camera coordinates, we use the camera's
**intrinsic matrix** K.

The intrinsic matrix encodes the camera's internal geometry:

```
        [ fx   0   cx ]
  K  =  [  0  fy   cy ]
        [  0   0    1 ]

  where:
    fx, fy = focal lengths (in pixels)
    cx, cy = principal point (image center, in pixels)
```

The relationship between a 3D point (X_c, Y_c, Z_c) in camera coordinates and its
pixel projection (u, v) is:

```
  d * [u]     [ fx   0   cx ] [ X_c ]
      [v]  =  [  0  fy   cy ] [ Y_c ]
      [1]     [  0   0    1 ] [ Z_c ]
```

Inverting this, given pixel (u, v) and depth d:

```
  [ X_c ]         [ u ]         [ (u - cx) / fx ]
  [ Y_c ] = K^-1 * d * [ v ] = d * [ (v - cy) / fy ]
  [ Z_c ]         [ 1 ]         [       1        ]
```

So each (pixel, depth) pair gives us a concrete 3D point in the camera's coordinate
system.

### 3.4 Transforming to World (Ego-Vehicle) Coordinates

Each camera has a known position and orientation relative to the ego vehicle. This is
described by the **extrinsic matrix** T_ego_cam (a 4x4 transformation matrix):

```
  T_ego_cam = [ R  | t ]     (3x3 rotation, 3x1 translation)
              [ 0  | 1 ]
```

To convert from camera coordinates to ego-vehicle (world) coordinates:

```
  P_world = T_ego_cam * P_camera_homogeneous
```

Putting it all together, for pixel (u, v) at depth d:

```
  P_world = T_ego_cam * K^{-1} * [u*d, v*d, d, 1]^T
```

This single equation is the geometric foundation of PETR's 3D position embedding.
It converts every (pixel, depth) pair from every camera into a unified 3D coordinate
system centered on the ego vehicle.

### 3.5 The Full Pipeline: Coordinate Normalization

Before feeding the 3D coordinates into the network, PETR normalizes them:

```
  x_norm = (x - x_min) / (x_max - x_min)
  y_norm = (y - y_min) / (y_max - y_min)
  z_norm = (z - z_min) / (z_max - z_min)
```

where (x_min, x_max, y_min, y_max, z_min, z_max) define the detection range (e.g.,
-51.2m to 51.2m in X and Y, -5m to 3m in Z). This brings all coordinates into [0, 1].

### 3.6 MLP Encoding: From 3D Coordinates to Feature Space

The normalized 3D coordinates (x, y, z) are just 3 numbers. We need to map them to
the same dimensionality as the image features (typically 256). This is done by a
small MLP (Multi-Layer Perceptron):

```
  PE_3d = MLP(x_norm, y_norm, z_norm)
  
  MLP architecture:
    Linear(3, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 256)
```

But wait -- each pixel has D=64 depth bins, giving 64 different 3D points. The MLP
processes all 64 points, and the resulting 64 position embeddings are aggregated
(typically by a weighted sum or max-pooling over the depth dimension) into a single
256-d position embedding per pixel.

The output is a tensor of shape [N_cameras x H_f x W_f x 256], where each element
encodes "what 3D locations this pixel could correspond to."

### 3.7 Adding Position Embedding to Image Features

Finally, the 3D position embeddings are added to the image features:

```
  F_position_aware = F_image + PE_3d
```

where:
- F_image: the backbone's output features [N_cameras x H_f x W_f x 256]
- PE_3d: the 3D position embeddings [N_cameras x H_f x W_f x 256]

After this addition, each feature vector "knows" not just what it sees (appearance from
F_image) but also where it could be in 3D space (geometry from PE_3d).

### 3.8 Summary: The Complete 3D PE Pipeline

```
  For each camera c:
    For each pixel (u, v) in feature map:
      For each depth bin d_k (k = 1..64):
        1. Compute camera coords:  P_cam = K^{-1} * [u*d_k, v*d_k, d_k]
        2. Transform to world:     P_world = T_ego_cam * P_cam
        3. Normalize:              P_norm = normalize(P_world)
        4. Encode with MLP:        pe_k = MLP(P_norm)
      5. Aggregate over depth:     PE_3d(u,v) = Aggregate({pe_1, ..., pe_64})
    6. Add to image features:      F(u,v) = F_image(u,v) + PE_3d(u,v)
```

---

## 4. Why This Works Intuitively

### 4.1 Attention as Soft Nearest-Neighbor Lookup

In a transformer, cross-attention works like this:
- A **query** (representing "I'm looking for a car at position (10, 5, 0)") computes
  a dot product with all **keys** (the position-aware features).
- High dot product = high attention weight = "this feature is relevant to my query."

If the features have been tagged with 3D position information via PE, then:
- A query encoding position (10, 5, 0) will naturally produce high dot products with
  features whose PE encodes positions near (10, 5, 0).
- Features from cameras that can see the point (10, 5, 0) will have matching PEs.
- Features from cameras that cannot see that point will have very different PEs, so
  they get low attention weights automatically.

```
  Query: "Find car at (10, 5, 0)"         Attention weights:
      |                                     
      |                                    Feature at PE~(10, 5, 0.5)  --> HIGH (0.7)
      +-----> dot product with all ----->  Feature at PE~(3, 12, 0)    --> LOW  (0.01)
              position-aware features       Feature at PE~(10, 4.8, 0)  --> HIGH (0.6)
                                           Feature at PE~(-5, 20, 0)   --> LOW  (0.001)
```

### 4.2 No Explicit Projection Needed at Runtime

Unlike DETR3D, which must project 3D points onto images at every decoder layer,
PETR computes the 3D PE only once during feature encoding. After that, standard
attention handles the 3D reasoning implicitly. This means:
- No camera projection matrices needed during the decoder forward pass
- No feature sampling at specific 2D locations
- No unstable gradients through geometric projection operations

### 4.3 Natural Multi-Camera Fusion

Because all cameras' features are encoded in the same world coordinate system, a
query can attend to features from ALL cameras simultaneously in a single attention
operation. Features from overlapping camera views that see the same 3D point will
have similar 3D PEs, allowing the model to naturally fuse information from multiple
viewpoints.

```
  Front-left camera:    Feature at pixel (100, 50) --> PE encodes 3D point (12, 3, 0.5)
  Front camera:         Feature at pixel (200, 50) --> PE encodes 3D point (12, 3.1, 0.5)
  
  Both features get high attention from a query looking for an object near (12, 3, 0.5).
  The model learns to combine both viewpoints.
```

### 4.4 Robustness to Calibration Errors

Small errors in camera calibration shift the projected 2D locations in DETR3D-style
methods, causing features to be sampled from wrong positions. In PETR, calibration
errors shift the 3D PE values slightly, but the attention mechanism can still match
"approximately correct" positions through its learned soft-matching behavior. This
makes PETR more robust to imperfect calibration in practice.

---

## 5. PETRv2: Extensions to the Base Framework

PETRv2 (ICCV 2023) extends PETR with four key improvements:

### 5.1 Temporal Feature Alignment

**The problem**: A single frame gives limited information about object motion. Velocity
estimation is impossible without temporal context.

**The solution**: Use the previous frame's features alongside the current frame's
features. But there is a subtlety -- the ego vehicle has moved between frames, so the
3D coordinate systems are different.

**How it works**:
1. Store the previous frame's position-aware features (F_prev with PE_3d_prev).
2. At the current frame, compute the ego-motion transformation T between frames.
3. Re-compute the 3D PE for the previous frame's features using the current frame's
   coordinate system:
   ```
   PE_3d_prev_aligned = MLP(T_current_from_prev * P_prev_world)
   ```
4. Concatenate current and previous features:
   ```
   F_temporal = Concat(F_current + PE_3d_current, F_prev + PE_3d_prev_aligned)
   ```
5. Object queries attend to this combined temporal feature set.

Now the model can see how objects have moved between frames, enabling velocity
estimation.

### 5.2 Combined 2D + 3D Position Embedding

**Insight**: 3D PE encodes geometric information, but discards fine-grained 2D spatial
information like texture edges and local patterns.

**Solution**: Add a learnable 2D positional encoding alongside the 3D PE:
```
  F = F_image + PE_3d + PE_2d
```

The 2D PE is a standard learnable positional embedding (like in ViT or DETR). It helps
the model retain awareness of where features are within each image, complementing the
3D geometric information.

### 5.3 Multi-Task Capability

PETRv2 adds optional auxiliary heads:
- **BEV segmentation**: Predict drivable area, lane markings from the shared features
- **Depth estimation**: Predict per-pixel depth as a supervision signal for the backbone

These auxiliary tasks provide additional gradient signals that improve the backbone's
3D understanding, even if only detection is needed at deployment.

### 5.4 Linear Increasing Discretization (LID)

**Problem**: Uniform depth bins (e.g., every 0.94m from 1m to 61m) waste resolution
on far-away regions where objects are small and hard to detect, while under-sampling
near regions where precise localization matters most for safety.

**Solution**: Non-uniform depth bins that are denser near the vehicle and sparser far
away:

```
  Uniform (baseline):
  |--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
  1m                                                      61m
  
  LID (Linear Increasing Discretization):
  ||||||||||--|---|------|-----------|---------------------|
  1m      5m   10m     20m         35m                   61m
  
  Finer resolution close up (where safety matters most)
```

This gives better depth resolution for nearby objects (the ones most relevant for
collision avoidance) without increasing the total number of depth bins.

### 5.5 PETRv2 Performance Improvement

| Model | mAP | NDS | mAVE (velocity error) |
|-------|-----|-----|-----------------------|
| PETR (R50) | 31.3 | 38.1 | 0.93 |
| PETRv2 (R50) | 34.6 | 42.1 | 0.41 |

Key observations:
- +3.3 mAP from temporal fusion and improved PE
- mAVE (velocity error) drops by over 50% -- temporal alignment enables velocity
  estimation that was impossible in single-frame PETR
- NDS improvement (+4.0) is even larger than mAP improvement because NDS weights
  velocity and attribute accuracy

---

## 6. StreamPETR: Object-Centric Temporal Modeling

### 6.1 The Problem with Feature-Level Temporal Fusion

Previous temporal methods store and align **entire feature maps** across frames:
- BEVFormer: stores 200x200x256 BEV features per frame, aligns with deformable
  attention. For T=4 frames of history: 200x200x256x4 = 40 million values.
- PETRv2: stores position-aware features from the previous frame and concatenates
  them. For 6 cameras x 16x44x256: ~1.7 million values per frame.

These approaches have several problems:
1. **Memory**: Storing dense feature maps is expensive, especially for long histories.
2. **Compute**: Aligning features across frames requires warping or re-encoding.
3. **Scalability**: Memory grows linearly (or worse) with temporal window length.
4. **Redundancy**: Most features describe static background (road, buildings). Only
   a small fraction corresponds to dynamic objects we actually detect.

### 6.2 The Key Insight: Object-Level Propagation

StreamPETR (ICCV 2023) asks a fundamental question: **for detection, do we really need
to propagate entire feature maps? Or can we just propagate the detection results
(queries) themselves?**

Consider: a transformer decoder uses ~900 object queries to detect objects. After
processing a frame, the top queries already encode rich information about detected
objects -- their positions, appearances, and even motion patterns. Why not just pass
these queries to the next frame?

This is far more efficient:
- 256 propagated queries x 256 dimensions = **65,536 values** (65K)
- vs. 200x200x256 BEV features = **10,240,000 values** (10M)

A factor of **157x** less memory for temporal context.

### 6.3 Query Propagation Mechanism: Step by Step

Here is how StreamPETR propagates queries across frames:

**Frame t processing**:
1. Extract image features from all 6 cameras.
2. Generate 3D PE and create position-aware features (same as base PETR).
3. Initialize queries: 644 fresh queries + 256 propagated queries from frame (t-1) = 900 total.
4. Run transformer decoder: queries attend to position-aware features.
5. Each query predicts a 3D bounding box (or "no object").
6. **Selection**: From the 900 queries, select the top-256 by confidence score.
7. **Store**: Save these 256 queries for the next frame.

**Between frames (t -> t+1)**:
1. **Ego-motion compensation**: Each query has an associated 3D reference point.
   Transform these reference points from frame t's coordinate system to frame (t+1)'s:
   ```
   ref_point_t+1 = T_{t+1 <- t} * ref_point_t
   ```
2. **Velocity extrapolation**: Each query also predicts the object's velocity (vx, vy).
   Extrapolate the reference point forward in time:
   ```
   ref_point_t+1 = ref_point_t+1 + velocity * delta_t
   ```
3. The query embeddings themselves are carried forward unchanged (they contain
   appearance and motion memory).

```
  Frame t:                               Frame t+1:
  
  900 queries                            900 queries
  [644 fresh + 256 propagated]           [644 fresh + 256 propagated from t]
       |                                      |
       v                                      v
  Transformer Decoder                    Transformer Decoder
       |                                      |
       v                                      v
  Predictions + confidence               Predictions + confidence
       |                                      |
       v                                      v
  Select top-256 -----> ego-motion -----> Inject as propagated queries
                        + velocity
                        extrapolation
```

### 6.4 Why This Works: The Power of Object Queries

A single object query after decoder processing encodes:
- The object's 3D position (via its reference point)
- The object's appearance (via features gathered through cross-attention)
- The object's motion history (accumulated through propagation)
- The object's class and attributes

Propagating these queries is like passing a detective's complete case file to the next
shift. The next frame's decoder does not need to rediscover the object from scratch --
it already has a strong prior about where the object is, what it looks like, and where
it is going.

### 6.5 Motion-Aware Layer Normalization

StreamPETR introduces a subtle but powerful technique to inject ego-motion awareness
into the transformer without explicit feature warping.

**Standard Layer Normalization**:
```
  LayerNorm(x) = gamma * (x - mean) / std + beta
```
Here gamma and beta are fixed learnable parameters (vectors of dimension 256).

**Motion-Aware Layer Normalization**:
```
  MotionLN(x, M) = gamma(M) * (x - mean) / std + beta(M)
```
Here gamma and beta are **functions of the ego-motion matrix M**:
```
  ego_motion_embedding = MLP_motion(flatten(M))    # 4x4 matrix -> 256-d vector
  gamma(M) = Linear_gamma(ego_motion_embedding)    # 256-d -> 256-d scale
  beta(M) = Linear_beta(ego_motion_embedding)      # 256-d -> 256-d shift
```

**Why this matters**: The ego-motion matrix M tells the model "the vehicle moved 0.5m
forward and turned 2 degrees left since the last frame." By modulating every LayerNorm
in the transformer with this information, the model implicitly knows how to reinterpret
propagated queries in the new coordinate frame -- without explicitly transforming every
feature.

This is more efficient than:
- Warping all BEV features (BEVFormer approach: expensive spatial transformer)
- Re-computing 3D PE for previous features (PETRv2 approach: extra MLP forward pass)

### 6.6 Emergent Tracking Without a Tracker

A remarkable property of StreamPETR: because the same query tracks the same object
across frames (via propagation), you get **object tracking for free**.

In traditional detection + tracking pipelines:
1. Detect objects in each frame independently.
2. Run a separate tracking algorithm (e.g., Hungarian matching, DeepSORT) to associate
   detections across frames.

In StreamPETR:
1. Query #47 detects a car in frame t.
2. Query #47 is propagated to frame t+1 (with position updated by ego-motion + velocity).
3. Query #47 detects the same car in frame t+1.
4. The query identity IS the track identity. No separate tracker needed.

This "tracking by propagation" approach is conceptually similar to TrackFormer and MOTR
for 2D tracking, but StreamPETR achieves it naturally as a byproduct of temporal
query propagation.

---

## 7. Comprehensive Comparison: PETR vs BEVFormer vs DETR3D

| Aspect | DETR3D | PETR / PETRv2 | BEVFormer | StreamPETR |
|--------|--------|---------------|-----------|------------|
| **3D Representation** | None (implicit via projection) | None (implicit via 3D PE in features) | Explicit BEV grid (200x200) | None (implicit in queries) |
| **How 3D info enters features** | 3D ref points projected to 2D, features sampled | 3D PE added to features once | Deformable attention from BEV to image features | 3D PE + propagated query positions |
| **Attention type** | Sampling at projected 2D locations | Global cross-attention | Deformable cross-attention | Global cross-attention |
| **Temporal approach** | None (single-frame) | Concatenate previous features (PETRv2) | BEV-level temporal self-attention | Object query propagation |
| **Temporal memory cost** | N/A | ~1.7M values/frame | ~10M values/frame | ~65K values/frame |
| **GPU memory (R50)** | ~10 GB | ~14 GB | ~18 GB | ~8 GB |
| **Speed (R50, A100)** | ~12 FPS | ~8 FPS (PETRv2) | ~4 FPS | ~30 FPS |
| **Multi-task suitability** | Low (no shared spatial repr.) | Medium (can add BEV head) | High (BEV supports segmentation) | Medium (detection-focused) |
| **Calibration sensitivity** | High (projection errors) | Low (soft matching via attention) | Medium | Low |
| **Strengths** | Simple geometry, no learned depth | Simple architecture, robust | Rich spatial repr., strong multi-task | Very fast, memory-efficient, free tracking |
| **Weaknesses** | Limited receptive field per query | Global attention is O(n^2) | Slow, memory-hungry | Less suited for dense prediction tasks |

### Key Tradeoffs Explained

**BEVFormer vs StreamPETR (speed vs. versatility)**:
BEVFormer's explicit BEV grid is ideal when you need multiple outputs (detection +
segmentation + motion prediction) because all tasks share the same spatial
representation. StreamPETR is 7-8x faster but its object-centric design makes dense
prediction tasks (like BEV segmentation) harder to add.

**DETR3D vs PETR (projection vs. encoding)**:
Both avoid explicit BEV construction, but PETR's approach of encoding position into
features (rather than projecting at runtime) gives better gradient flow and multi-camera
handling. PETR consistently outperforms DETR3D by 2-4 mAP.

**PETR vs StreamPETR (feature-temporal vs. object-temporal)**:
StreamPETR shows that for detection tasks, propagating 256 object queries is far more
efficient than propagating thousands of feature vectors, while achieving better
performance due to the focused, task-specific nature of the temporal information.

---

## 8. Performance Tables with Interpretation

### 8.1 nuScenes Validation Set Results

| Model | Backbone | Image Size | mAP | NDS | mATE | mASE | mAOE | mAVE | FPS |
|-------|----------|-----------|-----|-----|------|------|------|------|-----|
| DETR3D | ResNet-101 | 900x1600 | 34.9 | 42.2 | 0.716 | 0.268 | 0.379 | 0.842 | ~12 |
| PETR | ResNet-50 | 512x1408 | 31.3 | 38.1 | 0.768 | 0.278 | 0.564 | 0.930 | ~10 |
| PETR | ResNet-101 | 512x1408 | 35.7 | 42.1 | 0.710 | 0.270 | 0.490 | 0.885 | ~8 |
| PETR | VoVNet-99 | 512x1408 | 37.8 | 44.2 | 0.680 | 0.267 | 0.453 | 0.860 | ~7 |
| PETRv2 | ResNet-50 | 512x1408 | 34.6 | 42.1 | 0.739 | 0.274 | 0.487 | 0.413 | ~8 |
| PETRv2 | ResNet-101 | 512x1408 | 38.3 | 45.5 | 0.690 | 0.265 | 0.432 | 0.390 | ~6 |
| BEVFormer-S | ResNet-101 | 900x1600 | 37.5 | 44.8 | 0.695 | 0.272 | 0.391 | 0.439 | ~4 |
| BEVFormer-B | ResNet-101 | 900x1600 | 41.6 | 51.7 | 0.673 | 0.274 | 0.372 | 0.394 | ~2 |
| StreamPETR | ResNet-50 | 512x1408 | 38.4 | 44.9 | 0.702 | 0.272 | 0.470 | 0.413 | ~30 |
| StreamPETR | ResNet-101 | 512x1408 | 40.2 | 47.1 | 0.680 | 0.268 | 0.440 | 0.380 | ~20 |
| StreamPETR | VoVNet-99 | 512x1408 | 45.0 | 55.0 | 0.613 | 0.258 | 0.367 | 0.276 | ~15 |
| StreamPETR | ViT-L (EVA02) | 512x1408 | 55.2 | 63.6 | 0.501 | 0.243 | 0.303 | 0.212 | ~8 |

### 8.2 Metric Explanations

- **mAP** (mean Average Precision): Primary detection quality metric. Higher = better.
  Measures how well the model detects and localizes objects across all 10 nuScenes
  classes.
- **NDS** (nuScenes Detection Score): Composite metric = 0.5 * mAP + 0.5 * mean(mATE,
  mASE, mAOE, mAVE, mAAE). Captures overall detection quality including attributes.
- **mATE** (mean Translation Error): How far off the predicted center is from ground
  truth (in meters). Lower = better.
- **mASE** (mean Scale Error): How well the predicted box size matches reality. Lower =
  better.
- **mAOE** (mean Orientation Error): How well the predicted heading matches reality (in
  radians). Lower = better.
- **mAVE** (mean Velocity Error): How well the predicted velocity matches reality (in
  m/s). Lower = better. Single-frame methods have high mAVE because they cannot
  estimate velocity.

### 8.3 Key Observations

**1. Temporal methods dramatically improve velocity estimation**:
- PETR (single-frame): mAVE = 0.930 m/s
- PETRv2 (2-frame temporal): mAVE = 0.413 m/s (55% reduction)
- StreamPETR (streaming temporal): mAVE = 0.413 m/s (same quality, but much faster)

**2. StreamPETR achieves the best speed-accuracy tradeoff**:
- StreamPETR R50 (38.4 mAP, 30 FPS) vs. BEVFormer-S R101 (37.5 mAP, 4 FPS):
  Similar accuracy but 7.5x faster with a smaller backbone.

**3. Backbone matters enormously**:
- StreamPETR with ViT-L reaches 55.2 mAP, which is 16.8 mAP higher than with R50.
  The backbone contributes more to performance than any architectural innovation.

**4. The PETR paradigm scales well**:
- PETR (2022): 31.3 mAP @ 10 FPS
- PETRv2 (2023): 34.6 mAP @ 8 FPS (+3.3 mAP, same paradigm, temporal extension)
- StreamPETR (2023): 38.4 mAP @ 30 FPS (+7.1 mAP vs PETR, 3x faster)
- StreamPETR + ViT-L: 55.2 mAP @ 8 FPS (paradigm supports large models)

---

## 9. Key Takeaways for Practitioners

### 9.1 When to Choose PETR/StreamPETR

Choose **StreamPETR** when:
- Your primary task is 3D object detection (possibly with tracking)
- Inference speed matters (real-time requirement, limited compute budget)
- You want built-in temporal reasoning without heavy memory overhead
- You need tracking output without a separate tracker module

Choose **BEVFormer** instead when:
- You need multiple outputs from the same model (detection + BEV segmentation + motion
  prediction + planning)
- You have generous compute budget and can tolerate 2-4 FPS
- Dense spatial prediction is a first-class requirement

Choose **DETR3D** when:
- You want the simplest possible architecture for prototyping
- You are working on a single-camera setup where global attention is unnecessary
- You need a strong baseline to compare against

### 9.2 Implementation Tips

1. **3D PE computation is the key code path**: The frustum generation and coordinate
   transformation code must be correct. Off-by-one errors in depth bins or incorrect
   intrinsic/extrinsic matrices will silently degrade performance.

2. **Depth range matters**: The detection range (x_min, x_max, y_min, y_max, z_min,
   z_max) and depth range (d_min, d_max, n_bins) must match your deployment scenario.
   Urban driving: 1-60m. Highway: 1-100m+.

3. **Global attention is expensive**: For high-resolution feature maps, PETR's global
   cross-attention has O(N_queries * N_features) complexity. Keep feature map resolution
   reasonable (1/32 or 1/16 of input).

4. **Query propagation needs velocity**: StreamPETR's query propagation works best when
   velocity prediction is accurate. Use temporal supervision (GT velocity) during
   training.

5. **Backbone pre-training is critical**: With small datasets, a well-pretrained
   backbone (e.g., FCOS3D pre-training, or CLIP/EVA02 initialization) provides
   significant gains.

### 9.3 Common Pitfalls

- **Forgetting to update 3D PE when camera parameters change**: If you do data
  augmentation that changes the image (resize, crop), you must update K accordingly
  before computing 3D PE.
- **Incorrect ego-motion for temporal fusion**: The transformation between frames must
  be precise. Use high-quality odometry or SLAM, not just GPS.
- **Over-relying on global attention**: For very long-range detection, global attention
  may spread attention too thin. Consider combining with deformable attention for
  distant objects.

### 9.4 The Big Picture

The PETR paradigm represents a shift from **explicit geometric computation** to
**implicit geometric encoding**:

```
  Traditional (DETR3D):    Geometry happens during attention (projection)
  PETR paradigm:           Geometry happens before attention (encoding)
  
  Result: Simpler decoder, better gradients, faster inference
```

StreamPETR extends this philosophy to temporal reasoning:

```
  Traditional (BEVFormer): Temporal happens at feature level (dense, expensive)
  StreamPETR:              Temporal happens at object level (sparse, cheap)
  
  Result: 7-8x faster, free tracking, comparable accuracy
```

### 9.5 The Evolution of Ideas

```
  DETR3D (CoRL 2021)
    "Use 3D reference points, project to 2D to sample features"
        |
        | Insight: projection is expensive and hurts gradients
        v
  PETR (ECCV 2022)
    "Encode 3D position INTO features, let attention handle the rest"
        |
        | Insight: need temporal context for velocity
        v
  PETRv2 (ICCV 2023)
    "Add temporal feature alignment + 2D PE + multi-task"
        |
        | Insight: propagating features is wasteful; propagate queries instead
        v
  StreamPETR (ICCV 2023)
    "Propagate object queries, not features. Motion-aware LayerNorm."
        |
        | Result: 30 FPS real-time detection with free tracking
        v
  [Future work]
    Sparse queries + end-to-end planning?
```

---

## 10. References

1. Liu, Y., Wang, T., Zhang, X., Sun, J. "PETR: Position Embedding Transformation for
   Multi-View 3D Object Detection." ECCV 2022.

2. Liu, Y., Jia, J., Zhang, X., et al. "PETRv2: A Unified Framework for 3D Perception
   from Multi-Camera Images." ICCV 2023.

3. Wang, S., Liu, Y., Wang, T., et al. "Exploring Object-Centric Temporal Modeling for
   Efficient Multi-View 3D Object Detection (StreamPETR)." ICCV 2023.

4. Wang, Y., Guizilini, V., Zhang, T., et al. "DETR3D: 3D Object Detection from
   Multi-view Images via 3D-to-2D Queries." CoRL 2021.

5. Li, Z., Wang, W., Li, H., et al. "BEVFormer: Learning Bird's-Eye-View
   Representation from Multi-Camera Images via Spatiotemporal Transformers." ECCV 2022.

6. Philion, J., Fidler, S. "Lift, Splat, Shoot: Encoding Images From Arbitrary Camera
   Rigs by Implicitly Unprojecting to 3D." ECCV 2020.

7. Huang, J., Huang, G., Zhu, Z., et al. "BEVDet: High-Performance Multi-Camera 3D
   Object Detection in Bird-Eye-View." arXiv 2021.

8. Li, Y., Ge, Z., Yu, G., et al. "BEVDepth: Acquisition of Reliable Depth for
   Multi-view 3D Object Detection." AAAI 2023.

9. Carion, N., Massa, F., Synnaeve, G., et al. "End-to-End Object Detection with
   Transformers (DETR)." ECCV 2020.

10. Zeng, Y., Da, T., Hu, X., et al. "MOTR: End-to-End Multiple-Object Tracking with
    Transformer." ECCV 2022.

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| K | Camera intrinsic matrix (3x3) |
| T_ego_cam | Camera-to-ego extrinsic transformation (4x4) |
| (u, v) | Pixel coordinates in the image |
| d | Depth value (meters from camera) |
| D | Number of depth bins |
| H_f, W_f | Feature map height and width |
| PE_3d | 3D position embedding (256-d vector) |
| F_image | Image backbone features |
| MLP | Multi-Layer Perceptron |
| mAP | mean Average Precision |
| NDS | nuScenes Detection Score |
| mAVE | mean Average Velocity Error |
| BEV | Bird's-Eye-View |
| R, t | Rotation matrix and translation vector |

## Appendix B: Coordinate Systems in Autonomous Driving

Understanding coordinate systems is essential for working with PETR:

```
  Camera coordinate system:        Ego-vehicle coordinate system:
  
         Z (forward/depth)                   X (forward)
         |                                   |
         |                                   |
         |______ X (right)                   |______ Y (left)
        /                                   /
       /                                   /
      Y (down)                            Z (up)
```

- **Camera coordinates**: X points right, Y points down, Z points forward (into the
  scene). This is the standard computer vision convention.
- **Ego-vehicle coordinates**: X points forward (direction of travel), Y points left,
  Z points up. This is the standard robotics/automotive convention.

The extrinsic matrix T_ego_cam converts from camera to ego-vehicle coordinates.
Multiple cameras each have their own extrinsic matrix. The intrinsic matrix K is
specific to each camera model and does not change between frames.

---

*End of document. Total coverage: the fundamental depth ambiguity problem, three
paradigms for camera-based 3D detection, PETR's 3D PE mechanism from first principles,
PETRv2 temporal/multi-task extensions, StreamPETR object-centric temporal modeling,
comprehensive comparison tables, and practical guidance for implementation.*
