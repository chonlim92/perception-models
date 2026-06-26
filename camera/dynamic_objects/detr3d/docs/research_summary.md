# DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

A tutorial-style research summary for newcomers to autonomous driving perception
and transformer-based 3D object detection.

Paper: "DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries"
Authors: Yue Wang, Vitor Guizilini, Timanogo Zhang, Yilun Wang, Hang Zhao, Justin Solomon
Venue: Conference on Robot Learning (CoRL) 2022

---

## 1. The Detection Problem

### What Is 3D Object Detection from Cameras?

An autonomous vehicle needs to understand the 3D world around it: where are the
cars, pedestrians, cyclists, and other objects? It needs to know not just that
"there is a car" but precisely WHERE in 3D space that car is -- its center
position (x, y, z), its dimensions (length, width, height), its orientation
(which way it faces), and ideally its velocity.

LiDAR sensors provide direct 3D measurements (point clouds), making this task
relatively straightforward. But LiDAR is expensive and sparse. Cameras are cheap,
dense, and carry rich semantic information -- but they lose depth.

The fundamental question: can we recover 3D object locations from 2D images alone?

### Why Is This Hard?

Two fundamental ambiguities make camera-based 3D detection challenging:

**Depth Ambiguity:**
A camera captures a 2D projection of the 3D world. A single pixel could
correspond to ANY point along a ray extending from the camera center:

```
                    Near object (small car, close)
                   /
  Camera --------*----------*------------------> Ray into scene
  [lens]          \
                   Far object (large truck, far away)

  Both project to the SAME pixel location!
```

Without additional cues (context, ground plane, known sizes), we cannot tell if a
small nearby object or a large distant object produced a given image region.

**Scale Ambiguity:**
Related to depth ambiguity: an object's apparent size in the image depends on
both its true physical size AND its distance from the camera:

```
  Image plane:     [  car  ]     <- 100 pixels wide

  Possibility A:   Small car at 10m distance
  Possibility B:   Large truck at 30m distance
  Both produce the same 100-pixel bounding box!
```

**Multi-view Complication:**
An autonomous vehicle typically uses 6 cameras covering 360 degrees. Objects may
appear in one or multiple views. We must fuse information across these views
consistently -- the same car seen by the front-left and front cameras must be
detected exactly once, not twice.

```
       Front-Left    Front    Front-Right
           \          |          /
            \         |         /
             +--------+--------+
             |                 |
  Left ------|      EGO       |------ Right
             |                 |
             +--------+--------+
            /         |         \
           /          |          \
       Rear-Left     Rear     Rear-Right
```

---

## 2. DETR Recap for Newcomers

Before understanding DETR3D, we need to understand DETR -- the 2D detection
transformer that started it all.

### What Is DETR?

DETR (DEtection TRansformer) was introduced by Carion et al. at ECCV 2020. It
reimagined object detection as a DIRECT SET PREDICTION problem, removing many
hand-designed components that plagued prior detectors (anchors, non-maximum
suppression, region proposals).

Traditional detectors work like this:
1. Generate thousands of candidate boxes (anchors or proposals)
2. Classify each candidate
3. Remove duplicates via Non-Maximum Suppression (NMS)

DETR works like this:
1. Feed image through a CNN backbone
2. Feed features through a transformer encoder-decoder
3. Output a FIXED SET of N predictions directly -- done.

### Set Prediction: A Fixed Number of Output Slots

DETR always outputs exactly N predictions (e.g., N=100 for COCO). Each prediction
is either an object (class + bounding box) or "no object" (background). Because
the model outputs a fixed-size SET with no duplicates, there is no need for NMS.

```
  Input Image --> CNN Backbone --> Transformer --> N predictions
                                                   |
                                                   +-- Pred 1: car, box=[...]
                                                   +-- Pred 2: person, box=[...]
                                                   +-- Pred 3: no object
                                                   +-- Pred 4: no object
                                                   +-- ...
                                                   +-- Pred N: no object
```

### Hungarian Matching: Optimal Assignment During Training

During training, we need to assign each prediction to a ground-truth object (or
to "no object"). This is done via the Hungarian algorithm -- an optimal bipartite
matching that minimizes the total assignment cost:

```
  Predictions:       Ground Truth:
  +---------+        +---------+
  | Pred 1  |------->| GT Car  |    cost = class_loss + box_loss
  | Pred 2  |------->| GT Ped  |
  | Pred 3  |------->| (none)  |    assigned "no object"
  | ...     |        |         |
  | Pred N  |------->| (none)  |
  +---------+        +---------+

  Hungarian algorithm finds the minimum-cost one-to-one matching.
```

The matching cost considers:
- Classification probability (does prediction match the GT class?)
- Bounding box similarity (L1 distance + generalized IoU)

Once matched, the loss is computed only on the assigned pairs.

### Object Queries: Learnable Detection Slots

The key mechanism enabling set prediction is "object queries." These are N
learnable embedding vectors that are fed as input to the transformer decoder.
Each query acts as a "slot" that learns to detect one object:

```
  Object Query 1: "I will look for objects in the top-left region"
  Object Query 2: "I will look for large objects in the center"
  Object Query 3: "I will look for small objects near edges"
  ...
  Object Query N: "I will detect the Nth object if it exists"
```

Through training, each query specializes: some queries become responsible for
certain spatial regions, certain object sizes, or certain categories. The queries
attend to the image features via cross-attention, gathering the information they
need to make their prediction.

### Why DETR Was Revolutionary

Before DETR:
- Detectors needed hand-designed anchors (aspect ratios, scales, densities)
- NMS was a non-differentiable post-processing step
- The pipeline was complex: backbone -> RPN -> RoI pooling -> classification head

After DETR:
- End-to-end trainable: loss goes directly from predictions to ground truth
- No anchors: the model learns what to look for
- No NMS: set prediction inherently avoids duplicates
- Conceptually simple: backbone + transformer + prediction heads

---

## 3. DETR3D's Key Insight: 3D Reference Points Projected to Cameras

### The Core Idea

Most camera-based 3D detectors before DETR3D worked in one of two ways:

1. **Lift 2D to 3D:** Estimate depth for every pixel, then "lift" 2D features
   into 3D space to form a volumetric representation (e.g., LSS, BEVDet).

2. **Work in 2D, predict 3D:** Detect objects in 2D images, then regress their
   3D properties (e.g., FCOS3D).

DETR3D introduces a third paradigm: **operate in 3D, sample from 2D.**

The idea: maintain learnable 3D reference points (positions in 3D world space).
For each reference point, PROJECT it down to each camera's image plane using
known camera geometry. Then SAMPLE the 2D image features at those projected
locations. The sampled features tell the model what the image "sees" at that 3D
location.

```
  3D World Space                     2D Image Planes
  ==============                     ================

      * P_3d                          Camera 1 (Front):
      |                               +------------------+
      | project using                 |        x <------ sampled feature
      | K * [R|t]                     |                  |
      |                               +------------------+
      v
  p_2d = K * [R|t] * P_3d            Camera 2 (Front-Left):
                                      +------------------+
                                      |   x <----------- sampled feature
                                      |                  |
                                      +------------------+

  The 3D point P_3d is projected onto visible cameras.
  Features are sampled at projected locations via bilinear interpolation.
  Sampled features are aggregated (averaged) to form the query's input.
```

### Why NO Explicit BEV Is Needed

Bird's Eye View (BEV) methods construct a dense 2D grid representing the scene
from above. Every cell in this grid must be filled with features -- even cells
corresponding to empty space. This is expensive.

DETR3D skips BEV entirely:

```
  BEV Methods:                        DETR3D:
  ============                        =======

  +---+---+---+---+---+              Only 900 query points
  | . | . | . | . | . |              in 3D space:
  +---+---+---+---+---+
  | . | . | * | . | . |                  *     *
  +---+---+---+---+---+                     *
  | . | * | . | . | . |                *        *
  +---+---+---+---+---+                   *  *
  | . | . | . | . | . |                *       *
  +---+---+---+---+---+
                                      (sparse, only where objects might be)
  200x200 = 40,000 cells
  (dense, mostly empty)
```

DETR3D's queries directly ask: "Is there an object at THIS 3D location?" Each
query refines its 3D reference point across decoder layers, homing in on actual
object centers. This is fundamentally sparse and efficient.

---

## 4. Feature Sampling Mechanism in Detail

### Step 1: Geometric Projection

Each object query has an associated 3D reference point P_3d = (X, Y, Z) in the
ego-vehicle coordinate frame. To sample image features, we project this point
onto each camera's image plane:

```
  p_2d = K * [R | t] * P_3d
```

Where:
- P_3d is the 3D point in homogeneous coordinates [X, Y, Z, 1]^T
- [R | t] is the 4x4 extrinsic matrix (world-to-camera transformation)
  - R: 3x3 rotation matrix
  - t: 3x1 translation vector
- K is the 3x3 intrinsic matrix (camera parameters)
  - fx, fy: focal lengths
  - cx, cy: principal point

Expanding the math:

```
  [u]       [fx  0  cx] [r11 r12 r13 tx] [X]
  [v] = s * [ 0 fy  cy] [r21 r22 r23 ty] [Y]
  [1]       [ 0  0   1] [r31 r32 r33 tz] [Z]
                                          [1]

  where s is a scaling factor (depth), and (u, v) are pixel coordinates.
```

We check visibility: the point must have positive depth (in front of camera) and
the projected pixel (u, v) must lie within image bounds.

### Step 2: Bilinear Interpolation

The projected pixel coordinates (u, v) are generally NOT at integer pixel
locations. We use bilinear interpolation to sample features at sub-pixel
precision from the feature map:

```
  Feature map (H x W):

  +-------+-------+
  | F(i,j)|F(i,j+1)|     Projected point (u,v) lands here: *
  +-------+-------+
  |F(i+1,j)|F(i+1,j+1)|
  +-------+-------+

  Bilinear interpolation:
  F(u,v) = (1-a)(1-b)*F(i,j) + a*(1-b)*F(i,j+1)
          + (1-a)*b*F(i+1,j) + a*b*F(i+1,j+1)

  where a = u - floor(u), b = v - floor(v)
```

This gives us a feature vector at the exact projected location, even when it
falls between grid cells. Multi-scale features from an FPN (Feature Pyramid
Network) are sampled for richer representation.

### Step 3: Multi-View Aggregation

A 3D reference point may be visible in multiple cameras (e.g., a car at the
boundary between front and front-left views). DETR3D projects the point onto ALL
cameras, checks visibility, and AVERAGES the sampled features:

```
  P_3d visible in Camera 1 and Camera 3:

  Camera 1 feature: f_1 = [0.2, 0.5, 0.1, ...]
  Camera 2 feature: (not visible -- point behind camera)
  Camera 3 feature: f_3 = [0.3, 0.4, 0.2, ...]

  Aggregated feature = mean(f_1, f_3) = [0.25, 0.45, 0.15, ...]
```

This simple averaging provides multi-view fusion without complex attention
mechanisms or explicit feature warping.

### Step 4: Iterative Refinement Across Decoder Layers

DETR3D uses 6 transformer decoder layers. Each layer:
1. Self-attention: queries attend to each other (reason about inter-object
   relationships, suppress duplicates)
2. Feature sampling: project current reference points, sample features
3. Predict offset: each query predicts a REFINEMENT to its reference point
4. Update: reference point moves closer to the true object center

```
  Layer 1: Initial reference point   [-----*----->]  rough estimate
  Layer 2: Refined reference point   [--------*-->]  better
  Layer 3: Further refined           [---------*->]  closer
  Layer 4:                           [----------*]   accurate
  Layer 5:                           [----------*]   fine-tuned
  Layer 6: Final reference point     [----------*]   final prediction
```

This iterative process is critical -- early layers do coarse localization while
later layers do fine-grained refinement.

---

## 5. Advantages and Disadvantages

### Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Architectural simplicity** | No BEV encoder, no depth network, no voxelization. Just backbone + transformer decoder with geometric projection. |
| **No explicit depth estimation** | Avoids the ill-posed monocular depth estimation problem entirely. Depth is implicitly handled by the 3D reference point positions. |
| **Memory efficient** | Only 900 sparse queries vs. 200x200=40,000 dense BEV cells. Significantly lower GPU memory. |
| **No quantization artifacts** | BEV methods discretize space into a grid, losing precision. DETR3D operates in continuous 3D space. |
| **No NMS needed** | Set prediction with Hungarian matching eliminates post-processing. |
| **Fast inference** | Sparse computation pattern: O(N_queries x N_cameras) instead of O(BEV_H x BEV_W x N_cameras). |
| **Elegant formulation** | The 3D-to-2D projection is mathematically clean and easy to implement. Camera calibration is used directly. |

### Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| **No spatial BEV representation** | Cannot easily share features with downstream tasks (lane detection, motion planning) that benefit from a structured BEV map. |
| **Single-frame only** | No built-in temporal reasoning. Cannot leverage motion cues across frames for better depth estimation or velocity prediction. |
| **Weaker depth localization** | Without explicit depth supervision or BEV structure, depth estimates are less precise -- reflected in higher mATE scores. |
| **Limited query count** | 900 queries cap the number of detectable objects. Dense scenes or distant small objects may be missed. |
| **Simple feature aggregation** | Averaging multi-view features loses information about which view is more reliable or informative. |
| **No attention over image context** | Sampling at a single projected point misses contextual information from the surrounding image region. |

---

## 6. Comparison Table: DETR3D vs BEVFormer vs PETR

| Aspect | DETR3D | BEVFormer | PETR |
|--------|--------|-----------|------|
| **Core mechanism** | 3D-to-2D geometric projection + sampling | Dense BEV grid with spatial cross-attention | 3D position embedding added to 2D features |
| **3D representation** | None (implicit via sparse queries) | Explicit dense BEV feature map (200x200) | None (implicit via 3D-aware features) |
| **How features are obtained** | Bilinear sampling at projected 3D points | Deformable attention from BEV cells to images | Global cross-attention with 3D PE |
| **Geometric prior** | Hard geometric projection (camera matrices) | Learned deformable offsets around projected references | 3D coordinate encoding as position embedding |
| **Temporal modeling** | None (single-frame) | Temporal self-attention on BEV features | StreamPETR variant adds temporal propagation |
| **Multi-task friendly** | Low (no shared spatial representation) | High (BEV map usable for segmentation, planning) | Low (no shared spatial representation) |
| **Computational cost** | Low (sparse: 900 queries x 6 cameras) | High (dense: 200x200 BEV x 6 cameras) | Medium (global attention over all image features) |
| **Memory footprint** | Low | High | Medium |
| **Depth handling** | Bypassed entirely | Learned via attention offsets | Encoded in position embeddings |
| **nuScenes NDS** | 0.479 (w/ CBGS) | 0.517 (base) / 0.569 (temporal) | 0.455 (PETRv1) / 0.504 (PETRv2) |
| **nuScenes mAP** | 0.412 (w/ CBGS) | 0.416 (base) / 0.481 (temporal) | 0.391 (PETRv1) / 0.421 (PETRv2) |
| **Decoder layers** | 6 | 6 | 6 |
| **Object queries** | 900 | 900 | 900 |
| **NMS required** | No | No | No |
| **Year** | 2021 (CoRL 2022) | 2022 (ECCV 2022) | 2022 (ECCV 2022) |

**Key distinctions in plain language:**

- **DETR3D** says: "I have 3D points. I project them to cameras and read what
  the image says at those locations."
- **BEVFormer** says: "I build a full bird's-eye-view map by querying every
  camera from every BEV cell. This map captures the whole scene."
- **PETR** says: "I encode 3D position information INTO the image features
  themselves. Then queries can attend globally without explicit projection."

---

## 7. Results and Impact

### Key Results on nuScenes

| Configuration | Backbone | NDS | mAP | mATE | mASE | mAOE |
|---------------|----------|-----|-----|------|------|------|
| DETR3D (no CBGS) | ResNet-101 | 0.425 | 0.346 | 0.716 | 0.268 | 0.379 |
| DETR3D (w/ CBGS) | ResNet-101 | 0.479 | 0.412 | 0.641 | 0.255 | 0.394 |
| DETR3D | VoVNet-99 | 0.479 | 0.412 | 0.641 | 0.255 | 0.394 |

Metric definitions:
- **NDS** (nuScenes Detection Score): composite metric combining mAP with
  localization errors (translation, scale, orientation, velocity, attributes)
- **mAP** (mean Average Precision): detection accuracy across 10 classes
- **mATE** (mean Average Translation Error): distance error in meters (lower=better)
- **mASE** (mean Average Scale Error): size estimation error (lower=better)
- **mAOE** (mean Average Orientation Error): heading angle error (lower=better)

### Training Configuration

- **Object queries:** 900 (more than max objects in any nuScenes scene)
- **Decoder layers:** 6 (iterative refinement)
- **Losses:** Focal loss (classification) + L1 loss (bounding box regression)
- **Matching:** Hungarian algorithm for one-to-one GT assignment
- **CBGS:** Class-Balanced Grouping and Sampling -- oversamples rare classes
  (construction vehicles, barriers, traffic cones) to combat severe imbalance
- **Data augmentation:** Standard image augmentation (flip, resize, color jitter)
- **Backbone:** ResNet-101 with DCN (Deformable Convolutions) and FPN

### Ablation Highlights

- Increasing decoder layers 1->6: +5 NDS improvement (iterative refinement is key)
- Multi-scale FPN features: +2 NDS vs. single-scale
- CBGS sampling: +5.4 NDS and +6.6 mAP (class imbalance is a major challenge)
- 3D-to-2D projection vs. naive concatenation: large margin improvement

### Impact on the Field

DETR3D was a foundational work that established several paradigms now standard in
camera-based 3D detection:

1. **Proved camera-only viability:** Showed competitive 3D detection without
   LiDAR or explicit depth supervision, inspiring an entire line of research.

2. **Query-based 3D detection:** The idea of learnable 3D queries attending to
   image features became the dominant paradigm (BEVFormer, PETR, StreamPETR,
   Far3D, SparseBEV all build on this).

3. **Geometry-guided attention:** Using camera calibration matrices to guide
   where queries attend in images (rather than learning attention from scratch)
   proved both effective and data-efficient.

4. **Simplicity as strength:** The clean architecture made DETR3D highly
   extensible. Follow-up works added:
   - Temporal fusion (StreamPETR, BEVFormer)
   - Better feature aggregation (deformable attention in BEVFormer)
   - Depth supervision (BEVDepth, BEVStereo)
   - LiDAR-camera fusion (TransFusion, BEVFusion)

5. **Industry adoption:** The sparse query paradigm is computationally friendly
   for deployment on autonomous vehicle hardware, where memory and latency
   budgets are strict.

---

## 8. References

1. Wang, Y., Guizilini, V., Zhang, T., Wang, Y., Zhao, H., & Solomon, J.
   (2022). DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D
   Queries. Conference on Robot Learning (CoRL).

2. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko,
   S. (2020). End-to-End Object Detection with Transformers (DETR). ECCV.

3. Li, Z., Wang, W., Li, H., Xie, E., Sima, C., Lu, T., Qiao, Y., & Dai, J.
   (2022). BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera
   Images via Spatiotemporal Transformers. ECCV.

4. Liu, Y., Wang, T., Zhang, X., & Sun, J. (2022). PETR: Position Embedding
   Transformation for Multi-View 3D Object Detection. ECCV.

5. Philion, J., & Fidler, S. (2020). Lift, Splat, Shoot: Encoding Images from
   Arbitrary Camera Rigs by Implicitly Unprojecting to 3D. ECCV.

6. Huang, J., Huang, G., Zhu, Z., Ye, Y., & Du, D. (2021). BEVDet: High-
   Performance Multi-Camera 3D Object Detection in Bird-Eye-View. arXiv.

7. Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q.,
   Krishnan, A., Pan, Y., Baldan, G., & Beijbom, O. (2020). nuScenes: A
   Multimodal Dataset for Autonomous Driving. CVPR.

---

## Summary Diagram: DETR3D End-to-End Pipeline

```
  Multi-view Images (6 cameras)
       |
       v
  +-------------------+
  | CNN Backbone      |   (ResNet-101 + FPN)
  | (shared weights)  |
  +-------------------+
       |
       v
  Multi-scale Feature Maps (one per camera)
       |                                    +---------------------------+
       |                                    | Learnable Object Queries  |
       |                                    | (900 queries, 256-dim)    |
       |                                    +---------------------------+
       |                                                 |
       v                                                 v
  +------------------------------------------------------------------+
  |                    Transformer Decoder (x6 layers)                |
  |                                                                  |
  |  For each layer:                                                 |
  |    1. Self-attention among 900 queries                           |
  |    2. Each query predicts a 3D reference point                   |
  |    3. Project reference point to all cameras: p = K*[R|t]*P      |
  |    4. Bilinear-sample features at projected locations            |
  |    5. Average features from visible cameras                      |
  |    6. Update query with sampled features                         |
  |    7. Predict reference point offset (iterative refinement)      |
  |                                                                  |
  +------------------------------------------------------------------+
       |
       v
  +-------------------+
  | Prediction Heads  |
  |  - Class (focal)  |
  |  - 3D Box (L1)    |
  |  - Velocity       |
  +-------------------+
       |
       v
  900 predictions --> Hungarian Matching --> Loss
  (inference: filter by confidence threshold)
```

---

End of document.
