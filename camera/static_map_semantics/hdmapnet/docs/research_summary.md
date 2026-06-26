# HDMapNet: Online HD Map Construction

## Paper Reference

- **Title:** HDMapNet: An Online HD Map Construction and Evaluation Framework
- **Authors:** Qi Li, Yue Wang, Yilun Wang, Hang Zhao
- **Venue:** ICRA 2022
- **Institution:** MIT, Tsinghua University

---

## 1. What is Online HD Map Construction?

### Why Self-Driving Cars Need HD Maps

A self-driving car needs to know the structure of the road far beyond what
simple GPS can tell it. HD (high-definition) maps encode:

- **Lane dividers** -- dashed, solid, double-yellow lines
- **Road boundaries** -- curbs, guardrails, road edges
- **Pedestrian crossings** -- zebra stripes, crosswalk regions
- **Traffic signs/signals** -- stop signs, yield, traffic lights (position)
- **Connectivity** -- which lane connects to which at an intersection

Without this information, the planner cannot decide: "Am I allowed to change
lanes here?", "Where does the road curve ahead?", or "Where must I stop?"

```
  Traditional Perception Stack:
  +-----------+     +----------+     +----------+
  | Cameras / | --> | Detect   | --> | Planner  | --> actuators
  | LiDAR     |     | objects  |     | (needs   |
  +-----------+     +----------+     |  MAP!)   |
                                     +----------+
```

### Offline Mapping vs. Online Construction

```
  OFFLINE (Survey Vehicle)               ONLINE (Onboard, Real-Time)
  ========================               ===========================
  - Fleet of mapping cars               - The ego vehicle itself
  - Expensive LiDAR rigs                - 6 surround-view cameras
  - Centimeter-accurate SLAM            - CNN/Transformer inference
  - Months to map a city               - Runs every frame (~10 Hz)
  - Static: stale within weeks          - Always up-to-date
  - Requires pre-built map delivery     - No map dependency at test time
```

| Dimension         | Offline                  | Online                        |
|-------------------|--------------------------|-------------------------------|
| Sensor cost       | $500K+ mapping rig       | $2K camera suite              |
| Freshness         | Weeks/months stale       | Real-time                     |
| Coverage          | Only mapped roads        | Anywhere the car drives       |
| Accuracy          | cm-level (when fresh)    | 10-30 cm typical              |
| Compute           | Offline cluster          | Onboard GPU (real-time)       |
| Failure mode      | Out-of-date map          | Perception errors             |

**The core insight of HDMapNet:** We can train a neural network to produce
a local HD map "on-the-fly" from surround-view camera images, eliminating
the dependency on pre-built offline maps.

---

## 2. Camera-to-BEV View Transforms

The fundamental challenge: cameras produce images in **perspective view**,
but planning operates in **Bird's Eye View (BEV)**. We need a transform
from 2D image pixels to a 2D grid on the ground plane.

```
  Camera Image (perspective)          BEV Grid (top-down)
  +---------------------------+       +------------------+
  |        sky                |       |   . . . . . .   |
  |    /         \            |       |   . . L . . .   |  L = lane
  |   /  lane     \           |  -->  |   . . L . . .   |
  |  /   lines     \          |       |   . . L . . .   |
  | /________________\        |       |   . . . . . .   |
  +---------------------------+       +------------------+
```

### 2.1 IPM (Inverse Perspective Mapping)

**Core idea:** If the ground is flat, there is an exact geometric
relationship (a homography) between image pixels and ground-plane points.

#### Mathematical Formulation

A camera with intrinsic matrix K and extrinsic [R|t] projects a 3D world
point X = (X, Y, Z, 1)^T onto image pixel x = (u, v, 1)^T:

```
  s * x = K * [R | t] * X
```

For points on the ground plane Z = 0, the third column of R drops out:

```
  s * [u]     [f_x  0   c_x] [r11  r12  t_x] [X]
      [v]  =  [0   f_y  c_y] [r21  r22  t_y] [Y]
      [1]     [0    0    1  ] [r31  r32  t_z] [1]
```

This gives us a 3x3 homography matrix H:

```
  H = K * [r1 | r2 | t]

  where r1, r2 are the first two columns of R.
```

The inverse mapping (BEV -> image) is simply H^{-1}:

```
  x_bev = H^{-1} * x_img
```

#### IPM Algorithm

```
  For each cell (X, Y) in the BEV grid:
    1. Compute corresponding pixel: (u, v) = H * (X, Y, 1)^T
    2. Sample feature/color from the image at (u, v)
    3. Place it in the BEV cell
```

#### Strengths and Weaknesses

```
  STRENGTHS:                         WEAKNESSES:
  + Exact when ground IS flat        - FAILS on hills, ramps, bridges
  + No learned parameters            - 3D objects (cars, poles) get
  + Very fast (just a warp)            smeared/distorted
  + Works with any camera config     - Cannot handle occlusions
                                     - No depth reasoning at all
```

**Failure case visualization:**

```
  Side view of a slope:

       actual road          IPM assumes this
       /                    __________________
      /                     (flat ground)
     /
    /

  Result: lane markings on the slope get projected to WRONG BEV positions.
```

### 2.2 LSS (Lift-Splat-Shoot)

**Paper:** "Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs"
(Philion & Fidler, ECCV 2020)

**Core idea:** Instead of assuming flat ground, predict a depth distribution
for every pixel, "lift" image features into a 3D point cloud, then "splat"
(sum-pool) them onto the BEV grid.

#### Step 1: Lift

For each pixel (u, v) in the image feature map, the network predicts a
discrete depth distribution alpha(d) over D depth bins:

```
  alpha(u,v,d) = softmax(depth_net(feature[u,v]))    for d in {d_1, ..., d_D}
```

Each pixel's feature c(u,v) is then "lifted" to D 3D points along its ray:

```
  c_3d(u, v, d) = alpha(u, v, d) * c(u, v)
```

The 3D position of each point is computed by unprojecting:

```
  P(u, v, d) = d * K^{-1} * [u, v, 1]^T
```

Then transformed to ego-vehicle coordinates via the camera extrinsic:

```
  P_ego = R^{-1} * (P_cam - t)
```

#### Step 2: Splat

The lifted 3D points form a "frustum" point cloud. We discretize the BEV
plane into a grid and sum-pool all points that fall into each cell:

```
  BEV[i, j] = SUM over all points P where
               floor(P.x / res) == i  AND  floor(P.y / res) == j
              of c_3d(P)
```

In practice, this is implemented efficiently with "pillar pooling" --
sorting points by their BEV cell index and using cumulative sums.

#### Step 3: Shoot (optional)

Apply a 2D CNN on the BEV feature map for downstream tasks (motion
planning, segmentation, detection).

#### Full Pipeline Diagram

```
  +--------+    +----------+    +---------+    +--------+    +----------+
  | Camera |    | Backbone |    | Depth   |    | Lift   |    | Splat    |
  | Images | -> | (ResNet/ | -> | Network | -> | to 3D  | -> | onto BEV |
  | N views|    | EfficNet)|    | per-pix |    | frustum|    | grid     |
  +--------+    +----------+    +---------+    +--------+    +----------+
                                                                  |
                                                                  v
                                                             +---------+
                                                             | BEV     |
                                                             | Feature |
                                                             | Map     |
                                                             +---------+
```

#### Comparison: IPM vs LSS

```
  +----------------+-------------------+----------------------------+
  | Property       | IPM               | LSS                        |
  +----------------+-------------------+----------------------------+
  | Ground plane   | REQUIRED (flat)   | Not required               |
  | 3D objects     | Distorted/smeared | Correctly placed in BEV    |
  | Learned?       | No (geometric)    | Yes (depth network)        |
  | Compute cost   | Very low          | Moderate-high              |
  | Accuracy       | Good on highways  | Good everywhere            |
  | Training data  | None needed       | Needs depth supervision    |
  |                |                   | (or self-supervised depth) |
  +----------------+-------------------+----------------------------+
```

---

## 3. Multi-Task Learning in HDMapNet

HDMapNet's key architectural insight: after computing a BEV feature map
(using either IPM or LSS), apply **three parallel heads** that together
enable vectorized map construction.

```
  BEV Feature Map (H_bev x W_bev x C)
         |
         +---> Semantic Segmentation Head ---> per-pixel class
         |
         +---> Instance Embedding Head ------> per-pixel embedding vector
         |
         +---> Direction Prediction Head ----> per-pixel angle/direction
```

### 3.1 Semantic Segmentation Head

**Goal:** For each BEV cell, predict which map element class it belongs to.

**Classes (on nuScenes):**
- Lane divider (solid, dashed)
- Pedestrian crossing
- Road boundary (curb)
- Background (no map element)

**Architecture:** A small 2D CNN (or 1x1 conv stack) applied to the BEV
feature map, producing a (H x W x num_classes) tensor.

**Loss function:** Standard cross-entropy with class weighting (map elements
are sparse -- most cells are background):

```
  L_seg = - (1/N) * SUM_i [ w_{y_i} * log(p_i[y_i]) ]

  where:
    y_i = ground truth class for cell i
    p_i = softmax prediction for cell i
    w_c = class weight (higher for rare classes)
```

**Visualization:**

```
  BEV Semantic Map (top-down view of a T-intersection):

  . . . . B B B B B B B . . . .     B = road boundary
  . . . . . . . . . . . . . . .     L = lane divider
  . . . . . . L . . . . . . . .     P = pedestrian crossing
  . . . . . . L . . . . . . . .     . = background
  P P P P P P L P P P P P P P P
  . . . . . . L . . . . . . . .
  . . . . . . L . . . . . . . .
  . . . . B B B B B B B . . . .
```

### 3.2 Instance Embedding Head

**Goal:** Pixels belonging to the same map element instance should have
similar embeddings; pixels from different instances should have distant
embeddings.

**Why we need this:** Semantic segmentation alone cannot distinguish
between two parallel lane dividers -- they have the same class. We need
instance-level discrimination to vectorize them separately.

**Architecture:** A CNN head that outputs a D-dimensional embedding vector
e_i for each BEV cell.

**Loss function:** Discriminative loss (from "Semantic Instance Segmentation
with a Discriminative Loss Function", De Brabandere et al.):

```
  L_instance = L_pull + L_push

  L_pull = (1/K) * SUM_{k=1}^{K} (1/N_k) * SUM_{i in S_k}
           [max(0, ||e_i - mu_k|| - delta_pull)]^2

  L_push = (1/K(K-1)) * SUM_{k_a != k_b}
           [max(0, delta_push - ||mu_{k_a} - mu_{k_b}||)]^2

  where:
    K = number of instances
    S_k = set of pixels belonging to instance k
    mu_k = mean embedding of instance k
    delta_pull = pull margin (e.g., 0.5)
    delta_push = push margin (e.g., 1.5)
```

**Intuition:**

```
  Embedding Space (2D slice):

       o o         x x           L_pull: pulls o's together,
      o   o       x   x                   pulls x's together
       o o         x x
                                  L_push: pushes cluster centers apart
    |<--- delta_push --->|
```

### 3.3 Direction Prediction Head

**Goal:** For each BEV cell that contains a map element, predict the local
direction (tangent angle) of that element at that location.

**Why we need this:** During post-processing, direction helps us:
1. Connect fragmented segments correctly
2. Determine the ordering of points along a polyline
3. Resolve ambiguities at intersections where lines cross

**Architecture:** A CNN head predicting a 2D unit vector (cos theta, sin theta)
for each cell:

```
  d_i = (cos(theta_i), sin(theta_i))
```

**Loss function:** Cosine similarity loss (or L1 on the direction vector):

```
  L_dir = (1/N_fg) * SUM_{i in foreground}
          (1 - cos_similarity(d_i_pred, d_i_gt))

  where cos_similarity(a, b) = (a . b) / (||a|| * ||b||)
```

**Visualization:**

```
  Direction field on a curved lane divider:

  . . . . . . . . .
  . . . / / / . . .      Arrows show predicted direction
  . . / / . . . . .      at each foreground pixel
  . / / . . . . . .
  / / . . . . . . .      theta ~ 45 degrees here
  . . . . . . . . .
```

### 3.4 Total Training Loss

```
  L_total = lambda_seg * L_seg + lambda_inst * L_instance + lambda_dir * L_dir

  Typical values: lambda_seg = 1.0, lambda_inst = 1.0, lambda_dir = 0.2
```

---

## 4. Post-Processing Pipeline

The neural network outputs dense BEV maps. To produce vectorized polylines
(the format planners actually consume), HDMapNet applies a multi-step
post-processing pipeline.

### Pipeline Overview

```
  +------------+     +-----------+     +-------------+     +------------+
  | Semantic   | --> | Threshold | --> | Skeletonize | --> | Trace &    |
  | + Instance |     | & Cluster |     | (thin to    |     | Vectorize  |
  | + Direction|     |           |     |  1px lines) |     | (polylines)|
  +------------+     +-----------+     +-------------+     +------------+
```

### Step 1: Threshold Semantic Map

```python
# Pseudo-code
binary_mask = (semantic_prob[:, class_c] > threshold)  # e.g., threshold=0.5
```

This produces a binary mask for each map element class.

### Step 2: Instance Clustering

Group foreground pixels into instances using the embedding vectors:

```
Algorithm:
  1. Take all foreground pixels (from step 1)
  2. Run mean-shift clustering (or DBSCAN) in embedding space
  3. Each cluster = one map element instance
```

```
  Before clustering:          After clustering:
  . . X X . X X . .          . . A A . B B . .
  . . X X . X X . .    -->   . . A A . B B . .
  . . X X . X X . .          . . A A . B B . .
  (all same class)            (two distinct instances)
```

### Step 3: Skeletonize

For thin structures (lane dividers), reduce the binary mask to a 1-pixel-wide
skeleton using morphological thinning:

```
  Before:              After:
  . X X X .            . . X . .
  . X X X .    -->     . . X . .
  . X X X .            . . X . .
  . X X X .            . . X . .
```

For area-like structures (pedestrian crossings), we extract the contour
instead.

### Step 4: Trace Connected Components and Vectorize

```
Algorithm:
  1. Find connected components in the skeleton
  2. For each component, find endpoints (pixels with 1 neighbor)
  3. Trace from one endpoint to the other, collecting ordered points
  4. Sub-sample points to create a polyline with N vertices
  5. Use direction predictions to orient the polyline consistently
```

**Output format:** Each map element is a polyline [(x1,y1), (x2,y2), ..., (xN,yN)]

```
  Skeleton pixels:        Vectorized polyline:

  . . * . . . .           . . *---------* . .
  . * . . . . .               |
  * . . . . . .           *---+
  . . . . . . .           (3 vertices)
```

### Direction-Guided Connection

When skeletonization produces gaps, the direction field helps bridge them:

```
  Fragment A:  ---->  (direction points right)
                          gap
  Fragment B:         ---->  (direction also points right, aligned)

  Decision: Connect A to B (directions are compatible)
```

---

## 5. Comparison to Later Methods

### 5.1 MapTR (ICLR 2023)

**Key insight:** Why post-process at all? Directly predict vectorized map
elements end-to-end using a Transformer decoder.

```
  HDMapNet Pipeline:
  Images -> BEV -> Dense Maps -> Post-process -> Vectors  (fragile!)

  MapTR Pipeline:
  Images -> BEV -> Transformer Decoder -> Vectors directly  (end-to-end!)
```

**Architecture:**

```
  +--------+    +---------+    +-------------+    +-----------+
  | Camera | -> | BEV     | -> | Transformer | -> | Polyline  |
  | Images |    | Encoder |    | Decoder     |    | Vertices  |
  +--------+    +---------+    +-------------+    +-----------+
                                     ^
                                     |
                               Point Queries
                               (learnable embeddings,
                                one set per map element)
```

**How MapTR works:**
1. A set of learnable "instance queries" (like DETR) each represent one
   potential map element
2. Each instance query spawns N "point queries" -- one per vertex of the
   predicted polyline
3. The Transformer decoder attends to BEV features and outputs (x, y)
   coordinates for each point
4. Hungarian matching assigns predictions to ground-truth polylines
5. Loss: Chamfer distance between predicted and GT point sets (permutation-
   invariant within each polyline)

**Advantages over HDMapNet:**
- No post-processing (no thresholding, skeletonization, or tracing)
- End-to-end trainable (gradients flow from vector loss to backbone)
- Handles topology naturally (T-intersections, merges)
- Faster inference (no clustering step)

**Point query mechanism:**

```
  Instance Query Q_k (represents "lane divider #k"):
    |
    +-- Point Query Q_k,1 --> predicts vertex (x1, y1)
    +-- Point Query Q_k,2 --> predicts vertex (x2, y2)
    +-- Point Query Q_k,3 --> predicts vertex (x3, y3)
    ...
    +-- Point Query Q_k,N --> predicts vertex (xN, yN)
```

### 5.2 StreamMapNet (CVPR 2024)

**Key insight:** Single-frame map predictions are temporally inconsistent --
the same lane divider jitters between frames. StreamMapNet adds temporal
fusion.

```
  Frame t-2    Frame t-1    Frame t (current)
  +------+     +------+     +------+
  | BEV  |     | BEV  |     | BEV  |
  | feat |     | feat |     | feat |
  +--+---+     +--+---+     +--+---+
     |            |            |
     +-----+------+-----+-----+
           |             |
     +-----v-------------v-----+
     |  Temporal Fusion Module  |
     |  (propagate + fuse       |
     |   past BEV features      |
     |   with ego-motion comp.) |
     +------------+-------------+
                  |
                  v
          +-------+-------+
          | Map Prediction |
          | (current frame)|
          +---------------+
```

**How temporal fusion works:**
1. Cache BEV features from previous frames
2. Warp cached features to current ego-vehicle coordinate frame using
   odometry (ego-motion compensation)
3. Fuse warped historical features with current features via attention
   or concatenation + conv
4. Predict map elements from the fused representation

**Benefits:**
- Temporal consistency (same element predicted in same location across frames)
- Handles occlusions (element hidden in frame t may be visible in t-1)
- Reduces false positives (require multi-frame agreement)
- Longer effective range (accumulate evidence over time)

### Comparison Table

```
  +------------------+----------+----------+--------------+
  | Method           | HDMapNet | MapTR    | StreamMapNet |
  +------------------+----------+----------+--------------+
  | Year             | 2022     | 2023     | 2024         |
  | Output format    | Raster   | Vector   | Vector       |
  | Post-processing  | Yes      | No       | No           |
  | End-to-end       | No       | Yes      | Yes          |
  | Temporal fusion  | No       | No       | Yes          |
  | Backbone         | ResNet   | ResNet   | ResNet       |
  | Decoder          | CNN head | DETR-like| DETR + temp. |
  | nuScenes mAP     | ~30      | ~50      | ~55+         |
  +------------------+----------+----------+--------------+
```

---

## 6. Results, Evaluation, and Limitations

### 6.1 Evaluation on nuScenes

The nuScenes dataset provides:
- 6 surround-view cameras (360-degree coverage)
- 1000 scenes, 20s each, 2Hz annotation
- HD map annotations for lane dividers, road boundaries, pedestrian crossings

**HDMapNet evaluation protocol (introduced by the paper itself):**

| Metric             | Description                                    |
|--------------------|------------------------------------------------|
| IoU (semantic)     | Intersection-over-Union of rasterized maps     |
| AP (chamfer)       | Average Precision using Chamfer distance       |
|                    | between predicted and GT polylines             |
| Vectorized mAP     | Mean AP across all map element categories      |

**Reported results (HDMapNet, camera-only, nuScenes val):**

```
  Map Element          | Semantic IoU | Vectorized AP
  ---------------------|-------------|---------------
  Lane Divider         |    38.4     |    21.7
  Pedestrian Crossing  |    39.1     |    18.4
  Road Boundary        |    44.3     |    39.3
  ---------------------|-------------|---------------
  Mean                 |    40.6     |    26.5
```

**With LiDAR fusion (for reference):**

```
  Mean Semantic IoU: 53.2  (+12.6 over camera-only)
```

### 6.2 Ablation Studies (Key Findings)

1. **View transform matters:** LSS significantly outperforms IPM
   (IoU 40.6 vs 32.1) because roads are not perfectly flat.

2. **Instance embedding helps vectorization:** Without it, connected
   parallel lines merge into one element.

3. **Direction prediction helps:** Reduces broken polylines by ~15%
   in the vectorization step.

4. **Multi-camera fusion:** Using all 6 cameras >> using front camera
   only (360-degree coverage critical for intersections).

### 6.3 Limitations

#### Projection Errors
```
  Even LSS has depth prediction errors. At range > 50m, depth
  uncertainty grows quadratically, causing BEV placement errors:

  Depth error at 50m: ~2m  --> BEV error: ~0.5m
  Depth error at 80m: ~5m  --> BEV error: ~2.0m  (unacceptable)
```

#### Post-Processing Fragility

The skeletonization + tracing pipeline is brittle:

```
  Problem 1: Noise creates spurious branches
  
  Ground truth:    HDMapNet output:      After skeletonize:
  . . | . .        . . X . .             . . | . .
  . . | . .        . X X . .             . ./| . .    <-- spurious branch!
  . . | . .        . . X . .             . . | . .
  
  Problem 2: Gaps break connectivity
  
  Ground truth:    HDMapNet output:      After skeletonize:
  . . | . .        . . X . .             . . | . .
  . . | . .        . . . . .   (gap!)    . . . . .    <-- broken into 2!
  . . | . .        . . X . .             . . | . .
```

#### No Temporal Consistency

```
  Frame t:    predicts lane at x=5.0m
  Frame t+1:  predicts lane at x=5.3m   (jitter!)
  Frame t+2:  predicts lane at x=4.8m   (jitter!)

  The planner sees a "dancing" lane -- unusable without smoothing.
```

#### Limited Range

Camera-based depth prediction degrades with distance. HDMapNet typically
operates within 30m x 60m BEV range around the ego vehicle (vs. 100m+
for LiDAR-based or offline maps).

#### No Elevation Modeling

The BEV representation is inherently 2D (flat). It cannot represent:
- Overpasses (two roads stacked vertically)
- Multi-level parking structures
- Road grade information needed for speed planning

---

## 7. Architecture Summary (End-to-End Data Flow)

```
  +-----------------------------------------------------------------+
  |                         HDMapNet                                  |
  +-----------------------------------------------------------------+
  |                                                                   |
  |  INPUT: N camera images (e.g., 6 surround-view, 1600x900)       |
  |                                                                   |
  |  +-------------------+                                           |
  |  | Image Backbone    |  ResNet-50 or EfficientNet                |
  |  | (shared weights)  |  Output: N feature maps, 1/8 resolution  |
  |  +--------+----------+                                           |
  |           |                                                       |
  |  +--------v----------+                                           |
  |  | View Transform    |  IPM (homography warp)                    |
  |  | (one of two)      |  OR LSS (predict depth, lift, splat)      |
  |  +--------+----------+                                           |
  |           |                                                       |
  |           v                                                       |
  |  +-------------------+                                           |
  |  | BEV Feature Map   |  Shape: (H_bev x W_bev x C)              |
  |  | (fused from all   |  e.g., (200 x 400 x 256) at 0.3m/pixel  |
  |  |  cameras)         |                                           |
  |  +--------+----------+                                           |
  |           |                                                       |
  |     +-----+-----+-----+                                         |
  |     |           |       |                                        |
  |     v           v       v                                        |
  |  +------+  +-------+  +------+                                  |
  |  | Sem. |  | Inst. |  | Dir. |                                  |
  |  | Head |  | Head  |  | Head |                                  |
  |  +--+---+  +---+---+  +--+---+                                  |
  |     |          |          |                                       |
  |     v          v          v                                       |
  |  class map  embeddings  directions                               |
  |  (H x W x K) (H x W x D) (H x W x 2)                           |
  |                                                                   |
  +-----------------------------------------------------------------+
  |                                                                   |
  |  POST-PROCESSING:                                                |
  |  1. Threshold semantic map -> binary masks per class             |
  |  2. Cluster embeddings -> instance masks                         |
  |  3. Skeletonize each instance mask                               |
  |  4. Trace skeleton -> ordered point sequence                     |
  |  5. Use direction to orient + connect fragments                  |
  |  6. Output: set of polylines per class                           |
  |                                                                   |
  +-----------------------------------------------------------------+
  |                                                                   |
  |  OUTPUT: Vectorized local HD map                                 |
  |  {(class_id, [(x1,y1), (x2,y2), ..., (xN,yN)]), ...}           |
  |                                                                   |
  +-----------------------------------------------------------------+
```

---

## 8. Key Takeaways for a PyTorch Engineer

### What You Need to Implement This

1. **Backbone:** Standard torchvision ResNet or timm EfficientNet.
   Nothing special.

2. **View Transform (LSS):**
   - A small MLP/conv that takes (C,) image features and outputs (D,)
     depth logits per pixel
   - An outer product: `lifted = depth_probs.unsqueeze(-1) * features.unsqueeze(-2)`
   - Efficient BEV pooling (use `torch_scatter` or custom CUDA kernels
     for voxel pooling)

3. **Heads:** Three parallel `nn.Sequential` stacks of Conv2d + BN + ReLU.
   Minimal -- the heavy lifting is in the backbone and view transform.

4. **Loss:** Multi-task loss is just a weighted sum. Use
   `torch.nn.CrossEntropyLoss` for semantic, custom discriminative loss
   for instance, and cosine loss for direction.

5. **Post-processing:** Happens in NumPy/OpenCV (not differentiable).
   Use `skimage.morphology.skeletonize`, `scipy.ndimage.label`, and
   `sklearn.cluster.MeanShift`.

### Common Implementation Pitfalls

```
  PITFALL                              FIX
  -------                              ---
  BEV pooling is slow on CPU           Use CUDA scatter/gather ops
  Depth bins too few (< 40)            Use 64-128 bins for good resolution
  Class imbalance kills rare classes   Use focal loss or heavy class weights
  Instance loss explodes early         Clip gradients, warmup lambda_inst
  Homography assumes pinhole           Apply distortion correction first
  Memory OOM with 6 cameras            Process cameras sequentially, pool
```

### The Big Picture: Where HDMapNet Sits in the Field

```
  2020: LSS (view transform)
    |
  2022: HDMapNet (first full online map pipeline)     <-- THIS PAPER
    |
  2023: MapTR (end-to-end vectorized, no post-proc)
    |
  2024: StreamMapNet (temporal consistency)
    |
  2024+: MapTracker, Neural Map Prior, ...
         (unified with planning, world models)
```

HDMapNet is historically important as the **first comprehensive framework**
for camera-based online HD map construction. While later methods (MapTR,
StreamMapNet) surpass it in accuracy and elegance, understanding HDMapNet's
design choices -- the three-head architecture, the post-processing pipeline,
and its limitations -- provides essential context for understanding why the
field evolved toward end-to-end Transformer-based approaches.

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| BEV | Bird's Eye View -- top-down 2D representation |
| IPM | Inverse Perspective Mapping -- homography-based view transform |
| LSS | Lift-Splat-Shoot -- learned depth-based view transform |
| Frustum | 3D volume formed by a camera's field of view |
| Pillar pooling | Efficient operation to collapse 3D points onto 2D BEV grid |
| Polyline | Ordered sequence of (x,y) points representing a map element |
| Chamfer distance | Metric measuring dissimilarity between two point sets |
| Instance embedding | Per-pixel vector learned to cluster same-instance pixels |
| Skeletonization | Morphological operation reducing shapes to 1-pixel-wide lines |
| nuScenes | Large-scale autonomous driving dataset (1000 scenes, Boston + Singapore) |
| mAP | Mean Average Precision (detection/vectorization quality metric) |

---

## 10. Suggested Reading Order

1. **LSS** (Philion & Fidler, ECCV 2020) -- understand the view transform
2. **HDMapNet** (this paper, ICRA 2022) -- the full map construction pipeline
3. **MapTR** (Liao et al., ICLR 2023) -- end-to-end vectorization
4. **StreamMapNet** (Yuan et al., CVPR 2024) -- temporal fusion
5. **VectorMapNet** (Liu et al., ICML 2023) -- autoregressive vectorization

Each builds on the limitations of its predecessor, and together they tell
the story of how the field moved from "detect dense pixels then post-process"
to "predict vectors directly with Transformers."
