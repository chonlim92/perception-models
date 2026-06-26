# Deep Dive: Core Perception Concepts

This document provides in-depth explanations of the fundamental concepts used across all models in this repository. Read this if you want to understand the "why" behind each technique.

---

## 1. Deformable Attention (Used by: BEVFormer, MapTR)

### Standard Multi-Head Attention (Refresher)

```
Q (queries): what we're asking about
K (keys): what we're searching through
V (values): what we actually read

Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

Problem: Q @ K^T is [N_q × N_k] — if N_k is large (e.g., all image pixels),
this is O(N_q × N_k) = extremely expensive!
```

### Deformable Attention (The Solution)

Instead of attending to ALL positions, attend to only K learned positions:

```
Standard Attention:                     Deformable Attention:
Query attends to ALL keys               Query attends to K=4 learned positions
                                        (position = reference + learned offset)

┌────────────────────────┐              ┌────────────────────────┐
│ · · · · · · · · · · · │              │ · · · · · · · · · · · │
│ · · · · · · · · · · · │              │ · · · × · · · · · · · │
│ · · · · · · · · · · · │              │ · · · · · · · × · · · │
│ · · · · · · · · · · · │              │ · · · · · · · · · · · │
│ · · · · · · · · · · · │              │ · · · · · × · · · · · │
│ · · · · · · · · · · · │              │ · · · · · · · · × · · │
└────────────────────────┘              └────────────────────────┘
O(N_q × H × W) = expensive!            O(N_q × K) = cheap! (K=4 typically)

The K positions are: reference_point + learned_offset
  - reference_point: a fixed position (e.g., center of the BEV cell)
  - learned_offset: predicted by the network (depends on the query)
  - attention_weight: also predicted (how much to weight each sample)
```

### Why Deformable Attention is Perfect for BEV

In BEVFormer's spatial cross-attention:
- Each BEV query has a reference point in 3D (at position x, y, at multiple heights z)
- These 3D points project to specific 2D locations in each camera
- The network learns small offsets around these projections
- Only a few samples per query (K=4 per level × 4 levels = 16 total samples)

This is ~100× more efficient than attending to all pixels!

---

## 2. Lift-Splat-Shoot (LSS) View Transform

### The Problem

We have 2D image features and need 3D/BEV features. How?

### The Solution: Predict Depth

```
Step 1: LIFT — Create a depth distribution per pixel

For each pixel (u, v) in the image:
  Network predicts: depth_distribution [D] (e.g., D=60 bins from 1m to 60m)
  Softmax over D bins: P(depth = d_i | pixel)
  
  Image feature at (u, v): f ∈ R^C
  
  "Lifted" feature at depth d_i: f_3d = P(depth=d_i) × f
  
  This creates a frustum of features:
  For each pixel, we have D 3D points along its camera ray

                    · depth bin 60m
                   /
                  · depth bin 30m
                 /
                · depth bin 15m
               /
              · depth bin 5m
             /
   [Camera]──· depth bin 1m
             
  Each · has: feature × probability

Step 2: SPLAT — Pool frustum points into BEV grid

  All frustum points from ALL cameras fall into the BEV grid.
  For each BEV cell, SUM the features of all points that fall in it.
  
  (This is essentially: point cloud → voxelization → height sum = BEV)

Step 3: SHOOT — Process BEV features with a 2D CNN

  Apply a U-Net or ResNet on the BEV feature map for further processing.
```

### Tensor Shapes (Concrete Example)

```
Input:  6 images × [3, 256, 704] (after resize)
Backbone: 6 × [C, 16, 44]  (1/16 resolution, C=256)

Depth prediction: 6 × [D, 16, 44]  (D=60 depth bins)
Outer product:    6 × [C, D, 16, 44]  (features at each depth)

Frustum creation: 6 × 16 × 44 × 60 = 253,440 3D points total
Each point has: (x, y, z) position + C-dim feature

Splat to BEV grid: [C, 200, 200]  (200×200 BEV at 0.5m resolution)
```

---

## 3. 3D Sparse Convolutions (Used by: CenterPoint)

### Why Normal 3D Conv Doesn't Work

A LiDAR point cloud occupies ~0.01% of the 3D volume. Using dense 3D convolutions:
- Would require storing the full [D, H, W] volume in memory
- 99.99% of computations would be on empty space
- Completely impractical for real-time

### Sparse 3D Convolution

Only compute convolution at positions where data exists:

```
Dense 3D Conv:                          Sparse 3D Conv:
Process ALL voxels                      Process ONLY occupied voxels

[□ □ □ □ □ □ □ □]                     [□ □ □ □ □ □ □ □]
[□ □ ■ □ □ □ □ □]  ← occupied         [    ■            ]  ← only these
[□ □ □ □ □ ■ □ □]  ← occupied         [          ■      ]  
[□ □ □ □ □ □ □ □]                     [                  ]
[□ □ □ ■ □ □ □ □]  ← occupied         [      ■          ]

Memory: W×H×D×C (enormous)             Memory: N_occupied × C (tiny)
FLOPs: W×H×D×C×K³ (enormous)           FLOPs: N_occupied × C × K³ (small)
```

### Two Types of Sparse Conv

```
1. Submanifold Sparse Conv
   Output is non-zero ONLY where input is non-zero.
   The sparsity pattern is preserved exactly.
   Good for: most layers (preserves sparsity)

2. Regular Sparse Conv (with stride/dilation)
   Output can be non-zero at neighbors of input positions.
   Allows slight growth of the active set.
   Good for: downsampling layers (merge nearby voxels)

In CenterPoint's backbone:
  - 4 stages of submanifold sparse conv + 1 regular sparse conv (downsample) each
  - Input: ~40,000 voxels
  - After stage 4: ~10,000 voxels (2× downsampled in x,y)
  - Height collapse: max-pool along z → 2D BEV (dense, ~80×80 to 200×200)
```

---

## 4. Hungarian Matching (Used by: BEVFormer, MapTR, DETR3D, PETR)

### The Set Prediction Problem

DETR-style models predict a FIXED number of outputs (e.g., 900 queries → 900 predicted boxes).
The ground truth has a VARIABLE number of objects (e.g., 15 cars, 3 pedestrians = 18 total).

**Problem**: Which prediction corresponds to which ground truth?

### Hungarian Algorithm

Finds the OPTIMAL one-to-one assignment that minimizes total cost:

```
                Predicted Boxes (N=900)
                P1    P2    P3    P4    ...   P900
GT Box G1    [ 0.2  5.1  0.8  12.3  ...  7.8  ]
GT Box G2    [ 4.5  0.3  3.2   0.9  ...  6.1  ]
GT Box G3    [ 7.2  2.1  0.1   8.4  ...  0.5  ]

Cost matrix: cost(Gi, Pj) = λ_cls × class_cost + λ_box × L1_cost + λ_iou × IoU_cost

Hungarian algorithm finds minimum-cost 1:1 matching:
  G1 → P1 (cost 0.2)
  G2 → P2 (cost 0.3)
  G3 → P3 (cost 0.1)
  All other predictions → "no object" (background)
  
Then compute loss ONLY on matched pairs:
  L = Σ_{matched} (L_cls + L_box + L_iou)
  + Σ_{unmatched_preds} L_cls(pred=background)
```

### Why Not Just Use NMS + Anchors?

| Anchors + NMS | Hungarian Matching |
|--------------|-------------------|
| Must design anchor sizes/ratios | No anchor design needed |
| Duplicate predictions → NMS post-processing | No duplicates by design |
| NMS threshold is a fragile hyperparameter | No NMS needed |
| Difficult with polylines/maps | Works perfectly for any output type |

---

## 5. Center-Based Detection (Used by: CenterPoint)

### Anchor-Based vs Center-Based

```
Anchor-Based (PointPillars):                 Center-Based (CenterPoint):

Pre-define anchors at each grid cell:        Predict heatmap of object centers:
┌──┬──┬──┬──┬──┬──┐                        ┌──────────────────────┐
│⊠⊠│⊠⊠│⊠⊠│⊠⊠│⊠⊠│⊠⊠│  × num_classes       │                      │
│⊠⊠│⊠⊠│⊠⊠│⊠⊠│⊠⊠│⊠⊠│  × num_sizes         │      ·  ●            │  Gaussian peaks
│⊠⊠│⊠⊠│⊠⊠│⊠⊠│⊠⊠│⊠⊠│                      │  ●         ·         │  at centers
└──┴──┴──┴──┴──┴──┘                        │         ●            │
                                            └──────────────────────┘
Each anchor: classify + regress offset
Many anchors → many parameters → NMS needed  Find peaks → regress attributes
                                             One peak = one object → no NMS!

Hyperparameters needed:                      Hyperparameters needed:
- Anchor sizes per class                     - Gaussian sigma (from GT box size)
- Anchor rotations (0°, 90°)                 - That's it
- IoU thresholds for pos/neg
- NMS thresholds per class
```

### Gaussian Focal Loss

The heatmap GT has Gaussian blobs (not hard 0/1):

```
GT Heatmap for one class:

  0  0  0  0  0  0  0  0
  0  0  .1 .2 .1 0  0  0     Gaussian blob centered at object
  0  .1 .5 .9 .5 .1 0  0     Peak = 1.0 at center
  0  .2 .9  1 .9 .2 0  0     Falls off with σ proportional to box size
  0  .1 .5 .9 .5 .1 0  0
  0  0  .1 .2 .1 0  0  0
  0  0  0  0  0  0  0  0

Loss: Modified focal loss
  For positive (GT > 0): -(1-p)^α × log(p) × weight
  For negative (GT = 0): -(p)^β × (1-GT)^γ × log(1-p)
  
  β=4 means: near-center cells (GT=0.9) get MUCH less negative penalty
  This prevents penalizing predictions that are "almost right" (1 pixel off)
```

---

## 6. Ego-Motion Warping for Temporal Fusion

### The Problem

Between frames, the car moves. If we simply concatenate BEV features from different times,
features won't be aligned — the same road surface appears at different grid positions!

### The Solution: Warp Past BEV to Current Frame

```
At time t-1:                     At time t:
Car was here →  [×]              Car is now here →  [×]
                                 (moved forward and turned slightly)

BEV at t-1:                      BEV at t (same physical world):
┌──────────────┐                 ┌──────────────┐
│   building   │                 │ building     │   ← same building
│     [car]    │                 │  [car]       │      but shifted in grid!
│  ════════    │                 │    ════════  │   ← same road
│   lane       │                 │  lane        │      but shifted!
└──────────────┘                 └──────────────┘

WITHOUT warping: features misaligned, temporal fusion fails
WITH warping: transform t-1 BEV so physical locations match t BEV grid
```

### How Warping Works (Math)

```python
# ego_pose_t: [4,4] matrix transforming ego frame at t → world
# ego_pose_t1: [4,4] matrix transforming ego frame at t-1 → world

# Transform from t-1 ego frame to t ego frame:
T_t1_to_t = inv(ego_pose_t) @ ego_pose_t1  # [4, 4]

# For 2D BEV warping, we only need the 2D part:
R = T_t1_to_t[:2, :2]  # [2, 2] rotation in xy-plane
t = T_t1_to_t[:2, 3]   # [2] translation in xy

# Create grid of current BEV coordinates
grid_x = linspace(x_min, x_max, W)  # physical x coordinates
grid_y = linspace(y_min, y_max, H)  # physical y coordinates
grid = meshgrid(grid_x, grid_y)     # [H, W, 2]

# Transform to find where each current cell was in past BEV
past_coords = (R.T @ (grid - t).T).T  # [H, W, 2]

# Normalize to [-1, 1] for grid_sample
past_coords_normalized = 2 * (past_coords - [x_min, y_min]) / [x_range, y_range] - 1

# Sample past BEV features at transformed coordinates
warped_bev = F.grid_sample(past_bev_features, past_coords_normalized)
```

---

## 7. Pillar Encoding (Used by: PointPillars, RadarPillarNet)

### Core Idea: Turn 3D Points into a 2D "Pseudo-Image"

```
Step 1: Divide ground plane into grid cells (pillars = vertical columns)

Top-down view of grid:
┌──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  Each cell = 0.25m × 0.25m
├──┼──┼──┼──┼──┼──┼──┼──┤  (or larger for radar: 0.5m × 0.5m)
│  │··│  │  │  │  │  │  │  Points in this cell form one "pillar"
├──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │··│  │  │  │  │  · = points falling in this x,y cell
├──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │··│  │  │
└──┴──┴──┴──┴──┴──┴──┴──┘

Step 2: Augment each point with additional features

Original point: (x, y, z, intensity)
Augmented:     (x, y, z, intensity, x-xc, y-yc, z-zc, x-xp, y-yp)
  where xc, yc, zc = centroid of pillar
        xp, yp = x,y center of the pillar cell

Step 3: PointNet per pillar

For each non-empty pillar (e.g., containing M points, padded to max P=20):
  Input: [P, 9] (P points × 9 features)
  Shared MLP: Linear(9, 64) → BN → ReLU
  Max-pool over P points: [64]  (one feature vector per pillar)

Step 4: Scatter to pseudo-image

Non-empty pillars have features. Place them at their (i, j) grid position:
  pseudo_image[i, j, :] = pillar_feature  (for non-empty pillars)
  pseudo_image[i, j, :] = 0               (for empty pillars)
  
Result: [C, H, W] — a normal 2D "image" that we can process with 2D CNNs!

Step 5: Standard 2D backbone (ResNet + FPN)

  Apply regular 2D convolutions → multi-scale features → detection head
```

### Why Pillars Are Faster Than Voxels

- Pillars: 2D grid → 2D convolutions (fast, well-optimized on GPUs)
- Voxels: 3D grid → 3D convolutions (expensive, even sparse ones)
- PointPillars achieves 62 Hz vs CenterPoint's 11 Hz

---

## 8. Range Image Projection (Used by: RangeNet++)

### The Idea: Make LiDAR Look Like a Camera Image

A spinning LiDAR scans the world in spherical coordinates. If we "unroll" this:

```
LiDAR spins 360° horizontally, with N vertical beams (e.g., 64)

Spherical coordinates:
  azimuth φ: horizontal angle (0° to 360°)
  elevation θ: vertical angle (depends on beam, e.g., -25° to +3°)
  range r: distance to the point

Projection:
  (x, y, z) → (φ, θ, r)
  φ = atan2(y, x)
  θ = asin(z / sqrt(x² + y² + z²))
  r = sqrt(x² + y² + z²)

Image coordinates:
  u = (1 - φ/π) × W/2        (column, W=2048 typically)
  v = (1 - (θ-θ_min)/(θ_max-θ_min)) × H  (row, H=64 for 64-beam)

Result: a 2D "range image" of shape [H, W] = [64, 2048]
Each pixel stores: (range, x, y, z, intensity) = 5 channels
```

```
Range Image Example (top = up, left/right = around car):

    0°                     180°                    360°
  ┌─────────────────────────────────────────────────────┐
  │▓▓▓▓▓▓▓▓   ▓▓▓▓▓▓   ▓▓▓▓▓▓▓▓▓▓▓▓   ▓▓▓▓▓▓▓▓▓▓│ ← buildings
  │░░   ░░░░   ░░░░░░░   ░░░░░░░   ░░░░░░   ░░░░░░░│ ← cars/objects
  │████████████████████████████████████████████████████│ ← ground
  │████████████████████████████████████████████████████│ ← ground
  └─────────────────────────────────────────────────────┘
  
  Now apply any 2D CNN (DarkNet-53 in RangeNet++)!
  The CNN doesn't know this came from LiDAR — it's just an "image."
```

### KNN Post-Processing

Range images have a problem: boundaries between objects get blurred by the CNN.
Fix: for each point, check its K nearest neighbors in 3D space and vote on the label.

---

## 9. Cylindrical Voxelization (Used by: Cylinder3D)

### The Problem with Cartesian Voxels

LiDAR point density is NOT uniform in Cartesian space:
- Very dense near the sensor (many points per voxel)
- Very sparse far away (empty voxels waste memory)

```
Cartesian grid (wasteful):              Cylindrical grid (efficient):

  ┌──┬──┬──┬──┬──┬──┬──┬──┐            ┌──┬──┬──┬──┬──┬──┬──┬──┐
  │  │  │  │  │  │  │  │  │ ← far      │  │  │  │  │  │  │  │  │ ← far
  │  │  │  │  │  │  │  │  │ (sparse)    │  │  │  │  │  │  │  │  │ (uniform!)
  │  │  │  │  │  │  │  │  │            │  │  │  │  │  │  │  │  │
  │  │  │██│██│██│██│  │  │ ← mid      │  │  │ · │ · │ · │  │  │ ← mid
  │  │██│██│██│██│██│██│  │            │  │ · │ · │ · │ · │ · │  │
  │  │██│██│██│██│██│██│  │ ← near     │ · │ · │ · │ · │ · │ · │ · │ ← near
  │  │██│██│██│██│██│██│  │ (dense!)    │ · │ · │ · │ · │ · │ · │ · │ (uniform!)
  └──┴──┴──┴──┴──┴──┴──┴──┘            └──┴──┴──┴──┴──┴──┴──┴──┘
                                        
  Cartesian: near cells have 100+ pts,   Cylindrical: each cell has ~same
  far cells have 0-1 pts                  number of points (balanced)
```

### Cylindrical Coordinates

```
(x, y, z) → (r, θ, z)
  r = sqrt(x² + y²)     (radial distance from sensor)
  θ = atan2(y, x)       (azimuth angle)
  z = z                  (height, unchanged)

Grid:
  r: [0, 50m] divided into 480 bins (finer near, coarser far)
  θ: [0, 2π] divided into 360 bins (1° each)
  z: [-3, 1m] divided into 32 bins
```

---

## 10. The Transformer Decoder for Detection

### How Object Queries Work

All DETR-style models (BEVFormer, DETR3D, PETR, MapTR) use this pattern:

```
Object Queries: N learned embeddings (e.g., N=900)
  Each query is responsible for detecting AT MOST one object.
  Queries are initialized as learned parameters (or from heuristics).

Decoder (typically 6 layers):
  Each layer does:
  
  1. Self-Attention among queries
     Purpose: queries communicate — "I'm detecting a car here,
              you don't need to detect the same car"
     
  2. Cross-Attention to features (BEV or image)
     Purpose: queries gather information from the feature map
              to understand what object (if any) they should detect
     
  3. FFN (Feed-Forward Network)
     Purpose: process gathered information
     
  4. Prediction Head
     Each query predicts: class_logits + box_parameters
     If nothing is there: query predicts "no object" class

After decoder:
  N queries → N predictions (most will be "no object")
  Hungarian matching assigns GT objects to predictions
  Loss computed only on matched pairs
```

---

## Summary: How Everything Connects

```
                    ┌──────────────────────────────────────────────┐
                    │            COMMON BACKBONE IDEAS              │
                    │  ResNet/EfficientNet (images)                 │
                    │  PointNet (points)                            │
                    │  Sparse 3D Conv (voxels)                      │
                    └──────────┬──────────────────────┬────────────┘
                               │                      │
              ┌────────────────┴────┐    ┌────────────┴─────────────┐
              │  CAMERA-TO-BEV       │    │  POINT-CLOUD-TO-BEV      │
              │  • LSS (depth pred)  │    │  • Voxelize + collapse   │
              │  • Deformable attn   │    │  • Pillar + scatter      │
              │  • 3D PE             │    │  • Cylindrical           │
              │  • GKT               │    │  • Range image           │
              └─────────┬────────────┘    └───────────┬─────────────┘
                        │                              │
                        └──────────┬───────────────────┘
                                   │
                    ┌──────────────┴──────────────────┐
                    │        BEV FEATURES              │
                    │  Unified 2D representation       │
                    │  [C, H, W] feature map           │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼───────────────────────┐
              │                    │                        │
    ┌─────────┴──────┐  ┌────────┴─────────┐  ┌──────────┴───────┐
    │  TEMPORAL       │  │  DETECTION HEAD   │  │  MAP HEAD         │
    │  • BEV warp    │  │  • Center heatmap │  │  • Query decoder  │
    │  • Temporal attn│  │  • DETR decoder   │  │  • Set prediction │
    │  • Query prop  │  │  • Anchor-based   │  │  • Polyline pred  │
    └────────────────┘  └──────────────────┘  └──────────────────┘
                                   │                        │
                        ┌──────────┴────┐        ┌─────────┴──────┐
                        │   3D BOXES     │        │  VECTORIZED    │
                        │  + TRACKING    │        │  HD MAP        │
                        └───────────────┘        └────────────────┘
```

Every model in this repository is a specific combination of choices from these building blocks.
Understanding these core concepts lets you understand any new perception paper that comes out!
