# MapTR: Research Summary

## Paper Overview

**MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction**
- Authors: Bencheng Liao, Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Qian Zhang, Wenyu Liu, Chang Huang
- Venue: ICLR 2023
- Follow-up: MapTRv2 (extended version with improved performance)

MapTR introduces a structured end-to-end framework for online vectorized HD map construction from multi-camera sensor inputs. Unlike prior methods that produce rasterized map representations or use autoregressive point generation, MapTR directly outputs vectorized map elements as sets of ordered points in a single forward pass.

---

## What is Vectorized Map Construction?

### HD Maps in Autonomous Driving

High-Definition (HD) maps are a critical component of the autonomous driving stack. Unlike consumer navigation maps (Google Maps, Waze) that store roads as coarse polylines with lane counts, HD maps encode precise geometric and semantic information about the road environment at centimeter-level accuracy:

- **Lane dividers**: The painted lines separating lanes (solid, dashed, double-yellow)
- **Road boundaries**: Curbs, guardrails, and road edges that define drivable area
- **Pedestrian crossings**: Crosswalk polygons where pedestrians have right-of-way
- **Stop lines**: Where vehicles must stop at intersections
- **Lane centerlines**: The ideal driving path within each lane

These elements matter to downstream planning because a self-driving car needs to know exactly where it can drive, where lane changes are legal, where to yield to pedestrians, and where to stop.

### The Traditional Approach: Offline HD Maps

Historically, HD maps were built offline by specialized survey vehicles equipped with high-precision LiDAR and GPS. Human annotators then labeled these maps by hand. The problems:

1. **Cost**: Survey vehicles and human annotation are extremely expensive
2. **Freshness**: Road construction, new lane markings, and changed boundaries make maps stale
3. **Coverage**: Only major roads in select cities get mapped
4. **Storage**: Centimeter-resolution maps for an entire country require enormous storage

This motivates **online HD map construction**: building the map in real-time from the vehicle's own sensors (cameras, LiDAR) as it drives.

### Two Paradigms: Rasterized vs. Vectorized

There are two fundamentally different ways to represent map elements as neural network outputs:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RASTERIZED APPROACH                                │
│                                                                       │
│  Multi-camera    BEV Feature     Semantic         Post-processing    │
│  Images      -->  Extraction  -->  Segmentation -->  Vectorization   │
│                                                                       │
│  Output: Dense pixel grid (e.g., 200x200 BEV map)                   │
│                                                                       │
│  ┌────────────────────┐       ┌────────────────────┐                 │
│  │ . . . . . . . . .  │       │ After post-proc:   │                 │
│  │ . . X X . . . . .  │       │                    │                 │
│  │ . X X X X . . . .  │  -->  │ Polyline: [(2,1),  │                 │
│  │ . . X X X X . . .  │       │  (3,2),(4,3),      │                 │
│  │ . . . X X X . . .  │       │  (5,4),(6,5)]      │                 │
│  │ . . . . X X . . .  │       │                    │                 │
│  └────────────────────┘       └────────────────────┘                 │
│   BEV segmentation mask        Vectorized output                     │
│                                                                       │
│  Problems:                                                            │
│  - Resolution limited by grid size                                   │
│  - Post-processing is complex (skeletonization, grouping)            │
│  - No native instance awareness (which pixels = which lane?)         │
│  - Artifacts at junctions and intersections                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    VECTORIZED APPROACH (MapTR)                        │
│                                                                       │
│  Multi-camera    BEV Feature     Learned Queries    Direct Vector    │
│  Images      -->  Extraction  -->  + Decoder     -->  Output         │
│                                                                       │
│  Output: Set of polylines/polygons as ordered point sequences        │
│                                                                       │
│  ┌────────────────────┐                                              │
│  │                    │                                              │
│  │    *---*---*       │  Lane divider: [(2.3, 1.1), (3.7, 2.4),     │
│  │   /         \      │                 (5.1, 3.8), (6.4, 5.2)]     │
│  │  *           *     │                                              │
│  │   \         /      │  Road boundary: [(1.0, 0.5), (1.2, 2.3),    │
│  │    *---*---*       │                  (1.5, 4.1), (1.8, 6.0)]    │
│  │                    │                                              │
│  └────────────────────┘                                              │
│   Direct polyline/polygon prediction                                 │
│                                                                       │
│  Advantages:                                                          │
│  - Continuous coordinates (sub-pixel precision)                      │
│  - Native instance-level output                                      │
│  - No post-processing needed                                         │
│  - Directly usable by downstream planner                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Vectorized is Better for Downstream Planning

The planning module in an autonomous driving stack needs to reason about:
- "Can I change lanes here?" --> needs to know if the divider is dashed or solid
- "Where exactly is the road edge?" --> needs precise boundary geometry
- "Is there a crosswalk ahead?" --> needs the polygon extent

A rasterized BEV segmentation map gives you pixels. The planner then has to:
1. Group pixels into instances (which blob is which lane line?)
2. Extract centerlines from thick pixel blobs (skeletonization)
3. Order the extracted points into sequences (which direction does the line go?)
4. Smooth and simplify the result

Each step introduces errors and latency. With vectorized output, the planner receives exactly what it needs: a list of polylines and polygons with semantic labels, ready for geometric reasoning. No intermediate conversion needed.

---

## Core Innovation: Permutation-Equivalent Modeling

### The Fundamental Problem

Map elements (lane dividers, road boundaries, pedestrian crossings) are geometric primitives represented as ordered point sequences (polylines or polygons). However, the same geometric shape can be described by multiple equivalent point orderings:

- A **polyline** with N points has **2 equivalent representations** (forward and reverse traversal)
- A **polygon** with N points has **2N equivalent representations** (N starting points x 2 directions)

Traditional approaches that enforce a single canonical ordering create an artificial constraint that makes learning harder and introduces ambiguity in the ground truth.

### Permutation Equivalence: Detailed Explanation with Examples

#### Polyline Equivalence (2 permutations)

A polyline (like a lane divider) connects point A to point B. Whether you describe it as going from left-to-right or right-to-left, it is the same physical line:

```
Forward:   A -----> B -----> C -----> D
           (1,2)   (3,4)   (5,6)   (7,8)

Reverse:   D -----> C -----> B -----> A
           (7,8)   (5,6)   (3,4)   (1,2)

Both describe the SAME lane divider in the real world.
```

So for a polyline with N points, the equivalent set is:
- `{(p_1, p_2, ..., p_N), (p_N, p_{N-1}, ..., p_1)}`

#### Polygon Equivalence (2N permutations)

A polygon (like a crosswalk) is a closed shape. You can start describing it from any vertex, and you can go clockwise or counterclockwise. Consider a rectangle ABCD:

```
            A --------- B
            |           |
            |  CROSSWALK|
            |           |
            D --------- C
```

**Cyclic permutations** (starting from different vertices, same direction):

```
Starting at A, clockwise: [A, B, C, D]
Starting at B, clockwise: [B, C, D, A]
Starting at C, clockwise: [C, D, A, B]
Starting at D, clockwise: [D, A, B, C]
```

All four describe the exact same rectangle. The shape is identical regardless of which corner you call "first."

**Direction equivalence** (counterclockwise traversal):

```
Starting at A, counter-clockwise: [A, D, C, B]
Starting at B, counter-clockwise: [B, A, D, C]
Starting at C, counter-clockwise: [C, B, A, D]
Starting at D, counter-clockwise: [D, C, B, A]
```

These four also describe the same rectangle, just traversed in the opposite direction.

**Total: 4 cyclic rotations x 2 directions = 8 = 2N permutations** (for N=4 vertices).

More generally, for a polygon with N vertices:

```
Number of equivalent permutations = 2N

    = N (cyclic rotations) x 2 (clockwise + counterclockwise)
```

#### Why This Matters: The Training Problem

Imagine you are training a neural network to predict a crosswalk polygon. The ground truth annotation says:

```
GT ordering: [A=(1,1), B=(5,1), C=(5,3), D=(1,3)]
```

Your model predicts:

```
Predicted:   [C=(5,3), D=(1,3), A=(1,1), B=(5,1)]
```

This is a **perfect prediction** -- it describes the exact same rectangle! But if you naively compute L1 loss between corresponding points:

```
Loss = |A - C| + |B - D| + |C - A| + |D - B|
     = |(1,1)-(5,3)| + |(5,1)-(1,3)| + |(5,3)-(1,1)| + |(1,3)-(5,1)|
     = (4+2) + (4+2) + (4+2) + (4+2)
     = 24   <-- LARGE LOSS for a PERFECT prediction!
```

With MapTR's permutation-equivalent matching:

```
Try all 2N=8 permutations, find the one matching the prediction's ordering:
  GT permutation [C, D, A, B] matches prediction [C, D, A, B]

Loss = |C - C| + |D - D| + |A - A| + |B - B|
     = 0 + 0 + 0 + 0
     = 0   <-- Correct! Zero loss for a perfect prediction.
```

The training loss is computed as the **minimum** over all equivalent permutations:

```
L_point = min_{sigma in Sigma} sum_{i=1}^{N} ||p_i^pred - p_{sigma(i)}^gt||
```

where Sigma is the set of all 2N (polygon) or 2 (polyline) equivalent permutations.

#### Visual Summary of Polygon Permutation Equivalence

```
All 8 orderings below represent the SAME square:

   1---2       2---3       3---4       4---1
   |   |       |   |       |   |       |   |
   4---3       1---4       2---1       3---2

   (CW from 1) (CW from 2) (CW from 3) (CW from 4)

   1---4       4---3       3---2       2---1
   |   |       |   |       |   |       |   |
   2---3       1---2       4---1       3---4

   (CCW from 1)(CCW from 4)(CCW from 3)(CCW from 2)
```

---

## Hierarchical Bipartite Matching

MapTR employs a two-level matching strategy inspired by DETR but extended to handle structured point sets.

### Level 1: Instance-Level Matching

- Uses Hungarian algorithm to find optimal one-to-one assignment between predicted map instances and ground truth instances
- Matching cost considers both classification confidence and geometric similarity (point-set distance)
- Each predicted query is matched to at most one ground truth element

### Level 2: Point-Level Matching

- After instance matching, for each matched pair, finds the optimal point-level correspondence
- For polylines: selects the better of forward/reverse ordering (2 candidates)
- For polygons: evaluates all 2N cyclic permutations and selects the minimum-cost one
- Point-level matching uses Chamfer distance or L1 distance between corresponding points

### Concrete Worked Example

Let us walk through the full hierarchical matching process step by step.

**Setup:**
- Ground truth has **3 map elements**: GT_1 (lane divider, polyline), GT_2 (crosswalk, polygon), GT_3 (road boundary, polyline)
- Model predicts **10 queries**: Q_1, Q_2, ..., Q_10 (each with a class prediction and N points)

**Step 1: Compute the Cost Matrix (10 x 3)**

For each (query, GT) pair, compute a matching cost that combines:
- Classification cost: how well the predicted class matches the GT class
- Geometric cost: distance between predicted points and GT points (using best permutation)

```
              GT_1        GT_2        GT_3
         ┌──────────┬──────────┬──────────┐
  Q_1    │   2.1    │   8.4    │   3.7    │
  Q_2    │   7.3    │   1.2    │   6.5    │
  Q_3    │   5.8    │   6.1    │   2.3    │
  Q_4    │   3.4    │   9.2    │   4.8    │
  Q_5    │   6.7    │   4.5    │   7.1    │
  Q_6    │   8.9    │   3.8    │   5.6    │
  Q_7    │   4.2    │   7.7    │   8.3    │
  Q_8    │   9.1    │   5.3    │   3.1    │
  Q_9    │   6.4    │   8.8    │   6.9    │
  Q_10   │   7.8    │   6.2    │   5.4    │
         └──────────┴──────────┴──────────┘
```

**Step 2: Hungarian Algorithm**

The Hungarian algorithm finds the minimum-cost one-to-one assignment:
- Q_1 --> GT_1 (cost 2.1)
- Q_2 --> GT_2 (cost 1.2)
- Q_3 --> GT_3 (cost 2.3)

Unmatched queries (Q_4 through Q_10) are assigned "no object" and supervised with a background/no-class label.

**Step 3: Point-Level Permutation Matching**

For each matched pair, we now find the best point-level correspondence:

*Example: Q_2 matched to GT_2 (crosswalk polygon with 4 vertices)*

```
GT_2 vertices:  A=(2,1), B=(6,1), C=(6,4), D=(2,4)
Q_2 predicted:  (5.9,3.8), (2.1,3.9), (2.0,1.1), (5.8,1.0)
```

The model's prediction looks like it started at vertex C and went counterclockwise. Let us check all 8 permutations:

```
Permutation [A,B,C,D]:  ||(5.9,3.8)-(2,1)|| + ||(2.1,3.9)-(6,1)|| + ... = 18.2
Permutation [B,C,D,A]:  ||(5.9,3.8)-(6,1)|| + ||(2.1,3.9)-(6,4)|| + ... = 14.7
Permutation [C,D,A,B]:  ||(5.9,3.8)-(6,4)|| + ||(2.1,3.9)-(2,4)|| + ... = 0.6  <-- MIN
Permutation [D,A,B,C]:  ||(5.9,3.8)-(2,4)|| + ||(2.1,3.9)-(2,1)|| + ... = 16.3
Permutation [A,D,C,B]:  ||(5.9,3.8)-(2,1)|| + ||(2.1,3.9)-(2,4)|| + ... = 15.1
Permutation [D,C,B,A]:  ||(5.9,3.8)-(2,4)|| + ||(2.1,3.9)-(6,4)|| + ... = 13.8
Permutation [C,B,A,D]:  ||(5.9,3.8)-(6,4)|| + ||(2.1,3.9)-(6,1)|| + ... = 12.4
Permutation [B,A,D,C]:  ||(5.9,3.8)-(6,1)|| + ||(2.1,3.9)-(2,1)|| + ... = 11.9
```

Best permutation: [C, D, A, B] with total L1 distance = 0.6

This means the model predicted [C, D, A, B] ordering nearly perfectly. The loss is computed only against this best permutation, so the model is not penalized for choosing a different (but valid) starting point.

**Step 4: Compute Final Loss**

```
L_total = L_classification + lambda_1 * L_point + lambda_2 * L_direction

where L_point uses the minimum-permutation distance found in Step 3.
```

### Computational Complexity Analysis

This hierarchical approach decouples the combinatorial complexity:
- Instance matching: O(M^3) via Hungarian algorithm (M = number of queries, typically 50-100)
- Point matching: O(N) for polylines, O(N^2) for polygons (N = points per element, typically 20)
- Total: O(M^3 + M * N^2) -- tractable for real-time training

Without hierarchical decomposition, you would need to jointly optimize over all possible (instance, point) assignments simultaneously, which is combinatorially explosive.

---

## GKT: Geometry-guided Kernel Transformer

### The Camera-to-BEV Challenge

MapTR takes multi-camera images as input (typically 6 cameras providing 360-degree coverage) and needs to produce map elements in Bird's Eye View (BEV) space. The fundamental challenge:

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│   Camera images are in PERSPECTIVE view:                     │
│                                                               │
│      Front-left    Front     Front-right                     │
│      ┌───────┐  ┌───────┐  ┌───────┐                        │
│      │  /    │  │   |   │  │    \  │                        │
│      │ / lane│  │   |   │  │lane \ │                        │
│      │/  line│  │   |   │  │line  \│                        │
│      └───────┘  └───────┘  └───────┘                        │
│                                                               │
│      Rear-left    Rear      Rear-right                       │
│      ┌───────┐  ┌───────┐  ┌───────┐                        │
│      │       │  │   |   │  │       │                        │
│      │       │  │   |   │  │       │                        │
│      │       │  │   |   │  │       │                        │
│      └───────┘  └───────┘  └───────┘                        │
│                                                               │
│   Map elements exist in BEV (top-down) space:                │
│                                                               │
│      ┌─────────────────────────┐                             │
│      │         * * *           │                             │
│      │        *     *          │                             │
│      │  ------*--E--*------    │  E = ego vehicle            │
│      │        *     *          │  --- = lane dividers        │
│      │         * * *           │  * = road boundary          │
│      └─────────────────────────┘                             │
│                                                               │
│   Challenge: How to go from 2D perspective -> BEV?           │
└─────────────────────────────────────────────────────────────┘
```

### How GKT Works

GKT (Geometry-guided Kernel Transformer) is MapTR's approach to lifting multi-camera 2D features into a unified BEV representation. The key idea:

1. **Define a BEV grid**: Create a grid of BEV query positions (e.g., 200x100 covering 60m x 30m around the ego vehicle)

2. **Project BEV positions to cameras**: For each BEV position, use known camera intrinsics and extrinsics to determine which camera pixel(s) correspond to that 3D location. Since depth is unknown, project a range of heights (e.g., ground plane to 3m above ground).

3. **Geometry-guided sampling**: For each BEV query, sample features from the relevant camera image locations using deformable attention. The sampling locations are initialized by the geometric projection, then refined by learned offsets.

4. **Aggregate with attention**: Combine sampled features using cross-attention, where the BEV query attends to its geometrically-relevant image features.

```
BEV Query Position (x, y) in world coordinates
         |
         v
    Camera Projection (using known calibration)
         |
         v
    Sample region in image feature map
         |
         v
    Deformable Cross-Attention
         |
         v
    BEV Feature at (x, y)
```

### Comparison with Other View Transformation Methods

| Method | Approach | Depth | Speed | Quality |
|--------|----------|-------|-------|---------|
| **LSS (Lift-Splat-Shoot)** | Predict explicit depth distribution per pixel, scatter to 3D voxels, collapse to BEV | Learned | Medium | Good but noisy depth |
| **BEVDet** | LSS variant with data augmentation and better training | Learned | Medium | Better than LSS |
| **BEVFormer** | Spatial cross-attention: BEV queries attend to multi-camera features using deformable attention at projected 3D reference points | Implicit | Slow (dense attention) | High quality |
| **GKT (MapTR)** | Geometry-guided kernels: pre-compute sampling locations from camera geometry, use lightweight attention | Implicit | Fast | Good balance |

**Why GKT is fast:** Instead of dense attention over all image pixels (BEVFormer) or explicit depth prediction (LSS), GKT uses camera geometry to restrict attention to a small number of relevant image locations per BEV query. This makes the view transformation lightweight enough for real-time online mapping.

### The Full MapTR Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        MapTR Architecture                           │
│                                                                     │
│  6 Camera Images                                                    │
│       │                                                             │
│       v                                                             │
│  ┌──────────────┐                                                   │
│  │  Backbone    │  (ResNet-50 or VoVNet-99)                        │
│  │  + FPN       │  Extracts multi-scale 2D features                │
│  └──────────────┘                                                   │
│       │                                                             │
│       v                                                             │
│  ┌──────────────┐                                                   │
│  │  GKT View   │  Projects 2D features to BEV                     │
│  │  Transform  │  using geometry-guided sampling                   │
│  └──────────────┘                                                   │
│       │                                                             │
│       v                                                             │
│  ┌──────────────┐                                                   │
│  │  BEV        │  H x W feature map in bird's eye view            │
│  │  Features   │  (e.g., 200 x 100 x 256)                        │
│  └──────────────┘                                                   │
│       │                                                             │
│       v                                                             │
│  ┌──────────────┐                                                   │
│  │  Transformer │  M instance queries, each with N point queries   │
│  │  Decoder    │  Self-attention + Cross-attention to BEV          │
│  └──────────────┘                                                   │
│       │                                                             │
│       v                                                             │
│  ┌──────────────┐                                                   │
│  │  Prediction  │  Per-query: class label + N 2D point coords     │
│  │  Heads      │  Output: set of vectorized map elements          │
│  └──────────────┘                                                   │
│       │                                                             │
│       v                                                             │
│  Hierarchical Bipartite Matching (training only)                   │
│  --> Permutation-equivalent point-level loss                       │
└────────────────────────────────────────────────────────────────────┘
```

---

## Comparison with Prior Methods

### HDMapNet (Li et al., 2022)

| Aspect | HDMapNet | MapTR |
|--------|----------|-------|
| Output format | Rasterized segmentation map | Vectorized point sets |
| Post-processing | Complex vectorization pipeline (skeletonization, pixel grouping) | Direct vector output, no post-processing |
| Instance awareness | Requires grouping/clustering | Native instance-level prediction |
| Accuracy | Limited by rasterization resolution | Sub-pixel continuous coordinates |
| Speed | Slow due to post-processing | Real-time capable |

### VectorMapNet (Liu et al., 2023)

| Aspect | VectorMapNet | MapTR |
|--------|-------------|-------|
| Point generation | Autoregressive (sequential) | Parallel (single-shot) |
| Inference speed | Slow (N forward passes per element) | Fast (single forward pass) |
| Error accumulation | Yes (sequential dependency) | No (independent point prediction) |
| Ordering constraint | Fixed canonical ordering | Permutation-equivalent (flexible) |
| Scalability | Poor with many points | Constant inference time |

### Key Advantages of MapTR

1. **End-to-end vectorized output**: No rasterization or post-processing needed
2. **Parallel prediction**: All points predicted simultaneously
3. **Permutation invariance**: No artificial ordering constraints
4. **Real-time performance**: 25+ FPS on standard hardware
5. **Unified framework**: Same architecture handles polylines and polygons

---

## Comparison Table with Contemporary Methods

The following table compares MapTR against the broader landscape of online HD map construction methods. All numbers are on the nuScenes validation set with a ResNet-50 backbone unless otherwise noted.

| Method | Representation | Temporal | mAP | FPS | Key Innovation |
|--------|---------------|----------|-----|-----|----------------|
| HDMapNet | Rasterized | No | 21.7 | 3.2 | BEV segmentation + post-proc vectorization |
| VectorMapNet | Vector (autoregressive) | No | 36.1 | 2.9 | Sequential point generation with coarse-to-fine |
| MapTR | Vector (parallel) | No | 43.2 | 25.1 | Permutation-equivalent modeling |
| MapTRv2 | Vector (parallel) | No | 50.3 | 21.8 | One-to-many matching + decoupled attention |
| StreamMapNet | Vector (parallel) | Yes | 47.8 | 18.3 | Temporal BEV feature fusion across frames |
| BeMapNet | Bezier curves | No | 44.2 | 15.7 | Bezier curve representation (fewer params per element) |
| PivotNet | Vector (pivot-based) | No | 42.8 | 19.4 | Pivot points + dynamic point sampling |
| HIMap | Vector (parallel) | No | 48.1 | 20.3 | Hierarchical instance-point interactions |

### Reading This Table

- **Representation**: How map elements are encoded in the model output
  - *Rasterized*: BEV segmentation masks require post-processing
  - *Vector (autoregressive)*: Points generated one-by-one, slow but captures dependencies
  - *Vector (parallel)*: All points in one forward pass, fast
  - *Bezier curves*: Parametric curves, compact but limited to smooth shapes
- **Temporal**: Whether the method fuses information across multiple time steps
- **mAP**: Mean Average Precision at thresholds 0.5m, 1.0m, 1.5m (Chamfer distance)
- **FPS**: Frames per second on a single NVIDIA 3090 GPU

### Key Insight from the Table

MapTR achieved a massive speed improvement (25.1 FPS vs. 2.9-3.2 FPS) while simultaneously improving accuracy over prior work. This was the first method to make online vectorized mapping viable for real-time deployment. MapTRv2 then pushed accuracy further (50.3 mAP) with only a modest speed reduction.

---

## MapTRv2 Improvements

MapTRv2 extends the original MapTR with several enhancements for faster convergence and higher accuracy.

### 1. Auxiliary One-to-Many Matching

**Problem it solves:** In the original MapTR with one-to-one Hungarian matching, each GT element supervises only one query per training step. With M=50 queries and 15-20 GT elements per scene, most queries receive only background supervision. This leads to slow convergence.

**Solution:** Add auxiliary one-to-many matching heads during training:
- The primary head still uses one-to-one matching (for inference)
- K auxiliary heads each use one-to-many matching: each GT element is matched to K predictions
- This provides K-times denser supervision signal

```
┌───────────────────────────────────────────────────────────────┐
│                                                                 │
│  Original MapTR (one-to-one):                                  │
│                                                                 │
│  Queries:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8  Q9  Q10          │
│              \       |       /                                  │
│               \      |      /                                   │
│                GT1  GT2  GT3                                    │
│                                                                 │
│  Only 3 queries get geometry supervision per step!             │
│                                                                 │
├───────────────────────────────────────────────────────────────┤
│                                                                 │
│  MapTRv2 (one-to-many auxiliary):                              │
│                                                                 │
│  Queries:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8  Q9  Q10          │
│            \\  \   ||   /  //  \\  |   /                       │
│             \\  \  ||  /  //    \\ | //                         │
│              GT1  GT2  GT3   GT1 GT2 GT3                       │
│              (primary head)    (auxiliary head)                 │
│                                                                 │
│  More queries receive geometry supervision --> faster learning │
│                                                                 │
└───────────────────────────────────────────────────────────────┘
```

- Similar in spirit to Group DETR / Hybrid DETR / DN-DETR approaches
- Auxiliary heads are removed during inference (no speed penalty)
- Convergence improves significantly: 24-epoch MapTRv2 matches 110-epoch MapTR

### 2. Decoupled Self-Attention

The original MapTR applies self-attention across all queries jointly. With M instances and N points per instance, this means attention over M*N tokens.

**Problem:** O((M*N)^2) attention is expensive and mixes instance-level and point-level reasoning:

```
Original: All M*N query tokens attend to each other
          Complexity: O((M*N)^2) = O((50*20)^2) = O(1,000,000)
```

MapTRv2 decouples this into two stages:

```
Stage 1: Instance-level self-attention
  - Aggregate each instance's N point features into 1 instance feature
  - Self-attention among M instance features
  - Complexity: O(M^2) = O(2500)
  - Purpose: Global map structure reasoning
    (e.g., "lane dividers should be parallel")

Stage 2: Point-level self-attention
  - For each instance, self-attention among its N point features
  - Done independently for each of M instances
  - Complexity: M * O(N^2) = 50 * O(400) = O(20,000)
  - Purpose: Local geometry refinement
    (e.g., "these points should form a smooth curve")

Total: O(M^2 + M*N^2) = O(2500 + 20,000) = O(22,500)
vs. original: O(1,000,000)

~44x reduction in attention computation!
```

Benefits:
- Reduces computational complexity dramatically
- Allows instance queries to capture global map structure
- Allows point queries to focus on local geometric detail
- Better gradient flow for both instance classification and point regression
- Architectural inductive bias matches the problem structure

### 3. Dense Supervision via Auxiliary BEV Segmentation Head

**Motivation:** The BEV feature encoder must learn to produce good BEV representations, but with only vectorized supervision (sparse point locations), gradient signal to the encoder is sparse.

**Solution:** Add an auxiliary rasterized segmentation head:

```
BEV Features --> Auxiliary Seg Head --> Dense BEV segmentation map
      |                                        |
      |                                        v
      |                              Dense pixel-level loss
      |                              (complements sparse vector loss)
      v
  Transformer Decoder --> Vectorized prediction (primary task)
```

- Predicts a rasterized map as additional supervision during training
- Provides dense pixel-level gradients to the BEV feature encoder
- Helps the encoder learn better spatial representations
- Removed during inference (no speed penalty)
- Complementary to the primary vectorized prediction: dense supervision helps the backbone, sparse supervision trains the decoder

### 4. Improved Training Strategy

- **More aggressive data augmentation**: Random flipping, rotation, and scaling in BEV space
- **Better learning rate scheduling**: Cosine annealing with warmup
- **Extended training**: 110 epochs for best results (vs. 24 in original paper)
- **Loss balancing**: Tuned relative weights between classification, point, and direction losses

### Combined Effect

| Improvement | mAP Gain | Speed Impact |
|-------------|----------|-------------|
| One-to-many matching | +2.5 | None (removed at inference) |
| Decoupled attention | +0.8 | Slight speedup |
| BEV segmentation aux | +0.5 | None (removed at inference) |
| Training strategy | +0.7 | None |
| **Total over MapTR** | **+4.5** | **Minimal** |

---

## Key Performance Metrics

### nuScenes Benchmark Results

| Model | Backbone | Epochs | mAP | FPS |
|-------|----------|--------|-----|-----|
| MapTR | ResNet-50 | 24 | 43.2 | 25.1 |
| MapTR | ResNet-50 | 110 | 46.3 | 25.1 |
| MapTRv2 | ResNet-50 | 24 | 46.7 | 21.8 |
| MapTRv2 | ResNet-50 | 110 | 50.3 | 21.8 |
| MapTRv2 | VoVNet-99 | 110 | 53.9 | 14.1 |

### Comparison with Baselines (nuScenes, ResNet-50, 24 epochs)

| Method | mAP | FPS |
|--------|-----|-----|
| HDMapNet | 21.7 | 3.2 |
| VectorMapNet | 36.1 | 2.9 |
| MapTR | 43.2 | 25.1 |
| MapTRv2 | 46.7 | 21.8 |

### Per-Category Performance (MapTRv2, R50, 110ep)

| Category | AP@0.5m | AP@1.0m | AP@1.5m |
|----------|---------|---------|---------|
| Pedestrian Crossing | 38.7 | 55.2 | 61.4 |
| Lane Divider | 42.1 | 58.9 | 65.3 |
| Road Boundary | 45.6 | 61.7 | 67.8 |

### Understanding the mAP Metric

The mAP metric for map construction uses **Chamfer distance** thresholds rather than IoU (unlike object detection):

1. For each predicted map element matched to a GT element, compute the Chamfer distance between them
2. A prediction is considered "correct" if Chamfer distance < threshold (0.5m, 1.0m, or 1.5m)
3. Compute precision-recall curves and average precision per category per threshold
4. mAP = mean across all categories and thresholds

The thresholds represent real-world accuracy requirements:
- **0.5m threshold**: Very strict -- requires sub-meter accuracy
- **1.0m threshold**: Moderate -- acceptable for most planning tasks
- **1.5m threshold**: Lenient -- captures general structure

---

## Practical Implementation Notes

### Memory Requirements and Batch Size

| Configuration | GPU Memory | Recommended Batch Size |
|---------------|-----------|----------------------|
| MapTR, R50, 6 cameras | ~18 GB | 1-2 per GPU |
| MapTR, R50, 6 cameras + GradCheckpoint | ~12 GB | 2-4 per GPU |
| MapTRv2, R50, 6 cameras | ~22 GB | 1-2 per GPU |
| MapTRv2, VoVNet-99 | ~28 GB | 1 per GPU |

Key memory considerations:
- **BEV feature map** (200x100x256) is relatively lightweight
- **Multi-scale image features** from 6 cameras are the dominant memory cost
- **Transformer decoder** with M=50 instances x N=20 points per instance is manageable
- **Permutation search** during training adds memory for polygon elements (storing 2N distance computations)

Practical tips:
- Use gradient checkpointing for the backbone to save ~6 GB
- FP16 mixed precision reduces memory by ~30% with minimal accuracy loss
- Reduce image resolution (from 900x1600 to 448x800) if memory-constrained -- accuracy drops ~2-3 mAP

### Training Time and Convergence Behavior

Typical training times on 8x A100 GPUs:

| Setting | 24 epochs | 110 epochs |
|---------|-----------|------------|
| MapTR, R50 | ~12 hours | ~55 hours |
| MapTRv2, R50 | ~14 hours | ~60 hours |
| MapTRv2, VoVNet-99 | ~20 hours | ~85 hours |

Convergence behavior:
- **Epochs 1-5**: Loss drops rapidly as model learns coarse BEV structure
- **Epochs 5-15**: Instance-level matching stabilizes, point regression improves
- **Epochs 15-24**: Fine-grained geometry refinement, diminishing returns start
- **Epochs 24-60**: Continued slow improvement in difficult cases (thin elements, distant objects)
- **Epochs 60-110**: Marginal gains, mainly on edge cases

```
mAP
 50 |                                              ______
    |                                        _____/
 45 |                                  _____/
    |                            _____/
 40 |                      _____/
    |                 ____/
 35 |            ____/
    |       ____/
 30 |  ____/
    | /
 25 |/
    └───────────────────────────────────────────────────── Epoch
    0    10   20   30   40   50   60   70   80   90  110
         |         |                             |
         Rapid     Diminishing returns           Marginal
         learning  start here                    gains
```

**MapTRv2's faster convergence**: Thanks to one-to-many matching, MapTRv2 at 24 epochs already exceeds MapTR at 110 epochs. This is because each GT element supervises multiple queries, providing much denser gradient signal.

### Common Failure Cases

Understanding where MapTR struggles helps in adapting it to new scenarios:

**1. Thin and Small Elements**
```
Problem: Very short lane dividers or narrow crosswalks
Cause: Few BEV pixels cover the element, weak feature signal
Symptom: Predicted as background (missed detection)
Mitigation: Increase BEV resolution, add focal loss for small elements
```

**2. Occlusion by Large Vehicles**
```
Problem: Trucks/buses blocking camera view of road markings
Cause: No direct visual evidence of the occluded element
Symptom: Missing or distorted predictions in occluded regions
Mitigation: Temporal fusion (StreamMapNet), multi-frame aggregation
```

**3. Complex Intersections**
```
Problem: Multiple overlapping elements at intersections
Cause: High density of map elements in small area, ambiguous associations
Symptom: Merged predictions, incorrect topology
Mitigation: More queries, higher BEV resolution at intersections
```

**4. Distant Elements (>50m)**
```
Problem: Elements far from the ego vehicle
Cause: Tiny pixel footprint in perspective images, severe perspective distortion
Symptom: Inaccurate geometry, noisy point positions
Mitigation: Multi-scale BEV features, longer perception range training
```

**5. Weather and Lighting Degradation**
```
Problem: Rain, night, glare washing out road markings
Cause: Reduced visual contrast of lane lines
Symptom: Lower recall in adverse conditions
Mitigation: Data augmentation (brightness/contrast), pre-training on diverse data
```

**6. Curved Roads and Roundabouts**
```
Problem: Highly curved polylines require many points to represent
Cause: Fixed N points per element may not capture tight curvature
Symptom: Angular/jagged predictions on curves
Mitigation: Adaptive point count, Bezier parameterization (BeMapNet approach)
```

### Tips for Adapting to New Datasets

If you are applying MapTR to a dataset other than nuScenes:

**1. Define your map element taxonomy**
- What element types do you need? (lane lines, boundaries, crosswalks, stop lines, etc.)
- Are they polylines or polygons?
- How many points per element is appropriate?

**2. Calibrate the BEV range and resolution**
- nuScenes default: [-30m, 30m] x [-15m, 15m] at 0.3m resolution
- Highway scenarios: extend to [-60m, 60m] x [-30m, 30m]
- Urban parking: tighten to [-15m, 15m] x [-10m, 10m] with finer resolution

**3. Adjust the number of queries**
- nuScenes has ~20-30 elements per frame --> 50 queries is sufficient
- Dense urban scenes might need 100-200 queries
- Sparse highway scenes might only need 20-30 queries

**4. Handle class imbalance**
- Some element types (stop lines) are rare vs. common ones (lane dividers)
- Use class-weighted focal loss or repeat rare-class samples

**5. Camera configuration matters**
- nuScenes uses 6 cameras with known extrinsics
- Different camera counts or placements require re-calibrating GKT projections
- Fisheye cameras need distortion-aware projection

**6. Annotation format conversion**
- Convert your annotations to: list of (class_label, ordered_point_sequence) per frame
- Ensure polygons are properly closed (first point != last point in MapTR format)
- Normalize coordinates to the BEV range

### PyTorch Implementation Sketch

For a Staff Engineer familiar with PyTorch, here is a conceptual sketch of the key components:

```python
class MapTRDecoder(nn.Module):
    """Simplified MapTR decoder structure."""
    
    def __init__(self, num_instances=50, num_points=20, d_model=256):
        super().__init__()
        # Instance queries: learnable embeddings
        self.instance_queries = nn.Embedding(num_instances, d_model)
        # Point queries: learnable embeddings (shared across instances)
        self.point_queries = nn.Embedding(num_points, d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            MapTRDecoderLayer(d_model) for _ in range(6)
        ])
        
        # Prediction heads
        self.class_head = nn.Linear(d_model, num_classes)
        self.point_head = nn.Linear(d_model, 2)  # (x, y) per point
    
    def forward(self, bev_features):
        """
        bev_features: (B, C, H, W) - BEV feature map from GKT
        Returns: class_preds (B, M, num_classes), point_preds (B, M, N, 2)
        """
        B = bev_features.shape[0]
        M = self.instance_queries.weight.shape[0]
        N = self.point_queries.weight.shape[0]
        
        # Combine instance + point queries
        # Each instance query is combined with each point query
        inst_q = self.instance_queries.weight  # (M, C)
        pt_q = self.point_queries.weight       # (N, C)
        queries = inst_q[:, None, :] + pt_q[None, :, :]  # (M, N, C)
        queries = queries.reshape(M * N, -1).unsqueeze(0).expand(B, -1, -1)
        
        # Run through decoder layers
        for layer in self.layers:
            queries = layer(queries, bev_features)
        
        # Predict
        queries = queries.reshape(B, M, N, -1)
        class_preds = self.class_head(queries[:, :, 0, :])  # Use first point for class
        point_preds = self.point_head(queries).sigmoid()     # Normalized coords
        
        return class_preds, point_preds


def permutation_equivalent_loss(pred_points, gt_points, element_type):
    """
    Compute minimum loss over all equivalent permutations.
    
    pred_points: (N, 2) - predicted point sequence
    gt_points: (N, 2) - ground truth point sequence
    element_type: 'polyline' or 'polygon'
    """
    N = gt_points.shape[0]
    
    if element_type == 'polyline':
        # Only 2 permutations: forward and reverse
        loss_fwd = F.l1_loss(pred_points, gt_points, reduction='sum')
        loss_rev = F.l1_loss(pred_points, gt_points.flip(0), reduction='sum')
        return min(loss_fwd, loss_rev)
    
    elif element_type == 'polygon':
        # 2N permutations: N cyclic shifts x 2 directions
        min_loss = float('inf')
        
        for shift in range(N):
            # Cyclic shift
            shifted = torch.roll(gt_points, shifts=-shift, dims=0)
            loss_cw = F.l1_loss(pred_points, shifted, reduction='sum')
            
            # Reversed direction
            reversed_shifted = shifted.flip(0)
            loss_ccw = F.l1_loss(pred_points, reversed_shifted, reduction='sum')
            
            min_loss = min(min_loss, loss_cw.item(), loss_ccw.item())
        
        return min_loss
```

Note: The actual implementation vectorizes the permutation search for efficiency (computing all permutations in parallel via matrix operations rather than a Python loop).

---

## Significance and Impact

MapTR established vectorized HD map construction as a viable real-time perception task, demonstrating that:

1. Structured outputs (point sets) can be learned end-to-end without intermediate rasterization
2. Permutation-equivalent modeling is essential for geometric primitive prediction
3. Hierarchical matching effectively decomposes the assignment problem
4. The approach generalizes across different map element types (lines, polygons)

The framework has become a foundational baseline for subsequent work in online mapping, including StreamMapNet (temporal fusion), PivotNet (pivot-based representation), and BeMapNet (Bezier curve modeling).

### Broader Impact on the Field

MapTR shifted the community's approach to online mapping:

**Before MapTR (2021-2022):**
- Rasterized approaches dominated (HDMapNet, InstaGraM)
- Vectorized approaches existed but were slow (VectorMapNet's autoregressive generation)
- Real-time online mapping seemed incompatible with vector output

**After MapTR (2023-present):**
- Parallel vectorized prediction became the default paradigm
- Permutation-equivalent losses adopted by nearly all follow-up works
- Focus shifted to temporal fusion, topology reasoning, and longer-range prediction
- Industry adoption: several companies now use MapTR-style architectures in production

---

## References

- Liao, B., et al. "MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction." ICLR 2023.
- Liao, B., et al. "MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction." arXiv 2023.
- Li, Q., et al. "HDMapNet: An Online HD Map Construction and Evaluation Framework." ICRA 2022.
- Liu, Y., et al. "VectorMapNet: End-to-end Vectorized HD Map Learning." ICML 2023.
- Yuan, T., et al. "StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction." WACV 2024.
- Qiao, L., et al. "BeMapNet: End-to-End Vectorized HD-Map Construction with Piecewise Bezier Curve." CVPR 2023.
- Ding, W., et al. "PivotNet: Vectorized Pivot Learning for End-to-end HD Map Construction." ICCV 2023.
- Philion, J., Fidler, S. "Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D." ECCV 2020.
- Li, Z., et al. "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers." ECCV 2022.
