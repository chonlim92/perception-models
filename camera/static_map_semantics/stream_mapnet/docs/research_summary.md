# StreamMapNet: Comprehensive Research Summary

**Paper:** "StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction"
**Authors:** Tianyuan Yuan, Yicheng Liu, Yue Wang, Yilun Wang, Hang Zhao
**Venue:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2024
**arXiv:** 2308.12570

This document is written for a Staff AI Engineer who knows PyTorch well but is new to
autonomous driving and HD maps. It builds concepts from the ground up.

---

## 1. What Is HD Map Construction?

### Regular Navigation Maps vs HD Maps

A regular navigation map (Google Maps, Waze) gives you road-level topology: "Turn left on
Main Street." It operates at ~5-meter accuracy and stores roads as simplified lines.

An HD (High-Definition) map operates at ~5-CENTIMETER accuracy and stores the fine-grained
geometric and semantic structure of the road:

```
Regular Nav Map (what you see on your phone):
+--------------------------------------------------+
|                                                  |
|    ====== Main St ======                         |
|              |                                    |
|              | Oak Ave                            |
|              |                                    |
+--------------------------------------------------+
Just road connectivity. No lane info. No boundaries.


HD Map (what an autonomous car needs):
+--------------------------------------------------+
|   [curb]  |lane1|lane2||lane3|lane4|  [curb]     |
|   ........|.....|.....||.....|.....|..........   |
|           | --> | --> || <-- | <-- |              |
|   ........|.....|.....||.....|.....|..........   |
|           |     |     ||     |     |              |
|   --------+-----+-----++-----+-----+--------    |
|           |  ped xing  ||  ped xing |            |
|   ========|============||===========|========    |
+--------------------------------------------------+

Contains: lane dividers, road boundaries, pedestrian crossings,
          lane directions, centerlines, stop lines, traffic signs...
```

### What an HD Map Contains

| Element              | Description                                    | Precision  |
|---------------------|------------------------------------------------|------------|
| Lane dividers       | Lines separating adjacent lanes                | ~5 cm      |
| Road boundaries     | Edge of drivable surface (curbs, barriers)     | ~5 cm      |
| Pedestrian crossings| Zebra crossings, marked pedestrian areas       | ~10 cm     |
| Centerlines         | Center of each lane (used for path following)  | ~5 cm      |
| Stop lines          | Where the car must stop at intersections       | ~5 cm      |
| Traffic signs/lights| Location and type of signals                   | ~10 cm     |
| Road surface        | Material, slope, banking angle                 | varies     |

### Why the Autonomous Driving Stack Needs HD Maps

An autonomous vehicle has a layered software architecture:

```
+--------------------+
|     Perception     |  <-- "What is around me?" (objects, lanes, free space)
+--------------------+
         |
         v
+--------------------+
|     Prediction     |  <-- "Where will other agents go?" (needs lane info)
+--------------------+
         |
         v
+--------------------+
|      Planning      |  <-- "What path should I take?" (needs road structure)
+--------------------+
         |
         v
+--------------------+
|      Control       |  <-- "What steering/throttle/brake?" (executes the plan)
+--------------------+
```

Each downstream module depends on the map:

- **Prediction module:** To predict where a pedestrian or car will go, you need to know
  where lanes are. A car approaching an intersection will likely follow lane geometry.
  Without lane information, prediction degenerates to physics-only motion models.

- **Planning module:** To decide your own path, you need road boundaries (where CAN you
  drive?), lane dividers (which lane are you in?), and crossings (where should you yield?).
  Planning without an HD map is like navigating a building without a floor plan.

- **Behavior layer:** Lane-change decisions require knowing how many lanes exist, their
  widths, and their connectivity (which lane merges into which).

### The Problem with Pre-Built Maps

Traditionally, companies like Waymo send mapping vehicles to pre-scan every road with LiDAR,
then humans annotate lane markings offline. This approach has critical limitations:

1. **Scalability:** You cannot pre-map every road in every city.
2. **Freshness:** Construction, new markings, and road changes make old maps stale.
3. **Cost:** Human annotation of centimeter-accurate maps is extremely expensive.

This motivates ONLINE HD map construction: build the map in real-time from the car's own
cameras as it drives. StreamMapNet addresses exactly this problem.

---

## 2. Vectorized Maps vs Rasterized Maps

These are two fundamentally different ways to represent the predicted map.

### Rasterized Representation

A rasterized map is a pixel grid in bird's-eye view (BEV). Each pixel is classified as
belonging to a category (lane divider, road boundary, pedestrian crossing, background).

```
Rasterized BEV Map (200 x 100 grid, each cell = 0.3m):
+--------------------------------------------------+
|  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .   |
|  .  .  .  D  .  .  D  .  .  D  .  .  .  .  .   |  D = divider pixel
|  .  .  .  D  .  .  D  .  .  D  .  .  .  .  .   |  B = boundary pixel
|  .  .  .  D  .  .  D  .  .  D  .  .  .  .  .   |  P = ped crossing pixel
|  B  B  B  D  P  P  D  P  P  D  B  B  B  B  B   |  . = background
|  .  .  .  D  .  .  D  .  .  D  .  .  .  .  .   |
|  .  .  .  D  .  .  D  .  .  D  .  .  .  .  .   |
+--------------------------------------------------+

Storage: H x W x C tensor (e.g., 200 x 100 x 3 classes)
This is just semantic segmentation in BEV.
```

**Pros of rasterized:**
- Simple to produce (standard segmentation head)
- Dense supervision at every pixel

**Cons of rasterized:**
- No explicit structure: pixels don't know they belong to the same line
- Post-processing needed to extract polylines (skeletonization, tracing)
- Lossy: resolution limits accuracy (a 0.3m grid cannot represent 5cm accuracy)
- Large: storing a 200x100x3 tensor for every frame is wasteful
- No topology: cannot tell which lane connects to which

### Vectorized Representation

A vectorized map stores map elements as ordered sequences of (x, y) points (polylines):

```
Vectorized Map (compact, structured):

Lane Divider 1:  [(2.1, -15.0), (2.1, -10.0), (2.1, -5.0), (2.1, 0.0), (2.0, 5.0)]
Lane Divider 2:  [(5.5, -15.0), (5.5, -10.0), (5.5, -5.0), (5.5, 0.0), (5.4, 5.0)]
Road Boundary 1: [(0.0, -15.0), (0.0, -10.0), (0.0, -5.0), (0.0, 0.0), (0.0, 5.0)]
Ped Crossing 1:  [(2.1, 0.0), (5.5, 0.0), (5.5, 1.0), (2.1, 1.0)]  (polygon)

Visualized in BEV:

     y (forward)
     ^
     |       D1      D2
     |       |       |
  5  +       *       *          * = control point
     |       |       |
  0  +--B1---*==P1===*---B2--   B = boundary points
     |       |       |          P = ped crossing points
 -5  +       *       *          D = divider points
     |       |       |
-15  +       *       *
     +---+---+---+---+----> x (lateral)
     0       2.1     5.5
```

**Pros of vectorized:**
- Compact: a few hundred floats vs a 200x100 grid
- Precise: points can be at arbitrary coordinates, not snapped to grid
- Structured: each polyline is a distinct instance with explicit connectivity
- Topology-aware: you know which line is which, and their order
- Directly usable by planning (planning algorithms work with polylines, not pixel grids)

**Cons of vectorized:**
- Harder to train: requires set prediction or autoregressive generation
- Supervision is trickier (need point-level matching)

### Why the Field Moved to Vectorized

The planning module thinks in terms of lanes and paths, not pixels. A vectorized map can be
directly consumed by a planner:

```
Planner input (from vectorized map):
  - "I am in lane bounded by Divider1 (left) and Divider2 (right)"
  - "Lane width = 3.4m, I am centered at 3.8m lateral offset"
  - "Pedestrian crossing at y=0.0, check for pedestrians"

Planner input (from rasterized map):
  - Raw pixel grid... needs post-processing to extract the same info
```

StreamMapNet produces VECTORIZED output: a set of polylines, each with a class label.

---

## 3. Map Elements Explained

StreamMapNet predicts three categories of map elements. Here is what each one means
physically and why it matters for driving.

### Lane Dividers

Lane dividers are the painted lines separating adjacent lanes of traffic.

```
Real world (driver's perspective):         BEV (bird's eye):

    |   |   |   |   |                       | lane | lane | lane |
    | . | . | . | . |   <-- dashed lines     |      |      |      |
    |   |   |   |   |       (lane change OK) |  .   |  .   |  .   |
    |___|___|___|___|                         |      |      |      |
    |   |   |   |   |                         |  .   |  .   |  .   |
    | | | | | | | | |   <-- solid lines       |      |      |      |
    |___|___|___|___|       (no lane change)   | |   | |   | |    |

In vectorized form, each divider is one polyline:
  divider_left  = [(x0,y0), (x1,y1), ..., (xN,yN)]
  divider_right = [(x0,y0), (x1,y1), ..., (xN,yN)]
```

**Why they matter:** Lane-keeping requires knowing where lane boundaries are. The planning
module uses dividers to compute lateral offset and decide if a lane change is legal.

### Road Boundaries

Road boundaries mark the edge of the drivable area -- curbs, barriers, guardrails, or the
edge of pavement.

```
BEV view of road boundaries:

    guardrail                            guardrail
    =========                            =========
    |  lane  |  lane  |  lane  |  lane  |
    |        |        |        |        |
    |        |        |        |        |
    =========                            =========
    curb                                 curb

Boundary polylines:
  left_boundary  = [(x0,y0), ..., (xN,yN)]  -- left edge of road
  right_boundary = [(x0,y0), ..., (xN,yN)]  -- right edge of road
```

**Why they matter:** Leaving the drivable area means hitting a curb or barrier. This is
safety-critical -- the planner must never generate a path beyond road boundaries.

### Pedestrian Crossings

Pedestrian crossings are marked areas where pedestrians have right-of-way.

```
BEV view of pedestrian crossing:

         |        |        |
         | lane 1 | lane 2 |
    -----+--------+--------+-----
    |////|////////|////////|////|   <-- pedestrian crossing (polygon)
    |////|////////|////////|////|       typically 2-4m wide
    -----+--------+--------+-----
         | lane 1 | lane 2 |
         |        |        |

In vectorized form, a crossing is a polygon:
  crossing = [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]  -- 4 corners
```

**Why they matter:** When a crossing is detected ahead, the behavior planner must:
1. Check if any pedestrian is near or in the crossing
2. Reduce speed and prepare to yield
3. Stop if a pedestrian is crossing

Without detecting crossings, the car would drive through at full speed.

---

## 4. Why Temporal Fusion Matters for Mapping

### Single-Frame Problems

Imagine you are driving and a single camera captures one frame:

```
Frame at time t (single camera, limited FOV):

    Camera sees:                What is actually there:

    /        \                  +-----------+
   / visible  \                 | full road |
  /   region   \                | with all  |
 /     only     \               | markings  |
/________________\              +-----------+

Problems:
1. Limited FOV: camera sees only part of the scene
2. Occlusion: truck blocks view of lane markings behind it
3. Distance: far-away markings are blurry, few pixels
4. Weather: rain, glare make single frame unreliable
```

A single-frame model must guess at occluded or far-away elements from one viewpoint.
This leads to noisy, incomplete predictions.

### Multi-Frame Benefits

As the car drives forward, it observes the same road region from multiple viewpoints:

```
Time t:     Car is HERE           Sees road ahead (far, blurry)
            [CAR]-->
                     .............. (lane markings, far away)

Time t+1:       Car is HERE       Same markings, now closer
                [CAR]-->
                 :::::::::::::..... (markings clearer)

Time t+2:           Car is HERE   Same markings, right under car
                    [CAR]-->
                    ||||||||||||    (markings crystal clear)

By fusing information from t, t+1, t+2:
- Markings that were blurry at t are confirmed by clear observation at t+2
- Occluded regions at t may become visible at t+1 from a different angle
- Noise averages out across multiple observations
```

### Temporal Consistency for Downstream Modules

The planning module runs at 10 Hz. If the map flickers (lane appears, disappears, shifts):

```
Frame 1: Planner sees 3 lanes -> plans for lane 2
Frame 2: Map flickers, shows 2 lanes -> planner replans for lane 1
Frame 3: Map shows 3 lanes again -> planner replans for lane 2

Result: The car oscillates, creating jerky uncomfortable motion.
Worse: flickering boundaries could cause the planner to briefly "see"
       space that does not exist, risking a collision.
```

Temporal fusion produces STABLE predictions:

```
Frame 1: Hidden state accumulates evidence -> 3 lanes (confidence: 0.7)
Frame 2: More evidence accumulated          -> 3 lanes (confidence: 0.9)
Frame 3: Strong accumulated evidence        -> 3 lanes (confidence: 0.95)

Result: Stable map, smooth planning, comfortable ride.
```

### Why Flickering Maps Are Dangerous

For safety-critical autonomous driving:
- A lane boundary that disappears for one frame might cause a planner to initiate an
  unsafe lane change into oncoming traffic.
- A pedestrian crossing that flickers off might cause the car to not yield to pedestrians.
- Temporal consistency is not just a nice-to-have -- it is a safety requirement.

---

## 5. StreamMapNet's Key Innovation Explained from Scratch

### The Core Idea: Streaming with a Hidden State

StreamMapNet borrows from recurrent neural networks (RNNs) the idea of carrying forward
a hidden state. But instead of processing a sequence of tokens, it processes a sequence
of driving frames:

```
Traditional RNN:                    StreamMapNet:

  h0 -> [RNN] -> h1                  H0 -> [Warp+Fuse] -> H1
         ^                                    ^
         |                                    |
        x1 (input token)                   BEV_1 (current BEV features)

  h1 -> [RNN] -> h2                  H1 -> [Warp+Fuse] -> H2
         ^                                    ^
         |                                    |
        x2 (input token)                   BEV_2 (current BEV features)
```

The key difference from a standard RNN: before feeding the hidden state forward,
StreamMapNet WARPS it using the vehicle's ego-motion. This spatial alignment is critical
because the car has physically moved between frames.

### Ego-Motion Warping: Why and How

Between frame t and frame t+1, the car moves (say, 1 meter forward and 0.02 radians to
the right). The BEV feature map from frame t is in the PREVIOUS coordinate frame.
To combine it with frame t+1's features, we must align them spatially.

```
Frame t coordinate system:              Frame t+1 coordinate system:
(car was HERE)                          (car is now HERE)

      ^ y                                     ^ y
      |                                       |
      |   [lane marking at (3, 10)]           |   [same marking, now at (2.98, 9)]
      |                                       |
   [CAR] --> x                             [CAR] --> x

The marking didn't move in the world, but its coordinates changed
because the car (and its coordinate frame) moved forward by 1m.
```

Ego-motion warping applies the inverse of the vehicle's motion to resample the old
hidden state into the new coordinate frame:

```python
# Conceptual implementation
def warp_hidden_state(H_prev, ego_motion_matrix):
    """
    H_prev: (B, C, H, W) - previous hidden state in BEV
    ego_motion_matrix: (B, 3, 3) - transformation from t-1 to t coordinates
    """
    # Create sampling grid in current frame coordinates
    grid = make_bev_grid(H_prev.shape[-2:])  # (H, W, 2)

    # Transform to previous frame coordinates
    grid_in_prev = apply_transform(grid, ego_motion_matrix.inverse())

    # Resample previous hidden state at transformed locations
    H_warped = F.grid_sample(H_prev, grid_in_prev, mode='bilinear')

    return H_warped  # Now aligned to current frame
```

### No Re-Processing of History

This is the critical efficiency advantage. Consider alternatives:

```
Approach A: "Re-process all history" (BEVFormer-style)
  Frame 5: encode frame 1, 2, 3, 4, 5 -> attend to all 5 BEV features -> predict
  Cost: O(T) per frame, where T = number of history frames
  Memory: must store T BEV feature tensors

Approach B: "Streaming" (StreamMapNet)
  Frame 5: take H_4 (single tensor), warp it, fuse with BEV_5 -> predict
  Cost: O(1) per frame, regardless of history length
  Memory: one hidden state tensor only

  +-------+     +-------+     +-------+     +-------+     +-------+
  | BEV_1 |     | BEV_2 |     | BEV_3 |     | BEV_4 |     | BEV_5 |
  +---+---+     +---+---+     +---+---+     +---+---+     +---+---+
      |             |             |             |             |
      v             v             v             v             v
  [Fuse(H0)] -> [Fuse(H1)] -> [Fuse(H2)] -> [Fuse(H3)] -> [Fuse(H4)] -> H5
       H1           H2           H3           H4           H5
      |             |             |             |             |
      v             v             v             v             v
   pred_1       pred_2       pred_3       pred_4       pred_5
```

At frame 5, StreamMapNet has effectively "seen" all 5 frames of information
(compressed into H4), but only does one forward pass through the backbone.

### The Full Forward Pass (Single Frame)

```
Input: 6 surround-view camera images at time t, ego-motion from t-1 to t, H_{t-1}

Step 1: Image Backbone (ResNet-50)
  6 images -> 6 feature maps (each C x H/16 x W/16)

Step 2: BEV Encoder (Lift-Splat-Shoot or cross-attention)
  6 feature maps -> single BEV feature grid (C x BEV_H x BEV_W)
  e.g., 256 x 200 x 100 (covering 60m x 30m around the car)

Step 3: Temporal Module
  H_warped = warp(H_{t-1}, ego_motion)       # align to current frame
  H_t = fuse(BEV_t, H_warped)               # combine current + history

Step 4: Map Decoder (DETR-style)
  learnable queries (one per map element) attend to H_t
  output: set of polylines with class labels and point coordinates

Step 5: Hungarian Matching + Loss
  match predicted polylines to ground-truth polylines (permutation-invariant)
  loss = classification loss + point regression loss (Chamfer or L1)
```

---

## 6. How StreamMapNet Compares to Prior Work

### HDMapNet (Li et al., ICRA 2022)

**What it does:** Predicts a rasterized BEV segmentation map. Each pixel is classified as
lane divider, boundary, or crossing.

**Architecture:**
```
6 cameras -> Backbone -> IPM (Inverse Perspective Mapping) -> BEV features
                                                                  |
                                                                  v
                                                        Segmentation head
                                                                  |
                                                                  v
                                                     Pixel-level predictions
                                                                  |
                                                     Post-processing (skeleton)
                                                                  |
                                                                  v
                                                          Vectorized output
```

**What it got right:** Proved camera-only BEV map construction is feasible.
**What it missed:** Rasterized output requires fragile post-processing. No temporal info.
No instance-level understanding (which pixels form which lane line?).

### VectorMapNet (Liu et al., ICML 2023)

**What it does:** First end-to-end vectorized map prediction. Uses an autoregressive
transformer to generate polyline points one-by-one.

**Architecture:**
```
6 cameras -> Backbone -> BEV features -> Element detector (proposes instances)
                                              |
                                              v
                                    Autoregressive decoder
                                    "Given class + previous points,
                                     predict next point"
                                    point_1 -> point_2 -> ... -> point_N
```

**What it got right:** First to predict vectorized output directly without post-processing.
Showed that transformers can learn map structure.
**What it missed:** Autoregressive generation is SLOW (sequential, cannot parallelize).
Error accumulates along the sequence. No temporal fusion.

### MapTR (Liao et al., ICLR 2023)

**What it does:** Parallel set prediction of vectorized map elements using a DETR-like
architecture with Hungarian matching. Each polyline is predicted in one shot.

**Architecture:**
```
6 cameras -> Backbone -> BEV features -> Transformer decoder with:
                                           - Learnable instance queries
                                           - Learnable point queries per instance
                                           - Parallel prediction of all points
                                         |
                                         v
                                    Set of polylines (all at once)
                                         |
                                    Hungarian matching to GT
```

**What it got right:** Fast parallel decoding. Permutation-invariant matching handles the
unordered nature of map elements elegantly. Strong accuracy jump over prior work.
**What it missed:** Still single-frame. No temporal information. Predictions flicker.

### StreamMapNet (Yuan et al., WACV 2024)

**What it adds on top of MapTR:**

```
MapTR pipeline + Temporal Streaming Module:

  H_{t-1} ----[warp with ego-motion]----> H_warped
                                              |
  BEV_t ------------------------------------->+---[fusion]---> H_t
                                                                |
                                                                v
                                                         Map Decoder
                                                                |
                                                                v
                                                        Polyline predictions
```

**What it got right:** Temporal fusion with minimal overhead. Streaming design scales to
arbitrary history length at constant cost. First temporal method for vectorized mapping.

### Summary Comparison Table

| Feature              | HDMapNet     | VectorMapNet | MapTR        | StreamMapNet   |
|---------------------|--------------|--------------|--------------|----------------|
| Year                | 2022         | 2023         | 2023         | 2024           |
| Output format       | Rasterized   | Vectorized   | Vectorized   | Vectorized     |
| Decoder type        | CNN seg head | Autoregressive| Parallel set | Parallel set   |
| Temporal fusion     | None         | None         | None         | Streaming      |
| Matching strategy   | Per-pixel    | Sequential   | Hungarian    | Hungarian      |
| Inference speed     | Fast         | Slow         | Fast         | Fast (+<10%)   |
| Temporal consistency| Poor         | Poor         | Poor         | Strong         |
| nuScenes mAP       | 23.4         | 36.1         | 50.3         | 54.1           |
| Post-processing     | Required     | None         | None         | None           |

---

## 7. Paper Results and What They Mean

### nuScenes Validation Set (24 epochs, ResNet-50)

| Method         | Divider AP | Ped Crossing AP | Boundary AP | mAP  |
|---------------|-----------|----------------|-------------|------|
| HDMapNet      | 18.5      | 14.1           | 37.6        | 23.4 |
| VectorMapNet  | 36.2      | 28.5           | 43.5        | 36.1 |
| MapTR         | 51.5      | 46.3           | 53.1        | 50.3 |
| **StreamMapNet** | **56.3** | **50.1**      | **55.8**    | **54.1** |

**Interpretation:**
- StreamMapNet gains +3.8 mAP over MapTR, its single-frame baseline.
- The gain is consistent across all three element types.
- Dividers and crossings benefit most from temporal fusion (they are thin, easily occluded).
- Boundaries benefit less (they are large structures visible from most viewpoints).

### Argoverse 2 Validation Set

| Method         | Divider AP | Ped Crossing AP | Boundary AP | mAP  |
|---------------|-----------|----------------|-------------|------|
| MapTR         | 58.7      | 52.1           | 60.3        | 57.0 |
| **StreamMapNet** | **62.4** | **56.8**      | **63.1**    | **60.8** |

**Interpretation:**
- +3.8 mAP gain also holds on Argoverse 2, confirming generalization.
- Argoverse 2 has different sensor configurations (7 cameras vs 6, different resolution).
- Higher absolute numbers because Argoverse 2 annotations are cleaner and more consistent.

### Ablation Study: What Each Component Contributes

| Configuration                          | mAP  | Delta vs Baseline |
|---------------------------------------|------|-------------------|
| Single-frame (MapTR baseline)         | 50.3 | --                |
| + Ego-motion warping only             | 52.1 | +1.8              |
| + Temporal concatenation fusion       | 52.8 | +2.5              |
| + Temporal cross-attention fusion     | 54.1 | +3.8              |
| + Multi-frame propagation (3 frames)  | 54.1 | +3.8              |

**Key findings:**
1. **Ego-motion warping alone helps (+1.8):** Simply aligning and concatenating past features
   without any learned fusion already improves accuracy. This proves spatial alignment matters.

2. **Learned fusion adds more (+2.5 to +3.8):** Cross-attention fusion outperforms naive
   concatenation because it can selectively attend to useful history and ignore stale info.

3. **Diminishing returns beyond 2 frames:** Going from 2-frame to 3-frame propagation does
   not help further, suggesting the hidden state already captures relevant history efficiently.

### Computational Cost Analysis

| Method                        | Extra FLOPs/frame | Extra Memory | Total Latency |
|------------------------------|-------------------|--------------|---------------|
| MapTR (single-frame)         | 0                 | 0            | 42 ms         |
| BEVFormer temporal (4 frames)| ~3.2 GFLOPs       | 4x BEV feats | 68 ms         |
| StreamMapNet                 | ~0.8 GFLOPs       | 1x hidden    | 46 ms         |

**Interpretation:**
- StreamMapNet adds only ~4 ms latency (< 10% overhead) for a +3.8 mAP gain.
- BEVFormer's temporal approach costs 4x more memory and 26ms more latency.
- At deployment on embedded hardware (Orin, Xavier), this efficiency difference matters.

---

## 8. Limitations and Future Directions

### Current Limitations

**1. Error Accumulation in Long Sequences**
The hidden state can accumulate drift over very long sequences. Small errors in ego-motion
estimation compound over hundreds of frames. In practice, GPS/IMU on modern vehicles provides
sufficiently accurate ego-motion, but this remains a theoretical concern for degraded-sensor
conditions.

**2. Scene Transitions**
In datasets like nuScenes, sequences have hard cuts between different locations. The hidden
state from one scene is meaningless for the next. The model must learn to implicitly "reset"
-- or a simple heuristic (reset on large ego-motion jumps) can be applied at inference.

**3. Camera-Only Limitation**
StreamMapNet uses only camera inputs. LiDAR provides direct 3D geometry that could improve
BEV feature quality, especially in challenging lighting or weather. A multi-modal extension
would be straightforward but requires LiDAR sensors.

**4. Static Map Assumption**
The method assumes map elements are permanent. Construction zones, temporary lane markings,
or closed lanes are not explicitly handled. The hidden state might "remember" a lane marking
that has been physically removed during the drive.

**5. Fixed Perception Range**
The BEV grid covers a fixed area (typically 60m forward, 30m lateral). Elements beyond this
range are not predicted, even though they might be visible in high-resolution images.

### Future Directions

**1. Multi-Modal Fusion (Camera + LiDAR)**
Adding LiDAR point clouds to improve BEV feature quality. The streaming paradigm is
orthogonal to the sensor modality -- it works with any BEV encoder.

**2. Longer-Range Prediction**
Extending the perception range beyond 60m using higher resolution or multi-scale BEV grids.
Important for highway driving where the car needs to see lane structure 100m+ ahead.

**3. Dynamic Map Elements**
Detecting temporary changes (construction cones, temporary barriers) and distinguishing
them from permanent map structure. Could use change detection between the current observation
and the propagated hidden state.

**4. Uncertainty Estimation**
Providing confidence estimates per map element. Downstream planning can then be more
conservative in regions where the map is uncertain.

**5. End-to-End Training with Planning**
Training the map predictor jointly with the planning module, so the map representation
is optimized for planning quality, not just geometric accuracy.

**6. Foundation Models for BEV**
Using large pre-trained vision foundation models (DINOv2, SAM) as the image backbone,
potentially improving generalization to unseen road types and geographies.

---

## Summary for the Practitioner

If you are implementing or extending StreamMapNet, the mental model is:

1. You have a standard BEV perception pipeline (cameras -> backbone -> BEV features).
2. You add a lightweight temporal module that maintains ONE hidden state tensor.
3. Each frame: warp the old hidden state using vehicle odometry, fuse with new BEV features.
4. Feed the fused representation to a DETR-style decoder that outputs polylines.
5. Train with sequences, supervise every frame, use Hungarian matching.

The magic is in the streaming design: you get the benefit of multi-frame fusion at nearly
single-frame cost. The ego-motion warping ensures spatial consistency, and the learned fusion
module decides how much to trust history vs. current observation.

This approach is now the foundation for many subsequent works in temporal online mapping.

---

## References

- Yuan, T., Liu, Y., Wang, Y., Wang, Y., & Zhao, H. (2024). StreamMapNet: Streaming
  Mapping Network for Vectorized Online HD Map Construction. WACV 2024.
- Liao, B., et al. (2023). MapTR: Structured Modeling and Learning for Online Vectorized
  HD Map Construction. ICLR 2023.
- Li, Q., et al. (2022). HDMapNet: An Online HD Map Construction and Evaluation Framework.
  ICRA 2022.
- Liu, Y., et al. (2023). VectorMapNet: End-to-end Vectorized HD Map Learning. ICML 2023.
- Li, Z., et al. (2022). BEVFormer: Learning Bird's-Eye-View Representation from
  Multi-Camera Images via Spatiotemporal Transformers. ECCV 2022.
- Philion, J. & Fidler, S. (2020). Lift, Splat, Shoot: Encoding Images from Arbitrary
  Camera Rigs by Implicitly Unprojecting to 3D. ECCV 2020.
