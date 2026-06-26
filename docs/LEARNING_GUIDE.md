# Autonomous Driving Perception — Complete Learning Guide

This guide is designed for engineers who know deep learning (PyTorch, CNNs, Transformers) but are new to autonomous driving perception. By the end, you'll understand how every model in this repository works and how they fit together.

---

## Part 1: The Big Picture

### What Does a Self-Driving Car Need to See?

An autonomous vehicle must answer these questions at 10-20+ Hz:

1. **Where can I drive?** → Road surface, lanes, boundaries (Static Map Semantics)
2. **What objects are around me?** → Cars, trucks, pedestrians, cyclists (Dynamic Objects)
3. **Where are those objects going?** → Velocity, trajectory prediction (Tracking)
4. **How confident am I?** → Uncertainty estimation for safe decision-making

### The Perception → Planning → Control Pipeline

```
     Sensors              Perception              Prediction            Planning
  ┌───────────┐      ┌──────────────────┐     ┌────────────┐      ┌────────────┐
  │ 6 Cameras │      │ Object Detection │     │ Trajectory │      │   Route    │
  │ 1 LiDAR   │─────→│ Map Construction │────→│  Forecast  │─────→│  Planning  │──→ Control
  │ 5 Radars  │      │ Segmentation     │     │            │      │            │
  └───────────┘      └──────────────────┘     └────────────┘      └────────────┘
                                                                         │
                      THIS REPOSITORY                                    ↓
                      covers this step                              Steer/Brake/Gas
```

### Why Not Just Use 2D Detection?

You might wonder: "Why not just run YOLO/Faster R-CNN on camera images for 2D bounding boxes?"

**Because the planning module needs 3D information:**
- Distance to objects (for safe following distance)
- Object velocity (for collision prediction)
- Road geometry in metric space (for path planning)
- Free space (where can the car physically drive?)

2D bounding boxes tell you "there's a car in the image" but not "there's a car 15 meters ahead moving at 30 km/h in the left lane."

---

## Part 2: Fundamental Representations

### 2.1 How to Represent 3D Objects

A 3D bounding box in autonomous driving is parameterized as:

```
┌────────────────────────────────────────┐
│  3D Box = (cx, cy, cz, w, l, h, θ)    │
│                                        │
│  cx, cy, cz: center position (meters)  │
│  w: width (lateral extent)             │
│  l: length (longitudinal extent)       │
│  h: height (vertical extent)           │
│  θ: yaw angle (rotation about z-axis) │
│                                        │
│  For tracking, we also predict:        │
│  vx, vy: velocity (m/s)               │
└────────────────────────────────────────┘

                     Top View (BEV)              Side View
                    
                    ←── l ──→                    ←── l ──→
                    ┌────────┐                   ┌────────┐
                ↑   │        │                ↑  │        │
                w   │  (cx,  │                h  │  (cx,  │
                ↓   │   cy)  │                ↓  │   cz)  │
                    └────────┘                   └────────┘
                         ↗ θ (yaw)
```

### 2.2 Bird's Eye View (BEV) — The Unified Representation

BEV is a 2D grid where each cell represents a physical area of the ground plane:

```
BEV Grid Specification:
- x_range: [-50m, +50m] (lateral, left-right)
- y_range: [-50m, +50m] (longitudinal, front-back)  
- resolution: 0.5 m/pixel
- Grid size: 200 × 200 pixels
- Each pixel = 0.5m × 0.5m ground area

Physical World                          BEV Feature Map
(top-down view)                        (what the network sees)

   50m ┌──────────────────┐            ┌──────────────────┐
       │    car A          │            │  ■■               │  Channel 0: occupancy
       │         car B     │  ───→      │       ■■          │  Channel 1: height
       │                   │            │                   │  Channel 2: velocity_x
       │    [EGO CAR]      │            │    [center]       │  ...
       │                   │            │                   │  Channel C: class features
  -50m └──────────────────┘            └──────────────────┘
      -50m              50m             0              200
```

**Why BEV is so powerful:**
1. All sensors project naturally to the same BEV plane
2. Distance measurements are trivial (just pixel distance × resolution)
3. Planning operates in BEV space (path = sequence of BEV coordinates)
4. Temporal alignment is straightforward (apply ego-motion rotation + translation)

### 2.3 Point Cloud Representations

LiDAR gives us unstructured 3D points. We need to impose structure for neural networks:

| Representation | How It Works | Pros | Cons | Models |
|---------------|-------------|------|------|--------|
| **Raw Points** | Process points directly with PointNet | No information loss | Slow, limited receptive field | PointNet++ |
| **Voxels** | Divide space into 3D grid cells | Enables 3D convolutions | Quantization, memory-heavy | CenterPoint |
| **Pillars** | Divide into vertical columns (2D grid) | Fast (avoids 3D conv), real-time | Loses height detail | PointPillars |
| **Range Image** | Spherical projection to 2D image | Use fast 2D CNNs | Quantization, occlusion | RangeNet++ |
| **Cylinders** | Cylindrical coordinate grid | Matches LiDAR distribution | Custom convolutions | Cylinder3D |

### 2.4 Map Representations

Two ways to represent road structure:

```
Rasterized (pixel-based)                Vectorized (polyline-based)
┌────────────────────┐                  Points:
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                  Lane 1: [(x1,y1), (x2,y2), (x3,y3), ...]
│░░░░░░░░░░░░░░░░░░│                  Lane 2: [(x1,y1), (x2,y2), ...]
│████████████████████│                  Boundary: [(x1,y1), ...]
│░░░░░░░░░░░░░░░░░░│                  
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                  Advantages:
└────────────────────┘                  - Compact (few points vs millions of pixels)
                                        - Directly usable by planner
Advantages:                             - Sub-pixel accuracy
- Simple (per-pixel classification)     - No post-processing needed
- Uses standard segmentation networks   
                                        Used by: StreamMapNet, MapTR
Used by: HDMapNet
```

---

## Part 3: Camera-Based Perception (The Hardest Problem)

### 3.1 The Core Challenge: Depth Ambiguity

Cameras give us 2D images but we need 3D information. This is fundamentally ill-posed:
- A person 10m away and a toy 1m away can produce the same 2D appearance
- We must LEARN depth from context (ground plane, object sizes, perspective cues)

### 3.2 Three Approaches to Camera→3D

```
Approach 1: EXPLICIT DEPTH (LSS, HDMapNet, StreamMapNet)
┌─────────┐     ┌────────────┐     ┌─────────────┐     ┌─────────┐
│  Image  │────→│ Predict    │────→│ Outer Product│────→│  Splat  │──→ BEV
│ Features│     │ Depth Dist │     │ features×depth│    │ to Grid │
└─────────┘     └────────────┘     └─────────────┘     └─────────┘

For each pixel, predict a probability distribution over depth bins (e.g., 60 bins from 1m to 60m).
Then "lift" each pixel into 3D by weighting features by depth probability.
Finally "splat" all 3D points into the BEV grid.

Approach 2: ATTENTION-BASED (BEVFormer)
┌─────────┐     ┌────────────┐     ┌─────────────┐     ┌─────────┐
│  Image  │     │ BEV Query  │────→│ Deformable  │────→│ Updated │──→ BEV
│ Features│←────│ Positions  │     │ Cross-Attn  │     │  BEV    │
└─────────┘     └────────────┘     └─────────────┘     └─────────┘

BEV grid positions generate 3D reference points.
These project to 2D image locations.
Deformable attention samples image features at those locations.
No explicit depth prediction needed — learned implicitly through attention.

Approach 3: POSITION EMBEDDING (PETR)
┌─────────┐     ┌────────────┐     ┌─────────────┐     ┌─────────┐
│  Image  │     │ 3D Position│     │   Element-  │     │  Cross  │──→ 3D
│ Features│────→│  Encoding  │────→│   wise Add  │────→│  Attn   │   Boxes
└─────────┘     └────────────┘     └─────────────┘     └─────────┘

Encode the 3D position of each pixel (computed from camera geometry) as a positional embedding.
Add to image features so they "know" their 3D location.
Object queries attend to these 3D-aware features.
No explicit BEV representation needed.
```

### 3.3 BEVFormer Deep Dive (Most Important Camera Model)

BEVFormer combines the best ideas into one architecture:

```
Input: 6 cameras × (3, 900, 1600)
                │
                ↓
    ┌──────────────────────────┐
    │ ResNet-101 + FPN Backbone │  Extract multi-scale image features
    │ Output: 6 × 4 scales     │  C3: 1/8, C4: 1/16, C5: 1/32, C6: 1/64
    └──────────────────────────┘
                │
                ↓
    ┌──────────────────────────────────────────────────┐
    │ BEV Encoder (6 layers, each contains):           │
    │                                                   │
    │  ┌──────────────────────────────────────────┐    │
    │  │ 1. Temporal Self-Attention                │    │
    │  │    - Take BEV from previous frame         │    │
    │  │    - Warp it using ego-motion (R, t)      │    │
    │  │    - Current BEV queries attend to past   │    │
    │  │    - Uses deformable attention            │    │
    │  └──────────────────────────────────────────┘    │
    │                     ↓                             │
    │  ┌──────────────────────────────────────────┐    │
    │  │ 2. Spatial Cross-Attention                │    │
    │  │    - For each BEV query at position (x,y):│    │
    │  │      a. Create reference points at         │    │
    │  │         heights z={-2, -1, 0, 1, 2, 3}m  │    │
    │  │      b. Project each 3D point to cameras  │    │
    │  │         using K @ [R|t] (intrinsic×extr)  │    │
    │  │      c. Sample image features at those    │    │
    │  │         projected 2D locations            │    │
    │  │      d. Deformable: learn small offsets   │    │
    │  │         around reference for fine detail  │    │
    │  │      e. Weight & sum → updated BEV feat   │    │
    │  └──────────────────────────────────────────┘    │
    │                     ↓                             │
    │  ┌──────────────────────────────────────────┐    │
    │  │ 3. Feed-Forward Network (FFN)             │    │
    │  │    - Linear → GELU → Linear → Residual   │    │
    │  └──────────────────────────────────────────┘    │
    │                                                   │
    └──────────────────────────────────────────────────┘
                │
                ↓  BEV Features: [200, 200, 256]
    ┌──────────────────────────┐
    │ DETR-style Decoder       │  900 object queries
    │ Cross-attention to BEV   │  attend to BEV features
    │ Self-attention           │  queries interact
    │ FFN                      │  predict outputs
    └──────────────────────────┘
                │
                ↓
    ┌──────────────────────────┐
    │ Detection Head            │  Per query, predict:
    │ - class: softmax(10 cls) │  cx, cy, cz, w, l, h
    │ - box: Linear→10 values  │  sin(θ), cos(θ), vx, vy
    └──────────────────────────┘
```

---

## Part 4: LiDAR-Based Perception

### 4.1 Why LiDAR is Different from Images

```
Camera image:              LiDAR point cloud:
- Dense (millions of px)   - Sparse (~34k points)
- Regular grid             - Unstructured, unordered
- 2D (no depth)           - True 3D (x, y, z per point)
- Rich semantics          - Geometry only (+ intensity)
- Affected by lighting    - Works in all lighting
```

### 4.2 Point Cloud Processing Evolution

```
                  PointNet (2017)
                  "Can we learn directly from points?"
                  Process each point independently → max pool
                  Problem: No LOCAL features (only global)
                        │
                        ↓
                  PointNet++ (2017)
                  "Add hierarchical local structure"
                  FPS → Ball Query → Local PointNet → Repeat
                  Problem: Too slow for real-time detection
                        │
                ┌───────┴───────┐
                ↓               ↓
          VoxelNet (2018)    PointPillars (2019)
          "Voxelize and use   "Use pillars (vertical columns)
           3D convolutions"    and 2D CNN for speed"
          Accurate but slow   62 Hz real-time!
                │               
                ↓               
          SECOND / CenterPoint (2020-2021)
          "Sparse 3D conv + center-based detection + tracking"
          Best accuracy with reasonable speed
```

### 4.3 How PointNet++ Works (Foundation)

The key insight: learn features at multiple SCALES by hierarchically subsampling:

```
Input: N=16384 points
       │
       ↓ SA Layer 1 (Set Abstraction)
       │  
       │  Step 1: FPS (Farthest Point Sampling)
       │  Select 4096 "center" points that are spread out
       │  (iteratively pick the point farthest from all selected points)
       │  
       │  Step 2: Ball Query (radius=0.2m, K=32)
       │  For each center, find up to 32 neighbors within radius
       │  
       │  Step 3: Local PointNet
       │  For each group of 32 neighbors:
       │    - Subtract center (relative coordinates)
       │    - Shared MLP: [3+C] → 64 → 128
       │    - Max pool over 32 points → one 128-dim feature
       │  
       │  Output: 4096 points with 128-dim features
       │
       ↓ SA Layer 2
       │  FPS → 1024 centers, Ball Query (r=0.4m), MLP: 128→256
       │  Output: 1024 points with 256-dim features
       │
       ↓ SA Layer 3
       │  FPS → 256 centers, Ball Query (r=0.8m), MLP: 256→512
       │  Output: 256 points with 512-dim features
       │
       ↓ Detection Head or Feature Propagation (for segmentation)
```

### 4.4 How CenterPoint Works (Best LiDAR Detector)

```
Input: Point Cloud [N, 5]
       │
       ↓ Voxelization
       │  Divide space into tiny 3D cells (e.g., 0.075m × 0.075m × 0.2m)
       │  Only non-empty voxels are stored (sparse!)
       │  Each voxel: average of points inside → one feature vector
       │
       ↓ 3D Sparse Convolutions (4 stages)
       │  Like regular 3D conv, but ONLY on occupied voxels
       │  Submanifold conv: output only where input exists
       │  Regular sparse conv: allows slight dilation
       │  Output: sparse 3D feature volume
       │
       ↓ BEV Collapse (height compression)
       │  Take max/mean along Z-axis → 2D BEV feature map
       │
       ↓ 2D BEV Backbone (ResNet-style)
       │  Standard 2D convolutions on the BEV map
       │  Multi-scale feature extraction
       │
       ↓ Detection Heads
       │  ┌─────────────────────────────────────┐
       │  │ Center Heatmap: [H, W, num_classes] │
       │  │ Each class has its own heatmap       │
       │  │ Gaussian blob at each object center  │
       │  │ Trained with Gaussian Focal Loss     │
       │  ├─────────────────────────────────────┤
       │  │ Regression heads (per center):       │
       │  │ - offset: [2] sub-voxel offset       │
       │  │ - height: [1] z-center               │
       │  │ - size: [3] width, length, height    │
       │  │ - rotation: [2] sin(θ), cos(θ)      │
       │  │ - velocity: [2] vx, vy              │
       │  └─────────────────────────────────────┘
       │
       ↓ Inference: Find peaks in heatmap (local maxima)
       │  For each peak, read regression values
       │  No NMS needed! (one peak = one object)
       │
       ↓ Tracking (greedy matching)
         For frame t and t-1:
         - Predict where t-1 objects are at time t using their velocity
         - Match predictions to t detections by center distance
         - Unmatched detections = new tracks
         - Unmatched predictions = lost tracks (keep for N frames)
```

---

## Part 5: Temporal Modeling (Using Past Frames)

### 5.1 Why Temporal Matters

Single-frame perception:
- Cannot estimate velocity (need at least 2 frames)
- Misses occluded objects (visible from other viewpoints as car moves)
- Unstable predictions (different output each frame = dangerous for planning)

### 5.2 How to Align Past Frames to Current

The car moves between frames. To reuse past BEV features, we must **warp** them:

```
Ego-motion between frame t-1 and frame t:
  Rotation matrix R (yaw change) + Translation vector t (displacement)

Warping BEV features:
  1. Get ego-motion transform: T_{t-1→t} = T_t^{-1} @ T_{t-1}
  2. For each pixel (i, j) in current BEV grid:
     a. Convert to physical coordinates: (x, y) = (i * res + x_min, j * res + y_min)
     b. Transform to past frame: (x', y') = R^{-1} @ ((x, y) - t)
     c. Convert back to pixel: (i', j') = ((x'-x_min)/res, (y'-y_min)/res)
  3. Use grid_sample to bilinearly interpolate past features at (i', j')
  
Result: past BEV features aligned to current ego frame!
```

### 5.3 Temporal Fusion Methods

| Method | How It Works | Used By | Pros | Cons |
|--------|-------------|---------|------|------|
| **Concatenation** | Stack warped past BEV with current, conv to fuse | Simple baselines | Easy to implement | Limited temporal reasoning |
| **Temporal Attention** | Current BEV queries attend to warped past BEV | BEVFormer | Flexible, selective | More compute |
| **Query Propagation** | Pass object queries from t-1 to t | StreamPETR | Memory-efficient | Object-level only |
| **GRU/RNN** | Update state recurrently | StreamMapNet | Compact state | Gradient issues |
| **Velocity Tracking** | Predict velocity, match across frames | CenterPoint | No training needed | Simple assumptions |

---

## Part 6: Training Perception Models

### 6.1 Common Loss Functions

| Loss | Used For | Formula | Why It Works |
|------|---------|---------|-------------|
| **Focal Loss** | Classification | -α(1-p)^γ log(p) | Handles class imbalance (many easy negatives) |
| **Smooth L1** | Box regression | 0.5x² if |x|<1, else |x|-0.5 | Less sensitive to outliers than L2 |
| **Gaussian Focal** | Heatmap | Like focal but with Gaussian GT | One peak per object, smooth gradients |
| **Hungarian Loss** | Set prediction | Match pred↔GT first, then supervise | Handles permutation (order doesn't matter) |
| **Chamfer Loss** | Map polylines | Mean of closest-point distances | Works with different point counts |
| **Lovász Loss** | Segmentation | Differentiable IoU approximation | Directly optimizes IoU metric |

### 6.2 Training Schedule (Typical)

```
Epochs:   1    5    10   15   20   24 (total)
LR:      ─────────────────────────────
         │ warmup │  constant  │ cosine│
         │ linear │            │ decay │
         │ 0→2e-4 │   2e-4    │ →1e-6 │
```

### 6.3 Data Augmentation for 3D

```
Standard augmentation pipeline for LiDAR detection:

1. Random Horizontal Flip (p=0.5)
   - Flip all points: x → -x
   - Flip all boxes: cx → -cx, yaw → -yaw

2. Random Global Rotation (uniform [-π/4, π/4])
   - Rotate entire scene around z-axis
   - Apply to points AND boxes

3. Random Global Scaling ([0.95, 1.05])
   - Scale all coordinates
   - Apply to points AND box centers/sizes

4. GT Sampling (CenterPoint's secret weapon)
   - Pre-extract all GT objects from training set
   - Randomly paste 15 cars, 3 pedestrians, 5 cyclists into the scene
   - Check for collisions before placing
   - Dramatic improvement for rare classes!
```

### 6.4 Multi-GPU Training

```
# Distributed Data Parallel (DDP) — standard for perception training
python -m torch.distributed.launch --nproc_per_node=8 train.py --config cfg.yaml

# What happens:
# - Same model copied to 8 GPUs
# - Dataset split into 8 shards (each GPU sees different data)
# - Forward pass: independent on each GPU
# - Backward pass: gradients averaged across GPUs (all-reduce)
# - Effective batch size = per_gpu_batch × num_gpus

# Learning rate scaling rule:
# If base_lr works for batch_size=2 on 1 GPU,
# use lr = base_lr × num_gpus for multi-GPU
# (linear scaling rule from Goyal et al.)
```

---

## Part 7: Evaluation and Debugging

### 7.1 How to Interpret Results

```
Good results (BEVFormer-level):
  mAP: 0.517, NDS: 0.596
  ATE: 0.582, ASE: 0.256, AOE: 0.375, AVE: 0.286, AAE: 0.187

What each number means:
  mAP 0.517 → "On average, 51.7% of objects are correctly detected"
  NDS 0.596 → "Overall quality (detection + localization) is 59.6%"
  ATE 0.582 → "Detected objects are off by ~58cm on average"
  ASE 0.256 → "Size estimation error is ~25.6%"
  AOE 0.375 → "Orientation is off by ~0.375 radians (~21.5°)"
  AVE 0.286 → "Velocity error is ~0.29 m/s"

Red flags:
  mAP < 0.30 → Something is wrong (bad augmentation, learning rate, or data loading bug)
  ATE > 1.5m → Box centers are very inaccurate (depth estimation failing)
  AVE > 1.0 → Velocity estimation broken (temporal model not working)
```

### 7.2 Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss explodes (NaN) | LR too high, or bad data sample | Reduce LR, add gradient clipping |
| Loss doesn't decrease | LR too low, or bug in loss function | Increase LR, verify loss computation |
| Good mAP but bad ATE | Detection works but depth is off | Check camera calibration loading |
| Good AP far, bad AP near | Ego-motion handling bug | Verify coordinate transforms |
| Predictions flicker between frames | Temporal model not helping | Check BEV warping, ego-motion alignment |
| One class very low AP | Too few examples | Use GT sampling augmentation |
| Training crashes with OOM | BEV resolution too high, or batch too large | Reduce bev_h/bev_w, or reduce batch size |

---

## Part 8: From Research to Production

### 8.1 What Changes in Production

| Aspect | Research | Production |
|--------|---------|-----------|
| Speed | 4 FPS ok | Must be 10+ FPS (safety-critical) |
| Input | Pre-recorded data | Live sensor streams |
| Accuracy | Best benchmark score | Consistent, never catastrophically wrong |
| Failure mode | "Average metric ok" | "Worst case must be safe" |
| Testing | Val split | Billions of simulated miles |

### 8.2 Deployment Optimizations

- **TensorRT**: Convert PyTorch model to NVIDIA's optimized inference engine (2-5× speedup)
- **Quantization**: FP16 or INT8 inference (1.5-3× speedup with minimal accuracy loss)
- **Model pruning**: Remove less important weights (20-50% smaller)
- **Knowledge distillation**: Train smaller student model from large teacher

### 8.3 The Scenario Trees System

In production, you need to manage millions of recorded driving scenarios:

```
"The model fails in rain at night at intersections."
  → HOW do you find more data for this?
  → HOW do you know if your fix helps?
  → HOW do you ensure you haven't broken sunny-day performance?

Answer: Functional Scenario Trees
  1. Auto-tag all recordings (weather, time, road type, traffic)
  2. Mine for corner cases (unusual scenarios, near-misses)
  3. Generate balanced training splits (ensure coverage)
  4. Track test coverage per scenario type
  5. Detect distribution shift (new scenarios not in training data)
```

---

## Part 9: Quick Reference — Model Selection Guide

### When to Use Which Model

| Your Task | Best Model | Why |
|-----------|-----------|-----|
| Camera-only 3D detection (best accuracy) | BEVFormer | Deformable attention + temporal = SOTA |
| Camera-only 3D detection (efficient) | StreamPETR | No explicit BEV, query-based temporal |
| HD map from cameras (temporal) | StreamMapNet | Temporal fusion → stable, accurate maps |
| HD map from cameras (single-frame) | MapTR | Elegant set prediction formulation |
| LiDAR detection + tracking | CenterPoint | Best LiDAR detector with built-in tracking |
| LiDAR detection (real-time, 62 Hz) | PointPillars | Fast enough for production |
| LiDAR semantic segmentation (accurate) | Cylinder3D | Cylindrical coords match LiDAR distribution |
| LiDAR semantic segmentation (fast) | RangeNet++ | Standard 2D CNN on range image |
| Radar-only detection | RadarPillarNet | Handles radar sparsity well |
| Camera + Radar fusion | CRAFT | Cross-attention fuses complementary info |
| Data/scenario management | Scenario Trees | Auto-tag, mine, balance your data |

---

## Part 10: Next Steps After This Repository

1. **Multi-Task Models**: Combine detection + mapping + segmentation in one model (BEVerse, UniAD)
2. **Occupancy Networks**: Predict 3D voxel occupancy instead of boxes (Tesla, SurroundOcc)
3. **End-to-End Driving**: Perception directly outputs planning actions (UniAD, ThinkTwice)
4. **Foundation Models**: Large pre-trained vision models adapted for driving (DINOv2, SAM)
5. **Simulation**: Test perception in CARLA, SUMO, or internal simulators
6. **V2X Communication**: Perception augmented by infrastructure sensors
