# Radar Occupancy Grid Mapping — Research Summary and Teaching Guide

This document teaches radar-based occupancy grid mapping from scratch. It is written for someone who understands basic deep learning but is new to radar perception and occupancy grids for autonomous driving.

---

## What Is an Occupancy Grid?

An occupancy grid divides the world into a regular grid of cells and asks one question per cell: **is this space occupied or free?**

```
Occupancy Grid (Bird's Eye View):

   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
   │  │  │  │  │  │  │  │  │  │  │  Key:
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤  ░░ = Free (drivable)
   │  │  │  │  │██│██│  │  │  │  │  ██ = Occupied (obstacle)
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤     = Unknown
   │  │  │░░│░░│░░│░░│░░│  │  │  │
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
   │  │░░│░░│░░│░░│░░│░░│░░│  │  │  Resolution: 0.2m × 0.2m
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤  Range: 50m × 50m
   │  │░░│░░│░░│▓▓│░░│░░│░░│  │  │  Grid size: 250 × 250 cells
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
   │  │░░│░░│░░│░░│░░│░░│░░│  │  │  ▓▓ = Ego vehicle (center)
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
   │  │  │░░│░░│░░│░░│░░│  │  │  │
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
   │  │  │  │  │██│██│██│  │  │  │
   ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
   │  │  │  │  │  │  │  │  │  │  │
   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

### Why Occupancy Grids Matter for Autonomous Driving

1. **Planning needs free space**: The planner needs to know WHERE the car can drive — not just where objects are, but where they AREN'T
2. **Handles arbitrary shapes**: Object detection gives bounding boxes; occupancy gives the actual shape (walls, guardrails, arbitrary obstacles)
3. **Sensor-agnostic output**: Camera, LiDAR, and radar can all produce occupancy grids — and they can be fused together
4. **Handles unknown objects**: A shopping cart, a fallen tree, debris — things not in your object detector's class list still show as "occupied"

### Why Radar Specifically?

Radar is the **most robust** automotive sensor:

```
Weather Performance:
                    Camera    LiDAR    Radar
  Clear day          ★★★★★    ★★★★★    ★★★★★
  Light rain         ★★★★     ★★★★     ★★★★★
  Heavy rain         ★★       ★★★      ★★★★★
  Fog                ★        ★★       ★★★★★
  Snow               ★★       ★★★      ★★★★★
  Dust storm         ★        ★        ★★★★★
  Direct sunlight    ★★       ★★★★★    ★★★★★
  Night              ★★       ★★★★★    ★★★★★
```

But radar has ONE major weakness: **sparsity**.

---

## The Radar Sparsity Problem

A single radar frame gives you only 100-300 detections. Compare:
- Camera: 1920×1080 = 2 million pixels
- LiDAR: 30,000-300,000 points per scan
- Radar: 100-300 detections

```
Single Frame Comparison (Bird's Eye View):

LiDAR Point Cloud:                    Radar Detections:
··············································    ·   ·       ·
·  ····  ········  ·  ·  ·  ··               ·     ·
·  ·  ·  ·  ·  ·  ·  ·  ·                        ·
·  ····  ····  ·  ·  ·  ·  ·                ·
·  ·  ·  ····  ·  ·  ·  ·  ·       ·
·  ····  ·  ·  ·  ·  ·  ·  ·  ·             ·
                                        ·

Dense → can see everything            Sparse → huge gaps!
```

### How Occupancy Grids Solve Sparsity: Temporal Accumulation

The ego vehicle moves over time. Radar scans the environment from different positions. By **accumulating** multiple frames (after compensating for ego motion), we build up a dense representation:

```
Frame 1:        Frame 2:        Frame 3:        Accumulated:
(sparse)        (sparse)        (sparse)        (DENSE!)

  ·   ·           · ·            · ·          · · · · · ·
    ·           ·     ·        ·    ·         · · · · · ·
  ·                 ·           ·   ·         · · · · · ·
      ·          ·               ·            · · · · · ·

Each frame adds a few points.    After 10-20 frames:
                                  a complete picture emerges!
```

This is WHY temporal fusion is CRITICAL for radar (and optional for LiDAR which is already dense).

---

## Radar Sensor Fundamentals

### What Does Radar Measure?

Each radar detection provides:

| Measurement | Symbol | Accuracy | Description |
|-------------|--------|----------|-------------|
| Range | r | ±0.1m | Distance to target |
| Azimuth angle | θ | ±1-2° | Horizontal direction |
| Elevation angle | φ | ±5-10° | Vertical direction (poor!) |
| Radial velocity | v_r | ±0.1 m/s | Speed toward/away from sensor |
| RCS | σ | ~2 dB | Radar cross section (reflectivity) |
| SNR | - | - | Signal-to-noise ratio (confidence) |

### Why Is Angular Resolution So Poor?

Angular resolution ∝ wavelength / antenna_aperture.

- LiDAR wavelength: 905 nm or 1550 nm (tiny!) → excellent angular resolution
- Radar wavelength: 3.9 mm (77 GHz) → 4000× larger → much worse resolution

Even with MIMO antenna arrays (12 Tx × 16 Rx = 192 virtual elements), automotive radar achieves ~1° azimuth at best, vs LiDAR's ~0.1°.

### Radar Artifacts and Challenges

```
Multipath Reflections:

  Real car                    Ghost detection
     │                            │
     ▼                            ▼
   ┌───┐                       (ghost)
   │CAR│                          .
   └───┘                          .
     /\                          /
    /  \  ←── radar beam       / ←── reflected off guardrail
   /    \   bounces off       /      then hits car
  ▓▓▓▓▓▓▓  guardrail       ▓▓▓▓▓▓▓▓▓▓

Result: you see TWO cars — one real, one phantom!

Other challenges:
- Ground clutter: road surface causes false returns at close range
- Extended targets: a truck produces 5-10 detections along its side
- Missing detections: some materials (plastic, rubber) are radar-invisible
```

---

## Approach 1: Classical Inverse Sensor Model (ISM)

### Core Idea

For each radar detection, we can reason about what it tells us:
- **Between the sensor and the detection**: space MUST be free (beam traveled through it)
- **At the detection range**: space is likely OCCUPIED (something reflected the beam)
- **Beyond the detection**: UNKNOWN (beam was absorbed/reflected, no information)

```
Inverse Sensor Model for ONE Detection:

   Sensor              Detection          Beyond
     │                    │                 │
     ▼                    ▼                 ▼
  ═══════════════════════╪═════════════════···
  ░░░░░░░░░░░░░░░░░░░░░░██████···············
  
  P(occupied):
  low (FREE)              HIGH            0.5 (UNKNOWN)
  
  This is modeled as a cone in 2D/3D (not a line) because
  radar has finite angular resolution:
  
              /░░░░░░░░░████\
    Sensor ──<░░░░░░░░░░████ >──→ Unknown
              \░░░░░░░░░████/
                     ↑
           Cone width = angular resolution
```

### Log-Odds Bayesian Fusion

Each new observation updates our belief about occupancy using Bayes' rule. In log-odds form:

```
L(occ | z_{1:t}) = L(occ | z_t) + L(occ | z_{1:t-1}) - L_0

Where:
  L(x) = log(P(x) / (1 - P(x)))  (log-odds transform)
  L_0 = log(0.5 / 0.5) = 0        (prior, no information)
  
Advantages of log-odds:
  - Multiplication of probabilities → simple addition
  - Numerically stable (no underflow to 0 or overflow to 1)
  - Temporal fusion = just add the new observation
  
Back to probability:
  P(occ) = 1 / (1 + exp(-L(occ)))
```

### Step-by-Step: Building an Occupancy Grid Classically

```python
import numpy as np

class ClassicalRadarOccupancy:
    def __init__(self, grid_size=250, resolution=0.2, range_max=50.0):
        self.grid = np.zeros((grid_size, grid_size))  # log-odds
        self.resolution = resolution
        self.range_max = range_max
        
    def update(self, radar_detections, ego_to_world):
        """Update grid with new radar detections."""
        for det in radar_detections:
            r, theta = det['range'], det['azimuth']
            
            # Bresenham ray-casting from sensor to detection
            cells_free = self.ray_cast(0, 0, r, theta)
            cells_occupied = self.get_occupied_cells(r, theta)
            
            # Update log-odds
            for cell in cells_free:
                self.grid[cell] += L_FREE  # e.g., -0.4
            for cell in cells_occupied:
                self.grid[cell] += L_OCC   # e.g., +0.85
                
    def get_probability(self):
        """Convert log-odds to probability."""
        return 1.0 / (1.0 + np.exp(-self.grid))
```

### Advantages and Limitations

| Aspect | Classical ISM |
|--------|--------------|
| Training data needed | None |
| Interpretability | Fully transparent |
| Real-time capable | Yes (simple math) |
| Handles clutter | Poorly (no learning) |
| Adapts to sensor | Manual tuning needed |
| Accuracy | Moderate |
| Semantic info | No (occupied/free only) |

---

## Approach 2: Neural Occupancy Prediction (Learned)

### Why Deep Learning?

The classical approach has fixed rules. But real radar data is messy:
- Some clutter patterns are predictable (ground bounce near ego)
- Some objects consistently appear in certain configurations
- The relationship between sparse detections and true occupancy is learnable

A neural network can learn these patterns from data.

### Architecture: PillarOccNet

```
┌─────────────────────────────────────────────────────────────────┐
│                     PillarOccNet Architecture                      │
│                                                                   │
│  Radar Points ─→ Pillar Encoding ─→ BEV Pseudo-Image              │
│  [N, 7]          [P, 64]           [64, 250, 250]                │
│  (r, θ, z,                                                       │
│   v_r, RCS,                                                      │
│   x, y)                                                          │
│                                                                   │
│  BEV Pseudo-Image ─→ 2D U-Net Encoder-Decoder                   │
│  [64, 250, 250]      ├─ Encoder: ResNet-18 (2D)                 │
│                       │  [64→128→256→512] with stride-2          │
│                       └─ Decoder: Transposed convolutions         │
│                          [512→256→128→64] with skip connections   │
│                                                                   │
│  Decoder Output ─→ Prediction Heads                              │
│  [64, 250, 250]    ├─ Occupancy Head: Conv 1×1 → [1, 250, 250]  │
│                     │  (sigmoid → probability per cell)           │
│                     └─ Semantic Head: Conv 1×1 → [C, 250, 250]   │
│                        (softmax → class per cell: road, sidewalk, │
│                         vehicle, building, vegetation, unknown)    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Pillar Encoding for Radar (Detailed)

```
Step 1: Discretize ground plane into pillars (like PointPillars)
  Grid: 250 × 250 cells, each 0.2m × 0.2m
  
Step 2: Assign each radar detection to its pillar
  Most pillars are EMPTY (only 100-300 detections in 62,500 cells!)
  
Step 3: Per-pillar PointNet
  For pillar with M detections (typically 1-3 for radar):
    Input features per detection: [r, θ, z, v_r, RCS, x-x_c, y-y_c]
    
    Shared MLP: Linear(7, 64) → BN → ReLU
    Max-pool over M detections → [64]
    
Step 4: Scatter to BEV
  Place pillar features at grid positions → [64, 250, 250]
  Empty pillars → all zeros (99.5%+ of cells!)
```

### Temporal Fusion in Neural Approach

```
Option A: Feature Concatenation
  
  BEV_t   [64, 250, 250]  ←── current frame
  BEV_t-1 [64, 250, 250]  ←── ego-motion warped
  BEV_t-2 [64, 250, 250]  ←── ego-motion warped
  BEV_t-3 [64, 250, 250]  ←── ego-motion warped
  ─────────────────────────────
  Concat → [256, 250, 250] → Conv 1×1 → [64, 250, 250] → U-Net

Option B: Recurrent Fusion (GRU/ConvLSTM)
  
  For each new frame:
    BEV_t ─→ ┌────────┐ ─→ hidden_t [64, 250, 250]
             │ConvGRU │
    h_{t-1} →└────────┘
  
  hidden_t captures all temporal history in a compact state.
  No need to store past N frames!

Option C: Attention-Based (Transformer)
  
  Stack BEV features from T frames: [T, 64, 250, 250]
  Flatten spatial: [T, 64, 62500]
  Self-attention over temporal dimension
  Collapse to single frame
  
  Most powerful but most expensive.
```

### Training the Neural Approach

**Ground Truth Generation:**

Ground truth occupancy grids are created from LiDAR (which is dense enough to be treated as "truth"):

```
Training Pipeline:

  LiDAR Point Cloud (dense, 100k+ points)
     │
     ▼
  Voxelize into grid → mark cells with LiDAR points as OCCUPIED
     │
     ▼
  Ray-cast from LiDAR origin → mark traversed cells as FREE
     │
     ▼
  Remaining cells → UNKNOWN (ignore in loss computation)
     │
     ▼
  GT Occupancy Grid [H, W] with values {0=free, 1=occupied, -1=ignore}
```

**Loss Function:**

```python
# Binary cross-entropy with class balancing
# (most cells are free, so occupied cells get higher weight)

loss = -w_occ * gt * log(pred) - w_free * (1-gt) * log(1-pred)

# Typical weights:
w_occ = 5.0   # occupied cells are rare, weight them more
w_free = 1.0  # free cells are common

# Only compute loss where gt != UNKNOWN:
mask = (gt != -1)
loss = (loss * mask).sum() / mask.sum()
```

---

## Approach 3: Hybrid (Classical + Neural)

The best of both worlds: use classical ISM as a strong prior, then refine with a neural network.

```
Hybrid Pipeline:

  Radar Detections ─→ Classical ISM ─→ Accumulated Grid (log-odds)
       │                                      │
       │                                      ▼
       │                              [1, 250, 250] (prior)
       │                                      │
       └─→ Pillar Encoding ─→ BEV Features    │
              [64, 250, 250]                   │
                    │                          │
                    └──── Concatenate ──────────┘
                              │
                              ▼
                    [65, 250, 250]
                              │
                              ▼
                        2D U-Net
                              │
                              ▼
                    Refined Occupancy [1, 250, 250]

The classical prior gives the network:
- Strong initialization (already pretty good!)
- Free space rays (geometric reasoning the CNN can't easily learn)
- Temporal accumulation without needing recurrent networks
```

---

## Comparison to LiDAR Occupancy

| Aspect | Radar Occupancy | LiDAR Occupancy |
|--------|----------------|-----------------|
| Density per frame | ~100-300 points | ~30,000-300,000 points |
| Weather robustness | Excellent (all conditions) | Degrades in rain/fog |
| Range accuracy | ±0.1m | ±0.02m |
| Angular resolution | ~1-2° azimuth | ~0.1° |
| Elevation resolution | ~5-10° (very poor) | ~0.2° |
| Velocity info | Yes (Doppler, per detection) | No (requires multi-frame) |
| Cost | Low ($50-200 per sensor) | High ($1,000-75,000) |
| Temporal need | CRITICAL (too sparse otherwise) | Optional (already dense) |
| Clutter level | High (multipath, ground) | Low (clean measurements) |
| Grid quality (single frame) | Very poor | Excellent |
| Grid quality (10 frames accumulated) | Good | Excellent |
| Grid quality (50 frames accumulated) | Very good | Excellent |

---

## Key Results in Literature

### Occupancy Prediction Accuracy

| Method | Sensor | IoU (Occupied) | IoU (Free) | Note |
|--------|--------|---------------|------------|------|
| Classical ISM | Radar (1 frame) | 12% | 45% | Terrible from single frame |
| Classical ISM | Radar (20 frames) | 48% | 78% | Much better with accumulation |
| Neural (single frame) | Radar (1 frame) | 28% | 65% | Network fills gaps |
| Neural (temporal) | Radar (5 frames) | 55% | 82% | Best learned approach |
| Hybrid | Radar (20 frames) | 58% | 85% | Best overall radar approach |
| Classical ISM | LiDAR (1 frame) | 72% | 92% | LiDAR baseline for reference |

### Processing Speed

| Method | Latency | Hardware |
|--------|---------|----------|
| Classical ISM (20 frames) | 5 ms | CPU |
| Neural PillarOccNet (single frame) | 12 ms | GPU (RTX 3090) |
| Neural with temporal (5 frames) | 18 ms | GPU (RTX 3090) |
| Hybrid (ISM + Neural refinement) | 15 ms | CPU + GPU |

All approaches easily meet real-time requirements (< 100 ms).

---

## Key References

1. **Werber et al.** (2015) "Automotive Radar Gridmap Representations" — Classical ISM for automotive radar, cone model
2. **Prophet et al.** (2019) "Semantic Segmentation of Radar Occupancy Grids" — CNN on accumulated radar grids
3. **Krämer et al.** (2020) "Deep Radar Occupancy Grids" — End-to-end learned radar occupancy prediction
4. **Sless et al.** (2019) "Road Scene Understanding by Occupancy Grid" — Temporal fusion approaches comparison
5. **Scheiner et al.** (2020) "Object Detection for Automotive Radar" — Radar point cloud processing fundamentals
6. **Lim et al.** (2021) "Radar and Camera Early Fusion for Vehicle Detection" — Multi-modal occupancy
7. **Harley et al.** (2023) "Simple-BEV: What Really Matters for BEV Perception" — BEV representation comparison

---

## Practical Implementation Notes

### Grid Resolution Selection

```
Trade-off: resolution vs computation vs coverage

  Resolution    Grid Size (50m range)    Memory       Use Case
  ──────────    ────────────────────     ──────       ────────
  0.1m          1000 × 1000             4 MB         Research (high detail)
  0.2m          500 × 500               1 MB         Production (good balance)
  0.5m          200 × 200               160 KB       Fast prototype
  1.0m          100 × 100               40 KB        Long-range only
```

### Ego-Motion Compensation

Critical for temporal fusion. Past frames must be warped to current frame:

```python
def warp_grid_to_current(past_grid, past_ego_pose, current_ego_pose):
    """Warp a past occupancy grid to the current ego frame."""
    # Compute relative transform: past_ego → current_ego
    T_past_to_current = np.linalg.inv(current_ego_pose) @ past_ego_pose
    
    # Extract 2D rotation + translation (ignore z for BEV)
    R = T_past_to_current[:2, :2]
    t = T_past_to_current[:2, 3]
    
    # Apply affine warp to grid (using scipy or torch grid_sample)
    warped_grid = affine_warp(past_grid, R, t, resolution=0.2)
    return warped_grid
```

### Dealing with Dynamic Objects

Static occupancy grids assume the world is static. Moving objects create "smearing":

```
Problem: A car driving past leaves a trail of occupied cells

  Frame 1:   Frame 2:   Frame 3:   Accumulated (WRONG):
  
     ██         ██         ██        ██ ██ ██ ← ghost trail!
                                      ↑
                              Car was here in past frames
                              but has moved on!

Solutions:
1. Decay factor: L_t = α * L_{t-1} + L_new  (α=0.9 means old info fades)
2. Velocity gating: Doppler v_r > threshold → don't accumulate (it's moving)
3. Separate static/dynamic grids: Two grids — one for static, one for dynamic
4. Neural learned: Network learns to ignore dynamic objects in accumulation
```

### Radar-Specific Preprocessing

```python
def preprocess_radar_detections(raw_detections):
    """Filter and enhance raw radar detections before grid mapping."""
    detections = raw_detections.copy()
    
    # 1. Remove near-field clutter (ground bounce)
    detections = detections[detections['range'] > 1.5]  # Ignore < 1.5m
    
    # 2. SNR threshold (remove weak/unreliable detections)
    detections = detections[detections['snr'] > 10.0]  # dB
    
    # 3. Velocity-based static/dynamic separation
    static_mask = np.abs(detections['v_radial'] - ego_v_radial) < 0.5
    static_detections = detections[static_mask]
    dynamic_detections = detections[~static_mask]
    
    # 4. RCS-based confidence weighting
    # Higher RCS = more reliable detection
    detections['weight'] = np.clip(detections['rcs'] / 20.0, 0.1, 1.0)
    
    return static_detections, dynamic_detections
```

---

## Our Implementation

We provide three approaches with complete training pipelines:

### 1. Classical ISM
- Configurable cone-shaped sensor model (beam width, range resolution)
- Log-odds Bayesian fusion with configurable update weights
- Temporal accumulation over N frames with ego-motion compensation
- Velocity gating for dynamic object handling
- No training required — works out of the box

### 2. Neural PillarOccNet
- Pillar encoding (7 features: r, θ, z, v_r, RCS, x_offset, y_offset)
- 2D U-Net with ResNet-18 encoder + transposed conv decoder
- Dual-head output: occupancy (binary) + semantics (6 classes)
- Optional temporal concatenation (1-5 frames)
- Trained with weighted BCE loss + class-balanced semantic loss

### 3. Hybrid
- Classical ISM provides accumulated prior (strong geometric features)
- Neural refinement learns to fix ISM errors (clutter removal, gap filling)
- Best accuracy with less training data than pure neural
- More interpretable: can visualize what the ISM gives vs what the NN changes

---

## Summary: When to Use Which Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Quick prototype, no labeled data | Classical ISM |
| Best accuracy, have LiDAR-derived GT | Neural with temporal |
| Production (robustness + accuracy) | Hybrid |
| Real-time embedded (limited GPU) | Classical ISM or Neural (single frame) |
| Research / ablation studies | All three for comparison |

The occupancy grid is often the FINAL fusion output in a perception stack — cameras, LiDAR, and radar each produce their own grid, which are then merged into a unified world model for planning.
