# CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer

## Research Summary

### Paper Reference

Kim, Y., Shin, J., Kim, S., Lee, I., Choi, J. W., & Kum, D. (2023). *CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer*. IROS 2023.

---

## 1. Why Fuse Camera + Radar?

### 1.1 The Fundamental Complementarity

Camera and radar are complementary sensors in the most fundamental sense:

- **Camera provides WHAT** — rich semantics, appearance, texture, object class, color
- **Radar provides WHERE** — precise range, radial velocity, all-weather reliability

Neither sensor alone solves 3D object detection well:

```
Scene: A blue truck at 45 meters, moving at 30 km/h in the right lane

Camera alone sees:
+----------------------------------+
|                                  |
|        [Blue rectangle]          |  "That looks like a truck"
|        "Truck-shaped blob"       |  "But how far away? 30m? 50m? 80m?"
|                                  |  "How fast? Hard to tell from pixels"
+----------------------------------+

Radar alone detects:
    . (single point at x=45m, y=2m, vr=8.3 m/s)
    "Something at 45m, moving 30km/h"
    "But what IS it? Car? Truck? Bus? Sign?"

Camera + Radar together:
    "Blue truck at 45m, right lane, moving 30 km/h"
    → Complete understanding!
```

### 1.2 Complementary Strengths Table

| Capability | Camera | Radar | Camera+Radar Fused |
|------------|--------|-------|-------------------|
| Object classification | Excellent | Poor (RCS only) | Excellent |
| Range estimation | Poor (monocular depth) | Excellent (±0.3m) | Excellent |
| Lateral position | Good (angular) | Poor (~2° beam) | Good |
| Velocity measurement | Poor (optical flow) | Excellent (Doppler ±0.1m/s) | Excellent |
| Angular resolution | Excellent (~0.03°) | Poor (~1.5-2°) | Good |
| Weather: rain/fog | Degraded significantly | Robust | Robust |
| Night operation | Fails without lights | Unaffected | Works |
| Direct sunlight/glare | Degraded | Unaffected | Robust |
| Cost per sensor | $20-100 | $50-200 | $500 total suite |
| Data density | Dense (millions of pixels) | Extremely sparse (30-100 pts) | Complementary |

### 1.3 Why Not Just Use LiDAR?

LiDAR provides excellent 3D geometry but:
- Costs $1,000-$10,000+ per unit (vs ~$500 for full camera+radar suite)
- Degrades in heavy rain, fog, snow
- Has no direct velocity measurement
- Camera+radar is already standard on production vehicles (ADAS Level 2+)

The economic argument is compelling: every new car already ships with cameras and radars for ADAS. Making perception work with these existing sensors enables mass-market autonomy.

---

## 2. Fusion Taxonomy: Early vs Late vs Feature-Level

### 2.1 Overview

```
                    SENSOR FUSION STRATEGIES
                    
Early Fusion          Feature-Level Fusion       Late Fusion
(Data-level)          (CRAFT's approach)         (Decision-level)

Camera pixels ─┐      Camera ──→ CNN ──┐         Camera ──→ CNN ──→ Dets_C ─┐
               ├→ Net  Radar ──→ Enc ──┼→ Fuse → Head     Radar ──→ Enc ──→ Dets_R ─┼→ Merge
Radar points ──┘       ↑              ↑           ↑                                    ↑
                       Separate       Cross-       Separate                 NMS/
                       backbones      attention    pipelines               voting
```

### 2.2 Early Fusion

Concatenate raw sensor data before any processing:

**Approach:** Project radar points onto image, add as extra channels
- Image input becomes (H, W, 3+radar_channels)
- Single network processes everything together

**Pros:**
- Network sees all raw information
- Can learn arbitrary cross-modal correlations
- Simple architecture

**Cons:**
- Heterogeneous data formats (dense image vs sparse points)
- Radar's spatial resolution is much coarser — projection creates sparse, noisy channels
- Difficult to leverage pre-trained image backbones
- Examples: early radar-camera papers, some CenterFusion variants

### 2.3 Late Fusion

Run separate detectors, merge final predictions:

**Approach:** Independent detector per modality → combine bounding boxes
- Camera detector outputs 3D boxes from monocular depth
- Radar detector outputs 3D boxes from radar points
- Merge via NMS, voting, or learned fusion

**Pros:**
- Each network fully optimized for its modality
- Easy to add/remove modalities
- Graceful degradation if one sensor fails

**Cons:**
- Misses cross-modal synergies during feature learning
- Redundant computation
- Hard to resolve conflicts between modalities
- Examples: simple NMS-based fusion baselines

### 2.4 Feature-Level Fusion (CRAFT's Approach)

Separate backbones extract modality-specific features, then fuse at feature level:

**Approach:**
- Camera backbone (ResNet + FPN) extracts rich image features
- Radar backbone (pillar encoding) extracts spatial/velocity features
- Cross-attention transformer fuses features from both modalities
- Shared detection head predicts from fused features

**Pros:**
- Best of both worlds: modality-specific processing + cross-modal learning
- Attention mechanism learns WHICH features to combine
- Can handle spatial misalignment between modalities
- Leverages pre-trained backbones for each modality

**Cons:**
- More complex architecture
- Alignment/calibration errors affect fusion quality
- Examples: CRAFT, TransFusion, BEVFusion

---

## 3. CRAFT's Architecture: Step-by-Step Pipeline

### 3.1 Overall Architecture

```
                         CRAFT Architecture

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  Camera Images (6x)      Radar Points (accumulated sweeps)      │
  │       │                           │                             │
  │       ▼                           ▼                             │
  │  ┌──────────┐              ┌─────────────┐                      │
  │  │ ResNet   │              │ PillarNet   │                      │
  │  │ + FPN    │              │ (BEV enc.)  │                      │
  │  └────┬─────┘              └──────┬──────┘                      │
  │       │                           │                             │
  │       │  Image Features           │  Radar BEV Features         │
  │       │  (multi-scale)            │                             │
  │       │                           │                             │
  │       └───────────┐   ┌───────────┘                             │
  │                   ▼   ▼                                         │
  │          ┌─────────────────────┐                                │
  │          │  Spatio-Contextual  │                                │
  │          │  Fusion Transformer │                                │
  │          │  (Cross-Attention)  │                                │
  │          └──────────┬──────────┘                                │
  │                     │                                           │
  │                     ▼                                           │
  │            ┌────────────────┐                                   │
  │            │ Detection Head │                                   │
  │            │ (Heatmap+Reg.) │                                   │
  │            └────────┬───────┘                                   │
  │                     │                                           │
  │                     ▼                                           │
  │            3D Bounding Boxes                                    │
  │            (class, position, size, rotation, velocity)          │
  └─────────────────────────────────────────────────────────────────┘
```

### 3.2 Step 1: Camera Feature Extraction

The camera branch processes surround-view images (typically 6 cameras covering 360°):

- **Backbone:** ResNet-50 or EfficientNet-B4 (pre-trained on ImageNet)
- **Neck:** Feature Pyramid Network (FPN) produces multi-scale features
- **Output:** Feature maps at 1/8, 1/16, 1/32 resolution for each camera
- **Key:** These features are rich in semantic information (object appearance, class, texture)

### 3.3 Step 2: Radar Feature Extraction

The radar branch encodes sparse radar points into a BEV feature map:

- **Input features per point:** (x, y, z, vx_comp, vy_comp, rcs, timestamp)
- **Encoding:** Pillar-based (similar to PointPillars)
  - Divide BEV plane into pillars (e.g., 0.4m x 0.4m)
  - Mini-PointNet encodes points within each pillar
  - Scatter to 2D pseudo-image
- **2D backbone:** Light CNN processes the BEV pseudo-image
- **Output:** Dense BEV feature map with radar spatial + velocity information

### 3.4 Step 3: Spatial Alignment via Projection

To fuse radar and camera features, we need spatial correspondence:

```
Radar point p_radar = (x, y, z) in 3D world coordinates

Project to image:
  p_image = K * [R | t] * [x, y, z, 1]^T

Where:
  K = camera intrinsic matrix (3x3): focal length, principal point
  [R|t] = radar-to-camera extrinsic (3x4): rotation + translation

Result: (u, v) pixel coordinates where radar point appears in the image
```

This projection tells the transformer: "for radar point at (45m, 2m, 0m), look at pixel (823, 412) in camera 2 for semantic information."

### 3.5 Step 4: Spatio-Contextual Fusion Transformer (SCFT)

The core innovation — cross-attention between radar and camera features:

```
Cross-Attention Mechanism:

  Query (Q):  Radar BEV features at each spatial location
              "I'm a radar detection at (45m, 2m). What am I?"

  Key (K):    Image features at projected locations (+ neighbors)
              "Here are the visual features near where this radar point projects"

  Value (V):  Image features (same as Key)
              "Here is the semantic content to transfer to the radar feature"

  Attention(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V

  Position encoding includes:
    - 3D position of the radar point
    - 2D projected image coordinates
    - Depth-aware encoding (closer objects get higher resolution attention)
```

**Why cross-attention (not simple concatenation)?**
- Attention learns WHICH image features are relevant to each radar point
- Handles spatial misalignment (radar projection isn't pixel-perfect due to ~2° beam width)
- Deformable attention: learns offsets to sample most informative image locations
- Naturally handles multi-scale (attends to different FPN levels based on object size)

### 3.6 Step 5: Contextual Enhancement

Beyond point-to-pixel correspondence, CRAFT enhances fusion with:

1. **Multi-scale feature aggregation:** For each radar point, gather features from multiple FPN levels around the projection
2. **Deformable sampling:** Learn offset positions to attend to (not just the exact projection)
3. **Gating mechanism:** Learned gates weight reliability of each modality
4. **Temporal context:** Optional incorporation of previous frame features

### 3.7 Step 6: Detection Head

The fused BEV features feed into a standard detection head:
- **Heatmap:** Class-specific center prediction (CenterPoint-style)
- **Regression:** Box size (w, l, h), height (z), rotation (sin, cos), velocity (vx, vy)
- **Loss:** Gaussian focal loss for heatmap + L1 for regression

---

## 4. How Radar Provides WHERE, Camera Provides WHAT

### 4.1 Intuitive Example

```
Scenario: Highway driving, 3 objects ahead

Radar detects:             Camera sees:
                           ┌──────────────────────────────────┐
  . (50m, 0m, v=0)        │      [truck]     [car]           │
  . (35m, 3m, v=12m/s)    │   Truck is large, blue           │
  . (80m, -1m, v=8m/s)    │   Car is small, red, close       │
                           │   Something far away...          │
                           └──────────────────────────────────┘

Radar knows:               Camera knows:
- "Object at 50m, static"  - "Large blue vehicle"
- "Object at 35m, fast"    - "Small red vehicle"
- "Object at 80m, medium"  - "Hard to see, maybe a car?"

FUSED understanding:
- "Parked blue truck at 50m"
- "Red car at 35m doing 43 km/h (passing us)"
- "Vehicle at 80m doing 29 km/h (we're catching up)"
```

### 4.2 Where Each Modality Excels

**Radar excels at:**
- Long-range detection (80-200m where camera depth estimation fails)
- Velocity estimation (direct measurement, not indirect)
- Detecting objects in rain/fog/darkness
- Distinguishing moving from stationary objects

**Camera excels at:**
- Fine-grained classification (sedan vs SUV vs truck)
- Lateral position refinement (narrow angular accuracy)
- Detecting non-metallic objects (pedestrians, cyclists — low radar return)
- Scene context (lanes, signs, traffic lights)

**The fusion helps most when:**
- Objects are far away (camera depth uncertain, radar range accurate)
- Objects are fast-moving (radar Doppler directly measures speed)
- Weather is poor (camera degraded, radar reliable)
- Classification is ambiguous (radar can't tell car from van, camera can)

---

## 5. Challenges in Camera-Radar Fusion

### 5.1 Spatial Misalignment

Radar has ~2° angular resolution vs camera's ~0.03°. A single radar "detection" could correspond to a 2-meter-wide area at 60m range:

```
                    Radar beam width at 60m
                    ├───── ~2m ─────┤
  ┌─────────────────────────────────────────────┐
  │              ┌────────┐                     │ Camera image
  │              │ actual │                     │
  │              │ object │                     │
  │              └────────┘                     │
  │         ↑                                   │
  │    Radar point projects                     │
  │    HERE (but could be                       │
  │    anywhere in ~30px range)                 │
  └─────────────────────────────────────────────┘
```

**CRAFT's solution:** Deformable attention with learned offsets — the network learns to look in a region around the projection rather than at a single pixel.

### 5.2 Temporal Misalignment

Camera and radar don't trigger at exactly the same timestamp:
- Camera: ~30 Hz capture
- Radar: ~13 Hz measurement cycles
- Up to ~30ms offset between corresponding measurements

For a car at 30 m/s, 30ms offset = ~1m position error. CRAFT addresses this via ego-motion compensation and temporal encoding.

### 5.3 Elevation Ambiguity

Most automotive radars have no height measurement:

```
Radar detects something at range=40m, azimuth=5°
But at what height?

         │ Could be overhead sign
         │ Could be vehicle
         │ Could be curb reflection
    ─────┼──────────────────── Ground plane
         │
    Radar cannot tell!
```

A radar point projects to a VERTICAL LINE in the image, not a single point. CRAFT must search along this epipolar line for the correct image association.

### 5.4 Ghost Targets (Multipath)

Radar signals bounce off multiple surfaces:

```
    Radar ─────── Signal ──────→ Road surface
                                      │
                                      ▼ bounces up
                                 ┌─────────┐
                                 │  Car     │
                                 └─────────┘
                                      │
                                      ▼ bounces back via road
                               Ghost appears BELOW road surface

    Result: radar reports a detection underground!
```

CRAFT handles this by cross-referencing with camera — if camera sees nothing at the projected location, the radar point is likely a ghost.

### 5.5 Extreme Sparsity Mismatch

- Camera: 1920 x 1080 = ~2 million pixels per image, 6 cameras = ~12M values
- Radar: 30-100 points per frame after CFAR detection

This is a 100,000:1 density mismatch. CRAFT's attention mechanism naturally handles this — each sparse radar query can attend to relevant dense image regions.

---

## 6. Comparison with Related Methods

### 6.1 Camera-Radar Fusion Methods

| Method | Year | Fusion Strategy | Temporal | nuScenes NDS | nuScenes mAP |
|--------|------|----------------|----------|--------------|--------------|
| CenterFusion | 2021 | Frustum association | No | ~45.3 | ~32.6 |
| RCBEV | 2022 | BEV concatenation | No | ~49.7 | ~37.1 |
| CRAFT | 2023 | Spatio-contextual attention | No | ~56.1 | ~41.2 |
| CRN | 2023 | Cross-modal attention | Yes | ~54.2 | ~38.5 |

### 6.2 Cross-Modal Comparison (Same Benchmark: nuScenes val)

| Method | Modalities | NDS | mAP | Notes |
|--------|-----------|-----|-----|-------|
| BEVDet | Camera only | 48.8 | 34.9 | BEV from depth estimation |
| CenterPoint | LiDAR only | 67.3 | 60.3 | Gold standard |
| RadarPillarNet | Radar only | ~35 | ~22 | Very challenging |
| CRAFT | Camera + Radar | ~56 | ~41 | Best of low-cost sensors |
| TransFusion | LiDAR + Camera | 71.3 | 67.5 | Expensive but best |
| BEVFusion | LiDAR + Camera | 72.9 | 70.2 | State-of-the-art |

### 6.3 Key Takeaway

Camera+Radar fusion (~56 NDS) significantly outperforms either modality alone (camera ~49, radar ~35) and approaches the lower end of LiDAR-only performance (~65-67 NDS) at a fraction of the cost. However, there remains a substantial gap to LiDAR+Camera methods (~71-73 NDS).

---

## 7. CRAFT's Key Contributions

### Contribution 1: Spatio-Contextual Fusion Transformer (SCFT)

The novel fusion architecture bridges the modality gap through:
- Learnable spatial alignment beyond simple geometric projection
- Contextual attention that selectively fuses relevant features
- Principled handling of sparsity mismatch

### Contribution 2: Dual-Branch Feature Extraction

Separate modality-specific branches preserve unique sensor characteristics:
- Camera: established image architectures (ResNet + FPN)
- Radar: pillar-based encoding with sparse convolutions
- Each optimized independently before fusion

### Contribution 3: Proposal-Level Fusion Strategy

Fusion at the object/proposal level rather than raw data:
- Generate initial proposals from each modality
- Cross-reference using SCFT
- More robust to sensor noise and calibration errors

### Contribution 4: Comprehensive Evaluation

Thorough ablation studies showing:
- Per-modality contribution to overall performance
- Effectiveness of attention vs simpler fusion
- Per-class analysis (radar helps most for large/fast objects)
- Robustness under sensor degradation

---

## 8. Practical Considerations for Implementation

### 8.1 Calibration Requirements

Camera-radar fusion is highly sensitive to extrinsic calibration:
- Projection error of even 1-2° causes radar points to project onto wrong objects
- Auto-calibration / online calibration refinement is important
- Temporal calibration (synchronization) is equally critical

### 8.2 Radar Preprocessing

Before feeding to the network:
1. Ego-motion compensate accumulated sweeps (5-10 sweeps typical)
2. Filter obvious ground returns (z < -1.5m after compensation)
3. Compute compensated velocities: vx_comp = vr - v_ego * cos(azimuth)
4. Normalize RCS values to reasonable range

### 8.3 Training Tips

- Pre-train camera backbone on image classification (ImageNet)
- Pre-train radar branch on radar-only detection
- Fine-tune jointly with lower learning rate for pre-trained components
- Data augmentation: random flip, rotation, scaling in BEV space
- Handle missing sensors gracefully (dropout one modality during training)

### 8.4 Common Failure Cases

- **Pedestrians/cyclists:** Very low radar return (small RCS), often only 0-1 radar points
- **Close objects:** Radar minimum range ~1-2m creates blind spot
- **Lateral precision:** Radar's poor angular resolution limits lateral accuracy
- **Occluded objects:** Neither sensor can detect fully occluded targets
- **Multi-path near guardrails:** Ghost targets confuse fusion

---

## 9. Future Directions

### 9.1 4D Imaging Radar

Next-generation radars with elevation resolution (4D: range, azimuth, elevation, Doppler):
- 10-100x more points per frame
- Height measurement eliminates elevation ambiguity
- Closer to LiDAR-like point clouds
- Will likely close the NDS gap significantly

### 9.2 Temporal Fusion

Incorporating temporal information across frames:
- Track-level fusion (associate detections over time)
- Recurrent BEV features (memory of past observations)
- Temporal accumulation improves both density and confidence

### 9.3 Self-Supervised Pre-training

- Contrastive learning between camera and radar representations
- Masked prediction: predict radar features from camera (and vice versa)
- Knowledge distillation from LiDAR teacher models

---

## 10. Significance and Impact

CRAFT represents an important step toward production-ready perception:

1. **Mass-market viability:** Uses sensors already in production vehicles ($500 vs $10K+)
2. **All-weather operation:** Maintains performance in rain, fog, darkness
3. **Velocity estimation:** Direct measurement critical for prediction/planning
4. **Principled fusion:** Establishes attention-based cross-modal fusion as the paradigm
5. **Practical deployment:** Camera+radar is the only sensor suite scalable to millions of vehicles

The camera-radar fusion field is rapidly advancing, with 4D imaging radar and improved transformers likely to close the remaining gap to LiDAR-based methods.

---

## References

1. Kim, Y., et al. "CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer." IROS 2023.
2. Nabati, R., Qi, H. "CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection." WACV 2021.
3. Liu, Z., et al. "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation." ICRA 2023.
4. Bai, X., et al. "TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers." CVPR 2022.
5. Caesar, H., et al. "nuScenes: A Multimodal Dataset for Autonomous Driving." CVPR 2020.
6. Lang, A.H., et al. "PointPillars: Fast Encoders for Object Detection from Point Clouds." CVPR 2019.
7. Yin, T., et al. "Center-based 3D Object Detection and Tracking." CVPR 2021.
