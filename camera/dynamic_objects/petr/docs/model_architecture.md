# Model Architecture - PETR / PETRv2 / StreamPETR

## Overview

This document provides a detailed architectural description of the PETR family of models. All three models share the same fundamental principle: encoding 3D spatial information into 2D image features via position embeddings, then using transformer decoders for detection. They differ primarily in how temporal information is incorporated.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PETR / StreamPETR Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Multi-View     Image Backbone    Feature Pyramid     3D Position          │
│  Images    ──── + FPN          ── Network         ──  Embedding    ─┐      │
│  (6 cams)       (ResNet-50)       (Multi-scale)       Generation    │      │
│                                                                      │      │
│                                                                      ▼      │
│  Object     ┌── Transformer ◄── Position-Aware Features ◄───────────┘      │
│  Queries ───┤   Decoder         (F_image + PE_3d)                           │
│  (900)      │   (6 layers)                                                  │
│             │                                                               │
│             └── Detection Head ── Predictions                               │
│                  (cls + reg)      [class, box, velocity]                     │
│                                                                             │
│  ┌─────────── StreamPETR Addition ──────────────┐                          │
│  │  Query Propagation: top-K queries from       │                          │
│  │  previous frame -> ego-motion compensate     │                          │
│  │  -> inject as propagated queries             │                          │
│  └──────────────────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Image Backbone + FPN

### Image Backbone (ResNet-50)

Extracts multi-scale features from each camera image independently.

```
Input: 6 images, each [3, 900, 1600] (RGB, H, W)

ResNet-50 Architecture:
  Conv1 (7x7, stride 2) + BN + ReLU + MaxPool (stride 2)
  │
  ├── Stage 1 (C2): [256, 225, 400]   ── (not used, too fine)
  ├── Stage 2 (C3): [512, 113, 200]   ── out_indices[0]
  ├── Stage 3 (C4): [1024, 57, 100]   ── out_indices[1]
  └── Stage 4 (C5): [2048, 29, 50]    ── out_indices[2]

Total parameters: ~23.5M (ResNet-50)
```

### Feature Pyramid Network (FPN)

Fuses multi-scale features into a unified representation.

```
FPN Architecture:

  C5 [2048, 29, 50] ─── 1x1 conv ──── P5 [256, 29, 50]
                                          │
  C4 [1024, 57, 100] ── 1x1 conv ── + ←─┘(upsample 2x)
                                     │
                                     P4 [256, 57, 100]
                                          │
  C3 [512, 113, 200] ── 1x1 conv ── + ←─┘(upsample 2x)
                                     │
                                     P3 [256, 113, 200]

Output per camera: 3 feature maps at different scales
  P3: [256, 113, 200]  (1/8 resolution)
  P4: [256, 57, 100]   (1/16 resolution)
  P5: [256, 29, 50]    (1/32 resolution)

Total for 6 cameras (flattened):
  6 * (113*200 + 57*100 + 29*50) = 6 * (22600 + 5700 + 1450) = 178,500 tokens
```

---

## Component 2: 3D Position Embedding Generation

This is PETR's core innovation. It transforms camera frustum coordinates into 3D world space and encodes them into position embeddings that are added to image features.

### Step 2.1: Frustum Point Generation

For each camera and each feature map level, generate a grid of 3D frustum points:

```
For a feature map of size [H_feat, W_feat]:

1. Generate 2D pixel grid:
   u = linspace(0, W_img-1, W_feat)  # Horizontal pixel coords
   v = linspace(0, H_img-1, H_feat)  # Vertical pixel coords

2. Generate depth bins:
   d = linspace(depth_start, depth_end, depth_num)  # 64 depth values
   d = [1.0, 1.96, 2.91, ..., 61.2]  # meters

3. Create 3D frustum grid:
   frustum_points[h, w, d] = (u[w], v[h], depth[d])
   Shape: [H_feat, W_feat, D, 3]  # (height, width, depth, xyz)

Example for P3 (113x200 feature map, 64 depth bins):
   frustum shape = [113, 200, 64, 3] = 1,446,400 points per camera
```

### Step 2.2: Camera-to-World Transformation

Transform frustum points from pixel coordinates to 3D world (ego) coordinates:

```
For each frustum point (u, v, d):

1. Unproject to camera 3D coordinates:
   [x_cam]       [u * d]
   [y_cam] = K^{-1} * [v * d]
   [z_cam]       [  d  ]

2. Transform to ego frame:
   [x_ego]              [x_cam]
   [y_ego] = R_cam2ego * [y_cam] + t_cam2ego
   [z_ego]              [z_cam]

Where:
   K = camera intrinsic matrix (3x3)
   R_cam2ego = camera-to-ego rotation (3x3)
   t_cam2ego = camera-to-ego translation (3x1)

Result: 3D points in ego vehicle coordinate system
   Shape per camera: [H_feat, W_feat, D, 3]
```

### Step 2.3: Coordinate Normalization

Normalize 3D coordinates to [-1, 1] range using the perception volume:

```
pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

x_norm = (x_ego - x_min) / (x_max - x_min)  # [0, 1]
y_norm = (y_ego - y_min) / (y_max - y_min)  # [0, 1]
z_norm = (z_ego - z_min) / (z_max - z_min)  # [0, 1]

# Some implementations use [-1, 1] range instead:
x_norm = 2 * x_norm - 1
```

### Step 2.4: MLP Encoding

Feed normalized 3D coordinates through a learned MLP:

```
Position MLP:
  Input: [x_norm, y_norm, z_norm]  ── shape [*, 3]
    │
    Linear(3, 1024) + ReLU
    │
    Linear(1024, 256)
    │
  Output: position_embedding  ── shape [*, 256]

This MLP is shared across all cameras and all feature levels.
Parameters: 3*1024 + 1024 + 1024*256 + 256 = ~266K parameters
```

### Step 2.5: Aggregation Over Depth

The frustum has a depth dimension that must be collapsed to match feature map shape:

```
frustum PE shape: [H_feat, W_feat, D, 256]

Aggregation strategy (weighted sum over depth):
  PE_aggregated[h, w] = sum_d(PE[h, w, d]) / D
  
  OR (learned depth weights):
  PE_aggregated[h, w] = sum_d(alpha[d] * PE[h, w, d])
  
Final shape: [H_feat, W_feat, 256]  # Same as feature map
```

### Step 2.6: Add to Image Features

```
F_position_aware = F_image + PE_3d

Where:
  F_image: [num_cams * H * W, 256]  (flattened multi-camera features)
  PE_3d:   [num_cams * H * W, 256]  (corresponding position embeddings)
  F_position_aware: [num_cams * H * W, 256]  (input to decoder cross-attention)
```

### Diagram: 3D Position Embedding Pipeline

```
Camera Images (6x [3, 900, 1600])
         │
         ▼
┌─────────────────────┐
│  Backbone + FPN     │ ──── Image Features: 6x [256, H, W]
└─────────────────────┘
         │
         │    ┌──────────────────────────────────────────────────────┐
         │    │  3D Position Embedding Generation                    │
         │    │                                                      │
         │    │  Camera Intrinsics K ──┐                            │
         │    │  Camera Extrinsics  ───┤                            │
         │    │  Depth Bins         ───┼── Frustum Points [H,W,D,3] │
         │    │                        │         │                   │
         │    │                        │    cam-to-ego transform     │
         │    │                        │         │                   │
         │    │                        │    3D World Points [H,W,D,3]│
         │    │                        │         │                   │
         │    │                        │    Normalize to [-1,1]      │
         │    │                        │         │                   │
         │    │                        │    MLP(3 → 1024 → 256)     │
         │    │                        │         │                   │
         │    │                        │    Depth Aggregation        │
         │    │                        │         │                   │
         │    │                        │    PE_3d [H, W, 256]       │
         │    └──────────────────────────────────┼───────────────────┘
         │                                       │
         ▼                                       ▼
    F_image [H, W, 256]        +         PE_3d [H, W, 256]
         │                                       │
         └──────────────── + ────────────────────┘
                           │
                           ▼
              F_position_aware [H, W, 256]
                    (per camera, then flatten across all cameras)
```

---

## Component 3: Transformer Decoder

### Standard Cross-Attention (NOT Deformable)

PETR uses **standard multi-head attention** in its decoder, unlike BEVFormer which uses deformable attention. This is possible because 3D PE already encodes spatial information, so the attention mechanism can learn to focus on the correct spatial locations without explicit guidance.

### Decoder Layer Architecture

```
Each decoder layer (repeated 6 times):

  Input: queries [900, 256], memory [178500, 256]
    │
    ├── Self-Attention ──── queries attend to other queries
    │   │                   (models inter-object relationships)
    │   LayerNorm + Dropout + Residual
    │
    ├── Cross-Attention ─── queries attend to position-aware features
    │   │                   (extracts relevant information from images)
    │   │                   Q: queries [900, 256]
    │   │                   K: F_position_aware [178500, 256]
    │   │                   V: F_image [178500, 256]
    │   LayerNorm + Dropout + Residual
    │
    └── Feed-Forward Net ── Non-linear transformation
        │                   Linear(256, 2048) + ReLU + Linear(2048, 256)
        LayerNorm + Dropout + Residual
    │
  Output: refined queries [900, 256]
```

### Attention Computation Details

```
Multi-Head Cross-Attention (8 heads, dim=256):
  head_dim = 256 / 8 = 32

  For each head i:
    Q_i = queries @ W_q_i    # [900, 32]
    K_i = memory @ W_k_i     # [178500, 32]
    V_i = memory @ W_v_i     # [178500, 32]
    
    attn_i = softmax(Q_i @ K_i^T / sqrt(32))  # [900, 178500]
    out_i = attn_i @ V_i     # [900, 32]
  
  output = Concat(out_1, ..., out_8) @ W_o  # [900, 256]

Note: The attention matrix is [900, 178500] which is large but manageable
with mixed precision training. This is why PETR is slightly slower than
methods using deformable attention (which samples only ~4 points).
```

### Object Queries

```
Query initialization:
  - Learnable query embeddings: [900, 256] (trained parameters)
  - Learnable reference points: [900, 3] (initial 3D positions, normalized)
  
  During cross-attention:
    Q = query_embedding + position_encoding(reference_point)
    K = F_position_aware  (image features + 3D PE)
    V = F_image           (raw image features)
```

---

## Component 4: Detection Head

### Prediction Branches

```
For each decoder layer output (auxiliary loss) and final output:

  Classification Branch:
    queries [900, 256]
      │
      Linear(256, 256) + ReLU
      │
      Linear(256, 256) + ReLU
      │
      Linear(256, 10)  ── class logits (10 classes)
      │
      Sigmoid  ── class probabilities [900, 10]

  Regression Branch:
    queries [900, 256]
      │
      Linear(256, 256) + ReLU
      │
      Linear(256, 256) + ReLU
      │
      Linear(256, 10)  ── box parameters [900, 10]
                           (cx, cy, cz, w, l, h, sin, cos, vx, vy)
```

### Reference Point Refinement

```
The regression output is a RESIDUAL added to the reference point:

  predicted_cx = reference_point_x + sigmoid(reg_output[0]) * range_x
  predicted_cy = reference_point_y + sigmoid(reg_output[1]) * range_y
  predicted_cz = reference_point_z + sigmoid(reg_output[2]) * range_z
  
  (dimensions and velocities are predicted directly, not as residuals)
```

---

## Component 5: StreamPETR Query Propagation

### Query Propagation Mechanism

StreamPETR's key addition is propagating queries across frames:

```
Frame t-1:                              Frame t:
┌────────────────────┐                  ┌────────────────────┐
│ Decoder Output     │                  │                    │
│ 900 queries        │                  │  Fresh Queries     │
│   │                │                  │  (644 random init) │
│   ├── Top-K (256)  │──── propagate ──►│        +           │
│   │   by score     │     ┌─────┐      │  Propagated (256)  │
│   │                │     │ Ego │      │  = 900 total       │
│   └── Discard rest │     │Motion│      │                    │
│                    │     │Comp. │      │   ┌── Decoder ──┐  │
└────────────────────┘     └──┬──┘      │   │  6 layers   │  │
                              │         │   └─────────────┘  │
                              │         │         │          │
                              └────────►│   Detection Head   │
                                        └────────────────────┘
```

### Detailed Propagation Steps

```python
# After processing frame t-1:
scores = classification_head(decoder_output)  # [900, 10]
max_scores = scores.max(dim=-1)  # [900]

# Step 1: Select top-K queries
topk_indices = max_scores.topk(256).indices
propagated_queries = decoder_output[topk_indices]  # [256, 256]
propagated_ref_points = reference_points[topk_indices]  # [256, 3]

# Step 2: Ego-motion compensation
# T_curr_from_prev: 4x4 transformation matrix
propagated_ref_points_homo = cat([propagated_ref_points, ones], dim=-1)  # [256, 4]
propagated_ref_points = (T_curr_from_prev @ propagated_ref_points_homo.T).T[:, :3]  # [256, 3]

# Step 3: Velocity-based position prediction
# Extrapolate position using predicted velocity and time delta
dt = timestamp_curr - timestamp_prev  # seconds
predicted_velocity = regression_head(propagated_queries)[:, 8:10]  # [256, 2] (vx, vy)
propagated_ref_points[:, 0] += predicted_velocity[:, 0] * dt  # x += vx * dt
propagated_ref_points[:, 1] += predicted_velocity[:, 1] * dt  # y += vy * dt

# Step 4: Compose with fresh queries
fresh_queries = learnable_query_embeddings[:644]  # [644, 256]
fresh_ref_points = learnable_reference_points[:644]  # [644, 3]

all_queries = cat([propagated_queries, fresh_queries], dim=0)  # [900, 256]
all_ref_points = cat([propagated_ref_points, fresh_ref_points], dim=0)  # [900, 3]
```

### Memory Buffer

```
StreamPETR maintains a circular buffer of historical queries:

Memory Buffer (max length = 512):
┌──────────────────────────────────────────────────────────────┐
│ Frame t-3    Frame t-2    Frame t-1                          │
│ [256 queries] [256 queries] [256 queries] ... [padding]       │
│                                                              │
│ Total stored: min(num_frames * 256, 512) query vectors       │
│ Selection for propagation: top-256 from most recent frame    │
└──────────────────────────────────────────────────────────────┘

Benefits:
- Constant memory regardless of sequence length
- Objects tracked even through brief occlusions
- Natural handling of object entry/exit (new objects get fresh queries)
```

---

## Component 6: Motion-Aware Layer Norm (StreamPETR)

### Concept

Standard LayerNorm has fixed scale/shift parameters. Motion-aware LayerNorm makes these parameters dynamic, conditioned on the ego-motion between frames:

```
Standard LayerNorm:
  output = gamma * (x - mean) / std + beta
  (gamma, beta are learned but fixed at inference)

Motion-Aware LayerNorm:
  motion_embed = MLP(ego_motion_matrix.flatten())  # [256]
  gamma_dynamic = gamma * (1 + scale_MLP(motion_embed))
  beta_dynamic = beta + shift_MLP(motion_embed)
  output = gamma_dynamic * (x - mean) / std + beta_dynamic
```

### Architecture

```
Ego-Motion Matrix (4x4) → Flatten (16-d)
    │
    Linear(16, 256) + ReLU
    │
    Linear(256, 256)
    │
    motion_embed [256]
    │
    ├── Scale MLP: Linear(256, 256) ── scale_factor [256]
    │
    └── Shift MLP: Linear(256, 256) ── shift_factor [256]
    
Applied in every transformer layer's LayerNorm:
  gamma_modulated = gamma * (1 + scale_factor)
  beta_modulated = beta + shift_factor
```

### Why This Works

- The ego vehicle moves between frames, changing the reference frame
- Without motion awareness, the transformer must implicitly figure out the ego-motion from the position embeddings (which change each frame)
- Motion-aware LayerNorm explicitly tells the transformer "the world shifted by this much" so it can focus on learning object-level dynamics

---

## Model Variants Summary

### PETR (Base)

```
Components: Backbone + FPN + 3D PE + Transformer Decoder + Detection Head
Temporal: None (single-frame)
Queries: 900 learnable, randomly initialized
Memory: ~6 GB (single-frame)
Speed: ~10 FPS
```

### PETRv2

```
Components: Backbone + FPN + 3D PE + 2D PE + Temporal Alignment + Decoder + Head
Temporal: Feature-level (concatenate previous frame's position-aware features)
Queries: 900 learnable, randomly initialized
Memory: ~14 GB (due to storing previous frame features)
Speed: ~8 FPS
```

### StreamPETR

```
Components: Backbone + FPN + 3D PE + Query Propagation + Motion-Aware LN + Decoder + Head
Temporal: Object-level (propagate top-K queries across frames)
Queries: 256 propagated + 644 fresh = 900 total
Memory: ~8 GB (only propagate 256 query vectors)
Speed: ~30 FPS
```

---

## Parameter Count Breakdown

| Component | Parameters | Notes |
|-----------|-----------|-------|
| ResNet-50 backbone | 23.5M | Pretrained, LR multiplier 0.1x |
| FPN | 2.1M | 3 levels, 256-d output |
| 3D PE MLP | 0.27M | Small but critical |
| Transformer decoder (6 layers) | 9.4M | Self-attn + cross-attn + FFN |
| Detection head (cls + reg) | 0.8M | 2-layer FC + output |
| Query embeddings | 0.23M | 900 * 256 |
| **Total (PETR)** | **~36M** | |
| + Motion-aware LN (StreamPETR) | +0.5M | Per-layer modulation |
| + Query propagation (StreamPETR) | +0.1M | Selection + transform |
| **Total (StreamPETR)** | **~37M** | |

---

## Inference Pipeline

### PETR Single-Frame Inference

```
1. Load 6 camera images                          [~2ms]
2. Backbone + FPN forward pass                    [~15ms]
3. Generate 3D Position Embeddings                [~5ms]
   - Frustum generation (can be precomputed)
   - Camera-to-ego transform
   - MLP encoding
4. Flatten features across cameras                [~1ms]
5. Transformer decoder (6 layers)                 [~60ms]
   - Self-attention: O(900^2 * 256)
   - Cross-attention: O(900 * 178500 * 256)  ← bottleneck
6. Detection head prediction                      [~2ms]
7. Post-processing (NMS, thresholding)           [~5ms]

Total: ~90ms per frame (~10 FPS)
```

### StreamPETR Streaming Inference

```
1. Load 6 camera images                          [~2ms]
2. Backbone + FPN forward pass                    [~15ms]
3. Generate 3D Position Embeddings                [~5ms]
4. Retrieve propagated queries from buffer        [~0.1ms]
5. Apply ego-motion compensation to queries       [~0.1ms]
6. Compose query set (256 prop + 644 fresh)      [~0.1ms]
7. Transformer decoder (6 layers)                 [~8ms]
   - Same architecture but optimized:
   - Memory-efficient attention implementation
   - FP16 inference
8. Detection head prediction                      [~2ms]
9. Select top-K queries for next frame           [~0.1ms]
10. Post-processing                              [~2ms]

Total: ~35ms per frame (~30 FPS)

Note: StreamPETR is faster partly because optimized implementations
can batch the attention more efficiently when queries are primed
with good initial positions (propagated queries).
```
