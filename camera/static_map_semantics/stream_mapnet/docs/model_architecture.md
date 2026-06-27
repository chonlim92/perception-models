# StreamMapNet: Model Architecture

## Overview and Motivation

StreamMapNet is an end-to-end neural network that takes **surround-view camera images** as input and produces **vectorized HD map elements** as output. The fundamental challenge it solves is:

> "Given 6 camera images from a car at time t, predict the layout of lanes, road boundaries, and crosswalks in the surrounding 60m x 30m area -- and do this consistently across time."

This requires solving several sub-problems:
1. **Multi-view fusion**: Combine information from 6 cameras with different viewpoints
2. **2D-to-3D lifting**: Go from 2D image pixels to a 3D/BEV (Bird's Eye View) representation
3. **Temporal aggregation**: Accumulate evidence across time as the car moves
4. **Structured prediction**: Output clean, ordered polylines (not pixels)

---

## Full Pipeline Diagram

```
 ===============================================================================
                         StreamMapNet Full Pipeline
 ===============================================================================

 INPUT: 6 surround-view cameras at time t
 +-------+  +-------+  +-------+  +-------+  +-------+  +-------+
 | FRONT |  |FRONT_L|  |FRONT_R|  | BACK  |  |BACK_L |  |BACK_R |
 +---+---+  +---+---+  +---+---+  +---+---+  +---+---+  +---+---+
     |           |           |          |           |          |
     +-----+-----+-----+----+-----+----+-----+----+-----+----+
           |                       |                      |
           v                       v                      v
 STAGE 1:  [=== ResNet-50 Backbone + FPN Neck (shared weights) ===]
           Input: (B*6, 3, 256, 704)
           Output: (B*6, 256, 32, 88) multi-scale features
                              |
                              v
 STAGE 2:  [======= Lift-Splat-Shoot BEV Transform =======]
           (a) Depth prediction: (B*6, 256, 32, 88) -> (B*6, 59, 32, 88)
           (b) Outer product "Lift": -> (B*6, 256, 59, 32, 88) frustum
           (c) Voxel pooling "Splat": -> (B, 256, 200, 100) raw BEV
           (d) BEV encoder: -> (B, 256, 200, 100) refined BEV
                              |
                              v
 STAGE 3:  [========== Temporal Fusion Module ==========]
           (a) Retrieve hidden state H_{t-1}
           (b) Warp H_{t-1} to current frame using ego-motion
           (c) Cross-attention: BEV_t attends to Warped(H_{t-1})
           (d) Gated fusion -> H_t
           Input: (B, 256, 200, 100) current + (B, 256, 200, 100) warped history
           Output: (B, 256, 200, 100) fused BEV features
                              |
                              +--------> Store H_t for next frame
                              |
                              v
 STAGE 4:  [====== Map Element Transformer Decoder ======]
           (a) Learnable queries: (B, 150, 256) -- one per potential element
           (b) Self-attention among queries (avoid duplicates)
           (c) Cross-attention to BEV features (find map elements)
           (d) 6 decoder layers with iterative refinement
           Output: (B, 150, 256) decoded query embeddings
                              |
                              v
 STAGE 5:  [========== Prediction Heads ==========]
           Classification: (B, 150, 256) -> (B, 150, 4)  [3 classes + background]
           Point Regression: (B, 150, 256) -> (B, 150, 20, 2) [20 points x (x,y)]
                              |
                              v
 OUTPUT:   Set of vectorized map elements
           Each element = class_label + 20 ordered (x, y) points in [0, 1]
           Denormalized: points map to [-30m, +30m] x [-15m, +15m]

 ===============================================================================
```

---

## Stage 1: Image Backbone (ResNet-50 + FPN)

### Why We Need Multi-Scale Features

Camera images contain map elements at various scales:
- A nearby lane divider occupies many pixels (large scale)
- A distant pedestrian crossing is just a few pixels (small scale)
- Road boundaries extend across the entire image

Multi-scale features let the model "see" at different resolutions simultaneously.

### ResNet-50 Backbone

ResNet-50 is a convolutional network pretrained on ImageNet. It processes each camera image independently (shared weights across all 6 cameras):

```
Input: (B*6, 3, 256, 704) -- all 6 cameras flattened into batch dimension

Stem (conv 7x7, stride 2, maxpool):  (B*6, 64,  64, 176)
Layer 1 (C2, 3 bottleneck blocks):   (B*6, 256, 64, 176)  stride 4
Layer 2 (C3, 4 bottleneck blocks):   (B*6, 512, 32, 88)   stride 8
Layer 3 (C4, 6 bottleneck blocks):   (B*6, 1024, 16, 44)  stride 16
Layer 4 (C5, 3 bottleneck blocks):   (B*6, 2048, 8, 22)   stride 32
```

We extract features from layers C3, C4, C5 (out_indices=[1, 2, 3]).

**Frozen stages**: Layer 1 is frozen (no gradient updates) because early features (edges, textures) are well-captured by ImageNet pretraining. This saves memory and prevents overfitting.

### Feature Pyramid Network (FPN)

FPN combines multi-scale features through a top-down pathway with lateral connections:

```
                 Top-Down Pathway
                 ================

C5 (2048, 8, 22) ---[1x1 conv]---> P5 (256, 8, 22)
                                        |
                                    [upsample 2x]
                                        |
C4 (1024, 16, 44) --[1x1 conv]--> + --> P4 (256, 16, 44)  <-- [3x3 conv]
                                        |
                                    [upsample 2x]
                                        |
C3 (512, 32, 88) ---[1x1 conv]--> + --> P3 (256, 32, 88)  <-- [3x3 conv]

Output: [P3, P4, P5] each with 256 channels
Primary level for BEV: P3 (256, 32, 88)
```

**Why FPN?** Without it, high-level features (C5) have good semantics but poor spatial resolution. FPN propagates high-level semantics back to high-resolution levels through the top-down pathway.

---

## Stage 2: BEV Transform (Lift-Splat-Shoot)

### The Fundamental Challenge

Cameras produce 2D images, but the map lives in 3D space (or rather, on the 2D ground plane in BEV). The core question:

> "For each pixel in the image, WHERE in 3D space does it correspond to?"

This is ambiguous because a single pixel could correspond to any point along its ray. LSS resolves this by predicting a probability distribution over depth.

### Step 2a: Depth Distribution Prediction

For each pixel in the feature map, predict WHERE along its camera ray the content is located:

```
Input:  (B*6, 256, 32, 88) -- image features (one level from FPN)
        
DepthNet: Conv2d(256, 256, 3x3) -> BN -> ReLU -> Conv2d(256, 59, 1x1) -> Softmax

Output: (B*6, 59, 32, 88) -- probability over 59 depth bins
```

Each of the 59 bins represents a 1-meter interval from 1m to 60m.

**Why categorical (softmax) instead of regressing a single depth?**

A single pixel might be at the boundary of two surfaces (e.g., road at 10m and a building at 30m). A categorical distribution can express this multi-modal uncertainty. Regression would average to 20m -- which is wrong for both surfaces.

```
Example depth distribution for one pixel:

Depth (m):  1  2  3  4  5  6  7  8  9  10 11 12 ... 59
Prob:       .  .  .  .  .  .  .  .  .  ##  .  .  ...  .
                                        ^
                                   Road surface at ~10m

Another pixel (boundary):
Depth (m):  1  2  3  4  5  6  7  8  9  10 11 12 ... 30 31 ... 59
Prob:       .  .  .  .  .  .  .  .  .  #  .  .  ... #  .  ...  .
                                       ^              ^
                                   Road 10m      Building 30m
```

### Step 2b: The "Lift" -- Creating Frustum Features

The outer product of image features and depth probabilities creates a 3D feature volume:

```
context_features: (B*6, C, H, W) = (B*6, 256, 32, 88)
   |
   |  [Reduce channels: Conv2d(256, 64)]
   v
reduced_features: (B*6, 64, 32, 88)

depth_probs: (B*6, 59, 32, 88)

Outer product (broadcast multiply):
   reduced_features: (B*6, 64,  1, 32, 88)  -- add depth dim
   depth_probs:      (B*6,  1, 59, 32, 88)  -- add channel dim
                     -------------------------
   frustum_features: (B*6, 64, 59, 32, 88)  -- 3D feature volume!
```

**Intuition**: Each pixel now becomes a column of 59 points in 3D space. The depth probability tells us how much "weight" to assign to each depth. If the network is confident a pixel is at 10m, that depth slice gets high weight.

### Step 2c: The "Splat" -- Voxel Pooling to BEV

Now we project each frustum point into ego-vehicle 3D coordinates and accumulate into a BEV grid:

```
For each point (u, v, d) in the frustum:
  1. Unproject to camera coordinates:
     X_cam = (u - cx) * d / fx
     Y_cam = (v - cy) * d / fy
     Z_cam = d
     
  2. Transform to ego coordinates using extrinsics (cam-to-ego):
     [X_ego]         [X_cam]
     [Y_ego] = R *   [Y_cam]  + T
     [Z_ego]         [Z_cam]
     
  3. Assign to BEV grid cell:
     grid_x = floor((X_ego - x_min) / resolution)
     grid_y = floor((Y_ego - y_min) / resolution)
     
  4. Add weighted features to that grid cell
```

The BEV grid parameters:
```
X range: [-30.0m, +30.0m]  (60m forward/backward)
Y range: [-15.0m, +15.0m]  (30m left/right)
Resolution: 0.3m per cell
Grid size: 200 cells (X) x 100 cells (Y)
```

**Pooling**: Multiple points from multiple cameras may land in the same BEV cell. We sum their features (scatter_add). This naturally handles overlap between cameras.

```
Output after splatting: (B, 64, 200, 100) -- raw BEV feature map
```

### Step 2d: BEV Encoder

A small CNN refines the raw BEV features:

```
Input:  (B, 64, 200, 100)
Conv3x3 + BN + ReLU: (B, 256, 200, 100)
Conv3x3 + BN + ReLU: (B, 256, 200, 100)
Output: (B, 256, 200, 100) -- final BEV features
```

This smooths out noise from the discrete voxel pooling and increases the channel dimension for richer representations.

### Complete LSS Tensor Flow

| Step | Input Shape | Output Shape | Operation |
|------|------------|--------------|-----------|
| FPN features | - | (B*6, 256, 32, 88) | From backbone |
| Depth prediction | (B*6, 256, 32, 88) | (B*6, 59, 32, 88) | DepthNet + softmax |
| Channel reduction | (B*6, 256, 32, 88) | (B*6, 64, 32, 88) | Conv2d 1x1 |
| Outer product (Lift) | 64-ch + 59-depth | (B*6, 64, 59, 32, 88) | Broadcast multiply |
| Splat to BEV | (B*6, 64, 59, 32, 88) | (B, 64, 200, 100) | Voxel pool (scatter_add) |
| BEV Encoder | (B, 64, 200, 100) | (B, 256, 200, 100) | 2x Conv3x3+BN+ReLU |

---

## Stage 3: Temporal Fusion Module

### Why Past Frames Help

Consider a car driving forward. At time t, the front cameras see what is ahead. At time t+1, the car has moved forward by ~0.5m. Now:
- What was previously at the edge of the front camera's view is now centered
- The rear cameras now see road that was behind us
- Occluded areas may become visible from the new angle

By accumulating observations across time, the model builds a MORE COMPLETE map:

```
Time t:   Car sees road ahead, but lane dividers are partially occluded by a truck
Time t+1: Car has moved; truck is now at a different angle; dividers partly visible
Time t+2: Car passed the truck; dividers fully visible from behind

Single-frame at t: Can't see dividers -> MISSES THEM
StreamMapNet at t: Has accumulated info from t-1, t-2, ... -> DETECTS THEM
```

### The Streaming Hidden State

StreamMapNet maintains a hidden state H_t that encodes all useful information from past frames. This is analogous to an RNN:

```
               Temporal Information Flow
               =========================

Frame t-2:  BEV_{t-2} -----> H_{t-2}
                                |
                          [ego-motion warp]
                                |
                                v
Frame t-1:  BEV_{t-1} ---[fuse]--> H_{t-1}
                                     |
                               [ego-motion warp]
                                     |
                                     v
Frame t:    BEV_t --------[fuse]-------> H_t -----> Decoder -> Map Output
                                          |
                                    [stored for t+1]
```

**Key insight**: We never re-process past images. We only carry forward ONE hidden state tensor, regardless of how many frames have passed. This gives O(1) cost per frame.

### Step 3a: Ego-Motion Warping

When the car moves from time t-1 to time t, the BEV features from t-1 are now in the wrong coordinate frame. We must spatially align them.

**The math**:

Given ego-motion matrix T_{prev_to_curr} (4x4 rigid body transformation):

```
T = [R  t]     R = 3x3 rotation
    [0  1]     t = 3x1 translation

For a BEV grid point p_curr = (x, y, 0, 1) in the current frame,
we want to find where it was in the previous frame:

p_prev = T_curr_to_prev @ p_curr = T_prev_to_curr^{-1} @ p_curr
```

**Implementation**:

```python
# 1. Create grid of BEV coordinates in current frame
xs = linspace(-30, +30, 200)  # 200 cells, x-axis
ys = linspace(-15, +15, 100)  # 100 cells, y-axis
grid = meshgrid(ys, xs)       # (100, 200, 2) in meters

# 2. Apply inverse ego-motion to find corresponding previous-frame locations
grid_homo = [grid_x, grid_y, 0, 1]          # homogeneous coords
grid_prev = T_inv @ grid_homo               # where each cell WAS

# 3. Normalize to [-1, 1] for grid_sample
norm_x = (grid_prev_x - x_min) / (x_max - x_min) * 2 - 1
norm_y = (grid_prev_y - y_min) / (y_max - y_min) * 2 - 1

# 4. Sample previous features at those locations
warped = F.grid_sample(H_prev, grid, mode='bilinear', padding_mode='zeros')
```

**Zero padding**: Areas that are "new" (the car moved, revealing previously unseen road) get zero features -- the model has no information there yet.

```
Example: Car moves 1m forward

  Previous BEV:              Current BEV after warping:
  +------------------+       +------------------+
  |  History info    |       | zeros (new area) |  <- car moved into new area
  |  about road     |       +------------------+
  |  behind us      |       |  History info    |
  |                  |       |  about road     |
  +------------------+       |  (shifted back) |
                             +------------------+
```

### Step 3b: Cross-Attention Fusion

After warping, we fuse current observations with warped history:

```python
# Flatten spatial dimensions for attention
current_flat = current_bev.flatten(2).permute(0, 2, 1)  # (B, 20000, 256)
warped_flat  = warped_prev.flatten(2).permute(0, 2, 1)  # (B, 20000, 256)

# Cross-attention: "What useful information does the past provide?"
# Query = current frame (what do I need?)
# Key/Value = warped past (what can the past offer?)
Q = query_proj(current_flat)   # (B, 20000, 256)
K = key_proj(warped_flat)      # (B, 20000, 256)
V = value_proj(warped_flat)    # (B, 20000, 256)

attention_weights = softmax(Q @ K^T / sqrt(d))  # (B, 20000, 20000)
attn_output = attention_weights @ V              # (B, 20000, 256)

# Residual + FFN
fused = LayerNorm(current_flat + attn_output)
fused = LayerNorm(fused + FFN(fused))
```

### Step 3c: Gated Fusion

A learnable gate controls how much temporal information to incorporate at each spatial location:

```python
# Gate: per-location blend between current and fused
gate_input = concat([current_flat, fused], dim=-1)  # (B, 20000, 512)
gate_weight = sigmoid(Linear(gate_input))            # (B, 20000, 256) in [0, 1]

# Blend: gate=1 means use fused (temporal), gate=0 means use current only
output = gate_weight * fused + (1 - gate_weight) * current_flat
```

**Why gating?** Some areas benefit from temporal info (persistent road structure), while others should rely only on current observation (moving objects, areas with high ego-motion blur). The gate learns this automatically.

### Temporal Tensor Dimensions

| Tensor | Shape | Description |
|--------|-------|-------------|
| current_bev | (B, 256, 200, 100) | Fresh BEV from current cameras |
| H_prev | (B, 256, 200, 100) | Stored hidden state from previous frame |
| ego_motion | (B, 4, 4) | prev-to-current transformation |
| T_inv | (B, 4, 4) | current-to-previous (for warping) |
| warp_grid | (B, 100, 200, 2) | Sampling coordinates |
| warped_prev | (B, 256, 200, 100) | Previous state in current coords |
| fused_bev (H_t) | (B, 256, 200, 100) | Output = new hidden state |

### Multi-Frame Extension

For temporal_window > 1, the module maintains a buffer of N previous states:

```
Buffer = [H_{t-3}, H_{t-2}, H_{t-1}]  (for window=3)

Each frame is independently warped to current coordinates:
  warped_3 = warp(H_{t-3}, T_{t-3 to t})
  warped_2 = warp(H_{t-2}, T_{t-2 to t})
  warped_1 = warp(H_{t-1}, T_{t-1 to t})

Sequential fusion: oldest to newest:
  fused = cross_attn(current_bev, warped_3)
  fused = cross_attn(fused, warped_2)
  fused = cross_attn(fused, warped_1)

Additional channel concatenation + 1x1 conv for richer aggregation.
```

---

## Stage 4: Map Element Decoder (Transformer)

### What Are Map Queries?

Map queries are learnable embeddings -- vectors that the network optimizes during training. Each query "learns" to detect one map element:

```
query_embeddings = nn.Embedding(150, 256)  # 150 learnable 256-dim vectors

# After training, different queries specialize:
# Query 0-49:  tend to detect lane dividers
# Query 50-99: tend to detect road boundaries  
# Query 100-149: tend to detect pedestrian crossings
```

**Analogy**: Think of each query as a "detector agent" that roams the BEV looking for its assigned type of map element. Through self-attention, agents communicate to avoid detecting the same element twice.

### Decoder Layer Structure

Each of the 6 decoder layers performs:

```
Input: queries (B, 150, 256) + BEV memory (B, 20000, 256)

(1) Self-Attention (queries talk to each other):
    "Is anyone else detecting the same lane divider I found?"
    queries = LayerNorm(queries + SelfAttn(queries, queries, queries))

(2) Cross-Attention (queries look at BEV features):
    "Where in the BEV are my map elements?"
    queries = LayerNorm(queries + CrossAttn(queries, BEV, BEV))

(3) Feed-Forward Network:
    queries = LayerNorm(queries + FFN(queries))

Output: queries (B, 150, 256) -- refined
```

### Cross-Attention to BEV Features

This is where queries actually "read" the map content:

```
                  Cross-Attention Mechanism
                  ========================

  Queries (150 elements)         BEV Feature Map (200 x 100)
  +---+---+---+...+---+         +---------------------------+
  | q1| q2| q3|   |q150|        |  BEV features contain     |
  +---+---+---+...+---+         |  spatial info about road   |
       |                         |  structure, lanes, etc.    |
       | Q = Linear(queries)     +---------------------------+
       |                              |
       |     K, V = Linear(BEV)       |
       |          |                   |
       +----------+----> Attention ---+
                         weights
                           |
                           v
                  Weighted sum of BEV features
                  (each query extracts relevant spatial info)
```

### Deformable Attention (Efficiency)

Full cross-attention to 200x100 = 20,000 BEV positions is expensive (O(N*M) = O(150 * 20000)). Deformable attention samples only a few key positions per query:

```
Instead of attending to ALL 20,000 positions:
  Query -> predict 4 sampling offsets per head
  Sample BEV features at only those 4 positions
  Weighted average of sampled values

Complexity: O(150 * 4 * 8 heads) = O(4800) instead of O(3,000,000)
```

### Iterative Reference Point Refinement

Each decoder layer also predicts a small correction to reference points:

```
Layer 0: Initial reference points (learned, e.g., uniform grid)
         queries -> predict delta_points -> update reference_points

Layer 1: Refined reference points
         queries -> predict delta_points -> further refine

...

Layer 5: Final highly-refined reference points = predicted polyline
```

### Decoder Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_queries | 150 | Max map elements to detect |
| num_decoder_layers | 6 | Iterative refinement depth |
| d_model | 256 | Query/key/value dimension |
| num_heads | 8 | Multi-head attention heads |
| ffn_dim | 512 | Feed-forward hidden dimension |
| dropout | 0.1 | Regularization |
| deformable_points | 4 | Sampling points per head |

---

## Stage 5: Prediction Heads

### Classification Head

Predicts what class each query represents (or "no object"):

```python
ClassHead = Sequential(
    Linear(256, 256), ReLU(), LayerNorm(256),
    Linear(256, 4)  # 3 map classes + 1 background
)

Input:  (B, 150, 256) query embeddings
Output: (B, 150, 4)   class logits
                       [lane_div, road_bound, ped_cross, no_object]
```

### Polyline Regression Head

Predicts 20 ordered (x, y) points defining the shape of each map element:

```python
PointHead = Sequential(
    Linear(256, 256), ReLU(),
    Linear(256, 20 * 2)  # 20 points x 2 coordinates
)

Input:  (B, 150, 256) query embeddings
Output: (B, 150, 40) -> reshape to (B, 150, 20, 2) -> sigmoid to [0, 1]
```

**Sigmoid activation**: Forces coordinates into [0, 1] range. During evaluation, these are denormalized:
```
x_meters = x_norm * (x_max - x_min) + x_min = x_norm * 60 - 30
y_meters = y_norm * (y_max - y_min) + y_min = y_norm * 30 - 15
```

### Deep Supervision (Auxiliary Outputs)

Predictions are generated at EVERY decoder layer (not just the final one):

```
Layer 0 -> cls_logits_0, pred_points_0  (coarse)
Layer 1 -> cls_logits_1, pred_points_1
Layer 2 -> cls_logits_2, pred_points_2
Layer 3 -> cls_logits_3, pred_points_3
Layer 4 -> cls_logits_4, pred_points_4
Layer 5 -> cls_logits_5, pred_points_5  (final, best quality)
```

All layers contribute to the loss during training. This helps gradient flow to early layers.

---

## Output Format

### Per-Frame Predictions

```python
output = {
    'pred_logits': tensor(B, 150, 4),     # Class logits (softmax for probs)
    'pred_points': tensor(B, 150, 20, 2), # Normalized [0,1] polyline points
    'aux_outputs': [                       # Intermediate layer outputs
        {'pred_logits': ..., 'pred_points': ...},  # Layer 0
        {'pred_logits': ..., 'pred_points': ...},  # Layer 1
        # ... layers 2-4
    ]
}
```

### Post-Processing for Inference

```python
# 1. Get class probabilities
probs = softmax(pred_logits, dim=-1)           # (B, 150, 4)

# 2. Get predicted class and confidence (exclude background)
scores, labels = probs[:, :, :3].max(dim=-1)   # (B, 150)

# 3. Filter by confidence threshold
keep = scores > 0.3

# 4. Denormalize points to meters
points_meters = pred_points * [60, 30] + [-30, -15]  # (B, 150, 20, 2)
```

### What the Output Represents

```
Example output for one frame:

Element 0: class=lane_divider, score=0.92, points=[(−5.2, −2.1), (−4.8, −1.5), ..., (2.3, 3.4)]
Element 1: class=road_boundary, score=0.87, points=[(−15.0, −12.3), ..., (−15.0, 12.1)]
Element 2: class=ped_crossing, score=0.76, points=[(8.1, −3.2), ..., (8.1, 3.2)]
...
Element 11: class=lane_divider, score=0.45, points=[...]
Element 12: class=background, score=0.08 -> DISCARDED (below threshold)
...
Element 149: class=background, score=0.02 -> DISCARDED
```

Typically, a scene has 10-30 actual map elements; the remaining queries predict "background."

---

## Memory Footprint and Compute Budget

### Per-Frame Inference (B=1, ResNet-50, 200x100 BEV)

| Component | FLOPs | GPU Memory | Latency (A100) |
|-----------|-------|------------|----------------|
| Backbone (6 images) | 24.6G | 1.2 GB | 12 ms |
| LSS depth + splat | 8.2G | 2.1 GB | 8 ms |
| BEV encoder | 4.8G | 0.6 GB | 4 ms |
| Temporal fusion | 0.8G | 0.2 GB | 2 ms |
| Map decoder (6 layers) | 3.2G | 0.3 GB | 6 ms |
| Prediction heads | 0.1G | <0.1 GB | <1 ms |
| **Total** | **41.7G** | **4.4 GB** | **~33 ms** |

**Inference speed**: ~30 FPS on A100, ~15 FPS on RTX 3090

### Why Streaming is Memory-Efficient

| Approach | Memory for History | Compute for History |
|----------|-------------------|---------------------|
| No temporal | 0 | 0 |
| Re-encode past frames (N=4) | 4 x image features + 4 x BEV | 4x backbone + 4x LSS |
| BEVFormer (buffer 4 BEVs) | 4 x (256, 200, 100) = 80 MB | Temporal self-attention |
| **StreamMapNet (1 hidden state)** | **1 x (256, 200, 100) = 20 MB** | **1x warp + 1x cross-attn** |

StreamMapNet stores only ONE tensor of size (256, 200, 100) = 5.12M values = 20 MB in FP32. This is constant regardless of how many frames have been processed.

---

## Model Variants

### Backbone Options

| Backbone | Params | FLOPs (per image) | mAP (nuScenes) |
|----------|--------|-------------------|-----------------|
| ResNet-18 | 11.7M | 1.8G | 48.2 |
| ResNet-50 | 25.6M | 4.1G | 54.1 |
| ResNet-101 | 44.5M | 7.8G | 55.8 |
| Swin-Tiny | 28.3M | 4.5G | 56.2 |

### BEV Resolution Options

| Resolution | Grid Size | BEV Memory | mAP |
|-----------|-----------|------------|-----|
| 0.15 m/px | 400 x 200 | ~4.2 GB | 55.3 |
| 0.30 m/px | 200 x 100 | ~1.1 GB | 54.1 |
| 0.60 m/px | 100 x 50 | ~0.3 GB | 50.8 |

### Query Count Effect

| N_queries | mAP | Recall | Notes |
|-----------|-----|--------|-------|
| 30 | 52.1 | 78.3% | May miss elements in complex intersections |
| 50 | 53.5 | 83.2% | Good for simple roads |
| 100 | 54.0 | 87.5% | Balanced |
| 150 | 54.1 | 89.1% | Default, handles complex scenes |
| 200 | 54.2 | 89.8% | Diminishing returns |

---

## Key Design Decisions Summarized

| Decision | Choice | Rationale |
|----------|--------|-----------|
| 2D-to-BEV method | LSS (Lift-Splat-Shoot) | Explicit depth reasoning, geometry-aware |
| Temporal propagation | Streaming hidden state | O(1) compute and memory per frame |
| Temporal alignment | Ego-motion warping | Accurate spatial alignment via known pose |
| Temporal fusion | Cross-attention + gate | Selective info retrieval from history |
| Map decoder | DETR-style transformer | Parallel set prediction, permutation-invariant |
| Output format | Fixed-point polylines | Fixed K=20 points, simpler than autoregressive |
| Matching | Hungarian algorithm | Optimal assignment for set prediction loss |
| Point normalization | Sigmoid to [0,1] | Stable optimization, bounded output |

---

## References

- Philion & Fidler (2020). Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs. ECCV 2020.
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- Lin et al. (2017). Feature Pyramid Networks for Object Detection. CVPR 2017.
- Carion et al. (2020). End-to-End Object Detection with Transformers (DETR). ECCV 2020.
- Zhu et al. (2021). Deformable DETR: Deformable Transformers for End-to-End Object Detection. ICLR 2021.
- Yuan et al. (2024). StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction. WACV 2024.

---

## Hierarchical Lane Positional Embeddings (Enhancement)

This model has been enhanced with topology-aware hierarchical lane positional embeddings for lane detection.

### Architecture

StreamMapNet's decoder uses deformable cross-attention to attend to BEV features, with reference point refinement across 6 layers. The hierarchical lane PE is integrated via **position injection to Q/K only** (not to the residual stream), preserving the clean residual pathway that is critical for StreamMapNet's temporal fusion stability. The lane PE provides topology-aware structure to the attention mechanism without corrupting the feature representations that flow into the temporal hidden state. BEV positional interpolation is performed in fp32 to maintain precision when the BEV resolution changes between training and inference.

### Key Components

- `HierarchicalLanePositionalEmbedding` class: Generates sinusoidal positional embeddings encoding lane topology (lane index, boundary type, point index). The embeddings are injected only into Q and K projections of the self-attention layers, biasing attention patterns toward topologically-related queries without modifying the residual stream values.
- `LaneStreamDecoder` class: StreamMapNet-adapted decoder that applies lane PE to Q/K projections only. Uses a pre-computed decoupled attention mask (block-diagonal, 25 blocks of 40x40) to restrict self-attention to intra-lane interactions. BEV positional features used in cross-attention are interpolated in fp32 when spatial dimensions differ from the pre-computed grid.

### Implementation Details

- Position injection to Q/K only (not residual stream) -- prevents topology bias from contaminating temporal hidden state propagation
- Pre-computed decoupled mask: (1000, 1000) boolean block-diagonal, restricts self-attention to within-lane groups
- fp32 BEV positional interpolation: when BEV features are at a different resolution than the reference grid, `F.interpolate` is performed in fp32 before casting back to the working precision
- Compatible with StreamMapNet's temporal fusion: the hidden state H_t carries only content features (no PE leakage), so ego-motion warping remains geometrically correct
- Mask and PE are pre-computed at initialization and cached; device transfer handled via `_apply` override

### Design Decisions

- Balanced magnitude initialization (sinusoidal scaled to 0.02)
- Dropout 0.05 (reduced for geometric tasks)
- Inference caching with device-safe invalidation (_apply override)
- get_lane_mask returns clone for safety

### Query Layout

```
Total: 1000 queries (25 lanes x 2 lines x 20 points)
[0-19]:    Lane 0, Left boundary, points 0-19
[20-39]:   Lane 0, Right boundary, points 0-19
[40-59]:   Lane 1, Left boundary, points 0-19
...
[980-999]: Lane 24, Right boundary, points 0-19
```

### Usage Example

```python
from models.lane_pe import HierarchicalLanePositionalEmbedding

# Initialize for StreamMapNet (Q/K injection mode)
lane_pe = HierarchicalLanePositionalEmbedding(
    embed_dim=256,
    num_lanes=25,
    points_per_lane=20,
    num_boundaries=2
)

# Get topology PE and mask
topo_pe = lane_pe()                      # (1000, 256)
lane_mask = lane_pe.get_lane_mask()      # (1000, 1000) boolean

# Inject into Q/K only (not residual stream)
Q = W_q(lane_queries) + topo_pe          # topology-biased queries
K = W_k(lane_queries) + topo_pe          # topology-biased keys
V = W_v(lane_queries)                    # unmodified values

# Self-attention with decoupled mask
attn_logits = Q @ K.transpose(-1, -2) / sqrt(d_k)
attn_logits = attn_logits.masked_fill(~lane_mask, float('-inf'))
attn_output = softmax(attn_logits, dim=-1) @ V

# BEV cross-attention with fp32 positional interpolation
bev_pos = F.interpolate(bev_pos_grid.float(), size=(H_bev, W_bev), mode='bilinear').to(lane_queries.dtype)
output = deformable_cross_attention(lane_queries, bev_features + bev_pos, reference_points)
```
