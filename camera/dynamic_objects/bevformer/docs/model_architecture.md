# BEVFormer: Complete Model Architecture Guide

## Every Component Explained with Tensor Shapes, Math, and Diagrams

This document provides a complete walkthrough of BEVFormer's architecture for a Staff AI Engineer who knows PyTorch but is new to autonomous driving perception. Every component is explained with INPUT -> PROCESS -> OUTPUT, exact tensor shapes, and the reasoning behind design choices.

---

## 1. End-to-End Data Flow

### 1.1 Full Pipeline with Tensor Shapes

```
INPUT
=====
6 camera images: (B, 6, 3, 900, 1600)  -- B=batch, 6 cameras, RGB, H=900, W=1600
Camera calibration: (B, 6, 4, 4)        -- 6 lidar2img projection matrices
Previous BEV: (B, 256, 200, 200)         -- from last frame (or zeros if t=0)
Ego-motion: (B, 4, 4)                   -- transformation between frames

STAGE 1: IMAGE BACKBONE + FPN
==============================
Reshape: (B*6, 3, 900, 1600)            -- batch all cameras together
ResNet-101-DCN:
  Stage 1: (B*6, 256, 225, 400)         -- stride 4
  Stage 2: (B*6, 512, 113, 200)         -- stride 8     --> FPN input
  Stage 3: (B*6, 1024, 57, 100)         -- stride 16    --> FPN input
  Stage 4: (B*6, 2048, 29, 50)          -- stride 32    --> FPN input
FPN output:
  Level 0: (B*6, 256, 113, 200)         -- 1/8 resolution (fine detail)
  Level 1: (B*6, 256, 57, 100)          -- 1/16 resolution
  Level 2: (B*6, 256, 29, 50)           -- 1/32 resolution (coarse context)
Reshape back: (B, 6, num_levels, H_i*W_i, 256) for attention

STAGE 2: BEV ENCODER (6 layers)
================================
Input: bev_queries = learnable (B, 40000, 256) + positional encoding
For each of 6 encoder layers:
  Temporal Self-Attention:
    Q: (B, 40000, 256)       -- current BEV queries
    K,V: (B, 80000, 256)     -- concat(current, aligned_previous)
    Output: (B, 40000, 256)
  Spatial Cross-Attention:
    Q: (B, 40000, 256)       -- BEV queries
    K,V: image features      -- from all cameras and scales
    Output: (B, 40000, 256)
  FFN:
    Input: (B, 40000, 256)
    Hidden: (B, 40000, 512)
    Output: (B, 40000, 256)

Output: BEV features (B, 40000, 256) -> reshape to (B, 256, 200, 200)

STAGE 3: DETECTION DECODER (6 layers)
======================================
Input: object_queries = learnable (B, 900, 256)
       reference_points = learnable (B, 900, 2)
For each of 6 decoder layers:
  Self-Attention: (B, 900, 256) -> (B, 900, 256)
  Cross-Attention to BEV: (B, 900, 256) x (B, 40000, 256) -> (B, 900, 256)
  FFN: (B, 900, 256) -> (B, 900, 256)
  Reference point refinement: delta (B, 900, 2)

STAGE 4: DETECTION HEADS
=========================
Classification: (B, 900, 256) -> MLP -> (B, 900, 10)   -- 10 class logits
Regression: (B, 900, 256) -> MLP -> (B, 900, 10)       -- 10 box params

OUTPUT
======
Per decoder layer (for auxiliary losses):
  cls_scores: (B, 900, 10)
  bbox_preds: (B, 900, 10)  -- [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
```

---

## 2. Image Backbone: ResNet-101-DCN

### 2.1 What ResNet Does

ResNet-101 is a convolutional neural network with 101 layers that converts raw RGB images into hierarchical feature maps. Each stage doubles the number of channels while halving spatial resolution, creating features at multiple levels of abstraction:

- Early stages: detect edges, textures, colors (low-level)
- Middle stages: detect parts (wheels, windows, heads)
- Late stages: detect full objects and their relationships (high-level)

### 2.2 What Are Deformable Convolutions (DCN)?

Standard convolutions sample a fixed 3x3 grid of positions:

```
Standard 3x3 Conv:              Deformable Conv:
Fixed sampling grid             Learned offset per position

+---+---+---+                   +       +
| x | x | x |                       x       x
+---+---+---+                     x
| x | x | x |                         x
+---+---+---+                   x           x
| x | x | x |                       x   x
+---+---+---+                         x

Every pixel uses the                Each pixel learns WHERE
exact same grid pattern.            to sample -- offsets are input-dependent.
```

**Why DCN in stages 3-4?** Objects in driving scenes have diverse shapes and orientations. A standard conv with a rigid grid cannot adapt to deformable objects (turning trucks, leaning pedestrians). DCN learns to deform its sampling grid to match the object's shape, improving feature extraction for non-rigid objects.

**How DCN works:**
1. A parallel conv layer predicts 2D offsets for each sampling position: `offsets = conv(input)` -- shape (B, 2*K*K, H, W) where K=3 for 3x3
2. The main conv samples at `original_position + learned_offset` using bilinear interpolation
3. This is fully differentiable and trained end-to-end

### 2.3 Feature Pyramid Network (FPN)

FPN creates multi-scale features with a uniform channel dimension by combining deep (semantic) and shallow (high-resolution) features:

```
                  Backbone                    FPN (top-down + lateral)
                  ========                    ========================

    C2 (512ch, 113x200)  ----[1x1 conv]---> P2 (256ch, 113x200)  Level 0
           |                                       ^
           v                                       | (upsample 2x + add)
    C3 (1024ch, 57x100)  ----[1x1 conv]---> P3 (256ch, 57x100)   Level 1
           |                                       ^
           v                                       | (upsample 2x + add)
    C4 (2048ch, 29x50)   ----[1x1 conv]---> P4 (256ch, 29x50)    Level 2
```

**Why multi-scale?**
- Small objects (pedestrians, traffic cones) need high-resolution features -> Level 0
- Large objects (trucks, buses) need large receptive fields -> Level 2
- Multi-scale deformable attention lets BEVFormer sample from ALL levels simultaneously

### 2.4 Exact Tensor Shapes Through Backbone

For a single 900x1600 image:

| Component | Output Shape | Channels | Spatial | Notes |
|-----------|-------------|----------|---------|-------|
| Input | (3, 900, 1600) | 3 | 900x1600 | RGB |
| Stem (7x7 conv + pool) | (64, 225, 400) | 64 | 225x400 | Stride 4 |
| Stage 1 (3 blocks) | (256, 225, 400) | 256 | 225x400 | No downsampling |
| Stage 2 (4 blocks) | (512, 113, 200) | 512 | 113x200 | Stride 8 total |
| Stage 3 (23 blocks, DCN) | (1024, 57, 100) | 1024 | 57x100 | Stride 16 total |
| Stage 4 (3 blocks, DCN) | (2048, 29, 50) | 2048 | 29x50 | Stride 32 total |
| FPN Level 0 | (256, 113, 200) | 256 | 113x200 | From C2 |
| FPN Level 1 | (256, 57, 100) | 256 | 57x100 | From C3 |
| FPN Level 2 | (256, 29, 50) | 256 | 29x50 | From C4 |

**Total image feature tokens (per camera):** 113*200 + 57*100 + 29*50 = 22,600 + 5,700 + 1,450 = 29,750
**Total across 6 cameras:** 6 x 29,750 = 178,500 tokens of dimension 256

---

## 3. BEV Queries: The Foundation

### 3.1 What Are BEV Queries?

BEV queries are LEARNABLE parameter vectors (one per grid cell) that will be iteratively filled with information from camera images. Think of them as "empty slots" that ask: "What is in this patch of ground?"

```python
# Initialization
bev_queries = nn.Embedding(200 * 200, 256)  # 40,000 learnable 256-dim vectors
bev_pos = LearnedPositionalEncoding(num_feats=128, row_num_embed=200, col_num_embed=200)
```

### 3.2 Physical Interpretation

```
Grid cell (i, j) represents the ground patch:
  x_center = (i + 0.5) / 200 * 102.4 - 51.2  meters (forward/back)
  y_center = (j + 0.5) / 200 * 102.4 - 51.2  meters (left/right)
  size: 0.512m x 0.512m

Example:
  Cell (0, 0):     x=-51.0m, y=-51.0m   (far back-right)
  Cell (100, 100): x=0.26m,  y=0.26m    (directly at ego vehicle)
  Cell (199, 199): x=50.7m,  y=50.7m    (far front-left)
```

### 3.3 3D Reference Points (Pillars)

Each BEV query gets associated 3D reference points at multiple heights. These define WHERE in 3D space this query should look:

```python
# For BEV position (x, y), generate pillar of reference points:
z_values = [-1.0, 1.0, 3.0, 5.0]  # 4 heights in meters

# Result: reference_points shape (B, 40000, 4, 3)
# Each BEV query has 4 points: (x, y, z1), (x, y, z2), (x, y, z3), (x, y, z4)
```

**Why multiple heights?** A single (x, y) position might contain a road surface (z=0), a car body (z=1.5), a truck top (z=3.5), or a traffic sign (z=4). Sampling at multiple heights ensures we capture features at all relevant elevations.

---

## 4. Spatial Cross-Attention (SCA): The Core Innovation

### 4.1 Purpose

Spatial Cross-Attention answers: "For each BEV grid cell, what camera image features are relevant?" It does this by projecting 3D reference points to camera images and using deformable attention to sample features.

### 4.2 Step-by-Step Algorithm

```
Input:
  bev_queries: (B, 40000, 256)           -- queries to fill
  image_features: (B, 6, 3, *, 256)      -- multi-cam, multi-scale features
  reference_points_3d: (B, 40000, 4, 3)  -- 3D pillar points
  lidar2img: (B, 6, 4, 4)               -- projection matrices

Algorithm:
  1. Project all reference points to all cameras:
     ref_2d = project(reference_points_3d, lidar2img)
     # Shape: (B, 6, 40000, 4, 2) -- 6 cams, 40k queries, 4 heights, (u,v)

  2. Determine valid projections (within image bounds):
     valid = (ref_2d[..., 0] >= 0) & (ref_2d[..., 0] < W) &
             (ref_2d[..., 1] >= 0) & (ref_2d[..., 1] < H) &
             (depth > 0)
     # Shape: (B, 6, 40000, 4) boolean mask

  3. For each BEV query, for each valid camera-height pair:
     Apply deformable attention:
       - 8 heads, 4 sampling points per head
       - Sampling offsets learned from query: offsets = Linear(query)
       - Attention weights learned from query: weights = Linear(query)
       - Sample features at (ref_2d + offsets) from multi-scale image features
       - Weighted sum of sampled features

  4. Aggregate across cameras and heights:
     output[q] = weighted_sum(features from all valid cameras/heights for query q)

Output:
  updated_bev_queries: (B, 40000, 256)
```

### 4.3 The 3D-to-2D Projection Math

```python
def project_3d_to_cameras(ref_3d, lidar2img):
    """
    ref_3d: (B, N, 4, 3) -- N queries, 4 heights, (x,y,z)
    lidar2img: (B, 6, 4, 4) -- projection matrices per camera
    
    Returns: (B, 6, N, 4, 2) pixel coordinates
    """
    # Make homogeneous: (B, N, 4, 4) with w=1
    ref_homo = F.pad(ref_3d, (0, 1), value=1.0)  # append 1
    
    # Project to each camera: (B, 6, N, 4, 4)
    # lidar2img @ ref_homo^T for each camera
    ref_cam = torch.einsum('bnij,bqhj->bnqhi', lidar2img, ref_homo)
    
    # Perspective divide
    depth = ref_cam[..., 2:3]  # z coordinate = depth
    ref_2d = ref_cam[..., :2] / torch.clamp(depth, min=1e-5)  # u, v
    
    # Normalize to [0, 1] for deformable attention
    ref_2d[..., 0] /= W  # normalize u
    ref_2d[..., 1] /= H  # normalize v
    
    return ref_2d, depth
```

### 4.4 Deformable Attention in Detail

For each BEV query at reference point `p` in a camera image:

```
For head m = 1..8:
  For sample point k = 1..4:
    # Compute sampling location
    offset_mk = Linear_offset(query)[m, k]   # learned 2D offset (dx, dy)
    sample_loc = p + offset_mk               # actual sampling location
    
    # Sample feature via bilinear interpolation
    feature_mk = bilinear_sample(image_features, sample_loc)  # (256/8 = 32)-dim
    
    # Compute attention weight
    weight_mk = softmax(Linear_weight(query)[m, :])[k]  # scalar, sums to 1 over k
    
  # Aggregate within head
  head_output_m = sum_k(weight_mk * feature_mk)  # 32-dim

# Combine all heads
output = Linear_out(concat(head_1, ..., head_8))  # 256-dim
```

### 4.5 SCA Parameters Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Attention heads | 8 | Multiple attention patterns |
| Sampling points per head | 4 | Sparse sampling around reference |
| Reference heights | 4 (-1, 1, 3, 5 m) | Cover vertical extent |
| Feature scales | 3 (from FPN) | Multi-scale receptive field |
| Typical valid cameras per query | 1-3 | Only attend to relevant views |

---

## 5. Temporal Self-Attention (TSA)

### 5.1 Purpose

TSA enables the current frame's BEV queries to leverage information from the previous frame, aligned by ego-motion. This provides:
1. Velocity estimation (how far did objects move between frames?)
2. Temporal smoothing (reduce flickering detections)
3. Occluded object memory (object hidden in current frame may have been visible before)

### 5.2 Ego-Motion Alignment

```python
def align_previous_bev(prev_bev, ego_motion_matrix):
    """
    Warp previous BEV features to current ego frame.
    
    prev_bev: (B, 256, 200, 200) -- previous frame features
    ego_motion_matrix: (B, 4, 4) -- transform from current to previous ego frame
    """
    # Create grid of current BEV coordinates
    xs = torch.linspace(-51.2, 51.2, 200)
    ys = torch.linspace(-51.2, 51.2, 200)
    grid_y, grid_x = torch.meshgrid(ys, xs)
    
    # Transform current positions to previous frame
    # "Where was this position in the previous BEV?"
    current_pts = torch.stack([grid_x, grid_y, 
                               torch.zeros_like(grid_x),
                               torch.ones_like(grid_x)], dim=-1)  # (200, 200, 4)
    
    prev_pts = (ego_motion_matrix @ current_pts.unsqueeze(-1)).squeeze(-1)
    
    # Normalize to [-1, 1] for grid_sample
    sample_grid = prev_pts[..., :2] / 51.2  # (200, 200, 2)
    
    # Bilinear sampling from previous BEV
    aligned = F.grid_sample(prev_bev, sample_grid, align_corners=True)
    return aligned  # (B, 256, 200, 200)
```

### 5.3 Temporal Deformable Self-Attention

After alignment:

```
Q = current_bev_queries              (B, 40000, 256)
K = concat(current, aligned_prev)    (B, 80000, 256)
V = concat(current, aligned_prev)    (B, 80000, 256)
reference_points = bev_grid_centers  (B, 40000, 2)  -- normalized [0,1]

Output = DeformableAttention(Q, K, V, reference_points)
  - 8 heads, 4 sampling points per head
  - Samples from both current and previous features
  - Network learns whether to attend to current or previous
```

### 5.4 First Frame Handling

```python
if prev_bev is None:  # First frame of sequence
    prev_bev = bev_queries.clone()  # Use current queries as "previous"
    # TSA degenerates to standard self-attention
```

---

## 6. BEV Encoder: Stacking It All Together

### 6.1 Layer Structure

Each of the 6 encoder layers applies (in order):

```
Input: x (B, 40000, 256)
  |
  |-- x = x + TSA(LayerNorm(x), aligned_prev_bev)      # Temporal
  |
  |-- x = x + SCA(LayerNorm(x), image_features)        # Spatial
  |
  |-- x = x + FFN(LayerNorm(x))                        # Feed-forward
  |
Output: x (B, 40000, 256)
```

This is **pre-norm** (LayerNorm before attention, not after). Pre-norm transformers train more stably because the residual path is unperturbed.

### 6.2 Feed-Forward Network

```python
class FFN(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(256, 512)   # expand
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(512, 256)   # contract
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
```

### 6.3 Progressive Refinement

Through 6 layers, BEV features are progressively refined:
- Layer 1-2: Coarse filling (basic object presence)
- Layer 3-4: Spatial refinement (object boundaries)
- Layer 5-6: Fine detail (small objects, precise localization)

---

## 7. Detection Decoder (DETR-style)

### 7.1 What Are Object Queries?

Object queries are 900 learnable "slots" -- each one MIGHT become a detection. The decoder's job is to have each query specialize to one object (or background).

```python
object_queries = nn.Embedding(900, 256)       # learnable content
reference_points = nn.Linear(256, 2)          # learnable 2D positions in BEV
```

### 7.2 Self-Attention Among Queries

**Purpose:** Prevent duplicate detections. If query A and query B are both trying to detect the same car, self-attention lets them "communicate" and one can defer to the other.

```
Q = K = V = LayerNorm(object_queries)  (B, 900, 256)
Output = MultiHeadAttention(Q, K, V)   (B, 900, 256)
  - Standard (not deformable) attention
  - 8 heads, head_dim = 32
  - Full attention between all 900 queries
  - Cost: 900 x 900 = 810,000 (manageable)
```

### 7.3 Cross-Attention to BEV Features

**Purpose:** Each query "looks" at the BEV feature map to find its object.

```
Q = LayerNorm(object_queries)    (B, 900, 256)
K = V = bev_features.flatten()   (B, 40000, 256)
ref = reference_points           (B, 900, 2)  -- where to look in BEV

Output = DeformableAttention(Q, K, V, ref)  (B, 900, 256)
  - 8 heads, 4 sampling points per head
  - Samples BEV features near the reference point
```

### 7.4 Iterative Reference Point Refinement

After each decoder layer, reference points are updated:

```python
for layer in decoder_layers:
    output = layer(queries, bev_features, reference_points)
    
    # Predict offset to refine reference point
    delta = regression_branch(output)[..., :2]  # (B, 900, 2)
    reference_points = (reference_points + delta.sigmoid()).detach()
```

**Why detach?** Stopping gradient at reference points prevents training instability. Each layer gets clean reference points without backpropagation through the entire refinement chain.

---

## 8. Detection Heads

### 8.1 Classification Head

```python
class ClassificationHead(nn.Module):
    # 3-layer MLP: 256 -> 256 -> 256 -> 10
    def forward(self, x):  # x: (B, 900, 256)
        x = self.relu(self.linear1(x))   # (B, 900, 256)
        x = self.relu(self.linear2(x))   # (B, 900, 256)
        x = self.linear3(x)              # (B, 900, 10) -- raw logits
        return x  # Apply sigmoid for probabilities
```

**Loss:** Focal Loss (alpha=0.25, gamma=2.0, weight=2.0)

Focal Loss down-weights easy negatives (most of the 900 queries are background), focusing learning on hard positives and hard negatives.

### 8.2 Regression Head

```python
class RegressionHead(nn.Module):
    # 3-layer MLP: 256 -> 256 -> 256 -> 10
    def forward(self, x):  # x: (B, 900, 256)
        x = self.relu(self.linear1(x))   # (B, 900, 256)
        x = self.relu(self.linear2(x))   # (B, 900, 256)
        x = self.linear3(x)              # (B, 900, 10) -- box parameters
        return x
```

**Loss:** L1 Loss (weight=0.25)

### 8.3 The 10 Regression Parameters Explained

| Index | Parameter | Range | Encoding | What It Represents |
|-------|-----------|-------|----------|-------------------|
| 0 | cx | [-51.2, 51.2] m | sigmoid * range | Center X position |
| 1 | cy | [-51.2, 51.2] m | sigmoid * range | Center Y position |
| 2 | cz | [-5.0, 3.0] m | direct | Center Z (height above ground) |
| 3 | w | [0, 20] m | exp(log-space) | Box width |
| 4 | l | [0, 20] m | exp(log-space) | Box length |
| 5 | h | [0, 10] m | exp(log-space) | Box height |
| 6 | sin(yaw) | [-1, 1] | direct | Sin of heading angle |
| 7 | cos(yaw) | [-1, 1] | direct | Cos of heading angle |
| 8 | vx | [-20, 20] m/s | direct | Velocity X |
| 9 | vy | [-20, 20] m/s | direct | Velocity Y |

**Why sin/cos instead of angle directly?**
- Angles have discontinuities (359 deg and 1 deg are close but numerically far)
- sin/cos representation is continuous: sin(359 deg) is close to sin(1 deg)
- atan2(sin, cos) recovers the angle without ambiguity

**Why exp for sizes?**
- Sizes must be positive. exp() guarantees positivity.
- The network predicts in log-space, which makes proportional errors equal for large and small objects.

### 8.4 Hungarian Matching

During training, we must assign each of the 900 predictions to a ground truth object (or background). This is done using the Hungarian algorithm:

```
Cost matrix: (900 predictions) x (N_gt objects)
  cost[i, j] = focal_loss(pred_class[i], gt_class[j]) 
             + L1_loss(pred_box[i], gt_box[j])

Hungarian algorithm finds the minimum-cost bipartite matching.
Unmatched predictions are assigned to "background" class.
```

---

## 9. Parameter Count Breakdown

### 9.1 BEVFormer-Base (ResNet-101-DCN)

| Component | Parameters | % of Total | Notes |
|-----------|-----------|------------|-------|
| ResNet-101-DCN backbone | 44.5M | 63.5% | Pretrained on ImageNet + FCOS3D |
| FPN | 3.5M | 5.0% | 1x1 convs + 3x3 convs |
| BEV Queries (learnable) | 10.2M | 14.6% | 40,000 x 256 |
| BEV Encoder (6 layers) | 6.0M | 8.6% | TSA + SCA + FFN per layer |
| Detection Decoder (6 layers) | 4.9M | 7.0% | Self-attn + cross-attn + FFN |
| Detection Heads | 0.3M | 0.4% | Classification + regression MLPs |
| Positional Encodings | 0.6M | 0.9% | Learned positional embeddings |
| **Total** | **~70.0M** | **100%** | |

**Key observation:** The backbone dominates (63.5%). The "novel" BEVFormer components (encoder + decoder) are only ~16% of parameters. This is why pretrained backbone weights matter so much for final performance.

---

## 10. Memory and Compute Analysis

### 10.1 GPU Memory Breakdown (Training, batch_size=1, FP16)

| Component | Memory | Notes |
|-----------|--------|-------|
| Model parameters | 0.3 GB | 70M params x 4 bytes (FP32 master copy) |
| Optimizer states (AdamW) | 0.6 GB | Momentum + variance |
| Image features (6 cameras, 3 scales) | 2.5 GB | Stored for backward pass |
| BEV encoder activations | 4.0 GB | 6 layers of attention maps |
| Detection decoder activations | 1.0 GB | 6 layers, smaller attention maps |
| Gradient tensors | 3.0 GB | Same size as activations |
| Temporary buffers | 1.5 GB | Intermediate computations |
| **Total (approximate)** | **~13 GB (FP16)** | |
| **Total (FP32)** | **~22 GB** | |

### 10.2 Computational Bottleneck

The BEV encoder is the most expensive component:
- 40,000 queries x 6 cameras x 4 heights x 8 heads x 4 points = many sampling operations
- 6 layers multiplies this cost
- This is why the encoder takes 36% of inference time

### 10.3 Inference Timing (A100 GPU)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Backbone + FPN | 45 | 42% |
| BEV Encoder | 38 | 36% |
| Detection Decoder | 15 | 14% |
| Post-processing | 8 | 8% |
| **Total** | **~106 ms** | **~9.4 FPS** |

### 10.4 Strategies for Reducing Memory/Compute

| Strategy | Memory Savings | Speed Impact | Accuracy Impact |
|----------|---------------|--------------|-----------------|
| Gradient checkpointing (backbone) | -3 GB | +20% time | None |
| Reduce BEV: 200x200 -> 100x100 | -8 GB | +40% faster | -3 NDS |
| Reduce encoder: 6 -> 3 layers | -2 GB | +25% faster | -2 NDS |
| Reduce queries: 900 -> 300 | -0.5 GB | +10% faster | -1 NDS |
| FP16 mixed precision | -40% total | -20% time | Negligible |
| Reduce image res: 900x1600 -> 450x800 | -4 GB | +30% faster | -3 NDS |

---

## 11. Model Variants

| Variant | Backbone | BEV Size | Encoder Layers | Params | NDS | Use Case |
|---------|----------|----------|----------------|--------|-----|----------|
| BEVFormer-Tiny | ResNet-50 | 50x50 | 3 | ~35M | ~42 | Debugging, rapid iteration |
| BEVFormer-Small | ResNet-101 | 100x100 | 6 | ~55M | 47.8 | Limited GPU memory |
| BEVFormer-Base | ResNet-101-DCN | 200x200 | 6 | ~70M | 56.9 | Standard research |
| BEVFormer-Large | V2-99 | 200x200 | 6 | ~78M | 59.2 | Maximum accuracy |

---

## 12. Loss Computation and Training

### 12.1 Auxiliary Losses

Every decoder layer produces predictions, and loss is computed at EVERY layer (not just the last). This provides gradient signal to early layers:

```python
total_loss = 0
for layer_idx in range(6):
    cls_scores = cls_head(decoder_output[layer_idx])  # (B, 900, 10)
    bbox_preds = reg_head(decoder_output[layer_idx])  # (B, 900, 10)
    
    # Hungarian matching (may differ per layer)
    matched = hungarian_match(cls_scores, bbox_preds, gt_boxes, gt_labels)
    
    # Compute losses
    loss_cls = focal_loss(cls_scores, matched_labels)
    loss_reg = l1_loss(bbox_preds[matched_indices], gt_boxes[matched_indices])
    
    total_loss += loss_cls * 2.0 + loss_reg * 0.25
```

### 12.2 Loss Weights

| Loss | Weight | Rationale |
|------|--------|-----------|
| Classification (Focal) | 2.0 | Higher weight because most queries are background |
| Regression (L1) | 0.25 | Lower weight because box params have larger magnitude |
| GIoU | 0.0 | Not used (3D IoU is expensive and noisy) |
