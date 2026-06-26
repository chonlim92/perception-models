# BEVFormer: Model Architecture

## Detailed Layer-by-Layer Architecture

This document provides a comprehensive description of BEVFormer's architecture, including all components, dimensions, and computation flow.

---

## 1. Architecture Summary

```
Input: 6 camera images (900 x 1600 x 3)
  |
  v
[Image Backbone: ResNet-101-DCN + FPN]
  |
  +--> C3: 6 x 256 x 113 x 200  (1/8 scale)
  +--> C4: 6 x 256 x 57 x 100   (1/16 scale)
  +--> C5: 6 x 256 x 29 x 50    (1/32 scale)
  |
  v
[BEV Encoder: 6 Transformer Encoder Layers]
  Input: BEV Queries (200 x 200 x 256)
  |
  | Each layer:
  |   1. Temporal Self-Attention
  |   2. Spatial Cross-Attention  
  |   3. Feed-Forward Network
  |
  v
BEV Features: 200 x 200 x 256
  |
  v
[Detection Decoder: 6 Transformer Decoder Layers]
  Input: 900 Object Queries (900 x 256)
  |
  | Each layer:
  |   1. Self-Attention (among queries)
  |   2. Cross-Attention (to BEV features)
  |   3. Feed-Forward Network
  |
  v
[Detection Heads]
  +--> Classification: 900 x 10
  +--> Regression: 900 x 10 (cx, cy, cz, w, l, h, sin, cos, vx, vy)
```

---

## 2. Input Specification

| Property | Value |
|----------|-------|
| Number of cameras | 6 |
| Image resolution | 900 (H) x 1600 (W) |
| Color channels | 3 (RGB) |
| Pixel range | [0, 255] normalized to [0, 1] |
| Normalization | ImageNet mean/std |
| Input tensor shape | `(B, 6, 3, 900, 1600)` |

### Image Preprocessing

```python
# Standard normalization
mean = [0.485, 0.456, 0.406]  # ImageNet
std = [0.229, 0.224, 0.225]   # ImageNet

# Data augmentation (training only)
# - Random resize: scale [0.38, 0.55] relative to original
# - Random crop to 900x1600
# - Random horizontal flip
# - PhotoMetricDistortion (brightness, contrast, saturation, hue)
# - Grid mask augmentation
```

---

## 3. Image Backbone

### 3.1 ResNet-101 with DCN

BEVFormer uses ResNet-101 pretrained on ImageNet, enhanced with Deformable Convolutional Networks (DCN) in stages 3 and 4.

| Stage | Output Name | Output Size | Channels | Stride | DCN |
|-------|-------------|-------------|----------|--------|-----|
| Stem | - | 225 x 400 | 64 | 4 | No |
| Stage 1 | C1 | 225 x 400 | 256 | 4 | No |
| Stage 2 | C2 | 113 x 200 | 512 | 8 | No |
| Stage 3 | C3 | 57 x 100 | 1024 | 16 | Yes |
| Stage 4 | C4 | 29 x 50 | 2048 | 32 | Yes |

**Note:** Sizes shown for a single 900x1600 input image.

### 3.2 Feature Pyramid Network (FPN)

FPN produces multi-scale features with a unified channel dimension:

```
C4 (2048 ch) ----[1x1 conv]--> P4 (256 ch, 29x50)
                                  |
                    [upsample 2x + add]
                                  |
C3 (1024 ch) ----[1x1 conv]--> P3 (256 ch, 57x100)
                                  |
                    [upsample 2x + add]
                                  |
C2 (512 ch) -----[1x1 conv]--> P2 (256 ch, 113x200)
```

| FPN Level | Spatial Size (per camera) | Channels | Scale Factor |
|-----------|--------------------------|----------|--------------|
| P2 (Level 0) | 113 x 200 | 256 | 1/8 |
| P3 (Level 1) | 57 x 100 | 256 | 1/16 |
| P4 (Level 2) | 29 x 50 | 256 | 1/32 |

**Total image features per sample:**
- 6 cameras x 3 levels x 256 channels
- Combined spatial elements: 6 x (113x200 + 57x100 + 29x50) = 6 x 29,750 = 178,500 tokens

### 3.3 Alternative Backbone: VoVNet-99 (V2-99)

Used in BEVFormer-Large for better performance:

| Property | ResNet-101-DCN | V2-99 |
|----------|---------------|-------|
| Parameters | 44.5M | 52.7M |
| FLOPs (per image) | 7.8G | 14.3G |
| ImageNet Top-1 | 77.4% | 81.2% |
| nuScenes NDS improvement | - | +2.3 |

---

## 4. BEV Queries

### 4.1 Specification

| Property | Value |
|----------|-------|
| Grid size | 200 x 200 |
| Embedding dimension | 256 |
| Spatial resolution | 0.512m per grid cell |
| X range | [-51.2m, 51.2m] |
| Y range | [-51.2m, 51.2m] |
| Total queries | 40,000 |
| Learnable parameters | 40,000 x 256 = 10.24M |

### 4.2 BEV Query Initialization

```python
# Learnable BEV query embeddings
bev_queries = nn.Embedding(200 * 200, 256)  # (40000, 256)

# Fixed positional encoding for BEV grid
bev_pos = LearnedPositionalEncoding(
    num_feats=128,  # 256/2 for sin+cos
    row_num_embed=200,
    col_num_embed=200
)
```

### 4.3 3D Reference Points

Each BEV query has associated 3D reference points for spatial cross-attention:

```python
# Reference point generation
# For BEV grid position (i, j):
x = (i + 0.5) / 200 * 102.4 - 51.2  # X coordinate in meters
y = (j + 0.5) / 200 * 102.4 - 51.2  # Y coordinate in meters

# Sample N_ref heights along Z-axis
z_values = [-1.0, 1.0, 3.0, 5.0]  # 4 reference heights (meters)

# Total 3D reference points: 200 x 200 x 4 = 160,000
```

---

## 5. Spatial Cross-Attention (SCA)

### 5.1 Overview

Spatial cross-attention enables BEV queries to aggregate features from multi-camera image features based on geometric projection.

```
BEV Query (at position x,y)
    |
    | Generate 3D reference points at multiple heights
    v
3D Points: [(x, y, z1), (x, y, z2), (x, y, z3), (x, y, z4)]
    |
    | Project to each camera using calibration
    v
2D Points per camera: [(u1, v1), (u2, v2), ...]
    |
    | Filter: keep only valid projections (within image bounds)
    v
Valid camera-point pairs
    |
    | Deformable attention around each 2D point
    v
Aggregated feature for BEV query
```

### 5.2 Detailed Computation

```python
def spatial_cross_attention(bev_queries, image_features, reference_points, 
                             lidar2img_transforms):
    """
    Args:
        bev_queries: (B, H*W, C) = (B, 40000, 256)
        image_features: (B, N_cam, N_levels, H_i*W_i, C)
        reference_points: (B, H*W, N_ref, 3) = (B, 40000, 4, 3)
        lidar2img_transforms: (B, N_cam, 4, 4)
    
    Returns:
        output: (B, H*W, C) = (B, 40000, 256)
    """
    B, num_queries, num_ref, _ = reference_points.shape
    
    # Step 1: Project 3D reference points to all cameras
    # reference_points_3d: (B, 40000, 4, 3) -> (B, 40000, 4, 4) [homogeneous]
    ref_3d_homo = torch.cat([reference_points, 
                              torch.ones_like(reference_points[..., :1])], dim=-1)
    
    # Project to each camera: (B, N_cam, 40000, 4, 2) pixel coordinates
    ref_2d = project_to_cameras(ref_3d_homo, lidar2img_transforms)
    
    # Step 2: Determine valid projections (within image bounds)
    valid_mask = (ref_2d[..., 0] >= 0) & (ref_2d[..., 0] < W) & \
                 (ref_2d[..., 1] >= 0) & (ref_2d[..., 1] < H)
    
    # Step 3: Apply deformable attention for each valid projection
    # Attention with 8 heads, 4 sampling points per head
    output = deformable_attention(
        query=bev_queries,
        reference_points=ref_2d,  # normalized to [0, 1]
        input_flatten=image_features,
        spatial_shapes=spatial_shapes,
        sampling_offsets=self.sampling_offsets(bev_queries),  # learned offsets
        attention_weights=self.attention_weights(bev_queries)  # learned weights
    )
    
    return output
```

### 5.3 SCA Parameters

| Parameter | Value |
|-----------|-------|
| Number of attention heads | 8 |
| Sampling points per head | 4 |
| Number of reference heights (N_ref) | 4 |
| Reference height values | [-1.0, 1.0, 3.0, 5.0] m |
| Total sampling points per query | 8 heads x 4 points x 4 heights x N_valid_cams |
| Feature levels attended | 3 (from FPN) |

### 5.4 Deformable Attention Mechanism

```
For each BEV query q at position (x, y):
  For each reference height z_k (k = 1..4):
    For each camera c (c = 1..6):
      if project(x, y, z_k) is visible in camera c:
        ref_2d = project(x, y, z_k, cam_c)  # 2D reference point
        For each head h (h = 1..8):
          For each sample s (s = 1..4):
            offset = learned_offset(q, h, s)  # 2D offset
            sample_loc = ref_2d + offset
            feature = bilinear_sample(image_features[c], sample_loc)
            weight = attention_weight(q, h, s)
        weighted_feature = sum(weight * feature)
  
  output = linear_projection(concatenate(all_head_features))
```

---

## 6. Temporal Self-Attention (TSA)

### 6.1 Overview

Temporal self-attention allows current BEV features to attend to the previous frame's BEV features, aligned using ego-motion.

```
BEV Features (t-1)                BEV Queries (t)
      |                                 |
      | Ego-motion alignment            |
      v                                 v
Aligned BEV (t-1) --------> Deformable Self-Attention
                                        |
                                        v
                              Updated BEV Queries (t)
```

### 6.2 Ego-Motion Alignment

```python
def align_previous_bev(prev_bev, ego_motion):
    """
    Align previous BEV features to current ego frame.
    
    Args:
        prev_bev: (B, C, H, W) = (B, 256, 200, 200)
        ego_motion: (B, 4, 4) transformation from t-1 to t
    
    Returns:
        aligned_bev: (B, C, H, W) = (B, 256, 200, 200)
    """
    # Generate grid coordinates for current BEV
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-51.2, 51.2, 200),
        torch.linspace(-51.2, 51.2, 200)
    )
    grid = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x), 
                        torch.ones_like(grid_x)], dim=-1)  # (200, 200, 4)
    
    # Transform current grid to previous frame
    grid_prev = (torch.inverse(ego_motion) @ grid.unsqueeze(-1)).squeeze(-1)
    
    # Normalize to [-1, 1] for grid_sample
    grid_norm = grid_prev[..., :2] / 51.2  # normalize by BEV range
    
    # Bilinear interpolation
    aligned_bev = F.grid_sample(prev_bev, grid_norm, align_corners=True)
    
    return aligned_bev
```

### 6.3 Temporal Deformable Self-Attention

```python
def temporal_self_attention(bev_queries, prev_bev_aligned):
    """
    Args:
        bev_queries: (B, H*W, C) = (B, 40000, 256) current queries
        prev_bev_aligned: (B, H*W, C) = (B, 40000, 256) aligned previous BEV
    
    Returns:
        output: (B, H*W, C) = (B, 40000, 256)
    """
    # Concatenate current and previous as key/value
    # Current queries attend to both current positions and previous features
    kv = torch.cat([bev_queries, prev_bev_aligned], dim=1)  # (B, 80000, 256)
    
    # Reference points: center of each BEV query position
    ref_points = get_bev_reference_points(200, 200)  # (B, 40000, 1, 2)
    
    # Deformable attention
    output = deformable_attention(
        query=bev_queries,
        reference_points=ref_points,
        input_flatten=kv,
        num_heads=8,
        num_points=4
    )
    
    return output
```

### 6.4 TSA Parameters

| Parameter | Value |
|-----------|-------|
| Number of attention heads | 8 |
| Sampling points per head | 4 |
| Temporal frames | 1 previous frame (default) |
| Temporal interval | 0.5s (at 2 Hz keyframe rate) |
| Key/Value size | 2x BEV (current + previous) = 80,000 tokens |

### 6.5 First Frame Handling

At the first frame of a sequence (no previous BEV available):
- `prev_bev_aligned` is set to a copy of the current `bev_queries`
- TSA degenerates to standard self-attention over the BEV grid
- This ensures the model works for single-frame inference

---

## 7. BEV Encoder

### 7.1 Structure

The BEV encoder consists of 6 identical layers, each with three sub-layers:

```
Input: BEV Queries (B, 40000, 256)
  |
  v
[Layer 1]
  |-- LayerNorm
  |-- Temporal Self-Attention
  |-- Residual Connection
  |
  |-- LayerNorm
  |-- Spatial Cross-Attention
  |-- Residual Connection
  |
  |-- LayerNorm
  |-- Feed-Forward Network
  |-- Residual Connection
  |
  v
[Layer 2] ... [Layer 6]
  |
  v
Output: BEV Features (B, 40000, 256) -> reshape to (B, 256, 200, 200)
```

### 7.2 Feed-Forward Network (FFN)

```python
class FFN(nn.Module):
    def __init__(self, embed_dim=256, feedforward_dim=512, dropout=0.1):
        self.linear1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(512, 256)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

### 7.3 Encoder Layer Parameters

| Component | Parameters per Layer |
|-----------|---------------------|
| Temporal Self-Attention | |
| - Query/Key/Value projections | 3 x (256 x 256) = 196,608 |
| - Sampling offsets | 8 x 4 x 2 x 256 = 16,384 |
| - Attention weights | 8 x 4 x 256 = 8,192 |
| - Output projection | 256 x 256 = 65,536 |
| Spatial Cross-Attention | |
| - Query/Key/Value projections | 3 x (256 x 256) = 196,608 |
| - Sampling offsets | 8 x 4 x 3 x 4 x 256 = 98,304 |
| - Attention weights | 8 x 4 x 3 x 4 x 256 = 98,304 |
| - Output projection | 256 x 256 = 65,536 |
| FFN | |
| - Linear 1 | 256 x 512 + 512 = 131,584 |
| - Linear 2 | 512 x 256 + 256 = 131,328 |
| Layer Norms (x3) | 3 x (256 + 256) = 1,536 |
| **Total per layer** | **~1.0M** |
| **Total encoder (6 layers)** | **~6.0M** |

---

## 8. Detection Decoder

### 8.1 Overview

The detection decoder follows the Deformable DETR design, using learnable object queries that interact with the BEV features.

### 8.2 Object Queries

| Property | Value |
|----------|-------|
| Number of queries | 900 |
| Query dimension | 256 |
| Learnable reference points | 900 x 2 (x, y in BEV) |
| Total query parameters | 900 x 256 + 900 x 2 = 232,200 |

### 8.3 Decoder Layer Structure

```
Object Queries (900, 256)
  |
  v
[Decoder Layer 1-6]
  |-- LayerNorm
  |-- Multi-Head Self-Attention (among 900 queries)
  |      - 8 heads, head_dim=32
  |      - Prevents duplicate detections
  |-- Residual Connection
  |
  |-- LayerNorm
  |-- Cross-Attention to BEV Features
  |      - Deformable attention
  |      - 8 heads, 4 sampling points per head
  |      - Reference points: learned 2D positions in BEV
  |-- Residual Connection
  |
  |-- LayerNorm
  |-- Feed-Forward Network (256 -> 512 -> 256)
  |-- Residual Connection
  |
  v
Updated Object Queries (900, 256)
```

### 8.4 Iterative Refinement

- Reference points are refined at each decoder layer
- Each layer predicts a residual offset to the reference point
- This iterative refinement improves localization across layers

```python
# Iterative box refinement
for layer_idx in range(6):
    output = decoder_layer(queries, bev_features, reference_points)
    
    # Refine reference points
    delta = regression_head[layer_idx](output)  # (900, 2) offset
    reference_points = reference_points + delta.sigmoid()
    reference_points = reference_points.detach()  # Stop gradient
```

### 8.5 Decoder Parameters

| Component | Parameters per Layer |
|-----------|---------------------|
| Self-Attention | |
| - Q/K/V projections | 3 x (256 x 256) = 196,608 |
| - Output projection | 256 x 256 = 65,536 |
| Cross-Attention (deformable) | |
| - Q/K/V projections | 3 x (256 x 256) = 196,608 |
| - Sampling offsets | 8 x 4 x 2 x 256 = 16,384 |
| - Attention weights | 8 x 4 x 256 = 8,192 |
| - Output projection | 256 x 256 = 65,536 |
| FFN | |
| - Linear 1 | 256 x 512 + 512 = 131,584 |
| - Linear 2 | 512 x 256 + 256 = 131,328 |
| Layer Norms (x3) | 3 x (256 + 256) = 1,536 |
| **Total per layer** | **~0.81M** |
| **Total decoder (6 layers)** | **~4.9M** |

---

## 9. Detection Heads

### 9.1 Classification Head

```python
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=256, num_classes=10):
        self.layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 detection classes
        )
    # Output: (B, 900, 10) class logits
    # Loss: Focal Loss (alpha=0.25, gamma=2.0)
```

### 9.2 Regression Head

```python
class RegressionHead(nn.Module):
    def __init__(self, embed_dim=256, num_params=10):
        self.layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 box parameters
        )
    # Output: (B, 900, 10)
    # Parameters: [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
    # Loss: L1 Loss
```

### 9.3 Regression Parameter Details

| Parameter | Index | Range | Encoding |
|-----------|-------|-------|----------|
| cx | 0 | [-51.2, 51.2] m | Sigmoid x range |
| cy | 1 | [-51.2, 51.2] m | Sigmoid x range |
| cz | 2 | [-5.0, 3.0] m | Direct regression |
| w | 3 | [0, 20] m | Exponential (log-space) |
| l | 4 | [0, 20] m | Exponential (log-space) |
| h | 5 | [0, 10] m | Exponential (log-space) |
| sin(yaw) | 6 | [-1, 1] | Direct regression |
| cos(yaw) | 7 | [-1, 1] | Direct regression |
| vx | 8 | [-20, 20] m/s | Direct regression |
| vy | 9 | [-20, 20] m/s | Direct regression |

### 9.4 Head Parameters

| Component | Parameters |
|-----------|-----------|
| Classification head (shared across layers) | 256x256 + 256x256 + 256x10 = 133,898 |
| Regression head (shared across layers) | 256x256 + 256x256 + 256x10 = 133,898 |
| Reference point head (per decoder layer) | 6 x (256x2) = 3,072 |
| **Total head parameters** | **~271K** |

---

## 10. Parameter Count Breakdown

### 10.1 BEVFormer-Base (ResNet-101-DCN)

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Backbone (ResNet-101-DCN) | 44.5M | 63.5% |
| FPN | 3.5M | 5.0% |
| BEV Queries (learnable) | 10.2M | 14.6% |
| BEV Encoder (6 layers) | 6.0M | 8.6% |
| Detection Decoder (6 layers) | 4.9M | 7.0% |
| Detection Heads | 0.3M | 0.4% |
| Positional Encodings | 0.6M | 0.9% |
| **Total** | **~70.0M** | **100%** |

### 10.2 BEVFormer-Large (V2-99)

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Backbone (V2-99) | 52.7M | 67.3% |
| FPN | 3.5M | 4.5% |
| BEV Queries (learnable) | 10.2M | 13.0% |
| BEV Encoder (6 layers) | 6.0M | 7.7% |
| Detection Decoder (6 layers) | 4.9M | 6.3% |
| Detection Heads | 0.3M | 0.4% |
| Positional Encodings | 0.6M | 0.8% |
| **Total** | **~78.2M** | **100%** |

---

## 11. Computation Flow Diagram

### 11.1 Training Forward Pass

```
Step 1: Image Feature Extraction
=========================================
Input images: (B, 6, 3, 900, 1600)
  |
  | Reshape to (B*6, 3, 900, 1600) for batch processing
  v
Backbone(ResNet-101-DCN):
  Stage1: (B*6, 256, 225, 400)
  Stage2: (B*6, 512, 113, 200)  
  Stage3: (B*6, 1024, 57, 100)
  Stage4: (B*6, 2048, 29, 50)
  |
  v
FPN:
  Level0: (B*6, 256, 113, 200)  [1/8]
  Level1: (B*6, 256, 57, 100)   [1/16]
  Level2: (B*6, 256, 29, 50)    [1/32]
  |
  | Reshape to (B, 6, 3_levels, *, 256)
  v
Multi-scale image features ready

Step 2: BEV Encoder (iterative)
=========================================
Initialize: bev_queries = learnable_embedding  (B, 40000, 256)
Load: prev_bev (from previous frame, or copy of bev_queries if t=0)

For each encoder layer (1-6):
  |
  |-- Temporal Self-Attention:
  |     Q = LayerNorm(bev_queries)           (B, 40000, 256)
  |     K,V = concat(bev_queries, aligned_prev_bev)  (B, 80000, 256)
  |     ref_pts = bev_grid_centers            (B, 40000, 1, 2)
  |     out = DeformAttn(Q, K, V, ref_pts)   (B, 40000, 256)
  |     bev_queries = bev_queries + out       (residual)
  |
  |-- Spatial Cross-Attention:
  |     Q = LayerNorm(bev_queries)           (B, 40000, 256)
  |     K,V = multi_camera_features          (B, 6, *, 256)
  |     ref_pts_3d = bev_to_3d(bev_grid, heights)  (B, 40000, 4, 3)
  |     ref_pts_2d = project(ref_pts_3d, cam_params) (B, 40000, N_cam, 4, 2)
  |     out = SpatialCrossAttn(Q, K, V, ref_pts_2d) (B, 40000, 256)
  |     bev_queries = bev_queries + out       (residual)
  |
  |-- FFN:
  |     out = FFN(LayerNorm(bev_queries))    (B, 40000, 256)
  |     bev_queries = bev_queries + out       (residual)

Output: bev_features = bev_queries.reshape(B, 256, 200, 200)

Step 3: Detection Decoder
=========================================
Initialize: object_queries = learnable_embedding  (B, 900, 256)
            reference_points = learnable_2d_points (B, 900, 2)

For each decoder layer (1-6):
  |
  |-- Self-Attention:
  |     Q = K = V = LayerNorm(object_queries)  (B, 900, 256)
  |     out = MultiHeadAttn(Q, K, V)           (B, 900, 256)
  |     object_queries = object_queries + out
  |
  |-- Cross-Attention to BEV:
  |     Q = LayerNorm(object_queries)          (B, 900, 256)
  |     K,V = bev_features.flatten()           (B, 40000, 256)
  |     ref = reference_points                  (B, 900, 1, 2)
  |     out = DeformAttn(Q, K, V, ref)         (B, 900, 256)
  |     object_queries = object_queries + out
  |
  |-- FFN:
  |     out = FFN(LayerNorm(object_queries))   (B, 900, 256)
  |     object_queries = object_queries + out
  |
  |-- Reference Point Refinement:
  |     delta = reg_branch(object_queries)[:, :2]  (B, 900, 2)
  |     reference_points = (reference_points + delta).detach()

Step 4: Prediction
=========================================
For each decoder layer output (auxiliary losses):
  cls_scores = cls_head(object_queries)    (B, 900, 10)
  bbox_preds = reg_head(object_queries)    (B, 900, 10)

Step 5: Loss Computation
=========================================
Hungarian matching: match predicted boxes to GT
Losses:
  L_cls = FocalLoss(cls_scores, matched_labels)
  L_reg = L1Loss(bbox_preds, matched_boxes)
  L_total = L_cls + L_reg  (summed over all decoder layers)
```

### 11.2 Memory Usage Estimates (Training, batch_size=1)

| Component | GPU Memory |
|-----------|-----------|
| Image features (6 cameras, 3 scales) | ~2.5 GB |
| BEV queries + gradients | ~0.5 GB |
| BEV encoder activations | ~4.0 GB |
| Detection decoder activations | ~1.0 GB |
| Model parameters | ~0.3 GB |
| Optimizer states (AdamW) | ~0.6 GB |
| Miscellaneous (buffers, temp) | ~1.0 GB |
| **Total (approximate)** | **~10 GB** |

With mixed precision (FP16): ~6-7 GB

---

## 12. Inference Pipeline

```
Input: 6 camera images + calibration + ego_pose
  |
  v
[Backbone + FPN]: Extract multi-scale features
  |
  v
[Load previous BEV] from cache (or initialize if first frame)
  |
  v
[BEV Encoder]: Generate current BEV features
  |
  v
[Cache current BEV] for next frame
  |
  v
[Detection Decoder]: Decode object queries
  |
  v
[Detection Heads]: Predict classes and boxes
  |
  v
[Post-processing]:
  - Score thresholding (> 0.3)
  - Convert from normalized to metric coordinates
  - NMS or top-K selection (300 boxes)
  |
  v
Output: List of 3D bounding boxes with class, confidence, velocity
```

### Inference Timing (A100 GPU)

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Backbone + FPN | 45 | 42% |
| BEV Encoder | 38 | 36% |
| Detection Decoder | 15 | 14% |
| Post-processing | 8 | 8% |
| **Total** | **~106 ms** | **~9.4 FPS** |

---

## 13. Model Variants

| Variant | Backbone | BEV Size | Encoder Layers | Decoder Layers | Queries | NDS |
|---------|----------|----------|----------------|----------------|---------|-----|
| BEVFormer-Tiny | ResNet-50 | 50x50 | 3 | 6 | 300 | ~42 |
| BEVFormer-Small | ResNet-101 | 100x100 | 6 | 6 | 900 | 47.8 |
| BEVFormer-Base | ResNet-101-DCN | 200x200 | 6 | 6 | 900 | 56.9 |
| BEVFormer-Large | V2-99 | 200x200 | 6 | 6 | 900 | 59.2 |
