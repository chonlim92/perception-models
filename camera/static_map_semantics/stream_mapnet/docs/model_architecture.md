# StreamMapNet: Model Architecture

## Architecture Overview

StreamMapNet follows a modular encoder-decoder architecture with temporal propagation:

```
Input: 6 surround-view cameras (3x256x704 each)
    │
    ▼
[1] Image Backbone (ResNet-50 + FPN)
    │   → Multi-scale features per camera
    ▼
[2] BEV Transform (Lift-Splat-Shoot with depth prediction)
    │   → Bird's-eye-view feature map (C x 200 x 100)
    ▼
[3] Temporal Fusion Module
    │   → Fuse current BEV with warped history
    ▼
[4] Map Decoder (Transformer with deformable attention)
    │   → Decode learnable map queries
    ▼
[5] Prediction Heads (Classification + Polyline regression)
    │   → N queries x (class_logits + K points x 2 coords)
    ▼
Output: Vectorized map elements
```

---

## Stage 1: Image Backbone

### Architecture: ResNet-50 + FPN

**Input:** 6 camera images, each of size (3, 256, 704) after resize and normalization.  
Batch dimension: (B, 6, 3, 256, 704) reshaped to (B*6, 3, 256, 704) for backbone processing.

### ResNet-50 Layers

| Layer | Output Size | Channels | Notes |
|-------|-------------|----------|-------|
| Stem (conv1 + pool) | 64 x 176 | 64 | 7x7 conv, stride 2, + 3x3 maxpool |
| Layer 1 (C2) | 64 x 176 | 256 | 3 bottleneck blocks |
| Layer 2 (C3) | 32 x 88 | 512 | 4 bottleneck blocks, stride 2 |
| Layer 3 (C4) | 16 x 44 | 1024 | 6 bottleneck blocks, stride 2 |
| Layer 4 (C5) | 8 x 22 | 2048 | 3 bottleneck blocks, stride 2 |

### Feature Pyramid Network (FPN)

FPN produces multi-scale features with uniform channel dimension (C=256):

| FPN Level | Source | Output Size | Channels |
|-----------|--------|-------------|----------|
| P2 | C2 + upsampled P3 | 64 x 176 | 256 |
| P3 | C3 + upsampled P4 | 32 x 88 | 256 |
| P4 | C4 + upsampled P5 | 16 x 44 | 256 |
| P5 | C5 | 8 x 22 | 256 |

**Used for BEV transform:** Typically P3 and P4 (or just P4 depending on configuration).

**Output per camera:** Feature maps at selected scales, e.g., (256, 16, 44) from P4.  
**Total backbone output:** (B*6, 256, 16, 44) for the selected FPN level.

---

## Stage 2: BEV Transform (Lift-Splat-Shoot)

### Overview

The Lift-Splat-Shoot (LSS) module transforms perspective camera features into a unified bird's-eye-view (BEV) representation by predicting depth distributions and projecting features into 3D space.

### Step 2a: Depth Prediction

For each pixel in the feature map, predict a categorical depth distribution over D discrete depth bins:

```
Depth network: Conv2d(256, D) where D = number of depth bins
```

| Parameter | Value |
|-----------|-------|
| Depth range | [2.0m, 50.0m] |
| Number of bins (D) | 48 |
| Bin spacing | Uniform (1.0m per bin) |
| Activation | Softmax over depth dimension |

**Input:** (B*6, 256, 16, 44) image features  
**Output:** (B*6, D, 16, 44) depth probability distribution

### Step 2b: Outer Product (Lift)

Create a depth-aware 3D feature volume by taking the outer product of image features and depth probabilities:

```python
# context_features: (B*6, C, H_feat, W_feat) = (B*6, 64, 16, 44)
# depth_probs: (B*6, D, H_feat, W_feat) = (B*6, 48, 16, 44)

# Outer product creates frustum features
frustum_features = context_features.unsqueeze(2) * depth_probs.unsqueeze(1)
# Result: (B*6, C, D, H_feat, W_feat) = (B*6, 64, 48, 16, 44)
```

Note: A separate context network reduces channels from 256 to 64 (C_context) before the outer product to manage memory.

### Step 2c: Splat to BEV Grid

Project the 3D frustum features onto the BEV plane using camera calibration:

1. **Create frustum point cloud:** For each (u, v, d) in the frustum, compute the 3D point in ego frame using camera intrinsics and extrinsics.

2. **Voxelize:** Assign each 3D point to a BEV grid cell based on its (x, y) position.

3. **Pool:** Sum all features that fall into the same BEV grid cell (pillar pooling).

```python
# BEV grid parameters
BEV_X_RANGE = [-30.0, 30.0]   # meters, 60m total
BEV_Y_RANGE = [-15.0, 15.0]   # meters, 30m total
BEV_RESOLUTION = 0.3           # meters per pixel
BEV_H = 200                    # 60.0 / 0.3 = 200 pixels (longitudinal)
BEV_W = 100                    # 30.0 / 0.3 = 100 pixels (lateral)
```

**Output:** (B, C_bev, 200, 100) where C_bev = 64

### Step 2d: BEV Encoder

Optional convolutional encoder to refine the raw BEV features:

```python
BEVEncoder = nn.Sequential(
    ResBlock(64, 128, stride=2),   # (B, 128, 100, 50)
    ResBlock(128, 256, stride=2),  # (B, 256, 50, 25)
)
# Then upsample back:
BEVNeck = nn.Sequential(
    Upsample(256, 128),            # (B, 128, 100, 50)
    Upsample(128, 256),            # (B, 256, 200, 100)
)
```

**Final BEV features:** (B, 256, 200, 100)

---

## Stage 3: Temporal Fusion Module

### Overview

The temporal fusion module incorporates information from previous frames by:
1. Warping the previous frame's hidden state to the current coordinate frame using ego-motion
2. Fusing the warped history with current BEV features

### Step 3a: Ego-Motion Warping

Given the relative pose transformation from frame t-1 to frame t:

```python
def generate_warp_grid(ego_motion_matrix, bev_h, bev_w, bev_range):
    """
    Generate sampling grid for warping previous BEV to current frame.
    
    Args:
        ego_motion_matrix: (B, 4, 4) transformation from current to previous
        bev_h, bev_w: spatial dimensions of BEV grid (200, 100)
        bev_range: [x_min, y_min, x_max, y_max] = [-30, -15, 30, 15]
    
    Returns:
        grid: (B, bev_h, bev_w, 2) normalized sampling coordinates
    """
    # Create BEV coordinate meshgrid
    x = torch.linspace(bev_range[0], bev_range[2], bev_w)  # [-30, 30]
    y = torch.linspace(bev_range[1], bev_range[3], bev_h)  # [-15, 15]
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Current frame BEV points (homogeneous)
    ones = torch.ones_like(xx)
    zeros = torch.zeros_like(xx)
    points_curr = torch.stack([xx, yy, zeros, ones], dim=-1)  # (H, W, 4)
    
    # Transform to previous frame coordinates
    points_prev = torch.einsum('bij,hwj->bhwi', ego_motion_matrix, points_curr)
    
    # Normalize to [-1, 1] for grid_sample
    grid_x = (points_prev[..., 0] - bev_range[0]) / (bev_range[2] - bev_range[0]) * 2 - 1
    grid_y = (points_prev[..., 1] - bev_range[1]) / (bev_range[3] - bev_range[1]) * 2 - 1
    
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)
    return grid
```

```python
# Apply warping
warp_grid = generate_warp_grid(ego_motion, BEV_H, BEV_W, BEV_RANGE)
H_warped = F.grid_sample(
    H_prev,           # (B, C, 200, 100) previous hidden state
    warp_grid,        # (B, 200, 100, 2) sampling coordinates
    mode='bilinear',
    padding_mode='zeros',  # Out-of-range areas get zero features
    align_corners=True
)
# H_warped: (B, C, 200, 100) - previous features aligned to current frame
```

### Step 3b: Temporal Attention Fusion

The primary fusion mechanism uses cross-attention:

```python
class TemporalFusionModule(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, bev_current, h_warped):
        """
        Args:
            bev_current: (B, C, H, W) current BEV features
            h_warped: (B, C, H, W) warped previous hidden state
        
        Returns:
            h_new: (B, C, H, W) fused hidden state
        """
        B, C, H, W = bev_current.shape
        
        # Flatten spatial dims for attention
        q = bev_current.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        kv = h_warped.flatten(2).permute(0, 2, 1)    # (B, HW, C)
        
        # Cross-attention: current queries attend to history
        attn_out, _ = self.cross_attn(q, kv, kv)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        
        # Gated fusion with current BEV
        bev_flat = bev_current.flatten(2).permute(0, 2, 1)
        gate_input = torch.cat([q, bev_flat], dim=-1)
        gate_weight = self.gate(gate_input)  # (B, HW, C)
        
        h_new = gate_weight * q + (1 - gate_weight) * bev_flat
        h_new = h_new.permute(0, 2, 1).reshape(B, C, H, W)
        
        return h_new
```

### Tensor Dimensions Through Temporal Fusion

| Tensor | Shape | Description |
|--------|-------|-------------|
| BEV_current | (B, 256, 200, 100) | Current frame BEV features |
| H_prev | (B, 256, 200, 100) | Previous hidden state (stored) |
| ego_motion | (B, 4, 4) | Current-to-previous transform |
| warp_grid | (B, 200, 100, 2) | Sampling grid |
| H_warped | (B, 256, 200, 100) | Warped previous state |
| H_new | (B, 256, 200, 100) | Fused hidden state (output) |

---

## Stage 4: Map Decoder (Transformer)

### Architecture: DETR-style Decoder with Deformable Attention

The map decoder uses learnable queries to detect and localize map elements in the BEV feature map.

### Learnable Map Queries

```python
# Query initialization
num_queries = 50        # Number of map element queries (adjustable, typically 50-120)
query_dim = 256         # Query embedding dimension
num_points = 20         # Points per polyline (K)

map_queries = nn.Embedding(num_queries, query_dim)  # Learnable query embeddings
reference_points = nn.Linear(query_dim, num_points * 2)  # Initial reference points
```

### Decoder Layer Structure

Each decoder layer contains:

```python
class MapDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_levels=1, num_points_attn=4):
        super().__init__()
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Deformable cross-attention to BEV features
        self.cross_attn = DeformableAttention(
            d_model=d_model,
            n_heads=nhead,
            n_levels=num_levels,
            n_points=num_points_attn,  # sampling points per attention head
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, queries, bev_features, reference_points):
        """
        Args:
            queries: (B, N_queries, C) = (B, 50, 256)
            bev_features: (B, C, H, W) = (B, 256, 200, 100)
            reference_points: (B, N_queries, K, 2) = (B, 50, 20, 2)
        """
        # Self-attention
        q = self.norm1(queries + self.self_attn(queries, queries, queries)[0])
        
        # Deformable cross-attention to BEV
        q = self.norm2(q + self.cross_attn(q, bev_features, reference_points))
        
        # FFN
        q = self.norm3(q + self.ffn(q))
        
        return q
```

### Deformable Attention Details

Deformable attention samples a small set of key positions around reference points instead of attending to the full BEV feature map:

```python
class DeformableAttention(nn.Module):
    """
    Multi-scale deformable attention for efficient BEV feature sampling.
    Instead of attending to all 200x100=20000 BEV positions,
    samples only n_points positions per query per head.
    """
    def __init__(self, d_model=256, n_heads=8, n_levels=1, n_points=4):
        super().__init__()
        self.n_heads = n_heads
        self.n_points = n_points
        self.n_levels = n_levels
        
        # Predict sampling offsets: each head samples n_points per reference point
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
```

### Decoder Stack Configuration

| Parameter | Value |
|-----------|-------|
| Number of decoder layers | 6 |
| Hidden dimension | 256 |
| Number of attention heads | 8 |
| Deformable attention points | 4 per head |
| FFN dimension | 1024 |
| Dropout | 0.1 |

### Reference Point Iterative Refinement

After each decoder layer, reference points are refined:

```python
# In the decoder forward pass:
for layer_idx, decoder_layer in enumerate(self.decoder_layers):
    queries = decoder_layer(queries, bev_features, reference_points)
    
    # Refine reference points
    delta_points = self.point_refinement[layer_idx](queries)  # (B, N, K*2)
    delta_points = delta_points.reshape(B, N, K, 2)
    reference_points = reference_points + delta_points.sigmoid() * 0.1  # Small refinement
```

---

## Stage 5: Prediction Heads

### Classification Head

```python
class ClassificationHead(nn.Module):
    def __init__(self, d_model=256, num_classes=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )
    
    def forward(self, queries):
        """
        Args:
            queries: (B, N_queries, C) = (B, 50, 256)
        Returns:
            class_logits: (B, N_queries, num_classes) = (B, 50, 3)
        """
        return self.fc(queries)
```

### Polyline Regression Head

```python
class PolylineHead(nn.Module):
    def __init__(self, d_model=256, num_points=20):
        super().__init__()
        self.num_points = num_points
        self.reg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_points * 2),  # K points x 2 coordinates
        )
    
    def forward(self, queries):
        """
        Args:
            queries: (B, N_queries, C) = (B, 50, 256)
        Returns:
            points: (B, N_queries, K, 2) = (B, 50, 20, 2) in normalized coords
        """
        output = self.reg(queries)  # (B, 50, 40)
        output = output.reshape(-1, self.num_points, 2)  # (B*50, 20, 2)
        output = output.sigmoid()  # Normalize to [0, 1]
        return output.reshape(-1, queries.shape[1], self.num_points, 2)
```

### Final Output Format

```python
# Model output dictionary
output = {
    'class_logits': (B, N_queries, num_classes),  # (B, 50, 3)
    'pred_points': (B, N_queries, K, 2),          # (B, 50, 20, 2) normalized [0,1]
    # Auxiliary outputs from intermediate decoder layers (for deep supervision)
    'aux_outputs': [
        {'class_logits': ..., 'pred_points': ...},  # Layer 0
        {'class_logits': ..., 'pred_points': ...},  # Layer 1
        ...                                          # Layers 2-4
    ]
}
```

---

## Complete Tensor Flow Summary

| Stage | Input Shape | Output Shape | Module |
|-------|-------------|--------------|--------|
| Camera input | (B, 6, 3, 256, 704) | - | - |
| Reshape for backbone | (B*6, 3, 256, 704) | (B*6, 256, 16, 44) | ResNet-50 + FPN (P4) |
| Depth prediction | (B*6, 256, 16, 44) | (B*6, 48, 16, 44) | Conv2d + Softmax |
| Context features | (B*6, 256, 16, 44) | (B*6, 64, 16, 44) | Conv2d (channel reduction) |
| Frustum features (Lift) | (B*6, 64, 16, 44) + depth | (B*6, 64, 48, 16, 44) | Outer product |
| Splat to BEV | (B*6, 64, 48, 16, 44) | (B, 64, 200, 100) | Voxel pooling |
| BEV encoder | (B, 64, 200, 100) | (B, 256, 200, 100) | ResNet blocks + neck |
| Temporal warp | (B, 256, 200, 100) + pose | (B, 256, 200, 100) | grid_sample |
| Temporal fusion | (B, 256, 200, 100) x2 | (B, 256, 200, 100) | Cross-attention + gate |
| Map decoder | (B, 50, 256) queries + BEV | (B, 50, 256) | 6x Decoder layers |
| Classification | (B, 50, 256) | (B, 50, 3) | MLP head |
| Point regression | (B, 50, 256) | (B, 50, 20, 2) | MLP head + sigmoid |

---

## Model Variants

### Backbone Options

| Backbone | Params | FLOPs | mAP (nuScenes) |
|----------|--------|-------|-----------------|
| ResNet-18 | 11.7M | 1.8G | 48.2 |
| ResNet-50 | 25.6M | 4.1G | 54.1 |
| ResNet-101 | 44.5M | 7.8G | 55.8 |
| Swin-Tiny | 28.3M | 4.5G | 56.2 |

### BEV Resolution Options

| Resolution | Grid Size | Memory | mAP |
|-----------|-----------|--------|-----|
| 0.15 m/px | 400 x 200 | ~4.2 GB | 55.3 |
| 0.30 m/px | 200 x 100 | ~1.1 GB | 54.1 |
| 0.60 m/px | 100 x 50 | ~0.3 GB | 50.8 |

### Number of Queries

| N_queries | mAP | Recall | Notes |
|-----------|-----|--------|-------|
| 30 | 52.1 | 78.3% | May miss elements in complex scenes |
| 50 | 54.1 | 85.7% | Default setting |
| 100 | 54.3 | 89.1% | Slightly better recall, more compute |
| 150 | 54.2 | 89.8% | Diminishing returns |

---

## Memory and Compute Budget

### Per-frame Inference (B=1, ResNet-50, 200x100 BEV)

| Component | FLOPs | Memory | Time (A100) |
|-----------|-------|--------|-------------|
| Backbone (6 images) | 24.6G | 1.2 GB | 12 ms |
| LSS depth + splat | 8.2G | 2.1 GB | 8 ms |
| BEV encoder | 4.8G | 0.6 GB | 4 ms |
| Temporal fusion | 0.8G | 0.2 GB | 2 ms |
| Map decoder (6 layers) | 3.2G | 0.3 GB | 6 ms |
| Prediction heads | 0.1G | <0.1 GB | <1 ms |
| **Total** | **41.7G** | **4.4 GB** | **~33 ms** |

**Inference speed:** ~30 FPS on NVIDIA A100, ~15 FPS on NVIDIA RTX 3090
