# MapTR: Model Architecture

## Pipeline Overview

MapTR follows an encoder-decoder architecture that transforms multi-camera images into vectorized map element predictions in a single forward pass:

```
Multi-Camera Images (6 x H x W x 3)
         │
         ▼
┌─────────────────────┐
│   Image Backbone    │  (ResNet-50 / VoVNet-99 + FPN)
│   Per-camera 2D     │
│   feature extraction│
└─────────┬───────────┘
          │  Multi-scale 2D features
          ▼
┌─────────────────────┐
│   BEV Encoder       │  (GKT: Geometry-guided Kernel Transformer)
│   Perspective → BEV │
│   feature transform │
└─────────┬───────────┘
          │  BEV feature map (C x H_bev x W_bev)
          ▼
┌─────────────────────┐
│   Map Decoder       │  (Transformer decoder with hierarchical queries)
│   Instance + Point  │
│   query refinement  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Prediction Heads  │  (Classification + Point Regression)
│   Per-instance:     │
│   - class label     │
│   - N_pts x 2 coords│
└─────────────────────┘
```

---

## Image Backbone

### Architecture

The image backbone extracts multi-scale 2D features from each camera image independently.

**ResNet-50 + FPN (Default)**:

| Component | Output Shape | Stride |
|-----------|-------------|--------|
| ResNet-50 stem | H/4 x W/4 x 64 | 4 |
| ResNet-50 layer1 (C2) | H/4 x W/4 x 256 | 4 |
| ResNet-50 layer2 (C3) | H/8 x W/8 x 512 | 8 |
| ResNet-50 layer3 (C4) | H/16 x W/16 x 1024 | 16 |
| ResNet-50 layer4 (C5) | H/32 x W/32 x 2048 | 32 |
| FPN P3 | H/8 x W/8 x 256 | 8 |
| FPN P4 | H/16 x W/16 x 256 | 16 |
| FPN P5 | H/32 x W/32 x 256 | 32 |

**Processing flow**:
1. Each of the 6 camera images is processed independently through the backbone
2. FPN produces multi-scale features at strides 8, 16, 32
3. Features from all cameras are concatenated for BEV transformation

### Alternative Backbones

| Backbone | Parameters | FLOPs | mAP | FPS |
|----------|-----------|-------|-----|-----|
| ResNet-50 | 25.6M | 4.1G per image | 50.3 | 21.8 |
| ResNet-101 | 44.5M | 7.8G per image | 52.1 | 16.4 |
| VoVNet-99 | 37.5M | 6.8G per image | 53.9 | 14.1 |
| Swin-Tiny | 28.3M | 4.5G per image | 51.7 | 18.2 |

### Input Preprocessing

```python
# Standard input configuration
input_config = {
    "image_size": (800, 480),     # W x H after resize (or 1600 x 900 for full res)
    "normalize_mean": [0.485, 0.456, 0.406],   # ImageNet statistics
    "normalize_std": [0.229, 0.224, 0.225],
    "num_cameras": 6
}
```

---

## BEV Encoder: Geometry-guided Kernel Transformer (GKT)

### Purpose

The GKT module transforms multi-camera perspective-view features into a unified Bird's Eye View (BEV) feature representation. This is a critical step that bridges the 2D image domain with the 3D spatial domain where map elements exist.

### Architecture

GKT uses camera geometry (intrinsics and extrinsics) to guide the attention-based feature lifting:

```
2D Image Features (per camera)
         │
         ▼
┌─────────────────────────────┐
│  Camera-Aware Positional    │
│  Encoding                   │
│  (encodes 3D ray direction  │
│   for each 2D pixel)        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Geometry-Guided Attention  │
│                             │
│  BEV queries attend to      │
│  image features weighted    │
│  by geometric projection    │
│  likelihood                 │
└─────────────┬───────────────┘
              │
              ▼
    BEV Feature Map (C x H_bev x W_bev)
```

### Key Components

**1. BEV Query Grid**

A learnable grid of query embeddings representing the BEV space:
- Grid size: H_bev x W_bev (typically 200 x 100 for 60m x 30m at 0.3m resolution)
- Each grid cell has a learnable embedding of dimension C = 256
- Positional encoding encodes the (x, y) meter coordinates of each BEV cell

**2. Geometry-Guided Kernels**

For each BEV query position (x, y):
1. Project the 3D point (x, y, z=0) into each camera using known calibration:
   ```
   p_cam = K @ T_ego2cam @ [x, y, 0, 1]^T
   u, v = p_cam[0:2] / p_cam[2]
   ```
2. For valid projections (within image bounds), compute attention weights based on:
   - Geometric distance between projected point and image feature positions
   - Learned offset kernels around the projection center
   - Depth uncertainty (features farther from projection center get lower weight)

**3. Cross-Attention Mechanism**

```python
# Simplified GKT cross-attention
def gkt_attention(bev_queries, img_features, proj_coords, cameras):
    """
    bev_queries: (H_bev * W_bev, C) - BEV query embeddings
    img_features: (N_cam, H_img * W_img, C) - 2D image features
    proj_coords: (H_bev * W_bev, N_cam, 2) - projected 2D coordinates
    """
    for each bev_query at position (bx, by):
        # Find which cameras see this BEV location
        visible_cams = get_visible_cameras(bx, by, cameras)
        
        for cam in visible_cams:
            # Get projected image coordinate
            u, v = proj_coords[bev_idx, cam]
            
            # Sample kernel of features around projection
            kernel_features = sample_kernel(img_features[cam], u, v, kernel_size=3)
            
            # Geometry-aware attention weights
            attn_weights = compute_attention(bev_query, kernel_features, distance_to_center)
            
            # Aggregate
            bev_feature += attn_weights @ kernel_features
    
    return bev_features  # (H_bev * W_bev, C)
```

### BEV Feature Map Output

| Property | Value |
|----------|-------|
| Spatial dimensions | 200 x 100 (H_bev x W_bev) |
| Feature channels | 256 |
| Spatial resolution | 0.3m per cell |
| Coverage | 60m x 30m |
| Encoding | Dense BEV feature grid |

---

## Map Decoder

### Overview

The Map Decoder is a transformer-based decoder that uses hierarchical queries to predict map element instances and their point-level geometry simultaneously.

### Hierarchical Query Structure

MapTR uses a two-level query hierarchy:

**Instance Queries** (M total, typically M = 50):
- Each instance query represents a potential map element
- Learnable embeddings initialized randomly
- Capture instance-level semantics (category, existence)

**Point Queries** (N_pts per instance, typically N_pts = 20):
- Each point query represents a specific point within an instance
- Shared across instances (same point queries used for all instances)
- Capture geometric position within the element

**Combined Query Construction**:
```python
# Query structure
instance_queries = nn.Embedding(M, C)          # (M, C) - learnable
point_queries = nn.Embedding(N_pts, C)         # (N_pts, C) - learnable

# Combined queries for decoder input: (M * N_pts, C)
combined_queries = instance_queries.unsqueeze(1) + point_queries.unsqueeze(0)
# Shape: (M, N_pts, C) → flatten to (M * N_pts, C)
```

### Decoder Layers

Each decoder layer consists of three attention operations:

```
┌─────────────────────────────────┐
│ Layer l                          │
│                                  │
│  1. Self-Attention               │
│     (queries interact with       │
│      each other)                 │
│                                  │
│  2. Cross-Attention              │
│     (queries attend to BEV       │
│      features)                   │
│                                  │
│  3. Feed-Forward Network         │
│     (per-query feature           │
│      transformation)             │
│                                  │
└─────────────────────────────────┘
         × L layers (L = 6)
```

**1. Self-Attention Among Queries**

In the original MapTR:
- All M x N_pts queries attend to each other
- Computational complexity: O((M * N_pts)^2) = O((50 * 20)^2) = O(1,000,000)

In MapTRv2 (decoupled self-attention):
- Instance-level self-attention: O(M^2) - instance queries interact
- Point-level self-attention: O(M * N_pts^2) - points within each instance interact
- Total: O(M^2 + M * N_pts^2) = O(2500 + 50*400) = O(22,500) — much cheaper

**2. Cross-Attention to BEV Features**

- Each query attends to the BEV feature map using deformable attention
- Reference points are initialized from learnable positions or previous layer predictions
- Deformable attention samples K points (K=4) around each reference point
- Multi-scale attention across BEV feature pyramid levels

```python
# Deformable cross-attention
def deformable_cross_attention(queries, bev_features, reference_points):
    """
    queries: (M * N_pts, C)
    bev_features: (H_bev * W_bev, C)
    reference_points: (M * N_pts, 2) - normalized BEV coordinates
    """
    # Predict sampling offsets from queries
    offsets = offset_network(queries)  # (M * N_pts, num_heads, K, 2)
    
    # Sample BEV features at reference_point + offsets
    sampling_locations = reference_points + offsets
    sampled_features = bilinear_sample(bev_features, sampling_locations)
    
    # Attention-weighted aggregation
    attention_weights = attn_network(queries)  # (M * N_pts, num_heads, K)
    output = (attention_weights * sampled_features).sum(dim=-2)
    
    return output
```

**3. Feed-Forward Network**

- Two linear layers with ReLU activation
- Hidden dimension: 2048 (expansion ratio 8x from C=256)
- Applied independently to each query

### Decoder Configuration

| Parameter | Value |
|-----------|-------|
| Number of decoder layers | 6 |
| Hidden dimension (C) | 256 |
| Number of attention heads | 8 |
| FFN hidden dimension | 2048 |
| Number of instance queries (M) | 50 |
| Number of point queries (N_pts) | 20 |
| Deformable attention points (K) | 4 |
| Dropout rate | 0.1 |

---

## Prediction Heads

### Classification Head

Predicts the category for each instance query:

```python
class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=3):
        self.fc = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object"
    
    def forward(self, instance_features):
        # instance_features: (M, C) - aggregated from point queries
        # Aggregate by mean-pooling point features per instance
        logits = self.fc(instance_features)  # (M, num_classes + 1)
        return logits
```

Categories:
- 0: Pedestrian crossing
- 1: Lane divider
- 2: Road boundary
- 3: No object (background)

### Point Regression Head

Predicts 2D coordinates for each point in each instance:

```python
class PointRegressionHead(nn.Module):
    def __init__(self, hidden_dim=256):
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # (x, y) coordinates
        )
    
    def forward(self, point_features):
        # point_features: (M, N_pts, C)
        coords = self.fc(point_features)  # (M, N_pts, 2)
        coords = coords.sigmoid()  # Normalize to [0, 1]
        return coords
```

### Iterative Refinement

Point positions are refined across decoder layers:
- Layer 0: Predict initial coordinates from learnable reference points
- Layer l: Predict coordinate **offsets** from previous layer's coordinates
- Final coordinates: Sum of reference points and all offsets

```python
# Iterative refinement across decoder layers
reference_points = initial_reference_points  # (M, N_pts, 2), learnable

for layer in decoder_layers:
    # Decoder layer updates query features
    query_features = layer(query_features, bev_features, reference_points)
    
    # Predict offset from current reference
    offset = regression_head(query_features)  # (M, N_pts, 2)
    
    # Update reference points (with gradient detach for stability)
    reference_points = (reference_points + offset).detach()
```

---

## MapTRv2 Architectural Additions

### 1. Decoupled Self-Attention

Replaces the monolithic self-attention with two separate mechanisms:

```python
class DecoupledSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, M, N_pts):
        self.instance_self_attn = nn.MultiheadAttention(d_model, nhead)
        self.point_self_attn = nn.MultiheadAttention(d_model, nhead)
    
    def forward(self, queries):
        # queries: (M, N_pts, C)
        
        # Instance-level: aggregate points, attend across instances
        instance_features = queries.mean(dim=1)  # (M, C)
        instance_features = self.instance_self_attn(
            instance_features, instance_features, instance_features
        )  # (M, C)
        
        # Point-level: attend among points within each instance
        for i in range(M):
            point_features = queries[i]  # (N_pts, C)
            queries[i] = self.point_self_attn(
                point_features, point_features, point_features
            )  # (N_pts, C)
        
        # Combine: broadcast instance features back to points
        queries = queries + instance_features.unsqueeze(1)
        
        return queries
```

**Benefits**:
- Instance self-attention captures global map structure (spatial relationships between elements)
- Point self-attention captures local geometry (smoothness, continuity within an element)
- Dramatically reduced memory and computation

### 2. Auxiliary Dense Prediction Head

An additional branch that predicts a rasterized BEV segmentation map:

```python
class AuxiliaryDenseHead(nn.Module):
    def __init__(self, bev_channels=256, num_classes=3):
        self.conv_layers = nn.Sequential(
            nn.Conv2d(bev_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, bev_features):
        # bev_features: (B, C, H_bev, W_bev)
        segmentation = self.conv_layers(bev_features)  # (B, num_classes, H_bev, W_bev)
        return segmentation
```

**Purpose**:
- Provides dense pixel-level supervision to the BEV encoder
- Ground truth: Rasterized version of vectorized map annotations
- Loss: Binary cross-entropy per class per BEV pixel
- Only used during training (removed at inference)

### 3. One-to-Many Matching Auxiliary Heads

Additional decoder heads with relaxed matching:

```python
class OneToManyHead(nn.Module):
    """
    Auxiliary head where each GT element is matched to K predictions.
    Provides denser supervision during training.
    """
    def __init__(self, K=5):
        self.K = K  # Number of predictions matched per GT
        # Same architecture as primary head
        self.cls_head = ClassificationHead()
        self.reg_head = PointRegressionHead()
    
    def forward(self, query_features):
        cls_pred = self.cls_head(query_features)
        pts_pred = self.reg_head(query_features)
        return cls_pred, pts_pred
    
    def compute_loss(self, predictions, ground_truth):
        # Top-K matching: each GT matched to K best predictions
        # Loss computed for all K matches (not just best one)
        pass
```

---

## Full Model Configuration

### MapTR (Original)

```python
model_config = {
    # Backbone
    "backbone": "ResNet-50",
    "neck": "FPN",
    "neck_out_channels": 256,
    
    # BEV Encoder
    "bev_encoder": "GKT",
    "bev_h": 200,
    "bev_w": 100,
    "bev_channels": 256,
    
    # Map Decoder
    "num_decoder_layers": 6,
    "hidden_dim": 256,
    "num_heads": 8,
    "ffn_dim": 2048,
    "num_queries": 50,        # Instance queries (M)
    "num_points": 20,          # Points per instance (N_pts)
    "num_classes": 3,
    
    # Deformable Attention
    "num_levels": 1,           # Single BEV level
    "num_sampling_points": 4,
    
    # Prediction
    "iterative_refinement": True,
    "aux_loss": True           # Loss at each decoder layer
}
```

### MapTRv2

```python
model_config_v2 = {
    **model_config,  # Inherits all MapTR config
    
    # MapTRv2 additions
    "decoupled_self_attention": True,
    "aux_dense_head": True,
    "one_to_many_matching": True,
    "one_to_many_K": 5,
    "aux_dense_loss_weight": 2.0,
    "one_to_many_loss_weight": 1.0,
}
```

---

## Model Size and Computational Cost

| Component | Parameters | FLOPs (per frame) |
|-----------|-----------|-------------------|
| ResNet-50 backbone | 25.6M | 24.6G (6 images) |
| FPN neck | 3.5M | 1.2G |
| GKT BEV encoder | 8.2M | 12.4G |
| Map Decoder (6 layers) | 12.1M | 8.3G |
| Prediction heads | 0.8M | 0.1G |
| **Total (MapTR)** | **~50.2M** | **~46.6G** |
| + MapTRv2 additions | +2.4M | +3.1G |
| **Total (MapTRv2)** | **~52.6M** | **~49.7G** |

### Inference Speed

| Model | GPU | Resolution | FPS |
|-------|-----|-----------|-----|
| MapTR (R50) | RTX 3090 | 800 x 480 | 25.1 |
| MapTRv2 (R50) | RTX 3090 | 800 x 480 | 21.8 |
| MapTR (R50) | V100 | 800 x 480 | 18.7 |
| MapTRv2 (VoV-99) | RTX 3090 | 800 x 480 | 14.1 |

---

## Data Flow Summary

```
Input:
  6 images: (6, 3, 480, 800)
  6 intrinsics: (6, 3, 3)
  6 extrinsics: (6, 4, 4)

After Backbone + FPN:
  6 x multi-scale features: (6, 256, 60, 100), (6, 256, 30, 50), (6, 256, 15, 25)

After GKT:
  BEV features: (1, 256, 200, 100)

Decoder Input:
  Queries: (50 * 20, 256) = (1000, 256)
  Reference points: (50, 20, 2)

Decoder Output:
  Updated queries: (50, 20, 256)

Predictions:
  Classification: (50, 4)        # 3 classes + background
  Point coords: (50, 20, 2)     # Normalized [0,1] coordinates
```

---

## Key Design Decisions

1. **Parallel point prediction**: All 20 points per instance predicted simultaneously (not autoregressively), enabling fast inference.

2. **Shared point queries**: The same N_pts point embeddings are reused across all instances, reducing parameters and encouraging the model to learn generalizable point-level patterns.

3. **Iterative refinement**: Progressive coordinate refinement across decoder layers improves accuracy without additional parameters in the prediction head.

4. **Deformable attention**: Enables efficient cross-attention to large BEV feature maps without quadratic memory cost.

5. **Auxiliary losses at every decoder layer**: Provides gradients throughout the decoder depth, improving training stability.
