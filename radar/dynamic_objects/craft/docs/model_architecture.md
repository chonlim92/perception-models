# CRAFT: Model Architecture

## Dual-Branch Architecture with Spatio-Contextual Fusion Transformer

---

## 1. Architecture Overview

CRAFT employs a dual-branch architecture that processes camera and radar data independently before fusing them through a Spatio-Contextual Fusion Transformer (SCFT). This design preserves modality-specific feature representations while enabling cross-modal interaction.

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CRAFT Architecture                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────┐                                      │
│  │         CAMERA BRANCH             │                                      │
│  │                                   │                                      │
│  │  6x Images ──► Backbone ──► FPN ──┼──► Image Features                   │
│  │  (1600x900)   (ResNet-50)         │    (C x H x W per view)             │
│  └───────────────────────────────────┘                                      │
│                                          │                                  │
│                                          ▼                                  │
│                              ┌────────────────────────┐                     │
│                              │  Spatio-Contextual     │                     │
│                              │  Fusion Transformer    │──► Fused            │
│                              │  (SCFT)                │    Features          │
│                              └────────────────────────┘                     │
│                                          ▲                                  │
│                                          │                                  │
│  ┌───────────────────────────────────┐                                      │
│  │         RADAR BRANCH              │                                      │
│  │                                   │                                      │
│  │  Radar Pts ──► Pillar ──► Sparse ─┼──► BEV Features                     │
│  │  (N x 18)     Encode    Conv      │    (C x X x Y)                      │
│  └───────────────────────────────────┘                                      │
│                                                                             │
│                                          │                                  │
│                                          ▼                                  │
│                              ┌────────────────────────┐                     │
│                              │   Detection Head       │                     │
│                              │   (Anchor-Free)        │──► 3D Boxes         │
│                              └────────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Camera Branch

### 2.1 Input Specification

| Parameter | Value |
|-----------|-------|
| Number of views | 6 (surround cameras) |
| Input resolution | 1600 x 900 (original) |
| Training resolution | 704 x 256 or 800 x 448 (resized) |
| Color space | RGB, normalized to ImageNet statistics |
| Input tensor shape | `(B, 6, 3, H, W)` |

### 2.2 Backbone: ResNet-50 / EfficientNet-B4

The camera branch uses a standard image classification backbone pre-trained on ImageNet:

**ResNet-50 Configuration (default):**

| Layer | Output Shape | Details |
|-------|-------------|---------|
| Input | (B*6, 3, 256, 704) | Batched multi-view images |
| Conv1 | (B*6, 64, 128, 352) | 7x7, stride 2, BN, ReLU |
| MaxPool | (B*6, 64, 64, 176) | 3x3, stride 2 |
| Layer1 (C2) | (B*6, 256, 64, 176) | 3x Bottleneck blocks |
| Layer2 (C3) | (B*6, 512, 32, 88) | 4x Bottleneck blocks |
| Layer3 (C4) | (B*6, 1024, 16, 44) | 6x Bottleneck blocks |
| Layer4 (C5) | (B*6, 2048, 8, 22) | 3x Bottleneck blocks |

**EfficientNet-B4 Configuration (alternative):**

| Layer | Output Shape | Details |
|-------|-------------|---------|
| Input | (B*6, 3, 256, 704) | Batched multi-view images |
| Stage 1 | (B*6, 24, 128, 352) | MBConv1, k3x3 |
| Stage 2 | (B*6, 32, 64, 176) | MBConv6, k3x3 |
| Stage 3 | (B*6, 56, 32, 88) | MBConv6, k5x5 |
| Stage 4 | (B*6, 112, 16, 44) | MBConv6, k3x3 |
| Stage 5 | (B*6, 160, 16, 44) | MBConv6, k5x5 |
| Stage 6 | (B*6, 272, 8, 22) | MBConv6, k5x5 |
| Stage 7 | (B*6, 448, 8, 22) | MBConv6, k3x3 |

### 2.3 Feature Pyramid Network (FPN)

The FPN aggregates multi-scale features from the backbone to produce rich representations at multiple resolutions:

```python
class CameraFPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale camera feature extraction.
    
    Takes features from backbone stages C3, C4, C5 and produces
    multi-scale feature maps P3, P4, P5.
    """
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=256):
        super().__init__()
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_c5 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral_c3 = nn.Conv2d(in_channels[0], out_channels, 1)
        
        # Top-down pathway (3x3 conv after upsampling + addition)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
    def forward(self, c3, c4, c5):
        # Top-down pathway
        p5 = self.lateral_c5(c5)                           # (B*6, 256, 8, 22)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2)  # (B*6, 256, 16, 44)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2)  # (B*6, 256, 32, 88)
        
        # Smooth
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        
        return p3, p4, p5
```

**FPN Output Dimensions:**

| Level | Spatial Size | Channels | Stride | Receptive Field |
|-------|-------------|----------|--------|-----------------|
| P3 | 32 x 88 | 256 | 8x | Medium objects |
| P4 | 16 x 44 | 256 | 16x | Large objects |
| P5 | 8 x 22 | 256 | 32x | Very large objects |

### 2.4 Camera Feature Summary

The camera branch outputs multi-scale feature maps for each of the 6 views:

```
Output: {
    "P3": (B, 6, 256, 32, 88),   # High-resolution, fine details
    "P4": (B, 6, 256, 16, 44),   # Medium-resolution
    "P5": (B, 6, 256, 8, 22),    # Low-resolution, large receptive field
}
```

---

## 3. Radar Branch

### 3.1 Input Specification

| Parameter | Value |
|-----------|-------|
| Input points per frame | ~100-600 (after sweep accumulation) |
| Features per point | 18 (x, y, z, vx, vy, vx_comp, vy_comp, RCS, ...) |
| Used features | 5-7 (x, y, z, vx_comp, vy_comp, RCS, timestamp_offset) |
| BEV grid range | [-51.2, 51.2] x [-51.2, 51.2] meters |
| BEV grid resolution | 0.2 m per pixel (or 0.4 m) |
| BEV grid size | 512 x 512 (at 0.2m) or 256 x 256 (at 0.4m) |
| Number of sweeps | 3-6 (accumulated) |

### 3.2 Pillar Encoding

The pillar encoding converts sparse radar points into a pseudo-image (BEV) representation:

```python
class RadarPillarEncoder(nn.Module):
    """
    PointPillars-style encoding for radar point cloud.
    
    Converts sparse 3D points into a dense 2D BEV pseudo-image
    by grouping points into vertical pillars and encoding with PointNet.
    """
    def __init__(self, 
                 in_channels=7,          # [x, y, z, vx, vy, RCS, dt]
                 pillar_channels=64,
                 max_points_per_pillar=20,
                 max_pillars=10000,
                 grid_size=(512, 512),
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        super().__init__()
        
        # Augmented features: original + (x_c, y_c, z_c, x_p, y_p)
        augmented_channels = in_channels + 5
        
        # PointNet-style MLP for pillar feature extraction
        self.pfn = nn.Sequential(
            nn.Linear(augmented_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, pillar_channels),
            nn.BatchNorm1d(pillar_channels),
            nn.ReLU(),
        )
        
        self.max_points = max_points_per_pillar
        self.max_pillars = max_pillars
        self.grid_size = grid_size
        self.pc_range = point_cloud_range
    
    def forward(self, radar_points, coords):
        """
        Args:
            radar_points: (N_total, max_points_per_pillar, in_channels+5)
            coords: (N_total, 3) - pillar grid coordinates [batch, x, y]
        
        Returns:
            bev_features: (B, pillar_channels, grid_H, grid_W)
        """
        # Apply PointNet to each pillar
        features = self.pfn(radar_points)  # (N_total, max_points, 64)
        
        # Max pooling across points in each pillar
        features = features.max(dim=1)[0]  # (N_total, 64)
        
        # Scatter to BEV grid
        bev = self.scatter_to_bev(features, coords)  # (B, 64, 512, 512)
        
        return bev
```

**Pillar Feature Augmentation:**

For each point in a pillar, the following features are computed:
- Original features: `[x, y, z, vx_comp, vy_comp, RCS, timestamp_offset]`
- Offset from pillar center: `[x - x_c, y - y_c, z - z_c]`
- Offset from pillar position: `[x - x_p, y - y_p]`

Total: 7 + 5 = 12 features per point

### 3.3 Sparse Convolutional Network

After pillar encoding, sparse convolutions process the BEV representation efficiently:

```python
class RadarSparseBackbone(nn.Module):
    """
    Sparse convolutional backbone for radar BEV features.
    
    Uses submanifold sparse convolutions for efficiency (most pillars are empty).
    """
    def __init__(self, in_channels=64, layer_nums=[3, 5, 5]):
        super().__init__()
        
        # Block 1: 512x512 -> 256x256
        self.block1 = nn.Sequential(
            SparseConv2d(64, 64, 3, stride=2, padding=1),   # Downsample
            SparseBatchNorm(64),
            nn.ReLU(),
            *[SubmanifoldSparseConv2d(64, 64, 3, padding=1) 
              for _ in range(layer_nums[0] - 1)],
        )
        
        # Block 2: 256x256 -> 128x128
        self.block2 = nn.Sequential(
            SparseConv2d(64, 128, 3, stride=2, padding=1),  # Downsample
            SparseBatchNorm(128),
            nn.ReLU(),
            *[SubmanifoldSparseConv2d(128, 128, 3, padding=1)
              for _ in range(layer_nums[1] - 1)],
        )
        
        # Block 3: 128x128 -> 64x64 (or keep at 128x128)
        self.block3 = nn.Sequential(
            SparseConv2d(128, 256, 3, stride=2, padding=1), # Downsample
            SparseBatchNorm(256),
            nn.ReLU(),
            *[SubmanifoldSparseConv2d(256, 256, 3, padding=1)
              for _ in range(layer_nums[2] - 1)],
        )
        
        # Convert sparse to dense for downstream processing
        self.to_dense = SparseToDense()
    
    def forward(self, x):
        """
        Args:
            x: Sparse BEV tensor (B, 64, 512, 512)
        
        Returns:
            multi_scale_features: List of BEV feature maps at different scales
        """
        x1 = self.block1(x)    # (B, 64, 256, 256)
        x2 = self.block2(x1)   # (B, 128, 128, 128)
        x3 = self.block3(x2)   # (B, 256, 64, 64)
        
        # Convert to dense tensors
        d1 = self.to_dense(x1)  # (B, 64, 256, 256)
        d2 = self.to_dense(x2)  # (B, 128, 128, 128)
        d3 = self.to_dense(x3)  # (B, 256, 64, 64)
        
        return [d1, d2, d3]
```

### 3.4 Radar BEV Neck

A secondary processing stage aggregates multi-scale radar BEV features:

```python
class RadarBEVNeck(nn.Module):
    """
    Neck network to fuse multi-scale radar BEV features into
    a unified BEV representation.
    """
    def __init__(self, in_channels=[64, 128, 256], out_channels=256):
        super().__init__()
        
        # Upsample lower-resolution features
        self.up_block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Fusion conv
        total_channels = 64 + 128 + 256  # = 448
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, features):
        """
        Args:
            features: [d1 (256x256), d2 (128x128), d3 (64x64)]
        
        Returns:
            bev_features: (B, 256, 256, 256) unified BEV features
        """
        d1, d2, d3 = features
        
        d2_up = self.up_block2(d2)   # (B, 128, 256, 256)
        d3_up = self.up_block3(d3)   # (B, 256, 256, 256)
        
        fused = torch.cat([d1, d2_up, d3_up], dim=1)  # (B, 448, 256, 256)
        bev = self.fusion_conv(fused)                   # (B, 256, 256, 256)
        
        return bev
```

**Radar Branch Output:**

```
Output: (B, 256, 256, 256)  # BEV feature map
# Each pixel covers 0.4m x 0.4m ground area
# Total coverage: 102.4m x 102.4m around ego vehicle
```

---

## 4. Spatio-Contextual Fusion Transformer (SCFT)

### 4.1 Fusion Strategy Overview

The SCFT is the core innovation of CRAFT. It fuses radar BEV features with camera image features through a transformer-based cross-attention mechanism that respects the spatial geometry between modalities.

```
┌─────────────────────────────────────────────────────────────┐
│              Spatio-Contextual Fusion Transformer            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Radar BEV Features ──► Query Generation ──┐               │
│  (B, 256, 256, 256)    (Linear + PE)       │               │
│                                            ▼               │
│                                    ┌──────────────┐        │
│                                    │   Cross-     │        │
│  Camera Features ──► Key/Value ───►│   Attention  │──► Fused │
│  (B, 6, 256, H, W)   Generation   │   Layers     │   Features│
│                       (Linear + PE)└──────────────┘        │
│                                            │               │
│                                            ▼               │
│                                    ┌──────────────┐        │
│                                    │    Feed-     │        │
│                                    │   Forward    │──► Output│
│                                    │    Network   │        │
│                                    └──────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Radar-to-Image Projection

The spatial alignment between radar BEV and camera images is established through geometric projection:

```python
class RadarToImageProjection(nn.Module):
    """
    Projects radar BEV grid positions onto image planes of all cameras.
    This establishes the geometric correspondence for cross-attention.
    """
    def __init__(self, bev_size=(256, 256), 
                 bev_range=(-51.2, 51.2, -51.2, 51.2),
                 image_size=(256, 704)):
        super().__init__()
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.image_size = image_size
    
    def forward(self, camera_intrinsics, camera_extrinsics):
        """
        Compute projection matrices from BEV grid to each camera.
        
        Args:
            camera_intrinsics: (B, 6, 3, 3)
            camera_extrinsics: (B, 6, 4, 4) - camera-to-ego transforms
        
        Returns:
            reference_points: (B, 6, H_bev*W_bev, 2) - projected 2D coords
            valid_mask: (B, 6, H_bev*W_bev) - visibility mask per camera
        """
        B = camera_intrinsics.shape[0]
        H, W = self.bev_size
        
        # Generate BEV grid coordinates (3D world positions at ground level)
        x = torch.linspace(self.bev_range[0], self.bev_range[1], W)
        y = torch.linspace(self.bev_range[2], self.bev_range[3], H)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid_z = torch.zeros_like(grid_x)  # Ground plane
        
        # World points: (H*W, 3)
        world_points = torch.stack([grid_x.flatten(), 
                                     grid_y.flatten(), 
                                     grid_z.flatten()], dim=-1)
        
        reference_points = []
        valid_masks = []
        
        for cam_idx in range(6):
            # Get camera parameters
            K = camera_intrinsics[:, cam_idx]          # (B, 3, 3)
            E = camera_extrinsics[:, cam_idx]          # (B, 4, 4)
            
            # Ego to camera transform
            R = E[:, :3, :3]                           # (B, 3, 3)
            t = E[:, :3, 3]                            # (B, 3)
            
            # Project: p_cam = R^T * (p_world - t)
            points_cam = torch.einsum('bij,nj->bni', R.transpose(-1,-2), 
                                       world_points.unsqueeze(0) - t.unsqueeze(1))
            
            # Image projection: p_img = K * p_cam
            points_img = torch.einsum('bij,bnj->bni', K, points_cam)
            
            # Normalize by depth
            depth = points_img[..., 2:3]
            uv = points_img[..., :2] / (depth + 1e-8)
            
            # Validity check
            valid = (depth.squeeze(-1) > 0.1) & \
                    (uv[..., 0] >= 0) & (uv[..., 0] < self.image_size[1]) & \
                    (uv[..., 1] >= 0) & (uv[..., 1] < self.image_size[0])
            
            # Normalize to [0, 1] for grid sampling
            uv_norm = uv.clone()
            uv_norm[..., 0] /= self.image_size[1]
            uv_norm[..., 1] /= self.image_size[0]
            
            reference_points.append(uv_norm)
            valid_masks.append(valid)
        
        return torch.stack(reference_points, dim=1), \
               torch.stack(valid_masks, dim=1)
```

### 4.3 Cross-Attention Mechanism

```python
class SpatioContextualCrossAttention(nn.Module):
    """
    Cross-attention between radar BEV queries and camera image key/values.
    
    Uses deformable attention with projected reference points for
    efficient and geometrically-aware feature fusion.
    """
    def __init__(self, 
                 embed_dim=256,
                 num_heads=8,
                 num_levels=3,        # Multi-scale image features
                 num_points=4,        # Deformable attention sampling points
                 num_cameras=6,
                 dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_cameras = num_cameras
        
        # Query projection
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        # Sampling offsets for deformable attention
        self.sampling_offsets = nn.Linear(
            embed_dim, 
            num_heads * num_levels * num_cameras * num_points * 2
        )
        
        # Attention weights
        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_cameras * num_points
        )
        
        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
    
    def forward(self, query, image_features, reference_points, valid_mask):
        """
        Args:
            query: (B, N_q, C) - radar BEV features flattened
            image_features: List[(B, 6, C, H_l, W_l)] - multi-scale camera features
            reference_points: (B, 6, N_q, 2) - projected BEV positions on images
            valid_mask: (B, 6, N_q) - visibility mask
        
        Returns:
            output: (B, N_q, C) - fused features
        """
        B, N_q, C = query.shape
        
        # Project queries
        query = self.query_proj(query)
        
        # Compute sampling offsets around reference points
        offsets = self.sampling_offsets(query)  # (B, N_q, H*L*cam*P*2)
        offsets = offsets.view(B, N_q, self.num_heads, self.num_levels, 
                              self.num_cameras, self.num_points, 2)
        
        # Compute attention weights
        weights = self.attention_weights(query)  # (B, N_q, H*L*cam*P)
        weights = weights.view(B, N_q, self.num_heads, self.num_levels,
                              self.num_cameras, self.num_points)
        weights = F.softmax(weights, dim=-1)
        
        # Apply validity mask (zero out weights for invisible reference points)
        # valid_mask: (B, 6, N_q) -> expand to match weights shape
        
        # Sample features from image feature maps at offset positions
        # (Deformable attention sampling)
        sampled_features = self._sample_features(
            image_features, reference_points, offsets
        )
        
        # Weighted sum of sampled features
        output = torch.einsum('bnhlcp,bnhlcpd->bnd', weights, sampled_features)
        
        # Output projection
        output = self.output_proj(output)
        output = self.dropout(output)
        
        return output
```

### 4.4 Fusion Transformer Layers

The complete SCFT consists of multiple transformer decoder layers:

```python
class SCFTLayer(nn.Module):
    """Single layer of the Spatio-Contextual Fusion Transformer."""
    
    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=512, dropout=0.1):
        super().__init__()
        
        # Self-attention on BEV queries
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        
        # Cross-attention: radar queries attend to camera features
        self.cross_attn = SpatioContextualCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=3,
            num_points=4,
            num_cameras=6,
            dropout=dropout
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, query, image_features, reference_points, valid_mask):
        """
        Args:
            query: (B, N_q, C) - BEV query features
            image_features: Multi-scale camera features
            reference_points: Projected BEV-to-image coordinates
            valid_mask: Camera visibility mask
        
        Returns:
            query: (B, N_q, C) - updated BEV features
        """
        # Self-attention (BEV spatial reasoning)
        q = self.self_attn_norm(query)
        q2 = self.self_attn(q, q, q)[0]
        query = query + q2
        
        # Cross-attention (camera feature fusion)
        q = self.cross_attn_norm(query)
        q2 = self.cross_attn(q, image_features, reference_points, valid_mask)
        query = query + q2
        
        # FFN
        q = self.ffn_norm(query)
        query = query + self.ffn(q)
        
        return query


class SCFT(nn.Module):
    """
    Complete Spatio-Contextual Fusion Transformer.
    
    Stacks multiple SCFTLayers to progressively refine the fusion
    between radar BEV and camera image features.
    """
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6, 
                 ffn_dim=512, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            SCFTLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # BEV positional encoding
        self.bev_pos_encoding = LearnableBEVPositionalEncoding(
            embed_dim=embed_dim, 
            bev_h=256, 
            bev_w=256
        )
    
    def forward(self, radar_bev, image_features, reference_points, valid_mask):
        """
        Args:
            radar_bev: (B, C, H, W) - Radar BEV feature map
            image_features: Multi-scale camera features
            reference_points: (B, 6, H*W, 2)
            valid_mask: (B, 6, H*W)
        
        Returns:
            fused_bev: (B, C, H, W) - Fused BEV feature map
        """
        B, C, H, W = radar_bev.shape
        
        # Flatten BEV to sequence
        query = radar_bev.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        
        # Add BEV positional encoding
        query = query + self.bev_pos_encoding()
        
        # Apply transformer layers
        for layer in self.layers:
            query = layer(query, image_features, reference_points, valid_mask)
        
        # Reshape back to BEV
        fused_bev = query.permute(0, 2, 1).view(B, C, H, W)
        
        return fused_bev
```

### 4.5 SCFT Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Embedding dimension | 256 | Feature channel width |
| Number of attention heads | 8 | Head dimension = 32 |
| Number of transformer layers | 6 | Depth of fusion |
| FFN hidden dimension | 512 | 2x expansion ratio |
| Number of deformable points | 4 | Sampling points per head |
| Dropout rate | 0.1 | Regularization |
| Number of multi-scale levels | 3 | P3, P4, P5 from camera FPN |
| BEV query resolution | 256 x 256 | 65,536 query positions |

---

## 5. Detection Head

### 5.1 Anchor-Free 3D Box Prediction

CRAFT uses an anchor-free detection head that predicts 3D bounding boxes directly from the fused BEV features:

```python
class CRAFTDetectionHead(nn.Module):
    """
    Anchor-free 3D object detection head.
    
    Predicts per-pixel:
    - Heatmap (classification)
    - 3D bounding box parameters
    - Velocity
    - Attribute (optional)
    """
    def __init__(self, in_channels=256, num_classes=10, 
                 num_conv_layers=2, common_heads=True):
        super().__init__()
        
        # Shared convolutional trunk
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Task-specific heads
        # 1. Heatmap head (classification)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1),
        )
        
        # 2. Center offset (sub-pixel refinement)
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1),  # (dx, dy) offset
        )
        
        # 3. Height head (z-center and height)
        self.height_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1),  # (z_center, height)
        )
        
        # 4. Size head (width, length)
        self.size_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1),  # (log_w, log_l)
        )
        
        # 5. Rotation head (sin, cos of yaw)
        self.rotation_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1),  # (sin(yaw), cos(yaw))
        )
        
        # 6. Velocity head
        self.velocity_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 2, 1),  # (vx, vy)
        )
    
    def forward(self, fused_bev):
        """
        Args:
            fused_bev: (B, 256, 256, 256) - Fused BEV features
        
        Returns:
            predictions: Dict of prediction tensors
        """
        # Shared feature processing
        shared = self.shared_conv(fused_bev)  # (B, 256, 256, 256)
        
        # Task-specific predictions
        predictions = {
            'heatmap': torch.sigmoid(self.heatmap_head(shared)),  # (B, 10, 256, 256)
            'offset': self.offset_head(shared),                    # (B, 2, 256, 256)
            'height': self.height_head(shared),                    # (B, 2, 256, 256)
            'size': self.size_head(shared),                        # (B, 2, 256, 256)
            'rotation': self.rotation_head(shared),                # (B, 2, 256, 256)
            'velocity': self.velocity_head(shared),                # (B, 2, 256, 256)
        }
        
        return predictions
```

### 5.2 Detection Decoding

```python
def decode_predictions(predictions, bev_range, score_threshold=0.1, nms_threshold=0.2):
    """
    Decode network predictions into 3D bounding boxes.
    
    Args:
        predictions: Dict of prediction tensors from detection head
        bev_range: (x_min, x_max, y_min, y_max) meters
        score_threshold: Minimum confidence for detection
        nms_threshold: IoU threshold for BEV NMS
    
    Returns:
        boxes_3d: (N, 9) - [x, y, z, w, l, h, yaw, vx, vy]
        scores: (N,) - confidence scores
        labels: (N,) - class indices
    """
    heatmap = predictions['heatmap']      # (B, C, H, W)
    offset = predictions['offset']         # (B, 2, H, W)
    height = predictions['height']         # (B, 2, H, W)
    size = predictions['size']             # (B, 2, H, W)
    rotation = predictions['rotation']     # (B, 2, H, W)
    velocity = predictions['velocity']     # (B, 2, H, W)
    
    B, num_classes, H, W = heatmap.shape
    
    # Find local maxima in heatmap (peak detection)
    heatmap_peaks = nms_2d(heatmap, kernel_size=3)
    
    # Extract top-K detections
    topk_scores, topk_inds = heatmap_peaks.view(B, -1).topk(500)
    
    # Decode each detection
    for b in range(B):
        for k in range(500):
            score = topk_scores[b, k]
            if score < score_threshold:
                break
            
            # Get pixel position
            cls = topk_inds[b, k] // (H * W)
            pos = topk_inds[b, k] % (H * W)
            py, px = pos // W, pos % W
            
            # Decode center position
            x = (px + offset[b, 0, py, px]) / W * (bev_range[1] - bev_range[0]) + bev_range[0]
            y = (py + offset[b, 1, py, px]) / H * (bev_range[3] - bev_range[2]) + bev_range[2]
            z = height[b, 0, py, px]         # z-center
            h = height[b, 1, py, px].exp()   # height
            
            # Decode size
            w = size[b, 0, py, px].exp()     # width
            l = size[b, 1, py, px].exp()     # length
            
            # Decode rotation
            sin_yaw = rotation[b, 0, py, px]
            cos_yaw = rotation[b, 1, py, px]
            yaw = torch.atan2(sin_yaw, cos_yaw)
            
            # Decode velocity
            vx = velocity[b, 0, py, px]
            vy = velocity[b, 1, py, px]
    
    # Apply BEV NMS
    # ...
    
    return boxes_3d, scores, labels
```

### 5.3 Detection Head Output Summary

| Output | Shape | Description |
|--------|-------|-------------|
| Heatmap | (B, 10, 256, 256) | Per-class center probability |
| Offset | (B, 2, 256, 256) | Sub-pixel center refinement (dx, dy) |
| Height | (B, 2, 256, 256) | z-center and box height |
| Size | (B, 2, 256, 256) | log(width), log(length) |
| Rotation | (B, 2, 256, 256) | sin(yaw), cos(yaw) |
| Velocity | (B, 2, 256, 256) | vx, vy in ego frame |

---

## 6. Complete Model Specifications

### 6.1 Layer Count and Parameter Summary

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Camera Backbone (ResNet-50) | 23.5M | Pre-trained on ImageNet |
| Camera FPN | 3.3M | 256-channel outputs |
| Radar Pillar Encoder | 0.1M | Lightweight PointNet |
| Radar Sparse Backbone | 2.8M | Sparse convolutions |
| Radar BEV Neck | 1.5M | Multi-scale fusion |
| SCFT (6 layers) | 12.4M | Core fusion module |
| Detection Head | 4.2M | Anchor-free predictions |
| **Total** | **~47.8M** | Full model |

### 6.2 Computational Complexity

| Component | FLOPs | Latency (A100) |
|-----------|-------|-----------------|
| Camera Backbone + FPN | ~120 GFLOPs | ~15 ms |
| Radar Branch | ~8 GFLOPs | ~3 ms |
| SCFT | ~45 GFLOPs | ~20 ms |
| Detection Head | ~15 GFLOPs | ~5 ms |
| **Total** | **~188 GFLOPs** | **~43 ms (~23 FPS)** |

### 6.3 Memory Requirements

| Setting | GPU Memory |
|---------|-----------|
| Training (batch_size=1) | ~12 GB |
| Training (batch_size=4) | ~38 GB |
| Inference (batch_size=1) | ~5 GB |
| Model weights (FP32) | ~192 MB |
| Model weights (FP16) | ~96 MB |

### 6.4 Input/Output Dimensions Summary

```
INPUTS:
  Camera: (B, 6, 3, 256, 704)     # 6 views, RGB, H x W
  Radar:  (B, N, 7)                # N points, 7 features each
  Intrinsics: (B, 6, 3, 3)        # Per-camera intrinsics
  Extrinsics: (B, 6, 4, 4)        # Per-camera extrinsics

INTERMEDIATE:
  Camera FPN P3: (B, 6, 256, 32, 88)
  Camera FPN P4: (B, 6, 256, 16, 44)
  Camera FPN P5: (B, 6, 256, 8, 22)
  Radar BEV: (B, 256, 256, 256)
  Fused BEV: (B, 256, 256, 256)

OUTPUTS:
  Heatmap:  (B, 10, 256, 256)     # Class probabilities
  Offset:   (B, 2, 256, 256)      # Center refinement
  Height:   (B, 2, 256, 256)      # z-center, box height
  Size:     (B, 2, 256, 256)      # width, length (log-scale)
  Rotation: (B, 2, 256, 256)      # sin(yaw), cos(yaw)
  Velocity: (B, 2, 256, 256)      # vx, vy
  
  Final Detections: List of (x, y, z, w, l, h, yaw, vx, vy, class, score)
```
