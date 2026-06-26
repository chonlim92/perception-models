# HDMapNet: Model Architecture

A detailed architecture reference for PyTorch engineers working on the HDMapNet
codebase. This document covers every stage of the pipeline from multi-camera
input to vectorized map output, with tensor shapes, math, and design rationale.

---

## 1. Overall Pipeline (ASCII Diagram)

```
                         Multi-Camera Images
                         (B, 6, 3, H, W)
                               |
                               v
              +-------------------------------+
              |   Image Backbone + FPN        |
              |  (EfficientNet-B0 / ResNet-50)|
              +-------------------------------+
                               |
                    Multi-scale features
                   (B, 6, C, H/8, W/8) ...
                               |
                               v
              +-------------------------------+
              |       View Transform          |
              |      (IPM  or  LSS)           |
              +-------------------------------+
                               |
                        BEV Features
                      (B, C, Hbev, Wbev)
                               |
                               v
              +-------------------------------+
              |        BEV Encoder            |
              |       (U-Net style)           |
              +-------------------------------+
                               |
              +-------+-------+-------+
              |       |               |
              v       v               v
         Semantic  Instance       Direction
          Head      Head            Head
      (B,Cls,H,W) (B,E,H,W)    (B,2,H,W)
```

---

## 2. Image Backbone with Feature Pyramid Network

### 2.1 EfficientNet-B0

EfficientNet-B0 is the default backbone. Its core building block is the
**Mobile Inverted Bottleneck Convolution (MBConv)** with Squeeze-and-Excite:

```
MBConv Block:
  Input (B, Cin, H, W)
      |
      v
  1x1 Conv (expand ratio t) --> (B, t*Cin, H, W)
      |
      v
  Depthwise Conv kxk, stride s --> (B, t*Cin, H/s, W/s)
      |
      v
  Squeeze-and-Excite:
      GlobalAvgPool --> (B, t*Cin, 1, 1)
      FC(t*Cin, t*Cin/r) --> ReLU
      FC(t*Cin/r, t*Cin) --> Sigmoid
      Scale feature map element-wise
      |
      v
  1x1 Conv (project) --> (B, Cout, H/s, W/s)
      |
      v
  (+ skip connection if stride=1 and Cin==Cout)
```

**Compound Scaling** -- EfficientNet uses a compound coefficient phi to
uniformly scale depth (alpha^phi), width (beta^phi), and resolution
(gamma^phi) under the constraint alpha * beta^2 * gamma^2 ~ 2. For B0:
alpha=1.2, beta=1.1, gamma=1.15, phi=0 (baseline).

EfficientNet-B0 stage structure:

| Stage | Operator   | Resolution | Channels | Layers |
|-------|-----------|------------|----------|--------|
| 1     | MBConv1   | 112x112    | 16       | 1      |
| 2     | MBConv6   | 112x112    | 24       | 2      |
| 3     | MBConv6   | 56x56      | 40       | 2      |
| 4     | MBConv6   | 28x28      | 80       | 3      |
| 5     | MBConv6   | 14x14      | 112      | 3      |
| 6     | MBConv6   | 14x14      | 192      | 4      |
| 7     | MBConv6   | 7x7        | 320      | 1      |

We tap features at stages 3 (1/8), 5 (1/16), and 7 (1/32) for the FPN.

### 2.2 ResNet-50 Alternative

When compute budget allows, ResNet-50 replaces EfficientNet-B0. The standard
residual block uses the bottleneck design:

```
Bottleneck:
  Input (B, Cin, H, W)
      |------------------+
      v                  |
  1x1 Conv --> BN --> ReLU    (reduce: Cin -> Cin/4)
      |                  |
      v                  |
  3x3 Conv --> BN --> ReLU    (spatial: Cin/4 -> Cin/4)
      |                  |
      v                  |
  1x1 Conv --> BN            (expand: Cin/4 -> Cout)
      |                  |
      v                  |
  + <--------------------+    (skip, with 1x1 proj if Cin != Cout)
      |
      v
    ReLU
```

Feature extraction points:
- C3: output of layer2, stride 8, 512 channels
- C4: output of layer3, stride 16, 1024 channels
- C5: output of layer4, stride 32, 2048 channels

### 2.3 Feature Pyramid Network (FPN)

The FPN fuses multi-scale features via lateral connections and top-down
upsampling:

```
            C5 (1/32)         P5 (1/32)
              |                  ^
              v                  |
         1x1 Conv(C5->256) ---->+----> 3x3 Conv --> P5
                                |
            C4 (1/16)           | (upsample 2x)
              |                 v
              v                 +
         1x1 Conv(C4->256) --->+----> 3x3 Conv --> P4 (1/16)
                                |
            C3 (1/8)            | (upsample 2x)
              |                 v
              v                 +
         1x1 Conv(C3->256) --->+----> 3x3 Conv --> P3 (1/8)
```

The lateral connections are 1x1 convolutions that reduce channel count to a
uniform `fpn_dim` (typically 64 or 256). The top-down pathway upsamples via
`F.interpolate(mode='bilinear')` and adds element-wise to the lateral output.

For HDMapNet's default configuration with EfficientNet-B0 and input size
(3, 128, 352), the FPN output at P3 scale is the primary feature used
downstream:

```python
# Pseudocode
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=64):
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        # features = [C3, C4, C5]
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i], scale_factor=2, mode='bilinear'
            )
        return [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
```

### 2.4 Input/Output Shapes

| Configuration | Input Shape | P3 (1/8) | P4 (1/16) | P5 (1/32) |
|---------------|-------------|-----------|-----------|-----------|
| Small (128x352) | (B,6,3,128,352) | (B,6,64,16,44) | (B,6,64,8,22) | (B,6,64,4,11) |
| Large (900x1600) | (B,6,3,900,1600) | (B,6,64,113,200) | (B,6,64,57,100) | (B,6,64,29,50) |

The backbone processes each camera independently (or batched as B*N_cams):

```python
# Reshape for batch processing
imgs = imgs.view(B * N_cams, C, H, W)       # (B*6, 3, 128, 352)
feats = backbone(imgs)                       # (B*6, 64, 16, 44) at P3
feats = feats.view(B, N_cams, C_feat, H_f, W_f)  # (B, 6, 64, 16, 44)
```

---

## 3. IPM Variant (View Transform Option 1)

### 3.1 Inverse Perspective Mapping Fundamentals

IPM warps perspective images to a Bird's-Eye View using the assumption that
all observed points lie on the ground plane (z=0). This is a reasonable
approximation for road surfaces and lane markings.

### 3.2 Homography Derivation

Given camera intrinsic matrix K (3x3) and extrinsic [R|t] mapping world to
camera coordinates, a 3D point P_w = (X, Y, Z)^T projects as:

```
p = K [R | t] P_w

where p = (u, v, 1)^T in homogeneous pixel coords
```

Under the flat ground assumption Z = 0, the third column of R is eliminated:

```
    [u]       [r1 r2 r3 | t]   [X]
s * [v] = K * [         |  ] * [Y]
    [1]       [         |  ]   [0]
                                [1]

    [u]       [r1 r2 | t]   [X]
s * [v] = K * [      |  ] * [Y]
    [1]       [      |  ]   [1]
```

The 3x3 homography H mapping BEV coordinates (X,Y) to pixel (u,v) is:

```
H = K * [r1 | r2 | t]
```

where r1, r2 are the first two columns of R, and t is the translation.

To warp an image TO BEV, we need the inverse mapping H^{-1} that maps
BEV (x_bev, y_bev) -> pixel (u, v):

```
H_inv = H^{-1}    (3x3 matrix)

[u']       [x_bev]
[v'] = H * [y_bev]
[1 ]       [1    ]
```

Actually, for `grid_sample`, we need the inverse: given a BEV location,
where does it sample from in the image. So we use H^{-1}:

```
[x_bev]              [u]
[y_bev] = H^{-1} *  [v]
[  1  ]              [1]
```

Wait -- let's be precise. `grid_sample` takes an output-space grid and maps
each output pixel to its source location in the input. If output = BEV and
input = perspective image:

```
For each (x_bev, y_bev) in BEV grid:
    (u, v) = project(x_bev, y_bev) using H
    sample pixel from image at (u, v)
```

So the sampling grid is computed as:

```python
# BEV grid: (H_bev, W_bev, 2)  -- (x_bev, y_bev) in meters
# H: (3, 3) homography per camera

# Homogeneous BEV coords: (H_bev*W_bev, 3)
bev_pts = torch.stack([x_grid, y_grid, ones], dim=-1)

# Project to pixel: (H_bev*W_bev, 3)
pixel_pts = (H @ bev_pts.T).T      # (N, 3)
pixel_pts = pixel_pts[:, :2] / pixel_pts[:, 2:3]  # perspective divide

# Normalize to [-1, 1] for grid_sample
grid_u = 2 * pixel_pts[:, 0] / (W_img - 1) - 1
grid_v = 2 * pixel_pts[:, 1] / (H_img - 1) - 1
grid = torch.stack([grid_u, grid_v], dim=-1).view(H_bev, W_bev, 2)
```

### 3.3 Differentiable Warping with grid_sample

```python
# Per-camera warping
# feats: (B, N_cams, C, H_feat, W_feat)
# grids: (N_cams, H_bev, W_bev, 2)  -- precomputed per camera

bev_feats = []
for cam_idx in range(N_cams):
    feat_cam = feats[:, cam_idx]  # (B, C, H_feat, W_feat)
    grid_cam = grids[cam_idx].unsqueeze(0).expand(B, -1, -1, -1)
    warped = F.grid_sample(
        feat_cam, grid_cam,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # (B, C, H_bev, W_bev)
    bev_feats.append(warped)

# Aggregate: element-wise mean or max over cameras
bev = torch.stack(bev_feats, dim=1).mean(dim=1)  # (B, C, H_bev, W_bev)
```

### 3.4 Stitching Multiple Cameras

Each camera has its own H_i. After warping all 6 cameras independently, BEV
features are aggregated. Overlapping regions (where multiple cameras observe
the same ground patch) use mean pooling or learned attention weights:

```
BEV_unified = (1/K) * sum_{i: visible}( warp_i(feat_i) )
```

where K is the number of cameras covering each BEV cell.

### 3.5 Tensor Shape Trace (IPM)

```
Input:           (B, 6, 3, 128, 352)
After backbone:  (B, 6, 64, 16, 44)     -- 1/8 scale P3 features
Per-cam warp:    (B, 64, 200, 200)       -- per camera, then aggregate
After stitch:    (B, 64, 200, 200)       -- unified BEV
```

### 3.6 Limitations of IPM

- **Flat ground only**: elevated objects (vehicles, poles) create ghost
  artifacts stretching toward the camera
- **No depth reasoning**: cannot handle multi-layer structures (overpasses)
- **Fixed geometry**: homography must be recomputed if camera moves

---

## 4. LSS Variant (View Transform Option 2)

Lift-Splat-Shoot (LSS) is a learned view transform that does NOT assume a
flat ground plane. Instead, it predicts per-pixel depth distributions and
"lifts" 2D features into 3D, then projects (splats) into BEV.

### 4.1 Depth Prediction Network

A small network (typically a 1x1 convolution head on the image features)
produces a categorical depth distribution for each spatial location:

```python
class DepthNet(nn.Module):
    def __init__(self, in_channels, num_depth_bins):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels, num_depth_bins, kernel_size=1)

    def forward(self, feat):
        # feat: (B*N, C, H_f, W_f)
        depth_logits = self.depth_conv(feat)       # (B*N, D, H_f, W_f)
        depth_probs = depth_logits.softmax(dim=1)  # (B*N, D, H_f, W_f)
        return depth_probs
```

Depth bins are linearly spaced between `d_min` and `d_max`:

```
D = 41 bins (default)
d_min = 4.0 m
d_max = 44.0 m
bin_size = (d_max - d_min) / (D - 1) = 1.0 m
depths = [4.0, 5.0, 6.0, ..., 44.0]
```

### 4.2 Frustum Creation

For each pixel (h, w) in the feature map, we create D candidate 3D points
along its viewing ray at each discrete depth:

```
For pixel (h, w) at depth d_k:
    u = w * stride    (map back to original image coords)
    v = h * stride
    X_cam = (u - cx) * d_k / fx
    Y_cam = (v - cy) * d_k / fy
    Z_cam = d_k
```

This creates a "frustum" of shape (D, H_f, W_f, 3) per camera, containing
the 3D coordinates of each potential point in camera frame.

Transform to ego frame using camera extrinsics:

```
P_ego = R_cam2ego @ P_cam + t_cam2ego
```

### 4.3 Outer Product: Feature x Depth

The key insight of LSS: create a volumetric feature by taking the outer
product of the image feature vector and the depth probability:

```python
# feat:  (B, N, C, H_f, W_f) -- image features
# depth: (B, N, D, H_f, W_f) -- depth probabilities

# Outer product via broadcasting
volume = feat.unsqueeze(3) * depth.unsqueeze(2)
# feat:   (B, N, C, 1, H_f, W_f)
# depth:  (B, N, 1, D, H_f, W_f)
# volume: (B, N, C, D, H_f, W_f)
```

Mathematically, for each pixel (h, w):

```
v(c, d) = f(c) * p(d)

where:
  f in R^C  is the feature vector at that pixel
  p in R^D  is the depth probability at that pixel
  v in R^{C x D}  is the resulting volume
```

This is an expected-value formulation: the feature at depth d_k is weighted
by the probability that the actual surface is at d_k.

### 4.4 Splat into Voxel Grid

Each point in the frustum has:
- A 3D location in ego frame (known from geometry)
- A feature vector (from the outer product)

We "splat" (scatter-add) these features into a discrete voxel grid:

```python
# Voxel grid: (X_bins, Y_bins, Z_bins) covering ego-centric region
# e.g., X: [-50m, 50m], Y: [-50m, 50m], Z: [-10m, 10m]
# Resolution: 0.5m -> 200 x 200 x 40

# Compute voxel indices for each frustum point
voxel_x = ((point_x - x_min) / voxel_size).long()
voxel_y = ((point_y - y_min) / voxel_size).long()
voxel_z = ((point_z - z_min) / voxel_size).long()

# Flatten to linear index for scatter_add
linear_idx = voxel_x * (Y_bins * Z_bins) + voxel_y * Z_bins + voxel_z

# Scatter features into voxel grid
voxel_grid = torch.zeros(B, C, X_bins * Y_bins * Z_bins)
voxel_grid.scatter_add_(2, linear_idx.expand(B, C, -1), volume_flat)
voxel_grid = voxel_grid.view(B, C, X_bins, Y_bins, Z_bins)
```

### 4.5 Collapse Height to BEV

Sum (or max-pool) along the Z (height) axis to obtain a 2D BEV feature map:

```python
bev = voxel_grid.sum(dim=-1)  # (B, C, X_bins, Y_bins) = (B, 64, 200, 200)
```

### 4.6 Tensor Shape Trace (LSS)

```
Input:             (B, 6, 3, 128, 352)
After backbone:    (B, 6, 64, 16, 44)        -- C=64, 1/8 scale
Depth prediction:  (B, 6, 41, 16, 44)        -- D=41 depth bins
Outer product:     (B, 6, 64, 41, 16, 44)    -- lifted volume per camera
After splat:       (B, 64, 200, 200, 10)     -- 3D voxel grid (Z=10)
After Z-collapse:  (B, 64, 200, 200)         -- BEV feature map
```

### 4.7 Implementation Detail: Efficient Splatting

The naive scatter is slow. The actual implementation uses a "pillar" approach:

1. Pre-sort frustum points by their (x, y) voxel assignment
2. Use cumulative sum within each pillar for efficient aggregation
3. Implemented via custom CUDA kernels or `torch.unique` + `scatter_add_`

```python
# Efficient pillar pooling (simplified)
# Group points by BEV cell, sum features within each cell
bev_idx = voxel_x * Y_bins + voxel_y   # (B, N*D*H*W)
bev_flat = torch.zeros(B, C, X_bins * Y_bins, device=device)
bev_flat.scatter_add_(2, bev_idx.unsqueeze(1).expand(-1, C, -1), feats_flat)
bev = bev_flat.view(B, C, X_bins, Y_bins)
```

---

## 5. BEV Encoder (U-Net Architecture)

The BEV encoder refines the raw BEV features using a U-Net-style
encoder-decoder with skip connections.

### 5.1 Architecture Diagram

```
BEV Input (B, 64, 200, 200)
         |
         v
    +----------+
    | Enc Blk 1|  Conv(64,128,3,s=2) + BN + ReLU + Conv(128,128,3) + BN + ReLU
    +----------+
         |  (B, 128, 100, 100)          skip_1
         |------------------------------->|
         v                               |
    +----------+                          |
    | Enc Blk 2|  Conv(128,256,3,s=2) + BN + ReLU + Conv(256,256,3) + BN + ReLU
    +----------+                          |
         |  (B, 256, 50, 50)    skip_2    |
         |------------------->|           |
         v                    |           |
    +----------+              |           |
    | Bottleneck|  Conv(256,512,3,s=2) + BN + ReLU + Conv(512,512,3) + BN + ReLU
    +----------+              |           |
         |  (B, 512, 25, 25)  |           |
         v                    |           |
    +----------+              |           |
    | Dec Blk 2|  Upsample(2x) + Cat(skip_2) + Conv(768,256,3) + BN + ReLU
    +----------+              |           |
         |  (B, 256, 50, 50)              |
         v                                |
    +----------+                          |
    | Dec Blk 1|  Upsample(2x) + Cat(skip_1) + Conv(384,128,3) + BN + ReLU
    +----------+                          |
         |  (B, 128, 100, 100)            |
         v
    +----------+
    | Final Up |  Upsample(2x) + Conv(128,128,3) + BN + ReLU
    +----------+
         |
         v
    BEV Output (B, 128, 200, 200)
```

### 5.2 Encoder Block Detail

```python
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
```

### 5.3 Decoder Block Detail

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
```

### 5.4 Tensor Shape Trace (BEV Encoder)

```
Input:      (B, 64, 200, 200)
Enc1:       (B, 128, 100, 100)
Enc2:       (B, 256, 50, 50)
Bottleneck: (B, 512, 25, 25)
Dec2:       (B, 256, 50, 50)     -- cat(512+256) -> conv -> 256
Dec1:       (B, 128, 100, 100)   -- cat(256+128) -> conv -> 128
Final:      (B, 128, 200, 200)   -- upsample + conv
```

---

## 6. Three Prediction Heads

All three heads operate on the same BEV feature map and produce outputs at
the same spatial resolution (200 x 200 by default, representing the
surrounding area at 0.5m per pixel = 100m x 100m coverage).

### 6.1 Semantic Segmentation Head

Predicts per-cell map element classes (e.g., road boundary, lane divider,
pedestrian crossing).

```python
class SemanticHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )

    def forward(self, bev_feat):
        return self.conv(bev_feat)  # (B, num_classes, H, W)
```

- **Output**: (B, num_classes, 200, 200)
- **Activation**: Sigmoid (multi-label, each class independent)
- **Loss**: Binary Cross-Entropy

```
L_semantic = -1/(N*C*H*W) * sum[ y*log(p) + (1-y)*log(1-p) ]
```

### 6.2 Instance Embedding Head

Produces dense embeddings for discriminative instance segmentation. Points
belonging to the same map element (e.g., same lane line) should have similar
embeddings, while different instances should be far apart.

```python
class InstanceHead(nn.Module):
    def __init__(self, in_channels, embed_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, embed_dim, 1),
        )

    def forward(self, bev_feat):
        return self.conv(bev_feat)  # (B, embed_dim, H, W)
```

- **Output**: (B, 16, 200, 200)
- **Activation**: None (raw embeddings)
- **Loss**: Discriminative loss (Brabandere et al., 2017)

The discriminative loss has three terms:

```
L_var   = (1/C) * sum_c [ (1/Nc) * sum_i max(||mu_c - e_i|| - delta_v, 0)^2 ]
L_dist  = (1/C*(C-1)) * sum_{ca != cb} max(2*delta_d - ||mu_ca - mu_cb||, 0)^2
L_reg   = (1/C) * sum_c ||mu_c||

L_instance = alpha * L_var + beta * L_dist + gamma * L_reg
```

where:
- `mu_c` = mean embedding of instance c
- `e_i` = embedding at pixel i
- `delta_v` = variance margin (pull threshold, typically 0.5)
- `delta_d` = distance margin (push threshold, typically 1.5)
- `alpha, beta, gamma` = loss weights (1.0, 1.0, 0.001)

### 6.3 Direction Head

Predicts the local tangent direction of map elements at each BEV cell. This
is crucial for resolving ambiguities when multiple lines cross or merge.

```python
class DirectionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, 1),
        )

    def forward(self, bev_feat):
        return self.conv(bev_feat)  # (B, 2, H, W)
```

- **Output**: (B, 2, 200, 200) -- predicts (cos(theta), sin(theta))
- **Activation**: None (the network learns to output values in [-1, 1])
- **Loss**: L1 loss on unit-direction targets

```
L_direction = (1/N) * sum_i [ |cos_pred_i - cos_gt_i| + |sin_pred_i - sin_gt_i| ]
```

The direction is only supervised at pixels that belong to a map element
(using the semantic mask as a gate).

---

## 7. Post-Processing Pipeline

The post-processing converts dense rasterized predictions into vectorized
polylines suitable for downstream planning.

### 7.1 Pipeline Overview

```
Semantic Logits (B, Cls, 200, 200)
        |
        v
   Threshold (p > 0.5)
        |
        v
   Binary Masks per class
        |
        v
   Morphological Thinning (skeletonize)
        |
        v
   Skeleton (1-pixel wide)
        |
        v
   Connected Components
        |
        v
   Per-component:
        |
        v
   Trace skeleton to ordered polyline
        |
        v
   Direction-guided intersection resolution
        |
        v
   Vectorized Map Lines [(x1,y1), (x2,y2), ...]
```

### 7.2 Step Details

**Thresholding**:
```python
semantic_mask = (semantic_logits.sigmoid() > threshold)  # threshold=0.5
```

**Skeletonization** (morphological thinning):
```python
from skimage.morphology import skeletonize
skeleton = skeletonize(semantic_mask.numpy())
# Reduces thick predictions to 1-pixel centerlines
```

**Connected Component Analysis**:
```python
from scipy.ndimage import label
labeled, num_features = label(skeleton)
# Each connected region gets a unique integer ID
```

**Skeleton Tracing**:
Convert the skeleton pixel set into an ordered sequence of points by walking
along the skeleton from endpoints, choosing the next pixel by proximity.

**Direction-Guided Intersection Resolution**:
At junctions where multiple skeleton branches meet, the direction predictions
disambiguate which incoming branch connects to which outgoing branch:

```python
# At junction pixel (x, y):
direction = direction_pred[:, :, y, x]  # (cos_theta, sin_theta)
theta = atan2(sin_theta, cos_theta)

# Match incoming/outgoing branches by angular compatibility
# Branch pair with smallest angular difference gets connected
```

### 7.3 Instance-Based Alternative

When using instance embeddings, an alternative post-processing pipeline:

1. Threshold semantic predictions
2. Extract embeddings at foreground pixels
3. Cluster embeddings (mean-shift or HDBSCAN)
4. Each cluster = one map instance
5. Fit polyline through each cluster's pixel coordinates

```python
from sklearn.cluster import MeanShift

fg_mask = semantic_mask[class_idx]  # foreground for one class
embeddings = instance_output[:, fg_mask]  # (embed_dim, N_fg)
clustering = MeanShift(bandwidth=0.5).fit(embeddings.T)
labels = clustering.labels_
```

---

## 8. Complete Tensor Shape Trace (End-to-End)

```
============================================================
STAGE                           SHAPE
============================================================
Input images                    (B, 6, 3, 128, 352)
------------------------------------------------------------
Reshape for backbone            (B*6, 3, 128, 352)
EfficientNet-B0 stage 3         (B*6, 40, 16, 44)
EfficientNet-B0 stage 5         (B*6, 112, 8, 22)
EfficientNet-B0 stage 7         (B*6, 320, 4, 11)
------------------------------------------------------------
FPN lateral + top-down          
  P3                            (B*6, 64, 16, 44)
  P4                            (B*6, 64, 8, 22)
  P5                            (B*6, 64, 4, 11)
------------------------------------------------------------
Use P3, reshape back            (B, 6, 64, 16, 44)
============================================================
VIEW TRANSFORM (LSS variant)
------------------------------------------------------------
Depth network                   (B, 6, 41, 16, 44)
Outer product (lift)            (B, 6, 64, 41, 16, 44)
Frustum 3D coords               (6, 41, 16, 44, 3)
After ego transform             (B, 6, 41*16*44, 3) = (B,6,28864,3)
Splat to voxel grid             (B, 64, 200, 200, 10)
Collapse Z (sum)                (B, 64, 200, 200)
============================================================
BEV ENCODER (U-Net)
------------------------------------------------------------
Enc block 1 (s=2)              (B, 128, 100, 100)
Enc block 2 (s=2)              (B, 256, 50, 50)
Bottleneck (s=2)               (B, 512, 25, 25)
Dec block 2 (up + skip)        (B, 256, 50, 50)
Dec block 1 (up + skip)        (B, 128, 100, 100)
Final upsample                 (B, 128, 200, 200)
============================================================
PREDICTION HEADS
------------------------------------------------------------
Semantic head (1x1 conv)       (B, 3, 200, 200)
Instance head (1x1 conv)       (B, 16, 200, 200)
Direction head (1x1 conv)      (B, 2, 200, 200)
============================================================
```

---

## 9. Loss Function Summary

The total training loss is a weighted sum:

```
L_total = w_sem * L_semantic + w_inst * L_instance + w_dir * L_direction

Default weights:
  w_sem  = 1.0
  w_inst = 1.0
  w_dir  = 0.2
```

| Loss | Formula | Notes |
|------|---------|-------|
| L_semantic | BCE(sigmoid(pred), target) | Multi-label, per-class |
| L_instance | L_var + L_dist + L_reg | Pull-push discriminative |
| L_direction | L1(pred_dir, gt_dir) | Masked to foreground only |

---

## 10. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-7 |
| LR schedule | OneCycleLR (max_lr=1e-3) |
| Batch size | 4 (per GPU) |
| BEV resolution | 200 x 200 (0.5 m/pixel) |
| BEV range | [-50m, 50m] x [-25m, 25m] (asymmetric) or [-50m, 50m]^2 |
| Depth bins (LSS) | 41 (4m to 44m, 1m spacing) |
| Embed dim | 16 |
| Num classes | 3 (boundary, divider, crossing) |
| Epochs | 30 |

---

## 11. Design Decisions and Trade-offs

### IPM vs LSS

| Aspect | IPM | LSS |
|--------|-----|-----|
| Ground assumption | Required (z=0) | Not required |
| Elevated objects | Ghosting artifacts | Handled correctly |
| Compute cost | Lower (no depth net) | Higher (depth + scatter) |
| Training | No learned transform | End-to-end learned |
| Calibration sensitivity | High (exact H needed) | Moderate (learned compensation) |

### Why Three Heads?

- **Semantic alone** cannot distinguish overlapping instances
- **Instance alone** cannot classify what type of element it is
- **Direction alone** is meaningless without knowing where lines exist

Together they enable:
1. Identify which cells contain map elements (semantic)
2. Group cells into individual polylines (instance)
3. Determine traversal order and resolve intersections (direction)

### BEV Resolution Choice

At 0.5m per pixel with a 200x200 grid:
- Coverage: 100m x 100m around ego vehicle
- Sufficient for HD map elements (lane width ~3.5m = 7 pixels)
- Manageable compute: 40,000 cells vs. millions in raw images

---

## 12. Inference Pseudocode

```python
class HDMapNet(nn.Module):
    def __init__(self, backbone, fpn, view_transform, bev_encoder,
                 semantic_head, instance_head, direction_head):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.view_transform = view_transform  # IPM or LSS
        self.bev_encoder = bev_encoder
        self.semantic_head = semantic_head
        self.instance_head = instance_head
        self.direction_head = direction_head

    def forward(self, imgs, intrinsics, extrinsics):
        """
        Args:
            imgs: (B, N_cams, 3, H, W) multi-camera images
            intrinsics: (B, N_cams, 3, 3) camera intrinsic matrices
            extrinsics: (B, N_cams, 4, 4) cam-to-ego transforms

        Returns:
            semantic: (B, num_classes, H_bev, W_bev)
            instance: (B, embed_dim, H_bev, W_bev)
            direction: (B, 2, H_bev, W_bev)
        """
        B, N, C, H, W = imgs.shape

        # 1. Extract features per camera
        imgs_flat = imgs.view(B * N, C, H, W)
        feats = self.backbone(imgs_flat)          # multi-scale list
        feats = self.fpn(feats)                   # [P3, P4, P5]
        feat = feats[0]                           # use P3: (B*N, 64, H/8, W/8)
        feat = feat.view(B, N, *feat.shape[1:])   # (B, N, 64, H/8, W/8)

        # 2. Transform to BEV
        bev = self.view_transform(feat, intrinsics, extrinsics)
        # (B, 64, 200, 200)

        # 3. Encode BEV features
        bev = self.bev_encoder(bev)               # (B, 128, 200, 200)

        # 4. Predict
        semantic = self.semantic_head(bev)        # (B, 3, 200, 200)
        instance = self.instance_head(bev)        # (B, 16, 200, 200)
        direction = self.direction_head(bev)      # (B, 2, 200, 200)

        return semantic, instance, direction
```

---

## 13. Coordinate System Conventions

```
Ego Vehicle Frame (right-handed):
    X: forward (driving direction)
    Y: left
    Z: up

BEV Grid Indexing:
    Row i  -> Y axis (i=0 is leftmost, i=199 is rightmost)
    Col j  -> X axis (j=0 is rear, j=199 is front)

    bev[b, c, i, j] corresponds to world point:
        x = x_min + j * resolution
        y = y_max - i * resolution

Camera Frame:
    X: right
    Y: down
    Z: forward (into the scene)
```

---

## 14. References

- Li, Z. et al. "HDMapNet: An Online HD Map Construction and Evaluation Framework." ICRA 2022.
- Philion, J. & Fidler, S. "Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs." ECCV 2020.
- Tan, M. & Le, Q. "EfficientNet: Rethinking Model Scaling for CNNs." ICML 2019.
- De Brabandere, B. et al. "Semantic Instance Segmentation with a Discriminative Loss Function." CVPRW 2017.
- Lin, T.-Y. et al. "Feature Pyramid Networks for Object Detection." CVPR 2017.
