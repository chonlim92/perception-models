# Cylinder3D: Model Architecture

## Overview

Cylinder3D is a 3D semantic segmentation network built on three core components:

1. **Cylindrical voxelization** to partition the LiDAR point cloud
2. **Asymmetric 3D sparse convolution U-Net** for voxel-level feature learning
3. **Point-wise refinement module** to recover fine-grained predictions

---

## Architecture Diagram (Text)

```
Input Point Cloud (N × 4: x, y, z, intensity)
         │
         ▼
┌─────────────────────────────┐
│  Cylindrical Voxelization   │
│  (x,y,z) → (r, θ, z)       │
│  Grid: 480 × 360 × 32      │
│  Per-voxel: MLP features    │
└─────────────────────────────┘
         │
         ▼ Sparse Tensor (occupied voxels only)
┌─────────────────────────────────────────────────────────────┐
│           Asymmetric 3D U-Net (Sparse Convolutions)         │
│                                                             │
│  Encoder:                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │
│  │ Block 1  │──▶│ Block 2  │──▶│ Block 3  │──▶│Block 4 │ │
│  │ 32 ch    │   │ 64 ch    │   │ 128 ch   │   │256 ch  │ │
│  │ 480×360  │   │ 240×180  │   │ 120×90   │   │60×45   │ │
│  │ ×32      │   │ ×16      │   │ ×8       │   │×4      │ │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └───┬────┘ │
│       │skip          │skip          │skip          │       │
│       │              │              │              │       │
│  Decoder:            │              │              │       │
│  ┌──────────┐   ┌────┴─────┐   ┌───┴──────┐   ┌──┴────┐ │
│  │ Block 5  │◀──│ Block 6  │◀──│ Block 7  │◀──│Block 8│ │
│  │ 32 ch    │   │ 64 ch    │   │ 128 ch   │   │256 ch │ │
│  │ 480×360  │   │ 240×180  │   │ 120×90   │   │60×45  │ │
│  │ ×32      │   │ ×16      │   │ ×8       │   │×4     │ │
│  └──────────┘   └──────────┘   └──────────┘   └───────┘ │
│                                                             │
│  Each block contains:                                       │
│    - Asymmetric Residual Convolution                        │
│    - DDCMod (Dimension-Decomposition Context Modeling)      │
│    - Batch Normalization + LeakyReLU                        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼ Voxel features (F_voxel)
┌─────────────────────────────┐
│  Point-wise Refinement      │
│  Concat: [F_voxel, F_point] │
│  MLP: 256→128→64→C          │
│  Output: N × C (per-point)  │
└─────────────────────────────┘
         │
         ▼
   Per-point Predictions (N × 19 classes)
```

---

## Component 1: Cylindrical Voxelization

### Coordinate Transform

```python
# Cartesian to Cylindrical
r     = sqrt(x^2 + y^2)       # radial distance [0, 50m]
theta = atan2(y, x) + pi      # azimuth angle [0, 2*pi]
z     = z                     # height [-3m, 1m]
```

### Grid Partition

| Dimension | Range | Bins | Resolution |
|-----------|-------|------|------------|
| r (radial) | [0, 50] m | 480 | 10.42 cm |
| theta (azimuth) | [0, 2*pi] rad | 360 | 1.0° |
| z (height) | [-3, 1] m | 32 | 12.5 cm |

**Total grid size:** 480 x 360 x 32 = 5,529,600 voxels
**Typical occupancy:** ~0.5-0.7% (sparse)

### Per-Voxel Feature Computation

For each occupied voxel, point features are aggregated:

```python
# For each voxel v with points {p_1, ..., p_k}:
voxel_features = PointMLP([
    # Input per point: [x, y, z, intensity, r, theta, z_cyl, dx, dy, dz]
    # where dx, dy, dz = offset from voxel center
    Linear(10, 64),
    BatchNorm1d(64),
    LeakyReLU(0.1),
    Linear(64, 128),
    BatchNorm1d(128),
    LeakyReLU(0.1),
    Linear(128, 128),
])

# Aggregation: max pooling over points in voxel
voxel_feature = max_pool(voxel_features)  # shape: (128,)
```

**Input features per point (10-dimensional):**
- `x, y, z` - Cartesian coordinates
- `intensity` - LiDAR return strength
- `r, theta, z_cyl` - Cylindrical coordinates
- `dx, dy, dz` - Offset from voxel center (provides sub-voxel localization)

---

## Component 2: Asymmetric 3D Sparse Convolution U-Net

### Sparse Convolution Backbone

The network uses **sparse convolutions** (via the `spconv` library) that only compute on occupied voxels, making it computationally efficient despite the large grid.

### Encoder Architecture

| Stage | Input Resolution | Output Resolution | Channels | Stride |
|-------|-----------------|-------------------|----------|--------|
| Stage 1 | 480×360×32 | 480×360×32 | 32 → 64 | 1 |
| Down 1 | 480×360×32 | 240×180×16 | 64 | 2 |
| Stage 2 | 240×180×16 | 240×180×16 | 64 → 128 | 1 |
| Down 2 | 240×180×16 | 120×90×8 | 128 | 2 |
| Stage 3 | 120×90×8 | 120×90×8 | 128 → 256 | 1 |
| Down 3 | 120×90×8 | 60×45×4 | 256 | 2 |
| Stage 4 | 60×45×4 | 60×45×4 | 256 → 256 | 1 |

### Decoder Architecture

| Stage | Input Resolution | Output Resolution | Channels | Stride |
|-------|-----------------|-------------------|----------|--------|
| Up 1 | 60×45×4 | 120×90×8 | 256 → 128 | 2 |
| Stage 5 | 120×90×8 | 120×90×8 | 256 → 128 | 1 (+ skip) |
| Up 2 | 120×90×8 | 240×180×16 | 128 → 64 | 2 |
| Stage 6 | 240×180×16 | 240×180×16 | 128 → 64 | 1 (+ skip) |
| Up 3 | 240×180×16 | 480×360×32 | 64 → 32 | 2 |
| Stage 7 | 480×360×32 | 480×360×32 | 64 → 32 | 1 (+ skip) |

### Asymmetric Residual Block

Each stage contains one or more asymmetric residual blocks:

```
Input (C channels)
  │
  ├──────────────────────────────────────┐ (identity/1×1 skip)
  │                                      │
  ▼                                      │
[3×1×3 Sparse Conv, C→C]                 │
[BatchNorm + LeakyReLU]                  │
  │                                      │
  ▼                                      │
[1×3×3 Sparse Conv, C→C]                 │
[BatchNorm + LeakyReLU]                  │
  │                                      │
  ▼                                      │
[3×3×3 Sparse Conv, C→C]                 │
[BatchNorm]                              │
  │                                      │
  ▼                                      │
  ────────── Element-wise Add ◀──────────┘
  │
  ▼
[LeakyReLU]
  │
  ▼
Output (C channels)
```

**Kernel dimension semantics:**
- `3×1×3` (r × theta × z): Captures radial-height patterns (depth layering)
- `1×3×3` (r × theta × z): Captures azimuthal-height patterns (lateral structure)
- `3×3×3` (r × theta × z): Full 3D context integration

### Dimension-Decomposition Based Context Modeling (DDCMod)

DDCMod is applied after certain asymmetric blocks to capture long-range dependencies:

```
Input (C channels)
  │
  ├─→ [K×1×1 Conv, C→C/4] → BN → ReLU    (radial context)
  │
  ├─→ [1×K×1 Conv, C→C/4] → BN → ReLU    (azimuthal context)
  │
  ├─→ [1×1×K Conv, C→C/4] → BN → ReLU    (height context)
  │
  └─→ [1×1×1 Conv, C→C/4] → BN → ReLU    (point-wise)
      │
      ▼
  Concatenate (C channels total)
      │
      ▼
  [1×1×1 Conv, C→C] → BN → ReLU
      │
      ▼
  Output (C channels)
```

**Typical K values:** 7, 9, or 11 (enabling receptive fields spanning large spatial extents along each axis)

---

## Component 3: Point-wise Refinement Module

The refinement module recovers per-point predictions from voxel features by combining:
- Voxel-level features (from the U-Net output)
- Original point-level features (from the initial MLP)

```
For each point p_i in voxel v:
  
  F_voxel = UNet_output[v]              # C_voxel-dimensional
  F_point = PointMLP_output[p_i]        # C_point-dimensional
  
  F_combined = Concat([F_voxel, F_point])  # (C_voxel + C_point)-dimensional
  
  Refinement MLP:
    Linear(C_voxel + C_point, 256)
    BatchNorm1d(256)
    LeakyReLU(0.1)
    Dropout(0.3)
    Linear(256, 128)
    BatchNorm1d(128)
    LeakyReLU(0.1)
    Dropout(0.3)
    Linear(128, 64)
    BatchNorm1d(64)
    LeakyReLU(0.1)
    Linear(64, num_classes)   # 19 for SemanticKITTI, 16 for nuScenes
```

This module is critical for handling:
- **Voxel boundary effects:** Multiple objects in one voxel get differentiated
- **Fine-grained geometry:** Sub-voxel details preserved from raw points
- **Feature fusion:** Low-level geometric cues complement high-level semantic features

---

## Network Configuration Details

### Channel Progression

```
Voxel Feature MLP: 10 → 64 → 128 → 128
Encoder: 128 → 32 → 64 → 128 → 256
Bottleneck: 256
Decoder: 256 → 128 → 64 → 32
Refinement MLP: (32+128) → 256 → 128 → 64 → C
```

### Activation Functions

- **LeakyReLU** with negative slope 0.1 throughout
- Applied after BatchNorm in convolution blocks
- No activation on final classification layer (logits)

### Normalization

- **BatchNorm3d** (or sparse equivalent) for all convolutional layers
- **BatchNorm1d** for MLP layers
- Momentum: 0.1 (PyTorch default)
- Epsilon: 1e-5

### Dropout

- **Dropout rate:** 0.3 in refinement MLP
- No dropout in sparse convolution backbone (BatchNorm provides regularization)

---

## Parameter Count and Computational Cost

### Model Size

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Voxel Feature MLP | ~25K | 0.4% |
| Encoder (Stages 1-4) | ~3.2M | 52% |
| Decoder (Stages 5-7) | ~2.4M | 39% |
| DDCMod modules | ~350K | 5.7% |
| Refinement MLP | ~180K | 2.9% |
| **Total** | **~6.15M** | **100%** |

### Computational Cost (FLOPs)

| Component | GFLOPs (approx.) | Notes |
|-----------|------------------|-------|
| Voxel Feature MLP | ~0.8 | Depends on occupied voxels |
| Sparse Conv Encoder | ~12.5 | Proportional to active sites |
| Sparse Conv Decoder | ~8.3 | Proportional to active sites |
| DDCMod | ~2.1 | Large kernels but sparse |
| Refinement MLP | ~1.5 | Per-point operations |
| **Total** | **~25.2** | Varies with point count |

### Memory Footprint

| Configuration | GPU Memory |
|---------------|-----------|
| Batch size 2, training | ~6-8 GB |
| Batch size 4, training | ~12-14 GB |
| Batch size 1, inference | ~3-4 GB |

---

## Comparison with Related Architectures

| Architecture | Representation | Convolution | Parameters | mIoU (SK) |
|-------------|---------------|-------------|-----------|-----------|
| MinkowskiNet42 | Cartesian voxel | Sparse 3D | ~37.9M | 63.1% |
| SPVNAS | Cartesian voxel (NAS) | Sparse 3D | ~12.5M | 66.4% |
| PolarNet | Polar BEV | Dense 2D/3D | ~13.6M | 54.3% |
| RangeNet++ | Range image | Dense 2D | ~50.4M | 52.2% |
| **Cylinder3D** | **Cylindrical voxel** | **Sparse 3D asymmetric** | **~6.15M** | **68.9%** |

Cylinder3D achieves the best accuracy with the fewest parameters, demonstrating the efficiency of the cylindrical representation and asymmetric convolutions.

---

## Implementation Notes

### Sparse Convolution Library

Cylinder3D relies on `spconv` (Sparse Convolution library by Yan et al.) for efficient sparse 3D convolutions:

```python
import spconv.pytorch as spconv

# Example sparse convolution layer
self.conv = spconv.SubMConv3d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    padding=1,
    bias=False,
    indice_key="subm1"  # reuse indices for efficiency
)

# Downsampling with strided sparse conv
self.down = spconv.SparseConv3d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=2,
    padding=1,
    bias=False,
    indice_key="down1"
)

# Upsampling with inverse sparse conv
self.up = spconv.SparseInverseConv3d(
    in_channels=128,
    out_channels=64,
    kernel_size=3,
    indice_key="down1"  # reuses indices from downsampling
)
```

### Key `spconv` Concepts

- **SubManifold Convolutions (`SubMConv3d`):** Only produce output at locations where input exists (preserves sparsity)
- **Regular Sparse Convolutions (`SparseConv3d`):** Can generate new active sites (used for downsampling)
- **Inverse Sparse Convolutions (`SparseInverseConv3d`):** Upsampling by reversing the downsampling pattern
- **Indice Keys:** Allow reuse of computed neighbor indices across layers for efficiency

### Input Tensor Construction

```python
# Construct sparse tensor from voxelized point cloud
coords = torch.tensor(voxel_coords, dtype=torch.int32)  # (M, 4): [batch, r, theta, z]
feats = torch.tensor(voxel_features, dtype=torch.float32)  # (M, C)
spatial_shape = [480, 360, 32]  # grid dimensions

sparse_input = spconv.SparseConvTensor(
    features=feats,
    indices=coords,
    spatial_shape=spatial_shape,
    batch_size=batch_size
)
```
