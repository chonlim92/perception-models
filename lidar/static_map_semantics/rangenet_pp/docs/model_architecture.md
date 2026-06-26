# RangeNet++: Model Architecture

## Architecture Overview

RangeNet++ consists of three main stages:

1. **Spherical Projection:** Convert 3D point cloud to 2D range image
2. **2D CNN (Encoder-Decoder):** Semantic segmentation on range image using DarkNet-53 backbone with U-Net decoder
3. **KNN Post-Processing:** Transfer predictions back to 3D and fix boundary artifacts

```
3D Point Cloud  -->  Spherical Projection  -->  Range Image (64x2048x5)
                                                       |
                                                       v
                                              DarkNet-53 Encoder
                                                       |
                                                       v
                                              U-Net Decoder (skip connections)
                                                       |
                                                       v
                                              Per-pixel Predictions (64x2048x20)
                                                       |
                                                       v
                                              KNN Post-Processing
                                                       |
                                                       v
                                              3D Point-wise Labels
```

---

## Stage 1: Spherical Projection

### Mathematical Formulation

Given a 3D point `p = (x, y, z)` with intensity `i`, compute the range image coordinates:

**Step 1: Compute range**
```
r = sqrt(x^2 + y^2 + z^2)
```

**Step 2: Compute spherical coordinates**
```
theta = arcsin(z / r)           # elevation angle (pitch)
phi = arctan2(y, x)             # azimuth angle (yaw)
```

**Step 3: Map to pixel coordinates**
```
u = 0.5 * (1 + phi / pi) * W             # horizontal pixel [0, W-1]
v = (1 - (theta - fov_down) / fov) * H   # vertical pixel [0, H-1]
```

Where:
- `W` = image width (2048 or 1024)
- `H` = image height (64, matching number of beams)
- `fov_up` = +2.0 degrees (upper vertical FOV limit)
- `fov_down` = -24.8 degrees (lower vertical FOV limit)
- `fov` = `fov_up - fov_down` = 26.8 degrees (total vertical FOV)

**Step 4: Construct range image channels**

For pixel `(u, v)`, the 5-channel representation is:
```
range_image[v, u, 0] = r          # range (distance)
range_image[v, u, 1] = x          # x coordinate
range_image[v, u, 2] = y          # y coordinate
range_image[v, u, 3] = z          # z coordinate
range_image[v, u, 4] = intensity  # remission value
```

### Projection Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| H | 64 | Image height (beam count) |
| W | 2048 (or 1024) | Image width (azimuth resolution) |
| fov_up | +2.0 deg | Upper FOV limit |
| fov_down | -24.8 deg | Lower FOV limit |
| fov_total | 26.8 deg | Total vertical FOV |

### Handling Conflicts

When multiple points project to the same pixel:
- **Strategy:** Keep the point with the minimum range value (closest point).
- **Rationale:** Mimics natural occlusion - closer objects block farther ones.
- **Data structure:** Maintain a depth buffer during projection.

### Normalization

Input channels are normalized before feeding to the network:
```
range_norm = (r - mean_r) / std_r
x_norm = (x - mean_x) / std_x
y_norm = (y - mean_y) / std_y
z_norm = (z - mean_z) / std_z
intensity_norm = (intensity - mean_i) / std_i
```

Statistics computed over the training set.

---

## Stage 2: DarkNet-53 Backbone (Encoder)

### Overview

DarkNet-53 is the backbone from YOLOv3, adapted for dense prediction. It uses residual connections and aggressive downsampling through strided convolutions (no max pooling).

### Building Blocks

**Darknet Residual Block:**
```
Input
  |
  +--> Conv 1x1 (reduce channels) --> BN --> LeakyReLU
  |         |
  |         v
  |    Conv 3x3 (same channels) --> BN --> LeakyReLU
  |         |
  +----<----+ (residual addition)
  |
Output
```

### Encoder Architecture

```
Layer               Output Size      Channels    Details
----------------------------------------------------------------------
Input               64 x 2048        5           Range image
Conv 3x3            64 x 2048        32          stride=1, pad=1
Conv 3x3            32 x 1024        64          stride=2, pad=1 (downsample)
Residual Block x1   32 x 1024        64          DarkNet residual
Conv 3x3            16 x 512         128         stride=2 (downsample)
Residual Block x2   16 x 512         128         DarkNet residual
Conv 3x3            8 x 256          256         stride=2 (downsample)
Residual Block x8   8 x 256          256         DarkNet residual
Conv 3x3            4 x 128          512         stride=2 (downsample)
Residual Block x8   4 x 128          512         DarkNet residual
Conv 3x3            2 x 64           1024        stride=2 (downsample)
Residual Block x4   2 x 64           1024        DarkNet residual (bottleneck)
```

### Key Design Choices

1. **No max pooling:** All downsampling via strided convolutions (preserves more information).
2. **LeakyReLU:** `f(x) = max(0.1x, x)` instead of ReLU (prevents dead neurons).
3. **Batch Normalization:** After every convolution layer.
4. **Residual connections:** Enable training of deep network (53 layers).

### Layer Count Breakdown (53 layers)

- 1 initial conv layer
- 2 x (1 conv + 1 residual block) = 2 + 2x2 = 6
- 1 conv + 2 residual blocks = 1 + 2x2 = 5
- 1 conv + 8 residual blocks = 1 + 8x2 = 17
- 1 conv + 8 residual blocks = 1 + 8x2 = 17
- 1 conv + 4 residual blocks = 1 + 4x2 = 9
- **Total:** 1 + 6 + 5 + 17 + 17 + 9 = 53 convolutional layers (accounting for residual block convolutions)

---

## Stage 2: U-Net Decoder

### Overview

The decoder mirrors the encoder with upsampling operations and skip connections from the encoder at matching resolutions.

### Decoder Architecture

```
Layer               Input Size       Output Size     Details
----------------------------------------------------------------------
Upsample + Conv     2 x 64          4 x 128         Bilinear up + Conv 3x3
Skip Connection     + 4 x 128       4 x 128         Concat from encoder (512 ch)
Conv Block          4 x 128         4 x 128         Conv 3x3 + BN + ReLU
Upsample + Conv     4 x 128         8 x 256         Bilinear up + Conv 3x3
Skip Connection     + 8 x 256       8 x 256         Concat from encoder (256 ch)
Conv Block          8 x 256         8 x 256         Conv 3x3 + BN + ReLU
Upsample + Conv     8 x 256         16 x 512        Bilinear up + Conv 3x3
Skip Connection     + 16 x 512      16 x 512        Concat from encoder (128 ch)
Conv Block          16 x 512        16 x 512        Conv 3x3 + BN + ReLU
Upsample + Conv     16 x 512        32 x 1024       Bilinear up + Conv 3x3
Skip Connection     + 32 x 1024     32 x 1024       Concat from encoder (64 ch)
Conv Block          32 x 1024       32 x 1024       Conv 3x3 + BN + ReLU
Upsample + Conv     32 x 1024       64 x 2048       Bilinear up + Conv 3x3
Conv Block          64 x 2048       64 x 2048       Conv 3x3 + BN + ReLU
----------------------------------------------------------------------
Classification Head 64 x 2048       64 x 2048 x 20  Conv 1x1 (num_classes)
```

### Skip Connections

Skip connections concatenate encoder features with decoder features at matching spatial resolutions:
- Resolution 4x128: encoder features (512 ch) + decoder features
- Resolution 8x256: encoder features (256 ch) + decoder features
- Resolution 16x512: encoder features (128 ch) + decoder features
- Resolution 32x1024: encoder features (64 ch) + decoder features

After concatenation, a 1x1 convolution reduces the channel count.

### Upsampling Strategy

- **Method:** Bilinear interpolation followed by convolution (no transposed convolutions to avoid checkerboard artifacts).
- **Scale factor:** 2x in both height and width per stage.

---

## Stage 3: KNN Post-Processing

### Motivation

The range image projection introduces two types of artifacts when mapping predictions back to 3D:
1. **Discretization errors:** Multiple points in the same pixel get the same label.
2. **Boundary bleeding:** Labels near object boundaries may be incorrect due to the fixed-grid representation.

### Algorithm

```
Input: 
  - 3D points P = {p_1, ..., p_N}
  - Range image predictions L_range (H x W)
  - Projection mapping: point_i -> pixel (u_i, v_i)

For each point p_i:
  1. Get initial label: l_i = L_range[v_i, u_i]
  2. Find K nearest neighbors of p_i in 3D space: {n_1, ..., n_K}
  3. For each neighbor n_j, get its range image label: l_j
  4. Compute distance weight: w_j = 1 / (d(p_i, n_j) + epsilon)
  5. Weighted vote:
     For each class c:
       score_c = sum(w_j * [l_j == c] for j in 1..K)
  6. Final label: l_i = argmax_c(score_c)

Output: Refined 3D labels {l_1, ..., l_N}
```

### Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| K | 5 | Number of nearest neighbors |
| Search radius | 1.0 m | Maximum search distance |
| cutoff | 1.0 | Distance cutoff for weighting |
| epsilon | 1e-6 | Prevent division by zero |

### Implementation Details

1. **KD-Tree construction:** Build a KD-tree over all 3D points for efficient nearest neighbor search.
2. **GPU acceleration:** Use CUDA-based KNN search for real-time performance.
3. **Batch processing:** Process all points in parallel on GPU.
4. **Runtime:** ~5ms per scan on a modern GPU.

### Effect on Performance

| Method | mIoU (%) | Notes |
|--------|----------|-------|
| RangeNet53 (no KNN) | 49.9 | Raw range image predictions |
| RangeNet53++ (K=1) | 50.8 | Nearest point refinement |
| RangeNet53++ (K=5) | 52.2 | Best setting |
| RangeNet53++ (K=7) | 51.9 | Slight oversmoothing |
| RangeNet53++ (K=11) | 51.3 | Too much smoothing |

---

## Classification Head

### Structure

```
Decoder Output (64 x 2048 x C_decoder)
       |
       v
  Dropout (p=0.01)
       |
       v
  Conv 1x1 (C_decoder -> num_classes)
       |
       v
  Output Logits (64 x 2048 x 20)
```

### Output

- **Channels:** 20 (19 semantic classes + 1 unlabeled)
- **Activation:** None (raw logits); softmax applied during inference
- **Loss:** Weighted cross-entropy during training

---

## Model Summary

### Parameters

| Component | Parameters (approx.) |
|-----------|---------------------|
| DarkNet-53 Encoder | ~40M |
| U-Net Decoder | ~10M |
| Classification Head | ~0.02M |
| **Total** | **~50M** |

### Memory Requirements

| Configuration | Training (batch=4) | Inference (batch=1) |
|---------------|-------------------|---------------------|
| 64x2048 | ~8 GB GPU RAM | ~2 GB GPU RAM |
| 64x1024 | ~5 GB GPU RAM | ~1.5 GB GPU RAM |

### Computational Cost

| Stage | Time (ms) | Notes |
|-------|-----------|-------|
| Spherical Projection | 1-2 | CPU or GPU |
| DarkNet-53 Encoder | 12-15 | GPU (FP32) |
| U-Net Decoder | 5-7 | GPU (FP32) |
| KNN Post-Processing | 3-5 | GPU |
| **Total** | **~20-25** | Single NVIDIA GTX 1080Ti |

### Receptive Field

The encoder's 5 levels of 2x downsampling give the bottleneck features a theoretical receptive field covering the entire input image, allowing the network to capture global context (e.g., road surfaces span the full width of the range image).

---

## Alternative Backbones

### DarkNet-21 (Lighter Variant)

A smaller version with fewer residual blocks:
- 1 + 1 + 2 + 4 + 4 + 2 = 21 layers
- ~25M parameters
- ~14ms inference
- 47.4% mIoU (without KNN)

### SqueezeSeg / SqueezeSegV2

Earlier range-image methods using SqueezeNet-like backbones:
- Much fewer parameters (~1M)
- Faster inference but lower accuracy
- No skip connections in original versions

### Comparison

| Backbone | Params | Speed | mIoU |
|----------|--------|-------|------|
| SqueezeSeg | ~1M | 12ms | 29.5% |
| SqueezeSegV2 | ~1M | 15ms | 39.6% |
| DarkNet-21 | ~25M | 14ms | 47.4% |
| DarkNet-53 | ~50M | 20ms | 49.9% |
| DarkNet-53 + KNN | ~50M | 25ms | 52.2% |
