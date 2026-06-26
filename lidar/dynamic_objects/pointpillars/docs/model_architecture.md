# PointPillars Model Architecture

## 1. Overview / Data Flow

PointPillars converts raw LiDAR point clouds into 3D object detections by encoding point cloud data into a Bird's Eye View (BEV) pseudo-image, then applying standard 2D convolutional detection networks.

```
+-------------+     +--------------+     +-----------------+     +---------+
| Point Cloud | --> | Pillarization| --> | Pillar Feature  | --> | Scatter |
| (N, 4)     |     | (grid disc.) |     | Net (PointNet)  |     |         |
+-------------+     +--------------+     +-----------------+     +---------+
                                                                       |
                                                                       v
+------------+     +------+     +-------------+     +------------------------------+
| Detection  | <-- | Neck | <-- | 2D Backbone | <-- | Pseudo-Image (B, 64, H, W)  |
| Head (SSD) |     | (FPN)|     | (CNN)       |     |                              |
+------------+     +------+     +-------------+     +------------------------------+
       |
       v
+-------------------+
| 3D Bounding Boxes |
| (x,y,z,w,l,h,yaw)|
+-------------------+
```

**End-to-end pipeline:**

```
Point Cloud -> Pillarization -> Pillar Feature Net -> Scatter -> Pseudo-Image
           -> 2D Backbone -> Neck (FPN) -> Detection Head -> NMS -> 3D Boxes
```

---

## 2. Pillar Feature Net

The Pillar Feature Net converts the raw, unordered point cloud into a structured set of learned pillar features using a simplified PointNet.

### 2.1 Pillarization

The x-y plane is discretized into a uniform grid of pillars (vertical columns with infinite extent in z).

- **Grid resolution (KITTI):** x: [-39.68, 39.68] m, y: [0, 69.12] m
- **Pillar size:** 0.16 m x 0.16 m
- **Grid dimensions:** 496 (x) x 432 (y) pillars
- Each LiDAR point is assigned to the pillar whose x-y cell it falls into
- Points are randomly sampled if a pillar exceeds `max_points_per_pillar` (typically 32 or 100)
- Non-empty pillars are capped at `max_pillars` (typically 12000 or 16000)

### 2.2 Point Feature Augmentation

Each point within a pillar is augmented from 4 raw features to 9 features:

| Feature | Description |
|---------|-------------|
| x, y, z | Original 3D coordinates |
| intensity | LiDAR reflectance value |
| xc, yc, zc | Offset from arithmetic mean of all points in the pillar |
| xp, yp | Offset from the pillar's geometric center (x-y grid cell center) |

```
Original point:     (x, y, z, intensity)          -> 4 features
+ Pillar mean offset: (x-xmean, y-ymean, z-zmean) -> +3 features
+ Pillar center offset: (x-xcenter, y-ycenter)     -> +2 features
                                                   = 9 features total
```

### 2.3 PointNet Encoding

A simplified PointNet is applied independently to each pillar:

```
Input: (B, max_pillars, max_points, 9)
         |
         v
  Linear(9, 64)          -- fully connected, shared across all points
         |
         v
  BatchNorm1d(64)        -- normalize features
         |
         v
  ReLU                   -- non-linear activation
         |
         v
  MaxPool(dim=2)         -- pool across points dimension (max_points)
         |
         v
Output: (B, max_pillars, 64)
```

### 2.4 Tensor Shape Summary

```
Input point cloud:           (N_points, 4)         [variable per sample]
After pillarization:         (B, P, N, 9)          P=max_pillars, N=max_points
After Linear + BN + ReLU:    (B, P, N, 64)
After MaxPool over points:   (B, P, 64)            one vector per pillar
```

Where:
- `B` = batch size
- `P` = max_pillars (e.g., 12000)
- `N` = max_points_per_pillar (e.g., 32)

---

## 3. Scatter Operation

The scatter operation is the key bridge between the irregular point cloud representation and dense 2D convolutions.

### 3.1 Mechanism

Each pillar's 64-dimensional feature vector is placed ("scattered") back to its corresponding (x, y) position in the BEV grid:

```
Pillar features: (B, P, 64)     +     Pillar coordinates: (B, P, 2)
                                  |
                                  v
                    Pseudo-Image: (B, 64, H, W)
```

### 3.2 Implementation Details

```python
# Pseudo-code for scatter
pseudo_image = torch.zeros(B, 64, H, W)  # (B, 64, 496, 432) for KITTI

for each non-empty pillar i:
    x_idx, y_idx = pillar_coords[i]
    pseudo_image[:, :, x_idx, y_idx] = pillar_features[:, i, :]
```

### 3.3 Key Properties

- **Output shape (KITTI):** `(B, 64, 496, 432)`
- **Empty pillars remain zero** -- most of the pseudo-image is sparse
- **Key insight:** This converts the irregular, unordered point cloud into a dense, regularly-structured tensor that can be processed by standard 2D convolutional neural networks
- **No learnable parameters** -- scatter is a purely geometric placement operation
- **Efficient:** Only non-empty pillars need processing; empty space costs no compute in the Pillar Feature Net

---

## 4. 2D CNN Backbone

The backbone is a series of convolutional blocks that progressively downsample the pseudo-image while increasing channel depth.

### 4.1 Architecture

```
Input: (B, 64, 496, 432)
         |
         v
+------------------+
| Block 1          |
| Stride-2 conv   |
| + 3x Conv-BN-ReLU|
| Channels: 64    |
+------------------+
         |
Output: (B, 64, 248, 216)
         |
         v
+------------------+
| Block 2          |
| Stride-2 conv   |
| + 5x Conv-BN-ReLU|
| Channels: 128   |
+------------------+
         |
Output: (B, 128, 124, 108)
         |
         v
+------------------+
| Block 3          |
| Stride-2 conv   |
| + 5x Conv-BN-ReLU|
| Channels: 256   |
+------------------+
         |
Output: (B, 256, 62, 54)
```

### 4.2 Block Details

Each block consists of:

1. **Initial stride-2 convolution:** Downsamples spatial dimensions by 2x and changes channel count
2. **Subsequent stride-1 convolutions:** Maintain spatial dimensions, refine features

```
Block structure:
  Conv2d(C_in, C_out, kernel=3, stride=2, padding=1) -> BN -> ReLU
  Conv2d(C_out, C_out, kernel=3, stride=1, padding=1) -> BN -> ReLU  (repeated S times)
```

| Block | Input Shape | Output Shape | Channels | Num Layers (stride-1) | Total Convs |
|-------|-------------|--------------|----------|-----------------------|-------------|
| 1 | (B, 64, 496, 432) | (B, 64, 248, 216) | 64 | 3 | 4 |
| 2 | (B, 64, 248, 216) | (B, 128, 124, 108) | 128 | 5 | 6 |
| 3 | (B, 128, 124, 108) | (B, 256, 62, 54) | 256 | 5 | 6 |

---

## 5. Neck (Multi-Scale Feature Fusion)

The neck (Feature Pyramid Network style) upsamples multi-scale backbone features to a common resolution and concatenates them.

### 5.1 Architecture

```
Block 1 output                Block 2 output               Block 3 output
(B, 64, 248, 216)            (B, 128, 124, 108)           (B, 256, 62, 54)
       |                            |                            |
       v                            v                            v
  DeconvTranspose              DeconvTranspose              DeconvTranspose
  stride=1 (x1)               stride=2 (x2)               stride=4 (x4)
  64 -> 128 channels           128 -> 128 channels          256 -> 128 channels
       |                            |                            |
       v                            v                            v
(B, 128, 248, 216)           (B, 128, 248, 216)           (B, 128, 248, 216)
       |                            |                            |
       +----------------------------+----------------------------+
                                    |
                                    v
                            Concatenate (dim=1)
                                    |
                                    v
                          (B, 384, 248, 216)
```

### 5.2 Upsampling Details

| Source | Upsample Factor | Transposed Conv Stride | Input Channels | Output Channels | Output Shape |
|--------|-----------------|------------------------|----------------|-----------------|--------------|
| Block 1 | x1 | 1 | 64 | 128 | (B, 128, 248, 216) |
| Block 2 | x2 | 2 | 128 | 128 | (B, 128, 248, 216) |
| Block 3 | x4 | 4 | 256 | 128 | (B, 128, 248, 216) |

Each upsampling path: `ConvTranspose2d -> BatchNorm2d -> ReLU`

**Concatenated output:** `(B, 128*3, 248, 216) = (B, 384, 248, 216)`

---

## 6. SSD Detection Head

The detection head predicts per-anchor classification scores, bounding box regressions, and direction classifications.

### 6.1 Anchor Design

Anchors are pre-defined 3D boxes placed at every spatial location of the feature map:

- **Classes (KITTI):** Car, Pedestrian, Cyclist (3 classes)
- **Rotations per class:** 2 (0 and 90 degrees)
- **Total anchors per spatial location:** `num_classes * num_rotations = 3 * 2 = 6`
- **Anchor dimensions:** Pre-defined (w, l, h, z_center) per class based on dataset statistics

### 6.2 Per-Anchor Predictions

For each anchor at each spatial location, the head predicts:

| Output | Dimensions | Description |
|--------|-----------|-------------|
| Classification | num_classes | Class probability (focal loss) |
| Box Regression | 7 | (dx, dy, dz, dw, dl, dh, d_theta) residuals |
| Direction | 2 | Binary classification for heading disambiguation |

### 6.3 Output Tensor Shapes

```
Input from Neck: (B, 384, 248, 216)
         |
         +----> Conv2d -> cls_pred:  (B, num_anchors * num_classes, 248, 216)
         |                           = (B, 18, 248, 216)  [6 anchors * 3 classes]
         |
         +----> Conv2d -> box_pred:  (B, num_anchors * 7, 248, 216)
         |                           = (B, 42, 248, 216)  [6 anchors * 7 params]
         |
         +----> Conv2d -> dir_pred:  (B, num_anchors * 2, 248, 216)
                                     = (B, 12, 248, 216)  [6 anchors * 2 bins]
```

### 6.4 Target Assignment

Training targets are assigned via IoU-based matching:

- **Positive match:** anchor-to-GT IoU >= positive threshold (e.g., 0.6 for Car)
- **Negative match:** anchor-to-GT IoU < negative threshold (e.g., 0.45 for Car)
- **Ignored:** anchors between thresholds are excluded from loss computation
- IoU is computed on BEV (Bird's Eye View) rotated bounding boxes

### 6.5 Loss Functions

```
Total Loss = (1/N_pos) * [L_cls + beta_reg * L_reg + beta_dir * L_dir]

where:
  L_cls  = Focal Loss (handles class imbalance)
  L_reg  = SmoothL1 Loss (box regression residuals)
  L_dir  = Binary Cross-Entropy (direction classification)
```

---

## 7. Post-Processing

### 7.1 Pipeline

```
Raw predictions -> Score Threshold -> Decode Boxes -> Direction Fix -> NMS -> Final Detections
```

### 7.2 Score Thresholding

- Apply sigmoid to classification logits
- Filter out predictions below score threshold (e.g., 0.1)
- Retain top-K candidates per class (e.g., K=500) for efficiency

### 7.3 Box Decoding

Predicted residuals are decoded relative to anchor boxes:

```
x = dx * diagonal + x_anchor
y = dy * diagonal + y_anchor
z = dz * h_anchor + z_anchor
w = exp(dw) * w_anchor
l = exp(dl) * l_anchor
h = exp(dh) * h_anchor
theta = d_theta + theta_anchor
```

Where `diagonal = sqrt(w_anchor^2 + l_anchor^2)`.

### 7.4 Direction Classification

The direction classifier resolves the 180-degree heading ambiguity inherent in box regression:

- The regression head can only predict theta within a limited range
- Direction classification provides a binary signal: is the heading in [0, pi) or [pi, 2*pi)?
- If direction bin disagrees with regressed angle, flip heading by pi

### 7.5 Rotated NMS

- Non-Maximum Suppression is applied per class using rotated IoU
- IoU threshold typically 0.01-0.1 (class-dependent)
- Computes overlap between oriented BEV rectangles
- Keeps highest-confidence detection, suppresses overlapping lower-confidence duplicates

### 7.6 Final Output

```
Per detection:
  - 3D bounding box: (x, y, z, width, length, height, heading)  [7 params]
  - Class label: Car / Pedestrian / Cyclist
  - Confidence score: [0, 1]
```

---

## Complete Tensor Flow Summary

```
Stage                         | Tensor Shape                    | Notes
------------------------------|---------------------------------|---------------------------
Raw point cloud               | (N, 4)                         | Variable N per scan
After pillarization           | (B, 12000, 32, 9)              | Fixed size, zero-padded
After PFN Linear+BN+ReLU     | (B, 12000, 32, 64)             | Per-point features
After PFN MaxPool             | (B, 12000, 64)                 | Per-pillar features
After Scatter                 | (B, 64, 496, 432)              | BEV pseudo-image
After Backbone Block 1        | (B, 64, 248, 216)              | 2x downsample
After Backbone Block 2        | (B, 128, 124, 108)             | 4x downsample
After Backbone Block 3        | (B, 256, 62, 54)               | 8x downsample
After Neck upsample + concat  | (B, 384, 248, 216)             | Multi-scale fusion
Detection Head - cls           | (B, 18, 248, 216)              | 6 anchors * 3 classes
Detection Head - reg           | (B, 42, 248, 216)              | 6 anchors * 7 box params
Detection Head - dir           | (B, 12, 248, 216)              | 6 anchors * 2 direction bins
After NMS                     | List[(M, 9)]                   | M detections: box7+cls+score
```

---

## References

- Lang, A. H., et al. "PointPillars: Fast Encoders for Object Detection from Point Clouds." CVPR 2019.
- Implementation reference: OpenPCDet, MMDetection3D
