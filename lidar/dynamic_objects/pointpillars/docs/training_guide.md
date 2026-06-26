# PointPillars Training Guide

This document provides a comprehensive guide to training PointPillars models for 3D object detection from LiDAR point clouds. It covers anchor generation, data augmentation, learning rate scheduling, and practical tips for achieving state-of-the-art performance on KITTI and nuScenes benchmarks.

---

## 1. Anchor Generation Strategy

PointPillars uses predefined anchor boxes on the Bird's Eye View (BEV) feature map to detect 3D objects. The anchor generation strategy is critical for matching ground-truth boxes during training.

### 1.1 Anchor Placement on the BEV Grid

Anchors are placed at every spatial location of the BEV feature map. Given the feature map resolution (typically downsampled by 2x from the pseudo-image), each cell corresponds to a physical region in the point cloud space.

```
BEV feature map size: (H/2, W/2)
Anchor stride: 2 * voxel_size (e.g., 0.32m for KITTI with 0.16m pillars)
```

At each grid cell `(i, j)`, anchors are placed at the center of the corresponding physical location:

```python
x_center = x_min + (j + 0.5) * anchor_stride_x
y_center = y_min + (i + 0.5) * anchor_stride_y
```

### 1.2 Per-Class Anchor Sizes

Anchor dimensions are derived from the mean dimensions of each class in the training set:

| Class       | Length (m) | Width (m) | Height (m) |
|-------------|-----------|-----------|------------|
| Car         | 3.9       | 1.6       | 1.56       |
| Pedestrian  | 0.8       | 0.6       | 1.73       |
| Cyclist     | 1.76      | 0.6       | 1.73       |

For nuScenes, additional classes include:

| Class            | Length (m) | Width (m) | Height (m) |
|------------------|-----------|-----------|------------|
| Car              | 4.63      | 1.97      | 1.74       |
| Truck            | 6.93      | 2.51      | 2.84       |
| Bus              | 11.1      | 2.94      | 3.47       |
| Motorcycle       | 2.11      | 0.77      | 1.47       |
| Pedestrian       | 0.73      | 0.67      | 1.77       |

### 1.3 Two Rotations Per Anchor

At each spatial location, two anchors are generated per class with different orientations:

- **Rotation 0**: Aligned with the x-axis (heading forward)
- **Rotation pi/2**: Rotated 90 degrees (heading sideways)

This doubles the number of anchors but ensures objects at arbitrary orientations have at least one anchor with reasonable IoU overlap.

```python
anchor_rotations = [0, np.pi / 2]  # radians
```

### 1.4 Anchor Height Assignment

The z-center of each anchor is fixed per class based on the mean height of objects in the training set:

```python
anchor_z_centers = {
    'Car': -1.0,          # center of typical car above ground
    'Pedestrian': -0.6,   # center of typical pedestrian
    'Cyclist': -0.6,      # center of typical cyclist
}
```

This fixed assignment avoids searching over the z-dimension, which is reasonable for LiDAR scenes where objects rest on approximately flat ground.

### 1.5 IoU-Based Target Assignment

Each anchor is assigned as positive, negative, or ignored based on its 2D BEV IoU with ground-truth boxes:

| Assignment | Condition |
|-----------|-----------|
| Positive  | IoU >= positive_threshold (0.6 for Car, 0.5 for Ped/Cyc) |
| Negative  | IoU < negative_threshold (0.45 for Car, 0.35 for Ped/Cyc) |
| Ignore    | negative_threshold <= IoU < positive_threshold |

Additionally, for each ground-truth box, the anchor with the highest IoU is always assigned as positive regardless of threshold.

### 1.6 Residual Encoding Scheme

Positive anchors encode their regression targets as residuals relative to the matched anchor:

```
dx = (x_gt - x_a) / d_a
dy = (y_gt - y_a) / d_a
dz = (z_gt - z_a) / h_a
dw = log(w_gt / w_a)
dl = log(l_gt / l_a)
dh = log(h_gt / h_a)
dtheta = sin(theta_gt - theta_a)
```

Where:
- `d_a = sqrt(l_a^2 + w_a^2)` is the diagonal of the anchor in BEV
- `(x_a, y_a, z_a, w_a, l_a, h_a, theta_a)` are anchor parameters
- `(x_gt, y_gt, z_gt, w_gt, l_gt, h_gt, theta_gt)` are ground-truth parameters

The sine encoding for angle avoids the discontinuity at +/- pi. A separate direction classification head handles the sign ambiguity.

---

## 2. Data Augmentation

Data augmentation is essential for PointPillars to generalize well, especially given the limited diversity of driving datasets.

### 2.1 Ground-Truth Database Sampling (Copy-Paste)

This is the most impactful augmentation technique for 3D object detection.

**Pre-processing (offline):**

1. Extract all ground-truth objects from the training set
2. For each object, store:
   - The 3D bounding box parameters
   - All LiDAR points falling within the box
   - Class label and difficulty level

```python
# Database structure
gt_database = {
    'Car': [
        {'box3d': [...], 'points': np.array(...), 'difficulty': 0},
        ...
    ],
    'Pedestrian': [...],
    'Cyclist': [...],
}
```

**During training:**

1. Randomly sample N objects per class from the database:
   - Car: sample 15 instances
   - Pedestrian: sample 10 instances
   - Cyclist: sample 10 instances

2. For each sampled object, attempt to paste it into the current scene:
   - Place at random valid locations
   - Check for collisions with existing objects (BEV IoU < 0.0)
   - Check for collisions with other sampled objects
   - If no collision, add the object's points to the scene and its box to the label

3. Remove points from the original scene that fall within newly placed boxes (occlusion handling)

```python
def check_collision(new_box, existing_boxes, threshold=0.0):
    """Reject placement if BEV IoU exceeds threshold."""
    ious = bev_iou(new_box, existing_boxes)
    return np.all(ious <= threshold)
```

### 2.2 Global Rotation

Rotate the entire point cloud and all bounding boxes around the z-axis:

```python
rotation_angle = np.random.uniform(-np.pi/4, np.pi/4)

# Rotate points
rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle),  np.cos(angle), 0],
    [0,              0,             1]
])
points[:, :3] = points[:, :3] @ rotation_matrix.T

# Rotate box centers and headings
boxes[:, :2] = boxes[:, :2] @ rotation_matrix[:2, :2].T
boxes[:, 6] += rotation_angle  # update heading
```

### 2.3 Global Scaling

Scale all point coordinates and box dimensions by a uniform random factor:

```python
scale_factor = np.random.uniform(0.95, 1.05)

points[:, :3] *= scale_factor
boxes[:, :6] *= scale_factor  # scale x, y, z, w, l, h
# heading (boxes[:, 6]) remains unchanged
```

### 2.4 Global Translation

Shift the entire scene by a random offset along each axis:

```python
translation = np.random.normal(0, 0.2, size=3)  # std = 0.2m per axis

points[:, :3] += translation
boxes[:, :3] += translation  # shift box centers
```

### 2.5 Random Flip

Mirror the scene along the x-axis or y-axis (or both) with 50% probability each:

```python
# Flip along x-axis (left-right)
if np.random.random() > 0.5:
    points[:, 1] = -points[:, 1]
    boxes[:, 1] = -boxes[:, 1]
    boxes[:, 6] = -boxes[:, 6]  # negate heading

# Flip along y-axis (front-back)
if np.random.random() > 0.5:
    points[:, 0] = -points[:, 0]
    boxes[:, 0] = -boxes[:, 0]
    boxes[:, 6] = np.pi - boxes[:, 6]  # flip heading
```

### 2.6 Point Shuffling

Randomize the order of points within each pillar to prevent the network from relying on point ordering:

```python
for pillar in pillars:
    np.random.shuffle(pillar.point_indices)
```

This ensures permutation invariance is learned rather than imposed solely by the architecture.

### Augmentation Pipeline Order

The augmentations are applied in this sequence:

1. GT database sampling (copy-paste)
2. Random flip
3. Global rotation
4. Global scaling
5. Global translation
6. Point shuffling (during pillar creation)

---

## 3. One-Cycle Learning Rate Policy

PointPillars uses the one-cycle learning rate policy for training, which enables super-convergence.

### 3.1 Schedule Structure

```
|-- Warmup (40%) --|------- Cosine Decay (60%) -------|
lr/10 --> max_lr      max_lr --> lr/1000
```

- **Phase 1 (0-40% of training):** Linear warmup from `max_lr / 10` to `max_lr`
- **Phase 2 (40-100% of training):** Cosine annealing from `max_lr` to `max_lr / 1000`

### 3.2 Implementation

```python
from torch.optim.lr_scheduler import OneCycleLR

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)

scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=total_steps,
    pct_start=0.4,           # 40% warmup
    anneal_strategy='cos',   # cosine decay
    div_factor=10,           # initial_lr = max_lr / 10
    final_div_factor=1000,   # final_lr = max_lr / 1000
)
```

### 3.3 Why One-Cycle Works Well

- **Super-convergence:** The high learning rate in the middle of training allows the optimizer to escape sharp minima and find flatter loss landscapes that generalize better.
- **Implicit regularization:** Large learning rates act as regularizers by injecting noise into the optimization process, reducing the need for explicit regularization like dropout.
- **Faster training:** One-cycle often converges in fewer epochs than traditional step-decay schedules.
- **Momentum coupling:** The original one-cycle policy also cycles momentum inversely to learning rate (high lr = low momentum, low lr = high momentum), further improving convergence.

### 3.4 Typical max_lr Values

| Dataset   | max_lr  | Optimizer | Weight Decay |
|-----------|---------|-----------|--------------|
| KITTI     | 2e-3    | AdamW     | 0.01         |
| nuScenes  | 1e-3    | AdamW     | 0.01         |

The lower learning rate for nuScenes accounts for larger batch sizes and the more complex multi-class detection task.

---

## 4. Gradient Clipping

### 4.1 Clip by Norm

Gradient clipping prevents exploding gradients that can destabilize training, particularly in the early epochs when the network outputs are far from ground truth:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
```

| Dataset   | max_norm |
|-----------|----------|
| KITTI     | 10       |
| nuScenes  | 35       |

### 4.2 When Gradient Clipping Matters

- **Early training:** Before the network learns reasonable predictions, gradients from the regression loss can be extremely large.
- **Rare classes:** Sudden large gradients from hard examples of rare classes (pedestrians, cyclists) can destabilize the entire network.
- **GT database sampling:** Pasting many objects into a scene can create unusually dense training examples with large aggregate gradients.

The higher clip norm for nuScenes (35 vs 10) accommodates the larger number of classes and more complex scenes, where gradients are naturally larger.

---

## 5. Class-Balanced Sampling

### 5.1 The Class Imbalance Problem

Autonomous driving datasets exhibit severe class imbalance:

| Class       | KITTI (approx. instances) | Ratio |
|-------------|--------------------------|-------|
| Car         | 28,000                   | 1.0x  |
| Pedestrian  | 4,500                    | 0.16x |
| Cyclist     | 1,600                    | 0.06x |

Without mitigation, the network becomes biased toward the dominant class (Car) and fails on rare classes.

### 5.2 GT Database Sampling for Balance

The copy-paste augmentation (Section 2.1) directly addresses imbalance by sampling more instances of rare classes:

```python
sample_counts = {
    'Car': 15,          # Already abundant, moderate sampling
    'Pedestrian': 10,   # Rare, aggressive sampling
    'Cyclist': 10,      # Very rare, aggressive sampling
}
```

This effectively oversamples rare classes at the scene level, ensuring the network sees sufficient examples during training.

### 5.3 Focal Loss for Hard Negatives

The classification head uses focal loss to down-weight easy negatives (empty anchors far from any object):

```python
focal_loss = -alpha * (1 - p)^gamma * log(p)
```

With `alpha = 0.25` and `gamma = 2.0`:

- Easy negatives (p close to 0) get weight close to 0
- Hard negatives and all positives receive full weight
- This focuses the classification loss on informative examples

### 5.4 Combined Effect

The three mechanisms work together:

1. **GT database sampling** ensures rare classes appear frequently in training scenes
2. **Focal loss** ensures the classification loss focuses on hard examples
3. **Per-class anchor thresholds** (lower IoU thresholds for small objects) ensure rare/small objects can match anchors

---

## 6. KITTI vs nuScenes Training Differences

### 6.1 Configuration Comparison

| Parameter              | KITTI                    | nuScenes                   |
|-----------------------|--------------------------|----------------------------|
| Epochs                | 80                       | 20                         |
| Batch size            | 6                        | 4                          |
| Point cloud range (x) | [0, 69.12] m            | [-51.2, 51.2] m           |
| Point cloud range (y) | [-39.68, 39.68] m       | [-51.2, 51.2] m           |
| Point cloud range (z) | [-3, 1] m               | [-5, 3] m                 |
| Coverage              | Front-view only (90 deg) | 360-degree                 |
| Number of classes     | 3                        | 10                         |
| Pillar size (x, y)    | 0.16 m                  | 0.2 m                     |
| Max pillars           | 12,000                   | 30,000                     |
| Max points/pillar     | 100                      | 20                         |
| Input sweeps          | 1 (single frame)        | 10 (multi-sweep)           |
| Velocity prediction   | No                       | Yes (vx, vy)              |
| max_lr                | 2e-3                     | 1e-3                       |
| Gradient clip norm    | 10                       | 35                         |

### 6.2 nuScenes Multi-Sweep Input

nuScenes concatenates points from multiple LiDAR sweeps (typically 10) to increase point density:

```python
# Each point has: x, y, z, intensity, time_lag
# time_lag indicates which sweep the point came from (0 = current, negative = past)
point_features = 5  # (x, y, z, intensity, time_lag)
```

### 6.3 Velocity Prediction (nuScenes Only)

The regression head predicts two additional values (vx, vy) representing object velocity:

```python
# nuScenes box encoding: (dx, dy, dz, dw, dl, dh, dtheta, dvx, dvy)
regression_channels = 9  # vs 7 for KITTI
```

Velocity is supervised using the difference between current and previous annotations.

### 6.4 Fade Strategy (nuScenes)

In the last 5 epochs of nuScenes training, data augmentation is gradually disabled:

```python
if epoch >= (total_epochs - 5):
    disable_gt_sampling = True
    disable_global_augmentation = True
```

This allows the network to fine-tune on clean, unaugmented data distribution, improving final performance by 1-2 mAP.

---

## 7. Memory and Compute Requirements

### 7.1 Hardware Requirements

| Resource              | KITTI                   | nuScenes                 |
|-----------------------|-------------------------|--------------------------|
| GPU memory per sample | ~4 GB                   | ~8 GB                    |
| Recommended GPUs      | 4x V100 (32 GB)        | 8x V100 (32 GB)         |
| Training time         | ~4 hours                | ~20 hours                |
| Inference speed       | ~5 ms/frame (62 Hz)    | ~8 ms/frame (125 Hz theoretical, limited by sensor rate) |

### 7.2 Inference Latency Breakdown

| Component                | Time (ms) | Percentage |
|--------------------------|-----------|------------|
| Pillar Feature Net       | 0.5       | 10%        |
| Scatter to pseudo-image  | 0.1       | 2%         |
| Backbone (2D CNN)        | 3.2       | 64%        |
| SSD Detection Head       | 0.7       | 14%        |
| NMS post-processing      | 0.5       | 10%        |
| **Total**                | **~5.0**  | **100%**   |

### 7.3 Bottleneck Analysis

- **Backbone dominates:** The 2D CNN backbone (typically a modified ResNet or VGG variant) accounts for ~64% of inference time. This is because it processes the dense pseudo-image across multiple resolution levels.
- **Pillar Feature Net is efficient:** Despite processing variable numbers of points, the PointNet-like pillar feature extraction is fast due to simple MLP operations.
- **NMS is fixed-cost:** Non-maximum suppression operates on a relatively small number of confident detections and adds negligible latency.

### 7.4 Memory Optimization Tips

- Use mixed precision training (FP16) to reduce memory by ~40%
- Reduce max_pillars if GPU memory is limited (at slight accuracy cost)
- Use gradient accumulation to simulate larger batch sizes on fewer GPUs

```python
# Mixed precision training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 8. Tips and Common Issues

### 8.1 Pretrained Backbone (Optional)

While PointPillars can be trained from scratch, using a backbone pretrained on ImageNet (adapted for the pseudo-image input) can improve convergence speed:

- Pretrained initialization helps when training data is limited
- For KITTI (small dataset), pretraining can improve AP by 1-2%
- For nuScenes (large dataset), the benefit is marginal
- Ensure the first convolutional layer is re-initialized (different input channels)

### 8.2 Monitoring Class-Wise AP

Track per-class Average Precision (AP) during validation:

```python
# Log per-class metrics every N epochs
metrics = evaluate(model, val_loader)
for cls in ['Car', 'Pedestrian', 'Cyclist']:
    log(f'{cls}_AP_BEV_R40: {metrics[cls]["bev"]:.2f}')
    log(f'{cls}_AP_3D_R40: {metrics[cls]["3d"]:.2f}')
```

Watch for:
- Car AP converging first (most data, largest objects)
- Pedestrian/Cyclist AP lagging behind (less data, smaller objects)
- Large gap between BEV AP and 3D AP indicating height estimation issues

### 8.3 Learning Rate Tuning

**If loss explodes:**
- Reduce `max_lr` by 2-5x
- Increase warmup percentage (e.g., from 40% to 50%)
- Reduce gradient clip norm

**If loss plateaus early:**
- Increase `max_lr` by 1.5-2x
- Ensure augmentation is working correctly
- Check that positive anchors are being generated (verify IoU thresholds)

**If training is unstable (oscillating loss):**
- Reduce batch size
- Increase gradient clip norm slightly
- Check for NaN values in point cloud data

### 8.4 Batch Size vs Learning Rate Scaling

Follow the linear scaling rule when adjusting batch size:

```
new_lr = base_lr * (new_batch_size / base_batch_size)
```

| Batch Size | max_lr (KITTI) |
|-----------|----------------|
| 2         | 0.67e-3        |
| 4         | 1.33e-3        |
| 6 (base)  | 2.0e-3         |
| 8         | 2.67e-3        |
| 12        | 4.0e-3         |

For very large batch sizes (>16), consider using a square root scaling rule instead of linear to avoid instability.

### 8.5 Common Failure Modes

| Issue | Symptom | Solution |
|-------|---------|----------|
| No detections | AP = 0 for all classes | Check anchor sizes match data, verify positive anchor assignment |
| Only detecting cars | Ped/Cyc AP = 0 | Increase GT sampling for rare classes, lower IoU thresholds |
| Poor localization | High recall but low AP | Check regression target encoding, verify loss weights |
| NaN loss | Training crashes | Reduce learning rate, add epsilon to log operations, check input data |
| Overfitting | Train AP >> Val AP | Increase augmentation strength, add dropout, reduce model size |
| Poor orientation | High AP but wrong headings | Check direction classification head, verify angle encoding |

### 8.6 Recommended Training Workflow

1. **Verify data loading:** Visualize a few training samples with augmentation to confirm correctness
2. **Sanity check:** Overfit on a single batch to ensure the model can memorize
3. **Short run:** Train for 10% of total epochs with aggressive monitoring
4. **Full training:** Run complete training with validation every 5 epochs
5. **Evaluate:** Compute AP on the full validation set with official evaluation code
6. **Tune:** Adjust hyperparameters based on per-class results

```bash
# Example training command (OpenPCDet framework)
python train.py \
    --cfg_file cfgs/kitti_models/pointpillar.yaml \
    --batch_size 6 \
    --epochs 80 \
    --workers 8 \
    --extra_tag experiment_v1
```

---

## Summary

Training PointPillars effectively requires attention to several interrelated components:

- **Anchor design** must match the target class statistics
- **Data augmentation** (especially GT database sampling) is critical for handling class imbalance and limited data diversity
- **One-cycle learning rate** enables fast, stable convergence
- **Gradient clipping** prevents training instabilities
- **Dataset-specific tuning** (KITTI vs nuScenes) accounts for differences in scale, coverage, and class complexity

When all components are properly configured, PointPillars achieves strong 3D detection performance while maintaining real-time inference speed (~62 Hz), making it one of the most practical architectures for autonomous driving perception pipelines.
