# PointPillars Training Guide

This document provides a comprehensive guide to training PointPillars models for 3D
object detection from LiDAR point clouds. It is written for readers who are new to
training deep learning models for autonomous driving and covers all aspects from
hardware setup to debugging failed training runs.

---

## 1. Prerequisites

### 1.1 GPU Setup

Training PointPillars requires a CUDA-capable NVIDIA GPU. The model is relatively
lightweight compared to other 3D detectors, making it accessible on consumer hardware:

| GPU | VRAM | Batch Size | Training Time (KITTI) |
|-----|:----:|:----------:|:---------------------:|
| RTX 2080 Ti | 11 GB | 4 | ~6 hours |
| RTX 3090 | 24 GB | 8 | ~3 hours |
| V100 (32 GB) | 32 GB | 8 | ~4 hours |
| A100 (40 GB) | 40 GB | 16 | ~2 hours |
| RTX 3060 (budget) | 12 GB | 2-3 | ~10 hours |

Minimum requirements:
- NVIDIA GPU with at least 8 GB VRAM
- CUDA 11.x and cuDNN 8.x installed
- At least 32 GB system RAM (for data loading)
- SSD storage recommended (for faster data I/O)

### 1.2 Understanding the Training Loop

For readers unfamiliar with neural network training, here is the basic structure:

```
for epoch in range(total_epochs):          # Repeat over entire dataset
    for batch in dataloader:               # Process data in small batches
        predictions = model(batch)         # Forward pass: compute predictions
        loss = compute_loss(predictions, ground_truth)  # How wrong are we?
        loss.backward()                    # Backward pass: compute gradients
        optimizer.step()                   # Update model weights
        scheduler.step()                   # Update learning rate

    if epoch % eval_interval == 0:
        evaluate(model, val_set)           # Check performance on held-out data
```

Key terms:
- **Epoch:** One complete pass through all training data
- **Batch:** A small group of samples processed together (e.g., 4-8 point clouds)
- **Loss:** A number measuring how wrong the predictions are (lower = better)
- **Gradient:** The direction to adjust weights to reduce loss
- **Learning rate:** How large each weight adjustment is

---

## 2. Anchor Generation Strategy

### 2.1 Why Anchors?

PointPillars uses an anchor-based detection approach. Instead of predicting bounding
boxes from scratch, the network predicts small adjustments to predefined box templates
called anchors. This design choice has important benefits:

**Strong geometric prior:** Objects of the same class have predictable sizes. A car is
almost always approximately 3.9m x 1.6m x 1.56m. By starting from this template, the
network only needs to learn small corrections (e.g., "+0.2m longer than typical").

**Simplified regression:** Predicting small residuals (offsets from a template) is
mathematically easier for a neural network than predicting absolute values. The
residuals cluster around zero, which is a stable regime for gradient-based optimization.

**Multi-class support:** Different anchor sizes for each class allow simultaneous
detection of vastly different objects (a 4m car and a 0.8m pedestrian) with the same
detection head.

### 2.2 How Anchor Sizes Are Chosen

Anchor sizes come from computing statistics over the training set:

```python
# For each class, compute mean dimensions across all training instances:
car_stats = {
    'length_mean': 3.9,    # meters
    'width_mean': 1.6,
    'height_mean': 1.56,
    'z_center_mean': -1.0  # center height above ground
}
pedestrian_stats = {
    'length_mean': 0.8,
    'width_mean': 0.6,
    'height_mean': 1.73,
    'z_center_mean': -0.6
}
cyclist_stats = {
    'length_mean': 1.76,
    'width_mean': 0.6,
    'height_mean': 1.73,
    'z_center_mean': -0.6
}
```

These statistics ensure anchors closely match the actual object sizes, minimizing the
residuals the network must predict.

### 2.3 Worked Example: Anchors at Grid Cell (i, j)

Consider grid cell (i=100, j=150) on the 248x216 feature map.

```
Physical location:
  x_center = x_min + (j + 0.5) * anchor_stride_x = 0.0 + 150.5 * 0.32 = 48.16 m
  y_center = y_min + (i + 0.5) * anchor_stride_y = -39.68 + 100.5 * 0.32 = -7.52 m

At this location, 6 anchors are placed:

Anchor 1: Car, rotation=0
  box = (48.16, -7.52, -1.0, 1.6, 3.9, 1.56, 0.0)
         x      y      z    w    l    h     theta

Anchor 2: Car, rotation=pi/2
  box = (48.16, -7.52, -1.0, 1.6, 3.9, 1.56, pi/2)

Anchor 3: Pedestrian, rotation=0
  box = (48.16, -7.52, -0.6, 0.6, 0.8, 1.73, 0.0)

Anchor 4: Pedestrian, rotation=pi/2
  box = (48.16, -7.52, -0.6, 0.6, 0.8, 1.73, pi/2)

Anchor 5: Cyclist, rotation=0
  box = (48.16, -7.52, -0.6, 0.6, 1.76, 1.73, 0.0)

Anchor 6: Cyclist, rotation=pi/2
  box = (48.16, -7.52, -0.6, 0.6, 1.76, 1.73, pi/2)
```

### 2.4 IoU Target Assignment

During training, each anchor is labeled by comparing its overlap (IoU) with ground-truth
boxes:

```
                     IoU with GT boxes
                     |
    Negative         | Ignored        | Positive
    (background)     | (excluded)     | (contains object)
    |<-------|------->|<----->|<------>|
    0       0.35    0.45    0.5     0.6     1.0
            ^Ped     ^Car    ^Ped    ^Car
            neg_thr  neg_thr pos_thr pos_thr

For Car class:
  IoU >= 0.6  --> POSITIVE (this anchor matches a car)
  IoU < 0.45  --> NEGATIVE (this anchor is background)
  0.45 <= IoU < 0.6 --> IGNORED (ambiguous, excluded from loss)

For Pedestrian/Cyclist:
  IoU >= 0.5  --> POSITIVE
  IoU < 0.35  --> NEGATIVE
  0.35 <= IoU < 0.5 --> IGNORED
```

Additionally, for each ground-truth object, the single best-matching anchor (highest IoU)
is always forced to be positive, ensuring every object has at least one anchor assigned.

### 2.5 Residual Encoding

Positive anchors encode their regression targets as offsets from the anchor:

```
Given:
  Ground truth: (x_gt, y_gt, z_gt, w_gt, l_gt, h_gt, theta_gt)
  Matched anchor: (x_a, y_a, z_a, w_a, l_a, h_a, theta_a)
  Anchor diagonal: d_a = sqrt(l_a^2 + w_a^2)

Regression targets:
  dx = (x_gt - x_a) / d_a          (position offset, normalized by diagonal)
  dy = (y_gt - y_a) / d_a          (position offset, normalized by diagonal)
  dz = (z_gt - z_a) / h_a          (height offset, normalized by anchor height)
  dw = log(w_gt / w_a)             (log-ratio of widths)
  dl = log(l_gt / l_a)             (log-ratio of lengths)
  dh = log(h_gt / h_a)             (log-ratio of heights)
  dtheta = sin(theta_gt - theta_a)  (angular difference via sine)
```

Why this encoding works:
- Position offsets normalized by diagonal make the targets scale-invariant
- Log-ratios for dimensions ensure positivity and symmetric treatment of
  larger/smaller objects
- Sine encoding for angle avoids discontinuity at +/- pi
- All targets cluster around zero, which neural networks learn efficiently

---

## 3. Data Augmentation

Data augmentation is critical for PointPillars to generalize well. Without augmentation,
the network overfits to the limited diversity of driving datasets (KITTI has only 3,712
training samples).

### 3.1 Ground-Truth Database Sampling (Most Important)

This is the single most impactful augmentation technique. It "copy-pastes" objects from
other training scenes into the current scene, dramatically increasing diversity.

```
BEFORE GT Database Sampling:            AFTER GT Database Sampling:

   Scene with 2 cars:                   Scene with 2 original + 3 pasted cars,
                                        2 pasted pedestrians:
   ::::::::::::::::::::::::             ::::::::::::::::::::::::
   :::::  **  :::::::::::::             :::## **  :::::::::::::
   ::::: **** :::::::::::::             ::: ## **** ::: ** ::::
   :::::  **  :::::::::::::             :::    **  :::  ** ::::
   ::::::::::::::::::::::::             :: ## :::::::::: ** ::::
   :::::::::  ***  ::::::::             :: ##  :: *** :: ** ::::
   ::::::::: ***** ::::::::             :::::: :: *** ::::  ::::
   :::::::::  ***  ::::::::             :::::: ::  *  ::::::::::
   ::::::::::::::::::::::::             ::::::::::::::::::::::::

   * = original car                     * = original + pasted cars
   # = pasted pedestrians               # = pasted pedestrians
```

**How it works:**

1. **Offline preparation:** Extract all ground-truth objects from the training set. For
   each object, store its 3D bounding box and all LiDAR points within it.

2. **During training:** For each training scene:
   - Randomly sample objects from the database (15 cars, 10 pedestrians, 10 cyclists)
   - For each sampled object, attempt to place it at a valid location
   - Check for collisions with existing objects (reject if BEV IoU > 0)
   - Add the object's points to the scene and its label to the annotations
   - Remove original scene points that fall within newly placed boxes

```python
sample_counts = {
    'Car': 15,          # Paste up to 15 additional cars
    'Pedestrian': 10,   # Paste up to 10 additional pedestrians
    'Cyclist': 10,      # Paste up to 10 additional cyclists
}
```

**Why this is so effective:**
- Directly addresses class imbalance (more rare objects per scene)
- Increases scene diversity beyond what exists in the dataset
- Objects are "real" (actual LiDAR measurements, not synthetic)
- Provides the network with varied object contexts

### 3.2 Global Rotation

Rotate the entire point cloud and all bounding boxes around the vertical (z) axis:

```python
rotation_angle = np.random.uniform(-np.pi/4, np.pi/4)  # +/- 45 degrees

# Rotate all points
rotation_matrix = [[cos(a), -sin(a), 0],
                   [sin(a),  cos(a), 0],
                   [0,       0,      1]]
points[:, :3] = points[:, :3] @ rotation_matrix.T

# Rotate box centers and update headings
boxes[:, :2] = boxes[:, :2] @ rotation_matrix[:2, :2].T
boxes[:, 6] += rotation_angle
```

**Physical interpretation:** The ego vehicle approaches the scene from a slightly
different angle. This teaches the network that objects can appear at any orientation
relative to the sensor, not just the specific angles captured during data collection.

### 3.3 Global Scaling

Scale all coordinates uniformly:

```python
scale_factor = np.random.uniform(0.95, 1.05)  # +/- 5%

points[:, :3] *= scale_factor
boxes[:, :6] *= scale_factor  # scale position and dimensions
# heading remains unchanged
```

**Physical interpretation:** Objects at slightly different distances appear slightly
larger or smaller. This teaches the network to handle natural size variation.

### 3.4 Global Translation

Shift the entire scene by a random offset:

```python
translation = np.random.normal(0, 0.2, size=3)  # std = 0.2m per axis

points[:, :3] += translation
boxes[:, :3] += translation  # shift centers
```

**Physical interpretation:** The LiDAR sensor is mounted at a slightly different position.
This makes the network robust to calibration variations.

### 3.5 Random Flip

Mirror the scene along the x-axis or y-axis (50% probability each):

```python
if np.random.random() > 0.5:
    points[:, 1] = -points[:, 1]     # flip y
    boxes[:, 1] = -boxes[:, 1]       # flip box y
    boxes[:, 6] = -boxes[:, 6]       # negate heading

if np.random.random() > 0.5:
    points[:, 0] = -points[:, 0]     # flip x
    boxes[:, 0] = -boxes[:, 0]       # flip box x
    boxes[:, 6] = np.pi - boxes[:, 6]  # flip heading
```

**Physical interpretation:** Driving scenes are roughly symmetric -- a car approaching
from the left is as likely as from the right. Flipping doubles the effective dataset size.

### 3.6 Pipeline Order

Augmentations are applied in a specific sequence:

```
1. GT Database Sampling  (add objects first, before geometric transforms)
2. Random Flip           (coarse geometric)
3. Global Rotation       (medium geometric)
4. Global Scaling        (fine geometric)
5. Global Translation    (fine geometric)
6. Point Shuffling       (during pillar creation)
```

The order matters:
- GT sampling must come first so that pasted objects undergo the same geometric
  transforms as the rest of the scene.
- Flipping before rotation avoids creating impossible scenes (flipping after rotation
  could undo the rotation in unexpected ways).
- Fine adjustments (scaling, translation) come last to avoid being amplified by
  subsequent coarse transforms.

---

## 4. One-Cycle Learning Rate Policy

### 4.1 What Is Super-Convergence?

Super-convergence is a phenomenon where networks trained with a specific learning rate
schedule converge to better solutions in fewer iterations than traditional training.
The one-cycle policy exploits this by using a much higher maximum learning rate than
conventional training, cycling through it in a structured way.

### 4.2 The One-Cycle Schedule

```
Learning Rate over Training:

max_lr  ____________________________
       /                            \
      /                              \
     /                                \_______________
    /                                                 \
max_lr/10                                              max_lr/1000
    |                                                   |
    0%            40%                                 100%
    |--- Warmup ---|------------ Cosine Decay ----------|

Phase 1 (0% to 40%): Linear warmup from max_lr/10 to max_lr
Phase 2 (40% to 100%): Cosine decay from max_lr to max_lr/1000
```

### 4.3 Why Start Low, Go High, Then Decay

**Start low (warmup):** In early training, the network's predictions are random. Large
learning rates applied to random gradients cause instability. Starting at max_lr/10
allows the network to find a reasonable region of parameter space before increasing the
learning rate.

**Go high (peak):** The high learning rate phase serves two purposes:
1. **Exploration:** Large steps allow the optimizer to escape sharp local minima and
   find flatter regions of the loss landscape.
2. **Implicit regularization:** High learning rates inject noise into optimization,
   acting as a regularizer that improves generalization.

**Decay (settle):** After exploring, the learning rate decays to allow the optimizer to
settle into a good minimum. The final very low rate (max_lr/1000) provides fine-tuning
of the solution.

### 4.4 Momentum Coupling

The one-cycle policy optionally couples momentum inversely with learning rate:

```
When LR is high  --> Momentum is low  (0.85)
When LR is low   --> Momentum is high (0.95)

Rationale:
- High LR + low momentum = maximum exploration (high noise, less smoothing)
- Low LR + high momentum = maximum convergence (less noise, more smoothing)
```

### 4.5 Implementation

```python
from torch.optim.lr_scheduler import OneCycleLR

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)

scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,           # Peak learning rate
    total_steps=total_steps, # Total training iterations
    pct_start=0.4,           # 40% warmup
    anneal_strategy='cos',   # Cosine decay in phase 2
    div_factor=10,           # initial_lr = max_lr / 10
    final_div_factor=1000,   # final_lr = max_lr / 1000
)

# Call scheduler.step() after EVERY batch (not every epoch)
```

---

## 5. Gradient Clipping

### 5.1 What It Does

Gradient clipping limits the magnitude of gradients during training to prevent
catastrophically large weight updates:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
```

If the total gradient norm exceeds max_norm, all gradients are scaled down proportionally:

```
If ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

### 5.2 When It Matters

- **Early training:** Before predictions are reasonable, regression losses can produce
  enormous gradients (predicting a box 100m away from the true location).
- **GT database sampling:** Scenes with many pasted objects can produce unusually large
  aggregate gradients from the sum of many object losses.
- **Rare classes:** A scene with many pedestrians (normally rare) can produce a spike
  in gradient magnitude.

### 5.3 Settings

| Dataset | max_norm | Rationale |
|---------|:--------:|-----------|
| KITTI | 10 | Smaller scenes, fewer objects, more stable gradients |
| nuScenes | 35 | Larger scenes, 10 classes, more complex, larger gradients |

---

## 6. Class-Balanced Sampling

### 6.1 The Imbalance Problem

Autonomous driving datasets have extreme class imbalance:

```
KITTI Training Set Object Counts:

Car:         |========================================| ~28,000 instances
Pedestrian:  |======                                 | ~4,500 instances (6x fewer)
Cyclist:     |==                                     | ~1,600 instances (17x fewer)

Without mitigation:
- Network becomes biased toward Cars (sees them constantly)
- Pedestrian/Cyclist performance suffers (rarely sees them)
- This is DANGEROUS: failing to detect a pedestrian is worse than
  failing to detect a car (pedestrians are more vulnerable)
```

### 6.2 Three Mechanisms Working Together

**Mechanism 1: GT Database Sampling**
Directly increases the number of rare objects in each training scene:
- Sample 15 cars, 10 pedestrians, 10 cyclists per scene
- Relative to natural occurrence, pedestrians get 3x boost, cyclists get 10x boost

**Mechanism 2: Focal Loss**
The classification loss automatically down-weights easy examples (background anchors
that are obviously negative):

```
focal_loss = -alpha * (1 - p_t)^gamma * log(p_t)
  alpha = 0.25, gamma = 2.0

Example:
  Easy negative (p_t = 0.95): weight = (1-0.95)^2 = 0.0025 (nearly zero)
  Hard negative (p_t = 0.50): weight = (1-0.50)^2 = 0.25
  Positive (p_t = 0.30):      weight = (1-0.30)^2 = 0.49
```

This focuses the loss on informative examples rather than the overwhelming number of
trivially classified background anchors.

**Mechanism 3: Lower IoU Thresholds for Small Objects**
Pedestrians and cyclists use lower positive thresholds (0.5 vs 0.6 for cars), making it
easier for small objects to match anchors. This is necessary because small objects have
inherently lower maximum IoU with any fixed-size anchor.

---

## 7. KITTI vs nuScenes Training Differences

### 7.1 Configuration Comparison

| Parameter | KITTI | nuScenes | Why Different |
|-----------|-------|----------|---------------|
| Epochs | 80 | 20 | nuScenes is 7x larger, fewer epochs needed |
| Batch size | 6 | 4 | nuScenes has more points per scene, more memory |
| Point cloud range (x) | [0, 69.12] m | [-51.2, 51.2] m | nuScenes is 360-degree |
| Point cloud range (y) | [-39.68, 39.68] m | [-51.2, 51.2] m | nuScenes is wider |
| Point cloud range (z) | [-3, 1] m | [-5, 3] m | nuScenes has taller objects |
| Coverage | Front 90 degrees | Full 360 degrees | Different sensor setup |
| Number of classes | 3 | 10 | nuScenes has more diverse objects |
| Pillar size (x, y) | 0.16 m | 0.2 m | nuScenes: larger range needs coarser grid |
| Max pillars | 12,000 | 30,000 | 360-degree has more non-empty pillars |
| Max points/pillar | 100 | 20 | nuScenes uses multi-sweep (denser) |
| Input sweeps | 1 | 10 | Multi-sweep for density |
| Velocity prediction | No | Yes (vx, vy) | nuScenes provides velocity labels |
| max_lr | 2e-3 | 1e-3 | nuScenes is more complex, needs stability |
| Gradient clip norm | 10 | 35 | nuScenes has larger gradient magnitudes |

### 7.2 Key Differences Explained

**360-degree vs front-only:** KITTI uses a single forward-facing LiDAR with limited field
of view. nuScenes provides full 360-degree coverage, requiring a much larger detection
range and more pillars to cover the full circle.

**Single-frame vs multi-sweep:** KITTI processes one LiDAR scan at a time. nuScenes
aggregates 10 consecutive sweeps (about 0.5 seconds of data) to increase point density,
compensating for the sparser 32-beam LiDAR (vs KITTI's 64-beam sensor).

**Class count:** KITTI detects 3 classes (Car, Pedestrian, Cyclist). nuScenes detects 10
classes (Car, Truck, Bus, Trailer, Construction Vehicle, Pedestrian, Motorcycle, Bicycle,
Barrier, Traffic Cone), requiring a more complex detection head and more diverse anchors.

---

## 8. Multi-Sweep Input (nuScenes)

### 8.1 Why Multi-Sweep?

The nuScenes LiDAR (Velodyne VLP-32) produces sparser point clouds than KITTI's sensor
(Velodyne HDL-64). To compensate, multiple consecutive sweeps are aggregated:

```
Single sweep (32-beam):           10 sweeps aggregated:
  ~30,000 points                    ~300,000 points
  Sparse, hard to detect            Dense, similar to 64-beam
  distant objects                    single sweep
```

### 8.2 How It Works

Points from past sweeps are transformed to the current frame's coordinate system using
ego-vehicle pose information, then concatenated:

```python
# Each point in multi-sweep has 5 features:
# (x, y, z, intensity, time_lag)
#
# time_lag = 0.0 for current sweep
# time_lag = -0.05 for 1 sweep ago (50ms earlier)
# time_lag = -0.10 for 2 sweeps ago
# ... up to -0.45 for 9 sweeps ago

point_features = 5  # vs 4 for single-sweep KITTI
```

The time_lag feature allows the network to learn that older points may have moved
(if they belong to a moving object) or remained static (if they belong to background).

---

## 9. Velocity Prediction

### 9.1 What Is Predicted

For nuScenes, the regression head outputs 9 values instead of 7:

```
KITTI:    (dx, dy, dz, dw, dl, dh, dtheta)         -- 7 values
nuScenes: (dx, dy, dz, dw, dl, dh, dtheta, dvx, dvy)  -- 9 values
                                            ^^^^^^^^
                                            velocity components
```

### 9.2 Supervision Signal

Velocity ground truth comes from the difference between an object's position in
consecutive annotated frames (sampled at 2 Hz in nuScenes):

```
v_x = (x_current - x_previous) / delta_t
v_y = (y_current - y_previous) / delta_t

where delta_t = 0.5 seconds (annotation frequency)
```

### 9.3 Why Velocity Matters

Velocity estimation enables:
- **Tracking:** Predicting where objects will be in the next frame
- **Motion planning:** Anticipating the future trajectory of other vehicles
- **Static vs moving classification:** Distinguishing parked cars from moving ones

---

## 10. Fade Strategy

### 10.1 What It Is

In the final 5 epochs of nuScenes training, all data augmentation is disabled:

```python
if epoch >= (total_epochs - 5):
    disable_gt_sampling = True
    disable_global_rotation = True
    disable_global_scaling = True
    disable_global_translation = True
    disable_random_flip = True
```

### 10.2 Why Disable Augmentation at the End

During training with augmentation, the network learns on a modified data distribution
(rotated, scaled, with extra pasted objects). The real evaluation data has none of these
modifications. By training the final epochs WITHOUT augmentation, the network fine-tunes
its batch normalization statistics and final layer weights to match the true data
distribution.

This typically improves performance by 1-2 mAP. The effect is stronger for nuScenes
(which uses aggressive augmentation) than for KITTI.

---

## 11. Memory and Compute

### 11.1 Hardware Requirements

| Configuration | GPU | Batch Size | Memory Usage | Training Time |
|---------------|-----|:----------:|:------------:|:-------------:|
| Minimal (KITTI) | RTX 2080 Ti (11GB) | 2 | ~8 GB | ~10 hours |
| Recommended (KITTI) | RTX 3090 (24GB) | 6 | ~14 GB | ~4 hours |
| Full (nuScenes) | 4x V100 (32GB each) | 4 per GPU | ~24 GB/GPU | ~20 hours |
| Fast (nuScenes) | 8x A100 (40GB each) | 4 per GPU | ~28 GB/GPU | ~6 hours |

### 11.2 Reducing Memory for Smaller Setups

If your GPU has limited memory, apply these techniques in order:

1. **Reduce batch size** (most effective): Each point cloud uses ~4 GB during training.
   Batch size 2 instead of 6 saves ~16 GB.

2. **Mixed precision training (FP16):** Reduces memory by ~40% with minimal accuracy loss:
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

3. **Reduce max_pillars:** From 12,000 to 8,000 saves ~25% encoder memory at ~1 AP cost.

4. **Gradient accumulation:** Simulate larger batch sizes without more memory:
```python
accumulation_steps = 3  # Effective batch size = actual_bs * 3
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 11.3 Inference Latency Breakdown

| Component | Time (ms) | Percentage |
|-----------|:---------:|:----------:|
| Pillar Feature Net | 0.5 | 10% |
| Scatter to pseudo-image | 0.1 | 2% |
| Backbone (2D CNN) | 3.2 | 64% |
| SSD Detection Head | 0.7 | 14% |
| NMS post-processing | 0.5 | 10% |
| **Total** | **~5.0** | **100%** |

---

## 12. Practical Tips

### 12.1 Pretrained Backbone

While PointPillars trains from scratch effectively, using a backbone pretrained on
ImageNet can speed up convergence:
- Useful when training data is limited (KITTI: 3,712 samples)
- Less useful for large datasets (nuScenes: 28,130 samples)
- Re-initialize the first convolution layer (different input channels: 64 vs 3)

### 12.2 Monitoring Training

Track these metrics during training:

```python
# Every N iterations, log:
log('total_loss', loss.item())
log('cls_loss', cls_loss.item())
log('reg_loss', reg_loss.item())
log('dir_loss', dir_loss.item())
log('learning_rate', scheduler.get_last_lr())
log('num_positive_anchors', num_pos)

# Every eval interval, log:
for cls in ['Car', 'Pedestrian', 'Cyclist']:
    log(f'{cls}_3D_AP_Moderate', metrics[cls]['3d_moderate'])
    log(f'{cls}_BEV_AP_Moderate', metrics[cls]['bev_moderate'])
```

Watch for:
- Car AP converges first (most data, largest objects)
- Ped/Cyc AP lags 10-20 epochs behind Car
- Large gap between BEV AP and 3D AP indicates height estimation problems
- num_positive_anchors should be 50-200 per sample (if 0, anchors are wrong)

### 12.3 Learning Rate Tuning

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Loss explodes (goes to infinity) | LR too high | Reduce max_lr by 2-5x |
| Loss oscillates wildly | LR too high or batch too small | Reduce max_lr or increase batch |
| Loss plateaus early (epoch 5-10) | LR too low or broken augmentation | Increase max_lr by 1.5-2x |
| Loss NaN | Numerical instability | Add epsilon to log, reduce LR, check data |
| Train loss low but val AP poor | Overfitting | More augmentation, reduce model capacity |

### 12.4 Batch Size and Learning Rate Scaling

When changing batch size, scale the learning rate proportionally:

```
new_lr = base_lr * (new_batch_size / base_batch_size)

Example:
  Base: batch_size=6, max_lr=2e-3
  New:  batch_size=4, max_lr = 2e-3 * (4/6) = 1.33e-3
  New:  batch_size=12, max_lr = 2e-3 * (12/6) = 4e-3
```

For batch sizes larger than 16, use square root scaling instead of linear to avoid
instability.

---

## 13. Training Commands

### 13.1 KITTI Training (Single Class)

```bash
python tensorflow/train.py \
    --config configs/pointpillars_kitti_car.yaml \
    --data_root data/kitti/processed \
    --output_dir experiments/pp_kitti_car \
    --batch_size 4 \
    --epochs 80 \
    --learning_rate 0.002 \
    --weight_decay 0.01 \
    --grad_clip_norm 10
```

This command:
- Loads the car-only configuration (single class detection)
- Reads preprocessed KITTI data from data/kitti/processed
- Saves checkpoints and logs to experiments/pp_kitti_car
- Uses batch size 4 (reduce to 2 if GPU memory is limited)
- Trains for 80 epochs with one-cycle LR peaking at 0.002

### 13.2 KITTI Training (Multi-Class)

```bash
python tensorflow/train.py \
    --config configs/pointpillars_kitti_3class.yaml \
    --data_root data/kitti/processed \
    --output_dir experiments/pp_kitti_3class \
    --batch_size 4 \
    --epochs 160 \
    --learning_rate 0.002
```

Multi-class training uses more epochs (160 vs 80) because rare classes (Cyclist) need
more exposure to converge.

### 13.3 nuScenes Training

```bash
python tensorflow/train.py \
    --config configs/pointpillars_nuscenes.yaml \
    --data_root data/nuscenes/processed \
    --output_dir experiments/pp_nuscenes \
    --batch_size 4 \
    --epochs 20 \
    --learning_rate 0.001 \
    --grad_clip_norm 35 \
    --num_sweeps 10
```

### 13.4 Resuming Training

```bash
python tensorflow/train.py \
    --config configs/pointpillars_kitti_car.yaml \
    --data_root data/kitti/processed \
    --output_dir experiments/pp_kitti_car \
    --resume experiments/pp_kitti_car/checkpoint_epoch_40.pth
```

---

## 14. Debugging Training

### 14.1 Systematic Debugging Approach

When training produces unexpected results, follow this systematic process:

```
Step 1: Verify data loading
  - Visualize a training sample (points + boxes)
  - Confirm boxes align with point clusters
  - Check coordinate frame (x=forward, y=left, z=up)

Step 2: Verify anchor assignment
  - Print number of positive anchors per sample
  - Should be 50-200 for KITTI; if 0, anchors are misconfigured
  - Visualize which anchors match which GT boxes

Step 3: Overfit on one sample
  - Train on a single sample for 1000 iterations
  - Loss should drop to near zero (< 0.1)
  - If not, there is a bug in the model or loss computation

Step 4: Check loss components
  - If cls_loss is high: anchor assignment or focal loss issue
  - If reg_loss is high: residual encoding or smooth L1 issue
  - If dir_loss is high: direction target computation issue

Step 5: Check predictions
  - After 10 epochs, run inference on training data
  - Predictions should roughly match GT (even if imperfect)
  - If predictions are all at (0,0,0): scatter or backbone issue
```

### 14.2 Common Issues and Solutions

| Issue | Diagnostic | Root Cause | Solution |
|-------|-----------|------------|----------|
| AP = 0 for all classes | No positive anchors | Anchor sizes don't match data | Recompute anchor stats from training set |
| Car AP good, Ped/Cyc AP = 0 | Positive anchors only for Car | IoU thresholds too high for small objects | Lower Ped/Cyc thresholds to 0.5/0.35 |
| High recall, low AP | Many false positives | Score threshold too low or NMS broken | Check NMS IoU threshold, verify class scores |
| Loss is NaN | Check for inf values | Division by zero or log(0) | Add epsilon to log operations, clip values |
| Very slow training | GPU utilization low | Data loading bottleneck | Increase num_workers, use SSD, prefetch |
| Detections all same size | Regression not learning | Regression target encoding wrong | Verify residual encoding matches decoding |
| Poor orientation | Direction loss not decreasing | Wrong angle convention | Verify theta is consistent between GT and anchors |

### 14.3 Visualization for Debugging

Always visualize before and after augmentation:

```python
# Visualize a training sample
import open3d as o3d

# 1. Show point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(intensity_to_color(points[:, 3]))

# 2. Show ground truth boxes
for box in gt_boxes:
    bbox = create_oriented_bbox(box)
    bbox.color = (1, 0, 0)  # Red for GT

# 3. Show anchors that matched
for anchor in positive_anchors:
    bbox = create_oriented_bbox(anchor)
    bbox.color = (0, 1, 0)  # Green for matched anchors

o3d.visualization.draw_geometries([pcd] + gt_bboxes + anchor_bboxes)
```

### 14.4 Sanity Check: Overfitting on One Batch

The most reliable debugging technique is to verify the model can memorize a single batch:

```python
# Train on a single batch for 1000 iterations
single_batch = next(iter(train_loader))

model.train()
for step in range(1000):
    optimizer.zero_grad()
    loss = model(single_batch)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# Expected:
# Step 0:    loss = 8.5000
# Step 100:  loss = 1.2000
# Step 500:  loss = 0.0500
# Step 1000: loss = 0.0010  <-- near zero = model works
```

If loss does NOT reach near zero:
- Bug in model forward pass (check tensor shapes)
- Bug in loss computation (check target encoding)
- Bug in data loading (check input values are reasonable)

### 14.5 Verifying Anchor Assignment

A common failure mode is zero positive anchors. Check this explicitly:

```python
# After computing anchor-GT assignments:
num_pos = (labels == 1).sum().item()
num_neg = (labels == 0).sum().item()
num_ign = (labels == -1).sum().item()

print(f"Positive: {num_pos}, Negative: {num_neg}, Ignored: {num_ign}")

# Expected for KITTI (per sample):
#   Positive: 50-200 (depends on number of GT objects)
#   Negative: ~300,000 (majority)
#   Ignored: ~20,000

# If Positive = 0:
#   - Check anchor sizes match GT object sizes
#   - Check coordinate frames are consistent
#   - Check IoU computation is correct
#   - Verify GT boxes are in the detection range
```

### 14.6 Loss Component Analysis

When the total loss decreases but AP does not improve, examine individual components:

```python
# Log individual loss components:
log({
    'cls_loss': cls_loss.item(),      # Should decrease steadily
    'reg_loss': reg_loss.item(),      # Should decrease, may plateau
    'dir_loss': dir_loss.item(),      # Should decrease quickly
    'num_pos': num_pos,               # Should be stable (50-200)
})

# Interpretation:
# High cls_loss + normal reg_loss: Classification is the bottleneck
#   -> Check focal loss parameters (alpha, gamma)
#   -> Check positive/negative balance
#
# Low cls_loss + high reg_loss: Localization is the bottleneck
#   -> Check regression target encoding
#   -> Visualize predicted vs GT boxes
#
# All losses low but AP poor: Post-processing issue
#   -> Check NMS threshold
#   -> Check score threshold
#   -> Check box decoding
```

### 14.7 Data Pipeline Verification Checklist

Before starting a full training run, verify each component:

```
[ ] Point cloud loads correctly (check shape, range, units)
[ ] Labels load correctly (check box format matches code expectation)
[ ] Coordinate frame is consistent (x=forward, y=left, z=up for KITTI)
[ ] Point cloud range clips correctly (no points outside detection volume)
[ ] Pillarization produces non-empty pillars (check count > 0)
[ ] Feature augmentation produces 9 features (check values are reasonable)
[ ] GT database sampling places valid objects (visualize augmented scene)
[ ] Geometric augmentations apply to BOTH points and boxes
[ ] Anchor generation covers the feature map (check spatial distribution)
[ ] IoU computation gives reasonable values (test with known overlapping boxes)
[ ] Positive anchor count is 50-200 per sample
[ ] Loss components are all finite (no NaN or Inf)
[ ] Gradient norms are reasonable (check before and after clipping)
```

---

## 15. Training Recipes Summary

### 15.1 KITTI Car-Only Recipe

```yaml
# configs/pointpillars_kitti_car.yaml (key parameters)
dataset: kitti
classes: [Car]
epochs: 80
batch_size: 6
optimizer: AdamW
max_lr: 0.002
weight_decay: 0.01
scheduler: one_cycle (pct_start=0.4)
grad_clip_norm: 10
point_cloud_range: [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size: [0.16, 0.16, 4]
max_pillars: 12000
max_points_per_pillar: 32
augmentation:
  gt_sampling: {Car: 15}
  rotation: [-pi/4, pi/4]
  scaling: [0.95, 1.05]
  translation_std: 0.2
  flip: [x, y]
```

### 15.2 KITTI Multi-Class Recipe

```yaml
# configs/pointpillars_kitti_3class.yaml (key differences)
classes: [Car, Pedestrian, Cyclist]
epochs: 160  # More epochs for rare classes
augmentation:
  gt_sampling: {Car: 15, Pedestrian: 10, Cyclist: 10}
anchors:
  Car: {size: [1.6, 3.9, 1.56], z: -1.0, iou_pos: 0.6, iou_neg: 0.45}
  Pedestrian: {size: [0.6, 0.8, 1.73], z: -0.6, iou_pos: 0.5, iou_neg: 0.35}
  Cyclist: {size: [0.6, 1.76, 1.73], z: -0.6, iou_pos: 0.5, iou_neg: 0.35}
```

### 15.3 nuScenes Recipe

```yaml
# configs/pointpillars_nuscenes.yaml (key differences)
dataset: nuscenes
classes: [car, truck, bus, trailer, construction_vehicle,
          pedestrian, motorcycle, bicycle, barrier, traffic_cone]
epochs: 20
batch_size: 4
max_lr: 0.001
grad_clip_norm: 35
point_cloud_range: [-51.2, -51.2, -5, 51.2, 51.2, 3]
voxel_size: [0.2, 0.2, 8]
max_pillars: 30000
max_points_per_pillar: 20
num_sweeps: 10
predict_velocity: true
fade_epochs: 5  # Disable augmentation in last 5 epochs
```

---

## 16. Expected Training Milestones

### 16.1 KITTI Car-Only Training Timeline

| Epoch | Loss | Car 3D AP (Mod) | Notes |
|:-----:|:----:|:---------------:|-------|
| 1 | ~8.5 | - | Random predictions, loss dropping fast |
| 5 | ~2.1 | ~45% | Basic detection working |
| 10 | ~1.5 | ~60% | Most cars detected, localization improving |
| 20 | ~1.2 | ~68% | Good baseline performance |
| 40 | ~0.9 | ~74% | Near final performance |
| 60 | ~0.8 | ~76% | Diminishing returns |
| 80 | ~0.7 | ~77-78% | Converged |

### 16.2 When to Stop Training Early

Stop training if:
- Loss has not decreased for 20+ epochs (fully converged)
- Validation AP has not improved for 15+ epochs (overfitting beginning)
- Loss is NaN or Inf (broken training, needs debugging)

Continue training if:
- Loss is still decreasing steadily
- Validation AP is still improving
- Rare class AP (Ped, Cyc) is still climbing (they converge later)

---

## Summary

Training PointPillars effectively requires attention to several interrelated components:

- **Anchor design** must match the target class statistics
- **Data augmentation** (especially GT database sampling) is critical for handling class
  imbalance and limited data diversity
- **One-cycle learning rate** enables fast, stable convergence
- **Gradient clipping** prevents training instabilities
- **Dataset-specific tuning** (KITTI vs nuScenes) accounts for differences in scale,
  coverage, and class complexity
- **Systematic debugging** catches issues early before wasting GPU hours

When all components are properly configured, PointPillars achieves strong 3D detection
performance while maintaining real-time inference speed (~62 Hz), making it one of the
most practical architectures for autonomous driving perception pipelines.
