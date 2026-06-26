# Training Guide: RadarPillarNet

## 1. Overview

This document describes the complete training procedure for RadarPillarNet, including
data preparation, hyperparameter selection, augmentation strategies, and practical tips
for achieving optimal performance on the nuScenes radar detection benchmark.

## 2. Data Preparation

### 2.1 Prerequisites

- nuScenes dataset v1.0 (full dataset, not mini)
- nuScenes devkit installed (`pip install nuscenes-devkit`)
- Sufficient storage: ~400 GB for full dataset, ~5 GB for preprocessed radar data

### 2.2 Data Preprocessing Steps

```bash
# Step 1: Generate radar point cloud database (multi-sweep accumulation)
python scripts/create_radar_data.py \
    --dataroot /data/nuscenes \
    --output /data/nuscenes/radar_pillarnet_preprocessed \
    --n_sweeps 6 \
    --version v1.0-trainval

# Step 2: Create ground truth database for GT-sampling augmentation
python scripts/create_gt_database.py \
    --dataroot /data/nuscenes \
    --radar_data /data/nuscenes/radar_pillarnet_preprocessed \
    --output /data/nuscenes/radar_gt_database \
    --n_sweeps 6

# Step 3: Generate info files (metadata for data loading)
python scripts/create_infos.py \
    --dataroot /data/nuscenes \
    --output /data/nuscenes/radar_pillarnet_infos \
    --version v1.0-trainval
```

### 2.3 Preprocessed Data Structure

```
radar_pillarnet_preprocessed/
├── train/
│   ├── 000000.pkl    # Accumulated radar points + metadata
│   ├── 000001.pkl
│   └── ...
├── val/
│   ├── 000000.pkl
│   └── ...
├── gt_database/
│   ├── car/          # Cropped radar points per GT object
│   ├── truck/
│   ├── pedestrian/
│   └── ...
└── infos/
    ├── train_infos.pkl
    └── val_infos.pkl
```

## 3. Training Configuration

### 3.1 Core Hyperparameters

```yaml
# Model
model:
  voxel_size: [0.2, 0.2, 8.0]  # Pillar resolution (x, y, z)
  point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  max_points_per_pillar: 32
  max_pillars: 16000
  num_features: 9

# Training
training:
  batch_size: 4          # Per GPU
  num_epochs: 80
  learning_rate: 2e-4    # Peak learning rate (one-cycle)
  weight_decay: 0.01
  grad_clip_norm: 35.0
  num_workers: 4

# Optimizer
optimizer:
  type: AdamW
  betas: [0.95, 0.99]
  eps: 1.0e-8

# Learning rate schedule
lr_schedule:
  type: OneCycleLR
  max_lr: 2e-4
  div_factor: 10         # Initial LR = max_lr / div_factor = 2e-5
  pct_start: 0.4         # 40% warmup phase
  anneal_strategy: cos   # Cosine annealing after peak
  final_div_factor: 100  # Final LR = max_lr / final_div_factor = 2e-6
```

### 3.2 Loss Function Configuration

```yaml
losses:
  classification:
    type: FocalLoss
    alpha: 0.25
    gamma: 2.0
    weight: 1.0

  regression:
    type: SmoothL1Loss
    beta: 1.0
    weight: 2.0
    code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # dx,dy,dz,dw,dl,dh,dyaw

  direction:
    type: CrossEntropyLoss
    weight: 0.2

  velocity:
    type: SmoothL1Loss
    beta: 1.0
    weight: 0.2
    code_weights: [1.0, 1.0]  # vx, vy
```

### 3.3 Anchor Configuration

```yaml
anchors:
  car:
    sizes: [[4.63, 1.97, 1.74]]
    rotations: [0, 1.5708]
    matched_threshold: 0.6
    unmatched_threshold: 0.45

  truck:
    sizes: [[6.93, 2.51, 2.84]]
    rotations: [0, 1.5708]
    matched_threshold: 0.55
    unmatched_threshold: 0.4

  bus:
    sizes: [[10.5, 2.94, 3.47]]
    rotations: [0, 1.5708]
    matched_threshold: 0.55
    unmatched_threshold: 0.4

  pedestrian:
    sizes: [[0.73, 0.67, 1.77]]
    rotations: [0, 1.5708]
    matched_threshold: 0.6
    unmatched_threshold: 0.4

  motorcycle:
    sizes: [[2.11, 0.77, 1.47]]
    rotations: [0, 1.5708]
    matched_threshold: 0.5
    unmatched_threshold: 0.35

  bicycle:
    sizes: [[1.70, 0.60, 1.28]]
    rotations: [0, 1.5708]
    matched_threshold: 0.5
    unmatched_threshold: 0.35
```

## 4. Data Augmentation

### 4.1 Radar-Specific Augmentation Pipeline

```yaml
augmentation:
  # Ground truth sampling (most impactful augmentation for radar)
  gt_sampling:
    enabled: true
    rates:
      car: 3            # Sample up to 3 additional cars
      truck: 3
      bus: 2
      pedestrian: 4
      motorcycle: 3
      bicycle: 3
    min_points: 1       # Minimum radar points in GT sample
    transform_velocity: true  # Transform velocity vectors with rotation

  # Global rotation
  global_rotation:
    enabled: true
    range: [-0.7854, 0.7854]  # ±45 degrees (pi/4)

  # Global scaling
  global_scaling:
    enabled: true
    range: [0.95, 1.05]       # ±5% scale

  # Random horizontal flip
  random_flip:
    enabled: true
    probability: 0.5
    axes: [x, y]              # Flip along X or Y axis

  # Global translation
  global_translation:
    enabled: true
    std: [0.5, 0.5, 0.3]     # Standard deviation in x, y, z (meters)
```

### 4.2 GT-Sampling with Velocity Transform

When inserting ground truth objects during augmentation, the velocity vectors must be
transformed consistently with the spatial augmentation:

```python
def transform_gt_sample(gt_points, gt_box, rotation_angle, flip_x, flip_y):
    """Transform a GT sample including its velocity vectors."""
    # Rotate points and box
    rot_matrix = rotation_matrix_z(rotation_angle)
    gt_points[:, :3] = gt_points[:, :3] @ rot_matrix.T
    gt_box[:3] = gt_box[:3] @ rot_matrix.T
    gt_box[6] += rotation_angle  # yaw

    # Transform velocity (vx_comp, vy_comp are at indices 4, 5)
    gt_points[:, 4:6] = gt_points[:, 4:6] @ rot_matrix[:2, :2].T

    # Handle flips
    if flip_x:
        gt_points[:, 1] *= -1
        gt_points[:, 5] *= -1  # vy_comp
        gt_box[1] *= -1
        gt_box[6] = -gt_box[6]

    if flip_y:
        gt_points[:, 0] *= -1
        gt_points[:, 4] *= -1  # vx_comp
        gt_box[0] *= -1
        gt_box[6] = np.pi - gt_box[6]

    return gt_points, gt_box
```

### 4.3 Augmentation Order

The augmentation pipeline is applied in the following order:

1. GT-sampling (insert additional objects)
2. Random flip (horizontal)
3. Global rotation
4. Global scaling
5. Global translation
6. Range filtering (remove points outside grid)

## 5. Multi-GPU Training

### 5.1 Distributed Training Setup

```bash
# Single node, 4 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/train.py \
    --config configs/radar_pillarnet_nuscenes.yaml \
    --launcher pytorch

# Multi-node (2 nodes, 4 GPUs each)
# Node 0:
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="node0_ip" \
    --master_port=29500 \
    scripts/train.py \
    --config configs/radar_pillarnet_nuscenes.yaml

# Node 1:
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="node0_ip" \
    --master_port=29500 \
    scripts/train.py \
    --config configs/radar_pillarnet_nuscenes.yaml
```

### 5.2 Batch Size Scaling

When using multiple GPUs, scale the learning rate linearly:

```
effective_batch_size = batch_size_per_gpu * num_gpus
scaled_lr = base_lr * (effective_batch_size / 4)

Example: 4 GPUs, batch_size=4 per GPU -> effective=16, lr = 2e-4 * 4 = 8e-4
```

### 5.3 SyncBatchNorm

Enable synchronized batch normalization for multi-GPU training:

```python
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

## 6. Training Monitoring

### 6.1 Key Metrics to Track

| Metric | Expected Behavior | Warning Sign |
|--------|------------------|--------------|
| Total loss | Decreasing, stabilizes ~epoch 60 | Oscillating or increasing |
| Classification loss | Rapid initial decrease | Stuck above 1.0 after epoch 10 |
| Regression loss | Gradual decrease | Sudden spikes |
| Velocity loss | Slow decrease | Not decreasing at all |
| Learning rate | One-cycle shape | N/A |
| GPU memory | Stable | Growing (memory leak) |

### 6.2 Checkpointing Strategy

```yaml
checkpoint:
  save_interval: 5        # Save every 5 epochs
  keep_last: 5            # Keep last 5 checkpoints
  save_best: true         # Save best based on val NDS
  eval_interval: 5        # Evaluate every 5 epochs
```

## 7. Common Issues and Solutions

### 7.1 Training Divergence

**Symptom:** Loss explodes or becomes NaN after a few iterations.

**Solutions:**
- Reduce initial learning rate (try 1e-4 instead of 2e-4)
- Increase gradient clipping norm threshold
- Check for corrupted data samples (NaN in radar points)
- Ensure velocity values are properly compensated (not raw sensor values)

### 7.2 Slow Convergence

**Symptom:** Loss decreases very slowly, model underperforms expected metrics.

**Solutions:**
- Enable GT-sampling augmentation (most impactful for sparse radar data)
- Increase number of training epochs to 100
- Verify multi-sweep accumulation is working correctly (check point counts)
- Ensure ego-motion compensation is applied correctly

### 7.3 Overfitting

**Symptom:** Training loss continues decreasing but validation metrics plateau or degrade.

**Solutions:**
- Increase augmentation strength (rotation range, flip probability)
- Reduce model size (fewer channels in backbone)
- Add dropout in the detection head
- Reduce GT-sampling rates

### 7.4 GPU Out of Memory

**Symptom:** CUDA OOM error during training.

**Solutions:**
- Reduce batch size (minimum viable: 2 per GPU)
- Reduce max_pillars from 16000 to 12000
- Use gradient accumulation to simulate larger batches
- Enable mixed precision training (FP16)

### 7.5 Poor Velocity Predictions

**Symptom:** mAVE metric is high (bad velocity estimation).

**Solutions:**
- Increase velocity loss weight from 0.2 to 0.5
- Verify ego-motion compensation in preprocessing
- Check that velocity augmentation transform is correct
- Ensure velocity regression targets use global frame velocities

## 8. Mixed Precision Training

Enable AMP (Automatic Mixed Precision) for faster training with lower memory:

```python
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
    scaler.step(optimizer)
    scaler.update()
```

Expected speedup: ~1.5x training speed, ~30% memory reduction.

## 9. Training Timeline

| Phase | Epochs | Expected Duration (4x V100) |
|-------|--------|------------------------------|
| Warmup | 0-32 | ~8 hours |
| Peak LR | 32-40 | ~2 hours |
| Annealing | 40-80 | ~10 hours |
| **Total** | 0-80 | **~20 hours** |
