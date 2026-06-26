# Training Guide: CenterPoint

## Overview

This guide covers the complete training procedure for CenterPoint, including optimizer configuration, learning rate scheduling, data augmentation strategies, loss functions, and the two-stage training schedule.

---

## Optimizer Configuration

### Adam / AdamW

CenterPoint uses AdamW (Adam with decoupled weight decay) as the primary optimizer:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.99),
    weight_decay=0.01,
    eps=1e-8,
)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base learning rate | 0.001 | Scaled linearly with batch size |
| Beta1 | 0.9 | Momentum for first moment |
| Beta2 | 0.99 | Momentum for second moment |
| Weight decay | 0.01 | Decoupled (AdamW style) |
| Epsilon | 1e-8 | Numerical stability |
| Gradient clipping | max_norm=35 | Clip gradient L2 norm |

### Learning Rate Scaling

When using different batch sizes, scale the learning rate linearly:

```python
# Reference: lr=0.001 for batch_size=4 on 8 GPUs (effective batch 32)
effective_batch = batch_size_per_gpu * num_gpus
lr = 0.001 * (effective_batch / 32)
```

---

## Learning Rate Schedule: OneCycle

CenterPoint uses the OneCycle learning rate policy (Smith & Topin, 2018):

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    total_steps=total_epochs * steps_per_epoch,
    pct_start=0.4,          # 40% of training for warmup phase
    anneal_strategy='cos',  # Cosine annealing
    div_factor=10,          # initial_lr = max_lr / 10
    final_div_factor=100,   # final_lr = initial_lr / 100
)
```

### OneCycle Phases

```
LR
│    ╱‾‾‾‾╲
│   ╱      ╲
│  ╱        ╲
│ ╱          ╲
│╱            ╲________
└──────────────────────── Epoch
   Warmup    Anneal  Final
  (0→40%)  (40→90%)  (90→100%)
```

| Phase | Epoch Range | LR Range | Description |
|-------|-------------|----------|-------------|
| Warmup | 0 - 8 (40%) | 0.0001 -> 0.001 | Linear increase to max LR |
| Cosine Anneal | 8 - 18 (40-90%) | 0.001 -> 0.0001 | Cosine decay |
| Final Decay | 18 - 20 (90-100%) | 0.0001 -> 0.000001 | Fine-grained convergence |

### Momentum Schedule (inverse of LR)

OneCycle also varies momentum inversely with learning rate:

```python
# Momentum range (inversely coupled to LR)
momentum_range = (0.85, 0.95)
# When LR is high -> momentum is low (0.85)
# When LR is low -> momentum is high (0.95)
```

---

## Training Duration

### nuScenes

| Configuration | Epochs | Iterations | Wall Time (8x A100) |
|--------------|--------|------------|---------------------|
| CenterPoint-Voxel (1st stage) | 20 | ~14,000 | ~18 hours |
| CenterPoint-Voxel (2nd stage) | 20 | ~14,000 | ~22 hours |
| CenterPoint-Pillar (1st stage) | 20 | ~14,000 | ~12 hours |

### Waymo

| Configuration | Epochs | Iterations | Wall Time (8x A100) |
|--------------|--------|------------|---------------------|
| CenterPoint-Voxel (1st stage) | 36 | ~120,000 | ~72 hours |
| CenterPoint-Pillar (1st stage) | 36 | ~120,000 | ~48 hours |

---

## Fade Strategy (Augmentation Annealing)

### Concept

The fade strategy disables strong data augmentation during the last few epochs of training. This allows the model to fine-tune on clean, unaugmented data, improving final accuracy:

```python
class FadeStrategy:
    """Disable augmentation in the final N epochs."""
    
    def __init__(self, total_epochs=20, fade_epochs=5):
        self.total_epochs = total_epochs
        self.fade_epoch_start = total_epochs - fade_epochs  # epoch 15
    
    def should_augment(self, current_epoch):
        return current_epoch < self.fade_epoch_start
    
    def get_augmentation_config(self, current_epoch):
        if self.should_augment(current_epoch):
            return {
                'gt_sampling': True,
                'random_rotation': [-0.785, 0.785],     # [-pi/4, pi/4]
                'random_scaling': [0.95, 1.05],
                'random_translation': [-0.2, 0.2],
                'random_flip': True,
            }
        else:
            # Fade: disable all augmentation
            return {
                'gt_sampling': False,
                'random_rotation': [0.0, 0.0],
                'random_scaling': [1.0, 1.0],
                'random_translation': [0.0, 0.0],
                'random_flip': False,
            }
```

### Schedule

| Epoch | Augmentation | Notes |
|-------|-------------|-------|
| 1-15 | Full augmentation | GT sampling + geometric transforms |
| 16-20 | No augmentation (fade) | Clean data only, fine-tune |

---

## Data Augmentation

### GT Sampling (Ground Truth Database Augmentation)

GT sampling is the most impactful augmentation for LiDAR detection. It pastes ground truth objects (with their LiDAR points) from a pre-computed database into the current scene:

```python
class GTSampling:
    """
    Paste ground truth objects from a database into the training scene
    to increase object density and class balance.
    """
    
    def __init__(self, db_info_path, sample_groups):
        # Load pre-computed database of GT objects
        self.db_infos = load_pickle(db_info_path)
        
        # How many objects to sample per class
        self.sample_groups = sample_groups
        # Example: {'car': 15, 'truck': 3, 'bus': 4, 'pedestrian': 10, ...}
    
    def __call__(self, points, gt_boxes, gt_names):
        """
        Args:
            points: [N, 5] current scene points
            gt_boxes: [M, 9] existing GT boxes
            gt_names: [M] existing GT class names
        Returns:
            augmented_points: points with sampled objects added
            augmented_boxes: boxes with sampled objects added
        """
        sampled_boxes = []
        sampled_points = []
        
        for class_name, num_samples in self.sample_groups.items():
            # Count existing objects of this class
            existing = (gt_names == class_name).sum()
            num_to_sample = max(0, num_samples - existing)
            
            if num_to_sample == 0:
                continue
            
            # Randomly select from database
            candidates = self.db_infos[class_name]
            selected = np.random.choice(candidates, num_to_sample, replace=False)
            
            for info in selected:
                # Load stored points for this object
                obj_points = load_points(info['path'])
                
                # Check for collision with existing boxes
                if not self._check_collision(info['box'], gt_boxes, sampled_boxes):
                    sampled_boxes.append(info['box'])
                    sampled_points.append(obj_points)
        
        # Merge sampled objects into scene
        if sampled_boxes:
            # Remove scene points inside sampled box regions
            points = self._remove_points_in_boxes(points, sampled_boxes)
            
            # Add sampled points
            all_sampled_points = np.concatenate(sampled_points)
            points = np.concatenate([points, all_sampled_points])
            
            # Update ground truth
            gt_boxes = np.concatenate([gt_boxes, np.array(sampled_boxes)])
            gt_names = np.concatenate([gt_names, [info['name'] for info in selected]])
        
        return points, gt_boxes, gt_names
```

### GT Sampling Configuration

```python
GT_SAMPLE_GROUPS = {
    'car': 15,
    'truck': 3,
    'construction_vehicle': 7,
    'bus': 4,
    'trailer': 6,
    'barrier': 10,
    'motorcycle': 6,
    'bicycle': 6,
    'pedestrian': 10,
    'traffic_cone': 10,
}

# Minimum number of LiDAR points for a GT object to be included in database
MIN_POINTS_FOR_DB = {
    'car': 5,
    'truck': 5,
    'bus': 5,
    'pedestrian': 5,
    'motorcycle': 5,
    'bicycle': 5,
    'traffic_cone': 5,
    'barrier': 5,
    'trailer': 5,
    'construction_vehicle': 5,
}
```

### Geometric Augmentations

```python
class GeometricAugmentation:
    """Standard geometric augmentations for 3D point clouds."""
    
    def random_rotation(self, points, boxes, rotation_range=[-0.785, 0.785]):
        """Random yaw rotation around Z-axis."""
        angle = np.random.uniform(*rotation_range)
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        points[:, :3] = points[:, :3] @ rot_matrix.T
        boxes[:, :3] = boxes[:, :3] @ rot_matrix.T
        boxes[:, 6] += angle  # Update yaw
        # Also rotate velocity
        boxes[:, 7:9] = boxes[:, 7:9] @ rot_matrix[:2, :2].T
        return points, boxes
    
    def random_scaling(self, points, boxes, scale_range=[0.95, 1.05]):
        """Random uniform scaling."""
        scale = np.random.uniform(*scale_range)
        points[:, :3] *= scale
        boxes[:, :3] *= scale
        boxes[:, 3:6] *= scale  # Scale dimensions
        boxes[:, 7:9] *= scale  # Scale velocity
        return points, boxes
    
    def random_flip(self, points, boxes, axes=['x', 'y']):
        """Random flip along X and/or Y axis."""
        for axis in axes:
            if np.random.random() < 0.5:
                if axis == 'x':
                    points[:, 1] = -points[:, 1]
                    boxes[:, 1] = -boxes[:, 1]
                    boxes[:, 6] = -boxes[:, 6]  # Flip yaw
                    boxes[:, 8] = -boxes[:, 8]  # Flip vy
                elif axis == 'y':
                    points[:, 0] = -points[:, 0]
                    boxes[:, 0] = -boxes[:, 0]
                    boxes[:, 6] = np.pi - boxes[:, 6]  # Flip yaw
                    boxes[:, 7] = -boxes[:, 7]  # Flip vx
        return points, boxes
    
    def random_translation(self, points, boxes, translation_std=[0.2, 0.2, 0.2]):
        """Random global translation."""
        translation = np.random.normal(0, translation_std, size=3)
        points[:, :3] += translation
        boxes[:, :3] += translation
        return points, boxes
```

---

## Loss Functions

### Overall Loss

```python
total_loss = (
    w_heatmap * gaussian_focal_loss +
    w_offset * l1_loss_offset +
    w_height * l1_loss_height +
    w_size * l1_loss_size +
    w_rotation * l1_loss_rotation +
    w_velocity * l1_loss_velocity
)
```

### Loss Weights

| Loss Component | Weight | Notes |
|---------------|--------|-------|
| Heatmap (Gaussian Focal) | 1.0 | Center detection quality |
| Offset (L1) | 2.0 | Sub-voxel localization precision |
| Height (L1) | 2.0 | Vertical localization |
| Size (L1, log-space) | 0.2 | Lower weight since log-space compresses range |
| Rotation (L1, sin/cos) | 1.0 | Heading estimation |
| Velocity (L1) | 0.2 | Lower weight, noisier supervision |

### Regression Loss Details

```python
def compute_regression_loss(predictions, targets, mask):
    """
    Compute L1 regression loss at positive (GT center) locations only.
    
    Args:
        predictions: dict of predicted maps {offset, height, size, rot, vel}
        targets: dict of ground truth maps
        mask: [B, H, W] binary mask of GT center locations
    """
    losses = {}
    
    # Only compute loss at GT center locations
    num_pos = mask.sum().clamp(min=1)
    
    for key in ['offset', 'height', 'size', 'rotation', 'velocity']:
        pred = predictions[key]  # [B, C, H, W]
        target = targets[key]    # [B, C, H, W]
        
        # Gather predictions at positive locations
        pred_pos = pred[mask.unsqueeze(1).expand_as(pred)]
        target_pos = target[mask.unsqueeze(1).expand_as(target)]
        
        # L1 loss
        losses[key] = F.l1_loss(pred_pos, target_pos, reduction='sum') / num_pos
    
    return losses
```

---

## Two-Stage Training Schedule

### Stage 1: Train Detection Network

```python
# Stage 1: Train backbone + BEV + center heads
stage1_config = {
    'epochs': 20,
    'optimizer': 'AdamW',
    'lr': 0.001,
    'scheduler': 'OneCycleLR',
    'batch_size': 4,  # per GPU
    'num_gpus': 8,
    'fade_epochs': 5,
    'components': ['voxelizer', '3d_backbone', 'bev_collapse', '2d_backbone', 'center_heads'],
}
```

### Stage 2: Train Refinement Module

```python
# Stage 2: Freeze stage 1, train second-stage MLP
stage2_config = {
    'epochs': 20,
    'optimizer': 'AdamW',
    'lr': 0.0001,  # Lower LR for refinement
    'scheduler': 'OneCycleLR',
    'batch_size': 4,
    'num_gpus': 8,
    'fade_epochs': 5,
    'freeze': ['voxelizer', '3d_backbone', 'bev_collapse', '2d_backbone', 'center_heads'],
    'train': ['second_stage_extractor', 'second_stage_mlp'],
    'loss': {
        'cls': 'BinaryCrossEntropy',
        'reg': 'SmoothL1Loss',
        'cls_weight': 1.0,
        'reg_weight': 2.0,
    },
}
```

### Alternative: End-to-End Two-Stage Training

Some implementations train both stages jointly:

```python
# Joint training (less common, potentially unstable)
joint_config = {
    'epochs': 20,
    'optimizer': 'AdamW',
    'lr': 0.001,
    'stage1_loss_weight': 1.0,
    'stage2_loss_weight': 0.25,  # Lower weight for stability
    'detach_proposals': True,    # Stop gradient from stage2 to stage1
}
```

---

## Training Pipeline Summary

```
Epoch 1-15 (with augmentation):
├── Load batch (multi-sweep aggregation)
├── Apply GT sampling (paste objects from database)
├── Apply geometric augmentation (rotation, scaling, flip, translation)
├── Voxelize augmented point cloud
├── Forward pass through model
├── Compute losses (heatmap + regression heads)
├── Backward pass + gradient clipping (max_norm=35)
├── Optimizer step + scheduler step
└── Log metrics (loss, grad_norm, lr)

Epoch 16-20 (fade, no augmentation):
├── Load batch (multi-sweep aggregation)
├── NO augmentation applied
├── Voxelize clean point cloud
├── Forward pass through model
├── Compute losses
├── Backward pass + gradient clipping
├── Optimizer step + scheduler step
└── Log metrics
```

---

## Distributed Training

### Multi-GPU Configuration

```python
# PyTorch DistributedDataParallel (DDP)
distributed_config = {
    'backend': 'nccl',
    'num_gpus': 8,
    'batch_size_per_gpu': 4,
    'sync_bn': True,          # Synchronized batch normalization
    'find_unused_parameters': False,
}
```

### Launch Command

```bash
# 8-GPU training on a single node
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    tools/train.py \
    --config configs/centerpoint_voxel.yaml \
    --work_dir work_dirs/centerpoint_voxel_nuscenes
```

---

## Hyperparameter Summary

| Hyperparameter | nuScenes Value | Waymo Value |
|---------------|----------------|-------------|
| Epochs | 20 | 36 |
| Base LR | 0.001 | 0.003 |
| Batch size (per GPU) | 4 | 4 |
| Num GPUs | 8 | 8 |
| Weight decay | 0.01 | 0.01 |
| Gradient clip norm | 35 | 35 |
| OneCycle pct_start | 0.4 | 0.4 |
| Fade epochs | 5 | 5 |
| GT sample (car) | 15 | 15 |
| Rotation range | [-pi/4, pi/4] | [-pi/4, pi/4] |
| Scale range | [0.95, 1.05] | [0.95, 1.05] |
| Double flip | Yes | Yes |

---

## Common Training Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| NaN loss | Loss becomes NaN early | Reduce LR, check data loading, verify voxelization bounds |
| Slow convergence | mAP plateaus below 50% | Verify GT sampling is working, check augmentation parameters |
| Memory OOM | CUDA out of memory | Reduce batch size, reduce max_voxels, use gradient accumulation |
| Sparse conv errors | Runtime errors in spconv | Ensure spconv version matches CUDA version, check voxel coordinates |
| Poor small object detection | Low AP for pedestrians/cones | Increase GT sampling rate for small objects, verify min_points filtering |
| Unstable two-stage | Stage 2 degrades results | Ensure proposals are detached, verify IoU assignment thresholds |
