# PointNet++ Training Guide

## 1. Overview

This guide covers the training procedure for PointNet++ models across classification, segmentation, and 3D object detection tasks. It includes optimizer configuration, learning rate scheduling, data augmentation strategies, mixed precision training, and multi-GPU scaling.

---

## 2. Optimizer Configuration

### 2.1 Adam Optimizer (Recommended)

Adam is the standard optimizer for PointNet++ training due to its adaptive learning rates handling the varying gradient magnitudes across SA layers.

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,           # Initial learning rate
    betas=(0.9, 0.999), # Momentum parameters
    eps=1e-8,           # Numerical stability
    weight_decay=1e-4   # L2 regularization
)
```

### 2.2 AdamW (For Detection Tasks)

For 3D detection with larger models, AdamW provides better generalization through decoupled weight decay:

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01    # Higher weight decay, decoupled from LR
)
```

### 2.3 SGD with Momentum (Alternative)

For fine-tuning or when Adam leads to overfitting on small datasets:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
```

---

## 3. Learning Rate Schedule

### 3.1 Cosine Annealing (Recommended)

Cosine annealing provides smooth decay with optional warm restarts:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

# Standard cosine decay
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=200,          # Total epochs
    eta_min=1e-5        # Minimum learning rate
)

# With warm restarts (for longer training)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=50,             # First restart period
    T_mult=2,           # Period multiplier after each restart
    eta_min=1e-5
)
```

### 3.2 Step Decay (Original Paper)

The original PointNet++ paper uses step decay:

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(
    optimizer,
    step_size=20,       # Decay every 20 epochs
    gamma=0.7           # Multiply LR by 0.7
)
# LR progression: 0.001, 0.0007, 0.00049, 0.000343, ...
```

### 3.3 One-Cycle Policy (For Detection)

One-cycle achieves faster convergence for detection tasks:

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=total_epochs * steps_per_epoch,
    pct_start=0.4,       # 40% warmup
    anneal_strategy='cos',
    div_factor=10,       # Initial LR = max_lr / 10
    final_div_factor=100 # Final LR = max_lr / (10 * 100)
)
```

### 3.4 Linear Warmup + Cosine Decay

```python
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_warmup_cosine_scheduler(
    optimizer, warmup_epochs=5, total_epochs=200
)
```

---

## 4. Batch Size Considerations

### 4.1 Memory Scaling

Point cloud batch sizes are constrained by GPU memory due to the ball query expansion:

```
Memory per sample ≈ npoints × nsample × feature_dim × 4 bytes × num_SA_layers

Example (Detection, npoints=16384):
  SA1: 4096 × 64 × 132 × 4 = 137 MB
  SA2: 1024 × 64 × 259 × 4 = 68 MB
  SA3: 256 × 64 × 515 × 4 = 34 MB
  SA4: 64 × 64 × 1027 × 4 = 17 MB
  Overhead (gradients, activations): ~2x
  Total per sample: ~500 MB
  
  GPU 24GB: batch_size ≈ 8-12
  GPU 48GB: batch_size ≈ 16-24
```

### 4.2 Recommended Batch Sizes

| Task | Points/Sample | GPU (24GB) | GPU (48GB) | GPU (80GB) |
|------|---------------|------------|------------|------------|
| Classification (1024 pts) | 1024 | 32-48 | 64-96 | 128-192 |
| Classification (4096 pts) | 4096 | 24-32 | 48-64 | 96-128 |
| Segmentation (8192 pts) | 8192 | 12-16 | 24-32 | 48-64 |
| Detection (16384 pts) | 16384 | 4-8 | 8-16 | 16-32 |
| Detection (32768 pts) | 32768 | 2-4 | 4-8 | 8-16 |

### 4.3 Gradient Accumulation

When the effective batch size is too small, use gradient accumulation:

```python
accumulation_steps = 4  # Effective batch = batch_size × accumulation_steps

for i, (points, labels) in enumerate(dataloader):
    loss = model(points, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5. Data Augmentation

### 5.1 Random Rotation Around Z-Axis

Essential for outdoor LiDAR where object heading is arbitrary:

```python
def random_rotation_z(points, angle_range=(-np.pi, np.pi)):
    """Rotate point cloud randomly around Z-axis (vertical).
    
    Args:
        points: (N, 3+C) point cloud
        angle_range: tuple of (min_angle, max_angle) in radians
    """
    theta = np.random.uniform(*angle_range)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    rotation_matrix = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ])
    
    points[:, :3] = points[:, :3] @ rotation_matrix.T
    return points, theta  # Return angle to also rotate box labels
```

### 5.2 Random Flip Along X and Y Axes

```python
def random_flip(points, boxes=None, prob=0.5):
    """Random flip along X-axis and/or Y-axis.
    
    Args:
        points: (N, 3+C) point cloud
        boxes: (M, 7) bounding boxes [x,y,z,w,h,l,yaw] or None
    """
    # Flip along X-axis (left-right)
    if np.random.random() < prob:
        points[:, 1] = -points[:, 1]  # Negate Y
        if boxes is not None:
            boxes[:, 1] = -boxes[:, 1]  # Negate box Y
            boxes[:, 6] = -boxes[:, 6]  # Negate yaw
    
    # Flip along Y-axis (front-back) — less common, use carefully
    if np.random.random() < prob:
        points[:, 0] = -points[:, 0]  # Negate X
        if boxes is not None:
            boxes[:, 0] = -boxes[:, 0]
            boxes[:, 6] = np.pi - boxes[:, 6]
    
    return points, boxes
```

### 5.3 Random Scaling

```python
def random_scaling(points, boxes=None, scale_range=(0.95, 1.05)):
    """Uniform random scaling of point cloud.
    
    Args:
        points: (N, 3+C) point cloud
        boxes: (M, 7) bounding boxes or None
        scale_range: (min_scale, max_scale) tuple
    """
    scale = np.random.uniform(*scale_range)
    
    points[:, :3] *= scale
    
    if boxes is not None:
        boxes[:, :3] *= scale  # Scale center position
        boxes[:, 3:6] *= scale  # Scale dimensions
    
    return points, boxes
```

### 5.4 Point Jittering (Gaussian Noise)

```python
def point_jittering(points, sigma=0.01, clip=0.05):
    """Add Gaussian noise to point positions.
    
    Args:
        points: (N, 3+C) point cloud
        sigma: standard deviation of noise
        clip: maximum noise magnitude
    """
    N = points.shape[0]
    noise = np.clip(
        np.random.randn(N, 3) * sigma,
        -clip, clip
    )
    points[:, :3] += noise
    return points
```

### 5.5 Random Point Dropout

```python
def random_point_dropout(points, max_dropout_ratio=0.875):
    """Randomly drop points (density augmentation for MSG training).
    
    Args:
        points: (N, 3+C) point cloud
        max_dropout_ratio: maximum fraction of points to remove
    """
    dropout_ratio = np.random.uniform(0, max_dropout_ratio)
    drop_idx = np.where(np.random.random(points.shape[0]) < dropout_ratio)[0]
    
    if len(drop_idx) > 0:
        # Replace dropped points with first point (avoids changing array size)
        points[drop_idx] = points[0]
    
    return points
```

### 5.6 GT-Aug (Ground Truth Augmentation for Detection)

Paste ground truth objects from a database into the current scene:

```python
def gt_augmentation(points, boxes, labels, db_sampler, max_samples_per_class):
    """Insert ground truth objects from a pre-computed database.
    
    Args:
        points: (N, 4) current scene point cloud
        boxes: (M, 7) current scene boxes
        labels: (M,) current scene labels
        db_sampler: database of pre-extracted object point clouds
        max_samples_per_class: dict {class_name: max_count}
    """
    for class_name, max_count in max_samples_per_class.items():
        # Count existing objects of this class
        existing = (labels == class_to_id[class_name]).sum()
        needed = max(0, max_count - existing)
        
        if needed == 0:
            continue
        
        # Sample from database
        sampled = db_sampler.sample(class_name, needed)
        
        for obj in sampled:
            # Check for collision with existing boxes
            if not check_collision(obj['box'], boxes):
                points = np.concatenate([points, obj['points']], axis=0)
                boxes = np.concatenate([boxes, obj['box'][np.newaxis]], axis=0)
                labels = np.concatenate([labels, [class_to_id[class_name]]])
    
    return points, boxes, labels
```

### 5.7 Complete Augmentation Pipeline

```python
class PointCloudAugmentor:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, points, boxes=None, labels=None):
        """Apply full augmentation pipeline.
        
        Order matters: GT-aug → flip → rotation → scaling → jitter → dropout
        """
        # 1. GT Augmentation (detection only, before geometric transforms)
        if self.config.get('gt_aug') and boxes is not None:
            points, boxes, labels = gt_augmentation(
                points, boxes, labels,
                self.config['db_sampler'],
                self.config['max_samples_per_class']
            )
        
        # 2. Random flip
        if self.config.get('random_flip', True):
            points, boxes = random_flip(points, boxes, prob=0.5)
        
        # 3. Random rotation around Z
        if self.config.get('random_rotation', True):
            angle_range = self.config.get('rotation_range', (-np.pi/4, np.pi/4))
            points, theta = random_rotation_z(points, angle_range)
            if boxes is not None:
                # Rotate box centers
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                boxes[:, :2] = boxes[:, :2] @ rot.T
                boxes[:, 6] += theta
        
        # 4. Random scaling
        if self.config.get('random_scaling', True):
            scale_range = self.config.get('scale_range', (0.95, 1.05))
            points, boxes = random_scaling(points, boxes, scale_range)
        
        # 5. Point jittering
        if self.config.get('point_jitter', True):
            sigma = self.config.get('jitter_sigma', 0.01)
            points = point_jittering(points, sigma=sigma)
        
        # 6. Random dropout (for MSG training)
        if self.config.get('random_dropout', False):
            points = random_point_dropout(points)
        
        return points, boxes, labels
```

---

## 6. Loss Functions

### 6.1 Classification Loss

```python
# Standard cross-entropy with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

loss = criterion(logits, labels)  # logits: (B, C), labels: (B,)
```

### 6.2 Segmentation Loss

```python
# Weighted cross-entropy for class imbalance
class_weights = compute_class_weights(train_dataset)  # Inverse frequency
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Or Focal Loss for extreme imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### 6.3 Detection Loss (Multi-Task)

```python
class DetectionLoss(nn.Module):
    def __init__(self, cls_weight=1.0, reg_weight=2.0, dir_weight=0.2):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.dir_weight = dir_weight
    
    def forward(self, pred_cls, pred_reg, pred_dir, targets):
        # Classification: Focal Loss
        cls_loss = focal_loss(pred_cls, targets['cls_labels'])
        
        # Regression: Smooth L1 (only for positive samples)
        pos_mask = targets['cls_labels'] > 0
        reg_loss = F.smooth_l1_loss(
            pred_reg[pos_mask],
            targets['reg_targets'][pos_mask],
            beta=1.0/9.0
        )
        
        # Direction: Binary cross-entropy
        dir_loss = F.binary_cross_entropy_with_logits(
            pred_dir[pos_mask],
            targets['dir_labels'][pos_mask]
        )
        
        total = (self.cls_weight * cls_loss +
                 self.reg_weight * reg_loss +
                 self.dir_weight * dir_loss)
        
        return total, {
            'cls_loss': cls_loss.item(),
            'reg_loss': reg_loss.item(),
            'dir_loss': dir_loss.item()
        }
```

---

## 7. Mixed Precision Training

### 7.1 PyTorch AMP (Automatic Mixed Precision)

Mixed precision significantly reduces memory usage and speeds up training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for points, labels in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        # Forward pass in FP16
        logits = model(points.cuda())
        loss = criterion(logits, labels.cuda())
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    # Unscale gradients and clip
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    
    # Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()
```

### 7.2 Operations to Keep in FP32

Some operations are numerically sensitive and should remain in FP32:

```python
# These are automatically handled by autocast, but worth noting:
# - Softmax / log-softmax
# - Loss computation
# - Batch normalization running statistics
# - Reduction operations (sum, mean over large dims)

# Custom CUDA ops (FPS, ball query) typically operate in FP32
# Only the MLP computations benefit from FP16
```

### 7.3 Memory Savings with AMP

| Configuration | FP32 Memory | AMP Memory | Speedup |
|---------------|-------------|------------|---------|
| Cls (B=32, 1024 pts) | 4.2 GB | 2.8 GB | 1.3x |
| Seg (B=16, 8192 pts) | 12.1 GB | 7.8 GB | 1.4x |
| Det (B=8, 16384 pts) | 18.5 GB | 11.2 GB | 1.5x |

---

## 8. Multi-GPU Training Strategy

### 8.1 DataParallel (Simple, Single Node)

```python
# Simple but inefficient (GIL bottleneck, memory imbalance)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

### 8.2 DistributedDataParallel (Recommended)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_ddp(rank, world_size, config):
    setup_ddp(rank, world_size)
    
    model = PointNetPP(config).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Distributed sampler ensures no data overlap between GPUs
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        train_one_epoch(model, dataloader, optimizer, scheduler)

# Launch with torchrun:
# torchrun --nproc_per_node=4 train.py
```

### 8.3 Scaling Rules

When scaling from 1 GPU to N GPUs:

```
Effective batch size = batch_per_gpu × N_gpus
Learning rate scaling: LR_new = LR_base × sqrt(N_gpus)  (square root scaling)
  or: LR_new = LR_base × N_gpus  (linear scaling, with warmup)

Example:
  1 GPU:  batch=8,  LR=0.001
  4 GPUs: batch=32, LR=0.002 (sqrt scaling)
  8 GPUs: batch=64, LR=0.003 (sqrt scaling)
```

### 8.4 Synchronized Batch Normalization

Critical when effective per-GPU batch size is small:

```python
# Convert all BN layers to SyncBN
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model.cuda(rank), device_ids=[rank])
```

---

## 9. Training Recipes

### 9.1 Classification (ModelNet40)

```yaml
# Total training time: ~4 hours on 1× RTX 3090
model:
  npoints: 1024
  use_normals: true  # XYZ + normal = 6 channels

training:
  epochs: 200
  batch_size: 24
  optimizer: adam
  lr: 0.001
  weight_decay: 0.0001
  scheduler: step
  step_size: 20
  gamma: 0.7
  
augmentation:
  random_rotation: true  # Full 360° around vertical
  random_scale: [0.8, 1.2]
  point_jitter_sigma: 0.01
  point_jitter_clip: 0.05
  random_dropout: true  # For MSG variant
  
expected_results:
  accuracy: 91.9%  # MSG variant
```

### 9.2 Part Segmentation (ShapeNet)

```yaml
# Total training time: ~12 hours on 1× RTX 3090
model:
  npoints: 2048
  num_parts: 50  # Total parts across all categories
  num_categories: 16

training:
  epochs: 250
  batch_size: 16
  optimizer: adam
  lr: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  eta_min: 0.00001
  warmup_epochs: 5

augmentation:
  random_rotation: true
  random_scale: [0.9, 1.1]
  point_jitter_sigma: 0.005
  random_dropout: false

expected_results:
  mIoU_instance: 85.1%
  mIoU_class: 81.9%
```

### 9.3 Semantic Segmentation (SemanticKITTI)

```yaml
# Total training time: ~48 hours on 4× RTX 3090
model:
  npoints: 8192  # Per sub-cloud (scene is divided into blocks)
  num_classes: 19
  use_msg: true

training:
  epochs: 100
  batch_size: 8  # Per GPU
  num_gpus: 4
  optimizer: adamw
  lr: 0.002
  weight_decay: 0.01
  scheduler: one_cycle
  max_lr: 0.01
  
augmentation:
  random_rotation_z: [-pi, pi]
  random_flip_x: true
  random_flip_y: true
  random_scale: [0.95, 1.05]
  point_jitter_sigma: 0.02
  color_jitter: true  # If using color features

loss:
  type: weighted_cross_entropy
  class_weights: inverse_log_frequency
```

### 9.4 3D Object Detection (KITTI)

```yaml
# Total training time: ~24 hours on 4× RTX 3090
model:
  npoints: 16384
  num_classes: 3  # Car, Pedestrian, Cyclist
  use_two_stage: true

training:
  epochs: 80
  batch_size: 4  # Per GPU (large point clouds)
  num_gpus: 4
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.01
  scheduler: one_cycle
  max_lr: 0.01
  pct_start: 0.4
  grad_clip_norm: 10.0

augmentation:
  gt_augmentation: true
  max_samples: {car: 15, pedestrian: 10, cyclist: 10}
  random_rotation_z: [-0.785, 0.785]  # ±45°
  random_flip_x: true
  random_scale: [0.95, 1.05]
  point_jitter_sigma: 0.01
  
loss:
  cls_weight: 1.0
  reg_weight: 2.0
  dir_weight: 0.2
  
nms:
  iou_threshold: 0.7  # For car proposals
  score_threshold: 0.1
  max_proposals: 100

expected_results:  # KITTI val, moderate difficulty
  car_3d_ap: ~75%
  pedestrian_3d_ap: ~55%
  cyclist_3d_ap: ~60%
```

---

## 10. Training Monitoring

### 10.1 Key Metrics to Track

```python
# Log these every N iterations:
metrics = {
    'train/loss': total_loss,
    'train/cls_loss': cls_loss,
    'train/reg_loss': reg_loss,  # Detection only
    'train/lr': scheduler.get_last_lr()[0],
    'train/grad_norm': grad_norm,
    'train/batch_time': batch_time,
    'memory/allocated_gb': torch.cuda.memory_allocated() / 1e9,
    'memory/reserved_gb': torch.cuda.memory_reserved() / 1e9,
}

# Log these every epoch:
val_metrics = {
    'val/loss': val_loss,
    'val/accuracy': accuracy,        # Classification
    'val/mIoU': mean_iou,            # Segmentation
    'val/mAP': mean_ap,              # Detection
}
```

### 10.2 Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False  # Don't stop
        self.counter += 1
        return self.counter >= self.patience  # Stop if patience exceeded
```

### 10.3 Checkpointing

```python
def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
    }, path)

# Save best model and periodic checkpoints
if val_metric > best_metric:
    best_metric = val_metric
    save_checkpoint(model, optimizer, scheduler, epoch, best_metric,
                    'checkpoints/best_model.pth')

if epoch % 10 == 0:
    save_checkpoint(model, optimizer, scheduler, epoch, best_metric,
                    f'checkpoints/epoch_{epoch:03d}.pth')
```

---

## 11. Common Training Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| NaN loss | Loss becomes NaN after few epochs | Reduce LR, check for zero-division in custom ops, enable AMP grad scaling |
| Slow convergence | Loss plateaus early | Increase LR, add warmup, check data augmentation strength |
| Overfitting | Train acc high, val acc low | Increase dropout, stronger augmentation, weight decay, early stopping |
| OOM errors | CUDA out of memory | Reduce batch size, enable AMP, reduce npoints/nsample |
| FPS slow on CPU | Training bottlenecked by data loading | Move FPS to GPU, use CUDA implementation, increase num_workers |
| Unstable BN | Oscillating loss with small batch | Use GroupNorm or SyncBN, increase accumulation steps |
| Class imbalance | Poor detection of rare classes | Class-weighted loss, GT-augmentation, focal loss |
| Heading ambiguity | High rotation error | Use sin/cos parameterization, direction classification |
