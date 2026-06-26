# Cylinder3D: Training Guide

## Overview

This guide covers the complete training procedure for Cylinder3D, including loss functions, optimization, data augmentation, distributed training, and practical tips for achieving optimal performance.

---

## Loss Function

Cylinder3D uses a **combined loss** of weighted cross-entropy and Lovasz-softmax:

```
L_total = L_wce + L_lovasz
```

### Weighted Cross-Entropy Loss

```python
L_wce = -sum(w_c * y_c * log(p_c)) / N

where:
  w_c = class weight for class c
  y_c = ground truth one-hot label
  p_c = predicted probability for class c
  N   = number of valid points (excluding ignore_label)
```

**Class weight computation:**

```python
# Inverse square root frequency weighting
class_counts = count_points_per_class(training_set)
class_freq = class_counts / class_counts.sum()
class_weights = 1.0 / np.sqrt(class_freq)
class_weights = class_weights / class_weights.sum() * num_classes

# Alternative: inverse log frequency (used in some configs)
class_weights = 1.0 / np.log(1.02 + class_freq)
```

**Typical SemanticKITTI class weights:**

| Class | Weight | Class | Weight |
|-------|--------|-------|--------|
| car | 2.8 | parking | 5.1 |
| bicycle | 12.3 | sidewalk | 2.4 |
| motorcycle | 14.7 | other-ground | 5.8 |
| truck | 6.2 | building | 1.9 |
| other-vehicle | 5.9 | fence | 3.5 |
| person | 9.8 | vegetation | 1.6 |
| bicyclist | 8.4 | trunk | 5.3 |
| motorcyclist | 18.2 | terrain | 2.4 |
| road | 1.2 | pole | 5.6 |
| | | traffic-sign | 6.1 |

### Lovasz-Softmax Loss

The Lovasz-softmax loss directly optimizes the IoU metric through a convex surrogate:

```python
from lovasz_losses import lovasz_softmax

L_lovasz = lovasz_softmax(
    probas=F.softmax(logits, dim=1),  # predicted probabilities
    labels=targets,                    # ground truth labels
    classes='present',                 # only compute for classes present in batch
    ignore=0                           # ignore unlabeled class
)
```

**Properties:**
- Differentiable surrogate for the Jaccard index (IoU)
- Directly optimizes per-class IoU, addressing class imbalance
- Complements cross-entropy by providing gradient signal proportional to IoU improvement
- `classes='present'` avoids penalizing absent classes in a mini-batch

### Loss Weighting

```python
# Default configuration
loss = weighted_cross_entropy(logits, labels) + lovasz_softmax(probs, labels)

# Some configurations use a scaling factor
loss = 1.0 * L_wce + 1.0 * L_lovasz  # equal weighting (default)
# or
loss = 1.0 * L_wce + 1.5 * L_lovasz  # slightly more emphasis on IoU
```

---

## Optimizer Configuration

### Primary Optimizer: Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,              # initial learning rate
    betas=(0.9, 0.999),   # momentum parameters
    eps=1e-8,             # numerical stability
    weight_decay=1e-4     # L2 regularization
)
```

### Learning Rate Schedule

**Cosine annealing with warm-up:**

```python
# Warm-up for first 5% of training
warmup_epochs = 2  # out of 40 total

# Cosine decay from peak lr to min lr
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs - warmup_epochs,
    eta_min=1e-6  # minimum learning rate
)

# Warm-up implementation
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + cos(pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

**Alternative: Step decay (simpler):**

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=15,       # decay every 15 epochs
    gamma=0.1           # multiply lr by 0.1
)
```

---

## Data Augmentation

### Geometric Augmentations

```python
# 1. Random Rotation around Z-axis
angle = np.random.uniform(-np.pi, np.pi)  # full 360° rotation
rotation_matrix = np.array([
    [cos(angle), -sin(angle), 0],
    [sin(angle),  cos(angle), 0],
    [0,           0,          1]
])
points[:, :3] = points[:, :3] @ rotation_matrix.T

# 2. Random Flip (X-axis and/or Y-axis)
if np.random.random() > 0.5:
    points[:, 0] = -points[:, 0]  # flip X
if np.random.random() > 0.5:
    points[:, 1] = -points[:, 1]  # flip Y

# 3. Random Scaling
scale = np.random.uniform(0.95, 1.05)  # ±5% scale
points[:, :3] *= scale

# 4. Random Translation
translate = np.random.uniform(-0.2, 0.2, size=(1, 3))  # ±20cm
points[:, :3] += translate

# 5. Random Jitter (point-level noise)
noise = np.random.normal(0, 0.02, size=points[:, :3].shape)  # 2cm std
points[:, :3] += noise
```

### Point-Level Augmentations

```python
# 6. Random Point Dropout
dropout_ratio = np.random.uniform(0.0, 0.1)  # drop up to 10% of points
mask = np.random.random(len(points)) > dropout_ratio
points = points[mask]
labels = labels[mask]

# 7. Random Intensity Jitter
intensity_noise = np.random.normal(0, 0.05, size=points[:, 3:4].shape)
points[:, 3:4] = np.clip(points[:, 3:4] + intensity_noise, 0.0, 1.0)
```

### Mix3D Augmentation (Optional Enhancement)

```python
# Combine two training scenes for additional diversity
# Mix3D: paste points from another scene into current scene
if np.random.random() > 0.5:
    other_points, other_labels = load_random_scene()
    # Apply random transform to other scene
    other_points = apply_random_transform(other_points)
    # Concatenate
    points = np.concatenate([points, other_points], axis=0)
    labels = np.concatenate([labels, other_labels], axis=0)
```

### Augmentation Probabilities

| Augmentation | Probability | Range |
|-------------|-------------|-------|
| Z-axis rotation | 1.0 | [-pi, pi] |
| X-flip | 0.5 | binary |
| Y-flip | 0.5 | binary |
| Scale | 1.0 | [0.95, 1.05] |
| Translation | 1.0 | [-0.2m, 0.2m] |
| Point jitter | 0.5 | sigma=0.02m |
| Point dropout | 0.5 | [0%, 10%] |
| Intensity jitter | 0.5 | sigma=0.05 |

---

## Training Configuration

### Hyperparameters

| Parameter | SemanticKITTI | nuScenes |
|-----------|---------------|----------|
| Batch size (per GPU) | 2 | 4 |
| Total epochs | 40 | 50 |
| Initial learning rate | 1e-3 | 1e-3 |
| Minimum learning rate | 1e-6 | 1e-6 |
| Weight decay | 1e-4 | 1e-4 |
| Warmup epochs | 2 | 2 |
| Gradient clip norm | 10.0 | 10.0 |
| Grid size | 480×360×32 | 480×360×32 |
| Max points per voxel | 10 | 10 |
| Ignore label | 0 | 0 |

### Training Duration

| Configuration | GPUs | Batch Size | Time per Epoch | Total Time |
|---------------|------|-----------|----------------|------------|
| 1× RTX 3090 | 1 | 2 | ~45 min | ~30 hours |
| 2× RTX 3090 | 2 | 4 | ~25 min | ~17 hours |
| 4× V100 (32GB) | 4 | 8 | ~15 min | ~10 hours |
| 8× A100 (40GB) | 8 | 16 | ~8 min | ~5.5 hours |

---

## Mixed Precision Training

Mixed precision (FP16) training reduces memory usage and improves throughput:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch['points'], batch['coords'])
        loss = criterion(outputs, batch['labels'])
    
    # Scaled backward pass
    scaler.scale(loss).backward()
    
    # Unscale gradients for clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    
    # Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- ~40% reduction in GPU memory usage
- ~1.5-2× training speed improvement on Volta/Ampere GPUs
- Minimal impact on final accuracy (<0.1% mIoU difference)

**Caveats:**
- Sparse convolutions in some `spconv` versions may not fully support FP16
- Keep loss computation in FP32 (autocast handles this automatically)
- Monitor for NaN/Inf gradients (GradScaler handles overflow)

---

## Multi-GPU Training with DistributedDataParallel

### Setup

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def train(rank, world_size, args):
    setup_distributed(rank, world_size)
    
    # Model
    model = Cylinder3D(num_classes=20).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Data
    train_dataset = SemanticKITTIDataset(split='train')
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=sparse_collate_fn
    )
    
    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # important for shuffling
        train_one_epoch(model, train_loader, optimizer, scaler, rank)
```

### Launch Command

```bash
# Single node, 4 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train.py \
    --batch_size_per_gpu 2 \
    --epochs 40 \
    --lr 1e-3

# Alternative with torchrun (PyTorch >= 1.10)
torchrun --nproc_per_node=4 --master_port=29500 \
    train.py --batch_size_per_gpu 2 --epochs 40
```

### Scaling Learning Rate

When using multiple GPUs, scale the learning rate linearly:

```python
# Linear scaling rule
effective_batch_size = batch_size_per_gpu * world_size
base_lr = 1e-3  # for batch_size 2
scaled_lr = base_lr * (effective_batch_size / 2)  # scale relative to base
# Cap at 4e-3 to avoid instability
scaled_lr = min(scaled_lr, 4e-3)
```

---

## Tips for Handling Class Imbalance

### Strategy 1: Class-Weighted Loss (Primary)

Already described above. The weighted cross-entropy ensures rare classes contribute meaningful gradients.

### Strategy 2: Lovasz-Softmax (Primary)

Directly optimizes per-class IoU, naturally balancing the contribution of rare classes.

### Strategy 3: Class-Balanced Sampling

```python
# Over-sample scans containing rare classes
class_scan_counts = compute_class_presence_per_scan(dataset)

# Weight each scan by inverse frequency of its rarest class
scan_weights = []
for scan_idx in range(len(dataset)):
    classes_present = get_classes_in_scan(scan_idx)
    min_freq = min(class_freq[c] for c in classes_present)
    scan_weights.append(1.0 / np.sqrt(min_freq))

sampler = WeightedRandomSampler(scan_weights, num_samples=len(dataset))
```

### Strategy 4: Copy-Paste Augmentation for Rare Objects

```python
# Extract instances of rare classes and paste into training scenes
rare_classes = ['bicycle', 'motorcycle', 'person', 'motorcyclist']

def copy_paste_augment(points, labels, instance_bank):
    """Paste rare-class instances into the current scene."""
    for cls in rare_classes:
        if np.random.random() > 0.5:
            instance = random.choice(instance_bank[cls])
            # Random placement
            placement = random_ground_position(points)
            instance_shifted = instance + placement
            points = np.concatenate([points, instance_shifted[:, :4]])
            labels = np.concatenate([labels, instance_shifted[:, 4]])
    return points, labels
```

### Strategy 5: Focal Loss Variant (Optional)

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

---

## Training Monitoring

### Key Metrics to Track

| Metric | Frequency | Expected Trend |
|--------|-----------|----------------|
| Training loss | Every iteration | Steady decrease |
| Validation mIoU | Every epoch | Increase, plateau ~epoch 30-35 |
| Per-class IoU | Every 5 epochs | Rare classes improve slowly |
| Learning rate | Every epoch | Follows schedule |
| GPU memory | Continuous | Stable (watch for leaks) |
| Gradient norm | Every iteration | Should stay <10 after clipping |

### Logging Example

```python
# Weights & Biases logging
import wandb

wandb.init(project="cylinder3d", config=args)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_miou, per_class_iou = validate(...)
    
    wandb.log({
        'epoch': epoch,
        'train/loss': train_loss,
        'val/mIoU': val_miou,
        'val/car_iou': per_class_iou['car'],
        'val/person_iou': per_class_iou['person'],
        'lr': optimizer.param_groups[0]['lr'],
    })
```

### Expected Training Curves (SemanticKITTI)

| Epoch | Train Loss | Val mIoU |
|-------|-----------|----------|
| 1 | 2.8 | 15% |
| 5 | 1.2 | 45% |
| 10 | 0.8 | 55% |
| 20 | 0.5 | 62% |
| 30 | 0.35 | 65% |
| 40 | 0.28 | 67% |

---

## Checkpointing and Resumption

```python
# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),  # unwrap DDP
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_miou': best_miou,
    }, path)

# Resume from checkpoint
def load_checkpoint(model, optimizer, scheduler, scaler, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_miou']
```

---

## Common Training Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| NaN loss | Loss becomes NaN after few epochs | Reduce lr to 5e-4; check data preprocessing |
| GPU OOM | CUDA out of memory | Reduce batch size; enable mixed precision |
| Low rare-class IoU | Person/bicycle IoU stays near 0 | Increase class weights; add copy-paste augmentation |
| Overfitting | Val mIoU peaks early then declines | Add more augmentation; increase weight decay |
| Slow convergence | mIoU stuck below 55% after 20 epochs | Check learning rate; verify data loading correctness |
| spconv errors | Hash collision or index errors | Update spconv; reduce grid size slightly |
| Gradient explosion | Gradient norm spikes >100 | Ensure grad clipping is active; check for label errors |

---

## Reproducibility

```python
# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable for exact reproducibility

# Note: torch.backends.cudnn.benchmark = True is faster but non-deterministic
# For production training, use benchmark=True for speed
```

---

## Hardware Recommendations

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| GPU | RTX 2080 Ti (11GB) | RTX 3090 (24GB) | A100 (40/80GB) |
| GPU Count | 1 | 2-4 | 8 |
| RAM | 32 GB | 64 GB | 128 GB |
| Storage | 256 GB SSD | 1 TB NVMe | 2 TB NVMe |
| CPU | 8 cores | 16 cores | 32+ cores |

**Storage note:** SemanticKITTI requires ~80 GB; nuScenes requires ~400 GB. Use NVMe SSD for data loading to avoid I/O bottlenecks.
