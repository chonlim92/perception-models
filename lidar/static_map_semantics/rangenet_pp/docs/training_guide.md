# RangeNet++: Training Guide

## Overview

Training RangeNet++ involves learning per-pixel semantic segmentation on range images derived from 3D LiDAR point clouds. The training process requires careful handling of empty pixels, class imbalance, and data augmentation specific to the range image representation.

---

## Training Pipeline

```
Raw Point Clouds (.bin)
       |
       v
  Spherical Projection  -->  Range Images (64x2048x5)
       |
       v
  Data Augmentation (random rotation, flip, dropout)
       |
       v
  Normalization (per-channel mean/std)
       |
       v
  DarkNet-53 + U-Net (forward pass)
       |
       v
  Weighted Cross-Entropy Loss (with mask for empty pixels)
       |
       v
  Backpropagation + SGD/Adam Update
```

---

## Range Image Training

### Input Preparation

1. **Load point cloud** from `.bin` file (N x 4: x, y, z, intensity).
2. **Load labels** from `.label` file (N x 1: semantic class per point).
3. **Project to range image:**
   - Point coordinates -> 5-channel range image (64 x 2048 x 5)
   - Point labels -> label image (64 x 2048 x 1)
4. **Generate validity mask:** Binary mask indicating pixels with valid projections.
5. **Normalize** input channels using training set statistics.

### Training Configuration

```yaml
# Typical training hyperparameters
batch_size: 4                    # Limited by GPU memory at 64x2048
epochs: 150                      # ~150 epochs for convergence
learning_rate: 0.01              # Initial learning rate
lr_schedule: cosine_annealing    # With warm restarts
warmup_epochs: 1                 # Linear warmup
weight_decay: 0.0001             # L2 regularization
momentum: 0.9                    # SGD momentum
optimizer: SGD                   # SGD with momentum (or Adam)
```

---

## Handling Empty Pixels (Masking)

### Problem

Not all pixels in the range image correspond to valid LiDAR points. Approximately 36% of pixels at 64x2048 resolution are empty (no point projects there). These occur due to:
- Points beyond sensor range
- Sky regions (no returns)
- Gaps between beams at far distances

### Masking Strategy

```python
# During data loading
valid_mask = (range_image[:, :, 0] > 0)  # Range > 0 means valid point

# During loss computation
logits = model(range_image)  # (B, C, H, W)
loss = F.cross_entropy(logits, labels, reduction='none')  # (B, H, W)

# Apply validity mask
loss = loss * valid_mask.float()

# Normalize by number of valid pixels
loss = loss.sum() / valid_mask.float().sum()
```

### Empty Pixel Values

Empty pixels are set to:
- All 5 input channels = 0.0 (before normalization)
- Label = 0 (unlabeled, also masked from loss)

---

## Class Weighting (Inverse Frequency)

### Problem

SemanticKITTI has severe class imbalance. Road and vegetation dominate (~33% combined), while motorcyclist and traffic-sign are extremely rare (<0.2%).

### Inverse Frequency Weighting

```python
# Compute class frequencies from training set
class_counts = count_points_per_class(training_set)
total_points = sum(class_counts)
class_freq = class_counts / total_points

# Inverse frequency weight (with smoothing)
epsilon = 1e-3  # Prevent division by zero
class_weights = 1.0 / (class_freq + epsilon)

# Normalize weights to sum to num_classes
class_weights = class_weights / class_weights.sum() * num_classes
```

### Alternative: Log-Inverse Weighting

```python
# Logarithmic dampening (prevents extreme weights)
class_weights = 1.0 / np.log(1.02 + class_freq)
```

### Typical Class Weights (SemanticKITTI)

| Class | Name | Approx. Weight |
|-------|------|---------------|
| 1 | car | 3.5 |
| 2 | bicycle | 15.0 |
| 3 | motorcycle | 20.0 |
| 4 | truck | 8.0 |
| 5 | other-vehicle | 12.0 |
| 6 | person | 25.0 |
| 7 | bicyclist | 30.0 |
| 8 | motorcyclist | 50.0 |
| 9 | road | 1.0 |
| 10 | parking | 4.0 |
| 11 | sidewalk | 2.0 |
| 12 | other-ground | 6.0 |
| 13 | building | 1.5 |
| 14 | fence | 4.0 |
| 15 | vegetation | 1.0 |
| 16 | trunk | 8.0 |
| 17 | terrain | 2.5 |
| 18 | pole | 12.0 |
| 19 | traffic-sign | 18.0 |

### Implementation in Loss

```python
# Weighted cross-entropy loss
weights = torch.tensor(class_weights, device=device)
criterion = nn.CrossEntropyLoss(
    weight=weights,
    ignore_index=0,     # Ignore unlabeled class
    reduction='none'    # Apply mask manually
)
```

---

## Data Augmentation

### Random Rotation (Yaw)

Rotate the point cloud around the z-axis before projection. In range image space, this corresponds to a horizontal circular shift.

```python
# Random rotation around z-axis
angle = np.random.uniform(0, 2 * np.pi)

# In range image: horizontal shift
shift = int(angle / (2 * np.pi) * W)
range_image = np.roll(range_image, shift, axis=1)
label_image = np.roll(label_image, shift, axis=1)
mask = np.roll(mask, shift, axis=1)

# Also update x, y channels (rotate coordinates)
cos_a, sin_a = np.cos(angle), np.sin(angle)
x_new = range_image[:, :, 1] * cos_a - range_image[:, :, 2] * sin_a
y_new = range_image[:, :, 1] * sin_a + range_image[:, :, 2] * cos_a
range_image[:, :, 1] = x_new
range_image[:, :, 2] = y_new
```

### Random Horizontal Flip

Flip the range image horizontally (mirror the scene left-to-right).

```python
if np.random.random() > 0.5:
    range_image = np.flip(range_image, axis=1).copy()
    label_image = np.flip(label_image, axis=1).copy()
    mask = np.flip(mask, axis=1).copy()
    
    # Negate y-coordinate after flip
    range_image[:, :, 2] = -range_image[:, :, 2]
```

### Random Point Dropout

Randomly drop a percentage of valid pixels to simulate sensor noise and varying point density.

```python
# Random dropout of valid pixels
dropout_rate = np.random.uniform(0.0, 0.1)  # Drop 0-10% of points
dropout_mask = np.random.random(range_image.shape[:2]) > dropout_rate
range_image = range_image * dropout_mask[:, :, np.newaxis]
# Mark dropped pixels as invalid
mask = mask & dropout_mask
```

### Random Intensity Jitter

Add noise to the intensity channel to improve robustness.

```python
# Intensity noise
noise = np.random.normal(0, 0.02, range_image[:, :, 4].shape)
range_image[:, :, 4] = np.clip(range_image[:, :, 4] + noise, 0, 1)
```

### Augmentation Summary

| Augmentation | Probability | Range | Effect |
|-------------|-------------|-------|--------|
| Random rotation | 1.0 | [0, 360] deg | Horizontal shift in range image |
| Horizontal flip | 0.5 | - | Mirror scene |
| Point dropout | 0.5 | [0, 10]% | Simulate sparse returns |
| Intensity jitter | 0.5 | N(0, 0.02) | Robustness to intensity variation |

---

## Learning Rate Schedule

### Cosine Annealing with Warm Restarts

```python
# PyTorch implementation
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,          # Initial restart period (epochs)
    T_mult=2,        # Period multiplier after each restart
    eta_min=1e-5     # Minimum learning rate
)
```

### Warmup

```python
# Linear warmup for first epoch
def warmup_lr(epoch, warmup_epochs=1, base_lr=0.01):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr
```

---

## Training Procedure

### Step-by-Step

1. **Prepare dataset:**
   - Pre-compute range images from point clouds (or compute on-the-fly)
   - Compute training set mean/std per channel
   - Compute class weights from label statistics

2. **Initialize model:**
   - DarkNet-53 encoder (optionally pre-trained on ImageNet)
   - Random initialization for decoder and classification head

3. **Training loop:**
   ```python
   for epoch in range(num_epochs):
       model.train()
       for batch in dataloader:
           range_img, labels, mask = batch
           
           # Forward pass
           logits = model(range_img)
           
           # Compute weighted, masked loss
           loss = criterion(logits, labels)
           loss = (loss * mask.float()).sum() / mask.float().sum()
           
           # Backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
       
       scheduler.step()
       
       # Validation
       if epoch % val_interval == 0:
           validate(model, val_loader)
   ```

4. **Validation:**
   - Compute mIoU on sequence 08 (validation set)
   - Save best checkpoint based on validation mIoU

5. **Post-training:**
   - Apply KNN post-processing
   - Evaluate on test set (submit to benchmark server)

### Pre-training

- **ImageNet pre-training:** DarkNet-53 can be initialized with weights pre-trained on ImageNet for image classification. This provides faster convergence and slightly better final accuracy (~0.5-1.0 mIoU improvement).
- **Adaptation:** Since the input has 5 channels (not 3 RGB), the first convolutional layer is randomly initialized or adapted by duplicating/averaging the pre-trained 3-channel weights.

---

## Training Tips

### Convergence

- Monitor training loss and validation mIoU per epoch.
- Typical convergence: ~100-150 epochs for full accuracy.
- If training loss stagnates, try reducing learning rate or increasing augmentation.

### GPU Memory Management

- Batch size 4 for 64x2048 on a 12GB GPU.
- Use gradient accumulation for larger effective batch sizes.
- Mixed precision training (FP16) can reduce memory by ~40%.

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Class imbalance | Rare classes always predicted as background | Increase class weights, use focal loss |
| Empty pixel artifacts | Model predicts for empty regions | Ensure proper masking in loss |
| Overfitting | Val mIoU plateaus while train loss decreases | More augmentation, dropout, weight decay |
| Unstable training | Loss spikes | Reduce learning rate, gradient clipping |
| Boundary errors | Poor per-class IoU on thin objects | KNN post-processing, boundary-aware loss |

### Mixed Precision Training

```python
# PyTorch AMP
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast():
        logits = model(range_img)
        loss = criterion(logits, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Distributed Training

For multi-GPU training:

```python
# PyTorch DistributedDataParallel
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

- Linear scaling rule: LR = base_lr * num_gpus
- Batch size per GPU remains the same (effective batch = batch_per_gpu * num_gpus)
- Sync batch normalization across GPUs for best results

---

## Checkpointing

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_miou': best_miou,
}, f'checkpoint_epoch_{epoch}.pth')

# Resume from checkpoint
checkpoint = torch.load('checkpoint_epoch_100.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```
