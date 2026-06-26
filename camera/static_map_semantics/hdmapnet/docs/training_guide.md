# HDMapNet Training Guide

A practical guide for PyTorch engineers on training HDMapNet for BEV semantic map prediction from multi-camera inputs.

---

## 1. Prerequisites and Environment Setup

### Python and PyTorch

- **Python**: 3.8 or higher (3.9 recommended for best compatibility)
- **PyTorch**: 1.9+ with CUDA support (1.11+ preferred for stable AMP)
- **CUDA**: 11.1 or higher (11.3+ for A100 support)
- **cuDNN**: 8.1+

### Key Dependencies

```bash
pip install nuscenes-devkit==1.1.9
pip install efficientnet-pytorch==0.7.1
pip install torchvision>=0.10.0
pip install opencv-python>=4.5.0
pip install scikit-image>=0.18.0
pip install tensorboard>=2.6.0
pip install einops>=0.3.0
pip install pyquaternion>=0.9.9
pip install numpy>=1.20.0
pip install matplotlib>=3.4.0
pip install tqdm
```

### Recommended: Conda Environment

```bash
conda create -n hdmapnet python=3.9 -y
conda activate hdmapnet
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Hardware Requirements

| Setup | GPUs | VRAM | Batch Size | Notes |
|-------|------|------|------------|-------|
| Recommended | 4x NVIDIA A100 | 40GB each | 4 per GPU | Full resolution, fastest convergence |
| Standard | 4x NVIDIA V100 | 32GB each | 3 per GPU | Slightly reduced resolution |
| Minimum | 2x NVIDIA RTX 3090 | 24GB each | 2 per GPU | Gradient accumulation needed |
| Dev/Debug | 1x RTX 3090 | 24GB | 1 | Mini dataset only, not for full training |

**Storage**: At least 500GB free for the full nuScenes dataset plus checkpoints.

**RAM**: 64GB+ system memory recommended (data loading with multiple workers).

---

## 2. Dataset Preparation

### nuScenes Data Structure

Download the full nuScenes dataset (v1.0-trainval, approximately 400GB) or the mini split (~4GB) for development:

```bash
# Directory structure after extraction
data/
  nuscenes/
    maps/               # HD map raster and vector data
    samples/            # Keyframe sensor data (images, lidar, radar)
      CAM_FRONT/
      CAM_FRONT_LEFT/
      CAM_FRONT_RIGHT/
      CAM_BACK/
      CAM_BACK_LEFT/
      CAM_BACK_RIGHT/
    sweeps/             # Non-keyframe sensor data (not used)
    v1.0-trainval/      # Metadata JSON files
      sample.json
      sample_data.json
      ego_pose.json
      calibrated_sensor.json
      map.json
      ...
```

### Symlink Setup

If your data lives on a separate drive, create symlinks:

```bash
mkdir -p ./data
ln -s /mnt/datasets/nuscenes ./data/nuscenes
```

For Windows:

```powershell
mklink /D .\data\nuscenes D:\datasets\nuscenes
```

### Map Annotation Format

HDMapNet uses three map element classes from the nuScenes map expansion:

1. **Road boundaries** (lane dividers, road edges)
2. **Lane dividers** (dashed lines separating lanes)
3. **Pedestrian crossings**

Each map element is stored as a set of vectorized polylines. A polyline is a sequence of 2D points in the global coordinate frame:

```python
# Example: a lane divider polyline
polyline = np.array([
    [305.2, 1102.4],
    [305.8, 1104.1],
    [306.5, 1105.9],
    ...
])  # shape: (N_points, 2) in global (x, y) meters
```

### Rasterization of Vector Maps to BEV Ground Truth

The training pipeline rasterizes these polylines into binary BEV masks. This happens in the dataset `__getitem__` method:

```python
import cv2
import numpy as np

def vectormap_to_raster(vectors, patch_box, patch_angle, canvas_size):
    """
    Rasterize vector polylines to a BEV binary mask.

    Args:
        vectors: list of polylines, each shape (N, 2) in global coords
        patch_box: (x_center, y_center, height, width) of BEV patch
        patch_angle: rotation angle of the patch (ego heading)
        canvas_size: (H, W) of output raster, e.g. (200, 200)

    Returns:
        mask: binary array of shape (H, W)
    """
    mask = np.zeros(canvas_size, dtype=np.uint8)

    for polyline in vectors:
        # Transform global coords to patch-local pixel coords
        local_pts = global_to_patch(polyline, patch_box, patch_angle, canvas_size)
        local_pts = local_pts.astype(np.int32)

        # Draw polyline with thickness (3 pixels for boundaries)
        cv2.polylines(mask, [local_pts], isClosed=False, color=1, thickness=3)

    return mask
```

Ground truth consists of three channels (one per class) at 200x200 resolution:

```python
# Final GT shape: (3, 200, 200) - multi-label binary masks
gt_semantic = np.stack([road_boundary_mask, lane_divider_mask, crossing_mask], axis=0)
```

### Data Splits

The official nuScenes split:

- **Train**: 700 scenes (28,130 keyframe samples)
- **Val**: 150 scenes (6,019 keyframe samples)

Each keyframe sample provides 6 surround-view camera images with associated intrinsics and extrinsics.

---

## 3. Loss Functions

This is the most critical section for understanding training behavior. HDMapNet uses a multi-task loss combining semantic, instance, and direction heads.

### 3.1 Semantic Segmentation Loss

Binary cross-entropy (BCE) per class. The map classes are **not mutually exclusive** (a pixel can be both a road boundary and a lane divider), so we use independent sigmoid activations rather than softmax:

```python
import torch
import torch.nn.functional as F

def semantic_loss(pred, target):
    """
    Args:
        pred: (B, C, H, W) raw logits, C=3 classes
        target: (B, C, H, W) binary ground truth

    Returns:
        loss: scalar
    """
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
    return loss
```

**Tip**: For class imbalance (road boundaries are thin lines), use pos_weight:

```python
# Compute positive class weight from dataset statistics
# Approximate ratio of negative to positive pixels per class
pos_weight = torch.tensor([5.0, 4.0, 8.0]).view(1, 3, 1, 1).cuda()
loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
```

### 3.2 Instance Embedding Loss (Discriminative Loss)

Based on Brabandere et al., "Semantic Instance Segmentation with a Discriminative Loss Function" (2017). The network predicts a D-dimensional embedding for each BEV pixel, and the loss encourages:

- Embeddings of the **same instance** to cluster together (pull loss)
- Cluster centers of **different instances** to separate (push loss)
- All cluster centers to stay near the origin (regularization)

```python
def discriminative_loss(embedding, instance_mask, delta_v=0.5, delta_d=3.0):
    """
    Args:
        embedding: (B, D, H, W) - predicted embeddings, D=16 typical
        instance_mask: (B, H, W) - integer instance IDs (0 = background)
        delta_v: pull margin (hinge for intra-cluster variance)
        delta_d: push margin (hinge for inter-cluster distance)

    Returns:
        loss_pull, loss_push, loss_reg: scalar tensors
    """
    batch_size = embedding.shape[0]
    loss_pull_total = 0.0
    loss_push_total = 0.0
    loss_reg_total = 0.0

    for b in range(batch_size):
        emb = embedding[b]  # (D, H, W)
        inst = instance_mask[b]  # (H, W)

        instance_ids = inst.unique()
        instance_ids = instance_ids[instance_ids != 0]  # exclude background
        n_instances = len(instance_ids)

        if n_instances == 0:
            continue

        centers = []
        for inst_id in instance_ids:
            mask = (inst == inst_id).unsqueeze(0).float()  # (1, H, W)
            center = (emb * mask).sum(dim=(1, 2)) / mask.sum()  # (D,)
            centers.append(center)

            # Pull loss: ||e_i - mu_c|| - delta_v, clamped to >= 0
            diff = emb - center.view(-1, 1, 1)  # (D, H, W)
            dist = torch.norm(diff, dim=0) * mask.squeeze(0)  # (H, W)
            pull = torch.clamp(dist - delta_v, min=0.0) ** 2
            loss_pull_total += pull.sum() / (mask.sum() + 1e-6)

        centers = torch.stack(centers)  # (N, D)

        # Push loss: 2*delta_d - ||mu_a - mu_b||, clamped to >= 0
        if n_instances > 1:
            for i in range(n_instances):
                for j in range(i + 1, n_instances):
                    dist = torch.norm(centers[i] - centers[j])
                    push = torch.clamp(2 * delta_d - dist, min=0.0) ** 2
                    loss_push_total += push
            loss_push_total /= (n_instances * (n_instances - 1) / 2)

        # Regularization: centers toward origin
        loss_reg_total += torch.norm(centers, dim=1).mean()

    loss_pull = loss_pull_total / batch_size
    loss_push = loss_push_total / batch_size
    loss_reg = loss_reg_total / batch_size

    return loss_pull, loss_push, loss_reg
```

**Key hyperparameters**:

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `delta_v` (pull margin) | 0.5 | Larger = more tolerance for intra-cluster spread |
| `delta_d` (push margin) | 3.0 | Larger = clusters pushed further apart |
| Embedding dim `D` | 16 | Higher = easier separation but more compute |

### 3.3 Direction Loss

Predicts the local direction of each map element at every positive semantic pixel. Direction is represented as (cos_theta, sin_theta) to avoid angle wrapping issues:

```python
def direction_loss(pred_dir, gt_dir, semantic_mask):
    """
    Args:
        pred_dir: (B, 2, H, W) - predicted (cos, sin) direction
        gt_dir: (B, 2, H, W) - ground truth direction from polyline tangent
        semantic_mask: (B, 1, H, W) - binary mask of positive semantic pixels

    Returns:
        loss: scalar
    """
    # Only compute loss on pixels that belong to a map element
    mask = semantic_mask.expand_as(pred_dir)  # (B, 2, H, W)
    n_positive = mask.sum() + 1e-6

    loss = F.l1_loss(pred_dir * mask, gt_dir * mask, reduction='sum') / n_positive
    return loss
```

The ground truth direction is computed from the polyline tangent vector at each rasterized pixel. For a polyline segment from point A to point B:

```python
tangent = B - A
tangent_normalized = tangent / (np.linalg.norm(tangent) + 1e-6)
cos_theta, sin_theta = tangent_normalized[0], tangent_normalized[1]
```

### 3.4 Total Loss

```python
# Loss combination
L_total = w_sem * L_semantic + w_inst * L_instance + w_dir * L_direction

# Where L_instance = pull + push + 0.001 * reg
L_instance = loss_pull + loss_push + 0.001 * loss_reg
```

**Typical loss weights**:

```yaml
loss_weights:
  semantic: 1.0
  instance: 1.0
  direction: 0.5
```

These weights work well as-is. If instance separation is poor, increase `w_inst` to 2.0. If direction predictions are noisy, increase `w_dir` to 1.0.

---

## 4. Training Configuration

### Full Configuration Example

```yaml
# config/hdmapnet_efficientb4_200x200.yaml

model:
  backbone: efficientnet-b4
  bev_encoder_channels: 256
  bev_encoder_layers: 4
  embedding_dim: 16
  num_classes: 3          # road boundary, lane divider, crossing
  direction_dim: 2        # (cos, sin)
  depth_bins: 64          # for view transform (LSS-style)
  depth_range: [2.0, 50.0]

data:
  dataset: nuscenes
  version: v1.0-trainval
  dataroot: ./data/nuscenes
  input_size: [128, 352]  # H, W per camera (downsampled from 900x1600)
  bev_size: [200, 200]    # BEV grid resolution
  bev_resolution: 0.3     # meters per pixel
  bev_range: 60.0         # total coverage = 200 * 0.3 = 60m in each direction
  num_cameras: 6
  augmentation:
    random_flip: true      # horizontal flip (flip both images and BEV GT)
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    bev_rotation: 0.0      # degrees, set >0 for BEV augmentation

training:
  optimizer: adam
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  scheduler: cosine_annealing
  warmup_epochs: 2
  total_epochs: 30
  batch_size_per_gpu: 4
  num_workers: 4
  gradient_accumulation_steps: 1

  loss_weights:
    semantic: 1.0
    instance: 1.0
    direction: 0.5

  discriminative_loss:
    delta_v: 0.5
    delta_d: 3.0
    embedding_dim: 16

distributed:
  backend: nccl
  sync_batchnorm: true

checkpoint:
  save_dir: ./checkpoints/hdmapnet
  save_every: 5           # save every N epochs
  keep_last: 3            # keep last N checkpoints

logging:
  tensorboard_dir: ./runs/hdmapnet
  log_interval: 50        # log every N iterations
  vis_interval: 500       # visualize predictions every N iterations
```

### Optimizer Details

Adam with cosine annealing and linear warmup:

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

# Warmup for 2 epochs, then cosine decay
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                         milestones=[warmup_steps])
```

### Input/Output Dimensions Summary

```
Input:  6 cameras x (3, 128, 352)   -> (B, 6, 3, 128, 352)
Output: BEV semantic  (B, 3, 200, 200)   - per-class logits
        BEV instance  (B, 16, 200, 200)  - embedding vectors
        BEV direction (B, 2, 200, 200)   - (cos, sin) per pixel
```

---

## 5. Training Pipeline Walkthrough

### DataLoader

```python
from torch.utils.data import DataLoader, DistributedSampler
from dataset import HDMapNetDataset

train_dataset = HDMapNetDataset(
    dataroot='./data/nuscenes',
    version='v1.0-trainval',
    split='train',
    input_size=(128, 352),
    bev_size=(200, 200),
    bev_resolution=0.3,
)

train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
```

Each sample from the DataLoader contains:

```python
{
    'images': Tensor(6, 3, 128, 352),      # 6 camera images, normalized
    'intrinsics': Tensor(6, 3, 3),          # camera intrinsic matrices
    'extrinsics': Tensor(6, 4, 4),          # camera-to-ego transforms
    'gt_semantic': Tensor(3, 200, 200),     # multi-label BEV semantic GT
    'gt_instance': Tensor(200, 200),        # integer instance IDs
    'gt_direction': Tensor(2, 200, 200),    # (cos, sin) direction GT
}
```

### Forward Pass

The model architecture follows these stages:

1. **Image Backbone**: EfficientNet-B4 extracts features from each camera independently
2. **View Transform** (Lift-Splat-Shoot style): Projects 2D features to 3D using predicted depth distributions, then collapses to BEV
3. **BEV Encoder**: Convolutional neck that refines the BEV feature map
4. **Task Heads**: Separate heads for semantic, instance, and direction predictions

```python
def training_step(batch, model, optimizer, scheduler, loss_weights):
    images = batch['images'].cuda()           # (B, 6, 3, H, W)
    intrinsics = batch['intrinsics'].cuda()   # (B, 6, 3, 3)
    extrinsics = batch['extrinsics'].cuda()   # (B, 6, 4, 4)
    gt_sem = batch['gt_semantic'].cuda()      # (B, 3, 200, 200)
    gt_inst = batch['gt_instance'].cuda()     # (B, 200, 200)
    gt_dir = batch['gt_direction'].cuda()     # (B, 2, 200, 200)

    # Forward
    pred_sem, pred_inst, pred_dir = model(images, intrinsics, extrinsics)

    # Losses
    l_sem = semantic_loss(pred_sem, gt_sem)
    l_pull, l_push, l_reg = discriminative_loss(pred_inst, gt_inst)
    l_inst = l_pull + l_push + 0.001 * l_reg
    sem_mask = (gt_sem.sum(dim=1, keepdim=True) > 0).float()
    l_dir = direction_loss(pred_dir, gt_dir, sem_mask)

    total_loss = (loss_weights['semantic'] * l_sem +
                  loss_weights['instance'] * l_inst +
                  loss_weights['direction'] * l_dir)

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    scheduler.step()

    return {
        'total_loss': total_loss.item(),
        'sem_loss': l_sem.item(),
        'inst_loss': l_inst.item(),
        'dir_loss': l_dir.item(),
    }
```

### Gradient Accumulation

For smaller GPU setups where you cannot fit batch_size=4 per GPU:

```python
accumulation_steps = 2  # effective batch = real_batch * accumulation_steps

for i, batch in enumerate(train_loader):
    loss = compute_loss(batch, model) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

---

## 6. Common Training Issues and Solutions

### 6.1 View Transform Instability

**Symptom**: Depth predictions collapse to a single bin early in training, BEV features are blank or noisy.

**Cause**: The depth distribution head produces uniform or degenerate predictions before the backbone learns meaningful features.

**Solution**: Use a warmup schedule for the view transform:

```python
# Option A: Freeze depth head for first 2 epochs
if epoch < 2:
    for param in model.view_transform.depth_head.parameters():
        param.requires_grad = False
else:
    for param in model.view_transform.depth_head.parameters():
        param.requires_grad = True

# Option B: Lower learning rate for depth head
param_groups = [
    {'params': model.backbone.parameters(), 'lr': 1e-3},
    {'params': model.view_transform.depth_head.parameters(), 'lr': 1e-4},
    {'params': model.bev_encoder.parameters(), 'lr': 1e-3},
    {'params': model.heads.parameters(), 'lr': 1e-3},
]
optimizer = optim.Adam(param_groups)
```

### 6.2 Class Imbalance

**Symptom**: Model predicts mostly background; thin structures (road boundaries) have very low IoU.

**Cause**: Positive pixels are less than 5% of the BEV grid for boundary classes.

**Solutions**:

```python
# Solution 1: Focal loss
def focal_bce_loss(pred, target, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()

# Solution 2: Per-class positive weight (precomputed from dataset)
pos_weight = torch.tensor([5.0, 4.0, 8.0]).cuda()  # neg/pos ratio per class
loss = F.binary_cross_entropy_with_logits(
    pred, target, pos_weight=pos_weight.view(1, 3, 1, 1)
)

# Solution 3: Online hard example mining (OHEM)
def ohem_bce_loss(pred, target, keep_ratio=0.7):
    loss_map = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    loss_flat = loss_map.view(-1)
    k = int(loss_flat.numel() * keep_ratio)
    topk_loss, _ = torch.topk(loss_flat, k)
    return topk_loss.mean()
```

### 6.3 Instance Embedding Not Separating

**Symptom**: All instance embeddings converge to similar values; post-processing clustering fails.

**Causes and Fixes**:

| Cause | Fix |
|-------|-----|
| Push margin too small | Increase `delta_d` from 3.0 to 4.0 or 5.0 |
| Embedding dim too low | Increase from 16 to 32 |
| Instance loss underweighted | Increase `w_inst` to 2.0 |
| Too few instances per image | Verify data loading; ensure instance IDs are correct |

**Debugging tip**: Log the mean inter-cluster distance and intra-cluster variance to TensorBoard. They should diverge over training.

### 6.4 GPU Out of Memory (OOM)

**Symptom**: CUDA OOM during forward or backward pass.

**Solutions** (in order of preference):

```python
# 1. Reduce input resolution
input_size: [96, 256]  # instead of [128, 352]

# 2. Fewer depth bins
depth_bins: 32  # instead of 64

# 3. Gradient checkpointing on the backbone
from torch.utils.checkpoint import checkpoint_sequential

class CheckpointedBackbone(nn.Module):
    def forward(self, x):
        # Checkpoint every 2 layers
        return checkpoint_sequential(self.layers, 2, x)

# 4. Mixed precision training (AMP)
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    pred_sem, pred_inst, pred_dir = model(images, intrinsics, extrinsics)
    loss = compute_total_loss(pred_sem, pred_inst, pred_dir, gt)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 5. Reduce BEV encoder channels
bev_encoder_channels: 128  # instead of 256
```

### 6.5 Training Loss Plateaus

**Symptom**: Loss stops decreasing after epoch 10-15 but validation IoU is still below target.

**Solutions**:
- Reduce learning rate by 10x manually and continue for 10 more epochs
- Add BEV rotation augmentation (random rotation of ground truth and predicted BEV)
- Increase model capacity (use EfficientNet-B5 or ResNet-101 backbone)
- Verify data augmentation is correctly applied to both images AND ground truth

---

## 7. Monitoring and Logging

### Key Metrics to Track

```python
# Semantic IoU per class (computed on validation set)
def compute_iou(pred_binary, gt_binary):
    intersection = (pred_binary & gt_binary).sum()
    union = (pred_binary | gt_binary).sum()
    return intersection / (union + 1e-6)

# Per-class evaluation
class_names = ['road_boundary', 'lane_divider', 'pedestrian_crossing']
thresholds = [0.5, 0.5, 0.5]  # sigmoid threshold for binarization

for cls_idx, cls_name in enumerate(class_names):
    pred_cls = torch.sigmoid(pred_sem[:, cls_idx]) > thresholds[cls_idx]
    gt_cls = gt_sem[:, cls_idx].bool()
    iou = compute_iou(pred_cls, gt_cls)
    writer.add_scalar(f'val/iou_{cls_name}', iou, global_step)
```

**Expected IoU ranges** (full training, 30 epochs):

| Class | IoU Range |
|-------|-----------|
| Road boundary | 38-45% |
| Lane divider | 42-50% |
| Pedestrian crossing | 25-35% |
| Mean | 38-43% |

### TensorBoard Logging Setup

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs/hdmapnet_exp01')

# During training, log every N iterations
if step % log_interval == 0:
    writer.add_scalar('train/total_loss', losses['total_loss'], step)
    writer.add_scalar('train/sem_loss', losses['sem_loss'], step)
    writer.add_scalar('train/inst_loss', losses['inst_loss'], step)
    writer.add_scalar('train/dir_loss', losses['dir_loss'], step)
    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], step)

# Visualize BEV predictions periodically
if step % vis_interval == 0:
    vis = visualize_bev_predictions(pred_sem, gt_sem)
    writer.add_image('train/bev_prediction', vis, step, dataformats='HWC')
```

### Visualization: BEV Predictions vs Ground Truth

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_bev_predictions(pred_sem, gt_sem, idx=0):
    """
    Create side-by-side visualization of prediction and ground truth.
    """
    pred = torch.sigmoid(pred_sem[idx]).cpu().detach().numpy()  # (3, 200, 200)
    gt = gt_sem[idx].cpu().numpy()  # (3, 200, 200)

    # Color coding: R=road boundary, G=lane divider, B=crossing
    pred_rgb = np.stack([pred[0], pred[1], pred[2]], axis=-1)  # (200, 200, 3)
    gt_rgb = np.stack([gt[0], gt[1], gt[2]], axis=-1)

    # Concatenate side by side
    combined = np.concatenate([gt_rgb, pred_rgb], axis=1)  # (200, 400, 3)
    combined = (combined * 255).astype(np.uint8)

    return combined
```

### Checkpointing Strategy

```python
def save_checkpoint(model, optimizer, scheduler, epoch, step, save_dir, keep_last=3):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.module.state_dict(),  # unwrap DDP
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    path = os.path.join(save_dir, f'checkpoint_epoch{epoch:02d}.pth')
    torch.save(checkpoint, path)

    # Clean up old checkpoints
    checkpoints = sorted(glob.glob(os.path.join(save_dir, 'checkpoint_*.pth')))
    while len(checkpoints) > keep_last:
        os.remove(checkpoints.pop(0))
```

Save checkpoints every 5 epochs and always keep the best validation checkpoint:

```python
if val_mean_iou > best_iou:
    best_iou = val_mean_iou
    torch.save(checkpoint, os.path.join(save_dir, 'best.pth'))
```

---

## 8. Multi-GPU Training

### DistributedDataParallel (DDP) Setup

HDMapNet should be trained with DDP for best performance. Avoid DataParallel (DP) as it has a single-process bottleneck.

**Launch script**:

```bash
# 4 GPUs on a single node
torchrun --nproc_per_node=4 \
    train.py \
    --config config/hdmapnet_efficientb4_200x200.yaml \
    --dist-backend nccl
```

**Training script setup**:

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup_distributed()

    # Build model
    model = HDMapNet(config).cuda(local_rank)

    # Sync BatchNorm - critical for BEV encoder
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # DataLoader with DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size_per_gpu'],
        sampler=train_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    # Training loop
    for epoch in range(config['training']['total_epochs']):
        train_sampler.set_epoch(epoch)  # shuffle differently each epoch
        train_one_epoch(model, train_loader, optimizer, scheduler, epoch)

        # Validate only on rank 0 (or all ranks with gathered metrics)
        if local_rank == 0:
            validate(model, val_loader, epoch)

    dist.destroy_process_group()
```

### Sync BatchNorm

**Why it matters**: The BEV encoder operates on the fused BEV feature map. With regular BatchNorm, each GPU normalizes using only its local batch statistics. Since BEV features can vary significantly between samples (different scenes, different map densities), SyncBatchNorm provides more stable normalization by computing statistics across all GPUs.

```python
# Convert all BatchNorm layers to SyncBatchNorm
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

# Alternative: only convert BEV encoder BatchNorms (less communication overhead)
def convert_bev_encoder_syncbn(model):
    for name, module in model.named_modules():
        if 'bev_encoder' in name and isinstance(module, nn.BatchNorm2d):
            parent = get_parent_module(model, name)
            setattr(parent, name.split('.')[-1],
                    nn.SyncBatchNorm.convert_sync_batchnorm(module))
```

### Learning Rate Scaling

When scaling from 1 GPU to N GPUs, the effective batch size increases by N. Apply linear learning rate scaling:

```python
# Base config: 1 GPU, batch_size=4, lr=1e-3
# Scaled config: 4 GPUs, batch_size=4 per GPU (16 total), lr=4e-3

num_gpus = torch.cuda.device_count()
base_lr = config['training']['learning_rate']
scaled_lr = base_lr * num_gpus

# With warmup to stabilize the larger learning rate
warmup_epochs = 2  # more warmup for larger effective batch
```

**Caution**: Linear scaling works up to a point. For 8+ GPUs, consider using square root scaling or simply keeping the learning rate at 2e-3 with more warmup.

### Multi-Node Training

For clusters with multiple nodes:

```bash
# Node 0 (master)
torchrun --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train.py --config config/hdmapnet_efficientb4_200x200.yaml

# Node 1
torchrun --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=10.0.0.1 \
    --master_port=29500 \
    train.py --config config/hdmapnet_efficientb4_200x200.yaml
```

---

## 9. Quick Reference: Training Commands

### Full Training (4x A100)

```bash
torchrun --nproc_per_node=4 train.py \
    --config config/hdmapnet_efficientb4_200x200.yaml \
    --epochs 30 \
    --batch-size 4 \
    --lr 4e-3 \
    --save-dir checkpoints/hdmapnet_full
```

### Development Run (mini dataset, 1 GPU)

```bash
python train.py \
    --config config/hdmapnet_efficientb4_200x200.yaml \
    --data-version v1.0-mini \
    --epochs 5 \
    --batch-size 2 \
    --lr 1e-3 \
    --no-distributed
```

### Resume from Checkpoint

```bash
torchrun --nproc_per_node=4 train.py \
    --config config/hdmapnet_efficientb4_200x200.yaml \
    --resume checkpoints/hdmapnet_full/checkpoint_epoch15.pth
```

### Evaluation Only

```bash
python evaluate.py \
    --config config/hdmapnet_efficientb4_200x200.yaml \
    --checkpoint checkpoints/hdmapnet_full/best.pth \
    --split val \
    --visualize
```

---

## 10. Tips and Best Practices

1. **Always validate on the full val set** every epoch, not just a subset. IoU on partial val sets is misleading due to scene-level variance.

2. **Gradient clipping is essential**. Without `clip_grad_norm_(max_norm=5.0)`, the discriminative loss can produce large gradients that destabilize training.

3. **Mixed precision (AMP) is safe** for this model. The only concern is the discriminative loss computation, which should remain in float32:

   ```python
   with torch.cuda.amp.autocast():
       pred_sem, pred_inst, pred_dir = model(images, intrinsics, extrinsics)
       l_sem = semantic_loss(pred_sem, gt_sem)
       l_dir = direction_loss(pred_dir, gt_dir, sem_mask)

   # Compute instance loss in full precision
   with torch.cuda.amp.autocast(enabled=False):
       l_inst = discriminative_loss(pred_inst.float(), gt_inst)
   ```

4. **Convergence timeline**: Expect the model to converge around epoch 24. Semantic IoU improves rapidly in epochs 1-10, then gradually to epoch 24. Instance quality improves mostly in epochs 5-20.

5. **Data loading bottleneck**: Image decoding and rasterization can be slow. Pre-compute and cache the BEV ground truth:

   ```bash
   python preprocess_gt.py --dataroot ./data/nuscenes --output ./data/nuscenes_gt_cache
   ```

6. **Reproducibility**: Set seeds for deterministic training:

   ```python
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   np.random.seed(42)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False  # slightly slower but reproducible
   ```

7. **BEV augmentation**: Random rotation of the BEV grid provides a significant boost (+1-2% mIoU) but requires rotating both the ground truth masks and the camera extrinsics. Implement carefully:

   ```python
   def rotate_bev_augmentation(gt_sem, gt_inst, gt_dir, extrinsics, angle_deg):
       angle_rad = np.radians(angle_deg)
       # Rotate GT maps using affine transform
       # Rotate extrinsics by composing a yaw rotation
       # Rotate direction vectors (cos, sin) by the same angle
       ...
   ```

8. **Do not freeze the backbone**. Unlike object detection where ImageNet pretraining is very useful, the view transform requires gradients to flow through the backbone to learn depth-aware features. Fine-tune from ImageNet weights but keep all layers trainable.
