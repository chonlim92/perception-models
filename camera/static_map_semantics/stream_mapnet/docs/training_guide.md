# StreamMapNet: Training Guide

## Environment Setup

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1x NVIDIA RTX 3090 (24 GB) | 8x NVIDIA A100 (40/80 GB) |
| CPU | 8 cores | 32+ cores |
| RAM | 32 GB | 128 GB |
| Storage | 200 GB SSD | 1 TB NVMe SSD |
| CUDA | 11.3+ | 11.7+ |
| Python | 3.8 | 3.8 |

### Python Environment Setup

```bash
# Create conda environment
conda create -n streammapnet python=3.8 -y
conda activate streammapnet

# Install PyTorch (CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install mmcv and mmdet3d ecosystem
pip install openmim
mim install mmcv-full==1.7.1
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0

# Install mmdet3d from source (required for BEV transform)
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.1.1
pip install -e .
cd ..

# Install additional dependencies
pip install nuscenes-devkit==1.1.11
pip install av2==0.2.1
pip install einops==0.6.1
pip install scipy==1.10.1
pip install shapely==2.0.1
pip install scikit-image==0.21.0
pip install tensorboard==2.13.0
```

### Full Requirements (requirements.txt)

```
torch>=1.9.0,<2.0.0
torchvision>=0.10.0,<0.15.0
mmcv-full>=1.5.0,<=1.7.1
mmdet>=2.25.0,<=2.28.2
mmsegmentation>=0.29.0,<=0.30.0
mmdet3d>=1.0.0,<=1.1.1
nuscenes-devkit>=1.1.9
av2>=0.2.0
numpy>=1.21.0,<1.24.0
scipy>=1.7.0
shapely>=1.8.0
einops>=0.4.0
scikit-image>=0.19.0
tensorboard>=2.10.0
pyquaternion>=0.9.9
timm>=0.6.0
```

### Verify Installation

```python
import torch
import mmcv
import mmdet
import mmdet3d

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"mmcv: {mmcv.__version__}")
print(f"mmdet: {mmdet.__version__}")
print(f"mmdet3d: {mmdet3d.__version__}")
```

---

## Training Procedure

### Data Preparation (Pre-requisite)

Ensure ground truth annotations are generated before training (see data_collection.md):

```bash
# Generate nuScenes GT
python tools/create_data.py nuscenes \
    --root-path /data/nuscenes \
    --out-dir /data/nuscenes/stream_mapnet_gt \
    --version v1.0-trainval

# Verify GT files exist
ls /data/nuscenes/stream_mapnet_gt/
# Expected: nuscenes_infos_train.pkl, nuscenes_infos_val.pkl
```

### Training Schedule

| Phase | Epochs | Learning Rate | Notes |
|-------|--------|---------------|-------|
| Warmup | 0-1 (500 iters) | 0 → 6e-4 (linear) | Linear warmup from 0 |
| Main training | 1-24 | 6e-4 | Cosine annealing |
| Cooldown | 20-24 | 6e-4 → 6e-6 | Final LR decay |

**Total epochs:** 24  
**Effective batch size:** 32 (4 GPUs x 8 samples per GPU) or equivalent  
**Samples per epoch:** ~28,000 (nuScenes trainval)

### Optimizer Configuration

```python
optimizer = dict(
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),           # Lower LR for pretrained backbone
            'temporal_fusion': dict(lr_mult=1.0),    # Full LR for temporal module
            'map_decoder': dict(lr_mult=1.0),        # Full LR for decoder
            'bev_encoder': dict(lr_mult=0.5),        # Moderate LR for BEV
        }
    )
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=0.01,  # Final LR = 6e-4 * 0.01 = 6e-6
)
```

### Single-GPU Training

```bash
python tools/train.py configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    --work-dir work_dirs/streammapnet_r50_24ep \
    --gpu-ids 0
```

### Multi-GPU Training (DDP)

```bash
# 4-GPU training with Distributed Data Parallel
bash tools/dist_train.sh \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    4 \
    --work-dir work_dirs/streammapnet_r50_4gpu

# 8-GPU training
bash tools/dist_train.sh \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    8 \
    --work-dir work_dirs/streammapnet_r50_8gpu
```

The `dist_train.sh` script internally calls:
```bash
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py $CONFIG \
    --launcher pytorch \
    "${@:3}"
```

### Multi-Node Training

```bash
# Node 0 (master)
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    tools/train.py configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    --launcher pytorch

# Node 1
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    tools/train.py configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    --launcher pytorch
```

---

## Loss Functions

### Overview

StreamMapNet uses a multi-component loss with Hungarian matching:

```
L_total = lambda_cls * L_cls + lambda_pts * L_pts + lambda_dir * L_dir
```

Default loss weights:
| Component | Weight (lambda) | Description |
|-----------|----------------|-------------|
| L_cls | 2.0 | Classification loss |
| L_pts | 5.0 | Point regression loss |
| L_dir | 0.005 | Direction-aware loss |

### 1. Hungarian Matching (Permutation-Invariant Assignment)

Before computing losses, predictions must be matched to ground truth elements. StreamMapNet uses the Hungarian algorithm for optimal bipartite matching:

```python
from scipy.optimize import linear_sum_assignment

def hungarian_matching(pred_logits, pred_points, gt_labels, gt_points):
    """
    Find optimal assignment between predictions and ground truth.
    
    Args:
        pred_logits: (N_queries, num_classes) classification scores
        pred_points: (N_queries, K, 2) predicted polyline points
        gt_labels: (N_gt,) ground truth class labels
        gt_points: (N_gt, K, 2) ground truth polyline points
    
    Returns:
        matched_indices: list of (pred_idx, gt_idx) pairs
    """
    N_pred = pred_logits.shape[0]
    N_gt = gt_labels.shape[0]
    
    # Compute cost matrix (N_pred x N_gt)
    # Classification cost
    cls_cost = -pred_logits[:, gt_labels]  # (N_pred, N_gt)
    
    # L1 point cost
    pts_cost = torch.cdist(
        pred_points.flatten(1),  # (N_pred, K*2)
        gt_points.flatten(1),    # (N_gt, K*2)
        p=1
    ) / (K * 2)  # Normalize by number of coordinates
    
    # Combined cost
    cost_matrix = 2.0 * cls_cost + 5.0 * pts_cost
    
    # Solve assignment (scipy on CPU)
    cost_np = cost_matrix.detach().cpu().numpy()
    pred_indices, gt_indices = linear_sum_assignment(cost_np)
    
    return list(zip(pred_indices, gt_indices))
```

### 2. Classification Loss (Focal Loss)

Focal loss addresses class imbalance between matched (positive) and unmatched (negative/background) queries:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, pred_logits, targets, matched_indices):
        """
        Args:
            pred_logits: (B, N_queries, num_classes+1) including background
            targets: (B, N_gt) class labels
            matched_indices: list of (pred_idx, gt_idx) per batch
        """
        # Create target tensor (background class for unmatched queries)
        target_classes = torch.full(
            (pred_logits.shape[0], pred_logits.shape[1]),
            self.num_classes,  # Background class index
            device=pred_logits.device
        )
        
        # Assign matched GT classes
        for b, (pred_idx, gt_idx) in enumerate(matched_indices):
            target_classes[b, pred_idx] = targets[b, gt_idx]
        
        # Compute focal loss
        pred_probs = pred_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, 
            F.one_hot(target_classes, self.num_classes + 1).float(),
            reduction='none'
        )
        p_t = pred_probs * target_classes + (1 - pred_probs) * (1 - target_classes)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        return (focal_weight * ce_loss).mean()
```

### 3. Point-to-Point L1 Loss

For matched query-GT pairs, compute the L1 distance between predicted and GT points:

```python
def point_l1_loss(pred_points, gt_points, matched_indices):
    """
    Args:
        pred_points: (B, N_queries, K, 2) predicted points in [0, 1]
        gt_points: (B, N_gt, K, 2) ground truth points in [0, 1]
        matched_indices: assignment from Hungarian matching
    
    Returns:
        loss: scalar L1 loss over matched pairs
    """
    total_loss = 0
    num_matched = 0
    
    for b, (pred_idx, gt_idx) in enumerate(matched_indices):
        if len(pred_idx) == 0:
            continue
        
        pred_matched = pred_points[b, pred_idx]  # (M, K, 2)
        gt_matched = gt_points[b, gt_idx]        # (M, K, 2)
        
        # L1 loss per point
        loss = F.l1_loss(pred_matched, gt_matched, reduction='sum')
        total_loss += loss
        num_matched += len(pred_idx) * K * 2
    
    return total_loss / max(num_matched, 1)
```

### 4. Direction-Aware Loss

Accounts for the ambiguity in polyline direction (a polyline can be traversed in either direction):

```python
def direction_aware_point_loss(pred_points, gt_points, matched_indices):
    """
    Compute L1 loss considering both forward and reverse directions.
    For each matched pair, use the direction that gives lower loss.
    """
    total_loss = 0
    num_matched = 0
    
    for b, (pred_idx, gt_idx) in enumerate(matched_indices):
        if len(pred_idx) == 0:
            continue
        
        pred_matched = pred_points[b, pred_idx]  # (M, K, 2)
        gt_matched = gt_points[b, gt_idx]        # (M, K, 2)
        
        # Forward direction loss
        loss_forward = F.l1_loss(
            pred_matched, gt_matched, reduction='none'
        ).sum(dim=(-1, -2))  # (M,)
        
        # Reverse direction loss
        gt_reversed = gt_matched.flip(dims=[1])  # Reverse point order
        loss_reverse = F.l1_loss(
            pred_matched, gt_reversed, reduction='none'
        ).sum(dim=(-1, -2))  # (M,)
        
        # Take minimum per element
        loss = torch.min(loss_forward, loss_reverse).sum()
        total_loss += loss
        num_matched += len(pred_idx)
    
    return total_loss / max(num_matched, 1)
```

### 5. Deep Supervision (Auxiliary Losses)

Losses are applied at each decoder layer (not just the final one):

```python
def compute_total_loss(outputs, gt_labels, gt_points):
    """Apply loss at each decoder layer with decreasing weight."""
    total_loss = 0
    
    # Final layer loss (full weight)
    matched = hungarian_matching(outputs['class_logits'], outputs['pred_points'],
                                  gt_labels, gt_points)
    total_loss += compute_layer_loss(outputs, gt_labels, gt_points, matched)
    
    # Auxiliary losses (intermediate layers)
    for aux_idx, aux_output in enumerate(outputs['aux_outputs']):
        aux_matched = hungarian_matching(aux_output['class_logits'], 
                                          aux_output['pred_points'],
                                          gt_labels, gt_points)
        aux_loss = compute_layer_loss(aux_output, gt_labels, gt_points, aux_matched)
        total_loss += aux_loss  # Same weight as final layer (MapTR convention)
    
    return total_loss
```

---

## Temporal Sequence Training

### Sequence Batching Strategy

StreamMapNet requires sequential frames for temporal propagation. During training:

1. **Sequence construction:** Each training sample is a sequence of T consecutive frames from the same scene.
2. **Hidden state propagation:** The hidden state is propagated through the sequence during forward pass.
3. **Gradient flow:** Gradients flow through the temporal fusion across the sequence (truncated BPTT).

```python
# Training configuration for temporal sequences
data_config = dict(
    sequence_length=8,          # Number of consecutive frames per sequence
    stride=1,                   # Frame stride (1 = every keyframe at 2 Hz)
    shuffle_sequences=True,     # Shuffle sequence order (not frame order within)
    reset_hidden_at_scene_boundary=True,  # Reset when scene changes
)
```

### Sequence DataLoader

```python
class TemporalSequenceDataset:
    """Dataset that returns sequences of consecutive frames."""
    
    def __init__(self, data_root, ann_file, sequence_length=8):
        self.sequence_length = sequence_length
        self.sequences = self._build_sequences()  # List of frame index lists
    
    def _build_sequences(self):
        """Group frames into temporal sequences within scenes."""
        sequences = []
        for scene_frames in self.frames_by_scene:
            # Slide window over scene frames
            for start_idx in range(0, len(scene_frames) - self.sequence_length + 1):
                seq = scene_frames[start_idx:start_idx + self.sequence_length]
                sequences.append(seq)
        return sequences
    
    def __getitem__(self, idx):
        """Return a full sequence of T frames."""
        frame_indices = self.sequences[idx]
        
        sequence_data = []
        for frame_idx in frame_indices:
            data = {
                'images': self._load_images(frame_idx),         # (6, 3, H, W)
                'ego_pose': self._get_ego_pose(frame_idx),      # (4, 4)
                'gt_vectors': self._get_gt_vectors(frame_idx),  # (N, K, 2)
                'gt_labels': self._get_gt_labels(frame_idx),    # (N,)
                'ego_motion': self._get_relative_pose(frame_idx),  # (4, 4) to prev
            }
            sequence_data.append(data)
        
        return sequence_data
```

### Training Loop with Temporal Propagation

```python
def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    
    for batch_sequences in dataloader:
        optimizer.zero_grad()
        
        total_loss = 0
        hidden_state = None  # Reset at sequence start
        
        # Process each frame in the sequence
        for t, frame_data in enumerate(batch_sequences):
            images = frame_data['images'].cuda()           # (B, 6, 3, H, W)
            ego_motion = frame_data['ego_motion'].cuda()   # (B, 4, 4)
            gt_vectors = frame_data['gt_vectors'].cuda()   # (B, N_gt, K, 2)
            gt_labels = frame_data['gt_labels'].cuda()     # (B, N_gt)
            
            # Forward pass with temporal state
            outputs, hidden_state = model(
                images, 
                ego_motion=ego_motion,
                hidden_state=hidden_state,
            )
            
            # Compute loss for this frame
            frame_loss = compute_total_loss(outputs, gt_labels, gt_vectors)
            total_loss += frame_loss
            
            # Detach hidden state periodically to limit memory
            if t % 4 == 3:  # Every 4 frames
                hidden_state = hidden_state.detach()
        
        # Average loss over sequence
        total_loss = total_loss / len(batch_sequences)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
        
        optimizer.step()
```

---

## Mixed Precision Training

### Configuration

```python
# Enable mixed precision with PyTorch AMP
fp16_config = dict(
    enabled=True,
    loss_scale='dynamic',          # Dynamic loss scaling
    initial_loss_scale=2**16,      # Starting scale factor
    growth_interval=2000,          # Increase scale every N steps
)
```

### Implementation

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs, hidden_state = model(images, ego_motion, hidden_state)
        loss = compute_total_loss(outputs, gt_labels, gt_vectors)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
    scaler.step(optimizer)
    scaler.update()
```

**Memory savings with FP16:**
| Setting | GPU Memory (B=8, 1 GPU) | Training Time (24 ep) |
|---------|------------------------|----------------------|
| FP32 | ~22 GB | ~48 hours (8x A100) |
| FP16 (AMP) | ~14 GB | ~32 hours (8x A100) |

---

## Training Configuration File

### Complete Config Example

```python
# configs/streammapnet/streammapnet_r50_24ep_nuscenes.py

_base_ = ['../_base_/default_runtime.py']

# Model
model = dict(
    type='StreamMapNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),  # C3, C4, C5
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3,
    ),
    bev_constructor=dict(
        type='LSSTransform',
        in_channels=256,
        out_channels=64,
        image_size=[256, 704],
        feature_size=[16, 44],
        xbound=[-30.0, 30.0, 0.3],
        ybound=[-15.0, 15.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[2.0, 50.0, 1.0],
    ),
    bev_encoder=dict(
        type='BEVEncoder',
        in_channels=64,
        out_channels=256,
        num_layers=4,
    ),
    temporal_fusion=dict(
        type='TemporalCrossAttention',
        embed_dim=256,
        num_heads=8,
        num_layers=1,
        use_gate=True,
    ),
    map_decoder=dict(
        type='MapTransformerDecoder',
        num_layers=6,
        num_queries=50,
        embed_dim=256,
        num_heads=8,
        num_points=20,
        num_classes=3,
        ffn_dim=1024,
        dropout=0.1,
        deformable_points=4,
    ),
    loss=dict(
        cls_weight=2.0,
        pts_weight=5.0,
        dir_weight=0.005,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ),
)

# Data
dataset_type = 'NuScenesMapDataset'
data_root = '/data/nuscenes/'
input_size = (256, 704)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='stream_mapnet_gt/nuscenes_infos_train.pkl',
        sequence_length=8,
        pipeline=[
            dict(type='LoadMultiViewImages'),
            dict(type='ResizeMultiViewImages', size=input_size),
            dict(type='NormalizeMultiViewImages',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]),
            dict(type='LoadMapAnnotations'),
            dict(type='RandomRotateMap', angle=(-22.5, 22.5)),
            dict(type='RandomScaleMap', scale=(0.95, 1.05)),
        ],
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='stream_mapnet_gt/nuscenes_infos_val.pkl',
        sequence_length=1,  # Single frame for validation
        pipeline=[
            dict(type='LoadMultiViewImages'),
            dict(type='ResizeMultiViewImages', size=input_size),
            dict(type='NormalizeMultiViewImages',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]),
            dict(type='LoadMapAnnotations'),
        ],
    ),
)

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=6e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)}
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=0.01,
)

# Runtime
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=4, metric='chamfer')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)
```

---

## Training Tips and Best Practices

### Gradient Accumulation (for limited GPU memory)

```bash
# Effective batch size 32 with 1 GPU (accumulate 4 steps x batch 8)
python tools/train.py configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    --cfg-options optimizer_config.cumulative_iters=4
```

### Resume Training

```bash
# Resume from checkpoint
python tools/train.py configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    --resume-from work_dirs/streammapnet_r50/epoch_12.pth
```

### Freeze Backbone Initially

For faster initial convergence, freeze backbone for first few epochs:

```python
# In config
custom_hooks = [
    dict(type='FreezeHook', module='backbone', freeze_epochs=2),
]
```

### Common Training Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| NaN loss | Loss becomes NaN after few iterations | Reduce LR, check data loading, enable gradient clipping |
| OOM | CUDA out of memory | Reduce batch size, enable FP16, reduce sequence length |
| Slow convergence | mAP stuck below 40 after 10 epochs | Check backbone pretrained weights are loaded, verify data augmentation |
| Temporal instability | Loss spikes at sequence boundaries | Ensure hidden state reset at scene boundaries |
| Poor recall | Many GT elements unmatched | Increase num_queries (50 → 100) |

### Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir work_dirs/streammapnet_r50/tf_logs --port 6006
```

Key metrics to monitor:
- `loss_cls`: Should decrease steadily (target < 0.5)
- `loss_pts`: Should decrease (target < 0.01)
- `loss_dir`: Should decrease (target < 0.005)
- `learning_rate`: Verify warmup and cosine schedule
- `grad_norm`: Should be < 35 (clipping threshold)

### Expected Training Timeline

| GPUs | Batch Size | Epochs | Wall Time | mAP (val) |
|------|-----------|--------|-----------|-----------|
| 1x A100 | 8 | 24 | ~4 days | 54.1 |
| 4x A100 | 32 | 24 | ~24 hours | 54.1 |
| 8x A100 | 64 | 24 | ~14 hours | 53.8* |

*Large batch sizes may require LR scaling: `lr = base_lr * (batch_size / 32)`

---

## Ablation Training Configs

### Without Temporal Fusion (single-frame baseline)

```python
# Disable temporal module
model = dict(
    temporal_fusion=None,  # No temporal fusion
)
data = dict(
    train=dict(sequence_length=1),  # No sequences needed
)
```

### With Different Temporal Lengths

```python
# Short temporal context
data = dict(train=dict(sequence_length=4))

# Long temporal context
data = dict(train=dict(sequence_length=16))
```

### Backbone Comparison

```bash
# ResNet-18 (lighter)
python tools/train.py configs/streammapnet/streammapnet_r18_24ep_nuscenes.py

# ResNet-101 (heavier)
python tools/train.py configs/streammapnet/streammapnet_r101_24ep_nuscenes.py

# Swin-Tiny
python tools/train.py configs/streammapnet/streammapnet_swin_t_24ep_nuscenes.py
```
