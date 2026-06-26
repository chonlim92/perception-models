# StreamMapNet: Training Guide

## Prerequisites

Before starting, ensure you have:
1. A working GPU environment (NVIDIA GPU with CUDA support)
2. The nuScenes dataset downloaded (see [data_collection.md](data_collection.md))
3. Ground truth annotations generated (see Data Preparation below)

---

## Environment Setup

### System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | 1x RTX 3090 (24 GB) | 4-8x A100 (40/80 GB) | VRAM limits batch size |
| CPU | 8 cores | 32+ cores | Data loading bottleneck |
| RAM | 32 GB | 128 GB | Map API loads full JSON into memory |
| Storage | 200 GB SSD | 1 TB NVMe SSD | Random access during training |
| CUDA | 11.3+ | 11.7+ | Must match PyTorch build |
| Python | 3.8 | 3.8 | mmdet3d compatibility |

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

# Install mmdet3d from source (required for BEV transform utilities)
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d && git checkout v1.1.1
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
pip install pyquaternion>=0.9.9
pip install timm>=0.6.0
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
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"mmcv: {mmcv.__version__}")
print(f"mmdet: {mmdet.__version__}")
print(f"mmdet3d: {mmdet3d.__version__}")
```

---

## Data Preparation

### Step 1: Generate Ground Truth Annotations

StreamMapNet requires pre-computed vectorized map ground truth. This step converts nuScenes map expansion data into per-frame pickle files:

```bash
# Generate nuScenes GT (takes ~30 minutes for trainval)
python tools/create_data.py nuscenes \
    --root-path /data/nuscenes \
    --out-dir /data/nuscenes/stream_mapnet_gt \
    --version v1.0-trainval

# Verify output
ls /data/nuscenes/stream_mapnet_gt/
# Expected: nuscenes_infos_train.pkl, nuscenes_infos_val.pkl
```

### Step 2: Generate Temporal Sequence Index

The sequence index defines which frames form valid temporal sequences for training:

```bash
python tools/create_sequence_index.py \
    --root-path /data/nuscenes \
    --version v1.0-trainval \
    --sequence-length 8 \
    --output /data/nuscenes/stream_mapnet_gt/sequence_index.pkl
```

### What the GT Contains

Each pickle file stores per-sample data:
```python
{
    'sample_token': str,           # Unique frame identifier
    'ego_translation': (3,),       # Vehicle position in global frame
    'ego_rotation': (4,),          # Quaternion orientation
    'lane_dividers': [(20, 2), ...],       # List of polylines in ego frame
    'road_boundaries': [(20, 2), ...],     # Each is K=20 points x (x, y)
    'pedestrian_crossings': [(20, 2), ...],
}
```

---

## Configuration File Explained

The configuration file controls every aspect of training. Here is a complete annotated example:

### Model Configuration

```python
model = dict(
    type='StreamMapNet',
    
    # --- Image Backbone ---
    backbone=dict(
        type='ResNet',
        depth=50,                    # ResNet-50 (alternatives: 18, 101)
        num_stages=4,                # Use all 4 ResNet stages
        out_indices=(1, 2, 3),       # Extract C3, C4, C5 features
        frozen_stages=1,             # Freeze stem + layer1 (saves memory, stable training)
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,              # Keep BN in eval mode even during training
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    
    # --- Feature Pyramid Neck ---
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],  # From C3, C4, C5
        out_channels=256,                # Unified channel dimension
        num_outs=3,                      # 3 output levels
    ),
    
    # --- BEV Transform (Lift-Splat-Shoot) ---
    bev_constructor=dict(
        type='LSSTransform',
        in_channels=256,             # From FPN
        out_channels=64,             # BEV channels before encoder
        image_size=[256, 704],       # Input image size (H, W)
        feature_size=[16, 44],       # Feature map size after backbone
        xbound=[-30.0, 30.0, 0.3],  # [min, max, resolution] in meters
        ybound=[-15.0, 15.0, 0.3],  # [min, max, resolution] in meters
        zbound=[-10.0, 10.0, 20.0], # Vertical range (single bin)
        dbound=[2.0, 50.0, 1.0],    # Depth [min, max, bin_size]
        # Results in: 59 depth bins, BEV grid 200x100
    ),
    
    # --- BEV Encoder ---
    bev_encoder=dict(
        type='BEVEncoder',
        in_channels=64,              # From LSS
        out_channels=256,            # Final BEV feature dimension
        num_layers=4,                # Convolutional refinement layers
    ),
    
    # --- Temporal Fusion ---
    temporal_fusion=dict(
        type='TemporalCrossAttention',
        embed_dim=256,               # Must match BEV encoder output
        num_heads=8,                 # Multi-head attention
        num_layers=1,                # Single fusion layer (lightweight)
        use_gate=True,               # Learnable gate for blending
        # Setting this to None disables temporal fusion (single-frame baseline)
    ),
    
    # --- Map Decoder ---
    map_decoder=dict(
        type='MapTransformerDecoder',
        num_layers=6,                # 6 decoder layers (iterative refinement)
        num_queries=150,             # Max detectable map elements per frame
        embed_dim=256,               # Query/key/value dimension
        num_heads=8,                 # Attention heads
        num_points=20,               # Points per polyline (K)
        num_classes=3,               # lane_div, road_bound, ped_cross
        ffn_dim=1024,                # Feed-forward hidden dim
        dropout=0.1,                 # Regularization
        deformable_points=4,         # Sampling points for deformable attention
    ),
    
    # --- Loss Configuration ---
    loss=dict(
        cls_weight=2.0,              # Classification loss weight
        pts_weight=5.0,              # Point regression loss weight
        dir_weight=0.005,            # Direction-aware loss weight
        focal_alpha=0.25,            # Focal loss alpha (class balance)
        focal_gamma=2.0,             # Focal loss gamma (hard example focus)
    ),
)
```

### Data Configuration

```python
data = dict(
    samples_per_gpu=8,               # Batch size per GPU
    workers_per_gpu=4,               # DataLoader workers per GPU
    train=dict(
        type='NuScenesMapDataset',
        data_root='/data/nuscenes/',
        ann_file='stream_mapnet_gt/nuscenes_infos_train.pkl',
        sequence_length=8,           # Frames per temporal sequence
        pipeline=[
            dict(type='LoadMultiViewImages'),
            dict(type='ResizeMultiViewImages', size=(256, 704)),
            dict(type='NormalizeMultiViewImages',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]),
            dict(type='LoadMapAnnotations'),
            dict(type='RandomRotateMap', angle=(-22.5, 22.5)),
            dict(type='RandomScaleMap', scale=(0.95, 1.05)),
        ],
    ),
    val=dict(
        type='NuScenesMapDataset',
        data_root='/data/nuscenes/',
        ann_file='stream_mapnet_gt/nuscenes_infos_val.pkl',
        sequence_length=1,           # Single frame for validation
        pipeline=[
            dict(type='LoadMultiViewImages'),
            dict(type='ResizeMultiViewImages', size=(256, 704)),
            dict(type='NormalizeMultiViewImages',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375]),
            dict(type='LoadMapAnnotations'),
        ],
    ),
)
```

---

## Loss Functions Explained

### Why We Need Hungarian Matching

StreamMapNet predicts a SET of map elements (150 queries), and the ground truth is also a SET (typically 10-30 elements per frame). The challenge: predictions have no natural ordering -- which prediction should be compared to which GT element?

**Hungarian matching** finds the optimal one-to-one assignment that minimizes the total matching cost:

```
Example: 150 predictions, 15 GT elements

Step 1: Compute cost matrix (150 x 15)
  - For each (prediction, GT) pair, cost = cls_cost + point_cost

Step 2: Solve assignment using Hungarian algorithm (scipy.optimize.linear_sum_assignment)
  - Result: 15 matched pairs (prediction_i <-> GT_j)
  - Remaining 135 predictions are "background" (no-object class)

Step 3: Compute loss only on matched pairs
  - Matched pairs: classification loss + point loss + direction loss
  - Unmatched predictions: classification loss toward "background" class
```

```python
# Cost matrix construction
def compute_matching_cost(pred_logits, pred_points, gt_labels, gt_points):
    """
    pred_logits: (N_queries, num_classes) = (150, 4)
    pred_points: (N_queries, K, 2) = (150, 20, 2)
    gt_labels: (N_gt,) = (15,)
    gt_points: (N_gt, K, 2) = (15, 20, 2)
    """
    # Classification cost: negative probability of correct class
    probs = pred_logits.softmax(dim=-1)         # (150, 4)
    cls_cost = -probs[:, gt_labels]             # (150, 15)
    
    # Point L1 cost: average L1 distance between point sets
    pts_cost = torch.cdist(
        pred_points.flatten(1),                  # (150, 40)
        gt_points.flatten(1),                    # (15, 40)
        p=1
    ) / (20 * 2)  # Normalize by number of coordinates
    
    # Total cost
    cost = 2.0 * cls_cost + 5.0 * pts_cost      # (150, 15)
    
    # Solve with Hungarian algorithm
    pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
    return pred_idx, gt_idx
```

### Classification Loss (Focal Loss)

Most queries (135/150 in our example) predict "background." Standard cross-entropy would be dominated by these easy negatives. Focal loss down-weights easy examples:

```
FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

Where:
  p_t = model's predicted probability for the correct class
  alpha = 0.25 (balances positive/negative)
  gamma = 2.0 (focuses on hard examples)

When p_t is high (easy, correct prediction): (1 - p_t)^gamma is small -> low loss
When p_t is low (hard, wrong prediction): (1 - p_t)^gamma is large -> high loss
```

### Point-to-Point L1 Loss

For each matched (prediction, GT) pair, compute L1 distance between corresponding points:

```python
# pred_matched: (M, 20, 2) matched predictions in [0, 1]
# gt_matched: (M, 20, 2) corresponding GT in [0, 1]

loss_pts = F.l1_loss(pred_matched, gt_matched, reduction='mean')
```

**Why L1 instead of L2?** L1 is more robust to outlier points. If one point is far off, L2 squares the error (dominates the loss), while L1 treats it linearly.

### Direction-Aware Loss

A road boundary from point A to point B is the same element as from B to A. Without direction handling, a correctly-detected polyline with reversed point order would be penalized:

```python
def direction_aware_loss(pred_pts, gt_pts):
    """Compute L1 loss for both directions, take minimum."""
    # Forward direction
    loss_forward = F.l1_loss(pred_pts, gt_pts, reduction='none').sum(dim=(-1, -2))
    
    # Reversed direction
    gt_reversed = gt_pts.flip(dims=[1])  # Reverse point order
    loss_reverse = F.l1_loss(pred_pts, gt_reversed, reduction='none').sum(dim=(-1, -2))
    
    # Take minimum per element
    loss = torch.min(loss_forward, loss_reverse)
    return loss.mean()
```

### Deep Supervision

Losses are computed at EVERY decoder layer, not just the final one:

```python
total_loss = 0

# Final layer loss (full weight)
final_matched = hungarian_match(outputs['pred_logits'], outputs['pred_points'], gt)
total_loss += compute_loss(outputs, gt, final_matched)

# Auxiliary losses from intermediate layers
for aux_output in outputs['aux_outputs']:
    aux_matched = hungarian_match(aux_output['pred_logits'], aux_output['pred_points'], gt)
    total_loss += compute_loss(aux_output, gt, aux_matched)  # Same weight
```

**Why?** Without deep supervision, gradients must flow through all 6 decoder layers to reach early layers. Deep supervision provides direct gradient signal to each layer, enabling faster convergence. Each layer independently learns to improve predictions.

---

## Temporal Sequence Training

### How Batching Works with History

Unlike standard image classification where each sample is independent, StreamMapNet processes sequences of consecutive frames. The hidden state propagates through the sequence:

```
Training Batch = 1 sequence of T=8 frames from the SAME scene

Frame 0: images_0 -> backbone -> BEV_0 -> temporal(BEV_0, None) -> H_0 -> decoder -> loss_0
Frame 1: images_1 -> backbone -> BEV_1 -> temporal(BEV_1, H_0) -> H_1 -> decoder -> loss_1
Frame 2: images_2 -> backbone -> BEV_2 -> temporal(BEV_2, H_1) -> H_2 -> decoder -> loss_2
...
Frame 7: images_7 -> backbone -> BEV_7 -> temporal(BEV_7, H_6) -> H_7 -> decoder -> loss_7

Total loss = (loss_0 + loss_1 + ... + loss_7) / 8
```

### Sequence Construction

Sequences are built from consecutive keyframes within a scene:

```python
class TemporalSequenceDataset:
    def _build_sequences(self):
        """Slide a window over each scene's frames."""
        sequences = []
        for scene in self.scenes:
            frames = scene.keyframes  # Ordered by time (2 Hz)
            # Sliding window with stride 1
            for start in range(len(frames) - self.seq_length + 1):
                seq = frames[start : start + self.seq_length]
                sequences.append(seq)
        return sequences
        # Example: scene with 40 frames, seq_length=8
        # -> 33 possible sequences: [0:8], [1:9], ..., [32:40]
```

### Hidden State Management During Training

```python
def train_one_epoch(model, dataloader, optimizer):
    for batch_sequence in dataloader:
        optimizer.zero_grad()
        total_loss = 0
        hidden_state = None  # Reset at start of each sequence
        
        for t, frame_data in enumerate(batch_sequence):
            # Forward pass with temporal state
            outputs, hidden_state = model(
                frame_data['images'],
                frame_data['ego_motion'],
                hidden_state=hidden_state,
            )
            
            # Compute loss for this frame
            frame_loss = compute_total_loss(outputs, frame_data['gt'])
            total_loss += frame_loss
            
            # CRITICAL: Detach hidden state every 4 frames
            # This limits memory usage by stopping gradient flow
            if t % 4 == 3:
                hidden_state = hidden_state.detach()
        
        # Average loss over sequence length
        loss = total_loss / len(batch_sequence)
        loss.backward()
        
        # Gradient clipping (important for temporal models)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
        optimizer.step()
```

### Why Detach Every 4 Frames?

Without detachment, gradients would flow through ALL temporal connections in the sequence. For 8 frames, this means storing 8 copies of the BEV features for backpropagation -- quickly exhausting GPU memory.

Detaching every 4 frames means:
- Gradients flow through at most 4 temporal steps
- Memory is bounded (4x BEV features for backprop)
- The model still learns temporal fusion (just with shorter gradient paths)

### Scene Boundary Handling

When a new scene begins (different location, different time), the hidden state must be reset:

```python
# During training
data_config = dict(
    reset_hidden_at_scene_boundary=True,  # Automatic reset
)

# During inference
model.reset_temporal_state()  # Call when starting a new scene
```

### Drop History Augmentation

To make the model robust to missing history (first frame of a scene, sensor dropout), randomly drop the hidden state during training:

```python
# 20% of the time, pretend there is no history
if random.random() < 0.2:
    hidden_state = None  # Force single-frame mode
```

This prevents the model from becoming overly dependent on temporal information.

---

## Training Schedule

### Optimizer Configuration

```python
optimizer = dict(
    type='AdamW',
    lr=6e-4,                         # Base learning rate
    weight_decay=0.01,               # L2 regularization
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),       # 10x lower for pretrained backbone
            'temporal_fusion': dict(lr_mult=1.0), # Full LR for new modules
            'map_decoder': dict(lr_mult=1.0),
            'bev_encoder': dict(lr_mult=0.5),    # Moderate for BEV
        }
    )
)
```

**Why different learning rates?** The backbone is pretrained on ImageNet -- its features are already good. Large updates would destroy useful pretrained representations. New modules (temporal, decoder) need to be trained from scratch, so they use full learning rate.

### Learning Rate Schedule

```
Phase         | Epochs    | Learning Rate          | Notes
--------------+-----------+------------------------+---------------------------
Warmup        | 0-1       | 0 -> 6e-4 (linear)    | Ramp up over 500 iters
Main training | 1-20      | 6e-4 (cosine decay)   | Gradual annealing
Cooldown      | 20-24     | ~6e-5 -> 6e-6         | Fine-tuning at low LR
```

### Training Commands

```bash
# Single-GPU training
python tools/train.py configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    --work-dir work_dirs/streammapnet_r50_24ep

# 4-GPU Distributed Data Parallel
bash tools/dist_train.sh \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py 4 \
    --work-dir work_dirs/streammapnet_r50_4gpu

# 8-GPU training
bash tools/dist_train.sh \
    configs/streammapnet/streammapnet_r50_24ep_nuscenes.py 8 \
    --work-dir work_dirs/streammapnet_r50_8gpu
```

### Multi-Node Training

```bash
# Node 0 (master)
python -m torch.distributed.launch \
    --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr="192.168.1.100" --master_port=29500 \
    tools/train.py configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    --launcher pytorch

# Node 1
python -m torch.distributed.launch \
    --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr="192.168.1.100" --master_port=29500 \
    tools/train.py configs/streammapnet/streammapnet_r50_24ep_nuscenes.py \
    --launcher pytorch
```

---

## Mixed Precision Training

### Setup

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # FP16 for forward pass
        outputs, hidden_state = model(images, ego_motion, hidden_state)
        loss = compute_total_loss(outputs, gt)
    
    scaler.scale(loss).backward()       # Scale loss to prevent underflow
    scaler.unscale_(optimizer)           # Unscale for gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
    scaler.step(optimizer)
    scaler.update()                      # Adjust scale factor
```

### Memory Savings

| Setting | GPU Memory (B=8, 1 GPU) | Training Time (24 ep, 8x A100) |
|---------|------------------------|---------------------------------|
| FP32 | ~22 GB | ~48 hours |
| FP16 (AMP) | ~14 GB | ~32 hours |

Mixed precision nearly halves memory usage and reduces training time by ~33%.

---

## Training Tips and Best Practices

### Backbone Freezing Strategy

For faster initial convergence, freeze the backbone for the first 2 epochs:

```python
custom_hooks = [
    dict(type='FreezeHook', module='backbone', freeze_epochs=2),
]
```

This lets the randomly-initialized decoder and temporal modules stabilize before the backbone features start changing.

### Gradient Accumulation (Limited GPU Memory)

If you cannot fit batch size 8 per GPU:

```bash
# Effective batch 32 with 1 GPU: accumulate 4 steps x batch 8
python tools/train.py config.py \
    --cfg-options optimizer_config.cumulative_iters=4 data.samples_per_gpu=8
```

### Resume from Checkpoint

```bash
python tools/train.py config.py --resume-from work_dirs/checkpoint/epoch_12.pth
```

### Learning Rate Scaling

When increasing total batch size, scale the learning rate linearly:

```
lr = base_lr * (total_batch_size / 32)

Example: 8 GPUs x 8 samples = 64 total batch
         lr = 6e-4 * (64/32) = 1.2e-3
```

---

## Monitoring Training

### Launch TensorBoard

```bash
tensorboard --logdir work_dirs/streammapnet_r50/tf_logs --port 6006
```

### Key Metrics to Watch

| Metric | Expected Trend | Healthy Range | Alarm |
|--------|----------------|---------------|-------|
| loss_cls | Steady decrease | < 0.5 by epoch 10 | Stuck > 1.0 |
| loss_pts | Steady decrease | < 0.01 by epoch 15 | Stuck > 0.05 |
| loss_dir | Slow decrease | < 0.005 | N/A |
| learning_rate | Warmup then cosine | Verify schedule shape | Flat = bug |
| grad_norm | < 35 (clip threshold) | 1-30 typical | Constant 35 = unstable |

### What "Good" Training Looks Like

```
Epoch 1:  loss_cls=1.8, loss_pts=0.08, mAP=~5   (learning to separate classes)
Epoch 5:  loss_cls=0.9, loss_pts=0.03, mAP=~25  (finding map elements)
Epoch 10: loss_cls=0.5, loss_pts=0.015, mAP=~40 (refining positions)
Epoch 15: loss_cls=0.4, loss_pts=0.010, mAP=~48 (fine-tuning)
Epoch 20: loss_cls=0.35, loss_pts=0.008, mAP=~52 (converging)
Epoch 24: loss_cls=0.32, loss_pts=0.007, mAP=~54 (final)
```

### Expected Training Timeline

| GPUs | Batch Size | Epochs | Wall Time | Final mAP |
|------|-----------|--------|-----------|-----------|
| 1x RTX 3090 | 4 | 24 | ~5 days | 54.1 |
| 1x A100 | 8 | 24 | ~4 days | 54.1 |
| 4x A100 | 32 | 24 | ~24 hours | 54.1 |
| 8x A100 | 64 | 24 | ~14 hours | 53.8* |

*Large batch may require LR scaling or longer warmup.

---

## Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| NaN loss | Loss becomes NaN after few iterations | Reduce LR to 2e-4; enable gradient clipping; check data (missing images?) |
| OOM | CUDA out of memory | Reduce batch size; enable FP16; reduce sequence_length; use gradient accumulation |
| Slow convergence | mAP stuck below 40 after 10 epochs | Verify pretrained weights loaded (check for "Pretrained" in log); check data augmentation is not too strong |
| Temporal instability | Loss spikes at sequence boundaries | Ensure hidden state reset at scene boundaries; reduce sequence_length |
| Poor recall | Many GT elements unmatched (>50% FN) | Increase num_queries (150 -> 200); check GT generation (too many elements being filtered?) |
| All predictions are background | cls loss stays high, no map elements predicted | Learning rate too low for decoder; increase lr_mult for map_decoder; check loss weights |
| Gradient explosion | grad_norm always at clip threshold | Reduce LR; check for numerical instability in depth prediction (log-softmax) |
| Data loading bottleneck | GPU utilization < 50% | Increase num_workers; move data to SSD; pre-extract images |

---

## Ablation Training Configs

### Single-Frame Baseline (No Temporal)

```python
# Disable temporal module
model = dict(temporal_fusion=None)
data = dict(train=dict(sequence_length=1))  # No sequences needed
```

### Different Temporal Window Sizes

```python
# Short context (3 frames)
data = dict(train=dict(sequence_length=4))
model = dict(temporal_fusion=dict(temporal_window=3))

# Long context (8 frames)  
data = dict(train=dict(sequence_length=9))
model = dict(temporal_fusion=dict(temporal_window=8))
```

### Backbone Comparison

```bash
# ResNet-18 (faster, lower accuracy)
python tools/train.py configs/streammapnet/streammapnet_r18_24ep_nuscenes.py

# ResNet-101 (slower, higher accuracy)
python tools/train.py configs/streammapnet/streammapnet_r101_24ep_nuscenes.py

# Swin-Tiny (transformer backbone)
python tools/train.py configs/streammapnet/streammapnet_swin_t_24ep_nuscenes.py
```

### Temporal Fusion Method Comparison

```python
# Concatenation + Conv (simple baseline)
model = dict(temporal_fusion=dict(type='TemporalConcat'))

# Cross-attention without gate
model = dict(temporal_fusion=dict(type='TemporalCrossAttention', use_gate=False))

# Cross-attention with gate (default, best)
model = dict(temporal_fusion=dict(type='TemporalCrossAttention', use_gate=True))
```

---

## Advanced Training Techniques

### Curriculum Learning for Temporal Length

Start with short sequences and increase over training:

```python
# Custom hook to increase sequence length
class TemporalCurriculumHook:
    def before_epoch(self, runner):
        epoch = runner.epoch
        if epoch < 5:
            runner.data_loader.dataset.sequence_length = 3
        elif epoch < 12:
            runner.data_loader.dataset.sequence_length = 5
        else:
            runner.data_loader.dataset.sequence_length = 8
```

### EMA (Exponential Moving Average)

For more stable final model:

```python
ema_config = dict(
    enabled=True,
    decay=0.9999,
    update_interval=1,
)
```

### Evaluation During Training

```python
evaluation = dict(
    interval=4,          # Evaluate every 4 epochs
    metric='chamfer',    # Primary metric
    save_best='mAP',    # Save checkpoint with best mAP
)
```

---

## References

- Yuan et al. (2024). StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction. WACV 2024.
- Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization (AdamW). ICLR 2019.
- Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV 2017.
- Kuhn (1955). The Hungarian Method for the Assignment Problem.
- Micikevicius et al. (2018). Mixed Precision Training. ICLR 2018.
