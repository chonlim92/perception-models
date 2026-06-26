# BEVFormer: Training Guide

## Step-by-Step Training Instructions

This guide covers the complete training pipeline for BEVFormer, from environment setup to model evaluation.

---

## 1. Environment Setup

### 1.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x RTX 3090 (24GB) | 8x A100 (80GB) |
| CPU | 8 cores | 32+ cores |
| RAM | 32 GB | 128 GB |
| Storage | 200 GB SSD | 500 GB NVMe SSD |
| CUDA | 11.3+ | 11.8 |

### 1.2 Software Dependencies

```bash
# Create conda environment
conda create -n bevformer python=3.8 -y
conda activate bevformer

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install mmcv and mmdet (OpenMMLab ecosystem)
pip install openmim
mim install mmcv-full==1.7.1
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0

# Install mmdet3d
pip install mmdet3d==1.0.0rc6

# Install additional dependencies
pip install numpy==1.23.5
pip install nuscenes-devkit==1.1.10
pip install pyquaternion
pip install motmetrics
pip install scikit-image
pip install einops
pip install flash-attn  # Optional: for efficient attention

# Install BEVFormer-specific ops
cd projects/mmdet3d_plugin/ops
python setup.py develop
```

### 1.3 Verify Installation

```python
import torch
import mmcv
import mmdet
import mmdet3d

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"mmcv: {mmcv.__version__}")
print(f"mmdet: {mmdet.__version__}")
print(f"mmdet3d: {mmdet3d.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### 1.4 Clone Repository

```bash
git clone https://github.com/fundamentalvision/BEVFormer.git
cd BEVFormer

# Install in development mode
pip install -e .
```

---

## 2. Data Preparation

### 2.1 Dataset Directory Structure

```bash
# Expected data directory
data/
└── nuscenes/
    ├── maps/
    ├── samples/
    │   ├── CAM_FRONT/
    │   ├── CAM_FRONT_LEFT/
    │   ├── CAM_FRONT_RIGHT/
    │   ├── CAM_BACK/
    │   ├── CAM_BACK_LEFT/
    │   └── CAM_BACK_RIGHT/
    ├── sweeps/
    │   ├── CAM_FRONT/
    │   └── ...
    ├── v1.0-trainval/
    │   ├── attribute.json
    │   ├── calibrated_sensor.json
    │   ├── category.json
    │   ├── ego_pose.json
    │   ├── instance.json
    │   ├── log.json
    │   ├── map.json
    │   ├── sample.json
    │   ├── sample_annotation.json
    │   ├── sample_data.json
    │   ├── scene.json
    │   ├── sensor.json
    │   └── visibility.json
    └── can_bus/          # CAN bus data for ego-motion
```

### 2.2 Download CAN Bus Data

```bash
# CAN bus data is needed for ego-motion in temporal self-attention
# Download from the nuScenes expansion pack
wget https://www.nuscenes.org/data/can_bus.zip
unzip can_bus.zip -d data/nuscenes/
```

### 2.3 Generate Info Files

```bash
# Generate training and validation info pickle files
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-trainval \
    --canbus ./data/nuscenes

# This creates:
# data/nuscenes/nuscenes_infos_temporal_train.pkl
# data/nuscenes/nuscenes_infos_temporal_val.pkl
```

### 2.4 Info File Contents

Each entry in the info pickle file contains:

```python
info = {
    'lidar_path': str,              # Path to LiDAR file (reference)
    'token': str,                   # Sample token
    'sweeps': list,                 # Sweep data
    'cams': {                       # Per-camera info
        'CAM_FRONT': {
            'data_path': str,       # Image file path
            'type': 'CAM_FRONT',
            'sample_data_token': str,
            'sensor2ego_translation': [3],
            'sensor2ego_rotation': [4],      # quaternion
            'ego2global_translation': [3],
            'ego2global_rotation': [4],
            'sensor2lidar_translation': [3],
            'sensor2lidar_rotation': [4],
            'cam_intrinsic': [[3x3]],
            'timestamp': int,
        },
        # ... other cameras
    },
    'lidar2ego_translation': [3],
    'lidar2ego_rotation': [4],
    'ego2global_translation': [3],
    'ego2global_rotation': [4],
    'timestamp': int,
    'gt_boxes': np.array,           # (N, 9) [x,y,z,w,l,h,yaw,vx,vy]
    'gt_names': np.array,           # (N,) class names
    'gt_velocity': np.array,        # (N, 2) [vx, vy]
    'num_lidar_pts': np.array,      # (N,) points per object
    'num_radar_pts': np.array,      # (N,) radar points per object
    'valid_flag': np.array,         # (N,) boolean validity
    'prev': str,                    # Previous sample token
    'next': str,                    # Next sample token
    'can_bus': np.array,            # (18,) CAN bus data
}
```

### 2.5 Verify Data Preparation

```bash
# Quick verification
python -c "
import pickle
with open('data/nuscenes/nuscenes_infos_temporal_train.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'Training samples: {len(data[\"infos\"])}')
print(f'Sample keys: {data[\"infos\"][0].keys()}')
print(f'Cameras: {list(data[\"infos\"][0][\"cams\"].keys())}')
"
```

---

## 3. Configuration

### 3.1 Key Configuration Parameters

```python
# projects/configs/bevformer/bevformer_base.py

# Model
model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    
    # Backbone
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),  # C2, C3, C4
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    
    # Neck (FPN)
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    
    # BEV Encoder
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=200,
        bev_w=200,
        num_query=900,
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        
        # Transformer
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
            
            # Encoder config
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        # Temporal Self-Attention
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1,
                        ),
                        # Spatial Cross-Attention
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=4,
                            ),
                            embed_dims=256,
                        ),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'),
                ),
            ),
            
            # Decoder config
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'),
                ),
            ),
        ),
        
        # Loss
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),  # Not used for 3D
    ),
)

# Point cloud range
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
```

### 3.2 Data Pipeline Configuration

```python
# Training data pipeline
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d',
                                        'img']),
]

# Validation data pipeline  
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1600, 900),
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=class_names,
                  with_label=False),
             dict(type='CustomCollect3D', keys=['img']),
         ]),
]
```

---

## 4. Training Commands

### 4.1 Single-GPU Training (for debugging)

```bash
# Single GPU training
python tools/train.py \
    projects/configs/bevformer/bevformer_base.py \
    --work-dir work_dirs/bevformer_base \
    --gpu-ids 0

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    projects/configs/bevformer/bevformer_base.py \
    --work-dir work_dirs/bevformer_base
```

### 4.2 Multi-GPU Training (Distributed Data Parallel)

```bash
# 8-GPU DDP training (recommended)
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_base.py \
    8 \
    --work-dir work_dirs/bevformer_base

# Alternative: using torchrun
torchrun --nproc_per_node=8 \
    tools/train.py \
    projects/configs/bevformer/bevformer_base.py \
    --work-dir work_dirs/bevformer_base \
    --launcher pytorch

# Multi-node training (2 nodes, 8 GPUs each)
# Node 0 (master):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr="node0_ip" --master_port=29500 \
    tools/train.py projects/configs/bevformer/bevformer_base.py \
    --work-dir work_dirs/bevformer_base --launcher pytorch

# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr="node0_ip" --master_port=29500 \
    tools/train.py projects/configs/bevformer/bevformer_base.py \
    --work-dir work_dirs/bevformer_base --launcher pytorch
```

### 4.3 Resume Training

```bash
# Resume from latest checkpoint
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_base.py \
    8 \
    --work-dir work_dirs/bevformer_base \
    --resume-from work_dirs/bevformer_base/latest.pth
```

---

## 5. Hyperparameters

### 5.1 Optimizer Configuration

```python
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  # Lower LR for pretrained backbone
        }
    ),
)

optimizer_config = dict(
    type='Fp16OptimizerHook',  # Mixed precision
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
)
```

### 5.2 Learning Rate Schedule

```python
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
```

```
Learning Rate Schedule:
  
  lr
  ^
  |    /\
  |   /  \    Cosine Annealing
  |  /    \
  | /      \
  |/        \________________
  +--+--------+------------>  iterations
  0  500     12000        24000
     (warmup) (peak)      (end)
  
  Peak LR: 2e-4
  Min LR: 2e-7 (min_lr_ratio=1e-3)
  Backbone LR: 2e-5 (10x lower)
```

### 5.3 Training Schedule

| Parameter | Value |
|-----------|-------|
| Total epochs | 24 |
| Batch size per GPU | 1 |
| Total batch size (8 GPU) | 8 |
| Iterations per epoch | ~3,500 (28,130 samples / 8 batch) |
| Total iterations | ~84,000 |
| Warmup iterations | 500 |
| Warmup strategy | Linear |
| LR schedule | Cosine annealing |
| Weight decay | 0.01 |
| Gradient clipping | max_norm=35 |

### 5.4 Loss Weights

| Loss | Weight | Description |
|------|--------|-------------|
| Focal Loss (classification) | 2.0 | Class prediction |
| L1 Loss (regression) | 0.25 | Box parameter regression |
| GIoU Loss | 0.0 | Not used (3D boxes lack efficient IoU) |

### 5.5 Data Augmentation

| Augmentation | Parameters |
|--------------|-----------|
| PhotoMetricDistortion | brightness_delta=32, contrast_range=[0.5, 1.5] |
| Grid Mask | ratio=0.5, prob=0.7 |
| Random Resize | scale=[0.38, 0.55] |
| Normalization | ImageNet mean/std |

---

## 6. Training Time Estimates

### 6.1 Per-Iteration Timing

| GPU | Batch/GPU | Time per Iteration | Memory per GPU |
|-----|-----------|-------------------|----------------|
| A100 (80GB) | 1 | ~1.2s | ~18 GB |
| A100 (40GB) | 1 | ~1.3s | ~18 GB |
| V100 (32GB) | 1 | ~2.5s | ~22 GB |
| RTX 3090 (24GB) | 1 | ~3.0s | ~22 GB |

### 6.2 Total Training Time

| Setup | Time (24 epochs) |
|-------|-------------------|
| 8x A100 (80GB) | ~28 hours |
| 8x A100 (40GB) | ~30 hours |
| 4x A100 (80GB) | ~48 hours |
| 8x V100 (32GB) | ~58 hours |
| 8x RTX 3090 | ~70 hours |
| 4x RTX 3090 | ~130 hours |
| 1x A100 (80GB) | ~210 hours |

### 6.3 Epoch Milestones

| Epoch | NDS (val) | Notes |
|-------|-----------|-------|
| 6 | ~40 | Initial convergence |
| 12 | ~46 | Moderate performance |
| 18 | ~49 | Near convergence |
| 24 | ~51.7 | Final (best) |

---

## 7. Multi-GPU Setup with DDP

### 7.1 DDP Configuration

```python
# In config file
dist_params = dict(backend='nccl')
```

### 7.2 GPU Memory Optimization

```python
# Gradient checkpointing (saves memory, adds ~20% compute)
model = dict(
    img_backbone=dict(
        with_cp=True,  # Checkpoint activations in backbone
    ),
)

# Reduce BEV resolution for limited memory
# 100x100 BEV instead of 200x200
pts_bbox_head=dict(
    bev_h=100,
    bev_w=100,
)
```

### 7.3 NCCL Environment Variables

```bash
# Recommended NCCL settings for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# For multi-node
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
```

---

## 8. Mixed Precision Training

### 8.1 Configuration

```python
# FP16 training (default for BEVFormer)
fp16 = dict(loss_scale='dynamic')

# This enables:
# - FP16 forward pass (backbone, encoder, decoder)
# - FP32 loss computation
# - Dynamic loss scaling to prevent underflow
```

### 8.2 Memory Savings

| Precision | Memory per GPU | Speed Impact |
|-----------|---------------|--------------|
| FP32 | ~30 GB | 1.0x |
| FP16 (mixed) | ~18 GB | 1.3x faster |

### 8.3 Numerical Stability Notes

- Layer normalization is kept in FP32 for stability
- Attention softmax computed in FP32
- Loss computation in FP32
- Some deformable attention operations may need FP32 fallback

---

## 9. Checkpointing Strategy

### 9.1 Default Checkpoint Configuration

```python
checkpoint_config = dict(
    interval=1,          # Save every epoch
    max_keep_ckpts=5,    # Keep last 5 checkpoints
)

# Evaluation during training
evaluation = dict(
    interval=24,         # Evaluate at last epoch only (to save time)
    pipeline=test_pipeline,
)
```

### 9.2 Checkpoint Contents

Each checkpoint (~1 GB) contains:

```python
checkpoint = {
    'meta': {
        'mmdet_version': str,
        'config': str,
        'CLASSES': tuple,
        'epoch': int,
        'iter': int,
    },
    'state_dict': OrderedDict,      # Model parameters
    'optimizer': dict,               # Optimizer state
}
```

### 9.3 Checkpoint Management

```bash
# List checkpoints
ls work_dirs/bevformer_base/

# Typical output:
# epoch_1.pth   (1.0 GB)
# epoch_2.pth   (1.0 GB)
# ...
# epoch_24.pth  (1.0 GB)
# latest.pth -> epoch_24.pth
# bevformer_base.log
# bevformer_base.log.json
# config.py (copy of training config)
```

### 9.4 Loading Pretrained Backbone

```python
# In config file
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
# This loads only backbone weights, other layers are randomly initialized
```

### 9.5 Converting Checkpoints

```bash
# Convert checkpoint for inference only (remove optimizer)
python tools/convert_checkpoint.py \
    work_dirs/bevformer_base/epoch_24.pth \
    work_dirs/bevformer_base/bevformer_base_inference.pth \
    --no-optimizer
```

---

## 10. Common Issues and Solutions

### 10.1 Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# 1. Reduce BEV resolution
pts_bbox_head=dict(bev_h=100, bev_w=100)  # Instead of 200x200

# 2. Reduce number of encoder layers
encoder=dict(num_layers=3)  # Instead of 6

# 3. Reduce number of object queries
num_query=300  # Instead of 900

# 4. Enable gradient checkpointing
img_backbone=dict(with_cp=True)

# 5. Reduce image resolution in data pipeline
dict(type='ResizeMultiViewImage', size=(450, 800))

# 6. Use smaller batch size (already 1 per GPU typically)
```

### 10.2 NaN Loss

**Symptom:** Loss becomes NaN during training

**Solutions:**
```bash
# 1. Reduce learning rate
optimizer = dict(lr=1e-4)  # Instead of 2e-4

# 2. Increase warmup
lr_config = dict(warmup_iters=1000)

# 3. Reduce gradient clip
optimizer_config = dict(grad_clip=dict(max_norm=10))

# 4. Disable FP16
# Remove: fp16 = dict(loss_scale='dynamic')

# 5. Check data: ensure no corrupted images
python tools/verify_data.py --data-path data/nuscenes
```

### 10.3 Slow Training

**Symptom:** Each iteration takes much longer than expected

**Solutions:**
```bash
# 1. Increase number of data loading workers
data = dict(workers_per_gpu=4)  # Default is often 4, try 8

# 2. Enable persistent workers
data = dict(persistent_workers=True)

# 3. Ensure data is on fast storage (SSD, not HDD)
# Check I/O with: iostat -x 1

# 4. Pin memory
data = dict(pin_memory=True)

# 5. Profile to identify bottleneck
python -m torch.profiler tools/train.py ...
```

### 10.4 Low mAP After Training

**Symptom:** Model trains successfully but mAP is significantly below expected

**Solutions:**
```bash
# 1. Verify pretrained backbone is loaded
# Check training log for: "load model from: ckpts/r101_dcn_fcos3d_pretrain.pth"

# 2. Verify data preprocessing
python tools/visualize_data.py --config projects/configs/bevformer/bevformer_base.py

# 3. Check temporal pairs
# Ensure prev/next tokens are correctly linked in info files

# 4. Verify calibration matrices
# Ensure lidar2img transforms produce valid projections

# 5. Train longer
# BEVFormer may need full 24 epochs to converge
```

### 10.5 Deformable Attention CUDA Error

**Symptom:** `RuntimeError: Error in deformable attention CUDA kernel`

**Solutions:**
```bash
# 1. Rebuild custom ops
cd projects/mmdet3d_plugin/ops
rm -rf build/
python setup.py develop

# 2. Ensure CUDA version matches PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# 3. Try different CUDA architecture
TORCH_CUDA_ARCH_LIST="8.0" python setup.py develop  # For A100
TORCH_CUDA_ARCH_LIST="8.6" python setup.py develop  # For RTX 3090
```

### 10.6 Temporal Features Not Working

**Symptom:** Model performance is same with/without temporal frames

**Solutions:**
```python
# 1. Verify queue_length is set correctly in config
data = dict(
    train=dict(
        queue_length=4,  # Number of temporal frames
    )
)

# 2. Check that can_bus data is loaded
# Training log should show can_bus fields

# 3. Verify ego-motion computation
# Check that prev_bev is being rotated/shifted correctly

# 4. Ensure video_test_mode is True for evaluation
model = dict(video_test_mode=True)
```

---

## 11. Training Monitoring

### 11.1 Tensorboard

```bash
# Launch tensorboard
tensorboard --logdir work_dirs/bevformer_base --port 6006

# Key metrics to monitor:
# - loss (total loss, should decrease)
# - loss_cls (classification loss)
# - loss_bbox (regression loss)
# - grad_norm (should be stable, not exploding)
# - lr (learning rate schedule)
```

### 11.2 Training Log Analysis

```bash
# Parse training log for loss curve
python tools/analysis_tools/analyze_logs.py plot_curve \
    work_dirs/bevformer_base/bevformer_base.log.json \
    --keys loss loss_cls loss_bbox \
    --out loss_curve.png
```

### 11.3 Expected Loss Progression

| Epoch | Total Loss | Cls Loss | Bbox Loss |
|-------|-----------|----------|-----------|
| 1 | ~12.0 | ~8.0 | ~4.0 |
| 6 | ~7.5 | ~5.0 | ~2.5 |
| 12 | ~6.0 | ~4.0 | ~2.0 |
| 18 | ~5.5 | ~3.5 | ~2.0 |
| 24 | ~5.2 | ~3.3 | ~1.9 |

---

## 12. Advanced Training Techniques

### 12.1 CBGS (Class-Balanced Grouping and Sampling)

```python
# Enable class-balanced sampling to handle class imbalance
dataset_type = 'CustomNuScenesDataset'
data = dict(
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            ...
        ),
    ),
)
```

### 12.2 Progressive Training

```bash
# Stage 1: Train without temporal (faster convergence)
# Use bevformer_base_no_temporal.py config for 12 epochs

# Stage 2: Fine-tune with temporal
# Load stage 1 checkpoint, train with temporal for 12 more epochs
load_from = 'work_dirs/bevformer_no_temporal/epoch_12.pth'
```

### 12.3 Knowledge Distillation

```python
# Use LiDAR-based teacher model for distillation
model = dict(
    type='BEVFormerDistill',
    teacher_config='configs/centerpoint_base.py',
    teacher_checkpoint='ckpts/centerpoint.pth',
    distill_loss_weight=1.0,
)
```

### 12.4 EMA (Exponential Moving Average)

```python
# Enable EMA for more stable training
custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.001,
        priority=49,
    ),
]
```

---

## 13. Evaluation During/After Training

### 13.1 Single Checkpoint Evaluation

```bash
# Evaluate a specific checkpoint
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    8 \
    --eval bbox

# Single GPU evaluation
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --eval bbox \
    --gpu-ids 0
```

### 13.2 Test-Time Augmentation

```python
# Enable TTA in config (flip augmentation)
test_pipeline = [
    ...
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1600, 900),
         pts_scale_ratio=1,
         flip=True,  # Enable horizontal flip
         transforms=[...]),
]
```

### 13.3 Submission to nuScenes Benchmark

```bash
# Generate submission file
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --format-only \
    --eval-options jsonfile_prefix=results/bevformer_base

# This creates: results/bevformer_base/results_nusc.json
# Upload to eval.ai/nuscenes for official evaluation
```
