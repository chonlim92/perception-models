# BEVFormer: Complete Training Guide

## Step-by-Step Tutorial with Explanations

This guide takes you from zero to a trained BEVFormer model. Every step includes not just WHAT to do, but WHY each choice was made. Written for a Staff AI Engineer who knows PyTorch but is new to autonomous driving perception training.

---

## 1. Prerequisites

### 1.1 Why BEVFormer Needs Significant GPU Resources

BEVFormer is memory-intensive because of the combination of:
1. **Large multi-camera input:** 6 images x 3 scales x 256 channels = 178,500 feature tokens stored for backprop
2. **Large BEV grid:** 40,000 queries x 256 dimensions, refined over 6 layers
3. **Attention maps:** Deformable attention stores offsets and weights for all sampling points
4. **Temporal caching:** Previous frame BEV features must be kept in memory

### 1.2 Hardware Requirements

| Resource | Minimum (can train) | Recommended | Ideal |
|----------|-------------------|-------------|-------|
| GPU | 1x RTX 3090 (24GB) | 4x A100 (40GB) | 8x A100 (80GB) |
| GPU Memory | 24 GB | 40 GB | 80 GB |
| CPU Cores | 8 | 32 | 64 |
| System RAM | 32 GB | 128 GB | 256 GB |
| Storage | 200 GB SSD | 500 GB NVMe | 1 TB NVMe |
| CUDA | 11.3+ | 11.8 | 11.8 or 12.1 |

**Why these numbers?**
- 24 GB GPU: Fits batch_size=1 with FP16, BEV 200x200 (just barely)
- 128 GB RAM: Data loading with 4 workers, each loading 6 images + metadata
- NVMe SSD: Random access to ~240,000 JPEG images during training -- HDD would bottleneck

### 1.3 CUDA/cuDNN Compatibility

| PyTorch | CUDA | cuDNN | Status |
|---------|------|-------|--------|
| 2.0.1 | 11.8 | 8.7 | Recommended (stable) |
| 1.13.1 | 11.7 | 8.5 | Also works |
| 2.1.0 | 12.1 | 8.9 | Works but may need custom ops rebuild |

**Critical:** The CUDA version used to compile deformable attention CUDA kernels MUST match your runtime CUDA. Mismatches cause cryptic segfaults.

---

## 2. Environment Setup

### 2.1 What Is the OpenMMLab Ecosystem?

BEVFormer is built on the OpenMMLab framework, which provides:
- **mmcv:** Core utilities (data loading, config system, hooks, runners)
- **mmdet:** 2D object detection components (backbones, necks, losses)
- **mmdet3d:** 3D detection extensions (point cloud processing, 3D heads, nuScenes eval)

Think of it as an ML framework above PyTorch -- it handles training loops, distributed training, config management, and checkpointing so you focus on model architecture.

### 2.2 Step-by-Step Installation

```bash
# Step 1: Create isolated conda environment
conda create -n bevformer python=3.8 -y
conda activate bevformer

# Why Python 3.8? OpenMMLab packages have verified compatibility with 3.8.
# Python 3.9+ may work but can have edge cases with mmcv build.

# Step 2: Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Why torch 2.0.1? Stable release with good CUDA 11.8 support.
# The --index-url ensures you get the CUDA-enabled version (not CPU-only).

# Step 3: Install mmcv (the foundation layer)
pip install openmim
mim install mmcv-full==1.7.1

# Why mmcv-full (not mmcv)? "full" includes CUDA-compiled custom ops
# (deformable conv, RoI ops, etc). Without "full", deformable attention
# will not have its CUDA kernel and will be extremely slow.

# Step 4: Install mmdet and mmsegmentation
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0

# Step 5: Install mmdet3d
pip install mmdet3d==1.0.0rc6

# Step 6: Install additional dependencies
pip install numpy==1.23.5           # Pinned for pickle compatibility
pip install nuscenes-devkit==1.1.10  # nuScenes data loading and evaluation
pip install pyquaternion             # Quaternion math for rotations
pip install einops                   # Tensor reshaping utilities
pip install scikit-image             # Image processing for augmentation

# Step 7: Build BEVFormer custom CUDA ops (deformable attention)
cd projects/mmdet3d_plugin/ops
python setup.py develop
cd ../../..

# This compiles custom CUDA kernels for multi-scale deformable attention.
# If this fails, check that nvcc version matches torch.version.cuda.
```

### 2.3 Verify Installation

```python
import torch
import mmcv
import mmdet
import mmdet3d
from mmcv.ops import MultiScaleDeformableAttention

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"mmcv: {mmcv.__version__}")
print(f"mmdet: {mmdet.__version__}")
print(f"mmdet3d: {mmdet3d.__version__}")
print(f"Deformable attention: OK")  # If import didn't fail
```

### 2.4 Common Installation Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: mmcv._ext` | mmcv compiled without CUDA | `pip install mmcv-full==1.7.1` (note: "full") |
| `RuntimeError: CUDA error` on deformable attn | Wrong CUDA arch | `TORCH_CUDA_ARCH_LIST="8.0" python setup.py develop` |
| `ImportError: cannot import MultiScaleDeformableAttention` | Custom ops not built | `cd ops && python setup.py develop` |
| `nvcc not found` | CUDA toolkit not in PATH | `export PATH=/usr/local/cuda/bin:$PATH` |

---

## 3. Data Preparation

### 3.1 Download (See data_collection.md for full details)

```bash
# Quick start with mini dataset (4 GB, for testing pipeline)
# Download v1.0-mini from nuscenes.org
tar -xzf v1.0-mini.tgz -C data/nuscenes/

# Full dataset (72 GB minimum for camera-only)
# Download metadata + camera blobs + can_bus (see data_collection.md)
```

### 3.2 Generate Info Pickle Files

```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-trainval \
    --canbus ./data/nuscenes

# What this produces:
#   nuscenes_infos_temporal_train.pkl  (~1.5 GB)
#   nuscenes_infos_temporal_val.pkl    (~0.3 GB)
#
# What's inside:
#   - Pre-computed camera calibration matrices (lidar2img)
#   - Temporal links (prev/next sample tokens)
#   - Ground truth boxes converted to ego frame
#   - CAN bus data per sample
#   - File paths to all camera images
```

### 3.3 Verify Data is Correct

```bash
python -c "
import pickle
with open('data/nuscenes/nuscenes_infos_temporal_train.pkl', 'rb') as f:
    data = pickle.load(f)
infos = data['infos']
print(f'Training samples: {len(infos)}')
print(f'First sample keys: {list(infos[0].keys())}')
print(f'Cameras: {list(infos[0][\"cams\"].keys())}')
print(f'GT boxes shape: {infos[0][\"gt_boxes\"].shape}')
print(f'Has can_bus: {\"can_bus\" in infos[0]}')
print(f'Has prev token: {infos[0][\"prev\"] != \"\"}')
"
# Expected output:
# Training samples: 28130
# Cameras: ['CAM_FRONT', 'CAM_FRONT_RIGHT', ...]
# GT boxes shape: (N, 9)
# Has can_bus: True
# Has prev token: True (for most samples)
```

---

## 4. Understanding the Config File

The config file controls EVERYTHING about training. Let us explain the key sections.

### 4.1 Model Configuration

```python
model = dict(
    type='BEVFormer',
    use_grid_mask=True,       # Grid masking augmentation on image features
                              # Randomly masks out rectangular regions
                              # Forces the model to not rely on specific regions
    
    video_test_mode=True,     # During eval, process frames sequentially
                              # (temporal BEV carries over between frames)
                              # If False, each frame is processed independently
    
    img_backbone=dict(
        type='ResNet',
        depth=101,            # 101 layers (vs 50 or 152)
        num_stages=4,
        out_indices=(1, 2, 3),  # Output C2, C3, C4 (for FPN)
        frozen_stages=1,      # Freeze stage 1 (stem + first block)
                              # WHY: early features (edges, textures) are
                              # universal -- no need to fine-tune them
        norm_cfg=dict(type='BN2d', requires_grad=False),
                              # Freeze BatchNorm statistics
                              # WHY: with batch_size=1, BN stats are noisy
                              # Using frozen ImageNet BN stats is more stable
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
                              # DCN only in stages 3-4
                              # WHY: early stages extract low-level features
                              # that don't benefit from deformation
    ),
)
```

### 4.2 Optimizer Configuration

```python
optimizer = dict(
    type='AdamW',           # Adam with weight decay (decoupled)
    lr=2e-4,                # Base learning rate
    weight_decay=0.01,      # L2 regularization to prevent overfitting
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  # 10x LOWER LR for backbone
            # WHY: backbone is pretrained, needs gentle fine-tuning
            # Without this, pretrained features get destroyed early in training
        }
    ),
)

optimizer_config = dict(
    type='Fp16OptimizerHook',   # Mixed precision training
    loss_scale='dynamic',        # Automatically adjust loss scale
    grad_clip=dict(max_norm=35, norm_type=2),
    # WHY grad_clip at 35: Transformers can have gradient explosions
    # especially early in training. Clipping prevents NaN but is lenient
    # enough to not slow convergence.
)
```

### 4.3 Learning Rate Schedule

```python
lr_config = dict(
    policy='CosineAnnealing',   # Cosine decay from peak to min
    warmup='linear',            # Linear warmup at start
    warmup_iters=500,           # Warmup for 500 iterations
    warmup_ratio=1.0 / 3,      # Start warmup at 1/3 of base LR
    min_lr_ratio=1e-3,          # End at 1/1000 of base LR
)
```

```
Learning Rate Over Training:

LR
^
|  Peak: 2e-4
|    /\
|   /  \
|  /    \        Cosine Annealing
| /      \
|/        \
+--+-------\------------------> iterations
0  500      \
 (warmup)    \_______________  Min: 2e-7
             12000         84000
```

**Why warmup?** Transformers are unstable at initialization. Random attention weights cause large, noisy gradients. Starting with a small LR lets the model establish reasonable attention patterns before ramping up learning.

**Why cosine annealing?** Smooth LR reduction (vs step decay) avoids abrupt loss spikes. The cosine shape spends more time at moderate LR (productive learning) and less time at very low LR (diminishing returns).

---

## 5. Training Commands

### 5.1 Single-GPU Training

```bash
# Basic single-GPU training
python tools/train.py \
    projects/configs/bevformer/bevformer_base.py \
    --work-dir work_dirs/bevformer_base \
    --gpu-ids 0

# What each flag means:
# --work-dir: Where to save checkpoints, logs, and config copy
# --gpu-ids 0: Use GPU 0
```

### 5.2 Multi-GPU Training (DDP)

```bash
# 8-GPU Distributed Data Parallel training
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_base.py \
    8 \
    --work-dir work_dirs/bevformer_base

# Equivalent using torchrun directly:
torchrun --nproc_per_node=8 \
    tools/train.py \
    projects/configs/bevformer/bevformer_base.py \
    --work-dir work_dirs/bevformer_base \
    --launcher pytorch
```

### 5.3 Resume From Checkpoint

```bash
./tools/dist_train.sh \
    projects/configs/bevformer/bevformer_base.py \
    8 \
    --work-dir work_dirs/bevformer_base \
    --resume-from work_dirs/bevformer_base/latest.pth

# WHY resume-from (not load-from):
# --resume-from: Loads model + optimizer + scheduler + epoch counter
#                Training continues exactly where it left off
# --load-from:   Loads only model weights, resets optimizer/scheduler
#                Used for fine-tuning or transfer learning
```

---

## 6. What Happens During Training

### 6.1 Expected Loss Progression

| Epoch | Total Loss | Cls Loss | Bbox Loss | Notes |
|-------|-----------|----------|-----------|-------|
| 1 | ~12.0 | ~8.0 | ~4.0 | High loss, model is learning basic attention patterns |
| 3 | ~9.0 | ~6.0 | ~3.0 | Detection starting to work |
| 6 | ~7.5 | ~5.0 | ~2.5 | Reasonable detections, still many FP/FN |
| 12 | ~6.0 | ~4.0 | ~2.0 | Good detections, refining quality |
| 18 | ~5.5 | ~3.5 | ~2.0 | Near convergence |
| 24 | ~5.2 | ~3.3 | ~1.9 | Converged (best model) |

### 6.2 Signs of Healthy Training

- Loss decreases monotonically (with noise) through first 6 epochs
- Gradient norm is stable (not exploding): should be 5-20, not >100
- Learning rate follows expected schedule (check tensorboard)
- GPU utilization >90% (not I/O bottlenecked)
- No NaN in loss

### 6.3 Signs of Unhealthy Training

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Loss increases after epoch 1 | LR too high | Reduce to 1e-4 |
| Loss goes to NaN | Gradient explosion | Check data, reduce LR, increase grad_clip |
| Loss stuck (no decrease) | LR too low or broken data | Check data pipeline, increase LR |
| GPU utilization <50% | I/O bottleneck | Use SSD, increase workers, enable persistent_workers |
| OOM after a few iterations | Memory leak or variable-size data | Check batch collation, use fixed-size padding |

---

## 7. Multi-GPU Training with DDP

### 7.1 How DDP Works

Distributed Data Parallel:
1. Model is replicated to each GPU
2. Data is split across GPUs (each GPU gets different samples)
3. Each GPU computes forward + backward independently
4. Gradients are AVERAGED across all GPUs (all-reduce)
5. Each GPU applies the same gradient update -> models stay synchronized

```
GPU 0: batch[0:1] -> loss -> grads -> |
GPU 1: batch[1:2] -> loss -> grads -> |-- all_reduce(grads) / N_gpus
GPU 2: batch[2:3] -> loss -> grads -> |     -> update weights
...                                    |
GPU 7: batch[7:8] -> loss -> grads -> |
```

### 7.2 Linear Scaling Rule

With more GPUs, effective batch size increases. To maintain training dynamics:

```
LR_new = LR_base * (effective_batch_size / reference_batch_size)

BEVFormer default: 2e-4 for 8 GPUs with batch_size=1 (effective batch = 8)
If using 4 GPUs: LR = 2e-4 * (4/8) = 1e-4
If using 16 GPUs: LR = 2e-4 * (16/8) = 4e-4
```

### 7.3 NCCL Configuration

```bash
# For multi-GPU on single machine (usually automatic)
export NCCL_DEBUG=INFO          # Verbose NCCL logging (for debugging)
export NCCL_SOCKET_IFNAME=eth0  # Network interface (for multi-node)
export NCCL_IB_DISABLE=0        # Enable InfiniBand (if available)

# For multi-node training
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
```

---

## 8. Mixed Precision Training

### 8.1 What Is Mixed Precision?

Mixed precision uses FP16 (16-bit floating point) for most computations while keeping FP32 (32-bit) for numerically sensitive operations:

```
FP32 (32-bit): 1 sign + 8 exponent + 23 mantissa bits
  Range: +/- 3.4e38, Precision: ~7 decimal digits
  
FP16 (16-bit): 1 sign + 5 exponent + 10 mantissa bits
  Range: +/- 65504, Precision: ~3 decimal digits
  2x less memory, 2-8x faster on Tensor Cores
```

### 8.2 Configuration

```python
fp16 = dict(loss_scale='dynamic')
# 'dynamic' means: start with a large loss scale (e.g., 2^15),
# if overflow is detected, reduce scale by 2x
# if no overflow for 2000 steps, increase scale by 2x
```

### 8.3 What Stays in FP32 and Why

| Operation | Precision | Why |
|-----------|-----------|-----|
| LayerNorm | FP32 | Normalization requires high precision for mean/variance |
| Softmax | FP32 | exp() can overflow in FP16 range |
| Loss computation | FP32 | Small loss values would underflow in FP16 |
| Gradient accumulation | FP32 | Prevents rounding errors from accumulating |
| Everything else | FP16 | Safe and 2x faster |

### 8.4 Memory Savings

| Configuration | Memory per GPU | Training Speed |
|---------------|---------------|----------------|
| FP32 | ~30 GB | 1.0x (baseline) |
| FP16 (mixed) | ~18 GB | 1.3x faster |

---

## 9. Common Issues and Fixes

### 9.1 Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory. Tried to allocate X GiB`

**Systematic approach to reducing memory:**

```python
# Priority 1: Enable FP16 (saves ~40%)
fp16 = dict(loss_scale='dynamic')

# Priority 2: Gradient checkpointing on backbone (saves ~3 GB)
img_backbone=dict(with_cp=True)

# Priority 3: Reduce BEV resolution (saves ~8 GB but hurts accuracy)
pts_bbox_head=dict(bev_h=100, bev_w=100)  # 100x100 instead of 200x200

# Priority 4: Reduce encoder layers (saves ~2 GB)
encoder=dict(num_layers=3)  # instead of 6

# Priority 5: Reduce image resolution
# Add to data pipeline: dict(type='ResizeMultiViewImage', size=(450, 800))

# Priority 6: Reduce object queries
num_query=300  # instead of 900
```

### 9.2 NaN Loss

**Symptom:** Loss becomes NaN (Not a Number), training diverges

**Debugging checklist:**

```bash
# 1. Is it a data issue?
# Check for corrupted images or invalid calibration matrices
python tools/verify_data.py --data-path data/nuscenes

# 2. Is the learning rate too high?
# Try 1e-4 instead of 2e-4
optimizer = dict(lr=1e-4)

# 3. Are gradients exploding?
# Reduce gradient clip
optimizer_config = dict(grad_clip=dict(max_norm=10))  # instead of 35

# 4. Is FP16 causing underflow?
# Try without FP16 first
# Remove: fp16 = dict(loss_scale='dynamic')

# 5. Is deformable attention producing invalid values?
# Check reference points are in valid range [0, 1]
# Check sampling locations are within feature map bounds
```

### 9.3 Loss Not Decreasing (Plateau)

**Symptom:** Loss stays flat for many epochs

**Possible causes and fixes:**

```python
# 1. Backbone not loaded correctly
# Verify in log: "load model from: ckpts/r101_dcn_fcos3d_pretrain.pth"
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'

# 2. Learning rate too low
# If using fewer GPUs, LR might be scaled too aggressively
optimizer = dict(lr=2e-4)  # don't scale below this for BEVFormer

# 3. Data pipeline issue
# Verify images are being loaded correctly
# Add debug visualization:
python tools/visualize_data.py --config configs/bevformer/bevformer_base.py

# 4. Temporal features not working
# Check queue_length setting
data = dict(train=dict(queue_length=4))
```

### 9.4 Deformable Attention CUDA Errors

**Symptom:** `RuntimeError: Error in deformable attention forward CUDA kernel`

```bash
# Fix 1: Rebuild custom ops with correct CUDA architecture
cd projects/mmdet3d_plugin/ops
rm -rf build/
TORCH_CUDA_ARCH_LIST="8.0" python setup.py develop  # A100
TORCH_CUDA_ARCH_LIST="8.6" python setup.py develop  # RTX 3090
TORCH_CUDA_ARCH_LIST="8.9" python setup.py develop  # RTX 4090

# Fix 2: Verify CUDA versions match
python -c "import torch; print(torch.version.cuda)"  # Should match nvcc
nvcc --version  # Should print same CUDA version

# Fix 3: Check mmcv CUDA ops
python -c "from mmcv.ops import MultiScaleDeformableAttention; print('OK')"
```

### 9.5 Temporal Features Not Helping

**Symptom:** Performance with temporal is same as without

```python
# Check 1: video_test_mode must be True for evaluation
model = dict(video_test_mode=True)

# Check 2: queue_length must be > 1
data = dict(train=dict(queue_length=4))

# Check 3: can_bus data must be loaded
# In training log, verify can_bus fields are present

# Check 4: use_can_bus must be True in transformer config
transformer=dict(use_can_bus=True, rotate_prev_bev=True, use_shift=True)
```

---

## 10. Checkpointing

### 10.1 Configuration

```python
checkpoint_config = dict(
    interval=1,          # Save every epoch
    max_keep_ckpts=5,    # Keep only last 5 (save disk space)
)

# Evaluation frequency
evaluation = dict(
    interval=24,         # Only evaluate at final epoch (saves time)
    pipeline=test_pipeline,
)
```

### 10.2 Checkpoint Contents

Each checkpoint file (~1 GB) contains:
- Model state dict (all parameters)
- Optimizer state dict (momentum, variance for AdamW)
- Training metadata (epoch, iteration, config)

### 10.3 Converting for Inference

```bash
# Remove optimizer (saves 50% disk space)
python tools/convert_checkpoint.py \
    work_dirs/bevformer_base/epoch_24.pth \
    work_dirs/bevformer_base/bevformer_base_inference.pth \
    --no-optimizer

# Result: ~500 MB (model weights only)
```

---

## 11. Training Time and Cost Estimates

### 11.1 Training Duration

| Setup | Time (24 epochs) | GPU-Hours | Est. Cloud Cost |
|-------|-------------------|-----------|-----------------|
| 8x A100 (80GB) | 28 hours | 224 | $900 |
| 8x A100 (40GB) | 30 hours | 240 | $960 |
| 4x A100 (80GB) | 48 hours | 192 | $770 |
| 8x V100 (32GB) | 58 hours | 464 | $930 |
| 8x RTX 3090 | 70 hours | 560 | $560 |
| 4x RTX 3090 | 130 hours | 520 | $520 |
| 1x A100 (80GB) | 210 hours | 210 | $840 |

### 11.2 Epoch Milestones

| Epoch | Expected NDS (val) | Notes |
|-------|-------------------|-------|
| 6 | ~40 | Basic detection working |
| 12 | ~46 | Good detection, rough quality |
| 18 | ~49 | Near convergence |
| 24 | ~51.7 | Final (best performance) |

---

## 12. Advanced Techniques

### 12.1 CBGS (Class-Balanced Group Sampling)

**Problem:** nuScenes has 48x more cars than construction vehicles. Without balancing, the model barely learns rare classes.

**Solution:** CBGS oversamples scenes containing rare classes so each class appears equally often during training.

```python
data = dict(
    train=dict(
        type='CBGSDataset',
        dataset=dict(type='CustomNuScenesDataset', ...),
    ),
)
```

**Impact:** +1.4 NDS from CBGS. Primarily helps rare classes (construction_vehicle, trailer).

### 12.2 Progressive Training

Train in stages for better convergence:

```bash
# Stage 1: Train without temporal (faster convergence)
# Config: queue_length=1, 12 epochs
# Result: model learns basic spatial detection

# Stage 2: Fine-tune with temporal
# Config: queue_length=4, 12 more epochs
# Load: stage 1 checkpoint
load_from = 'work_dirs/bevformer_no_temporal/epoch_12.pth'
```

**Why this helps:** Temporal self-attention depends on good spatial features (it aligns the previous BEV which must already be meaningful). Training spatial features first gives temporal attention a better starting point.

### 12.3 Knowledge Distillation from LiDAR

Use a trained LiDAR-based model to provide additional supervision:

```python
model = dict(
    type='BEVFormerDistill',
    teacher_config='configs/centerpoint_base.py',
    teacher_checkpoint='ckpts/centerpoint.pth',
    distill_loss_weight=1.0,
    # Teacher provides dense BEV features as soft targets
    # Student learns to match LiDAR-quality BEV from cameras alone
)
```

### 12.4 EMA (Exponential Moving Average)

Keep a running average of model weights for more stable evaluation:

```python
custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        momentum=0.001,  # EMA decay rate
        priority=49,
    ),
]
# EMA model typically gives +0.3-0.5 NDS vs raw model
```

---

## 13. Evaluation After Training

### 13.1 Evaluate Best Checkpoint

```bash
# Multi-GPU evaluation (faster)
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    8 \
    --eval bbox

# Single-GPU evaluation
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --eval bbox \
    --gpu-ids 0
```

### 13.2 What to Look For in Results

```
Expected output:
  mAP: 0.416 (+/- 0.005)    -- detection accuracy
  NDS: 0.517 (+/- 0.005)    -- overall score
  mAVE: 0.394 (+/- 0.015)   -- velocity error (sensitive to temporal)

If mAP < 0.38: Check backbone pretrain, training epochs, data pipeline
If mAVE > 0.6: Check temporal fusion (can_bus, queue_length, video_test_mode)
If NDS < 0.49: Something is significantly wrong -- systematic debugging needed
```

### 13.3 Submit to Leaderboard

```bash
# Generate submission file for nuScenes test set
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --format-only \
    --eval-options jsonfile_prefix=results/bevformer_submit

# Upload results/bevformer_submit/results_nusc.json to:
# https://eval.ai/web/challenges/challenge-page/356/
```

---

## 14. Training Monitoring

### 14.1 TensorBoard

```bash
tensorboard --logdir work_dirs/bevformer_base --port 6006

# Key curves to monitor:
# loss: Should decrease monotonically (with noise)
# loss_cls: Classification loss (should converge to ~3.3)
# loss_bbox: Box regression loss (should converge to ~1.9)
# grad_norm: Should be stable 5-20 (not exploding >100)
# lr: Should follow warmup + cosine schedule
```

### 14.2 Log Analysis

```bash
# Plot loss curves from log file
python tools/analysis_tools/analyze_logs.py plot_curve \
    work_dirs/bevformer_base/bevformer_base.log.json \
    --keys loss loss_cls loss_bbox \
    --out loss_curve.png
```

### 14.3 Quick Health Check During Training

```bash
# Check training is progressing (run periodically)
tail -5 work_dirs/bevformer_base/bevformer_base.log

# Expected output after ~1 epoch:
# Epoch [1][3500/3516]  lr: 2.000e-04, loss: 7.482, loss_cls: 4.921, loss_bbox: 2.561
# Gradient norm should be shown if enabled
```
