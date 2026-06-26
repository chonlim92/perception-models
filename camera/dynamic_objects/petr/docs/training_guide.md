# Training Guide - PETR / PETRv2 / StreamPETR

## Overview

This guide covers practical training procedures for the PETR family of models, including hardware requirements, multi-GPU setup, hyperparameter tuning, and model-specific training strategies.

---

## Hardware Requirements

### GPU Memory Requirements

| Model | Backbone | Batch Size | GPU Memory | Min GPUs |
|-------|----------|-----------|------------|----------|
| PETR | ResNet-50 | 1 per GPU | ~6 GB | 8x V100/A100 |
| PETR | ResNet-101 | 1 per GPU | ~8 GB | 8x V100/A100 |
| PETRv2 | ResNet-50 | 1 per GPU | ~10 GB | 8x V100/A100 |
| PETRv2 | ResNet-101 | 1 per GPU | ~14 GB | 8x A100 |
| StreamPETR | ResNet-50 | 1 per GPU | ~8 GB | 8x V100/A100 |
| StreamPETR | VoVNet-99 | 1 per GPU | ~16 GB | 8x A100 |
| StreamPETR | ViT-L | 1 per GPU | ~32 GB | 8x A100-80G |

### Training Time Estimates (8x A100 40GB)

| Model | Epochs | Time per Epoch | Total Training |
|-------|--------|---------------|----------------|
| PETR (R50) | 24 | ~45 min | ~18 hours |
| PETRv2 (R50) | 24 | ~60 min | ~24 hours |
| StreamPETR (R50) | 24 | ~55 min | ~22 hours |
| StreamPETR (VoVNet-99) | 24 | ~90 min | ~36 hours |

---

## Training Commands

### PETR Base Training

```bash
# Single-node, 8 GPU training
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    tools/train.py \
    configs/petr_r50_nuscenes.yaml \
    --launcher pytorch \
    --work-dir work_dirs/petr_r50

# With torchrun (preferred for PyTorch >= 1.9)
torchrun --nproc_per_node=8 \
    tools/train.py \
    configs/petr_r50_nuscenes.yaml \
    --work-dir work_dirs/petr_r50
```

### PETRv2 Training

```bash
# PETRv2 requires temporal info files
torchrun --nproc_per_node=8 \
    tools/train.py \
    configs/petrv2_r50_nuscenes.yaml \
    --work-dir work_dirs/petrv2_r50

# Optional: Initialize from PETR checkpoint for faster convergence
torchrun --nproc_per_node=8 \
    tools/train.py \
    configs/petrv2_r50_nuscenes.yaml \
    --work-dir work_dirs/petrv2_r50 \
    --load-from work_dirs/petr_r50/epoch_24.pth
```

### StreamPETR Training

```bash
# StreamPETR streaming training
torchrun --nproc_per_node=8 \
    tools/train.py \
    configs/stream_petr_r50_nuscenes.yaml \
    --work-dir work_dirs/stream_petr_r50

# Resume from checkpoint (e.g., after interruption)
torchrun --nproc_per_node=8 \
    tools/train.py \
    configs/stream_petr_r50_nuscenes.yaml \
    --work-dir work_dirs/stream_petr_r50 \
    --resume-from work_dirs/stream_petr_r50/latest.pth
```

---

## Learning Rate Schedule

### Cosine Annealing with Linear Warmup

All PETR variants use the same LR schedule:

```
LR Schedule:
  
  lr_max = 2e-4
  warmup_iters = 500
  warmup_ratio = 0.33
  
  Phase 1 (Warmup, iterations 0-500):
    lr(t) = lr_max * warmup_ratio + (lr_max - lr_max*warmup_ratio) * t/500
    lr(0) = 6.6e-5, lr(500) = 2e-4
  
  Phase 2 (Cosine Decay, iterations 500 to end):
    lr(t) = lr_max * 0.5 * (1 + cos(pi * (t-500) / (total_iters-500)))
    lr(end) = lr_max * min_lr_ratio = 2e-7

Graphical representation:
  lr
  2e-4 ─────╲
  |    /     ╲
  |   /       ╲╲
  |  /          ╲╲╲
  | /              ╲╲╲╲╲╲
  |/                       ╲╲╲╲╲
  ├──────┼─────────────────────────── iterations
  0    500                        end
  warmup        cosine decay
```

### Layer-wise Learning Rate Decay

Different parts of the model use different learning rates:

| Component | LR Multiplier | Effective LR | Rationale |
|-----------|--------------|-------------|-----------|
| Backbone (frozen stages) | 0.0 | 0 | Preserve low-level features |
| Backbone (unfrozen stages) | 0.1 | 2e-5 | Slow fine-tuning |
| FPN | 1.0 | 2e-4 | Train from scratch |
| 3D PE MLP | 1.0 | 2e-4 | Critical, train fully |
| Transformer decoder | 1.0 | 2e-4 | Train from scratch |
| Detection head | 1.0 | 2e-4 | Train from scratch |

---

## Backbone Freezing Strategies

### Strategy 1: Freeze Stage 1 Only (Default)

```yaml
backbone:
  frozen_stages: 1  # Freeze conv1 + bn1 + layer1
```

- Keeps low-level features (edges, textures) fixed
- Reduces memory by ~15%
- Recommended for most training runs

### Strategy 2: Progressive Unfreezing

```python
# Unfreeze backbone stages gradually during training
schedule = {
    0: frozen_stages=3,    # Epochs 0-5: only train decoder
    6: frozen_stages=2,    # Epochs 6-11: unfreeze stage 4
    12: frozen_stages=1,   # Epochs 12-23: unfreeze stage 3
}
```

- Can improve final performance by ~0.5 mAP
- Reduces risk of catastrophic forgetting
- More complex training script required

### Strategy 3: Full Backbone Freeze (for large backbones)

```yaml
backbone:
  frozen_stages: 4  # Freeze entire backbone
  norm_eval: true
```

- Used when backbone is very large (ViT-L, Swin-L)
- Significantly reduces memory and training time
- Relies on backbone being pretrained on a large dataset (ImageNet-22K, etc.)

---

## Class-Balanced Grouping and Sampling (CBGS)

### Problem

nuScenes has severe class imbalance:

| Class | Instances | Proportion |
|-------|-----------|-----------|
| car | ~650K | 46% |
| pedestrian | ~220K | 15% |
| barrier | ~140K | 10% |
| traffic_cone | ~100K | 7% |
| truck | ~90K | 6% |
| trailer | ~55K | 4% |
| bus | ~30K | 2% |
| motorcycle | ~25K | 2% |
| bicycle | ~15K | 1% |
| construction_vehicle | ~12K | <1% |

### CBGS Solution

CBGS rebalances the dataset by oversampling frames containing rare classes:

```python
# CBGS assigns a sampling weight to each training sample
# Weight is proportional to the rarest class it contains
sample_weight[i] = 1.0 / class_frequency[rarest_class_in_sample_i]

# During training, samples are drawn with probability proportional to weight
# This means frames with construction_vehicles are sampled ~50x more often
```

### When to Use CBGS

| Scenario | Use CBGS? | Notes |
|----------|----------|-------|
| Standard training (24 epochs) | No | May overfit rare classes |
| Longer training (60+ epochs) | Yes | Helps with rare class recall |
| Competition settings | Yes | Often gives +1-2 mAP boost |
| Quick experimentation | No | Faster convergence without |

### Config

```yaml
training:
  cbgs: true
  # When CBGS is enabled, one "epoch" sees each sample ~1 time on average
  # but rare-class samples are repeated more often
  # Adjust epochs accordingly (usually 2x fewer epochs needed)
  epochs: 12  # Instead of 24 without CBGS
```

---

## Data Augmentation for Multi-View Consistency

### Key Constraint

When augmenting multi-view camera data, augmentations must maintain geometric consistency across views. Unlike single-image augmentation, you cannot independently augment each camera.

### Safe Augmentations (Applied Consistently)

1. **Global Rotation/Translation (BEV space)**
   - Rotate/translate all cameras and annotations together
   - Maintains cross-camera consistency
   ```yaml
   rotation: [-5.4, 5.4]  # degrees, applied to ego frame
   ```

2. **Global Scaling**
   - Scale all images by the same factor
   - Adjust intrinsics accordingly
   ```yaml
   resize: [0.38, 0.55]  # uniform across all cameras
   ```

3. **Horizontal Flip (with camera swapping)**
   - Flip all images AND swap left/right camera pairs:
     - FRONT_LEFT <-> FRONT_RIGHT
     - BACK_LEFT <-> BACK_RIGHT
   - Flip annotations in BEV space
   ```yaml
   flip: 0.5  # probability, applied to all cameras simultaneously
   ```

4. **GridMask (per-image, OK)**
   - Random rectangular masking applied independently per image
   - Does not affect geometry, so per-image application is safe
   ```yaml
   grid_mask:
     enabled: true
     probability: 0.7
   ```

5. **Color Jitter (per-image, OK)**
   - Photometric augmentation applied independently
   - Does not affect 3D geometry
   ```yaml
   color_jitter:
     enabled: true
     brightness: 0.2
   ```

### Unsafe Augmentations (Avoid or Apply Carefully)

| Augmentation | Issue | Mitigation |
|-------------|-------|-----------|
| Per-image crop | Breaks calibration | Adjust intrinsics if cropping |
| Per-image rotation | Misaligns 3D PE | Never rotate individual images |
| Per-image resize | Different scale per cam | Always resize uniformly |
| Random erasing | Could erase same object in all views | Apply per-image (OK) |

### Temporal Augmentation Consistency (PETRv2/StreamPETR)

For temporal models, augmentations must also be consistent across time:

```python
# All frames in a temporal window must share:
# - Same resize factor
# - Same flip decision
# - Same rotation angle

# GridMask and color jitter can differ across time
# (represents natural lighting changes)
```

---

## Model-Specific Training Notes

### PETR-Specific

1. **3D PE is critical**: If the 3D Position Embedding doesn't converge (check attention maps), training will fail completely
2. **Attention memory**: Global attention over 178K tokens is memory-intensive. Use FP16.
3. **Query initialization**: Random query positions work fine; no special initialization needed

### PETRv2-Specific

1. **Temporal warmup**: First few epochs may have unstable temporal features (previous frame features are random). Consider initializing from a trained PETR checkpoint.
2. **CAN bus dependency**: Ensure ego-motion data is correctly loaded. Silent failures here mean temporal alignment does nothing.
3. **Memory budget**: Storing previous frame features doubles the feature memory. Monitor GPU usage.

### StreamPETR-Specific

1. **Query propagation warmup**: In early training, propagated queries are low-quality (model hasn't learned to detect yet). This is fine - the model learns to use them gradually.
2. **Sequence length during training**: Training with frames_per_clip=4 gives good results. Longer clips marginally help but use more memory.
3. **Random temporal drop**: Dropping temporal connection with 10% probability during training makes the model robust to sequence boundaries.
4. **Gradient detachment**: Always detach gradients through propagated queries to prevent memory explosion during backprop-through-time.

---

## Common Training Issues and Solutions

### Issue: Loss Explodes in Early Training

**Symptoms**: NaN loss or extremely high values in first 100 iterations.

**Solutions**:
- Reduce initial learning rate (warmup_ratio from 0.33 to 0.1)
- Increase warmup iterations (500 -> 1000)
- Check that input normalization is correct (ImageNet mean/std)
- Verify camera intrinsics/extrinsics are loaded correctly

### Issue: mAP Plateaus Below Expected

**Symptoms**: mAP stuck at ~25 when ~31 expected for PETR-R50.

**Solutions**:
- Verify 3D PE is computed correctly (visualize attention maps)
- Check pc_range matches between position_embedding and data config
- Ensure Hungarian matching is working (monitor matching statistics)
- Try disabling augmentation to rule out data pipeline issues

### Issue: Velocity Prediction is Poor (High mAVE)

**Symptoms**: mAVE > 1.0 m/s (expected ~0.9 for PETR, ~0.4 for PETRv2).

**Solutions**:
- Verify velocity GT is in ego frame (not global frame)
- For temporal models: check ego-motion compensation is applied
- Reduce velocity code weight if it destabilizes training
- Increase training epochs (velocity needs more data to converge)

### Issue: GPU Out of Memory

**Symptoms**: CUDA OOM during training.

**Solutions**:
- Enable mixed precision (FP16): saves ~40% memory
- Reduce image resolution (resize: [0.3, 0.45])
- Use gradient checkpointing for backbone
- Reduce num_queries from 900 to 600 (slight performance drop)
- Use gradient accumulation (accumulate_grad_batches: 2) with half batch_size

### Issue: StreamPETR Queries Don't Propagate Well

**Symptoms**: Propagated queries have low confidence, model relies mostly on fresh queries.

**Solutions**:
- Lower propagation_threshold to 0.0 (propagate all top-K regardless of confidence)
- Initialize from a trained PETR checkpoint so queries start meaningful
- Increase num_propagated_queries from 256 to 384
- Check ego-motion compensation is correct (visualize propagated reference points)

---

## Distributed Training Setup

### Single-Node Multi-GPU

```bash
# Using PyTorch distributed launch
export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    tools/train.py \
    configs/stream_petr_r50_nuscenes.yaml
```

### Multi-Node Training

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<MASTER_IP> \
    --master_port=29500 \
    tools/train.py \
    configs/stream_petr_r50_nuscenes.yaml

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<MASTER_IP> \
    --master_port=29500 \
    tools/train.py \
    configs/stream_petr_r50_nuscenes.yaml
```

### SLURM Cluster

```bash
#!/bin/bash
#SBATCH --job-name=stream_petr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=5
#SBATCH --mem=256G
#SBATCH --time=24:00:00

srun torchrun \
    --nproc_per_node=8 \
    tools/train.py \
    configs/stream_petr_r50_nuscenes.yaml \
    --work-dir work_dirs/stream_petr_r50 \
    --launcher slurm
```

---

## Training Monitoring

### Key Metrics to Watch

| Metric | Expected Behavior | Red Flag |
|--------|-------------------|----------|
| cls_loss | Decreases steadily, ~1.5 -> ~0.3 | Stuck above 1.0 after epoch 5 |
| bbox_loss | Decreases, ~5.0 -> ~1.5 | Oscillates wildly |
| total_loss | Smooth decrease | NaN or spikes |
| lr | Cosine curve | Flat (schedule not applied) |
| grad_norm | ~10-30 (clipped at 35) | Consistently hitting clip limit |
| memory | Stable after warmup | Monotonic increase (memory leak) |

### Logging Configuration

```bash
# TensorBoard logging
tensorboard --logdir work_dirs/ --port 6006

# Weights & Biases (if configured)
wandb login
# Add to config: training.logger: wandb
```

### Validation During Training

```yaml
training:
  eval_interval: 1  # Validate every epoch
  # Early stopping (optional):
  early_stopping:
    patience: 5  # Stop if NDS doesn't improve for 5 epochs
    metric: NDS
    mode: max
```
