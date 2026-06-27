# Radar Occupancy Grid Mapping — Training Guide

## Environment Setup, Training Commands, and Hyperparameter Tuning

---

## 1. Prerequisites and Environment Setup

### 1.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1080 Ti (11GB) | NVIDIA RTX 3090 (24GB) or A100 |
| CPU | 8 cores | 16+ cores |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB SSD (for nuScenes) | 500 GB NVMe SSD |
| GPU Memory | 11 GB (batch_size=8) | 24 GB (batch_size=16) |

### 1.2 Software Dependencies

**PyTorch environment:**

```bash
# Create conda environment
conda create -n radar_occ python=3.9
conda activate radar_occ

# Install PyTorch with CUDA
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install numpy>=1.21.0
pip install pyyaml>=6.0
pip install tqdm>=4.65.0
pip install tensorboard>=2.12.0
pip install nuscenes-devkit>=1.1.9
pip install scipy>=1.9.0
pip install matplotlib>=3.6.0

# Verify GPU is accessible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**TensorFlow environment (alternative):**

```bash
conda create -n radar_occ_tf python=3.9
conda activate radar_occ_tf

pip install tensorflow>=2.12.0
pip install numpy>=1.21.0
pip install pyyaml>=6.0
pip install nuscenes-devkit>=1.1.9

# Verify GPU
python -c "import tensorflow as tf; print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')"
```

### 1.3 Data Preparation

```bash
# 1. Download nuScenes (requires registration at nuscenes.org)
mkdir -p data/nuscenes
# Download and extract v1.0-trainval to data/nuscenes/

# 2. Verify data structure
ls data/nuscenes/
# Expected: maps/ samples/ sweeps/ v1.0-trainval/

# 3. Generate occupancy ground truth from LiDAR
python tools/generate_occupancy_gt.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --dataroot data/nuscenes \
    --output_dir data/nuscenes/occupancy_gt \
    --num_sweeps 10 \
    --split train

python tools/generate_occupancy_gt.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --dataroot data/nuscenes \
    --output_dir data/nuscenes/occupancy_gt \
    --num_sweeps 10 \
    --split val

# 4. Verify GT generation
python -c "
import numpy as np
gt = np.load('data/nuscenes/occupancy_gt/train/sample_000001.npz')
print(f'Shape: {gt[\"occupancy\"].shape}')
print(f'Labels: free={np.sum(gt[\"occupancy\"]==0)}, occ={np.sum(gt[\"occupancy\"]==1)}, unk={np.sum(gt[\"occupancy\"]==2)}')
"
```

---

## 2. Configuration File Explanation

The training configuration lives in `configs/radar_occupancy_nuscenes.yaml`. Here is each section explained:

### 2.1 Dataset Configuration

```yaml
dataset:
  name: nuscenes
  version: v1.0-trainval
  root: data/nuscenes
  radar_sensors:              # All 5 radar sensors used
    - RADAR_FRONT
    - RADAR_FRONT_LEFT
    - RADAR_FRONT_RIGHT
    - RADAR_BACK_LEFT
    - RADAR_BACK_RIGHT
  num_sweeps: 6              # Accumulate 6 sweeps per sample (~0.5s of data)
  max_points_per_sweep: 300  # Cap points per sweep (memory management)
  min_rcs: -5.0              # Filter weak detections below this RCS (dBsm)
  remove_invalid: true       # Remove detections flagged as invalid by firmware
```

### 2.2 Grid Configuration

```yaml
grid:
  x_range: [-50.0, 50.0]    # 100m total coverage in X
  y_range: [-50.0, 50.0]    # 100m total coverage in Y
  z_range: [-3.0, 5.0]      # Height range to include
  cell_size: 0.5             # Each cell is 0.5m x 0.5m
  grid_size: [200, 200]     # 200x200 = 40,000 cells total
```

### 2.3 Model Configuration

```yaml
model:
  type: temporal_pillar_occ_net  # Options: classical_ism | pillar_occ_net | temporal_pillar_occ_net
  
  pillar:
    max_points_per_pillar: 20    # Max radar detections per grid cell
    max_pillars: 12000           # Max non-empty cells (radar is sparse, rarely >2000)
    input_features: 9            # [x, y, z, rcs, vr, dt, x_center, y_center, z_center]
    pillar_features: 64          # Output dimension of pillar PointNet
  
  backbone:
    encoder_channels: [64, 128, 256, 512]  # U-Net encoder channel progression
    decoder_channels: [256, 128, 64]       # U-Net decoder (with skip connections)
    use_skip_connections: true
    norm_type: batch_norm
    activation: relu
  
  temporal:
    num_frames: 5                # Current frame + 4 past frames
    fusion_method: concat_conv   # Options: concat_conv | attention | gru
    temporal_conv_channels: 64   # Output channels after temporal fusion
  
  heads:
    occupancy:
      num_classes: 1             # Binary: occupied or not
      threshold: 0.5             # Inference threshold
    semantics:
      num_classes: 5             # Free, Vehicle, Pedestrian, Barrier, Other
      enabled: true              # Set false to train occupancy-only
```

### 2.4 Training Configuration

```yaml
training:
  batch_size: 16              # Per-GPU batch size
  num_epochs: 50              # Total training epochs
  optimizer:
    type: adamw               # AdamW with decoupled weight decay
    lr: 0.001                 # Base learning rate
    weight_decay: 0.01        # L2 regularization
    betas: [0.9, 0.999]      # Adam momentum parameters
  scheduler:
    type: cosine              # Cosine annealing with warmup
    min_lr: 0.00001           # Minimum LR at end of training
    warmup_epochs: 5          # Linear warmup for first 5 epochs
  loss:
    occupancy_weight: 1.0     # Weight for binary occupancy loss
    semantic_weight: 0.5      # Weight for semantic segmentation loss
    focal_alpha: 0.75         # Focal loss: weight for positive (occupied) class
    focal_gamma: 2.0          # Focal loss: focusing parameter
    class_weights: [1.0, 5.0, 10.0, 8.0, 3.0]  # Per-class weights for semantics
  augmentation:
    random_flip: true         # Horizontal flip (50% probability)
    random_rotate: [-45.0, 45.0]  # Random rotation range (degrees)
    random_scale: [0.9, 1.1]      # Random scale range
    point_dropout: 0.1             # Random point removal probability
```

---

## 3. Training Commands

### 3.1 PyTorch Training

**Single-frame model (PillarOccNet):**

```bash
python pytorch/train.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --output_dir outputs/pillar_occ_net
```

**Temporal model (TemporalPillarOccNet):**

```bash
python pytorch/train.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --output_dir outputs/temporal_occ_net
```

**Resume from checkpoint:**

```bash
python pytorch/train.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --output_dir outputs/temporal_occ_net \
    --resume outputs/temporal_occ_net/latest.pth
```

### 3.2 TensorFlow Training

```bash
python tensorflow/train.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --output_dir outputs/tf_radar_occ

# Resume training
python tensorflow/train.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --output_dir outputs/tf_radar_occ \
    --resume outputs/tf_radar_occ/checkpoints
```

### 3.3 Quick Validation Run (Mini Dataset)

```bash
# Use the nuScenes mini split for quick testing
python pytorch/train.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --output_dir outputs/test_run

# Override config values for a quick test:
# Edit config to use: version: v1.0-mini, num_epochs: 3, batch_size: 4
```

---

## 4. Multi-GPU Training

### 4.1 PyTorch Distributed Data Parallel (DDP)

```bash
# Single-node, 4 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    pytorch/train.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --output_dir outputs/multi_gpu_run

# Using torchrun (PyTorch 1.10+, recommended)
torchrun --nproc_per_node=4 \
    pytorch/train.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --output_dir outputs/multi_gpu_run
```

### 4.2 Batch Size Scaling

When using multiple GPUs, scale the learning rate linearly with effective batch size:

```
Effective batch size = per_gpu_batch_size * num_gpus
Learning rate = base_lr * (effective_batch_size / reference_batch_size)

Example:
  1 GPU, batch_size=16, lr=0.001 (reference)
  4 GPUs, batch_size=16, lr=0.004 (4x scaling)
  
  Or keep lr=0.001 and use batch_size=4 per GPU for same effective batch.
```

### 4.3 TensorFlow Multi-GPU (MirroredStrategy)

```python
# The TF training script handles multi-GPU automatically via MirroredStrategy
# Just ensure multiple GPUs are visible:
CUDA_VISIBLE_DEVICES=0,1,2,3 python tensorflow/train.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --output_dir outputs/tf_multi_gpu
```

---

## 5. Hyperparameter Tuning Recommendations

### 5.1 Most Impactful Hyperparameters (Ranked)

| Rank | Hyperparameter | Default | Tuning Range | Impact |
|------|---------------|---------|-------------|--------|
| 1 | num_frames (temporal) | 5 | 1-10 | High: more frames = better, diminishing returns after 5 |
| 2 | learning_rate | 0.001 | 0.0003-0.003 | High: too high diverges, too low converges slowly |
| 3 | focal_alpha | 0.75 | 0.5-0.9 | Medium: balances occupied vs free class |
| 4 | num_sweeps (input) | 6 | 1-12 | Medium: more sweeps = denser input |
| 5 | cell_size | 0.5 | 0.25-1.0 | Medium: finer = more detail but more computation |
| 6 | focal_gamma | 2.0 | 1.0-3.0 | Low-Medium: focuses on hard examples |
| 7 | weight_decay | 0.01 | 0.001-0.05 | Low: prevents overfitting |
| 8 | batch_size | 16 | 8-32 | Low: larger is slightly better if GPU allows |

### 5.2 Tuning Strategy

**Phase 1: Find stable learning rate (1-2 hours)**

```bash
# Try 3 learning rates
for lr in 0.0003 0.001 0.003; do
    # Modify config and train for 5 epochs
    python pytorch/train.py \
        --config configs/radar_occupancy_nuscenes.yaml \
        --output_dir outputs/lr_sweep_${lr}
done
# Pick the LR where loss decreases fastest without instability
```

**Phase 2: Tune temporal depth (4-8 hours)**

```bash
# Compare temporal frame counts
for nf in 1 3 5 7; do
    python pytorch/train.py \
        --config configs/radar_occupancy_nframes_${nf}.yaml \
        --output_dir outputs/temporal_sweep_${nf}
done
```

**Phase 3: Tune loss balance (4-8 hours)**

```bash
# Adjust focal_alpha for occupancy class balance
# Higher alpha = more weight on occupied class
# If occupied_recall is low: increase alpha (e.g., 0.8-0.9)
# If occupied_precision is low: decrease alpha (e.g., 0.5-0.6)
```

### 5.3 Temporal Fusion Method Selection

| Method | mIoU | Memory | Speed | When to Use |
|--------|------|--------|-------|-------------|
| concat_conv | 0.685 | High (C*T channels) | Fast | Default choice, good balance |
| attention | 0.695 | Very High | Slow | Best accuracy, research setting |
| gru | 0.680 | Low (recurrent state) | Medium | Deployment (low memory) |

---

## 6. Monitoring with TensorBoard

### 6.1 Launch TensorBoard

```bash
# PyTorch training logs
tensorboard --logdir outputs/temporal_occ_net --port 6006

# TensorFlow training logs
tensorboard --logdir outputs/tf_radar_occ/logs --port 6006

# Compare multiple runs
tensorboard --logdir outputs/ --port 6006
```

### 6.2 Key Metrics to Monitor

| Metric | Healthy Pattern | Warning Sign |
|--------|----------------|-------------|
| train/loss | Smooth decrease, 2.0 -> 0.3 | Spikes, NaN, plateau before epoch 10 |
| val/loss | Decreasing, slightly above train | Increasing (overfitting) |
| train/occ_iou | Increasing, 0.2 -> 0.5+ | Stuck below 0.15 after 5 epochs |
| val/mean_iou | Increasing, should plateau | Decreasing after initial rise |
| learning_rate | Matches schedule curve | Unexpected jumps or zeros |
| grad_norm | 0.1 - 10.0 | > 50 (exploding), < 0.01 (vanishing) |

### 6.3 TensorBoard Scalars for This Model

The PyTorch training script logs every 50 batches:
- `Loss` (total, occupancy component, semantic component)
- `OccIoU` (training occupied IoU)
- `FreeIoU` (training free IoU)
- Validation metrics at end of each epoch

The TensorFlow script logs:
- `loss/total`, `loss/occupancy`, `loss/semantic`
- `metrics/miou`, `metrics/occupied_iou`, `metrics/free_iou`
- `lr` (current learning rate)
- `grad_norm` (gradient magnitude)

---

## 7. Common Issues and Solutions

### 7.1 Training Fails to Start

| Issue | Symptom | Solution |
|-------|---------|----------|
| CUDA OOM | `RuntimeError: CUDA out of memory` | Reduce `batch_size` to 8 or 4 |
| Data not found | `FileNotFoundError` on nuScenes path | Verify `dataset.root` in config |
| GT not generated | `KeyError: 'occupancy_gt'` | Run GT generation script first |
| Wrong CUDA version | `torch.cuda.is_available() = False` | Reinstall PyTorch with correct CUDA |

### 7.2 Training Instability

| Issue | Symptom | Solution |
|-------|---------|----------|
| Loss explodes | NaN or very large values | Reduce `lr` by 3-5x, reduce `clip_grad_norm` to 5.0 |
| Loss plateaus | No improvement after 10 epochs | Increase `lr` by 2x, check data pipeline |
| Oscillating loss | Loss bounces up and down | Reduce `lr`, increase `batch_size` |
| Overfitting | Val loss increases while train decreases | Add augmentation, increase `weight_decay` |

### 7.3 Poor Final Performance

| Issue | Symptom | Solution |
|-------|---------|----------|
| Low occupied IoU | Below 0.40 on val | Increase `focal_alpha`, add more sweeps |
| Low free IoU | Below 0.70 on val | Decrease `focal_alpha`, check GT quality |
| Semantic mIoU low | Per-class IoUs very unbalanced | Adjust `class_weights`, add GT-paste augmentation |
| Good train, bad val | Large generalization gap | More augmentation, reduce model capacity |

### 7.4 Data Pipeline Issues

```python
# Debug: check a single batch from the dataloader
from dataset import RadarOccupancyDataset
from torch.utils.data import DataLoader
import yaml

with open("configs/radar_occupancy_nuscenes.yaml") as f:
    config = yaml.safe_load(f)

dataset = RadarOccupancyDataset(config, split="train")
loader = DataLoader(dataset, batch_size=2, shuffle=False)
batch = next(iter(loader))

print(f"Pillar features shape: {batch['pillar_features'].shape}")
print(f"Pillar indices shape: {batch['pillar_indices'].shape}")
print(f"Num pillars: {batch['num_pillars']}")
print(f"Occupancy GT shape: {batch['occupancy_gt'].shape}")
print(f"GT unique values: {batch['occupancy_gt'].unique()}")
print(f"Occupied ratio: {(batch['occupancy_gt']==1).float().mean():.4f}")
```

---

## 8. Training Timeline and Checkpointing

### 8.1 Expected Training Duration

| Configuration | Hardware | Epochs | Time |
|--------------|----------|--------|------|
| PillarOccNet, batch=16 | 1x RTX 3090 | 50 | ~8 hours |
| TemporalPillarOccNet, batch=16 | 1x RTX 3090 | 50 | ~14 hours |
| TemporalPillarOccNet, batch=16 | 4x RTX 3090 | 50 | ~4 hours |
| TemporalPillarOccNet, batch=16 | 1x A100 | 50 | ~10 hours |

### 8.2 Checkpoint Structure

```
outputs/radar_occ/
├── latest.pth          # Most recent epoch checkpoint
├── best.pth            # Best validation mIoU checkpoint
└── config.yaml         # Copy of training config (for reproducibility)

# Each checkpoint contains:
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "train_metrics": {"loss": float, "occ_iou": float, "free_iou": float},
    "val_metrics": {"loss": float, "occ_iou": float, "free_iou": float},
    "best_iou": float,
    "config": dict
}
```

### 8.3 When to Stop Training

Training typically converges by epoch 40-50. Signs that training is complete:

1. Validation mIoU has not improved for 10 consecutive epochs
2. Learning rate has decayed below `min_lr`
3. Training and validation IoU curves have both flattened

Use the `best.pth` checkpoint for final evaluation, not `latest.pth`.

---

## 9. Reproducibility Checklist

To reproduce results exactly:

- [ ] Set random seed: `torch.manual_seed(42); np.random.seed(42)`
- [ ] Use deterministic operations: `torch.use_deterministic_algorithms(True)`
- [ ] Pin exact package versions in `requirements.txt`
- [ ] Use same number of GPUs (DDP behavior changes with GPU count)
- [ ] Same data split (nuScenes v1.0-trainval has fixed train/val split)
- [ ] Same GT generation parameters (num_sweeps, height filter)
- [ ] Log and save the full config YAML with each experiment
