# MapTR: Training Guide

## Overview

Training MapTR involves a hierarchical matching strategy that jointly optimizes instance-level assignment and point-level correspondence. This guide covers the loss formulation, training schedule, optimization details, and practical multi-GPU training setup.

---

## Loss Architecture

### Hierarchical Matching Loss

MapTR's training objective operates at two levels, combining instance matching (which predicted query corresponds to which ground truth element) with point matching (which ordering of points within the matched pair gives the best correspondence).

```
Total Loss = λ_cls * L_cls + λ_pts * L_pts + λ_dir * L_dir + [λ_dense * L_dense]

Where:
  L_cls  = Classification loss (focal loss)
  L_pts  = Point-set loss (Chamfer distance or L1)
  L_dir  = Direction loss (ordering penalty)
  L_dense = Auxiliary dense prediction loss (MapTRv2 only)
```

---

## Step 1: Instance-Level Matching (Hungarian Algorithm)

### Matching Cost Matrix

For M predicted instances and G ground truth instances, compute an M x G cost matrix:

```python
def compute_matching_cost(pred_classes, pred_points, gt_classes, gt_points):
    """
    pred_classes: (M, num_classes+1) - classification logits
    pred_points: (M, N_pts, 2) - predicted point coordinates
    gt_classes: (G,) - ground truth class labels
    gt_points: (G, N_pts, 2) - ground truth point sequences
    """
    M, G = pred_classes.shape[0], gt_classes.shape[0]
    cost_matrix = torch.zeros(M, G)
    
    for i in range(M):
        for j in range(G):
            # Classification cost
            cls_cost = -pred_classes[i, gt_classes[j]]  # Negative log-prob of correct class
            
            # Point-set cost (minimum over equivalent permutations)
            pts_cost = compute_permutation_aware_distance(
                pred_points[i], gt_points[j], gt_type[j]
            )
            
            cost_matrix[i, j] = λ_cls_cost * cls_cost + λ_pts_cost * pts_cost
    
    return cost_matrix
```

### Permutation-Aware Distance

```python
def compute_permutation_aware_distance(pred_pts, gt_pts, element_type):
    """
    Compute minimum distance over all equivalent permutations of gt_pts.
    """
    if element_type == "polyline":
        # 2 equivalent orderings: forward and reverse
        dist_forward = chamfer_distance(pred_pts, gt_pts)
        dist_reverse = chamfer_distance(pred_pts, gt_pts.flip(0))
        return min(dist_forward, dist_reverse)
    
    elif element_type == "polygon":
        # 2N equivalent orderings: N starting points x 2 directions
        N = gt_pts.shape[0]
        min_dist = float('inf')
        
        for start in range(N):
            # Forward direction from this starting point
            shifted = torch.roll(gt_pts, -start, dims=0)
            dist = chamfer_distance(pred_pts, shifted)
            min_dist = min(min_dist, dist)
            
            # Reverse direction from this starting point
            reversed_shifted = shifted.flip(0)
            dist = chamfer_distance(pred_pts, reversed_shifted)
            min_dist = min(min_dist, dist)
        
        return min_dist
```

### Hungarian Algorithm

```python
from scipy.optimize import linear_sum_assignment

def hungarian_matching(cost_matrix):
    """
    Find optimal one-to-one assignment minimizing total cost.
    Returns: matched pairs (pred_idx, gt_idx)
    """
    row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
    return list(zip(row_indices, col_indices))
```

---

## Step 2: Point-Level Matching

After instance matching, for each matched pair (pred_i, gt_j), determine the optimal point ordering:

```python
def find_best_permutation(pred_pts, gt_pts, element_type):
    """
    Find the permutation of gt_pts that minimizes L1 distance to pred_pts.
    Returns the optimally ordered gt_pts.
    """
    if element_type == "polyline":
        dist_fwd = (pred_pts - gt_pts).abs().sum()
        dist_rev = (pred_pts - gt_pts.flip(0)).abs().sum()
        if dist_fwd <= dist_rev:
            return gt_pts, "forward"
        else:
            return gt_pts.flip(0), "reverse"
    
    elif element_type == "polygon":
        N = gt_pts.shape[0]
        best_dist = float('inf')
        best_perm = gt_pts
        
        for start in range(N):
            for direction in ["forward", "reverse"]:
                shifted = torch.roll(gt_pts, -start, dims=0)
                if direction == "reverse":
                    shifted = shifted.flip(0)
                
                dist = (pred_pts - shifted).abs().sum()
                if dist < best_dist:
                    best_dist = dist
                    best_perm = shifted
        
        return best_perm, best_dist
```

---

## Loss Functions

### Classification Loss: Focal Loss

```python
def focal_loss(pred_logits, gt_labels, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance (most queries are background).
    
    pred_logits: (M, num_classes+1) - raw logits
    gt_labels: (M,) - target labels (num_classes = background)
    """
    probs = torch.softmax(pred_logits, dim=-1)
    
    # For each prediction, get probability of target class
    pt = probs.gather(1, gt_labels.unsqueeze(1)).squeeze(1)
    
    # Focal weight: (1 - pt)^gamma
    focal_weight = (1 - pt) ** gamma
    
    # Alpha weighting for positive/negative
    alpha_t = torch.where(gt_labels < num_classes, alpha, 1 - alpha)
    
    loss = -alpha_t * focal_weight * torch.log(pt + 1e-8)
    return loss.mean()
```

**Parameters**:
- α = 0.25 (weight for foreground classes)
- γ = 2.0 (focusing parameter)
- Background class receives (1-α) weight

### Point-Set Loss: Chamfer Distance

The primary geometric loss measures the distance between predicted and matched ground truth point sets:

```python
def chamfer_distance(pred_pts, gt_pts):
    """
    Symmetric Chamfer distance between two point sets.
    
    pred_pts: (N_pts, 2)
    gt_pts: (N_pts, 2)
    """
    # Since points are already in correspondence (after permutation matching),
    # use ordered L1 distance
    return (pred_pts - gt_pts).abs().mean()
```

**Note**: In MapTR, because the permutation matching provides point-to-point correspondence, the "Chamfer distance" effectively reduces to ordered L1 distance between corresponding points after optimal permutation selection.

**Alternative**: Some configurations use the proper Chamfer distance (nearest-neighbor based) as:

```python
def unordered_chamfer_distance(pred_pts, gt_pts):
    """
    Proper Chamfer distance (nearest neighbor based).
    Used in matching cost; ordered L1 used in final loss.
    """
    # pred_to_gt: for each pred point, distance to nearest gt point
    dist_matrix = torch.cdist(pred_pts, gt_pts, p=1)  # (N, N) pairwise L1
    pred_to_gt = dist_matrix.min(dim=1)[0].mean()
    gt_to_pred = dist_matrix.min(dim=0)[0].mean()
    return (pred_to_gt + gt_to_pred) / 2
```

### Direction Loss

Penalizes predictions that have the correct point positions but reversed ordering:

```python
def direction_loss(pred_pts, gt_pts_ordered):
    """
    Encourages the predicted points to follow the same traversal direction
    as the (optimally permuted) ground truth.
    
    Computes cosine similarity between consecutive displacement vectors.
    """
    # Displacement vectors between consecutive points
    pred_displacements = pred_pts[1:] - pred_pts[:-1]  # (N-1, 2)
    gt_displacements = gt_pts_ordered[1:] - gt_pts_ordered[:-1]  # (N-1, 2)
    
    # Cosine similarity between corresponding displacements
    cos_sim = F.cosine_similarity(pred_displacements, gt_displacements, dim=-1)
    
    # Loss: penalize negative cosine similarity (opposite direction)
    loss = (1 - cos_sim).mean()
    return loss
```

### Auxiliary Dense Prediction Loss (MapTRv2)

```python
def dense_segmentation_loss(pred_seg, gt_seg):
    """
    Binary cross-entropy for auxiliary BEV segmentation head.
    
    pred_seg: (B, num_classes, H_bev, W_bev) - predicted segmentation
    gt_seg: (B, num_classes, H_bev, W_bev) - rasterized ground truth
    """
    loss = F.binary_cross_entropy_with_logits(pred_seg, gt_seg, reduction='mean')
    return loss
```

Ground truth rasterization:
- Each map element's polyline/polygon is rasterized onto the BEV grid
- Line thickness: 1-2 pixels for polylines
- Filled polygon for pedestrian crossings

---

## Loss Weights

| Loss Component | Symbol | Weight (MapTR) | Weight (MapTRv2) |
|---------------|--------|---------------|-----------------|
| Classification (focal) | λ_cls | 2.0 | 2.0 |
| Point regression (L1) | λ_pts | 5.0 | 5.0 |
| Direction loss | λ_dir | 0.005 | 0.005 |
| Dense segmentation | λ_dense | - | 2.0 |
| One-to-many cls | λ_o2m_cls | - | 2.0 |
| One-to-many pts | λ_o2m_pts | - | 5.0 |

### Intermediate Layer Losses

Loss is applied at every decoder layer output (not just the final layer):

```python
total_loss = 0
for layer_idx in range(num_decoder_layers):
    layer_loss = compute_loss(layer_predictions[layer_idx], ground_truth)
    total_loss += layer_loss  # Equal weight for each layer
```

---

## MapTRv2: Auxiliary One-to-Many Matching

### Motivation

Standard one-to-one Hungarian matching provides sparse supervision (each GT element supervises only 1 query). This leads to slow convergence, especially in early training when most queries are unmatched.

### Mechanism

```python
def one_to_many_matching(pred_classes, pred_points, gt_classes, gt_points, K=5):
    """
    Each ground truth element is matched to top-K predictions.
    Provides K times denser supervision.
    """
    M, G = pred_classes.shape[0], gt_classes.shape[0]
    cost_matrix = compute_matching_cost(pred_classes, pred_points, gt_classes, gt_points)
    
    matches = []
    for gt_idx in range(G):
        # Find top-K predictions with lowest cost for this GT
        costs_for_gt = cost_matrix[:, gt_idx]
        topk_pred_indices = costs_for_gt.topk(K, largest=False).indices
        
        for pred_idx in topk_pred_indices:
            matches.append((pred_idx.item(), gt_idx))
    
    return matches
```

### Training Integration

```python
# Primary loss: one-to-one matching (standard Hungarian)
primary_loss = compute_hungarian_loss(predictions, ground_truth)

# Auxiliary loss: one-to-many matching (K matches per GT)
aux_predictions = auxiliary_head(query_features)
aux_loss = compute_one_to_many_loss(aux_predictions, ground_truth, K=5)

# Combined
total_loss = primary_loss + λ_aux * aux_loss
```

**Key details**:
- The auxiliary one-to-many head shares the decoder features but has separate prediction heads
- Only the primary one-to-one head is used at inference
- Typically K = 5 or K = 6 auxiliary matches per GT element
- Auxiliary loss weight: 1.0 (same scale as primary)

---

## Training Schedule

### MapTR Standard (24 epochs)

```python
training_config = {
    "optimizer": "AdamW",
    "base_lr": 6e-4,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    
    "lr_scheduler": "CosineAnnealing",
    "warmup_epochs": 1,
    "warmup_lr": 1e-6,
    "min_lr": 1e-6,
    
    "total_epochs": 24,
    "batch_size_per_gpu": 4,
    "num_gpus": 8,
    "effective_batch_size": 32,
    
    "gradient_clip_norm": 35.0,
    "fp16": True,  # Mixed precision training
}
```

### MapTR Extended (110 epochs)

```python
training_config_extended = {
    **training_config,
    "total_epochs": 110,
    "lr_scheduler": "CosineAnnealing",
    "warmup_epochs": 5,
    # Same base_lr, decays more gradually
}
```

### MapTRv2 (24 epochs, faster convergence)

```python
training_config_v2 = {
    **training_config,
    "total_epochs": 24,
    # One-to-many matching accelerates convergence
    # Achieves similar performance to MapTR-110ep in just 24 epochs
}
```

### Learning Rate Schedule

```
LR
 ↑
6e-4 ─────┐
          │ \
          │  \  Cosine decay
          │   \
          │    \
1e-6 ─────│─────\────────────
          │      \___________
          └──────────────────→ Epochs
     0   1                  24
     ↑
  Warmup
```

---

## Multi-GPU Training Setup

### Hardware Requirements

| Configuration | GPUs | Memory per GPU | Training Time (24ep) |
|--------------|------|---------------|---------------------|
| Minimum | 4x RTX 3090 | 24 GB | ~20 hours |
| Standard | 8x A100 40GB | 40 GB | ~8 hours |
| Fast | 8x A100 80GB | 80 GB | ~6 hours |

### Distributed Training Launch

```bash
# Using PyTorch Distributed Data Parallel (DDP)
# 8 GPU setup on single node

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=29500

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    tools/train.py \
    --config configs/maptr_r50_24ep.py \
    --launcher pytorch \
    --work-dir work_dirs/maptr_r50_24ep
```

### Using torchrun (Recommended for PyTorch >= 1.10)

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    tools/train.py \
    --config configs/maptr_r50_24ep.py \
    --work-dir work_dirs/maptr_r50_24ep
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
    tools/train.py --config configs/maptr_r50_24ep.py

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<MASTER_IP> \
    --master_port=29500 \
    tools/train.py --config configs/maptr_r50_24ep.py
```

---

## Training Configuration Details

### Data Configuration

```python
data_config = {
    "dataset": "nuScenes",
    "version": "v1.0-trainval",
    "data_root": "data/nuscenes/",
    
    "train_split": "train",
    "val_split": "val",
    
    "input_size": (800, 480),
    "num_cameras": 6,
    
    "perception_range": [-30.0, 30.0, -15.0, 15.0],
    "num_points_per_element": 20,
    "max_elements_per_frame": 100,
    "map_classes": ["ped_crossing", "divider", "boundary"],
    
    "augmentation": {
        "random_flip": True,
        "flip_prob": 0.5,
        "random_resize": [0.8, 1.2],
        "color_jitter": True,
        "normalize": True,
    },
    
    "dataloader": {
        "batch_size": 4,          # Per GPU
        "num_workers": 4,         # Per GPU
        "pin_memory": True,
        "drop_last": True,
        "shuffle": True,
    }
}
```

### Backbone Initialization

```python
backbone_config = {
    "type": "ResNet",
    "depth": 50,
    "num_stages": 4,
    "out_indices": (1, 2, 3),      # C3, C4, C5
    "frozen_stages": 1,             # Freeze stem + stage1
    "norm_eval": True,
    "pretrained": "torchvision://resnet50",  # ImageNet pretrained
}
```

### Optimizer Configuration

```python
optimizer_config = {
    "type": "AdamW",
    "lr": 6e-4,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    
    # Per-parameter learning rate adjustments
    "paramwise_cfg": {
        "backbone": {"lr_mult": 0.1},        # Lower LR for pretrained backbone
        "neck": {"lr_mult": 0.5},
        "bev_encoder": {"lr_mult": 1.0},
        "decoder": {"lr_mult": 1.0},
        "heads": {"lr_mult": 1.0},
    }
}
```

---

## Training Monitoring

### Key Metrics to Track

| Metric | Expected Behavior | Concern If... |
|--------|-------------------|---------------|
| Total loss | Decreasing, smooth | Diverges or oscillates |
| Classification loss | Rapid decrease (epochs 1-3) | Stuck above 1.0 after epoch 5 |
| Point regression loss | Steady decrease | Plateaus very early |
| Direction loss | Gradual decrease | Increases after initial drop |
| Matching accuracy | Increasing (GT matched %) | Below 50% after epoch 10 |
| mAP (validation) | Steady improvement | No improvement for 5+ epochs |

### Logging Configuration

```python
log_config = {
    "interval": 50,           # Log every 50 iterations
    "hooks": [
        {"type": "TextLoggerHook"},
        {"type": "TensorboardLoggerHook"},
    ]
}

# Checkpoint saving
checkpoint_config = {
    "interval": 1,            # Save every epoch
    "max_keep_ckpts": 5,      # Keep latest 5
}

# Validation
evaluation = {
    "interval": 1,            # Validate every epoch
    "metric": "chamfer",
}
```

---

## Common Training Issues and Solutions

### Issue: Loss Divergence

**Symptoms**: Loss increases rapidly after a few iterations
**Solutions**:
- Reduce learning rate (try 3e-4 or 1e-4)
- Increase gradient clip norm
- Check for NaN in data loading (corrupted calibration matrices)
- Ensure correct normalization of coordinates to [0, 1]

### Issue: Slow Convergence

**Symptoms**: mAP improves very slowly, many unmatched queries
**Solutions**:
- Switch to MapTRv2 with one-to-many matching
- Increase number of instance queries (50 → 100)
- Verify backbone is pretrained (not random init)
- Check that augmentation is not too aggressive

### Issue: Poor Classification but Good Regression

**Symptoms**: Many false positives; predicted points are geometrically correct but assigned wrong class
**Solutions**:
- Increase classification loss weight
- Increase focal loss gamma (more focus on hard examples)
- Verify class balance in dataset; adjust alpha per class

### Issue: Good Classification but Poor Geometry

**Symptoms**: Correct detections but points are imprecise
**Solutions**:
- Increase point regression loss weight
- Increase number of decoder layers (6 → 8)
- Verify coordinate normalization is consistent between prediction and GT
- Check perception range configuration matches data preparation

### Issue: GPU Out of Memory

**Solutions**:
- Reduce batch size per GPU (4 → 2)
- Reduce input resolution (800x480 → 640x384)
- Use gradient checkpointing for backbone
- Reduce number of queries (50 → 30) or points (20 → 10)
- Enable mixed precision (FP16) training

---

## Ablation-Informed Training Tips

Based on ablation studies from the papers:

1. **Permutation equivalence is critical**: Removing it (forcing canonical ordering) drops mAP by ~5 points
2. **Hierarchical matching outperforms flat matching**: Joint point matching across all instances is worse
3. **Iterative refinement helps**: Each decoder layer improves mAP by ~0.5-1.0 points
4. **Backbone freezing**: Freezing first stage provides slight regularization benefit
5. **Point count sensitivity**: N_pts=20 is the sweet spot; N_pts=10 loses ~2 mAP, N_pts=50 gains <0.5 mAP
6. **Query count**: 50 queries sufficient for most scenes; 100 queries helps in dense urban areas but slows training

---

## Reproducibility Checklist

- [ ] Same random seed (default: 42)
- [ ] Same PyTorch version (>= 1.10)
- [ ] Same CUDA version (>= 11.3)
- [ ] Same cuDNN version (deterministic mode enabled)
- [ ] Same number of GPUs (affects batch norm statistics and effective batch size)
- [ ] Same data preprocessing pipeline
- [ ] ImageNet pretrained backbone weights match exactly
- [ ] SyncBatchNorm enabled for multi-GPU training
- [ ] Gradient clipping value matches config
- [ ] Same augmentation random seed per epoch
