# Training Guide: DETR3D

## Overview

DETR3D follows the DETR-style set prediction training paradigm. Training involves bipartite matching between predictions and ground truth, combined with classification and regression losses. This guide covers the complete training configuration, loss functions, optimization strategy, and data augmentation pipeline.

---

## Set Prediction Training

### Formulation
DETR3D outputs a fixed set of N=900 predictions per sample. During training, each prediction must be assigned to either a ground-truth object or the "no object" class. This assignment is determined by Hungarian matching.

### Why Set Prediction?
- **No NMS needed:** One-to-one assignment prevents duplicate detections by design
- **End-to-end training:** No hand-crafted post-processing components that break gradient flow
- **Global reasoning:** The matching considers all predictions and all ground-truth objects simultaneously
- **Permutation invariance:** The loss is invariant to the ordering of predictions

---

## Hungarian Matching

### Algorithm
At each training step, the Hungarian algorithm finds the optimal bipartite matching between the set of N predictions and the set of M ground-truth objects (where M << N, typically M < 50):

```
optimal_matching = Hungarian(cost_matrix)
```

Where `cost_matrix` is an N x M matrix with each element representing the cost of assigning prediction i to ground-truth j.

### Matching Cost
The matching cost for assigning prediction i to ground-truth object j:

```
C(i, j) = lambda_cls * C_cls(i, j) + lambda_L1 * C_L1(i, j)
```

Where:
- **Classification cost:** `C_cls(i, j) = -p_i(c_j)` (negative predicted probability for the true class)
- **L1 regression cost:** `C_L1(i, j) = ||bbox_pred_i - bbox_gt_j||_1` (L1 distance between predicted and ground-truth box parameters)
- **Weights:** `lambda_cls = 2.0`, `lambda_L1 = 0.25` (typical values)

### Matching Details
- Only the top (non-background) class probability is used for matching cost
- The L1 cost uses normalized box parameters (center divided by detection range, log-scale dimensions)
- Unmatched predictions (N - M predictions) are assigned to the "no object" background class
- Matching is computed independently per sample in the batch
- Matching is performed **without gradients** (used only to determine assignment)

---

## Loss Functions

### Total Loss
```
L_total = L_cls + lambda_bbox * L_bbox + sum(L_aux_l for l in decoder_layers[:-1])
```

### Classification Loss: Focal Loss
```
L_cls = FocalLoss(pred_cls, target_cls)
```

- **Focal loss formula:** `FL(p) = -alpha * (1 - p)^gamma * log(p)`
- **Parameters:**
  - `alpha = 0.25` (class balance weight)
  - `gamma = 2.0` (focusing parameter to down-weight easy examples)
- **Applied to:** All 900 predictions (matched get their GT class, unmatched get background)
- **Motivation:** Addresses the extreme class imbalance between foreground (few matched) and background (many unmatched) predictions
- **Weight:** `lambda_cls = 2.0`

### Bounding Box Regression Loss: L1 Loss
```
L_bbox = L1Loss(pred_bbox, target_bbox)
```

- **Applied to:** Only matched predictions (foreground objects)
- **Targets:** Normalized box parameters:
  - Center: `(cx - x_min) / (x_max - x_min)` (normalized to [0, 1])
  - Dimensions: `log(w)`, `log(l)`, `log(h)`
  - Orientation: `sin(yaw)`, `cos(yaw)` (avoids angle wrapping issues)
  - Velocity: `vx`, `vy` (in m/s, directly supervised)
- **Weight:** `lambda_bbox = 0.25`
- **Note:** Unlike DETR for 2D detection, no GIoU loss is used (3D IoU is computationally expensive)

### Auxiliary Losses
- **Applied at:** Every intermediate decoder layer (layers 1-5), not just the final layer
- **Same loss functions:** Focal loss + L1 loss with the same weights
- **Independent matching:** Hungarian matching is performed independently at each layer
- **Purpose:** Provides gradient signal to early decoder layers, accelerating convergence
- **Auxiliary loss weight:** 1.0 (same as final layer, no decay)

### Loss Weights Summary

| Loss Component | Weight | Applied To |
|----------------|--------|-----------|
| Focal classification loss | 2.0 | All predictions |
| L1 bbox regression loss | 0.25 | Matched predictions only |
| Auxiliary losses (per layer) | 1.0 x (same as above) | Each decoder layer independently |

---

## Optimizer Configuration

### AdamW Optimizer
```python
optimizer = AdamW(
    params=model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### Parameter Group Configuration
Different learning rates for different components:

| Parameter Group | Learning Rate | Weight Decay |
|-----------------|--------------|--------------|
| Backbone (ResNet-101) | 2e-5 (0.1x base) | 0.01 |
| FPN | 2e-4 (base) | 0.01 |
| Transformer decoder | 2e-4 (base) | 0.01 |
| Detection heads | 2e-4 (base) | 0.01 |
| Query embeddings | 2e-4 (base) | 0.0 (no decay) |
| Bias terms | 2e-4 (base) | 0.0 (no decay) |

### Why AdamW?
- Decoupled weight decay regularization (proper L2 regularization independent of learning rate)
- Well-suited for transformer architectures
- Stable training with diverse parameter scales across backbone and transformer

---

## Learning Rate Schedule

### Cosine Annealing
```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_epochs,  # 24 epochs
    eta_min=2e-7         # minimum LR (1/1000 of base)
)
```

### Schedule Characteristics
- **Warmup:** Linear warmup from 0 to base LR over first 500 iterations
- **Decay:** Smooth cosine decay from base LR to minimum LR over remaining training
- **No restarts:** Single cosine cycle across all training epochs
- **Final LR:** ~2e-7 (1000x smaller than peak)

### Learning Rate Curve
```
LR
2e-4 |    /\
     |   /  \
     |  /    \
     | /      \___
     |/           \__
2e-7 |________________\___
     0    6   12  18  24  epochs
      warmup  cosine annealing
```

---

## Training Configuration

### Batch Size and Hardware
- **Batch size:** 1 sample per GPU (each sample = 6 camera images)
- **Number of GPUs:** 8 (NVIDIA V100 32GB or A100 40GB)
- **Effective batch size:** 8
- **Gradient accumulation:** Not typically used (batch of 8 is sufficient)
- **Mixed precision:** FP16 with dynamic loss scaling (reduces memory, enables larger resolution)

### Training Duration
- **Epochs:** 24 (standard), 36 (extended for higher performance)
- **Iterations per epoch:** ~3,500 (28,130 keyframes / 8 effective batch size)
- **Total iterations:** ~84,000 (24 epochs)
- **Training time:** ~40-48 hours on 8x V100 GPUs

### Checkpoint Strategy
- Save checkpoint every epoch
- Best model selected by NDS on validation set
- Last 5 checkpoints retained for model selection

---

## Data Augmentation

### Spatial Augmentations

#### Random Horizontal Flip
- **Probability:** 0.5
- **Applied to:** All 6 camera images simultaneously
- **Box adjustment:** Mirror X-coordinate of box centers, negate yaw angle
- **Camera adjustment:** Update camera extrinsics to reflect the flip

#### Random Scale (Resize)
- **Scale range:** [0.95, 1.05] (conservative) or [0.85, 1.15] (aggressive)
- **Applied to:** Image resolution (affects intrinsic focal length proportionally)
- **Box adjustment:** Not needed (3D annotations remain in physical coordinates)

#### Random Rotation (BEV)
- **Rotation range:** [-22.5, 22.5] degrees around the Z-axis (up)
- **Applied to:** All annotations and camera extrinsics in ego frame
- **Effect:** Simulates the ego vehicle approaching the scene from a slightly different angle

### Photometric Augmentations

#### Color Jitter
- Brightness: +/- 0.2
- Contrast: +/- 0.2
- Saturation: +/- 0.2
- Hue: +/- 0.1
- Applied independently to each camera image

#### Random Erasing (GridMask)
- Probability: 0.3
- Erases random rectangular patches in images
- Simulates occlusion and forces the model to use multiple views

### Temporal Augmentations
- **Random frame skip:** Occasionally skip a keyframe to vary temporal spacing
- **Copy-paste (advanced):** Paste 3D objects from other scenes (less common for DETR3D)

---

## Class-Balanced Grouping and Sampling (CBGS)

### Motivation
nuScenes has severe class imbalance (e.g., ~400K cars vs. ~8K bicycles). Without balancing, the model overwhelmingly learns to detect cars and performs poorly on rare classes.

### CBGS Strategy
1. **Grouping:** For each sample, identify which rare classes are present
2. **Oversampling:** Samples containing rare classes are duplicated/oversampled to balance class frequency
3. **Effective epochs:** With CBGS, one "epoch" sees each sample multiple times for rare-class scenes
4. **Sampling weights:** Inversely proportional to class frequency

### CBGS Implementation
```python
class CBGSDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.sample_indices = self._compute_balanced_indices()

    def _compute_balanced_indices(self):
        # For each class, find all samples containing that class
        # Oversample rare class samples to match the most common class frequency
        cls_to_samples = defaultdict(list)
        for idx, sample in enumerate(self.dataset):
            for cls in sample.classes_present:
                cls_to_samples[cls].append(idx)

        # Balance: repeat rare-class samples
        max_count = max(len(v) for v in cls_to_samples.values())
        balanced_indices = []
        for cls, indices in cls_to_samples.items():
            repeat_factor = max_count // len(indices)
            balanced_indices.extend(indices * repeat_factor)

        return balanced_indices
```

### CBGS Impact on Performance
| Setting | NDS | mAP | Training Time |
|---------|-----|-----|---------------|
| Without CBGS | 0.425 | 0.346 | ~40 hours |
| With CBGS | 0.479 | 0.412 | ~60 hours (more iterations per epoch) |

CBGS provides +5.4 NDS and +6.6 mAP improvement, making it essential for competitive results.

---

## Training Pipeline

### Step-by-Step Training Flow

```
1. Load batch of samples (each = 6 multi-view images + annotations)
2. Apply data augmentation (flip, scale, rotate, color jitter)
3. Forward pass:
   a. Extract multi-scale features via backbone + FPN (6 images)
   b. Initialize object queries and reference points
   c. Run transformer decoder (6 layers)
   d. Compute predictions at each layer via detection heads
4. Compute losses:
   a. For each decoder layer:
      - Run Hungarian matching (predictions vs. ground truth)
      - Compute focal loss (classification)
      - Compute L1 loss (regression, matched pairs only)
   b. Sum all layer losses
5. Backward pass + gradient clipping (max_norm=35.0)
6. Optimizer step + LR scheduler step
7. Log metrics (loss, learning rate)
```

### Gradient Clipping
- **Method:** Gradient norm clipping
- **Max norm:** 35.0
- **Purpose:** Prevents training instability from large gradients in early training or on rare difficult samples

---

## Training Tips and Common Issues

### Convergence
- DETR3D typically requires the full 24 epochs to converge (unlike CNN detectors that plateau earlier)
- NDS improves steadily throughout training; do not stop early based on apparent plateaus
- Validation NDS usually peaks at epoch 22-24

### Common Failure Modes
| Issue | Symptom | Solution |
|-------|---------|----------|
| Divergence | Loss explodes in first 100 iterations | Reduce LR, increase warmup, check data loading |
| All background | Model predicts "no object" for everything | Reduce focal loss gamma, check matching cost weights |
| Duplicate detections | Same object detected multiple times | Increase self-attention layers, check query count |
| Poor rare-class performance | High mAP on car, low on bicycle | Enable CBGS, increase training epochs |
| Poor localization | High mATE / mAOE | Check camera calibration, verify coordinate transforms |

### Initialization
- **Backbone:** ImageNet pre-trained (required; random init does not converge well)
- **FPN:** Random initialization with Kaiming uniform
- **Transformer decoder:** Xavier uniform initialization
- **Detection heads:** Bias initialized for background prior (prevents early training instability):
  ```python
  # Initialize classification bias to predict "no object" initially
  nn.init.constant_(cls_head.bias, -4.6)  # sigmoid(-4.6) ≈ 0.01
  ```
- **Query embeddings:** Normal distribution (mean=0, std=1.0)

### Memory Optimization
- **Gradient checkpointing:** Enable for backbone to trade compute for memory (saves ~30% GPU memory)
- **FP16 training:** Use PyTorch AMP for 2x memory reduction with minimal accuracy impact
- **Reduced image resolution:** Training at 448 x 800 instead of 900 x 1600 speeds up 4x with ~2 NDS loss

---

## Distributed Training Setup

### Configuration (8 GPU)
```bash
# Launch distributed training with PyTorch DDP
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    tools/train.py \
    configs/detr3d/detr3d_res101_gridmask.py \
    --launcher pytorch \
    --work-dir work_dirs/detr3d_r101_cbgs_24e
```

### Distributed Settings
- **Backend:** NCCL
- **Sync batch norm:** Not used (batch size 1 per GPU makes BN less effective; uses group norm or layer norm in transformer)
- **Gradient synchronization:** Averaged across GPUs after each step
- **Data parallelism:** Each GPU processes a different sample

---

## Evaluation During Training

### Validation Schedule
- Evaluate on full validation set every 4 epochs (or every epoch in final phase)
- Primary metric: NDS (nuScenes Detection Score)
- Secondary metric: mAP

### Best Model Selection
- Track best NDS on validation set
- Save "best" checkpoint separately from periodic checkpoints
- Report final results using the best checkpoint (not the last epoch)
