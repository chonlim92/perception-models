# Model Architecture — Radar Occupancy Grid Mapping

## Overview

We implement three approaches with increasing complexity:

1. **Classical ISM** — No learning, Bayesian probability update
2. **PillarOccNet** — Neural network on radar pillars
3. **Temporal PillarOccNet** — Multi-frame neural fusion

## 1. Classical Inverse Sensor Model (ISM)

### Sensor Model

For each radar detection at range `r` with azimuth `θ`:

```
P(occupied | cell) =
  - 0.9  if cell is at detection location (within resolution cell)
  - 0.3  if cell is between sensor and detection (free ray)
  - 0.5  if cell is beyond detection (unknown)
```

The cone-shaped model accounts for:
- Range uncertainty: Gaussian spread σ_r = 0.2m at detection range
- Angular uncertainty: Gaussian spread σ_θ = 1.0° in azimuth

### Log-Odds Bayesian Update

Convert probability to log-odds for efficient temporal fusion:
```
L(x) = log(P(x) / (1 - P(x)))

L_t(cell) = L_{t-1}(cell) + L_sensor(cell) - L_prior
```

Where:
- L_prior = log(0.5/0.5) = 0 (uniform prior)
- Clamp to [-5, 5] to prevent overconfidence

### Update per frame:
```
For each radar detection (r_i, θ_i, rcs_i):
  1. Compute ray from sensor origin to detection
  2. For each cell along ray:
     - If before detection: update with L_free = log(0.3/0.7) = -0.847
     - If at detection: update with L_occ = log(0.9/0.1) = 2.197
     - Weight by RCS (stronger return = more confident)
```

## 2. PillarOccNet (Neural, Single Frame)

### Architecture

```
Input: Radar points (N × 6) [x, y, z, rcs, vr_comp, dt]
           ↓
┌─────────────────────────┐
│  Pillar Feature Net      │  (same as PointPillars)
│  - Voxelize to pillars   │  Pillar size: 0.5m × 0.5m
│  - PointNet per pillar   │  Max points per pillar: 20
│  - Output: C features    │  C = 64
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│  Scatter to BEV Grid     │  200 × 200 × 64
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│  2D U-Net Backbone       │
│  Encoder:                │
│    Block1: 64→128, /2    │  100 × 100
│    Block2: 128→256, /2   │  50 × 50
│    Block3: 256→512, /2   │  25 × 25
│  Decoder:                │
│    Up3: 512→256, ×2      │  50 × 50
│    Up2: 256→128, ×2      │  100 × 100
│    Up1: 128→64, ×2       │  200 × 200
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│  Output Heads            │
│  - Occupancy: 200×200×1  │  Sigmoid → P(occupied)
│  - Semantics: 200×200×K  │  Softmax → class probs
└─────────────────────────┘
```

### Dimensions

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Input pillars | (N, 20, 9) | — |
| Pillar PointNet | (N, 64) | 12K |
| BEV scatter | (200, 200, 64) | — |
| Encoder Block 1 | (100, 100, 128) | 148K |
| Encoder Block 2 | (50, 50, 256) | 590K |
| Encoder Block 3 | (25, 25, 512) | 2.4M |
| Decoder Block 3 | (50, 50, 256) | 1.8M |
| Decoder Block 2 | (100, 100, 128) | 442K |
| Decoder Block 1 | (200, 200, 64) | 111K |
| Occupancy head | (200, 200, 1) | 65 |
| Semantic head | (200, 200, 5) | 325 |
| **Total** | — | **~5.5M** |

### Each Encoder Block

```python
ConvBlock = [
    Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
    BatchNorm2d(out_ch),
    ReLU(),
    Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
    BatchNorm2d(out_ch),
    ReLU(),
]
```

### Each Decoder Block

```python
UpBlock = [
    ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),  # Upsample
    BatchNorm2d(out_ch),
    ReLU(),
    Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),  # Fuse skip
    BatchNorm2d(out_ch),
    ReLU(),
]
```

## 3. Temporal PillarOccNet (Multi-Frame)

### Temporal Extension

Add temporal context by concatenating BEV features from past T frames:

```
Frame t-T: Radar → Pillars → BEV features (ego-compensated)
Frame t-1: Radar → Pillars → BEV features (ego-compensated)
Frame t:   Radar → Pillars → BEV features
         ↓
Concatenate along channel dim: (200, 200, 64×(T+1))
         ↓
Temporal Conv: reduce to (200, 200, 64)
         ↓
Standard U-Net Backbone → Occupancy + Semantics
```

### Ego-Motion Compensation

Previous frames' BEV features are warped to current ego frame:
```python
# T_ego: 4x4 transformation from frame t-k to frame t
# For each cell (i, j) in current grid:
#   1. Compute world coordinate: (x_w, y_w) = cell_to_world(i, j)
#   2. Transform to past frame: (x_p, y_p) = inv(T_ego) @ (x_w, y_w)
#   3. Sample from past BEV: F_past[x_p, y_p] via bilinear interpolation
```

Uses `F.grid_sample` for differentiable warping.

### Temporal Attention (Advanced)

Alternative to concatenation — use cross-attention:
```
Q = current_bev_features  (200×200, 64)
K = past_bev_features     (200×200×T, 64)
V = past_bev_features     (200×200×T, 64)

Attention(Q, K, V) with spatial locality (local window)
```

## Loss Functions

### Occupancy Loss (Binary)
```
L_occ = FocalLoss(pred_occ, gt_occ, α=0.75, γ=2.0)
```
Focal loss to handle class imbalance (most cells are free/unknown).

### Semantic Loss
```
L_sem = CrossEntropy(pred_sem, gt_sem, weight=[1.0, 5.0, 10.0, 8.0, 3.0])
```
Class weights to balance rare classes (pedestrians, barriers).

### Total Loss
```
L = λ_occ × L_occ + λ_sem × L_sem
λ_occ = 1.0, λ_sem = 0.5
```

## Inference

1. Accumulate N radar sweeps (ego-compensated)
2. Run pillar encoding + U-Net forward pass
3. Threshold occupancy at P > 0.5
4. Apply semantic argmax for occupied cells
5. Optional: temporal smoothing with exponential moving average
