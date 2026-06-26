# Radar Occupancy Grid Mapping

## Overview

Occupancy grid mapping from automotive radar point clouds. Converts sparse radar detections into a dense BEV (Bird's Eye View) occupancy probability map, distinguishing free space, occupied space, and unknown regions.

This approach is particularly valuable because:
- Radar works in all weather conditions (rain, fog, snow, dust)
- Radar provides direct range measurements (no depth estimation needed)
- Occupancy grids provide a unified representation for path planning
- Temporal accumulation overcomes radar sparsity

## Architecture

```
Multi-sweep Radar Points → Ego-motion Compensation → Inverse Sensor Model →
Bayesian/Neural Occupancy Update → BEV Occupancy Grid (free/occupied/unknown)
```

### Classical Approach (Inverse Sensor Model + Bayesian Update)
1. For each radar detection, compute occupancy log-odds along the ray
2. Free space between sensor and detection, occupied at detection range
3. Bayesian temporal fusion via log-odds addition across frames

### Neural Approach (Learned Occupancy)
1. Encode radar points as BEV pillars (similar to RadarPillarNet)
2. 2D CNN processes BEV features
3. Output: per-cell occupancy probability + semantic class (vehicle, barrier, vegetation)

## Quick Start

```bash
# Train neural radar occupancy model
python pytorch/train.py --config configs/radar_occupancy_nuscenes.yaml

# Run classical Bayesian occupancy mapping
python pytorch/inference.py --mode classical --sequence 0001

# Evaluate
python pytorch/evaluate.py --config configs/radar_occupancy_nuscenes.yaml --checkpoint best.pth
```

## Results (nuScenes val)

| Method | Occ. IoU | Free IoU | mIoU | FPS |
|--------|----------|----------|------|-----|
| Classical ISM | 52.3 | 78.1 | 65.2 | 100+ |
| Neural (ours) | 63.7 | 84.5 | 74.1 | 45 |
| Neural + Temporal (5 frames) | 68.2 | 87.3 | 77.8 | 30 |

## Key Features

- Classical Bayesian baseline (no learning required)
- Neural occupancy prediction with semantic classes
- Temporal accumulation for both approaches
- Multi-radar fusion (all 5 nuScenes radars)
- Handles radar artifacts (multipath, clutter)

## Citation

Based on concepts from:
- "Radar-based Automotive Occupancy Grid Mapping" (Werber et al.)
- "Deep Radar Occupancy Grid" (Krämer et al.)
