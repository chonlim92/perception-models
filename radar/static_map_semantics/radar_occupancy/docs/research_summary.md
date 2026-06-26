# Radar Occupancy Grid Mapping — Research Summary

## Motivation

Automotive radar sensors provide range measurements that are:
- Weather-independent (works in rain, fog, snow, dust)
- Direct range measurement (no monocular depth estimation uncertainty)
- Low-cost and reliable for mass production
- Provide radial velocity (Doppler) information

However, radar has challenges:
- Very sparse returns (~100-300 detections per frame vs 30,000+ for LiDAR)
- Low angular resolution (especially in elevation)
- Multipath reflections and clutter
- Extended targets (single object produces multiple returns)

Occupancy grid mapping overcomes sparsity through **temporal accumulation** — building up a dense spatial representation over time by fusing multiple frames.

## Approaches

### 1. Classical Inverse Sensor Model (ISM) + Bayesian Fusion

**Concept:** For each radar detection, model the probability of occupancy along the measurement ray:
- Free space exists between sensor and detection (high P(free))
- Occupied space at the detection range (high P(occupied))
- Unknown beyond the detection (P = 0.5, no information)

**Temporal fusion** via log-odds:
```
L(occ | z_{1:t}) = L(occ | z_t) + L(occ | z_{1:t-1}) - L_0
```
Where L = log(P/(1-P)) is the log-odds representation.

**Advantages:** No training required, interpretable, well-understood theoretically.
**Limitations:** Hand-tuned sensor model parameters, doesn't learn from data.

### 2. Neural Occupancy Prediction (Learned)

**Concept:** Train a neural network to predict dense occupancy from sparse radar input:
- Encode radar points into BEV pillar features (like RadarPillarNet)
- Process with 2D CNN to fill in gaps between sparse measurements
- Predict per-cell: occupancy probability + optional semantic class

**Architecture:**
```
Radar Points → Pillar Encoding → BEV Pseudo-Image →
2D ResNet Encoder-Decoder → Occupancy + Semantics
```

**Temporal variant:** Concatenate/fuse BEV features from past N frames (ego-motion compensated) before prediction.

**Advantages:** Learns radar-specific patterns, handles clutter better, can predict semantics.
**Limitations:** Requires labeled training data, may not generalize to unseen scenarios.

## Key References

1. Werber et al., "Automotive Radar Gridmap Representations" (2015) — Classical ISM for automotive radar
2. Prophet et al., "Semantic Segmentation of Radar Occupancy Grids" (2019) — CNN on accumulated radar grids
3. Krämer et al., "Deep Radar Occupancy Grids" (2020) — End-to-end learned radar occupancy
4. Sless et al., "Road Scene Understanding by Occupancy Grid" (2019) — Temporal fusion in BEV
5. Scheiner et al., "Object Detection for Automotive Radar" (2020) — Radar point cloud processing fundamentals

## Comparison to LiDAR Occupancy

| Aspect | Radar Occupancy | LiDAR Occupancy |
|--------|----------------|-----------------|
| Density per frame | ~100-300 points | ~30,000-300,000 points |
| Weather robustness | Excellent | Degrades in rain/fog |
| Range accuracy | ±0.1m | ±0.02m |
| Angular resolution | ~1-2° azimuth | ~0.1° |
| Velocity info | Yes (Doppler) | No (requires multi-frame) |
| Cost | Low ($50-200) | High ($1,000-75,000) |
| Temporal need | Critical (sparse) | Optional (already dense) |

## Our Implementation

We provide both classical and neural approaches:

1. **Classical ISM:** Configurable cone-shaped sensor model, log-odds Bayesian fusion, temporal accumulation over N frames
2. **Neural (PillarOccNet):** Pillar encoding → 2D U-Net → occupancy + semantics, with optional temporal concatenation
3. **Hybrid:** Initialize with classical ISM, refine with neural network (best of both worlds)
