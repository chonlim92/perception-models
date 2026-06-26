# PETR / PETRv2 / StreamPETR - Research Summary

## Overview

The PETR family of models represents a paradigm shift in camera-based 3D object detection for autonomous driving. Unlike methods that explicitly construct Bird's-Eye-View (BEV) representations (BEVFormer, BEVDet) or project 3D reference points onto images (DETR3D), PETR introduces **3D Position Embedding (3D PE)** that implicitly encodes 3D spatial information into 2D image features, enabling a standard transformer decoder to reason about 3D space through global cross-attention.

---

## PETR: Position Embedding Transformation (ECCV 2022)

### Core Innovation: 3D Position Embedding

PETR's key insight is that instead of performing explicit geometric projection operations during attention (like DETR3D), you can **pre-encode 3D positional information into image features** so that a standard transformer decoder naturally learns 3D spatial relationships.

#### How 3D Position Embedding Works

1. **Frustum Point Generation**: For each camera, generate a 3D frustum grid by combining:
   - 2D pixel coordinates (u, v) from the image feature map
   - Discretized depth values (e.g., 64 bins from 1m to 61.2m)
   - Result: a set of 3D points in camera coordinate space

2. **Camera-to-World Transformation**: Transform frustum points from camera coordinates to 3D world (ego vehicle) coordinates using camera intrinsics and extrinsics:
   ```
   P_world = T_ego_cam * K^{-1} * [u*d, v*d, d, 1]^T
   ```
   where K is the camera intrinsic matrix, T_ego_cam is the camera-to-ego transformation, and d is the depth value.

3. **MLP Encoding**: Feed the 3D world coordinates (x, y, z) through a learned MLP:
   ```
   PE_3d = MLP(normalize(x, y, z))
   ```
   The MLP maps 3D coordinates to the same dimension as image features (typically 256-d).

4. **Feature Addition**: Add the 3D position embeddings to the corresponding image features:
   ```
   F_position_aware = F_image + PE_3d
   ```

5. **Global Cross-Attention**: Object queries attend to these position-aware features from ALL cameras simultaneously. Because spatial information is already encoded in the features, standard (non-deformable) attention suffices.

#### Why This Works

- The 3D PE creates a **continuous 3D coordinate field** overlaid on 2D features
- During cross-attention, a query looking for "a car at position (10, 5, 0)" will naturally attend to features that have been imbued with 3D PE values near (10, 5, 0)
- The model learns the correspondence between query positions and feature positions through the attention mechanism
- No explicit projection or sampling is needed at inference time

### Comparison to DETR3D

| Aspect | DETR3D | PETR |
|--------|--------|------|
| 3D-to-2D mapping | Explicit projection of 3D reference points onto images | Implicit via 3D PE added to features |
| Attention type | Sampling at projected 2D locations | Global attention over all features |
| Geometric operations | Required at every decoder layer | Only once during PE generation |
| Multi-camera handling | Project to each camera separately | Natural via global attention |
| Gradient flow | Through projection (can be unstable) | Direct through attention weights |

**Advantages of PETR over DETR3D:**
- Simpler architecture (no projection operations in attention)
- Better gradient flow (no geometric bottleneck)
- Naturally handles multi-camera overlap regions
- More robust to calibration errors

### Comparison to BEVFormer

| Aspect | BEVFormer | PETR |
|--------|-----------|------|
| Representation | Explicit BEV grid (200x200) | Implicit in position-aware features |
| Memory usage | High (BEV grid + temporal history) | Lower (no explicit BEV) |
| Temporal fusion | BEV-level alignment + attention | N/A (base PETR is single-frame) |
| Multi-task support | Natural (BEV supports segmentation) | Less natural without explicit BEV |
| Computation | Deformable attention + BEV construction | Global attention (can be expensive) |

**Key tradeoff:** BEVFormer's explicit BEV is more versatile for downstream tasks (segmentation, planning) but requires more memory and computation. PETR is simpler and can be faster for detection-only tasks.

### Performance (nuScenes val)

| Model | Backbone | mAP | NDS | FPS |
|-------|----------|-----|-----|-----|
| PETR | ResNet-50 | 31.3 | 38.1 | ~10 |
| PETR | ResNet-101 | 35.7 | 42.1 | ~8 |
| PETR | VoVNet-99 | 37.8 | 44.2 | ~7 |

---

## PETRv2: A Unified Framework (ICCV 2023)

### Extensions Over PETR

1. **Temporal Feature Alignment**
   - Transforms previous frame's 3D PE to current frame using ego-motion matrix
   - Previous frame features (with aligned 3D PE) are concatenated with current features
   - Object queries attend to temporally fused features in cross-attention
   - Enables velocity estimation and motion understanding

2. **2D + 3D Position Embedding**
   - Adds learnable 2D positional encoding alongside 3D PE
   - 2D PE helps retain fine-grained spatial information (texture, edges)
   - Combined encoding: `F = F_image + PE_3d + PE_2d`

3. **Multi-task Learning Framework**
   - Detection head (primary task)
   - Optional BEV segmentation head
   - Optional depth estimation auxiliary task
   - Shared backbone and encoder, task-specific heads

4. **Linear Increasing Discretization (LID)**
   - Non-uniform depth discretization: finer bins at close range
   - Better accuracy for nearby objects (most safety-critical)

### Performance Improvement

| Model | mAP | NDS | mAVE |
|-------|-----|-----|------|
| PETR (R50) | 31.3 | 38.1 | 0.93 |
| PETRv2 (R50) | 34.6 | 42.1 | 0.41 |

The ~3 mAP improvement comes primarily from temporal fusion, with a dramatic improvement in velocity estimation (mAVE reduced by >50%) due to temporal reasoning.

---

## StreamPETR: Object-Centric Temporal Modeling (ICCV 2023)

### Motivation

Previous temporal methods operate at the **feature level**:
- BEVFormer: aligns and fuses BEV features across frames (40K features per frame)
- PETRv2: concatenates previous frame's position-aware features

These approaches are memory-intensive and scale poorly with temporal history. StreamPETR asks: **can we do temporal modeling at the object level instead?**

### Core Innovation: Query Propagation

Instead of propagating dense features, StreamPETR propagates **object queries** across frames:

1. **Query Selection**: After each frame, select top-K confident queries (K=256)
2. **Ego-Motion Compensation**: Transform selected queries' reference points to the next frame's coordinate system using the ego-motion matrix
3. **Velocity Prediction**: Extrapolate query positions based on predicted velocities
4. **Query Injection**: In the next frame, use propagated queries alongside fresh queries (total = 900)

This means temporal context is carried by ~256 query vectors (256 * 256 = 65K parameters) rather than dense feature maps (200 * 200 * 256 = 10M parameters for BEV).

### Motion-Aware Layer Norm

StreamPETR introduces motion-aware layer normalization:
- The 4x4 ego-motion matrix is encoded by an MLP into scale/shift parameters
- These parameters modulate the LayerNorm in transformer layers
- This injects ego-motion awareness into the entire transformer computation
- No need to explicitly transform all features or reference points

### Memory and Compute Benefits

| Metric | BEVFormer | PETRv2 | StreamPETR |
|--------|-----------|--------|------------|
| Temporal memory | ~40K BEV features/frame | ~80K features/frame | ~256 queries/frame |
| Memory scaling | O(H*W*T) | O(N_feat*T) | O(N_queries) |
| FPS (R50, A100) | ~4 | ~8 | ~30 |
| GPU Memory | ~18 GB | ~14 GB | ~8 GB |

### Emergent Tracking

Because queries maintain identity across frames (same query tracks same object), StreamPETR naturally produces object tracks without an explicit tracking module. This is conceptually similar to how MOTR/TrackFormer work for 2D tracking.

### Performance

| Model | Backbone | mAP | NDS | FPS |
|-------|----------|-----|-----|-----|
| StreamPETR | ResNet-50 | 38.4 | 44.9 | ~30 |
| StreamPETR | ResNet-101 | 40.2 | 47.1 | ~20 |
| StreamPETR | VoVNet-99 | 45.0 | 55.0 | ~15 |
| StreamPETR | ViT-L (EVA02) | 55.2 | 63.6 | ~8 |

StreamPETR with ResNet-50 already outperforms PETRv2 with ResNet-101, while being 3-4x faster.

---

## Key Takeaways

1. **3D PE is a powerful abstraction**: Encoding 3D information into 2D features enables standard attention to perform 3D reasoning, avoiding complex geometric operations.

2. **Object-centric > Feature-centric temporal modeling**: For detection tasks, propagating object queries is far more efficient than propagating dense feature maps.

3. **Motion-aware normalization is effective**: Simple modifications to LayerNorm can inject ego-motion awareness without expensive feature warping.

4. **The PETR paradigm scales well**: From PETR to StreamPETR, the approach went from ~31 mAP/10 FPS to ~55 mAP/8 FPS (with larger backbone), showing the paradigm's scalability.

5. **Streaming perception is the future**: Real-time perception requires methods that process frames sequentially and incrementally, rather than batch-processing temporal windows.

---

## References

1. Liu, Y., et al. "PETR: Position Embedding Transformation for Multi-View 3D Object Detection." ECCV 2022.
2. Liu, Y., et al. "PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images." ICCV 2023.
3. Wang, S., et al. "Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection." ICCV 2023.
4. Wang, Y., et al. "DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries." CoRL 2021.
5. Li, Z., et al. "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers." ECCV 2022.
