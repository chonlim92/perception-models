# DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

## Paper Overview

- **Title:** DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries
- **Authors:** Yue Wang, Vitor Campagnolo Guizilini, Timanogo Zhang, Yilun Wang, Hang Zhao, Justin Solomon
- **Venue:** Conference on Robot Learning (CoRL), 2021
- **Key Contribution:** A framework that performs 3D object detection from multi-view camera images by back-projecting learned 3D reference points onto 2D image feature maps, eliminating the need for explicit depth estimation or bird's-eye-view (BEV) representations.

---

## Core Paradigm: 3D-to-2D Query Projection

DETR3D introduces a fundamentally different approach to multi-view 3D object detection. Instead of lifting 2D features into 3D space (as in LSS or BEVDet), DETR3D operates in the reverse direction:

1. **Learnable 3D Reference Points:** The model maintains a set of learnable 3D reference points in world coordinates. Each object query in the transformer decoder is associated with a predicted 3D reference point.

2. **Projection to 2D Image Planes:** Given the known camera calibration matrices (intrinsics K and extrinsics [R|t]), each 3D reference point is projected onto all camera image planes using standard pinhole camera geometry:
   ```
   p_2d = K * [R|t] * p_3d
   ```

3. **Feature Sampling:** At each projected 2D location, image features are sampled from the corresponding camera's feature map using bilinear interpolation. Features from multiple cameras (where the point is visible) are aggregated.

4. **Iterative Refinement:** The transformer decoder refines both the 3D reference points and the object queries across multiple decoder layers, progressively improving localization accuracy.

This approach avoids quantization errors inherent in voxel-based BEV representations and sidesteps the ill-posed nature of monocular depth estimation.

---

## Comparison to BEVFormer

| Aspect | DETR3D | BEVFormer |
|--------|--------|-----------|
| **3D Representation** | No explicit 3D representation; operates via sparse 3D-to-2D projections | Constructs explicit BEV feature map via spatial cross-attention |
| **Feature Aggregation** | Samples features at sparse projected points | Dense BEV grid queries multi-view features at each BEV cell |
| **Temporal Modeling** | Single-frame (no built-in temporal fusion) | Temporal self-attention fuses BEV features across time steps |
| **Computational Pattern** | Sparse: O(num_queries * num_cameras) | Dense: O(BEV_H * BEV_W * num_cameras) |
| **Memory Efficiency** | More memory efficient due to sparse queries | Higher memory due to dense BEV grid |
| **Spatial Reasoning** | Implicit via attention among queries | Explicit via structured BEV grid |
| **Performance (nuScenes)** | NDS ~42.5 (without CBGS), ~47.9 (with CBGS) | NDS ~51.7 (base), ~56.9 (with temporal) |
| **Architecture Complexity** | Simpler, fewer components | More complex with BEV encoder and temporal module |

Key insight: DETR3D trades off some detection accuracy for architectural simplicity and computational efficiency. It was a pioneering work that demonstrated transformer-based camera-only 3D detection was viable, paving the way for BEVFormer and subsequent methods.

---

## DETR Heritage

DETR3D directly inherits core design principles from the original DETR (DEtection TRansformer) by Carlin et al. (ECCV 2020):

### Set Prediction Framework
- Detection is formulated as a direct set prediction problem, outputting a fixed-size set of N predictions in parallel (no NMS post-processing required).
- Each prediction consists of a class label and a 3D bounding box parameterization.
- The number of queries (typically 900) is set larger than the expected number of objects in any scene.

### Hungarian Matching
- During training, a bipartite matching algorithm (Hungarian algorithm) finds the optimal one-to-one assignment between predicted and ground-truth objects.
- The matching cost considers both classification confidence and bounding box regression quality.
- Unmatched predictions are assigned the "no object" class, encouraging the model to produce exactly the right number of detections.

### Transformer Architecture
- **Self-attention among queries:** Enables reasoning about inter-object relationships (e.g., spatial arrangements, preventing duplicate detections).
- **Cross-attention via feature sampling:** Instead of standard cross-attention over flattened image features (as in 2D DETR), DETR3D uses projected feature sampling as its cross-attention mechanism.
- **Multi-layer refinement:** Typically 6 decoder layers, each progressively refining predictions.

### Differences from 2D DETR
- No encoder (backbone features are used directly without a transformer encoder, reducing computation).
- 3D reference points replace 2D positional queries.
- Cross-attention is replaced by geometric projection + bilinear sampling.
- Output space is 3D (center, dimensions, rotation, velocity) rather than 2D boxes.

---

## Key Results on nuScenes

### Main Results (nuScenes val set)

| Model | Backbone | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|-------|----------|-----|-----|------|------|------|------|------|
| DETR3D (w/o CBGS) | ResNet-101 | 0.425 | 0.346 | 0.716 | 0.268 | 0.379 | 0.842 | 0.200 |
| DETR3D (w/ CBGS) | ResNet-101 | 0.479 | 0.412 | 0.641 | 0.255 | 0.394 | 0.845 | 0.133 |
| DETR3D | VoVNet-99 | 0.479 | 0.412 | 0.641 | 0.255 | 0.394 | 0.845 | 0.133 |

### nuScenes Test Set (Leaderboard)

| Model | NDS | mAP |
|-------|-----|-----|
| DETR3D (ResNet-101-DCN, CBGS) | 0.479 | 0.412 |

### Performance Highlights
- **First competitive camera-only method:** Demonstrated that camera-only 3D detection could approach LiDAR-based methods on nuScenes without explicit depth supervision.
- **No post-processing:** Achieves results without NMS, test-time augmentation, or model ensembling in the base configuration.
- **Efficient inference:** Faster than contemporary BEV-based methods due to sparse query formulation.
- **CBGS impact:** Class-balanced grouping and sampling provides a significant boost (+5.4 NDS, +6.6 mAP), addressing the severe class imbalance in nuScenes.

### Ablation Studies
- Increasing the number of decoder layers from 1 to 6 improves NDS by ~5 points.
- Multi-scale features (FPN) provide ~2 point improvement over single-scale.
- The 3D-to-2D projection mechanism outperforms naive feature concatenation by a large margin.
- Iterative reference point refinement across decoder layers is critical for accurate localization.

---

## Significance and Impact

DETR3D established several important paradigms for the field:

1. **Camera-only 3D detection viability:** Proved that competitive 3D detection is achievable without LiDAR or explicit depth estimation.
2. **Geometric projection as attention:** Introduced the idea of using camera geometry to guide cross-attention, which influenced subsequent works (PETR, BEVFormer, StreamPETR).
3. **Query-based 3D detection:** Pioneered the use of learnable 3D queries for multi-view detection, now standard in the field.
4. **Simplicity:** The architectural simplicity made it highly extensible, spawning numerous follow-up works that added temporal modeling, BEV representations, and improved feature aggregation.
