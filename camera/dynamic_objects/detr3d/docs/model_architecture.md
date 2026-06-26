# Model Architecture: DETR3D

## Architecture Overview

DETR3D is a transformer-based 3D object detection framework that operates on multi-view camera images. The architecture consists of four main components:

1. **Image Backbone + FPN:** Extracts multi-scale 2D features from each camera view independently
2. **3D Object Queries with Reference Points:** Learnable queries that encode potential object locations in 3D
3. **3D-to-2D Projection and Feature Sampling:** Geometric projection using camera matrices to sample relevant image features
4. **Transformer Decoder + Detection Head:** Iterative refinement of object queries and final prediction

```
Multi-view Images (6 cameras)
        │
        ▼
┌─────────────────────────┐
│  ResNet-101 + FPN        │  (shared weights across all views)
│  → Multi-scale features  │
└─────────────────────────┘
        │
        ▼ Feature maps: {P2, P3, P4, P5} per camera
        │
┌─────────────────────────────────────────┐
│  Transformer Decoder (6 layers)          │
│  ┌─────────────────────────────────┐    │
│  │ Self-Attention (query-to-query)  │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │ 3D-to-2D Projection + Sampling  │    │  ← Camera intrinsics K
│  │ (replaces cross-attention)       │    │  ← Camera extrinsics [R|t]
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │ FFN (Feed-Forward Network)       │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │ Reference Point Refinement       │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Detection Head           │
│  → Class logits (10+1)   │
│  → 3D BBox regression    │
│  → Velocity prediction   │
└─────────────────────────┘
```

---

## Component 1: Backbone (ResNet-101 + FPN)

### ResNet-101 Backbone
- **Architecture:** Standard ResNet-101 with 101 layers, pre-trained on ImageNet
- **Input:** Each camera image independently, resized to 900 x 1600 (H x W) or similar
- **Feature extraction stages:**
  - Stage 1 (C1): 64 channels, stride 2 (not used in FPN)
  - Stage 2 (C2): 256 channels, stride 4
  - Stage 3 (C3): 512 channels, stride 8
  - Stage 4 (C4): 1024 channels, stride 16
  - Stage 5 (C5): 2048 channels, stride 32

### Feature Pyramid Network (FPN)
- **Purpose:** Generate multi-scale feature maps for detecting objects at different distances/sizes
- **Architecture:** Top-down pathway with lateral connections
- **Output feature levels:**
  - P2: stride 4, 256 channels (highest resolution, for nearby small objects)
  - P3: stride 8, 256 channels
  - P4: stride 16, 256 channels (primary level for most objects)
  - P5: stride 32, 256 channels (lowest resolution, for distant/large objects)
- **Channel dimension:** All FPN levels are projected to 256 channels

### Weight Sharing
- The backbone and FPN weights are **shared** across all 6 camera views
- Each camera image is processed independently (no cross-view feature interaction in the backbone)
- This design ensures computational efficiency and parameter efficiency

### Optional Enhancements
- **DCN (Deformable Convolution Networks):** DCNv2 in the last stage of ResNet improves feature alignment, adding ~1-2 NDS points
- **VoVNet-99:** Alternative backbone with better efficiency-accuracy trade-off
- **Pre-training:** ImageNet pre-training is essential; FCOS3D pre-training provides additional boost (~1 NDS)

---

## Component 2: Learnable 3D Object Queries

### Query Design
- **Number of queries:** 900 (fixed, set larger than maximum expected objects per scene)
- **Query dimension:** 256 (matching FPN feature dimension)
- **Initialization:** Learned embeddings, randomly initialized before training

### 3D Reference Points
Each object query is associated with a 3D reference point that represents its hypothesized location:

- **Initial reference points:** Predicted from the query embedding via a small MLP:
  ```
  ref_point_3d = sigmoid(MLP(query_embedding))  # normalized to [0, 1]^3
  ```
- **Coordinate range:** Reference points are initialized in a normalized space and then mapped to the physical detection range (e.g., [-51.2m, 51.2m] for X/Y, [-5m, 3m] for Z)
- **Learnable:** Both the query embeddings and the reference point prediction network are learned end-to-end

### Query Semantics
- Each query is intended to detect at most one object (enforced by Hungarian matching)
- Queries implicitly specialize during training: some learn to detect cars, others pedestrians, etc.
- No explicit spatial anchoring (unlike anchor-based methods): queries can detect objects anywhere in the scene

---

## Component 3: 3D-to-2D Projection and Feature Sampling

This is the core innovation of DETR3D, replacing standard cross-attention with geometry-guided feature sampling.

### Projection Pipeline

**Step 1: Transform to Camera Coordinates**
```
p_cam = T_cam_from_ego @ T_ego_from_global @ p_3d
```
Where:
- `p_3d`: 3D reference point [x, y, z, 1]^T in world/ego coordinates
- `T_ego_from_global`: Ego pose inverse (4x4 transformation matrix)
- `T_cam_from_ego`: Camera extrinsic inverse (4x4, rotation R and translation t)

**Step 2: Project to Image Plane**
```
[u, v, d]^T = K @ p_cam[:3]
u_pixel = u / d
v_pixel = v / d
```
Where:
- `K`: 3x3 camera intrinsic matrix containing focal lengths (fx, fy) and principal point (cx, cy)
- `d`: depth (must be positive for point to be in front of camera)

**Step 3: Normalize to Feature Map Coordinates**
```
u_norm = u_pixel / image_width   # [0, 1]
v_norm = v_pixel / image_height  # [0, 1]
```

### Bilinear Feature Sampling
- At each projected 2D location (u_norm, v_norm), features are sampled using bilinear interpolation from the corresponding FPN feature map
- **Multi-scale sampling:** Features are sampled from all FPN levels and summed (or from a selected level based on reference point depth)
- **Grid sample operation:** Uses PyTorch's `F.grid_sample` with bilinear interpolation and zero padding for out-of-bounds locations

### Multi-Camera Aggregation
For each 3D reference point projected to multiple cameras:
1. Check visibility: depth > 0 and projected coordinates within image bounds
2. Sample features from each visible camera's feature maps
3. Aggregate features: element-wise mean across all visible cameras
4. If not visible in any camera: feature is set to zero vector

### Feature Sampling Code (Simplified)
```python
def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    # reference_points: (B, num_queries, 3) in ego coordinates
    # mlvl_feats: list of (B, num_cams, C, H_i, W_i) per FPN level

    # Project 3D points to all camera image planes
    lidar2img = img_metas['lidar2img']  # (B, num_cams, 4, 4)
    reference_points_cam = project_to_cameras(reference_points, lidar2img)
    # reference_points_cam: (B, num_cams, num_queries, 2) normalized coords

    # Check visibility mask
    mask = (reference_points_cam[..., 0] > 0) & (reference_points_cam[..., 0] < 1) & \
           (reference_points_cam[..., 1] > 0) & (reference_points_cam[..., 1] < 1) & \
           (depths > 0)

    # Sample from each FPN level
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        sampled = F.grid_sample(feat, reference_points_cam)  # bilinear sampling
        sampled_feats.append(sampled)

    # Aggregate across levels and cameras
    sampled_feats = torch.stack(sampled_feats).sum(0)  # sum over levels
    sampled_feats = sampled_feats * mask  # zero out invisible
    sampled_feats = sampled_feats.mean(dim=cam_dim)  # mean over cameras

    return sampled_feats
```

---

## Component 4: Transformer Decoder

### Decoder Architecture
- **Number of layers:** 6 (default)
- **Hidden dimension:** 256
- **Number of attention heads:** 8
- **FFN intermediate dimension:** 512 (2x expansion)
- **Dropout:** 0.1
- **Activation:** ReLU in FFN

### Each Decoder Layer Contains:

#### 1. Multi-Head Self-Attention
- **Input:** Object query embeddings (900 x 256)
- **Purpose:** Enable reasoning about inter-object relationships
  - Prevents duplicate detections of the same object
  - Models spatial relationships between objects
  - Enables contextual reasoning (e.g., cars often appear near roads)
- **Positional encoding:** Query position embeddings added to keys and queries
- **Complexity:** O(num_queries^2 * dim) = O(900^2 * 256)

#### 2. Cross-Attention via Feature Sampling (3D-to-2D Projection)
- **Replaces standard cross-attention:** Instead of attending over flattened image features, uses geometric projection
- **Input:** Query embeddings + 3D reference points + multi-view feature maps
- **Process:**
  1. Project each query's 3D reference point to all camera images
  2. Sample features at projected locations
  3. Aggregate sampled features as the "cross-attention output"
- **Advantage:** O(num_queries * num_cameras) instead of O(num_queries * H*W*num_cameras)
- **No attention weights:** Feature sampling is deterministic given the geometry (no learned attention map)

#### 3. Feed-Forward Network (FFN)
- Two-layer MLP with ReLU activation
- `FFN(x) = Linear(ReLU(Linear(x)))`
- Provides per-query non-linear feature transformation
- Residual connection and layer normalization around each sub-layer

#### 4. Reference Point Refinement
- After each decoder layer, reference points are updated:
  ```
  ref_point_new = ref_point_old + delta_ref(query_output)
  ```
- `delta_ref`: Small MLP that predicts position offset
- This iterative refinement enables coarse-to-fine localization
- Each layer's predictions are supervised (auxiliary losses)

### Decoder Information Flow
```
Input: query_embeddings (900 x 256), ref_points (900 x 3)

For each decoder layer l = 1, ..., 6:
    1. q = LayerNorm(query + self_attn(query, query, query))
    2. sampled_feats = project_and_sample(ref_points, features, cameras)
    3. q = LayerNorm(q + linear(sampled_feats))  # "cross-attention"
    4. q = LayerNorm(q + FFN(q))
    5. ref_points = ref_points + MLP_refine(q)  # refine positions
    6. predictions_l = detection_head(q, ref_points)  # auxiliary output

Output: final predictions from layer 6 (and auxiliary predictions from layers 1-5)
```

---

## Component 5: Detection Head

### Classification Head
- **Input:** Query feature (256-dim)
- **Architecture:** 2-layer MLP with ReLU, output dimension = num_classes + 1 = 11
- **Output:** Class logits for 10 object classes + 1 background/no-object class
- **Activation:** No softmax during training (applied during inference for probability)
- **Shared weights:** Same classification head applied at every decoder layer

### 3D Bounding Box Regression Head
- **Input:** Query feature (256-dim) + reference point (3-dim)
- **Architecture:** 2-layer MLP with ReLU
- **Output:** 10-dimensional vector:
  - `[delta_cx, delta_cy, delta_cz]`: Offset from reference point to box center
  - `[log(w), log(l), log(h)]`: Log-scale dimensions (width, length, height)
  - `[sin(yaw), cos(yaw)]`: Orientation as sine/cosine
  - `[vx, vy]`: Velocity in X and Y directions
- **Final center:** `center = reference_point + [delta_cx, delta_cy, delta_cz]`
- **Shared weights:** Same regression head applied at every decoder layer

### Attribute Head (Optional)
- **Input:** Query feature (256-dim)
- **Architecture:** Linear layer, output dimension = 8 (max attribute classes)
- **Output:** Attribute class logits (activity state prediction)
- **Class-conditional:** Only certain attributes are valid for each object class

### Output Format
For each of the 900 queries, the model outputs:
```python
{
    'cls_scores': tensor(900, 11),        # class logits
    'bbox_preds': tensor(900, 10),        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]
    'attr_preds': tensor(900, 8),         # attribute logits (optional)
}
```

---

## Model Dimensions Summary

| Component | Parameter | Value |
|-----------|-----------|-------|
| Input image size | H x W | 900 x 1600 |
| Number of cameras | - | 6 |
| Backbone | - | ResNet-101 |
| FPN output channels | C | 256 |
| FPN levels | - | 4 (P2, P3, P4, P5) |
| Number of queries | N_q | 900 |
| Query dimension | d | 256 |
| Decoder layers | L | 6 |
| Attention heads | H | 8 |
| FFN hidden dim | d_ff | 512 |
| Detection range (X/Y) | - | [-51.2m, 51.2m] |
| Detection range (Z) | - | [-5.0m, 3.0m] |
| Total parameters | - | ~55M (backbone) + ~10M (decoder + heads) |

---

## Inference Pipeline

### Step-by-Step Inference
1. **Load 6 camera images** and their associated calibration matrices
2. **Extract features:** Pass each image through ResNet-101 + FPN (batched across cameras)
3. **Initialize queries:** Load learned query embeddings and predict initial reference points
4. **Decoder forward pass:** Run 6 decoder layers with projection-based feature sampling
5. **Apply detection head:** Get class scores and bounding box predictions from final layer
6. **Post-processing:**
   - Apply sigmoid to class scores to get probabilities
   - Filter predictions by confidence threshold (e.g., 0.1)
   - No NMS required (set prediction formulation inherently produces non-duplicate outputs)
   - Convert predictions from normalized space to physical coordinates

### Inference Speed
- **GPU:** ~6-8 FPS on single NVIDIA V100
- **Latency breakdown:**
  - Backbone + FPN: ~60% of total time
  - Transformer decoder: ~30% of total time
  - Detection head + post-processing: ~10% of total time
- **Memory:** ~8 GB GPU memory for inference with batch size 1
