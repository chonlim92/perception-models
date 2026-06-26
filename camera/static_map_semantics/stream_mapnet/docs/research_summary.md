# StreamMapNet: Research Summary

## Paper Overview

**Title:** Streaming Mapping Network for Vectorized Online HD Map Construction  
**Authors:** Tianyuan Yuan, Yicheng Liu, Yue Wang, Yilun Wang, Hang Zhao  
**Venue:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2024  
**arXiv:** 2308.12570  

StreamMapNet introduces a streaming paradigm for online HD map construction from surround-view camera images. Unlike prior single-frame methods that process each timestamp independently, StreamMapNet propagates temporal information across frames using a lightweight hidden-state mechanism, achieving significant improvements in both accuracy and temporal consistency without re-processing historical frames.

---

## Core Motivation

Online HD map construction from camera images is critical for autonomous driving systems that cannot rely solely on pre-built HD maps. Prior approaches fall into two categories:

1. **Single-frame methods** (HDMapNet, VectorMapNet, MapTR): Process each frame independently, leading to temporally inconsistent predictions and failure to exploit redundancy across consecutive frames.
2. **Multi-frame methods** (BEVFormer temporal self-attention): Re-process history frames or store large BEV feature buffers, incurring significant computational overhead.

StreamMapNet proposes a third path: **streaming propagation** that carries forward a compact hidden state (warped BEV features) without re-encoding past images.

---

## Streaming Paradigm vs Single-Frame Approaches

### Single-Frame Pipeline (MapTR, HDMapNet)

```
Frame t:  Images → Backbone → BEV → Decoder → Map Elements
Frame t+1: Images → Backbone → BEV → Decoder → Map Elements  (independent)
```

Each frame is processed in isolation. No information from frame t is available at frame t+1. This leads to:
- Flickering predictions between consecutive frames
- Failure to recover occluded or partially visible elements
- No accumulation of evidence over time

### Streaming Pipeline (StreamMapNet)

```
Frame t:   Images → Backbone → BEV_t → Fuse(BEV_t, Warp(H_{t-1})) → H_t → Decoder → Map Elements
Frame t+1: Images → Backbone → BEV_{t+1} → Fuse(BEV_{t+1}, Warp(H_t)) → H_{t+1} → Decoder → Map Elements
```

Key differences:
- Hidden state H_t encodes accumulated temporal information
- Ego-motion warping aligns previous hidden state to current coordinate frame
- Fusion module combines current observations with propagated history
- No re-processing of past images; constant memory footprint regardless of history length

---

## Temporal Propagation Mechanism

### Step 1: Ego-Motion Warping

Given the ego-motion transformation matrix T_{t→t+1} between consecutive frames (obtained from vehicle odometry or pose estimation):

1. Construct a 2D sampling grid in the current frame's BEV coordinate system
2. Transform grid coordinates to the previous frame's coordinate system using T_{t→t+1}^{-1}
3. Use `grid_sample` (bilinear interpolation) to resample the previous hidden state H_{t-1} onto the current coordinate frame

```python
# Pseudo-code for ego-motion warping
grid = create_bev_grid(H, W)  # (H, W, 2) normalized coordinates
grid_prev = transform_grid(grid, T_curr_to_prev)  # apply inverse ego-motion
H_warped = F.grid_sample(H_prev, grid_prev, mode='bilinear', align_corners=True)
```

### Step 2: Temporal Fusion

The warped previous hidden state H_warped is fused with the current BEV features BEV_t through a temporal attention mechanism:

1. **Concatenation + Conv:** Simple fusion baseline concatenating H_warped and BEV_t along channel dimension followed by 1x1 convolution
2. **Temporal Cross-Attention (primary method):** Queries from current BEV attend to keys/values from warped history, enabling selective information retrieval
3. **Gated Fusion:** Learnable gate determines per-location how much to rely on current observation vs. history

The fused representation becomes the new hidden state H_t, which is both passed to the decoder for current-frame prediction and propagated to the next frame.

### Step 3: Hidden State Update

```
H_t = FusionModule(BEV_t, Warp(H_{t-1}, ego_motion))
```

The hidden state has the same spatial dimensions as the BEV feature map (e.g., 200x100) but may differ in channel dimension. It serves dual purposes:
- Input to the map decoder for producing predictions at time t
- Carried forward (after warping) to provide temporal context at time t+1

---

## Comparison Table

| Aspect | HDMapNet | VectorMapNet | MapTR | StreamMapNet |
|--------|----------|--------------|-------|--------------|
| **Year** | 2022 | 2023 | 2023 | 2024 (WACV) |
| **Output Format** | Rasterized segmentation | Vectorized polylines | Vectorized polylines | Vectorized polylines |
| **Temporal Info** | None | None | None | Streaming hidden state |
| **BEV Method** | IPM + learned | Cross-attention | GKT / LSS | LSS with depth |
| **Decoder** | CNN segmentation head | Autoregressive transformer | DETR-like parallel | DETR-like parallel |
| **Map Representation** | Pixel masks | Ordered point sequences | Fixed-point polylines | Fixed-point polylines |
| **Backbone** | EfficientNet-B0 | ResNet-50 | ResNet-50 | ResNet-50 |
| **Matching** | Per-pixel | Sequential | Hungarian (permutation-invariant) | Hungarian (permutation-invariant) |
| **Temporal Consistency** | Poor (flickering) | Poor | Poor | Strong (propagated state) |
| **Inference Speed** | Fast (no decoder) | Slow (autoregressive) | Fast | Fast (minimal overhead) |
| **nuScenes mAP** | ~30.0 | ~36.1 | ~50.3 | ~54.1 |

---

## Key Metrics and Improvements

### nuScenes val set (24-epoch schedule, ResNet-50 backbone)

| Method | Divider AP | Ped Crossing AP | Boundary AP | mAP |
|--------|-----------|----------------|-------------|-----|
| HDMapNet | 18.5 | 14.1 | 37.6 | 23.4 |
| VectorMapNet | 36.2 | 28.5 | 43.5 | 36.1 |
| MapTR | 51.5 | 46.3 | 53.1 | 50.3 |
| MapTRv2 | 55.7 | 49.2 | 57.4 | 54.1 |
| **StreamMapNet** | **56.3** | **50.1** | **55.8** | **54.1** |

### Temporal Fusion Ablation (improvements over single-frame baseline)

| Configuration | mAP | Delta |
|--------------|-----|-------|
| Single-frame (no temporal) | 50.3 | - |
| + Ego-motion warping only | 52.1 | +1.8 |
| + Temporal concatenation | 52.8 | +2.5 |
| + Temporal cross-attention | 54.1 | +3.8 |
| + Multi-frame propagation (3 frames) | 54.1 | +3.8 |

### Argoverse 2 val set

| Method | Divider AP | Ped Crossing AP | Boundary AP | mAP |
|--------|-----------|----------------|-------------|-----|
| MapTR | 58.7 | 52.1 | 60.3 | 57.0 |
| **StreamMapNet** | **62.4** | **56.8** | **63.1** | **60.8** |

---

## Streaming Design: No Re-Processing of History

The fundamental design principle of StreamMapNet is computational efficiency through streaming:

1. **Constant-time processing:** Each new frame requires only one forward pass through the image backbone and BEV encoder, regardless of how many historical frames have been observed.

2. **Fixed memory footprint:** Only one hidden state tensor (same size as BEV features) is stored, not a buffer of past BEV features or raw images.

3. **No redundant computation:** Unlike BEVFormer's temporal self-attention which stores and attends to multiple past BEV features, StreamMapNet compresses all history into a single propagated hidden state.

4. **Graceful degradation:** If the hidden state becomes stale (e.g., after a long gap or scene change), the model can still produce reasonable predictions from the current frame alone, as the fusion module learns to gate historical information.

5. **Training with sequences:** During training, sequences of consecutive frames are processed in order, with gradients flowing through the warping and fusion operations. This teaches the network to produce hidden states that are useful for future frames.

### Computational Cost Comparison

| Method | Additional FLOPs per frame (temporal) | Memory for history |
|--------|--------------------------------------|-------------------|
| MapTR (single-frame) | 0 | 0 |
| BEVFormer (4-frame buffer) | ~3.2G (temporal self-attn) | 4x BEV features |
| StreamMapNet | ~0.8G (warp + fusion) | 1x hidden state |

---

## Key Contributions

1. **Streaming paradigm for online mapping:** First work to introduce a streaming hidden-state approach specifically for vectorized HD map construction.

2. **Ego-motion-aware temporal propagation:** Principled spatial alignment of historical features using ego-motion, critical for maintaining spatial accuracy as the vehicle moves.

3. **Lightweight temporal fusion:** The temporal module adds minimal computational overhead (<10% of total inference time) while providing significant accuracy gains.

4. **Temporal consistency:** Beyond raw mAP improvements, StreamMapNet produces temporally smooth predictions that are more suitable for downstream planning modules.

5. **Multi-dataset validation:** Evaluated on both nuScenes and Argoverse 2, demonstrating generalization of the streaming approach across different sensor configurations and annotation styles.

---

## Limitations and Future Directions

- **Error accumulation:** Long sequences may accumulate drift in the warped hidden state, though ego-motion from high-quality IMU/GPS mitigates this.
- **Scene transitions:** Abrupt scene changes (cuts in dataset) can corrupt the hidden state; the model must learn to reset.
- **Single-modality:** The paper focuses on camera-only input; LiDAR fusion could further improve BEV quality.
- **Static map assumption:** The method assumes map elements are static; dynamic elements in the map (construction zones, temporary markings) are not explicitly handled.

---

## References

- Yuan, T., Liu, Y., Wang, Y., Wang, Y., & Zhao, H. (2024). StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction. WACV 2024.
- Liao, B., et al. (2023). MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction. ICLR 2023.
- Li, Q., et al. (2022). HDMapNet: An Online HD Map Construction and Evaluation Framework. ICRA 2022.
- Liu, Y., et al. (2023). VectorMapNet: End-to-end Vectorized HD Map Learning. ICML 2023.
- Li, Z., et al. (2022). BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers. ECCV 2022.
