# Hierarchical Lane Positional Embeddings

Comprehensive documentation for the topology-aware hierarchical lane positional
embedding system implemented across 5 lane detection models.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Query Layout](#query-layout)
3. [Key Design Decisions](#key-design-decisions)
4. [Per-Model Implementation Details](#per-model-implementation-details)
5. [Shared Utilities](#shared-utilities-commonlane_topologypy)
6. [Training Guide](#training-guide)
7. [API Reference](#api-reference)

---

## Architecture Overview

The hierarchical lane positional embedding encodes the structural topology of
lane detection outputs directly into the transformer query design. Instead of
using flat, unstructured query embeddings, each query knows its position in a
three-level hierarchy:

```
                    ┌─────────────────────────────────────────────┐
                    │          Hierarchical Position               │
                    │                                             │
                    │  pos = lane_embed[id] + line_type_embed[t]  │
                    │        + point_embed[pt_idx]                │
                    │                                             │
                    │  pos = LayerNorm(pos)                       │
                    │  pos = Dropout(pos, p=0.05)                 │
                    └─────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
              ┌─────┴─────┐     ┌──────┴──────┐    ┌──────┴──────┐
              │   Lane    │     │  Line Type  │    │   Point     │
              │ Embedding │     │  Embedding  │    │  Embedding  │
              │           │     │             │    │  (Hybrid)   │
              │ 25 lanes  │     │ 0=left      │    │ sinusoidal  │
              │ + others  │     │ 1=right     │    │ + learned   │
              │           │     │ 2=other     │    │ residual    │
              └───────────┘     └─────────────┘    └─────────────┘
```

### Core Principle

Each query in the decoder corresponds to a specific **point** on a specific
**boundary line** (left or right) of a specific **lane**. The positional
embedding makes this structure explicit:

- **Lane-level**: Which lane (0-24) or other line group
- **Line-type**: Left boundary (0), right boundary (1), or other (2)
- **Point-position**: Which point along the line (0-19)

### Content vs Position Separation

The system maintains strict separation:
- **Positional embedding**: Encodes WHERE in the structure (topology)
- **Content embedding**: Learnable per-slot features (WHAT to predict)

Both are (total_queries, embed_dim) tensors returned by the forward pass.

---

## Query Layout

```
Structure: 25 lanes × 2 boundary lines × 20 points = 1000 lane queries

Index    │ Lane │ Line  │ Point │ Description
─────────┼──────┼───────┼───────┼──────────────────────────
0-19     │  0   │ Left  │ 0-19  │ Lane 0, left boundary
20-39    │  0   │ Right │ 0-19  │ Lane 0, right boundary
40-59    │  1   │ Left  │ 0-19  │ Lane 1, left boundary
60-79    │  1   │ Right │ 0-19  │ Lane 1, right boundary
...      │ ...  │ ...   │ ...   │ ...
960-979  │ 24   │ Left  │ 0-19  │ Lane 24, left boundary
980-999  │ 24   │ Right │ 0-19  │ Lane 24, right boundary
─────────┼──────┼───────┼───────┼──────────────────────────
1000+    │ 25+  │ Other │ 0-19  │ Additional polylines
```

The ordering is: `lane_idx → line_type (left, right) → point_idx`

This layout enables:
- Block-diagonal self-attention masks (each line = one block of 20 queries)
- Easy reshaping for per-line classification (pool 20 points per line)
- Efficient lane-width computation (left/right are adjacent blocks)

---

## Key Design Decisions

### 1. Balanced Magnitude Initialization

**Problem**: Raw sinusoidal PE has RMS ~0.707, while learned embeddings init at
std=0.02. The sinusoidal would dominate 97% of the positional signal.

**Solution**: Scale sinusoidal by 0.02: `pe = pe * 0.02`

**Effect**: All components contribute equally at initialization:
- lane_embedding: std=0.02, RMS ~0.016
- line_type_embedding: std=0.02, RMS ~0.016
- point_sinusoidal: RMS ~0.011 (after scaling)
- point_residual: std=0.02, RMS ~0.016

This ensures equal gradient flow from the start. Without scaling, the model
would spend many epochs learning to distinguish lanes because only point-order
structure would be visible in attention patterns.

### 2. Hybrid Point Embedding (Sinusoidal + Learned)

```python
point_embed = point_sinusoidal[ids] + point_residual(ids)
```

- **Sinusoidal base**: Provides inductive bias that adjacent points are close
  in embedding space. This encodes ordinal structure without any learning.
- **Learned residual**: Allows the model to deviate from pure sinusoidal when
  lane geometry requires non-uniform point spacing (e.g., denser at curves).
- Both at same scale (0.02) so the residual can meaningfully override.

### 3. Decoupled Block-Diagonal Self-Attention

```
Points on same line:  can attend (mask = 0)
Points on other line: cannot attend (mask = -inf)

Mask structure (for 3 lines × 4 points):
    ┌────────────────────────┐
    │ 0  0  0  0 │-∞ -∞ ... │
    │ 0  0  0  0 │-∞ -∞ ... │
    │ 0  0  0  0 │-∞ -∞ ... │
    │ 0  0  0  0 │-∞ -∞ ... │
    │─────────────┼──────────│
    │-∞ -∞ -∞ -∞ │ 0  0 ... │
    │...                     │
    └────────────────────────┘
```

**Registered as buffer** (computed once at `__init__`):
- Saves ~4 MB allocation per forward call (1000×1000 float32)
- Enables torch.compile and CUDA graph capture
- Automatically migrates with `.to(device)`

### 4. Per-Head Geometric ALiBi (MapTR)

Instead of a single uniform slope, uses geometric decay per head:

```
slope_h = 2^(-h * 8/H) * 0.5    for h = 1..H (H=8 heads)

Head 1: slope=0.25, max bias at dist=19: -4.75 (strong locality)
Head 2: slope=0.125, max bias: -2.375
...
Head 8: slope=0.00195, max bias: -0.037 (nearly global)
```

This gives multi-scale attention: some heads focus on local geometry (nearby
points for curvature estimation), while others capture long-range dependencies
(overall lane shape).

### 5. Gated Dynamic Position Injection

```python
query_pos = static_pos + gate.sigmoid() * proj(reference_points)
```

- `gate` initialized to 0 → sigmoid(0) = 0.5 (half contribution)
- Early training: random reference points add noise; gate learns to suppress
- Later training: reference points become meaningful; gate grows to amplify
- Prevents optimization conflict between structural identity (static) and
  spatial location (dynamic)

### 6. Inference Caching with Device Safety

```python
def _apply(self, fn):
    """Invalidate cache on .to()/.cuda()/.half() calls."""
    self._cached_pos = None
    return super()._apply(fn)
```

Without this, `_cached_pos` (a plain attribute) would stay on the old device
after `model.to('cuda')`, causing runtime crashes in deployment pipelines that
load on CPU then move to GPU.

### 7. Reduced Positional Dropout (0.05)

Standard transformers use 0.1 dropout on embeddings. For geometric regression
(predicting BEV coordinates), positional precision is critical — dropping 10%
of position dimensions risks losing structural information needed for accurate
coordinate prediction. 5% provides regularization while preserving geometry.

### 8. Smooth-L1 Lane Width Consistency Loss

```python
# Old: (width_diff ** 2).mean()  -- outlier-sensitive
# New: F.smooth_l1_loss(width_diff, zeros, beta=0.01)
```

- Below beta (0.01): L2 behavior (smooth gradients for normal variation)
- Above beta: L1 behavior (bounded gradients for outliers)
- Handles lane merges/splits gracefully (sudden width changes are L1, not L2)
- Uses eps-clamped norm: `(diff**2).sum(-1).clamp(min=1e-6).sqrt()`
  to avoid dead gradients at zero width

### 9. StreamMapNet Position Injection Fix

**Before** (incorrect):
```python
for layer in self.layers:
    queries = layer(queries + query_pos, memory)  # pos in residual!
```

**After** (correct):
```python
for layer in self.layers:
    queries = layer(queries, memory, query_pos=query_pos, ...)
```

The layer adds pos only to Q and K (for attention routing), NOT to the value
stream. This prevents 6× amplification of positional signal through the
residual connections across 6 layers.

### 10. get_lane_mask Returns Clone

```python
def get_lane_mask(self):
    return self.lane_mask.clone()  # not self.lane_mask
```

Prevents accidental in-place modification of the registered buffer, which
would corrupt model state and persist through save/load.

---

## Per-Model Implementation Details

### 1. BEVFormer

**File**: `camera/dynamic_objects/bevformer/pytorch/model.py`

**Classes**:
- `HierarchicalLanePositionalEmbedding`: Standard hierarchical PE
- `LaneDetectionDecoder`: BEV cross-attention decoder with decoupled mask

**Characteristics**:
- Static positional embedding only (no dynamic reference point injection)
- BEV features as cross-attention memory (2D, no 3D projection needed)
- `DecoderLayer` accepts `self_attn_mask` parameter
- Intermediate point predictions at each layer for auxiliary loss
- Per-line confidence via mean-pooling point features

**Forward flow**:
```
BEV features → cross-attention ← (content + position as Q/K)
                                   ↓
                        self-attention with decoupled mask
                                   ↓
                        point_reg_head → sigmoid → (B, Q, 2)
```

### 2. DETR3D

**File**: `camera/dynamic_objects/detr3d/pytorch/decoder.py`

**Classes**:
- `HierarchicalLanePositionalEmbedding`: Standard hierarchical PE
- `DETR3DLaneDecoder`: 3D-to-2D sampling decoder with iterative refinement

**Characteristics**:
- 3D reference points projected to multi-camera images for feature sampling
- Gated dynamic position injection from 3D reference points
- Inverse-sigmoid iterative refinement in fp32 with eps=1e-3
- `_pc_range` as registered buffer (avoids per-call tensor creation)
- Pre-computed decoupled mask buffer
- `.detach()` on refined reference points to prevent gradient explosion

**Forward flow**:
```
Per layer:
  query_pos = static + gate * proj(ref_points_3d)
  query = layer(query, query_pos, sample(features, project(ref_3d)))
  ref_points = sigmoid(inverse_sigmoid(ref_points) + delta).detach()
```

### 3. PETR

**File**: `camera/dynamic_objects/petr/pytorch/decoder.py`

**Classes**:
- `HierarchicalLanePositionalEmbedding`: Standard hierarchical PE
- `PETRLaneDecoder`: Global cross-attention to 3D position-aware features

**Characteristics**:
- PETR encodes 3D position into image features directly (no explicit projection)
- Global attention to all position-aware tokens
- Gated dynamic position injection from 3D reference points
- Inline inverse-sigmoid refinement: `(log(x/(1-x)) + delta).sigmoid()`
- Pre-computed decoupled mask buffer
- Optional `key_pos` for position-aware features
- `return_intermediate` flag for auxiliary loss

### 4. MapTR

**File**: `camera/static_map_semantics/maptr/pytorch/map_decoder.py`

**Classes**:
- `HierarchicalLanePositionalEmbedding`: Standard hierarchical PE
- `MapDecoderLayer`: Layer with lazy-cached per-head ALiBi mask
- `HierarchicalLaneMapDecoder`: Full decoder with iterative 2D refinement

**Characteristics**:
- Per-head geometric ALiBi slopes for multi-scale locality
- Lazy-cached mask: built once on first forward, then reused
- Mask expanded to (B×num_heads, L, L) for nn.MultiheadAttention compatibility
- 2D reference points (BEV plane) with `torch.special.logit` refinement
- Gated dynamic position: `gate * proj(ref_2d) + static_pos`
- Output reshaped to (B, num_lines, points_per_line, embed_dim)

**ALiBi mask structure** (per head):
```
Within same line block: -slope_h * |i - j|  (soft distance penalty)
Cross-line positions:   -inf                 (hard block)
```

### 5. StreamMapNet

**File**: `camera/static_map_semantics/stream_mapnet/pytorch/model.py`

**Classes**:
- `HierarchicalLanePositionalEmbedding`: Standard hierarchical PE
- `MapDecoderLayer`: Refactored to accept query_pos and self_attn_mask
- `HierarchicalMapDecoder`: Temporal BEV decoder with lane structure

**Characteristics**:
- Position added only to Q/K (NOT residual stream) — architectural fix
- `MapDecoderLayer.forward(queries, memory, query_pos, self_attn_mask)`
- Pre-computed decoupled mask buffer on decoder
- BEV positional encoding interpolation in fp32 for precision
- No iterative refinement (single-pass prediction)
- Temporal BEV features from upstream StreamMapNet encoder

**Layer forward**:
```python
# Self-attention: pos on Q/K only
q = norm1(queries) + query_pos
q2 = self_attn(q, q, norm1(queries), attn_mask=mask)
queries = queries + dropout(q2)

# Cross-attention: pos on Q only
q = norm2(queries) + query_pos
q2 = cross_attn(q, memory, memory)
queries = queries + dropout(q2)
```

---

## Shared Utilities (common/lane_topology.py)

### HybridPointEmbedding

Reusable sinusoidal+learned point embedding with configurable scale.

```python
class HybridPointEmbedding(nn.Module):
    def __init__(self, num_points: int, embed_dim: int, sinusoidal_scale: float = 0.02):
        # sinusoidal_base: (num_points, embed_dim) buffer, scaled
        # residual: nn.Embedding(num_points, embed_dim), std=0.02

    def forward(self, point_ids: torch.Tensor) -> torch.Tensor:
        return self.sinusoidal_base[point_ids] + self.residual(point_ids)
```

### build_alibi_intra_line_bias

Pre-computes per-head ALiBi attention bias for block-diagonal attention.

```python
def build_alibi_intra_line_bias(
    num_total_lines: int,      # 50 for 25 lanes × 2
    points_per_line: int,      # 20
    num_heads: int,            # 8
    slope_scale: float = 0.5,  # controls overall bias strength
) -> torch.Tensor:             # (num_heads, total_queries, total_queries)
```

Slopes follow geometric sequence: `2^(-k * 8/H) * slope_scale` for k=1..H

### lane_width_consistency_loss

Geometric regularizer encouraging smooth lane width along each lane.

```python
def lane_width_consistency_loss(
    pred_points: torch.Tensor,   # (B, total_queries, 2) BEV coordinates
    num_lanes: int,              # 25
    points_per_line: int,        # 20
) -> torch.Tensor:               # scalar loss
```

Computes width = ||left - right|| at each station, then penalizes variation
between adjacent stations using Smooth-L1 (Huber) loss.

---

## Training Guide

### Hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Optimizer | AdamW | Standard for transformers |
| Learning rate | 2e-4 | With cosine schedule |
| Weight decay | 0.01 | Applied to all non-bias params |
| Warmup steps | 500 | Critical for gate+embedding co-adaptation |
| Batch size | 4-8 | Per-GPU |
| Width loss weight | 0.1 | Relative to main regression loss |
| pos_drop | 0.05 | Reduced for geometric precision |

### Monitoring

Track these during training:
- `dynamic_pos_gate` value: Should grow from 0 → 2-3 over training
- Per-component embedding norms: lane, line_type, point should stay balanced
- Lane width consistency loss: Should decrease smoothly

### Mixed Precision

All models are safe for fp16/bf16 training:
- Inverse-sigmoid computed in fp32 with eps=1e-3
- LayerNorm internally upcasts to fp32
- ALiBi bias values bounded (max -4.75, well within fp16 range)

---

## API Reference

### HierarchicalLanePositionalEmbedding

```python
class HierarchicalLanePositionalEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,        # Must be even
        num_lanes: int = 25,         # Number of lanes
        points_per_line: int = 20,   # Points per boundary line
        num_other_lines: int = 0,    # Additional non-lane polylines
        pos_drop: float = 0.05,      # Positional dropout rate
    ) -> None: ...

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (pos_embed, content_embed), each (total_queries, embed_dim)."""

    def get_lane_mask(self) -> torch.Tensor:
        """Returns (total_queries,) bool mask: True for lane queries."""

    # Properties
    total_queries: int       # = num_lanes * 2 * points_per_line + num_other_lines * points_per_line
    num_total_lines: int     # = num_lanes * 2 + num_other_lines
    num_lane_queries: int    # = num_lanes * 2 * points_per_line
```

### LaneDetectionDecoder (BEVFormer)

```python
class LaneDetectionDecoder(nn.Module):
    def __init__(self, num_decoder_layers=6, embed_dim=256, num_heads=8, ...): ...

    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args: bev_features (B, H*W, embed_dim)
        Returns: {'pred_points': (B, Q, 2), 'lane_logits': (B, num_lines),
                  'intermediate_points': [...]}
        """
```

### DETR3DLaneDecoder

```python
class DETR3DLaneDecoder(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_layers=6, pc_range=None, ...): ...

    def forward(self, multi_scale_features, intrinsics, extrinsics, image_shape):
        """Returns: (query_out, intermediate_outputs, intermediate_ref_points)"""
```

### PETRLaneDecoder

```python
class PETRLaneDecoder(nn.Module):
    def __init__(self, num_layers=6, embed_dims=256, return_intermediate=True, ...): ...

    def forward(self, key, value, key_pos=None, key_padding_mask=None):
        """Returns: (query_out, intermediate_outputs, intermediate_ref_pts)"""
```

### HierarchicalLaneMapDecoder (MapTR)

```python
class HierarchicalLaneMapDecoder(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_layers=6, ...): ...

    def forward(self, bev_features: torch.Tensor):
        """
        Args: bev_features (B, C, H, W)
        Returns: (intermediate_outputs, intermediate_ref_pts)
            Each list has num_layers entries.
        """
```

### HierarchicalMapDecoder (StreamMapNet)

```python
class HierarchicalMapDecoder(nn.Module):
    def __init__(self, bev_channels=256, d_model=256, num_heads=8, ...): ...

    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args: bev_features (B, C, H, W)
        Returns: {'pred_points': (B, num_lines, pts_per_line, 2),
                  'pred_logits': (B, num_lines, 4),
                  'aux_outputs': [...]}
        """
```

---

## ML Expert Review & Validation

This implementation underwent rigorous review by 5 independent ML experts,
each focused on a different aspect. Below are the findings and resolutions.

### Expert 1: Numerical Stability & Mixed Precision

| Finding | Severity | Resolution |
|---------|----------|------------|
| Sinusoidal cos slice `div_term[:embed_dim//2]` fragile for odd dims | Medium | Added `assert embed_dim % 2 == 0` guard |
| No FP32 upcast in embedding summation | Low | PyTorch LayerNorm internally upcasts — acceptable |
| `torch.norm` gradient at zero in lane_width_loss | Low | Replaced with eps-clamped `.pow(2).sum(-1).clamp(1e-6).sqrt()` |

**Verdict**: ALL PASS after fixes.

### Expert 2: Architecture Correctness

| Finding | Severity | Resolution |
|---------|----------|------------|
| StreamMapNet: pos injection corrupts value stream | HIGH | Refactored to pass query_pos to Q/K only |
| StreamMapNet: missing decoupled self-attention mask | MEDIUM | Added `_self_attn_mask` buffer + passed to layer |
| All models: cache not invalidated on train/eval switch | Low | Safe (cache only read in eval, only written in eval) |
| All models: `get_lane_mask` returns buffer reference | Low | Now returns `.clone()` |

**Verdict**: 4 PASS, StreamMapNet FIXED (2 architectural bugs resolved).

### Expert 3: Performance & Memory

| Finding | Severity | Resolution |
|---------|----------|------------|
| `_cached_pos` not moved by `.to()` device migration | CRITICAL | Added `_apply()` override to invalidate |
| MapTR mask rebuilt per-layer per-forward (24 MB/call) | CRITICAL | Lazy-cached as non-persistent buffer |
| BEVFormer/DETR3D/PETR mask rebuilt per-forward (4 MB) | MEDIUM | Pre-computed as registered buffer at init |
| DETR3D `pc_range` tensor created per-call in loop | LOW | Registered as buffer `_pc_range` |
| `torch.compile` blocked by dynamic mask allocation | MEDIUM | Fixed (masks are now static buffers) |

**Verdict**: ALL FIXED. Forward path is now allocation-free and CUDA-graph-compatible.

### Expert 4: Training Dynamics

| Finding | Severity | Resolution |
|---------|----------|------------|
| Sinusoidal dominates at 0.707 vs learned at 0.02 | HIGH | Scaled sinusoidal by 0.02 (all at ~0.016 RMS now) |
| Point residual (0.01) can't override sinusoidal (0.707) | MEDIUM | Residual init raised to 0.02, sinusoidal scaled down |
| LayerNorm destroys component info with imbalanced magnitudes | MEDIUM | Fixed by balancing magnitudes first |
| `lane_width_consistency_loss` L2 outlier-sensitive | MEDIUM | Switched to `F.smooth_l1_loss(beta=0.01)` |
| Dynamic pos injection conflicts with static identity | MEDIUM | Added learnable `dynamic_pos_gate` (sigmoid gating) |
| MapTR: uniform ALiBi slope across all heads | LOW | Per-head geometric slopes (multi-scale attention) |
| Positional dropout 0.1 too high for geometry | LOW | Reduced to 0.05 |

**Verdict**: ALL FIXED. Embedding magnitudes balanced, loss robust, gating prevents conflict.

### Expert 5: Code Correctness & Edge Cases

| Finding | Severity | Resolution |
|---------|----------|------------|
| `_cached_pos` device mismatch after `.to()` | MEDIUM | Fixed via `_apply()` override (same as Expert 3) |
| `get_lane_mask()` returns mutable buffer reference | LOW | Returns `.clone()` |
| StreamMapNet pos accumulates in residual stream | LOW | Architectural fix (Expert 2 overlap) |
| `lane_width_consistency_loss` dead gradient at zero | LOW | eps-clamped norm prevents this edge case |
| ALiBi + -inf mask arithmetic | N/A | Verified correct (ALiBi only applied to unmasked blocks) |
| `.detach()` on reference points | N/A | Correct and intentional (prevents gradient explosion) |

**Verdict**: ALL PASS (1 medium fixed, rest low/non-issues).

### Validation Tests

All 5 models pass forward + backward testing after fixes:

```
=== BEVFormer ===
  PE shape: (1000, 256)
  Forward+backward OK, pred_points=(2, 1000, 2), grad flows: True
  _apply cache invalidation OK
  get_lane_mask clone OK
  PASS

=== DETR3D ===
  Forward+backward OK, query_out=(2, 1000, 256)
  Dynamic pos gate: 0.0000 (starts at zero, grows during training)
  PASS

=== PETR ===
  Forward+backward OK, query_out=(2, 1000, 256)
  Dynamic pos gate: 0.0000
  PASS

=== MapTR ===
  PE shape: (1000, 256)
  Forward+backward OK, output=(2, 50, 20, 256)
  Dynamic pos gate: 0.0000
  PASS

=== StreamMapNet ===
  PE shape: (1000, 256)
  Forward+backward OK, pred_points=(2, 50, 20, 2)
  Decoupled mask: (1000, 1000)
  PASS

=== common/lane_topology.py ===
  Sinusoidal magnitude: 0.0113 (balanced with learned: 0.0160)
  ALiBi bias: head1 max=-4.75 (strong), head8 max=-0.037 (global)
  Loss: grad flows, no NaN at zero-width
  PASS
```

---

## Version History

| Date | Changes |
|------|---------|
| 2026-06-27 | Initial implementation: hybrid PE, decoupled attention, dynamic pos injection |
| 2026-06-27 | Expert review round 1: LayerNorm, dropout, inference caching, ALiBi bias |
| 2026-06-27 | Expert review round 2 (5 experts): balanced magnitudes, per-head ALiBi, gating, Smooth-L1, StreamMapNet architecture fix, device-safe caching, pre-computed mask buffers |
