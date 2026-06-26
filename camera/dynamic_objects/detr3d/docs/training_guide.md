# Training Guide: DETR3D

This guide explains how to train a DETR3D model from scratch. It is written for readers
who are new to transformer-based 3D object detection. Every concept is introduced from
first principles before diving into implementation details.

---

## Training Fundamentals for Set Prediction

### What Makes Set Prediction Different from Traditional Detection?

Traditional object detection models (such as YOLO, Faster R-CNN, SSD) follow an
anchor-based paradigm:

1. Place thousands of predefined "anchor boxes" densely across the image.
2. Each anchor predicts: (a) whether it contains an object, and (b) a refinement to its
   position and size.
3. After prediction, many anchors overlap on the same object. A post-processing step
   called Non-Maximum Suppression (NMS) removes duplicates by keeping only the
   highest-confidence prediction in each cluster.

```
Traditional Detection Pipeline
================================

  Image --> Backbone --> Dense Anchors --> Predictions --> NMS --> Final Detections
                         (thousands)      (thousands)           (tens)

  Problems:
  - NMS is a hand-crafted heuristic (IoU threshold must be tuned)
  - NMS is not differentiable (breaks gradient flow)
  - Anchors require careful design (sizes, aspect ratios, densities)
```

DETR3D uses a fundamentally different approach called **set prediction**:

1. The model outputs exactly N=900 predictions in parallel (no anchors).
2. During training, Hungarian matching assigns each ground-truth object to exactly one
   prediction slot.
3. The remaining unmatched slots are trained to predict "no object" (background).
4. At inference time, predictions are simply filtered by confidence -- no NMS needed.

```
Set Prediction Pipeline (DETR3D)
==================================

  Image --> Backbone --> Object Queries --> Transformer Decoder --> 900 Predictions
                        (900 learned)       (self-attention        (filter by
                                             prevents duplicates)   confidence)

  Advantages:
  - No NMS required (model learns to suppress duplicates)
  - End-to-end differentiable (gradients flow through everything)
  - No anchor design needed (queries are learned)
  - Global reasoning (every query sees every other query)
```

### How Does the Model Learn to Avoid Duplicates?

The key mechanism is **self-attention between queries**. In each transformer decoder
layer, every query attends to every other query. Through this communication channel,
queries learn to:

- Specialize on different spatial regions or object types
- "Claim" an object and signal other queries to back off
- Avoid predicting the same object twice

This is enforced by the training signal: if two queries both predict the same object,
one of them will be matched to that ground-truth (getting a positive training signal)
while the other will be told it should have predicted "background" (getting a negative
training signal). Over many training iterations, queries learn to coordinate.

### Why 900 Queries?

The number 900 is chosen to be significantly larger than the maximum number of objects
expected in any single scene. In nuScenes driving scenes, a typical frame contains
20-50 annotated objects. By having 900 queries (much more than needed), the model has
ample capacity to detect all objects without running out of "slots."

The trade-off: more queries = more computation, but also more detection capacity. In
practice, most queries learn to predict "no object" for any given scene.

---

## Hungarian Matching

### The Assignment Problem

During training, we have:
- N = 900 model predictions (each with a class probability and a bounding box)
- M ground-truth objects (typically 20-50 per scene, always M << N)

We need to find the **optimal one-to-one assignment** between predictions and
ground-truth objects. This is a classic combinatorial optimization problem called
**bipartite matching**.

```
Bipartite Matching Visualization
=================================

  Predictions (N=900)          Ground Truth (M=3)
  +----------+                 +----------+
  | pred_0   |----cost_00----->| gt_A     |
  | pred_1   |----cost_10----->| (car)    |
  | pred_2   |----cost_20----->+----------+
  | pred_3   |----cost_01----->+----------+
  | ...      |----cost_11----->| gt_B     |
  | pred_899 |----cost_21----->| (ped)    |
  +----------+                 +----------+
                               +----------+
                               | gt_C     |
                               | (truck)  |
                               +----------+

  Goal: Find the assignment of predictions to GT objects that
        minimizes total cost. Each GT gets exactly one prediction.
        The remaining 897 predictions are assigned to "background."
```

### The Matching Cost Function

For each possible pairing (prediction_i, ground_truth_j), we compute a cost:

```
C(i, j) = lambda_cls * C_cls(i, j) + lambda_L1 * C_L1(i, j)
```

Where:
- C_cls(i, j) = -p_i(c_j): Negative predicted probability for the true class of gt_j
- C_L1(i, j) = ||bbox_pred_i - bbox_gt_j||_1: L1 distance between predicted and GT
  bounding box parameters (normalized)
- lambda_cls = 2.0 (classification cost weight)
- lambda_L1 = 0.25 (regression cost weight)

The intuition: a low cost means the prediction is already close to the ground-truth
(high probability for the correct class AND close bounding box). The Hungarian
algorithm finds the assignment that minimizes the total cost across all matched pairs.

### Worked Example: 3 Predictions, 2 Ground-Truth Objects

Consider a training sample with 2 ground-truth objects:
- gt_A: car at position (10.0, 5.0, -1.0), size (4.5, 1.8, 1.5)
- gt_B: pedestrian at position (3.0, 2.0, 0.0), size (0.6, 0.6, 1.7)

And 3 predictions (simplified from 900 for illustration):
- pred_0: predicts car with prob=0.8, position (9.5, 5.2, -0.8)
- pred_1: predicts pedestrian with prob=0.6, position (4.0, 2.5, 0.2)
- pred_2: predicts car with prob=0.3, position (15.0, 8.0, -1.0)

**Step 1: Compute cost matrix (3 x 2)**

For C_cls, we use the negative probability of the correct class:
- pred_0 vs gt_A (car): C_cls = -0.8
- pred_0 vs gt_B (ped): C_cls = -0.05 (pred_0's pedestrian prob is low)
- pred_1 vs gt_A (car): C_cls = -0.1 (pred_1's car prob is low)
- pred_1 vs gt_B (ped): C_cls = -0.6
- pred_2 vs gt_A (car): C_cls = -0.3
- pred_2 vs gt_B (ped): C_cls = -0.02

For C_L1, we compute L1 distance of normalized box parameters:
- pred_0 vs gt_A: ||norm(9.5,5.2,-0.8) - norm(10,5,-1)||_1 = 0.12
- pred_0 vs gt_B: ||norm(9.5,5.2,-0.8) - norm(3,2,0)||_1 = 1.85
- pred_1 vs gt_A: ||norm(4,2.5,0.2) - norm(10,5,-1)||_1 = 1.52
- pred_1 vs gt_B: ||norm(4,2.5,0.2) - norm(3,2,0)||_1 = 0.28
- pred_2 vs gt_A: ||norm(15,8,-1) - norm(10,5,-1)||_1 = 1.20
- pred_2 vs gt_B: ||norm(15,8,-1) - norm(3,2,0)||_1 = 2.95

**Step 2: Total cost matrix** (lambda_cls=2.0, lambda_L1=0.25)

```
              gt_A (car)    gt_B (pedestrian)
pred_0        2*(-0.8) + 0.25*(0.12)  = -1.57    2*(-0.05) + 0.25*(1.85) = 0.36
pred_1        2*(-0.1) + 0.25*(1.52)  =  0.18    2*(-0.6) + 0.25*(0.28) = -1.13
pred_2        2*(-0.3) + 0.25*(1.20)  = -0.30    2*(-0.02) + 0.25*(2.95) = 0.70
```

**Step 3: Hungarian algorithm finds optimal assignment**

The algorithm tries all valid assignments and picks the minimum total cost:
- Assignment 1: pred_0->gt_A, pred_1->gt_B: total = -1.57 + (-1.13) = -2.70
- Assignment 2: pred_0->gt_A, pred_2->gt_B: total = -1.57 + 0.70 = -0.87
- Assignment 3: pred_0->gt_B, pred_1->gt_A: total = 0.36 + 0.18 = 0.54
- Assignment 4: pred_0->gt_B, pred_2->gt_A: total = 0.36 + (-0.30) = 0.06
- Assignment 5: pred_1->gt_A, pred_2->gt_B: total = 0.18 + 0.70 = 0.88
- Assignment 6: pred_1->gt_B, pred_2->gt_A: total = -1.13 + (-0.30) = -1.43

**Optimal: Assignment 1** (pred_0 -> gt_A, pred_1 -> gt_B, cost = -2.70)

Result:
- pred_0 is supervised to detect the car at (10, 5, -1)
- pred_1 is supervised to detect the pedestrian at (3, 2, 0)
- pred_2 is supervised to predict "no object" (background)

### Implementation Details

```python
from scipy.optimize import linear_sum_assignment

# cost_matrix shape: (num_predictions, num_gt_objects)
# Returns (row_indices, col_indices) for optimal assignment
row_ind, col_ind = linear_sum_assignment(cost_matrix)
```

Key points:
- The cost matrix is computed **without gradients** (detached from the computation
  graph). It is used only to determine the assignment, not to compute the loss.
- Matching is performed independently for each sample in the batch.
- The Hungarian algorithm has complexity O(N^3) in the worst case, but since N=900
  and M is small, it runs in milliseconds.

---

## Loss Functions

### Total Loss Formula

After Hungarian matching determines the assignment, we compute the training loss:

```
L_total = L_cls + lambda_bbox * L_bbox + sum(L_aux_l for each decoder layer 1..5)
```

The total loss is the sum of:
1. Classification loss (all 900 predictions)
2. Bounding box regression loss (only matched predictions)
3. Auxiliary losses from intermediate decoder layers

### Focal Loss for Classification

**The problem:** Of 900 predictions, only ~20-50 are matched to ground-truth objects.
The remaining 850-880 are background. If we use standard cross-entropy loss, the
overwhelming number of "easy background" examples dominates the gradient, and the model
never learns to detect objects.

**The solution: Focal Loss** (Lin et al., 2017)

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Where:
- p_t = predicted probability for the correct class
- alpha_t = 0.25 (class balance weight; lower for background to counterbalance its
  frequency)
- gamma = 2.0 (focusing parameter)

**How the focusing parameter gamma works:**

The key term is (1 - p_t)^gamma. This down-weights easy examples (where p_t is high)
and focuses training on hard examples (where p_t is low).

```
Effect of Gamma on Loss Weight
===============================

  p_t (confidence   (1-p_t)^2        Effective      Interpretation
  for correct       (focusing         loss
  class)            factor)           multiplier
  ----------------------------------------------------------------
  0.9 (easy)        (0.1)^2 = 0.01   1% of normal   Almost ignored
  0.8 (easy)        (0.2)^2 = 0.04   4% of normal   Heavily reduced
  0.5 (moderate)    (0.5)^2 = 0.25   25% of normal  Moderate reduction
  0.2 (hard)        (0.8)^2 = 0.64   64% of normal  Slight reduction
  0.1 (very hard)   (0.9)^2 = 0.81   81% of normal  Barely reduced
  0.01 (extreme)    (0.99)^2 = 0.98  98% of normal  Full loss


  Loss
   |
   |  *                          <-- gamma=0 (standard cross-entropy)
   |   *
   |    *
   |     **
   |       ***
   |          *****
   |               **********
   |__________________________ p_t
   0                        1

  Loss
   |
   |  *                          <-- gamma=2 (focal loss)
   |   *
   |    *
   |     *
   |      *
   |       *
   |        **
   |          ***
   |             ********____
   |__________________________ p_t
   0                        1

  With gamma=2, easy examples (high p_t) contribute almost nothing
  to the loss, letting the model focus on hard examples.
```

Configuration:
- Loss weight: lambda_cls = 2.0
- Applied to: all 900 predictions
- Matched predictions have their ground-truth class as target
- Unmatched predictions have "background" as target

### L1 Loss for Bounding Box Regression

The regression loss is a simple L1 (absolute difference) loss between predicted and
ground-truth bounding box parameters:

```
L_bbox = (1/M) * sum(||bbox_pred_i - bbox_gt_matched_i||_1 for i in matched_pairs)
```

**Regression targets (normalized):**

| Parameter | Normalization | Example |
|-----------|--------------|---------|
| cx (center x) | (cx - x_min) / (x_max - x_min) | (10 - (-51.2)) / 102.4 = 0.598 |
| cy (center y) | (cy - y_min) / (y_max - y_min) | (5 - (-51.2)) / 102.4 = 0.549 |
| cz (center z) | (cz - z_min) / (z_max - z_min) | (-1 - (-5)) / 8 = 0.500 |
| w (width) | log(w) | log(4.5) = 1.504 |
| l (length) | log(l) | log(1.8) = 0.588 |
| h (height) | log(h) | log(1.5) = 0.405 |
| yaw | sin(yaw), cos(yaw) | sin(0.3), cos(0.3) |
| vx (velocity x) | raw m/s | 5.0 |
| vy (velocity y) | raw m/s | -2.0 |

Configuration:
- Loss weight: lambda_bbox = 0.25
- Applied to: only matched predictions (foreground objects)

**Why L1 loss instead of Smooth L1?**

Smooth L1 reduces the gradient magnitude for small errors (errors < 1.0 get a
quadratic treatment). While this prevents exploding gradients for large errors, it also
makes the model less sensitive to small localization errors. For precise 3D localization,
we want the model to be equally motivated to fix a 0.1m error as a 1.0m error. L1 loss
provides this uniform gradient:

```
          L1 Loss                    Smooth L1 Loss
  Loss |     /\                Loss |       /
       |    /  \                    |      /
       |   /    \                   |     /
       |  /      \                  |   _/    <-- reduced gradient
       | /        \                 |  /          near zero
       |/          \                | /
  -----+------------ error    -----+------------ error
       0                            0

  L1: gradient is always +1 or -1   Smooth L1: gradient approaches 0 near error=0
  (constant push to reduce error)   (less push to fix small errors)
```

**Why no IoU loss?**

In 2D detection, Generalized IoU (GIoU) loss is common because 2D IoU is cheap to
compute and provides a scale-invariant signal. However, 3D IoU computation is expensive
(requires complex polygon intersection in 3D), and the normalized L1 targets already
account for scale differences through log-scale dimensions. The simpler L1 loss works
well in practice.

### Auxiliary Losses: Supervising Intermediate Decoder Layers

DETR3D has 6 decoder layers. Without auxiliary losses, only the final layer (layer 6)
receives direct supervision. Layers 1-5 only get gradients backpropagated through the
remaining layers, which can lead to vanishing gradients and slow convergence.

**Solution:** Apply the same loss (focal + L1) at every decoder layer independently.

```
Auxiliary Loss Flow
====================

  Layer 1 output --> Hungarian Matching --> Focal + L1 Loss --> L_aux_1
  Layer 2 output --> Hungarian Matching --> Focal + L1 Loss --> L_aux_2
  Layer 3 output --> Hungarian Matching --> Focal + L1 Loss --> L_aux_3
  Layer 4 output --> Hungarian Matching --> Focal + L1 Loss --> L_aux_4
  Layer 5 output --> Hungarian Matching --> Focal + L1 Loss --> L_aux_5
  Layer 6 output --> Hungarian Matching --> Focal + L1 Loss --> L_final

  L_total = L_final + L_aux_1 + L_aux_2 + L_aux_3 + L_aux_4 + L_aux_5
```

Key details:
- Hungarian matching is performed **independently** at each layer (the assignment may
  differ between layers since intermediate predictions are less refined)
- Auxiliary loss weight: 1.0 (same as the final layer, no decay)
- This provides strong gradient signals to early layers, significantly accelerating
  convergence (ablations show ~3-5 NDS improvement from auxiliary losses)

### Loss Weights Summary

| Component | Weight | Applied To | Purpose |
|-----------|--------|-----------|---------|
| Focal classification loss | 2.0 | All 900 predictions | Detect vs. background |
| L1 bounding box loss | 0.25 | Matched predictions only | Localize objects |
| Auxiliary losses (per layer) | 1.0 each | Each decoder layer | Stabilize training |

---

## Optimizer and Learning Rate Schedule

### Why AdamW Over SGD for Transformers?

Transformers have parameters with vastly different gradient magnitudes and scales:
- Backbone convolution weights: gradients are relatively uniform
- Attention QKV projections: can have very large or very small gradients
- Query embeddings: learned from scratch, need aggressive updates
- Layer normalization: small parameters with large relative gradients

SGD uses a single learning rate (scaled by momentum) for all parameters. This creates a
dilemma: a learning rate good for the backbone may be too aggressive for attention
weights, or vice versa.

**Adam** solves this by maintaining per-parameter adaptive learning rates based on the
first and second moments of gradients. Each parameter effectively gets its own learning
rate that adapts to its gradient statistics.

**AdamW** (Adam with decoupled Weight decay) adds proper L2 regularization. In standard
Adam, weight decay is coupled with the gradient, leading to incorrect regularization for
parameters with large gradient magnitudes. AdamW decouples them:

```
Standard Adam:    param = param - lr * (grad + weight_decay * param) / sqrt(v)
                  (weight decay is scaled by adaptive learning rate -- incorrect)

AdamW:            param = param - lr * grad / sqrt(v) - lr * weight_decay * param
                  (weight decay is independent of adaptive rate -- correct)
```

### AdamW Configuration

```python
optimizer = torch.optim.AdamW(
    params=param_groups,
    lr=2e-4,           # Base learning rate
    weight_decay=0.01, # L2 regularization strength
    betas=(0.9, 0.999),# Momentum decay rates (first and second moment)
    eps=1e-8           # Numerical stability
)
```

### Layer-wise Learning Rate (Parameter Groups)

Different model components need different learning rates:

| Parameter Group | Learning Rate | Weight Decay | Rationale |
|----------------|--------------|--------------|-----------|
| Backbone (ResNet-101) | 2e-5 (0.1x base) | 0.01 | Pretrained; large LR would destroy learned features |
| FPN | 2e-4 (base) | 0.01 | Random init; needs to learn from scratch |
| Transformer decoder | 2e-4 (base) | 0.01 | Random init; core learning component |
| Detection heads | 2e-4 (base) | 0.01 | Random init; final prediction layers |
| Query embeddings | 2e-4 (base) | 0.0 | No weight decay (they are learned positions, not weights) |
| Bias terms | 2e-4 (base) | 0.0 | Standard practice: no weight decay on biases |

**Why does the backbone need a lower learning rate?**

The backbone (ResNet-101) is pretrained on ImageNet with millions of images. Its weights
already encode powerful visual features (edges, textures, shapes). If we apply a large
learning rate, these carefully learned features are destroyed in the first few training
iterations, and the model must relearn them from scratch on a much smaller dataset.

By using 0.1x the base learning rate, the backbone is "fine-tuned" -- its features are
gently adapted to the 3D detection task without catastrophic forgetting.

### Cosine Annealing Schedule

The learning rate follows a cosine curve from the peak value down to a minimum:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=24,       # Total epochs
    eta_min=2e-7    # Minimum LR (1/1000 of base)
)
```

Combined with linear warmup:

```
Learning Rate Schedule
=======================

  LR
  2e-4 |         ___
       |        /   \
       |       /     \
       |      /       \
       |     /         \___
       |    /              \___
       |   /                   \____
  2e-7 |__/                         \___
       |__________________________________ epochs
       0  warmup  6    12    18    24
          (500
          iters)

  Phase 1 (warmup): Linear increase from 0 to 2e-4 over 500 iterations
  Phase 2 (cosine): Smooth decay from 2e-4 to 2e-7 over remaining training
```

**Intuition behind this schedule:**

1. **Warmup** (first 500 iterations): Start with a tiny LR to let the randomly
   initialized transformer decoder stabilize. Without warmup, large initial gradients
   can cause divergence.

2. **Peak** (after warmup): The model is stable; apply full learning rate for maximum
   learning speed.

3. **Cosine decay** (rest of training): Gradually reduce the LR. Early in training,
   the model makes large adjustments (learning major patterns). Late in training,
   smaller LR allows fine-grained refinement without overshooting.

4. **Final LR** (2e-7, 1000x smaller than peak): Very gentle updates in the last
   epochs, allowing the model to settle into a good local minimum.

---

## Data Augmentation

Data augmentation artificially increases the diversity of the training data by applying
random transformations. This prevents overfitting and teaches the model to be robust
to variations it will encounter in the real world.

### Random Horizontal Flip (Probability: 0.5)

**What it does:** Mirrors all 6 camera images left-to-right, and correspondingly flips
all annotations.

**Adjustments required:**
- Mirror X-coordinate of all box centers: cx_new = -cx_old
- Negate yaw angle: yaw_new = -yaw_old (or equivalently: pi - yaw_old)
- Negate X-component of velocity: vx_new = -vx_old
- Update camera extrinsic matrices to reflect the mirrored viewing direction

**Why it helps:** Without this augmentation, the model might learn a bias toward
detecting objects on one side of the road (e.g., always expecting oncoming traffic
on the left in right-hand-drive countries). Flipping ensures the model works equally
well regardless of which side objects appear on.

### Random Scale (Range: [0.95, 1.05])

**What it does:** Resizes all camera images by a random scale factor within the range.

**Adjustments required:**
- Scale the camera intrinsic focal lengths: fx_new = fx * scale, fy_new = fy * scale
- No adjustment needed for 3D annotations (they remain in physical coordinates)

**Why it helps:** Objects at different distances appear at different scales in the
image. A car at 10m occupies many pixels; the same car at 50m occupies few pixels.
Scale augmentation teaches the model that the same object can appear at slightly
different sizes, improving robustness to distance variation.

Note: A conservative range [0.95, 1.05] is used because aggressive scaling would make
the projection geometry inconsistent with the augmented intrinsics.

### Random BEV Rotation (Range: [-22.5, 22.5] degrees)

**What it does:** Rotates all annotations and camera extrinsic matrices around the
vertical Z-axis (the "up" direction) by a random angle.

**Adjustments required:**
- Rotate all 3D box centers: apply rotation matrix around Z-axis
- Rotate all yaw angles: yaw_new = yaw_old + rotation_angle
- Rotate velocity vectors: [vx_new, vy_new] = R @ [vx_old, vy_old]
- Update all camera extrinsic matrices: T_new = T_old @ R_z^(-1)

**Why it helps:** In the real world, the ego vehicle approaches intersections,
parking lots, and other scenes from many different angles. Without rotation
augmentation, the model might learn to expect certain spatial arrangements only from
specific viewpoints.

### Color Jitter

**What it does:** Randomly adjusts brightness, contrast, saturation, and hue of each
camera image independently.

**Parameters:**
- Brightness: +/- 0.2
- Contrast: +/- 0.2
- Saturation: +/- 0.2
- Hue: +/- 0.1

**Why it helps:** Real driving encounters diverse lighting conditions:
- Bright sunlight vs. overcast sky
- Direct sun causing glare vs. shadows under bridges
- Artificial lighting at night vs. natural daylight
- Rain making surfaces reflective

Color jitter simulates these variations, teaching the model to rely on shape and
structure rather than specific colors or brightness levels.

### GridMask / Random Erasing (Probability: 0.3)

**What it does:** Erases random rectangular patches in the images, replacing them with
zeros (black).

**Why it helps:**
1. **Simulates occlusion:** In real driving, objects are frequently partially occluded
   by other vehicles, poles, signs, etc. GridMask teaches the model to detect objects
   even when parts are missing.
2. **Forces multi-view usage:** If a region is erased in one camera view, the model
   must rely on other camera views that can see the same 3D location. This strengthens
   the multi-view aggregation mechanism.
3. **Regularization:** Prevents the model from relying too heavily on any single local
   feature.

---

## CBGS: Class-Balanced Grouping and Sampling

### The Problem: Severe Class Imbalance in nuScenes

The nuScenes dataset has extreme imbalance in the number of annotations per class:

```
Class Distribution in nuScenes Training Set (approximate)
==========================================================

  Car                 |================================================| ~400,000
  Pedestrian          |====================|                             ~160,000
  Barrier             |===============|                                  ~120,000
  Traffic Cone        |============|                                     ~100,000
  Truck               |=======|                                          ~60,000
  Trailer             |====|                                             ~35,000
  Bus                 |===|                                              ~25,000
  Motorcycle          |==|                                               ~15,000
  Construction Veh.   |=|                                                ~10,000
  Bicycle             |=|                                                ~8,000

  Ratio of most common to rarest: 400,000 / 8,000 = 50:1
```

**What happens without balancing:** The model sees 50x more cars than bicycles during
training. The loss is dominated by car examples. The model learns to detect cars very
well but fails catastrophically on rare classes (bicycles, construction vehicles).

### The Solution: CBGS (Class-Balanced Grouping and Sampling)

CBGS oversamples training scenes that contain rare classes:

1. **Grouping:** For each training scene, determine which object classes are present.
2. **Counting:** Count how many scenes contain each class.
3. **Oversampling:** Repeat scenes containing rare classes so that each class appears
   approximately the same number of times across an "epoch."

```python
class CBGSDataset:
    """Wraps a dataset to provide class-balanced sampling."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.sample_indices = self._build_balanced_indices()

    def _build_balanced_indices(self):
        # Step 1: Find which classes appear in each sample
        cls_to_samples = defaultdict(list)
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            for cls in sample.classes_present:
                cls_to_samples[cls].append(idx)

        # Step 2: Compute oversampling factors
        max_count = max(len(indices) for indices in cls_to_samples.values())

        # Step 3: Oversample rare classes
        balanced_indices = []
        for cls, indices in cls_to_samples.items():
            repeat_factor = max_count // len(indices)
            balanced_indices.extend(indices * repeat_factor)

        random.shuffle(balanced_indices)
        return balanced_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        real_idx = self.sample_indices[idx]
        return self.dataset[real_idx]
```

### CBGS Impact on Performance

| Setting | NDS | mAP | Training Time | Notes |
|---------|-----|-----|---------------|-------|
| Without CBGS | 0.425 | 0.346 | ~40 hours | Rare classes have very low AP |
| With CBGS | 0.479 | 0.412 | ~60 hours | All classes improve significantly |
| **Improvement** | **+0.054** | **+0.066** | **+50%** | **Essential for competitive results** |

The +5.4 NDS and +6.6 mAP improvement makes CBGS one of the most important training
techniques for nuScenes. The cost is ~50% longer training (because each "epoch" now
contains more samples due to oversampling).

### Class-Level Impact of CBGS

| Class | AP without CBGS | AP with CBGS | Improvement |
|-------|----------------|-------------|-------------|
| Car | 0.58 | 0.62 | +0.04 |
| Bicycle | 0.12 | 0.24 | +0.12 |
| Construction Vehicle | 0.05 | 0.11 | +0.06 |
| Pedestrian | 0.38 | 0.43 | +0.05 |

The largest improvements are on the rarest classes, as expected.

---

## Training Pipeline

### Complete Training Flow Diagram

```
DETR3D Training Pipeline (one iteration)
==========================================

  +-------------------------------------------+
  | 1. DATA LOADING                           |
  |   - Load 6 camera images (900x1600 each)  |
  |   - Load camera intrinsics K (6x 3x3)    |
  |   - Load camera extrinsics [R|t] (6x 4x4)|
  |   - Load ground-truth annotations         |
  +-------------------------------------------+
                    |
                    v
  +-------------------------------------------+
  | 2. DATA AUGMENTATION                      |
  |   - Random horizontal flip (p=0.5)        |
  |   - Random scale [0.95, 1.05]             |
  |   - Random BEV rotation [-22.5, 22.5] deg |
  |   - Color jitter (per camera)             |
  |   - GridMask (p=0.3)                      |
  +-------------------------------------------+
                    |
                    v
  +-------------------------------------------+
  | 3. FORWARD PASS                           |
  |   a. Backbone + FPN: extract features     |
  |      Input: 6 images (900x1600x3)         |
  |      Output: 4 FPN levels per camera      |
  |   b. Initialize 900 object queries        |
  |   c. Predict initial reference points     |
  |   d. Transformer decoder (6 layers):      |
  |      - Self-attention between queries     |
  |      - Project ref points to all cameras  |
  |      - Sample features at projections     |
  |      - FFN transformation                 |
  |      - Refine reference points            |
  |   e. Detection heads at each layer        |
  +-------------------------------------------+
                    |
                    v
  +-------------------------------------------+
  | 4. LOSS COMPUTATION (per decoder layer)   |
  |   a. Run Hungarian matching               |
  |      (900 preds vs M GT objects)          |
  |   b. Compute focal loss (all 900 preds)   |
  |   c. Compute L1 loss (matched preds only) |
  |   d. Sum across all 6 decoder layers      |
  +-------------------------------------------+
                    |
                    v
  +-------------------------------------------+
  | 5. BACKWARD PASS                          |
  |   - Compute gradients for all parameters  |
  |   - Gradient clipping: max_norm = 35.0    |
  +-------------------------------------------+
                    |
                    v
  +-------------------------------------------+
  | 6. OPTIMIZER STEP                         |
  |   - AdamW update with layer-wise LR       |
  |   - LR scheduler step (cosine annealing)  |
  +-------------------------------------------+
                    |
                    v
  +-------------------------------------------+
  | 7. LOGGING                                |
  |   - Total loss, cls loss, bbox loss       |
  |   - Learning rate                         |
  |   - GPU memory usage                      |
  +-------------------------------------------+
```

### Hardware and Timing

| Parameter | Value |
|-----------|-------|
| Batch size per GPU | 1 (= 6 camera images) |
| Number of GPUs | 8 (NVIDIA V100 32GB or A100 40GB) |
| Effective batch size | 8 |
| Epochs | 24 (standard) |
| Keyframes in training set | 28,130 |
| Iterations per epoch | ~3,500 (28,130 / 8) |
| Total iterations | ~84,000 |
| Training time | ~40-48 hours on 8x V100 |
| GPU memory per GPU | ~28 GB (FP16) or ~24 GB (with gradient checkpointing) |
| Mixed precision | FP16 with dynamic loss scaling |

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
```

**Why gradient clipping?**
- In early training iterations, the model makes large errors, producing large gradients
- Rare difficult samples (heavily occluded objects, extreme distances) can produce
  unusually large gradients
- Without clipping, these large gradients can cause the model to make catastrophically
  large parameter updates, destabilizing or diverging training
- The max_norm=35.0 threshold clips the global gradient norm (L2 norm across all
  parameters) to 35.0, preventing any single step from being too large

---

## Distributed Training

### What is Distributed Training?

Training DETR3D on a single GPU would take over two weeks. Distributed Data Parallel
(DDP) training splits the workload across multiple GPUs to reduce wall-clock time.

**How DDP works:**

```
Distributed Data Parallel Training (8 GPUs)
=============================================

  Training Data
  (28,130 keyframes)
        |
        | DataLoader with DistributedSampler
        | (each GPU gets a different 1/8 of the data)
        v
  +-------+-------+-------+-------+-------+-------+-------+-------+
  | GPU 0 | GPU 1 | GPU 2 | GPU 3 | GPU 4 | GPU 5 | GPU 6 | GPU 7 |
  | sample| sample| sample| sample| sample| sample| sample| sample|
  | #0    | #1    | #2    | #3    | #4    | #5    | #6    | #7    |
  +-------+-------+-------+-------+-------+-------+-------+-------+
        |         |         |         |         |         |
        | Forward pass (independent per GPU)
        | Compute loss and gradients
        v         v         v         v         v         v
  +-------+-------+-------+-------+-------+-------+-------+-------+
  | grad  | grad  | grad  | grad  | grad  | grad  | grad  | grad  |
  | GPU 0 | GPU 1 | GPU 2 | GPU 3 | GPU 4 | GPU 5 | GPU 6 | GPU 7 |
  +-------+-------+-------+-------+-------+-------+-------+-------+
        |         |         |         |         |         |
        +----+----+----+----+----+----+----+----+
             | AllReduce (average gradients via NCCL)
             v
  +--------------------------------------------------+
  | Averaged gradient (same on all GPUs)             |
  +--------------------------------------------------+
        |
        | Optimizer step (same update on all GPUs)
        v
  Model weights stay synchronized across all GPUs
```

Key concepts:
- **Each GPU processes a different training sample** (data parallelism)
- **Model weights are identical** across all GPUs at all times
- **Gradients are averaged** across GPUs after each backward pass (AllReduce)
- **Effective batch size** = batch_per_gpu * num_gpus = 1 * 8 = 8

### Launch Commands

```bash
# Modern approach using torchrun (PyTorch >= 1.9)
torchrun --nproc_per_node=8 train.py \
    --config configs/detr3d_r101_nuscenes.yaml \
    --work-dir ./work_dirs/detr3d_r101 \
    --launcher pytorch

# Legacy approach using torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    train.py \
    --config configs/detr3d_r101_nuscenes.yaml \
    --work-dir ./work_dirs/detr3d_r101 \
    --launcher pytorch
```

### Important Distributed Training Considerations

| Topic | Detail |
|-------|--------|
| Communication backend | NCCL (optimized for NVIDIA GPU-to-GPU communication) |
| Batch normalization | Not used in transformer decoder (LayerNorm instead). Backbone BN layers are frozen from pretrained weights. |
| Gradient synchronization | Automatic via DDP wrapper (all-reduce after backward) |
| Random seeds | Different per GPU (ensures different augmentations per sample) |
| Checkpointing | Save from rank 0 only (avoid duplicate writes) |
| Logging | Log from rank 0 only (avoid duplicate log entries) |

### Scaling Rules

If you use a different number of GPUs than 8, adjust accordingly:

| GPUs | Effective Batch Size | Recommended Base LR | Training Time |
|------|---------------------|--------------------|----|
| 4 | 4 | 1e-4 (0.5x) | ~80-96 hours |
| 8 | 8 | 2e-4 (1.0x) | ~40-48 hours |
| 16 | 16 | 4e-4 (2.0x) | ~20-24 hours |

The **linear scaling rule**: when you double the batch size, double the learning rate.
This approximately preserves the magnitude of parameter updates per iteration.

---

## Common Issues and Solutions

### Expanded Troubleshooting Table

| Issue | Symptom | Likely Cause | Solution |
|-------|---------|-------------|----------|
| Training divergence | Loss explodes to NaN or infinity within first 100 iterations | Learning rate too high for initialization | Reduce base LR to 1e-4, increase warmup to 1000 iterations, check for infinite values in data |
| All-background predictions | Model predicts "no object" for all 900 queries; mAP stays at 0 | Classification bias too negative, or matching cost misconfigured | Check cls_head bias initialization (should be ~-4.6, not -10), verify lambda_cls and lambda_L1 weights |
| Duplicate detections | Same object detected by multiple queries; evaluation shows false positives | Self-attention not working effectively | Verify positional encoding is added to self-attention keys/queries, increase decoder layers to 6 |
| Poor rare-class performance | Car AP is 0.55 but bicycle AP is 0.05 | Class imbalance in training data | Enable CBGS, increase training epochs to 36 |
| High mATE (poor localization) | Objects detected but positions are off by >1m | Camera calibration error or projection bug | Verify lidar2img matrices, check coordinate conventions (is Z up or forward?), visualize projected reference points |
| High mAOE (wrong orientations) | Heading predictions are consistently wrong | Sin/cos encoding bug or yaw convention mismatch | Check if yaw is measured from X-axis or Y-axis, verify sin/cos order in regression targets |
| High mAVE (bad velocity) | Velocity predictions are unreliable | Expected for single-frame model; or velocity annotations not loaded correctly | This is inherent to DETR3D (single-frame); verify velocity labels are in correct coordinate frame |
| NaN in loss | NaN appears after some iterations | Numerical instability, often in focal loss log computation | Add epsilon to log: log(p + 1e-7), enable gradient clipping, check for empty GT samples |
| Out of Memory (OOM) | CUDA OOM error during training | Image resolution too high or batch size too large | Reduce resolution (448x800), enable gradient checkpointing, use FP16 |
| Slow convergence | NDS still below 0.30 at epoch 12 | Normal for DETR-style models | Continue training; DETR3D needs 24 full epochs. Performance improves steadily. |
| Training-validation gap | Training loss is low but validation NDS is poor | Overfitting to training data | Reduce augmentation strength, add dropout (0.1), use weight decay |
| Inconsistent results between runs | Different random seeds give very different NDS | Normal variance is ~0.5-1.0 NDS | Average across 3 runs for reliable comparison; ensure deterministic data loading |

### Diagnostic Flowchart

```
Model not training well?
|
+-- Is loss diverging (NaN/Inf)?
|   YES --> Reduce LR, add warmup, check data for invalid values
|
+-- Is loss decreasing but mAP stays at 0?
|   YES --> Check Hungarian matching (are predictions assigned correctly?)
|       --> Check classification bias initialization
|       --> Visualize: are reference points in reasonable locations?
|
+-- Is mAP > 0 but NDS is low?
|   YES --> Check which TP metric is worst
|       --> mATE high? --> Camera calibration problem
|       --> mAOE high? --> Yaw convention mismatch
|       --> mAVE high? --> Expected (single-frame model)
|
+-- Is performance good on cars but bad on rare classes?
    YES --> Enable CBGS, train for more epochs
```

---

## Practical Tips

### Starting from Pretrained Weights

**Backbone initialization is critical:**
- Always use ImageNet-pretrained ResNet-101 (random init does not converge in 24 epochs)
- For best results, use FCOS3D-pretrained backbone (gains ~1 NDS over ImageNet-only)
- FCOS3D pretraining teaches the backbone to extract features useful for 3D tasks

```python
# Load pretrained backbone
backbone_state = torch.load('pretrained/resnet101_imagenet.pth')
model.backbone.load_state_dict(backbone_state, strict=False)

# Or load FCOS3D-pretrained for better initialization
fcos3d_state = torch.load('pretrained/fcos3d_r101_nuscenes.pth')
model.backbone.load_state_dict(
    {k.replace('backbone.', ''): v
     for k, v in fcos3d_state.items() if 'backbone' in k},
    strict=False
)
```

**Detection head initialization:**
```python
# Initialize classification bias to predict background initially
# sigmoid(-4.6) = 0.01 -- model starts by predicting 1% chance of any class
# This prevents initial training instability from overconfident random predictions
nn.init.constant_(model.cls_head[-1].bias, -4.6)
```

**Transformer decoder initialization:**
```python
# Xavier uniform for linear layers in transformer
for p in model.decoder.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```

**Query embedding initialization:**
```python
# Normal distribution for query embeddings
nn.init.normal_(model.query_embedding.weight, mean=0, std=1.0)
```

### Debugging Training

**Monitor these metrics during training:**
1. Total loss (should decrease steadily)
2. Classification loss (should decrease; plateau is OK)
3. Regression loss (should decrease more slowly than cls loss)
4. Number of matched objects per iteration (should be stable ~20-50)
5. Learning rate (verify schedule looks correct)

**Visualization checkpoints (save every 4 epochs):**
- Project reference points onto camera images: are they near objects?
- Visualize predictions: are boxes reasonable in size and orientation?
- Check per-class detection: are all classes being detected?

**Quick sanity checks:**
```python
# Check 1: Reference points should be within detection range
ref_points = model.get_reference_points()  # (900, 3) in [0, 1]
assert ref_points.min() >= 0 and ref_points.max() <= 1

# Check 2: Classification should not be all one class
pred_classes = predictions['cls_scores'].argmax(dim=-1)
unique_classes = pred_classes.unique()
assert len(unique_classes) > 1, "All predictions are same class!"

# Check 3: Bounding box sizes should be reasonable
pred_sizes = predictions['bbox_preds'][:, 3:6].exp()  # log-scale -> actual
assert pred_sizes.max() < 50.0, "Predicted size > 50m is unreasonable"
```

### Memory Optimization Techniques

| Technique | Memory Saved | NDS Impact | Implementation |
|-----------|-------------|-----------|----------------|
| FP16 mixed precision | ~50% | < 0.1 NDS loss | `torch.cuda.amp.autocast()` |
| Gradient checkpointing (backbone) | ~30% | None | `model.backbone.enable_gradient_checkpointing()` |
| Reduced resolution (448x800) | ~75% | ~2 NDS loss | Change config `input_size` |
| Fewer queries (600 instead of 900) | ~10% | ~0.5 NDS loss | Change config `num_queries` |
| Fewer FPN levels (3 instead of 4) | ~15% | ~1 NDS loss | Change config `fpn_levels` |

**Priority order for memory optimization:**
1. Enable FP16 (free performance, negligible accuracy impact)
2. Enable gradient checkpointing (trades compute for memory)
3. Reduce image resolution (significant memory savings with moderate accuracy loss)
4. Reduce queries only as last resort

### Validation During Training

- Evaluate on full validation set every 4 epochs (epochs 4, 8, 12, 16, 20, 24)
- In the final phase (epochs 20-24), evaluate every epoch for fine-grained selection
- Primary selection metric: NDS (combines detection and localization quality)
- Best model typically comes from epochs 22-24
- Keep the last 5 checkpoints to allow post-hoc model selection

```python
# Checkpoint strategy
checkpoint_config = {
    'interval': 1,           # Save every epoch
    'max_keep_ckpts': 5,     # Keep last 5
    'save_best': True,       # Track best NDS
    'best_metric': 'NDS',
}
```

### Training Timeline (What to Expect)

| Epoch Range | Expected NDS | What is Happening |
|-------------|-------------|-------------------|
| 1-4 | 0.15-0.25 | Model learning basic detection; many false positives |
| 5-8 | 0.25-0.35 | Reference points becoming more meaningful; fewer FPs |
| 9-12 | 0.35-0.40 | Localization improving; per-class specialization emerging |
| 13-16 | 0.38-0.42 | Refinement; slow but steady improvement |
| 17-20 | 0.40-0.44 | Fine-tuning; model approaching convergence |
| 21-24 | 0.42-0.48 | Final refinement; best checkpoint usually here |

Do not be alarmed by slow progress in epochs 9-16. DETR-style models have a
characteristically long convergence period compared to anchor-based detectors that
plateau much earlier.

---

## Summary

Training DETR3D requires understanding several interconnected components:

1. **Set prediction** eliminates NMS but requires Hungarian matching
2. **Focal loss** handles the extreme foreground/background imbalance
3. **Auxiliary losses** provide gradient signals to all decoder layers
4. **AdamW with layer-wise LR** respects the pretrained backbone
5. **CBGS** addresses class imbalance (essential for competitive results)
6. **Distributed training** on 8 GPUs reduces wall-clock time to ~2 days

The most common failure mode is poor initialization (especially the classification bias)
or misconfigured camera calibration. When in doubt, visualize reference point projections
to verify the geometric pipeline is correct.
