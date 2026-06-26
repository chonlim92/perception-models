# Evaluation Guide: DETR3D on nuScenes

This guide explains how to evaluate a trained DETR3D model and how to interpret the
results. It is written for readers who are new to 3D object detection evaluation.
Every metric is explained from first principles with worked examples.

---

## What is Model Evaluation?

### Why We Evaluate

Training teaches a model to detect objects. Evaluation measures how well it learned.
We evaluate on data the model has **never seen** during training (the validation set)
to test whether it generalizes to new situations rather than memorizing training
examples.

For autonomous driving, evaluation is especially critical because detection quality
directly impacts safety. A model that misses pedestrians or places detected cars in
the wrong location could lead to dangerous driving decisions.

### The nuScenes Dataset

nuScenes is the standard benchmark for 3D object detection in autonomous driving:

- **1000 driving scenes** recorded in Boston and Singapore
- **6 cameras** providing 360-degree coverage around the vehicle
- **10 object classes:** car, truck, bus, trailer, construction vehicle, pedestrian,
  motorcycle, bicycle, barrier, traffic cone
- **Training set:** 28,130 keyframes (700 scenes)
- **Validation set:** 6,019 keyframes (150 scenes)
- **Test set:** 6,008 keyframes (150 scenes, labels held by organizers)

### The Key Question

The fundamental question evaluation answers is:

> "Can this model detect objects well enough for safe autonomous driving?"

This requires measuring both:
1. **Detection completeness:** Does the model find all objects? (recall)
2. **Detection quality:** Are the detected positions, sizes, orientations accurate? (precision and TP metrics)

---

## NDS: nuScenes Detection Score

### The Formula

NDS is the single number that ranks methods on the nuScenes leaderboard:

```
NDS = (1/10) * [5 * mAP + (1 - min(1, mATE)) + (1 - min(1, mASE))
                        + (1 - min(1, mAOE)) + (1 - min(1, mAVE))
                        + (1 - min(1, mAAE))]
```

### Breaking Down the Formula

The formula has two parts:

**Part 1: Detection ability (50% of NDS)**
- `5 * mAP` captures how many objects the model correctly detects
- mAP ranges from 0 to 1, so this contributes 0 to 5 points

**Part 2: Detection quality (50% of NDS, 10% each)**
- `(1 - min(1, mATE))`: Translation accuracy (0 = bad, 1 = perfect)
- `(1 - min(1, mASE))`: Scale/size accuracy
- `(1 - min(1, mAOE))`: Orientation accuracy
- `(1 - min(1, mAVE))`: Velocity accuracy
- `(1 - min(1, mAAE))`: Attribute accuracy

Each TP metric is an error (lower is better). Subtracting from 1 converts to a score
(higher is better). The `min(1, ...)` clamp prevents any single catastrophic metric
from contributing negatively.

### Worked Example: Computing DETR3D's NDS

Let us compute NDS for DETR3D (ResNet-101, with CBGS) step by step:

**Given metrics:**
- mAP = 0.412
- mATE = 0.641 m
- mASE = 0.255
- mAOE = 0.394 rad
- mAVE = 0.845 m/s
- mAAE = 0.133

**Step 1: Compute detection contribution**
```
5 * mAP = 5 * 0.412 = 2.060
```

**Step 2: Compute TP quality contributions**
```
(1 - min(1, 0.641)) = 1 - 0.641 = 0.359   (translation)
(1 - min(1, 0.255)) = 1 - 0.255 = 0.745   (scale)
(1 - min(1, 0.394)) = 1 - 0.394 = 0.606   (orientation)
(1 - min(1, 0.845)) = 1 - 0.845 = 0.155   (velocity)
(1 - min(1, 0.133)) = 1 - 0.133 = 0.867   (attribute)
```

**Step 3: Sum and normalize**
```
Sum = 2.060 + 0.359 + 0.745 + 0.606 + 0.155 + 0.867 = 4.792
NDS = (1/10) * 4.792 = 0.4792
```

**Result: NDS = 0.479**

### What Does NDS = 0.479 Mean?

To put this in context:

| NDS Range | Interpretation | Example Methods |
|-----------|---------------|-----------------|
| 0.30-0.40 | Early camera-only methods | FCOS3D, PGD |
| 0.40-0.50 | Competitive camera-only | DETR3D, PETR |
| 0.50-0.60 | Strong camera-only | BEVFormer, StreamPETR |
| 0.60-0.70 | State-of-the-art camera | Latest temporal methods |
| 0.70-0.75 | LiDAR-based methods | CenterPoint, TransFusion |
| 0.75-0.80 | Best LiDAR + Camera fusion | BEVFusion |

DETR3D at NDS=0.479 is a solid camera-only baseline that proved the concept of
query-based 3D detection. It is not production-ready for autonomous driving but
established the architectural foundation for stronger methods.

### Why Not Just Use mAP?

mAP only measures whether objects are detected (recall) and whether detections are
correct (precision). It says nothing about the quality of those detections.

Consider two models:
- Model A: Detects a car 3.9m from its true position (counts as TP at 4m threshold)
- Model B: Detects the same car 0.2m from its true position

Both get the same mAP credit, but Model B is far more useful for autonomous driving.
NDS captures this difference through the TP metrics.

---

## mAP: Mean Average Precision

### Key Difference from 2D Detection: Center Distance Matching

In 2D object detection (COCO, Pascal VOC), a prediction is matched to a ground-truth
object based on Intersection over Union (IoU). In nuScenes 3D detection, matching uses
**BEV center distance** instead:

```
Match criterion: Euclidean distance in the Bird's Eye View (X-Y) plane

  distance = sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)

  Note: Height (Z-axis) is NOT used for matching.
```

**Why center distance instead of 3D IoU?**
1. 3D IoU computation is expensive (complex polygon intersection in 3D space)
2. Center distance is more intuitive for driving: "how far off is the predicted
   position from the true position?"
3. Height matters less for driving decisions than horizontal position
4. Center distance is continuous and well-behaved (no sharp transitions like IoU)

### The Four Distance Thresholds

AP is computed at 4 different distance thresholds and averaged:

```
AP_class = (1/4) * [AP@0.5m + AP@1.0m + AP@2.0m + AP@4.0m]
```

| Threshold | Value | Scenario | Difficulty |
|-----------|-------|----------|-----------|
| d1 | 0.5 meters | Very precise (parking, low-speed) | Hardest |
| d2 | 1.0 meters | Standard urban driving | Hard |
| d3 | 2.0 meters | Moderate tolerance | Moderate |
| d4 | 4.0 meters | Generous (detection awareness) | Easiest |

**Why multiple thresholds?** Different driving scenarios require different precision:
- Parking in a tight spot requires sub-meter accuracy
- Highway driving at 120 km/h can tolerate larger errors for distant objects
- Averaging across thresholds gives a balanced assessment

### How mAP is Computed

```
mAP = (1/10) * sum(AP_class for each of 10 classes)
```

For each class at each threshold:

1. **Rank all predictions** by confidence score (highest first)
2. **Match predictions to ground-truth** objects greedily:
   - For each prediction (in confidence order), find the nearest unmatched GT
   - If distance < threshold: True Positive (TP)
   - If distance >= threshold or no GT available: False Positive (FP)
3. **Compute precision and recall** at each confidence cutoff
4. **Compute AP** as area under the precision-recall curve (101-point interpolation)

### Matching Rules

- **One-to-one:** Each ground-truth object can match at most one prediction
- **Greedy by confidence:** Higher-confidence predictions get first pick
- **Distance tie-breaking:** If two predictions are equidistant, the higher confidence wins
- **No duplicate matching:** Once a GT is matched, it is removed from the pool

### Worked Example: 5 Predictions, 3 Ground-Truth Cars

**Setup:**
- 3 ground-truth cars: GT_A at (10, 5), GT_B at (20, 8), GT_C at (35, -3)
- 5 model predictions (ranked by confidence):

| Prediction | Position | Confidence | Nearest GT | Distance |
|-----------|----------|-----------|-----------|----------|
| P1 | (10.3, 5.1) | 0.95 | GT_A | 0.32m |
| P2 | (19.5, 7.8) | 0.85 | GT_B | 0.54m |
| P3 | (25.0, 10.0) | 0.70 | GT_B | 5.39m |
| P4 | (34.2, -2.5) | 0.60 | GT_C | 0.94m |
| P5 | (50.0, 0.0) | 0.30 | - | >10m |

**Evaluation at threshold = 2.0m:**

Processing predictions in confidence order:

| Step | Prediction | Match | Result | Precision | Recall |
|------|-----------|-------|--------|-----------|--------|
| 1 | P1 (conf=0.95) | GT_A (0.32m < 2m) | TP | 1/1 = 1.00 | 1/3 = 0.33 |
| 2 | P2 (conf=0.85) | GT_B (0.54m < 2m) | TP | 2/2 = 1.00 | 2/3 = 0.67 |
| 3 | P3 (conf=0.70) | GT_B already matched, no other GT within 2m | FP | 2/3 = 0.67 | 2/3 = 0.67 |
| 4 | P4 (conf=0.60) | GT_C (0.94m < 2m) | TP | 3/4 = 0.75 | 3/3 = 1.00 |
| 5 | P5 (conf=0.30) | No GT within 2m | FP | 3/5 = 0.60 | 3/3 = 1.00 |

**Precision-Recall curve:**
```
Precision
1.0  |  *---*
     |       \
0.75 |        *
     |         \
0.67 |          *
0.60 |           *
     |
0.0  |_________________________
     0    0.33  0.67  1.0  Recall
```

**AP at 2.0m threshold** = area under this curve (using interpolation) = approximately 0.89

This would be repeated for all 4 thresholds (0.5m, 1.0m, 2.0m, 4.0m), then averaged to
get AP_car.

---

## True Positive (TP) Metrics

### What Are TP Metrics?

Once we know which predictions are True Positives (correctly matched to ground-truth
objects), we measure the **quality** of those detections. A prediction can be "correct"
(within the distance threshold) but still be imprecise.

TP metrics answer: "For the objects we DID detect, how accurately did we detect them?"

### mATE: Mean Average Translation Error

**Formula:**
```
mATE = (1/|TP|) * sum(||center_pred - center_gt||_2 for each TP)
```

**Unit:** Meters (BEV Euclidean distance)

**DETR3D value:** 0.641 m

**Real-world interpretation:**

> "On average, detected objects are positioned 64cm from their true location."

Is this good enough for driving?

```
What 0.64m error looks like:
==============================

  True car position     Predicted position
       [====]                [====]
       ^                     ^
       |<--- 0.64m --->|

  A typical car is ~1.8m wide.
  So 0.64m error = about 1/3 of a car width.

  For highway driving at 120 km/h: Marginal -- planning needs better accuracy
  For urban driving at 50 km/h:    Acceptable for most scenarios
  For parking at 5 km/h:           Not precise enough for tight spaces
```

**Why DETR3D has relatively high mATE:**
Camera-based depth estimation is inherently ambiguous. Without LiDAR's direct distance
measurement, the model must infer depth from visual cues (perspective, object size,
texture gradients). This is particularly difficult for distant objects.

### mASE: Mean Average Scale Error

**Formula:**
```
mASE = (1/|TP|) * sum(1 - IoU_3D(pred_box_aligned, gt_box_aligned) for each TP)
```

Where boxes are aligned (same center, same orientation) to isolate size error.

**Unit:** Dimensionless, range [0, 1], where 0 is perfect

**DETR3D value:** 0.255

**Interpretation:**

> "Predicted box sizes overlap about 74.5% with true sizes (1 - 0.255 = 0.745 IoU)."

This is actually quite good. Object sizes are relatively constrained:
- Cars are always approximately 4.5m x 1.8m x 1.5m
- Pedestrians are always approximately 0.6m x 0.6m x 1.7m
- The model learns these typical sizes quickly and produces accurate size predictions

**Why mASE is usually not the bottleneck:** Object classes have consistent physical
dimensions. A car detected as 4.3m long instead of 4.5m is a 4% error -- acceptable
for most planning scenarios.

### mAOE: Mean Average Orientation Error

**Formula:**
```
mAOE = (1/|TP|) * sum(|angle_diff(yaw_pred, yaw_gt)| for each TP)
```

Where angle_diff computes the smallest angular difference (handling wraparound).

**Unit:** Radians

**DETR3D value:** 0.394 rad = approximately 22.6 degrees

**Interpretation:**

> "Heading predictions are off by about 23 degrees on average."

```
What 23-degree orientation error looks like:
=============================================

  True heading:        Predicted heading:

      ^                     /
      |                    /
      |  (car facing      /   (23 degrees off)
      |   this way)      /
     [CAR]             [CAR]

  For a car moving at 50 km/h (14 m/s):
  - In 2 seconds, it travels 28 meters
  - With 23-degree error, the predicted endpoint is:
    28 * sin(23) = 11 meters off from the true endpoint
  - This is a significant error for trajectory prediction!
```

**Why orientation is hard:** Camera images show objects from specific viewpoints.
A car seen from directly behind is harder to orient than one seen from the side.
Also, small and distant objects have few pixels to determine heading from.

**Symmetry handling:** For barriers and traffic cones (roughly symmetric), orientation
error is computed modulo 180 degrees (pi radians).

### mAVE: Mean Average Velocity Error

**Formula:**
```
mAVE = (1/|TP|) * sum(||vel_pred - vel_gt||_2 for each TP)
```

**Unit:** Meters per second (m/s)

**DETR3D value:** 0.845 m/s = approximately 3.0 km/h

**Interpretation:**

> "Velocity predictions are off by about 3 km/h on average."

This is DETR3D's **weakest metric** relative to temporal methods. The reason is
fundamental: DETR3D is a **single-frame model**. It processes one timestamp of images
with no temporal context. It cannot directly observe motion -- it can only infer
velocity from indirect visual cues:

- Motion blur
- Wheel orientation (turning vs. straight)
- Relative positioning in subsequent frames (not available to single-frame model)
- Brake lights illuminated (deceleration)

Compare to BEVFormer with temporal fusion: mAVE = 0.304 m/s (~1 km/h) -- temporal
observation of actual motion across frames is much more accurate.

**Classes evaluated:** Velocity is only meaningful for moving objects:
- Evaluated: car, truck, bus, trailer, construction vehicle, pedestrian, motorcycle, bicycle
- NOT evaluated: barrier, traffic cone (static objects)

### mAAE: Mean Average Attribute Error

**Formula:**
```
mAAE = (1/|TP|) * sum(1{attr_pred != attr_gt} for each TP)
```

**Unit:** Dimensionless (1 - accuracy), range [0, 1]

**DETR3D value:** 0.133

**Interpretation:**

> "87% of detected objects have correctly predicted activity state."

Attributes describe what objects are doing:
- **Vehicles:** moving / stopped / parked
- **Pedestrians:** moving / standing / sitting
- **Cycles:** with_rider / without_rider

This is relatively easy to predict because attributes correlate strongly with visual
appearance (parked cars are aligned with curbs, moving pedestrians show stride pose).

---

## Per-Class Analysis

### Why Some Classes Are Harder Than Others

| Class | AP | mATE | Key Challenges |
|-------|----|----|-------|
| Car | 0.56 | 0.58m | Most common (400K+ annotations); consistent shape; large size; many training examples |
| Traffic Cone | 0.51 | 0.45m | Distinctive orange color and conical shape; small but very consistent appearance |
| Barrier | 0.46 | 0.55m | Large, static, consistent rectangular shape; often in groups (easier to detect) |
| Pedestrian | 0.43 | 0.65m | Common but small; highly variable appearance (clothing, pose); articulated body |
| Bus | 0.39 | 0.70m | Large and distinctive; but rare and often partially occluded (only part visible) |
| Truck | 0.33 | 0.72m | Moderate frequency; wide variety of truck types (box truck, flatbed, tanker) |
| Motorcycle | 0.33 | 0.68m | Small, thin profile; often occluded by rider; similar to bicycle from some angles |
| Bicycle | 0.24 | 0.72m | Very small, thin wire-frame structure; extremely rare (8K annotations); easily missed |
| Trailer | 0.20 | 0.95m | Very large (hard to localize center from partial views); rare; often largely occluded |
| Construction Vehicle | 0.11 | 0.88m | Extremely rare (~10K); huge shape variation (crane, excavator, bulldozer, etc.) |

### Three Factors That Determine Performance

**1. Training data frequency** -- More examples = better learned representation
```
  AP vs. Training Annotations (approximate relationship)
  ====================================================

  AP
  0.6 |                                        * Car
      |
  0.5 |                          * Traffic Cone
      |                     * Barrier
  0.4 |                * Pedestrian
      |          * Bus
  0.3 |     * Truck  * Motorcycle
      |
  0.2 |  * Bicycle    * Trailer
      |
  0.1 |  * Construction Vehicle
      |___________________________________________
      8K    25K   60K  100K  160K       400K  Annotations
```

**2. Object physical size** -- Larger objects occupy more pixels = more information
- Car (4.5m): ~100-200 pixels wide at 20m distance
- Pedestrian (0.6m): ~15-30 pixels wide at 20m distance
- Traffic cone (0.4m): ~10-20 pixels wide at 20m distance (but very distinctive color)

**3. Intra-class variability** -- Consistent shapes are easier to learn
- Car: All cars look roughly similar (box on wheels)
- Construction vehicle: Crane looks nothing like a bulldozer

---

## Distance Analysis

### Why Performance Degrades with Range

| Distance | Approx. mAP | Approx. mATE | Key Factors |
|----------|-------------|-------------|-------------|
| 0-10m | ~0.65 | ~0.30m | High pixel resolution, multiple camera overlap, strong features |
| 10-20m | ~0.55 | ~0.50m | Good quality, primary driving range |
| 20-30m | ~0.40 | ~0.70m | Significant pixel reduction, depth ambiguity increases |
| 30-40m | ~0.25 | ~1.00m | Only large objects reliably detected |
| 40-50m | ~0.15 | ~1.30m | Very few pixels per object, mostly large vehicles |
| >50m | ~0.05 | ~1.80m | Barely detectable, extreme depth uncertainty |

### The Physics Behind Distance Degradation

**Pixel coverage shrinks inversely with distance:**

```
Object pixel width = (focal_length * object_width) / distance

Example: Car (width=1.8m), focal_length=1000 pixels

  Distance    Pixel Width    Relative to 10m
  --------    -----------    ---------------
  10m         180 pixels     1.0x
  20m          90 pixels     0.5x
  30m          60 pixels     0.33x
  40m          45 pixels     0.25x
  50m          36 pixels     0.20x
  100m         18 pixels     0.10x
```

At 50m, a car is only 36 pixels wide. At that resolution, it is difficult to
distinguish a car from a truck, estimate its orientation, or precisely locate its center.

**Depth ambiguity grows quadratically:**

Camera projection is fundamentally ambiguous about depth. An object at distance d
projects to the same pixel location as a smaller object at distance d/2. The model must
resolve this ambiguity using contextual cues (perspective lines, known object sizes,
ground plane). These cues become weaker with distance.

**Camera coverage decreases:**

```
Multi-Camera Coverage at Different Distances
=============================================

  Close (5m):   Visible in 2-3 cameras (overlap zones)
                --> Multi-view aggregation helps localization

  Medium (20m): Visible in 1-2 cameras
                --> Some multi-view benefit

  Far (50m):    Visible in exactly 1 camera
                --> No multi-view aggregation possible
                --> Single-view depth estimation only
```

---

## Comparison with Other Methods

### Camera-Only Methods on nuScenes Val Set

| Method | Year | Backbone | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE | Key Innovation |
|--------|------|----------|-----|-----|------|------|------|------|------|---------------|
| DETR3D | 2021 | R101-DCN | 0.479 | 0.412 | 0.641 | 0.255 | 0.394 | 0.845 | 0.133 | 3D-to-2D projection |
| PETR | 2022 | R101-DCN | 0.504 | 0.441 | 0.593 | 0.249 | 0.383 | 0.808 | 0.132 | 3D position embeddings |
| BEVDet | 2022 | R101 | 0.488 | 0.424 | 0.524 | 0.242 | 0.373 | 0.950 | 0.148 | Explicit view transform |
| BEVFormer | 2022 | R101-DCN | 0.517 | 0.448 | 0.582 | 0.256 | 0.375 | 0.378 | 0.126 | Spatial cross-attention |
| BEVFormer-T | 2022 | R101-DCN | 0.569 | 0.481 | 0.545 | 0.247 | 0.362 | 0.304 | 0.120 | + Temporal fusion |

### Interpreting the Comparison

**DETR3D vs. PETR (+2.5 NDS):**
PETR replaces DETR3D's explicit geometric projection with learned 3D position
embeddings added to image features. This provides a slight advantage because the model
can learn to compensate for calibration inaccuracies, but loses the explicit geometric
prior.

**DETR3D vs. BEVDet (+0.9 NDS):**
BEVDet constructs an explicit BEV representation via a view transform module (predicting
depth distributions). It has much better mATE (0.524 vs 0.641) because the explicit
depth prediction helps localization. However, it has worse mAVE (0.950 vs 0.845) --
the BEV construction step adds noise that hurts velocity estimation.

**DETR3D vs. BEVFormer (+3.8 NDS):**
BEVFormer uses a dense BEV grid with spatial cross-attention, giving it explicit spatial
structure. Its biggest advantage is temporal fusion (dramatically reducing mAVE from
0.845 to 0.378 then 0.304), which requires observing the same objects across time.

**Key takeaway:** DETR3D's main limitations are:
1. **No temporal information** -> high mAVE (velocity error)
2. **No explicit depth estimation** -> moderate mATE (translation error)
3. **No dense BEV representation** -> cannot easily extend to other tasks (planning, motion prediction)

---

## Running Evaluation

### Prerequisites

```bash
# Install the official nuScenes development kit
pip install nuscenes-devkit==1.1.10
```

### Step-by-Step Evaluation Pipeline

```bash
# Step 1: Generate predictions on the validation set
# This runs the trained model on all 6019 validation keyframes
python tools/test.py \
    configs/detr3d/detr3d_res101_gridmask.py \
    checkpoints/detr3d_r101_cbgs_24e.pth \
    --eval bbox \
    --out results/detr3d_val_predictions.pkl

# Step 2: Convert predictions to official nuScenes submission format
# The official evaluator requires a specific JSON format
python tools/convert_results.py \
    --results results/detr3d_val_predictions.pkl \
    --output results/detr3d_results_nusc.json

# Step 3: Run official nuScenes evaluation
# This computes all metrics (mAP, NDS, per-class APs, TP metrics)
python tools/nusc_eval.py \
    --result_path results/detr3d_results_nusc.json \
    --eval_set val \
    --dataroot data/nuscenes/ \
    --output_dir eval_results/
```

### Expected Output

```
=== nuScenes Detection Evaluation ===
Evaluation time: 120.5 seconds

=== Per-class results ===
  car:                  AP: 0.562, ATE: 0.580, ASE: 0.160, AOE: 0.120, AVE: 0.900, AAE: 0.150
  truck:                AP: 0.327, ATE: 0.720, ASE: 0.230, AOE: 0.200, AVE: 0.800, AAE: 0.220
  bus:                  AP: 0.389, ATE: 0.700, ASE: 0.200, AOE: 0.080, AVE: 1.100, AAE: 0.180
  trailer:              AP: 0.200, ATE: 0.950, ASE: 0.240, AOE: 0.550, AVE: 0.500, AAE: 0.100
  construction_vehicle: AP: 0.112, ATE: 0.880, ASE: 0.480, AOE: 1.050, AVE: 0.120, AAE: 0.350
  pedestrian:           AP: 0.425, ATE: 0.650, ASE: 0.280, AOE: 0.620, AVE: 0.850, AAE: 0.200
  motorcycle:           AP: 0.328, ATE: 0.680, ASE: 0.260, AOE: 0.550, AVE: 1.100, AAE: 0.120
  bicycle:              AP: 0.240, ATE: 0.720, ASE: 0.280, AOE: 0.700, AVE: 0.450, AAE: 0.020
  barrier:              AP: 0.463, ATE: 0.550, ASE: 0.300, AOE: 0.150, AVE: NaN,   AAE: NaN
  traffic_cone:         AP: 0.511, ATE: 0.450, ASE: 0.340, AOE: NaN,   AVE: NaN,   AAE: NaN

=== Overall metrics ===
  mAP:  0.4120
  mATE: 0.6410
  mASE: 0.2550
  mAOE: 0.3940
  mAVE: 0.8450
  mAAE: 0.1330
  NDS:  0.4790
```

### Result File Format

Predictions must be submitted as a JSON file with this structure:

```json
{
  "meta": {
    "use_camera": true,
    "use_lidar": false,
    "use_radar": false,
    "use_map": false,
    "use_external": false
  },
  "results": {
    "<sample_token>": [
      {
        "sample_token": "abc123...",
        "translation": [10.5, 5.2, -0.8],
        "size": [4.5, 1.8, 1.5],
        "rotation": [0.707, 0.0, 0.0, 0.707],
        "velocity": [5.0, -2.0],
        "detection_name": "car",
        "detection_score": 0.95,
        "attribute_name": "vehicle.moving"
      }
    ]
  }
}
```

**Field descriptions:**
| Field | Format | Description |
|-------|--------|-------------|
| translation | [x, y, z] | 3D center in global coordinates (meters) |
| size | [w, l, h] | Width, length, height (meters) |
| rotation | [qw, qx, qy, qz] | Orientation as quaternion |
| velocity | [vx, vy] | Velocity in X and Y (m/s, global frame) |
| detection_name | string | One of 10 class names |
| detection_score | float [0, 1] | Model confidence |
| attribute_name | string | Activity state (e.g., "vehicle.moving") |

### Evaluation Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max predictions per sample | 500 | Predictions beyond this are discarded |
| Max evaluation distance | 50m (most classes), 60m (trucks/buses) | Objects beyond this are ignored |
| Minimum confidence | None | All predictions evaluated via PR curve |
| Coordinate frame | Global | All predictions must be in global (world) frame |
| Matching criterion | BEV center distance | Height (Z) not used for matching |

---

## Debugging Poor Results

### Systematic Debugging Approach

When evaluation results are worse than expected, follow this systematic approach:

**Step 1: Identify the worst metric**

| Metric | Expected | Warning | Likely Problem Area |
|--------|----------|---------|---------------------|
| mAP | > 0.35 | < 0.30 | Detection backbone or feature extraction |
| mATE | < 0.70 | > 0.80 | Camera calibration or projection pipeline |
| mASE | < 0.27 | > 0.35 | Size regression targets or normalization |
| mAOE | < 0.40 | > 0.50 | Orientation encoding (sin/cos) or yaw convention |
| mAVE | < 0.90 | > 1.20 | Velocity annotation loading or coordinate frame |
| mAAE | < 0.15 | > 0.30 | Attribute label encoding or class-conditional logic |

**Step 2: Check per-class results**

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| All classes uniformly low | Backbone or feature extraction problem | Verify pretrained weights loaded correctly |
| Only rare classes low | Class imbalance | Enable CBGS |
| One class catastrophically bad | Class-specific bug | Check that class's annotation loading |
| Close objects good, far objects bad | Resolution or depth issue | Expected behavior for cameras |

**Step 3: Visualize predictions**

The most effective debugging tool is visual inspection:

```bash
# Visualize predictions overlaid on camera images
python scripts/visualize_results.py \
    --predictions results/val_predictions.pkl \
    --infos ./data/nuscenes/infos/detr3d_infos_val.pkl \
    --data-root ./data/nuscenes \
    --output-dir ./debug_vis/ \
    --mode camera \
    --show-gt  # Show ground-truth boxes alongside predictions
```

**What to look for:**
- Are predicted boxes systematically shifted in one direction? (calibration error)
- Are predictions clustered in certain image regions? (bias in learned queries)
- Are orientations always wrong by ~90 or 180 degrees? (convention mismatch)
- Are there many missed objects at specific distances? (FPN level assignment issue)

**Step 4: Check the data pipeline**

```python
# Verify camera calibration matrices
for cam in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']:
    lidar2img = sample_info['lidar2img'][cam]  # Should be 4x4 matrix
    assert lidar2img.shape == (4, 4)
    # Project a known 3D point and verify it lands in the expected camera
    test_point = np.array([10, 0, 0, 1])  # 10m ahead, should be in FRONT camera
    pixel = lidar2img @ test_point
    pixel = pixel[:2] / pixel[2]
    assert 0 < pixel[0] < image_width, f"Point ahead should be in FRONT camera!"
```

### Common Coordinate Convention Issues

The most frequent source of bugs is mismatched coordinate conventions:

```
nuScenes Coordinate Systems
============================

  Global/LiDAR frame:       Camera frame:
      Z (up)                    Y (down)
      |                         |
      |                         |
      +--- Y (left)             +--- X (right)
     /                         /
    X (forward)               Z (forward/depth)

  Common mistakes:
  - Mixing up X (forward in LiDAR) with Z (forward in camera)
  - Forgetting that camera Y points DOWN
  - Using wrong rotation convention (intrinsic vs. extrinsic)
```

---

## Evaluation for Deployment

### What Metrics Matter for a Real Self-Driving Car?

Academic benchmarks like nuScenes measure average performance across all scenarios. For
deployment, we care about **worst-case** performance in **safety-critical** situations.

### Safety-Critical Metrics

**1. Recall at close range (0-20m) -- MUST be near 100%**

Missing a pedestrian at 10m means the car has less than 1 second to react at urban
speeds. The nuScenes mAP metric does not explicitly measure close-range recall, but
this is the most important metric for deployment:

```
Required recall for safe driving:

  Distance    Speed     Reaction Time    Required Recall
  --------    -----     -------------    ---------------
  0-5m        Any       < 0.3s           > 99.9%
  5-10m       30 km/h   ~1.2s            > 99.5%
  10-20m      50 km/h   ~1.4s            > 99.0%
  20-30m      80 km/h   ~1.0s            > 98.0%
  30-50m      120 km/h  ~1.0s            > 95.0%
```

**2. mATE at close range -- must be < 0.3m**

For close-range objects, position errors directly translate to collision risk. If a
pedestrian is detected 50cm to the left of their true position, the car might drive
too close.

**3. mAVE -- critical for trajectory prediction**

Velocity accuracy determines how well the planner can predict where objects will be in
the future. For a car at 50 km/h (14 m/s), a velocity error of 0.85 m/s means the
predicted position 2 seconds ahead is off by 1.7m.

**4. False positive rate -- must be extremely low**

A phantom detection (false positive) in the ego lane can cause unnecessary emergency
braking. At highway speed, this is dangerous for following vehicles. The acceptable
false positive rate is < 1 per 10,000 km driven.

### Beyond NDS: Deployment Considerations

| Factor | NDS Captures? | Real-World Impact |
|--------|--------------|-------------------|
| Average detection quality | Yes | Baseline capability |
| Worst-case misses (pedestrians) | Partially (via recall in mAP) | Safety-critical |
| False positives causing braking | No (mAP penalizes FPs on average) | Safety-critical |
| Weather/lighting robustness | No | Essential for all-weather operation |
| Latency (must run at 10+ Hz) | No | Real-time requirement |
| Edge cases (unusual objects) | No | Long-tail safety |
| Degradation on dirty cameras | No | Sensor maintenance |
| Confidence calibration | No | Planning trust level |

### DETR3D's Deployment Readiness

At NDS = 0.479, DETR3D is **NOT ready for production autonomous driving**. Here is why:

| Requirement | Threshold | DETR3D | Gap |
|-------------|-----------|--------|-----|
| Close-range recall | > 99% | ~95% | 4% gap (misses 1 in 20 close objects) |
| mATE | < 0.3m | 0.641m | 2x too high |
| mAVE | < 0.5 m/s | 0.845 m/s | 1.7x too high |
| NDS overall | > 0.65 | 0.479 | 0.17 gap |
| Latency (V100) | < 100ms | ~150ms | 1.5x too slow |

**DETR3D's value is architectural, not operational:**
- It proved that query-based camera detection works
- It established the 3D-to-2D projection paradigm
- It enabled BEVFormer, StreamPETR, and other production-closer methods
- Production systems build on DETR3D's ideas but add: temporal fusion, larger backbones,
  test-time augmentation, sensor fusion, and extensive engineering

### What Would Be Needed for Production?

1. **Temporal fusion** (observe objects across multiple frames) -> reduce mAVE to < 0.3 m/s
2. **Larger/better backbone** (Swin-L, InternImage) -> improve mAP and mATE
3. **Multi-sensor fusion** (camera + LiDAR + radar) -> robust depth and velocity
4. **Test-time augmentation** -> improve recall
5. **Extensive validation** (millions of miles of simulation + real-world testing)
6. **Confidence calibration** -> planning can trust the model's uncertainty estimates
7. **Redundancy** -> multiple independent detection systems that cross-check each other

---

## Summary

Evaluating DETR3D on nuScenes involves understanding:

1. **NDS** combines detection (mAP) and quality (5 TP metrics) into one score
2. **mAP** measures detection using BEV center distance matching at 4 thresholds
3. **TP metrics** measure how precisely matched detections are localized, sized, oriented
4. **Per-class analysis** reveals which objects are hardest (rare + small + variable)
5. **Distance analysis** shows fundamental camera limitations at range
6. **Deployment evaluation** requires much stricter thresholds than academic benchmarks

DETR3D at NDS=0.479 is a strong research baseline that established important
architectural paradigms. Closing the gap to production-ready detection requires
temporal information, multi-sensor fusion, and extensive engineering beyond the core
detection model.
