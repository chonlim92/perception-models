# Evaluation Guide: DETR3D on nuScenes

## Overview

DETR3D is evaluated using the official nuScenes detection metrics. The primary metric is the nuScenes Detection Score (NDS), which combines mean Average Precision (mAP) with a set of True Positive (TP) metrics that measure localization, size, orientation, velocity, and attribute prediction quality.

---

## nuScenes Detection Score (NDS)

### Definition
NDS is the single aggregate metric used to rank methods on the nuScenes leaderboard:

```
NDS = (1/10) * [5 * mAP + sum(1 - min(1, TP_metric) for each of 5 TP metrics)]
```

Expanded:
```
NDS = (1/10) * [5 * mAP + (1 - min(1, mATE)) + (1 - min(1, mASE))
                        + (1 - min(1, mAOE)) + (1 - min(1, mAVE))
                        + (1 - min(1, mAAE))]
```

### NDS Properties
- **Range:** [0, 1] where 1 is perfect
- **Balanced weighting:** mAP contributes 50% and TP metrics contribute 50% (10% each)
- **Intuition:** A model that detects all objects (high mAP) but localizes them poorly (high ATE) will score lower than a model with moderate detection but precise localization
- **Clamping:** TP error metrics are clamped to [0, 1] to prevent a single catastrophic metric from dominating

### Why NDS Over mAP Alone?
- mAP only measures detection recall at various thresholds but says nothing about prediction quality
- A detection that is 3 meters off from the true position would count as a true positive under the 4.0m threshold, but is useless for autonomous driving
- NDS penalizes poor localization, incorrect size/orientation, and wrong velocity/attribute predictions
- This incentivizes methods that produce high-quality, deployment-ready predictions

---

## Mean Average Precision (mAP)

### Definition
mAP is the mean of per-class Average Precision (AP) across all 10 detection classes:

```
mAP = (1/10) * sum(AP_c for c in classes)
```

### AP Computation

#### Matching Criterion: BEV Center Distance
Unlike 2D object detection (which uses IoU for matching), nuScenes uses Bird's Eye View (BEV) center distance:

- A prediction matches a ground-truth if the Euclidean distance between their centers in the BEV (X-Y) plane is below a threshold
- Height (Z-axis) is NOT considered for matching

#### Distance Thresholds
AP is computed at 4 distance thresholds and averaged:

| Threshold | Value | Target Scenario |
|-----------|-------|-----------------|
| d1 | 0.5 meters | Precise localization (parking, low-speed) |
| d2 | 1.0 meters | Standard urban driving |
| d3 | 2.0 meters | Moderate tolerance |
| d4 | 4.0 meters | Generous threshold (distant objects) |

```
AP_class = (1/4) * [AP@0.5m + AP@1.0m + AP@2.0m + AP@4.0m]
```

#### Precision-Recall Curve
For each class and distance threshold:
1. Rank all predictions by confidence score (descending)
2. For each prediction in order:
   - Find the closest unmatched ground-truth within the distance threshold
   - If found: True Positive (TP)
   - If not found: False Positive (FP)
3. Compute precision and recall at each confidence threshold
4. Compute AP as the area under the precision-recall curve (using 101-point interpolation)

#### Matching Rules
- **One-to-one matching:** Each ground-truth can match at most one prediction (the highest confidence one within threshold)
- **Greedy matching:** Predictions are processed in confidence order; ties are broken by distance
- **No duplicate matching:** Once a GT is matched, it cannot be matched again

### mAP by Class (Typical DETR3D Results)

| Class | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m | Mean AP |
|-------|---------|---------|---------|---------|---------|
| Car | 0.35 | 0.55 | 0.65 | 0.70 | 0.56 |
| Truck | 0.15 | 0.30 | 0.40 | 0.45 | 0.33 |
| Bus | 0.20 | 0.35 | 0.45 | 0.55 | 0.39 |
| Trailer | 0.05 | 0.15 | 0.25 | 0.35 | 0.20 |
| Construction Vehicle | 0.02 | 0.08 | 0.15 | 0.20 | 0.11 |
| Pedestrian | 0.25 | 0.40 | 0.50 | 0.55 | 0.43 |
| Motorcycle | 0.15 | 0.30 | 0.40 | 0.45 | 0.33 |
| Bicycle | 0.10 | 0.20 | 0.30 | 0.35 | 0.24 |
| Barrier | 0.25 | 0.45 | 0.55 | 0.60 | 0.46 |
| Traffic Cone | 0.30 | 0.50 | 0.60 | 0.65 | 0.51 |

---

## True Positive (TP) Metrics

TP metrics are computed only on True Positive detections (correctly matched predictions). They measure the quality of predictions beyond simple detection.

### Average Translation Error (ATE)

```
ATE = (1/|TP|) * sum(||center_pred - center_gt||_2 for tp in TP)
```

- **Unit:** Meters
- **Computation:** Euclidean distance between predicted and ground-truth centers in 2D BEV (X-Y plane)
- **Typical DETR3D value:** 0.64 - 0.72 m
- **Interpretation:** On average, detected objects are localized within ~0.65m of their true BEV position
- **Key insight for DETR3D:** ATE is the weakest TP metric for camera-only methods due to depth estimation difficulty from monocular/multi-view images

### Average Scale Error (ASE)

```
ASE = (1/|TP|) * sum(1 - IOU_3D(pred_box_size, gt_box_size) for tp in TP)
```

- **Unit:** Dimensionless (1 - 3D IoU of axis-aligned boxes at same center)
- **Range:** [0, 1] where 0 is perfect size match
- **Computation:** Compute 3D IoU between predicted and GT boxes after aligning their centers and orientations (isolates size error from translation/rotation error)
- **Typical DETR3D value:** 0.25 - 0.27
- **Interpretation:** Predicted box sizes overlap approximately 73-75% with ground-truth sizes

### Average Orientation Error (AOE)

```
AOE = (1/|TP|) * sum(min_angle_diff(yaw_pred, yaw_gt) for tp in TP)
```

- **Unit:** Radians
- **Computation:** Smallest angle between predicted and ground-truth yaw orientations
- **Range:** [0, pi] (accounts for 180-degree ambiguity for symmetric objects)
- **Symmetry handling:** For barriers and traffic cones, the error is computed modulo pi (180 degrees) since they are roughly symmetric
- **Typical DETR3D value:** 0.37 - 0.40 rad (~21-23 degrees)
- **Interpretation:** Average heading prediction error is about 21-23 degrees

### Average Velocity Error (AVE)

```
AVE = (1/|TP|) * sum(||vel_pred - vel_gt||_2 for tp in TP)
```

- **Unit:** Meters per second (m/s)
- **Computation:** L2 distance between predicted and ground-truth velocity vectors in the BEV plane
- **Typical DETR3D value:** 0.84 - 0.88 m/s
- **Interpretation:** Velocity predictions are off by about 0.85 m/s on average (~3 km/h)
- **Challenge for DETR3D:** Without temporal information (single-frame model), velocity estimation relies entirely on visual cues (motion blur, wheel orientation, relative positioning)
- **Note:** Computed only for classes where velocity is meaningful (vehicles, pedestrians, cyclists); not computed for barriers and traffic cones

### Average Attribute Error (AAE)

```
AAE = (1/|TP|) * sum(1 - correct_attribute(pred, gt) for tp in TP)
```

- **Unit:** Dimensionless (1 - accuracy)
- **Range:** [0, 1] where 0 is perfect attribute prediction
- **Computation:** Fraction of true positive detections where the predicted attribute is incorrect
- **Typical DETR3D value:** 0.13 - 0.20
- **Interpretation:** 80-87% of detected objects have correctly predicted activity state
- **Attributes evaluated:** Vehicle state (moving/stopped/parked), pedestrian state (moving/standing/sitting), cycle state (with_rider/without_rider)

---

## Class-Specific Evaluation

### Per-Class Metric Reporting
All metrics are reported both as class-averages (mAP, mATE, etc.) and per-class:

```
Per-class results:
  car:                  AP=0.56, ATE=0.58, ASE=0.16, AOE=0.12, AVE=0.90, AAE=0.15
  truck:                AP=0.33, ATE=0.72, ASE=0.23, AOE=0.20, AVE=0.80, AAE=0.22
  bus:                  AP=0.39, ATE=0.70, ASE=0.20, AOE=0.08, AVE=1.10, AAE=0.18
  trailer:              AP=0.20, ATE=0.95, ASE=0.24, AOE=0.55, AVE=0.50, AAE=0.10
  construction_vehicle: AP=0.11, ATE=0.88, ASE=0.48, AOE=1.05, AVE=0.12, AAE=0.35
  pedestrian:           AP=0.43, ATE=0.65, ASE=0.28, AOE=0.62, AVE=0.85, AAE=0.20
  motorcycle:           AP=0.33, ATE=0.68, ASE=0.26, AOE=0.55, AVE=1.10, AAE=0.12
  bicycle:              AP=0.24, ATE=0.72, ASE=0.28, AOE=0.70, AVE=0.45, AAE=0.02
  barrier:              AP=0.46, ATE=0.55, ASE=0.30, AOE=0.15, AVE=NaN,  AAE=NaN
  traffic_cone:         AP=0.51, ATE=0.45, ASE=0.34, AOE=NaN,  AVE=NaN,  AAE=NaN
```

### Class-Specific Observations for DETR3D
- **Cars:** Best AP due to abundance in training data; moderate ATE
- **Traffic cones:** High AP despite small size (distinctive appearance and consistent shape)
- **Trailers:** Low AP due to rarity and large size (hard to localize center from partial views)
- **Construction vehicles:** Lowest AP due to extreme rarity and high intra-class variation
- **Pedestrians:** Moderate AP but high AOE (orientation harder to determine for small objects)

---

## Performance Breakdown by Distance

### Distance-Stratified Analysis
While not part of the official nuScenes metrics, analyzing performance by distance reveals important characteristics:

| Distance Range | Approx. mAP | Approx. mATE | Notes |
|---------------|------|------|-------|
| 0 - 10 meters | 0.65 | 0.30 m | High resolution features, multiple camera coverage |
| 10 - 20 meters | 0.55 | 0.50 m | Good performance, primary driving range |
| 20 - 30 meters | 0.40 | 0.70 m | Moderate degradation, fewer pixels per object |
| 30 - 40 meters | 0.25 | 1.00 m | Significant degradation for small objects |
| 40 - 50 meters | 0.15 | 1.30 m | Only large objects (trucks, buses) reliably detected |
| > 50 meters | 0.05 | 1.80 m | Very few detections; mostly large vehicles |

### Distance-Related Challenges for DETR3D
- **Depth ambiguity:** Camera-based depth estimation degrades quadratically with distance
- **Feature resolution:** Objects at 50m occupy very few pixels (~20-50 pixels wide for cars)
- **Multi-camera coverage:** Distant objects appear in fewer cameras (sometimes only one)
- **Reference point accuracy:** Initial 3D reference points must be more precise for distant objects

---

## Evaluation Protocol

### Official Evaluation Code
```python
from nuscenes.eval.detection.evaluate import DetectionEval

eval_config = config_factory('detection_cvpr_2019')
nusc_eval = DetectionEval(
    nusc=nusc,
    config=eval_config,
    result_path='results/detr3d_results.json',
    eval_set='val',
    output_dir='eval_output/',
    verbose=True
)
metrics = nusc_eval.main()
```

### Result File Format
Predictions must be submitted as a JSON file:
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
        "sample_token": "<sample_token>",
        "translation": [x, y, z],
        "size": [w, l, h],
        "rotation": [qw, qx, qy, qz],
        "velocity": [vx, vy],
        "detection_name": "car",
        "detection_score": 0.95,
        "attribute_name": "vehicle.moving"
      }
    ]
  }
}
```

### Evaluation Configuration
- **Max predictions per sample:** 500 (predictions beyond this are discarded)
- **Max distance for evaluation:** Class-specific (50m for most, 60m for trucks/buses)
- **Min detection score:** No minimum threshold (all predictions are evaluated via PR curve)
- **Coordinate frame:** All predictions must be in the global frame

---

## Comparison with Other Methods

### nuScenes Detection Leaderboard (Camera-Only Methods)

| Method | Backbone | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|--------|----------|-----|-----|------|------|------|------|------|
| DETR3D | R101-DCN | 0.479 | 0.412 | 0.641 | 0.255 | 0.394 | 0.845 | 0.133 |
| PETR | R101-DCN | 0.504 | 0.441 | 0.593 | 0.249 | 0.383 | 0.808 | 0.132 |
| BEVDet | R101 | 0.488 | 0.424 | 0.524 | 0.242 | 0.373 | 0.950 | 0.148 |
| BEVFormer | R101-DCN | 0.517 | 0.448 | 0.582 | 0.256 | 0.375 | 0.378 | 0.126 |
| BEVFormer (temporal) | R101-DCN | 0.569 | 0.481 | 0.545 | 0.247 | 0.362 | 0.304 | 0.120 |

### Key Takeaways
- DETR3D established the query-based paradigm; subsequent methods improved upon it
- BEVFormer's temporal modeling dramatically reduces AVE (velocity error)
- DETR3D's mATE is higher than BEV methods (depth estimation is less precise without explicit BEV construction)
- DETR3D remains competitive on mASE and mAAE (size and attribute prediction)

---

## Running Evaluation

### Prerequisites
```bash
pip install nuscenes-devkit
```

### Full Evaluation Pipeline
```bash
# 1. Generate predictions on validation set
python tools/test.py \
    configs/detr3d/detr3d_res101_gridmask.py \
    checkpoints/detr3d_r101_cbgs_24e.pth \
    --eval bbox \
    --out results/detr3d_val_predictions.pkl

# 2. Convert to nuScenes submission format
python tools/convert_results.py \
    --results results/detr3d_val_predictions.pkl \
    --output results/detr3d_results_nusc.json

# 3. Run official evaluation
python tools/nusc_eval.py \
    --result_path results/detr3d_results_nusc.json \
    --eval_set val \
    --dataroot data/nuscenes/ \
    --output_dir eval_results/
```

### Expected Outputs
```
=== nuScenes Detection Evaluation ===
mAP:  0.4120
mATE: 0.6410
mASE: 0.2550
mAOE: 0.3940
mAVE: 0.8450
mAAE: 0.1330
NDS:  0.4790
```

---

## Evaluation Best Practices

### Fair Comparison
- Always report whether CBGS was used during training (significant impact on results)
- Specify exact backbone (ResNet-101 vs. VoVNet-99 vs. ResNet-101-DCN)
- Report whether test-time augmentation (TTA) was applied
- Indicate training epochs (24 vs. 36 vs. longer schedules)
- Note image resolution used during evaluation

### Debugging Poor Results
| Metric | Below Expected | Likely Cause |
|--------|---------------|--------------|
| mAP << 0.35 | Detection failures | Check backbone initialization, verify data loading |
| mATE > 0.80 | Poor localization | Check camera calibration, projection code |
| mASE > 0.35 | Wrong sizes | Check size regression targets, log-scale encoding |
| mAOE > 0.50 | Wrong orientations | Check sin/cos encoding, yaw convention |
| mAVE > 1.20 | Bad velocity | Check velocity annotation loading, coordinate frame |
| mAAE > 0.30 | Wrong attributes | Check attribute label encoding, class-conditional logic |

### Confidence Threshold Selection
- For evaluation: use all predictions (no threshold needed; PR curve handles this)
- For deployment: tune confidence threshold per class on validation set
- Typical operating thresholds: 0.3-0.5 for vehicles, 0.2-0.4 for VRUs, 0.15-0.3 for static objects
