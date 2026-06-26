# BEVFormer: Evaluation Guide

## Understanding 3D Detection Metrics from First Principles

This guide teaches you everything about evaluating 3D object detection models on the nuScenes benchmark. It starts from basic concepts (precision, recall) and builds up to the full nuScenes Detection Score (NDS), explaining what each metric means physically and how to interpret results.

---

## 1. Why Special Metrics for 3D Detection?

### 1.1 Why Not Just Use 2D mAP?

In 2D object detection (ImageNet, COCO), you evaluate using Intersection over Union (IoU) between predicted and ground-truth bounding boxes. If IoU > 0.5, it is a true positive.

This does NOT work for 3D detection because:

1. **Depth errors dominate:** A prediction with perfect 2D overlap but 5m depth error is useless for driving. 2D IoU cannot capture this.

2. **Size matters differently in 3D:** A car predicted at the right position but with wrong height is still dangerous (you might try to drive under it). 3D IoU is complex to compute and does not cleanly separate position vs. size vs. orientation errors.

3. **Velocity and attributes matter:** For driving, knowing an object is MOVING (and how fast) is as important as knowing where it is. 2D metrics cannot capture this.

### 1.2 The nuScenes Philosophy

nuScenes designed their metrics to answer: "How useful is this detection for downstream driving decisions?"

Their insight: separate DETECTION (did you find it?) from QUALITY (how precisely did you characterize it?). This gives:
- **mAP**: Did you find the objects? (detection performance)
- **TP metrics (mATE, mASE, mAOE, mAVE, mAAE)**: For the objects you found, how good are your estimates? (quality of true positives)
- **NDS**: A single number combining both

---

## 2. Precision and Recall (Foundations)

### 2.1 Definitions with a Driving Example

Imagine your model processes one frame and predicts 10 bounding boxes. In reality, there are 8 objects in the scene.

```
Predictions: 10 boxes
Ground truth: 8 objects

Matching results:
  - 6 predictions match a ground truth object  -> True Positives (TP)
  - 4 predictions have no matching GT object   -> False Positives (FP)
  - 2 GT objects have no matching prediction   -> False Negatives (FN)
```

**Precision** = TP / (TP + FP) = 6 / 10 = 0.60
"Of all boxes I predicted, 60% are real objects."
High precision means: when I say there is a car, there probably IS a car.

**Recall** = TP / (TP + FN) = 6 / 8 = 0.75
"Of all real objects, I found 75%."
High recall means: I find most objects, even if I sometimes hallucinate extras.

### 2.2 The Precision-Recall Tradeoff

Every detection model outputs a confidence score (0 to 1) for each prediction. By varying the confidence threshold:

- **High threshold (e.g., 0.9):** Only keep very confident predictions. High precision (few false alarms) but low recall (miss uncertain objects).
- **Low threshold (e.g., 0.1):** Keep everything. High recall (find everything) but low precision (many false alarms).

```
    Precision
    1.0 |*
        | *
        |  *
        |   *
        |    **
        |      ***
        |         ****
        |             *****
    0.0 +--------------------> Recall
        0.0                1.0

    The Precision-Recall (P-R) Curve
    Area under this curve = Average Precision (AP)
```

### 2.3 Why This Matters for Driving

- **False Positive (phantom detection):** Your car brakes for a non-existent pedestrian. Annoying but safe.
- **False Negative (missed detection):** Your car does not see a real pedestrian. Potentially fatal.

For autonomous driving, recall at reasonable precision is critical. You would rather have some false alarms than miss a single pedestrian.

---

## 3. Average Precision (AP) in nuScenes

### 3.1 Distance-Based Matching (Not IoU!)

Unlike COCO/KITTI which use IoU for matching, nuScenes uses **2D center distance** on the ground plane:

```
match_distance = sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)

A prediction matches a GT if: match_distance < threshold
```

Why center distance instead of IoU?
1. Simpler and faster to compute in 3D
2. Cleanly separates detection (finding the object) from quality (size/orientation)
3. IoU in 3D is problematic for thin/elongated objects

### 3.2 Multiple Distance Thresholds

nuScenes evaluates at 4 distance thresholds and averages:

| Threshold | What it measures | Challenge level |
|-----------|------------------|-----------------|
| 0.5 m | Very precise localization | Hard (must be within 0.5m) |
| 1.0 m | Good localization | Moderate |
| 2.0 m | Reasonable localization | Forgiving |
| 4.0 m | Approximate detection | Easy (just find it roughly) |

**AP for one class at one threshold:**

```
1. Sort all predictions by confidence (highest first)
2. For each prediction (in order):
   - If it matches an unmatched GT within threshold: mark as TP
   - If no matching GT: mark as FP
3. Compute precision and recall at each step
4. Compute area under the interpolated P-R curve = AP
```

**AP for one class (averaged over thresholds):**
```
AP_class = mean(AP_0.5m, AP_1.0m, AP_2.0m, AP_4.0m)
```

**mAP (mean over all classes):**
```
mAP = mean(AP_class for each of 10 classes)
```

### 3.3 Worked Example

Imagine class "car" with threshold 2.0m. Your model produces 5 car predictions ranked by confidence:

```
Pred  Confidence  Distance to nearest GT  Match?  Cumulative TP  Precision  Recall
----  ----------  ----------------------  ------  -------------  ---------  ------
 1     0.95       0.8m                    TP      1              1/1=1.00   1/4=0.25
 2     0.90       1.5m                    TP      2              2/2=1.00   2/4=0.50
 3     0.80       3.5m                    FP      2              2/3=0.67   2/4=0.50
 4     0.70       1.2m                    TP      3              3/4=0.75   3/4=0.75
 5     0.60       0.3m                    TP      4              4/5=0.80   4/4=1.00

(Assume 4 GT cars total)

P-R points: (0.25, 1.0), (0.50, 1.0), (0.50, 0.67), (0.75, 0.75), (1.0, 0.80)
AP = area under interpolated P-R curve ~ 0.85
```

---

## 4. nuScenes Detection Score (NDS)

### 4.1 The Formula

```
NDS = (1/10) * [5 * mAP + sum of (1 - min(1, TP_metric)) for 5 TP metrics]

Expanded:
NDS = (1/10) * [5*mAP + (1-min(1,mATE)) + (1-min(1,mASE)) + (1-min(1,mAOE))
                      + (1-min(1,mAVE)) + (1-min(1,mAAE))]
```

### 4.2 Why This Formula?

- **50% weight on mAP:** Finding objects matters
- **50% weight on TP quality:** Characterizing found objects also matters
- **min(1, metric):** Caps the penalty at 1.0 (no bonus for being worse than 1.0 in any metric; also no bonus for being better than 0.0)
- **1 - metric:** Converts "error" (lower is better) to "score" (higher is better)
- **(1/10) normalization:** Keeps NDS in [0, 1] range

### 4.3 What Good NDS Values Look Like

| NDS Range | Interpretation | Examples |
|-----------|---------------|----------|
| 0.0 - 0.3 | Poor | Random or heavily broken models |
| 0.3 - 0.4 | Below average | Early camera-only methods (2020) |
| 0.4 - 0.5 | Decent | Basic monocular methods, FCOS3D |
| 0.5 - 0.6 | Strong | BEVFormer-Base (0.517 val, 0.569 test) |
| 0.6 - 0.7 | Excellent | BEVFormer-Large, best camera-only (2023) |
| 0.7+ | State-of-the-art | LiDAR methods, multi-modal fusion |

---

## 5. True Positive Quality Metrics (Detailed)

For every True Positive detection (a prediction correctly matched to a GT object), nuScenes measures how GOOD the match is across 5 dimensions.

### 5.1 mATE -- Mean Average Translation Error

**What it measures:** How far is your predicted center from the true center?

**Formula:**
```
ATE = sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)    [2D ground-plane distance]
mATE = mean of ATE across all TPs, across all classes
```

**Physical intuition:** "Your detected car's center is on average 0.67m away from where it actually is."

**What is good vs bad:**
| mATE Value | Interpretation | Impact on Driving |
|------------|---------------|-------------------|
| < 0.3 m | Excellent | Sub-lane-width accuracy |
| 0.3 - 0.5 m | Very good | Within a parking space width |
| 0.5 - 0.8 m | Good | BEVFormer territory (0.673) |
| 0.8 - 1.2 m | Moderate | Might confuse adjacent lanes |
| > 1.5 m | Poor | Object might be in wrong lane |

**Which classes are hardest?**
- trailer (mATE ~1.04m): Very long, center is far from visible part
- construction_vehicle (mATE ~1.06m): Rare, variable shape
- car (mATE ~0.46m): Common, regular shape, easiest

**What causes high mATE:**
- Low BEV resolution (each cell = 0.512m, so inherent quantization)
- Depth errors (camera-based methods struggle at distance)
- Few pixels on distant objects

### 5.2 mASE -- Mean Average Scale Error

**What it measures:** How wrong is your predicted box size, ignoring position and orientation?

**Formula:**
```
ASE = 1 - IoU_3D(pred_box_aligned, gt_box_aligned)

Where "aligned" means: move both boxes to origin, align orientations,
then compute 3D IoU (only size differs).

mASE = mean of ASE across all TPs
```

**Physical intuition:** "Your predicted box overlaps with the true box by 73% when position and orientation are removed. The 27% error is from size mismatch."

**What is good vs bad:**
| mASE Value | Interpretation |
|------------|---------------|
| < 0.20 | Excellent size estimation |
| 0.20 - 0.30 | Good (BEVFormer: 0.274) |
| 0.30 - 0.40 | Moderate |
| > 0.40 | Poor -- significant size errors |

**Which classes are hardest?**
- construction_vehicle (mASE ~0.48): Hugely variable size (small bobcat vs giant crane)
- traffic_cone (mASE ~0.34): Very small, hard to estimate precisely
- car (mASE ~0.15): Very consistent size across instances

### 5.3 mAOE -- Mean Average Orientation Error

**What it measures:** How wrong is your predicted heading angle?

**Formula:**
```
AOE = smallest angle between predicted and GT yaw
    = min(|yaw_pred - yaw_gt|, 2*pi - |yaw_pred - yaw_gt|)

For rotationally symmetric objects (barrier, traffic_cone):
    AOE is computed modulo pi (180-degree ambiguity allowed)

mAOE = mean of AOE across all TPs
```

**Physical intuition:** "Your predicted car heading is off by about 21 degrees on average" (0.372 rad).

**What is good vs bad:**
| mAOE Value | Degrees | Interpretation |
|------------|---------|---------------|
| < 0.1 rad | < 6 deg | Excellent |
| 0.1 - 0.3 | 6-17 deg | Very good |
| 0.3 - 0.5 | 17-29 deg | Good (BEVFormer: 0.372 = ~21 deg) |
| 0.5 - 1.0 | 29-57 deg | Moderate -- might confuse direction |
| > 1.0 rad | > 57 deg | Poor -- essentially wrong direction |

**Which classes are hardest?**
- construction_vehicle (mAOE ~1.13 rad): Irregular shape, unclear "front"
- bicycle (mAOE ~0.66 rad): Thin, often only a few pixels
- car (mAOE ~0.08 rad): Clear front/back distinction

### 5.4 mAVE -- Mean Average Velocity Error

**What it measures:** How wrong is your predicted velocity?

**Formula:**
```
AVE = sqrt((vx_pred - vx_gt)^2 + (vy_pred - vy_gt)^2)   [L2 norm of velocity difference]
mAVE = mean of AVE across all TPs

Note: Only computed for MOVING object classes (vehicles, pedestrians, cyclists).
      Static classes (barrier, traffic_cone) are excluded.
```

**Physical intuition:** "Your predicted velocity is off by 0.39 m/s on average" (about 1.4 km/h).

**What is good vs bad:**
| mAVE Value | km/h equivalent | Interpretation |
|------------|----------------|---------------|
| < 0.3 m/s | < 1.1 km/h | Excellent |
| 0.3 - 0.5 | 1.1 - 1.8 km/h | Good (BEVFormer: 0.394) |
| 0.5 - 1.0 | 1.8 - 3.6 km/h | Moderate |
| > 1.0 | > 3.6 km/h | Poor -- cannot trust velocity |
| > 2.0 | > 7.2 km/h | Very poor -- basically no velocity info |

**Why velocity matters for driving:**
- Prediction of future positions depends on velocity
- Time-to-collision calculations need velocity
- Decision to yield vs. proceed depends on approaching speed

**Key insight:** Without temporal fusion, mAVE is typically >0.8 m/s. BEVFormer's temporal self-attention reduces this to 0.394 -- demonstrating that temporal fusion is essential for velocity estimation.

### 5.5 mAAE -- Mean Average Attribute Error

**What it measures:** Did you correctly predict the object's attribute (behavioral state)?

**Attributes in nuScenes:**
- Vehicles: {moving, stopped, parked}
- Pedestrians: {moving, standing, sitting_lying_down}
- Cyclists: {with_rider, without_rider}

**Formula:**
```
AAE = 1 - accuracy(predicted_attribute, gt_attribute)
mAAE = mean fraction of incorrect attribute predictions across TPs
```

**Physical intuition:** "You confused a parked car with a stopped car (or vice versa) about 20% of the time."

**Why it matters:** A parked car will never move, so you can pass close. A stopped car might suddenly start moving, requiring more caution.

---

## 6. How to Interpret Results

### 6.1 Common Patterns and What They Mean

**Pattern: High mAP but high mAVE (>0.8)**
- Diagnosis: Model has no temporal fusion
- Fix: Enable temporal self-attention, verify can_bus data is loaded

**Pattern: Low mAP but low TP errors**
- Diagnosis: Model detects few objects but those it detects are precise
- This suggests: confidence threshold too high, or model under-trained
- Fix: Lower score threshold, train more epochs

**Pattern: High mATE (>1.0m)**
- Diagnosis: Poor localization, likely depth-related
- Causes: Wrong calibration matrices, low BEV resolution, poor backbone
- Fix: Verify calibration, increase BEV resolution, use better pretrained backbone

**Pattern: Good cars/pedestrians but terrible construction_vehicle/trailer**
- Diagnosis: Class imbalance (rare classes under-represented)
- Fix: Enable CBGS (class-balanced group sampling), increase training epochs

### 6.2 Reading Per-Class Results

```
Example output:
         Class    AP    ATE    ASE    AOE    AVE    AAE
           car 0.594  0.462  0.154  0.081  0.359  0.177
         truck 0.388  0.692  0.207  0.096  0.348  0.198
           bus 0.445  0.723  0.197  0.051  0.846  0.245
       trailer 0.205  1.040  0.243  0.557  0.232  0.092
construction_v 0.091  1.058  0.481  1.125  0.121  0.362
    pedestrian 0.449  0.704  0.295  0.592  0.432  0.216
    motorcycle 0.393  0.616  0.261  0.449  0.601  0.003
       bicycle 0.331  0.607  0.270  0.661  0.255  0.009
       barrier 0.534  0.557  0.284  0.136  nan    nan
  traffic_cone 0.532  0.404  0.342  nan    nan    nan
```

**How to read this:**
- `car` AP 0.594: Best detected class (common, large, regular shape)
- `construction_v` AP 0.091: Worst class (rare, variable appearance)
- `bus` AVE 0.846: High velocity error for buses (they accelerate/decelerate differently)
- `barrier`/`traffic_cone` AVE=nan: Static objects, velocity not evaluated
- `traffic_cone` AOE=nan: Symmetric object, orientation not evaluated

### 6.3 Fair Comparison Checklist

When comparing BEVFormer to other papers, verify:

- [ ] Same dataset split (v1.0-trainval, standard train/val division)
- [ ] Same input resolution (900x1600 unless noted)
- [ ] Same backbone pretraining (FCOS3D pretrained vs ImageNet-only: ~3 NDS difference!)
- [ ] Same number of training epochs (24)
- [ ] Same data augmentation (grid mask, photometric distortion)
- [ ] Test-time augmentation (TTA) noted if used (adds ~1-2 NDS)
- [ ] Single model vs ensemble clearly stated

**Common sources of unfair comparison:**
- Paper A uses V2-99 backbone (stronger) vs Paper B uses ResNet-50 (weaker)
- Paper A uses 48 epochs vs Paper B uses 24 epochs
- Paper A uses TTA (flip) vs Paper B without TTA

---

## 7. Per-Class Analysis

### 7.1 Why Some Classes Are Much Harder

| Class | # Annotations (train) | Avg Size (m) | Avg Distance | AP | Primary Challenge |
|-------|----------------------|--------------|--------------|-----|-------------------|
| car | ~340,000 | 4.6 x 1.9 x 1.7 | 25m | 0.594 | None (easiest) |
| pedestrian | ~160,000 | 0.7 x 0.7 x 1.8 | 20m | 0.449 | Small, variable pose |
| barrier | ~120,000 | 0.5 x 2.5 x 1.0 | 15m | 0.534 | Static, regular |
| traffic_cone | ~70,000 | 0.4 x 0.4 x 1.0 | 12m | 0.532 | Very small but distinctive color |
| truck | ~65,000 | 6.9 x 2.5 x 2.8 | 30m | 0.388 | Large, sometimes far |
| bus | ~12,000 | 11.1 x 2.9 x 3.5 | 35m | 0.445 | Rare, very large |
| trailer | ~20,000 | 12.3 x 2.9 x 3.9 | 40m | 0.205 | Very large, often occluded |
| motorcycle | ~15,000 | 2.1 x 0.8 x 1.5 | 22m | 0.393 | Small, fast-moving |
| bicycle | ~10,000 | 1.7 x 0.6 x 1.3 | 18m | 0.331 | Very small, rare |
| construction_v | ~7,000 | 6.4 x 2.8 x 3.1 | 30m | 0.091 | Extremely rare, variable shape |

### 7.2 The "Long Tail" Problem

Construction vehicles have 48x fewer training examples than cars. Even with class-balanced sampling, the network cannot learn all the visual variations of cranes, excavators, dump trucks, and cement mixers from just 7,000 examples.

### 7.3 Distance Distribution Effects

Most annotations are within 30m. Beyond 40m, objects occupy very few pixels:
```
At 20m: a car is ~200 pixels wide in the image
At 40m: a car is ~100 pixels wide
At 60m: a car is ~65 pixels wide
At 80m: a car is ~50 pixels wide  (hard to even identify class)
```

---

## 8. Temporal Consistency Evaluation

### 8.1 Why Single-Frame Metrics Are Not Enough

Imagine a model that detects a parked car in frame 1, misses it in frame 2, detects it again in frame 3. Per-frame mAP looks fine (2/3 recall), but the driving experience is terrible -- the car "flickers" in and out of existence, confusing the planner.

### 8.2 Tracking-Based Metrics

| Metric | Full Name | BEVFormer-Base | What It Measures |
|--------|-----------|----------------|------------------|
| AMOTA | Avg Multi-Object Tracking Accuracy | 0.412 | Overall tracking quality |
| AMOTP | Avg Multi-Object Tracking Precision | 1.132m | Localization of tracked objects |
| ID Switches | Identity switches | 842 | How often tracked ID changes |
| Fragmentation | Track fragmentation | 1,247 | How often tracks break/resume |

### 8.3 Temporal Consistency Improvements

| Metric | Single Frame | With Temporal | Improvement |
|--------|-------------|---------------|-------------|
| False Positives / frame | 4.2 | 3.1 | -26% fewer phantom detections |
| False Negatives / frame | 5.8 | 4.9 | -16% fewer missed objects |
| Box Jitter (ATE std) | 0.31m | 0.19m | -39% more stable positions |
| Velocity Error | 0.842 m/s | 0.394 m/s | -53% better velocity |
| Heading Jitter | 0.12 rad | 0.08 rad | -33% more stable orientation |

Temporal fusion makes detections significantly more stable across frames, which is crucial for downstream tracking and planning.

---

## 9. Running Evaluation

### 9.1 Standard Evaluation Command

```bash
# Multi-GPU evaluation (faster)
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    8 \
    --eval bbox

# Single-GPU evaluation
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --eval bbox \
    --gpu-ids 0
```

### 9.2 Understanding the Output

The evaluation produces a table like:

```
---------- nuScenes Detection Evaluation ----------
mAP: 0.4163
mATE: 0.6728
mASE: 0.2731
mAOE: 0.3718
mAVE: 0.3944
mAAE: 0.1981
NDS: 0.5170
```

**Interpreting these numbers:**
- mAP 0.416: Detecting about 42% of objects (averaged across distances/classes)
- mATE 0.673: Average position error is 67cm
- NDS 0.517: Overall score in [0,1] -- solid camera-only performance

### 9.3 Generating Submission Files

```bash
# Generate JSON for leaderboard submission (nuScenes test set)
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --format-only \
    --eval-options jsonfile_prefix=results/bevformer_submit

# Output: results/bevformer_submit/results_nusc.json
# Submit to: https://eval.ai/web/challenges/challenge-page/356/
```

### 9.4 Visualization for Debugging

```bash
# Visualize predictions overlaid on camera images and BEV
python tools/visualize.py \
    projects/configs/bevformer/bevformer_base.py \
    work_dirs/bevformer_base/epoch_24.pth \
    --show-dir visualizations/ \
    --show-bev \
    --show-cameras \
    --score-thr 0.3
```

---

## 10. Debugging Poor Results

### 10.1 Systematic Diagnosis

```
Is the problem DETECTION (low mAP) or QUALITY (high TP errors)?

Low mAP (< 0.35):
  |-- Check recall: are objects being missed?
  |     |-- Yes: backbone features may be poor (check pretrained weights)
  |     |-- Yes at distance: BEV resolution too low
  |     +-- Yes for rare classes: enable CBGS
  |-- Check precision: too many false positives?
  |     |-- Yes: model overconfident (lower LR, more training)
  |     +-- Yes in specific regions: check calibration

High mATE (> 0.8):
  |-- Check calibration matrices (are they correct for your data?)
  |-- Check BEV resolution (100x100 has 1m quantization error)
  |-- Check backbone (stronger backbone = better features = better depth)
  +-- Distance analysis: if only far objects, this is expected behavior

High mAVE (> 0.6):
  |-- Is temporal fusion enabled? (check video_test_mode=True)
  |-- Is can_bus data loaded? (check data pipeline)
  |-- Is ego-motion computation correct? (check coordinate transforms)
  +-- Is queue_length >= 2? (need at least 1 previous frame)

High mAOE (> 0.5):
  |-- Objects with ambiguous front/back (common for trucks from behind)
  |-- Check sin/cos regression -- is loss converging?
  +-- May need more training epochs for orientation to converge
```

### 10.2 Quick Sanity Checks

```python
# 1. Verify predictions are in correct range
# cx, cy should be in [-51.2, 51.2], NOT normalized [0, 1]
# velocities should be in m/s (typical: -20 to 20), NOT km/h

# 2. Check score distribution
# If all scores > 0.9: model is overconfident (reduce confidence)
# If all scores < 0.1: model is underconfident (might need training)

# 3. Verify temporal pairs
# Print prev/next tokens -- if all "None", temporal is broken
```

---

## 11. Benchmark Comparison

### 11.1 Camera-Only Methods (nuScenes Test Set, as of 2023)

| Method | Year | Backbone | NDS | mAP | mAVE | Key Innovation |
|--------|------|----------|-----|-----|------|----------------|
| FCOS3D | 2021 | R101 | 0.428 | 0.358 | 1.434 | Per-image monocular |
| DETR3D | 2022 | V2-99 | 0.479 | 0.412 | 0.845 | 3D reference points |
| PETR | 2022 | V2-99 | 0.504 | 0.441 | 0.808 | 3D position embedding |
| BEVDet4D | 2022 | Swin-B | 0.515 | 0.421 | 0.556 | Depth + temporal |
| **BEVFormer** | **2022** | **R101-DCN** | **0.569** | **0.481** | **0.378** | **Attn BEV + temporal** |
| **BEVFormer** | **2022** | **V2-99** | **0.592** | **0.517** | **0.322** | **Larger backbone** |
| StreamPETR | 2023 | V2-99 | 0.592 | 0.504 | 0.283 | PETR + streaming temporal |
| SOLOFusion | 2023 | R101-DCN | 0.582 | 0.483 | 0.246 | Long-term temporal |

### 11.2 Historical Context

BEVFormer was published in 2022 and was the top camera-only method at that time. By 2023, several methods (StreamPETR, SOLOFusion, Far3D) matched or slightly exceeded it, often by building on BEVFormer's insights (temporal fusion, BEV representation). BEVFormer remains historically significant as the method that proved attention-based BEV construction works.

### 11.3 Gap to LiDAR

| Method | Sensor | NDS | Gap |
|--------|--------|-----|-----|
| BEVFormer-Large | Camera | 0.592 | baseline |
| CenterPoint | LiDAR | 0.673 | +8.1 |
| TransFusion-L | LiDAR | 0.702 | +11.0 |
| BEVFusion | Camera+LiDAR | 0.714 | +12.2 |

The camera-to-LiDAR gap is primarily in: (a) long-range detection (>50m), (b) precise localization (mATE), and (c) adverse weather robustness.

---

## 12. Reproducing Published Results

### 12.1 Expected Results with Tolerance

| Metric | Expected | Acceptable Range | Outside Range? |
|--------|----------|-----------------|----------------|
| NDS | 0.517 | 0.512 - 0.522 | Check pretrained weights |
| mAP | 0.416 | 0.411 - 0.421 | Check training schedule |
| mATE | 0.673 | 0.660 - 0.690 | Check BEV resolution |
| mAVE | 0.394 | 0.380 - 0.410 | Check temporal fusion |

### 12.2 Common Reproduction Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong backbone pretrain | NDS ~0.48 instead of ~0.52 | Use FCOS3D pretrained weights |
| Missing can_bus data | mAVE ~0.8 (poor velocity) | Download and extract can_bus |
| queue_length=1 | No temporal benefit | Set queue_length=4 |
| Wrong mmcv version | CUDA kernel errors | Use mmcv-full==1.7.1 exactly |
| Random seed variation | +/- 0.5 NDS | Run 3 seeds, report mean |

---

## 13. Evaluation Configuration Reference

### 13.1 Class Evaluation Ranges

```python
# Maximum range for evaluating each class
class_range = {
    'car': 50,              # meters
    'truck': 50,
    'bus': 50,
    'trailer': 50,
    'construction_vehicle': 50,
    'pedestrian': 40,       # shorter range (smaller objects)
    'motorcycle': 40,
    'bicycle': 40,
    'barrier': 30,          # static, typically close
    'traffic_cone': 30,
}
```

### 13.2 Post-Processing for Evaluation

```python
test_cfg = dict(
    pts=dict(
        score_threshold=0.0,    # Keep all for eval (AP computation needs full ranking)
        max_per_sample=300,     # Max predictions per frame
        nms_type=None,          # DETR-style: no NMS needed
    )
)
```

Note: For evaluation, keep score_threshold=0.0 so the full precision-recall curve can be computed. For deployment, use score_threshold=0.3 or higher to reduce false positives.
