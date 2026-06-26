# RangeNet++: Evaluation Guide

## Overview

Evaluation of RangeNet++ follows the SemanticKITTI benchmark protocol, using mean Intersection over Union (mIoU) as the primary metric. This guide covers evaluation metrics, per-class results, inference speed benchmarks, and the effect of KNN post-processing.

---

## Evaluation Metrics

### Mean Intersection over Union (mIoU)

The primary metric is mIoU computed over 19 semantic classes (excluding unlabeled):

```
IoU_c = TP_c / (TP_c + FP_c + FN_c)

mIoU = (1/19) * sum(IoU_c for c in 1..19)
```

Where:
- `TP_c`: True positives for class c (correctly predicted as c)
- `FP_c`: False positives for class c (incorrectly predicted as c)
- `FN_c`: False negatives for class c (should be c but predicted as something else)

### Per-Class IoU

Individual class IoU reveals performance on specific object categories:

```python
def compute_iou(pred, target, num_classes=20):
    ious = []
    for c in range(1, num_classes):  # Skip class 0 (unlabeled)
        pred_c = (pred == c)
        target_c = (target == c)
        
        tp = (pred_c & target_c).sum()
        fp = (pred_c & ~target_c).sum()
        fn = (~pred_c & target_c).sum()
        
        if tp + fp + fn == 0:
            ious.append(float('nan'))  # Class not present
        else:
            ious.append(tp / (tp + fp + fn))
    
    return np.nanmean(ious), ious
```

### Additional Metrics

| Metric | Description |
|--------|-------------|
| Overall Accuracy (OA) | Percentage of correctly classified points |
| Frequency-Weighted IoU (fwIoU) | IoU weighted by class frequency |
| Per-class Precision | TP / (TP + FP) per class |
| Per-class Recall | TP / (TP + FN) per class |

---

## Results on SemanticKITTI Test Set

### Overall Performance

| Model | mIoU (%) | Inference (ms) | FPS |
|-------|----------|----------------|-----|
| RangeNet21 | 47.4 | 14 | 71 |
| RangeNet53 | 49.9 | 20 | 50 |
| RangeNet53++ (KNN) | 52.2 | 25 | 40 |

### Per-Class IoU Results (RangeNet53++)

| Class | IoU (%) | Category |
|-------|---------|----------|
| car | 91.4 | Vehicle |
| bicycle | 26.2 | Vehicle |
| motorcycle | 25.7 | Vehicle |
| truck | 34.2 | Vehicle |
| other-vehicle | 20.0 | Vehicle |
| person | 45.6 | Human |
| bicyclist | 33.6 | Human |
| motorcyclist | 4.6 | Human |
| road | 91.8 | Ground |
| parking | 64.8 | Ground |
| sidewalk | 75.0 | Ground |
| other-ground | 27.8 | Ground |
| building | 87.4 | Structure |
| fence | 58.6 | Structure |
| vegetation | 80.5 | Nature |
| trunk | 55.1 | Nature |
| terrain | 64.6 | Nature |
| pole | 47.9 | Object |
| traffic-sign | 55.9 | Object |

### Performance by Category

| Category | Avg IoU (%) | Classes |
|----------|-------------|---------|
| Ground surfaces | 64.9 | road, parking, sidewalk, other-ground |
| Vehicles | 39.5 | car, bicycle, motorcycle, truck, other-vehicle |
| Structures | 73.0 | building, fence |
| Nature | 66.7 | vegetation, trunk, terrain |
| Humans | 27.9 | person, bicyclist, motorcyclist |
| Small objects | 51.9 | pole, traffic-sign |

---

## Analysis of Results

### Strong Performance (IoU > 70%)

- **Road (91.8%):** Large, flat surface with distinctive range/height signature. Consistent appearance.
- **Car (91.4%):** Most common object, abundant training samples, distinctive shape.
- **Building (87.4%):** Large vertical surfaces, consistent height profiles.
- **Vegetation (80.5%):** Distinctive "fuzzy" return pattern from leaves/branches.
- **Sidewalk (75.0%):** Adjacent to road, elevation difference provides clear boundary.

### Moderate Performance (IoU 40-70%)

- **Parking (64.8%):** Similar to road but with different context (adjacent to buildings).
- **Terrain (64.6%):** Distinguishable from road by lack of smoothness.
- **Fence (58.6%):** Partially transparent structures; mixed returns.
- **Traffic-sign (55.9%):** Small but reflective; good intensity contrast.
- **Trunk (55.1%):** Thin vertical structures, few points but distinctive.

### Weak Performance (IoU < 40%)

- **Motorcyclist (4.6%):** Extremely rare class, very few training samples.
- **Other-vehicle (20.0%):** Heterogeneous class (buses, trains, construction vehicles).
- **Motorcycle (25.7%):** Rare, small, easily confused with bicycle.
- **Bicycle (26.2%):** Very few points at distance, thin frame structure.
- **Other-ground (27.8%):** Ambiguous class definition, confused with road/parking.
- **Bicyclist (33.6%):** Rare, combined person+bicycle hard to distinguish.
- **Truck (34.2%):** Less common, variable sizes/shapes.

### Key Observations

1. **Class frequency correlates with performance:** Common classes (road, car, building) perform best.
2. **Size matters:** Large objects (buildings, road) easier than small objects (poles, signs).
3. **Range image limitations:** Thin objects (bicycles, poles) lose detail in projection.
4. **Category confusion:** Similar-shaped classes (motorcycle vs. bicycle, road vs. parking) show high confusion.

---

## Inference Speed Benchmarks

### Hardware Configurations

| GPU | CNN (ms) | KNN (ms) | Total (ms) | FPS |
|-----|----------|----------|-----------|-----|
| NVIDIA GTX 1080 Ti | 22 | 5 | 27 | 37 |
| NVIDIA RTX 2080 Ti | 16 | 4 | 20 | 50 |
| NVIDIA V100 | 14 | 3 | 17 | 59 |
| NVIDIA RTX 3090 | 11 | 3 | 14 | 71 |

### Latency Breakdown

| Stage | Time (ms) | Percentage |
|-------|-----------|------------|
| Point cloud loading | 1-2 | 5% |
| Spherical projection | 1-2 | 5% |
| CNN forward pass | 15-20 | 65% |
| Softmax + argmax | 0.5 | 2% |
| KNN post-processing | 3-5 | 18% |
| Label back-projection | 0.5-1 | 5% |
| **Total** | **~22-30** | 100% |

### Resolution vs. Speed

| Resolution | CNN (ms) | mIoU (%) | Points/sec |
|-----------|----------|----------|------------|
| 64 x 512 | 8 | 46.1 | ~15M |
| 64 x 1024 | 12 | 50.3 | ~13M |
| 64 x 2048 | 20 | 52.2 | ~6.5M |

### Batch Processing

| Batch Size | Throughput (scans/sec) | Latency (ms/scan) |
|------------|----------------------|-------------------|
| 1 | 40-50 | 20-25 |
| 2 | 55-65 | 31-36 |
| 4 | 70-80 | 50-57 |
| 8 | 85-95 | 84-94 |

---

## Effect of KNN Post-Processing

### Ablation Study

| Configuration | mIoU (%) | Delta |
|--------------|----------|-------|
| No KNN (direct projection) | 49.9 | baseline |
| KNN K=1 | 50.8 | +0.9 |
| KNN K=3 | 51.7 | +1.8 |
| KNN K=5 | 52.2 | +2.3 |
| KNN K=7 | 51.9 | +2.0 |
| KNN K=9 | 51.5 | +1.6 |
| KNN K=11 | 51.3 | +1.4 |

### Per-Class Impact of KNN (K=5)

| Class | Without KNN | With KNN | Improvement |
|-------|-------------|----------|-------------|
| car | 89.8 | 91.4 | +1.6 |
| bicycle | 22.1 | 26.2 | +4.1 |
| motorcycle | 22.5 | 25.7 | +3.2 |
| person | 41.2 | 45.6 | +4.4 |
| pole | 42.3 | 47.9 | +5.6 |
| traffic-sign | 50.4 | 55.9 | +5.5 |
| trunk | 49.8 | 55.1 | +5.3 |
| building | 86.2 | 87.4 | +1.2 |
| road | 91.1 | 91.8 | +0.7 |
| vegetation | 79.0 | 80.5 | +1.5 |

### Analysis

- **Largest improvement:** Thin/small objects (poles +5.6, traffic-signs +5.5, trunks +5.3) benefit most from 3D spatial smoothing.
- **Moderate improvement:** Objects with complex boundaries (persons +4.4, bicycles +4.1).
- **Smallest improvement:** Large uniform surfaces (road +0.7, building +1.2) already have clean predictions.
- **Optimal K:** K=5 provides the best balance; larger K causes oversmoothing, especially for thin objects.

---

## Evaluation Procedure

### Step 1: Prepare Predictions

```python
# Run inference on test sequences
model.eval()
with torch.no_grad():
    for scan_path in test_scans:
        # Load and project
        points = load_point_cloud(scan_path)
        range_img, proj_idx = spherical_projection(points)
        
        # CNN prediction
        logits = model(range_img.unsqueeze(0).cuda())
        pred_range = logits.argmax(dim=1).squeeze()
        
        # KNN post-processing
        pred_3d = knn_post_process(points, pred_range, proj_idx, k=5)
        
        # Save predictions
        save_predictions(pred_3d, output_path)
```

### Step 2: Format for Submission

Predictions must match the label format:
```
predictions/
  sequences/
    11/
      predictions/
        000000.label    # uint32, only lower 16 bits used
        000001.label
        ...
    12/
    ...
    21/
```

### Step 3: Submit to Benchmark

Upload predictions to the SemanticKITTI benchmark server:
- Website: http://www.semantic-kitti.org/tasks.html#semseg
- Format: ZIP file containing the predictions directory structure
- Evaluation: Server computes mIoU over all test sequences

### Local Validation

For local development, evaluate on sequence 08:

```python
# Evaluate on validation set
from utils.evaluation import compute_miou

all_preds = []
all_labels = []

for scan_path, label_path in val_dataset:
    pred = inference(model, scan_path)
    label = load_labels(label_path)
    all_preds.append(pred)
    all_labels.append(label)

miou, per_class_iou = compute_miou(all_preds, all_labels, num_classes=20)
print(f"Validation mIoU: {miou:.1f}%")
for cls_id, cls_iou in enumerate(per_class_iou):
    print(f"  Class {cls_id}: {cls_iou:.1f}%")
```

---

## Comparison with Other Methods

### SemanticKITTI Benchmark (Test Set)

| Method | Type | mIoU (%) | FPS |
|--------|------|----------|-----|
| PointNet | Point | 14.6 | ~2 |
| PointNet++ | Point | 20.1 | ~1 |
| TangentConv | Point | 35.9 | ~1 |
| SqueezeSeg | Range | 29.5 | 83 |
| SqueezeSegV2 | Range | 39.7 | 67 |
| RangeNet53++ | Range | 52.2 | 40 |
| SalsaNext | Range | 59.5 | 24 |
| MinkowskiNet | Voxel | 63.1 | ~5 |
| Cylinder3D | Voxel | 67.8 | ~4 |
| SPVCNN | Voxel+Point | 63.8 | ~7 |

### Key Takeaways

1. **RangeNet++ offers the best speed-accuracy tradeoff** among methods available at publication time (2019).
2. **Range-image methods dominate in speed** but trail voxel-based methods in accuracy.
3. **KNN post-processing closes the gap** between naive range projection and 3D-native methods.
4. **Subsequent range-image methods** (SalsaNext, FIDNet, CENet) have further improved on RangeNet++'s foundation.

---

## Reproducing Results

### Validation Set (Sequence 08) Expected Results

When properly trained, expect these approximate results on the validation set:

| Metric | Expected Value |
|--------|---------------|
| mIoU (with KNN) | 52-54% |
| mIoU (without KNN) | 49-51% |
| Overall Accuracy | 89-91% |
| Road IoU | 90-93% |
| Car IoU | 90-93% |
| Person IoU | 40-50% |

### Common Issues Affecting Results

| Issue | Effect on mIoU | Solution |
|-------|---------------|----------|
| Wrong label mapping | -10-20% | Use official SemanticKITTI mapping |
| No class weighting | -3-5% | Apply inverse frequency weights |
| No KNN | -2-3% | Enable KNN post-processing |
| Wrong FOV parameters | -5-10% | Use exact sensor specs |
| No augmentation | -2-3% | Enable rotation + flip + dropout |
| Insufficient epochs | -2-5% | Train for 150+ epochs |
