# Cylinder3D: Evaluation Guide

## Overview

This guide covers the evaluation protocols, metrics, submission formats, and expected results for Cylinder3D on the SemanticKITTI and nuScenes-lidarseg benchmarks.

---

## Mean Intersection-over-Union (mIoU)

### Definition

The primary metric for LiDAR semantic segmentation is **mean Intersection-over-Union (mIoU)**:

```
IoU_c = TP_c / (TP_c + FP_c + FN_c)

where for class c:
  TP_c = True Positives  (correctly predicted as class c)
  FP_c = False Positives (incorrectly predicted as class c)
  FN_c = False Negatives (ground truth is class c, predicted differently)

mIoU = (1/C) * sum(IoU_c for c in evaluated_classes)

where C = number of evaluated classes (19 for SemanticKITTI, 16 for nuScenes)
```

### Properties

- **Range:** [0, 1] or equivalently [0%, 100%]
- **Per-class averaging:** Each class contributes equally regardless of frequency
- **Ignore class:** Unlabeled/noise points are excluded from both predictions and evaluation
- **Symmetric:** Penalizes both over-prediction (FP) and under-prediction (FN)

### Implementation

```python
import numpy as np

def compute_miou(predictions, labels, num_classes, ignore_label=0):
    """
    Compute per-class IoU and mIoU.
    
    Args:
        predictions: np.array of shape (N,), predicted class IDs
        labels: np.array of shape (N,), ground truth class IDs
        num_classes: int, total number of classes including ignore
        ignore_label: int, class ID to ignore in evaluation
    
    Returns:
        miou: float, mean IoU across valid classes
        per_class_iou: dict, IoU for each class
    """
    # Build confusion matrix
    valid_mask = labels != ignore_label
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(predictions, labels):
        confusion[gt, pred] += 1
    
    # Compute IoU per class
    per_class_iou = {}
    valid_classes = []
    
    for c in range(num_classes):
        if c == ignore_label:
            continue
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        
        if tp + fp + fn == 0:
            continue  # class not present in evaluation
        
        iou = tp / (tp + fp + fn)
        per_class_iou[c] = iou
        valid_classes.append(iou)
    
    miou = np.mean(valid_classes) if valid_classes else 0.0
    return miou, per_class_iou


def compute_miou_fast(predictions, labels, num_classes, ignore_label=0):
    """Vectorized version using np.bincount for efficiency."""
    valid = labels != ignore_label
    pred_valid = predictions[valid]
    label_valid = labels[valid]
    
    # Flatten to 1D index for confusion matrix
    conf_idx = label_valid * num_classes + pred_valid
    confusion = np.bincount(conf_idx, minlength=num_classes**2)
    confusion = confusion.reshape(num_classes, num_classes)
    
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    
    # Avoid division by zero
    denom = tp + fp + fn
    valid_classes = (denom > 0) & (np.arange(num_classes) != ignore_label)
    
    iou = np.zeros(num_classes)
    iou[valid_classes] = tp[valid_classes] / denom[valid_classes]
    
    miou = iou[valid_classes].mean()
    return miou, iou
```

---

## Per-Class IoU Evaluation

### SemanticKITTI Expected Results (Cylinder3D)

| Class ID | Class Name | IoU (%) | Difficulty |
|----------|-----------|---------|------------|
| 1 | car | 97.1 | Easy |
| 2 | bicycle | 67.6 | Hard |
| 3 | motorcycle | 64.0 | Hard |
| 4 | truck | 69.4 | Medium |
| 5 | other-vehicle | 48.2 | Hard |
| 6 | person | 69.4 | Medium |
| 7 | bicyclist | 75.5 | Medium |
| 8 | motorcyclist | 42.5 | Very Hard |
| 9 | road | 91.4 | Easy |
| 10 | parking | 64.5 | Medium |
| 11 | sidewalk | 75.5 | Medium |
| 12 | other-ground | 22.3 | Very Hard |
| 13 | building | 90.2 | Easy |
| 14 | fence | 60.5 | Medium |
| 15 | vegetation | 84.3 | Easy |
| 16 | trunk | 69.1 | Medium |
| 17 | terrain | 72.3 | Medium |
| 18 | pole | 62.8 | Medium |
| 19 | traffic-sign | 52.5 | Hard |
| — | **mIoU** | **68.9** | — |

### nuScenes-lidarseg Expected Results (Cylinder3D)

| Class ID | Class Name | IoU (%) |
|----------|-----------|---------|
| 1 | barrier | 76.4 |
| 2 | bicycle | 30.6 |
| 3 | bus | 89.5 |
| 4 | car | 86.1 |
| 5 | construction_vehicle | 56.2 |
| 6 | motorcycle | 60.8 |
| 7 | pedestrian | 75.3 |
| 8 | traffic_cone | 68.2 |
| 9 | trailer | 63.7 |
| 10 | truck | 79.4 |
| 11 | driveable_surface | 96.8 |
| 12 | other_flat | 69.3 |
| 13 | sidewalk | 75.8 |
| 14 | terrain | 71.6 |
| 15 | manmade | 89.1 |
| 16 | vegetation | 85.7 |
| — | **mIoU** | **76.1** |

---

## SemanticKITTI Leaderboard Submission

### Submission Format

Predictions must be submitted as `.label` files matching the test set structure:

```
submission/
├── sequences/
│   ├── 11/
│   │   └── predictions/
│   │       ├── 000000.label
│   │       ├── 000001.label
│   │       └── ...
│   ├── 12/
│   │   └── predictions/
│   │       └── ...
│   └── ...  (sequences 11-21)
└── description.txt (optional)
```

### Label File Format

```python
# Each .label file: N × uint32 (matching the point count of corresponding .bin)
# Lower 16 bits: semantic label (using ORIGINAL label IDs, not mapped IDs)
# Upper 16 bits: instance ID (0 if not providing instance labels)

# Inverse learning map (evaluation → raw label IDs)
inv_learning_map = {
    0: 0,     # unlabeled
    1: 10,    # car
    2: 11,    # bicycle
    3: 15,    # motorcycle
    4: 18,    # truck
    5: 20,    # other-vehicle
    6: 30,    # person
    7: 31,    # bicyclist
    8: 32,    # motorcyclist
    9: 40,    # road
    10: 44,   # parking
    11: 48,   # sidewalk
    12: 49,   # other-ground
    13: 50,   # building
    14: 51,   # fence
    15: 70,   # vegetation
    16: 71,   # trunk
    17: 72,   # terrain
    18: 80,   # pole
    19: 81,   # traffic-sign
}

# Writing prediction labels
def write_predictions(pred_labels_mapped, output_path):
    """
    Args:
        pred_labels_mapped: np.array of shape (N,), class IDs in evaluation space (1-19)
        output_path: path to .label file
    """
    # Map back to original label space
    raw_labels = np.vectorize(inv_learning_map.get)(pred_labels_mapped)
    
    # Write as uint32 (no instance info → upper 16 bits = 0)
    raw_labels = raw_labels.astype(np.uint32)
    raw_labels.tofile(output_path)
```

### Submission Steps

1. **Generate predictions** for all test sequences (11-21)
2. **Verify** point counts match exactly (each `.label` must have same N as `.bin`)
3. **Package** into a zip file maintaining the directory structure
4. **Submit** at [codalab.lisn.upsaclay.fr/competitions/6280](https://codalab.lisn.upsaclay.fr/competitions/6280)

### Validation Before Submission

```python
import os
import numpy as np

def validate_submission(submission_dir, dataset_dir):
    """Check submission validity before uploading."""
    errors = []
    
    for seq in range(11, 22):
        seq_str = f'{seq:02d}'
        pred_dir = os.path.join(submission_dir, 'sequences', seq_str, 'predictions')
        velodyne_dir = os.path.join(dataset_dir, 'sequences', seq_str, 'velodyne')
        
        bin_files = sorted(os.listdir(velodyne_dir))
        
        for bin_file in bin_files:
            frame = bin_file.replace('.bin', '')
            label_file = os.path.join(pred_dir, f'{frame}.label')
            bin_path = os.path.join(velodyne_dir, bin_file)
            
            # Check file exists
            if not os.path.exists(label_file):
                errors.append(f'Missing: {label_file}')
                continue
            
            # Check point count matches
            n_points = os.path.getsize(bin_path) // (4 * 4)  # 4 floats × 4 bytes
            n_labels = os.path.getsize(label_file) // 4       # 1 uint32 × 4 bytes
            
            if n_points != n_labels:
                errors.append(f'{label_file}: {n_labels} labels vs {n_points} points')
            
            # Check label values are valid
            labels = np.fromfile(label_file, dtype=np.uint32)
            semantic = labels & 0xFFFF
            valid_ids = set(inv_learning_map.values())
            invalid = set(np.unique(semantic)) - valid_ids
            if invalid:
                errors.append(f'{label_file}: invalid label IDs {invalid}')
    
    return errors
```

---

## nuScenes-lidarseg Evaluation Protocol

### Submission Format

nuScenes uses a JSON-based submission format:

```
submission/
├── lidarseg/
│   └── test/
│       ├── {sample_data_token}_lidarseg.bin   # for each test sample
│       └── ...
└── test_submission.json
```

### test_submission.json

```json
{
    "meta": {
        "use_camera": false,
        "use_lidar": true,
        "use_radar": false,
        "use_map": false,
        "use_external": false
    }
}
```

### Label File Format (nuScenes)

```python
# Each .bin file: N × uint8 (matching point count of the sample)
# Each byte is the predicted class index (0-16)

def write_nuscenes_prediction(pred_labels, output_path):
    """
    Args:
        pred_labels: np.array of shape (N,), class IDs (0-16)
        output_path: path to _lidarseg.bin file
    """
    pred_labels = pred_labels.astype(np.uint8)
    pred_labels.tofile(output_path)
```

### Evaluation Metrics (nuScenes)

nuScenes-lidarseg reports:

| Metric | Description |
|--------|-------------|
| mIoU | Mean IoU across 16 classes (primary metric) |
| fwIoU | Frequency-weighted IoU |
| Per-class IoU | Individual class performance |

### Submission Steps

1. Generate predictions for all test keyframes
2. Create `test_submission.json` with metadata
3. Package as zip
4. Submit at [eval.ai/web/challenges/challenge-page/720](https://eval.ai/web/challenges/challenge-page/720)

---

## Inference Pipeline

### Standard Inference

```python
import torch
import numpy as np

def inference_single_scan(model, point_cloud_path, config):
    """Run inference on a single LiDAR scan."""
    model.eval()
    
    # Load point cloud
    points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
    
    # Preprocess: cylindrical voxelization
    voxel_coords, voxel_features, point_to_voxel = cylindrical_voxelize(
        points, 
        grid_size=config.grid_size,       # [480, 360, 32]
        point_cloud_range=config.range     # [r_min, r_max, theta_min, theta_max, z_min, z_max]
    )
    
    # To tensors
    coords_tensor = torch.from_numpy(voxel_coords).int().cuda()
    feats_tensor = torch.from_numpy(voxel_features).float().cuda()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            logits = model(feats_tensor, coords_tensor, points)  # (N, C)
    
    # Predictions
    predictions = logits.argmax(dim=1).cpu().numpy()  # (N,)
    
    return predictions
```

### Test-Time Augmentation (TTA)

TTA improves mIoU by ~2-3% at the cost of increased inference time:

```python
def inference_with_tta(model, points, config, num_rotations=4, flip=True):
    """
    Test-time augmentation with rotations and flips.
    
    Combines predictions from multiple augmented views.
    """
    all_logits = []
    
    # Rotation augmentations
    angles = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    
    for angle in angles:
        # Rotate points
        rotated = rotate_points_z(points.copy(), angle)
        logits = inference_single(model, rotated, config)
        all_logits.append(logits)
        
        # Flip augmentations
        if flip:
            # X-flip
            flipped_x = rotated.copy()
            flipped_x[:, 0] *= -1
            logits_fx = inference_single(model, flipped_x, config)
            all_logits.append(logits_fx)
            
            # Y-flip
            flipped_y = rotated.copy()
            flipped_y[:, 1] *= -1
            logits_fy = inference_single(model, flipped_y, config)
            all_logits.append(logits_fy)
    
    # Average logits (or probabilities) across augmentations
    avg_logits = torch.stack(all_logits).mean(dim=0)
    predictions = avg_logits.argmax(dim=1)
    
    return predictions
```

---

## Inference Speed Benchmarks

### Single-Scan Latency

| Hardware | Batch Size | Latency (ms) | FPS | Notes |
|----------|-----------|-------------|-----|-------|
| RTX 2080 Ti (11GB) | 1 | 180 | 5.6 | Without TTA |
| RTX 3090 (24GB) | 1 | 120 | 8.3 | Without TTA |
| V100 (32GB) | 1 | 140 | 7.1 | Without TTA |
| A100 (40GB) | 1 | 85 | 11.8 | Without TTA |
| A100 (80GB) | 1 | 80 | 12.5 | Without TTA |
| RTX 3090 (TTA×12) | 1 | 1440 | 0.7 | 4 rotations + flips |

### Latency Breakdown (RTX 3090)

| Component | Time (ms) | Percentage |
|-----------|----------|------------|
| Voxelization (CPU→GPU) | 15 | 12.5% |
| Sparse Conv Encoder | 45 | 37.5% |
| Sparse Conv Decoder | 35 | 29.2% |
| Point Refinement | 18 | 15.0% |
| Post-processing | 7 | 5.8% |
| **Total** | **120** | **100%** |

### Throughput (Points per Second)

| Hardware | Points/sec (millions) |
|----------|---------------------|
| RTX 3090 | ~1.0M |
| A100 | ~1.5M |

### Comparison with Other Methods

| Method | mIoU (%) | Latency (ms) | Parameters |
|--------|----------|-------------|-----------|
| RangeNet++ | 52.2 | 80 | 50.4M |
| MinkowskiNet | 63.1 | 200 | 37.9M |
| SPVNAS | 66.4 | 180 | 12.5M |
| PolarNet | 54.3 | 90 | 13.6M |
| **Cylinder3D** | **68.9** | **120** | **6.15M** |

---

## Evaluation on Validation Set

### Running Evaluation Locally

```python
def evaluate_val_set(model, val_loader, num_classes=20, ignore_label=0):
    """Full validation set evaluation."""
    model.eval()
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            logits = model(batch['features'].cuda(), batch['coords'].cuda(), batch['points'])
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch['labels'].numpy()
            
            # Accumulate confusion matrix
            valid = labels != ignore_label
            confusion_matrix += compute_confusion(
                preds[valid], labels[valid], num_classes
            )
    
    # Compute metrics from accumulated confusion matrix
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    
    denom = tp + fp + fn
    per_class_iou = np.zeros(num_classes)
    valid_classes = []
    
    for c in range(1, num_classes):  # skip ignore_label=0
        if denom[c] > 0:
            per_class_iou[c] = tp[c] / denom[c]
            valid_classes.append(per_class_iou[c])
    
    miou = np.mean(valid_classes)
    
    # Overall accuracy
    overall_acc = tp[1:].sum() / confusion_matrix[1:, :].sum()
    
    return {
        'mIoU': miou * 100,
        'per_class_iou': per_class_iou * 100,
        'overall_accuracy': overall_acc * 100,
        'confusion_matrix': confusion_matrix
    }
```

### Validation Results Interpretation

**Expected validation results (SemanticKITTI sequence 08):**

| Metric | Expected Value |
|--------|---------------|
| mIoU | 65.0 – 67.5% |
| Overall accuracy | 90 – 92% |
| Road IoU | >90% |
| Car IoU | >95% |
| Person IoU | >60% |
| Motorcyclist IoU | >35% |

**Note:** Validation results are typically 1-2% lower than test results due to:
- Sequence 08 having slightly different class distribution
- Test set evaluation using hidden ground truth (no opportunity for overfitting)

---

## Error Analysis

### Confusion Pattern Analysis

Common confusion patterns in Cylinder3D:

| Predicted As | Commonly Confused With | Reason |
|-------------|----------------------|--------|
| bicycle | motorcycle | Similar shape at distance |
| person | pole | Thin vertical structures |
| motorcyclist | bicyclist | Similar pose |
| other-ground | road, parking | Ambiguous flat surfaces |
| fence | vegetation | Often adjacent; similar height |
| traffic-sign | pole | Mounted on poles |
| trunk | pole | Similar vertical cylinders |

### Distance-Based Performance

| Distance Range | mIoU (%) | Notes |
|---------------|----------|-------|
| 0 – 10 m | 78.5 | High point density |
| 10 – 20 m | 72.3 | Good density |
| 20 – 30 m | 65.1 | Moderate density |
| 30 – 50 m | 55.8 | Sparse, performance degrades |
| > 50 m | 42.3 | Very sparse |

### Visualization

```python
# Generate colored prediction visualization
def visualize_predictions(points, predictions, color_map):
    """Create colored point cloud for visualization."""
    colors = np.array([color_map[label] for label in predictions])
    
    # Save as .ply for visualization in CloudCompare/Open3D
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud("prediction_vis.ply", pcd)
```

---

## Reproducing Paper Results

### Checklist

- [ ] Use correct data splits (train: 00-07,09-10; val: 08; test: 11-21)
- [ ] Apply learning_map before training (28 → 19 classes + ignore)
- [ ] Grid size: 480 × 360 × 32
- [ ] Training: 40 epochs, Adam, lr=1e-3, weight_decay=1e-4
- [ ] Loss: WCE + Lovasz-softmax
- [ ] Augmentation: rotation, flip, scale, translate
- [ ] Evaluate with ignore_label=0 excluded from mIoU computation
- [ ] For test submission: map predictions back to original label space
- [ ] For TTA: use 4 rotations + X/Y flips (12 forward passes total)

### Known Differences from Paper

- `spconv` version affects exact results (v1.x vs v2.x API differences)
- PyTorch version and CUDA version can cause ~0.2% mIoU variation
- Random seed affects final result by ~0.3-0.5% mIoU
- Paper results may include unreported tricks (BN sync, specific augmentation order)
