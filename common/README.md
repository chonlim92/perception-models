# Common Utilities

Shared components used across all perception models in this repository. These utilities provide the foundation that all models build upon — data loading, evaluation metrics, coordinate transformations, and visualization.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Datasets Module](#datasets-module)
3. [Metrics Module](#metrics-module)
4. [Transforms Module](#transforms-module)
5. [Visualization Module](#visualization-module)
6. [Registry System](#registry-system)
7. [Usage Examples](#usage-examples)

---

## Architecture Overview

```
common/
├── datasets/
│   ├── nuscenes_dataset.py     # Full nuScenes loader (all sensors)
│   └── kitti_dataset.py        # KITTI 3D detection loader
├── metrics/
│   ├── detection_metrics.py    # mAP, NDS, TP errors
│   ├── segmentation_metrics.py # mIoU, per-class IoU
│   ├── map_metrics.py          # Chamfer distance, vectorized AP
│   ├── tracking_metrics.py     # AMOTA, AMOTP, IDS
│   └── temporal_metrics.py     # Map consistency, streaming AP
├── transforms/
│   ├── augmentations.py        # 3D and 2D data augmentation
│   └── coordinates.py          # Coordinate system conversions
├── visualization/
│   ├── bev_visualizer.py       # BEV plotting (boxes, maps, predictions)
│   ├── pointcloud_viz.py       # 3D point cloud with Open3D
│   └── image_viz.py            # 2D image overlays
└── registry.py                 # Decorator-based model/dataset registry
```

---

## Datasets Module

### nuScenes Dataset Loader (`nuscenes_dataset.py`)

The nuScenes dataset loader handles the complexity of loading multi-modal, multi-frame data from the nuScenes dataset. It supports all sensor types and temporal sequences.

#### What It Loads

```
One sample from NuScenesDataset returns:
┌─────────────────────────────────────────────────────────────────┐
│                                                                   │
│  images: dict                                                     │
│    CAM_FRONT:      Tensor [3, H, W]  (normalized RGB)            │
│    CAM_FRONT_LEFT: Tensor [3, H, W]                              │
│    CAM_FRONT_RIGHT: Tensor [3, H, W]                             │
│    CAM_BACK:       Tensor [3, H, W]                              │
│    CAM_BACK_LEFT:  Tensor [3, H, W]                              │
│    CAM_BACK_RIGHT: Tensor [3, H, W]                              │
│                                                                   │
│  lidar_points: Tensor [N, 5]  (x, y, z, intensity, ring_index)  │
│                                                                   │
│  radar_points: dict (per radar sensor)                            │
│    RADAR_FRONT: Tensor [M, 18]  (x, y, z, dyn_prop, rcs, vx...)│
│                                                                   │
│  calibration: dict                                                │
│    intrinsics: dict of [3, 3] matrices per camera                │
│    extrinsics: dict of [4, 4] matrices (sensor → ego)            │
│                                                                   │
│  ego_pose: Tensor [4, 4]  (ego → world transform)               │
│                                                                   │
│  annotations: dict (for training)                                 │
│    boxes_3d: Tensor [K, 9]  (cx, cy, cz, w, l, h, yaw, vx, vy) │
│    labels: Tensor [K]  (class indices)                           │
│    track_ids: Tensor [K]  (tracking IDs for MOT)                 │
│                                                                   │
│  temporal: list of past samples (if sequence_length > 1)         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Features

- **Multi-modal**: Loads cameras, LiDAR, and radar simultaneously
- **Temporal sequences**: Loads past N frames with ego-pose for temporal models
- **Configurable sensors**: Load only the sensors you need (saves memory)
- **Coordinate alignment**: All data aligned to the ego frame at the current timestamp
- **Lazy loading**: Images/point clouds loaded on demand, not cached in RAM

#### Usage

```python
from common.datasets.nuscenes_dataset import NuScenesDataset

# Full multi-modal dataset
dataset = NuScenesDataset(
    root="data/nuscenes",
    split="train",                    # "train", "val", "test", "mini_train", "mini_val"
    sensors=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
             "LIDAR_TOP"],
    sequence_length=4,                # Load 4 frames (current + 3 past) for temporal models
    image_size=(900, 1600),           # Resize images
    point_cloud_range=[-50, -50, -5, 50, 50, 3],  # x_min, y_min, z_min, x_max, y_max, z_max
)

sample = dataset[0]
print(sample['images']['CAM_FRONT'].shape)  # [3, 900, 1600]
print(sample['lidar_points'].shape)          # [N, 5]
print(sample['annotations']['boxes_3d'].shape)  # [K, 9]
```

### KITTI Dataset Loader (`kitti_dataset.py`)

KITTI is an older but still widely-used benchmark, particularly for LiDAR-based 3D detection.

```python
from common.datasets.kitti_dataset import KITTIDataset

dataset = KITTIDataset(
    root="data/kitti",
    split="training",
    sensors=["image_2", "velodyne"],  # Left camera + LiDAR
)
```

---

## Metrics Module

### Detection Metrics (`detection_metrics.py`)

Implements the official nuScenes detection evaluation protocol.

#### How nuScenes Detection Evaluation Works

```
Step 1: Match predictions to ground truth
  - For each class, compute center-distance between all pred-GT pairs
  - Match using greedy assignment at thresholds: [0.5, 1.0, 2.0, 4.0] meters
  - A prediction is a True Positive (TP) if center distance < threshold

Step 2: Compute AP per class per threshold
  - Sort predictions by confidence score
  - Walk down the list, computing precision/recall at each point
  - AP = area under the precision-recall curve (40-point interpolation)

Step 3: Compute TP errors (for matched predictions only)
  - ATE: |pred_center - gt_center|  (Euclidean)
  - ASE: 1 - IoU(pred_box, gt_box) after alignment
  - AOE: |pred_yaw - gt_yaw| (smallest arc)
  - AVE: |pred_velocity - gt_velocity|
  - AAE: 1{pred_attribute != gt_attribute}

Step 4: Aggregate
  - mAP = mean of AP over all classes and distance thresholds
  - mTP_errors = mean of each TP error over all classes
  - NDS = (1/10) * [5 * mAP + sum(max(1 - mTP_error, 0))]
```

#### Usage

```python
from common.metrics.detection_metrics import compute_nuscenes_metrics

# predictions: list of dicts with 'translation', 'size', 'rotation', 'velocity', 'detection_score', 'detection_name'
# ground_truth: list of dicts in same format (from dataset annotations)
results = compute_nuscenes_metrics(predictions, ground_truth)

print(f"mAP:  {results['mAP']:.3f}")
print(f"NDS:  {results['NDS']:.3f}")
print(f"ATE:  {results['ATE']:.3f}")
print(f"ASE:  {results['ASE']:.3f}")
print(f"AOE:  {results['AOE']:.3f}")
print(f"AVE:  {results['AVE']:.3f}")
print(f"AAE:  {results['AAE']:.3f}")

# Per-class breakdown
for cls_name, cls_ap in results['per_class_ap'].items():
    print(f"  {cls_name}: AP={cls_ap:.3f}")
```

### Segmentation Metrics (`segmentation_metrics.py`)

Implements mean Intersection over Union (mIoU) for per-point or per-pixel semantic labeling.

```python
from common.metrics.segmentation_metrics import compute_miou

# predictions: Tensor [N] of class indices
# targets: Tensor [N] of class indices
miou, per_class_iou = compute_miou(predictions, targets, num_classes=20)
print(f"mIoU: {miou:.1f}%")
for cls_name, iou in zip(class_names, per_class_iou):
    print(f"  {cls_name}: {iou:.1f}%")
```

### Map Metrics (`map_metrics.py`)

Evaluates vectorized HD map predictions using Chamfer distance and AP.

```python
from common.metrics.map_metrics import compute_map_metrics

# pred_lines: list of polylines, each [N_points, 2] in BEV coordinates
# gt_lines: list of GT polylines
results = compute_map_metrics(
    pred_lines, gt_lines,
    thresholds=[0.5, 1.0, 1.5],  # meters
    categories=['lane_divider', 'road_boundary', 'pedestrian_crossing']
)
print(f"Chamfer Distance: {results['chamfer']:.2f}m")
print(f"AP@0.5m: {results['AP_0.5']:.1f}%")
print(f"AP@1.0m: {results['AP_1.0']:.1f}%")
```

### Tracking Metrics (`tracking_metrics.py`)

Implements AMOTA/AMOTP and other multi-object tracking metrics.

```python
from common.metrics.tracking_metrics import compute_tracking_metrics

# tracks: list of per-frame track predictions with track_ids
# gt_tracks: ground truth tracks
results = compute_tracking_metrics(tracks, gt_tracks)
print(f"AMOTA: {results['AMOTA']:.3f}")
print(f"AMOTP: {results['AMOTP']:.3f}")
print(f"ID Switches: {results['IDS']}")
```

### Temporal Metrics (`temporal_metrics.py`)

Measures temporal consistency of predictions across frames.

```python
from common.metrics.temporal_metrics import compute_map_consistency, compute_streaming_ap

# Measures how stable map predictions are between consecutive frames
consistency = compute_map_consistency(map_predictions_t0, map_predictions_t1, ego_motion)
print(f"Map Consistency: {consistency:.3f}")
```

---

## Transforms Module

### Augmentations (`augmentations.py`)

Data augmentation for 3D perception training. Handles the complexity of augmenting multiple data types consistently (you can't rotate an image without also rotating the 3D boxes and point clouds).

#### Available Augmentations

| Augmentation | Applies To | Description |
|-------------|-----------|-------------|
| `RandomFlipHorizontal` | Points, boxes, images | Flip along x-axis (left/right) |
| `RandomRotation` | Points, boxes | Random yaw rotation [-pi/4, pi/4] |
| `RandomScale` | Points, boxes | Scale [0.95, 1.05] |
| `RandomTranslation` | Points, boxes | Shift [±0.2m, ±0.2m, ±0.1m] |
| `PointDropout` | Points | Randomly drop 5% of points |
| `PointJitter` | Points | Add Gaussian noise (std=0.01) |
| `ColorJitter` | Images | Brightness/contrast/saturation |
| `ImageNormalize` | Images | Normalize to ImageNet stats |
| `GTSampling` | Points, boxes | Copy objects from other scenes |

#### Usage

```python
from common.transforms.augmentations import (
    Compose3D, RandomFlipHorizontal, RandomRotation, 
    RandomScale, GTSampling, ImageNormalize
)

# Training augmentations
train_transforms = Compose3D([
    RandomFlipHorizontal(prob=0.5),
    RandomRotation(angle_range=[-0.785, 0.785]),  # ±45 degrees
    RandomScale(scale_range=[0.95, 1.05]),
    GTSampling(db_path="data/gt_database/"),       # Paste GT objects from other scenes
    ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply to sample (augments all modalities consistently)
augmented_sample = train_transforms(sample)
```

### Coordinate Transforms (`coordinates.py`)

Handles all the coordinate system conversions needed in autonomous driving.

#### Coordinate Systems Explained

```
                     World Frame (global map)
                        x: East
                        y: North
                        z: Up
                           │
                     ego_pose (4x4)
                     changes every frame
                           │
                           ↓
                     Ego Frame (vehicle body)
                        x: Forward (driving direction)
                        y: Left
                        z: Up
                        Origin: rear axle center
                           │
              ┌────────────┼────────────────┐
              │            │                │
        cam_extrinsic  lidar_extrinsic  radar_extrinsic
        (4x4, fixed)  (4x4, fixed)    (4x4, fixed)
              │            │                │
              ↓            ↓                ↓
        Camera Frame  LiDAR Frame      Radar Frame
        x: right      x: right         x: forward
        y: down       y: forward       y: left
        z: forward    z: up            z: up
```

#### Key Transforms

```python
from common.transforms.coordinates import (
    CameraToEgo, EgoToCamera,
    EgoToWorld, WorldToEgo,
    WorldToBEV, BEVToWorld,
    project_to_image, unproject_from_image
)

# Convert LiDAR points to ego frame
ego_points = lidar_extrinsic @ lidar_points  # [4, 4] @ [4, N] = [4, N]

# Convert ego frame to world
world_points = ego_pose @ ego_points

# Project 3D point to camera pixel
# pixel = intrinsic @ extrinsic @ point_3d
pixel_coords = project_to_image(
    points_3d,           # [N, 3] in ego frame
    camera_intrinsic,    # [3, 3]
    camera_extrinsic,    # [4, 4] ego→camera
)  # Returns [N, 2] pixel coordinates + [N] depths

# Convert world coordinates to BEV grid indices
bev_indices = WorldToBEV(
    world_points,
    x_range=[-50, 50],   # meters
    y_range=[-50, 50],   # meters
    resolution=0.5,      # meters per pixel
)  # Returns [N, 2] grid indices (col, row)
```

---

## Visualization Module

### BEV Visualizer (`bev_visualizer.py`)

Creates top-down visualizations of detections, maps, and predictions.

```python
from common.visualization.bev_visualizer import BEVVisualizer

viz = BEVVisualizer(
    x_range=[-50, 50],    # meters
    y_range=[-50, 50],    # meters
    resolution=0.1,       # meters per pixel
)

# Draw ground truth boxes (green) and predictions (red)
viz.draw_boxes(gt_boxes, color='green', label='GT')
viz.draw_boxes(pred_boxes, color='red', label='Predictions')

# Draw map elements
viz.draw_polylines(lane_lines, color='yellow', linewidth=2)
viz.draw_polylines(road_boundaries, color='white', linewidth=1)

# Draw point cloud (top-down)
viz.draw_points(lidar_points[:, :2], color='gray', size=0.5)

# Save or show
viz.save("output_bev.png")
viz.show()
```

### Point Cloud Visualizer (`pointcloud_viz.py`)

Interactive 3D visualization using Open3D.

```python
from common.visualization.pointcloud_viz import PointCloudVisualizer

viz = PointCloudVisualizer()
viz.add_points(lidar_points, colormap='height')  # Color by height
viz.add_boxes_3d(gt_boxes, color=[0, 1, 0])      # Green GT boxes
viz.add_boxes_3d(pred_boxes, color=[1, 0, 0])    # Red predictions
viz.show()  # Opens interactive 3D viewer
```

### Image Visualizer (`image_viz.py`)

Overlays 3D information onto camera images.

```python
from common.visualization.image_viz import ImageVisualizer

viz = ImageVisualizer()
# Project 3D boxes onto camera image
viz.draw_projected_boxes(
    image, boxes_3d, camera_intrinsic, camera_extrinsic,
    color=(0, 255, 0), thickness=2
)
viz.save("image_with_boxes.png")
```

---

## Registry System

The registry provides a clean way to instantiate models, datasets, and metrics by name.

```python
from common.registry import Registry

MODELS = Registry("models")
DATASETS = Registry("datasets")

# Register a model
@MODELS.register("bevformer")
class BEVFormer(nn.Module):
    ...

# Register a dataset
@DATASETS.register("nuscenes")
class NuScenesDataset(Dataset):
    ...

# Later, instantiate by name (useful for config-driven training)
model = MODELS.build("bevformer", num_classes=10, bev_size=(200, 200))
dataset = DATASETS.build("nuscenes", root="data/nuscenes", split="train")
```

---

## Usage Examples

### Complete Training Pipeline

```python
import torch
from torch.utils.data import DataLoader
from common.datasets.nuscenes_dataset import NuScenesDataset
from common.metrics.detection_metrics import compute_nuscenes_metrics
from common.transforms.augmentations import Compose3D, RandomFlipHorizontal, RandomRotation
from common.visualization.bev_visualizer import BEVVisualizer

# 1. Setup data
train_transforms = Compose3D([
    RandomFlipHorizontal(prob=0.5),
    RandomRotation(angle_range=[-0.785, 0.785]),
])

dataset = NuScenesDataset(
    root="data/nuscenes",
    split="train",
    sensors=["CAM_FRONT", "LIDAR_TOP"],
    transforms=train_transforms,
)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# 2. Training loop
for batch in loader:
    images = batch['images']          # Dict of camera tensors
    points = batch['lidar_points']    # [B, N, 5]
    targets = batch['annotations']    # Dict with boxes_3d, labels
    
    # Forward pass through your model
    predictions = model(images, points)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

# 3. Evaluation
results = compute_nuscenes_metrics(all_predictions, all_ground_truth)
print(f"mAP: {results['mAP']:.3f}, NDS: {results['NDS']:.3f}")

# 4. Visualization
viz = BEVVisualizer(x_range=[-50, 50], y_range=[-50, 50])
viz.draw_boxes(predictions, color='red')
viz.draw_boxes(ground_truth, color='green')
viz.save("eval_result.png")
```

### Coordinate Transform Chain

```python
from common.transforms.coordinates import project_to_image

# Task: Draw LiDAR points on camera image

# 1. LiDAR points are in LiDAR frame
lidar_points = sample['lidar_points'][:, :3]  # [N, 3]

# 2. Transform to ego frame
lidar_to_ego = sample['calibration']['extrinsics']['LIDAR_TOP']  # [4, 4]
points_homo = torch.cat([lidar_points, torch.ones(N, 1)], dim=1)  # [N, 4]
points_ego = (lidar_to_ego @ points_homo.T).T[:, :3]  # [N, 3]

# 3. Project to camera image
cam_intrinsic = sample['calibration']['intrinsics']['CAM_FRONT']
cam_extrinsic = sample['calibration']['extrinsics']['CAM_FRONT']
pixels, depths = project_to_image(points_ego, cam_intrinsic, cam_extrinsic)

# 4. Filter points behind camera or outside image
valid = (depths > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] >= 0) & (pixels[:, 1] < H)
```

---

## Dependencies

The common utilities require:
- `torch >= 1.12`
- `numpy`
- `nuscenes-devkit` (for nuScenes data loading)
- `open3d` (for 3D visualization)
- `matplotlib` (for BEV and image plotting)
- `scipy` (for Hungarian matching in metrics)
- `pyquaternion` (for rotation handling)
