# Common Utilities

Shared components used across all perception models in this repository.

## Modules

### datasets/
Data loaders for standard autonomous driving datasets.

| File | Description |
|------|-------------|
| `nuscenes_dataset.py` | Full nuScenes loader (6 cameras, LiDAR, 5 radars, ego pose, calibration, temporal sequences) |
| `kitti_dataset.py` | KITTI 3D detection loader (stereo cameras + LiDAR + 3D boxes) |

### metrics/
Evaluation metrics for all perception tasks.

| File | Description |
|------|-------------|
| `detection_metrics.py` | mAP, NDS, ATE, ASE, AOE, AVE, AAE (nuScenes-style center-distance matching) |
| `segmentation_metrics.py` | mIoU, per-class IoU, pixel accuracy |
| `map_metrics.py` | Chamfer distance, Frechet distance, AP for vectorized map elements |
| `tracking_metrics.py` | AMOTA, AMOTP, MOTA, IDF1, ID switches, track fragmentation |
| `temporal_metrics.py` | Map consistency, streaming AP (latency-aware), temporal smoothness |

### transforms/
Data augmentation and coordinate system utilities.

| File | Description |
|------|-------------|
| `augmentations.py` | 3D augmentations (flip, rotate, scale, translate), image augmentations (color jitter, normalize), BEV augmentations |
| `coordinates.py` | Coordinate conversions (camera↔LiDAR↔ego↔world↔BEV), projection matrices, homogeneous transforms |

### visualization/
Plotting and visualization tools.

| File | Description |
|------|-------------|
| `bev_visualizer.py` | BEV plots with 3D boxes, map elements, predictions vs GT |
| `pointcloud_viz.py` | 3D point cloud visualization with Open3D |
| `image_viz.py` | 2D image overlays (projected boxes, segmentation masks, lane lines) |

### registry.py
Decorator-based model/dataset/metric registry for clean model instantiation.

## Usage

```python
from common.datasets.nuscenes_dataset import NuScenesDataset
from common.metrics.detection_metrics import compute_nuscenes_metrics
from common.transforms.coordinates import CameraToEgo, EgoToWorld
from common.visualization.bev_visualizer import BEVVisualizer

# Load data
dataset = NuScenesDataset(root="data/nuscenes", split="train", sensors=["CAM_FRONT", "LIDAR_TOP"])

# Evaluate
results = compute_nuscenes_metrics(predictions, ground_truth)
print(f"mAP: {results['mAP']:.3f}, NDS: {results['NDS']:.3f}")

# Visualize
viz = BEVVisualizer(x_range=[-50, 50], y_range=[-50, 50])
viz.draw_boxes(predictions, color='red')
viz.draw_boxes(ground_truth, color='green')
viz.save("output.png")
```
