# CenterPoint: Center-based 3D Object Detection and Tracking

CenterPoint is an anchor-free 3D object detection and tracking framework for LiDAR point clouds. It represents detected objects as points (centers) in Bird's Eye View (BEV), predicts a heatmap of object centers along with regression attributes (size, rotation, velocity), and performs simple online tracking via closest-distance matching with velocity prediction.

This approach eliminates the need for hand-crafted anchor configurations and achieves state-of-the-art performance on both the nuScenes and Waymo Open Dataset benchmarks.

---

## Architecture

```
Point Cloud (N x 5: x, y, z, intensity, timestamp)
       |
       v
+------------------+
| Dynamic Voxeliz. |  --> Voxel Grid (H x W x D x C_in)
+------------------+      e.g. (1440 x 1440 x 40 x 5)
       |
       v
+------------------+
| 3D Sparse Conv   |  --> Sparse Feature Volume (H x W x D x C)
| Backbone         |      e.g. (1440 x 1440 x 2 x 128)
+------------------+
       |
       v
+------------------+
| BEV Collapse     |  --> 2D BEV Feature Map (H x W x C')
| (Height Flatten) |      e.g. (1440 x 1440 x 256)
+------------------+
       |
       v
+------------------+
| 2D Conv Backbone |  --> Multi-scale BEV Features
| + Neck (FPN)     |      e.g. (360 x 360 x 512)
+------------------+
       |
       v
+------------------+
| Center Heads     |  --> Per-class heatmaps + regression branches
| (Multi-task)     |      heatmap: (360 x 360 x num_classes)
|                  |      reg: center_offset(2), height(1), size(3),
|                  |           rotation(2), velocity(2)
+------------------+
       |
       v
+------------------+
| Decode + NMS     |  --> 3D Bounding Boxes (K x 11)
+------------------+      [x, y, z, w, l, h, yaw, vx, vy, score, class]
       |
       v
+------------------+
| Online Tracker   |  --> Tracked Objects with IDs
| (Greedy Match)   |      Velocity-based motion prediction
+------------------+      + Closest-distance association
```

---

## Key Features

- **Anchor-free detection**: No anchor tuning required. Objects are represented as center points in BEV, eliminating the need for hand-crafted anchor sizes and aspect ratios.
- **Center-based representation**: Gaussian heatmap supervision with focal loss drives the network to predict object centers directly.
- **Velocity estimation**: Multi-sweep LiDAR input (up to 10 sweeps) enables per-object velocity regression for motion-aware detection.
- **Simple yet effective tracking**: Online tracking via greedy closest-distance matching with velocity-based motion prediction. No appearance features, no learned association, no graph optimization.
- **Multi-task heads**: Separate detection heads for different class groups (e.g., vehicles, pedestrians, cyclists) allow class-specific regression targets.
- **Dataset support**: Full pipelines for nuScenes and Waymo Open Dataset, including data preparation, training, and evaluation scripts.
- **TensorFlow 2.x implementation**: Built with TensorFlow 2.x and supports mixed precision (FP16) training for faster convergence and lower memory usage.

---

## Installation

### Requirements

- Python >= 3.8
- TensorFlow >= 2.10
- CUDA >= 11.2 (for GPU training)
- cuDNN >= 8.1

### Install Dependencies

```bash
pip install tensorflow>=2.10
pip install numpy scipy open3d nuscenes-devkit matplotlib tqdm pyquaternion
pip install numba  # for fast voxelization
```

### Optional: Verify GPU Setup

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Data Preparation

### 1. Download nuScenes

Download the full nuScenes dataset (v1.0-trainval) from [https://www.nuscenes.org/download](https://www.nuscenes.org/download) and extract it to `data/nuscenes/`.

### 2. Generate Info Files and GT Database

```bash
python scripts/prepare_data.py \
    --data-root data/nuscenes \
    --version v1.0-trainval \
    --max-sweeps 10
```

### 3. Expected Directory Structure

```
data/nuscenes/
├── maps/
├── samples/
│   └── LIDAR_TOP/
├── sweeps/
│   └── LIDAR_TOP/
├── v1.0-trainval/
│   ├── sample.json
│   ├── sample_data.json
│   ├── ego_pose.json
│   └── ...
├── infos_train.pkl        # generated
├── infos_val.pkl          # generated
└── gt_database/           # generated
    ├── car/
    ├── pedestrian/
    ├── bicycle/
    └── ...
```

---

## Training

### Single GPU

```bash
python tensorflow/train.py \
    --data-root data/nuscenes \
    --epochs 20 \
    --batch-size 4 \
    --lr 1e-3 \
    --mixed-precision
```

### Multi-GPU (Distributed)

```bash
python tensorflow/train.py \
    --data-root data/nuscenes \
    --epochs 20 \
    --batch-size 8 \
    --gpus 4 \
    --lr 1e-3 \
    --mixed-precision
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Total training epochs |
| `--batch-size` | 4 | Per-GPU batch size |
| `--lr` | 1e-3 | Initial learning rate |
| `--weight-decay` | 0.01 | AdamW weight decay |
| `--voxel-size` | [0.075, 0.075, 0.2] | Voxel dimensions in meters (x, y, z) |
| `--point-cloud-range` | [-54, -54, -5, 54, 54, 3] | Detection range [xmin, ymin, zmin, xmax, ymax, zmax] |
| `--max-sweeps` | 10 | Number of LiDAR sweeps to aggregate |
| `--mixed-precision` | False | Enable FP16 mixed precision training |
| `--grad-clip` | 35.0 | Gradient clipping norm |
| `--fade-epochs` | 5 | Disable data augmentation for last N epochs |

---

## Evaluation

```bash
python tensorflow/evaluate.py \
    --model-path checkpoints/best \
    --data-root data/nuscenes \
    --batch-size 8
```

This produces nuScenes detection and tracking metrics including mAP, NDS, AMOTA, and per-class breakdowns.

---

## Inference and Tracking Demo

```bash
python tensorflow/inference.py \
    --model-path checkpoints/best \
    --input data/nuscenes/samples/LIDAR_TOP/ \
    --output results/ \
    --visualize
```

The inference script performs detection and tracking on a sequence of LiDAR frames and optionally renders 3D bounding boxes in BEV for visualization.

---

## Results

### nuScenes Validation Set

| Model | mAP | NDS | AMOTA | FPS |
|-------|-----|-----|-------|-----|
| CenterPoint-Pillar | 50.3 | 60.2 | 63.8 | 25 |
| CenterPoint-Voxel | 56.4 | 64.8 | 67.2 | 16 |

- **mAP**: Mean Average Precision (detection quality)
- **NDS**: nuScenes Detection Score (composite metric including mAP, translation, scale, orientation, velocity, and attribute errors)
- **AMOTA**: Average Multi-Object Tracking Accuracy
- **FPS**: Frames per second on a single NVIDIA A100 GPU

---

## Comparison to Baselines

| Method | mAP | NDS | Notes |
|--------|-----|-----|-------|
| PointPillars | 40.1 | 55.0 | Pillar-based, fast inference |
| SECOND | 48.0 | 59.2 | Sparse convolution backbone |
| CenterPoint (ours) | 56.4 | 64.8 | Center-based, anchor-free |

CenterPoint achieves significant improvements over anchor-based methods by eliminating anchor design choices and directly regressing object properties from center points.

---

## Citation

```bibtex
@article{yin2021center,
  title={Center-based 3D Object Detection and Tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Krahenbuhl, Philipp},
  journal={CVPR},
  year={2021}
}
```

---

## License

Please refer to the LICENSE file in the repository root for usage terms.
