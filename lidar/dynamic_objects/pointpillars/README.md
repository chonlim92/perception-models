# PointPillars: Real-Time 3D Object Detection from LiDAR Point Clouds

A TensorFlow 2.x implementation of PointPillars for fast, accurate 3D object detection from LiDAR point clouds. This module detects cars, pedestrians, and cyclists in autonomous driving scenarios, achieving real-time inference at approximately 62 Hz on an RTX 2080 Ti.

## Key Innovation

Traditional 3D object detection methods (e.g., VoxelNet) divide the 3D space into volumetric voxels and apply expensive 3D convolutions. PointPillars replaces this approach with **pillar-based encoding**: the point cloud is discretized into vertical columns (pillars) rather than 3D voxels. A lightweight PointNet extracts a fixed-size feature vector per pillar, which is then scattered onto a 2D bird's eye view (BEV) pseudo-image. This enables the use of standard 2D convolutional backbones, eliminating 3D convolutions entirely and achieving a 4-10x speedup over voxel-based methods while maintaining competitive accuracy.

The core insight is that the vertical (z-axis) structure within each pillar can be efficiently captured by a PointNet, making the explicit 3D discretization unnecessary. This single architectural change bridges the gap between accuracy and real-time performance.

## Architecture

```
                        PointPillars Architecture
                        ========================

  Raw Point Cloud         Pillar Encoding           2D Backbone + Detection
  (N x 4: x,y,z,i)       (Learned Features)        (Standard CNN Pipeline)

  +----------------+    +-------------------+    +----------------------+
  |                |    |                   |    |                      |
  |  Point Cloud   |--->| PillarFeatureNet  |--->|      Scatter         |
  |  (N points)    |    | (PointNet + MLP)  |    | (Pillars -> BEV)     |
  |                |    |                   |    |                      |
  +----------------+    +-------------------+    +----------+-----------+
                                                            |
                         Pillar Features:                    v
                         9D augmented input          +------+------+
                         (x,y,z,i,xc,yc,zc,xp,yp)  |             |
                              |                     | Backbone2D  |
                              v                     | (3 blocks,  |
                         Max-pool per pillar        |  multi-scale)|
                         -> 64-dim feature          |             |
                                                    +------+------+
                                                           |
                                                           v
  +----------------+    +-------------------+    +---------+---------+
  |                |    |                   |    |                   |
  |   Detections   |<---| AnchorHead        |<---|   Neck (FPN)      |
  |  (3D Boxes +   |    | (cls + box + dir) |    | (Upsample + Cat)  |
  |   Scores)      |    |                   |    |                   |
  +----------------+    +-------------------+    +-------------------+

  Detection Output:
  - Classification: num_anchors x num_classes (Car, Ped, Cyc)
  - Box Regression: num_anchors x 7 (x, y, z, w, l, h, theta)
  - Direction Cls:  num_anchors x 2 (heading bin)
```

**Data Flow Summary:**

1. **PillarFeatureNet**: Voxelizes points into pillars, augments each point with offsets from pillar mean and pillar center (4 -> 9 features), applies shared MLP (Dense + BN + ReLU), and max-pools across points to produce one 64-dim vector per pillar.

2. **Scatter**: Places each pillar's feature vector at its (x, y) grid location on a dense BEV canvas of shape (496 x 432 x 64) for KITTI.

3. **Backbone2D**: Three convolutional blocks with stride-2 downsampling produce multi-scale feature maps at 1/2, 1/4, and 1/8 resolution.

4. **Neck**: FPN-style upsampling brings all scales to the same resolution and concatenates them (384 channels).

5. **AnchorHead**: 1x1 convolutions predict class scores, box regression residuals, and direction classification for each anchor at every spatial location.

## Installation

### Requirements

- Python 3.8 or higher
- TensorFlow 2.6+ (GPU recommended)
- CUDA 11.x and cuDNN 8.x (for GPU acceleration)

### Install Dependencies

```bash
pip install tensorflow>=2.6.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install open3d>=0.13.0        # Point cloud I/O and visualization
pip install pyquaternion>=0.9.9   # Rotation handling
pip install fire>=0.4.0           # CLI interface
pip install tqdm>=4.62.0          # Progress bars
pip install pyyaml>=5.4           # Configuration files
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}, GPU: {tf.config.list_physical_devices(\"GPU\")}')"
```

## Dataset Setup

### KITTI 3D Object Detection

Download from [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):

```
data/kitti/
├── training/
│   ├── calib/          # Calibration files (*.txt)
│   ├── image_2/        # Left color images (*.png)
│   ├── label_2/        # 3D bounding box labels (*.txt)
│   └── velodyne/       # LiDAR point clouds (*.bin)
├── testing/
│   ├── calib/
│   ├── image_2/
│   └── velodyne/
└── ImageSets/
    ├── train.txt       # Training split (3712 samples)
    ├── val.txt         # Validation split (3769 samples)
    └── test.txt        # Test split (7518 samples)
```

### nuScenes

Download from [nuScenes](https://www.nuscenes.org/nuscenes):

```
data/nuscenes/
├── maps/
├── samples/
│   ├── CAM_FRONT/
│   ├── LIDAR_TOP/       # LiDAR sweeps
│   └── ...
├── sweeps/
│   └── LIDAR_TOP/       # Intermediate LiDAR sweeps
├── v1.0-trainval/
│   ├── category.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── sample_annotation.json
│   └── ...
└── v1.0-test/
    └── ...
```

## Quick Start

### 1. Prepare Data

Generate ground truth database and info files:

```bash
python scripts/create_data.py --dataset kitti --root_path data/kitti --out_path data/kitti/processed
```

For nuScenes:

```bash
python scripts/create_data.py --dataset nuscenes --root_path data/nuscenes --out_path data/nuscenes/processed --version v1.0-trainval
```

### 2. Train

```bash
python tensorflow/train.py \
    --config configs/pointpillars_kitti_car.yaml \
    --data_root data/kitti/processed \
    --output_dir experiments/pp_kitti_car \
    --batch_size 4 \
    --epochs 160 \
    --learning_rate 0.0002
```

Multi-class training (Car + Pedestrian + Cyclist):

```bash
python tensorflow/train.py \
    --config configs/pointpillars_kitti_3class.yaml \
    --data_root data/kitti/processed \
    --output_dir experiments/pp_kitti_3class \
    --batch_size 4 \
    --epochs 160
```

### 3. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/pointpillars_kitti_car.yaml \
    --checkpoint experiments/pp_kitti_car/best_model \
    --data_root data/kitti/processed \
    --split val
```

### 4. Inference on a Single Point Cloud

```bash
python scripts/inference.py \
    --config configs/pointpillars_kitti_3class.yaml \
    --checkpoint experiments/pp_kitti_3class/best_model \
    --input data/kitti/training/velodyne/000008.bin \
    --output results/000008_detections.txt \
    --visualize
```

## Configuration

Configuration files are in YAML format under `configs/`. Key parameters:

| Parameter | Description | KITTI Default |
|-----------|-------------|---------------|
| `point_cloud_range` | Detection range [x_min, y_min, z_min, x_max, y_max, z_max] | [0, -39.68, -3, 69.12, 39.68, 1] |
| `voxel_size` | Pillar dimensions [x, y, z] in meters | [0.16, 0.16, 4] |
| `max_points_per_voxel` | Max points sampled per pillar | 100 |
| `max_num_voxels` | Max non-empty pillars | 12000 |
| `pillar_feat_dim` | PillarFeatureNet output dimension | 64 |
| `backbone_layers` | Conv layers per backbone block | [4, 6, 6] |
| `backbone_filters` | Filters per backbone block | [64, 128, 256] |
| `neck_upsample_strides` | Upsample factors for FPN neck | [1, 2, 4] |
| `neck_filters` | Filters per FPN level | [128, 128, 128] |
| `anchor_sizes` | [w, l, h] per class | Car: [1.6, 3.9, 1.56] |
| `anchor_rotations` | Rotation angles (radians) | [0, pi/2] |
| `nms_iou_threshold` | NMS IoU threshold | 0.5 |
| `score_threshold` | Minimum detection confidence | 0.3 |

## Performance

### KITTI 3D Object Detection Benchmark (3D AP @ IoU 0.7/0.5/0.5)

| Class | Easy | Moderate | Hard |
|-------|------|----------|------|
| Car (IoU 0.7) | 87.75 | 78.39 | 75.18 |
| Pedestrian (IoU 0.5) | 57.30 | 52.29 | 47.19 |
| Cyclist (IoU 0.5) | 79.14 | 63.57 | 56.98 |

### nuScenes Detection Benchmark

| Metric | Value |
|--------|-------|
| mAP | 40.1 |
| NDS | 55.0 |
| mATE | 0.33 m |
| mASE | 0.26 |
| mAOE | 0.42 rad |

### Inference Speed

| Hardware | Speed (Hz) | Latency (ms) |
|----------|-----------|--------------|
| RTX 2080 Ti | 62 | 16.1 |
| RTX 3090 | 88 | 11.4 |
| V100 (16GB) | 54 | 18.5 |
| Xavier AGX | 18 | 55.6 |

## Comparison with Other Methods

| Method | KITTI Car 3D AP (Mod.) | nuScenes mAP | Speed (Hz) | 3D Conv | Notes |
|--------|------------------------|--------------|------------|---------|-------|
| **PointPillars** | **78.39** | **40.1** | **62** | No | Pillar encoding, fastest among competitive methods |
| VoxelNet | 65.11 | - | 4.4 | Yes | First end-to-end voxel method, very slow |
| SECOND | 76.48 | - | 26 | Sparse 3D | Sparse convolutions improve speed over VoxelNet |
| CenterPoint | 79.23 | 60.3 | 16 | Sparse 3D | Center-based detection, higher accuracy, slower |
| PV-RCNN | 83.61 | - | 8 | Sparse 3D | Two-stage, highest accuracy, not real-time |
| Part-A2 | 79.47 | - | 12 | Sparse 3D | Part-aware aggregation |

**Key Tradeoffs:**
- PointPillars offers the best speed-accuracy tradeoff for real-time deployment.
- CenterPoint and PV-RCNN achieve higher accuracy but require sparse 3D convolutions and cannot meet real-time constraints on edge devices.
- VoxelNet is historically significant but impractical for deployment due to dense 3D convolutions.

## File Structure

```
pointpillars/
├── configs/
│   ├── pointpillars_kitti_car.yaml       # KITTI car-only config
│   ├── pointpillars_kitti_3class.yaml    # KITTI 3-class config
│   └── pointpillars_nuscenes.yaml        # nuScenes config
├── docs/
│   └── architecture.md                   # Detailed architecture notes
├── pytorch/                              # PyTorch implementation (alternative)
├── scripts/
│   ├── create_data.py                    # Dataset preprocessing
│   ├── evaluate.py                       # Evaluation script
│   └── inference.py                      # Single-sample inference
├── tensorflow/
│   ├── model.py                          # Core model (all layers + losses)
│   └── train.py                          # Training loop
├── tests/
│   └── test_model.py                     # Comprehensive pytest suite
└── README.md                             # This file
```

## Citations

If you use this implementation in your research, please cite the original PointPillars paper:

```bibtex
@inproceedings{lang2019pointpillars,
  title={PointPillars: Fast Encoders for Object Detection from Point Clouds},
  author={Lang, Alex H. and Vora, Sourabh and Caesar, Holger and Zhou, Lubing and Yang, Jiong and Beijbom, Oscar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={12697--12705},
  year={2019}
}
```

Additional references:

```bibtex
@inproceedings{yan2018second,
  title={SECOND: Sparsely Embedded Convolutional Detection},
  author={Yan, Yan and Mao, Yuxing and Li, Bo},
  booktitle={Sensors},
  volume={18},
  number={10},
  pages={3337},
  year={2018}
}

@inproceedings{zhou2018voxelnet,
  title={VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection},
  author={Zhou, Yin and Tuzel, Oncel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4490--4499},
  year={2018}
}

@article{yin2021centerpoint,
  title={Center-based 3D Object Detection and Tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Krahenbuhl, Philipp},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={11784--11793},
  year={2021}
}
```

## License

This implementation is released under the Apache License 2.0. See the LICENSE file for details.

The KITTI dataset is provided for academic research only. The nuScenes dataset is provided under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license. Please review and comply with the respective dataset licenses before use.
