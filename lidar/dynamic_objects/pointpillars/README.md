# PointPillars: Real-Time 3D Object Detection from LiDAR Point Clouds

A TensorFlow 2.x implementation of PointPillars for fast, accurate 3D object detection
from LiDAR point clouds. This module detects cars, pedestrians, and cyclists in autonomous
driving scenarios, achieving real-time inference at approximately 62 Hz on an RTX 2080 Ti.

---

## What Problem Does This Solve?

### The 3D Detection Challenge

Autonomous vehicles must understand the 3D world around them in real time. A LiDAR sensor
mounted on the vehicle produces a "point cloud" -- a collection of 100,000+ 3D points
representing the surfaces of nearby objects (cars, pedestrians, cyclists, buildings, roads).

The perception system must process this point cloud and output:
- **Where** each object is (3D position: x, y, z)
- **How big** it is (dimensions: width, length, height)
- **Which direction** it faces (heading angle)
- **What** it is (class: car, pedestrian, cyclist)

All of this must happen fast enough to react to the environment.

### Why Real-Time Matters

A LiDAR sensor produces a new scan every 50-100 milliseconds (10-20 Hz). The detection
system must process each scan before the next one arrives. If processing takes longer
than the scan period, the vehicle is "driving blind" -- making decisions based on stale
information.

```
                    LiDAR Timeline (10 Hz sensor)
    |----100ms----|----100ms----|----100ms----|----100ms----|
    ^             ^             ^             ^             ^
    Scan 1        Scan 2        Scan 3        Scan 4        Scan 5

    If detection takes 500ms (VoxelNet at 2 Hz):
    |=========================500ms=========================|
    ^                                                       ^
    Start Scan 1                                            Finish!
    (Scans 2, 3, 4, 5 are all MISSED -- driving blind)

    If detection takes 16ms (PointPillars at 62 Hz):
    |16ms|
    ^    ^
    Start Finish  (84ms of headroom for other tasks)
```

### What Makes PointPillars Special

Traditional 3D detection methods (VoxelNet, SECOND) divide 3D space into small cubes
and apply expensive 3D convolutions. PointPillars takes a radically different approach:

1. Divide the ground plane into vertical columns called "pillars"
2. Use a small neural network (PointNet) to compress each pillar into a feature vector
3. Arrange these features on a 2D grid (creating a "pseudo-image")
4. Process with standard 2D convolutions (fast, GPU-optimized)

This eliminates ALL 3D convolutions, achieving 31x faster processing than VoxelNet
while maintaining competitive accuracy.

---

## How It Works (Intuition)

Think of PointPillars like taking an aerial photograph of a city from a helicopter.
From above, you can see where every car, person, and cyclist is located -- their 2D
footprint on the ground tells you everything needed for navigation. You do not need to
process the full 3D volume of the city to find objects on the street.

PointPillars applies this same logic to LiDAR point clouds. Rather than processing
points in full 3D (which requires expensive 3D convolutions), it first "squashes" the
height information into a compact encoding per ground-plane location. The result is a
2D map -- a "pseudo-image" where each pixel contains rich information about what is
above that ground location (how tall, what shape, what surface properties).

This pseudo-image is then processed by a standard 2D convolutional neural network --
the same type of network used for image classification and object detection, which has
been optimized on GPUs for decades. The network produces 3D bounding boxes (including
height and heading) by combining the encoded height information with spatial patterns
learned from the 2D layout.

The key insight is that for objects resting on the ground (cars, pedestrians, cyclists),
their identity and location are primarily determined by their 2D footprint in the bird's
eye view. The height axis can be compressed early without losing the information needed
for accurate detection.

---

## Architecture

```
                           PointPillars Architecture
    ========================================================================

    STAGE 1: Point Cloud Encoding         STAGE 2: 2D Feature Extraction
    (Irregular -> Regular)                 (Standard CNN Operations)

    +----------------+                     +---------------------------+
    |  Raw Points    |                     |    Pseudo-Image           |
    |  (N x 4)      |                     |    (B, 64, 496, 432)     |
    |  x, y, z, i   |                     |    [like a 64-channel     |
    +-------+--------+                     |     aerial photograph]    |
            |                              +-------------+-------------+
            v                                            |
    +-------+--------+                                   v
    | Pillarization  |                     +-------------+-------------+
    | Assign each    |                     |    2D Backbone            |
    | point to x-y   |                     |    Block 1: (B,64,248,216)|
    | grid cell      |                     |    Block 2: (B,128,124,108|
    +-------+--------+                     |    Block 3: (B,256,62,54) |
            |                              +-------------+-------------+
            v                                            |
    +-------+--------+                                   v
    | Feature        |                     +-------------+-------------+
    | Augmentation   |                     |    Neck (FPN)             |
    | 4 -> 9 features|                     |    Upsample + Concat     |
    | per point      |                     |    Output: (B,384,248,216)|
    +-------+--------+                     +-------------+-------------+
            |                                            |
            v                                            v
    +-------+--------+                     +-------------+-------------+
    | PointNet       |                     |    Detection Head (SSD)   |
    | per Pillar     |                     |    cls: (B, 18, 248, 216) |
    | Linear(9,64)   |                     |    box: (B, 42, 248, 216) |
    | + BN + ReLU    |                     |    dir: (B, 12, 248, 216) |
    | + MaxPool      |                     +-------------+-------------+
    +-------+--------+                                   |
            |                                            v
            v                              +-------------+-------------+
    +-------+--------+                     |    Post-Processing        |
    | Scatter to     |                     |    Score Threshold        |
    | 2D Grid        |-------------------->|    Box Decoding           |
    | (B, P, 64) ->  |                     |    Rotated NMS            |
    | (B, 64, H, W)  |                     +-------------+-------------+
    +----------------+                                   |
                                                         v
                                           +-------------+-------------+
                                           |  3D Detections            |
                                           |  (x,y,z,w,l,h,yaw,cls)   |
                                           +---------------------------+
```

**Data Flow Summary:**

1. **Pillar Feature Net**: Assigns points to x-y grid cells (pillars), augments each
   point with 9 features (x,y,z,intensity + offsets from pillar mean and center), applies
   a shared linear layer (9->64) + BatchNorm + ReLU, and max-pools across points to
   produce one 64-dim vector per pillar.

2. **Scatter**: Places each pillar's feature vector at its (x,y) grid location on a
   dense BEV canvas of shape (496 x 432 x 64) for KITTI.

3. **Backbone2D**: Three convolutional blocks with stride-2 downsampling produce
   multi-scale feature maps at 1/2, 1/4, and 1/8 resolution.

4. **Neck (FPN)**: Upsamples all scales to the same resolution and concatenates them
   (384 channels total).

5. **Detection Head**: 1x1 convolutions predict class scores, box regression residuals,
   and direction classification for each of 6 anchors at every spatial location.

6. **Post-Processing**: Decodes box residuals, applies score thresholding and rotated
   NMS to produce final 3D detections.

---

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

Confirm TensorFlow sees your GPU:

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}, GPU: {tf.config.list_physical_devices(\"GPU\")}')"
```

Expected output (with GPU):
```
TensorFlow 2.10.0, GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| No GPU detected | Verify CUDA installation: `nvidia-smi` should show your GPU |
| CUDA version mismatch | Match TF version to CUDA: TF 2.10 needs CUDA 11.2 |
| Out of memory during import | Set `TF_GPU_ALLOCATOR=cuda_malloc_async` |
| Import errors | Verify all dependencies: `pip check` |
| Slow training start | First epoch is slow due to tf.function tracing; this is normal |

---

## Dataset Setup

### KITTI 3D Object Detection

Download from [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):

```
data/kitti/
├── training/
│   ├── calib/          # Calibration files (7481 .txt files)
│   ├── image_2/        # Left color images (7481 .png files)
│   ├── label_2/        # 3D bounding box labels (7481 .txt files)
│   └── velodyne/       # LiDAR point clouds (7481 .bin files)
├── testing/
│   ├── calib/          # (7518 .txt files)
│   ├── image_2/        # (7518 .png files)
│   └── velodyne/       # (7518 .bin files)
└── ImageSets/
    ├── train.txt       # Training split (3712 samples)
    ├── val.txt         # Validation split (3769 samples)
    └── test.txt        # Test split (7518 samples)
```

Verify download:
```bash
# Check file counts
ls data/kitti/training/velodyne/ | wc -l    # Should be 7481
ls data/kitti/training/label_2/ | wc -l     # Should be 7481

# Check a sample point cloud
python -c "import numpy as np; pc = np.fromfile('data/kitti/training/velodyne/000000.bin', dtype=np.float32).reshape(-1, 4); print(f'Points: {pc.shape[0]}, Range: x[{pc[:,0].min():.1f}, {pc[:,0].max():.1f}]')"
```

### nuScenes

Download from [nuScenes](https://www.nuscenes.org/nuscenes):

```
data/nuscenes/
├── maps/
├── samples/
│   ├── CAM_FRONT/
│   ├── LIDAR_TOP/       # LiDAR keyframe sweeps
│   └── ...
├── sweeps/
│   └── LIDAR_TOP/       # Intermediate LiDAR sweeps (for multi-sweep)
├── v1.0-trainval/
│   ├── category.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── sample_annotation.json
│   └── ...
└── v1.0-test/
    └── ...
```

---

## Quick Start

### 1. Prepare Data

Generate ground truth database and preprocessing files:

```bash
# KITTI preprocessing (~5 minutes)
# This creates: info files (metadata), GT database (for augmentation)
python scripts/create_data.py \
    --dataset kitti \
    --root_path data/kitti \
    --out_path data/kitti/processed
```

Expected output:
```
Processing training split...
  Extracting info for 3712 samples... done (42s)
  Building GT database: 28742 Car, 4487 Ped, 1627 Cyc objects
Processing validation split...
  Extracting info for 3769 samples... done (43s)
Saved to data/kitti/processed/
```

For nuScenes:
```bash
# nuScenes preprocessing (~30 minutes due to dataset size)
python scripts/create_data.py \
    --dataset nuscenes \
    --root_path data/nuscenes \
    --out_path data/nuscenes/processed \
    --version v1.0-trainval
```

### 2. Train

```bash
# Train on KITTI (Car only, ~4 hours on RTX 3090)
python tensorflow/train.py \
    --config configs/pointpillars_kitti_car.yaml \
    --data_root data/kitti/processed \
    --output_dir experiments/pp_kitti_car \
    --batch_size 4 \
    --epochs 80 \
    --learning_rate 0.002
```

Expected training progress:
```
Epoch [1/80] Loss: 8.42, LR: 0.0002, Pos_anchors: 127/batch
Epoch [5/80] Loss: 2.13, LR: 0.0012, Pos_anchors: 142/batch
Epoch [20/80] Loss: 1.21, LR: 0.0020, Pos_anchors: 156/batch
Epoch [40/80] Loss: 0.92, LR: 0.0015, Pos_anchors: 148/batch
Epoch [80/80] Loss: 0.73, LR: 0.000002, Pos_anchors: 151/batch
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

Expected output:
```
Car AP (R40):
  3D  AP: Easy=87.75, Moderate=78.39, Hard=75.18
  BEV AP: Easy=90.12, Moderate=87.56, Hard=85.23
Inference speed: 62.1 Hz (16.1 ms/frame)
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

This produces a visualization showing the point cloud with colored 3D bounding boxes
around detected objects, and saves detection results in KITTI format.

---

## Configuration

Configuration files are in YAML format under `configs/`. Each parameter controls a
specific physical or architectural property:

| Parameter | Description | KITTI Default | Physical Meaning |
|-----------|-------------|:-------------:|------------------|
| `point_cloud_range` | [x_min, y_min, z_min, x_max, y_max, z_max] | [0, -39.68, -3, 69.12, 39.68, 1] | Detection volume in meters (front/side/height) |
| `voxel_size` | Pillar [x, y, z] in meters | [0.16, 0.16, 4] | Ground resolution: 16cm cells, full height |
| `max_points_per_voxel` | Points sampled per pillar | 100 | Limits memory; excess randomly dropped |
| `max_num_voxels` | Max non-empty pillars | 12000 | Limits encoder computation |
| `pillar_feat_dim` | PointNet output dimension | 64 | Richness of per-pillar representation |
| `backbone_layers` | Conv layers per block | [4, 6, 6] | Network depth at each scale |
| `backbone_filters` | Channels per block | [64, 128, 256] | Network width at each scale |
| `neck_upsample_strides` | Upsample factors | [1, 2, 4] | How much to enlarge each scale |
| `neck_filters` | Channels per FPN level | [128, 128, 128] | Width of fused features |
| `anchor_sizes` | [w, l, h] per class | Car: [1.6, 3.9, 1.56] | Template box dimensions (meters) |
| `anchor_rotations` | Angles (radians) | [0, pi/2] | Two orientations per class |
| `nms_iou_threshold` | NMS IoU cutoff | 0.5 | How much overlap triggers suppression |
| `score_threshold` | Min detection confidence | 0.3 | Below this, predictions are discarded |

---

## Performance

### KITTI 3D Object Detection Benchmark (3D AP @ IoU 0.7/0.5/0.5)

| Class | Easy | Moderate | Hard |
|-------|:----:|:--------:|:----:|
| Car (IoU 0.7) | 87.75 | 78.39 | 75.18 |
| Pedestrian (IoU 0.5) | 57.30 | 52.29 | 47.19 |
| Cyclist (IoU 0.5) | 79.14 | 63.57 | 56.98 |

The Moderate difficulty is the primary metric for comparison. PointPillars achieves 78.39%
AP for cars, meaning it correctly detects about 78% of moderately difficult cars with
high localization accuracy.

### nuScenes Detection Benchmark

| Metric | Value | Interpretation |
|--------|:-----:|----------------|
| mAP | 40.1 | Mean detection accuracy across 10 classes |
| NDS | 55.0 | Combined detection + localization quality |
| mATE | 0.33 m | Average position error (about 1 foot) |
| mASE | 0.26 | Average size error (scale mismatch) |
| mAOE | 0.42 rad | Average heading error (~24 degrees) |

### Inference Speed

| Hardware | Speed (Hz) | Latency (ms) | Suitable For |
|----------|:---------:|:------------:|--------------|
| RTX 2080 Ti | 62 | 16.1 | Development, real-time inference |
| RTX 3090 | 88 | 11.4 | Fast development |
| V100 (16GB) | 54 | 18.5 | Cloud deployment |
| Xavier AGX | 18 | 55.6 | Edge/vehicle deployment |

All measurements are with batch size 1, including NMS post-processing, after 50 warmup
iterations.

---

## Comparison with Other Methods

| Method | KITTI Car 3D AP (Mod.) | nuScenes mAP | Speed (Hz) | 3D Conv | Notes |
|--------|:----------------------:|:------------:|:----------:|:-------:|-------|
| **PointPillars** | **78.39** | **40.1** | **62** | No | Pillar encoding, fastest |
| VoxelNet | 65.11 | - | 4.4 | Yes (dense) | First end-to-end, too slow |
| SECOND | 76.48 | - | 26 | Sparse 3D | Good accuracy, moderate speed |
| CenterPoint | 79.23 | 60.3 | 16 | Sparse 3D | Anchor-free, higher accuracy |
| PV-RCNN | 83.61 | - | 8 | Sparse 3D | Two-stage, highest accuracy |
| Part-A2 | 79.47 | - | 12 | Sparse 3D | Part-aware aggregation |

**Key Tradeoffs:**
- PointPillars offers the best speed-accuracy tradeoff for real-time deployment. At 62 Hz,
  it processes frames 6x faster than the LiDAR sensor produces them, leaving substantial
  headroom for other perception tasks.
- CenterPoint and PV-RCNN achieve higher accuracy but require sparse 3D convolutions and
  cannot meet real-time constraints on edge devices.
- VoxelNet is historically significant but impractical for deployment (500ms per frame).
- For applications where accuracy is paramount and latency is less critical (e.g., offline
  annotation), PV-RCNN is preferred. For real-time driving, PointPillars is the standard.

---

## Documentation Guide

Detailed documentation is available in the `docs/` directory:

| Document | What It Covers |
|----------|---------------|
| [docs/research_summary.md](docs/research_summary.md) | Why point clouds need encoding, evolution from VoxelNet to PointPillars, speed analysis, comparisons with other methods, radar adaptation |
| [docs/model_architecture.md](docs/model_architecture.md) | Layer-by-layer architecture walkthrough, tensor shapes, pillar creation, PointNet encoding, scatter, backbone, FPN, detection head, NMS |
| [docs/training_guide.md](docs/training_guide.md) | Anchor generation, data augmentation (GT sampling, geometric transforms), one-cycle LR, KITTI vs nuScenes training, debugging |
| [docs/evaluation_guide.md](docs/evaluation_guide.md) | KITTI and nuScenes metrics, inference speed measurement, ablation studies, deployment considerations |
| [docs/annotation_guide.md](docs/annotation_guide.md) | Data annotation standards and procedures |
| [docs/data_collection.md](docs/data_collection.md) | Data collection protocols and requirements |

For newcomers, the recommended reading order is:
1. This README (overview)
2. `docs/research_summary.md` (understand the problem and solution)
3. `docs/model_architecture.md` (understand how the model works)
4. `docs/training_guide.md` (learn to train the model)
5. `docs/evaluation_guide.md` (learn to evaluate results)

---

## File Structure

```
pointpillars/
├── configs/
│   ├── pointpillars_kitti_car.yaml       # KITTI car-only config
│   ├── pointpillars_kitti_3class.yaml    # KITTI 3-class config
│   └── pointpillars_nuscenes.yaml        # nuScenes config
├── docs/
│   ├── research_summary.md              # Research background and comparisons
│   ├── model_architecture.md            # Detailed architecture documentation
│   ├── training_guide.md                # Training procedures and tips
│   ├── evaluation_guide.md              # Evaluation metrics and ablations
│   ├── annotation_guide.md              # Data annotation standards
│   └── data_collection.md              # Data collection protocols
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

---

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

---

## Frequently Asked Questions

### What hardware do I need to run this?

For inference only, any NVIDIA GPU with at least 4 GB VRAM (e.g., GTX 1070 or newer)
is sufficient. For training, 8 GB VRAM minimum (RTX 2080 Ti or V100 recommended). CPU-only
execution is possible but approximately 50x slower and not recommended.

### How does PointPillars handle different LiDAR sensors?

PointPillars is sensor-agnostic. It works with any LiDAR that produces (x, y, z, intensity)
point clouds. Different sensors may require adjusting:
- `point_cloud_range`: Match the sensor's maximum range
- `voxel_size`: Sparser sensors benefit from larger pillars
- `max_num_voxels`: More beams produce more non-empty pillars
- `max_points_per_voxel`: Adjust based on point density

### Can PointPillars detect objects not in the training set?

No. PointPillars is a supervised detector -- it can only detect object classes it was
trained on. To detect new classes, you need labeled training data for those classes and
must retrain or fine-tune the model.

### How accurate is PointPillars at long range?

Accuracy degrades significantly with distance due to point sparsity:
- 0-20m: ~92% AP (Car), many points per object
- 20-40m: ~81% AP (Car), moderate points
- 40-60m: ~64% AP (Car), few points
- 60-80m: ~41% AP (Car), very few points

For safety-critical applications at long range, sensor fusion (combining LiDAR with
camera and radar) is recommended.

### How does weather affect performance?

LiDAR performance degrades in adverse weather:
- Light rain: ~5% AP reduction (some laser returns scattered by droplets)
- Heavy rain: ~15-20% AP reduction (significant point cloud degradation)
- Fog: ~10-25% AP reduction depending on density
- Snow: ~5-15% AP reduction (snowflakes produce false points)

PointPillars itself does not explicitly handle weather degradation. Production systems
typically add preprocessing steps (noise filtering) or fuse with radar (weather-robust).

### Can I use this with a camera-only system?

No. PointPillars requires 3D LiDAR point clouds as input. For camera-only 3D detection,
consider methods like BEVDet, DETR3D, or FCOS3D. However, PointPillars can be combined
with camera features through fusion architectures like BEVFusion.

### What is the difference between this and CenterPoint?

Both use pillar-based encoding, but differ in the detection head:
- PointPillars: Anchor-based (predefined box templates, requires NMS)
- CenterPoint: Anchor-free (predicts object centers as heatmap peaks, minimal NMS)

CenterPoint achieves higher accuracy (+20 mAP on nuScenes) but is 4x slower. For
real-time applications, PointPillars remains competitive.

---

## Known Limitations

1. **Distance dependence:** Detection quality degrades significantly beyond 50m due to
   point sparsity. Objects at the edge of the detection range may be missed entirely.

2. **Small objects:** Very small objects (traffic cones, debris) produce very few LiDAR
   returns and are difficult to detect reliably.

3. **Vertical ambiguity:** The pillar representation compresses height information,
   making it harder to distinguish objects at different heights in the same (x,y) column
   (e.g., a bridge overpass vs. a vehicle below it).

4. **No temporal reasoning:** Each frame is processed independently. Moving objects are
   not tracked across frames, and motion information is not used (except in the nuScenes
   multi-sweep variant).

5. **Fixed detection range:** The model's detection range is fixed at training time.
   Objects outside this range are invisible to the model regardless of their size.

---

## Understanding the Output Format

When you run inference, the model outputs 3D bounding boxes. Each detection contains:

```
Field           Description                              Units / Convention
-----------     -----------                              ------------------
x, y, z         Center of the 3D box                     meters, LiDAR frame
w, l, h         Width, length, height of the box         meters
yaw             Rotation around the vertical (z) axis    radians
class_id        Object category (0=Car, 1=Ped, 2=Cyc)   integer
score           Detection confidence                     [0.0, 1.0]
```

Converting from LiDAR frame to camera frame (for KITTI submission):

```
# KITTI provides calibration matrices:
# P2: camera projection matrix (3x4)
# R0_rect: rectification rotation (3x3)
# Tr_velo_to_cam: LiDAR-to-camera transform (3x4)

# Transform 3D center from LiDAR to camera frame:
point_cam = R0_rect @ Tr_velo_to_cam @ [x, y, z, 1].T

# Note: KITTI camera frame has Y pointing down, Z pointing forward
# LiDAR frame has X pointing forward, Y pointing left, Z pointing up
```

---

## Troubleshooting Common Issues

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| OOM during training | Batch size too large or too many pillars | Reduce batch_size or max_pillars |
| AP stuck at 0% | Coordinate frame mismatch | Check LiDAR-to-camera calibration |
| Training loss explodes | Learning rate too high | Reduce max_lr, check gradient clipping |
| Very slow training | Data loading bottleneck | Increase num_workers, enable prefetch |
| Low AP on Pedestrian only | Too few training samples | Increase GT sampling ratio for Pedestrian |
| NaN in loss | Invalid point cloud or bad augmentation | Add input validation, check for empty pillars |
| Inference much slower than expected | Not using GPU or debug mode | Verify CUDA available, use @tf.function |

---

## Contributing

Contributions are welcome. When submitting changes, please:

1. Run the existing test suite: `python -m pytest tests/ -v`
2. Verify training converges on a small subset (100 samples, 5 epochs)
3. Report before/after AP numbers if changing model architecture or augmentation
4. Follow the existing code style and documentation conventions
5. Include a clear description of what problem your change solves

---

## License

This implementation is released under the Apache License 2.0. See the LICENSE file for details.

The KITTI dataset is provided for academic research only. The nuScenes dataset is provided
under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license. Please review
and comply with the respective dataset licenses before use.
