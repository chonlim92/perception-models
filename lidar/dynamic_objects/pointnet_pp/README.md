# PointNet++ for 3D Point Cloud Processing

PointNet++ is a hierarchical neural network that applies PointNet recursively on nested partitions of the input point set. Unlike PointNet, which processes all points in a single global aggregation step and therefore struggles to capture local geometric structures, PointNet++ introduces a hierarchical feature learning framework that progressively abstracts larger and larger local regions. This allows the network to learn fine-grained local patterns at multiple scales while maintaining permutation invariance.

This implementation supports three primary tasks for autonomous driving perception:

- **3D Object Classification** -- Categorize entire point clouds into predefined classes (e.g., Car, Pedestrian, Cyclist).
- **3D Object Detection** -- Localize and classify objects in a scene by predicting oriented 3D bounding boxes.
- **3D Semantic Segmentation** -- Assign a class label to every point in the input cloud.

Key advantages over the original PointNet:

- Captures local geometric structures through hierarchical grouping
- Handles non-uniform point density via Multi-Scale Grouping (MSG) and Multi-Resolution Grouping (MRG)
- Achieves significantly better performance on fine-grained recognition tasks

---

## Architecture Overview

```
INPUT POINT CLOUD (N x 3+C)
         |
         v
+--------------------------------------------------+
|           SET ABSTRACTION LAYER 1 (SA1)          |
|                                                  |
|  Sampling: FPS selects N1 centroids              |
|  Grouping: Ball query with radius r1             |
|  PointNet: Shared MLPs -> Max Pool               |
|                                                  |
|  MSG Variant:                                    |
|  +----------+  +----------+  +----------+       |
|  | radius=r1|  | radius=r2|  | radius=r3|       |
|  | scale 1  |  | scale 2  |  | scale 3  |       |
|  +----+-----+  +----+-----+  +----+-----+       |
|       |              |              |             |
|       +---------+----+----+---------+            |
|                 | Concat  |                      |
|                 +---------+                      |
+--------------------------------------------------+
         |
         v  (N1 x d1)
+--------------------------------------------------+
|           SET ABSTRACTION LAYER 2 (SA2)          |
|                                                  |
|  Sampling: FPS selects N2 centroids (N2 < N1)   |
|  Grouping: Ball query with radius r2 > r1       |
|  PointNet: Shared MLPs -> Max Pool               |
+--------------------------------------------------+
         |
         v  (N2 x d2)
+--------------------------------------------------+
|           SET ABSTRACTION LAYER 3 (SA3)          |
|                                                  |
|  Sampling: FPS selects N3 centroids (N3 < N2)   |
|  Grouping: Ball query with radius r3 > r2       |
|  PointNet: Shared MLPs -> Max Pool               |
+--------------------------------------------------+
         |
         v  (N3 x d3)
         |
    +----+----+
    |         |
    v         v
+-------+  +-------------------------------------------+
| CLASS |  |     SEGMENTATION (Feature Propagation)    |
+-------+  +-------------------------------------------+
    |              |
    v              v
+-------+  +--------------------------------------------------+
| FC    |  |     FEATURE PROPAGATION LAYER 1 (FP1)            |
| Layers|  |                                                  |
|  |    |  |  Interpolate: distance-weighted from N3 to N2    |
|  v    |  |  Skip connection: concatenate SA2 features       |
| Pred  |  |  Unit PointNet: Shared MLPs                      |
+-------+  +--------------------------------------------------+
                   |
                   v  (N2 x d2')
           +--------------------------------------------------+
           |     FEATURE PROPAGATION LAYER 2 (FP2)            |
           |                                                  |
           |  Interpolate: distance-weighted from N2 to N1    |
           |  Skip connection: concatenate SA1 features       |
           |  Unit PointNet: Shared MLPs                      |
           +--------------------------------------------------+
                   |
                   v  (N1 x d1')
           +--------------------------------------------------+
           |     FEATURE PROPAGATION LAYER 3 (FP3)            |
           |                                                  |
           |  Interpolate: distance-weighted from N1 to N     |
           |  Skip connection: concatenate input features     |
           |  Unit PointNet: Shared MLPs                      |
           +--------------------------------------------------+
                   |
                   v  (N x d0')
           +--------------------------------------------------+
           |  Per-Point Classification (1x1 Conv -> Softmax)  |
           +--------------------------------------------------+
                   |
                   v
           PER-POINT SEMANTIC LABELS (N x num_classes)
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.10 or higher (GPU version recommended)
- CUDA 11.2+ and cuDNN 8.1+ (for GPU acceleration)

### Install via pip

```bash
cd lidar/dynamic_objects/pointnet_pp
pip install -r requirements.txt
```

### Requirements

The `requirements.txt` contains:

```
tensorflow>=2.10.0
numpy>=1.22.0
open3d>=0.15.0
scipy>=1.8.0
scikit-learn>=1.1.0
pyyaml>=6.0
tqdm>=4.64.0
matplotlib>=3.5.0
tensorboard>=2.10.0
h5py>=3.7.0
```

### Verify Installation

```bash
python -c "import tensorflow as tf; print(tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## Project Structure

```
pointnet_pp/
|-- configs/
|   |-- classification.yaml       # Config for ModelNet40 classification
|   |-- detection_kitti.yaml      # Config for KITTI 3D object detection
|   |-- segmentation_s3dis.yaml   # Config for S3DIS semantic segmentation
|   +-- msg_config.yaml           # Multi-Scale Grouping parameters
|
|-- tensorflow/
|   |-- models/
|   |   |-- pointnet2_cls.py      # Classification model
|   |   |-- pointnet2_det.py      # Detection model
|   |   +-- pointnet2_seg.py      # Segmentation model
|   |-- layers/
|   |   |-- set_abstraction.py    # Set Abstraction (SA) layer
|   |   |-- feature_propagation.py# Feature Propagation (FP) layer
|   |   |-- pointnet_sa_module.py # PointNet SA module with MSG
|   |   |-- sampling.py           # Farthest Point Sampling
|   |   |-- grouping.py           # Ball query and kNN grouping
|   |   +-- tf_ops/               # Custom TF ops (FPS, ball query)
|   |-- losses/
|   |   |-- focal_loss.py         # Focal loss for class imbalance
|   |   +-- detection_loss.py     # Combined cls + bbox regression loss
|   |-- train.py                  # Training entry point
|   |-- evaluate.py               # Evaluation entry point
|   +-- inference.py              # Single-sample inference
|
|-- pytorch/
|   |-- models/                   # PyTorch model implementations
|   |-- layers/                   # PyTorch layer implementations
|   |-- train.py
|   |-- evaluate.py
|   +-- inference.py
|
|-- scripts/
|   |-- download_data.sh          # Download KITTI and ModelNet40
|   |-- prepare_data.py           # Convert raw data to training format
|   |-- visualize_results.py      # 3D and BEV visualization
|   +-- export_saved_model.py     # Export to TF SavedModel / ONNX
|
|-- tests/
|   |-- test_set_abstraction.py   # Unit tests for SA layers
|   |-- test_feature_propagation.py
|   |-- test_sampling.py
|   +-- test_model_forward.py     # End-to-end forward pass tests
|
|-- docs/
|   |-- architecture.md           # Detailed architecture documentation
|   +-- training_guide.md         # Extended training guide
|
+-- README.md                     # This file
```

---

## Dataset Preparation

### KITTI 3D Object Detection Dataset

The KITTI dataset is the primary benchmark used for 3D object detection evaluation in autonomous driving.

**Step 1: Download the dataset**

```bash
# Run the automated download script
bash scripts/download_data.sh --dataset kitti --output_dir /data/kitti

# This downloads:
#   - Velodyne point clouds (29 GB)
#   - Camera calibration files
#   - Training labels
#   - Left color images (for reference)
```

Alternatively, download manually from https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d:

1. Download "Velodyne point clouds" (data_object_velodyne.zip)
2. Download "Training labels" (data_object_label_2.zip)
3. Download "Camera calibration matrices" (data_object_calib.zip)

**Step 2: Organize the directory structure**

After downloading, ensure the following layout:

```
/data/kitti/
|-- training/
|   |-- velodyne/        # 7481 .bin point cloud files
|   |-- label_2/         # 7481 .txt label files
|   +-- calib/           # 7481 .txt calibration files
+-- testing/
    |-- velodyne/        # 7518 .bin point cloud files
    +-- calib/           # 7518 .txt calibration files
```

**Step 3: Prepare the data for training**

```bash
python scripts/prepare_data.py \
    --dataset kitti \
    --input_dir /data/kitti \
    --output_dir /data/kitti_processed \
    --num_points 16384 \
    --split_ratio 0.8 \
    --create_val_split
```

This script performs:
- Point cloud cropping to the forward-facing region
- Ground truth box extraction and class mapping
- Train/validation split generation (using the standard KITTI split or a custom ratio)
- Point cloud normalization and downsampling to a fixed number of points

### ModelNet40 (for Classification)

```bash
bash scripts/download_data.sh --dataset modelnet40 --output_dir /data/modelnet40
python scripts/prepare_data.py --dataset modelnet40 --input_dir /data/modelnet40 --output_dir /data/modelnet40_processed --num_points 1024
```

### S3DIS (for Semantic Segmentation)

```bash
bash scripts/download_data.sh --dataset s3dis --output_dir /data/s3dis
python scripts/prepare_data.py --dataset s3dis --input_dir /data/s3dis --output_dir /data/s3dis_processed --num_points 4096 --block_size 1.0
```

---

## Training

### Classification (ModelNet40)

```bash
python tensorflow/train.py \
    --config configs/classification.yaml \
    --data_dir /data/modelnet40_processed \
    --num_points 1024 \
    --batch_size 32 \
    --epochs 250 \
    --learning_rate 0.001 \
    --optimizer adam \
    --scheduler cosine \
    --use_msg \
    --log_dir logs/cls_modelnet40
```

### 3D Object Detection (KITTI)

```bash
python tensorflow/train.py \
    --config configs/detection_kitti.yaml \
    --data_dir /data/kitti_processed \
    --num_points 16384 \
    --batch_size 8 \
    --epochs 200 \
    --learning_rate 0.002 \
    --optimizer adam \
    --scheduler step \
    --step_size 40 \
    --gamma 0.5 \
    --use_msg \
    --augment \
    --log_dir logs/det_kitti
```

### Semantic Segmentation (S3DIS)

```bash
python tensorflow/train.py \
    --config configs/segmentation_s3dis.yaml \
    --data_dir /data/s3dis_processed \
    --num_points 4096 \
    --batch_size 16 \
    --epochs 150 \
    --learning_rate 0.001 \
    --optimizer adam \
    --scheduler cosine \
    --use_msg \
    --log_dir logs/seg_s3dis
```

### Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--num_points` | Number of input points per sample | 1024 (cls), 4096 (seg), 16384 (det) |
| `--use_msg` | Enable Multi-Scale Grouping | Recommended for non-uniform density |
| `--batch_size` | Training batch size | 8-32 depending on GPU memory |
| `--learning_rate` | Initial learning rate | 0.001-0.002 |
| `--scheduler` | LR schedule type | cosine, step, exponential |
| `--augment` | Enable data augmentation | Random rotation, jitter, scaling |
| `--dropout_rate` | Dropout in FC layers | 0.4-0.5 |
| `--bn_momentum` | Batch normalization momentum | 0.9 |

### Mixed Precision Training

Enable mixed precision (FP16) to reduce memory usage and accelerate training on compatible GPUs (Volta and later):

```bash
python tensorflow/train.py \
    --config configs/detection_kitti.yaml \
    --data_dir /data/kitti_processed \
    --mixed_precision \
    --batch_size 16 \
    --log_dir logs/det_kitti_fp16
```

### Multi-GPU Training

For distributed training across multiple GPUs using TensorFlow MirroredStrategy:

```bash
python tensorflow/train.py \
    --config configs/detection_kitti.yaml \
    --data_dir /data/kitti_processed \
    --multi_gpu \
    --gpus 0,1,2,3 \
    --batch_size 32 \
    --log_dir logs/det_kitti_multigpu
```

The effective batch size equals `batch_size * num_gpus`. Learning rate is automatically scaled linearly with the number of GPUs, with a warmup period of 5 epochs.

---

## Evaluation

### Evaluate a Trained Model

```bash
# Classification
python tensorflow/evaluate.py \
    --config configs/classification.yaml \
    --data_dir /data/modelnet40_processed \
    --checkpoint logs/cls_modelnet40/best_model.h5 \
    --num_points 1024 \
    --batch_size 32

# Detection
python tensorflow/evaluate.py \
    --config configs/detection_kitti.yaml \
    --data_dir /data/kitti_processed \
    --checkpoint logs/det_kitti/best_model.h5 \
    --num_points 16384 \
    --iou_threshold 0.7 \
    --score_threshold 0.3

# Segmentation
python tensorflow/evaluate.py \
    --config configs/segmentation_s3dis.yaml \
    --data_dir /data/s3dis_processed \
    --checkpoint logs/seg_s3dis/best_model.h5 \
    --num_points 4096 \
    --batch_size 16
```

### Expected Output Format

**Classification:**

```
=== Classification Results ===
Overall Accuracy: 92.8%
Mean Class Accuracy: 90.1%

Per-class accuracy:
  airplane    : 99.0%
  bathtub     : 93.5%
  bed         : 97.0%
  ...
```

**Detection (KITTI format):**

```
=== 3D Object Detection Results (IoU=0.7) ===
              Easy     Moderate    Hard
  Car         85.42    76.13       68.91
  Pedestrian  62.34    55.87       50.12
  Cyclist     72.15    61.03       56.78

Mean AP (moderate): 64.34
```

**Segmentation:**

```
=== Semantic Segmentation Results ===
Overall Accuracy: 87.3%
Mean IoU: 63.5%

Per-class IoU:
  ceiling  : 92.1%
  floor    : 97.4%
  wall     : 81.2%
  ...
```

---

## Inference

### Single Point Cloud Inference

```bash
python tensorflow/inference.py \
    --config configs/detection_kitti.yaml \
    --checkpoint logs/det_kitti/best_model.h5 \
    --input /data/kitti/training/velodyne/000001.bin \
    --task detection \
    --num_points 16384 \
    --score_threshold 0.5 \
    --output results/000001_predictions.txt
```

### Programmatic Inference

```python
from tensorflow.inference import PointNet2Predictor

predictor = PointNet2Predictor(
    config_path="configs/detection_kitti.yaml",
    checkpoint_path="logs/det_kitti/best_model.h5",
    task="detection"
)

# Load a raw point cloud (N x 4: x, y, z, intensity)
import numpy as np
points = np.fromfile("000001.bin", dtype=np.float32).reshape(-1, 4)

# Run inference
predictions = predictor.predict(points, score_threshold=0.5)

for pred in predictions:
    print(f"Class: {pred['class']}, Score: {pred['score']:.3f}, "
          f"Box: {pred['bbox_3d']}")
```

### Batch Inference

```bash
python tensorflow/inference.py \
    --config configs/detection_kitti.yaml \
    --checkpoint logs/det_kitti/best_model.h5 \
    --input_dir /data/kitti/testing/velodyne/ \
    --task detection \
    --num_points 16384 \
    --score_threshold 0.5 \
    --output_dir results/kitti_test/ \
    --batch_size 4
```

This processes all `.bin` files in the input directory and writes per-frame prediction files to the output directory.

---

## Visualization

The `scripts/visualize_results.py` tool provides interactive 3D visualization and Bird's Eye View (BEV) rendering of point clouds with predictions.

### 3D Interactive Visualization

```bash
python scripts/visualize_results.py \
    --input /data/kitti/training/velodyne/000001.bin \
    --predictions results/000001_predictions.txt \
    --ground_truth /data/kitti/training/label_2/000001.txt \
    --calib /data/kitti/training/calib/000001.txt \
    --mode 3d
```

### Bird's Eye View (BEV) Visualization

```bash
python scripts/visualize_results.py \
    --input /data/kitti/training/velodyne/000001.bin \
    --predictions results/000001_predictions.txt \
    --ground_truth /data/kitti/training/label_2/000001.txt \
    --mode bev \
    --x_range -40 40 \
    --y_range -40 40 \
    --resolution 0.1 \
    --save_path figures/000001_bev.png
```

### Segmentation Visualization

```bash
python scripts/visualize_results.py \
    --input /data/s3dis_processed/Area_5/office_1.npy \
    --predictions results/office_1_seg.npy \
    --mode 3d \
    --color_by segmentation \
    --class_map configs/s3dis_classes.yaml
```

### Options

| Flag | Description |
|------|-------------|
| `--mode` | Visualization mode: `3d` (interactive Open3D) or `bev` (matplotlib) |
| `--color_by` | Coloring scheme: `height`, `intensity`, `segmentation`, `class` |
| `--show_gt` | Overlay ground truth boxes (green) alongside predictions (red) |
| `--save_path` | Save rendering to file instead of displaying interactively |
| `--point_size` | Point rendering size (default: 2.0) |

---

## Results

### Classification -- ModelNet40

| Method | Input | Overall Accuracy | Mean Class Accuracy |
|--------|-------|-----------------|---------------------|
| PointNet++ (SSG) | 1024 points | 91.9% | 89.2% |
| PointNet++ (MSG) | 1024 points | 92.8% | 90.1% |
| PointNet++ (MSG + normals) | 5000 points + normals | 93.3% | 91.0% |

### 3D Object Detection -- KITTI val split (AP 3D, IoU=0.7 for Car, 0.5 for Ped/Cyc)

| Class | Easy | Moderate | Hard |
|-------|------|----------|------|
| Car | 85.42 | 76.13 | 68.91 |
| Pedestrian | 62.34 | 55.87 | 50.12 |
| Cyclist | 72.15 | 61.03 | 56.78 |

### Semantic Segmentation -- S3DIS Area 5

| Method | mIoU | Overall Accuracy |
|--------|------|-----------------|
| PointNet++ (SSG) | 60.1% | 85.7% |
| PointNet++ (MSG) | 63.5% | 87.3% |

---

## Citation

If you use this implementation in your research, please cite the original PointNet++ paper:

```
@inproceedings{qi2017pointnetplusplus,
  title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author={Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={30},
  year={2017}
}
```

Additionally, if you use the detection components, consider citing:

```
@inproceedings{qi2018frustum,
  title={Frustum PointNets for 3D Object Detection from RGB-D Data},
  author={Qi, Charles R. and Liu, Wei and Wu, Chenxia and Su, Hao and Guibas, Leonidas J.},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={918--927},
  year={2018}
}
```

---

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

---

## Acknowledgments

- The original PointNet++ implementation by Charles R. Qi et al. at Stanford University.
- The KITTI Vision Benchmark Suite by Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago.
- The S3DIS dataset by Stanford University.
- Open3D for point cloud visualization utilities.
- The TensorFlow and PyTorch communities for framework support.
