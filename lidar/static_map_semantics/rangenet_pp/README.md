# RangeNet++

Implementation and documentation for **RangeNet++: Fast and Accurate LiDAR Semantic Segmentation** (Milioto et al., IROS 2019).

## Overview

RangeNet++ performs semantic segmentation of 3D LiDAR point clouds by:

1. **Projecting** the 3D point cloud to a 2D range image via spherical projection
2. **Segmenting** the range image with a 2D CNN (DarkNet-53 encoder + U-Net decoder)
3. **Post-processing** with GPU-accelerated KNN to refine labels back in 3D space

This approach achieves real-time performance (~50Hz) while maintaining competitive accuracy on the SemanticKITTI benchmark.

## Key Features

- **Real-time inference:** 20-25ms per scan on a single GPU (40-50 FPS)
- **19-class segmentation** on SemanticKITTI (52.2% mIoU)
- **Range image representation:** Converts irregular 3D points to a structured 2D grid
- **KNN post-processing:** Fixes boundary artifacts from projection (+2.3 mIoU)
- **DarkNet-53 backbone:** Deep residual network without max pooling

## Architecture

```
Point Cloud (N x 4) --> Spherical Projection --> Range Image (64 x 2048 x 5)
                                                        |
                                                  DarkNet-53 Encoder
                                                        |
                                                  U-Net Decoder (skip connections)
                                                        |
                                                  Per-pixel Labels (64 x 2048 x 20)
                                                        |
                                                  KNN Post-Processing (K=5)
                                                        |
                                                  3D Point Labels (N x 1)
```

## Directory Structure

```
rangenet_pp/
├── README.md                           # This file
├── configs/
│   ├── rangenet_pp_semantickitti.yaml   # Full SemanticKITTI training config
│   └── rangenet_pp_darknet53.yaml      # DarkNet-53 backbone architecture config
├── docs/
│   ├── research_summary.md             # Paper summary and method comparison
│   ├── data_collection.md              # SemanticKITTI dataset details
│   ├── annotation_guide.md             # Class definitions and label mapping
│   ├── model_architecture.md           # Detailed architecture description
│   ├── training_guide.md               # Training procedure and tips
│   └── evaluation_guide.md             # Metrics, results, and benchmarks
├── pytorch/                            # PyTorch implementation (placeholder)
├── tensorflow/                         # TensorFlow implementation (placeholder)
├── scripts/                            # Utility scripts (placeholder)
└── tests/                              # Unit tests (placeholder)
```

## Installation

### Requirements

- Python >= 3.7
- PyTorch >= 1.7 (with CUDA support)
- NumPy >= 1.19
- PyYAML >= 5.3
- scikit-learn (for KNN baseline)
- Open3D (optional, for visualization)

### Setup

```bash
# Clone and enter directory
cd rangenet_pp

# Create conda environment
conda create -n rangenet python=3.8
conda activate rangenet

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# Install dependencies
pip install numpy pyyaml scikit-learn open3d tensorboard
```

## Dataset Preparation

### SemanticKITTI

1. Download the [KITTI Odometry Benchmark](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) velodyne point clouds.
2. Download [SemanticKITTI labels](http://www.semantic-kitti.org/dataset.html).
3. Organize as:

```
/data/semantickitti/dataset/sequences/
├── 00/
│   ├── velodyne/       # .bin point clouds
│   ├── labels/         # .label semantic annotations
│   ├── calib.txt
│   └── poses.txt
├── 01/
...
└── 21/
```

4. Update `dataset.root` in `configs/rangenet_pp_semantickitti.yaml`.

## Usage

### Training

```bash
# Train with DarkNet-53 backbone on SemanticKITTI
python pytorch/train.py \
    --config configs/rangenet_pp_semantickitti.yaml \
    --backbone configs/rangenet_pp_darknet53.yaml \
    --gpu 0

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 \
    pytorch/train.py \
    --config configs/rangenet_pp_semantickitti.yaml \
    --backbone configs/rangenet_pp_darknet53.yaml \
    --distributed
```

### Inference

```bash
# Run inference on a single scan
python pytorch/infer.py \
    --config configs/rangenet_pp_semantickitti.yaml \
    --checkpoint checkpoints/best_model.pth \
    --input /path/to/scan.bin \
    --output /path/to/predictions/

# Run inference on full test set
python pytorch/infer.py \
    --config configs/rangenet_pp_semantickitti.yaml \
    --checkpoint checkpoints/best_model.pth \
    --sequences 11 12 13 14 15 16 17 18 19 20 21 \
    --output predictions/
```

### Evaluation

```bash
# Evaluate on validation set (sequence 08)
python pytorch/evaluate.py \
    --config configs/rangenet_pp_semantickitti.yaml \
    --checkpoint checkpoints/best_model.pth \
    --split val

# Evaluate with KNN post-processing
python pytorch/evaluate.py \
    --config configs/rangenet_pp_semantickitti.yaml \
    --checkpoint checkpoints/best_model.pth \
    --split val \
    --knn --knn_k 5

# Generate predictions for benchmark submission
python pytorch/evaluate.py \
    --config configs/rangenet_pp_semantickitti.yaml \
    --checkpoint checkpoints/best_model.pth \
    --split test \
    --knn --knn_k 5 \
    --save_predictions predictions/
```

### Visualization

```bash
# Visualize predictions on a scan
python scripts/visualize.py \
    --scan /path/to/scan.bin \
    --predictions /path/to/prediction.label \
    --config configs/rangenet_pp_semantickitti.yaml
```

## Pretrained Models

| Model | Backbone | KNN | mIoU (val) | mIoU (test) | Download |
|-------|----------|-----|-----------|------------|----------|
| RangeNet21 | DarkNet-21 | No | 47.4% | 47.4% | [link] |
| RangeNet53 | DarkNet-53 | No | 49.9% | 49.9% | [link] |
| RangeNet53++ | DarkNet-53 | K=5 | 52.2% | 52.2% | [link] |

Pretrained weights available from the [original authors' repository](https://github.com/PRBonn/lidar-bonnetal).

## Results

### SemanticKITTI Test Set (19 classes)

| Method | mIoU | FPS | Real-time |
|--------|------|-----|-----------|
| PointNet++ | 20.1% | ~1 | No |
| SqueezeSeg | 29.5% | 83 | Yes |
| SqueezeSegV2 | 39.7% | 67 | Yes |
| **RangeNet53++** | **52.2%** | **40** | **Yes** |

### Per-Class Highlights

- Best: Road (91.8%), Car (91.4%), Building (87.4%)
- Challenging: Motorcyclist (4.6%), Other-vehicle (20.0%), Motorcycle (25.7%)

## Configuration

Key parameters in `configs/rangenet_pp_semantickitti.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `range_image.width` | 2048 | Azimuth resolution (1024 for faster) |
| `range_image.height` | 64 | Vertical resolution (beam count) |
| `training.batch_size` | 4 | Batch size (limited by GPU memory) |
| `training.epochs` | 150 | Number of training epochs |
| `training.optimizer.learning_rate` | 0.01 | Initial learning rate |
| `post_processing.knn.k` | 5 | KNN neighbors for post-processing |
| `post_processing.knn.search_radius` | 1.0 | KNN search radius (meters) |

## References

```bibtex
@inproceedings{milioto2019rangenet,
  title={RangeNet++: Fast and Accurate LiDAR Semantic Segmentation},
  author={Milioto, Andres and Vizzo, Ignacio and Behley, Jens and Stachniss, Cyrill},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2019}
}

@inproceedings{behley2019semantickitti,
  title={SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences},
  author={Behley, Jens and Garbade, Martin and Milioto, Andres and Quenzel, Jan and Behnke, Sven and Stachniss, Cyrill and Gall, Juergen},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## License

This documentation and configuration is provided for research and educational purposes. The original RangeNet++ implementation is available under the MIT License from the [Photogrammetry & Robotics Lab, University of Bonn](https://github.com/PRBonn/lidar-bonnetal).
