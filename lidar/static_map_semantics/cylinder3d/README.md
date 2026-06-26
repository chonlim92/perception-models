# Cylinder3D: Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation

**Paper:** [Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation](https://arxiv.org/abs/2011.10033) (CVPR 2021)

**Authors:** Xinge Zhu, Hui Zhou, Tai Wang, Fangzhou Hong, Yuexin Ma, Wei Li, Hongsheng Li, Dahua Lin

## Description

Cylinder3D is a framework for outdoor LiDAR point cloud semantic segmentation. It addresses the challenges of sparsity and varying density in driving-scene point clouds through:

1. **Cylindrical Partition** - Partitions the 3D space using cylindrical coordinates (rho, theta, z) instead of Cartesian voxels, providing a more balanced point distribution that better matches the scanning pattern of rotating LiDAR sensors.

2. **Asymmetrical 3D Convolution Networks** - Exploits the geometric properties of the cylindrical partition with asymmetrical convolution kernels, enabling more effective feature extraction with reduced computational cost.

3. **Point-wise Refinement Module** - Refines voxel-level predictions back to point-level through a lightweight point head that incorporates both voxel features and original point features.

## Architecture Overview

```
Input Point Cloud (N x 4)
        |
        v
Cylindrical Partition (480 x 360 x 32)
        |
        v
Asymmetrical 3D Sparse Convolution Encoder
  [32] -> [64] -> [128] -> [256] -> [256]
        |
        v
Asymmetrical 3D Sparse Convolution Decoder
  [256] -> [128] -> [64] -> [32]
        |
        v
Dimension-Decomposition Context Module (DDCM)
        |
        v
Point-wise Refinement Head
        |
        v
Per-point Semantic Labels (N x num_classes)
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.8.0
- CUDA >= 11.1
- NumPy >= 1.20.0
- numba >= 0.53.0
- PyYAML >= 5.4
- scipy >= 1.6.0
- spconv-cu114 >= 2.1 (or matching CUDA version)
- torch-scatter >= 2.0.8
- tqdm >= 4.60.0
- tensorboard >= 2.5.0

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd cylinder3d

# Create conda environment
conda create -n cylinder3d python=3.8 -y
conda activate cylinder3d

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install spconv
pip install spconv-cu113

# Install remaining dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Training

```bash
# Train on SemanticKITTI
python scripts/train.py --config configs/cylinder3d_semantickitti.yaml

# Train on nuScenes
python scripts/train.py --config configs/cylinder3d_nuscenes.yaml

# Resume training from checkpoint
python scripts/train.py --config configs/cylinder3d_semantickitti.yaml \
    --resume checkpoints/cylinder3d_semantickitti/epoch_20.pth

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 \
    scripts/train.py --config configs/cylinder3d_semantickitti.yaml
```

### Evaluation

```bash
# Evaluate on SemanticKITTI validation set
python scripts/evaluate.py --config configs/cylinder3d_semantickitti.yaml \
    --checkpoint checkpoints/cylinder3d_semantickitti/best_model.pth

# Evaluate with test-time augmentation
python scripts/evaluate.py --config configs/cylinder3d_semantickitti.yaml \
    --checkpoint checkpoints/cylinder3d_semantickitti/best_model.pth \
    --tta
```

### Inference

```bash
# Run inference on a single scan
python scripts/infer.py --config configs/cylinder3d_semantickitti.yaml \
    --checkpoint checkpoints/cylinder3d_semantickitti/best_model.pth \
    --input /path/to/scan.bin \
    --output /path/to/output/

# Run inference on a directory of scans
python scripts/infer.py --config configs/cylinder3d_semantickitti.yaml \
    --checkpoint checkpoints/cylinder3d_semantickitti/best_model.pth \
    --input /path/to/scans/ \
    --output /path/to/predictions/
```

## Project Structure

```
cylinder3d/
├── configs/
│   ├── cylinder3d_semantickitti.yaml   # SemanticKITTI training config
│   ├── cylinder3d_nuscenes.yaml        # nuScenes training config
│   └── label_mapping/                  # Label mapping files
│       ├── semantic-kitti.yaml
│       └── nuscenes.yaml
├── pytorch/
│   ├── models/
│   │   ├── cylinder3d.py              # Main model architecture
│   │   ├── backbone.py                # Asymmetrical 3D sparse convolution
│   │   ├── segmentation_head.py       # Point-wise refinement head
│   │   └── losses.py                  # CE + Lovasz-Softmax losses
│   ├── datasets/
│   │   ├── semantickitti.py           # SemanticKITTI dataloader
│   │   ├── nuscenes_dataset.py        # nuScenes dataloader
│   │   └── augmentation.py            # Data augmentation
│   └── utils/
│       ├── cylinder_utils.py          # Cylindrical partition utilities
│       ├── metrics.py                 # mIoU computation
│       └── visualization.py           # Point cloud visualization
├── scripts/
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   └── infer.py                       # Inference script
├── tests/
│   ├── test_model.py                  # Model unit tests
│   ├── test_dataloader.py             # Dataset unit tests
│   └── test_partition.py              # Cylindrical partition tests
├── docs/
│   └── architecture.md               # Detailed architecture documentation
├── README.md
└── requirements.txt
```

## Results

### SemanticKITTI (Validation Set - Sequence 08)

| Method | mIoU (%) | car | bicycle | motorcycle | truck | other-veh | person | bicyclist | motorcyclist | road | parking | sidewalk | other-gnd | building | fence | vegetation | trunk | terrain | pole | traffic-sign |
|--------|----------|-----|---------|------------|-------|-----------|--------|-----------|--------------|------|---------|----------|-----------|----------|-------|------------|-------|---------|------|--------------|
| Cylinder3D | **67.8** | 97.1 | 67.6 | 64.0 | 59.0 | 58.6 | 73.9 | 67.9 | 36.0 | 91.4 | 65.1 | 75.5 | 32.3 | 91.0 | 66.5 | 85.4 | 71.8 | 68.5 | 62.6 | 65.6 |

### SemanticKITTI (Test Set - Online Benchmark)

| Method | mIoU (%) |
|--------|----------|
| Cylinder3D | 68.9 |
| Cylinder3D (TTA) | 72.2 |

### nuScenes (Validation Set)

| Method | mIoU (%) | barrier | bicycle | bus | car | constr. | motorcycle | pedestrian | traffic cone | trailer | truck | driv. surf. | other flat | sidewalk | terrain | manmade | vegetation |
|--------|----------|---------|---------|-----|-----|---------|------------|------------|--------------|---------|-------|-------------|------------|----------|---------|---------|------------|
| Cylinder3D | **76.1** | 76.4 | 40.3 | 91.2 | 93.8 | 51.3 | 78.0 | 78.9 | 64.9 | 62.1 | 84.4 | 96.8 | 71.6 | 76.4 | 75.4 | 90.5 | 87.4 |

### nuScenes (Test Set - Online Benchmark)

| Method | mIoU (%) |
|--------|----------|
| Cylinder3D | 77.2 |

## Citation

```bibtex
@inproceedings{zhu2021cylindrical,
  title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={9939--9948},
  year={2021}
}
```

## License

This project is released under the Apache 2.0 License. See the LICENSE file for details.

## Acknowledgements

- [spconv](https://github.com/traveller59/spconv) for sparse convolution operations
- [SemanticKITTI](http://www.semantic-kitti.org/) for the LiDAR segmentation benchmark
- [nuScenes](https://www.nuscenes.org/) for the autonomous driving dataset
- [Lovasz-Softmax](https://github.com/bermanmaxim/LovsizSoftmax) for the Lovasz loss implementation
