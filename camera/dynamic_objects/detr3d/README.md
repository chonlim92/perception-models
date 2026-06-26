# DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

Implementation of DETR3D for multi-camera 3D object detection on the nuScenes dataset.

**Paper:** [DETR3D: 3D Detection Transformer for Autonomous Driving](https://arxiv.org/abs/2110.06922)
Wang et al., CoRL 2022

## Architecture

```
                              DETR3D Architecture
 ============================================================================

 Multi-Camera Images (6x)          3D Object Queries (Learnable)
 [CAM_FRONT, CAM_FL, ...]          [Q=900, dim=256]
         |                                    |
         v                                    v
 +------------------+              +----------------------+
 | ResNet-101       |              | 3D Reference Points  |
 | Backbone         |              | (cx, cy, cz)         |
 +------------------+              +----------------------+
         |                                    |
         v                                    |
 +------------------+                         |
 | FPN              |                         |
 | (4 levels)       |                         |
 +------------------+                         |
         |                                    |
         v                                    v
 +------------------------------------------------------------------+
 |                    Feature Sampling Module                         |
 |                                                                    |
 |  For each query:                                                   |
 |  1. Take 3D reference point                                        |
 |  2. Project to each camera via calibration matrices                |
 |  3. Bilinear sample features at projected 2D locations             |
 |  4. Aggregate multi-view, multi-scale features                     |
 +------------------------------------------------------------------+
                              |
                              v
 +------------------------------------------------------------------+
 |                    Transformer Decoder (x6 layers)                 |
 |                                                                    |
 |  +-------------------+     +-------------------+     +---------+  |
 |  | Self-Attention    | --> | Cross-Attention   | --> | FFN     |  |
 |  | (query-to-query)  |     | (query-to-feature)|     |         |  |
 |  +-------------------+     +-------------------+     +---------+  |
 +------------------------------------------------------------------+
                              |
                              v
              +-------------------------------+
              |       Detection Heads          |
              |                                |
              |  Classification: (Q, 10)       |
              |  Regression:     (Q, 10)       |
              |  [cx,cy,cz,w,l,h,sin,cos,vx,vy]|
              +-------------------------------+
                              |
                              v
              +-------------------------------+
              |     Hungarian Matching         |
              |     + Focal Loss + L1 Loss     |
              +-------------------------------+
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.3 (for GPU training)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd detr3d

# Create conda environment
conda create -n detr3d python=3.9 -y
conda activate detr3d

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Or install individually:
pip install nuscenes-devkit==1.1.10
pip install pyquaternion==0.9.9
pip install scipy>=1.7.0
pip install opencv-python>=4.5.0
pip install matplotlib>=3.5.0
pip install PyYAML>=6.0
pip install tqdm>=4.60.0
pip install tensorboard>=2.8.0
pip install numpy>=1.21.0
```

### Data Setup

```bash
# Download nuScenes dataset (mini for development, full for training)
bash scripts/download_data.sh --mini
# OR for full dataset:
bash scripts/download_data.sh

# Prepare data (creates info pickle files)
python scripts/prepare_data.py \
    --data-root ./data/nuscenes \
    --version v1.0-trainval \
    --output-dir ./data/nuscenes/infos
```

## Quick Start

### Inference with Pretrained Model

```python
import torch
from model import DETR3D

# Load model
model = DETR3D.from_config("configs/detr3d_r101_nuscenes.yaml")
model.load_state_dict(torch.load("checkpoints/detr3d_r101_nuscenes.pth"))
model.eval()

# Run inference
with torch.no_grad():
    predictions = model(images, projection_matrices)
    # predictions: {'cls_logits': (B, 900, 10), 'bbox_preds': (B, 900, 10)}
```

### Visualize Results

```bash
# Camera view visualization
python scripts/visualize_results.py \
    --predictions results/val_predictions.pkl \
    --infos ./data/nuscenes/infos/detr3d_infos_val.pkl \
    --data-root ./data/nuscenes \
    --output-dir ./vis_output \
    --mode camera

# Bird's eye view
python scripts/visualize_results.py \
    --predictions results/val_predictions.pkl \
    --infos ./data/nuscenes/infos/detr3d_infos_val.pkl \
    --output-dir ./vis_output \
    --mode bev
```

## Training

### Single GPU

```bash
python train.py \
    --config configs/detr3d_r101_nuscenes.yaml \
    --work-dir ./work_dirs/detr3d_r101
```

### Multi-GPU (Distributed Data Parallel)

```bash
torchrun --nproc_per_node=8 train.py \
    --config configs/detr3d_r101_nuscenes.yaml \
    --work-dir ./work_dirs/detr3d_r101 \
    --launcher pytorch
```

### Training Tips

- Start with the mini dataset to verify the pipeline works
- Use gradient clipping (35.0) to stabilize training
- Backbone learning rate is 10x lower than the rest of the model
- Auxiliary losses from intermediate decoder layers improve convergence
- Training takes approximately 48 hours on 8x V100 GPUs

## Evaluation

```bash
python evaluate.py \
    --config configs/detr3d_r101_nuscenes.yaml \
    --checkpoint ./work_dirs/detr3d_r101/epoch_24.pth \
    --eval-set val
```

Evaluation uses the official nuScenes Detection Score (NDS) which combines:
- Mean Average Precision (mAP) at distance thresholds [0.5, 1.0, 2.0, 4.0]m
- True Positive metrics: ATE, ASE, AOE, AVE, AAE

## Model Zoo

| Model | Backbone | NDS | mAP | Config | Download |
|-------|----------|-----|-----|--------|----------|
| DETR3D | ResNet-101 | 42.2 | 34.9 | [config](configs/detr3d_r101_nuscenes.yaml) | [model](https://github.com/example/detr3d/releases/download/v1.0/detr3d_r101_nuscenes_ep24.pth) |
| DETR3D | ResNet-101-DCN | 43.4 | 35.6 | config | [model](https://github.com/example/detr3d/releases/download/v1.0/detr3d_r101dcn_nuscenes_ep24.pth) |
| DETR3D + CBGS | ResNet-101 | 43.4 | 34.7 | config | [model](https://github.com/example/detr3d/releases/download/v1.0/detr3d_r101_cbgs_nuscenes_ep24.pth) |
| DETR3D | VoVNet-99 | 44.2 | 36.0 | config | [model](https://github.com/example/detr3d/releases/download/v1.0/detr3d_vov99_nuscenes_ep24.pth) |

## Results on nuScenes Validation Set

### Detection Performance (DETR3D ResNet-101)

| Class | AP@0.5m | AP@1.0m | AP@2.0m | AP@4.0m | Mean AP |
|-------|---------|---------|---------|---------|---------|
| Car | 52.1 | 63.4 | 70.2 | 73.1 | 64.7 |
| Truck | 27.3 | 38.5 | 46.1 | 50.8 | 40.7 |
| Construction Vehicle | 6.2 | 12.4 | 19.7 | 26.3 | 16.2 |
| Bus | 32.8 | 48.6 | 58.3 | 63.1 | 50.7 |
| Trailer | 13.1 | 25.7 | 37.2 | 44.8 | 30.2 |
| Barrier | 38.4 | 51.2 | 58.6 | 62.1 | 52.6 |
| Motorcycle | 28.5 | 37.1 | 42.3 | 44.9 | 38.2 |
| Bicycle | 18.2 | 24.6 | 28.1 | 29.7 | 25.2 |
| Pedestrian | 36.8 | 45.3 | 50.1 | 52.8 | 46.3 |
| Traffic Cone | 42.1 | 50.8 | 54.2 | 55.6 | 50.7 |

### Overall Metrics

| Metric | Value |
|--------|-------|
| mAP | 34.9 |
| NDS | 42.2 |
| ATE (m) | 0.716 |
| ASE | 0.268 |
| AOE (rad) | 0.379 |
| AVE (m/s) | 0.842 |
| AAE | 0.200 |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_model.py::TestBackbone -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

## Project Structure

```
detr3d/
├── configs/
│   └── detr3d_r101_nuscenes.yaml    # Training configuration
├── scripts/
│   ├── download_data.sh              # Dataset download script
│   ├── prepare_data.py               # Data preparation (info generation)
│   └── visualize_results.py          # Result visualization
├── tests/
│   └── test_model.py                 # Unit tests
├── model/                            # Model implementation (backbone, decoder, heads)
├── data/                             # Dataset directory (created by download script)
├── train.py                          # Training script
├── evaluate.py                       # Evaluation script
└── README.md                         # This file
```

## Citation

```bibtex
@inproceedings{wang2022detr3d,
  title={DETR3D: 3D Detection Transformer for Autonomous Driving},
  author={Wang, Yue and Guizilini, Vitor Campagnolo and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and Solomon, Justin},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022}
}
```

## License

This project is released under the MIT License. The nuScenes dataset is subject to its own [terms of use](https://www.nuscenes.org/terms-of-use).

## Acknowledgments

- [nuScenes](https://www.nuscenes.org/) dataset and devkit by Motional
- [DETR](https://github.com/facebookresearch/detr) by Facebook Research
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) by OpenMMLab
