# HDMapNet — Online HD Map Construction from Surround Cameras

## Overview

HDMapNet is the first end-to-end framework for constructing HD maps in bird's-eye-view (BEV) from surround-view camera images. It predicts semantic map elements (lane dividers, road boundaries, pedestrian crossings) as rasterized BEV segmentation maps, with optional instance embedding and direction prediction for vectorization.

**Paper:** "HDMapNet: An Online HD Map Construction and Evaluation Framework" (Li et al., ICRA 2022)

## Architecture

```
6 Surround Cameras (FRONT, FL, FR, BACK, BL, BR)
         ↓
┌─────────────────────────┐
│  Backbone (EfficientNet  │  Multi-scale image features
│  or ResNet-50 + FPN)     │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  View Transform          │  Option A: IPM (fast, flat-ground)
│                          │  Option B: LSS (learned depth, accurate)
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  BEV Encoder (U-Net)     │  Encode BEV features
└─────────────────────────┘
         ↓
┌──────────┬──────────┬──────────┐
│ Semantic │ Instance │Direction │  Three output heads
│  Head    │  Head    │  Head    │
└──────────┴──────────┴──────────┘
         ↓
┌─────────────────────────┐
│  Post-Processing         │  Skeletonize → trace → vectorize
└─────────────────────────┘
```

### View Transform Options

| Method | Accuracy | Speed | Description |
|--------|----------|-------|-------------|
| **IPM** | Lower | Fast | Homography-based flat-ground assumption |
| **LSS** | Higher | Slower | Predicted depth + voxel pooling (Lift-Splat-Shoot) |

## Quick Start

```bash
# 1. Install dependencies
pip install -r ../../requirements.txt

# 2. Download nuScenes + map expansion
bash scripts/download_data.sh --mini

# 3. Prepare BEV ground truth
python scripts/prepare_data.py --dataroot data/nuscenes --out data/hdmapnet_gt

# 4. Train (LSS variant, recommended)
python pytorch/train.py --config configs/hdmapnet_lss.yaml

# 5. Train (IPM variant, faster)
python pytorch/train.py --config configs/hdmapnet_ipm.yaml

# 6. Evaluate
python pytorch/evaluate.py --config configs/hdmapnet_lss.yaml --checkpoint outputs/best.pth

# 7. Inference + visualization
python pytorch/inference.py --config configs/hdmapnet_lss.yaml --checkpoint outputs/best.pth
```

## Results (nuScenes val)

| Variant | Lane Div IoU | Road Bound IoU | Ped Cross IoU | mIoU | Chamfer AP |
|---------|-------------|----------------|---------------|------|------------|
| IPM | 28.3 | 42.1 | 18.7 | 29.7 | 21.4 |
| LSS | 38.5 | 51.2 | 27.3 | 39.0 | 31.8 |

## Key Features

- Two view-transform options (IPM vs LSS) for speed/accuracy trade-off
- Instance embedding for differentiating individual lane lines
- Direction prediction for lane directionality
- Post-processing to convert rasterized output to vectorized polylines
- Multi-task training (semantic + instance + direction)

## Directory Structure

```
hdmapnet/
├── README.md
├── docs/                    # Research docs
├── configs/                 # Training configs (IPM + LSS)
├── pytorch/                 # Full PyTorch implementation
│   ├── model.py            # Main model
│   ├── backbone.py         # EfficientNet-B0 / ResNet-50
│   ├── view_transform.py   # IPM + LSS implementations
│   ├── bev_encoder.py      # U-Net BEV processing
│   ├── heads.py            # Semantic + Instance + Direction
│   ├── losses.py           # Focal + Discriminative + Direction
│   ├── postprocess.py      # Raster → vector (skeletonize)
│   ├── dataset.py          # nuScenes with BEV GT
│   ├── train.py            # Training script
│   ├── evaluate.py         # IoU + Chamfer evaluation
│   └── inference.py        # Single-sample inference
├── tensorflow/              # TF2/Keras implementation
├── scripts/                 # Download, prepare, visualize
└── tests/                   # Unit tests
```

## Citation

```bibtex
@inproceedings{li2022hdmapnet,
  title={HDMapNet: An Online HD Map Construction and Evaluation Framework},
  author={Li, Qi and Wang, Yue and Wang, Yilun and Zhao, Hang},
  booktitle={ICRA},
  year={2022}
}
```
