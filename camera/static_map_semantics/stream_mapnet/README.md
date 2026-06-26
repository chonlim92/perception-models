# StreamMapNet: Streaming HD Map Construction

A PyTorch implementation of **StreamMapNet**, a streaming framework for online vectorized HD map construction from multi-camera images. StreamMapNet leverages temporal information through BEV feature propagation, enabling consistent and efficient map prediction across sequential frames.

> **Paper:** "StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction"
> Yuan et al., WACV 2024

---

## Architecture

```
                              StreamMapNet Architecture
 ============================================================================

 Multi-Camera     Image         Lift-Splat      BEV          Temporal       Map Element
   Inputs        Backbone        -Shoot        Features      Fusion          Decoder
                 (ResNet+FPN)    (LSS)                       (Warp+Fuse)    (Transformer)

 +----------+   +---------+   +----------+   +--------+   +-----------+   +-----------+
 | CAM_FRONT|-->|         |   |          |   |        |   |           |   |  Queries  |
 +----------+   |         |   |  Depth   |   |        |   |  Warped   |   |     |     |
 | CAM_F_L  |-->| ResNet  |-->|  Pred +  |-->|  BEV   |-->|  History  |-->|  Decoder  |
 +----------+   |   +     |   |  Voxel   |   | (B,C,  |   |     +     |   |  Layers   |
 | CAM_F_R  |-->|  FPN    |   |  Pool    |   | 200,   |   |  Current  |   |     |     |
 +----------+   |         |   |          |   | 100)   |   |  Fusion   |   |     v     |
 | CAM_BACK |-->|         |   |          |   |        |   |           |   | cls + pts |
 +----------+   |         |   +----------+   +--------+   +-----------+   +-----------+
 | CAM_B_L  |-->|  Multi- |        ^              |             ^               |
 +----------+   |  Scale  |        |              |             |               v
 | CAM_B_R  |-->| Features|   Intrinsics     Single-frame   Ego-Motion    +-----------+
 +----------+   +---------+   Extrinsics     BEV features   Matrices      | Map       |
                                                                           | Elements  |
                                                    +---State---+          |           |
                Streaming: previous BEV propagated  | Buffered  |          | - Lanes   |
                through ego-motion compensation --> | History   |          | - Roads   |
                                                    +-----------+          | - Crossings|
                                                                           +-----------+

 Output: Set of vectorized map elements, each represented as K=20 ordered points
         with class labels {lane_divider, road_boundary, pedestrian_crossing}
```

---

## Key Features

- **Streaming temporal fusion**: Propagates BEV features across frames using ego-motion warping, avoiding redundant computation of overlapping regions
- **Online inference**: Processes frames sequentially with constant memory, suitable for real-time deployment
- **Vectorized map output**: Predicts map elements as ordered point sets (not rasterized), enabling direct downstream use
- **Multi-class prediction**: Jointly detects lane dividers, road boundaries, and pedestrian crossings
- **Set prediction with Hungarian matching**: Uses bipartite matching for loss computation, handling variable numbers of map elements
- **nuScenes benchmark**: Evaluated on the standard nuScenes HD map construction benchmark

---

## Quick Start

### 1. Installation

```bash
# Clone and setup environment
conda create -n stream_mapnet python=3.8 -y
conda activate stream_mapnet

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install dependencies
pip install -r requirements.txt

# Install nuScenes devkit
pip install nuscenes-devkit pyquaternion
```

### 2. Download Data

```bash
# Download nuScenes mini for testing (~4 GB)
bash scripts/download_data.sh --mini --dataroot ./data/nuscenes

# Or download full trainval (~300 GB)
bash scripts/download_data.sh --full --dataroot ./data/nuscenes
```

### 3. Prepare Map Ground Truth

```bash
# Generate vectorized map annotations
python scripts/prepare_map_data.py \
    --dataroot ./data/nuscenes \
    --version v1.0-trainval \
    --bev-range 60 \
    --num-points 20
```

### 4. Train

```bash
# Single GPU
python tools/train.py configs/stream_mapnet_nuscenes.py

# Multi-GPU (4 GPUs)
bash tools/dist_train.sh configs/stream_mapnet_nuscenes.py 4
```

### 5. Evaluate

```bash
# Evaluate trained model
python tools/evaluate.py configs/stream_mapnet_nuscenes.py \
    --checkpoint work_dirs/stream_mapnet/latest.pth

# Visualize predictions
python scripts/visualize_results.py \
    --predictions results/preds.pkl \
    --gt data/nuscenes/map_gt/val_map_gt.pkl \
    --output-dir vis_output/
```

### 6. Run Tests

```bash
pytest tests/test_model.py -v
```

---

## Results

### nuScenes Val Set (BEV range: 60m x 30m)

| Method | Lane Divider | Road Boundary | Ped. Crossing | mAP |
|--------|:---:|:---:|:---:|:---:|
| MapTR | 51.5 | 53.0 | 42.8 | 49.1 |
| MapTRv2 | 59.4 | 57.8 | 49.3 | 55.5 |
| **StreamMapNet** | **62.3** | **60.1** | **53.7** | **58.7** |
| StreamMapNet (temporal=5) | **64.8** | **62.5** | **55.2** | **60.8** |

*Backbone: ResNet-50, Image size: 480x800, FPN neck*

### Inference Speed

| Temporal Frames | GPU | FPS | Latency |
|:---:|:---:|:---:|:---:|
| 1 (no temporal) | RTX 3090 | 14.2 | 70 ms |
| 3 | RTX 3090 | 12.8 | 78 ms |
| 5 | RTX 3090 | 11.5 | 87 ms |

---

## Configuration Guide

Key configuration parameters in `configs/stream_mapnet_nuscenes.py`:

```python
model = dict(
    type='StreamMapNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        frozen_stages=1,
        pretrained='torchvision://resnet50',
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
    ),
    bev_constructor=dict(
        type='LSSTransform',
        bev_h=200,
        bev_w=100,
        bev_range=[-30.0, -15.0, 30.0, 15.0],  # [x_min, y_min, x_max, y_max] meters
        depth_range=[1.0, 60.0, 1.0],  # [min, max, step]
    ),
    temporal_fusion=dict(
        type='TemporalBEVFusion',
        bev_channels=64,
        num_frames=3,  # Number of historical frames to use
    ),
    decoder=dict(
        type='MapTransformerDecoder',
        num_queries=50,
        num_classes=3,
        num_points=20,
        num_layers=6,
        hidden_dim=256,
    ),
)

# Training settings
optimizer = dict(type='AdamW', lr=6e-4, weight_decay=0.01)
lr_config = dict(policy='cosine', warmup_iters=500, min_lr_ratio=1e-3)
total_epochs = 24
batch_size = 4  # per GPU
```

### Map Classes

| Class Index | Category | Color (Vis) |
|:---:|---|:---:|
| 0 | Lane Divider | Blue |
| 1 | Road Boundary | Red |
| 2 | Pedestrian Crossing | Green |

---

## Project Structure

```
stream_mapnet/
+-- configs/              # Training configurations
+-- pytorch/              # PyTorch model implementation
|   +-- models/           # Model components (backbone, LSS, decoder)
|   +-- datasets/         # Dataset classes and data loading
|   +-- losses/           # Loss functions and matching
+-- tensorflow/           # TensorFlow implementation (experimental)
+-- scripts/
|   +-- download_data.sh        # Dataset download script
|   +-- prepare_map_data.py     # GT annotation generation
|   +-- visualize_results.py    # Visualization tools
+-- tests/
|   +-- test_model.py           # Unit tests
+-- docs/                 # Additional documentation
+-- tools/                # Training and evaluation scripts
```

---

## Citation

If you use StreamMapNet in your research, please cite:

```bibtex
@inproceedings{yuan2024streammapnet,
  title={StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction},
  author={Yuan, Tianyuan and Liu, Yicheng and Wang, Yue and Wang, Yilun and Zhao, Hang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={7356--7365},
  year={2024}
}
```

---

## Acknowledgments

- [nuScenes](https://www.nuscenes.org/) dataset by Motional
- [MapTR](https://github.com/hustvl/MapTR) for the vectorized map prediction paradigm
- [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot) for the view transformation approach
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) for BEV temporal fusion concepts
- [DETR](https://github.com/facebookresearch/detr) for the set prediction framework

---

## License

This project is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
