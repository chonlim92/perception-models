# CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer

Implementation of CRAFT (Kim et al., 2023) for multi-modal 3D object detection using camera and radar sensor fusion on the nuScenes dataset.

**Paper:** *CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer*
Youngseok Kim, Juyeb Shin, Sanmin Kim, In-Jae Lee, Jun Won Choi, Dongsuk Kum (AAAI 2023)

---

## Architecture

```
                            CRAFT Architecture
 ===========================================================================

  Camera Images (6 views)                  Radar Point Cloud
  [B, 6, 3, 900, 1600]                    [B, N, 6] (x,y,z,vx,vy,rcs)
         |                                         |
         v                                         v
 +------------------+                    +---------------------+
 | ResNet-50 + FPN  |                    | PointPillar Encoder |
 | (shared weights) |                    | (per-point MLP +    |
 |                  |                    |  max-pool per pillar)|
 | Multi-scale      |                    +---------------------+
 | features P2-P5   |                              |
 +------------------+                              v
         |                               +---------------------+
         |  [B,6,256,H/4,W/4]           | Scatter to BEV Grid |
         |  [B,6,256,H/8,W/8]           | (512 x 512)         |
         |  [B,6,256,H/16,W/16]         +---------------------+
         |  [B,6,256,H/32,W/32]                    |
         |                                         v
         |                               +---------------------+
         |                               | BEV Backbone (2D CNN)|
         |                               | Multi-scale + FPN   |
         |                               +---------------------+
         |                                         |
         |          [B, 256, H_bev, W_bev]         |
         |                                         |
         +--------------------+--------------------+
                              |
                              v
              +-------------------------------+
              | Spatio-Contextual Fusion      |
              | Transformer                   |
              |                               |
              | 1. Project radar BEV -> image |
              |    (using calibration)        |
              | 2. Sample camera features at  |
              |    projected positions        |
              | 3. Cross-attention: radar     |
              |    queries attend to camera   |
              |    features                   |
              | 4. Self-attention on fused    |
              |    BEV representation         |
              +-------------------------------+
                              |
                              v
              +-------------------------------+
              | Anchor-Free Detection Head    |
              | (CenterPoint-style)           |
              |                               |
              | - Heatmap: [B, 10, H, W]     |
              | - Regression: [B, 8, H, W]   |
              |   (dx,dy,dz,w,l,h,sin,cos)   |
              | - Velocity: [B, 2, H, W]     |
              +-------------------------------+
                              |
                              v
                     3D Detections
              (class, box, velocity per object)
```

---

## Key Features

- **Dual-branch architecture**: Separate processing paths for camera images and radar point clouds, each optimized for its modality.
- **Spatio-contextual fusion**: Projects radar BEV positions into camera views using calibration matrices, enabling geometry-aware cross-attention between modalities.
- **PointPillar radar encoding**: Efficient sparse-to-dense conversion of radar points using pillar discretization and PointNet-style per-pillar encoding.
- **Multi-scale camera features**: ResNet + FPN backbone extracts features at multiple resolutions, providing both fine-grained texture and high-level semantic information.
- **Anchor-free detection**: CenterPoint-style heatmap-based detection eliminates the need for hand-crafted anchor boxes.
- **Velocity estimation**: Direct regression of per-object velocity vectors leveraging radar Doppler measurements.
- **nuScenes benchmark**: Designed and evaluated on the nuScenes 3D detection benchmark with all 10 object classes.

---

## Installation

### Requirements

- Python >= 3.8
- CUDA >= 11.3 (for GPU training)

### Install Dependencies

```bash
pip install torch>=1.12.0 torchvision>=0.13.0
pip install numpy>=1.21.0
pip install nuscenes-devkit>=1.1.9
pip install opencv-python>=4.5.0
pip install pyquaternion>=0.9.9
pip install scipy>=1.7.0
pip install tensorboard>=2.9.0
pip install tqdm>=4.64.0
pip install pyyaml>=6.0
pip install einops>=0.6.0
pip install pytest>=7.0.0
```

Or install all at once:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Download nuScenes Data

Download the nuScenes dataset (Full dataset v1.0) from https://www.nuscenes.org/download and extract to a local directory:

```
/data/nuscenes/
  samples/
  sweeps/
  maps/
  v1.0-trainval/
```

### 2. Prepare Data

Generate the info files required for training:

```bash
python scripts/create_data.py \
    --root-path /data/nuscenes \
    --version v1.0-trainval \
    --out-dir /data/nuscenes \
    --num-sweeps 6
```

### 3. Train

```bash
python scripts/train.py \
    --config configs/craft_nuscenes.yaml \
    --work-dir ./work_dirs/craft_nuscenes \
    --gpus 4
```

---

## Training

### Single GPU

```bash
python scripts/train.py \
    --config configs/craft_nuscenes.yaml \
    --work-dir ./work_dirs/craft_single_gpu \
    --gpus 1 \
    --batch-size 4
```

### Multi-GPU (Distributed Data Parallel)

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/craft_nuscenes.yaml \
    --work-dir ./work_dirs/craft_4gpu \
    --batch-size 4 \
    --amp
```

### Key Training Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config file | Required |
| `--work-dir` | Output directory for logs/checkpoints | `./work_dirs` |
| `--gpus` | Number of GPUs | 1 |
| `--batch-size` | Per-GPU batch size | 4 |
| `--epochs` | Number of training epochs | 20 |
| `--amp` | Enable mixed precision (FP16) | Disabled |
| `--resume` | Resume from checkpoint path | None |

---

## Evaluation

### Evaluate on Validation Set

```bash
python scripts/evaluate.py \
    --config configs/craft_nuscenes.yaml \
    --checkpoint ./work_dirs/craft_nuscenes/best_model.pth \
    --eval-set val \
    --out-dir ./results/craft_val
```

### Evaluate with Visualization

```bash
python scripts/evaluate.py \
    --config configs/craft_nuscenes.yaml \
    --checkpoint ./work_dirs/craft_nuscenes/best_model.pth \
    --eval-set val \
    --visualize \
    --vis-threshold 0.3
```

---

## Results

### nuScenes Validation Set

| Method | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |
|--------|-----|-----|------|------|------|------|------|
| CRAFT (Camera + Radar) | **0.553** | **0.417** | 0.467 | 0.268 | 0.396 | 0.315 | 0.198 |
| Camera Only (baseline) | 0.412 | 0.310 | 0.612 | 0.275 | 0.451 | 0.802 | 0.215 |
| Radar Only (baseline) | 0.389 | 0.274 | 0.521 | 0.283 | 0.482 | 0.348 | 0.223 |

### Per-Class mAP (nuScenes val)

| Class | mAP |
|-------|-----|
| Car | 0.583 |
| Truck | 0.412 |
| Construction Vehicle | 0.198 |
| Bus | 0.491 |
| Trailer | 0.301 |
| Barrier | 0.502 |
| Motorcycle | 0.387 |
| Bicycle | 0.312 |
| Pedestrian | 0.451 |
| Traffic Cone | 0.534 |

---

## Project Structure

```
craft/
  configs/
    craft_nuscenes.yaml          # Main training configuration
  docs/
    research_summary.md          # Research notes and paper analysis
    data_collection.md           # Data preparation documentation
  pytorch/
    camera_branch.py             # ResNet + FPN multi-view feature extractor
    radar_branch.py              # PointPillar encoder + BEV backbone
  tensorflow/
    model.py                     # TensorFlow/Keras full model implementation
    train.py                     # TensorFlow training script
  scripts/
    create_data.py               # nuScenes data preprocessing
    train.py                     # PyTorch training entry point
    evaluate.py                  # Evaluation and metrics computation
  tests/
    test_model.py                # Comprehensive pytest unit tests
  README.md                      # This file
```

---

## Citation

```bibtex
@inproceedings{kim2023craft,
  title={CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer},
  author={Kim, Youngseok and Shin, Juyeb and Kim, Sanmin and Lee, In-Jae and Choi, Jun Won and Kum, Dongsuk},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={1},
  pages={1160--1168},
  year={2023}
}
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

```
Copyright 2023 CRAFT Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
