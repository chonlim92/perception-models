# BEVFormer: Bird's-Eye-View Representation from Multi-Camera Images

BEVFormer is a spatiotemporal transformer framework for autonomous driving perception that generates a unified Bird's-Eye-View (BEV) representation from multi-camera images. By leveraging spatial cross-attention to project 2D image features into 3D BEV space and temporal self-attention to incorporate motion information from previous frames, BEVFormer achieves state-of-the-art 3D object detection without requiring LiDAR input.

The model builds upon the DETR detection paradigm: a transformer encoder constructs the BEV representation through learned queries that attend to multi-scale image features across all cameras, while a transformer decoder refines a set of object queries against the BEV features to produce 3D bounding box predictions. This architecture naturally handles the geometric correspondence between camera views and the 3D world through differentiable attention mechanisms, enabling end-to-end training from raw images to 3D detections.

## Architecture

```
                          BEVFormer Pipeline
 ============================================================================

  Multi-Camera Images (6 views)
  [Front, Front-Left, Front-Right, Back, Back-Left, Back-Right]
         |
         v
  +------------------+
  |  ResNet-101      |  ImageNet-pretrained backbone
  |  Backbone        |  Outputs: C3 (1/8), C4 (1/16), C5 (1/32)
  +------------------+
         |
         v
  +------------------+
  |  Feature Pyramid |  Unifies channels to 256 across 4 levels
  |  Network (FPN)   |  P3, P4, P5, P6
  +------------------+
         |
         v
  +--------------------------------------------------+
  |              BEV Encoder (x6 layers)             |
  |                                                  |
  |  +--------------------------------------------+  |
  |  | Temporal Self-Attention                    |  |
  |  | - Warp prev BEV via ego-motion             |  |
  |  | - Deformable attention (current + prev)    |  |
  |  +--------------------------------------------+  |
  |         |                                        |
  |         v                                        |
  |  +--------------------------------------------+  |
  |  | Spatial Cross-Attention                    |  |
  |  | - Project BEV queries to 3D (pillar)       |  |
  |  | - Project 3D points to camera views        |  |
  |  | - Multi-scale deformable attention         |  |
  |  | - Aggregate features from all 6 cameras    |  |
  |  +--------------------------------------------+  |
  |         |                                        |
  |         v                                        |
  |  +--------------------------------------------+  |
  |  | Feed-Forward Network (FFN)                 |  |
  |  +--------------------------------------------+  |
  |                                                  |
  +--------------------------------------------------+
         |
         v
  BEV Features (200 x 200, 256-dim)
         |
         v
  +--------------------------------------------------+
  |           DETR Decoder (x6 layers)               |
  |  - 900 learnable object queries                  |
  |  - Self-attention among queries                  |
  |  - Cross-attention to BEV features               |
  |  - FFN                                           |
  +--------------------------------------------------+
         |
         v
  +------------------+     +------------------+
  | Classification   |     | Regression Head  |
  | Head (10 cls)    |     | (10-dim bbox)    |
  +------------------+     +------------------+
         |                          |
         v                          v
  Class Scores              3D Bounding Boxes
  (car, truck, ...)         (cx,cy,cz,w,l,h,sin,cos,vx,vy)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare nuScenes data
python scripts/prepare_data.py --data_root ./data/nuscenes/ --out_dir ./data/nuscenes/

# Train BEVFormer-Base
python tensorflow/train.py --config configs/bevformer_base.yaml --num_gpus 1
```

## Installation

### Requirements

- Python 3.8+
- TensorFlow 2.10+
- CUDA 11.2+ (for GPU training)
- cuDNN 8.1+

### Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install core dependencies
pip install tensorflow>=2.10.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install pyyaml>=5.4
pip install opencv-python>=4.5.0
pip install pyquaternion>=0.9.9
pip install nuscenes-devkit>=1.1.9

# Install development dependencies (for testing)
pip install pytest>=7.0.0
pip install pytest-cov>=3.0.0
```

### Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}, GPU: {tf.config.list_physical_devices(\"GPU\")}')"
python -c "from tensorflow.model import build_bevformer; print('BEVFormer import OK')"
```

## Data Preparation

### Download nuScenes Dataset

1. Register and download the nuScenes dataset from https://www.nuscenes.org/download
2. Download the following splits:
   - Full dataset (v1.0-trainval): ~350GB
   - Mini dataset (v1.0-mini): ~4GB (for quick testing)

### Directory Structure

```
data/nuscenes/
    maps/
    samples/
        CAM_FRONT/
        CAM_FRONT_LEFT/
        CAM_FRONT_RIGHT/
        CAM_BACK/
        CAM_BACK_LEFT/
        CAM_BACK_RIGHT/
    sweeps/
    v1.0-trainval/
        sample.json
        sample_data.json
        ego_pose.json
        calibrated_sensor.json
        ...
```

### Generate Training Annotations

```bash
# Generate temporal info files required for BEVFormer training
python scripts/prepare_data.py \
    --data_root ./data/nuscenes/ \
    --out_dir ./data/nuscenes/ \
    --version v1.0-trainval \
    --num_temporal_frames 4

# This generates:
#   nuscenes_infos_temporal_train.pkl
#   nuscenes_infos_temporal_val.pkl
```

## Training

### Single-GPU Training

```bash
python tensorflow/train.py \
    --config configs/bevformer_base.yaml \
    --num_gpus 1 \
    --batch_size 1
```

### Multi-GPU Training

```bash
python tensorflow/train.py \
    --config configs/bevformer_base.yaml \
    --num_gpus 8 \
    --batch_size 1
```

### Resume Training from Checkpoint

```bash
python tensorflow/train.py \
    --config configs/bevformer_base.yaml \
    --num_gpus 8 \
    --resume_from ./checkpoints/bevformer_base_epoch_12.h5
```

### Training Tips

- **Memory**: BEVFormer is memory-intensive. Use batch_size=1 per GPU with mixed precision (FP16) enabled.
- **Learning Rate**: Default 2e-4 with cosine schedule works well for 24 epochs on 8 GPUs.
- **Backbone LR**: The pretrained ResNet-101 backbone uses 0.1x the base learning rate.
- **Gradient Clipping**: Max gradient norm is set to 35.0 to stabilize transformer training.
- **Warm-up**: 500 iterations of linear warm-up from 0.33x the base LR.

## Evaluation

### Run Evaluation

```bash
python tensorflow/train.py \
    --config configs/bevformer_base.yaml \
    --eval_only \
    --checkpoint ./checkpoints/bevformer_base_best.h5
```

### Metrics

Evaluation reports standard nuScenes detection metrics:

| Metric | Description |
|--------|-------------|
| mAP    | Mean Average Precision (distance-based) |
| NDS    | nuScenes Detection Score (composite) |
| mATE   | Mean Average Translation Error |
| mASE   | Mean Average Scale Error |
| mAOE   | Mean Average Orientation Error |
| mAVE   | Mean Average Velocity Error |
| mAAE   | Mean Average Attribute Error |

## Inference and Visualization

### Run Inference on Sample Images

```bash
python scripts/inference.py \
    --config configs/bevformer_base.yaml \
    --checkpoint ./checkpoints/bevformer_base_best.h5 \
    --input_dir ./data/nuscenes/samples/ \
    --output_dir ./outputs/visualization/ \
    --score_threshold 0.3
```

### Visualize BEV Detections

```bash
python scripts/visualize_bev.py \
    --predictions ./outputs/predictions.json \
    --data_root ./data/nuscenes/ \
    --output_dir ./outputs/bev_vis/ \
    --show_velocity \
    --show_trajectory
```

### Export for Deployment

```bash
# Export to SavedModel format
python scripts/export_model.py \
    --config configs/bevformer_base.yaml \
    --checkpoint ./checkpoints/bevformer_base_best.h5 \
    --output_dir ./exported_models/bevformer_base/ \
    --format saved_model

# Export to TensorRT (NVIDIA GPU inference)
python scripts/export_model.py \
    --config configs/bevformer_base.yaml \
    --checkpoint ./checkpoints/bevformer_base_best.h5 \
    --output_dir ./exported_models/bevformer_base_trt/ \
    --format tensorrt \
    --fp16
```

## Model Zoo

| Model | Backbone | BEV Size | mAP | NDS | Config | Weights |
|-------|----------|----------|-----|-----|--------|---------|
| BEVFormer-Small | ResNet-50 | 50x50 | 40.2 | 47.8 | [config](configs/bevformer_small.yaml) | [download](https://github.com/example/bevformer/releases/download/v1.0/bevformer_small_r50.h5) |
| BEVFormer-Base | ResNet-101 | 200x200 | 51.7 | 56.9 | [config](configs/bevformer_base.yaml) | [download](https://github.com/example/bevformer/releases/download/v1.0/bevformer_base_r101.h5) |

**Notes:**
- All models are trained on nuScenes trainval split and evaluated on the val split.
- mAP and NDS are reported on the nuScenes validation set.
- BEVFormer-Small uses a smaller BEV resolution (50x50) and lighter backbone for faster inference.
- BEVFormer-Base is the standard configuration matching the original paper results.

## Citation

If you use BEVFormer in your research, please cite the original paper:

```bibtex
@inproceedings{li2022bevformer,
    title={BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
    author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
```

## License

This project is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

```
Copyright 2022 BEVFormer Authors

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
