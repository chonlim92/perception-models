# PETR / PETRv2 / StreamPETR - TensorFlow Implementation

Multi-view 3D object detection using Position Embedding Transformation with optional temporal modeling for autonomous driving perception.

## Overview

This repository provides a TensorFlow 2 implementation of the PETR family of models for camera-based 3D object detection on the nuScenes dataset:

- **PETR** (ECCV 2022): Introduces 3D position-aware features by transforming camera frustum coordinates into 3D space via MLP encoding, enabling a standard transformer decoder to perform multi-view 3D detection without explicit depth estimation.
- **PETRv2** (ICCV 2023): Extends PETR with temporal alignment of 3D position embeddings and supports multi-task learning (detection + segmentation).
- **StreamPETR** (ICCV 2023): Replaces frame-level temporal fusion with object-centric temporal modeling. Propagates queries across frames using ego-motion compensation and motion-aware layer normalization, achieving long-term temporal reasoning with constant memory.

## Key Innovations

| Feature | Description |
|---------|-------------|
| 3D Position Embedding | Camera frustum points projected to world coordinates and encoded via MLP, providing geometry-aware features without LiDAR |
| Object-Centric Temporal | Queries carry object state across frames (StreamPETR), unlike BEV-based temporal fusion that operates on the feature map |
| Motion-Aware LayerNorm | Ego-motion embedding modulates layer normalization, helping the model adapt to dynamic viewpoint changes |
| Hungarian Matching | Bipartite matching between predictions and ground truth for set-based loss computation |

## Installation

### Requirements

```
tensorflow>=2.10.0
tensorflow-addons>=0.19.0
numpy>=1.22.0
scipy>=1.8.0
pyyaml>=6.0
opencv-python>=4.6.0
nuscenes-devkit>=1.1.9
Pillow>=9.0.0
tqdm>=4.64.0
matplotlib>=3.5.0
```

### Setup

```bash
# Clone and enter directory
cd perception-models/camera/dynamic_objects/petr

# Install dependencies
pip install -r requirements.txt

# (Optional) Install in development mode
pip install -e .
```

## Quick Start

### 1. Download Data

```bash
# Download nuScenes mini split (for testing)
bash scripts/download_data.sh --split mini --output_dir ./data/nuscenes

# Download pretrained backbone
bash scripts/download_data.sh --backbone --backbone_dir ./data/pretrained

# For full training, download trainval split (~400GB)
bash scripts/download_data.sh --split trainval --output_dir ./data/nuscenes
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py \
    --data_root ./data/nuscenes \
    --version v1.0-mini \
    --output_dir ./data/infos
```

### 3. Train

```bash
# PETR base
python tensorflow/train.py --config configs/petr_r50.yaml --output_dir ./output/petr

# StreamPETR (temporal)
python tensorflow/train.py --config configs/stream_petr_r50.yaml --output_dir ./output/stream_petr

# Multi-GPU
python tensorflow/train.py --config configs/petr_r50.yaml --gpus 0,1,2,3
```

### 4. Evaluate

```bash
python tensorflow/evaluate.py \
    --config configs/petr_r50.yaml \
    --checkpoint ./output/petr/checkpoints/ckpt-24 \
    --data_info ./data/infos/petr_infos_val_v1_0-trainval.pkl \
    --data_root ./data/nuscenes \
    --output ./eval_results.json
```

### 5. Inference

```bash
python tensorflow/inference.py \
    --config configs/petr_r50.yaml \
    --model_path ./output/petr/saved_model \
    --input ./data/infos/petr_infos_val_v1_0-mini.pkl \
    --output ./inference_results.pkl \
    --score_threshold 0.3
```

### 6. Visualize

```bash
python scripts/visualize_results.py \
    --results ./inference_results.pkl \
    --data_info ./data/infos/petr_infos_val_v1_0-mini.pkl \
    --data_root ./data/nuscenes \
    --output_dir ./visualizations \
    --show_bev \
    --create_video
```

## Model Variants and Configuration

### PETR Base (petr_r50.yaml)

```yaml
model:
  num_classes: 10
  embed_dims: 256
  num_queries: 900
  num_decoder_layers: 6
  num_heads: 8
  num_depth_bins: 64
  depth_range: [1.0, 61.0]
  temporal: false
training:
  batch_size: 2
  num_epochs: 24
  learning_rate: 2.0e-4
  weight_decay: 0.01
  mixed_precision: true
```

### StreamPETR (stream_petr_r50.yaml)

```yaml
model:
  num_classes: 10
  embed_dims: 256
  num_queries: 900
  num_decoder_layers: 6
  num_heads: 8
  num_depth_bins: 64
  depth_range: [1.0, 61.0]
  temporal: true
  num_propagated_queries: 256
training:
  batch_size: 2
  num_epochs: 24
  learning_rate: 2.0e-4
```

## Performance Benchmarks

Results reported in the original papers on nuScenes val set:

| Model | Backbone | mAP | NDS | FPS |
|-------|----------|-----|-----|-----|
| PETR | ResNet50 | 0.313 | 0.381 | 15.3 |
| PETR | ResNet101 | 0.357 | 0.421 | 11.2 |
| PETRv2 | ResNet50 | 0.349 | 0.422 | 14.8 |
| PETRv2 | V2-99 | 0.421 | 0.524 | 9.7 |
| StreamPETR | ResNet50 | 0.384 | 0.450 | 31.7 |
| StreamPETR | V2-99 | 0.450 | 0.550 | 17.1 |

Key observations:
- StreamPETR achieves higher FPS than frame-level methods by avoiding feature-level temporal fusion
- Temporal modeling (StreamPETR) consistently improves both mAP and NDS over single-frame PETR
- The object-centric approach scales better with sequence length than BEV-based methods

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_model.py::TestPositionEmbedding3D -v

# Run with coverage
pytest tests/ --cov=tensorflow --cov-report=term-missing
```

## File Structure

```
petr/
├── tensorflow/
│   ├── model.py          # PETR/StreamPETR model (backbone, FPN, 3D PE, decoder, head)
│   ├── train.py          # Training script (multi-GPU, mixed precision, cosine LR)
│   ├── evaluate.py       # Evaluation (nuScenes metrics: mAP, NDS, TP errors)
│   └── inference.py      # Inference (SavedModel/checkpoint, NMS, batch support)
├── scripts/
│   ├── download_data.sh  # Download nuScenes + pretrained weights
│   ├── prepare_data.py   # Generate training info files from nuScenes DB
│   └── visualize_results.py  # Multi-view + BEV + temporal visualization
├── tests/
│   └── test_model.py     # Comprehensive model tests (pytest)
├── configs/              # YAML configuration files (create as needed)
└── README.md
```

## Citation

```bibtex
@inproceedings{liu2022petr,
  title={PETR: Position Embedding Transformation for Multi-View 3D Object Detection},
  author={Liu, Yingfei and Wang, Tiancai and Zhang, Xiangyu and Sun, Jian},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}

@inproceedings{liu2023petrv2,
  title={PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images},
  author={Liu, Yingfei and Yan, Junjie and Jia, Fan and Li, Shuailin and Gao, Aqi and Wang, Tiancai and Zhang, Xiangyu},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2023}
}

@inproceedings{wang2023streampetr,
  title={Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection},
  author={Wang, Shihao and Liu, Yingfei and Wang, Tiancai and Li, Ying and Zhang, Xiangyu},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## License

This implementation is for research purposes. The nuScenes dataset is subject to its own [license terms](https://www.nuscenes.org/terms-of-use).
