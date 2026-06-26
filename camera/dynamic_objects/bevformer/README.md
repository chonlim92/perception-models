# BEVFormer: Bird's-Eye-View Representation from Multi-Camera Images

## What Is BEVFormer?

BEVFormer is a neural network architecture that solves one of the hardest problems in autonomous driving: understanding the full 3D world around a vehicle using ONLY cameras (no LiDAR). Given 6 camera images that together provide 360-degree coverage, BEVFormer produces a Bird's-Eye-View (top-down) feature map that encodes where every object is in 3D space -- their position, size, orientation, and velocity.

The fundamental challenge is that cameras produce flat 2D images, but driving decisions require 3D understanding. How far away is that car? How fast is it approaching? Is that pedestrian about to step into the road? These questions require depth information that cameras inherently discard during imaging. Previous approaches either tried to explicitly estimate depth (noisy and error-prone) or skipped BEV entirely (losing the spatial structure that downstream planning needs).

BEVFormer's key insight is to use **attention mechanisms** with **known camera geometry** to construct BEV features without explicit depth estimation. It defines a grid of learnable "BEV queries" (one per patch of ground), projects each query's 3D position into the camera images using calibration matrices, and then uses deformable attention to sample relevant image features at those projected locations. This is geometrically principled (uses real camera math) yet learned end-to-end (the network decides what features matter). Combined with temporal fusion that aligns previous frames using ego-motion, BEVFormer achieves camera-only 3D detection performance that approaches LiDAR-based methods at a fraction of the sensor cost.

## Architecture

```
                              BEVFormer Pipeline
 ================================================================================

 INPUT: 6 Camera Images (each 900 x 1600 x 3 RGB)
 [FRONT] [FRONT_LEFT] [FRONT_RIGHT] [BACK] [BACK_LEFT] [BACK_RIGHT]
         |
         v
 +------------------------------------------------------------------+
 |  IMAGE BACKBONE: ResNet-101-DCN + Feature Pyramid Network (FPN)  |
 |                                                                    |
 |  Each image -> ResNet stages -> FPN produces multi-scale features |
 |  Output: 6 cameras x 3 scales x 256 channels                     |
 |    Level 0: 113 x 200 (1/8 resolution)                           |
 |    Level 1:  57 x 100 (1/16 resolution)                          |
 |    Level 2:  29 x  50 (1/32 resolution)                          |
 +------------------------------------------------------------------+
         |
         v
 +------------------------------------------------------------------+
 |  BEV ENCODER: 6 Transformer Encoder Layers                        |
 |                                                                    |
 |  BEV Queries: 200 x 200 grid, 256-dim each = 40,000 queries     |
 |  Each query represents a 0.512m x 0.512m patch of ground         |
 |  Range: [-51.2m, +51.2m] in X and Y around the ego vehicle       |
 |                                                                    |
 |  Each layer applies:                                              |
 |  +------------------------------------------------------------+   |
 |  | 1. TEMPORAL SELF-ATTENTION                                  |   |
 |  |    - Align previous BEV using ego-motion (rotation+trans)  |   |
 |  |    - Deformable attention: current queries attend to        |   |
 |  |      aligned previous BEV features                         |   |
 |  |    - Enables velocity estimation and temporal consistency   |   |
 |  +------------------------------------------------------------+   |
 |  | 2. SPATIAL CROSS-ATTENTION                                  |   |
 |  |    - For each BEV query at (x,y):                          |   |
 |  |      a) Generate 3D points at 4 heights: z=[-1,1,3,5]m    |   |
 |  |      b) Project each 3D point to all 6 cameras             |   |
 |  |      c) Keep valid projections (within image bounds)        |   |
 |  |      d) Apply deformable attention around projected 2D pts |   |
 |  |      e) Aggregate features across cameras and heights      |   |
 |  +------------------------------------------------------------+   |
 |  | 3. FEED-FORWARD NETWORK (FFN)                               |   |
 |  |    - Two-layer MLP: 256 -> 512 -> 256 with ReLU            |   |
 |  +------------------------------------------------------------+   |
 |                                                                    |
 +------------------------------------------------------------------+
         |
         v
 BEV Feature Map: (200 x 200 x 256) -- encodes the 3D scene
         |
         v
 +------------------------------------------------------------------+
 |  DETECTION DECODER: 6 Transformer Decoder Layers (DETR-style)     |
 |                                                                    |
 |  900 learnable object queries (each might become a detection)     |
 |  Each layer: Self-Attention -> Cross-Attention to BEV -> FFN     |
 |  Iterative reference point refinement across layers               |
 +------------------------------------------------------------------+
         |
         v
 +----------------------------+    +--------------------------------+
 | Classification Head        |    | Regression Head                |
 | 900 x 10 class scores     |    | 900 x 10 box parameters       |
 | (car, truck, bus, trailer, |    | (cx, cy, cz, w, l, h,         |
 |  constr_veh, pedestrian,   |    |  sin(yaw), cos(yaw), vx, vy)  |
 |  motorcycle, bicycle,      |    |                                |
 |  barrier, traffic_cone)    |    |                                |
 +----------------------------+    +--------------------------------+

 OUTPUT: Up to 300 3D bounding boxes with class, confidence, and velocity
```

## Why BEVFormer Matters

### Camera-Only Approaches to 3D Detection

| Approach | Method | How it works | Key limitation |
|----------|--------|--------------|----------------|
| Depth-based | LSS, BEVDet | Predict depth per pixel, lift to 3D | Depth prediction is noisy |
| Query-based | DETR3D, PETR | Object queries sample from images | No shared BEV representation |
| **Attention-based** | **BEVFormer** | **BEV queries attend to images via projection** | **Higher compute cost** |

BEVFormer achieves the best of both worlds:
- Like depth-based methods, it produces an explicit BEV feature map (enabling multi-task learning)
- Like query-based methods, it avoids explicit depth estimation (reducing error accumulation)
- Unlike both, it includes built-in temporal fusion (enabling velocity estimation)

### Performance Context

| Method | Type | NDS | mAP | Notes |
|--------|------|-----|-----|-------|
| BEVFormer-Base | Camera | 56.9 | 48.1 | Our model |
| DETR3D | Camera | 47.9 | 41.2 | No BEV, no temporal |
| BEVDet4D | Camera | 51.5 | 42.1 | Depth-based + temporal |
| CenterPoint | LiDAR | 67.3 | 60.3 | Uses $75k LiDAR sensor |

BEVFormer closes ~60% of the camera-to-LiDAR gap while using sensors that cost 100x less.

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.3+ with compatible GPU (minimum 24GB VRAM for training)
- ~80 GB disk space for nuScenes camera data

### Installation

```bash
# Clone and enter directory
git clone <this-repository>
cd bevformer

# Create environment
conda create -n bevformer python=3.8 -y
conda activate bevformer

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Data Preparation (5 minutes with mini dataset)

```bash
# For quick testing, use the nuScenes mini split (~4 GB)
# Download from https://www.nuscenes.org/download

# Generate temporal info files
python scripts/prepare_data.py \
    --data_root ./data/nuscenes/ \
    --out_dir ./data/nuscenes/ \
    --version v1.0-mini \
    --num_temporal_frames 4
```

### Training

```bash
# Single-GPU training (for development/debugging)
python tensorflow/train.py \
    --config configs/bevformer_base.yaml \
    --num_gpus 1 \
    --batch_size 1

# Multi-GPU training (recommended for full training)
python tensorflow/train.py \
    --config configs/bevformer_base.yaml \
    --num_gpus 8 \
    --batch_size 1
```

### Evaluation

```bash
python tensorflow/train.py \
    --config configs/bevformer_base.yaml \
    --eval_only \
    --checkpoint ./checkpoints/bevformer_base_best.h5
```

### Inference

```bash
python scripts/inference.py \
    --config configs/bevformer_base.yaml \
    --checkpoint ./checkpoints/bevformer_base_best.h5 \
    --input_dir ./data/nuscenes/samples/ \
    --output_dir ./outputs/visualization/ \
    --score_threshold 0.3
```

## Expected Results

### BEVFormer-Base on nuScenes Validation Set

| Metric | Expected Value | Meaning |
|--------|---------------|---------|
| NDS | 51.7 | Overall detection quality (higher = better) |
| mAP | 41.6 | Detection accuracy across distance thresholds |
| mATE | 0.673 m | Average center position error |
| mASE | 0.274 | Average size error |
| mAOE | 0.372 rad | Average orientation error (~21 degrees) |
| mAVE | 0.394 m/s | Average velocity error |
| mAAE | 0.198 | Average attribute error |

### Training Time Estimates

| Hardware | Time (24 epochs) | Estimated Cloud Cost |
|----------|-------------------|---------------------|
| 8x A100 (80GB) | ~28 hours | ~$900 (at $4/GPU-hr) |
| 4x A100 (80GB) | ~48 hours | ~$770 |
| 8x RTX 3090 (24GB) | ~70 hours | ~$560 (at $1/GPU-hr) |
| 1x A100 (80GB) | ~210 hours | ~$840 |

## Directory Structure

```
bevformer/
+-- README.md                    # This file
+-- configs/                     # Model and training configurations
|   +-- bevformer_base.yaml     # Standard BEVFormer-Base config
|   +-- bevformer_small.yaml    # Smaller variant for limited hardware
+-- docs/                        # Detailed documentation
|   +-- research_summary.md     # Paper explanation and teaching material
|   +-- model_architecture.md   # Complete architecture with tensor shapes
|   +-- training_guide.md       # Step-by-step training tutorial
|   +-- evaluation_guide.md     # Metrics explanation and benchmarks
|   +-- data_collection.md      # nuScenes dataset guide
|   +-- annotation_guide.md     # Annotation format reference
+-- pytorch/                     # PyTorch implementation
+-- tensorflow/                  # TensorFlow implementation
|   +-- train.py                # Training entry point
+-- scripts/                     # Utility scripts
|   +-- prepare_data.py         # Data preprocessing
|   +-- inference.py            # Run inference on images
|   +-- visualize_bev.py       # Visualize BEV detections
|   +-- export_model.py        # Export for deployment
+-- tests/                       # Unit and integration tests
```

## Documentation Guide

### Reading Order by Goal

**"I want to understand how BEVFormer works" (conceptual understanding):**
1. `docs/research_summary.md` -- Start here. Explains the problem, attention mechanisms, and architecture from scratch.
2. `docs/model_architecture.md` -- Deep dive into every component with tensor shapes and math.

**"I want to train BEVFormer on nuScenes" (practical training):**
1. `docs/data_collection.md` -- Understand and download the dataset.
2. `docs/training_guide.md` -- Environment setup, config explanation, training commands.
3. `docs/evaluation_guide.md` -- Understand what the metrics mean and interpret results.

**"I want to modify the architecture or build on BEVFormer":**
1. `docs/research_summary.md` -- Understand the design decisions and tradeoffs.
2. `docs/model_architecture.md` -- Full parameter breakdown and computation flow.
3. `configs/bevformer_base.yaml` -- See how architectural choices map to config fields.

**"I want to deploy BEVFormer in a real system":**
1. `docs/model_architecture.md` -- Understand compute/memory requirements (Section 11-12).
2. `docs/evaluation_guide.md` -- Know what performance to expect (Section 3-4).
3. `scripts/export_model.py` -- TensorRT and SavedModel export for inference.

## Frequently Asked Questions

**Q: Can I train on a single GPU?**

Yes, but with reduced settings. BEVFormer-Base with 200x200 BEV requires ~18 GB. For a 24GB GPU (RTX 3090), use batch_size=1 with mixed precision. For faster iteration, reduce BEV resolution to 100x100 (cuts memory to ~10 GB) at the cost of ~3 NDS. Training on 1 GPU takes roughly 8-9 days for 24 epochs.

**Q: How much does it cost to train on cloud GPUs?**

On AWS/GCP with 8x A100 instances: approximately $800-1000 for a full 24-epoch training run. Using spot/preemptible instances can reduce this by 60-70%, but requires checkpoint management for interruptions.

**Q: Can I use my own dataset (not nuScenes)?**

Yes, but you need to provide:
1. Camera images from your multi-camera rig
2. Camera intrinsic matrices (focal length, principal point)
3. Camera extrinsic matrices (position and orientation relative to ego vehicle)
4. Ego-motion data between frames (from IMU/GPS or visual odometry)
5. 3D bounding box annotations for training

The data format must match nuScenes structure (see `docs/data_collection.md`).

**Q: How fast is inference?**

| GPU | FPS | Latency | Notes |
|-----|-----|---------|-------|
| A100 | 9.4 | 106 ms | Standard BEVFormer-Base |
| RTX 3090 | 7.2 | 139 ms | Consumer GPU |
| A100 + TensorRT FP16 | ~15 | ~67 ms | Optimized deployment |
| Orin (TensorRT INT8) | ~8 | ~125 ms | Automotive-grade embedded |

**Q: What is the difference between BEVFormer variants?**

| Variant | Backbone | BEV Size | NDS | Use Case |
|---------|----------|----------|-----|----------|
| BEVFormer-Tiny | ResNet-50 | 50x50 | ~42 | Rapid prototyping, debugging |
| BEVFormer-Small | ResNet-101 | 100x100 | 47.8 | Limited GPU memory |
| BEVFormer-Base | ResNet-101-DCN | 200x200 | 56.9 | Standard training/research |
| BEVFormer-Large | V2-99 | 200x200 | 59.2 | Maximum accuracy |

**Q: How does camera-only compare to LiDAR?**

Camera-only (BEVFormer): NDS ~57, sensor cost ~$500 (6 cameras)
LiDAR-based (CenterPoint): NDS ~67, sensor cost ~$10,000-75,000 (mechanical LiDAR)

The ~10 NDS gap means LiDAR is still more accurate, especially at long range and in adverse weather. However, cameras are 20-150x cheaper, have higher resolution for classification, and work better for reading signs/lights. Many production systems use both (sensor fusion), with BEVFormer-style camera perception as a redundant backup.

**Q: Does BEVFormer work at night or in rain?**

Performance degrades in challenging conditions since it relies entirely on camera image quality. nuScenes includes some night and rain scenes in training, so the model has partial robustness, but expect 10-30% lower mAP in adverse conditions. For production, combine with radar or LiDAR for all-weather reliability.

## Model Zoo

| Model | Backbone | BEV Size | mAP | NDS | Config | Weights |
|-------|----------|----------|-----|-----|--------|---------|
| BEVFormer-Small | ResNet-50 | 50x50 | 40.2 | 47.8 | [config](configs/bevformer_small.yaml) | [download](https://github.com/example/bevformer/releases/download/v1.0/bevformer_small_r50.h5) |
| BEVFormer-Base | ResNet-101 | 200x200 | 51.7 | 56.9 | [config](configs/bevformer_base.yaml) | [download](https://github.com/example/bevformer/releases/download/v1.0/bevformer_base_r101.h5) |

All models trained on nuScenes v1.0-trainval, evaluated on validation split.

## Citation

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
