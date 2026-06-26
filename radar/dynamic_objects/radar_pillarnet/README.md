# RadarPillarNet: 3D Object Detection from Radar Point Clouds

## Overview

RadarPillarNet is a deep learning model for 3D object detection using automotive radar
point clouds. It adapts the PointPillars architecture for the unique characteristics of
radar data: extreme sparsity (~100-300 points per sweep), direct Doppler velocity
measurements, and high noise from multipath reflections.

The model achieves competitive performance on the nuScenes benchmark for radar-only
3D detection by combining multi-sweep accumulation, radar-specific feature engineering,
and velocity regression.

## Architecture

```
Radar Point Cloud (N x 9)
    |
    v
[Pillar Encoding] --> [PointNet] --> [Scatter to BEV]
    |
    v
[2D CNN Backbone (3 blocks)] --> [FPN Neck]
    |
    v
[Detection Head: cls + box + dir + velocity]
    |
    v
3D Bounding Boxes + Velocity
```

**Key design choices:**
- 9 input features per point: x, y, z, RCS, vx_comp, vy_comp, dt, xc, yc
- 6-sweep temporal accumulation with ego-motion compensation
- 512x512 BEV grid at 0.2m resolution (102.4m x 102.4m coverage)
- Anchor-based detection with class-specific anchor sizes
- Dedicated velocity regression head for direct speed estimation
- ~4.6M parameters, ~67 FPS on NVIDIA V100

## Results

### nuScenes Validation Set

| Model | Sweeps | mAP | NDS | mATE | mASE | mAOE | mAVE |
|-------|--------|-----|-----|------|------|------|------|
| RadarPillarNet | 6 | 23.4 | 35.8 | 0.72 | 0.28 | 0.58 | 0.89 |
| RadarPillarNet | 10 | 25.1 | 37.2 | 0.69 | 0.27 | 0.55 | 0.82 |

### Per-Class AP (6 sweeps)

| car | truck | bus | ped | moto | bicycle |
|-----|-------|-----|-----|------|---------|
| 42.1 | 22.8 | 28.5 | 20.5 | 18.2 | 9.8 |

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.3
- nuScenes devkit

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd radar_pillarnet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Install nuScenes devkit
pip install nuscenes-devkit
```

### Dependencies

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
scipy>=1.7.0
numba>=0.54.0
nuscenes-devkit>=1.1.9
tensorboard>=2.8.0
pyyaml>=5.4
tqdm>=4.62
```

## Quick Start

### Data Preparation

```bash
# Download nuScenes dataset to /data/nuscenes/
# Then run preprocessing:

# Step 1: Create preprocessed radar data
python scripts/create_radar_data.py \
    --dataroot /data/nuscenes \
    --output /data/nuscenes/radar_preprocessed \
    --n_sweeps 6

# Step 2: Create GT database for augmentation
python scripts/create_gt_database.py \
    --dataroot /data/nuscenes \
    --radar_data /data/nuscenes/radar_preprocessed \
    --output /data/nuscenes/radar_gt_database

# Step 3: Generate info files
python scripts/create_infos.py \
    --dataroot /data/nuscenes \
    --output /data/nuscenes/radar_infos
```

### Training

```bash
# Single GPU training
python scripts/train.py \
    --config configs/radar_pillarnet_nuscenes.yaml \
    --dataroot /data/nuscenes

# Multi-GPU training (4 GPUs)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/radar_pillarnet_nuscenes.yaml \
    --dataroot /data/nuscenes \
    --launcher pytorch
```

### Evaluation

```bash
# Evaluate on validation set
python scripts/evaluate.py \
    --config configs/radar_pillarnet_nuscenes.yaml \
    --checkpoint checkpoints/radar_pillarnet_epoch80.pth \
    --dataroot /data/nuscenes \
    --eval_set val
```

### Visualization

```bash
# Visualize detections in BEV
python scripts/visualize.py \
    --config configs/radar_pillarnet_nuscenes.yaml \
    --checkpoint checkpoints/radar_pillarnet_epoch80.pth \
    --dataroot /data/nuscenes \
    --num_samples 10 \
    --output_dir visualizations/
```

## Project Structure

```
radar_pillarnet/
├── configs/                    # Training configuration files
├── docs/                       # Documentation
│   ├── research_summary.md     # Background research and motivations
│   ├── data_collection.md      # Data format and collection details
│   ├── annotation_guide.md     # Annotation standards
│   ├── model_architecture.md   # Detailed architecture description
│   ├── training_guide.md       # Training procedures and tips
│   └── evaluation_guide.md     # Metrics and evaluation
├── pytorch/                    # PyTorch implementation
├── tensorflow/                 # TensorFlow implementation
├── scripts/                    # Training, evaluation, preprocessing scripts
├── tests/                      # Unit tests
└── README.md                   # This file
```

## Configuration

Key configuration parameters in `configs/radar_pillarnet_nuscenes.yaml`:

```yaml
model:
  voxel_size: [0.2, 0.2, 8.0]
  point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  max_points_per_pillar: 32
  max_pillars: 16000
  n_sweeps: 6

training:
  batch_size: 4
  num_epochs: 80
  learning_rate: 2e-4
  optimizer: AdamW
  scheduler: OneCycleLR
```

## Citation

If you use RadarPillarNet in your research, please cite:

```bibtex
@inproceedings{radarpillarnet2024,
  title={RadarPillarNet: Pillar-based 3D Object Detection from Radar Point Clouds},
  author={Perception Team},
  year={2024}
}
```

Related works:

```bibtex
@inproceedings{lang2019pointpillars,
  title={PointPillars: Fast Encoders for Object Detection from Point Clouds},
  author={Lang, Alex H. and Vora, Sourabh and Caesar, Holger and Zhou, Lubing and Yang, Jiong and Beijbom, Oscar},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{caesar2020nuscenes,
  title={nuScenes: A Multimodal Dataset for Autonomous Driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H. and Vora, Sourabh and Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={CVPR},
  year={2020}
}
```

## License

This project is released under the Apache 2.0 License. See LICENSE for details.
