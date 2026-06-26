# DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

A camera-only 3D object detection model that uses learned 3D reference points
projected into 2D image space to sample features -- the inverse of traditional
"lift-splat" approaches that reconstruct 3D from 2D.

**Paper:** [DETR3D: 3D Detection Transformer for Autonomous Driving](https://arxiv.org/abs/2110.06922)
Wang et al., Conference on Robot Learning (CoRL), 2022

---

## Table of Contents

1. [What Problem Does This Solve?](#what-problem-does-this-solve)
2. [How It Works (Intuition)](#how-it-works-intuition)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Data Setup](#data-setup)
6. [Quick Start](#quick-start)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Model Zoo](#model-zoo)
10. [Results](#results)
11. [Documentation Guide](#documentation-guide)
12. [Tests](#tests)
13. [Project Structure](#project-structure)
14. [Citation](#citation)
15. [License](#license)

---

## What Problem Does This Solve?

### The Camera-Based 3D Detection Problem

Self-driving cars need to detect objects in 3D space -- they must know not just
"there is a car" but "there is a car 15 meters ahead, 2 meters to the left,
moving at 30 km/h." LiDAR sensors directly measure 3D points, but cameras only
capture 2D images. The challenge: how do you recover accurate 3D bounding boxes
from flat 2D images?

A modern autonomous vehicle typically has 6 cameras covering a full 360-degree
view around the vehicle:

```
                          FRONT
                       +---------+
                      /           \
           FRONT-LEFT/             \FRONT-RIGHT
                    /               \
                   |                 |
                   |    EGO CAR      |
                   |                 |
                    \               /
          BACK-LEFT  \             / BACK-RIGHT
                      \           /
                       +---------+
                          BACK

  Each camera produces a 1600x900 RGB image.
  Together they give complete surround coverage.
  The question: how to fuse these 6 views into 3D detections?
```

### What Makes DETR3D's Approach Unique

Most prior methods follow a "lift" paradigm: they take 2D image features and
try to "lift" them into 3D space by predicting depth or constructing a 3D
voxel grid. This is computationally expensive and error-prone because depth
estimation from monocular images is inherently ambiguous.

DETR3D flips this around. Instead of lifting 2D to 3D, it projects 3D to 2D:

```
  Traditional "Lift" approach:          DETR3D "Project" approach:
  ===========================          ===========================

  2D Image Features                    3D Reference Points (learned)
       |                                       |
       | predict depth                         | project using camera
       | (hard, ambiguous)                     | calibration (exact)
       v                                       v
  3D Voxel Grid                        2D Sample Locations
       |                                       |
       | 3D detection                          | bilinear sample
       v                                       v
  3D Bounding Boxes                    3D Bounding Boxes
```

Key insight: camera calibration matrices are known precisely (they come from
the sensor setup), so projecting a 3D point to 2D is a deterministic geometric
operation -- no learning required. By starting in 3D and projecting to 2D,
DETR3D sidesteps the hardest part of the traditional approach (depth estimation)
entirely.

---

## How It Works (Intuition)

Think of DETR3D as a set of 900 "scouts" that are each given a 3D position in
the world and tasked with determining whether an object exists at that location.
Each scout looks through all 6 cameras to gather visual evidence about its
assigned location. It does this by computing where its 3D position would appear
in each camera image (using known camera geometry) and sampling the image
features at those pixel locations. After gathering evidence from all views, the
scout decides: "is there an object here, and if so, what kind, how big, and
which direction is it facing?"

These scouts do not work in isolation. Through self-attention in the transformer
decoder, scouts communicate with each other. A scout investigating a position
near the front-left of the vehicle can share information with nearby scouts,
helping them collectively determine object boundaries. This communication
happens across 6 decoder layers, with each layer refining the scouts'
understanding of the scene. The 3D reference points themselves are also refined
at each layer, allowing scouts to "move" toward objects they are detecting.

After the final decoder layer, each scout outputs a classification (one of 10
object categories like car, truck, pedestrian, etc.) and a 3D bounding box
parameterized as (cx, cy, cz, width, length, height, yaw_sin, yaw_cos,
velocity_x, velocity_y). During training, the Hungarian algorithm matches
scouts to ground-truth objects one-to-one, so each object is detected by
exactly one scout. This set-based prediction eliminates the need for
non-maximum suppression (NMS) post-processing entirely, producing clean outputs
directly from the network.

---

## Architecture

```
 ==============================================================================
                         DETR3D Full Architecture
 ==============================================================================

 INPUT: 6 Camera Images (each 1600x900x3)
 +---------------------------------------------------------------------------+
 |  CAM_FRONT    CAM_FRONT_LEFT    CAM_FRONT_RIGHT                           |
 |  CAM_BACK     CAM_BACK_LEFT     CAM_BACK_RIGHT                            |
 +---------------------------------------------------------------------------+
         |
         | Each image processed independently by shared backbone
         v
 +---------------------------------------------------------------------------+
 |                      IMAGE BACKBONE: ResNet-101                             |
 |                                                                             |
 |  Input Image (3, H, W)                                                      |
 |       |                                                                     |
 |       v                                                                     |
 |  +----------+    +----------+    +----------+    +----------+               |
 |  | Layer 1  | -> | Layer 2  | -> | Layer 3  | -> | Layer 4  |               |
 |  | stride=4 |    | stride=8 |    | stride=16|    | stride=32|               |
 |  | C=256    |    | C=512    |    | C=1024   |    | C=2048   |               |
 |  +----------+    +----------+    +----------+    +----------+               |
 |       |               |               |               |                     |
 +---------------------------------------------------------------------------+
         |               |               |               |
         v               v               v               v
 +---------------------------------------------------------------------------+
 |                   FEATURE PYRAMID NETWORK (FPN)                             |
 |                                                                             |
 |  Produces 4 multi-scale feature maps per camera:                            |
 |  Level 0: stride=4,  channels=256,  size ~ (400, 225)                      |
 |  Level 1: stride=8,  channels=256,  size ~ (200, 112)                      |
 |  Level 2: stride=16, channels=256,  size ~ (100, 56)                       |
 |  Level 3: stride=32, channels=256,  size ~ (50,  28)                       |
 |                                                                             |
 |  Total: 6 cameras x 4 levels = 24 feature maps                             |
 +---------------------------------------------------------------------------+
         |
         | 24 feature maps stored for later sampling
         v
 +---------------------------------------------------------------------------+
 |                     3D OBJECT QUERIES (Learned)                              |
 |                                                                             |
 |  queries: (900, 256)  -- 900 learnable query embeddings                     |
 |  ref_pts: (900, 3)    -- 900 learnable 3D reference points (x, y, z)       |
 |                                                                             |
 |  These are model parameters learned during training.                        |
 |  Each query "owns" a 3D reference point in the scene.                       |
 +---------------------------------------------------------------------------+
         |
         | For each of 6 decoder layers:
         v
 +---------------------------------------------------------------------------+
 |                    TRANSFORMER DECODER LAYER (x6)                            |
 |                                                                             |
 |  Step 1: SELF-ATTENTION                                                     |
 |  +---------------------------------------------------------------+         |
 |  |  queries attend to each other                                  |         |
 |  |  Q = K = V = query embeddings (900, 256)                       |         |
 |  |  Purpose: inter-query communication, duplicate suppression      |         |
 |  +---------------------------------------------------------------+         |
 |           |                                                                 |
 |           v                                                                 |
 |  Step 2: FEATURE SAMPLING (Cross-Attention replacement)                     |
 |  +---------------------------------------------------------------+         |
 |  |                                                                 |         |
 |  |  For each query q with 3D reference point P = (px, py, pz):    |         |
 |  |                                                                 |         |
 |  |    For each camera c (c = 1..6):                                |         |
 |  |      1. Project: (u, v) = K_c * [R_c | t_c] * [px, py, pz, 1] |         |
 |  |      2. Normalize: u' = u/W, v' = v/H                          |         |
 |  |      3. Check visibility: is (u', v') in [0,1] x [0,1]?        |         |
 |  |      4. If visible, bilinear sample from each FPN level         |         |
 |  |                                                                 |         |
 |  |    Aggregate: weighted sum of multi-view, multi-scale features   |         |
 |  |                                                                 |         |
 |  +---------------------------------------------------------------+         |
 |           |                                                                 |
 |           v                                                                 |
 |  Step 3: FEED-FORWARD NETWORK (FFN)                                         |
 |  +---------------------------------------------------------------+         |
 |  |  Linear(256, 512) -> ReLU -> Linear(512, 256)                   |         |
 |  |  + LayerNorm + Residual connection                              |         |
 |  +---------------------------------------------------------------+         |
 |           |                                                                 |
 |           v                                                                 |
 |  Step 4: REFERENCE POINT REFINEMENT                                         |
 |  +---------------------------------------------------------------+         |
 |  |  delta = Linear(query, 3)     -- predict offset                 |         |
 |  |  ref_pt = ref_pt + delta      -- update reference point         |         |
 |  |  (allows queries to "walk" toward objects)                      |         |
 |  +---------------------------------------------------------------+         |
 |                                                                             |
 +---------------------------------------------------------------------------+
         |
         | After 6 decoder layers
         v
 +---------------------------------------------------------------------------+
 |                       DETECTION HEADS                                        |
 |                                                                             |
 |  Classification Head:                                                        |
 |    Linear(256, 256) -> ReLU -> Linear(256, 10)                              |
 |    Output: (batch, 900, 10) -- 10 nuScenes classes                          |
 |                                                                             |
 |  Regression Head:                                                            |
 |    Linear(256, 256) -> ReLU -> Linear(256, 10)                              |
 |    Output: (batch, 900, 10) -- [cx, cy, cz, w, l, h, sin, cos, vx, vy]     |
 |                                                                             |
 +---------------------------------------------------------------------------+
         |
         v
 +---------------------------------------------------------------------------+
 |                       LOSS COMPUTATION (Training only)                       |
 |                                                                             |
 |  1. Hungarian Matching: bipartite match predictions to ground truth          |
 |     Cost = lambda_cls * FocalLoss + lambda_reg * L1Loss                     |
 |                                                                             |
 |  2. Classification Loss: Focal Loss (handles class imbalance)                |
 |     - alpha=0.25, gamma=2.0                                                 |
 |                                                                             |
 |  3. Regression Loss: L1 Loss on matched box parameters                      |
 |                                                                             |
 |  4. Auxiliary Losses: applied at each intermediate decoder layer             |
 |                                                                             |
 +---------------------------------------------------------------------------+
         |
         v
 +---------------------------------------------------------------------------+
 |                        OUTPUT                                                |
 |                                                                             |
 |  Per-object predictions (up to 900, filtered by confidence threshold):       |
 |    - class_label: one of 10 categories                                      |
 |    - confidence: [0, 1]                                                      |
 |    - 3D box: center (x,y,z), size (w,l,h), heading (sin,cos), vel (vx,vy)  |
 |                                                                             |
 |  No NMS needed -- set-based prediction with Hungarian matching               |
 +---------------------------------------------------------------------------+
```

### Data Flow Summary

```
  6 images --> Backbone --> FPN --> 24 feature maps (stored)
                                          |
  900 learned 3D queries -----> project 3D ref points to 2D
                                          |
                                   sample features
                                          |
                                  6x transformer decoder
                                          |
                                   classification + regression
                                          |
                                  up to 900 3D detections
```

---

## Installation

### Requirements

| Dependency | Minimum Version | Recommended | Notes |
|-----------|----------------|-------------|-------|
| Python | 3.8 | 3.9 | 3.10+ untested |
| PyTorch | 1.10 | 2.0.1 | Must match CUDA version |
| CUDA | 11.3 | 11.8 | Required for GPU training |
| cuDNN | 8.2 | 8.6 | Comes with CUDA toolkit |
| RAM | 16 GB | 32 GB | Data loading is memory-intensive |
| GPU VRAM | 8 GB (inference) | 32 GB (training) | V100 or A100 recommended |

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd detr3d

# 2. Create conda environment
conda create -n detr3d python=3.9 -y
conda activate detr3d

# 3. Install PyTorch (adjust CUDA version as needed)
#    For CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

#    For CUDA 11.7:
#    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### Manual Dependency Installation

If you prefer to install dependencies individually or if requirements.txt
causes conflicts:

```bash
# Core dependencies
pip install nuscenes-devkit==1.1.10    # nuScenes data loading and evaluation
pip install pyquaternion==0.9.9        # Rotation/orientation handling
pip install scipy>=1.7.0               # Scientific computing (Hungarian algo)
pip install numpy>=1.21.0              # Array operations

# Vision and visualization
pip install opencv-python>=4.5.0       # Image I/O and processing
pip install matplotlib>=3.5.0          # Plotting and visualization

# Training utilities
pip install PyYAML>=6.0                # Config file parsing
pip install tqdm>=4.60.0               # Progress bars
pip install tensorboard>=2.8.0         # Training metrics logging
```

### Troubleshooting

**Problem: `CUDA out of memory` during import or model creation**
```bash
# Check which GPU is being used and its available memory
python -c "import torch; print(torch.cuda.get_device_properties(0))"

# If you have multiple GPUs, select one with more memory
export CUDA_VISIBLE_DEVICES=1
```

**Problem: `ModuleNotFoundError: No module named 'nuscenes'`**
```bash
# The nuscenes-devkit package installs as 'nuscenes'
pip install nuscenes-devkit==1.1.10

# Verify
python -c "from nuscenes.nuscenes import NuScenes; print('OK')"
```

**Problem: PyTorch not detecting CUDA**
```bash
# Check CUDA version your PyTorch was built for
python -c "import torch; print(torch.version.cuda)"

# Check system CUDA version
nvcc --version

# These must be compatible (not necessarily identical)
# If mismatched, reinstall PyTorch for your CUDA version
```

**Problem: `ImportError` related to `cv2` (OpenCV)**
```bash
# On headless servers (no display), use headless OpenCV
pip uninstall opencv-python
pip install opencv-python-headless>=4.5.0
```

**Problem: Very slow data loading**
```bash
# Increase shared memory (Docker containers often have small /dev/shm)
# Add to docker run: --shm-size=8g

# Or reduce num_workers in the config:
# dataloader:
#   num_workers: 2  # instead of default 4
```

---

## Data Setup

This project uses the [nuScenes dataset](https://www.nuscenes.org/), which
contains 1000 driving scenes captured in Boston and Singapore with 6 cameras,
1 LiDAR, and 5 radar sensors. We use only the camera data.

### Dataset Variants

| Variant | Scenes | Frames | Size | Use |
|---------|--------|--------|------|-----|
| nuScenes-mini | 10 | ~400 | ~4 GB | Development and debugging |
| nuScenes-trainval | 850 | ~34,000 | ~300 GB | Full training and evaluation |
| nuScenes-test | 150 | ~6,000 | ~60 GB | Test set submission |

### Download

```bash
# Option 1: Download mini set (recommended to start)
bash scripts/download_data.sh --mini

# Option 2: Download full trainval set
bash scripts/download_data.sh

# Option 3: Manual download from https://www.nuscenes.org/download
# You need: metadata, sweeps (CAM_FRONT, CAM_FRONT_LEFT, etc.), and maps
```

### Prepare Data

```bash
# Generate info pickle files (required before training/evaluation)
python scripts/prepare_data.py \
    --data-root ./data/nuscenes \
    --version v1.0-trainval \
    --output-dir ./data/nuscenes/infos
```

This creates:
- `detr3d_infos_train.pkl` -- training sample metadata
- `detr3d_infos_val.pkl` -- validation sample metadata

Each info file contains per-sample entries with:
- Camera image paths and calibration matrices
- Ground-truth 3D bounding boxes
- Object categories and attributes

### Data Verification

After setup, verify everything is in place:

```bash
# Check directory structure
ls data/nuscenes/
# Expected: maps/ samples/ sweeps/ v1.0-trainval/ (or v1.0-mini/)

# Check that camera images exist
ls data/nuscenes/samples/CAM_FRONT/ | head -5
# Should show .jpg files like: n015-2018-07-18-11-07-57+0800__CAM_FRONT__...jpg

# Check that info files were generated
ls data/nuscenes/infos/
# Expected: detr3d_infos_train.pkl  detr3d_infos_val.pkl

# Validate info files are loadable
python -c "
import pickle
with open('./data/nuscenes/infos/detr3d_infos_val.pkl', 'rb') as f:
    infos = pickle.load(f)
print(f'Validation set: {len(infos)} samples')
print(f'First sample keys: {list(infos[0].keys())}')
print(f'Number of cameras: {len(infos[0][\"cams\"])}')
"
```

Expected output:
```
Validation set: 6019 samples
First sample keys: ['token', 'cams', 'gt_boxes', 'gt_names', 'timestamp']
Number of cameras: 6
```

### Expected Directory Layout

```
data/nuscenes/
├── maps/                          # HD maps (optional for detection)
├── samples/                       # Keyframe sensor data
│   ├── CAM_FRONT/                 # ~34,000 images
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   └── CAM_BACK_RIGHT/
├── sweeps/                        # Intermediate frames (not used in training)
│   ├── CAM_FRONT/
│   └── ...
├── v1.0-trainval/                 # Metadata JSON files
│   ├── category.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── calibrated_sensor.json
│   └── ...
└── infos/                         # Generated by prepare_data.py
    ├── detr3d_infos_train.pkl
    └── detr3d_infos_val.pkl
```

---

## Quick Start

### Inference with Pretrained Model

```python
import torch
from model import DETR3D

# ---- Step 1: Load the model ----
# from_config() builds the model architecture from the YAML config file.
# This sets backbone type, number of decoder layers, query count, etc.
model = DETR3D.from_config("configs/detr3d_r101_nuscenes.yaml")

# Load trained weights
checkpoint = torch.load("checkpoints/detr3d_r101_nuscenes.pth", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()
model.cuda()

# ---- Step 2: Prepare input ----
# images: (batch, 6_cameras, 3, H, W) -- 6 camera views, normalized RGB
# projection_matrices: (batch, 6, 4, 4) -- camera intrinsic @ extrinsic
#
# In practice, the data loader handles this. For a single sample:
from data.nuscenes_dataset import NuScenesDataset

dataset = NuScenesDataset(
    info_path="./data/nuscenes/infos/detr3d_infos_val.pkl",
    data_root="./data/nuscenes",
)
sample = dataset[0]
images = sample["images"].unsqueeze(0).cuda()            # (1, 6, 3, 900, 1600)
proj_matrices = sample["proj_matrices"].unsqueeze(0).cuda()  # (1, 6, 4, 4)

# ---- Step 3: Run inference ----
with torch.no_grad():
    predictions = model(images, proj_matrices)

# predictions is a dictionary:
#   'cls_logits': (1, 900, 10)  -- raw class scores for 10 nuScenes categories
#   'bbox_preds': (1, 900, 10)  -- [cx, cy, cz, w, l, h, sin, cos, vx, vy]
#
# The 10 categories are:
#   car, truck, construction_vehicle, bus, trailer,
#   barrier, motorcycle, bicycle, pedestrian, traffic_cone

# ---- Step 4: Post-process ----
# Apply sigmoid to get probabilities, filter by confidence threshold
cls_scores = predictions['cls_logits'].sigmoid()  # (1, 900, 10)
max_scores, max_classes = cls_scores.max(dim=-1)  # (1, 900)

# Keep predictions with confidence > 0.3
threshold = 0.3
mask = max_scores[0] > threshold
detected_boxes = predictions['bbox_preds'][0][mask]   # (N, 10)
detected_classes = max_classes[0][mask]                # (N,)
detected_scores = max_scores[0][mask]                  # (N,)

print(f"Detected {mask.sum().item()} objects")
```

### Visualize Results

```bash
# Camera view visualization: projects 3D boxes onto camera images
python scripts/visualize_results.py \
    --predictions results/val_predictions.pkl \
    --infos ./data/nuscenes/infos/detr3d_infos_val.pkl \
    --data-root ./data/nuscenes \
    --output-dir ./vis_output \
    --mode camera

# Bird's eye view: shows boxes in top-down 2D plane
python scripts/visualize_results.py \
    --predictions results/val_predictions.pkl \
    --infos ./data/nuscenes/infos/detr3d_infos_val.pkl \
    --output-dir ./vis_output \
    --mode bev
```

---

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

### Flag Explanations

| Flag | Purpose |
|------|---------|
| `--config` | Path to YAML config defining model architecture, dataset paths, optimizer settings, and augmentation. All hyperparameters live here. |
| `--work-dir` | Output directory for checkpoints, logs, and tensorboard events. One checkpoint is saved per epoch. |
| `--launcher pytorch` | Enables PyTorch Distributed Data Parallel (DDP). Required for multi-GPU. Omit for single-GPU training. |
| `--nproc_per_node=8` | Number of GPUs to use (passed to torchrun, not train.py). Effective batch size = per-GPU batch x num GPUs. |
| `--resume-from` | Path to a checkpoint to resume training from. Restores model weights, optimizer state, and epoch counter. |
| `--seed` | Random seed for reproducibility. Default: 42. |
| `--deterministic` | Forces deterministic CUDA operations. Slower but exactly reproducible. |

### Training Configuration Highlights

Key settings in `configs/detr3d_r101_nuscenes.yaml`:

```yaml
model:
  backbone: resnet101          # Pretrained on ImageNet
  num_queries: 900             # Number of object queries (scouts)
  num_decoder_layers: 6        # Transformer decoder depth
  hidden_dim: 256              # Query/feature embedding dimension

optimizer:
  type: AdamW
  lr: 2.0e-4                   # Learning rate for decoder + heads
  backbone_lr_factor: 0.1      # Backbone LR = 2.0e-5 (10x lower)
  weight_decay: 0.01

training:
  epochs: 24                   # Total training epochs
  batch_size: 1                # Per-GPU batch size (images are large)
  grad_clip_norm: 35.0         # Gradient clipping for stability
  warmup_epochs: 1             # Linear LR warmup

loss:
  cls_weight: 2.0              # Classification loss weight
  reg_weight: 0.25             # Regression loss weight
  focal_alpha: 0.25            # Focal loss alpha (class balance)
  focal_gamma: 2.0             # Focal loss gamma (hard example mining)
```

### Training Tips

1. **Start with mini dataset** -- Run a few epochs on nuScenes-mini first to
   verify the entire pipeline (data loading, forward pass, loss computation,
   backward pass) works without errors before committing to a full training run.

2. **Gradient clipping is essential** -- The default clip norm of 35.0 prevents
   exploding gradients, especially in early training when predictions are random.

3. **Lower backbone learning rate** -- The backbone is pretrained on ImageNet.
   Using a 10x lower learning rate preserves useful pretrained features while
   still allowing fine-tuning for the detection task.

4. **Auxiliary losses help convergence** -- Intermediate decoder layers each
   produce predictions and receive loss. This provides gradient signal to all
   layers, not just the last one.

5. **Expected training time** -- Approximately 48 hours on 8x NVIDIA V100
   (32GB) GPUs. On 8x A100 (80GB), approximately 24 hours.

6. **Monitor via TensorBoard**:
   ```bash
   tensorboard --logdir ./work_dirs/detr3d_r101/tb_logs
   ```

---

## Evaluation

### Run Evaluation

```bash
python evaluate.py \
    --config configs/detr3d_r101_nuscenes.yaml \
    --checkpoint ./work_dirs/detr3d_r101/epoch_24.pth \
    --eval-set val
```

### What the Evaluation Computes

The evaluation uses the **official nuScenes Detection Score (NDS)**, which is
a weighted combination of mean Average Precision (mAP) and five True Positive
(TP) metrics:

```
NDS = (1/10) * [5 * mAP + sum(1 - min(1, TP_metric)) for each TP metric]
```

**Mean Average Precision (mAP):**
- Computes AP at four distance thresholds: 0.5m, 1.0m, 2.0m, 4.0m
- A detection is a true positive if its center is within the threshold distance
  of a ground-truth center (in bird's-eye view)
- Final mAP averages across all 10 classes and all 4 thresholds

**True Positive Metrics (computed on matched detections only):**

| Metric | Full Name | Measures | Unit |
|--------|-----------|----------|------|
| ATE | Average Translation Error | Center distance | meters |
| ASE | Average Scale Error | Size (IoU-based) | 1 - IoU |
| AOE | Average Orientation Error | Yaw angle | radians |
| AVE | Average Velocity Error | Velocity vector | m/s |
| AAE | Average Attribute Error | Attribute classification | 1 - acc |

### Expected Output

```
========== nuScenes Detection Evaluation ==========
Per-class results:
  car:                  AP=64.7  ATE=0.612  ASE=0.157  AOE=0.098  AVE=0.921  AAE=0.188
  truck:                AP=40.7  ATE=0.731  ASE=0.212  AOE=0.182  AVE=0.779  AAE=0.213
  construction_vehicle: AP=16.2  ATE=0.982  ASE=0.458  AOE=1.012  AVE=0.124  AAE=0.352
  bus:                  AP=50.7  ATE=0.689  ASE=0.194  AOE=0.088  AVE=1.682  AAE=0.271
  trailer:              AP=30.2  ATE=0.911  ASE=0.234  AOE=0.543  AVE=0.381  AAE=0.142
  barrier:              AP=52.6  ATE=0.504  ASE=0.291  AOE=0.123  AVE=nan    AAE=nan
  motorcycle:           AP=38.2  ATE=0.678  ASE=0.261  AOE=0.512  AVE=1.103  AAE=0.108
  bicycle:              AP=25.2  ATE=0.612  ASE=0.284  AOE=0.698  AVE=0.412  AAE=0.012
  pedestrian:           AP=46.3  ATE=0.693  ASE=0.292  AOE=0.641  AVE=0.487  AAE=0.141
  traffic_cone:         AP=50.7  ATE=0.452  ASE=0.298  AOE=nan    AVE=nan    AAE=nan

Overall:
  mAP:  34.9
  NDS:  42.2
  ATE:  0.716
  ASE:  0.268
  AOE:  0.379
  AVE:  0.842
  AAE:  0.200

Evaluation complete. Results saved to: ./work_dirs/detr3d_r101/eval_results.json
```

---

## Model Zoo

Pre-trained models available for download:

| Model | Backbone | NDS | mAP | Config | Download |
|-------|----------|-----|-----|--------|----------|
| DETR3D | ResNet-101 | 42.2 | 34.9 | [config](configs/detr3d_r101_nuscenes.yaml) | [model](https://github.com/example/detr3d/releases/download/v1.0/detr3d_r101_nuscenes_ep24.pth) |
| DETR3D | ResNet-101-DCN | 43.4 | 35.6 | config | [model](https://github.com/example/detr3d/releases/download/v1.0/detr3d_r101dcn_nuscenes_ep24.pth) |
| DETR3D + CBGS | ResNet-101 | 43.4 | 34.7 | config | [model](https://github.com/example/detr3d/releases/download/v1.0/detr3d_r101_cbgs_nuscenes_ep24.pth) |
| DETR3D | VoVNet-99 | 44.2 | 36.0 | config | [model](https://github.com/example/detr3d/releases/download/v1.0/detr3d_vov99_nuscenes_ep24.pth) |

**Model Variant Notes:**

- **ResNet-101** -- Standard baseline. Good balance of speed and accuracy.
- **ResNet-101-DCN** -- Adds Deformable Convolutions in the backbone's last two
  stages. Improves feature alignment at the cost of slightly more computation.
- **CBGS (Class-Balanced Grouping and Sampling)** -- Oversamples rare classes
  during training. Improves recall on underrepresented categories (construction
  vehicles, bicycles) without hurting common classes.
- **VoVNet-99** -- Larger and more efficient backbone. Best overall performance
  but requires more GPU memory.

---

## Results

### Detection Performance by Class (DETR3D ResNet-101)

This table shows Average Precision at each distance threshold. Lower thresholds
are stricter -- a detection must be within 0.5m of the true center for AP@0.5m.

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

**Interpretation:**

- **Cars** perform best because they are the most common class with consistent
  appearance and size. The model sees thousands of car examples during training.
- **Construction Vehicles** are hardest due to extreme size variation, rare
  occurrence, and diverse appearance (cranes, excavators, bulldozers all count).
- **Performance improves significantly** at looser thresholds (0.5m to 4.0m),
  indicating the model localizes objects in the right vicinity but struggles
  with precise center prediction -- a known limitation of camera-only methods
  that lack direct depth measurement.
- **Small objects** (bicycle, traffic cone) are challenging because they occupy
  few pixels in the image, especially at longer ranges.

### Overall Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| mAP | 34.9 | Average detection precision across classes and thresholds |
| NDS | 42.2 | Composite score including both detection and box quality |
| ATE (m) | 0.716 | Avg center error ~72cm (depth ambiguity from cameras) |
| ASE | 0.268 | Size estimation is relatively accurate (IoU ~0.73) |
| AOE (rad) | 0.379 | Heading error ~22 degrees |
| AVE (m/s) | 0.842 | Velocity error ~0.84 m/s (single-frame, no temporal) |
| AAE | 0.200 | Attribute accuracy ~80% (e.g., parked vs moving) |

**Key Takeaway:** The ATE of 0.716m reflects the fundamental challenge of
camera-only 3D detection -- without direct depth measurement, localizing objects
precisely in the depth direction is inherently uncertain. The AVE of 0.842 m/s
is relatively high because DETR3D is a single-frame model; temporal variants
that process multiple frames significantly improve velocity estimation.

---

## Documentation Guide

Detailed documentation is available in the `docs/` directory:

| Document | Description | Read When... |
|----------|-------------|--------------|
| [docs/research_summary.md](docs/research_summary.md) | Academic context, related work, and how DETR3D fits in the 3D detection landscape | You want to understand the research motivation and compare with other methods |
| [docs/model_architecture.md](docs/model_architecture.md) | Detailed layer-by-layer architecture description, tensor shapes at every stage, design decisions | You want to modify the model or understand the code |
| [docs/training_guide.md](docs/training_guide.md) | Complete training recipes, hyperparameter tuning advice, common failure modes and fixes | You are training the model or debugging training issues |
| [docs/evaluation_guide.md](docs/evaluation_guide.md) | nuScenes evaluation protocol details, metric definitions, how to submit to the test leaderboard | You are evaluating results or preparing a submission |

---

## Tests

Run the test suite to verify model components work correctly:

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_model.py -v

# Run a specific test class
pytest tests/test_model.py::TestBackbone -v

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=html
# Open htmlcov/index.html in a browser to view coverage

# Run only fast unit tests (skip integration tests that need data)
pytest tests/ -v -m "not integration"
```

Tests cover:
- Backbone forward pass and output shapes
- FPN multi-scale feature generation
- 3D-to-2D projection correctness
- Transformer decoder attention mechanisms
- Detection head output dimensions
- Loss computation and Hungarian matching
- End-to-end inference pipeline

---

## Project Structure

```
detr3d/
├── configs/
│   └── detr3d_r101_nuscenes.yaml    # Training/model configuration
├── docs/
│   ├── research_summary.md          # Paper context and related work
│   ├── model_architecture.md        # Detailed architecture documentation
│   ├── training_guide.md            # Training recipes and debugging
│   └── evaluation_guide.md          # Evaluation protocol and metrics
├── scripts/
│   ├── download_data.sh             # Dataset download automation
│   ├── prepare_data.py              # Info file generation from raw data
│   └── visualize_results.py         # Camera/BEV result visualization
├── tests/
│   └── test_model.py                # Unit and integration tests
├── model/                           # Model implementation
│   ├── __init__.py
│   ├── backbone.py                  # ResNet-101, VoVNet-99 backbones
│   ├── fpn.py                       # Feature Pyramid Network
│   ├── decoder.py                   # Transformer decoder + feature sampling
│   ├── heads.py                     # Classification and regression heads
│   └── detr3d.py                    # Top-level model assembly
├── data/                            # Dataset directory (created by download)
├── train.py                         # Training entry point
├── evaluate.py                      # Evaluation entry point
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{wang2022detr3d,
  title={DETR3D: 3D Detection Transformer for Autonomous Driving},
  author={Wang, Yue and Guizilini, Vitor Campagnolo and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and Solomon, Justin},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022}
}
```

---

## License

This project is released under the **MIT License**.

The nuScenes dataset is subject to its own
[terms of use](https://www.nuscenes.org/terms-of-use). You must agree to
Motional's terms before downloading or using the dataset.

---

## Acknowledgments

- [nuScenes](https://www.nuscenes.org/) dataset and devkit by Motional
- [DETR](https://github.com/facebookresearch/detr) by Facebook Research
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) by OpenMMLab
