# PETR / PETRv2 / StreamPETR - Camera-Based 3D Object Detection

A TensorFlow 2 implementation of the PETR family of models for detecting 3D objects
from multiple camera views on an autonomous vehicle. These models take six camera images
covering the full 360-degree view around a car and produce 3D bounding boxes for every
detected object -- including its class (car, pedestrian, cyclist, etc.), 3D position,
dimensions, heading angle, and velocity -- all without requiring expensive LiDAR sensors.

---

## What Problem Does This Solve?

### The Autonomous Driving Perception Challenge

An autonomous vehicle must understand the 3D world around it in real time. It needs to
answer questions like: "Where are the other cars? How fast are they going? Is that
pedestrian about to step into my path?" The answers take the form of **3D bounding
boxes** -- rectangles drawn in three-dimensional space around every object of interest.

Traditional approaches use LiDAR (Light Detection And Ranging), which fires laser pulses
and measures how long they take to bounce back. LiDAR directly gives you depth -- the
distance to every surface -- making 3D detection relatively straightforward. However,
LiDAR sensors cost thousands of dollars, produce sparse point clouds (typically 32-128
vertical lines), and provide no color or texture information.

**Cameras** are the alternative: they are cheap (under $50 each), capture rich color and
texture detail at high resolution, and are already standard equipment on every production
vehicle. The catch? A camera image is a 2D projection of the 3D world. Depth information
is lost during the imaging process. A small car 10 meters away can look identical to a
large truck 50 meters away in a single image.

This project implements PETR, PETRv2, and StreamPETR -- a family of neural network models
that solve this problem by teaching a transformer to reason about 3D geometry directly
from 2D image features, without ever explicitly reconstructing a depth map or 3D point
cloud.

### Sensor Setup

The nuScenes dataset (and most modern autonomous vehicles) uses six cameras arranged
around the vehicle to achieve a complete 360-degree field of view:

```
                        FRONT
                     ____________
                    /   CAM_F    \
                   / (70-degree)  \
     FRONT_LEFT  /                 \  FRONT_RIGHT
    ____________/                   \____________
   /  CAM_FL   |                   |  CAM_FR    \
  / (70-deg)   |                   |  (70-deg)   \
 /             |       EGO         |              \
|              |     VEHICLE       |               |
 \             |     [=====]       |              /
  \            |                   |             /
   \___________\                   /____________/
   /  CAM_BL    \                 /  CAM_BR    \
  / (70-deg)     \               / (70-deg)     \
 /   BACK_LEFT    \_____________/  BACK_RIGHT    \
 \                 /  CAM_B    \                  /
  \               / (70-degree) \                /
   \_____________/    BACK       \______________/
```

Each camera captures images at 900x1600 pixels, 12 Hz. Together they provide complete
surround coverage with overlapping fields of view for redundancy.

### What Goes In, What Comes Out

```
INPUT:                                    OUTPUT:
                                          
 6 Camera Images                          3D Bounding Boxes (per object):
 (each 900 x 1600 pixels)                 
                                           +-- class: "car"
 +-------+  +-------+  +-------+          |-- position: (x=12.3, y=-3.1, z=0.5) meters
 | Front |  | F-Left|  |F-Right|          |-- dimensions: (w=1.9, l=4.7, h=1.5) meters
 |  Cam  |  |  Cam  |  |  Cam  |   --->   |-- heading: 45 degrees
 +-------+  +-------+  +-------+          |-- velocity: (vx=8.2, vy=0.1) m/s
 +-------+  +-------+  +-------+          +-- confidence: 0.92
 | Back  |  | B-Left|  |B-Right|          
 |  Cam  |  |  Cam  |  |  Cam  |          ... repeated for every detected object
 +-------+  +-------+  +-------+          (up to 300 objects per frame)
                                          
 + Camera calibration matrices            Covering 10 classes:
 + Ego-vehicle pose (for temporal)        car, truck, bus, trailer, barrier,
                                          construction_vehicle, motorcycle,
                                          bicycle, pedestrian, traffic_cone
```

The model processes all six views simultaneously and reasons about object positions in
a unified 3D coordinate system centered on the ego vehicle. It outputs detections in the
ego vehicle frame, where X points forward, Y points left, and Z points up.

---

## How Does It Work? (High-Level Intuition)

### The Core Idea: Giving 2D Features a Sense of 3D Position

The fundamental insight behind PETR is surprisingly elegant. Consider that a transformer
can already learn to relate different image patches through attention -- the problem is
that it has no idea where those patches exist in 3D space. A pixel showing a car bumper
at (row=450, col=800) in the front camera image could be at 10 meters or 50 meters
away. Without 3D awareness, the transformer cannot produce 3D bounding boxes.

PETR's solution: **encode 3D position information directly into the image features**.
For every pixel in every camera, the model creates a set of hypothetical 3D points along
the camera ray (from 1 meter to 61 meters depth, at 64 evenly spaced intervals). It
then projects these 3D coordinates into a learned embedding and adds them to the image
features. After this step, each image feature "knows" not just what it shows (a bumper,
a wheel, a head), but also where in 3D space it could possibly be.

Once features are position-aware, a standard transformer decoder with object queries can
attend to them and naturally reason in 3D -- selecting the correct depth hypothesis for
each detection through learned attention patterns.

### The Temporal Extension: Remembering Objects Across Frames

A single camera frame provides limited information. But if you observe the same car
across multiple frames, you can estimate its velocity, confirm its existence despite
momentary occlusions, and improve your position estimate. This is temporal modeling.

**PETRv2** approaches temporal modeling by aligning previous frames' position-aware
features to the current frame's coordinate system (compensating for ego-vehicle motion),
then letting the transformer attend to both current and historical features.

**StreamPETR** takes a fundamentally different and more efficient approach: instead of
storing and re-processing all previous image features, it only propagates the **object
queries** from the previous frame. Since queries are compact vectors (256 queries of 256
dimensions = 64K parameters vs. 40,000+ BEV feature cells), this is dramatically cheaper.
Propagated queries carry the "memory" of previously detected objects and are ego-motion
compensated before being fed back into the current frame's decoder.

### An Analogy

Think of it like reading a book in a foreign language with a spatial dictionary:

- **Without PETR**: You see all the words (image features) but have no grammar (3D
  structure). You cannot construct meaningful sentences (3D detections).
- **With PETR**: Each word is annotated with its grammatical role (3D position embedding).
  Now you can parse the sentence even in a language you have never seen before.
- **With StreamPETR**: You remember the characters from previous chapters (propagated
  queries) and use that context to understand new scenes faster, without re-reading
  earlier chapters (re-processing old frames).

---

## Key Innovations

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| 3D Position Embedding (3D PE) | Camera frustum points projected to world coordinates and encoded via MLP, providing geometry-aware features | Eliminates the need for explicit depth estimation or Bird's Eye View (BEV) construction -- the model implicitly learns depth through attention |
| Object-Centric Temporal Modeling | Queries carry object state across frames (StreamPETR), rather than fusing dense feature maps | Memory usage is constant regardless of history length; 256 queries vs. 40K BEV cells gives ~150x memory reduction |
| Motion-Aware Layer Normalization | Ego-motion embedding modulates LayerNorm scale and shift parameters | The transformer adapts to viewpoint changes without explicitly transforming all features, enabling streaming inference |
| Linear Increasing Discretization (LID) | Non-uniform depth binning with finer resolution at close range | Objects nearby (where precision matters most) get more depth hypotheses, improving detection of pedestrians and cyclists |
| Hungarian Matching | Bipartite matching between predictions and ground truth for set-based loss | No need for hand-crafted anchor boxes or non-maximum suppression during training -- the model learns a clean one-to-one assignment |
| Ego-Motion Compensation | Previous-frame query positions are transformed to the current ego-vehicle coordinate frame | Propagated queries remain geometrically valid despite vehicle movement, enabling stable temporal tracking |

---

## Model Variants

The PETR family consists of three models with increasing capability. Here is a summary
to help you choose:

### PETR (ECCV 2022) -- Single-Frame Baseline

The foundation model. Processes each frame independently with no temporal context.
Best for understanding the core 3D position embedding idea or for applications where
latency is critical and temporal information is unavailable.

- Input: 6 camera images from a single timestamp
- Output: 3D detections for that frame only
- Strengths: Simplest to train, lowest memory, easiest to debug
- Limitations: No velocity estimation from motion, misses occluded objects

### PETRv2 (ICCV 2023) -- Temporal Feature Fusion + Multi-Task

Extends PETR by aligning and concatenating features from previous frames. Also supports
auxiliary tasks like BEV segmentation (drivable area, lane boundaries).

- Input: 6 camera images from current + previous frame(s)
- Output: 3D detections with improved accuracy, optional BEV segmentation map
- Strengths: Better depth accuracy from temporal cues, multi-task flexibility
- Limitations: Memory scales with number of temporal frames, moderate speed

### StreamPETR (ICCV 2023) -- Efficient Streaming Temporal

The state-of-the-art variant. Propagates lightweight object queries instead of dense
features. Achieves the best accuracy at the highest speed.

- Input: 6 camera images + propagated queries from previous frame
- Output: 3D detections with strong temporal consistency and accurate velocities
- Strengths: Real-time (30+ FPS), constant memory, natural object tracking
- Limitations: Slightly more complex training (streaming simulation), needs sequential data

### Quick Decision Guide

| Scenario | Recommended Model |
|----------|-------------------|
| Learning/prototyping, want simplest setup | PETR |
| Need multi-task (detection + segmentation) | PETRv2 |
| Production deployment, need speed + accuracy | StreamPETR |
| Limited GPU memory (single GPU training) | PETR |
| Downstream tracking pipeline | StreamPETR (queries provide implicit tracking) |

---

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

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: No module named 'tensorflow'` | Install TensorFlow: `pip install tensorflow>=2.10.0`. For GPU support, ensure CUDA 11.x and cuDNN 8.x are installed. |
| `nuscenes` import fails | Install the devkit: `pip install nuscenes-devkit>=1.1.9` |
| Out of memory during training | Reduce `batch_size` to 1, enable `mixed_precision: true`, or reduce `img_size` in config |
| CUDA version mismatch | Check `nvidia-smi` output and match TF version to your CUDA: TF 2.10 needs CUDA 11.2, TF 2.12+ needs CUDA 11.8 |
| Slow data loading | Increase `workers_per_gpu` (default 4), ensure data is on SSD not network drive |
| `scipy.optimize.linear_sum_assignment` errors | Update scipy: `pip install scipy>=1.8.0` (Hungarian matcher depends on this) |

---

## Quick Start

### 1. Download Data

The nuScenes dataset is the standard benchmark for multi-view 3D detection. Start with
the "mini" split (4GB) for testing your setup, then graduate to the full "trainval"
split (400GB) for real training.

```bash
# Download nuScenes mini split (4GB, ~10 scenes, good for testing pipeline)
bash scripts/download_data.sh --split mini --output_dir ./data/nuscenes

# Download pretrained ImageNet backbone weights (ResNet-50)
# These initialize the image feature extractor for faster convergence
bash scripts/download_data.sh --backbone --backbone_dir ./data/pretrained

# For full training, download the trainval split (~400GB, 700 training + 150 val scenes)
bash scripts/download_data.sh --split trainval --output_dir ./data/nuscenes
```

### 2. Prepare Data

This step pre-processes the raw nuScenes database into pickle files containing all the
metadata needed for training: camera intrinsics/extrinsics, ego poses, annotation boxes,
and frame-to-frame temporal linkage. This is done once and the results are cached.

```bash
python scripts/prepare_data.py \
    --data_root ./data/nuscenes \
    --version v1.0-mini \
    --output_dir ./data/infos
```

For temporal models (PETRv2, StreamPETR), the script also builds sequence graphs
linking consecutive frames and computing inter-frame ego-motion matrices.

### 3. Train

Training uses distributed data-parallel across multiple GPUs. The default configs expect
8 GPUs (effective batch size 8). You can train on fewer GPUs by increasing gradient
accumulation.

```bash
# PETR base (single-frame, simplest model)
python tensorflow/train.py --config configs/petr_r50.yaml --output_dir ./output/petr

# StreamPETR (temporal, recommended for best results)
python tensorflow/train.py --config configs/stream_petr_r50.yaml --output_dir ./output/stream_petr

# Multi-GPU training (specify GPU IDs)
python tensorflow/train.py --config configs/petr_r50.yaml --gpus 0,1,2,3

# Single-GPU with gradient accumulation (simulates larger batch)
python tensorflow/train.py --config configs/petr_r50.yaml --gpus 0 \
    --override training.accumulate_grad_batches=8
```

Training typically takes 24 epochs (~20 hours on 8x A100 for PETR-R50).

### 4. Evaluate

Run the official nuScenes evaluation protocol to compute mAP, NDS, and per-class metrics.

```bash
python tensorflow/evaluate.py \
    --config configs/petr_r50.yaml \
    --checkpoint ./output/petr/checkpoints/ckpt-24 \
    --data_info ./data/infos/petr_infos_val_v1_0-trainval.pkl \
    --data_root ./data/nuscenes \
    --output ./eval_results.json
```

### 5. Inference

Run the trained model on new data to produce detection results.

```bash
python tensorflow/inference.py \
    --config configs/petr_r50.yaml \
    --model_path ./output/petr/saved_model \
    --input ./data/infos/petr_infos_val_v1_0-mini.pkl \
    --output ./inference_results.pkl \
    --score_threshold 0.3
```

### 6. Visualize

Generate multi-view images with projected 3D boxes and a bird's eye view plot.

```bash
python scripts/visualize_results.py \
    --results ./inference_results.pkl \
    --data_info ./data/infos/petr_infos_val_v1_0-mini.pkl \
    --data_root ./data/nuscenes \
    --output_dir ./visualizations \
    --show_bev \
    --create_video
```

---

## Configuration

Configuration files are in `configs/` in YAML format. Here is an explanation of the
most important hyperparameters and when you might want to change them:

### Model Architecture Parameters

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `model.decoder.num_queries` | 900 | Maximum number of objects the model can detect per frame | Reduce to 300-500 for highway scenes (fewer objects); increase for dense urban |
| `model.decoder.embed_dims` | 256 | Feature dimension throughout the transformer | 512 for stronger models (2x memory), 128 for faster lightweight models |
| `model.decoder.num_decoder_layers` | 6 | Number of iterative refinement layers | 3-4 for faster inference at slight accuracy cost |
| `model.decoder.num_heads` | 8 | Attention heads in multi-head attention | Generally keep at 8; must divide embed_dims evenly |
| `model.position_embedding.depth_num` | 64 | Number of depth hypotheses per pixel ray | More bins = finer depth resolution but more memory |
| `model.position_embedding.pc_range` | [-51.2, ..., 51.2] | 3D perception volume in meters | Expand for highway (need further range), shrink for parking lots |

### Training Parameters

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `training.lr` | 2e-4 | Base learning rate | Scale linearly with effective batch size (2e-4 for batch=8) |
| `training.epochs` | 24 | Total training duration | 24 is standard; use 36-48 with CBGS for rare-class improvement |
| `training.batch_size` | 1 | Samples per GPU per step | Limited by GPU memory; use gradient accumulation instead |
| `training.backbone_lr_mult` | 0.1 | LR multiplier for pretrained backbone | 0.1 prevents catastrophic forgetting; use 1.0 if training from scratch |
| `training.grad_clip` | 35.0 | Maximum gradient norm | Lower (10-25) if training is unstable; higher if gradients are too aggressively clipped |
| `training.mixed_precision` | true | Use FP16 where safe | Disable if you see NaN losses (rare with dynamic loss scaling) |

### StreamPETR-Specific Parameters

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `model.query_propagation.num_propagated_queries` | 256 | Queries carried from previous frame | More = stronger temporal memory, fewer = more fresh detection capacity |
| `model.query_propagation.memory_len` | 512 | Historical query buffer size | Increase for long-term tracking in slow-moving scenarios |
| `model.motion_aware_layer_norm.enabled` | true | Inject ego-motion into LayerNorm | Always keep enabled for temporal models |
| `training.streaming.frames_per_clip` | 4 | Sequence length during training | Longer = better temporal learning but more memory |

### Loss Parameters

| Parameter | Default | Description | When to Change |
|-----------|---------|-------------|----------------|
| `loss.cls_loss.weight` | 2.0 | Classification loss importance | Increase if too many false negatives; decrease if too many false positives |
| `loss.bbox_loss.weight` | 0.25 | Box regression loss importance | Increase if localization is poor despite good classification |
| `loss.cls_loss.gamma` | 2.0 | Focal loss focusing parameter | Higher (3-5) focuses more on hard examples; lower (0-1) treats all examples more equally |

---

## Performance Benchmarks

Results reported in the original papers on nuScenes val set:

| Model | Backbone | mAP | NDS | FPS | GPU Memory |
|-------|----------|-----|-----|-----|------------|
| PETR | ResNet-50 | 0.313 | 0.381 | 15.3 | ~6 GB |
| PETR | ResNet-101 | 0.357 | 0.421 | 11.2 | ~8 GB |
| PETRv2 | ResNet-50 | 0.349 | 0.422 | 14.8 | ~10 GB |
| PETRv2 | V2-99 | 0.421 | 0.524 | 9.7 | ~16 GB |
| StreamPETR | ResNet-50 | 0.384 | 0.450 | 31.7 | ~8 GB |
| StreamPETR | V2-99 | 0.450 | 0.550 | 17.1 | ~16 GB |

### How to Read These Numbers

- **mAP (Mean Average Precision)**: The primary detection accuracy metric. Measures how
  well the model finds objects at various distance thresholds (0.5m, 1m, 2m, 4m center
  distance in BEV). A score of 0.45 means the model correctly finds and localizes 45%
  of objects across all thresholds and classes. State-of-the-art camera-only methods
  reach ~0.50-0.55 on nuScenes val.

- **NDS (nuScenes Detection Score)**: A composite metric that combines mAP with five
  True Positive error metrics (translation, scale, orientation, velocity, attribute).
  NDS rewards not just finding objects but also accurately estimating their properties.
  Formula: NDS = (5 * mAP + sum(1 - min(1, TP_error))) / 10

- **FPS (Frames Per Second)**: Inference speed measured on a single NVIDIA A100 GPU.
  Real-time autonomous driving requires at least 10 FPS for the perception pipeline.
  StreamPETR's 30+ FPS leaves ample headroom for other pipeline stages.

### Key Observations

- StreamPETR achieves higher FPS than frame-level methods by avoiding feature-level
  temporal fusion. It processes only current-frame images at full resolution while
  recycling compact query vectors from the past.

- Temporal modeling (StreamPETR/PETRv2) consistently improves both mAP and NDS over
  single-frame PETR. The biggest gains are in velocity estimation (mAVE) since motion
  is only observable across multiple frames.

- The object-centric approach (StreamPETR) scales better with sequence length than
  BEV-based methods. Adding more temporal history costs almost nothing in memory, while
  BEV temporal fusion (like BEVFormer) linearly increases memory with history length.

- Larger backbones (V2-99 vs ResNet-50) improve accuracy significantly (+7 mAP) but
  at the cost of speed and memory. For deployment, ResNet-50 with StreamPETR provides
  the best accuracy-speed tradeoff.

---

## Documentation Guide

Detailed documentation is available in the `docs/` directory. Each document serves a
specific purpose:

| Document | What It Covers | Read This When... |
|----------|---------------|-------------------|
| [research_summary.md](docs/research_summary.md) | First-principles explanation of the PETR approach, why each design decision was made, comparisons with alternatives (BEVFormer, DETR3D, etc.) | You want deep understanding of the theory and motivation |
| [model_architecture.md](docs/model_architecture.md) | Detailed architecture diagrams, layer-by-layer description, tensor shapes at each stage, attention pattern analysis | You are implementing changes or debugging the model |
| [data_collection.md](docs/data_collection.md) | nuScenes dataset specifications, sensor configuration, download procedures, directory structure, coordinate systems | You are setting up data for the first time or working with a new dataset |
| [annotation_guide.md](docs/annotation_guide.md) | 3D bounding box format, coordinate conventions, class definitions, ego-frame transforms, how annotations map to loss targets | You are confused about data formats or adapting to a new annotation scheme |
| [training_guide.md](docs/training_guide.md) | Hardware requirements, multi-GPU setup, hyperparameter tuning recipes, convergence troubleshooting, model-specific training strategies | You are training the model and want to optimize results |
| [evaluation_guide.md](docs/evaluation_guide.md) | nuScenes metrics explained, evaluation protocol, speed benchmarking, per-class analysis, ablation study reproduction | You are interpreting results or preparing a benchmark submission |

Recommended reading order for newcomers:
1. This README (you are here)
2. `docs/research_summary.md` -- understand the "why"
3. `docs/data_collection.md` -- understand the data
4. `docs/model_architecture.md` -- understand the "how"
5. `docs/training_guide.md` -- train your first model

---

## Running Tests

The test suite validates model components, loss computation, and data pipeline correctness.

```bash
# Run all tests
pytest tests/ -v

# Run specific test class (e.g., position embedding module)
pytest tests/test_model.py::TestPositionEmbedding3D -v

# Run with coverage report
pytest tests/ --cov=tensorflow --cov-report=term-missing

# Run only fast unit tests (skip integration tests that need data)
pytest tests/ -v -m "not integration"
```

Key test classes:
- `TestPositionEmbedding3D` -- validates frustum generation and 3D PE encoding
- `TestTransformerDecoder` -- checks attention shapes and gradient flow
- `TestHungarianMatcher` -- verifies bipartite matching correctness
- `TestDetectionHead` -- tests classification and regression output shapes
- `TestStreamPETRPropagation` -- validates query propagation and ego-motion compensation

---

## File Structure

```
petr/
├── tensorflow/
│   ├── model.py          # Core model: backbone (ResNet), FPN neck, 3D position
│   │                     # embedding generator, transformer decoder, detection head.
│   │                     # Contains both PETR and StreamPETR architectures with
│   │                     # conditional temporal components.
│   ├── train.py          # Training loop: multi-GPU distributed training, mixed
│   │                     # precision, cosine LR schedule with warmup, gradient
│   │                     # clipping, checkpoint saving, and streaming simulation
│   │                     # for StreamPETR.
│   ├── evaluate.py       # Evaluation script: runs inference on val set, computes
│   │                     # official nuScenes metrics (mAP, NDS, per-class AP),
│   │                     # and generates a structured JSON report.
│   ├── inference.py      # Inference pipeline: loads SavedModel or checkpoint,
│   │                     # supports batch processing, applies score filtering and
│   │                     # NMS, outputs pickle files for downstream use.
│   └── __init__.py       # Package initialization
├── pytorch/
│   ├── model.py          # PyTorch reference implementation (mirrors tensorflow/)
│   ├── backbone.py       # ResNet/VoVNet backbone with pretrained weight loading
│   ├── position_embedding_3d.py  # 3D PE generation: frustum creation, camera
│   │                     # projection, MLP encoding, LID discretization
│   ├── temporal.py       # Temporal modules: PETRv2 feature alignment, StreamPETR
│   │                     # query propagation, motion-aware LayerNorm
│   ├── decoder.py        # Transformer decoder: multi-head attention, FFN, query
│   │                     # initialization, iterative refinement
│   ├── heads.py          # Detection head: classification + regression branches
│   ├── losses.py         # Loss functions: focal loss, L1, Hungarian matcher
│   ├── dataset.py        # nuScenes dataset loader with temporal sequencing
│   ├── train.py          # PyTorch training script
│   ├── evaluate.py       # PyTorch evaluation script
│   ├── inference.py      # PyTorch inference script
│   └── __init__.py       # Package initialization
├── scripts/
│   ├── download_data.sh  # Downloads nuScenes splits (mini/trainval/test) and
│   │                     # pretrained backbone weights from official sources
│   ├── prepare_data.py   # Generates annotation pickle files from raw nuScenes
│   │                     # database: extracts calibration, poses, boxes, and
│   │                     # builds temporal frame linkage graphs
│   └── visualize_results.py  # Visualization: projects 3D boxes onto multi-view
│                         # images, generates BEV plots, creates video sequences,
│                         # and supports side-by-side GT vs prediction comparison
├── configs/
│   ├── petr_r50_nuscenes.yaml       # PETR base config (single-frame, ResNet-50)
│   ├── petrv2_r50_nuscenes.yaml     # PETRv2 config (temporal + multi-task)
│   └── stream_petr_r50_nuscenes.yaml # StreamPETR config (streaming temporal)
├── tests/
│   ├── test_model.py     # Comprehensive pytest suite: unit tests for every module,
│   │                     # integration tests for forward pass, gradient checks
│   └── __init__.py       # Test package initialization
├── docs/
│   ├── research_summary.md      # First-principles explanation of PETR approach
│   ├── model_architecture.md    # Detailed architecture and tensor flow diagrams
│   ├── data_collection.md       # nuScenes dataset setup and format documentation
│   ├── annotation_guide.md      # 3D bounding box conventions and coordinate systems
│   ├── training_guide.md        # Practical training recipes and troubleshooting
│   └── evaluation_guide.md      # Metrics explanation and benchmarking guide
└── README.md             # This file
```

---

## Citation

If you use this implementation in your research, please cite the original papers:

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

---

## License

This implementation is for research purposes. The nuScenes dataset is subject to its own
[license terms](https://www.nuscenes.org/terms-of-use). Please review and comply with
the nuScenes Terms of Use before downloading or using the dataset.
