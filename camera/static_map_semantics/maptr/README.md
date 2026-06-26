# MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction

Implementation of **MapTR** (ICLR 2023) and **MapTRv2** for online vectorized HD map construction from multi-camera images.

**Paper:** [MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction](https://arxiv.org/abs/2208.14437)
**Authors:** Bencheng Liao, Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Qian Zhang, Wenyu Liu, Chang Huang

---

## Key Contributions

1. **Permutation-Equivalent Modeling**: Map elements (polylines) are modeled as ordered point sets with permutation invariance. The model considers all equivalent point orderings (cyclic shifts and direction reversals) during training, eliminating the need for canonical ordering.

2. **Hierarchical Bipartite Matching**: A two-level matching strategy:
   - *Instance-level*: Hungarian matching assigns predicted queries to ground truth map elements
   - *Point-level*: Finds the optimal permutation of points within each matched pair

3. **Unified Architecture**: End-to-end transformer-based architecture that directly predicts vectorized map elements without intermediate rasterization.

4. **MapTRv2 Improvements**: Auxiliary one-to-many matching for faster convergence, decoupled self-attention for efficiency, and dense BEV supervision.

---

## Architecture Overview

```
Multi-Camera Images (6 views)
        |
        v
+------------------+
| ResNet-50 + FPN  |  (Multi-scale feature extraction)
+------------------+
        |
        v
+------------------+
|       GKT        |  (Geometry-guided Kernel Transformer)
| Perspective->BEV |  (Projects image features to BEV using camera geometry)
+------------------+
        |
        v
+------------------+
|   Map Decoder    |  (Transformer decoder with hierarchical queries)
| Instance queries |  (N_q instance embeddings + N_p point embeddings)
| + Point queries  |  (Self-attention + Cross-attention to BEV)
+------------------+
        |
        v
+------------------+
|   Output Heads   |
| Classification   |  -> per-instance class logits
| Point Regression |  -> per-instance ordered point coordinates
+------------------+
```

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0 (or TensorFlow >= 2.8.0)
- torchvision >= 0.10.0
- scipy >= 1.7.0
- numpy >= 1.20.0
- matplotlib >= 3.4.0
- Pillow >= 8.0.0
- PyYAML >= 5.4.0
- nuscenes-devkit >= 1.1.9

### Install

```bash
# Clone the repository
cd camera/static_map_semantics/maptr

# Install dependencies (PyTorch)
pip install torch torchvision scipy numpy matplotlib Pillow pyyaml
pip install nuscenes-devkit

# Or for TensorFlow
pip install tensorflow scipy numpy matplotlib Pillow pyyaml
pip install nuscenes-devkit
```

---

## Quick Start

### 1. Download Data

```bash
# Download nuScenes with map expansion
export NUSCENES_TOKEN=your_token_here
bash scripts/download_data.sh --dataset nuscenes --output_dir /data/nuscenes

# For mini dataset (quick testing)
bash scripts/download_data.sh --dataset nuscenes --output_dir /data/nuscenes --version v1.0-mini
```

### 2. Prepare Data

```bash
python scripts/prepare_data.py \
    --nuscenes_root /data/nuscenes \
    --output_dir data/processed \
    --version v1.0-trainval \
    --num_workers 8 \
    --num_points 20
```

### 3. Train

```bash
# MapTR v1 (24 epochs)
python pytorch/train.py \
    --data_root data/processed \
    --config configs/maptr_r50_nuscenes.yaml \
    --epochs 24 \
    --batch_size 4 \
    --lr 6e-4 \
    --num_gpus 8 \
    --work_dir work_dirs/maptr_r50

# MapTRv2 (110 epochs, with auxiliary losses)
python pytorch/train.py \
    --data_root data/processed \
    --config configs/maptr_v2_nuscenes.yaml \
    --epochs 110 \
    --batch_size 4 \
    --lr 6e-4 \
    --num_gpus 8 \
    --work_dir work_dirs/maptr_v2
```

### 4. Evaluate

```bash
python pytorch/evaluate.py \
    --checkpoint work_dirs/maptr_r50/epoch_24.pth \
    --data_root data/processed \
    --ann_file data/processed/maptr_val.pkl \
    --output results/eval_results.json
```

### 5. Inference & Visualization

```bash
python pytorch/inference.py \
    --checkpoint work_dirs/maptr_r50/epoch_24.pth \
    --data_root data/processed \
    --sample_idx 0 \
    --output_dir results/visualizations \
    --confidence_threshold 0.3
```

---

## Results

### nuScenes Val Set

| Method | Backbone | Epochs | Ped Crossing | Divider | Boundary | mAP |
|--------|----------|--------|:------------:|:-------:|:--------:|:---:|
| MapTR | ResNet-50 | 24 | 46.3 | 51.5 | 53.1 | 50.3 |
| MapTR | ResNet-50 | 110 | 50.3 | 55.1 | 57.2 | 54.2 |
| MapTRv2 | ResNet-50 | 24 | 52.1 | 57.8 | 59.3 | 56.4 |
| MapTRv2 | ResNet-50 | 110 | 55.7 | 60.2 | 62.1 | 59.3 |

*AP computed with Chamfer distance thresholds at 0.5m, 1.0m, and 1.5m.*

### Inference Speed

| Method | Backbone | Input Size | FPS (A100) |
|--------|----------|:----------:|:----------:|
| MapTR | ResNet-50 | 480x800 | 15.1 |
| MapTRv2 | ResNet-50 | 480x800 | 14.3 |

---

## Directory Structure

```
maptr/
├── configs/
│   ├── maptr_r50_nuscenes.yaml      # MapTR v1 configuration
│   └── maptr_v2_nuscenes.yaml       # MapTRv2 configuration
├── docs/
│   ├── research_summary.md          # Paper contributions & comparisons
│   ├── data_collection.md           # Dataset information
│   ├── annotation_guide.md          # Vectorized map annotation format
│   ├── model_architecture.md        # Detailed architecture description
│   ├── training_guide.md            # Training procedures & losses
│   └── evaluation_guide.md          # Evaluation metrics & protocol
├── pytorch/
│   ├── model.py                     # Main MapTR/MapTRv2 model
│   ├── backbone.py                  # ResNet-50 + FPN
│   ├── gkt.py                       # Geometry-guided Kernel Transformer
│   ├── map_decoder.py               # Transformer decoder with hierarchical queries
│   ├── heads.py                     # Classification + point regression heads
│   ├── losses.py                    # Hierarchical matching + losses
│   ├── dataset.py                   # nuScenes map dataset loader
│   ├── train.py                     # Training script (DDP, AMP)
│   ├── evaluate.py                  # Chamfer AP evaluation
│   └── inference.py                 # Inference + visualization
├── tensorflow/
│   ├── model.py                     # TF2/Keras MapTR implementation
│   ├── train.py                     # TF2 training script
│   ├── evaluate.py                  # TF2 evaluation
│   └── inference.py                 # TF2 inference + visualization
├── scripts/
│   ├── download_data.sh             # Data download script
│   ├── prepare_data.py              # Data preparation (vectorized GT extraction)
│   └── visualize_results.py         # Prediction vs GT visualization
├── tests/
│   └── test_model.py                # Comprehensive model tests
└── README.md                        # This file
```

---

## Key Design Decisions

### Permutation Equivalence
Map elements like lane dividers can be described equally well starting from either end. MapTR handles this by evaluating all valid permutations (cyclic shifts + reversals) during point-level matching and using the minimum-cost assignment.

### Hierarchical Queries
Instead of a flat set of queries, MapTR uses a structured decomposition:
- **Instance queries** (N_q): Each represents one map element
- **Point queries** (N_p): Each represents a vertex position within an element
- Combined via broadcast addition: query[i,j] = instance[i] + point[j]

### GKT (Geometry-guided Kernel Transformer)
Projects BEV grid locations back onto camera images using known camera geometry, then aggregates features using attention with learned deformable offsets around the projected locations.

---

## Citation

```bibtex
@inproceedings{liao2023maptr,
  title={MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction},
  author={Liao, Bencheng and Chen, Shaoyu and Wang, Xinggang and Cheng, Tianheng and Zhang, Qian and Liu, Wenyu and Huang, Chang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}

@article{liao2023maptrv2,
  title={MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction},
  author={Liao, Bencheng and Chen, Shaoyu and Zhang, Yunchi and Jiang, Bo and Zhang, Qian and Liu, Wenyu and Huang, Chang and Wang, Xinggang},
  journal={arXiv preprint arXiv:2308.05736},
  year={2023}
}
```

---

## License

This implementation is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The nuScenes dataset is subject to its own [Terms of Use](https://www.nuscenes.org/terms-of-use).
