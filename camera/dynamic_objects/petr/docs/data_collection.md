# Data Collection - nuScenes Dataset Setup

## Overview

PETR, PETRv2, and StreamPETR are trained and evaluated on the nuScenes dataset, a large-scale autonomous driving dataset collected in Boston and Singapore. This document covers dataset acquisition, directory structure, annotation format, and preparation steps specific to the PETR family.

---

## nuScenes Dataset Specifications

### Sensor Configuration

| Sensor | Count | Details |
|--------|-------|---------|
| Camera | 6 | Full 360-degree surround coverage |
| LiDAR | 1 | 32-beam Velodyne HDL-32E (for ground truth generation) |
| RADAR | 5 | Continental ARS 408-21 (not used in camera-only methods) |
| IMU/GPS | 1 | For ego-motion and localization |

### Camera Configuration

The 6 cameras provide complete 360-degree horizontal coverage:

| Camera Name | Field of View | Resolution | Position |
|-------------|--------------|------------|----------|
| CAM_FRONT | 70 degrees | 1600 x 900 | Front center |
| CAM_FRONT_LEFT | 70 degrees | 1600 x 900 | Front left 55 degrees |
| CAM_FRONT_RIGHT | 70 degrees | 1600 x 900 | Front right 55 degrees |
| CAM_BACK | 110 degrees | 1600 x 900 | Rear center |
| CAM_BACK_LEFT | 70 degrees | 1600 x 900 | Rear left 55 degrees |
| CAM_BACK_RIGHT | 70 degrees | 1600 x 900 | Rear right 55 degrees |

### Dataset Statistics

| Split | Scenes | Keyframes | Sweeps | Annotations |
|-------|--------|-----------|--------|-------------|
| Train | 700 | 28,130 | ~280K | ~1.4M 3D boxes |
| Val | 150 | 6,019 | ~60K | ~300K 3D boxes |
| Test | 150 | 6,008 | ~60K | Hidden |
| **Total** | **1000** | **40,157** | ~400K | ~1.7M+ |

### Temporal Structure

- **Keyframe rate**: 2 Hz (every 0.5 seconds)
- **Sweep rate**: 12 Hz (camera) / 20 Hz (LiDAR)
- **Scene duration**: ~20 seconds each
- **Frames per scene**: ~40 keyframes, ~240 sweeps

For StreamPETR's temporal modeling, keyframe sequences within scenes provide the temporal supervision signal.

---

## Data Download Instructions

### Prerequisites

1. Register at [nuScenes website](https://www.nuscenes.org/)
2. Accept the dataset license agreement
3. Ensure sufficient storage: ~400 GB for full dataset

### Download Options

#### Option 1: Full Dataset (Recommended)

```bash
# Install nuScenes devkit
pip install nuscenes-devkit

# Download via the website or using the following structure:
# Visit https://www.nuscenes.org/nuscenes#download
# Download:
#   - Full dataset (v1.0): metadata + sweeps + maps
#   - Can skip: LiDAR sweeps if only training camera models (saves ~200GB)
```

#### Option 2: Mini Dataset (for development/debugging)

```bash
# Download the mini split (10 scenes, ~4 GB)
# Useful for verifying data pipeline before full training
# https://www.nuscenes.org/nuscenes#download -> v1.0-mini
```

### Download Checklist

- [ ] Metadata (v1.0-trainval): `v1.0-trainval_meta.tgz` (~1.4 GB)
- [ ] Camera blobs (keyframes): `v1.0-trainval01_blobs.tgz` through `v1.0-trainval10_blobs.tgz` (~120 GB total)
- [ ] Camera sweeps (optional, for temporal): individual sweep packages
- [ ] Maps: `nuScenes-map-expansion-v1.3.zip` (~700 MB)
- [ ] CAN bus data (for ego-motion): `can_bus.zip` (~2 GB) **Required for PETRv2/StreamPETR**

---

## Directory Structure

After extraction, organize the dataset as follows:

```
data/nuscenes/
├── maps/                          # HD maps
│   ├── basemap/
│   ├── expansion/
│   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
│   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
│   ├── 53992ee3023e5494b90c316c183be829.png
│   └── 93406b464a165eaba6d9de76c24571eb.png
├── samples/                       # Keyframe data (annotated at 2 Hz)
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── LIDAR_TOP/                 # LiDAR keyframes (for GT generation)
│   ├── RADAR_BACK_LEFT/
│   ├── RADAR_BACK_RIGHT/
│   ├── RADAR_FRONT/
│   ├── RADAR_FRONT_LEFT/
│   └── RADAR_FRONT_RIGHT/
├── sweeps/                        # Inter-keyframe data (12 Hz camera)
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   └── LIDAR_TOP/
├── v1.0-trainval/                 # Metadata JSON files
│   ├── attribute.json
│   ├── calibrated_sensor.json     # Camera intrinsics & extrinsics
│   ├── category.json              # Object categories
│   ├── ego_pose.json              # Ego vehicle poses (for temporal)
│   ├── instance.json              # Object instance tracking
│   ├── log.json
│   ├── map.json
│   ├── sample.json                # Keyframe references
│   ├── sample_annotation.json     # 3D bounding box annotations
│   ├── sample_data.json           # Sensor data references
│   ├── scene.json                 # Scene metadata
│   ├── sensor.json
│   └── visibility.json            # Visibility levels
├── v1.0-test/                     # Test split metadata (no annotations)
│   └── ... (same structure as trainval)
└── can_bus/                       # CAN bus data for ego-motion
    ├── scene-0001.json
    ├── scene-0002.json
    └── ...
```

---

## Data Preparation

### Step 1: Generate Info Files

PETR uses pre-processed pickle files containing all necessary metadata for efficient data loading. Generate them using MMDetection3D tools:

```bash
# Generate standard info files
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes

# This creates:
# - nuscenes_infos_train.pkl
# - nuscenes_infos_val.pkl
```

### Step 2: Generate Temporal Info Files (Required for PETRv2/StreamPETR)

```bash
# Generate info files with temporal frame links
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-trainval \
    --with-temporal

# This creates:
# - nuscenes_infos_temporal_train.pkl
# - nuscenes_infos_temporal_val.pkl
```

### Step 3: Verify Dataset

```python
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)

# Verify scene count
assert len(nusc.scene) == 850  # 700 train + 150 val

# Verify sample count
assert len(nusc.sample) == 34149  # Total keyframes

# Verify camera data availability
sample = nusc.sample[0]
for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
    assert cam in sample['data']
```

---

## Info File Format

The pickle files contain a list of dictionaries, one per keyframe sample:

```python
info = {
    'lidar_path': str,              # Path to LiDAR point cloud
    'token': str,                    # Unique sample token
    'timestamp': int,                # Unix timestamp (microseconds)
    'sweeps': list,                  # List of inter-keyframe sweeps

    # Camera data for all 6 cameras
    'cams': {
        'CAM_FRONT': {
            'data_path': str,        # Image file path
            'type': str,             # Camera name
            'sensor2lidar_rotation': np.array,  # 3x3 rotation matrix
            'sensor2lidar_translation': np.array,  # 3-d translation vector
            'sensor2ego_rotation': np.array,
            'sensor2ego_translation': np.array,
            'ego2global_rotation': np.array,
            'ego2global_translation': np.array,
            'cam_intrinsic': np.array,  # 3x3 intrinsic matrix
            'timestamp': int,
        },
        # ... same for other 5 cameras
    },

    # Ego pose
    'ego2global_rotation': np.array,    # 3x3 rotation
    'ego2global_translation': np.array, # 3-d translation

    # Ground truth annotations
    'gt_boxes': np.array,            # (N, 9): cx,cy,cz,w,l,h,yaw,vx,vy
    'gt_names': list,                # List of class names
    'gt_velocity': np.array,         # (N, 2): vx, vy
    'num_lidar_pts': np.array,       # Number of LiDAR points per box
    'num_radar_pts': np.array,       # Number of RADAR points per box

    # Temporal info (for PETRv2/StreamPETR)
    'prev': str or None,             # Token of previous keyframe
    'next': str or None,             # Token of next keyframe
    'scene_token': str,              # Scene this sample belongs to
}
```

---

## Temporal Sequence Requirements for StreamPETR

StreamPETR requires sequential frame access for query propagation during both training and inference.

### Training Requirements

1. **Sequential Loading**: Frames must be loaded in temporal order within a scene
2. **Ego-Motion Data**: Transformation matrices between consecutive frames are essential for:
   - Compensating ego-motion in propagated query positions
   - Motion-aware layer normalization
3. **Instance Tracking**: Ground truth instance IDs enable temporal loss computation

### Key Temporal Fields

```python
# Additional fields in temporal info files
temporal_info = {
    # Previous frame reference
    'prev_info': {
        'token': str,
        'ego2global_rotation': np.array,
        'ego2global_translation': np.array,
        'timestamp': int,
    },

    # Ego-motion from previous to current frame
    'ego_motion': np.array,  # 4x4 transformation matrix

    # Instance tracking
    'instance_tokens': list,  # Per-box instance identifiers (same ID across frames)
}
```

### Handling Sequence Boundaries

- First frame of a scene: no previous frame available, queries are randomly initialized
- Large time gaps (>2 seconds): treat as sequence boundary, reset temporal state
- Dropped frames: use ego-motion interpolation to fill gaps

---

## Storage Requirements

| Component | Size | Required for |
|-----------|------|-------------|
| Camera keyframes | ~120 GB | All models |
| Camera sweeps | ~60 GB | Optional (higher-rate data) |
| LiDAR keyframes | ~150 GB | GT generation only |
| Metadata + annotations | ~2 GB | All models |
| CAN bus data | ~2 GB | PETRv2, StreamPETR |
| Processed info files | ~1 GB | All models |
| **Total (camera-only)** | **~185 GB** | PETR/PETRv2/StreamPETR |
| **Total (full dataset)** | **~400 GB** | If LiDAR/RADAR needed |

---

## Common Issues and Solutions

### Issue: Missing CAN bus data

**Symptom**: Ego-motion matrices are all identity, temporal fusion produces no improvement.

**Solution**: Download `can_bus.zip` separately from nuScenes downloads page and extract to `data/nuscenes/can_bus/`.

### Issue: Mismatched calibration data

**Symptom**: 3D Position Embeddings project to wrong image locations.

**Solution**: Verify `calibrated_sensor.json` is from the correct dataset version. Check that camera intrinsics match the actual image resolution.

### Issue: Temporal frame linking fails

**Symptom**: Training crashes with "prev frame not found" errors.

**Solution**: Regenerate info files with `--with-temporal` flag. Ensure all scenes have complete keyframe sequences.

### Issue: Insufficient disk I/O during training

**Symptom**: Data loading becomes the bottleneck, GPU utilization drops.

**Solution**: 
- Use SSD storage for the dataset
- Increase `workers_per_gpu` (4-8)
- Pre-cache image paths in RAM
- Consider symlinking frequently accessed data to fast storage
