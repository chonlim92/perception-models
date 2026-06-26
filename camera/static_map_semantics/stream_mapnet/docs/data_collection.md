# StreamMapNet: Data Collection Guide

## Overview

StreamMapNet is evaluated on two primary datasets for online HD map construction:
1. **nuScenes** (with map expansion pack) - primary benchmark
2. **Argoverse 2** - secondary benchmark for generalization

Both datasets provide surround-view camera images with corresponding ego-pose information and vectorized HD map annotations.

---

## nuScenes Dataset

### Dataset Versions

| Version | Scenes | Frames | Size | Purpose |
|---------|--------|--------|------|---------|
| v1.0-trainval | 850 | ~400K | ~350 GB | Full training and validation |
| v1.0-mini | 10 | ~4K | ~4 GB | Quick prototyping and debugging |
| v1.0-test | 150 | ~70K | ~60 GB | Official test submission |

### Required Components

#### 1. Core Dataset (v1.0-trainval)

The base nuScenes dataset provides:
- **6 surround-view cameras:** FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT
- **Camera calibration:** Intrinsic and extrinsic parameters for all cameras
- **Ego-pose:** 6-DoF vehicle pose at each timestamp (from IMU/GPS fusion)
- **Timestamps:** Synchronized timestamps across all sensors (2 Hz keyframes)

#### 2. Map Expansion Pack (CRITICAL)

The map expansion pack is **required** for vectorized HD map ground truth. It provides:
- Vectorized map layers for 4 locations: Boston Seaport, Singapore OneNorth, Singapore Queenstown, Singapore Holland Village
- Map elements: lane dividers, road boundaries, pedestrian crossings, road segments, walkways, stop lines, carpark areas
- Format: JSON-based vector format accessible via `NuScenesMap` API

Without the map expansion, only rasterized semantic maps from the base dataset are available, which are insufficient for vectorized map training.

### Download Instructions

#### Prerequisites

```bash
# Install nuscenes-devkit
pip install nuscenes-devkit==1.1.11

# Alternative: install from source for latest features
pip install git+https://github.com/nutonomy/nuscenes-devkit.git
```

#### Step 1: Register and Download

1. Create an account at https://www.nuscenes.org/
2. Accept the Terms of Use for the nuScenes dataset
3. Navigate to the Download page

#### Step 2: Download Core Dataset

```bash
# Option A: Using the official download script (recommended)
# Download metadata + all camera data (skip LiDAR/RADAR if not needed)
mkdir -p /data/nuscenes
cd /data/nuscenes

# Download these tarballs:
# - v1.0-trainval_meta.tgz (metadata, ~300 MB)
# - v1.0-trainval01_blobs.tgz through v1.0-trainval10_blobs.tgz (camera images)

# Extract all:
for f in v1.0-trainval*.tgz; do
    tar -xzf "$f"
done
```

#### Step 3: Download Map Expansion

```bash
# Download nuScenes-map-expansion-v1.3.zip from the download page
# This contains vectorized map annotations for all 4 locations

wget https://www.nuscenes.org/data/nuScenes-map-expansion-v1.3.zip
unzip nuScenes-map-expansion-v1.3.zip -d /data/nuscenes/

# The maps directory should be placed at the same level as other nuScenes directories
```

#### Step 4: Verify Installation

```python
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

# Verify core dataset
nusc = NuScenes(version='v1.0-trainval', dataroot='/data/nuscenes', verbose=True)
print(f"Number of scenes: {len(nusc.scene)}")
print(f"Number of samples: {len(nusc.sample)}")

# Verify map expansion
nusc_map = NuScenesMap(dataroot='/data/nuscenes', map_name='singapore-onenorth')
print(f"Map layers: {nusc_map.non_geometric_layers}")
print(f"Number of road segments: {len(nusc_map.road_segment)}")
```

### Directory Structure After Setup

```
/data/nuscenes/
├── maps/
│   ├── basemap/                          # Rasterized basemaps (PNG)
│   ├── expansion/                        # Vectorized map expansion
│   │   ├── boston-seaport.json           # ~50 MB per map
│   │   ├── singapore-hollandvillage.json
│   │   ├── singapore-onenorth.json
│   │   └── singapore-queenstown.json
│   ├── prediction/                       # (optional, for prediction tasks)
│   └── 36092f0b03a857c6a3403e25b4b7aab3.png  # Rasterized map images
├── samples/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   └── CAM_FRONT_RIGHT/
├── sweeps/                               # Inter-keyframe images (optional)
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   └── CAM_FRONT_RIGHT/
├── v1.0-trainval/
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── scene.json
│   ├── sensor.json
│   └── visibility.json
└── v1.0-mini/                            # (if downloaded)
    └── ... (same structure as v1.0-trainval)
```

---

## Argoverse 2 Dataset

### Overview

Argoverse 2 provides a complementary evaluation setting with:
- **7 ring cameras** (surround-view) + 2 stereo cameras
- Higher resolution images (2048x1550)
- Larger perception range
- Different geographic locations (6 US cities: Austin, Detroit, Miami, Palo Alto, Pittsburgh, Washington D.C.)

### Required Components

| Component | Scenes | Size | Description |
|-----------|--------|------|-------------|
| Sensor dataset (train) | 700 | ~1 TB | Full sensor suite |
| Sensor dataset (val) | 150 | ~200 GB | Validation split |
| Map data | - | ~2 GB | Vectorized HD maps per log |

### Download Instructions

```bash
# Install Argoverse 2 API
pip install av2==0.2.1

# Install s5cmd for fast parallel downloads
# Linux:
wget https://github.com/peak/s5cmd/releases/download/v2.1.0/s5cmd_2.1.0_Linux-64bit.tar.gz
tar -xzf s5cmd_2.1.0_Linux-64bit.tar.gz

# Download sensor dataset (camera data + maps)
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/train/*" /data/argoverse2/sensor/train/
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/*" /data/argoverse2/sensor/val/

# Or selectively download only camera + map data:
# Each log directory contains:
#   sensors/cameras/ring_*/*.jpg
#   map/
#   city_SE3_egovehicle.feather
#   calibration/
```

### Directory Structure After Setup

```
/data/argoverse2/
├── sensor/
│   ├── train/
│   │   ├── 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a/
│   │   │   ├── calibration/
│   │   │   │   ├── egovehicle_SE3_sensor.feather
│   │   │   │   └── intrinsics.feather
│   │   │   ├── city_SE3_egovehicle.feather    # Ego-pose trajectory
│   │   │   ├── map/
│   │   │   │   ├── log_map_archive.json       # Vector map for this log
│   │   │   │   ├── nearby_centerlines.feather
│   │   │   │   └── ...
│   │   │   └── sensors/
│   │   │       └── cameras/
│   │   │           ├── ring_front_center/
│   │   │           ├── ring_front_left/
│   │   │           ├── ring_front_right/
│   │   │           ├── ring_rear_left/
│   │   │           ├── ring_rear_right/
│   │   │           ├── ring_side_left/
│   │   │           └── ring_side_right/
│   │   └── .../
│   └── val/
│       └── ... (same structure)
└── ...
```

### Verify Argoverse 2 Setup

```python
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path

# Load a log
data_root = Path("/data/argoverse2/sensor/val")
log_id = "00a6ffc1-6ce9-3bc3-a060-6006e9893a1a"

loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)
log_map = ArgoverseStaticMap.from_map_dir(data_root / log_id / "map", build_raster=False)

# Access map elements
print(f"Lane segments: {len(log_map.vector_lane_segments)}")
print(f"Pedestrian crossings: {len(log_map.vector_pedestrian_crossings)}")
```

---

## Storage Requirements

### Minimum Requirements (nuScenes only, camera data)

| Component | Size | Notes |
|-----------|------|-------|
| v1.0-mini (metadata + cameras) | ~4 GB | For debugging only |
| v1.0-trainval metadata | ~300 MB | JSON annotation files |
| v1.0-trainval camera keyframes | ~80 GB | 6 cameras, ~34K keyframes |
| v1.0-trainval camera sweeps | ~250 GB | Inter-keyframe images (optional) |
| Map expansion pack | ~200 MB | Vectorized map JSONs |
| **Total (keyframes only)** | **~80 GB** | Minimum for training |
| **Total (with sweeps)** | **~350 GB** | Full dataset |

### Full Setup (both datasets)

| Component | Size |
|-----------|------|
| nuScenes v1.0-trainval (cameras + maps) | ~80-350 GB |
| Argoverse 2 sensor (cameras + maps) | ~1.2 TB |
| Generated GT cache (nuScenes) | ~5 GB |
| Generated GT cache (Argoverse 2) | ~10 GB |
| **Recommended free space** | **~1.6 TB** |

### Disk I/O Considerations

- Training reads images randomly from sequences; SSD is strongly recommended
- Map expansion JSON files are loaded entirely into memory during GT generation (~2 GB RAM per map)
- Consider creating symlinks if datasets are on separate drives:
  ```bash
  ln -s /ssd1/nuscenes /data/nuscenes
  ln -s /ssd2/argoverse2 /data/argoverse2
  ```

---

## Data Preprocessing

### Generate Ground Truth Cache

StreamMapNet requires pre-generated ground truth map annotations in ego-vehicle coordinates. This is done once before training:

```bash
# Generate nuScenes GT annotations
python tools/create_data.py nuscenes \
    --root-path /data/nuscenes \
    --out-dir /data/nuscenes/stream_mapnet_gt \
    --extra-tag nuscenes \
    --version v1.0-trainval

# Generate Argoverse 2 GT annotations
python tools/create_data.py argoverse2 \
    --root-path /data/argoverse2/sensor \
    --out-dir /data/argoverse2/stream_mapnet_gt \
    --extra-tag argoverse2
```

This preprocessing step:
1. Loads vectorized map elements from the map expansion
2. Transforms them to ego-vehicle coordinates using ego-pose
3. Clips to the perception range (e.g., [-30m, 30m] x [-15m, 15m])
4. Samples fixed number of points per polyline/polygon
5. Saves as pickle files for efficient loading during training

### Temporal Sequence Index

StreamMapNet also requires a sequence index file that defines which frames form temporal sequences:

```bash
# Generate sequence indices for temporal training
python tools/create_sequence_index.py \
    --root-path /data/nuscenes \
    --version v1.0-trainval \
    --sequence-length 8 \
    --output /data/nuscenes/stream_mapnet_gt/sequence_index.pkl
```

---

## Common Issues and Troubleshooting

### Issue: Missing map expansion

```
FileNotFoundError: Map file not found: /data/nuscenes/maps/expansion/boston-seaport.json
```
**Solution:** Download and extract the map expansion pack to `/data/nuscenes/maps/expansion/`.

### Issue: Ego-pose discontinuities

Some nuScenes scenes have ego-pose jumps at scene boundaries. StreamMapNet handles this by resetting the hidden state at scene boundaries during training.

### Issue: Insufficient disk space

For limited storage, consider:
1. Using v1.0-mini for development (4 GB)
2. Downloading only keyframe images (skip sweeps)
3. Processing GT annotations incrementally per scene

### Issue: Slow data loading

- Pre-extract images to uncompressed format for faster I/O
- Use `num_workers > 0` in DataLoader
- Place frequently accessed metadata on SSD
- Consider LMDB or memory-mapped files for GT annotations
