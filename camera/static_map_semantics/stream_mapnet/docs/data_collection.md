# StreamMapNet: Data Collection Guide

A comprehensive guide for Staff AI Engineers who know PyTorch but are new to
autonomous driving datasets. This document explains what data StreamMapNet needs,
where it comes from, and how every piece fits together.

---

## 1. Overview: What Data StreamMapNet Needs and Why

StreamMapNet constructs vectorized HD maps from surround-view camera images in
real time. The model takes in multi-camera images and outputs polylines/polygons
representing static map elements (lane dividers, road boundaries, pedestrian
crossings) in bird's-eye view (BEV).

To train this model, you need:

| Data Component              | Purpose                                           |
|-----------------------------|---------------------------------------------------|
| 6 surround-view camera imgs | Visual input to the model                         |
| Camera intrinsics           | Project 3D points into image planes               |
| Camera extrinsics           | Relate each camera to the ego vehicle             |
| Ego-pose (per frame)        | Locate the vehicle in world coordinates           |
| Vectorized HD map           | Ground truth for supervised training              |
| Temporal frame sequences    | Enable the temporal fusion module (8 frames)      |

The model predicts 150 map elements (50 per class), each as a polyline of 20
ordered 2D points within the BEV range of [-30m, +30m] x [-15m, +15m].

---

## 2. nuScenes Dataset Explained from Scratch

### What is nuScenes?

nuScenes (pronounced "new-scenes") is a large-scale autonomous driving dataset
released by Motional (formerly nuTonomy) in 2019. It has become the de facto
benchmark for BEV perception research.

Key facts:
- 1000 driving scenes (850 train/val + 150 test)
- Collected in Boston (Seaport district) and Singapore (3 neighborhoods)
- Full multi-sensor suite: cameras, LiDAR, RADAR, IMU, GPS
- Diverse conditions: day, night, rain, construction zones
- Fully annotated 3D bounding boxes for 23 object classes
- Creative Commons BY-NC-SA 4.0 license (non-commercial research)

### Sensor Setup

The nuScenes ego vehicle carries:
- 6 surround-view cameras (1600x900, 12 Hz capture, 2 Hz keyframe annotation)
- 1 spinning LiDAR (Velodyne HDL-32E, 20 Hz)
- 5 RADAR sensors (Continental ARS 408-21, 13 Hz)
- IMU + GPS for ego-motion

### Camera Layout (ASCII Diagram)

```
                    FRONT (70 deg FOV)
                        |
           FRONT_LEFT   |   FRONT_RIGHT
          (70 deg)  \   |   /  (70 deg)
                     \  |  /
                      \ | /
        +--------------[EGO]--------------+
        |              VEHICLE            |
        +---------------------------------+
                      / | \
                     /  |  \
          (70 deg)  /   |   \  (70 deg)
           BACK_LEFT    |    BACK_RIGHT
                        |
                    BACK (110 deg FOV)
```

Coverage notes:
- FRONT, FRONT_LEFT, FRONT_RIGHT: each ~70-degree horizontal FOV
- BACK: wider ~110-degree FOV (fisheye-like)
- BACK_LEFT, BACK_RIGHT: each ~70-degree horizontal FOV
- Together they provide full 360-degree surround coverage with overlap

### Frame Rate and Timing

- Keyframes: 2 Hz (one annotated sample every 0.5 seconds)
- Sweeps: 12 Hz (unannotated intermediate frames between keyframes)
- Each keyframe has full annotation; sweeps have sensor data only
- StreamMapNet uses keyframes for training (GT available only at 2 Hz)
- Temporal window: 8 consecutive keyframes = 4 seconds of driving

### Scenes

- Each scene is a ~20-second driving clip (approximately 40 keyframes)
- Scenes are continuous segments extracted from longer driving logs
- 850 scenes for train/val, 150 for test
- Train/val split: 700 train + 150 val (official split)

### Why nuScenes is the Standard for BEV Perception

1. Full surround cameras with known calibration (needed for LSS/BEV lift)
2. Synchronized multi-sensor data with precise timestamps
3. Rich map annotations via the map expansion pack
4. Established leaderboard with reproducible baselines
5. Active community and well-maintained Python devkit

---

## 3. nuScenes Map Expansion: What It Adds

### Base nuScenes vs. Map Expansion

The BASE nuScenes dataset provides:
- 3D bounding box annotations for dynamic objects (cars, pedestrians, etc.)
- Ego-pose trajectory
- Camera/LiDAR/RADAR data
- Rasterized semantic maps (low-resolution PNGs -- not useful for vectorized models)

The MAP EXPANSION (v1.3) adds:
- High-definition vectorized map layers for all 4 collection locations
- Geometric primitives: polylines and polygons in global coordinates
- Queryable via the NuScenesMap Python API
- This is what provides ground truth for StreamMapNet

### Four Mapped Locations

| Location                 | City      | Character                        |
|--------------------------|-----------|----------------------------------|
| boston-seaport            | Boston    | Wide roads, grid layout, port    |
| singapore-onenorth       | Singapore | Research park, curves, junctions |
| singapore-queenstown     | Singapore | Residential, dense intersections |
| singapore-hollandvillage | Singapore | Mixed-use, narrow roads          |

Each location has one JSON file (~50 MB) containing all map elements.

### Map Layers Available

| Layer          | Geometry | Description                               |
|----------------|----------|-------------------------------------------|
| lane_divider   | Polyline | Lines separating adjacent lanes           |
| road_segment   | Polygon  | Drivable road surface areas               |
| road_boundary  | Polyline | Physical edges of the road (curbs, walls) |
| ped_crossing   | Polygon  | Pedestrian crosswalk areas                |
| walkway        | Polygon  | Sidewalks and footpaths                   |
| stop_line      | Polyline | Stop lines at intersections               |
| carpark_area   | Polygon  | Parking lot regions                       |

StreamMapNet uses 3 of these for its primary benchmark:
- lane_divider (polylines)
- road_boundary (derived from road_segment polygon exteriors)
- ped_crossing (derived from polygon exteriors)

### Map Format

Each map JSON contains records like:

```json
{
  "lane_divider": [
    {
      "token": "abc123...",
      "line": {"type": "LineString", "coordinates": [[x1,y1], [x2,y2], ...]},
      "road_segment_token": "def456..."
    }
  ],
  "ped_crossing": [
    {
      "token": "ghi789...",
      "polygon": {"type": "Polygon", "coordinates": [[[x1,y1], [x2,y2], ...]]}
    }
  ]
}
```

Coordinates are in a global frame (meters, fixed per city).

### NuScenesMap API

```python
from nuscenes.map_expansion.map_api import NuScenesMap

nusc_map = NuScenesMap(dataroot='/data/nuscenes', map_name='boston-seaport')

# Query all lane dividers within a rectangular patch
patch = (x_min, y_min, x_max, y_max)  # global coords in meters
records = nusc_map.get_records_in_patch(patch, ['lane_divider', 'ped_crossing'])

# Get the geometry of a specific lane divider
line_token = records['lane_divider'][0]
line = nusc_map.get(line_token)  # returns dict with 'line' key
coords = nusc_map.discretize_lanes([line_token], resolution_meters=0.5)
```

---

## 4. How Vectorized Ground Truth is Generated

### Step-by-Step Process

The script `scripts/prepare_map_data.py` performs these steps for every keyframe:

```
For each sample in the dataset:
  1. Get ego-pose (translation [x, y, z] + rotation quaternion)
  2. Define query patch centered on ego: [ego_x +/- 30m, ego_y +/- 15m]
  3. Query NuScenesMap API for all elements within that patch
  4. Transform each element from global coordinates to ego-vehicle frame
  5. Clip polylines to the BEV perception range
  6. Resample each polyline to exactly K=20 evenly-spaced points
  7. Save per-sample result to a pickle file
```

### Transformation Math: Global to Ego

The ego-pose provides a 4x4 transformation matrix T_ego_to_global:

```
T_ego_to_global = [R | t]    (3x3 rotation, 3x1 translation)
                  [0 | 1]

To go from global to ego:
  T_global_to_ego = inverse(T_ego_to_global)

For a point p_global = [x, y, z, 1]^T:
  p_ego = T_global_to_ego @ p_global
```

In practice (2D map elements, z=0):

```python
import numpy as np
from pyquaternion import Quaternion

# ego_pose has 'translation' and 'rotation' (quaternion)
translation = np.array(ego_pose['translation'])  # [x, y, z]
rotation = Quaternion(ego_pose['rotation'])       # [w, x, y, z]

# Build rotation matrix (3x3)
R = rotation.rotation_matrix

# Transform global points to ego frame
# points_global: shape (N, 2) -- map element coordinates
points_3d = np.hstack([points_global, np.zeros((N, 1))])  # add z=0
points_centered = points_3d - translation  # translate
points_ego = (R.T @ points_centered.T).T   # rotate (inverse = transpose for rotation)

# Keep only x, y for BEV
points_ego_2d = points_ego[:, :2]
```

### Polyline Resampling

Map elements have variable numbers of vertices. The model expects fixed-size
tensors, so we resample every polyline to exactly 20 points:

```python
def resample_polyline(points, num_samples=20):
    """Resample a polyline to num_samples evenly-spaced points along arc length."""
    # Compute cumulative arc length
    diffs = np.diff(points, axis=0)                    # (N-1, 2)
    segment_lengths = np.linalg.norm(diffs, axis=1)    # (N-1,)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length < 1e-6:
        return np.repeat(points[:1], num_samples, axis=0)

    # Generate evenly-spaced sample distances
    sample_distances = np.linspace(0, total_length, num_samples)

    # Interpolate x and y independently along arc length
    resampled_x = np.interp(sample_distances, cumulative_length, points[:, 0])
    resampled_y = np.interp(sample_distances, cumulative_length, points[:, 1])

    return np.stack([resampled_x, resampled_y], axis=1)  # (20, 2)
```

This ensures the model always outputs a fixed (20, 2) tensor per element.

### Why We Clip to BEV Range

The model only predicts map elements within its perception range (60m x 30m
centered on the ego vehicle). Elements outside this range:
- Cannot be seen by the cameras (too far or occluded)
- Would waste model capacity on unobservable predictions
- Must be clipped to the range boundary, not simply discarded

Clipping preserves the portion of a long polyline that falls within range.

### Minimum Polyline Length Filter

After clipping, polylines shorter than 1.0 meter are discarded. These are
typically artifacts of clipping at the BEV boundary or degenerate map geometries.

---

## 5. Coordinate Systems (Critical)

Understanding coordinate frames is essential. Bugs here are silent and fatal.

### Global Frame (World Coordinates)

- Fixed coordinate system for each city map
- Origin: arbitrary reference point in the mapped area
- Axes: roughly aligned with cardinal directions
- Units: meters
- Used by: NuScenesMap API, ego-pose translations

### Ego-Vehicle Frame

- Origin: center of the rear axle (projected to ground)
- X-axis: points FORWARD (longitudinal, direction of travel)
- Y-axis: points LEFT (lateral)
- Z-axis: points UP
- Units: meters
- Changes every frame as the vehicle moves

### BEV Grid Frame

- 2D discretization of the ego-vehicle frame (top-down view)
- X range: [-30.0m, +30.0m] = 60m total (forward/backward)
- Y range: [-15.0m, +15.0m] = 30m total (left/right)
- Resolution: 0.3 meters per pixel (cell)
- Grid size: 200 pixels (x-axis) x 100 pixels (y-axis)
- Origin in grid: pixel (100, 50) corresponds to ego position (0, 0)

### Normalized Frame (Model Output)

- The model outputs polyline coordinates in [0, 1] range
- Denormalization maps back to ego-vehicle meters:
  - x_meters = x_norm * 60.0 - 30.0   (maps [0,1] to [-30, +30])
  - y_meters = y_norm * 30.0 - 15.0   (maps [0,1] to [-15, +15])
- This normalization helps the transformer decoder converge faster

### ASCII Diagram: All Coordinate Frames

```
GLOBAL FRAME (fixed per city)            EGO-VEHICLE FRAME (moves with car)
 North ^                                        ^ X (forward)
       |                                        |
       |                                        |
       +-------> East                  Y <------+------
                                    (left)      |     (right)
                                                v

BEV GRID FRAME (discretized ego)         NORMALIZED FRAME (model output)
  col 0          col 99                     (0,0)-----------(1,0)
   +------ ... ------+  row 0                |               |
   |                 |                        |               |
   |    [ego at      |                        |   [ego at     |
   |   row 100,      |                        |  (0.5, 0.5)]  |
   |   col 50]       |                        |               |
   |                 |                        |               |
   +------ ... ------+  row 199             (0,1)-----------(1,1)
```

### Conversion Functions

```python
def ego_to_normalized(points_ego, x_range=(-30, 30), y_range=(-15, 15)):
    """Convert ego-frame meters to [0, 1] normalized coordinates."""
    x_norm = (points_ego[:, 0] - x_range[0]) / (x_range[1] - x_range[0])
    y_norm = (points_ego[:, 1] - y_range[0]) / (y_range[1] - y_range[0])
    return np.stack([x_norm, y_norm], axis=1)

def normalized_to_ego(points_norm, x_range=(-30, 30), y_range=(-15, 15)):
    """Convert [0, 1] normalized coordinates back to ego-frame meters."""
    x_ego = points_norm[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    y_ego = points_norm[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    return np.stack([x_ego, y_ego], axis=1)

def ego_to_grid(points_ego, x_range=(-30, 30), y_range=(-15, 15),
                grid_size=(200, 100)):
    """Convert ego-frame meters to BEV grid pixel indices."""
    x_norm = (points_ego[:, 0] - x_range[0]) / (x_range[1] - x_range[0])
    y_norm = (points_ego[:, 1] - y_range[0]) / (y_range[1] - y_range[0])
    col = (y_norm * grid_size[1]).astype(int)
    row = (x_norm * grid_size[0]).astype(int)
    return np.stack([row, col], axis=1)
```

---

## 6. Data Pipeline: Raw nuScenes to Training-Ready Format

### Step 1: Download nuScenes (Cameras + Metadata + Map Expansion)

```bash
mkdir -p /data/nuscenes && cd /data/nuscenes

# Download from https://www.nuscenes.org/download (requires account)
# Required files:
#   v1.0-trainval_meta.tgz           (~300 MB)  -- metadata JSONs
#   v1.0-trainval01_blobs.tgz        (~15 GB each, 10 parts) -- sensor data
#   nuScenes-map-expansion-v1.3.zip  (~200 MB)  -- vectorized maps

# Extract metadata
tar -xzf v1.0-trainval_meta.tgz

# Extract camera blobs (can skip LiDAR/RADAR blobs to save space)
for i in $(seq -w 1 10); do
    tar -xzf "v1.0-trainval${i}_blobs.tgz"
done

# Extract map expansion
unzip nuScenes-map-expansion-v1.3.zip -d /data/nuscenes/
```

Directory after Step 1:
```
/data/nuscenes/
├── maps/expansion/          <-- 4 city JSON files
├── samples/CAM_*/           <-- keyframe images (6 camera dirs)
├── sweeps/CAM_*/            <-- inter-keyframe images (optional)
└── v1.0-trainval/           <-- 12 metadata JSON files
```

### Step 2: Generate Ground Truth Pickles

```bash
cd /path/to/stream_mapnet

python scripts/prepare_map_data.py \
    --dataroot /data/nuscenes \
    --version v1.0-trainval \
    --output-dir /data/nuscenes/stream_mapnet_gt \
    --map-classes lane_divider road_boundary ped_crossing \
    --num-points 20 \
    --bev-range -30 30 -15 15
```

This produces:
```
/data/nuscenes/stream_mapnet_gt/
├── train_map_gt.pkl          <-- all training samples
├── val_map_gt.pkl            <-- all validation samples
└── per_sample/               <-- individual sample pickles (optional)
    ├── sample_000000.pkl
    ├── sample_000001.pkl
    └── ...
```

Each pickle entry contains:
```python
{
    'token': 'abc123...',                    # nuScenes sample token
    'ego_pose': np.array(4x4),              # ego-to-global transform
    'map_elements': {
        'lane_divider': [np.array(20,2), ...],   # list of polylines
        'road_boundary': [np.array(20,2), ...],
        'ped_crossing': [np.array(20,2), ...],
    },
    'camera_info': {
        'CAM_FRONT': {'intrinsic': ..., 'extrinsic': ..., 'img_path': ...},
        ...  # 6 cameras total
    }
}
```

### Step 3: Generate Temporal Sequence Index

```bash
python scripts/prepare_map_data.py \
    --dataroot /data/nuscenes \
    --version v1.0-trainval \
    --output-dir /data/nuscenes/stream_mapnet_gt \
    --generate-sequence-index \
    --sequence-length 8
```

This creates `sequence_index.pkl` mapping each sample to its temporal neighbors:
```python
# sequence_index[sample_token] = [token_t-7, token_t-6, ..., token_t-1, token_t]
# If fewer than 8 prior frames exist (scene start), earlier slots are None
```

### Step 4: Verify Data Integrity

```python
import pickle
import numpy as np

with open('/data/nuscenes/stream_mapnet_gt/train_map_gt.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Total training samples: {len(data['samples'])}")

# Check a sample
sample = data['samples'][0]
for cls_name, elements in sample['map_elements'].items():
    print(f"  {cls_name}: {len(elements)} elements")
    if elements:
        print(f"    Shape: {elements[0].shape}")  # should be (20, 2)
        print(f"    X range: [{elements[0][:,0].min():.1f}, {elements[0][:,0].max():.1f}]")
        print(f"    Y range: [{elements[0][:,1].min():.1f}, {elements[0][:,1].max():.1f}]")
```

### Storage Requirements Per Component

| Component                     | Size      | Required? |
|-------------------------------|-----------|-----------|
| Metadata (v1.0-trainval)      | ~300 MB   | Yes       |
| Camera keyframes (samples/)   | ~80 GB    | Yes       |
| Camera sweeps (sweeps/)       | ~250 GB   | No        |
| Map expansion (maps/)         | ~200 MB   | Yes       |
| GT pickle cache               | ~2-5 GB   | Generated |
| Sequence index                | ~50 MB    | Generated |
| **Minimum total**             | **~85 GB**| --        |

---

## 7. Argoverse 2 Dataset (Secondary Benchmark)

### Overview

Argoverse 2 (AV2) is a complementary dataset from Argo AI (2022) with higher
resolution and more US cities. It tests generalization beyond nuScenes.

### Key Differences from nuScenes

| Feature            | nuScenes                  | Argoverse 2               |
|--------------------|---------------------------|---------------------------|
| Cameras            | 6 surround-view           | 7 ring + 2 stereo         |
| Resolution         | 1600 x 900               | 2048 x 1550              |
| Keyframe rate      | 2 Hz                      | 10 Hz                     |
| Locations          | 2 cities (Boston, SG)     | 6 US cities               |
| Map format         | Per-city global map        | Per-log local map         |
| Map access         | NuScenesMap API           | ArgoverseStaticMap API    |
| Train scenes       | 700                       | 700                       |
| Val scenes         | 150                       | 150                       |
| Total size         | ~350 GB (full)            | ~1.2 TB (full)            |

### 7 Ring Cameras

```
                ring_front_center
                      |
    ring_front_left   |   ring_front_right
                \     |     /
                 \    |    /
                  \   |   /
        +----------[EGO]----------+
        |          VEHICLE        |
        +-------------------------+
    ring_side_left   / \   ring_side_right
                    /   \
     ring_rear_left     ring_rear_right
```

### Per-Log Maps

Unlike nuScenes (one large map per city), AV2 provides a separate map file per
driving log. This means:
- No need to query a large city-wide map
- Map is already local to the driving route
- Accessed via `ArgoverseStaticMap.from_map_dir()`

### Download Instructions

```bash
# Install prerequisites
pip install av2==0.2.1

# Install s5cmd for fast parallel S3 downloads
wget https://github.com/peak/s5cmd/releases/download/v2.1.0/s5cmd_2.1.0_Linux-64bit.tar.gz
tar -xzf s5cmd_2.1.0_Linux-64bit.tar.gz && sudo mv s5cmd /usr/local/bin/

# Download sensor dataset (no AWS credentials needed -- public bucket)
mkdir -p /data/argoverse2/sensor
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/train/*" /data/argoverse2/sensor/train/
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/*" /data/argoverse2/sensor/val/
```

### Verify AV2 Setup

```python
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path

data_root = Path("/data/argoverse2/sensor/val")
log_ids = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
print(f"Number of validation logs: {len(log_ids)}")

# Load first log's map
log_map = ArgoverseStaticMap.from_map_dir(
    data_root / log_ids[0] / "map", build_raster=False
)
print(f"Lane segments: {len(log_map.vector_lane_segments)}")
print(f"Pedestrian crossings: {len(log_map.vector_pedestrian_crossings)}")
```

---

## 8. Dataset Statistics

### nuScenes Map Statistics

| Split | Scenes | Keyframes | Avg elements/frame |
|-------|--------|-----------|-------------------|
| Train | 700    | ~28,130   | ~25-40            |
| Val   | 150    | ~6,019    | ~25-40            |
| Total | 850    | ~34,149   | --                |

### Average Map Elements Per Frame (nuScenes, by class)

| Class          | Avg count/frame | Typical polyline length |
|----------------|-----------------|------------------------|
| lane_divider   | 15-25           | 10-50 meters           |
| road_boundary  | 5-12            | 15-60 meters           |
| ped_crossing   | 2-8             | 5-15 meters            |

### Class Frequency Distribution

Lane dividers dominate (~55% of all elements), followed by road boundaries
(~30%), then pedestrian crossings (~15%). This imbalance is why StreamMapNet
allocates queries per class (50 each) rather than using a single shared pool.

### Polyline Characteristics After Resampling

- All polylines: exactly 20 points, shape (20, 2)
- Point spacing: varies by original length (longer lines = wider spacing)
- Coordinate range: [-30, +30] x [-15, +15] meters (ego frame)
- Padding value for absent elements: -10000.0 (clearly out of range)
- Maximum elements per sample: 50 (padded if fewer)

---

## 9. Storage Requirements

### Complete Storage Table

| Component                          | Size        | Notes                      |
|------------------------------------|-------------|----------------------------|
| nuScenes v1.0-mini                 | ~4 GB       | 10 scenes, for debugging   |
| nuScenes v1.0-trainval (cameras)   | ~80 GB      | Keyframes only, minimum    |
| nuScenes v1.0-trainval (full)      | ~350 GB     | Includes sweeps + LiDAR    |
| nuScenes map expansion             | ~200 MB     | 4 city JSON files          |
| nuScenes GT cache (generated)      | ~2-5 GB     | Pickle files               |
| Argoverse 2 sensor (train)         | ~1 TB       | 700 logs, all sensors      |
| Argoverse 2 sensor (val)           | ~200 GB     | 150 logs                   |
| Argoverse 2 GT cache (generated)   | ~10 GB      | Pickle files               |
| **Minimum (nuScenes only)**        | **~85 GB**  | Cameras + maps + GT cache  |
| **Full setup (both datasets)**     | **~1.6 TB** | Everything                 |

### Disk I/O Considerations

- SSD strongly recommended: training loads random images from random sequences
- HDD throughput (~100 MB/s) becomes bottleneck with batch_size=4, num_workers=4
- NVMe SSD (~3 GB/s) eliminates I/O as training bottleneck
- Map JSON files (~50 MB each) are loaded entirely into RAM during preprocessing

### Symlink Strategy for Multi-Drive Setups

```bash
# If your SSD is small, keep only actively-used data there
# Put raw tarballs on HDD, extracted data on SSD

# Example: SSD at /ssd, HDD at /hdd
ln -s /ssd/nuscenes_extracted /data/nuscenes
ln -s /ssd/stream_mapnet_gt /data/nuscenes/stream_mapnet_gt

# Or split by access pattern:
# Frequently accessed (images): SSD
# Rarely accessed (raw tarballs): HDD
ln -s /ssd/nuscenes/samples /data/nuscenes/samples
ln -s /hdd/nuscenes/sweeps /data/nuscenes/sweeps
```

---

## 10. Common Issues and Troubleshooting

### Missing Map Expansion

**Symptom:**
```
FileNotFoundError: /data/nuscenes/maps/expansion/boston-seaport.json
```

**Cause:** Map expansion pack not downloaded or extracted to wrong location.

**Fix:**
```bash
# Download from nuScenes website
wget https://www.nuscenes.org/data/nuScenes-map-expansion-v1.3.zip
unzip nuScenes-map-expansion-v1.3.zip -d /data/nuscenes/
# Verify: ls /data/nuscenes/maps/expansion/*.json should show 4 files
```

### Ego-Pose Discontinuities

**Symptom:** Large jumps in predicted map between consecutive frames.

**Cause:** nuScenes scenes are extracted from longer logs. At scene boundaries,
ego-pose can jump. Also, GPS drift can cause small discontinuities mid-scene.

**Solution:** StreamMapNet resets its temporal hidden state at scene boundaries.
The sequence index marks scene starts with `None` entries. During inference,
check for ego-pose jumps > 5m between frames as a safety reset trigger.

### Insufficient Disk Space

**Symptom:** Extraction fails mid-way, or training cannot find images.

**Solutions (in order of preference):**
1. Use v1.0-mini (4 GB) for development and debugging
2. Download only keyframe camera data (skip sweeps, LiDAR, RADAR): ~80 GB
3. Process GT annotations incrementally and delete raw map JSONs after
4. Use cloud storage with local caching (e.g., goofys for S3)

### Slow Data Loading

**Symptom:** GPU utilization below 50%, training throughput limited by CPU/IO.

**Solutions:**

```python
# 1. Increase DataLoader workers
train_loader = DataLoader(dataset, batch_size=4, num_workers=8, pin_memory=True)

# 2. Use LMDB for image storage (pre-convert once)
# This gives ~3x speedup over reading individual JPEGs
import lmdb
env = lmdb.open('/data/nuscenes/samples_lmdb', map_size=100*1024**3)

# 3. Memory-map the GT pickle (for large annotation files)
import mmap
with open('train_map_gt.pkl', 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

# 4. Pre-decode and resize images to training resolution (256x704)
# Saves repeated JPEG decode + resize during training
```

### Verify Data Integrity Checklist

```bash
# 1. Check metadata is complete
python -c "
from nuscenes.nuscenes import NuScenes
nusc = NuScenes('v1.0-trainval', '/data/nuscenes', verbose=True)
assert len(nusc.scene) == 850, f'Expected 850 scenes, got {len(nusc.scene)}'
assert len(nusc.sample) > 34000, f'Too few samples: {len(nusc.sample)}'
print('Metadata OK')
"

# 2. Check map expansion
python -c "
from nuscenes.map_expansion.map_api import NuScenesMap
for loc in ['boston-seaport','singapore-onenorth','singapore-queenstown','singapore-hollandvillage']:
    m = NuScenesMap('/data/nuscenes', loc)
    assert len(m.lane_divider) > 0, f'No lane dividers in {loc}'
print('Map expansion OK')
"

# 3. Check camera images exist
python -c "
import os
cam_dir = '/data/nuscenes/samples/CAM_FRONT'
n = len(os.listdir(cam_dir))
assert n > 34000, f'Only {n} front camera images found'
print(f'Camera images OK: {n} files in CAM_FRONT')
"
```

---

## 11. Data Augmentation for Map Training

### Image-Level Augmentations

Applied independently to each camera image before BEV lifting:

| Augmentation   | Parameters                        | Effect                       |
|----------------|-----------------------------------|------------------------------|
| Resize         | Scale factor 0.4-0.5 (to 256x704)| Reduces compute, train input |
| Random crop    | After resize, jitter +/- 10px    | Robustness to framing        |
| Color jitter   | Brightness/contrast/saturation 0.2| Robustness to lighting       |
| GridMask       | Ratio=0.5, grid size=7           | Regularization via occlusion |

Important: when you resize the image, you MUST scale the camera intrinsics
accordingly. Otherwise the BEV lift will project to wrong locations.

```python
# If original image is 1600x900 and we resize to 704x256:
scale_x = 704 / 1600  # 0.44
scale_y = 256 / 900   # 0.284
intrinsic[0, :] *= scale_x  # fx, cx
intrinsic[1, :] *= scale_y  # fy, cy
```

### BEV-Level Augmentations

Applied to the ground truth map annotations in ego-vehicle frame:

| Augmentation    | Parameters             | Effect                          |
|-----------------|------------------------|---------------------------------|
| Random rotation | +/- 22.5 degrees       | Rotation invariance             |
| Random scaling  | 0.9 - 1.1x            | Scale invariance                |
| Random flip     | Left-right (Y-axis)    | Mirror invariance               |

These augmentations modify both the BEV feature map AND the GT polylines
consistently:

```python
def augment_bev(points_ego, rotation_deg, scale, flip_y):
    """Apply BEV augmentation to polyline points in ego frame."""
    theta = np.radians(rotation_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    points = points_ego * scale       # scale
    points = points @ R.T             # rotate
    if flip_y:
        points[:, 1] *= -1           # mirror left-right
    return points
```

### Temporal Consistency Requirements

StreamMapNet processes sequences of 8 frames. Augmentations must respect this:

| Augmentation Type     | Must be consistent across sequence? | Why                         |
|-----------------------|-------------------------------------|-----------------------------|
| Image color jitter    | NO  (per-frame is fine)             | Model should handle varying light |
| Image resize/crop     | YES (same scale for all frames)     | Camera intrinsics must match      |
| BEV rotation/scale    | YES (same transform for all frames) | Temporal fusion aligns features   |
| BEV flip              | YES (same flip for all frames)      | Ego-motion direction must be consistent |

If BEV augmentations differ between frames in a sequence, the temporal fusion
module will try to align features that are inconsistently transformed, causing
training instability and degraded performance.

```python
# Correct: sample augmentation params once per sequence
aug_params = sample_augmentation()  # rotation, scale, flip
for frame in sequence:
    frame['map_elements'] = augment_bev(frame['map_elements'], **aug_params)
    frame['ego_motion'] = augment_ego_motion(frame['ego_motion'], **aug_params)
```

### Augmentation During Evaluation

All augmentations are disabled during validation and testing. Images are
deterministically resized to (256, 704) and GT annotations are used as-is in
ego-vehicle coordinates.

---

## Quick Reference: End-to-End Setup Commands

```bash
# 1. Install dependencies
pip install nuscenes-devkit==1.1.11 pyquaternion numpy

# 2. Download data (see sections above for details)
#    - nuScenes trainval cameras + metadata
#    - nuScenes map expansion v1.3

# 3. Generate GT annotations
python scripts/prepare_map_data.py \
    --dataroot /data/nuscenes \
    --version v1.0-trainval \
    --output-dir /data/nuscenes/stream_mapnet_gt \
    --map-classes lane_divider road_boundary ped_crossing \
    --num-points 20 \
    --bev-range -30 30 -15 15

# 4. Generate sequence index
python scripts/prepare_map_data.py \
    --dataroot /data/nuscenes \
    --version v1.0-trainval \
    --output-dir /data/nuscenes/stream_mapnet_gt \
    --generate-sequence-index \
    --sequence-length 8

# 5. Verify everything
python -c "
import pickle
with open('/data/nuscenes/stream_mapnet_gt/train_map_gt.pkl','rb') as f:
    d = pickle.load(f)
print(f'Ready: {len(d[\"samples\"])} training samples')
"

# 6. Start training (see training_guide.md)
python train.py --config configs/stream_mapnet_nuscenes.yaml
```
