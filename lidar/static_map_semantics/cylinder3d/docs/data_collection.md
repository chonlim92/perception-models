# Cylinder3D: Data Collection Guide

## Overview

Cylinder3D is trained and evaluated on two primary LiDAR semantic segmentation benchmarks:

1. **SemanticKITTI** - Large-scale outdoor driving dataset with Velodyne HDL-64E
2. **nuScenes-lidarseg** - Multi-modal autonomous driving dataset with 32-beam LiDAR

---

## SemanticKITTI Dataset

### General Information

| Property | Value |
|----------|-------|
| Source | KITTI Odometry Benchmark (extended) |
| Sensor | Velodyne HDL-64E rotating LiDAR |
| Beams | 64 |
| Points per scan | ~120,000 (varies: 80K–130K) |
| Frequency | 10 Hz |
| Total scans | 43,552 |
| Annotated scans | 43,552 (all have semantic labels) |
| Classes | 28 raw → 19 evaluation classes + 1 unlabeled |
| Location | Karlsruhe, Germany |
| Driving scenarios | Urban, suburban, highway, residential |

### Data Splits

| Split | Sequences | Scans | Usage |
|-------|-----------|-------|-------|
| Train | 00–07, 09–10 | 19,130 | Model training |
| Validation | 08 | 4,071 | Development / ablation |
| Test | 11–21 | 20,351 | Leaderboard evaluation (labels hidden) |

### Sensor Specifications: Velodyne HDL-64E

| Parameter | Value |
|-----------|-------|
| Beams | 64 |
| Vertical FOV | -24.8° to +2.0° (26.8° total) |
| Horizontal FOV | 360° |
| Range | 120 m (max) |
| Angular resolution (horizontal) | 0.08° – 0.35° |
| Angular resolution (vertical) | ~0.4° average |
| Rotation rate | 5–20 Hz (10 Hz in KITTI) |
| Points per second | ~1.3 million |
| Accuracy | ±2 cm |
| Return mode | Dual return |

### File Format

**Point cloud files:** `.bin` (binary)
```
Location: dataset/sequences/{seq_id}/velodyne/{frame_id}.bin

Format: N × 4 float32 (little-endian)
  - Column 0: x (meters, forward)
  - Column 1: y (meters, left)
  - Column 2: z (meters, up)
  - Column 3: remission/intensity (0.0 – 1.0)

Reading example (Python):
  points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
  xyz = points[:, :3]
  intensity = points[:, 3]
```

**Label files:** `.label` (binary)
```
Location: dataset/sequences/{seq_id}/labels/{frame_id}.label

Format: N × 1 uint32 (little-endian)
  - Lower 16 bits: semantic label ID
  - Upper 16 bits: instance ID

Reading example (Python):
  labels = np.fromfile(path, dtype=np.uint32)
  semantic = labels & 0xFFFF
  instance = labels >> 16
```

**Calibration files:** `.txt`
```
Location: dataset/sequences/{seq_id}/calib.txt

Contains:
  - P0–P3: camera projection matrices (3×4)
  - Tr: LiDAR-to-camera transform (3×4)
```

**Pose files:** `.txt`
```
Location: dataset/sequences/{seq_id}/poses.txt

Format: One 4×4 homogeneous transform per line (12 values, row-major, last row [0,0,0,1] omitted)
```

### Directory Structure

```
SemanticKITTI/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   └── ...
│   │   ├── labels/
│   │   │   ├── 000000.label
│   │   │   ├── 000001.label
│   │   │   └── ...
│   │   ├── calib.txt
│   │   └── poses.txt
│   ├── 01/
│   │   └── ...
│   └── ...
└── semantic-kitti.yaml  (class definitions, splits, color map)
```

### Collection Environment

- **Vehicle:** Volkswagen Passat B6
- **Mounting:** LiDAR on roof rack, approximately 1.73 m above ground
- **GPS/IMU:** OXTS RT 3003 (for pose ground truth)
- **Cameras:** 2× PointGrey Flea2 (color), 2× PointGrey Flea2 (grayscale)
- **Conditions:** Dry weather, daytime, various traffic densities
- **Speed range:** 0–80 km/h

---

## nuScenes-lidarseg Dataset

### General Information

| Property | Value |
|----------|-------|
| Source | nuScenes dataset (Motional/Aptiv) |
| Sensor | Velodyne HDL-32E rotating LiDAR |
| Beams | 32 |
| Points per scan | ~34,000 (varies: 25K–40K) |
| Frequency | 20 Hz (keyframes at 2 Hz) |
| Total keyframes | 40,000 (annotated) |
| Scenes | 1,000 (20-second clips) |
| Classes | 32 raw → 16 evaluation classes |
| Locations | Boston (Seaport), Singapore (Queenstown, Holland Village, One-North) |
| Driving scenarios | Dense urban, construction zones, rain, night |

### Data Splits

| Split | Scenes | Keyframes | Usage |
|-------|--------|-----------|-------|
| Train | 700 | 28,130 | Model training |
| Validation | 150 | 6,019 | Development / ablation |
| Test | 150 | 6,008 | Leaderboard evaluation |

### Sensor Specifications: Velodyne HDL-32E

| Parameter | Value |
|-----------|-------|
| Beams | 32 |
| Vertical FOV | -30.67° to +10.67° (41.33° total) |
| Horizontal FOV | 360° |
| Range | 100 m (max), 80 m (typical) |
| Angular resolution (horizontal) | 0.1° – 0.4° |
| Angular resolution (vertical) | ~1.33° average |
| Rotation rate | 20 Hz |
| Points per second | ~695,000 |
| Accuracy | ±2 cm |
| Return mode | Dual return |

### File Format

**Point cloud files:** `.pcd.bin` (binary)
```
Location: data/sets/nuscenes/samples/LIDAR_TOP/{token}.pcd.bin

Format: N × 5 float32 (little-endian)
  - Column 0: x (meters, ego-vehicle frame)
  - Column 1: y (meters, ego-vehicle frame)
  - Column 2: z (meters, ego-vehicle frame)
  - Column 3: intensity (0.0 – 255.0)
  - Column 4: ring index (beam number, float-encoded)

Reading example (Python):
  points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
  xyz = points[:, :3]
  intensity = points[:, 3]
  ring = points[:, 4].astype(np.int32)
```

**Label files:** `.lidarseg.bin` (binary)
```
Location: data/sets/nuscenes/lidarseg/v1.0-{split}/{token}_lidarseg.bin

Format: N × 1 uint8
  - Each byte is the semantic class label (0–31)

Reading example (Python):
  labels = np.fromfile(path, dtype=np.uint8)
```

**Metadata:** JSON-based database
```
nuScenes uses a relational database stored in JSON files:
  - sample.json: keyframe metadata
  - sample_data.json: sensor data references
  - ego_pose.json: vehicle poses
  - calibrated_sensor.json: sensor extrinsics/intrinsics
  - lidarseg.json: maps sample_data tokens to label files
```

### Directory Structure

```
nuscenes/
├── samples/
│   ├── LIDAR_TOP/
│   │   ├── n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530.bin
│   │   └── ...
│   ├── CAM_FRONT/
│   └── ...
├── sweeps/
│   ├── LIDAR_TOP/
│   └── ...
├── lidarseg/
│   └── v1.0-trainval/
│       ├── {token}_lidarseg.bin
│       └── ...
├── v1.0-trainval/
│   ├── sample.json
│   ├── sample_data.json
│   ├── ego_pose.json
│   ├── calibrated_sensor.json
│   ├── lidarseg.json
│   └── ...
└── nuscenes_infos_{train|val}.pkl  (preprocessed metadata)
```

### Collection Environment

- **Vehicles:** Renault Zoe electric vehicles
- **Mounting:** LiDAR on roof, approximately 1.84 m above ground
- **GPS/IMU:** NovAtel SPAN-CPT (RTK-corrected)
- **Cameras:** 6× cameras (360° coverage)
- **Radar:** 5× Continental ARS 408-21 radar
- **Conditions:** Day, night, rain, construction, heavy traffic
- **Speed range:** 0–60 km/h (urban driving)
- **Countries:** USA (Boston), Singapore

---

## Data Preprocessing for Cylinder3D

### Coordinate Range Clipping

Before cylindrical voxelization, points are clipped to a working volume:

```python
# SemanticKITTI typical ranges
x_range = [-50.0, 50.0]   # meters
y_range = [-50.0, 50.0]   # meters
z_range = [-3.0, 1.0]     # meters (below ground to above vehicle)

# Derived cylindrical ranges
r_range = [0.0, 50.0]     # meters (radial)
theta_range = [0.0, 2*pi] # radians (full rotation)
z_range = [-3.0, 1.0]     # meters (height, same as Cartesian)
```

### Point Features Used

| Feature | Source | Notes |
|---------|--------|-------|
| x, y, z | Raw point cloud | Converted to cylindrical for voxelization |
| Intensity | Sensor return | Normalized to [0, 1] |
| r, θ, z | Computed | Cylindrical coordinates |
| Δx, Δy, Δz | Computed | Offset from voxel center |

### Voxelization Statistics

| Dataset | Avg occupied voxels | Grid fill rate |
|---------|-------------------|----------------|
| SemanticKITTI | ~35,000 | ~0.63% of 480×360×32 |
| nuScenes | ~12,000 | ~0.22% of 480×360×32 |

The extreme sparsity justifies the use of sparse convolution operations.

---

## Data Download and Setup

### SemanticKITTI

```bash
# Download from http://www.semantic-kitti.org/dataset.html
# Requires registration; total size ~80 GB

# Velodyne point clouds (from KITTI Odometry)
wget http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip  # ~80 GB

# SemanticKITTI labels
wget http://www.semantic-kitti.org/assets/data_odometry_labels.zip  # ~700 MB
```

### nuScenes-lidarseg

```bash
# Download from https://www.nuscenes.org/nuscenes#download
# Requires registration; total size ~400 GB (full dataset)

# For lidarseg only, minimum required:
# 1. Full dataset v1.0 (keyframes + metadata)
# 2. lidarseg annotations addon

# Using nuscenes-devkit:
pip install nuscenes-devkit
python -c "from nuscenes import NuScenes; nusc = NuScenes(version='v1.0-trainval', dataroot='/data/nuscenes')"
```

---

## Key Differences Between Datasets

| Aspect | SemanticKITTI | nuScenes-lidarseg |
|--------|---------------|-------------------|
| Beam count | 64 | 32 |
| Point density | ~120K/scan | ~34K/scan |
| Range | 120 m | 100 m |
| Annotation classes | 19 (eval) | 16 (eval) |
| Geographic diversity | Single city | 2 countries |
| Weather diversity | Dry only | Rain, night included |
| Label format | Binary uint32 | Binary uint8 |
| Instance labels | Yes (upper 16 bits) | Separate (via detection boxes) |
| Temporal info | Sequential scans | Keyframes at 2 Hz |
