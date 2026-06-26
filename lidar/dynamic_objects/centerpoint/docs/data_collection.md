# Data Collection: LiDAR Point Cloud Datasets

## Overview

CenterPoint is evaluated on two major autonomous driving datasets: nuScenes and Waymo Open Dataset. Both provide high-quality 3D LiDAR point clouds with comprehensive annotations for detection and tracking.

---

## nuScenes Dataset

### LiDAR Sensor Specifications

| Parameter | Value |
|-----------|-------|
| Sensor | Velodyne HDL-32E |
| Beams | 32 channels |
| Rotation rate | 20 Hz |
| Points per sweep | ~34,000 (after motion compensation) |
| Range | 70 m effective |
| Vertical FOV | -30.67 to +10.67 degrees |
| Angular resolution (horizontal) | 0.1 - 0.4 degrees |
| Return mode | Strongest return |

### Dataset Statistics

| Split | Scenes | Frames (keyframes) | Annotated objects |
|-------|--------|-------------------|-------------------|
| Train | 700 | 28,130 | ~1.4M boxes |
| Val | 150 | 6,019 | ~300K boxes |
| Test | 150 | 6,008 | - (held out) |

### Multi-Sweep Aggregation

Since a single 32-beam LiDAR sweep is relatively sparse (~34K points), CenterPoint aggregates multiple consecutive sweeps to densify the point cloud:

- **Number of sweeps:** 10 (current frame + 9 previous frames)
- **Time span:** 0.5 seconds (at 20 Hz keyframe rate, with 2 Hz annotation rate)
- **Motion compensation:** Previous sweeps are transformed to the current ego-vehicle coordinate frame using:
  1. Ego-motion compensation via IMU/odometry poses
  2. Rigid body transformation: `P_current = T_ego_current_inv @ T_ego_past @ P_past`
- **Time encoding:** Each point is augmented with a time lag feature `dt` indicating how old the sweep is (0.0 for current, up to 0.45 for the oldest sweep)
- **Result:** ~300,000 points per aggregated frame (10x density improvement)

### Point Feature Vector

Each point in the aggregated cloud has the following features:

```
[x, y, z, intensity, time_lag]
```

- `x, y, z`: 3D coordinates in the current ego-vehicle frame (meters)
- `intensity`: Reflectance value normalized to [0, 1]
- `time_lag`: Time difference from the current sweep (seconds, 0.0 to 0.45)

### Voxelization Parameters (nuScenes)

| Parameter | Value |
|-----------|-------|
| Voxel size (x, y, z) | [0.075, 0.075, 0.2] m |
| Point cloud range | [-54, -54, -5.0, 54, 54, 3.0] m |
| Grid dimensions | [1440, 1440, 40] |
| Max points per voxel | 10 |
| Max voxels | 120,000 (train) / 160,000 (test) |
| Feature encoding | Mean VFE (mean of point features within voxel) |

### Detection Classes (10 classes)

| Class | Examples |
|-------|----------|
| car | Sedans, SUVs, hatchbacks |
| truck | Pickup trucks, delivery trucks |
| bus | City buses, school buses |
| trailer | Semi-trailers, cargo trailers |
| construction_vehicle | Excavators, cranes, bulldozers |
| pedestrian | Adults, children |
| motorcycle | Motorcycles with riders |
| bicycle | Bicycles with riders |
| traffic_cone | Standard traffic cones |
| barrier | Jersey barriers, concrete barriers |

---

## Waymo Open Dataset

### LiDAR Sensor Specifications

| Parameter | Value |
|-----------|-------|
| Sensor | Custom 64-beam top LiDAR + 4 short-range LiDARs |
| Top LiDAR beams | 64 channels |
| Short-range LiDAR beams | 200 lines each |
| Rotation rate | 10 Hz |
| Points per frame (top) | ~177,000 |
| Points per frame (all 5) | ~230,000 |
| Range (top) | 75 m |
| Vertical FOV (top) | -17.6 to +2.4 degrees |

### Dataset Statistics

| Split | Sequences | Frames | Annotated objects |
|-------|-----------|--------|-------------------|
| Train | 798 | 158,361 | ~9.9M boxes |
| Val | 202 | 40,077 | ~2.5M boxes |
| Test | 150 | 29,854 | - (held out) |

### Multi-Sweep Aggregation (Waymo)

- **Number of sweeps:** Typically 2-3 (Waymo frames are already denser due to 64-beam LiDAR)
- **Time span:** 0.1-0.2 seconds
- **Motion compensation:** Same ego-motion transformation approach as nuScenes

### Voxelization Parameters (Waymo)

| Parameter | Value |
|-----------|-------|
| Voxel size (x, y, z) | [0.1, 0.1, 0.15] m |
| Point cloud range | [-75.2, -75.2, -2.0, 75.2, 75.2, 4.0] m |
| Grid dimensions | [1504, 1504, 40] |
| Max points per voxel | 5 |
| Max voxels | 150,000 (train) / 200,000 (test) |
| Feature encoding | Mean VFE |

### Detection Classes (3 classes)

| Class | Description |
|-------|-------------|
| Vehicle | Cars, trucks, buses, motorcycles |
| Pedestrian | Adults, children, wheelchair users |
| Cyclist | Bicyclists, motorcyclists |

---

## Data Format and Storage

### nuScenes Data Format

```
nuscenes/
├── maps/                    # HD maps (not used in CenterPoint)
├── samples/                 # Keyframe sensor data
│   └── LIDAR_TOP/          # .pcd.bin files (binary point clouds)
├── sweeps/                  # Non-keyframe sensor data (for multi-sweep)
│   └── LIDAR_TOP/          # .pcd.bin files
├── v1.0-trainval/          # Annotation JSON files
│   ├── sample.json         # Keyframe timestamps and tokens
│   ├── sample_data.json    # Sensor data references
│   ├── sample_annotation.json  # 3D bounding box annotations
│   ├── instance.json       # Object instance (track) metadata
│   ├── ego_pose.json       # Ego vehicle poses
│   ├── calibrated_sensor.json  # Sensor extrinsics/intrinsics
│   └── ...
└── v1.0-test/              # Test set annotations (no boxes)
```

### Point Cloud Binary Format (.pcd.bin)

```python
# nuScenes: 5 floats per point
# [x, y, z, intensity, ring_index]
points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)

# Waymo: stored in TFRecord format
# Each frame contains range images that are converted to point clouds
```

### Preprocessed Data (for training efficiency)

CenterPoint typically preprocesses raw data into a serialized format:

```python
# Info dict per frame (saved as .pkl)
info = {
    'lidar_path': str,           # Path to point cloud file
    'token': str,                # Unique frame identifier
    'sweeps': List[dict],        # Metadata for aggregated sweeps
    'timestamp': int,            # Frame timestamp (microseconds)
    'ego2global_translation': np.ndarray,  # [3,]
    'ego2global_rotation': np.ndarray,     # [4,] quaternion
    'gt_boxes': np.ndarray,      # [N, 9] (x,y,z,w,l,h,yaw,vx,vy)
    'gt_names': List[str],       # [N,] class names
    'gt_velocity': np.ndarray,   # [N, 2] (vx, vy) in global frame
    'num_lidar_pts': np.ndarray, # [N,] points per box
    'num_radar_pts': np.ndarray, # [N,] radar points per box
}
```

---

## Data Loading Pipeline

### Training Pipeline

```
1. Load point cloud binary file
2. Load sweep metadata and aggregate multi-sweep points
3. Apply ego-motion compensation to align sweeps
4. Append time_lag feature to each point
5. Apply data augmentation:
   a. GT sampling (paste ground truth boxes from database)
   b. Random global rotation [-π/4, π/4]
   c. Random global scaling [0.95, 1.05]
   d. Random global translation [-0.2, 0.2] m
   e. Random flip (x-axis and y-axis)
6. Filter points outside point_cloud_range
7. Voxelize points into 3D grid
8. Generate target heatmaps and regression targets
```

### Evaluation Pipeline

```
1. Load point cloud binary file
2. Load sweep metadata and aggregate multi-sweep points
3. Apply ego-motion compensation
4. Append time_lag feature
5. Filter points outside point_cloud_range
6. Voxelize points into 3D grid
7. Run model inference
8. Post-process detections (peak extraction, decode boxes)
9. Transform predictions to global frame for evaluation
```

---

## Hardware Requirements for Data Processing

| Task | Recommended Hardware |
|------|---------------------|
| Data preprocessing | 32+ CPU cores, 64 GB RAM |
| Training (nuScenes) | 8x NVIDIA A100 (40GB) or 8x V100 (32GB) |
| Training (Waymo) | 8x NVIDIA A100 (80GB) recommended |
| Inference | 1x NVIDIA RTX 3090 / A100 |
| Storage (nuScenes full) | ~400 GB |
| Storage (Waymo full) | ~2 TB |
