# LiDAR Data Collection and Formats

## 1. Overview

This document describes the LiDAR data formats, acquisition setup, and preprocessing requirements for training PointNet++ models on autonomous driving datasets, focusing on KITTI and nuScenes.

---

## 2. Data Acquisition Setup

### 2.1 Velodyne HDL-64E (KITTI)

| Parameter | Value |
|-----------|-------|
| Sensor | Velodyne HDL-64E S2 |
| Channels | 64 laser beams |
| Vertical FOV | -24.8° to +2.0° |
| Horizontal FOV | 360° |
| Range | 120 m |
| Points per revolution | ~120,000 |
| Rotation rate | 10 Hz |
| Angular resolution (horizontal) | 0.08° - 0.35° |
| Angular resolution (vertical) | ~0.4° |
| Accuracy | ±2 cm |
| Mounting position | Roof-mounted, centered |
| Height above ground | ~1.73 m |

### 2.2 Velodyne HDL-32E (nuScenes)

| Parameter | Value |
|-----------|-------|
| Sensor | Velodyne HDL-32E |
| Channels | 32 laser beams |
| Vertical FOV | -30.67° to +10.67° |
| Horizontal FOV | 360° |
| Range | 100 m |
| Points per revolution | ~34,000 (single return) |
| Rotation rate | 20 Hz |
| Angular resolution (vertical) | ~1.33° |
| Mounting position | Roof-mounted, centered |
| Height above ground | ~1.84 m |

### 2.3 Sensor Calibration

Both datasets provide extrinsic calibration matrices relating:
- LiDAR frame to camera frames
- LiDAR frame to vehicle/IMU frame
- Camera intrinsic parameters

---

## 3. Velodyne Point Cloud Format (.bin)

### 3.1 Binary Format Specification

KITTI stores point clouds as flat binary files with `.bin` extension:

```
File structure:
[x1, y1, z1, r1, x2, y2, z2, r2, ..., xN, yN, zN, rN]

Each point: 4 × float32 = 16 bytes
- x: float32 - X coordinate in meters (forward)
- y: float32 - Y coordinate in meters (left)
- z: float32 - Z coordinate in meters (up)
- r: float32 - Reflectance/intensity (0.0 to 1.0)
```

### 3.2 Reading Point Clouds (Python)

```python
import numpy as np

def load_kitti_bin(filepath):
    """Load a KITTI .bin point cloud file.
    
    Args:
        filepath: Path to .bin file
        
    Returns:
        points: np.ndarray of shape (N, 4) with columns [x, y, z, reflectance]
    """
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    return points

# Example usage
points = load_kitti_bin("000001.bin")
xyz = points[:, :3]          # (N, 3) coordinates
reflectance = points[:, 3]   # (N,) intensity values
```

### 3.3 Point Cloud Statistics (KITTI)

| Statistic | Value |
|-----------|-------|
| Average points per frame | ~120,000 |
| Min points per frame | ~80,000 |
| Max points per frame | ~130,000 |
| X range (forward) | [0, 70] m (filtered front view) |
| Y range (lateral) | [-40, 40] m |
| Z range (vertical) | [-3, 1] m |
| Reflectance range | [0, 1] (normalized) |

---

## 4. KITTI Dataset Structure

### 4.1 Directory Layout

```
kitti/
├── training/
│   ├── velodyne/          # Point clouds (.bin)
│   │   ├── 000000.bin
│   │   ├── 000001.bin
│   │   └── ...
│   ├── label_2/           # 2D/3D annotations (.txt)
│   │   ├── 000000.txt
│   │   └── ...
│   ├── calib/             # Calibration files (.txt)
│   │   ├── 000000.txt
│   │   └── ...
│   ├── image_2/           # Left color camera (PNG)
│   │   └── ...
│   └── planes/            # Ground plane parameters
│       └── ...
├── testing/
│   ├── velodyne/
│   ├── calib/
│   └── image_2/
└── ImageSets/
    ├── train.txt          # 3712 samples
    ├── val.txt            # 3769 samples
    └── test.txt           # 7518 samples
```

### 4.2 KITTI Calibration File Format

```
P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 ...  (3x4 projection matrix, cam0)
P1: ...  (cam1)
P2: ...  (cam2, left color - primary)
P3: ...  (cam3, right color)
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 ...  (3x3 rectification rotation)
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 ...  (3x4, LiDAR→cam0)
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 ...  (3x4, IMU→LiDAR)
```

### 4.3 Coordinate Transformation (KITTI)

```python
import numpy as np

def load_calib(calib_path):
    """Parse KITTI calibration file."""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            key, *values = line.strip().split()
            key = key.rstrip(':')
            calib[key] = np.array([float(v) for v in values])
    
    # Reshape matrices
    calib['P2'] = calib['P2'].reshape(3, 4)
    calib['R0_rect'] = np.eye(4)
    calib['R0_rect'][:3, :3] = calib['R0_rect_raw'].reshape(3, 3)
    calib['Tr_velo_to_cam'] = np.eye(4)
    calib['Tr_velo_to_cam'][:3, :4] = calib['Tr_velo_to_cam_raw'].reshape(3, 4)
    return calib

def lidar_to_camera(points_lidar, calib):
    """Transform points from LiDAR frame to camera frame.
    
    Args:
        points_lidar: (N, 3) points in LiDAR coordinates
        calib: calibration dictionary
    """
    N = points_lidar.shape[0]
    points_hom = np.hstack([points_lidar, np.ones((N, 1))])  # (N, 4)
    points_cam = (calib['R0_rect'] @ calib['Tr_velo_to_cam'] @ points_hom.T).T
    return points_cam[:, :3]
```

---

## 5. nuScenes Dataset Structure

### 5.1 Directory Layout

```
nuscenes/
├── v1.0-trainval/         # Metadata (JSON)
│   ├── sample.json
│   ├── sample_data.json
│   ├── sample_annotation.json
│   ├── ego_pose.json
│   ├── calibrated_sensor.json
│   ├── sensor.json
│   ├── instance.json
│   ├── category.json
│   ├── attribute.json
│   ├── scene.json
│   ├── log.json
│   └── map.json
├── samples/               # Keyframe sensor data
│   ├── LIDAR_TOP/         # Point clouds (.pcd.bin)
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   └── CAM_BACK_RIGHT/
├── sweeps/                # Non-keyframe sensor data (intermediate)
│   ├── LIDAR_TOP/
│   └── ...
└── maps/                  # HD maps
```

### 5.2 nuScenes Point Cloud Format (.pcd.bin)

```
File structure: same as KITTI .bin
[x1, y1, z1, r1, ring1, x2, y2, z2, r2, ring2, ...]

Each point: 5 × float32 = 20 bytes
- x: float32 - X coordinate in ego frame (forward)
- y: float32 - Y coordinate in ego frame (left)
- z: float32 - Z coordinate in ego frame (up)
- intensity: float32 - Reflectance
- ring_index: float32 - Laser ring index (0-31)
```

```python
def load_nuscenes_bin(filepath):
    """Load a nuScenes .pcd.bin point cloud file."""
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
    return points  # (N, 5): x, y, z, intensity, ring_index
```

### 5.3 nuScenes Coordinate System

- **Ego frame:** X forward, Y left, Z up (right-handed)
- **Global frame:** East-North-Up (ENU)
- **Sensor frame:** relative to ego via calibrated_sensor.json

Each sample_data entry has:
- `ego_pose`: ego vehicle pose in global frame (translation + rotation quaternion)
- `calibrated_sensor`: sensor pose relative to ego (translation + rotation quaternion)

### 5.4 Multi-Sweep Aggregation

nuScenes LiDAR operates at 20 Hz but keyframes are annotated at 2 Hz. To increase point density, multiple sweeps are aggregated:

```python
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

def aggregate_sweeps(nusc, sample_token, nsweeps=10):
    """Aggregate multiple LiDAR sweeps into a single point cloud."""
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    
    # Get reference pose
    ref_sd = nusc.get('sample_data', lidar_token)
    ref_pose = nusc.get('ego_pose', ref_sd['ego_pose_token'])
    ref_cs = nusc.get('calibrated_sensor', ref_sd['calibrated_sensor_token'])
    
    all_points = []
    current_token = lidar_token
    
    for _ in range(nsweeps):
        sd = nusc.get('sample_data', current_token)
        pc = LidarPointCloud.from_file(nusc.dataroot / sd['filename'])
        
        # Transform to ego frame, then to global, then to reference frame
        # ... (transformation chain)
        
        all_points.append(pc.points.T)  # (N, 4)
        
        if sd['prev'] == '':
            break
        current_token = sd['prev']
    
    return np.concatenate(all_points, axis=0)
```

---

## 6. Coordinate Systems

### 6.1 KITTI Coordinate Systems

```
LiDAR Frame (Velodyne):        Camera Frame (cam2):
    z (up)                         y (down)
    |                              |
    |                              |
    |_____ x (forward)             |_____ x (right)
   /                              /
  y (left)                       z (forward/depth)

Transformation: cam = R0_rect × Tr_velo_to_cam × lidar
```

### 6.2 nuScenes Coordinate Systems

```
Ego Vehicle Frame:             Global Frame:
    x (forward)                    North (y)
    |                              |
    |                              |
    |_____ y (left)                |_____ East (x)
   /                              /
  z (up)                         Up (z)
```

### 6.3 Common Pitfalls

1. **KITTI LiDAR→Camera:** The LiDAR X-axis (forward) maps to Camera Z-axis (depth)
2. **nuScenes rotations:** Stored as quaternions (w, x, y, z), not Euler angles
3. **KITTI labels:** Annotations are in camera frame, NOT LiDAR frame
4. **Height offset:** LiDAR is mounted ~1.73m above ground; Z=0 is at sensor height

---

## 7. Preprocessing Requirements

### 7.1 Point Cloud Filtering

```python
def preprocess_kitti_pointcloud(points, config):
    """Standard preprocessing for KITTI point clouds.
    
    Args:
        points: (N, 4) raw point cloud [x, y, z, reflectance]
        config: preprocessing parameters
    
    Returns:
        filtered_points: (M, 4) preprocessed points
    """
    # 1. Range filtering (remove points outside detection range)
    x_range = config.get('x_range', [0, 70.4])
    y_range = config.get('y_range', [-40, 40])
    z_range = config.get('z_range', [-3, 1])
    
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    points = points[mask]
    
    # 2. Remove points on ego vehicle (within ~1m radius at sensor height)
    ego_mask = ~(
        (np.abs(points[:, 0]) < 0.5) &
        (np.abs(points[:, 1]) < 1.0) &
        (points[:, 2] > -0.5)
    )
    points = points[ego_mask]
    
    # 3. Optional: subsample to fixed number of points
    npoints = config.get('npoints', None)
    if npoints and points.shape[0] > npoints:
        choice = np.random.choice(points.shape[0], npoints, replace=False)
        points = points[choice]
    elif npoints and points.shape[0] < npoints:
        # Pad by repeating random points
        choice = np.random.choice(points.shape[0], npoints - points.shape[0], replace=True)
        points = np.concatenate([points, points[choice]], axis=0)
    
    return points
```

### 7.2 Ground Plane Removal

```python
def remove_ground_plane(points, height_threshold=-1.5):
    """Simple height-based ground removal.
    
    For production use, consider RANSAC plane fitting.
    """
    mask = points[:, 2] > height_threshold
    return points[mask]

def ransac_ground_removal(points, distance_threshold=0.2, max_iterations=100):
    """RANSAC-based ground plane removal."""
    from sklearn.linear_model import RANSACRegressor
    
    # Fit plane: z = ax + by + c
    X = points[:, :2]  # x, y
    z = points[:, 2]
    
    ransac = RANSACRegressor(residual_threshold=distance_threshold,
                             max_trials=max_iterations)
    ransac.fit(X, z)
    
    inlier_mask = ransac.inlier_mask_
    non_ground = points[~inlier_mask]
    return non_ground
```

### 7.3 Normalization

```python
def normalize_point_cloud(points):
    """Normalize point cloud to unit sphere (for classification tasks)."""
    centroid = np.mean(points[:, :3], axis=0)
    points[:, :3] -= centroid
    max_dist = np.max(np.sqrt(np.sum(points[:, :3]**2, axis=1)))
    points[:, :3] /= max_dist
    return points

def normalize_reflectance(points):
    """Normalize reflectance to [0, 1] range."""
    points[:, 3] = np.clip(points[:, 3], 0, 1)
    return points
```

### 7.4 Data Storage Formats for Training

For efficient training, preprocessed data is typically stored as:

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| `.bin` (raw) | KITTI standard | Compact, fast I/O | No metadata |
| `.npy` | NumPy arrays | Fast loading, typed | Single array only |
| `.npz` | Multiple arrays | Points + labels together | Slight overhead |
| `.h5` (HDF5) | Large datasets | Chunked, compressed | Complex API |
| `.pkl` | Python objects | Flexible structure | Not portable |

### 7.5 Dataset Statistics for Normalization

```
KITTI Training Set (7481 frames):
  Points per frame: mean=119,672, std=12,834
  X range: [0, 79.6] m
  Y range: [-50.2, 50.1] m  
  Z range: [-4.9, 2.4] m
  Reflectance: [0, 1.0], mean=0.21

nuScenes (28130 keyframes):
  Points per frame (single sweep): mean=34,720, std=3,218
  Points per frame (10 sweeps): mean=300,000+
  X range: [-51.2, 51.2] m
  Y range: [-51.2, 51.2] m
  Z range: [-5.0, 3.0] m
```

---

## 8. Data Loading Pipeline

### 8.1 PyTorch Dataset Example

```python
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class KITTIPointCloudDataset(Dataset):
    def __init__(self, root_dir, split='train', npoints=16384, augment=True):
        self.root_dir = root_dir
        self.npoints = npoints
        self.augment = augment
        
        # Load split file
        split_file = os.path.join(root_dir, 'ImageSets', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.sample_ids = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load point cloud
        bin_path = os.path.join(self.root_dir, 'training', 'velodyne', f'{sample_id}.bin')
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        
        # Preprocess
        points = self._filter_points(points)
        points = self._subsample(points)
        
        if self.augment:
            points = self._augment(points)
        
        return torch.from_numpy(points).float()
    
    def _filter_points(self, points):
        mask = (points[:, 0] > 0) & (points[:, 0] < 70.4)
        mask &= (np.abs(points[:, 1]) < 40)
        mask &= (points[:, 2] > -3) & (points[:, 2] < 1)
        return points[mask]
    
    def _subsample(self, points):
        if points.shape[0] >= self.npoints:
            choice = np.random.choice(points.shape[0], self.npoints, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.npoints, replace=True)
        return points[choice]
    
    def _augment(self, points):
        # Random rotation around Z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        points[:, :3] = points[:, :3] @ rot.T
        return points
```

---

## 9. Hardware and Collection Requirements

### 9.1 Sensor Synchronization

- LiDAR timestamps must be synchronized with camera and GPS/IMU
- KITTI uses hardware trigger synchronization
- nuScenes uses software timestamp matching (within 50ms tolerance)

### 9.2 Collection Conditions

| Condition | KITTI | nuScenes |
|-----------|-------|----------|
| Location | Karlsruhe, Germany | Boston, Singapore |
| Weather | Clear, overcast | Clear, rain, night |
| Scenarios | Highway, urban, rural | Dense urban |
| Duration | ~6 hours | 5.5 hours (annotated) |
| Total frames (annotated) | 7,481 train + 7,518 test | 28,130 keyframes |

### 9.3 Known Data Artifacts

1. **Rolling shutter effect:** Velodyne scans take ~100ms per revolution; fast ego-motion causes distortion
2. **Multi-return artifacts:** Some surfaces produce multiple returns (glass, rain)
3. **Blind spots:** Directly above/below sensor, behind mounting hardware
4. **Crosstalk:** Multiple LiDARs can interfere with each other
5. **Motion compensation:** Required for aggregated sweeps (ego-motion correction)
