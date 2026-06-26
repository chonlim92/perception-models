# Data Collection Guide for PointPillars

This guide covers the acquisition, structure, and preprocessing of LiDAR point cloud data from the KITTI and nuScenes datasets for training and evaluating PointPillars-based 3D object detection models.

---

## 1. KITTI Dataset

### 1.1 Sensor Specifications

The KITTI dataset uses a **Velodyne HDL-64E** spinning LiDAR mounted on the roof of the data collection vehicle.

| Parameter | Value |
|-----------|-------|
| Number of beams | 64 |
| Rotation rate | 10 Hz |
| Points per scan | ~120,000 |
| Maximum range | 120 m |
| Vertical FOV | -24.9 to +2.0 degrees |
| Horizontal FOV | 360 degrees |
| Angular resolution (vertical) | ~0.4 degrees |
| Angular resolution (horizontal) | ~0.08 degrees (at 10 Hz) |
| Wavelength | 905 nm |

### 1.2 Data Format

Point clouds are stored as **binary `.bin` files**. Each file contains a flat array of `float32` values with 4 values per point:

```
[x1, y1, z1, intensity1, x2, y2, z2, intensity2, ...]
```

- **x, y, z**: 3D coordinates in meters (LiDAR frame)
- **intensity**: Reflectance intensity, typically in the range [0, 1] (some versions use [0, 255])
- **Data type**: `float32` (4 bytes per value, 16 bytes per point)
- **File size**: Approximately 1.9 MB per scan (~120,000 points x 16 bytes)

Reading a KITTI `.bin` file in Python:

```python
import numpy as np

point_cloud = np.fromfile("000000.bin", dtype=np.float32).reshape(-1, 4)
# point_cloud shape: (N, 4) where columns are [x, y, z, intensity]
```

### 1.3 Coordinate Systems

KITTI uses multiple coordinate frames. Understanding the transformations between them is critical for projecting 3D detections into images and vice versa.

**LiDAR Frame (Velodyne)**
- x-axis: forward
- y-axis: left
- z-axis: up
- Origin: center of the LiDAR sensor

**Camera Frame (Reference Camera 0)**
- x-axis: right
- y-axis: down
- z-axis: forward
- Origin: center of the left grayscale camera (camera 0)

**Calibration Matrices**

Each frame has an associated calibration file (`calib/XXXXXX.txt`) containing:

| Matrix | Shape | Description |
|--------|-------|-------------|
| `P0` | 3x4 | Projection matrix for camera 0 (left grayscale) |
| `P1` | 3x4 | Projection matrix for camera 1 (right grayscale) |
| `P2` | 3x4 | Projection matrix for camera 2 (left color) |
| `P3` | 3x4 | Projection matrix for camera 3 (right color) |
| `R0_rect` | 3x3 | Rectification rotation matrix (aligns camera 0 coordinate system) |
| `Tr_velo_to_cam` | 3x4 | Rigid transformation from Velodyne LiDAR frame to camera 0 frame |

**Projection from LiDAR to image (camera 2):**

```python
# Transform LiDAR point to camera 2 image coordinates
# p_lidar: (4, N) homogeneous coordinates [x, y, z, 1]
p_cam = P2 @ R0_rect @ Tr_velo_to_cam @ p_lidar
# Normalize by z to get pixel coordinates
u = p_cam[0] / p_cam[2]
v = p_cam[1] / p_cam[2]
```

### 1.4 Training/Validation Split

The KITTI 3D Object Detection benchmark provides:

| Split | Frames | Notes |
|-------|--------|-------|
| Training set (total) | 7,481 | With ground-truth labels |
| Test set | 7,518 | No public ground-truth (submit to server) |

The standard community split of the training set:

| Subset | Frames | Frame indices |
|--------|--------|---------------|
| Train | 3,712 | From `ImageSets/train.txt` |
| Val | 3,769 | From `ImageSets/val.txt` |

This split was originally proposed by Chen et al. (MV3D) and is widely adopted in the literature (including PointPillars, SECOND, VoxelNet, etc.).

### 1.5 Download Instructions

1. Register at the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
2. Download the following files:
   - **Velodyne point clouds** (29 GB): `data_object_velodyne.zip`
   - **Camera images** (12 GB): `data_object_image_2.zip`
   - **Calibration files** (16 MB): `data_object_calib.zip`
   - **Training labels** (5 MB): `data_object_label_2.zip`
   - **ImageSets** (split files): available from most PointPillars repositories

3. Extract to the following directory structure.

### 1.6 Directory Structure

```
kitti/
├── ImageSets/
│   ├── train.txt                    # 3712 frame indices for training
│   ├── val.txt                      # 3769 frame indices for validation
│   ├── trainval.txt                 # 7481 frame indices (all training)
│   └── test.txt                     # 7518 frame indices for testing
├── training/
│   ├── velodyne/
│   │   ├── 000000.bin               # Point cloud (float32, Nx4)
│   │   ├── 000001.bin
│   │   └── ... (7481 files)
│   ├── image_2/
│   │   ├── 000000.png               # Left color camera image
│   │   ├── 000001.png
│   │   └── ... (7481 files)
│   ├── calib/
│   │   ├── 000000.txt               # Calibration matrices
│   │   ├── 000001.txt
│   │   └── ... (7481 files)
│   └── label_2/
│       ├── 000000.txt               # 3D bounding box annotations
│       ├── 000001.txt
│       └── ... (7481 files)
└── testing/
    ├── velodyne/
    │   └── ... (7518 files)
    ├── image_2/
    │   └── ... (7518 files)
    └── calib/
        └── ... (7518 files)
```

---

## 2. nuScenes Dataset

### 2.1 Sensor Specifications

The nuScenes dataset uses a **Velodyne HDL-32E** spinning LiDAR mounted on the roof of the data collection vehicle.

| Parameter | Value |
|-----------|-------|
| Number of beams | 32 |
| Rotation rate | 20 Hz |
| Points per scan | ~30,000 |
| Maximum range | 70 m (effective) |
| Vertical FOV | -30.67 to +10.67 degrees |
| Horizontal FOV | 360 degrees |
| Angular resolution (vertical) | ~1.33 degrees |

### 2.2 Dataset Statistics

| Parameter | Value |
|-----------|-------|
| Total scenes | 1,000 |
| Scene duration | ~20 seconds each |
| Keyframe annotation rate | 2 Hz (every 0.5 seconds) |
| Total keyframes | ~40,000 |
| Sweep rate (raw LiDAR) | 20 Hz |
| Sweeps between keyframes | 10 |
| Annotated object classes | 23 (10 used for detection benchmark) |
| Train/Val/Test split | 700 / 150 / 150 scenes |

The standard practice is to **aggregate 10 sweeps** (0.5 seconds of data) into a single point cloud for each keyframe to increase point density, compensating for the lower beam count compared to KITTI.

### 2.3 Data Format

Point clouds are stored as **binary `.bin` files** organized in sweep directories:

- Each `.bin` file contains `float32` values with **5 values per point**: `(x, y, z, intensity, ring_index)`
- Alternatively, some versions use `.pcd.bin` format with `(x, y, z, intensity, timestamp)` as 5 channels
- Points are in the **sensor frame** at the time of capture

Reading a nuScenes `.bin` file in Python:

```python
import numpy as np

# nuScenes stores 5 float32 values per point
point_cloud = np.fromfile("sweep_lidar.bin", dtype=np.float32).reshape(-1, 5)
# Columns: [x, y, z, intensity, ring_index]

# For PointPillars, typically use only (x, y, z, intensity)
points = point_cloud[:, :4]
```

### 2.4 Coordinate Systems

nuScenes defines three primary coordinate frames:

**Sensor Frame (LiDAR)**
- x-axis: right
- y-axis: forward
- z-axis: up
- Origin: center of the LiDAR sensor

**Ego Vehicle Frame**
- x-axis: right
- y-axis: forward
- z-axis: up
- Origin: rear axle center, projected to ground

**Global Frame**
- Fixed world coordinate system
- Used for tracking and multi-frame aggregation

**Transformations:**

```python
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

# Sensor to ego vehicle
sensor_to_ego = transform_matrix(
    cs_record['translation'],
    Quaternion(cs_record['rotation'])
)

# Ego vehicle to global
ego_to_global = transform_matrix(
    pose_record['translation'],
    Quaternion(pose_record['rotation'])
)

# Full chain: sensor -> ego -> global
sensor_to_global = ego_to_global @ sensor_to_ego
```

### 2.5 Download Instructions

1. Register at [nuScenes.org](https://www.nuscenes.org/nuscenes)
2. Download options:
   - **Full dataset** (~400 GB): All 1,000 scenes with all sensor data
   - **Mini dataset** (~4 GB): 10 scenes for development and debugging
   - **nuScenes-lidarseg** (optional): Per-point semantic labels

3. Use the nuScenes devkit for data loading:

```bash
pip install nuscenes-devkit
```

4. Set the `NUSCENES` environment variable or specify the dataroot:

```python
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='/data/nuscenes')
```

### 2.6 Directory Structure

```
nuscenes/
├── v1.0-trainval/                   # Metadata JSON files
│   ├── scene.json                   # Scene descriptions
│   ├── sample.json                  # Keyframe references
│   ├── sample_data.json             # All sensor data references
│   ├── sample_annotation.json       # 3D bounding box annotations
│   ├── ego_pose.json                # Vehicle pose at each timestamp
│   ├── calibrated_sensor.json       # Sensor extrinsics/intrinsics
│   ├── sensor.json                  # Sensor metadata
│   ├── instance.json                # Object instance tracking
│   ├── category.json                # Object categories
│   ├── attribute.json               # Object attributes
│   ├── visibility.json              # Visibility levels
│   ├── log.json                     # Log/drive session info
│   └── map.json                     # Map references
├── samples/                         # Keyframe sensor data (2 Hz)
│   ├── LIDAR_TOP/
│   │   ├── n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin
│   │   └── ...
│   ├── CAM_FRONT/
│   │   └── ...
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── RADAR_FRONT/
│   ├── RADAR_FRONT_LEFT/
│   ├── RADAR_FRONT_RIGHT/
│   ├── RADAR_BACK_LEFT/
│   └── RADAR_BACK_RIGHT/
├── sweeps/                          # Non-keyframe sensor data (20 Hz)
│   ├── LIDAR_TOP/
│   │   └── ... (all intermediate LiDAR sweeps)
│   ├── CAM_FRONT/
│   │   └── ...
│   └── ... (same structure as samples/)
└── maps/                            # HD map data
    ├── 36092f0b03a857c6a3403e25b4b7aab3.png
    └── ...
```

---

## 3. Point Cloud Preprocessing

Before feeding point clouds into PointPillars, several preprocessing steps are applied to normalize and filter the data.

### 3.1 Range Filtering (Point Cloud Cropping)

The raw point cloud is cropped to a region of interest defined by `point_cloud_range`. Points outside this range are discarded.

```python
# Typical point_cloud_range for KITTI: [x_min, y_min, z_min, x_max, y_max, z_max]
# In LiDAR frame (x-forward, y-left, z-up)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

def filter_point_cloud_range(points, pc_range):
    """
    Filter points to keep only those within the specified range.
    
    Args:
        points: (N, 4) array of [x, y, z, intensity]
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    
    Returns:
        Filtered points within the range.
    """
    mask = (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) &
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) &
        (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
    )
    return points[mask]
```

Common range configurations:

| Dataset | x range (m) | y range (m) | z range (m) |
|---------|-------------|-------------|-------------|
| KITTI (front only) | [0, 69.12] | [-39.68, 39.68] | [-3, 1] |
| KITTI (full 360) | [-69.12, 69.12] | [-69.12, 69.12] | [-3, 1] |
| nuScenes | [-50, 50] | [-50, 50] | [-5, 3] |

### 3.2 Ground Removal (Optional)

Removing ground plane points can reduce computational load and improve detection of low-height objects. Two common methods:

**RANSAC-based Ground Plane Estimation:**

```python
from sklearn.linear_model import RANSACRegressor

def remove_ground_ransac(points, height_threshold=0.2, n_iterations=100):
    """
    Remove ground points using RANSAC plane fitting.
    
    Args:
        points: (N, 4) array of [x, y, z, intensity]
        height_threshold: Distance threshold to classify as ground
        n_iterations: Number of RANSAC iterations
    
    Returns:
        non_ground_points: Points above the fitted ground plane
    """
    # Use x, y to predict z
    X = points[:, :2]   # x, y
    z = points[:, 2]    # z
    
    ransac = RANSACRegressor(
        residual_threshold=height_threshold,
        max_trials=n_iterations
    )
    ransac.fit(X, z)
    
    # Inliers are ground points
    ground_mask = ransac.inlier_mask_
    non_ground_points = points[~ground_mask]
    
    return non_ground_points
```

**Cloth Simulation Filter (CSF):**

```python
import CSF

def remove_ground_csf(points, cloth_resolution=0.5, class_threshold=0.5):
    """
    Remove ground points using Cloth Simulation Filter.
    
    Args:
        points: (N, 4) array of [x, y, z, intensity]
        cloth_resolution: Grid resolution for the cloth
        class_threshold: Height threshold for ground classification
    
    Returns:
        non_ground_points: Points classified as non-ground
    """
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cloth_resolution
    csf.params.class_threshold = class_threshold
    
    csf.setPointCloud(points[:, :3])
    ground_indices = CSF.VecInt()
    non_ground_indices = CSF.VecInt()
    csf.do_filtering(ground_indices, non_ground_indices)
    
    return points[np.array(non_ground_indices)]
```

Note: PointPillars typically does **not** use ground removal by default, as the network learns to handle ground points implicitly. Ground removal is more common in classical pipelines or as an optional augmentation.

### 3.3 Intensity Normalization

Different LiDAR sensors and datasets encode intensity differently. Normalization ensures consistent input distributions.

```python
def normalize_intensity(points, method='scale_to_unit'):
    """
    Normalize point intensity values.
    
    Args:
        points: (N, 4) array of [x, y, z, intensity]
        method: Normalization method
    
    Returns:
        Points with normalized intensity.
    """
    if method == 'scale_to_unit':
        # Scale from [0, 255] to [0, 1]
        if points[:, 3].max() > 1.0:
            points[:, 3] = points[:, 3] / 255.0
    elif method == 'min_max':
        # Min-max normalization
        i_min = points[:, 3].min()
        i_max = points[:, 3].max()
        if i_max > i_min:
            points[:, 3] = (points[:, 3] - i_min) / (i_max - i_min)
    elif method == 'log':
        # Log normalization (useful for highly skewed distributions)
        points[:, 3] = np.log1p(points[:, 3])
        points[:, 3] = points[:, 3] / points[:, 3].max()
    
    return points
```

- **KITTI**: Intensity is already in [0, 1] in most versions. Verify before normalizing.
- **nuScenes**: Intensity values can vary; typically normalized to [0, 1] by dividing by 255.

### 3.4 Multi-Sweep Aggregation (nuScenes)

Since the HDL-32E produces only ~30k points per sweep (compared to ~120k for KITTI's HDL-64E), multiple sweeps are aggregated to increase point density. This requires **motion compensation** to account for ego-vehicle movement between sweeps.

```python
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import numpy as np

def aggregate_sweeps(nusc, sample_token, nsweeps=10):
    """
    Aggregate multiple LiDAR sweeps into a single point cloud
    with time compensation for ego-motion.
    
    Args:
        nusc: NuScenes database object
        sample_token: Token of the target keyframe
        nsweeps: Number of sweeps to aggregate (including the keyframe)
    
    Returns:
        points: (N, 5) array of [x, y, z, intensity, time_lag]
    """
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    
    # Get reference pose (current keyframe)
    ref_sd = nusc.get('sample_data', lidar_token)
    ref_pose = nusc.get('ego_pose', ref_sd['ego_pose_token'])
    ref_cs = nusc.get('calibrated_sensor', ref_sd['calibrated_sensor_token'])
    
    # Reference transforms
    ref_from_car = transform_matrix(
        ref_cs['translation'], Quaternion(ref_cs['rotation']), inverse=True
    )
    car_from_global = transform_matrix(
        ref_pose['translation'], Quaternion(ref_pose['rotation']), inverse=True
    )
    
    all_points = []
    current_token = lidar_token
    
    for i in range(nsweeps):
        current_sd = nusc.get('sample_data', current_token)
        
        # Load point cloud
        pc = LidarPointCloud.from_file(
            nusc.dataroot / current_sd['filename']
        )
        
        # Get current sweep pose
        current_pose = nusc.get('ego_pose', current_sd['ego_pose_token'])
        current_cs = nusc.get(
            'calibrated_sensor', current_sd['calibrated_sensor_token']
        )
        
        # Transform: current sensor -> current ego -> global -> ref ego -> ref sensor
        car_from_current = transform_matrix(
            current_cs['translation'], Quaternion(current_cs['rotation'])
        )
        global_from_car = transform_matrix(
            current_pose['translation'], Quaternion(current_pose['rotation'])
        )
        
        # Full transformation to reference frame
        trans_matrix = ref_from_car @ car_from_global @ global_from_car @ car_from_current
        pc.transform(trans_matrix)
        
        # Compute time lag relative to keyframe
        time_lag = ref_sd['timestamp'] - current_sd['timestamp']
        time_lag_seconds = time_lag / 1e6  # Convert microseconds to seconds
        
        # Append time lag as additional feature
        n_points = pc.points.shape[1]
        times = np.full((1, n_points), time_lag_seconds)
        points_with_time = np.vstack([pc.points, times])  # (5, N)
        
        all_points.append(points_with_time.T)  # (N, 5)
        
        # Move to previous sweep
        if current_sd['prev'] == '':
            break
        current_token = current_sd['prev']
    
    # Concatenate all sweeps
    aggregated = np.concatenate(all_points, axis=0)  # (M, 5)
    
    return aggregated  # [x, y, z, intensity, time_lag]
```

Key considerations for multi-sweep aggregation:
- **Time compensation**: Each point gets a `time_lag` feature indicating how old it is relative to the keyframe
- **Motion artifacts**: Fast-moving objects leave ghost trails; the time feature helps the network reason about this
- **Point count**: Aggregating 10 sweeps yields ~300k points, comparable to or exceeding KITTI density
- **Memory**: More sweeps increase computation; 10 is the standard trade-off

### 3.5 Coordinate Frame Transformation

When using data from different sources or combining LiDAR with camera data, coordinate transformations are essential.

```python
def transform_lidar_to_camera(points, Tr_velo_to_cam, R0_rect):
    """
    Transform points from LiDAR frame to rectified camera frame (KITTI).
    
    Args:
        points: (N, 3) or (N, 4) point cloud in LiDAR frame
        Tr_velo_to_cam: (3, 4) LiDAR to camera transformation
        R0_rect: (3, 3) Rectification rotation
    
    Returns:
        Points in rectified camera frame.
    """
    n_points = points.shape[0]
    xyz = points[:, :3]
    
    # Convert to homogeneous coordinates
    ones = np.ones((n_points, 1))
    xyz_hom = np.hstack([xyz, ones])  # (N, 4)
    
    # Apply velo_to_cam transformation
    xyz_cam = (Tr_velo_to_cam @ xyz_hom.T).T  # (N, 3)
    
    # Apply rectification
    xyz_rect = (R0_rect @ xyz_cam.T).T  # (N, 3)
    
    return xyz_rect


def transform_nuscenes_to_ego(points, calibrated_sensor):
    """
    Transform points from sensor frame to ego vehicle frame (nuScenes).
    
    Args:
        points: (N, 3+) point cloud in sensor frame
        calibrated_sensor: dict with 'translation' and 'rotation'
    
    Returns:
        Points in ego vehicle frame.
    """
    from pyquaternion import Quaternion
    
    rotation = Quaternion(calibrated_sensor['rotation']).rotation_matrix
    translation = np.array(calibrated_sensor['translation'])
    
    xyz = points[:, :3]
    xyz_ego = (rotation @ xyz.T).T + translation
    
    # Preserve additional features (intensity, etc.)
    if points.shape[1] > 3:
        return np.hstack([xyz_ego, points[:, 3:]])
    return xyz_ego
```

---

## 4. Summary of Key Differences

| Aspect | KITTI | nuScenes |
|--------|-------|----------|
| LiDAR sensor | HDL-64E (64 beams) | HDL-32E (32 beams) |
| Points per scan | ~120,000 | ~30,000 |
| Rotation rate | 10 Hz | 20 Hz |
| Annotation frames | 7,481 (train) | ~40,000 keyframes |
| Multi-sweep | Not needed | 10 sweeps standard |
| Point features | (x, y, z, intensity) | (x, y, z, intensity, time_lag) |
| Coordinate convention | x-fwd, y-left, z-up (LiDAR) | x-right, y-fwd, z-up (sensor) |
| Range (effective) | 120 m | 70 m |
| Detection classes | 3 (Car, Pedestrian, Cyclist) | 10 |
| Difficulty levels | Easy, Moderate, Hard | By distance and visibility |

---

## 5. Quick Start Checklist

1. Download and extract the dataset to the correct directory structure
2. Verify file integrity (check file counts match expected numbers)
3. Configure `point_cloud_range` in the model config for your target dataset
4. Set up the train/val split (use standard splits for reproducibility)
5. Apply intensity normalization appropriate for the dataset
6. For nuScenes: configure multi-sweep aggregation (default: 10 sweeps)
7. Verify coordinate frame conventions match your model's expectations
8. Run a visualization sanity check (project points onto camera images)
