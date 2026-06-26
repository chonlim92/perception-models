# BEVFormer: Data Collection Guide

## nuScenes Dataset Requirements

BEVFormer is trained and evaluated on the **nuScenes** dataset, a large-scale autonomous driving dataset designed for holistic 3D perception research.

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Total scenes | 1,000 |
| Scene duration | ~20 seconds each |
| Total keyframes | 40,000 (annotated at 2 Hz) |
| Total sweep frames | ~1,400,000 (unannotated at ~12 Hz) |
| Cameras | 6 (full 360-degree surround) |
| LiDAR | 1 (32-beam, 360-degree) |
| Radar | 5 (long-range + short-range) |
| GPS/IMU | Yes (for ego-pose) |
| Location | Boston, Singapore |
| Weather/Time | Day, night, rain, overcast |
| Annotation classes | 23 (10 used for detection) |
| Split | Train: 700 / Val: 150 / Test: 150 |

### Key Dataset Properties for BEVFormer

- **Keyframe annotations at 2 Hz:** BEVFormer uses these keyframes for training with temporal pairs (current + previous frame at 0.5s interval)
- **Full calibration data:** Camera intrinsics and extrinsics are provided per frame, enabling the spatial cross-attention projection
- **Ego-pose data:** Required for temporal self-attention alignment between frames
- **Sweep data (optional):** Intermediate unannotated frames can be used for denser temporal modeling

---

## 2. Camera Setup

### 2.1 Camera Configuration

nuScenes uses 6 cameras providing a full 360-degree horizontal field of view:

```
                    FRONT
                   /     \
          FRONT_LEFT     FRONT_RIGHT
              |               |
          BACK_LEFT      BACK_RIGHT
                   \     /
                    BACK
```

| Camera | Horizontal FOV | Orientation | Primary Coverage |
|--------|---------------|-------------|-----------------|
| CAM_FRONT | 70 deg | Forward | 0 deg (forward) |
| CAM_FRONT_LEFT | 70 deg | Forward-left | ~55 deg left |
| CAM_FRONT_RIGHT | 70 deg | Forward-right | ~55 deg right |
| CAM_BACK | 110 deg | Backward | 180 deg (rear) |
| CAM_BACK_LEFT | 70 deg | Backward-left | ~110 deg left |
| CAM_BACK_RIGHT | 70 deg | Backward-right | ~110 deg right |

### 2.2 Camera Specifications

| Property | Value |
|----------|-------|
| Resolution | 1600 x 900 pixels |
| Sensor | 1/2.7" CMOS |
| Frame rate | 12 Hz (keyframes at 2 Hz) |
| Color depth | 8-bit RGB |
| Format | JPEG |
| Lens type | Wide-angle (varies per position) |

### 2.3 Camera Overlap

- Adjacent cameras have ~10-15 degrees of overlap
- The front camera and front-left/front-right cameras overlap in the forward-diagonal directions
- Back camera has wider FOV (110 deg) to compensate for the larger gap between rear cameras

### 2.4 Relevance to BEVFormer

BEVFormer's spatial cross-attention projects 3D reference points onto all 6 camera image planes. For each BEV query:
- Reference points are projected to every camera
- Only cameras where the projection falls within valid image bounds contribute features
- This naturally handles the 360-degree coverage without explicit view assignment

---

## 3. Download Instructions

### 3.1 Registration and Access

1. Visit [https://www.nuscenes.org/](https://www.nuscenes.org/)
2. Create a free account (academic/commercial)
3. Accept the Terms of Use
4. Navigate to the Download page

### 3.2 Dataset Versions

| Version | Size | Use Case |
|---------|------|----------|
| Full dataset (v1.0-trainval) | ~400 GB | Full training and validation |
| Full dataset (v1.0-test) | ~60 GB | Test set submission |
| Mini dataset (v1.0-mini) | ~4 GB | Development and debugging |

### 3.3 Download Commands

```bash
# Option 1: Direct download via browser
# Navigate to nuscenes.org/download and download archives

# Option 2: Using the official download script
pip install nuscenes-devkit
python -c "from nuscenes.utils.download import download; download('v1.0-trainval', '/data/nuscenes')"

# Option 3: AWS CLI (faster for large downloads)
# Requires AWS credentials from nuscenes.org
aws s3 cp s3://nuscenes/v1.0-trainval/ /data/nuscenes/ --recursive --no-sign-request
```

### 3.4 Required Archives for BEVFormer

For the **full dataset**, download these archives:

```
v1.0-trainval_meta.tgz          (~300 MB)  - Annotations, calibration, ego-pose
v1.0-trainval01_blobs.tgz       (~70 GB)   - Camera images batch 1
v1.0-trainval02_blobs.tgz       (~70 GB)   - Camera images batch 2
v1.0-trainval03_blobs.tgz       (~70 GB)   - Camera images batch 3
v1.0-trainval04_blobs.tgz       (~70 GB)   - Camera images batch 4
v1.0-trainval05_blobs.tgz       (~70 GB)   - Camera images batch 5
v1.0-trainval06_blobs.tgz       (~70 GB)   - Camera images batch 6
v1.0-trainval07_blobs.tgz       (~20 GB)   - Camera images batch 7
v1.0-trainval08_blobs.tgz       (~20 GB)   - Camera images batch 8
v1.0-trainval09_blobs.tgz       (~20 GB)   - Camera images batch 9
v1.0-trainval10_blobs.tgz       (~20 GB)   - Camera images batch 10
```

**Note:** BEVFormer is camera-only, so LiDAR point cloud archives are NOT required for training. However, LiDAR data may be needed for:
- Generating ground truth depth maps (if using depth supervision)
- Evaluation scripts that reference point cloud data

### 3.5 Mini Dataset (for development)

```bash
# Download mini split for quick testing
wget https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xzf v1.0-mini.tgz -C /data/nuscenes/
```

The mini dataset contains 10 scenes (4 train, 4 val, 2 test) and is sufficient for verifying the data pipeline.

---

## 4. Data Format

### 4.1 Directory Structure

```
nuscenes/
├── maps/                          # Map raster images
│   ├── basemap/
│   └── expansion/
├── samples/                       # Keyframe sensor data (2 Hz)
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── LIDAR_TOP/
│   ├── RADAR_FRONT/
│   ├── RADAR_FRONT_LEFT/
│   ├── RADAR_FRONT_RIGHT/
│   ├── RADAR_BACK_LEFT/
│   └── RADAR_BACK_RIGHT/
├── sweeps/                        # Intermediate frames (~12 Hz)
│   ├── CAM_FRONT/
│   ├── ...
│   └── LIDAR_TOP/
└── v1.0-trainval/                 # Metadata JSON files
    ├── attribute.json
    ├── calibrated_sensor.json
    ├── category.json
    ├── ego_pose.json
    ├── instance.json
    ├── log.json
    ├── map.json
    ├── sample.json
    ├── sample_annotation.json
    ├── sample_data.json
    ├── scene.json
    ├── sensor.json
    └── visibility.json
```

### 4.2 Key Metadata Tables

#### sample.json
Each sample represents a keyframe timestamp with all sensors synchronized:
```json
{
    "token": "ca9a282c9e77460f8360f564131a8af5",
    "timestamp": 1532402927647951,
    "prev": "39586f9d59004284a7114a68825e8eec",
    "next": "01fc50f68de944ee9c437b4e235b7d28",
    "scene_token": "cc8c0bf57f984915a77078b10eb33198"
}
```

#### sample_data.json
Links each sensor reading to its file and calibration:
```json
{
    "token": "5ace90b379af485b9dcb1584b01e7212",
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "ego_pose_token": "5ace90b379af485b9dcb1584b01e7212",
    "calibrated_sensor_token": "2fde3d3376ea42a8a561df595e001cc7",
    "timestamp": 1532402927612460,
    "fileformat": "jpg",
    "is_key_frame": true,
    "height": 900,
    "width": 1600,
    "filename": "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg",
    "prev": "a5e7c5f2e3aa4bcaaa3a70e6a3a2429e",
    "next": "9b25e0e1064746ed8caa49e0c2e1b859"
}
```

#### sample_annotation.json
3D bounding box annotations for each object instance:
```json
{
    "token": "70aecbe9b64f4722ab3c230f9a1d7149",
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "instance_token": "e91afa15647c4c4994f19a6013df6bb8",
    "visibility_token": "4",
    "attribute_tokens": ["cb5118da1ab342aa947717dc53544259"],
    "translation": [373.214, 1130.48, 1.25],
    "size": [0.621, 0.669, 1.642],
    "rotation": [0.9831, 0.0, 0.0, -0.1830],
    "velocity": [0.0, 0.0],
    "num_lidar_pts": 5,
    "num_radar_pts": 2,
    "prev": "",
    "next": "a480a36b9a5e4ebfbbeb3c1c85fed621"
}
```

#### ego_pose.json
Vehicle pose at each sensor reading timestamp:
```json
{
    "token": "5ace90b379af485b9dcb1584b01e7212",
    "timestamp": 1532402927612460,
    "rotation": [0.5720, -0.0016, 0.0130, -0.8201],
    "translation": [410.77, 1137.28, 0.0]
}
```

#### calibrated_sensor.json
Sensor calibration (extrinsics + intrinsics):
```json
{
    "token": "2fde3d3376ea42a8a561df595e001cc7",
    "sensor_token": "ec4b5d41840a509984f7ec36419d4c09",
    "translation": [1.70, 0.00, 1.51],
    "rotation": [0.5077, -0.4973, 0.4978, -0.4972],
    "camera_intrinsic": [
        [1266.417, 0.0, 816.267],
        [0.0, 1266.417, 491.507],
        [0.0, 0.0, 1.0]
    ]
}
```

---

## 5. Storage Requirements

### 5.1 Full Dataset

| Component | Size | Required for BEVFormer? |
|-----------|------|------------------------|
| Camera images (samples) | ~70 GB | Yes |
| Camera images (sweeps) | ~270 GB | Optional (for dense temporal) |
| LiDAR point clouds | ~50 GB | No (camera-only) |
| Radar data | ~10 GB | No |
| Metadata (JSON) | ~300 MB | Yes |
| Maps | ~2 GB | Optional (for map tasks) |
| **Total (minimum for BEVFormer)** | **~72 GB** | |
| **Total (full dataset)** | **~400 GB** | |

### 5.2 Training Storage Requirements

Beyond the raw dataset, training BEVFormer requires additional storage:

| Component | Size |
|-----------|------|
| Preprocessed info files (pkl) | ~2 GB |
| Model checkpoints (per epoch) | ~1 GB each |
| Training logs | ~100 MB |
| Tensorboard events | ~500 MB |
| **Recommended free space** | **~100 GB** (dataset + working space) |

### 5.3 Storage Recommendations

- Use **SSD storage** for the dataset directory to avoid I/O bottlenecks during training
- If using NFS/shared storage, ensure sufficient bandwidth (>1 GB/s)
- Consider creating symbolic links if dataset and working directories are on different volumes:
  ```bash
  ln -s /fast_ssd/nuscenes /data/nuscenes
  ```

---

## 6. Sensor Calibration

### 6.1 Coordinate Systems

nuScenes defines four coordinate systems:

```
Global Frame (world)
    |
    | ego_pose (rotation + translation)
    v
Ego Vehicle Frame
    |
    | calibrated_sensor (rotation + translation)
    v
Sensor Frame (camera/lidar/radar)
    |
    | camera_intrinsic (for cameras only)
    v
Image Frame (pixels)
```

All coordinate systems follow a **right-handed** convention:
- **Global frame:** Fixed world coordinate system (East-North-Up)
- **Ego frame:** Origin at rear axle center, X-forward, Y-left, Z-up
- **Camera frame:** X-right, Y-down, Z-forward (standard camera convention)
- **LiDAR frame:** X-forward, Y-left, Z-up (same as ego)

### 6.2 Intrinsic Calibration

Camera intrinsics define the projection from 3D camera coordinates to 2D pixel coordinates:

```
K = [fx  0  cx]     3x3 intrinsic matrix
    [ 0 fy  cy]
    [ 0  0   1]
```

| Parameter | Description | Typical Value (CAM_FRONT) |
|-----------|-------------|--------------------------|
| fx | Focal length (x) | ~1266 pixels |
| fy | Focal length (y) | ~1266 pixels |
| cx | Principal point (x) | ~816 pixels |
| cy | Principal point (y) | ~491 pixels |

**Note:** nuScenes provides already-undistorted images, so no distortion coefficients are needed.

### 6.3 Extrinsic Calibration

Extrinsics define the transformation from the sensor frame to the ego vehicle frame:

```
T_ego_sensor = [R | t]     4x4 transformation matrix
               [0 | 1]

Where:
  R = 3x3 rotation matrix (from quaternion in calibrated_sensor.json)
  t = 3x1 translation vector (from translation in calibrated_sensor.json)
```

#### Computing the Full Projection (3D World to 2D Pixel)

For BEVFormer's spatial cross-attention, the full projection chain is:

```python
# 1. World to ego
T_ego_global = np.eye(4)
T_ego_global[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
T_ego_global[:3, 3] = ego_pose['translation']
T_global_ego = np.linalg.inv(T_ego_global)

# 2. Ego to camera
T_cam_ego = np.eye(4)
T_cam_ego[:3, :3] = Quaternion(calibrated_sensor['rotation']).rotation_matrix
T_cam_ego[:3, 3] = calibrated_sensor['translation']
T_ego_cam = np.linalg.inv(T_cam_ego)

# 3. Full transformation: world to camera
T_cam_global = T_ego_cam @ T_global_ego

# 4. Project to image
point_cam = T_cam_global @ point_world_homogeneous
point_img = K @ point_cam[:3]
pixel = point_img[:2] / point_img[2]  # Perspective division
```

### 6.4 Calibration Quality

- nuScenes calibration is performed once per log sequence (not per frame)
- Calibration accuracy is typically within ~1 pixel reprojection error
- For BEVFormer, calibration errors propagate to BEV feature misalignment
- Consider data augmentation that simulates calibration perturbation for robustness

### 6.5 BEVFormer-Specific Calibration Usage

BEVFormer uses calibration in two key places:

1. **Spatial Cross-Attention:** Projects 3D reference points `(x, y, z)` in the BEV grid to 2D pixel locations `(u, v)` in each camera image
2. **Temporal Self-Attention:** Uses ego-pose difference between consecutive frames to spatially align BEV features

The calibration matrices are typically precomputed and passed as part of the data pipeline metadata (not loaded per-sample during training).

---

## 7. Data Quality Considerations

### 7.1 Known Issues

- Some frames have slight timestamp misalignment between cameras (~50ms)
- A small number of annotations may have incorrect velocity labels
- Night scenes have lower annotation density due to visibility limitations
- Some rare object categories have very few training examples

### 7.2 Data Filtering

BEVFormer's data pipeline typically filters:
- Frames where ego-pose is unavailable or unreliable
- Samples at the boundaries of scenes (where temporal pairs cannot be formed)
- Annotations with zero LiDAR points (fully occluded objects, excluded from evaluation)

### 7.3 Recommended Preprocessing

```bash
# Generate info files for BEVFormer training
python tools/create_data.py nuscenes \
    --root-path /data/nuscenes \
    --out-dir /data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-trainval \
    --canbus /data/nuscenes
```

This creates pickle files containing:
- Per-sample metadata (calibration, ego-pose, file paths)
- Annotation data in a format ready for the data loader
- Temporal sample pairs for consecutive frame loading
