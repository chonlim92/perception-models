# CRAFT: Data Collection and Sensor Configuration

## Dataset and Sensor Specifications

---

## 1. nuScenes Dataset Overview

### Dataset Summary

CRAFT is trained and evaluated on the nuScenes dataset, which is the standard benchmark for 3D object detection in autonomous driving. The dataset provides synchronized multi-modal sensor data from a full autonomous driving sensor suite.

| Property | Value |
|----------|-------|
| Total scenes | 1,000 |
| Scene duration | ~20 seconds each |
| Training samples | 28,130 keyframes |
| Validation samples | 6,019 keyframes |
| Test samples | 6,008 keyframes |
| Annotation frequency | 2 Hz (keyframes) |
| Sensor data frequency | 12-13 Hz (sweeps) |
| Location | Boston and Singapore |
| Driving conditions | Day, night, rain, construction zones |
| Annotated object classes | 10 (for detection task) |

### Object Classes for Detection

1. **Car** - Standard passenger vehicles
2. **Truck** - Large commercial vehicles
3. **Bus** - Public transit and coach buses
4. **Trailer** - Articulated trailers
5. **Construction Vehicle** - Excavators, cranes, bulldozers
6. **Pedestrian** - People walking, standing, or running
7. **Motorcycle** - Two-wheeled motorized vehicles with riders
8. **Bicycle** - Human-powered two-wheeled vehicles
9. **Traffic Cone** - Road construction markers
10. **Barrier** - Road barriers and dividers

### Dataset Splits

```
nuScenes v1.0
├── v1.0-trainval/
│   ├── train: 700 scenes (28,130 keyframes)
│   └── val: 150 scenes (6,019 keyframes)
├── v1.0-test/
│   └── test: 150 scenes (6,008 keyframes)
└── v1.0-mini/
    └── mini: 10 scenes (404 keyframes, for development)
```

---

## 2. Camera Specifications

### Hardware Configuration

The nuScenes vehicle is equipped with **6 cameras** providing full 360-degree surround coverage:

| Camera | Position | Horizontal FOV | Orientation |
|--------|----------|---------------|-------------|
| CAM_FRONT | Front windshield, center | ~70° | Forward |
| CAM_FRONT_LEFT | Front left A-pillar | ~70° | 55° left of forward |
| CAM_FRONT_RIGHT | Front right A-pillar | ~70° | 55° right of forward |
| CAM_BACK | Rear window, center | ~110° | Backward |
| CAM_BACK_LEFT | Rear left quarter | ~70° | 110° left of forward |
| CAM_BACK_RIGHT | Rear right quarter | ~70° | 110° right of forward |

### Camera Technical Specifications

| Parameter | Value |
|-----------|-------|
| Sensor type | CMOS image sensor |
| Resolution | 1600 x 900 pixels |
| Color depth | 24-bit RGB (8 bits per channel) |
| Frame rate | 12 Hz |
| Image format | JPEG (stored as .jpg) |
| Lens type | Wide-angle automotive lens |
| Dynamic range | High Dynamic Range (HDR) capable |
| Trigger | Hardware-triggered for synchronization |
| Mounting height | ~1.5m above ground (varies by position) |

### Image Characteristics

- **Exposure:** Auto-exposure with HDR fusion for handling high-contrast scenes
- **White balance:** Auto white balance calibrated for outdoor driving
- **Distortion:** Barrel distortion present (correctable via intrinsic parameters)
- **Rolling shutter:** Present; compensated during calibration
- **Artifacts:** Occasional lens flare, rain drops, sun glare (representative of real driving)

### Camera Intrinsic Parameters (Typical)

```python
# Example intrinsic matrix for CAM_FRONT
K = [[1266.417, 0.0,      816.267],
     [0.0,      1266.417, 491.507],
     [0.0,      0.0,      1.0    ]]

# Distortion coefficients (radial + tangential)
distortion = [k1, k2, p1, p2, k3]
```

### Coverage Geometry

The 6 cameras provide overlapping fields of view that together cover the full 360-degree azimuth around the vehicle:

```
              CAM_FRONT (70°)
                  ▲
                 /|\
                / | \
   CAM_FRONT_LEFT  |  CAM_FRONT_RIGHT
        (70°)  \  |  /  (70°)
                \ | /
     CAM_BACK_LEFT | CAM_BACK_RIGHT
        (70°)     |    (70°)
                  ▼
             CAM_BACK (110°)
```

Adjacent cameras have approximately 10-15 degrees of overlap to ensure no blind spots.

---

## 3. Radar Specifications

### Hardware Configuration

The nuScenes vehicle is equipped with **5 Continental ARS 408-21 radar sensors**:

| Radar | Position | Orientation | Primary Coverage |
|-------|----------|-------------|-----------------|
| RADAR_FRONT | Front bumper, center | Forward | 0° ± 60° |
| RADAR_FRONT_LEFT | Front left corner | 60° left | Side/forward-left |
| RADAR_FRONT_RIGHT | Front right corner | 60° right | Side/forward-right |
| RADAR_BACK_LEFT | Rear left corner | 150° left | Rear-left |
| RADAR_BACK_RIGHT | Rear right corner | 150° right | Rear-right |

### Radar Technical Specifications

| Parameter | Value |
|-----------|-------|
| Operating frequency | 77 GHz (W-band) |
| Modulation | FMCW (Frequency Modulated Continuous Wave) |
| Range | 0.2 - 250 m (long range mode) |
| Range resolution | ~0.4 m |
| Range accuracy | ±0.1 m |
| Velocity range | -400 to +200 km/h (radial) |
| Velocity resolution | ~0.12 m/s |
| Velocity accuracy | ±0.1 m/s |
| Azimuth FOV | ±60° (detection), ±9° (tracking, long range) |
| Azimuth resolution | ~1.5° - 3.0° (varies with mode) |
| Elevation FOV | Limited (±5° typical) |
| Elevation resolution | Not resolved (2D radar) |
| Update rate | 13 Hz |
| Output format | Detected object list (not raw spectrum) |

### Radar Measurement Parameters per Detection

Each radar detection provides the following measurements:

| Measurement | Description | Unit |
|-------------|-------------|------|
| x | Longitudinal distance | meters |
| y | Lateral distance | meters |
| z | Height (estimated, low accuracy) | meters |
| vx | Longitudinal velocity (compensated) | m/s |
| vy | Lateral velocity (compensated) | m/s |
| vx_comp | Ego-motion compensated vx | m/s |
| vy_comp | Ego-motion compensated vy | m/s |
| RCS | Radar Cross Section | dBsm |
| dynProp | Dynamic property (moving/stationary) | categorical |
| pdh0 | Probability of false alarm | probability |

### Radar Point Cloud Characteristics

- **Sparsity:** Typically 30-100 detections per frame per radar (vs. ~300,000 points for LiDAR)
- **2D nature:** No reliable elevation information (points are projected to ground plane)
- **Noise:** Higher false alarm rate than LiDAR, including ghost targets
- **Velocity:** Direct radial velocity measurement is the key advantage
- **RCS variation:** Provides information about target size/material

### Radar Coordinate System

```
        +x (forward)
         ^
         |
  +y <---+--- -y
  (left)  |  (right)
         |
         v
        -x (backward)
```

Height (z) is approximately at sensor mounting height with limited accuracy.

---

## 4. Sensor Synchronization

### Temporal Synchronization

All sensors in the nuScenes dataset are hardware-synchronized using a common clock:

| Aspect | Specification |
|--------|--------------|
| Synchronization method | PTP (Precision Time Protocol) + hardware trigger |
| Camera trigger frequency | 12 Hz |
| Radar output frequency | 13 Hz |
| Maximum inter-sensor timestamp difference | < 50 ms at keyframes |
| Keyframe annotation rate | 2 Hz |
| Interpolation between keyframes | Supported via ego-pose |

### Synchronization Architecture

```
┌─────────────────┐
│  Master Clock   │
│  (GPS + PTP)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│Camera │ │ Radar │
│Trigger│ │ Sync  │
│ 12 Hz │ │ 13 Hz │
└───────┘ └───────┘
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│6 Cams │ │5 Radar│
│Capture│ │Output │
└───────┘ └───────┘
```

### Keyframe Selection

- Keyframes are selected at 2 Hz from the 12 Hz camera stream
- At each keyframe, the closest radar sweep (within 50 ms) is associated
- Annotations are provided only at keyframes
- Intermediate frames (sweeps) can be used for temporal aggregation

### Temporal Aggregation for Radar

Due to radar sparsity, CRAFT (and many radar-based methods) aggregates radar detections across multiple sweeps:

```python
# Typical radar accumulation strategy
n_sweeps = 6  # Accumulate 6 radar sweeps (~0.5 seconds)
# Each sweep is transformed to the current keyframe coordinate system
# using ego-motion compensation (vehicle odometry + GPS/IMU)
```

This increases the radar point density from ~100 to ~600 points per scene while introducing motion compensation requirements.

---

## 5. Calibration Data

### Extrinsic Calibration

Each sensor has a 4x4 homogeneous transformation matrix defining its pose relative to the vehicle reference frame (ego frame):

```python
# Sensor-to-ego transformation
T_sensor_to_ego = {
    "translation": [x, y, z],          # meters
    "rotation": [w, x, y, z],          # quaternion (scalar-first)
    "token": "unique_calibration_id"
}

# Example: RADAR_FRONT to ego
T_radar_front = {
    "translation": [3.412, 0.0, 0.5],   # 3.4m forward, centered, 0.5m high
    "rotation": [1.0, 0.0, 0.0, 0.0]    # Identity (pointing forward)
}
```

### Intrinsic Calibration (Cameras Only)

```python
camera_intrinsic = {
    "fx": 1266.417,    # Focal length x (pixels)
    "fy": 1266.417,    # Focal length y (pixels)
    "cx": 816.267,     # Principal point x (pixels)
    "cy": 491.507,     # Principal point y (pixels)
    "distortion": [k1, k2, p1, p2, k3]  # Distortion coefficients
}
```

### Coordinate Frame Transformations

To project radar points into camera images (essential for CRAFT's fusion):

```python
import numpy as np
from pyquaternion import Quaternion

def radar_to_image(point_radar, calib_radar, calib_camera, camera_intrinsic):
    """
    Transform a 3D radar point to 2D image coordinates.
    
    Args:
        point_radar: [x, y, z] in radar frame
        calib_radar: radar extrinsic calibration
        calib_camera: camera extrinsic calibration
        camera_intrinsic: 3x3 camera matrix
    
    Returns:
        [u, v] pixel coordinates in image
    """
    # Step 1: Radar frame -> Ego frame
    R_radar = Quaternion(calib_radar['rotation']).rotation_matrix
    t_radar = np.array(calib_radar['translation'])
    point_ego = R_radar @ point_radar + t_radar
    
    # Step 2: Ego frame -> Camera frame
    R_cam = Quaternion(calib_camera['rotation']).rotation_matrix
    t_cam = np.array(calib_camera['translation'])
    point_cam = R_cam.T @ (point_ego - t_cam)
    
    # Step 3: Camera frame -> Image plane
    # Only valid if point is in front of camera (z > 0)
    if point_cam[2] <= 0:
        return None  # Behind camera
    
    point_img = camera_intrinsic @ point_cam
    u = point_img[0] / point_img[2]
    v = point_img[1] / point_img[2]
    
    return [u, v]
```

### Ego-Motion Data

For temporal alignment and radar sweep accumulation:

```python
ego_pose = {
    "translation": [x, y, z],      # Global position (meters)
    "rotation": [w, x, y, z],      # Global orientation (quaternion)
    "timestamp": 1535385096904799   # Microseconds since epoch
}
```

### Calibration Quality

| Aspect | Typical Accuracy |
|--------|-----------------|
| Camera-to-ego rotation | < 0.1° |
| Camera-to-ego translation | < 1 cm |
| Radar-to-ego rotation | < 0.5° |
| Radar-to-ego translation | < 2 cm |
| Cross-sensor temporal alignment | < 5 ms |
| Ego-pose (GPS/IMU) position | < 10 cm |
| Ego-pose (GPS/IMU) heading | < 0.1° |

---

## 6. Data Preprocessing for CRAFT

### Camera Data Preprocessing

1. **Image loading:** Decode JPEG to RGB tensor (3 x 900 x 1600)
2. **Normalization:** ImageNet mean/std normalization
3. **Augmentation:** Random crop, color jitter, horizontal flip (training only)
4. **Resizing:** Typically resized to target resolution (e.g., 448 x 800 or 256 x 704)

### Radar Data Preprocessing

1. **Point loading:** Load radar detections for current keyframe
2. **Sweep accumulation:** Aggregate n_sweeps (typically 3-6) past radar frames
3. **Ego-motion compensation:** Transform past sweeps to current frame coordinates
4. **Feature extraction:** Extract per-point features [x, y, z, vx_comp, vy_comp, RCS]
5. **Filtering:** Remove stationary clutter (optional, based on dynProp)
6. **Coordinate transformation:** Convert to ego frame or BEV grid

### Data Format Summary

```python
# Single training sample structure
sample = {
    # Camera data: 6 images
    "images": {
        "CAM_FRONT": np.ndarray,       # (3, H, W)
        "CAM_FRONT_LEFT": np.ndarray,
        "CAM_FRONT_RIGHT": np.ndarray,
        "CAM_BACK": np.ndarray,
        "CAM_BACK_LEFT": np.ndarray,
        "CAM_BACK_RIGHT": np.ndarray,
    },
    # Radar data: accumulated point cloud
    "radar_points": np.ndarray,  # (N, 18) - N points, 18 features
    # Calibration
    "camera_intrinsics": dict,   # Per-camera intrinsic matrices
    "camera_extrinsics": dict,   # Per-camera extrinsic poses
    "radar_extrinsics": dict,    # Per-radar extrinsic poses
    # Annotations
    "gt_boxes": np.ndarray,      # (M, 9) - M objects [x,y,z,w,l,h,yaw,vx,vy]
    "gt_labels": np.ndarray,     # (M,) - Class indices
}
```
