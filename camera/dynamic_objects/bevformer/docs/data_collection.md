# BEVFormer: Data Collection and Dataset Guide

## Understanding the nuScenes Dataset from First Principles

This guide teaches you everything about the nuScenes dataset that BEVFormer uses -- from sensor setup and coordinate systems to download instructions and annotation formats. Written for engineers who know deep learning but are new to autonomous driving datasets.

---

## 1. What Is nuScenes?

### 1.1 History and Significance

nuScenes (short for "nuTonomy Scenes") was released in 2019 by nuTonomy (later acquired by Aptiv/Motional). It was the first large-scale autonomous driving dataset to provide:
- Full 360-degree camera coverage (6 cameras)
- Synchronized multi-modal data (cameras, LiDAR, radar, GPS/IMU)
- 3D bounding box annotations with velocity and attributes
- Temporal tracking annotations (same object across frames)

### 1.2 Why nuScenes Became the Standard for Camera-Based 3D Detection

| Dataset | Cameras | 360-deg | Velocity Labels | Temporal Tracking | BEV methods built on it |
|---------|---------|---------|-----------------|-------------------|-----------------------|
| KITTI (2012) | 2 (stereo) | No (front only) | No | No | Few |
| Waymo Open (2019) | 5 | Yes | No (must derive) | Yes | Some |
| **nuScenes (2019)** | **6** | **Yes** | **Yes** | **Yes** | **Most** (BEVFormer, BEVDet, PETR, etc.) |
| Argoverse 2 (2021) | 7 | Yes | No | Yes | Growing |

nuScenes became the standard because: (a) full 360 coverage matches real deployment, (b) velocity annotations enable temporal model evaluation, (c) the evaluation server and metrics (NDS) became widely adopted, and (d) the dataset size is manageable (not petabytes like Waymo).

### 1.3 Dataset Statistics

| Property | Value |
|----------|-------|
| Total driving scenes | 1,000 (each ~20 seconds) |
| Annotated keyframes | 40,000 (at 2 Hz) |
| Total camera frames (sweeps) | ~1,400,000 (at 12 Hz) |
| Number of cameras | 6 (full 360-degree surround) |
| LiDAR | 1 (Velodyne HDL-32E, 32-beam, 20 Hz) |
| Radar | 5 (Continental ARS 408-21) |
| GPS/IMU | Yes (Novatel SPAN-CPT) |
| Locations | Boston (USA), Singapore |
| Conditions | Day, night, rain, overcast |
| Annotation classes | 23 total, 10 used for detection task |
| Data split | Train: 700 scenes / Val: 150 / Test: 150 |
| Total annotations | ~1.4 million 3D bounding boxes |

---

## 2. Sensor Setup

### 2.1 Vehicle and Camera Arrangement

```
                    TOP-DOWN VIEW OF VEHICLE
    ================================================================
    
                        CAM_FRONT (70 deg FOV)
                            |
                        +---+---+
                       /    |    \
                      /     |     \
    CAM_FRONT_LEFT   /      |      \   CAM_FRONT_RIGHT
    (70 deg FOV)    /       |       \  (70 deg FOV)
                   /        |        \
                  /    +---------+    \
                 /     |         |     \
                |      |  EGO    |      |
                |      | VEHICLE |      |
                |      |    *    |      |    * = origin (rear axle center)
                 \     |         |     /
                  \    +---------+    /
                   \        |        /
    CAM_BACK_LEFT   \       |       /   CAM_BACK_RIGHT
    (70 deg FOV)     \      |      /    (70 deg FOV)
                      \     |     /
                       \    |    /
                        +---+---+
                            |
                        CAM_BACK (110 deg FOV)
    
    ================================================================
    
    Camera positions (approximate, relative to rear axle):
    
    Camera           X (forward)   Y (left)   Z (up)   Heading
    ------           -----------   --------   ------   -------
    CAM_FRONT         1.70 m        0.00 m    1.51 m   0 deg
    CAM_FRONT_LEFT    1.69 m        0.46 m    1.49 m   +55 deg
    CAM_FRONT_RIGHT   1.69 m       -0.46 m    1.49 m   -55 deg
    CAM_BACK          0.02 m        0.00 m    1.57 m   180 deg
    CAM_BACK_LEFT     1.03 m        0.48 m    1.56 m   +110 deg
    CAM_BACK_RIGHT    1.03 m       -0.48 m    1.56 m   -110 deg
```

### 2.2 Field of View and Coverage

```
    360-DEGREE COVERAGE MAP (viewed from above):
    
                    0 deg (forward)
                        |
              FRONT_LEFT | FRONT_RIGHT
                   \     |     /
                    \  FRONT  /
                     \ (70) /
              (70)    \   /    (70)
                 ------[*]------
              (70)    /   \    (70)
                     / (110)\
                    / BACK   \
                   /     |    \
              BACK_LEFT  | BACK_RIGHT
                        |
                    180 deg (backward)
    
    Overlap regions: ~10-15 degrees between adjacent cameras
    Total coverage: Full 360 degrees (no blind spots in horizontal plane)
    Vertical FOV: approximately 40-50 degrees per camera
```

### 2.3 Why This Specific Setup?

1. **6 cameras (not 4 or 8):** 6 cameras with 70-degree FOV (plus one 110-degree rear) is the minimum for complete 360-degree coverage with reasonable overlap. Fewer cameras would leave blind spots; more would increase cost and data bandwidth without proportional benefit.

2. **Front camera at 70 deg (not wider):** Narrower FOV means higher pixel density at distance, which is critical for detecting far objects ahead (where driving decisions are most time-critical).

3. **Back camera at 110 deg:** The rear has a larger gap between the two back cameras. A wider FOV compensates for this, ensuring no blind spot behind the vehicle.

4. **Cameras mounted high (~1.5m):** Reduces occlusion from the hood and provides better viewing angles for nearby objects.

### 2.4 Camera Specifications

| Property | Value |
|----------|-------|
| Resolution | 1600 x 900 pixels |
| Sensor type | 1/2.7" CMOS |
| Frame rate | 12 Hz (keyframes selected at 2 Hz) |
| Color depth | 8-bit RGB (24-bit total) |
| File format | JPEG (lossy compression) |
| Lens type | Wide-angle, varies by position |
| Images undistorted | Yes (factory calibrated, distortion removed) |

---

## 3. Coordinate Systems (In-Depth)

Understanding coordinate systems is CRITICAL for BEVFormer because the model constantly transforms between them (image pixels <-> camera frame <-> ego frame <-> global frame).

### 3.1 Global Frame

```
    Global Frame (fixed to Earth/map):
    
         North (Y)
         ^
         |
         |
         +-------> East (X)
        /
       /
      v
    Down (-Z)    ...but convention is Z-up, so:
    
    Actually:   X = East
                Y = North  
                Z = Up
    
    Origin: An arbitrary fixed point in the map
    Used for: Absolute positioning, HD map alignment
```

The global frame does NOT move with the vehicle. It is a fixed reference anchored to the world.

### 3.2 Ego Vehicle Frame

```
    Ego Vehicle Frame (moves with car):
    
         Z (Up)
         ^
         |
         |
         +-------> Y (Left, driver side for right-hand-drive)
        /
       /
      v
    X (Forward, direction of travel)
    
    Wait -- that's ambiguous. Let me be precise:
    
    X-axis: Points FORWARD (direction the car faces)
    Y-axis: Points LEFT (driver's left for left-hand drive)
    Z-axis: Points UP
    
    Origin: Center of rear axle, projected onto ground plane
```

**Why rear axle?** The rear axle is the center of rotation for a car (front wheels steer, rear wheels follow). This makes kinematic calculations simpler.

### 3.3 Camera Frame

```
    Camera Frame (standard computer vision convention):
    
    Looking OUT through the camera:
    
         Y (Down)
         ^
         |
         |  Z (Forward = optical axis, into the scene)
         | /
         |/
         +-------> X (Right)
    
    Origin: Camera optical center (pinhole)
```

**IMPORTANT:** The camera frame has Y pointing DOWN. This is different from the ego frame (Y-left) and global frame (Y-north). This convention comes from image processing where row 0 is at the top.

### 3.4 Image Frame (Pixel Coordinates)

```
    Image Frame:
    
    (0, 0) -----> u (column, 0 to 1599)
       |
       |
       v
       v (row, 0 to 899)
    
    Origin: Top-left corner of the image
    u-axis: Points right (column index)
    v-axis: Points down (row index)
```

### 3.5 Transformation Chain

To go from a 3D point in the world to a 2D pixel in an image:

```
    Global Frame          Ego Frame            Camera Frame         Image Frame
    (world coords)  -->  (vehicle coords)  -->  (camera coords)  -->  (pixels)
    
    [x_global]     ego_pose       [x_ego]      calib_sensor     [x_cam]      intrinsics    [u]
    [y_global]  ------------>     [y_ego]   -------------->     [y_cam]   ------------>    [v]
    [z_global]   (inverse)        [z_ego]    (inverse)          [z_cam]    (project)
    [ 1      ]                    [ 1   ]                       [ 1   ]
```

For BEVFormer, the relevant transformation is: **Ego Frame -> Image Frame** (project BEV positions to camera pixels).

---

## 4. Camera Intrinsic and Extrinsic Matrices

### 4.1 Intrinsics: The Pinhole Camera Model

The intrinsic matrix describes how a camera projects 3D points (in camera coordinates) to 2D image pixels.

**The Pinhole Camera Model:**

```
    Physical picture:
    
    3D Point P -------- lens/pinhole -------- Image Plane
    (X, Y, Z)              |                  (u, v)
                        focal length f
                           |<----->|
    
    By similar triangles:
        u = f * X / Z + cx
        v = f * Y / Z + cy
    
    Where:
        f = focal length (distance from pinhole to image plane, in pixels)
        cx, cy = principal point (where the optical axis hits the image)
```

**The 3x3 Intrinsic Matrix K:**

```
    K = [fx   0   cx]
        [ 0  fy   cy]
        [ 0   0    1]
    
    Typical values for nuScenes CAM_FRONT:
    K = [1266.4    0     816.3]
        [   0   1266.4   491.5]
        [   0      0       1  ]
```

| Parameter | Symbol | Meaning | Typical Value |
|-----------|--------|---------|---------------|
| Focal length X | fx | Horizontal magnification (pixels/radian) | ~1266 |
| Focal length Y | fy | Vertical magnification (usually = fx) | ~1266 |
| Principal point X | cx | Image center column (should be ~W/2) | ~816 |
| Principal point Y | cy | Image center row (should be ~H/2) | ~491 |

**Worked example: Project a 3D point to pixel**

```
Point in camera frame: P_cam = (2.0, -0.5, 10.0) meters
  (2m to the right, 0.5m above optical axis, 10m ahead)

Using K = [[1266, 0, 816], [0, 1266, 491], [0, 0, 1]]:

p = K @ [2.0, -0.5, 10.0]^T = [1266*2 + 816*10, 1266*(-0.5) + 491*10, 10]
                                = [2532 + 8160, -633 + 4910, 10]
                                = [10692, 4277, 10]

Pixel: u = 10692 / 10 = 1069.2
       v = 4277 / 10 = 427.7

So the point appears at pixel (1069, 428) -- right of center, above center.
This makes sense: the point is to the right and above the optical axis.
```

### 4.2 Extrinsics: Rigid Body Transformation

The extrinsic parameters describe where the camera is physically mounted on the vehicle.

**Components:**
- Rotation R (3x3 matrix): How the camera is oriented relative to the ego frame
- Translation t (3x1 vector): Where the camera is positioned relative to the ego frame

**The 4x4 Homogeneous Transformation Matrix:**

```
T_sensor_to_ego = [R  t]     transforms points FROM sensor TO ego
                  [0  1]

T_ego_to_sensor = T_sensor_to_ego^(-1)    transforms points FROM ego TO sensor
```

**In nuScenes,** `calibrated_sensor.json` provides `rotation` (as quaternion) and `translation` that represent the sensor-to-ego transformation.

**Quaternion to Rotation Matrix:**

nuScenes stores rotations as quaternions `[w, x, y, z]`. Convert to rotation matrix:

```python
from pyquaternion import Quaternion

q = Quaternion(w=0.5077, x=-0.4973, y=0.4978, z=-0.4972)
R = q.rotation_matrix  # 3x3 numpy array
```

### 4.3 The Full Projection Pipeline (BEVFormer Uses This)

Given a 3D point in the ego frame, project it to pixel coordinates in a specific camera:

```python
import numpy as np

def project_ego_to_pixel(point_ego, T_sensor_to_ego, K):
    """
    Project a 3D point from ego frame to image pixel coordinates.
    
    Args:
        point_ego: [x, y, z] in ego vehicle frame (meters)
        T_sensor_to_ego: 4x4 transformation (sensor->ego)
        K: 3x3 camera intrinsic matrix
    
    Returns:
        (u, v): pixel coordinates, or None if behind camera
    """
    # Step 1: Ego frame -> Camera frame
    T_ego_to_sensor = np.linalg.inv(T_sensor_to_ego)
    point_homo = np.array([*point_ego, 1.0])  # [x, y, z, 1]
    point_cam = T_ego_to_sensor @ point_homo   # [x_cam, y_cam, z_cam, 1]
    
    # Step 2: Check if point is in front of camera
    if point_cam[2] <= 0:
        return None  # Behind camera -- invisible
    
    # Step 3: Camera frame -> Pixel (perspective projection)
    point_img = K @ point_cam[:3]  # [u*z, v*z, z]
    u = point_img[0] / point_img[2]  # perspective divide
    v = point_img[1] / point_img[2]
    
    # Step 4: Check if within image bounds
    if 0 <= u < 1600 and 0 <= v < 900:
        return (u, v)
    else:
        return None  # Outside image bounds
```

### 4.4 The Combined lidar2img Matrix

BEVFormer precomputes a single 4x4 matrix that combines ego->camera and camera->pixel in one step. In the code, this is often called `lidar2img` (because the ego frame is aligned with the LiDAR frame in nuScenes):

```python
# Precomputed for efficiency
lidar2img = K_4x4 @ T_ego_to_cam  # 4x4 combined projection matrix

# Then projection is just:
p_homo = lidar2img @ [x, y, z, 1]^T
u = p_homo[0] / p_homo[2]
v = p_homo[1] / p_homo[2]
```

---

## 5. Data Format

### 5.1 Relational Structure

nuScenes uses a relational database structure stored as JSON files. The relationships:

```
    scene (1000 scenes, ~20 sec each)
      |
      +-- has many --> sample (keyframes, 40 per scene at 2 Hz)
           |
           +-- has many --> sample_data (one per sensor per frame)
           |    |
           |    +-- links to --> calibrated_sensor (calibration for that sensor)
           |    +-- links to --> ego_pose (vehicle pose at that timestamp)
           |    +-- has file --> filename (path to image/pointcloud)
           |
           +-- has many --> sample_annotation (3D boxes for objects in this frame)
                |
                +-- links to --> instance (tracking: same physical object across time)
                +-- links to --> attribute (behavioral state: moving/parked/etc)
                +-- links to --> visibility (what fraction is visible)
```

### 5.2 Key JSON Files

**scene.json** -- Top-level container (one entry per driving scene):
```json
{
    "token": "cc8c0bf57f984915a77078b10eb33198",
    "log_token": "7e25a2c8ea1f41c5b0dce7b8a0449e7e",
    "nbr_samples": 39,
    "first_sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "last_sample_token": "9fad1d386f3e4a9fb51f259e5fdd3a5f",
    "name": "scene-0001",
    "description": "Construction site, rainy weather"
}
```

**sample.json** -- A keyframe (synchronized across all sensors):
```json
{
    "token": "ca9a282c9e77460f8360f564131a8af5",
    "timestamp": 1532402927647951,
    "prev": "",
    "next": "39586f9d59004284a7114a68825e8eec",
    "scene_token": "cc8c0bf57f984915a77078b10eb33198"
}
```

The `prev` and `next` fields link consecutive keyframes. BEVFormer uses these to find temporal pairs for temporal self-attention.

**sample_data.json** -- Per-sensor data for each timestamp:
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
    "filename": "samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg"
}
```

**calibrated_sensor.json** -- Camera calibration (one per camera-log pair):
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

**ego_pose.json** -- Vehicle position/orientation at each timestamp:
```json
{
    "token": "5ace90b379af485b9dcb1584b01e7212",
    "timestamp": 1532402927612460,
    "rotation": [0.5720, -0.0016, 0.0130, -0.8201],
    "translation": [410.77, 1137.28, 0.0]
}
```

**sample_annotation.json** -- 3D bounding boxes:
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

### 5.3 Annotation Format Details

| Field | Format | Meaning |
|-------|--------|---------|
| translation | [x, y, z] | Center of box in GLOBAL frame (meters) |
| size | [w, l, h] | Width, length, height of box (meters) |
| rotation | [w, x, y, z] | Quaternion orientation in GLOBAL frame |
| velocity | [vx, vy] | Velocity in GLOBAL frame (m/s), 2D only |
| num_lidar_pts | int | LiDAR points inside box (0 = fully occluded) |
| instance_token | str | Unique ID for this physical object across time |
| prev/next | str | Links to same object in previous/next frame |

**Important notes:**
- Annotations with `num_lidar_pts=0` are excluded from evaluation (object is fully occluded)
- `velocity` is only available for keyframes and may be [0,0] for static objects
- `size` uses [width, length, height] convention (width = shorter dimension)

---

## 6. Download Instructions

### 6.1 Step-by-Step Download

**Step 1: Create Account**
1. Go to https://www.nuscenes.org/
2. Click "Sign Up" (free for academic and commercial use)
3. Accept the Terms of Use

**Step 2: Choose What to Download**

For BEVFormer (camera-only), you need:

| Archive | Size | Required? | Contains |
|---------|------|-----------|----------|
| v1.0-trainval_meta.tgz | ~300 MB | YES | All JSON metadata + annotations |
| v1.0-trainval01_blobs.tgz through v1.0-trainval10_blobs.tgz | ~70 GB total | YES | Camera images |
| LiDAR archives | ~50 GB | NO | Point clouds (not needed for BEVFormer) |
| Radar archives | ~10 GB | NO | Radar data (not needed) |
| can_bus.zip | ~300 MB | YES | CAN bus data for ego-motion |
| v1.0-mini.tgz | ~4 GB | OPTIONAL | Small subset for testing |

**Minimum download for BEVFormer: ~72 GB** (metadata + camera images + can_bus)

**Step 3: Download**

```bash
# Create data directory
mkdir -p /data/nuscenes

# Option A: Browser download (slow but simple)
# Go to nuscenes.org/download, click each archive, save to /data/nuscenes/

# Option B: Command line (if you have the download links)
cd /data/nuscenes
wget <meta_link> -O v1.0-trainval_meta.tgz
wget <blob01_link> -O v1.0-trainval01_blobs.tgz
# ... repeat for all blob archives

# Option C: nuScenes devkit
pip install nuscenes-devkit
python -c "
from nuscenes.utils.download import download
download('v1.0-trainval', '/data/nuscenes')
"
```

**Step 4: Extract**

```bash
cd /data/nuscenes

# Extract metadata (annotations, calibration, etc.)
tar -xzf v1.0-trainval_meta.tgz

# Extract camera images (one archive at a time to manage disk space)
for i in $(seq -w 1 10); do
    tar -xzf v1.0-trainval${i}_blobs.tgz
    echo "Extracted blob archive ${i}"
done

# Extract CAN bus data
unzip can_bus.zip
```

**Step 5: Verify**

```bash
# Check directory structure
ls /data/nuscenes/
# Expected: maps/ samples/ sweeps/ v1.0-trainval/ can_bus/

# Check camera images exist
ls /data/nuscenes/samples/CAM_FRONT/ | head -5
# Expected: n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243012465.jpg ...

# Count images (should be ~40,000 keyframes x 6 cameras = ~240,000)
find /data/nuscenes/samples/CAM_* -name "*.jpg" | wc -l
# Expected: approximately 240,000

# Verify metadata
python -c "
from nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='/data/nuscenes')
print(f'Scenes: {len(nusc.scene)}')
print(f'Samples: {len(nusc.sample)}')
print(f'Sample data: {len(nusc.sample_data)}')
"
# Expected: Scenes: 1000, Samples: ~40000, Sample data: ~400000+
```

---

## 7. Data Splits

### 7.1 Official Splits

| Split | Scenes | Keyframes | Purpose |
|-------|--------|-----------|---------|
| Train | 700 | ~28,130 | Model training |
| Validation | 150 | ~6,019 | Local evaluation, hyperparameter tuning |
| Test | 150 | ~6,008 | Leaderboard submission only (no local eval) |

### 7.2 Mini Split (for Development)

| Split | Scenes | Keyframes | Purpose |
|-------|--------|-----------|---------|
| Mini-train | 7 | ~283 | Quick code verification |
| Mini-val | 3 | ~123 | Quick evaluation test |

Use the mini split to verify your data pipeline works before downloading 70+ GB.

### 7.3 Geographic Diversity

- **Boston scenes:** Urban driving, typical US road infrastructure
- **Singapore scenes:** Dense urban, different driving conventions (left-hand drive), tropical weather

The splits are geographically balanced -- both cities appear in train, val, and test.

---

## 8. CAN Bus Data (Ego-Motion)

### 8.1 What Is CAN Bus?

CAN (Controller Area Network) is the internal communication bus in modern vehicles. It carries real-time data from the vehicle's sensors: wheel speed, steering angle, accelerometer, gyroscope, GPS, etc.

### 8.2 Why BEVFormer Needs It

BEVFormer's temporal self-attention must align previous BEV features to the current frame. This requires knowing exactly how the ego vehicle moved between frames. The CAN bus provides precise ego-motion information (translation + rotation).

### 8.3 What nuScenes Provides

The `can_bus` expansion pack provides 18-dimensional vectors per frame:

```python
can_bus = np.array([
    x, y, z,           # Global position (meters)
    qw, qx, qy, qz,   # Global orientation (quaternion)
    vx, vy, vz,        # Linear velocity (m/s)
    ax, ay, az,         # Linear acceleration (m/s^2)
    wx, wy, wz,         # Angular velocity (rad/s)
    speed               # Scalar speed (m/s)
])  # shape: (18,)
```

### 8.4 Download and Integration

```bash
# Download CAN bus data
wget https://www.nuscenes.org/data/can_bus.zip
unzip can_bus.zip -d /data/nuscenes/

# Verify
ls /data/nuscenes/can_bus/
# Expected: scene-0001.json, scene-0002.json, ...
```

---

## 9. Data Preprocessing for BEVFormer

### 9.1 Generating Info Files

BEVFormer requires preprocessed pickle files that combine all metadata into an efficient format:

```bash
python tools/create_data.py nuscenes \
    --root-path /data/nuscenes \
    --out-dir /data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-trainval \
    --canbus /data/nuscenes

# Output files:
#   /data/nuscenes/nuscenes_infos_temporal_train.pkl  (~1.5 GB)
#   /data/nuscenes/nuscenes_infos_temporal_val.pkl    (~0.3 GB)
```

### 9.2 What the Info Files Contain

Each entry in the info pickle represents one keyframe and contains:

```python
info = {
    'lidar_path': str,              # Reference path (for token matching)
    'token': str,                   # Unique sample identifier
    'cams': {                       # Per-camera information
        'CAM_FRONT': {
            'data_path': str,       # Path to JPEG image
            'sensor2ego_translation': [3],   # Camera position in ego frame
            'sensor2ego_rotation': [4],      # Camera orientation (quaternion)
            'ego2global_translation': [3],   # Ego position in global frame
            'ego2global_rotation': [4],      # Ego orientation (quaternion)
            'cam_intrinsic': [[3,3]],        # 3x3 intrinsic matrix
        },
        'CAM_FRONT_LEFT': {...},
        'CAM_FRONT_RIGHT': {...},
        'CAM_BACK': {...},
        'CAM_BACK_LEFT': {...},
        'CAM_BACK_RIGHT': {...},
    },
    'gt_boxes': np.array,           # (N, 9): [x,y,z,w,l,h,yaw,vx,vy] in ego frame
    'gt_names': np.array,           # (N,) class names
    'num_lidar_pts': np.array,      # (N,) LiDAR points per object
    'valid_flag': np.array,         # (N,) boolean: has enough points to evaluate?
    'prev': str,                    # Token of previous keyframe (for temporal)
    'next': str,                    # Token of next keyframe
    'can_bus': np.array,            # (18,) CAN bus data
}
```

### 9.3 How Temporal Pairs Are Formed

BEVFormer loads multiple consecutive frames. During data loading:

```python
# For current sample at time t:
#   Load current frame info
#   Follow 'prev' tokens to get previous frames: t-1, t-2, t-3
#   Each previous frame provides:
#     - ego_pose (for alignment transformation)
#     - This is used to compute the ego-motion matrix between frames
#
# The model processes these as a temporal queue:
#   queue = [frame_t-3, frame_t-2, frame_t-1, frame_t]
#   The BEV from t-1 is cached and aligned to t using ego-motion
```

---

## 10. Data Quality Considerations

### 10.1 Known Issues

1. **Timestamp misalignment:** Cameras are not perfectly synchronized. There can be up to ~50ms offset between cameras in the same keyframe. For a car moving at 30 m/s, this is ~1.5m of motion.

2. **Velocity label noise:** Some velocity annotations are derived from tracking and may be noisy, especially for objects that appear/disappear between frames.

3. **Night scene annotation density:** Fewer annotations in very dark scenes because human annotators cannot see occluded objects.

4. **Calibration stability:** Calibration is per-log (one driving session), not per-frame. Minor vibrations during driving are not accounted for.

### 10.2 Data Filtering in BEVFormer

The BEVFormer data pipeline filters:
- Annotations with `num_lidar_pts=0` (fully occluded, not evaluatable)
- Annotations outside the point cloud range `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]`
- First frames of scenes (no previous frame for temporal)
- Corrupt or missing images (rare)

### 10.3 Class Distribution (Imbalance)

```
Class                  # Train Annotations  Relative Frequency
car                    ~340,000             ████████████████████  48.6%
pedestrian             ~160,000             ████████████          22.9%
barrier                ~120,000             █████████             17.1%
traffic_cone           ~ 70,000             █████                 10.0%
truck                  ~ 65,000             █████                  9.3%
trailer                ~ 20,000             ██                     2.9%
motorcycle             ~ 15,000             █                      2.1%
bus                    ~ 12,000             █                      1.7%
bicycle                ~ 10,000             █                      1.4%
construction_vehicle   ~  7,000             █                      1.0%
```

The 48x imbalance between `car` and `construction_vehicle` motivates the use of class-balanced sampling (CBGS) during training.

---

## 11. Storage and Performance Recommendations

### 11.1 Storage Requirements

| Component | Size | Required? |
|-----------|------|-----------|
| Camera keyframes (samples/) | ~35 GB | Yes |
| Camera sweeps (sweeps/) | ~240 GB | Optional (denser temporal) |
| Metadata JSON | ~300 MB | Yes |
| CAN bus | ~300 MB | Yes |
| Preprocessed info files | ~2 GB | Generated locally |
| Model checkpoints | ~1 GB each | Generated during training |
| **Minimum total** | **~40 GB** | |
| **Recommended free space** | **~100 GB** | (includes working room) |

### 11.2 I/O Performance

BEVFormer loads 6 images per sample (each ~200KB JPEG). At batch_size=1 with 4 temporal frames, that is 24 image reads per iteration.

**Recommendations:**
- Use SSD storage (not HDD) -- random reads of 200KB files benefit enormously from SSD
- If using network storage (NFS), ensure >1 GB/s bandwidth
- Enable `persistent_workers=True` in dataloader to avoid re-spawning processes
- Use `num_workers=4` per GPU (increase if I/O is the bottleneck)

```bash
# Check if I/O is your bottleneck:
# If GPU utilization < 90% during training, I/O is likely the issue
nvidia-smi  # Check GPU utilization during training
iostat -x 1  # Check disk I/O saturation
```
