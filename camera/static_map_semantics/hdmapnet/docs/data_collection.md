# Data Collection and Preparation Guide for HDMapNet

This guide covers everything a PyTorch engineer needs to collect, prepare, and load
training data for HDMapNet. Whether you are working with nuScenes or building a custom
data pipeline, this document provides the practical details to get from raw sensor data
to model-ready tensors.

---

## 1. Overview: What Data Does HDMapNet Need?

HDMapNet predicts vectorized HD map elements in bird's-eye view (BEV) from surround
camera images. The model requires the following inputs and ground truth at training time:

**Model Inputs:**

| Input | Shape / Format | Description |
|-------|---------------|-------------|
| Multi-view camera images | 6 x (3, H, W) | RGB images from surround cameras |
| Camera intrinsic matrices | 6 x (3, 3) | Per-camera K matrix (focal length, principal point) |
| Camera extrinsic matrices | 6 x (4, 4) | Ego vehicle frame to camera frame transform |
| Ego pose | (4, 4) | World frame to ego vehicle frame transform |

**Ground Truth Targets:**

| Target | Shape / Format | Description |
|--------|---------------|-------------|
| Semantic BEV map | (C, H_bev, W_bev) | Binary masks per map class |
| Instance BEV map | (H_bev, W_bev) | Unique integer ID per polyline instance |
| Direction BEV map | (2, H_bev, W_bev) | Unit tangent vector (dx, dy) at each positive pixel |

The standard BEV grid is 200x200 pixels covering 60m x 30m around the ego vehicle
(0.3 m/pixel resolution in the longitudinal axis, 0.15 m/pixel lateral -- or symmetric
depending on configuration).

---

## 2. nuScenes Dataset

### 2.1 Dataset Statistics

| Property | Value |
|----------|-------|
| Total scenes | 1000 |
| Scene duration | ~20 seconds each |
| Keyframe rate | 2 Hz (40 keyframes per scene) |
| Total keyframes | ~40,000 |
| Training split | 700 scenes (~28,000 samples) |
| Validation split | 150 scenes (~6,000 samples) |
| Test split | 150 scenes (held out, no public labels) |
| Cities covered | Boston, Singapore (11 map areas) |
| Full download size | ~400 GB |
| Mini split (dev) | ~4 GB (10 scenes) |

### 2.2 Camera Configuration

nuScenes uses 6 cameras providing 360-degree surround coverage:

| Camera | Field of View | Horizontal Angle |
|--------|--------------|-----------------|
| CAM_FRONT | 70 deg | 0 deg (forward) |
| CAM_FRONT_LEFT | 70 deg | ~55 deg left |
| CAM_FRONT_RIGHT | 70 deg | ~55 deg right |
| CAM_BACK | 110 deg | 180 deg (rear) |
| CAM_BACK_LEFT | 70 deg | ~110 deg left |
| CAM_BACK_RIGHT | 70 deg | ~110 deg right |

All cameras capture at **1600 x 900** pixel resolution.

### 2.3 Other Sensors (Reference)

- **LiDAR**: 32-beam Velodyne, 20 Hz rotation, ~300k points/sweep
  - Not used in camera-only HDMapNet but useful for validation/debugging
- **RADAR**: 5 radars, typically unused for map prediction
- **IMU/GPS**: Provides ego pose, critical for map alignment

### 2.4 Map Annotations

The nuScenes map expansion provides vector maps for all 11 areas:

- **Lane dividers**: Center lines separating lanes
- **Pedestrian crossings**: Polygonal crosswalk regions (represented as polylines for HDMapNet)
- **Road boundaries**: Edge of drivable surface

Map elements are stored as **polylines** (ordered sequences of 2D points) in the
**world coordinate frame** (global UTM-like coordinates per city).

### 2.5 Download and Setup

```bash
# Directory structure after download
nuscenes/
    v1.0-trainval/          # Metadata JSON files
        sample.json
        sample_data.json
        ego_pose.json
        calibrated_sensor.json
        ...
    samples/                # Keyframe sensor data
        CAM_FRONT/
        CAM_FRONT_LEFT/
        CAM_FRONT_RIGHT/
        CAM_BACK/
        CAM_BACK_LEFT/
        CAM_BACK_RIGHT/
        LIDAR_TOP/
    sweeps/                 # Non-keyframe sensor data (higher rate)
    maps/                   # Vector map files (.json)
        expansion/          # Map expansion with detailed annotations
```

```bash
# Using the mini split for development (recommended first step)
pip install nuscenes-devkit

# In Python:
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/data/nuscenes')
```

---

## 3. Camera Setup and Coordinate Systems

### 3.1 Surround Camera Layout (Top-Down View)

```
                        FRONT
                     ___________
                    /           \
                   /  CAM_FRONT  \
                  /   (70 deg)    \
     CAM_FRONT  /                  \  CAM_FRONT
     _LEFT     /                    \    _RIGHT
    (70 deg)  /                      \  (70 deg)
             |                        |
             |      EGO VEHICLE       |
             |       (top view)       |
             |          +X            |
             |          |             |
             |     +Y<--+             |
             |                        |
    CAM_BACK  \                      /  CAM_BACK
     _LEFT     \                    /    _RIGHT
    (70 deg)    \                  /   (70 deg)
                 \   CAM_BACK    /
                  \  (110 deg)  /
                   \___________/

                       BACK
```

### 3.2 Coordinate Frames

HDMapNet operates across four coordinate frames. Understanding the transform chain is
critical for correct projection.

```
    World Frame              Ego Frame              Camera Frame           Image Frame
    (global map)         (vehicle body)           (optical center)        (pixel coords)
         |                     |                       |                       |
         |   T_world_to_ego   |   T_ego_to_cam       |   K (intrinsic)      |
         | -----------------> | ------------------> | ------------------>   |
         |    (ego_pose)      |    (extrinsic)       |   (projection)       |
```

**World Frame:**
- Origin: fixed global reference (per city map tile)
- Used for: map annotations, ego pose trajectories

**Ego Vehicle Frame:**
- Origin: center of rear axle (nuScenes convention)
- X: forward, Y: left, Z: up
- Changes every timestep as vehicle moves

**Camera Frame:**
- Origin: optical center of camera
- Z: forward (into scene), X: right, Y: down
- Each camera has its own frame

**Image Frame:**
- Origin: top-left corner of image
- u: right (columns), v: down (rows)
- Units: pixels

### 3.3 Transform Chain: World Point to Image Pixel

Given a 3D point `P_world` in world coordinates, the full projection to pixel `(u, v)`:

```python
import numpy as np

def world_to_pixel(P_world, ego_pose, extrinsic, intrinsic):
    """
    Project a world-frame 3D point to image pixel coordinates.

    Args:
        P_world: (3,) or (4,) point in world frame
        ego_pose: (4, 4) world-to-ego transform (inverse of ego pose)
        extrinsic: (4, 4) ego-to-camera transform
        intrinsic: (3, 3) camera intrinsic matrix K

    Returns:
        (u, v) pixel coordinates
    """
    # Homogeneous coordinates
    if P_world.shape[0] == 3:
        P_world = np.append(P_world, 1.0)

    # World -> Ego
    P_ego = ego_pose @ P_world  # (4,)

    # Ego -> Camera
    P_cam = extrinsic @ P_ego   # (4,)

    # Camera -> Image (perspective projection)
    P_img = intrinsic @ P_cam[:3]  # (3,)

    # Normalize by depth
    u = P_img[0] / P_img[2]
    v = P_img[1] / P_img[2]

    return u, v
```

### 3.4 Intrinsic Matrix K

The 3x3 camera intrinsic matrix encodes focal length and principal point:

```
K = | fx   0   cx |
    |  0  fy   cy |
    |  0   0    1 |
```

Where:
- `fx, fy`: focal lengths in pixels (typically 1250-1400 for nuScenes cameras)
- `cx, cy`: principal point (approximately image center: 800, 450)

### 3.5 Extrinsic Matrix (Ego to Camera)

The 4x4 extrinsic transforms points from ego frame to camera frame:

```
T_ego_to_cam = | R  t |    R: (3,3) rotation matrix
               | 0  1 |    t: (3,1) translation vector
```

In nuScenes, the calibrated sensor record stores the **sensor-to-ego** transform.
You must invert it to get ego-to-camera:

```python
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

# nuScenes provides sensor_to_ego
cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
sensor_to_ego = transform_matrix(
    cs_record['translation'],
    Quaternion(cs_record['rotation']),
    inverse=False
)
# Invert to get ego_to_sensor (what we need)
ego_to_camera = np.linalg.inv(sensor_to_ego)
```

---

## 4. Map Annotation Format

### 4.1 Vector Map Structure

HD map elements in nuScenes are stored as **vector polylines** -- ordered sequences
of 2D (x, y) points in the world coordinate frame.

```python
# Example: a lane divider polyline
lane_divider = {
    "token": "abc123...",
    "line": [
        [345.2, 1102.4],   # Point 0 (x, y) in world frame
        [345.8, 1104.1],   # Point 1
        [346.5, 1106.0],   # Point 2
        ...                 # Typically 10-100 points per element
    ]
}
```

### 4.2 Three Map Classes for HDMapNet

| Class Index | Map Element | Representation | Typical Count per Sample |
|-------------|------------|----------------|-------------------------|
| 0 | Lane Divider | Polyline | 5-30 instances |
| 1 | Pedestrian Crossing | Polygon boundary as polyline | 0-5 instances |
| 2 | Road Boundary | Polyline | 5-20 instances |

### 4.3 Querying Map Elements

The nuScenes map API provides spatial queries to retrieve elements near the ego vehicle:

```python
from nuscenes.map_expansion.map_api import NuScenesMap

nusc_map = NuScenesMap(dataroot='/data/nuscenes', map_name='singapore-onenorth')

# Get map elements within a bounding box around ego
patch_box = (ego_x, ego_y, patch_height, patch_width)  # center_x, center_y, h, w
patch_angle = ego_yaw_degrees

# Retrieve lane dividers as polylines
lane_records = nusc_map.get_records_in_patch(patch_box, ['lane_divider'], 'intersect')
```

### 4.4 Coordinate Transform: World to Ego-Centric

Map annotations are in world coordinates but HDMapNet needs them in ego-centric BEV.
The transformation:

```python
def transform_polyline_to_ego(polyline_world, ego_pose_matrix):
    """
    Transform a 2D polyline from world frame to ego-centric frame.

    Args:
        polyline_world: (N, 2) array of (x, y) points in world frame
        ego_pose_matrix: (4, 4) ego-to-world transform

    Returns:
        polyline_ego: (N, 2) array of (x, y) points in ego frame
    """
    # Invert ego pose: world_to_ego
    world_to_ego = np.linalg.inv(ego_pose_matrix)

    # Lift 2D to 3D (z=0 for map elements on ground plane)
    N = polyline_world.shape[0]
    points_3d = np.zeros((N, 4))
    points_3d[:, :2] = polyline_world
    points_3d[:, 2] = 0.0  # ground plane
    points_3d[:, 3] = 1.0  # homogeneous

    # Transform
    points_ego = (world_to_ego @ points_3d.T).T  # (N, 4)

    return points_ego[:, :2]  # Return (x, y) in ego frame
```

---

## 5. Ground Truth Generation Pipeline

### 5.1 Pipeline Overview

For each training sample (one keyframe), the GT generation pipeline produces three
outputs on the BEV grid:

```
Input: sample_token
         |
         v
    [Get Ego Pose] --> ego_pose (4x4)
         |
         v
    [Query Map Elements within perception range]
         |
         v
    [Transform polylines: world -> ego frame]
         |
         v
    [Filter: keep only elements within BEV grid bounds]
         |
         v
    +----+----+----+
    |         |         |
    v         v         v
[Semantic] [Instance] [Direction]
  masks       map        map
```

### 5.2 BEV Grid Configuration

```python
# Standard HDMapNet BEV configuration
bev_config = {
    'xbound': [-30.0, 30.0, 0.15],  # [min, max, resolution] in meters
    'ybound': [-15.0, 15.0, 0.15],  # lateral range
    # OR for 200x200 grid:
    'xbound': [-30.0, 30.0, 0.3],
    'ybound': [-15.0, 15.0, 0.15],
}

# Grid dimensions
nx = int((xbound[1] - xbound[0]) / xbound[2])  # 200
ny = int((ybound[1] - ybound[0]) / ybound[2])  # 200
```

Note: The exact bounds and resolution vary by implementation. Common configurations:
- 60m x 30m at 0.3m/pixel -> 200 x 100 grid
- 60m x 30m at 0.15m/pixel -> 400 x 200 grid
- Symmetric 60m x 60m at 0.3m/pixel -> 200 x 200 grid

### 5.3 Generating Semantic Masks

For each map class, rasterize all polylines of that class onto a binary BEV grid:

```python
import cv2
import numpy as np

def generate_semantic_mask(polylines_ego, bev_config, line_thickness=2):
    """
    Rasterize polylines onto a BEV binary mask.

    Args:
        polylines_ego: list of (N_i, 2) arrays, each a polyline in ego frame
        bev_config: dict with xbound, ybound
        line_thickness: pixel thickness for drawing polylines

    Returns:
        mask: (H, W) binary uint8 array
    """
    xmin, xmax, xres = bev_config['xbound']
    ymin, ymax, yres = bev_config['ybound']

    W = int((xmax - xmin) / xres)
    H = int((ymax - ymin) / yres)
    mask = np.zeros((H, W), dtype=np.uint8)

    for polyline in polylines_ego:
        # Convert metric coordinates to pixel indices
        px = ((polyline[:, 0] - xmin) / xres).astype(np.int32)
        py = ((polyline[:, 1] - ymin) / yres).astype(np.int32)
        pts = np.stack([px, py], axis=-1)

        # Draw polyline on mask
        cv2.polylines(mask, [pts], isClosed=False,
                      color=1, thickness=line_thickness)

    return mask
```

Final semantic GT shape: `(3, H_bev, W_bev)` -- one binary channel per map class.

### 5.4 Generating Instance Map

Each polyline instance receives a unique integer ID:

```python
def generate_instance_map(polylines_ego, bev_config, line_thickness=2):
    """
    Rasterize polylines with unique instance IDs.

    Returns:
        instance_map: (H, W) int32 array where 0 = background,
                      1..N = instance IDs
    """
    xmin, xmax, xres = bev_config['xbound']
    ymin, ymax, yres = bev_config['ybound']

    W = int((xmax - xmin) / xres)
    H = int((ymax - ymin) / yres)
    instance_map = np.zeros((H, W), dtype=np.int32)

    for idx, polyline in enumerate(polylines_ego, start=1):
        px = ((polyline[:, 0] - xmin) / xres).astype(np.int32)
        py = ((polyline[:, 1] - ymin) / yres).astype(np.int32)
        pts = np.stack([px, py], axis=-1)

        # Draw with instance ID as color value
        cv2.polylines(instance_map, [pts], isClosed=False,
                      color=int(idx), thickness=line_thickness)

    return instance_map
```

### 5.5 Generating Direction Map

At each positive pixel, compute the tangent direction along the polyline:

```python
def generate_direction_map(polylines_ego, bev_config, line_thickness=2):
    """
    Generate per-pixel direction vectors tangent to polylines.

    Returns:
        direction_map: (2, H, W) float32 array with (dx, dy) unit vectors
    """
    xmin, xmax, xres = bev_config['xbound']
    ymin, ymax, yres = bev_config['ybound']

    W = int((xmax - xmin) / xres)
    H = int((ymax - ymin) / yres)
    direction_map = np.zeros((2, H, W), dtype=np.float32)

    for polyline in polylines_ego:
        # Compute tangent at each segment midpoint
        for i in range(len(polyline) - 1):
            p0 = polyline[i]
            p1 = polyline[i + 1]

            # Direction vector (normalized)
            d = p1 - p0
            length = np.linalg.norm(d)
            if length < 1e-6:
                continue
            d_norm = d / length

            # Rasterize this segment
            px0 = int((p0[0] - xmin) / xres)
            py0 = int((p0[1] - ymin) / yres)
            px1 = int((p1[0] - xmin) / xres)
            py1 = int((p1[1] - ymin) / yres)

            # Use Bresenham or cv2 to get pixels along segment
            temp = np.zeros((H, W), dtype=np.uint8)
            cv2.line(temp, (px0, py0), (px1, py1), 1, line_thickness)

            # Assign direction to those pixels
            segment_mask = temp > 0
            direction_map[0][segment_mask] = d_norm[0]
            direction_map[1][segment_mask] = d_norm[1]

    return direction_map
```

### 5.6 Putting It All Together

```python
def generate_gt_for_sample(nusc, nusc_map, sample_token, bev_config):
    """
    Complete GT generation for one sample.
    """
    sample = nusc.get('sample', sample_token)

    # 1. Get ego pose
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_pose = make_4x4(ego_pose_record['translation'],
                        ego_pose_record['rotation'])

    # 2. Define perception patch in world frame
    ego_x, ego_y = ego_pose_record['translation'][:2]
    patch_size = 60.0  # meters
    patch_box = (ego_x, ego_y, patch_size, patch_size)
    patch_angle = quaternion_to_yaw(ego_pose_record['rotation'])

    # 3. Query map elements
    map_classes = {
        0: 'lane_divider',
        1: 'ped_crossing',
        2: 'road_boundary',
    }

    semantic_masks = []
    all_polylines = []

    for class_idx, class_name in map_classes.items():
        records = nusc_map.get_records_in_patch(patch_box, [class_name], 'intersect')
        polylines_world = extract_polylines(nusc_map, records[class_name])

        # 4. Transform to ego frame
        polylines_ego = [transform_polyline_to_ego(p, ego_pose)
                         for p in polylines_world]

        # 5. Generate masks
        semantic_masks.append(
            generate_semantic_mask(polylines_ego, bev_config))
        all_polylines.extend(polylines_ego)

    semantic_gt = np.stack(semantic_masks, axis=0)  # (3, H, W)
    instance_gt = generate_instance_map(all_polylines, bev_config)
    direction_gt = generate_direction_map(all_polylines, bev_config)

    return semantic_gt, instance_gt, direction_gt
```

---

## 6. Data Augmentation Strategies

### 6.1 Image-Level Augmentations

These apply independently to each camera image and do not affect geometry:

```python
import torchvision.transforms as T

image_augmentation = T.Compose([
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
    # Note: no random crop or resize here -- geometry must stay consistent
])
```

### 6.2 Horizontal Flip (Requires Coupled Transform)

Horizontal flip is the most impactful geometric augmentation but requires careful
coordination across images, extrinsics, and BEV ground truth:

```python
def apply_horizontal_flip(images, extrinsics, intrinsics, bev_gt):
    """
    Apply synchronized horizontal flip to all data.

    When flipping:
    1. Flip each image left-right
    2. Swap left/right camera pairs
    3. Adjust extrinsic matrices (negate Y translation, adjust rotation)
    4. Adjust intrinsic cx (principal point)
    5. Flip BEV GT along lateral axis
    """
    # 1. Flip images
    flipped_images = [img.flip(-1) for img in images]  # flip W dimension

    # 2. Swap camera pairs
    # FRONT_LEFT <-> FRONT_RIGHT, BACK_LEFT <-> BACK_RIGHT
    # FRONT and BACK stay (but are still horizontally flipped)
    cam_order = [0, 2, 1, 3, 5, 4]  # Original: FL, FR, BL, BR -> swap pairs
    flipped_images = [flipped_images[i] for i in cam_order]

    # 3. Adjust intrinsics: flip principal point
    flipped_intrinsics = intrinsics.clone()
    img_w = images[0].shape[-1]
    flipped_intrinsics[:, 0, 2] = img_w - intrinsics[:, 0, 2]  # cx' = W - cx

    # 4. Adjust extrinsics: negate lateral component
    flipped_extrinsics = extrinsics.clone()
    flipped_extrinsics[:, 1, 3] *= -1  # negate Y translation
    # Also negate appropriate rotation components
    flipped_extrinsics[:, 0, 1] *= -1
    flipped_extrinsics[:, 1, 0] *= -1
    flipped_extrinsics[:, 2, 1] *= -1
    flipped_extrinsics[:, 1, 2] *= -1

    flipped_extrinsics = flipped_extrinsics[cam_order]
    flipped_intrinsics = flipped_intrinsics[cam_order]

    # 5. Flip BEV GT along lateral (Y) axis
    flipped_bev = bev_gt.flip(-1)  # flip along width (lateral)

    # Also flip direction Y component
    # direction_gt[1] *= -1 if present

    return flipped_images, flipped_extrinsics, flipped_intrinsics, flipped_bev
```

### 6.3 BEV Rotation Augmentation

Rotate both the prediction space and GT together by a random angle:

```python
import torch.nn.functional as F

def rotate_bev(bev_gt, angle_deg):
    """
    Rotate BEV ground truth by angle_deg around center.
    Must apply same rotation to model output or adjust ego pose accordingly.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Rotation matrix for affine_grid
    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0]
    ], dtype=torch.float32).unsqueeze(0)

    grid = F.affine_grid(theta, bev_gt.unsqueeze(0).shape, align_corners=False)
    rotated = F.grid_sample(bev_gt.unsqueeze(0).float(), grid,
                            mode='nearest', align_corners=False)

    return rotated.squeeze(0)
```

### 6.4 Augmentations to Avoid

| Augmentation | Reason to Skip |
|-------------|---------------|
| Vertical flip | Breaks gravity assumption; sky/ground swap is unrealistic |
| Random crop (image) | Invalidates intrinsic matrix unless K is updated |
| Independent per-camera geometric transforms | Breaks multi-view consistency |
| BEV scale augmentation | Changes effective resolution, complicates loss |

---

## 7. Custom Dataset Collection (Non-nuScenes)

If you are collecting data from your own vehicle platform, the following requirements
must be met.

### 7.1 Camera Calibration

**Intrinsic Calibration (per camera, one-time):**

1. Print a checkerboard pattern (e.g., 9x6 inner corners, 30mm squares)
2. Capture 20-50 images of the checkerboard at various angles and distances
3. Use OpenCV calibration:

```python
import cv2

# Find checkerboard corners in each image
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

# After collecting all corner sets:
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_size, None, None
)
# K is your 3x3 intrinsic matrix
# dist_coeffs: lens distortion (undistort images before training)
```

**Extrinsic Calibration (camera-to-vehicle, after mounting):**

1. Use a known target (e.g., large checkerboard) at a measured position relative to vehicle
2. Or use LiDAR-camera cross-calibration if LiDAR is available
3. Result: 6-DOF transform from vehicle body frame to each camera optical center

### 7.2 Synchronized Multi-Camera Capture

**Critical requirement:** All cameras must be hardware-synchronized or software-synchronized
to within 5ms. Unsynchronized images create inconsistent multi-view geometry.

Hardware sync options:
- External trigger signal (GPIO pulse to all cameras simultaneously)
- IEEE 1588 PTP (Precision Time Protocol) over Ethernet
- Camera-specific sync cables (e.g., FLIR/LUCID hardware trigger)

Minimum specs per camera:
- Resolution: 1280x720 or higher
- Frame rate: 10 Hz minimum (2 Hz keyframes selected for training)
- Global shutter preferred (rolling shutter causes motion artifacts)
- HDR capability recommended for varying lighting

### 7.3 Ego Pose Requirements

The ego pose must be accurate to within 10cm for map alignment to work:

| Method | Typical Accuracy | Suitable? |
|--------|-----------------|-----------|
| Consumer GPS | 2-5 meters | No |
| RTK GPS | 1-2 cm | Yes |
| RTK GPS + IMU (fused) | 1-5 cm | Yes (recommended) |
| SLAM (visual/LiDAR) | 5-30 cm | Marginal |
| Post-processed PPK | 1-3 cm | Yes |

Recommended setup: **RTK-corrected GNSS receiver + tactical-grade IMU** with
Kalman filter fusion (e.g., NovAtel SPAN, Applanix POS LV).

### 7.4 Map Annotation

Options for obtaining HD map ground truth:

1. **Commercial HD maps** (HERE, TomTom, Mobileye RoadBook)
   - Pro: Professional quality, wide coverage
   - Con: Expensive licensing, format conversion needed

2. **OpenStreetMap + manual refinement**
   - Pro: Free, global coverage
   - Con: Inaccurate lane-level detail, significant manual work

3. **Manual annotation from LiDAR/aerial imagery**
   - Tools: QGIS, LabelMe3D, custom annotation tools
   - Create polylines by clicking points along lane boundaries
   - Export as ordered (x, y) point sequences

4. **Automated extraction from LiDAR point clouds**
   - Detect road surfaces, then extract boundaries algorithmically
   - Requires manual verification and correction

### 7.5 Minimum Data Requirements

| Aspect | Minimum | Recommended |
|--------|---------|-------------|
| Number of scenes | 100 | 500+ |
| Scene duration | 10 seconds | 20 seconds |
| Keyframes per scene | 20 | 40 |
| Total keyframes | 2,000 | 20,000+ |
| Weather conditions | 2 (dry, rain) | 4+ (add night, fog) |
| Road types | Urban | Urban + highway + suburban |
| Geographic diversity | 1 area | 3+ areas |

---

## 8. Data Loading and Preprocessing

### 8.1 PyTorch Dataset Class Structure

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class HDMapNetDataset(Dataset):
    """
    PyTorch Dataset for HDMapNet training.
    """

    # ImageNet normalization stats
    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMG_STD = torch.tensor([0.229, 0.224, 0.225])

    CAMERA_NAMES = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
    ]

    def __init__(self, nusc, nusc_map, split='train',
                 img_size=(128, 352),  # (H, W) after resize
                 bev_config=None,
                 augment=True):
        """
        Args:
            nusc: NuScenes instance
            nusc_map: dict of NuScenesMap instances keyed by map name
            split: 'train' or 'val'
            img_size: target image size (H, W) after resize
            bev_config: dict with xbound, ybound
            augment: whether to apply data augmentation
        """
        self.nusc = nusc
        self.nusc_map = nusc_map
        self.img_size = img_size
        self.bev_config = bev_config or {
            'xbound': [-30.0, 30.0, 0.3],
            'ybound': [-15.0, 15.0, 0.15],
        }
        self.augment = augment

        # Collect sample tokens for this split
        self.sample_tokens = self._get_split_tokens(split)

        # Pre-compute and cache calibration data
        self._cache_calibration()

    def _cache_calibration(self):
        """
        Cache intrinsic/extrinsic matrices to avoid repeated lookups.
        Calibration is constant per sensor, so only store unique values.
        """
        self.calibration_cache = {}
        for sample_token in self.sample_tokens:
            sample = self.nusc.get('sample', sample_token)
            calib = {}
            for cam_name in self.CAMERA_NAMES:
                sd_token = sample['data'][cam_name]
                sd_record = self.nusc.get('sample_data', sd_token)
                cs_record = self.nusc.get('calibrated_sensor',
                                          sd_record['calibrated_sensor_token'])

                intrinsic = np.array(cs_record['camera_intrinsic'])  # (3, 3)
                sensor_to_ego = make_transform_matrix(
                    cs_record['translation'], cs_record['rotation'])

                calib[cam_name] = {
                    'intrinsic': intrinsic,
                    'sensor_to_ego': sensor_to_ego,
                }
            self.calibration_cache[sample_token] = calib

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)

        # --- Load images ---
        images = []
        intrinsics = []
        extrinsics = []

        for cam_name in self.CAMERA_NAMES:
            sd_token = sample['data'][cam_name]
            sd_record = self.nusc.get('sample_data', sd_token)

            # Load image
            img_path = self.nusc.get_sample_data_path(sd_token)
            img = Image.open(img_path).convert('RGB')

            # Resize
            orig_size = img.size  # (W, H)
            img = img.resize((self.img_size[1], self.img_size[0]),
                             Image.BILINEAR)

            # To tensor and normalize
            img_tensor = torch.from_numpy(
                np.array(img)).permute(2, 0, 1).float() / 255.0
            img_tensor = (img_tensor - self.IMG_MEAN[:, None, None]) / \
                         self.IMG_STD[:, None, None]

            images.append(img_tensor)

            # Calibration (adjust intrinsic for resize)
            calib = self.calibration_cache[sample_token][cam_name]
            K = calib['intrinsic'].copy()

            # Scale intrinsic matrix for resized image
            scale_x = self.img_size[1] / orig_size[0]
            scale_y = self.img_size[0] / orig_size[1]
            K[0, :] *= scale_x  # fx, cx
            K[1, :] *= scale_y  # fy, cy

            intrinsics.append(K)

            # Extrinsic: ego_to_camera
            ego_to_cam = np.linalg.inv(calib['sensor_to_ego'])
            extrinsics.append(ego_to_cam)

        images = torch.stack(images, dim=0)          # (6, 3, H, W)
        intrinsics = torch.tensor(np.stack(intrinsics), dtype=torch.float32)  # (6, 3, 3)
        extrinsics = torch.tensor(np.stack(extrinsics), dtype=torch.float32)  # (6, 4, 4)

        # --- Generate BEV ground truth ---
        semantic_gt, instance_gt, direction_gt = generate_gt_for_sample(
            self.nusc, self.nusc_map, sample_token, self.bev_config)

        semantic_gt = torch.tensor(semantic_gt, dtype=torch.float32)
        instance_gt = torch.tensor(instance_gt, dtype=torch.long)
        direction_gt = torch.tensor(direction_gt, dtype=torch.float32)

        # --- Augmentation ---
        if self.augment:
            images, intrinsics, extrinsics, semantic_gt = self._apply_augmentation(
                images, intrinsics, extrinsics, semantic_gt)

        return {
            'images': images,              # (6, 3, H, W)
            'intrinsics': intrinsics,      # (6, 3, 3)
            'extrinsics': extrinsics,      # (6, 4, 4)
            'semantic_gt': semantic_gt,    # (3, H_bev, W_bev)
            'instance_gt': instance_gt,    # (H_bev, W_bev)
            'direction_gt': direction_gt,  # (2, H_bev, W_bev)
        }
```

### 8.2 Image Normalization

Always normalize with ImageNet statistics (the backbone is typically pretrained on ImageNet):

```python
# Standard ImageNet normalization
mean = [0.485, 0.456, 0.406]  # RGB channel means
std = [0.229, 0.224, 0.225]   # RGB channel stds

# Applied as: normalized = (pixel_value / 255.0 - mean) / std
```

### 8.3 Resize and Crop Strategy

The original nuScenes images (1600x900) are too large for efficient training.
Common resize strategies:

| Strategy | Target Size | Notes |
|----------|-------------|-------|
| Direct resize | 128x352 or 256x704 | Simple, slight aspect ratio change |
| Resize + center crop | Various | Maintains aspect ratio, loses edges |
| Resize preserving ratio | 224x400 | Pads to target size |

**Important:** When resizing, you must update the intrinsic matrix K accordingly:

```python
def adjust_intrinsic_for_resize(K, orig_h, orig_w, new_h, new_w):
    """Scale intrinsic matrix when image is resized."""
    K_new = K.copy()
    K_new[0, 0] *= new_w / orig_w  # fx
    K_new[0, 2] *= new_w / orig_w  # cx
    K_new[1, 1] *= new_h / orig_h  # fy
    K_new[1, 2] *= new_h / orig_h  # cy
    return K_new
```

If you crop the image, also shift the principal point:

```python
def adjust_intrinsic_for_crop(K, crop_x, crop_y):
    """Shift intrinsic matrix when image is cropped."""
    K_new = K.copy()
    K_new[0, 2] -= crop_x  # cx
    K_new[1, 2] -= crop_y  # cy
    return K_new
```

### 8.4 Caching Strategies

GT generation can be expensive (map queries, rasterization). Common caching approaches:

```python
# Option 1: Pre-compute GT and save to disk
import pickle

def precompute_all_gt(nusc, nusc_map, sample_tokens, bev_config, cache_dir):
    """Run once before training to cache all GT maps."""
    os.makedirs(cache_dir, exist_ok=True)
    for token in tqdm(sample_tokens):
        cache_path = os.path.join(cache_dir, f'{token}.pkl')
        if os.path.exists(cache_path):
            continue
        sem, inst, dirn = generate_gt_for_sample(nusc, nusc_map, token, bev_config)
        with open(cache_path, 'wb') as f:
            pickle.dump({'semantic': sem, 'instance': inst, 'direction': dirn}, f)

# Option 2: Use lmdb for fast random access
import lmdb

def create_lmdb_cache(sample_tokens, gt_data, lmdb_path):
    """Store GT in LMDB for fast memory-mapped access."""
    env = lmdb.open(lmdb_path, map_size=50 * 1024**3)  # 50GB
    with env.begin(write=True) as txn:
        for token, data in zip(sample_tokens, gt_data):
            txn.put(token.encode(), pickle.dumps(data))
```

### 8.5 DataLoader Configuration

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=4,           # 6 cameras x 4 batches = 24 images per step
    shuffle=True,
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    drop_last=True,         # Avoid variable batch size
    persistent_workers=True # Keep worker processes alive between epochs
)
```

**Memory considerations:**
- 6 images at 256x704x3 float32 = ~13 MB per sample
- Batch of 4 = ~52 MB just for images
- Add extrinsics, intrinsics, GT maps: ~60 MB total per batch
- With num_workers=4 prefetching: budget ~240 MB for data loading

---

## Summary Checklist

Before starting training, verify:

- [ ] nuScenes or custom dataset downloaded and extracted
- [ ] Camera intrinsics calibrated and stored per camera
- [ ] Camera extrinsics (sensor-to-ego transforms) available
- [ ] Ego pose available for every keyframe (accurate to <10cm)
- [ ] Map data available in vector polyline format
- [ ] BEV grid configuration chosen (bounds, resolution)
- [ ] GT generation pipeline tested on a few samples
- [ ] GT cached to disk (pickle or LMDB) for training speed
- [ ] Image normalization uses ImageNet stats
- [ ] Intrinsic matrix adjusted for any image resize/crop
- [ ] Horizontal flip augmentation correctly swaps cameras and flips GT
- [ ] DataLoader configured with appropriate batch size and workers
