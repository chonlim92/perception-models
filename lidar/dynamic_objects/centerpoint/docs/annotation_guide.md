# Annotation Guide: 3D Bounding Boxes and Tracking

## Overview

CenterPoint relies on high-quality 3D bounding box annotations with tracking IDs and velocity ground truth. This guide covers the annotation format, coordinate systems, and quality requirements for both nuScenes and Waymo datasets.

---

## 3D Bounding Box Representation

### Box Parameters (9 DoF + metadata)

Each annotated object is described by a 3D oriented bounding box:

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| Center X | x | meters | Object center in ego/global frame |
| Center Y | y | meters | Object center in ego/global frame |
| Center Z | z | meters | Object center (geometric center of box) |
| Width | w | meters | Extent along the local Y-axis (lateral) |
| Length | l | meters | Extent along the local X-axis (longitudinal) |
| Height | h | meters | Extent along the local Z-axis (vertical) |
| Yaw | theta | radians | Rotation around the Z-axis (heading) |
| Velocity X | vx | m/s | Longitudinal velocity in global frame |
| Velocity Y | vy | m/s | Lateral velocity in global frame |

### Box Convention

```
        l (length)
    ┌─────────────┐
    │             │
w   │      * ──── │ ──→ heading (yaw = 0 points to +x)
    │   (center)  │
    └─────────────┘

* Center is the geometric center of the box
* Yaw angle: counter-clockwise from the positive x-axis
* Height center: z is the center height, not the bottom
```

---

## Track-Level Annotations

### Tracking ID Assignment

Each object instance receives a unique tracking ID that persists across the entire scene (sequence):

```python
# nuScenes: instance_token (UUID string)
# Example: "e1f6b1df98b24511a418503a7c6e8e1c"

# Waymo: object_id (integer, unique within sequence)
# Example: 12345
```

### Tracking Requirements

- **Consistency:** The same physical object must have the same tracking ID across all frames in a scene.
- **Occlusion handling:** Tracking IDs persist through brief occlusions (object not visible for a few frames but reappears).
- **Unique assignment:** Each tracking ID maps to exactly one physical object; no ID reuse within a scene.
- **Lifecycle:** Track starts when object first appears, ends when object permanently leaves the scene.

### Track Metadata

| Field | Description |
|-------|-------------|
| instance_token | Unique identifier for the track (nuScenes) |
| category | Object class (e.g., "car", "pedestrian") |
| first_annotation | Token of first frame where object appears |
| last_annotation | Token of last frame where object appears |
| nbr_annotations | Total number of annotated frames for this track |

---

## Velocity Ground Truth

### Computation from Consecutive Frames

Velocity annotations are derived from the displacement between consecutive annotated frames:

```python
def compute_velocity(current_box, prev_box, dt):
    """
    Compute velocity from consecutive frame annotations.
    
    Args:
        current_box: [x, y, z, w, l, h, yaw] in global frame at time t
        prev_box: [x, y, z, w, l, h, yaw] in global frame at time t-1
        dt: time difference between frames (seconds)
    
    Returns:
        velocity: [vx, vy] in global frame (m/s)
    """
    # Displacement in global frame
    dx = current_box[0] - prev_box[0]
    dy = current_box[1] - prev_box[1]
    
    # Velocity = displacement / time
    vx = dx / dt
    vy = dy / dt
    
    return np.array([vx, vy])
```

### Velocity Annotation Details

- **Frame rate:** nuScenes annotations at 2 Hz (keyframes), Waymo at 10 Hz.
- **Time delta:** nuScenes dt = 0.5s between keyframes, Waymo dt = 0.1s.
- **Smoothing:** Some implementations apply low-pass filtering to reduce noise.
- **Edge cases:**
  - First frame of a track: velocity set to NaN or [0, 0].
  - Stationary objects: velocity close to [0, 0] (within annotation noise).
  - Objects appearing after occlusion: velocity computed only when two consecutive annotations exist.

### nuScenes Velocity Convention

```python
# nuScenes provides velocity in the GLOBAL coordinate frame
# Conversion to ego frame if needed:
velocity_ego = global_to_ego_rotation @ velocity_global
```

---

## nuScenes Annotation Format

### Sample Annotation JSON Structure

```json
{
    "token": "a1b2c3d4e5f6...",
    "sample_token": "f6e5d4c3b2a1...",
    "instance_token": "1a2b3c4d5e6f...",
    "attribute_tokens": ["attr_token_1"],
    "visibility_token": "4",
    "translation": [100.5, 200.3, 1.2],
    "size": [1.8, 4.5, 1.5],
    "rotation": [0.707, 0.0, 0.0, 0.707],
    "velocity": [5.2, -1.3],
    "num_lidar_pts": 245,
    "num_radar_pts": 3,
    "next": "next_annotation_token",
    "prev": "prev_annotation_token"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| token | str | Unique annotation token |
| sample_token | str | Reference to the keyframe |
| instance_token | str | Tracking ID (persistent across frames) |
| attribute_tokens | List[str] | Activity/state attributes |
| visibility_token | str | Visibility level (1-4) |
| translation | [x, y, z] | Center position in global frame (meters) |
| size | [w, l, h] | Box dimensions (width, length, height) |
| rotation | [w, x, y, z] | Orientation as quaternion (global frame) |
| velocity | [vx, vy] | Velocity in global frame (m/s), can be NaN |
| num_lidar_pts | int | Number of LiDAR points inside the box |
| num_radar_pts | int | Number of radar points inside the box |
| next | str | Token of annotation in next keyframe (same instance) |
| prev | str | Token of annotation in previous keyframe (same instance) |

### Attributes

Attributes encode the state or activity of an object:

| Category | Attribute | Description |
|----------|-----------|-------------|
| vehicle | vehicle.moving | Vehicle is in motion |
| vehicle | vehicle.stopped | Vehicle is temporarily stopped (e.g., at light) |
| vehicle | vehicle.parked | Vehicle is parked |
| pedestrian | pedestrian.moving | Pedestrian is walking/running |
| pedestrian | pedestrian.standing | Pedestrian is stationary |
| pedestrian | pedestrian.sitting_lying_down | Pedestrian is seated or lying |
| cycle | cycle.with_rider | Bicycle/motorcycle has a rider |
| cycle | cycle.without_rider | Bicycle/motorcycle is unoccupied |

### Visibility Levels

| Level | Token | Description | LiDAR point percentage visible |
|-------|-------|-------------|-------------------------------|
| 1 | "1" | 0-40% visible | Heavily occluded |
| 2 | "2" | 40-60% visible | Partially occluded |
| 3 | "3" | 60-80% visible | Slightly occluded |
| 4 | "4" | 80-100% visible | Fully visible |

---

## Coordinate Transformations

### Coordinate Frames

CenterPoint operates across multiple coordinate frames:

1. **LiDAR frame:** Origin at the LiDAR sensor, x-forward, y-left, z-up.
2. **Ego-vehicle frame:** Origin at the rear axle center, x-forward, y-left, z-up.
3. **Global frame:** A fixed world reference frame (first frame of the log).

### Transformation Matrices

```python
# LiDAR to Ego transformation (from calibration)
T_ego_lidar = np.eye(4)
T_ego_lidar[:3, :3] = quaternion_to_rotation_matrix(calib['rotation'])
T_ego_lidar[:3, 3] = calib['translation']

# Ego to Global transformation (from ego_pose)
T_global_ego = np.eye(4)
T_global_ego[:3, :3] = quaternion_to_rotation_matrix(ego_pose['rotation'])
T_global_ego[:3, 3] = ego_pose['translation']

# Full chain: LiDAR -> Ego -> Global
T_global_lidar = T_global_ego @ T_ego_lidar
```

### Transforming Annotations to Ego/LiDAR Frame

```python
def transform_box_global_to_ego(box, ego_pose):
    """
    Transform a 3D box from global frame to ego-vehicle frame.
    
    Args:
        box: dict with 'translation' [3], 'rotation' [4] (quaternion), 'size' [3]
        ego_pose: dict with 'translation' [3], 'rotation' [4]
    
    Returns:
        box_ego: transformed box in ego-vehicle frame
    """
    # Inverse ego pose
    R_ego = quaternion_to_rotation_matrix(ego_pose['rotation'])
    t_ego = np.array(ego_pose['translation'])
    
    # Transform center
    center_global = np.array(box['translation'])
    center_ego = R_ego.T @ (center_global - t_ego)
    
    # Transform rotation
    q_global = Quaternion(box['rotation'])
    q_ego_inv = Quaternion(ego_pose['rotation']).inverse
    q_ego = q_ego_inv * q_global
    
    # Velocity transformation (rotate only, no translation)
    if box.get('velocity') is not None:
        vel_global = np.array([box['velocity'][0], box['velocity'][1], 0.0])
        vel_ego = R_ego.T @ vel_global
    
    return {
        'translation': center_ego.tolist(),
        'rotation': q_ego.elements.tolist(),
        'size': box['size'],  # Size is frame-invariant
        'velocity': vel_ego[:2].tolist() if box.get('velocity') else None
    }
```

### Multi-Sweep Point Transformation

```python
def aggregate_sweeps(current_sample, nsweeps=10):
    """
    Aggregate multiple LiDAR sweeps into the current ego frame.
    """
    points_all = []
    
    # Current sweep (time_lag = 0)
    current_pc = load_pointcloud(current_sample['lidar_path'])
    current_pc = np.hstack([current_pc, np.zeros((len(current_pc), 1))])  # time_lag=0
    points_all.append(current_pc)
    
    # Previous sweeps
    for i, sweep in enumerate(current_sample['sweeps'][:nsweeps-1]):
        # Load sweep points
        sweep_pc = load_pointcloud(sweep['data_path'])
        
        # Transform from sweep's ego frame to current ego frame
        # sweep_ego -> global -> current_ego
        T_global_sweep_ego = sweep['ego2global']
        T_current_ego_global = np.linalg.inv(current_sample['ego2global'])
        T_current_sweep = T_current_ego_global @ T_global_sweep_ego
        
        # Also account for LiDAR calibration if different
        T_sweep_lidar = sweep['lidar2ego']
        T_current_lidar = current_sample['lidar2ego']
        
        # Full transform: sweep_lidar -> sweep_ego -> global -> current_ego -> current_lidar
        transform = np.linalg.inv(T_current_lidar) @ T_current_ego_global @ T_global_sweep_ego @ T_sweep_lidar
        
        # Apply transformation
        sweep_pc_hom = np.hstack([sweep_pc[:, :3], np.ones((len(sweep_pc), 1))])
        sweep_pc_transformed = (transform @ sweep_pc_hom.T).T[:, :3]
        
        # Add time lag
        time_lag = current_sample['timestamp'] - sweep['timestamp']
        time_lag_sec = time_lag * 1e-6  # microseconds to seconds
        
        # Combine: [x, y, z, intensity, time_lag]
        sweep_features = np.hstack([
            sweep_pc_transformed,
            sweep_pc[:, 3:4],  # intensity
            np.full((len(sweep_pc), 1), time_lag_sec)
        ])
        points_all.append(sweep_features)
    
    return np.concatenate(points_all, axis=0)
```

---

## Annotation Quality Requirements

### Bounding Box Accuracy

| Criterion | Requirement |
|-----------|-------------|
| Center position error | < 0.1 m (vehicles), < 0.05 m (pedestrians) |
| Size error | < 0.1 m per dimension |
| Heading angle error | < 5 degrees |
| Tracking ID consistency | 100% correct across visible frames |

### Quality Checks

1. **Temporal consistency:** Box sizes should be nearly constant across frames for the same track (allow small variations from annotation noise).
2. **Physical plausibility:** Velocity should be consistent with displacement between frames.
3. **LiDAR point count:** Objects with 0 LiDAR points should be excluded from training (not visible to sensor).
4. **Ground plane alignment:** Bottom of box should roughly align with ground surface.

### Filtering Criteria for Training

```python
# Minimum LiDAR points to include an annotation in training
MIN_POINTS = {
    'car': 5,
    'truck': 5,
    'bus': 5,
    'trailer': 5,
    'construction_vehicle': 5,
    'pedestrian': 2,
    'motorcycle': 2,
    'bicycle': 2,
    'traffic_cone': 2,
    'barrier': 5,
}

# Filter annotations
valid_annotations = [
    ann for ann in annotations
    if ann['num_lidar_pts'] >= MIN_POINTS[ann['category']]
]
```

---

## Ground Truth Generation for CenterPoint

### Heatmap Target Generation

```python
def generate_heatmap_target(boxes, classes, grid_size, voxel_size, pc_range):
    """
    Generate ground truth heatmaps for CenterPoint training.
    
    For each object, render a 2D Gaussian at the object's BEV center.
    """
    num_classes = max(classes) + 1
    heatmap = np.zeros((num_classes, grid_size[1], grid_size[0]))
    
    for box, cls in zip(boxes, classes):
        # Convert box center to BEV grid coordinates
        cx = (box[0] - pc_range[0]) / voxel_size[0] / downsample_factor
        cy = (box[1] - pc_range[1]) / voxel_size[1] / downsample_factor
        
        # Compute Gaussian radius based on box size
        w_pixels = box[3] / voxel_size[0] / downsample_factor
        l_pixels = box[4] / voxel_size[1] / downsample_factor
        radius = gaussian_radius((l_pixels, w_pixels), min_overlap=0.1)
        radius = max(int(radius), 2)
        
        # Draw Gaussian on heatmap
        draw_gaussian(heatmap[cls], center=(int(cx), int(cy)), radius=radius)
    
    return heatmap
```

### Regression Target Assignment

```python
def generate_regression_targets(boxes, grid_size, voxel_size, pc_range, downsample):
    """
    Generate regression targets at each object center location.
    """
    offset_target = np.zeros((2, grid_size[1], grid_size[0]))  # sub-voxel offset
    height_target = np.zeros((1, grid_size[1], grid_size[0]))  # z center
    size_target = np.zeros((3, grid_size[1], grid_size[0]))    # log(w), log(l), log(h)
    rot_target = np.zeros((2, grid_size[1], grid_size[0]))     # sin(yaw), cos(yaw)
    vel_target = np.zeros((2, grid_size[1], grid_size[0]))     # vx, vy
    
    for box in boxes:
        # Grid coordinate (float)
        cx = (box[0] - pc_range[0]) / (voxel_size[0] * downsample)
        cy = (box[1] - pc_range[1]) / (voxel_size[1] * downsample)
        
        # Integer grid position
        cx_int, cy_int = int(cx), int(cy)
        
        # Sub-voxel offset (fractional part)
        offset_target[0, cy_int, cx_int] = cx - cx_int
        offset_target[1, cy_int, cx_int] = cy - cy_int
        
        # Height (absolute)
        height_target[0, cy_int, cx_int] = box[2]
        
        # Size (log-normalized)
        size_target[0, cy_int, cx_int] = np.log(box[3])  # log(w)
        size_target[1, cy_int, cx_int] = np.log(box[4])  # log(l)
        size_target[2, cy_int, cx_int] = np.log(box[5])  # log(h)
        
        # Rotation
        rot_target[0, cy_int, cx_int] = np.sin(box[6])
        rot_target[1, cy_int, cx_int] = np.cos(box[6])
        
        # Velocity
        vel_target[0, cy_int, cx_int] = box[7]  # vx
        vel_target[1, cy_int, cx_int] = box[8]  # vy
    
    return offset_target, height_target, size_target, rot_target, vel_target
```
