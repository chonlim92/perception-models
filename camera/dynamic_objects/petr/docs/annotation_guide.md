# Annotation Guide - 3D Bounding Box Format and Conventions

## Overview

This guide documents the annotation format used for PETR/PETRv2/StreamPETR training on the nuScenes dataset. Understanding these conventions is critical for correct model training, loss computation, and evaluation.

---

## 3D Bounding Box Format

### Box Parameterization

Each 3D bounding box is represented by a 10-dimensional vector (code_size=10):

```
[cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
```

| Index | Parameter | Description | Unit | Range |
|-------|-----------|-------------|------|-------|
| 0 | cx | Center x-coordinate (longitudinal) | meters | [-51.2, 51.2] |
| 1 | cy | Center y-coordinate (lateral) | meters | [-51.2, 51.2] |
| 2 | cz | Center z-coordinate (vertical) | meters | [-5.0, 3.0] |
| 3 | w | Width (lateral extent) | meters | [0, ~10] |
| 4 | l | Length (longitudinal extent) | meters | [0, ~20] |
| 5 | h | Height (vertical extent) | meters | [0, ~5] |
| 6 | sin(yaw) | Sine of heading angle | unitless | [-1, 1] |
| 7 | cos(yaw) | Cosine of heading angle | unitless | [-1, 1] |
| 8 | vx | Velocity x-component | m/s | [-30, 30] |
| 9 | vy | Velocity y-component | m/s | [-30, 30] |

### Why sin/cos Instead of Raw Angle?

The heading angle (yaw) is decomposed into sin(yaw) and cos(yaw) because:
1. **Continuity**: Raw angles have a discontinuity at +/-pi. Regression targets with discontinuities cause training instability.
2. **Smooth Regression**: sin/cos values are bounded [-1, 1] and smooth, making L1 regression well-behaved.
3. **Unique Representation**: atan2(sin(yaw), cos(yaw)) uniquely recovers the angle in [-pi, pi].

### Box Corner Convention

The bounding box is defined with the center at `(cx, cy, cz)` and extends:
- Width (w): +/- w/2 along the lateral axis
- Length (l): +/- l/2 along the longitudinal axis (heading direction)
- Height (h): from cz - h/2 to cz + h/2 along vertical axis

The 8 corners can be computed as:
```python
corners = [
    [+l/2, +w/2, +h/2],  # Front-right-top
    [+l/2, -w/2, +h/2],  # Front-left-top
    [-l/2, -w/2, +h/2],  # Rear-left-top
    [-l/2, +w/2, +h/2],  # Rear-right-top
    [+l/2, +w/2, -h/2],  # Front-right-bottom
    [+l/2, -w/2, -h/2],  # Front-left-bottom
    [-l/2, -w/2, -h/2],  # Rear-left-bottom
    [-l/2, +w/2, -h/2],  # Rear-right-bottom
]
# Then rotate by yaw and translate by (cx, cy, cz)
```

---

## nuScenes Category Mapping

### Detection Classes (10 classes)

| Class Index | Class Name | nuScenes Category | Typical Size (l,w,h) |
|-------------|-----------|-------------------|---------------------|
| 0 | car | vehicle.car | 4.6, 1.9, 1.7 |
| 1 | truck | vehicle.truck | 6.9, 2.5, 2.8 |
| 2 | construction_vehicle | vehicle.construction | 6.4, 2.9, 3.2 |
| 3 | bus | vehicle.bus.bendy, vehicle.bus.rigid | 11.0, 2.9, 3.5 |
| 4 | trailer | vehicle.trailer | 12.3, 2.9, 3.9 |
| 5 | barrier | movable_object.barrier | 0.5, 2.5, 1.0 |
| 6 | motorcycle | vehicle.motorcycle | 2.1, 0.8, 1.5 |
| 7 | bicycle | vehicle.bicycle | 1.7, 0.6, 1.3 |
| 8 | pedestrian | human.pedestrian.* | 0.7, 0.7, 1.8 |
| 9 | traffic_cone | movable_object.trafficcone | 0.4, 0.4, 1.1 |

### Category Mapping Details

nuScenes has a hierarchical category system. Multiple subcategories map to each detection class:

```python
CATEGORY_MAPPING = {
    'car': [
        'vehicle.car',
    ],
    'truck': [
        'vehicle.truck',
    ],
    'construction_vehicle': [
        'vehicle.construction',
    ],
    'bus': [
        'vehicle.bus.bendy',
        'vehicle.bus.rigid',
    ],
    'trailer': [
        'vehicle.trailer',
    ],
    'barrier': [
        'movable_object.barrier',
    ],
    'motorcycle': [
        'vehicle.motorcycle',
    ],
    'bicycle': [
        'vehicle.bicycle',
    ],
    'pedestrian': [
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.police_officer',
    ],
    'traffic_cone': [
        'movable_object.trafficcone',
    ],
}
```

### Ignored Categories

The following nuScenes categories are NOT included in the 10-class detection benchmark:
- `animal` - too rare
- `movable_object.debris` - too rare
- `movable_object.pushable_pullable` - too diverse
- `static_object.bicycle_rack` - static infrastructure
- `vehicle.emergency.ambulance` - too rare
- `vehicle.emergency.police` - too rare

---

## Temporal Annotations

### Instance Tracking

Each object has a unique `instance_token` that persists across all frames in a scene:

```python
# Example: tracking a car across frames
frame_0: {'instance_token': 'abc123', 'class': 'car', 'position': [10.0, 5.0, 0.5]}
frame_1: {'instance_token': 'abc123', 'class': 'car', 'position': [10.5, 5.1, 0.5]}
frame_2: {'instance_token': 'abc123', 'class': 'car', 'position': [11.0, 5.2, 0.5]}
```

### Temporal Annotation Properties

| Property | Description |
|----------|-------------|
| instance_token | Unique ID for object instance (persistent across frames) |
| first_annotation_token | Token of first annotation of this instance in scene |
| last_annotation_token | Token of last annotation of this instance in scene |
| nbr_annotations | Total number of annotations for this instance |
| visibility | Visibility level at each frame (1-4) |

### Visibility Levels

| Level | Description | Meaning |
|-------|-------------|---------|
| 1 | 0-40% visible | Heavily occluded |
| 2 | 40-60% visible | Partially occluded |
| 3 | 60-80% visible | Mostly visible |
| 4 | 80-100% visible | Fully or nearly fully visible |

### Velocity Computation

Object velocities are computed from sequential annotations:
```python
# Velocity is computed in the GLOBAL frame, then transformed to ego frame
v_global = (position_t - position_{t-1}) / (time_t - time_{t-1})

# Transform to ego frame for the current timestamp
v_ego = R_ego2global^{-1} @ v_global
# Only xy components are used: vx, vy (vertical velocity is ignored)
```

---

## Coordinate Systems

### 1. Global Frame (World Coordinates)

- **Origin**: Fixed reference point in the map
- **Axes**: East (x), North (y), Up (z) - right-handed
- **Usage**: Absolute positioning, map alignment, ego-motion computation
- **Note**: nuScenes uses a global frame per log (not per scene)

### 2. Ego Vehicle Frame

- **Origin**: Center of the rear axle, projected to ground plane
- **Axes**: 
  - x: Forward (longitudinal, direction of travel)
  - y: Left (lateral)
  - z: Up (vertical)
- **Usage**: Primary frame for detection outputs, model predictions
- **Transformation**: ego2global (rotation + translation) available per timestamp

### 3. Camera Frame

- **Origin**: Camera optical center
- **Axes**:
  - x: Right (image horizontal)
  - y: Down (image vertical)
  - z: Forward (optical axis, depth)
- **Usage**: Image feature extraction, frustum point generation
- **Transformation**: sensor2ego (rotation + translation) per calibrated sensor

### 4. LiDAR Frame

- **Origin**: LiDAR sensor center
- **Axes**:
  - x: Forward
  - y: Left
  - z: Up
- **Usage**: Ground truth box annotation (boxes are annotated in LiDAR frame)
- **Transformation**: sensor2ego (rotation + translation)

### Coordinate Transformations

```
Camera Frame  --[cam_intrinsic^{-1}]--> Camera 3D
Camera 3D     --[sensor2ego]----------> Ego Frame
Ego Frame     --[ego2global]----------> Global Frame
LiDAR Frame   --[sensor2ego]----------> Ego Frame
```

#### Key Transformation Matrices

```python
# Camera intrinsic matrix (3x3)
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

# Pixel to camera ray (for frustum generation)
# [u, v, 1]^T -> K^{-1} @ [u, v, 1]^T = [x/z, y/z, 1]^T

# Camera-to-ego (4x4 homogeneous)
T_ego_cam = [[R_3x3, t_3x1],
             [0 0 0,     1]]

# Ego-to-global (4x4 homogeneous, changes every timestamp)
T_global_ego = [[R_3x3, t_3x1],
                [0 0 0,     1]]
```

### PETR's Use of Coordinate Systems

1. **Frustum Generation**: Generate points in camera frame using K^{-1} and depth bins
2. **World Coordinates**: Transform frustum points to ego frame using sensor2ego
3. **Normalization**: Normalize 3D coordinates to [-1, 1] using pc_range
4. **MLP Encoding**: Encode normalized coordinates to 256-d position embeddings

For StreamPETR's ego-motion compensation:
```python
# Transform previous frame query position to current frame
# prev_pos is in previous ego frame
# Need: T_curr_ego <- T_global <- T_prev_ego

T_curr_from_prev = T_global_curr_ego^{-1} @ T_global_prev_ego
curr_pos = T_curr_from_prev @ [prev_pos, 1]^T
```

---

## Annotation Quality Notes

### Known Limitations

1. **Velocity noise**: Computed from position differences, can be noisy for slow-moving objects
2. **Far-range accuracy**: Boxes beyond 50m have higher annotation uncertainty
3. **Occlusion handling**: Heavily occluded objects (visibility=1) may have less accurate annotations
4. **Temporal gaps**: Not all objects have annotations at every keyframe (may enter/exit scene)

### Best Practices for Training

1. **Filter by visibility**: Consider filtering visibility=1 objects during training for cleaner supervision
2. **Filter by LiDAR points**: Objects with very few LiDAR points (<5) may have unreliable boxes
3. **Velocity clamping**: Clamp predicted velocities to reasonable ranges during inference
4. **Range filtering**: Focus evaluation on objects within the detection range (typically 50m radius)

---

## Annotation File Examples

### sample_annotation.json Entry

```json
{
    "token": "ef63a697930c4b20a6b9791f423351da",
    "sample_token": "ca9a282c9e77460f8360f564131a8af5",
    "instance_token": "e92a1a9a74034e71a7f7748b10528c28",
    "visibility_token": "4",
    "attribute_tokens": ["cb5118da1ab342aa947717dc53544259"],
    "translation": [373.256, 1130.419, 0.8],
    "size": [0.621, 0.669, 1.642],
    "rotation": [0.9831, 0.0, 0.0, -0.1831],
    "prev": "a28ca11f23c34f30a1a4fda39cfad3e6",
    "next": "1e86b547f14d4e08b3d288e2776f3a08",
    "num_lidar_pts": 5,
    "num_radar_pts": 0
}
```

Field descriptions:
- `translation`: [x, y, z] center position in **global** frame (meters)
- `size`: [w, l, h] width, length, height (meters)
- `rotation`: [w, x, y, z] quaternion representing heading in global frame
- `prev`/`next`: tokens linking to same instance's annotation in adjacent keyframes
- `num_lidar_pts`: number of LiDAR points inside the box (quality indicator)

### Converting to Model Input Format

```python
import numpy as np
from pyquaternion import Quaternion

def annotation_to_model_format(ann, ego2global_rotation, ego2global_translation):
    """Convert nuScenes annotation to model training format."""
    # Position: global -> ego frame
    pos_global = np.array(ann['translation'])
    pos_ego = ego2global_rotation.T @ (pos_global - ego2global_translation)

    # Dimensions: [w, l, h] in nuScenes -> keep same order
    w, l, h = ann['size']

    # Rotation: quaternion -> yaw angle in ego frame
    q_global = Quaternion(ann['rotation'])
    # Remove ego rotation to get heading in ego frame
    q_ego = Quaternion(matrix=ego2global_rotation.T) * q_global
    yaw = q_ego.yaw_pitch_roll[0]  # Extract yaw

    # Velocity (computed from consecutive annotations)
    vx, vy = compute_velocity(ann)  # In ego frame

    # Final format: [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
    return np.array([
        pos_ego[0], pos_ego[1], pos_ego[2],
        w, l, h,
        np.sin(yaw), np.cos(yaw),
        vx, vy
    ])
```
