# 3D Bounding Box Annotation Guide

## 1. Overview

This guide covers the annotation formats and quality requirements for 3D object detection labels used with PointNet++ models in autonomous driving. It covers the KITTI and nuScenes annotation systems, coordinate conventions, and quality standards.

---

## 2. 3D Bounding Box Representation

### 2.1 General Parameterization

A 3D bounding box is defined by 7 degrees of freedom (7-DoF):

```
(x, y, z, w, h, l, θ)

- (x, y, z): Center position of the box
- w: Width (lateral extent)
- h: Height (vertical extent)  
- l: Length (longitudinal extent)
- θ: Yaw angle (rotation around the vertical/up axis)
```

Note: Roll and pitch are typically assumed to be zero for objects on flat ground.

### 2.2 Orientation Convention

```
Top-down view (Z-up, KITTI LiDAR frame):

         Length (l)
    ┌──────────────────┐
    │                  │  Width (w)
    │        ● center  │
    │                  │
    └──────────────────┘
    
    θ = 0: aligned with X-axis (forward)
    θ = π/2: aligned with Y-axis (left)
    
    Rotation is counter-clockwise when viewed from above (right-hand rule around Z).
```

### 2.3 Box Corner Computation

```python
import numpy as np

def compute_box_corners_3d(center, dimensions, yaw):
    """Compute the 8 corners of a 3D bounding box.
    
    Args:
        center: (3,) array [x, y, z] - box center
        dimensions: (3,) array [w, h, l] - width, height, length
        yaw: float - rotation angle around vertical axis (radians)
    
    Returns:
        corners: (8, 3) array of corner positions
    """
    w, h, l = dimensions
    
    # 8 corners in local frame (centered at origin)
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    z_corners = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners = np.array([x_corners, y_corners, z_corners])  # (3, 8)
    
    # Rotation matrix (around Z-axis for LiDAR frame)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,            1]
    ])
    
    corners = R @ corners  # Rotate
    corners = corners.T + center  # Translate
    
    return corners  # (8, 3)
```

---

## 3. Coordinate Systems

### 3.1 KITTI Coordinate Frames

```
┌─────────────────────────────────────────────────────────────────┐
│ KITTI uses THREE coordinate systems:                             │
│                                                                  │
│ 1. LiDAR (Velodyne):  X=forward, Y=left,  Z=up                │
│ 2. Camera (rect):     X=right,   Y=down,  Z=forward            │
│ 3. Image:             u=right,   v=down   (pixel coordinates)   │
│                                                                  │
│ IMPORTANT: KITTI labels are in the CAMERA coordinate system!    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 KITTI Label-to-LiDAR Conversion

Since KITTI labels are in camera coordinates but PointNet++ operates in LiDAR coordinates, conversion is essential:

```python
def kitti_camera_to_lidar(x, y, z, Tr_velo_to_cam, R0_rect):
    """Convert KITTI camera coordinates to LiDAR coordinates.
    
    In camera frame: X=right, Y=down, Z=forward
    In LiDAR frame:  X=forward, Y=left, Z=up
    """
    # Invert the transformation chain
    # cam = R0_rect @ Tr_velo_to_cam @ lidar
    # lidar = Tr_velo_to_cam^{-1} @ R0_rect^{-1} @ cam
    
    R0_inv = np.linalg.inv(R0_rect[:3, :3])
    Tr_inv = np.linalg.inv(Tr_velo_to_cam)
    
    point_cam = np.array([x, y, z, 1.0])
    point_rect = np.append(R0_inv @ point_cam[:3], 1.0)
    point_lidar = Tr_inv @ point_rect
    
    return point_lidar[:3]

def kitti_dimensions_cam_to_lidar(h, w, l):
    """Convert dimensions from camera to LiDAR frame.
    
    Camera: h=Y extent, w=X extent, l=Z extent
    LiDAR:  h=Z extent, w=Y extent, l=X extent
    """
    # In LiDAR frame:
    # height (Z) = camera h (Y)
    # width (Y) = camera w (X)  
    # length (X) = camera l (Z)
    return h, w, l  # Dimensions stay the same, axes differ

def kitti_rotation_cam_to_lidar(ry):
    """Convert rotation_y (camera frame) to yaw (LiDAR frame).
    
    Camera rotation_y: rotation around camera Y-axis (pointing down)
    LiDAR yaw: rotation around LiDAR Z-axis (pointing up)
    
    The relationship is: yaw_lidar = -(ry + π/2)
    (accounting for 90° rotation between frames)
    """
    return -(ry + np.pi / 2)
```

### 3.3 nuScenes Coordinate Frame

```
nuScenes uses a consistent right-handed coordinate system:
- Ego vehicle frame: X=forward, Y=left, Z=up
- Global frame: X=East, Y=North, Z=up
- Annotations are in the GLOBAL frame

Conversion to ego frame requires ego_pose at the annotation timestamp.
```

---

## 4. KITTI Label Format

### 4.1 File Structure

Each frame has a corresponding `.txt` file in `label_2/` with one object per line:

```
type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z ry
```

### 4.2 Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Object class: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare |
| `truncated` | float | Truncation level [0, 1]: fraction of object outside image boundary |
| `occluded` | int | Occlusion state: 0=fully visible, 1=partly, 2=largely, 3=unknown |
| `alpha` | float | Observation angle [-π, π]: angle of object relative to camera center |
| `bbox` | 4×float | 2D bounding box in image: left, top, right, bottom (pixels) |
| `h` | float | Height in meters (camera Y-axis extent) |
| `w` | float | Width in meters (camera X-axis extent) |
| `l` | float | Length in meters (camera Z-axis extent) |
| `x` | float | 3D center X in camera coordinates (meters, right) |
| `y` | float | 3D center Y in camera coordinates (meters, down) |
| `z` | float | 3D center Z in camera coordinates (meters, forward/depth) |
| `ry` | float | Rotation around camera Y-axis [-π, π] |

### 4.3 Example KITTI Label

```
Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
Pedestrian 0.00 2 0.48 561.87 187.30 575.97 214.68 1.72 0.53 0.41 -1.46 1.78 24.53 0.44
Cyclist 0.12 1 -2.04 369.45 170.21 421.30 232.50 1.75 0.59 1.82 -6.48 1.69 14.22 -2.12
```

### 4.4 Parsing KITTI Labels

```python
class KITTIObject:
    def __init__(self, line):
        parts = line.strip().split()
        self.type = parts[0]
        self.truncated = float(parts[1])
        self.occluded = int(parts[2])
        self.alpha = float(parts[3])
        self.bbox_2d = [float(x) for x in parts[4:8]]  # left, top, right, bottom
        self.h = float(parts[8])   # height (m)
        self.w = float(parts[9])   # width (m)
        self.l = float(parts[10])  # length (m)
        self.x = float(parts[11])  # center x in camera frame
        self.y = float(parts[12])  # center y in camera frame
        self.z = float(parts[13])  # center z in camera frame (depth)
        self.ry = float(parts[14]) # rotation_y
    
    @property
    def dimensions(self):
        """Returns (h, w, l) in meters."""
        return (self.h, self.w, self.l)
    
    @property
    def center_cam(self):
        """3D center in camera coordinates."""
        return np.array([self.x, self.y, self.z])

def load_kitti_labels(label_path):
    """Load all objects from a KITTI label file."""
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('DontCare'):
                objects.append(KITTIObject(line))
    return objects
```

### 4.5 KITTI Difficulty Levels

| Difficulty | Min BBox Height | Max Occlusion | Max Truncation |
|------------|-----------------|---------------|----------------|
| Easy | 40 px | Fully visible (0) | 15% |
| Moderate | 25 px | Partly occluded (1) | 30% |
| Hard | 25 px | Largely occluded (2) | 50% |

---

## 5. nuScenes Annotation Format

### 5.1 Annotation Structure (JSON)

nuScenes uses a relational database structure stored as JSON files:

```json
// sample_annotation.json entry
{
    "token": "abcdef1234567890",
    "sample_token": "sample_001",
    "instance_token": "instance_car_42",
    "attribute_tokens": ["attr_moving"],
    "visibility_token": "4",
    "translation": [373.214, 1130.48, 1.25],
    "size": [1.92, 4.60, 1.68],
    "rotation": [0.9998, 0.0, 0.0, 0.0175],
    "num_lidar_pts": 327,
    "num_radar_pts": 5,
    "category_name": "vehicle.car",
    "next": "next_annotation_token",
    "prev": "prev_annotation_token"
}
```

### 5.2 Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `translation` | [x, y, z] | Center in global frame (meters) |
| `size` | [w, l, h] | Width, length, height (meters) |
| `rotation` | [w, x, y, z] | Orientation quaternion (global frame) |
| `num_lidar_pts` | int | Number of LiDAR points inside the box |
| `category_name` | string | Hierarchical category (e.g., "vehicle.car") |
| `visibility_token` | string | Visibility level: 1(0-40%), 2(40-60%), 3(60-80%), 4(80-100%) |
| `attribute_tokens` | list | Motion/pose attributes |

### 5.3 nuScenes Categories

```
vehicle.car
vehicle.truck
vehicle.bus.bendy
vehicle.bus.rigid
vehicle.construction
vehicle.motorcycle
vehicle.bicycle
vehicle.trailer
vehicle.emergency.ambulance
vehicle.emergency.police
human.pedestrian.adult
human.pedestrian.child
human.pedestrian.construction_worker
human.pedestrian.police_officer
movable_object.barrier
movable_object.trafficcone
movable_object.pushable_pullable
movable_object.debris
static_object.bicycle_rack
```

### 5.4 nuScenes Detection Classes (10 classes for evaluation)

```python
NUSCENES_DETECTION_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]
```

### 5.5 Converting nuScenes Annotations to LiDAR Frame

```python
from pyquaternion import Quaternion
import numpy as np

def nuscenes_annotation_to_lidar(annotation, ego_pose, calibrated_sensor):
    """Convert nuScenes annotation from global to LiDAR sensor frame.
    
    Args:
        annotation: dict with 'translation', 'size', 'rotation'
        ego_pose: dict with 'translation', 'rotation' (ego in global)
        calibrated_sensor: dict with 'translation', 'rotation' (sensor in ego)
    
    Returns:
        center_lidar: (3,) center in LiDAR frame
        size: (3,) [w, l, h]
        yaw_lidar: float rotation around Z-axis in LiDAR frame
    """
    # Annotation in global frame
    center_global = np.array(annotation['translation'])
    quat_global = Quaternion(annotation['rotation'])
    
    # Step 1: Global to ego frame
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    center_ego = ego_rotation.inverse.rotate(center_global - ego_translation)
    quat_ego = ego_rotation.inverse * quat_global
    
    # Step 2: Ego to sensor (LiDAR) frame
    sensor_translation = np.array(calibrated_sensor['translation'])
    sensor_rotation = Quaternion(calibrated_sensor['rotation'])
    
    center_lidar = sensor_rotation.inverse.rotate(center_ego - sensor_translation)
    quat_lidar = sensor_rotation.inverse * quat_ego
    
    # Extract yaw angle (rotation around Z-axis)
    # For a Z-up frame, yaw is the rotation around Z
    yaw_lidar = quat_lidar.yaw_pitch_roll[0]
    
    size = annotation['size']  # [w, l, h]
    
    return center_lidar, size, yaw_lidar
```

---

## 6. Annotation Quality Requirements

### 6.1 Geometric Accuracy

| Metric | Requirement |
|--------|-------------|
| Center position error | < 10 cm (for objects within 30m) |
| Center position error | < 30 cm (for objects 30-70m) |
| Dimension error | < 10% of actual size |
| Rotation error | < 5° for well-defined heading |
| Rotation error | < 15° for ambiguous heading (e.g., square objects) |

### 6.2 Completeness Requirements

- All objects of target classes within the annotated range must be labeled
- KITTI: annotate all objects visible in the front camera image AND within 70m
- nuScenes: annotate all objects within 50m with at least 1 LiDAR point
- Objects with fewer than 5 LiDAR points may be marked as "DontCare"

### 6.3 Consistency Requirements

- **Temporal consistency:** Same object must have the same instance ID across frames
- **Size consistency:** Dimensions of a tracked object should not vary by more than 5% frame-to-frame
- **Heading continuity:** Orientation should change smoothly (no sudden 180° flips)

### 6.4 Special Cases

| Scenario | Handling |
|----------|----------|
| Partially visible (truncated) | Annotate full extent, mark truncated flag |
| Heavily occluded (>80%) | Annotate if identifiable, mark occluded flag |
| Parked vehicles | Include, mark as "stopped" attribute |
| Grouped pedestrians | Annotate each individual separately |
| Articulated vehicles (bus+trailer) | Annotate each rigid body separately |
| Objects on vehicle (bicycle on car rack) | Annotate as part of the carrier vehicle |

### 6.5 Heading Direction Convention

```
KITTI:
- Heading angle (ry) is the angle between object's forward direction and camera Z-axis
- For cars: forward = direction the car is facing
- Range: [-π, π], 0 = facing away from camera

nuScenes:
- Rotation quaternion encodes full 3D orientation
- Heading = direction of the object's front face
- For cars: front = hood direction
- Converted to yaw: angle from global X-axis (East), counter-clockwise positive
```

---

## 7. Annotation Tools

### 7.1 Common 3D Annotation Tools

| Tool | Features | Format |
|------|----------|--------|
| SUSTechPOINTS | Open-source, web-based, LiDAR+camera fusion | Custom JSON |
| CVAT | Open-source, 3D cuboid support | CVAT XML, KITTI |
| Labelbox | Commercial, AI-assisted | Custom JSON |
| Scale AI | Commercial, full-service annotation | Multiple |
| BasicAI | Commercial, sensor fusion | KITTI, nuScenes |

### 7.2 Annotation Workflow

```
1. Load point cloud + synchronized camera images
2. Identify objects (using both LiDAR intensity and camera color)
3. Place initial 3D box (approximate position and size)
4. Refine box dimensions using point distribution
5. Adjust orientation using point pattern + camera appearance
6. Verify from multiple viewpoints (BEV, front view, side view)
7. Assign class label and attributes
8. Link to track ID for temporal consistency
9. QA review (automated checks + human review)
```

### 7.3 Quality Assurance Checks

```python
def validate_annotation(obj, points_in_box):
    """Automated QA checks for a single annotation."""
    errors = []
    
    # Check 1: Reasonable dimensions
    class_dims = {
        'Car': {'h': (1.2, 2.0), 'w': (1.4, 2.2), 'l': (3.0, 5.5)},
        'Pedestrian': {'h': (1.4, 2.0), 'w': (0.3, 0.8), 'l': (0.3, 0.8)},
        'Cyclist': {'h': (1.4, 2.0), 'w': (0.4, 0.8), 'l': (1.4, 2.2)},
    }
    if obj.type in class_dims:
        dims = class_dims[obj.type]
        if not (dims['h'][0] <= obj.h <= dims['h'][1]):
            errors.append(f"Height {obj.h:.2f} outside range for {obj.type}")
        if not (dims['w'][0] <= obj.w <= dims['w'][1]):
            errors.append(f"Width {obj.w:.2f} outside range for {obj.type}")
        if not (dims['l'][0] <= obj.l <= dims['l'][1]):
            errors.append(f"Length {obj.l:.2f} outside range for {obj.type}")
    
    # Check 2: Points actually inside box
    if points_in_box < 5:
        errors.append(f"Only {points_in_box} points in box (min: 5)")
    
    # Check 3: Box not underground
    ground_z = -1.7  # approximate ground level relative to LiDAR
    if (obj.z - obj.h/2) < ground_z - 0.5:
        errors.append(f"Box bottom below ground plane")
    
    # Check 4: Box not floating
    if (obj.z - obj.h/2) > ground_z + 0.5:
        errors.append(f"Box floating above ground")
    
    return errors
```

---

## 8. IoU Computation for 3D Boxes

### 8.1 3D IoU (Intersection over Union)

```python
from shapely.geometry import Polygon
import numpy as np

def iou_3d(box1, box2):
    """Compute 3D IoU between two bounding boxes.
    
    Each box: (x, y, z, w, h, l, yaw) in LiDAR frame
    """
    # Decompose
    c1, d1, y1 = box1[:3], box1[3:6], box1[6]
    c2, d2, y2 = box2[:3], box2[3:6], box2[6]
    
    # BEV (Bird's Eye View) IoU via Shapely
    corners1_bev = get_bev_corners(c1[:2], d1[0], d1[2], y1)  # (4, 2)
    corners2_bev = get_bev_corners(c2[:2], d2[0], d2[2], y2)  # (4, 2)
    
    poly1 = Polygon(corners1_bev)
    poly2 = Polygon(corners2_bev)
    
    inter_area = poly1.intersection(poly2).area
    
    # Height overlap
    z_min1, z_max1 = c1[2] - d1[1]/2, c1[2] + d1[1]/2
    z_min2, z_max2 = c2[2] - d2[1]/2, c2[2] + d2[1]/2
    
    z_overlap = max(0, min(z_max1, z_max2) - max(z_min1, z_min2))
    
    # 3D intersection and union
    inter_3d = inter_area * z_overlap
    vol1 = d1[0] * d1[1] * d1[2]
    vol2 = d2[0] * d2[1] * d2[2]
    union_3d = vol1 + vol2 - inter_3d
    
    return inter_3d / (union_3d + 1e-8)

def get_bev_corners(center_xy, width, length, yaw):
    """Get 4 BEV corners of a rotated box."""
    cos, sin = np.cos(yaw), np.sin(yaw)
    dx = np.array([length/2 * cos, length/2 * sin])
    dy = np.array([-width/2 * sin, width/2 * cos])
    
    corners = np.array([
        center_xy + dx + dy,
        center_xy + dx - dy,
        center_xy - dx - dy,
        center_xy - dx + dy,
    ])
    return corners
```

---

## 9. Label Statistics

### 9.1 KITTI Object Distribution (Training Set)

| Class | Count | Avg Points | Avg Distance |
|-------|-------|------------|--------------|
| Car | 28,742 | 486 | 25.3 m |
| Van | 2,914 | 524 | 24.1 m |
| Truck | 1,094 | 612 | 30.7 m |
| Pedestrian | 4,487 | 98 | 18.6 m |
| Cyclist | 1,627 | 143 | 17.2 m |
| Tram | 511 | 1,240 | 32.5 m |
| Misc | 973 | 215 | 22.8 m |

### 9.2 Typical Object Dimensions (meters)

| Class | Height | Width | Length |
|-------|--------|-------|--------|
| Car | 1.52 ± 0.13 | 1.63 ± 0.10 | 3.88 ± 0.43 |
| Pedestrian | 1.76 ± 0.11 | 0.66 ± 0.10 | 0.84 ± 0.16 |
| Cyclist | 1.73 ± 0.10 | 0.60 ± 0.08 | 1.76 ± 0.16 |
| Van | 2.19 ± 0.24 | 1.93 ± 0.15 | 5.07 ± 0.57 |
| Truck | 3.14 ± 0.60 | 2.51 ± 0.25 | 9.45 ± 3.21 |
