# BEVFormer: Annotation Guide

## nuScenes 3D Bounding Box Annotation Format

This document describes the annotation format used in the nuScenes dataset for training and evaluating BEVFormer's 3D object detection.

---

## 1. Detection Classes

BEVFormer uses the **nuScenes detection task** which defines 10 object classes:

| Class | Description | Typical Size (W x L x H) | Frequency |
|-------|-------------|--------------------------|-----------|
| `car` | Standard passenger vehicles | 1.9 x 4.6 x 1.7 m | Very common |
| `truck` | Large cargo vehicles, pickups | 2.5 x 6.9 x 2.8 m | Common |
| `bus` | Public transit buses, shuttles | 2.9 x 11.0 x 3.5 m | Moderate |
| `trailer` | Towed cargo containers | 2.5 x 12.3 x 4.0 m | Moderate |
| `construction_vehicle` | Cranes, excavators, bulldozers | 2.8 x 5.7 x 3.2 m | Rare |
| `pedestrian` | People walking, standing, sitting | 0.7 x 0.7 x 1.8 m | Very common |
| `motorcycle` | Motorcycles with/without rider | 0.8 x 2.2 x 1.5 m | Moderate |
| `bicycle` | Bicycles with/without rider | 0.6 x 1.7 x 1.3 m | Moderate |
| `barrier` | Road barriers, guardrails | 0.6 x 1.7 x 1.0 m | Common |
| `traffic_cone` | Orange/yellow traffic cones | 0.4 x 0.4 x 1.0 m | Common |

### Class Hierarchy Mapping

nuScenes has 23 fine-grained categories that are mapped to 10 detection classes:

```
car                    <- vehicle.car
truck                  <- vehicle.truck
bus                    <- vehicle.bus.bendy, vehicle.bus.rigid
trailer                <- vehicle.trailer
construction_vehicle   <- vehicle.construction
pedestrian             <- human.pedestrian.adult, human.pedestrian.child,
                          human.pedestrian.construction_worker,
                          human.pedestrian.police_officer
motorcycle             <- vehicle.motorcycle
bicycle                <- vehicle.bicycle
barrier                <- movable_object.barrier
traffic_cone           <- movable_object.trafficcone
```

**Ignored classes** (not evaluated): `animal`, `debris`, `pushable_pullable`, `personal_mobility`, `stroller`, `wheelchair`, `ambulance`, `police`

---

## 2. 3D Bounding Box Format

### 2.1 Box Parameterization

Each 3D bounding box annotation contains:

| Parameter | Format | Description |
|-----------|--------|-------------|
| `translation` | `[x, y, z]` | Center of the box in **global** coordinates (meters) |
| `size` | `[w, l, h]` | Width, length, height of the box (meters) |
| `rotation` | `[w, x, y, z]` | Orientation as a quaternion (global frame) |
| `velocity` | `[vx, vy]` | Velocity in the global frame (m/s), only x and y |

### 2.2 Detailed Parameter Descriptions

#### Translation (Center Position)
```
translation = [x, y, z]

x: East-West position in global frame (meters)
y: North-South position in global frame (meters)  
z: Height of the box CENTER above ground (meters)
   - For a car: z ≈ half the car height above ground
   - For a pedestrian: z ≈ 0.9m (half of standing height)
```

#### Size (Dimensions)
```
size = [width, length, height]

width (w):  Lateral extent (perpendicular to heading direction)
length (l): Longitudinal extent (along heading direction)
height (h): Vertical extent

Note: Size is always positive and orientation-independent
The box extends ±w/2, ±l/2, ±h/2 from the center
```

#### Rotation (Orientation)
```
rotation = [w, x, y, z]  (quaternion, scalar-first format)

- Represents rotation from a canonical orientation to the object's heading
- Canonical: object faces positive X-axis in global frame
- Yaw angle (heading) is the primary rotation for ground vehicles
- For most vehicles, only the yaw component is significant
  (roll and pitch are nearly zero on flat roads)

Conversion to yaw angle:
  yaw = 2 * atan2(z, w)  (for planar rotation about Z-axis)
```

#### Velocity
```
velocity = [vx, vy]

vx: Velocity in global X direction (m/s)
vy: Velocity in global Y direction (m/s)

- Computed from annotation displacement between consecutive keyframes
- vz is not provided (vertical velocity assumed negligible)
- Stationary objects have velocity = [0.0, 0.0]
- NaN values indicate velocity could not be computed (single-frame annotations)
```

### 2.3 Box Corner Computation

To compute the 8 corners of a 3D bounding box:

```python
import numpy as np
from pyquaternion import Quaternion

def get_box_corners(translation, size, rotation):
    """
    Compute 8 corners of a 3D bounding box.
    
    Args:
        translation: [x, y, z] center position
        size: [w, l, h] box dimensions
        rotation: [w, x, y, z] quaternion
    
    Returns:
        corners: (8, 3) array of corner positions in global frame
    """
    w, l, h = size
    
    # 8 corners in box-local coordinates (centered at origin)
    corners = np.array([
        [-w/2, -l/2, -h/2],
        [-w/2, -l/2,  h/2],
        [-w/2,  l/2, -h/2],
        [-w/2,  l/2,  h/2],
        [ w/2, -l/2, -h/2],
        [ w/2, -l/2,  h/2],
        [ w/2,  l/2, -h/2],
        [ w/2,  l/2,  h/2],
    ])
    
    # Rotate corners to global orientation
    rot_matrix = Quaternion(rotation).rotation_matrix  # 3x3
    corners_global = (rot_matrix @ corners.T).T  # (8, 3)
    
    # Translate to global position
    corners_global += np.array(translation)
    
    return corners_global
```

---

## 3. Attributes

Attributes provide additional state information for each annotation:

### 3.1 Vehicle Attributes

| Attribute | Description | Applies To |
|-----------|-------------|------------|
| `vehicle.moving` | Vehicle is in motion | car, truck, bus, trailer, construction_vehicle |
| `vehicle.parked` | Vehicle is parked (engine off) | car, truck, bus, trailer, construction_vehicle |
| `vehicle.stopped` | Vehicle is stopped but ready to move | car, truck, bus, trailer, construction_vehicle |

### 3.2 Pedestrian Attributes

| Attribute | Description | Applies To |
|-----------|-------------|------------|
| `pedestrian.moving` | Person is walking/running | pedestrian |
| `pedestrian.standing` | Person is standing still | pedestrian |
| `pedestrian.sitting_lying_down` | Person is sitting or lying | pedestrian |

### 3.3 Cycle Attributes

| Attribute | Description | Applies To |
|-----------|-------------|------------|
| `cycle.with_rider` | Has a person on it | motorcycle, bicycle |
| `cycle.without_rider` | No person on it (parked) | motorcycle, bicycle |

### 3.4 Attribute Usage in BEVFormer

- Attributes are **not directly predicted** by the standard BEVFormer detection head
- However, they can be added as an additional classification branch if needed
- During evaluation, attribute accuracy contributes to the AAE (Average Attribute Error) metric

---

## 4. Coordinate Systems

### 4.1 Global Frame

```
Origin: Fixed reference point in the map
X-axis: Points East
Y-axis: Points North  
Z-axis: Points Up
Unit: Meters

All annotations (translation, rotation, velocity) are in this frame.
```

### 4.2 Ego Vehicle Frame

```
Origin: Center of the rear axle, projected to ground
X-axis: Points forward (vehicle heading direction)
Y-axis: Points left
Z-axis: Points up
Unit: Meters

Transformation from global to ego:
  T_ego = inverse(ego_pose)
  point_ego = T_ego @ point_global
```

### 4.3 Sensor (Camera) Frame

```
Origin: Camera optical center
X-axis: Points right (in image)
Y-axis: Points down (in image)
Z-axis: Points forward (out of camera)
Unit: Meters

Transformation from ego to camera:
  T_cam = inverse(calibrated_sensor)
  point_cam = T_cam @ point_ego
```

### 4.4 Image (Pixel) Frame

```
Origin: Top-left corner of image
u-axis: Points right (column index)
v-axis: Points down (row index)
Unit: Pixels

Projection from camera to image:
  [u, v, 1]^T = (1/z) * K @ [x, y, z]^T
  where K is the 3x3 intrinsic matrix
```

### 4.5 LiDAR Frame

```
Origin: LiDAR sensor center
X-axis: Points forward
Y-axis: Points left
Z-axis: Points up
Unit: Meters

Note: In nuScenes, the LiDAR frame is aligned with the ego frame
(same orientation, slightly different origin due to sensor mounting)
```

### 4.6 Coordinate Transform Chain for BEVFormer

BEVFormer operates primarily in the **ego vehicle frame** for its BEV grid:

```
                    Global Frame
                         |
     ego_pose            |            ego_pose (prev frame)
    (current)            |            
         v               |               v
    Ego Frame ------> BEV Grid <------ Ego Frame (prev)
    (current)         (200x200)        (temporal alignment)
         |
    calibrated_sensor (per camera)
         v
    Camera Frame (x6)
         |
    camera_intrinsic
         v
    Image Pixels (x6)
```

### 4.7 Converting Annotations to Ego Frame

```python
from pyquaternion import Quaternion
import numpy as np

def global_to_ego(annotation, ego_pose):
    """Convert annotation from global to ego vehicle frame."""
    
    # Ego pose
    ego_rotation = Quaternion(ego_pose['rotation'])
    ego_translation = np.array(ego_pose['translation'])
    
    # Transform center
    center_global = np.array(annotation['translation'])
    center_ego = ego_rotation.inverse.rotate(center_global - ego_translation)
    
    # Transform rotation
    rot_global = Quaternion(annotation['rotation'])
    rot_ego = ego_rotation.inverse * rot_global
    
    # Transform velocity
    vel_global = np.array([*annotation['velocity'], 0.0])  # Add vz=0
    vel_ego = ego_rotation.inverse.rotate(vel_global)
    
    return {
        'translation': center_ego.tolist(),
        'size': annotation['size'],  # Size is frame-independent
        'rotation': rot_ego.elements.tolist(),
        'velocity': vel_ego[:2].tolist()
    }
```

---

## 5. Visibility Levels

Each annotation has a visibility token indicating what fraction of the object is visible:

| Level | Token | Description | Visibility Range |
|-------|-------|-------------|-----------------|
| 0 | `"1"` | Not visible | 0% |
| 1 | `"2"` | Partially visible | 1-40% |
| 2 | `"3"` | Mostly visible | 40-60% |
| 3 | `"4"` | Fully visible | 60-100% |

### 5.1 Visibility Assessment Criteria

Visibility is determined by:
- Occlusion by other objects
- Truncation at image boundaries
- Self-occlusion (only visible from certain angles)

### 5.2 Impact on Evaluation

- **Training:** All visibility levels are used during training
- **Evaluation:** The nuScenes detection challenge evaluates on objects with **at least 1 LiDAR point** (effectively filtering fully invisible objects)
- Objects with `num_lidar_pts == 0` are ignored during evaluation

### 5.3 Visibility and BEVFormer Performance

BEVFormer's performance varies significantly with visibility:

| Visibility Level | Relative AP (approximate) |
|-----------------|--------------------------|
| Fully visible (60-100%) | 1.0x (baseline) |
| Mostly visible (40-60%) | 0.8x |
| Partially visible (1-40%) | 0.5x |

Temporal fusion helps with partially visible objects by aggregating information across frames where the object may be more visible.

---

## 6. Annotation Statistics

### 6.1 Class Distribution (Training Set)

| Class | Annotations | % of Total | Avg per Frame |
|-------|-------------|-----------|---------------|
| car | 367,187 | 43.2% | 13.2 |
| pedestrian | 182,787 | 21.5% | 6.6 |
| barrier | 107,507 | 12.6% | 3.9 |
| traffic_cone | 74,906 | 8.8% | 2.7 |
| truck | 53,631 | 6.3% | 1.9 |
| trailer | 19,218 | 2.3% | 0.7 |
| bus | 11,375 | 1.3% | 0.4 |
| motorcycle | 8,856 | 1.0% | 0.3 |
| construction_vehicle | 10,843 | 1.3% | 0.4 |
| bicycle | 8,185 | 1.0% | 0.3 |

### 6.2 Class Imbalance Implications

- **Car** and **pedestrian** dominate the dataset, leading to higher AP for these classes
- **Construction_vehicle**, **motorcycle**, and **bicycle** are rare, making them harder to detect
- BEVFormer uses class-balanced sampling and CBGS (Class-Balanced Grouping and Sampling) to mitigate imbalance

### 6.3 Distance Distribution

| Distance Range | % of Annotations | Typical AP Impact |
|----------------|-----------------|-------------------|
| 0-20m | 35% | Highest AP |
| 20-40m | 30% | Moderate AP |
| 40-60m | 20% | Lower AP |
| 60-80m | 10% | Significantly lower AP |
| 80-100m | 5% | Lowest AP |

---

## 7. Annotation Quality and Guidelines

### 7.1 Annotation Protocol

nuScenes annotations were created using the following process:

1. **3D annotation in point cloud:** Annotators place 3D boxes in the LiDAR point cloud view
2. **Multi-view verification:** Boxes are projected to all camera views for visual verification
3. **Temporal tracking:** Objects are tracked across frames to ensure consistent instance IDs
4. **Quality review:** Annotations undergo multi-stage review for accuracy

### 7.2 Annotation Accuracy

| Metric | Typical Accuracy |
|--------|-----------------|
| Center position (x, y) | ±0.1m |
| Center height (z) | ±0.2m |
| Size (w, l, h) | ±0.1m |
| Heading angle | ±5 degrees |
| Velocity | ±0.5 m/s |

### 7.3 Known Annotation Limitations

- **Velocity noise:** Computed from frame-to-frame displacement, may be noisy for slowly moving objects
- **Size consistency:** Some instances have slightly varying size annotations across frames
- **Far-field accuracy:** Objects beyond 50m have less accurate annotations due to sparse LiDAR coverage
- **Attribute ambiguity:** The stopped/parked distinction can be subjective

### 7.4 BEVFormer Training Targets

For BEVFormer, annotations are converted to the following regression targets:

```python
# Detection head targets (per object query)
targets = {
    'labels': int,           # Class index [0, 9]
    'boxes': {
        'cx': float,         # Center x in ego frame (normalized)
        'cy': float,         # Center y in ego frame (normalized)
        'cz': float,         # Center z in ego frame
        'w': float,          # Width (log-space)
        'l': float,          # Length (log-space)
        'h': float,          # Height (log-space)
        'sin_yaw': float,    # sin(heading angle)
        'cos_yaw': float,    # cos(heading angle)
        'vx': float,         # Velocity x in ego frame
        'vy': float,         # Velocity y in ego frame
    }
}
# Total regression parameters: 10
# (cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy)
```

---

## 8. Working with Annotations

### 8.1 Loading Annotations with nuscenes-devkit

```python
from nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='/data/nuscenes')

# Get a sample
sample = nusc.sample[0]

# Get all annotations for this sample
annotations = [nusc.get('sample_annotation', token) 
               for token in sample['anns']]

# Get category name
for ann in annotations:
    instance = nusc.get('instance', ann['instance_token'])
    category = nusc.get('category', instance['category_token'])
    print(f"Class: {category['name']}, Position: {ann['translation']}")
```

### 8.2 Visualizing Annotations

```python
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

# Create a Box object
box = Box(
    center=ann['translation'],
    size=ann['size'],
    orientation=Quaternion(ann['rotation']),
    velocity=(*ann['velocity'], 0)
)

# Project box to camera image
corners_2d = view_points(box.corners(), camera_intrinsic, normalize=True)
```

### 8.3 Filtering Annotations for BEVFormer

```python
# Standard filtering for BEVFormer training
DETECTION_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone'
]

def filter_annotations(annotations, nusc):
    """Filter annotations relevant for BEVFormer detection."""
    filtered = []
    for ann in annotations:
        # Get category
        instance = nusc.get('instance', ann['instance_token'])
        category = nusc.get('category', instance['category_token'])
        
        # Check if in detection classes
        cat_name = category['name']
        det_class = None
        for dc in DETECTION_CLASSES:
            if dc in cat_name:
                det_class = dc
                break
        
        if det_class is None:
            continue
        
        # Filter by number of LiDAR points (must be visible)
        if ann['num_lidar_pts'] == 0:
            continue
        
        filtered.append({
            'class': det_class,
            'translation': ann['translation'],
            'size': ann['size'],
            'rotation': ann['rotation'],
            'velocity': ann['velocity'],
            'num_lidar_pts': ann['num_lidar_pts'],
            'visibility': ann['visibility_token']
        })
    
    return filtered
```
