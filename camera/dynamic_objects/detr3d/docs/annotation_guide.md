# Annotation Guide: nuScenes 3D Bounding Boxes for DETR3D

## Overview

This guide describes the 3D bounding box annotation format, object classes, and coordinate systems used in the nuScenes dataset as consumed by DETR3D. Understanding these conventions is critical for correct model training, evaluation, and deployment.

---

## 3D Bounding Box Format

Each annotated object is represented by a 3D bounding box with the following parameters:

### Box Parameterization

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| Center X | `center_x` | meters | X-coordinate of box center in global frame |
| Center Y | `center_y` | meters | Y-coordinate of box center in global frame |
| Center Z | `center_z` | meters | Z-coordinate of box center in global frame |
| Width | `width` | meters | Extent along the local Y-axis (lateral) |
| Length | `length` | meters | Extent along the local X-axis (longitudinal) |
| Height | `height` | meters | Extent along the local Z-axis (vertical) |
| Yaw | `yaw` | radians | Rotation angle around the Z-axis (up) |

### Box Convention Details
- **Center position:** The center of the box is defined as the geometric centroid of the 3D cuboid, located at half-height above the ground plane.
- **Orientation (yaw):** Measured as counter-clockwise rotation from the positive X-axis when viewed from above. Range: [-pi, pi].
- **No pitch/roll:** Only yaw rotation is annotated; objects are assumed to be on a locally flat ground plane (pitch and roll are zero).
- **Size ordering:** Width < Length for elongated objects (cars, trucks). Width is the shorter lateral dimension.

### nuScenes Storage Format
In the nuScenes database, annotations are stored as:
```json
{
  "translation": [center_x, center_y, center_z],
  "size": [width, length, height],
  "rotation": [w, x, y, z]
}
```
- `translation`: 3D center position in the global coordinate frame (meters)
- `size`: [width, length, height] in meters
- `rotation`: Orientation as a quaternion (w, x, y, z format), representing yaw-only rotation around the up-axis

### DETR3D Internal Representation
For training and inference, DETR3D converts annotations to a 9-DOF vector:
```
[center_x, center_y, center_z, width, length, height, sin(yaw), cos(yaw), velocity_x, velocity_y]
```
Using sin/cos representation for yaw avoids discontinuity at +/-pi boundaries.

---

## Object Classes (10 Detection Categories)

### Class Definitions

| # | Class Name | Description | Typical Dimensions (L x W x H) |
|---|-----------|-------------|----------------------------------|
| 1 | **car** | Personal automobiles, SUVs, vans (< 3.5 tons) | 4.6 x 1.9 x 1.7 m |
| 2 | **truck** | Rigid trucks, pickup trucks (> 3.5 tons) | 6.9 x 2.5 x 2.8 m |
| 3 | **bus** | Public transit buses, school buses, shuttles | 11.0 x 2.9 x 3.5 m |
| 4 | **trailer** | Towed cargo containers, semi-trailers | 12.0 x 2.9 x 3.8 m |
| 5 | **construction_vehicle** | Bulldozers, excavators, cranes, cement mixers | 6.4 x 2.8 x 3.2 m |
| 6 | **pedestrian** | Adults, children, persons in wheelchairs | 0.7 x 0.7 x 1.8 m |
| 7 | **motorcycle** | Motorcycles with or without rider | 2.2 x 0.8 x 1.5 m |
| 8 | **bicycle** | Bicycles with or without rider | 1.8 x 0.6 x 1.5 m |
| 9 | **barrier** | Road barriers, jersey barriers, concrete dividers | 2.5 x 0.6 x 1.0 m |
| 10 | **traffic_cone** | Traffic cones, delineator posts | 0.5 x 0.5 x 1.0 m |

### Class Hierarchy and Grouping
The 10 detection classes are derived from a broader 23-class taxonomy in nuScenes. The grouping for detection evaluation is:

- **Vehicles:** car, truck, bus, trailer, construction_vehicle
- **Vulnerable Road Users (VRU):** pedestrian, motorcycle, bicycle
- **Static Objects:** barrier, traffic_cone

### Annotation Inclusion Criteria
- Object must have at least 1 LiDAR point or be clearly visible in at least 1 camera
- Objects beyond 50 meters may have reduced annotation completeness
- Heavily occluded objects (< 20% visible) are still annotated if identifiable
- Parked vehicles are annotated (velocity = 0)
- Annotation range: typically within 50-60 meters of the ego vehicle

---

## Velocity Annotations

### Format
Each object has an associated velocity vector:
```
velocity: [vx, vy]  (meters per second, in global frame)
```

### Velocity Estimation Method
- Computed from the displacement of the object's center between consecutive annotated keyframes (at 2 Hz)
- `vx = (x_t - x_{t-1}) / delta_t` where `delta_t = 0.5s`
- Only horizontal velocity is provided (no vertical velocity component)
- Stationary objects have velocity [0.0, 0.0]

### Special Cases
- **First/last frame in scene:** Velocity may be NaN or computed from the nearest available pair
- **Newly appeared objects:** Velocity set to NaN for the first frame of track
- **Static classes:** Barriers and traffic cones always have velocity [0.0, 0.0]

### Velocity in DETR3D
- DETR3D predicts velocity as part of its output head: 2 additional regression targets (vx, vy)
- Velocity prediction is evaluated via the AVE (Average Velocity Error) metric
- During training, velocity loss is only applied to objects with valid (non-NaN) velocity annotations

---

## Attribute Annotations

### Visibility Levels
Each annotation includes a visibility rating indicating what fraction of the object is visible (not occluded or truncated):

| Level | Name | Visibility Range | Description |
|-------|------|------------------|-------------|
| 1 | `visibility_0_20` | 0-20% | Mostly occluded, barely visible |
| 2 | `visibility_20_40` | 20-40% | Significantly occluded |
| 3 | `visibility_40_60` | 40-60% | Partially occluded |
| 4 | `visibility_60_80` | 60-80% | Mostly visible |
| 5 | `visibility_80_100` | 80-100% | Fully or nearly fully visible |

### Activity State Attributes
Objects have class-specific activity attributes:

**Vehicles (car, truck, bus, trailer, construction_vehicle):**
| Attribute | Description |
|-----------|-------------|
| `vehicle.moving` | Currently in motion |
| `vehicle.stopped` | Temporarily stopped (e.g., at traffic light) |
| `vehicle.parked` | Parked with no intent to move |

**Pedestrians:**
| Attribute | Description |
|-----------|-------------|
| `pedestrian.moving` | Walking or running |
| `pedestrian.standing` | Standing still |
| `pedestrian.sitting_lying_down` | Seated or lying on ground |

**Bicycles and Motorcycles:**
| Attribute | Description |
|-----------|-------------|
| `cycle.with_rider` | Has a person actively riding |
| `cycle.without_rider` | Parked or without active rider |

### Attribute Usage in DETR3D
- Attributes are evaluated via the AAE (Average Attribute Error) metric
- DETR3D predicts attributes as an auxiliary classification task
- Attribute prediction is class-conditional (different attribute sets per class)

---

## Coordinate Systems

### Global Frame (World Coordinates)
- **Origin:** Fixed reference point defined per log/map
- **X-axis:** East
- **Y-axis:** North
- **Z-axis:** Up (right-handed coordinate system)
- **Usage:** Annotations are stored in this frame; provides absolute positioning
- **Note:** Consistent within a single scene but may differ between scenes from different logs

### Ego-Vehicle Frame
- **Origin:** Center of the rear axle, projected to ground level
- **X-axis:** Forward (vehicle heading direction)
- **Y-axis:** Left
- **Z-axis:** Up
- **Conventions:** Right-handed coordinate system, follows automotive ISO 8855
- **Transformation:** `ego_pose` provides the ego-vehicle-to-global transform at each timestamp
- **Usage:** DETR3D internally operates in ego-vehicle coordinates for detection

### Sensor (Camera) Frame
- **Origin:** Camera optical center
- **X-axis:** Right (image u-direction)
- **Y-axis:** Down (image v-direction)
- **Z-axis:** Forward (optical axis, depth direction)
- **Convention:** Standard computer vision camera coordinate convention
- **Transformation:** `calibrated_sensor` provides sensor-to-ego-vehicle transform (fixed per sensor)

### Image Frame (Pixel Coordinates)
- **Origin:** Top-left corner of the image
- **U-axis:** Right (column index)
- **V-axis:** Down (row index)
- **Units:** Pixels
- **Transformation:** Camera intrinsic matrix K maps 3D camera coordinates to 2D pixel coordinates:
  ```
  [u]       [fx  0  cx] [X/Z]
  [v]   =   [0  fy  cy] [Y/Z]
  [1]       [0   0   1] [ 1 ]
  ```

### Coordinate Transformation Chain
To project a 3D annotation to a camera image (as DETR3D does internally):

```
Global Frame
    │
    ├── (ego_pose inverse) ──→ Ego-Vehicle Frame
    │
    ├── (calibrated_sensor inverse) ──→ Sensor (Camera) Frame
    │
    └── (intrinsic matrix K) ──→ Image Frame (pixels)
```

Full projection equation:
```
p_image = K @ T_sensor_from_ego @ T_ego_from_global @ p_global
```

Where:
- `T_ego_from_global` = inverse of ego_pose (4x4 matrix)
- `T_sensor_from_ego` = inverse of calibrated_sensor extrinsic (4x4 matrix)
- `K` = 3x3 camera intrinsic matrix (extended to 3x4 for homogeneous coordinates)

### Visibility Check
After projection, a point is considered visible in a camera if:
1. The depth (Z in camera frame) is positive (point is in front of the camera)
2. The projected pixel coordinates fall within image bounds: `0 <= u < W` and `0 <= v < H`
3. The point is not occluded by closer geometry (not explicitly checked in DETR3D)

---

## Annotation Quality and Guidelines

### Annotator Instructions
- Annotate the full extent of the object even if partially occluded
- Bounding box should be as tight as possible while fully enclosing the object
- For articulated vehicles (truck + trailer), annotate separately
- Pedestrian groups: annotate each individual separately
- Minimum size: objects must subtend at least 2 LiDAR points or 20 pixels in any camera

### Quality Metrics
- **Inter-annotator agreement:** ~0.1m position error, ~5 degrees heading error
- **Annotation pipeline:** Professional annotators using 3D point cloud + multi-camera visualization tools
- **Review process:** Two-stage with automated consistency checks and human review
- **Track consistency:** Annotations maintain consistent object IDs across keyframes within a scene (tracking annotations)

### Known Limitations
- Distant objects (> 50m) may have less precise annotations due to sparse LiDAR points
- Fast-moving objects may have slight motion blur in camera images
- Night scenes have reduced visibility and potentially noisier annotations
- Some edge cases (e.g., vehicle on tow truck) have inconsistent labeling conventions
