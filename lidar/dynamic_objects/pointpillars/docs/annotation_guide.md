# 3D Object Detection Annotation Guide for PointPillars

This guide documents the annotation formats, anchor definitions, difficulty levels, and coordinate frame conventions used in 3D object detection with PointPillars.

---

## 1. KITTI Annotation Format

The KITTI dataset uses a plain-text label format with one object per line. Each line contains **15 space-separated fields**.

### Field Definitions

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `type` | string | Object class: `Car`, `Van`, `Truck`, `Pedestrian`, `Person_sitting`, `Cyclist`, `Tram`, `Misc`, or `DontCare` |
| 2 | `truncated` | float | Fraction of the object that lies outside the image boundary. Range: 0.0 (fully visible) to 1.0 (fully truncated). |
| 3 | `occluded` | int | Occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown |
| 4 | `alpha` | float | Observation angle of the object in radians, ranging from -pi to pi. This is the angle between the object's forward direction and the ray from the camera center to the object center projected onto the ground plane. |
| 5-8 | `bbox` | 4 floats | 2D bounding box in the image: `x1, y1, x2, y2` (left, top, right, bottom) in pixels. 0-indexed. |
| 9-11 | `dimensions` | 3 floats | 3D object dimensions in meters: `height, width, length` (h, w, l) |
| 12-14 | `location` | 3 floats | 3D object center location `x, y, z` in **camera coordinates** (meters). Note: `y` points downward in KITTI camera frame, and the location refers to the bottom-center of the 3D bounding box. |
| 15 | `rotation_y` | float | Rotation around the Y-axis in camera coordinates, in radians. Range: -pi to pi. 0 means the object faces the positive X-axis of the camera. |

### Example Annotation Line

```
Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
```

Breakdown:

| Field | Value | Meaning |
|-------|-------|---------|
| type | `Car` | Object is a car |
| truncated | `0.00` | Not truncated (fully within image) |
| occluded | `0` | Fully visible |
| alpha | `-1.58` | Observation angle approximately -pi/2 |
| bbox | `587.01 173.33 614.12 200.12` | 2D box: left=587, top=173, right=614, bottom=200 |
| dimensions | `1.65 1.67 3.64` | Height=1.65m, Width=1.67m, Length=3.64m |
| location | `-0.65 1.71 46.70` | x=-0.65m, y=1.71m, z=46.70m in camera frame |
| rotation_y | `-1.59` | Heading approximately -pi/2 (facing left relative to camera) |

### DontCare Regions

```
DontCare -1 -1 -10 503.89 169.71 590.61 190.13 -1 -1 -1 -1000 -1000 -1000 -10
```

`DontCare` annotations mark regions that contain objects which are too ambiguous to label (e.g., distant clusters, heavily occluded objects). During evaluation:

- Predictions that overlap with `DontCare` regions are **neither penalized nor rewarded**.
- This prevents false positives in ambiguous areas from unfairly lowering precision.
- The non-meaningful fields are filled with placeholder values (`-1`, `-1000`, `-10`).

---

## 2. nuScenes Annotation Format

nuScenes uses a structured JSON format with richer metadata than KITTI. Each annotation (called a `sample_annotation`) contains:

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `translation` | `[x, y, z]` | Center position of the 3D bounding box in the **global frame** (meters). `z` represents the geometric center height, not the bottom face. |
| `size` | `[w, l, h]` | Bounding box extent: width, length, height in meters. |
| `rotation` | `[w, x, y, z]` | Orientation as a unit quaternion in the global frame. The quaternion encodes the rotation from the object's canonical orientation to its actual orientation. |
| `velocity` | `[vx, vy]` | Velocity in m/s in the global frame (x and y components only; z velocity is not annotated). |
| `attribute` | string | Behavioral state of the object (see below). |
| `visibility` | int | Visibility level token (see below). |
| `category_name` | string | Hierarchical class name, e.g., `vehicle.car`, `human.pedestrian.adult`. |
| `instance_token` | string | Unique ID linking annotations of the same object across frames (enables tracking). |

### Attributes

Attributes describe the behavioral state of an annotated object:

| Category | Attribute | Description |
|----------|-----------|-------------|
| Vehicle | `vehicle.moving` | Vehicle is in motion |
| Vehicle | `vehicle.stopped` | Vehicle is stopped (temporarily) |
| Vehicle | `vehicle.parked` | Vehicle is parked (engine off) |
| Pedestrian | `pedestrian.moving` | Pedestrian is walking/running |
| Pedestrian | `pedestrian.standing` | Pedestrian is standing still |
| Pedestrian | `pedestrian.sitting_lying_down` | Pedestrian is seated or lying |
| Cyclist | `cycle.with_rider` | Bicycle/motorcycle has a rider |
| Cyclist | `cycle.without_rider` | Bicycle/motorcycle is parked without rider |

### Visibility Levels

| Level | Label | Description |
|-------|-------|-------------|
| 0 | Unknown | Visibility not determined |
| 1 | 0-40% | Most of the object is NOT visible |
| 2 | 40-60% | Object is partially visible |
| 3 | 60-80% | Most of the object is visible |
| 4 | 80-100% | Object is (nearly) fully visible |

### Example (JSON)

```json
{
  "sample_token": "ca9a282c9e77460f8360f564131a8af5",
  "translation": [373.214, 1130.48, 1.25],
  "size": [1.92, 4.01, 1.44],
  "rotation": [0.9999, 0.0, 0.0, 0.0175],
  "velocity": [4.52, 0.31],
  "category_name": "vehicle.car",
  "attribute_tokens": ["vehicle.moving"],
  "visibility_token": "4",
  "instance_token": "a7d0722bfe164cd6b7f4546e2acf3d56"
}
```

---

## 3. Anchor Definitions

PointPillars uses predefined anchors (prior boxes) to regress 3D bounding box parameters. Anchors are critical for matching ground truth during training and for decoding predictions at inference.

### Deriving Anchor Sizes from Training Data

Anchor dimensions are computed from the **mean dimensions of each class** in the training set:

```python
# Example: compute mean anchor sizes from KITTI training labels
import numpy as np

# Collected dimensions [height, width, length] per class
car_dims = np.array([[1.52, 1.63, 3.88], [1.50, 1.62, 3.85], ...])
ped_dims = np.array([[1.73, 0.67, 0.87], [1.70, 0.65, 0.90], ...])
cyc_dims = np.array([[1.73, 0.60, 1.76], [1.75, 0.58, 1.78], ...])

# Mean anchor sizes (h, w, l)
anchor_sizes = {
    "Car":        np.mean(car_dims, axis=0),   # ~ [1.52, 1.64, 3.88]
    "Pedestrian": np.mean(ped_dims, axis=0),   # ~ [1.73, 0.67, 0.87]
    "Cyclist":    np.mean(cyc_dims, axis=0),   # ~ [1.73, 0.60, 1.78]
}
```

Typical anchor sizes used in PointPillars for KITTI:

| Class | Height (m) | Width (m) | Length (m) |
|-------|-----------|-----------|------------|
| Car | 1.52 | 1.64 | 3.88 |
| Pedestrian | 1.73 | 0.67 | 0.87 |
| Cyclist | 1.73 | 0.60 | 1.78 |

### Anchor Rotation Angles

For each class, anchors are placed at **two rotation angles** to cover the dominant orientations:

- **0 radians** (aligned with the x-axis)
- **pi/2 radians** (perpendicular, aligned with the y-axis)

This means at every spatial position on the BEV grid, there are `num_classes x 2` anchors (one per rotation per class).

### Anchor Height Placement

Anchors are placed at a fixed height derived from the mean z-center of each class:

| Class | Anchor z-center (m) | Description |
|-------|---------------------|-------------|
| Car | -1.00 | Center of a typical car at ~0.7m above ground (KITTI LiDAR frame) |
| Pedestrian | -0.60 | Center of a typical pedestrian at ~0.9m above ground |
| Cyclist | -0.60 | Center of a typical cyclist at ~0.9m above ground |

Note: In KITTI LiDAR frame, the z-axis points upward and the LiDAR sensor is mounted at approximately z=0 (about 1.73m above ground). Negative z-center values correspond to objects below the sensor.

### IoU Matching Thresholds

During training, each anchor is assigned as positive, negative, or ignored based on its IoU (Intersection over Union) with ground truth boxes:

| Class | Positive Threshold | Negative Threshold |
|-------|-------------------|-------------------|
| Car | >= 0.60 | < 0.45 |
| Pedestrian | >= 0.50 | < 0.35 |
| Cyclist | >= 0.50 | < 0.35 |

Assignment rules:

1. **Positive anchor**: IoU with any ground truth box >= positive threshold, OR the anchor has the highest IoU with a particular ground truth (even if below threshold).
2. **Negative anchor**: IoU with ALL ground truth boxes < negative threshold.
3. **Ignored anchor**: IoU falls between negative and positive thresholds. These anchors do not contribute to the loss.

---

## 4. KITTI Difficulty Levels

KITTI defines three difficulty levels that determine which objects are evaluated. An object is assigned to the hardest level whose criteria it still satisfies.

### Criteria

| Criterion | Easy | Moderate | Hard |
|-----------|------|----------|------|
| Min. bounding box height | 40 px | 25 px | 25 px |
| Max. occlusion level | 0 (fully visible) | 1 (partly occluded) | 2 (largely occluded) |
| Max. truncation | 15% | 30% | 50% |

### Detailed Definitions

**Easy**
- The 2D bounding box height is at least 40 pixels.
- The object is fully visible (occlusion = 0).
- At most 15% of the object extends outside the image (truncation <= 0.15).

**Moderate**
- The 2D bounding box height is at least 25 pixels.
- The object is at most partly occluded (occlusion <= 1).
- At most 30% of the object extends outside the image (truncation <= 0.30).

**Hard**
- The 2D bounding box height is at least 25 pixels.
- The object is at most largely occluded (occlusion <= 2).
- At most 50% of the object extends outside the image (truncation <= 0.50).

### How Difficulty Affects Evaluation

- Each difficulty level is evaluated **independently**. The "Moderate" evaluation includes all objects satisfying moderate criteria (which is a superset of "Easy").
- The official KITTI benchmark reports **Average Precision (AP)** at each difficulty level.
- Since the KITTI update (2019), evaluation uses **40-point interpolation** of the precision-recall curve (previously 11-point).
- For 3D detection, IoU thresholds used for matching predictions to ground truth:
  - Car: IoU >= 0.70
  - Pedestrian: IoU >= 0.50
  - Cyclist: IoU >= 0.50

---

## 5. Coordinate Frame Conventions

### KITTI Coordinate Frames

KITTI defines several coordinate frames. The two most relevant for 3D detection are:

**Camera Frame (reference frame for labels)**
```
        z (forward/depth)
       /
      /
     /_________ x (right)
     |
     |
     | y (down)
```

- Origin: left camera optical center
- X-axis: points right
- Y-axis: points **down**
- Z-axis: points forward (into the scene)

**LiDAR Frame**
```
     z (up)
     |
     |
     |_________ x (forward)
    /
   /
  / y (left)
```

- Origin: Velodyne LiDAR sensor center
- X-axis: points forward
- Y-axis: points **left**
- Z-axis: points **up**

### nuScenes Coordinate Frames

**Ego Vehicle Frame**
```
     z (up)
     |
     |
     |_________ x (forward)
    /
   /
  / y (left)
```

- Origin: rear axle center projected to ground
- X-axis: forward
- Y-axis: left
- Z-axis: up

**Global Frame**
- A fixed world coordinate frame (East-North-Up or map-relative).
- Annotations are provided in the global frame.
- Transformations from ego to global are given per sample as a 4x4 matrix (rotation + translation).

### Transformation Between KITTI Camera and LiDAR Frames

The KITTI dataset provides calibration matrices to convert between frames:

```
# Camera-to-LiDAR transformation
# Given a point P_cam = [x_cam, y_cam, z_cam] in camera frame,
# convert to LiDAR frame P_lidar = [x_lidar, y_lidar, z_lidar]:

x_lidar =  z_cam          # camera depth    -> LiDAR forward
y_lidar = -x_cam          # camera right    -> LiDAR left (negated)
z_lidar = -y_cam          # camera down     -> LiDAR up (negated)
```

More precisely, the full transformation uses the Velodyne-to-camera calibration matrix `Tr_velo_to_cam` provided in the calibration files:

```python
import numpy as np

# From KITTI calib file
# Tr_velo_to_cam: 3x4 matrix [R | t] (LiDAR -> Camera)
# To go Camera -> LiDAR, invert:
R = Tr_velo_to_cam[:, :3]       # 3x3 rotation
t = Tr_velo_to_cam[:, 3]        # 3x1 translation

# Camera -> LiDAR
R_inv = R.T
t_inv = -R.T @ t

# Transform a point from camera to LiDAR
p_cam = np.array([x_cam, y_cam, z_cam])
p_lidar = R_inv @ p_cam + t_inv
```

### Rotation: `rotation_y` to LiDAR Heading

In KITTI:
- `rotation_y` is the rotation around the **camera Y-axis** (which points down).
- A value of 0 means the object's front faces the camera X-axis (right).
- Positive `rotation_y` rotates clockwise when viewed from above (since Y points down).

To convert `rotation_y` to a heading angle in the LiDAR frame:

```python
# rotation_y (camera frame) -> heading (LiDAR frame)
# In LiDAR frame, heading is rotation around z-axis (up)
# heading = 0 means facing forward (positive x in LiDAR)

heading_lidar = -(rotation_y + np.pi / 2)

# Normalize to [-pi, pi]
heading_lidar = (heading_lidar + np.pi) % (2 * np.pi) - np.pi
```

Explanation:
1. The camera X-axis corresponds to the negative LiDAR Y-axis.
2. Camera Z-axis (forward) corresponds to LiDAR X-axis (forward).
3. The pi/2 offset accounts for the 90-degree rotation between the axes.
4. The negation accounts for the sign convention change (camera Y is down, LiDAR Z is up, resulting in opposite rotation direction).

### nuScenes: Ego to Global

```python
from pyquaternion import Quaternion

# Given from sample data
ego_to_global_rotation = Quaternion(w, x, y, z)  # unit quaternion
ego_to_global_translation = np.array([tx, ty, tz])

# Transform point from ego frame to global frame
p_ego = np.array([x_ego, y_ego, z_ego])
p_global = ego_to_global_rotation.rotate(p_ego) + ego_to_global_translation

# Transform heading from ego to global
heading_ego = some_angle
heading_global = heading_ego + ego_to_global_rotation.yaw_pitch_roll[0]
```

### Summary of Frame Conventions

| Property | KITTI Camera | KITTI LiDAR | nuScenes Ego | nuScenes Global |
|----------|-------------|-------------|--------------|-----------------|
| Forward | +Z | +X | +X | Map-defined |
| Right | +X | -Y | -Y | Map-defined |
| Up | -Y | +Z | +Z | +Z |
| Rotation axis | Y (down) | Z (up) | Z (up) | Z (up) |
| Rotation sign | CW from top | CCW from top | CCW from top | CCW from top |
| Labels provided in | Yes (KITTI) | No (derived) | No (derived) | Yes (nuScenes) |

---

## References

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
- [nuScenes Dataset Schema](https://www.nuscenes.org/nuscenes#data-format)
- Lang, A. H., et al. "PointPillars: Fast Encoders for Object Detection from Point Clouds." CVPR 2019.
- [KITTI Object Development Kit](https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt)
