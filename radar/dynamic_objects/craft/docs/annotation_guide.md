# CRAFT: Annotation Guide

## 3D Bounding Box Annotations with Radar-Camera Cross-Reference

---

## 1. Overview

This guide describes the annotation methodology and format used for training and evaluating the CRAFT model. The annotations follow the nuScenes dataset standard, with additional considerations for radar-camera cross-modal consistency.

### Annotation Scope

| Aspect | Specification |
|--------|--------------|
| Annotation type | 3D bounding boxes with attributes |
| Coordinate frame | Global (map) frame, convertible to ego frame |
| Annotation rate | 2 Hz (at keyframes) |
| Annotated classes | 10 detection classes |
| Additional attributes | Visibility, activity, vehicle state |
| Instance tracking | Unique instance IDs across frames |

---

## 2. 3D Bounding Box Annotations

### 2.1 Bounding Box Representation

Each annotated object is represented as a 3D oriented bounding box with the following parameters:

```python
annotation = {
    "token": "unique_annotation_id",
    "sample_token": "associated_keyframe_id",
    "instance_token": "object_instance_id",  # Consistent across frames
    
    # 3D box parameters (in global frame)
    "translation": [x, y, z],          # Center position (meters)
    "size": [width, length, height],    # Box dimensions (meters) [w, l, h]
    "rotation": [w, x, y, z],          # Orientation quaternion (scalar-first)
    
    # Velocity (for moving objects)
    "velocity": [vx, vy],              # m/s in global frame (z-velocity not annotated)
    
    # Classification
    "category_name": "vehicle.car",     # Hierarchical class label
    
    # Visibility
    "visibility_token": "4",            # 1=0-40%, 2=40-60%, 3=60-80%, 4=80-100%
    
    # Attributes
    "attribute_tokens": ["attr_token"], # Activity/state attributes
    
    # Tracking
    "prev": "previous_annotation_token",
    "next": "next_annotation_token",
    "num_lidar_pts": 15,                # LiDAR points inside box (quality metric)
    "num_radar_pts": 3,                 # Radar points inside box
}
```

### 2.2 nuScenes Annotation Format Details

#### Translation (Center Position)

The translation vector `[x, y, z]` represents the center of the 3D bounding box in the global coordinate frame:

- **x:** East-West position (meters)
- **y:** North-South position (meters)  
- **z:** Height above ground reference (meters, represents box center height)

To convert to ego frame:
```python
from pyquaternion import Quaternion
import numpy as np

def global_to_ego(annotation, ego_pose):
    """Convert annotation from global to ego frame."""
    # Translation
    pos_global = np.array(annotation['translation'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    pos_ego = ego_rotation.inverse.rotate(pos_global - ego_translation)
    
    # Rotation
    box_rotation = Quaternion(annotation['rotation'])
    rot_ego = ego_rotation.inverse * box_rotation
    
    return pos_ego, rot_ego
```

#### Size (Dimensions)

The size vector `[width, length, height]` follows the nuScenes convention:

- **width (w):** Lateral extent (perpendicular to heading direction)
- **length (l):** Longitudinal extent (along heading direction)
- **height (h):** Vertical extent

Typical dimensions by class:

| Class | Width (m) | Length (m) | Height (m) |
|-------|-----------|------------|------------|
| Car | 1.7 - 2.1 | 3.8 - 5.2 | 1.4 - 1.8 |
| Truck | 2.2 - 2.8 | 5.0 - 12.0 | 2.5 - 4.0 |
| Bus | 2.5 - 3.0 | 8.0 - 14.0 | 3.0 - 4.0 |
| Trailer | 2.4 - 2.8 | 6.0 - 16.0 | 2.5 - 4.5 |
| Construction Vehicle | 2.0 - 3.5 | 3.0 - 8.0 | 2.0 - 4.0 |
| Pedestrian | 0.4 - 0.8 | 0.4 - 0.8 | 1.5 - 2.0 |
| Motorcycle | 0.6 - 1.0 | 1.8 - 2.5 | 1.2 - 1.8 |
| Bicycle | 0.5 - 0.8 | 1.5 - 2.0 | 1.2 - 1.8 |
| Traffic Cone | 0.3 - 0.5 | 0.3 - 0.5 | 0.6 - 1.0 |
| Barrier | 0.4 - 0.8 | 0.5 - 2.5 | 0.8 - 1.2 |

#### Rotation (Quaternion)

Orientation is stored as a unit quaternion `[w, x, y, z]` (scalar-first convention):

```python
from pyquaternion import Quaternion

# Example: Car facing 45 degrees to the right
q = Quaternion(axis=[0, 0, 1], angle=-np.pi/4)
# Results in: [w, x, y, z] = [0.924, 0.0, 0.0, -0.383]

# Convert quaternion to yaw angle (rotation around z-axis)
def quaternion_to_yaw(q):
    """Extract yaw angle from quaternion."""
    q = Quaternion(q)
    # Project to z-axis rotation
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    yaw = np.arctan2(v[1], v[0])
    return yaw
```

The rotation represents the heading direction of the object:
- Yaw = 0: Object facing along positive x-axis (East)
- Yaw = π/2: Object facing along positive y-axis (North)
- Only yaw is typically used for ground vehicles (roll/pitch ≈ 0)

### 2.3 Velocity Annotation

Velocity is annotated as a 2D vector `[vx, vy]` in the global frame:

- Computed from consecutive annotations of the same instance
- Smoothed across multiple frames to reduce noise
- Only lateral (vx, vy) components; vertical velocity (vz) is not annotated
- Stationary objects have velocity [0, 0]

```python
# Velocity computation from tracking
def compute_velocity(current_ann, prev_ann, dt):
    """Compute velocity from consecutive annotations."""
    pos_curr = np.array(current_ann['translation'][:2])
    pos_prev = np.array(prev_ann['translation'][:2])
    velocity = (pos_curr - pos_prev) / dt
    return velocity  # [vx, vy] in m/s
```

---

## 3. Radar-Camera Cross-Reference

### 3.1 Projecting Annotations to Radar Space

For CRAFT, annotations must be consistent when viewed from both sensor modalities:

```python
def get_radar_points_in_box(annotation, radar_points, ego_pose, radar_calib):
    """
    Find radar points that fall within an annotated 3D bounding box.
    
    Returns:
        radar_indices: Indices of radar points inside the box
        association_quality: Metric for how well radar supports this annotation
    """
    # Transform annotation to ego frame
    box_center_ego, box_rot_ego = global_to_ego(annotation, ego_pose)
    box_size = annotation['size']  # [w, l, h]
    
    # Transform radar points to ego frame
    R_radar = Quaternion(radar_calib['rotation']).rotation_matrix
    t_radar = np.array(radar_calib['translation'])
    points_ego = (R_radar @ radar_points[:, :3].T).T + t_radar
    
    # Check which points fall inside the oriented bounding box
    # Transform points to box-local frame
    box_rot_matrix = box_rot_ego.rotation_matrix
    points_local = (box_rot_matrix.T @ (points_ego - box_center_ego).T).T
    
    # Check bounds
    half_size = np.array(box_size) / 2.0
    inside = np.all(np.abs(points_local) <= half_size, axis=1)
    
    return np.where(inside)[0]
```

### 3.2 Projecting Annotations to Camera Space

```python
def project_box_to_image(annotation, ego_pose, camera_calib, camera_intrinsic):
    """
    Project 3D bounding box corners to 2D image plane.
    
    Returns:
        corners_2d: 8 corner points projected to image (8, 2)
        bbox_2d: Tight 2D bounding box [x_min, y_min, x_max, y_max]
        visible: Whether box is visible in this camera
    """
    # Get 3D corners of the box
    box_center_ego, box_rot_ego = global_to_ego(annotation, ego_pose)
    w, l, h = annotation['size']
    
    # 8 corners in box-local frame
    corners_local = np.array([
        [-w/2, -l/2, -h/2], [-w/2, -l/2,  h/2],
        [-w/2,  l/2, -h/2], [-w/2,  l/2,  h/2],
        [ w/2, -l/2, -h/2], [ w/2, -l/2,  h/2],
        [ w/2,  l/2, -h/2], [ w/2,  l/2,  h/2],
    ])
    
    # Transform to ego frame
    corners_ego = (box_rot_ego.rotation_matrix @ corners_local.T).T + box_center_ego
    
    # Transform to camera frame
    R_cam = Quaternion(camera_calib['rotation']).rotation_matrix
    t_cam = np.array(camera_calib['translation'])
    corners_cam = (R_cam.T @ (corners_ego - t_cam).T).T
    
    # Check visibility (all corners must be in front of camera)
    if np.any(corners_cam[:, 2] <= 0):
        return None, None, False
    
    # Project to image
    corners_img = (camera_intrinsic @ corners_cam.T).T
    corners_2d = corners_img[:, :2] / corners_img[:, 2:3]
    
    # Compute tight 2D bbox
    bbox_2d = [
        corners_2d[:, 0].min(), corners_2d[:, 1].min(),
        corners_2d[:, 0].max(), corners_2d[:, 1].max()
    ]
    
    # Check if within image bounds (1600 x 900)
    visible = (bbox_2d[2] > 0 and bbox_2d[0] < 1600 and
               bbox_2d[3] > 0 and bbox_2d[1] < 900)
    
    return corners_2d, bbox_2d, visible
```

### 3.3 Cross-Modal Consistency Validation

For each annotation, verify consistency across modalities:

| Check | Criterion | Action if Failed |
|-------|-----------|------------------|
| Radar-image alignment | Radar point projects within 2D bbox in image | Re-check calibration |
| Velocity consistency | Radar Doppler matches annotation velocity (±2 m/s) | Flag for review |
| RCS plausibility | RCS magnitude consistent with object class/size | Note but allow |
| Multi-radar agreement | Same object detected by multiple radars with consistent position | Verify ego-motion |
| Temporal continuity | Box position consistent with velocity over time | Smooth annotation |

---

## 4. Handling Radar Ghost Targets

### 4.1 Types of Radar Artifacts

Automotive radar is susceptible to several types of false detections that must be handled in the annotation process:

#### Multi-Path Reflections

Multi-path occurs when the radar signal bounces off multiple surfaces before returning:

```
                  ┌─── Wall/Guardrail
                  │
    Radar ───────►│──────► Real Object
         ◄────────│◄──────
                  │
         Direct path: Correct detection at true range
         
    Radar ──► Wall ──► Object ──► Wall ──► Radar
         Multi-path: Ghost at incorrect (longer) range
```

**Characteristics of multi-path ghosts:**
- Appear at distances greater than the true target (extra path length)
- Often appear behind walls, guardrails, or other flat surfaces
- May have similar velocity to the real target (or doubled Doppler)
- Typically have lower RCS than the real target

#### Ground Clutter

- Static radar returns from road surface irregularities
- More common at close range and low grazing angles
- Typically have zero or very low velocity
- Can be filtered using `dynProp` flag

#### Interference

- Returns from other vehicles' radar systems
- Random, non-persistent detections
- No consistent velocity or RCS pattern
- Usually filtered at the radar firmware level

#### Mirror Reflections (Specular)

- Strong returns from flat metallic surfaces at perpendicular angles
- Can appear as phantom objects behind barriers or in tunnels
- Often have very high RCS values
- Position may oscillate as viewing angle changes

### 4.2 Annotation Strategy for Ghost Targets

**DO NOT annotate ghost targets as real objects.** The annotation process should:

1. **Identify ghosts by cross-referencing with camera:**
   - If a radar detection has no corresponding visual evidence in any camera view, flag as potential ghost
   - Exception: Objects beyond camera range but within radar range (rare at nuScenes annotation distances)

2. **Use temporal persistence:**
   - Real objects appear consistently across multiple frames with smooth trajectories
   - Ghosts tend to flicker (appear/disappear) or have erratic motion

3. **Apply physical plausibility checks:**
   - Objects cannot be inside solid structures (walls, buildings)
   - Velocity must be physically possible for the detected class
   - Position must be reachable (not through barriers)

4. **Document known ghost-prone scenarios:**
   - Tunnels and underpasses (multiple reflective surfaces)
   - Highway guardrails (metallic, flat surfaces)
   - Parking garages (confined spaces with many reflectors)
   - Construction zones (irregular metallic structures)

### 4.3 Ghost Target Filtering in CRAFT Training

During training, CRAFT handles radar noise through:

```python
# Radar point filtering strategy
def filter_radar_points(radar_points, config):
    """
    Filter radar points to reduce ghost targets for training.
    
    Args:
        radar_points: (N, features) array of radar detections
        config: Filtering configuration
    
    Returns:
        filtered_points: Cleaned radar point cloud
    """
    mask = np.ones(len(radar_points), dtype=bool)
    
    # 1. Remove points with high false alarm probability
    if config.filter_by_pdh0:
        mask &= radar_points[:, 'pdh0_idx'] < config.pdh0_threshold  # e.g., < 0.5
    
    # 2. Remove points with invalid dynamic property
    if config.filter_by_dynprop:
        valid_dynprops = [0, 1, 2, 3, 4, 5, 6]  # Moving and stationary
        mask &= np.isin(radar_points[:, 'dynprop_idx'], valid_dynprops)
    
    # 3. Remove points outside valid range
    ranges = np.sqrt(radar_points[:, 0]**2 + radar_points[:, 1]**2)
    mask &= (ranges > config.min_range) & (ranges < config.max_range)
    
    # 4. Optionally remove stationary clutter
    if config.remove_stationary:
        velocities = np.sqrt(radar_points[:, 'vx_idx']**2 + 
                           radar_points[:, 'vy_idx']**2)
        mask &= velocities > config.min_velocity  # e.g., > 0.5 m/s
    
    # 5. RCS-based filtering (very low RCS often indicates noise)
    if config.filter_by_rcs:
        mask &= radar_points[:, 'rcs_idx'] > config.min_rcs  # e.g., > -5 dBsm
    
    return radar_points[mask]
```

### 4.4 Training Label Assignment with Ghost Awareness

```python
def assign_radar_to_gt(radar_points, gt_boxes, config):
    """
    Assign radar points to ground truth boxes, handling unassigned points
    that may be ghosts or background.
    
    Returns:
        assignments: Per-point GT box index (-1 for background/ghost)
        confidence: Per-assignment confidence score
    """
    assignments = np.full(len(radar_points), -1, dtype=int)
    confidence = np.zeros(len(radar_points))
    
    for i, point in enumerate(radar_points):
        for j, box in enumerate(gt_boxes):
            if point_in_box(point[:3], box):
                assignments[i] = j
                # Higher confidence if velocity matches
                radar_vel = np.array([point[3], point[4]])  # vx, vy
                gt_vel = box[7:9]  # GT velocity
                vel_diff = np.linalg.norm(radar_vel - gt_vel)
                confidence[i] = max(0, 1.0 - vel_diff / 5.0)
                break
    
    return assignments, confidence
```

---

## 5. Quality Assurance for Cross-Modal Annotations

### 5.1 Automated Quality Checks

| Check ID | Description | Threshold | Severity |
|----------|-------------|-----------|----------|
| QA-001 | Box fully outside sensor range | > 80m from ego | Warning |
| QA-002 | Zero radar points in large vehicle box | Car/Truck/Bus | Review |
| QA-003 | Annotation velocity vs radar Doppler mismatch | > 5 m/s diff | Error |
| QA-004 | Box overlaps with another box (IoU > threshold) | IoU > 0.3 | Error |
| QA-005 | Box size outside class-typical range | > 3σ from mean | Warning |
| QA-006 | Tracking discontinuity (ID switch) | Position jump > 5m | Error |
| QA-007 | Object floating above or below ground | |z_bottom| > 0.5m | Warning |
| QA-008 | Rotation discontinuity in tracking | > 30°/frame | Review |
| QA-009 | Radar points outside projected 2D bbox | > 50px offset | Warning |
| QA-010 | Visibility label inconsistent with occlusion | Visible but occluded | Review |

### 5.2 Manual Review Protocol

For annotations flagged by automated checks:

1. **Level 1 Review (Single Annotator):**
   - Verify object presence in camera images (all 6 views)
   - Check radar point association (are assigned points plausible?)
   - Confirm box dimensions match visual appearance
   - Verify heading direction matches vehicle/pedestrian orientation

2. **Level 2 Review (Cross-Check):**
   - Independent annotator reviews flagged boxes
   - Compare velocity annotation with radar Doppler measurements
   - Verify temporal consistency across neighboring keyframes
   - Check that multi-path ghosts are not accidentally annotated

3. **Level 3 Review (Expert):**
   - Resolve disagreements between Level 1 and Level 2
   - Handle ambiguous cases (partial occlusions, distant objects)
   - Make final decisions on edge cases (ghost vs. real, class ambiguity)

### 5.3 Cross-Modal Annotation Verification Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                  Annotation Pipeline                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐       │
│  │ LiDAR    │───►│ Initial  │───►│ Automated QA     │       │
│  │ Annotate │    │ 3D Boxes │    │ Checks (QA-001   │       │
│  └──────────┘    └──────────┘    │ through QA-010)  │       │
│                                  └────────┬─────────┘       │
│                                           │                  │
│                              ┌────────────┼────────────┐     │
│                              │ Pass       │ Fail       │     │
│                              ▼            ▼            │     │
│                    ┌──────────────┐  ┌──────────┐     │     │
│                    │ Camera Cross │  │ Manual   │     │     │
│                    │ Validation   │  │ Review   │     │     │
│                    └──────┬───────┘  └────┬─────┘     │     │
│                           │               │           │     │
│                           ▼               ▼           │     │
│                    ┌──────────────┐  ┌──────────┐    │     │
│                    │ Radar Cross  │  │ Correct  │    │     │
│                    │ Validation   │  │ & Re-QA  │    │     │
│                    └──────┬───────┘  └────┬─────┘    │     │
│                           │               │          │     │
│                           ▼               ▼          │     │
│                    ┌─────────────────────────────┐   │     │
│                    │   Final Annotation Database  │   │     │
│                    └─────────────────────────────┘   │     │
│                                                      │     │
└──────────────────────────────────────────────────────────────┘
```

### 5.4 Metrics for Annotation Quality

| Metric | Description | Target |
|--------|-------------|--------|
| Inter-annotator agreement (IoU) | Mean IoU between independent annotations | > 0.85 |
| Radar association rate | % of annotations with >= 1 radar point | > 70% for cars |
| Camera visibility rate | % of annotations visible in >= 1 camera | > 95% |
| Tracking continuity | % of tracks without ID switches | > 98% |
| Velocity annotation accuracy | RMSE vs radar-derived velocity | < 1.0 m/s |
| Temporal smoothness | Mean acceleration (should be bounded) | < 5 m/s^2 |

### 5.5 Known Annotation Challenges

1. **Distant objects (> 50m):** Radar may detect them but cameras lack resolution for precise annotation
2. **Occluded objects:** Visible to radar (pass-through certain materials) but not to camera
3. **Small objects (cones, pedestrians):** May have zero radar returns due to low RCS
4. **Fast-moving motorcycles:** Radar Doppler may disagree with annotation velocity due to multi-path
5. **Parked vehicles:** Many radar returns but zero velocity creates ambiguity with static background
6. **Object boundaries:** Radar point association is ambiguous when objects are close together
7. **Height estimation:** Radar provides minimal height information, complicating z-annotation verification

---

## 6. Annotation Tools and Workflow

### 6.1 Recommended Annotation Tools

- **3D Annotation:** SUSTechPOINTS, CVAT 3D, or nuScenes devkit visualization tools
- **Cross-modal verification:** Custom overlay tools projecting 3D boxes onto camera images
- **Radar visualization:** BEV plots with radar points colored by velocity/RCS

### 6.2 Workflow for New Data

1. **Initial annotation in LiDAR/camera** (primary modality for shape/position)
2. **Radar overlay verification** (confirm radar points align with annotations)
3. **Velocity annotation refinement** (use radar Doppler to improve velocity labels)
4. **Ghost target review** (identify and remove false radar-only annotations)
5. **Final cross-modal consistency check** (all three modalities agree)
6. **Export in nuScenes format** (JSON annotation files with proper tokens and linking)
