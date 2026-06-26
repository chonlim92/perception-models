# Annotation Guide: 3D Object Detection from Radar Point Clouds

## 1. Overview

This document defines the annotation standards, procedures, and quality criteria for
3D bounding box annotations used in training and evaluating the RadarPillarNet model.
While annotations are typically created from LiDAR and camera data (which provide
higher spatial resolution), they are applied to radar-only detection tasks during
training and evaluation.

## 2. 3D Bounding Box Format

### 2.1 Annotation Representation

Each annotated object is represented as a 3D bounding box with the following parameters:

| Parameter | Description | Unit |
|-----------|-------------|------|
| x | Center position X (ego frame) | meters |
| y | Center position Y (ego frame) | meters |
| z | Center position Z (ego frame) | meters |
| w | Width (lateral extent) | meters |
| l | Length (longitudinal extent) | meters |
| h | Height (vertical extent) | meters |
| yaw | Heading angle (rotation about Z-axis) | radians |
| vx | Velocity in X (global frame) | m/s |
| vy | Velocity in Y (global frame) | m/s |

### 2.2 Coordinate System

- **Origin:** Center of the rear axle of the ego vehicle, projected to ground level
- **X-axis:** Points forward (direction of travel)
- **Y-axis:** Points left (driver's side for left-hand drive)
- **Z-axis:** Points up
- **Yaw:** Counter-clockwise rotation from the positive X-axis in the X-Y plane

### 2.3 Box Center Convention

The (x, y, z) coordinates specify the geometric center of the 3D bounding box, which
is at mid-height of the object. Note: some frameworks use bottom-center; ensure
consistency during data loading.

## 3. Object Class Definitions

### 3.1 Detection Classes

The following 10 classes are used for nuScenes 3D detection:

| Class | Description | Typical Dimensions (L x W x H) |
|-------|-------------|-------------------------------|
| car | Passenger vehicles, sedans, SUVs, hatchbacks | 4.6 x 1.9 x 1.7 m |
| truck | Large commercial vehicles, delivery trucks | 6.9 x 2.5 x 2.9 m |
| bus | Public transit buses, coaches | 11.0 x 2.9 x 3.5 m |
| trailer | Towed cargo containers | 12.3 x 2.9 x 4.0 m |
| construction_vehicle | Cranes, excavators, bulldozers | 6.4 x 2.8 x 3.1 m |
| pedestrian | Walking or standing persons | 0.7 x 0.7 x 1.8 m |
| motorcycle | Two-wheeled motorized vehicles with rider | 2.1 x 0.8 x 1.5 m |
| bicycle | Two-wheeled human-powered vehicles with rider | 1.7 x 0.6 x 1.3 m |
| traffic_cone | Road construction cones | 0.4 x 0.4 x 1.0 m |
| barrier | Road barriers, jersey walls | 2.5 x 0.6 x 1.0 m |

### 3.2 Attribute Annotations

Additional attributes supplement each bounding box:

- **Visibility:** 0-40%, 40-60%, 60-80%, 80-100% (occlusion level)
- **Activity state (vehicles):** moving, stopped, parked, with_rider/without_rider
- **Pose (pedestrians):** standing, sitting_lying_down, moving

## 4. Radar-Specific Annotation Challenges

### 4.1 Ghost Detections and Multipath

Radar signals reflect off multiple surfaces before returning to the sensor, creating
phantom detections at incorrect locations:

- **Guardrail multipath:** A vehicle near a guardrail may produce a "mirror" detection
  on the opposite side of the guardrail
- **Underpass/tunnel reflections:** Overhead structures create ground-level ghost points
- **Inter-vehicle bouncing:** Radar signals bouncing between closely spaced vehicles

**Annotation impact:** Annotations are based on the true object location (from LiDAR/camera).
Ghost detections will appear as false negatives during training, which the model must
learn to ignore.

### 4.2 Limited Elevation Information

Automotive radars provide minimal height information:

- Most detections are projected to approximately z=0 (ground plane)
- Overhead objects (bridges, signs) may appear at ground level
- Stacked objects (vehicle on a trailer) cannot be vertically resolved

**Annotation impact:** The z-coordinate and height (h) of annotations are derived from
LiDAR data, not radar measurements. During inference, the model must predict height
from learned priors rather than direct measurement.

### 4.3 Association Ambiguity

Due to radar's coarse angular resolution, it is often unclear which detection belongs
to which object:

- A single radar return may originate from the front, side, or rear of a vehicle
- Multiple objects at similar ranges may produce merged detections
- Small objects near large objects may not produce independent detections

**Annotation impact:** Ground truth association (which points belong to which box) is
based on spatial containment. Points within the 3D bounding box are considered true
positives for that object.

### 4.4 Missing Detections for Small Objects

Radar has inherently low detection probability for small objects:

- Pedestrians: ~30-50% detection rate per frame (improved with accumulation)
- Bicycles: ~40-60% detection rate per frame
- Traffic cones: ~10-20% detection rate (very low RCS)

**Annotation impact:** Many annotated objects will have zero radar points inside their
bounding box in any given frame. Multi-sweep accumulation partially addresses this,
but some classes remain challenging for radar-only detection.

### 4.5 Velocity Ambiguity

The measured radial velocity depends on the relative geometry between radar and target:

- Only the component along the line of sight is directly measured
- Crossing targets (perpendicular motion) show near-zero radial velocity
- Compensated velocities may contain errors from ego-motion estimation

**Annotation impact:** Velocity annotations (vx, vy) represent the true 2D velocity
of the object derived from tracking across frames, not from radar Doppler directly.

## 5. Annotation Quality for Radar-Only Detection

### 5.1 Annotation Standards

When creating or verifying annotations for radar-only evaluation:

1. **Primary labeling modality:** Use LiDAR point clouds and camera images for annotation
   creation, as they provide sufficient resolution for accurate box placement
2. **Minimum point threshold:** Objects with fewer than 1 radar point (across accumulated
   sweeps) within the bounding box should be flagged but retained in the ground truth
3. **Distance-dependent quality:** Annotation accuracy naturally decreases with range;
   focus quality assurance on objects within 50m
4. **Velocity verification:** Cross-check annotated velocities against radar compensated
   velocities when points are available within the box

### 5.2 Quality Metrics

| Criterion | Acceptable Range |
|-----------|-----------------|
| Position accuracy (x, y) | ±0.3 m for vehicles, ±0.5 m for VRU |
| Size accuracy (l, w, h) | ±0.3 m |
| Heading accuracy (yaw) | ±5 degrees |
| Velocity accuracy (vx, vy) | ±0.5 m/s |

### 5.3 Special Handling

- **Occluded objects:** Include in ground truth if any part is visible to any sensor;
  mark with appropriate visibility attribute
- **Distant objects (>70m):** Include but expect lower detection rates from radar
- **Parked vehicles:** Include and annotate with zero velocity
- **Groups of pedestrians:** Annotate each individual separately

## 6. Annotation Tools and Workflow

### 6.1 Recommended Workflow

1. Generate initial annotations from LiDAR point cloud using 3D annotation tools
2. Refine using camera images for class verification
3. Interpolate between keyframes for smooth trajectories
4. Verify velocity consistency across the track
5. Cross-check with radar data to validate radar-visible objects

### 6.2 Validation Checks

Automated validation scripts should verify:

- No overlapping bounding boxes of the same class
- Velocity consistency between consecutive frames
- Size consistency across the track (objects don't change size)
- All required attributes are present and valid
- Class-size consistency (e.g., cars aren't truck-sized)

## 7. Evaluation Considerations

### 7.1 Distance-Stratified Evaluation

Due to radar's range-dependent performance, evaluate separately at:

- Near range: 0-30 m
- Mid range: 30-50 m
- Far range: 50-70 m
- Extended range: 70+ m

### 7.2 Class-Specific Expectations

Not all classes are equally detectable by radar:

- **High confidence:** car, truck, bus (large RCS, many radar returns)
- **Medium confidence:** motorcycle, construction_vehicle (moderate RCS)
- **Low confidence:** pedestrian, bicycle (small RCS, few returns)
- **Very low confidence:** traffic_cone, barrier (minimal RCS, static)
