# Data Collection: nuScenes Dataset for DETR3D

## Overview

DETR3D is trained and evaluated on the nuScenes dataset, a large-scale autonomous driving benchmark developed by Motional (formerly nuTonomy). The dataset provides synchronized multi-modal sensor data collected in challenging urban driving scenarios across two cities with diverse traffic conditions.

---

## Multi-Camera Setup

The nuScenes data collection vehicle is equipped with 6 cameras providing full 360-degree surround coverage:

### Camera Configuration

| Camera | Field of View | Orientation | Primary Coverage |
|--------|--------------|-------------|-----------------|
| CAM_FRONT | 70 degrees | Forward-facing | Road ahead, vehicles, pedestrians |
| CAM_FRONT_LEFT | 70 degrees | 55 degrees left of center | Left-forward intersection area |
| CAM_FRONT_RIGHT | 70 degrees | 55 degrees right of center | Right-forward intersection area |
| CAM_BACK | 110 degrees | Rear-facing | Vehicles behind, reversing |
| CAM_BACK_LEFT | 70 degrees | 110 degrees left of center | Left-rear blind spot area |
| CAM_BACK_RIGHT | 70 degrees | 110 degrees right of center | Right-rear blind spot area |

### Camera Specifications
- **Resolution:** 1600 x 900 pixels
- **Frame rate:** 12 Hz (captured), 2 Hz (annotated keyframes)
- **Image format:** JPEG
- **Color space:** RGB, 8-bit per channel
- **Sensor type:** Basler acA1600-60gc machine vision cameras
- **Mounting:** Roof-mounted camera rig ensuring minimal occlusion and vibration

### Coverage Geometry
- The 6 cameras collectively cover the full 360-degree horizontal field around the ego vehicle.
- There is slight overlap between adjacent cameras (approximately 10-15 degrees), enabling cross-camera consistency verification.
- The back camera has a wider FOV (110 degrees) compared to the other five (70 degrees) to compensate for fewer rear cameras.
- Vertical FOV coverage captures both nearby ground-level objects and overhead structures.

---

## LiDAR Sensor (Ground Truth Generation)

### 32-Beam LiDAR
- **Sensor:** Velodyne HDL-32E
- **Beams:** 32 channels
- **Rotation rate:** 20 Hz
- **Range:** Up to 70 meters (effective), 100 meters (maximum)
- **Points per second:** ~1.39 million
- **Vertical FOV:** +10.67 degrees to -30.67 degrees
- **Angular resolution:** ~1.33 degrees vertical, ~0.1-0.4 degrees horizontal
- **Mounting:** Roof-mounted, centered on vehicle roofline

### Role in DETR3D Pipeline
The LiDAR sensor is **not used as input** to DETR3D during inference. Its role is exclusively for:
- Generating accurate 3D ground-truth bounding box annotations
- Providing point cloud data for annotators to precisely localize objects in 3D space
- Validating annotation quality through point cloud density analysis
- Enabling comparison with LiDAR-based detection methods on the same benchmark

---

## Radar Sensors

### Configuration
- **Sensor count:** 5 Continental ARS 408-21 radar units
- **Placement:** Front, front-left, front-right, back-left, back-right
- **Range:** Up to 250 meters
- **Output:** Radar point cloud with velocity information (Doppler)
- **Frame rate:** 13 Hz

### Radar Data Characteristics
- Provides sparse detections with radial velocity measurements
- Useful for velocity estimation ground truth validation
- Not used as input to DETR3D but available in the dataset for multi-modal fusion research
- Each radar return includes: x, y position, radial velocity, RCS (radar cross-section), and detection confidence

---

## Data Format and Timing

### Keyframes (Annotated Samples)
- **Frequency:** 2 Hz (one keyframe every 0.5 seconds)
- **Content:** Full sensor suite synchronized capture with 3D bounding box annotations
- **Total keyframes:** ~400,000 across the full dataset
- **Annotation completeness:** All visible objects within annotation range are labeled
- **Synchronization:** All 6 cameras and LiDAR are timestamp-synchronized within 50ms tolerance

### Sweeps (Intermediate Frames)
- **Frequency:** ~12 Hz for cameras, 20 Hz for LiDAR
- **Content:** Raw sensor data without annotations
- **Purpose:** Enable temporal modeling, motion estimation, and data augmentation
- **Availability:** ~10 sweeps between consecutive keyframes (for cameras)
- **Usage in DETR3D:** Sweeps can be used for temporal data augmentation but are not annotated

### Data Organization
```
nuScenes/
├── maps/                    # HD maps (lane boundaries, walkways, etc.)
├── samples/                 # Keyframe sensor data (annotated)
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── LIDAR_TOP/
│   ├── RADAR_FRONT/
│   └── ...
├── sweeps/                  # Intermediate frames (unannotated)
│   ├── CAM_FRONT/
│   └── ...
├── v1.0-trainval/          # Annotation JSON files
│   ├── sample.json
│   ├── sample_data.json
│   ├── sample_annotation.json
│   ├── ego_pose.json
│   ├── calibrated_sensor.json
│   ├── scene.json
│   ├── log.json
│   └── ...
└── v1.0-test/              # Test set (annotations withheld)
```

### Calibration Data
- **Intrinsic calibration:** Camera focal lengths, principal points, distortion coefficients
- **Extrinsic calibration:** 6-DOF transformation from each sensor to the ego-vehicle frame
- **Ego pose:** Vehicle position and orientation in the global coordinate frame at each timestamp
- **Format:** Stored as 4x4 transformation matrices and 3x3 intrinsic matrices in JSON

---

## Geographic Locations

### Boston, Massachusetts (USA)
- **Neighborhood:** Seaport District (Boston Harbor area)
- **Characteristics:**
  - Dense urban environment with high-rise buildings
  - Construction zones with heavy machinery
  - Mixed traffic: pedestrians, cyclists, cars, trucks, buses
  - Complex intersections with traffic signals
  - Weather: varied (clear, cloudy, rain)
  - Driving side: Right-hand traffic

### Singapore
- **Neighborhood:** One-North district (technology and business park area)
- **Characteristics:**
  - Tropical urban environment
  - Dense vegetation alongside roads
  - Motorcycles and scooters prevalent
  - Multi-lane roads with complex merging
  - Weather: tropical (clear, overcast, rain, post-rain wet surfaces)
  - Driving side: Left-hand traffic
  - Different road markings and signage conventions

### Geographic Diversity Value
- Two continents provide diversity in driving behaviors, traffic patterns, and road infrastructure
- Different driving sides (left vs. right) stress-test the model's rotational invariance
- Varied weather conditions test robustness across environmental conditions
- Different urban planning styles create diverse scene layouts

---

## Dataset Statistics

### Scene Breakdown
- **Total scenes:** 1,000 (each approximately 20 seconds long)
- **Training set:** 700 scenes (28,130 keyframes)
- **Validation set:** 150 scenes (6,019 keyframes)
- **Test set:** 150 scenes (6,008 keyframes)
- **Total annotated 3D boxes:** ~1.4 million across all keyframes

### Object Distribution
| Class | Approximate Count | Frequency |
|-------|-------------------|-----------|
| Car | ~400,000 | Very common |
| Pedestrian | ~180,000 | Common |
| Barrier | ~130,000 | Common |
| Traffic Cone | ~80,000 | Moderate |
| Truck | ~70,000 | Moderate |
| Trailer | ~20,000 | Rare |
| Bus | ~12,000 | Rare |
| Construction Vehicle | ~11,000 | Rare |
| Motorcycle | ~9,000 | Rare |
| Bicycle | ~8,000 | Rare |

This severe class imbalance motivates the use of CBGS (Class-Balanced Grouping and Sampling) during DETR3D training.

---

## Data Collection Process

### Vehicle Platform
- Modified Renault Zoe electric vehicle
- Fully integrated sensor suite with rigid mounting
- GPS/IMU for precise ego-motion tracking (Novatel SPAN-CPT)
- Custom data acquisition system with hardware timestamping

### Collection Protocol
- Driven by professional operators in normal traffic conditions
- Routes cover varied scenarios: highways, intersections, parking lots, residential areas
- Minimum 20-second continuous driving per scene
- Data collected during day and night, across different weather conditions
- Approximately 15 hours of total driving data

### Quality Assurance
- Sensor calibration verified before and after each collection run
- GPS/IMU accuracy: centimeter-level positioning
- Clock synchronization: hardware PPS (pulse-per-second) signal ensures <1ms timing accuracy
- Post-processing: ego-motion refined using SLAM for globally consistent poses
