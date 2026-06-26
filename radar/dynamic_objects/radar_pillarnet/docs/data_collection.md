# Data Collection: Radar Point Clouds for RadarPillarNet

## 1. Overview

This document describes the radar data collection setup, data formats, and preprocessing
strategies used for training and evaluating the RadarPillarNet model. The primary dataset
is nuScenes, which provides synchronized multi-modal sensor data including radar.

## 2. nuScenes Radar Sensor Setup

### 2.1 Sensor Configuration

The nuScenes dataset uses **5 Continental ARS408-21** long-range radar sensors providing
full 360-degree coverage around the ego vehicle:

| Sensor ID | Position | Orientation | FOV (azimuth) |
|-----------|----------|-------------|----------------|
| RADAR_FRONT | Front bumper, center | Forward (0 deg) | ±60 degrees |
| RADAR_FRONT_LEFT | Front bumper, left | ~55 deg left | ±60 degrees |
| RADAR_FRONT_RIGHT | Front bumper, right | ~55 deg right | ±60 degrees |
| RADAR_BACK_LEFT | Rear bumper, left | ~135 deg left | ±60 degrees |
| RADAR_BACK_RIGHT | Rear bumper, right | ~135 deg right | ±60 degrees |

### 2.2 Continental ARS408-21 Specifications

- **Measurement range:** 0.2 m to 250 m (long range mode)
- **Range accuracy:** ±0.1 m (typical)
- **Velocity range:** -400 to +200 km/h
- **Velocity accuracy:** ±0.1 m/s
- **Cycle time:** 72 ms (approx. 13.9 Hz)
- **Detection capability:** Up to 100 tracked objects per cycle
- **Output format:** Cluster list and object list (nuScenes uses cluster list)

### 2.3 Coverage Overlap

The five radars provide overlapping coverage in the forward and rear sectors:

```
                 FRONT
                 (±60°)
                  /  \
    FRONT_LEFT  /    \  FRONT_RIGHT
    (±60°)     /      \    (±60°)
              /        \
    ---------[  EGO   ]----------
              \        /
    BACK_LEFT  \      /  BACK_RIGHT
    (±60°)     \    /    (±60°)
                \  /
```

Overlap regions provide redundant detections that improve reliability.

## 3. Radar Data Format

### 3.1 Point Cloud Fields

Each radar detection (point) contains the following fields in the nuScenes dataset:

| Field | Type | Description | Unit |
|-------|------|-------------|------|
| x | float32 | Position in sensor X (forward) | meters |
| y | float32 | Position in sensor Y (left) | meters |
| z | float32 | Position in sensor Z (up) | meters |
| dyn_prop | int8 | Dynamic property classification | enum |
| id | int16 | Measurement ID | - |
| rcs | float32 | Radar cross section | dBsm |
| vx | float32 | Velocity in X (sensor frame) | m/s |
| vy | float32 | Velocity in Y (sensor frame) | m/s |
| vx_comp | float32 | Compensated velocity in X (ego-motion removed) | m/s |
| vy_comp | float32 | Compensated velocity in Y (ego-motion removed) | m/s |
| is_quality_valid | bool | Quality flag | - |
| ambig_state | int8 | Ambiguity state | enum |
| x_rms | int8 | X position standard deviation | index |
| y_rms | int8 | Y position standard deviation | index |
| invalid_state | int8 | Invalid state flag | enum |
| pdh0 | int8 | Probability of false detection | index |
| vx_rms | int8 | X velocity standard deviation | index |
| vy_rms | int8 | Y velocity standard deviation | index |

### 3.2 Dynamic Property (dyn_prop) Values

The `dyn_prop` field classifies each detection's motion state:

| Value | Description |
|-------|-------------|
| 0 | Moving |
| 1 | Stationary |
| 2 | Oncoming |
| 3 | Stationary candidate |
| 4 | Unknown |
| 5 | Crossing stationary |
| 6 | Crossing moving |
| 7 | Stopped |

### 3.3 Data Storage Format

In the nuScenes devkit, radar point clouds are stored as binary `.pcd` files:

- Format: Point Cloud Data (PCD) version 0.7
- Encoding: Binary
- Typical file size: 2-10 KB per sweep (due to low point count)
- One file per radar sensor per keyframe

## 4. Multi-Sweep Accumulation Strategy

### 4.1 Default Configuration

The standard accumulation configuration for RadarPillarNet:

```
n_sweeps: 6          # Number of sweeps to accumulate (including current)
sweep_interval: 0.05 # Seconds between sweeps (~20 Hz effective)
max_time_lag: 0.5    # Maximum time window for accumulation (seconds)
```

### 4.2 Accumulation Procedure

For each sample at time t, the accumulation procedure is:

1. Load the current sweep radar points for all 5 sensors
2. For each of the previous (n_sweeps - 1) sweeps:
   a. Identify the closest radar sweep within the time window
   b. Load the radar points from that sweep
   c. Compute the ego-motion transform between sweep time and current time
   d. Transform historical points into the current ego-vehicle frame
   e. Append a relative timestamp feature (dt) to each point
3. Concatenate all transformed points into a single point cloud
4. Filter points outside the detection range

### 4.3 Ego-Motion Compensation

The transformation pipeline for a point from sweep at time t_past to the current frame at
time t_now:

```
T_global_from_ego_past   = ego_pose(t_past)
T_global_from_ego_now    = ego_pose(t_now)
T_ego_now_from_ego_past  = inv(T_global_from_ego_now) @ T_global_from_ego_past

T_ego_from_sensor        = calibration(sensor_id)

# Full transform for a point in sensor frame at t_past to ego frame at t_now:
P_ego_now = T_ego_now_from_ego_past @ T_ego_from_sensor @ P_sensor_past
```

### 4.4 Coordinate Frame Definitions

- **Sensor frame:** Origin at radar sensor, X forward, Y left, Z up
- **Ego frame:** Origin at rear axle center, X forward, Y left, Z up
- **Global frame:** UTM-like global coordinate system

## 5. Data Statistics

### 5.1 Point Count Distribution (after 6-sweep accumulation)

| Statistic | All 5 Radars Combined |
|-----------|----------------------|
| Mean points per sample | ~1,200 |
| Median points per sample | ~1,100 |
| Min points per sample | ~300 |
| Max points per sample | ~3,000 |
| Std. dev. | ~400 |

### 5.2 RCS Distribution

| Statistic | Value (dBsm) |
|-----------|--------------|
| Mean | 5.2 |
| Std. dev. | 8.7 |
| Min | -64.0 |
| Max | 47.0 |
| 25th percentile | -0.5 |
| 75th percentile | 11.0 |

### 5.3 Velocity Distribution (compensated)

| Statistic | vx_comp (m/s) | vy_comp (m/s) |
|-----------|---------------|---------------|
| Mean | -0.3 | 0.0 |
| Std. dev. | 3.8 | 1.5 |
| Range | [-30, +30] | [-15, +15] |

### 5.4 Spatial Distribution

Most radar points concentrate in the forward-facing direction due to the narrower
beam pattern and typical driving scenarios:

- ~40% of points from RADAR_FRONT
- ~20% from RADAR_FRONT_LEFT
- ~20% from RADAR_FRONT_RIGHT
- ~10% from RADAR_BACK_LEFT
- ~10% from RADAR_BACK_RIGHT

## 6. Data Preprocessing

### 6.1 Point Filtering

Before training, apply the following filters:

```python
# Range filter (meters)
x_range = [-51.2, 51.2]
y_range = [-51.2, 51.2]
z_range = [-5.0, 3.0]

# Remove invalid detections
keep = (points['invalid_state'] == 0)
keep &= (points['pdh0'] < 4)  # Low false alarm probability
```

### 6.2 Feature Selection

The final feature vector per point for RadarPillarNet (9 features):

| Feature | Description | Normalization |
|---------|-------------|---------------|
| x | Position X in ego frame | Raw (meters) |
| y | Position Y in ego frame | Raw (meters) |
| z | Position Z in ego frame | Raw (meters) |
| rcs | Radar cross section | Raw (dBsm) |
| vx_comp | Compensated velocity X | Raw (m/s) |
| vy_comp | Compensated velocity Y | Raw (m/s) |
| dt | Relative timestamp | Normalized [0, 1] |
| xc | X offset to pillar center | Raw (meters) |
| yc | Y offset to pillar center | Raw (meters) |

### 6.3 Dataset Splits

nuScenes official splits for the detection task:

| Split | Scenes | Keyframes |
|-------|--------|-----------|
| Train | 700 | 28,130 |
| Validation | 150 | 6,019 |
| Test | 150 | 6,008 |

## 7. Data Quality Considerations

### 7.1 Known Issues

- **Height ambiguity:** The z-coordinate from radar is unreliable; some implementations
  fix z=0 for all radar points
- **Velocity noise near zero:** Compensated velocities for truly stationary objects may
  show small residual values due to ego-motion compensation errors
- **Ghost detections:** Multipath reflections create false points, especially near
  metallic structures and guardrails
- **Sensor synchronization:** Radar operates asynchronously; timestamps may not align
  perfectly with keyframe timestamps

### 7.2 Mitigation Strategies

- Use `invalid_state` and `pdh0` fields to filter low-quality detections
- Apply velocity thresholding for static/dynamic classification
- Leverage multi-sweep consistency to identify persistent vs. transient detections
- Use `dyn_prop` as an additional input feature rather than hard filtering
