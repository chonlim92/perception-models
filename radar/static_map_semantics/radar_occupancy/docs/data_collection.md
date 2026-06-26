# Data Collection — Radar Occupancy Grid Mapping

## Dataset: nuScenes

### Radar Sensor Setup

nuScenes uses **5 Continental ARS 408-21** long-range radar sensors:
- RADAR_FRONT: forward-facing, center
- RADAR_FRONT_LEFT: 60° left of center
- RADAR_FRONT_RIGHT: 60° right of center
- RADAR_BACK_LEFT: 120° left of center (rear)
- RADAR_BACK_RIGHT: 120° right of center (rear)

### Radar Specifications (ARS 408-21)

| Parameter | Value |
|-----------|-------|
| Max range | 250 m |
| Range resolution | 0.39 m |
| Range accuracy | ±0.1 m |
| Azimuth FOV | ±60° |
| Azimuth resolution | 1.8° |
| Elevation FOV | ±5° |
| Update rate | 13 Hz |
| Doppler range | ±69 m/s |
| Velocity resolution | 0.16 m/s |

### Radar Point Format (nuScenes)

Each radar detection contains 18 features:
```
x, y, z                  # Position in sensor frame (meters)
dyn_prop                 # Dynamic property (moving/stationary classification)
id                       # Detection ID
rcs                      # Radar Cross Section (dBsm)
vx, vy                   # Velocity components (m/s, compensated)
vx_comp, vy_comp         # Velocity (ego-motion compensated)
is_quality_valid         # Quality flag
ambig_state              # Ambiguity state
x_rms, y_rms             # Position uncertainty (m)
invalid_state            # Validity flag
pdh0                     # False alarm probability
vx_rms, vy_rms           # Velocity uncertainty (m/s)
```

### Download Instructions

```bash
# 1. Register at https://www.nuscenes.org/
# 2. Download the full dataset (or mini for testing)

# Mini split (~4GB, 10 scenes)
wget https://www.nuscenes.org/data/v1.0-mini.tgz

# Full dataset (~85GB)
wget https://www.nuscenes.org/data/v1.0-trainval01_blobs.tgz
wget https://www.nuscenes.org/data/v1.0-trainval02_blobs.tgz
# ... (multiple archives)

# 3. Extract to data directory
mkdir -p data/nuscenes
tar -xzf v1.0-mini.tgz -C data/nuscenes/
```

### Multi-Sweep Accumulation

Due to radar sparsity, we accumulate multiple sweeps:
- Collect past N radar sweeps (typically N=6, ~0.5 seconds)
- Transform each sweep to current ego frame using ego-pose
- Concatenate all points with timestamp offset as additional feature

```python
# Accumulation produces ~600-1800 points vs ~100-300 from single sweep
features_per_point = [x, y, z, rcs, vr_comp, dt]  # dt = time offset from current
```

## Ground Truth Generation

### From LiDAR (Proxy Ground Truth)

Since dense radar occupancy GT doesn't exist, we derive it from LiDAR:

1. **LiDAR-based occupancy:** Use dense LiDAR point cloud as ground truth for occupied cells
2. **Ray-casting for free space:** Cast rays from LiDAR origin through each detection — cells traversed are free
3. **Aggregate multiple LiDAR sweeps** for denser ground truth

### From 3D Bounding Boxes

Alternative: use annotated 3D boxes as occupied regions:
- Rasterize 3D boxes into BEV grid
- Everything else within LiDAR range = free
- Beyond LiDAR range = unknown

### Grid Parameters

| Parameter | Value |
|-----------|-------|
| BEV range X | [-50m, 50m] |
| BEV range Y | [-50m, 50m] |
| Cell size | 0.25m × 0.25m |
| Grid size | 400 × 400 cells |
| Height range | [-3m, 5m] |
| Classes | Free (0), Occupied (1), Unknown (2) |
| Semantic classes | Free, Vehicle, Pedestrian, Barrier, Other |

## Data Pipeline

```
Raw radar PCD files → Multi-sweep accumulation (ego-motion compensated) →
Filter (remove invalid, RCS threshold) → Normalize features →
Create input BEV grid (pillar encoding OR scatter) →
Load GT occupancy (from LiDAR/boxes) → Apply augmentations → Batch
```
