# RangeNet++: Data Collection

## SemanticKITTI Dataset

### Overview

SemanticKITTI is the primary dataset used for training and evaluating RangeNet++. It provides dense point-wise semantic annotations for the full KITTI Odometry Benchmark sequences.

- **Paper:** "SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences" (Behley et al., ICCV 2019)
- **Website:** http://www.semantic-kitti.org
- **Total scans:** 43,552 LiDAR scans
- **Total annotated points:** ~4.549 billion points
- **Classes:** 28 classes (mapped to 19 evaluation classes + 1 unlabeled)
- **Sensor:** Velodyne HDL-64E

### Sequence Split

| Split | Sequences | Scans | Purpose |
|-------|-----------|-------|---------|
| Training | 00-07, 09-10 | 19,130 | Model training |
| Validation | 08 | 4,071 | Hyperparameter tuning |
| Test | 11-21 | 20,351 | Official evaluation (labels withheld) |

### Sequence Details

| Sequence | Scans | Environment |
|----------|-------|-------------|
| 00 | 4,541 | Urban, residential |
| 01 | 1,101 | Highway |
| 02 | 4,661 | Urban, residential |
| 03 | 801 | Rural |
| 04 | 271 | Rural |
| 05 | 2,761 | Urban, residential |
| 06 | 1,101 | Urban, residential |
| 07 | 1,101 | Urban, residential |
| 08 | 4,071 | Urban, residential |
| 09 | 1,591 | Urban, residential |
| 10 | 1,201 | Urban, residential |
| 11-21 | ~20,351 | Various (test set) |

### Data Format

Each scan is stored as:
- **Point cloud:** `.bin` file (N x 4 float32: x, y, z, remission/intensity)
- **Labels:** `.label` file (N x uint32: lower 16 bits = semantic label, upper 16 bits = instance ID)

```
dataset/
  sequences/
    00/
      velodyne/        # Point clouds (.bin)
        000000.bin
        000001.bin
        ...
      labels/          # Semantic labels (.label)
        000000.label
        000001.label
        ...
      calib.txt        # Calibration data
      poses.txt        # Vehicle poses (for multi-scan)
    01/
    ...
    21/
```

### Point Cloud Format

Each `.bin` file contains points as contiguous float32 values:
```
[x1, y1, z1, intensity1, x2, y2, z2, intensity2, ...]
```

- **x, y, z:** 3D coordinates in the sensor frame (meters)
- **intensity:** Remission/reflectance value (0.0 - 1.0, normalized)

### Label Format

Each `.label` file contains uint32 values:
```
semantic_label = label & 0xFFFF        # Lower 16 bits
instance_id    = (label >> 16) & 0xFFFF # Upper 16 bits
```

---

## Velodyne HDL-64E Sensor

### Specifications

| Parameter | Value |
|-----------|-------|
| Model | Velodyne HDL-64E S2 |
| Beams | 64 laser/detector pairs |
| Rotation rate | 10 Hz (configurable 5-20 Hz) |
| Points per revolution | ~130,000 (at 10 Hz) |
| Points per second | ~1.3 million |
| Vertical FOV | -24.8 to +2.0 degrees (26.8 total) |
| Vertical angular resolution | ~0.4 degrees (non-uniform) |
| Horizontal FOV | 360 degrees |
| Horizontal angular resolution | ~0.08-0.35 degrees (dependent on rotation rate) |
| Range | 120 m (up to 50 m for pavement) |
| Range accuracy | +/- 2 cm (1 sigma) |
| Wavelength | 905 nm |

### Beam Configuration

The 64 beams are divided into two blocks:
- **Upper block (32 beams):** -8.33 to +2.0 degrees (finer spacing above horizon)
- **Lower block (32 beams):** -24.8 to -8.53 degrees (captures ground and nearby objects)

The non-uniform vertical spacing means more beams are concentrated near the horizon where most objects of interest (vehicles, pedestrians) appear.

### Mounting Configuration (KITTI)

- **Height:** ~1.73 m above ground
- **Position:** Roof-mounted on Volkswagen Passat station wagon
- **Orientation:** Level with ground plane

### Output Characteristics

- **Coordinate frame:** Right-handed, x-forward, y-left, z-up
- **Point density:** Decreases with distance (1/r^2)
- **Typical scan points:** 100,000 - 130,000 points per rotation
- **Shadow effects:** Occlusion causes gaps behind objects
- **Intensity:** Depends on surface reflectivity, angle of incidence, and range

---

## Data Preprocessing for RangeNet++

### Range Image Generation

The 3D point cloud is projected to a 2D range image using spherical projection:

1. **Compute spherical coordinates** for each point (x, y, z):
   - Azimuth: `phi = atan2(y, x)`
   - Elevation: `theta = arcsin(z / sqrt(x^2 + y^2 + z^2))`
   - Range: `r = sqrt(x^2 + y^2 + z^2)`

2. **Map to pixel coordinates:**
   - `u = 0.5 * (1 + phi / pi) * W`
   - `v = (1 - (theta - fov_down) / fov) * H`

3. **Fill range image** (H x W x 5):
   - Channel 0: Range (r)
   - Channel 1: x coordinate
   - Channel 2: y coordinate
   - Channel 3: z coordinate
   - Channel 4: Intensity/remission

### Image Dimensions

| Configuration | Height (H) | Width (W) | Notes |
|---------------|-----------|----------|-------|
| Full resolution | 64 | 2048 | Best accuracy |
| Half resolution | 64 | 1024 | Faster inference |

### Handling Multiple Points per Pixel

When multiple 3D points project to the same pixel:
- Keep the point with the **smallest range** (closest to sensor)
- This mimics natural occlusion behavior

### Empty Pixel Handling

Pixels with no projected points are set to zero across all channels and masked during training (loss not computed for empty pixels).

---

## Data Statistics

### Point Distribution per Scan

- **Average points per scan:** ~120,000
- **Range image occupancy:** ~64% (at 64x2048 resolution)
- **Empty pixels:** ~36% (sky, beyond max range)

### Class Distribution (Training Set)

The dataset exhibits significant class imbalance:

| Class | Percentage of Points |
|-------|---------------------|
| Road | ~15% |
| Vegetation | ~18% |
| Building | ~12% |
| Car | ~4% |
| Sidewalk | ~8% |
| Terrain | ~7% |
| Other-ground | ~3% |
| Fence | ~4% |
| Pole | ~0.5% |
| Traffic-sign | ~0.2% |
| Person | ~0.1% |
| Bicyclist | ~0.1% |
| Motorcyclist | ~0.01% |

This severe imbalance necessitates class-weighted loss functions during training.
