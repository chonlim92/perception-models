# Radar Occupancy Grid Mapping — Annotation Guide

## Ground Truth Generation, Labeling Conventions, and Quality Control

---

## 1. Overview

This guide describes how occupancy ground truth is generated for training and evaluating radar-based occupancy grid models. Since radar alone is too sparse to provide ground truth directly, we derive labels from dense LiDAR point clouds, which serve as a proxy for the true occupancy state of the environment.

### Annotation Scope

| Aspect | Specification |
|--------|--------------|
| Ground truth source | LiDAR point cloud (32/64-beam spinning LiDAR) |
| Annotation type | Per-cell occupancy labels in BEV grid |
| Cell resolution | 0.5m x 0.5m |
| Grid dimensions | 200 x 200 cells (100m x 100m coverage) |
| Label set | Free (0), Occupied (1), Unknown (2) |
| Semantic classes | Free, Vehicle, Pedestrian, Barrier, Other |
| Update rate | 2 Hz (at nuScenes keyframes) |
| Coordinate frame | Ego vehicle frame at each keyframe |

---

## 2. Occupancy Ground Truth Generation from LiDAR

### 2.1 Why LiDAR as Ground Truth?

LiDAR provides 30,000-300,000 points per scan (vs radar's 100-300), giving a sufficiently dense representation to determine which cells are occupied and which are free. The process relies on two fundamental geometric observations:

1. **Occupied cells**: Cells containing LiDAR return points are occupied (something reflected the laser)
2. **Free cells**: Cells along the ray between the sensor and the return point must be free (the laser passed through them unobstructed)

### 2.2 Step-by-Step Ground Truth Pipeline

```
LiDAR Point Cloud (100k+ points per scan)
    |
    v
[Step 1] Aggregate multiple LiDAR sweeps (ego-motion compensated)
    |
    v
[Step 2] Filter points by height range (z in [-3m, 5m])
    |
    v
[Step 3] Project points to BEV grid -> mark cells as OCCUPIED (1)
    |
    v
[Step 4] Ray-cast from LiDAR origin through each point -> mark traversed cells as FREE (0)
    |
    v
[Step 5] Mark all remaining cells as UNKNOWN (2) -> these are ignored during training
    |
    v
Ground Truth Occupancy Grid [200, 200] with values {0, 1, 2}
```

### 2.3 Implementation Details

```python
import numpy as np

def generate_occupancy_gt(lidar_points, lidar_origin, grid_config):
    """
    Generate occupancy ground truth from a LiDAR point cloud.
    
    Args:
        lidar_points: (N, 3) array of [x, y, z] in ego frame
        lidar_origin: (3,) sensor origin in ego frame
        grid_config: dict with x_range, y_range, cell_size, grid_size
        
    Returns:
        occupancy_gt: (H, W) array with values 0=free, 1=occupied, 2=unknown
    """
    H, W = grid_config["grid_size"]
    x_range = grid_config["x_range"]
    y_range = grid_config["y_range"]
    cell_size = grid_config["cell_size"]
    
    # Initialize all cells as UNKNOWN
    gt = np.full((H, W), fill_value=2, dtype=np.int32)
    
    # Step 1: Mark OCCUPIED cells (cells containing LiDAR points)
    for point in lidar_points:
        gx = int((point[0] - x_range[0]) / cell_size)
        gy = int((point[1] - y_range[0]) / cell_size)
        if 0 <= gx < H and 0 <= gy < W:
            gt[gx, gy] = 1  # OCCUPIED
    
    # Step 2: Ray-cast to mark FREE cells
    origin_gx = int((lidar_origin[0] - x_range[0]) / cell_size)
    origin_gy = int((lidar_origin[1] - y_range[0]) / cell_size)
    
    for point in lidar_points:
        end_gx = int((point[0] - x_range[0]) / cell_size)
        end_gy = int((point[1] - y_range[0]) / cell_size)
        
        # Bresenham ray from origin to point (exclusive of endpoint)
        ray_cells = bresenham_2d(origin_gx, origin_gy, end_gx, end_gy)
        for (rx, ry) in ray_cells[:-1]:  # Exclude the endpoint (occupied)
            if 0 <= rx < H and 0 <= ry < W:
                if gt[rx, ry] == 2:  # Only overwrite UNKNOWN, not OCCUPIED
                    gt[rx, ry] = 0   # FREE
    
    return gt
```

### 2.4 Multi-Sweep Aggregation for Denser Ground Truth

Single LiDAR scans may miss thin structures. Aggregating multiple sweeps improves coverage:

```python
def generate_dense_gt(lidar_sweeps, ego_poses, current_pose, grid_config):
    """
    Aggregate multiple LiDAR sweeps into a single dense ground truth.
    
    Args:
        lidar_sweeps: list of (N_i, 3) point clouds in sensor frame
        ego_poses: list of (4, 4) ego poses for each sweep
        current_pose: (4, 4) current keyframe ego pose
        grid_config: grid configuration dictionary
    """
    all_points = []
    
    for points, pose in zip(lidar_sweeps, ego_poses):
        # Transform to current ego frame
        T_relative = np.linalg.inv(current_pose) @ pose
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        transformed = (T_relative @ points_homo.T).T[:, :3]
        all_points.append(transformed)
    
    aggregated = np.vstack(all_points)
    
    # Filter by height range
    z_mask = (aggregated[:, 2] > -3.0) & (aggregated[:, 2] < 5.0)
    aggregated = aggregated[z_mask]
    
    # Generate GT from aggregated dense cloud
    lidar_origin = np.array([0.0, 0.0, 1.8])  # Approximate LiDAR height on ego
    return generate_occupancy_gt(aggregated, lidar_origin, grid_config)
```

---

## 3. Labeling Conventions

### 3.1 Binary Occupancy Labels

| Label Value | Meaning | Description |
|-------------|---------|-------------|
| 0 | Free | Cell is confirmed empty (LiDAR ray passed through) |
| 1 | Occupied | Cell contains a physical object (LiDAR returned a point here) |
| 2 | Unknown | No LiDAR information available (behind obstacles, beyond range) |

**Critical rules:**
- Unknown cells are **ignored** in both loss computation and metric evaluation
- A cell is marked occupied if **any** LiDAR point falls within it
- Free cells must have explicit evidence (ray traversal), not absence of evidence

### 3.2 Semantic Class Definitions

When the semantic head is enabled, occupied cells are further classified:

| Class ID | Class Name | Definition | Examples |
|----------|-----------|------------|----------|
| 0 | Free | Drivable/traversable space | Road surface, sidewalks (when accessible) |
| 1 | Vehicle | Motorized transport | Cars, trucks, buses, motorcycles |
| 2 | Pedestrian | Humans and personal mobility | People, wheelchairs, strollers |
| 3 | Barrier | Physical separation structures | Guardrails, walls, fences, jersey barriers |
| 4 | Other | All other occupied space | Buildings, vegetation, poles, signs, debris |

**Semantic label sources:**
- Derived from nuScenes 3D bounding box annotations (rasterized to grid)
- Non-annotated occupied cells default to class 4 (Other)
- Static infrastructure (buildings, walls) comes from map annotations where available

### 3.3 Coordinate Convention

```
BEV Grid Layout (ego at center):

     Y (forward)
     ^
     |
     |    ┌───────────────────────┐
     |    │                       │
     |    │     Grid [200x200]    │
     |    │                       │
     |    │          ▓▓           │  ▓▓ = Ego vehicle
     |    │        (100,100)      │
     |    │                       │
     |    │                       │
     |    └───────────────────────┘
     └────────────────────────────────> X (right)
     
Grid index (0,0) = bottom-left = world (-50m, -50m)
Grid index (199,199) = top-right = world (+50m, +50m)
Cell (i, j) covers: x in [x_range[0] + i*cell_size, x_range[0] + (i+1)*cell_size]
                     y in [y_range[0] + j*cell_size, y_range[0] + (j+1)*cell_size]
```

---

## 4. Quality Control Procedures

### 4.1 Automated Quality Checks

| Check ID | Description | Criterion | Severity |
|----------|-------------|-----------|----------|
| QC-001 | Occupied cell ratio | 2-20% of valid cells are occupied | Warning if outside |
| QC-002 | Unknown cell ratio | 20-60% of total cells are unknown | Warning if outside |
| QC-003 | LiDAR point density | At least 10,000 points in grid range | Error if below |
| QC-004 | Free space connectivity | Free cells form connected region from ego | Error if not |
| QC-005 | Ego cell clear | Cells around ego (3x3) must be free | Error if not |
| QC-006 | Temporal consistency | IoU between adjacent keyframe GTs > 0.7 | Warning if below |
| QC-007 | Semantic-occupancy agreement | All semantic-labeled cells are occupied | Error if not |
| QC-008 | Height filter validity | No ground points marking road as occupied | Review if > 5% |

### 4.2 Validation Against 3D Annotations

Cross-reference with nuScenes 3D bounding box annotations:

```python
def validate_gt_against_boxes(occupancy_gt, gt_boxes_bev, grid_config):
    """
    Validate occupancy GT consistency with annotated 3D boxes.
    
    Checks:
    1. All annotated object footprints should overlap with occupied cells
    2. No occupied cells inside annotated "free space" corridors
    """
    issues = []
    
    for box in gt_boxes_bev:
        # Rasterize box footprint to grid
        box_cells = rasterize_box_to_grid(box, grid_config)
        
        # Check overlap: at least 50% of box cells should be occupied
        occupied_in_box = sum(1 for c in box_cells if occupancy_gt[c] == 1)
        overlap_ratio = occupied_in_box / max(len(box_cells), 1)
        
        if overlap_ratio < 0.3:
            issues.append({
                "type": "low_box_coverage",
                "box_id": box["token"],
                "overlap": overlap_ratio,
                "severity": "warning"
            })
    
    return issues
```

### 4.3 Manual Review Triggers

The following conditions trigger manual inspection:

1. **Anomalous occupancy patterns**: Straight lines of occupied cells (calibration error)
2. **Large unknown regions within free space**: Indicates LiDAR occlusion or failure
3. **Temporal flickering**: A cell alternates between occupied and free across frames
4. **Edge artifacts**: Grid boundary cells showing unexplained occupancy patterns

### 4.4 Known Limitations of LiDAR-Derived Ground Truth

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| LiDAR occlusion | Areas behind large vehicles have no GT | Mark as unknown (2), ignore in loss |
| Thin structures | Fences/poles may be missed | Multi-sweep aggregation helps |
| Moving objects | Objects move between sweeps, smearing GT | Use single sweep for dynamic objects |
| Ground reflections | Wet roads reflect LiDAR | Height filtering (z > -0.3m for occupancy) |
| Glass/transparent | LiDAR passes through glass | Accept as limitation; radar may see these |
| Maximum range | LiDAR range ~70-100m; beyond is unknown | Cells beyond range stay unknown |

---

## 5. Tools and Workflow

### 5.1 Data Preparation Tools

**nuScenes devkit** (primary tool for data access):

```bash
pip install nuscenes-devkit

# Verify installation
python -c "from nuscenes.nuscenes import NuScenes; print('OK')"
```

**Ground truth generation script:**

```bash
# Generate occupancy GT for the full training set
python tools/generate_occupancy_gt.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --dataroot data/nuscenes \
    --output_dir data/nuscenes/occupancy_gt \
    --num_sweeps 10 \
    --split train

# Generate for validation
python tools/generate_occupancy_gt.py \
    --config configs/radar_occupancy_nuscenes.yaml \
    --dataroot data/nuscenes \
    --output_dir data/nuscenes/occupancy_gt \
    --num_sweeps 10 \
    --split val
```

### 5.2 Visualization Tools

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_occupancy_gt(gt_grid, title="Occupancy Ground Truth"):
    """Visualize occupancy ground truth with color coding."""
    # Color map: free=green, occupied=red, unknown=gray
    vis = np.zeros((*gt_grid.shape, 3))
    vis[gt_grid == 0] = [0.2, 0.8, 0.2]   # Free = green
    vis[gt_grid == 1] = [0.9, 0.1, 0.1]   # Occupied = red
    vis[gt_grid == 2] = [0.5, 0.5, 0.5]   # Unknown = gray
    
    plt.figure(figsize=(8, 8))
    plt.imshow(vis, origin='lower')
    plt.title(title)
    plt.xlabel("X cells")
    plt.ylabel("Y cells")
    
    # Mark ego position
    cx, cy = gt_grid.shape[0] // 2, gt_grid.shape[1] // 2
    plt.plot(cx, cy, 'b*', markersize=15, label='Ego')
    plt.legend()
    plt.tight_layout()
    plt.savefig("occupancy_gt_visualization.png", dpi=150)
    plt.close()
```

### 5.3 Complete Annotation Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Occupancy GT Generation Pipeline                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Load nuScenes scene                                             │
│     └─ Access LiDAR sweeps, ego poses, calibrations                 │
│                                                                     │
│  2. For each keyframe sample:                                       │
│     ├─ Collect N LiDAR sweeps (current + past)                      │
│     ├─ Ego-motion compensate all sweeps to current frame            │
│     ├─ Aggregate point cloud (filter by height and range)           │
│     └─ Generate binary occupancy via voxelization + ray-casting     │
│                                                                     │
│  3. (Optional) Generate semantic labels:                            │
│     ├─ Rasterize annotated 3D boxes into BEV grid                   │
│     ├─ Assign semantic class to occupied cells within boxes          │
│     └─ Label remaining occupied cells as "Other"                    │
│                                                                     │
│  4. Quality control:                                                │
│     ├─ Run automated QC checks (QC-001 through QC-008)              │
│     ├─ Flag anomalous samples for manual review                     │
│     └─ Validate against 3D box annotations                          │
│                                                                     │
│  5. Export:                                                         │
│     ├─ Save as .npz files (one per keyframe sample)                  │
│     ├─ Format: {"occupancy": (H,W), "semantic": (H,W), "meta": {}} │
│     └─ Generate split files (train.txt, val.txt)                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Data Format on Disk

```
data/nuscenes/occupancy_gt/
├── train/
│   ├── sample_000001.npz   # {"occupancy": (200,200), "semantic": (200,200)}
│   ├── sample_000002.npz
│   └── ...
├── val/
│   ├── sample_000001.npz
│   └── ...
├── train.txt               # List of sample tokens
└── val.txt
```

Each `.npz` file contains:
- `occupancy`: int32 array (H, W) with values {0, 1, 2}
- `semantic`: int32 array (H, W) with class indices {0, 1, 2, 3, 4} (2 = ignore)
- `sample_token`: string identifier linking to nuScenes database
- `num_lidar_points`: int, total points used for GT generation

---

## 6. Guidelines for Custom Datasets

If creating occupancy ground truth for non-nuScenes data:

1. **Minimum LiDAR density**: Ensure at least 16-beam LiDAR for adequate free-space ray coverage
2. **Ego pose accuracy**: Sub-centimeter pose accuracy is required for multi-sweep aggregation
3. **Height filtering**: Adjust z-range based on sensor mounting height and terrain
4. **Dynamic object handling**: Use single-sweep GT for cells near annotated moving objects, multi-sweep for static regions
5. **Validation**: Always cross-validate GT against camera imagery to catch systematic errors
6. **Consistency**: Use identical grid parameters (cell size, range) between GT generation and model training
