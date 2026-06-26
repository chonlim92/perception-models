# StreamMapNet: Annotation Guide

## Overview

StreamMapNet operates on vectorized map annotations representing static road structures as polylines and polygons. This document describes the annotation format, ground truth generation pipeline, and the transformation from global map coordinates to ego-vehicle-centric representations.

---

## Map Element Categories

StreamMapNet uses three primary map element categories for training and evaluation:

| Category | Geometry Type | Description | Examples |
|----------|--------------|-------------|----------|
| **Lane Divider** | Polyline | Lines separating lanes | White/yellow lane markings, dashed/solid lines |
| **Road Boundary** | Polyline | Edges of drivable surface | Curbs, road edges, barriers |
| **Pedestrian Crossing** | Polygon | Areas designated for pedestrian crossing | Zebra crossings, marked crosswalks |

---

## Vectorized Map Annotation Format

### Polyline Representation

Lane dividers and road boundaries are represented as **ordered polylines**:

```
polyline = [(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)]
```

- Each polyline is a sequence of 2D points in a coordinate frame
- Points are ordered along the direction of the road element
- The number of raw annotation points varies per element (typically 5-200+)
- For training, polylines are resampled to a fixed number of points (see Point Sampling below)

### Polygon Representation

Pedestrian crossings are represented as **closed polygons**:

```
polygon = [(x_1, y_1), (x_2, y_2), ..., (x_M, y_M), (x_1, y_1)]
```

- The first and last points are identical (closed contour)
- Points define the boundary of the crossing area
- Typically quadrilateral (4 corner points) but can be more complex
- For training, the polygon boundary is treated as a closed polyline and resampled

---

## nuScenes Map Expansion Format

### NuScenesMap API

The nuScenes map expansion provides vectorized annotations accessible through the `NuScenesMap` API:

```python
from nuscenes.map_expansion.map_api import NuScenesMap

nusc_map = NuScenesMap(dataroot='/data/nuscenes', map_name='singapore-onenorth')
```

### Map Layers

| Layer Name | Type | Used in StreamMapNet |
|-----------|------|---------------------|
| `lane_divider` | Line | Yes (Lane Divider class) |
| `road_divider` | Line | Yes (Lane Divider class) |
| `road_segment` | Polygon | No (too coarse) |
| `lane` | Polygon | No (use dividers instead) |
| `ped_crossing` | Polygon | Yes (Pedestrian Crossing class) |
| `walkway` | Polygon | No |
| `stop_line` | Line | No (optional) |
| `carpark_area` | Polygon | No |
| `road_block` | Polygon | Yes (Road Boundary class - boundary extraction) |

### Accessing Raw Annotations

```python
# Get all lane dividers
lane_divider_tokens = nusc_map.lane_divider

for token in lane_divider_tokens:
    record = nusc_map.get('lane_divider', token)
    # record contains:
    # - 'token': unique identifier
    # - 'line_token': reference to underlying line geometry
    # - 'lane_divider_segments': list of segment info
    
    # Get the actual geometry
    line = nusc_map.get('line', record['line_token'])
    nodes = [nusc_map.get('node', node_token) for node_token in line['node_tokens']]
    coords = [(node['x'], node['y']) for node in nodes]
    # coords is in global map coordinates (meters)

# Get pedestrian crossings
ped_crossing_tokens = nusc_map.ped_crossing

for token in ped_crossing_tokens:
    record = nusc_map.get('ped_crossing', token)
    polygon_token = record['polygon_token']
    polygon = nusc_map.get('polygon', polygon_token)
    exterior_nodes = [nusc_map.get('node', nt) for nt in polygon['exterior_node_tokens']]
    coords = [(node['x'], node['y']) for node in exterior_nodes]
    # coords defines the polygon boundary in global coordinates
```

### Road Boundary Extraction

Road boundaries are not directly annotated in nuScenes. They are derived from `road_segment` or `road_block` polygon boundaries:

```python
from shapely.geometry import MultiPolygon, box
from shapely.ops import unary_union

# Get all road block polygons
road_polygons = []
for token in nusc_map.road_block:
    record = nusc_map.get('road_block', token)
    polygon = nusc_map.get('polygon', record['polygon_token'])
    nodes = [nusc_map.get('node', nt) for nt in polygon['exterior_node_tokens']]
    coords = [(n['x'], n['y']) for n in nodes]
    road_polygons.append(Polygon(coords))

# Union all road polygons
road_union = unary_union(road_polygons)

# Extract boundary as polylines
if road_union.geom_type == 'MultiPolygon':
    boundaries = []
    for poly in road_union.geoms:
        boundaries.append(list(poly.exterior.coords))
else:
    boundaries = [list(road_union.exterior.coords)]
```

---

## Coordinate Frame Transformations

### Global Map Coordinates to Ego-Vehicle Frame

Map annotations in nuScenes are stored in a global coordinate system (per city map). For training, they must be transformed to the ego-vehicle coordinate frame at each timestamp.

#### Step 1: Get Ego-Pose

```python
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np

nusc = NuScenes(version='v1.0-trainval', dataroot='/data/nuscenes')

# Get sample and its ego-pose
sample = nusc.sample[0]
sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])

# Ego pose: global -> ego transformation
ego_translation = np.array(ego_pose['translation'])  # [x, y, z] in global
ego_rotation = Quaternion(ego_pose['rotation'])       # quaternion
```

#### Step 2: Transform Map Points to Ego Frame

```python
def global_to_ego(points_global, ego_translation, ego_rotation):
    """
    Transform points from global map coordinates to ego-vehicle frame.
    
    Args:
        points_global: (N, 2) or (N, 3) array in global coordinates
        ego_translation: (3,) ego position in global frame
        ego_rotation: Quaternion representing ego orientation
    
    Returns:
        points_ego: (N, 2) points in ego-vehicle frame (x-forward, y-left)
    """
    # Add z=0 if 2D points
    if points_global.shape[1] == 2:
        points_global = np.column_stack([points_global, np.zeros(len(points_global))])
    
    # Translate: shift origin to ego position
    points_centered = points_global - ego_translation
    
    # Rotate: align with ego heading
    rotation_matrix = ego_rotation.inverse.rotation_matrix
    points_ego = (rotation_matrix @ points_centered.T).T
    
    return points_ego[:, :2]  # Return only x, y
```

#### Step 3: Clip to Perception Range

```python
# StreamMapNet perception range
PERCEPTION_RANGE = {
    'x_min': -30.0,  # meters behind ego
    'x_max': 30.0,   # meters ahead of ego
    'y_min': -15.0,  # meters to the right
    'y_max': 15.0,   # meters to the left
}

def clip_polyline_to_range(polyline_ego, perception_range):
    """
    Clip a polyline to the perception range.
    Segments crossing the boundary are interpolated.
    """
    from shapely.geometry import LineString, box
    
    line = LineString(polyline_ego)
    roi = box(
        perception_range['x_min'], perception_range['y_min'],
        perception_range['x_max'], perception_range['y_max']
    )
    
    clipped = line.intersection(roi)
    
    if clipped.is_empty:
        return []
    elif clipped.geom_type == 'LineString':
        return [np.array(clipped.coords)]
    elif clipped.geom_type == 'MultiLineString':
        return [np.array(seg.coords) for seg in clipped.geoms]
    return []
```

---

## Point Sampling Along Polylines

### Fixed-Point Resampling

StreamMapNet represents each map element as a fixed number of points (K), regardless of the original annotation density. This is critical for:
- Uniform representation across elements of different lengths
- Compatibility with the permutation-invariant matching loss
- Fixed output dimension of the prediction head

#### Sampling Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K (num_points) | 20 | Number of sampled points per element |
| Sampling method | Uniform arc-length | Equal spacing along the polyline |

#### Implementation

```python
import numpy as np
from scipy.interpolate import interp1d

def resample_polyline(polyline, num_points=20):
    """
    Resample a polyline to a fixed number of equally-spaced points.
    
    Args:
        polyline: (M, 2) array of original polyline vertices
        num_points: target number of points (K=20 in StreamMapNet)
    
    Returns:
        resampled: (num_points, 2) array of uniformly sampled points
    """
    # Compute cumulative arc length
    diffs = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    if total_length < 1e-6:
        # Degenerate polyline (all points coincide)
        return np.tile(polyline[0], (num_points, 1))
    
    # Normalize to [0, 1]
    cumulative_normalized = cumulative_length / total_length
    
    # Create interpolation function
    fx = interp1d(cumulative_normalized, polyline[:, 0], kind='linear')
    fy = interp1d(cumulative_normalized, polyline[:, 1], kind='linear')
    
    # Sample at uniform intervals
    sample_points = np.linspace(0, 1, num_points)
    resampled_x = fx(sample_points)
    resampled_y = fy(sample_points)
    
    return np.stack([resampled_x, resampled_y], axis=1)
```

### Polygon Resampling

For pedestrian crossings (polygons), the boundary is treated as a closed polyline:

```python
def resample_polygon(polygon, num_points=20):
    """
    Resample a polygon boundary to fixed points.
    The polygon is treated as a closed polyline.
    
    Args:
        polygon: (M, 2) array of polygon vertices (first == last for closed)
        num_points: target number of points
    
    Returns:
        resampled: (num_points, 2) array
    """
    # Ensure polygon is closed
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0:1]])
    
    # Resample as a polyline (the last point will be near the first)
    return resample_polyline(polygon, num_points)
```

---

## Ground Truth Generation Pipeline

### Complete Pipeline Overview

```
Raw nuScenes Map (global coords)
    │
    ▼
[1] Load map elements from NuScenesMap API
    │
    ▼
[2] For each sample timestamp:
    │   - Get ego-pose (translation + rotation)
    │   - Transform all map elements to ego frame
    │
    ▼
[3] Clip elements to perception range
    │   - Discard elements fully outside range
    │   - Clip elements crossing boundary
    │
    ▼
[4] Filter by minimum length
    │   - Discard very short segments (< 2m)
    │
    ▼
[5] Resample to fixed K points
    │   - Uniform arc-length sampling
    │
    ▼
[6] Normalize coordinates to [-1, 1]
    │   - x_norm = (x - x_center) / x_range
    │   - y_norm = (y - y_center) / y_range
    │
    ▼
[7] Assign class labels
    │   - 0: lane_divider
    │   - 1: road_boundary
    │   - 2: pedestrian_crossing
    │
    ▼
Ground Truth: {
    'vectors': (N_elements, K, 2),  # normalized coordinates
    'labels': (N_elements,),         # class indices
}
```

### Implementation

```python
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

class StreamMapNetGTGenerator:
    """Generate ground truth annotations for StreamMapNet."""
    
    CLASSES = ['lane_divider', 'road_boundary', 'ped_crossing']
    NUM_POINTS = 20
    PERCEPTION_RANGE = [-30.0, -15.0, 30.0, 15.0]  # x_min, y_min, x_max, y_max
    MIN_LENGTH = 2.0  # meters
    
    def __init__(self, nusc, nusc_map):
        self.nusc = nusc
        self.nusc_map = nusc_map
    
    def generate_gt_for_sample(self, sample_token):
        """Generate GT map elements for a single sample."""
        sample = self.nusc.get('sample', sample_token)
        sample_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        
        # Get ego transformation
        ego_trans = np.array(ego_pose['translation'])
        ego_rot = Quaternion(ego_pose['rotation'])
        
        vectors = []
        labels = []
        
        # Process lane dividers
        for element in self._get_lane_dividers():
            element_ego = self._transform_to_ego(element, ego_trans, ego_rot)
            clipped = self._clip_to_range(element_ego)
            for segment in clipped:
                if self._compute_length(segment) >= self.MIN_LENGTH:
                    resampled = resample_polyline(segment, self.NUM_POINTS)
                    normalized = self._normalize(resampled)
                    vectors.append(normalized)
                    labels.append(0)
        
        # Process road boundaries
        for element in self._get_road_boundaries():
            element_ego = self._transform_to_ego(element, ego_trans, ego_rot)
            clipped = self._clip_to_range(element_ego)
            for segment in clipped:
                if self._compute_length(segment) >= self.MIN_LENGTH:
                    resampled = resample_polyline(segment, self.NUM_POINTS)
                    normalized = self._normalize(resampled)
                    vectors.append(normalized)
                    labels.append(1)
        
        # Process pedestrian crossings
        for element in self._get_ped_crossings():
            element_ego = self._transform_to_ego(element, ego_trans, ego_rot)
            if self._is_in_range(element_ego):
                resampled = resample_polygon(element_ego, self.NUM_POINTS)
                normalized = self._normalize(resampled)
                vectors.append(normalized)
                labels.append(2)
        
        return {
            'vectors': np.array(vectors),   # (N, 20, 2)
            'labels': np.array(labels),     # (N,)
        }
    
    def _normalize(self, points):
        """Normalize points to [-1, 1] based on perception range."""
        x_min, y_min, x_max, y_max = self.PERCEPTION_RANGE
        points_norm = points.copy()
        points_norm[:, 0] = (points[:, 0] - (x_min + x_max) / 2) / ((x_max - x_min) / 2)
        points_norm[:, 1] = (points[:, 1] - (y_min + y_max) / 2) / ((y_max - y_min) / 2)
        return points_norm
```

---

## Annotation Statistics

### nuScenes Dataset Statistics

| Metric | Value |
|--------|-------|
| Average elements per frame | ~35-50 |
| Average lane dividers per frame | ~15-25 |
| Average road boundaries per frame | ~10-15 |
| Average ped crossings per frame | ~2-5 |
| Average polyline length | ~15-40 m |
| Max elements per frame | ~120 |
| Min elements per frame | ~5 |

### Element Length Distribution

| Element Type | Mean Length | Std Length | Min | Max |
|-------------|-----------|-----------|-----|-----|
| Lane Divider | 22.3 m | 12.1 m | 2.0 m | 60.0 m |
| Road Boundary | 28.7 m | 15.4 m | 2.0 m | 60.0 m |
| Ped Crossing (perimeter) | 18.5 m | 6.2 m | 8.0 m | 40.0 m |

---

## Direction Handling

### Polyline Direction Convention

StreamMapNet uses a direction-aware loss that requires consistent polyline direction:

- **Lane dividers:** Direction follows traffic flow (start → end in driving direction)
- **Road boundaries:** Direction follows the right-hand rule (road surface on the left of the polyline direction)
- **Pedestrian crossings:** Counter-clockwise ordering of boundary points

### Direction Augmentation

During training, polyline direction is augmented:
- With 50% probability, the direction is reversed
- The loss function accounts for both directions (bidirectional matching)

```python
def compute_directed_loss(pred_points, gt_points):
    """
    Compute direction-aware loss considering both directions.
    """
    # Forward direction loss
    loss_forward = F.l1_loss(pred_points, gt_points, reduction='none').sum(-1).mean(-1)
    
    # Reverse direction loss
    gt_reversed = gt_points.flip(dims=[1])  # Reverse point order
    loss_reverse = F.l1_loss(pred_points, gt_reversed, reduction='none').sum(-1).mean(-1)
    
    # Take minimum of both directions
    loss = torch.min(loss_forward, loss_reverse)
    return loss
```

---

## Quality Considerations

### Common Annotation Artifacts

1. **Self-intersecting polylines:** Some raw annotations contain self-intersections, especially at road curves. These are handled by the clipping step.

2. **Duplicate elements:** Overlapping lane dividers at intersections. Deduplication is applied during GT generation.

3. **Missing elements:** Some map areas have incomplete annotations. The model learns to handle variable-density GT.

4. **Coordinate precision:** nuScenes map coordinates have centimeter-level precision, sufficient for training.

### Data Augmentation on Annotations

During training, augmentations are applied consistently to both images and GT annotations:
- **Random rotation** ([-22.5, 22.5] degrees): Rotate both BEV features and GT points
- **Random scaling** ([0.95, 1.05]): Scale perception range and GT coordinates
- **Random translation** ([-5m, 5m]): Shift both BEV center and GT coordinates
