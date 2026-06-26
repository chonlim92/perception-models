# MapTR: Annotation Guide

## Overview

This guide describes the vectorized map annotation format used by MapTR, including point ordering conventions, coordinate systems, interpolation strategies, and category-specific geometric properties. The annotations define ground truth map elements as ordered point sequences that the model learns to predict.

---

## Coordinate System

### Ego-Vehicle Centered Frame

All map annotations are expressed in the ego-vehicle coordinate system:

```
        +X (forward / longitudinal)
         ^
         |
         |
  +Y <---O (ego vehicle origin)
         |
         |
         v
        -Y (right of vehicle)
```

| Axis | Direction | Range (standard) |
|------|-----------|-----------------|
| X | Forward (longitudinal) | [-30m, +30m] |
| Y | Left (lateral) | [-15m, +15m] |
| Z | Up (not used in BEV) | N/A |

**Convention**: The origin is at the center of the rear axle of the ego vehicle, projected onto the ground plane. The X-axis points forward along the vehicle heading, Y-axis points to the left.

### BEV Plane Projection

Map elements are represented in the 2D Bird's Eye View (BEV) plane:
- All 3D map coordinates are projected onto Z=0 (ground plane)
- Each point is described by (x, y) coordinates in meters
- The perception range defines valid annotation bounds

### Normalized Coordinates

For model training, raw meter coordinates are normalized to [0, 1]:

```
x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)
```

For standard range [-30, 30] x [-15, 15]:
```
x_norm = (x + 30) / 60
y_norm = (y + 15) / 30
```

---

## Vectorized Map Format

### Polyline Representation

A polyline map element (lane divider, road boundary) is an **open** curve represented as an ordered sequence of 2D points:

```
polyline = [(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)]
```

Properties:
- Start point ≠ End point (open curve)
- Points are ordered sequentially along the curve
- Adjacent points are connected by straight line segments
- The full curve is the piecewise-linear interpolation of these points

### Polygon Representation

A polygon map element (pedestrian crossing) is a **closed** curve represented as an ordered sequence of 2D points:

```
polygon = [(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)]
```

Properties:
- Conceptually closed: an implicit edge connects (x_N, y_N) back to (x_1, y_1)
- Points are ordered sequentially along the polygon boundary
- Can be traversed clockwise or counterclockwise
- Any point can serve as the starting point

### Element Data Structure

```python
annotation = {
    "instance_id": int,          # Unique identifier for this element instance
    "category": str,             # "ped_crossing" | "divider" | "boundary"
    "type": str,                 # "polygon" | "polyline"
    "points": np.ndarray,        # Shape: (N_pts, 2), dtype: float32
    "num_raw_points": int,       # Original number of vertices before resampling
    "attributes": dict           # Optional category-specific attributes
}
```

---

## Point Ordering Conventions

### Polyline Ordering (Lane Dividers, Road Boundaries)

For polylines, a consistent traversal direction is defined per category:

**Lane Dividers**:
- Primary direction: Following traffic flow direction (same as adjacent lane direction)
- If bidirectional: Left-to-right relative to ego vehicle forward direction
- Fallback: Bottom-to-top in BEV (increasing X coordinate)

**Road Boundaries**:
- Primary direction: Following the natural road direction (parallel to adjacent lanes)
- Convention: Ordered so that the road surface is on the left side of the traversal direction

### Polygon Ordering (Pedestrian Crossings)

For polygons, the ordering follows:
- **Traversal direction**: Counter-clockwise (CCW) when viewed from above (standard mathematical convention)
- **Starting point**: Topmost-leftmost vertex (minimum X, then minimum Y as tiebreaker)
- Note: Due to permutation-equivalent modeling, the specific starting point and direction matter less during training

### Permutation Equivalence in Training

Despite defining conventions above, MapTR's key innovation is that it does **not** require the model to predict a specific ordering. During training:

- **Polylines**: Both forward and reverse orderings are considered equivalent. The loss is computed as min(loss_forward, loss_reverse).
- **Polygons**: All 2N orderings (N starting points x 2 directions) are considered equivalent. The loss is min over all 2N permutations.

This means annotation ordering is a convention for data consistency, but the model is free to predict any equivalent ordering.

---

## Fixed Point Count: Interpolation and Sampling

### Why Fixed Point Count?

MapTR requires all elements to have exactly **N_pts** points (default: 20) regardless of their original vertex count. This enables:
- Batched tensor operations during training
- Consistent query structure in the transformer decoder
- Direct point-to-point correspondence in loss computation

### Resampling Algorithm

Raw map annotations have variable numbers of vertices. The resampling procedure:

```python
def resample_polyline(points, N_pts=20):
    """
    Resample a polyline/polygon to exactly N_pts points
    via uniform arc-length interpolation.
    """
    # Step 1: Compute cumulative arc length
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    # Step 2: Define uniform sample positions along arc length
    if element_type == "polyline":
        # Endpoints included: sample at 0, L/(N-1), 2L/(N-1), ..., L
        sample_distances = np.linspace(0, total_length, N_pts)
    elif element_type == "polygon":
        # Exclude endpoint (implicit closure): sample at 0, L/N, 2L/N, ..., (N-1)L/N
        sample_distances = np.linspace(0, total_length, N_pts, endpoint=False)
    
    # Step 3: Linear interpolation at each sample distance
    resampled = np.zeros((N_pts, 2))
    for i, d in enumerate(sample_distances):
        # Find segment containing distance d
        idx = np.searchsorted(cumulative_length, d, side='right') - 1
        idx = np.clip(idx, 0, len(points) - 2)
        
        # Interpolation parameter within segment
        seg_start = cumulative_length[idx]
        seg_end = cumulative_length[idx + 1]
        t = (d - seg_start) / (seg_end - seg_start + 1e-6)
        
        # Linear interpolation
        resampled[i] = points[idx] * (1 - t) + points[idx + 1] * t
    
    return resampled
```

### Key Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| N_pts (default) | 20 | Fixed for all categories |
| Interpolation method | Linear (piecewise) | Between consecutive original vertices |
| Sampling strategy | Uniform arc-length | Equal spacing along curve length |
| Endpoint handling (polyline) | Inclusive | First and last sample = first and last point |
| Endpoint handling (polygon) | Exclusive | Last sample ≠ first sample (implicit closure) |

### Effect of N_pts Choice

| N_pts | Geometric Fidelity | Computation Cost | Typical Use |
|-------|-------------------|-----------------|-------------|
| 10 | Lower (may miss fine curves) | Lower | Fast inference |
| 20 | Standard (good balance) | Medium | Default setting |
| 50 | Higher (captures tight curves) | Higher | High-fidelity experiments |

---

## Annotation Categories and Geometric Properties

### Pedestrian Crossing

| Property | Description |
|----------|-------------|
| Geometry type | Closed polygon |
| Typical shape | Rectangle (4 corners) |
| Raw vertices | Usually 4 (rectangular) or 6-8 (irregular) |
| Resampled to | 20 points along polygon perimeter |
| Typical width | 2-6 meters |
| Typical length | 3-15 meters (spans lane width) |
| Orientation | Perpendicular to traffic flow |
| Permutation group size | 2N = 40 equivalent orderings |

**Annotation rules**:
- The polygon boundary traces the outer edge of the crossing marking
- All four sides of a rectangular crossing are included
- Irregular crossings (non-rectangular) follow the actual painted boundary
- Crossings partially outside perception range are clipped to the boundary

### Lane Divider

| Property | Description |
|----------|-------------|
| Geometry type | Open polyline |
| Typical shape | Smooth curve following lane markings |
| Raw vertices | 10-50 (depending on curvature) |
| Resampled to | 20 points along polyline |
| Typical length | 10-60 meters (clipped to perception range) |
| Curvature | Low (straight roads) to high (intersections) |
| Permutation group size | 2 equivalent orderings |

**Annotation rules**:
- Traces the center of the painted lane marking
- Each continuous marking is a separate instance (broken lines are grouped into one divider)
- Intersections may have dividers that split or merge (each segment is a separate instance)
- Double lines: annotated as a single divider at the centerline between the two painted lines

### Road Boundary

| Property | Description |
|----------|-------------|
| Geometry type | Open polyline |
| Typical shape | Follows road edge (curb, barrier, or road edge) |
| Raw vertices | 15-80 (often more complex than dividers) |
| Resampled to | 20 points along polyline |
| Typical length | 15-60 meters (clipped to perception range) |
| Curvature | Variable (follows road geometry and obstacles) |
| Permutation group size | 2 equivalent orderings |

**Annotation rules**:
- Traces the boundary between drivable and non-drivable surface
- Includes curbs, barriers, fences, vegetation edges
- Each continuous boundary segment is a separate instance
- T-intersections and junctions create boundary endpoints

---

## Clipping to Perception Range

When map elements extend beyond the perception range, they are clipped:

### Polyline Clipping

```python
def clip_polyline_to_range(points, x_range, y_range):
    """
    Clip a polyline to the rectangular perception range.
    Uses Liang-Barsky or Cohen-Sutherland line clipping per segment.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    clipped_segments = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        # Clip segment [p1, p2] to rectangle
        clipped = clip_line_segment(p1, p2, x_min, x_max, y_min, y_max)
        if clipped is not None:
            clipped_segments.append(clipped)
    
    # Merge consecutive clipped segments into continuous polylines
    return merge_segments(clipped_segments)
```

### Polygon Clipping

- Uses Sutherland-Hodgman algorithm to clip polygon against rectangular boundary
- Result may have more vertices than the original (clipping adds intersection points)
- Final result is resampled to N_pts points

### Edge Cases

| Scenario | Handling |
|----------|----------|
| Element fully inside range | Keep as-is |
| Element fully outside range | Discard |
| Element partially inside | Clip and keep visible portion |
| Very short clipped segment (< 2m) | Discard (too small to be meaningful) |
| Polygon becomes degenerate after clipping | Discard |

---

## Quality Assurance Checks

### Geometric Validity

1. **Minimum length**: Polylines must be at least 2 meters long after clipping
2. **Minimum area**: Polygons must have area > 1 m² after clipping
3. **No self-intersection**: Polylines should not cross themselves
4. **Point spacing**: After resampling, adjacent points should be roughly equally spaced
5. **No duplicate points**: Consecutive points must not be identical (causes zero-length segments)

### Annotation Consistency

1. **Category correctness**: Element assigned to correct semantic category
2. **Instance separation**: Distinct physical elements are separate instances
3. **Completeness**: All visible map elements within perception range are annotated
4. **Temporal consistency**: Same physical element gets consistent instance ID across frames

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Crossing annotated as divider | Category confusion | Review semantic definition |
| Gap in road boundary | Occluded or missing annotation | Interpolate through gap or split into segments |
| Overlapping instances | Duplicate annotations | Merge or remove duplicate |
| Very short elements | Aggressive clipping | Apply minimum length threshold |
| Irregular point spacing | Non-uniform raw annotation | Arc-length resampling fixes this |

---

## Annotation Format: File Structure

### Per-Scene Annotation File

```json
{
    "scene_id": "scene-0001",
    "frames": [
        {
            "frame_id": "frame-001",
            "timestamp": 1532402927647951,
            "ego_pose": {
                "translation": [x, y, z],
                "rotation": [w, x, y, z]
            },
            "map_elements": [
                {
                    "instance_id": 1,
                    "category": "divider",
                    "type": "polyline",
                    "points": [[x1,y1], [x2,y2], ..., [x20,y20]]
                },
                {
                    "instance_id": 2,
                    "category": "ped_crossing",
                    "type": "polygon", 
                    "points": [[x1,y1], [x2,y2], ..., [x20,y20]]
                }
            ]
        }
    ]
}
```

### Coordinate Value Ranges

After normalization to [0, 1]:
- x = 0.0 corresponds to x_min = -30m (behind ego)
- x = 1.0 corresponds to x_max = +30m (in front of ego)
- y = 0.0 corresponds to y_min = -15m (right side)
- y = 1.0 corresponds to y_max = +15m (left side)

---

## Summary of Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N_pts | 20 | Points per element |
| Perception X | [-30, 30] m | Longitudinal range |
| Perception Y | [-15, 15] m | Lateral range |
| Categories | 3 | ped_crossing, divider, boundary |
| Polyline permutations | 2 | Forward/reverse |
| Polygon permutations | 2N = 40 | All cyclic orderings |
| Min polyline length | 2 m | After clipping |
| Min polygon area | 1 m² | After clipping |
| Coordinate frame | Ego-centric BEV | Z=0 ground plane |
| Normalization | [0, 1] | Relative to perception range |
