"""
Post-processing for HDMapNet predictions.

Converts dense BEV predictions (semantic masks, instance embeddings, directions)
into vectorized polyline representations:
1. Threshold semantic mask and skeletonize
2. Sample points along skeleton to form polylines
3. Group by instance embedding (DBSCAN clustering)
4. Assign direction to each polyline
5. Apply NMS to remove duplicates
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import label as scipy_label
from sklearn.cluster import DBSCAN
from collections import defaultdict


def morphological_thin(binary_mask, max_iterations=50):
    """Apply morphological thinning to produce a skeleton.

    Uses iterative hit-or-miss erosion with structuring elements
    to produce a 1-pixel wide skeleton of the input binary mask.

    Args:
        binary_mask: 2D numpy array (H, W) of type bool or uint8.
        max_iterations: Maximum number of thinning iterations.

    Returns:
        Skeleton mask as boolean numpy array (H, W).
    """
    mask = binary_mask.astype(np.uint8).copy()

    # 8 structuring element pairs for thinning (Guo-Hall or Zhang-Suen style)
    # Each pair: (foreground pattern, background pattern)
    struct_elements = [
        # Element 0: top edge
        (np.array([[0, 0, 0], [2, 1, 2], [1, 1, 1]], dtype=np.int8),),
        # Element 1: top-right corner
        (np.array([[2, 0, 0], [1, 1, 0], [2, 1, 2]], dtype=np.int8),),
        # Element 2: right edge
        (np.array([[1, 2, 0], [1, 1, 0], [1, 2, 0]], dtype=np.int8),),
        # Element 3: bottom-right corner
        (np.array([[2, 1, 2], [1, 1, 0], [2, 0, 0]], dtype=np.int8),),
        # Element 4: bottom edge
        (np.array([[1, 1, 1], [2, 1, 2], [0, 0, 0]], dtype=np.int8),),
        # Element 5: bottom-left corner
        (np.array([[2, 1, 2], [0, 1, 1], [0, 0, 2]], dtype=np.int8),),
        # Element 6: left edge
        (np.array([[0, 2, 1], [0, 1, 1], [0, 2, 1]], dtype=np.int8),),
        # Element 7: top-left corner
        (np.array([[0, 0, 2], [0, 1, 1], [2, 1, 2]], dtype=np.int8),),
    ]

    for _ in range(max_iterations):
        changed = False
        for se_tuple in struct_elements:
            se = se_tuple[0]
            # Create hit and miss structuring elements
            # 1 = must be foreground, 0 = must be background, 2 = don't care
            hit = (se == 1).astype(np.uint8)
            miss = (se == 0).astype(np.uint8)

            # Hit-or-miss: pixels matching the pattern
            eroded_fg = ndimage.binary_erosion(mask, hit)
            eroded_bg = ndimage.binary_erosion(1 - mask, miss)
            matches = eroded_fg & eroded_bg

            if matches.any():
                mask[matches] = 0
                changed = True

        if not changed:
            break

    return mask.astype(bool)


def skeleton_to_points(skeleton, sample_spacing=2):
    """Sample points along a skeleton to create polyline vertices.

    Uses connected component labeling and then traces each component
    by following neighbors to produce ordered point sequences.

    Args:
        skeleton: Boolean skeleton mask (H, W).
        sample_spacing: Sample every N-th pixel along the skeleton.

    Returns:
        List of polylines, each a numpy array of shape (num_points, 2) in (row, col).
    """
    if not skeleton.any():
        return []

    # Label connected components
    labeled, num_features = scipy_label(skeleton)
    polylines = []

    for comp_id in range(1, num_features + 1):
        component = (labeled == comp_id)
        points = np.argwhere(component)  # (num_pixels, 2) as (row, col)

        if len(points) < 3:
            continue

        # Order points by tracing connectivity
        ordered = _trace_component(component, points)

        # Sample at regular intervals
        if len(ordered) > sample_spacing:
            indices = list(range(0, len(ordered), sample_spacing))
            if indices[-1] != len(ordered) - 1:
                indices.append(len(ordered) - 1)
            sampled = ordered[indices]
        else:
            sampled = ordered

        if len(sampled) >= 2:
            polylines.append(sampled)

    return polylines


def _trace_component(component, points):
    """Trace a skeleton component to produce ordered point sequence.

    Starts from an endpoint (degree-1 pixel) and follows neighbors.

    Args:
        component: Boolean mask for this connected component.
        points: All skeleton pixels for this component (N, 2).

    Returns:
        Ordered numpy array (N, 2) of points along the skeleton.
    """
    H, W = component.shape

    # Find degrees (number of 8-connected skeleton neighbors)
    padded = np.pad(component.astype(np.uint8), 1, mode='constant')
    degrees = np.zeros_like(component, dtype=np.int32)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = padded[1 + dy: H + 1 + dy, 1 + dx: W + 1 + dx]
            degrees += shifted
    degrees = degrees * component

    # Find endpoints (degree 1) as start points
    endpoints = np.argwhere((degrees == 1) & component)
    if len(endpoints) > 0:
        start = tuple(endpoints[0])
    else:
        # No endpoints (closed loop), start from any point
        start = tuple(points[0])

    # Trace by following neighbors
    visited = set()
    ordered = []
    current = start
    visited.add(current)
    ordered.append(current)

    while True:
        r, c = current
        found_next = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and component[nr, nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    ordered.append((nr, nc))
                    current = (nr, nc)
                    found_next = True
                    break
            if found_next:
                break

        if not found_next:
            break

    return np.array(ordered)


def cluster_instances(embeddings, semantic_mask, eps=1.5, min_samples=5):
    """Group semantic pixels into instances using DBSCAN on embeddings.

    Args:
        embeddings: Instance embedding map (E, H, W) numpy array.
        semantic_mask: Binary semantic mask (H, W) indicating valid pixels.
        eps: DBSCAN epsilon (neighborhood radius in embedding space).
        min_samples: Minimum samples per cluster.

    Returns:
        instance_map: (H, W) integer array with instance IDs (0 = background).
    """
    E, H, W = embeddings.shape
    instance_map = np.zeros((H, W), dtype=np.int32)

    valid_pixels = np.argwhere(semantic_mask > 0)  # (N, 2)
    if len(valid_pixels) < min_samples:
        return instance_map

    # Extract embeddings at valid pixels
    rows, cols = valid_pixels[:, 0], valid_pixels[:, 1]
    pixel_embeddings = embeddings[:, rows, cols].T  # (N, E)

    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = clustering.fit_predict(pixel_embeddings)

    # Assign cluster labels to instance map (offset by 1, -1 is noise -> 0)
    for i, (r, c) in enumerate(valid_pixels):
        if labels[i] >= 0:
            instance_map[r, c] = labels[i] + 1

    return instance_map


def assign_directions(polyline, direction_map):
    """Assign direction to a polyline based on the direction map.

    Computes the average direction along the polyline from the predicted
    direction map, then orients the polyline consistently.

    Args:
        polyline: Numpy array (N, 2) of points in (row, col) format.
        direction_map: Direction prediction (2, H, W) numpy array.

    Returns:
        Oriented polyline (N, 2) with consistent direction.
    """
    if len(polyline) < 2:
        return polyline

    # Sample direction along polyline points
    rows = polyline[:, 0].astype(int)
    cols = polyline[:, 1].astype(int)

    H, W = direction_map.shape[1], direction_map.shape[2]
    rows = np.clip(rows, 0, H - 1)
    cols = np.clip(cols, 0, W - 1)

    dx = direction_map[0, rows, cols]  # direction x components
    dy = direction_map[1, rows, cols]  # direction y components

    # Average direction along polyline
    avg_dx = np.mean(dx)
    avg_dy = np.mean(dy)

    # Compute polyline tangent direction (start to end)
    tangent = polyline[-1] - polyline[0]
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-6:
        return polyline

    tangent = tangent / tangent_norm

    # Check if predicted direction aligns with polyline direction
    # tangent is in (row, col) = (dy, dx) format
    dot_product = tangent[1] * avg_dx + tangent[0] * avg_dy

    # If directions are opposed, reverse the polyline
    if dot_product < 0:
        polyline = polyline[::-1].copy()

    return polyline


def polyline_nms(polylines, distance_threshold=5.0):
    """Non-maximum suppression for polylines.

    Removes duplicate polylines that are too close to each other,
    keeping the longer one in each overlapping pair.

    Args:
        polylines: List of polyline arrays, each (N_i, 2).
        distance_threshold: Maximum Chamfer distance to consider as duplicate.

    Returns:
        Filtered list of polylines.
    """
    if len(polylines) <= 1:
        return polylines

    # Sort by length (longer first)
    polylines_sorted = sorted(polylines, key=lambda p: len(p), reverse=True)

    keep = []
    suppressed = set()

    for i, poly_i in enumerate(polylines_sorted):
        if i in suppressed:
            continue
        keep.append(poly_i)

        for j in range(i + 1, len(polylines_sorted)):
            if j in suppressed:
                continue
            poly_j = polylines_sorted[j]

            # Compute asymmetric Chamfer distance (mean of min distances)
            dist = _chamfer_distance(poly_i, poly_j)
            if dist < distance_threshold:
                suppressed.add(j)

    return keep


def _chamfer_distance(poly_a, poly_b):
    """Compute symmetric Chamfer distance between two polylines.

    Args:
        poly_a: (N, 2) array.
        poly_b: (M, 2) array.

    Returns:
        Symmetric Chamfer distance (scalar).
    """
    # Distances from A to B
    diff_ab = poly_a[:, None, :] - poly_b[None, :, :]  # (N, M, 2)
    dist_ab = np.linalg.norm(diff_ab, axis=-1)  # (N, M)
    min_ab = dist_ab.min(axis=1).mean()  # mean of min distance from A to B

    # Distances from B to A
    min_ba = dist_ab.min(axis=0).mean()  # mean of min distance from B to A

    return (min_ab + min_ba) / 2.0


def vectorize_predictions(
    semantic_pred,
    instance_embedding,
    direction_pred,
    semantic_threshold=0.5,
    dbscan_eps=1.5,
    dbscan_min_samples=5,
    nms_threshold=5.0,
    sample_spacing=2,
    xbound=None,
    ybound=None,
):
    """Full post-processing pipeline to convert dense predictions to polylines.

    Args:
        semantic_pred: Semantic prediction (num_classes, H, W) as probabilities [0,1].
        instance_embedding: Instance embeddings (E, H, W).
        direction_pred: Direction vectors (2, H, W).
        semantic_threshold: Threshold for binary semantic mask.
        dbscan_eps: DBSCAN epsilon for instance clustering.
        dbscan_min_samples: DBSCAN minimum samples.
        nms_threshold: Distance threshold for polyline NMS.
        sample_spacing: Point sampling interval along skeleton.
        xbound: [xmin, xmax, res] for converting pixel to meters. If None, return in pixels.
        ybound: [ymin, ymax, res] for converting pixel to meters.

    Returns:
        Dict mapping class_id to list of polylines (each polyline is Nx2 array).
        If xbound/ybound provided, coordinates are in meters; otherwise in pixels.
    """
    num_classes, H, W = semantic_pred.shape
    results = {}

    for cls_id in range(num_classes):
        # Threshold semantic mask
        cls_mask = (semantic_pred[cls_id] > semantic_threshold).astype(np.uint8)

        if cls_mask.sum() < 10:
            results[cls_id] = []
            continue

        # Skeletonize
        skeleton = morphological_thin(cls_mask)

        # Instance clustering on this class
        cls_instance_map = cluster_instances(
            instance_embedding, cls_mask, eps=dbscan_eps, min_samples=dbscan_min_samples
        )

        # Get unique instance IDs
        instance_ids = np.unique(cls_instance_map)
        instance_ids = instance_ids[instance_ids > 0]

        class_polylines = []

        if len(instance_ids) == 0:
            # No instances found, just use skeleton directly
            polylines = skeleton_to_points(skeleton, sample_spacing=sample_spacing)
            for poly in polylines:
                poly = assign_directions(poly, direction_pred)
                class_polylines.append(poly)
        else:
            # Process each instance separately
            for inst_id in instance_ids:
                inst_mask = (cls_instance_map == inst_id).astype(np.uint8)
                inst_skeleton = skeleton & inst_mask.astype(bool)

                if inst_skeleton.sum() < 3:
                    # If skeleton doesn't overlap instance well, use instance mask
                    inst_skeleton = morphological_thin(inst_mask)

                polylines = skeleton_to_points(inst_skeleton, sample_spacing=sample_spacing)
                for poly in polylines:
                    poly = assign_directions(poly, direction_pred)
                    class_polylines.append(poly)

        # NMS to remove duplicates
        class_polylines = polyline_nms(class_polylines, distance_threshold=nms_threshold)

        # Convert from pixel to metric coordinates if bounds provided
        if xbound is not None and ybound is not None:
            metric_polylines = []
            for poly in class_polylines:
                metric_poly = np.zeros_like(poly, dtype=np.float64)
                # row -> y coordinate
                metric_poly[:, 0] = poly[:, 0] * ybound[2] + ybound[0]
                # col -> x coordinate
                metric_poly[:, 1] = poly[:, 1] * xbound[2] + xbound[0]
                metric_polylines.append(metric_poly)
            class_polylines = metric_polylines

        results[cls_id] = class_polylines

    return results
