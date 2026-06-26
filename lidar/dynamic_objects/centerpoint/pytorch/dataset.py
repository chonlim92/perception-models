"""
CenterPoint Dataset Module.

Implements NuScenes and Waymo dataset loaders for CenterPoint 3D object detection
training, including multi-sweep aggregation, ground truth generation, augmentation
pipelines, and voxelization.
"""

import copy
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
DEFAULT_VOXEL_SIZE = [0.075, 0.075, 0.2]
DEFAULT_MAX_POINTS_PER_VOXEL = 10
DEFAULT_MAX_VOXELS = 120000
DEFAULT_NUM_SWEEPS = 10

NUSCENES_CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

WAYMO_CLASS_NAMES = [
    "Vehicle",
    "Pedestrian",
    "Cyclist",
]


# ---------------------------------------------------------------------------
# Utility: 3D rotation matrix around Z-axis
# ---------------------------------------------------------------------------


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Create a 3x3 rotation matrix for rotation around the Z-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix as numpy array.
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Utility: Voxelization (numpy-based for dataset preprocessing)
# ---------------------------------------------------------------------------


def voxelize_points(
    points: np.ndarray,
    voxel_size: List[float],
    point_cloud_range: List[float],
    max_points_per_voxel: int = 10,
    max_voxels: int = 120000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize a point cloud using numpy.

    Assigns points to voxels defined by voxel_size within the given range.
    Each voxel stores up to max_points_per_voxel points (zero-padded).

    Args:
        points: (N, C) point cloud array with at least (x, y, z).
        voxel_size: [vx, vy, vz] voxel dimensions in meters.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        max_points_per_voxel: Maximum number of points stored per voxel.
        max_voxels: Maximum number of voxels to return.

    Returns:
        voxels: (M, max_points_per_voxel, C) padded point features per voxel.
        coordinates: (M, 3) voxel grid coordinates in (z, y, x) order.
        num_points_per_voxel: (M,) actual point count in each voxel.
    """
    voxel_size = np.array(voxel_size, dtype=np.float32)
    range_min = np.array(point_cloud_range[:3], dtype=np.float32)
    range_max = np.array(point_cloud_range[3:], dtype=np.float32)

    grid_size = np.round((range_max - range_min) / voxel_size).astype(np.int64)

    # Filter points outside the valid range
    mask = (
        (points[:, 0] >= range_min[0]) & (points[:, 0] < range_max[0]) &
        (points[:, 1] >= range_min[1]) & (points[:, 1] < range_max[1]) &
        (points[:, 2] >= range_min[2]) & (points[:, 2] < range_max[2])
    )
    points = points[mask]

    if points.shape[0] == 0:
        num_features = points.shape[1] if len(points.shape) > 1 else 4
        voxels = np.zeros((0, max_points_per_voxel, num_features), dtype=np.float32)
        coordinates = np.zeros((0, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros((0,), dtype=np.int32)
        return voxels, coordinates, num_points_per_voxel

    # Compute voxel coordinates for each point
    coords = np.floor((points[:, :3] - range_min) / voxel_size).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, grid_size[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, grid_size[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, grid_size[2] - 1)

    # Linearize indices for grouping
    linear_idx = (
        coords[:, 2] * (grid_size[1] * grid_size[0]) +
        coords[:, 1] * grid_size[0] +
        coords[:, 0]
    )

    # Sort by linear index for efficient grouping
    sort_order = np.argsort(linear_idx)
    linear_idx_sorted = linear_idx[sort_order]
    points_sorted = points[sort_order]
    coords_sorted = coords[sort_order]

    # Find unique voxels and their boundaries
    unique_indices, first_occurrences, counts = np.unique(
        linear_idx_sorted, return_index=True, return_counts=True
    )

    num_voxels = min(len(unique_indices), max_voxels)
    num_features = points.shape[1]

    voxels = np.zeros((num_voxels, max_points_per_voxel, num_features), dtype=np.float32)
    coordinates = np.zeros((num_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros((num_voxels,), dtype=np.int32)

    for i in range(num_voxels):
        start = first_occurrences[i]
        count = int(counts[i])
        n_points = min(count, max_points_per_voxel)

        voxels[i, :n_points] = points_sorted[start:start + n_points]
        num_points_per_voxel[i] = n_points

        # Store coordinates in (z, y, x) order for sparse convolution
        c = coords_sorted[start]
        coordinates[i] = [c[2], c[1], c[0]]

    return voxels, coordinates, num_points_per_voxel


# ---------------------------------------------------------------------------
# Utility: Gaussian heatmap generation for BEV targets
# ---------------------------------------------------------------------------


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.5) -> float:
    """Compute the Gaussian radius for a detection box in BEV.

    Following CenterNet/CenterPoint, computes the minimum radius such that a
    pair of circles with that radius centered on the ground truth and prediction
    have IoU >= min_overlap.

    Args:
        det_size: (height, width) of the detection box in the BEV grid.
        min_overlap: Minimum required IoU overlap.

    Returns:
        Gaussian radius value.
    """
    height, width = det_size

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4.0 * a1 * c1)
    r1 = (b1 + sq1) / 2.0

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4.0 * a2 * c2)
    r2 = (b2 + sq2) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4.0 * a3 * c3)
    r3 = (b3 + sq3) / 2.0

    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
    """Draw a 2D Gaussian on a heatmap at the given center.

    Args:
        heatmap: (H, W) array to draw on (modified in place).
        center: (x, y) integer pixel coordinates of the Gaussian center.
        radius: Integer radius of the Gaussian kernel.

    Returns:
        The modified heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = _gaussian_2d(diameter, sigma=diameter / 6.0)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]

    left = min(x, radius)
    right = min(width - x, radius + 1)
    top = min(y, radius)
    bottom = min(height - y, radius + 1)

    if left <= 0 or right <= 0 or top <= 0 or bottom <= 0:
        return heatmap

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


def _gaussian_2d(diameter: int, sigma: float) -> np.ndarray:
    """Generate a 2D Gaussian kernel.

    Args:
        diameter: Size of the kernel (diameter x diameter).
        sigma: Standard deviation of the Gaussian.

    Returns:
        (diameter, diameter) normalized Gaussian kernel with peak = 1.
    """
    m = np.arange(0, diameter, dtype=np.float32) - diameter // 2
    x, y = np.meshgrid(m, m)
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0.0
    return h


# ---------------------------------------------------------------------------
# Ground truth target generation
# ---------------------------------------------------------------------------


def generate_targets(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    point_cloud_range: List[float],
    voxel_size: List[float],
    num_classes: int,
    gaussian_overlap: float = 0.1,
    min_radius: int = 2,
) -> Dict[str, np.ndarray]:
    """Generate CenterPoint training targets: heatmaps and regression targets.

    For each ground truth box, a Gaussian is drawn at the BEV center location on
    the class-specific heatmap. Regression targets (offset, size, rotation, velocity)
    are stored at those center locations.

    Args:
        gt_boxes: (N, 9) array with columns [x, y, z, w, l, h, yaw, vx, vy].
        gt_classes: (N,) integer class labels (0-indexed).
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy, vz] voxel dimensions.
        num_classes: Total number of object classes.
        gaussian_overlap: Minimum overlap for Gaussian radius computation.
        min_radius: Minimum Gaussian radius in pixels.

    Returns:
        Dictionary containing:
            - heatmap: (num_classes, H, W) Gaussian heatmaps.
            - reg_targets: (N, 8) regression targets [offset_x, offset_y, z, log(w), log(l), log(h), sin(yaw), cos(yaw)].
            - velocity_targets: (N, 2) velocity targets [vx, vy].
            - target_indices: (N,) linear indices into the BEV grid for each target.
            - target_mask: (N,) binary mask indicating valid targets.
    """
    range_min = np.array(point_cloud_range[:3], dtype=np.float32)
    range_max = np.array(point_cloud_range[3:], dtype=np.float32)
    voxel_sz = np.array(voxel_size, dtype=np.float32)

    # BEV grid dimensions
    grid_size_x = int(np.round((range_max[0] - range_min[0]) / voxel_sz[0]))
    grid_size_y = int(np.round((range_max[1] - range_min[1]) / voxel_sz[1]))

    heatmap = np.zeros((num_classes, grid_size_y, grid_size_x), dtype=np.float32)

    max_objects = gt_boxes.shape[0]
    reg_targets = np.zeros((max_objects, 8), dtype=np.float32)
    velocity_targets = np.zeros((max_objects, 2), dtype=np.float32)
    target_indices = np.zeros((max_objects,), dtype=np.int64)
    target_mask = np.zeros((max_objects,), dtype=np.float32)

    for i in range(max_objects):
        cls_id = int(gt_classes[i])
        if cls_id < 0 or cls_id >= num_classes:
            continue

        x, y, z = gt_boxes[i, 0], gt_boxes[i, 1], gt_boxes[i, 2]
        w, l, h = gt_boxes[i, 3], gt_boxes[i, 4], gt_boxes[i, 5]
        yaw = gt_boxes[i, 6]
        vx, vy = gt_boxes[i, 7], gt_boxes[i, 8]

        # Convert to BEV pixel coordinates
        cx = (x - range_min[0]) / voxel_sz[0]
        cy = (y - range_min[1]) / voxel_sz[1]

        # Integer center (grid cell)
        cx_int = int(cx)
        cy_int = int(cy)

        if cx_int < 0 or cx_int >= grid_size_x or cy_int < 0 or cy_int >= grid_size_y:
            continue

        # Sub-pixel offset
        offset_x = cx - cx_int
        offset_y = cy - cy_int

        # Compute Gaussian radius based on box footprint in BEV
        box_w_pixels = w / voxel_sz[0]
        box_l_pixels = l / voxel_sz[1]
        radius = max(
            int(gaussian_radius((box_l_pixels, box_w_pixels), min_overlap=gaussian_overlap)),
            min_radius,
        )

        # Draw Gaussian on class heatmap
        draw_gaussian(heatmap[cls_id], center=(cx_int, cy_int), radius=radius)

        # Regression targets
        reg_targets[i, 0] = offset_x
        reg_targets[i, 1] = offset_y
        reg_targets[i, 2] = z
        reg_targets[i, 3] = np.log(max(w, 1e-6))
        reg_targets[i, 4] = np.log(max(l, 1e-6))
        reg_targets[i, 5] = np.log(max(h, 1e-6))
        reg_targets[i, 6] = np.sin(yaw)
        reg_targets[i, 7] = np.cos(yaw)

        # Velocity
        velocity_targets[i, 0] = vx
        velocity_targets[i, 1] = vy

        # Linear index in BEV grid
        target_indices[i] = cy_int * grid_size_x + cx_int
        target_mask[i] = 1.0

    return {
        "heatmap": heatmap,
        "reg_targets": reg_targets,
        "velocity_targets": velocity_targets,
        "target_indices": target_indices,
        "target_mask": target_mask,
    }


# ---------------------------------------------------------------------------
# GT Database creation for GT-sampling augmentation
# ---------------------------------------------------------------------------


def create_gt_database(
    dataset_root: str,
    info_path: str,
    class_names: List[str],
    output_path: Optional[str] = None,
    num_sweeps: int = 1,
    point_cloud_range: Optional[List[float]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract ground truth boxes and their interior points to create a GT database.

    For each annotated object in the dataset, extracts the points falling inside
    the 3D bounding box and saves them as individual .bin files. This database is
    used for the GT-sampling augmentation during training.

    Args:
        dataset_root: Root directory of the dataset.
        info_path: Path to the dataset info pickle file (e.g., nuscenes_infos_train.pkl).
        class_names: List of class names to include.
        output_path: Directory to store the GT database files. Defaults to
            dataset_root / "gt_database".
        num_sweeps: Number of sweeps to aggregate for point extraction.
        point_cloud_range: Optional point cloud range to filter points.

    Returns:
        Dictionary mapping class name to list of database entries. Each entry
        contains keys: name, path, box3d, num_points_in_gt, difficulty.
    """
    dataset_root = Path(dataset_root)
    if output_path is None:
        output_path = str(dataset_root / "gt_database")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(info_path, "rb") as f:
        infos = pickle.load(f)

    db: Dict[str, List[Dict[str, Any]]] = {name: [] for name in class_names}

    for idx, info in enumerate(infos):
        # Load the point cloud for the keyframe
        lidar_path = dataset_root / info["lidar_path"]
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)

        # Aggregate additional sweeps if requested
        if num_sweeps > 1 and "sweeps" in info:
            sweep_points_list = [points]
            for sweep_info in info["sweeps"][: num_sweeps - 1]:
                sweep_path = dataset_root / sweep_info["lidar_path"]
                if not sweep_path.exists():
                    continue
                sweep_pts = np.fromfile(str(sweep_path), dtype=np.float32).reshape(-1, 5)

                # Apply ego-motion compensation
                if "transform_matrix" in sweep_info:
                    transform = np.array(sweep_info["transform_matrix"], dtype=np.float32)
                    # Transform sweep points to keyframe coordinate system
                    pts_hom = np.ones((sweep_pts.shape[0], 4), dtype=np.float32)
                    pts_hom[:, :3] = sweep_pts[:, :3]
                    pts_hom = pts_hom @ transform.T
                    sweep_pts[:, :3] = pts_hom[:, :3]

                sweep_points_list.append(sweep_pts)
            points = np.concatenate(sweep_points_list, axis=0)

        # Filter by range if specified
        if point_cloud_range is not None:
            pcr = np.array(point_cloud_range, dtype=np.float32)
            mask = (
                (points[:, 0] >= pcr[0]) & (points[:, 0] < pcr[3]) &
                (points[:, 1] >= pcr[1]) & (points[:, 1] < pcr[4]) &
                (points[:, 2] >= pcr[2]) & (points[:, 2] < pcr[5])
            )
            points = points[mask]

        # Process each GT box in this frame
        gt_boxes = np.array(info["gt_boxes"], dtype=np.float32)  # (N, 9)
        gt_names = info["gt_names"]  # List[str]

        for j in range(len(gt_names)):
            name = gt_names[j]
            if name not in class_names:
                continue

            box = gt_boxes[j]  # [x, y, z, w, l, h, yaw, vx, vy]
            cx, cy, cz = box[0], box[1], box[2]
            bw, bl, bh = box[3], box[4], box[5]
            yaw = box[6]

            # Find points inside the 3D box
            # Transform points to box-local coordinates
            rot = rotation_matrix_z(-yaw)
            translated = points[:, :3] - np.array([cx, cy, cz], dtype=np.float32)
            rotated = translated @ rot.T

            # Check containment in axis-aligned box
            half_w, half_l, half_h = bw / 2.0, bl / 2.0, bh / 2.0
            in_box_mask = (
                (np.abs(rotated[:, 0]) <= half_w) &
                (np.abs(rotated[:, 1]) <= half_l) &
                (np.abs(rotated[:, 2]) <= half_h)
            )

            interior_points = points[in_box_mask].copy()
            num_interior = interior_points.shape[0]

            if num_interior == 0:
                continue

            # Translate points to be relative to box center
            interior_points[:, :3] -= np.array([cx, cy, cz], dtype=np.float32)

            # Save to file
            filename = f"{idx}_{name}_{j}.bin"
            filepath = output_dir / filename
            interior_points.astype(np.float32).tofile(str(filepath))

            entry = {
                "name": name,
                "path": str(filepath.relative_to(dataset_root)),
                "box3d": box.tolist(),
                "num_points_in_gt": num_interior,
                "difficulty": _compute_difficulty(num_interior),
            }
            db[name].append(entry)

    # Save the database info
    db_info_path = output_dir / "gt_database_info.pkl"
    with open(str(db_info_path), "wb") as f:
        pickle.dump(db, f)

    return db


def _compute_difficulty(num_points: int) -> int:
    """Assign difficulty level based on point count.

    Args:
        num_points: Number of lidar points in the ground truth box.

    Returns:
        Difficulty level: 0 (easy, >= 15 points), 1 (moderate, >= 7),
        2 (hard, < 7).
    """
    if num_points >= 15:
        return 0
    elif num_points >= 7:
        return 1
    else:
        return 2


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------


class GTDatabaseSampler:
    """Ground truth sampling augmentation.

    Pastes ground truth objects from a pre-built database into the current scene
    to increase the diversity of training examples, especially for rare classes.

    Args:
        database_path: Path to the GT database info pickle file.
        dataset_root: Root directory for resolving relative bin file paths.
        class_names: List of class names to sample.
        sample_counts: Dictionary mapping class name to number of samples to add.
        min_points: Minimum number of points required in a GT sample.
        point_cloud_range: Valid point cloud range for filtering.
    """

    def __init__(
        self,
        database_path: str,
        dataset_root: str,
        class_names: List[str],
        sample_counts: Optional[Dict[str, int]] = None,
        min_points: int = 5,
        point_cloud_range: Optional[List[float]] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.class_names = class_names
        self.min_points = min_points
        self.point_cloud_range = point_cloud_range

        if sample_counts is None:
            self.sample_counts = {name: 15 for name in class_names}
        else:
            self.sample_counts = sample_counts

        # Load the database
        with open(database_path, "rb") as f:
            self.db = pickle.load(f)

        # Filter by minimum points
        for name in list(self.db.keys()):
            self.db[name] = [
                entry for entry in self.db[name]
                if entry["num_points_in_gt"] >= self.min_points
            ]

    def sample(
        self,
        gt_boxes: np.ndarray,
        gt_names: List[str],
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Sample GT objects and paste them into the scene.

        Args:
            gt_boxes: (N, 9) existing ground truth boxes.
            gt_names: List of existing GT class names.
            points: (M, C) point cloud.

        Returns:
            Tuple of (augmented_gt_boxes, augmented_points, augmented_gt_names).
        """
        sampled_boxes_list: List[np.ndarray] = []
        sampled_points_list: List[np.ndarray] = []
        sampled_names_list: List[str] = []

        existing_boxes = gt_boxes.copy() if gt_boxes.shape[0] > 0 else np.zeros((0, 9), dtype=np.float32)

        for class_name in self.class_names:
            if class_name not in self.db or len(self.db[class_name]) == 0:
                continue

            # Count how many of this class already exist
            existing_count = sum(1 for n in gt_names if n == class_name)
            num_to_sample = max(0, self.sample_counts.get(class_name, 0) - existing_count)

            if num_to_sample <= 0:
                continue

            # Randomly select samples from the database
            available = self.db[class_name]
            indices = np.random.choice(
                len(available), size=min(num_to_sample, len(available)), replace=False
            )

            for sample_idx in indices:
                entry = available[sample_idx]
                box = np.array(entry["box3d"], dtype=np.float32)

                # Check collision with existing boxes
                if existing_boxes.shape[0] > 0 and _boxes_bev_iou_check(
                    box[np.newaxis, :7], existing_boxes[:, :7]
                ):
                    continue

                # Load the sampled points
                pts_path = self.dataset_root / entry["path"]
                if not pts_path.exists():
                    continue
                sampled_pts = np.fromfile(str(pts_path), dtype=np.float32)
                num_features = points.shape[1] if len(points.shape) > 1 else 5
                sampled_pts = sampled_pts.reshape(-1, num_features)

                # Translate points back to global position (they were stored relative to box center)
                sampled_pts[:, 0] += box[0]
                sampled_pts[:, 1] += box[1]
                sampled_pts[:, 2] += box[2]

                sampled_boxes_list.append(box[np.newaxis, :])
                sampled_points_list.append(sampled_pts)
                sampled_names_list.append(class_name)

                # Update existing boxes for collision checking
                existing_boxes = np.concatenate(
                    [existing_boxes, box[np.newaxis, :]], axis=0
                )

        if len(sampled_boxes_list) == 0:
            return gt_boxes, points, gt_names

        sampled_boxes = np.concatenate(sampled_boxes_list, axis=0)
        sampled_points = np.concatenate(sampled_points_list, axis=0)

        # Remove points from original cloud that fall inside sampled boxes
        points = _remove_points_in_boxes(points, sampled_boxes)

        # Combine
        augmented_boxes = np.concatenate([gt_boxes, sampled_boxes], axis=0)
        augmented_points = np.concatenate([points, sampled_points], axis=0)
        augmented_names = list(gt_names) + sampled_names_list

        return augmented_boxes, augmented_points, augmented_names


def _boxes_bev_iou_check(
    query_box: np.ndarray,
    existing_boxes: np.ndarray,
    threshold: float = 0.05,
) -> bool:
    """Check if query box overlaps with any existing box in BEV using axis-aligned approximation.

    Args:
        query_box: (1, 7) box [x, y, z, w, l, h, yaw].
        existing_boxes: (N, 7) existing boxes.
        threshold: Distance threshold as a fraction of box diagonal.

    Returns:
        True if there is a collision, False otherwise.
    """
    # Simple center distance check as fast collision proxy
    qx, qy = query_box[0, 0], query_box[0, 1]
    ex, ey = existing_boxes[:, 0], existing_boxes[:, 1]
    qw, ql = query_box[0, 3], query_box[0, 4]

    distances = np.sqrt((qx - ex) ** 2 + (qy - ey) ** 2)
    # Minimum clearance: half-diagonal of query box
    min_clearance = np.sqrt(qw ** 2 + ql ** 2) / 2.0 + threshold

    return bool(np.any(distances < min_clearance))


def _remove_points_in_boxes(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Remove points that fall inside any of the given 3D boxes.

    Args:
        points: (N, C) point cloud.
        boxes: (M, 9) bounding boxes [x, y, z, w, l, h, yaw, vx, vy].

    Returns:
        Filtered point cloud with interior points removed.
    """
    if boxes.shape[0] == 0:
        return points

    mask = np.ones(points.shape[0], dtype=bool)

    for i in range(boxes.shape[0]):
        cx, cy, cz = boxes[i, 0], boxes[i, 1], boxes[i, 2]
        bw, bl, bh = boxes[i, 3], boxes[i, 4], boxes[i, 5]
        yaw = boxes[i, 6]

        rot = rotation_matrix_z(-yaw)
        translated = points[:, :3] - np.array([cx, cy, cz], dtype=np.float32)
        rotated = translated @ rot.T

        in_box = (
            (np.abs(rotated[:, 0]) <= bw / 2.0) &
            (np.abs(rotated[:, 1]) <= bl / 2.0) &
            (np.abs(rotated[:, 2]) <= bh / 2.0)
        )
        mask &= ~in_box

    return points[mask]


def augment_point_cloud(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    rotation_range: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
    flip_x: bool = True,
    flip_y: bool = True,
    scale_range: Tuple[float, float] = (0.95, 1.05),
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random augmentations to point cloud and GT boxes.

    Applies random rotation around Z, random flipping along X and Y axes,
    and random uniform scaling.

    Args:
        points: (N, C) point cloud (at least x, y, z in first 3 columns).
        gt_boxes: (M, 9) boxes [x, y, z, w, l, h, yaw, vx, vy].
        rotation_range: (min_angle, max_angle) in radians for random Z rotation.
        flip_x: Whether to randomly flip along the X-axis.
        flip_y: Whether to randomly flip along the Y-axis.
        scale_range: (min_scale, max_scale) for random uniform scaling.

    Returns:
        Tuple of (augmented_points, augmented_gt_boxes).
    """
    points = points.copy()
    gt_boxes = gt_boxes.copy()

    # Random rotation around Z-axis
    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    rot_mat = rotation_matrix_z(angle)

    points[:, :3] = points[:, :3] @ rot_mat.T
    gt_boxes[:, :3] = gt_boxes[:, :3] @ rot_mat.T
    gt_boxes[:, 6] += angle

    # Rotate velocity vectors
    if gt_boxes.shape[1] >= 9:
        vel = gt_boxes[:, 7:9].copy()
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        gt_boxes[:, 7] = vel[:, 0] * cos_a - vel[:, 1] * sin_a
        gt_boxes[:, 8] = vel[:, 0] * sin_a + vel[:, 1] * cos_a

    # Random flip along X-axis
    if flip_x and np.random.random() < 0.5:
        points[:, 1] = -points[:, 1]
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        if gt_boxes.shape[1] >= 9:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    # Random flip along Y-axis
    if flip_y and np.random.random() < 0.5:
        points[:, 0] = -points[:, 0]
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = np.pi - gt_boxes[:, 6]
        if gt_boxes.shape[1] >= 9:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    # Random scaling
    scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= scale
    gt_boxes[:, :3] *= scale
    gt_boxes[:, 3:6] *= scale  # scale box dimensions
    if gt_boxes.shape[1] >= 9:
        gt_boxes[:, 7:9] *= scale  # scale velocity

    return points, gt_boxes


# ---------------------------------------------------------------------------
# NuScenes Dataset
# ---------------------------------------------------------------------------


class NuScenesDataset(Dataset):
    """NuScenes dataset for CenterPoint 3D object detection.

    Loads multi-sweep point clouds with ego-motion compensation, generates
    heatmap-based training targets, and applies data augmentation including
    GT-sampling, random rotation, flip, and scaling.

    Args:
        dataset_root: Path to the NuScenes dataset root.
        info_path: Path to the preprocessed info pickle file.
        class_names: List of detection class names.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy, vz] voxel dimensions.
        num_sweeps: Number of lidar sweeps to aggregate.
        max_points_per_voxel: Maximum points per voxel.
        max_voxels: Maximum voxels to generate.
        training: Whether this is for training (enables augmentation).
        gt_database_path: Path to GT database pickle for GT-sampling. None to disable.
        gt_sample_counts: Dict mapping class name to number of samples per scene.
        augmentation: Whether to apply geometric augmentations.
    """

    def __init__(
        self,
        dataset_root: str,
        info_path: str,
        class_names: Optional[List[str]] = None,
        point_cloud_range: Optional[List[float]] = None,
        voxel_size: Optional[List[float]] = None,
        num_sweeps: int = DEFAULT_NUM_SWEEPS,
        max_points_per_voxel: int = DEFAULT_MAX_POINTS_PER_VOXEL,
        max_voxels: int = DEFAULT_MAX_VOXELS,
        training: bool = True,
        gt_database_path: Optional[str] = None,
        gt_sample_counts: Optional[Dict[str, int]] = None,
        augmentation: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.class_names = class_names if class_names is not None else NUSCENES_CLASS_NAMES
        self.point_cloud_range = point_cloud_range if point_cloud_range is not None else DEFAULT_POINT_CLOUD_RANGE
        self.voxel_size = voxel_size if voxel_size is not None else DEFAULT_VOXEL_SIZE
        self.num_sweeps = num_sweeps
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.training = training
        self.augmentation = augmentation and training

        # Load dataset info
        with open(info_path, "rb") as f:
            self.infos = pickle.load(f)

        # Build class name to index mapping
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        self.num_classes = len(self.class_names)

        # Initialize GT sampler if database path is provided
        self.gt_sampler: Optional[GTDatabaseSampler] = None
        if gt_database_path is not None and training:
            self.gt_sampler = GTDatabaseSampler(
                database_path=gt_database_path,
                dataset_root=str(self.dataset_root),
                class_names=self.class_names,
                sample_counts=gt_sample_counts,
                point_cloud_range=self.point_cloud_range,
            )

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single training/inference sample.

        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing:
                - voxels: (M, max_pts_per_voxel, C) voxel features.
                - coordinates: (M, 3) voxel coordinates (z, y, x).
                - num_points_per_voxel: (M,) point counts.
                - targets: dict of heatmap and regression targets (training only).
                - metadata: dict with token, lidar_path, etc.
        """
        info = copy.deepcopy(self.infos[idx])

        # Load multi-sweep point cloud
        points = self._load_sweeps(info)

        # Load annotations
        gt_boxes, gt_names, gt_classes = self._load_annotations(info)

        # Apply GT sampling augmentation
        if self.gt_sampler is not None and gt_boxes.shape[0] >= 0:
            gt_boxes, points, gt_names = self.gt_sampler.sample(
                gt_boxes, gt_names, points
            )
            # Recompute class indices after sampling
            gt_classes = np.array(
                [self.class_to_idx.get(n, -1) for n in gt_names], dtype=np.int32
            )

        # Apply geometric augmentations
        if self.augmentation and gt_boxes.shape[0] > 0:
            points, gt_boxes = augment_point_cloud(
                points, gt_boxes,
                rotation_range=(-np.pi / 4, np.pi / 4),
                flip_x=True,
                flip_y=True,
                scale_range=(0.95, 1.05),
            )

        # Filter GT boxes: only keep valid classes and boxes within range
        valid_mask = gt_classes >= 0
        if self.training:
            range_mask = self._boxes_in_range(gt_boxes)
            valid_mask = valid_mask & range_mask
        gt_boxes = gt_boxes[valid_mask]
        gt_classes = gt_classes[valid_mask]

        # Voxelize the point cloud
        voxels, coordinates, num_points_per_voxel = self.get_voxels(points)

        # Generate targets
        targets = {}
        if self.training:
            targets = generate_targets(
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size,
                num_classes=self.num_classes,
            )

        metadata = {
            "token": info.get("token", f"sample_{idx}"),
            "lidar_path": info.get("lidar_path", ""),
            "num_points": points.shape[0],
        }

        return {
            "voxels": voxels.astype(np.float32),
            "coordinates": coordinates.astype(np.int32),
            "num_points_per_voxel": num_points_per_voxel.astype(np.int32),
            "targets": targets,
            "metadata": metadata,
        }

    def _load_sweeps(self, info: Dict[str, Any]) -> np.ndarray:
        """Load and aggregate multi-sweep point clouds with ego-motion compensation.

        Args:
            info: Sample info dictionary containing lidar_path and sweeps.

        Returns:
            (N, 5) aggregated point cloud [x, y, z, intensity, time_lag].
        """
        # Load keyframe
        lidar_path = self.dataset_root / info["lidar_path"]
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)

        # Add zero time lag for keyframe
        if points.shape[1] == 4:
            time_lag = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.hstack([points, time_lag])

        sweep_points_list = [points]

        # Aggregate past sweeps
        sweeps = info.get("sweeps", [])
        num_additional_sweeps = min(self.num_sweeps - 1, len(sweeps))

        for i in range(num_additional_sweeps):
            sweep = sweeps[i]
            sweep_path = self.dataset_root / sweep["lidar_path"]

            if not sweep_path.exists():
                continue

            sweep_pts = np.fromfile(str(sweep_path), dtype=np.float32).reshape(-1, 5)

            # Apply ego-motion compensation: transform sweep points to keyframe coords
            if "transform_matrix" in sweep:
                transform = np.array(sweep["transform_matrix"], dtype=np.float32)  # 4x4
                num_pts = sweep_pts.shape[0]
                pts_hom = np.ones((num_pts, 4), dtype=np.float32)
                pts_hom[:, :3] = sweep_pts[:, :3]
                pts_hom = pts_hom @ transform.T
                sweep_pts[:, :3] = pts_hom[:, :3]

            # Set time lag
            time_lag_val = sweep.get("time_lag", (i + 1) * 0.05)
            if sweep_pts.shape[1] == 4:
                time_lag = np.full((sweep_pts.shape[0], 1), time_lag_val, dtype=np.float32)
                sweep_pts = np.hstack([sweep_pts, time_lag])
            else:
                sweep_pts[:, 4] = time_lag_val

            sweep_points_list.append(sweep_pts)

        points = np.concatenate(sweep_points_list, axis=0)
        return points

    def _load_annotations(
        self, info: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Load 3D bounding box annotations from info.

        Args:
            info: Sample info dictionary.

        Returns:
            Tuple of (gt_boxes, gt_names, gt_classes) where:
                - gt_boxes: (N, 9) [x, y, z, w, l, h, yaw, vx, vy]
                - gt_names: list of class name strings
                - gt_classes: (N,) integer class indices (0-indexed, -1 for unknown)
        """
        if "gt_boxes" not in info or "gt_names" not in info:
            return (
                np.zeros((0, 9), dtype=np.float32),
                [],
                np.zeros((0,), dtype=np.int32),
            )

        gt_boxes = np.array(info["gt_boxes"], dtype=np.float32)
        gt_names = list(info["gt_names"])

        # Ensure boxes have 9 columns (pad velocity if missing)
        if gt_boxes.ndim == 1:
            gt_boxes = gt_boxes.reshape(-1, gt_boxes.shape[0])
        if gt_boxes.shape[1] < 9:
            padding = np.zeros(
                (gt_boxes.shape[0], 9 - gt_boxes.shape[1]), dtype=np.float32
            )
            gt_boxes = np.hstack([gt_boxes, padding])

        gt_classes = np.array(
            [self.class_to_idx.get(name, -1) for name in gt_names], dtype=np.int32
        )

        return gt_boxes, gt_names, gt_classes

    def _boxes_in_range(self, boxes: np.ndarray) -> np.ndarray:
        """Check which boxes have their center within the point cloud range.

        Args:
            boxes: (N, 9) bounding boxes.

        Returns:
            (N,) boolean mask of boxes within range.
        """
        if boxes.shape[0] == 0:
            return np.zeros((0,), dtype=bool)

        pcr = self.point_cloud_range
        mask = (
            (boxes[:, 0] >= pcr[0]) & (boxes[:, 0] <= pcr[3]) &
            (boxes[:, 1] >= pcr[1]) & (boxes[:, 1] <= pcr[4]) &
            (boxes[:, 2] >= pcr[2]) & (boxes[:, 2] <= pcr[5])
        )
        return mask

    def get_voxels(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxelize the point cloud.

        Args:
            points: (N, C) point cloud array.

        Returns:
            Tuple of (voxels, coordinates, num_points_per_voxel).
        """
        return voxelize_points(
            points=points,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_points_per_voxel=self.max_points_per_voxel,
            max_voxels=self.max_voxels,
        )


# ---------------------------------------------------------------------------
# Waymo Dataset
# ---------------------------------------------------------------------------


class WaymoDataset(Dataset):
    """Waymo Open Dataset loader for CenterPoint 3D object detection.

    Similar to NuScenesDataset but handles Waymo-specific data format:
    - Point clouds stored as .bin files with (x, y, z, intensity, elongation)
    - Annotations include tracking IDs
    - Different coordinate conventions

    Args:
        dataset_root: Path to the Waymo dataset root.
        info_path: Path to the preprocessed info pickle file.
        class_names: List of detection class names.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy, vz] voxel dimensions.
        num_sweeps: Number of lidar sweeps to aggregate.
        max_points_per_voxel: Maximum points per voxel.
        max_voxels: Maximum voxels to generate.
        training: Whether this is for training (enables augmentation).
        gt_database_path: Path to GT database pickle for GT-sampling.
        gt_sample_counts: Dict mapping class name to number of samples per scene.
        augmentation: Whether to apply geometric augmentations.
    """

    def __init__(
        self,
        dataset_root: str,
        info_path: str,
        class_names: Optional[List[str]] = None,
        point_cloud_range: Optional[List[float]] = None,
        voxel_size: Optional[List[float]] = None,
        num_sweeps: int = DEFAULT_NUM_SWEEPS,
        max_points_per_voxel: int = DEFAULT_MAX_POINTS_PER_VOXEL,
        max_voxels: int = DEFAULT_MAX_VOXELS,
        training: bool = True,
        gt_database_path: Optional[str] = None,
        gt_sample_counts: Optional[Dict[str, int]] = None,
        augmentation: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.class_names = class_names if class_names is not None else WAYMO_CLASS_NAMES
        self.point_cloud_range = point_cloud_range if point_cloud_range is not None else DEFAULT_POINT_CLOUD_RANGE
        self.voxel_size = voxel_size if voxel_size is not None else DEFAULT_VOXEL_SIZE
        self.num_sweeps = num_sweeps
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        self.training = training
        self.augmentation = augmentation and training

        # Load dataset info
        with open(info_path, "rb") as f:
            self.infos = pickle.load(f)

        # Build class name to index mapping
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        self.num_classes = len(self.class_names)

        # Initialize GT sampler
        self.gt_sampler: Optional[GTDatabaseSampler] = None
        if gt_database_path is not None and training:
            self.gt_sampler = GTDatabaseSampler(
                database_path=gt_database_path,
                dataset_root=str(self.dataset_root),
                class_names=self.class_names,
                sample_counts=gt_sample_counts,
                point_cloud_range=self.point_cloud_range,
            )

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single training/inference sample.

        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing voxels, coordinates, num_points_per_voxel,
            targets, and metadata.
        """
        info = copy.deepcopy(self.infos[idx])

        # Load multi-sweep point cloud
        points = self._load_sweeps(info)

        # Load annotations
        gt_boxes, gt_names, gt_classes, tracking_ids = self._load_annotations(info)

        # Apply GT sampling augmentation
        if self.gt_sampler is not None and gt_boxes.shape[0] >= 0:
            gt_boxes, points, gt_names = self.gt_sampler.sample(
                gt_boxes, gt_names, points
            )
            gt_classes = np.array(
                [self.class_to_idx.get(n, -1) for n in gt_names], dtype=np.int32
            )

        # Apply geometric augmentations
        if self.augmentation and gt_boxes.shape[0] > 0:
            points, gt_boxes = augment_point_cloud(
                points, gt_boxes,
                rotation_range=(-np.pi / 4, np.pi / 4),
                flip_x=True,
                flip_y=True,
                scale_range=(0.95, 1.05),
            )

        # Filter GT boxes
        valid_mask = gt_classes >= 0
        if self.training:
            range_mask = self._boxes_in_range(gt_boxes)
            valid_mask = valid_mask & range_mask
        gt_boxes = gt_boxes[valid_mask]
        gt_classes = gt_classes[valid_mask]

        # Voxelize
        voxels, coordinates, num_points_per_voxel = self.get_voxels(points)

        # Generate targets
        targets = {}
        if self.training:
            targets = generate_targets(
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size,
                num_classes=self.num_classes,
            )

        metadata = {
            "sequence_id": info.get("sequence_id", ""),
            "frame_id": info.get("frame_id", idx),
            "lidar_path": info.get("lidar_path", ""),
            "num_points": points.shape[0],
            "tracking_ids": tracking_ids[valid_mask].tolist() if tracking_ids is not None else [],
        }

        return {
            "voxels": voxels.astype(np.float32),
            "coordinates": coordinates.astype(np.int32),
            "num_points_per_voxel": num_points_per_voxel.astype(np.int32),
            "targets": targets,
            "metadata": metadata,
        }

    def _load_sweeps(self, info: Dict[str, Any]) -> np.ndarray:
        """Load and aggregate multi-sweep point clouds with ego-motion compensation.

        Waymo point clouds have 5 features: (x, y, z, intensity, elongation).
        A time_lag column is appended as the 6th feature.

        Args:
            info: Sample info dictionary.

        Returns:
            (N, 6) aggregated point cloud [x, y, z, intensity, elongation, time_lag].
        """
        lidar_path = self.dataset_root / info["lidar_path"]
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)

        # Append time lag = 0 for keyframe
        time_lag = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack([points, time_lag])

        sweep_points_list = [points]

        # Aggregate past sweeps
        sweeps = info.get("sweeps", [])
        num_additional_sweeps = min(self.num_sweeps - 1, len(sweeps))

        for i in range(num_additional_sweeps):
            sweep = sweeps[i]
            sweep_path = self.dataset_root / sweep["lidar_path"]

            if not sweep_path.exists():
                continue

            sweep_pts = np.fromfile(str(sweep_path), dtype=np.float32).reshape(-1, 5)

            # Apply ego-motion compensation via provided pose/transform
            if "pose" in sweep and "ref_pose" in info:
                # Waymo stores global poses; compute relative transform
                sweep_pose = np.array(sweep["pose"], dtype=np.float32).reshape(4, 4)
                ref_pose = np.array(info["ref_pose"], dtype=np.float32).reshape(4, 4)
                # Transform: ref_pose_inv @ sweep_pose maps sweep to ref frame
                transform = np.linalg.inv(ref_pose) @ sweep_pose

                num_pts = sweep_pts.shape[0]
                pts_hom = np.ones((num_pts, 4), dtype=np.float32)
                pts_hom[:, :3] = sweep_pts[:, :3]
                pts_hom = pts_hom @ transform.T
                sweep_pts[:, :3] = pts_hom[:, :3]
            elif "transform_matrix" in sweep:
                transform = np.array(sweep["transform_matrix"], dtype=np.float32).reshape(4, 4)
                num_pts = sweep_pts.shape[0]
                pts_hom = np.ones((num_pts, 4), dtype=np.float32)
                pts_hom[:, :3] = sweep_pts[:, :3]
                pts_hom = pts_hom @ transform.T
                sweep_pts[:, :3] = pts_hom[:, :3]

            # Append time lag
            time_lag_val = sweep.get("time_lag", (i + 1) * 0.1)
            time_col = np.full((sweep_pts.shape[0], 1), time_lag_val, dtype=np.float32)
            sweep_pts = np.hstack([sweep_pts, time_col])

            sweep_points_list.append(sweep_pts)

        points = np.concatenate(sweep_points_list, axis=0)
        return points

    def _load_annotations(
        self, info: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str], np.ndarray, Optional[np.ndarray]]:
        """Load 3D bounding box annotations from Waymo info.

        Args:
            info: Sample info dictionary.

        Returns:
            Tuple of (gt_boxes, gt_names, gt_classes, tracking_ids) where:
                - gt_boxes: (N, 9) [x, y, z, w, l, h, yaw, vx, vy]
                - gt_names: list of class name strings
                - gt_classes: (N,) integer class indices
                - tracking_ids: (N,) integer tracking IDs or None
        """
        if "gt_boxes" not in info or "gt_names" not in info:
            return (
                np.zeros((0, 9), dtype=np.float32),
                [],
                np.zeros((0,), dtype=np.int32),
                None,
            )

        gt_boxes = np.array(info["gt_boxes"], dtype=np.float32)
        gt_names = list(info["gt_names"])

        # Ensure 9 columns
        if gt_boxes.ndim == 1:
            gt_boxes = gt_boxes.reshape(-1, gt_boxes.shape[0])
        if gt_boxes.shape[1] < 9:
            padding = np.zeros(
                (gt_boxes.shape[0], 9 - gt_boxes.shape[1]), dtype=np.float32
            )
            gt_boxes = np.hstack([gt_boxes, padding])

        gt_classes = np.array(
            [self.class_to_idx.get(name, -1) for name in gt_names], dtype=np.int32
        )

        # Tracking IDs
        tracking_ids = None
        if "tracking_ids" in info:
            tracking_ids = np.array(info["tracking_ids"], dtype=np.int64)

        return gt_boxes, gt_names, gt_classes, tracking_ids

    def _boxes_in_range(self, boxes: np.ndarray) -> np.ndarray:
        """Check which boxes have their center within the point cloud range.

        Args:
            boxes: (N, 9) bounding boxes.

        Returns:
            (N,) boolean mask.
        """
        if boxes.shape[0] == 0:
            return np.zeros((0,), dtype=bool)

        pcr = self.point_cloud_range
        mask = (
            (boxes[:, 0] >= pcr[0]) & (boxes[:, 0] <= pcr[3]) &
            (boxes[:, 1] >= pcr[1]) & (boxes[:, 1] <= pcr[4]) &
            (boxes[:, 2] >= pcr[2]) & (boxes[:, 2] <= pcr[5])
        )
        return mask

    def get_voxels(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxelize the point cloud.

        Args:
            points: (N, C) point cloud array.

        Returns:
            Tuple of (voxels, coordinates, num_points_per_voxel).
        """
        return voxelize_points(
            points=points,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_points_per_voxel=self.max_points_per_voxel,
            max_voxels=self.max_voxels,
        )


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching variable-size voxelized data.

    Voxel tensors have variable lengths across samples, so they cannot be
    simply stacked. This function concatenates them and prepends a batch index
    to the coordinates tensor.

    Args:
        batch: List of sample dictionaries from the dataset __getitem__.

    Returns:
        Batched dictionary with:
            - voxels: (M_total, max_pts_per_voxel, C) concatenated voxel features.
            - coordinates: (M_total, 4) with batch index prepended: (batch_id, z, y, x).
            - num_points_per_voxel: (M_total,) concatenated point counts.
            - targets: batched target tensors (stacked heatmaps, padded regression).
            - metadata: list of per-sample metadata dicts.
    """
    voxels_list: List[np.ndarray] = []
    coords_list: List[np.ndarray] = []
    num_points_list: List[np.ndarray] = []
    targets_list: List[Dict[str, np.ndarray]] = []
    metadata_list: List[Dict[str, Any]] = []

    for i, sample in enumerate(batch):
        voxels = sample["voxels"]
        coords = sample["coordinates"]
        num_pts = sample["num_points_per_voxel"]

        # Prepend batch index to coordinates
        batch_col = np.full((coords.shape[0], 1), i, dtype=np.int32)
        coords_with_batch = np.hstack([batch_col, coords])

        voxels_list.append(voxels)
        coords_list.append(coords_with_batch)
        num_points_list.append(num_pts)
        targets_list.append(sample.get("targets", {}))
        metadata_list.append(sample.get("metadata", {}))

    # Concatenate variable-length voxel data
    batched_voxels = torch.from_numpy(np.concatenate(voxels_list, axis=0))
    batched_coords = torch.from_numpy(np.concatenate(coords_list, axis=0))
    batched_num_points = torch.from_numpy(np.concatenate(num_points_list, axis=0))

    # Batch targets: stack heatmaps, pad regression targets
    batched_targets = _collate_targets(targets_list)

    return {
        "voxels": batched_voxels,
        "coordinates": batched_coords,
        "num_points_per_voxel": batched_num_points,
        "targets": batched_targets,
        "metadata": metadata_list,
    }


def _collate_targets(
    targets_list: List[Dict[str, np.ndarray]],
) -> Dict[str, torch.Tensor]:
    """Collate per-sample targets into batched tensors.

    Args:
        targets_list: List of target dictionaries from each sample.

    Returns:
        Dictionary of batched target tensors.
    """
    if not targets_list or not targets_list[0]:
        return {}

    batch_size = len(targets_list)

    # Stack heatmaps directly (all same spatial size)
    heatmaps = np.stack([t["heatmap"] for t in targets_list], axis=0)

    # Pad regression targets to the maximum number of objects across the batch
    max_objects = max(t["reg_targets"].shape[0] for t in targets_list)

    reg_targets = np.zeros((batch_size, max_objects, 8), dtype=np.float32)
    velocity_targets = np.zeros((batch_size, max_objects, 2), dtype=np.float32)
    target_indices = np.zeros((batch_size, max_objects), dtype=np.int64)
    target_mask = np.zeros((batch_size, max_objects), dtype=np.float32)

    for i, t in enumerate(targets_list):
        n = t["reg_targets"].shape[0]
        reg_targets[i, :n] = t["reg_targets"]
        velocity_targets[i, :n] = t["velocity_targets"]
        target_indices[i, :n] = t["target_indices"]
        target_mask[i, :n] = t["target_mask"]

    return {
        "heatmap": torch.from_numpy(heatmaps),
        "reg_targets": torch.from_numpy(reg_targets),
        "velocity_targets": torch.from_numpy(velocity_targets),
        "target_indices": torch.from_numpy(target_indices),
        "target_mask": torch.from_numpy(target_mask),
    }
