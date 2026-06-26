"""
Point cloud datasets for PointPillars 3D object detection.

Provides PyTorch Dataset implementations for KITTI and nuScenes benchmarks,
along with data augmentation utilities and a custom collate function for
variable-size batching.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Return a 3x3 rotation matrix around the Z-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix as float64 ndarray.
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def boxes_camera_to_lidar(
    boxes_cam: np.ndarray, R0_rect: np.ndarray, Tr_velo_to_cam: np.ndarray
) -> np.ndarray:
    """Convert 3-D boxes from camera frame to LiDAR (velodyne) frame.

    Camera box format:  (x, y, z, h, w, l, ry) -- KITTI camera convention
    LiDAR box format:   (x, y, z, dx, dy, dz, heading) -- right-hand LiDAR frame

    Args:
        boxes_cam: (N, 7) boxes in camera coordinates.
        R0_rect: (3, 3) rectifying rotation matrix.
        Tr_velo_to_cam: (3, 4) velodyne-to-camera transform.

    Returns:
        (N, 7) boxes in LiDAR coordinates.
    """
    if boxes_cam.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)

    # Build inverse transform: camera -> velodyne
    R0_inv = np.linalg.inv(R0_rect)
    Tr_rot = Tr_velo_to_cam[:, :3]
    Tr_trans = Tr_velo_to_cam[:, 3]
    Tr_rot_inv = np.linalg.inv(Tr_rot)

    # Convert location
    locs_cam = boxes_cam[:, :3]  # (N, 3)
    locs_rect = (R0_inv @ locs_cam.T).T  # (N, 3) in unrectified camera frame
    locs_lidar = (Tr_rot_inv @ (locs_rect - Tr_trans).T).T  # (N, 3)

    # Convert dimensions: camera (h, w, l) -> lidar (dx, dy, dz)
    h = boxes_cam[:, 3]
    w = boxes_cam[:, 4]
    l = boxes_cam[:, 5]  # noqa: E741
    dx = l
    dy = w
    dz = h

    # Convert rotation: camera ry -> lidar heading
    # In KITTI, ry is rotation around camera Y-axis (downward).  In LiDAR frame
    # this corresponds to rotation around Z-axis with a -pi/2 offset.
    ry = boxes_cam[:, 6]
    heading = -(ry + np.pi / 2.0)

    boxes_lidar = np.stack([locs_lidar[:, 0], locs_lidar[:, 1], locs_lidar[:, 2],
                            dx, dy, dz, heading], axis=1).astype(np.float32)
    return boxes_lidar


def filter_points_by_range(points: np.ndarray, point_cloud_range: Sequence[float]) -> np.ndarray:
    """Remove points outside the given 3-D range.

    Args:
        points: (N, C) point cloud, first three columns are x, y, z.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        Filtered subset of points within the range.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range[:6]
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return points[mask]


def filter_boxes_by_range(boxes: np.ndarray, point_cloud_range: Sequence[float]) -> np.ndarray:
    """Return boolean mask for boxes whose centers fall inside the range.

    Args:
        boxes: (N, 7) boxes with center (x, y, z) in columns 0-2.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        Boolean mask of length N.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range[:6]
    mask = (
        (boxes[:, 0] >= x_min)
        & (boxes[:, 0] <= x_max)
        & (boxes[:, 1] >= y_min)
        & (boxes[:, 1] <= y_max)
        & (boxes[:, 2] >= z_min)
        & (boxes[:, 2] <= z_max)
    )
    return mask


def boxes3d_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert (N, 7) boxes to 8-corner representation for collision checking.

    Each box is parameterized as (cx, cy, cz, dx, dy, dz, heading).

    Args:
        boxes: (N, 7) array.

    Returns:
        (N, 8, 3) corner coordinates.
    """
    n = boxes.shape[0]
    # Unit box corners relative to center
    # dx along x, dy along y, dz along z
    template = np.array(
        [
            [1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
            [-1, 1, -1],
            [1, 1, 1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    ) * 0.5  # (8, 3)

    corners_all = np.zeros((n, 8, 3), dtype=np.float32)
    for i in range(n):
        cx, cy, cz, dx, dy, dz, heading = boxes[i]
        # Scale template by box dimensions
        scaled = template * np.array([dx, dy, dz], dtype=np.float32)  # (8, 3)
        # Rotate around Z
        rot = rotation_matrix_z(heading).astype(np.float32)
        rotated = (rot @ scaled.T).T  # (8, 3)
        # Translate
        corners_all[i] = rotated + np.array([cx, cy, cz], dtype=np.float32)
    return corners_all


def boxes_bev_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute axis-aligned BEV IoU between two sets of boxes.

    Uses the bounding rectangle of rotated boxes for a fast approximation.

    Args:
        boxes_a: (M, 7) first set.
        boxes_b: (N, 7) second set.

    Returns:
        (M, N) IoU matrix.
    """
    # Compute axis-aligned bounding rectangles in BEV from corners
    corners_a = boxes3d_to_corners(boxes_a)[:, :, :2]  # (M, 8, 2)
    corners_b = boxes3d_to_corners(boxes_b)[:, :, :2]  # (N, 8, 2)

    min_a = corners_a.min(axis=1)  # (M, 2)
    max_a = corners_a.max(axis=1)  # (M, 2)
    min_b = corners_b.min(axis=1)  # (N, 2)
    max_b = corners_b.max(axis=1)  # (N, 2)

    area_a = (max_a[:, 0] - min_a[:, 0]) * (max_a[:, 1] - min_a[:, 1])  # (M,)
    area_b = (max_b[:, 0] - min_b[:, 0]) * (max_b[:, 1] - min_b[:, 1])  # (N,)

    m = boxes_a.shape[0]
    n = boxes_b.shape[0]
    iou = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        # Intersection coordinates
        inter_min = np.maximum(min_a[i], min_b)  # (N, 2)
        inter_max = np.minimum(max_a[i], max_b)  # (N, 2)
        inter_wh = np.maximum(inter_max - inter_min, 0.0)  # (N, 2)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]  # (N,)
        union_area = area_a[i] + area_b - inter_area
        valid = union_area > 0
        iou[i, valid] = inter_area[valid] / union_area[valid]
    return iou


# ---------------------------------------------------------------------------
# Data augmentation functions
# ---------------------------------------------------------------------------


def random_global_rotation(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    rotation_range: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a random global rotation around the Z-axis to the point cloud and boxes.

    Args:
        points: (N, C) point cloud; first three columns are x, y, z.
        gt_boxes: (M, 7) ground-truth boxes (cx, cy, cz, dx, dy, dz, heading).
        rotation_range: (min_angle, max_angle) in radians.

    Returns:
        Tuple of rotated (points, gt_boxes).
    """
    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    rot = rotation_matrix_z(angle).astype(np.float32)

    # Rotate point coordinates
    points_out = points.copy()
    points_out[:, :3] = (rot @ points[:, :3].T).T

    # Rotate box centers and headings
    gt_boxes_out = gt_boxes.copy()
    gt_boxes_out[:, :3] = (rot @ gt_boxes[:, :3].T).T
    gt_boxes_out[:, 6] += angle

    return points_out, gt_boxes_out


def random_global_scaling(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    scale_range: Tuple[float, float] = (0.95, 1.05),
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a random global scaling to the point cloud and boxes.

    Args:
        points: (N, C) point cloud.
        gt_boxes: (M, 7) ground-truth boxes.
        scale_range: (min_scale, max_scale).

    Returns:
        Tuple of scaled (points, gt_boxes).
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])

    points_out = points.copy()
    points_out[:, :3] *= scale

    gt_boxes_out = gt_boxes.copy()
    gt_boxes_out[:, :3] *= scale  # centers
    gt_boxes_out[:, 3:6] *= scale  # dimensions

    return points_out, gt_boxes_out


def random_flip(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    along_x: bool = True,
    along_y: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly flip the point cloud and boxes along specified axes.

    Flipping along X means negating the Y coordinate (left-right mirror).
    Flipping along Y means negating the X coordinate (front-back mirror).
    Each axis is flipped independently with probability 0.5.

    Args:
        points: (N, C) point cloud.
        gt_boxes: (M, 7) ground-truth boxes.
        along_x: Whether to consider flipping along X-axis (negate Y).
        along_y: Whether to consider flipping along Y-axis (negate X).

    Returns:
        Tuple of (points, gt_boxes) after flipping.
    """
    points_out = points.copy()
    gt_boxes_out = gt_boxes.copy()

    if along_x and np.random.random() < 0.5:
        points_out[:, 1] = -points_out[:, 1]
        gt_boxes_out[:, 1] = -gt_boxes_out[:, 1]
        gt_boxes_out[:, 6] = -gt_boxes_out[:, 6]

    if along_y and np.random.random() < 0.5:
        points_out[:, 0] = -points_out[:, 0]
        gt_boxes_out[:, 0] = -gt_boxes_out[:, 0]
        gt_boxes_out[:, 6] = np.pi - gt_boxes_out[:, 6]

    return points_out, gt_boxes_out


def gt_database_sampling(
    points: np.ndarray,
    gt_boxes: np.ndarray,
    gt_names: np.ndarray,
    db_infos: Dict[str, List[Dict[str, Any]]],
    sample_groups: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground-truth database augmentation (copy-paste).

    Samples additional annotated objects from a pre-built database and pastes
    them into the current scene, avoiding collisions with existing boxes.

    Args:
        points: (N, C) current scene point cloud.
        gt_boxes: (M, 7) existing ground-truth boxes.
        gt_names: (M,) class names for existing boxes.
        db_infos: Mapping from class name to list of DB sample dicts. Each dict
            must contain:
                - 'box3d_lidar': (7,) box parameters in LiDAR frame
                - 'path': str path to the saved point cloud segment (.bin)
                - 'name': str class name
                - 'num_points_in_gt': int number of points
        sample_groups: Mapping from class name to the desired *total* number of
            instances of that class in the augmented scene.

    Returns:
        Tuple of augmented (points, gt_boxes, gt_names).
    """
    existing_boxes = gt_boxes.copy()
    all_new_points: List[np.ndarray] = []
    all_new_boxes: List[np.ndarray] = []
    all_new_names: List[str] = []

    for class_name, target_count in sample_groups.items():
        if class_name not in db_infos or len(db_infos[class_name]) == 0:
            continue

        # How many of this class already exist?
        current_count = int((gt_names == class_name).sum())
        num_to_sample = max(0, target_count - current_count)
        if num_to_sample == 0:
            continue

        # Randomly select candidates from DB
        db_list = db_infos[class_name]
        indices = np.random.choice(len(db_list), size=min(num_to_sample * 3, len(db_list)), replace=False)
        sampled_count = 0

        for idx in indices:
            if sampled_count >= num_to_sample:
                break
            info = db_list[idx]
            sample_box = np.array(info["box3d_lidar"], dtype=np.float32).reshape(1, 7)

            # Collision check with all existing + already-added boxes
            if existing_boxes.shape[0] > 0:
                iou = boxes_bev_iou(sample_box, existing_boxes)
                if iou.max() > 0.0:
                    continue  # Collision detected, skip this sample

            # Load points for the sampled object
            sample_points_path = Path(info["path"])
            if not sample_points_path.exists():
                continue
            obj_points = np.fromfile(str(sample_points_path), dtype=np.float32).reshape(-1, points.shape[1])

            # Add to scene
            all_new_points.append(obj_points)
            all_new_boxes.append(sample_box)
            all_new_names.append(info["name"])
            existing_boxes = np.concatenate([existing_boxes, sample_box], axis=0)
            sampled_count += 1

    # Remove points inside newly-added boxes from the original cloud
    if len(all_new_boxes) > 0:
        new_boxes_arr = np.concatenate(all_new_boxes, axis=0)  # (K, 7)
        # Remove original points that fall inside any new box (simple AABB check)
        corners = boxes3d_to_corners(new_boxes_arr)  # (K, 8, 3)
        bev_min = corners[:, :, :2].min(axis=1)  # (K, 2)
        bev_max = corners[:, :, :2].max(axis=1)  # (K, 2)
        z_min = corners[:, :, 2].min(axis=1)  # (K,)
        z_max = corners[:, :, 2].max(axis=1)  # (K,)

        keep_mask = np.ones(points.shape[0], dtype=bool)
        for k in range(new_boxes_arr.shape[0]):
            in_box = (
                (points[:, 0] >= bev_min[k, 0])
                & (points[:, 0] <= bev_max[k, 0])
                & (points[:, 1] >= bev_min[k, 1])
                & (points[:, 1] <= bev_max[k, 1])
                & (points[:, 2] >= z_min[k])
                & (points[:, 2] <= z_max[k])
            )
            keep_mask &= ~in_box
        points = points[keep_mask]

        # Concatenate all new points
        all_new_points_arr = np.concatenate(all_new_points, axis=0)
        points = np.concatenate([points, all_new_points_arr], axis=0)

        # Update boxes and names
        gt_boxes = np.concatenate([gt_boxes, new_boxes_arr], axis=0)
        gt_names = np.concatenate([gt_names, np.array(all_new_names)])

    return points, gt_boxes, gt_names


# ---------------------------------------------------------------------------
# KITTI Dataset
# ---------------------------------------------------------------------------


class KITTIDataset(Dataset):
    """PyTorch Dataset for the KITTI 3D object detection benchmark.

    Loads point clouds from .bin files, labels from KITTI-format .txt annotation
    files, and calibration data to transform annotations into the LiDAR frame.

    Directory layout expected::

        root/
            velodyne/     *.bin  (N, 4) float32 point clouds
            label_2/      *.txt  KITTI annotations
            calib/        *.txt  calibration matrices
            ImageSets/    train.txt / val.txt / test.txt

    Args:
        root_dir: Path to the KITTI data root.
        split: One of 'train', 'val', 'test', or 'trainval'.
        class_names: List of class names to keep (e.g. ['Car', 'Pedestrian', 'Cyclist']).
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        augment: Whether to apply data augmentation.
        db_infos_path: Optional path to a ground-truth database pickle for GT sampling.
        sample_groups: Class-to-count mapping for GT database sampling.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        class_names: Optional[List[str]] = None,
        point_cloud_range: Optional[List[float]] = None,
        augment: bool = False,
        db_infos_path: Optional[str] = None,
        sample_groups: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.class_names = class_names if class_names is not None else ["Car", "Pedestrian", "Cyclist"]
        self.point_cloud_range = point_cloud_range if point_cloud_range is not None else [0, -39.68, -3, 69.12, 39.68, 1]
        self.augment = augment
        self.sample_groups = sample_groups if sample_groups is not None else {}

        # Load frame indices
        split_file = self.root_dir / "ImageSets" / f"{split}.txt"
        with open(split_file, "r") as f:
            self.frame_ids = [line.strip() for line in f.readlines() if line.strip()]

        # Load GT database if specified
        self.db_infos: Dict[str, List[Dict[str, Any]]] = {}
        if db_infos_path is not None and Path(db_infos_path).exists():
            with open(db_infos_path, "rb") as f:
                self.db_infos = pickle.load(f)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _load_point_cloud(self, frame_id: str) -> np.ndarray:
        """Load a velodyne point cloud from a .bin file.

        Args:
            frame_id: Six-digit frame identifier (e.g. '000042').

        Returns:
            (N, 4) float32 array of (x, y, z, intensity).
        """
        bin_path = self.root_dir / "velodyne" / f"{frame_id}.bin"
        points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
        return points

    def _load_calibration(self, frame_id: str) -> Dict[str, np.ndarray]:
        """Parse a KITTI calibration file.

        Args:
            frame_id: Frame identifier.

        Returns:
            Dict with keys 'P0'..'P3', 'R0_rect', 'Tr_velo_to_cam'.
        """
        calib_path = self.root_dir / "calib" / f"{frame_id}.txt"
        calib: Dict[str, np.ndarray] = {}
        with open(calib_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                values = np.array([float(v) for v in value.strip().split()], dtype=np.float64)
                if key in ("P0", "P1", "P2", "P3"):
                    calib[key] = values.reshape(3, 4)
                elif key == "R0_rect":
                    calib["R0_rect"] = values.reshape(3, 3)
                elif key in ("Tr_velo_to_cam", "Tr_velo_cam"):
                    calib["Tr_velo_to_cam"] = values.reshape(3, 4)
        return calib

    def _load_labels(self, frame_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load KITTI label file and return camera-frame boxes and class names.

        KITTI label format per line:
            type truncated occluded alpha bbox2d(4) dimensions(3) location(3) rotation_y [score]

        Args:
            frame_id: Frame identifier.

        Returns:
            Tuple of:
                boxes_cam: (N, 7) array [x, y, z, h, w, l, ry] in camera frame.
                names: (N,) array of class name strings.
        """
        label_path = self.root_dir / "label_2" / f"{frame_id}.txt"
        boxes_list: List[np.ndarray] = []
        names_list: List[str] = []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                obj_type = parts[0]
                if obj_type == "DontCare":
                    continue
                # Filter by class names
                if obj_type not in self.class_names:
                    continue

                # Parse fields
                # truncated = float(parts[1])
                # occluded = int(parts[2])
                # alpha = float(parts[3])
                # bbox2d = [float(parts[i]) for i in range(4, 8)]
                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])  # noqa: E741
                x = float(parts[11])
                y = float(parts[12])
                z = float(parts[13])
                ry = float(parts[14])

                boxes_list.append(np.array([x, y, z, h, w, l, ry], dtype=np.float32))
                names_list.append(obj_type)

        if len(boxes_list) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.array([], dtype="<U16")

        boxes_cam = np.stack(boxes_list, axis=0)
        names = np.array(names_list)
        return boxes_cam, names

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve a single training sample.

        Returns:
            Dict with keys:
                'points': (N, 4) float32 tensor of point cloud.
                'gt_boxes': (M, 7) float32 tensor of ground-truth boxes in LiDAR frame.
                'gt_names': list of M class name strings.
                'frame_id': str frame identifier.
        """
        frame_id = self.frame_ids[index]

        # Load data
        points = self._load_point_cloud(frame_id)
        calib = self._load_calibration(frame_id)

        # Load and convert labels
        boxes_cam, gt_names = self._load_labels(frame_id)
        if boxes_cam.shape[0] > 0:
            gt_boxes = boxes_camera_to_lidar(
                boxes_cam, calib["R0_rect"], calib["Tr_velo_to_cam"]
            )
        else:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)

        # Filter points by range
        points = filter_points_by_range(points, self.point_cloud_range)

        # Filter boxes by range
        if gt_boxes.shape[0] > 0:
            range_mask = filter_boxes_by_range(gt_boxes, self.point_cloud_range)
            gt_boxes = gt_boxes[range_mask]
            gt_names = gt_names[range_mask]

        # Data augmentation
        if self.augment:
            # GT database sampling
            if self.db_infos and self.sample_groups:
                points, gt_boxes, gt_names = gt_database_sampling(
                    points, gt_boxes, gt_names, self.db_infos, self.sample_groups
                )

            # Random flip
            points, gt_boxes = random_flip(points, gt_boxes, along_x=True, along_y=False)

            # Random global rotation
            points, gt_boxes = random_global_rotation(
                points, gt_boxes, rotation_range=(-np.pi / 4, np.pi / 4)
            )

            # Random global scaling
            points, gt_boxes = random_global_scaling(
                points, gt_boxes, scale_range=(0.95, 1.05)
            )

            # Re-filter after augmentation
            points = filter_points_by_range(points, self.point_cloud_range)
            if gt_boxes.shape[0] > 0:
                range_mask = filter_boxes_by_range(gt_boxes, self.point_cloud_range)
                gt_boxes = gt_boxes[range_mask]
                gt_names = gt_names[range_mask]

        return {
            "points": torch.from_numpy(points).float(),
            "gt_boxes": torch.from_numpy(gt_boxes).float(),
            "gt_names": gt_names.tolist() if isinstance(gt_names, np.ndarray) else list(gt_names),
            "frame_id": frame_id,
        }


# ---------------------------------------------------------------------------
# nuScenes Dataset
# ---------------------------------------------------------------------------


class NuScenesDataset(Dataset):
    """PyTorch Dataset for the nuScenes 3D object detection benchmark.

    Expects a pre-processed info pickle file (generated by standard nuScenes
    data-preparation scripts) containing per-sample metadata including paths
    to .bin point cloud files and annotation information.

    Each info dict should contain:
        - 'lidar_path': str, path to the .bin point cloud file
        - 'token': str, unique sample token
        - 'gt_boxes': (M, 7) np.ndarray of boxes in LiDAR frame
        - 'gt_names': list of M class name strings

    nuScenes .bin format: (N, 5) float32 -- (x, y, z, intensity, ring_index)

    Args:
        info_path: Path to the pre-processed info pickle file.
        class_names: List of class names to retain.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        augment: Whether to apply data augmentation.
        max_sweeps: Number of LiDAR sweeps to aggregate (1 = keyframe only).
        db_infos_path: Optional path to GT database pickle.
        sample_groups: Class-to-count mapping for GT database sampling.
    """

    def __init__(
        self,
        info_path: str,
        class_names: Optional[List[str]] = None,
        point_cloud_range: Optional[List[float]] = None,
        augment: bool = False,
        max_sweeps: int = 10,
        db_infos_path: Optional[str] = None,
        sample_groups: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.info_path = Path(info_path)
        self.class_names = class_names if class_names is not None else [
            "car", "truck", "construction_vehicle", "bus", "trailer",
            "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
        ]
        self.point_cloud_range = point_cloud_range if point_cloud_range is not None else [
            -51.2, -51.2, -5.0, 51.2, 51.2, 3.0
        ]
        self.augment = augment
        self.max_sweeps = max_sweeps
        self.sample_groups = sample_groups if sample_groups is not None else {}

        # Load info list
        with open(str(self.info_path), "rb") as f:
            self.infos: List[Dict[str, Any]] = pickle.load(f)

        # Load GT database if specified
        self.db_infos: Dict[str, List[Dict[str, Any]]] = {}
        if db_infos_path is not None and Path(db_infos_path).exists():
            with open(db_infos_path, "rb") as f:
                self.db_infos = pickle.load(f)

    def __len__(self) -> int:
        return len(self.infos)

    def _load_point_cloud(self, info: Dict[str, Any]) -> np.ndarray:
        """Load and aggregate point cloud sweeps for a nuScenes sample.

        The keyframe point cloud has 5 channels (x, y, z, intensity, ring_index).
        Additional sweeps are transformed to the keyframe coordinate system
        using the provided sensor transforms, then concatenated. The ring_index
        channel is preserved.

        Args:
            info: Sample info dict containing 'lidar_path' and optionally 'sweeps'.

        Returns:
            (N, 5) float32 array.
        """
        # Load keyframe
        lidar_path = info["lidar_path"]
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

        # Aggregate sweeps if available
        if "sweeps" in info and self.max_sweeps > 1:
            sweep_list = info["sweeps"]
            num_sweeps = min(len(sweep_list), self.max_sweeps - 1)
            if num_sweeps > 0:
                # Randomly sample sweeps for diversity
                chosen_indices = np.random.choice(len(sweep_list), size=num_sweeps, replace=False)
                for sw_idx in chosen_indices:
                    sweep = sweep_list[sw_idx]
                    sweep_points = np.fromfile(sweep["lidar_path"], dtype=np.float32).reshape(-1, 5)

                    # Transform sweep points to keyframe coordinates
                    if "transform_matrix" in sweep:
                        transform = np.array(sweep["transform_matrix"], dtype=np.float64)  # (4, 4)
                        xyz = sweep_points[:, :3]
                        ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
                        xyz_hom = np.concatenate([xyz.astype(np.float64), ones], axis=1)  # (N, 4)
                        xyz_transformed = (transform @ xyz_hom.T).T[:, :3]  # (N, 3)
                        sweep_points = sweep_points.copy()
                        sweep_points[:, :3] = xyz_transformed.astype(np.float32)

                    points = np.concatenate([points, sweep_points], axis=0)

        return points

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve a single training sample.

        Returns:
            Dict with keys:
                'points': (N, 5) float32 tensor.
                'gt_boxes': (M, 7) float32 tensor.
                'gt_names': list of M class name strings.
                'frame_id': str sample token.
        """
        info = copy.deepcopy(self.infos[index])

        # Load point cloud
        points = self._load_point_cloud(info)

        # Load annotations
        gt_boxes = np.array(info.get("gt_boxes", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
        gt_names_raw = info.get("gt_names", [])
        gt_names = np.array(gt_names_raw) if len(gt_names_raw) > 0 else np.array([], dtype="<U32")

        # Filter by class name
        if gt_boxes.shape[0] > 0 and gt_names.shape[0] > 0:
            class_mask = np.array([name in self.class_names for name in gt_names], dtype=bool)
            gt_boxes = gt_boxes[class_mask]
            gt_names = gt_names[class_mask]

        # Filter points by range
        points = filter_points_by_range(points, self.point_cloud_range)

        # Filter boxes by range
        if gt_boxes.shape[0] > 0:
            range_mask = filter_boxes_by_range(gt_boxes, self.point_cloud_range)
            gt_boxes = gt_boxes[range_mask]
            gt_names = gt_names[range_mask]

        # Data augmentation
        if self.augment:
            # GT database sampling
            if self.db_infos and self.sample_groups:
                points, gt_boxes, gt_names = gt_database_sampling(
                    points, gt_boxes, gt_names, self.db_infos, self.sample_groups
                )

            # Random flip
            points, gt_boxes = random_flip(points, gt_boxes, along_x=True, along_y=True)

            # Random global rotation
            points, gt_boxes = random_global_rotation(
                points, gt_boxes, rotation_range=(-np.pi / 4, np.pi / 4)
            )

            # Random global scaling
            points, gt_boxes = random_global_scaling(
                points, gt_boxes, scale_range=(0.95, 1.05)
            )

            # Re-filter after augmentation
            points = filter_points_by_range(points, self.point_cloud_range)
            if gt_boxes.shape[0] > 0:
                range_mask = filter_boxes_by_range(gt_boxes, self.point_cloud_range)
                gt_boxes = gt_boxes[range_mask]
                gt_names = gt_names[range_mask]

        # Determine frame ID
        frame_id = info.get("token", info.get("frame_id", str(index)))

        return {
            "points": torch.from_numpy(points).float(),
            "gt_boxes": torch.from_numpy(gt_boxes).float(),
            "gt_names": gt_names.tolist() if isinstance(gt_names, np.ndarray) else list(gt_names),
            "frame_id": frame_id,
        }


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for DataLoader with variable-size point clouds.

    Handles variable numbers of points per sample by concatenating all point
    clouds and prepending a batch index column.  Ground-truth boxes are padded
    to the maximum count in the batch.

    Args:
        batch: List of sample dicts from Dataset.__getitem__.

    Returns:
        Dict with:
            'points': (sum(N_i), C+1) float32 tensor with batch index in column 0.
            'gt_boxes': (B, M_max, 7) float32 tensor, zero-padded.
            'gt_names': list of B lists of names.
            'frame_ids': list of B frame id strings.
            'batch_size': int.
    """
    batch_size = len(batch)
    points_list: List[torch.Tensor] = []
    gt_boxes_list: List[torch.Tensor] = []
    gt_names_list: List[List[str]] = []
    frame_ids: List[str] = []

    for i, sample in enumerate(batch):
        # Add batch index as column 0
        pts = sample["points"]  # (N_i, C)
        batch_idx = torch.full((pts.shape[0], 1), fill_value=i, dtype=torch.float32)
        pts_with_batch = torch.cat([batch_idx, pts], dim=1)  # (N_i, C+1)
        points_list.append(pts_with_batch)

        gt_boxes_list.append(sample["gt_boxes"])
        gt_names_list.append(sample["gt_names"])
        frame_ids.append(sample["frame_id"])

    # Concatenate all points
    all_points = torch.cat(points_list, dim=0)  # (sum(N_i), C+1)

    # Pad gt_boxes to max count
    max_num_boxes = max(boxes.shape[0] for boxes in gt_boxes_list) if gt_boxes_list else 0
    if max_num_boxes == 0:
        padded_gt_boxes = torch.zeros((batch_size, 0, 7), dtype=torch.float32)
    else:
        padded_gt_boxes = torch.zeros((batch_size, max_num_boxes, 7), dtype=torch.float32)
        for i, boxes in enumerate(gt_boxes_list):
            if boxes.shape[0] > 0:
                padded_gt_boxes[i, : boxes.shape[0], :] = boxes

    return {
        "points": all_points,
        "gt_boxes": padded_gt_boxes,
        "gt_names": gt_names_list,
        "frame_ids": frame_ids,
        "batch_size": batch_size,
    }
