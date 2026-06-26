"""
Dataset classes for KITTI and nuScenes point cloud data.

Provides data loading, label parsing, augmentation, and batching
utilities for 3D object detection training.
"""

import os
import math
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset


# ============================================================================
# Augmentation Functions
# ============================================================================


def random_rotate_along_z(
    points: np.ndarray, angle_range: float = math.pi / 4
) -> Tuple[np.ndarray, float]:
    """
    Randomly rotate point cloud around the Z (up) axis.

    Args:
        points: Point cloud, shape (N, 3+)
        angle_range: Maximum rotation angle in radians (symmetric range)

    Returns:
        Rotated points and the rotation angle applied
    """
    angle = np.random.uniform(-angle_range, angle_range)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rotation_matrix = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    points_rotated = points.copy()
    points_rotated[:, :3] = points[:, :3] @ rotation_matrix.T

    return points_rotated, angle


def random_flip_along_x(
    points: np.ndarray, probability: float = 0.5
) -> Tuple[np.ndarray, bool]:
    """
    Randomly flip the point cloud along the X axis (mirror Y coordinates).

    Args:
        points: Point cloud, shape (N, 3+)
        probability: Probability of flipping

    Returns:
        (Possibly flipped) points and whether flip was applied
    """
    flipped = np.random.random() < probability
    if flipped:
        points_out = points.copy()
        points_out[:, 1] = -points_out[:, 1]
        return points_out, True
    return points, False


def random_scale_points(
    points: np.ndarray, scale_range: Tuple[float, float] = (0.95, 1.05)
) -> Tuple[np.ndarray, float]:
    """
    Randomly scale the point cloud uniformly.

    Args:
        points: Point cloud, shape (N, 3+)
        scale_range: (min_scale, max_scale) tuple

    Returns:
        Scaled points and the scale factor applied
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    points_scaled = points.copy()
    points_scaled[:, :3] *= scale
    return points_scaled, scale


def random_jitter_points(
    points: np.ndarray, sigma: float = 0.01, clip: float = 0.05
) -> np.ndarray:
    """
    Add random Gaussian noise to point coordinates.

    Args:
        points: Point cloud, shape (N, 3+)
        sigma: Standard deviation of the noise
        clip: Maximum absolute noise value (clipped)

    Returns:
        Jittered points
    """
    noise = np.clip(np.random.randn(points.shape[0], 3) * sigma, -clip, clip)
    points_jittered = points.copy()
    points_jittered[:, :3] += noise.astype(np.float32)
    return points_jittered


def augment_labels_rotation(
    labels: List[Dict], angle: float
) -> List[Dict]:
    """Apply rotation augmentation to label boxes."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    augmented = []
    for label in labels:
        new_label = label.copy()
        x, y, z = label["center"]
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        new_label["center"] = [new_x, new_y, z]
        new_label["yaw"] = label["yaw"] + angle
        augmented.append(new_label)
    return augmented


def augment_labels_flip_x(labels: List[Dict]) -> List[Dict]:
    """Apply X-flip augmentation to label boxes."""
    augmented = []
    for label in labels:
        new_label = label.copy()
        x, y, z = label["center"]
        new_label["center"] = [x, -y, z]
        new_label["yaw"] = -label["yaw"]
        augmented.append(new_label)
    return augmented


def augment_labels_scale(labels: List[Dict], scale: float) -> List[Dict]:
    """Apply scale augmentation to label boxes."""
    augmented = []
    for label in labels:
        new_label = label.copy()
        x, y, z = label["center"]
        new_label["center"] = [x * scale, y * scale, z * scale]
        w, h, l = label["size"]
        new_label["size"] = [w * scale, h * scale, l * scale]
        augmented.append(new_label)
    return augmented


# ============================================================================
# KITTI Dataset
# ============================================================================


KITTI_CLASSES = ["Car", "Pedestrian", "Cyclist", "Van", "Truck", "Person_sitting",
                 "Tram", "Misc", "DontCare"]
KITTI_CLASS_TO_IDX = {cls: i for i, cls in enumerate(KITTI_CLASSES)}


class KITTIDataset(Dataset):
    """
    KITTI 3D Object Detection dataset.

    Loads .bin point cloud files and .txt label files from the KITTI format.

    Directory structure expected:
        root/
            velodyne/       # .bin point cloud files (N x 4 float32: x,y,z,intensity)
            label_2/        # .txt label files

    Args:
        root: Path to the dataset root directory
        split: 'train' or 'val' (uses split file if available, else all files)
        npoints: Number of points to subsample to (default 16384)
        augment: Whether to apply data augmentation
        classes: List of class names to use (default: Car, Pedestrian, Cyclist)
        point_range: Optional [xmin, ymin, zmin, xmax, ymax, zmax] to filter
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        npoints: int = 16384,
        augment: bool = True,
        classes: Optional[List[str]] = None,
        point_range: Optional[List[float]] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.npoints = npoints
        self.augment = augment and (split == "train")
        self.classes = classes or ["Car", "Pedestrian", "Cyclist"]
        self.class_to_idx = {cls: i + 1 for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes) + 1  # +1 for background

        # Default KITTI LiDAR range
        if point_range is None:
            self.point_range = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]
        else:
            self.point_range = point_range

        # Find all sample indices
        velodyne_dir = self.root / "velodyne"
        label_dir = self.root / "label_2"

        # Try to load split file
        split_file = self.root / f"{split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                self.sample_ids = [line.strip() for line in f.readlines()]
        else:
            # Use all files in velodyne directory
            if velodyne_dir.exists():
                self.sample_ids = sorted(
                    [f.stem for f in velodyne_dir.glob("*.bin")]
                )
            else:
                self.sample_ids = []

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.sample_ids[idx]

        # Load point cloud (.bin: N x 4 float32)
        pc_path = self.root / "velodyne" / f"{sample_id}.bin"
        points = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)

        # Load labels
        label_path = self.root / "label_2" / f"{sample_id}.txt"
        labels = self._parse_kitti_labels(label_path)

        # Filter points by range
        points = self._filter_points_by_range(points)

        # Apply augmentations
        if self.augment:
            points, labels = self._apply_augmentations(points, labels)

        # Subsample to fixed number of points
        points = self._subsample(points)

        # Convert to tensors
        xyz = torch.from_numpy(points[:, :3]).float()
        features = torch.from_numpy(points[:, 3:]).float()  # intensity

        # Encode labels as targets
        targets = self._encode_targets(labels, xyz)

        return {
            "xyz": xyz,                  # (npoints, 3)
            "features": features,        # (npoints, 1)
            "targets": targets,
            "sample_id": sample_id,
        }

    def _parse_kitti_labels(self, label_path: Path) -> List[Dict]:
        """Parse a KITTI label .txt file."""
        labels = []
        if not label_path.exists():
            return labels

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                cls_name = parts[0]
                if cls_name not in self.classes:
                    continue

                # KITTI format: type truncated occluded alpha
                # bbox2d(4) dimensions(3: h,w,l) location(3: x,y,z) rotation_y
                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])
                x = float(parts[11])
                y = float(parts[12])
                z = float(parts[13])
                yaw = float(parts[14])

                # KITTI uses camera coordinates; convert to LiDAR frame:
                # cam_x -> lidar_z, cam_y -> -lidar_x, cam_z -> -lidar_y
                # For simplicity, we store in the coordinate system of the
                # point cloud directly
                labels.append({
                    "class": cls_name,
                    "class_idx": self.class_to_idx[cls_name],
                    "center": [x, y, z - h / 2.0],  # move to box center
                    "size": [w, l, h],  # width, length, height
                    "yaw": yaw,
                })

        return labels

    def _filter_points_by_range(self, points: np.ndarray) -> np.ndarray:
        """Filter points outside the specified spatial range."""
        xmin, ymin, zmin, xmax, ymax, zmax = self.point_range
        mask = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
            & (points[:, 2] >= zmin)
            & (points[:, 2] <= zmax)
        )
        return points[mask]

    def _subsample(self, points: np.ndarray) -> np.ndarray:
        """Subsample or pad to a fixed number of points."""
        n = points.shape[0]
        if n == 0:
            # Return zeros if no points after filtering
            return np.zeros((self.npoints, points.shape[1]), dtype=np.float32)

        if n >= self.npoints:
            # Random subsample
            indices = np.random.choice(n, self.npoints, replace=False)
        else:
            # Pad by repeating random points
            indices = np.concatenate([
                np.arange(n),
                np.random.choice(n, self.npoints - n, replace=True),
            ])
        return points[indices]

    def _apply_augmentations(
        self, points: np.ndarray, labels: List[Dict]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Apply random augmentations to points and labels."""
        # Random rotation around Z
        points, angle = random_rotate_along_z(points, angle_range=math.pi / 4)
        labels = augment_labels_rotation(labels, angle)

        # Random flip along X
        points, flipped = random_flip_along_x(points)
        if flipped:
            labels = augment_labels_flip_x(labels)

        # Random scale
        points, scale = random_scale_points(points, scale_range=(0.95, 1.05))
        labels = augment_labels_scale(labels, scale)

        # Random jitter (points only, not labels)
        points = random_jitter_points(points, sigma=0.01, clip=0.05)

        return points, labels

    def _encode_targets(
        self, labels: List[Dict], xyz: torch.Tensor
    ) -> Dict:
        """
        Encode labels as per-point targets for detection.

        For each point, assigns it to the nearest GT box (if within the box).
        """
        N = xyz.shape[0]
        # Default: all background
        cls_targets = torch.zeros(N, dtype=torch.long)
        center_targets = torch.zeros(N, 3, dtype=torch.float32)
        size_targets = torch.zeros(N, 3, dtype=torch.float32)
        angle_targets = torch.zeros(N, 1, dtype=torch.float32)
        mask = torch.zeros(N, dtype=torch.float32)

        for label in labels:
            cx, cy, cz = label["center"]
            w, l, h = label["size"]

            # Simple box membership: check if point is within the box
            # (axis-aligned approximation for assignment)
            in_box = (
                (xyz[:, 0] >= cx - w / 2)
                & (xyz[:, 0] <= cx + w / 2)
                & (xyz[:, 1] >= cy - l / 2)
                & (xyz[:, 1] <= cy + l / 2)
                & (xyz[:, 2] >= cz - h / 2)
                & (xyz[:, 2] <= cz + h / 2)
            )

            cls_targets[in_box] = label["class_idx"]
            # Center offset from point to box center
            center_targets[in_box, 0] = cx - xyz[in_box, 0]
            center_targets[in_box, 1] = cy - xyz[in_box, 1]
            center_targets[in_box, 2] = cz - xyz[in_box, 2]
            size_targets[in_box, 0] = w
            size_targets[in_box, 1] = l
            size_targets[in_box, 2] = h
            angle_targets[in_box, 0] = label["yaw"]
            mask[in_box] = 1.0

        return {
            "cls": cls_targets,
            "center": center_targets,
            "size": size_targets,
            "angle": angle_targets,
            "mask": mask,
        }


# ============================================================================
# nuScenes Dataset
# ============================================================================


NUSCENES_CLASSES = [
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
]
NUSCENES_CLASS_TO_IDX = {cls: i for i, cls in enumerate(NUSCENES_CLASSES)}


class NuScenesDataset(Dataset):
    """
    nuScenes 3D Object Detection dataset.

    Loads point cloud files and annotations in nuScenes format.
    Point clouds are expected as .bin files (N x 5 float32: x,y,z,intensity,ring).

    Directory structure expected:
        root/
            samples/
                LIDAR_TOP/     # .bin or .pcd.bin files
            v1.0-trainval/     # or v1.0-mini
                sample_data.json
                sample_annotation.json
                ...

    For simplicity, this implementation expects pre-extracted data:
        root/
            points/            # .bin files (N x 5 float32)
            labels/            # .txt files with box annotations
            split/
                train.txt
                val.txt

    Args:
        root: Path to dataset root
        split: 'train' or 'val'
        npoints: Number of points to subsample to
        augment: Whether to apply augmentation
        classes: List of class names
        point_range: Spatial filter [xmin, ymin, zmin, xmax, ymax, zmax]
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        npoints: int = 32768,
        augment: bool = True,
        classes: Optional[List[str]] = None,
        point_range: Optional[List[float]] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.npoints = npoints
        self.augment = augment and (split == "train")
        self.classes = classes or NUSCENES_CLASSES
        self.class_to_idx = {cls: i + 1 for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes) + 1  # +1 for background

        # nuScenes has a larger range than KITTI
        if point_range is None:
            self.point_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        else:
            self.point_range = point_range

        # Load sample IDs from split file
        split_file = self.root / "split" / f"{split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                self.sample_ids = [line.strip() for line in f.readlines()]
        else:
            points_dir = self.root / "points"
            if points_dir.exists():
                self.sample_ids = sorted([f.stem for f in points_dir.glob("*.bin")])
            else:
                self.sample_ids = []

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.sample_ids[idx]

        # Load point cloud (N x 5: x, y, z, intensity, ring_index)
        pc_path = self.root / "points" / f"{sample_id}.bin"
        points = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 5)

        # Load labels
        label_path = self.root / "labels" / f"{sample_id}.txt"
        labels = self._parse_nuscenes_labels(label_path)

        # Filter by range
        points = self._filter_points_by_range(points)

        # Augmentation
        if self.augment:
            points, labels = self._apply_augmentations(points, labels)

        # Subsample
        points = self._subsample(points)

        # Convert to tensors
        xyz = torch.from_numpy(points[:, :3]).float()
        features = torch.from_numpy(points[:, 3:5]).float()  # intensity + ring

        targets = self._encode_targets(labels, xyz)

        return {
            "xyz": xyz,                  # (npoints, 3)
            "features": features,        # (npoints, 2)
            "targets": targets,
            "sample_id": sample_id,
        }

    def _parse_nuscenes_labels(self, label_path: Path) -> List[Dict]:
        """
        Parse nuScenes-style label file.

        Expected format per line:
            class_name x y z w l h yaw
        """
        labels = []
        if not label_path.exists():
            return labels

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 8:
                    continue

                cls_name = parts[0]
                if cls_name not in self.classes:
                    continue

                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                w = float(parts[4])
                l = float(parts[5])
                h = float(parts[6])
                yaw = float(parts[7])

                labels.append({
                    "class": cls_name,
                    "class_idx": self.class_to_idx[cls_name],
                    "center": [x, y, z],
                    "size": [w, l, h],
                    "yaw": yaw,
                })

        return labels

    def _filter_points_by_range(self, points: np.ndarray) -> np.ndarray:
        """Filter points by spatial range."""
        xmin, ymin, zmin, xmax, ymax, zmax = self.point_range
        mask = (
            (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
            & (points[:, 2] >= zmin)
            & (points[:, 2] <= zmax)
        )
        return points[mask]

    def _subsample(self, points: np.ndarray) -> np.ndarray:
        """Subsample or pad point cloud to fixed size."""
        n = points.shape[0]
        if n == 0:
            return np.zeros((self.npoints, points.shape[1]), dtype=np.float32)

        if n >= self.npoints:
            indices = np.random.choice(n, self.npoints, replace=False)
        else:
            indices = np.concatenate([
                np.arange(n),
                np.random.choice(n, self.npoints - n, replace=True),
            ])
        return points[indices]

    def _apply_augmentations(
        self, points: np.ndarray, labels: List[Dict]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Apply augmentations."""
        points, angle = random_rotate_along_z(points, angle_range=math.pi / 4)
        labels = augment_labels_rotation(labels, angle)

        points, flipped = random_flip_along_x(points)
        if flipped:
            labels = augment_labels_flip_x(labels)

        points, scale = random_scale_points(points, scale_range=(0.95, 1.05))
        labels = augment_labels_scale(labels, scale)

        points = random_jitter_points(points, sigma=0.01, clip=0.05)

        return points, labels

    def _encode_targets(
        self, labels: List[Dict], xyz: torch.Tensor
    ) -> Dict:
        """Encode labels as per-point targets."""
        N = xyz.shape[0]
        cls_targets = torch.zeros(N, dtype=torch.long)
        center_targets = torch.zeros(N, 3, dtype=torch.float32)
        size_targets = torch.zeros(N, 3, dtype=torch.float32)
        angle_targets = torch.zeros(N, 1, dtype=torch.float32)
        mask = torch.zeros(N, dtype=torch.float32)

        for label in labels:
            cx, cy, cz = label["center"]
            w, l, h = label["size"]

            in_box = (
                (xyz[:, 0] >= cx - w / 2)
                & (xyz[:, 0] <= cx + w / 2)
                & (xyz[:, 1] >= cy - l / 2)
                & (xyz[:, 1] <= cy + l / 2)
                & (xyz[:, 2] >= cz - h / 2)
                & (xyz[:, 2] <= cz + h / 2)
            )

            cls_targets[in_box] = label["class_idx"]
            center_targets[in_box, 0] = cx - xyz[in_box, 0]
            center_targets[in_box, 1] = cy - xyz[in_box, 1]
            center_targets[in_box, 2] = cz - xyz[in_box, 2]
            size_targets[in_box, 0] = w
            size_targets[in_box, 1] = l
            size_targets[in_box, 2] = h
            angle_targets[in_box, 0] = label["yaw"]
            mask[in_box] = 1.0

        return {
            "cls": cls_targets,
            "center": center_targets,
            "size": size_targets,
            "angle": angle_targets,
            "mask": mask,
        }


# ============================================================================
# Collate Function
# ============================================================================


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-size point clouds.

    Pads point clouds to the maximum size in the batch and creates
    a validity mask.

    Args:
        batch: List of sample dictionaries from the dataset

    Returns:
        Batched dictionary with padded tensors
    """
    # If all samples already have the same size (fixed npoints), standard stack works
    batch_size = len(batch)

    # Check if xyz shapes are uniform
    xyz_shapes = [sample["xyz"].shape[0] for sample in batch]
    max_npoints = max(xyz_shapes)

    # Determine feature dimension
    feat_dim = batch[0]["features"].shape[1] if batch[0]["features"].dim() > 1 else 1

    # Allocate padded tensors
    xyz_batch = torch.zeros(batch_size, max_npoints, 3)
    features_batch = torch.zeros(batch_size, max_npoints, feat_dim)
    point_mask = torch.zeros(batch_size, max_npoints, dtype=torch.bool)

    # Target tensors
    cls_batch = torch.zeros(batch_size, max_npoints, dtype=torch.long)
    center_batch = torch.zeros(batch_size, max_npoints, 3)
    size_batch = torch.zeros(batch_size, max_npoints, 3)
    angle_batch = torch.zeros(batch_size, max_npoints, 1)
    target_mask_batch = torch.zeros(batch_size, max_npoints)

    sample_ids = []

    for i, sample in enumerate(batch):
        n = sample["xyz"].shape[0]
        xyz_batch[i, :n] = sample["xyz"]
        if sample["features"].dim() == 1:
            features_batch[i, :n, 0] = sample["features"]
        else:
            features_batch[i, :n] = sample["features"]
        point_mask[i, :n] = True

        targets = sample["targets"]
        cls_batch[i, :n] = targets["cls"]
        center_batch[i, :n] = targets["center"]
        size_batch[i, :n] = targets["size"]
        angle_batch[i, :n] = targets["angle"]
        target_mask_batch[i, :n] = targets["mask"]

        sample_ids.append(sample["sample_id"])

    return {
        "xyz": xyz_batch,
        "features": features_batch,
        "point_mask": point_mask,
        "targets": {
            "cls": cls_batch,
            "center": center_batch,
            "size": size_batch,
            "angle": angle_batch,
            "mask": target_mask_batch,
        },
        "sample_ids": sample_ids,
    }
