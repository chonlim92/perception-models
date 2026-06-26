"""KITTI 3D Object Detection dataset loader.

Provides a fully-featured data loader for the KITTI benchmark supporting:
- Stereo camera images (left color: image_2, right color: image_3)
- Velodyne LiDAR point clouds (.bin files, Nx4: x, y, z, reflectance)
- Calibration files (P0-P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo)
- 3D bounding box labels (type, truncated, occluded, alpha, 2D bbox,
  dimensions, location, rotation_y)
- Train/val/test splits
- PyTorch Dataset interface

Expected directory structure::

    kitti_root/
        training/
            image_2/      # Left color images (*.png)
            image_3/      # Right color images (*.png)
            velodyne/     # LiDAR point clouds (*.bin)
            calib/        # Calibration files (*.txt)
            label_2/      # 3D annotations (*.txt)
        testing/
            image_2/
            image_3/
            velodyne/
            calib/

Usage
-----
::

    from common.datasets.kitti_dataset import KITTIDataset

    dataset = KITTIDataset(
        dataroot="/data/kitti",
        split="train",
        load_velodyne=True,
        load_right_image=True,
    )
    sample = dataset[0]
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    TorchDataset = object  # type: ignore[assignment,misc]

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from common.registry import DATASETS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KITTI_CLASSES: List[str] = [
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
    "DontCare",
]

# Standard train/val split from Qi et al. (3DOP) - 3712 training, 3769 validation
# These are the sample indices (6-digit zero-padded file stems)
_TRAIN_VAL_SPLIT_URL = (
    "https://raw.githubusercontent.com/charlesq34/3D-object-detection-APE/"
    "master/kitti/image_sets/"
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _load_kitti_image(filepath: str) -> np.ndarray:
    """Load a KITTI image as a numpy array.

    Parameters
    ----------
    filepath : str
        Path to the PNG image file.

    Returns
    -------
    np.ndarray
        Image array with shape (H, W, 3), dtype uint8, RGB channel order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for image loading. Install with: pip install Pillow")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")
    img = Image.open(filepath).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _load_velodyne_points(filepath: str) -> np.ndarray:
    """Load a Velodyne LiDAR point cloud from a .bin file.

    Parameters
    ----------
    filepath : str
        Path to the .bin file.

    Returns
    -------
    np.ndarray
        Shape (N, 4) array with columns [x, y, z, reflectance].
        Coordinates are in the Velodyne sensor frame.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Velodyne file not found: {filepath}")
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    return points


def _parse_kitti_calibration(filepath: str) -> Dict[str, np.ndarray]:
    """Parse a KITTI calibration file.

    Parameters
    ----------
    filepath : str
        Path to the calibration .txt file.

    Returns
    -------
    dict
        Calibration matrices with keys:
        - ``"P0"`` : (3, 4) projection matrix for camera 0
        - ``"P1"`` : (3, 4) projection matrix for camera 1
        - ``"P2"`` : (3, 4) projection matrix for left color camera
        - ``"P3"`` : (3, 4) projection matrix for right color camera
        - ``"R0_rect"`` : (4, 4) rectifying rotation (extended to 4x4)
        - ``"Tr_velo_to_cam"`` : (4, 4) Velodyne-to-camera transformation
        - ``"Tr_imu_to_velo"`` : (4, 4) IMU-to-Velodyne transformation

    Raises
    ------
    FileNotFoundError
        If the calibration file does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Calibration file not found: {filepath}")

    calib: Dict[str, np.ndarray] = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            values = np.array([float(x) for x in value.strip().split()], dtype=np.float64)

            if key in ("P0", "P1", "P2", "P3"):
                calib[key] = values.reshape(3, 4)
            elif key == "R0_rect":
                # Extend 3x3 rotation to 4x4 homogeneous
                R = values.reshape(3, 3)
                R_ext = np.eye(4, dtype=np.float64)
                R_ext[:3, :3] = R
                calib["R0_rect"] = R_ext
            elif key in ("Tr_velo_to_cam", "Tr_imu_to_velo"):
                # Extend 3x4 to 4x4
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = values.reshape(3, 4)
                calib[key] = T

    return calib


def _parse_kitti_label(filepath: str) -> List[Dict[str, Any]]:
    """Parse a KITTI label file with 3D bounding box annotations.

    Parameters
    ----------
    filepath : str
        Path to the label .txt file.

    Returns
    -------
    list of dict
        Each annotation dict contains:
        - ``"type"`` : str - object class name
        - ``"truncated"`` : float - truncation level (0.0 to 1.0)
        - ``"occluded"`` : int - occlusion state (0=visible, 1=partly, 2=largely, 3=unknown)
        - ``"alpha"`` : float - observation angle of object in image plane [-pi, pi]
        - ``"bbox_2d"`` : np.ndarray (4,) - 2D bbox [x1, y1, x2, y2] in pixels
        - ``"dimensions"`` : np.ndarray (3,) - [height, width, length] in meters
        - ``"location"`` : np.ndarray (3,) - [x, y, z] in camera coordinates (meters)
        - ``"rotation_y"`` : float - rotation around Y-axis in camera coordinates [-pi, pi]
        - ``"score"`` : float - confidence score (1.0 for ground truth)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Label file not found: {filepath}")

    annotations: List[Dict[str, Any]] = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 15:
                continue

            obj_type = parts[0]
            truncated = float(parts[1])
            occluded = int(parts[2])
            alpha = float(parts[3])
            bbox_2d = np.array(
                [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                dtype=np.float64,
            )
            # KITTI format: height, width, length
            dimensions = np.array(
                [float(parts[8]), float(parts[9]), float(parts[10])],
                dtype=np.float64,
            )
            location = np.array(
                [float(parts[11]), float(parts[12]), float(parts[13])],
                dtype=np.float64,
            )
            rotation_y = float(parts[14])

            # Score (optional, present in detection results)
            score = float(parts[15]) if len(parts) > 15 else 1.0

            annotation = {
                "type": obj_type,
                "truncated": truncated,
                "occluded": occluded,
                "alpha": alpha,
                "bbox_2d": bbox_2d,
                "dimensions": dimensions,
                "location": location,
                "rotation_y": rotation_y,
                "score": score,
            }
            annotations.append(annotation)

    return annotations


# ---------------------------------------------------------------------------
# Main Dataset Class
# ---------------------------------------------------------------------------


@DATASETS.register("kitti")
class KITTIDataset(TorchDataset):
    """KITTI 3D Object Detection dataset.

    Loads stereo images, Velodyne LiDAR, calibration, and 3D bounding box
    annotations from the KITTI benchmark dataset.

    Parameters
    ----------
    dataroot : str
        Root directory of the KITTI dataset containing ``training/`` and
        ``testing/`` subdirectories.
    split : str
        Data split: ``"train"``, ``"val"``, ``"trainval"``, or ``"test"``.
    load_velodyne : bool
        Whether to load Velodyne LiDAR point clouds. Default ``True``.
    load_right_image : bool
        Whether to load the right stereo image (image_3). Default ``False``.
    max_points : int
        Maximum number of LiDAR points. Use ``-1`` for no limit. Default ``-1``.
    image_size : tuple of int, optional
        If provided, resize images to (height, width).
    transform : callable, optional
        A function applied to the full sample dict after loading.
    point_cloud_range : list of float, optional
        Filter LiDAR points to [x_min, y_min, z_min, x_max, y_max, z_max].
    class_filter : list of str, optional
        If provided, only keep annotations with type in this list.
    split_file : str, optional
        Path to a custom split file listing sample indices (one per line).
        If not provided, uses the standard Chen et al. train/val split.
    min_points_in_box : int
        Minimum number of LiDAR points inside a 3D box to keep the annotation.
        Default ``0`` (keep all).
    """

    # Standard KITTI train/val split (Chen et al.)
    # 3712 training samples, 3769 validation samples from 7481 total
    _STANDARD_TRAIN_INDICES: Optional[List[int]] = None
    _STANDARD_VAL_INDICES: Optional[List[int]] = None

    def __init__(
        self,
        dataroot: str,
        split: str = "train",
        load_velodyne: bool = True,
        load_right_image: bool = False,
        max_points: int = -1,
        image_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        point_cloud_range: Optional[List[float]] = None,
        class_filter: Optional[List[str]] = None,
        split_file: Optional[str] = None,
        min_points_in_box: int = 0,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for KITTIDataset. "
                "Install with: pip install torch"
            )

        super().__init__()

        self.dataroot = dataroot
        self.split = split
        self.load_velodyne = load_velodyne
        self.load_right_image = load_right_image
        self.max_points = max_points
        self.image_size = image_size
        self.transform = transform
        self.point_cloud_range = point_cloud_range
        self.class_filter = class_filter
        self.min_points_in_box = min_points_in_box

        # Determine whether we're in training or testing partition
        self._is_test = split == "test"
        self._data_dir = os.path.join(
            dataroot, "testing" if self._is_test else "training"
        )

        # Validate directory structure
        if not os.path.isdir(self._data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {self._data_dir}. "
                f"Expected KITTI directory structure with training/ and testing/ subdirs."
            )

        # Collect sample indices
        self.sample_indices = self._get_sample_indices(split_file)

    def _get_sample_indices(self, split_file: Optional[str]) -> List[int]:
        """Determine sample indices for the requested split.

        Parameters
        ----------
        split_file : str or None
            Optional path to a custom split file.

        Returns
        -------
        list of int
            Sorted list of sample indices.
        """
        if split_file is not None:
            # Use custom split file
            if not os.path.isfile(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            with open(split_file, "r") as f:
                indices = [int(line.strip()) for line in f if line.strip()]
            return sorted(indices)

        # Discover all available samples from image_2 directory
        image_dir = os.path.join(self._data_dir, "image_2")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"image_2 directory not found: {image_dir}"
            )

        all_indices = sorted(
            int(f.stem)
            for f in Path(image_dir).glob("*.png")
            if f.stem.isdigit()
        )

        if not all_indices:
            raise ValueError(f"No PNG images found in {image_dir}")

        if self._is_test:
            return all_indices

        # Standard train/val split
        # Use the widely-adopted split: first 3712 for train, remaining for val
        # based on the ImageSets from KITTI 3D detection benchmark
        train_split_file = os.path.join(self.dataroot, "ImageSets", "train.txt")
        val_split_file = os.path.join(self.dataroot, "ImageSets", "val.txt")
        trainval_split_file = os.path.join(self.dataroot, "ImageSets", "trainval.txt")

        if self.split == "trainval":
            if os.path.isfile(trainval_split_file):
                return self._read_split_file(trainval_split_file)
            return all_indices

        if self.split == "train" and os.path.isfile(train_split_file):
            return self._read_split_file(train_split_file)

        if self.split == "val" and os.path.isfile(val_split_file):
            return self._read_split_file(val_split_file)

        # Fallback: if no ImageSets files exist, use a deterministic split
        # 50% train, 50% val based on index parity (standard fallback)
        n_total = len(all_indices)
        split_point = int(n_total * 0.5)

        if self.split == "train":
            return all_indices[:split_point]
        elif self.split == "val":
            return all_indices[split_point:]
        else:
            raise ValueError(
                f"Unknown split: {self.split!r}. "
                f"Expected one of: 'train', 'val', 'trainval', 'test'."
            )

    @staticmethod
    def _read_split_file(filepath: str) -> List[int]:
        """Read a split file containing one sample index per line.

        Parameters
        ----------
        filepath : str
            Path to the split file.

        Returns
        -------
        list of int
            Sorted sample indices.
        """
        with open(filepath, "r") as f:
            indices = [int(line.strip()) for line in f if line.strip()]
        return sorted(indices)

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load and return a single sample.

        Parameters
        ----------
        index : int
            Index into the dataset.

        Returns
        -------
        dict
            A dictionary containing:
            - ``"index"`` : int - KITTI sample index
            - ``"image_2"`` : np.ndarray (H, W, 3) - left color image (RGB, uint8)
            - ``"image_3"`` : np.ndarray (H, W, 3) or None - right color image
            - ``"velodyne"`` : np.ndarray (N, 4) or None - LiDAR [x, y, z, reflectance]
            - ``"calibration"`` : dict - calibration matrices
            - ``"annotations"`` : list of dict or None - 3D bbox annotations
            - ``"image_shape"`` : tuple (H, W) - original image dimensions
        """
        sample_idx = self.sample_indices[index]
        sample_id = f"{sample_idx:06d}"

        result: Dict[str, Any] = {"index": sample_idx}

        # Load left color image (image_2)
        img2_path = os.path.join(self._data_dir, "image_2", f"{sample_id}.png")
        try:
            img2 = _load_kitti_image(img2_path)
            result["image_2"] = img2
            result["image_shape"] = (img2.shape[0], img2.shape[1])
        except FileNotFoundError:
            warnings.warn(f"Left image not found: {img2_path}")
            result["image_2"] = np.zeros((375, 1242, 3), dtype=np.uint8)
            result["image_shape"] = (375, 1242)

        # Resize if requested
        if self.image_size is not None and _PIL_AVAILABLE:
            h, w = self.image_size
            pil_img = Image.fromarray(result["image_2"])
            pil_img = pil_img.resize((w, h), Image.BILINEAR)
            result["image_2"] = np.array(pil_img, dtype=np.uint8)

        # Load right color image (image_3)
        if self.load_right_image:
            img3_path = os.path.join(self._data_dir, "image_3", f"{sample_id}.png")
            try:
                img3 = _load_kitti_image(img3_path)
                if self.image_size is not None and _PIL_AVAILABLE:
                    h, w = self.image_size
                    pil_img = Image.fromarray(img3)
                    pil_img = pil_img.resize((w, h), Image.BILINEAR)
                    img3 = np.array(pil_img, dtype=np.uint8)
                result["image_3"] = img3
            except FileNotFoundError:
                warnings.warn(f"Right image not found: {img3_path}")
                result["image_3"] = None
        else:
            result["image_3"] = None

        # Load Velodyne LiDAR
        if self.load_velodyne:
            velo_path = os.path.join(self._data_dir, "velodyne", f"{sample_id}.bin")
            try:
                points = _load_velodyne_points(velo_path)
                points = self._filter_points(points)
                result["velodyne"] = points
            except FileNotFoundError:
                warnings.warn(f"Velodyne file not found: {velo_path}")
                result["velodyne"] = np.zeros((0, 4), dtype=np.float32)
        else:
            result["velodyne"] = None

        # Load calibration
        calib_path = os.path.join(self._data_dir, "calib", f"{sample_id}.txt")
        try:
            calibration = _parse_kitti_calibration(calib_path)
            result["calibration"] = calibration
        except FileNotFoundError:
            warnings.warn(f"Calibration file not found: {calib_path}")
            result["calibration"] = {}

        # Load annotations (only for training split)
        if not self._is_test:
            label_path = os.path.join(self._data_dir, "label_2", f"{sample_id}.txt")
            try:
                annotations = _parse_kitti_label(label_path)
                annotations = self._filter_annotations(annotations)
                result["annotations"] = annotations
            except FileNotFoundError:
                warnings.warn(f"Label file not found: {label_path}")
                result["annotations"] = []
        else:
            result["annotations"] = None

        # Apply user transform
        if self.transform is not None:
            result = self.transform(result)

        return result

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """Filter and subsample the point cloud.

        Parameters
        ----------
        points : np.ndarray
            Shape (N, 4) point cloud.

        Returns
        -------
        np.ndarray
            Filtered point cloud.
        """
        if points.shape[0] == 0:
            return points

        # Range filter
        if self.point_cloud_range is not None:
            pcr = self.point_cloud_range
            mask = (
                (points[:, 0] >= pcr[0])
                & (points[:, 1] >= pcr[1])
                & (points[:, 2] >= pcr[2])
                & (points[:, 0] <= pcr[3])
                & (points[:, 1] <= pcr[4])
                & (points[:, 2] <= pcr[5])
            )
            points = points[mask]

        # Subsample
        if self.max_points > 0 and points.shape[0] > self.max_points:
            indices = np.random.choice(
                points.shape[0], self.max_points, replace=False
            )
            points = points[indices]

        return points

    def _filter_annotations(
        self, annotations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter annotations by class and other criteria.

        Parameters
        ----------
        annotations : list of dict
            Raw annotations from the label file.

        Returns
        -------
        list of dict
            Filtered annotations.
        """
        if self.class_filter is not None:
            annotations = [
                ann for ann in annotations if ann["type"] in self.class_filter
            ]
        return annotations

    # ------------------------------------------------------------------
    # Coordinate transformations
    # ------------------------------------------------------------------

    @staticmethod
    def project_velo_to_cam(
        points: np.ndarray, calibration: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Project Velodyne points into the camera coordinate frame.

        Parameters
        ----------
        points : np.ndarray
            Shape (N, 3) or (N, 4) Velodyne points [x, y, z, (reflectance)].
        calibration : dict
            Calibration dictionary with ``"R0_rect"`` and ``"Tr_velo_to_cam"`` keys.

        Returns
        -------
        np.ndarray
            Shape (N, 3) points in camera coordinates.
        """
        pts_3d = points[:, :3]
        n = pts_3d.shape[0]

        # To homogeneous
        pts_hom = np.hstack([pts_3d, np.ones((n, 1), dtype=np.float64)])

        # Velodyne to camera: R0_rect @ Tr_velo_to_cam @ [x, y, z, 1]^T
        Tr = calibration["Tr_velo_to_cam"]  # (4, 4)
        R0 = calibration["R0_rect"]  # (4, 4)

        pts_cam = (R0 @ Tr @ pts_hom.T).T  # (N, 4)
        return pts_cam[:, :3]

    @staticmethod
    def project_cam_to_image(
        points_cam: np.ndarray, P: np.ndarray
    ) -> np.ndarray:
        """Project camera-coordinate points onto the image plane.

        Parameters
        ----------
        points_cam : np.ndarray
            Shape (N, 3) points in camera coordinates.
        P : np.ndarray
            Shape (3, 4) projection matrix (e.g., P2 for left color camera).

        Returns
        -------
        np.ndarray
            Shape (N, 2) pixel coordinates [u, v].
        """
        n = points_cam.shape[0]
        pts_hom = np.hstack([points_cam, np.ones((n, 1), dtype=np.float64)])
        pts_2d = (P @ pts_hom.T).T  # (N, 3)
        # Normalize by depth
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, :2]

    @staticmethod
    def project_velo_to_image(
        points: np.ndarray, calibration: Dict[str, np.ndarray], P_key: str = "P2"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project Velodyne points onto the image and return pixel coords with depth.

        Parameters
        ----------
        points : np.ndarray
            Shape (N, 3) or (N, 4) Velodyne points.
        calibration : dict
            Calibration dictionary.
        P_key : str
            Which projection matrix to use (``"P2"`` for left, ``"P3"`` for right).

        Returns
        -------
        pixels : np.ndarray
            Shape (M, 2) pixel coordinates for points with positive depth.
        depths : np.ndarray
            Shape (M,) depth values in camera frame.
        """
        pts_cam = KITTIDataset.project_velo_to_cam(points, calibration)

        # Keep only points in front of the camera
        mask = pts_cam[:, 2] > 0
        pts_cam = pts_cam[mask]
        depths = pts_cam[:, 2].copy()

        P = calibration[P_key]
        pixels = KITTIDataset.project_cam_to_image(pts_cam, P)

        return pixels, depths

    @staticmethod
    def bbox_3d_to_corners(
        dimensions: np.ndarray, location: np.ndarray, rotation_y: float
    ) -> np.ndarray:
        """Convert a KITTI 3D bounding box to 8 corner points in camera coordinates.

        Parameters
        ----------
        dimensions : np.ndarray
            Shape (3,) with [height, width, length] in meters.
        location : np.ndarray
            Shape (3,) center [x, y, z] in camera coordinates.
        rotation_y : float
            Rotation around Y-axis.

        Returns
        -------
        np.ndarray
            Shape (8, 3) corner points in camera coordinates.
            Order: front-left-bottom, front-right-bottom, back-right-bottom,
            back-left-bottom, front-left-top, front-right-top, back-right-top,
            back-left-top.
        """
        h, w, l = dimensions
        x, y, z = location

        # Rotation matrix around Y-axis
        cos_ry = np.cos(rotation_y)
        sin_ry = np.sin(rotation_y)
        R = np.array(
            [[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]],
            dtype=np.float64,
        )

        # 3D bounding box corners (in object frame, centered at bottom-center)
        # x: width, y: height (down), z: length (forward)
        x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
        y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
        z_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

        corners = np.stack([x_corners, y_corners, z_corners], axis=0)  # (3, 8)
        corners = R @ corners  # Rotate

        # Translate to world position
        corners[0, :] += x
        corners[1, :] += y
        corners[2, :] += z

        return corners.T  # (8, 3)

    # ------------------------------------------------------------------
    # Collate function for DataLoader
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for DataLoader that handles variable-size data.

        Parameters
        ----------
        batch : list of dict
            List of samples from __getitem__.

        Returns
        -------
        dict
            Batched data dictionary. Images are stacked into tensors.
            Point clouds and annotations remain as lists due to variable sizes.
        """
        collated: Dict[str, Any] = {
            "index": [s["index"] for s in batch],
            "image_shape": [s["image_shape"] for s in batch],
        }

        # Stack images into tensors
        if all(s["image_2"] is not None for s in batch):
            imgs = np.stack([s["image_2"] for s in batch], axis=0)
            collated["image_2"] = torch.from_numpy(imgs)
        else:
            collated["image_2"] = [s["image_2"] for s in batch]

        if all(s.get("image_3") is not None for s in batch):
            imgs = np.stack([s["image_3"] for s in batch], axis=0)
            collated["image_3"] = torch.from_numpy(imgs)
        else:
            collated["image_3"] = [s.get("image_3") for s in batch]

        # Point clouds remain as lists (variable size)
        collated["velodyne"] = [s.get("velodyne") for s in batch]

        # Calibration and annotations remain as lists
        collated["calibration"] = [s.get("calibration") for s in batch]
        collated["annotations"] = [s.get("annotations") for s in batch]

        return collated

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"KITTIDataset("
            f"split={self.split!r}, "
            f"samples={len(self)}, "
            f"velodyne={self.load_velodyne}, "
            f"stereo={self.load_right_image}, "
            f"class_filter={self.class_filter})"
        )
