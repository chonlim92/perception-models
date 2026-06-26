#!/usr/bin/env python3
"""
prepare_data.py - Data preparation for PointPillars training.

Processes KITTI and nuScenes datasets into efficient pickle-based info files
for fast loading during training. Creates ground truth databases for
copy-paste augmentation.

Features:
    - KITTI: Parse calibration, labels, create train/val info dicts, GT database
    - nuScenes: Parse database tables, generate info files with annotations
    - Compute dataset statistics (point cloud ranges, class distributions)
    - Save as pickle files for fast DataLoader access

Usage:
    python prepare_data.py --dataset kitti --data-root ./data
    python prepare_data.py --dataset nuscenes --data-root ./data
    python prepare_data.py --dataset all --data-root ./data --workers 8
"""

import argparse
import collections
import json
import logging
import os
import pickle
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

KITTI_CLASSES = ["Car", "Pedestrian", "Cyclist", "Van", "Truck", "Person_sitting",
                 "Tram", "Misc", "DontCare"]

KITTI_DETECTION_CLASSES = ["Car", "Pedestrian", "Cyclist"]

NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# Point cloud range for KITTI (x_min, y_min, z_min, x_max, y_max, z_max)
KITTI_POINT_CLOUD_RANGE = [0, -39.68, -3, 69.12, 39.68, 1]

# Point cloud range for nuScenes
NUSCENES_POINT_CLOUD_RANGE = [-50, -50, -5, 50, 50, 3]


# ============================================================================
# KITTI Calibration Parser
# ============================================================================

class KITTICalibration:
    """Parse and store KITTI calibration data.

    KITTI calibration files contain projection matrices for cameras (P0-P3),
    rectification matrix (R0_rect), and transformation from velodyne to camera
    coordinate system (Tr_velo_to_cam).
    """

    def __init__(self, calib_filepath: str):
        """Load calibration from file.

        Args:
            calib_filepath: Path to KITTI calibration .txt file.
        """
        calib_data = self._read_calib_file(calib_filepath)

        self.P0 = calib_data["P0"].reshape(3, 4)
        self.P1 = calib_data["P1"].reshape(3, 4)
        self.P2 = calib_data["P2"].reshape(3, 4)
        self.P3 = calib_data["P3"].reshape(3, 4)
        self.R0_rect = calib_data["R0_rect"].reshape(3, 3)
        self.Tr_velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
        self.Tr_imu_to_velo = calib_data.get(
            "Tr_imu_to_velo", np.zeros(12)
        ).reshape(3, 4)

    def _read_calib_file(self, filepath: str) -> Dict[str, np.ndarray]:
        """Read calibration file and return dict of numpy arrays.

        Args:
            filepath: Path to calibration text file.

        Returns:
            Dictionary mapping calibration names to numpy arrays.
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split(":", 1)
                data[key.strip()] = np.array(
                    [float(x) for x in value.strip().split()], dtype=np.float32
                )
        return data

    def cart_to_hom(self, pts: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates to homogeneous.

        Args:
            pts: (N, 3) array of 3D points.

        Returns:
            (N, 4) array with ones appended.
        """
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        return np.concatenate([pts, ones], axis=1)

    def lidar_to_camera(self, pts_lidar: np.ndarray) -> np.ndarray:
        """Transform points from lidar to camera coordinate system.

        Args:
            pts_lidar: (N, 3) points in lidar coordinates.

        Returns:
            (N, 3) points in camera coordinates.
        """
        pts_hom = self.cart_to_hom(pts_lidar)
        pts_cam = pts_hom @ self.Tr_velo_to_cam.T
        pts_rect = pts_cam @ self.R0_rect.T
        return pts_rect

    def camera_to_lidar(self, pts_cam: np.ndarray) -> np.ndarray:
        """Transform points from camera to lidar coordinate system.

        Args:
            pts_cam: (N, 3) points in camera coordinates.

        Returns:
            (N, 3) points in lidar coordinates.
        """
        R0_inv = np.linalg.inv(self.R0_rect)
        Tr_inv = np.linalg.inv(
            np.vstack([self.Tr_velo_to_cam, [0, 0, 0, 1]])
        )[:3, :]
        pts_unrect = pts_cam @ R0_inv.T
        pts_hom = self.cart_to_hom(pts_unrect)
        pts_lidar = pts_hom @ Tr_inv.T
        return pts_lidar

    def project_to_image(self, pts_3d: np.ndarray) -> np.ndarray:
        """Project 3D points (in camera frame) to image plane using P2.

        Args:
            pts_3d: (N, 3) points in rectified camera coordinates.

        Returns:
            (N, 2) pixel coordinates.
        """
        pts_hom = self.cart_to_hom(pts_3d)
        pts_2d = pts_hom @ self.P2.T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, :2]

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Export calibration matrices as dictionary for serialization."""
        return {
            "P0": self.P0,
            "P1": self.P1,
            "P2": self.P2,
            "P3": self.P3,
            "R0_rect": self.R0_rect,
            "Tr_velo_to_cam": self.Tr_velo_to_cam,
            "Tr_imu_to_velo": self.Tr_imu_to_velo,
        }


# ============================================================================
# KITTI Label Parser
# ============================================================================

def parse_kitti_label(label_filepath: str) -> List[Dict[str, Any]]:
    """Parse KITTI label file into list of annotation dicts.

    Each line in KITTI label file contains:
    type, truncated, occluded, alpha, bbox(4), dimensions(3), location(3), rotation_y

    Args:
        label_filepath: Path to KITTI label .txt file.

    Returns:
        List of annotation dictionaries with keys:
            - class_name: object class string
            - truncated: float [0,1] truncation level
            - occluded: int {0,1,2,3} occlusion state
            - alpha: float observation angle
            - bbox: [x1, y1, x2, y2] 2D bounding box in image
            - dimensions: [h, w, l] 3D box dimensions in meters
            - location: [x, y, z] 3D center in camera coordinates
            - rotation_y: float rotation around Y-axis in camera frame
    """
    annotations = []

    if not os.path.exists(label_filepath):
        return annotations

    with open(label_filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            annotation = {
                "class_name": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": np.array([float(x) for x in parts[4:8]], dtype=np.float32),
                "dimensions": np.array(
                    [float(parts[8]), float(parts[9]), float(parts[10])],
                    dtype=np.float32,
                ),  # h, w, l
                "location": np.array(
                    [float(parts[11]), float(parts[12]), float(parts[13])],
                    dtype=np.float32,
                ),  # x, y, z in camera coords
                "rotation_y": float(parts[14]),
            }

            # Parse optional score field (for detection results)
            if len(parts) > 15:
                annotation["score"] = float(parts[15])

            annotations.append(annotation)

    return annotations


def boxes_camera_to_lidar(
    boxes_cam: np.ndarray, calib: KITTICalibration
) -> np.ndarray:
    """Convert 3D boxes from camera to lidar coordinate system.

    Args:
        boxes_cam: (N, 7) array with [x, y, z, h, w, l, ry] in camera frame.
        calib: KITTICalibration instance.

    Returns:
        (N, 7) array with [x, y, z, dx, dy, dz, heading] in lidar frame.
    """
    centers = boxes_cam[:, :3]
    dims = boxes_cam[:, 3:6]  # h, w, l in camera
    rotations = boxes_cam[:, 6]

    # Transform centers from camera to lidar
    centers_lidar = calib.camera_to_lidar(centers)

    # In camera: h, w, l; In lidar: dx(l), dy(w), dz(h)
    dims_lidar = dims[:, [2, 1, 0]]  # l, w, h

    # Rotation: camera ry -> lidar heading
    # In camera frame, ry is rotation around Y (down). In lidar, heading is around Z (up).
    headings = -(rotations + np.pi / 2)

    boxes_lidar = np.column_stack([centers_lidar, dims_lidar, headings])
    return boxes_lidar.astype(np.float32)


# ============================================================================
# Point Cloud I/O
# ============================================================================

def load_point_cloud(filepath: str) -> np.ndarray:
    """Load KITTI-format point cloud from binary file.

    Args:
        filepath: Path to .bin file containing float32 x,y,z,intensity.

    Returns:
        (N, 4) numpy array of points.
    """
    points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    return points


def points_in_box(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Find points inside a rotated 3D bounding box (in lidar frame).

    Args:
        points: (N, 3+) array of point coordinates.
        box: (7,) array [cx, cy, cz, dx, dy, dz, heading].

    Returns:
        Boolean mask of shape (N,) indicating which points are inside.
    """
    cx, cy, cz, dx, dy, dz, heading = box

    # Translate points to box center
    shifted = points[:, :3] - np.array([cx, cy, cz])

    # Rotate points to box-aligned frame
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    rot_matrix = np.array([[cos_h, -sin_h, 0],
                           [sin_h, cos_h, 0],
                           [0, 0, 1]], dtype=np.float32)
    aligned = shifted @ rot_matrix.T

    # Check if aligned points are within half-dimensions
    mask_x = np.abs(aligned[:, 0]) <= (dx / 2)
    mask_y = np.abs(aligned[:, 1]) <= (dy / 2)
    mask_z = np.abs(aligned[:, 2]) <= (dz / 2)

    return mask_x & mask_y & mask_z


# ============================================================================
# KITTI Data Preparation
# ============================================================================

def create_kitti_info(
    sample_idx: int,
    velodyne_dir: str,
    label_dir: str,
    calib_dir: str,
    image_dir: str,
    has_label: bool = True,
) -> Dict[str, Any]:
    """Create info dict for a single KITTI sample.

    Args:
        sample_idx: Integer index of the sample (e.g., 0 for 000000).
        velodyne_dir: Path to velodyne directory.
        label_dir: Path to label_2 directory.
        calib_dir: Path to calib directory.
        image_dir: Path to image_2 directory.
        has_label: Whether this sample has ground truth labels.

    Returns:
        Dictionary with sample metadata, calibration, and annotations.
    """
    idx_str = f"{sample_idx:06d}"

    # Paths
    velodyne_path = os.path.join(velodyne_dir, f"{idx_str}.bin")
    calib_path = os.path.join(calib_dir, f"{idx_str}.txt")
    image_path = os.path.join(image_dir, f"{idx_str}.png")
    label_path = os.path.join(label_dir, f"{idx_str}.txt") if has_label else None

    # Basic info
    info = {
        "sample_idx": sample_idx,
        "point_cloud": {
            "num_features": 4,
            "velodyne_path": velodyne_path,
        },
        "image": {
            "image_path": image_path,
            "image_shape": None,  # Will be filled if image exists
        },
        "calib": None,
        "annos": None,
    }

    # Parse calibration
    if os.path.exists(calib_path):
        calib = KITTICalibration(calib_path)
        info["calib"] = calib.to_dict()
    else:
        logger.warning(f"Calibration file not found: {calib_path}")
        return info

    # Try to get image shape
    if os.path.exists(image_path):
        try:
            # Read PNG header to get dimensions without loading full image
            with open(image_path, "rb") as img_f:
                img_f.read(8)  # PNG signature
                img_f.read(4)  # IHDR length
                img_f.read(4)  # IHDR tag
                width = struct.unpack(">I", img_f.read(4))[0]
                height = struct.unpack(">I", img_f.read(4))[0]
                info["image"]["image_shape"] = np.array(
                    [height, width], dtype=np.int32
                )
        except (IOError, struct.error):
            info["image"]["image_shape"] = np.array([375, 1242], dtype=np.int32)

    # Parse annotations
    if has_label and label_path and os.path.exists(label_path):
        raw_annos = parse_kitti_label(label_path)

        if raw_annos:
            annos = {
                "name": np.array([a["class_name"] for a in raw_annos]),
                "truncated": np.array(
                    [a["truncated"] for a in raw_annos], dtype=np.float32
                ),
                "occluded": np.array(
                    [a["occluded"] for a in raw_annos], dtype=np.int32
                ),
                "alpha": np.array(
                    [a["alpha"] for a in raw_annos], dtype=np.float32
                ),
                "bbox": np.array(
                    [a["bbox"] for a in raw_annos], dtype=np.float32
                ),
                "dimensions": np.array(
                    [a["dimensions"] for a in raw_annos], dtype=np.float32
                ),
                "location": np.array(
                    [a["location"] for a in raw_annos], dtype=np.float32
                ),
                "rotation_y": np.array(
                    [a["rotation_y"] for a in raw_annos], dtype=np.float32
                ),
            }

            # Compute boxes in lidar frame
            num_objects = len(raw_annos)
            if num_objects > 0:
                boxes_cam = np.column_stack([
                    annos["location"],
                    annos["dimensions"],
                    annos["rotation_y"].reshape(-1, 1),
                ])
                annos["boxes_lidar"] = boxes_camera_to_lidar(boxes_cam, calib)

            # Compute difficulty level for each object
            annos["difficulty"] = compute_kitti_difficulty(annos)

            info["annos"] = annos

    return info


def compute_kitti_difficulty(annos: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute KITTI difficulty level for each annotation.

    Difficulty is based on bbox height, occlusion, and truncation:
    - Easy:     height >= 40px, occlusion <= 0, truncation <= 0.15
    - Moderate: height >= 25px, occlusion <= 1, truncation <= 0.30
    - Hard:     height >= 25px, occlusion <= 2, truncation <= 0.50

    Args:
        annos: Annotation dictionary with bbox, occluded, truncated.

    Returns:
        Array of difficulty levels (-1=unknown, 0=easy, 1=moderate, 2=hard).
    """
    heights = annos["bbox"][:, 3] - annos["bbox"][:, 1]
    occlusion = annos["occluded"]
    truncation = annos["truncated"]

    num_objs = len(heights)
    difficulty = np.full(num_objs, -1, dtype=np.int32)

    for i in range(num_objs):
        h = heights[i]
        occ = occlusion[i]
        trunc = truncation[i]

        if h >= 40 and occ <= 0 and trunc <= 0.15:
            difficulty[i] = 0  # Easy
        elif h >= 25 and occ <= 1 and trunc <= 0.30:
            difficulty[i] = 1  # Moderate
        elif h >= 25 and occ <= 2 and trunc <= 0.50:
            difficulty[i] = 2  # Hard
        else:
            difficulty[i] = -1  # Unknown / DontCare

    return difficulty


def prepare_kitti_dataset(
    data_root: str, workers: int = 4, classes: Optional[List[str]] = None
) -> None:
    """Prepare complete KITTI dataset for PointPillars training.

    Creates info pickle files and ground truth database.

    Args:
        data_root: Root directory containing kitti/ subdirectory.
        workers: Number of parallel workers for processing.
        classes: List of class names to include (default: Car, Pedestrian, Cyclist).
    """
    if classes is None:
        classes = KITTI_DETECTION_CLASSES

    kitti_root = os.path.join(data_root, "kitti")
    velodyne_dir = os.path.join(kitti_root, "training", "velodyne")
    label_dir = os.path.join(kitti_root, "training", "label_2")
    calib_dir = os.path.join(kitti_root, "training", "calib")
    image_dir = os.path.join(kitti_root, "training", "image_2")
    imagesets_dir = os.path.join(kitti_root, "ImageSets")

    output_dir = os.path.join(kitti_root, "kitti_infos")
    gt_database_dir = os.path.join(kitti_root, "gt_database")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gt_database_dir, exist_ok=True)

    # Load split files
    train_ids = _load_split_file(os.path.join(imagesets_dir, "train.txt"))
    val_ids = _load_split_file(os.path.join(imagesets_dir, "val.txt"))
    test_ids = _load_split_file(os.path.join(imagesets_dir, "test.txt"))

    logger.info(f"KITTI splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Process training infos
    logger.info("Processing KITTI training samples...")
    train_infos = _process_kitti_samples(
        train_ids, velodyne_dir, label_dir, calib_dir, image_dir,
        has_label=True, workers=workers
    )

    # Process validation infos
    logger.info("Processing KITTI validation samples...")
    val_infos = _process_kitti_samples(
        val_ids, velodyne_dir, label_dir, calib_dir, image_dir,
        has_label=True, workers=workers
    )

    # Process test infos (no labels)
    logger.info("Processing KITTI test samples...")
    test_velodyne_dir = os.path.join(kitti_root, "testing", "velodyne")
    test_calib_dir = os.path.join(kitti_root, "testing", "calib")
    test_image_dir = os.path.join(kitti_root, "testing", "image_2")
    test_infos = _process_kitti_samples(
        test_ids, test_velodyne_dir, None, test_calib_dir, test_image_dir,
        has_label=False, workers=workers
    )

    # Save info files
    train_info_path = os.path.join(output_dir, "kitti_infos_train.pkl")
    val_info_path = os.path.join(output_dir, "kitti_infos_val.pkl")
    trainval_info_path = os.path.join(output_dir, "kitti_infos_trainval.pkl")
    test_info_path = os.path.join(output_dir, "kitti_infos_test.pkl")

    with open(train_info_path, "wb") as f:
        pickle.dump(train_infos, f)
    logger.info(f"Saved: {train_info_path} ({len(train_infos)} samples)")

    with open(val_info_path, "wb") as f:
        pickle.dump(val_infos, f)
    logger.info(f"Saved: {val_info_path} ({len(val_infos)} samples)")

    with open(trainval_info_path, "wb") as f:
        pickle.dump(train_infos + val_infos, f)
    logger.info(f"Saved: {trainval_info_path} ({len(train_infos) + len(val_infos)} samples)")

    with open(test_info_path, "wb") as f:
        pickle.dump(test_infos, f)
    logger.info(f"Saved: {test_info_path} ({len(test_infos)} samples)")

    # Create ground truth database for copy-paste augmentation
    logger.info("Creating KITTI ground truth database...")
    gt_db_info = create_kitti_gt_database(
        train_infos, velodyne_dir, gt_database_dir, classes
    )

    gt_db_path = os.path.join(output_dir, "kitti_dbinfos_train.pkl")
    with open(gt_db_path, "wb") as f:
        pickle.dump(gt_db_info, f)
    logger.info(f"Saved GT database info: {gt_db_path}")

    # Compute and save dataset statistics
    logger.info("Computing KITTI dataset statistics...")
    stats = compute_kitti_statistics(train_infos, velodyne_dir, classes)
    stats_path = os.path.join(output_dir, "kitti_statistics.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    logger.info(f"Saved statistics: {stats_path}")

    _print_kitti_statistics(stats)


def _load_split_file(filepath: str) -> List[int]:
    """Load sample indices from split file.

    Args:
        filepath: Path to split text file (one index per line).

    Returns:
        List of integer sample indices.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Split file not found: {filepath}")
        return []

    indices = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                indices.append(int(line))
    return indices


def _process_kitti_samples(
    sample_ids: List[int],
    velodyne_dir: str,
    label_dir: Optional[str],
    calib_dir: str,
    image_dir: str,
    has_label: bool,
    workers: int,
) -> List[Dict[str, Any]]:
    """Process multiple KITTI samples in parallel.

    Args:
        sample_ids: List of sample indices.
        velodyne_dir: Path to velodyne directory.
        label_dir: Path to label directory (None for test).
        calib_dir: Path to calibration directory.
        image_dir: Path to image directory.
        has_label: Whether samples have labels.
        workers: Number of parallel workers.

    Returns:
        List of info dictionaries.
    """
    infos = []

    if workers <= 1:
        for idx in sample_ids:
            info = create_kitti_info(
                idx, velodyne_dir, label_dir or "", calib_dir, image_dir, has_label
            )
            if info is not None:
                infos.append(info)
            if len(infos) % 500 == 0 and len(infos) > 0:
                logger.info(f"  Processed {len(infos)}/{len(sample_ids)} samples")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx in sample_ids:
                future = executor.submit(
                    create_kitti_info,
                    idx, velodyne_dir, label_dir or "", calib_dir, image_dir, has_label,
                )
                futures[future] = idx

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    infos.append(result)
                completed += 1
                if completed % 500 == 0:
                    logger.info(f"  Processed {completed}/{len(sample_ids)} samples")

    # Sort by sample index for reproducibility
    infos.sort(key=lambda x: x["sample_idx"])
    return infos


def create_kitti_gt_database(
    infos: List[Dict[str, Any]],
    velodyne_dir: str,
    gt_database_dir: str,
    classes: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Create ground truth database for copy-paste augmentation.

    Extracts point clouds within each ground truth bounding box and saves
    them as individual files. This enables the GT-Aug strategy where GT
    boxes with their points are pasted into new scenes during training.

    Args:
        infos: List of training info dictionaries.
        velodyne_dir: Path to velodyne point cloud directory.
        gt_database_dir: Output directory for GT point cloud crops.
        classes: List of class names to include.

    Returns:
        Dictionary mapping class names to lists of GT sample info dicts.
    """
    db_infos = {cls: [] for cls in classes}
    total_saved = 0

    for info_idx, info in enumerate(infos):
        sample_idx = info["sample_idx"]
        annos = info.get("annos")
        calib_dict = info.get("calib")

        if annos is None or calib_dict is None:
            continue

        # Load point cloud
        pc_path = os.path.join(velodyne_dir, f"{sample_idx:06d}.bin")
        if not os.path.exists(pc_path):
            continue

        points = load_point_cloud(pc_path)

        names = annos["name"]
        boxes_lidar = annos.get("boxes_lidar")
        difficulty = annos.get("difficulty", np.zeros(len(names), dtype=np.int32))

        if boxes_lidar is None:
            continue

        for obj_idx in range(len(names)):
            class_name = names[obj_idx]
            if class_name not in classes:
                continue

            box = boxes_lidar[obj_idx]

            # Find points inside this box
            mask = points_in_box(points, box)
            num_points_in_box = mask.sum()

            if num_points_in_box == 0:
                continue

            # Extract points and shift to box center
            gt_points = points[mask].copy()

            # Save GT point cloud
            gt_filename = f"{sample_idx:06d}_{class_name}_{obj_idx}.bin"
            gt_filepath = os.path.join(gt_database_dir, gt_filename)
            gt_points.astype(np.float32).tofile(gt_filepath)

            # Create database entry
            db_entry = {
                "name": class_name,
                "path": gt_filepath,
                "sample_idx": sample_idx,
                "gt_idx": obj_idx,
                "box3d_lidar": box,
                "num_points_in_gt": int(num_points_in_box),
                "difficulty": int(difficulty[obj_idx]),
                "bbox": annos["bbox"][obj_idx],
                "truncated": float(annos["truncated"][obj_idx]),
                "occluded": int(annos["occluded"][obj_idx]),
            }
            db_infos[class_name].append(db_entry)
            total_saved += 1

        if (info_idx + 1) % 200 == 0:
            logger.info(
                f"  GT database: processed {info_idx + 1}/{len(infos)} samples, "
                f"saved {total_saved} objects"
            )

    # Log summary
    for cls, entries in db_infos.items():
        logger.info(f"  GT database class '{cls}': {len(entries)} samples")

    return db_infos


def compute_kitti_statistics(
    infos: List[Dict[str, Any]],
    velodyne_dir: str,
    classes: List[str],
    max_samples: int = 500,
) -> Dict[str, Any]:
    """Compute dataset statistics for normalization and augmentation parameters.

    Args:
        infos: List of info dictionaries.
        velodyne_dir: Path to velodyne directory.
        classes: List of class names.
        max_samples: Maximum number of samples for point cloud stats.

    Returns:
        Dictionary with statistics (ranges, distributions, means, etc.).
    """
    stats = {
        "point_cloud": {"xyz_min": None, "xyz_max": None, "xyz_mean": None, "num_points": []},
        "class_distribution": collections.Counter(),
        "box_sizes": {cls: [] for cls in classes},
        "box_rotations": {cls: [] for cls in classes},
    }

    # Point cloud range statistics (sample a subset for efficiency)
    sample_indices = np.random.choice(
        len(infos), min(max_samples, len(infos)), replace=False
    )

    all_mins = []
    all_maxs = []
    all_means = []

    for i in sample_indices:
        pc_path = infos[i]["point_cloud"]["velodyne_path"]
        if not os.path.exists(pc_path):
            continue

        points = load_point_cloud(pc_path)
        stats["point_cloud"]["num_points"].append(len(points))
        all_mins.append(points[:, :3].min(axis=0))
        all_maxs.append(points[:, :3].max(axis=0))
        all_means.append(points[:, :3].mean(axis=0))

    if all_mins:
        stats["point_cloud"]["xyz_min"] = np.array(all_mins).min(axis=0).tolist()
        stats["point_cloud"]["xyz_max"] = np.array(all_maxs).max(axis=0).tolist()
        stats["point_cloud"]["xyz_mean"] = np.array(all_means).mean(axis=0).tolist()
        stats["point_cloud"]["num_points_mean"] = float(
            np.mean(stats["point_cloud"]["num_points"])
        )
        stats["point_cloud"]["num_points_std"] = float(
            np.std(stats["point_cloud"]["num_points"])
        )

    # Class and box size statistics
    for info in infos:
        annos = info.get("annos")
        if annos is None:
            continue

        for i, name in enumerate(annos["name"]):
            stats["class_distribution"][name] += 1

            if name in classes and "boxes_lidar" in annos:
                box = annos["boxes_lidar"][i]
                stats["box_sizes"][name].append(box[3:6].tolist())  # dx, dy, dz
                stats["box_rotations"][name].append(float(box[6]))

    # Compute box size means and stds per class
    stats["box_size_stats"] = {}
    for cls in classes:
        sizes = stats["box_sizes"][cls]
        if sizes:
            sizes_arr = np.array(sizes)
            stats["box_size_stats"][cls] = {
                "mean": sizes_arr.mean(axis=0).tolist(),
                "std": sizes_arr.std(axis=0).tolist(),
                "min": sizes_arr.min(axis=0).tolist(),
                "max": sizes_arr.max(axis=0).tolist(),
            }

    # Convert box_sizes lists to summary (don't save full lists)
    del stats["box_sizes"]
    del stats["box_rotations"]

    return stats


def _print_kitti_statistics(stats: Dict[str, Any]) -> None:
    """Pretty-print dataset statistics."""
    logger.info("=" * 60)
    logger.info("KITTI Dataset Statistics")
    logger.info("=" * 60)

    pc_stats = stats["point_cloud"]
    if pc_stats["xyz_min"] is not None:
        logger.info(f"Point Cloud XYZ range:")
        logger.info(f"  X: [{pc_stats['xyz_min'][0]:.2f}, {pc_stats['xyz_max'][0]:.2f}]")
        logger.info(f"  Y: [{pc_stats['xyz_min'][1]:.2f}, {pc_stats['xyz_max'][1]:.2f}]")
        logger.info(f"  Z: [{pc_stats['xyz_min'][2]:.2f}, {pc_stats['xyz_max'][2]:.2f}]")
        logger.info(f"  Mean points/scan: {pc_stats['num_points_mean']:.0f} +/- {pc_stats['num_points_std']:.0f}")

    logger.info(f"\nClass distribution:")
    for cls, count in sorted(stats["class_distribution"].items(), key=lambda x: -x[1]):
        logger.info(f"  {cls:20s}: {count:6d}")

    logger.info(f"\nBox size statistics (dx, dy, dz in lidar frame):")
    for cls, box_stats in stats.get("box_size_stats", {}).items():
        logger.info(
            f"  {cls:12s}: mean=[{box_stats['mean'][0]:.2f}, {box_stats['mean'][1]:.2f}, "
            f"{box_stats['mean'][2]:.2f}] "
            f"std=[{box_stats['std'][0]:.2f}, {box_stats['std'][1]:.2f}, "
            f"{box_stats['std'][2]:.2f}]"
        )


# ============================================================================
# nuScenes Data Preparation
# ============================================================================

class NuScenesParser:
    """Parse nuScenes database tables for PointPillars training.

    Reads the nuScenes JSON database tables and generates info dictionaries
    with point cloud paths, annotations in the lidar frame, ego poses,
    and timestamps.
    """

    def __init__(self, dataroot: str, version: str = "v1.0-mini"):
        """Initialize nuScenes parser.

        Args:
            dataroot: Root directory of nuScenes dataset.
            version: Dataset version string (e.g., 'v1.0-mini', 'v1.0-trainval').
        """
        self.dataroot = dataroot
        self.version = version
        self.table_dir = os.path.join(dataroot, version)

        # Load database tables
        self.tables = {}
        table_names = [
            "category", "attribute", "visibility", "instance", "sensor",
            "calibrated_sensor", "ego_pose", "log", "scene", "sample",
            "sample_data", "sample_annotation", "map",
        ]

        for table_name in table_names:
            table_path = os.path.join(self.table_dir, f"{table_name}.json")
            if os.path.exists(table_path):
                with open(table_path, "r") as f:
                    self.tables[table_name] = json.load(f)
                logger.info(f"  Loaded {table_name}: {len(self.tables[table_name])} records")
            else:
                self.tables[table_name] = []
                logger.warning(f"  Table not found: {table_path}")

        # Build token-to-record lookup tables
        self._build_indices()

    def _build_indices(self) -> None:
        """Build fast lookup indices from token to record."""
        self.token_to_record = {}
        for table_name, records in self.tables.items():
            self.token_to_record[table_name] = {}
            for record in records:
                if "token" in record:
                    self.token_to_record[table_name][record["token"]] = record

    def get_record(self, table_name: str, token: str) -> Dict[str, Any]:
        """Get a record by table name and token.

        Args:
            table_name: Name of the database table.
            token: Unique identifier token.

        Returns:
            Record dictionary.
        """
        return self.token_to_record[table_name][token]

    def get_sample_data_path(self, sample_data_token: str) -> str:
        """Get full file path for a sample_data record.

        Args:
            sample_data_token: Token of the sample_data record.

        Returns:
            Absolute file path.
        """
        sd_record = self.get_record("sample_data", sample_data_token)
        return os.path.join(self.dataroot, sd_record["filename"])

    def get_ego_pose(self, ego_pose_token: str) -> Dict[str, Any]:
        """Get ego vehicle pose.

        Args:
            ego_pose_token: Token of the ego_pose record.

        Returns:
            Dictionary with 'translation' and 'rotation' (quaternion).
        """
        return self.get_record("ego_pose", ego_pose_token)

    def get_sensor_transform(self, calibrated_sensor_token: str) -> Dict[str, Any]:
        """Get sensor calibration (extrinsics + intrinsics).

        Args:
            calibrated_sensor_token: Token of the calibrated_sensor record.

        Returns:
            Dictionary with 'translation', 'rotation', and optionally 'camera_intrinsic'.
        """
        return self.get_record("calibrated_sensor", calibrated_sensor_token)

    def quaternion_to_rotation_matrix(self, quaternion: List[float]) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

        Args:
            quaternion: [w, x, y, z] quaternion.

        Returns:
            (3, 3) rotation matrix.
        """
        w, x, y, z = quaternion
        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
            [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float64)
        return R

    def get_lidar_to_global(self, sample_data_record: Dict) -> np.ndarray:
        """Compute transformation matrix from lidar frame to global frame.

        Args:
            sample_data_record: A sample_data record for the lidar sensor.

        Returns:
            (4, 4) transformation matrix.
        """
        # Sensor to ego
        cs_record = self.get_record(
            "calibrated_sensor", sample_data_record["calibrated_sensor_token"]
        )
        sensor_to_ego = np.eye(4, dtype=np.float64)
        sensor_to_ego[:3, :3] = self.quaternion_to_rotation_matrix(cs_record["rotation"])
        sensor_to_ego[:3, 3] = cs_record["translation"]

        # Ego to global
        ep_record = self.get_record(
            "ego_pose", sample_data_record["ego_pose_token"]
        )
        ego_to_global = np.eye(4, dtype=np.float64)
        ego_to_global[:3, :3] = self.quaternion_to_rotation_matrix(ep_record["rotation"])
        ego_to_global[:3, 3] = ep_record["translation"]

        # Compose: lidar -> ego -> global
        lidar_to_global = ego_to_global @ sensor_to_ego
        return lidar_to_global

    def get_annotation_in_lidar(
        self, annotation_token: str, lidar_to_global: np.ndarray
    ) -> Dict[str, Any]:
        """Transform an annotation from global frame to lidar frame.

        Args:
            annotation_token: Token of the sample_annotation.
            lidar_to_global: (4, 4) lidar-to-global transformation.

        Returns:
            Dictionary with annotation in lidar frame.
        """
        ann = self.get_record("sample_annotation", annotation_token)
        category = self.get_record("category", ann["category_token"])

        # Global position and orientation
        center_global = np.array(ann["translation"], dtype=np.float64)
        size = np.array(ann["size"], dtype=np.float32)  # width, length, height in nuScenes
        rotation_quat = ann["rotation"]  # [w, x, y, z]

        # Transform center to lidar frame
        global_to_lidar = np.linalg.inv(lidar_to_global)
        center_hom = np.append(center_global, 1.0)
        center_lidar = (global_to_lidar @ center_hom)[:3]

        # Transform rotation to lidar frame
        R_global = self.quaternion_to_rotation_matrix(rotation_quat)
        R_global_to_lidar = global_to_lidar[:3, :3]
        R_lidar = R_global_to_lidar @ R_global

        # Extract yaw angle from rotation matrix (around z-axis in lidar frame)
        yaw = np.arctan2(R_lidar[1, 0], R_lidar[0, 0])

        # nuScenes size convention: [width, length, height]
        # Convert to [dx, dy, dz] in lidar frame: [length, width, height]
        dx = size[1]  # length
        dy = size[0]  # width
        dz = size[2]  # height

        return {
            "token": annotation_token,
            "category_name": category["name"],
            "center_lidar": center_lidar.astype(np.float32),
            "size": np.array([dx, dy, dz], dtype=np.float32),
            "yaw": float(yaw),
            "box3d_lidar": np.array(
                [center_lidar[0], center_lidar[1], center_lidar[2], dx, dy, dz, yaw],
                dtype=np.float32,
            ),
            "num_lidar_pts": ann["num_lidar_pts"],
            "num_radar_pts": ann["num_radar_pts"],
            "instance_token": ann["instance_token"],
            "visibility_token": ann["visibility_token"],
            "attribute_tokens": ann["attribute_tokens"],
        }


def prepare_nuscenes_dataset(
    data_root: str, version: str = "v1.0-mini", workers: int = 4
) -> None:
    """Prepare nuScenes dataset for PointPillars training.

    Args:
        data_root: Root directory containing nuscenes/ subdirectory.
        version: nuScenes version string.
        workers: Number of parallel workers.
    """
    nuscenes_root = os.path.join(data_root, "nuscenes")
    output_dir = os.path.join(nuscenes_root, "nuscenes_infos")
    gt_database_dir = os.path.join(nuscenes_root, "gt_database")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gt_database_dir, exist_ok=True)

    logger.info(f"Parsing nuScenes database ({version})...")
    parser = NuScenesParser(nuscenes_root, version)

    # Get all samples
    samples = parser.tables.get("sample", [])
    if not samples:
        logger.error("No samples found in nuScenes database.")
        return

    logger.info(f"Found {len(samples)} samples in {version}")

    # Process each sample
    infos = []
    for sample_idx, sample in enumerate(samples):
        info = _create_nuscenes_info(parser, sample, nuscenes_root)
        if info is not None:
            infos.append(info)

        if (sample_idx + 1) % 50 == 0:
            logger.info(f"  Processed {sample_idx + 1}/{len(samples)} samples")

    logger.info(f"Generated {len(infos)} valid info records")

    # Split into train/val for mini (all scenes in mini are usable)
    # For v1.0-mini: use first 7 scenes for train, last 3 for val
    scene_tokens = list({info["scene_token"] for info in infos})
    scene_tokens.sort()

    train_scene_count = max(1, int(len(scene_tokens) * 0.7))
    train_scenes = set(scene_tokens[:train_scene_count])
    val_scenes = set(scene_tokens[train_scene_count:])

    train_infos = [info for info in infos if info["scene_token"] in train_scenes]
    val_infos = [info for info in infos if info["scene_token"] in val_scenes]

    logger.info(f"Split: train={len(train_infos)}, val={len(val_infos)}")

    # Save info files
    train_path = os.path.join(output_dir, f"nuscenes_infos_{version}_train.pkl")
    val_path = os.path.join(output_dir, f"nuscenes_infos_{version}_val.pkl")

    with open(train_path, "wb") as f:
        pickle.dump(train_infos, f)
    logger.info(f"Saved: {train_path}")

    with open(val_path, "wb") as f:
        pickle.dump(val_infos, f)
    logger.info(f"Saved: {val_path}")

    # Create GT database
    logger.info("Creating nuScenes ground truth database...")
    gt_db_info = _create_nuscenes_gt_database(
        parser, train_infos, nuscenes_root, gt_database_dir
    )

    gt_db_path = os.path.join(output_dir, f"nuscenes_dbinfos_{version}_train.pkl")
    with open(gt_db_path, "wb") as f:
        pickle.dump(gt_db_info, f)
    logger.info(f"Saved GT database: {gt_db_path}")

    # Compute statistics
    logger.info("Computing nuScenes statistics...")
    stats = _compute_nuscenes_statistics(infos, nuscenes_root)
    stats_path = os.path.join(output_dir, f"nuscenes_statistics_{version}.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    logger.info(f"Saved statistics: {stats_path}")
    _print_nuscenes_statistics(stats)


def _create_nuscenes_info(
    parser: NuScenesParser, sample: Dict, dataroot: str
) -> Optional[Dict[str, Any]]:
    """Create info dictionary for a single nuScenes sample.

    Args:
        parser: NuScenesParser instance.
        sample: Sample record from nuScenes database.
        dataroot: Root directory of the dataset.

    Returns:
        Info dictionary or None if lidar data not available.
    """
    # Get lidar sample data
    lidar_token = sample["data"].get("LIDAR_TOP")
    if lidar_token is None:
        return None

    lidar_sd = parser.get_record("sample_data", lidar_token)
    lidar_path = os.path.join(dataroot, lidar_sd["filename"])

    # Compute lidar-to-global transform
    lidar_to_global = parser.get_lidar_to_global(lidar_sd)

    # Get ego pose
    ego_pose = parser.get_record("ego_pose", lidar_sd["ego_pose_token"])

    # Process annotations
    annotations = []
    for ann_token in sample["anns"]:
        ann_info = parser.get_annotation_in_lidar(ann_token, lidar_to_global)
        annotations.append(ann_info)

    # Map category names to detection classes
    mapped_annotations = []
    for ann in annotations:
        det_class = _map_nuscenes_category(ann["category_name"])
        if det_class is not None:
            ann["detection_name"] = det_class
            mapped_annotations.append(ann)

    info = {
        "token": sample["token"],
        "scene_token": sample["scene_token"],
        "timestamp": sample["timestamp"],
        "lidar_path": lidar_path,
        "lidar_token": lidar_token,
        "lidar_to_global": lidar_to_global.astype(np.float32),
        "ego_pose": {
            "translation": np.array(ego_pose["translation"], dtype=np.float32),
            "rotation": ego_pose["rotation"],
        },
        "annotations": mapped_annotations,
        "num_lidar_pts": lidar_sd.get("num_lidar_pts", -1),
    }

    # Add sweep information (intermediate lidar frames between keyframes)
    sweeps = _get_lidar_sweeps(parser, lidar_sd, max_sweeps=10)
    info["sweeps"] = sweeps

    return info


def _get_lidar_sweeps(
    parser: NuScenesParser, lidar_sd: Dict, max_sweeps: int = 10
) -> List[Dict[str, Any]]:
    """Get intermediate lidar sweeps between keyframes.

    Args:
        parser: NuScenesParser instance.
        lidar_sd: Current keyframe's lidar sample_data record.
        max_sweeps: Maximum number of sweeps to collect.

    Returns:
        List of sweep info dictionaries.
    """
    sweeps = []
    current_sd = lidar_sd

    # Traverse backwards through linked list of sample_data
    while len(sweeps) < max_sweeps:
        prev_token = current_sd.get("prev", "")
        if not prev_token:
            break

        prev_sd = parser.get_record("sample_data", prev_token)
        sweep_info = {
            "lidar_path": os.path.join(
                parser.dataroot, prev_sd["filename"]
            ),
            "timestamp": prev_sd["timestamp"],
            "sensor_to_ego": None,
            "ego_to_global": None,
        }

        # Sensor to ego transform for this sweep
        cs_record = parser.get_record(
            "calibrated_sensor", prev_sd["calibrated_sensor_token"]
        )
        sensor_to_ego = np.eye(4, dtype=np.float32)
        sensor_to_ego[:3, :3] = parser.quaternion_to_rotation_matrix(
            cs_record["rotation"]
        ).astype(np.float32)
        sensor_to_ego[:3, 3] = cs_record["translation"]
        sweep_info["sensor_to_ego"] = sensor_to_ego

        # Ego to global transform for this sweep
        ep_record = parser.get_record("ego_pose", prev_sd["ego_pose_token"])
        ego_to_global = np.eye(4, dtype=np.float32)
        ego_to_global[:3, :3] = parser.quaternion_to_rotation_matrix(
            ep_record["rotation"]
        ).astype(np.float32)
        ego_to_global[:3, 3] = ep_record["translation"]
        sweep_info["ego_to_global"] = ego_to_global

        sweeps.append(sweep_info)
        current_sd = prev_sd

    return sweeps


def _map_nuscenes_category(category_name: str) -> Optional[str]:
    """Map nuScenes fine-grained category to detection class.

    Args:
        category_name: Full nuScenes category name (e.g., 'vehicle.car').

    Returns:
        Detection class name or None if not a detection class.
    """
    category_map = {
        "vehicle.car": "car",
        "vehicle.truck": "truck",
        "vehicle.construction": "construction_vehicle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.trailer": "trailer",
        "movable_object.barrier": "barrier",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.bicycle": "bicycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
    }

    for prefix, det_name in category_map.items():
        if category_name.startswith(prefix):
            return det_name

    return None


def _create_nuscenes_gt_database(
    parser: NuScenesParser,
    infos: List[Dict[str, Any]],
    dataroot: str,
    gt_database_dir: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Create nuScenes ground truth database for copy-paste augmentation.

    Args:
        parser: NuScenesParser instance.
        infos: List of training info dictionaries.
        dataroot: nuScenes root directory.
        gt_database_dir: Output directory for GT crops.

    Returns:
        Dictionary mapping detection class names to GT sample lists.
    """
    db_infos = {cls: [] for cls in NUSCENES_CLASSES}

    for info_idx, info in enumerate(infos):
        lidar_path = info["lidar_path"]
        if not os.path.exists(lidar_path):
            continue

        # Load nuScenes lidar (5 channels: x, y, z, intensity, ring_index)
        points = np.fromfile(lidar_path, dtype=np.float32)
        # nuScenes has 5 features per point
        num_features = 5
        if points.size % num_features != 0:
            num_features = 4
        points = points.reshape(-1, num_features)

        for ann_idx, ann in enumerate(info["annotations"]):
            det_name = ann.get("detection_name")
            if det_name is None or det_name not in NUSCENES_CLASSES:
                continue

            box = ann["box3d_lidar"]
            mask = points_in_box(points, box)
            num_pts = mask.sum()

            if num_pts < 1:
                continue

            gt_points = points[mask].copy()

            # Save GT points
            gt_filename = f"{info['token']}_{det_name}_{ann_idx}.bin"
            gt_filepath = os.path.join(gt_database_dir, gt_filename)
            gt_points.astype(np.float32).tofile(gt_filepath)

            db_entry = {
                "name": det_name,
                "path": gt_filepath,
                "token": info["token"],
                "gt_idx": ann_idx,
                "box3d_lidar": box,
                "num_points_in_gt": int(num_pts),
                "instance_token": ann.get("instance_token", ""),
            }
            db_infos[det_name].append(db_entry)

        if (info_idx + 1) % 50 == 0:
            logger.info(f"  GT database: {info_idx + 1}/{len(infos)} samples processed")

    for cls, entries in db_infos.items():
        if entries:
            logger.info(f"  nuScenes GT class '{cls}': {len(entries)} samples")

    return db_infos


def _compute_nuscenes_statistics(
    infos: List[Dict[str, Any]], dataroot: str, max_samples: int = 200
) -> Dict[str, Any]:
    """Compute nuScenes dataset statistics.

    Args:
        infos: List of info dictionaries.
        dataroot: nuScenes root directory.
        max_samples: Maximum samples for point cloud stats.

    Returns:
        Statistics dictionary.
    """
    stats = {
        "point_cloud": {"xyz_min": None, "xyz_max": None, "xyz_mean": None, "num_points": []},
        "class_distribution": collections.Counter(),
        "box_size_stats": {},
    }

    # Point cloud statistics
    sample_indices = np.random.choice(
        len(infos), min(max_samples, len(infos)), replace=False
    )

    all_mins = []
    all_maxs = []

    for i in sample_indices:
        lidar_path = infos[i]["lidar_path"]
        if not os.path.exists(lidar_path):
            continue

        points = np.fromfile(lidar_path, dtype=np.float32)
        num_features = 5 if points.size % 5 == 0 else 4
        points = points.reshape(-1, num_features)

        stats["point_cloud"]["num_points"].append(len(points))
        all_mins.append(points[:, :3].min(axis=0))
        all_maxs.append(points[:, :3].max(axis=0))

    if all_mins:
        stats["point_cloud"]["xyz_min"] = np.array(all_mins).min(axis=0).tolist()
        stats["point_cloud"]["xyz_max"] = np.array(all_maxs).max(axis=0).tolist()
        stats["point_cloud"]["num_points_mean"] = float(
            np.mean(stats["point_cloud"]["num_points"])
        )

    # Class distribution and box sizes
    box_sizes = {cls: [] for cls in NUSCENES_CLASSES}

    for info in infos:
        for ann in info["annotations"]:
            det_name = ann.get("detection_name")
            if det_name:
                stats["class_distribution"][det_name] += 1
                box_sizes[det_name].append(ann["size"].tolist())

    for cls in NUSCENES_CLASSES:
        if box_sizes[cls]:
            sizes_arr = np.array(box_sizes[cls])
            stats["box_size_stats"][cls] = {
                "mean": sizes_arr.mean(axis=0).tolist(),
                "std": sizes_arr.std(axis=0).tolist(),
                "count": len(box_sizes[cls]),
            }

    return stats


def _print_nuscenes_statistics(stats: Dict[str, Any]) -> None:
    """Pretty-print nuScenes statistics."""
    logger.info("=" * 60)
    logger.info("nuScenes Dataset Statistics")
    logger.info("=" * 60)

    pc_stats = stats["point_cloud"]
    if pc_stats["xyz_min"] is not None:
        logger.info(f"Point Cloud XYZ range:")
        logger.info(f"  X: [{pc_stats['xyz_min'][0]:.2f}, {pc_stats['xyz_max'][0]:.2f}]")
        logger.info(f"  Y: [{pc_stats['xyz_min'][1]:.2f}, {pc_stats['xyz_max'][1]:.2f}]")
        logger.info(f"  Z: [{pc_stats['xyz_min'][2]:.2f}, {pc_stats['xyz_max'][2]:.2f}]")
        logger.info(f"  Mean points/scan: {pc_stats['num_points_mean']:.0f}")

    logger.info(f"\nClass distribution:")
    for cls, count in sorted(stats["class_distribution"].items(), key=lambda x: -x[1]):
        logger.info(f"  {cls:25s}: {count:6d}")

    logger.info(f"\nBox size statistics (dx, dy, dz):")
    for cls, box_stats in stats.get("box_size_stats", {}).items():
        if box_stats:
            logger.info(
                f"  {cls:25s}: mean=[{box_stats['mean'][0]:.2f}, "
                f"{box_stats['mean'][1]:.2f}, {box_stats['mean'][2]:.2f}] "
                f"(n={box_stats['count']})"
            )


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare KITTI and nuScenes datasets for PointPillars training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py --dataset kitti --data-root ./data
  python prepare_data.py --dataset nuscenes --data-root ./data --version v1.0-mini
  python prepare_data.py --dataset all --data-root ./data --workers 8
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["kitti", "nuscenes", "all"],
        help="Which dataset to prepare.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing dataset subdirectories (kitti/, nuscenes/).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="nuScenes version (default: v1.0-mini).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="KITTI classes to include (default: Car Pedestrian Cyclist).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for data preparation."""
    args = parse_arguments()

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("PointPillars Data Preparation")
    logger.info("=" * 60)
    logger.info(f"Dataset:   {args.dataset}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Workers:   {args.workers}")

    if not os.path.isdir(args.data_root):
        logger.error(f"Data root does not exist: {args.data_root}")
        sys.exit(1)

    if args.dataset in ("kitti", "all"):
        kitti_dir = os.path.join(args.data_root, "kitti")
        if not os.path.isdir(kitti_dir):
            logger.error(f"KITTI directory not found: {kitti_dir}")
            if args.dataset == "kitti":
                sys.exit(1)
        else:
            logger.info("\n--- Preparing KITTI dataset ---")
            prepare_kitti_dataset(args.data_root, args.workers, args.classes)

    if args.dataset in ("nuscenes", "all"):
        nuscenes_dir = os.path.join(args.data_root, "nuscenes")
        if not os.path.isdir(nuscenes_dir):
            logger.error(f"nuScenes directory not found: {nuscenes_dir}")
            if args.dataset == "nuscenes":
                sys.exit(1)
        else:
            logger.info("\n--- Preparing nuScenes dataset ---")
            prepare_nuscenes_dataset(args.data_root, args.version, args.workers)

    elapsed = time.time() - start_time
    logger.info(f"\nData preparation completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
