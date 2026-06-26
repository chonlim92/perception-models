"""
NuScenes radar dataset for RadarPillarNet training and evaluation.

Loads multi-sweep radar point clouds from the nuScenes dataset, applies ego-motion
compensation to align historical sweeps to the current keyframe, and provides
data augmentation for 3D object detection training.

Point features per sample: [x, y, z, rcs, vr_compensated, time_delta] (6 dims)
The pillar encoder later augments these with [x_c, y_c, z_c] offsets to get 9 dims.

Ground truth format:
    - gt_boxes: (N, 7) [x, y, z, w, l, h, yaw] in ego frame
    - gt_labels: (N,) integer class indices (0-indexed)
    - gt_velocity: (N, 2) [vx, vy] in ego frame

Reference:
    Caesar et al., "nuScenes: A multimodal dataset for autonomous driving", CVPR 2020
"""

from __future__ import annotations

import copy
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .pillar_encoder import create_pillars
from .radar_preprocessing import (
    RadarSweep,
    compensate_ego_motion,
    accumulate_sweeps,
)


# =============================================================================
# Constants
# =============================================================================

POINT_CLOUD_RANGE: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
PILLAR_SIZE: List[float] = [0.4, 0.4, 8.0]
MAX_POINTS_PER_PILLAR: int = 20
MAX_PILLARS: int = 12000

# nuScenes class mapping for radar detection
NUSCENES_RADAR_CLASSES: Dict[str, int] = {
    "car": 0,
    "truck": 1,
    "pedestrian": 2,
    "bicycle": 3,
}

# Inverse mapping
NUSCENES_RADAR_CLASS_NAMES: List[str] = ["car", "truck", "pedestrian", "bicycle"]

# nuScenes radar channels (5 radar sensors)
RADAR_CHANNELS: List[str] = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]


# =============================================================================
# Helper functions for nuScenes data loading
# =============================================================================


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a 3x3 rotation matrix.

    Uses the Hamilton convention: q = [w, x, y, z].

    Args:
        quaternion: (4,) array [w, x, y, z] representing a unit quaternion.

    Returns:
        (3, 3) rotation matrix.
    """
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    # Precompute repeated terms
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotation = np.array(
        [
            [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
        ],
        dtype=np.float64,
    )
    return rotation


def make_transform_matrix(
    translation: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """Construct a 4x4 SE(3) transformation matrix from translation and quaternion.

    Args:
        translation: (3,) translation vector [x, y, z].
        rotation: (4,) quaternion [w, x, y, z].

    Returns:
        (4, 4) homogeneous transformation matrix.
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(rotation)
    transform[:3, 3] = translation
    return transform


def load_radar_pointcloud(
    filepath: str,
) -> np.ndarray:
    """Load a nuScenes radar point cloud from a .pcd binary file.

    nuScenes radar .pcd files contain 18 features per point:
        [x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp,
         is_quality_valid, ambig_state, x_rms, y_rms, invalid_state,
         pdh0, vx_rms, vy_rms]

    We extract and return: [x, y, z, rcs, vr_compensated, vr_raw]
    where vr_compensated = sqrt(vx_comp^2 + vy_comp^2) with sign from radial direction
    and vr_raw = sqrt(vx^2 + vy^2) with sign from radial direction.

    Args:
        filepath: Path to the .pcd.bin file.

    Returns:
        (N, 6) array with columns [x, y, z, rcs, vr_compensated, vr_raw].
        Returns empty (0, 6) array if file cannot be loaded.
    """
    if not os.path.exists(filepath):
        return np.zeros((0, 6), dtype=np.float32)

    # nuScenes stores radar as flat binary with 18 float32 per point
    points = np.fromfile(filepath, dtype=np.float32)

    if points.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Reshape to (N, 18)
    n_features = 18
    if points.size % n_features != 0:
        # Try with 5 features (older format or different storage)
        n_features = 5
        if points.size % n_features != 0:
            return np.zeros((0, 6), dtype=np.float32)

    points = points.reshape(-1, n_features)
    n_points = points.shape[0]

    if n_features == 18:
        # Standard nuScenes radar format
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        rcs = points[:, 5]
        vx_raw = points[:, 6]
        vy_raw = points[:, 7]
        vx_comp = points[:, 8]
        vy_comp = points[:, 9]

        # Compute radial velocity (signed projection onto radial direction)
        range_xy = np.sqrt(x ** 2 + y ** 2).clip(min=1e-6)
        # Radial direction unit vector in x-y plane
        rad_x = x / range_xy
        rad_y = y / range_xy

        # Project velocity onto radial direction for scalar radial velocity
        vr_compensated = vx_comp * rad_x + vy_comp * rad_y
        vr_raw = vx_raw * rad_x + vy_raw * rad_y

        result = np.column_stack([x, y, z, rcs, vr_compensated, vr_raw])
    else:
        # Minimal format: [x, y, z, rcs, vr]
        result = np.zeros((n_points, 6), dtype=np.float32)
        result[:, :n_features] = points[:, :min(n_features, 6)]

    return result.astype(np.float32)


def load_nuscenes_infos(info_path: str) -> List[Dict[str, Any]]:
    """Load pre-processed nuScenes info file.

    The info file is a pickle containing a list of sample dicts with fields:
        - token: str, sample token
        - timestamp: float, sample timestamp in microseconds
        - ego_pose: dict with 'translation' (3,) and 'rotation' (4,)
        - radar_paths: dict mapping channel name -> file path
        - radar_sweeps: dict mapping channel name -> list of sweep dicts
        - gt_boxes: (N, 7) array [x, y, z, w, l, h, yaw]
        - gt_names: list of N class name strings
        - gt_velocity: (N, 2) array [vx, vy]

    Args:
        info_path: Path to the .pkl info file.

    Returns:
        List of sample info dictionaries.
    """
    with open(info_path, "rb") as f:
        infos = pickle.load(f)
    return infos


def get_sweep_transform(
    sweep_info: Dict[str, Any],
    current_ego_pose: np.ndarray,
    current_sensor_calibration: np.ndarray,
) -> np.ndarray:
    """Compute the transformation from a sweep's sensor frame to the current ego frame.

    Transform chain: sweep_sensor -> sweep_ego -> global -> current_ego -> current_sensor
    For BEV detection we stop at current_ego (no need to go back to sensor frame).

    Simplified: T = inv(current_ego_pose) @ sweep_ego_pose @ sweep_sensor_calibration

    Args:
        sweep_info: Dictionary containing sweep ego_pose and calibrated_sensor transforms.
        current_ego_pose: (4, 4) current keyframe ego-to-global transform.
        current_sensor_calibration: (4, 4) sensor-to-ego calibration for the current frame.

    Returns:
        (4, 4) transformation matrix from sweep sensor frame to current ego frame.
    """
    # Sweep sensor -> sweep ego
    sweep_sensor_to_ego = make_transform_matrix(
        np.array(sweep_info["sensor2ego_translation"], dtype=np.float64),
        np.array(sweep_info["sensor2ego_rotation"], dtype=np.float64),
    )

    # Sweep ego -> global
    sweep_ego_to_global = make_transform_matrix(
        np.array(sweep_info["ego2global_translation"], dtype=np.float64),
        np.array(sweep_info["ego2global_rotation"], dtype=np.float64),
    )

    # Current ego -> global (invert to get global -> current ego)
    global_to_current_ego = np.linalg.inv(current_ego_pose)

    # Full chain: sweep_sensor -> sweep_ego -> global -> current_ego
    transform = global_to_current_ego @ sweep_ego_to_global @ sweep_sensor_to_ego

    return transform


def transform_points_to_ego(
    points: np.ndarray,
    sensor_calibration: np.ndarray,
) -> np.ndarray:
    """Transform points from sensor frame to ego-vehicle frame.

    Applies the sensor-to-ego calibration matrix to point positions.
    Velocities (radial) are preserved as-is since they are sensor-relative scalars.

    Args:
        points: (N, 6) array [x, y, z, rcs, vr_comp, vr_raw] in sensor frame.
        sensor_calibration: (4, 4) sensor-to-ego transformation matrix.

    Returns:
        (N, 6) array with positions in ego frame, other features preserved.
    """
    if points.shape[0] == 0:
        return points.copy()

    result = points.copy()

    # Transform positions
    positions = np.ones((points.shape[0], 4), dtype=np.float64)
    positions[:, :3] = points[:, :3]
    transformed = (sensor_calibration @ positions.T).T  # (N, 4)
    result[:, :3] = transformed[:, :3].astype(np.float32)

    return result


# =============================================================================
# Data augmentation
# =============================================================================


class DataAugmentor:
    """Applies data augmentation to radar point clouds and ground truth boxes.

    Supported augmentations:
    - Random horizontal flip (along x-axis and/or y-axis)
    - Global rotation around z-axis
    - Global scaling
    - Ground truth sampling (copy-paste augmentation)

    All augmentations are applied consistently to both points and boxes.
    """

    def __init__(
        self,
        enable_flip: bool = True,
        enable_rotation: bool = True,
        enable_scaling: bool = True,
        enable_gt_sampling: bool = True,
        rotation_range: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        flip_probability: float = 0.5,
        gt_database_path: Optional[str] = None,
        max_samples_per_class: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize data augmentor.

        Args:
            enable_flip: Whether to apply random flipping.
            enable_rotation: Whether to apply global rotation.
            enable_scaling: Whether to apply global scaling.
            enable_gt_sampling: Whether to apply ground truth sampling.
            rotation_range: (min_angle, max_angle) in radians for rotation.
            scale_range: (min_scale, max_scale) for uniform scaling.
            flip_probability: Probability of applying each flip axis.
            gt_database_path: Path to ground truth database pickle file for sampling.
            max_samples_per_class: Max GT samples to paste per class.
                Defaults: {'car': 15, 'truck': 3, 'pedestrian': 10, 'bicycle': 10}
        """
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.enable_scaling = enable_scaling
        self.enable_gt_sampling = enable_gt_sampling
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_probability = flip_probability
        self.gt_database_path = gt_database_path

        if max_samples_per_class is None:
            max_samples_per_class = {
                "car": 15,
                "truck": 3,
                "pedestrian": 10,
                "bicycle": 10,
            }
        self.max_samples_per_class = max_samples_per_class

        # Load GT database if available
        self._gt_database: Optional[Dict[str, List[Dict[str, Any]]]] = None
        if enable_gt_sampling and gt_database_path and os.path.exists(gt_database_path):
            self._load_gt_database(gt_database_path)

    def _load_gt_database(self, db_path: str) -> None:
        """Load ground truth database for copy-paste augmentation.

        The GT database is a pickle file containing a dict:
            class_name -> list of {
                'box': (7,) [x, y, z, w, l, h, yaw],
                'points': (M, 6) radar points within the box,
                'velocity': (2,) [vx, vy],
                'num_points': int
            }

        Args:
            db_path: Path to the GT database pickle file.
        """
        with open(db_path, "rb") as f:
            self._gt_database = pickle.load(f)

    def _check_box_collision(
        self,
        new_box: np.ndarray,
        existing_boxes: np.ndarray,
        margin: float = 0.5,
    ) -> bool:
        """Check if a new box collides with existing boxes (BEV axis-aligned check).

        Uses a conservative axis-aligned bounding box overlap test in BEV.

        Args:
            new_box: (7,) [x, y, z, w, l, h, yaw] new box to check.
            existing_boxes: (N, 7) existing boxes.
            margin: Extra margin in meters to prevent near-misses.

        Returns:
            True if collision detected (box should not be placed).
        """
        if existing_boxes.shape[0] == 0:
            return False

        # Use maximum of w, l as radius for conservative BEV check
        new_radius = max(new_box[3], new_box[4]) / 2.0 + margin
        existing_radii = np.maximum(existing_boxes[:, 3], existing_boxes[:, 4]) / 2.0 + margin

        # BEV distance
        dx = existing_boxes[:, 0] - new_box[0]
        dy = existing_boxes[:, 1] - new_box[1]
        distances = np.sqrt(dx ** 2 + dy ** 2)

        # Collision if any distance is less than sum of radii
        min_distances = new_radius + existing_radii
        return bool(np.any(distances < min_distances))

    def sample_ground_truth(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        gt_velocity: np.ndarray,
        gt_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Sample ground truth objects from database and paste into the scene.

        For each class, samples up to max_samples_per_class objects from the
        GT database and places them in the scene if they don't collide with
        existing objects.

        Args:
            points: (N, 6) point cloud.
            gt_boxes: (M, 7) ground truth boxes.
            gt_labels: (M,) class labels.
            gt_velocity: (M, 2) ground truth velocities.
            gt_names: List of M class name strings.

        Returns:
            Tuple of augmented (points, gt_boxes, gt_labels, gt_velocity, gt_names).
        """
        if self._gt_database is None:
            return points, gt_boxes, gt_labels, gt_velocity, gt_names

        new_points_list = [points]
        new_boxes_list = [gt_boxes]
        new_labels_list = [gt_labels]
        new_velocity_list = [gt_velocity]
        new_names_list = list(gt_names)

        # Current combined boxes for collision checking
        combined_boxes = gt_boxes.copy()

        for class_name, max_samples in self.max_samples_per_class.items():
            if class_name not in self._gt_database:
                continue

            db_entries = self._gt_database[class_name]
            if len(db_entries) == 0:
                continue

            class_idx = NUSCENES_RADAR_CLASSES.get(class_name)
            if class_idx is None:
                continue

            # Count existing objects of this class
            existing_count = np.sum(gt_labels == class_idx) if gt_labels.size > 0 else 0
            num_to_sample = max(0, max_samples - int(existing_count))

            if num_to_sample == 0:
                continue

            # Randomly sample from database
            sample_indices = np.random.choice(
                len(db_entries), min(num_to_sample, len(db_entries)), replace=False
            )

            for idx in sample_indices:
                entry = db_entries[idx]
                sampled_box = np.array(entry["box"], dtype=np.float32)
                sampled_points = np.array(entry["points"], dtype=np.float32)
                sampled_velocity = np.array(
                    entry.get("velocity", [0.0, 0.0]), dtype=np.float32
                )

                # Check for collision with existing boxes
                if self._check_box_collision(sampled_box, combined_boxes):
                    continue

                # Check point cloud range
                if (
                    sampled_box[0] < POINT_CLOUD_RANGE[0]
                    or sampled_box[0] > POINT_CLOUD_RANGE[3]
                    or sampled_box[1] < POINT_CLOUD_RANGE[1]
                    or sampled_box[1] > POINT_CLOUD_RANGE[4]
                ):
                    continue

                # Add sampled object
                if sampled_points.shape[0] > 0:
                    new_points_list.append(sampled_points)
                new_boxes_list.append(sampled_box.reshape(1, 7))
                new_labels_list.append(np.array([class_idx], dtype=np.int64))
                new_velocity_list.append(sampled_velocity.reshape(1, 2))
                new_names_list.append(class_name)

                # Update combined boxes for subsequent collision checks
                combined_boxes = np.vstack([combined_boxes, sampled_box.reshape(1, 7)])

        # Concatenate results
        aug_points = np.concatenate(new_points_list, axis=0)
        aug_boxes = (
            np.concatenate(new_boxes_list, axis=0)
            if len(new_boxes_list) > 1
            else gt_boxes
        )
        aug_labels = (
            np.concatenate(new_labels_list, axis=0)
            if len(new_labels_list) > 1
            else gt_labels
        )
        aug_velocity = (
            np.concatenate(new_velocity_list, axis=0)
            if len(new_velocity_list) > 1
            else gt_velocity
        )

        return aug_points, aug_boxes, aug_labels, aug_velocity, new_names_list

    def random_flip(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        gt_velocity: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random horizontal flipping along x and/or y axes.

        When flipping along x: negate x coordinates and yaw, negate vx.
        When flipping along y: negate y coordinates and yaw, negate vy.

        Args:
            points: (N, 6) point cloud [x, y, z, rcs, vr_comp, dt].
            gt_boxes: (M, 7) ground truth boxes [x, y, z, w, l, h, yaw].
            gt_velocity: (M, 2) ground truth velocities [vx, vy].

        Returns:
            Tuple of augmented (points, gt_boxes, gt_velocity).
        """
        # Flip along x-axis (left-right)
        if np.random.random() < self.flip_probability:
            points[:, 1] = -points[:, 1]
            if gt_boxes.shape[0] > 0:
                gt_boxes[:, 1] = -gt_boxes[:, 1]
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            if gt_velocity.shape[0] > 0:
                gt_velocity[:, 1] = -gt_velocity[:, 1]

        # Flip along y-axis (front-back)
        if np.random.random() < self.flip_probability:
            points[:, 0] = -points[:, 0]
            if gt_boxes.shape[0] > 0:
                gt_boxes[:, 0] = -gt_boxes[:, 0]
                gt_boxes[:, 6] = -(gt_boxes[:, 6] - np.pi)
            if gt_velocity.shape[0] > 0:
                gt_velocity[:, 0] = -gt_velocity[:, 0]

        return points, gt_boxes, gt_velocity

    def global_rotation(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        gt_velocity: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random global rotation around the z-axis.

        Rotates all point coordinates, box centers, box yaw, and velocity vectors
        by a uniformly sampled angle.

        Args:
            points: (N, 6) point cloud [x, y, z, rcs, vr_comp, dt].
            gt_boxes: (M, 7) ground truth boxes [x, y, z, w, l, h, yaw].
            gt_velocity: (M, 2) ground truth velocities [vx, vy].

        Returns:
            Tuple of augmented (points, gt_boxes, gt_velocity).
        """
        angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rotation matrix for x-y plane
        rot_matrix = np.array(
            [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32
        )

        # Rotate point positions (x, y)
        points[:, :2] = points[:, :2] @ rot_matrix.T

        # Rotate box centers and yaw
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :2] = gt_boxes[:, :2] @ rot_matrix.T
            gt_boxes[:, 6] += angle

        # Rotate velocity vectors
        if gt_velocity.shape[0] > 0:
            gt_velocity[:, :2] = gt_velocity[:, :2] @ rot_matrix.T

        return points, gt_boxes, gt_velocity

    def global_scaling(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        gt_velocity: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random global uniform scaling.

        Scales point positions, box centers, box dimensions, and velocity magnitudes.

        Args:
            points: (N, 6) point cloud [x, y, z, rcs, vr_comp, dt].
            gt_boxes: (M, 7) ground truth boxes [x, y, z, w, l, h, yaw].
            gt_velocity: (M, 2) ground truth velocities [vx, vy].

        Returns:
            Tuple of augmented (points, gt_boxes, gt_velocity).
        """
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Scale point positions
        points[:, :3] *= scale

        # Scale box centers and dimensions (not yaw)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] *= scale  # center position
            gt_boxes[:, 3:6] *= scale  # dimensions (w, l, h)

        # Scale velocity magnitudes
        if gt_velocity.shape[0] > 0:
            gt_velocity *= scale

        return points, gt_boxes, gt_velocity

    def __call__(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        gt_velocity: np.ndarray,
        gt_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Apply all enabled augmentations sequentially.

        Order: GT sampling -> flip -> rotation -> scaling.

        Args:
            points: (N, 6) point cloud.
            gt_boxes: (M, 7) ground truth boxes.
            gt_labels: (M,) class labels.
            gt_velocity: (M, 2) ground truth velocities.
            gt_names: List of M class name strings.

        Returns:
            Tuple of augmented (points, gt_boxes, gt_labels, gt_velocity, gt_names).
        """
        # GT sampling (must be applied first, before geometric augmentations)
        if self.enable_gt_sampling:
            points, gt_boxes, gt_labels, gt_velocity, gt_names = (
                self.sample_ground_truth(
                    points, gt_boxes, gt_labels, gt_velocity, gt_names
                )
            )

        # Random flip
        if self.enable_flip:
            points, gt_boxes, gt_velocity = self.random_flip(
                points, gt_boxes, gt_velocity
            )

        # Global rotation
        if self.enable_rotation:
            points, gt_boxes, gt_velocity = self.global_rotation(
                points, gt_boxes, gt_velocity
            )

        # Global scaling
        if self.enable_scaling:
            points, gt_boxes, gt_velocity = self.global_scaling(
                points, gt_boxes, gt_velocity
            )

        return points, gt_boxes, gt_labels, gt_velocity, gt_names


# =============================================================================
# Main dataset class
# =============================================================================


class NuScenesRadarDataset(Dataset):
    """NuScenes radar point cloud dataset for 3D object detection.

    Loads multi-sweep accumulated radar point clouds from the nuScenes dataset
    with ego-motion compensation. Provides data augmentation for training and
    returns pre-processed samples ready for pillarization in the collate function.

    The dataset expects pre-generated info files (pickle format) containing
    metadata about each sample, including file paths, ego poses, calibrations,
    and ground truth annotations.

    Each sample returns:
        - points: (N, 6) float32 [x, y, z, rcs, vr_compensated, time_delta]
        - gt_boxes: (M, 7) float32 [x, y, z, w, l, h, yaw]
        - gt_labels: (M,) int64 class indices
        - gt_velocity: (M, 2) float32 [vx, vy]
        - metadata: dict with sample token, timestamp, etc.
    """

    def __init__(
        self,
        data_root: str,
        info_path: str,
        split: str = "train",
        num_sweeps: int = 6,
        point_cloud_range: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        augmentor: Optional[DataAugmentor] = None,
        min_points_in_gt: int = 1,
        load_interval: int = 1,
    ) -> None:
        """Initialize the nuScenes radar dataset.

        Args:
            data_root: Root directory of the nuScenes dataset (contains 'sweeps/', 'samples/', etc.).
            info_path: Path to the pre-generated info pickle file.
            split: Dataset split ('train', 'val', or 'test').
            num_sweeps: Number of radar sweeps to accumulate (including current keyframe).
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] detection range.
                Defaults to [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0].
            class_names: List of class names to detect. Defaults to
                ['car', 'truck', 'pedestrian', 'bicycle'].
            augmentor: Data augmentation module. Pass None to disable augmentation.
            min_points_in_gt: Minimum radar points required in a GT box to keep it.
                Boxes with fewer points are filtered out during training.
            load_interval: Sub-sampling interval for dataset (1 = use all samples).
        """
        self.data_root = data_root
        self.split = split
        self.num_sweeps = num_sweeps
        self.min_points_in_gt = min_points_in_gt
        self.augmentor = augmentor

        if point_cloud_range is None:
            point_cloud_range = POINT_CLOUD_RANGE.copy()
        self.point_cloud_range = point_cloud_range

        if class_names is None:
            class_names = NUSCENES_RADAR_CLASS_NAMES.copy()
        self.class_names = class_names

        # Build class name to index mapping
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        # Load info file
        self.infos = load_nuscenes_infos(info_path)

        # Apply load interval (subsampling)
        if load_interval > 1:
            self.infos = self.infos[::load_interval]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.infos)

    def _load_radar_sweeps(
        self, info: Dict[str, Any]
    ) -> np.ndarray:
        """Load and accumulate multi-sweep radar point cloud for a sample.

        Loads the current keyframe radar data from all radar sensors, plus
        historical sweeps, transforms everything to the current ego frame,
        and concatenates with time deltas.

        Args:
            info: Sample info dictionary from the info file.

        Returns:
            (N, 7) array [x, y, z, rcs, vr_compensated, vr_raw, time_delta]
            in the current ego-vehicle frame.
        """
        # Current ego pose (ego -> global)
        ego_translation = np.array(
            info["ego2global_translation"], dtype=np.float64
        )
        ego_rotation = np.array(info["ego2global_rotation"], dtype=np.float64)
        current_ego_pose = make_transform_matrix(ego_translation, ego_rotation)
        current_timestamp = info["timestamp"] / 1e6  # Convert microseconds to seconds

        all_points = []

        # Load current keyframe from all radar channels
        for channel in RADAR_CHANNELS:
            # Get radar data path
            radar_info = info.get("radars", {}).get(channel, None)
            if radar_info is None:
                # Try alternative info format
                radar_path = info.get("radar_paths", {}).get(channel, None)
                if radar_path is None:
                    continue
                radar_info = {"data_path": radar_path}

            # Load current keyframe points
            radar_path = radar_info.get("data_path", "")
            if not os.path.isabs(radar_path):
                radar_path = os.path.join(self.data_root, radar_path)

            points = load_radar_pointcloud(radar_path)
            if points.shape[0] == 0:
                continue

            # Get sensor calibration (sensor -> ego)
            sensor_translation = np.array(
                radar_info.get(
                    "sensor2ego_translation",
                    info.get("radar_calibrations", {}).get(channel, {}).get(
                        "sensor2ego_translation", [0.0, 0.0, 0.0]
                    ),
                ),
                dtype=np.float64,
            )
            sensor_rotation = np.array(
                radar_info.get(
                    "sensor2ego_rotation",
                    info.get("radar_calibrations", {}).get(channel, {}).get(
                        "sensor2ego_rotation", [1.0, 0.0, 0.0, 0.0]
                    ),
                ),
                dtype=np.float64,
            )
            sensor_calibration = make_transform_matrix(
                sensor_translation, sensor_rotation
            )

            # Transform to ego frame
            points_ego = transform_points_to_ego(points, sensor_calibration)

            # Add time delta = 0 for current keyframe
            points_with_dt = np.column_stack(
                [points_ego, np.zeros(points_ego.shape[0], dtype=np.float32)]
            )
            all_points.append(points_with_dt)

            # Load historical sweeps for this channel
            sweeps_info = radar_info.get("sweeps", [])
            if not sweeps_info:
                sweeps_info = info.get("radar_sweeps", {}).get(channel, [])

            num_history = min(self.num_sweeps - 1, len(sweeps_info))
            for s_idx in range(num_history):
                sweep = sweeps_info[s_idx]

                sweep_path = sweep.get("data_path", "")
                if not os.path.isabs(sweep_path):
                    sweep_path = os.path.join(self.data_root, sweep_path)

                sweep_points = load_radar_pointcloud(sweep_path)
                if sweep_points.shape[0] == 0:
                    continue

                # Compute transform from sweep sensor frame to current ego frame
                sweep_transform = get_sweep_transform(
                    sweep, current_ego_pose, sensor_calibration
                )

                # Transform sweep points to current ego frame
                sweep_points_ego = transform_points_to_ego(
                    sweep_points, sweep_transform
                )

                # Compute time delta
                sweep_timestamp = sweep.get("timestamp", 0) / 1e6
                time_delta = current_timestamp - sweep_timestamp

                # Append time delta
                sweep_with_dt = np.column_stack(
                    [
                        sweep_points_ego,
                        np.full(
                            sweep_points_ego.shape[0], time_delta, dtype=np.float32
                        ),
                    ]
                )
                all_points.append(sweep_with_dt)

        # Concatenate all points
        if len(all_points) == 0:
            return np.zeros((0, 7), dtype=np.float32)

        accumulated = np.concatenate(all_points, axis=0).astype(np.float32)
        return accumulated

    def _filter_points_by_range(self, points: np.ndarray) -> np.ndarray:
        """Filter points to keep only those within the detection range.

        Args:
            points: (N, D) array where first 3 columns are x, y, z.

        Returns:
            (M, D) array with M <= N points within range.
        """
        mask = (
            (points[:, 0] >= self.point_cloud_range[0])
            & (points[:, 0] <= self.point_cloud_range[3])
            & (points[:, 1] >= self.point_cloud_range[1])
            & (points[:, 1] <= self.point_cloud_range[4])
            & (points[:, 2] >= self.point_cloud_range[2])
            & (points[:, 2] <= self.point_cloud_range[5])
        )
        return points[mask]

    def _get_gt_annotations(
        self, info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Extract ground truth annotations from sample info.

        Filters annotations to keep only classes in self.class_names.

        Args:
            info: Sample info dictionary.

        Returns:
            Tuple of:
                gt_boxes: (M, 7) array [x, y, z, w, l, h, yaw]
                gt_labels: (M,) array of integer class indices
                gt_velocity: (M, 2) array [vx, vy]
                gt_names: list of M class name strings
        """
        gt_boxes = np.array(info.get("gt_boxes", []), dtype=np.float32)
        gt_names = info.get("gt_names", [])
        gt_velocity = np.array(
            info.get("gt_velocity", []), dtype=np.float32
        )

        if len(gt_boxes) == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
                [],
            )

        # Ensure correct shapes
        if gt_boxes.ndim == 1:
            gt_boxes = gt_boxes.reshape(-1, 7)
        if gt_velocity.ndim == 1 and gt_velocity.size > 0:
            gt_velocity = gt_velocity.reshape(-1, 2)
        if gt_velocity.shape[0] == 0:
            gt_velocity = np.zeros((gt_boxes.shape[0], 2), dtype=np.float32)

        # Filter to target classes
        keep_mask = np.array(
            [name in self.class_to_idx for name in gt_names], dtype=bool
        )

        if not np.any(keep_mask):
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
                [],
            )

        gt_boxes = gt_boxes[keep_mask]
        gt_velocity = gt_velocity[keep_mask]
        gt_names = [name for name, keep in zip(gt_names, keep_mask) if keep]
        gt_labels = np.array(
            [self.class_to_idx[name] for name in gt_names], dtype=np.int64
        )

        return gt_boxes, gt_labels, gt_velocity, gt_names

    def _filter_gt_by_points(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        gt_velocity: np.ndarray,
        gt_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Remove ground truth boxes that contain fewer radar points than the threshold.

        Uses a simplified check: counts points within the axis-aligned bounding box
        of each rotated GT box (conservative, may count extra points).

        Args:
            points: (N, 6) point cloud [x, y, z, ...].
            gt_boxes: (M, 7) boxes [x, y, z, w, l, h, yaw].
            gt_labels: (M,) class labels.
            gt_velocity: (M, 2) velocities.
            gt_names: List of M class names.

        Returns:
            Filtered (gt_boxes, gt_labels, gt_velocity, gt_names).
        """
        if gt_boxes.shape[0] == 0 or self.min_points_in_gt <= 0:
            return gt_boxes, gt_labels, gt_velocity, gt_names

        keep_mask = np.zeros(gt_boxes.shape[0], dtype=bool)

        for i in range(gt_boxes.shape[0]):
            cx, cy, cz = gt_boxes[i, 0], gt_boxes[i, 1], gt_boxes[i, 2]
            w, l, h, yaw = gt_boxes[i, 3], gt_boxes[i, 4], gt_boxes[i, 5], gt_boxes[i, 6]

            # Rotate points into box-local frame for accurate counting
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)

            # Translate points relative to box center
            dx = points[:, 0] - cx
            dy = points[:, 1] - cy
            dz = points[:, 2] - cz

            # Rotate into box frame
            local_x = cos_yaw * dx - sin_yaw * dy
            local_y = sin_yaw * dx + cos_yaw * dy

            # Check if inside box
            inside = (
                (np.abs(local_x) <= w / 2.0)
                & (np.abs(local_y) <= l / 2.0)
                & (np.abs(dz) <= h / 2.0)
            )
            num_points_in_box = int(np.sum(inside))
            keep_mask[i] = num_points_in_box >= self.min_points_in_gt

        gt_boxes = gt_boxes[keep_mask]
        gt_labels = gt_labels[keep_mask]
        gt_velocity = gt_velocity[keep_mask]
        gt_names = [name for name, keep in zip(gt_names, keep_mask) if keep]

        return gt_boxes, gt_labels, gt_velocity, gt_names

    def _filter_gt_by_range(
        self,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        gt_velocity: np.ndarray,
        gt_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Remove ground truth boxes whose centers are outside the detection range.

        Args:
            gt_boxes: (M, 7) boxes.
            gt_labels: (M,) labels.
            gt_velocity: (M, 2) velocities.
            gt_names: List of M names.

        Returns:
            Filtered (gt_boxes, gt_labels, gt_velocity, gt_names).
        """
        if gt_boxes.shape[0] == 0:
            return gt_boxes, gt_labels, gt_velocity, gt_names

        mask = (
            (gt_boxes[:, 0] >= self.point_cloud_range[0])
            & (gt_boxes[:, 0] <= self.point_cloud_range[3])
            & (gt_boxes[:, 1] >= self.point_cloud_range[1])
            & (gt_boxes[:, 1] <= self.point_cloud_range[4])
            & (gt_boxes[:, 2] >= self.point_cloud_range[2])
            & (gt_boxes[:, 2] <= self.point_cloud_range[5])
        )

        gt_boxes = gt_boxes[mask]
        gt_labels = gt_labels[mask]
        gt_velocity = gt_velocity[mask]
        gt_names = [name for name, keep in zip(gt_names, mask) if keep]

        return gt_boxes, gt_labels, gt_velocity, gt_names

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single training/validation sample.

        Args:
            index: Sample index.

        Returns:
            Dict containing:
                'points': (N, 6) float32 array [x, y, z, rcs, vr_compensated, time_delta]
                'gt_boxes': (M, 7) float32 array [x, y, z, w, l, h, yaw]
                'gt_labels': (M,) int64 array of class indices
                'gt_velocity': (M, 2) float32 array [vx, vy]
                'metadata': dict with 'token', 'timestamp', 'num_points', 'num_gt_boxes'
        """
        info = copy.deepcopy(self.infos[index])

        # Load accumulated radar point cloud (N, 7): [x, y, z, rcs, vr_comp, vr_raw, dt]
        raw_points = self._load_radar_sweeps(info)

        # Filter by detection range
        raw_points = self._filter_points_by_range(raw_points)

        # Select features for model input: [x, y, z, rcs, vr_compensated, time_delta]
        # From accumulated (N, 7): drop vr_raw (index 5), keep [0,1,2,3,4,6]
        if raw_points.shape[0] > 0:
            points = raw_points[:, [0, 1, 2, 3, 4, 6]].astype(np.float32)
        else:
            points = np.zeros((0, 6), dtype=np.float32)

        # Get ground truth annotations
        gt_boxes, gt_labels, gt_velocity, gt_names = self._get_gt_annotations(info)

        # Filter GT boxes outside detection range
        gt_boxes, gt_labels, gt_velocity, gt_names = self._filter_gt_by_range(
            gt_boxes, gt_labels, gt_velocity, gt_names
        )

        # Filter GT boxes with insufficient radar points
        if self.split == "train" and self.min_points_in_gt > 0:
            gt_boxes, gt_labels, gt_velocity, gt_names = self._filter_gt_by_points(
                points, gt_boxes, gt_labels, gt_velocity, gt_names
            )

        # Apply data augmentation (training only)
        if self.augmentor is not None and self.split == "train":
            points, gt_boxes, gt_labels, gt_velocity, gt_names = self.augmentor(
                points, gt_boxes, gt_labels, gt_velocity, gt_names
            )

            # Re-filter points after augmentation (rotation/scaling may push out of range)
            points = self._filter_points_by_range(
                np.column_stack([points, np.zeros((points.shape[0], 1))])
                if points.shape[1] < 7
                else points
            )
            if points.shape[1] > 6:
                points = points[:, :6]

        # Build metadata
        metadata = {
            "token": info.get("token", ""),
            "timestamp": info.get("timestamp", 0),
            "num_points": points.shape[0],
            "num_gt_boxes": gt_boxes.shape[0],
            "sample_idx": index,
        }

        return {
            "points": points.astype(np.float32),
            "gt_boxes": gt_boxes.astype(np.float32),
            "gt_labels": gt_labels.astype(np.int64),
            "gt_velocity": gt_velocity.astype(np.float32),
            "metadata": metadata,
        }


# =============================================================================
# Collate function with pillarization
# =============================================================================


class RadarCollateFunction:
    """Custom collate function that creates pillars from variable-length point clouds.

    Handles batching of samples with different point counts by:
    1. Creating pillars for each sample independently using create_pillars()
    2. Stacking pillar tensors into fixed-size batch tensors
    3. Padding ground truth boxes/labels to the maximum count in the batch

    This class is designed to be used as the collate_fn argument to DataLoader.

    Usage:
        collate_fn = RadarCollateFunction(
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            pillar_size=[0.4, 0.4, 8.0],
            max_points_per_pillar=20,
            max_pillars=12000,
        )
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    """

    def __init__(
        self,
        point_cloud_range: Optional[List[float]] = None,
        pillar_size: Optional[List[float]] = None,
        max_points_per_pillar: int = MAX_POINTS_PER_PILLAR,
        max_pillars: int = MAX_PILLARS,
        max_gt_boxes: int = 500,
    ) -> None:
        """Initialize the collate function.

        Args:
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
                Defaults to [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0].
            pillar_size: [dx, dy, dz] in meters. Defaults to [0.4, 0.4, 8.0].
            max_points_per_pillar: Maximum points per pillar for padding/truncation.
            max_pillars: Maximum number of non-empty pillars per sample.
            max_gt_boxes: Maximum number of GT boxes to pad to in the batch.
        """
        if point_cloud_range is None:
            point_cloud_range = POINT_CLOUD_RANGE.copy()
        if pillar_size is None:
            pillar_size = PILLAR_SIZE.copy()

        self.point_cloud_range = point_cloud_range
        self.pillar_size = pillar_size
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars
        self.max_gt_boxes = max_gt_boxes

    def _pillarize_sample(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create pillars from a single sample's point cloud.

        Wraps the create_pillars() function, handling the case where points
        may have fewer than 7 columns (pads with zeros for the extra column
        expected by create_pillars).

        Args:
            points: (N, 6) float32 array [x, y, z, rcs, vr_comp, time_delta].

        Returns:
            Tuple of:
                pillars: (max_pillars, max_points_per_pillar, 9) float32
                pillar_indices: (max_pillars, 3) int32 [batch_idx, gx, gy]
                num_points_per_pillar: (max_pillars,) int32
        """
        if points.shape[0] == 0:
            pillars = np.zeros(
                (self.max_pillars, self.max_points_per_pillar, 9),
                dtype=np.float32,
            )
            pillar_indices = np.zeros((self.max_pillars, 3), dtype=np.int32)
            num_points = np.zeros(self.max_pillars, dtype=np.int32)
            return pillars, pillar_indices, num_points

        # create_pillars expects (N, 7) but only uses first 6 features.
        # Pad with zeros if needed to ensure >= 7 columns.
        if points.shape[1] < 7:
            padding = np.zeros(
                (points.shape[0], 7 - points.shape[1]), dtype=np.float32
            )
            points_padded = np.column_stack([points, padding])
        else:
            points_padded = points

        pillars, pillar_indices, num_points = create_pillars(
            points=points_padded,
            point_range=self.point_cloud_range,
            pillar_size=self.pillar_size,
            max_points_per_pillar=self.max_points_per_pillar,
            max_pillars=self.max_pillars,
        )

        return pillars, pillar_indices, num_points

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collate a list of samples into a batched tensor dict.

        Performs pillarization on each sample and stacks results.
        Pads ground truth to the maximum number of boxes in the batch.

        Args:
            batch: List of sample dicts from NuScenesRadarDataset.__getitem__().

        Returns:
            Dict containing:
                'pillars': (B, max_pillars, max_points_per_pillar, 9) float32 tensor
                'pillar_indices': (B, max_pillars, 3) int32 tensor
                'num_points_per_pillar': (B, max_pillars) int32 tensor
                'gt_boxes': (B, max_gt, 7) float32 tensor (zero-padded)
                'gt_labels': (B, max_gt) int64 tensor (-1 for padding)
                'gt_velocity': (B, max_gt, 2) float32 tensor (zero-padded)
                'num_gt_boxes': (B,) int32 tensor with actual GT count per sample
                'metadata': list of B metadata dicts
        """
        batch_size = len(batch)

        # Pillarize each sample
        all_pillars = []
        all_pillar_indices = []
        all_num_points = []

        for sample in batch:
            points = sample["points"]
            pillars, pillar_indices, num_points = self._pillarize_sample(points)
            all_pillars.append(pillars)
            all_pillar_indices.append(pillar_indices)
            all_num_points.append(num_points)

        # Stack pillars into batch tensors
        pillars_batch = torch.from_numpy(np.stack(all_pillars, axis=0))
        pillar_indices_batch = torch.from_numpy(
            np.stack(all_pillar_indices, axis=0)
        )
        num_points_batch = torch.from_numpy(np.stack(all_num_points, axis=0))

        # Determine max GT boxes in this batch (capped by max_gt_boxes)
        max_gt_in_batch = max(
            sample["gt_boxes"].shape[0] for sample in batch
        )
        max_gt = min(max_gt_in_batch, self.max_gt_boxes)

        # Pad ground truth annotations
        gt_boxes_padded = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
        gt_labels_padded = np.full((batch_size, max_gt), -1, dtype=np.int64)
        gt_velocity_padded = np.zeros((batch_size, max_gt, 2), dtype=np.float32)
        num_gt_boxes = np.zeros(batch_size, dtype=np.int32)

        for i, sample in enumerate(batch):
            n_gt = min(sample["gt_boxes"].shape[0], max_gt)
            if n_gt > 0:
                gt_boxes_padded[i, :n_gt] = sample["gt_boxes"][:n_gt]
                gt_labels_padded[i, :n_gt] = sample["gt_labels"][:n_gt]
                gt_velocity_padded[i, :n_gt] = sample["gt_velocity"][:n_gt]
            num_gt_boxes[i] = n_gt

        # Collect metadata
        metadata_list = [sample["metadata"] for sample in batch]

        return {
            "pillars": pillars_batch,
            "pillar_indices": pillar_indices_batch,
            "num_points_per_pillar": num_points_batch,
            "gt_boxes": torch.from_numpy(gt_boxes_padded),
            "gt_labels": torch.from_numpy(gt_labels_padded),
            "gt_velocity": torch.from_numpy(gt_velocity_padded),
            "num_gt_boxes": torch.from_numpy(num_gt_boxes),
            "metadata": metadata_list,
        }


# =============================================================================
# Convenience factory functions
# =============================================================================


def build_nuscenes_radar_dataset(
    data_root: str,
    info_path: str,
    split: str = "train",
    num_sweeps: int = 6,
    gt_database_path: Optional[str] = None,
    augmentation: bool = True,
    load_interval: int = 1,
) -> NuScenesRadarDataset:
    """Factory function to create a NuScenesRadarDataset with standard configuration.

    Provides sensible defaults for training and validation splits.

    Args:
        data_root: Root directory of nuScenes dataset.
        info_path: Path to the info pickle file.
        split: 'train', 'val', or 'test'.
        num_sweeps: Number of sweeps to accumulate.
        gt_database_path: Path to GT database for sampling augmentation.
        augmentation: Whether to enable data augmentation (ignored for val/test).
        load_interval: Sub-sampling interval.

    Returns:
        Configured NuScenesRadarDataset instance.
    """
    augmentor = None
    if split == "train" and augmentation:
        augmentor = DataAugmentor(
            enable_flip=True,
            enable_rotation=True,
            enable_scaling=True,
            enable_gt_sampling=gt_database_path is not None,
            rotation_range=(-np.pi / 4, np.pi / 4),
            scale_range=(0.95, 1.05),
            flip_probability=0.5,
            gt_database_path=gt_database_path,
        )

    dataset = NuScenesRadarDataset(
        data_root=data_root,
        info_path=info_path,
        split=split,
        num_sweeps=num_sweeps,
        point_cloud_range=POINT_CLOUD_RANGE.copy(),
        class_names=NUSCENES_RADAR_CLASS_NAMES.copy(),
        augmentor=augmentor,
        min_points_in_gt=1 if split == "train" else 0,
        load_interval=load_interval,
    )

    return dataset


def build_dataloader(
    dataset: NuScenesRadarDataset,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader with the radar collate function.

    Args:
        dataset: NuScenesRadarDataset instance.
        batch_size: Samples per batch.
        num_workers: Parallel data loading workers.
        shuffle: Whether to shuffle (defaults to True for train, False otherwise).
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop incomplete last batch.

    Returns:
        Configured DataLoader.
    """
    if shuffle is None:
        shuffle = dataset.split == "train"

    collate_fn = RadarCollateFunction(
        point_cloud_range=dataset.point_cloud_range,
        pillar_size=PILLAR_SIZE.copy(),
        max_points_per_pillar=MAX_POINTS_PER_PILLAR,
        max_pillars=MAX_PILLARS,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader
