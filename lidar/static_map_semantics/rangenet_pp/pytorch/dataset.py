"""SemanticKITTI dataset for RangeNet++ range image segmentation.

Loads point cloud (.bin) and label (.label) files from the SemanticKITTI
dataset, performs spherical projection, and applies data augmentation.

Dataset structure:
    dataset/
      sequences/
        00/ ... 10/ (training+validation)
          velodyne/  -> xxxxxx.bin (N, 4) float32
          labels/    -> xxxxxx.label (N,) uint32
        11/ ... 21/ (test, no labels)

Training: sequences 00-07, 09-10
Validation: sequence 08
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from .spherical_projection import SphericalProjection


# SemanticKITTI label mapping: raw label ID -> training label (0-19)
# 0 = unlabeled/ignored, 1-19 = evaluated classes
SEMANTICKITTI_LABEL_MAP = {
    0: 0,       # unlabeled
    1: 0,       # outlier
    10: 1,      # car
    11: 2,      # bicycle
    13: 5,      # bus (-> other-vehicle)
    15: 3,      # motorcycle
    16: 5,      # on-rails (-> other-vehicle)
    18: 4,      # truck
    20: 5,      # other-vehicle
    30: 6,      # person
    31: 7,      # bicyclist
    32: 8,      # motorcyclist
    40: 9,      # road
    44: 10,     # parking
    48: 11,     # sidewalk
    49: 12,     # other-ground
    50: 13,     # building
    51: 14,     # fence
    52: 0,      # other-structure (-> unlabeled)
    60: 9,      # lane-marking (-> road)
    70: 15,     # vegetation
    71: 16,     # trunk
    72: 17,     # terrain
    80: 18,     # pole
    81: 19,     # traffic-sign
    99: 0,      # other-object (-> unlabeled)
    252: 1,     # moving-car (-> car)
    253: 7,     # moving-bicyclist (-> bicyclist)
    254: 6,     # moving-person (-> person)
    255: 8,     # moving-motorcyclist (-> motorcyclist)
    256: 5,     # moving-on-rails (-> other-vehicle)
    257: 5,     # moving-bus (-> other-vehicle)
    258: 4,     # moving-truck (-> truck)
    259: 5,     # moving-other-vehicle (-> other-vehicle)
}

# Class names for the 20 training classes
SEMANTICKITTI_CLASS_NAMES = [
    "unlabeled",       # 0
    "car",             # 1
    "bicycle",         # 2
    "motorcycle",      # 3
    "truck",           # 4
    "other-vehicle",   # 5
    "person",          # 6
    "bicyclist",       # 7
    "motorcyclist",    # 8
    "road",            # 9
    "parking",         # 10
    "sidewalk",        # 11
    "other-ground",    # 12
    "building",        # 13
    "fence",           # 14
    "vegetation",      # 15
    "trunk",           # 16
    "terrain",         # 17
    "pole",            # 18
    "traffic-sign",    # 19
]

# Default split: train and validation sequences
TRAIN_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
VAL_SEQUENCES = ["08"]


def map_labels_to_training(raw_labels: np.ndarray) -> np.ndarray:
    """Map raw SemanticKITTI labels to training labels (0-19).

    Args:
        raw_labels: (N,) uint16 array of raw semantic label IDs.

    Returns:
        (N,) int32 array of training labels in range [0, 19].
    """
    mapped = np.zeros_like(raw_labels, dtype=np.int32)
    for raw_id, train_id in SEMANTICKITTI_LABEL_MAP.items():
        mapped[raw_labels == raw_id] = train_id
    return mapped


class SemanticKITTIRangeDataset(Dataset):
    """SemanticKITTI dataset producing range images for RangeNet++.

    Each sample returns:
        - range_image: (5, H, W) float32 tensor [range, x, y, z, intensity]
        - label_image: (H, W) long tensor of semantic class labels
        - mask: (H, W) bool tensor indicating valid (non-empty) pixels
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        height: int = 64,
        width: int = 2048,
        fov_up: float = 2.0,
        fov_down: float = -24.8,
        augment: bool = False,
        max_points: int = 150000,
    ):
        """
        Args:
            root: Path to SemanticKITTI dataset root (containing 'sequences/' dir).
            split: 'train' or 'val'.
            height: Range image height.
            width: Range image width.
            fov_up: Upper vertical FOV in degrees.
            fov_down: Lower vertical FOV in degrees.
            augment: Whether to apply data augmentation (training only).
            max_points: Maximum number of points to load (for memory).
        """
        super().__init__()
        self.root = root
        self.split = split
        self.height = height
        self.width = width
        self.augment = augment and (split == "train")
        self.max_points = max_points

        # Spherical projection
        self.projector = SphericalProjection(
            height=height,
            width=width,
            fov_up=fov_up,
            fov_down=fov_down,
        )

        # Determine sequences for this split
        if split == "train":
            sequences = TRAIN_SEQUENCES
        elif split == "val":
            sequences = VAL_SEQUENCES
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'val'.")

        # Collect all scan file paths
        self.scan_files: List[str] = []
        self.label_files: List[str] = []

        sequences_dir = os.path.join(root, "sequences")
        for seq in sequences:
            velodyne_dir = os.path.join(sequences_dir, seq, "velodyne")
            labels_dir = os.path.join(sequences_dir, seq, "labels")

            if not os.path.isdir(velodyne_dir):
                continue

            scan_names = sorted(os.listdir(velodyne_dir))
            for scan_name in scan_names:
                if not scan_name.endswith(".bin"):
                    continue
                scan_path = os.path.join(velodyne_dir, scan_name)
                label_name = scan_name.replace(".bin", ".label")
                label_path = os.path.join(labels_dir, label_name)

                self.scan_files.append(scan_path)
                self.label_files.append(label_path)

        # Build label lookup table for fast mapping
        self._label_lut = np.zeros(260 * 256, dtype=np.int32)
        for raw_id, train_id in SEMANTICKITTI_LABEL_MAP.items():
            self._label_lut[raw_id] = train_id

    def __len__(self) -> int:
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process a single scan.

        Returns:
            Dictionary with:
                'range_image': (5, H, W) float32 tensor
                'label_image': (H, W) long tensor
                'mask': (H, W) bool tensor (True for valid pixels)
        """
        # Load point cloud: (N, 4) float32 [x, y, z, intensity]
        points = np.fromfile(self.scan_files[idx], dtype=np.float32).reshape(-1, 4)

        # Load labels: (N,) uint32, lower 16 bits = semantic label
        if os.path.exists(self.label_files[idx]):
            labels_raw = np.fromfile(self.label_files[idx], dtype=np.uint32)
            semantic_labels = (labels_raw & 0xFFFF).astype(np.uint16)
        else:
            semantic_labels = np.zeros(points.shape[0], dtype=np.uint16)

        # Subsample if too many points
        if points.shape[0] > self.max_points:
            choice = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[choice]
            semantic_labels = semantic_labels[choice]

        # Map to training labels
        training_labels = self._label_lut[semantic_labels].astype(np.int32)

        # Apply data augmentation
        if self.augment:
            points, training_labels = self._augment(points, training_labels)

        # Project to range image
        range_image, pixel_to_point, point_to_pixel = (
            self.projector.project_points_to_range_image_fast(points)
        )

        # Create label image from projected labels
        label_image = np.zeros((self.height, self.width), dtype=np.int32)
        valid_pixels = pixel_to_point >= 0
        rows, cols = np.where(valid_pixels)
        point_indices = pixel_to_point[rows, cols]
        label_image[rows, cols] = training_labels[point_indices]

        # Valid pixel mask (non-empty)
        mask = range_image[0] > 0

        # Normalize range image channels for network input
        range_image_normalized = self._normalize_range_image(range_image)

        # Convert to tensors
        range_tensor = torch.from_numpy(range_image_normalized).float()
        label_tensor = torch.from_numpy(label_image).long()
        mask_tensor = torch.from_numpy(mask).bool()

        return {
            "range_image": range_tensor,
            "label_image": label_tensor,
            "mask": mask_tensor,
        }

    def _normalize_range_image(self, range_image: np.ndarray) -> np.ndarray:
        """Normalize range image channels to reasonable ranges.

        Channel normalization:
            - range: divide by max_range (80m)
            - x, y, z: divide by max_range (80m)
            - intensity: already in [0, 1] typically, clamp to [0, 1]

        Args:
            range_image: (5, H, W) raw range image.

        Returns:
            (5, H, W) normalized range image.
        """
        normalized = range_image.copy()
        max_range = 80.0

        # Range channel
        normalized[0] = normalized[0] / max_range

        # XYZ channels
        normalized[1] = normalized[1] / max_range
        normalized[2] = normalized[2] / max_range
        normalized[3] = normalized[3] / max_range

        # Intensity: clip to [0, 1]
        normalized[4] = np.clip(normalized[4], 0.0, 1.0)

        return normalized

    def _augment(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation.

        Augmentations:
            1. Random rotation around z-axis (yaw)
            2. Random point dropout (5% of points)
            3. Intensity noise (Gaussian, sigma=0.02)

        Args:
            points: (N, 4) point cloud.
            labels: (N,) training labels.

        Returns:
            Augmented points and labels.
        """
        N = points.shape[0]

        # 1. Random rotation around z-axis
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1],
        ], dtype=np.float32)
        points[:, :3] = points[:, :3] @ rotation_matrix.T

        # 2. Random point dropout (keep 95% of points)
        dropout_rate = 0.05
        keep_mask = np.random.random(N) > dropout_rate
        if keep_mask.sum() > 100:  # ensure we keep at least some points
            points = points[keep_mask]
            labels = labels[keep_mask]

        # 3. Intensity noise
        intensity_noise = np.random.normal(0, 0.02, size=points.shape[0]).astype(np.float32)
        points[:, 3] = np.clip(points[:, 3] + intensity_noise, 0.0, 1.0)

        return points, labels

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights from dataset statistics.

        Scans a subset of the dataset to estimate class frequencies,
        then returns inverse-frequency weights.
        """
        from .losses import get_default_semantickitti_weights
        return get_default_semantickitti_weights(num_classes=20)


class SemanticKITTIRangeInferenceDataset(Dataset):
    """Inference-only dataset (no labels) for RangeNet++.

    Returns range images and the projection mapping needed for KNN post-processing.
    """

    def __init__(
        self,
        scan_paths: List[str],
        height: int = 64,
        width: int = 2048,
        fov_up: float = 2.0,
        fov_down: float = -24.8,
    ):
        """
        Args:
            scan_paths: List of paths to .bin point cloud files.
            height: Range image height.
            width: Range image width.
            fov_up: Upper FOV in degrees.
            fov_down: Lower FOV in degrees.
        """
        super().__init__()
        self.scan_paths = scan_paths
        self.projector = SphericalProjection(
            height=height, width=width, fov_up=fov_up, fov_down=fov_down
        )

    def __len__(self) -> int:
        return len(self.scan_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Load a scan for inference.

        Returns:
            Dictionary with:
                'range_image': (5, H, W) float32 tensor (normalized)
                'points': (N, 4) float32 numpy array (original)
                'pixel_to_point': (H, W) int32 numpy array
                'point_to_pixel': (N, 2) int32 numpy array
        """
        points = np.fromfile(self.scan_paths[idx], dtype=np.float32).reshape(-1, 4)

        range_image, pixel_to_point, point_to_pixel = (
            self.projector.project_points_to_range_image_fast(points)
        )

        # Normalize
        normalized = range_image.copy()
        max_range = 80.0
        normalized[0] /= max_range
        normalized[1] /= max_range
        normalized[2] /= max_range
        normalized[3] /= max_range
        normalized[4] = np.clip(normalized[4], 0.0, 1.0)

        range_tensor = torch.from_numpy(normalized).float()

        return {
            "range_image": range_tensor,
            "points": points,
            "pixel_to_point": pixel_to_point,
            "point_to_pixel": point_to_pixel,
        }
