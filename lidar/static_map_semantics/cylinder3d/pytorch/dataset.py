"""
Dataset loading for Cylinder3D semantic segmentation.

Supports SemanticKITTI and nuScenes LiDAR segmentation datasets with full
augmentation pipelines and proper label remapping.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ==============================================================================
# SemanticKITTI Constants
# ==============================================================================

SEMANTICKITTI_CLASSES = [
    'unlabeled',      # 0
    'car',            # 1
    'bicycle',        # 2
    'motorcycle',     # 3
    'truck',          # 4
    'other-vehicle',  # 5
    'person',         # 6
    'bicyclist',      # 7
    'motorcyclist',   # 8
    'road',           # 9
    'parking',        # 10
    'sidewalk',       # 11
    'other-ground',   # 12
    'building',       # 13
    'fence',          # 14
    'vegetation',     # 15
    'trunk',          # 16
    'terrain',        # 17
    'pole',           # 18
    'traffic-sign',   # 19
]

# Maps raw SemanticKITTI label IDs to learning (training) class IDs [0..19].
# 0 = unlabeled/ignored class.
SEMANTICKITTI_LEARNING_MAP = {
    0: 0,       # "unlabeled"
    1: 0,       # "outlier" mapped to "unlabeled"
    10: 1,      # "car"
    11: 2,      # "bicycle"
    13: 5,      # "bus" mapped to "other-vehicle"
    15: 3,      # "motorcycle"
    16: 5,      # "on-rails" mapped to "other-vehicle"
    18: 4,      # "truck"
    20: 5,      # "other-vehicle"
    30: 6,      # "person"
    31: 7,      # "bicyclist"
    32: 8,      # "motorcyclist"
    40: 9,      # "road"
    44: 10,     # "parking"
    48: 11,     # "sidewalk"
    49: 12,     # "other-ground"
    50: 13,     # "building"
    51: 14,     # "fence"
    52: 0,      # "other-structure" mapped to "unlabeled"
    60: 9,      # "lane-marking" mapped to "road"
    70: 15,     # "vegetation"
    71: 16,     # "trunk"
    72: 17,     # "terrain"
    80: 18,     # "pole"
    81: 19,     # "traffic-sign"
    99: 0,      # "other-object" mapped to "unlabeled"
    252: 1,     # "moving-car" mapped to "car"
    253: 7,     # "moving-bicyclist" mapped to "bicyclist"
    254: 6,     # "moving-person" mapped to "person"
    255: 8,     # "moving-motorcyclist" mapped to "motorcyclist"
    256: 5,     # "moving-on-rails" mapped to "other-vehicle"
    257: 5,     # "moving-bus" mapped to "other-vehicle"
    258: 4,     # "moving-truck" mapped to "truck"
    259: 5,     # "moving-other-vehicle" mapped to "other-vehicle"
}

# BGR color map for SemanticKITTI visualization (20 classes)
SEMANTICKITTI_COLOR_MAP = {
    0:  [0, 0, 0],          # unlabeled - black
    1:  [245, 150, 100],    # car - orange
    2:  [245, 230, 100],    # bicycle - yellow
    3:  [150, 60, 30],      # motorcycle - dark orange
    4:  [180, 30, 80],      # truck - dark magenta
    5:  [255, 0, 0],        # other-vehicle - red
    6:  [30, 30, 255],      # person - blue
    7:  [200, 40, 255],     # bicyclist - purple
    8:  [90, 30, 150],      # motorcyclist - dark purple
    9:  [255, 0, 255],      # road - magenta
    10: [255, 150, 255],    # parking - light magenta
    11: [75, 0, 75],        # sidewalk - dark purple
    12: [75, 0, 175],       # other-ground - indigo
    13: [0, 200, 255],      # building - cyan
    14: [50, 120, 255],     # fence - light blue
    15: [0, 175, 0],        # vegetation - green
    16: [0, 60, 135],       # trunk - dark brown
    17: [80, 240, 150],     # terrain - light green
    18: [150, 240, 255],    # pole - light cyan
    19: [0, 0, 255],        # traffic-sign - blue
}


# ==============================================================================
# nuScenes Constants
# ==============================================================================

NUSCENES_CLASSES = [
    'noise',              # 0 - unlabeled/noise
    'barrier',            # 1
    'bicycle',            # 2
    'bus',                # 3
    'car',                # 4
    'construction_vehicle',  # 5
    'motorcycle',         # 6
    'pedestrian',         # 7
    'traffic_cone',       # 8
    'trailer',            # 9
    'truck',              # 10
    'driveable_surface',  # 11
    'other_flat',         # 12
    'sidewalk',           # 13
    'terrain',            # 14
    'manmade',            # 15
    'vegetation',         # 16
]

# nuScenes lidarseg general class mapping (raw label -> training label)
# nuScenes has 32 raw classes mapped to 16 + 1 unlabeled
NUSCENES_LEARNING_MAP = {
    0: 0,    # noise
    1: 0,    # animal -> noise (rare)
    2: 7,    # human.pedestrian.adult -> pedestrian
    3: 7,    # human.pedestrian.child -> pedestrian
    4: 7,    # human.pedestrian.construction_worker -> pedestrian
    5: 0,    # human.pedestrian.personal_mobility -> noise
    6: 7,    # human.pedestrian.police_officer -> pedestrian
    7: 0,    # human.pedestrian.stroller -> noise
    8: 0,    # human.pedestrian.wheelchair -> noise
    9: 1,    # movable_object.barrier -> barrier
    10: 0,   # movable_object.debris -> noise
    11: 0,   # movable_object.pushable_pullable -> noise
    12: 8,   # movable_object.trafficcone -> traffic_cone
    13: 0,   # static_object.bicycle_rack -> noise
    14: 2,   # vehicle.bicycle -> bicycle
    15: 3,   # vehicle.bus.bendy -> bus
    16: 3,   # vehicle.bus.rigid -> bus
    17: 4,   # vehicle.car -> car
    18: 5,   # vehicle.construction -> construction_vehicle
    19: 0,   # vehicle.emergency.ambulance -> noise
    20: 0,   # vehicle.emergency.police -> noise
    21: 6,   # vehicle.motorcycle -> motorcycle
    22: 9,   # vehicle.trailer -> trailer
    23: 10,  # vehicle.truck -> truck
    24: 11,  # flat.driveable_surface -> driveable_surface
    25: 12,  # flat.other -> other_flat
    26: 13,  # flat.sidewalk -> sidewalk
    27: 14,  # flat.terrain -> terrain
    28: 15,  # static.manmade -> manmade
    29: 0,   # static.other -> noise
    30: 16,  # static.vegetation -> vegetation
    31: 0,   # vehicle.ego -> noise (always excluded)
}

# RGB color map for nuScenes visualization (17 classes)
NUSCENES_COLOR_MAP = {
    0:  [0, 0, 0],          # noise - black
    1:  [255, 120, 50],     # barrier - orange
    2:  [255, 192, 203],    # bicycle - pink
    3:  [255, 255, 0],      # bus - yellow
    4:  [0, 150, 245],      # car - blue
    5:  [0, 255, 255],      # construction_vehicle - cyan
    6:  [200, 180, 0],      # motorcycle - dark yellow
    7:  [255, 0, 0],        # pedestrian - red
    8:  [255, 240, 150],    # traffic_cone - light yellow
    9:  [135, 60, 0],       # trailer - brown
    10: [160, 32, 240],     # truck - purple
    11: [255, 0, 255],      # driveable_surface - magenta
    12: [139, 137, 137],    # other_flat - gray
    13: [75, 0, 75],        # sidewalk - dark purple
    14: [150, 240, 80],     # terrain - light green
    15: [230, 230, 250],    # manmade - lavender
    16: [0, 175, 0],        # vegetation - green
}


# ==============================================================================
# SemanticKITTI Dataset
# ==============================================================================

class SemanticKITTIDataset(Dataset):
    """
    Dataset loader for SemanticKITTI LiDAR semantic segmentation.

    File structure expected:
        root/
          sequences/
            00/
              velodyne/
                000000.bin
                000001.bin
                ...
              labels/
                000000.label
                000001.label
                ...
            01/
              ...

    Each .bin file contains N x 4 float32 values (x, y, z, remission).
    Each .label file contains N x uint32 values where:
        - lower 16 bits = semantic label
        - upper 16 bits = instance ID
    """

    def __init__(
        self,
        root: str,
        sequences: List[str],
        config: Optional[Dict] = None,
        augment: bool = True,
    ):
        """
        Args:
            root: Path to SemanticKITTI dataset root (contains 'sequences/' folder).
            sequences: List of sequence IDs to load, e.g. ['00', '01', ..., '10'].
            config: Optional configuration dict with keys:
                - 'learning_map': custom label remapping dict (default: SEMANTICKITTI_LEARNING_MAP)
                - 'max_points': max number of points to keep per scan (default: None = all)
                - 'min_points': min number of points required (default: 1024)
                - 'augmentation': dict of augmentation parameters
            augment: Whether to apply data augmentation.
        """
        super().__init__()

        self.root = root
        self.sequences = sequences
        self.augment = augment
        self.config = config or {}

        # Label remapping
        self.learning_map = self.config.get('learning_map', SEMANTICKITTI_LEARNING_MAP)
        self._build_label_lut()

        # Point count limits
        self.max_points = self.config.get('max_points', None)
        self.min_points = self.config.get('min_points', 1024)

        # Augmentation parameters
        aug_config = self.config.get('augmentation', {})
        self.aug_rotation_range = aug_config.get('rotation_range', [-np.pi, np.pi])
        self.aug_flip_prob = aug_config.get('flip_prob', 0.5)
        self.aug_scale_range = aug_config.get('scale_range', [0.95, 1.05])
        self.aug_translate_std = aug_config.get('translate_std', 0.1)
        self.aug_dropout_prob = aug_config.get('dropout_prob', 0.1)
        self.aug_dropout_ratio = aug_config.get('dropout_ratio', 0.05)

        # Discover all scan files
        self.scan_files = []
        self.label_files = []
        self._discover_files()

    def _build_label_lut(self):
        """Build a lookup table for fast label remapping."""
        # Maximum possible label value in SemanticKITTI raw labels
        max_label = max(self.learning_map.keys()) + 1
        # Allocate LUT - default to 0 (unlabeled) for any unknown labels
        self.label_lut = np.zeros(max_label + 100, dtype=np.int32)
        for raw_label, mapped_label in self.learning_map.items():
            self.label_lut[raw_label] = mapped_label

    def _discover_files(self):
        """Walk through sequences to find all .bin and .label file pairs."""
        for seq in self.sequences:
            seq_str = str(seq).zfill(2)
            velodyne_dir = os.path.join(self.root, 'sequences', seq_str, 'velodyne')
            labels_dir = os.path.join(self.root, 'sequences', seq_str, 'labels')

            if not os.path.isdir(velodyne_dir):
                raise FileNotFoundError(
                    f"Velodyne directory not found: {velodyne_dir}"
                )

            # Find all .bin files and sort them
            bin_files = sorted(glob.glob(os.path.join(velodyne_dir, '*.bin')))

            for bin_file in bin_files:
                # Derive corresponding label file
                scan_name = os.path.splitext(os.path.basename(bin_file))[0]
                label_file = os.path.join(labels_dir, scan_name + '.label')

                # Only include if label file exists (for training/validation)
                if os.path.isfile(label_file):
                    self.scan_files.append(bin_file)
                    self.label_files.append(label_file)
                elif not os.path.isdir(labels_dir):
                    # Test set: no labels available, use dummy
                    self.scan_files.append(bin_file)
                    self.label_files.append(None)

        if len(self.scan_files) == 0:
            raise RuntimeError(
                f"No scan files found in sequences {self.sequences} under {self.root}"
            )

    def __len__(self) -> int:
        """Return total number of scans in the dataset."""
        return len(self.scan_files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Load a single scan and its labels.

        Args:
            index: Index of the scan to load.

        Returns:
            Dictionary with:
                'points': torch.FloatTensor of shape (N, 4) - x, y, z, remission
                'labels': torch.LongTensor of shape (N,) - semantic class labels [0..19]
        """
        # Load point cloud
        scan_file = self.scan_files[index]
        points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)

        # Load labels
        label_file = self.label_files[index]
        if label_file is not None:
            raw_labels = np.fromfile(label_file, dtype=np.uint32)
            # Extract semantic label from lower 16 bits
            semantic_labels = (raw_labels & 0xFFFF).astype(np.int32)
            # Remap using LUT
            labels = self.label_lut[semantic_labels]
        else:
            # No labels (test set) - return zeros
            labels = np.zeros(points.shape[0], dtype=np.int32)

        # Validate point count
        if points.shape[0] < self.min_points:
            # Pad with zeros if too few points (edge case)
            pad_count = self.min_points - points.shape[0]
            points = np.vstack([points, np.zeros((pad_count, 4), dtype=np.float32)])
            labels = np.concatenate([labels, np.zeros(pad_count, dtype=np.int32)])

        # Limit point count if configured
        if self.max_points is not None and points.shape[0] > self.max_points:
            choice = np.random.choice(points.shape[0], self.max_points, replace=False)
            choice.sort()
            points = points[choice]
            labels = labels[choice]

        # Apply augmentations
        if self.augment:
            points, labels = self._augment(points, labels)

        # Convert to tensors
        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels.astype(np.int64)).long()

        return {
            'points': points_tensor,
            'labels': labels_tensor,
        }

    def _augment(
        self, points: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to the point cloud.

        Augmentations applied:
            1. Random rotation around z-axis
            2. Random flip along x and/or y axis
            3. Random scaling
            4. Random translation
            5. Random point dropout

        Args:
            points: (N, 4) float32 array of x, y, z, remission.
            labels: (N,) int32 array of semantic labels.

        Returns:
            Augmented points and labels.
        """
        xyz = points[:, :3]
        remission = points[:, 3:]

        # 1. Random rotation around z-axis
        theta = np.random.uniform(self.aug_rotation_range[0], self.aug_rotation_range[1])
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation_matrix = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float32)
        xyz = xyz @ rotation_matrix.T

        # 2. Random flip along x-axis
        if np.random.random() < self.aug_flip_prob:
            xyz[:, 0] = -xyz[:, 0]

        # 3. Random flip along y-axis
        if np.random.random() < self.aug_flip_prob:
            xyz[:, 1] = -xyz[:, 1]

        # 4. Random scaling
        scale = np.random.uniform(self.aug_scale_range[0], self.aug_scale_range[1])
        xyz *= scale

        # 5. Random translation
        translation = np.random.normal(0.0, self.aug_translate_std, size=(1, 3)).astype(np.float32)
        xyz += translation

        # 6. Random point dropout
        if np.random.random() < self.aug_dropout_prob:
            num_points = xyz.shape[0]
            num_drop = int(num_points * self.aug_dropout_ratio)
            if num_drop > 0 and num_points - num_drop >= self.min_points:
                keep_mask = np.ones(num_points, dtype=bool)
                drop_indices = np.random.choice(num_points, num_drop, replace=False)
                keep_mask[drop_indices] = False
                xyz = xyz[keep_mask]
                remission = remission[keep_mask]
                labels = labels[keep_mask]

        # Reassemble points
        points = np.concatenate([xyz, remission], axis=1)

        return points, labels


# ==============================================================================
# nuScenes LiDAR Segmentation Dataset
# ==============================================================================

class NuScenesLidarSegDataset(Dataset):
    """
    Dataset loader for nuScenes LiDAR segmentation.

    File structure expected:
        root/
          samples/
            LIDAR_TOP/
              n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin
              ...
          lidarseg/
            v1.0-trainval/
              n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377_lidarseg.bin
              ...
          v1.0-trainval/
            lidarseg.json
            ...

    Each .pcd.bin file contains N x 5 float32 values (x, y, z, intensity, ring_index).
    Each lidarseg .bin file contains N x uint8 values (semantic label per point).
    """

    def __init__(
        self,
        root: str,
        sequences: Optional[List[str]] = None,
        config: Optional[Dict] = None,
        augment: bool = True,
    ):
        """
        Args:
            root: Path to nuScenes dataset root.
            sequences: List of sample tokens or split specification.
                If None, discovers all available samples from lidarseg folder.
            config: Optional configuration dict with keys:
                - 'version': dataset version string (default: 'v1.0-trainval')
                - 'learning_map': custom label remapping dict
                - 'max_points': max number of points to keep per scan
                - 'min_points': min number of points required
                - 'augmentation': augmentation parameters dict
            augment: Whether to apply data augmentation.
        """
        super().__init__()

        self.root = root
        self.augment = augment
        self.config = config or {}

        # Dataset version
        self.version = self.config.get('version', 'v1.0-trainval')

        # Label remapping
        self.learning_map = self.config.get('learning_map', NUSCENES_LEARNING_MAP)
        self._build_label_lut()

        # Point count limits
        self.max_points = self.config.get('max_points', None)
        self.min_points = self.config.get('min_points', 1024)

        # Augmentation parameters
        aug_config = self.config.get('augmentation', {})
        self.aug_rotation_range = aug_config.get('rotation_range', [-np.pi, np.pi])
        self.aug_flip_prob = aug_config.get('flip_prob', 0.5)
        self.aug_scale_range = aug_config.get('scale_range', [0.95, 1.05])
        self.aug_translate_std = aug_config.get('translate_std', 0.1)
        self.aug_dropout_prob = aug_config.get('dropout_prob', 0.1)
        self.aug_dropout_ratio = aug_config.get('dropout_ratio', 0.05)

        # Discover files
        self.scan_files = []
        self.label_files = []
        self._discover_files(sequences)

    def _build_label_lut(self):
        """Build a lookup table for fast label remapping."""
        max_label = max(self.learning_map.keys()) + 1
        self.label_lut = np.zeros(max_label + 10, dtype=np.int32)
        for raw_label, mapped_label in self.learning_map.items():
            self.label_lut[raw_label] = mapped_label

    def _discover_files(self, sequences: Optional[List[str]]):
        """
        Discover scan and label file pairs.

        If sequences is provided as a list of sample tokens, only those are loaded.
        Otherwise, all available lidarseg .bin files are used.
        """
        lidar_dir = os.path.join(self.root, 'samples', 'LIDAR_TOP')
        lidarseg_dir = os.path.join(self.root, 'lidarseg', self.version)

        if not os.path.isdir(lidar_dir):
            raise FileNotFoundError(f"LIDAR_TOP directory not found: {lidar_dir}")

        if sequences is not None:
            # Sequences here are treated as sample tokens or filenames
            for token in sequences:
                # Try to find matching .pcd.bin
                scan_pattern = os.path.join(lidar_dir, f'*{token}*.pcd.bin')
                matches = glob.glob(scan_pattern)
                if matches:
                    scan_file = matches[0]
                else:
                    # Try direct filename
                    scan_file = os.path.join(lidar_dir, token + '.pcd.bin')
                    if not os.path.isfile(scan_file):
                        continue

                # Derive label file
                scan_basename = os.path.splitext(os.path.splitext(
                    os.path.basename(scan_file)
                )[0])[0]  # Remove .pcd.bin
                label_file = os.path.join(
                    lidarseg_dir, scan_basename + '_lidarseg.bin'
                )

                if os.path.isfile(label_file):
                    self.scan_files.append(scan_file)
                    self.label_files.append(label_file)
        else:
            # Discover all available labeled samples
            if os.path.isdir(lidarseg_dir):
                label_files = sorted(glob.glob(
                    os.path.join(lidarseg_dir, '*_lidarseg.bin')
                ))
                for label_file in label_files:
                    # Derive scan file from label file name
                    label_basename = os.path.basename(label_file)
                    # Remove '_lidarseg.bin' suffix to get the scan token
                    scan_token = label_basename.replace('_lidarseg.bin', '')
                    scan_file = os.path.join(lidar_dir, scan_token + '.pcd.bin')

                    if os.path.isfile(scan_file):
                        self.scan_files.append(scan_file)
                        self.label_files.append(label_file)
            else:
                # Fallback: load all scans without labels
                scan_files = sorted(glob.glob(os.path.join(lidar_dir, '*.pcd.bin')))
                for scan_file in scan_files:
                    self.scan_files.append(scan_file)
                    self.label_files.append(None)

        if len(self.scan_files) == 0:
            raise RuntimeError(
                f"No scan files found in nuScenes dataset at {self.root}"
            )

    def __len__(self) -> int:
        """Return total number of scans in the dataset."""
        return len(self.scan_files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Load a single scan and its labels.

        Args:
            index: Index of the scan to load.

        Returns:
            Dictionary with:
                'points': torch.FloatTensor of shape (N, 4) - x, y, z, intensity
                'labels': torch.LongTensor of shape (N,) - semantic class labels [0..16]
        """
        # Load point cloud - nuScenes stores 5 floats: x, y, z, intensity, ring_index
        scan_file = self.scan_files[index]
        raw_points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 5)
        # Keep x, y, z, intensity (drop ring_index)
        points = raw_points[:, :4].copy()

        # Load labels
        label_file = self.label_files[index]
        if label_file is not None:
            raw_labels = np.fromfile(label_file, dtype=np.uint8).astype(np.int32)
            # Remap using LUT
            labels = self.label_lut[raw_labels]
        else:
            labels = np.zeros(points.shape[0], dtype=np.int32)

        # Ensure label count matches point count
        if labels.shape[0] != points.shape[0]:
            min_count = min(labels.shape[0], points.shape[0])
            points = points[:min_count]
            labels = labels[:min_count]

        # Validate point count
        if points.shape[0] < self.min_points:
            pad_count = self.min_points - points.shape[0]
            points = np.vstack([points, np.zeros((pad_count, 4), dtype=np.float32)])
            labels = np.concatenate([labels, np.zeros(pad_count, dtype=np.int32)])

        # Limit point count if configured
        if self.max_points is not None and points.shape[0] > self.max_points:
            choice = np.random.choice(points.shape[0], self.max_points, replace=False)
            choice.sort()
            points = points[choice]
            labels = labels[choice]

        # Apply augmentations
        if self.augment:
            points, labels = self._augment(points, labels)

        # Convert to tensors
        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels.astype(np.int64)).long()

        return {
            'points': points_tensor,
            'labels': labels_tensor,
        }

    def _augment(
        self, points: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to the point cloud.

        Same augmentations as SemanticKITTI:
            1. Random rotation around z-axis
            2. Random flip along x and/or y axis
            3. Random scaling
            4. Random translation
            5. Random point dropout
        """
        xyz = points[:, :3]
        features = points[:, 3:]

        # 1. Random rotation around z-axis
        theta = np.random.uniform(self.aug_rotation_range[0], self.aug_rotation_range[1])
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation_matrix = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float32)
        xyz = xyz @ rotation_matrix.T

        # 2. Random flip along x-axis
        if np.random.random() < self.aug_flip_prob:
            xyz[:, 0] = -xyz[:, 0]

        # 3. Random flip along y-axis
        if np.random.random() < self.aug_flip_prob:
            xyz[:, 1] = -xyz[:, 1]

        # 4. Random scaling
        scale = np.random.uniform(self.aug_scale_range[0], self.aug_scale_range[1])
        xyz *= scale

        # 5. Random translation
        translation = np.random.normal(0.0, self.aug_translate_std, size=(1, 3)).astype(np.float32)
        xyz += translation

        # 6. Random point dropout
        if np.random.random() < self.aug_dropout_prob:
            num_points = xyz.shape[0]
            num_drop = int(num_points * self.aug_dropout_ratio)
            if num_drop > 0 and num_points - num_drop >= self.min_points:
                keep_mask = np.ones(num_points, dtype=bool)
                drop_indices = np.random.choice(num_points, num_drop, replace=False)
                keep_mask[drop_indices] = False
                xyz = xyz[keep_mask]
                features = features[keep_mask]
                labels = labels[keep_mask]

        # Reassemble points
        points = np.concatenate([xyz, features], axis=1)

        return points, labels


# ==============================================================================
# Collate Function
# ==============================================================================

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-size point clouds.

    Since each scan has a different number of points, we cannot simply stack
    them into a single tensor. Instead, we concatenate all points and labels,
    and provide batch indices and point counts for reconstruction.

    Args:
        batch: List of dicts from __getitem__, each with 'points' (N_i, 4)
               and 'labels' (N_i,).

    Returns:
        Dictionary with:
            'points': torch.FloatTensor of shape (sum(N_i), 4) - all points concatenated
            'labels': torch.LongTensor of shape (sum(N_i),) - all labels concatenated
            'batch_indices': torch.LongTensor of shape (sum(N_i),) - batch index per point
            'point_counts': torch.LongTensor of shape (B,) - number of points per sample
    """
    points_list = []
    labels_list = []
    batch_indices_list = []
    point_counts = []

    for batch_idx, sample in enumerate(batch):
        pts = sample['points']
        lbl = sample['labels']
        num_points = pts.shape[0]

        points_list.append(pts)
        labels_list.append(lbl)
        batch_indices_list.append(
            torch.full((num_points,), batch_idx, dtype=torch.long)
        )
        point_counts.append(num_points)

    return {
        'points': torch.cat(points_list, dim=0),
        'labels': torch.cat(labels_list, dim=0),
        'batch_indices': torch.cat(batch_indices_list, dim=0),
        'point_counts': torch.tensor(point_counts, dtype=torch.long),
    }


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_semantickitti_color_map_array() -> np.ndarray:
    """
    Return SemanticKITTI color map as a numpy array of shape (20, 3).

    Colors are in RGB format with values [0, 255].
    """
    num_classes = len(SEMANTICKITTI_CLASSES)
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    for class_id, color in SEMANTICKITTI_COLOR_MAP.items():
        color_map[class_id] = color
    return color_map


def get_nuscenes_color_map_array() -> np.ndarray:
    """
    Return nuScenes color map as a numpy array of shape (17, 3).

    Colors are in RGB format with values [0, 255].
    """
    num_classes = len(NUSCENES_CLASSES)
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    for class_id, color in NUSCENES_COLOR_MAP.items():
        color_map[class_id] = color
    return color_map


def colorize_labels(
    labels: np.ndarray,
    dataset: str = 'semantickitti',
) -> np.ndarray:
    """
    Convert integer labels to RGB colors for visualization.

    Args:
        labels: (N,) integer array of class labels.
        dataset: 'semantickitti' or 'nuscenes'.

    Returns:
        (N, 3) uint8 array of RGB colors.
    """
    if dataset == 'semantickitti':
        color_map = get_semantickitti_color_map_array()
    elif dataset == 'nuscenes':
        color_map = get_nuscenes_color_map_array()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'semantickitti' or 'nuscenes'.")

    # Clip labels to valid range
    labels_clipped = np.clip(labels, 0, color_map.shape[0] - 1)
    return color_map[labels_clipped]


def build_semantickitti_splits() -> Dict[str, List[str]]:
    """
    Return the standard SemanticKITTI train/val/test split.

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to sequence ID lists.
    """
    return {
        'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
        'val': ['08'],
        'test': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'],
    }
