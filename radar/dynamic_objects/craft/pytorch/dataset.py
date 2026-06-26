"""
NuScenes Synchronized Camera + Radar Dataset for CRAFT.

Provides a fully-featured PyTorch dataset for the CRAFT model (Camera-Radar 3D Object
Detection with Spatio-Contextual Fusion Transformer) on the nuScenes benchmark.

Supports:
    - 6 surround-view camera images (FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT)
    - 5 radar point clouds (FRONT, FRONT_LEFT, FRONT_RIGHT, BACK_LEFT, BACK_RIGHT)
    - Temporal radar sweep accumulation (up to 6 past sweeps for denser point clouds)
    - Sensor calibration: camera intrinsics, extrinsics, and radar-to-ego transforms
    - 3D bounding box annotations (translation, size, rotation, velocity, class)
    - Synchronized data augmentation (global flip, radar rotation/scale, image color jitter)
    - Custom collate function for variable-length radar point clouds
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


# =============================================================================
# Constants
# =============================================================================

CAMERA_NAMES: List[str] = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

RADAR_NAMES: List[str] = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT',
]

CLASS_NAMES: List[str] = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

# Default radar feature dimensions to use from raw nuScenes radar points
# [x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, ...]
# Indices: 0=x, 1=y, 2=z, 3=dyn_prop, 4=id, 5=rcs, 8=vx_comp, 9=vy_comp, 18=pdh0
DEFAULT_RADAR_USE_DIMS: List[int] = [0, 1, 2, 3, 4, 5, 8, 9, 18]


# =============================================================================
# Dataset
# =============================================================================


class NuScenesRadarCameraDataset(Dataset):
    """
    Synchronized camera and radar dataset for CRAFT on nuScenes.

    Each sample provides:
        - images: (6, 3, H, W) float tensor of normalized camera images
        - intrinsics: (6, 3, 3) camera intrinsic matrices
        - extrinsics: (6, 4, 4) camera-to-ego transformation matrices
        - radar_points: (N, D) float tensor of accumulated radar points in ego frame
        - radar_mask: (N,) bool tensor indicating valid (non-padded) points
        - gt_boxes: (M, 9) float tensor [x, y, z, w, l, h, yaw, vx, vy]
        - gt_labels: (M,) int tensor of class indices
        - gt_mask: (M,) bool tensor indicating valid (non-padded) annotations

    Args:
        data_root: Root directory of nuScenes data (e.g. '/data/nuscenes').
        info_path: Path to the preprocessed info pickle file containing sample
                   metadata, camera paths, radar paths, calibration, and annotations.
        img_size: Target image size as (height, width) after resizing.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] defining
                          the valid spatial extent for radar points.
        num_sweeps: Number of radar sweeps to accumulate (including current).
        radar_use_dims: Indices of radar point dimensions to retain.
        max_radar_points: Maximum number of radar points after accumulation.
        max_boxes: Maximum number of GT bounding boxes per sample.
        class_names: List of detection class names.
        augmentation: Dictionary of augmentation parameters.
        is_train: Whether this is for training (enables augmentation).
    """

    def __init__(
        self,
        data_root: str,
        info_path: str,
        img_size: Tuple[int, int] = (900, 1600),
        point_cloud_range: List[float] = None,
        num_sweeps: int = 6,
        radar_use_dims: List[int] = None,
        max_radar_points: int = 30000,
        max_boxes: int = 300,
        class_names: List[str] = None,
        augmentation: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
    ) -> None:
        """
        Initialize the NuScenes radar-camera dataset.

        Args:
            data_root: Root directory of nuScenes data.
            info_path: Path to preprocessed info pickle file.
            img_size: Target image size (H, W) after resizing.
            point_cloud_range: Spatial range for filtering radar points.
            num_sweeps: Number of radar sweeps to accumulate.
            radar_use_dims: Indices of radar point features to use.
            max_radar_points: Max number of points (for padding/truncation).
            max_boxes: Maximum number of GT boxes per sample.
            class_names: Detection class names.
            augmentation: Augmentation config dict.
            is_train: Whether this is a training split.
        """
        super().__init__()

        self.data_root = data_root
        self.img_size = img_size
        self.num_sweeps = num_sweeps
        self.max_radar_points = max_radar_points
        self.max_boxes = max_boxes
        self.is_train = is_train

        # Point cloud range
        if point_cloud_range is None:
            self.point_cloud_range = np.array(
                [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32
            )
        else:
            self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)

        # Radar feature selection
        self.radar_use_dims = radar_use_dims if radar_use_dims is not None else DEFAULT_RADAR_USE_DIMS

        # Class names
        self.class_names = class_names if class_names is not None else CLASS_NAMES
        self.class_name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        # Augmentation config
        self.augmentation = augmentation if augmentation is not None else {}

        # Load info pickle
        if not os.path.isfile(info_path):
            raise FileNotFoundError(f"Info file not found: {info_path}")
        with open(info_path, 'rb') as f:
            data = pickle.load(f)

        # Expected pickle structure:
        # {
        #     'infos': [
        #         {
        #             'token': str,
        #             'timestamp': int,
        #             'ego_pose': {'translation': [x,y,z], 'rotation': [w,x,y,z]},
        #             'cams': {
        #                 'CAM_FRONT': {
        #                     'filename': 'samples/CAM_FRONT/n015-...-1234.jpg',
        #                     'intrinsic': [[fx,0,cx],[0,fy,cy],[0,0,1]],
        #                     'extrinsic': [[r00,...,t0],...,[0,0,0,1]],
        #                     'sensor2ego_translation': [x,y,z],
        #                     'sensor2ego_rotation': [w,x,y,z],
        #                 }, ...
        #             },
        #             'radars': {
        #                 'RADAR_FRONT': {
        #                     'filename': 'samples/RADAR_FRONT/n015-...-1234.pcd',
        #                     'sensor2ego_translation': [x,y,z],
        #                     'sensor2ego_rotation': [w,x,y,z],
        #                     'sweeps': [
        #                         {
        #                             'filename': 'sweeps/RADAR_FRONT/...',
        #                             'sensor2ego_translation': [...],
        #                             'sensor2ego_rotation': [...],
        #                             'timestamp': int,
        #                         }, ...
        #                     ],
        #                 }, ...
        #             },
        #             'gt_boxes': np.ndarray (M, 9),  # x,y,z,w,l,h,yaw,vx,vy
        #             'gt_names': list of str,
        #         }, ...
        #     ]
        # }
        self.infos: List[Dict[str, Any]] = data['infos']

        # Image transforms
        self._img_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Color jitter for training augmentation
        color_jitter_cfg = self.augmentation.get('color_jitter', {})
        if self.is_train and color_jitter_cfg:
            self._color_jitter = transforms.ColorJitter(
                brightness=color_jitter_cfg.get('brightness', 0.2),
                contrast=color_jitter_cfg.get('contrast', 0.2),
                saturation=color_jitter_cfg.get('saturation', 0.2),
                hue=color_jitter_cfg.get('hue', 0.1),
            )
        else:
            self._color_jitter = None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.infos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with synchronized camera and radar data.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing images, calibration, radar points,
            and 3D bounding box annotations.
        """
        info = self.infos[idx]

        # Determine global augmentation parameters for this sample
        aug_params = self._sample_augmentation_params()

        # Load multi-camera images with calibration
        images, intrinsics, extrinsics = self._load_cameras(info, aug_params)

        # Load and accumulate radar point clouds
        radar_points = self._load_radar(info, aug_params)

        # Load ground truth annotations
        gt_boxes, gt_labels, gt_mask = self._load_annotations(info, aug_params)

        # Pad or truncate radar points to fixed size
        radar_points, radar_mask = self._pad_radar_points(radar_points)

        return {
            'images': images,             # (6, 3, H, W)
            'intrinsics': intrinsics,     # (6, 3, 3)
            'extrinsics': extrinsics,     # (6, 4, 4)
            'radar_points': radar_points, # (max_radar_points, D)
            'radar_mask': radar_mask,     # (max_radar_points,)
            'gt_boxes': gt_boxes,         # (max_boxes, 9)
            'gt_labels': gt_labels,       # (max_boxes,)
            'gt_mask': gt_mask,           # (max_boxes,)
        }

    # =========================================================================
    # Augmentation
    # =========================================================================

    def _sample_augmentation_params(self) -> Dict[str, Any]:
        """
        Sample random augmentation parameters for a single sample.

        All modalities share the same global flip to maintain spatial consistency
        between camera images and radar point clouds.

        Returns:
            Dictionary of augmentation parameters:
                - flip_x: Whether to flip along the x-axis (left-right).
                - rotation: Random rotation angle in radians.
                - scale: Random scale factor.
                - apply_color_jitter: Whether to apply color jitter to images.
        """
        params: Dict[str, Any] = {
            'flip_x': False,
            'rotation': 0.0,
            'scale': 1.0,
            'apply_color_jitter': False,
        }

        if not self.is_train:
            return params

        # Global horizontal flip (consistent across camera and radar)
        flip_prob = self.augmentation.get('flip_prob', 0.5)
        if np.random.rand() < flip_prob:
            params['flip_x'] = True

        # Rotation for radar BEV (does not affect camera images directly)
        rotation_range = self.augmentation.get('rotation_range', [-0.3925, 0.3925])
        params['rotation'] = np.random.uniform(rotation_range[0], rotation_range[1])

        # Scale for radar BEV
        scale_range = self.augmentation.get('scale_range', [0.95, 1.05])
        params['scale'] = np.random.uniform(scale_range[0], scale_range[1])

        # Color jitter
        if self._color_jitter is not None:
            params['apply_color_jitter'] = True

        return params

    # =========================================================================
    # Camera Loading
    # =========================================================================

    def _load_cameras(
        self, info: Dict[str, Any], aug_params: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and preprocess all 6 camera images with calibration matrices.

        Applies augmentation: horizontal flip (consistent with radar) and
        optional color jitter during training.

        Args:
            info: Sample info dict containing camera metadata.
            aug_params: Augmentation parameters for this sample.

        Returns:
            images: (6, 3, H, W) float tensor of normalized images.
            intrinsics: (6, 3, 3) float tensor of camera intrinsic matrices.
            extrinsics: (6, 4, 4) float tensor of camera-to-ego transforms.
        """
        images_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        extrinsics_list: List[torch.Tensor] = []

        flip_x = aug_params['flip_x']

        for cam_name in CAMERA_NAMES:
            cam_data = info['cams'][cam_name]

            # Load image
            img_path = os.path.join(self.data_root, cam_data['filename'])
            img = Image.open(img_path).convert('RGB')

            # Compute resize scale for intrinsic adjustment
            orig_w, orig_h = img.size
            target_h, target_w = self.img_size
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            # Resize
            img = img.resize((target_w, target_h), Image.BILINEAR)

            # Apply color jitter (training only)
            if aug_params['apply_color_jitter']:
                img = self._color_jitter(img)

            # Apply horizontal flip
            if flip_x:
                img = TF.hflip(img)

            # Convert to tensor and normalize
            img_tensor = TF.to_tensor(img)  # (3, H, W), float [0, 1]
            img_tensor = self._img_normalize(img_tensor)
            images_list.append(img_tensor)

            # Intrinsic matrix: adjust for resize
            intrinsic = np.array(cam_data['intrinsic'], dtype=np.float32)
            intrinsic[0, :] *= scale_x  # fx, s, cx
            intrinsic[1, :] *= scale_y  # fy, cy

            # Adjust intrinsics for horizontal flip
            if flip_x:
                intrinsic[0, 2] = target_w - intrinsic[0, 2]

            intrinsics_list.append(torch.from_numpy(intrinsic))

            # Extrinsic: camera-to-ego 4x4 transform
            extrinsic = np.array(cam_data['extrinsic'], dtype=np.float32)

            # For horizontal flip: mirror the camera x-axis in ego frame
            if flip_x:
                # Negate the x component of translation
                extrinsic[0, 3] = -extrinsic[0, 3]
                # Negate x-row of rotation (first row)
                extrinsic[0, :3] = -extrinsic[0, :3]

            extrinsics_list.append(torch.from_numpy(extrinsic))

        images = torch.stack(images_list, dim=0)
        intrinsics = torch.stack(intrinsics_list, dim=0)
        extrinsics = torch.stack(extrinsics_list, dim=0)

        return images, intrinsics, extrinsics

    # =========================================================================
    # Radar Loading
    # =========================================================================

    def _load_radar(
        self, info: Dict[str, Any], aug_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Load and accumulate radar point clouds from all 5 radar sensors.

        For each radar sensor, loads the current keyframe and up to (num_sweeps - 1)
        past sweeps. All points are transformed into the current ego vehicle frame
        using sensor calibration and ego pose transformations.

        Args:
            info: Sample info dict containing radar metadata.
            aug_params: Augmentation parameters for this sample.

        Returns:
            points: (N, D) float array of accumulated radar points in ego frame,
                    where D is len(radar_use_dims).
        """
        all_points: List[np.ndarray] = []

        for radar_name in RADAR_NAMES:
            radar_data = info['radars'][radar_name]

            # Build sensor-to-ego transform for keyframe
            sensor2ego = self._build_transform(
                radar_data['sensor2ego_translation'],
                radar_data['sensor2ego_rotation'],
            )

            # Load current keyframe radar points
            keyframe_path = os.path.join(self.data_root, radar_data['filename'])
            keyframe_points = self._load_radar_file(keyframe_path)

            if keyframe_points.shape[0] > 0:
                # Transform to ego frame
                ego_points = self._transform_radar_points(keyframe_points, sensor2ego)
                all_points.append(ego_points)

            # Load past sweeps
            sweeps = radar_data.get('sweeps', [])
            num_past_sweeps = min(self.num_sweeps - 1, len(sweeps))

            for i in range(num_past_sweeps):
                sweep = sweeps[i]
                sweep_path = os.path.join(self.data_root, sweep['filename'])
                sweep_points = self._load_radar_file(sweep_path)

                if sweep_points.shape[0] == 0:
                    continue

                # Build sweep sensor-to-ego transform
                sweep_sensor2ego = self._build_transform(
                    sweep['sensor2ego_translation'],
                    sweep['sensor2ego_rotation'],
                )

                # If the sweep has a different ego pose than the keyframe,
                # we need to compensate: sweep_sensor -> sweep_ego -> global -> keyframe_ego
                if 'ego_pose' in sweep and 'ego_pose' in info:
                    sweep_ego2global = self._build_transform(
                        sweep['ego_pose']['translation'],
                        sweep['ego_pose']['rotation'],
                    )
                    keyframe_global2ego = np.linalg.inv(
                        self._build_transform(
                            info['ego_pose']['translation'],
                            info['ego_pose']['rotation'],
                        )
                    )
                    # Full chain: sweep_sensor -> sweep_ego -> global -> keyframe_ego
                    full_transform = keyframe_global2ego @ sweep_ego2global @ sweep_sensor2ego
                else:
                    # Assume same ego frame (simplified case)
                    full_transform = sweep_sensor2ego

                ego_points = self._transform_radar_points(sweep_points, full_transform)
                all_points.append(ego_points)

        # Concatenate all radar points
        if len(all_points) == 0:
            # No valid radar points; return empty array with correct feature dim
            points = np.zeros((0, len(self.radar_use_dims)), dtype=np.float32)
        else:
            points = np.concatenate(all_points, axis=0)

        # Filter points within range
        points = self._filter_points_in_range(points)

        # Apply radar augmentations (flip, rotation, scale)
        points = self._augment_radar(points, aug_params)

        return points

    def _load_radar_file(self, filepath: str) -> np.ndarray:
        """
        Load a single radar point cloud file.

        Supports nuScenes PCD format (binary numpy) or precomputed .bin files.
        Returns raw radar points with all 18 features from nuScenes radar spec.

        Args:
            filepath: Absolute path to the radar PCD or bin file.

        Returns:
            points: (N, 18) float array of radar detections, or empty (0, 18)
                    if the file cannot be loaded.
        """
        if not os.path.isfile(filepath):
            return np.zeros((0, 18), dtype=np.float32)

        # nuScenes stores radar as .pcd files with custom binary format
        # The standard approach uses nuscenes-devkit's PointCloud.from_file,
        # but for efficiency we load the precomputed .bin representation
        if filepath.endswith('.bin'):
            points = np.fromfile(filepath, dtype=np.float32)
            if points.size == 0:
                return np.zeros((0, 18), dtype=np.float32)
            points = points.reshape(-1, 18)
        elif filepath.endswith('.pcd'):
            points = self._load_pcd(filepath)
        else:
            return np.zeros((0, 18), dtype=np.float32)

        return points

    def _load_pcd(self, filepath: str) -> np.ndarray:
        """
        Load a nuScenes radar PCD file.

        nuScenes radar PCD files contain a text header followed by binary data.
        Each point has 18 float32 fields: x, y, z, dyn_prop, id, rcs, vx, vy,
        vx_comp, vy_comp, is_quality_valid, ambig_state, x_rms, y_rms,
        invalid_state, pdh0, vx_rms, vy_rms.

        Args:
            filepath: Path to the .pcd file.

        Returns:
            points: (N, 18) float32 array of radar detections.
        """
        num_fields = 18

        with open(filepath, 'rb') as f:
            # Parse header to find data offset
            header_lines = []
            num_points = 0
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                header_lines.append(line)
                if line.startswith('POINTS'):
                    num_points = int(line.split()[-1])
                if line.startswith('DATA'):
                    break

            if num_points == 0:
                return np.zeros((0, num_fields), dtype=np.float32)

            # Read binary data
            data = np.frombuffer(
                f.read(num_points * num_fields * 4),
                dtype=np.float32,
            )

        if data.size < num_points * num_fields:
            return np.zeros((0, num_fields), dtype=np.float32)

        points = data.reshape(num_points, num_fields)
        return points

    def _build_transform(
        self, translation: List[float], rotation: List[float]
    ) -> np.ndarray:
        """
        Build a 4x4 homogeneous transformation matrix from translation and quaternion.

        Args:
            translation: [x, y, z] translation vector.
            rotation: [w, x, y, z] quaternion (scalar-first convention).

        Returns:
            transform: (4, 4) float64 transformation matrix.
        """
        transform = np.eye(4, dtype=np.float64)
        quat = Quaternion(rotation)
        transform[:3, :3] = quat.rotation_matrix
        transform[:3, 3] = np.array(translation, dtype=np.float64)
        return transform

    def _transform_radar_points(
        self, points: np.ndarray, transform: np.ndarray
    ) -> np.ndarray:
        """
        Transform radar points from sensor frame to target frame.

        Transforms the spatial coordinates (x, y, z) and compensated velocities
        (vx_comp, vy_comp) using the given transformation matrix. Selects only
        the configured radar_use_dims from the result.

        Args:
            points: (N, 18) raw radar points in sensor frame.
            transform: (4, 4) transformation matrix (sensor-to-target).

        Returns:
            transformed: (N, D) radar points with selected features in target frame,
                         where D = len(self.radar_use_dims).
        """
        n_points = points.shape[0]
        if n_points == 0:
            return np.zeros((0, len(self.radar_use_dims)), dtype=np.float32)

        # Transform xyz positions
        xyz = points[:, :3]  # (N, 3)
        ones = np.ones((n_points, 1), dtype=np.float64)
        xyz_homo = np.concatenate([xyz.astype(np.float64), ones], axis=1)  # (N, 4)
        xyz_transformed = (transform @ xyz_homo.T).T[:, :3]  # (N, 3)

        # Transform compensated velocities (rotation only, no translation)
        # Indices 8, 9 are vx_comp, vy_comp in nuScenes radar
        rotation = transform[:3, :3]
        vx_comp = points[:, 8]
        vy_comp = points[:, 9]
        vz_comp = np.zeros_like(vx_comp)
        vel_sensor = np.stack([vx_comp, vy_comp, vz_comp], axis=1)  # (N, 3)
        vel_transformed = (rotation @ vel_sensor.astype(np.float64).T).T  # (N, 3)

        # Rebuild full point array with transformed coordinates and velocities
        points_out = points.copy().astype(np.float32)
        points_out[:, 0] = xyz_transformed[:, 0].astype(np.float32)
        points_out[:, 1] = xyz_transformed[:, 1].astype(np.float32)
        points_out[:, 2] = xyz_transformed[:, 2].astype(np.float32)
        points_out[:, 8] = vel_transformed[:, 0].astype(np.float32)
        points_out[:, 9] = vel_transformed[:, 1].astype(np.float32)

        # Select desired dimensions
        points_selected = points_out[:, self.radar_use_dims]

        return points_selected

    def _filter_points_in_range(self, points: np.ndarray) -> np.ndarray:
        """
        Filter radar points to those within the configured point cloud range.

        The spatial coordinates (first 3 dimensions of radar_use_dims which
        correspond to x, y, z) are checked against point_cloud_range.

        Args:
            points: (N, D) radar points in ego frame.

        Returns:
            filtered: (M, D) points within the valid range, M <= N.
        """
        if points.shape[0] == 0:
            return points

        # x, y, z are always the first three columns after dim selection
        # since radar_use_dims starts with [0, 1, 2, ...]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        x_min, y_min, z_min = self.point_cloud_range[:3]
        x_max, y_max, z_max = self.point_cloud_range[3:]

        mask = (
            (x >= x_min) & (x <= x_max) &
            (y >= y_min) & (y <= y_max) &
            (z >= z_min) & (z <= z_max)
        )

        return points[mask]

    def _augment_radar(
        self, points: np.ndarray, aug_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply spatial augmentations to radar points.

        Augmentations applied:
            1. Horizontal flip along x-axis (consistent with camera flip)
            2. Random rotation around z-axis (yaw)
            3. Random uniform scaling

        All augmentations are applied to x, y coordinates and velocities
        (vx_comp, vy_comp) where applicable.

        Args:
            points: (N, D) radar points in ego frame.
            aug_params: Augmentation parameters (flip_x, rotation, scale).

        Returns:
            augmented: (N, D) augmented radar points.
        """
        if points.shape[0] == 0:
            return points

        points = points.copy()

        # Horizontal flip along x-axis
        if aug_params['flip_x']:
            points[:, 0] = -points[:, 0]
            # Flip vx_comp if it is among selected dims
            # In radar_use_dims [0,1,2,3,4,5,8,9,18], index 6 is vx_comp (orig dim 8)
            vx_idx = self._get_selected_dim_index(8)
            if vx_idx is not None:
                points[:, vx_idx] = -points[:, vx_idx]

        # Random rotation around z-axis
        rotation = aug_params['rotation']
        if abs(rotation) > 1e-6:
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)

            # Rotate x, y
            x = points[:, 0].copy()
            y = points[:, 1].copy()
            points[:, 0] = x * cos_r - y * sin_r
            points[:, 1] = x * sin_r + y * cos_r

            # Rotate vx, vy
            vx_idx = self._get_selected_dim_index(8)
            vy_idx = self._get_selected_dim_index(9)
            if vx_idx is not None and vy_idx is not None:
                vx = points[:, vx_idx].copy()
                vy = points[:, vy_idx].copy()
                points[:, vx_idx] = vx * cos_r - vy * sin_r
                points[:, vy_idx] = vx * sin_r + vy * cos_r

        # Random scaling (positions only)
        scale = aug_params['scale']
        if abs(scale - 1.0) > 1e-6:
            points[:, 0] *= scale
            points[:, 1] *= scale
            points[:, 2] *= scale

        return points

    def _get_selected_dim_index(self, original_dim: int) -> Optional[int]:
        """
        Get the index of an original radar dimension within the selected dims.

        Args:
            original_dim: The dimension index in the full 18-feature radar point.

        Returns:
            The index within self.radar_use_dims, or None if not selected.
        """
        try:
            return self.radar_use_dims.index(original_dim)
        except ValueError:
            return None

    # =========================================================================
    # Radar Padding
    # =========================================================================

    def _pad_radar_points(
        self, points: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad or truncate radar points to a fixed size for batching.

        If there are more points than max_radar_points, a random subset is
        selected during training (all points are kept during evaluation by
        taking the first max_radar_points).

        Args:
            points: (N, D) radar points.

        Returns:
            padded_points: (max_radar_points, D) float tensor.
            mask: (max_radar_points,) bool tensor where True = valid point.
        """
        n_points = points.shape[0]
        n_features = len(self.radar_use_dims)

        padded = np.zeros((self.max_radar_points, n_features), dtype=np.float32)
        mask = np.zeros(self.max_radar_points, dtype=np.bool_)

        if n_points == 0:
            return torch.from_numpy(padded), torch.from_numpy(mask)

        if n_points > self.max_radar_points:
            # Subsample
            if self.is_train:
                indices = np.random.choice(
                    n_points, self.max_radar_points, replace=False
                )
            else:
                indices = np.arange(self.max_radar_points)
            padded[:] = points[indices]
            mask[:] = True
        else:
            padded[:n_points] = points
            mask[:n_points] = True

        return torch.from_numpy(padded), torch.from_numpy(mask)

    # =========================================================================
    # Annotation Loading
    # =========================================================================

    def _load_annotations(
        self, info: Dict[str, Any], aug_params: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load 3D bounding box annotations and apply augmentations.

        Each box is represented as [x, y, z, w, l, h, yaw, vx, vy] where
        (x, y, z) is the center, (w, l, h) is the size, yaw is the heading
        angle, and (vx, vy) is the velocity.

        Augmentations applied consistently with radar:
            - Horizontal flip: negate x, yaw, vx
            - Rotation: rotate center (x, y) and yaw
            - Scale: scale center (x, y, z) and size (w, l, h)

        Args:
            info: Sample info dict containing gt_boxes and gt_names.
            aug_params: Augmentation parameters.

        Returns:
            gt_boxes: (max_boxes, 9) float tensor of bounding boxes.
            gt_labels: (max_boxes,) int tensor of class indices (-1 for padding).
            gt_mask: (max_boxes,) bool tensor indicating valid annotations.
        """
        gt_boxes_arr = info.get('gt_boxes', np.zeros((0, 9), dtype=np.float32))
        gt_names = info.get('gt_names', [])

        # Filter to classes we care about
        valid_indices: List[int] = []
        valid_labels: List[int] = []
        for i, name in enumerate(gt_names):
            if name in self.class_name_to_idx:
                valid_indices.append(i)
                valid_labels.append(self.class_name_to_idx[name])

        if len(valid_indices) > 0:
            boxes = gt_boxes_arr[valid_indices].astype(np.float32)
            labels = np.array(valid_labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 9), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        # Apply augmentations to boxes
        boxes = self._augment_boxes(boxes, aug_params)

        # Pad to fixed size
        n_boxes = min(len(boxes), self.max_boxes)

        gt_boxes_padded = torch.zeros((self.max_boxes, 9), dtype=torch.float32)
        gt_labels_padded = torch.full(
            (self.max_boxes,), fill_value=-1, dtype=torch.long
        )
        gt_mask_padded = torch.zeros(self.max_boxes, dtype=torch.bool)

        if n_boxes > 0:
            gt_boxes_padded[:n_boxes] = torch.from_numpy(boxes[:n_boxes])
            gt_labels_padded[:n_boxes] = torch.from_numpy(labels[:n_boxes])
            gt_mask_padded[:n_boxes] = True

        return gt_boxes_padded, gt_labels_padded, gt_mask_padded

    def _augment_boxes(
        self, boxes: np.ndarray, aug_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply spatial augmentations to 3D bounding boxes.

        Must be consistent with the radar point cloud augmentations.

        Args:
            boxes: (N, 9) array of boxes [x, y, z, w, l, h, yaw, vx, vy].
            aug_params: Augmentation parameters (flip_x, rotation, scale).

        Returns:
            augmented: (N, 9) augmented bounding boxes.
        """
        if boxes.shape[0] == 0:
            return boxes

        boxes = boxes.copy()

        # Horizontal flip along x-axis
        if aug_params['flip_x']:
            boxes[:, 0] = -boxes[:, 0]    # x
            boxes[:, 6] = -boxes[:, 6]    # yaw (negate heading)
            boxes[:, 7] = -boxes[:, 7]    # vx

        # Random rotation around z-axis
        rotation = aug_params['rotation']
        if abs(rotation) > 1e-6:
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)

            # Rotate center (x, y)
            x = boxes[:, 0].copy()
            y = boxes[:, 1].copy()
            boxes[:, 0] = x * cos_r - y * sin_r
            boxes[:, 1] = x * sin_r + y * cos_r

            # Rotate heading
            boxes[:, 6] += rotation

            # Rotate velocity (vx, vy)
            vx = boxes[:, 7].copy()
            vy = boxes[:, 8].copy()
            boxes[:, 7] = vx * cos_r - vy * sin_r
            boxes[:, 8] = vx * sin_r + vy * cos_r

        # Random scaling
        scale = aug_params['scale']
        if abs(scale - 1.0) > 1e-6:
            boxes[:, 0:3] *= scale    # center (x, y, z)
            boxes[:, 3:6] *= scale    # size (w, l, h)
            boxes[:, 7:9] *= scale    # velocity (vx, vy)

        return boxes


# =============================================================================
# Collate Function
# =============================================================================


def collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for NuScenesRadarCameraDataset.

    Handles variable-length radar point clouds by using the pre-padded tensors
    and their associated masks. All tensors are stacked along batch dimension.

    Args:
        batch: List of sample dicts from __getitem__.

    Returns:
        Collated batch dict with batch dimension on all tensors.
    """
    images = torch.stack([s['images'] for s in batch], dim=0)
    intrinsics = torch.stack([s['intrinsics'] for s in batch], dim=0)
    extrinsics = torch.stack([s['extrinsics'] for s in batch], dim=0)
    radar_points = torch.stack([s['radar_points'] for s in batch], dim=0)
    radar_mask = torch.stack([s['radar_mask'] for s in batch], dim=0)
    gt_boxes = torch.stack([s['gt_boxes'] for s in batch], dim=0)
    gt_labels = torch.stack([s['gt_labels'] for s in batch], dim=0)
    gt_mask = torch.stack([s['gt_mask'] for s in batch], dim=0)

    return {
        'images': images,             # (B, 6, 3, H, W)
        'intrinsics': intrinsics,     # (B, 6, 3, 3)
        'extrinsics': extrinsics,     # (B, 6, 4, 4)
        'radar_points': radar_points, # (B, max_radar_points, D)
        'radar_mask': radar_mask,     # (B, max_radar_points)
        'gt_boxes': gt_boxes,         # (B, max_boxes, 9)
        'gt_labels': gt_labels,       # (B, max_boxes)
        'gt_mask': gt_mask,           # (B, max_boxes)
    }


# =============================================================================
# Factory Functions
# =============================================================================


def build_dataset(
    data_root: str,
    info_path: str,
    img_size: Tuple[int, int] = (900, 1600),
    point_cloud_range: Optional[List[float]] = None,
    num_sweeps: int = 6,
    radar_use_dims: Optional[List[int]] = None,
    max_radar_points: int = 30000,
    max_boxes: int = 300,
    class_names: Optional[List[str]] = None,
    augmentation: Optional[Dict[str, Any]] = None,
    is_train: bool = True,
) -> NuScenesRadarCameraDataset:
    """
    Build a NuScenesRadarCameraDataset instance.

    Args:
        data_root: Root directory of nuScenes data.
        info_path: Path to preprocessed info pickle file.
        img_size: Target image size (H, W).
        point_cloud_range: Spatial range for filtering radar points.
        num_sweeps: Number of radar sweeps to accumulate.
        radar_use_dims: Indices of radar point features to use.
        max_radar_points: Maximum number of radar points per sample.
        max_boxes: Maximum number of GT boxes per sample.
        class_names: Detection class names.
        augmentation: Augmentation configuration dict.
        is_train: Whether this is a training split.

    Returns:
        Configured NuScenesRadarCameraDataset instance.
    """
    return NuScenesRadarCameraDataset(
        data_root=data_root,
        info_path=info_path,
        img_size=img_size,
        point_cloud_range=point_cloud_range,
        num_sweeps=num_sweeps,
        radar_use_dims=radar_use_dims,
        max_radar_points=max_radar_points,
        max_boxes=max_boxes,
        class_names=class_names,
        augmentation=augmentation,
        is_train=is_train,
    )


def build_dataloader(
    data_root: str,
    info_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (900, 1600),
    point_cloud_range: Optional[List[float]] = None,
    num_sweeps: int = 6,
    radar_use_dims: Optional[List[int]] = None,
    max_radar_points: int = 30000,
    max_boxes: int = 300,
    class_names: Optional[List[str]] = None,
    augmentation: Optional[Dict[str, Any]] = None,
    is_train: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for the CRAFT radar-camera dataset.

    Args:
        data_root: Root directory of nuScenes data.
        info_path: Path to preprocessed info pickle file.
        batch_size: Number of samples per batch.
        num_workers: Number of data loading worker processes.
        img_size: Target image size (H, W).
        point_cloud_range: Spatial range for filtering radar points.
        num_sweeps: Number of radar sweeps to accumulate.
        radar_use_dims: Indices of radar point features to use.
        max_radar_points: Maximum number of radar points per sample.
        max_boxes: Maximum number of GT boxes per sample.
        class_names: Detection class names.
        augmentation: Augmentation configuration dict.
        is_train: Whether this is a training split.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Configured DataLoader instance.
    """
    dataset = build_dataset(
        data_root=data_root,
        info_path=info_path,
        img_size=img_size,
        point_cloud_range=point_cloud_range,
        num_sweeps=num_sweeps,
        radar_use_dims=radar_use_dims,
        max_radar_points=max_radar_points,
        max_boxes=max_boxes,
        class_names=class_names,
        augmentation=augmentation,
        is_train=is_train,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_train,
        persistent_workers=num_workers > 0,
    )

    return dataloader


def build_dataloader_from_config(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Build train and val DataLoaders from a CRAFT configuration dictionary.

    Expects a config structure matching craft_nuscenes.yaml:
        config['data']['root_path'], config['data']['info_path']['train'], etc.

    Args:
        config: Full experiment configuration dictionary.

    Returns:
        Dictionary with 'train' and 'val' DataLoader instances.
    """
    data_cfg = config['data']
    class_names = config.get('class_names', CLASS_NAMES)

    # Extract augmentation config
    augmentation = data_cfg.get('augmentation', {})

    # Common kwargs
    common_kwargs = {
        'data_root': data_cfg['root_path'],
        'img_size': tuple(data_cfg['image']['size']),
        'point_cloud_range': data_cfg['point_cloud']['range'],
        'num_sweeps': data_cfg['point_cloud']['num_sweeps'],
        'radar_use_dims': data_cfg['point_cloud']['radar_use_dims'],
        'max_radar_points': data_cfg['point_cloud']['max_pillars'],
        'class_names': class_names,
        'num_workers': data_cfg.get('num_workers', 4),
        'pin_memory': data_cfg.get('pin_memory', True),
    }

    # Build train loader
    train_loader = build_dataloader(
        info_path=data_cfg['info_path']['train'],
        batch_size=config.get('batch_size', 4),
        augmentation=augmentation,
        is_train=True,
        **common_kwargs,
    )

    # Build val loader
    val_loader = build_dataloader(
        info_path=data_cfg['info_path']['val'],
        batch_size=config.get('batch_size', 4),
        augmentation=None,
        is_train=False,
        **common_kwargs,
    )

    return {'train': train_loader, 'val': val_loader}
