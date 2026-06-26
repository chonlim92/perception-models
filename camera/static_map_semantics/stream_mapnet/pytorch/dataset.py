"""
NuScenes Map Dataset for StreamMapNet.

Provides temporal sequences of multi-camera images with vectorized map ground truth.
Each sample includes 6 camera images, camera calibration, ego-motion between frames,
and ground truth map elements (lane dividers, road boundaries, pedestrian crossings)
as ordered point sequences.
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


class NuScenesMapDataset(Dataset):
    """
    Dataset for StreamMapNet training/evaluation on nuScenes.

    Each item returns a dict with:
        - images: (num_cams, 3, H, W) float tensor, normalized
        - intrinsics: (num_cams, 3, 3) camera intrinsic matrices
        - extrinsics: (num_cams, 4, 4) camera-to-ego transforms
        - ego_motion: (4, 4) transform from previous ego frame to current
        - gt_labels: (max_elements,) class labels, padded with -1
        - gt_points: (max_elements, num_points, 2) point coordinates in [0,1]
        - gt_mask: (max_elements,) bool mask for valid elements
    """

    MAP_CLASSES: List[str] = ['lane_divider', 'road_boundary', 'ped_crossing']
    CAMERA_NAMES: List[str] = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT',
    ]

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        img_size: Tuple[int, int] = (256, 704),
        x_range: Tuple[float, float] = (-30.0, 30.0),
        y_range: Tuple[float, float] = (-15.0, 15.0),
        num_points: int = 20,
        max_elements: int = 50,
        is_train: bool = True,
    ) -> None:
        """
        Initialize the NuScenes map dataset.

        Args:
            data_root: Root directory of nuScenes data
                       (e.g. '/data/nuscenes').
            ann_file: Path to the annotation pickle file containing
                      pre-computed GT map elements and sample metadata.
            img_size: Target image size (H, W) after resizing.
            x_range: BEV x-axis range in meters (left/right of ego).
            y_range: BEV y-axis range in meters (behind/ahead of ego).
            num_points: Fixed number of points per map element polyline.
            max_elements: Maximum number of GT map elements per sample.
            is_train: Whether this is for training (enables augmentation).
        """
        super().__init__()

        self.data_root = data_root
        self.img_size = img_size
        self.x_range = x_range
        self.y_range = y_range
        self.num_points = num_points
        self.max_elements = max_elements
        self.is_train = is_train

        # Load annotation pickle
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}"
            )
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        # Expected pickle structure:
        # {
        #     'samples': [
        #         {
        #             'token': str,
        #             'scene_token': str,
        #             'timestamp': int,
        #             'is_first_frame': bool,
        #             'ego_pose': {'translation': [x,y,z], 'rotation': [w,x,y,z]},
        #             'cams': {
        #                 'CAM_FRONT': {
        #                     'filename': 'samples/CAM_FRONT/n015-...-1234.jpg',
        #                     'intrinsic': [[fx,0,cx],[0,fy,cy],[0,0,1]],
        #                     'extrinsic': [[r00,...,t0],...,[0,0,0,1]],
        #                 },
        #                 ...
        #             },
        #             'map_elements': [
        #                 {
        #                     'class': 'lane_divider',
        #                     'points': [[x1,y1],[x2,y2],...],  # ego coords
        #                 },
        #                 ...
        #             ],
        #         },
        #         ...
        #     ]
        # }
        self.samples: List[Dict[str, Any]] = data['samples']

        # Build scene-aware index for temporal lookups
        self._build_scene_index()

        # Image transforms
        self.img_transform = self._build_img_transform()

    def _build_scene_index(self) -> None:
        """Build mapping from sample index to previous sample in same scene."""
        self.prev_indices: List[Optional[int]] = [None] * len(self.samples)
        scene_last_idx: Dict[str, int] = {}

        for idx, sample in enumerate(self.samples):
            scene_token = sample['scene_token']
            if sample.get('is_first_frame', False):
                # First frame in scene: no previous frame
                self.prev_indices[idx] = None
            elif scene_token in scene_last_idx:
                self.prev_indices[idx] = scene_last_idx[scene_token]
            else:
                self.prev_indices[idx] = None
            scene_last_idx[scene_token] = idx

    def _build_img_transform(self) -> transforms.Compose:
        """Build image preprocessing pipeline."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing images, calibration, ego motion,
            and ground truth map elements.
        """
        sample = self.samples[idx]

        # Load multi-camera images
        images, intrinsics, extrinsics = self._load_images(sample)

        # Compute ego motion from previous frame
        ego_motion = self._get_ego_motion(idx)

        # Process ground truth map elements
        gt_labels, gt_points, gt_mask = self._process_gt(sample)

        return {
            'images': images,           # (6, 3, H, W)
            'intrinsics': intrinsics,   # (6, 3, 3)
            'extrinsics': extrinsics,   # (6, 4, 4)
            'ego_motion': ego_motion,   # (4, 4)
            'gt_labels': gt_labels,     # (max_elements,)
            'gt_points': gt_points,     # (max_elements, num_points, 2)
            'gt_mask': gt_mask,         # (max_elements,)
        }

    def _load_images(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and transform multi-camera images with calibration.

        Args:
            sample: Sample metadata dict containing camera info.

        Returns:
            images: (num_cams, 3, H, W) normalized float tensor.
            intrinsics: (num_cams, 3, 3) camera intrinsic matrices.
            extrinsics: (num_cams, 4, 4) camera-to-ego transform matrices.
        """
        num_cams = len(self.CAMERA_NAMES)
        images_list: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        extrinsics_list: List[torch.Tensor] = []

        for cam_name in self.CAMERA_NAMES:
            cam_data = sample['cams'][cam_name]

            # Load image
            img_path = os.path.join(self.data_root, cam_data['filename'])
            if not os.path.isfile(img_path):
                raise FileNotFoundError(
                    f"Camera image not found: {img_path}"
                )
            img = Image.open(img_path).convert('RGB')

            # Compute resize scale factors for intrinsic adjustment
            orig_w, orig_h = img.size
            target_h, target_w = self.img_size
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            # Apply image transforms (resize + normalize)
            img_tensor = self.img_transform(img)
            images_list.append(img_tensor)

            # Adjust intrinsics for resize
            intrinsic = np.array(cam_data['intrinsic'], dtype=np.float32)
            intrinsic[0, :] *= scale_x  # fx, cx
            intrinsic[1, :] *= scale_y  # fy, cy
            intrinsics_list.append(torch.from_numpy(intrinsic))

            # Extrinsic: camera-to-ego 4x4 transform
            extrinsic = np.array(cam_data['extrinsic'], dtype=np.float32)
            extrinsics_list.append(torch.from_numpy(extrinsic))

        images = torch.stack(images_list, dim=0)        # (6, 3, H, W)
        intrinsics = torch.stack(intrinsics_list, dim=0)  # (6, 3, 3)
        extrinsics = torch.stack(extrinsics_list, dim=0)  # (6, 4, 4)

        return images, intrinsics, extrinsics

    def _get_ego_motion(self, idx: int) -> torch.Tensor:
        """
        Compute the 4x4 relative transform from previous ego frame to current.

        For the first frame in a scene, returns identity matrix.

        The ego motion T_prev->curr transforms a point in the previous ego
        coordinate frame into the current ego coordinate frame:
            p_curr = T_prev->curr @ p_prev

        Args:
            idx: Current sample index.

        Returns:
            ego_motion: (4, 4) float tensor representing the relative transform.
        """
        prev_idx = self.prev_indices[idx]

        if prev_idx is None:
            # First frame in scene: no motion
            return torch.eye(4, dtype=torch.float32)

        curr_sample = self.samples[idx]
        prev_sample = self.samples[prev_idx]

        # Current ego pose (global frame)
        curr_trans = np.array(
            curr_sample['ego_pose']['translation'], dtype=np.float64
        )
        curr_rot = Quaternion(curr_sample['ego_pose']['rotation'])

        # Previous ego pose (global frame)
        prev_trans = np.array(
            prev_sample['ego_pose']['translation'], dtype=np.float64
        )
        prev_rot = Quaternion(prev_sample['ego_pose']['rotation'])

        # Compute T_global_to_curr: transforms from global to current ego frame
        curr_rot_inv = curr_rot.inverse
        # T_prev_to_curr = T_global_to_curr @ T_prev_to_global
        # T_prev_to_global: translation=prev_trans, rotation=prev_rot
        # T_global_to_curr: translation=-R_curr_inv @ curr_trans, rotation=R_curr_inv

        # Relative rotation: R_curr_inv * R_prev
        rel_rot = curr_rot_inv * prev_rot
        rel_rot_matrix = rel_rot.rotation_matrix  # (3, 3)

        # Relative translation
        rel_trans = curr_rot_inv.rotate(prev_trans - curr_trans)

        # Build 4x4 transform matrix
        ego_motion = np.eye(4, dtype=np.float32)
        ego_motion[:3, :3] = rel_rot_matrix.astype(np.float32)
        ego_motion[:3, 3] = rel_trans.astype(np.float32)

        return torch.from_numpy(ego_motion)

    def _normalize_points(
        self, points: np.ndarray
    ) -> np.ndarray:
        """
        Transform map element points from ego coordinates to [0, 1] normalized
        BEV coordinates.

        The BEV coordinate system:
            x_bev = (x_ego - x_min) / (x_max - x_min)
            y_bev = (y_ego - y_min) / (y_max - y_min)

        Points outside the BEV range are clipped to [0, 1].

        Args:
            points: (N, 2) array of points in ego coordinates (x, y).

        Returns:
            normalized: (N, 2) array of points in [0, 1] BEV coordinates.
        """
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        normalized = np.zeros_like(points, dtype=np.float32)
        normalized[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        normalized[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Clip to [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized

    def _interpolate_points(
        self, points: np.ndarray, num_points: int
    ) -> np.ndarray:
        """
        Resample a polyline to a fixed number of equally-spaced points.

        Uses linear interpolation along the cumulative arc length.

        Args:
            points: (M, 2) array of polyline vertices.
            num_points: Target number of output points.

        Returns:
            resampled: (num_points, 2) array of evenly-spaced points.
        """
        if len(points) < 2:
            # Degenerate case: duplicate the single point
            return np.tile(points[0], (num_points, 1))

        # Compute cumulative arc length
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative = np.zeros(len(points), dtype=np.float64)
        cumulative[1:] = np.cumsum(segment_lengths)

        total_length = cumulative[-1]
        if total_length < 1e-8:
            # All points are identical
            return np.tile(points[0], (num_points, 1)).astype(np.float32)

        # Target arc lengths for uniform spacing
        targets = np.linspace(0.0, total_length, num_points)

        # Interpolate x and y independently along arc length
        resampled = np.zeros((num_points, 2), dtype=np.float32)
        resampled[:, 0] = np.interp(targets, cumulative, points[:, 0])
        resampled[:, 1] = np.interp(targets, cumulative, points[:, 1])

        return resampled

    def _process_gt(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process ground truth map elements into fixed-size padded tensors.

        Args:
            sample: Sample metadata dict containing map_elements.

        Returns:
            gt_labels: (max_elements,) int tensor of class indices, -1 for padding.
            gt_points: (max_elements, num_points, 2) float tensor of normalized
                       BEV point coordinates.
            gt_mask: (max_elements,) bool tensor indicating valid elements.
        """
        map_elements = sample.get('map_elements', [])

        gt_labels = torch.full(
            (self.max_elements,), fill_value=-1, dtype=torch.long
        )
        gt_points = torch.zeros(
            (self.max_elements, self.num_points, 2), dtype=torch.float32
        )
        gt_mask = torch.zeros(self.max_elements, dtype=torch.bool)

        valid_count = 0
        for element in map_elements:
            if valid_count >= self.max_elements:
                break

            cls_name = element['class']
            if cls_name not in self.MAP_CLASSES:
                continue

            points = np.array(element['points'], dtype=np.float64)
            if len(points) < 2:
                continue

            # Filter elements completely outside BEV range
            x_min, x_max = self.x_range
            y_min, y_max = self.y_range
            if (points[:, 0].max() < x_min or points[:, 0].min() > x_max or
                    points[:, 1].max() < y_min or points[:, 1].min() > y_max):
                continue

            # Resample to fixed number of points
            resampled = self._interpolate_points(points, self.num_points)

            # Normalize to [0, 1] BEV coordinates
            normalized = self._normalize_points(resampled)

            # Store
            cls_idx = self.MAP_CLASSES.index(cls_name)
            gt_labels[valid_count] = cls_idx
            gt_points[valid_count] = torch.from_numpy(normalized)
            gt_mask[valid_count] = True
            valid_count += 1

        return gt_labels, gt_points, gt_mask


def collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for NuScenesMapDataset.

    Pads GT elements to the maximum number of valid elements across the batch
    for memory efficiency. All other tensors are simply stacked.

    Args:
        batch: List of sample dicts from __getitem__.

    Returns:
        Collated batch dict with an additional batch dimension on all tensors.
    """
    # Stack fixed-size tensors directly
    images = torch.stack([s['images'] for s in batch], dim=0)
    intrinsics = torch.stack([s['intrinsics'] for s in batch], dim=0)
    extrinsics = torch.stack([s['extrinsics'] for s in batch], dim=0)
    ego_motion = torch.stack([s['ego_motion'] for s in batch], dim=0)

    # For GT elements, find the max number of valid elements in this batch
    # and pad accordingly for efficient batching
    batch_size = len(batch)
    max_valid = max(
        int(s['gt_mask'].sum().item()) for s in batch
    )
    # Use at least 1 to avoid zero-size tensors
    max_valid = max(max_valid, 1)

    num_points = batch[0]['gt_points'].shape[1]

    gt_labels = torch.full(
        (batch_size, max_valid), fill_value=-1, dtype=torch.long
    )
    gt_points = torch.zeros(
        (batch_size, max_valid, num_points, 2), dtype=torch.float32
    )
    gt_mask = torch.zeros(
        (batch_size, max_valid), dtype=torch.bool
    )

    for i, sample in enumerate(batch):
        n_valid = int(sample['gt_mask'].sum().item())
        if n_valid > 0:
            gt_labels[i, :n_valid] = sample['gt_labels'][:n_valid]
            gt_points[i, :n_valid] = sample['gt_points'][:n_valid]
            gt_mask[i, :n_valid] = True

    return {
        'images': images,           # (B, 6, 3, H, W)
        'intrinsics': intrinsics,   # (B, 6, 3, 3)
        'extrinsics': extrinsics,   # (B, 6, 4, 4)
        'ego_motion': ego_motion,   # (B, 4, 4)
        'gt_labels': gt_labels,     # (B, max_valid)
        'gt_points': gt_points,     # (B, max_valid, num_points, 2)
        'gt_mask': gt_mask,         # (B, max_valid)
    }


def build_dataloader(
    data_root: str,
    ann_file: str,
    batch_size: int = 4,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (256, 704),
    x_range: Tuple[float, float] = (-30.0, 30.0),
    y_range: Tuple[float, float] = (-15.0, 15.0),
    num_points: int = 20,
    max_elements: int = 50,
    is_train: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for NuScenes map dataset.

    Args:
        data_root: Root directory of nuScenes data.
        ann_file: Path to annotation pickle file.
        batch_size: Number of samples per batch.
        num_workers: Number of data loading workers.
        img_size: Target image size (H, W).
        x_range: BEV x-axis range in meters.
        y_range: BEV y-axis range in meters.
        num_points: Fixed number of points per map element.
        max_elements: Maximum number of GT elements per sample.
        is_train: Whether this is for training.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Configured DataLoader instance.
    """
    dataset = NuScenesMapDataset(
        data_root=data_root,
        ann_file=ann_file,
        img_size=img_size,
        x_range=x_range,
        y_range=y_range,
        num_points=num_points,
        max_elements=max_elements,
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
