"""
NuScenes multi-camera dataset for DETR3D 3D object detection.

Provides a PyTorch Dataset that loads 6-camera images with calibration data
and 3D bounding box annotations from the nuScenes dataset.
"""

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from typing import Dict, List, Optional, Tuple

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box

import torchvision.transforms.functional as F


# Camera names in the standard nuScenes multi-camera setup
CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

# nuScenes category name to class index mapping
# 10 detection classes used in nuScenes detection benchmark
CATEGORY_MAP = {
    'car': 0,
    'truck': 1,
    'construction_vehicle': 2,
    'bus': 3,
    'trailer': 4,
    'barrier': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'pedestrian': 8,
    'traffic_cone': 9,
}

# Expanded mapping from nuScenes fine-grained categories to the 10 detection classes
CATEGORY_NAME_TO_CLASS = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
}

# Point cloud range: [x_min, y_min, z_min, x_max, y_max, z_max]
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# Code size: (cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy)
CODE_SIZE = 10

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Original nuScenes image size (H, W)
ORIGINAL_IMAGE_SIZE = (900, 1600)


class NuScenesDataset(torch.utils.data.Dataset):
    """
    NuScenes multi-camera dataset for DETR3D 3D object detection.

    Loads 6 surround-view camera images with their calibration matrices and
    3D bounding box annotations transformed to the ego/lidar coordinate frame.

    Args:
        data_root: Path to the nuScenes dataset root directory.
        version: NuScenes dataset version (e.g., 'v1.0-trainval', 'v1.0-mini').
        split: Dataset split ('train' or 'val').
        image_size: Target image size as (H, W) after resizing.
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        max_objects: Maximum number of objects per sample for padding.
        use_lidar_coord: If True, annotations are in lidar frame; otherwise ego frame.
    """

    def __init__(
        self,
        data_root: str,
        version: str = 'v1.0-trainval',
        split: str = 'train',
        image_size: Tuple[int, int] = (256, 704),
        pc_range: Optional[List[float]] = None,
        max_objects: int = 300,
        use_lidar_coord: bool = True,
    ):
        super().__init__()

        self.data_root = data_root
        self.version = version
        self.split = split
        self.image_size = image_size  # (H, W)
        self.pc_range = pc_range if pc_range is not None else PC_RANGE
        self.max_objects = max_objects
        self.use_lidar_coord = use_lidar_coord

        # Initialize nuScenes devkit
        self.nusc = NuScenes(
            version=version,
            dataroot=data_root,
            verbose=False,
        )

        # Get scene names for the split
        split_scenes = create_splits_scenes()
        if split == 'train':
            scene_names = split_scenes['train']
        elif split == 'val':
            scene_names = split_scenes['val']
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'.")

        # Get scene tokens that belong to the split
        scene_tokens = set()
        for scene in self.nusc.scene:
            if scene['name'] in scene_names:
                scene_tokens.add(scene['token'])

        # Collect all sample tokens from matching scenes
        self.sample_tokens = []
        for sample in self.nusc.sample:
            if sample['scene_token'] in scene_tokens:
                self.sample_tokens.append(sample['token'])

        # Sort sample tokens for reproducibility
        self.sample_tokens.sort()

        # Compute resize scale factors
        self.resize_scale_h = self.image_size[0] / ORIGINAL_IMAGE_SIZE[0]
        self.resize_scale_w = self.image_size[1] / ORIGINAL_IMAGE_SIZE[1]

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with multi-camera images, calibrations, and annotations.

        Returns:
            Dictionary with:
                'images': (num_cams, 3, H, W) float32 tensor, normalized
                'intrinsics': (num_cams, 3, 3) float32 tensor, adjusted for resize
                'extrinsics': (num_cams, 4, 4) float32 tensor, lidar-to-camera transforms
                'labels': (num_objects,) int64 tensor, class indices
                'boxes_3d': (num_objects, 10) float32 tensor
        """
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)

        # Get lidar sensor calibration for coordinate transform
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_calib = self.nusc.get(
            'calibrated_sensor', lidar_data['calibrated_sensor_token']
        )
        lidar_ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        # Lidar sensor to ego transform
        lidar_sensor2ego = self._get_transform_matrix(
            lidar_calib['translation'], lidar_calib['rotation']
        )
        # Ego to global transform (at lidar timestamp)
        lidar_ego2global = self._get_transform_matrix(
            lidar_ego_pose['translation'], lidar_ego_pose['rotation']
        )

        # Load camera data
        images = []
        intrinsics = []
        extrinsics = []

        for cam_name in CAMERAS:
            cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
            cam_calib = self.nusc.get(
                'calibrated_sensor', cam_data['calibrated_sensor_token']
            )
            cam_ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

            # Load and preprocess image
            img_path = self.nusc.get_sample_data_path(sample['data'][cam_name])
            img = Image.open(img_path).convert('RGB')
            img = img.resize(
                (self.image_size[1], self.image_size[0]),  # PIL uses (W, H)
                Image.BILINEAR,
            )
            # Convert to tensor and normalize
            img_tensor = F.to_tensor(img)  # (3, H, W), float [0, 1]
            img_tensor = F.normalize(img_tensor, IMAGENET_MEAN, IMAGENET_STD)
            images.append(img_tensor)

            # Intrinsic matrix (3x3), adjusted for resize
            intrinsic = np.array(cam_calib['camera_intrinsic'], dtype=np.float32)
            intrinsic[0, :] *= self.resize_scale_w  # scale fx, cx
            intrinsic[1, :] *= self.resize_scale_h  # scale fy, cy
            intrinsics.append(torch.from_numpy(intrinsic))

            # Compute lidar-to-camera extrinsic transform
            # lidar2cam = cam_sensor2ego^{-1} @ cam_ego2global^{-1} @ lidar_ego2global @ lidar_sensor2ego
            cam_sensor2ego = self._get_transform_matrix(
                cam_calib['translation'], cam_calib['rotation']
            )
            cam_ego2global = self._get_transform_matrix(
                cam_ego_pose['translation'], cam_ego_pose['rotation']
            )

            # Full transform: lidar sensor -> lidar ego -> global -> cam ego -> cam sensor
            lidar2global = lidar_ego2global @ lidar_sensor2ego
            global2cam = np.linalg.inv(cam_ego2global) @ np.eye(4)
            cam_ego2cam_sensor = np.linalg.inv(cam_sensor2ego)

            lidar2cam = cam_ego2cam_sensor @ np.linalg.inv(cam_ego2global) @ lidar2global
            extrinsics.append(torch.from_numpy(lidar2cam.astype(np.float32)))

        images = torch.stack(images, dim=0)  # (num_cams, 3, H, W)
        intrinsics = torch.stack(intrinsics, dim=0)  # (num_cams, 3, 3)
        extrinsics = torch.stack(extrinsics, dim=0)  # (num_cams, 4, 4)

        # Process annotations
        labels, boxes_3d = self._get_annotations(sample, lidar_calib, lidar_ego_pose)

        return {
            'images': images,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'labels': labels,
            'boxes_3d': boxes_3d,
        }

    def _get_transform_matrix(
        self, translation: List[float], rotation: List[float]
    ) -> np.ndarray:
        """
        Construct a 4x4 homogeneous transformation matrix from translation and quaternion.

        Args:
            translation: [x, y, z] translation vector.
            rotation: [w, x, y, z] quaternion (nuScenes format).

        Returns:
            4x4 transformation matrix as numpy float64 array.
        """
        transform = np.eye(4, dtype=np.float64)
        quat = Quaternion(rotation)
        transform[:3, :3] = quat.rotation_matrix
        transform[:3, 3] = np.array(translation)
        return transform

    def _get_annotations(
        self,
        sample: dict,
        lidar_calib: dict,
        lidar_ego_pose: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract 3D bounding box annotations for a sample.

        Annotations are transformed to the lidar coordinate frame and encoded as:
        (cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy)

        Args:
            sample: NuScenes sample record.
            lidar_calib: Calibrated sensor record for LIDAR_TOP.
            lidar_ego_pose: Ego pose record at lidar timestamp.

        Returns:
            labels: (N,) int64 tensor of class indices.
            boxes_3d: (N, 10) float32 tensor of encoded box parameters.
        """
        labels = []
        boxes_3d = []

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)

            # Map category to class index
            category_name = ann['category_name']
            class_name = None
            for prefix, cls in CATEGORY_NAME_TO_CLASS.items():
                if category_name.startswith(prefix):
                    class_name = cls
                    break

            if class_name is None:
                continue  # Skip annotations not in our class set

            class_idx = CATEGORY_MAP[class_name]

            # Create Box in global frame
            box = Box(
                center=ann['translation'],
                size=ann['size'],  # (w, l, h) in nuScenes
                orientation=Quaternion(ann['rotation']),
                velocity=self.nusc.box_velocity(ann_token)[:2]  # vx, vy in global
            )

            # Transform box from global to ego frame
            box.translate(-np.array(lidar_ego_pose['translation']))
            box.rotate(Quaternion(lidar_ego_pose['rotation']).inverse)

            if self.use_lidar_coord:
                # Transform from ego to lidar sensor frame
                box.translate(-np.array(lidar_calib['translation']))
                box.rotate(Quaternion(lidar_calib['rotation']).inverse)

            # Extract center (x, y, z)
            cx, cy, cz = box.center

            # Extract size: nuScenes Box stores (w, l, h)
            w, l, h = box.wlh

            # Extract yaw angle from quaternion
            # In nuScenes, yaw is rotation around the up (z) axis
            yaw = self._quaternion_to_yaw(box.orientation)
            sin_yaw = np.sin(yaw)
            cos_yaw = np.cos(yaw)

            # Velocity (vx, vy) - transform to local frame
            velocity = box.velocity[:2]
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(2)

            # Encode box: (cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy)
            box_encoded = np.array(
                [cx, cy, cz, w, l, h, sin_yaw, cos_yaw, velocity[0], velocity[1]],
                dtype=np.float32,
            )

            # Filter boxes outside the point cloud range
            if not self._is_in_pc_range(cx, cy, cz):
                continue

            labels.append(class_idx)
            boxes_3d.append(box_encoded)

        if len(labels) == 0:
            labels = torch.zeros(0, dtype=torch.int64)
            boxes_3d = torch.zeros(0, CODE_SIZE, dtype=torch.float32)
        else:
            labels = torch.tensor(labels, dtype=torch.int64)
            boxes_3d = torch.from_numpy(np.stack(boxes_3d, axis=0))

        return labels, boxes_3d

    def _quaternion_to_yaw(self, q: Quaternion) -> float:
        """
        Extract yaw angle (rotation around z-axis) from a quaternion.

        Args:
            q: Quaternion representing orientation.

        Returns:
            Yaw angle in radians.
        """
        # Yaw from rotation matrix: atan2(R[1,0], R[0,0])
        rot_mat = q.rotation_matrix
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        return yaw

    def _is_in_pc_range(self, x: float, y: float, z: float) -> bool:
        """
        Check if a 3D point is within the configured point cloud range.

        Args:
            x, y, z: 3D coordinates.

        Returns:
            True if the point is within range.
        """
        return (
            self.pc_range[0] <= x <= self.pc_range[3]
            and self.pc_range[1] <= y <= self.pc_range[4]
            and self.pc_range[2] <= z <= self.pc_range[5]
        )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that handles variable number of annotations per sample.

    Pads labels and boxes_3d to the maximum number of objects in the batch.
    Padded labels are set to -1 (ignore index) and padded boxes are set to 0.

    Args:
        batch: List of sample dictionaries from NuScenesDataset.__getitem__.

    Returns:
        Collated dictionary with:
            'images': (B, num_cams, 3, H, W) float32 tensor
            'intrinsics': (B, num_cams, 3, 3) float32 tensor
            'extrinsics': (B, num_cams, 4, 4) float32 tensor
            'labels': (B, max_objects) int64 tensor, padded with -1
            'boxes_3d': (B, max_objects, 10) float32 tensor, padded with 0
            'num_objects': (B,) int64 tensor, actual number of objects per sample
    """
    batch_size = len(batch)

    # Stack fixed-size tensors
    images = torch.stack([b['images'] for b in batch], dim=0)
    intrinsics = torch.stack([b['intrinsics'] for b in batch], dim=0)
    extrinsics = torch.stack([b['extrinsics'] for b in batch], dim=0)

    # Find the maximum number of objects in this batch
    num_objects_list = [b['labels'].shape[0] for b in batch]
    max_objects = max(num_objects_list) if num_objects_list else 0
    # Ensure at least 1 slot to avoid empty tensor issues
    max_objects = max(max_objects, 1)

    # Pad labels and boxes
    labels = torch.full((batch_size, max_objects), -1, dtype=torch.int64)
    boxes_3d = torch.zeros(batch_size, max_objects, CODE_SIZE, dtype=torch.float32)
    num_objects = torch.zeros(batch_size, dtype=torch.int64)

    for i, b in enumerate(batch):
        n = b['labels'].shape[0]
        num_objects[i] = n
        if n > 0:
            labels[i, :n] = b['labels']
            boxes_3d[i, :n] = b['boxes_3d']

    return {
        'images': images,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'labels': labels,
        'boxes_3d': boxes_3d,
        'num_objects': num_objects,
    }
