"""
nuScenes dataset for PETR/PETRv2/StreamPETR.

Loads multi-view camera images, camera calibration (intrinsics/extrinsics),
3D bounding box annotations, temporal sequences, and ego-motion data.
Supports data augmentation with consistent camera matrix updates.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    HAS_NUSCENES = True
except ImportError:
    HAS_NUSCENES = False


# nuScenes camera names
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# nuScenes detection classes
DETECTION_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


class NuScenesDataset(Dataset):
    """nuScenes dataset for multi-view 3D object detection with PETR.

    Args:
        data_root: Path to the nuScenes dataset root directory.
        ann_file: Path to the annotation info pickle/json file (preprocessed).
        split: Dataset split ('train', 'val', 'test').
        num_cameras: Number of camera views to load (default 6).
        img_size: Target image size (H, W) after resizing.
        num_temporal_frames: Number of previous frames to load for
            temporal models (0 for PETR, >=1 for StreamPETR).
        pc_range: Point cloud range for coordinate normalization.
        augmentation: Whether to apply data augmentation.
        use_nuscenes_sdk: Whether to use nuscenes-devkit for data loading.
            If False, expects preprocessed annotation files.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: Optional[str] = None,
        split: str = "train",
        num_cameras: int = 6,
        img_size: Tuple[int, int] = (900, 1600),
        num_temporal_frames: int = 0,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        augmentation: bool = True,
        use_nuscenes_sdk: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.num_cameras = num_cameras
        self.img_size = img_size
        self.num_temporal_frames = num_temporal_frames
        self.pc_range = pc_range
        self.augmentation = augmentation and (split == "train")
        self.use_nuscenes_sdk = use_nuscenes_sdk

        # Load data info
        if use_nuscenes_sdk and HAS_NUSCENES:
            version = "v1.0-trainval" if split in ("train", "val") else "v1.0-test"
            self.nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
            self.samples = self._load_samples_from_sdk()
        elif ann_file is not None:
            self.nusc = None
            self.samples = self._load_samples_from_file(ann_file)
        else:
            raise ValueError(
                "Either provide ann_file or install nuscenes-devkit for SDK loading."
            )

        # Class name to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(DETECTION_CLASSES)}
        self.num_classes = len(DETECTION_CLASSES)

    def _load_samples_from_sdk(self) -> List[Dict[str, Any]]:
        """Load sample information using nuScenes SDK."""
        samples = []
        scene_splits = self._get_split_scenes()

        for scene in self.nusc.scene:
            if scene["name"] not in scene_splits:
                continue

            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = self.nusc.get("sample", sample_token)
                sample_info = {
                    "token": sample_token,
                    "timestamp": sample["timestamp"],
                    "scene_token": scene["token"],
                    "prev": sample["prev"],
                    "next": sample["next"],
                }
                samples.append(sample_info)
                sample_token = sample["next"]

        return samples

    def _get_split_scenes(self) -> List[str]:
        """Get scene names for the current split."""
        # Use official nuScenes split
        if self.split == "train":
            split_file = "train"
        elif self.split == "val":
            split_file = "val"
        else:
            split_file = "test"

        # Try to load from nuscenes split definitions
        try:
            from nuscenes.utils.splits import create_splits_scenes
            splits = create_splits_scenes()
            return splits[split_file]
        except (ImportError, KeyError):
            # Return all scenes if splits not available
            return [s["name"] for s in self.nusc.scene]

    def _load_samples_from_file(self, ann_file: str) -> List[Dict[str, Any]]:
        """Load preprocessed sample information from a JSON file."""
        with open(ann_file, "r") as f:
            data = json.load(f)
        return data["samples"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a training/evaluation sample.

        Returns:
            Dictionary containing:
                'images': Tensor (N_cams, 3, H, W) multi-view images.
                'intrinsics': Tensor (N_cams, 3, 3) camera intrinsics.
                'extrinsics': Tensor (N_cams, 4, 4) camera-to-ego transforms.
                'gt_labels': Tensor (num_gt,) class indices.
                'gt_bboxes': Tensor (num_gt, 10) bounding boxes
                    [cx, cy, cz, w, l, h, sin, cos, vx, vy].
                'ego_motion': Tensor (4, 4) ego-motion from prev to current
                    (identity if first frame or no temporal).
                'ego_motion_vec': Tensor (6,) velocity vector
                    [vx, vy, vz, wx, wy, wz].
                'prev_images': Optional tensor (T, N_cams, 3, H, W) previous frames.
                'prev_intrinsics': Optional tensor (T, N_cams, 3, 3).
                'prev_extrinsics': Optional tensor (T, N_cams, 4, 4).
                'prev_ego_motions': Optional tensor (T, 4, 4).
        """
        sample_info = self.samples[idx]

        if self.use_nuscenes_sdk and self.nusc is not None:
            return self._load_sample_sdk(sample_info)
        else:
            return self._load_sample_preprocessed(sample_info)

    def _load_sample_sdk(self, sample_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load sample using nuScenes SDK."""
        sample = self.nusc.get("sample", sample_info["token"])

        # Load multi-view camera data
        images = []
        intrinsics = []
        extrinsics = []

        for cam_name in CAMERA_NAMES[: self.num_cameras]:
            cam_data = self.nusc.get("sample_data", sample["data"][cam_name])
            img, K, T_cam2ego = self._load_camera_data(cam_data)
            images.append(img)
            intrinsics.append(K)
            extrinsics.append(T_cam2ego)

        images = torch.stack(images, dim=0)  # (N, 3, H, W)
        intrinsics = torch.stack(intrinsics, dim=0)  # (N, 3, 3)
        extrinsics = torch.stack(extrinsics, dim=0)  # (N, 4, 4)

        # Load annotations
        gt_labels, gt_bboxes = self._load_annotations(sample)

        # Compute ego-motion
        ego_motion, ego_motion_vec = self._compute_ego_motion(sample_info)

        # Apply augmentation
        if self.augmentation:
            images, intrinsics, gt_bboxes = self._augment(
                images, intrinsics, gt_bboxes
            )

        result = {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "gt_labels": gt_labels,
            "gt_bboxes": gt_bboxes,
            "ego_motion": ego_motion,
            "ego_motion_vec": ego_motion_vec,
        }

        # Load temporal frames if needed
        if self.num_temporal_frames > 0:
            temporal_data = self._load_temporal_frames(sample_info)
            result.update(temporal_data)

        return result

    def _load_camera_data(
        self, cam_data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load image and camera calibration for a single camera.

        Args:
            cam_data: nuScenes sample_data record for the camera.

        Returns:
            Tuple of (image, intrinsics, extrinsics):
                image: (3, H, W) tensor normalized to [0, 1].
                intrinsics: (3, 3) camera intrinsic matrix.
                extrinsics: (4, 4) camera-to-ego transformation.
        """
        # Load image
        img_path = os.path.join(self.data_root, cam_data["filename"])
        img = Image.open(img_path).convert("RGB")

        # Resize to target size
        orig_w, orig_h = img.size
        target_h, target_w = self.img_size
        img = img.resize((target_w, target_h), Image.BILINEAR)

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, H, W)

        # Load calibration
        calib = self.nusc.get(
            "calibrated_sensor", cam_data["calibrated_sensor_token"]
        )

        # Intrinsics (3x3)
        K = torch.tensor(calib["camera_intrinsic"], dtype=torch.float32)  # (3, 3)

        # Adjust intrinsics for image resize
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        K[0, :] *= scale_x
        K[1, :] *= scale_y

        # Extrinsics: sensor-to-ego transform
        rotation = Quaternion(calib["rotation"]).rotation_matrix
        translation = np.array(calib["translation"])
        T_cam2ego = np.eye(4, dtype=np.float32)
        T_cam2ego[:3, :3] = rotation
        T_cam2ego[:3, 3] = translation
        T_cam2ego = torch.from_numpy(T_cam2ego)

        return img_tensor, K, T_cam2ego

    def _load_annotations(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load 3D bounding box annotations for a sample.

        Args:
            sample: nuScenes sample record.

        Returns:
            Tuple of (labels, bboxes):
                labels: (num_gt,) tensor of class indices.
                bboxes: (num_gt, 10) tensor of [cx,cy,cz,w,l,h,sin,cos,vx,vy].
        """
        annotations = [
            self.nusc.get("sample_annotation", token)
            for token in sample["anns"]
        ]

        labels = []
        bboxes = []

        for ann in annotations:
            # Get class name (map to detection categories)
            class_name = self._map_class_name(ann["category_name"])
            if class_name not in self.class_to_idx:
                continue

            label = self.class_to_idx[class_name]

            # Get box in ego frame
            # Center (x, y, z)
            center = np.array(ann["translation"], dtype=np.float32)
            # Size (w, l, h) - nuScenes uses (w, l, h) format
            size = np.array(ann["size"], dtype=np.float32)
            # Rotation (yaw from quaternion)
            quat = Quaternion(ann["rotation"])
            yaw = quat.yaw_pitch_roll[0]

            # Velocity (vx, vy) in global frame
            velocity = self.nusc.box_velocity(ann["token"])[:2]
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(2, dtype=np.float32)

            # Transform annotation from global to ego frame
            ego_pose = self.nusc.get(
                "ego_pose",
                self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])[
                    "ego_pose_token"
                ],
            )
            ego_rotation = Quaternion(ego_pose["rotation"])
            ego_translation = np.array(ego_pose["translation"])

            # Transform center to ego frame
            center_ego = (
                ego_rotation.inverse.rotation_matrix @ (center - ego_translation)
            )

            # Transform yaw to ego frame
            yaw_ego = yaw - ego_rotation.yaw_pitch_roll[0]

            # Check if within perception range
            x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
            if not (
                x_min <= center_ego[0] <= x_max
                and y_min <= center_ego[1] <= y_max
                and z_min <= center_ego[2] <= z_max
            ):
                continue

            # Encode as [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy]
            bbox = np.array(
                [
                    center_ego[0],
                    center_ego[1],
                    center_ego[2],
                    size[0],
                    size[1],
                    size[2],
                    np.sin(yaw_ego),
                    np.cos(yaw_ego),
                    velocity[0],
                    velocity[1],
                ],
                dtype=np.float32,
            )

            labels.append(label)
            bboxes.append(bbox)

        if len(labels) == 0:
            return (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, 10, dtype=torch.float32),
            )

        return (
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(np.array(bboxes), dtype=torch.float32),
        )

    def _map_class_name(self, category_name: str) -> str:
        """Map nuScenes category name to detection class name."""
        mapping = {
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
        for prefix, cls_name in mapping.items():
            if category_name.startswith(prefix):
                return cls_name
        return category_name

    def _compute_ego_motion(
        self, sample_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ego-motion transformation from previous to current frame.

        Args:
            sample_info: Current sample info dictionary.

        Returns:
            Tuple of (ego_motion_matrix, ego_motion_vector):
                ego_motion_matrix: (4, 4) transformation from prev to curr.
                ego_motion_vector: (6,) [vx, vy, vz, wx, wy, wz].
        """
        if not sample_info["prev"] or self.nusc is None:
            return torch.eye(4, dtype=torch.float32), torch.zeros(6, dtype=torch.float32)

        # Get current and previous ego poses
        curr_sample = self.nusc.get("sample", sample_info["token"])
        prev_sample = self.nusc.get("sample", sample_info["prev"])

        curr_lidar = self.nusc.get("sample_data", curr_sample["data"]["LIDAR_TOP"])
        prev_lidar = self.nusc.get("sample_data", prev_sample["data"]["LIDAR_TOP"])

        curr_ego = self.nusc.get("ego_pose", curr_lidar["ego_pose_token"])
        prev_ego = self.nusc.get("ego_pose", prev_lidar["ego_pose_token"])

        # Compute relative transformation: T_curr_prev
        curr_rot = Quaternion(curr_ego["rotation"]).rotation_matrix
        curr_trans = np.array(curr_ego["translation"])
        T_global_curr = np.eye(4, dtype=np.float32)
        T_global_curr[:3, :3] = curr_rot
        T_global_curr[:3, 3] = curr_trans

        prev_rot = Quaternion(prev_ego["rotation"]).rotation_matrix
        prev_trans = np.array(prev_ego["translation"])
        T_global_prev = np.eye(4, dtype=np.float32)
        T_global_prev[:3, :3] = prev_rot
        T_global_prev[:3, 3] = prev_trans

        # T_curr_prev = T_curr_global @ T_global_prev = inv(T_global_curr) @ T_global_prev
        T_curr_global = np.linalg.inv(T_global_curr)
        T_curr_prev = T_curr_global @ T_global_prev

        # Compute velocity vector
        dt = (curr_sample["timestamp"] - prev_sample["timestamp"]) / 1e6  # seconds
        if dt > 0:
            translation_diff = T_curr_prev[:3, 3]
            velocity = translation_diff / dt  # vx, vy, vz in m/s

            # Angular velocity from rotation
            R_diff = T_curr_prev[:3, :3]
            # Approximate angular velocity from rotation matrix
            # Using log map: theta = arccos((trace(R)-1)/2)
            trace_val = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
            theta = np.arccos(trace_val)
            if abs(theta) > 1e-6:
                omega_hat = (R_diff - R_diff.T) / (2 * np.sin(theta)) * theta
                angular_vel = np.array(
                    [omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]]
                ) / dt
            else:
                angular_vel = np.zeros(3, dtype=np.float32)
        else:
            velocity = np.zeros(3, dtype=np.float32)
            angular_vel = np.zeros(3, dtype=np.float32)

        ego_motion_mat = torch.from_numpy(T_curr_prev.astype(np.float32))
        ego_motion_vec = torch.from_numpy(
            np.concatenate([velocity, angular_vel]).astype(np.float32)
        )

        return ego_motion_mat, ego_motion_vec

    def _load_temporal_frames(
        self, sample_info: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Load previous frame data for temporal models.

        Args:
            sample_info: Current sample info.

        Returns:
            Dictionary with temporal data tensors.
        """
        prev_images_list = []
        prev_intrinsics_list = []
        prev_extrinsics_list = []
        prev_ego_motions_list = []

        prev_token = sample_info["prev"]
        for t in range(self.num_temporal_frames):
            if not prev_token or self.nusc is None:
                # Pad with zeros if no previous frame available
                H, W = self.img_size
                prev_images_list.append(
                    torch.zeros(self.num_cameras, 3, H, W, dtype=torch.float32)
                )
                prev_intrinsics_list.append(
                    torch.eye(3, dtype=torch.float32)
                    .unsqueeze(0)
                    .expand(self.num_cameras, -1, -1)
                )
                prev_extrinsics_list.append(
                    torch.eye(4, dtype=torch.float32)
                    .unsqueeze(0)
                    .expand(self.num_cameras, -1, -1)
                )
                prev_ego_motions_list.append(
                    torch.eye(4, dtype=torch.float32)
                )
                break

            prev_sample = self.nusc.get("sample", prev_token)

            # Load camera data for previous frame
            imgs = []
            Ks = []
            Ts = []
            for cam_name in CAMERA_NAMES[: self.num_cameras]:
                cam_data = self.nusc.get(
                    "sample_data", prev_sample["data"][cam_name]
                )
                img, K, T = self._load_camera_data(cam_data)
                imgs.append(img)
                Ks.append(K)
                Ts.append(T)

            prev_images_list.append(torch.stack(imgs, dim=0))
            prev_intrinsics_list.append(torch.stack(Ks, dim=0))
            prev_extrinsics_list.append(torch.stack(Ts, dim=0))

            # Ego-motion from this previous frame to next
            prev_info = {"token": prev_token, "prev": prev_sample["prev"]}
            ego_mat, _ = self._compute_ego_motion(prev_info)
            prev_ego_motions_list.append(ego_mat)

            prev_token = prev_sample["prev"]

        result: Dict[str, torch.Tensor] = {}
        if prev_images_list:
            result["prev_images"] = torch.stack(prev_images_list, dim=0)
            result["prev_intrinsics"] = torch.stack(prev_intrinsics_list, dim=0)
            result["prev_extrinsics"] = torch.stack(prev_extrinsics_list, dim=0)
            result["prev_ego_motions"] = torch.stack(prev_ego_motions_list, dim=0)

        return result

    def _augment(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply data augmentation with consistent camera matrix updates.

        Augmentations applied:
        - Random horizontal flip (with corresponding bbox and intrinsics update)
        - Random resize (scale factor 0.9-1.1, with intrinsics update)
        - Color jitter (brightness, contrast, saturation)

        Args:
            images: (N, 3, H, W) multi-view images.
            intrinsics: (N, 3, 3) camera intrinsics.
            gt_bboxes: (num_gt, 10) bounding boxes.

        Returns:
            Augmented (images, intrinsics, gt_bboxes).
        """
        N, C, H, W = images.shape

        # Random horizontal flip (50% probability)
        if torch.rand(1).item() > 0.5:
            images = images.flip(dims=[-1])  # Flip width dimension

            # Update intrinsics: cx -> W - cx
            intrinsics = intrinsics.clone()
            intrinsics[:, 0, 2] = W - intrinsics[:, 0, 2]

            # Update bbox: flip y coordinate (lateral axis in ego frame)
            if gt_bboxes.numel() > 0:
                gt_bboxes = gt_bboxes.clone()
                gt_bboxes[:, 1] = -gt_bboxes[:, 1]  # Flip cy
                gt_bboxes[:, 6] = -gt_bboxes[:, 6]  # Flip sin(yaw)
                gt_bboxes[:, 9] = -gt_bboxes[:, 9]  # Flip vy

        # Random resize (scale 0.9 to 1.1)
        scale = 0.9 + torch.rand(1).item() * 0.2
        new_H = int(H * scale)
        new_W = int(W * scale)

        images_resized = torch.nn.functional.interpolate(
            images, size=(new_H, new_W), mode="bilinear", align_corners=False
        )

        # Pad or crop back to original size
        if new_H >= H and new_W >= W:
            # Crop center
            start_h = (new_H - H) // 2
            start_w = (new_W - W) // 2
            images = images_resized[:, :, start_h : start_h + H, start_w : start_w + W]
            # Update intrinsics for crop
            intrinsics = intrinsics.clone()
            intrinsics[:, 0, :] *= scale
            intrinsics[:, 1, :] *= scale
            intrinsics[:, 0, 2] -= start_w
            intrinsics[:, 1, 2] -= start_h
        else:
            # Pad with zeros
            pad_h = max(0, H - new_H)
            pad_w = max(0, W - new_W)
            images = torch.nn.functional.pad(
                images_resized, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
            images = images[:, :, :H, :W]
            # Update intrinsics for scale
            intrinsics = intrinsics.clone()
            intrinsics[:, 0, :] *= scale
            intrinsics[:, 1, :] *= scale

        # Color jitter
        brightness = 0.9 + torch.rand(1).item() * 0.2
        contrast = 0.9 + torch.rand(1).item() * 0.2
        saturation = 0.9 + torch.rand(1).item() * 0.2

        images = images * brightness
        mean = images.mean(dim=1, keepdim=True)
        images = (images - mean) * contrast + mean

        # Saturation adjustment
        gray = images.mean(dim=1, keepdim=True)
        images = gray + (images - gray) * saturation
        images = images.clamp(0, 1)

        return images, intrinsics, gt_bboxes

    def _load_sample_preprocessed(
        self, sample_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load sample from preprocessed files (non-SDK path).

        Expects sample_info to contain file paths for images and annotations.

        Args:
            sample_info: Dictionary with paths and metadata.

        Returns:
            Same format as _load_sample_sdk.
        """
        # Load images
        images = []
        for cam_path in sample_info["image_paths"][: self.num_cameras]:
            full_path = os.path.join(self.data_root, cam_path)
            img = Image.open(full_path).convert("RGB")
            target_h, target_w = self.img_size
            img = img.resize((target_w, target_h), Image.BILINEAR)
            img_tensor = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)
            images.append(img_tensor)

        images = torch.stack(images, dim=0)

        # Load calibration
        intrinsics = torch.tensor(
            sample_info["intrinsics"], dtype=torch.float32
        )[: self.num_cameras]
        extrinsics = torch.tensor(
            sample_info["extrinsics"], dtype=torch.float32
        )[: self.num_cameras]

        # Load annotations
        gt_labels = torch.tensor(sample_info["gt_labels"], dtype=torch.long)
        gt_bboxes = torch.tensor(sample_info["gt_bboxes"], dtype=torch.float32)

        # Ego motion
        if "ego_motion" in sample_info:
            ego_motion = torch.tensor(sample_info["ego_motion"], dtype=torch.float32)
        else:
            ego_motion = torch.eye(4, dtype=torch.float32)

        if "ego_motion_vec" in sample_info:
            ego_motion_vec = torch.tensor(
                sample_info["ego_motion_vec"], dtype=torch.float32
            )
        else:
            ego_motion_vec = torch.zeros(6, dtype=torch.float32)

        result = {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "gt_labels": gt_labels,
            "gt_bboxes": gt_bboxes,
            "ego_motion": ego_motion,
            "ego_motion_vec": ego_motion_vec,
        }

        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collation function for variable-size annotations.

    Stacks fixed-size tensors (images, calibration) and keeps
    variable-size tensors (annotations) as lists.

    Args:
        batch: List of sample dictionaries from __getitem__.

    Returns:
        Batched dictionary with stacked tensors where possible.
    """
    collated: Dict[str, Any] = {}

    # Stack fixed-size tensors
    collated["images"] = torch.stack([s["images"] for s in batch], dim=0)
    collated["intrinsics"] = torch.stack([s["intrinsics"] for s in batch], dim=0)
    collated["extrinsics"] = torch.stack([s["extrinsics"] for s in batch], dim=0)
    collated["ego_motion"] = torch.stack([s["ego_motion"] for s in batch], dim=0)
    collated["ego_motion_vec"] = torch.stack(
        [s["ego_motion_vec"] for s in batch], dim=0
    )

    # Keep annotations as lists (variable size)
    collated["gt_labels"] = [s["gt_labels"] for s in batch]
    collated["gt_bboxes"] = [s["gt_bboxes"] for s in batch]

    # Temporal data (if present)
    if "prev_images" in batch[0]:
        collated["prev_images"] = torch.stack(
            [s["prev_images"] for s in batch], dim=0
        )
        collated["prev_intrinsics"] = torch.stack(
            [s["prev_intrinsics"] for s in batch], dim=0
        )
        collated["prev_extrinsics"] = torch.stack(
            [s["prev_extrinsics"] for s in batch], dim=0
        )
        collated["prev_ego_motions"] = torch.stack(
            [s["prev_ego_motions"] for s in batch], dim=0
        )

    return collated
