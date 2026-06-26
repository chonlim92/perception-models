"""
NuScenes Map Dataset for MapTR.

Loads multi-camera images, camera calibration, and vectorized map annotations
from nuScenes for training and evaluation of MapTR models.
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:
    Image = None


# nuScenes camera names in canonical order
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class ImageTransform:
    """
    Image preprocessing: resize, normalize (ImageNet mean/std), and pad to
    a size divisible by a given factor.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (480, 800),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        to_rgb: bool = True,
        size_divisor: int = 32,
    ):
        """
        Args:
            target_size: (H, W) to resize images to.
            mean: Per-channel mean for normalization (RGB, ImageNet default).
            std: Per-channel std for normalization (RGB, ImageNet default).
            to_rgb: If True, convert BGR to RGB (for OpenCV-loaded images).
            size_divisor: Pad image dimensions to be divisible by this value.
        """
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[torch.Tensor, float, float, Tuple[int, int]]:
        """
        Transform a single image.

        Args:
            img: numpy array of shape (H, W, 3) in uint8, either RGB or BGR.

        Returns:
            img_tensor: Transformed image tensor of shape (3, H_pad, W_pad).
            scale_h: Height scale factor applied during resize.
            scale_w: Width scale factor applied during resize.
            pad_shape: (H_pad, W_pad) after padding.
        """
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = self.target_size

        # Resize
        if Image is not None:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
            img = np.array(pil_img)
        else:
            # Fallback using simple numpy resize (nearest neighbor approximation)
            import cv2
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        scale_h = target_h / orig_h
        scale_w = target_w / orig_w

        # Convert to RGB if needed
        if self.to_rgb and img.shape[2] == 3:
            # Assume input is already RGB from PIL; skip conversion
            pass

        # Normalize to float [0, 1] then standardize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # Transpose to (3, H, W)
        img = img.transpose(2, 0, 1)

        # Pad to be divisible by size_divisor
        _, h, w = img.shape
        pad_h = (self.size_divisor - h % self.size_divisor) % self.size_divisor
        pad_w = (self.size_divisor - w % self.size_divisor) % self.size_divisor

        if pad_h > 0 or pad_w > 0:
            img = np.pad(
                img, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )

        img_tensor = torch.from_numpy(img.copy())
        pad_shape = (img_tensor.shape[1], img_tensor.shape[2])

        return img_tensor, scale_h, scale_w, pad_shape


class NuScenesMapDataset(Dataset):
    """
    NuScenes dataset for vectorized HD map construction with MapTR.

    Loads 6 surround-view camera images, their calibration matrices, and
    vectorized map annotations (polylines) for map element classes.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        pipeline: Optional[Dict[str, Any]] = None,
        map_classes: Optional[List[str]] = None,
        num_points_per_instance: int = 20,
        coord_range: Optional[List[float]] = None,
    ):
        """
        Args:
            data_root: Root path of nuScenes dataset (containing 'samples/', 'sweeps/', etc.).
            ann_file: Path to annotation file (JSON or pickle) with scene/sample info.
            pipeline: Dict specifying image transform parameters. Keys:
                - target_size: (H, W) tuple
                - mean, std: normalization params
                - size_divisor: int
            map_classes: List of map element class names to use.
            num_points_per_instance: Fixed number of points to resample each polyline to.
            coord_range: [x_min, y_min, x_max, y_max] for BEV coordinate normalization.
        """
        self.data_root = data_root
        self.ann_file = ann_file
        self.num_points_per_instance = num_points_per_instance

        if map_classes is None:
            self.map_classes = ["ped_crossing", "divider", "boundary"]
        else:
            self.map_classes = map_classes
        self.num_classes = len(self.map_classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.map_classes)}

        if coord_range is None:
            self.coord_range = [-30.0, -15.0, 30.0, 15.0]
        else:
            self.coord_range = coord_range

        # Build image transform pipeline
        if pipeline is None:
            pipeline = {}
        self.img_transform = ImageTransform(
            target_size=pipeline.get("target_size", (480, 800)),
            mean=pipeline.get("mean", (0.485, 0.456, 0.406)),
            std=pipeline.get("std", (0.229, 0.224, 0.225)),
            to_rgb=pipeline.get("to_rgb", True),
            size_divisor=pipeline.get("size_divisor", 32),
        )

        # Load annotations
        self.samples = self._load_annotations()

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load and parse the annotation file.

        Expected format (JSON or pickle):
        {
            "infos": [
                {
                    "token": "sample_token_string",
                    "timestamp": 1234567890,
                    "cams": {
                        "CAM_FRONT": {
                            "data_path": "samples/CAM_FRONT/xxx.jpg",
                            "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                            "extrinsics": [[r00,...,t0], ..., [0,0,0,1]]  # 4x4 cam-to-ego
                        },
                        ...
                    },
                    "map_annotations": [
                        {
                            "class": "divider",
                            "points": [[x1, y1], [x2, y2], ...]  # BEV coordinates in meters
                        },
                        ...
                    ]
                },
                ...
            ]
        }

        Returns:
            List of sample info dicts.
        """
        ann_path = self.ann_file
        if not os.path.isabs(ann_path):
            ann_path = os.path.join(self.data_root, ann_path)

        if ann_path.endswith(".pkl") or ann_path.endswith(".pickle"):
            with open(ann_path, "rb") as f:
                data = pickle.load(f)
        elif ann_path.endswith(".json"):
            with open(ann_path, "r") as f:
                data = json.load(f)
        else:
            # Try pickle first, then JSON
            try:
                with open(ann_path, "rb") as f:
                    data = pickle.load(f)
            except (pickle.UnpicklingError, UnicodeDecodeError):
                with open(ann_path, "r") as f:
                    data = json.load(f)

        # Support both flat list and dict with "infos" key
        if isinstance(data, dict):
            samples = data.get("infos", data.get("samples", []))
        elif isinstance(data, list):
            samples = data
        else:
            raise ValueError(f"Unsupported annotation file format: {type(data)}")

        return samples

    def _resample_polyline(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """
        Resample a polyline to a fixed number of equally-spaced points via
        linear interpolation along the cumulative arc length.

        Args:
            points: Array of shape (N, 2) with 2D coordinates.
            num_points: Target number of points after resampling.

        Returns:
            Resampled points array of shape (num_points, 2).
        """
        if len(points) == 0:
            return np.zeros((num_points, 2), dtype=np.float32)

        if len(points) == 1:
            return np.tile(points[0], (num_points, 1)).astype(np.float32)

        # Compute cumulative arc length
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]

        if total_length < 1e-8:
            # Degenerate polyline (all points coincident)
            return np.tile(points[0], (num_points, 1)).astype(np.float32)

        # Desired sample positions along the arc length
        target_lengths = np.linspace(0.0, total_length, num_points)

        # Interpolate x and y separately
        resampled = np.zeros((num_points, 2), dtype=np.float32)
        resampled[:, 0] = np.interp(target_lengths, cumulative_lengths, points[:, 0])
        resampled[:, 1] = np.interp(target_lengths, cumulative_lengths, points[:, 1])

        return resampled

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize BEV points from metric coordinates to [0, 1] range
        based on the configured coord_range.

        Args:
            points: Array of shape (N, 2) in meters.

        Returns:
            Normalized points in [0, 1] range, shape (N, 2).
        """
        x_min, y_min, x_max, y_max = self.coord_range
        normalized = np.zeros_like(points)
        normalized[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        normalized[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        # Clamp to [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a single sample.

        Returns:
            Dict with keys:
                - images: Tensor [6, 3, H, W] (preprocessed multi-camera images)
                - intrinsics: Tensor [6, 3, 3] camera intrinsic matrices
                - extrinsics: Tensor [6, 4, 4] camera-to-ego extrinsic matrices
                - gt_labels: Tensor [num_instances] class indices
                - gt_points: Tensor [num_instances, num_points_per_instance, 2] normalized coords
                - sample_token: str identifier (if available)
        """
        sample_info = self.samples[idx]

        # ---- Load camera images and calibration ----
        images = []
        intrinsics = []
        extrinsics = []

        cam_data = sample_info.get("cams", sample_info.get("cameras", {}))

        for cam_name in CAMERA_NAMES:
            cam_info = cam_data.get(cam_name, {})

            # Load image
            img_path = cam_info.get("data_path", cam_info.get("filename", ""))
            if not os.path.isabs(img_path):
                img_path = os.path.join(self.data_root, img_path)

            if Image is not None and os.path.exists(img_path):
                img = np.array(Image.open(img_path).convert("RGB"))
            else:
                # Fallback: create placeholder if file doesn't exist (for testing)
                img = np.zeros((900, 1600, 3), dtype=np.uint8)

            # Apply image transforms
            img_tensor, scale_h, scale_w, pad_shape = self.img_transform(img)
            images.append(img_tensor)

            # Load intrinsics (3x3)
            intr = cam_info.get("intrinsics", cam_info.get("cam_intrinsic", None))
            if intr is not None:
                intr = np.array(intr, dtype=np.float32).reshape(3, 3)
                # Adjust intrinsics for resize
                intr[0, :] *= scale_w
                intr[1, :] *= scale_h
            else:
                intr = np.eye(3, dtype=np.float32)
            intrinsics.append(torch.from_numpy(intr))

            # Load extrinsics (4x4 camera-to-ego or sensor-to-lidar)
            extr = cam_info.get("extrinsics", None)
            if extr is None:
                # Try alternative keys: sensor2ego combined from rotation + translation
                sensor2ego_rot = cam_info.get("sensor2ego_rotation", None)
                sensor2ego_trans = cam_info.get("sensor2ego_translation", None)
                if sensor2ego_rot is not None and sensor2ego_trans is not None:
                    extr = self._compose_extrinsics(
                        np.array(sensor2ego_rot, dtype=np.float32),
                        np.array(sensor2ego_trans, dtype=np.float32),
                    )
                else:
                    extr = np.eye(4, dtype=np.float32)
            else:
                extr = np.array(extr, dtype=np.float32).reshape(4, 4)
            extrinsics.append(torch.from_numpy(extr))

        # Stack images: [6, 3, H, W]
        images_tensor = torch.stack(images, dim=0)
        intrinsics_tensor = torch.stack(intrinsics, dim=0)  # [6, 3, 3]
        extrinsics_tensor = torch.stack(extrinsics, dim=0)  # [6, 4, 4]

        # ---- Load map annotations ----
        map_anns = sample_info.get(
            "map_annotations", sample_info.get("vectors", sample_info.get("gts", []))
        )

        gt_labels = []
        gt_points = []

        for ann in map_anns:
            # Get class label
            cls_name = ann.get("class", ann.get("type", ann.get("cls_name", "")))
            if cls_name not in self.class_to_idx:
                continue

            label = self.class_to_idx[cls_name]

            # Get polyline points
            pts = ann.get("points", ann.get("pts", ann.get("vertices", [])))
            pts = np.array(pts, dtype=np.float32)

            if len(pts) < 2:
                continue

            # Ensure 2D
            if pts.ndim == 1:
                pts = pts.reshape(-1, 2)
            elif pts.shape[1] > 2:
                pts = pts[:, :2]  # Take only x, y

            # Resample to fixed number of points
            pts_resampled = self._resample_polyline(pts, self.num_points_per_instance)

            # Normalize to [0, 1]
            pts_normalized = self._normalize_points(pts_resampled)

            gt_labels.append(label)
            gt_points.append(pts_normalized)

        # Convert to tensors
        if len(gt_labels) > 0:
            gt_labels_tensor = torch.tensor(gt_labels, dtype=torch.long)
            gt_points_tensor = torch.from_numpy(np.stack(gt_points, axis=0))  # [N, num_pts, 2]
        else:
            gt_labels_tensor = torch.zeros(0, dtype=torch.long)
            gt_points_tensor = torch.zeros(
                (0, self.num_points_per_instance, 2), dtype=torch.float32
            )

        result = {
            "images": images_tensor,
            "intrinsics": intrinsics_tensor,
            "extrinsics": extrinsics_tensor,
            "gt_labels": gt_labels_tensor,
            "gt_points": gt_points_tensor,
        }

        # Include sample token if available
        token = sample_info.get("token", sample_info.get("sample_token", None))
        if token is not None:
            result["sample_token"] = token

        return result

    @staticmethod
    def _compose_extrinsics(
        rotation: np.ndarray, translation: np.ndarray
    ) -> np.ndarray:
        """
        Compose a 4x4 transformation matrix from a rotation (quaternion or 3x3)
        and translation vector.

        Args:
            rotation: Either a quaternion [w, x, y, z] or a 3x3 rotation matrix.
            translation: Translation vector [x, y, z].

        Returns:
            4x4 transformation matrix.
        """
        T = np.eye(4, dtype=np.float32)

        if rotation.shape == (3, 3):
            T[:3, :3] = rotation
        elif rotation.shape == (4,):
            # Quaternion [w, x, y, z] to rotation matrix
            w, x, y, z = rotation
            T[:3, :3] = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
            ], dtype=np.float32)
        elif rotation.size == 9:
            T[:3, :3] = rotation.reshape(3, 3)
        else:
            raise ValueError(f"Unsupported rotation shape: {rotation.shape}")

        translation = translation.flatten()
        T[:3, 3] = translation[:3]

        return T

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function that pads variable-length GT annotations
        to the maximum number of instances in the batch.

        Args:
            batch: List of sample dicts from __getitem__.

        Returns:
            Batched dict with:
                - images: [B, 6, 3, H, W]
                - intrinsics: [B, 6, 3, 3]
                - extrinsics: [B, 6, 4, 4]
                - gt_labels: [B, max_instances] (padded with -1)
                - gt_points: [B, max_instances, num_points, 2] (padded with 0)
                - gt_masks: [B, max_instances] bool mask (True = valid instance)
        """
        batch_size = len(batch)

        # Stack fixed-size tensors
        images = torch.stack([s["images"] for s in batch], dim=0)
        intrinsics = torch.stack([s["intrinsics"] for s in batch], dim=0)
        extrinsics = torch.stack([s["extrinsics"] for s in batch], dim=0)

        # Find max number of instances in this batch
        num_instances = [s["gt_labels"].shape[0] for s in batch]
        max_instances = max(num_instances) if num_instances else 0

        if max_instances == 0:
            # No ground truth in any sample
            num_points = batch[0]["gt_points"].shape[1] if batch[0]["gt_points"].numel() > 0 else 20
            gt_labels = torch.full((batch_size, 1), -1, dtype=torch.long)
            gt_points = torch.zeros((batch_size, 1, num_points, 2), dtype=torch.float32)
            gt_masks = torch.zeros((batch_size, 1), dtype=torch.bool)
        else:
            num_points = batch[0]["gt_points"].shape[1]
            gt_labels = torch.full((batch_size, max_instances), -1, dtype=torch.long)
            gt_points = torch.zeros(
                (batch_size, max_instances, num_points, 2), dtype=torch.float32
            )
            gt_masks = torch.zeros((batch_size, max_instances), dtype=torch.bool)

            for i, sample in enumerate(batch):
                n = num_instances[i]
                if n > 0:
                    gt_labels[i, :n] = sample["gt_labels"]
                    gt_points[i, :n] = sample["gt_points"]
                    gt_masks[i, :n] = True

        collated = {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "gt_labels": gt_labels,
            "gt_points": gt_points,
            "gt_masks": gt_masks,
        }

        # Collect sample tokens if present
        if "sample_token" in batch[0]:
            collated["sample_tokens"] = [s.get("sample_token", "") for s in batch]

        return collated
