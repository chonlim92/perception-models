"""Data augmentation transforms for 3D perception in autonomous driving.

Provides point cloud, image, and BEV augmentations following conventions from
BEVFormer/BEVDet. All transforms operate on sample dictionaries and properly
update all affected fields (points, boxes, images, intrinsics, extrinsics).
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class BaseTransform:
    """Base class for all transforms.

    All transforms operate on a sample dict and return the modified dict.
    Transforms should handle the case where expected keys are missing gracefully.
    """

    def __call__(self, data: Dict) -> Dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms: List of transform instances to apply in order.
    """

    def __init__(self, transforms: List[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(self, data: Dict) -> Dict:
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}(["]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append("])")
        return "\n".join(lines)


# ===========================================================================
# 3D Point Cloud Augmentations
# ===========================================================================


class RandomFlip3D(BaseTransform):
    """Randomly flip point cloud and boxes along X or Y axis.

    Args:
        flip_x_prob: Probability of flipping along X-axis. Default 0.5.
        flip_y_prob: Probability of flipping along Y-axis. Default 0.5.
    """

    def __init__(self, flip_x_prob: float = 0.5, flip_y_prob: float = 0.5) -> None:
        self.flip_x_prob = flip_x_prob
        self.flip_y_prob = flip_y_prob

    def __call__(self, data: Dict) -> Dict:
        flip_x = np.random.rand() < self.flip_x_prob
        flip_y = np.random.rand() < self.flip_y_prob

        if "points" in data:
            points = data["points"]
            if flip_x:
                points[:, 0] = -points[:, 0]
            if flip_y:
                points[:, 1] = -points[:, 1]
            data["points"] = points

        if "gt_boxes_3d" in data:
            # Boxes format: (N, 7+) -> [x, y, z, dx, dy, dz, yaw, ...]
            boxes = data["gt_boxes_3d"]
            if boxes.shape[0] > 0:
                if flip_x:
                    boxes[:, 0] = -boxes[:, 0]
                    boxes[:, 6] = -(boxes[:, 6])  # negate yaw
                if flip_y:
                    boxes[:, 1] = -boxes[:, 1]
                    boxes[:, 6] = np.pi - boxes[:, 6]  # mirror yaw about X-axis
            data["gt_boxes_3d"] = boxes

        # Store flip state for downstream use
        data.setdefault("flip_x", False)
        data.setdefault("flip_y", False)
        if flip_x:
            data["flip_x"] = not data["flip_x"]
        if flip_y:
            data["flip_y"] = not data["flip_y"]

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"flip_x_prob={self.flip_x_prob}, flip_y_prob={self.flip_y_prob})"
        )


class RandomRotate3D(BaseTransform):
    """Randomly rotate point cloud and boxes around Z-axis.

    Args:
        rotation_range: (min_angle, max_angle) in radians. Default (-pi/4, pi/4).
        prob: Probability of applying the rotation. Default 1.0.
    """

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
        prob: float = 1.0,
    ) -> None:
        self.rotation_range = rotation_range
        self.prob = prob

    def _rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Create 3x3 rotation matrix about Z-axis."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ])

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        angle = np.random.uniform(*self.rotation_range)
        rot_mat = self._rotation_matrix_z(angle)

        if "points" in data:
            points = data["points"]
            points[:, :3] = (rot_mat @ points[:, :3].T).T
            data["points"] = points

        if "gt_boxes_3d" in data:
            boxes = data["gt_boxes_3d"]
            if boxes.shape[0] > 0:
                # Rotate center positions
                boxes[:, :3] = (rot_mat @ boxes[:, :3].T).T
                # Adjust heading
                boxes[:, 6] += angle
            data["gt_boxes_3d"] = boxes

        data["rotation_angle"] = angle
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"rotation_range={self.rotation_range}, prob={self.prob})"
        )


class RandomScale3D(BaseTransform):
    """Randomly scale point cloud and boxes.

    Args:
        scale_range: (min_scale, max_scale). Default (0.95, 1.05).
        uniform: If True, apply same scale to all axes. If False, sample per axis.
        prob: Probability of applying. Default 1.0.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        uniform: bool = True,
        prob: float = 1.0,
    ) -> None:
        self.scale_range = scale_range
        self.uniform = uniform
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        if self.uniform:
            scale = np.random.uniform(*self.scale_range)
            scale_vec = np.array([scale, scale, scale])
        else:
            scale_vec = np.random.uniform(
                self.scale_range[0], self.scale_range[1], size=3
            )

        if "points" in data:
            points = data["points"]
            points[:, :3] *= scale_vec
            data["points"] = points

        if "gt_boxes_3d" in data:
            boxes = data["gt_boxes_3d"]
            if boxes.shape[0] > 0:
                # Scale center position
                boxes[:, :3] *= scale_vec
                # Scale dimensions (dx, dy, dz)
                boxes[:, 3:6] *= scale_vec
            data["gt_boxes_3d"] = boxes

        data["scale_factor_3d"] = scale_vec
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scale_range={self.scale_range}, uniform={self.uniform}, "
            f"prob={self.prob})"
        )


class RandomTranslate3D(BaseTransform):
    """Randomly translate point cloud and boxes.

    Args:
        translation_std: Standard deviation of Gaussian noise per axis (x, y, z).
            Default (0.2, 0.2, 0.2).
        prob: Probability of applying. Default 1.0.
    """

    def __init__(
        self,
        translation_std: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        prob: float = 1.0,
    ) -> None:
        self.translation_std = np.array(translation_std)
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        trans = np.random.normal(0, self.translation_std)

        if "points" in data:
            data["points"][:, :3] += trans

        if "gt_boxes_3d" in data:
            boxes = data["gt_boxes_3d"]
            if boxes.shape[0] > 0:
                boxes[:, :3] += trans
            data["gt_boxes_3d"] = boxes

        data["translation_3d"] = trans
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"translation_std={tuple(self.translation_std)}, prob={self.prob})"
        )


class RandomDropPoints(BaseTransform):
    """Randomly remove a fraction of points from the point cloud.

    Args:
        drop_ratio_range: (min_ratio, max_ratio) of points to drop.
            Default (0.0, 0.1).
        prob: Probability of applying. Default 0.5.
    """

    def __init__(
        self,
        drop_ratio_range: Tuple[float, float] = (0.0, 0.1),
        prob: float = 0.5,
    ) -> None:
        self.drop_ratio_range = drop_ratio_range
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        if "points" not in data:
            return data

        points = data["points"]
        N = points.shape[0]
        drop_ratio = np.random.uniform(*self.drop_ratio_range)
        keep_num = max(1, int(N * (1 - drop_ratio)))

        indices = np.random.choice(N, keep_num, replace=False)
        indices.sort()
        data["points"] = points[indices]

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"drop_ratio_range={self.drop_ratio_range}, prob={self.prob})"
        )


class GlobalRotScaleTrans(BaseTransform):
    """Combined global rotation, scaling, and translation augmentation.

    This is the standard combined augmentation used in most LiDAR-based
    3D detectors (PointPillars, CenterPoint, BEVDet, etc.).

    Args:
        rotation_range: (min_angle, max_angle) in radians. Default (-pi/4, pi/4).
        scale_range: (min_scale, max_scale). Default (0.95, 1.05).
        translation_std: Per-axis translation std. Default (0.0, 0.0, 0.0).
        prob: Probability of applying. Default 1.0.
    """

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translation_std: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        prob: float = 1.0,
    ) -> None:
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_std = np.array(translation_std)
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        # Sample augmentation parameters
        angle = np.random.uniform(*self.rotation_range)
        scale = np.random.uniform(*self.scale_range)
        trans = np.random.normal(0, self.translation_std)

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot_mat = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ])

        if "points" in data:
            points = data["points"]
            points[:, :3] = (rot_mat @ points[:, :3].T).T * scale + trans
            data["points"] = points

        if "gt_boxes_3d" in data:
            boxes = data["gt_boxes_3d"]
            if boxes.shape[0] > 0:
                # Rotate + scale center
                boxes[:, :3] = (rot_mat @ boxes[:, :3].T).T * scale + trans
                # Scale dimensions
                boxes[:, 3:6] *= scale
                # Rotate heading
                boxes[:, 6] += angle
            data["gt_boxes_3d"] = boxes

        data["global_rot_angle"] = angle
        data["global_scale"] = scale
        data["global_translation"] = trans
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"rotation_range={self.rotation_range}, "
            f"scale_range={self.scale_range}, "
            f"translation_std={tuple(self.translation_std)}, "
            f"prob={self.prob})"
        )


# ===========================================================================
# Image Augmentations
# ===========================================================================


class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortions: brightness, contrast, saturation, hue.

    Follows the augmentation pipeline from SSD/DETR adapted for multi-camera
    autonomous driving setups.

    Args:
        brightness_delta: Maximum brightness change. Default 32.
        contrast_range: (lower, upper) contrast factor range. Default (0.5, 1.5).
        saturation_range: (lower, upper) saturation factor range. Default (0.5, 1.5).
        hue_delta: Maximum hue shift in degrees. Default 18.
        prob: Probability of applying. Default 0.5.
    """

    def __init__(
        self,
        brightness_delta: float = 32.0,
        contrast_range: Tuple[float, float] = (0.5, 1.5),
        saturation_range: Tuple[float, float] = (0.5, 1.5),
        hue_delta: float = 18.0,
        prob: float = 0.5,
    ) -> None:
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta
        self.prob = prob

    def _convert_to_hsv(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR/RGB uint8 image to HSV float."""
        img = img.astype(np.float32) / 255.0
        max_c = img.max(axis=-1)
        min_c = img.min(axis=-1)
        diff = max_c - min_c

        h = np.zeros_like(max_c)
        s = np.zeros_like(max_c)
        v = max_c

        # Saturation
        mask = max_c > 0
        s[mask] = diff[mask] / max_c[mask]

        # Hue
        mask = diff > 0
        idx = (img[..., 0] == max_c) & mask
        h[idx] = 60.0 * (img[idx, 1] - img[idx, 2]) / diff[idx]
        idx = (img[..., 1] == max_c) & mask
        h[idx] = 120.0 + 60.0 * (img[idx, 2] - img[idx, 0]) / diff[idx]
        idx = (img[..., 2] == max_c) & mask
        h[idx] = 240.0 + 60.0 * (img[idx, 0] - img[idx, 1]) / diff[idx]
        h[h < 0] += 360.0

        return np.stack([h, s, v], axis=-1)

    def _convert_from_hsv(self, hsv: np.ndarray) -> np.ndarray:
        """Convert HSV float image back to uint8 RGB."""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = h % 360.0

        c = v * s
        h_prime = h / 60.0
        x = c * (1 - np.abs(h_prime % 2 - 1))
        m = v - c

        rgb = np.zeros_like(hsv)
        for i in range(6):
            mask = (h_prime >= i) & (h_prime < i + 1)
            if i == 0:
                rgb[mask] = np.stack([c[mask], x[mask], np.zeros_like(c[mask])], axis=-1)
            elif i == 1:
                rgb[mask] = np.stack([x[mask], c[mask], np.zeros_like(c[mask])], axis=-1)
            elif i == 2:
                rgb[mask] = np.stack([np.zeros_like(c[mask]), c[mask], x[mask]], axis=-1)
            elif i == 3:
                rgb[mask] = np.stack([np.zeros_like(c[mask]), x[mask], c[mask]], axis=-1)
            elif i == 4:
                rgb[mask] = np.stack([x[mask], np.zeros_like(c[mask]), c[mask]], axis=-1)
            else:
                rgb[mask] = np.stack([c[mask], np.zeros_like(c[mask]), x[mask]], axis=-1)

        rgb += m[..., None]
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def _distort_single(self, img: np.ndarray) -> np.ndarray:
        """Apply distortions to a single image (H, W, 3) uint8."""
        img = img.astype(np.float32)

        # Brightness
        if np.random.rand() < 0.5:
            delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
            img += delta

        # Contrast (applied before or after color distortions randomly)
        contrast_first = np.random.rand() < 0.5
        if contrast_first and np.random.rand() < 0.5:
            alpha = np.random.uniform(*self.contrast_range)
            img *= alpha

        # Saturation and Hue in HSV space
        img_clipped = np.clip(img, 0, 255).astype(np.uint8)
        hsv = self._convert_to_hsv(img_clipped)

        if np.random.rand() < 0.5:
            hsv[..., 1] *= np.random.uniform(*self.saturation_range)
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)

        if np.random.rand() < 0.5:
            hsv[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)

        img = self._convert_from_hsv(hsv).astype(np.float32)

        # Contrast (if not applied earlier)
        if not contrast_first and np.random.rand() < 0.5:
            alpha = np.random.uniform(*self.contrast_range)
            img *= alpha

        return np.clip(img, 0, 255).astype(np.uint8)

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        if "images" in data:
            images = data["images"]
            # Handle list of images (multi-camera) or single image
            if isinstance(images, list):
                data["images"] = [self._distort_single(img) for img in images]
            elif images.ndim == 4:
                # (num_cams, H, W, 3)
                data["images"] = np.stack(
                    [self._distort_single(images[i]) for i in range(images.shape[0])]
                )
            else:
                data["images"] = self._distort_single(images)

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brightness_delta={self.brightness_delta}, "
            f"contrast_range={self.contrast_range}, "
            f"saturation_range={self.saturation_range}, "
            f"hue_delta={self.hue_delta}, prob={self.prob})"
        )


class ImageNormalize(BaseTransform):
    """Normalize images with mean and standard deviation.

    Converts uint8 images to float32 and normalizes channel-wise.

    Args:
        mean: Per-channel mean values. Default ImageNet mean.
        std: Per-channel standard deviation. Default ImageNet std.
        to_rgb: If True, convert BGR to RGB before normalizing. Default False.
    """

    def __init__(
        self,
        mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
        std: Tuple[float, ...] = (58.395, 57.12, 57.375),
        to_rgb: bool = False,
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize a single image."""
        img = img.astype(np.float32)
        if self.to_rgb and img.ndim == 3 and img.shape[2] == 3:
            img = img[..., ::-1]
        img = (img - self.mean) / self.std
        return img

    def __call__(self, data: Dict) -> Dict:
        if "images" in data:
            images = data["images"]
            if isinstance(images, list):
                data["images"] = [self._normalize(img) for img in images]
            elif images.ndim == 4:
                data["images"] = np.stack(
                    [self._normalize(images[i]) for i in range(images.shape[0])]
                )
            else:
                data["images"] = self._normalize(images)

        data["img_norm_cfg"] = {"mean": self.mean, "std": self.std, "to_rgb": self.to_rgb}
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mean={tuple(self.mean)}, std={tuple(self.std)}, to_rgb={self.to_rgb})"
        )


class ImageResize(BaseTransform):
    """Resize images with aspect ratio handling.

    Supports multi-camera setups. Updates intrinsic matrices accordingly.

    Args:
        target_size: (H, W) target size.
        keep_ratio: If True, resize with aspect ratio preserved and pad.
            Default True.
        prob: Probability of applying. Default 1.0.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (900, 1600),
        keep_ratio: bool = True,
        prob: float = 1.0,
    ) -> None:
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.prob = prob

    def _resize_single(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Resize a single image and return scale factors."""
        h, w = img.shape[:2]
        target_h, target_w = self.target_size

        if self.keep_ratio:
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
            scale_y, scale_x = new_h / h, new_w / w
        else:
            new_h, new_w = target_h, target_w
            scale_y, scale_x = new_h / h, new_w / w

        # Simple bilinear-like resize using numpy (nearest for efficiency)
        row_indices = (np.arange(new_h) / scale_y).astype(int)
        col_indices = (np.arange(new_w) / scale_x).astype(int)
        row_indices = np.clip(row_indices, 0, h - 1)
        col_indices = np.clip(col_indices, 0, w - 1)

        resized = img[row_indices][:, col_indices]

        if self.keep_ratio:
            # Pad to target size
            padded = np.zeros((target_h, target_w) + img.shape[2:], dtype=img.dtype)
            padded[:new_h, :new_w] = resized
            return padded, scale_x, scale_y
        else:
            return resized, scale_x, scale_y

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        if "images" not in data:
            return data

        images = data["images"]
        is_list = isinstance(images, list)
        if not is_list and images.ndim == 3:
            images = [images]
            single = True
        elif not is_list and images.ndim == 4:
            images = [images[i] for i in range(images.shape[0])]
            single = False
        else:
            single = False

        resized_images = []
        scale_factors = []

        for img in images:
            resized, sx, sy = self._resize_single(img)
            resized_images.append(resized)
            scale_factors.append((sx, sy))

        # Update intrinsics if available
        if "intrinsics" in data:
            intrinsics = data["intrinsics"]
            if isinstance(intrinsics, np.ndarray) and intrinsics.ndim == 3:
                # (num_cams, 3, 3) or (num_cams, 4, 4)
                for i, (sx, sy) in enumerate(scale_factors):
                    if i < intrinsics.shape[0]:
                        intrinsics[i, 0, :] *= sx  # fx, skew, cx
                        intrinsics[i, 1, :] *= sy  # fy, cy
                data["intrinsics"] = intrinsics
            elif isinstance(intrinsics, list):
                for i, (sx, sy) in enumerate(scale_factors):
                    if i < len(intrinsics):
                        intrinsics[i][0, :] *= sx
                        intrinsics[i][1, :] *= sy
                data["intrinsics"] = intrinsics

        if single:
            data["images"] = resized_images[0]
        elif is_list:
            data["images"] = resized_images
        else:
            data["images"] = np.stack(resized_images)

        data["img_scale_factors"] = scale_factors
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target_size={self.target_size}, keep_ratio={self.keep_ratio}, "
            f"prob={self.prob})"
        )


class RandomCropImage(BaseTransform):
    """Random crop of image(s) with corresponding intrinsic matrix update.

    Crops a fixed-size patch from the image and adjusts the principal point
    in the intrinsic matrix.

    Args:
        crop_size: (H, W) size of the crop.
        prob: Probability of applying. Default 0.5.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int] = (320, 800),
        prob: float = 0.5,
    ) -> None:
        self.crop_size = crop_size
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        if "images" not in data:
            return data

        images = data["images"]
        is_list = isinstance(images, list)
        if not is_list and images.ndim == 3:
            images = [images]
            single = True
        elif not is_list and images.ndim == 4:
            images = [images[i] for i in range(images.shape[0])]
            single = False
        else:
            single = False

        crop_h, crop_w = self.crop_size
        cropped_images = []
        crop_offsets = []

        for img in images:
            h, w = img.shape[:2]
            # Ensure crop fits
            ch = min(crop_h, h)
            cw = min(crop_w, w)

            top = np.random.randint(0, max(h - ch, 0) + 1)
            left = np.random.randint(0, max(w - cw, 0) + 1)

            cropped = img[top : top + ch, left : left + cw]
            cropped_images.append(cropped)
            crop_offsets.append((left, top))

        # Update intrinsics: shift principal point by crop offset
        if "intrinsics" in data:
            intrinsics = data["intrinsics"]
            if isinstance(intrinsics, np.ndarray) and intrinsics.ndim == 3:
                for i, (dx, dy) in enumerate(crop_offsets):
                    if i < intrinsics.shape[0]:
                        intrinsics[i, 0, 2] -= dx  # cx
                        intrinsics[i, 1, 2] -= dy  # cy
                data["intrinsics"] = intrinsics
            elif isinstance(intrinsics, list):
                for i, (dx, dy) in enumerate(crop_offsets):
                    if i < len(intrinsics):
                        intrinsics[i][0, 2] -= dx
                        intrinsics[i][1, 2] -= dy
                data["intrinsics"] = intrinsics

        if single:
            data["images"] = cropped_images[0]
        elif is_list:
            data["images"] = cropped_images
        else:
            data["images"] = np.stack(cropped_images) if all(
                c.shape == cropped_images[0].shape for c in cropped_images
            ) else cropped_images

        data["crop_offsets"] = crop_offsets
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"crop_size={self.crop_size}, prob={self.prob})"
        )


class GridMask(BaseTransform):
    """Grid-based masking augmentation for images.

    Masks out regular grid patterns from images to improve robustness.
    Originally from GridMask paper (Chen et al., 2020).

    Args:
        use_h: Whether to mask along height. Default True.
        use_w: Whether to mask along width. Default True.
        ratio: Ratio of masked region within each grid cell. Default 0.5.
        grid_size_range: (min, max) grid spacing in pixels. Default (64, 128).
        prob: Probability of applying. Default 0.5.
    """

    def __init__(
        self,
        use_h: bool = True,
        use_w: bool = True,
        ratio: float = 0.5,
        grid_size_range: Tuple[int, int] = (64, 128),
        prob: float = 0.5,
    ) -> None:
        self.use_h = use_h
        self.use_w = use_w
        self.ratio = ratio
        self.grid_size_range = grid_size_range
        self.prob = prob

    def _generate_mask(self, h: int, w: int) -> np.ndarray:
        """Generate a binary grid mask (1 = keep, 0 = mask)."""
        d = np.random.randint(self.grid_size_range[0], self.grid_size_range[1])
        mask_len = int(d * self.ratio + 0.5)

        mask = np.ones((h, w), dtype=np.float32)

        if self.use_h:
            offset_h = np.random.randint(0, d)
            for start in range(offset_h, h, d):
                end = min(start + mask_len, h)
                mask[start:end, :] = 0

        if self.use_w:
            offset_w = np.random.randint(0, d)
            mask_w = np.ones((h, w), dtype=np.float32)
            for start in range(offset_w, w, d):
                end = min(start + mask_len, w)
                mask_w[:, start:end] = 0
            mask *= mask_w

        return mask

    def _apply_single(self, img: np.ndarray) -> np.ndarray:
        """Apply grid mask to a single image."""
        h, w = img.shape[:2]
        mask = self._generate_mask(h, w)
        if img.ndim == 3:
            mask = mask[..., None]
        return (img * mask).astype(img.dtype)

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        if "images" not in data:
            return data

        images = data["images"]
        if isinstance(images, list):
            data["images"] = [self._apply_single(img) for img in images]
        elif images.ndim == 4:
            data["images"] = np.stack(
                [self._apply_single(images[i]) for i in range(images.shape[0])]
            )
        else:
            data["images"] = self._apply_single(images)

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ratio={self.ratio}, grid_size_range={self.grid_size_range}, "
            f"prob={self.prob})"
        )


# ===========================================================================
# BEV (Bird's Eye View) Augmentations
# ===========================================================================


class BEVRandomFlip(BaseTransform):
    """Random flip in BEV space, consistently updating points and images.

    When flipping in BEV, both the point cloud and camera extrinsics must be
    updated so that image features projected to BEV remain consistent.

    Args:
        flip_x_prob: Probability of flipping along X (lateral). Default 0.5.
        flip_y_prob: Probability of flipping along Y (longitudinal). Default 0.0.
    """

    def __init__(
        self,
        flip_x_prob: float = 0.5,
        flip_y_prob: float = 0.0,
    ) -> None:
        self.flip_x_prob = flip_x_prob
        self.flip_y_prob = flip_y_prob

    def __call__(self, data: Dict) -> Dict:
        flip_x = np.random.rand() < self.flip_x_prob
        flip_y = np.random.rand() < self.flip_y_prob

        if not flip_x and not flip_y:
            return data

        # Flip points
        if "points" in data:
            points = data["points"]
            if flip_x:
                points[:, 0] = -points[:, 0]
            if flip_y:
                points[:, 1] = -points[:, 1]
            data["points"] = points

        # Flip boxes
        if "gt_boxes_3d" in data:
            boxes = data["gt_boxes_3d"]
            if boxes.shape[0] > 0:
                if flip_x:
                    boxes[:, 0] = -boxes[:, 0]
                    boxes[:, 6] = -(boxes[:, 6])
                if flip_y:
                    boxes[:, 1] = -boxes[:, 1]
                    boxes[:, 6] = np.pi - boxes[:, 6]
            data["gt_boxes_3d"] = boxes

        # Update camera extrinsics (lidar2camera or lidar2img)
        # Flipping the LiDAR frame is equivalent to applying a reflection matrix
        if "extrinsics" in data:
            extrinsics = data["extrinsics"]
            # Reflection matrix for the flip
            flip_mat = np.eye(4)
            if flip_x:
                flip_mat[0, 0] = -1
            if flip_y:
                flip_mat[1, 1] = -1

            if isinstance(extrinsics, np.ndarray) and extrinsics.ndim == 3:
                # (num_cams, 4, 4)
                for i in range(extrinsics.shape[0]):
                    extrinsics[i] = extrinsics[i] @ flip_mat
                data["extrinsics"] = extrinsics
            elif isinstance(extrinsics, list):
                for i in range(len(extrinsics)):
                    extrinsics[i] = extrinsics[i] @ flip_mat
                data["extrinsics"] = extrinsics

        # If images are indexed by camera, we may need to swap camera order
        # (e.g., left camera becomes right camera when flipping X)
        # This is dataset-specific; store flip state for downstream handling.
        data["bev_flip_x"] = flip_x
        data["bev_flip_y"] = flip_y

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"flip_x_prob={self.flip_x_prob}, flip_y_prob={self.flip_y_prob})"
        )


class BEVRandomRotate(BaseTransform):
    """Random rotation in BEV space, updating camera extrinsics.

    Rotates the entire scene around the Z-axis in ego/LiDAR frame.
    Camera extrinsics are updated so that image-to-BEV projections remain
    consistent.

    Args:
        rotation_range: (min_angle, max_angle) in radians. Default (-pi/4, pi/4).
        prob: Probability of applying. Default 1.0.
    """

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
        prob: float = 1.0,
    ) -> None:
        self.rotation_range = rotation_range
        self.prob = prob

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        angle = np.random.uniform(*self.rotation_range)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 4x4 rotation about Z
        rot_4x4 = np.eye(4)
        rot_4x4[0, 0] = cos_a
        rot_4x4[0, 1] = -sin_a
        rot_4x4[1, 0] = sin_a
        rot_4x4[1, 1] = cos_a

        rot_3x3 = rot_4x4[:3, :3]

        # Rotate points
        if "points" in data:
            points = data["points"]
            points[:, :3] = (rot_3x3 @ points[:, :3].T).T
            data["points"] = points

        # Rotate boxes
        if "gt_boxes_3d" in data:
            boxes = data["gt_boxes_3d"]
            if boxes.shape[0] > 0:
                boxes[:, :3] = (rot_3x3 @ boxes[:, :3].T).T
                boxes[:, 6] += angle
            data["gt_boxes_3d"] = boxes

        # Update extrinsics: E_new = E_old @ R_inv (since points rotated by R)
        # If E maps from LiDAR to camera: p_cam = E @ p_lidar
        # After rotation: p_lidar_new = R @ p_lidar_old
        # So E_new = E_old @ R_inv to get same p_cam from new p_lidar_new
        rot_inv_4x4 = np.eye(4)
        rot_inv_4x4[0, 0] = cos_a
        rot_inv_4x4[0, 1] = sin_a
        rot_inv_4x4[1, 0] = -sin_a
        rot_inv_4x4[1, 1] = cos_a

        if "extrinsics" in data:
            extrinsics = data["extrinsics"]
            if isinstance(extrinsics, np.ndarray) and extrinsics.ndim == 3:
                for i in range(extrinsics.shape[0]):
                    extrinsics[i] = extrinsics[i] @ rot_inv_4x4
                data["extrinsics"] = extrinsics
            elif isinstance(extrinsics, list):
                for i in range(len(extrinsics)):
                    extrinsics[i] = extrinsics[i] @ rot_inv_4x4
                data["extrinsics"] = extrinsics

        data["bev_rotation_angle"] = angle
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"rotation_range={self.rotation_range}, prob={self.prob})"
        )


class BEVGridMask(BaseTransform):
    """Grid-based masking in BEV feature space.

    Applies a grid mask pattern to BEV features or to projected image features
    in BEV coordinates. Useful for regularization in BEV-based models.

    Args:
        ratio: Ratio of masked area within each cell. Default 0.5.
        grid_size_range: (min, max) grid spacing in BEV pixels. Default (30, 60).
        prob: Probability of applying. Default 0.5.
    """

    def __init__(
        self,
        ratio: float = 0.5,
        grid_size_range: Tuple[int, int] = (30, 60),
        prob: float = 0.5,
    ) -> None:
        self.ratio = ratio
        self.grid_size_range = grid_size_range
        self.prob = prob

    def _generate_bev_mask(self, h: int, w: int) -> np.ndarray:
        """Generate a grid mask for BEV feature map."""
        d = np.random.randint(self.grid_size_range[0], self.grid_size_range[1])
        mask_len = int(d * self.ratio + 0.5)

        mask = np.ones((h, w), dtype=np.float32)

        offset_h = np.random.randint(0, d)
        for start in range(offset_h, h, d):
            end = min(start + mask_len, h)
            mask[start:end, :] = 0

        offset_w = np.random.randint(0, d)
        mask_w = np.ones((h, w), dtype=np.float32)
        for start in range(offset_w, w, d):
            end = min(start + mask_len, w)
            mask_w[:, start:end] = 0
        mask *= mask_w

        return mask

    def __call__(self, data: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return data

        # Apply to BEV features if present
        if "bev_features" in data:
            bev = data["bev_features"]
            if bev.ndim == 3:
                # (C, H, W) or (H, W, C)
                h, w = bev.shape[-2], bev.shape[-1]
                mask = self._generate_bev_mask(h, w)
                if bev.shape[0] != h:  # (C, H, W)
                    bev = bev * mask[None, :, :]
                else:  # (H, W, C)
                    bev = bev * mask[:, :, None]
                data["bev_features"] = bev
            elif bev.ndim == 2:
                h, w = bev.shape
                mask = self._generate_bev_mask(h, w)
                data["bev_features"] = bev * mask

        # Also apply to images used for BEV projection (common in training)
        if "images" in data and "bev_features" not in data:
            images = data["images"]
            if isinstance(images, list):
                for i in range(len(images)):
                    h, w = images[i].shape[:2]
                    mask = self._generate_bev_mask(h, w)
                    if images[i].ndim == 3:
                        images[i] = (images[i] * mask[:, :, None]).astype(images[i].dtype)
                    else:
                        images[i] = (images[i] * mask).astype(images[i].dtype)
                data["images"] = images
            elif isinstance(images, np.ndarray) and images.ndim == 4:
                for i in range(images.shape[0]):
                    h, w = images[i].shape[:2]
                    mask = self._generate_bev_mask(h, w)
                    images[i] = (images[i] * mask[:, :, None]).astype(images.dtype)
                data["images"] = images

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ratio={self.ratio}, grid_size_range={self.grid_size_range}, "
            f"prob={self.prob})"
        )
