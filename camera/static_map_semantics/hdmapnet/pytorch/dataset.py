"""
nuScenes HD Map dataset for HDMapNet.

Loads multi-camera images with intrinsics/extrinsics, renders BEV ground truth
maps (semantic masks, instance maps, direction maps) from nuScenes map annotations.
Supports data augmentation (random flip, color jitter).
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# Map layer names to class indices
MAP_CLASSES = {
    "road_divider": 0,
    "lane_divider": 0,
    "ped_crossing": 2,
    "road_segment": 1,
    "road_boundary": 1,
}

# Class names for reference
CLASS_NAMES = ["divider", "boundary", "crossing"]


class NuScenesHDMapDataset(Dataset):
    """nuScenes dataset for HDMapNet training.

    Expected data structure:
        dataroot/
            samples/
                CAM_FRONT/
                CAM_FRONT_LEFT/
                CAM_FRONT_RIGHT/
                CAM_BACK/
                CAM_BACK_LEFT/
                CAM_BACK_RIGHT/
            annotations/
                hdmap_annotations.json  (or generated from nuScenes devkit)

    The annotations file should contain per-sample entries with:
        - camera file paths
        - camera intrinsics and extrinsics
        - map polyline annotations in ego frame
    """

    CAMERA_NAMES = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]

    def __init__(
        self,
        dataroot,
        ann_file,
        image_size=(128, 352),
        xbound=(-30.0, 30.0, 0.3),
        ybound=(-15.0, 15.0, 0.3),
        num_classes=3,
        augment=True,
        thickness=5,
    ):
        """
        Args:
            dataroot: Root directory of the nuScenes dataset.
            ann_file: Path to annotation JSON file.
            image_size: Target (H, W) for resized images.
            xbound: (xmin, xmax, resolution) for BEV x-axis.
            ybound: (ymin, ymax, resolution) for BEV y-axis.
            num_classes: Number of semantic map classes.
            augment: Whether to apply data augmentation.
            thickness: Line thickness for rendering map elements on BEV.
        """
        super().__init__()
        self.dataroot = dataroot
        self.image_size = image_size
        self.xbound = xbound
        self.ybound = ybound
        self.num_classes = num_classes
        self.augment = augment
        self.thickness = thickness

        # BEV grid dimensions
        self.bev_h = int((ybound[1] - ybound[0]) / ybound[2])
        self.bev_w = int((xbound[1] - xbound[0]) / xbound[2])

        # Load annotations
        with open(ann_file, "r") as f:
            self.annotations = json.load(f)

        # Image normalization
        self.img_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Color jitter for augmentation
        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )

    def __len__(self):
        return len(self.annotations)

    def _load_image(self, filepath):
        """Load and resize an image.

        Args:
            filepath: Path to image file.

        Returns:
            PIL Image resized to self.image_size.
        """
        img = Image.open(filepath).convert("RGB")
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        return img

    def _adjust_intrinsics(self, intrinsics, original_size, target_size):
        """Adjust camera intrinsics after image resize.

        Args:
            intrinsics: 3x3 numpy intrinsic matrix.
            original_size: (orig_H, orig_W).
            target_size: (target_H, target_W).

        Returns:
            Adjusted 3x3 intrinsic matrix.
        """
        sx = target_size[1] / original_size[1]
        sy = target_size[0] / original_size[0]

        adjusted = intrinsics.copy()
        adjusted[0, 0] *= sx  # fx
        adjusted[0, 2] *= sx  # cx
        adjusted[1, 1] *= sy  # fy
        adjusted[1, 2] *= sy  # cy
        return adjusted

    def _world_to_bev_pixel(self, x, y):
        """Convert world coordinates (meters) to BEV pixel coordinates.

        Args:
            x: X coordinate in ego frame (meters).
            y: Y coordinate in ego frame (meters).

        Returns:
            (col, row) pixel coordinates in BEV grid.
        """
        col = (x - self.xbound[0]) / self.xbound[2]
        row = (y - self.ybound[0]) / self.ybound[2]
        return col, row

    def _render_map_gt(self, annotation):
        """Render BEV ground truth maps from annotation polylines.

        Args:
            annotation: Dict with 'map_elements' key containing polyline definitions.

        Returns:
            semantic_map: (num_classes, bev_h, bev_w) binary masks.
            instance_map: (bev_h, bev_w) integer instance IDs.
            direction_map: (2, bev_h, bev_w) direction vectors.
        """
        semantic_map = np.zeros((self.num_classes, self.bev_h, self.bev_w), dtype=np.float32)
        instance_map = np.zeros((self.bev_h, self.bev_w), dtype=np.int32)
        direction_map = np.zeros((2, self.bev_h, self.bev_w), dtype=np.float32)

        instance_counter = 0

        map_elements = annotation.get("map_elements", [])

        for element in map_elements:
            class_name = element.get("class", "")
            if class_name not in MAP_CLASSES:
                continue

            cls_id = MAP_CLASSES[class_name]
            points = np.array(element["points"])  # (N, 2) or (N, 3) in ego frame

            if len(points) < 2:
                continue

            # Take x, y coordinates (ignore z if present)
            xy = points[:, :2]

            # Convert to BEV pixel coordinates
            bev_points = []
            for pt in xy:
                col, row = self._world_to_bev_pixel(pt[0], pt[1])
                bev_points.append([col, row])
            bev_points = np.array(bev_points)

            # Draw polyline on semantic map and instance map
            instance_counter += 1
            self._draw_polyline(
                semantic_map[cls_id], instance_map, direction_map,
                bev_points, instance_counter
            )

        return semantic_map, instance_map, direction_map

    def _draw_polyline(self, semantic_layer, instance_map, direction_map, points, instance_id):
        """Draw a polyline onto semantic, instance, and direction maps.

        Uses Bresenham-style line drawing with configurable thickness.

        Args:
            semantic_layer: (bev_h, bev_w) array for the class.
            instance_map: (bev_h, bev_w) array for instance IDs.
            direction_map: (2, bev_h, bev_w) array for directions.
            points: (N, 2) array of (col, row) in BEV pixel coords.
            instance_id: Integer instance ID to assign.
        """
        H, W = semantic_layer.shape
        half_thick = self.thickness // 2

        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            # Direction for this segment
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx * dx + dy * dy)
            if length < 1e-6:
                continue
            dx /= length
            dy /= length

            # Rasterize line segment using linear interpolation
            num_steps = max(int(length * 2), 1)
            for t in np.linspace(0, 1, num_steps):
                cx = x0 + t * (x1 - x0)
                cy = y0 + t * (y1 - y0)

                # Draw with thickness
                for ddy in range(-half_thick, half_thick + 1):
                    for ddx in range(-half_thick, half_thick + 1):
                        px = int(round(cx + ddx))
                        py = int(round(cy + ddy))
                        if 0 <= px < W and 0 <= py < H:
                            semantic_layer[py, px] = 1.0
                            instance_map[py, px] = instance_id
                            direction_map[0, py, px] = dx
                            direction_map[1, py, px] = dy

    def _augment_flip(self, images, intrinsics, extrinsics, semantic_map, instance_map, direction_map):
        """Apply random horizontal flip augmentation.

        Flips all camera images, adjusts intrinsics/extrinsics, and flips BEV maps.

        Args:
            images: List of PIL Images.
            intrinsics: (N, 3, 3) numpy array.
            extrinsics: (N, 4, 4) numpy array.
            semantic_map: (C, H, W) numpy array.
            instance_map: (H, W) numpy array.
            direction_map: (2, H, W) numpy array.

        Returns:
            Flipped versions of all inputs.
        """
        if np.random.random() > 0.5:
            return images, intrinsics, extrinsics, semantic_map, instance_map, direction_map

        # Flip images
        flipped_images = [TF.hflip(img) for img in images]

        # Adjust intrinsics (flip cx)
        flipped_intrinsics = intrinsics.copy()
        for i in range(len(flipped_intrinsics)):
            flipped_intrinsics[i, 0, 2] = self.image_size[1] - flipped_intrinsics[i, 0, 2]

        # Flip BEV maps horizontally
        flipped_semantic = semantic_map[:, :, ::-1].copy()
        flipped_instance = instance_map[:, ::-1].copy()
        flipped_direction = direction_map[:, :, ::-1].copy()
        flipped_direction[0] = -flipped_direction[0]  # Flip x direction

        # Swap left/right cameras and adjust extrinsics
        # Camera order: FRONT, FRONT_RIGHT, BACK_RIGHT, BACK, BACK_LEFT, FRONT_LEFT
        # After flip: FRONT, FRONT_LEFT, BACK_LEFT, BACK, BACK_RIGHT, FRONT_RIGHT
        swap_indices = [0, 5, 4, 3, 2, 1]
        flipped_images = [flipped_images[i] for i in swap_indices]
        flipped_intrinsics = flipped_intrinsics[swap_indices]

        flipped_extrinsics = extrinsics[swap_indices].copy()
        # Negate x translation and rotation around y-axis
        for i in range(len(flipped_extrinsics)):
            flipped_extrinsics[i, 0, 3] = -flipped_extrinsics[i, 0, 3]
            flipped_extrinsics[i, 0, 1] = -flipped_extrinsics[i, 0, 1]
            flipped_extrinsics[i, 0, 2] = -flipped_extrinsics[i, 0, 2]
            flipped_extrinsics[i, 1, 0] = -flipped_extrinsics[i, 1, 0]
            flipped_extrinsics[i, 2, 0] = -flipped_extrinsics[i, 2, 0]

        return (
            flipped_images, flipped_intrinsics, flipped_extrinsics,
            flipped_semantic, flipped_instance, flipped_direction,
        )

    def __getitem__(self, index):
        """Get a single training sample.

        Args:
            index: Dataset index.

        Returns:
            Dict with keys:
                - 'images': (N_cams, 3, H, W) float tensor, normalized.
                - 'intrinsics': (N_cams, 3, 3) float tensor.
                - 'extrinsics': (N_cams, 4, 4) float tensor.
                - 'semantic_map': (num_classes, bev_h, bev_w) float tensor.
                - 'instance_map': (bev_h, bev_w) long tensor.
                - 'direction_map': (2, bev_h, bev_w) float tensor.
        """
        ann = self.annotations[index]

        # Load camera images and parameters
        images = []
        intrinsics_list = []
        extrinsics_list = []

        for cam_idx, cam_name in enumerate(self.CAMERA_NAMES):
            cam_info = ann["cameras"][cam_name]

            # Load image
            img_path = os.path.join(self.dataroot, cam_info["filepath"])
            img = self._load_image(img_path)

            # Get intrinsics and adjust for resize
            K = np.array(cam_info["intrinsics"], dtype=np.float32).reshape(3, 3)
            orig_size = cam_info.get("image_size", [900, 1600])  # (H, W)
            K = self._adjust_intrinsics(K, orig_size, self.image_size)

            # Get extrinsics (4x4 transformation matrix)
            E = np.array(cam_info["extrinsics"], dtype=np.float32).reshape(4, 4)

            images.append(img)
            intrinsics_list.append(K)
            extrinsics_list.append(E)

        intrinsics = np.stack(intrinsics_list, axis=0)  # (N, 3, 3)
        extrinsics = np.stack(extrinsics_list, axis=0)  # (N, 4, 4)

        # Render ground truth maps
        semantic_map, instance_map, direction_map = self._render_map_gt(ann)

        # Data augmentation
        if self.augment:
            images, intrinsics, extrinsics, semantic_map, instance_map, direction_map = (
                self._augment_flip(
                    images, intrinsics, extrinsics, semantic_map, instance_map, direction_map
                )
            )
            # Color jitter (applied independently to each camera)
            images = [self.color_jitter(img) for img in images]

        # Convert images to tensors and normalize
        img_tensors = []
        for img in images:
            img_tensor = TF.to_tensor(img)  # (3, H, W), [0, 1]
            img_tensor = self.img_normalize(img_tensor)
            img_tensors.append(img_tensor)

        images_tensor = torch.stack(img_tensors, dim=0)  # (N, 3, H, W)

        return {
            "images": images_tensor,
            "intrinsics": torch.from_numpy(intrinsics),
            "extrinsics": torch.from_numpy(extrinsics),
            "semantic_map": torch.from_numpy(semantic_map),
            "instance_map": torch.from_numpy(instance_map).long(),
            "direction_map": torch.from_numpy(direction_map),
        }


def collate_fn(batch):
    """Custom collation function for the HDMap dataset.

    Simply stacks all tensors along the batch dimension.

    Args:
        batch: List of sample dicts from __getitem__.

    Returns:
        Collated dict with batched tensors.
    """
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return collated
