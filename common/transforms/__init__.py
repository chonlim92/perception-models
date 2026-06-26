"""Transform utilities for autonomous driving perception models.

This module provides:
- Data augmentation transforms for point clouds, images, and BEV features
- Coordinate system conversions (camera, LiDAR, ego, world, BEV, image)
"""

from common.transforms.augmentations import (
    BaseTransform,
    BEVGridMask,
    BEVRandomFlip,
    BEVRandomRotate,
    Compose,
    GlobalRotScaleTrans,
    GridMask,
    ImageNormalize,
    ImageResize,
    PhotoMetricDistortion,
    RandomCropImage,
    RandomDropPoints,
    RandomFlip3D,
    RandomRotate3D,
    RandomScale3D,
    RandomTranslate3D,
)
from common.transforms.coordinates import (
    apply_transform,
    bev_to_world,
    camera_to_lidar,
    compose_transforms,
    create_projection_matrix,
    ego_to_lidar,
    ego_to_world,
    euler_to_rotation_matrix,
    inverse_transform,
    lidar_to_camera,
    lidar_to_ego,
    make_transform,
    points_in_frustum,
    project_points_to_image,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler,
    world_to_bev,
    world_to_ego,
)

__all__ = [
    # Base / composition
    "BaseTransform",
    "Compose",
    # 3D Point Cloud Augmentations
    "RandomFlip3D",
    "RandomRotate3D",
    "RandomScale3D",
    "RandomTranslate3D",
    "RandomDropPoints",
    "GlobalRotScaleTrans",
    # Image Augmentations
    "PhotoMetricDistortion",
    "ImageNormalize",
    "ImageResize",
    "RandomCropImage",
    "GridMask",
    # BEV Augmentations
    "BEVRandomFlip",
    "BEVRandomRotate",
    "BEVGridMask",
    # Coordinate conversions
    "quaternion_to_rotation_matrix",
    "euler_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "make_transform",
    "inverse_transform",
    "compose_transforms",
    "apply_transform",
    "lidar_to_camera",
    "camera_to_lidar",
    "lidar_to_ego",
    "ego_to_lidar",
    "ego_to_world",
    "world_to_ego",
    "create_projection_matrix",
    "project_points_to_image",
    "world_to_bev",
    "bev_to_world",
    "points_in_frustum",
]
