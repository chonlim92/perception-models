"""
CenterPoint PyTorch Implementation.

A center-based 3D object detection framework for LiDAR point clouds.
Architecture: Voxelization -> 3D Sparse CNN -> BEV Backbone -> Center Head.
"""

from .voxelization import DynamicVoxelization, PillarFeatureExtraction, points_to_voxel
from .sparse_backbone import (
    SparseTensor,
    SparseConv3d,
    SubmanifoldSparseConv3d,
    SparseBasicBlock,
    SparseCNNBackbone,
)
from .bev_backbone import SparseToBEV, BEVBackbone, BEVFeatureNet
from .center_head import CenterHead, SeparateHead, gaussian_radius, draw_gaussian
from .two_stage import CenterPointTwoStage, PointFeatureExtractor, RefinementHead
from .losses import GaussianFocalLoss, RegLoss, CenterPointLoss

__all__ = [
    # Voxelization
    'DynamicVoxelization',
    'PillarFeatureExtraction',
    'points_to_voxel',
    # Sparse Backbone
    'SparseTensor',
    'SparseConv3d',
    'SubmanifoldSparseConv3d',
    'SparseBasicBlock',
    'SparseCNNBackbone',
    # BEV Backbone
    'SparseToBEV',
    'BEVBackbone',
    'BEVFeatureNet',
    # Detection Head
    'CenterHead',
    'SeparateHead',
    'gaussian_radius',
    'draw_gaussian',
    # Two-Stage
    'CenterPointTwoStage',
    'PointFeatureExtractor',
    'RefinementHead',
    # Losses
    'GaussianFocalLoss',
    'RegLoss',
    'CenterPointLoss',
]
