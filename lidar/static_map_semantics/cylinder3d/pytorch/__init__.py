"""
Cylinder3D: Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation.

This package implements the Cylinder3D architecture for 3D semantic segmentation
of LiDAR point clouds, based on the paper:
"Cylinder3D: An Effective 3D Framework for Driving-scene LiDAR Semantic Segmentation"

Components:
    - CylindricalPartition: Converts point clouds to cylindrical voxel representation
    - AsymmetricConvBlock / DDCMod: Asymmetric 3D convolution building blocks
    - Cylinder3DBackbone: U-Net style encoder-decoder backbone
    - PointRefinementModule: Per-point prediction refinement
    - Cylinder3D: Full model integrating all components
    - LovaszSoftmaxLoss / CombinedLoss: Training losses
"""

from .cylindrical_partition import CylindricalPartition
from .asymmetric_convolution import AsymmetricConvBlock, DDCMod, AsymmetricResBlock
from .backbone import Cylinder3DBackbone
from .point_refinement import PointRefinementModule
from .model import Cylinder3D, create_cylinder3d
from .losses import LovaszSoftmaxLoss, CombinedLoss, WeightedCrossEntropyLoss

__all__ = [
    "CylindricalPartition",
    "AsymmetricConvBlock",
    "DDCMod",
    "AsymmetricResBlock",
    "Cylinder3DBackbone",
    "PointRefinementModule",
    "Cylinder3D",
    "create_cylinder3d",
    "LovaszSoftmaxLoss",
    "CombinedLoss",
    "WeightedCrossEntropyLoss",
]
