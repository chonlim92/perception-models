"""
PointPillars PyTorch Implementation
====================================

A complete implementation of the PointPillars 3D object detection model for
LiDAR point clouds, based on:

    "PointPillars: Fast Encoders for Object Detection from Point Clouds"
    Lang et al., CVPR 2019

Modules:
    - pillar_feature_net: PillarFeatureNet for encoding point cloud into pillar features
    - scatter: PointPillarsScatter for creating BEV pseudo-images
    - backbone: BaseBEVBackbone 2D CNN with FPN neck
    - anchors: AnchorGenerator for 3D anchor box generation and encoding
    - anchor_head: SSD-style detection head with NMS
    - losses: Focal loss, Smooth L1, direction classification loss
    - dataset: KITTI and nuScenes dataset loaders with augmentation
    - model: Main PointPillars model combining all components
    - train: Training script with distributed support
    - evaluate: KITTI and nuScenes evaluation metrics
    - inference: Real-time inference with visualization
"""

from .pillar_feature_net import PillarFeatureNet
from .scatter import PointPillarsScatter
from .backbone import BaseBEVBackbone
from .anchors import AnchorGenerator, encode_boxes, decode_boxes
from .anchor_head import AnchorHead
from .losses import FocalLoss, WeightedSmoothL1Loss, DirectionClassificationLoss, PointPillarsLoss
from .model import PointPillars

__version__ = "1.0.0"
__all__ = [
    "PillarFeatureNet",
    "PointPillarsScatter",
    "BaseBEVBackbone",
    "AnchorGenerator",
    "AnchorHead",
    "FocalLoss",
    "WeightedSmoothL1Loss",
    "DirectionClassificationLoss",
    "PointPillarsLoss",
    "PointPillars",
    "encode_boxes",
    "decode_boxes",
]
