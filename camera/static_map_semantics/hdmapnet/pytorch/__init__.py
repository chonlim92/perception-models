"""
HDMapNet: An Online HD Map Construction and Evaluation Framework
PyTorch Implementation

Paper: Li et al., "HDMapNet: An Online HD Map Construction and Evaluation Framework", ICRA 2022

This package provides a complete implementation including:
- Multi-camera feature extraction with EfficientNet-B0 / ResNet-50 backbones
- IPM and LSS (Lift-Splat-Shoot) view transforms to BEV
- BEV encoder with residual blocks
- Semantic segmentation, instance embedding, and direction prediction heads
- Training, evaluation, and inference pipelines
"""

from .model import HDMapNet
from .backbone import EfficientNetB0Backbone, ResNet50Backbone
from .view_transform import IPMTransform, LSSTransform
from .bev_encoder import BEVEncoder
from .heads import SemanticHead, InstanceHead, DirectionHead
from .losses import HDMapNetLoss, SemanticLoss, DiscriminativeLoss, DirectionLoss

__all__ = [
    "HDMapNet",
    "EfficientNetB0Backbone",
    "ResNet50Backbone",
    "IPMTransform",
    "LSSTransform",
    "BEVEncoder",
    "SemanticHead",
    "InstanceHead",
    "DirectionHead",
    "HDMapNetLoss",
    "SemanticLoss",
    "DiscriminativeLoss",
    "DirectionLoss",
]
