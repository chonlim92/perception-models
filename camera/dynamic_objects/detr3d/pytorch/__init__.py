"""
DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries.

PyTorch implementation of the DETR3D architecture for camera-based 3D
object detection in autonomous driving scenarios.

Modules:
    backbone: ResNet-101 + FPN multi-scale feature extractor
    feature_sampling: 3D-to-2D projection and bilinear feature sampling
    decoder: Transformer decoder with feature sampling cross-attention
    heads: Classification and 3D bounding box regression heads
    losses: Hungarian matching + focal loss + L1 regression loss
"""

from .backbone import ResNet101FPN, FPN
from .feature_sampling import (
    project_points_to_cameras,
    normalize_pixel_coords,
    feature_sampling,
    DETR3DFeatureSampler,
)
from .decoder import (
    DETR3DCrossAttention,
    DETR3DTransformerDecoderLayer,
    DETR3DTransformerDecoder,
)
from .heads import (
    MLP,
    DETR3DClassificationHead,
    DETR3DRegressionHead,
    DETR3DHead,
)
from .losses import (
    focal_loss,
    l1_loss,
    smooth_l1_loss,
    HungarianMatcher,
    DETR3DLoss,
)
from .model import DETR3D, DETR3DPostProcessor

__all__ = [
    'ResNet101FPN',
    'FPN',
    'project_points_to_cameras',
    'normalize_pixel_coords',
    'feature_sampling',
    'DETR3DFeatureSampler',
    'DETR3DCrossAttention',
    'DETR3DTransformerDecoderLayer',
    'DETR3DTransformerDecoder',
    'MLP',
    'DETR3DClassificationHead',
    'DETR3DRegressionHead',
    'DETR3DHead',
    'focal_loss',
    'l1_loss',
    'smooth_l1_loss',
    'HungarianMatcher',
    'DETR3DLoss',
    'DETR3D',
    'DETR3DPostProcessor',
]
