"""
PETR / PETRv2 / StreamPETR - Position Embedding TRansformation for 3D Object Detection.

This package implements camera-only 3D object detection using position-aware
transformers. The key innovation is encoding 3D world coordinates into image
features via learnable position embeddings, enabling a standard transformer
decoder to perform 3D detection without explicit depth estimation.

Variants:
    - PETR: Single-frame 3D detection with 3D position embedding.
    - PETRv2: Multi-frame temporal fusion via aligned position-aware features.
    - StreamPETR: Streaming detection with query propagation across frames.

References:
    - PETR: Position Embedding Transformation for Multi-View 3D Object Detection
      (Liu et al., ECCV 2022)
    - PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images
      (Liu et al., ICCV 2023)
    - StreamPETR: Exploring Object-Centric Temporal Modeling for Efficient
      Multi-View 3D Object Detection (Wang et al., ICCV 2023)
"""

from .backbone import BackboneWithFPN, FPN, ResNet50Backbone
from .decoder import PETRTransformerDecoder, TransformerDecoderLayer
from .heads import PETRDetectionHead, VelocityHead
from .losses import FocalLoss, HungarianMatcher, L1Loss, PETRLoss
from .model import PETRConfig, PETRModel
from .position_embedding_3d import PositionEmbedding3D
from .temporal import (
    EgoMotionCompensation,
    MotionAwareLayerNorm,
    QueryPropagation,
    TemporalMemory,
)

__all__ = [
    # Model
    "PETRModel",
    "PETRConfig",
    # Backbone
    "ResNet50Backbone",
    "FPN",
    "BackboneWithFPN",
    # Core modules
    "PositionEmbedding3D",
    "PETRTransformerDecoder",
    "TransformerDecoderLayer",
    # Heads
    "PETRDetectionHead",
    "VelocityHead",
    # Losses
    "FocalLoss",
    "L1Loss",
    "HungarianMatcher",
    "PETRLoss",
    # Temporal
    "QueryPropagation",
    "MotionAwareLayerNorm",
    "TemporalMemory",
    "EgoMotionCompensation",
]
