"""Cylinder3D TensorFlow 2 implementation for LiDAR semantic segmentation.

This package provides a complete TF2 implementation of the Cylinder3D model
for 3D point cloud semantic segmentation on the SemanticKITTI dataset.

Modules:
    model: Cylinder3D model architecture (backbone, partition, refinement, losses)
    train: Training script with mixed precision, multi-GPU, data augmentation
    evaluate: Evaluation script computing per-class IoU metrics
    inference: Inference script for generating predictions and colored PLY output
"""

from .model import Cylinder3DModel, CombinedLoss

__version__ = "1.0.0"
__all__ = ["Cylinder3DModel", "CombinedLoss"]
