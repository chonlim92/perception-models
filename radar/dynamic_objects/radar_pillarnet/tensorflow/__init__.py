"""
RadarPillarNet - PointPillars adapted for automotive radar point clouds.

TensorFlow 2.x / Keras implementation.

This package implements the RadarPillarNet architecture for 3D object detection
from radar point clouds. Key adaptations from standard PointPillars (LiDAR):
- Multi-sweep accumulation with ego-motion compensation
- Radar-specific features: RCS, radial velocity, time delta
- Larger pillar sizes (0.4m vs 0.16m) due to radar sparsity
- Velocity regression head for moving object tracking
- Clutter filtering based on dynamic properties

Architecture:
    PillarEncoder -> PillarScatter -> BEVBackbone -> AnchorHead

Reference:
    Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds", CVPR 2019
    Adapted for radar with multi-sweep accumulation and velocity estimation.
"""

from .model import (
    PillarEncoder,
    PillarScatter,
    BEVBackbone,
    AnchorHead,
    RadarPillarNet,
    DEFAULT_CONFIG,
    build_radar_pillarnet,
)

__all__ = [
    "PillarEncoder",
    "PillarScatter",
    "BEVBackbone",
    "AnchorHead",
    "RadarPillarNet",
    "DEFAULT_CONFIG",
    "build_radar_pillarnet",
]

__version__ = "1.0.0"
