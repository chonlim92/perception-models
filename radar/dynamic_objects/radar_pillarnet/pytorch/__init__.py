"""
RadarPillarNet - PointPillars adapted for automotive radar point clouds.

This package implements the RadarPillarNet architecture for 3D object detection
from radar point clouds. Key adaptations from standard PointPillars (LiDAR):
- Multi-sweep accumulation with ego-motion compensation
- Radar-specific features: RCS, radial velocity, time delta
- Larger pillar sizes (0.4m vs 0.16m) due to radar sparsity
- Velocity regression head for moving object tracking
- Clutter filtering based on dynamic properties

Architecture:
    RadarPreprocessing -> PillarEncoder -> PillarScatter -> RadarBEVBackbone -> RadarAnchorHead

Reference:
    Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds", CVPR 2019
    Adapted for radar with multi-sweep accumulation and velocity estimation.
"""

from .model import RadarPillarNet
from .pillar_encoder import PillarEncoder, PillarScatter
from .backbone import RadarBackbone, RadarFPN, RadarBEVBackbone
from .heads import RadarAnchorHead, AnchorGenerator
from .losses import RadarPillarNetLoss
from .radar_preprocessing import (
    RadarMultiSweepAccumulator,
    RadarClutterFilter,
    compensate_ego_motion,
    accumulate_sweeps,
)

__all__ = [
    "RadarPillarNet",
    "PillarEncoder",
    "PillarScatter",
    "RadarBackbone",
    "RadarFPN",
    "RadarBEVBackbone",
    "RadarAnchorHead",
    "AnchorGenerator",
    "RadarPillarNetLoss",
    "RadarMultiSweepAccumulator",
    "RadarClutterFilter",
    "compensate_ego_motion",
    "accumulate_sweeps",
]

__version__ = "1.0.0"
