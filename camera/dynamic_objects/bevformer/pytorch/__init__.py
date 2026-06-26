"""BEVFormer PyTorch implementation.

Multi-camera 3D object detection using spatiotemporal transformers
to construct Bird's-Eye-View representations.
"""

from .backbone import ResNetFPN
from .spatial_cross_attention import BEVFormerSpatialCrossAttention
from .temporal_self_attention import TemporalSelfAttention
from .bev_encoder import BEVFormerEncoder
from .decoder import BEVFormerDecoder
from .heads import BEVFormerHead
from .losses import BEVFormerLoss, HungarianMatcher
from .model import BEVFormer
from .dataset import NuScenesDataset

__all__ = [
    "ResNetFPN",
    "BEVFormerSpatialCrossAttention",
    "TemporalSelfAttention",
    "BEVFormerEncoder",
    "BEVFormerDecoder",
    "BEVFormerHead",
    "BEVFormerLoss",
    "HungarianMatcher",
    "BEVFormer",
    "NuScenesDataset",
]
