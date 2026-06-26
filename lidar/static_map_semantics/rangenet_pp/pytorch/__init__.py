"""RangeNet++ PyTorch implementation.

Fast and Accurate LiDAR Semantic Segmentation using range image representation
with DarkNet-53 encoder, U-Net decoder, and KNN post-processing.

Reference:
    "RangeNet++: Fast and Accurate LiDAR Semantic Segmentation"
    Milioto et al., IROS 2019
"""

from .model import RangeNetPP, RangeNetPPWithAux, build_model
from .backbone import DarkNet53Backbone
from .decoder import RangeNetDecoder
from .spherical_projection import SphericalProjection, SphericalProjectionTorch
from .losses import (
    WeightedCrossEntropyLoss,
    LovaszSoftmaxLoss,
    CombinedLoss,
    get_default_semantickitti_weights,
    compute_class_weights,
)
from .knn_postprocess import (
    knn_postprocess_numpy,
    knn_postprocess_numpy_fast,
    knn_postprocess_torch,
    knn_postprocess_torch_vectorized,
)
from .dataset import (
    SemanticKITTIRangeDataset,
    SemanticKITTIRangeInferenceDataset,
    SEMANTICKITTI_CLASS_NAMES,
    SEMANTICKITTI_LABEL_MAP,
)

__all__ = [
    "RangeNetPP",
    "RangeNetPPWithAux",
    "build_model",
    "DarkNet53Backbone",
    "RangeNetDecoder",
    "SphericalProjection",
    "SphericalProjectionTorch",
    "WeightedCrossEntropyLoss",
    "LovaszSoftmaxLoss",
    "CombinedLoss",
    "get_default_semantickitti_weights",
    "compute_class_weights",
    "knn_postprocess_numpy",
    "knn_postprocess_numpy_fast",
    "knn_postprocess_torch",
    "knn_postprocess_torch_vectorized",
    "SemanticKITTIRangeDataset",
    "SemanticKITTIRangeInferenceDataset",
    "SEMANTICKITTI_CLASS_NAMES",
    "SEMANTICKITTI_LABEL_MAP",
]
