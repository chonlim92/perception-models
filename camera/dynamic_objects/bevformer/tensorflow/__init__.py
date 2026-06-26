"""BEVFormer TensorFlow/Keras implementation.

Multi-camera 3D object detection using spatiotemporal transformers
to construct Bird's-Eye-View representations.
"""

from .model import BEVFormerModel

__all__ = ["BEVFormerModel"]
