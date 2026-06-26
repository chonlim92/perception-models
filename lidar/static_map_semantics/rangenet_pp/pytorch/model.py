"""Complete RangeNet++ model for LiDAR semantic segmentation.

Combines the DarkNet-53 encoder with a U-Net decoder to produce
per-pixel semantic predictions on range images.

Reference:
  "RangeNet++: Fast and Accurate LiDAR Semantic Segmentation"
  (Milioto et al., IROS 2019)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .backbone import DarkNet53Backbone
from .decoder import RangeNetDecoder


class RangeNetPP(nn.Module):
    """RangeNet++ full model: DarkNet-53 encoder + U-Net decoder.

    Input: Range image (B, 5, H, W) where 5 channels = [range, x, y, z, intensity]
    Output: Logits (B, num_classes, H, W)

    Default configuration matches SemanticKITTI:
        H=64, W=2048 (or 1024), num_classes=20
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Configuration dictionary with keys:
                - in_channels (int): Number of input channels (default: 5).
                - num_classes (int): Number of semantic classes (default: 20).
                - height (int): Range image height (default: 64).
                - width (int): Range image width (default: 2048).
                - dropout_p (float): Dropout probability (default: 0.01).
                - encoder_channels (list): Channel progression (default: [64,128,256,512,1024]).
        """
        super().__init__()

        if config is None:
            config = {}

        self.in_channels = config.get("in_channels", 5)
        self.num_classes = config.get("num_classes", 20)
        self.height = config.get("height", 64)
        self.width = config.get("width", 2048)
        self.dropout_p = config.get("dropout_p", 0.01)
        self.encoder_channels = config.get("encoder_channels", [64, 128, 256, 512, 1024])

        # Encoder: DarkNet-53 backbone
        self.encoder = DarkNet53Backbone(in_channels=self.in_channels)

        # Decoder: U-Net style upsampling with skip connections
        self.decoder = RangeNetDecoder(
            encoder_channels=self.encoder_channels,
            num_classes=self.num_classes,
            dropout_p=self.dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input range image (B, 5, H, W).

        Returns:
            Logits (B, num_classes, H, W).
        """
        input_size = (x.shape[2], x.shape[3])

        # Encode: extract multi-scale features
        encoder_features = self.encoder(x)

        # Decode: upsample with skip connections
        logits = self.decoder(encoder_features, input_size=input_size)

        return logits

    def get_num_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RangeNetPP":
        """Create model from configuration dictionary."""
        return cls(config=config)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Return default configuration for SemanticKITTI."""
        return {
            "in_channels": 5,
            "num_classes": 20,
            "height": 64,
            "width": 2048,
            "dropout_p": 0.01,
            "encoder_channels": [64, 128, 256, 512, 1024],
        }


class RangeNetPPWithAux(nn.Module):
    """RangeNet++ with auxiliary classification head for deep supervision.

    Adds an auxiliary loss at an intermediate decoder stage to help
    gradient flow during training.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        if config is None:
            config = {}

        self.in_channels = config.get("in_channels", 5)
        self.num_classes = config.get("num_classes", 20)
        self.height = config.get("height", 64)
        self.width = config.get("width", 2048)
        self.dropout_p = config.get("dropout_p", 0.01)
        self.encoder_channels = config.get("encoder_channels", [64, 128, 256, 512, 1024])

        # Encoder
        self.encoder = DarkNet53Backbone(in_channels=self.in_channels)

        # Decoder
        self.decoder = RangeNetDecoder(
            encoder_channels=self.encoder_channels,
            num_classes=self.num_classes,
            dropout_p=self.dropout_p,
        )

        # Auxiliary head from stage3 features (256 channels at 1/8 resolution)
        self.aux_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=self.dropout_p),
            nn.Conv2d(128, self.num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with auxiliary output.

        Args:
            x: Input range image (B, 5, H, W).

        Returns:
            Dictionary with:
                'logits': Main output (B, num_classes, H, W)
                'aux_logits': Auxiliary output (B, num_classes, H/8, W/8)
        """
        input_size = (x.shape[2], x.shape[3])

        encoder_features = self.encoder(x)
        logits = self.decoder(encoder_features, input_size=input_size)

        # Auxiliary head from stage3 (index 2)
        aux_logits = self.aux_head(encoder_features[2])

        return {"logits": logits, "aux_logits": aux_logits}


def build_model(config: Optional[Dict[str, Any]] = None, aux_loss: bool = False) -> nn.Module:
    """Factory function to build RangeNet++ model.

    Args:
        config: Model configuration dictionary.
        aux_loss: Whether to include auxiliary classification head.

    Returns:
        RangeNet++ model instance.
    """
    if aux_loss:
        return RangeNetPPWithAux(config=config)
    return RangeNetPP(config=config)
