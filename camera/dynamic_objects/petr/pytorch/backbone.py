"""
ResNet-50 backbone with Feature Pyramid Network (FPN) for PETR.

Extracts multi-scale features from multi-view camera images using a
pretrained ResNet-50 and fuses them via lateral connections and
top-down pathway in FPN.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    """ResNet-50 backbone with configurable frozen stages.

    Args:
        pretrained: Whether to use ImageNet pretrained weights.
        frozen_stages: Number of stages to freeze (0-4). Stage 0 is the
            stem (conv1 + bn1), stages 1-4 are the residual layer groups.
        out_indices: Indices of stages whose outputs to return (0-indexed
            from layer1). Default (0,1,2,3) returns C2,C3,C4,C5.
    """

    def __init__(
        self,
        pretrained: bool = True,
        frozen_stages: int = 1,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices

        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)

        # Stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Residual stages
        self.layer1 = resnet.layer1  # C2: stride 4, 256 channels
        self.layer2 = resnet.layer2  # C3: stride 8, 512 channels
        self.layer3 = resnet.layer3  # C4: stride 16, 1024 channels
        self.layer4 = resnet.layer4  # C5: stride 32, 2048 channels

        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze parameters in stages up to frozen_stages."""
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()

        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f"layer{i}")
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "ResNet50Backbone":
        """Override train to keep frozen stages in eval mode."""
        super().train(mode)
        self._freeze_stages()
        if mode:
            # Keep frozen BN layers in eval mode
            if self.frozen_stages >= 0:
                self.bn1.eval()
            for i in range(1, self.frozen_stages + 1):
                layer = getattr(self, f"layer{i}")
                layer.eval()
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            List of feature maps at selected stages. For default
            out_indices=(0,1,2,3), returns features with channels
            [256, 512, 1024, 2048] at strides [4, 8, 16, 32].
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i, layer in enumerate(layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return outs


class FPN(nn.Module):
    """Feature Pyramid Network with lateral connections and top-down pathway.

    Takes multi-scale feature maps from the backbone and produces
    feature maps of uniform channel dimension at each scale.

    Args:
        in_channels: List of input channel dimensions from backbone
            (e.g., [256, 512, 1024, 2048] for ResNet-50).
        out_channels: Number of output channels for all FPN levels.
        num_outs: Number of output feature map levels. If greater than
            len(in_channels), extra levels are produced via stride-2
            convolutions on the last feature map.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_outs: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_ins = len(in_channels)

        # Lateral (1x1) convolutions to reduce channel dimensions
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            )

        # Output (3x3) convolutions to remove aliasing from upsampling
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

        # Extra downsampling layers if num_outs > num_ins
        self.extra_convs = nn.ModuleList()
        for i in range(num_outs - self.num_ins):
            self.extra_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Build FPN feature pyramid.

        Args:
            inputs: Multi-scale feature maps from backbone, ordered from
                lowest resolution to highest resolution (C2, C3, C4, C5).

        Returns:
            List of FPN feature maps (P2, P3, P4, P5, [P6...]) each with
            out_channels channels.
        """
        assert len(inputs) == self.num_ins

        # Build lateral features
        laterals = [
            self.lateral_convs[i](inputs[i]) for i in range(self.num_ins)
        ]

        # Top-down pathway: add upsampled higher-level features to
        # lower-level lateral features
        for i in range(self.num_ins - 2, -1, -1):
            h, w = laterals[i].shape[2:]
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=(h, w), mode="bilinear", align_corners=False
            )

        # Apply 3x3 convolutions to produce final outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # Generate extra levels via stride-2 convolutions
        if self.num_outs > self.num_ins:
            extra_input = outs[-1]
            for extra_conv in self.extra_convs:
                extra_input = F.relu(extra_conv(extra_input))
                outs.append(extra_input)

        return outs


class BackboneWithFPN(nn.Module):
    """Combined ResNet-50 backbone and FPN neck for multi-view images.

    Processes B*N_cameras images jointly through the backbone and FPN,
    returning multi-scale features suitable for PETR's 3D position
    embedding.

    Args:
        pretrained: Whether to use pretrained ResNet-50 weights.
        frozen_stages: Number of backbone stages to freeze.
        fpn_out_channels: FPN output channel dimension.
        fpn_num_outs: Number of FPN output levels.
    """

    def __init__(
        self,
        pretrained: bool = True,
        frozen_stages: int = 1,
        fpn_out_channels: int = 256,
        fpn_num_outs: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = ResNet50Backbone(
            pretrained=pretrained,
            frozen_stages=frozen_stages,
            out_indices=(0, 1, 2, 3),
        )
        self.fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=fpn_out_channels,
            num_outs=fpn_num_outs,
        )

    def forward(
        self, images: torch.Tensor
    ) -> List[torch.Tensor]:
        """Extract multi-scale features from multi-view images.

        Args:
            images: Tensor of shape (B, N_cams, 3, H, W) or (B*N_cams, 3, H, W).

        Returns:
            List of FPN feature maps. Each has shape
            (B*N_cams, C, H_i, W_i) where C = fpn_out_channels.
        """
        if images.dim() == 5:
            batch_size, num_cams = images.shape[:2]
            images = images.flatten(0, 1)  # (B*N, 3, H, W)

        backbone_feats = self.backbone(images)
        fpn_feats = self.fpn(backbone_feats)

        return fpn_feats
