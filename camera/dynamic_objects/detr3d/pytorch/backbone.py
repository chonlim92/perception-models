"""
DETR3D Backbone: ResNet-101 + Feature Pyramid Network (FPN).

Extracts multi-scale features from input images at strides 8, 16, 32, 64.
Uses torchvision pretrained ResNet-101 as the base encoder and builds a
standard FPN with lateral connections and top-down pathway.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict


class FPN(nn.Module):
    """Feature Pyramid Network.

    Takes multi-scale feature maps from the backbone (C2, C3, C4, C5) and
    produces pyramid features (P2, P3, P4, P5) each with `out_channels` channels.
    Includes lateral connections (1x1 conv) and top-down upsampling pathway,
    followed by 3x3 conv to reduce aliasing.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        """
        Args:
            in_channels_list: Number of channels for each input feature level
                              [C2_channels, C3_channels, C4_channels, C5_channels].
            out_channels: Number of output channels for all pyramid levels.
        """
        super().__init__()
        self.out_channels = out_channels

        # Lateral 1x1 convolutions to reduce channel dimension
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            )

        # 3x3 convolutions applied after top-down addition to reduce aliasing
        self.output_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature tensors [C2, C3, C4, C5] from backbone.

        Returns:
            List of pyramid feature tensors [P2, P3, P4, P5], each with
            `out_channels` channels.
        """
        assert len(features) == len(self.lateral_convs)

        # Build lateral features (1x1 conv)
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]

        # Top-down pathway: from highest level (smallest spatial) downward
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher level and add to lower level
            h, w = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=(h, w), mode='bilinear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply 3x3 conv to each level
        outputs = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

        return outputs


class ResNet101FPN(nn.Module):
    """ResNet-101 backbone with FPN.

    Uses torchvision's pretrained ResNet-101 to extract features at multiple
    scales (C2, C3, C4, C5 corresponding to layer1-4 outputs), then applies
    FPN to produce P2, P3, P4, P5 features at strides 8, 16, 32, 64.
    """

    def __init__(
        self,
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        frozen_stages: int = 1,
    ):
        """
        Args:
            pretrained: Whether to use pretrained ResNet-101 weights.
            fpn_out_channels: Number of output channels for FPN levels.
            frozen_stages: Number of stages to freeze (0=none, 1=stem+layer1, etc.).
        """
        super().__init__()
        self.fpn_out_channels = fpn_out_channels
        self.frozen_stages = frozen_stages

        # Load pretrained ResNet-101
        if pretrained:
            resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet101(weights=None)

        # Stem: conv1, bn1, relu, maxpool
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Residual layers
        self.layer1 = resnet.layer1  # stride 4, 256 channels (C2)
        self.layer2 = resnet.layer2  # stride 8, 512 channels (C3)
        self.layer3 = resnet.layer3  # stride 16, 1024 channels (C4)
        self.layer4 = resnet.layer4  # stride 32, 2048 channels (C5)

        # FPN: takes C2(256), C3(512), C4(1024), C5(2048)
        # Note: We use C3, C4, C5 at strides 8, 16, 32 and add an extra level P5
        # for stride 64 via a stride-2 conv on P5.
        # Standard DETR3D uses features at strides 8, 16, 32, 64.
        # We produce P3, P4, P5 from FPN and P6 via stride-2 max pool on P5.
        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=fpn_out_channels,
        )

        # Extra level P6 via stride-2 conv on C5 (for stride 64)
        self.p6_conv = nn.Conv2d(2048, fpn_out_channels, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_uniform_(self.p6_conv.weight, a=1)
        nn.init.constant_(self.p6_conv.bias, 0)

        # Freeze stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters in early stages."""
        if self.frozen_stages >= 0:
            # Freeze stem
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()

        frozen_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(min(self.frozen_stages, 4)):
            layer = frozen_layers[i]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train to keep frozen stages in eval mode."""
        super().train(mode)
        self._freeze_stages()
        if mode:
            # Keep frozen BN layers in eval
            if self.frozen_stages >= 0:
                self.bn1.eval()
            frozen_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for i in range(min(self.frozen_stages, 4)):
                frozen_layers[i].eval()
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary with keys 'p2', 'p3', 'p4', 'p5', 'p6' mapping to
            feature tensors. The effective strides are:
                p2: stride 4 (from layer1)
                p3: stride 8 (from layer2)
                p4: stride 16 (from layer3)
                p5: stride 32 (from layer4)
                p6: stride 64 (from extra conv on C5)

            For DETR3D multi-scale features at strides 8, 16, 32, 64 we use
            p3, p4, p5, p6.
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        c2 = self.layer1(x)   # stride 4
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32

        # FPN
        fpn_features = self.fpn([c2, c3, c4, c5])
        p2, p3, p4, p5 = fpn_features

        # Extra level P6 for stride 64
        p6 = self.p6_conv(c5)

        return {
            'p2': p2,  # stride 4
            'p3': p3,  # stride 8
            'p4': p4,  # stride 16
            'p5': p5,  # stride 32
            'p6': p6,  # stride 64
        }

    def get_multi_scale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Convenience method returning features at strides 8, 16, 32, 64.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            List of 4 feature tensors at strides [8, 16, 32, 64].
        """
        feat_dict = self.forward(x)
        return [feat_dict['p3'], feat_dict['p4'], feat_dict['p5'], feat_dict['p6']]
