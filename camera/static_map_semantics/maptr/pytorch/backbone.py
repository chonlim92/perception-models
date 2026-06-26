"""
ResNet-50 + FPN backbone for MapTR.

Produces multi-scale feature maps from multi-camera images.
Input:  [B, N_cams, 3, H, W]
Output: list of feature maps at multiple scales per camera
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class Bottleneck(nn.Module):
    """Standard ResNet Bottleneck block with 1x1 -> 3x3 -> 1x1 convolutions."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    """
    Full ResNet-50 implementation producing 4 feature map stages.

    Output channels: [256, 512, 1024, 2048] at strides [4, 8, 16, 32].
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers: ResNet-50 = [3, 4, 6, 3] blocks
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self._initialize_weights()

        if pretrained:
            self._load_pretrained()

    def _make_layer(self, mid_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [Bottleneck(self.in_channels, mid_channels, stride, downsample)]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_channels, mid_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-init the last BN in each residual branch
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _load_pretrained(self):
        """Load pretrained weights from torchvision ResNet-50."""
        try:
            from torchvision.models import resnet50, ResNet50_Weights

            pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except (ImportError, TypeError):
            # Fallback for older torchvision versions
            from torchvision.models import resnet50

            pretrained_model = resnet50(pretrained=True)

        # Copy matching parameters
        own_state = self.state_dict()
        pretrained_state = pretrained_model.state_dict()
        for name, param in pretrained_state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] single image tensor

        Returns:
            List of 4 feature maps: C2, C3, C4, C5 with channels [256, 512, 1024, 2048]
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        c2 = self.layer1(x)   # stride 4,  channels 256
        c3 = self.layer2(c2)  # stride 8,  channels 512
        c4 = self.layer3(c3)  # stride 16, channels 1024
        c5 = self.layer4(c4)  # stride 32, channels 2048

        return [c2, c3, c4, c5]


class FPN(nn.Module):
    """
    Feature Pyramid Network.

    Takes multi-scale backbone features and produces unified-channel feature maps
    with top-down lateral connections.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_output_levels: int = 4,
    ):
        """
        Args:
            in_channels_list: Channel counts for each input level (e.g. [256, 512, 1024, 2048])
            out_channels: Unified output channel dimension
            num_output_levels: Number of output feature levels
        """
        super().__init__()
        self.num_input_levels = len(in_channels_list)
        self.num_output_levels = num_output_levels

        # Lateral 1x1 convolutions to reduce channels
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=True)
            )

        # Output 3x3 convolutions to reduce aliasing after upsampling
        self.output_convs = nn.ModuleList()
        for _ in range(self.num_input_levels):
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
            )

        # Extra levels via strided convolution if needed
        self.extra_convs = nn.ModuleList()
        for i in range(num_output_levels - self.num_input_levels):
            if i == 0:
                extra_in = in_channels_list[-1]
            else:
                extra_in = out_channels
            self.extra_convs.append(
                nn.Conv2d(extra_in, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of backbone features [C2, C3, C4, C5]

        Returns:
            List of FPN features [P2, P3, P4, P5, (P6, ...)]
        """
        assert len(features) == self.num_input_levels

        # Build lateral features
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down path with element-wise addition
        for i in range(self.num_input_levels - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode="bilinear", align_corners=False
            )
            laterals[i] = laterals[i] + upsampled

        # Apply output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        # Generate extra levels
        if self.extra_convs:
            extra_input = features[-1]
            for i, extra_conv in enumerate(self.extra_convs):
                if i == 0:
                    outputs.append(F.relu(extra_conv(extra_input)))
                else:
                    outputs.append(F.relu(extra_conv(outputs[-1])))

        return outputs


class ResNet50FPN(nn.Module):
    """
    Combined ResNet-50 + FPN backbone for MapTR.

    Processes multi-camera images and returns multi-scale FPN features per camera.

    Input:  [B, N_cams, 3, H, W]
    Output: List of [B * N_cams, C_out, H_i, W_i] feature maps at each FPN level
    """

    def __init__(
        self,
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        num_fpn_levels: int = 4,
    ):
        """
        Args:
            pretrained: Whether to load ImageNet pretrained weights for ResNet-50
            fpn_out_channels: Number of output channels for all FPN levels
            num_fpn_levels: Number of FPN output levels
        """
        super().__init__()
        self.resnet = ResNet50(pretrained=pretrained)
        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=fpn_out_channels,
            num_output_levels=num_fpn_levels,
        )
        self.out_channels = fpn_out_channels
        self.num_levels = num_fpn_levels

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            images: Multi-camera images [B, N_cams, 3, H, W]

        Returns:
            List of multi-scale features, each [B * N_cams, fpn_out_channels, H_i, W_i]
            where H_i, W_i correspond to stride 4, 8, 16, 32 (and more if extra levels).
        """
        B, N, C, H, W = images.shape

        # Flatten batch and camera dimensions for shared backbone processing
        x = images.reshape(B * N, C, H, W)

        # Extract multi-scale features from ResNet-50
        backbone_features = self.resnet(x)

        # Build FPN feature pyramid
        fpn_features = self.fpn(backbone_features)

        return fpn_features

    def forward_with_cam_split(self, images: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Alternative forward that returns features split per camera.

        Args:
            images: [B, N_cams, 3, H, W]

        Returns:
            List (per level) of list (per camera) of [B, C, H_i, W_i] features
        """
        B, N, C, H, W = images.shape
        fpn_features = self.forward(images)

        # Split back into per-camera features
        per_cam_features = []
        for feat in fpn_features:
            _, C_out, H_i, W_i = feat.shape
            feat_reshaped = feat.reshape(B, N, C_out, H_i, W_i)
            cam_list = [feat_reshaped[:, i] for i in range(N)]
            per_cam_features.append(cam_list)

        return per_cam_features
