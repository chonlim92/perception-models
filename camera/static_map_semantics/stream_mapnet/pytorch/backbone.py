"""
StreamMapNet Backbone: ResNet-50 + Feature Pyramid Network (FPN)

Extracts multi-scale image features from surround-view camera inputs (6 cameras).
Produces C3 and C4 level features suitable for BEV transformation via LSS.

Architecture:
    Input: (B*6, 3, H, W) multi-camera images
    ResNet-50 layers -> C2 (stride 4), C3 (stride 8), C4 (stride 16), C5 (stride 32)
    FPN lateral + top-down pathway -> P3 (stride 8), P4 (stride 16)
    Output: C3-level features (B*6, C_fpn, H/8, W/8)
            C4-level features (B*6, C_fpn, H/16, W/16)

Reference:
    - He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
    - Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
    - Philion & Fidler, "Lift, Splat, Shoot", ECCV 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, Tuple


class FPNBlock(nn.Module):
    """Single FPN lateral connection + top-down fusion block.

    Takes a high-resolution lateral feature and a low-resolution top-down feature,
    fuses them via element-wise addition after matching channels and spatial dims.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Number of channels from the backbone feature map.
            out_channels: Number of output channels after lateral 1x1 conv.
        """
        super().__init__()
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.lateral_bn = nn.BatchNorm2d(out_channels)
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.output_bn = nn.BatchNorm2d(out_channels)

    def forward(self, lateral_input: torch.Tensor, top_down_input: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            lateral_input: Feature map from backbone layer, (B, C_in, H, W).
            top_down_input: Upsampled feature from higher FPN level, (B, C_out, H', W').
                           If None, this is the top-most level (P5).

        Returns:
            Fused feature map, (B, C_out, H, W).
        """
        lateral = self.lateral_bn(self.lateral_conv(lateral_input))

        if top_down_input is not None:
            # Upsample top-down to match lateral spatial dimensions
            top_down = F.interpolate(
                top_down_input,
                size=lateral.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            lateral = lateral + top_down

        out = self.output_bn(self.output_conv(lateral))
        out = F.relu(out, inplace=True)
        return out


class ResNet50FPN(nn.Module):
    """ResNet-50 backbone with Feature Pyramid Network for multi-camera feature extraction.

    Processes surround-view camera images (typically 6 cameras) in a batched manner.
    Outputs multi-scale features at C3 (stride 8) and C4 (stride 16) levels, which
    are used downstream by the LSS-based BEV transformation module.

    The backbone uses ImageNet-pretrained weights and the FPN is initialized with
    Kaiming initialization for stable training.
    """

    def __init__(
        self,
        fpn_out_channels: int = 256,
        pretrained: bool = True,
        freeze_bn: bool = True,
    ):
        """
        Args:
            fpn_out_channels: Number of output channels for each FPN level.
            pretrained: Whether to use ImageNet-pretrained ResNet-50 weights.
            freeze_bn: Whether to freeze BatchNorm layers in the ResNet backbone.
                      Recommended when using small batch sizes per GPU.
        """
        super().__init__()
        self.fpn_out_channels = fpn_out_channels
        self.freeze_bn = freeze_bn

        # Load pretrained ResNet-50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            backbone = resnet50(weights=weights)
        else:
            backbone = resnet50(weights=None)

        # Extract backbone stages
        # stem: conv1 + bn1 + relu + maxpool -> stride 4, 64 channels
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        # ResNet layers
        self.layer1 = backbone.layer1  # C2: stride 4, 256 channels
        self.layer2 = backbone.layer2  # C3: stride 8, 512 channels
        self.layer3 = backbone.layer3  # C4: stride 16, 1024 channels
        self.layer4 = backbone.layer4  # C5: stride 32, 2048 channels

        # FPN lateral connections and output convolutions
        # We build top-down pathway from C5 -> C4 -> C3
        self.fpn_c5 = FPNBlock(in_channels=2048, out_channels=fpn_out_channels)
        self.fpn_c4 = FPNBlock(in_channels=1024, out_channels=fpn_out_channels)
        self.fpn_c3 = FPNBlock(in_channels=512, out_channels=fpn_out_channels)

        # Initialize FPN weights
        self._init_fpn_weights()

        # Optionally freeze backbone batch norm
        if self.freeze_bn:
            self._freeze_backbone_bn()

    def _init_fpn_weights(self):
        """Initialize FPN convolution layers with Kaiming normal initialization."""
        for module in [self.fpn_c5, self.fpn_c4, self.fpn_c3]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def _freeze_backbone_bn(self):
        """Freeze all BatchNorm layers in the ResNet backbone.

        Sets them to eval mode and disables gradient computation for their parameters.
        This is important when fine-tuning with small batch sizes, as BN statistics
        become unreliable.
        """
        for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train to keep frozen BN layers in eval mode."""
        super().train(mode)
        if mode and self.freeze_bn:
            self._freeze_backbone_bn()
        return self

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale FPN features from multi-camera images.

        Args:
            images: Input tensor of shape (B*num_cams, 3, H, W) where num_cams
                   is typically 6 for surround-view setups. Images should be
                   normalized with ImageNet mean/std.
                   Typical input size: (B*6, 3, 256, 704) or (B*6, 3, 480, 800).

        Returns:
            Dictionary with keys:
                'c3': FPN features at stride 8, shape (B*num_cams, C_fpn, H/8, W/8)
                'c4': FPN features at stride 16, shape (B*num_cams, C_fpn, H/16, W/16)

            For input (B*6, 3, 256, 704):
                'c3': (B*6, 256, 32, 88)
                'c4': (B*6, 256, 16, 44)
        """
        # Backbone forward pass
        x = self.stem(images)       # (B*6, 64, H/4, W/4)
        c2 = self.layer1(x)         # (B*6, 256, H/4, W/4)
        c3 = self.layer2(c2)        # (B*6, 512, H/8, W/8)
        c4 = self.layer3(c3)        # (B*6, 1024, H/16, W/16)
        c5 = self.layer4(c4)        # (B*6, 2048, H/32, W/32)

        # FPN top-down pathway
        p5 = self.fpn_c5(c5, None)          # (B*6, 256, H/32, W/32)
        p4 = self.fpn_c4(c4, p5)            # (B*6, 256, H/16, W/16)
        p3 = self.fpn_c3(c3, p4)            # (B*6, 256, H/8, W/8)

        return {
            "c3": p3,
            "c4": p4,
        }


class StreamMapNetBackbone(nn.Module):
    """Full backbone wrapper that handles multi-camera batching and feature extraction.

    This module reshapes multi-camera inputs from (B, num_cams, 3, H, W) to
    (B*num_cams, 3, H, W), runs through ResNet50-FPN, and reshapes outputs back
    to (B, num_cams, C, H', W') format for downstream BEV transformation.
    """

    def __init__(
        self,
        num_cams: int = 6,
        fpn_out_channels: int = 256,
        pretrained: bool = True,
        freeze_bn: bool = True,
    ):
        """
        Args:
            num_cams: Number of surround-view cameras. Default 6 for typical
                     autonomous driving setup (front, front-left, front-right,
                     back, back-left, back-right).
            fpn_out_channels: Channel dimension of FPN output features.
            pretrained: Use ImageNet-pretrained ResNet-50 weights.
            freeze_bn: Freeze backbone BN layers during training.
        """
        super().__init__()
        self.num_cams = num_cams
        self.backbone = ResNet50FPN(
            fpn_out_channels=fpn_out_channels,
            pretrained=pretrained,
            freeze_bn=freeze_bn,
        )

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from multi-camera images.

        Args:
            images: (B, num_cams, 3, H, W) multi-camera input images,
                   normalized with ImageNet statistics.

        Returns:
            c3_features: (B, num_cams, C_fpn, H/8, W/8) features at stride 8.
            c4_features: (B, num_cams, C_fpn, H/16, W/16) features at stride 16.
        """
        B, N, C, H, W = images.shape
        assert N == self.num_cams, f"Expected {self.num_cams} cameras, got {N}"

        # Flatten batch and camera dimensions for efficient processing
        images_flat = images.view(B * N, C, H, W)

        # Extract FPN features
        features = self.backbone(images_flat)

        # Reshape back to (B, num_cams, C_fpn, H', W')
        c3 = features["c3"]
        c4 = features["c4"]

        _, C_fpn, H3, W3 = c3.shape
        _, _, H4, W4 = c4.shape

        c3 = c3.view(B, N, C_fpn, H3, W3)
        c4 = c4.view(B, N, C_fpn, H4, W4)

        return c3, c4


if __name__ == "__main__":
    # Demonstration with typical autonomous driving dimensions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    batch_size = 2
    num_cams = 6
    img_h, img_w = 256, 704  # Typical BEV perception input resolution

    # Create model
    model = StreamMapNetBackbone(
        num_cams=num_cams,
        fpn_out_channels=256,
        pretrained=False,  # Set False for quick testing without downloading weights
        freeze_bn=True,
    ).to(device)

    # Synthetic input: 6 cameras, RGB images
    images = torch.randn(batch_size, num_cams, 3, img_h, img_w, device=device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        c3_feat, c4_feat = model(images)

    print(f"Input shape:  {images.shape}")
    print(f"C3 features:  {c3_feat.shape}")  # Expected: (2, 6, 256, 32, 88)
    print(f"C4 features:  {c4_feat.shape}")  # Expected: (2, 6, 256, 16, 44)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
