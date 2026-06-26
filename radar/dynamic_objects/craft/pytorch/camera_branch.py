"""
Camera Branch for CRAFT (Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer).

Implements a multi-view camera feature extraction pipeline using ResNet + FPN.
Each of the 6 nuScenes camera views is processed through a shared backbone to produce
multi-scale feature maps suitable for downstream BEV transformation and fusion.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models import ResNet50_Weights, ResNet101_Weights


class FPN(nn.Module):
    """Feature Pyramid Network.

    Merges multi-scale features from a backbone (C2-C5) using lateral connections
    and a top-down pathway to produce feature maps (P2-P5) with uniform channel
    dimension.

    Args:
        in_channels_list: List of input channel sizes for each backbone level [C2, C3, C4, C5].
        out_channels: Number of output channels for all FPN levels (default: 256).
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256) -> None:
        super().__init__()
        self.out_channels = out_channels

        # Lateral 1x1 convolutions to reduce channel dimensions
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            )

        # 3x3 convolutions applied after top-down merging to reduce aliasing
        self.output_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize FPN weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass through FPN.

        Args:
            features: List of backbone feature maps [C2, C3, C4, C5], ordered from
                      highest resolution to lowest resolution.

        Returns:
            List of FPN output feature maps [P2, P3, P4, P5], ordered from
            highest resolution to lowest resolution.
        """
        assert len(features) == len(self.lateral_convs), (
            f"Expected {len(self.lateral_convs)} feature levels, got {len(features)}"
        )

        # Apply lateral convolutions
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]

        # Top-down pathway: propagate from coarsest (P5) to finest (P2)
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample coarser level and add to finer level
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="nearest",
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply output 3x3 convolutions
        outputs = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

        return outputs


class MultiViewCameraBackbone(nn.Module):
    """Multi-view camera feature extractor using ResNet + FPN.

    Processes multiple camera views (e.g., 6 views for nuScenes) through a shared
    ResNet backbone followed by a Feature Pyramid Network. Each view is processed
    independently to produce multi-scale feature maps.

    Args:
        backbone_name: ResNet variant to use. One of 'resnet50' or 'resnet101'.
        pretrained: Whether to load ImageNet pretrained weights.
        fpn_out_channels: Number of output channels for all FPN levels.
        num_cameras: Number of camera views (default: 6 for nuScenes).
        frozen_stages: Number of backbone stages to freeze (0-4). Stage 0 is the
                       stem (conv1 + bn1). Stages 1-4 correspond to layer1-layer4.
    """

    # Channel dimensions for each ResNet layer output
    RESNET_CHANNELS = {
        "resnet50": [256, 512, 1024, 2048],   # C2, C3, C4, C5
        "resnet101": [256, 512, 1024, 2048],  # C2, C3, C4, C5
    }

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        num_cameras: int = 6,
        frozen_stages: int = 1,
    ) -> None:
        super().__init__()

        if backbone_name not in self.RESNET_CHANNELS:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                f"Choose from {list(self.RESNET_CHANNELS.keys())}"
            )

        self.backbone_name = backbone_name
        self.fpn_out_channels = fpn_out_channels
        self.num_cameras = num_cameras
        self.frozen_stages = frozen_stages

        # Build the ResNet backbone
        self.backbone = self._build_backbone(backbone_name, pretrained)

        # Build FPN
        in_channels_list = self.RESNET_CHANNELS[backbone_name]
        self.fpn = FPN(in_channels_list, out_channels=fpn_out_channels)

        # Freeze stages if required
        self._freeze_stages()

    def _build_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        """Build ResNet backbone and return it without the classification head.

        Args:
            backbone_name: Name of the backbone variant.
            pretrained: Whether to use pretrained ImageNet weights.

        Returns:
            ResNet backbone module (without avgpool and fc layers).
        """
        if backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet50(weights=weights)
        elif backbone_name == "resnet101":
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = resnet101(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Remove the average pooling and fully connected layers
        # We only need the feature extraction layers
        del backbone.avgpool
        del backbone.fc

        return backbone

    def _freeze_stages(self) -> None:
        """Freeze backbone stages up to the specified level.

        Stage 0: stem (conv1 + bn1)
        Stage 1-4: layer1 - layer4
        """
        if self.frozen_stages >= 0:
            # Freeze stem
            self.backbone.conv1.requires_grad_(False)
            self.backbone.bn1.requires_grad_(False)
            self.backbone.bn1.eval()

        # Freeze layer stages
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self.backbone, f"layer{i}", None)
            if layer is not None:
                layer.requires_grad_(False)
                # Set BatchNorm layers to eval mode
                for module in layer.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        module.eval()

    def _extract_backbone_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from the ResNet backbone.

        Args:
            x: Input tensor of shape [B_flat, 3, H, W] where B_flat = B * N_cams.

        Returns:
            List of feature tensors [C2, C3, C4, C5] at decreasing resolutions.
        """
        backbone = self.backbone

        # Stem: conv1 -> bn1 -> relu -> maxpool
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        # Residual layers
        c2 = backbone.layer1(x)   # stride 4,  channels: 256
        c3 = backbone.layer2(c2)  # stride 8,  channels: 512
        c4 = backbone.layer3(c3)  # stride 16, channels: 1024
        c5 = backbone.layer4(c4)  # stride 32, channels: 2048

        return [c2, c3, c4, c5]

    def forward(
        self, images: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        """Process multi-view camera images through backbone + FPN.

        Args:
            images: Input tensor of shape [B, N_cams, 3, H, W].
                    B: batch size
                    N_cams: number of camera views (default 6)
                    3: RGB channels
                    H, W: image height and width

        Returns:
            Dictionary with key 'features' mapping to a list of 4 tensors (P2-P5),
            each of shape [B, N_cams, C, H_i, W_i] where:
                C = fpn_out_channels (256)
                H_i, W_i = H/(2^(i+1)), W/(2^(i+1)) for level i (i=1..4)
        """
        B, N, C, H, W = images.shape
        assert N == self.num_cameras, (
            f"Expected {self.num_cameras} camera views, got {N}"
        )
        assert C == 3, f"Expected 3 input channels (RGB), got {C}"

        # Reshape to process all views through backbone simultaneously
        # [B, N_cams, 3, H, W] -> [B*N_cams, 3, H, W]
        images_flat = images.reshape(B * N, C, H, W)

        # Extract multi-scale backbone features
        backbone_features = self._extract_backbone_features(images_flat)

        # Apply FPN to get P2-P5
        fpn_features = self.fpn(backbone_features)

        # Reshape back to per-view format
        # [B*N_cams, C_fpn, H_i, W_i] -> [B, N_cams, C_fpn, H_i, W_i]
        output_features = []
        for feat in fpn_features:
            _, C_fpn, H_feat, W_feat = feat.shape
            feat_reshaped = feat.reshape(B, N, C_fpn, H_feat, W_feat)
            output_features.append(feat_reshaped)

        return {"features": output_features}

    def train(self, mode: bool = True) -> "MultiViewCameraBackbone":
        """Override train to keep frozen BatchNorm layers in eval mode."""
        super().train(mode)
        if mode:
            self._freeze_stages()
        return self

    def get_output_info(self) -> Dict[str, any]:
        """Return metadata about the output feature maps.

        Returns:
            Dictionary with output channel count and number of FPN levels.
        """
        return {
            "num_levels": 4,  # P2, P3, P4, P5
            "out_channels": self.fpn_out_channels,
            "strides": [4, 8, 16, 32],  # Relative to input image
        }


def build_camera_branch(
    backbone_name: str = "resnet50",
    pretrained: bool = True,
    fpn_out_channels: int = 256,
    num_cameras: int = 6,
    frozen_stages: int = 1,
) -> MultiViewCameraBackbone:
    """Factory function to build the camera branch.

    Args:
        backbone_name: ResNet variant ('resnet50' or 'resnet101').
        pretrained: Whether to load ImageNet pretrained weights.
        fpn_out_channels: Output channels for all FPN levels.
        num_cameras: Number of camera views.
        frozen_stages: Number of backbone stages to freeze.

    Returns:
        Configured MultiViewCameraBackbone instance.
    """
    return MultiViewCameraBackbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        fpn_out_channels=fpn_out_channels,
        num_cameras=num_cameras,
        frozen_stages=frozen_stages,
    )


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_camera_branch(
        backbone_name="resnet50",
        pretrained=False,  # Avoid downloading weights for test
        fpn_out_channels=256,
        num_cameras=6,
        frozen_stages=1,
    ).to(device)

    # Simulate nuScenes multi-view input: 6 cameras, 900x1600 images
    # Using smaller size for quick test
    batch_size = 2
    num_cams = 6
    H, W = 256, 448  # Reduced size for testing

    dummy_input = torch.randn(batch_size, num_cams, 3, H, W, device=device)

    with torch.no_grad():
        output = model(dummy_input)

    print("Camera Branch Output:")
    print(f"  Number of FPN levels: {len(output['features'])}")
    for i, feat in enumerate(output["features"]):
        print(f"  P{i+2}: shape = {feat.shape}")

    # Verify output shapes
    for i, feat in enumerate(output["features"]):
        stride = 2 ** (i + 2)
        expected_h = H // stride
        expected_w = W // stride
        assert feat.shape == (batch_size, num_cams, 256, expected_h, expected_w), (
            f"P{i+2} shape mismatch: expected "
            f"{(batch_size, num_cams, 256, expected_h, expected_w)}, got {feat.shape}"
        )

    print("\nAll shape checks passed!")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
