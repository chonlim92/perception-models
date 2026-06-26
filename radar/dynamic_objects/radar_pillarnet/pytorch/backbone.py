"""
2D BEV backbone with multi-scale feature fusion for RadarPillarNet.

The backbone processes the BEV pseudo-image from the pillar scatter module
through a series of strided convolutional blocks to extract multi-scale features.
A Feature Pyramid Network (FPN) then upsamples and concatenates features from
all scales to produce a dense, multi-resolution feature map for the detection head.

Architecture:
    Block1: 64 -> 64 channels, stride=1, 3 conv layers (1x spatial)
    Block2: 64 -> 128 channels, stride=2, 5 conv layers (1/2 spatial)
    Block3: 128 -> 256 channels, stride=2, 5 conv layers (1/4 spatial)

FPN:
    Deconv Block1: 64 -> 128, stride=1 (keeps 1x resolution)
    Deconv Block2: 128 -> 128, stride=2 (upsamples 1/2 -> 1x)
    Deconv Block3: 256 -> 128, stride=4 (upsamples 1/4 -> 1x)
    Output: concatenation of all three = 384 channels at 1x resolution
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """A single convolutional block: Conv2d -> BatchNorm2d -> ReLU.

    Used as the basic building block for the backbone's strided conv layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        """Initialize convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: conv -> bn -> relu.

        Args:
            x: (B, C_in, H, W) input feature map.

        Returns:
            (B, C_out, H', W') output feature map.
        """
        return self.relu(self.bn(self.conv(x)))


class RadarBackbone(nn.Module):
    """Multi-scale 2D convolutional backbone for BEV feature extraction.

    Processes the BEV pseudo-image through three blocks of increasing depth
    and decreasing spatial resolution. Each block starts with a strided
    convolution for downsampling, followed by same-resolution convolutions.

    Block configuration:
        Block 1: input 64ch, output 64ch, stride 1, 3 layers
        Block 2: input 64ch, output 128ch, stride 2, 5 layers
        Block 3: input 128ch, output 256ch, stride 2, 5 layers
    """

    def __init__(
        self,
        in_channels: int = 64,
        layer_nums: List[int] = None,
        layer_strides: List[int] = None,
        num_filters: List[int] = None,
    ) -> None:
        """Initialize the backbone.

        Args:
            in_channels: Number of input channels from pillar encoder (default 64).
            layer_nums: Number of conv layers per block (default [3, 5, 5]).
            layer_strides: Stride for the first conv in each block (default [1, 2, 2]).
            num_filters: Output channels per block (default [64, 128, 256]).
        """
        super().__init__()

        if layer_nums is None:
            layer_nums = [3, 5, 5]
        if layer_strides is None:
            layer_strides = [1, 2, 2]
        if num_filters is None:
            num_filters = [64, 128, 256]

        assert len(layer_nums) == len(layer_strides) == len(num_filters)
        self.num_blocks = len(layer_nums)

        # Build convolutional blocks
        blocks = []
        current_channels = in_channels

        for block_idx in range(self.num_blocks):
            layers = []
            out_ch = num_filters[block_idx]
            stride = layer_strides[block_idx]
            n_layers = layer_nums[block_idx]

            # First layer with potential stride for downsampling
            layers.append(
                ConvBlock(
                    current_channels,
                    out_ch,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                )
            )

            # Remaining layers at stride=1
            for _ in range(1, n_layers):
                layers.append(
                    ConvBlock(
                        out_ch,
                        out_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )

            blocks.append(nn.Sequential(*layers))
            current_channels = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = num_filters

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through all blocks, returning multi-scale features.

        Args:
            x: (B, 64, H, W) BEV pseudo-image from pillar scatter.

        Returns:
            List of feature maps at each scale:
                - block1_out: (B, 64, H, W) - stride 1
                - block2_out: (B, 128, H/2, W/2) - stride 2
                - block3_out: (B, 256, H/4, W/4) - stride 4
        """
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs


class RadarFPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion.

    Upsamples feature maps from all backbone scales to the same resolution
    and concatenates them to produce a rich, multi-scale feature map.

    Deconvolution layers:
        - Scale 1 (64ch, 1x): DeConv 64->128, stride=1 (no upsampling)
        - Scale 2 (128ch, 1/2x): DeConv 128->128, stride=2 (2x upsampling)
        - Scale 3 (256ch, 1/4x): DeConv 256->128, stride=4 (4x upsampling)

    Output: Concatenation of all upsampled features = 384 channels
    """

    def __init__(
        self,
        in_channels: List[int] = None,
        out_channels: List[int] = None,
        upsample_strides: List[int] = None,
    ) -> None:
        """Initialize FPN with deconvolution layers.

        Args:
            in_channels: Input channels from each backbone block (default [64, 128, 256]).
            out_channels: Output channels for each deconv block (default [128, 128, 128]).
            upsample_strides: Upsampling stride for each deconv (default [1, 2, 4]).
        """
        super().__init__()

        if in_channels is None:
            in_channels = [64, 128, 256]
        if out_channels is None:
            out_channels = [128, 128, 128]
        if upsample_strides is None:
            upsample_strides = [1, 2, 4]

        assert len(in_channels) == len(out_channels) == len(upsample_strides)

        self.deconv_blocks = nn.ModuleList()

        for i in range(len(in_channels)):
            stride = upsample_strides[i]
            if stride >= 1:
                # Use transposed convolution for upsampling
                deconv = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels[i],
                        out_channels[i],
                        kernel_size=stride,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                )
            else:
                # stride < 1 means downsampling (not used in default config)
                actual_stride = int(round(1.0 / stride))
                deconv = nn.Sequential(
                    nn.Conv2d(
                        in_channels[i],
                        out_channels[i],
                        kernel_size=actual_stride,
                        stride=actual_stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                )

            self.deconv_blocks.append(deconv)

        self.total_out_channels = sum(out_channels)  # 384 with defaults

    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """Upsample and concatenate multi-scale features.

        Args:
            multi_scale_features: List of tensors from backbone blocks:
                - (B, 64, H, W)
                - (B, 128, H/2, W/2)
                - (B, 256, H/4, W/4)

        Returns:
            (B, 384, H, W) concatenated multi-scale feature map.
        """
        upsampled = []
        for i, feat in enumerate(multi_scale_features):
            up = self.deconv_blocks[i](feat)
            upsampled.append(up)

        # All features should now have the same spatial dimensions
        # Handle potential size mismatches due to rounding
        target_h = upsampled[0].shape[2]
        target_w = upsampled[0].shape[3]

        aligned = [upsampled[0]]
        for i in range(1, len(upsampled)):
            if upsampled[i].shape[2] != target_h or upsampled[i].shape[3] != target_w:
                aligned.append(
                    nn.functional.interpolate(
                        upsampled[i],
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            else:
                aligned.append(upsampled[i])

        # Concatenate along channel dimension
        return torch.cat(aligned, dim=1)  # (B, 384, H, W)


class RadarBEVBackbone(nn.Module):
    """Combined BEV backbone: RadarBackbone + RadarFPN.

    This is the main backbone module that takes the BEV pseudo-image
    and produces the final multi-scale fused feature map for detection heads.
    """

    def __init__(
        self,
        in_channels: int = 64,
        layer_nums: List[int] = None,
        layer_strides: List[int] = None,
        num_filters: List[int] = None,
        upsample_strides: List[int] = None,
        num_upsample_filters: List[int] = None,
    ) -> None:
        """Initialize combined backbone.

        Args:
            in_channels: Input channels from pillar scatter.
            layer_nums: Conv layers per backbone block (default [3, 5, 5]).
            layer_strides: Strides per backbone block (default [1, 2, 2]).
            num_filters: Output channels per backbone block (default [64, 128, 256]).
            upsample_strides: FPN upsampling strides (default [1, 2, 4]).
            num_upsample_filters: FPN output channels (default [128, 128, 128]).
        """
        super().__init__()

        if layer_nums is None:
            layer_nums = [3, 5, 5]
        if layer_strides is None:
            layer_strides = [1, 2, 2]
        if num_filters is None:
            num_filters = [64, 128, 256]
        if upsample_strides is None:
            upsample_strides = [1, 2, 4]
        if num_upsample_filters is None:
            num_upsample_filters = [128, 128, 128]

        self.backbone = RadarBackbone(
            in_channels=in_channels,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
        )

        self.fpn = RadarFPN(
            in_channels=num_filters,
            out_channels=num_upsample_filters,
            upsample_strides=upsample_strides,
        )

        self.out_channels = self.fpn.total_out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and FPN.

        Args:
            x: (B, 64, H, W) BEV pseudo-image.

        Returns:
            (B, 384, H, W) multi-scale fused feature map.
        """
        multi_scale = self.backbone(x)
        fused = self.fpn(multi_scale)
        return fused
