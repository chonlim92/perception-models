"""PointPillars 2D CNN Backbone for Bird's-Eye View (BEV) pseudo-image feature extraction.

This module implements the BaseBEVBackbone, a multi-scale 2D convolutional backbone
that processes the BEV pseudo-image produced by the pillar feature net and scatter
operation. It follows the architecture described in:

    Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds",
    CVPR 2019.

The backbone consists of multiple downsampling blocks followed by a Feature Pyramid
Network (FPN)-style neck that upsamples and concatenates multi-scale features into a
single dense feature map suitable for detection heads.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    """Multi-scale 2D CNN backbone with FPN-style neck for BEV pseudo-images.

    The backbone is composed of sequential downsampling blocks. Each block begins
    with a stride-2 convolution for spatial downsampling, followed by several
    convolution-BatchNorm-ReLU layers at the same resolution. The neck upsamples
    each block's output to a common spatial resolution and concatenates them along
    the channel dimension, producing a rich multi-scale feature map.

    Args:
        in_channels: Number of input channels from the pillar scatter output
            (i.e., the pillar feature dimension). Typically 64.
        layer_nums: Number of convolutional layers (excluding the initial stride-2
            conv) in each block. For example, [3, 5, 5] means the first block has
            1 + 3 = 4 convolutions total, the second has 1 + 5 = 6, etc.
        layer_strides: Stride of the first convolution in each block. Typically
            [2, 2, 2] for progressive 2x downsampling.
        num_filters: Output channel count for each block. For example, [64, 128, 256].
        upsample_strides: Upsample factor for each block's output in the neck.
            Values >= 1 use transposed convolutions; a stride of 1 means a 1x1 conv
            with no spatial change.
        num_upsample_filters: Output channel count for each upsample branch.
            For example, [128, 128, 128] yields a final concatenated feature map
            with 384 channels.

    Example:
        >>> backbone = BaseBEVBackbone(
        ...     in_channels=64,
        ...     layer_nums=[3, 5, 5],
        ...     layer_strides=[2, 2, 2],
        ...     num_filters=[64, 128, 256],
        ...     upsample_strides=[1, 2, 4],
        ...     num_upsample_filters=[128, 128, 128],
        ... )
        >>> # BEV pseudo-image: batch=2, channels=64, height=496, width=432
        >>> x = torch.randn(2, 64, 496, 432)
        >>> out = backbone(x)
        >>> out.shape
        torch.Size([2, 384, 248, 216])
    """

    def __init__(
        self,
        in_channels: int = 64,
        layer_nums: List[int] | None = None,
        layer_strides: List[int] | None = None,
        num_filters: List[int] | None = None,
        upsample_strides: List[int] | None = None,
        num_upsample_filters: List[int] | None = None,
    ) -> None:
        super().__init__()

        # Apply defaults matching the original PointPillars paper configuration
        if layer_nums is None:
            layer_nums = [3, 5, 5]
        if layer_strides is None:
            layer_strides = [2, 2, 2]
        if num_filters is None:
            num_filters = [64, 128, 256]
        if upsample_strides is None:
            upsample_strides = [1, 2, 4]
        if num_upsample_filters is None:
            num_upsample_filters = [128, 128, 128]

        assert len(layer_nums) == len(layer_strides) == len(num_filters), (
            f"layer_nums ({len(layer_nums)}), layer_strides ({len(layer_strides)}), "
            f"and num_filters ({len(num_filters)}) must have the same length."
        )
        assert len(num_filters) == len(upsample_strides) == len(num_upsample_filters), (
            f"num_filters ({len(num_filters)}), upsample_strides ({len(upsample_strides)}), "
            f"and num_upsample_filters ({len(num_upsample_filters)}) must have the same length."
        )

        self.num_blocks = len(layer_nums)
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_filters = num_filters
        self.upsample_strides = upsample_strides
        self.num_upsample_filters = num_upsample_filters

        # =====================================================================
        # Build downsampling blocks
        # =====================================================================
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for block_idx in range(self.num_blocks):
            out_channels = num_filters[block_idx]
            stride = layer_strides[block_idx]
            num_layers = layer_nums[block_idx]

            # Each block starts with a stride-2 (or specified stride) convolution
            # for spatial downsampling, followed by num_layers same-resolution convs.
            block_layers: List[nn.Module] = []

            # Initial strided convolution for downsampling
            block_layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
            )
            block_layers.append(nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01))
            block_layers.append(nn.ReLU(inplace=True))

            # Subsequent same-resolution convolutions
            for _ in range(num_layers):
                block_layers.append(
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )
                block_layers.append(nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01))
                block_layers.append(nn.ReLU(inplace=True))

            self.blocks.append(nn.Sequential(*block_layers))
            current_channels = out_channels

        # =====================================================================
        # Build FPN-style upsample (neck) branches
        # =====================================================================
        self.deblocks = nn.ModuleList()

        for block_idx in range(self.num_blocks):
            in_ch = num_filters[block_idx]
            out_ch = num_upsample_filters[block_idx]
            us_stride = upsample_strides[block_idx]

            if us_stride >= 1:
                # Use transposed convolution for upsampling
                self.deblocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=us_stride,
                            stride=us_stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                # Fractional stride means additional downsampling (rare but supported)
                actual_stride = round(1.0 / us_stride)
                self.deblocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=actual_stride,
                            stride=actual_stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True),
                    )
                )

        # Total output channels is the sum of all upsample filter counts
        self.num_output_channels = sum(num_upsample_filters)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming normal initialization to all convolutional layers.

        Conv2d and ConvTranspose2d layers use Kaiming normal with fan-out mode
        and ReLU nonlinearity. BatchNorm layers are initialized with weight=1
        and bias=0.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone and neck.

        Processes the BEV pseudo-image through sequential downsampling blocks,
        then upsamples each block's output to a common resolution and concatenates
        them along the channel dimension.

        Args:
            x: Input BEV pseudo-image tensor of shape (B, C, H, W) where
                B is batch size, C is in_channels, H and W are spatial dimensions.

        Returns:
            Concatenated multi-scale feature map of shape
            (B, sum(num_upsample_filters), H', W') where H' and W' depend on
            the upsample configuration. With default settings and input spatial
            size H x W, the output is at H/2 x W/2 resolution with
            sum(num_upsample_filters) channels.
        """
        block_outputs: List[torch.Tensor] = []

        # Pass through each downsampling block sequentially
        current_input = x
        for block in self.blocks:
            current_input = block(current_input)
            block_outputs.append(current_input)

        # Upsample each block's output and concatenate
        upsampled_features: List[torch.Tensor] = []
        for idx, deblock in enumerate(self.deblocks):
            upsampled_features.append(deblock(block_outputs[idx]))

        # Concatenate along channel dimension
        output = torch.cat(upsampled_features, dim=1)

        return output

    def get_output_channels(self) -> int:
        """Return the number of output channels after concatenation.

        This is useful for downstream detection heads that need to know
        the input channel count.

        Returns:
            Total number of output channels (sum of num_upsample_filters).
        """
        return self.num_output_channels
