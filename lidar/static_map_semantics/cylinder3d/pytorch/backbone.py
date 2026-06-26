"""
Cylinder3D Backbone: U-Net style encoder-decoder network.

Processes the 3D cylindrical voxel grid through an encoder-decoder architecture
with skip connections. Uses asymmetric convolutions throughout to efficiently
model the anisotropic structure of the cylindrical representation.

NOTE: This implementation uses standard dense 3D convolutions (torch.nn.Conv3d).
For production use with large-scale point clouds, consider replacing with sparse
convolution libraries such as:
    - MinkowskiEngine (https://github.com/NVIDIA/MinkowskiEngine)
    - SpConv (https://github.com/traveller59/spconv)
    - TorchSparse (https://github.com/mit-han-lab/torchsparse)
These sparse alternatives avoid computation on empty voxels and significantly
reduce memory usage and runtime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .asymmetric_convolution import (
    AsymmetricResBlock,
    AsymmetricDownBlock,
    AsymmetricUpBlock,
    DDCMod,
)


class Cylinder3DBackbone(nn.Module):
    """
    U-Net style encoder-decoder backbone for Cylinder3D.

    Architecture:
        Encoder:
            Stage 0: input_channels -> 32  (stem, no downsampling)
            Stage 1: 32 -> 64             (downsample 2x)
            Stage 2: 64 -> 128            (downsample 2x)
            Stage 3: 128 -> 256           (downsample 2x)
            Stage 4: 256 -> 256           (downsample 2x)
            Bottleneck: DDCMod at 256 channels

        Decoder:
            Stage 4: 256 + 256 -> 256     (upsample 2x, skip from enc stage 3)
            Stage 3: 256 + 128 -> 128     (upsample 2x, skip from enc stage 2)
            Stage 2: 128 + 64 -> 64      (upsample 2x, skip from enc stage 1)
            Stage 1: 64 + 32 -> 32       (upsample 2x, skip from enc stage 0)

        Head: 1x1x1 conv for per-voxel classification

    Args:
        input_channels: Number of input feature channels per voxel.
                        Default: 9 (from CylindricalPartition point features)
        num_classes: Number of semantic classes for segmentation.
                     Default: 20 (SemanticKITTI classes)
        base_channels: Base channel count for the first encoder stage.
                       Default: 32
        encoder_channels: Channel counts for encoder stages.
                          Default: [32, 64, 128, 256, 256]
    """

    def __init__(
        self,
        input_channels: int = 9,
        num_classes: int = 20,
        base_channels: int = 32,
        encoder_channels: Optional[List[int]] = None,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256, 256]

        self.num_classes = num_classes
        self.encoder_channels = encoder_channels

        # Stem: project input features to base channels
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            AsymmetricResBlock(base_channels, encoder_channels[0], stride=1),
        )

        # Encoder stages (each downsamples by 2x)
        self.encoder_stages = nn.ModuleList()
        for i in range(1, len(encoder_channels)):
            self.encoder_stages.append(
                AsymmetricDownBlock(
                    in_channels=encoder_channels[i - 1],
                    out_channels=encoder_channels[i],
                    num_blocks=2,
                )
            )

        # Bottleneck: DDCMod for global context modeling
        self.bottleneck = DDCMod(encoder_channels[-1])

        # Decoder stages (each upsamples by 2x with skip connections)
        decoder_channels = list(reversed(encoder_channels[:-1]))  # [128, 64, 32]
        self.decoder_stages = nn.ModuleList()

        dec_in = encoder_channels[-1]
        for i, dec_out in enumerate(decoder_channels):
            # Skip comes from encoder at the corresponding resolution
            skip_ch = encoder_channels[len(encoder_channels) - 2 - i]
            self.decoder_stages.append(
                AsymmetricUpBlock(
                    in_channels=dec_in,
                    skip_channels=skip_ch,
                    out_channels=dec_out,
                )
            )
            dec_in = dec_out

        # Classification head: 1x1x1 conv to produce per-voxel class scores
        self.head = nn.Sequential(
            nn.Conv3d(decoder_channels[-1], decoder_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm3d(decoder_channels[-1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(decoder_channels[-1], num_classes, kernel_size=1, bias=True),
        )

    def forward(
        self, voxel_volume: torch.Tensor
    ) -> dict:
        """
        Forward pass through the U-Net backbone.

        Args:
            voxel_volume: (B, C, D_rho, D_theta, D_z) dense voxel feature volume.
                          For sparse inputs, this should be the densified volume
                          (with zeros at unoccupied voxels).

        Returns:
            Dictionary containing:
                - 'voxel_logits': (B, num_classes, D_rho, D_theta, D_z) per-voxel class scores
                - 'voxel_features': (B, C_dec, D_rho, D_theta, D_z) decoder features
                                    before classification head (useful for point refinement)
        """
        # Encoder
        encoder_features = []

        # Stem (no downsampling)
        x = self.stem(voxel_volume)
        encoder_features.append(x)

        # Encoder stages (with downsampling)
        for stage in self.encoder_stages:
            x = stage(x)
            encoder_features.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (with upsampling and skip connections)
        # encoder_features: [stem_out, enc1_out, enc2_out, enc3_out, enc4_out]
        # We skip the last encoder feature (it goes through bottleneck) and
        # use the rest as skip connections in reverse order
        for i, stage in enumerate(self.decoder_stages):
            skip_idx = len(encoder_features) - 2 - i
            skip = encoder_features[skip_idx]
            x = stage(x, skip)

        # Save decoder features before head (for point refinement)
        decoder_features = x

        # Classification head
        voxel_logits = self.head(x)

        return {
            "voxel_logits": voxel_logits,
            "voxel_features": decoder_features,
        }
