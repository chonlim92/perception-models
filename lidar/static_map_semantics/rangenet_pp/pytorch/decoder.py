"""U-Net decoder with skip connections for RangeNet++.

Progressive upsampling decoder that takes multi-scale encoder features and
produces full-resolution semantic predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DecoderBlock(nn.Module):
    """Single decoder block: upsample -> concat skip -> conv -> conv.

    Uses bilinear upsampling followed by two 3x3 convolutions with
    BatchNorm and LeakyReLU activation.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout_p: float = 0.01):
        """
        Args:
            in_channels: Number of channels from the lower-resolution feature map.
            skip_channels: Number of channels from the encoder skip connection.
            out_channels: Number of output channels after this block.
            dropout_p: Dropout probability for regularization.
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # After concatenation, channels = in_channels + skip_channels
        concat_channels = in_channels + skip_channels

        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_p)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout2 = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map from deeper layer (B, in_channels, H, W)
            skip: Skip connection from encoder (B, skip_channels, 2H, 2W)

        Returns:
            Upsampled and refined feature map (B, out_channels, 2H, 2W)
        """
        x = self.upsample(x)

        # Handle potential size mismatch due to odd input dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.dropout1(self.act1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.act2(self.bn2(self.conv2(x))))
        return x


class RangeNetDecoder(nn.Module):
    """U-Net style decoder for RangeNet++.

    Takes 5 encoder feature maps and progressively upsamples while
    concatenating skip connections from the encoder.

    Architecture:
        stage5 (1024) -> upsample + concat stage4 (512) -> 512
        512 -> upsample + concat stage3 (256) -> 256
        256 -> upsample + concat stage2 (128) -> 128
        128 -> upsample + concat stage1 (64) -> 64
        64 -> upsample -> 32 -> final conv -> num_classes
    """

    def __init__(
        self,
        encoder_channels: List[int] = None,
        num_classes: int = 20,
        dropout_p: float = 0.01,
    ):
        """
        Args:
            encoder_channels: Channel counts from encoder stages [64, 128, 256, 512, 1024].
            num_classes: Number of semantic classes.
            dropout_p: Dropout probability.
        """
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512, 1024]

        # Decoder blocks: from deepest to shallowest
        # Block 1: 1024 + skip(512) -> 512
        self.dec4 = DecoderBlock(
            in_channels=encoder_channels[4],
            skip_channels=encoder_channels[3],
            out_channels=encoder_channels[3],
            dropout_p=dropout_p,
        )
        # Block 2: 512 + skip(256) -> 256
        self.dec3 = DecoderBlock(
            in_channels=encoder_channels[3],
            skip_channels=encoder_channels[2],
            out_channels=encoder_channels[2],
            dropout_p=dropout_p,
        )
        # Block 3: 256 + skip(128) -> 128
        self.dec2 = DecoderBlock(
            in_channels=encoder_channels[2],
            skip_channels=encoder_channels[1],
            out_channels=encoder_channels[1],
            dropout_p=dropout_p,
        )
        # Block 4: 128 + skip(64) -> 64
        self.dec1 = DecoderBlock(
            in_channels=encoder_channels[1],
            skip_channels=encoder_channels[0],
            out_channels=encoder_channels[0],
            dropout_p=dropout_p,
        )

        # Final upsampling to original resolution (encoder input was downsampled by stage1)
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(32, num_classes, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_features: List[torch.Tensor], input_size: tuple = None) -> torch.Tensor:
        """Forward pass.

        Args:
            encoder_features: List of encoder feature maps [s1, s2, s3, s4, s5].
                s1: (B, 64, H/2, W/2)
                s2: (B, 128, H/4, W/4)
                s3: (B, 256, H/8, W/8)
                s4: (B, 512, H/16, W/16)
                s5: (B, 1024, H/32, W/32)
            input_size: Original input (H, W) for final size matching.

        Returns:
            Logits tensor (B, num_classes, H, W)
        """
        s1, s2, s3, s4, s5 = encoder_features

        # Decoder path
        x = self.dec4(s5, s4)  # (B, 512, H/16, W/16)
        x = self.dec3(x, s3)   # (B, 256, H/8, W/8)
        x = self.dec2(x, s2)   # (B, 128, H/4, W/4)
        x = self.dec1(x, s1)   # (B, 64, H/2, W/2)

        # Final upsampling to full resolution
        x = self.final_upsample(x)  # (B, 64, H, W)
        x = self.final_conv(x)       # (B, num_classes, H, W)

        # Ensure output matches input spatial dimensions
        if input_size is not None and x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)

        return x
