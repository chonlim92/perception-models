"""
Prediction heads for HDMapNet.

Three heads operate on the BEV feature map:
1. SemanticHead: Per-class binary segmentation (divider, boundary, crossing)
2. InstanceHead: Dense embedding vector for discriminative instance grouping
3. DirectionHead: 2D direction vector (dx, dy) at each BEV pixel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SemanticHead(nn.Module):
    """Semantic segmentation head for HD map elements.

    Produces per-class binary segmentation maps for map element classes:
    - Class 0: Divider (lane dividers)
    - Class 1: Boundary (road boundaries)
    - Class 2: Crossing (pedestrian crossings)
    """

    def __init__(self, in_channels, num_classes=3, mid_channels=64):
        """
        Args:
            in_channels: Number of input feature channels.
            num_classes: Number of semantic classes (default 3).
            mid_channels: Intermediate channel count.
        """
        super().__init__()
        self.num_classes = num_classes

        self.head = nn.Sequential(
            ConvBlock(in_channels, mid_channels, kernel_size=3, padding=1),
            ConvBlock(mid_channels, mid_channels, kernel_size=3, padding=1),
            ConvBlock(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: BEV features (B, C, H, W).

        Returns:
            Semantic logits (B, num_classes, H, W). Apply sigmoid for probabilities.
        """
        return self.head(x)


class InstanceHead(nn.Module):
    """Instance embedding head for discriminative loss-based instance grouping.

    Produces a dense embedding vector at each BEV pixel. Pixels belonging to
    the same instance are pulled together, while different instances are pushed apart.
    """

    def __init__(self, in_channels, embedding_dim=16, mid_channels=64):
        """
        Args:
            in_channels: Number of input feature channels.
            embedding_dim: Dimensionality of the embedding vector.
            mid_channels: Intermediate channel count.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        self.head = nn.Sequential(
            ConvBlock(in_channels, mid_channels, kernel_size=3, padding=1),
            ConvBlock(mid_channels, mid_channels, kernel_size=3, padding=1),
            ConvBlock(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels, embedding_dim, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: BEV features (B, C, H, W).

        Returns:
            Instance embeddings (B, embedding_dim, H, W).
        """
        return self.head(x)


class DirectionHead(nn.Module):
    """Direction prediction head.

    Predicts a 2D direction vector (dx, dy) at each BEV pixel, indicating
    the local tangent direction of map elements (useful for polyline ordering).
    """

    def __init__(self, in_channels, mid_channels=64):
        """
        Args:
            in_channels: Number of input feature channels.
            mid_channels: Intermediate channel count.
        """
        super().__init__()

        self.head = nn.Sequential(
            ConvBlock(in_channels, mid_channels, kernel_size=3, padding=1),
            ConvBlock(mid_channels, mid_channels, kernel_size=3, padding=1),
            ConvBlock(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 2, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: BEV features (B, C, H, W).

        Returns:
            Direction vectors (B, 2, H, W) where channel 0 is dx and channel 1 is dy.
        """
        direction = self.head(x)
        # Normalize to unit vectors
        norm = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
        direction = direction / norm
        return direction
