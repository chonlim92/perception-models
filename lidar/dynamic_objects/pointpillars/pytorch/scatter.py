"""PointPillars Scatter module for converting sparse pillar features to a dense BEV pseudo-image.

This module implements the scatter operation described in the PointPillars paper
(Lang et al., 2019), which takes encoded pillar features and their corresponding
grid coordinates and produces a dense bird's-eye view (BEV) pseudo-image tensor
suitable for downstream 2D convolutional processing.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class PointPillarsScatter(nn.Module):
    """Scatter pillar features into a dense BEV pseudo-image.

    This module takes sparse pillar features and their grid coordinates, then
    places each pillar's feature vector at the corresponding (x, y) location
    in a dense 2D canvas, producing a pseudo-image of shape (B, C, H, W).

    Args:
        output_shape: Tuple (H, W) specifying the height and width of the
            output BEV canvas in grid cells.
        num_channels: The number of feature channels C per pillar.

    Example:
        >>> scatter = PointPillarsScatter(output_shape=(496, 432), num_channels=64)
        >>> pillar_features = torch.randn(20000, 64)  # P pillars, C channels
        >>> coords = torch.randint(0, 432, (20000, 2))  # P pillar grid indices (row, col)
        >>> coords[:, 0] = torch.randint(0, 496, (20000,))
        >>> output = scatter(pillar_features, coords, batch_size=1)
        >>> output.shape
        torch.Size([1, 64, 496, 432])
    """

    def __init__(self, output_shape: Tuple[int, int], num_channels: int) -> None:
        """Initialize PointPillarsScatter.

        Args:
            output_shape: Tuple of (H, W) for the spatial dimensions of the
                output BEV pseudo-image.
            num_channels: Integer specifying the number of feature channels C.
        """
        super().__init__()
        self.output_shape = output_shape
        self.num_channels = num_channels
        self.height: int = output_shape[0]
        self.width: int = output_shape[1]

    def forward(
        self,
        pillar_features: Tensor,
        coords: Tensor,
        batch_size: int,
    ) -> Tensor:
        """Scatter sparse pillar features onto a dense BEV canvas.

        Takes pillar features and their grid coordinates and produces a dense
        pseudo-image by placing each pillar's feature vector at the appropriate
        spatial location. Unoccupied locations remain zero-filled.

        Args:
            pillar_features: Tensor of shape (P, C) containing the encoded
                feature vector for each pillar, where P is the total number of
                non-empty pillars across all samples in the batch and C is the
                feature dimension (must equal self.num_channels).
            coords: Tensor of shape (P, 3) containing batch index and grid
                coordinates for each pillar. Column 0 is the batch index,
                column 1 is the row index (y), and column 2 is the column
                index (x). Alternatively, if coords has shape (P, 2), it is
                interpreted as (row, col) and all pillars are assumed to belong
                to a single sample (batch index 0).
            batch_size: The number of samples in the current batch.

        Returns:
            Dense BEV pseudo-image tensor of shape (B, C, H, W) where B is
            batch_size, C is num_channels, H is the canvas height, and W is
            the canvas width.

        Raises:
            ValueError: If pillar_features channel dimension does not match
                num_channels.
        """
        if pillar_features.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected pillar features with {self.num_channels} channels, "
                f"but got {pillar_features.shape[1]} channels."
            )

        # Create the dense canvas initialized to zeros
        # Shape: (B, C, H, W)
        canvas = torch.zeros(
            batch_size,
            self.num_channels,
            self.height,
            self.width,
            dtype=pillar_features.dtype,
            device=pillar_features.device,
        )

        # Handle the case where there are no pillars
        if pillar_features.shape[0] == 0:
            return canvas

        # Parse coordinates based on their shape
        if coords.shape[1] == 2:
            # coords is (P, 2) with (row, col) — no batch index provided
            # Assign all pillars to their respective batch samples by splitting
            # evenly, or assume single batch if batch_size == 1
            if batch_size == 1:
                batch_indices = torch.zeros(
                    coords.shape[0], dtype=torch.long, device=coords.device
                )
                row_indices = coords[:, 0].long()
                col_indices = coords[:, 1].long()
            else:
                # When coords lack batch index and batch_size > 1, we assume
                # pillars are evenly distributed across batches in order
                num_pillars = coords.shape[0]
                pillars_per_sample = num_pillars // batch_size
                batch_indices = torch.arange(
                    batch_size, device=coords.device
                ).repeat_interleave(pillars_per_sample)
                # Handle remainder pillars (assign to last batch element)
                remainder = num_pillars - pillars_per_sample * batch_size
                if remainder > 0:
                    last_batch = torch.full(
                        (remainder,), batch_size - 1, dtype=torch.long, device=coords.device
                    )
                    batch_indices = torch.cat([batch_indices, last_batch], dim=0)
                row_indices = coords[:, 0].long()
                col_indices = coords[:, 1].long()
        elif coords.shape[1] >= 3:
            # coords is (P, 3+) with (batch_idx, row, col)
            batch_indices = coords[:, 0].long()
            row_indices = coords[:, 1].long()
            col_indices = coords[:, 2].long()
        else:
            raise ValueError(
                f"coords must have at least 2 columns, got shape {coords.shape}"
            )

        # Clamp indices to valid range to prevent out-of-bounds access
        row_indices = row_indices.clamp(0, self.height - 1)
        col_indices = col_indices.clamp(0, self.width - 1)
        batch_indices = batch_indices.clamp(0, batch_size - 1)

        # Scatter pillar features to the canvas using advanced indexing
        # pillar_features shape: (P, C) -> we need to assign each pillar's
        # C-dimensional feature vector to canvas[batch, :, row, col]
        canvas[batch_indices, :, row_indices, col_indices] = pillar_features

        return canvas

    def extra_repr(self) -> str:
        """Return a string representation of module parameters."""
        return (
            f"num_channels={self.num_channels}, "
            f"output_shape=({self.height}, {self.width})"
        )
