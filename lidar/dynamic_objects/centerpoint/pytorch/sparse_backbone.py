"""
CenterPoint 3D Sparse Convolution Backbone.

Implements a 3D sparse CNN backbone using gather/scatter-based sparse convolution.
This implementation uses dense tensors with masking as a portable fallback when
the spconv library is not available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


class SparseTensor:
    """Lightweight sparse tensor representation for 3D feature volumes.

    Stores only occupied voxel features and their coordinates, avoiding
    the memory cost of dense 3D tensors.

    Attributes:
        features: (N, C) tensor of voxel features.
        indices: (N, 4) tensor of (batch_idx, z, y, x) integer coordinates.
        spatial_shape: (D, H, W) spatial dimensions of the volume.
        batch_size: Number of samples in the batch.
    """

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: Tuple[int, int, int],
        batch_size: int,
    ):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    @property
    def device(self):
        return self.features.device

    @property
    def dtype(self):
        return self.features.dtype


def build_kernel_offsets(kernel_size: int, ndim: int = 3) -> torch.Tensor:
    """Generate all kernel offset positions for a 3D convolution kernel.

    Args:
        kernel_size: Size of the cubic kernel.
        ndim: Number of spatial dimensions (default 3).

    Returns:
        offsets: (kernel_size^ndim, ndim) tensor of offset vectors.
    """
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1)
    if ndim == 3:
        grid = torch.meshgrid(coords, coords, coords, indexing='ij')
    else:
        grid = torch.meshgrid(*[coords] * ndim, indexing='ij')
    offsets = torch.stack([g.reshape(-1) for g in grid], dim=1)
    return offsets


def build_hash_table(
    indices: torch.Tensor, spatial_shape: Tuple[int, int, int]
) -> Dict[str, torch.Tensor]:
    """Build a hash table mapping spatial coordinates to feature indices.

    Uses linearized coordinates as keys for O(1) lookup.

    Args:
        indices: (N, 4) integer coordinates (batch, z, y, x).
        spatial_shape: (D, H, W) spatial dimensions.

    Returns:
        Dictionary with 'linear_indices' and 'hash_table' tensors.
    """
    D, H, W = spatial_shape
    batch_idx = indices[:, 0]
    z, y, x = indices[:, 1], indices[:, 2], indices[:, 3]

    # Linearize: batch * D*H*W + z * H*W + y * W + x
    linear = batch_idx * (D * H * W) + z * (H * W) + y * W + x
    max_linear = indices[:, 0].max().item() * D * H * W + D * H * W

    # Create hash table: linear_coord -> index in features
    hash_table = torch.full((int(max_linear) + 1,), -1, dtype=torch.long, device=indices.device)
    feature_indices = torch.arange(indices.shape[0], device=indices.device)
    hash_table[linear] = feature_indices

    return {'linear_indices': linear, 'hash_table': hash_table, 'max_val': int(max_linear)}


class SparseConv3d(nn.Module):
    """3D Sparse Convolution using gather-scatter operations.

    Computes convolution only at output locations where the strided grid
    has valid positions. For stride > 1, output locations are the strided
    subset of input locations.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel (cubic).
        stride: Convolution stride.
        padding: Padding (applied conceptually to the sparse volume).
        bias: Whether to include bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        num_kernel_positions = kernel_size ** 3
        self.weight = nn.Parameter(
            torch.empty(num_kernel_positions, in_channels, out_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # Initialize weights
        nn.init.kaiming_normal_(self.weight.view(-1, out_channels), mode='fan_out')

        # Register kernel offsets
        offsets = build_kernel_offsets(kernel_size, ndim=3)
        self.register_buffer('kernel_offsets', offsets)

    def forward(self, sparse_input: SparseTensor) -> SparseTensor:
        """
        Args:
            sparse_input: SparseTensor with features and indices.

        Returns:
            SparseTensor with convolved features at output locations.
        """
        features = sparse_input.features  # (N, C_in)
        indices = sparse_input.indices  # (N, 4) [batch, z, y, x]
        spatial_shape = sparse_input.spatial_shape
        batch_size = sparse_input.batch_size
        device = features.device

        D, H, W = spatial_shape

        # Compute output spatial shape
        D_out = (D + 2 * self.padding - self.kernel_size) // self.stride + 1
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_spatial_shape = (D_out, H_out, W_out)

        # Determine output locations
        if self.stride == 1:
            # Output at same locations as input
            out_indices = indices.clone()
        else:
            # Output at strided positions that have at least one input neighbor
            out_z = indices[:, 1] // self.stride
            out_y = indices[:, 2] // self.stride
            out_x = indices[:, 3] // self.stride

            out_coords = torch.stack([indices[:, 0], out_z, out_y, out_x], dim=1)

            # Keep unique output positions
            # Linearize for uniqueness
            linear_out = (
                out_coords[:, 0] * (D_out * H_out * W_out) +
                out_coords[:, 1] * (H_out * W_out) +
                out_coords[:, 2] * W_out +
                out_coords[:, 3]
            )
            unique_linear, inverse_map = torch.unique(linear_out, return_inverse=True)

            # Recover 3D coords from linear
            batch_out = unique_linear // (D_out * H_out * W_out)
            remainder = unique_linear % (D_out * H_out * W_out)
            z_out = remainder // (H_out * W_out)
            remainder2 = remainder % (H_out * W_out)
            y_out = remainder2 // W_out
            x_out = remainder2 % W_out

            out_indices = torch.stack([batch_out, z_out, y_out, x_out], dim=1).int()

        num_out = out_indices.shape[0]

        # Build input hash table for neighbor lookup
        input_linear = (
            indices[:, 0].long() * (D * H * W) +
            indices[:, 1].long() * (H * W) +
            indices[:, 2].long() * W +
            indices[:, 3].long()
        )
        max_linear_val = batch_size * D * H * W
        hash_table = torch.full((max_linear_val,), -1, dtype=torch.long, device=device)
        hash_table[input_linear] = torch.arange(features.shape[0], device=device)

        # For each output location, gather neighbor features and apply kernel weights
        out_features = torch.zeros(num_out, self.out_channels, dtype=features.dtype, device=device)

        kernel_offsets = self.kernel_offsets.to(device)  # (K^3, 3)

        for k_idx in range(kernel_offsets.shape[0]):
            offset = kernel_offsets[k_idx]  # (3,) [dz, dy, dx]

            # Compute input neighbor coordinates for each output position
            if self.stride == 1:
                neighbor_z = out_indices[:, 1].long() + offset[0].long()
                neighbor_y = out_indices[:, 2].long() + offset[1].long()
                neighbor_x = out_indices[:, 3].long() + offset[2].long()
            else:
                # For strided conv: output position maps to input at stride*out + offset
                neighbor_z = out_indices[:, 1].long() * self.stride + offset[0].long() + self.padding
                neighbor_y = out_indices[:, 2].long() * self.stride + offset[1].long() + self.padding
                neighbor_x = out_indices[:, 3].long() * self.stride + offset[2].long() + self.padding

            neighbor_batch = out_indices[:, 0].long()

            # Check bounds
            valid = (
                (neighbor_z >= 0) & (neighbor_z < D) &
                (neighbor_y >= 0) & (neighbor_y < H) &
                (neighbor_x >= 0) & (neighbor_x < W)
            )

            if not valid.any():
                continue

            # Linearize neighbor coords
            neighbor_linear = (
                neighbor_batch * (D * H * W) +
                neighbor_z * (H * W) +
                neighbor_y * W +
                neighbor_x
            )

            # Clamp for safe indexing
            neighbor_linear = neighbor_linear.clamp(0, max_linear_val - 1)

            # Look up in hash table
            neighbor_feat_idx = hash_table[neighbor_linear]  # (num_out,)

            # Valid if in bounds AND found in hash table
            found = (neighbor_feat_idx >= 0) & valid

            if not found.any():
                continue

            # Gather features from found neighbors
            valid_indices = torch.where(found)[0]
            gathered_features = features[neighbor_feat_idx[valid_indices]]  # (num_valid, C_in)

            # Apply kernel weight for this position
            weight_k = self.weight[k_idx]  # (C_in, C_out)
            contribution = gathered_features @ weight_k  # (num_valid, C_out)

            out_features[valid_indices] += contribution

        if self.bias is not None:
            out_features = out_features + self.bias.unsqueeze(0)

        return SparseTensor(
            features=out_features,
            indices=out_indices,
            spatial_shape=out_spatial_shape,
            batch_size=batch_size,
        )


class SubmanifoldSparseConv3d(nn.Module):
    """Submanifold Sparse Convolution: output only at locations where input exists.

    This ensures the sparsity pattern is preserved (no dilation of active sites).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size (cubic).
        bias: Whether to use bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        num_kernel_positions = kernel_size ** 3
        self.weight = nn.Parameter(
            torch.empty(num_kernel_positions, in_channels, out_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_normal_(self.weight.view(-1, out_channels), mode='fan_out')

        offsets = build_kernel_offsets(kernel_size, ndim=3)
        self.register_buffer('kernel_offsets', offsets)

    def forward(self, sparse_input: SparseTensor) -> SparseTensor:
        """
        Args:
            sparse_input: Input SparseTensor.

        Returns:
            SparseTensor with same indices but transformed features.
        """
        features = sparse_input.features  # (N, C_in)
        indices = sparse_input.indices  # (N, 4)
        spatial_shape = sparse_input.spatial_shape
        batch_size = sparse_input.batch_size
        device = features.device

        D, H, W = spatial_shape
        N = features.shape[0]

        # Build hash table for input lookup
        input_linear = (
            indices[:, 0].long() * (D * H * W) +
            indices[:, 1].long() * (H * W) +
            indices[:, 2].long() * W +
            indices[:, 3].long()
        )
        max_linear_val = batch_size * D * H * W
        hash_table = torch.full((max_linear_val,), -1, dtype=torch.long, device=device)
        hash_table[input_linear] = torch.arange(N, device=device)

        # Output at same locations as input (submanifold property)
        out_features = torch.zeros(N, self.out_channels, dtype=features.dtype, device=device)

        kernel_offsets = self.kernel_offsets.to(device)

        for k_idx in range(kernel_offsets.shape[0]):
            offset = kernel_offsets[k_idx]

            # Neighbor coordinates
            neighbor_z = indices[:, 1].long() + offset[0].long()
            neighbor_y = indices[:, 2].long() + offset[1].long()
            neighbor_x = indices[:, 3].long() + offset[2].long()
            neighbor_batch = indices[:, 0].long()

            # Bounds check
            valid = (
                (neighbor_z >= 0) & (neighbor_z < D) &
                (neighbor_y >= 0) & (neighbor_y < H) &
                (neighbor_x >= 0) & (neighbor_x < W)
            )

            if not valid.any():
                continue

            neighbor_linear = (
                neighbor_batch * (D * H * W) +
                neighbor_z * (H * W) +
                neighbor_y * W +
                neighbor_x
            )
            neighbor_linear = neighbor_linear.clamp(0, max_linear_val - 1)

            neighbor_feat_idx = hash_table[neighbor_linear]
            found = (neighbor_feat_idx >= 0) & valid

            if not found.any():
                continue

            valid_out_indices = torch.where(found)[0]
            gathered_features = features[neighbor_feat_idx[valid_out_indices]]

            weight_k = self.weight[k_idx]  # (C_in, C_out)
            contribution = gathered_features @ weight_k

            out_features[valid_out_indices] += contribution

        if self.bias is not None:
            out_features = out_features + self.bias.unsqueeze(0)

        return SparseTensor(
            features=out_features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )


class SparseBatchNorm(nn.Module):
    """Batch normalization for SparseTensor features."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, sparse_input: SparseTensor) -> SparseTensor:
        features = self.bn(sparse_input.features)
        return SparseTensor(
            features=features,
            indices=sparse_input.indices,
            spatial_shape=sparse_input.spatial_shape,
            batch_size=sparse_input.batch_size,
        )


class SparseReLU(nn.Module):
    """ReLU activation for SparseTensor features."""

    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, sparse_input: SparseTensor) -> SparseTensor:
        features = F.relu(sparse_input.features, inplace=self.inplace)
        return SparseTensor(
            features=features,
            indices=sparse_input.indices,
            spatial_shape=sparse_input.spatial_shape,
            batch_size=sparse_input.batch_size,
        )


class SparseBasicBlock(nn.Module):
    """Sparse residual block: SubMConv3d -> BN -> ReLU -> SubMConv3d -> BN + residual.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = SubmanifoldSparseConv3d(in_channels, out_channels, kernel_size)
        self.bn1 = SparseBatchNorm(out_channels)
        self.relu1 = SparseReLU(inplace=True)
        self.conv2 = SubmanifoldSparseConv3d(out_channels, out_channels, kernel_size)
        self.bn2 = SparseBatchNorm(out_channels)
        self.relu2 = SparseReLU(inplace=True)

        # If channel dimensions change, need a projection
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                SubmanifoldSparseConv3d(in_channels, out_channels, kernel_size=1),
                SparseBatchNorm(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, sparse_input: SparseTensor) -> SparseTensor:
        identity = sparse_input

        out = self.conv1(sparse_input)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        if self.downsample is not None:
            for layer in self.downsample:
                identity = layer(identity)

        out = SparseTensor(
            features=out.features + identity.features,
            indices=out.indices,
            spatial_shape=out.spatial_shape,
            batch_size=out.batch_size,
        )
        out = self.relu2(out)
        return out


class SparseCNNBackbone(nn.Module):
    """3D Sparse CNN Backbone with 4 stages.

    Architecture:
        - Initial submanifold conv to set channel count
        - Stage 1: SparseBasicBlock(s) at channels[0], then strided SparseConv3d (stride=2)
        - Stage 2: SparseBasicBlock(s) at channels[1], then strided SparseConv3d (stride=2)
        - Stage 3: SparseBasicBlock(s) at channels[2], then strided SparseConv3d (stride=2)
        - Stage 4: SparseBasicBlock(s) at channels[3]

    Output: Sparse features at stride 8 (after 3 downsampling stages).

    Args:
        in_channels: Number of input voxel feature channels.
        channels: Channel dimensions for each stage [16, 32, 64, 128].
        num_blocks: Number of SparseBasicBlocks per stage.
        spatial_shape: Input spatial shape (D, H, W) of the voxel grid.
    """

    def __init__(
        self,
        in_channels: int = 4,
        channels: List[int] = [16, 32, 64, 128],
        num_blocks: List[int] = [2, 2, 2, 2],
        spatial_shape: Tuple[int, int, int] = (40, 1440, 1440),
    ):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.channels = channels

        # Initial convolution to map input features to first channel dim
        self.input_conv = SubmanifoldSparseConv3d(in_channels, channels[0], kernel_size=3)
        self.input_bn = SparseBatchNorm(channels[0])
        self.input_relu = SparseReLU(inplace=True)

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev_channels = channels[0]
        for stage_idx, (ch, n_blocks) in enumerate(zip(channels, num_blocks)):
            # Build blocks for this stage
            blocks = nn.ModuleList()
            for block_idx in range(n_blocks):
                block_in_ch = prev_channels if block_idx == 0 else ch
                blocks.append(SparseBasicBlock(block_in_ch, ch, kernel_size=3))
            self.stages.append(blocks)
            prev_channels = ch

            # Add strided convolution for downsampling (except last stage)
            if stage_idx < len(channels) - 1:
                self.downsamples.append(
                    SparseConv3d(ch, channels[stage_idx + 1], kernel_size=3, stride=2, padding=1)
                )
                self.downsamples.append(SparseBatchNorm(channels[stage_idx + 1]))
                self.downsamples.append(SparseReLU(inplace=True))

        self.out_channels = channels[-1]

    def forward(self, sparse_input: SparseTensor) -> SparseTensor:
        """
        Args:
            sparse_input: SparseTensor from voxelization (features at voxel locations).

        Returns:
            SparseTensor with features at stride-8 resolution.
        """
        # Initial conv
        x = self.input_conv(sparse_input)
        x = self.input_bn(x)
        x = self.input_relu(x)

        # Process stages
        downsample_idx = 0
        for stage_idx, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x)

            # Apply strided downsampling (except for last stage)
            if stage_idx < len(self.channels) - 1:
                x = self.downsamples[downsample_idx * 3](x)      # SparseConv3d
                x = self.downsamples[downsample_idx * 3 + 1](x)  # BN
                x = self.downsamples[downsample_idx * 3 + 2](x)  # ReLU
                downsample_idx += 1

        return x

    @staticmethod
    def from_voxels(
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        spatial_shape: Tuple[int, int, int],
        batch_size: int,
    ) -> SparseTensor:
        """Create a SparseTensor from voxelization output.

        Args:
            voxel_features: (M, C) voxel features.
            voxel_coords: (M, 4) coords with batch index (batch, z, y, x).
            spatial_shape: (D, H, W) spatial dimensions.
            batch_size: Number of samples.

        Returns:
            SparseTensor ready for backbone processing.
        """
        return SparseTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
