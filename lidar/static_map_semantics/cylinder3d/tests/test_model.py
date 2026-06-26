#!/usr/bin/env python3
"""
test_model.py - Unit tests for Cylinder3D model components.

Tests cover:
  - Cylindrical partition (coordinate transform and voxelization)
  - Asymmetric convolution block
  - Dimension-decomposed convolution module (DDCMod)
  - Backbone forward pass
  - Point-level refinement MLP
  - Full model end-to-end forward pass
  - Lovasz-Softmax loss function
  - Dataset loading

Run with:
  pytest tests/test_model.py -v
"""

import os
import struct
import tempfile
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_point_cloud():
    """Generate a random point cloud (N=1000, 4 channels: x, y, z, intensity)."""
    rng = np.random.default_rng(42)
    # Simulate LiDAR-like distribution
    n_points = 1000
    # Random angles and distances
    theta = rng.uniform(-np.pi, np.pi, n_points)
    rho = rng.uniform(1.0, 50.0, n_points)
    z = rng.uniform(-3.0, 5.0, n_points)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    intensity = rng.uniform(0.0, 1.0, n_points)
    points = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
    return points


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def num_classes():
    """Number of semantic classes (SemanticKITTI: 20)."""
    return 20


@pytest.fixture
def grid_size():
    """Cylindrical grid dimensions [D_rho, D_theta, D_z]."""
    return [480, 360, 32]


@pytest.fixture
def small_grid_size():
    """Small grid for fast testing."""
    return [16, 16, 8]


# =============================================================================
# Helper Modules (Minimal implementations for testing)
# =============================================================================


class CylindricalPartition(nn.Module):
    """Convert Cartesian point cloud to cylindrical coordinates and voxelize."""

    def __init__(self, grid_size, max_bound, min_bound):
        super().__init__()
        self.grid_size = np.array(grid_size)
        self.max_bound = np.array(max_bound)
        self.min_bound = np.array(min_bound)
        self.intervals = (self.max_bound - self.min_bound) / self.grid_size

    def cart2cyl(self, points_xyz):
        """Convert Cartesian (x, y, z) to cylindrical (rho, theta, z)."""
        x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
        rho = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)  # [-pi, pi]
        return torch.stack([rho, theta, z], dim=-1)

    def get_grid_indices(self, cyl_coords):
        """Convert cylindrical coordinates to grid indices."""
        # Clamp to bounds
        cyl_np = cyl_coords.detach().cpu().numpy()
        clamped = np.clip(cyl_np, self.min_bound, self.max_bound - 1e-6)
        # Compute indices
        indices = ((clamped - self.min_bound) / self.intervals).astype(np.int32)
        # Clamp indices to valid range
        for dim in range(3):
            indices[:, dim] = np.clip(indices[:, dim], 0, self.grid_size[dim] - 1)
        return indices

    def forward(self, points):
        """
        Args:
            points: (N, 4) tensor [x, y, z, intensity]

        Returns:
            cyl_coords: (N, 3) cylindrical coordinates
            grid_indices: (N, 3) voxel grid indices
            voxel_features: (D, H, W, C) sparse voxel feature grid
        """
        xyz = points[:, :3]
        cyl_coords = self.cart2cyl(xyz)
        grid_indices = self.get_grid_indices(cyl_coords)

        # Create sparse voxel grid (simple mean pooling)
        feat_dim = points.shape[1]
        voxel_grid = np.zeros(
            (*self.grid_size, feat_dim), dtype=np.float32
        )
        voxel_count = np.zeros(self.grid_size, dtype=np.int32)

        points_np = points.detach().cpu().numpy()
        for i in range(len(points)):
            idx = tuple(grid_indices[i])
            voxel_grid[idx] += points_np[i]
            voxel_count[idx] += 1

        # Average features in each voxel
        mask = voxel_count > 0
        voxel_grid[mask] /= voxel_count[mask][..., None]

        return cyl_coords, grid_indices, torch.from_numpy(voxel_grid)


class AsymmetricConvBlock(nn.Module):
    """Asymmetric 3D convolution block (decomposes 3x3x3 into asymmetric kernels)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, D, H, W) tensor

        Returns:
            (B, C_out, D, H, W) tensor
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DDCMod(nn.Module):
    """Dimension-Decomposed Convolution Module.

    Decomposes 3D convolution into three 1D convolutions along each axis,
    producing more efficient feature extraction in cylindrical space.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 3

        # Decomposed convolutions along each dimension
        self.conv_d = nn.Conv3d(in_channels, mid_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_h = nn.Conv3d(in_channels, mid_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.conv_w = nn.Conv3d(in_channels, out_channels - 2 * mid_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1))

        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, D, H, W)

        Returns:
            (B, C_out, D, H, W)
        """
        out_d = self.conv_d(x)
        out_h = self.conv_h(x)
        out_w = self.conv_w(x)
        out = torch.cat([out_d, out_h, out_w], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Cylinder3DBackbone(nn.Module):
    """Simplified Cylinder3D backbone for testing."""

    def __init__(self, in_channels, num_classes, grid_size):
        super().__init__()
        self.grid_size = grid_size

        # Encoder
        self.enc1 = AsymmetricConvBlock(in_channels, 32)
        self.enc2 = AsymmetricConvBlock(32, 64)
        self.ddc = DDCMod(64, 64)

        # Decoder
        self.dec1 = AsymmetricConvBlock(64, 32)
        self.output_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) voxel features

        Returns:
            (B, num_classes, D, H, W) per-voxel class scores
        """
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.ddc(x)
        x = self.dec1(x)
        x = self.output_conv(x)
        return x


class PointRefinementMLP(nn.Module):
    """Point-wise MLP for refining voxel predictions at the point level."""

    def __init__(self, in_features, num_classes, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, point_features):
        """
        Args:
            point_features: (N, in_features) per-point features

        Returns:
            (N, num_classes) per-point class logits
        """
        return self.mlp(point_features)


class Cylinder3DModel(nn.Module):
    """Full Cylinder3D model (simplified for testing)."""

    def __init__(self, num_classes=20, grid_size=(16, 16, 8), in_channels=4):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size

        self.backbone = Cylinder3DBackbone(in_channels, num_classes, grid_size)
        self.point_refine = PointRefinementMLP(
            in_features=num_classes + in_channels,
            num_classes=num_classes,
        )

    def forward(self, voxel_features, point_features, grid_indices):
        """
        Args:
            voxel_features: (B, C, D, H, W) voxel input
            point_features: (N, 4) raw point features
            grid_indices: (N, 3) grid indices for each point

        Returns:
            dict with 'voxel_logits' and 'point_logits'
        """
        # Backbone processes voxel grid
        voxel_logits = self.backbone(voxel_features)

        # Gather voxel predictions for each point
        B = voxel_features.shape[0]
        # For simplicity, assume single batch
        voxel_preds_per_point = voxel_logits[
            0, :,
            grid_indices[:, 0],
            grid_indices[:, 1],
            grid_indices[:, 2],
        ].T  # (N, num_classes)

        # Concatenate with raw point features for refinement
        refine_input = torch.cat([voxel_preds_per_point, point_features], dim=1)
        point_logits = self.point_refine(refine_input)

        return {
            "voxel_logits": voxel_logits,
            "point_logits": point_logits,
        }


# =============================================================================
# Lovasz-Softmax Loss (simplified for testing)
# =============================================================================


def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovasz-Softmax loss (flat version).

    Args:
        probas: (N, C) class probabilities at each prediction
        labels: (N,) ground truth labels
        classes: 'all', 'present', or list of class indices

    Returns:
        Lovasz-Softmax loss value
    """
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.size(1)
    losses = []

    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes

    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == "present" and fg.sum() == 0) or fg.sum() == 0:
            continue
        if C == 1:
            fg_class = 1.0 - probas[:, 0]
        else:
            fg_class = 1.0 - probas[:, c]

        errors = (fg - fg_class).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=probas.device)

    return torch.stack(losses).mean()


def lovasz_softmax_loss(logits, labels, classes="present", ignore_index=0):
    """Lovasz-Softmax loss for semantic segmentation.

    Args:
        logits: (N, C) raw logits
        labels: (N,) ground truth labels
        classes: which classes to include
        ignore_index: class index to ignore

    Returns:
        Loss scalar
    """
    probas = F.softmax(logits, dim=1)

    # Filter ignore index
    valid = labels != ignore_index
    if valid.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    vprobas = probas[valid]
    vlabels = labels[valid]

    return lovasz_softmax_flat(vprobas, vlabels, classes=classes)


# =============================================================================
# Tests
# =============================================================================


class TestCylindricalPartition:
    """Tests for cylindrical coordinate transform and voxelization."""

    def test_coordinate_transform(self, random_point_cloud):
        """Verify correct Cartesian to cylindrical coordinate conversion."""
        points = torch.from_numpy(random_point_cloud)
        partition = CylindricalPartition(
            grid_size=[16, 16, 8],
            max_bound=[50.0, np.pi, 5.0],
            min_bound=[0.0, -np.pi, -3.0],
        )

        cyl_coords = partition.cart2cyl(points[:, :3])

        # rho should be non-negative
        assert (cyl_coords[:, 0] >= 0).all(), "rho must be non-negative"

        # theta should be in [-pi, pi]
        assert (cyl_coords[:, 1] >= -np.pi).all(), "theta must be >= -pi"
        assert (cyl_coords[:, 1] <= np.pi).all(), "theta must be <= pi"

        # z should match input z
        assert torch.allclose(
            cyl_coords[:, 2], points[:, 2], atol=1e-6
        ), "z coordinate should be preserved"

        # Verify rho = sqrt(x^2 + y^2)
        expected_rho = torch.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        assert torch.allclose(
            cyl_coords[:, 0], expected_rho, atol=1e-6
        ), "rho should equal sqrt(x^2 + y^2)"

    def test_grid_indices_within_bounds(self, random_point_cloud):
        """Verify grid indices are within valid bounds."""
        grid_size = [16, 16, 8]
        points = torch.from_numpy(random_point_cloud)
        partition = CylindricalPartition(
            grid_size=grid_size,
            max_bound=[50.0, np.pi, 5.0],
            min_bound=[0.0, -np.pi, -3.0],
        )

        cyl_coords = partition.cart2cyl(points[:, :3])
        grid_indices = partition.get_grid_indices(cyl_coords)

        # All indices should be within [0, grid_size-1]
        for dim in range(3):
            assert (grid_indices[:, dim] >= 0).all(), (
                f"Grid index dim {dim} has negative values"
            )
            assert (grid_indices[:, dim] < grid_size[dim]).all(), (
                f"Grid index dim {dim} exceeds bound {grid_size[dim]}"
            )

    def test_voxel_feature_shapes(self, random_point_cloud):
        """Verify voxelized features have correct shape."""
        grid_size = [16, 16, 8]
        points = torch.from_numpy(random_point_cloud)
        partition = CylindricalPartition(
            grid_size=grid_size,
            max_bound=[50.0, np.pi, 5.0],
            min_bound=[0.0, -np.pi, -3.0],
        )

        cyl_coords, grid_indices, voxel_features = partition(points)

        # Check shapes
        assert cyl_coords.shape == (len(points), 3)
        assert grid_indices.shape == (len(points), 3)
        assert voxel_features.shape == (16, 16, 8, 4), (
            f"Expected (16,16,8,4) got {voxel_features.shape}"
        )

    def test_voxel_features_dtype(self, random_point_cloud):
        """Verify voxel features are float32."""
        points = torch.from_numpy(random_point_cloud)
        partition = CylindricalPartition(
            grid_size=[8, 8, 4],
            max_bound=[50.0, np.pi, 5.0],
            min_bound=[0.0, -np.pi, -3.0],
        )

        _, _, voxel_features = partition(points)
        assert voxel_features.dtype == torch.float32


class TestAsymmetricConvBlock:
    """Tests for asymmetric 3D convolution block."""

    def test_output_shape_matches_spatial_dims(self, device):
        """Verify output spatial dimensions match input."""
        in_channels, out_channels = 32, 64
        block = AsymmetricConvBlock(in_channels, out_channels).to(device)

        D, H, W = 16, 16, 8
        x = torch.randn(2, in_channels, D, H, W, device=device)
        out = block(x)

        assert out.shape == (2, out_channels, D, H, W), (
            f"Expected shape (2, {out_channels}, {D}, {H}, {W}), got {out.shape}"
        )

    def test_parameter_count_reasonable(self):
        """Verify parameter count is within expected range."""
        block = AsymmetricConvBlock(32, 64)
        num_params = sum(p.numel() for p in block.parameters())

        # Conv3d(32,64,(3,1,1)): 32*64*3 + 64 = 6208
        # Conv3d(64,64,(1,3,1)): 64*64*3 + 64 = 12352
        # Conv3d(64,64,(1,1,3)): 64*64*3 + 64 = 12352
        # BN(64): 128
        # Total ~ 31040
        assert 10000 < num_params < 100000, (
            f"Parameter count {num_params} seems unreasonable"
        )

    def test_different_spatial_sizes(self, device):
        """Test with various spatial dimensions."""
        block = AsymmetricConvBlock(16, 32).to(device)

        for D, H, W in [(8, 8, 4), (32, 32, 16), (4, 12, 6)]:
            x = torch.randn(1, 16, D, H, W, device=device)
            out = block(x)
            assert out.shape == (1, 32, D, H, W)


class TestDDCMod:
    """Tests for Dimension-Decomposed Convolution Module."""

    def test_output_shape(self, device):
        """Verify DDCMod produces correct output shape."""
        in_ch, out_ch = 64, 96
        ddc = DDCMod(in_ch, out_ch).to(device)

        D, H, W = 16, 16, 8
        x = torch.randn(2, in_ch, D, H, W, device=device)
        out = ddc(x)

        assert out.shape == (2, out_ch, D, H, W), (
            f"Expected (2, {out_ch}, {D}, {H}, {W}), got {out.shape}"
        )

    def test_decomposition_channels(self):
        """Verify channel decomposition adds up correctly."""
        in_ch, out_ch = 32, 96
        ddc = DDCMod(in_ch, out_ch)

        # mid_channels = 96 // 3 = 32
        # conv_d: 32, conv_h: 32, conv_w: 96 - 64 = 32
        # Total: 32 + 32 + 32 = 96
        x = torch.randn(1, in_ch, 8, 8, 4)
        out = ddc(x)
        assert out.shape[1] == out_ch

    def test_gradient_flow(self, device):
        """Verify gradients flow through DDCMod."""
        ddc = DDCMod(32, 64).to(device)
        x = torch.randn(1, 32, 8, 8, 4, device=device, requires_grad=True)
        out = ddc(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradient did not flow to input"
        assert x.grad.shape == x.shape


class TestBackboneForward:
    """Tests for Cylinder3D backbone."""

    def test_output_shape(self, device, num_classes, small_grid_size):
        """Verify backbone output is [B, num_classes, D, H, W]."""
        B = 2
        in_channels = 4
        D, H, W = small_grid_size

        backbone = Cylinder3DBackbone(in_channels, num_classes, small_grid_size).to(device)
        x = torch.randn(B, in_channels, D, H, W, device=device)
        out = backbone(x)

        expected_shape = (B, num_classes, D, H, W)
        assert out.shape == expected_shape, (
            f"Expected {expected_shape}, got {out.shape}"
        )

    def test_single_sample(self, device, num_classes, small_grid_size):
        """Test with batch size 1."""
        D, H, W = small_grid_size
        backbone = Cylinder3DBackbone(4, num_classes, small_grid_size).to(device)
        x = torch.randn(1, 4, D, H, W, device=device)
        out = backbone(x)

        assert out.shape == (1, num_classes, D, H, W)

    def test_output_dtype(self, device, small_grid_size):
        """Verify output is float32."""
        D, H, W = small_grid_size
        backbone = Cylinder3DBackbone(4, 20, small_grid_size).to(device)
        x = torch.randn(1, 4, D, H, W, device=device)
        out = backbone(x)

        assert out.dtype == torch.float32


class TestPointRefinement:
    """Tests for point-level refinement MLP."""

    def test_output_shape(self, device, num_classes):
        """Verify MLP produces [N, num_classes] output."""
        N = 500
        in_features = num_classes + 4  # voxel logits + point features
        mlp = PointRefinementMLP(in_features, num_classes).to(device)

        x = torch.randn(N, in_features, device=device)
        out = mlp(x)

        assert out.shape == (N, num_classes), (
            f"Expected ({N}, {num_classes}), got {out.shape}"
        )

    def test_various_point_counts(self, device, num_classes):
        """Test with different numbers of points."""
        in_features = num_classes + 4
        mlp = PointRefinementMLP(in_features, num_classes).to(device)

        for N in [1, 100, 1000, 5000]:
            x = torch.randn(N, in_features, device=device)
            out = mlp(x)
            assert out.shape == (N, num_classes)

    def test_gradient_flow(self, device, num_classes):
        """Verify gradients flow through the MLP."""
        in_features = num_classes + 4
        mlp = PointRefinementMLP(in_features, num_classes).to(device)

        x = torch.randn(100, in_features, device=device, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestFullModelForward:
    """Tests for end-to-end Cylinder3D model forward pass."""

    def test_output_dict_keys(self, device, num_classes, small_grid_size):
        """Verify output dict has correct keys."""
        D, H, W = small_grid_size
        model = Cylinder3DModel(
            num_classes=num_classes, grid_size=small_grid_size, in_channels=4
        ).to(device)

        N = 200
        voxel_features = torch.randn(1, 4, D, H, W, device=device)
        point_features = torch.randn(N, 4, device=device)
        grid_indices = torch.randint(0, min(D, H, W), (N, 3), device=device)
        # Clamp each dim properly
        grid_indices[:, 0] = grid_indices[:, 0] % D
        grid_indices[:, 1] = grid_indices[:, 1] % H
        grid_indices[:, 2] = grid_indices[:, 2] % W

        output = model(voxel_features, point_features, grid_indices)

        assert "voxel_logits" in output, "Missing 'voxel_logits' key"
        assert "point_logits" in output, "Missing 'point_logits' key"

    def test_output_shapes(self, device, num_classes, small_grid_size):
        """Verify output shapes are correct."""
        D, H, W = small_grid_size
        model = Cylinder3DModel(
            num_classes=num_classes, grid_size=small_grid_size, in_channels=4
        ).to(device)

        N = 300
        voxel_features = torch.randn(1, 4, D, H, W, device=device)
        point_features = torch.randn(N, 4, device=device)
        grid_indices = torch.zeros(N, 3, dtype=torch.long, device=device)
        grid_indices[:, 0] = torch.randint(0, D, (N,))
        grid_indices[:, 1] = torch.randint(0, H, (N,))
        grid_indices[:, 2] = torch.randint(0, W, (N,))

        output = model(voxel_features, point_features, grid_indices)

        assert output["voxel_logits"].shape == (1, num_classes, D, H, W)
        assert output["point_logits"].shape == (N, num_classes)

    def test_end_to_end_gradient(self, device, num_classes, small_grid_size):
        """Verify end-to-end gradient flow."""
        D, H, W = small_grid_size
        model = Cylinder3DModel(
            num_classes=num_classes, grid_size=small_grid_size, in_channels=4
        ).to(device)

        N = 100
        voxel_features = torch.randn(1, 4, D, H, W, device=device, requires_grad=True)
        point_features = torch.randn(N, 4, device=device, requires_grad=True)
        grid_indices = torch.zeros(N, 3, dtype=torch.long, device=device)
        grid_indices[:, 0] = torch.randint(0, D, (N,))
        grid_indices[:, 1] = torch.randint(0, H, (N,))
        grid_indices[:, 2] = torch.randint(0, W, (N,))

        output = model(voxel_features, point_features, grid_indices)
        loss = output["point_logits"].sum() + output["voxel_logits"].sum()
        loss.backward()

        assert voxel_features.grad is not None
        assert point_features.grad is not None


class TestLovaszLoss:
    """Tests for Lovasz-Softmax loss function."""

    def test_loss_is_differentiable(self, device):
        """Verify Lovasz loss is differentiable."""
        num_classes = 5
        N = 100

        logits = torch.randn(N, num_classes, device=device, requires_grad=True)
        labels = torch.randint(0, num_classes, (N,), device=device)

        loss = lovasz_softmax_loss(logits, labels, ignore_index=-1)
        loss.backward()

        assert logits.grad is not None, "Gradient not computed"
        assert not torch.isnan(logits.grad).any(), "NaN in gradients"

    def test_perfect_predictions_low_loss(self, device):
        """Verify loss is near zero for perfect predictions."""
        num_classes = 5
        N = 200

        # Create perfect predictions (one-hot with high confidence)
        labels = torch.randint(1, num_classes, (N,), device=device)  # avoid class 0
        logits = torch.zeros(N, num_classes, device=device)
        # Set very high logit for correct class
        for i in range(N):
            logits[i, labels[i]] = 100.0

        loss = lovasz_softmax_loss(logits, labels, ignore_index=0)

        assert loss.item() < 0.01, (
            f"Loss for perfect predictions should be ~0, got {loss.item()}"
        )

    def test_loss_positive_for_random(self, device):
        """Verify loss is positive for random predictions."""
        num_classes = 10
        N = 200

        logits = torch.randn(N, num_classes, device=device, requires_grad=True)
        labels = torch.randint(1, num_classes, (N,), device=device)

        loss = lovasz_softmax_loss(logits, labels, ignore_index=0)

        assert loss.item() > 0, "Loss should be positive for random predictions"

    def test_loss_ignores_specified_class(self, device):
        """Verify ignore_index is respected."""
        num_classes = 5
        N = 100

        logits = torch.randn(N, num_classes, device=device, requires_grad=True)
        # All labels are the ignore class
        labels = torch.zeros(N, dtype=torch.long, device=device)

        loss = lovasz_softmax_loss(logits, labels, ignore_index=0)

        assert loss.item() == 0.0, (
            f"Loss should be 0 when all labels are ignored, got {loss.item()}"
        )


class TestDatasetLoading:
    """Tests for dataset loading with mock .bin/.label files."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create a mock dataset directory with .bin and .label files."""
        seq_dir = tmp_path / "sequences" / "00"
        vel_dir = seq_dir / "velodyne"
        lab_dir = seq_dir / "labels"
        vel_dir.mkdir(parents=True)
        lab_dir.mkdir(parents=True)

        # Create 5 mock scan/label pairs
        rng = np.random.default_rng(123)
        for i in range(5):
            # Random point cloud (N=100 points, 4 floats each)
            n_points = 100
            points = rng.standard_normal((n_points, 4)).astype(np.float32)
            points[:, 3] = np.abs(points[:, 3])  # intensity non-negative

            # Save .bin file
            bin_path = vel_dir / f"{i:06d}.bin"
            points.tofile(str(bin_path))

            # Create labels (uint32: lower 16 bits = semantic)
            # Use valid SemanticKITTI label IDs
            valid_labels = [0, 10, 11, 13, 15, 40, 48, 50, 70, 71, 72, 80, 81]
            semantic = rng.choice(valid_labels, n_points).astype(np.uint16)
            instance = rng.integers(0, 10, n_points).astype(np.uint16)
            combined = (instance.astype(np.uint32) << 16) | semantic.astype(np.uint32)

            # Save .label file
            label_path = lab_dir / f"{i:06d}.label"
            combined.tofile(str(label_path))

        return tmp_path

    def test_bin_loading_shape(self, mock_dataset_dir):
        """Verify .bin files load with correct shape (N, 4)."""
        bin_path = mock_dataset_dir / "sequences" / "00" / "velodyne" / "000000.bin"
        points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)

        assert points.shape == (100, 4), f"Expected (100, 4), got {points.shape}"
        assert points.dtype == np.float32

    def test_label_loading_shape(self, mock_dataset_dir):
        """Verify .label files load with correct shape (N,)."""
        label_path = mock_dataset_dir / "sequences" / "00" / "labels" / "000000.label"
        raw = np.fromfile(str(label_path), dtype=np.uint32)
        semantic = raw & 0xFFFF

        assert semantic.shape == (100,), f"Expected (100,), got {semantic.shape}"

    def test_point_label_count_match(self, mock_dataset_dir):
        """Verify point count matches label count for each scan."""
        vel_dir = mock_dataset_dir / "sequences" / "00" / "velodyne"
        lab_dir = mock_dataset_dir / "sequences" / "00" / "labels"

        for bin_file in sorted(vel_dir.glob("*.bin")):
            label_file = lab_dir / bin_file.name.replace(".bin", ".label")
            assert label_file.exists(), f"Missing label for {bin_file.name}"

            points = np.fromfile(str(bin_file), dtype=np.float32).reshape(-1, 4)
            labels = np.fromfile(str(label_file), dtype=np.uint32)

            assert len(points) == len(labels), (
                f"Mismatch in {bin_file.name}: "
                f"{len(points)} points vs {len(labels)} labels"
            )

    def test_label_values_valid(self, mock_dataset_dir):
        """Verify label values are valid SemanticKITTI IDs."""
        label_path = mock_dataset_dir / "sequences" / "00" / "labels" / "000000.label"
        raw = np.fromfile(str(label_path), dtype=np.uint32)
        semantic = raw & 0xFFFF

        # All semantic labels should be mappable
        learning_map = {
            0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4,
            20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11,
            49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16,
            72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6,
            255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
        }

        valid_raw_labels = set(learning_map.keys())
        unique_labels = set(semantic.tolist())

        for label in unique_labels:
            assert label in valid_raw_labels, (
                f"Invalid label {label} not in learning map"
            )

    def test_mapped_labels_in_range(self, mock_dataset_dir):
        """Verify mapped labels are in [0, 19]."""
        label_path = mock_dataset_dir / "sequences" / "00" / "labels" / "000000.label"
        raw = np.fromfile(str(label_path), dtype=np.uint32)
        semantic = raw & 0xFFFF

        learning_map = {
            0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4,
            20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11,
            49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16,
            72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6,
            255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
        }

        mapped = np.zeros_like(semantic)
        for raw_id, mapped_id in learning_map.items():
            mapped[semantic == raw_id] = mapped_id

        assert mapped.min() >= 0, f"Mapped label below 0: {mapped.min()}"
        assert mapped.max() <= 19, f"Mapped label above 19: {mapped.max()}"

    def test_intensity_non_negative(self, mock_dataset_dir):
        """Verify intensity channel is non-negative."""
        bin_path = mock_dataset_dir / "sequences" / "00" / "velodyne" / "000000.bin"
        points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
        intensity = points[:, 3]

        assert (intensity >= 0).all(), "Intensity values should be non-negative"

    def test_all_scans_have_labels(self, mock_dataset_dir):
        """Verify every scan file has a corresponding label file."""
        vel_dir = mock_dataset_dir / "sequences" / "00" / "velodyne"
        lab_dir = mock_dataset_dir / "sequences" / "00" / "labels"

        bin_files = sorted(vel_dir.glob("*.bin"))
        assert len(bin_files) == 5, f"Expected 5 .bin files, got {len(bin_files)}"

        for bin_file in bin_files:
            label_file = lab_dir / bin_file.name.replace(".bin", ".label")
            assert label_file.exists(), f"Missing label for {bin_file.name}"


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
