"""
StreamMapNet BEV Transform: Lift-Splat-Shoot (LSS) Implementation

Transforms perspective image features into a Bird's Eye View (BEV) representation
using predicted depth distributions and known camera geometry.

Pipeline:
    1. Depth Prediction: Image features -> per-pixel depth distribution (D bins)
    2. Create Frustum: Build a 3D frustum grid for each pixel at each depth bin
    3. Lift: Outer product of features with depth probs -> pseudo point cloud
    4. Splat: Project points to ego frame using camera params, scatter into BEV grid

Reference:
    - Philion & Fidler, "Lift, Splat, Shoot: Encoding Images from Arbitrary
      Camera Rigs by Implicitly Unprojecting to 3D", ECCV 2020
    - Li et al., "StreamMapNet: Streaming Mapping Network for Vectorized Online
      HD Map Construction", WACV 2024

BEV Grid Configuration:
    - X range: [-30.0, 30.0] meters (lateral)
    - Y range: [-15.0, 15.0] meters (longitudinal)
    - Resolution: 200 x 100 (0.3m per cell)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DepthNet(nn.Module):
    """Predicts per-pixel depth distribution from image features.

    Takes image feature maps and produces a categorical depth distribution
    over D discrete depth bins for each spatial location. The output represents
    the probability that each pixel's content lies at each depth.

    Architecture:
        Conv2d(C_in, C_in, 3) -> BN -> ReLU ->
        Conv2d(C_in, C_in, 3) -> BN -> ReLU ->
        Conv2d(C_in, D, 1) -> Softmax(dim=1)
    """

    def __init__(
        self,
        in_channels: int = 256,
        mid_channels: int = 256,
        num_depth_bins: int = 60,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
    ):
        """
        Args:
            in_channels: Number of input feature channels (from FPN).
            mid_channels: Number of intermediate channels in depth head.
            num_depth_bins: Number of discrete depth bins (D).
            depth_min: Minimum depth in meters.
            depth_max: Maximum depth in meters.
        """
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max

        # Depth bin centers (uniform spacing)
        depth_bin_edges = torch.linspace(depth_min, depth_max, num_depth_bins + 1)
        depth_bin_centers = 0.5 * (depth_bin_edges[:-1] + depth_bin_edges[1:])
        self.register_buffer("depth_bins", depth_bin_centers)

        # Depth prediction network
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_depth_bins, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize convolution layers with Kaiming normal."""
        for m in self.depth_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict depth distribution for each spatial location.

        Args:
            features: Image features, (B*N, C, H_feat, W_feat).

        Returns:
            Depth probabilities, (B*N, D, H_feat, W_feat), summing to 1 along D.
        """
        depth_logits = self.depth_head(features)  # (B*N, D, H_feat, W_feat)
        depth_probs = F.softmax(depth_logits, dim=1)
        return depth_probs


class FrustumGrid(nn.Module):
    """Creates a 3D frustum point grid in camera coordinates.

    For each pixel in the feature map, creates D points at the depth bin centers.
    The frustum is defined in normalized image coordinates and then unprojected
    to 3D using camera intrinsics.

    The frustum grid is (D, H_feat, W_feat, 3) where the last dimension is (u, v, d)
    in pixel coordinates before unprojection, or (x, y, z) in camera frame after.
    """

    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        img_height: int,
        img_width: int,
        num_depth_bins: int = 60,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
        downsample_factor: int = 8,
    ):
        """
        Args:
            feat_height: Height of the feature map.
            feat_width: Width of the feature map.
            img_height: Original image height.
            img_width: Original image width.
            num_depth_bins: Number of discrete depth bins.
            depth_min: Minimum depth in meters.
            depth_max: Maximum depth in meters.
            downsample_factor: Spatial downsample factor from image to features.
        """
        super().__init__()
        self.feat_height = feat_height
        self.feat_width = feat_width
        self.img_height = img_height
        self.img_width = img_width
        self.num_depth_bins = num_depth_bins
        self.downsample_factor = downsample_factor

        # Create depth bins
        depth_bins = torch.linspace(depth_min, depth_max, num_depth_bins)

        # Create frustum grid: pixel coordinates at each depth
        # Feature pixel centers mapped back to image coordinates
        xs = torch.linspace(0, img_width - 1, feat_width)
        ys = torch.linspace(0, img_height - 1, feat_height)

        # Grid of (u, v) pixel coordinates: (H_feat, W_feat, 2)
        # Note: meshgrid with indexing='ij' gives (H, W) ordering
        ys_grid, xs_grid = torch.meshgrid(ys, xs, indexing="ij")

        # Expand to include depth dimension: (D, H_feat, W_feat, 3)
        # Last dim is (x_pixel, y_pixel, depth)
        frustum = torch.zeros(num_depth_bins, feat_height, feat_width, 3)
        frustum[:, :, :, 0] = xs_grid.unsqueeze(0).expand(num_depth_bins, -1, -1)
        frustum[:, :, :, 1] = ys_grid.unsqueeze(0).expand(num_depth_bins, -1, -1)
        frustum[:, :, :, 2] = depth_bins.view(-1, 1, 1).expand(-1, feat_height, feat_width)

        self.register_buffer("frustum", frustum)  # (D, H_feat, W_feat, 3)

    def forward(self) -> torch.Tensor:
        """Return the frustum grid.

        Returns:
            Frustum tensor of shape (D, H_feat, W_feat, 3) where last dim is
            (u_pixel, v_pixel, depth_meters).
        """
        return self.frustum


class LiftSplatShoot(nn.Module):
    """Lift-Splat-Shoot module for BEV feature generation.

    Transforms 2D image features from multiple cameras into a unified BEV
    feature map using predicted depth distributions and known camera geometry.

    The process:
        1. Predict depth distribution for each feature pixel
        2. Lift: Create a 3D pseudo point cloud by multiplying features with depth probs
        3. Project frustum points to ego frame using camera intrinsics/extrinsics
        4. Splat: Pool the lifted features into a discretized BEV grid

    BEV Grid:
        Default covers [-30, 30] x [-15, 15] meters at 200 x 100 resolution.
    """

    def __init__(
        self,
        in_channels: int = 256,
        feat_height: int = 32,
        feat_width: int = 88,
        img_height: int = 256,
        img_width: int = 704,
        num_depth_bins: int = 60,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
        downsample_factor: int = 8,
        bev_x_range: Tuple[float, float] = (-30.0, 30.0),
        bev_y_range: Tuple[float, float] = (-15.0, 15.0),
        bev_resolution: Tuple[int, int] = (200, 100),
        bev_channels: int = 256,
    ):
        """
        Args:
            in_channels: Input feature channels from backbone/FPN.
            feat_height: Feature map height (after backbone downsampling).
            feat_width: Feature map width (after backbone downsampling).
            img_height: Original input image height.
            img_width: Original input image width.
            num_depth_bins: Number of depth discretization bins (D).
            depth_min: Minimum depth in meters.
            depth_max: Maximum depth in meters.
            downsample_factor: Spatial downsampling factor of features vs input.
            bev_x_range: BEV grid lateral range in meters (min, max).
            bev_y_range: BEV grid longitudinal range in meters (min, max).
            bev_resolution: BEV grid resolution (W_bev, H_bev).
            bev_channels: Output BEV feature channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.feat_height = feat_height
        self.feat_width = feat_width
        self.num_depth_bins = num_depth_bins
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.bev_w, self.bev_h = bev_resolution

        # Depth prediction network
        self.depth_net = DepthNet(
            in_channels=in_channels,
            mid_channels=in_channels,
            num_depth_bins=num_depth_bins,
            depth_min=depth_min,
            depth_max=depth_max,
        )

        # Frustum grid creation
        self.frustum_grid = FrustumGrid(
            feat_height=feat_height,
            feat_width=feat_width,
            img_height=img_height,
            img_width=img_width,
            num_depth_bins=num_depth_bins,
            depth_min=depth_min,
            depth_max=depth_max,
            downsample_factor=downsample_factor,
        )

        # BEV grid parameters
        self.bev_x_step = (bev_x_range[1] - bev_x_range[0]) / self.bev_w
        self.bev_y_step = (bev_y_range[1] - bev_y_range[0]) / self.bev_h

        # BEV feature compression: reduce channels after splatting
        self.bev_compress = nn.Sequential(
            nn.Conv2d(in_channels, bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

        self._init_bev_compress()

    def _init_bev_compress(self):
        """Initialize BEV compression convolutions."""
        for m in self.bev_compress.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def unproject_frustum_to_3d(
        self,
        frustum: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Unproject frustum points from pixel coords to ego-vehicle 3D coordinates.

        Transforms (u, v, d) pixel+depth tuples through:
            1. Camera intrinsics inverse: pixel -> camera 3D
            2. Camera extrinsics: camera 3D -> ego-vehicle 3D

        Args:
            frustum: (D, H_feat, W_feat, 3) frustum grid with (u, v, depth).
            intrinsics: (B*N, 3, 3) camera intrinsic matrices.
            extrinsics: (B*N, 4, 4) camera-to-ego extrinsic matrices (cam2ego).

        Returns:
            ego_points: (B*N, D, H_feat, W_feat, 3) 3D points in ego frame.
        """
        BN = intrinsics.shape[0]
        D, H, W, _ = frustum.shape

        # Extract pixel coords and depth
        # frustum[..., 0] = u (x pixel), frustum[..., 1] = v (y pixel), frustum[..., 2] = d
        points = frustum.clone()  # (D, H, W, 3)

        # Convert pixel coordinates to normalized camera coordinates using intrinsics
        # p_cam = K^-1 @ [u*d, v*d, d]^T
        # First, create homogeneous pixel coordinates scaled by depth
        u = points[:, :, :, 0:1]  # (D, H, W, 1)
        v = points[:, :, :, 1:2]  # (D, H, W, 1)
        d = points[:, :, :, 2:3]  # (D, H, W, 1)

        # Scale pixel coords by depth: [u*d, v*d, d]
        points_scaled = torch.cat([u * d, v * d, d], dim=-1)  # (D, H, W, 3)

        # Reshape for batch matrix multiplication
        points_flat = points_scaled.view(1, D * H * W, 3).expand(BN, -1, -1)  # (BN, D*H*W, 3)

        # Apply inverse intrinsics: camera_coords = K^-1 @ pixel_coords
        K_inv = torch.inverse(intrinsics)  # (BN, 3, 3)
        cam_points = torch.bmm(points_flat, K_inv.transpose(1, 2))  # (BN, D*H*W, 3)

        # Apply extrinsics (camera to ego transformation)
        # extrinsics is cam2ego: 4x4 matrix [R|t]
        R = extrinsics[:, :3, :3]  # (BN, 3, 3) rotation
        t = extrinsics[:, :3, 3:4]  # (BN, 3, 1) translation

        # ego_points = R @ cam_points + t
        ego_points = torch.bmm(cam_points, R.transpose(1, 2)) + t.transpose(1, 2)  # (BN, D*H*W, 3)

        # Reshape back to spatial grid
        ego_points = ego_points.view(BN, D, H, W, 3)

        return ego_points

    def voxel_pooling(
        self,
        ego_points: torch.Tensor,
        lifted_features: torch.Tensor,
        batch_size: int,
        num_cams: int,
    ) -> torch.Tensor:
        """Pool lifted features into a BEV grid using scatter operations.

        For each lifted 3D point, determine which BEV grid cell it falls into,
        then sum all features that map to the same cell.

        Args:
            ego_points: (B*N, D, H_feat, W_feat, 3) 3D points in ego frame.
            lifted_features: (B*N, C, D, H_feat, W_feat) features at each 3D point.
            batch_size: Batch size B.
            num_cams: Number of cameras N.

        Returns:
            bev_features: (B, C, H_bev, W_bev) BEV feature map.
        """
        BN, C, D, H, W = lifted_features.shape
        device = lifted_features.device

        # Extract x, y coordinates in ego frame
        # ego_points: (BN, D, H, W, 3) -> x, y, z
        x = ego_points[:, :, :, :, 0]  # (BN, D, H, W) - lateral
        y = ego_points[:, :, :, :, 1]  # (BN, D, H, W) - longitudinal

        # Convert continuous coordinates to BEV grid indices
        # x -> bev_col (W_bev), y -> bev_row (H_bev)
        bev_col = ((x - self.bev_x_range[0]) / self.bev_x_step).long()  # (BN, D, H, W)
        bev_row = ((y - self.bev_y_range[0]) / self.bev_y_step).long()  # (BN, D, H, W)

        # Create batch indices
        batch_indices = torch.arange(batch_size, device=device)
        # Map camera index to batch: cameras [0..N-1] -> batch 0, [N..2N-1] -> batch 1, etc.
        cam_to_batch = batch_indices.repeat_interleave(num_cams)  # (BN,)
        batch_idx = cam_to_batch.view(BN, 1, 1, 1).expand(-1, D, H, W)  # (BN, D, H, W)

        # Valid mask: points that fall within the BEV grid
        valid = (
            (bev_col >= 0) & (bev_col < self.bev_w) &
            (bev_row >= 0) & (bev_row < self.bev_h)
        )  # (BN, D, H, W)

        # Flatten spatial dimensions for scatter
        # Compute linear index into (B, H_bev, W_bev) grid
        linear_idx = (
            batch_idx * self.bev_h * self.bev_w +
            bev_row * self.bev_w +
            bev_col
        )  # (BN, D, H, W)

        # Apply validity mask
        linear_idx[~valid] = 0  # Will be masked out

        # Flatten everything for scatter_add
        linear_idx_flat = linear_idx.view(-1)  # (BN*D*H*W,)
        valid_flat = valid.view(-1)  # (BN*D*H*W,)

        # Reshape features: (BN, C, D, H, W) -> (C, BN*D*H*W)
        features_flat = lifted_features.permute(1, 0, 2, 3, 4).reshape(C, -1)  # (C, BN*D*H*W)

        # Zero out invalid features
        features_flat[:, ~valid_flat] = 0.0

        # Scatter add into BEV grid
        bev_flat = torch.zeros(C, batch_size * self.bev_h * self.bev_w, device=device)
        linear_idx_expanded = linear_idx_flat.unsqueeze(0).expand(C, -1)  # (C, BN*D*H*W)
        bev_flat.scatter_add_(1, linear_idx_expanded, features_flat)

        # Reshape to (B, C, H_bev, W_bev)
        bev_features = bev_flat.view(C, batch_size, self.bev_h, self.bev_w)
        bev_features = bev_features.permute(1, 0, 2, 3)  # (B, C, H_bev, W_bev)

        return bev_features

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        batch_size: int,
        num_cams: int = 6,
    ) -> torch.Tensor:
        """Transform image features to BEV representation via LSS.

        Full pipeline: depth prediction -> lift -> splat -> BEV compression.

        Args:
            features: (B*N, C, H_feat, W_feat) image features from backbone.
                     Typical shape: (B*6, 256, 32, 88).
            intrinsics: (B*N, 3, 3) camera intrinsic matrices.
                       Contains focal lengths and principal points:
                       [[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]]
            extrinsics: (B*N, 4, 4) camera-to-ego extrinsic matrices.
                       Transforms points from camera frame to ego-vehicle frame:
                       [[R(3x3) | t(3x1)],
                        [0 0 0  |   1   ]]
            batch_size: Number of samples in the batch (B).
            num_cams: Number of cameras per sample (N). Default 6.

        Returns:
            bev_features: (B, C_bev, H_bev, W_bev) BEV feature map.
                         Default shape: (B, 256, 100, 200).
        """
        BN, C, H_feat, W_feat = features.shape
        assert BN == batch_size * num_cams, (
            f"Feature batch dim {BN} != batch_size({batch_size}) * num_cams({num_cams})"
        )

        # Step 1: Predict depth distribution
        depth_probs = self.depth_net(features)  # (BN, D, H_feat, W_feat)

        # Step 2: Lift - outer product of features and depth probabilities
        # features: (BN, C, H_feat, W_feat) -> (BN, C, 1, H_feat, W_feat)
        # depth_probs: (BN, D, H_feat, W_feat) -> (BN, 1, D, H_feat, W_feat)
        features_expanded = features.unsqueeze(2)  # (BN, C, 1, H, W)
        depth_expanded = depth_probs.unsqueeze(1)  # (BN, 1, D, H, W)

        # Lifted features: each feature vector weighted by depth probability
        lifted_features = features_expanded * depth_expanded  # (BN, C, D, H, W)

        # Step 3: Get frustum points and project to ego frame
        frustum = self.frustum_grid()  # (D, H_feat, W_feat, 3)
        ego_points = self.unproject_frustum_to_3d(
            frustum, intrinsics, extrinsics
        )  # (BN, D, H_feat, W_feat, 3)

        # Step 4: Splat - voxel pooling into BEV grid
        bev_features = self.voxel_pooling(
            ego_points, lifted_features, batch_size, num_cams
        )  # (B, C, H_bev, W_bev)

        # Step 5: BEV feature compression
        bev_features = self.bev_compress(bev_features)  # (B, C_bev, H_bev, W_bev)

        return bev_features


class StreamMapNetBEVTransform(nn.Module):
    """High-level wrapper combining backbone features with LSS BEV transformation.

    Accepts multi-camera features (already extracted by backbone) and camera
    parameters, producing a unified BEV feature map for downstream map prediction.
    """

    def __init__(
        self,
        in_channels: int = 256,
        img_height: int = 256,
        img_width: int = 704,
        feat_height: int = 32,
        feat_width: int = 88,
        num_depth_bins: int = 60,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
        bev_x_range: Tuple[float, float] = (-30.0, 30.0),
        bev_y_range: Tuple[float, float] = (-15.0, 15.0),
        bev_resolution: Tuple[int, int] = (200, 100),
        bev_channels: int = 256,
        num_cams: int = 6,
    ):
        """
        Args:
            in_channels: Feature channels from backbone.
            img_height: Original image height.
            img_width: Original image width.
            feat_height: Feature map height.
            feat_width: Feature map width.
            num_depth_bins: Number of depth bins for LSS.
            depth_min: Minimum depth (meters).
            depth_max: Maximum depth (meters).
            bev_x_range: BEV lateral range (meters).
            bev_y_range: BEV longitudinal range (meters).
            bev_resolution: (W_bev, H_bev) grid dimensions.
            bev_channels: Output BEV feature channels.
            num_cams: Number of cameras.
        """
        super().__init__()
        self.num_cams = num_cams

        self.lss = LiftSplatShoot(
            in_channels=in_channels,
            feat_height=feat_height,
            feat_width=feat_width,
            img_height=img_height,
            img_width=img_width,
            num_depth_bins=num_depth_bins,
            depth_min=depth_min,
            depth_max=depth_max,
            downsample_factor=img_height // feat_height,
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
            bev_resolution=bev_resolution,
            bev_channels=bev_channels,
        )

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Generate BEV features from multi-camera image features.

        Args:
            features: (B, N, C, H_feat, W_feat) multi-camera features.
            intrinsics: (B, N, 3, 3) camera intrinsic matrices.
            extrinsics: (B, N, 4, 4) camera-to-ego extrinsic matrices.

        Returns:
            bev: (B, C_bev, H_bev, W_bev) BEV feature map.
        """
        B, N, C, H, W = features.shape
        assert N == self.num_cams

        # Flatten batch and camera dims
        features_flat = features.view(B * N, C, H, W)
        intrinsics_flat = intrinsics.view(B * N, 3, 3)
        extrinsics_flat = extrinsics.view(B * N, 4, 4)

        # Run LSS
        bev = self.lss(
            features_flat,
            intrinsics_flat,
            extrinsics_flat,
            batch_size=B,
            num_cams=N,
        )

        return bev


if __name__ == "__main__":
    # Demonstration with typical autonomous driving setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    batch_size = 2
    num_cams = 6
    in_channels = 256
    feat_h, feat_w = 32, 88  # Feature map dims (from 256x704 input at stride 8)
    img_h, img_w = 256, 704

    print("=" * 60)
    print("StreamMapNet BEV Transform (LSS) - Demo")
    print("=" * 60)

    # Create BEV transform module
    bev_transform = StreamMapNetBEVTransform(
        in_channels=in_channels,
        img_height=img_h,
        img_width=img_w,
        feat_height=feat_h,
        feat_width=feat_w,
        num_depth_bins=60,
        depth_min=1.0,
        depth_max=60.0,
        bev_x_range=(-30.0, 30.0),
        bev_y_range=(-15.0, 15.0),
        bev_resolution=(200, 100),
        bev_channels=256,
        num_cams=num_cams,
    ).to(device)

    # Synthetic inputs
    features = torch.randn(batch_size, num_cams, in_channels, feat_h, feat_w, device=device)

    # Realistic camera intrinsics (focal length ~450px for 704px width)
    fx, fy = 450.0, 450.0
    cx, cy = 352.0, 128.0
    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], device=device)
    intrinsics = K.unsqueeze(0).unsqueeze(0).expand(batch_size, num_cams, -1, -1).clone()

    # Realistic camera extrinsics (6 cameras around the vehicle)
    # cam2ego transforms for: front, front-left, front-right, back, back-left, back-right
    extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(
        batch_size, num_cams, -1, -1
    ).clone()

    # Set approximate camera positions (meters from ego center)
    cam_positions = torch.tensor([
        [1.5, 0.0, 1.5],    # Front camera
        [1.0, -0.8, 1.5],   # Front-left
        [1.0, 0.8, 1.5],    # Front-right
        [-1.5, 0.0, 1.5],   # Back
        [-1.0, -0.8, 1.5],  # Back-left
        [-1.0, 0.8, 1.5],   # Back-right
    ], device=device)

    for b in range(batch_size):
        for n in range(num_cams):
            extrinsics[b, n, :3, 3] = cam_positions[n]

    # Forward pass
    bev_transform.eval()
    with torch.no_grad():
        bev_features = bev_transform(features, intrinsics, extrinsics)

    print(f"\nInput features shape:  {features.shape}")
    print(f"Intrinsics shape:      {intrinsics.shape}")
    print(f"Extrinsics shape:      {extrinsics.shape}")
    print(f"Output BEV shape:      {bev_features.shape}")
    print(f"\nBEV grid coverage:")
    print(f"  X (lateral):      [-30.0, 30.0] m  ->  200 cells (0.30 m/cell)")
    print(f"  Y (longitudinal): [-15.0, 15.0] m  ->  100 cells (0.30 m/cell)")

    # Parameter count
    total_params = sum(p.numel() for p in bev_transform.parameters())
    trainable_params = sum(p.numel() for p in bev_transform.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Verify depth probabilities sum to 1
    with torch.no_grad():
        depth_probs = bev_transform.lss.depth_net(features.view(-1, in_channels, feat_h, feat_w))
        depth_sum = depth_probs.sum(dim=1)
        print(f"\nDepth prob sum (should be ~1.0): {depth_sum.mean().item():.6f}")
