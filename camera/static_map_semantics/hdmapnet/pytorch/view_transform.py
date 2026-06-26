"""
View transform modules for HDMapNet.

Provides two methods to transform perspective camera features to BEV:
1. IPM (Inverse Perspective Mapping): Assumes flat ground plane (z=0),
   computes homography from camera parameters, and warps features.
2. LSS (Lift-Splat-Shoot): Predicts per-pixel depth distribution,
   lifts 2D features to 3D frustum point cloud, and splats into BEV grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IPMTransform(nn.Module):
    """Inverse Perspective Mapping view transform.

    Computes homography matrices from camera intrinsics and extrinsics
    assuming a flat ground plane (z=0), then warps perspective features
    to the BEV plane using grid_sample.
    """

    def __init__(self, xbound, ybound, image_size, feature_stride=8):
        """
        Args:
            xbound: [xmin, xmax, resolution] for BEV x-axis (meters).
            ybound: [ymin, ymax, resolution] for BEV y-axis (meters).
            image_size: (H, W) of the input image.
            feature_stride: Downsampling factor of feature map relative to input.
        """
        super().__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.image_size = image_size
        self.feature_stride = feature_stride

        # BEV grid dimensions
        self.bev_w = int((xbound[1] - xbound[0]) / xbound[2])
        self.bev_h = int((ybound[1] - ybound[0]) / ybound[2])

        # Create BEV grid coordinates in world frame (z=0 plane)
        xs = torch.linspace(xbound[0] + xbound[2] / 2, xbound[1] - xbound[2] / 2, self.bev_w)
        ys = torch.linspace(ybound[0] + ybound[2] / 2, ybound[1] - ybound[2] / 2, self.bev_h)
        # Grid shape: (bev_h, bev_w, 3) with homogeneous coords [x, y, 1]
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        # World points on ground plane: (x, y, z=0) -> homogeneous (x, y, 0, 1)
        ones = torch.ones_like(xx)
        zeros = torch.zeros_like(xx)
        # (bev_h, bev_w, 4)
        self.register_buffer(
            "world_coords",
            torch.stack([xx, ys.unsqueeze(1).expand_as(xx), zeros, ones], dim=-1),
        )
        # Actually we need (x, y, 0, 1) in ego frame
        self.register_buffer(
            "bev_points",
            torch.stack([xx, yy, zeros, ones], dim=-1).reshape(-1, 4),
        )

    def get_pixel_coords(self, intrinsics, extrinsics):
        """Project BEV world points to image pixel coordinates.

        Args:
            intrinsics: Camera intrinsic matrices (B, N, 3, 3).
            extrinsics: Camera extrinsic matrices (B, N, 4, 4) - world to camera.

        Returns:
            Normalized pixel coordinates (B, N, bev_h, bev_w, 2) in [-1, 1].
        """
        B, N = intrinsics.shape[:2]
        device = intrinsics.device

        # BEV points in ego/world frame: (num_points, 4)
        points = self.bev_points.to(device)  # (bev_h*bev_w, 4)

        # Transform to camera frame: P_cam = extrinsics @ P_world
        # extrinsics: (B, N, 4, 4), points: (P, 4)
        points_expanded = points.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # (B, N, P, 4)
        # (B, N, 4, 4) @ (B, N, 4, P) -> (B, N, 4, P)
        cam_points = torch.matmul(extrinsics, points_expanded.permute(0, 1, 3, 2))  # (B, N, 4, P)
        cam_points = cam_points.permute(0, 1, 3, 2)  # (B, N, P, 4)
        cam_points = cam_points[..., :3]  # (B, N, P, 3)

        # Project to image plane: p = K @ P_cam
        # intrinsics: (B, N, 3, 3), cam_points: (B, N, P, 3)
        pixel_coords = torch.matmul(intrinsics, cam_points.permute(0, 1, 3, 2))  # (B, N, 3, P)
        pixel_coords = pixel_coords.permute(0, 1, 3, 2)  # (B, N, P, 3)

        # Normalize by depth (z)
        depth = pixel_coords[..., 2:3].clamp(min=1e-5)
        pixel_coords = pixel_coords[..., :2] / depth  # (B, N, P, 2)

        # Account for feature stride
        pixel_coords = pixel_coords / self.feature_stride

        # Normalize to [-1, 1] for grid_sample
        feat_h = self.image_size[0] // self.feature_stride
        feat_w = self.image_size[1] // self.feature_stride
        pixel_coords[..., 0] = 2.0 * pixel_coords[..., 0] / (feat_w - 1) - 1.0
        pixel_coords[..., 1] = 2.0 * pixel_coords[..., 1] / (feat_h - 1) - 1.0

        # Reshape to (B, N, bev_h, bev_w, 2)
        pixel_coords = pixel_coords.reshape(B, N, self.bev_h, self.bev_w, 2)

        return pixel_coords, depth.reshape(B, N, self.bev_h, self.bev_w, 1)

    def forward(self, features, intrinsics, extrinsics):
        """
        Args:
            features: Multi-camera features (B, N, C, fH, fW).
            intrinsics: Camera intrinsic matrices (B, N, 3, 3).
            extrinsics: Camera extrinsic matrices (B, N, 4, 4).

        Returns:
            BEV feature map (B, C, bev_h, bev_w).
        """
        B, N, C, fH, fW = features.shape

        # Get pixel coordinates for BEV points in each camera
        pixel_coords, depths = self.get_pixel_coords(intrinsics, extrinsics)

        # Create validity mask: points must be in front of camera and within image
        valid = (
            (pixel_coords[..., 0] >= -1.0)
            & (pixel_coords[..., 0] <= 1.0)
            & (pixel_coords[..., 1] >= -1.0)
            & (pixel_coords[..., 1] <= 1.0)
            & (depths.squeeze(-1) > 0)
        )  # (B, N, bev_h, bev_w)

        # Warp features from each camera to BEV
        bev_features = torch.zeros(B, C, self.bev_h, self.bev_w, device=features.device)
        weight_sum = torch.zeros(B, 1, self.bev_h, self.bev_w, device=features.device)

        for n in range(N):
            cam_feat = features[:, n]  # (B, C, fH, fW)
            grid = pixel_coords[:, n]  # (B, bev_h, bev_w, 2)
            mask = valid[:, n].unsqueeze(1).float()  # (B, 1, bev_h, bev_w)

            warped = F.grid_sample(
                cam_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )  # (B, C, bev_h, bev_w)

            bev_features = bev_features + warped * mask
            weight_sum = weight_sum + mask

        # Average over cameras that see each BEV location
        weight_sum = weight_sum.clamp(min=1.0)
        bev_features = bev_features / weight_sum

        return bev_features


class CamEncode(nn.Module):
    """Camera encoder that predicts a depth distribution for each pixel
    in the feature map, used by LSS view transform."""

    def __init__(self, in_channels, depth_channels, mid_channels=128):
        """
        Args:
            in_channels: Number of input feature channels.
            depth_channels: Number of depth bins.
            mid_channels: Intermediate channel count.
        """
        super().__init__()
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, depth_channels + in_channels, kernel_size=1),
        )
        self.depth_channels = depth_channels
        self.in_channels = in_channels

    def forward(self, x):
        """
        Args:
            x: Feature map (B*N, C, fH, fW).

        Returns:
            depth: Depth distribution (B*N, D, fH, fW).
            context: Context features (B*N, C, fH, fW).
        """
        out = self.depth_net(x)
        depth = out[:, : self.depth_channels]
        context = out[:, self.depth_channels :]

        depth = depth.softmax(dim=1)
        return depth, context


class LSSTransform(nn.Module):
    """Lift-Splat-Shoot view transform.

    Predicts per-pixel depth distribution, creates a frustum point cloud
    by lifting 2D features to 3D, and splats features into a BEV voxel grid.
    """

    def __init__(self, in_channels, xbound, ybound, zbound, dbound, image_size, feature_stride=8):
        """
        Args:
            in_channels: Number of feature channels from backbone.
            xbound: [xmin, xmax, resolution] for BEV x-axis (meters).
            ybound: [ymin, ymax, resolution] for BEV y-axis (meters).
            zbound: [zmin, zmax, resolution] for BEV z-axis (meters).
            dbound: [dmin, dmax, resolution] for depth bins (meters).
            image_size: (H, W) of the input image.
            feature_stride: Downsampling factor of feature map.
        """
        super().__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.image_size = image_size
        self.feature_stride = feature_stride
        self.in_channels = in_channels

        # Grid dimensions
        self.bev_w = int((xbound[1] - xbound[0]) / xbound[2])
        self.bev_h = int((ybound[1] - ybound[0]) / ybound[2])
        self.bev_z = int((zbound[1] - zbound[0]) / zbound[2])

        # Depth bins
        self.depth_channels = int((dbound[1] - dbound[0]) / dbound[2])

        # Feature map size
        self.feat_h = image_size[0] // feature_stride
        self.feat_w = image_size[1] // feature_stride

        # Camera encoding network (predicts depth + context)
        self.cam_encode = CamEncode(in_channels, self.depth_channels, mid_channels=128)

        # Create frustum grid
        self.register_buffer("frustum", self._create_frustum())

    def _create_frustum(self):
        """Create a frustum grid of (D, fH, fW, 3) with pixel + depth coords.

        Each point stores (u * d, v * d, d) where u, v are pixel coordinates
        in the feature map and d is the depth value.
        """
        # Depth values
        ds = torch.arange(
            self.dbound[0], self.dbound[1], self.dbound[2]
        ).reshape(-1, 1, 1).expand(-1, self.feat_h, self.feat_w)

        D = ds.shape[0]

        # Pixel coordinates in feature map space
        xs = torch.linspace(0, self.image_size[1] - 1, self.feat_w).reshape(
            1, 1, self.feat_w
        ).expand(D, self.feat_h, -1)
        ys = torch.linspace(0, self.image_size[0] - 1, self.feat_h).reshape(
            1, self.feat_h, 1
        ).expand(D, -1, self.feat_w)

        # Frustum: (D, fH, fW, 3) with (u*d, v*d, d)
        frustum = torch.stack([xs * ds, ys * ds, ds], dim=-1)
        return frustum

    def get_geometry(self, intrinsics, extrinsics):
        """Unproject frustum points to 3D ego frame coordinates.

        Args:
            intrinsics: Camera intrinsics (B, N, 3, 3).
            extrinsics: Camera extrinsics (B, N, 4, 4) - camera to ego.

        Returns:
            geom: 3D coordinates (B, N, D, fH, fW, 3) in ego frame.
        """
        B, N = intrinsics.shape[:2]
        D = self.frustum.shape[0]

        # Frustum points: (D, fH, fW, 3)
        points = self.frustum.clone()

        # Unproject from image to camera frame: P_cam = K^-1 @ p
        # points shape: (D*fH*fW, 3)
        points_flat = points.reshape(-1, 3).T  # (3, D*fH*fW)

        # Expand for batch and cameras
        points_flat = points_flat.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)  # (B, N, 3, D*fH*fW)

        # Invert intrinsics
        inv_intrinsics = torch.inverse(intrinsics)  # (B, N, 3, 3)

        # Unproject to camera frame
        cam_points = torch.matmul(inv_intrinsics, points_flat)  # (B, N, 3, D*fH*fW)

        # Add homogeneous coordinate
        ones = torch.ones(B, N, 1, cam_points.shape[-1], device=cam_points.device)
        cam_points_h = torch.cat([cam_points, ones], dim=2)  # (B, N, 4, D*fH*fW)

        # For LSS, extrinsics are camera-to-ego (inverse of world-to-camera)
        # Transform to ego frame: P_ego = extrinsics @ P_cam
        ego_points = torch.matmul(extrinsics, cam_points_h)  # (B, N, 4, D*fH*fW)
        ego_points = ego_points[:, :, :3, :]  # (B, N, 3, D*fH*fW)

        # Reshape
        geom = ego_points.permute(0, 1, 3, 2).reshape(B, N, D, self.feat_h, self.feat_w, 3)
        return geom

    def voxel_pooling(self, geom, features):
        """Splat 3D features into BEV voxel grid using cumulative sum trick.

        Args:
            geom: 3D point positions (B, N, D, fH, fW, 3) in ego frame.
            features: Outer product features (B, N, D, fH, fW, C).

        Returns:
            BEV feature map (B, C, bev_h, bev_w).
        """
        B, N, D, fH, fW, C = features.shape
        device = features.device

        # Compute voxel indices
        # x -> column index in BEV
        voxel_x = ((geom[..., 0] - self.xbound[0]) / self.xbound[2]).long()
        # y -> row index in BEV
        voxel_y = ((geom[..., 1] - self.ybound[0]) / self.ybound[2]).long()
        # z -> height index
        voxel_z = ((geom[..., 2] - self.zbound[0]) / self.zbound[2]).long()

        # Valid mask
        valid = (
            (voxel_x >= 0) & (voxel_x < self.bev_w)
            & (voxel_y >= 0) & (voxel_y < self.bev_h)
            & (voxel_z >= 0) & (voxel_z < self.bev_z)
        )  # (B, N, D, fH, fW)

        # Flatten spatial dimensions
        Npixels = N * D * fH * fW

        bev_feature = torch.zeros(B, self.bev_h, self.bev_w, C, device=device)

        for b in range(B):
            # Get valid points for this batch
            mask = valid[b].reshape(-1)  # (Npixels,)
            cur_x = voxel_x[b].reshape(-1)[mask]  # (num_valid,)
            cur_y = voxel_y[b].reshape(-1)[mask]
            cur_feat = features[b].reshape(Npixels, C)[mask]  # (num_valid, C)

            # Compute linear indices for scatter
            indices = cur_y * self.bev_w + cur_x  # (num_valid,)

            # Scatter add features into BEV grid
            bev_flat = torch.zeros(self.bev_h * self.bev_w, C, device=device)
            indices_expanded = indices.unsqueeze(-1).expand(-1, C)
            bev_flat.scatter_add_(0, indices_expanded, cur_feat)

            bev_feature[b] = bev_flat.reshape(self.bev_h, self.bev_w, C)

        # (B, bev_h, bev_w, C) -> (B, C, bev_h, bev_w)
        bev_feature = bev_feature.permute(0, 3, 1, 2)
        return bev_feature

    def forward(self, features, intrinsics, extrinsics):
        """
        Args:
            features: Multi-camera features (B, N, C, fH, fW).
            intrinsics: Camera intrinsic matrices (B, N, 3, 3).
            extrinsics: Camera extrinsic matrices (B, N, 4, 4) - camera to ego.

        Returns:
            BEV feature map (B, C, bev_h, bev_w).
        """
        B, N, C, fH, fW = features.shape

        # Reshape for camera encoding
        x = features.reshape(B * N, C, fH, fW)

        # Predict depth distribution and context features
        depth, context = self.cam_encode(x)  # depth: (B*N, D, fH, fW), context: (B*N, C, fH, fW)

        D = depth.shape[1]
        context = context.reshape(B, N, C, fH, fW)
        depth = depth.reshape(B, N, D, fH, fW)

        # Outer product of depth and context: creates volumetric features
        # depth: (B, N, D, fH, fW) -> (B, N, D, fH, fW, 1)
        # context: (B, N, C, fH, fW) -> (B, N, 1, fH, fW, C)
        volume = depth.unsqueeze(-1) * context.permute(0, 1, 3, 4, 2).unsqueeze(2)
        # volume: (B, N, D, fH, fW, C)

        # Get 3D geometry
        geom = self.get_geometry(intrinsics, extrinsics)  # (B, N, D, fH, fW, 3)

        # Splat into BEV
        bev = self.voxel_pooling(geom, volume)  # (B, C, bev_h, bev_w)

        return bev
