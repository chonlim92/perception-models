"""
DETR3D Feature Sampling: 3D-to-2D projection and bilinear feature sampling.

Projects 3D reference points onto multi-view camera images using camera
intrinsics and extrinsics, then samples features from multi-scale feature
maps via bilinear interpolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def project_points_to_cameras(
    reference_points: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D reference points to all camera image planes.

    Transforms 3D points from world/ego coordinate frame to each camera's
    image plane using the extrinsic (lidar2cam) and intrinsic matrices.

    Args:
        reference_points: 3D points in world/ego frame, shape (N, 3).
        intrinsics: Camera intrinsic matrices, shape (num_cams, 3, 3).
        extrinsics: Lidar-to-camera transformation matrices (world-to-cam),
                    shape (num_cams, 4, 4).
        image_shape: (H, W) of the camera images.

    Returns:
        pixel_coords: 2D pixel coordinates, shape (num_cams, N, 2) in (u, v) format.
        valid_mask: Boolean mask indicating valid projections (depth > 0 and
                    within image bounds), shape (num_cams, N).
    """
    num_cams = intrinsics.shape[0]
    num_points = reference_points.shape[0]
    img_h, img_w = image_shape
    device = reference_points.device

    # Convert reference points to homogeneous coordinates: (N, 4)
    ones = torch.ones(num_points, 1, device=device, dtype=reference_points.dtype)
    points_homo = torch.cat([reference_points, ones], dim=1)  # (N, 4)

    # Transform to camera coordinates for all cameras
    # extrinsics: (num_cams, 4, 4), points_homo: (N, 4)
    # Result: (num_cams, N, 4)
    points_cam = torch.einsum('cij,nj->cni', extrinsics, points_homo)  # (num_cams, N, 4)

    # Extract x, y, z in camera frame (first 3 components)
    points_cam_xyz = points_cam[..., :3]  # (num_cams, N, 3)

    # Depth is z coordinate in camera frame
    depth = points_cam_xyz[..., 2]  # (num_cams, N)

    # Project to image plane using intrinsics: pixel = K @ point_cam
    # intrinsics: (num_cams, 3, 3), points_cam_xyz: (num_cams, N, 3)
    pixels = torch.einsum('cij,cnj->cni', intrinsics, points_cam_xyz)  # (num_cams, N, 3)

    # Normalize by depth (perspective division)
    # Avoid division by zero
    eps = 1e-5
    depth_safe = depth.clone()
    depth_safe[depth_safe.abs() < eps] = eps

    pixel_coords = pixels[..., :2] / depth_safe.unsqueeze(-1)  # (num_cams, N, 2)

    # Validity check: depth > 0 and within image bounds
    valid_depth = depth > eps
    valid_u = (pixel_coords[..., 0] >= 0) & (pixel_coords[..., 0] < img_w)
    valid_v = (pixel_coords[..., 1] >= 0) & (pixel_coords[..., 1] < img_h)
    valid_mask = valid_depth & valid_u & valid_v  # (num_cams, N)

    return pixel_coords, valid_mask


def normalize_pixel_coords(
    pixel_coords: torch.Tensor,
    image_shape: Tuple[int, int],
) -> torch.Tensor:
    """Normalize pixel coordinates from [0, W/H) to [-1, 1] for grid_sample.

    Args:
        pixel_coords: Pixel coordinates (u, v), shape (..., 2).
        image_shape: (H, W) of the feature map.

    Returns:
        Normalized coordinates in [-1, 1], shape (..., 2).
    """
    img_h, img_w = image_shape
    normalized = pixel_coords.clone()
    normalized[..., 0] = 2.0 * normalized[..., 0] / (img_w - 1) - 1.0  # u -> x in [-1, 1]
    normalized[..., 1] = 2.0 * normalized[..., 1] / (img_h - 1) - 1.0  # v -> y in [-1, 1]
    return normalized


def feature_sampling(
    pixel_coords: torch.Tensor,
    valid_mask: torch.Tensor,
    multi_scale_features: List[torch.Tensor],
    image_shape: Tuple[int, int],
) -> torch.Tensor:
    """Sample features from multi-scale feature maps at projected 2D locations.

    Uses bilinear interpolation via F.grid_sample to extract features at the
    projected pixel locations. Aggregates features across cameras using a
    weighted sum based on the validity mask, and across feature levels via mean.

    Args:
        pixel_coords: 2D pixel coordinates, shape (num_cams, N, 2).
        valid_mask: Validity mask, shape (num_cams, N).
        multi_scale_features: List of L feature tensors, each of shape
                              (B, num_cams, C, H_l, W_l) where H_l, W_l are the
                              spatial dimensions at level l.
        image_shape: (H, W) of the original input images (used to scale
                     pixel coordinates to each feature level).

    Returns:
        Sampled features aggregated across cameras and levels,
        shape (B, N, C) where C is the feature channel dimension.
    """
    num_cams, num_points, _ = pixel_coords.shape
    num_levels = len(multi_scale_features)
    batch_size = multi_scale_features[0].shape[0]
    feat_channels = multi_scale_features[0].shape[2]
    device = pixel_coords.device
    img_h, img_w = image_shape

    # Accumulator for features across levels
    all_level_features = torch.zeros(
        batch_size, num_points, feat_channels, device=device, dtype=multi_scale_features[0].dtype
    )

    for level_idx in range(num_levels):
        # Feature map at this level: (B, num_cams, C, H_l, W_l)
        feat_map = multi_scale_features[level_idx]
        _, _, C, feat_h, feat_w = feat_map.shape

        # Scale pixel coordinates to feature map resolution
        scale_u = feat_w / img_w
        scale_v = feat_h / img_h
        scaled_coords = pixel_coords.clone()  # (num_cams, N, 2)
        scaled_coords[..., 0] = scaled_coords[..., 0] * scale_u
        scaled_coords[..., 1] = scaled_coords[..., 1] * scale_v

        # Normalize to [-1, 1] for grid_sample
        norm_coords = normalize_pixel_coords(scaled_coords, (feat_h, feat_w))  # (num_cams, N, 2)

        # Reshape for grid_sample: need (B*num_cams, C, H_l, W_l) features
        # and (B*num_cams, 1, N, 2) grid
        feat_flat = feat_map.reshape(batch_size * num_cams, C, feat_h, feat_w)

        # Expand grid for batch: (num_cams, N, 2) -> (B*num_cams, 1, N, 2)
        grid = norm_coords.unsqueeze(1)  # (num_cams, 1, N, 2)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # (B, num_cams, 1, N, 2)
        grid = grid.reshape(batch_size * num_cams, 1, num_points, 2)

        # Bilinear sampling
        # F.grid_sample expects grid in (x, y) format which is (u_norm, v_norm)
        sampled = F.grid_sample(
            feat_flat, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )  # (B*num_cams, C, 1, N)

        # Reshape back: (B, num_cams, C, N)
        sampled = sampled.reshape(batch_size, num_cams, C, num_points)
        sampled = sampled.permute(0, 1, 3, 2)  # (B, num_cams, N, C)

        # Apply validity mask: (num_cams, N) -> (1, num_cams, N, 1) for broadcasting
        mask = valid_mask.unsqueeze(0).unsqueeze(-1).float()  # (1, num_cams, N, 1)
        mask = mask.expand(batch_size, -1, -1, -1)

        # Weighted sum across cameras
        masked_sampled = sampled * mask  # (B, num_cams, N, C)
        # Sum across cameras and normalize by number of valid cameras per point
        cam_sum = masked_sampled.sum(dim=1)  # (B, N, C)
        cam_count = mask.sum(dim=1).clamp(min=1.0)  # (B, N, 1)
        level_features = cam_sum / cam_count  # (B, N, C)

        all_level_features = all_level_features + level_features

    # Average across levels
    all_level_features = all_level_features / num_levels

    return all_level_features


class DETR3DFeatureSampler(nn.Module):
    """Module that encapsulates the full 3D-to-2D projection and feature sampling.

    Given 3D reference points and multi-view camera parameters, projects points
    to all camera views and samples multi-scale features via bilinear interpolation.
    """

    def __init__(self, embed_dims: int = 256):
        """
        Args:
            embed_dims: Dimensionality of the output features (must match FPN channels).
        """
        super().__init__()
        self.embed_dims = embed_dims
        # Learnable projection to map sampled features to query space
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        reference_points: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            reference_points: 3D reference points, shape (B, N, 3) in world frame.
            multi_scale_features: List of L feature maps, each (B, num_cams, C, H_l, W_l).
            intrinsics: Camera intrinsic matrices, shape (B, num_cams, 3, 3).
            extrinsics: World-to-camera transforms, shape (B, num_cams, 4, 4).
            image_shape: (H, W) of original input images.

        Returns:
            Sampled and projected features, shape (B, N, embed_dims).
        """
        batch_size, num_queries, _ = reference_points.shape
        device = reference_points.device

        sampled_features_batch = []

        for b in range(batch_size):
            # Get camera params for this batch element
            pts = reference_points[b]  # (N, 3)
            K = intrinsics[b]  # (num_cams, 3, 3)
            E = extrinsics[b]  # (num_cams, 4, 4)

            # Project 3D points to all cameras
            pixel_coords, valid_mask = project_points_to_cameras(pts, K, E, image_shape)

            # Get features for this batch element
            batch_features = [feat[b:b+1] for feat in multi_scale_features]

            # Sample features
            sampled = feature_sampling(
                pixel_coords, valid_mask, batch_features, image_shape
            )  # (1, N, C)

            sampled_features_batch.append(sampled.squeeze(0))

        # Stack batch: (B, N, C)
        sampled_features = torch.stack(sampled_features_batch, dim=0)

        # Project and add positional encoding
        output = self.output_proj(sampled_features)
        pos_encoding = self.position_encoder(reference_points)
        output = output + pos_encoding

        return output
