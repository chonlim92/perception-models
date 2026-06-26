"""
3D Position Embedding for PETR.

This module implements the key innovation of PETR: encoding 3D world
coordinates into image features via position-aware embeddings. For each
pixel in the multi-view images, a frustum of 3D points is generated along
the depth axis, transformed to the ego/world coordinate frame using camera
intrinsics and extrinsics, and then encoded via a learnable MLP to produce
position embeddings that are added to the image features.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionEmbedding3D(nn.Module):
    """3D-aware position embedding for PETR.

    Generates camera frustum coordinates for each pixel and depth bin,
    transforms them to the ego/world coordinate frame, and encodes the
    3D positions using an MLP to produce position-aware feature embeddings.

    Args:
        embed_dims: Dimension of the position embedding output (should
            match the image feature channel dimension).
        num_depth_bins: Number of depth bins to sample along each ray.
        depth_start: Near plane depth in meters.
        depth_end: Far plane depth in meters.
        depth_distribution: How to distribute depth bins ('linear' or 'log').
        with_multiview: If True, process multi-view images jointly.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_depth_bins: int = 64,
        depth_start: float = 1.0,
        depth_end: float = 60.0,
        depth_distribution: str = "linear",
        with_multiview: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_depth_bins = num_depth_bins
        self.depth_start = depth_start
        self.depth_end = depth_end
        self.depth_distribution = depth_distribution
        self.with_multiview = with_multiview

        # Generate fixed depth bins
        if depth_distribution == "linear":
            depths = torch.linspace(depth_start, depth_end, num_depth_bins)
        elif depth_distribution == "log":
            depths = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(depth_start)),
                    torch.log(torch.tensor(depth_end)),
                    num_depth_bins,
                )
            )
        else:
            raise ValueError(
                f"Unknown depth distribution: {depth_distribution}. "
                "Use 'linear' or 'log'."
            )
        self.register_buffer("depth_bins", depths)  # (D,)

        # MLP to encode 3D coordinates -> position embeddings
        # Input: 3D coordinates (x, y, z) = 3 channels
        # Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear
        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize MLP weights with Xavier uniform."""
        for m in self.position_encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def generate_frustum(
        self, feat_h: int, feat_w: int, device: torch.device
    ) -> torch.Tensor:
        """Generate a frustum of 3D points in normalized image coordinates.

        For each spatial location (u, v) and each depth bin d, produce a
        3D point (u*d, v*d, d) in camera coordinates (before unprojection).

        Args:
            feat_h: Height of the feature map.
            feat_w: Width of the feature map.
            device: Device to create tensors on.

        Returns:
            Frustum coordinates of shape (D, H, W, 3) where the last dim
            is (u_norm * d, v_norm * d, d). u_norm and v_norm are in [0, 1].
        """
        # Create grid of normalized pixel coordinates [0, 1]
        us = torch.linspace(0.5 / feat_w, 1.0 - 0.5 / feat_w, feat_w, device=device)
        vs = torch.linspace(0.5 / feat_h, 1.0 - 0.5 / feat_h, feat_h, device=device)
        depths = self.depth_bins.to(device)  # (D,)

        # Create meshgrid: (D, H, W) for each of u, v, d
        # v_grid: (H,), u_grid: (W,), d_grid: (D,)
        d_grid, v_grid, u_grid = torch.meshgrid(
            depths, vs, us, indexing="ij"
        )  # each (D, H, W)

        # Frustum points: normalized pixel coords scaled by depth
        # This represents rays from camera center through each pixel
        frustum = torch.stack(
            [u_grid * d_grid, v_grid * d_grid, d_grid], dim=-1
        )  # (D, H, W, 3)

        return frustum

    def frustum_to_world(
        self,
        frustum: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """Transform frustum points from normalized image coords to world coords.

        The transformation pipeline:
        1. Scale normalized coords (u*d, v*d, d) to pixel coords (px*d, py*d, d)
        2. Unproject using inverse intrinsics: K^{-1} @ [px*d, py*d, d]^T -> cam coords
        3. Transform to world using extrinsics: T_world_cam @ [x, y, z, 1]^T

        Args:
            frustum: Frustum points (D, H, W, 3) in normalized image coords.
            intrinsics: Camera intrinsic matrices (B, N, 3, 3) or (B, N, 4, 4).
            extrinsics: Camera-to-world (cam2ego) transforms (B, N, 4, 4).
            img_h: Original image height (before feature extraction).
            img_w: Original image width (before feature extraction).

        Returns:
            World coordinates of shape (B, N, D, H, W, 3).
        """
        batch_size, num_cams = intrinsics.shape[:2]
        D, H, W, _ = frustum.shape
        device = frustum.device

        # Scale from normalized [0,1] to pixel coordinates
        # frustum[..., 0] = u_norm * d -> pixel_x * d = u_norm * img_w * d / d * d
        # Actually: frustum stores (u_norm*d, v_norm*d, d)
        # pixel coords: px = u_norm * img_w, py = v_norm * img_h
        # So pixel_homogeneous * d = (px*d, py*d, d) = (u_norm*img_w*d, v_norm*img_h*d, d)
        scale = torch.tensor(
            [img_w, img_h, 1.0], device=device, dtype=frustum.dtype
        )
        frustum_pixel = frustum * scale  # (D, H, W, 3): (px*d, py*d, d)

        # Reshape for batch matrix operations
        # (D*H*W, 3) -> will broadcast with batch dims
        frustum_flat = frustum_pixel.reshape(-1, 3)  # (D*H*W, 3)

        # Extract 3x3 intrinsics if 4x4 was provided
        K = intrinsics[..., :3, :3]  # (B, N, 3, 3)

        # Compute inverse intrinsics
        K_inv = torch.inverse(K)  # (B, N, 3, 3)

        # Unproject: cam_coords = K_inv @ (px*d, py*d, d)^T
        # This gives 3D points in camera frame
        # (B, N, 3, 3) @ (D*H*W, 3, 1) -> need broadcasting
        frustum_expanded = frustum_flat.unsqueeze(0).unsqueeze(0)  # (1, 1, D*H*W, 3)
        frustum_expanded = frustum_expanded.expand(
            batch_size, num_cams, -1, -1
        )  # (B, N, D*H*W, 3)

        # Matrix multiply: K_inv @ frustum^T -> camera coords
        # K_inv: (B, N, 3, 3), frustum: (B, N, D*H*W, 3)
        cam_coords = torch.einsum(
            "bnij,bnpj->bnpi", K_inv, frustum_expanded
        )  # (B, N, D*H*W, 3)

        # Transform to world coordinates using extrinsics (cam2ego/cam2world)
        # Append homogeneous coordinate
        ones = torch.ones(
            *cam_coords.shape[:-1], 1, device=device, dtype=cam_coords.dtype
        )
        cam_coords_homo = torch.cat([cam_coords, ones], dim=-1)  # (B, N, D*H*W, 4)

        # Apply extrinsics: world = T_cam2world @ cam_homo
        # extrinsics: (B, N, 4, 4), cam_coords_homo: (B, N, D*H*W, 4)
        world_coords = torch.einsum(
            "bnij,bnpj->bnpi", extrinsics, cam_coords_homo
        )  # (B, N, D*H*W, 4)

        # Take xyz (drop homogeneous w)
        world_coords = world_coords[..., :3]  # (B, N, D*H*W, 3)

        # Reshape back to spatial layout
        world_coords = world_coords.reshape(
            batch_size, num_cams, D, H, W, 3
        )

        return world_coords

    def normalize_coords(
        self,
        coords: torch.Tensor,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> torch.Tensor:
        """Normalize world coordinates to [0, 1] range using point cloud range.

        Args:
            coords: World coordinates (..., 3).
            pc_range: (x_min, y_min, z_min, x_max, y_max, z_max) defining
                the perception range in world coordinates.

        Returns:
            Normalized coordinates (..., 3) in [0, 1].
        """
        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        mins = torch.tensor(
            [x_min, y_min, z_min], device=coords.device, dtype=coords.dtype
        )
        maxs = torch.tensor(
            [x_max, y_max, z_max], device=coords.device, dtype=coords.dtype
        )
        coords_norm = (coords - mins) / (maxs - mins)
        # Clamp to [0, 1] to avoid out-of-range embeddings
        coords_norm = coords_norm.clamp(0.0, 1.0)
        return coords_norm

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        img_shape: Tuple[int, int],
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> torch.Tensor:
        """Generate 3D position embeddings and add to image features.

        Args:
            features: Image features of shape (B, N_cams, C, H, W) where
                C should equal embed_dims.
            intrinsics: Camera intrinsic matrices (B, N_cams, 3, 3) or (B, N, 4, 4).
            extrinsics: Camera-to-ego transforms (B, N_cams, 4, 4).
            img_shape: Original image (H, W) before backbone downsampling.
            pc_range: Point cloud range for coordinate normalization.

        Returns:
            Position-aware features of shape (B, N_cams, C, D*H*W) where
            features have been augmented with learned 3D position embeddings.
            The D (depth) dimension is folded into the spatial dimension.
        """
        B, N, C, H, W = features.shape
        device = features.device
        img_h, img_w = img_shape

        # Step 1: Generate frustum in normalized image coordinates
        frustum = self.generate_frustum(H, W, device)  # (D, H, W, 3)

        # Step 2: Transform to world coordinates
        world_coords = self.frustum_to_world(
            frustum, intrinsics, extrinsics, img_h, img_w
        )  # (B, N, D, H, W, 3)

        # Step 3: Normalize coordinates to [0, 1]
        world_coords_norm = self.normalize_coords(world_coords, pc_range)

        # Step 4: Encode 3D positions with MLP
        # world_coords_norm: (B, N, D, H, W, 3)
        pos_embed = self.position_encoder(world_coords_norm)  # (B, N, D, H, W, C)

        # Step 5: Combine features with position embeddings
        # Expand features along depth dimension and add position embeddings
        # features: (B, N, C, H, W) -> (B, N, C, 1, H, W) -> (B, N, C, D, H, W)
        D = self.num_depth_bins
        features_expanded = features.unsqueeze(3).expand(
            B, N, C, D, H, W
        )  # (B, N, C, D, H, W)

        # pos_embed: (B, N, D, H, W, C) -> (B, N, C, D, H, W)
        pos_embed = pos_embed.permute(0, 1, 5, 2, 3, 4)  # (B, N, C, D, H, W)

        # Add position embeddings to features
        pos_aware_features = features_expanded + pos_embed  # (B, N, C, D, H, W)

        # Flatten spatial dimensions: (B, N, C, D*H*W)
        pos_aware_features = pos_aware_features.reshape(B, N, C, D * H * W)

        return pos_aware_features

    def get_coords_for_queries(
        self,
        reference_points: torch.Tensor,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> torch.Tensor:
        """Encode 3D reference point positions for object queries.

        Used to generate position embeddings for learnable object queries
        based on their predicted 3D reference points.

        Args:
            reference_points: Normalized 3D reference points (B, Q, 3)
                in [0, 1] range.
            pc_range: Point cloud range for denormalization (if needed).

        Returns:
            Position embeddings for queries (B, Q, C).
        """
        return self.position_encoder(reference_points)
