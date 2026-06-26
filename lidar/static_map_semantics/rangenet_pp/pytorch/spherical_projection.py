"""Spherical projection: Point cloud <-> Range image conversion.

Implements the projection used in RangeNet++ to convert 3D LiDAR point clouds
into 2D range images and back. The spherical projection maps each 3D point
to a pixel in the range image based on its azimuth (yaw) and elevation (pitch).
"""

import numpy as np
import torch
from typing import Tuple, Optional


class SphericalProjection:
    """Handles spherical projection of point clouds to range images.

    The projection uses the LiDAR's field of view parameters to map 3D points
    to a 2D grid. For Velodyne HDL-64E (SemanticKITTI):
        - fov_up = 2.0 degrees
        - fov_down = -24.8 degrees
        - Total vertical FOV = 26.8 degrees
    """

    def __init__(
        self,
        height: int = 64,
        width: int = 2048,
        fov_up: float = 2.0,
        fov_down: float = -24.8,
    ):
        """
        Args:
            height: Range image height (number of laser beams).
            width: Range image width (horizontal resolution).
            fov_up: Upper field of view limit in degrees.
            fov_down: Lower field of view limit in degrees.
        """
        self.height = height
        self.width = width
        self.fov_up = np.deg2rad(fov_up)
        self.fov_down = np.deg2rad(fov_down)
        self.fov_total = self.fov_up - self.fov_down  # positive value

    def project_points_to_range_image(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project a point cloud to a range image using spherical coordinates.

        Args:
            points: Point cloud array of shape (N, 4) with columns [x, y, z, intensity].
                    Coordinates are in the LiDAR frame.

        Returns:
            range_image: (5, H, W) float32 array with channels [range, x, y, z, intensity].
                         Invalid pixels are 0.
            label_map: (H, W) int32 array mapping each pixel to the point index (-1 for empty).
            point_to_pixel: (N, 2) int32 array mapping each point to (row, col) in the image.
                            Points not projected get (-1, -1).
        """
        N = points.shape[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3]

        # Compute range (distance from sensor)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # Filter out points with zero range (sensor origin)
        valid_mask = r > 1e-6

        # Compute azimuth angle (yaw) and elevation angle (pitch)
        yaw = np.arctan2(y, x)  # [-pi, pi]
        pitch = np.arcsin(np.clip(z / np.maximum(r, 1e-8), -1.0, 1.0))  # [-pi/2, pi/2]

        # Map angles to pixel coordinates
        # Column: yaw normalized to [0, 1] -> [0, W-1]
        # yaw goes from -pi to pi, we want left=pi, right=-pi (wrapping around)
        col = 0.5 * (1.0 + yaw / np.pi) * self.width
        col = np.floor(col).astype(np.int32)
        col = np.clip(col, 0, self.width - 1)

        # Row: pitch normalized within FOV -> [0, H-1]
        # fov_up is the top of the image (row 0), fov_down is the bottom (row H-1)
        row = (1.0 - (pitch - self.fov_down) / self.fov_total) * self.height
        row = np.floor(row).astype(np.int32)
        row = np.clip(row, 0, self.height - 1)

        # Filter points outside the vertical FOV
        in_fov = valid_mask & (pitch >= self.fov_down) & (pitch <= self.fov_up)

        # Initialize outputs
        range_image = np.zeros((5, self.height, self.width), dtype=np.float32)
        pixel_to_point = np.full((self.height, self.width), -1, dtype=np.int32)
        point_to_pixel = np.full((N, 2), -1, dtype=np.int32)

        # Depth ordering: process points from farthest to nearest so that
        # closer points overwrite farther ones in case of overlap
        order = np.argsort(-r)

        for idx in order:
            if not in_fov[idx]:
                continue
            ri = row[idx]
            ci = col[idx]

            range_image[0, ri, ci] = r[idx]
            range_image[1, ri, ci] = x[idx]
            range_image[2, ri, ci] = y[idx]
            range_image[3, ri, ci] = z[idx]
            range_image[4, ri, ci] = intensity[idx]

            pixel_to_point[ri, ci] = idx
            point_to_pixel[idx, 0] = ri
            point_to_pixel[idx, 1] = ci

        return range_image, pixel_to_point, point_to_pixel

    def project_points_to_range_image_fast(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized (fast) projection of point cloud to range image.

        Same interface as project_points_to_range_image but uses vectorized
        numpy operations for speed. Uses closest-point-wins policy.

        Args:
            points: (N, 4) array [x, y, z, intensity].

        Returns:
            range_image: (5, H, W) float32
            pixel_to_point: (H, W) int32, point index per pixel (-1 if empty)
            point_to_pixel: (N, 2) int32, (row, col) per point (-1 if not projected)
        """
        N = points.shape[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3]

        # Compute spherical coordinates
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        valid_mask = r > 1e-6

        yaw = np.arctan2(y, x)
        pitch = np.arcsin(np.clip(z / np.maximum(r, 1e-8), -1.0, 1.0))

        # Map to pixel coordinates
        col = np.floor(0.5 * (1.0 + yaw / np.pi) * self.width).astype(np.int32)
        col = np.clip(col, 0, self.width - 1)

        row = np.floor((1.0 - (pitch - self.fov_down) / self.fov_total) * self.height).astype(np.int32)
        row = np.clip(row, 0, self.height - 1)

        # Valid: within range and within vertical FOV
        in_fov = valid_mask & (pitch >= self.fov_down) & (pitch <= self.fov_up)

        # Initialize outputs
        range_image = np.zeros((5, self.height, self.width), dtype=np.float32)
        pixel_to_point = np.full((self.height, self.width), -1, dtype=np.int32)
        point_to_pixel = np.full((N, 2), -1, dtype=np.int32)

        # Get valid indices sorted by range (farthest first so closest overwrites)
        valid_indices = np.where(in_fov)[0]
        sorted_by_range = valid_indices[np.argsort(-r[valid_indices])]

        valid_rows = row[sorted_by_range]
        valid_cols = col[sorted_by_range]
        valid_r = r[sorted_by_range]
        valid_x = x[sorted_by_range]
        valid_y = y[sorted_by_range]
        valid_z = z[sorted_by_range]
        valid_intensity = intensity[sorted_by_range]

        # Write to range image (last write wins = closest point)
        range_image[0, valid_rows, valid_cols] = valid_r
        range_image[1, valid_rows, valid_cols] = valid_x
        range_image[2, valid_rows, valid_cols] = valid_y
        range_image[3, valid_rows, valid_cols] = valid_z
        range_image[4, valid_rows, valid_cols] = valid_intensity

        pixel_to_point[valid_rows, valid_cols] = sorted_by_range
        point_to_pixel[sorted_by_range, 0] = valid_rows
        point_to_pixel[sorted_by_range, 1] = valid_cols

        return range_image, pixel_to_point, point_to_pixel

    def range_image_to_points(
        self,
        range_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Back-project range image pixels to 3D points.

        Converts the range image representation back to a 3D point cloud.
        Uses the stored x, y, z channels directly if available.

        Args:
            range_image: (5, H, W) array with channels [range, x, y, z, intensity].

        Returns:
            points: (M, 4) array of [x, y, z, intensity] for valid pixels.
            pixel_indices: (M, 2) array of (row, col) for each returned point.
        """
        r_channel = range_image[0]
        valid_mask = r_channel > 1e-6

        rows, cols = np.where(valid_mask)
        M = len(rows)

        points = np.zeros((M, 4), dtype=np.float32)
        points[:, 0] = range_image[1, rows, cols]  # x
        points[:, 1] = range_image[2, rows, cols]  # y
        points[:, 2] = range_image[3, rows, cols]  # z
        points[:, 3] = range_image[4, rows, cols]  # intensity

        pixel_indices = np.stack([rows, cols], axis=1).astype(np.int32)

        return points, pixel_indices

    def range_image_to_points_from_angles(
        self,
        range_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Back-project using spherical coordinates (when x,y,z channels unavailable).

        Reconstructs 3D coordinates from range values and pixel positions.

        Args:
            range_image: (5, H, W) or at minimum (1, H, W) with range channel.

        Returns:
            points: (M, 4) array with [x, y, z, intensity].
            pixel_indices: (M, 2) array of (row, col).
        """
        r_channel = range_image[0]
        valid_mask = r_channel > 1e-6

        rows, cols = np.where(valid_mask)
        M = len(rows)

        # Reconstruct angles from pixel coordinates
        # col -> yaw
        yaw = ((cols.astype(np.float32) / self.width) * 2.0 - 1.0) * np.pi
        # row -> pitch
        pitch = self.fov_down + (1.0 - rows.astype(np.float32) / self.height) * self.fov_total

        r_values = r_channel[rows, cols]

        # Convert spherical to cartesian
        x = r_values * np.cos(pitch) * np.cos(yaw)
        y = r_values * np.cos(pitch) * np.sin(yaw)
        z = r_values * np.sin(pitch)

        points = np.zeros((M, 4), dtype=np.float32)
        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z
        if range_image.shape[0] >= 5:
            points[:, 3] = range_image[4, rows, cols]

        pixel_indices = np.stack([rows, cols], axis=1).astype(np.int32)
        return points, pixel_indices


class SphericalProjectionTorch:
    """PyTorch-based spherical projection for use in differentiable pipelines.

    Provides GPU-accelerated projection without loops.
    """

    def __init__(
        self,
        height: int = 64,
        width: int = 2048,
        fov_up: float = 2.0,
        fov_down: float = -24.8,
    ):
        self.height = height
        self.width = width
        self.fov_up = torch.deg2rad(torch.tensor(fov_up))
        self.fov_down = torch.deg2rad(torch.tensor(fov_down))
        self.fov_total = self.fov_up - self.fov_down

    def project(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project point cloud to range image using PyTorch operations.

        Args:
            points: (N, 4) tensor [x, y, z, intensity] on any device.

        Returns:
            range_image: (5, H, W) tensor.
            pixel_to_point: (H, W) long tensor with point indices (-1 for empty).
            point_to_pixel: (N, 2) long tensor with (row, col) per point (-1 if not projected).
        """
        device = points.device
        N = points.shape[0]

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3]

        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        valid = r > 1e-6

        yaw = torch.atan2(y, x)
        pitch = torch.asin(torch.clamp(z / torch.clamp(r, min=1e-8), -1.0, 1.0))

        fov_down = self.fov_down.to(device)
        fov_up = self.fov_up.to(device)
        fov_total = self.fov_total.to(device)

        col = torch.floor(0.5 * (1.0 + yaw / torch.pi) * self.width).long()
        col = torch.clamp(col, 0, self.width - 1)

        row = torch.floor((1.0 - (pitch - fov_down) / fov_total) * self.height).long()
        row = torch.clamp(row, 0, self.height - 1)

        in_fov = valid & (pitch >= fov_down) & (pitch <= fov_up)

        # Initialize outputs
        range_image = torch.zeros(5, self.height, self.width, device=device, dtype=torch.float32)
        pixel_to_point = torch.full((self.height, self.width), -1, device=device, dtype=torch.long)
        point_to_pixel = torch.full((N, 2), -1, device=device, dtype=torch.long)

        # Get valid point indices sorted by range (farthest first)
        valid_indices = torch.where(in_fov)[0]
        valid_ranges = r[valid_indices]
        sorted_order = torch.argsort(valid_ranges, descending=True)
        sorted_indices = valid_indices[sorted_order]

        v_rows = row[sorted_indices]
        v_cols = col[sorted_indices]

        # Write data (last write wins = closest point)
        range_image[0, v_rows, v_cols] = r[sorted_indices]
        range_image[1, v_rows, v_cols] = x[sorted_indices]
        range_image[2, v_rows, v_cols] = y[sorted_indices]
        range_image[3, v_rows, v_cols] = z[sorted_indices]
        range_image[4, v_rows, v_cols] = intensity[sorted_indices]

        pixel_to_point[v_rows, v_cols] = sorted_indices
        point_to_pixel[sorted_indices, 0] = v_rows
        point_to_pixel[sorted_indices, 1] = v_cols

        return range_image, pixel_to_point, point_to_pixel
