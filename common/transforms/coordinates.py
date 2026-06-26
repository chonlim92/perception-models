"""Coordinate system conversions for autonomous driving perception.

Supports transformations between camera, LiDAR, ego vehicle, and world frames,
as well as projections to BEV and image planes. Follows conventions used in
BEVFormer/BEVDet/nuScenes: right-handed coordinate systems with Z-up for LiDAR
and ego frames, and X-right Y-down Z-forward for camera frames.
"""

from typing import Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Rotation representations
# ---------------------------------------------------------------------------


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Args:
        q: Quaternion array of shape (..., 4) in (w, x, y, z) convention.

    Returns:
        Rotation matrix of shape (..., 3, 3).
    """
    q = np.asarray(q, dtype=np.float64)
    # Normalize
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Pre-compute products
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = np.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], axis=-1)

    return rot.reshape(q.shape[:-1] + (3, 3))


def euler_to_rotation_matrix(
    roll: float, pitch: float, yaw: float, order: str = "xyz"
) -> np.ndarray:
    """Convert Euler angles (in radians) to a 3x3 rotation matrix.

    The rotation is applied in the order specified: 'xyz' means R = Rz @ Ry @ Rx
    (intrinsic rotations about X, then Y, then Z).

    Args:
        roll: Rotation about X-axis in radians.
        pitch: Rotation about Y-axis in radians.
        yaw: Rotation about Z-axis in radians.
        order: Rotation order string (e.g., 'xyz', 'zyx'). Default 'xyz'.

    Returns:
        Rotation matrix of shape (3, 3).
    """
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r],
    ])
    Ry = np.array([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p],
    ])
    Rz = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1],
    ])

    axis_map = {"x": Rx, "y": Ry, "z": Rz}

    # Intrinsic rotations: apply first axis first (leftmost multiplication last)
    R = np.eye(3)
    for axis_char in order:
        R = axis_map[axis_char] @ R
    return R


def rotation_matrix_to_euler(R: np.ndarray, order: str = "xyz") -> Tuple[float, float, float]:
    """Extract Euler angles from a rotation matrix.

    Assumes intrinsic rotation order 'xyz' (Rz @ Ry @ Rx).

    Args:
        R: Rotation matrix of shape (3, 3).
        order: Rotation order (currently supports 'xyz').

    Returns:
        Tuple of (roll, pitch, yaw) in radians.
    """
    if order != "xyz":
        raise NotImplementedError(f"Only 'xyz' order is currently supported, got '{order}'")

    # For R = Rz @ Ry @ Rx:
    # R[2,0] = -sin(pitch)
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))

    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Homogeneous transforms
# ---------------------------------------------------------------------------


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix from rotation and translation.

    Args:
        R: Rotation matrix of shape (3, 3).
        t: Translation vector of shape (3,) or (3, 1).

    Returns:
        4x4 transformation matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).flatten()[:3]
    return T


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 4x4 rigid-body transformation matrix.

    For a rigid-body transform [R | t; 0 | 1], the inverse is [R^T | -R^T t; 0 | 1].

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        4x4 inverse transformation matrix.
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Compose multiple 4x4 transformation matrices (left to right application order).

    compose_transforms(T1, T2, T3) returns T3 @ T2 @ T1.
    A point transformed by T1, then T2, then T3: p' = T3 @ T2 @ T1 @ p.

    Args:
        *transforms: Variable number of 4x4 transformation matrices.

    Returns:
        Combined 4x4 transformation matrix.
    """
    result = np.eye(4, dtype=np.float64)
    for T in transforms:
        result = np.asarray(T, dtype=np.float64) @ result
    return result


# ---------------------------------------------------------------------------
# Frame conversions
# ---------------------------------------------------------------------------


def lidar_to_camera(
    points: np.ndarray, lidar2camera: np.ndarray
) -> np.ndarray:
    """Transform points from LiDAR frame to camera frame.

    Args:
        points: Point cloud of shape (N, 3) or (N, 4) in LiDAR coordinates.
        lidar2camera: 4x4 extrinsic matrix mapping LiDAR -> camera.

    Returns:
        Points in camera frame of shape (N, 3).
    """
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]
    if points.shape[1] == 3:
        pts_hom = np.hstack([points, np.ones((N, 1))])
    else:
        pts_hom = points.copy()
        pts_hom[:, 3] = 1.0

    pts_cam = (lidar2camera @ pts_hom.T).T
    return pts_cam[:, :3]


def camera_to_lidar(
    points: np.ndarray, lidar2camera: np.ndarray
) -> np.ndarray:
    """Transform points from camera frame to LiDAR frame.

    Args:
        points: Points of shape (N, 3) in camera coordinates.
        lidar2camera: 4x4 extrinsic matrix mapping LiDAR -> camera.

    Returns:
        Points in LiDAR frame of shape (N, 3).
    """
    camera2lidar = inverse_transform(lidar2camera)
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]
    pts_hom = np.hstack([points, np.ones((N, 1))])
    pts_lidar = (camera2lidar @ pts_hom.T).T
    return pts_lidar[:, :3]


def lidar_to_ego(points: np.ndarray, lidar2ego: np.ndarray) -> np.ndarray:
    """Transform points from LiDAR frame to ego vehicle frame.

    Args:
        points: Point cloud of shape (N, 3) in LiDAR coordinates.
        lidar2ego: 4x4 transformation from LiDAR to ego frame.

    Returns:
        Points in ego frame of shape (N, 3).
    """
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]
    pts_hom = np.hstack([points, np.ones((N, 1))])
    pts_ego = (lidar2ego @ pts_hom.T).T
    return pts_ego[:, :3]


def ego_to_lidar(points: np.ndarray, lidar2ego: np.ndarray) -> np.ndarray:
    """Transform points from ego vehicle frame to LiDAR frame.

    Args:
        points: Points of shape (N, 3) in ego coordinates.
        lidar2ego: 4x4 transformation from LiDAR to ego frame.

    Returns:
        Points in LiDAR frame of shape (N, 3).
    """
    ego2lidar = inverse_transform(lidar2ego)
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]
    pts_hom = np.hstack([points, np.ones((N, 1))])
    pts_lidar = (ego2lidar @ pts_hom.T).T
    return pts_lidar[:, :3]


def ego_to_world(points: np.ndarray, ego2world: np.ndarray) -> np.ndarray:
    """Transform points from ego vehicle frame to world frame.

    Args:
        points: Points of shape (N, 3) in ego coordinates.
        ego2world: 4x4 transformation from ego to world frame.

    Returns:
        Points in world frame of shape (N, 3).
    """
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]
    pts_hom = np.hstack([points, np.ones((N, 1))])
    pts_world = (ego2world @ pts_hom.T).T
    return pts_world[:, :3]


def world_to_ego(points: np.ndarray, ego2world: np.ndarray) -> np.ndarray:
    """Transform points from world frame to ego vehicle frame.

    Args:
        points: Points of shape (N, 3) in world coordinates.
        ego2world: 4x4 transformation from ego to world frame.

    Returns:
        Points in ego frame of shape (N, 3).
    """
    world2ego = inverse_transform(ego2world)
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]
    pts_hom = np.hstack([points, np.ones((N, 1))])
    pts_ego = (world2ego @ pts_hom.T).T
    return pts_ego[:, :3]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def create_projection_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    skew: float = 0.0,
) -> np.ndarray:
    """Create a 3x3 camera intrinsic (projection) matrix.

    Args:
        fx: Focal length in pixels (x-axis).
        fy: Focal length in pixels (y-axis).
        cx: Principal point x-coordinate.
        cy: Principal point y-coordinate.
        skew: Skew coefficient (default 0).

    Returns:
        3x3 camera intrinsic matrix K.
    """
    K = np.array([
        [fx, skew, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    return K


def project_points_to_image(
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    return_depth: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Project 3D points onto a 2D image plane.

    Args:
        points_3d: Points of shape (N, 3) in the frame defined by extrinsic.
            If extrinsic is None, points are assumed to be in camera frame.
        intrinsic: 3x3 camera intrinsic matrix K.
        extrinsic: Optional 4x4 matrix transforming points from their frame to
            camera frame. If None, points are already in camera frame.
        image_shape: Optional (H, W) tuple to filter points outside the image.
        return_depth: If True, also return depth values and validity mask.

    Returns:
        If return_depth is False:
            2D pixel coordinates of shape (N, 2) as (u, v). Points behind
            the camera are set to (-1, -1).
        If return_depth is True:
            Tuple of (uv, depth, mask) where mask indicates valid projections.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    N = points_3d.shape[0]

    # Transform to camera frame if extrinsic provided
    if extrinsic is not None:
        pts_hom = np.hstack([points_3d, np.ones((N, 1))])
        pts_cam = (extrinsic @ pts_hom.T).T[:, :3]
    else:
        pts_cam = points_3d

    # Depth in camera frame (Z-forward convention)
    depth = pts_cam[:, 2]

    # Project: uv = K @ [x, y, z]^T / z
    pts_proj = (intrinsic @ pts_cam.T).T  # (N, 3)
    valid_depth = depth > 1e-5
    uv = np.full((N, 2), -1.0, dtype=np.float64)
    uv[valid_depth, 0] = pts_proj[valid_depth, 0] / depth[valid_depth]
    uv[valid_depth, 1] = pts_proj[valid_depth, 1] / depth[valid_depth]

    # Image bounds check
    mask = valid_depth.copy()
    if image_shape is not None:
        H, W = image_shape
        in_bounds = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        mask = mask & in_bounds

    if return_depth:
        return uv, depth, mask
    return uv


# ---------------------------------------------------------------------------
# BEV conversions
# ---------------------------------------------------------------------------


def world_to_bev(
    points: np.ndarray,
    bev_origin: Tuple[float, float],
    bev_resolution: float,
    bev_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Convert 3D world coordinates to BEV pixel coordinates.

    The BEV map is a top-down view with the X-axis pointing right and Y-axis
    pointing down in pixel space. The world X maps to BEV column (u) and
    world Y maps to BEV row (v).

    Args:
        points: Points of shape (N, 2) or (N, 3). Only X and Y are used.
        bev_origin: (x_min, y_min) world coordinates of the BEV map origin
            (top-left corner).
        bev_resolution: Meters per pixel.
        bev_size: Optional (H, W) of the BEV map for clamping.

    Returns:
        BEV pixel coordinates of shape (N, 2) as (col, row) i.e., (u, v).
    """
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]

    u = (x - bev_origin[0]) / bev_resolution
    v = (y - bev_origin[1]) / bev_resolution

    bev_coords = np.stack([u, v], axis=-1)

    if bev_size is not None:
        H, W = bev_size
        bev_coords[:, 0] = np.clip(bev_coords[:, 0], 0, W - 1)
        bev_coords[:, 1] = np.clip(bev_coords[:, 1], 0, H - 1)

    return bev_coords


def bev_to_world(
    bev_coords: np.ndarray,
    bev_origin: Tuple[float, float],
    bev_resolution: float,
    z: float = 0.0,
) -> np.ndarray:
    """Convert BEV pixel coordinates back to 3D world coordinates.

    Args:
        bev_coords: BEV pixel coordinates of shape (N, 2) as (col, row).
        bev_origin: (x_min, y_min) world coordinates of BEV origin.
        bev_resolution: Meters per pixel.
        z: Z-coordinate to assign (default 0 = ground plane).

    Returns:
        World coordinates of shape (N, 3).
    """
    bev_coords = np.asarray(bev_coords, dtype=np.float64)
    x = bev_coords[:, 0] * bev_resolution + bev_origin[0]
    y = bev_coords[:, 1] * bev_resolution + bev_origin[1]
    z_arr = np.full_like(x, z)
    return np.stack([x, y, z_arr], axis=-1)


# ---------------------------------------------------------------------------
# Frustum operations
# ---------------------------------------------------------------------------


def points_in_frustum(
    points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_shape: Tuple[int, int],
    near: float = 0.1,
    far: float = 100.0,
) -> np.ndarray:
    """Check which 3D points lie inside a camera frustum.

    Performs frustum culling: determines which points project within the image
    bounds and lie within the near/far depth range.

    Args:
        points: Points of shape (N, 3) in the source frame (e.g., LiDAR/world).
        intrinsic: 3x3 camera intrinsic matrix K.
        extrinsic: 4x4 transformation from source frame to camera frame.
        image_shape: (H, W) of the image.
        near: Near clipping plane distance in meters.
        far: Far clipping plane distance in meters.

    Returns:
        Boolean mask of shape (N,) indicating which points are inside the frustum.
    """
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]
    H, W = image_shape

    # Transform to camera frame
    pts_hom = np.hstack([points, np.ones((N, 1))])
    pts_cam = (extrinsic @ pts_hom.T).T[:, :3]

    depth = pts_cam[:, 2]

    # Depth check
    depth_valid = (depth >= near) & (depth <= far)

    # Project to image
    pts_proj = (intrinsic @ pts_cam.T).T
    u = pts_proj[:, 0] / np.where(depth > 1e-5, depth, 1e-5)
    v = pts_proj[:, 1] / np.where(depth > 1e-5, depth, 1e-5)

    # Bounds check
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    return depth_valid & in_image


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid-body transform to a set of 3D points.

    Args:
        points: Points of shape (N, 3).
        T: 4x4 homogeneous transformation matrix.

    Returns:
        Transformed points of shape (N, 3).
    """
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]
    pts_hom = np.hstack([points, np.ones((N, 1))])
    transformed = (T @ pts_hom.T).T
    return transformed[:, :3]
