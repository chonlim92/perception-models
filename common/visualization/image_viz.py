"""2D image overlay visualization for autonomous driving perception.

Provides functions for projecting 3D boxes onto camera images, overlaying
segmentation masks, drawing 2D detections, multi-camera grid views, depth
map visualization, lane marking projection, and video output.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Default color palette for autonomous driving classes (BGR-style for OpenCV,
# but stored as RGB tuples in [0, 255])
# ---------------------------------------------------------------------------
DEFAULT_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (31, 119, 180),
    "truck": (255, 127, 14),
    "bus": (44, 160, 44),
    "trailer": (214, 39, 40),
    "construction_vehicle": (148, 103, 189),
    "pedestrian": (227, 119, 194),
    "motorcycle": (127, 127, 127),
    "bicycle": (188, 189, 34),
    "traffic_cone": (23, 190, 207),
    "barrier": (174, 199, 232),
}

# Semantic segmentation class colors (common autonomous driving classes)
SEGMENTATION_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),         # background / unlabeled
    1: (128, 64, 128),    # road
    2: (244, 35, 232),    # sidewalk
    3: (70, 70, 70),      # building
    4: (102, 102, 156),   # wall
    5: (190, 153, 153),   # fence
    6: (153, 153, 153),   # pole
    7: (250, 170, 30),    # traffic light
    8: (220, 220, 0),     # traffic sign
    9: (107, 142, 35),    # vegetation
    10: (152, 251, 152),  # terrain
    11: (70, 130, 180),   # sky
    12: (220, 20, 60),    # person
    13: (255, 0, 0),      # rider
    14: (0, 0, 142),      # car
    15: (0, 0, 70),       # truck
    16: (0, 60, 100),     # bus
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
}


# ---------------------------------------------------------------------------
# 3D-to-2D projection utilities
# ---------------------------------------------------------------------------


def project_points_to_image(
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D image coordinates.

    Parameters
    ----------
    points_3d : np.ndarray
        Shape (N, 3) points in world or camera coordinates.
    intrinsic : np.ndarray
        Shape (3, 3) camera intrinsic matrix.
    extrinsic : np.ndarray, optional
        Shape (4, 4) world-to-camera transformation. If None, points are
        assumed already in camera frame.
    image_shape : tuple (H, W), optional
        If provided, filter out points outside image bounds.

    Returns
    -------
    points_2d : np.ndarray
        Shape (M, 2) pixel coordinates [u, v].
    valid_mask : np.ndarray
        Boolean mask of shape (N,) indicating which points are valid
        (positive depth and within image bounds).
    """
    pts = points_3d.copy()

    # Transform to camera frame if extrinsic provided
    if extrinsic is not None:
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_cam = (extrinsic @ pts_h.T).T[:, :3]
    else:
        pts_cam = pts

    # Filter points behind camera
    valid_mask = pts_cam[:, 2] > 0.1

    # Project
    pts_proj = (intrinsic @ pts_cam.T).T
    pts_2d = pts_proj[:, :2] / (pts_proj[:, 2:3] + 1e-8)

    # Filter by image bounds
    if image_shape is not None:
        h, w = image_shape
        in_bounds = (
            (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
            (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
        )
        valid_mask &= in_bounds

    return pts_2d, valid_mask


def _box_3d_corners(
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """Compute 8 corners of a 3D bounding box.

    Parameters
    ----------
    center : np.ndarray
        [x, y, z] center.
    size : np.ndarray
        [length, width, height].
    yaw : float
        Heading angle (radians) around z-axis.

    Returns
    -------
    np.ndarray
        Shape (8, 3) corner positions.
    """
    l, w, h = size[0] / 2, size[1] / 2, size[2] / 2
    corners = np.array([
        [l, w, h], [l, -w, h], [-l, -w, h], [-l, w, h],
        [l, w, -h], [l, -w, -h], [-l, -w, -h], [-l, w, -h],
    ])

    cos, sin = np.cos(yaw), np.sin(yaw)
    rot = np.array([
        [cos, -sin, 0],
        [sin, cos, 0],
        [0, 0, 1],
    ])

    return corners @ rot.T + center


# Box edges (pairs of corner indices)
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
]

# Front face edges (for heading visualization)
_FRONT_EDGES = [(0, 1), (0, 4), (1, 5), (4, 5)]


# ---------------------------------------------------------------------------
# Image visualization functions
# ---------------------------------------------------------------------------


def draw_projected_3d_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    linewidth: int = 2,
    draw_front_face: bool = True,
) -> np.ndarray:
    """Project 3D bounding boxes onto a camera image and draw wireframes.

    Parameters
    ----------
    image : np.ndarray
        Shape (H, W, 3) RGB image (uint8).
    boxes : np.ndarray
        Shape (N, 7+): [x, y, z, length, width, height, yaw, ...].
    intrinsic : np.ndarray
        Shape (3, 3) camera intrinsic matrix.
    extrinsic : np.ndarray, optional
        Shape (4, 4) world-to-camera transform.
    classes : sequence of str, optional
        Class names for color coding.
    confidences : sequence of float, optional
        Confidence scores.
    linewidth : int
        Line thickness in pixels.
    draw_front_face : bool
        If True, draw front face edges thicker to indicate heading.

    Returns
    -------
    np.ndarray
        Annotated image (copy of input).
    """
    img = image.copy()
    h, w = img.shape[:2]

    for i in range(len(boxes)):
        center = boxes[i, :3]
        size = boxes[i, 3:6]
        yaw = boxes[i, 6]

        corners_3d = _box_3d_corners(center, size, yaw)
        corners_2d, valid = project_points_to_image(
            corners_3d, intrinsic, extrinsic, image_shape=(h, w)
        )

        if valid.sum() < 4:
            continue

        # Get color
        cls_name = classes[i] if classes else "car"
        color = DEFAULT_CLASS_COLORS.get(cls_name, (255, 255, 255))

        # Draw edges
        for edge in _BOX_EDGES:
            p1_idx, p2_idx = edge
            if not (valid[p1_idx] and valid[p2_idx]):
                continue
            p1 = corners_2d[p1_idx].astype(int)
            p2 = corners_2d[p2_idx].astype(int)

            lw = linewidth
            if draw_front_face and edge in _FRONT_EDGES:
                lw = linewidth * 2

            _draw_line(img, p1, p2, color, lw)

        # Draw label
        if valid[0]:
            label_pt = corners_2d[0].astype(int)
            label = cls_name
            if confidences is not None:
                label += f" {confidences[i]:.2f}"
            _draw_text(img, label, (label_pt[0], max(0, label_pt[1] - 5)), color)

    return img


def draw_2d_boxes(
    image: np.ndarray,
    boxes_2d: np.ndarray,
    classes: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    linewidth: int = 2,
) -> np.ndarray:
    """Draw 2D bounding boxes with class labels and confidence scores.

    Parameters
    ----------
    image : np.ndarray
        Shape (H, W, 3) RGB image.
    boxes_2d : np.ndarray
        Shape (N, 4) boxes as [x_min, y_min, x_max, y_max] in pixels.
    classes : sequence of str, optional
        Class names.
    confidences : sequence of float, optional
        Detection confidence scores.
    linewidth : int
        Box line thickness.

    Returns
    -------
    np.ndarray
        Annotated image.
    """
    img = image.copy()

    for i in range(len(boxes_2d)):
        x1, y1, x2, y2 = boxes_2d[i].astype(int)
        cls_name = classes[i] if classes else "object"
        color = DEFAULT_CLASS_COLORS.get(cls_name, (255, 255, 255))

        # Draw rectangle
        _draw_rect(img, (x1, y1), (x2, y2), color, linewidth)

        # Label
        label = cls_name
        if confidences is not None:
            label += f" {confidences[i]:.2f}"

        # Background for text
        _draw_text_with_bg(img, label, (x1, y1 - 2), color)

    return img


def overlay_segmentation(
    image: np.ndarray,
    seg_mask: np.ndarray,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a semantic segmentation mask on an image with transparency.

    Parameters
    ----------
    image : np.ndarray
        Shape (H, W, 3) RGB image (uint8).
    seg_mask : np.ndarray
        Shape (H, W) integer class labels.
    class_colors : dict, optional
        Mapping from class ID to RGB color tuple. Uses SEGMENTATION_COLORS
        if not provided.
    alpha : float
        Blend factor for the overlay (0=invisible, 1=opaque).

    Returns
    -------
    np.ndarray
        Blended image.
    """
    if class_colors is None:
        class_colors = SEGMENTATION_COLORS

    color_map = np.zeros_like(image)
    for cls_id, color in class_colors.items():
        mask = seg_mask == cls_id
        color_map[mask] = color

    blended = (image.astype(np.float32) * (1 - alpha) + color_map.astype(np.float32) * alpha)
    return blended.clip(0, 255).astype(np.uint8)


def overlay_depth_map(
    image: np.ndarray,
    depth: np.ndarray,
    alpha: float = 0.6,
    cmap_name: str = "magma",
    depth_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Overlay a depth map as a colormap on the image.

    Parameters
    ----------
    image : np.ndarray
        Shape (H, W, 3) RGB image.
    depth : np.ndarray
        Shape (H, W) depth values in meters.
    alpha : float
        Blend factor.
    cmap_name : str
        Matplotlib colormap name.
    depth_range : tuple, optional
        (min_depth, max_depth) for normalization. If None, uses data range.

    Returns
    -------
    np.ndarray
        Blended image with depth overlay.
    """
    cmap = plt.get_cmap(cmap_name)

    if depth_range is None:
        d_min = depth[depth > 0].min() if (depth > 0).any() else 0.0
        d_max = depth.max()
    else:
        d_min, d_max = depth_range

    normalized = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0.0, 1.0)
    depth_colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)

    # Only overlay where depth is valid (> 0)
    valid_mask = depth > 0
    blended = image.copy().astype(np.float32)
    blended[valid_mask] = (
        blended[valid_mask] * (1 - alpha) + depth_colored[valid_mask].astype(np.float32) * alpha
    )
    return blended.clip(0, 255).astype(np.uint8)


def draw_lane_markings(
    image: np.ndarray,
    lanes: List[np.ndarray],
    intrinsic: np.ndarray,
    extrinsic: Optional[np.ndarray] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    linewidth: int = 3,
) -> np.ndarray:
    """Project and draw lane markings / map elements onto an image.

    Parameters
    ----------
    image : np.ndarray
        Shape (H, W, 3) RGB image.
    lanes : list of np.ndarray
        Each lane is shape (M, 3) polyline in 3D world coordinates.
    intrinsic : np.ndarray
        Camera intrinsic matrix (3, 3).
    extrinsic : np.ndarray, optional
        World-to-camera transform (4, 4).
    colors : list of tuple, optional
        Colors for each lane. Cycles through defaults if not enough.
    linewidth : int
        Line thickness in pixels.

    Returns
    -------
    np.ndarray
        Annotated image.
    """
    img = image.copy()
    h, w = img.shape[:2]

    default_colors = [
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (0, 255, 0), (255, 128, 0), (128, 0, 255),
    ]

    for lane_idx, lane_pts in enumerate(lanes):
        pts_2d, valid = project_points_to_image(
            lane_pts, intrinsic, extrinsic, image_shape=(h, w)
        )

        if valid.sum() < 2:
            continue

        valid_pts = pts_2d[valid].astype(int)

        if colors and lane_idx < len(colors):
            color = colors[lane_idx]
        else:
            color = default_colors[lane_idx % len(default_colors)]

        # Draw as connected line segments
        for j in range(len(valid_pts) - 1):
            _draw_line(img, valid_pts[j], valid_pts[j + 1], color, linewidth)

    return img


def create_multicamera_grid(
    images: Dict[str, np.ndarray],
    layout: Optional[List[List[str]]] = None,
    cell_size: Tuple[int, int] = (480, 270),
    title: str = "",
) -> np.ndarray:
    """Arrange multiple camera images in a grid layout.

    Parameters
    ----------
    images : dict
        Mapping from camera name to image array (H, W, 3).
    layout : list of list of str, optional
        2D grid of camera names. Default layout for 6 surround cameras:
        [[front_left, front, front_right],
         [back_left,  back,  back_right]]
    cell_size : tuple (width, height)
        Target size for each image cell in pixels.
    title : str
        Optional title text.

    Returns
    -------
    np.ndarray
        Composed grid image.
    """
    if layout is None:
        # Default surround camera layout
        available = list(images.keys())
        if len(available) == 6:
            layout = [
                [available[0], available[1], available[2]],
                [available[3], available[4], available[5]],
            ]
        else:
            # Auto-arrange in rows of 3
            n_cols = min(3, len(available))
            n_rows = (len(available) + n_cols - 1) // n_cols
            layout = []
            idx = 0
            for r in range(n_rows):
                row = []
                for c in range(n_cols):
                    if idx < len(available):
                        row.append(available[idx])
                        idx += 1
                    else:
                        row.append("")
                layout.append(row)

    cell_w, cell_h = cell_size
    n_rows = len(layout)
    n_cols = max(len(row) for row in layout)

    # Add space for title
    title_height = 40 if title else 0
    # Add space for camera labels
    label_height = 25

    canvas = np.zeros(
        (n_rows * (cell_h + label_height) + title_height, n_cols * cell_w, 3),
        dtype=np.uint8,
    )

    for r, row in enumerate(layout):
        for c, cam_name in enumerate(row):
            if not cam_name or cam_name not in images:
                continue

            img = images[cam_name]
            # Resize to cell size
            resized = _resize_image(img, cell_w, cell_h)

            y_start = title_height + r * (cell_h + label_height) + label_height
            x_start = c * cell_w

            canvas[y_start:y_start + cell_h, x_start:x_start + cell_w] = resized

            # Draw camera label
            label_y = title_height + r * (cell_h + label_height)
            _draw_text(
                canvas,
                cam_name,
                (x_start + 5, label_y + 18),
                (255, 255, 255),
            )

    # Draw title
    if title:
        _draw_text(canvas, title, (10, 25), (255, 255, 255))

    return canvas


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------


def write_annotated_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 10.0,
    codec: str = "mp4v",
) -> None:
    """Write a sequence of annotated frames to a video file.

    Parameters
    ----------
    frames : list of np.ndarray
        List of (H, W, 3) RGB images.
    output_path : str
        Output video file path (e.g., 'output.mp4').
    fps : float
        Frames per second.
    codec : str
        FourCC codec code (e.g., 'mp4v', 'XVID', 'avc1').

    Raises
    ------
    ImportError
        If OpenCV is not available.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for video output. Install with: pip install opencv-python"
        )

    if len(frames) == 0:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    try:
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
    finally:
        writer.release()


def write_annotated_video_from_generator(
    frame_generator,
    output_path: str,
    frame_size: Tuple[int, int] = (1920, 1080),
    fps: float = 10.0,
    codec: str = "mp4v",
) -> None:
    """Write frames from a generator to a video (memory efficient).

    Parameters
    ----------
    frame_generator : iterable
        Yields (H, W, 3) RGB images.
    output_path : str
        Output video file path.
    frame_size : tuple (width, height)
        Expected frame dimensions.
    fps : float
        Frames per second.
    codec : str
        Video codec.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for video output. Install with: pip install opencv-python"
        )

    w, h = frame_size
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    try:
        for frame in frame_generator:
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Drawing primitives (pure numpy, no OpenCV dependency)
# ---------------------------------------------------------------------------


def _draw_line(
    img: np.ndarray,
    pt1: np.ndarray,
    pt2: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw a line on an image using Bresenham-like approach.

    Uses OpenCV if available for better performance, otherwise falls back
    to numpy-based rasterization.
    """
    try:
        import cv2
        cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, thickness)
        return
    except ImportError:
        pass

    # Numpy fallback using linear interpolation
    h, w = img.shape[:2]
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])

    n_steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    t = np.linspace(0, 1, n_steps + 1)
    xs = (x1 + t * (x2 - x1)).astype(int)
    ys = (y1 + t * (y2 - y1)).astype(int)

    # Clip to image bounds
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]

    # Thicken line
    for dx in range(-thickness // 2, thickness // 2 + 1):
        for dy in range(-thickness // 2, thickness // 2 + 1):
            xxs = np.clip(xs + dx, 0, w - 1)
            yys = np.clip(ys + dy, 0, h - 1)
            img[yys, xxs] = color


def _draw_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw a rectangle on an image."""
    try:
        import cv2
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return
    except ImportError:
        pass

    x1, y1 = pt1
    x2, y2 = pt2
    _draw_line(img, np.array([x1, y1]), np.array([x2, y1]), color, thickness)
    _draw_line(img, np.array([x2, y1]), np.array([x2, y2]), color, thickness)
    _draw_line(img, np.array([x2, y2]), np.array([x1, y2]), color, thickness)
    _draw_line(img, np.array([x1, y2]), np.array([x1, y1]), color, thickness)


def _draw_text(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw text on an image. Uses OpenCV if available."""
    try:
        import cv2
        cv2.putText(
            img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, thickness, cv2.LINE_AA,
        )
    except ImportError:
        # Minimal text rendering not feasible without cv2 - skip gracefully
        pass


def _draw_text_with_bg(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw text with a filled background rectangle."""
    try:
        import cv2
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        x, y = position
        cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
        # Text in contrasting color
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
        cv2.putText(
            img, text, (x, y - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX,
            scale, text_color, thickness, cv2.LINE_AA,
        )
    except ImportError:
        _draw_text(img, text, position, color, scale, thickness)


def _resize_image(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image to target dimensions."""
    try:
        import cv2
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        # Numpy nearest-neighbor resize fallback
        h, w = img.shape[:2]
        y_indices = np.linspace(0, h - 1, target_h).astype(int)
        x_indices = np.linspace(0, w - 1, target_w).astype(int)
        return img[np.ix_(y_indices, x_indices)]


# ---------------------------------------------------------------------------
# Matplotlib-based visualization (alternative for figure output)
# ---------------------------------------------------------------------------


def plot_image_with_boxes(
    image: np.ndarray,
    boxes_2d: Optional[np.ndarray] = None,
    classes: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    seg_mask: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None,
    title: str = "",
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """Create a matplotlib figure with image and overlaid annotations.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3).
    boxes_2d : np.ndarray, optional
        2D bounding boxes (N, 4) as [x1, y1, x2, y2].
    classes : sequence of str, optional
        Class names.
    confidences : sequence of float, optional
        Confidence scores.
    seg_mask : np.ndarray, optional
        Segmentation mask to overlay.
    depth_map : np.ndarray, optional
        Depth map to overlay.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Save figure to file.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    display_img = image.copy()

    if seg_mask is not None:
        display_img = overlay_segmentation(display_img, seg_mask, alpha=0.4)

    if depth_map is not None:
        display_img = overlay_depth_map(display_img, depth_map, alpha=0.5)

    ax.imshow(display_img)

    if boxes_2d is not None:
        for i in range(len(boxes_2d)):
            x1, y1, x2, y2 = boxes_2d[i]
            cls_name = classes[i] if classes else "object"
            color_rgb = DEFAULT_CLASS_COLORS.get(cls_name, (255, 255, 255))
            color = tuple(c / 255.0 for c in color_rgb)

            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)

            label = cls_name
            if confidences is not None:
                label += f" {confidences[i]:.2f}"
            ax.text(
                x1, y1 - 5, label,
                color="white", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
            )

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_multicamera_figure(
    images: Dict[str, np.ndarray],
    layout: Optional[List[List[str]]] = None,
    annotations: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Multi-Camera View",
    figsize: Tuple[float, float] = (18, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """Create a matplotlib figure with multiple camera views arranged in a grid.

    Parameters
    ----------
    images : dict
        Camera name to image mapping.
    layout : list of list of str, optional
        Grid layout of camera names.
    annotations : dict, optional
        Camera name to annotated image (if pre-annotated).
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if layout is None:
        available = list(images.keys())
        n_cols = min(3, len(available))
        n_rows = (len(available) + n_cols - 1) // n_cols
        layout = []
        idx = 0
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                if idx < len(available):
                    row.append(available[idx])
                    idx += 1
            layout.append(row)

    n_rows = len(layout)
    n_cols = max(len(row) for row in layout)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for r, row in enumerate(layout):
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(row) and row[c] in images:
                cam_name = row[c]
                if annotations and cam_name in annotations:
                    ax.imshow(annotations[cam_name])
                else:
                    ax.imshow(images[cam_name])
                ax.set_title(cam_name, fontsize=10)
            ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
