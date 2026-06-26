#!/usr/bin/env python3
"""
visualize_results.py - Visualization for PointPillars detection results.

Provides BEV (bird's eye view), 3D wireframe, and comparison visualizations
for point clouds and 3D bounding box detections.

Features:
    - BEV scatter plot with height-colored points and rotated box overlays
    - 3D matplotlib wireframe box rendering
    - Class-based color coding (Car=green, Pedestrian=blue, Cyclist=yellow)
    - Confidence score annotations
    - Ground truth vs predictions side-by-side comparison
    - Sequential frame video generation

Usage:
    python visualize_results.py bev --point-cloud data/000001.bin --detections results/000001.txt
    python visualize_results.py 3d --point-cloud data/000001.bin --detections results/000001.txt
    python visualize_results.py compare --point-cloud data/000001.bin --gt labels/000001.txt --pred results/000001.txt
    python visualize_results.py video --point-cloud-dir data/velodyne/ --detections-dir results/ --output video.mp4
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Use non-interactive backend for headless rendering
matplotlib.use("Agg")

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Class color mapping (RGB 0-1 scale)
CLASS_COLORS = {
    "Car": (0.0, 0.8, 0.0),          # green
    "car": (0.0, 0.8, 0.0),
    "Pedestrian": (0.2, 0.4, 1.0),   # blue
    "pedestrian": (0.2, 0.4, 1.0),
    "Cyclist": (1.0, 0.85, 0.0),     # yellow
    "cyclist": (1.0, 0.85, 0.0),
    "Van": (0.0, 0.6, 0.6),          # teal
    "Truck": (0.8, 0.4, 0.0),        # orange
    "truck": (0.8, 0.4, 0.0),
    "bus": (0.6, 0.0, 0.6),          # purple
    "motorcycle": (1.0, 0.5, 0.5),   # salmon
    "bicycle": (0.9, 0.9, 0.0),      # bright yellow
    "barrier": (0.5, 0.5, 0.5),      # gray
    "traffic_cone": (1.0, 0.6, 0.0), # orange
    "construction_vehicle": (0.4, 0.2, 0.0),  # brown
    "trailer": (0.7, 0.3, 0.5),      # mauve
}

DEFAULT_COLOR = (0.8, 0.0, 0.0)  # red for unknown classes

# Visualization parameters
BEV_X_RANGE = (-40, 80)      # meters (forward/backward)
BEV_Y_RANGE = (-40, 40)      # meters (left/right)
BEV_Z_RANGE = (-3, 1)        # meters (height, used for coloring)
POINT_SIZE = 0.3
FIGURE_DPI = 150

# ============================================================================
# Data Loading
# ============================================================================

def load_point_cloud(filepath: str) -> np.ndarray:
    """Load a point cloud from KITTI .bin format.

    Args:
        filepath: Path to .bin file (float32 x,y,z,intensity per point).

    Returns:
        (N, 4) numpy array of [x, y, z, intensity].
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Point cloud not found: {filepath}")

    points = np.fromfile(filepath, dtype=np.float32)

    # Auto-detect number of features
    if points.size % 5 == 0:
        points = points.reshape(-1, 5)
    elif points.size % 4 == 0:
        points = points.reshape(-1, 4)
    else:
        raise ValueError(
            f"Cannot determine point cloud format: {points.size} values "
            f"(not divisible by 4 or 5)"
        )

    return points


def load_detections_txt(filepath: str) -> List[Dict[str, Any]]:
    """Load detections from KITTI-format text file.

    Each line: class truncated occluded alpha x1 y1 x2 y2 h w l x y z ry [score]

    Args:
        filepath: Path to detection results .txt file.

    Returns:
        List of detection dictionaries with 3D box parameters.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Detections file not found: {filepath}")

    detections = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            det = {
                "class_name": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox_2d": [float(x) for x in parts[4:8]],
                "dimensions": [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
                "location": [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z (camera)
                "rotation_y": float(parts[14]),
                "score": float(parts[15]) if len(parts) > 15 else 1.0,
            }

            # Convert to lidar frame for visualization
            # KITTI camera -> lidar: x_lidar = z_cam, y_lidar = -x_cam, z_lidar = -y_cam
            cam_x, cam_y, cam_z = det["location"]
            h, w, l = det["dimensions"]
            ry = det["rotation_y"]

            det["center_lidar"] = np.array([cam_z, -cam_x, -cam_y + h / 2], dtype=np.float32)
            det["size_lidar"] = np.array([l, w, h], dtype=np.float32)  # dx, dy, dz
            det["heading"] = -(ry + np.pi / 2)

            detections.append(det)

    return detections


def load_detections_json(filepath: str) -> List[Dict[str, Any]]:
    """Load detections from JSON format.

    Expected format:
    [{"class_name": str, "center_lidar": [x,y,z], "size_lidar": [dx,dy,dz],
      "heading": float, "score": float}, ...]

    Args:
        filepath: Path to detection results .json file.

    Returns:
        List of detection dictionaries.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Detections file not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    detections = []
    for item in data:
        det = {
            "class_name": item["class_name"],
            "center_lidar": np.array(item["center_lidar"], dtype=np.float32),
            "size_lidar": np.array(item["size_lidar"], dtype=np.float32),
            "heading": float(item["heading"]),
            "score": float(item.get("score", 1.0)),
        }
        detections.append(det)

    return detections


def load_detections(filepath: str) -> List[Dict[str, Any]]:
    """Load detections from either txt or json format (auto-detected).

    Args:
        filepath: Path to detection results file.

    Returns:
        List of detection dictionaries.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        return load_detections_json(filepath)
    else:
        return load_detections_txt(filepath)


# ============================================================================
# Box Geometry Utilities
# ============================================================================

def get_box_corners_2d(center: np.ndarray, size: np.ndarray, heading: float) -> np.ndarray:
    """Compute 2D BEV corners of a rotated bounding box.

    Args:
        center: (2,) array [x, y] center of box.
        size: (2,) array [dx, dy] half-size not needed, full size.
        heading: Rotation angle in radians.

    Returns:
        (4, 2) array of corner coordinates.
    """
    dx, dy = size[0] / 2, size[1] / 2

    # Corners in box-local frame
    corners = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy],
    ], dtype=np.float32)

    # Rotate corners
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
    rotated = corners @ rotation.T

    # Translate to center
    rotated[:, 0] += center[0]
    rotated[:, 1] += center[1]

    return rotated


def get_box_corners_3d(
    center: np.ndarray, size: np.ndarray, heading: float
) -> np.ndarray:
    """Compute 3D corners of a rotated bounding box.

    Args:
        center: (3,) array [x, y, z] center of box.
        size: (3,) array [dx, dy, dz] full dimensions.
        heading: Rotation angle around z-axis in radians.

    Returns:
        (8, 3) array of corner coordinates.
        Order: bottom-face (4 corners), top-face (4 corners).
    """
    dx, dy, dz = size[0] / 2, size[1] / 2, size[2] / 2

    # 8 corners in box-local frame
    # Bottom face: 0-3, Top face: 4-7
    corners = np.array([
        [ dx,  dy, -dz],  # 0: front-right-bottom
        [ dx, -dy, -dz],  # 1: front-left-bottom
        [-dx, -dy, -dz],  # 2: back-left-bottom
        [-dx,  dy, -dz],  # 3: back-right-bottom
        [ dx,  dy,  dz],  # 4: front-right-top
        [ dx, -dy,  dz],  # 5: front-left-top
        [-dx, -dy,  dz],  # 6: back-left-top
        [-dx,  dy,  dz],  # 7: back-right-top
    ], dtype=np.float32)

    # Rotate around z-axis
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rotation = np.array([
        [cos_h, -sin_h, 0],
        [sin_h,  cos_h, 0],
        [0,      0,     1],
    ], dtype=np.float32)

    rotated = corners @ rotation.T

    # Translate to center
    rotated += center

    return rotated


# ============================================================================
# BEV Visualization
# ============================================================================

def draw_bev(
    points: np.ndarray,
    detections: List[Dict[str, Any]],
    ax: Optional[plt.Axes] = None,
    x_range: Tuple[float, float] = BEV_X_RANGE,
    y_range: Tuple[float, float] = BEV_Y_RANGE,
    z_range: Tuple[float, float] = BEV_Z_RANGE,
    point_size: float = POINT_SIZE,
    show_scores: bool = True,
    score_threshold: float = 0.0,
    title: str = "BEV Visualization",
) -> plt.Axes:
    """Draw bird's eye view visualization.

    Points are colored by height (z-value). Detections are drawn as
    rotated rectangles with class-specific colors.

    Args:
        points: (N, 3+) point cloud array.
        detections: List of detection dictionaries.
        ax: Matplotlib axes to draw on (created if None).
        x_range: (min, max) range for x-axis in meters.
        y_range: (min, max) range for y-axis in meters.
        z_range: (min, max) range for height coloring.
        point_size: Scatter plot point size.
        show_scores: Whether to annotate confidence scores.
        score_threshold: Minimum score to display.
        title: Plot title.

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Filter points within range
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    )
    pts = points[mask]

    # Color points by height (z-value)
    z_vals = np.clip(pts[:, 2], z_range[0], z_range[1])
    z_normalized = (z_vals - z_range[0]) / (z_range[1] - z_range[0])

    # Scatter plot: x-forward maps to plot-y, y-left maps to plot-x
    scatter = ax.scatter(
        pts[:, 1], pts[:, 0],
        c=z_normalized,
        cmap="viridis",
        s=point_size,
        alpha=0.6,
        edgecolors="none",
        rasterized=True,
    )

    # Draw detection boxes
    for det in detections:
        score = det.get("score", 1.0)
        if score < score_threshold:
            continue

        class_name = det["class_name"]
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        center = det["center_lidar"]
        size = det["size_lidar"]
        heading = det["heading"]

        # Get 2D BEV corners
        corners = get_box_corners_2d(
            center[:2], size[:2], heading
        )

        # Draw rotated rectangle (swap x,y for BEV plot orientation)
        polygon = plt.Polygon(
            corners[:, [1, 0]],  # swap x/y for plot coords
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
            linestyle="-",
        )
        ax.add_patch(polygon)

        # Draw heading direction (front arrow)
        front_center = (corners[0] + corners[1]) / 2
        box_center_2d = center[:2]
        ax.annotate(
            "",
            xy=(front_center[1], front_center[0]),
            xytext=(box_center_2d[1], box_center_2d[0]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

        # Score annotation
        if show_scores and score < 1.0:
            ax.text(
                center[1], center[0] + size[0] / 2 + 0.5,
                f"{class_name}\n{score:.2f}",
                color=color,
                fontsize=6,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        else:
            ax.text(
                center[1], center[0] + size[0] / 2 + 0.5,
                class_name,
                color=color,
                fontsize=6,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Draw ego vehicle marker
    ax.plot(0, 0, marker="^", color="red", markersize=10, zorder=5)
    ax.text(0.5, -1, "ego", color="red", fontsize=7, ha="center")

    ax.set_xlim(y_range)
    ax.set_ylim(x_range)
    ax.set_xlabel("Y (meters, left/right)")
    ax.set_ylabel("X (meters, forward)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return ax


# ============================================================================
# 3D Visualization
# ============================================================================

def draw_3d(
    points: np.ndarray,
    detections: List[Dict[str, Any]],
    ax: Optional[plt.Axes] = None,
    x_range: Tuple[float, float] = BEV_X_RANGE,
    y_range: Tuple[float, float] = BEV_Y_RANGE,
    z_range: Tuple[float, float] = (-3, 3),
    point_size: float = 0.1,
    show_scores: bool = True,
    score_threshold: float = 0.0,
    title: str = "3D Visualization",
    elevation: float = 30,
    azimuth: float = -60,
) -> plt.Axes:
    """Draw 3D wireframe box visualization using matplotlib projection.

    Args:
        points: (N, 3+) point cloud array.
        detections: List of detection dictionaries.
        ax: Matplotlib 3D axes (created if None).
        x_range: X range filter.
        y_range: Y range filter.
        z_range: Z range filter.
        point_size: Scatter point size.
        show_scores: Show confidence scores.
        score_threshold: Minimum score threshold.
        title: Plot title.
        elevation: Camera elevation angle.
        azimuth: Camera azimuth angle.

    Returns:
        The 3D matplotlib Axes object.
    """
    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

    # Filter points within range
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    pts = points[mask]

    # Subsample points for performance (3D rendering is slow)
    max_display_points = 50000
    if len(pts) > max_display_points:
        indices = np.random.choice(len(pts), max_display_points, replace=False)
        pts = pts[indices]

    # Color by height
    z_normalized = (pts[:, 2] - z_range[0]) / (z_range[1] - z_range[0])
    z_normalized = np.clip(z_normalized, 0, 1)

    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=z_normalized,
        cmap="viridis",
        s=point_size,
        alpha=0.3,
        depthshade=True,
    )

    # Draw 3D wireframe boxes
    for det in detections:
        score = det.get("score", 1.0)
        if score < score_threshold:
            continue

        class_name = det["class_name"]
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        center = det["center_lidar"]
        size = det["size_lidar"]
        heading = det["heading"]

        corners = get_box_corners_3d(center, size, heading)

        # Draw 12 edges of the 3D box
        # Bottom face edges: 0-1, 1-2, 2-3, 3-0
        # Top face edges: 4-5, 5-6, 6-7, 7-4
        # Vertical edges: 0-4, 1-5, 2-6, 3-7
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
        ]

        for edge in edges:
            xs = [corners[edge[0], 0], corners[edge[1], 0]]
            ys = [corners[edge[0], 1], corners[edge[1], 1]]
            zs = [corners[edge[0], 2], corners[edge[1], 2]]
            ax.plot(xs, ys, zs, color=color, linewidth=1.2)

        # Draw front face with different style to show heading
        front_edges = [[0, 1], [1, 5], [5, 4], [4, 0]]
        for edge in front_edges:
            xs = [corners[edge[0], 0], corners[edge[1], 0]]
            ys = [corners[edge[0], 1], corners[edge[1], 1]]
            zs = [corners[edge[0], 2], corners[edge[1], 2]]
            ax.plot(xs, ys, zs, color=color, linewidth=2.0, linestyle="--")

        # Score annotation in 3D
        if show_scores:
            label = f"{class_name}"
            if score < 1.0:
                label += f" {score:.2f}"
            ax.text(
                center[0], center[1], center[2] + size[2] / 2 + 0.3,
                label,
                color=color,
                fontsize=5,
                ha="center",
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.view_init(elev=elevation, azim=azimuth)

    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)

    return ax


# ============================================================================
# Comparison Visualization
# ============================================================================

def draw_comparison(
    points: np.ndarray,
    gt_detections: List[Dict[str, Any]],
    pred_detections: List[Dict[str, Any]],
    output_path: str,
    score_threshold: float = 0.3,
    title_prefix: str = "",
) -> None:
    """Draw side-by-side comparison of ground truth vs predictions.

    Args:
        points: (N, 3+) point cloud.
        gt_detections: Ground truth detection list.
        pred_detections: Predicted detection list.
        output_path: Path to save the comparison image.
        score_threshold: Minimum score for predictions.
        title_prefix: Prefix for plot titles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Left: Ground Truth
    draw_bev(
        points, gt_detections,
        ax=axes[0],
        show_scores=False,
        title=f"{title_prefix}Ground Truth",
    )

    # Right: Predictions
    draw_bev(
        points, pred_detections,
        ax=axes[1],
        show_scores=True,
        score_threshold=score_threshold,
        title=f"{title_prefix}Predictions (score >= {score_threshold:.2f})",
    )

    # Add legend
    legend_elements = []
    for cls, color in CLASS_COLORS.items():
        if cls[0].isupper():  # Only KITTI-style capitalized names for legend
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor=color, label=cls)
            )
        if len(legend_elements) >= 6:
            break

    axes[1].legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=8,
        framealpha=0.7,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison: {output_path}")


# ============================================================================
# Video Generation
# ============================================================================

def create_video(
    point_cloud_dir: str,
    detections_dir: str,
    output_path: str,
    gt_dir: Optional[str] = None,
    fps: int = 10,
    score_threshold: float = 0.3,
    max_frames: int = 500,
    mode: str = "bev",
) -> None:
    """Create video from sequential frames of point clouds and detections.

    Uses matplotlib animation saved as mp4 or assembles frames into video.

    Args:
        point_cloud_dir: Directory containing .bin point cloud files.
        detections_dir: Directory containing detection result files.
        output_path: Output video file path (.mp4).
        gt_dir: Optional ground truth directory for comparison mode.
        fps: Frames per second.
        score_threshold: Minimum detection score.
        max_frames: Maximum number of frames to process.
        mode: Visualization mode ('bev', '3d', 'compare').
    """
    # Find point cloud files
    pc_files = sorted([
        f for f in os.listdir(point_cloud_dir)
        if f.endswith(".bin")
    ])[:max_frames]

    if not pc_files:
        logger.error(f"No .bin files found in {point_cloud_dir}")
        return

    logger.info(f"Creating video from {len(pc_files)} frames...")

    # Create temporary directory for frame images
    frames_dir = os.path.join(os.path.dirname(output_path), ".video_frames_tmp")
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths = []

    for frame_idx, pc_file in enumerate(pc_files):
        pc_path = os.path.join(point_cloud_dir, pc_file)
        stem = os.path.splitext(pc_file)[0]

        # Load point cloud
        points = load_point_cloud(pc_path)

        # Load detections
        det_file_txt = os.path.join(detections_dir, f"{stem}.txt")
        det_file_json = os.path.join(detections_dir, f"{stem}.json")

        detections = []
        if os.path.exists(det_file_txt):
            detections = load_detections(det_file_txt)
        elif os.path.exists(det_file_json):
            detections = load_detections(det_file_json)

        # Generate frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")

        if mode == "compare" and gt_dir:
            gt_file = os.path.join(gt_dir, f"{stem}.txt")
            gt_dets = load_detections(gt_file) if os.path.exists(gt_file) else []
            draw_comparison(
                points, gt_dets, detections, frame_path,
                score_threshold=score_threshold,
                title_prefix=f"Frame {frame_idx:04d} - ",
            )
        elif mode == "3d":
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            draw_3d(
                points, detections, ax=ax,
                score_threshold=score_threshold,
                title=f"Frame {frame_idx:04d}",
            )
            plt.savefig(frame_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            draw_bev(
                points, detections, ax=ax,
                score_threshold=score_threshold,
                title=f"Frame {frame_idx:04d}",
            )
            plt.savefig(frame_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

        frame_paths.append(frame_path)

        if (frame_idx + 1) % 20 == 0:
            logger.info(f"  Rendered {frame_idx + 1}/{len(pc_files)} frames")

    # Assemble frames into video using matplotlib animation or ffmpeg
    _assemble_video(frame_paths, output_path, fps)

    # Clean up frame images
    for fp in frame_paths:
        os.remove(fp)
    os.rmdir(frames_dir)

    logger.info(f"Video saved: {output_path}")


def _assemble_video(frame_paths: List[str], output_path: str, fps: int) -> None:
    """Assemble frame images into a video file.

    Tries ffmpeg first for best quality, falls back to matplotlib animation.

    Args:
        frame_paths: List of frame image file paths.
        output_path: Output video file path.
        fps: Frames per second.
    """
    import subprocess
    import shutil

    # Try ffmpeg first
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        frames_dir = os.path.dirname(frame_paths[0])
        cmd = [
            ffmpeg_path,
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-profile:v", "high",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                logger.info(f"Video created with ffmpeg: {output_path}")
                return
            else:
                logger.warning(f"ffmpeg failed: {result.stderr[:200]}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("ffmpeg unavailable or timed out, falling back to imageio")

    # Fallback: use imageio if available
    try:
        import imageio.v3 as iio

        frames = []
        for fp in frame_paths:
            frame = iio.imread(fp)
            frames.append(frame)

        # Ensure output has .mp4 extension for proper codec
        if not output_path.endswith(".mp4"):
            output_path = output_path.rsplit(".", 1)[0] + ".mp4"

        iio.imwrite(output_path, frames, fps=fps, codec="libx264")
        logger.info(f"Video created with imageio: {output_path}")
        return
    except ImportError:
        logger.warning("imageio not available")

    # Last fallback: save as animated GIF with matplotlib
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig, ax = plt.subplots(figsize=(12, 10))

        def update(frame_idx):
            ax.clear()
            img = plt.imread(frame_paths[frame_idx])
            ax.imshow(img)
            ax.axis("off")
            return []

        anim = FuncAnimation(fig, update, frames=len(frame_paths), interval=1000 / fps)
        gif_path = output_path.rsplit(".", 1)[0] + ".gif"
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        logger.info(f"Animation saved as GIF: {gif_path}")
    except Exception as e:
        logger.error(f"All video backends failed: {e}")
        logger.info("Frame images were generated. Install ffmpeg or imageio for video output.")


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_bev(args: argparse.Namespace) -> None:
    """Execute BEV visualization command.

    Args:
        args: Parsed command line arguments.
    """
    logger.info(f"Loading point cloud: {args.point_cloud}")
    points = load_point_cloud(args.point_cloud)
    logger.info(f"  Loaded {len(points)} points")

    detections = []
    if args.detections:
        logger.info(f"Loading detections: {args.detections}")
        detections = load_detections(args.detections)
        logger.info(f"  Loaded {len(detections)} detections")

    # Filter by score
    if args.score_threshold > 0:
        detections = [d for d in detections if d.get("score", 1.0) >= args.score_threshold]
        logger.info(f"  After score filter (>= {args.score_threshold}): {len(detections)}")

    # Create BEV visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    draw_bev(
        points, detections, ax=ax,
        show_scores=args.show_scores,
        score_threshold=args.score_threshold,
        title=f"BEV - {os.path.basename(args.point_cloud)}",
    )

    output_path = args.output or _default_output_path(args.point_cloud, "bev")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved BEV image: {output_path}")


def cmd_3d(args: argparse.Namespace) -> None:
    """Execute 3D visualization command.

    Args:
        args: Parsed command line arguments.
    """
    logger.info(f"Loading point cloud: {args.point_cloud}")
    points = load_point_cloud(args.point_cloud)
    logger.info(f"  Loaded {len(points)} points")

    detections = []
    if args.detections:
        logger.info(f"Loading detections: {args.detections}")
        detections = load_detections(args.detections)
        logger.info(f"  Loaded {len(detections)} detections")

    if args.score_threshold > 0:
        detections = [d for d in detections if d.get("score", 1.0) >= args.score_threshold]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    draw_3d(
        points, detections, ax=ax,
        show_scores=args.show_scores,
        score_threshold=args.score_threshold,
        title=f"3D - {os.path.basename(args.point_cloud)}",
        elevation=args.elevation,
        azimuth=args.azimuth,
    )

    output_path = args.output or _default_output_path(args.point_cloud, "3d")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved 3D image: {output_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Execute comparison visualization command.

    Args:
        args: Parsed command line arguments.
    """
    logger.info(f"Loading point cloud: {args.point_cloud}")
    points = load_point_cloud(args.point_cloud)
    logger.info(f"  Loaded {len(points)} points")

    logger.info(f"Loading ground truth: {args.gt}")
    gt_dets = load_detections(args.gt)
    logger.info(f"  Loaded {len(gt_dets)} GT boxes")

    logger.info(f"Loading predictions: {args.pred}")
    pred_dets = load_detections(args.pred)
    logger.info(f"  Loaded {len(pred_dets)} predicted boxes")

    output_path = args.output or _default_output_path(args.point_cloud, "compare")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    draw_comparison(
        points, gt_dets, pred_dets, output_path,
        score_threshold=args.score_threshold,
    )


def cmd_video(args: argparse.Namespace) -> None:
    """Execute video generation command.

    Args:
        args: Parsed command line arguments.
    """
    output_path = args.output or "output_video.mp4"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    create_video(
        point_cloud_dir=args.point_cloud_dir,
        detections_dir=args.detections_dir,
        output_path=output_path,
        gt_dir=args.gt_dir,
        fps=args.fps,
        score_threshold=args.score_threshold,
        max_frames=args.max_frames,
        mode=args.mode,
    )


# ============================================================================
# Utility
# ============================================================================

def _default_output_path(input_path: str, suffix: str) -> str:
    """Generate default output path from input path.

    Args:
        input_path: Input file path.
        suffix: Suffix to add before extension.

    Returns:
        Output file path with suffix appended.
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(os.path.dirname(input_path), "..", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{stem}_{suffix}.png")


# ============================================================================
# Argument Parsing
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="PointPillars Detection Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s bev --point-cloud data/000001.bin --detections results/000001.txt
  %(prog)s 3d --point-cloud data/000001.bin --detections results/000001.txt --elevation 45
  %(prog)s compare --point-cloud data/000001.bin --gt labels/000001.txt --pred results/000001.txt
  %(prog)s video --point-cloud-dir data/velodyne/ --detections-dir results/ --output demo.mp4
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Visualization mode")

    # BEV subcommand
    bev_parser = subparsers.add_parser("bev", help="Bird's eye view visualization")
    bev_parser.add_argument(
        "--point-cloud", required=True, help="Path to .bin point cloud file"
    )
    bev_parser.add_argument(
        "--detections", default=None, help="Path to detections file (.txt or .json)"
    )
    bev_parser.add_argument(
        "--output", default=None, help="Output image path (default: auto-generated)"
    )
    bev_parser.add_argument(
        "--score-threshold", type=float, default=0.3, help="Min score to display (default: 0.3)"
    )
    bev_parser.add_argument(
        "--show-scores", action="store_true", default=True, help="Show confidence scores"
    )
    bev_parser.add_argument(
        "--no-scores", dest="show_scores", action="store_false", help="Hide confidence scores"
    )
    bev_parser.set_defaults(func=cmd_bev)

    # 3D subcommand
    three_d_parser = subparsers.add_parser("3d", help="3D wireframe visualization")
    three_d_parser.add_argument(
        "--point-cloud", required=True, help="Path to .bin point cloud file"
    )
    three_d_parser.add_argument(
        "--detections", default=None, help="Path to detections file (.txt or .json)"
    )
    three_d_parser.add_argument(
        "--output", default=None, help="Output image path"
    )
    three_d_parser.add_argument(
        "--score-threshold", type=float, default=0.3, help="Min score to display"
    )
    three_d_parser.add_argument(
        "--show-scores", action="store_true", default=True, help="Show confidence scores"
    )
    three_d_parser.add_argument(
        "--no-scores", dest="show_scores", action="store_false", help="Hide scores"
    )
    three_d_parser.add_argument(
        "--elevation", type=float, default=30, help="Camera elevation angle (default: 30)"
    )
    three_d_parser.add_argument(
        "--azimuth", type=float, default=-60, help="Camera azimuth angle (default: -60)"
    )
    three_d_parser.set_defaults(func=cmd_3d)

    # Compare subcommand
    compare_parser = subparsers.add_parser("compare", help="GT vs predictions comparison")
    compare_parser.add_argument(
        "--point-cloud", required=True, help="Path to .bin point cloud file"
    )
    compare_parser.add_argument(
        "--gt", required=True, help="Path to ground truth detections file"
    )
    compare_parser.add_argument(
        "--pred", required=True, help="Path to prediction detections file"
    )
    compare_parser.add_argument(
        "--output", default=None, help="Output image path"
    )
    compare_parser.add_argument(
        "--score-threshold", type=float, default=0.3, help="Min score for predictions"
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Video subcommand
    video_parser = subparsers.add_parser("video", help="Create video from sequential frames")
    video_parser.add_argument(
        "--point-cloud-dir", required=True, help="Directory with .bin point cloud files"
    )
    video_parser.add_argument(
        "--detections-dir", required=True, help="Directory with detection result files"
    )
    video_parser.add_argument(
        "--gt-dir", default=None, help="Ground truth directory (for compare mode)"
    )
    video_parser.add_argument(
        "--output", default="output_video.mp4", help="Output video path"
    )
    video_parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second (default: 10)"
    )
    video_parser.add_argument(
        "--score-threshold", type=float, default=0.3, help="Min score to display"
    )
    video_parser.add_argument(
        "--max-frames", type=int, default=500, help="Max frames to process (default: 500)"
    )
    video_parser.add_argument(
        "--mode", choices=["bev", "3d", "compare"], default="bev",
        help="Visualization mode for video frames (default: bev)"
    )
    video_parser.set_defaults(func=cmd_video)

    return parser


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
