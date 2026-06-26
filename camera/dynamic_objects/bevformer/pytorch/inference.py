"""Single-sample and sequence inference script with visualization for BEVFormer.

This module provides inference utilities for the BEVFormer model, including:
- Single-sample inference with multi-camera 3D box visualization
- Temporal sequence inference with BEV state propagation
- BEV top-down map visualization with oriented bounding boxes
- Video generation for full scene sequences

Requires PyTorch 2.x and matplotlib for visualization.

Example usage:
    # Single sample visualization
    python inference.py --config config.yaml --checkpoint model.pt --sample-idx 0

    # Full scene sequence video
    python inference.py --config config.yaml --checkpoint model.pt --sequence --output-dir results/
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .dataset import NuScenesDataset, collate_fn
from .model import BEVFormer

__all__ = [
    "get_color_map",
    "project_3d_box_to_image",
    "draw_3d_boxes_on_image",
    "draw_bev_map",
    "visualize_single_sample",
    "visualize_sequence",
    "run_inference",
]

logger = logging.getLogger(__name__)

# Class names for the 10-class detection task
CLASS_NAMES: list[str] = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# Camera names in canonical order
CAMERA_NAMES: list[str] = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# Point cloud range: [x_min, y_min, z_min, x_max, y_max, z_max]
PC_RANGE: list[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


def get_color_map(num_classes: int) -> list[tuple[int, int, int]]:
    """Generate a distinct color map for object classes.

    Uses a perceptually spaced colormap to ensure visual distinguishability
    between classes in both camera and BEV visualizations.

    Args:
        num_classes: Number of distinct classes to generate colors for.

    Returns:
        List of RGB tuples (each value in [0, 255]) indexed by class ID.
    """
    cmap = plt.cm.get_cmap("tab10", num_classes)
    colors: list[tuple[int, int, int]] = []
    for i in range(num_classes):
        rgba = cmap(i)
        colors.append((int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)))
    return colors


def _yaw_from_sincos(sin_yaw: float, cos_yaw: float) -> float:
    """Recover yaw angle from sin and cos components.

    Args:
        sin_yaw: Sine of the yaw angle.
        cos_yaw: Cosine of the yaw angle.

    Returns:
        Yaw angle in radians, in the range [-pi, pi].
    """
    return math.atan2(sin_yaw, cos_yaw)


def _get_3d_box_corners(box_3d: np.ndarray) -> np.ndarray:
    """Compute the 8 corners of a 3D bounding box in world/ego coordinates.

    The box is parameterized as [cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy].
    The corners are computed in the order:
        Bottom face: front-left, front-right, rear-right, rear-left
        Top face: front-left, front-right, rear-right, rear-left

    Args:
        box_3d: Array of shape (10,) with box parameters.

    Returns:
        Array of shape (8, 3) with 3D corner coordinates in ego frame.
    """
    cx, cy, cz, w, l, h, sin_yaw, cos_yaw = box_3d[:8]
    yaw = _yaw_from_sincos(sin_yaw, cos_yaw)

    # Half dimensions
    hw, hl, hh = w / 2.0, l / 2.0, h / 2.0

    # 8 corners in local object frame (x-forward, y-left, z-up convention)
    # nuscenes convention: x is length, y is width
    corners_local = np.array(
        [
            [hl, hw, -hh],   # front-left-bottom
            [hl, -hw, -hh],  # front-right-bottom
            [-hl, -hw, -hh], # rear-right-bottom
            [-hl, hw, -hh],  # rear-left-bottom
            [hl, hw, hh],    # front-left-top
            [hl, -hw, hh],   # front-right-top
            [-hl, -hw, hh],  # rear-right-top
            [-hl, hw, hh],   # rear-left-top
        ],
        dtype=np.float64,
    )

    # Rotation matrix around Z-axis
    rot = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0],
         [math.sin(yaw), math.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Rotate and translate
    corners_world = (rot @ corners_local.T).T + np.array([cx, cy, cz])
    return corners_world


def project_3d_box_to_image(
    box_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> Optional[np.ndarray]:
    """Project a 3D bounding box onto the image plane of a camera.

    Transforms 3D box corners from ego frame to camera frame using the
    extrinsic matrix, then projects to 2D using the intrinsic matrix.

    Args:
        box_3d: Array of shape (10,) with box parameters
            [cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy].
        intrinsic: Camera intrinsic matrix of shape (3, 3) or (4, 4).
        extrinsic: Camera extrinsic matrix of shape (4, 4), mapping from
            ego/world frame to camera frame.

    Returns:
        Array of shape (8, 2) with projected 2D corner coordinates (u, v),
        or None if the majority of box corners are behind the camera.
    """
    corners_3d = _get_3d_box_corners(box_3d)  # (8, 3)

    # Convert to homogeneous coordinates
    ones = np.ones((8, 1), dtype=np.float64)
    corners_homo = np.hstack([corners_3d, ones])  # (8, 4)

    # Transform to camera frame: extrinsic maps ego -> camera
    corners_cam = (extrinsic @ corners_homo.T).T  # (8, 4)
    corners_cam_xyz = corners_cam[:, :3]  # (8, 3)

    # Check depth: skip if majority of points are behind camera (z <= 0)
    depths = corners_cam_xyz[:, 2]
    if np.sum(depths > 0) < 4:
        return None

    # Project to image using intrinsic (handle 3x3 or 4x4)
    K = intrinsic[:3, :3] if intrinsic.shape[0] >= 3 else intrinsic

    # Only project points with positive depth
    projected = np.zeros((8, 2), dtype=np.float64)
    for i in range(8):
        if depths[i] > 0:
            pt = K @ corners_cam_xyz[i]
            projected[i, 0] = pt[0] / pt[2]
            projected[i, 1] = pt[1] / pt[2]
        else:
            # Clamp points behind camera to a large value (will be clipped later)
            projected[i, 0] = float("nan")
            projected[i, 1] = float("nan")

    return projected


def draw_3d_boxes_on_image(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    class_names: list[str],
    color_map: list[tuple[int, int, int]],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw projected 3D bounding boxes as wireframes on a camera image.

    Each box is drawn as a wireframe connecting the 8 projected corners,
    colored by class. A text label with class name and confidence score
    is placed near the top-left corner of each box.

    Args:
        image: Camera image array of shape (H, W, 3), uint8.
        boxes: Array of shape (K, 10) with box parameters.
        labels: Array of shape (K,) with integer class labels.
        scores: Array of shape (K,) with confidence scores.
        intrinsic: Camera intrinsic matrix of shape (3, 3) or (4, 4).
        extrinsic: Camera extrinsic matrix of shape (4, 4).
        class_names: List of class name strings.
        color_map: List of RGB tuples per class.
        ax: Optional matplotlib Axes to draw on. If None, creates new figure.

    Returns:
        The matplotlib Axes with the visualization.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 7))

    ax.imshow(image)
    ax.axis("off")

    h, w = image.shape[:2]

    # Define edges connecting corners for wireframe
    # Bottom face: 0-1, 1-2, 2-3, 3-0
    # Top face: 4-5, 5-6, 6-7, 7-4
    # Vertical edges: 0-4, 1-5, 2-6, 3-7
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]

    for i in range(len(boxes)):
        projected = project_3d_box_to_image(boxes[i], intrinsic, extrinsic)
        if projected is None:
            continue

        label_idx = int(labels[i])
        color = tuple(c / 255.0 for c in color_map[label_idx % len(color_map)])
        score = float(scores[i])

        # Draw edges
        for start, end in edges:
            p1 = projected[start]
            p2 = projected[end]

            # Skip edges with NaN (behind camera)
            if np.isnan(p1).any() or np.isnan(p2).any():
                continue

            # Clip to image boundaries for display check
            if (p1[0] < -w or p1[0] > 2 * w or p1[1] < -h or p1[1] > 2 * h):
                continue
            if (p2[0] < -w or p2[0] > 2 * w or p2[1] < -h or p2[1] > 2 * h):
                continue

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )

        # Add text label at top-left visible corner
        valid_pts = projected[~np.isnan(projected[:, 0])]
        if len(valid_pts) > 0:
            # Find top-left-most visible point
            text_x = np.clip(np.min(valid_pts[:, 0]), 0, w - 1)
            text_y = np.clip(np.min(valid_pts[:, 1]), 0, h - 1)

            if 0 <= text_x < w and 0 <= text_y < h:
                class_name = class_names[label_idx] if label_idx < len(class_names) else f"cls_{label_idx}"
                ax.text(
                    text_x,
                    text_y - 5,
                    f"{class_name}: {score:.2f}",
                    color="white",
                    fontsize=6,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                )

    return ax


def draw_bev_map(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    color_map: list[tuple[int, int, int]],
    pc_range: list[float],
    bev_resolution: int = 800,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw a top-down Bird's Eye View (BEV) visualization of detections.

    Renders detected objects as oriented rectangles on a BEV map, with
    velocity arrows, grid lines, range rings, ego vehicle marker, and a legend.

    Args:
        boxes: Array of shape (K, 10) with box parameters.
        labels: Array of shape (K,) with integer class labels.
        scores: Array of shape (K,) with confidence scores.
        class_names: List of class name strings.
        color_map: List of RGB tuples per class.
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        bev_resolution: Resolution of the BEV map in pixels.
        ax: Optional matplotlib Axes to draw on. If None, creates new figure.

    Returns:
        The matplotlib Axes with the BEV visualization.
    """
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    x_range = x_max - x_min
    y_range = y_max - y_min

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlabel("Y (m)", fontsize=9)
    ax.set_ylabel("X (m)", fontsize=9)
    ax.set_title("BEV Detection Map", fontsize=11, fontweight="bold")

    # Draw grid lines
    grid_spacing = 10.0
    for x in np.arange(x_min, x_max + grid_spacing, grid_spacing):
        ax.axhline(y=x, color="gray", linewidth=0.3, alpha=0.4)
    for y in np.arange(y_min, y_max + grid_spacing, grid_spacing):
        ax.axvline(x=y, color="gray", linewidth=0.3, alpha=0.4)

    # Draw range rings (concentric circles from ego)
    for radius in [10, 20, 30, 40, 50]:
        circle = plt.Circle(
            (0, 0), radius, fill=False, color="white", linewidth=0.5, alpha=0.3, linestyle="--"
        )
        ax.add_patch(circle)
        ax.text(
            0.5, radius + 0.5, f"{radius}m",
            color="white", fontsize=6, alpha=0.5, ha="left",
        )

    # Draw ego vehicle at center
    ego_length, ego_width = 4.5, 2.0
    ego_rect = mpatches.FancyBboxPatch(
        (-ego_width / 2, -ego_length / 2),
        ego_width,
        ego_length,
        boxstyle="round,pad=0.1",
        facecolor="white",
        edgecolor="yellow",
        linewidth=2,
        alpha=0.8,
    )
    ax.add_patch(ego_rect)
    # Ego direction arrow
    ax.annotate(
        "",
        xy=(0, ego_length / 2 + 1.5),
        xytext=(0, ego_length / 2),
        arrowprops=dict(arrowstyle="->", color="yellow", lw=2),
    )
    ax.text(0, -0.2, "EGO", color="black", fontsize=6, ha="center", va="center", fontweight="bold")

    # Draw detected objects
    legend_handles: dict[str, mpatches.Patch] = {}

    for i in range(len(boxes)):
        cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy = boxes[i]
        label_idx = int(labels[i])
        score = float(scores[i])
        color = tuple(c / 255.0 for c in color_map[label_idx % len(color_map)])
        class_name = class_names[label_idx] if label_idx < len(class_names) else f"cls_{label_idx}"

        yaw = _yaw_from_sincos(sin_yaw, cos_yaw)

        # Draw oriented rectangle (in BEV: x is forward, y is left)
        # matplotlib Rectangle uses bottom-left corner
        corners = np.array([
            [-l / 2, -w / 2],
            [l / 2, -w / 2],
            [l / 2, w / 2],
            [-l / 2, w / 2],
            [-l / 2, -w / 2],  # close polygon
        ])

        # Rotate corners by yaw
        rot = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)],
        ])
        corners_rotated = (rot @ corners.T).T + np.array([cx, cy])

        # Plot in BEV: swap x,y for display (y is horizontal, x is vertical)
        ax.plot(
            corners_rotated[:, 1],
            corners_rotated[:, 0],
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )
        ax.fill(
            corners_rotated[:, 1],
            corners_rotated[:, 0],
            color=color,
            alpha=0.3,
        )

        # Draw front face indicator (first edge)
        front_mid_x = (corners_rotated[0, 0] + corners_rotated[1, 0]) / 2
        front_mid_y = (corners_rotated[0, 1] + corners_rotated[1, 1]) / 2
        ax.plot(front_mid_y, front_mid_x, "o", color=color, markersize=3, alpha=0.9)

        # Draw velocity arrow
        vel_magnitude = math.sqrt(vx**2 + vy**2)
        if vel_magnitude > 0.5:  # Only show if velocity is significant
            arrow_scale = 1.0  # seconds of velocity to show
            ax.annotate(
                "",
                xy=(cy + vy * arrow_scale, cx + vx * arrow_scale),
                xytext=(cy, cx),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.7),
            )

        # Add score text
        ax.text(
            cy, cx + l / 2 + 1.0,
            f"{score:.2f}",
            color=color, fontsize=5, ha="center", va="bottom", alpha=0.8,
        )

        # Build legend
        if class_name not in legend_handles:
            legend_handles[class_name] = mpatches.Patch(color=color, label=class_name)

    # Add legend
    if legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            loc="upper right",
            fontsize=7,
            framealpha=0.7,
            facecolor="#2a2a4e",
            labelcolor="white",
        )

    return ax


def visualize_single_sample(
    model: BEVFormer,
    dataset: NuScenesDataset,
    sample_idx: int,
    output_dir: str,
    score_threshold: float = 0.3,
) -> Path:
    """Run inference on a single sample and create a multi-panel visualization.

    Creates a figure with:
    - Top row: 3 front cameras (FRONT_LEFT, FRONT, FRONT_RIGHT) with 3D boxes
    - Middle row: 3 rear cameras (BACK_LEFT, BACK, BACK_RIGHT) with 3D boxes
    - Right panel: BEV top-down view

    Args:
        model: The BEVFormer model in eval mode.
        dataset: The NuScenesDataset instance.
        sample_idx: Index of the sample in the dataset.
        output_dir: Directory to save the output visualization.
        score_threshold: Minimum confidence score to display detections.

    Returns:
        Path to the saved visualization PNG file.

    Raises:
        IndexError: If sample_idx is out of dataset range.
        RuntimeError: If model inference fails.
    """
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(
            f"sample_idx {sample_idx} out of range [0, {len(dataset) - 1}]"
        )

    os.makedirs(output_dir, exist_ok=True)
    color_map = get_color_map(len(CLASS_NAMES))

    # Load sample
    logger.info(f"Loading sample {sample_idx} from dataset...")
    sample = dataset[sample_idx]
    batch = collate_fn([sample])

    # Extract tensors
    device = next(model.parameters()).device
    images = batch["images"].to(device)           # (1, 6, 3, H, W)
    intrinsics = batch["intrinsics"].to(device)   # (1, 6, 3, 3) or (1, 6, 4, 4)
    extrinsics = batch["extrinsics"].to(device)   # (1, 6, 4, 4)
    ego_motion = batch.get("ego_motion")
    if ego_motion is not None:
        ego_motion = ego_motion.to(device)

    # Run inference
    logger.info("Running model inference...")
    with torch.no_grad():
        detections, _ = model.forward_test(
            images, intrinsics, extrinsics, ego_motion, prev_bev=None
        )

    # Extract detections for batch index 0
    det_scores = detections["scores"][0].cpu().numpy()   # (K,)
    det_labels = detections["labels"][0].cpu().numpy()   # (K,)
    det_boxes = detections["boxes"][0].cpu().numpy()     # (K, 10)

    # Filter by score threshold
    mask = det_scores >= score_threshold
    det_scores = det_scores[mask]
    det_labels = det_labels[mask]
    det_boxes = det_boxes[mask]

    logger.info(f"Detected {len(det_scores)} objects above threshold {score_threshold}")

    # Get images and camera parameters as numpy
    images_np = images[0].cpu().numpy()  # (6, 3, H, W)
    intrinsics_np = intrinsics[0].cpu().numpy()  # (6, 3, 3) or (6, 4, 4)
    extrinsics_np = extrinsics[0].cpu().numpy()  # (6, 4, 4)

    # Camera layout for visualization:
    # Top row:    FRONT_LEFT, FRONT, FRONT_RIGHT
    # Bottom row: BACK_LEFT,  BACK,  BACK_RIGHT
    cam_layout = [
        ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
        ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
    ]

    # Create figure with gridspec: 2 rows of cameras + BEV on the right
    fig = plt.figure(figsize=(24, 14), facecolor="black")
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.2], hspace=0.05, wspace=0.05)

    for row_idx, row_cams in enumerate(cam_layout):
        for col_idx, cam_name in enumerate(row_cams):
            cam_idx = CAMERA_NAMES.index(cam_name)
            img = images_np[cam_idx].transpose(1, 2, 0)  # (H, W, 3)

            # Denormalize if needed (assume [0, 1] or [-1, 1] range)
            if img.min() < 0:
                img = (img + 1.0) / 2.0
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            ax = fig.add_subplot(gs[row_idx, col_idx])
            draw_3d_boxes_on_image(
                img, det_boxes, det_labels, det_scores,
                intrinsics_np[cam_idx], extrinsics_np[cam_idx],
                CLASS_NAMES, color_map, ax=ax,
            )
            ax.set_title(cam_name, color="white", fontsize=10, pad=3)

    # BEV map spanning both rows on the right
    ax_bev = fig.add_subplot(gs[:, 3])
    draw_bev_map(
        det_boxes, det_labels, det_scores,
        CLASS_NAMES, color_map, PC_RANGE, ax=ax_bev,
    )

    # Save figure
    output_path = Path(output_dir) / f"sample_{sample_idx:06d}.png"
    fig.savefig(
        str(output_path),
        dpi=150,
        bbox_inches="tight",
        facecolor="black",
        pad_inches=0.1,
    )
    plt.close(fig)

    logger.info(f"Visualization saved to {output_path}")
    return output_path


def visualize_sequence(
    model: BEVFormer,
    dataset: NuScenesDataset,
    scene_token: str,
    output_dir: str,
    score_threshold: float = 0.3,
    fps: int = 2,
) -> Path:
    """Process all frames in a scene temporally and create a video visualization.

    Maintains the prev_bev state across frames to enable temporal fusion.
    For each frame, creates the multi-panel visualization and compiles
    all frames into a video using cv2.VideoWriter.

    Args:
        model: The BEVFormer model in eval mode.
        dataset: The NuScenesDataset instance.
        scene_token: Token identifying the scene to process.
        output_dir: Directory to save output frames and video.
        score_threshold: Minimum confidence score to display detections.
        fps: Frames per second for the output video.

    Returns:
        Path to the saved video file.

    Raises:
        ValueError: If scene_token is not found in the dataset.
        RuntimeError: If video encoding fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = Path(output_dir) / "frames" / scene_token
    os.makedirs(frames_dir, exist_ok=True)

    color_map = get_color_map(len(CLASS_NAMES))
    device = next(model.parameters()).device

    # Get sample indices for the scene
    scene_indices = _get_scene_sample_indices(dataset, scene_token)
    if not scene_indices:
        raise ValueError(f"Scene '{scene_token}' not found or has no samples in dataset.")

    logger.info(f"Processing scene '{scene_token}' with {len(scene_indices)} frames...")

    prev_bev = None
    frame_paths: list[Path] = []

    for frame_num, sample_idx in enumerate(scene_indices):
        logger.info(f"  Frame {frame_num + 1}/{len(scene_indices)} (sample_idx={sample_idx})")

        sample = dataset[sample_idx]
        batch = collate_fn([sample])

        images = batch["images"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        ego_motion = batch.get("ego_motion")
        if ego_motion is not None:
            ego_motion = ego_motion.to(device)

        # Run inference with temporal BEV propagation
        with torch.no_grad():
            detections, prev_bev = model.forward_test(
                images, intrinsics, extrinsics, ego_motion, prev_bev=prev_bev
            )

        # Extract detections
        det_scores = detections["scores"][0].cpu().numpy()
        det_labels = detections["labels"][0].cpu().numpy()
        det_boxes = detections["boxes"][0].cpu().numpy()

        # Filter by score threshold
        mask = det_scores >= score_threshold
        det_scores = det_scores[mask]
        det_labels = det_labels[mask]
        det_boxes = det_boxes[mask]

        # Get images and parameters
        images_np = images[0].cpu().numpy()
        intrinsics_np = intrinsics[0].cpu().numpy()
        extrinsics_np = extrinsics[0].cpu().numpy()

        # Create visualization frame
        fig = plt.figure(figsize=(24, 14), facecolor="black")
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.2], hspace=0.05, wspace=0.05)

        cam_layout = [
            ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
            ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
        ]

        for row_idx, row_cams in enumerate(cam_layout):
            for col_idx, cam_name in enumerate(row_cams):
                cam_idx = CAMERA_NAMES.index(cam_name)
                img = images_np[cam_idx].transpose(1, 2, 0)

                if img.min() < 0:
                    img = (img + 1.0) / 2.0
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                ax = fig.add_subplot(gs[row_idx, col_idx])
                draw_3d_boxes_on_image(
                    img, det_boxes, det_labels, det_scores,
                    intrinsics_np[cam_idx], extrinsics_np[cam_idx],
                    CLASS_NAMES, color_map, ax=ax,
                )
                ax.set_title(cam_name, color="white", fontsize=10, pad=3)

        ax_bev = fig.add_subplot(gs[:, 3])
        draw_bev_map(
            det_boxes, det_labels, det_scores,
            CLASS_NAMES, color_map, PC_RANGE, ax=ax_bev,
        )

        # Add frame info
        fig.suptitle(
            f"Scene: {scene_token} | Frame: {frame_num + 1}/{len(scene_indices)} | "
            f"Detections: {len(det_scores)}",
            color="white", fontsize=12, y=0.98,
        )

        # Save individual frame
        frame_path = frames_dir / f"frame_{frame_num:04d}.png"
        fig.savefig(
            str(frame_path),
            dpi=100,
            bbox_inches="tight",
            facecolor="black",
            pad_inches=0.1,
        )
        plt.close(fig)
        frame_paths.append(frame_path)

    # Compile frames into video
    video_path = Path(output_dir) / f"scene_{scene_token}.mp4"
    _compile_video(frame_paths, video_path, fps=fps)

    logger.info(f"Video saved to {video_path}")
    logger.info(f"Individual frames saved to {frames_dir}")
    return video_path


def _get_scene_sample_indices(dataset: NuScenesDataset, scene_token: str) -> list[int]:
    """Get dataset indices belonging to a specific scene in temporal order.

    Attempts to use the dataset's internal scene mapping. Falls back to
    iterating over all samples if no mapping is available.

    Args:
        dataset: The NuScenesDataset instance.
        scene_token: Token identifying the scene.

    Returns:
        Sorted list of dataset indices for the scene.
    """
    # Try accessing scene-to-sample mapping from dataset
    if hasattr(dataset, "scene_to_indices"):
        indices = dataset.scene_to_indices.get(scene_token, [])
        return sorted(indices)

    if hasattr(dataset, "samples"):
        indices = []
        for idx, sample_info in enumerate(dataset.samples):
            token = None
            if isinstance(sample_info, dict):
                token = sample_info.get("scene_token", sample_info.get("scene", None))
            elif hasattr(sample_info, "scene_token"):
                token = sample_info.scene_token
            if token == scene_token:
                indices.append(idx)
        return sorted(indices)

    # Fallback: try scene_tokens attribute
    if hasattr(dataset, "scene_tokens"):
        indices = [
            i for i, t in enumerate(dataset.scene_tokens)
            if t == scene_token
        ]
        return sorted(indices)

    logger.warning(
        f"Cannot determine scene sample mapping. "
        f"Returning empty list for scene_token='{scene_token}'."
    )
    return []


def _compile_video(frame_paths: list[Path], output_path: Path, fps: int = 2) -> None:
    """Compile a list of frame images into a video file.

    Uses cv2.VideoWriter if OpenCV is available, otherwise falls back to
    matplotlib.animation.

    Args:
        frame_paths: Ordered list of paths to frame PNG images.
        output_path: Path for the output video file.
        fps: Frames per second for the video.

    Raises:
        RuntimeError: If neither cv2 nor matplotlib animation can produce a video.
    """
    if not frame_paths:
        logger.warning("No frames to compile into video.")
        return

    try:
        import cv2

        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")

        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                writer.write(frame)
            else:
                logger.warning(f"Failed to read frame: {frame_path}, skipping.")

        writer.release()
        logger.info(f"Video compiled with cv2: {output_path} ({len(frame_paths)} frames, {fps} fps)")

    except ImportError:
        logger.info("cv2 not available, using matplotlib animation for video compilation.")
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        fig, ax = plt.subplots(1, 1, figsize=(24, 14))
        ax.axis("off")

        # Load all frames
        frame_images = []
        for fp in frame_paths:
            img = plt.imread(str(fp))
            frame_images.append(img)

        im = ax.imshow(frame_images[0])

        def update(frame_idx: int):
            im.set_data(frame_images[frame_idx])
            return [im]

        anim = FuncAnimation(fig, update, frames=len(frame_images), interval=1000 / fps, blit=True)

        try:
            writer = FFMpegWriter(fps=fps, codec="libx264")
            anim.save(str(output_path), writer=writer)
        except (RuntimeError, FileNotFoundError):
            # FFmpeg not available, try saving as GIF
            gif_path = output_path.with_suffix(".gif")
            anim.save(str(gif_path), writer="pillow", fps=fps)
            logger.warning(f"FFmpeg unavailable. Saved as GIF instead: {gif_path}")

        plt.close(fig)


def _print_detection_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> None:
    """Print a summary of detection counts per class.

    Args:
        scores: Array of detection scores.
        labels: Array of integer class labels.
        class_names: List of class name strings.
    """
    print("\n" + "=" * 50)
    print("DETECTION SUMMARY")
    print("=" * 50)
    print(f"Total detections: {len(scores)}")
    print("-" * 50)
    print(f"{'Class':<25} {'Count':>6} {'Avg Score':>10}")
    print("-" * 50)

    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        count = int(np.sum(mask))
        if count > 0:
            avg_score = float(np.mean(scores[mask]))
            print(f"{cls_name:<25} {count:>6} {avg_score:>10.3f}")
        else:
            print(f"{cls_name:<25} {count:>6} {'---':>10}")

    print("=" * 50 + "\n")


def run_inference() -> None:
    """Main inference entry point.

    Parses command-line arguments, loads the model and dataset, and
    dispatches to either single-sample or sequence visualization mode.
    Prints a summary of detections upon completion.
    """
    args = _parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for inference.")

    # Load config
    logger.info(f"Loading config from: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Build model
    logger.info("Building BEVFormer model...")
    model = BEVFormer(**config.get("model", {}))
    model = model.to(device)
    model.eval()

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Handle 'module.' prefix from DataParallel/DDP
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_state_dict[key] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    logger.info("Checkpoint loaded successfully.")

    # Build dataset
    logger.info("Building dataset...")
    dataset_cfg = config.get("dataset", {})
    if args.data_root:
        dataset_cfg["data_root"] = args.data_root
    dataset = NuScenesDataset(**dataset_cfg)
    logger.info(f"Dataset loaded with {len(dataset)} samples.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Dispatch to visualization mode
    if args.sequence:
        # Get scene token - use dataset's available scenes
        scene_token = _resolve_scene_token(dataset, args.sample_idx)
        if scene_token is None:
            logger.error(
                "Could not determine scene token. Provide --sample-idx pointing to "
                "a sample within the desired scene."
            )
            sys.exit(1)

        video_path = visualize_sequence(
            model, dataset, scene_token, args.output_dir, args.score_threshold
        )
        logger.info(f"Sequence visualization complete: {video_path}")

        # Print detection summary for last frame
        # Re-run last frame for summary
        scene_indices = _get_scene_sample_indices(dataset, scene_token)
        if scene_indices:
            _run_and_print_summary(model, dataset, scene_indices[-1], device, args.score_threshold)
    else:
        # Single sample mode
        output_path = visualize_single_sample(
            model, dataset, args.sample_idx, args.output_dir, args.score_threshold
        )
        logger.info(f"Single sample visualization complete: {output_path}")

        # Print detection summary
        _run_and_print_summary(model, dataset, args.sample_idx, device, args.score_threshold)


def _resolve_scene_token(dataset: NuScenesDataset, sample_idx: Optional[int]) -> Optional[str]:
    """Resolve a scene token from the dataset using sample_idx as a hint.

    Args:
        dataset: The NuScenesDataset instance.
        sample_idx: Optional sample index to look up the scene for.

    Returns:
        Scene token string, or None if it cannot be determined.
    """
    # If dataset has a list of scenes, use the first one or the one at sample_idx
    if hasattr(dataset, "scenes") and dataset.scenes:
        if sample_idx is not None and sample_idx < len(dataset.scenes):
            scene = dataset.scenes[sample_idx]
            if isinstance(scene, dict):
                return scene.get("token", scene.get("scene_token"))
            elif isinstance(scene, str):
                return scene
            elif hasattr(scene, "token"):
                return scene.token
        # Default to first scene
        scene = dataset.scenes[0]
        if isinstance(scene, dict):
            return scene.get("token", scene.get("scene_token"))
        elif isinstance(scene, str):
            return scene
        elif hasattr(scene, "token"):
            return scene.token

    # Try getting scene token from a specific sample
    if hasattr(dataset, "samples") and sample_idx is not None:
        idx = min(sample_idx, len(dataset.samples) - 1)
        sample_info = dataset.samples[idx]
        if isinstance(sample_info, dict):
            return sample_info.get("scene_token", sample_info.get("scene"))
        elif hasattr(sample_info, "scene_token"):
            return sample_info.scene_token

    return None


def _run_and_print_summary(
    model: BEVFormer,
    dataset: NuScenesDataset,
    sample_idx: int,
    device: torch.device,
    score_threshold: float,
) -> None:
    """Run inference on a single sample and print detection summary.

    Args:
        model: The BEVFormer model in eval mode.
        dataset: The NuScenesDataset instance.
        sample_idx: Index into the dataset.
        device: Torch device.
        score_threshold: Minimum score for filtering.
    """
    sample = dataset[sample_idx]
    batch = collate_fn([sample])

    images = batch["images"].to(device)
    intrinsics = batch["intrinsics"].to(device)
    extrinsics = batch["extrinsics"].to(device)
    ego_motion = batch.get("ego_motion")
    if ego_motion is not None:
        ego_motion = ego_motion.to(device)

    with torch.no_grad():
        detections, _ = model.forward_test(
            images, intrinsics, extrinsics, ego_motion, prev_bev=None
        )

    det_scores = detections["scores"][0].cpu().numpy()
    det_labels = detections["labels"][0].cpu().numpy()

    mask = det_scores >= score_threshold
    _print_detection_summary(det_scores[mask], det_labels[mask], CLASS_NAMES)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inference script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="BEVFormer Inference and Visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for the dataset. Overrides config if provided.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Index of the sample to visualize (single-sample mode).",
    )
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="If set, process a full scene sequence and output video.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/visualizations",
        help="Output directory for visualization results.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence score to display a detection.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use. Set to -1 for CPU.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Use non-interactive backend for headless environments
    matplotlib.use("Agg")
    run_inference()
