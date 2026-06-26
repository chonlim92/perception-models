#!/usr/bin/env python3
"""
visualize_results.py - Visualize DETR3D detection results.

Supports:
- Projecting 3D bounding boxes onto multi-camera images
- Bird's eye view (BEV) visualization
- Multi-camera grid layout (2x3 for all 6 cameras)
- Saving as images or video sequences

Usage:
    # Visualize predictions on camera images
    python scripts/visualize_results.py \
        --predictions results/predictions.pkl \
        --data-root ./data/nuscenes \
        --infos ./data/nuscenes/infos/detr3d_infos_val.pkl \
        --output-dir ./vis_output \
        --mode camera

    # Bird's eye view
    python scripts/visualize_results.py \
        --predictions results/predictions.pkl \
        --infos ./data/nuscenes/infos/detr3d_infos_val.pkl \
        --output-dir ./vis_output \
        --mode bev

    # Create video
    python scripts/visualize_results.py \
        --predictions results/predictions.pkl \
        --data-root ./data/nuscenes \
        --infos ./data/nuscenes/infos/detr3d_infos_val.pkl \
        --output-dir ./vis_output \
        --mode camera \
        --video
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# ============================================================================
# Constants
# ============================================================================

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CLASS_NAMES = [
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

# Color map for each class (BGR for OpenCV)
CLASS_COLORS_BGR = {
    "car": (0, 255, 0),
    "truck": (0, 200, 200),
    "construction_vehicle": (0, 150, 255),
    "bus": (255, 200, 0),
    "trailer": (200, 200, 0),
    "barrier": (128, 128, 128),
    "motorcycle": (255, 0, 200),
    "bicycle": (255, 100, 0),
    "pedestrian": (0, 0, 255),
    "traffic_cone": (0, 128, 255),
}

# Color map for matplotlib (RGB normalized)
CLASS_COLORS_RGB = {
    k: (v[2] / 255.0, v[1] / 255.0, v[0] / 255.0)
    for k, v in CLASS_COLORS_BGR.items()
}

# BEV visualization range (meters)
BEV_RANGE = 51.2
BEV_RESOLUTION = 0.1  # meters per pixel


# ============================================================================
# 3D Box Utilities
# ============================================================================


def get_box_corners_3d(
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """Compute 8 corners of a 3D bounding box.

    Args:
        center: [cx, cy, cz] center of box
        size: [w, l, h] width, length, height
        yaw: rotation angle around z-axis (radians)

    Returns:
        corners: (8, 3) array of corner coordinates in 3D
    """
    w, l, h = size[0], size[1], size[2]

    # 8 corners in local box frame (centered at origin)
    # Bottom face: 4 corners, Top face: 4 corners
    x_corners = np.array([w, w, -w, -w, w, w, -w, -w]) / 2.0
    y_corners = np.array([l, -l, -l, l, l, -l, -l, l]) / 2.0
    z_corners = np.array([-h, -h, -h, -h, h, h, h, h]) / 2.0

    # Rotation matrix around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array(
        [
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1],
        ]
    )

    # Rotate and translate corners
    corners = np.stack([x_corners, y_corners, z_corners], axis=0)  # (3, 8)
    corners = R @ corners  # (3, 8)
    corners = corners.T + center  # (8, 3)

    return corners


def project_points_to_image(
    points_3d: np.ndarray,
    viewmatrix: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) points in global/ego coordinates
        viewmatrix: (3, 4) projection matrix (intrinsic @ extrinsic)
        image_shape: (height, width) of target image

    Returns:
        points_2d: (N, 2) projected points [u, v]
        valid_mask: (N,) boolean mask for points in front of camera and in image
    """
    N = points_3d.shape[0]

    # Convert to homogeneous coordinates
    points_homo = np.concatenate(
        [points_3d, np.ones((N, 1))], axis=1
    )  # (N, 4)

    # Project: (3, 4) @ (4, N) -> (3, N)
    projected = viewmatrix @ points_homo.T  # (3, N)

    # Depth check (points must be in front of camera)
    depths = projected[2, :]
    valid_depth = depths > 0.1  # minimum depth threshold

    # Normalize by depth to get pixel coordinates
    points_2d = np.zeros((N, 2), dtype=np.float64)
    valid_indices = depths > 0.1
    points_2d[valid_indices, 0] = (
        projected[0, valid_indices] / depths[valid_indices]
    )
    points_2d[valid_indices, 1] = (
        projected[1, valid_indices] / depths[valid_indices]
    )

    # Check if points are within image bounds
    h, w = image_shape
    valid_u = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w)
    valid_v = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)

    valid_mask = valid_depth & valid_u & valid_v

    return points_2d, valid_mask


def draw_box_3d_on_image(
    image: np.ndarray,
    corners_2d: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw projected 3D bounding box edges on an image.

    The box has 12 edges: 4 bottom, 4 top, 4 vertical pillars.
    """
    # Define edge connections (pairs of corner indices)
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical pillars
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    corners_int = corners_2d.astype(np.int32)

    for start, end in edges:
        pt1 = tuple(corners_int[start])
        pt2 = tuple(corners_int[end])
        cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw front face with thicker line to indicate heading
    front_edges = [(0, 1), (4, 5), (0, 4), (1, 5)]
    for start, end in front_edges:
        pt1 = tuple(corners_int[start])
        pt2 = tuple(corners_int[end])
        cv2.line(image, pt1, pt2, color, thickness + 1, cv2.LINE_AA)

    return image


def draw_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw text label with background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    # Background rectangle
    cv2.rectangle(
        image,
        (x, y - text_size[1] - 4),
        (x + text_size[0] + 4, y + 2),
        color,
        -1,
    )
    # Text (white on colored background)
    cv2.putText(
        image,
        text,
        (x + 2, y - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return image


# ============================================================================
# Camera Visualization
# ============================================================================


def visualize_camera_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    viewmatrix: np.ndarray,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Draw 3D detection boxes projected onto a single camera image.

    Args:
        image: Camera image (H, W, 3) BGR
        detections: List of detection dicts with 'center', 'size', 'yaw',
                    'class_name', 'score'
        viewmatrix: (3, 4) camera projection matrix
        score_threshold: Minimum confidence to display

    Returns:
        Annotated image
    """
    vis_image = image.copy()
    h, w = vis_image.shape[:2]

    for det in detections:
        if det["score"] < score_threshold:
            continue

        class_name = det["class_name"]
        score = det["score"]
        color = CLASS_COLORS_BGR.get(class_name, (255, 255, 255))

        # Get 3D corners
        corners_3d = get_box_corners_3d(det["center"], det["size"], det["yaw"])

        # Project to image
        corners_2d, valid_mask = project_points_to_image(
            corners_3d, viewmatrix, (h, w)
        )

        # Only draw if at least 4 corners are visible
        if valid_mask.sum() < 4:
            continue

        # Draw the box
        draw_box_3d_on_image(vis_image, corners_2d, color, thickness=2)

        # Draw label at top-left corner of visible projection
        visible_corners = corners_2d[valid_mask]
        label_x = int(visible_corners[:, 0].min())
        label_y = int(visible_corners[:, 1].min())
        label_y = max(label_y, 15)

        label_text = f"{class_name} {score:.2f}"
        draw_label(vis_image, label_text, (label_x, label_y), color)

    return vis_image


def create_multicamera_grid(
    images: Dict[str, np.ndarray],
    target_size: Tuple[int, int] = (400, 711),
) -> np.ndarray:
    """Create a 2x3 grid visualization of all 6 cameras.

    Layout:
        [FRONT_LEFT]  [FRONT]  [FRONT_RIGHT]
        [BACK_LEFT]   [BACK]   [BACK_RIGHT]

    Args:
        images: Dict mapping camera name to annotated image
        target_size: (height, width) to resize each camera image

    Returns:
        Grid image (2*h, 3*w, 3)
    """
    grid_layout = [
        ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
        ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
    ]

    h, w = target_size
    grid = np.zeros((2 * h, 3 * w, 3), dtype=np.uint8)

    for row_idx, row in enumerate(grid_layout):
        for col_idx, cam_name in enumerate(row):
            if cam_name in images:
                img = images[cam_name]
                img_resized = cv2.resize(img, (w, h))

                # Add camera name label
                cv2.putText(
                    img_resized,
                    cam_name,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                y_start = row_idx * h
                x_start = col_idx * w
                grid[y_start : y_start + h, x_start : x_start + w] = img_resized

    return grid


# ============================================================================
# BEV Visualization
# ============================================================================


def create_bev_image(
    detections: List[Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    bev_range: float = BEV_RANGE,
    resolution: float = BEV_RESOLUTION,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Create bird's eye view visualization of detections.

    Args:
        detections: List of predicted detection dicts
        ground_truth: Optional list of GT annotation dicts
        bev_range: Range in meters from ego center
        resolution: Meters per pixel
        score_threshold: Minimum confidence to display

    Returns:
        BEV visualization image
    """
    # Image size
    img_size = int(2 * bev_range / resolution)
    bev_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Draw grid lines
    grid_spacing_m = 10.0  # meters
    grid_spacing_px = int(grid_spacing_m / resolution)
    center = img_size // 2

    for i in range(0, img_size, grid_spacing_px):
        cv2.line(bev_img, (i, 0), (i, img_size), (40, 40, 40), 1)
        cv2.line(bev_img, (0, i), (img_size, i), (40, 40, 40), 1)

    # Draw axes
    cv2.line(bev_img, (center, 0), (center, img_size), (60, 60, 60), 1)
    cv2.line(bev_img, (0, center), (img_size, center), (60, 60, 60), 1)

    # Draw ego vehicle
    ego_size_px = int(4.5 / resolution)  # ~4.5m car length
    ego_width_px = int(2.0 / resolution)
    cv2.rectangle(
        bev_img,
        (center - ego_width_px // 2, center - ego_size_px // 2),
        (center + ego_width_px // 2, center + ego_size_px // 2),
        (255, 255, 255),
        2,
    )

    def world_to_bev(x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to BEV pixel coordinates."""
        px = int(center + x / resolution)
        py = int(center - y / resolution)  # y is flipped
        return (px, py)

    def draw_bev_box(
        img: np.ndarray,
        det: Dict[str, Any],
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        """Draw a single detection as a rotated rectangle in BEV."""
        cx, cy = det["center"][0], det["center"][1]
        w, l = det["size"][0], det["size"][1]
        yaw = det["yaw"]

        # Compute 4 corners in world frame (top-down, ignoring z)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        dx = np.array([w / 2 * cos_yaw, w / 2 * sin_yaw])
        dy = np.array([-l / 2 * sin_yaw, l / 2 * cos_yaw])

        corners = np.array(
            [
                [cx + dx[0] + dy[0], cy + dx[1] + dy[1]],
                [cx + dx[0] - dy[0], cy + dx[1] - dy[1]],
                [cx - dx[0] - dy[0], cy - dx[1] - dy[1]],
                [cx - dx[0] + dy[0], cy - dx[1] + dy[1]],
            ]
        )

        # Convert to BEV pixels
        corners_px = np.array(
            [world_to_bev(c[0], c[1]) for c in corners], dtype=np.int32
        )

        # Draw rotated rectangle
        cv2.polylines(img, [corners_px], True, color, thickness, cv2.LINE_AA)

        # Draw heading direction (front edge)
        front_center = (
            (corners_px[0] + corners_px[1]) / 2
        ).astype(np.int32)
        cv2.circle(img, tuple(front_center), 3, color, -1)

    # Draw ground truth (if available) with dashed appearance (thinner lines)
    if ground_truth is not None:
        for gt in ground_truth:
            class_name = gt["class_name"]
            color = CLASS_COLORS_BGR.get(class_name, (128, 128, 128))
            # Slightly dimmer color for GT
            color = tuple(int(c * 0.6) for c in color)
            draw_bev_box(bev_img, gt, color, thickness=1)

    # Draw predictions
    for det in detections:
        if det["score"] < score_threshold:
            continue

        class_name = det["class_name"]
        color = CLASS_COLORS_BGR.get(class_name, (255, 255, 255))
        draw_bev_box(bev_img, det, color, thickness=2)

        # Draw score text
        px, py = world_to_bev(det["center"][0], det["center"][1])
        if 0 <= px < img_size and 0 <= py < img_size:
            score_text = f"{det['score']:.2f}"
            cv2.putText(
                bev_img,
                score_text,
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )

    # Add legend
    legend_x = 10
    legend_y = 30
    for cls_name in CLASS_NAMES:
        color = CLASS_COLORS_BGR[cls_name]
        cv2.rectangle(
            bev_img,
            (legend_x, legend_y - 12),
            (legend_x + 15, legend_y),
            color,
            -1,
        )
        cv2.putText(
            bev_img,
            cls_name,
            (legend_x + 20, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        legend_y += 20

    # Add range info
    cv2.putText(
        bev_img,
        f"Range: +/-{bev_range}m",
        (img_size - 180, img_size - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return bev_img


# ============================================================================
# Main Visualization Pipeline
# ============================================================================


def load_predictions(pred_path: str) -> List[Dict[str, Any]]:
    """Load prediction results from pickle file.

    Expected format: List of dicts, one per sample, each containing:
        - 'token': sample token
        - 'boxes': List of dicts with 'center', 'size', 'yaw', 'class_name', 'score'
    """
    with open(pred_path, "rb") as f:
        predictions = pickle.load(f)
    return predictions


def load_infos(infos_path: str) -> List[Dict[str, Any]]:
    """Load prepared info files."""
    with open(infos_path, "rb") as f:
        infos = pickle.load(f)
    return infos


def visualize_sample_cameras(
    sample_info: Dict[str, Any],
    detections: List[Dict[str, Any]],
    data_root: str,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Visualize detections on all 6 cameras for a single sample.

    Returns the multi-camera grid image.
    """
    camera_images = {}

    for cam_name in CAMERA_NAMES:
        cam_info = sample_info["cameras"][cam_name]
        img_path = os.path.join(data_root, cam_info["data_path"])

        # Load image
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
        else:
            # Create placeholder if image not available
            h, w = cam_info["height"], cam_info["width"]
            image = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                image,
                f"Image not found: {cam_info['data_path']}",
                (50, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Get view matrix for this camera
        viewmatrix = cam_info["viewmatrix"]

        # Draw detections
        annotated = visualize_camera_detections(
            image, detections, viewmatrix, score_threshold
        )
        camera_images[cam_name] = annotated

    # Create 2x3 grid
    grid = create_multicamera_grid(camera_images)
    return grid


def visualize_sample_bev(
    sample_info: Dict[str, Any],
    detections: List[Dict[str, Any]],
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Create BEV visualization for a single sample with predictions and GT."""
    ground_truth = []
    for ann in sample_info["annotations"]:
        ground_truth.append(
            {
                "center": ann["center_ego"],
                "size": ann["size"],
                "yaw": ann["yaw_ego"],
                "class_name": ann["class_name"],
            }
        )

    bev_img = create_bev_image(
        detections,
        ground_truth=ground_truth,
        score_threshold=score_threshold,
    )
    return bev_img


def run_visualization(
    predictions_path: str,
    infos_path: str,
    data_root: str,
    output_dir: str,
    mode: str = "camera",
    score_threshold: float = 0.3,
    max_samples: Optional[int] = None,
    create_video: bool = False,
    fps: int = 2,
) -> None:
    """Main visualization function.

    Args:
        predictions_path: Path to predictions pickle file
        infos_path: Path to info pickle file
        data_root: Path to nuScenes data root
        output_dir: Directory to save visualization outputs
        mode: 'camera', 'bev', or 'both'
        score_threshold: Minimum confidence to visualize
        max_samples: Limit number of samples to visualize
        create_video: Whether to create video from frames
        fps: Frames per second for video output
    """
    print(f"Loading predictions from: {predictions_path}")
    predictions = load_predictions(predictions_path)

    print(f"Loading infos from: {infos_path}")
    infos = load_infos(infos_path)

    # Build token-to-prediction mapping
    pred_by_token = {}
    for pred in predictions:
        pred_by_token[pred["token"]] = pred.get("boxes", [])

    os.makedirs(output_dir, exist_ok=True)

    num_samples = len(infos)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    print(f"Visualizing {num_samples} samples (mode={mode})...")

    video_frames_camera = []
    video_frames_bev = []

    for idx in range(num_samples):
        info = infos[idx]
        token = info["token"]
        detections = pred_by_token.get(token, [])

        if idx % 10 == 0:
            print(
                f"  Processing sample {idx + 1}/{num_samples} "
                f"({len(detections)} detections)..."
            )

        if mode in ("camera", "both"):
            grid_img = visualize_sample_cameras(
                info, detections, data_root, score_threshold
            )

            # Save individual frame
            out_path = os.path.join(output_dir, f"camera_{idx:06d}.jpg")
            cv2.imwrite(out_path, grid_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            if create_video:
                video_frames_camera.append(grid_img)

        if mode in ("bev", "both"):
            bev_img = visualize_sample_bev(info, detections, score_threshold)

            # Save individual frame
            out_path = os.path.join(output_dir, f"bev_{idx:06d}.jpg")
            cv2.imwrite(out_path, bev_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            if create_video:
                video_frames_bev.append(bev_img)

    # Create videos if requested
    if create_video and video_frames_camera:
        video_path = os.path.join(output_dir, "camera_detections.mp4")
        print(f"Creating camera video: {video_path}")
        h, w = video_frames_camera[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        for frame in video_frames_camera:
            writer.write(frame)
        writer.release()
        print(f"  Saved: {video_path}")

    if create_video and video_frames_bev:
        video_path = os.path.join(output_dir, "bev_detections.mp4")
        print(f"Creating BEV video: {video_path}")
        h, w = video_frames_bev[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        for frame in video_frames_bev:
            writer.write(frame)
        writer.release()
        print(f"  Saved: {video_path}")

    print(f"\nVisualization complete! Output saved to: {output_dir}")


def create_summary_figure(
    predictions_path: str,
    infos_path: str,
    output_path: str,
) -> None:
    """Create a matplotlib summary figure with detection statistics.

    Shows per-class score distributions, detection counts, and
    confidence histograms.
    """
    predictions = load_predictions(predictions_path)
    infos = load_infos(infos_path)

    # Collect all detection scores by class
    class_scores = {cls: [] for cls in CLASS_NAMES}
    class_counts_pred = {cls: 0 for cls in CLASS_NAMES}
    class_counts_gt = {cls: 0 for cls in CLASS_NAMES}

    for pred in predictions:
        for box in pred.get("boxes", []):
            cls = box["class_name"]
            if cls in class_scores:
                class_scores[cls].append(box["score"])
                class_counts_pred[cls] += 1

    for info in infos:
        for ann in info["annotations"]:
            cls = ann["class_name"]
            if cls in class_counts_gt:
                class_counts_gt[cls] += 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Detection counts comparison
    ax = axes[0, 0]
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    gt_counts = [class_counts_gt[c] for c in CLASS_NAMES]
    pred_counts = [class_counts_pred[c] for c in CLASS_NAMES]
    ax.bar(x - width / 2, gt_counts, width, label="Ground Truth", alpha=0.7)
    ax.bar(x + width / 2, pred_counts, width, label="Predictions", alpha=0.7)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Detection Counts: GT vs Predictions")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.legend()

    # Plot 2: Score distribution per class
    ax = axes[0, 1]
    scores_data = [class_scores[c] for c in CLASS_NAMES if class_scores[c]]
    labels = [c for c in CLASS_NAMES if class_scores[c]]
    if scores_data:
        ax.boxplot(scores_data, labels=labels)
        ax.set_xlabel("Class")
        ax.set_ylabel("Confidence Score")
        ax.set_title("Score Distribution by Class")
        ax.tick_params(axis="x", rotation=45)

    # Plot 3: Overall score histogram
    ax = axes[1, 0]
    all_scores = []
    for scores in class_scores.values():
        all_scores.extend(scores)
    if all_scores:
        ax.hist(all_scores, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(
            x=0.3, color="r", linestyle="--", label="Threshold=0.3"
        )
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        ax.set_title("Overall Score Distribution")
        ax.legend()

    # Plot 4: Per-class recall at different thresholds
    ax = axes[1, 1]
    thresholds = np.arange(0.1, 1.0, 0.05)
    for cls in CLASS_NAMES[:5]:  # Top 5 classes for readability
        if not class_scores[cls]:
            continue
        scores_arr = np.array(class_scores[cls])
        recall_curve = [
            (scores_arr >= t).sum() / max(class_counts_gt[cls], 1)
            for t in thresholds
        ]
        color = CLASS_COLORS_RGB.get(cls, (0.5, 0.5, 0.5))
        ax.plot(thresholds, recall_curve, label=cls, color=color)
    ax.set_xlabel("Score Threshold")
    ax.set_ylabel("Recall (approx)")
    ax.set_title("Approximate Recall vs Threshold")
    ax.legend(fontsize=8)
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary figure saved: {output_path}")


# ============================================================================
# Entry Point
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize DETR3D detection results"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions pickle file",
    )
    parser.add_argument(
        "--infos",
        type=str,
        required=True,
        help="Path to dataset info pickle file",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/nuscenes",
        help="Path to nuScenes data root (needed for camera mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vis_output",
        help="Output directory for visualization results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["camera", "bev", "both"],
        default="camera",
        help="Visualization mode: camera, bev, or both",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence score to display (default: 0.3)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to visualize",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Create video from visualization frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="FPS for video output (default: 2)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Create summary statistics figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.summary:
        summary_path = os.path.join(args.output_dir, "detection_summary.png")
        os.makedirs(args.output_dir, exist_ok=True)
        create_summary_figure(args.predictions, args.infos, summary_path)
    else:
        run_visualization(
            predictions_path=args.predictions,
            infos_path=args.infos,
            data_root=args.data_root,
            output_dir=args.output_dir,
            mode=args.mode,
            score_threshold=args.score_threshold,
            max_samples=args.max_samples,
            create_video=args.video,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
