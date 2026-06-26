#!/usr/bin/env python3
"""
Visualize BEVFormer predictions and ground truth.

Supports:
- 3D box projection onto camera images
- BEV top-down view with oriented bounding boxes
- Side-by-side GT vs prediction comparison
- Video generation for full scenes

Usage:
    python visualize_results.py --predictions preds.json --data_root data/nuscenes \
        --info_file nuscenes_infos_temporal_val.pkl --output_dir vis_output

    python visualize_results.py --predictions preds.json --data_root data/nuscenes \
        --info_file nuscenes_infos_temporal_val.pkl --sample_token <token> --output_dir vis_output

    python visualize_results.py --predictions preds.json --data_root data/nuscenes \
        --info_file nuscenes_infos_temporal_val.pkl --make_video --output_dir vis_output
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

matplotlib.use("Agg")


# Class-specific colors (BGR for OpenCV, will convert for matplotlib)
CLASS_COLORS = {
    "car": (0, 255, 0),           # Green
    "truck": (0, 200, 200),       # Yellow-ish
    "construction_vehicle": (0, 128, 255),  # Orange
    "bus": (255, 128, 0),         # Blue-ish
    "trailer": (128, 0, 255),     # Purple
    "barrier": (255, 0, 128),     # Pink
    "motorcycle": (0, 255, 255),  # Cyan
    "bicycle": (255, 255, 0),     # Light Blue
    "pedestrian": (0, 0, 255),    # Red
    "traffic_cone": (128, 128, 255),  # Light Red
}

# Camera layout for multi-camera visualization
CAMERA_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

# BEV visualization parameters
BEV_RANGE = 51.2  # meters from center
BEV_RESOLUTION = 0.256  # meters per pixel (200 pixels for 51.2m)
BEV_SIZE = 400  # pixels (400x400 for the full 102.4m range)


def bgr_to_rgb_normalized(color_bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert BGR (0-255) to RGB (0-1) for matplotlib."""
    return (color_bgr[2] / 255.0, color_bgr[1] / 255.0, color_bgr[0] / 255.0)


def get_3d_box_corners(bbox_3d: List[float]) -> np.ndarray:
    """
    Compute 8 corners of a 3D bounding box.

    Args:
        bbox_3d: [cx, cy, cz, w, l, h, yaw]

    Returns:
        corners: (8, 3) array of 3D corner coordinates
    """
    cx, cy, cz, w, l, h, yaw = bbox_3d

    # 8 corners in object frame (before rotation)
    # Order: bottom 4 corners then top 4 corners
    # Each face: front-left, front-right, back-right, back-left
    dx = l / 2.0
    dy = w / 2.0
    dz = h / 2.0

    corners = np.array([
        [ dx,  dy, -dz],  # front-left-bottom
        [ dx, -dy, -dz],  # front-right-bottom
        [-dx, -dy, -dz],  # back-right-bottom
        [-dx,  dy, -dz],  # back-left-bottom
        [ dx,  dy,  dz],  # front-left-top
        [ dx, -dy,  dz],  # front-right-top
        [-dx, -dy,  dz],  # back-right-top
        [-dx,  dy,  dz],  # back-left-top
    ], dtype=np.float64)

    # Rotation matrix around Z axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1],
    ], dtype=np.float64)

    # Rotate and translate
    corners = (rotation @ corners.T).T
    corners[:, 0] += cx
    corners[:, 1] += cy
    corners[:, 2] += cz

    return corners


def project_corners_to_image(
    corners_3d: np.ndarray,
    lidar2img: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Project 3D corners onto image plane using lidar2img matrix.

    Args:
        corners_3d: (8, 3) 3D corner coordinates in lidar/ego frame
        lidar2img: (4, 4) projection matrix

    Returns:
        corners_2d: (8, 2) pixel coordinates, or None if behind camera
        depths: (8,) depth values, or None
    """
    # Homogeneous coordinates
    corners_homo = np.hstack([corners_3d, np.ones((8, 1))])  # (8, 4)

    # Project
    projected = (lidar2img @ corners_homo.T).T  # (8, 4)

    # Check depth (z > 0 means in front of camera)
    depths = projected[:, 2]
    if np.all(depths <= 0):
        return None, None

    # Normalize by depth
    mask = depths > 0
    corners_2d = np.zeros((8, 2), dtype=np.float64)
    corners_2d[mask, 0] = projected[mask, 0] / depths[mask]
    corners_2d[mask, 1] = projected[mask, 1] / depths[mask]

    # For points behind camera, set to invalid
    corners_2d[~mask] = -1000

    return corners_2d, depths


def draw_3d_box_on_image(
    image: np.ndarray,
    corners_2d: np.ndarray,
    depths: np.ndarray,
    color: Tuple[int, int, int],
    label: str = "",
    score: float = 0.0,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a wireframe 3D box on the image.

    The box edges connect:
    - Bottom face: 0-1-2-3-0
    - Top face: 4-5-6-7-4
    - Pillars: 0-4, 1-5, 2-6, 3-7
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Edge definitions
    bottom_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]
    pillar_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
    # Front face highlighted
    front_edges = [(0, 1), (4, 5), (0, 4), (1, 5)]

    all_edges = bottom_edges + top_edges + pillar_edges

    def is_valid_point(pt: np.ndarray, depth: float) -> bool:
        return depth > 0 and 0 <= pt[0] < w and 0 <= pt[1] < h

    # Draw edges
    for i, j in all_edges:
        if depths[i] > 0 and depths[j] > 0:
            pt1 = (int(round(corners_2d[i, 0])), int(round(corners_2d[i, 1])))
            pt2 = (int(round(corners_2d[j, 0])), int(round(corners_2d[j, 1])))

            # Clip to reasonable bounds
            if (-w < pt1[0] < 2 * w and -h < pt1[1] < 2 * h and
                -w < pt2[0] < 2 * w and -h < pt2[1] < 2 * h):
                edge_thickness = thickness + 1 if (i, j) in front_edges else thickness
                cv2.line(img, pt1, pt2, color, edge_thickness)

    # Draw label with score
    if label:
        # Find the topmost visible corner for label placement
        valid_corners = [(corners_2d[i, 1], i) for i in range(8)
                        if depths[i] > 0 and 0 <= corners_2d[i, 0] < w and 0 <= corners_2d[i, 1] < h]
        if valid_corners:
            valid_corners.sort()
            top_idx = valid_corners[0][1]
            label_pos = (
                int(round(corners_2d[top_idx, 0])),
                max(15, int(round(corners_2d[top_idx, 1])) - 5),
            )

            text = f"{label} {score:.2f}" if score > 0 else label
            font_scale = 0.4
            font_thickness = 1

            # Background rectangle for readability
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            cv2.rectangle(
                img,
                (label_pos[0], label_pos[1] - text_h - 4),
                (label_pos[0] + text_w, label_pos[1] + 2),
                color,
                -1,
            )
            cv2.putText(
                img, text, (label_pos[0], label_pos[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness,
            )

    return img


def draw_boxes_on_camera(
    image: np.ndarray,
    boxes: List[Dict[str, Any]],
    lidar2img: np.ndarray,
    score_threshold: float = 0.3,
    is_gt: bool = False,
) -> np.ndarray:
    """
    Draw all 3D boxes on a camera image.

    Args:
        image: camera image (BGR)
        boxes: list of box dicts with 'bbox_3d', 'category', optionally 'score'
        lidar2img: 4x4 projection matrix for this camera
        score_threshold: minimum score to draw
        is_gt: if True, label as GT
    """
    img = image.copy()
    lidar2img_mat = np.array(lidar2img, dtype=np.float64)

    for box in boxes:
        score = box.get("score", 1.0)
        if score < score_threshold:
            continue

        category = box.get("category", "unknown")
        bbox_3d = box["bbox_3d"]
        color = CLASS_COLORS.get(category, (200, 200, 200))

        # Get 3D corners
        corners_3d = get_3d_box_corners(bbox_3d)

        # Project to image
        corners_2d, depths = project_corners_to_image(corners_3d, lidar2img_mat)
        if corners_2d is None:
            continue

        # Check if any corner is visible
        h, w = img.shape[:2]
        visible = False
        for i in range(8):
            if depths[i] > 0 and 0 <= corners_2d[i, 0] < w and 0 <= corners_2d[i, 1] < h:
                visible = True
                break

        if not visible:
            continue

        # Draw box
        label = f"GT:{category}" if is_gt else category
        img = draw_3d_box_on_image(
            img, corners_2d, depths, color,
            label=label, score=score if not is_gt else 0.0,
            thickness=2 if not is_gt else 1,
        )

    return img


def create_bev_image(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    score_threshold: float = 0.3,
    bev_size: int = BEV_SIZE,
    bev_range: float = BEV_RANGE,
) -> np.ndarray:
    """
    Create a BEV (Bird's Eye View) visualization.

    Shows:
    - Gray background representing the driving area
    - Ego vehicle at center
    - Predicted boxes as solid oriented rectangles
    - GT boxes as dashed outlines

    Returns:
        BEV image as BGR numpy array
    """
    # Create gray background
    bev_img = np.ones((bev_size, bev_size, 3), dtype=np.uint8) * 80

    # Draw grid lines (every 10 meters)
    meters_per_pixel = (2 * bev_range) / bev_size
    grid_spacing = 10.0  # meters
    grid_pixels = int(grid_spacing / meters_per_pixel)

    center = bev_size // 2
    for i in range(-int(bev_range / grid_spacing), int(bev_range / grid_spacing) + 1):
        offset = int(i * grid_pixels)
        # Vertical lines
        cv2.line(bev_img, (center + offset, 0), (center + offset, bev_size), (60, 60, 60), 1)
        # Horizontal lines
        cv2.line(bev_img, (0, center + offset), (bev_size, center + offset), (60, 60, 60), 1)

    # Draw range circles (every 20 meters)
    for r_meters in range(20, int(bev_range) + 1, 20):
        r_pixels = int(r_meters / meters_per_pixel)
        cv2.circle(bev_img, (center, center), r_pixels, (70, 70, 70), 1)

    # Draw ego vehicle at center (as a filled rectangle)
    ego_length = int(4.5 / meters_per_pixel)  # ~4.5m car length
    ego_width = int(2.0 / meters_per_pixel)   # ~2.0m car width
    ego_rect = np.array([
        [center - ego_width // 2, center - ego_length // 2],
        [center + ego_width // 2, center - ego_length // 2],
        [center + ego_width // 2, center + ego_length // 2],
        [center - ego_width // 2, center + ego_length // 2],
    ], dtype=np.int32)
    cv2.fillPoly(bev_img, [ego_rect], (200, 200, 200))
    cv2.polylines(bev_img, [ego_rect], True, (255, 255, 255), 1)

    # Draw forward direction arrow
    arrow_start = (center, center - ego_length // 2)
    arrow_end = (center, center - ego_length // 2 - int(3.0 / meters_per_pixel))
    cv2.arrowedLine(bev_img, arrow_start, arrow_end, (255, 255, 255), 2, tipLength=0.3)

    def world_to_bev(x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (ego frame) to BEV pixel coordinates."""
        # x is forward (up in image), y is left (left in image)
        px = center - int(y / meters_per_pixel)
        py = center - int(x / meters_per_pixel)
        return (px, py)

    def draw_oriented_box_bev(
        img: np.ndarray,
        bbox_3d: List[float],
        color: Tuple[int, int, int],
        thickness: int = 2,
        is_dashed: bool = False,
    ) -> np.ndarray:
        """Draw an oriented box in BEV."""
        cx, cy, cz, w, l, h, yaw = bbox_3d

        # 4 corners in BEV (top-down, ignoring z)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        dx = l / 2.0
        dy = w / 2.0

        corners_local = np.array([
            [ dx,  dy],
            [ dx, -dy],
            [-dx, -dy],
            [-dx,  dy],
        ])

        # Rotate
        rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners_world = (rot @ corners_local.T).T
        corners_world[:, 0] += cx
        corners_world[:, 1] += cy

        # Convert to pixel coordinates
        corners_px = np.array([world_to_bev(c[0], c[1]) for c in corners_world], dtype=np.int32)

        if is_dashed:
            # Draw dashed lines
            for i in range(4):
                pt1 = tuple(corners_px[i])
                pt2 = tuple(corners_px[(i + 1) % 4])
                _draw_dashed_line(img, pt1, pt2, color, thickness)
        else:
            cv2.polylines(img, [corners_px], True, color, thickness)

        # Draw heading indicator (line from center to front)
        front_center = world_to_bev(
            cx + dx * cos_yaw,
            cy + dx * sin_yaw,
        )
        center_px = world_to_bev(cx, cy)
        if not is_dashed:
            cv2.line(img, center_px, front_center, color, thickness)

        return img

    def _draw_dashed_line(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        dash_length: int = 5,
    ):
        """Draw a dashed line between two points."""
        dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        if dist < 1:
            return
        num_dashes = int(dist / dash_length)
        if num_dashes < 1:
            num_dashes = 1

        for i in range(0, num_dashes, 2):
            start_frac = i / num_dashes
            end_frac = min((i + 1) / num_dashes, 1.0)
            start = (
                int(pt1[0] + (pt2[0] - pt1[0]) * start_frac),
                int(pt1[1] + (pt2[1] - pt1[1]) * start_frac),
            )
            end = (
                int(pt1[0] + (pt2[0] - pt1[0]) * end_frac),
                int(pt1[1] + (pt2[1] - pt1[1]) * end_frac),
            )
            cv2.line(img, start, end, color, thickness)

    # Draw GT boxes (dashed, behind predictions)
    for gt in ground_truths:
        category = gt.get("category", "unknown")
        color = CLASS_COLORS.get(category, (200, 200, 200))
        # Slightly dimmer color for GT
        dim_color = tuple(max(0, c - 50) for c in color)
        bev_img = draw_oriented_box_bev(bev_img, gt["bbox_3d"], dim_color, thickness=1, is_dashed=True)

    # Draw predicted boxes (solid)
    for pred in predictions:
        score = pred.get("score", 1.0)
        if score < score_threshold:
            continue
        category = pred.get("category", "unknown")
        color = CLASS_COLORS.get(category, (200, 200, 200))
        bev_img = draw_oriented_box_bev(bev_img, pred["bbox_3d"], color, thickness=2, is_dashed=False)

    # Draw legend
    legend_y = 15
    for cat, color in CLASS_COLORS.items():
        cv2.rectangle(bev_img, (5, legend_y - 10), (15, legend_y), color, -1)
        cv2.putText(bev_img, cat, (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        legend_y += 15

    # Draw scale bar
    scale_bar_meters = 10.0
    scale_bar_pixels = int(scale_bar_meters / meters_per_pixel)
    cv2.line(bev_img, (bev_size - scale_bar_pixels - 10, bev_size - 20),
             (bev_size - 10, bev_size - 20), (255, 255, 255), 2)
    cv2.putText(bev_img, f"{scale_bar_meters:.0f}m",
                (bev_size - scale_bar_pixels - 10, bev_size - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return bev_img


def create_multicam_visualization(
    data_root: str,
    camera_infos: List[Dict[str, Any]],
    lidar2img_list: List[List],
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    score_threshold: float = 0.3,
) -> np.ndarray:
    """
    Create a multi-camera visualization with all 6 cameras arranged in a 2x3 grid.

    Returns:
        Combined image as BGR numpy array
    """
    # Map camera name to info
    cam_map = {info["cam_name"]: (info, lidar2img) for info, lidar2img
               in zip(camera_infos, lidar2img_list)}

    images = []
    target_height = 450
    target_width = 800

    for cam_name in CAMERA_NAMES:
        if cam_name in cam_map:
            cam_info, lidar2img = cam_map[cam_name]

            # Load image
            img_path = os.path.join(data_root, cam_info["filename"])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
            else:
                img = np.zeros((900, 1600, 3), dtype=np.uint8)
                cv2.putText(img, f"Image not found: {cam_info['filename']}",
                           (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw predictions
            img = draw_boxes_on_camera(img, predictions, lidar2img, score_threshold, is_gt=False)

            # Draw GT (dimmer/thinner)
            img = draw_boxes_on_camera(img, ground_truths, lidar2img, 0.0, is_gt=True)

            # Add camera name
            cv2.putText(img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Resize
            img = cv2.resize(img, (target_width, target_height))
        else:
            img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            cv2.putText(img, f"No data: {cam_name}", (50, target_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        images.append(img)

    # Arrange in 2x3 grid
    top_row = np.hstack(images[0:3])
    bottom_row = np.hstack(images[3:6])
    multicam = np.vstack([top_row, bottom_row])

    return multicam


def create_comparison_figure(
    data_root: str,
    sample_info: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    score_threshold: float = 0.3,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create a side-by-side comparison visualization.

    Layout:
    - Top: Multi-camera view with boxes
    - Bottom-left: BEV with predictions + GT
    - Bottom-right: Stats and legend

    Returns:
        Combined visualization image
    """
    camera_infos = sample_info["cameras"]
    lidar2img_list = sample_info["lidar2img"]

    # Create multi-camera visualization
    multicam = create_multicam_visualization(
        data_root, camera_infos, lidar2img_list,
        predictions, ground_truths, score_threshold,
    )

    # Create BEV visualization
    bev_img = create_bev_image(predictions, ground_truths, score_threshold)

    # Scale BEV to match the multi-camera width
    multicam_h, multicam_w = multicam.shape[:2]
    bev_target_h = int(multicam_w * 0.4)  # BEV height is 40% of multicam width
    bev_resized = cv2.resize(bev_img, (bev_target_h, bev_target_h))

    # Create stats panel
    stats_w = multicam_w - bev_target_h
    stats_h = bev_target_h
    stats_panel = np.ones((stats_h, stats_w, 3), dtype=np.uint8) * 40

    # Write stats
    y_offset = 30
    line_height = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    cv2.putText(stats_panel, "=== Detection Results ===", (10, y_offset),
                font, 0.6, (255, 255, 255), 1)
    y_offset += line_height + 10

    # Count predictions by category
    pred_counts = defaultdict(int)
    for pred in predictions:
        if pred.get("score", 1.0) >= score_threshold:
            pred_counts[pred.get("category", "unknown")] += 1

    gt_counts = defaultdict(int)
    for gt in ground_truths:
        gt_counts[gt.get("category", "unknown")] += 1

    cv2.putText(stats_panel, f"Score threshold: {score_threshold:.2f}", (10, y_offset),
                font, font_scale, (200, 200, 200), 1)
    y_offset += line_height

    cv2.putText(stats_panel, f"Predictions: {sum(pred_counts.values())}", (10, y_offset),
                font, font_scale, (0, 255, 0), 1)
    y_offset += line_height

    cv2.putText(stats_panel, f"Ground Truth: {sum(gt_counts.values())}", (10, y_offset),
                font, font_scale, (0, 200, 200), 1)
    y_offset += line_height + 10

    cv2.putText(stats_panel, "Category      Pred   GT", (10, y_offset),
                font, font_scale, (255, 255, 255), 1)
    y_offset += line_height

    cv2.line(stats_panel, (10, y_offset - 5), (stats_w - 10, y_offset - 5), (100, 100, 100), 1)
    y_offset += 5

    for cat in CLASS_COLORS:
        color = CLASS_COLORS[cat]
        pred_n = pred_counts.get(cat, 0)
        gt_n = gt_counts.get(cat, 0)
        text = f"{cat:18s} {pred_n:>3d}  {gt_n:>3d}"
        cv2.putText(stats_panel, text, (10, y_offset), font, font_scale, color, 1)
        y_offset += line_height

        if y_offset > stats_h - 20:
            break

    # Legend at the bottom
    y_offset = stats_h - 40
    cv2.putText(stats_panel, "Solid = Prediction, Dashed = GT", (10, y_offset),
                font, font_scale, (180, 180, 180), 1)

    # Combine bottom row
    bottom_row = np.hstack([bev_resized, stats_panel])

    # Combine all
    # Ensure widths match
    if bottom_row.shape[1] != multicam_w:
        bottom_row = cv2.resize(bottom_row, (multicam_w, bev_target_h))

    full_vis = np.vstack([multicam, bottom_row])

    if output_path:
        cv2.imwrite(output_path, full_vis)

    return full_vis


def load_predictions(pred_path: str) -> Dict[str, List[Dict]]:
    """
    Load predictions from JSON file.

    Expected format:
    [
        {
            "sample_token": "...",
            "boxes": [[cx, cy, cz, w, l, h, yaw], ...],
            "scores": [0.9, 0.8, ...],
            "labels": ["car", "pedestrian", ...]
        },
        ...
    ]

    Returns:
        Dict mapping sample_token -> list of box dicts
    """
    with open(pred_path, "r") as f:
        predictions_raw = json.load(f)

    predictions_by_token = {}
    for pred in predictions_raw:
        token = pred["sample_token"]
        boxes = []
        for i, bbox in enumerate(pred["boxes"]):
            box_dict = {
                "bbox_3d": bbox,
                "score": pred["scores"][i] if i < len(pred["scores"]) else 1.0,
                "category": pred["labels"][i] if i < len(pred["labels"]) else "unknown",
            }
            boxes.append(box_dict)
        predictions_by_token[token] = boxes

    return predictions_by_token


def load_ground_truth_from_info(info: Dict[str, Any]) -> List[Dict]:
    """Extract ground truth boxes from a sample info dict."""
    return info.get("annotations", [])


def generate_video(
    data_root: str,
    info_data: Dict[str, Any],
    predictions_by_token: Dict[str, List[Dict]],
    output_path: str,
    score_threshold: float = 0.3,
    fps: int = 2,
):
    """
    Generate a video for a sequence of samples.

    Groups samples by scene and creates one video per scene.
    """
    infos = info_data["infos"]

    # Group infos by scene
    scene_to_infos = defaultdict(list)
    for info in infos:
        scene_to_infos[info["scene_token"]].append(info)

    # Sort each scene by timestamp
    for scene_token in scene_to_infos:
        scene_to_infos[scene_token].sort(key=lambda x: x["timestamp"])

    video_count = 0
    for scene_token, scene_infos in scene_to_infos.items():
        if len(scene_infos) < 2:
            continue

        # Check if we have predictions for this scene
        has_predictions = any(
            info["token"] in predictions_by_token for info in scene_infos
        )
        if not has_predictions:
            continue

        video_count += 1
        scene_output = output_path.replace(".mp4", f"_scene_{scene_token[:8]}.mp4")

        print(f"  Generating video for scene {scene_token[:8]} "
              f"({len(scene_infos)} frames)...")

        writer = None

        for frame_idx, info in enumerate(scene_infos):
            token = info["token"]
            preds = predictions_by_token.get(token, [])
            gts = load_ground_truth_from_info(info)

            # Create comparison frame
            frame = create_comparison_figure(
                data_root, info, preds, gts, score_threshold
            )

            # Add frame number
            cv2.putText(frame, f"Frame {frame_idx + 1}/{len(scene_infos)}",
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(scene_output, fourcc, fps, (w, h))

            writer.write(frame)

        if writer is not None:
            writer.release()
            print(f"    Saved: {scene_output}")

    if video_count == 0:
        print("  [WARNING] No scenes with predictions found for video generation.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BEVFormer predictions and ground truth."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to nuScenes dataset root.",
    )
    parser.add_argument(
        "--info_file",
        type=str,
        required=True,
        help="Path to info pickle file (e.g., nuscenes_infos_temporal_val.pkl).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vis_output",
        help="Output directory for visualizations.",
    )
    parser.add_argument(
        "--sample_token",
        type=str,
        default=None,
        help="Visualize a single sample by token.",
    )
    parser.add_argument(
        "--make_video",
        action="store_true",
        help="Generate video for scene sequences.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Minimum score threshold for predictions (default: 0.3).",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  BEVFormer Visualization")
    print("=" * 60)
    print(f"  Predictions:    {args.predictions}")
    print(f"  Data root:      {args.data_root}")
    print(f"  Info file:      {args.info_file}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Score thresh:   {args.score_threshold}")
    print(f"  Make video:     {args.make_video}")
    if args.sample_token:
        print(f"  Sample token:   {args.sample_token}")
    print("=" * 60)

    # Load predictions
    print("\n[1/3] Loading predictions...")
    predictions_by_token = load_predictions(args.predictions)
    print(f"  Loaded predictions for {len(predictions_by_token)} samples")

    total_preds = sum(len(v) for v in predictions_by_token.values())
    print(f"  Total prediction boxes: {total_preds}")

    # Load info file
    print("\n[2/3] Loading info file...")
    with open(args.info_file, "rb") as f:
        info_data = pickle.load(f)

    infos = info_data["infos"]
    metadata = info_data.get("metadata", {})
    print(f"  Loaded {len(infos)} sample infos")
    print(f"  Version: {metadata.get('version', 'unknown')}")
    print(f"  Split: {metadata.get('split', 'unknown')}")

    # Build token-to-info mapping
    info_by_token = {info["token"]: info for info in infos}

    # Visualize
    print("\n[3/3] Generating visualizations...")

    if args.sample_token:
        # Single sample visualization
        if args.sample_token not in info_by_token:
            print(f"  [ERROR] Sample token not found: {args.sample_token}")
            print(f"  Available tokens (first 5): {list(info_by_token.keys())[:5]}")
            return

        info = info_by_token[args.sample_token]
        preds = predictions_by_token.get(args.sample_token, [])
        gts = load_ground_truth_from_info(info)

        print(f"  Visualizing sample: {args.sample_token}")
        print(f"  Predictions: {len(preds)}, GT: {len(gts)}")

        # Full comparison
        output_path = os.path.join(args.output_dir, f"vis_{args.sample_token[:8]}.png")
        create_comparison_figure(
            args.data_root, info, preds, gts, args.score_threshold, output_path
        )
        print(f"  Saved: {output_path}")

        # BEV only
        bev_path = os.path.join(args.output_dir, f"bev_{args.sample_token[:8]}.png")
        bev_img = create_bev_image(preds, gts, args.score_threshold)
        cv2.imwrite(bev_path, bev_img)
        print(f"  Saved: {bev_path}")

    elif args.make_video:
        # Video generation
        video_path = os.path.join(args.output_dir, "bevformer_results.mp4")
        generate_video(
            args.data_root, info_data, predictions_by_token,
            video_path, args.score_threshold,
        )

    else:
        # Visualize all samples that have predictions
        num_visualized = 0
        max_samples = 50  # Limit to avoid generating too many images

        for token, preds in predictions_by_token.items():
            if token not in info_by_token:
                continue

            info = info_by_token[token]
            gts = load_ground_truth_from_info(info)

            output_path = os.path.join(args.output_dir, f"vis_{token[:8]}.png")
            create_comparison_figure(
                args.data_root, info, preds, gts, args.score_threshold, output_path
            )

            num_visualized += 1
            if num_visualized % 10 == 0:
                print(f"  Visualized {num_visualized} samples...")

            if num_visualized >= max_samples:
                print(f"  Reached max samples limit ({max_samples}). "
                      f"Use --sample_token for specific samples.")
                break

        print(f"  Total visualizations generated: {num_visualized}")

    print(f"\n  Output saved to: {args.output_dir}")
    print("  Done!")


if __name__ == "__main__":
    main()
