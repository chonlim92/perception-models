#!/usr/bin/env python3
"""
visualize_results.py - Visualize CRAFT model predictions and ground truth.

This script loads prediction results and ground truth annotations, projects
3D bounding boxes onto camera images, creates BEV (bird's eye view) plots,
and generates multi-panel visualizations and videos.

Usage:
    # Visualize single sample
    python visualize_results.py \
        --predictions results/predictions.json \
        --infos data/craft_infos_val.pkl \
        --dataroot /data/nuscenes \
        --output-dir vis_output \
        --sample-token <token>

    # Visualize all samples and create video
    python visualize_results.py \
        --predictions results/predictions.json \
        --infos data/craft_infos_val.pkl \
        --dataroot /data/nuscenes \
        --output-dir vis_output \
        --make-video \
        --max-samples 100

    # BEV-only visualization
    python visualize_results.py \
        --predictions results/predictions.json \
        --infos data/craft_infos_val.pkl \
        --dataroot /data/nuscenes \
        --output-dir vis_output \
        --bev-only
"""

import argparse
import json
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
from matplotlib.gridspec import GridSpec
from pyquaternion import Quaternion

# Use non-interactive backend for server environments
matplotlib.use('Agg')


# =============================================================================
# Configuration
# =============================================================================

# Detection class colors (BGR for OpenCV, RGB for matplotlib)
CLASS_COLORS_RGB = {
    'car': (0.0, 0.5, 1.0),           # Blue
    'truck': (0.0, 0.8, 0.4),         # Green
    'construction_vehicle': (0.6, 0.4, 0.0),  # Brown
    'bus': (1.0, 0.6, 0.0),           # Orange
    'trailer': (0.5, 0.0, 0.5),       # Purple
    'barrier': (0.7, 0.7, 0.7),       # Gray
    'motorcycle': (1.0, 0.0, 0.5),    # Magenta
    'bicycle': (0.0, 1.0, 1.0),       # Cyan
    'pedestrian': (1.0, 0.0, 0.0),    # Red
    'traffic_cone': (1.0, 1.0, 0.0),  # Yellow
}

CLASS_COLORS_BGR = {
    k: (int(v[2] * 255), int(v[1] * 255), int(v[0] * 255))
    for k, v in CLASS_COLORS_RGB.items()
}

# GT color: green, Prediction color: red
GT_COLOR_RGB = (0.0, 0.8, 0.0)
GT_COLOR_BGR = (0, 204, 0)
PRED_COLOR_RGB = (1.0, 0.0, 0.0)
PRED_COLOR_BGR = (0, 0, 255)

# BEV plot configuration
BEV_RANGE_X = (-50.0, 50.0)  # meters
BEV_RANGE_Y = (-50.0, 50.0)  # meters
BEV_RESOLUTION = 0.1          # meters per pixel

# Camera ordering for multi-panel display
CAMERA_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]

# Camera layout in the multi-panel figure (row, col)
CAMERA_LAYOUT = {
    'CAM_FRONT_LEFT': (0, 0),
    'CAM_FRONT': (0, 1),
    'CAM_FRONT_RIGHT': (0, 2),
    'CAM_BACK_LEFT': (1, 0),
    'CAM_BACK': (1, 1),
    'CAM_BACK_RIGHT': (1, 2),
}


# =============================================================================
# 3D Bounding Box Utilities
# =============================================================================

def get_3d_box_corners(
    center: np.ndarray,
    size: np.ndarray,
    rotation: Quaternion,
) -> np.ndarray:
    """
    Compute the 8 corners of a 3D bounding box.

    The box is defined by its center, size [width, length, height], and rotation
    quaternion. Returns corners in the global/ego frame.

    Corner ordering:
        4 --- 5
       /|    /|
      7 --- 6 |
      | 0 --| 1
      |/    |/
      3 --- 2

    Args:
        center: (3,) box center [x, y, z].
        size: (3,) box dimensions [width, length, height].
        rotation: Quaternion representing box orientation.

    Returns:
        (8, 3) array of corner coordinates.
    """
    w, l, h = size[0], size[1], size[2]

    # 8 corners in box-local frame (centered at origin)
    # x: width, y: length, z: height
    corners_local = np.array([
        [-w / 2, -l / 2, -h / 2],  # 0: back-left-bottom
        [+w / 2, -l / 2, -h / 2],  # 1: back-right-bottom
        [+w / 2, +l / 2, -h / 2],  # 2: front-right-bottom
        [-w / 2, +l / 2, -h / 2],  # 3: front-left-bottom
        [-w / 2, -l / 2, +h / 2],  # 4: back-left-top
        [+w / 2, -l / 2, +h / 2],  # 5: back-right-top
        [+w / 2, +l / 2, +h / 2],  # 6: front-right-top
        [-w / 2, +l / 2, +h / 2],  # 7: front-left-top
    ])

    # Rotate corners to global frame
    rot_matrix = rotation.rotation_matrix
    corners_global = (rot_matrix @ corners_local.T).T + center

    return corners_global


def project_points_to_image(
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points onto a camera image plane.

    Args:
        points_3d: (N, 3) points in global/ego frame.
        intrinsic: (3, 3) camera intrinsic matrix.
        extrinsic: (4, 4) world-to-camera transformation matrix.
        image_shape: (height, width) of the target image.

    Returns:
        Tuple of:
            - (M, 2) projected pixel coordinates for valid points.
            - (M,) boolean mask indicating which points are valid (in front of
              camera and within image bounds).
    """
    N = points_3d.shape[0]
    if N == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

    # Convert to homogeneous coordinates
    ones = np.ones((N, 1), dtype=np.float64)
    points_h = np.hstack([points_3d, ones])  # (N, 4)

    # Transform to camera frame
    points_cam = (extrinsic @ points_h.T).T  # (N, 4)
    points_cam_3d = points_cam[:, :3]  # (N, 3)

    # Check depth (points must be in front of camera)
    depth = points_cam_3d[:, 2]
    valid_depth = depth > 0.1  # Minimum depth threshold

    # Project to image plane
    points_2d_h = (intrinsic @ points_cam_3d.T).T  # (N, 3)

    # Normalize by depth
    with np.errstate(divide='ignore', invalid='ignore'):
        points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]

    # Check image bounds
    h, w = image_shape
    valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w)
    valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)

    valid_mask = valid_depth & valid_x & valid_y

    return points_2d.astype(np.float32), valid_mask


def draw_3d_box_on_image(
    image: np.ndarray,
    corners_2d: np.ndarray,
    valid_mask: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
    label: str = '',
    score: float = -1.0,
) -> np.ndarray:
    """
    Draw a projected 3D bounding box on an image.

    The box is drawn as a wireframe with 12 edges connecting the 8 corners.

    Args:
        image: Input image (BGR, will be modified in-place).
        corners_2d: (8, 2) projected corner coordinates.
        valid_mask: (8,) boolean mask for valid corners.
        color: BGR color tuple.
        thickness: Line thickness in pixels.
        label: Optional class label to display.
        score: Optional confidence score to display.

    Returns:
        Modified image.
    """
    # Define the 12 edges of the box
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # Draw edges where both endpoints are valid
    for i, j in edges:
        if valid_mask[i] and valid_mask[j]:
            pt1 = tuple(corners_2d[i].astype(int))
            pt2 = tuple(corners_2d[j].astype(int))
            cv2.line(image, pt1, pt2, color, thickness)

    # Draw label if any valid corners exist
    if label and np.any(valid_mask):
        # Find the topmost valid corner for label placement
        valid_corners = corners_2d[valid_mask]
        top_idx = np.argmin(valid_corners[:, 1])
        label_pos = valid_corners[top_idx].astype(int)

        text = label
        if score >= 0:
            text = f"{label} {score:.2f}"

        # Draw text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, text_thickness)

        text_x = max(0, label_pos[0] - text_w // 2)
        text_y = max(text_h + 4, label_pos[1] - 5)

        cv2.rectangle(
            image,
            (text_x - 1, text_y - text_h - 3),
            (text_x + text_w + 1, text_y + 3),
            color,
            -1
        )
        cv2.putText(
            image, text,
            (text_x, text_y),
            font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA
        )

    return image


# =============================================================================
# BEV (Bird's Eye View) Visualization
# =============================================================================

def create_bev_plot(
    gt_boxes: List[Dict],
    pred_boxes: List[Dict],
    radar_points: Optional[np.ndarray] = None,
    ego_position: Optional[np.ndarray] = None,
    x_range: Tuple[float, float] = BEV_RANGE_X,
    y_range: Tuple[float, float] = BEV_RANGE_Y,
    figsize: Tuple[float, float] = (8, 8),
) -> plt.Figure:
    """
    Create a bird's eye view plot showing detections and radar points.

    The plot shows the scene from above, with the ego vehicle at center.
    Ground truth boxes are shown in green, predictions in class-specific colors
    with red outline.

    Args:
        gt_boxes: List of ground truth annotation dicts with 'translation',
                  'size', 'rotation', 'detection_name'.
        pred_boxes: List of prediction dicts with 'translation', 'size',
                    'rotation', 'detection_name', 'score'.
        radar_points: Optional (N, 3+) array of radar points [x, y, z, ...].
        ego_position: Optional (3,) ego position in global frame.
        x_range: (min, max) x-axis range in meters.
        y_range: (min, max) y-axis range in meters.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_title('Bird\'s Eye View', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Draw ego vehicle as a rectangle at origin
    ego_rect = plt.Rectangle(
        (-1.0, -2.0), 2.0, 4.0,
        linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7,
        label='Ego Vehicle'
    )
    ax.add_patch(ego_rect)

    # Draw radar points
    if radar_points is not None and len(radar_points) > 0:
        # Filter to BEV range
        mask = (
            (radar_points[:, 0] >= x_range[0]) & (radar_points[:, 0] <= x_range[1]) &
            (radar_points[:, 1] >= y_range[0]) & (radar_points[:, 1] <= y_range[1])
        )
        visible_pts = radar_points[mask]
        if len(visible_pts) > 0:
            ax.scatter(
                visible_pts[:, 0], visible_pts[:, 1],
                c='gray', s=3, alpha=0.5, zorder=1, label='Radar Points'
            )

            # Draw velocity vectors if available (columns 3 and 4)
            if visible_pts.shape[1] >= 5:
                # Subsample for clarity
                step = max(1, len(visible_pts) // 50)
                for pt in visible_pts[::step]:
                    vx, vy = pt[3], pt[4]
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed > 0.5:  # Only show significant velocities
                        ax.arrow(
                            pt[0], pt[1], vx * 0.5, vy * 0.5,
                            head_width=0.3, head_length=0.15,
                            fc='darkgray', ec='darkgray', alpha=0.6
                        )

    # Draw ground truth boxes
    for gt in gt_boxes:
        _draw_bev_box(
            ax, gt['translation'], gt['size'], gt['rotation'],
            color=GT_COLOR_RGB, linestyle='-', linewidth=2,
            label_text=gt.get('detection_name', ''),
        )

    # Draw prediction boxes
    for pred in pred_boxes:
        det_name = pred.get('detection_name', 'unknown')
        color = CLASS_COLORS_RGB.get(det_name, PRED_COLOR_RGB)
        score = pred.get('score', -1)
        label = f"{det_name}"
        if score >= 0:
            label += f" ({score:.2f})"

        _draw_bev_box(
            ax, pred['translation'], pred['size'], pred['rotation'],
            color=color, linestyle='--', linewidth=1.5,
            label_text=label,
        )

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='blue', label='Ego'),
        mpatches.Patch(facecolor='none', edgecolor=GT_COLOR_RGB, linewidth=2, label='Ground Truth'),
        mpatches.Patch(facecolor='none', edgecolor=PRED_COLOR_RGB, linewidth=1.5,
                       linestyle='--', label='Predictions'),
    ]
    if radar_points is not None and len(radar_points) > 0:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=5, label='Radar Points')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig


def _draw_bev_box(
    ax: plt.Axes,
    translation: List[float],
    size: List[float],
    rotation: List[float],
    color: Tuple[float, float, float],
    linestyle: str = '-',
    linewidth: float = 2,
    label_text: str = '',
) -> None:
    """
    Draw a single 3D bounding box in BEV (top-down projection).

    Args:
        ax: Matplotlib axes.
        translation: [x, y, z] center position.
        size: [w, l, h] box dimensions.
        rotation: [w, x, y, z] quaternion.
        color: RGB color tuple.
        linestyle: Line style for the box edges.
        linewidth: Line width.
        label_text: Optional text label.
    """
    center = np.array(translation[:2])
    w, l = size[0], size[1]
    q = Quaternion(rotation)

    # Get yaw angle from quaternion
    # Rotation around z-axis
    yaw = q.yaw_pitch_roll[0]

    # 4 corners in BEV (local frame)
    corners_local = np.array([
        [-w / 2, -l / 2],
        [+w / 2, -l / 2],
        [+w / 2, +l / 2],
        [-w / 2, +l / 2],
    ])

    # Rotation matrix for yaw
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    # Rotate and translate corners
    corners_global = (R @ corners_local.T).T + center

    # Draw box edges
    for i in range(4):
        j = (i + 1) % 4
        ax.plot(
            [corners_global[i, 0], corners_global[j, 0]],
            [corners_global[i, 1], corners_global[j, 1]],
            color=color, linestyle=linestyle, linewidth=linewidth
        )

    # Draw heading indicator (front edge is thicker)
    front_mid = (corners_global[2] + corners_global[3]) / 2
    ax.plot(
        [corners_global[2, 0], corners_global[3, 0]],
        [corners_global[2, 1], corners_global[3, 1]],
        color=color, linestyle=linestyle, linewidth=linewidth * 1.5
    )

    # Draw label
    if label_text:
        ax.text(
            center[0], center[1] + l / 2 + 1.0,
            label_text, fontsize=6, ha='center', va='bottom',
            color=color, alpha=0.8
        )


# =============================================================================
# Multi-Panel Visualization
# =============================================================================

def create_multipanel_figure(
    images: Dict[str, np.ndarray],
    bev_figure: plt.Figure,
    sample_token: str = '',
    scene_name: str = '',
) -> plt.Figure:
    """
    Create a combined multi-panel figure with 6 camera views and BEV.

    Layout:
        [CAM_FRONT_LEFT] [CAM_FRONT] [CAM_FRONT_RIGHT]
        [CAM_BACK_LEFT]  [CAM_BACK]  [CAM_BACK_RIGHT]
        [           BEV (spanning full width)          ]

    Args:
        images: Dictionary mapping camera channel names to annotated images (BGR).
        bev_figure: BEV matplotlib figure.
        sample_token: Sample token for title.
        scene_name: Scene name for title.

    Returns:
        matplotlib Figure with all panels.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.05, wspace=0.02,
                  height_ratios=[1, 1, 1.2])

    # Title
    title = f"CRAFT Visualization"
    if scene_name:
        title += f" | {scene_name}"
    if sample_token:
        title += f" | {sample_token[:8]}..."
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Draw camera images
    for cam_name, (row, col) in CAMERA_LAYOUT.items():
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(cam_name.replace('CAM_', ''), fontsize=9, pad=2)
        ax.axis('off')

        if cam_name in images:
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(images[cam_name], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')

    # Draw BEV in the bottom row (spanning all 3 columns)
    ax_bev = fig.add_subplot(gs[2, :])

    # Render the BEV figure into an image array
    bev_figure.canvas.draw()
    bev_array = np.frombuffer(bev_figure.canvas.tostring_rgb(), dtype=np.uint8)
    bev_array = bev_array.reshape(bev_figure.canvas.get_width_height()[::-1] + (3,))
    ax_bev.imshow(bev_array)
    ax_bev.axis('off')

    return fig


# =============================================================================
# Main Visualization Pipeline
# =============================================================================

class CRAFTVisualizer:
    """Main visualization class for CRAFT model results."""

    def __init__(
        self,
        predictions_path: str,
        infos_path: str,
        dataroot: str,
        output_dir: str,
        score_threshold: float = 0.3,
    ):
        """
        Initialize the visualizer.

        Args:
            predictions_path: Path to predictions JSON file.
            infos_path: Path to info pickle file (from prepare_data.py).
            dataroot: Path to nuScenes dataset root.
            output_dir: Directory for saving visualizations.
            score_threshold: Minimum confidence score for displaying predictions.
        """
        self.dataroot = dataroot
        self.output_dir = output_dir
        self.score_threshold = score_threshold

        os.makedirs(output_dir, exist_ok=True)

        # Load predictions
        print(f"Loading predictions from: {predictions_path}")
        with open(predictions_path, 'r') as f:
            self.predictions = json.load(f)

        # Handle different prediction formats
        if isinstance(self.predictions, dict):
            if 'results' in self.predictions:
                self.pred_by_sample = self.predictions['results']
            else:
                self.pred_by_sample = self.predictions
        elif isinstance(self.predictions, list):
            # Group predictions by sample_token
            self.pred_by_sample = {}
            for pred in self.predictions:
                token = pred.get('sample_token', '')
                if token not in self.pred_by_sample:
                    self.pred_by_sample[token] = []
                self.pred_by_sample[token].append(pred)
        else:
            raise ValueError(f"Unexpected predictions format: {type(self.predictions)}")

        print(f"  Loaded predictions for {len(self.pred_by_sample)} samples")

        # Load info pickle
        print(f"Loading info from: {infos_path}")
        with open(infos_path, 'rb') as f:
            self.infos = pickle.load(f)
        print(f"  Loaded {len(self.infos)} sample infos")

        # Build sample_token to info lookup
        self.info_by_token = {info['sample_token']: info for info in self.infos}

    def visualize_sample(
        self,
        sample_token: str,
        save_individual: bool = True,
        bev_only: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Visualize a single sample with predictions and ground truth.

        Args:
            sample_token: Token of the sample to visualize.
            save_individual: Whether to save individual frame as PNG.
            bev_only: If True, only generate BEV plot.

        Returns:
            Combined image as numpy array (BGR), or None on failure.
        """
        if sample_token not in self.info_by_token:
            print(f"  Warning: Sample {sample_token} not found in infos")
            return None

        info = self.info_by_token[sample_token]
        scene_name = info.get('scene_name', 'unknown')

        # Get predictions for this sample
        preds = self.pred_by_sample.get(sample_token, [])
        # Filter by score threshold
        preds = [p for p in preds if p.get('score', 0) >= self.score_threshold]

        # Get ground truth annotations
        gt_annotations = info.get('annotations', [])

        # Get ego2global for transforming to ego frame
        ego2global = np.array(info.get('ego2global', np.eye(4).tolist()))
        global2ego = np.linalg.inv(ego2global)

        # Transform GT and predictions to ego frame for BEV
        gt_boxes_ego = self._transform_boxes_to_ego(gt_annotations, global2ego)
        pred_boxes_ego = self._transform_boxes_to_ego(preds, global2ego)

        # Create BEV plot
        bev_fig = create_bev_plot(
            gt_boxes=gt_boxes_ego,
            pred_boxes=pred_boxes_ego,
            radar_points=None,  # Could load radar points here if desired
        )

        if bev_only:
            bev_path = os.path.join(self.output_dir, f"bev_{sample_token[:8]}.png")
            bev_fig.savefig(bev_path, dpi=150, bbox_inches='tight')
            plt.close(bev_fig)
            print(f"  Saved BEV: {bev_path}")
            return None

        # Load and annotate camera images
        annotated_images = {}
        for cam_name in CAMERA_ORDER:
            cam_info = info.get('cameras', {}).get(cam_name)
            if cam_info is None:
                continue

            # Load image
            img_path = os.path.join(self.dataroot, cam_info['data_path'])
            if not os.path.isfile(img_path):
                print(f"  Warning: Image not found: {img_path}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"  Warning: Failed to read image: {img_path}")
                continue

            # Get camera calibration
            intrinsic = np.array(cam_info['intrinsic'])
            sensor2ego = np.array(cam_info['sensor2ego'])
            cam_ego2global = np.array(cam_info['ego2global'])

            # World-to-camera transform
            world2cam = np.linalg.inv(sensor2ego) @ np.linalg.inv(cam_ego2global)

            img_shape = (image.shape[0], image.shape[1])

            # Draw ground truth boxes (green)
            for gt in gt_annotations:
                image = self._draw_box_on_camera(
                    image, gt, intrinsic, world2cam, img_shape,
                    color=GT_COLOR_BGR, thickness=2,
                    label=gt.get('detection_name', ''),
                    score=-1,
                )

            # Draw predictions (class-specific color)
            for pred in preds:
                det_name = pred.get('detection_name', 'unknown')
                color = CLASS_COLORS_BGR.get(det_name, PRED_COLOR_BGR)
                image = self._draw_box_on_camera(
                    image, pred, intrinsic, world2cam, img_shape,
                    color=color, thickness=2,
                    label=det_name,
                    score=pred.get('score', -1),
                )

            annotated_images[cam_name] = image

        # Create multi-panel figure
        fig = create_multipanel_figure(
            annotated_images, bev_fig,
            sample_token=sample_token,
            scene_name=scene_name,
        )

        # Save
        if save_individual:
            out_path = os.path.join(self.output_dir, f"vis_{sample_token[:8]}.png")
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            print(f"  Saved: {out_path}")

        # Convert figure to image array for video
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        plt.close(fig)
        plt.close(bev_fig)

        return img_bgr

    def _draw_box_on_camera(
        self,
        image: np.ndarray,
        box: Dict,
        intrinsic: np.ndarray,
        world2cam: np.ndarray,
        img_shape: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        label: str = '',
        score: float = -1,
    ) -> np.ndarray:
        """Draw a 3D bounding box projected onto a camera image."""
        translation = np.array(box['translation'])
        size = np.array(box['size'])
        rotation = Quaternion(box['rotation'])

        # Get 3D corners in world frame
        corners_3d = get_3d_box_corners(translation, size, rotation)

        # Project to image
        corners_2d, valid_mask = project_points_to_image(
            corners_3d, intrinsic, world2cam, img_shape
        )

        # Only draw if at least 2 corners are visible
        if np.sum(valid_mask) >= 2:
            image = draw_3d_box_on_image(
                image, corners_2d, valid_mask,
                color=color, thickness=thickness,
                label=label, score=score,
            )

        return image

    def _transform_boxes_to_ego(
        self,
        boxes: List[Dict],
        global2ego: np.ndarray,
    ) -> List[Dict]:
        """Transform bounding boxes from global frame to ego frame."""
        ego_boxes = []
        for box in boxes:
            translation = np.array(box['translation'])
            rotation = Quaternion(box['rotation'])

            # Transform center to ego frame
            center_h = np.array([*translation, 1.0])
            center_ego = (global2ego @ center_h)[:3]

            # Transform rotation to ego frame
            ego_rotation_matrix = global2ego[:3, :3]
            box_rot_matrix = rotation.rotation_matrix
            ego_box_rot = ego_rotation_matrix @ box_rot_matrix
            ego_quat = Quaternion(matrix=ego_box_rot)

            ego_box = {
                'translation': center_ego.tolist(),
                'size': box['size'],
                'rotation': [ego_quat.w, ego_quat.x, ego_quat.y, ego_quat.z],
                'detection_name': box.get('detection_name', 'unknown'),
                'score': box.get('score', -1),
            }
            ego_boxes.append(ego_box)

        return ego_boxes

    def visualize_all(
        self,
        max_samples: int = -1,
        bev_only: bool = False,
    ) -> List[np.ndarray]:
        """
        Visualize all samples in the info file.

        Args:
            max_samples: Maximum number of samples to visualize (-1 for all).
            bev_only: If True, only generate BEV plots.

        Returns:
            List of visualization images (BGR numpy arrays).
        """
        frames = []
        samples_to_process = self.infos
        if max_samples > 0:
            samples_to_process = samples_to_process[:max_samples]

        print(f"\nVisualizing {len(samples_to_process)} samples...")

        for i, info in enumerate(samples_to_process):
            if (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(samples_to_process)}...")

            sample_token = info['sample_token']
            frame = self.visualize_sample(
                sample_token,
                save_individual=True,
                bev_only=bev_only,
            )
            if frame is not None:
                frames.append(frame)

        print(f"  Completed {len(frames)} visualizations")
        return frames

    def create_video(
        self,
        frames: List[np.ndarray],
        output_filename: str = "craft_visualization.mp4",
        fps: float = 2.0,
    ) -> str:
        """
        Create a video from sequential visualization frames.

        Args:
            frames: List of frame images (BGR numpy arrays).
            output_filename: Output video filename.
            fps: Frames per second.

        Returns:
            Path to the created video file.
        """
        if not frames:
            print("  No frames to create video from")
            return ""

        video_path = os.path.join(self.output_dir, output_filename)

        # Get frame dimensions (use first frame as reference)
        h, w = frames[0].shape[:2]

        # Ensure all frames have the same dimensions
        target_size = (w, h)

        # Use H.264 codec (mp4v as fallback)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, target_size)

        if not writer.isOpened():
            # Try alternative codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = video_path.replace('.mp4', '.avi')
            writer = cv2.VideoWriter(video_path, fourcc, fps, target_size)

        if not writer.isOpened():
            print(f"  Error: Could not open video writer for {video_path}")
            return ""

        print(f"  Creating video: {video_path}")
        print(f"  Resolution: {w}x{h}, FPS: {fps}, Frames: {len(frames)}")

        for i, frame in enumerate(frames):
            # Resize if needed
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, target_size)
            writer.write(frame)

        writer.release()
        print(f"  Video saved: {video_path}")
        print(f"  Duration: {len(frames) / fps:.1f} seconds")

        return video_path


# =============================================================================
# Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize CRAFT model predictions and ground truth.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a specific sample
  python visualize_results.py \\
      --predictions results/predictions.json \\
      --infos data/craft_infos_val.pkl \\
      --dataroot /data/nuscenes \\
      --output-dir vis_output \\
      --sample-token abc123def456

  # Visualize all samples and create video
  python visualize_results.py \\
      --predictions results/predictions.json \\
      --infos data/craft_infos_val.pkl \\
      --dataroot /data/nuscenes \\
      --output-dir vis_output \\
      --make-video --fps 2 --max-samples 50

  # BEV-only mode (faster, no image loading)
  python visualize_results.py \\
      --predictions results/predictions.json \\
      --infos data/craft_infos_val.pkl \\
      --dataroot /data/nuscenes \\
      --output-dir vis_output \\
      --bev-only
        """
    )

    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--infos',
        type=str,
        required=True,
        help='Path to info pickle file (output of prepare_data.py)'
    )
    parser.add_argument(
        '--dataroot',
        type=str,
        required=True,
        help='Path to nuScenes dataset root directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory for saving visualizations'
    )
    parser.add_argument(
        '--sample-token',
        type=str,
        default=None,
        help='Visualize a specific sample by its token'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=-1,
        help='Maximum number of samples to visualize (-1 for all)'
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.3,
        help='Minimum confidence score for displaying predictions (default: 0.3)'
    )
    parser.add_argument(
        '--make-video',
        action='store_true',
        help='Create video from sequential frames'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=2.0,
        help='Frames per second for video output (default: 2.0)'
    )
    parser.add_argument(
        '--bev-only',
        action='store_true',
        help='Only generate BEV plots (skip camera image projections)'
    )
    parser.add_argument(
        '--video-filename',
        type=str,
        default='craft_visualization.mp4',
        help='Output video filename (default: craft_visualization.mp4)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not os.path.isfile(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)

    if not os.path.isfile(args.infos):
        print(f"Error: Info file not found: {args.infos}")
        sys.exit(1)

    if not os.path.isdir(args.dataroot):
        print(f"Error: Data root not found: {args.dataroot}")
        sys.exit(1)

    # Initialize visualizer
    visualizer = CRAFTVisualizer(
        predictions_path=args.predictions,
        infos_path=args.infos,
        dataroot=args.dataroot,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
    )

    # Run visualization
    if args.sample_token:
        # Visualize single sample
        print(f"\nVisualizing sample: {args.sample_token}")
        frame = visualizer.visualize_sample(
            args.sample_token,
            save_individual=True,
            bev_only=args.bev_only,
        )
        if frame is not None:
            print("  Done!")
        else:
            print("  Warning: No output generated")
    else:
        # Visualize multiple samples
        frames = visualizer.visualize_all(
            max_samples=args.max_samples,
            bev_only=args.bev_only,
        )

        # Create video if requested
        if args.make_video and frames:
            visualizer.create_video(
                frames,
                output_filename=args.video_filename,
                fps=args.fps,
            )

    print("\nVisualization complete!")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
