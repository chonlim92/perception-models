"""DETR3D Inference and Visualization Script.

Runs 3D object detection on nuScenes multi-camera samples using a trained
DETR3D model. Supports single-sample and batch-mode demos with projected
3D bounding box visualization across all six camera views.

Usage:
    python inference.py \
        --checkpoint /path/to/detr3d.pth \
        --data_root /path/to/nuscenes \
        --output_dir ./results \
        --score_threshold 0.3

    # Single sample:
    python inference.py \
        --checkpoint /path/to/detr3d.pth \
        --data_root /path/to/nuscenes \
        --sample_token <token> \
        --output_dir ./results
"""

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from model import DETR3D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Colors for each class (BGR format for OpenCV compatibility, but stored as RGB)
CLASS_COLORS = [
    (255, 158, 0),    # car - orange
    (255, 99, 71),    # truck - tomato
    (233, 150, 70),   # construction_vehicle - sandy brown
    (255, 69, 0),     # bus - red-orange
    (255, 140, 0),    # trailer - dark orange
    (0, 207, 191),    # barrier - teal
    (255, 61, 99),    # motorcycle - crimson
    (220, 20, 60),    # bicycle - crimson red
    (0, 0, 230),      # pedestrian - blue
    (47, 79, 79),     # traffic_cone - dark slate gray
]


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str, device: str = "cuda") -> DETR3D:
    """Create DETR3D model, load trained weights, and set to eval mode.

    Args:
        checkpoint_path: Path to the model checkpoint file (.pth or .pt).
        device: Device to place the model on ('cuda' or 'cpu').

    Returns:
        DETR3D model ready for inference.
    """
    model = DETR3D(
        num_classes=len(CLASS_NAMES),
        embed_dims=256,
        num_heads=8,
        ffn_dims=1024,
        num_layers=6,
        num_queries=900,
        dropout=0.0,
        pc_range=PC_RANGE,
        code_size=10,
        pretrained_backbone=False,
        fpn_out_channels=256,
        frozen_backbone_stages=-1,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove module. prefix if saved from DataParallel/DistributedDataParallel
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"[INFO] Model loaded from: {checkpoint_path}")
    print(f"[INFO] Model device: {device}")
    print(f"[INFO] Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def preprocess_image(
    image: Image.Image, target_size: Tuple[int, int] = (256, 704)
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize and normalize an image for model input.

    Args:
        image: PIL Image in RGB format.
        target_size: Target (height, width) for resizing.

    Returns:
        Tuple of (normalized_image as CHW float32 array, original_image as HWC uint8 array).
    """
    original_np = np.array(image)

    # Resize to target size
    image_resized = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    image_np = np.array(image_resized, dtype=np.float32) / 255.0

    # Normalize with ImageNet mean/std
    image_np = (image_np - IMAGE_MEAN) / IMAGE_STD

    # Convert HWC to CHW
    image_np = image_np.transpose(2, 0, 1)

    return image_np, original_np


def load_sample(
    nusc,
    sample_token: str,
    image_size: Tuple[int, int] = (256, 704),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[np.ndarray]]:
    """Load 6 camera images and calibration data for one nuScenes sample.

    Args:
        nusc: NuScenes dataset instance.
        sample_token: Token identifying the sample.
        image_size: Target (height, width) for model input images.

    Returns:
        Tuple of:
            - images: Tensor of shape (1, 6, 3, H, W) normalized for model.
            - intrinsics: Tensor of shape (1, 6, 3, 3) camera intrinsic matrices.
            - extrinsics: Tensor of shape (1, 6, 4, 4) camera-to-world extrinsic matrices.
            - images_original: List of 6 original images as numpy arrays (HWC, uint8).
    """
    from pyquaternion import Quaternion

    sample = nusc.get("sample", sample_token)

    images_processed = []
    images_original = []
    intrinsic_matrices = []
    extrinsic_matrices = []

    for cam_name in CAMERA_NAMES:
        # Get sample data for this camera
        cam_data = nusc.get("sample_data", sample["data"][cam_name])

        # Load image
        image_path = os.path.join(nusc.dataroot, cam_data["filename"])
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        image_processed, image_orig = preprocess_image(image, image_size)
        images_processed.append(image_processed)
        images_original.append(image_orig)

        # Get calibration: sensor -> ego -> global
        calibrated_sensor = nusc.get(
            "calibrated_sensor", cam_data["calibrated_sensor_token"]
        )
        ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])

        # Intrinsic matrix (3x3)
        intrinsic = np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float32)

        # Scale intrinsic to match resized image
        orig_h, orig_w = image_orig.shape[:2]
        scale_x = image_size[1] / orig_w
        scale_y = image_size[0] / orig_h
        intrinsic[0, :] *= scale_x
        intrinsic[1, :] *= scale_y
        intrinsic_matrices.append(intrinsic)

        # Extrinsic: camera-to-global transform (4x4)
        # sensor -> ego
        sensor_to_ego = np.eye(4, dtype=np.float32)
        sensor_to_ego[:3, :3] = Quaternion(
            calibrated_sensor["rotation"]
        ).rotation_matrix.astype(np.float32)
        sensor_to_ego[:3, 3] = np.array(
            calibrated_sensor["translation"], dtype=np.float32
        )

        # ego -> global
        ego_to_global = np.eye(4, dtype=np.float32)
        ego_to_global[:3, :3] = Quaternion(
            ego_pose["rotation"]
        ).rotation_matrix.astype(np.float32)
        ego_to_global[:3, 3] = np.array(ego_pose["translation"], dtype=np.float32)

        # camera-to-global = ego_to_global @ sensor_to_ego
        camera_to_global = ego_to_global @ sensor_to_ego
        extrinsic_matrices.append(camera_to_global)

    # Stack into tensors with batch dimension
    images_tensor = torch.from_numpy(
        np.stack(images_processed, axis=0)
    ).unsqueeze(0)  # (1, 6, 3, H, W)

    intrinsics_tensor = torch.from_numpy(
        np.stack(intrinsic_matrices, axis=0)
    ).unsqueeze(0)  # (1, 6, 3, 3)

    extrinsics_tensor = torch.from_numpy(
        np.stack(extrinsic_matrices, axis=0)
    ).unsqueeze(0)  # (1, 6, 4, 4)

    return images_tensor, intrinsics_tensor, extrinsics_tensor, images_original


# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------


def postprocess_predictions(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    score_threshold: float = 0.3,
    pc_range: Optional[List[float]] = None,
    max_detections: int = 300,
) -> Dict[str, np.ndarray]:
    """Post-process raw model predictions into detection results.

    Applies sigmoid to logits, filters by score threshold, and decodes
    bounding boxes to absolute coordinates.

    Args:
        pred_logits: Raw classification logits of shape (B, 900, num_classes).
            Only first batch element is processed.
        pred_boxes: Raw box predictions of shape (B, 900, 10).
            Encoded as (cx, cy, cz, w, l, h, sin, cos, vx, vy).
        score_threshold: Minimum confidence score to keep a detection.
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        max_detections: Maximum number of detections to return.

    Returns:
        Dictionary with:
            - 'scores': (N,) confidence scores.
            - 'labels': (N,) predicted class indices.
            - 'boxes_3d': (N, 10) decoded box parameters in absolute coordinates
              (cx_m, cy_m, cz_m, w_m, l_m, h_m, sin_yaw, cos_yaw, vx_m/s, vy_m/s).
    """
    if pc_range is None:
        pc_range = PC_RANGE

    # Take first batch element
    logits = pred_logits[0]  # (900, num_classes)
    boxes = pred_boxes[0]    # (900, 10)

    # Apply sigmoid to get class probabilities
    scores_all = logits.sigmoid()  # (900, num_classes)

    # Get max score per query across all classes
    max_scores, max_labels = scores_all.max(dim=-1)  # (900,), (900,)

    # Filter by threshold
    valid_mask = max_scores > score_threshold
    valid_scores = max_scores[valid_mask]
    valid_labels = max_labels[valid_mask]
    valid_boxes = boxes[valid_mask]

    # Sort by score and take top max_detections
    num_valid = valid_scores.shape[0]
    if num_valid == 0:
        return {
            "scores": np.array([], dtype=np.float32),
            "labels": np.array([], dtype=np.int64),
            "boxes_3d": np.zeros((0, 10), dtype=np.float32),
        }

    if num_valid > max_detections:
        topk_scores, topk_indices = valid_scores.topk(max_detections, sorted=True)
        topk_labels = valid_labels[topk_indices]
        topk_boxes = valid_boxes[topk_indices]
    else:
        sorted_indices = valid_scores.argsort(descending=True)
        topk_scores = valid_scores[sorted_indices]
        topk_labels = valid_labels[sorted_indices]
        topk_boxes = valid_boxes[sorted_indices]

    # Decode box center positions from normalized [0, 1] to absolute meters
    # The model predicts cx, cy, cz as sigmoid-normalized values within pc_range
    decoded_boxes = topk_boxes.clone()
    range_min = torch.tensor(pc_range[:3], device=decoded_boxes.device, dtype=decoded_boxes.dtype)
    range_max = torch.tensor(pc_range[3:], device=decoded_boxes.device, dtype=decoded_boxes.dtype)

    # Denormalize center position (cx, cy, cz)
    decoded_boxes[:, 0] = decoded_boxes[:, 0] * (range_max[0] - range_min[0]) + range_min[0]
    decoded_boxes[:, 1] = decoded_boxes[:, 1] * (range_max[1] - range_min[1]) + range_min[1]
    decoded_boxes[:, 2] = decoded_boxes[:, 2] * (range_max[2] - range_min[2]) + range_min[2]

    # w, l, h (indices 3, 4, 5) are predicted as exponential of learned offsets
    # For inference we take them directly as absolute values (meters)
    # sin, cos (indices 6, 7) are direct predictions of yaw angle components
    # vx, vy (indices 8, 9) are direct velocity predictions in m/s

    # Convert to numpy
    scores_np = topk_scores.detach().cpu().numpy().astype(np.float32)
    labels_np = topk_labels.detach().cpu().numpy().astype(np.int64)
    boxes_np = decoded_boxes.detach().cpu().numpy().astype(np.float32)

    return {
        "scores": scores_np,
        "labels": labels_np,
        "boxes_3d": boxes_np,
    }


# ---------------------------------------------------------------------------
# 3D Box Projection
# ---------------------------------------------------------------------------


def compute_3d_box_corners(box_3d: np.ndarray) -> np.ndarray:
    """Compute the 8 corners of a 3D bounding box in world coordinates.

    Args:
        box_3d: Array of shape (10,) with (cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy).
            Convention: x-forward, y-left, z-up for nuScenes global frame.
            w = width (y-axis), l = length (x-axis), h = height (z-axis).

    Returns:
        corners: Array of shape (8, 3) containing the 3D corner coordinates.
            Corner ordering: bottom face (4 corners) then top face (4 corners).
    """
    cx, cy, cz, w, l, h, sin_yaw, cos_yaw = box_3d[:8]

    # Half dimensions
    half_l = l / 2.0
    half_w = w / 2.0
    half_h = h / 2.0

    # 8 corners in box-local frame (before rotation)
    # Order: front-right-bottom, front-left-bottom, back-left-bottom, back-right-bottom,
    #        front-right-top, front-left-top, back-left-top, back-right-top
    corners_local = np.array([
        [ half_l,  half_w, -half_h],
        [ half_l, -half_w, -half_h],
        [-half_l, -half_w, -half_h],
        [-half_l,  half_w, -half_h],
        [ half_l,  half_w,  half_h],
        [ half_l, -half_w,  half_h],
        [-half_l, -half_w,  half_h],
        [-half_l,  half_w,  half_h],
    ], dtype=np.float64)

    # Rotation matrix around z-axis (yaw)
    rotation = np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw,  cos_yaw, 0.0],
        [0.0,      0.0,     1.0],
    ], dtype=np.float64)

    # Rotate and translate to world frame
    corners_world = (rotation @ corners_local.T).T + np.array([cx, cy, cz])

    return corners_world.astype(np.float32)


def project_3d_box_to_image(
    box_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, bool]:
    """Project a 3D bounding box onto a camera image.

    Args:
        box_3d: Array of shape (10,) with (cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy).
        intrinsic: Camera intrinsic matrix of shape (3, 3).
        extrinsic: Camera-to-world extrinsic matrix of shape (4, 4).
            This is the camera-to-global transform.
        image_shape: Tuple (height, width) of the target image.

    Returns:
        Tuple of:
            - corners_2d: Array of shape (8, 2) with projected 2D pixel coordinates.
            - visible: Boolean indicating if the box is at least partially visible
              in the camera view (at least one corner is in front of camera and within image).
    """
    # Get 3D corners in world/global frame
    corners_3d = compute_3d_box_corners(box_3d)  # (8, 3)

    # Transform from global to camera frame
    # extrinsic is camera-to-global, so we invert it to get global-to-camera
    global_to_camera = np.linalg.inv(extrinsic.astype(np.float64))

    # Convert corners to homogeneous coordinates
    ones = np.ones((8, 1), dtype=np.float64)
    corners_homo = np.hstack([corners_3d.astype(np.float64), ones])  # (8, 4)

    # Transform to camera frame
    corners_camera = (global_to_camera @ corners_homo.T).T[:, :3]  # (8, 3)

    # Check if any corner is in front of camera (z > 0)
    depth_mask = corners_camera[:, 2] > 0.1
    if not depth_mask.any():
        return np.zeros((8, 2), dtype=np.float32), False

    # Project to image plane using intrinsic matrix
    # Project all corners (even those behind camera for line continuity)
    corners_2d = np.zeros((8, 2), dtype=np.float32)
    for i in range(8):
        if corners_camera[i, 2] > 0.1:
            point_2d = intrinsic.astype(np.float64) @ corners_camera[i]
            corners_2d[i, 0] = point_2d[0] / point_2d[2]
            corners_2d[i, 1] = point_2d[1] / point_2d[2]
        else:
            # For points behind camera, project with a small positive depth
            # to maintain edge connectivity
            corners_2d[i, 0] = float("nan")
            corners_2d[i, 1] = float("nan")

    # Check if at least one projected corner is within image bounds
    img_h, img_w = image_shape
    valid_corners = depth_mask & (
        (corners_2d[:, 0] >= -img_w * 0.5) &
        (corners_2d[:, 0] <= img_w * 1.5) &
        (corners_2d[:, 1] >= -img_h * 0.5) &
        (corners_2d[:, 1] <= img_h * 1.5)
    )

    visible = valid_corners.any()

    return corners_2d, visible


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def draw_projected_box_3d(
    image: np.ndarray,
    corners_2d: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a projected 3D bounding box on an image.

    Draws the 12 edges of the box (4 bottom, 4 top, 4 vertical pillars).

    Args:
        image: Image array of shape (H, W, 3) in uint8 RGB format.
        corners_2d: Array of shape (8, 2) with projected 2D corner coordinates.
        color: RGB color tuple for the box edges.
        thickness: Line thickness in pixels.

    Returns:
        Image with the 3D box drawn on it.
    """
    import cv2

    image = image.copy()

    # Convert RGB to BGR for OpenCV
    color_bgr = (color[2], color[1], color[0])

    # Define the 12 edges of the box
    # Bottom face: 0-1, 1-2, 2-3, 3-0
    # Top face: 4-5, 5-6, 6-7, 7-4
    # Vertical pillars: 0-4, 1-5, 2-6, 3-7
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # pillars
    ]

    for start_idx, end_idx in edges:
        pt1 = corners_2d[start_idx]
        pt2 = corners_2d[end_idx]

        # Skip if either point has NaN coordinates (behind camera)
        if np.isnan(pt1).any() or np.isnan(pt2).any():
            continue

        pt1_int = (int(round(pt1[0])), int(round(pt1[1])))
        pt2_int = (int(round(pt2[0])), int(round(pt2[1])))

        cv2.line(image, pt1_int, pt2_int, color_bgr, thickness)

    # Draw front face with slightly thicker lines to indicate orientation
    front_edges = [(0, 1), (4, 5), (0, 4), (1, 5)]
    for start_idx, end_idx in front_edges:
        pt1 = corners_2d[start_idx]
        pt2 = corners_2d[end_idx]
        if np.isnan(pt1).any() or np.isnan(pt2).any():
            continue
        pt1_int = (int(round(pt1[0])), int(round(pt1[1])))
        pt2_int = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(image, pt1_int, pt2_int, color_bgr, thickness + 1)

    return image


def visualize_predictions(
    images_original: List[np.ndarray],
    predictions: Dict[str, np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    class_names: List[str],
    output_path: str,
) -> None:
    """Visualize 3D detection results on all 6 camera views.

    Projects 3D bounding boxes onto each camera image, draws them with
    class-specific colors, and arranges the 6 views in a 2x3 grid.

    Args:
        images_original: List of 6 original images as numpy arrays (H, W, 3), uint8.
        predictions: Dict with 'scores' (N,), 'labels' (N,), 'boxes_3d' (N, 10).
        intrinsics: Camera intrinsics of shape (6, 3, 3) for the resized images.
        extrinsics: Camera extrinsics of shape (6, 4, 4) camera-to-world.
        class_names: List of class name strings.
        output_path: File path to save the visualization image.
    """
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_cams = len(images_original)
    scores = predictions["scores"]
    labels = predictions["labels"]
    boxes_3d = predictions["boxes_3d"]
    num_detections = len(scores)

    # We need to rescale intrinsics to match original image dimensions for visualization
    # But we project onto the original images directly, so compute adjusted intrinsics
    annotated_images = []

    for cam_idx in range(num_cams):
        img = images_original[cam_idx].copy()
        img_h, img_w = img.shape[:2]

        # The provided intrinsics are for the resized (256, 704) input
        # We need to scale them back to the original image size
        cam_intrinsic = intrinsics[cam_idx].copy()
        scale_x = img_w / 704.0
        scale_y = img_h / 256.0
        cam_intrinsic[0, :] *= scale_x
        cam_intrinsic[1, :] *= scale_y

        cam_extrinsic = extrinsics[cam_idx]

        for det_idx in range(num_detections):
            box = boxes_3d[det_idx]
            label = labels[det_idx]
            score = scores[det_idx]
            color = CLASS_COLORS[label % len(CLASS_COLORS)]

            corners_2d, visible = project_3d_box_to_image(
                box, cam_intrinsic, cam_extrinsic, (img_h, img_w)
            )

            if not visible:
                continue

            # Draw the 3D box
            img = draw_projected_box_3d(img, corners_2d, color, thickness=2)

            # Add text label at the top-left visible corner
            valid_corners = corners_2d[~np.isnan(corners_2d[:, 0])]
            if len(valid_corners) > 0:
                # Find top-left-most visible corner for text placement
                text_x = int(np.clip(valid_corners[:, 0].min(), 0, img_w - 1))
                text_y = int(np.clip(valid_corners[:, 1].min() - 5, 10, img_h - 1))

                label_text = f"{class_names[label]} {score:.2f}"
                cv2.putText(
                    img,
                    label_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (color[2], color[1], color[0]),  # BGR
                    1,
                    cv2.LINE_AA,
                )

        # Add camera name title
        cv2.putText(
            img,
            CAMERA_NAMES[cam_idx],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        annotated_images.append(img)

    # Arrange in 2x3 grid using matplotlib
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    fig.suptitle(
        f"DETR3D Detections ({num_detections} objects)",
        fontsize=16,
        fontweight="bold",
    )

    # Grid layout:
    # Row 0: FRONT_LEFT, FRONT, FRONT_RIGHT
    # Row 1: BACK_LEFT, BACK, BACK_RIGHT
    grid_order = [1, 0, 2, 4, 3, 5]  # maps camera index to grid position

    grid_positions = [
        (0, 1),  # CAM_FRONT -> top center
        (0, 0),  # CAM_FRONT_LEFT -> top left
        (0, 2),  # CAM_FRONT_RIGHT -> top right
        (1, 1),  # CAM_BACK -> bottom center
        (1, 0),  # CAM_BACK_LEFT -> bottom left
        (1, 2),  # CAM_BACK_RIGHT -> bottom right
    ]

    for cam_idx, (row, col) in enumerate(grid_positions):
        ax = axes[row, col]
        # Convert BGR back to RGB for matplotlib if needed (images are already RGB)
        ax.imshow(annotated_images[cam_idx])
        ax.set_title(CAMERA_NAMES[cam_idx], fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"[INFO] Visualization saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main Inference Pipeline
# ---------------------------------------------------------------------------


def run_inference_single(
    model: DETR3D,
    nusc,
    sample_token: str,
    score_threshold: float,
    output_dir: str,
    device: str,
    image_size: Tuple[int, int] = (256, 704),
) -> Dict[str, np.ndarray]:
    """Run inference on a single nuScenes sample.

    Args:
        model: Loaded DETR3D model in eval mode.
        nusc: NuScenes dataset instance.
        sample_token: Token identifying the sample.
        score_threshold: Minimum confidence score.
        output_dir: Directory to save visualization.
        device: Device string.
        image_size: Model input image size (H, W).

    Returns:
        Predictions dict with 'scores', 'labels', 'boxes_3d'.
    """
    # Load sample data
    images, intrinsics, extrinsics, images_original = load_sample(
        nusc, sample_token, image_size
    )

    # Move to device
    images = images.to(device)
    intrinsics_dev = intrinsics.to(device)
    extrinsics_dev = extrinsics.to(device)

    # Run model inference
    with torch.no_grad():
        outputs = model(images, intrinsics_dev, extrinsics_dev, image_size)

    # Post-process
    predictions = postprocess_predictions(
        outputs["pred_logits"],
        outputs["pred_boxes"],
        score_threshold=score_threshold,
        pc_range=PC_RANGE,
    )

    # Visualize
    output_path = os.path.join(output_dir, f"det_{sample_token[:8]}.png")
    visualize_predictions(
        images_original,
        predictions,
        intrinsics[0].numpy(),   # (6, 3, 3)
        extrinsics[0].numpy(),   # (6, 4, 4)
        CLASS_NAMES,
        output_path,
    )

    return predictions


def print_detection_summary(
    predictions: Dict[str, np.ndarray],
    class_names: List[str],
) -> None:
    """Print a summary of detection results.

    Args:
        predictions: Dict with 'scores', 'labels', 'boxes_3d'.
        class_names: List of class name strings.
    """
    scores = predictions["scores"]
    labels = predictions["labels"]
    boxes_3d = predictions["boxes_3d"]

    num_dets = len(scores)
    print(f"\n{'='*60}")
    print(f"Detection Summary: {num_dets} objects detected")
    print(f"{'='*60}")

    if num_dets == 0:
        print("  No detections above threshold.")
        return

    # Count per class
    class_counts = {}
    for label in labels:
        name = class_names[label]
        class_counts[name] = class_counts.get(name, 0) + 1

    print(f"\n  {'Class':<25} {'Count':<8} {'Avg Score':<10}")
    print(f"  {'-'*43}")

    for cls_name in class_names:
        count = class_counts.get(cls_name, 0)
        if count > 0:
            cls_mask = labels == class_names.index(cls_name)
            avg_score = scores[cls_mask].mean()
            print(f"  {cls_name:<25} {count:<8} {avg_score:.3f}")

    print(f"\n  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Mean score: {scores.mean():.3f}")

    # Print top-5 detections with box details
    print(f"\n  Top-5 Detections:")
    print(f"  {'#':<4} {'Class':<20} {'Score':<8} {'Position (x,y,z)':<25} {'Size (w,l,h)':<20}")
    print(f"  {'-'*77}")
    for i in range(min(5, num_dets)):
        box = boxes_3d[i]
        cls_name = class_names[labels[i]]
        pos_str = f"({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f})"
        size_str = f"({box[3]:.1f}, {box[4]:.1f}, {box[5]:.1f})"
        print(f"  {i+1:<4} {cls_name:<20} {scores[i]:<8.3f} {pos_str:<25} {size_str:<20}")

    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DETR3D 3D Object Detection Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the nuScenes dataset root directory.",
    )
    parser.add_argument(
        "--sample_token",
        type=str,
        default=None,
        help="Specific sample token to process. If not provided, random samples are used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results",
        help="Directory to save visualization outputs.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Minimum confidence score for detections.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random samples to process in batch demo mode.",
    )

    return parser.parse_args()


def main():
    """Main entry point for DETR3D inference demo."""
    args = parse_args()

    print("=" * 60)
    print("  DETR3D 3D Object Detection - Inference Demo")
    print("=" * 60)
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Data root:        {args.data_root}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Score threshold:  {args.score_threshold}")
    print(f"  Device:           {args.device}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Load nuScenes dataset
    print("\n[INFO] Loading nuScenes dataset...")
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot=args.data_root,
        verbose=True,
    )

    # Determine which samples to process
    if args.sample_token is not None:
        # Single sample mode
        sample_tokens = [args.sample_token]
        print(f"\n[INFO] Processing single sample: {args.sample_token}")
    else:
        # Batch demo mode: select random samples from val split
        from nuscenes.utils.splits import create_splits_scenes

        split_scenes = create_splits_scenes()
        val_scene_names = set(split_scenes["val"])

        # Get sample tokens from validation scenes
        val_sample_tokens = []
        for scene in nusc.scene:
            if scene["name"] in val_scene_names:
                sample_token = scene["first_sample_token"]
                while sample_token:
                    val_sample_tokens.append(sample_token)
                    sample = nusc.get("sample", sample_token)
                    sample_token = sample["next"] if sample["next"] != "" else None

        if len(val_sample_tokens) == 0:
            print("[WARNING] No validation samples found. Using all samples.")
            val_sample_tokens = [s["token"] for s in nusc.sample]

        # Randomly select samples
        num_to_process = min(args.num_samples, len(val_sample_tokens))
        sample_tokens = random.sample(val_sample_tokens, num_to_process)
        print(f"\n[INFO] Processing {num_to_process} random validation samples.")

    # Run inference on each sample
    all_predictions = []
    total_detections = 0

    for idx, token in enumerate(sample_tokens):
        print(f"\n[{idx + 1}/{len(sample_tokens)}] Processing sample: {token}")

        try:
            predictions = run_inference_single(
                model=model,
                nusc=nusc,
                sample_token=token,
                score_threshold=args.score_threshold,
                output_dir=args.output_dir,
                device=args.device,
            )

            print_detection_summary(predictions, CLASS_NAMES)
            all_predictions.append(predictions)
            total_detections += len(predictions["scores"])

        except Exception as e:
            print(f"[ERROR] Failed to process sample {token}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print("\n" + "=" * 60)
    print("  INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Samples processed:   {len(all_predictions)}/{len(sample_tokens)}")
    print(f"  Total detections:    {total_detections}")
    if all_predictions:
        avg_dets = total_detections / len(all_predictions)
        print(f"  Avg dets/sample:     {avg_dets:.1f}")

        # Aggregate class statistics
        all_labels = np.concatenate([p["labels"] for p in all_predictions])
        all_scores = np.concatenate([p["scores"] for p in all_predictions])
        print(f"  Overall score range: [{all_scores.min():.3f}, {all_scores.max():.3f}]")
        print(f"\n  Class distribution across all samples:")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            count = (all_labels == cls_idx).sum()
            if count > 0:
                avg_sc = all_scores[all_labels == cls_idx].mean()
                print(f"    {cls_name:<25} {count:<6} (avg score: {avg_sc:.3f})")

    print(f"\n  Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
