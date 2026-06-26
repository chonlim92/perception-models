#!/usr/bin/env python3
"""BEVFormer Inference and Visualization Script (TensorFlow).

Runs inference on a single sample (6 camera images + calibration) and produces:
- 3D bounding box projections onto each camera image
- BEV (bird's-eye-view) top-down visualization with detected boxes
- Color-coded classes with confidence scores

Usage:
    python inference.py --config configs/bevformer_base.yaml \
                        --checkpoint ./work_dirs/bevformer_tf/checkpoints/ckpt-24 \
                        --sample_path /data/nuscenes/samples/sample_0001 \
                        --output_dir ./inference_results \
                        --score_threshold 0.3
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import tensorflow as tf
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bevformer.inference")


# =============================================================================
# Constants
# =============================================================================

NUSCENES_CLASSES = [
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

# Color palette for each class (BGR for OpenCV, RGB for matplotlib)
CLASS_COLORS_BGR = {
    "car": (0, 165, 255),           # Orange
    "truck": (0, 0, 255),           # Red
    "construction_vehicle": (0, 128, 128),  # Olive
    "bus": (255, 0, 0),             # Blue
    "trailer": (128, 0, 128),       # Purple
    "barrier": (128, 128, 128),     # Gray
    "motorcycle": (0, 255, 255),    # Yellow
    "bicycle": (0, 255, 0),         # Green
    "pedestrian": (255, 0, 255),    # Magenta
    "traffic_cone": (0, 128, 255),  # Dark Orange
}

CLASS_COLORS_RGB = {
    "car": (255, 165, 0),
    "truck": (255, 0, 0),
    "construction_vehicle": (128, 128, 0),
    "bus": (0, 0, 255),
    "trailer": (128, 0, 128),
    "barrier": (128, 128, 128),
    "motorcycle": (255, 255, 0),
    "bicycle": (0, 255, 0),
    "pedestrian": (255, 0, 255),
    "traffic_cone": (255, 128, 0),
}

# Normalized colors for matplotlib (0-1 range)
CLASS_COLORS_NORM = {
    cls: tuple(c / 255.0 for c in rgb)
    for cls, rgb in CLASS_COLORS_RGB.items()
}

# Camera names in nuScenes order
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_INFERENCE_CONFIG = {
    "model": {
        "bev_h": 200,
        "bev_w": 200,
        "num_classes": 10,
        "num_query": 900,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "embed_dims": 256,
        "num_heads": 8,
        "num_levels": 4,
    },
    "data": {
        "img_h": 900,
        "img_w": 1600,
        "input_h": 480,
        "input_w": 800,
        "num_cameras": 6,
    },
    "inference": {
        "score_threshold": 0.1,
        "max_per_frame": 300,
        "nms_iou_threshold": 0.5,
        "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    },
}


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load inference configuration from YAML, merged with defaults."""
    config = DEFAULT_INFERENCE_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            config = _deep_merge(config, user_config)
        logger.info(f"Loaded config from {config_path}")
    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# 3D Bounding Box Utilities
# =============================================================================


def get_3d_box_corners(box: np.ndarray) -> np.ndarray:
    """Compute 8 corners of a 3D bounding box from its parameterization.

    Args:
        box: (9,) array with cx, cy, cz, w, l, h, yaw, vx, vy.

    Returns:
        corners: (8, 3) array of 3D corner coordinates.
    """
    cx, cy, cz, w, l, h, yaw = box[0], box[1], box[2], box[3], box[4], box[5], box[6]

    # Half dimensions
    dx = w / 2.0
    dy = l / 2.0
    dz = h / 2.0

    # 8 corners in local frame (centered at box center)
    # Order: bottom-4 then top-4 (counterclockwise from front-left)
    corners_local = np.array([
        [ dx,  dy, -dz],  # front-left-bottom
        [ dx, -dy, -dz],  # front-right-bottom
        [-dx, -dy, -dz],  # rear-right-bottom
        [-dx,  dy, -dz],  # rear-left-bottom
        [ dx,  dy,  dz],  # front-left-top
        [ dx, -dy,  dz],  # front-right-top
        [-dx, -dy,  dz],  # rear-right-top
        [-dx,  dy,  dz],  # rear-left-top
    ], dtype=np.float64)

    # Rotation matrix around z-axis (yaw)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1],
    ], dtype=np.float64)

    # Rotate and translate
    corners_world = (rotation @ corners_local.T).T
    corners_world[:, 0] += cx
    corners_world[:, 1] += cy
    corners_world[:, 2] += cz

    return corners_world.astype(np.float32)


def project_3d_to_2d(
    corners_3d: np.ndarray, lidar2img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points onto a 2D image plane.

    Args:
        corners_3d: (N, 3) array of 3D points.
        lidar2img: (4, 4) projection matrix (lidar to image).

    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates.
        depth: (N,) array of depths (z-coordinate after projection).
    """
    num_points = corners_3d.shape[0]

    # Homogeneous coordinates
    corners_homo = np.concatenate(
        [corners_3d, np.ones((num_points, 1), dtype=np.float32)], axis=1
    )

    # Project: (4, 4) @ (4, N) -> (4, N)
    projected = (lidar2img @ corners_homo.T).T  # (N, 4)

    # Depth (z-coordinate after projection)
    depth = projected[:, 2].copy()

    # Normalize by depth to get pixel coordinates
    # Avoid division by zero
    valid_depth = depth.copy()
    valid_depth[valid_depth < 1e-5] = 1e-5

    points_2d = projected[:, :2] / valid_depth[:, np.newaxis]

    return points_2d.astype(np.float32), depth.astype(np.float32)


def is_box_visible(
    corners_2d: np.ndarray, depth: np.ndarray, img_h: int, img_w: int
) -> bool:
    """Check if a projected 3D box is at least partially visible in the image.

    Args:
        corners_2d: (8, 2) projected corner points.
        depth: (8,) depth values for each corner.
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        True if the box is at least partially visible.
    """
    # At least one corner must have positive depth
    if not np.any(depth > 0):
        return False

    # Check if any corner is within the image bounds (with margin)
    margin = 50
    in_bounds = (
        (corners_2d[:, 0] > -margin) &
        (corners_2d[:, 0] < img_w + margin) &
        (corners_2d[:, 1] > -margin) &
        (corners_2d[:, 1] < img_h + margin) &
        (depth > 0)
    )

    return np.any(in_bounds)


# =============================================================================
# NMS Post-Processing
# =============================================================================


def nms_3d_bev(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Perform class-aware NMS in BEV (bird's-eye-view) for 3D detections.

    Uses axis-aligned BEV IoU for NMS to suppress overlapping detections
    of the same class.

    Args:
        boxes: (N, 9) detected boxes: cx, cy, cz, w, l, h, yaw, vx, vy.
        scores: (N,) confidence scores.
        labels: (N,) class labels.
        iou_threshold: IoU threshold for suppression.

    Returns:
        keep_indices: Indices of boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    keep_indices = []
    unique_classes = np.unique(labels)

    for cls in unique_classes:
        cls_mask = labels == cls
        cls_indices = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]

        # Sort by score descending
        order = np.argsort(-cls_scores)
        cls_indices = cls_indices[order]
        cls_boxes = cls_boxes[order]

        # Compute BEV (axis-aligned) boxes: x_min, y_min, x_max, y_max
        half_w = np.abs(cls_boxes[:, 3]) / 2.0
        half_l = np.abs(cls_boxes[:, 4]) / 2.0
        bev_x1 = cls_boxes[:, 0] - half_w
        bev_y1 = cls_boxes[:, 1] - half_l
        bev_x2 = cls_boxes[:, 0] + half_w
        bev_y2 = cls_boxes[:, 1] + half_l

        areas = (bev_x2 - bev_x1) * (bev_y2 - bev_y1)

        suppressed = np.zeros(len(cls_indices), dtype=bool)

        for i in range(len(cls_indices)):
            if suppressed[i]:
                continue

            keep_indices.append(cls_indices[i])

            # Compute IoU with remaining boxes
            for j in range(i + 1, len(cls_indices)):
                if suppressed[j]:
                    continue

                # Intersection
                inter_x1 = max(bev_x1[i], bev_x1[j])
                inter_y1 = max(bev_y1[i], bev_y1[j])
                inter_x2 = min(bev_x2[i], bev_x2[j])
                inter_y2 = min(bev_y2[i], bev_y2[j])

                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                # Union
                union_area = areas[i] + areas[j] - inter_area
                iou = inter_area / max(union_area, 1e-6)

                if iou > iou_threshold:
                    suppressed[j] = True

    return np.array(keep_indices, dtype=np.int32)


# =============================================================================
# Model Loading and Inference
# =============================================================================


def load_model(config: Dict[str, Any], checkpoint_path: str) -> tf.keras.Model:
    """Build BEVFormer model and restore from checkpoint.

    Args:
        config: Model configuration dictionary.
        checkpoint_path: Path to TensorFlow checkpoint.

    Returns:
        Loaded model ready for inference.
    """
    logger.info("Building BEVFormer model...")

    # Import from model.py in the same directory
    try:
        from model import BEVFormer, build_bevformer, DEFAULT_CONFIG as MODEL_DEFAULT_CONFIG
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from model import BEVFormer, build_bevformer, DEFAULT_CONFIG as MODEL_DEFAULT_CONFIG

    model_cfg = config["model"]
    model_config = MODEL_DEFAULT_CONFIG.copy()
    model_config.update({
        "bev_h": model_cfg.get("bev_h", 200),
        "bev_w": model_cfg.get("bev_w", 200),
        "embed_dims": model_cfg.get("embed_dims", 256),
        "num_encoder_layers": model_cfg.get("num_encoder_layers", 6),
        "num_decoder_layers": model_cfg.get("num_decoder_layers", 6),
        "num_heads": model_cfg.get("num_heads", 8),
        "num_queries": model_cfg.get("num_query", 900),
        "num_classes": model_cfg.get("num_classes", 10),
        "num_levels": model_cfg.get("num_levels", 4),
    })

    model = build_bevformer(model_config)

    # Build model with dummy input
    input_h = config["data"].get("input_h", 480)
    input_w = config["data"].get("input_w", 800)
    dummy_inputs = {
        "images": tf.zeros([1, 6, input_h, input_w, 3]),
        "lidar2img": tf.zeros([1, 6, 4, 4]),
        "ego_motion": tf.eye(4, batch_shape=[1]),
        "prev_bev": None,
    }
    _ = model(dummy_inputs, training=False)

    # Restore checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    try:
        status.expect_partial()
    except Exception as e:
        logger.warning(f"Checkpoint restore note: {e}")

    logger.info("Model loaded successfully.")
    return model


def load_sample_data(
    sample_path: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Load a single sample's images and calibration data.

    Expected directory structure:
        sample_path/
            CAM_FRONT.jpg (or .png)
            CAM_FRONT_RIGHT.jpg
            CAM_FRONT_LEFT.jpg
            CAM_BACK.jpg
            CAM_BACK_LEFT.jpg
            CAM_BACK_RIGHT.jpg
            calibration.json (or calibration.npy)

    calibration.json format:
    {
        "lidar2img": {
            "CAM_FRONT": [[4x4 matrix]], ...
        }
    }

    Args:
        sample_path: Path to sample directory.
        config: Configuration dictionary.

    Returns:
        Dictionary with loaded data:
            - 'images': (6, input_h, input_w, 3) float32 normalized
            - 'images_raw': list of (H, W, 3) uint8 original images
            - 'lidar2img': (6, 4, 4) float32 calibration matrices
    """
    data_cfg = config["data"]
    input_h = data_cfg.get("input_h", 480)
    input_w = data_cfg.get("input_w", 800)
    img_h = data_cfg.get("img_h", 900)
    img_w = data_cfg.get("img_w", 1600)

    images_normalized = []
    images_raw = []

    # Load camera images
    for cam_name in CAMERA_NAMES:
        # Try multiple extensions
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(sample_path, f"{cam_name}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            logger.warning(f"Image not found for {cam_name} in {sample_path}")
            raw_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            norm_img = np.zeros((input_h, input_w, 3), dtype=np.float32)
        else:
            raw_img = cv2.imread(img_path)
            if raw_img is None:
                logger.warning(f"Failed to read image: {img_path}")
                raw_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                norm_img = np.zeros((input_h, input_w, 3), dtype=np.float32)
            else:
                # Convert BGR to RGB for processing
                raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

                # Resize for model input
                resized = cv2.resize(raw_img_rgb, (input_w, input_h))
                norm_img = resized.astype(np.float32) / 255.0

                # ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm_img = (norm_img - mean) / std

        images_raw.append(raw_img)
        images_normalized.append(norm_img)

    images_normalized = np.stack(images_normalized, axis=0)

    # Load calibration (lidar2img matrices)
    lidar2img = _load_calibration(sample_path, config)

    return {
        "images": images_normalized,
        "images_raw": images_raw,
        "lidar2img": lidar2img,
    }


def _load_calibration(sample_path: str, config: Dict[str, Any]) -> np.ndarray:
    """Load and adjust lidar2img calibration matrices.

    Supports JSON and numpy formats.

    Args:
        sample_path: Path to sample directory.
        config: Configuration dictionary.

    Returns:
        lidar2img: (6, 4, 4) float32 adjusted calibration matrices.
    """
    data_cfg = config["data"]
    input_h = data_cfg.get("input_h", 480)
    input_w = data_cfg.get("input_w", 800)
    img_h = data_cfg.get("img_h", 900)
    img_w = data_cfg.get("img_w", 1600)

    scale_x = input_w / img_w
    scale_y = input_h / img_h

    resize_matrix = np.array(
        [[scale_x, 0, 0, 0],
         [0, scale_y, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float32,
    )

    lidar2img = np.zeros((6, 4, 4), dtype=np.float32)

    # Try JSON format
    json_path = os.path.join(sample_path, "calibration.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            calib_data = json.load(f)

        l2i_data = calib_data.get("lidar2img", {})
        for i, cam_name in enumerate(CAMERA_NAMES):
            matrix = l2i_data.get(cam_name, np.eye(4).tolist())
            lidar2img[i] = resize_matrix @ np.array(matrix, dtype=np.float32)
        return lidar2img

    # Try numpy format
    npy_path = os.path.join(sample_path, "calibration.npy")
    if os.path.exists(npy_path):
        raw = np.load(npy_path)
        if raw.shape == (6, 4, 4):
            for i in range(6):
                lidar2img[i] = resize_matrix @ raw[i].astype(np.float32)
            return lidar2img

    # Try pickle format
    pkl_path = os.path.join(sample_path, "calibration.pkl")
    if os.path.exists(pkl_path):
        import pickle
        with open(pkl_path, "rb") as f:
            calib_data = pickle.load(f)
        if isinstance(calib_data, dict) and "lidar2img" in calib_data:
            raw = np.array(calib_data["lidar2img"], dtype=np.float32)
            if raw.shape == (6, 4, 4):
                for i in range(6):
                    lidar2img[i] = resize_matrix @ raw[i]
                return lidar2img

    logger.warning(
        f"No calibration file found in {sample_path}. "
        f"Using identity matrices (projections will be incorrect)."
    )
    for i in range(6):
        lidar2img[i] = np.eye(4, dtype=np.float32)

    return lidar2img


def run_inference(
    model: tf.keras.Model,
    images: np.ndarray,
    lidar2img: np.ndarray,
    config: Dict[str, Any],
    score_threshold: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Run model inference on a single sample and apply post-processing.

    Args:
        model: Loaded BEVFormer model.
        images: (6, input_h, input_w, 3) normalized images.
        lidar2img: (6, 4, 4) calibration matrices.
        config: Configuration dictionary.
        score_threshold: Minimum score to keep a detection.

    Returns:
        Dictionary with post-processed detections:
            - 'boxes': (N, 9) detected boxes: cx,cy,cz,w,l,h,yaw,vx,vy
            - 'scores': (N,) confidence scores
            - 'labels': (N,) class indices
            - 'class_names': List of class name strings
    """
    infer_cfg = config.get("inference", {})
    max_per_frame = infer_cfg.get("max_per_frame", 300)
    nms_iou_threshold = infer_cfg.get("nms_iou_threshold", 0.5)
    pc_range = infer_cfg.get("pc_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

    # Prepare model input
    images_batch = np.expand_dims(images, axis=0)  # (1, 6, H, W, 3)
    lidar2img_batch = np.expand_dims(lidar2img, axis=0)  # (1, 6, 4, 4)

    inputs = {
        "images": tf.constant(images_batch, dtype=tf.float32),
        "lidar2img": tf.constant(lidar2img_batch, dtype=tf.float32),
        "ego_motion": tf.eye(4, batch_shape=[1]),
        "prev_bev": None,
    }

    # Run inference
    logger.info("Running model inference...")
    start_time = time.time()
    outputs = model(inputs, training=False)
    inference_time = time.time() - start_time
    logger.info(f"Inference time: {inference_time * 1000:.1f} ms")

    # Extract predictions
    cls_logits = outputs["cls_logits"].numpy()[0]  # (num_queries, num_classes)
    bbox_preds = outputs["bbox_preds"].numpy()[0]  # (num_queries, 10)

    # Decode predictions
    cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))  # sigmoid
    max_scores = np.max(cls_probs, axis=-1)
    max_labels = np.argmax(cls_probs, axis=-1)

    # Filter by score threshold
    keep_mask = max_scores >= score_threshold
    if not np.any(keep_mask):
        logger.info("No detections above score threshold.")
        return {
            "boxes": np.zeros((0, 9), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int32),
            "class_names": [],
        }

    scores = max_scores[keep_mask]
    labels = max_labels[keep_mask]
    boxes_raw = bbox_preds[keep_mask]

    # Decode bounding boxes
    x_range = pc_range[3] - pc_range[0]
    y_range = pc_range[4] - pc_range[1]
    z_range = pc_range[5] - pc_range[2]

    cx = boxes_raw[:, 0] * x_range + pc_range[0]
    cy = boxes_raw[:, 1] * y_range + pc_range[1]
    cz = boxes_raw[:, 2] * z_range + pc_range[2]
    w = np.exp(boxes_raw[:, 3])
    l = np.exp(boxes_raw[:, 4])
    h = np.exp(boxes_raw[:, 5])
    sin_yaw = boxes_raw[:, 6]
    cos_yaw = boxes_raw[:, 7]
    vx = boxes_raw[:, 8]
    vy = boxes_raw[:, 9]

    yaw = np.arctan2(sin_yaw, cos_yaw)

    boxes = np.stack([cx, cy, cz, w, l, h, yaw, vx, vy], axis=-1)

    # Sort by score and keep top-k before NMS
    sort_indices = np.argsort(-scores)[:max_per_frame]
    boxes = boxes[sort_indices]
    scores = scores[sort_indices]
    labels = labels[sort_indices]

    # Apply NMS
    logger.info(f"Detections before NMS: {len(boxes)}")
    nms_keep = nms_3d_bev(boxes, scores, labels, iou_threshold=nms_iou_threshold)

    if len(nms_keep) > 0:
        boxes = boxes[nms_keep]
        scores = scores[nms_keep]
        labels = labels[nms_keep]
    else:
        boxes = np.zeros((0, 9), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int32)

    # Final top-k after NMS
    if len(boxes) > max_per_frame:
        top_k = np.argsort(-scores)[:max_per_frame]
        boxes = boxes[top_k]
        scores = scores[top_k]
        labels = labels[top_k]

    logger.info(f"Detections after NMS: {len(boxes)}")

    class_names = [NUSCENES_CLASSES[l] for l in labels]

    return {
        "boxes": boxes.astype(np.float32),
        "scores": scores.astype(np.float32),
        "labels": labels.astype(np.int32),
        "class_names": class_names,
    }


# =============================================================================
# Visualization: Camera View with 3D Box Projections
# =============================================================================


def draw_3d_box_on_image(
    image: np.ndarray,
    corners_2d: np.ndarray,
    depth: np.ndarray,
    color: Tuple[int, int, int],
    label_text: str,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a projected 3D bounding box on an image.

    Draws the 12 edges of a 3D box and the class label. Only draws edges
    where both endpoints have positive depth (visible to camera).

    Args:
        image: (H, W, 3) BGR image to draw on.
        corners_2d: (8, 2) projected 2D corner points.
        depth: (8,) depth values for each corner.
        color: BGR color tuple.
        label_text: Text label to display.
        thickness: Line thickness in pixels.

    Returns:
        Image with drawn box.
    """
    img = image.copy()

    # Define the 12 edges of a 3D box
    # Bottom face: 0-1, 1-2, 2-3, 3-0
    # Top face: 4-5, 5-6, 6-7, 7-4
    # Vertical edges: 0-4, 1-5, 2-6, 3-7
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]

    # Draw front face with thicker lines
    front_edges = [(0, 1), (4, 5), (0, 4), (1, 5)]

    for i, j in edges:
        # Only draw if both endpoints have positive depth
        if depth[i] > 0 and depth[j] > 0:
            pt1 = (int(corners_2d[i, 0]), int(corners_2d[i, 1]))
            pt2 = (int(corners_2d[j, 0]), int(corners_2d[j, 1]))

            line_thickness = thickness + 1 if (i, j) in front_edges else thickness
            cv2.line(img, pt1, pt2, color, line_thickness)

    # Draw label at the top-front edge midpoint
    visible_top = [(idx, corners_2d[idx]) for idx in [4, 5, 6, 7] if depth[idx] > 0]
    if visible_top:
        label_pt = visible_top[0][1]
        label_x = int(label_pt[0])
        label_y = int(label_pt[1]) - 5

        # Background rectangle for text readability
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        cv2.rectangle(
            img, (label_x, label_y - th - 2), (label_x + tw, label_y + 2),
            color, -1
        )
        cv2.putText(
            img, label_text, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )

    return img


def visualize_camera_views(
    images_raw: List[np.ndarray],
    detections: Dict[str, np.ndarray],
    lidar2img: np.ndarray,
    output_dir: str,
) -> None:
    """Render 3D box projections on all 6 camera views and save.

    Creates individual annotated camera images and a combined 6-camera mosaic.

    Args:
        images_raw: List of 6 raw BGR images.
        detections: Detection results dictionary.
        lidar2img: (6, 4, 4) original (un-resized) calibration matrices.
        output_dir: Directory to save visualizations.
    """
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]
    class_names = detections["class_names"]

    annotated_images = []

    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        img = images_raw[cam_idx].copy()
        img_h, img_w = img.shape[:2]
        cam_matrix = lidar2img[cam_idx]

        # Draw each detection
        for det_idx in range(len(boxes)):
            box = boxes[det_idx]
            score = scores[det_idx]
            cls_name = class_names[det_idx]

            # Get 3D corners
            corners_3d = get_3d_box_corners(box)

            # Project to this camera
            corners_2d, depth = project_3d_to_2d(corners_3d, cam_matrix)

            # Check visibility
            if not is_box_visible(corners_2d, depth, img_h, img_w):
                continue

            # Get color for this class
            color = CLASS_COLORS_BGR.get(cls_name, (255, 255, 255))

            # Label text
            label_text = f"{cls_name} {score:.2f}"

            # Draw box
            img = draw_3d_box_on_image(
                img, corners_2d, depth, color, label_text, thickness=2
            )

        # Add camera name overlay
        cv2.putText(
            img, cam_name, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA
        )

        annotated_images.append(img)

        # Save individual camera image
        cam_output_path = os.path.join(output_dir, f"{cam_name}_detections.jpg")
        cv2.imwrite(cam_output_path, img)

    # Create 6-camera mosaic (2 rows x 3 columns)
    _create_camera_mosaic(annotated_images, output_dir)


def _create_camera_mosaic(images: List[np.ndarray], output_dir: str) -> None:
    """Create a 2x3 mosaic of annotated camera images.

    Layout:
        FRONT_LEFT  |  FRONT  |  FRONT_RIGHT
        BACK_LEFT   |  BACK   |  BACK_RIGHT

    Args:
        images: List of 6 annotated BGR images (in nuScenes camera order).
        output_dir: Directory to save the mosaic.
    """
    # Reorder for display layout
    # Input order: FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT
    # Desired layout:
    #   Row 1: FRONT_LEFT(2), FRONT(0), FRONT_RIGHT(1)
    #   Row 2: BACK_LEFT(4), BACK(3), BACK_RIGHT(5)
    layout_indices = [2, 0, 1, 4, 3, 5]

    # Resize all images to the same size for mosaic
    target_h = 360
    target_w = 640
    resized = []
    for idx in layout_indices:
        img = images[idx]
        img_resized = cv2.resize(img, (target_w, target_h))
        resized.append(img_resized)

    # Assemble mosaic
    row1 = np.concatenate(resized[0:3], axis=1)
    row2 = np.concatenate(resized[3:6], axis=1)
    mosaic = np.concatenate([row1, row2], axis=0)

    mosaic_path = os.path.join(output_dir, "camera_mosaic.jpg")
    cv2.imwrite(mosaic_path, mosaic)
    logger.info(f"Camera mosaic saved: {mosaic_path}")


# =============================================================================
# Visualization: BEV Top-Down View
# =============================================================================


def visualize_bev(
    detections: Dict[str, np.ndarray],
    output_dir: str,
    pc_range: List[float] = None,
    bev_resolution: float = 0.2,
) -> None:
    """Render bird's-eye-view (top-down) visualization of detected boxes.

    Creates a top-down map showing detected objects as colored oriented
    rectangles with class labels and confidence scores.

    Args:
        detections: Detection results dictionary.
        output_dir: Directory to save the BEV visualization.
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        bev_resolution: Resolution in meters per pixel for the BEV image.
    """
    if pc_range is None:
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]
    class_names = detections["class_names"]

    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    x_range = x_max - x_min
    y_range = y_max - y_min

    # BEV image dimensions
    bev_w = int(x_range / bev_resolution)
    bev_h = int(y_range / bev_resolution)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d2d2d")
    ax.grid(True, alpha=0.2, color="white", linestyle="--")
    ax.set_xlabel("X (meters)", fontsize=10, color="white")
    ax.set_ylabel("Y (meters)", fontsize=10, color="white")
    ax.set_title("BEV Detection Results", fontsize=14, color="white", pad=10)
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#1a1a1a")

    # Draw ego vehicle at center
    ego_rect = plt.Rectangle(
        (-1.0, -2.0), 2.0, 4.0,
        linewidth=2, edgecolor="white", facecolor="gray", alpha=0.7
    )
    ax.add_patch(ego_rect)
    ax.annotate(
        "EGO", (0, 0), color="white", fontsize=8,
        ha="center", va="center", fontweight="bold"
    )

    # Draw range circles
    for r in [10, 20, 30, 40, 50]:
        circle = plt.Circle(
            (0, 0), r, fill=False, color="white", alpha=0.15, linestyle=":"
        )
        ax.add_patch(circle)
        ax.annotate(
            f"{r}m", (r * 0.707, r * 0.707),
            color="white", fontsize=7, alpha=0.4
        )

    # Draw detected boxes
    for det_idx in range(len(boxes)):
        box = boxes[det_idx]
        score = scores[det_idx]
        cls_name = class_names[det_idx]

        cx, cy = box[0], box[1]
        w, l = box[3], box[4]
        yaw = box[6]

        # Color for this class
        color = CLASS_COLORS_NORM.get(cls_name, (1.0, 1.0, 1.0))

        # Draw oriented rectangle
        corners = _get_bev_corners(cx, cy, w, l, yaw)

        # Create polygon
        polygon = plt.Polygon(
            corners, closed=True,
            fill=True, facecolor=color + (0.3,),
            edgecolor=color, linewidth=1.5
        )
        ax.add_patch(polygon)

        # Draw heading direction (front of vehicle)
        heading_x = cx + (l / 2.0) * np.cos(yaw)
        heading_y = cy + (l / 2.0) * np.sin(yaw)
        ax.annotate(
            "", xy=(heading_x, heading_y), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5)
        )

        # Draw velocity vector if non-zero
        vx, vy = box[7], box[8]
        vel_mag = np.sqrt(vx * vx + vy * vy)
        if vel_mag > 0.5:
            vel_scale = 2.0  # Scale factor for visualization
            ax.annotate(
                "", xy=(cx + vx * vel_scale, cy + vy * vel_scale),
                xytext=(cx, cy),
                arrowprops=dict(
                    arrowstyle="->", color="cyan", lw=1.0,
                    linestyle="dashed"
                )
            )

        # Label with class and score
        label = f"{cls_name}\n{score:.2f}"
        ax.annotate(
            label, (cx, cy + l / 2.0 + 1.0),
            color=color, fontsize=6, ha="center", va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6)
        )

    # Create legend
    legend_patches = []
    classes_in_scene = set(class_names)
    for cls in NUSCENES_CLASSES:
        if cls in classes_in_scene:
            color = CLASS_COLORS_NORM.get(cls, (1.0, 1.0, 1.0))
            patch = mpatches.Patch(color=color, label=cls)
            legend_patches.append(patch)

    if legend_patches:
        legend = ax.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=8,
            framealpha=0.7,
            facecolor="#333333",
            edgecolor="white",
        )
        for text in legend.get_texts():
            text.set_color("white")

    # Add detection count annotation
    count_text = f"Total detections: {len(boxes)}"
    ax.annotate(
        count_text, (x_min + 2, y_max - 2),
        color="white", fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7)
    )

    plt.tight_layout()
    bev_path = os.path.join(output_dir, "bev_detections.png")
    plt.savefig(bev_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"BEV visualization saved: {bev_path}")


def _get_bev_corners(
    cx: float, cy: float, w: float, l: float, yaw: float
) -> np.ndarray:
    """Get 4 BEV corners of a rotated rectangle.

    Args:
        cx, cy: Center coordinates.
        w: Width of the box.
        l: Length of the box.
        yaw: Heading angle in radians.

    Returns:
        corners: (4, 2) array of corner coordinates.
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Half dimensions
    hw = w / 2.0
    hl = l / 2.0

    # Corners in local frame
    corners_local = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ])

    # Rotation matrix
    R = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw],
    ])

    # Rotate and translate
    corners_world = (R @ corners_local.T).T
    corners_world[:, 0] += cx
    corners_world[:, 1] += cy

    return corners_world


# =============================================================================
# Results Summary
# =============================================================================


def save_detection_results(
    detections: Dict[str, np.ndarray],
    output_dir: str,
    inference_config: Dict[str, Any],
) -> None:
    """Save detection results to JSON for downstream processing.

    Args:
        detections: Detection results dictionary.
        output_dir: Directory to save results.
        inference_config: Configuration used for inference.
    """
    results = {
        "num_detections": len(detections["boxes"]),
        "detections": [],
        "config": {
            "score_threshold": inference_config.get("inference", {}).get(
                "score_threshold", 0.1
            ),
            "max_per_frame": inference_config.get("inference", {}).get(
                "max_per_frame", 300
            ),
        },
    }

    for i in range(len(detections["boxes"])):
        det = {
            "class": detections["class_names"][i],
            "class_id": int(detections["labels"][i]),
            "score": float(detections["scores"][i]),
            "box": {
                "cx": float(detections["boxes"][i, 0]),
                "cy": float(detections["boxes"][i, 1]),
                "cz": float(detections["boxes"][i, 2]),
                "w": float(detections["boxes"][i, 3]),
                "l": float(detections["boxes"][i, 4]),
                "h": float(detections["boxes"][i, 5]),
                "yaw": float(detections["boxes"][i, 6]),
                "vx": float(detections["boxes"][i, 7]),
                "vy": float(detections["boxes"][i, 8]),
            },
        }
        results["detections"].append(det)

    output_path = os.path.join(output_dir, "detections.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Detection results saved: {output_path}")


def print_detection_summary(detections: Dict[str, np.ndarray]) -> None:
    """Print a formatted summary of detection results.

    Args:
        detections: Detection results dictionary.
    """
    boxes = detections["boxes"]
    scores = detections["scores"]
    class_names = detections["class_names"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("  DETECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total detections: {len(boxes)}")
    logger.info("")

    if len(boxes) == 0:
        logger.info("  No detections found.")
        return

    # Per-class breakdown
    logger.info(f"  {'Class':<24}{'Count':<8}{'Avg Score':<12}{'Avg Dist':<10}")
    logger.info(f"  {'-' * 54}")

    class_stats = {}
    for i in range(len(boxes)):
        cls = class_names[i]
        if cls not in class_stats:
            class_stats[cls] = {"count": 0, "scores": [], "dists": []}
        class_stats[cls]["count"] += 1
        class_stats[cls]["scores"].append(scores[i])
        dist = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
        class_stats[cls]["dists"].append(dist)

    for cls in NUSCENES_CLASSES:
        if cls in class_stats:
            stats = class_stats[cls]
            avg_score = np.mean(stats["scores"])
            avg_dist = np.mean(stats["dists"])
            logger.info(
                f"  {cls:<24}{stats['count']:<8}{avg_score:<12.3f}{avg_dist:<10.1f}m"
            )

    logger.info(f"  {'-' * 54}")
    logger.info(
        f"  {'TOTAL':<24}{len(boxes):<8}"
        f"{np.mean(scores):<12.3f}{np.mean(np.sqrt(boxes[:, 0]**2 + boxes[:, 1]**2)):<10.1f}m"
    )
    logger.info("=" * 60)


# =============================================================================
# Command Line Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="BEVFormer Inference and Visualization (TensorFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to TensorFlow checkpoint for inference.",
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        required=True,
        help=(
            "Path to sample directory containing 6 camera images "
            "(CAM_FRONT.jpg, etc.) and calibration.json."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results",
        help="Directory to save inference visualizations and results.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.1,
        help="Minimum confidence score to display a detection.",
    )

    args = parser.parse_args()

    # Validate
    if args.score_threshold < 0 or args.score_threshold > 1:
        parser.error("--score_threshold must be in [0, 1]")

    return args


def main():
    """Main inference entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override score threshold from command line
    if "inference" not in config:
        config["inference"] = {}
    config["inference"]["score_threshold"] = args.score_threshold

    # Log system info
    logger.info("=" * 70)
    logger.info("BEVFormer Inference (TensorFlow)")
    logger.info("=" * 70)
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Sample path: {args.sample_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Score threshold: {args.score_threshold}")
    logger.info(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load model
        model = load_model(config, args.checkpoint)

        # Load sample data
        logger.info(f"Loading sample from: {args.sample_path}")
        sample_data = load_sample_data(args.sample_path, config)

        # Run inference
        detections = run_inference(
            model=model,
            images=sample_data["images"],
            lidar2img=sample_data["lidar2img"],
            config=config,
            score_threshold=args.score_threshold,
        )

        # Print detection summary
        print_detection_summary(detections)

        # Visualize camera views with 3D box projections
        logger.info("Generating camera view visualizations...")
        visualize_camera_views(
            images_raw=sample_data["images_raw"],
            detections=detections,
            lidar2img=sample_data["lidar2img"],
            output_dir=args.output_dir,
        )

        # Visualize BEV top-down view
        logger.info("Generating BEV visualization...")
        pc_range = config.get("inference", {}).get(
            "pc_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        )
        visualize_bev(
            detections=detections,
            output_dir=args.output_dir,
            pc_range=pc_range,
        )

        # Save detection results as JSON
        save_detection_results(detections, args.output_dir, config)

        logger.info("")
        logger.info("Inference complete. Output files:")
        logger.info(f"  Camera views: {args.output_dir}/<CAM_NAME>_detections.jpg")
        logger.info(f"  Camera mosaic: {args.output_dir}/camera_mosaic.jpg")
        logger.info(f"  BEV view: {args.output_dir}/bev_detections.png")
        logger.info(f"  Results JSON: {args.output_dir}/detections.json")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
