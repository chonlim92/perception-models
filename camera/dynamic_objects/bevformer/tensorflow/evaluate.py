#!/usr/bin/env python3
"""BEVFormer Evaluation Script (TensorFlow).

Runs full nuScenes detection evaluation on the validation set, computing:
- mAP at distance thresholds: 0.5, 1.0, 2.0, 4.0 meters
- NDS (nuScenes Detection Score)
- mATE, mASE, mAOE, mAVE, mAAE (True Positive error metrics)
- Per-class AP for all 10 nuScenes classes

Usage:
    python evaluate.py --config configs/bevformer_base.yaml \
                       --checkpoint ./work_dirs/bevformer_tf/checkpoints/ckpt-24 \
                       --data_root /data/nuscenes \
                       --output_dir ./eval_results \
                       --batch_size 1
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bevformer.evaluate")


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

# Distance thresholds for AP computation (in meters)
DISTANCE_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]

# nuScenes attribute mapping per class
CLASS_ATTRIBUTES = {
    "car": ["vehicle.moving", "vehicle.parked", "vehicle.stopped"],
    "truck": ["vehicle.moving", "vehicle.parked", "vehicle.stopped"],
    "construction_vehicle": ["vehicle.moving", "vehicle.parked", "vehicle.stopped"],
    "bus": ["vehicle.moving", "vehicle.stopped"],
    "trailer": ["vehicle.moving", "vehicle.parked", "vehicle.stopped"],
    "barrier": [""],
    "motorcycle": ["cycle.with_rider", "cycle.without_rider"],
    "bicycle": ["cycle.with_rider", "cycle.without_rider"],
    "pedestrian": [
        "pedestrian.moving",
        "pedestrian.standing",
        "pedestrian.sitting_lying_down",
    ],
    "traffic_cone": [""],
}

# nuScenes velocity error cap (m/s) for classes without velocity annotation
VELOCITY_ERROR_CLASSES = {
    "barrier": True,
    "traffic_cone": True,
}


# =============================================================================
# Configuration Loading
# =============================================================================

DEFAULT_EVAL_CONFIG = {
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
        "temporal_num_frames": 2,
    },
    "data": {
        "dataset_root": "/data/nuscenes",
        "ann_file_val": "data/nuscenes_infos_val.pkl",
        "img_h": 900,
        "img_w": 1600,
        "input_h": 480,
        "input_w": 800,
        "num_cameras": 6,
        "camera_names": [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ],
        "max_objects": 300,
    },
    "eval": {
        "score_threshold": 0.1,
        "max_detections_per_sample": 300,
        "distance_thresholds": [0.5, 1.0, 2.0, 4.0],
    },
}


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load evaluation configuration from YAML, merged with defaults."""
    config = DEFAULT_EVAL_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            config = _deep_merge(config, user_config)
        logger.info(f"Loaded config from {config_path}")
    else:
        if config_path:
            logger.warning(f"Config file {config_path} not found, using defaults.")
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
# Data Loading for Evaluation
# =============================================================================


class NuScenesEvalDataLoader:
    """Data loader for nuScenes validation set evaluation.

    Loads multi-camera images, calibration matrices, and ground truth
    annotations for evaluation without augmentation.
    """

    def __init__(self, config: Dict[str, Any], data_root: str):
        """Initialize evaluation data loader.

        Args:
            config: Full configuration dictionary.
            data_root: Path to nuScenes dataset root.
        """
        self.config = config
        self.data_root = data_root
        self.data_cfg = config["data"]
        self.model_cfg = config["model"]

        self.input_h = self.data_cfg["input_h"]
        self.input_w = self.data_cfg["input_w"]
        self.img_h = self.data_cfg["img_h"]
        self.img_w = self.data_cfg["img_w"]
        self.num_cameras = self.data_cfg["num_cameras"]
        self.camera_names = self.data_cfg["camera_names"]
        self.max_objects = self.data_cfg["max_objects"]

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load validation annotations from pickle file.

        Returns:
            List of annotation dictionaries, one per sample.
        """
        ann_file = self.data_cfg["ann_file_val"]
        ann_path = os.path.join(self.data_root, ann_file)

        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}. "
                f"Please generate nuScenes info files."
            )

        with open(ann_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "infos" in data:
            infos = data["infos"]
        elif isinstance(data, list):
            infos = data
        else:
            raise ValueError(f"Unexpected annotation format in {ann_path}")

        logger.info(f"Loaded {len(infos)} validation samples from {ann_path}")
        return infos

    def get_sample(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Load and preprocess a single evaluation sample.

        Args:
            info: Sample annotation dictionary.

        Returns:
            Dictionary with processed inputs and ground truth.
        """
        # Load multi-camera images
        images = self._load_images(info)

        # Load and adjust calibration matrices
        lidar2img = self._load_calibration(info)

        # Load ground truth
        gt_boxes, gt_labels, gt_names = self._load_ground_truth(info)

        # Sample metadata
        token = info.get("token", "")
        timestamp = info.get("timestamp", 0)

        return {
            "images": images,
            "lidar2img": lidar2img,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "gt_names": gt_names,
            "token": token,
            "timestamp": timestamp,
        }

    def _load_images(self, info: Dict[str, Any]) -> np.ndarray:
        """Load and preprocess multi-camera images.

        Args:
            info: Sample annotation dictionary.

        Returns:
            images: (num_cameras, input_h, input_w, 3) float32 normalized.
        """
        images = []
        cams = info.get("cams", {})

        for cam_name in self.camera_names:
            cam_info = cams.get(cam_name, {})
            cam_path = cam_info.get("data_path", "")

            if not os.path.isabs(cam_path):
                cam_path = os.path.join(self.data_root, cam_path)

            if os.path.exists(cam_path):
                raw = tf.io.read_file(cam_path)
                img = tf.io.decode_jpeg(raw, channels=3)
                img = tf.image.resize(img, [self.input_h, self.input_w])
                img = tf.cast(img, tf.float32) / 255.0
                # Apply ImageNet normalization
                mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
                std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
                img = (img - mean) / std
            else:
                img = tf.zeros(
                    [self.input_h, self.input_w, 3], dtype=tf.float32
                )

            images.append(img.numpy())

        return np.stack(images, axis=0)

    def _load_calibration(self, info: Dict[str, Any]) -> np.ndarray:
        """Load and adjust lidar2img calibration matrices.

        Args:
            info: Sample annotation dictionary.

        Returns:
            lidar2img: (num_cameras, 4, 4) float32 adjusted for image resize.
        """
        cams = info.get("cams", {})
        lidar2img_list = []

        scale_x = self.input_w / self.img_w
        scale_y = self.input_h / self.img_h
        resize_matrix = np.array(
            [[scale_x, 0, 0, 0],
             [0, scale_y, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        for cam_name in self.camera_names:
            cam_info = cams.get(cam_name, {})
            l2i = cam_info.get("lidar2img", np.eye(4, dtype=np.float32))
            l2i = np.array(l2i, dtype=np.float32)
            # Adjust for image resizing
            l2i_adjusted = resize_matrix @ l2i
            lidar2img_list.append(l2i_adjusted)

        return np.stack(lidar2img_list, axis=0)

    def _load_ground_truth(
        self, info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load ground truth boxes and labels.

        Args:
            info: Sample annotation dictionary.

        Returns:
            gt_boxes: (N, 9) float32 - cx, cy, cz, w, l, h, yaw, vx, vy
            gt_labels: (N,) int32
            gt_names: List of class name strings
        """
        gt_boxes = info.get("gt_boxes", np.zeros((0, 9), dtype=np.float32))
        gt_labels = info.get("gt_labels", np.zeros((0,), dtype=np.int32))
        gt_names = info.get("gt_names", [])

        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int32)

        # If gt_names not provided, derive from labels
        if len(gt_names) == 0 and len(gt_labels) > 0:
            gt_names = [NUSCENES_CLASSES[l] for l in gt_labels if l < len(NUSCENES_CLASSES)]

        return gt_boxes, gt_labels, gt_names

    def build_tf_dataset(self, infos: List[Dict[str, Any]], batch_size: int) -> tf.data.Dataset:
        """Build a tf.data.Dataset for batched evaluation.

        Args:
            infos: List of annotation dictionaries.
            batch_size: Batch size for evaluation.

        Returns:
            tf.data.Dataset yielding batched inputs.
        """
        num_cameras = self.num_cameras
        input_h = self.input_h
        input_w = self.input_w

        def generator():
            for info in infos:
                sample = self.get_sample(info)
                yield (
                    sample["images"],
                    sample["lidar2img"],
                    sample["token"],
                )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(num_cameras, input_h, input_w, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(num_cameras, 4, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string),
            ),
        )

        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


# =============================================================================
# Detection Post-Processing
# =============================================================================


def decode_predictions(
    cls_logits: np.ndarray,
    bbox_preds: np.ndarray,
    pc_range: List[float],
    score_threshold: float = 0.1,
    max_detections: int = 300,
) -> Dict[str, np.ndarray]:
    """Decode raw model predictions into detection results.

    Args:
        cls_logits: (num_queries, num_classes) classification logits.
        bbox_preds: (num_queries, 10) bounding box predictions
                    (normalized cx,cy,cz,w,l,h,sin,cos,vx,vy).
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        score_threshold: Minimum confidence score to keep a detection.
        max_detections: Maximum number of detections to return.

    Returns:
        Dictionary with:
            - 'boxes': (N, 9) decoded boxes: cx,cy,cz,w,l,h,yaw,vx,vy
            - 'scores': (N,) confidence scores
            - 'labels': (N,) class indices
    """
    # Compute class probabilities via sigmoid
    cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))  # sigmoid

    # Get maximum class score per query
    max_scores = np.max(cls_probs, axis=-1)
    max_labels = np.argmax(cls_probs, axis=-1)

    # Filter by score threshold
    keep_mask = max_scores >= score_threshold
    if not np.any(keep_mask):
        return {
            "boxes": np.zeros((0, 9), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int32),
        }

    scores = max_scores[keep_mask]
    labels = max_labels[keep_mask]
    boxes_raw = bbox_preds[keep_mask]

    # Decode bounding boxes from normalized to world coordinates
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

    # Convert sin/cos to yaw angle
    yaw = np.arctan2(sin_yaw, cos_yaw)

    # Stack decoded boxes: cx, cy, cz, w, l, h, yaw, vx, vy
    boxes = np.stack([cx, cy, cz, w, l, h, yaw, vx, vy], axis=-1)

    # Sort by score and keep top-k
    sort_indices = np.argsort(-scores)[:max_detections]
    boxes = boxes[sort_indices]
    scores = scores[sort_indices]
    labels = labels[sort_indices]

    return {
        "boxes": boxes.astype(np.float32),
        "scores": scores.astype(np.float32),
        "labels": labels.astype(np.int32),
    }


# =============================================================================
# nuScenes Evaluation Metrics
# =============================================================================


def compute_center_distance(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute 2D center distance between prediction and ground truth.

    Args:
        pred_box: (9,) predicted box [cx, cy, cz, w, l, h, yaw, vx, vy].
        gt_box: (9,) ground truth box [cx, cy, cz, w, l, h, yaw, vx, vy].

    Returns:
        Euclidean distance between centers in BEV (x, y).
    """
    dx = pred_box[0] - gt_box[0]
    dy = pred_box[1] - gt_box[1]
    return np.sqrt(dx * dx + dy * dy)


def compute_translation_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute 3D translation error (ATE).

    Args:
        pred_box: (9,) predicted box.
        gt_box: (9,) ground truth box.

    Returns:
        Euclidean distance between 3D centers.
    """
    diff = pred_box[:3] - gt_box[:3]
    return np.sqrt(np.sum(diff ** 2))


def compute_scale_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute scale error (ASE) as 1 - 3D IoU of aligned boxes.

    Approximates scale error using volume ratio instead of full 3D IoU
    for computational efficiency.

    Args:
        pred_box: (9,) predicted box.
        gt_box: (9,) ground truth box.

    Returns:
        Scale error in [0, 1].
    """
    # Volume of predicted and ground truth boxes
    pred_vol = abs(pred_box[3] * pred_box[4] * pred_box[5])
    gt_vol = abs(gt_box[3] * gt_box[4] * gt_box[5])

    if pred_vol < 1e-6 or gt_vol < 1e-6:
        return 1.0

    # Compute axis-aligned 3D IoU with boxes centered at origin
    pred_half = np.abs(pred_box[3:6]) / 2.0
    gt_half = np.abs(gt_box[3:6]) / 2.0

    # Intersection dimensions (boxes aligned at center)
    inter_dims = np.minimum(pred_half, gt_half) * 2.0
    inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]

    # Union
    union_vol = pred_vol + gt_vol - inter_vol
    iou = inter_vol / max(union_vol, 1e-6)

    return 1.0 - iou


def compute_orientation_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute orientation error (AOE) as absolute yaw difference.

    Args:
        pred_box: (9,) predicted box (yaw at index 6).
        gt_box: (9,) ground truth box (yaw at index 6).

    Returns:
        Absolute angular difference in radians, wrapped to [0, pi].
    """
    pred_yaw = pred_box[6]
    gt_yaw = gt_box[6]

    # Compute smallest angular difference
    diff = abs(pred_yaw - gt_yaw)
    diff = diff % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff

    return diff


def compute_velocity_error(pred_box: np.ndarray, gt_box: np.ndarray) -> float:
    """Compute velocity error (AVE) as L2 distance between velocity vectors.

    Args:
        pred_box: (9,) predicted box (vx at index 7, vy at index 8).
        gt_box: (9,) ground truth box (vx at index 7, vy at index 8).

    Returns:
        L2 distance between velocity vectors.
    """
    pred_vel = pred_box[7:9]
    gt_vel = gt_box[7:9]
    return np.sqrt(np.sum((pred_vel - gt_vel) ** 2))


def compute_attribute_error(
    pred_label: int, gt_label: int, pred_score: float
) -> float:
    """Compute attribute error (AAE).

    In the simplified evaluation, attribute error is binary: 0 if the
    class prediction matches ground truth, 1 otherwise. In full nuScenes
    evaluation, this would compare predicted attributes.

    Args:
        pred_label: Predicted class index.
        gt_label: Ground truth class index.
        pred_score: Prediction confidence (unused, for API compatibility).

    Returns:
        Attribute error: 0.0 if class matches, 1.0 otherwise.
    """
    return 0.0 if pred_label == gt_label else 1.0


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    distance_threshold: float,
    class_idx: int,
) -> Tuple[List[bool], int]:
    """Match predictions to ground truth for a single class at a given threshold.

    Uses greedy matching: predictions are sorted by score (descending), and
    each is matched to the closest unmatched ground truth within the distance
    threshold.

    Args:
        pred_boxes: (M, 9) predicted boxes.
        pred_scores: (M,) prediction scores.
        pred_labels: (M,) predicted class indices.
        gt_boxes: (K, 9) ground truth boxes.
        gt_labels: (K,) ground truth class indices.
        distance_threshold: Maximum center distance for a match.
        class_idx: Class index to evaluate.

    Returns:
        tp_list: List of booleans indicating TP (True) or FP (False) for each
                 prediction of this class, sorted by descending score.
        num_gt: Number of ground truth objects for this class.
    """
    # Filter predictions and GT for the target class
    pred_mask = pred_labels == class_idx
    gt_mask = gt_labels == class_idx

    class_pred_indices = np.where(pred_mask)[0]
    class_gt_indices = np.where(gt_mask)[0]

    num_gt = len(class_gt_indices)
    if len(class_pred_indices) == 0:
        return [], num_gt

    # Sort predictions by score (descending)
    class_scores = pred_scores[class_pred_indices]
    sort_order = np.argsort(-class_scores)
    sorted_pred_indices = class_pred_indices[sort_order]

    # Track which GT boxes have been matched
    gt_matched = np.zeros(num_gt, dtype=bool)
    tp_list = []

    for pred_idx in sorted_pred_indices:
        pred_box = pred_boxes[pred_idx]
        is_tp = False

        # Find closest unmatched GT
        best_dist = float("inf")
        best_gt_local_idx = -1

        for gt_local_idx, gt_global_idx in enumerate(class_gt_indices):
            if gt_matched[gt_local_idx]:
                continue

            dist = compute_center_distance(pred_box, gt_boxes[gt_global_idx])
            if dist < best_dist:
                best_dist = dist
                best_gt_local_idx = gt_local_idx

        # Check if match is within threshold
        if best_gt_local_idx >= 0 and best_dist <= distance_threshold:
            gt_matched[best_gt_local_idx] = True
            is_tp = True

        tp_list.append(is_tp)

    return tp_list, num_gt


def compute_ap(tp_list: List[bool], num_gt: int) -> float:
    """Compute Average Precision from TP/FP list using all-point interpolation.

    Args:
        tp_list: List of booleans indicating TP or FP (sorted by score desc).
        num_gt: Total number of ground truth objects.

    Returns:
        AP value in [0, 1].
    """
    if num_gt == 0:
        return 0.0

    if len(tp_list) == 0:
        return 0.0

    tp_array = np.array(tp_list, dtype=np.float32)
    fp_array = 1.0 - tp_array

    # Cumulative sums
    tp_cumsum = np.cumsum(tp_array)
    fp_cumsum = np.cumsum(fp_array)

    # Precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gt

    # Append sentinel values
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # Ensure precision is monotonically decreasing (all-point interpolation)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute area under PR curve using trapezoidal rule
    recall_diff = np.diff(recall)
    ap = np.sum(recall_diff * precision[1:])

    return float(ap)


def compute_tp_errors(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    distance_threshold: float = 2.0,
) -> Dict[str, Dict[str, float]]:
    """Compute True Positive errors for all classes.

    Matches predictions to ground truth at the given distance threshold, then
    computes error metrics for true positive detections.

    Args:
        pred_boxes: (M, 9) predicted boxes.
        pred_scores: (M,) prediction scores.
        pred_labels: (M,) predicted class indices.
        gt_boxes: (K, 9) ground truth boxes.
        gt_labels: (K,) ground truth class indices.
        distance_threshold: Distance threshold for matching (default 2.0m).

    Returns:
        Per-class dictionary of error metrics:
            {class_name: {ATE, ASE, AOE, AVE, AAE}}
    """
    errors = {}

    for class_idx, class_name in enumerate(NUSCENES_CLASSES):
        # Filter for this class
        pred_mask = pred_labels == class_idx
        gt_mask = gt_labels == class_idx

        class_pred_indices = np.where(pred_mask)[0]
        class_gt_indices = np.where(gt_mask)[0]

        if len(class_pred_indices) == 0 or len(class_gt_indices) == 0:
            errors[class_name] = {
                "ATE": 1.0,
                "ASE": 1.0,
                "AOE": np.pi,
                "AVE": 1.0,
                "AAE": 1.0,
            }
            continue

        # Sort by score descending
        class_scores = pred_scores[class_pred_indices]
        sort_order = np.argsort(-class_scores)
        sorted_pred_indices = class_pred_indices[sort_order]

        # Greedy matching
        gt_matched = np.zeros(len(class_gt_indices), dtype=bool)
        ate_list = []
        ase_list = []
        aoe_list = []
        ave_list = []
        aae_list = []

        for pred_idx in sorted_pred_indices:
            pred_box = pred_boxes[pred_idx]
            best_dist = float("inf")
            best_gt_local = -1

            for gt_local, gt_global in enumerate(class_gt_indices):
                if gt_matched[gt_local]:
                    continue
                dist = compute_center_distance(pred_box, gt_boxes[gt_global])
                if dist < best_dist:
                    best_dist = dist
                    best_gt_local = gt_local

            if best_gt_local >= 0 and best_dist <= distance_threshold:
                gt_matched[best_gt_local] = True
                gt_global_idx = class_gt_indices[best_gt_local]
                gt_box = gt_boxes[gt_global_idx]

                ate_list.append(compute_translation_error(pred_box, gt_box))
                ase_list.append(compute_scale_error(pred_box, gt_box))
                aoe_list.append(compute_orientation_error(pred_box, gt_box))

                if class_name not in VELOCITY_ERROR_CLASSES:
                    ave_list.append(compute_velocity_error(pred_box, gt_box))

                aae_list.append(
                    compute_attribute_error(
                        int(pred_labels[pred_idx]),
                        int(gt_labels[gt_global_idx]),
                        float(pred_scores[pred_idx]),
                    )
                )

        errors[class_name] = {
            "ATE": float(np.mean(ate_list)) if ate_list else 1.0,
            "ASE": float(np.mean(ase_list)) if ase_list else 1.0,
            "AOE": float(np.mean(aoe_list)) if aoe_list else np.pi,
            "AVE": float(np.mean(ave_list)) if ave_list else 1.0,
            "AAE": float(np.mean(aae_list)) if aae_list else 1.0,
        }

    return errors


def compute_nds(mAP: float, tp_errors: Dict[str, float]) -> float:
    """Compute nuScenes Detection Score (NDS).

    NDS = (1/10) * [5 * mAP + sum(1 - min(1, error_i))]

    where error_i are the 5 TP metrics normalized to [0, 1].

    Args:
        mAP: Mean Average Precision.
        tp_errors: Dictionary of mean TP error values
                   {mATE, mASE, mAOE, mAVE, mAAE}.

    Returns:
        NDS value in [0, 1].
    """
    # TP error normalization thresholds (cap errors at these values)
    error_caps = {
        "mATE": 1.0,   # meters
        "mASE": 1.0,   # ratio
        "mAOE": np.pi, # radians (normalized by pi below)
        "mAVE": 1.0,   # m/s
        "mAAE": 1.0,   # ratio
    }

    # Compute TP score: 1 - normalized error
    tp_scores = []
    for metric, cap in error_caps.items():
        error_val = tp_errors.get(metric, cap)
        # Normalize AOE by pi to bring to [0, 1]
        if metric == "mAOE":
            normalized = min(error_val / np.pi, 1.0)
        else:
            normalized = min(error_val / cap, 1.0)
        tp_scores.append(max(0.0, 1.0 - normalized))

    # NDS formula: weighted combination of mAP and TP scores
    nds = (5.0 * mAP + sum(tp_scores)) / 10.0
    return float(nds)


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================


class BEVFormerEvaluator:
    """Full evaluation pipeline for BEVFormer on nuScenes.

    Runs inference on the validation set, accumulates predictions, and computes
    all nuScenes detection metrics.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: str,
        data_root: str,
        output_dir: str,
        batch_size: int = 1,
    ):
        """Initialize evaluator.

        Args:
            config: Full configuration dictionary.
            checkpoint_path: Path to TensorFlow checkpoint.
            data_root: Path to nuScenes dataset root.
            output_dir: Directory to save evaluation results.
            batch_size: Batch size for inference.
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root
        self.output_dir = output_dir
        self.batch_size = batch_size

        self.model_cfg = config["model"]
        self.eval_cfg = config.get("eval", {})
        self.score_threshold = self.eval_cfg.get("score_threshold", 0.1)
        self.max_detections = self.eval_cfg.get("max_detections_per_sample", 300)

        # Point cloud range for box decoding
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        os.makedirs(output_dir, exist_ok=True)

        # Build model and load checkpoint
        self.model = self._build_and_load_model()

        # Data loader
        self.data_loader = NuScenesEvalDataLoader(config, data_root)

    def _build_and_load_model(self) -> tf.keras.Model:
        """Build BEVFormer model and load weights from checkpoint.

        Returns:
            Loaded model ready for inference.
        """
        logger.info("Building BEVFormer model...")

        # Import the model building function from model.py
        # Try to import from the same package
        try:
            from model import BEVFormer, build_bevformer, DEFAULT_CONFIG as MODEL_DEFAULT_CONFIG
        except ImportError:
            # Try relative import path
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from model import BEVFormer, build_bevformer, DEFAULT_CONFIG as MODEL_DEFAULT_CONFIG

        # Build model config from evaluation config
        model_config = MODEL_DEFAULT_CONFIG.copy()
        model_config.update({
            "bev_h": self.model_cfg.get("bev_h", 200),
            "bev_w": self.model_cfg.get("bev_w", 200),
            "embed_dims": self.model_cfg.get("embed_dims", 256),
            "num_encoder_layers": self.model_cfg.get("num_encoder_layers", 6),
            "num_decoder_layers": self.model_cfg.get("num_decoder_layers", 6),
            "num_heads": self.model_cfg.get("num_heads", 8),
            "num_queries": self.model_cfg.get("num_query", 900),
            "num_classes": self.model_cfg.get("num_classes", 10),
            "num_levels": self.model_cfg.get("num_levels", 4),
        })

        model = build_bevformer(model_config)

        # Build model with dummy input to initialize weights
        bev_h = model_config["bev_h"]
        bev_w = model_config["bev_w"]
        embed_dims = model_config["embed_dims"]
        input_h = self.config["data"].get("input_h", 480)
        input_w = self.config["data"].get("input_w", 800)

        dummy_inputs = {
            "images": tf.zeros([1, 6, input_h, input_w, 3]),
            "lidar2img": tf.zeros([1, 6, 4, 4]),
            "ego_motion": tf.eye(4, batch_shape=[1]),
            "prev_bev": None,
        }
        _ = model(dummy_inputs, training=False)

        # Load checkpoint
        logger.info(f"Loading checkpoint from: {self.checkpoint_path}")

        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(self.checkpoint_path)

        # Try expect_partial() for cases where optimizer state is missing
        try:
            status.expect_partial()
        except Exception as e:
            logger.warning(f"Checkpoint restore note: {e}")

        logger.info("Model loaded successfully.")
        return model

    def run_inference(self, images: np.ndarray, lidar2img: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a batch of samples.

        Args:
            images: (B, num_cameras, H, W, 3) multi-camera images.
            lidar2img: (B, num_cameras, 4, 4) calibration matrices.

        Returns:
            Dictionary with raw model outputs:
                - 'cls_logits': (B, num_queries, num_classes)
                - 'bbox_preds': (B, num_queries, code_size)
        """
        inputs = {
            "images": tf.constant(images, dtype=tf.float32),
            "lidar2img": tf.constant(lidar2img, dtype=tf.float32),
            "ego_motion": tf.eye(4, batch_shape=[images.shape[0]]),
            "prev_bev": None,
        }

        outputs = self.model(inputs, training=False)

        return {
            "cls_logits": outputs["cls_logits"].numpy(),
            "bbox_preds": outputs["bbox_preds"].numpy(),
        }

    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation on the validation set.

        Returns:
            Complete evaluation results dictionary.
        """
        logger.info("=" * 70)
        logger.info("Starting BEVFormer Evaluation")
        logger.info("=" * 70)

        # Load validation annotations
        infos = self.data_loader.load_annotations()
        num_samples = len(infos)
        logger.info(f"Evaluating on {num_samples} samples...")

        # Collect all predictions and ground truths
        all_predictions = []  # List of per-sample prediction dicts
        all_ground_truths = []  # List of per-sample GT dicts

        eval_start = time.time()
        batch_count = 0

        # Process samples in batches
        for batch_start in range(0, num_samples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_samples)
            batch_infos = infos[batch_start:batch_end]

            # Load batch data
            batch_images = []
            batch_lidar2img = []
            batch_gt = []

            for info in batch_infos:
                sample = self.data_loader.get_sample(info)
                batch_images.append(sample["images"])
                batch_lidar2img.append(sample["lidar2img"])
                batch_gt.append({
                    "boxes": sample["gt_boxes"],
                    "labels": sample["gt_labels"],
                    "names": sample["gt_names"],
                    "token": sample["token"],
                })

            images_batch = np.stack(batch_images, axis=0)
            lidar2img_batch = np.stack(batch_lidar2img, axis=0)

            # Run inference
            outputs = self.run_inference(images_batch, lidar2img_batch)

            # Decode predictions for each sample in batch
            for i in range(len(batch_infos)):
                preds = decode_predictions(
                    cls_logits=outputs["cls_logits"][i],
                    bbox_preds=outputs["bbox_preds"][i],
                    pc_range=self.pc_range,
                    score_threshold=self.score_threshold,
                    max_detections=self.max_detections,
                )
                preds["token"] = batch_gt[i]["token"]
                all_predictions.append(preds)
                all_ground_truths.append(batch_gt[i])

            batch_count += 1
            if batch_count % 100 == 0:
                elapsed = time.time() - eval_start
                speed = batch_end / elapsed
                eta = (num_samples - batch_end) / max(speed, 1e-6)
                logger.info(
                    f"  Progress: {batch_end}/{num_samples} samples "
                    f"({100.0 * batch_end / num_samples:.1f}%) "
                    f"Speed: {speed:.1f} samples/s, ETA: {eta:.0f}s"
                )

        eval_time = time.time() - eval_start
        logger.info(f"Inference complete: {num_samples} samples in {eval_time:.1f}s "
                    f"({num_samples / eval_time:.1f} samples/s)")

        # Compute metrics
        logger.info("Computing evaluation metrics...")
        results = self._compute_metrics(all_predictions, all_ground_truths)
        results["eval_time_seconds"] = eval_time
        results["num_samples"] = num_samples

        # Print and save results
        self._print_results(results)
        self._save_results(results)

        return results

    def _compute_metrics(
        self,
        predictions: List[Dict[str, np.ndarray]],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute all nuScenes detection metrics.

        Args:
            predictions: List of per-sample prediction dicts.
            ground_truths: List of per-sample GT dicts.

        Returns:
            Complete metrics dictionary.
        """
        num_samples = len(predictions)

        # Aggregate per-class AP at each distance threshold
        per_class_ap = {
            cls: {d: [] for d in DISTANCE_THRESHOLDS}
            for cls in NUSCENES_CLASSES
        }

        # Accumulate TP errors across all samples
        all_tp_errors = {
            cls: {"ATE": [], "ASE": [], "AOE": [], "AVE": [], "AAE": []}
            for cls in NUSCENES_CLASSES
        }

        # Accumulate predictions and GT across all samples for global AP calculation
        # Per-class accumulation across all samples
        global_tp_lists = {
            cls: {d: [] for d in DISTANCE_THRESHOLDS}
            for cls in NUSCENES_CLASSES
        }
        global_num_gt = {
            cls: 0 for cls in NUSCENES_CLASSES
        }

        for sample_idx in range(num_samples):
            pred = predictions[sample_idx]
            gt = ground_truths[sample_idx]

            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]

            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            # Count ground truth per class
            for class_idx, class_name in enumerate(NUSCENES_CLASSES):
                num_class_gt = np.sum(gt_labels == class_idx)
                global_num_gt[class_name] += int(num_class_gt)

            # Match predictions to GT at each distance threshold
            for dist_thresh in DISTANCE_THRESHOLDS:
                for class_idx, class_name in enumerate(NUSCENES_CLASSES):
                    tp_list, _ = match_predictions_to_gt(
                        pred_boxes, pred_scores, pred_labels,
                        gt_boxes, gt_labels,
                        distance_threshold=dist_thresh,
                        class_idx=class_idx,
                    )
                    global_tp_lists[class_name][dist_thresh].extend(tp_list)

            # Compute TP errors for this sample (at 2.0m threshold)
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                sample_errors = compute_tp_errors(
                    pred_boxes, pred_scores, pred_labels,
                    gt_boxes, gt_labels,
                    distance_threshold=2.0,
                )
                for cls_name, cls_errors in sample_errors.items():
                    for metric, value in cls_errors.items():
                        all_tp_errors[cls_name][metric].append(value)

        # Compute AP for each class at each threshold
        ap_results = {}
        for class_name in NUSCENES_CLASSES:
            ap_results[class_name] = {}
            for dist_thresh in DISTANCE_THRESHOLDS:
                tp_list = global_tp_lists[class_name][dist_thresh]
                num_gt = global_num_gt[class_name]
                ap = compute_ap(tp_list, num_gt)
                ap_results[class_name][dist_thresh] = ap

        # Compute mean AP across thresholds for each class
        per_class_map = {}
        for class_name in NUSCENES_CLASSES:
            aps = [ap_results[class_name][d] for d in DISTANCE_THRESHOLDS]
            per_class_map[class_name] = float(np.mean(aps))

        # Overall mAP
        mAP = float(np.mean(list(per_class_map.values())))

        # Compute mean TP errors across classes
        mean_tp_errors = {}
        for metric in ["ATE", "ASE", "AOE", "AVE", "AAE"]:
            class_means = []
            for class_name in NUSCENES_CLASSES:
                values = all_tp_errors[class_name][metric]
                if values:
                    class_means.append(float(np.mean(values)))
                else:
                    # Use maximum error if no TP matches
                    if metric == "AOE":
                        class_means.append(np.pi)
                    else:
                        class_means.append(1.0)
            mean_tp_errors[f"m{metric}"] = float(np.mean(class_means))

        # Compute NDS
        nds = compute_nds(mAP, mean_tp_errors)

        # Assemble results
        results = {
            "mAP": mAP,
            "NDS": nds,
            "mATE": mean_tp_errors["mATE"],
            "mASE": mean_tp_errors["mASE"],
            "mAOE": mean_tp_errors["mAOE"],
            "mAVE": mean_tp_errors["mAVE"],
            "mAAE": mean_tp_errors["mAAE"],
            "per_class_ap": per_class_map,
            "per_class_ap_per_threshold": {
                cls: {str(d): v for d, v in thresh_aps.items()}
                for cls, thresh_aps in ap_results.items()
            },
            "per_class_tp_errors": {
                cls: {
                    metric: float(np.mean(values)) if values else (
                        np.pi if metric == "AOE" else 1.0
                    )
                    for metric, values in errors.items()
                }
                for cls, errors in all_tp_errors.items()
            },
            "distance_thresholds": DISTANCE_THRESHOLDS,
            "score_threshold": self.score_threshold,
            "num_gt_per_class": global_num_gt,
        }

        return results

    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print evaluation results in formatted tables.

        Args:
            results: Complete evaluation results dictionary.
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("  EVALUATION RESULTS")
        logger.info("=" * 80)

        # Overall metrics
        logger.info("")
        logger.info("  Overall Metrics:")
        logger.info(f"  {'Metric':<12} {'Value':<12}")
        logger.info(f"  {'-' * 24}")
        logger.info(f"  {'mAP':<12} {results['mAP']:.4f}")
        logger.info(f"  {'NDS':<12} {results['NDS']:.4f}")
        logger.info(f"  {'mATE':<12} {results['mATE']:.4f}")
        logger.info(f"  {'mASE':<12} {results['mASE']:.4f}")
        logger.info(f"  {'mAOE':<12} {results['mAOE']:.4f}")
        logger.info(f"  {'mAVE':<12} {results['mAVE']:.4f}")
        logger.info(f"  {'mAAE':<12} {results['mAAE']:.4f}")

        # Per-class AP table
        logger.info("")
        logger.info("  Per-Class Average Precision (mAP across thresholds):")
        logger.info("")

        # Header
        header = f"  {'Class':<24}"
        for d in DISTANCE_THRESHOLDS:
            header += f"{'AP@' + str(d) + 'm':<10}"
        header += f"{'Mean AP':<10}"
        logger.info(header)
        logger.info(f"  {'-' * (24 + 10 * (len(DISTANCE_THRESHOLDS) + 1))}")

        per_class_ap = results["per_class_ap"]
        per_threshold = results["per_class_ap_per_threshold"]

        for class_name in NUSCENES_CLASSES:
            row = f"  {class_name:<24}"
            for d in DISTANCE_THRESHOLDS:
                val = per_threshold[class_name][str(d)]
                row += f"{val:.4f}    "
            row += f"{per_class_ap[class_name]:.4f}"
            logger.info(row)

        # Mean row
        logger.info(f"  {'-' * (24 + 10 * (len(DISTANCE_THRESHOLDS) + 1))}")
        mean_row = f"  {'MEAN':<24}"
        for d in DISTANCE_THRESHOLDS:
            vals = [per_threshold[cls][str(d)] for cls in NUSCENES_CLASSES]
            mean_row += f"{np.mean(vals):.4f}    "
        mean_row += f"{results['mAP']:.4f}"
        logger.info(mean_row)

        # Per-class TP errors
        logger.info("")
        logger.info("  Per-Class True Positive Errors:")
        logger.info("")
        tp_header = f"  {'Class':<24}{'ATE':<8}{'ASE':<8}{'AOE':<8}{'AVE':<8}{'AAE':<8}"
        logger.info(tp_header)
        logger.info(f"  {'-' * 64}")

        per_class_errors = results["per_class_tp_errors"]
        for class_name in NUSCENES_CLASSES:
            errs = per_class_errors[class_name]
            row = f"  {class_name:<24}"
            row += f"{errs['ATE']:.4f}  "
            row += f"{errs['ASE']:.4f}  "
            row += f"{errs['AOE']:.4f}  "
            row += f"{errs['AVE']:.4f}  "
            row += f"{errs['AAE']:.4f}"
            logger.info(row)

        logger.info(f"  {'-' * 64}")
        mean_err_row = f"  {'MEAN':<24}"
        mean_err_row += f"{results['mATE']:.4f}  "
        mean_err_row += f"{results['mASE']:.4f}  "
        mean_err_row += f"{results['mAOE']:.4f}  "
        mean_err_row += f"{results['mAVE']:.4f}  "
        mean_err_row += f"{results['mAAE']:.4f}"
        logger.info(mean_err_row)

        logger.info("")
        logger.info(f"  Evaluation time: {results.get('eval_time_seconds', 0):.1f}s")
        logger.info(f"  Number of samples: {results.get('num_samples', 0)}")
        logger.info("=" * 80)

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to JSON file.

        Args:
            results: Complete evaluation results dictionary.
        """
        # Convert numpy types to Python native for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            return obj

        results_serializable = convert_for_json(results)

        output_path = os.path.join(self.output_dir, "eval_results.json")
        with open(output_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

        # Also save a concise summary
        summary = {
            "mAP": results["mAP"],
            "NDS": results["NDS"],
            "mATE": results["mATE"],
            "mASE": results["mASE"],
            "mAOE": results["mAOE"],
            "mAVE": results["mAVE"],
            "mAAE": results["mAAE"],
            "per_class_ap": results["per_class_ap"],
        }
        summary_path = os.path.join(self.output_dir, "eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(convert_for_json(summary), f, indent=2)

        logger.info(f"Summary saved to: {summary_path}")


# =============================================================================
# Command Line Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="BEVFormer Evaluation Script (TensorFlow) - nuScenes Detection Metrics",
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
        help="Path to TensorFlow checkpoint to evaluate.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/nuscenes",
        help="Path to nuScenes dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")

    return args


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override data root if specified
    if args.data_root:
        config["data"]["dataset_root"] = args.data_root

    # Log system information
    logger.info("=" * 70)
    logger.info("BEVFormer Evaluation (TensorFlow)")
    logger.info("=" * 70)
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    # Enable memory growth for GPUs
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # Run evaluation
    try:
        evaluator = BEVFormerEvaluator(
            config=config,
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        results = evaluator.evaluate()

        # Print final summary
        logger.info("")
        logger.info("Final Results:")
        logger.info(f"  mAP = {results['mAP']:.4f}")
        logger.info(f"  NDS = {results['NDS']:.4f}")

    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
