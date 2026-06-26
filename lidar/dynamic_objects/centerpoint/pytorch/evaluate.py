"""
CenterPoint 3D Object Detection and Tracking - Evaluation Script

Complete evaluation pipeline implementing nuScenes-style detection metrics and
multi-object tracking metrics from scratch.

Detection Metrics:
    - mAP: Mean Average Precision using BEV center distance matching
      (thresholds: 0.5, 1.0, 2.0, 4.0 meters)
    - True Positive metrics: ATE, ASE, AOE, AVE, AAE
    - NDS: nuScenes Detection Score

Tracking Metrics (optional):
    - AMOTA: Average Multi-Object Tracking Accuracy
    - AMOTP: Average Multi-Object Tracking Precision
    - IDS: Identity Switches
    - FRAG: Track Fragmentations

Usage:
    python -m lidar.dynamic_objects.centerpoint.pytorch.evaluate \
        --checkpoint path/to/checkpoint.pth \
        --config path/to/config.yaml \
        --data-path path/to/nuscenes/ \
        --output-path path/to/results/ \
        --eval-tracking
"""

import argparse
import collections
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from .model import build_model_from_config
from .dataset import NuScenesDataset, collate_fn
from .tracker import CenterPointTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="CenterPoint Detection and Tracking Evaluation"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--output-path", type=str, default="./eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--eval-tracking", action="store_true", default=False,
        help="Enable tracking evaluation (runs tracker across sequences)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.1,
        help="Minimum detection score threshold"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False,
        help="Disable CUDA even if available"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Build model from config and load checkpoint weights.

    Args:
        config: Model configuration dictionary.
        checkpoint_path: Path to the saved checkpoint.
        device: Device to load model onto.

    Returns:
        Model in eval mode with loaded weights.
    """
    model = build_model_from_config(config["model"])

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Handle DDP-wrapped state dicts (keys prefixed with "module.")
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model with {num_params / 1e6:.2f}M parameters from {checkpoint_path}")

    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.1,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    """Run model inference on entire validation set.

    Args:
        model: CenterPoint model in eval mode.
        dataloader: Validation dataloader.
        device: Compute device.
        score_threshold: Minimum score for keeping detections.

    Returns:
        Tuple of (all_predictions, all_metadata) where:
            all_predictions: List of dicts per sample with keys:
                'boxes': (K, 9) numpy array [x, y, z, w, h, l, yaw, vx, vy]
                'scores': (K,) numpy array
                'labels': (K,) numpy integer array
            all_metadata: List of metadata dicts per sample.
    """
    all_predictions = []
    all_metadata = []

    total_samples = 0
    start_time = time.time()

    for batch_idx, batch_data in enumerate(dataloader):
        voxels = batch_data["voxels"].to(device, non_blocking=True)
        coordinates = batch_data["coordinates"].to(device, non_blocking=True)
        num_points_per_voxel = batch_data["num_points_per_voxel"].to(device, non_blocking=True)
        batch_size = len(batch_data["metadata"])

        with autocast(enabled=(device.type == "cuda")):
            predictions = model(
                voxels=voxels,
                coordinates=coordinates,
                num_points_per_voxel=num_points_per_voxel,
                batch_size=batch_size,
            )

        # Decode predictions from heatmaps to boxes
        if hasattr(model, "decode"):
            decoded = model.decode(predictions)
        elif hasattr(model, "module") and hasattr(model.module, "decode"):
            decoded = model.module.decode(predictions)
        else:
            # Fallback: assume predictions are already in list-of-dicts form
            decoded = predictions

        # Convert to numpy and filter by score
        for i, det in enumerate(decoded):
            if isinstance(det["boxes"], torch.Tensor):
                boxes = det["boxes"].cpu().numpy()
                scores = det["scores"].cpu().numpy()
                labels = det["labels"].cpu().numpy().astype(np.int32)
            else:
                boxes = np.asarray(det["boxes"])
                scores = np.asarray(det["scores"])
                labels = np.asarray(det["labels"], dtype=np.int32)

            # Apply score threshold
            keep_mask = scores >= score_threshold
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            labels = labels[keep_mask]

            all_predictions.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })

        all_metadata.extend(batch_data["metadata"])
        total_samples += batch_size

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Inference: {total_samples}/{len(dataloader.dataset)} samples, "
                f"{total_samples / elapsed:.1f} samples/sec"
            )

    elapsed = time.time() - start_time
    logger.info(
        f"Inference complete: {total_samples} samples in {elapsed:.1f}s "
        f"({total_samples / elapsed:.1f} samples/sec)"
    )

    return all_predictions, all_metadata


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def load_ground_truth(dataloader: DataLoader) -> List[Dict[str, np.ndarray]]:
    """Extract ground truth annotations from the dataset.

    Iterates through the dataset's stored info to retrieve GT boxes, class labels,
    velocities, and attributes without re-running the full __getitem__ logic.

    Args:
        dataloader: Validation dataloader.

    Returns:
        List of ground truth dicts per sample with keys:
            'boxes': (N, 9) [x, y, z, w, h, l, yaw, vx, vy]
            'labels': (N,) integer class indices
            'tracking_ids': (N,) integer instance IDs (if available)
            'attributes': (N,) integer attribute indices (if available)
    """
    dataset = dataloader.dataset
    all_gt = []

    class_names = dataset.class_names
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for idx in range(len(dataset)):
        info = dataset.infos[idx]

        gt_boxes = np.array(info.get("gt_boxes", np.zeros((0, 9))), dtype=np.float32)
        gt_names = info.get("gt_names", [])

        # Ensure 9 columns
        if gt_boxes.ndim == 1 and gt_boxes.shape[0] == 0:
            gt_boxes = np.zeros((0, 9), dtype=np.float32)
        elif gt_boxes.ndim == 2 and gt_boxes.shape[1] < 9:
            padding = np.zeros((gt_boxes.shape[0], 9 - gt_boxes.shape[1]), dtype=np.float32)
            gt_boxes = np.hstack([gt_boxes, padding])

        # Compute class labels
        gt_labels = np.array(
            [class_to_idx.get(name, -1) for name in gt_names], dtype=np.int32
        )

        # Filter to valid classes only
        valid_mask = gt_labels >= 0
        gt_boxes = gt_boxes[valid_mask]
        gt_labels = gt_labels[valid_mask]

        # Tracking IDs
        tracking_ids = None
        if "tracking_ids" in info:
            tracking_ids = np.array(info["tracking_ids"], dtype=np.int64)
            if len(tracking_ids) > 0:
                tracking_ids = tracking_ids[valid_mask]
            else:
                tracking_ids = np.zeros(len(gt_labels), dtype=np.int64)
        else:
            tracking_ids = np.zeros(len(gt_labels), dtype=np.int64)

        # Attributes (for AAE computation)
        attributes = None
        if "gt_attributes" in info:
            attributes = np.array(info["gt_attributes"], dtype=np.int32)
            if len(attributes) > 0:
                attributes = attributes[valid_mask]
            else:
                attributes = np.zeros(len(gt_labels), dtype=np.int32)
        else:
            attributes = np.zeros(len(gt_labels), dtype=np.int32)

        all_gt.append({
            "boxes": gt_boxes,
            "labels": gt_labels,
            "tracking_ids": tracking_ids,
            "attributes": attributes,
        })

    return all_gt


# ---------------------------------------------------------------------------
# Detection Evaluation: nuScenes-style mAP and TP metrics
# ---------------------------------------------------------------------------


def compute_center_distance(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute pairwise BEV center distances between predictions and ground truth.

    Args:
        pred_boxes: (M, 9) predicted bounding boxes.
        gt_boxes: (N, 9) ground truth bounding boxes.

    Returns:
        (M, N) distance matrix in BEV.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)

    pred_centers = pred_boxes[:, :2]  # (M, 2)
    gt_centers = gt_boxes[:, :2]  # (N, 2)

    # (M, 1, 2) - (1, N, 2) -> (M, N, 2)
    diff = pred_centers[:, np.newaxis, :] - gt_centers[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)  # (M, N)
    return distances


def compute_ap(
    scores: np.ndarray,
    tp_flags: np.ndarray,
    num_gt: int,
    min_recall: float = 0.1,
    min_precision: float = 0.1,
) -> float:
    """Compute Average Precision using nuScenes-style interpolation.

    The nuScenes AP computation:
    1. Sort detections by score descending.
    2. Compute precision-recall curve.
    3. Compute AP as the area under the precision-recall curve, but only
       considering recall values >= min_recall. The curve is interpolated
       using the maximum precision at each recall level.

    Args:
        scores: (K,) detection confidence scores.
        tp_flags: (K,) binary true-positive indicators (1=TP, 0=FP).
        num_gt: Total number of ground truth instances.
        min_recall: Minimum recall for AP computation (nuScenes uses 0.1).
        min_precision: Minimum precision for AP computation (nuScenes uses 0.1).

    Returns:
        Scalar AP value.
    """
    if num_gt == 0:
        return 0.0

    if len(scores) == 0:
        return 0.0

    # Sort by score descending
    sort_idx = np.argsort(-scores)
    tp_flags = tp_flags[sort_idx]

    # Cumulative TP and FP
    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum(1 - tp_flags)

    # Precision and recall curves
    recall = tp_cumsum / num_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Ensure starting point
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    # nuScenes-style: sample at 101 recall points
    recall_interp = np.linspace(0, 1, 101)
    precision_interp = np.zeros_like(recall_interp)

    for i, r in enumerate(recall_interp):
        # Maximum precision at recall >= r
        mask = recall >= r
        if mask.any():
            precision_interp[i] = precision[mask].max()
        else:
            precision_interp[i] = 0.0

    # Filter to valid range
    valid = (recall_interp >= min_recall) & (precision_interp >= min_precision)

    if not valid.any():
        return 0.0

    # AP is the mean precision in the valid region, normalized by the recall range
    # nuScenes: mean of precision values at sampled recall points above thresholds
    ap = precision_interp[valid].mean()

    # Scale by the fraction of recall range that is valid
    valid_recall_range = recall_interp[valid]
    if len(valid_recall_range) > 0:
        recall_range = valid_recall_range[-1] - valid_recall_range[0]
        # Normalized to full [0,1] range to match nuScenes
        ap = np.trapz(precision_interp[recall_interp >= min_recall],
                      recall_interp[recall_interp >= min_recall])
    else:
        ap = 0.0

    return float(ap)


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    distance_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Match predictions to ground truth using greedy center distance matching.

    Predictions are processed in order of decreasing score. Each prediction is
    matched to the nearest unmatched ground truth within the distance threshold.

    Args:
        pred_boxes: (M, 9) predicted boxes.
        pred_scores: (M,) scores.
        gt_boxes: (N, 9) ground truth boxes.
        distance_threshold: Maximum BEV center distance for a valid match.

    Returns:
        tp_flags: (M,) binary array, 1 for true positives.
        match_distances: (M,) BEV distance to matched GT (-1 if FP).
        gt_match_indices: (M,) index of matched GT for each prediction (-1 if FP).
    """
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)

    tp_flags = np.zeros(num_pred, dtype=np.float64)
    match_distances = np.full(num_pred, -1.0, dtype=np.float64)
    gt_match_indices = np.full(num_pred, -1, dtype=np.int64)

    if num_pred == 0 or num_gt == 0:
        return tp_flags, match_distances, gt_match_indices

    # Sort predictions by score descending
    sort_idx = np.argsort(-pred_scores)

    # Compute distance matrix
    dist_matrix = compute_center_distance(pred_boxes, gt_boxes)  # (M, N)

    matched_gt = set()

    for rank, pred_idx in enumerate(sort_idx):
        if num_gt == 0:
            break

        distances = dist_matrix[pred_idx]  # (N,)

        # Mask already-matched GTs
        for mg in matched_gt:
            distances[mg] = np.inf

        min_gt_idx = np.argmin(distances)
        min_dist = distances[min_gt_idx]

        if min_dist <= distance_threshold:
            tp_flags[pred_idx] = 1.0
            match_distances[pred_idx] = min_dist
            gt_match_indices[pred_idx] = min_gt_idx
            matched_gt.add(min_gt_idx)

    return tp_flags, match_distances, gt_match_indices


def compute_tp_metrics(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    gt_attributes: np.ndarray,
    pred_attributes: Optional[np.ndarray],
    tp_flags: np.ndarray,
    gt_match_indices: np.ndarray,
) -> Dict[str, float]:
    """Compute True Positive metrics for matched detections.

    Metrics computed only for TP detections (where tp_flags == 1):
    - ATE: Average Translation Error (Euclidean center distance in BEV)
    - ASE: Average Scale Error (1 - IoU of 3D volumes, approximated as
            1 - min(vol_pred, vol_gt) / max(vol_pred, vol_gt))
    - AOE: Average Orientation Error (smallest angle between pred and GT yaw)
    - AVE: Average Velocity Error (L2 norm of velocity difference)
    - AAE: Average Attribute Error (1 - accuracy of attribute classification)

    Args:
        pred_boxes: (M, 9) predicted boxes [x, y, z, w, h, l, yaw, vx, vy].
        gt_boxes: (N, 9) ground truth boxes.
        pred_labels: (M,) predicted class labels.
        gt_labels: (N,) ground truth class labels.
        gt_attributes: (N,) ground truth attribute indices.
        pred_attributes: (M,) predicted attribute indices (None if unavailable).
        tp_flags: (M,) binary TP indicators.
        gt_match_indices: (M,) matched GT index per prediction.

    Returns:
        Dict with ATE, ASE, AOE, AVE, AAE values.
    """
    tp_mask = tp_flags == 1.0
    if not tp_mask.any():
        return {"ATE": 1.0, "ASE": 1.0, "AOE": 1.0, "AVE": 1.0, "AAE": 1.0}

    tp_pred_boxes = pred_boxes[tp_mask]
    tp_gt_indices = gt_match_indices[tp_mask].astype(np.int64)
    tp_gt_boxes = gt_boxes[tp_gt_indices]

    # ATE: Euclidean center distance in BEV (x, y)
    translation_errors = np.linalg.norm(
        tp_pred_boxes[:, :2] - tp_gt_boxes[:, :2], axis=1
    )
    ate = float(np.mean(translation_errors))

    # ASE: Scale error using 3D IoU approximation
    # Volume = w * h * l (indices 3, 4, 5 in box representation)
    pred_volumes = tp_pred_boxes[:, 3] * tp_pred_boxes[:, 4] * tp_pred_boxes[:, 5]
    gt_volumes = tp_gt_boxes[:, 3] * tp_gt_boxes[:, 4] * tp_gt_boxes[:, 5]

    # Clamp volumes to avoid division by zero
    pred_volumes = np.clip(pred_volumes, 1e-6, None)
    gt_volumes = np.clip(gt_volumes, 1e-6, None)

    # Scale error: 1 - IOU_3D approximation using volume ratio
    # Approximate 3D IoU as min(vol) / max(vol) (assumes aligned boxes)
    min_vol = np.minimum(pred_volumes, gt_volumes)
    max_vol = np.maximum(pred_volumes, gt_volumes)
    iou_approx = min_vol / max_vol
    scale_errors = 1.0 - iou_approx
    ase = float(np.mean(scale_errors))

    # AOE: Orientation error (smallest angular difference)
    pred_yaw = tp_pred_boxes[:, 6]
    gt_yaw = tp_gt_boxes[:, 6]
    yaw_diff = pred_yaw - gt_yaw
    # Normalize to [-pi, pi]
    yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
    # Take absolute value (direction-agnostic for some classes)
    orientation_errors = np.abs(yaw_diff)
    aoe = float(np.mean(orientation_errors))

    # AVE: Velocity error (L2 norm of velocity difference)
    pred_vel = tp_pred_boxes[:, 7:9]
    gt_vel = tp_gt_boxes[:, 7:9]
    velocity_errors = np.linalg.norm(pred_vel - gt_vel, axis=1)
    ave = float(np.mean(velocity_errors))

    # AAE: Attribute error (1 - attribute classification accuracy)
    if pred_attributes is not None and gt_attributes is not None:
        tp_pred_attrs = pred_attributes[tp_mask]
        tp_gt_attrs = gt_attributes[tp_gt_indices]
        attr_correct = (tp_pred_attrs == tp_gt_attrs).astype(np.float64)
        aae = 1.0 - float(np.mean(attr_correct))
    else:
        # If attributes are not available, assign neutral error
        aae = 1.0

    return {"ATE": ate, "ASE": ase, "AOE": aoe, "AVE": ave, "AAE": aae}


def evaluate_detection(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    class_names: List[str],
    distance_thresholds: List[float] = None,
) -> Dict[str, Any]:
    """Run full nuScenes-style detection evaluation.

    Computes:
    - Per-class AP at each distance threshold
    - mAP (mean over classes and thresholds)
    - TP metrics (ATE, ASE, AOE, AVE, AAE) at the loosest threshold (4m)
    - NDS = 1/10 * [5*mAP + sum(max(1 - TP_err, 0.0))]

    Args:
        predictions: List of prediction dicts per sample.
        ground_truths: List of ground truth dicts per sample.
        class_names: Ordered list of class names.
        distance_thresholds: BEV center distance thresholds for matching.

    Returns:
        Comprehensive results dict with mAP, NDS, per-class metrics, and TP metrics.
    """
    if distance_thresholds is None:
        distance_thresholds = [0.5, 1.0, 2.0, 4.0]

    num_classes = len(class_names)
    num_thresholds = len(distance_thresholds)

    # Per-class, per-threshold AP storage
    ap_table = np.zeros((num_classes, num_thresholds), dtype=np.float64)

    # TP metric accumulators (computed at the largest threshold for nuScenes)
    tp_metric_threshold = max(distance_thresholds)
    per_class_tp_metrics = {}

    for cls_idx, cls_name in enumerate(class_names):
        # Gather all predictions and GT for this class across all samples
        all_scores = []
        all_tp_flags_per_thresh = {dt: [] for dt in distance_thresholds}
        total_gt = 0

        # For TP metrics at largest threshold
        all_pred_boxes_tp = []
        all_gt_boxes_tp = []
        all_tp_flags_largest = []
        all_gt_match_idx_largest = []
        all_gt_attributes_tp = []
        gt_offset = 0  # running offset for GT indices across samples

        for sample_idx in range(len(predictions)):
            pred = predictions[sample_idx]
            gt = ground_truths[sample_idx]

            # Filter predictions and GT for this class
            pred_cls_mask = pred["labels"] == cls_idx
            gt_cls_mask = gt["labels"] == cls_idx

            pred_boxes_cls = pred["boxes"][pred_cls_mask]
            pred_scores_cls = pred["scores"][pred_cls_mask]
            gt_boxes_cls = gt["boxes"][gt_cls_mask]
            gt_attrs_cls = gt["attributes"][gt_cls_mask]

            num_gt_cls = len(gt_boxes_cls)
            total_gt += num_gt_cls

            if len(pred_boxes_cls) == 0:
                gt_offset += num_gt_cls
                continue

            # For each distance threshold, compute matches
            for dt_idx, dist_thresh in enumerate(distance_thresholds):
                tp_flags, _, _ = match_predictions_to_gt(
                    pred_boxes_cls, pred_scores_cls, gt_boxes_cls, dist_thresh
                )
                all_tp_flags_per_thresh[dist_thresh].append(
                    (pred_scores_cls.copy(), tp_flags)
                )

            # TP metrics at largest threshold
            tp_flags_lg, match_dists, gt_match_idx = match_predictions_to_gt(
                pred_boxes_cls, pred_scores_cls, gt_boxes_cls, tp_metric_threshold
            )
            all_pred_boxes_tp.append(pred_boxes_cls)
            all_tp_flags_largest.append(tp_flags_lg)
            all_gt_match_idx_largest.append(gt_match_idx + gt_offset)
            all_gt_boxes_tp.append(gt_boxes_cls)
            all_gt_attributes_tp.append(gt_attrs_cls)

            gt_offset += num_gt_cls

        # Compute AP for each threshold
        for dt_idx, dist_thresh in enumerate(distance_thresholds):
            entries = all_tp_flags_per_thresh[dist_thresh]
            if not entries:
                ap_table[cls_idx, dt_idx] = 0.0
                continue

            all_scores_arr = np.concatenate([e[0] for e in entries])
            all_tp_arr = np.concatenate([e[1] for e in entries])

            ap = compute_ap(all_scores_arr, all_tp_arr, total_gt)
            ap_table[cls_idx, dt_idx] = ap

        # Compute TP metrics for this class
        if all_pred_boxes_tp:
            concat_pred_boxes = np.concatenate(all_pred_boxes_tp, axis=0)
            concat_tp_flags = np.concatenate(all_tp_flags_largest)
            concat_gt_match_idx = np.concatenate(all_gt_match_idx_largest)
            concat_gt_boxes = np.concatenate(all_gt_boxes_tp, axis=0) if all_gt_boxes_tp else np.zeros((0, 9))
            concat_gt_attrs = np.concatenate(all_gt_attributes_tp) if all_gt_attributes_tp else np.zeros(0, dtype=np.int32)

            tp_metrics = compute_tp_metrics(
                pred_boxes=concat_pred_boxes,
                gt_boxes=concat_gt_boxes,
                pred_labels=np.full(len(concat_pred_boxes), cls_idx, dtype=np.int32),
                gt_labels=np.full(len(concat_gt_boxes), cls_idx, dtype=np.int32),
                gt_attributes=concat_gt_attrs,
                pred_attributes=None,  # Attributes from predictions not typically available
                tp_flags=concat_tp_flags,
                gt_match_indices=concat_gt_match_idx,
            )
        else:
            tp_metrics = {"ATE": 1.0, "ASE": 1.0, "AOE": 1.0, "AVE": 1.0, "AAE": 1.0}

        per_class_tp_metrics[cls_name] = tp_metrics

    # Compute summary metrics
    per_class_ap = np.mean(ap_table, axis=1)  # Average over thresholds
    mAP = float(np.mean(per_class_ap))

    # Average TP metrics across classes
    avg_tp = {"ATE": 0.0, "ASE": 0.0, "AOE": 0.0, "AVE": 0.0, "AAE": 0.0}
    for cls_name in class_names:
        for metric in avg_tp:
            avg_tp[metric] += per_class_tp_metrics[cls_name][metric]
    for metric in avg_tp:
        avg_tp[metric] /= max(num_classes, 1)

    # NDS = 1/10 * [5 * mAP + sum(max(1 - TP_err, 0.0))]
    tp_score_sum = 0.0
    for metric_name in ["ATE", "ASE", "AOE", "AVE", "AAE"]:
        tp_score_sum += max(1.0 - avg_tp[metric_name], 0.0)
    nds = (5.0 * mAP + tp_score_sum) / 10.0

    # Build results dict
    results = {
        "mAP": mAP,
        "NDS": nds,
        "mean_tp_metrics": avg_tp,
        "per_class_ap": {
            cls_name: float(per_class_ap[i])
            for i, cls_name in enumerate(class_names)
        },
        "per_class_tp_metrics": per_class_tp_metrics,
        "ap_table": {
            cls_name: {
                f"{dt:.1f}m": float(ap_table[i, j])
                for j, dt in enumerate(distance_thresholds)
            }
            for i, cls_name in enumerate(class_names)
        },
        "distance_thresholds": distance_thresholds,
    }

    return results


# ---------------------------------------------------------------------------
# Tracking Evaluation: AMOTA, AMOTP, IDS, FRAG
# ---------------------------------------------------------------------------


def run_tracking(
    predictions: List[Dict[str, np.ndarray]],
    metadata: List[Dict[str, Any]],
    tracker_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run CenterPointTracker across sequences to produce tracking results.

    Groups frames by sequence_id and processes them in order of frame_id.
    Each sequence gets a fresh tracker instance.

    Args:
        predictions: List of per-frame prediction dicts.
        metadata: List of per-frame metadata dicts with sequence_id and frame_id.
        tracker_config: Optional config for tracker (max_age, min_hits, distance_threshold).

    Returns:
        List of per-frame tracking result dicts with keys:
            'track_ids': (K,) integer array of assigned track IDs.
            'boxes': (K, 9) tracked boxes.
            'scores': (K,) tracked scores.
            'labels': (K,) tracked class labels.
            'sequence_id': str
            'frame_id': int
    """
    if tracker_config is None:
        tracker_config = {"max_age": 3, "min_hits": 1, "distance_threshold": 2.0}

    # Group frame indices by sequence
    sequence_frames = collections.defaultdict(list)
    for idx, meta in enumerate(metadata):
        seq_id = meta.get("sequence_id", "default_sequence")
        frame_id = meta.get("frame_id", idx)
        sequence_frames[seq_id].append((frame_id, idx))

    # Sort frames within each sequence by frame_id
    for seq_id in sequence_frames:
        sequence_frames[seq_id].sort(key=lambda x: x[0])

    all_tracking_results = [None] * len(predictions)

    for seq_id, frames in sequence_frames.items():
        tracker = CenterPointTracker(
            max_age=tracker_config.get("max_age", 3),
            min_hits=tracker_config.get("min_hits", 1),
            distance_threshold=tracker_config.get("distance_threshold", 2.0),
        )

        for frame_id, global_idx in frames:
            pred = predictions[global_idx]
            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]

            # Predict step (extrapolate existing tracks)
            tracker.predict()

            # Prepare detections as list of (9,) arrays
            detections = [boxes[i] for i in range(len(boxes))] if len(boxes) > 0 else []
            det_scores = scores.tolist() if len(scores) > 0 else []
            det_class_ids = labels.tolist() if len(labels) > 0 else []

            # Update tracker with new detections
            tracker.update(detections, scores=det_scores, class_ids=det_class_ids)

            # Get active tracks
            active_tracks = tracker.get_active_tracks()

            # Assemble tracking output
            track_ids = np.array([t.track_id for t in active_tracks], dtype=np.int64)
            track_boxes = np.array([t.last_box for t in active_tracks], dtype=np.float64) if active_tracks else np.zeros((0, 9))
            track_scores = np.array([t.score for t in active_tracks], dtype=np.float64)
            track_labels = np.array([t.class_id for t in active_tracks], dtype=np.int32)

            all_tracking_results[global_idx] = {
                "track_ids": track_ids,
                "boxes": track_boxes,
                "scores": track_scores,
                "labels": track_labels,
                "sequence_id": seq_id,
                "frame_id": frame_id,
            }

    return all_tracking_results


def evaluate_tracking(
    tracking_results: List[Dict[str, Any]],
    ground_truths: List[Dict[str, np.ndarray]],
    metadata: List[Dict[str, Any]],
    class_names: List[str],
    distance_threshold: float = 2.0,
    num_recall_points: int = 40,
) -> Dict[str, Any]:
    """Compute multi-object tracking metrics.

    Implements AMOTA/AMOTP as defined in the nuScenes tracking benchmark:
    - For each recall threshold r in [1/N, 2/N, ..., 1.0]:
      - Keep only tracks/detections with score >= threshold that achieves recall r
      - Compute MOTA at that recall level
    - AMOTA = mean(MOTA over recall thresholds)
    - AMOTP = mean(MOTP over recall thresholds)

    Also computes:
    - IDS: Total identity switches across all sequences
    - FRAG: Total track fragmentations across all sequences

    Args:
        tracking_results: List of per-frame tracking dicts from run_tracking().
        ground_truths: List of per-frame ground truth dicts.
        metadata: List of per-frame metadata dicts.
        class_names: List of class names.
        distance_threshold: BEV center distance threshold for track-GT matching.
        num_recall_points: Number of recall thresholds for AMOTA computation.

    Returns:
        Dict with AMOTA, AMOTP, IDS, FRAG and per-class breakdowns.
    """
    num_classes = len(class_names)

    per_class_tracking_metrics = {}

    for cls_idx, cls_name in enumerate(class_names):
        # Group frames by sequence
        sequence_frames = collections.defaultdict(list)
        for idx, meta in enumerate(metadata):
            seq_id = meta.get("sequence_id", "default_sequence")
            frame_id = meta.get("frame_id", idx)
            sequence_frames[seq_id].append((frame_id, idx))

        for seq_id in sequence_frames:
            sequence_frames[seq_id].sort(key=lambda x: x[0])

        # Collect all track scores for this class to determine score thresholds
        all_track_scores = []
        total_gt_objects = 0

        for idx in range(len(tracking_results)):
            tr = tracking_results[idx]
            gt = ground_truths[idx]

            if tr is None:
                continue

            track_cls_mask = tr["labels"] == cls_idx
            gt_cls_mask = gt["labels"] == cls_idx

            all_track_scores.extend(tr["scores"][track_cls_mask].tolist())
            total_gt_objects += int(gt_cls_mask.sum())

        if total_gt_objects == 0:
            per_class_tracking_metrics[cls_name] = {
                "AMOTA": 0.0, "AMOTP": 0.0, "IDS": 0, "FRAG": 0,
            }
            continue

        # Determine score thresholds that achieve each recall level
        all_track_scores = np.array(all_track_scores)
        if len(all_track_scores) == 0:
            per_class_tracking_metrics[cls_name] = {
                "AMOTA": 0.0, "AMOTP": 0.0, "IDS": 0, "FRAG": 0,
            }
            continue

        # Sort scores descending to find thresholds
        sorted_scores = np.sort(all_track_scores)[::-1]

        # Compute MOTA/MOTP at multiple recall thresholds
        recall_thresholds = np.linspace(
            1.0 / num_recall_points, 1.0, num_recall_points
        )

        mota_values = []
        motp_values = []
        total_ids = 0
        total_frag = 0

        for recall_target in recall_thresholds:
            # Determine the score threshold that achieves this recall
            # recall = num_detections_above_threshold / total_gt
            # We want num_detections_above_threshold = recall_target * total_gt
            desired_num_dets = int(np.ceil(recall_target * total_gt_objects))

            if desired_num_dets > len(sorted_scores):
                score_thresh = 0.0
            elif desired_num_dets <= 0:
                score_thresh = sorted_scores[0] + 1.0  # impossibly high
            else:
                score_thresh = sorted_scores[min(desired_num_dets - 1, len(sorted_scores) - 1)]

            # Run MOTA computation at this score threshold
            mota, motp, ids, frag = _compute_mota_at_threshold(
                tracking_results=tracking_results,
                ground_truths=ground_truths,
                sequence_frames=sequence_frames,
                cls_idx=cls_idx,
                score_threshold=score_thresh,
                distance_threshold=distance_threshold,
            )

            mota_values.append(mota)
            motp_values.append(motp)
            total_ids += ids
            total_frag += frag

        # AMOTA = mean of clipped MOTA values over recall thresholds
        amota = float(np.mean(np.clip(mota_values, 0.0, 1.0)))
        # AMOTP = mean of MOTP values (only where MOTP is defined)
        valid_motp = [m for m in motp_values if m < np.inf]
        amotp = float(np.mean(valid_motp)) if valid_motp else float("inf")

        # IDS and FRAG: total across all thresholds averaged
        avg_ids = total_ids // max(num_recall_points, 1)
        avg_frag = total_frag // max(num_recall_points, 1)

        per_class_tracking_metrics[cls_name] = {
            "AMOTA": amota,
            "AMOTP": amotp,
            "IDS": avg_ids,
            "FRAG": avg_frag,
        }

    # Compute overall metrics as mean over classes
    overall_amota = float(np.mean([m["AMOTA"] for m in per_class_tracking_metrics.values()]))
    valid_amotps = [m["AMOTP"] for m in per_class_tracking_metrics.values() if m["AMOTP"] < np.inf]
    overall_amotp = float(np.mean(valid_amotps)) if valid_amotps else float("inf")
    overall_ids = sum(m["IDS"] for m in per_class_tracking_metrics.values())
    overall_frag = sum(m["FRAG"] for m in per_class_tracking_metrics.values())

    results = {
        "AMOTA": overall_amota,
        "AMOTP": overall_amotp,
        "IDS": overall_ids,
        "FRAG": overall_frag,
        "per_class": per_class_tracking_metrics,
    }

    return results


def _compute_mota_at_threshold(
    tracking_results: List[Dict[str, Any]],
    ground_truths: List[Dict[str, np.ndarray]],
    sequence_frames: Dict[str, List[Tuple[int, int]]],
    cls_idx: int,
    score_threshold: float,
    distance_threshold: float,
) -> Tuple[float, float, int, int]:
    """Compute MOTA, MOTP, IDS, and FRAG for a single class at a given score threshold.

    MOTA = 1 - (FP + FN + IDS) / total_GT
    MOTP = mean distance of all true positive matches

    IDS: occurs when a GT object is matched to a different track than in previous frame.
    FRAG: occurs when a track is interrupted (GT was matched, then unmatched, then matched again).

    Args:
        tracking_results: Per-frame tracking outputs.
        ground_truths: Per-frame ground truth.
        sequence_frames: Dict mapping seq_id -> [(frame_id, global_idx), ...].
        cls_idx: Class index to evaluate.
        score_threshold: Minimum track score to include.
        distance_threshold: Maximum matching distance.

    Returns:
        Tuple of (MOTA, MOTP, IDS, FRAG).
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_ids = 0
    total_frag = 0
    all_match_distances = []

    for seq_id, frames in sequence_frames.items():
        # Track the GT-to-track assignment across frames for this sequence
        # gt_to_track: maps (global_gt_offset + local_gt_idx) -> track_id
        prev_gt_to_track = {}
        # Track whether each GT was matched in previous frame (for FRAG)
        prev_gt_matched = {}

        for frame_id, global_idx in frames:
            tr = tracking_results[global_idx]
            gt = ground_truths[global_idx]

            if tr is None:
                # Count all GT as FN
                gt_cls_mask = gt["labels"] == cls_idx
                total_gt += int(gt_cls_mask.sum())
                total_fn += int(gt_cls_mask.sum())
                # Update prev state: all GT unmatched
                gt_tracking_ids = gt["tracking_ids"][gt_cls_mask]
                for gt_tid in gt_tracking_ids:
                    if prev_gt_matched.get(gt_tid, False):
                        # Was matched before, now unmatched -> potential frag
                        pass
                    prev_gt_matched[gt_tid] = False
                continue

            # Filter tracks by class and score threshold
            track_cls_mask = (tr["labels"] == cls_idx) & (tr["scores"] >= score_threshold)
            gt_cls_mask = gt["labels"] == cls_idx

            track_boxes = tr["boxes"][track_cls_mask]
            track_ids = tr["track_ids"][track_cls_mask]
            gt_boxes = gt["boxes"][gt_cls_mask]
            gt_tracking_ids = gt["tracking_ids"][gt_cls_mask]

            num_tracks = len(track_boxes)
            num_gt = len(gt_boxes)
            total_gt += num_gt

            if num_tracks == 0 and num_gt == 0:
                continue

            if num_tracks == 0:
                total_fn += num_gt
                for gt_tid in gt_tracking_ids:
                    if prev_gt_matched.get(gt_tid, False):
                        # Was matched, now unmatched
                        prev_gt_matched[gt_tid] = False
                continue

            if num_gt == 0:
                total_fp += num_tracks
                continue

            # Compute distance matrix between tracks and GT
            track_centers = track_boxes[:, :2]
            gt_centers = gt_boxes[:, :2]
            diff = track_centers[:, np.newaxis, :] - gt_centers[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)  # (num_tracks, num_gt)

            # Greedy matching
            matched_tracks_set = set()
            matched_gt_set = set()
            matches = []  # list of (track_local_idx, gt_local_idx)

            # Flatten and sort
            flat_indices = np.argsort(dist_matrix, axis=None)
            for flat_idx in flat_indices:
                t_idx = int(flat_idx // num_gt)
                g_idx = int(flat_idx % num_gt)

                if dist_matrix[t_idx, g_idx] > distance_threshold:
                    break

                if t_idx in matched_tracks_set or g_idx in matched_gt_set:
                    continue

                matched_tracks_set.add(t_idx)
                matched_gt_set.add(g_idx)
                matches.append((t_idx, g_idx))

                if len(matched_tracks_set) == num_tracks or len(matched_gt_set) == num_gt:
                    break

            # Count FP, FN
            fp_this_frame = num_tracks - len(matches)
            fn_this_frame = num_gt - len(matches)
            total_fp += fp_this_frame
            total_fn += fn_this_frame

            # Collect match distances for MOTP
            for t_idx, g_idx in matches:
                all_match_distances.append(dist_matrix[t_idx, g_idx])

            # Check for identity switches and fragmentations
            current_gt_to_track = {}
            for t_idx, g_idx in matches:
                gt_tid = gt_tracking_ids[g_idx]
                assigned_track_id = track_ids[t_idx]
                current_gt_to_track[gt_tid] = assigned_track_id

                # IDS: GT was previously matched to a different track
                if gt_tid in prev_gt_to_track:
                    if prev_gt_to_track[gt_tid] != assigned_track_id:
                        total_ids += 1

                # FRAG: GT was matched, then unmatched (in a prior frame), now matched again
                if gt_tid in prev_gt_matched:
                    if not prev_gt_matched[gt_tid]:
                        # Was unmatched in previous frame(s), now matched again
                        if gt_tid in prev_gt_to_track:
                            total_frag += 1

            # Update state for next frame
            # Mark which GT are matched this frame
            new_prev_gt_matched = {}
            for g_idx in range(num_gt):
                gt_tid = gt_tracking_ids[g_idx]
                new_prev_gt_matched[gt_tid] = g_idx in matched_gt_set

            prev_gt_to_track = current_gt_to_track
            prev_gt_matched = new_prev_gt_matched

    # Compute MOTA and MOTP
    if total_gt == 0:
        mota = 0.0
    else:
        mota = 1.0 - (total_fp + total_fn + total_ids) / total_gt

    if all_match_distances:
        motp = float(np.mean(all_match_distances))
    else:
        motp = float("inf")

    return mota, motp, total_ids, total_frag


# ---------------------------------------------------------------------------
# Results formatting and printing
# ---------------------------------------------------------------------------


def print_detection_results(results: Dict[str, Any], class_names: List[str]) -> None:
    """Print detection evaluation results as a formatted table.

    Args:
        results: Detection evaluation results dict.
        class_names: List of class names.
    """
    distance_thresholds = results["distance_thresholds"]

    # Header
    header = "Detection Evaluation Results"
    print("\n" + "=" * 80)
    print(f"{header:^80}")
    print("=" * 80)

    # Summary
    print(f"\n  mAP:  {results['mAP']:.4f}")
    print(f"  NDS:  {results['NDS']:.4f}")
    print()

    # TP Metrics
    tp = results["mean_tp_metrics"]
    print("  True Positive Metrics (mean across classes):")
    print(f"    ATE: {tp['ATE']:.4f}  ASE: {tp['ASE']:.4f}  AOE: {tp['AOE']:.4f}  "
          f"AVE: {tp['AVE']:.4f}  AAE: {tp['AAE']:.4f}")
    print()

    # Per-class AP table
    # Build header row
    thresh_headers = [f"{dt:.1f}m" for dt in distance_thresholds]
    col_width = 8
    class_col_width = 22

    row_format = f"  {{:<{class_col_width}}}" + f"{{:>{col_width}}}" * (len(thresh_headers) + 1)
    sep_line = "  " + "-" * (class_col_width + col_width * (len(thresh_headers) + 1))

    print("  Per-Class Average Precision:")
    print(sep_line)
    print(row_format.format("Class", *thresh_headers, "Mean"))
    print(sep_line)

    for cls_name in class_names:
        ap_vals = results["ap_table"][cls_name]
        row_data = [f"{ap_vals[f'{dt:.1f}m']:.4f}" for dt in distance_thresholds]
        mean_ap = results["per_class_ap"][cls_name]
        row_data.append(f"{mean_ap:.4f}")
        print(row_format.format(cls_name, *row_data))

    print(sep_line)

    # Mean row
    mean_per_thresh = []
    for dt in distance_thresholds:
        vals = [results["ap_table"][cn][f"{dt:.1f}m"] for cn in class_names]
        mean_per_thresh.append(f"{np.mean(vals):.4f}")
    mean_per_thresh.append(f"{results['mAP']:.4f}")
    print(row_format.format("MEAN", *mean_per_thresh))
    print(sep_line)

    # Per-class TP metrics table
    print("\n  Per-Class TP Metrics:")
    tp_row_format = f"  {{:<{class_col_width}}}" + f"{{:>{col_width}}}" * 5
    tp_sep = "  " + "-" * (class_col_width + col_width * 5)

    print(tp_sep)
    print(tp_row_format.format("Class", "ATE", "ASE", "AOE", "AVE", "AAE"))
    print(tp_sep)

    for cls_name in class_names:
        tp_cls = results["per_class_tp_metrics"][cls_name]
        print(tp_row_format.format(
            cls_name,
            f"{tp_cls['ATE']:.4f}",
            f"{tp_cls['ASE']:.4f}",
            f"{tp_cls['AOE']:.4f}",
            f"{tp_cls['AVE']:.4f}",
            f"{tp_cls['AAE']:.4f}",
        ))

    print(tp_sep)
    print(tp_row_format.format(
        "MEAN",
        f"{tp['ATE']:.4f}",
        f"{tp['ASE']:.4f}",
        f"{tp['AOE']:.4f}",
        f"{tp['AVE']:.4f}",
        f"{tp['AAE']:.4f}",
    ))
    print(tp_sep)
    print()


def print_tracking_results(results: Dict[str, Any], class_names: List[str]) -> None:
    """Print tracking evaluation results as a formatted table.

    Args:
        results: Tracking evaluation results dict.
        class_names: List of class names.
    """
    header = "Tracking Evaluation Results"
    print("\n" + "=" * 80)
    print(f"{header:^80}")
    print("=" * 80)

    # Summary
    print(f"\n  AMOTA:  {results['AMOTA']:.4f}")
    amotp_str = f"{results['AMOTP']:.4f}" if results['AMOTP'] < np.inf else "inf"
    print(f"  AMOTP:  {amotp_str}")
    print(f"  IDS:    {results['IDS']}")
    print(f"  FRAG:   {results['FRAG']}")
    print()

    # Per-class table
    col_width = 10
    class_col_width = 22
    row_format = f"  {{:<{class_col_width}}}" + f"{{:>{col_width}}}" * 4
    sep_line = "  " + "-" * (class_col_width + col_width * 4)

    print("  Per-Class Tracking Metrics:")
    print(sep_line)
    print(row_format.format("Class", "AMOTA", "AMOTP", "IDS", "FRAG"))
    print(sep_line)

    for cls_name in class_names:
        cls_metrics = results["per_class"][cls_name]
        amotp_val = f"{cls_metrics['AMOTP']:.4f}" if cls_metrics['AMOTP'] < np.inf else "inf"
        print(row_format.format(
            cls_name,
            f"{cls_metrics['AMOTA']:.4f}",
            amotp_val,
            str(cls_metrics['IDS']),
            str(cls_metrics['FRAG']),
        ))

    print(sep_line)

    # Mean row
    print(row_format.format(
        "MEAN",
        f"{results['AMOTA']:.4f}",
        amotp_str,
        str(results['IDS']),
        str(results['FRAG']),
    ))
    print(sep_line)
    print()


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------


def save_results(
    detection_results: Dict[str, Any],
    tracking_results: Optional[Dict[str, Any]],
    output_path: str,
) -> str:
    """Save evaluation results to JSON file.

    Args:
        detection_results: Detection evaluation results.
        tracking_results: Tracking evaluation results (or None).
        output_path: Directory to save results.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(output_path, exist_ok=True)

    # Convert numpy types to native Python for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, float) and obj == float("inf"):
            return "inf"
        return obj

    output = {
        "detection": convert_to_serializable(detection_results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if tracking_results is not None:
        output["tracking"] = convert_to_serializable(tracking_results)

    output_file = os.path.join(output_path, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info("CenterPoint Evaluation")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Data path: {args.data_path}")
    logger.info(f"  Output path: {args.output_path}")
    logger.info(f"  Tracking eval: {args.eval_tracking}")

    # Load configuration
    config = load_config(args.config)

    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"  Device: {device}")

    # Load model
    model = load_model(config, args.checkpoint, device)

    # Build validation dataset and dataloader
    dataset_cfg = config.get("dataset", {})
    class_names = dataset_cfg.get("class_names", [
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    ])

    val_dataset = NuScenesDataset(
        data_root=args.data_path,
        split="val",
        voxel_size=dataset_cfg.get("voxel_size", [0.075, 0.075, 0.2]),
        point_cloud_range=dataset_cfg.get(
            "point_cloud_range", [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        ),
        max_points_per_voxel=dataset_cfg.get("max_points_per_voxel", 10),
        max_voxels=dataset_cfg.get("max_voxels", {"train": 120000, "val": 160000}),
        class_names=class_names,
        augmentation={},  # No augmentation for evaluation
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_cuda,
        drop_last=False,
    )

    logger.info(f"  Validation samples: {len(val_dataset)}")

    # Run inference
    logger.info("Running inference on validation set...")
    eval_start = time.time()

    predictions, metadata = run_inference(
        model, val_dataloader, device, score_threshold=args.score_threshold
    )

    inference_time = time.time() - eval_start
    logger.info(f"Inference completed in {inference_time:.1f}s")

    # Load ground truth
    logger.info("Loading ground truth annotations...")
    ground_truths = load_ground_truth(val_dataloader)

    assert len(predictions) == len(ground_truths), (
        f"Prediction count ({len(predictions)}) != GT count ({len(ground_truths)})"
    )

    # Detection evaluation
    logger.info("Computing detection metrics...")
    det_start = time.time()

    detection_results = evaluate_detection(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=class_names,
        distance_thresholds=[0.5, 1.0, 2.0, 4.0],
    )

    det_time = time.time() - det_start
    logger.info(f"Detection evaluation completed in {det_time:.1f}s")

    # Print detection results
    print_detection_results(detection_results, class_names)

    # Tracking evaluation (optional)
    tracking_eval_results = None

    if args.eval_tracking:
        logger.info("Running tracker across sequences...")
        track_start = time.time()

        tracker_cfg = config.get("tracker", {
            "max_age": 3,
            "min_hits": 1,
            "distance_threshold": 2.0,
        })

        tracking_outputs = run_tracking(predictions, metadata, tracker_cfg)

        logger.info("Computing tracking metrics...")
        tracking_eval_results = evaluate_tracking(
            tracking_results=tracking_outputs,
            ground_truths=ground_truths,
            metadata=metadata,
            class_names=class_names,
            distance_threshold=tracker_cfg.get("distance_threshold", 2.0),
        )

        track_time = time.time() - track_start
        logger.info(f"Tracking evaluation completed in {track_time:.1f}s")

        # Print tracking results
        print_tracking_results(tracking_eval_results, class_names)

    # Save results to JSON
    output_file = save_results(detection_results, tracking_eval_results, args.output_path)

    # Final summary
    total_time = time.time() - eval_start
    print("\n" + "=" * 80)
    print(f"{'Evaluation Summary':^80}")
    print("=" * 80)
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  mAP:            {detection_results['mAP']:.4f}")
    print(f"  NDS:            {detection_results['NDS']:.4f}")
    if tracking_eval_results is not None:
        print(f"  AMOTA:          {tracking_eval_results['AMOTA']:.4f}")
        amotp_str = (f"{tracking_eval_results['AMOTP']:.4f}"
                     if tracking_eval_results['AMOTP'] < np.inf else "inf")
        print(f"  AMOTP:          {amotp_str}")
    print(f"  Results saved:  {output_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
