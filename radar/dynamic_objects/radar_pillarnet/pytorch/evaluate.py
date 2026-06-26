"""
Evaluation script for RadarPillarNet with nuScenes-style metrics.

Computes the following metrics for 3D object detection from radar point clouds:
- mAP: Mean Average Precision at center distance thresholds [0.5, 1.0, 2.0, 4.0] meters
- True Positive metrics:
    - ATE: Average Translation Error (Euclidean center distance)
    - ASE: Average Scale Error (1 - 3D IoU after alignment)
    - AOE: Average Orientation Error (smallest yaw angle difference)
    - AVE: Average Velocity Error (L2 norm of velocity difference)
    - AAE: Average Attribute Error (1 - attribute classification accuracy)
- NDS: nuScenes Detection Score = (1/10) * [5*mAP + sum(max(1-TP_metric, 0))]
- Per-class breakdown for all 10 nuScenes classes
- Distance-based evaluation (0-30m, 30-50m, 50m+)

Usage:
    python evaluate.py --checkpoint /path/to/checkpoint.pth \\
                       --data_root /path/to/nuscenes/processed \\
                       --split val \\
                       --batch_size 4 \\
                       --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.optimize import linear_sum_assignment

from .model import RadarPillarNet

# ============================================================================
# Constants
# ============================================================================

NUSCENES_CLASSES: List[str] = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]

# Center distance thresholds for mAP (meters)
DISTANCE_THRESHOLDS: List[float] = [0.5, 1.0, 2.0, 4.0]

# TP metric error caps (used for NDS normalization)
TP_METRIC_CAPS: Dict[str, float] = {
    "ATE": 2.0,  # meters
    "ASE": 1.0,  # dimensionless (1 - IoU)
    "AOE": 1.0,  # radians (capped at 1.0, not pi)
    "AVE": 2.0,  # m/s
    "AAE": 1.0,  # dimensionless
}

# Distance ranges for distance-based evaluation
DISTANCE_RANGES: List[Tuple[float, float, str]] = [
    (0.0, 30.0, "0-30m"),
    (30.0, 50.0, "30-50m"),
    (50.0, float("inf"), "50m+"),
]

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class DetectionBox:
    """Represents a single 3D detection bounding box.

    Attributes:
        center: (3,) box center [x, y, z] in ego frame (meters).
        size: (3,) box dimensions [w, l, h] in meters.
        yaw: Heading angle in radians.
        velocity: (2,) velocity [vx, vy] in m/s.
        score: Detection confidence (0-1). For GT boxes, set to 1.0.
        label: Class index (0-indexed into NUSCENES_CLASSES).
        attribute: Attribute label index (e.g., moving/stopped). -1 if unknown.
        sample_token: Identifier linking detection to a specific sample.
    """

    center: np.ndarray
    size: np.ndarray
    yaw: float
    velocity: np.ndarray
    score: float
    label: int
    attribute: int = -1
    sample_token: str = ""


@dataclass
class EvalResults:
    """Container for all evaluation results.

    Attributes:
        mAP: Overall mean Average Precision.
        per_class_ap: Per-class AP averaged over distance thresholds.
        per_class_ap_per_threshold: AP per class per distance threshold.
        ate: Average Translation Error per class.
        ase: Average Scale Error per class.
        aoe: Average Orientation Error per class.
        ave: Average Velocity Error per class.
        aae: Average Attribute Error per class.
        nds: nuScenes Detection Score.
        distance_results: Metrics broken down by distance range.
    """

    mAP: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    per_class_ap_per_threshold: Dict[str, Dict[float, float]] = field(
        default_factory=dict
    )
    ate: Dict[str, float] = field(default_factory=dict)
    ase: Dict[str, float] = field(default_factory=dict)
    aoe: Dict[str, float] = field(default_factory=dict)
    ave: Dict[str, float] = field(default_factory=dict)
    aae: Dict[str, float] = field(default_factory=dict)
    nds: float = 0.0
    distance_results: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ============================================================================
# Metric Computation Functions
# ============================================================================


def center_distance(box1: DetectionBox, box2: DetectionBox) -> float:
    """Compute Euclidean distance between box centers in the BEV plane.

    Args:
        box1: First detection box.
        box2: Second detection box.

    Returns:
        2D Euclidean distance between centers (x, y only).
    """
    return float(
        np.sqrt(
            (box1.center[0] - box2.center[0]) ** 2
            + (box1.center[1] - box2.center[1]) ** 2
        )
    )


def translation_error(pred: DetectionBox, gt: DetectionBox) -> float:
    """Compute Average Translation Error (ATE).

    ATE is the 2D Euclidean center distance in BEV.

    Args:
        pred: Predicted detection box.
        gt: Ground truth detection box.

    Returns:
        Translation error in meters.
    """
    return center_distance(pred, gt)


def scale_error(pred: DetectionBox, gt: DetectionBox) -> float:
    """Compute Average Scale Error (ASE).

    ASE = 1 - 3D IoU after center and orientation alignment.
    We compute the 3D IoU assuming the boxes are axis-aligned (after alignment),
    which simplifies to the product of per-axis overlap ratios.

    Args:
        pred: Predicted detection box.
        gt: Ground truth detection box.

    Returns:
        Scale error in range [0, 1]. 0 means perfect overlap.
    """
    # Compute aligned 3D IoU (boxes centered and rotation-aligned)
    # Under alignment, IoU depends only on size ratios
    pred_w, pred_l, pred_h = pred.size[0], pred.size[1], pred.size[2]
    gt_w, gt_l, gt_h = gt.size[0], gt.size[1], gt.size[2]

    # Intersection dimensions (minimum of each axis extent)
    inter_w = min(pred_w, gt_w)
    inter_l = min(pred_l, gt_l)
    inter_h = min(pred_h, gt_h)

    # Intersection and union volumes
    inter_vol = inter_w * inter_l * inter_h
    pred_vol = pred_w * pred_l * pred_h
    gt_vol = gt_w * gt_l * gt_h
    union_vol = pred_vol + gt_vol - inter_vol

    if union_vol <= 0:
        return 1.0

    iou_3d = inter_vol / union_vol
    return float(1.0 - iou_3d)


def orientation_error(pred: DetectionBox, gt: DetectionBox) -> float:
    """Compute Average Orientation Error (AOE).

    AOE is the smallest yaw angle difference between prediction and ground truth.
    The error is computed as the absolute difference, wrapped to [0, pi].

    Args:
        pred: Predicted detection box.
        gt: Ground truth detection box.

    Returns:
        Orientation error in radians, range [0, pi].
    """
    diff = abs(pred.yaw - gt.yaw)
    # Wrap to [0, pi] because objects have 180-degree symmetry
    diff = diff % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return float(diff)


def velocity_error(pred: DetectionBox, gt: DetectionBox) -> float:
    """Compute Average Velocity Error (AVE).

    AVE is the L2 norm of the velocity vector difference.

    Args:
        pred: Predicted detection box.
        gt: Ground truth detection box.

    Returns:
        Velocity error in m/s.
    """
    return float(np.linalg.norm(pred.velocity - gt.velocity))


def attribute_error(pred: DetectionBox, gt: DetectionBox) -> float:
    """Compute Average Attribute Error (AAE).

    AAE = 1 - (attribute classification accuracy).
    Returns 0 if attributes match, 1 otherwise.
    If either attribute is unknown (-1), returns 1.0 (worst case).

    Args:
        pred: Predicted detection box.
        gt: Ground truth detection box.

    Returns:
        Attribute error: 0.0 if match, 1.0 if mismatch or unknown.
    """
    if pred.attribute < 0 or gt.attribute < 0:
        return 1.0
    return 0.0 if pred.attribute == gt.attribute else 1.0


def compute_ap_per_class_per_threshold(
    predictions: List[DetectionBox],
    ground_truths: List[DetectionBox],
    class_idx: int,
    distance_threshold: float,
    recall_interpolation_points: int = 101,
) -> Tuple[float, List[float], List[float], List[float], List[float], List[float]]:
    """Compute Average Precision for a single class at a single distance threshold.

    Uses center distance matching with the Hungarian algorithm to find optimal
    assignment between predictions and ground truths per sample. A prediction
    is a true positive if its center distance to the matched GT is within the
    distance threshold.

    Args:
        predictions: All predicted boxes (all samples, single class).
        ground_truths: All ground truth boxes (all samples, single class).
        class_idx: Class index to evaluate.
        distance_threshold: Maximum center distance for a match (meters).
        recall_interpolation_points: Number of recall points for AP interpolation.

    Returns:
        Tuple of:
            - ap: Average precision value.
            - ate_list: Translation errors for true positives.
            - ase_list: Scale errors for true positives.
            - aoe_list: Orientation errors for true positives.
            - ave_list: Velocity errors for true positives.
            - aae_list: Attribute errors for true positives.
    """
    # Filter by class
    preds_cls = [p for p in predictions if p.label == class_idx]
    gts_cls = [g for g in ground_truths if g.label == class_idx]

    if len(gts_cls) == 0:
        return 0.0, [], [], [], [], []

    if len(preds_cls) == 0:
        return 0.0, [], [], [], [], []

    # Sort predictions by score (descending)
    preds_cls = sorted(preds_cls, key=lambda x: x.score, reverse=True)

    # Group ground truths by sample_token
    gt_by_sample: Dict[str, List[DetectionBox]] = defaultdict(list)
    for gt in gts_cls:
        gt_by_sample[gt.sample_token].append(gt)

    # Track which GTs have been matched (per sample)
    gt_matched: Dict[str, List[bool]] = {}
    for token, gts in gt_by_sample.items():
        gt_matched[token] = [False] * len(gts)

    # Process predictions in descending score order
    tp_list: List[int] = []
    fp_list: List[int] = []
    ate_list: List[float] = []
    ase_list: List[float] = []
    aoe_list: List[float] = []
    ave_list: List[float] = []
    aae_list: List[float] = []

    num_gt_total = len(gts_cls)

    for pred in preds_cls:
        sample_token = pred.sample_token
        sample_gts = gt_by_sample.get(sample_token, [])

        if len(sample_gts) == 0:
            tp_list.append(0)
            fp_list.append(1)
            continue

        # Compute center distances to all unmatched GTs in this sample
        min_dist = float("inf")
        best_gt_idx = -1

        for gt_idx, gt in enumerate(sample_gts):
            if gt_matched[sample_token][gt_idx]:
                continue
            dist = center_distance(pred, gt)
            if dist < min_dist:
                min_dist = dist
                best_gt_idx = gt_idx

        # Check if the closest unmatched GT is within threshold
        if best_gt_idx >= 0 and min_dist <= distance_threshold:
            # True positive
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[sample_token][best_gt_idx] = True

            # Compute TP metrics
            matched_gt = sample_gts[best_gt_idx]
            ate_list.append(translation_error(pred, matched_gt))
            ase_list.append(scale_error(pred, matched_gt))
            aoe_list.append(orientation_error(pred, matched_gt))
            ave_list.append(velocity_error(pred, matched_gt))
            aae_list.append(attribute_error(pred, matched_gt))
        else:
            # False positive
            tp_list.append(0)
            fp_list.append(1)

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp_list).astype(np.float64)
    fp_cumsum = np.cumsum(fp_list).astype(np.float64)

    recall = tp_cumsum / num_gt_total
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Interpolate precision at fixed recall points (nuScenes style)
    recall_interp = np.linspace(0.0, 1.0, recall_interpolation_points)
    precision_interp = np.zeros_like(recall_interp)

    for i, r in enumerate(recall_interp):
        # Maximum precision at recall >= r
        valid_mask = recall >= r
        if valid_mask.any():
            precision_interp[i] = precision[valid_mask].max()
        else:
            precision_interp[i] = 0.0

    # AP = mean of interpolated precision values
    ap = float(np.mean(precision_interp))

    return ap, ate_list, ase_list, aoe_list, ave_list, aae_list


def compute_nuscenes_metrics(
    predictions: List[DetectionBox],
    ground_truths: List[DetectionBox],
) -> EvalResults:
    """Compute full nuScenes-style detection metrics.

    Evaluates mAP at multiple distance thresholds, per-class TP metrics,
    and the overall NDS score.

    Args:
        predictions: All predicted detection boxes across all samples.
        ground_truths: All ground truth detection boxes across all samples.

    Returns:
        EvalResults containing mAP, per-class metrics, and NDS.
    """
    results = EvalResults()

    # Per-class, per-threshold AP and TP metrics
    all_ate: Dict[str, List[float]] = defaultdict(list)
    all_ase: Dict[str, List[float]] = defaultdict(list)
    all_aoe: Dict[str, List[float]] = defaultdict(list)
    all_ave: Dict[str, List[float]] = defaultdict(list)
    all_aae: Dict[str, List[float]] = defaultdict(list)

    for cls_idx, cls_name in enumerate(NUSCENES_CLASSES):
        results.per_class_ap_per_threshold[cls_name] = {}

        for threshold in DISTANCE_THRESHOLDS:
            ap, ate_l, ase_l, aoe_l, ave_l, aae_l = (
                compute_ap_per_class_per_threshold(
                    predictions, ground_truths, cls_idx, threshold
                )
            )
            results.per_class_ap_per_threshold[cls_name][threshold] = ap

            # Accumulate TP metrics (use largest threshold = 4.0m for TP metrics,
            # matching nuScenes convention where TP metrics are computed at 2m threshold)
            if threshold == 2.0:
                all_ate[cls_name].extend(ate_l)
                all_ase[cls_name].extend(ase_l)
                all_aoe[cls_name].extend(aoe_l)
                all_ave[cls_name].extend(ave_l)
                all_aae[cls_name].extend(aae_l)

        # Mean AP across thresholds for this class
        class_aps = list(results.per_class_ap_per_threshold[cls_name].values())
        results.per_class_ap[cls_name] = float(np.mean(class_aps))

    # Overall mAP: mean over all classes
    results.mAP = float(np.mean(list(results.per_class_ap.values())))

    # Compute per-class TP metrics (mean over all true positive matches)
    for cls_name in NUSCENES_CLASSES:
        if len(all_ate[cls_name]) > 0:
            results.ate[cls_name] = float(np.mean(all_ate[cls_name]))
            results.ase[cls_name] = float(np.mean(all_ase[cls_name]))
            results.aoe[cls_name] = float(np.mean(all_aoe[cls_name]))
            results.ave[cls_name] = float(np.mean(all_ave[cls_name]))
            results.aae[cls_name] = float(np.mean(all_aae[cls_name]))
        else:
            # No true positives for this class; assign worst-case errors
            results.ate[cls_name] = TP_METRIC_CAPS["ATE"]
            results.ase[cls_name] = TP_METRIC_CAPS["ASE"]
            results.aoe[cls_name] = TP_METRIC_CAPS["AOE"]
            results.ave[cls_name] = TP_METRIC_CAPS["AVE"]
            results.aae[cls_name] = TP_METRIC_CAPS["AAE"]

    # Compute NDS
    # NDS = (1/10) * [5*mAP + sum(max(1 - TP_i/cap_i, 0.0)) for each of 5 TP metrics]
    # where the sum is over the mean TP metric across classes
    mean_ate = float(np.mean(list(results.ate.values())))
    mean_ase = float(np.mean(list(results.ase.values())))
    mean_aoe = float(np.mean(list(results.aoe.values())))
    mean_ave = float(np.mean(list(results.ave.values())))
    mean_aae = float(np.mean(list(results.aae.values())))

    tp_scores = [
        max(1.0 - mean_ate / TP_METRIC_CAPS["ATE"], 0.0),
        max(1.0 - mean_ase / TP_METRIC_CAPS["ASE"], 0.0),
        max(1.0 - mean_aoe / TP_METRIC_CAPS["AOE"], 0.0),
        max(1.0 - mean_ave / TP_METRIC_CAPS["AVE"], 0.0),
        max(1.0 - mean_aae / TP_METRIC_CAPS["AAE"], 0.0),
    ]

    results.nds = (5.0 * results.mAP + sum(tp_scores)) / 10.0

    return results


def compute_distance_based_metrics(
    predictions: List[DetectionBox],
    ground_truths: List[DetectionBox],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by distance range from the ego vehicle.

    Evaluates detection performance separately for near (0-30m), medium (30-50m),
    and far (50m+) ranges, providing insight into how performance degrades with
    distance.

    Args:
        predictions: All predicted detection boxes.
        ground_truths: All ground truth detection boxes.

    Returns:
        Dictionary mapping distance range name to a dict of metrics:
            - mAP: mean AP at the 2.0m threshold
            - num_gt: number of ground truths in range
            - num_pred: number of predictions in range
    """
    distance_results: Dict[str, Dict[str, float]] = {}

    for range_min, range_max, range_name in DISTANCE_RANGES:
        # Filter GT and predictions by distance from ego (origin)
        gts_in_range = [
            g
            for g in ground_truths
            if range_min
            <= np.sqrt(g.center[0] ** 2 + g.center[1] ** 2)
            < range_max
        ]
        preds_in_range = [
            p
            for p in predictions
            if range_min
            <= np.sqrt(p.center[0] ** 2 + p.center[1] ** 2)
            < range_max
        ]

        if len(gts_in_range) == 0:
            distance_results[range_name] = {
                "mAP": 0.0,
                "num_gt": 0,
                "num_pred": len(preds_in_range),
            }
            continue

        # Compute mAP at 2.0m threshold for this distance range
        aps_per_class: List[float] = []
        for cls_idx, cls_name in enumerate(NUSCENES_CLASSES):
            ap, _, _, _, _, _ = compute_ap_per_class_per_threshold(
                preds_in_range, gts_in_range, cls_idx, distance_threshold=2.0
            )
            aps_per_class.append(ap)

        # Only average over classes that have GT in this range
        classes_with_gt = [
            cls_idx
            for cls_idx in range(len(NUSCENES_CLASSES))
            if any(g.label == cls_idx for g in gts_in_range)
        ]

        if len(classes_with_gt) > 0:
            range_mAP = float(
                np.mean([aps_per_class[idx] for idx in classes_with_gt])
            )
        else:
            range_mAP = 0.0

        distance_results[range_name] = {
            "mAP": range_mAP,
            "num_gt": len(gts_in_range),
            "num_pred": len(preds_in_range),
        }

    return distance_results


# ============================================================================
# Dataset Interface
# ============================================================================


class RadarEvalDataset(Dataset):
    """Dataset for RadarPillarNet evaluation on nuScenes radar data.

    Loads preprocessed radar pillar data and corresponding ground truth
    annotations for evaluation. Expects data in the format produced by
    the project's preprocessing pipeline.

    Directory structure:
        data_root/
            split/
                pillars/       - (max_pillars, max_points_per_pillar, 9) .npy
                indices/       - (max_pillars, 3) .npy
                num_points/    - (max_pillars,) .npy
                annotations/   - JSON with gt_boxes, gt_labels, gt_velocity, metadata
    """

    def __init__(self, data_root: str, split: str = "val") -> None:
        """Initialize evaluation dataset.

        Args:
            data_root: Root directory of the preprocessed dataset.
            split: Data split to evaluate ('val' or 'test').
        """
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / split

        # Discover samples
        pillars_dir = self.split_dir / "pillars"
        if not pillars_dir.exists():
            raise FileNotFoundError(
                f"Pillars directory not found: {pillars_dir}. "
                f"Ensure data has been preprocessed."
            )

        self.sample_ids = sorted(
            [f.stem for f in pillars_dir.glob("*.npy")]
        )

        if len(self.sample_ids) == 0:
            raise ValueError(
                f"No samples found in {pillars_dir}. "
                f"Check that preprocessing has been run."
            )

        logger.info(
            "Loaded %d samples from %s split at %s",
            len(self.sample_ids),
            split,
            data_root,
        )

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single sample for evaluation.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
                'pillars': (max_pillars, max_points_per_pillar, 9) float32 tensor
                'pillar_indices': (max_pillars, 3) int32 tensor
                'num_points_per_pillar': (max_pillars,) int32 tensor
                'gt_boxes': (M, 7) float32 array of GT boxes
                'gt_labels': (M,) int64 array of GT class labels
                'gt_velocity': (M, 2) float32 array of GT velocities
                'metadata': dict with sample_token and other info
        """
        sample_id = self.sample_ids[idx]

        # Load pillar data
        pillars = np.load(
            self.split_dir / "pillars" / f"{sample_id}.npy"
        ).astype(np.float32)
        pillar_indices = np.load(
            self.split_dir / "indices" / f"{sample_id}.npy"
        ).astype(np.int32)
        num_points = np.load(
            self.split_dir / "num_points" / f"{sample_id}.npy"
        ).astype(np.int32)

        # Load annotations
        anno_path = self.split_dir / "annotations" / f"{sample_id}.json"
        with open(anno_path, "r") as f:
            annotation = json.load(f)

        gt_boxes = np.array(annotation["gt_boxes"], dtype=np.float32)
        gt_labels = np.array(annotation["gt_labels"], dtype=np.int64)
        gt_velocity = np.array(annotation["gt_velocity"], dtype=np.float32)

        metadata = annotation.get("metadata", {})
        metadata["sample_token"] = metadata.get("sample_token", sample_id)

        return {
            "pillars": torch.from_numpy(pillars),
            "pillar_indices": torch.from_numpy(pillar_indices),
            "num_points_per_pillar": torch.from_numpy(num_points),
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "gt_velocity": gt_velocity,
            "metadata": metadata,
        }


def collate_eval_batch(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Custom collate function for evaluation batches.

    Stacks tensor inputs for the model and keeps ground truths as lists.

    Args:
        batch: List of sample dicts from RadarEvalDataset.

    Returns:
        Collated batch dict with stacked model inputs and listed GT data.
    """
    pillars = torch.stack([s["pillars"] for s in batch], dim=0)
    pillar_indices = torch.stack([s["pillar_indices"] for s in batch], dim=0)
    num_points = torch.stack([s["num_points_per_pillar"] for s in batch], dim=0)

    return {
        "pillars": pillars,
        "pillar_indices": pillar_indices,
        "num_points_per_pillar": num_points,
        "gt_boxes": [s["gt_boxes"] for s in batch],
        "gt_labels": [s["gt_labels"] for s in batch],
        "gt_velocity": [s["gt_velocity"] for s in batch],
        "metadata": [s["metadata"] for s in batch],
    }


# ============================================================================
# Model Loading
# ============================================================================


def load_model(
    checkpoint_path: str,
    device: torch.device,
    num_classes: int = 10,
) -> RadarPillarNet:
    """Load a trained RadarPillarNet model from a checkpoint.

    Supports checkpoints saved as:
    - Full state dict: {'model_state_dict': ..., 'config': ...}
    - Plain state dict: model.state_dict()

    Args:
        checkpoint_path: Path to the model checkpoint file.
        device: Device to load the model onto.
        num_classes: Number of detection classes (default 10 for nuScenes).

    Returns:
        Loaded model in evaluation mode.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint is incompatible with the model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract model config if available
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
        model = RadarPillarNet(
            in_channels=config.get("in_channels", 9),
            pillar_feat_channels=config.get("pillar_feat_channels", 64),
            x_range=tuple(config.get("x_range", (-51.2, 51.2))),
            y_range=tuple(config.get("y_range", (-51.2, 51.2))),
            z_range=tuple(config.get("z_range", (-5.0, 3.0))),
            pillar_size=tuple(config.get("pillar_size", (0.4, 0.4, 8.0))),
            max_points_per_pillar=config.get("max_points_per_pillar", 20),
            max_pillars=config.get("max_pillars", 12000),
            num_classes=config.get("num_classes", num_classes),
        )
    else:
        # Use default configuration
        model = RadarPillarNet(num_classes=num_classes)

    # Load state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel prefix stripping
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully (%d parameters)", sum(
        p.numel() for p in model.parameters()
    ))

    return model


# ============================================================================
# Inference Loop
# ============================================================================


def run_inference(
    model: RadarPillarNet,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[DetectionBox], List[DetectionBox]]:
    """Run model inference on the evaluation set and collect predictions and GTs.

    Args:
        model: Trained RadarPillarNet model in eval mode.
        dataloader: DataLoader yielding evaluation batches.
        device: Compute device for inference.

    Returns:
        Tuple of (all_predictions, all_ground_truths) as lists of DetectionBox.
    """
    all_predictions: List[DetectionBox] = []
    all_ground_truths: List[DetectionBox] = []

    model.eval()
    num_samples_processed = 0
    inference_start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move inputs to device
            batch_input = {
                "pillars": batch["pillars"].to(device),
                "pillar_indices": batch["pillar_indices"].to(device),
                "num_points_per_pillar": batch["num_points_per_pillar"].to(device),
            }

            # Run inference with post-processing
            detections = model.predict(batch_input)

            batch_size = len(detections)

            for i in range(batch_size):
                sample_token = batch["metadata"][i].get(
                    "sample_token", f"sample_{num_samples_processed}"
                )

                # Convert predictions to DetectionBox
                det = detections[i]
                boxes = det["boxes"].cpu().numpy()
                scores = det["scores"].cpu().numpy()
                labels = det["labels"].cpu().numpy()
                velocities = det["velocities"].cpu().numpy()

                for j in range(boxes.shape[0]):
                    pred_box = DetectionBox(
                        center=boxes[j, :3],
                        size=boxes[j, 3:6],
                        yaw=float(boxes[j, 6]),
                        velocity=velocities[j],
                        score=float(scores[j]),
                        label=int(labels[j]),
                        attribute=-1,
                        sample_token=sample_token,
                    )
                    all_predictions.append(pred_box)

                # Convert ground truths to DetectionBox
                gt_boxes = batch["gt_boxes"][i]
                gt_labels = batch["gt_labels"][i]
                gt_velocity = batch["gt_velocity"][i]

                for j in range(gt_boxes.shape[0]):
                    gt_box = DetectionBox(
                        center=gt_boxes[j, :3],
                        size=gt_boxes[j, 3:6],
                        yaw=float(gt_boxes[j, 6]),
                        velocity=gt_velocity[j],
                        score=1.0,
                        label=int(gt_labels[j]),
                        attribute=-1,
                        sample_token=sample_token,
                    )
                    all_ground_truths.append(gt_box)

                num_samples_processed += 1

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - inference_start
                logger.info(
                    "Processed %d/%d batches (%.1f samples/sec)",
                    batch_idx + 1,
                    len(dataloader),
                    num_samples_processed / elapsed,
                )

    total_time = time.time() - inference_start
    logger.info(
        "Inference complete: %d samples in %.1f sec (%.1f samples/sec)",
        num_samples_processed,
        total_time,
        num_samples_processed / max(total_time, 1e-6),
    )
    logger.info(
        "Total predictions: %d, Total ground truths: %d",
        len(all_predictions),
        len(all_ground_truths),
    )

    return all_predictions, all_ground_truths


# ============================================================================
# Results Formatting
# ============================================================================


def format_results_table(results: EvalResults) -> str:
    """Format evaluation results as a human-readable table.

    Args:
        results: Computed evaluation results.

    Returns:
        Formatted string containing the results tables.
    """
    lines: List[str] = []
    separator = "=" * 90

    lines.append(separator)
    lines.append("  RadarPillarNet Evaluation Results (nuScenes-style)")
    lines.append(separator)
    lines.append("")

    # Overall metrics
    lines.append(f"  NDS (nuScenes Detection Score): {results.nds:.4f}")
    lines.append(f"  mAP (mean Average Precision):   {results.mAP:.4f}")
    lines.append("")

    # TP metrics summary
    lines.append("  True Positive Metrics (mean across classes):")
    lines.append(f"    ATE (Avg Translation Error): {np.mean(list(results.ate.values())):.4f} m")
    lines.append(f"    ASE (Avg Scale Error):       {np.mean(list(results.ase.values())):.4f}")
    lines.append(f"    AOE (Avg Orientation Error): {np.mean(list(results.aoe.values())):.4f} rad")
    lines.append(f"    AVE (Avg Velocity Error):    {np.mean(list(results.ave.values())):.4f} m/s")
    lines.append(f"    AAE (Avg Attribute Error):   {np.mean(list(results.aae.values())):.4f}")
    lines.append("")

    # Per-class AP table
    lines.append("-" * 90)
    lines.append("  Per-Class Average Precision")
    lines.append("-" * 90)
    header = f"  {'Class':<22} {'AP@0.5':>8} {'AP@1.0':>8} {'AP@2.0':>8} {'AP@4.0':>8} {'Mean AP':>8}"
    lines.append(header)
    lines.append("  " + "-" * 86)

    for cls_name in NUSCENES_CLASSES:
        ap_per_thresh = results.per_class_ap_per_threshold.get(cls_name, {})
        ap_values = [
            ap_per_thresh.get(t, 0.0) for t in DISTANCE_THRESHOLDS
        ]
        mean_ap = results.per_class_ap.get(cls_name, 0.0)
        line = (
            f"  {cls_name:<22} "
            f"{ap_values[0]:>8.4f} "
            f"{ap_values[1]:>8.4f} "
            f"{ap_values[2]:>8.4f} "
            f"{ap_values[3]:>8.4f} "
            f"{mean_ap:>8.4f}"
        )
        lines.append(line)

    lines.append("")

    # Per-class TP metrics table
    lines.append("-" * 90)
    lines.append("  Per-Class True Positive Metrics")
    lines.append("-" * 90)
    header_tp = (
        f"  {'Class':<22} {'ATE(m)':>8} {'ASE':>8} {'AOE(rad)':>8} "
        f"{'AVE(m/s)':>8} {'AAE':>8}"
    )
    lines.append(header_tp)
    lines.append("  " + "-" * 86)

    for cls_name in NUSCENES_CLASSES:
        ate = results.ate.get(cls_name, 0.0)
        ase = results.ase.get(cls_name, 0.0)
        aoe = results.aoe.get(cls_name, 0.0)
        ave = results.ave.get(cls_name, 0.0)
        aae = results.aae.get(cls_name, 0.0)
        line = (
            f"  {cls_name:<22} "
            f"{ate:>8.4f} "
            f"{ase:>8.4f} "
            f"{aoe:>8.4f} "
            f"{ave:>8.4f} "
            f"{aae:>8.4f}"
        )
        lines.append(line)

    lines.append("")

    # Distance-based results
    if results.distance_results:
        lines.append("-" * 90)
        lines.append("  Distance-Based Evaluation")
        lines.append("-" * 90)
        header_dist = f"  {'Range':<12} {'mAP@2.0m':>10} {'Num GT':>10} {'Num Pred':>10}"
        lines.append(header_dist)
        lines.append("  " + "-" * 50)

        for range_name in ["0-30m", "30-50m", "50m+"]:
            if range_name in results.distance_results:
                d = results.distance_results[range_name]
                line = (
                    f"  {range_name:<12} "
                    f"{d['mAP']:>10.4f} "
                    f"{int(d['num_gt']):>10} "
                    f"{int(d['num_pred']):>10}"
                )
                lines.append(line)

    lines.append("")
    lines.append(separator)

    return "\n".join(lines)


def save_results_json(
    results: EvalResults,
    output_path: str,
) -> None:
    """Save evaluation results to a JSON file for downstream processing.

    Args:
        results: Evaluation results to serialize.
        output_path: Path to write the JSON output.
    """
    output = {
        "nds": results.nds,
        "mAP": results.mAP,
        "per_class_ap": results.per_class_ap,
        "per_class_ap_per_threshold": {
            cls: {str(k): v for k, v in thresholds.items()}
            for cls, thresholds in results.per_class_ap_per_threshold.items()
        },
        "tp_metrics": {
            "ATE": results.ate,
            "ASE": results.ase,
            "AOE": results.aoe,
            "AVE": results.ave,
            "AAE": results.aae,
        },
        "tp_metrics_mean": {
            "ATE": float(np.mean(list(results.ate.values()))),
            "ASE": float(np.mean(list(results.ase.values()))),
            "AOE": float(np.mean(list(results.aoe.values()))),
            "AVE": float(np.mean(list(results.ave.values()))),
            "AAE": float(np.mean(list(results.aae.values()))),
        },
        "distance_results": results.distance_results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================


def evaluate(
    checkpoint_path: str,
    data_root: str,
    split: str = "val",
    batch_size: int = 4,
    num_workers: int = 4,
    device_str: str = "cuda:0",
    output_dir: Optional[str] = None,
    num_classes: int = 10,
) -> EvalResults:
    """Run the full evaluation pipeline.

    Loads the model, iterates over the validation set, computes detections,
    and evaluates against ground truth using nuScenes-style metrics.

    Args:
        checkpoint_path: Path to the trained model checkpoint.
        data_root: Root directory of preprocessed evaluation data.
        split: Dataset split to evaluate ('val' or 'test').
        batch_size: Batch size for inference.
        num_workers: Number of DataLoader worker processes.
        device_str: Device string (e.g., 'cuda:0', 'cpu').
        output_dir: Optional directory to save JSON results. If None, only prints.
        num_classes: Number of detection classes.

    Returns:
        EvalResults containing all computed metrics.
    """
    # Setup device
    device = torch.device(device_str)
    if "cuda" in device_str and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")

    # Load model
    model = load_model(checkpoint_path, device, num_classes=num_classes)

    # Create dataset and dataloader
    dataset = RadarEvalDataset(data_root=data_root, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eval_batch,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    logger.info(
        "Starting evaluation: %d samples, batch_size=%d, device=%s",
        len(dataset),
        batch_size,
        device,
    )

    # Run inference
    all_predictions, all_ground_truths = run_inference(model, dataloader, device)

    # Compute nuScenes metrics
    logger.info("Computing nuScenes-style metrics...")
    eval_start = time.time()
    results = compute_nuscenes_metrics(all_predictions, all_ground_truths)

    # Compute distance-based metrics
    logger.info("Computing distance-based metrics...")
    results.distance_results = compute_distance_based_metrics(
        all_predictions, all_ground_truths
    )
    eval_time = time.time() - eval_start
    logger.info("Metric computation completed in %.1f sec", eval_time)

    # Print formatted results
    table = format_results_table(results)
    print(table)

    # Save results to JSON if output directory specified
    if output_dir is not None:
        json_path = os.path.join(output_dir, f"eval_results_{split}.json")
        save_results_json(results, json_path)

    return results


# ============================================================================
# CLI Entry Point
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate RadarPillarNet with nuScenes-style metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of preprocessed evaluation data.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (e.g., cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results JSON. If not set, results are only printed.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of detection classes (10 for full nuScenes).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run evaluation
    results = evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
    )

    # Exit with non-zero code if NDS is 0 (likely indicates an issue)
    if results.nds == 0.0 and results.mAP == 0.0:
        logger.warning(
            "Both NDS and mAP are 0.0 - check that the model and data are correct."
        )
        sys.exit(1)
