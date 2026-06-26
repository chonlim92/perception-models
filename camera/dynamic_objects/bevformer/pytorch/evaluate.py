"""Evaluation script for BEVFormer: computes nuScenes detection metrics.

Loads a trained model, runs inference on the validation set, and computes
standard nuScenes metrics including mAP, NDS, and per-class TP metrics
(ATE, ASE, AOE, AVE, AAE).

Usage:
    python evaluate.py --config ../configs/bevformer_base.yaml --checkpoint work_dirs/best.pth
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import yaml

__all__ = ["NuScenesEvaluator", "evaluate", "main"]

logger = logging.getLogger(__name__)

# nuScenes detection classes
NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


# =============================================================================
# NuScenes Evaluator
# =============================================================================


class NuScenesEvaluator:
    """Evaluator for nuScenes 3D object detection metrics.

    Computes mAP at multiple distance thresholds, NDS, and true positive
    metrics (ATE, ASE, AOE, AVE, AAE) following the official nuScenes
    detection evaluation protocol.
    """

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        distance_thresholds: Optional[List[float]] = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            classes: List of class names.
            distance_thresholds: Distance thresholds for AP computation in meters.
        """
        self.classes = classes or NUSCENES_CLASSES
        self.distance_thresholds = distance_thresholds or [0.5, 1.0, 2.0, 4.0]
        self.num_classes = len(self.classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        # Storage for predictions and ground truths
        self.predictions: List[Dict[str, np.ndarray]] = []
        self.ground_truths: List[Dict[str, np.ndarray]] = []

    def reset(self) -> None:
        """Clear all stored predictions and ground truths."""
        self.predictions.clear()
        self.ground_truths.clear()

    def add_batch(
        self,
        predictions: List[Dict[str, np.ndarray]],
        ground_truths: List[Dict[str, np.ndarray]],
    ) -> None:
        """Add a batch of predictions and ground truths.

        Args:
            predictions: List of dicts per sample, each with:
                'boxes': (K, 10) [cx, cy, cz, w, l, h, sin, cos, vx, vy]
                'scores': (K,) confidence scores
                'labels': (K,) class indices
            ground_truths: List of dicts per sample, each with:
                'boxes': (M, 10) [cx, cy, cz, w, l, h, sin, cos, vx, vy]
                'labels': (M,) class indices
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all nuScenes detection metrics.

        Returns:
            Dict containing:
                'mAP': mean Average Precision
                'NDS': nuScenes Detection Score
                'mATE': mean Average Translation Error
                'mASE': mean Average Scale Error
                'mAOE': mean Average Orientation Error
                'mAVE': mean Average Velocity Error
                'mAAE': mean Average Attribute Error
                'per_class': dict of per-class metrics
                'ap_per_class_threshold': per-class AP at each threshold
        """
        num_samples = len(self.predictions)
        if num_samples == 0:
            logger.warning("No predictions to evaluate")
            return self._empty_metrics()

        # Compute AP for each class at each distance threshold
        ap_matrix = np.zeros((self.num_classes, len(self.distance_thresholds)))
        tp_metrics_per_class = {
            cls: {"ATE": [], "ASE": [], "AOE": [], "AVE": [], "AAE": []}
            for cls in self.classes
        }

        for cls_idx, cls_name in enumerate(self.classes):
            for thresh_idx, threshold in enumerate(self.distance_thresholds):
                ap, tp_errors = self._compute_ap_for_class(
                    cls_idx, threshold
                )
                ap_matrix[cls_idx, thresh_idx] = ap

                # Collect TP metrics from the smallest threshold (most strict matching)
                if thresh_idx == 1:  # Use 1.0m threshold for TP metrics
                    for key, values in tp_errors.items():
                        tp_metrics_per_class[cls_name][key].extend(values)

        # mAP: mean over classes and thresholds
        mAP = float(np.mean(ap_matrix))

        # Per-class AP (mean over thresholds)
        per_class_ap = {
            cls: float(np.mean(ap_matrix[i]))
            for i, cls in enumerate(self.classes)
        }

        # TP metrics: mean over all matched TPs per class, then mean over classes
        mean_tp_metrics = {}
        per_class_tp = {}

        for metric_name in ["ATE", "ASE", "AOE", "AVE", "AAE"]:
            class_means = []
            for cls_name in self.classes:
                values = tp_metrics_per_class[cls_name][metric_name]
                if values:
                    cls_mean = float(np.mean(values))
                else:
                    cls_mean = 1.0  # Worst case if no TPs
                class_means.append(cls_mean)
                per_class_tp.setdefault(cls_name, {})[metric_name] = cls_mean

            mean_tp_metrics[f"m{metric_name}"] = float(np.mean(class_means))

        # NDS = 1/10 * [5*mAP + sum(max(1 - metric, 0) for 5 TP metrics)]
        tp_score_sum = sum(
            max(1.0 - mean_tp_metrics[f"m{m}"], 0.0)
            for m in ["ATE", "ASE", "AOE", "AVE", "AAE"]
        )
        NDS = (5.0 * mAP + tp_score_sum) / 10.0

        results = {
            "mAP": mAP,
            "NDS": NDS,
            **mean_tp_metrics,
            "per_class_ap": per_class_ap,
            "per_class_tp": per_class_tp,
            "ap_matrix": ap_matrix.tolist(),
            "distance_thresholds": self.distance_thresholds,
            "num_samples": num_samples,
        }

        return results

    def _compute_ap_for_class(
        self, cls_idx: int, distance_threshold: float
    ) -> Tuple[float, Dict[str, List[float]]]:
        """Compute AP for a single class at a single distance threshold.

        Uses center distance matching with greedy assignment (highest score first).

        Args:
            cls_idx: Class index.
            distance_threshold: Maximum center distance for a match (meters).

        Returns:
            Tuple of (AP value, dict of TP error lists).
        """
        # Collect all predictions and GTs for this class
        all_scores = []
        all_pred_boxes = []
        all_pred_sample_ids = []

        all_gt_boxes = []
        all_gt_sample_ids = []

        for sample_idx in range(len(self.predictions)):
            pred = self.predictions[sample_idx]
            gt = self.ground_truths[sample_idx]

            # Predictions for this class
            if pred["labels"].size > 0:
                pred_mask = pred["labels"] == cls_idx
                if pred_mask.any():
                    pred_boxes = pred["boxes"][pred_mask]
                    pred_scores = pred["scores"][pred_mask]
                    all_scores.extend(pred_scores.tolist())
                    all_pred_boxes.extend(pred_boxes.tolist())
                    all_pred_sample_ids.extend([sample_idx] * int(pred_mask.sum()))

            # Ground truths for this class
            if gt["labels"].size > 0:
                gt_mask = gt["labels"] == cls_idx
                if gt_mask.any():
                    gt_boxes = gt["boxes"][gt_mask]
                    all_gt_boxes.extend(gt_boxes.tolist())
                    all_gt_sample_ids.extend([sample_idx] * int(gt_mask.sum()))

        num_gt = len(all_gt_boxes)
        if num_gt == 0:
            return 0.0, {"ATE": [], "ASE": [], "AOE": [], "AVE": [], "AAE": []}

        if len(all_scores) == 0:
            return 0.0, {"ATE": [], "ASE": [], "AOE": [], "AVE": [], "AAE": []}

        # Sort predictions by score (descending)
        all_scores = np.array(all_scores)
        all_pred_boxes = np.array(all_pred_boxes)
        all_pred_sample_ids = np.array(all_pred_sample_ids)
        all_gt_boxes = np.array(all_gt_boxes)
        all_gt_sample_ids = np.array(all_gt_sample_ids)

        sorted_indices = np.argsort(-all_scores)
        all_scores = all_scores[sorted_indices]
        all_pred_boxes = all_pred_boxes[sorted_indices]
        all_pred_sample_ids = all_pred_sample_ids[sorted_indices]

        # Greedy matching
        gt_matched = np.zeros(num_gt, dtype=bool)
        tp = np.zeros(len(all_scores), dtype=bool)
        tp_errors: Dict[str, List[float]] = {
            "ATE": [], "ASE": [], "AOE": [], "AVE": [], "AAE": []
        }

        for pred_idx in range(len(all_scores)):
            pred_box = all_pred_boxes[pred_idx]
            pred_sample = all_pred_sample_ids[pred_idx]

            # Find GT boxes in the same sample
            gt_in_sample = np.where(all_gt_sample_ids == pred_sample)[0]
            if len(gt_in_sample) == 0:
                continue

            # Compute center distance to unmatched GTs in this sample
            best_dist = float("inf")
            best_gt_idx = -1

            for gt_idx in gt_in_sample:
                if gt_matched[gt_idx]:
                    continue
                gt_box = all_gt_boxes[gt_idx]
                # 2D center distance (x, y only)
                dist = np.sqrt(
                    (pred_box[0] - gt_box[0]) ** 2
                    + (pred_box[1] - gt_box[1]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = gt_idx

            if best_dist <= distance_threshold and best_gt_idx >= 0:
                tp[pred_idx] = True
                gt_matched[best_gt_idx] = True

                # Compute TP errors
                gt_box = all_gt_boxes[best_gt_idx]
                errors = self._compute_tp_errors(pred_box, gt_box)
                for key, val in errors.items():
                    tp_errors[key].append(val)

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(~tp)
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # AP using 40-point interpolation (nuScenes style)
        ap = self._compute_ap_40point(recalls, precisions)

        return ap, tp_errors

    def _compute_tp_errors(
        self, pred_box: np.ndarray, gt_box: np.ndarray
    ) -> Dict[str, float]:
        """Compute true positive errors between a matched prediction and GT.

        Both boxes format: [cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy]

        Args:
            pred_box: Predicted box (10,).
            gt_box: Ground truth box (10,).

        Returns:
            Dict with ATE, ASE, AOE, AVE, AAE error values.
        """
        # ATE: 2D Euclidean center distance (x, y)
        ate = float(np.sqrt(
            (pred_box[0] - gt_box[0]) ** 2 + (pred_box[1] - gt_box[1]) ** 2
        ))

        # ASE: 1 - IoU of the volume (approximated as 1 - scale similarity)
        # Scale similarity = min(pred, gt) / max(pred, gt) for each dimension
        pred_size = np.abs(pred_box[3:6])  # w, l, h
        gt_size = np.abs(gt_box[3:6])

        # Volume IoU approximation
        pred_vol = float(np.prod(np.maximum(pred_size, 0.01)))
        gt_vol = float(np.prod(np.maximum(gt_size, 0.01)))
        inter_size = np.minimum(pred_size, gt_size)
        inter_vol = float(np.prod(np.maximum(inter_size, 0.01)))
        iou = inter_vol / (pred_vol + gt_vol - inter_vol + 1e-8)
        ase = 1.0 - iou

        # AOE: Smallest angle difference between predicted and GT yaw
        pred_yaw = float(np.arctan2(pred_box[6], pred_box[7]))
        gt_yaw = float(np.arctan2(gt_box[6], gt_box[7]))
        aoe = float(np.abs(self._angle_diff(pred_yaw, gt_yaw)))

        # AVE: L2 of velocity difference
        pred_vel = pred_box[8:10]
        gt_vel = gt_box[8:10]
        ave = float(np.sqrt(np.sum((pred_vel - gt_vel) ** 2)))

        # AAE: Attribute error (set to 0 since we don't predict attributes)
        aae = 0.0

        return {"ATE": ate, "ASE": ase, "AOE": aoe, "AVE": ave, "AAE": aae}

    @staticmethod
    def _angle_diff(angle1: float, angle2: float) -> float:
        """Compute smallest angular difference.

        Args:
            angle1: First angle in radians.
            angle2: Second angle in radians.

        Returns:
            Signed angular difference in [-pi, pi].
        """
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    @staticmethod
    def _compute_ap_40point(
        recalls: np.ndarray, precisions: np.ndarray
    ) -> float:
        """Compute AP using 40-point interpolation (nuScenes style).

        Args:
            recalls: Cumulative recall values.
            precisions: Cumulative precision values.

        Returns:
            Average Precision value.
        """
        if len(recalls) == 0:
            return 0.0

        # 40 recall points from 0 to 1
        recall_thresholds = np.linspace(0, 1, 41)[1:]  # Exclude 0

        ap = 0.0
        for r_thresh in recall_thresholds:
            # Find max precision at recall >= threshold
            mask = recalls >= r_thresh
            if mask.any():
                ap += float(np.max(precisions[mask]))

        ap /= len(recall_thresholds)
        return ap

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dict when no predictions exist."""
        return {
            "mAP": 0.0,
            "NDS": 0.0,
            "mATE": 1.0,
            "mASE": 1.0,
            "mAOE": 1.0,
            "mAVE": 1.0,
            "mAAE": 1.0,
            "per_class_ap": {cls: 0.0 for cls in self.classes},
            "per_class_tp": {
                cls: {"ATE": 1.0, "ASE": 1.0, "AOE": 1.0, "AVE": 1.0, "AAE": 1.0}
                for cls in self.classes
            },
            "num_samples": 0,
        }


# =============================================================================
# Temporal Consistency Evaluation
# =============================================================================


class TemporalConsistencyEvaluator:
    """Evaluates prediction consistency across consecutive frames.

    Measures how smoothly detections evolve over time by tracking
    prediction centers and computing jitter metrics.
    """

    def __init__(
        self,
        distance_threshold: float = 2.0,
        classes: Optional[List[str]] = None,
    ) -> None:
        """Initialize temporal consistency evaluator.

        Args:
            distance_threshold: Max distance to associate detections across frames.
            classes: Class names.
        """
        self.distance_threshold = distance_threshold
        self.classes = classes or NUSCENES_CLASSES

        self.frame_predictions: List[Dict[str, np.ndarray]] = []
        self.position_diffs: List[float] = []
        self.velocity_diffs: List[float] = []

    def add_frame(self, predictions: Dict[str, np.ndarray]) -> None:
        """Add predictions from one frame.

        Args:
            predictions: Dict with 'boxes', 'scores', 'labels'.
        """
        if self.frame_predictions:
            prev = self.frame_predictions[-1]
            self._compute_consistency(prev, predictions)

        self.frame_predictions.append(predictions)

    def _compute_consistency(
        self,
        prev_preds: Dict[str, np.ndarray],
        curr_preds: Dict[str, np.ndarray],
    ) -> None:
        """Compute consistency between consecutive frame predictions."""
        if prev_preds["boxes"].size == 0 or curr_preds["boxes"].size == 0:
            return

        prev_centers = prev_preds["boxes"][:, :2]  # (N, 2)
        curr_centers = curr_preds["boxes"][:, :2]  # (M, 2)

        # Simple nearest-neighbor association
        for i in range(len(curr_centers)):
            dists = np.sqrt(np.sum((prev_centers - curr_centers[i:i+1]) ** 2, axis=1))
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]

            if min_dist < self.distance_threshold:
                self.position_diffs.append(min_dist)

                # Velocity consistency
                prev_vel = prev_preds["boxes"][min_idx, 8:10]
                curr_vel = curr_preds["boxes"][i, 8:10]
                vel_diff = float(np.sqrt(np.sum((prev_vel - curr_vel) ** 2)))
                self.velocity_diffs.append(vel_diff)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute temporal consistency metrics.

        Returns:
            Dict with consistency metrics.
        """
        if not self.position_diffs:
            return {
                "mean_position_jitter": 0.0,
                "mean_velocity_jitter": 0.0,
                "num_associations": 0,
            }

        return {
            "mean_position_jitter": float(np.mean(self.position_diffs)),
            "std_position_jitter": float(np.std(self.position_diffs)),
            "mean_velocity_jitter": float(np.mean(self.velocity_diffs)),
            "std_velocity_jitter": float(np.std(self.velocity_diffs)),
            "num_associations": len(self.position_diffs),
            "num_frames": len(self.frame_predictions),
        }


# =============================================================================
# Print Results
# =============================================================================


def print_results_table(results: Dict[str, Any]) -> None:
    """Print evaluation results in a formatted table.

    Args:
        results: Metrics dict from NuScenesEvaluator.compute_metrics().
    """
    print("\n" + "=" * 70)
    print("  nuScenes Detection Evaluation Results")
    print("=" * 70)

    # Overall metrics
    print(f"\n{'Metric':<15} {'Value':>10}")
    print("-" * 30)
    print(f"{'mAP':<15} {results['mAP']:>10.4f}")
    print(f"{'NDS':<15} {results['NDS']:>10.4f}")
    print(f"{'mATE':<15} {results.get('mATE', 0):>10.4f}")
    print(f"{'mASE':<15} {results.get('mASE', 0):>10.4f}")
    print(f"{'mAOE':<15} {results.get('mAOE', 0):>10.4f}")
    print(f"{'mAVE':<15} {results.get('mAVE', 0):>10.4f}")
    print(f"{'mAAE':<15} {results.get('mAAE', 0):>10.4f}")

    # Per-class AP
    if "per_class_ap" in results:
        print(f"\n{'Class':<25} {'AP':>8}")
        print("-" * 35)
        for cls_name, ap in results["per_class_ap"].items():
            print(f"  {cls_name:<23} {ap:>8.4f}")

    # Per-class TP metrics
    if "per_class_tp" in results:
        print(f"\n{'Class':<20} {'ATE':>7} {'ASE':>7} {'AOE':>7} {'AVE':>7} {'AAE':>7}")
        print("-" * 60)
        for cls_name, tp_dict in results["per_class_tp"].items():
            print(
                f"  {cls_name:<18} "
                f"{tp_dict['ATE']:>7.3f} "
                f"{tp_dict['ASE']:>7.3f} "
                f"{tp_dict['AOE']:>7.3f} "
                f"{tp_dict['AVE']:>7.3f} "
                f"{tp_dict['AAE']:>7.3f}"
            )

    # AP at different thresholds
    if "ap_matrix" in results:
        thresholds = results.get("distance_thresholds", [0.5, 1.0, 2.0, 4.0])
        classes = list(results.get("per_class_ap", {}).keys())
        ap_matrix = results["ap_matrix"]

        if classes and ap_matrix:
            header = f"{'Class':<20}" + "".join(f"{'d=' + str(t) + 'm':>10}" for t in thresholds)
            print(f"\n{header}")
            print("-" * (20 + 10 * len(thresholds)))
            for i, cls_name in enumerate(classes):
                row = f"  {cls_name:<18}"
                for j in range(len(thresholds)):
                    row += f"{ap_matrix[i][j]:>10.4f}"
                print(row)

    print("\n" + "=" * 70)


# =============================================================================
# Main Evaluation Function
# =============================================================================


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """Run evaluation on the validation set.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Results dictionary.
    """
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override data root if specified
    if args.data_root:
        config["data"]["data_root"] = args.data_root

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build model
    from .model import BEVFormer
    from .dataset import NuScenesDataset, collate_fn

    model_cfg = config["model"]
    bev_cfg = model_cfg["bev_encoder"]
    decoder_cfg = model_cfg["decoder"]
    head_cfg = model_cfg["head"]
    loss_cfg = config["loss"]

    model = BEVFormer(
        backbone_out_channels=model_cfg["neck"]["out_channels"],
        backbone_pretrained=False,  # Don't need pretrained for eval
        backbone_frozen_stages=model_cfg["backbone"]["frozen_stages"],
        embed_dim=bev_cfg["embed_dims"],
        bev_h=bev_cfg["bev_h"],
        bev_w=bev_cfg["bev_w"],
        num_encoder_layers=bev_cfg["num_encoder_layers"],
        num_heads=bev_cfg["num_heads"],
        num_points_spatial=bev_cfg["num_points_spatial"],
        num_points_temporal=bev_cfg["num_points_temporal"],
        num_levels=bev_cfg["num_levels"],
        num_cams=bev_cfg["num_cams"],
        num_ref_points=bev_cfg["num_ref_points"],
        pc_range=tuple(bev_cfg["pc_range"]),
        num_decoder_layers=decoder_cfg["num_decoder_layers"],
        num_queries=decoder_cfg["num_queries"],
        ffn_dim=decoder_cfg["ffn_dim"],
        dropout=decoder_cfg["dropout"],
        iterative_bbox_refinement=decoder_cfg["iterative_bbox_refinement"],
        num_classes=head_cfg["num_classes"],
        code_size=head_cfg["code_size"],
        num_reg_fcs=head_cfg["num_reg_fcs"],
        cls_weight=loss_cfg["cls_loss"]["weight"],
        bbox_weight=loss_cfg["bbox_loss"]["weight"],
        focal_alpha=loss_cfg["cls_loss"]["alpha"],
        focal_gamma=loss_cfg["cls_loss"]["gamma"],
        cls_cost=loss_cfg["matcher"]["cls_cost"],
        bbox_cost=loss_cfg["matcher"]["bbox_cost"],
        score_threshold=config["evaluation"]["nms"]["score_threshold"],
        max_detections=config["evaluation"]["nms"]["max_per_frame"],
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    # Build validation dataset
    data_cfg = config["data"]
    val_dataset = NuScenesDataset(
        data_root=data_cfg["data_root"],
        ann_file=data_cfg["val_ann"],
        img_size=tuple(data_cfg["img_size"]),
        num_temporal_frames=data_cfg["num_temporal_frames"],
        classes=data_cfg["classes"],
        augmentation_cfg=data_cfg.get("augmentation"),
        is_train=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    logger.info(f"Validation set: {len(val_dataset)} samples")

    # Run inference
    evaluator = NuScenesEvaluator(classes=data_cfg["classes"])
    temporal_evaluator = TemporalConsistencyEvaluator(classes=data_cfg["classes"])

    prev_bev = None
    total_time = 0.0
    num_frames = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["images"].to(device, non_blocking=True)
            intrinsics = batch["intrinsics"].to(device, non_blocking=True)
            extrinsics = batch["extrinsics"].to(device, non_blocking=True)
            ego_motion = batch["ego_motion"].to(device, non_blocking=True)
            prev_exists = batch["prev_exists"]

            # Reset prev_bev for new sequences
            if not all(prev_exists):
                prev_bev = None

            # Inference with timing
            start_time = time.time()
            with torch.amp.autocast("cuda", enabled=True):
                detections, new_bev = model.forward_test(
                    images=images,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    ego_motion=ego_motion,
                    prev_bev=prev_bev,
                )
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            total_time += elapsed
            num_frames += images.shape[0]

            prev_bev = new_bev.detach()

            # Convert to numpy for evaluation
            batch_size = images.shape[0]
            batch_preds = []
            batch_gts = []

            for b in range(batch_size):
                num_dets = int(detections["num_detections"][b].item())
                pred_dict = {
                    "boxes": detections["boxes"][b, :num_dets].cpu().numpy(),
                    "scores": detections["scores"][b, :num_dets].cpu().numpy(),
                    "labels": detections["labels"][b, :num_dets].cpu().numpy(),
                }
                batch_preds.append(pred_dict)

                # Ground truth
                gt_labels = batch["gt_labels"][b].numpy()
                valid = gt_labels >= 0
                gt_dict = {
                    "boxes": batch["gt_bboxes_3d"][b][valid].numpy(),
                    "labels": gt_labels[valid],
                }
                batch_gts.append(gt_dict)

                # Temporal consistency
                temporal_evaluator.add_frame(pred_dict)

            evaluator.add_batch(batch_preds, batch_gts)

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"Evaluated {batch_idx + 1}/{len(val_loader)} batches "
                    f"({num_frames / total_time:.1f} FPS)"
                )

    # Compute metrics
    logger.info("Computing metrics...")
    results = evaluator.compute_metrics()
    temporal_results = temporal_evaluator.compute_metrics()
    results["temporal_consistency"] = temporal_results

    # Timing
    if num_frames > 0:
        results["fps"] = num_frames / total_time
        results["avg_latency_ms"] = (total_time / num_frames) * 1000.0
    else:
        results["fps"] = 0.0
        results["avg_latency_ms"] = 0.0

    # Print results
    print_results_table(results)

    # Print temporal consistency
    print(f"\n{'Temporal Consistency':}")
    print(f"  Position jitter: {temporal_results.get('mean_position_jitter', 0):.3f}m")
    print(f"  Velocity jitter: {temporal_results.get('mean_velocity_jitter', 0):.3f}m/s")
    print(f"  Associations: {temporal_results.get('num_associations', 0)}")
    print(f"\n  Inference speed: {results['fps']:.1f} FPS ({results['avg_latency_ms']:.1f}ms)")

    # Save results to JSON
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # Convert numpy types for JSON serialization
        json_results = _to_json_serializable(results)
        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return results


def _to_json_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable version of the object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


# =============================================================================
# Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BEVFormer Evaluation Script"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override data root directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Inference batch size"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader workers"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID"
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
