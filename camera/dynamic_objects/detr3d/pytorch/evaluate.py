"""DETR3D evaluation script implementing nuScenes-style detection metrics.

Computes per-class Average Precision (AP) at BEV center distance thresholds,
mean AP (mAP), True Positive metrics (ATE, ASE, AOE, AVE, AAE), and the
nuScenes Detection Score (NDS).

Usage:
    python evaluate.py --checkpoint /path/to/checkpoint.pth \
                       --data_root /path/to/nuscenes \
                       --batch_size 4 \
                       --output_file results.json
"""

import argparse
import json
import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DETR3D, DETR3DPostProcessor
from dataset import NuScenesDataset, collate_fn, CATEGORY_MAP, PC_RANGE, CODE_SIZE


# Default evaluation configuration
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
DISTANCE_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]
DEFAULT_SCORE_THRESHOLD = 0.1
DEFAULT_MAX_DETECTIONS = 300


class NuScenesEvaluator:
    """Evaluator implementing nuScenes-style detection metrics.

    Computes:
        - Per-class AP at BEV center distance thresholds [0.5, 1.0, 2.0, 4.0]m
        - mAP (mean over all classes and thresholds)
        - True Positive metrics at 2.0m threshold:
            * ATE: Average Translation Error (3D Euclidean distance)
            * ASE: Average Scale Error (1 - volume IoU approximation)
            * AOE: Average Orientation Error (smallest yaw angle difference)
            * AVE: Average Velocity Error (L2 of velocity difference)
            * AAE: Average Attribute Error (simplified to 0)
        - NDS: nuScenes Detection Score

    Args:
        class_names: List of class names in order of class indices.
        distance_thresholds: List of BEV center distance thresholds in meters.
        tp_threshold: Distance threshold for True Positive metric computation.
    """

    def __init__(
        self,
        class_names: List[str] = None,
        distance_thresholds: List[float] = None,
        tp_threshold: float = 2.0,
    ):
        self.class_names = class_names if class_names is not None else CLASS_NAMES
        self.distance_thresholds = (
            distance_thresholds if distance_thresholds is not None
            else DISTANCE_THRESHOLDS
        )
        self.tp_threshold = tp_threshold
        self.num_classes = len(self.class_names)

    def evaluate(
        self,
        predictions: List[Dict[str, np.ndarray]],
        ground_truths: List[Dict[str, np.ndarray]],
    ) -> Dict:
        """Run full nuScenes-style evaluation.

        Args:
            predictions: List of prediction dicts per sample, each containing:
                - 'scores': (N,) confidence scores
                - 'labels': (N,) predicted class indices
                - 'boxes': (N, 10) predicted box parameters
                    (cx, cy, cz, w, l, h, sin, cos, vx, vy)
            ground_truths: List of GT dicts per sample, each containing:
                - 'labels': (M,) class indices
                - 'boxes': (M, 10) GT box parameters

        Returns:
            Dictionary with evaluation results including per-class AP,
            mAP, TP metrics, and NDS.
        """
        # Organize predictions and ground truths by class
        per_class_preds = self._organize_by_class(predictions, is_gt=False)
        per_class_gts = self._organize_by_class(ground_truths, is_gt=True)

        # Compute per-class AP at each distance threshold
        ap_results = {}  # {class_name: {threshold: ap_value}}
        for class_idx, class_name in enumerate(self.class_names):
            ap_results[class_name] = {}
            for threshold in self.distance_thresholds:
                ap = self._compute_ap(
                    per_class_preds[class_idx],
                    per_class_gts[class_idx],
                    threshold,
                )
                ap_results[class_name][threshold] = ap

        # Compute mAP (mean over all classes and thresholds)
        all_aps = []
        for class_name in self.class_names:
            for threshold in self.distance_thresholds:
                all_aps.append(ap_results[class_name][threshold])
        mAP = float(np.mean(all_aps)) if all_aps else 0.0

        # Compute True Positive metrics at the TP threshold (2.0m)
        tp_metrics = self._compute_tp_metrics(
            per_class_preds, per_class_gts, self.tp_threshold
        )

        # Compute NDS
        nds = self._compute_nds(mAP, tp_metrics)

        # Per-class mean AP (averaged over thresholds)
        per_class_mAP = {}
        for class_name in self.class_names:
            class_aps = [
                ap_results[class_name][t] for t in self.distance_thresholds
            ]
            per_class_mAP[class_name] = float(np.mean(class_aps))

        return {
            'mAP': mAP,
            'NDS': nds,
            'per_class_AP': ap_results,
            'per_class_mAP': per_class_mAP,
            'tp_metrics': tp_metrics,
            'distance_thresholds': self.distance_thresholds,
        }

    def _organize_by_class(
        self,
        data: List[Dict[str, np.ndarray]],
        is_gt: bool,
    ) -> Dict[int, List[Dict[str, np.ndarray]]]:
        """Organize predictions or GTs by class, preserving sample indices.

        Args:
            data: List of per-sample dicts.
            is_gt: Whether this is ground truth data.

        Returns:
            Dict mapping class_idx to list of per-sample entries.
            Each entry: {'boxes': (K, 10), 'scores': (K,) [preds only],
                         'sample_idx': int}
        """
        per_class = defaultdict(list)

        for sample_idx, sample_data in enumerate(data):
            labels = sample_data['labels']
            boxes = sample_data['boxes']

            if is_gt:
                for class_idx in range(self.num_classes):
                    mask = labels == class_idx
                    per_class[class_idx].append({
                        'boxes': boxes[mask],
                        'sample_idx': sample_idx,
                    })
            else:
                scores = sample_data['scores']
                for class_idx in range(self.num_classes):
                    mask = labels == class_idx
                    per_class[class_idx].append({
                        'boxes': boxes[mask],
                        'scores': scores[mask],
                        'sample_idx': sample_idx,
                    })

        return per_class

    def _compute_ap(
        self,
        class_preds: List[Dict[str, np.ndarray]],
        class_gts: List[Dict[str, np.ndarray]],
        distance_threshold: float,
    ) -> float:
        """Compute Average Precision for one class at one distance threshold.

        Uses BEV center distance (2D Euclidean of cx, cy) for matching.
        Greedy matching by descending confidence score.

        Args:
            class_preds: List of per-sample prediction dicts for this class.
            class_gts: List of per-sample GT dicts for this class.
            distance_threshold: Maximum BEV center distance for a match.

        Returns:
            AP value (float between 0 and 1).
        """
        # Count total number of ground truths
        total_gt = sum(entry['boxes'].shape[0] for entry in class_gts)
        if total_gt == 0:
            return 0.0

        # Gather all predictions across samples with their sample index
        all_preds = []
        for entry in class_preds:
            scores = entry['scores']
            boxes = entry['boxes']
            sample_idx = entry['sample_idx']
            for i in range(len(scores)):
                all_preds.append({
                    'score': scores[i],
                    'box': boxes[i],
                    'sample_idx': sample_idx,
                })

        if len(all_preds) == 0:
            return 0.0

        # Sort predictions by confidence (descending)
        all_preds.sort(key=lambda x: x['score'], reverse=True)

        # Track which GT boxes have been matched per sample
        gt_matched = {}
        for entry in class_gts:
            sample_idx = entry['sample_idx']
            num_gt = entry['boxes'].shape[0]
            gt_matched[sample_idx] = np.zeros(num_gt, dtype=bool)

        # Build a quick lookup for GT boxes per sample
        gt_boxes_by_sample = {}
        for entry in class_gts:
            gt_boxes_by_sample[entry['sample_idx']] = entry['boxes']

        # Compute TP/FP for each prediction
        tp = np.zeros(len(all_preds), dtype=np.float64)
        fp = np.zeros(len(all_preds), dtype=np.float64)

        for pred_idx, pred in enumerate(all_preds):
            sample_idx = pred['sample_idx']
            pred_box = pred['box']
            gt_boxes = gt_boxes_by_sample.get(sample_idx)

            if gt_boxes is None or gt_boxes.shape[0] == 0:
                fp[pred_idx] = 1.0
                continue

            # Compute BEV center distance to all GT boxes in the same sample
            # BEV distance uses cx (index 0) and cy (index 1)
            pred_center = pred_box[:2]  # (cx, cy)
            gt_centers = gt_boxes[:, :2]  # (num_gt, 2)
            distances = np.linalg.norm(gt_centers - pred_center, axis=1)

            # Find the closest unmatched GT
            sorted_gt_indices = np.argsort(distances)
            matched = False
            for gt_idx in sorted_gt_indices:
                if distances[gt_idx] > distance_threshold:
                    break
                if not gt_matched[sample_idx][gt_idx]:
                    # Match found
                    tp[pred_idx] = 1.0
                    gt_matched[sample_idx][gt_idx] = True
                    matched = True
                    break

            if not matched:
                fp[pred_idx] = 1.0

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / total_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Compute AP using all-point interpolation (nuScenes style)
        ap = self._compute_ap_from_pr(precision, recall)
        return ap

    def _compute_ap_from_pr(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
    ) -> float:
        """Compute AP from precision-recall curve using all-point interpolation.

        The nuScenes evaluation uses a 40-point interpolation (recall from 0 to 1
        in steps of 1/40). We sample precision at 101 recall thresholds for
        smoother AP computation consistent with COCO-style.

        Args:
            precision: Precision values at each detection.
            recall: Recall values at each detection.

        Returns:
            Average Precision value.
        """
        if len(precision) == 0:
            return 0.0

        # Prepend start point (recall=0, precision=1)
        recall = np.concatenate(([0.0], recall))
        precision = np.concatenate(([1.0], precision))

        # Make precision monotonically decreasing (right to left)
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        # Sample at 101 recall thresholds (0.0, 0.01, ..., 1.0)
        recall_thresholds = np.linspace(0.0, 1.0, 101)
        ap = 0.0
        for t in recall_thresholds:
            # Find the maximum precision at recall >= t
            mask = recall >= t
            if mask.any():
                ap += precision[mask].max()

        ap /= len(recall_thresholds)
        return float(ap)

    def _compute_tp_metrics(
        self,
        per_class_preds: Dict[int, List[Dict[str, np.ndarray]]],
        per_class_gts: Dict[int, List[Dict[str, np.ndarray]]],
        threshold: float,
    ) -> Dict[str, float]:
        """Compute True Positive metrics on matched predictions.

        Metrics are computed on predictions that match a GT at the given
        BEV center distance threshold. Results are averaged over all classes.

        Args:
            per_class_preds: Predictions organized by class.
            per_class_gts: Ground truths organized by class.
            threshold: BEV center distance threshold for matching.

        Returns:
            Dict with ATE, ASE, AOE, AVE, AAE values.
        """
        all_ate = []
        all_ase = []
        all_aoe = []
        all_ave = []

        for class_idx in range(self.num_classes):
            class_preds = per_class_preds[class_idx]
            class_gts = per_class_gts[class_idx]

            # Get matched pairs for this class
            matches = self._get_tp_matches(class_preds, class_gts, threshold)

            for pred_box, gt_box in matches:
                # ATE: 3D Euclidean distance of centers
                ate = np.linalg.norm(pred_box[:3] - gt_box[:3])
                all_ate.append(ate)

                # ASE: 1 - volume_IoU_approx
                # Approximate volume IoU as min(vol_pred, vol_gt) / max(vol_pred, vol_gt)
                pred_vol = pred_box[3] * pred_box[4] * pred_box[5]  # w * l * h
                gt_vol = gt_box[3] * gt_box[4] * gt_box[5]
                if max(pred_vol, gt_vol) > 0:
                    vol_iou = min(pred_vol, gt_vol) / max(pred_vol, gt_vol)
                else:
                    vol_iou = 0.0
                ase = 1.0 - vol_iou
                all_ase.append(ase)

                # AOE: Angular difference using sin/cos encoding
                # pred_box[6] = sin(yaw_pred), pred_box[7] = cos(yaw_pred)
                pred_yaw = math.atan2(pred_box[6], pred_box[7])
                gt_yaw = math.atan2(gt_box[6], gt_box[7])
                aoe = self._angle_diff(pred_yaw, gt_yaw)
                all_aoe.append(aoe)

                # AVE: L2 norm of velocity difference (vx, vy)
                pred_vel = pred_box[8:10]
                gt_vel = gt_box[8:10]
                ave = np.linalg.norm(pred_vel - gt_vel)
                all_ave.append(ave)

        # Average metrics (use nan-safe computation)
        tp_metrics = {
            'ATE': float(np.mean(all_ate)) if all_ate else 1.0,
            'ASE': float(np.mean(all_ase)) if all_ase else 1.0,
            'AOE': float(np.mean(all_aoe)) if all_aoe else 1.0,
            'AVE': float(np.mean(all_ave)) if all_ave else 1.0,
            'AAE': 0.0,  # Simplified: attributes not predicted
        }
        return tp_metrics

    def _get_tp_matches(
        self,
        class_preds: List[Dict[str, np.ndarray]],
        class_gts: List[Dict[str, np.ndarray]],
        threshold: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get matched (pred_box, gt_box) pairs for TP metric computation.

        Greedy matching by confidence score, same as AP computation.

        Args:
            class_preds: Per-sample predictions for one class.
            class_gts: Per-sample GTs for one class.
            threshold: BEV center distance threshold.

        Returns:
            List of (pred_box, gt_box) tuples for matched detections.
        """
        # Gather all predictions sorted by confidence
        all_preds = []
        for entry in class_preds:
            scores = entry['scores']
            boxes = entry['boxes']
            sample_idx = entry['sample_idx']
            for i in range(len(scores)):
                all_preds.append({
                    'score': scores[i],
                    'box': boxes[i],
                    'sample_idx': sample_idx,
                })

        if len(all_preds) == 0:
            return []

        all_preds.sort(key=lambda x: x['score'], reverse=True)

        # Track matched GTs
        gt_matched = {}
        gt_boxes_by_sample = {}
        for entry in class_gts:
            sample_idx = entry['sample_idx']
            gt_boxes_by_sample[sample_idx] = entry['boxes']
            gt_matched[sample_idx] = np.zeros(entry['boxes'].shape[0], dtype=bool)

        matches = []
        for pred in all_preds:
            sample_idx = pred['sample_idx']
            pred_box = pred['box']
            gt_boxes = gt_boxes_by_sample.get(sample_idx)

            if gt_boxes is None or gt_boxes.shape[0] == 0:
                continue

            # BEV center distance
            pred_center = pred_box[:2]
            gt_centers = gt_boxes[:, :2]
            distances = np.linalg.norm(gt_centers - pred_center, axis=1)

            sorted_gt_indices = np.argsort(distances)
            for gt_idx in sorted_gt_indices:
                if distances[gt_idx] > threshold:
                    break
                if not gt_matched[sample_idx][gt_idx]:
                    gt_matched[sample_idx][gt_idx] = True
                    matches.append((pred_box, gt_boxes[gt_idx]))
                    break

        return matches

    @staticmethod
    def _angle_diff(angle1: float, angle2: float) -> float:
        """Compute the smallest angular difference between two angles.

        Args:
            angle1: First angle in radians.
            angle2: Second angle in radians.

        Returns:
            Absolute angular difference in [0, pi].
        """
        diff = abs(angle1 - angle2)
        diff = diff % (2 * math.pi)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        return diff

    @staticmethod
    def _compute_nds(mAP: float, tp_metrics: Dict[str, float]) -> float:
        """Compute the nuScenes Detection Score (NDS).

        NDS = 1/10 * [5 * mAP + sum(max(1 - metric, 0) for each of 5 TP metrics)]

        Args:
            mAP: Mean Average Precision value.
            tp_metrics: Dict with ATE, ASE, AOE, AVE, AAE.

        Returns:
            NDS value.
        """
        tp_scores = []
        for metric_name in ['ATE', 'ASE', 'AOE', 'AVE', 'AAE']:
            metric_val = tp_metrics[metric_name]
            tp_scores.append(max(1.0 - metric_val, 0.0))

        nds = (5.0 * mAP + sum(tp_scores)) / 10.0
        return float(nds)


def decode_predictions(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    pc_range: List[float],
    score_threshold: float,
    max_detections: int,
) -> List[Dict[str, np.ndarray]]:
    """Post-process raw model predictions into evaluation-ready format.

    Applies sigmoid to logits, filters by score, takes top-k detections,
    and denormalizes box centers from [0,1] to absolute coordinates.

    Args:
        pred_logits: Raw classification logits (B, num_queries, num_classes).
        pred_boxes: Raw box predictions (B, num_queries, code_size).
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        score_threshold: Minimum confidence score to keep a detection.
        max_detections: Maximum number of detections per sample.

    Returns:
        List of dicts (one per sample) with:
            - 'scores': (N,) numpy array of confidence scores
            - 'labels': (N,) numpy array of class indices
            - 'boxes': (N, 10) numpy array of decoded box parameters
    """
    batch_size = pred_logits.shape[0]
    pred_scores = pred_logits.sigmoid()  # (B, Q, C)

    results = []
    for b in range(batch_size):
        scores = pred_scores[b]  # (Q, C)
        boxes = pred_boxes[b]  # (Q, code_size)

        # Get maximum score and class per query
        max_scores, max_labels = scores.max(dim=-1)  # (Q,), (Q,)

        # Filter by score threshold
        valid_mask = max_scores > score_threshold
        valid_scores = max_scores[valid_mask]
        valid_labels = max_labels[valid_mask]
        valid_boxes = boxes[valid_mask]

        # Take top-k by score
        num_valid = valid_scores.shape[0]
        if num_valid > max_detections:
            topk_scores, topk_indices = valid_scores.topk(
                max_detections, sorted=True
            )
            topk_labels = valid_labels[topk_indices]
            topk_boxes = valid_boxes[topk_indices]
        else:
            sorted_indices = valid_scores.argsort(descending=True)
            topk_scores = valid_scores[sorted_indices]
            topk_labels = valid_labels[sorted_indices]
            topk_boxes = valid_boxes[sorted_indices]

        # Denormalize box center (cx, cy, cz) from [0, 1] to absolute coords
        denorm_boxes = topk_boxes.clone()
        denorm_boxes[:, 0] = (
            denorm_boxes[:, 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        denorm_boxes[:, 1] = (
            denorm_boxes[:, 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        denorm_boxes[:, 2] = (
            denorm_boxes[:, 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        results.append({
            'scores': topk_scores.cpu().numpy(),
            'labels': topk_labels.cpu().numpy(),
            'boxes': denorm_boxes.cpu().numpy(),
        })

    return results


def prepare_ground_truths(
    batch: Dict[str, torch.Tensor],
) -> List[Dict[str, np.ndarray]]:
    """Extract ground truth annotations from a collated batch.

    Args:
        batch: Collated batch dict from the dataloader with keys:
            - 'labels': (B, max_objects) padded with -1
            - 'boxes_3d': (B, max_objects, 10)
            - 'num_objects': (B,) actual object counts

    Returns:
        List of GT dicts (one per sample) with:
            - 'labels': (M,) numpy array of class indices
            - 'boxes': (M, 10) numpy array of box parameters
    """
    batch_size = batch['labels'].shape[0]
    num_objects = batch['num_objects']
    labels = batch['labels']
    boxes = batch['boxes_3d']

    results = []
    for b in range(batch_size):
        n = int(num_objects[b].item())
        if n > 0:
            sample_labels = labels[b, :n].cpu().numpy()
            sample_boxes = boxes[b, :n].cpu().numpy()
        else:
            sample_labels = np.zeros(0, dtype=np.int64)
            sample_boxes = np.zeros((0, CODE_SIZE), dtype=np.float32)

        results.append({
            'labels': sample_labels,
            'boxes': sample_boxes,
        })

    return results


def load_model(
    checkpoint_path: str,
    num_classes: int = 10,
    pc_range: List[float] = None,
    device: torch.device = None,
) -> DETR3D:
    """Load a DETR3D model from a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (.pth file).
        num_classes: Number of object classes.
        pc_range: Point cloud range.
        device: Device to load the model onto.

    Returns:
        DETR3D model in eval mode.
    """
    if pc_range is None:
        pc_range = PC_RANGE
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DETR3D(
        num_classes=num_classes,
        pc_range=pc_range,
        pretrained_backbone=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if model was saved with DataParallel
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


def print_results(results: Dict) -> None:
    """Print evaluation results in a formatted table.

    Args:
        results: Evaluation results dict from NuScenesEvaluator.evaluate().
    """
    print("\n" + "=" * 80)
    print("DETR3D Evaluation Results")
    print("=" * 80)

    # Per-class AP table
    print("\nPer-Class Average Precision (AP) at BEV Center Distance Thresholds:")
    print("-" * 80)
    header = f"{'Class':<25}"
    for t in results['distance_thresholds']:
        header += f"{'AP@' + str(t) + 'm':<12}"
    header += f"{'Mean AP':<12}"
    print(header)
    print("-" * 80)

    for class_name in CLASS_NAMES:
        row = f"{class_name:<25}"
        for t in results['distance_thresholds']:
            ap_val = results['per_class_AP'][class_name][t]
            row += f"{ap_val:<12.4f}"
        row += f"{results['per_class_mAP'][class_name]:<12.4f}"
        print(row)

    print("-" * 80)

    # Summary metrics
    print(f"\n{'Metric':<30}{'Value':<15}")
    print("-" * 45)
    print(f"{'mAP':<30}{results['mAP']:<15.4f}")
    print(f"{'NDS':<30}{results['NDS']:<15.4f}")
    print("-" * 45)

    # TP metrics
    print(f"\nTrue Positive Metrics (at {2.0}m threshold):")
    print("-" * 45)
    tp_metrics = results['tp_metrics']
    print(f"{'ATE (Translation Error)':<30}{tp_metrics['ATE']:<15.4f}")
    print(f"{'ASE (Scale Error)':<30}{tp_metrics['ASE']:<15.4f}")
    print(f"{'AOE (Orientation Error)':<30}{tp_metrics['AOE']:<15.4f}")
    print(f"{'AVE (Velocity Error)':<30}{tp_metrics['AVE']:<15.4f}")
    print(f"{'AAE (Attribute Error)':<30}{tp_metrics['AAE']:<15.4f}")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate DETR3D model on nuScenes validation set'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional, overrides defaults)',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)',
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to nuScenes dataset root directory',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for inference (default: 4)',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers (default: 4)',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to save results as JSON (optional)',
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f'Score threshold for detections (default: {DEFAULT_SCORE_THRESHOLD})',
    )
    parser.add_argument(
        '--max_detections',
        type=int,
        default=DEFAULT_MAX_DETECTIONS,
        help=f'Maximum detections per sample (default: {DEFAULT_MAX_DETECTIONS})',
    )

    return parser.parse_args()


def main():
    """Main evaluation entry point.

    Loads model and dataset, runs inference on validation set, computes
    nuScenes-style metrics, prints results, and optionally saves to JSON.
    """
    args = parse_args()

    # Load config overrides if provided
    config = {}
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Resolve parameters (command-line args take precedence over config)
    pc_range = config.get('pc_range', PC_RANGE)
    num_classes = config.get('num_classes', 10)
    image_size = tuple(config.get('image_size', [256, 704]))
    version = config.get('version', 'v1.0-trainval')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=num_classes,
        pc_range=pc_range,
        device=device,
    )
    print("Model loaded successfully.")

    # Create validation dataset and dataloader
    print(f"Loading validation dataset from: {args.data_root}")
    val_dataset = NuScenesDataset(
        data_root=args.data_root,
        version=version,
        split='val',
        image_size=image_size,
        pc_range=pc_range,
    )
    print(f"Validation samples: {len(val_dataset)}")

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # Run inference
    print("Running inference...")
    all_predictions = []
    all_ground_truths = []

    total_batches = len(val_dataloader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Move inputs to device
            images = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)

            # Forward pass
            outputs = model(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_shape=image_size,
            )

            # Decode predictions
            batch_preds = decode_predictions(
                pred_logits=outputs['pred_logits'],
                pred_boxes=outputs['pred_boxes'],
                pc_range=pc_range,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
            )
            all_predictions.extend(batch_preds)

            # Extract ground truths
            batch_gts = prepare_ground_truths(batch)
            all_ground_truths.extend(batch_gts)

            # Progress reporting
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - start_time
                samples_done = (batch_idx + 1) * args.batch_size
                samples_per_sec = samples_done / elapsed if elapsed > 0 else 0
                print(
                    f"  Batch [{batch_idx + 1}/{total_batches}] "
                    f"({samples_done} samples, {samples_per_sec:.1f} samples/s)"
                )

    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.1f}s")
    print(f"Total predictions: {sum(p['scores'].shape[0] for p in all_predictions)}")
    print(f"Total ground truths: {sum(g['labels'].shape[0] for g in all_ground_truths)}")

    # Run evaluation
    print("\nComputing metrics...")
    evaluator = NuScenesEvaluator(
        class_names=CLASS_NAMES,
        distance_thresholds=DISTANCE_THRESHOLDS,
    )
    results = evaluator.evaluate(all_predictions, all_ground_truths)

    # Print results
    print_results(results)

    # Save results to JSON if requested
    if args.output_file is not None:
        # Convert numpy types for JSON serialization
        output_data = {
            'mAP': results['mAP'],
            'NDS': results['NDS'],
            'tp_metrics': results['tp_metrics'],
            'per_class_mAP': results['per_class_mAP'],
            'per_class_AP': {
                class_name: {
                    str(t): float(v)
                    for t, v in thresholds.items()
                }
                for class_name, thresholds in results['per_class_AP'].items()
            },
            'config': {
                'checkpoint': args.checkpoint,
                'data_root': args.data_root,
                'score_threshold': args.score_threshold,
                'max_detections': args.max_detections,
                'pc_range': pc_range,
                'distance_thresholds': DISTANCE_THRESHOLDS,
                'num_samples': len(all_predictions),
            },
        }

        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output_file}")


if __name__ == '__main__':
    main()
