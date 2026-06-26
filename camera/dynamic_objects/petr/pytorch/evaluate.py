"""
Evaluation script for PETR/PETRv2/StreamPETR.

Computes nuScenes-style metrics: mAP, NDS, mATE, mASE, mAOE, mAVE, mAAE.
Supports per-class AP at multiple distance thresholds, streaming evaluation
for StreamPETR, and FPS measurement.
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

from .dataset import DETECTION_CLASSES, NuScenesDataset, collate_fn
from .model import PETRConfig, PETRModel

logger = logging.getLogger(__name__)


def compute_iou_3d(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Compute 3D IoU between two sets of axis-aligned bounding boxes.

    For nuScenes evaluation, boxes are compared using center distance
    rather than IoU, but this is kept for completeness.

    Args:
        boxes_a: (N, 7) array [cx, cy, cz, w, l, h, yaw].
        boxes_b: (M, 7) array [cx, cy, cz, w, l, h, yaw].

    Returns:
        IoU matrix (N, M).
    """
    # Simplified BEV IoU (ignoring rotation for speed)
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]
    iou = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        ax1 = boxes_a[i, 0] - boxes_a[i, 3] / 2
        ax2 = boxes_a[i, 0] + boxes_a[i, 3] / 2
        ay1 = boxes_a[i, 1] - boxes_a[i, 4] / 2
        ay2 = boxes_a[i, 1] + boxes_a[i, 4] / 2

        for j in range(M):
            bx1 = boxes_b[j, 0] - boxes_b[j, 3] / 2
            bx2 = boxes_b[j, 0] + boxes_b[j, 3] / 2
            by1 = boxes_b[j, 1] - boxes_b[j, 4] / 2
            by2 = boxes_b[j, 1] + boxes_b[j, 4] / 2

            # Intersection
            ix1 = max(ax1, bx1)
            ix2 = min(ax2, bx2)
            iy1 = max(ay1, by1)
            iy2 = min(ay2, by2)

            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
            else:
                inter = 0.0

            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union = area_a + area_b - inter

            if union > 0:
                iou[i, j] = inter / union

    return iou


def compute_center_distance(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray
) -> np.ndarray:
    """Compute BEV center distance between predictions and ground truth.

    Args:
        pred_boxes: (N, 10) predicted boxes [cx, cy, cz, ...].
        gt_boxes: (M, 10) ground-truth boxes [cx, cy, cz, ...].

    Returns:
        Distance matrix (N, M) in meters (BEV distance).
    """
    pred_centers = pred_boxes[:, :2]  # (N, 2) - x, y
    gt_centers = gt_boxes[:, :2]  # (M, 2) - x, y

    # Compute pairwise Euclidean distance
    diff = pred_centers[:, None, :] - gt_centers[None, :, :]  # (N, M, 2)
    distances = np.linalg.norm(diff, axis=-1)  # (N, M)

    return distances


def compute_ap_per_class(
    pred_scores: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    class_id: int,
    distance_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0],
) -> Dict[str, float]:
    """Compute Average Precision for a single class at multiple thresholds.

    Uses center-distance matching (nuScenes style).

    Args:
        pred_scores: (N,) prediction confidence scores.
        pred_boxes: (N, 10) predicted bounding boxes.
        pred_labels: (N,) predicted class labels.
        gt_boxes: (M, 10) ground-truth bounding boxes.
        gt_labels: (M,) ground-truth class labels.
        class_id: Class index to evaluate.
        distance_thresholds: BEV distance thresholds for matching.

    Returns:
        Dictionary with AP at each threshold and mean AP.
    """
    # Filter by class
    pred_mask = pred_labels == class_id
    gt_mask = gt_labels == class_id

    pred_scores_cls = pred_scores[pred_mask]
    pred_boxes_cls = pred_boxes[pred_mask]
    gt_boxes_cls = gt_boxes[gt_mask]

    num_gt = gt_boxes_cls.shape[0]

    if num_gt == 0 and pred_scores_cls.shape[0] == 0:
        return {f"AP@{t}": 0.0 for t in distance_thresholds}

    if num_gt == 0:
        return {f"AP@{t}": 0.0 for t in distance_thresholds}

    if pred_scores_cls.shape[0] == 0:
        return {f"AP@{t}": 0.0 for t in distance_thresholds}

    # Sort predictions by score (descending)
    sort_idx = np.argsort(-pred_scores_cls)
    pred_scores_cls = pred_scores_cls[sort_idx]
    pred_boxes_cls = pred_boxes_cls[sort_idx]

    # Compute center distances
    distances = compute_center_distance(pred_boxes_cls, gt_boxes_cls)  # (N_pred, N_gt)

    results = {}
    for thresh in distance_thresholds:
        # Greedy matching at this threshold
        tp = np.zeros(len(pred_scores_cls), dtype=np.float32)
        fp = np.zeros(len(pred_scores_cls), dtype=np.float32)
        matched_gt = set()

        for pred_idx in range(len(pred_scores_cls)):
            # Find closest unmatched GT
            min_dist = float("inf")
            min_gt_idx = -1

            for gt_idx in range(num_gt):
                if gt_idx in matched_gt:
                    continue
                if distances[pred_idx, gt_idx] < min_dist:
                    min_dist = distances[pred_idx, gt_idx]
                    min_gt_idx = gt_idx

            if min_dist <= thresh and min_gt_idx >= 0:
                tp[pred_idx] = 1.0
                matched_gt.add(min_gt_idx)
            else:
                fp[pred_idx] = 1.0

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / num_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Compute AP using 11-point interpolation
        ap = 0.0
        for r_thresh in np.linspace(0, 1, 11):
            precisions_at_recall = precision[recall >= r_thresh]
            if len(precisions_at_recall) > 0:
                ap += np.max(precisions_at_recall)
        ap /= 11.0

        results[f"AP@{thresh}"] = float(ap)

    # Mean AP across thresholds
    results["mAP"] = float(np.mean([results[f"AP@{t}"] for t in distance_thresholds]))

    return results


def compute_translation_error(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, matched_pairs: List[Tuple[int, int]]
) -> float:
    """Compute mean Average Translation Error (mATE).

    Args:
        pred_boxes: Predicted boxes (N, 10).
        gt_boxes: Ground-truth boxes (M, 10).
        matched_pairs: List of (pred_idx, gt_idx) matched pairs.

    Returns:
        Mean translation error in meters.
    """
    if len(matched_pairs) == 0:
        return float("inf")

    errors = []
    for pred_idx, gt_idx in matched_pairs:
        pred_center = pred_boxes[pred_idx, :3]
        gt_center = gt_boxes[gt_idx, :3]
        error = np.linalg.norm(pred_center - gt_center)
        errors.append(error)

    return float(np.mean(errors))


def compute_scale_error(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, matched_pairs: List[Tuple[int, int]]
) -> float:
    """Compute mean Average Scale Error (mASE).

    1 - IoU of 3D bounding box dimensions (ignoring position and rotation).

    Args:
        pred_boxes: Predicted boxes (N, 10) with [cx,cy,cz,w,l,h,...].
        gt_boxes: Ground-truth boxes (M, 10).
        matched_pairs: List of (pred_idx, gt_idx) matched pairs.

    Returns:
        Mean scale error (1 - dimension IoU).
    """
    if len(matched_pairs) == 0:
        return 1.0

    errors = []
    for pred_idx, gt_idx in matched_pairs:
        pred_wlh = pred_boxes[pred_idx, 3:6]  # w, l, h
        gt_wlh = gt_boxes[gt_idx, 3:6]

        # Volume IoU of axis-aligned boxes centered at origin
        min_wlh = np.minimum(pred_wlh, gt_wlh)
        intersection = np.prod(min_wlh)
        union = np.prod(pred_wlh) + np.prod(gt_wlh) - intersection
        iou = intersection / max(union, 1e-8)
        errors.append(1.0 - iou)

    return float(np.mean(errors))


def compute_orientation_error(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, matched_pairs: List[Tuple[int, int]]
) -> float:
    """Compute mean Average Orientation Error (mAOE).

    Args:
        pred_boxes: Predicted boxes (N, 10) with [..., sin, cos, ...].
        gt_boxes: Ground-truth boxes (M, 10).
        matched_pairs: List of (pred_idx, gt_idx) matched pairs.

    Returns:
        Mean orientation error in radians.
    """
    if len(matched_pairs) == 0:
        return float("inf")

    errors = []
    for pred_idx, gt_idx in matched_pairs:
        pred_yaw = np.arctan2(pred_boxes[pred_idx, 6], pred_boxes[pred_idx, 7])
        gt_yaw = np.arctan2(gt_boxes[gt_idx, 6], gt_boxes[gt_idx, 7])

        # Angle difference wrapped to [-pi, pi]
        diff = pred_yaw - gt_yaw
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        errors.append(abs(diff))

    return float(np.mean(errors))


def compute_velocity_error(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, matched_pairs: List[Tuple[int, int]]
) -> float:
    """Compute mean Average Velocity Error (mAVE).

    Args:
        pred_boxes: Predicted boxes (N, 10) with [..., vx, vy].
        gt_boxes: Ground-truth boxes (M, 10).
        matched_pairs: List of (pred_idx, gt_idx) matched pairs.

    Returns:
        Mean velocity error in m/s.
    """
    if len(matched_pairs) == 0:
        return float("inf")

    errors = []
    for pred_idx, gt_idx in matched_pairs:
        pred_vel = pred_boxes[pred_idx, 8:10]
        gt_vel = gt_boxes[gt_idx, 8:10]
        error = np.linalg.norm(pred_vel - gt_vel)
        errors.append(error)

    return float(np.mean(errors))


def compute_attribute_error(
    pred_attrs: Optional[np.ndarray],
    gt_attrs: Optional[np.ndarray],
    matched_pairs: List[Tuple[int, int]],
) -> float:
    """Compute mean Average Attribute Error (mAAE).

    Args:
        pred_attrs: Predicted attributes (N,).
        gt_attrs: Ground-truth attributes (M,).
        matched_pairs: List of (pred_idx, gt_idx) matched pairs.

    Returns:
        Mean attribute error (1 - accuracy).
    """
    if pred_attrs is None or gt_attrs is None or len(matched_pairs) == 0:
        return 1.0

    correct = 0
    for pred_idx, gt_idx in matched_pairs:
        if pred_attrs[pred_idx] == gt_attrs[gt_idx]:
            correct += 1

    accuracy = correct / len(matched_pairs)
    return 1.0 - accuracy


def get_matched_pairs(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    distance_threshold: float = 2.0,
) -> List[Tuple[int, int]]:
    """Get matched prediction-GT pairs using center distance.

    Args:
        pred_boxes: (N, 10) predicted boxes.
        gt_boxes: (M, 10) ground-truth boxes.
        distance_threshold: Maximum matching distance.

    Returns:
        List of (pred_idx, gt_idx) pairs.
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return []

    distances = compute_center_distance(pred_boxes, gt_boxes)
    matched_pairs = []
    matched_gt = set()

    # Sort predictions by distance to closest GT
    min_dists = distances.min(axis=1)
    sort_idx = np.argsort(min_dists)

    for pred_idx in sort_idx:
        min_dist = float("inf")
        best_gt = -1
        for gt_idx in range(gt_boxes.shape[0]):
            if gt_idx in matched_gt:
                continue
            if distances[pred_idx, gt_idx] < min_dist:
                min_dist = distances[pred_idx, gt_idx]
                best_gt = gt_idx

        if min_dist <= distance_threshold and best_gt >= 0:
            matched_pairs.append((int(pred_idx), int(best_gt)))
            matched_gt.add(best_gt)

    return matched_pairs


def compute_nds(metrics: Dict[str, float]) -> float:
    """Compute nuScenes Detection Score (NDS).

    NDS = (1/10) * [5*mAP + sum(1 - min(1, metric)) for metric in TP metrics]
    where TP metrics are mATE, mASE, mAOE, mAVE, mAAE.

    Args:
        metrics: Dictionary with mAP and TP error metrics.

    Returns:
        NDS score in [0, 1].
    """
    mAP = metrics.get("mAP", 0.0)
    mATE = metrics.get("mATE", 1.0)
    mASE = metrics.get("mASE", 1.0)
    mAOE = metrics.get("mAOE", 1.0)
    mAVE = metrics.get("mAVE", 1.0)
    mAAE = metrics.get("mAAE", 1.0)

    # Cap errors at 1.0
    tp_scores = [
        max(0.0, 1.0 - min(1.0, mATE)),
        max(0.0, 1.0 - min(1.0, mASE)),
        max(0.0, 1.0 - min(1.0, mAOE)),
        max(0.0, 1.0 - min(1.0, mAVE)),
        max(0.0, 1.0 - min(1.0, mAAE)),
    ]

    nds = (5.0 * mAP + sum(tp_scores)) / 10.0
    return float(nds)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Run evaluation on the validation set.

    Args:
        model: Trained model.
        dataloader: Validation data loader.
        device: Evaluation device.
        config: Evaluation configuration.
        output_dir: Directory to save results.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    is_stream = config.get("model", {}).get("variant", "petr") == "streampetr"
    pc_range = tuple(config.get("model", {}).get(
        "pc_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    ))
    score_threshold = config.get("eval", {}).get("score_threshold", 0.3)

    # Collect all predictions and ground truths
    all_pred_scores = []
    all_pred_boxes = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    # FPS measurement
    total_time = 0.0
    num_samples = 0

    # Reset temporal state for streaming evaluation
    if hasattr(model, "module"):
        model.module.reset_temporal_state()
    else:
        model.reset_temporal_state()

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)

        ego_motion = batch.get("ego_motion")
        ego_motion_vec = batch.get("ego_motion_vec")
        if ego_motion is not None:
            ego_motion = ego_motion.to(device)
        if ego_motion_vec is not None:
            ego_motion_vec = ego_motion_vec.to(device)

        prev_images = batch.get("prev_images")
        prev_intrinsics = batch.get("prev_intrinsics")
        prev_extrinsics = batch.get("prev_extrinsics")
        prev_ego_motions = batch.get("prev_ego_motions")
        if prev_images is not None:
            prev_images = prev_images.to(device)
            prev_intrinsics = prev_intrinsics.to(device)
            prev_extrinsics = prev_extrinsics.to(device)
            prev_ego_motions = prev_ego_motions.to(device)

        # Measure inference time
        torch.cuda.synchronize()
        start = time.time()

        outputs = model(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            ego_motion=ego_motion,
            ego_motion_vec=ego_motion_vec,
            prev_images=prev_images,
            prev_intrinsics=prev_intrinsics,
            prev_extrinsics=prev_extrinsics,
            prev_ego_motions=prev_ego_motions,
        )

        torch.cuda.synchronize()
        elapsed = time.time() - start
        total_time += elapsed
        num_samples += images.shape[0]

        # Extract predictions from last decoder layer
        predictions = outputs["predictions"]
        cls_scores = predictions["cls_scores"][-1]  # (B, Q, num_classes)
        bbox_preds = predictions["bbox_preds"][-1]  # (B, Q, code_size)

        B = cls_scores.shape[0]

        for b in range(B):
            # Get per-query scores and labels
            scores_per_query = cls_scores[b].sigmoid()  # (Q, num_classes)
            max_scores, pred_labels_b = scores_per_query.max(dim=-1)  # (Q,)

            # Filter by score threshold
            keep_mask = max_scores > score_threshold
            scores_kept = max_scores[keep_mask].cpu().numpy()
            labels_kept = pred_labels_b[keep_mask].cpu().numpy()
            boxes_kept = bbox_preds[b][keep_mask].cpu().numpy()

            all_pred_scores.append(scores_kept)
            all_pred_boxes.append(boxes_kept)
            all_pred_labels.append(labels_kept)

            # Ground truth
            gt_labels_b = batch["gt_labels"][b].numpy()
            gt_bboxes_b = batch["gt_bboxes"][b].numpy()
            all_gt_labels.append(gt_labels_b)
            all_gt_boxes.append(gt_bboxes_b)

    # Compute metrics
    logger.info("Computing evaluation metrics...")

    # Per-class AP
    distance_thresholds = [0.5, 1.0, 2.0, 4.0]
    class_aps = {}
    all_matched_pairs = []

    # Concatenate all predictions and GTs across samples
    # (For simplicity, evaluate per-sample and average)
    per_sample_aps = {cls: [] for cls in DETECTION_CLASSES}

    for sample_idx in range(len(all_pred_scores)):
        pred_s = all_pred_scores[sample_idx]
        pred_b = all_pred_boxes[sample_idx]
        pred_l = all_pred_labels[sample_idx]
        gt_b = all_gt_boxes[sample_idx]
        gt_l = all_gt_labels[sample_idx]

        # Get matched pairs for TP metrics
        if pred_b.shape[0] > 0 and gt_b.shape[0] > 0:
            pairs = get_matched_pairs(pred_b, gt_b, distance_threshold=2.0)
            all_matched_pairs.append((pred_b, gt_b, pairs))

        for cls_idx, cls_name in enumerate(DETECTION_CLASSES):
            ap_results = compute_ap_per_class(
                pred_s, pred_b, pred_l, gt_b, gt_l, cls_idx, distance_thresholds
            )
            per_sample_aps[cls_name].append(ap_results.get("mAP", 0.0))

    # Mean AP across classes and samples
    for cls_name in DETECTION_CLASSES:
        aps = per_sample_aps[cls_name]
        class_aps[cls_name] = float(np.mean(aps)) if aps else 0.0

    mAP = float(np.mean(list(class_aps.values())))

    # TP metrics (computed on all matched pairs)
    all_pred_matched = []
    all_gt_matched = []
    all_pairs_flat = []

    offset_pred = 0
    offset_gt = 0
    for pred_b, gt_b, pairs in all_matched_pairs:
        for pi, gi in pairs:
            all_pairs_flat.append((offset_pred + pi, offset_gt + gi))
        all_pred_matched.append(pred_b)
        all_gt_matched.append(gt_b)
        offset_pred += pred_b.shape[0]
        offset_gt += gt_b.shape[0]

    if all_pred_matched:
        all_pred_concat = np.concatenate(all_pred_matched, axis=0)
        all_gt_concat = np.concatenate(all_gt_matched, axis=0)
    else:
        all_pred_concat = np.zeros((0, 10))
        all_gt_concat = np.zeros((0, 10))

    mATE = compute_translation_error(all_pred_concat, all_gt_concat, all_pairs_flat)
    mASE = compute_scale_error(all_pred_concat, all_gt_concat, all_pairs_flat)
    mAOE = compute_orientation_error(all_pred_concat, all_gt_concat, all_pairs_flat)
    mAVE = compute_velocity_error(all_pred_concat, all_gt_concat, all_pairs_flat)
    mAAE = 1.0  # Attribute not predicted in this implementation

    # Compute NDS
    metrics = {
        "mAP": mAP,
        "mATE": mATE,
        "mASE": mASE,
        "mAOE": mAOE,
        "mAVE": mAVE,
        "mAAE": mAAE,
    }
    nds = compute_nds(metrics)
    metrics["NDS"] = nds

    # FPS
    fps = num_samples / total_time if total_time > 0 else 0.0
    metrics["FPS"] = fps

    # Per-class AP
    metrics["per_class_AP"] = class_aps

    # Log results
    logger.info(f"{'='*60}")
    logger.info(f"Evaluation Results:")
    logger.info(f"  mAP:  {mAP:.4f}")
    logger.info(f"  NDS:  {nds:.4f}")
    logger.info(f"  mATE: {mATE:.4f}")
    logger.info(f"  mASE: {mASE:.4f}")
    logger.info(f"  mAOE: {mAOE:.4f}")
    logger.info(f"  mAVE: {mAVE:.4f}")
    logger.info(f"  mAAE: {mAAE:.4f}")
    logger.info(f"  FPS:  {fps:.1f}")
    logger.info(f"{'='*60}")
    logger.info(f"Per-class AP:")
    for cls_name, ap in class_aps.items():
        logger.info(f"  {cls_name:25s}: {ap:.4f}")
    logger.info(f"{'='*60}")

    # Save results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "eval_results.json")
        # Convert numpy types for JSON serialization
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                serializable_metrics[k] = {kk: float(vv) for kk, vv in v.items()}
            else:
                serializable_metrics[k] = float(v) if not isinstance(v, str) else v
        with open(results_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    return metrics


def main() -> None:
    """Entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate PETR/PETRv2/StreamPETR")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results", help="Output directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Evaluation batch size"
    )
    args = parser.parse_args()

    # Setup
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    data_config = config.get("data", {})

    # Build model
    petr_config = PETRConfig(**model_config)
    model = PETRModel(petr_config).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Build validation dataset
    dataset = NuScenesDataset(
        data_root=data_config.get("data_root", "/data/nuscenes"),
        ann_file=data_config.get("val_ann_file"),
        split="val",
        num_cameras=model_config.get("num_cameras", 6),
        img_size=tuple(model_config.get("img_size", [900, 1600])),
        num_temporal_frames=model_config.get("num_temporal_frames", 0),
        pc_range=tuple(model_config.get("pc_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])),
        augmentation=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Run evaluation
    metrics = evaluate(model, dataloader, device, config, args.output_dir)


if __name__ == "__main__":
    main()
