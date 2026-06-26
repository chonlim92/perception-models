"""
StreamMapNet - Evaluation Script (PyTorch)

Evaluates a trained StreamMapNet model on the validation set with temporal
propagation, computing standard vectorized map metrics:
  - Chamfer distance between predicted and GT point sets
  - AP at thresholds 0.5m, 1.0m, 1.5m
  - Per-class metrics (lane_divider, road_boundary, ped_crossing)
  - mAP (mean over classes and thresholds)
  - Temporal consistency between consecutive frames

Usage:
    python evaluate.py --config configs/stream_mapnet_base.yaml \
                       --checkpoint work_dirs/stream_mapnet/checkpoints/epoch_24.pth

    # With visualization
    python evaluate.py --config configs/stream_mapnet_base.yaml \
                       --checkpoint work_dirs/stream_mapnet/checkpoints/epoch_24.pth \
                       --visualize --vis_dir work_dirs/vis
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml

from model import StreamMapNet, build_stream_mapnet
from train import StreamMapNetDataset, collate_temporal_sequences


# =============================================================================
# Metrics Computation
# =============================================================================


def chamfer_distance(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
) -> float:
    """
    Compute symmetric Chamfer distance between two point sets.

    Args:
        pred_points: (N_pred, K, 2) predicted polyline points in meters
        gt_points: (N_gt, K, 2) ground truth polyline points in meters

    Returns:
        Symmetric Chamfer distance (mean of forward + backward).
    """
    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")

    # Flatten points per element: (N, K*2)
    pred_flat = pred_points.reshape(len(pred_points), -1)
    gt_flat = gt_points.reshape(len(gt_points), -1)

    # Compute pairwise L2 distances between all point sets
    # Using the mean point distance per element pair
    pred_expanded = pred_points[:, np.newaxis, :, :]  # (N_pred, 1, K, 2)
    gt_expanded = gt_points[np.newaxis, :, :, :]      # (1, N_gt, K, 2)

    # Per-point distance, then mean over points: (N_pred, N_gt)
    point_dists = np.linalg.norm(pred_expanded - gt_expanded, axis=-1)  # (N_p, N_g, K)
    element_dists = point_dists.mean(axis=-1)  # (N_pred, N_gt)

    # Forward: for each pred, distance to nearest GT
    forward_dist = element_dists.min(axis=1).mean()

    # Backward: for each GT, distance to nearest pred
    backward_dist = element_dists.min(axis=0).mean()

    return (forward_dist + backward_dist) / 2.0


def compute_ap_per_threshold(
    pred_points: np.ndarray,
    pred_scores: np.ndarray,
    gt_points: np.ndarray,
    threshold: float,
) -> float:
    """
    Compute Average Precision at a given Chamfer distance threshold.

    A prediction is considered a true positive if its Chamfer distance to
    the nearest unmatched GT element is below the threshold.

    Args:
        pred_points: (N_pred, K, 2) predicted points in meters
        pred_scores: (N_pred,) confidence scores
        gt_points: (N_gt, K, 2) ground truth points in meters
        threshold: Chamfer distance threshold in meters

    Returns:
        AP at the given threshold
    """
    N_pred = len(pred_points)
    N_gt = len(gt_points)

    if N_gt == 0:
        return 1.0 if N_pred == 0 else 0.0
    if N_pred == 0:
        return 0.0

    # Sort predictions by descending score
    sorted_indices = np.argsort(-pred_scores)
    pred_points_sorted = pred_points[sorted_indices]

    # Compute pairwise Chamfer distances
    # For efficiency, use mean per-point L2 distance
    pred_expanded = pred_points_sorted[:, np.newaxis, :, :]  # (N_pred, 1, K, 2)
    gt_expanded = gt_points[np.newaxis, :, :, :]             # (1, N_gt, K, 2)
    point_dists = np.linalg.norm(pred_expanded - gt_expanded, axis=-1)  # (N_p, N_g, K)
    element_dists = point_dists.mean(axis=-1)  # (N_pred, N_gt)

    # Greedy matching: assign each pred to nearest unmatched GT
    gt_matched = np.zeros(N_gt, dtype=bool)
    tp = np.zeros(N_pred, dtype=bool)

    for i in range(N_pred):
        # Find nearest unmatched GT
        unmatched_mask = ~gt_matched
        if not unmatched_mask.any():
            break

        dists_to_unmatched = element_dists[i].copy()
        dists_to_unmatched[gt_matched] = float("inf")
        nearest_gt = dists_to_unmatched.argmin()

        if dists_to_unmatched[nearest_gt] < threshold:
            tp[i] = True
            gt_matched[nearest_gt] = True

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp).astype(float)
    fp_cumsum = np.cumsum(~tp).astype(float)
    recall = tp_cumsum / N_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP using 11-point interpolation (PASCAL VOC style)
    ap = 0.0
    for r_thresh in np.linspace(0, 1, 11):
        prec_at_recall = precision[recall >= r_thresh]
        if len(prec_at_recall) > 0:
            ap += prec_at_recall.max()
    ap /= 11.0

    return ap


def compute_temporal_consistency(
    predictions_seq: List[Dict[str, np.ndarray]],
    ego_motions: List[np.ndarray],
    bev_x_range: Tuple[float, float] = (-30.0, 30.0),
    bev_y_range: Tuple[float, float] = (-15.0, 15.0),
) -> float:
    """
    Measure temporal consistency of predictions across consecutive frames.

    Warps predictions from frame t to frame t+1 using ego motion and
    measures agreement with actual predictions at frame t+1.

    Args:
        predictions_seq: List of prediction dicts per frame
        ego_motions: List of (4, 4) ego motion matrices (prev->curr)
        bev_x_range: BEV x range in meters
        bev_y_range: BEV y range in meters

    Returns:
        Mean temporal consistency score (lower = more consistent, in meters)
    """
    if len(predictions_seq) < 2:
        return 0.0

    consistencies = []

    for t in range(len(predictions_seq) - 1):
        pred_t = predictions_seq[t]
        pred_t1 = predictions_seq[t + 1]
        ego = ego_motions[t + 1]  # transforms from t to t+1

        pts_t = pred_t.get("points", np.zeros((0, 20, 2)))
        pts_t1 = pred_t1.get("points", np.zeros((0, 20, 2)))

        if len(pts_t) == 0 or len(pts_t1) == 0:
            continue

        # Convert normalized [0,1] points to meters
        x_range = bev_x_range[1] - bev_x_range[0]
        y_range = bev_y_range[1] - bev_y_range[0]
        pts_t_m = pts_t.copy()
        pts_t_m[..., 0] = pts_t_m[..., 0] * x_range + bev_x_range[0]
        pts_t_m[..., 1] = pts_t_m[..., 1] * y_range + bev_y_range[0]

        pts_t1_m = pts_t1.copy()
        pts_t1_m[..., 0] = pts_t1_m[..., 0] * x_range + bev_x_range[0]
        pts_t1_m[..., 1] = pts_t1_m[..., 1] * y_range + bev_y_range[0]

        # Warp pts_t to frame t+1 using ego motion
        N, K, _ = pts_t_m.shape
        pts_homo = np.ones((N, K, 4))
        pts_homo[..., 0] = pts_t_m[..., 0]
        pts_homo[..., 1] = pts_t_m[..., 1]
        pts_homo[..., 2] = 0.0

        # ego: (4, 4) prev -> current
        pts_warped = np.einsum("ij,nkj->nki", ego, pts_homo)  # (N, K, 4)
        pts_warped_2d = pts_warped[..., :2]  # (N, K, 2)

        # Compute Chamfer distance between warped predictions and actual predictions
        cd = chamfer_distance(pts_warped_2d, pts_t1_m)
        if np.isfinite(cd):
            consistencies.append(cd)

    return np.mean(consistencies) if consistencies else 0.0


# =============================================================================
# Evaluation Engine
# =============================================================================


class StreamMapNetEvaluator:
    """
    Evaluator for StreamMapNet.

    Accumulates predictions over the validation set and computes
    comprehensive metrics including mAP at multiple thresholds.
    """

    def __init__(
        self,
        config: dict,
        cd_thresholds: List[float] = [0.5, 1.0, 1.5],
        score_threshold: float = 0.3,
    ):
        self.config = config
        self.cd_thresholds = cd_thresholds
        self.score_threshold = score_threshold

        data_cfg = config.get("data", {})
        self.num_classes = data_cfg.get("num_classes", 3)
        self.map_classes = data_cfg.get("map_classes", [
            "lane_divider", "road_boundary", "ped_crossing"
        ])
        self.bev_x_range = tuple(data_cfg.get("bev_range", {}).get("x", [-30.0, 30.0]))
        self.bev_y_range = tuple(data_cfg.get("bev_range", {}).get("y", [-15.0, 15.0]))
        self.num_points = config.get("model", {}).get("map_decoder", {}).get(
            "num_points_per_query", 20
        )

        # Storage for per-class predictions and GT
        self.reset()

    def reset(self):
        """Reset accumulated predictions."""
        # Per-class storage: {class_idx: list of (pred_pts, pred_scores, gt_pts)}
        self.per_class_results = defaultdict(list)
        self.temporal_predictions = []  # list of sequences
        self.temporal_ego_motions = []

    def _denormalize_points(self, points: np.ndarray) -> np.ndarray:
        """Convert points from [0,1] normalized to meters."""
        x_range = self.bev_x_range[1] - self.bev_x_range[0]
        y_range = self.bev_y_range[1] - self.bev_y_range[0]
        pts_m = points.copy()
        pts_m[..., 0] = pts_m[..., 0] * x_range + self.bev_x_range[0]
        pts_m[..., 1] = pts_m[..., 1] * y_range + self.bev_y_range[0]
        return pts_m

    def update(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_points: torch.Tensor,
    ):
        """
        Add a batch of predictions and ground truth to the evaluator.

        Args:
            pred_logits: (B, N_q, C+1) classification logits
            pred_points: (B, N_q, K, 2) predicted points [0,1]
            gt_labels: list of (N_gt,) per-sample class labels
            gt_points: list of (N_gt, K, 2) per-sample GT points [0,1]
        """
        B = pred_logits.shape[0]
        probs = pred_logits.softmax(dim=-1).cpu().numpy()
        points_np = pred_points.cpu().numpy()

        for b in range(B):
            # Get predictions above threshold
            scores = probs[b, :, :-1].max(axis=-1)  # (N_q,)
            labels = probs[b, :, :-1].argmax(axis=-1)  # (N_q,)
            mask = scores > self.score_threshold

            pred_scores_b = scores[mask]
            pred_labels_b = labels[mask]
            pred_pts_b = points_np[b][mask]  # (N_keep, K, 2)

            # Ground truth
            gt_labels_b = gt_labels[b].cpu().numpy() if torch.is_tensor(gt_labels[b]) else gt_labels[b]
            gt_pts_b = gt_points[b].cpu().numpy() if torch.is_tensor(gt_points[b]) else gt_points[b]

            # Store per-class
            for cls_idx in range(self.num_classes):
                pred_mask_cls = pred_labels_b == cls_idx
                gt_mask_cls = gt_labels_b == cls_idx

                pred_pts_cls = self._denormalize_points(pred_pts_b[pred_mask_cls])
                pred_scores_cls = pred_scores_b[pred_mask_cls]
                gt_pts_cls = self._denormalize_points(gt_pts_b[gt_mask_cls])

                self.per_class_results[cls_idx].append({
                    "pred_points": pred_pts_cls,
                    "pred_scores": pred_scores_cls,
                    "gt_points": gt_pts_cls,
                })

    def update_temporal(
        self,
        sequence_predictions: List[Dict[str, np.ndarray]],
        sequence_ego_motions: List[np.ndarray],
    ):
        """Add a sequence of predictions for temporal consistency evaluation."""
        self.temporal_predictions.append(sequence_predictions)
        self.temporal_ego_motions.append(sequence_ego_motions)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Returns:
            dict with:
                - Per-class AP at each threshold
                - Per-class mAP (mean over thresholds)
                - Overall mAP (mean over classes and thresholds)
                - Per-class Chamfer distance
                - Temporal consistency
        """
        results = {}
        all_aps = []

        for cls_idx in range(self.num_classes):
            cls_name = self.map_classes[cls_idx]
            class_data = self.per_class_results[cls_idx]

            if not class_data:
                for thresh in self.cd_thresholds:
                    results[f"{cls_name}/AP_{thresh:.1f}m"] = 0.0
                results[f"{cls_name}/mAP"] = 0.0
                results[f"{cls_name}/chamfer_dist"] = float("inf")
                continue

            # Aggregate all predictions and GT for this class
            all_pred_pts = np.concatenate(
                [d["pred_points"] for d in class_data if len(d["pred_points"]) > 0],
                axis=0,
            ) if any(len(d["pred_points"]) > 0 for d in class_data) else np.zeros((0, self.num_points, 2))

            all_pred_scores = np.concatenate(
                [d["pred_scores"] for d in class_data if len(d["pred_scores"]) > 0],
                axis=0,
            ) if any(len(d["pred_scores"]) > 0 for d in class_data) else np.zeros(0)

            all_gt_pts = np.concatenate(
                [d["gt_points"] for d in class_data if len(d["gt_points"]) > 0],
                axis=0,
            ) if any(len(d["gt_points"]) > 0 for d in class_data) else np.zeros((0, self.num_points, 2))

            # AP at each threshold
            class_aps = []
            for thresh in self.cd_thresholds:
                ap = compute_ap_per_threshold(
                    all_pred_pts, all_pred_scores, all_gt_pts, thresh
                )
                results[f"{cls_name}/AP_{thresh:.1f}m"] = ap
                class_aps.append(ap)
                all_aps.append(ap)

            results[f"{cls_name}/mAP"] = np.mean(class_aps)

            # Chamfer distance (using top predictions)
            if len(all_pred_pts) > 0 and len(all_gt_pts) > 0:
                cd = chamfer_distance(all_pred_pts, all_gt_pts)
                results[f"{cls_name}/chamfer_dist"] = cd
            else:
                results[f"{cls_name}/chamfer_dist"] = float("inf")

        # Overall mAP
        results["mAP"] = np.mean(all_aps) if all_aps else 0.0

        # Temporal consistency
        if self.temporal_predictions:
            consistencies = []
            for seq_preds, seq_egos in zip(
                self.temporal_predictions, self.temporal_ego_motions
            ):
                tc = compute_temporal_consistency(
                    seq_preds, seq_egos, self.bev_x_range, self.bev_y_range
                )
                consistencies.append(tc)
            results["temporal_consistency"] = np.mean(consistencies)
        else:
            results["temporal_consistency"] = 0.0

        return results


# =============================================================================
# Evaluation Loop
# =============================================================================


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    evaluator: StreamMapNetEvaluator,
    device: torch.device,
    max_sequences: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run evaluation over the validation set with temporal propagation.

    Args:
        model: trained StreamMapNet model
        dataloader: validation dataloader
        evaluator: metrics evaluator
        device: compute device
        max_sequences: maximum number of sequences to evaluate (None = all)

    Returns:
        metrics dict
    """
    model.eval()
    evaluator.reset()

    num_sequences = 0
    total_frames = 0
    total_time = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if max_sequences and num_sequences >= max_sequences:
            break

        images = batch["images"].to(device)            # (B, T, N, 3, H, W)
        intrinsics = batch["intrinsics"].to(device)    # (B, T, N, 3, 3)
        extrinsics = batch["extrinsics"].to(device)    # (B, T, N, 4, 4)
        ego_motions = batch["ego_motions"].to(device)  # (B, T, 4, 4)
        targets = batch["targets"]                     # list of T lists

        B, T = images.shape[:2]

        # Reset temporal state for new sequences
        model_unwrapped = model.module if hasattr(model, "module") else model
        model_unwrapped.reset_temporal_state()

        # Process sequence frame by frame
        sequence_preds = [[] for _ in range(B)]

        for t in range(T):
            ego_motion_t = ego_motions[:, t] if t > 0 else None

            start = time.time()
            outputs = model(
                images[:, t],
                intrinsics[:, t],
                extrinsics[:, t],
                ego_motion=ego_motion_t,
            )
            total_time += time.time() - start
            total_frames += B

            # Update evaluator with this frame's predictions and GT
            frame_targets = targets[t]
            gt_labels_list = [ft["labels"] for ft in frame_targets]
            gt_points_list = [ft["points"] for ft in frame_targets]

            evaluator.update(
                outputs["pred_logits"],
                outputs["pred_points"],
                gt_labels_list,
                gt_points_list,
            )

            # Store predictions for temporal consistency
            probs = outputs["pred_logits"].softmax(dim=-1).cpu().numpy()
            pts_np = outputs["pred_points"].cpu().numpy()
            for b in range(B):
                scores = probs[b, :, :-1].max(axis=-1)
                mask = scores > evaluator.score_threshold
                sequence_preds[b].append({
                    "points": pts_np[b][mask],
                    "scores": scores[mask],
                })

        # Update temporal consistency
        for b in range(B):
            seq_egos = ego_motions[:, :, :, :].cpu().numpy()[b]  # (T, 4, 4)
            evaluator.update_temporal(
                sequence_preds[b],
                [seq_egos[t] for t in range(T)],
            )

        num_sequences += B

        if (batch_idx + 1) % 10 == 0:
            print(f"  Evaluated {num_sequences} sequences, {total_frames} frames")

    # Compute metrics
    metrics = evaluator.compute_metrics()

    # Add timing info
    if total_frames > 0:
        metrics["avg_time_per_frame_ms"] = (total_time / total_frames) * 1000
        metrics["fps"] = total_frames / total_time
    metrics["total_sequences"] = num_sequences
    metrics["total_frames"] = total_frames

    return metrics


# =============================================================================
# Results Formatting
# =============================================================================


def print_results_table(metrics: Dict[str, float], map_classes: List[str]):
    """Print evaluation results as a formatted table."""
    print("\n" + "=" * 80)
    print("StreamMapNet Evaluation Results")
    print("=" * 80)

    # Overall metrics
    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 50)
    print(f"{'mAP (overall)':<35} {metrics.get('mAP', 0.0):>10.4f}")
    print(f"{'Temporal Consistency (m)':<35} {metrics.get('temporal_consistency', 0.0):>10.4f}")
    print(f"{'FPS':<35} {metrics.get('fps', 0.0):>10.1f}")
    print(f"{'Avg Time/Frame (ms)':<35} {metrics.get('avg_time_per_frame_ms', 0.0):>10.1f}")

    # Per-class AP table
    print(f"\n{'Class':<20} {'AP@0.5m':>10} {'AP@1.0m':>10} {'AP@1.5m':>10} {'mAP':>10} {'CD (m)':>10}")
    print("-" * 80)
    for cls_name in map_classes:
        ap_05 = metrics.get(f"{cls_name}/AP_0.5m", 0.0)
        ap_10 = metrics.get(f"{cls_name}/AP_1.0m", 0.0)
        ap_15 = metrics.get(f"{cls_name}/AP_1.5m", 0.0)
        class_map = metrics.get(f"{cls_name}/mAP", 0.0)
        cd = metrics.get(f"{cls_name}/chamfer_dist", float("inf"))
        cd_str = f"{cd:.4f}" if np.isfinite(cd) else "inf"
        print(
            f"{cls_name:<20} {ap_05:>10.4f} {ap_10:>10.4f} {ap_15:>10.4f} "
            f"{class_map:>10.4f} {cd_str:>10}"
        )

    print("-" * 80)
    print(f"{'MEAN':<20} "
          f"{np.mean([metrics.get(f'{c}/AP_0.5m', 0) for c in map_classes]):>10.4f} "
          f"{np.mean([metrics.get(f'{c}/AP_1.0m', 0) for c in map_classes]):>10.4f} "
          f"{np.mean([metrics.get(f'{c}/AP_1.5m', 0) for c in map_classes]):>10.4f} "
          f"{metrics.get('mAP', 0.0):>10.4f}")
    print("=" * 80)

    # Summary
    print(f"\nEvaluated {metrics.get('total_sequences', 0)} sequences, "
          f"{metrics.get('total_frames', 0)} total frames.")


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate StreamMapNet")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_root", type=str, default="data/nuscenes",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--max_sequences", type=int, default=None,
        help="Maximum number of sequences to evaluate (None = all)",
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.3,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save BEV visualization of predictions",
    )
    parser.add_argument(
        "--vis_dir", type=str, default="work_dirs/vis",
        help="Directory to save visualizations",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"StreamMapNet Evaluation")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")

    # Build model
    model = build_stream_mapnet(config).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        print(f"  Loaded model from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded model weights")

    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Build validation dataset
    data_cfg = config.get("data", {})
    temporal_cfg = data_cfg.get("temporal", {})

    val_dataset = StreamMapNetDataset(
        data_root=args.data_root,
        split="val",
        sequence_length=temporal_cfg.get("window_size", 8),
        img_size=tuple(data_cfg.get("img_size", [256, 704])),
        num_cameras=data_cfg.get("num_cameras", 6),
        num_classes=data_cfg.get("num_classes", 3),
        num_points=config.get("model", {}).get("map_decoder", {}).get("num_points_per_query", 20),
        augment=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_temporal_sequences,
        drop_last=False,
    )

    print(f"  Validation set: {len(val_dataset)} sequences")
    print(f"  Batch size: {args.batch_size}")

    # Build evaluator
    eval_cfg = config.get("evaluation", {})
    cd_thresholds = eval_cfg.get("cd_thresholds", [0.5, 1.0, 1.5])

    evaluator = StreamMapNetEvaluator(
        config=config,
        cd_thresholds=cd_thresholds,
        score_threshold=args.score_threshold,
    )

    # Run evaluation
    print(f"\nRunning evaluation...")
    start_time = time.time()

    metrics = evaluate(
        model=model,
        dataloader=val_loader,
        evaluator=evaluator,
        device=device,
        max_sequences=args.max_sequences,
    )

    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.1f}s")

    # Print results
    map_classes = data_cfg.get("map_classes", ["lane_divider", "road_boundary", "ped_crossing"])
    print_results_table(metrics, map_classes)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types to Python types for JSON serialization
        metrics_serializable = {
            k: float(v) if isinstance(v, (np.floating, float)) else int(v)
            for k, v in metrics.items()
        }
        with open(output_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
