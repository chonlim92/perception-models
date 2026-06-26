"""
Evaluation script for HDMapNet.

Computes:
- IoU (Intersection over Union) per semantic class
- Chamfer distance between predicted and ground truth polylines
- Average Precision (AP) at multiple distance thresholds
- Prints a formatted results table
"""

import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import HDMapNet
from .dataset import NuScenesHDMapDataset, collate_fn
from .postprocess import vectorize_predictions


CLASS_NAMES = ["divider", "boundary", "crossing"]


def compute_iou_per_class(pred_semantic, gt_semantic, threshold=0.5):
    """Compute IoU for each semantic class.

    Args:
        pred_semantic: Predicted logits (B, C, H, W).
        gt_semantic: Ground truth binary masks (B, C, H, W).
        threshold: Threshold for binarizing predictions.

    Returns:
        Dict mapping class_idx to IoU value.
    """
    pred_binary = (torch.sigmoid(pred_semantic) > threshold).float()
    num_classes = pred_binary.shape[1]

    iou_dict = {}
    for c in range(num_classes):
        pred_c = pred_binary[:, c].reshape(-1)
        gt_c = gt_semantic[:, c].reshape(-1)

        intersection = (pred_c * gt_c).sum().item()
        union = ((pred_c + gt_c) > 0).float().sum().item()

        if union > 0:
            iou_dict[c] = intersection / union
        else:
            iou_dict[c] = float("nan")

    return iou_dict


def chamfer_distance_polylines(pred_polylines, gt_polylines):
    """Compute Chamfer distance between two sets of polylines.

    For each point in the prediction, find the nearest point in GT (and vice versa).
    Returns the average of both directions.

    Args:
        pred_polylines: List of numpy arrays, each (N_i, 2).
        gt_polylines: List of numpy arrays, each (M_j, 2).

    Returns:
        Chamfer distance (scalar). Returns inf if either set is empty.
    """
    if len(pred_polylines) == 0 or len(gt_polylines) == 0:
        return float("inf")

    # Concatenate all points from each set
    pred_points = np.concatenate(pred_polylines, axis=0)  # (P, 2)
    gt_points = np.concatenate(gt_polylines, axis=0)      # (G, 2)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")

    # Pred to GT: for each pred point, find nearest GT point
    diff_p2g = pred_points[:, None, :] - gt_points[None, :, :]  # (P, G, 2)
    dist_p2g = np.linalg.norm(diff_p2g, axis=-1)                # (P, G)
    min_p2g = dist_p2g.min(axis=1).mean()                        # scalar

    # GT to Pred: for each GT point, find nearest pred point
    min_g2p = dist_p2g.min(axis=0).mean()                        # scalar

    return (min_p2g + min_g2p) / 2.0


def compute_ap_at_threshold(pred_polylines, gt_polylines, threshold):
    """Compute Average Precision at a given distance threshold.

    A predicted polyline is a true positive if its Chamfer distance to the
    nearest unmatched GT polyline is below the threshold.

    Args:
        pred_polylines: List of predicted polylines.
        gt_polylines: List of GT polylines.
        threshold: Distance threshold in pixels or meters.

    Returns:
        AP value (float).
    """
    if len(gt_polylines) == 0:
        return 1.0 if len(pred_polylines) == 0 else 0.0
    if len(pred_polylines) == 0:
        return 0.0

    num_gt = len(gt_polylines)
    gt_matched = [False] * num_gt

    # Compute distance matrix between pred and gt polylines
    dist_matrix = np.zeros((len(pred_polylines), num_gt))
    for i, pred_pl in enumerate(pred_polylines):
        for j, gt_pl in enumerate(gt_polylines):
            dist_matrix[i, j] = chamfer_distance_polylines([pred_pl], [gt_pl])

    # Sort predictions by confidence (here we use length as proxy for confidence)
    pred_lengths = [len(p) for p in pred_polylines]
    sorted_indices = np.argsort(pred_lengths)[::-1]

    tp = np.zeros(len(pred_polylines))
    fp = np.zeros(len(pred_polylines))

    for rank, pred_idx in enumerate(sorted_indices):
        # Find closest unmatched GT
        min_dist = float("inf")
        min_gt_idx = -1

        for gt_idx in range(num_gt):
            if gt_matched[gt_idx]:
                continue
            d = dist_matrix[pred_idx, gt_idx]
            if d < min_dist:
                min_dist = d
                min_gt_idx = gt_idx

        if min_dist < threshold and min_gt_idx >= 0:
            tp[rank] = 1
            gt_matched[min_gt_idx] = True
        else:
            fp[rank] = 1

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP using all-points interpolation
    ap = 0.0
    for i in range(len(precisions)):
        if i == 0:
            ap += precisions[i] * recalls[i]
        else:
            ap += precisions[i] * (recalls[i] - recalls[i - 1])

    return ap


def evaluate_sample(
    pred_semantic, pred_instance, pred_direction,
    gt_semantic, gt_instance_map, gt_direction,
    config, ap_thresholds=(2.0, 5.0, 10.0),
):
    """Evaluate a single sample across all metrics.

    Args:
        pred_semantic: Predicted semantic logits (C, H, W) tensor.
        pred_instance: Predicted instance embeddings (E, H, W) tensor.
        pred_direction: Predicted directions (2, H, W) tensor.
        gt_semantic: GT semantic mask (C, H, W) tensor.
        gt_instance_map: GT instance map (H, W) tensor.
        gt_direction: GT direction map (2, H, W) tensor.
        config: Configuration dict with xbound, ybound.
        ap_thresholds: Tuple of AP distance thresholds.

    Returns:
        Dict with per-class metrics.
    """
    num_classes = pred_semantic.shape[0]
    xbound = config.get("xbound", [-30.0, 30.0, 0.3])
    ybound = config.get("ybound", [-15.0, 15.0, 0.3])

    # Convert predictions to numpy
    sem_prob = torch.sigmoid(pred_semantic).cpu().numpy()
    inst_emb = pred_instance.cpu().numpy()
    dir_pred = pred_direction.cpu().numpy()

    # Vectorize predictions
    pred_vectorized = vectorize_predictions(
        sem_prob, inst_emb, dir_pred,
        semantic_threshold=0.5,
        dbscan_eps=1.5,
        dbscan_min_samples=5,
        nms_threshold=5.0,
        xbound=xbound,
        ybound=ybound,
    )

    # Vectorize ground truth
    gt_sem_np = gt_semantic.cpu().numpy()
    gt_inst_np = gt_instance_map.cpu().numpy()
    gt_dir_np = gt_direction.cpu().numpy()

    gt_vectorized = vectorize_predictions(
        gt_sem_np, np.zeros_like(inst_emb), gt_dir_np,
        semantic_threshold=0.5,
        dbscan_eps=1.5,
        dbscan_min_samples=3,
        nms_threshold=2.0,
        xbound=xbound,
        ybound=ybound,
    )

    results = {}
    for c in range(num_classes):
        pred_polys = pred_vectorized.get(c, [])
        gt_polys = gt_vectorized.get(c, [])

        # Chamfer distance
        cd = chamfer_distance_polylines(pred_polys, gt_polys)

        # AP at thresholds
        ap_values = {}
        for thresh in ap_thresholds:
            ap = compute_ap_at_threshold(pred_polys, gt_polys, thresh)
            ap_values[thresh] = ap

        results[c] = {
            "chamfer_distance": cd,
            "ap": ap_values,
            "num_pred": len(pred_polys),
            "num_gt": len(gt_polys),
        }

    return results


@torch.no_grad()
def evaluate_full(model, val_loader, device, config):
    """Run full evaluation on the validation set.

    Args:
        model: Trained HDMapNet model.
        val_loader: Validation data loader.
        device: Torch device.
        config: Configuration dict.

    Returns:
        Aggregated evaluation metrics dict.
    """
    model.eval()
    num_classes = config.get("num_classes", 3)
    ap_thresholds = (2.0, 5.0, 10.0)

    # Accumulators for IoU
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    # Accumulators for Chamfer and AP
    all_chamfer = {c: [] for c in range(num_classes)}
    all_ap = {c: {t: [] for t in ap_thresholds} for c in range(num_classes)}
    num_samples = 0

    for batch_idx, batch in enumerate(val_loader):
        images = batch["images"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        semantic_gt = batch["semantic_map"].to(device)
        instance_gt = batch["instance_map"].to(device)
        direction_gt = batch["direction_map"].to(device)

        predictions = model(images, intrinsics, extrinsics)
        B = images.shape[0]

        # IoU computation
        iou_dict = compute_iou_per_class(predictions["semantic"], semantic_gt)
        pred_binary = (torch.sigmoid(predictions["semantic"]) > 0.5).float()
        for c in range(num_classes):
            pred_c = pred_binary[:, c].reshape(-1)
            gt_c = semantic_gt[:, c].reshape(-1)
            total_intersection[c] += (pred_c * gt_c).sum().item()
            total_union[c] += ((pred_c + gt_c) > 0).float().sum().item()

        # Per-sample vectorized metrics (only every 10th sample for speed)
        if batch_idx % 10 == 0:
            for b in range(min(B, 2)):  # Evaluate max 2 samples per batch for speed
                sample_results = evaluate_sample(
                    predictions["semantic"][b],
                    predictions["instance"][b],
                    predictions["direction"][b],
                    semantic_gt[b],
                    instance_gt[b],
                    direction_gt[b],
                    config,
                    ap_thresholds=ap_thresholds,
                )
                for c in range(num_classes):
                    if c in sample_results:
                        cd = sample_results[c]["chamfer_distance"]
                        if not np.isinf(cd):
                            all_chamfer[c].append(cd)
                        for t in ap_thresholds:
                            all_ap[c][t].append(sample_results[c]["ap"][t])

                num_samples += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  Evaluated {batch_idx + 1}/{len(val_loader)} batches...")

    # Compute final metrics
    iou_results = {}
    for c in range(num_classes):
        if total_union[c] > 0:
            iou_results[c] = total_intersection[c] / total_union[c]
        else:
            iou_results[c] = 0.0

    chamfer_results = {}
    for c in range(num_classes):
        if len(all_chamfer[c]) > 0:
            chamfer_results[c] = np.mean(all_chamfer[c])
        else:
            chamfer_results[c] = float("inf")

    ap_results = {}
    for c in range(num_classes):
        ap_results[c] = {}
        for t in ap_thresholds:
            if len(all_ap[c][t]) > 0:
                ap_results[c][t] = np.mean(all_ap[c][t])
            else:
                ap_results[c][t] = 0.0

    return {
        "iou": iou_results,
        "chamfer": chamfer_results,
        "ap": ap_results,
        "num_samples_vectorized": num_samples,
    }


def print_results_table(results, ap_thresholds=(2.0, 5.0, 10.0)):
    """Print evaluation results in a formatted table.

    Args:
        results: Dict from evaluate_full().
        ap_thresholds: AP thresholds used.
    """
    num_classes = len(results["iou"])

    print("\n" + "=" * 80)
    print("HDMapNet Evaluation Results")
    print("=" * 80)

    # Header
    header = f"{'Class':<12} {'IoU':<8}"
    header += f"{'CD':<10}"
    for t in ap_thresholds:
        header += f"{'AP@' + str(t):<10}"
    print(header)
    print("-" * 80)

    # Per-class rows
    mean_iou = 0.0
    mean_cd = 0.0
    mean_ap = {t: 0.0 for t in ap_thresholds}
    valid_classes = 0

    for c in range(num_classes):
        row = f"{CLASS_NAMES[c]:<12}"
        iou = results["iou"].get(c, 0.0)
        cd = results["chamfer"].get(c, float("inf"))
        row += f"{iou:.4f}  "
        row += f"{cd:.4f}    " if not np.isinf(cd) else f"{'inf':<10}"

        mean_iou += iou
        if not np.isinf(cd):
            mean_cd += cd

        for t in ap_thresholds:
            ap = results["ap"].get(c, {}).get(t, 0.0)
            row += f"{ap:.4f}    "
            mean_ap[t] += ap

        valid_classes += 1
        print(row)

    # Mean row
    print("-" * 80)
    row = f"{'Mean':<12}"
    row += f"{mean_iou / max(valid_classes, 1):.4f}  "
    row += f"{mean_cd / max(valid_classes, 1):.4f}    "
    for t in ap_thresholds:
        row += f"{mean_ap[t] / max(valid_classes, 1):.4f}    "
    print(row)
    print("=" * 80)

    print(f"\nTotal samples evaluated (vectorized): {results['num_samples_vectorized']}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="HDMapNet Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    # Merge with defaults
    from .train import get_default_config
    full_config = get_default_config()
    full_config.update(config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    from .train import build_model
    model = build_model(full_config).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Build validation loader
    val_dataset = NuScenesHDMapDataset(
        dataroot=full_config["dataroot"],
        ann_file=full_config["val_ann_file"],
        image_size=tuple(full_config["image_size"]),
        xbound=tuple(full_config["xbound"]),
        ybound=tuple(full_config["ybound"]),
        num_classes=full_config["num_classes"],
        augment=False,
        thickness=full_config["thickness"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=full_config["batch_size"],
        shuffle=False,
        num_workers=full_config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Validation set: {len(val_dataset)} samples")

    # Run evaluation
    results = evaluate_full(model, val_loader, device, full_config)

    # Print results
    print_results_table(results)


if __name__ == "__main__":
    main()
