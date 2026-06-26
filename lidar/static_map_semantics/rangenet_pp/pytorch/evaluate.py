"""Evaluation script for RangeNet++ on SemanticKITTI validation set.

Runs inference on validation sequence (08), applies KNN post-processing,
and reports per-class and mean IoU metrics with timing information.

Usage:
    python evaluate.py --checkpoint best_model.pth --data_root /path/to/semantickitti
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import RangeNetPP
from .dataset import (
    SemanticKITTIRangeDataset,
    SemanticKITTIRangeInferenceDataset,
    SEMANTICKITTI_CLASS_NAMES,
    VAL_SEQUENCES,
)
from .knn_postprocess import knn_postprocess_numpy_fast
from .train import IoUMetric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RangeNet++ on SemanticKITTI")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to SemanticKITTI dataset root")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 recommended for KNN post-processing)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--knn", action="store_true",
                        help="Apply KNN post-processing")
    parser.add_argument("--knn_k", type=int, default=5,
                        help="Number of neighbors for KNN")
    parser.add_argument("--knn_radius", type=float, default=1.0,
                        help="Search radius for KNN (meters)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> RangeNetPP:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        config: Model configuration dictionary.
        device: Target device.

    Returns:
        Loaded model in eval mode.
    """
    model = RangeNetPP(config=config)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both full checkpoint and raw state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if saved from DDP
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    model.eval()
    return model


def evaluate_range_image_only(
    model: RangeNetPP,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 20,
) -> dict:
    """Evaluate using only range image predictions (no KNN).

    Args:
        model: Trained model.
        dataloader: Validation dataloader.
        device: Computation device.
        num_classes: Number of semantic classes.

    Returns:
        Dictionary with metrics.
    """
    iou_metric = IoUMetric(num_classes=num_classes, ignore_index=0)
    total_time = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            range_image = batch["range_image"].to(device, non_blocking=True)
            label_image = batch["label_image"].to(device, non_blocking=True)

            start_t = time.time()
            logits = model(range_image)
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.time() - start_t

            total_time += elapsed
            num_samples += range_image.shape[0]

            preds = logits.argmax(dim=1)
            iou_metric.update(preds, label_image)

    per_class_iou, miou = iou_metric.compute_iou()
    fps = num_samples / total_time if total_time > 0 else 0.0

    return {
        "per_class_iou": per_class_iou,
        "miou": miou,
        "fps": fps,
        "num_samples": num_samples,
        "total_time": total_time,
    }


def evaluate_with_knn(
    model: RangeNetPP,
    data_root: str,
    device: torch.device,
    height: int = 64,
    width: int = 2048,
    num_classes: int = 20,
    knn_k: int = 5,
    knn_radius: float = 1.0,
) -> dict:
    """Evaluate with KNN post-processing for refined per-point labels.

    Processes scans individually since KNN requires original 3D coordinates.

    Args:
        model: Trained model.
        data_root: Path to SemanticKITTI root.
        device: Computation device.
        height: Range image height.
        width: Range image width.
        num_classes: Number of classes.
        knn_k: Number of KNN neighbors.
        knn_radius: KNN search radius in meters.

    Returns:
        Dictionary with metrics.
    """
    from .spherical_projection import SphericalProjection
    from .dataset import SEMANTICKITTI_LABEL_MAP

    projector = SphericalProjection(height=height, width=width)

    # Build label lookup table
    label_lut = np.zeros(260 * 256, dtype=np.int32)
    for raw_id, train_id in SEMANTICKITTI_LABEL_MAP.items():
        label_lut[raw_id] = train_id

    # Collect validation scan paths
    sequences_dir = os.path.join(data_root, "sequences")
    scan_paths = []
    label_paths = []
    for seq in VAL_SEQUENCES:
        velodyne_dir = os.path.join(sequences_dir, seq, "velodyne")
        labels_dir = os.path.join(sequences_dir, seq, "labels")
        if not os.path.isdir(velodyne_dir):
            continue
        for scan_name in sorted(os.listdir(velodyne_dir)):
            if scan_name.endswith(".bin"):
                scan_paths.append(os.path.join(velodyne_dir, scan_name))
                label_paths.append(os.path.join(labels_dir, scan_name.replace(".bin", ".label")))

    # Per-point IoU computation
    iou_metric = IoUMetric(num_classes=num_classes, ignore_index=0)
    total_inference_time = 0.0
    total_knn_time = 0.0
    num_scans = 0

    with torch.no_grad():
        for scan_path, label_path in zip(scan_paths, label_paths):
            # Load point cloud
            points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            N = points.shape[0]

            # Load ground truth labels
            labels_raw = np.fromfile(label_path, dtype=np.uint32)
            semantic_labels = (labels_raw & 0xFFFF).astype(np.uint16)
            gt_labels = label_lut[semantic_labels]

            # Project to range image
            range_image, pixel_to_point, point_to_pixel = (
                projector.project_points_to_range_image_fast(points)
            )

            # Normalize
            normalized = range_image.copy()
            max_range = 80.0
            normalized[0] /= max_range
            normalized[1] /= max_range
            normalized[2] /= max_range
            normalized[3] /= max_range
            normalized[4] = np.clip(normalized[4], 0.0, 1.0)

            # Run inference
            input_tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(device)

            start_t = time.time()
            logits = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_time = time.time() - start_t
            total_inference_time += inference_time

            # Get range image predictions
            pred_image = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

            # Apply KNN post-processing
            start_t = time.time()
            refined_labels = knn_postprocess_numpy_fast(
                predicted_labels_image=pred_image,
                points=points,
                pixel_to_point=pixel_to_point,
                point_to_pixel=point_to_pixel,
                k=knn_k,
                search_radius=knn_radius,
                num_classes=num_classes,
            )
            knn_time = time.time() - start_t
            total_knn_time += knn_time

            # Update IoU (per-point)
            pred_tensor = torch.from_numpy(refined_labels).long().unsqueeze(0)
            gt_tensor = torch.from_numpy(gt_labels).long().unsqueeze(0)
            iou_metric.update(pred_tensor, gt_tensor)

            num_scans += 1
            if num_scans % 100 == 0:
                print(f"  Processed {num_scans}/{len(scan_paths)} scans...")

    per_class_iou, miou = iou_metric.compute_iou()
    fps_inference = num_scans / total_inference_time if total_inference_time > 0 else 0.0
    fps_total = num_scans / (total_inference_time + total_knn_time) if (total_inference_time + total_knn_time) > 0 else 0.0

    return {
        "per_class_iou": per_class_iou,
        "miou": miou,
        "fps_inference": fps_inference,
        "fps_total": fps_total,
        "avg_inference_ms": (total_inference_time / num_scans) * 1000 if num_scans > 0 else 0,
        "avg_knn_ms": (total_knn_time / num_scans) * 1000 if num_scans > 0 else 0,
        "num_scans": num_scans,
    }


def print_results(results: dict, title: str = "Evaluation Results"):
    """Print formatted evaluation results."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

    per_class_iou = results["per_class_iou"]
    miou = results["miou"]

    print(f"\n{'Class':<20} {'IoU (%)':>10}")
    print("-" * 32)
    for i in range(1, 20):
        name = SEMANTICKITTI_CLASS_NAMES[i]
        iou_pct = per_class_iou[i] * 100
        print(f"  {name:<18} {iou_pct:>8.1f}")
    print("-" * 32)
    print(f"  {'Mean IoU':<18} {miou * 100:>8.1f}")
    print()

    # Timing information
    if "fps" in results:
        print(f"  Inference FPS: {results['fps']:.1f}")
    if "fps_inference" in results:
        print(f"  Network inference FPS: {results['fps_inference']:.1f}")
        print(f"  Avg inference time: {results['avg_inference_ms']:.1f} ms")
    if "fps_total" in results:
        print(f"  Total FPS (with KNN): {results['fps_total']:.1f}")
        print(f"  Avg KNN time: {results['avg_knn_ms']:.1f} ms")
    print(f"  Evaluated scans: {results.get('num_scans', results.get('num_samples', 0))}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Evaluating RangeNet++ on device: {device}")

    # Load model
    model_config = {
        "in_channels": 5,
        "num_classes": args.num_classes,
        "height": args.height,
        "width": args.width,
    }
    model = load_model(args.checkpoint, model_config, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.knn:
        # Evaluation with KNN post-processing (per-point metrics)
        print(f"Running evaluation with KNN post-processing (K={args.knn_k}, radius={args.knn_radius}m)")
        results = evaluate_with_knn(
            model=model,
            data_root=args.data_root,
            device=device,
            height=args.height,
            width=args.width,
            num_classes=args.num_classes,
            knn_k=args.knn_k,
            knn_radius=args.knn_radius,
        )
        print_results(results, title="RangeNet++ + KNN Post-Processing")
    else:
        # Evaluation on range image only
        print("Running evaluation on range image (no KNN)")
        val_dataset = SemanticKITTIRangeDataset(
            root=args.data_root,
            split="val",
            height=args.height,
            width=args.width,
            augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        results = evaluate_range_image_only(
            model=model,
            dataloader=val_loader,
            device=device,
            num_classes=args.num_classes,
        )
        print_results(results, title="RangeNet++ (Range Image Only)")


if __name__ == "__main__":
    main()
