"""
Radar Occupancy Grid Mapping — Evaluation Script

Evaluates occupancy prediction quality with IoU, accuracy, and per-class metrics.
"""

import argparse
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model, PillarOccNet, TemporalPillarOccNet
from dataset import RadarOccupancyDataset, TemporalRadarOccupancyDataset


class OccupancyMetrics:
    """Compute occupancy grid evaluation metrics."""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.total_cells = 0
        self.correct_cells = 0

    def update(self, pred, gt):
        """
        Args:
            pred: (H, W) predicted occupancy (0=free, 1=occupied)
            gt: (H, W) ground truth (0=free, 1=occupied, 2=unknown)
        """
        valid = gt != 2
        pred_valid = pred[valid]
        gt_valid = gt[valid]

        for gt_class in range(self.num_classes):
            for pred_class in range(self.num_classes):
                self.confusion_matrix[gt_class, pred_class] += \
                    ((gt_valid == gt_class) & (pred_valid == pred_class)).sum()

        self.total_cells += valid.sum()
        self.correct_cells += (pred_valid == gt_valid).sum()

    def compute(self):
        """Compute all metrics from confusion matrix."""
        results = {}

        results["accuracy"] = self.correct_cells / max(self.total_cells, 1)

        class_names = ["free", "occupied"]
        ious = []
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp
            iou = tp / max(tp + fp + fn, 1)
            ious.append(iou)
            results[f"{class_names[c]}_iou"] = iou
            results[f"{class_names[c]}_precision"] = tp / max(tp + fp, 1)
            results[f"{class_names[c]}_recall"] = tp / max(tp + fn, 1)

        results["mean_iou"] = np.mean(ious)

        return results

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self.total_cells = 0
        self.correct_cells = 0


def evaluate_neural(model, dataloader, device, threshold=0.5):
    """Evaluate neural occupancy model."""
    model.eval()
    metrics = OccupancyMetrics(num_classes=2)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pillar_features = batch["pillar_features"].to(device)
            pillar_indices = batch["pillar_indices"].to(device)
            num_pillars = batch["num_pillars"].to(device)
            occ_gt = batch["occupancy_gt"].numpy()

            outputs = model(pillar_features, pillar_indices, num_pillars)
            pred_occ = (torch.sigmoid(outputs["occupancy"].squeeze(1)) > threshold)
            pred_occ = pred_occ.cpu().numpy().astype(np.int64)

            for b in range(len(occ_gt)):
                metrics.update(pred_occ[b], occ_gt[b])

    return metrics.compute()


def evaluate_classical(config, nusc, val_samples, num_frames=20):
    """Evaluate classical ISM occupancy mapping."""
    from model import ClassicalISM

    ism = ClassicalISM(config)
    metrics = OccupancyMetrics(num_classes=2)

    dataset = RadarOccupancyDataset(config, split="val", nusc=nusc)

    for idx in tqdm(range(min(len(dataset), 200)), desc="Classical ISM"):
        sample = dataset[idx]
        occ_gt = sample["occupancy_gt"].numpy()

        ism.reset()

        radar_points = dataset._get_radar_points(dataset.samples[idx])
        ism.update(radar_points)

        pred_prob = ism.get_occupancy_probability()
        pred_occ = (pred_prob > 0.5).astype(np.int64)

        metrics.update(pred_occ, occ_gt)

    return metrics.compute()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Radar Occupancy Model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint (for neural models)")
    parser.add_argument("--mode", type=str, default="neural",
                       choices=["neural", "classical", "both"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ["neural", "both"]:
        print("=" * 60)
        print("Evaluating Neural Model")
        print("=" * 60)

        model_type = config["model"]["type"]
        is_temporal = model_type == "temporal_pillar_occ_net"

        if is_temporal:
            model = TemporalPillarOccNet(config).to(device)
            val_dataset = TemporalRadarOccupancyDataset(config, split="val")
        else:
            model = PillarOccNet(config).to(device)
            val_dataset = RadarOccupancyDataset(config, split="val")

        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint: {args.checkpoint}")

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["hardware"]["num_workers"],
        )

        neural_results = evaluate_neural(model, val_loader, device, args.threshold)

        print("\nNeural Model Results:")
        print("-" * 40)
        print(f"  Mean IoU:          {neural_results['mean_iou']:.4f}")
        print(f"  Occupied IoU:      {neural_results['occupied_iou']:.4f}")
        print(f"  Free Space IoU:    {neural_results['free_iou']:.4f}")
        print(f"  Accuracy:          {neural_results['accuracy']:.4f}")
        print(f"  Occupied Precision: {neural_results['occupied_precision']:.4f}")
        print(f"  Occupied Recall:   {neural_results['occupied_recall']:.4f}")

    if args.mode in ["classical", "both"]:
        print("\n" + "=" * 60)
        print("Evaluating Classical ISM")
        print("=" * 60)

        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(
            version=config["dataset"]["version"],
            dataroot=config["dataset"]["root"],
            verbose=False,
        )

        classical_results = evaluate_classical(config, nusc, None)

        print("\nClassical ISM Results:")
        print("-" * 40)
        print(f"  Mean IoU:          {classical_results['mean_iou']:.4f}")
        print(f"  Occupied IoU:      {classical_results['occupied_iou']:.4f}")
        print(f"  Free Space IoU:    {classical_results['free_iou']:.4f}")
        print(f"  Accuracy:          {classical_results['accuracy']:.4f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
