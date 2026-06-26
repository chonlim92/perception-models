"""
Radar Occupancy Grid Mapping — Inference Script

Run occupancy prediction on single samples and visualize results.
Supports both classical ISM and neural models.
"""

import argparse
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from model import build_model, ClassicalISM, PillarOccNet, TemporalPillarOccNet
from dataset import RadarOccupancyDataset


def visualize_occupancy(pred_occ, gt_occ=None, radar_points=None, config=None,
                       save_path=None, title="Radar Occupancy Grid"):
    """Visualize predicted occupancy grid.

    Args:
        pred_occ: (H, W) predicted occupancy (0=free, 1=occupied)
        gt_occ: (H, W) optional ground truth
        radar_points: (N, 6) optional radar points to overlay
        config: config dict for grid parameters
        save_path: path to save figure
    """
    n_plots = 1 + (gt_occ is not None) + (radar_points is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    cmap = ListedColormap(['white', 'green', 'black', 'gray'])

    plot_idx = 0

    if radar_points is not None:
        ax = axes[plot_idx]
        ax.set_title("Radar Points (BEV)")
        ax.set_xlim(config["grid"]["x_range"])
        ax.set_ylim(config["grid"]["y_range"])
        ax.scatter(radar_points[:, 0], radar_points[:, 1],
                  c=radar_points[:, 3], cmap='viridis', s=2, alpha=0.7)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    ax = axes[plot_idx]
    ax.set_title("Predicted Occupancy")
    vis_pred = np.zeros_like(pred_occ, dtype=np.float32)
    vis_pred[pred_occ == 0] = 0.2  # Free = light
    vis_pred[pred_occ == 1] = 0.9  # Occupied = dark
    ax.imshow(vis_pred.T, origin='lower', cmap='RdYlGn_r',
             vmin=0, vmax=1, aspect='equal')
    ax.set_xlabel("X cells")
    ax.set_ylabel("Y cells")
    plot_idx += 1

    if gt_occ is not None:
        ax = axes[plot_idx]
        ax.set_title("Ground Truth Occupancy")
        vis_gt = np.zeros_like(gt_occ, dtype=np.float32)
        vis_gt[gt_occ == 0] = 0.2
        vis_gt[gt_occ == 1] = 0.9
        vis_gt[gt_occ == 2] = 0.5  # Unknown = gray
        ax.imshow(vis_gt.T, origin='lower', cmap='RdYlGn_r',
                 vmin=0, vmax=1, aspect='equal')
        ax.set_xlabel("X cells")
        ax.set_ylabel("Y cells")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def run_classical_inference(config, sample_idx=0, num_frames=10, save_dir=None):
    """Run classical ISM inference and accumulate over multiple frames."""
    print("Running Classical ISM inference...")

    dataset = RadarOccupancyDataset(config, split="val")
    ism = ClassicalISM(config)

    sample = dataset[sample_idx]
    radar_points = dataset._get_radar_points(dataset.samples[sample_idx])

    ism.update(radar_points)

    prob_map = ism.get_occupancy_probability()
    pred_occ = (prob_map > 0.5).astype(np.int64)
    gt_occ = sample["occupancy_gt"].numpy()

    valid = gt_occ != 2
    if valid.any():
        correct = (pred_occ[valid] == gt_occ[valid]).sum()
        accuracy = correct / valid.sum()
        print(f"  Accuracy: {accuracy:.4f}")

        occ_gt_mask = gt_occ[valid] == 1
        occ_pred_mask = pred_occ[valid] == 1
        tp = (occ_gt_mask & occ_pred_mask).sum()
        fp = (~occ_gt_mask & occ_pred_mask).sum()
        fn = (occ_gt_mask & ~occ_pred_mask).sum()
        occ_iou = tp / max(tp + fp + fn, 1)
        print(f"  Occupied IoU: {occ_iou:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        visualize_occupancy(
            pred_occ, gt_occ, radar_points, config,
            save_path=os.path.join(save_dir, f"classical_sample_{sample_idx}.png"),
            title=f"Classical ISM — Sample {sample_idx}"
        )

    return pred_occ, prob_map


def run_neural_inference(config, checkpoint_path, sample_idx=0, save_dir=None):
    """Run neural model inference on a single sample."""
    print("Running Neural model inference...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = config["model"]["type"]
    if model_type == "temporal_pillar_occ_net":
        model = TemporalPillarOccNet(config).to(device)
    else:
        model = PillarOccNet(config).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")

    model.eval()

    dataset = RadarOccupancyDataset(config, split="val")
    sample = dataset[sample_idx]

    pillar_features = sample["pillar_features"].unsqueeze(0).to(device)
    pillar_indices = sample["pillar_indices"].unsqueeze(0).to(device)
    num_pillars = sample["num_pillars"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pillar_features, pillar_indices, num_pillars)

    pred_prob = torch.sigmoid(outputs["occupancy"][0, 0]).cpu().numpy()
    pred_occ = (pred_prob > 0.5).astype(np.int64)
    gt_occ = sample["occupancy_gt"].numpy()

    valid = gt_occ != 2
    if valid.any():
        correct = (pred_occ[valid] == gt_occ[valid]).sum()
        accuracy = correct / valid.sum()
        print(f"  Accuracy: {accuracy:.4f}")

    radar_points = dataset._get_radar_points(dataset.samples[sample_idx])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        visualize_occupancy(
            pred_occ, gt_occ, radar_points, config,
            save_path=os.path.join(save_dir, f"neural_sample_{sample_idx}.png"),
            title=f"Neural Occupancy — Sample {sample_idx}"
        )

    return pred_occ, pred_prob


def main():
    parser = argparse.ArgumentParser(description="Radar Occupancy Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", type=str, default="neural",
                       choices=["neural", "classical", "both"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="inference_results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.mode in ["classical", "both"]:
        run_classical_inference(config, args.sample_idx, save_dir=args.save_dir)

    if args.mode in ["neural", "both"]:
        run_neural_inference(config, args.checkpoint, args.sample_idx,
                           save_dir=args.save_dir)


if __name__ == "__main__":
    main()
