"""
Radar Occupancy Grid Mapping — Training Script

Trains PillarOccNet or TemporalPillarOccNet on nuScenes radar data.
"""

import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from model import PillarOccNet, TemporalPillarOccNet, build_model
from dataset import RadarOccupancyDataset, TemporalRadarOccupancyDataset


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in occupancy prediction."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) raw logits
            target: (B, H, W) with values 0=free, 1=occupied, 2=unknown
        """
        valid = target != 2  # Ignore unknown cells
        if not valid.any():
            return torch.tensor(0.0, device=pred.device)

        pred_flat = pred.squeeze(1)[valid]
        target_flat = target[valid].float()

        p = torch.sigmoid(pred_flat)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_flat, target_flat, reduction='none'
        )

        p_t = p * target_flat + (1 - p) * (1 - target_flat)
        alpha_t = self.alpha * target_flat + (1 - self.alpha) * (1 - target_flat)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = (focal_weight * ce_loss).mean()
        return loss


class SemanticLoss(nn.Module):
    """Weighted cross-entropy for semantic occupancy classes."""

    def __init__(self, class_weights=None, ignore_index=2):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred: (B, K, H, W) class logits
            target: (B, H, W) class indices
        """
        return nn.functional.cross_entropy(
            pred, target, weight=self.weight, ignore_index=self.ignore_index
        )


def compute_iou(pred_occ, gt_occ):
    """Compute IoU for occupancy and free space."""
    valid = gt_occ != 2

    pred_binary = (torch.sigmoid(pred_occ.squeeze(1)) > 0.5).long()
    gt_binary = gt_occ.clone()

    occ_pred = pred_binary[valid] == 1
    occ_gt = gt_binary[valid] == 1
    free_pred = pred_binary[valid] == 0
    free_gt = gt_binary[valid] == 0

    occ_intersection = (occ_pred & occ_gt).sum().float()
    occ_union = (occ_pred | occ_gt).sum().float()
    occ_iou = occ_intersection / (occ_union + 1e-6)

    free_intersection = (free_pred & free_gt).sum().float()
    free_union = (free_pred | free_gt).sum().float()
    free_iou = free_intersection / (free_union + 1e-6)

    return occ_iou.item(), free_iou.item()


def train_epoch(model, dataloader, optimizer, scaler, focal_loss, semantic_loss,
                config, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_occ_iou = 0.0
    total_free_iou = 0.0
    num_batches = 0

    loss_cfg = config["training"]["loss"]
    occ_weight = loss_cfg["occupancy_weight"]
    sem_weight = loss_cfg["semantic_weight"]

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        pillar_features = batch["pillar_features"].to(device)
        pillar_indices = batch["pillar_indices"].to(device)
        num_pillars = batch["num_pillars"].to(device)
        occ_gt = batch["occupancy_gt"].to(device)

        with autocast():
            outputs = model(pillar_features, pillar_indices, num_pillars)

            loss_occ = focal_loss(outputs["occupancy"], occ_gt)
            loss = occ_weight * loss_occ

            if "semantics" in outputs:
                loss_sem = semantic_loss(outputs["semantics"], occ_gt)
                loss += sem_weight * loss_sem

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            occ_iou, free_iou = compute_iou(outputs["occupancy"], occ_gt)

        total_loss += loss.item()
        total_occ_iou += occ_iou
        total_free_iou += free_iou
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"OccIoU: {occ_iou:.3f} FreeIoU: {free_iou:.3f}")

    return {
        "loss": total_loss / num_batches,
        "occ_iou": total_occ_iou / num_batches,
        "free_iou": total_free_iou / num_batches,
    }


def validate(model, dataloader, focal_loss, semantic_loss, config, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_occ_iou = 0.0
    total_free_iou = 0.0
    num_batches = 0

    loss_cfg = config["training"]["loss"]

    with torch.no_grad():
        for batch in dataloader:
            pillar_features = batch["pillar_features"].to(device)
            pillar_indices = batch["pillar_indices"].to(device)
            num_pillars = batch["num_pillars"].to(device)
            occ_gt = batch["occupancy_gt"].to(device)

            outputs = model(pillar_features, pillar_indices, num_pillars)

            loss_occ = focal_loss(outputs["occupancy"], occ_gt)
            loss = loss_cfg["occupancy_weight"] * loss_occ

            occ_iou, free_iou = compute_iou(outputs["occupancy"], occ_gt)

            total_loss += loss.item()
            total_occ_iou += occ_iou
            total_free_iou += free_iou
            num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "occ_iou": total_occ_iou / max(num_batches, 1),
        "free_iou": total_free_iou / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Radar Occupancy Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="outputs/radar_occ",
                       help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_type = config["model"]["type"]
    is_temporal = model_type == "temporal_pillar_occ_net"

    if is_temporal:
        train_dataset = TemporalRadarOccupancyDataset(config, split="train")
        val_dataset = TemporalRadarOccupancyDataset(config, split="val")
        model = TemporalPillarOccNet(config).to(device)
    else:
        train_dataset = RadarOccupancyDataset(config, split="train")
        val_dataset = RadarOccupancyDataset(config, split="val")
        model = PillarOccNet(config).to(device)

    print(f"Model: {model_type}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    train_cfg = config["training"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["optimizer"]["lr"],
        weight_decay=train_cfg["optimizer"]["weight_decay"],
        betas=tuple(train_cfg["optimizer"]["betas"]),
    )

    num_epochs = train_cfg["num_epochs"]
    warmup_epochs = train_cfg["scheduler"]["warmup_epochs"]
    min_lr = train_cfg["scheduler"]["min_lr"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return min_lr / train_cfg["optimizer"]["lr"] + \
               (1 - min_lr / train_cfg["optimizer"]["lr"]) * \
               0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_cfg = train_cfg["loss"]
    focal_loss = FocalLoss(alpha=loss_cfg["focal_alpha"], gamma=loss_cfg["focal_gamma"])
    semantic_loss = SemanticLoss(class_weights=loss_cfg.get("class_weights"))
    semantic_loss = semantic_loss.to(device)

    scaler = GradScaler(enabled=config["hardware"]["mixed_precision"])

    start_epoch = 0
    best_iou = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler,
            focal_loss, semantic_loss, config, device, epoch
        )

        val_metrics = validate(
            model, val_loader, focal_loss, semantic_loss, config, device
        )

        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        mean_iou = (val_metrics["occ_iou"] + val_metrics["free_iou"]) / 2

        print(f"\nEpoch {epoch}/{num_epochs} ({elapsed:.1f}s) LR: {current_lr:.6f}")
        print(f"  Train — Loss: {train_metrics['loss']:.4f} "
              f"OccIoU: {train_metrics['occ_iou']:.3f} "
              f"FreeIoU: {train_metrics['free_iou']:.3f}")
        print(f"  Val   — Loss: {val_metrics['loss']:.4f} "
              f"OccIoU: {val_metrics['occ_iou']:.3f} "
              f"FreeIoU: {val_metrics['free_iou']:.3f} "
              f"mIoU: {mean_iou:.3f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_iou": best_iou,
            "config": config,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "latest.pth"))

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pth"))
            print(f"  * New best mIoU: {best_iou:.3f}")

    print(f"\nTraining complete. Best mIoU: {best_iou:.3f}")


if __name__ == "__main__":
    main()
