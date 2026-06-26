"""
Full training script for PointNet++ 3D object detection / classification / segmentation.

Supports:
- Config file or command-line arguments
- Mixed precision training (torch.cuda.amp)
- CosineAnnealingLR scheduler
- TensorBoard logging
- Best model checkpointing
- Multi-task support (classification, detection, segmentation)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .model import PointNetPPClassification, PointNetPPDetection, PointNetPPSegmentation
from .losses import (
    PointNetPPClassificationLoss,
    PointNetPPDetectionLoss,
    PointNetPPSegmentationLoss,
)
from .dataset import KITTIDataset, NuScenesDataset, collate_fn


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PointNet++ Training Script for 3D Object Detection"
    )

    # Config file (overrides defaults, overridden by CLI args)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")

    # Task
    parser.add_argument("--task", type=str, default="detection",
                        choices=["classification", "detection", "segmentation"],
                        help="Training task")

    # Data
    parser.add_argument("--dataset", type=str, default="kitti",
                        choices=["kitti", "nuscenes"],
                        help="Dataset to use")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--npoints", type=int, default=16384,
                        help="Number of input points")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of classes (including background for detection)")
    parser.add_argument("--num_seg_classes", type=int, default=20,
                        help="Number of segmentation classes")

    # Model
    parser.add_argument("--in_channels", type=int, default=4,
                        help="Input channels (3 for xyz, 4 for xyz+intensity)")
    parser.add_argument("--use_msg", action="store_true",
                        help="Use multi-scale grouping")
    parser.add_argument("--num_angle_bins", type=int, default=12,
                        help="Number of angle bins for detection")

    # Training
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--lr_min", type=float, default=1e-5,
                        help="Minimum learning rate for cosine annealing")
    parser.add_argument("--grad_clip", type=float, default=10.0,
                        help="Gradient clipping max norm")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker count")

    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log every N iterations")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Load config file if provided
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        # Override defaults with config, but CLI args take priority
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    return args


def build_model(args):
    """Build the model based on the task."""
    if args.task == "classification":
        model = PointNetPPClassification(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            use_msg=args.use_msg,
        )
    elif args.task == "detection":
        model = PointNetPPDetection(
            num_classes=args.num_classes,
            in_channels=args.in_channels,
            num_angle_bins=args.num_angle_bins,
        )
    elif args.task == "segmentation":
        model = PointNetPPSegmentation(
            num_seg_classes=args.num_seg_classes,
            in_channels=args.in_channels,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return model


def build_criterion(args):
    """Build the loss function based on the task."""
    if args.task == "classification":
        return PointNetPPClassificationLoss()
    elif args.task == "detection":
        return PointNetPPDetectionLoss(num_angle_bins=args.num_angle_bins)
    elif args.task == "segmentation":
        return PointNetPPSegmentationLoss(num_classes=args.num_seg_classes)
    else:
        raise ValueError(f"Unknown task: {args.task}")


def build_dataset(args, split: str):
    """Build dataset based on configuration."""
    if args.dataset == "kitti":
        return KITTIDataset(
            root=args.data_root,
            split=split,
            npoints=args.npoints,
            augment=(split == "train"),
        )
    elif args.dataset == "nuscenes":
        return NuScenesDataset(
            root=args.data_root,
            split=split,
            npoints=args.npoints,
            augment=(split == "train"),
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch,
    args,
    writer=None,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_start = time.time()

    for batch_idx, batch_data in enumerate(dataloader):
        xyz = batch_data["xyz"].to(device)
        features = batch_data["features"].to(device)
        targets = batch_data["targets"]

        # Move targets to device
        if isinstance(targets, dict):
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in targets.items()}
        else:
            targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass with optional mixed precision
        with autocast(enabled=args.use_amp):
            if args.task == "classification":
                # For classification, features beyond xyz
                extra_feat = features if args.in_channels > 3 else None
                predictions = model(xyz, extra_feat)
                loss_dict = criterion(predictions, targets["cls"])
            elif args.task == "detection":
                extra_feat = features if args.in_channels > 3 else None
                predictions = model(xyz, extra_feat)
                loss_dict = criterion(predictions, targets)
            elif args.task == "segmentation":
                extra_feat = features if args.in_channels > 3 else None
                predictions = model(xyz, extra_feat)
                loss_dict = criterion(predictions, targets["cls"])

            loss = loss_dict["total_loss"]

        # Backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Logging
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - epoch_start
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"Time: {elapsed:.1f}s"
            )
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                for key, val in loss_dict.items():
                    if key != "total_loss" and isinstance(val, torch.Tensor):
                        writer.add_scalar(f"train/{key}", val.item(), global_step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, criterion, dataloader, device, args):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    for batch_data in dataloader:
        xyz = batch_data["xyz"].to(device)
        features = batch_data["features"].to(device)
        targets = batch_data["targets"]

        if isinstance(targets, dict):
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in targets.items()}
        else:
            targets = targets.to(device)

        if args.task == "classification":
            extra_feat = features if args.in_channels > 3 else None
            predictions = model(xyz, extra_feat)
            loss_dict = criterion(predictions, targets["cls"])
            # Accuracy
            pred_cls = torch.argmax(predictions, dim=-1)
            correct += (pred_cls == targets["cls"]).sum().item()
            total += targets["cls"].numel()
        elif args.task == "detection":
            extra_feat = features if args.in_channels > 3 else None
            predictions = model(xyz, extra_feat)
            loss_dict = criterion(predictions, targets)
        elif args.task == "segmentation":
            extra_feat = features if args.in_channels > 3 else None
            predictions = model(xyz, extra_feat)
            loss_dict = criterion(predictions, targets["cls"])
            # Per-point accuracy
            pred_cls = torch.argmax(predictions, dim=-1)
            correct += (pred_cls == targets["cls"]).sum().item()
            total += targets["cls"].numel()

        total_loss += loss_dict["total_loss"].item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1) if total > 0 else 0.0

    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_loss, path):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, path)


def main():
    """Main training entry point."""
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build datasets
    train_dataset = build_dataset(args, "train")
    val_dataset = build_dataset(args, "val")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Build data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Build model
    model = build_model(args)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Build loss
    criterion = build_criterion(args)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min,
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    # TensorBoard
    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_loss"]
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Task: {args.task}, Dataset: {args.dataset}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device,
            epoch, args, writer
        )

        # Validate
        val_loss, val_acc = validate(model, criterion, val_loader, device, args)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # Logging
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/val_accuracy", val_acc, epoch)
            writer.add_scalar("epoch/lr", current_lr, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_val_loss,
                checkpoint_dir / "best_model.pth"
            )
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_val_loss,
                checkpoint_dir / f"epoch_{epoch:04d}.pth"
            )

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, scaler, args.epochs - 1, best_val_loss,
        checkpoint_dir / "final_model.pth"
    )

    if writer is not None:
        writer.close()

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
