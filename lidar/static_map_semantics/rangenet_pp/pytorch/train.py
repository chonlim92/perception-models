"""Training script for RangeNet++ on SemanticKITTI.

Supports:
  - Single-GPU and multi-GPU (DistributedDataParallel) training
  - Mixed precision (torch.cuda.amp)
  - Polynomial or cosine learning rate decay
  - Periodic validation with mIoU reporting
  - Checkpoint saving and resuming

Usage:
  Single GPU:
    python train.py --data_root /path/to/semantickitti --epochs 150

  Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train.py --data_root /path/to/semantickitti --epochs 150
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Tuple

from .model import RangeNetPP
from .dataset import SemanticKITTIRangeDataset, SEMANTICKITTI_CLASS_NAMES
from .losses import CombinedLoss, get_default_semantickitti_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RangeNet++ on SemanticKITTI")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to SemanticKITTI dataset root")
    parser.add_argument("--height", type=int, default=64,
                        help="Range image height")
    parser.add_argument("--width", type=int, default=2048,
                        help="Range image width (2048 full, 1024 half)")

    # Model
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.01)

    # Training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_schedule", type=str, default="polynomial",
                        choices=["polynomial", "cosine"],
                        help="LR decay schedule")
    parser.add_argument("--poly_power", type=float, default=0.9,
                        help="Power for polynomial LR decay")
    parser.add_argument("--warmup_epochs", type=int, default=1)

    # Loss
    parser.add_argument("--loss_alpha", type=float, default=1.0,
                        help="Weight for cross-entropy loss")
    parser.add_argument("--loss_beta", type=float, default=1.5,
                        help="Weight for Lovasz loss")

    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--val_every", type=int, default=5,
                        help="Validate every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for DDP (set by torchrun)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file (overrides CLI args)")

    args = parser.parse_args()

    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


class IoUMetric:
    """Running IoU computation for semantic segmentation."""

    def __init__(self, num_classes: int, ignore_index: int = 0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes, dtype=np.int64)
        self.union = np.zeros(self.num_classes, dtype=np.int64)
        self.target_count = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with a batch of predictions and targets.

        Args:
            predictions: (B, H, W) predicted class indices.
            targets: (B, H, W) ground truth class indices.
        """
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()

        # Ignore unlabeled
        valid = target_np != self.ignore_index
        pred_np = pred_np[valid]
        target_np = target_np[valid]

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            pred_cls = pred_np == cls
            target_cls = target_np == cls
            self.intersection[cls] += np.logical_and(pred_cls, target_cls).sum()
            self.union[cls] += np.logical_or(pred_cls, target_cls).sum()
            self.target_count[cls] += target_cls.sum()

    def compute_iou(self) -> Tuple[np.ndarray, float]:
        """Compute per-class IoU and mean IoU.

        Returns:
            per_class_iou: (num_classes,) array
            miou: Mean IoU over evaluated classes (1-19)
        """
        per_class_iou = np.zeros(self.num_classes, dtype=np.float64)
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            if self.union[cls] > 0:
                per_class_iou[cls] = self.intersection[cls] / self.union[cls]

        # Mean IoU over evaluated classes (1 to num_classes-1)
        evaluated = per_class_iou[1:]
        evaluated_mask = self.union[1:] > 0
        if evaluated_mask.sum() > 0:
            miou = evaluated[evaluated_mask].mean()
        else:
            miou = 0.0

        return per_class_iou, miou


def get_lr_scheduler(optimizer, args, steps_per_epoch: int):
    """Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer.
        args: Training arguments.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        LR scheduler (step-based).
    """
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / max(warmup_steps, 1)
        else:
            # Decay phase
            progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
            if args.lr_schedule == "polynomial":
                return max(0.0, (1.0 - progress) ** args.poly_power)
            else:  # cosine
                return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    use_amp: bool,
    rank: int = 0,
) -> Dict[str, float]:
    """Train for one epoch.

    Returns:
        Dictionary with 'loss' and 'miou' for this epoch.
    """
    model.train()
    iou_metric = IoUMetric(num_classes=20, ignore_index=0)
    running_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        range_image = batch["range_image"].to(device, non_blocking=True)
        label_image = batch["label_image"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                logits = model(range_image)
                loss = criterion(logits, label_image)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(range_image)
            loss = criterion(logits, label_image)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Metrics
        running_loss += loss.item()
        num_batches += 1

        with torch.no_grad():
            preds = logits.argmax(dim=1)  # (B, H, W)
            iou_metric.update(preds, label_image)

        # Log progress
        if rank == 0 and (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss / num_batches
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.4f} LR: {current_lr:.6f}"
            )

    _, miou = iou_metric.compute_iou()
    avg_loss = running_loss / max(num_batches, 1)

    return {"loss": avg_loss, "miou": miou}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Run validation.

    Returns:
        Dictionary with 'loss', 'miou', and 'per_class_iou'.
    """
    model.eval()
    iou_metric = IoUMetric(num_classes=20, ignore_index=0)
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        range_image = batch["range_image"].to(device, non_blocking=True)
        label_image = batch["label_image"].to(device, non_blocking=True)

        if use_amp:
            with autocast():
                logits = model(range_image)
                loss = criterion(logits, label_image)
        else:
            logits = model(range_image)
            loss = criterion(logits, label_image)

        running_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=1)
        iou_metric.update(preds, label_image)

    per_class_iou, miou = iou_metric.compute_iou()
    avg_loss = running_loss / max(num_batches, 1)

    return {"loss": avg_loss, "miou": miou, "per_class_iou": per_class_iou}


def print_iou_table(per_class_iou: np.ndarray, miou: float):
    """Print a formatted table of per-class IoU results."""
    print("\n" + "=" * 50)
    print(f"{'Class':<20} {'IoU':>8}")
    print("-" * 50)
    for i in range(1, 20):
        name = SEMANTICKITTI_CLASS_NAMES[i]
        iou = per_class_iou[i] * 100
        print(f"{name:<20} {iou:>7.1f}%")
    print("-" * 50)
    print(f"{'Mean IoU':<20} {miou * 100:>7.1f}%")
    print("=" * 50 + "\n")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    miou: float,
    save_path: str,
    is_ddp: bool = False,
):
    """Save training checkpoint."""
    state_dict = model.module.state_dict() if is_ddp else model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "miou": miou,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, save_path)


def main():
    args = parse_args()

    # Setup distributed training
    is_ddp, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Training RangeNet++ on SemanticKITTI")
        print(f"  Device: {device}")
        print(f"  Distributed: {is_ddp} (world_size={world_size})")
        print(f"  Range image: {args.height}x{args.width}")
        print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
        print(f"  LR: {args.lr}, Schedule: {args.lr_schedule}")
        print(f"  AMP: {args.amp}")
        os.makedirs(args.save_dir, exist_ok=True)

    # Build model
    model_config = {
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        "height": args.height,
        "width": args.width,
        "dropout_p": args.dropout,
    }
    model = RangeNetPP(config=model_config).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    if rank == 0:
        raw_model = model.module if is_ddp else model
        params = raw_model.get_num_parameters()
        print(f"  Model parameters: {params['trainable']:,} trainable / {params['total']:,} total")

    # Datasets and dataloaders
    train_dataset = SemanticKITTIRangeDataset(
        root=args.data_root,
        split="train",
        height=args.height,
        width=args.width,
        augment=True,
    )
    val_dataset = SemanticKITTIRangeDataset(
        root=args.data_root,
        split="val",
        height=args.height,
        width=args.width,
        augment=False,
    )

    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Loss function
    class_weights = get_default_semantickitti_weights(num_classes=args.num_classes).to(device)
    criterion = CombinedLoss(
        class_weights=class_weights,
        ignore_index=0,
        alpha=args.loss_alpha,
        beta=args.loss_beta,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # LR scheduler
    steps_per_epoch = len(train_loader)
    scheduler = get_lr_scheduler(optimizer, args, steps_per_epoch)

    # AMP scaler
    scaler = GradScaler() if args.amp else None

    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        raw_model = model.module if is_ddp else model
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("miou", 0.0)
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if rank == 0:
            print(f"  Resumed from epoch {start_epoch}, best mIoU: {best_miou*100:.1f}%")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            epoch_start = time.time()
            print(f"\nEpoch {epoch}/{args.epochs-1}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            use_amp=args.amp,
            rank=rank,
        )

        if rank == 0:
            elapsed = time.time() - epoch_start
            print(
                f"  Train: Loss={train_metrics['loss']:.4f}, "
                f"mIoU={train_metrics['miou']*100:.1f}%, "
                f"Time={elapsed:.1f}s"
            )

        # Validation
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_metrics = validate(model, val_loader, criterion, device, args.amp)

            if rank == 0:
                print(
                    f"  Val: Loss={val_metrics['loss']:.4f}, "
                    f"mIoU={val_metrics['miou']*100:.1f}%"
                )
                print_iou_table(val_metrics["per_class_iou"], val_metrics["miou"])

                # Save best model
                if val_metrics["miou"] > best_miou:
                    best_miou = val_metrics["miou"]
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        epoch, best_miou,
                        os.path.join(args.save_dir, "best_model.pth"),
                        is_ddp=is_ddp,
                    )
                    print(f"  New best mIoU: {best_miou*100:.1f}%")

        # Periodic checkpoint
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, best_miou,
                os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pth"),
                is_ddp=is_ddp,
            )

    # Cleanup
    if is_ddp:
        dist.destroy_process_group()

    if rank == 0:
        print(f"\nTraining complete. Best mIoU: {best_miou*100:.1f}%")
        print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
