"""
Cylinder3D Training Script
Complete training pipeline with mixed precision, distributed training,
combined CE + Lovasz loss, cosine annealing with warmup, and mIoU evaluation.
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import yaml

from .model import Cylinder3D
from .dataset import SemanticKITTIDataset, collate_fn
from .losses import CombinedLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Cylinder3D Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device IDs (comma-separated, e.g. '0,1,2,3')",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory for training logs and checkpoints",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable DistributedDataParallel training",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and return YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(log_dir: str, rank: int = 0) -> logging.Logger:
    """Set up logger that writes to file and stdout (only on rank 0)."""
    logger = logging.getLogger("Cylinder3D")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "[%(asctime)s][Rank %(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "training.log"), mode="a"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.name = str(rank)
    return logger


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed(local_rank: int):
    """Initialize distributed process group."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    """Destroy distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
):
    """Save training checkpoint and optionally copy as best model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
):
    """Load checkpoint and restore model, optimizer, scheduler, and scaler states.

    Returns:
        dict: Checkpoint dictionary with metadata (epoch, best_miou, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle DDP state dict (remove 'module.' prefix if present)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def compute_miou(
    confusion_matrix: np.ndarray,
    ignore_index: int = 0,
) -> tuple:
    """Compute mean Intersection over Union from confusion matrix.

    Args:
        confusion_matrix: NxN confusion matrix (rows=true, cols=predicted).
        ignore_index: Class index to ignore (typically 0 for 'unlabeled').

    Returns:
        Tuple of (miou, per_class_iou) where per_class_iou is a numpy array.
    """
    # Compute IoU per class
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    # Avoid division by zero
    iou = np.zeros_like(intersection, dtype=np.float64)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]

    # Exclude ignore_index from mIoU computation
    mask = np.ones(len(iou), dtype=bool)
    if ignore_index is not None and 0 <= ignore_index < len(iou):
        mask[ignore_index] = False

    valid_classes = mask & valid
    if valid_classes.sum() == 0:
        return 0.0, iou

    miou = iou[valid_classes].mean()
    return float(miou), iou


class WarmupCosineScheduler:
    """Cosine annealing LR scheduler with linear warmup."""

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        self.current_epoch = 0

    def step(self):
        """Advance scheduler by one epoch."""
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.current_epoch / self.warmup_epochs
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs
            ):
                param_group["lr"] = base_lr * warmup_factor
        else:
            self.cosine_scheduler.step()

    def state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "cosine_scheduler": self.cosine_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict["current_epoch"]
        self.cosine_scheduler.load_state_dict(state_dict["cosine_scheduler"])

    def get_last_lr(self):
        """Return the current learning rate."""
        return [pg["lr"] for pg in self.optimizer.param_groups]


def build_dataloader(
    config: dict,
    split: str,
    distributed: bool = False,
):
    """Build dataset and dataloader from config.

    Args:
        config: Full config dict.
        split: One of 'train', 'val', 'test'.
        distributed: Whether to use DistributedSampler.

    Returns:
        Tuple of (dataloader, sampler or None).
    """
    dataset_config = config.get("dataset", {})
    data_root = dataset_config.get("data_root", "./data/SemanticKITTI")
    num_classes = dataset_config.get("num_classes", 20)
    grid_size = dataset_config.get("grid_size", [480, 360, 32])

    dataset = SemanticKITTIDataset(
        data_root=data_root,
        split=split,
        grid_size=grid_size,
        num_classes=num_classes,
    )

    train_config = config.get("train", {})
    batch_size = train_config.get("batch_size", 2)
    num_workers = train_config.get("num_workers", 4)

    sampler = None
    shuffle = split == "train"

    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=(split == "train"),
        )
        shuffle = False  # Sampler handles shuffling

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        sampler=sampler,
        collate_fn=collate_fn,
    )

    return dataloader, sampler


def build_model(config: dict, device: torch.device) -> nn.Module:
    """Build Cylinder3D model from config."""
    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})

    num_classes = dataset_config.get("num_classes", 20)
    grid_size = dataset_config.get("grid_size", [480, 360, 32])
    input_dims = model_config.get("input_dims", 9)
    init_channels = model_config.get("init_channels", 32)

    model = Cylinder3D(
        num_classes=num_classes,
        grid_size=grid_size,
        input_dims=input_dims,
        init_channels=init_channels,
    )
    model = model.to(device)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    log_interval: int = 10,
    max_grad_norm: float = 10.0,
):
    """Train for one epoch with mixed precision and gradient clipping.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        voxel_features = batch["voxel_features"].to(device, non_blocking=True)
        voxel_coords = batch["voxel_coords"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        point_to_voxel = batch["point_to_voxel"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision
        with autocast():
            predictions = model(voxel_features, voxel_coords, point_to_voxel)
            loss = criterion(predictions, targets)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first for correct norm computation)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss.item()
        total_loss += batch_loss

        # Logging
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch}] Iter [{batch_idx + 1}/{num_batches}] "
                f"Loss: {batch_loss:.4f} LR: {current_lr:.6f}"
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 0,
):
    """Validate model and compute mIoU.

    Returns:
        Tuple of (avg_loss, miou, per_class_iou).
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in dataloader:
        voxel_features = batch["voxel_features"].to(device, non_blocking=True)
        voxel_coords = batch["voxel_coords"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        point_to_voxel = batch["point_to_voxel"].to(device, non_blocking=True)

        with autocast():
            predictions = model(voxel_features, voxel_coords, point_to_voxel)
            loss = criterion(predictions, targets)

        total_loss += loss.item()

        # Compute confusion matrix
        preds = predictions.argmax(dim=1).cpu().numpy()
        labels = targets.cpu().numpy()

        # Only count valid labels
        valid_mask = (labels >= 0) & (labels < num_classes)
        preds_valid = preds[valid_mask]
        labels_valid = labels[valid_mask]

        np.add.at(
            confusion_matrix,
            (labels_valid, preds_valid),
            1,
        )

    avg_loss = total_loss / max(num_batches, 1)
    miou, per_class_iou = compute_miou(confusion_matrix, ignore_index=ignore_index)

    return avg_loss, miou, per_class_iou


def main():
    args = parse_args()

    # Determine rank and local_rank for distributed training
    local_rank = 0
    rank = 0
    world_size = 1

    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        setup_distributed(local_rank)

    # Set seed
    set_seed(args.seed + rank)

    # Load config
    config = load_config(args.config)

    # Setup logger
    logger = setup_logger(args.log_dir, rank=rank)
    logger.info(f"Starting Cylinder3D training with config: {args.config}")
    logger.info(f"Distributed: {args.distributed}, World size: {world_size}")

    # Device
    if args.distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        gpu_ids = [int(x) for x in args.gpu.split(",")]
        device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_ids[0])

    logger.info(f"Using device: {device}")

    # Build model
    model = build_model(config, device)
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # Wrap with DDP if distributed
    if args.distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # Build datasets
    train_loader, train_sampler = build_dataloader(
        config, split="train", distributed=args.distributed
    )
    val_loader, _ = build_dataloader(
        config, split="val", distributed=args.distributed
    )
    logger.info(
        f"Train samples: {len(train_loader.dataset)}, "
        f"Val samples: {len(val_loader.dataset)}"
    )

    # Loss function
    dataset_config = config.get("dataset", {})
    num_classes = dataset_config.get("num_classes", 20)
    ignore_index = dataset_config.get("ignore_index", 0)

    loss_config = config.get("loss", {})
    ce_weight = loss_config.get("ce_weight", 1.0)
    lovasz_weight = loss_config.get("lovasz_weight", 1.0)

    criterion = CombinedLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        ce_weight=ce_weight,
        lovasz_weight=lovasz_weight,
    ).to(device)

    # Optimizer
    train_config = config.get("train", {})
    lr = train_config.get("lr", 1e-3)
    weight_decay = train_config.get("weight_decay", 1e-4)

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler: cosine annealing with linear warmup
    total_epochs = train_config.get("epochs", 40)
    warmup_epochs = train_config.get("warmup_epochs", 5)
    min_lr = train_config.get("min_lr", 1e-6)

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        min_lr=min_lr,
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0.0

    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch = checkpoint.get("epoch", 0)
        best_miou = checkpoint.get("best_miou", 0.0)
        logger.info(
            f"Resumed from epoch {start_epoch} with best mIoU: {best_miou:.4f}"
        )

    # Training config
    log_interval = train_config.get("log_interval", 10)
    save_interval = train_config.get("save_interval", 5)
    max_grad_norm = train_config.get("max_grad_norm", 10.0)
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")

    logger.info(
        f"Training for {total_epochs} epochs "
        f"(warmup: {warmup_epochs}, starting at epoch {start_epoch})"
    )

    # Training loop
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train one epoch
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch + 1,
            logger=logger,
            log_interval=log_interval,
            max_grad_norm=max_grad_norm,
        )

        # Step scheduler after epoch
        scheduler.step()

        # Validation
        val_loss, val_miou, per_class_iou = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{epoch + 1}/{total_epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_miou:.4f} | LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Log per-class IoU
        iou_str = " | ".join(
            [f"C{i}: {iou:.3f}" for i, iou in enumerate(per_class_iou) if i != ignore_index]
        )
        logger.info(f"Per-class IoU: {iou_str}")

        # Save checkpoints (only rank 0)
        if rank == 0:
            is_best = val_miou > best_miou
            if is_best:
                best_miou = val_miou
                logger.info(f"New best mIoU: {best_miou:.4f}")

            # Prepare checkpoint state
            model_state = (
                model.module.state_dict()
                if args.distributed
                else model.state_dict()
            )
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_miou": best_miou,
                "config": config,
            }

            # Save best checkpoint
            save_checkpoint(
                checkpoint_state,
                is_best=is_best,
                checkpoint_dir=checkpoint_dir,
            )

            # Periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                save_checkpoint(
                    checkpoint_state,
                    is_best=False,
                    checkpoint_dir=checkpoint_dir,
                    filename=f"checkpoint_epoch_{epoch + 1}.pth",
                )

    # Final summary
    logger.info(f"Training complete. Best mIoU: {best_miou:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")

    # Cleanup
    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
