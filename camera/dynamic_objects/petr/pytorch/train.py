"""
Training script for PETR/PETRv2/StreamPETR.

Supports distributed training (DDP), mixed precision (AMP),
gradient accumulation, cosine learning rate scheduling with warmup,
checkpoint saving/loading, and logging.
"""

import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import yaml

from .dataset import NuScenesDataset, collate_fn
from .model import PETRConfig, PETRModel

logger = logging.getLogger(__name__)


def setup_logging(rank: int, output_dir: str) -> None:
    """Configure logging for distributed training."""
    log_format = "[%(asctime)s][Rank %(name)s] %(levelname)s: %(message)s"
    level = logging.INFO if rank == 0 else logging.WARNING

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(output_dir, f"train_rank{rank}.log"), mode="a"
            ),
        ],
    )
    logger.name = str(rank)


def setup_distributed() -> tuple:
    """Initialize distributed training environment.

    Returns:
        Tuple of (rank, world_size, device).
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=30),
        )
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_optimizer(
    model: nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """Build optimizer with layer-wise learning rate decay.

    Args:
        model: The model to optimize.
        config: Training config with optimizer settings.

    Returns:
        Configured optimizer.
    """
    lr = config.get("lr", 2e-4)
    weight_decay = config.get("weight_decay", 0.01)
    backbone_lr_mult = config.get("backbone_lr_mult", 0.1)

    # Separate backbone and non-backbone parameters
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {
            "params": backbone_params,
            "lr": lr * backbone_lr_mult,
            "weight_decay": weight_decay,
        },
    ]

    optimizer = AdamW(param_groups)
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Dict[str, Any], steps_per_epoch: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build learning rate scheduler with warmup.

    Args:
        optimizer: The optimizer.
        config: Training config with scheduler settings.
        steps_per_epoch: Number of optimization steps per epoch.

    Returns:
        Learning rate scheduler.
    """
    num_epochs = config.get("num_epochs", 24)
    warmup_epochs = config.get("warmup_epochs", 1)
    min_lr_ratio = config.get("min_lr_ratio", 1e-3)

    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    # Warmup: linear increase from 0 to base lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Main schedule: cosine annealing
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
    )

    # Combine warmup + cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    return scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    output_dir: str,
    is_best: bool = False,
) -> None:
    """Save training checkpoint.

    Args:
        model: The model (unwrapped from DDP if needed).
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        scaler: AMP grad scaler state.
        epoch: Current epoch number.
        global_step: Global training step.
        output_dir: Directory to save checkpoint.
        is_best: If True, also save as 'best.pth'.
    """
    # Unwrap DDP model
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")

    # Save latest symlink
    latest_path = os.path.join(output_dir, "latest.pth")
    torch.save(checkpoint, latest_path)

    if is_best:
        best_path = os.path.join(output_dir, "best.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to restore state (optional).
        scheduler: Scheduler to restore state (optional).
        scaler: AMP scaler to restore state (optional).

    Returns:
        Checkpoint metadata (epoch, global_step).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model state
    model_state = checkpoint["model"]
    if hasattr(model, "module"):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    # Load optimizer state
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    # Load scaler state
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', 'unknown')})"
    )

    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    config: Dict[str, Any],
    device: torch.device,
    rank: int,
) -> int:
    """Train for one epoch.

    Args:
        model: The model (possibly DDP-wrapped).
        dataloader: Training data loader.
        optimizer: Optimizer.
        scheduler: LR scheduler (stepped per iteration).
        scaler: AMP gradient scaler.
        epoch: Current epoch number.
        global_step: Current global step count.
        config: Training configuration.
        device: Training device.
        rank: Process rank.

    Returns:
        Updated global step count.
    """
    model.train()
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    log_interval = config.get("log_interval", 50)
    max_grad_norm = config.get("max_grad_norm", 35.0)
    use_amp = config.get("use_amp", True)
    is_stream = config.get("model", {}).get("variant", "petr") == "streampetr"

    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_bbox_loss = 0.0
    num_batches = 0
    start_time = time.time()

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        images = batch["images"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        gt_labels = [l.to(device) for l in batch["gt_labels"]]
        gt_bboxes = [b.to(device) for b in batch["gt_bboxes"]]

        # Optional temporal data
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

        # Forward pass with AMP
        with autocast(enabled=use_amp):
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
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
            )

            losses = outputs["losses"]
            loss = losses["loss_total"] / grad_accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # Logging
        epoch_loss += losses["loss_total"].item()
        epoch_cls_loss += losses["loss_cls"].item()
        epoch_bbox_loss += losses["loss_bbox"].item()
        num_batches += 1

        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * images.shape[0] / elapsed
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch [{epoch}][{batch_idx + 1}/{len(dataloader)}] "
                f"loss: {losses['loss_total'].item():.4f} "
                f"cls: {losses['loss_cls'].item():.4f} "
                f"bbox: {losses['loss_bbox'].item():.4f} "
                f"vel: {losses['loss_velocity'].item():.4f} "
                f"lr: {current_lr:.2e} "
                f"throughput: {samples_per_sec:.1f} samples/s"
            )

    # Epoch summary
    if rank == 0:
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_cls = epoch_cls_loss / max(num_batches, 1)
        avg_bbox = epoch_bbox_loss / max(num_batches, 1)
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch} summary: "
            f"avg_loss={avg_loss:.4f} "
            f"avg_cls={avg_cls:.4f} "
            f"avg_bbox={avg_bbox:.4f} "
            f"time={elapsed:.1f}s"
        )

    return global_step


def train(config_path: str, resume: Optional[str] = None) -> None:
    """Main training function.

    Args:
        config_path: Path to YAML configuration file.
        resume: Optional path to checkpoint to resume from.
    """
    # Load config
    config = load_config(config_path)
    model_config = config.get("model", {})
    train_config = config.get("training", {})
    data_config = config.get("data", {})

    # Setup distributed
    rank, world_size, device = setup_distributed()

    # Create output directory
    output_dir = train_config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(rank, output_dir)

    if rank == 0:
        logger.info(f"Training config: {config}")
        logger.info(f"World size: {world_size}, Device: {device}")

    # Build model
    petr_config = PETRConfig(**model_config)
    model = PETRModel(petr_config).to(device)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model: {petr_config.variant}, "
            f"params: {num_params / 1e6:.1f}M, "
            f"trainable: {num_trainable / 1e6:.1f}M"
        )

    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            find_unused_parameters=True,
        )

    # Build dataset and dataloader
    dataset = NuScenesDataset(
        data_root=data_config.get("data_root", "/data/nuscenes"),
        ann_file=data_config.get("ann_file"),
        split="train",
        num_cameras=model_config.get("num_cameras", 6),
        img_size=tuple(model_config.get("img_size", [900, 1600])),
        num_temporal_frames=model_config.get("num_temporal_frames", 0),
        pc_range=tuple(model_config.get("pc_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])),
        augmentation=train_config.get("augmentation", True),
    )

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.get("batch_size", 1),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, train_config)
    steps_per_epoch = len(dataloader) // train_config.get("gradient_accumulation_steps", 1)
    scheduler = build_scheduler(optimizer, train_config, steps_per_epoch)

    # AMP scaler
    scaler = GradScaler(enabled=train_config.get("use_amp", True))

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if resume:
        meta = load_checkpoint(resume, model, optimizer, scheduler, scaler)
        start_epoch = meta["epoch"] + 1
        global_step = meta["global_step"]
        logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")

    # Training loop
    num_epochs = train_config.get("num_epochs", 24)
    save_interval = train_config.get("save_interval", 1)

    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        # Reset temporal memory at start of each epoch (for StreamPETR)
        if hasattr(model, "module"):
            model.module.reset_temporal_state()
        else:
            model.reset_temporal_state()

        global_step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            config=train_config,
            device=device,
            rank=rank,
        )

        # Save checkpoint
        if rank == 0 and (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                output_dir=output_dir,
            )

    cleanup_distributed()
    if rank == 0:
        logger.info("Training complete.")


def main() -> None:
    """Entry point for training script."""
    parser = argparse.ArgumentParser(description="Train PETR/PETRv2/StreamPETR")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint for resuming"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Local rank for distributed training"
    )
    args = parser.parse_args()

    train(args.config, args.resume)


if __name__ == "__main__":
    main()
