"""Distributed training script for BEVFormer.

Supports multi-GPU training via PyTorch DDP with mixed precision, gradient
clipping, cosine LR scheduling with warmup, temporal BEV caching, and
TensorBoard/WandB logging.

Usage:
    torchrun --nproc_per_node=8 train.py --config ../configs/bevformer_base.yaml
"""

import argparse
import datetime
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import yaml

__all__ = ["train", "main"]

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# =============================================================================
# Learning Rate Scheduler
# =============================================================================


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay learning rate scheduler.

    Operates on a per-iteration basis for precise warmup control.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        total_iters: int,
        warmup_ratio: float = 0.33,
        min_lr_ratio: float = 1e-3,
    ) -> None:
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule.
            warmup_iters: Number of warmup iterations.
            total_iters: Total training iterations.
            warmup_ratio: LR multiplier at start of warmup.
            min_lr_ratio: Minimum LR ratio at end of cosine decay.
        """
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step_count = 0

    def step(self) -> None:
        """Update learning rate for the current iteration."""
        self._step_count += 1
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = self._get_lr(base_lr)

    def _get_lr(self, base_lr: float) -> float:
        """Compute learning rate for current step.

        Args:
            base_lr: Base learning rate for this param group.

        Returns:
            Current learning rate.
        """
        if self._step_count <= self.warmup_iters:
            # Linear warmup
            alpha = self._step_count / max(self.warmup_iters, 1)
            return base_lr * (self.warmup_ratio + (1.0 - self.warmup_ratio) * alpha)
        else:
            # Cosine decay
            progress = (self._step_count - self.warmup_iters) / max(
                self.total_iters - self.warmup_iters, 1
            )
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay)

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state."""
        return {"step_count": self._step_count}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self._step_count = state_dict["step_count"]


# =============================================================================
# Training Utilities
# =============================================================================


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training.

    Returns:
        Tuple of (rank, local_rank, world_size).
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=30),
        )
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Base random seed.
        rank: Process rank (adds offset for data diversity).
    """
    import random
    import numpy as np

    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Instantiate BEVFormer model from configuration.

    Args:
        config: Full configuration dict.
        device: Target device.

    Returns:
        BEVFormer model on device.
    """
    from .model import BEVFormer

    model_cfg = config["model"]
    bev_cfg = model_cfg["bev_encoder"]
    decoder_cfg = model_cfg["decoder"]
    head_cfg = model_cfg["head"]
    loss_cfg = config["loss"]

    model = BEVFormer(
        backbone_out_channels=model_cfg["neck"]["out_channels"],
        backbone_pretrained=model_cfg["backbone"]["pretrained"],
        backbone_frozen_stages=model_cfg["backbone"]["frozen_stages"],
        embed_dim=bev_cfg["embed_dims"],
        bev_h=bev_cfg["bev_h"],
        bev_w=bev_cfg["bev_w"],
        num_encoder_layers=bev_cfg["num_encoder_layers"],
        num_heads=bev_cfg["num_heads"],
        num_points_spatial=bev_cfg["num_points_spatial"],
        num_points_temporal=bev_cfg["num_points_temporal"],
        num_levels=bev_cfg["num_levels"],
        num_cams=bev_cfg["num_cams"],
        num_ref_points=bev_cfg["num_ref_points"],
        pc_range=tuple(bev_cfg["pc_range"]),
        num_decoder_layers=decoder_cfg["num_decoder_layers"],
        num_queries=decoder_cfg["num_queries"],
        ffn_dim=decoder_cfg["ffn_dim"],
        dropout=decoder_cfg["dropout"],
        iterative_bbox_refinement=decoder_cfg["iterative_bbox_refinement"],
        num_classes=head_cfg["num_classes"],
        code_size=head_cfg["code_size"],
        num_reg_fcs=head_cfg["num_reg_fcs"],
        cls_weight=loss_cfg["cls_loss"]["weight"],
        bbox_weight=loss_cfg["bbox_loss"]["weight"],
        focal_alpha=loss_cfg["cls_loss"]["alpha"],
        focal_gamma=loss_cfg["cls_loss"]["gamma"],
        cls_cost=loss_cfg["matcher"]["cls_cost"],
        bbox_cost=loss_cfg["matcher"]["bbox_cost"],
    )

    model = model.to(device)
    return model


def build_optimizer(
    model: nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with parameter groups.

    Separates backbone parameters (lower LR) from the rest. Excludes bias
    and LayerNorm parameters from weight decay.

    Args:
        model: BEVFormer model.
        config: Training configuration.

    Returns:
        Configured optimizer.
    """
    train_cfg = config["training"]
    lr = train_cfg["lr"]
    backbone_lr_mult = train_cfg["backbone_lr_mult"]
    weight_decay = train_cfg["weight_decay"]

    # Separate parameters into groups
    backbone_params_decay = []
    backbone_params_no_decay = []
    other_params_decay = []
    other_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_backbone = name.startswith("backbone.") or name.startswith("module.backbone.")
        no_decay = (
            "bias" in name
            or "norm" in name
            or "bn" in name
            or "layer_norm" in name
            or "LayerNorm" in name
        )

        if is_backbone:
            if no_decay:
                backbone_params_no_decay.append(param)
            else:
                backbone_params_decay.append(param)
        else:
            if no_decay:
                other_params_no_decay.append(param)
            else:
                other_params_decay.append(param)

    param_groups = [
        {"params": other_params_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": other_params_no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": backbone_params_decay, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": backbone_params_no_decay, "lr": lr * backbone_lr_mult, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(train_cfg["betas"]),
    )

    return optimizer


def build_dataloaders(
    config: Dict[str, Any], rank: int, world_size: int
) -> Tuple[DataLoader, DataLoader, DistributedSampler]:
    """Build training and validation data loaders.

    Args:
        config: Full configuration dict.
        rank: Current process rank.
        world_size: Total number of processes.

    Returns:
        Tuple of (train_loader, val_loader, train_sampler).
    """
    from .dataset import NuScenesDataset, collate_fn

    data_cfg = config["data"]
    train_cfg = config["training"]

    train_dataset = NuScenesDataset(
        data_root=data_cfg["data_root"],
        ann_file=data_cfg["train_ann"],
        img_size=tuple(data_cfg["img_size"]),
        num_temporal_frames=data_cfg["num_temporal_frames"],
        classes=data_cfg["classes"],
        augmentation_cfg=data_cfg.get("augmentation"),
        is_train=True,
    )

    val_dataset = NuScenesDataset(
        data_root=data_cfg["data_root"],
        ann_file=data_cfg["val_ann"],
        img_size=tuple(data_cfg["img_size"]),
        num_temporal_frames=data_cfg["num_temporal_frames"],
        classes=data_cfg["classes"],
        augmentation_cfg=data_cfg.get("augmentation"),
        is_train=False,
    )

    # Distributed sampler
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=data_cfg.get("workers_per_gpu", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        sampler=val_sampler,
        shuffle=False,
        num_workers=data_cfg.get("workers_per_gpu", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    work_dir: str,
    filename: str = "latest.pth",
) -> None:
    """Save training checkpoint.

    Args:
        model: Model (possibly wrapped in DDP).
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        scaler: Grad scaler state.
        epoch: Current epoch.
        best_metric: Best validation metric so far.
        work_dir: Output directory.
        filename: Checkpoint filename.
    """
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, filename)
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[WarmupCosineScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[int, float]:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional grad scaler to restore.

    Returns:
        Tuple of (start_epoch, best_metric).
    """
    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Load model weights
    state_dict = checkpoint["model"]
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    # Load optimizer, scheduler, scaler
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0.0)

    logger.info(f"Resumed from epoch {epoch}, best_metric={best_metric:.4f}")
    return epoch, best_metric


# =============================================================================
# Training Loop
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: Dict[str, Any],
    device: torch.device,
    rank: int,
    writer: Optional[Any] = None,
) -> float:
    """Train for one epoch.

    Args:
        model: DDP-wrapped model.
        train_loader: Training data loader.
        optimizer: Optimizer.
        scheduler: LR scheduler.
        scaler: Gradient scaler for mixed precision.
        epoch: Current epoch number.
        config: Training configuration.
        device: Training device.
        rank: Process rank.
        writer: TensorBoard writer (rank 0 only).

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    train_cfg = config["training"]
    grad_clip = train_cfg["grad_clip"]
    use_amp = train_cfg.get("mixed_precision", True)

    total_loss = 0.0
    num_batches = 0
    prev_bev = None
    log_interval = 50

    start_time = time.time()
    data_time = 0.0
    iter_start = time.time()

    global_step = epoch * len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        data_time += time.time() - iter_start

        # Move data to device
        images = batch["images"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        extrinsics = batch["extrinsics"].to(device, non_blocking=True)
        ego_motion = batch["ego_motion"].to(device, non_blocking=True)
        gt_bboxes_3d = batch["gt_bboxes_3d"].to(device, non_blocking=True)
        gt_labels = batch["gt_labels"].to(device, non_blocking=True)
        prev_exists = batch["prev_exists"]

        # Reset prev_bev if new sequence starts
        if not all(prev_exists):
            prev_bev = None

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss_dict, new_bev = model.module.forward_train(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                ego_motion=ego_motion,
                prev_bev=prev_bev,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels=gt_labels,
            ) if hasattr(model, "module") else model.forward_train(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                ego_motion=ego_motion,
                prev_bev=prev_bev,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels=gt_labels,
            )

        loss = loss_dict["total_loss"]

        # Cache BEV features for next iteration (detached)
        prev_bev = new_bev.detach()

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # LR scheduler step (per-iteration)
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Logging
        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            throughput = (batch_idx + 1) * images.shape[0] / elapsed
            current_lr = optimizer.param_groups[0]["lr"]
            gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

            log_msg = (
                f"Epoch [{epoch}][{batch_idx + 1}/{len(train_loader)}] "
                f"loss: {loss.item():.4f} "
                f"lr: {current_lr:.2e} "
                f"mem: {gpu_mem:.1f}GB "
                f"throughput: {throughput:.1f} samples/s"
            )

            # Add component losses
            for key, val in loss_dict.items():
                if key != "total_loss":
                    log_msg += f" {key}: {val.item():.4f}"

            logger.info(log_msg)

            # TensorBoard logging
            if writer is not None:
                writer.add_scalar("train/total_loss", loss.item(), global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                writer.add_scalar("train/gpu_memory_gb", gpu_mem, global_step)
                writer.add_scalar("train/throughput", throughput, global_step)
                for key, val in loss_dict.items():
                    if key != "total_loss":
                        writer.add_scalar(f"train/{key}", val.item(), global_step)

        iter_start = time.time()

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    """Run validation and compute metrics.

    Args:
        model: Model (DDP or plain).
        val_loader: Validation data loader.
        device: Device.
        config: Configuration.
        rank: Process rank.
        world_size: World size.

    Returns:
        Dict of validation metrics.
    """
    model.eval()
    use_amp = config["training"].get("mixed_precision", True)

    all_predictions = []
    all_gt_bboxes = []
    all_gt_labels = []
    prev_bev = None

    for batch_idx, batch in enumerate(val_loader):
        images = batch["images"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        extrinsics = batch["extrinsics"].to(device, non_blocking=True)
        ego_motion = batch["ego_motion"].to(device, non_blocking=True)
        prev_exists = batch["prev_exists"]

        if not all(prev_exists):
            prev_bev = None

        with torch.amp.autocast("cuda", enabled=use_amp):
            forward_fn = model.module.forward_test if hasattr(model, "module") else model.forward_test
            detections, new_bev = forward_fn(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                ego_motion=ego_motion,
                prev_bev=prev_bev,
            )

        prev_bev = new_bev.detach()

        # Collect predictions
        all_predictions.append({
            "scores": detections["scores"].cpu(),
            "labels": detections["labels"].cpu(),
            "boxes": detections["boxes"].cpu(),
            "num_detections": detections["num_detections"].cpu(),
        })
        all_gt_bboxes.append(batch["gt_bboxes_3d"])
        all_gt_labels.append(batch["gt_labels"])

    # Simple metric: compute average number of detections and loss-like proxy
    total_dets = sum(
        p["num_detections"].sum().item() for p in all_predictions
    )
    total_samples = len(val_loader.dataset) if val_loader.dataset else 1

    metrics = {
        "avg_detections_per_sample": total_dets / max(total_samples, 1),
        "num_val_samples": float(total_samples),
    }

    if rank == 0:
        logger.info(
            f"Validation: avg_detections={metrics['avg_detections_per_sample']:.1f} "
            f"over {total_samples} samples"
        )

    return metrics


# =============================================================================
# Main Training Function
# =============================================================================


def train(args: argparse.Namespace) -> None:
    """Main training function.

    Args:
        args: Parsed command-line arguments.
    """
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Setup logging
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f"[%(asctime)s][Rank {rank}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = load_config(args.config)
    train_cfg = config["training"]

    # Override from args
    if args.seed is not None:
        train_cfg["seed"] = args.seed

    # Set seeds
    set_seed(train_cfg.get("seed", 0), rank)

    # Work directory
    work_dir = args.work_dir or os.path.join("work_dirs", Path(args.config).stem)
    if rank == 0:
        os.makedirs(work_dir, exist_ok=True)
        logger.info(f"Work directory: {work_dir}")
        logger.info(f"Config: {args.config}")
        logger.info(f"World size: {world_size}")

    # Build model
    model = build_model(config, device)

    # Convert BatchNorm to SyncBatchNorm for multi-GPU
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model params: {num_params:,} total, {num_trainable:,} trainable")

    # Build optimizer and scheduler
    optimizer = build_optimizer(
        model.module if hasattr(model, "module") else model, config
    )

    # Compute total iterations
    # We'll set total_iters after building dataloaders
    train_loader, val_loader, train_sampler = build_dataloaders(
        config, rank, world_size
    )
    total_iters = len(train_loader) * train_cfg["epochs"]

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_iters=train_cfg["lr_schedule"]["warmup_iters"],
        total_iters=total_iters,
        warmup_ratio=train_cfg["lr_schedule"]["warmup_ratio"],
        min_lr_ratio=train_cfg["lr_schedule"]["min_lr_ratio"],
    )

    # Grad scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.get("mixed_precision", True))

    # Resume from checkpoint
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        start_epoch, best_metric = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch += 1  # Start from next epoch

    # TensorBoard writer
    writer = None
    if rank == 0:
        if args.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir=os.path.join(work_dir, "tb_logs"))
                logger.info("TensorBoard logging enabled")
            except ImportError:
                logger.warning("tensorboard not installed, skipping TB logging")

        if args.wandb:
            try:
                import wandb
                wandb.init(
                    project="bevformer",
                    config=config,
                    dir=work_dir,
                    name=Path(args.config).stem,
                )
                logger.info("WandB logging enabled")
            except ImportError:
                logger.warning("wandb not installed, skipping WandB logging")

    # Training loop
    try:
        for epoch in range(start_epoch, train_cfg["epochs"]):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Train one epoch
            epoch_start = time.time()
            avg_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                config=config,
                device=device,
                rank=rank,
                writer=writer,
            )
            epoch_time = time.time() - epoch_start

            if rank == 0:
                logger.info(
                    f"Epoch {epoch} completed in {epoch_time:.1f}s, "
                    f"avg_loss: {avg_loss:.4f}"
                )

            # Validation
            if (epoch + 1) % train_cfg.get("eval_interval", 1) == 0:
                metrics = validate(
                    model, val_loader, device, config, rank, world_size
                )

                if rank == 0 and writer is not None:
                    for key, val in metrics.items():
                        writer.add_scalar(f"val/{key}", val, epoch)

                # Track best metric (use avg_detections as proxy, or NDS if available)
                current_metric = metrics.get("avg_detections_per_sample", 0.0)
                if current_metric > best_metric:
                    best_metric = current_metric
                    if rank == 0:
                        save_checkpoint(
                            model, optimizer, scheduler, scaler,
                            epoch, best_metric, work_dir, "best.pth"
                        )

            # Save checkpoint
            if rank == 0 and (epoch + 1) % train_cfg.get("checkpoint_interval", 1) == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, best_metric, work_dir, f"epoch_{epoch}.pth"
                )
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, best_metric, work_dir, "latest.pth"
                )

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if writer is not None:
            writer.close()
        cleanup_distributed()

    if rank == 0:
        logger.info(f"Training completed! Best metric: {best_metric:.4f}")


# =============================================================================
# Entry Point
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BEVFormer Training Script"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--work-dir", type=str, default=None,
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--tensorboard", action="store_true",
        help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
