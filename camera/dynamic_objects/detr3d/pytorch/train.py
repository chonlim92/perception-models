"""
Training script for DETR3D 3D object detection.

Supports single-GPU and multi-GPU distributed training with mixed precision.
"""

import argparse
import os
import random
import time
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import yaml

from model import DETR3D, DETR3DLoss, DETR3DPostProcessor
from dataset import NuScenesDataset, collate_fn, PC_RANGE


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DETR3D 3D object detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detr3d_r101_nuscenes.yaml",
        help="Path to config yaml file",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Root path to dataset")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader workers")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./work_dirs/detr3d", help="Output directory"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Local rank for distributed training"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_args_into_config(args, config):
    """Override config values with CLI arguments where provided."""
    if args.data_root is not None:
        config.setdefault("data", {})["data_root"] = args.data_root
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    return config


def set_seed(seed, deterministic=False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def setup_distributed():
    """Initialize distributed training if applicable."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = rank % torch.cuda.device_count()
    else:
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()
    return True, rank, world_size, local_rank


def setup_logging(output_dir, rank=0):
    """Configure logging (only rank 0 logs to console and file)."""
    log_format = "[%(asctime)s %(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=datefmt,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(output_dir, "train.log")),
            ],
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def build_model(config):
    """Build DETR3D model from config."""
    model_cfg = config["model"]
    detection_range = model_cfg["detection_range"]
    pc_range = detection_range[0] + detection_range[1]  # flatten to [x_min, y_min, z_min, x_max, y_max, z_max]

    model = DETR3D(
        num_classes=model_cfg["num_classes"],
        embed_dims=model_cfg["query_dim"],
        num_queries=model_cfg["num_queries"],
        num_layers=model_cfg["num_decoder_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dims=model_cfg["ffn_dim"],
        dropout=model_cfg["dropout"],
        pc_range=pc_range,
        pretrained_backbone=True,
        frozen_backbone_stages=1,
    )
    return model


def build_criterion(config):
    """Build DETR3D loss criterion from config."""
    model_cfg = config["model"]

    # Loss config - use defaults if not specified
    loss_cfg = config.get("loss", {})
    criterion = DETR3DLoss(
        num_classes=model_cfg["num_classes"],
        cls_weight=loss_cfg.get("cls_weight", 2.0),
        reg_weight=loss_cfg.get("bbox_weight", 0.25),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
    )
    return criterion


def build_optimizer(model, config):
    """Build AdamW optimizer with parameter groups.

    Backbone parameters get lr * backbone_lr_factor.
    Weight decay is not applied to bias and normalization parameters.
    """
    training_cfg = config["training"]
    base_lr = training_cfg.get("learning_rate", training_cfg.get("lr", 2e-4))
    backbone_lr_factor = training_cfg.get("backbone_lr_factor", 0.1)
    weight_decay = training_cfg.get("weight_decay", 0.01)

    # Separate parameters into groups
    backbone_params_decay = []
    backbone_params_no_decay = []
    other_params_decay = []
    other_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_backbone = "backbone" in name
        # No weight decay for bias and norm parameters
        is_no_decay = "bias" in name or "norm" in name or "bn" in name

        if is_backbone and is_no_decay:
            backbone_params_no_decay.append(param)
        elif is_backbone:
            backbone_params_decay.append(param)
        elif is_no_decay:
            other_params_no_decay.append(param)
        else:
            other_params_decay.append(param)

    param_groups = [
        {
            "params": backbone_params_decay,
            "lr": base_lr * backbone_lr_factor,
            "weight_decay": weight_decay,
            "name": "backbone_decay",
        },
        {
            "params": backbone_params_no_decay,
            "lr": base_lr * backbone_lr_factor,
            "weight_decay": 0.0,
            "name": "backbone_no_decay",
        },
        {
            "params": other_params_decay,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "other_decay",
        },
        {
            "params": other_params_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "other_no_decay",
        },
    ]

    # Filter out empty parameter groups
    param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    """Build cosine annealing scheduler with linear warmup."""
    training_cfg = config["training"]
    epochs = training_cfg["epochs"]
    warmup_epochs = training_cfg.get("warmup_epochs", 1)

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def save_checkpoint(state, output_dir, filename="checkpoint.pth"):
    """Save training checkpoint."""
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint and restore training state."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle DDP state dict
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present (from DDP saving)
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    start_epoch = checkpoint.get("epoch", 0) + 1

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info(f"Resumed from epoch {start_epoch}")
    return start_epoch


def reduce_loss(loss_value, world_size):
    """Reduce loss across all processes for logging."""
    if world_size <= 1:
        return loss_value
    with torch.no_grad():
        reduced = loss_value.clone()
        dist.all_reduce(reduced)
        reduced /= world_size
    return reduced


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    scheduler,
    scaler,
    epoch,
    config,
    distributed,
    world_size,
    log_interval=50,
):
    """Train for one epoch."""
    model.train()
    criterion.train()

    training_cfg = config["training"]
    grad_clip_norm = training_cfg.get("grad_clip_norm", 35.0)
    use_aux_loss = training_cfg.get("aux_loss", True)

    total_loss_accum = 0.0
    cls_loss_accum = 0.0
    bbox_loss_accum = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(data_loader):
        # Move data to GPU
        images = batch["images"].cuda(non_blocking=True)
        intrinsics = batch["intrinsics"].cuda(non_blocking=True)
        extrinsics = batch["extrinsics"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)
        boxes_3d = batch["boxes_3d"].cuda(non_blocking=True)
        num_objects = batch["num_objects"]

        # Derive image shape from the images tensor
        image_shape = (images.shape[-2], images.shape[-1])

        # Build per-sample target list for the loss function
        targets = []
        for b in range(images.shape[0]):
            n = int(num_objects[b].item())
            targets.append({
                "labels": labels[b, :n],
                "boxes": boxes_3d[b, :n],
            })

        # Forward pass with mixed precision
        optimizer.zero_grad()

        with autocast():
            outputs = model(images, intrinsics, extrinsics, image_shape)

            # Compute loss (includes auxiliary losses internally)
            loss_dict = criterion(outputs, targets)
            total_loss = loss_dict["total_loss"]

        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()

        # Gradient clipping (unscale first for proper norm computation)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step (per-iteration)
        scheduler.step()

        # Accumulate losses for logging
        total_loss_accum += total_loss.item()
        cls_loss_accum += loss_dict["loss_cls"].item()
        bbox_loss_accum += loss_dict["loss_reg"].item()
        num_batches += 1

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_total_loss = total_loss_accum / num_batches
            avg_cls_loss = cls_loss_accum / num_batches
            avg_bbox_loss = bbox_loss_accum / num_batches
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            eta_seconds = elapsed / (batch_idx + 1) * (len(data_loader) - batch_idx - 1)

            if not distributed or dist.get_rank() == 0:
                logger.info(
                    f"Epoch [{epoch}][{batch_idx + 1}/{len(data_loader)}] "
                    f"total_loss: {avg_total_loss:.4f}, "
                    f"cls_loss: {avg_cls_loss:.4f}, "
                    f"bbox_loss: {avg_bbox_loss:.4f}, "
                    f"lr: {current_lr:.6f}, "
                    f"ETA: {int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
                )

            # Reset accumulators
            total_loss_accum = 0.0
            cls_loss_accum = 0.0
            bbox_loss_accum = 0.0
            num_batches = 0

    return total_loss.item()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    config = merge_args_into_config(args, config)

    # Setup distributed training
    distributed, rank, world_size, local_rank = setup_distributed()

    # Create output directory
    output_dir = args.output_dir
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Wait for rank 0 to create output dir
    if distributed:
        dist.barrier()

    # Setup logging
    setup_logging(output_dir, rank)
    logger.info(f"Config: {config}")
    logger.info(f"Distributed: {distributed}, World size: {world_size}, Rank: {rank}")

    # Set seed
    set_seed(args.seed + rank)

    # Training config
    training_cfg = config["training"]
    batch_size = training_cfg["batch_size"]
    epochs = training_cfg["epochs"]
    num_workers = training_cfg.get("num_workers", 4)
    checkpoint_interval = training_cfg.get("checkpoint_interval", 4)

    # Build dataset and dataloader
    data_cfg = config.get("data", {})
    data_root = data_cfg.get("data_root", data_cfg.get("root", "./data/nuscenes"))
    train_dataset = NuScenesDataset(
        data_root=data_root,
        split="train",
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Build model
    model = build_model(config)
    model.cuda()
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # Build loss criterion
    criterion = build_criterion(config)
    criterion.cuda()

    # Build optimizer
    optimizer = build_optimizer(model, config)

    # Build scheduler
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    # Wrap model with DDP
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    logger.info(f"Starting training from epoch {start_epoch} to {epochs}")
    logger.info(
        f"Batch size per GPU: {batch_size}, "
        f"Effective batch size: {batch_size * world_size}, "
        f"Steps per epoch: {steps_per_epoch}"
    )

    for epoch in range(start_epoch, epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        # Train one epoch
        train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            config=config,
            distributed=distributed,
            world_size=world_size,
        )

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch} completed in {epoch_time / 60:.1f} minutes"
        )

        # Save checkpoint
        if rank == 0 and (epoch + 1) % checkpoint_interval == 0:
            model_state = (
                model.module.state_dict() if distributed else model.state_dict()
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "scaler": scaler.state_dict(),
                },
                output_dir,
                filename=f"checkpoint_epoch_{epoch + 1}.pth",
            )

        # Save latest checkpoint every epoch (rank 0 only)
        if rank == 0:
            model_state = (
                model.module.state_dict() if distributed else model.state_dict()
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "scaler": scaler.state_dict(),
                },
                output_dir,
                filename="checkpoint_latest.pth",
            )

    # Final save
    if rank == 0:
        model_state = (
            model.module.state_dict() if distributed else model.state_dict()
        )
        save_checkpoint(
            {
                "epoch": epochs - 1,
                "model": model_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
                "scaler": scaler.state_dict(),
            },
            output_dir,
            filename="checkpoint_final.pth",
        )
        logger.info("Training completed successfully!")

    # Cleanup distributed
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
