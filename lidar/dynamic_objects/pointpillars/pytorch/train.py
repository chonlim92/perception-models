"""
PointPillars Training Script for 3D Object Detection.

Supports single-GPU and distributed (DDP) training with checkpoint management,
OneCycleLR scheduling, gradient clipping, and comprehensive logging.

Usage:
    Single GPU:
        python -m lidar.dynamic_objects.pointpillars.pytorch.train --config config.yaml

    Distributed (e.g., 4 GPUs):
        torchrun --nproc_per_node=4 -m lidar.dynamic_objects.pointpillars.pytorch.train \
            --config config.yaml --distributed

    Resume training:
        python -m lidar.dynamic_objects.pointpillars.pytorch.train \
            --config config.yaml --resume checkpoints/latest.pth
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import yaml

from .model import PointPillars
from .dataset import KITTIDataset, NuScenesDataset, collate_fn
from .losses import PointPillarsLoss


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train PointPillars 3D object detection model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training with DistributedDataParallel.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training (set automatically by torchrun).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply defaults for optional fields
    defaults = {
        "training": {
            "epochs": 80,
            "batch_size": 4,
            "num_workers": 4,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 10.0,
            "log_interval": 50,
            "save_interval": 5,
            "seed": 42,
            "pin_memory": True,
        },
        "scheduler": {
            "max_lr": 2e-4,
            "div_factor": 10.0,
            "pct_start": 0.4,
        },
        "checkpoint": {
            "save_dir": "checkpoints",
        },
    }

    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = {}
        for key, value in section_defaults.items():
            if key not in config[section]:
                config[section][key] = value

    return config


def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """
    Configure logging to both console and file.

    Args:
        log_dir: Directory where log files will be saved.
        rank: Process rank in distributed training (only rank 0 logs to file).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("pointpillars_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler for all ranks
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler only for rank 0
    if rank == 0:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed(local_rank: int) -> Tuple[int, int]:
    """
    Initialize the distributed process group for DDP training.

    Args:
        local_rank: Local rank of this process on the current node.

    Returns:
        Tuple of (rank, world_size) for the distributed group.
    """
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, world_size


def cleanup_distributed() -> None:
    """Destroy the distributed process group and release resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def build_dataset(config: Dict[str, Any], split: str) -> torch.utils.data.Dataset:
    """
    Instantiate the appropriate dataset based on configuration.

    Args:
        config: Full training configuration dictionary.
        split: Dataset split - 'train' or 'val'.

    Returns:
        Dataset instance for the specified split.

    Raises:
        ValueError: If the dataset type in config is not supported.
    """
    dataset_config = config["dataset"]
    dataset_type = dataset_config["type"].lower()
    data_root = dataset_config["data_root"]

    common_params = {
        "data_root": data_root,
        "split": split,
        "point_cloud_range": dataset_config.get(
            "point_cloud_range", [0, -39.68, -3, 69.12, 39.68, 1]
        ),
        "voxel_size": dataset_config.get("voxel_size", [0.16, 0.16, 4]),
        "max_points_per_voxel": dataset_config.get("max_points_per_voxel", 32),
        "max_voxels": dataset_config.get("max_voxels", 16000 if split == "train" else 40000),
    }

    if dataset_type == "kitti":
        return KITTIDataset(**common_params)
    elif dataset_type == "nuscenes":
        return NuScenesDataset(
            **common_params,
            version=dataset_config.get("version", "v1.0-trainval"),
            nsweeps=dataset_config.get("nsweeps", 10),
        )
    else:
        raise ValueError(
            f"Unsupported dataset type: '{dataset_type}'. "
            f"Supported types are: 'kitti', 'nuscenes'."
        )


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    config: Dict[str, Any],
    distributed: bool = False,
    is_training: bool = True,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Build a DataLoader with optional distributed sampling.

    Args:
        dataset: The dataset to load from.
        config: Training configuration dictionary.
        distributed: Whether to use distributed sampling.
        is_training: Whether this is for training (affects shuffling).

    Returns:
        Tuple of (DataLoader, sampler or None).
    """
    training_config = config["training"]
    batch_size = training_config["batch_size"]
    num_workers = training_config["num_workers"]
    pin_memory = training_config.get("pin_memory", True)

    sampler: Optional[DistributedSampler] = None

    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=is_training,
            drop_last=is_training,
        )
        shuffle = False
    else:
        shuffle = is_training

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=is_training,
    )

    return dataloader, sampler


def build_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Instantiate the PointPillars model from configuration.

    Args:
        config: Full training configuration dictionary.
        device: Device to place the model on.

    Returns:
        PointPillars model on the specified device.
    """
    model_config = config["model"]
    model = PointPillars(
        num_classes=model_config["num_classes"],
        point_cloud_range=config["dataset"].get(
            "point_cloud_range", [0, -39.68, -3, 69.12, 39.68, 1]
        ),
        voxel_size=config["dataset"].get("voxel_size", [0.16, 0.16, 4]),
        max_points_per_voxel=config["dataset"].get("max_points_per_voxel", 32),
        max_voxels=config["dataset"].get("max_voxels", 16000),
        num_point_features=model_config.get("num_point_features", 4),
        num_filters=model_config.get("num_filters", [64]),
        backbone_layer_nums=model_config.get("backbone_layer_nums", [3, 5, 5]),
        backbone_layer_strides=model_config.get("backbone_layer_strides", [2, 2, 2]),
        backbone_num_filters=model_config.get("backbone_num_filters", [64, 128, 256]),
        backbone_upsample_strides=model_config.get(
            "backbone_upsample_strides", [1, 2, 4]
        ),
        backbone_num_upsample_filters=model_config.get(
            "backbone_num_upsample_filters", [128, 128, 128]
        ),
        num_anchor_per_loc=model_config.get("num_anchor_per_loc", 2),
        anchor_sizes=model_config.get(
            "anchor_sizes", [[1.6, 3.9, 1.56]]
        ),
        anchor_rotations=model_config.get("anchor_rotations", [0, 1.5707963]),
        use_direction_classifier=model_config.get("use_direction_classifier", True),
        encode_background_as_zeros=model_config.get(
            "encode_background_as_zeros", True
        ),
        use_bev=model_config.get("use_bev", False),
    )
    model = model.to(device)
    return model


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> Adam:
    """
    Build Adam optimizer with configurable learning rate and weight decay.

    Args:
        model: The model whose parameters will be optimized.
        config: Training configuration dictionary.

    Returns:
        Configured Adam optimizer.
    """
    training_config = config["training"]
    lr = training_config.get("learning_rate", 2e-4)
    weight_decay = training_config.get("weight_decay", 0.01)

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer


def build_scheduler(
    optimizer: Adam,
    config: Dict[str, Any],
    steps_per_epoch: int,
    epochs: int,
) -> OneCycleLR:
    """
    Build OneCycleLR scheduler with configurable parameters.

    Args:
        optimizer: The optimizer to schedule.
        config: Full training configuration dictionary.
        steps_per_epoch: Number of optimizer steps per epoch.
        epochs: Total number of training epochs.

    Returns:
        Configured OneCycleLR scheduler.
    """
    scheduler_config = config["scheduler"]
    max_lr = scheduler_config.get("max_lr", config["training"].get("learning_rate", 2e-4))
    div_factor = scheduler_config.get("div_factor", 10.0)
    pct_start = scheduler_config.get("pct_start", 0.4)

    total_steps = steps_per_epoch * epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        div_factor=div_factor,
        pct_start=pct_start,
        anneal_strategy="cos",
        final_div_factor=1000.0,
    )
    return scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: Adam,
    scheduler: OneCycleLR,
    epoch: int,
    best_loss: float,
    save_path: str,
    is_distributed: bool = False,
) -> None:
    """
    Save a training checkpoint to disk.

    Args:
        model: The model (or DDP-wrapped model) to save.
        optimizer: The optimizer to save.
        scheduler: The LR scheduler to save.
        epoch: Current epoch number (0-indexed).
        best_loss: Best validation loss achieved so far.
        save_path: Full path where the checkpoint will be saved.
        is_distributed: Whether training is distributed (extracts module from DDP).
    """
    if is_distributed:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Adam] = None,
    scheduler: Optional[OneCycleLR] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float]:
    """
    Load a training checkpoint and restore model/optimizer/scheduler states.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: The model to restore weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        device: Device to map checkpoint tensors to.

    Returns:
        Tuple of (start_epoch, best_loss) from the checkpoint.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_loss = checkpoint.get("best_loss", float("inf"))

    return start_epoch, best_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: PointPillarsLoss,
    optimizer: Adam,
    scheduler: OneCycleLR,
    device: torch.device,
    epoch: int,
    max_grad_norm: float,
    log_interval: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Train the model for one full epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        criterion: Loss function computing cls, reg, and direction losses.
        optimizer: Optimizer for parameter updates.
        scheduler: Learning rate scheduler (stepped per batch).
        device: Device for computation.
        epoch: Current epoch number (for logging).
        max_grad_norm: Maximum gradient norm for clipping.
        log_interval: Print progress every N batches.
        logger: Logger instance.

    Returns:
        Dictionary of average losses: cls_loss, reg_loss, dir_loss, total_loss.
    """
    model.train()

    running_cls_loss = 0.0
    running_reg_loss = 0.0
    running_dir_loss = 0.0
    running_total_loss = 0.0
    num_batches = 0

    epoch_start_time = time.time()

    for batch_idx, batch_data in enumerate(dataloader):
        # Move batch data to device
        voxels = batch_data["voxels"].to(device, non_blocking=True)
        num_points = batch_data["num_points"].to(device, non_blocking=True)
        coordinates = batch_data["coordinates"].to(device, non_blocking=True)
        targets = {
            key: val.to(device, non_blocking=True)
            for key, val in batch_data["targets"].items()
        }

        # Forward pass
        predictions = model(voxels, num_points, coordinates)

        # Compute losses
        loss_dict = criterion(predictions, targets)
        cls_loss = loss_dict["cls_loss"]
        reg_loss = loss_dict["reg_loss"]
        dir_loss = loss_dict["dir_loss"]
        total_loss = loss_dict["total_loss"]

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # Optimizer and scheduler step
        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        running_cls_loss += cls_loss.item()
        running_reg_loss += reg_loss.item()
        running_dir_loss += dir_loss.item()
        running_total_loss += total_loss.item()
        num_batches += 1

        # Log progress at specified intervals
        if (batch_idx + 1) % log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            avg_total = running_total_loss / num_batches
            logger.info(
                f"Epoch [{epoch + 1}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {total_loss.item():.4f} (avg: {avg_total:.4f}) | "
                f"cls: {cls_loss.item():.4f} | reg: {reg_loss.item():.4f} | "
                f"dir: {dir_loss.item():.4f} | LR: {current_lr:.6f}"
            )

    epoch_duration = time.time() - epoch_start_time

    # Compute epoch averages
    avg_losses = {
        "cls_loss": running_cls_loss / max(num_batches, 1),
        "reg_loss": running_reg_loss / max(num_batches, 1),
        "dir_loss": running_dir_loss / max(num_batches, 1),
        "total_loss": running_total_loss / max(num_batches, 1),
    }

    logger.info(
        f"Epoch [{epoch + 1}] completed in {epoch_duration:.1f}s | "
        f"Avg Loss - total: {avg_losses['total_loss']:.4f}, "
        f"cls: {avg_losses['cls_loss']:.4f}, "
        f"reg: {avg_losses['reg_loss']:.4f}, "
        f"dir: {avg_losses['dir_loss']:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
    )

    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: PointPillarsLoss,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Evaluate the model on the validation set.

    Args:
        model: The model to evaluate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device for computation.
        logger: Logger instance.

    Returns:
        Dictionary of average validation losses.
    """
    model.eval()

    running_cls_loss = 0.0
    running_reg_loss = 0.0
    running_dir_loss = 0.0
    running_total_loss = 0.0
    num_batches = 0

    for batch_data in dataloader:
        voxels = batch_data["voxels"].to(device, non_blocking=True)
        num_points = batch_data["num_points"].to(device, non_blocking=True)
        coordinates = batch_data["coordinates"].to(device, non_blocking=True)
        targets = {
            key: val.to(device, non_blocking=True)
            for key, val in batch_data["targets"].items()
        }

        predictions = model(voxels, num_points, coordinates)
        loss_dict = criterion(predictions, targets)

        running_cls_loss += loss_dict["cls_loss"].item()
        running_reg_loss += loss_dict["reg_loss"].item()
        running_dir_loss += loss_dict["dir_loss"].item()
        running_total_loss += loss_dict["total_loss"].item()
        num_batches += 1

    avg_losses = {
        "cls_loss": running_cls_loss / max(num_batches, 1),
        "reg_loss": running_reg_loss / max(num_batches, 1),
        "dir_loss": running_dir_loss / max(num_batches, 1),
        "total_loss": running_total_loss / max(num_batches, 1),
    }

    logger.info(
        f"Validation | "
        f"Avg Loss - total: {avg_losses['total_loss']:.4f}, "
        f"cls: {avg_losses['cls_loss']:.4f}, "
        f"reg: {avg_losses['reg_loss']:.4f}, "
        f"dir: {avg_losses['dir_loss']:.4f}"
    )

    return avg_losses


def train(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """
    Main training loop orchestrating the full training pipeline.

    Handles model construction, optimizer/scheduler setup, checkpoint
    resumption, distributed training wrapping, and the epoch loop with
    validation and checkpoint saving.

    Args:
        config: Full training configuration dictionary.
        args: Parsed command-line arguments.
    """
    # Distributed setup
    distributed = args.distributed
    rank = 0
    world_size = 1
    local_rank = args.local_rank

    if distributed:
        # torchrun sets LOCAL_RANK env var
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        rank, world_size = setup_distributed(local_rank)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Reproducibility
    training_config = config["training"]
    seed = training_config.get("seed", 42)
    set_seed(seed + rank)

    # Logging
    checkpoint_dir = config["checkpoint"]["save_dir"]
    log_dir = os.path.join(checkpoint_dir, "logs")
    logger = setup_logging(log_dir, rank=rank)

    if rank == 0:
        logger.info(f"Training configuration: {config}")
        logger.info(f"Device: {device}, Distributed: {distributed}, World size: {world_size}")

    # Build datasets
    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")

    if rank == 0:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")

    # Build dataloaders
    train_loader, train_sampler = build_dataloader(
        train_dataset, config, distributed=distributed, is_training=True
    )
    val_loader, _ = build_dataloader(
        val_dataset, config, distributed=distributed, is_training=False
    )

    # Build model
    model = build_model(config, device)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
        )

    # Wrap model for distributed training
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # Build optimizer and scheduler
    epochs = training_config["epochs"]
    optimizer = build_optimizer(model, config)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch, epochs)

    # Build loss function
    loss_config = config.get("loss", {})
    criterion = PointPillarsLoss(
        num_classes=config["model"]["num_classes"],
        cls_weight=loss_config.get("cls_weight", 1.0),
        reg_weight=loss_config.get("reg_weight", 2.0),
        dir_weight=loss_config.get("dir_weight", 0.2),
    )

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float("inf")

    if args.resume is not None:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {args.resume}")

        # For DDP, load into the unwrapped model
        model_to_load = model.module if distributed else model
        start_epoch, best_loss = load_checkpoint(
            args.resume, model_to_load, optimizer, scheduler, device
        )

        if rank == 0:
            logger.info(
                f"Resumed at epoch {start_epoch} with best_loss={best_loss:.4f}"
            )

    # Training parameters
    max_grad_norm = training_config.get("max_grad_norm", 10.0)
    log_interval = training_config.get("log_interval", 50)
    save_interval = training_config.get("save_interval", 5)

    if rank == 0:
        logger.info(
            f"Starting training: epochs={epochs}, batch_size={training_config['batch_size']}, "
            f"lr={training_config['learning_rate']}, weight_decay={training_config['weight_decay']}"
        )
        logger.info(
            f"Scheduler: OneCycleLR max_lr={config['scheduler']['max_lr']}, "
            f"div_factor={config['scheduler']['div_factor']}, "
            f"pct_start={config['scheduler']['pct_start']}"
        )

    # Main training loop
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Set epoch for distributed sampler (ensures proper shuffling)
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train one epoch
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            max_grad_norm=max_grad_norm,
            log_interval=log_interval,
            logger=logger,
        )

        # Validation
        val_losses = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            logger=logger,
        )

        epoch_time = time.time() - epoch_start

        if rank == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] total time: {epoch_time:.1f}s | "
                f"Train loss: {train_losses['total_loss']:.4f} | "
                f"Val loss: {val_losses['total_loss']:.4f}"
            )

            val_total_loss = val_losses["total_loss"]

            # Save latest checkpoint
            latest_path = os.path.join(checkpoint_dir, "latest.pth")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_loss=best_loss,
                save_path=latest_path,
                is_distributed=distributed,
            )
            logger.info(f"Saved latest checkpoint: {latest_path}")

            # Save best model
            if val_total_loss < best_loss:
                best_loss = val_total_loss
                best_path = os.path.join(checkpoint_dir, "best.pth")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_loss=best_loss,
                    save_path=best_path,
                    is_distributed=distributed,
                )
                logger.info(
                    f"New best model saved: {best_path} (val_loss={best_loss:.4f})"
                )

            # Save periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                periodic_path = os.path.join(
                    checkpoint_dir, f"epoch_{epoch + 1:03d}.pth"
                )
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_loss=best_loss,
                    save_path=periodic_path,
                    is_distributed=distributed,
                )
                logger.info(f"Saved periodic checkpoint: {periodic_path}")

        # Synchronize all processes before next epoch
        if distributed:
            dist.barrier()

    # Cleanup
    if rank == 0:
        logger.info(
            f"Training complete. Best validation loss: {best_loss:.4f}"
        )

    if distributed:
        cleanup_distributed()


def main() -> None:
    """Main entry point: parse args, load config, and launch training."""
    args = parse_args()
    config = load_config(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
