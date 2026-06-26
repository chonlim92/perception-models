"""
Training script for HDMapNet.

Supports:
- Configuration via YAML config file
- Distributed training (DDP)
- Gradient accumulation
- AdamW optimizer with cosine annealing LR scheduler
- TensorBoard logging
- Periodic checkpointing and validation
"""

import os
import sys
import argparse
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from .model import HDMapNet
from .losses import HDMapNetLoss
from .dataset import NuScenesHDMapDataset, collate_fn


def get_default_config():
    """Return default training configuration."""
    return {
        # Data
        "dataroot": "/data/nuscenes",
        "train_ann_file": "/data/nuscenes/annotations/hdmap_train.json",
        "val_ann_file": "/data/nuscenes/annotations/hdmap_val.json",
        "image_size": [128, 352],
        "xbound": [-30.0, 30.0, 0.3],
        "ybound": [-15.0, 15.0, 0.3],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [4.0, 45.0, 1.0],
        "num_classes": 3,
        "thickness": 5,

        # Model
        "backbone": "efficientnet-b0",
        "pretrained_backbone": True,
        "backbone_out_channels": 64,
        "view_transform": "lss",
        "embedding_dim": 16,
        "bev_encoder_base_channels": 64,
        "head_mid_channels": 64,

        # Training
        "epochs": 30,
        "batch_size": 4,
        "num_workers": 4,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 1,
        "grad_accumulation_steps": 1,
        "clip_grad_norm": 5.0,

        # Loss weights
        "semantic_weight": 1.0,
        "instance_weight": 1.0,
        "direction_weight": 0.2,
        "use_focal": True,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "delta_v": 0.5,
        "delta_d": 3.0,

        # Logging & Checkpointing
        "log_dir": "./logs/hdmapnet",
        "checkpoint_dir": "./checkpoints/hdmapnet",
        "log_interval": 50,
        "val_interval": 1,
        "save_interval": 1,

        # Distributed
        "distributed": False,
        "local_rank": 0,
    }


def load_config(config_path):
    """Load configuration from YAML file and merge with defaults.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Merged configuration dict.
    """
    config = get_default_config()
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            config.update(user_config)
    return config


def build_model(config):
    """Build HDMapNet model from config.

    Args:
        config: Configuration dictionary.

    Returns:
        HDMapNet model instance.
    """
    model_config = {
        "backbone": config["backbone"],
        "pretrained_backbone": config["pretrained_backbone"],
        "backbone_out_channels": config["backbone_out_channels"],
        "view_transform": config["view_transform"],
        "xbound": config["xbound"],
        "ybound": config["ybound"],
        "zbound": config["zbound"],
        "dbound": config["dbound"],
        "image_size": config["image_size"],
        "num_classes": config["num_classes"],
        "embedding_dim": config["embedding_dim"],
        "bev_encoder_base_channels": config["bev_encoder_base_channels"],
        "head_mid_channels": config["head_mid_channels"],
    }
    return HDMapNet(model_config)


def build_dataloaders(config, distributed=False):
    """Build train and validation data loaders.

    Args:
        config: Configuration dictionary.
        distributed: Whether to use distributed samplers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = NuScenesHDMapDataset(
        dataroot=config["dataroot"],
        ann_file=config["train_ann_file"],
        image_size=tuple(config["image_size"]),
        xbound=tuple(config["xbound"]),
        ybound=tuple(config["ybound"]),
        num_classes=config["num_classes"],
        augment=True,
        thickness=config["thickness"],
    )

    val_dataset = NuScenesHDMapDataset(
        dataroot=config["dataroot"],
        ann_file=config["val_ann_file"],
        image_size=tuple(config["image_size"]),
        xbound=tuple(config["xbound"]),
        ybound=tuple(config["ybound"]),
        num_classes=config["num_classes"],
        augment=False,
        thickness=config["thickness"],
    )

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def build_optimizer_scheduler(model, config, steps_per_epoch):
    """Build AdamW optimizer and cosine annealing scheduler.

    Args:
        model: The model to optimize.
        config: Configuration dictionary.
        steps_per_epoch: Number of optimization steps per epoch.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    # Separate weight decay for different parameter groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"])

    # Cosine annealing with linear warmup
    warmup_steps = config["warmup_epochs"] * steps_per_epoch
    total_steps = config["epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_one_epoch(
    model, train_loader, criterion, optimizer, scheduler, device, config, epoch, writer, global_step
):
    """Train for one epoch.

    Args:
        model: The model.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: LR scheduler.
        device: Torch device.
        config: Configuration dict.
        epoch: Current epoch number.
        writer: TensorBoard writer.
        global_step: Current global step counter.

    Returns:
        Updated global_step.
    """
    model.train()
    accumulation_steps = config["grad_accumulation_steps"]
    log_interval = config["log_interval"]

    running_loss = 0.0
    running_sem_loss = 0.0
    running_inst_loss = 0.0
    running_dir_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        images = batch["images"].to(device)            # (B, N, 3, H, W)
        intrinsics = batch["intrinsics"].to(device)    # (B, N, 3, 3)
        extrinsics = batch["extrinsics"].to(device)    # (B, N, 4, 4)
        semantic_gt = batch["semantic_map"].to(device)  # (B, C, bev_h, bev_w)
        instance_gt = batch["instance_map"].to(device)  # (B, bev_h, bev_w)
        direction_gt = batch["direction_map"].to(device)  # (B, 2, bev_h, bev_w)

        # Forward pass
        predictions = model(images, intrinsics, extrinsics)

        targets = {
            "semantic": semantic_gt,
            "instance": instance_gt,
            "direction": direction_gt,
        }

        loss_dict = criterion(predictions, targets)
        loss = loss_dict["total"] / accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if config["clip_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # Logging
        running_loss += loss_dict["total"].item()
        running_sem_loss += loss_dict["semantic"].item()
        running_inst_loss += loss_dict["instance"].item()
        running_dir_loss += loss_dict["direction"].item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / num_batches
            avg_sem = running_sem_loss / num_batches
            avg_inst = running_inst_loss / num_batches
            avg_dir = running_dir_loss / num_batches

            if config["local_rank"] == 0:
                print(
                    f"  Epoch [{epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f} (Sem: {avg_sem:.4f}, Inst: {avg_inst:.4f}, Dir: {avg_dir:.4f}) "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )
                writer.add_scalar("train/total_loss", avg_loss, global_step)
                writer.add_scalar("train/semantic_loss", avg_sem, global_step)
                writer.add_scalar("train/instance_loss", avg_inst, global_step)
                writer.add_scalar("train/direction_loss", avg_dir, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            running_loss = 0.0
            running_sem_loss = 0.0
            running_inst_loss = 0.0
            running_dir_loss = 0.0
            num_batches = 0

    return global_step


@torch.no_grad()
def validate(model, val_loader, criterion, device, config, epoch, writer, global_step):
    """Run validation.

    Args:
        model: The model.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Torch device.
        config: Configuration dict.
        epoch: Current epoch number.
        writer: TensorBoard writer.
        global_step: Current global step.

    Returns:
        Dict with validation metrics.
    """
    model.eval()
    total_loss = 0.0
    total_sem_loss = 0.0
    total_inst_loss = 0.0
    total_dir_loss = 0.0
    num_batches = 0

    # IoU computation
    num_classes = config["num_classes"]
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    for batch in val_loader:
        images = batch["images"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        semantic_gt = batch["semantic_map"].to(device)
        instance_gt = batch["instance_map"].to(device)
        direction_gt = batch["direction_map"].to(device)

        predictions = model(images, intrinsics, extrinsics)
        targets = {
            "semantic": semantic_gt,
            "instance": instance_gt,
            "direction": direction_gt,
        }

        loss_dict = criterion(predictions, targets)
        total_loss += loss_dict["total"].item()
        total_sem_loss += loss_dict["semantic"].item()
        total_inst_loss += loss_dict["instance"].item()
        total_dir_loss += loss_dict["direction"].item()
        num_batches += 1

        # Compute IoU per class
        sem_pred = (predictions["semantic"].sigmoid() > 0.5).float()
        for c in range(num_classes):
            pred_c = sem_pred[:, c]
            gt_c = semantic_gt[:, c]
            intersection[c] += (pred_c * gt_c).sum()
            union[c] += ((pred_c + gt_c) > 0).float().sum()

    # Aggregate metrics
    avg_loss = total_loss / max(num_batches, 1)
    avg_sem = total_sem_loss / max(num_batches, 1)
    avg_inst = total_inst_loss / max(num_batches, 1)
    avg_dir = total_dir_loss / max(num_batches, 1)

    iou = intersection / union.clamp(min=1.0)
    mean_iou = iou.mean().item()

    if config["local_rank"] == 0:
        print(f"\n  Validation Epoch [{epoch}]:")
        print(f"    Loss: {avg_loss:.4f} (Sem: {avg_sem:.4f}, Inst: {avg_inst:.4f}, Dir: {avg_dir:.4f})")
        class_names = ["divider", "boundary", "crossing"]
        for c in range(num_classes):
            print(f"    IoU {class_names[c]}: {iou[c].item():.4f}")
        print(f"    Mean IoU: {mean_iou:.4f}\n")

        writer.add_scalar("val/total_loss", avg_loss, global_step)
        writer.add_scalar("val/semantic_loss", avg_sem, global_step)
        writer.add_scalar("val/instance_loss", avg_inst, global_step)
        writer.add_scalar("val/direction_loss", avg_dir, global_step)
        writer.add_scalar("val/mean_iou", mean_iou, global_step)
        for c in range(num_classes):
            writer.add_scalar(f"val/iou_{class_names[c]}", iou[c].item(), global_step)

    return {"loss": avg_loss, "mean_iou": mean_iou, "iou": iou.cpu().numpy()}


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, config, is_best=False):
    """Save a training checkpoint.

    Args:
        model: The model (possibly wrapped in DDP).
        optimizer: The optimizer.
        scheduler: The LR scheduler.
        epoch: Current epoch.
        global_step: Current global step.
        config: Configuration dict.
        is_best: Whether this is the best model so far.
    """
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # Get raw model state dict (unwrap DDP if needed)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }

    path = os.path.join(config["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, path)

    if is_best:
        best_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
        torch.save(checkpoint, best_path)

    # Keep only last 3 checkpoints
    existing = sorted(
        [f for f in os.listdir(config["checkpoint_dir"]) if f.startswith("checkpoint_epoch_")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    while len(existing) > 3:
        old_path = os.path.join(config["checkpoint_dir"], existing.pop(0))
        if os.path.exists(old_path):
            os.remove(old_path)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="HDMapNet Training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config["local_rank"] = args.local_rank

    # Setup distributed training
    distributed = config.get("distributed", False)
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["local_rank"] == 0:
        print("=" * 60)
        print("HDMapNet Training")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Backbone: {config['backbone']}")
        print(f"View transform: {config['view_transform']}")
        print(f"Image size: {config['image_size']}")
        print(f"BEV range: x{config['xbound']}, y{config['ybound']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Epochs: {config['epochs']}")
        print("=" * 60)

    # Build model
    model = build_model(config).to(device)
    if distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    # Build data loaders
    train_loader, val_loader = build_dataloaders(config, distributed=distributed)

    # Build optimizer and scheduler
    effective_steps_per_epoch = len(train_loader) // config["grad_accumulation_steps"]
    optimizer, scheduler = build_optimizer_scheduler(model, config, effective_steps_per_epoch)

    # Build loss
    criterion = HDMapNetLoss(
        semantic_weight=config["semantic_weight"],
        instance_weight=config["instance_weight"],
        direction_weight=config["direction_weight"],
        use_focal=config["use_focal"],
        focal_alpha=config["focal_alpha"],
        focal_gamma=config["focal_gamma"],
        delta_v=config["delta_v"],
        delta_d=config["delta_d"],
    )

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_iou = 0.0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        if config["local_rank"] == 0:
            print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # TensorBoard writer
    writer = None
    if config["local_rank"] == 0:
        os.makedirs(config["log_dir"], exist_ok=True)
        writer = SummaryWriter(log_dir=config["log_dir"])

    # Training loop
    for epoch in range(start_epoch, config["epochs"]):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        if config["local_rank"] == 0:
            print(f"\nEpoch {epoch}/{config['epochs'] - 1}")
            print("-" * 40)

        epoch_start = time.time()

        global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, config, epoch, writer, global_step,
        )

        epoch_time = time.time() - epoch_start
        if config["local_rank"] == 0:
            print(f"  Epoch time: {epoch_time:.1f}s")

        # Validation
        if (epoch + 1) % config["val_interval"] == 0:
            val_metrics = validate(model, val_loader, criterion, device, config, epoch, writer, global_step)

            is_best = val_metrics["mean_iou"] > best_iou
            if is_best:
                best_iou = val_metrics["mean_iou"]

            # Save checkpoint
            if config["local_rank"] == 0 and (epoch + 1) % config["save_interval"] == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, config, is_best=is_best)

    # Final save
    if config["local_rank"] == 0:
        save_checkpoint(model, optimizer, scheduler, config["epochs"] - 1, global_step, config)
        writer.close()
        print(f"\nTraining complete. Best IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
