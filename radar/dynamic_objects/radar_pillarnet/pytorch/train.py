"""
Training script for RadarPillarNet 3D object detection from radar point clouds.

Supports:
- Distributed training with DistributedDataParallel (multi-GPU)
- Mixed precision training with torch.cuda.amp
- Exponential Moving Average (EMA) of model weights
- One-cycle learning rate scheduling with AdamW optimizer
- Gradient clipping (max_norm=35)
- TensorBoard logging and periodic checkpoint saving
- YAML-based configuration

Usage:
    Single GPU:
        python train.py --config configs/radar_pillarnet.yaml

    Multi-GPU (distributed):
        torchrun --nproc_per_node=4 train.py --config configs/radar_pillarnet.yaml

    Override config values:
        python train.py --config configs/radar_pillarnet.yaml --batch_size 8 --lr 1e-3
"""

from __future__ import annotations

import argparse
import copy
import datetime
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")

from .model import RadarPillarNet
from .losses import RadarPillarNetLoss
from .heads import AnchorConfig


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(rank: int, output_dir: Path) -> logging.Logger:
    """Configure logging for the training process.

    Only rank 0 logs to console and file; other ranks log warnings+ to file only.

    Args:
        rank: Process rank in distributed training (0 for single GPU).
        output_dir: Directory where log files are written.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("radar_pillarnet")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s][Rank %(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Embed rank into logger name for identification
    logger = logging.getLogger(f"radar_pillarnet.rank{rank}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler for all ranks
    log_file = output_dir / f"train_rank{rank}.log"
    fh = logging.FileHandler(str(log_file), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler only for rank 0
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights that is updated as an exponential
    moving average of the training weights. The EMA model typically yields
    better validation performance than the instantaneous weights.

    EMA update rule:
        shadow = decay * shadow + (1 - decay) * current_params

    Args:
        model: The model whose parameters to track.
        decay: EMA decay factor (default 0.9999). Higher values = slower update.
        warmup_steps: Number of steps during which decay ramps from 0 to target.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
    ) -> None:
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Deep copy model parameters as shadow
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _get_decay(self) -> float:
        """Compute current decay with optional warmup ramp."""
        if self.step_count < self.warmup_steps:
            # Linear warmup from 0 to target decay
            return self.decay * (1 - math.exp(-self.step_count / self.warmup_steps))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA shadow parameters with current model parameters.

        Args:
            model: Model with current training weights.
        """
        self.step_count += 1
        decay = self._get_decay()

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Replace model parameters with EMA shadow parameters.

        Call this before validation or checkpoint saving to use EMA weights.
        Original parameters are backed up and can be restored with restore().

        Args:
            model: Model to apply shadow weights to.
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original model parameters after apply_shadow().

        Args:
            model: Model to restore original weights to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, Any]:
        """Serialize EMA state for checkpointing."""
        return {
            "shadow": self.shadow,
            "decay": self.decay,
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint."""
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]
        self.step_count = state_dict["step_count"]
        self.warmup_steps = state_dict["warmup_steps"]


# ---------------------------------------------------------------------------
# Dataset and collate
# ---------------------------------------------------------------------------

class RadarPillarNetDataset(Dataset):
    """Dataset wrapper for RadarPillarNet training.

    Expects pre-processed samples stored as individual .pt or .npz files.
    Each sample is a dictionary with keys:
        'pillars': (max_pillars, max_points_per_pillar, 9) float32
        'pillar_indices': (max_pillars, 3) int32
        'num_points_per_pillar': (max_pillars,) int32
        'gt_boxes': (M, 7) float32 - ground truth boxes [x,y,z,w,l,h,yaw]
        'gt_labels': (M,) int64 - class labels (0-indexed)
        'gt_velocity': (M, 2) float32 - ground truth velocity [vx, vy]
        'metadata': dict with scene/sample info

    Args:
        data_root: Root directory containing the data split.
        split: Dataset split ('train', 'val', 'test').
        info_path: Path to info pkl/json file listing all samples.
        max_pillars: Maximum number of pillars per sample.
        max_points_per_pillar: Maximum points per pillar.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        info_path: Optional[str] = None,
        max_pillars: int = 12000,
        max_points_per_pillar: int = 20,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.max_pillars = max_pillars
        self.max_points_per_pillar = max_points_per_pillar

        # Load sample list
        if info_path is not None:
            info_file = Path(info_path)
        else:
            info_file = self.data_root / f"{split}_info.pkl"

        if info_file.suffix == ".pkl":
            import pickle
            with open(info_file, "rb") as f:
                self.sample_infos: List[Dict[str, Any]] = pickle.load(f)
        elif info_file.suffix == ".json":
            import json
            with open(info_file, "r") as f:
                self.sample_infos = json.load(f)
        else:
            # Fallback: list all .pt files in split directory
            split_dir = self.data_root / split
            sample_files = sorted(split_dir.glob("*.pt"))
            self.sample_infos = [{"path": str(p)} for p in sample_files]

    def __len__(self) -> int:
        return len(self.sample_infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and return a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with pillars, indices, ground truth, and metadata.
        """
        info = self.sample_infos[idx]

        # Load sample data
        if "path" in info:
            sample_path = Path(info["path"])
            if not sample_path.is_absolute():
                sample_path = self.data_root / sample_path
        else:
            sample_path = self.data_root / self.split / f"{info['token']}.pt"

        sample = torch.load(str(sample_path), map_location="cpu", weights_only=False)

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching radar pillar samples.

    Handles variable-length ground truth boxes by padding to the maximum
    number of objects in the batch.

    Args:
        batch: List of sample dictionaries from the dataset.

    Returns:
        Batched dictionary with all tensors properly stacked/padded.
    """
    batch_size = len(batch)

    # Stack fixed-size tensors directly
    pillars = torch.stack([s["pillars"] for s in batch], dim=0)
    pillar_indices = torch.stack([s["pillar_indices"] for s in batch], dim=0)
    num_points_per_pillar = torch.stack(
        [s["num_points_per_pillar"] for s in batch], dim=0
    )

    # Pad variable-length ground truth to max objects in batch
    max_gt = max(s["gt_boxes"].shape[0] for s in batch)
    max_gt = max(max_gt, 1)  # Ensure at least 1 to avoid empty tensors

    gt_boxes = torch.zeros(batch_size, max_gt, 7, dtype=torch.float32)
    gt_labels = torch.full(
        (batch_size, max_gt), -1, dtype=torch.int64
    )  # -1 for padding
    gt_velocity = torch.zeros(batch_size, max_gt, 2, dtype=torch.float32)
    num_gt = torch.zeros(batch_size, dtype=torch.int32)

    for i, sample in enumerate(batch):
        n = sample["gt_boxes"].shape[0]
        if n > 0:
            gt_boxes[i, :n] = sample["gt_boxes"]
            gt_labels[i, :n] = sample["gt_labels"]
            gt_velocity[i, :n] = sample["gt_velocity"]
        num_gt[i] = n

    # Collect metadata
    metadata = [s.get("metadata", {}) for s in batch]

    return {
        "pillars": pillars,
        "pillar_indices": pillar_indices,
        "num_points_per_pillar": num_points_per_pillar,
        "gt_boxes": gt_boxes,
        "gt_labels": gt_labels,
        "gt_velocity": gt_velocity,
        "num_gt": num_gt,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_default_config() -> Dict[str, Any]:
    """Return default training configuration.

    Returns:
        Dictionary with all default hyperparameters and paths.
    """
    return {
        # Data
        "data_root": "/data/nuscenes/radar_pillars",
        "train_info": None,
        "val_info": None,
        "num_workers": 4,
        "pin_memory": True,
        # Model
        "model": {
            "in_channels": 9,
            "pillar_feat_channels": 64,
            "x_range": [-51.2, 51.2],
            "y_range": [-51.2, 51.2],
            "z_range": [-5.0, 3.0],
            "pillar_size": [0.4, 0.4, 8.0],
            "max_points_per_pillar": 20,
            "max_pillars": 12000,
            "layer_nums": [3, 5, 5],
            "layer_strides": [1, 2, 2],
            "num_filters": [64, 128, 256],
            "upsample_strides": [1, 2, 4],
            "num_upsample_filters": [128, 128, 128],
            "num_classes": 10,
            "nms_threshold": 0.2,
            "score_threshold": 0.1,
            "max_detections": 300,
        },
        # Loss
        "loss": {
            "cls_weight": 1.0,
            "reg_weight": 2.0,
            "vel_weight": 0.2,
            "dir_weight": 0.2,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "smooth_l1_beta": 1.0,
            "code_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        # Training
        "batch_size": 4,
        "epochs": 20,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "max_grad_norm": 35.0,
        "amp_enabled": True,
        # LR scheduler (OneCycleLR)
        "scheduler": {
            "max_lr": 1e-3,
            "pct_start": 0.4,
            "anneal_strategy": "cos",
            "div_factor": 10.0,
            "final_div_factor": 100.0,
        },
        # EMA
        "ema": {
            "enabled": True,
            "decay": 0.9999,
            "warmup_steps": 2000,
        },
        # Checkpointing
        "output_dir": "./output/radar_pillarnet",
        "save_every_n_epochs": 5,
        "val_every_n_epochs": 1,
        "resume_from": None,
        # Distributed
        "dist_backend": "nccl",
        "seed": 42,
    }


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file, merged with defaults.

    Args:
        config_path: Path to YAML config file, or None for defaults only.

    Returns:
        Merged configuration dictionary.
    """
    config = get_default_config()

    if config_path is not None and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        if user_config is not None:
            config = _deep_merge(config, user_config)

    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict.

    Args:
        base: Base configuration dictionary.
        override: Override values to merge in.

    Returns:
        Merged dictionary (base is mutated).
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# Distributed training utilities
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[int, int, bool]:
    """Initialize distributed training environment.

    Returns:
        Tuple of (rank, world_size, is_distributed).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=30),
        )
        torch.cuda.set_device(local_rank)
        return rank, world_size, True
    else:
        return 0, 1, False


def cleanup_distributed() -> None:
    """Destroy distributed process group if active."""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-reduce a tensor across all processes and average.

    Args:
        tensor: Tensor to reduce.
        world_size: Number of processes.

    Returns:
        Averaged tensor.
    """
    if world_size <= 1:
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    ema: Optional[ModelEMA],
    best_metric: float,
    config: Dict[str, Any],
    is_best: bool = False,
) -> None:
    """Save training checkpoint.

    Args:
        output_dir: Directory to save the checkpoint.
        epoch: Current epoch number (0-indexed).
        model: Model (may be DDP-wrapped).
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        scaler: GradScaler state (if using AMP).
        ema: EMA state (if enabled).
        best_metric: Best validation metric so far.
        config: Training configuration.
        is_best: Whether this is the best model so far.
    """
    # Extract model state dict from DDP wrapper if needed
    model_state = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "config": config,
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()

    # Save periodic checkpoint
    ckpt_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pth"
    torch.save(checkpoint, str(ckpt_path))

    # Save latest checkpoint (always overwritten)
    latest_path = output_dir / "checkpoint_latest.pth"
    torch.save(checkpoint, str(latest_path))

    # Save best checkpoint
    if is_best:
        best_path = output_dir / "checkpoint_best.pth"
        torch.save(checkpoint, str(best_path))


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    ema: Optional[ModelEMA] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float]:
    """Load a training checkpoint and restore state.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state into (should not be DDP-wrapped).
        optimizer: Optimizer to restore state (optional).
        scheduler: LR scheduler to restore state (optional).
        scaler: GradScaler to restore state (optional).
        ema: EMA to restore state (optional).
        device: Device to map checkpoint tensors to.

    Returns:
        Tuple of (start_epoch, best_metric).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("best_metric", float("inf"))

    return start_epoch, best_metric


# ---------------------------------------------------------------------------
# Target assignment (anchor matching)
# ---------------------------------------------------------------------------

def assign_targets(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_velocity: torch.Tensor,
    num_gt: torch.Tensor,
    anchors: torch.Tensor,
    anchor_configs: List[AnchorConfig],
    num_classes: int,
) -> Dict[str, torch.Tensor]:
    """Assign ground truth targets to anchors for loss computation.

    Uses IoU-based matching: anchors with IoU >= matched_threshold are positives,
    anchors with IoU < unmatched_threshold are negatives, and those in between
    are ignored during training.

    Args:
        gt_boxes: (B, M, 7) ground truth boxes.
        gt_labels: (B, M) ground truth class labels (0-indexed, -1 for padding).
        gt_velocity: (B, M, 2) ground truth velocities.
        num_gt: (B,) number of valid ground truth objects per sample.
        anchors: (H, W, A, 7) generated anchors.
        anchor_configs: Anchor config per class.
        num_classes: Total number of classes.

    Returns:
        Dictionary with target tensors for loss computation:
            'cls_targets': (B, N, num_classes) one-hot classification targets
            'box_targets': (B, N, 7) encoded box regression targets
            'vel_targets': (B, N, 2) velocity targets
            'dir_targets': (B, N) direction bin targets
            'pos_mask': (B, N) boolean mask for positive anchors
            'neg_mask': (B, N) boolean mask for negative anchors
    """
    from .heads import encode_boxes

    batch_size = gt_boxes.shape[0]
    device = gt_boxes.device

    # Flatten anchors
    h, w, num_anchors_per_loc, _ = anchors.shape
    total_anchors = h * w * num_anchors_per_loc
    anchors_flat = anchors.reshape(-1, 7).to(device)  # (N, 7)

    # Initialize targets
    cls_targets = torch.zeros(
        batch_size, total_anchors, num_classes, device=device, dtype=torch.float32
    )
    box_targets = torch.zeros(
        batch_size, total_anchors, 7, device=device, dtype=torch.float32
    )
    vel_targets = torch.zeros(
        batch_size, total_anchors, 2, device=device, dtype=torch.float32
    )
    dir_targets = torch.zeros(
        batch_size, total_anchors, device=device, dtype=torch.int64
    )
    pos_mask = torch.zeros(
        batch_size, total_anchors, device=device, dtype=torch.bool
    )
    neg_mask = torch.zeros(
        batch_size, total_anchors, device=device, dtype=torch.bool
    )

    # Global matched/unmatched thresholds (use minimum from all classes)
    matched_threshold = min(cfg.matched_threshold for cfg in anchor_configs)
    unmatched_threshold = min(cfg.unmatched_threshold for cfg in anchor_configs)

    for b in range(batch_size):
        n_gt = int(num_gt[b].item())
        if n_gt == 0:
            neg_mask[b, :] = True
            continue

        gt_b = gt_boxes[b, :n_gt]  # (n_gt, 7)
        labels_b = gt_labels[b, :n_gt]  # (n_gt,)
        vel_b = gt_velocity[b, :n_gt]  # (n_gt, 2)

        # Compute BEV IoU between anchors and GT using axis-aligned approximation
        iou_matrix = _compute_bev_iou(anchors_flat, gt_b)  # (N, n_gt)

        # For each anchor, find best matching GT
        max_iou_per_anchor, matched_gt_idx = iou_matrix.max(dim=1)  # (N,)

        # For each GT, find best matching anchor (to ensure every GT has >= 1 match)
        max_iou_per_gt, best_anchor_per_gt = iou_matrix.max(dim=0)  # (n_gt,)

        # Positive: IoU >= matched_threshold
        pos = max_iou_per_anchor >= matched_threshold

        # Negative: IoU < unmatched_threshold
        neg = max_iou_per_anchor < unmatched_threshold

        # Force best anchor per GT to be positive (handles low-IoU GTs)
        for gt_idx in range(n_gt):
            best_anchor = best_anchor_per_gt[gt_idx]
            pos[best_anchor] = True
            neg[best_anchor] = False
            matched_gt_idx[best_anchor] = gt_idx

        # Assign targets to positive anchors
        pos_mask[b] = pos
        neg_mask[b] = neg

        if pos.sum() > 0:
            pos_gt_idx = matched_gt_idx[pos]  # (P,)
            pos_gt_boxes = gt_b[pos_gt_idx]  # (P, 7)
            pos_gt_labels = labels_b[pos_gt_idx]  # (P,)
            pos_gt_vel = vel_b[pos_gt_idx]  # (P, 2)
            pos_anchors = anchors_flat[pos]  # (P, 7)

            # One-hot classification targets
            for i, label in enumerate(pos_gt_labels):
                if 0 <= label < num_classes:
                    cls_targets[b, pos.nonzero(as_tuple=True)[0][i], label] = 1.0

            # Box regression targets (encoded deltas)
            box_targets[b, pos] = encode_boxes(pos_gt_boxes, pos_anchors)

            # Velocity targets
            vel_targets[b, pos] = pos_gt_vel

            # Direction targets (0 if yaw in [-pi/2, pi/2], else 1)
            yaw = pos_gt_boxes[:, 6]
            dir_targets[b, pos] = (yaw > 0).long()

    return {
        "cls_targets": cls_targets,
        "box_targets": box_targets,
        "vel_targets": vel_targets,
        "dir_targets": dir_targets,
        "pos_mask": pos_mask,
        "neg_mask": neg_mask,
    }


def _compute_bev_iou(
    anchors: torch.Tensor, gt_boxes: torch.Tensor
) -> torch.Tensor:
    """Compute axis-aligned BEV IoU between anchors and ground truth boxes.

    Uses bounding-box approximation (ignoring rotation) for efficiency.

    Args:
        anchors: (N, 7) anchor boxes [x, y, z, w, l, h, theta].
        gt_boxes: (M, 7) ground truth boxes [x, y, z, w, l, h, theta].

    Returns:
        (N, M) IoU matrix.
    """
    # Convert to AABB in BEV: [x1, y1, x2, y2]
    anchor_x1 = anchors[:, 0] - anchors[:, 3] / 2  # (N,)
    anchor_y1 = anchors[:, 1] - anchors[:, 4] / 2
    anchor_x2 = anchors[:, 0] + anchors[:, 3] / 2
    anchor_y2 = anchors[:, 1] + anchors[:, 4] / 2

    gt_x1 = gt_boxes[:, 0] - gt_boxes[:, 3] / 2  # (M,)
    gt_y1 = gt_boxes[:, 1] - gt_boxes[:, 4] / 2
    gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 3] / 2
    gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 4] / 2

    # Compute pairwise intersection
    # (N, 1) vs (1, M) broadcasting
    inter_x1 = torch.max(anchor_x1.unsqueeze(1), gt_x1.unsqueeze(0))  # (N, M)
    inter_y1 = torch.max(anchor_y1.unsqueeze(1), gt_y1.unsqueeze(0))
    inter_x2 = torch.min(anchor_x2.unsqueeze(1), gt_x2.unsqueeze(0))
    inter_y2 = torch.min(anchor_y2.unsqueeze(1), gt_y2.unsqueeze(0))

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (
        inter_y2 - inter_y1
    ).clamp(min=0)  # (N, M)

    # Compute areas
    anchor_area = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)  # (N,)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)  # (M,)

    # Union
    union = anchor_area.unsqueeze(1) + gt_area.unsqueeze(0) - inter_area  # (N, M)

    iou = inter_area / union.clamp(min=1e-7)  # (N, M)

    return iou


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: RadarPillarNetLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    ema: Optional[ModelEMA],
    anchor_generator: Any,
    anchor_configs: List[AnchorConfig],
    epoch: int,
    config: Dict[str, Any],
    logger: logging.Logger,
    writer: Optional[SummaryWriter],
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model (possibly DDP-wrapped).
        dataloader: Training data loader.
        criterion: Loss function module.
        optimizer: Optimizer.
        scheduler: LR scheduler (stepped per iteration).
        scaler: GradScaler for mixed precision (None if disabled).
        ema: EMA tracker (None if disabled).
        anchor_generator: Anchor generator from model head.
        anchor_configs: Anchor configurations for target assignment.
        epoch: Current epoch number.
        config: Training configuration dictionary.
        logger: Logger instance.
        writer: TensorBoard SummaryWriter (None for non-rank-0).
        rank: Process rank.
        world_size: Total number of processes.

    Returns:
        Dictionary of average loss values for the epoch.
    """
    model.train()
    device = next(model.parameters()).device
    num_classes = config["model"]["num_classes"]
    max_grad_norm = config["max_grad_norm"]
    amp_enabled = config["amp_enabled"] and scaler is not None

    # Metrics accumulation
    loss_accum = {
        "total_loss": 0.0,
        "cls_loss": 0.0,
        "reg_loss": 0.0,
        "vel_loss": 0.0,
        "dir_loss": 0.0,
    }
    num_batches = 0
    epoch_start_time = time.time()

    # Generate anchors once (same for all samples)
    anchors = anchor_generator.generate_anchors(device)  # (H, W, A, 7)

    for batch_idx, batch_dict in enumerate(dataloader):
        iter_start_time = time.time()

        # Move inputs to device
        pillars = batch_dict["pillars"].to(device, non_blocking=True)
        pillar_indices = batch_dict["pillar_indices"].to(device, non_blocking=True)
        num_points_per_pillar = batch_dict["num_points_per_pillar"].to(
            device, non_blocking=True
        )
        gt_boxes = batch_dict["gt_boxes"].to(device, non_blocking=True)
        gt_labels = batch_dict["gt_labels"].to(device, non_blocking=True)
        gt_velocity = batch_dict["gt_velocity"].to(device, non_blocking=True)
        num_gt = batch_dict["num_gt"].to(device, non_blocking=True)

        # Prepare model input
        model_input = {
            "pillars": pillars,
            "pillar_indices": pillar_indices,
            "num_points_per_pillar": num_points_per_pillar,
        }

        # Assign targets
        targets = assign_targets(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_velocity=gt_velocity,
            num_gt=num_gt,
            anchors=anchors,
            anchor_configs=anchor_configs,
            num_classes=num_classes,
        )

        # Forward pass with mixed precision
        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            with autocast(device_type="cuda"):
                predictions = model(model_input)
                losses = criterion(
                    cls_preds=predictions["cls_preds"],
                    box_preds=predictions["box_preds"],
                    vel_preds=predictions["vel_preds"],
                    dir_preds=predictions["dir_preds"],
                    cls_targets=targets["cls_targets"],
                    box_targets=targets["box_targets"],
                    vel_targets=targets["vel_targets"],
                    dir_targets=targets["dir_targets"],
                    pos_mask=targets["pos_mask"],
                    neg_mask=targets["neg_mask"],
                )
                total_loss = losses["total_loss"]

            # Backward pass with scaled gradients
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(model_input)
            losses = criterion(
                cls_preds=predictions["cls_preds"],
                box_preds=predictions["box_preds"],
                vel_preds=predictions["vel_preds"],
                dir_preds=predictions["dir_preds"],
                cls_targets=targets["cls_targets"],
                box_targets=targets["box_targets"],
                vel_targets=targets["vel_targets"],
                dir_targets=targets["dir_targets"],
                pos_mask=targets["pos_mask"],
                neg_mask=targets["neg_mask"],
            )
            total_loss = losses["total_loss"]

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        # Step LR scheduler (per-iteration for OneCycleLR)
        scheduler.step()

        # Update EMA
        if ema is not None:
            ema_model = model.module if hasattr(model, "module") else model
            ema.update(ema_model)

        # Accumulate metrics
        batch_loss = {k: v.item() for k, v in losses.items()}
        for key in loss_accum:
            loss_accum[key] += batch_loss[key]
        num_batches += 1

        # Logging
        global_step = epoch * len(dataloader) + batch_idx
        iter_time = time.time() - iter_start_time

        if batch_idx % 50 == 0 and rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            pos_count = targets["pos_mask"].sum().item()
            neg_count = targets["neg_mask"].sum().item()

            logger.info(
                f"Epoch [{epoch}][{batch_idx}/{len(dataloader)}] "
                f"loss: {batch_loss['total_loss']:.4f} "
                f"(cls: {batch_loss['cls_loss']:.4f}, "
                f"reg: {batch_loss['reg_loss']:.4f}, "
                f"vel: {batch_loss['vel_loss']:.4f}, "
                f"dir: {batch_loss['dir_loss']:.4f}) "
                f"lr: {current_lr:.6f} "
                f"pos/neg: {pos_count}/{neg_count} "
                f"time: {iter_time:.3f}s"
            )

            if writer is not None:
                writer.add_scalar("train/total_loss", batch_loss["total_loss"], global_step)
                writer.add_scalar("train/cls_loss", batch_loss["cls_loss"], global_step)
                writer.add_scalar("train/reg_loss", batch_loss["reg_loss"], global_step)
                writer.add_scalar("train/vel_loss", batch_loss["vel_loss"], global_step)
                writer.add_scalar("train/dir_loss", batch_loss["dir_loss"], global_step)
                writer.add_scalar("train/learning_rate", current_lr, global_step)
                writer.add_scalar("train/pos_anchors", pos_count, global_step)
                writer.add_scalar("train/neg_anchors", neg_count, global_step)
                if amp_enabled:
                    writer.add_scalar(
                        "train/grad_scale", scaler.get_scale(), global_step
                    )

    # Compute epoch averages
    epoch_time = time.time() - epoch_start_time
    avg_losses = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}

    if rank == 0:
        logger.info(
            f"Epoch [{epoch}] completed in {epoch_time:.1f}s | "
            f"avg_loss: {avg_losses['total_loss']:.4f} "
            f"(cls: {avg_losses['cls_loss']:.4f}, "
            f"reg: {avg_losses['reg_loss']:.4f}, "
            f"vel: {avg_losses['vel_loss']:.4f}, "
            f"dir: {avg_losses['dir_loss']:.4f})"
        )

        if writer is not None:
            writer.add_scalar("epoch/total_loss", avg_losses["total_loss"], epoch)
            writer.add_scalar("epoch/cls_loss", avg_losses["cls_loss"], epoch)
            writer.add_scalar("epoch/reg_loss", avg_losses["reg_loss"], epoch)
            writer.add_scalar("epoch/vel_loss", avg_losses["vel_loss"], epoch)
            writer.add_scalar("epoch/dir_loss", avg_losses["dir_loss"], epoch)
            writer.add_scalar("epoch/epoch_time_sec", epoch_time, epoch)

    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: RadarPillarNetLoss,
    anchor_generator: Any,
    anchor_configs: List[AnchorConfig],
    config: Dict[str, Any],
    logger: logging.Logger,
    writer: Optional[SummaryWriter],
    epoch: int,
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    """Run validation and compute average losses.

    Args:
        model: Model (possibly DDP-wrapped).
        dataloader: Validation data loader.
        criterion: Loss function.
        anchor_generator: Anchor generator from model head.
        anchor_configs: Anchor configurations.
        config: Training configuration.
        logger: Logger instance.
        writer: TensorBoard writer (None for non-rank-0).
        epoch: Current epoch.
        rank: Process rank.
        world_size: Total processes.

    Returns:
        Dictionary of average validation loss values.
    """
    model.eval()
    device = next(model.parameters()).device
    num_classes = config["model"]["num_classes"]

    loss_accum = {
        "total_loss": 0.0,
        "cls_loss": 0.0,
        "reg_loss": 0.0,
        "vel_loss": 0.0,
        "dir_loss": 0.0,
    }
    num_batches = 0

    anchors = anchor_generator.generate_anchors(device)

    for batch_dict in dataloader:
        pillars = batch_dict["pillars"].to(device, non_blocking=True)
        pillar_indices = batch_dict["pillar_indices"].to(device, non_blocking=True)
        num_points_per_pillar = batch_dict["num_points_per_pillar"].to(
            device, non_blocking=True
        )
        gt_boxes = batch_dict["gt_boxes"].to(device, non_blocking=True)
        gt_labels = batch_dict["gt_labels"].to(device, non_blocking=True)
        gt_velocity = batch_dict["gt_velocity"].to(device, non_blocking=True)
        num_gt = batch_dict["num_gt"].to(device, non_blocking=True)

        model_input = {
            "pillars": pillars,
            "pillar_indices": pillar_indices,
            "num_points_per_pillar": num_points_per_pillar,
        }

        targets = assign_targets(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_velocity=gt_velocity,
            num_gt=num_gt,
            anchors=anchors,
            anchor_configs=anchor_configs,
            num_classes=num_classes,
        )

        predictions = model(model_input)
        losses = criterion(
            cls_preds=predictions["cls_preds"],
            box_preds=predictions["box_preds"],
            vel_preds=predictions["vel_preds"],
            dir_preds=predictions["dir_preds"],
            cls_targets=targets["cls_targets"],
            box_targets=targets["box_targets"],
            vel_targets=targets["vel_targets"],
            dir_targets=targets["dir_targets"],
            pos_mask=targets["pos_mask"],
            neg_mask=targets["neg_mask"],
        )

        for key in loss_accum:
            loss_accum[key] += losses[key].item()
        num_batches += 1

    avg_losses = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}

    # Reduce across processes for distributed validation
    if world_size > 1:
        for key in avg_losses:
            tensor = torch.tensor(avg_losses[key], device=device)
            tensor = reduce_tensor(tensor, world_size)
            avg_losses[key] = tensor.item()

    if rank == 0:
        logger.info(
            f"Validation [{epoch}] | "
            f"avg_loss: {avg_losses['total_loss']:.4f} "
            f"(cls: {avg_losses['cls_loss']:.4f}, "
            f"reg: {avg_losses['reg_loss']:.4f}, "
            f"vel: {avg_losses['vel_loss']:.4f}, "
            f"dir: {avg_losses['dir_loss']:.4f})"
        )

        if writer is not None:
            writer.add_scalar("val/total_loss", avg_losses["total_loss"], epoch)
            writer.add_scalar("val/cls_loss", avg_losses["cls_loss"], epoch)
            writer.add_scalar("val/reg_loss", avg_losses["reg_loss"], epoch)
            writer.add_scalar("val/vel_loss", avg_losses["vel_loss"], epoch)
            writer.add_scalar("val/dir_loss", avg_losses["dir_loss"], epoch)

    return avg_losses


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: Dict[str, Any]) -> None:
    """Main training entry point.

    Orchestrates the full training pipeline: setup distributed environment,
    build model/optimizer/scheduler, run training loop with validation,
    and save checkpoints.

    Args:
        config: Complete training configuration dictionary.
    """
    # Setup distributed
    rank, world_size, is_distributed = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Seed for reproducibility
    seed = config["seed"] + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Output directory
    output_dir = Path(config["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tensorboard").mkdir(exist_ok=True)

    # Synchronize before logging setup
    if is_distributed:
        dist.barrier()

    # Logging
    logger = setup_logging(rank, output_dir)
    logger.info(f"Training config: {config}")
    logger.info(f"Device: {device}, Distributed: {is_distributed}, World size: {world_size}")

    # TensorBoard writer (rank 0 only)
    writer: Optional[SummaryWriter] = None
    if rank == 0:
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    model_cfg = config["model"]
    model = RadarPillarNet(
        in_channels=model_cfg["in_channels"],
        pillar_feat_channels=model_cfg["pillar_feat_channels"],
        x_range=tuple(model_cfg["x_range"]),
        y_range=tuple(model_cfg["y_range"]),
        z_range=tuple(model_cfg["z_range"]),
        pillar_size=tuple(model_cfg["pillar_size"]),
        max_points_per_pillar=model_cfg["max_points_per_pillar"],
        max_pillars=model_cfg["max_pillars"],
        layer_nums=model_cfg.get("layer_nums"),
        layer_strides=model_cfg.get("layer_strides"),
        num_filters=model_cfg.get("num_filters"),
        upsample_strides=model_cfg.get("upsample_strides"),
        num_upsample_filters=model_cfg.get("num_upsample_filters"),
        num_classes=model_cfg["num_classes"],
        nms_threshold=model_cfg.get("nms_threshold", 0.2),
        score_threshold=model_cfg.get("score_threshold", 0.1),
        max_detections=model_cfg.get("max_detections", 300),
    )
    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
    )

    # Get anchor generator and configs from model head
    anchor_generator = model.head.anchor_generator
    anchor_configs = model.head.anchor_configs

    # -------------------------------------------------------------------------
    # Loss function
    # -------------------------------------------------------------------------
    loss_cfg = config["loss"]
    criterion = RadarPillarNetLoss(
        num_classes=model_cfg["num_classes"],
        cls_weight=loss_cfg["cls_weight"],
        reg_weight=loss_cfg["reg_weight"],
        vel_weight=loss_cfg["vel_weight"],
        dir_weight=loss_cfg["dir_weight"],
        focal_alpha=loss_cfg["focal_alpha"],
        focal_gamma=loss_cfg["focal_gamma"],
        smooth_l1_beta=loss_cfg["smooth_l1_beta"],
        code_weights=loss_cfg.get("code_weights"),
    ).to(device)

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=tuple(config["betas"]),
        weight_decay=config["weight_decay"],
    )

    # -------------------------------------------------------------------------
    # Dataset and DataLoader
    # -------------------------------------------------------------------------
    train_dataset = RadarPillarNetDataset(
        data_root=config["data_root"],
        split="train",
        info_path=config.get("train_info"),
        max_pillars=model_cfg["max_pillars"],
        max_points_per_pillar=model_cfg["max_points_per_pillar"],
    )

    val_dataset = RadarPillarNetDataset(
        data_root=config["data_root"],
        split="val",
        info_path=config.get("val_info"),
        max_pillars=model_cfg["max_pillars"],
        max_points_per_pillar=model_cfg["max_points_per_pillar"],
    )

    # Samplers
    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=config["num_workers"] > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=config["num_workers"] > 0,
    )

    logger.info(
        f"Dataset: {len(train_dataset)} train samples, {len(val_dataset)} val samples"
    )
    logger.info(
        f"DataLoader: batch_size={config['batch_size']}, "
        f"num_workers={config['num_workers']}, "
        f"iterations_per_epoch={len(train_loader)}"
    )

    # -------------------------------------------------------------------------
    # LR Scheduler (OneCycleLR - steps per iteration)
    # -------------------------------------------------------------------------
    sched_cfg = config["scheduler"]
    total_steps = config["epochs"] * len(train_loader)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=sched_cfg.get("max_lr", config["lr"]),
        total_steps=total_steps,
        pct_start=sched_cfg.get("pct_start", 0.4),
        anneal_strategy=sched_cfg.get("anneal_strategy", "cos"),
        div_factor=sched_cfg.get("div_factor", 10.0),
        final_div_factor=sched_cfg.get("final_div_factor", 100.0),
    )

    # -------------------------------------------------------------------------
    # Mixed precision scaler
    # -------------------------------------------------------------------------
    scaler: Optional[GradScaler] = None
    if config["amp_enabled"] and torch.cuda.is_available():
        scaler = GradScaler()
        logger.info("Mixed precision training (AMP) enabled")

    # -------------------------------------------------------------------------
    # EMA
    # -------------------------------------------------------------------------
    ema: Optional[ModelEMA] = None
    ema_cfg = config.get("ema", {})
    if ema_cfg.get("enabled", False):
        ema = ModelEMA(
            model,
            decay=ema_cfg.get("decay", 0.9999),
            warmup_steps=ema_cfg.get("warmup_steps", 2000),
        )
        logger.info(
            f"EMA enabled: decay={ema_cfg.get('decay', 0.9999)}, "
            f"warmup_steps={ema_cfg.get('warmup_steps', 2000)}"
        )

    # -------------------------------------------------------------------------
    # Resume from checkpoint
    # -------------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")

    if config.get("resume_from") is not None:
        resume_path = config["resume_from"]
        logger.info(f"Resuming from checkpoint: {resume_path}")
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            device=device,
        )
        logger.info(
            f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}"
        )

    # -------------------------------------------------------------------------
    # Distributed Data Parallel
    # -------------------------------------------------------------------------
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        logger.info(f"Wrapped model with DDP (rank={rank}, local_rank={local_rank})")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    logger.info(
        f"Starting training: epochs={config['epochs']}, "
        f"start_epoch={start_epoch}, total_steps={total_steps}"
    )

    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()

        # Set epoch for distributed sampler (ensures proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train one epoch
        train_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            anchor_generator=anchor_generator,
            anchor_configs=anchor_configs,
            epoch=epoch,
            config=config,
            logger=logger,
            writer=writer,
            rank=rank,
            world_size=world_size,
        )

        # Validation
        is_best = False
        if (epoch + 1) % config["val_every_n_epochs"] == 0:
            # Apply EMA weights for validation
            if ema is not None:
                ema_model = model.module if hasattr(model, "module") else model
                ema.apply_shadow(ema_model)

            val_losses = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                anchor_generator=anchor_generator,
                anchor_configs=anchor_configs,
                config=config,
                logger=logger,
                writer=writer,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
            )

            # Restore training weights after validation
            if ema is not None:
                ema_model = model.module if hasattr(model, "module") else model
                ema.restore(ema_model)

            # Track best model
            if val_losses["total_loss"] < best_val_loss:
                best_val_loss = val_losses["total_loss"]
                is_best = True
                logger.info(
                    f"New best validation loss: {best_val_loss:.4f} at epoch {epoch}"
                )

        # Save checkpoints (rank 0 only)
        if rank == 0:
            should_save = (
                is_best
                or (epoch + 1) % config["save_every_n_epochs"] == 0
                or (epoch + 1) == config["epochs"]
            )
            if should_save:
                save_checkpoint(
                    output_dir=output_dir,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    best_metric=best_val_loss,
                    config=config,
                    is_best=is_best,
                )
                logger.info(
                    f"Saved checkpoint at epoch {epoch} "
                    f"(is_best={is_best}, val_loss={best_val_loss:.4f})"
                )

        epoch_time = time.time() - epoch_start
        if rank == 0:
            logger.info(f"Epoch [{epoch}] total time: {epoch_time:.1f}s")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    if writer is not None:
        writer.close()

    cleanup_distributed()
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train RadarPillarNet for 3D radar object detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )

    # Override common config values from CLI
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs (informational; use torchrun for multi-GPU).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Disable exponential moving average.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Maximum gradient norm for clipping.",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--val_every_n_epochs",
        type=int,
        default=None,
        help="Run validation every N epochs.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point: parse args, load config, and launch training."""
    args = parse_args()

    # Load base config from YAML
    config = load_config(args.config)

    # Override with CLI arguments
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr
        config["scheduler"]["max_lr"] = args.lr
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.resume_from is not None:
        config["resume_from"] = args.resume_from
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_amp:
        config["amp_enabled"] = False
    if args.no_ema:
        config["ema"]["enabled"] = False
    if args.max_grad_norm is not None:
        config["max_grad_norm"] = args.max_grad_norm
    if args.save_every_n_epochs is not None:
        config["save_every_n_epochs"] = args.save_every_n_epochs
    if args.val_every_n_epochs is not None:
        config["val_every_n_epochs"] = args.val_every_n_epochs

    # Launch training
    train(config)


if __name__ == "__main__":
    main()
