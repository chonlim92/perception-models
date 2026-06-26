"""
StreamMapNet - Training Script (PyTorch)

Trains the StreamMapNet model for online HD map construction with temporal
sequence batching, ego-motion propagation, and multi-GPU distributed training.

Usage:
    # Single GPU
    python train.py --config configs/stream_mapnet_base.yaml

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 train.py --config configs/stream_mapnet_base.yaml

    # Resume from checkpoint
    python train.py --config configs/stream_mapnet_base.yaml --resume checkpoints/epoch_12.pth
"""

import argparse
import datetime
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import yaml

from model import StreamMapNet, build_stream_mapnet


# =============================================================================
# Loss Functions
# =============================================================================


class FocalLoss(nn.Module):
    """Focal loss for classification with class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw classification logits
            targets: (N,) integer class labels

        Returns:
            Scalar focal loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        probs = logits.softmax(dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


class StreamMapNetLoss(nn.Module):
    """
    Combined loss for StreamMapNet with Hungarian matching.

    Computes bipartite matching between predictions and ground truth,
    then applies classification (focal) and regression (L1) losses.
    """

    def __init__(self, config: dict):
        super().__init__()
        loss_cfg = config.get("loss", {})
        matching_cfg = loss_cfg.get("matching_cost", {})
        weights_cfg = loss_cfg.get("loss_weights", {})
        focal_cfg = loss_cfg.get("focal_loss", {})

        self.cls_weight = weights_cfg.get("cls_loss", 2.0)
        self.pts_weight = weights_cfg.get("point_loss", 5.0)
        self.dir_weight = weights_cfg.get("direction_loss", 0.5)
        self.aux_weight = loss_cfg.get("aux_loss_weight", 0.5)

        self.match_cls_weight = matching_cfg.get("cls_weight", 2.0)
        self.match_pts_weight = matching_cfg.get("point_weight", 5.0)

        self.num_classes = config.get("data", {}).get("num_classes", 3)

        self.focal_loss = FocalLoss(
            alpha=focal_cfg.get("alpha", 0.25),
            gamma=focal_cfg.get("gamma", 2.0),
        )

    @torch.no_grad()
    def hungarian_matching(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Hungarian matching between predictions and ground truth.

        Args:
            pred_logits: (N_q, num_classes+1) classification logits
            pred_points: (N_q, K, 2) predicted polyline points
            gt_labels: (N_gt,) class indices
            gt_points: (N_gt, K, 2) ground truth points

        Returns:
            pred_idx: (N_gt,) matched prediction indices
            gt_idx: (N_gt,) ground truth indices
        """
        from scipy.optimize import linear_sum_assignment

        N_q = pred_logits.shape[0]
        N_gt = gt_labels.shape[0]

        if N_gt == 0:
            return (
                torch.tensor([], dtype=torch.long, device=pred_logits.device),
                torch.tensor([], dtype=torch.long, device=pred_logits.device),
            )

        # Classification cost: negative probability of correct class
        probs = pred_logits.softmax(dim=-1)  # (N_q, C+1)
        cls_cost = -probs[:, gt_labels]  # (N_q, N_gt)

        # Point regression cost: L1 distance
        pts_flat_pred = pred_points.flatten(1)  # (N_q, K*2)
        pts_flat_gt = gt_points.flatten(1).to(pred_points.device)  # (N_gt, K*2)
        pts_cost = torch.cdist(pts_flat_pred, pts_flat_gt, p=1)  # (N_q, N_gt)

        # Combined cost
        cost = self.match_cls_weight * cls_cost + self.match_pts_weight * pts_cost
        cost_np = cost.cpu().numpy()

        row_idx, col_idx = linear_sum_assignment(cost_np)
        return (
            torch.tensor(row_idx, dtype=torch.long, device=pred_logits.device),
            torch.tensor(col_idx, dtype=torch.long, device=pred_logits.device),
        )

    def _compute_loss_single_layer(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a single decoder layer output.

        Args:
            pred_logits: (B, N_q, C+1)
            pred_points: (B, N_q, K, 2)
            targets: list of dicts with 'labels' and 'points'

        Returns:
            dict with loss_cls, loss_pts, loss_total
        """
        B = pred_logits.shape[0]
        N_q = pred_logits.shape[1]
        device = pred_logits.device

        total_cls_loss = torch.tensor(0.0, device=device)
        total_pts_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            gt_labels = targets[b]["labels"].to(device)
            gt_points = targets[b]["points"].to(device)

            # Hungarian matching
            pred_idx, gt_idx = self.hungarian_matching(
                pred_logits[b], pred_points[b], gt_labels, gt_points
            )

            # Classification loss (all queries)
            target_cls = torch.full(
                (N_q,), self.num_classes,  # background class
                dtype=torch.long, device=device,
            )
            if len(pred_idx) > 0:
                target_cls[pred_idx] = gt_labels[gt_idx]

            total_cls_loss += self.focal_loss(pred_logits[b], target_cls)

            # Point regression loss (matched queries only)
            if len(pred_idx) > 0:
                matched_pts = pred_points[b][pred_idx]  # (N_gt, K, 2)
                target_pts = gt_points[gt_idx]  # (N_gt, K, 2)
                total_pts_loss += F.l1_loss(matched_pts, target_pts)

        total_cls_loss /= B
        total_pts_loss /= B
        loss_total = self.cls_weight * total_cls_loss + self.pts_weight * total_pts_loss

        return {
            "loss_cls": total_cls_loss,
            "loss_pts": total_pts_loss,
            "loss_total": loss_total,
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss including auxiliary losses from intermediate layers.

        Args:
            outputs: model output dict with 'pred_logits', 'pred_points', 'aux_outputs'
            targets: list of B dicts with 'labels' (N_gt,) and 'points' (N_gt, K, 2)

        Returns:
            dict with all loss components and total loss
        """
        # Main loss (final decoder layer)
        losses = self._compute_loss_single_layer(
            outputs["pred_logits"], outputs["pred_points"], targets
        )

        # Auxiliary losses from intermediate decoder layers
        if "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                aux_losses = self._compute_loss_single_layer(
                    aux_out["pred_logits"], aux_out["pred_points"], targets
                )
                losses[f"aux_{i}_loss_cls"] = aux_losses["loss_cls"]
                losses[f"aux_{i}_loss_pts"] = aux_losses["loss_pts"]
                losses["loss_total"] += self.aux_weight * aux_losses["loss_total"]

        return losses


# =============================================================================
# Dataset (Temporal Sequence)
# =============================================================================


class StreamMapNetDataset(Dataset):
    """
    Dataset for StreamMapNet with temporal sequence batching.

    Each sample is a sequence of consecutive frames with associated
    camera data, ego poses, and map annotations.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        sequence_length: int = 8,
        img_size: Tuple[int, int] = (256, 704),
        num_cameras: int = 6,
        num_classes: int = 3,
        num_points: int = 20,
        augment: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.num_cameras = num_cameras
        self.num_classes = num_classes
        self.num_points = num_points
        self.augment = augment and (split == "train")

        # Load annotation index (scene_id, frame indices, paths)
        self.sequences = self._load_sequences()

        # Image normalization
        self.img_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1) / 255.0
        self.img_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1) / 255.0

    def _load_sequences(self) -> List[Dict]:
        """
        Load sequence annotations.

        In a real implementation, this would load from disk (nuScenes, Argoverse2).
        Here we generate synthetic metadata for demonstration.
        """
        sequences = []
        # Generate synthetic sequences for demonstration
        num_scenes = 100 if self.split == "train" else 20
        frames_per_scene = 40

        for scene_id in range(num_scenes):
            num_seqs = (frames_per_scene - self.sequence_length) + 1
            for start_idx in range(0, num_seqs, self.sequence_length // 2):
                sequences.append({
                    "scene_id": scene_id,
                    "start_frame": start_idx,
                    "length": self.sequence_length,
                })
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def _generate_sample_frame(self) -> Dict[str, torch.Tensor]:
        """
        Generate a synthetic training sample for one frame.

        In a real implementation, this loads actual images, calibration,
        and map annotations from the dataset.
        """
        H, W = self.img_size

        # Multi-camera images: (N, 3, H, W)
        images = torch.randn(self.num_cameras, 3, H, W) * 0.2 + 0.5
        images = images.clamp(0, 1)

        # Camera intrinsics: (N, 3, 3)
        intrinsics = torch.zeros(self.num_cameras, 3, 3)
        intrinsics[:, 0, 0] = 1260.0  # fx
        intrinsics[:, 1, 1] = 1260.0  # fy
        intrinsics[:, 0, 2] = W / 2.0  # cx
        intrinsics[:, 1, 2] = H / 2.0  # cy
        intrinsics[:, 2, 2] = 1.0

        # Camera extrinsics: (N, 4, 4) camera-to-ego
        extrinsics = torch.eye(4).unsqueeze(0).expand(self.num_cameras, -1, -1).clone()
        for cam in range(self.num_cameras):
            angle = cam * (2.0 * math.pi / self.num_cameras)
            extrinsics[cam, 0, 3] = 1.5 * math.cos(angle)
            extrinsics[cam, 1, 3] = 1.5 * math.sin(angle)
            extrinsics[cam, 2, 3] = 1.6

        # Ego motion: (4, 4) transformation from previous to current
        ego_motion = torch.eye(4)
        # Small forward translation + slight rotation
        ego_motion[0, 3] = random.uniform(0.3, 1.0)  # forward movement
        theta = random.uniform(-0.02, 0.02)  # small yaw
        ego_motion[0, 0] = math.cos(theta)
        ego_motion[0, 1] = -math.sin(theta)
        ego_motion[1, 0] = math.sin(theta)
        ego_motion[1, 1] = math.cos(theta)

        # Ground truth map elements
        num_elements = random.randint(5, 25)
        labels = torch.randint(0, self.num_classes, (num_elements,))
        # Points normalized to [0, 1] representing polylines in BEV
        points = torch.rand(num_elements, self.num_points, 2)
        # Make polylines smooth (sorted along one dimension)
        points[:, :, 0] = points[:, :, 0].sort(dim=1)[0]

        return {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "ego_motion": ego_motion,
            "labels": labels,
            "points": points,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a temporal sequence sample.

        Returns a dict with tensors for the sequence:
            images: (T, N, 3, H, W)
            intrinsics: (T, N, 3, 3)
            extrinsics: (T, N, 4, 4)
            ego_motions: (T, 4, 4)  identity for first frame
            targets: list of T dicts with 'labels' and 'points'
        """
        seq_info = self.sequences[idx]
        T = seq_info["length"]

        all_images = []
        all_intrinsics = []
        all_extrinsics = []
        all_ego_motions = []
        all_targets = []

        for t in range(T):
            frame = self._generate_sample_frame()

            # Normalize images
            images = frame["images"]  # (N, 3, H, W) in [0, 1]
            images = (images - self.img_mean) / self.img_std

            all_images.append(images)
            all_intrinsics.append(frame["intrinsics"])
            all_extrinsics.append(frame["extrinsics"])

            # First frame has identity ego motion
            if t == 0:
                all_ego_motions.append(torch.eye(4))
            else:
                all_ego_motions.append(frame["ego_motion"])

            all_targets.append({
                "labels": frame["labels"],
                "points": frame["points"],
            })

        return {
            "images": torch.stack(all_images),         # (T, N, 3, H, W)
            "intrinsics": torch.stack(all_intrinsics), # (T, N, 3, 3)
            "extrinsics": torch.stack(all_extrinsics), # (T, N, 4, 4)
            "ego_motions": torch.stack(all_ego_motions),  # (T, 4, 4)
            "targets": all_targets,                    # list of T dicts
        }


def collate_temporal_sequences(batch: List[Dict]) -> Dict:
    """
    Custom collate function for temporal sequences.

    Stacks batch dimension while keeping temporal targets as nested lists.
    """
    B = len(batch)
    T = batch[0]["images"].shape[0]

    images = torch.stack([b["images"] for b in batch])          # (B, T, N, 3, H, W)
    intrinsics = torch.stack([b["intrinsics"] for b in batch])  # (B, T, N, 3, 3)
    extrinsics = torch.stack([b["extrinsics"] for b in batch])  # (B, T, N, 4, 4)
    ego_motions = torch.stack([b["ego_motions"] for b in batch])  # (B, T, 4, 4)

    # Targets: list of T lists, each inner list has B dicts
    targets = []
    for t in range(T):
        frame_targets = [batch[b]["targets"][t] for b in range(B)]
        targets.append(frame_targets)

    return {
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "ego_motions": ego_motions,
        "targets": targets,
    }


# =============================================================================
# Learning Rate Schedule
# =============================================================================


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
):
    """Cosine annealing schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Distributed Training Utilities
# =============================================================================


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# =============================================================================
# Training Loop
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    criterion: StreamMapNetLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    epoch: int,
    config: dict,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
) -> Tuple[float, int]:
    """
    Train for one epoch over temporal sequences.

    For each batch, iterate over the temporal dimension, propagating
    BEV hidden state across frames within each sequence.
    """
    model.train()
    training_cfg = config.get("training", {})
    grad_clip = training_cfg.get("grad_clip", {}).get("max_norm", 35.0)
    log_interval = training_cfg.get("log_interval", 50)
    use_fp16 = training_cfg.get("fp16", {}).get("enabled", True)
    accumulate_steps = training_cfg.get("accumulate_grad_batches", 1)

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].cuda(non_blocking=True)        # (B, T, N, 3, H, W)
        intrinsics = batch["intrinsics"].cuda(non_blocking=True)  # (B, T, N, 3, 3)
        extrinsics = batch["extrinsics"].cuda(non_blocking=True)  # (B, T, N, 4, 4)
        ego_motions = batch["ego_motions"].cuda(non_blocking=True)  # (B, T, 4, 4)
        targets = batch["targets"]  # list of T lists of B dicts

        B, T = images.shape[:2]

        # Reset temporal state at the start of each new sequence
        if hasattr(model, "module"):
            model.module.reset_temporal_state()
        else:
            model.reset_temporal_state()

        # Accumulate loss over temporal steps
        sequence_loss = torch.tensor(0.0, device=images.device)

        for t in range(T):
            # Get ego motion (None for first frame, identity is also fine)
            ego_motion_t = ego_motions[:, t] if t > 0 else None

            # Forward pass with mixed precision
            with autocast(enabled=use_fp16):
                outputs = model(
                    images[:, t],      # (B, N, 3, H, W)
                    intrinsics[:, t],  # (B, N, 3, 3)
                    extrinsics[:, t],  # (B, N, 4, 4)
                    ego_motion=ego_motion_t,
                )

                # Compute loss for this frame
                frame_targets = targets[t]
                losses = criterion(outputs, frame_targets)
                frame_loss = losses["loss_total"] / T  # Average over sequence

            sequence_loss += frame_loss.detach()

            # Backward pass
            scaler.scale(frame_loss / accumulate_steps).backward()

        # Optimizer step (after processing full sequence)
        if (batch_idx + 1) % accumulate_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += sequence_loss.item()
        num_batches += 1
        global_step += 1

        # Logging
        if is_main_process() and (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / num_batches
            lr = optimizer.param_groups[0]["lr"]
            samples_per_sec = (batch_idx + 1) * B / elapsed

            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {sequence_loss.item():.4f} (avg: {avg_loss:.4f}) | "
                f"LR: {lr:.2e} | Speed: {samples_per_sec:.1f} seq/s"
            )

            if writer is not None:
                writer.add_scalar("train/loss", sequence_loss.item(), global_step)
                writer.add_scalar("train/loss_cls", losses["loss_cls"].item(), global_step)
                writer.add_scalar("train/loss_pts", losses["loss_pts"].item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)

    avg_epoch_loss = total_loss / max(num_batches, 1)
    return avg_epoch_loss, global_step


# =============================================================================
# Checkpoint Management
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    loss: float,
    save_path: str,
):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }
    torch.save(state, save_path)
    print(f"  Checkpoint saved: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[GradScaler] = None,
) -> Tuple[int, int]:
    """Load training checkpoint. Returns (start_epoch, global_step)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    print(f"  Resumed from epoch {epoch}, step {global_step}")
    return epoch, global_step


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train StreamMapNet")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint for resuming training",
    )
    parser.add_argument(
        "--work_dir", type=str, default="work_dirs/stream_mapnet",
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--data_root", type=str, default="data/nuscenes",
        help="Root directory of the dataset",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Set seed
    set_seed(args.seed + rank)

    # Create work directory
    work_dir = Path(args.work_dir)
    if is_main_process():
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "checkpoints").mkdir(exist_ok=True)

    # Logging
    writer = None
    if is_main_process():
        writer = SummaryWriter(log_dir=str(work_dir / "tensorboard"))
        print(f"StreamMapNet Training")
        print(f"  Config: {args.config}")
        print(f"  Work dir: {work_dir}")
        print(f"  World size: {world_size}")
        print(f"  Device: {device}")

    # Build model
    model = build_stream_mapnet(config).to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config.get("training", {}).get("dist", {}).get(
                "find_unused_parameters", True
            ),
        )

    # Build dataset and dataloader
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    temporal_cfg = data_cfg.get("temporal", {})

    train_dataset = StreamMapNetDataset(
        data_root=args.data_root,
        split="train",
        sequence_length=temporal_cfg.get("window_size", 8),
        img_size=tuple(data_cfg.get("img_size", [256, 704])),
        num_cameras=data_cfg.get("num_cameras", 6),
        num_classes=data_cfg.get("num_classes", 3),
        num_points=config.get("model", {}).get("map_decoder", {}).get("num_points_per_query", 20),
        augment=True,
    )

    sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get("batch_size", 4),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=training_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_temporal_sequences,
        drop_last=True,
    )

    # Build optimizer with layer-wise learning rate
    base_lr = training_cfg.get("optimizer", {}).get("lr", 2e-4)
    weight_decay = training_cfg.get("optimizer", {}).get("weight_decay", 0.01)
    backbone_lr_mult = (
        training_cfg.get("optimizer", {}).get("paramwise_cfg", {}).get("backbone_lr_mult", 0.1)
    )

    # Separate backbone parameters for different learning rate
    backbone_params = []
    other_params = []
    model_for_params = model.module if hasattr(model, "module") else model
    for name, param in model_for_params.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
            {"params": other_params, "lr": base_lr},
        ],
        weight_decay=weight_decay,
        betas=tuple(training_cfg.get("optimizer", {}).get("betas", [0.9, 0.999])),
    )

    # Build scheduler
    num_epochs = training_cfg.get("epochs", 24)
    warmup_iters = training_cfg.get("scheduler", {}).get("warmup", {}).get("warmup_iters", 500)
    total_iters = num_epochs * len(train_loader)
    min_lr_ratio = training_cfg.get("scheduler", {}).get("min_lr_ratio", 0.01)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_iters, total_iters, min_lr_ratio
    )

    # Mixed precision
    use_fp16 = training_cfg.get("fp16", {}).get("enabled", True)
    scaler = GradScaler(enabled=use_fp16)

    # Loss function
    criterion = StreamMapNetLoss(config).to(device)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch += 1  # Start from next epoch

    # Training loop
    if is_main_process():
        print(f"\nStarting training from epoch {start_epoch}")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Batch size per GPU: {training_cfg.get('batch_size', 4)}")
        print(f"  Effective batch size: {training_cfg.get('batch_size', 4) * world_size}")
        print(f"  Sequence length: {temporal_cfg.get('window_size', 8)}")
        print(f"  Total iterations: {total_iters}")
        print(f"  Warmup iterations: {warmup_iters}")

    checkpoint_cfg = training_cfg.get("checkpoint", {})
    save_interval = checkpoint_cfg.get("interval", 1)
    max_keep = checkpoint_cfg.get("max_keep", 5)

    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

        epoch_loss, global_step = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            config=config,
            writer=writer,
            global_step=global_step,
        )

        if is_main_process():
            print(f"  Epoch {epoch + 1} complete. Average loss: {epoch_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                ckpt_path = str(work_dir / "checkpoints" / f"epoch_{epoch + 1}.pth")
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, epoch_loss, ckpt_path,
                )

                # Remove old checkpoints beyond max_keep
                ckpts = sorted(
                    (work_dir / "checkpoints").glob("epoch_*.pth"),
                    key=lambda p: p.stat().st_mtime,
                )
                while len(ckpts) > max_keep:
                    oldest = ckpts.pop(0)
                    oldest.unlink()
                    print(f"  Removed old checkpoint: {oldest.name}")

            # Save latest
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, global_step, epoch_loss,
                str(work_dir / "checkpoints" / "latest.pth"),
            )

    # Cleanup
    if writer is not None:
        writer.close()
    cleanup_distributed()

    if is_main_process():
        print(f"\nTraining complete. Final loss: {epoch_loss:.4f}")
        print(f"Checkpoints saved to: {work_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
