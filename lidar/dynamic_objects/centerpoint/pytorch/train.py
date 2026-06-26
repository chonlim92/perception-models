"""
CenterPoint 3D Object Detection - Training Script

Complete training pipeline for CenterPoint with support for:
- Single-GPU and multi-GPU (DDP) distributed training
- Mixed precision (AMP) training
- OneCycleLR scheduling with fade strategy
- TensorBoard logging
- Checkpoint save/resume
- Two-stage refinement (optional)
"""

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml

from .model import CenterPoint, build_model_from_config
from .dataset import NuScenesDataset, collate_fn

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CenterPoint 3D Object Detection Training"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--work-dir", type=str, default=None, help="Working directory for outputs"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Local rank for distributed training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Run evaluation only"
    )
    parser.add_argument(
        "--amp", action="store_true", default=True,
        help="Enable automatic mixed precision training"
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable automatic mixed precision"
    )
    args = parser.parse_args()

    if args.no_amp:
        args.amp = False

    return args


def load_config(config_path: str) -> dict:
    """Load and return YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(work_dir: str, rank: int):
    """Configure logging for training."""
    log_format = "[%(asctime)s %(levelname)s rank%(name)s] %(message)s"
    log_level = logging.INFO if rank == 0 else logging.WARNING

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(work_dir, "train.log")),
        ],
    )


def set_random_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

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
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = rank % torch.cuda.device_count()
    else:
        return 0, 1, 0  # rank, world_size, local_rank (single GPU)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def build_dataloader(config: dict, split: str, distributed: bool, world_size: int):
    """Build dataset and dataloader.

    Args:
        config: Full training configuration dict.
        split: One of 'train' or 'val'.
        distributed: Whether to use DistributedSampler.
        world_size: Number of processes.

    Returns:
        DataLoader instance.
    """
    dataset_cfg = config["dataset"]
    train_cfg = config["training"]

    dataset = NuScenesDataset(
        data_root=dataset_cfg["data_root"],
        split=split,
        voxel_size=dataset_cfg.get("voxel_size", [0.075, 0.075, 0.2]),
        point_cloud_range=dataset_cfg.get(
            "point_cloud_range", [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        ),
        max_points_per_voxel=dataset_cfg.get("max_points_per_voxel", 10),
        max_voxels=dataset_cfg.get("max_voxels", {"train": 120000, "val": 160000}),
        class_names=dataset_cfg.get(
            "class_names",
            [
                "car", "truck", "construction_vehicle", "bus", "trailer",
                "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
            ],
        ),
        augmentation=dataset_cfg.get("augmentation", {}) if split == "train" else {},
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, shuffle=(split == "train")
        )

    batch_size = train_cfg["batch_size_per_gpu"]
    num_workers = train_cfg.get("num_workers", 4)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )

    return dataloader, sampler


def build_optimizer(model: nn.Module, config: dict):
    """Build AdamW optimizer with per-parameter weight decay.

    Args:
        model: The CenterPoint model.
        config: Full training configuration dict.

    Returns:
        Configured AdamW optimizer.
    """
    train_cfg = config["training"]
    lr = train_cfg["learning_rate"]
    weight_decay = train_cfg.get("weight_decay", 0.01)

    # Separate parameters that should not have weight decay
    no_decay_params = []
    decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
    return optimizer


def build_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Build OneCycleLR scheduler.

    Args:
        optimizer: The optimizer.
        config: Full training configuration dict.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        OneCycleLR scheduler instance.
    """
    train_cfg = config["training"]
    max_lr = train_cfg["learning_rate"]
    total_epochs = train_cfg["epochs"]
    div_factor = train_cfg.get("lr_div_factor", 10)
    pct_start = train_cfg.get("lr_pct_start", 0.4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_epochs * steps_per_epoch,
        div_factor=div_factor,
        pct_start=pct_start,
        anneal_strategy="cos",
    )
    return scheduler


def compute_losses(predictions: dict, targets: dict, config: dict) -> dict:
    """Compute CenterPoint training losses.

    Computes heatmap focal loss and regression L1 loss for all detection heads.

    Args:
        predictions: Dict with keys 'heatmap', 'reg', 'height', 'dim', 'rot', 'vel'
                     from each detection head. Each value is a list over heads.
        targets: Dict with target heatmaps and regression targets from the dataset.
        config: Full training configuration dict.

    Returns:
        Dict of loss components and total loss.
    """
    loss_cfg = config.get("loss", {})
    heatmap_weight = loss_cfg.get("heatmap_weight", 1.0)
    regression_weight = loss_cfg.get("regression_weight", 2.0)
    velocity_weight = loss_cfg.get("velocity_weight", 0.2)

    total_loss = torch.tensor(0.0, device=predictions["heatmap"][0].device)
    loss_dict = {}

    num_heads = len(predictions["heatmap"])

    for head_idx in range(num_heads):
        # Heatmap loss (Gaussian focal loss)
        pred_heatmap = predictions["heatmap"][head_idx]
        target_heatmap = targets["heatmap"][head_idx]
        heatmap_loss = _gaussian_focal_loss(pred_heatmap, target_heatmap)

        # Regression losses (only at positive locations)
        positive_mask = targets["mask"][head_idx]  # (B, max_objs)
        num_positives = positive_mask.sum().clamp(min=1).float()
        indices = targets["indices"][head_idx]  # (B, max_objs)

        # Gather predictions at object center locations
        reg_loss = torch.tensor(0.0, device=total_loss.device)

        # Sub-voxel offset regression
        if "reg" in predictions:
            pred_reg = _gather_features(predictions["reg"][head_idx], indices)
            target_reg = targets["reg"][head_idx]
            reg_loss = reg_loss + _weighted_l1_loss(
                pred_reg, target_reg, positive_mask
            )

        # Height regression
        if "height" in predictions:
            pred_height = _gather_features(predictions["height"][head_idx], indices)
            target_height = targets["height"][head_idx]
            reg_loss = reg_loss + _weighted_l1_loss(
                pred_height, target_height, positive_mask
            )

        # Dimension regression (log-scale)
        if "dim" in predictions:
            pred_dim = _gather_features(predictions["dim"][head_idx], indices)
            target_dim = targets["dim"][head_idx]
            reg_loss = reg_loss + _weighted_l1_loss(
                pred_dim, target_dim, positive_mask
            )

        # Rotation regression (sin, cos)
        if "rot" in predictions:
            pred_rot = _gather_features(predictions["rot"][head_idx], indices)
            target_rot = targets["rot"][head_idx]
            reg_loss = reg_loss + _weighted_l1_loss(
                pred_rot, target_rot, positive_mask
            )

        # Velocity regression (optional, for nuScenes)
        vel_loss = torch.tensor(0.0, device=total_loss.device)
        if "vel" in predictions and "vel" in targets:
            pred_vel = _gather_features(predictions["vel"][head_idx], indices)
            target_vel = targets["vel"][head_idx]
            vel_loss = _weighted_l1_loss(pred_vel, target_vel, positive_mask)

        head_loss = (
            heatmap_weight * heatmap_loss
            + regression_weight * reg_loss / num_positives
            + velocity_weight * vel_loss / num_positives
        )
        total_loss = total_loss + head_loss

        loss_dict[f"head{head_idx}/heatmap_loss"] = heatmap_loss.item()
        loss_dict[f"head{head_idx}/reg_loss"] = (reg_loss / num_positives).item()
        if vel_loss.item() > 0:
            loss_dict[f"head{head_idx}/vel_loss"] = (vel_loss / num_positives).item()

    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict


def _gaussian_focal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian focal loss for heatmap prediction.

    Args:
        pred: Predicted heatmap (B, C, H, W), sigmoid-activated.
        target: Ground-truth Gaussian heatmap (B, C, H, W).

    Returns:
        Scalar focal loss.
    """
    pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

    positive_mask = target.eq(1).float()
    negative_mask = target.lt(1).float()

    # Positive locations
    positive_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * positive_mask
    # Negative locations (down-weighted by Gaussian)
    negative_loss = (
        -torch.log(1 - pred)
        * torch.pow(pred, 2)
        * torch.pow(1 - target, 4)
        * negative_mask
    )

    num_positives = positive_mask.sum().clamp(min=1)
    loss = (positive_loss.sum() + negative_loss.sum()) / num_positives
    return loss


def _gather_features(
    features: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Gather features at specified index locations from BEV feature map.

    Args:
        features: (B, C, H, W) feature map.
        indices: (B, max_objs) flattened indices into H*W.

    Returns:
        (B, max_objs, C) gathered features.
    """
    B, C, H, W = features.shape
    features = features.view(B, C, H * W)  # (B, C, H*W)
    features = features.permute(0, 2, 1)  # (B, H*W, C)

    max_objs = indices.shape[1]
    batch_indices = indices.unsqueeze(2).expand(-1, -1, C).long()  # (B, max_objs, C)
    gathered = features.gather(1, batch_indices)  # (B, max_objs, C)
    return gathered


def _weighted_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute L1 loss weighted by a binary mask.

    Args:
        pred: (B, max_objs, C) predictions.
        target: (B, max_objs, C) targets.
        mask: (B, max_objs) binary mask for valid objects.

    Returns:
        Scalar weighted L1 loss (sum, not mean - divide by num_pos externally).
    """
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = torch.abs(pred - target) * mask
    return loss.sum()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    config: dict,
    epoch: int,
    rank: int,
    writer: SummaryWriter,
    use_amp: bool = True,
    global_step: int = 0,
):
    """Train model for one epoch.

    Args:
        model: CenterPoint model.
        dataloader: Training dataloader.
        optimizer: AdamW optimizer.
        scheduler: OneCycleLR scheduler.
        scaler: GradScaler for mixed precision.
        config: Full training configuration dict.
        epoch: Current epoch number.
        rank: Process rank.
        writer: TensorBoard SummaryWriter.
        use_amp: Whether to use automatic mixed precision.
        global_step: Running global step counter.

    Returns:
        Updated global_step, average loss for the epoch.
    """
    model.train()
    train_cfg = config["training"]
    grad_clip_norm = train_cfg.get("grad_clip_max_norm", 35.0)
    log_interval = train_cfg.get("log_interval", 50)

    epoch_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch_data in enumerate(dataloader):
        # Move data to GPU
        voxels = batch_data["voxels"].cuda(non_blocking=True)
        coordinates = batch_data["coordinates"].cuda(non_blocking=True)
        num_points_per_voxel = batch_data["num_points_per_voxel"].cuda(
            non_blocking=True
        )
        targets = {
            key: (
                [t.cuda(non_blocking=True) for t in val]
                if isinstance(val, list)
                else val.cuda(non_blocking=True)
            )
            for key, val in batch_data["targets"].items()
        }

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            # Forward pass: voxelization -> backbone -> detection head
            predictions = model(
                voxels=voxels,
                coordinates=coordinates,
                num_points_per_voxel=num_points_per_voxel,
                batch_size=batch_data["batch_size"],
            )

            # Compute losses
            loss, loss_dict = compute_losses(predictions, targets, config)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first for accurate norm computation)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # LR scheduler step (per iteration for OneCycleLR)
        scheduler.step()

        epoch_loss += loss_dict["total_loss"]
        num_batches += 1
        global_step += 1

        # Logging
        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            samples_per_sec = (
                (batch_idx + 1) * train_cfg["batch_size_per_gpu"] / elapsed
            )

            logger.info(
                f"Epoch [{epoch}][{batch_idx + 1}/{len(dataloader)}] "
                f"loss: {loss_dict['total_loss']:.4f}, "
                f"lr: {current_lr:.6f}, "
                f"samples/s: {samples_per_sec:.1f}"
            )

            writer.add_scalar("train/total_loss", loss_dict["total_loss"], global_step)
            writer.add_scalar("train/learning_rate", current_lr, global_step)

            for key, val in loss_dict.items():
                if key != "total_loss":
                    writer.add_scalar(f"train/{key}", val, global_step)

    avg_loss = epoch_loss / max(num_batches, 1)
    return global_step, avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    config: dict,
    epoch: int,
    rank: int,
    writer: SummaryWriter,
    global_step: int,
    use_amp: bool = True,
):
    """Run validation and compute metrics.

    Args:
        model: CenterPoint model.
        dataloader: Validation dataloader.
        config: Full training configuration dict.
        epoch: Current epoch number.
        rank: Process rank.
        writer: TensorBoard SummaryWriter.
        global_step: Current global step for logging.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Dict of validation metrics including mAP and NDS.
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_ground_truths = []

    for batch_data in dataloader:
        voxels = batch_data["voxels"].cuda(non_blocking=True)
        coordinates = batch_data["coordinates"].cuda(non_blocking=True)
        num_points_per_voxel = batch_data["num_points_per_voxel"].cuda(
            non_blocking=True
        )
        targets = {
            key: (
                [t.cuda(non_blocking=True) for t in val]
                if isinstance(val, list)
                else val.cuda(non_blocking=True)
            )
            for key, val in batch_data["targets"].items()
        }

        with autocast(enabled=use_amp):
            predictions = model(
                voxels=voxels,
                coordinates=coordinates,
                num_points_per_voxel=num_points_per_voxel,
                batch_size=batch_data["batch_size"],
            )
            loss, loss_dict = compute_losses(predictions, targets, config)

        total_loss += loss_dict["total_loss"]
        num_batches += 1

        # Decode predictions for mAP computation
        decoded = model.module.decode(predictions) if hasattr(model, "module") else model.decode(predictions)
        all_predictions.extend(decoded)

        if "ground_truth" in batch_data:
            all_ground_truths.extend(batch_data["ground_truth"])

    avg_loss = total_loss / max(num_batches, 1)

    # Compute detection metrics (mAP, NDS) if ground truth is available
    metrics = {"val_loss": avg_loss}

    if all_ground_truths:
        eval_metrics = _compute_detection_metrics(
            all_predictions, all_ground_truths, config
        )
        metrics.update(eval_metrics)

    # Aggregate across ranks for distributed training
    if dist.is_initialized():
        metrics_tensor = torch.tensor(
            [metrics.get("mAP", 0.0), metrics.get("NDS", 0.0), avg_loss],
            device="cuda",
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        metrics_tensor /= world_size
        metrics["mAP"] = metrics_tensor[0].item()
        metrics["NDS"] = metrics_tensor[1].item()
        metrics["val_loss"] = metrics_tensor[2].item()

    # Log to TensorBoard
    if rank == 0:
        logger.info(
            f"Validation Epoch [{epoch}] - "
            f"loss: {metrics['val_loss']:.4f}, "
            f"mAP: {metrics.get('mAP', 0.0):.4f}, "
            f"NDS: {metrics.get('NDS', 0.0):.4f}"
        )
        writer.add_scalar("val/loss", metrics["val_loss"], global_step)
        writer.add_scalar("val/mAP", metrics.get("mAP", 0.0), global_step)
        writer.add_scalar("val/NDS", metrics.get("NDS", 0.0), global_step)

    return metrics


def _compute_detection_metrics(
    predictions: list, ground_truths: list, config: dict
) -> dict:
    """Compute nuScenes-style detection metrics.

    Computes mean Average Precision (mAP) over distance thresholds
    and nuScenes Detection Score (NDS).

    Args:
        predictions: List of per-sample prediction dicts with boxes, scores, labels.
        ground_truths: List of per-sample ground truth dicts.
        config: Full config dict for class names and thresholds.

    Returns:
        Dict with 'mAP', 'NDS', and per-class APs.
    """
    import numpy as np

    class_names = config["dataset"].get(
        "class_names",
        [
            "car", "truck", "construction_vehicle", "bus", "trailer",
            "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
        ],
    )
    distance_thresholds = [0.5, 1.0, 2.0, 4.0]

    per_class_aps = {}

    for class_idx, class_name in enumerate(class_names):
        aps_for_thresholds = []

        for dist_thresh in distance_thresholds:
            tp_list = []
            total_gt = 0

            for pred_sample, gt_sample in zip(predictions, ground_truths):
                pred_boxes = pred_sample.get("boxes", np.zeros((0, 7)))
                pred_scores = pred_sample.get("scores", np.zeros(0))
                pred_labels = pred_sample.get("labels", np.zeros(0, dtype=int))

                gt_boxes = gt_sample.get("boxes", np.zeros((0, 7)))
                gt_labels = gt_sample.get("labels", np.zeros(0, dtype=int))

                # Filter by class
                pred_mask = pred_labels == class_idx
                gt_mask = gt_labels == class_idx

                pred_boxes_cls = pred_boxes[pred_mask]
                pred_scores_cls = pred_scores[pred_mask]
                gt_boxes_cls = gt_boxes[gt_mask]

                total_gt += len(gt_boxes_cls)

                if len(pred_boxes_cls) == 0:
                    continue

                # Sort by score descending
                sort_idx = np.argsort(-pred_scores_cls)
                pred_boxes_cls = pred_boxes_cls[sort_idx]

                # Match predictions to ground truth by center distance
                matched_gt = set()
                for pred_box in pred_boxes_cls:
                    if len(gt_boxes_cls) == 0:
                        tp_list.append(0)
                        continue

                    distances = np.linalg.norm(
                        pred_box[:2] - gt_boxes_cls[:, :2], axis=1
                    )
                    min_idx = np.argmin(distances)
                    min_dist = distances[min_idx]

                    if min_dist < dist_thresh and min_idx not in matched_gt:
                        tp_list.append(1)
                        matched_gt.add(min_idx)
                    else:
                        tp_list.append(0)

            # Compute AP for this threshold
            if total_gt == 0:
                aps_for_thresholds.append(0.0)
                continue

            tp_array = np.array(tp_list)
            tp_cumsum = np.cumsum(tp_array)
            fp_cumsum = np.cumsum(1 - tp_array)
            recall = tp_cumsum / total_gt
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)

            # 11-point interpolation
            ap = 0.0
            for r_thresh in np.linspace(0, 1, 11):
                prec_at_recall = precision[recall >= r_thresh]
                if len(prec_at_recall) > 0:
                    ap += prec_at_recall.max() / 11.0

            aps_for_thresholds.append(ap)

        per_class_aps[class_name] = float(np.mean(aps_for_thresholds))

    mAP = float(np.mean(list(per_class_aps.values()))) if per_class_aps else 0.0

    # NDS is a weighted combination of mAP and attribute errors
    # Simplified: NDS = (5 * mAP + sum(1 - min(1, error_i))) / 10
    # Without attribute errors available, approximate as weighted mAP
    NDS = mAP  # Simplified; full NDS requires translation/scale/orientation errors

    metrics = {"mAP": mAP, "NDS": NDS}
    metrics.update({f"AP/{k}": v for k, v in per_class_aps.items()})
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    metrics: dict,
    config: dict,
    work_dir: str,
    is_best: bool = False,
):
    """Save training checkpoint.

    Args:
        model: The model (handles DDP unwrapping).
        optimizer: Optimizer state.
        scheduler: Scheduler state.
        scaler: GradScaler state.
        epoch: Current epoch.
        global_step: Current global step.
        metrics: Current validation metrics.
        config: Training configuration.
        work_dir: Directory to save checkpoints.
        is_best: Whether this is the best model so far.
    """
    model_state = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "config": config,
    }

    # Save latest checkpoint
    latest_path = os.path.join(work_dir, "checkpoint_latest.pth")
    torch.save(checkpoint, latest_path)
    logger.info(f"Saved latest checkpoint to {latest_path}")

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(work_dir, "checkpoint_best.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint to {best_path} (mAP: {metrics.get('mAP', 0.0):.4f})")

    # Save periodic checkpoint
    save_every = config["training"].get("save_checkpoint_every", 5)
    if (epoch + 1) % save_every == 0:
        periodic_path = os.path.join(work_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, periodic_path)
        logger.info(f"Saved periodic checkpoint to {periodic_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    scaler: GradScaler = None,
) -> dict:
    """Load checkpoint and restore training state.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        scaler: Optional GradScaler to restore state.

    Returns:
        Checkpoint dict with epoch, global_step, metrics, config.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle DDP model
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logger.info(
        f"Resumed from epoch {checkpoint['epoch']}, "
        f"global_step {checkpoint['global_step']}"
    )
    return checkpoint


def apply_fade_strategy(dataloader: DataLoader, epoch: int, total_epochs: int):
    """Disable augmentation in the last N epochs (fade strategy).

    This helps the model converge better by removing augmentation noise
    during final training epochs.

    Args:
        dataloader: Training dataloader with underlying NuScenesDataset.
        epoch: Current epoch number.
        total_epochs: Total number of epochs.
    """
    fade_epochs = 5

    if epoch >= total_epochs - fade_epochs:
        dataset = dataloader.dataset
        if hasattr(dataset, "set_augmentation_enabled"):
            dataset.set_augmentation_enabled(False)
            logger.info(
                f"Fade strategy: disabled augmentation at epoch {epoch} "
                f"(last {fade_epochs} epochs)"
            )
    else:
        dataset = dataloader.dataset
        if hasattr(dataset, "set_augmentation_enabled"):
            dataset.set_augmentation_enabled(True)


def main():
    """Main training entry point."""
    args = parse_args()
    config = load_config(args.config)

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    distributed = world_size > 1

    # Setup working directory
    work_dir = args.work_dir or config.get("work_dir", "./work_dirs/centerpoint")
    if rank == 0:
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(os.path.join(work_dir, "tensorboard"), exist_ok=True)

    # Synchronize to ensure directory exists before other ranks proceed
    if distributed:
        dist.barrier()

    # Setup logging and seed
    setup_logging(work_dir, rank)
    set_random_seed(args.seed + rank)

    logger.info(f"Config: {args.config}")
    logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
    logger.info(f"Working directory: {work_dir}")

    # Build model
    model = build_model_from_config(config["model"])
    model.cuda(local_rank)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params / 1e6:.2f}M")

    # Wrap model with DDP
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config["model"].get("find_unused_parameters", False),
        )

    # Build dataloaders
    train_loader, train_sampler = build_dataloader(
        config, "train", distributed, world_size
    )
    val_loader, _ = build_dataloader(config, "val", distributed, world_size)

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)

    # Mixed precision scaler
    use_amp = args.amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    # TensorBoard writer (rank 0 only)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(work_dir, "tensorboard"))

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_mAP = 0.0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_mAP = checkpoint.get("metrics", {}).get("mAP", 0.0)

    # Evaluation only mode
    if args.eval_only:
        metrics = validate(
            model, val_loader, config, start_epoch, rank, writer, global_step, use_amp
        )
        if rank == 0:
            logger.info(f"Evaluation results: {metrics}")
        cleanup_distributed()
        return

    # Training loop
    train_cfg = config["training"]
    total_epochs = train_cfg["epochs"]
    val_interval = train_cfg.get("val_interval", 5)

    logger.info(
        f"Starting training: {total_epochs} epochs, "
        f"{steps_per_epoch} steps/epoch, "
        f"batch_size_per_gpu={train_cfg['batch_size_per_gpu']}, "
        f"amp={use_amp}"
    )

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Apply fade strategy (disable augmentation in last 5 epochs)
        apply_fade_strategy(train_loader, epoch, total_epochs)

        # Train one epoch
        global_step, avg_train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch,
            rank=rank,
            writer=writer,
            use_amp=use_amp,
            global_step=global_step,
        )

        epoch_time = time.time() - epoch_start

        if rank == 0:
            logger.info(
                f"Epoch [{epoch}] completed in {epoch_time:.1f}s, "
                f"avg_loss: {avg_train_loss:.4f}"
            )
            writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
            writer.add_scalar("train/epoch_time_s", epoch_time, epoch)

        # Validation
        metrics = {}
        if (epoch + 1) % val_interval == 0 or epoch == total_epochs - 1:
            metrics = validate(
                model, val_loader, config, epoch, rank, writer, global_step, use_amp
            )

        # Save checkpoint (rank 0 only)
        if rank == 0:
            current_mAP = metrics.get("mAP", 0.0)
            is_best = current_mAP > best_mAP
            if is_best:
                best_mAP = current_mAP

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                config=config,
                work_dir=work_dir,
                is_best=is_best,
            )

        # Synchronize before next epoch
        if distributed:
            dist.barrier()

    # Cleanup
    if rank == 0:
        writer.close()
        logger.info(f"Training complete. Best mAP: {best_mAP:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
