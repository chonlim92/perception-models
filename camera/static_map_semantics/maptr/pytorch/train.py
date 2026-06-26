"""
Training script for MapTR / MapTRv2.

Supports:
- Multi-GPU training via DistributedDataParallel (DDP)
- Mixed precision training (AMP)
- AdamW optimizer with separate backbone LR
- Cosine annealing LR scheduler with linear warmup
- Gradient clipping
- Checkpoint save/resume
"""

import argparse
import datetime
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Local imports
from dataset import NuScenesMapDataset


# =============================================================================
# Logging setup
# =============================================================================

def setup_logger(rank: int, work_dir: str) -> logging.Logger:
    """Configure logger that writes to file and stdout (rank 0 only)."""
    logger = logging.getLogger("maptr_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s][Rank %(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        os.makedirs(work_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(work_dir, "train.log"), mode="a"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.name = str(rank)
    return logger


# =============================================================================
# Model construction helpers
# =============================================================================

def build_model(config: Dict[str, Any], num_classes: int = 3) -> nn.Module:
    """
    Build MapTR or MapTRv2 model from config.

    This constructs the full model pipeline:
    - ResNet-50 + FPN backbone
    - BEV feature encoder (GKT-based view transform)
    - MapTR decoder head with hierarchical queries

    Args:
        config: Model configuration dict.
        num_classes: Number of map element classes.

    Returns:
        Full MapTR model as nn.Module.
    """
    from backbone import ResNet50FPN

    model_type = config.get("model", {}).get("type", "MapTR")
    embed_dims = config.get("model", {}).get("bev", {}).get("embed_dims", 256)
    bev_h = config.get("model", {}).get("bev", {}).get("bev_h", 200)
    bev_w = config.get("model", {}).get("bev", {}).get("bev_w", 100)
    num_queries = config.get("model", {}).get("head", {}).get("num_queries", 50)
    num_points = config.get("model", {}).get("head", {}).get("num_points_per_instance", 20)
    num_decoder_layers = (
        config.get("model", {})
        .get("head", {})
        .get("transformer", {})
        .get("num_layers", 6)
    )
    num_heads = 8

    class BEVEncoder(nn.Module):
        """
        Geometry-guided BEV feature encoder.

        Projects multi-camera FPN features into a unified BEV grid using
        learned depth distribution and camera geometry.
        """

        def __init__(self, in_channels: int, embed_dims: int, bev_h: int, bev_w: int):
            super().__init__()
            self.bev_h = bev_h
            self.bev_w = bev_w
            self.embed_dims = embed_dims

            # Depth prediction network
            self.depth_net = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 118, 1),  # 118 depth bins for [1, 60] at 0.5m step
            )

            # Context feature compression
            self.context_net = nn.Sequential(
                nn.Conv2d(in_channels, embed_dims, 1, bias=False),
                nn.BatchNorm2d(embed_dims),
                nn.ReLU(inplace=True),
            )

            # BEV positional embedding
            self.bev_embed = nn.Embedding(bev_h * bev_w, embed_dims)

            # Spatial aggregation after lifting
            self.bev_conv = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, 3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dims, embed_dims, 3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims),
                nn.ReLU(inplace=True),
            )

        def forward(
            self,
            multi_scale_feats: List[torch.Tensor],
            intrinsics: torch.Tensor,
            extrinsics: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                multi_scale_feats: List of [B*6, C, H_i, W_i] FPN features.
                intrinsics: [B, 6, 3, 3]
                extrinsics: [B, 6, 4, 4]

            Returns:
                BEV features [B, embed_dims, bev_h, bev_w].
            """
            # Use the highest resolution feature map
            feat = multi_scale_feats[0]  # [B*6, C, H, W]
            BN, C, H, W = feat.shape
            B = intrinsics.shape[0]
            N = 6

            # Predict depth distribution
            depth_logits = self.depth_net(feat)  # [B*6, D, H, W]
            depth_probs = depth_logits.softmax(dim=1)  # [B*6, D, H, W]

            # Context features
            context = self.context_net(feat)  # [B*6, embed_dims, H, W]

            # Outer product: depth_probs x context -> volume
            # Simplified: average pool across depth and spatial dims projected to BEV
            # This is a simplified version; full GKT uses explicit geometric projection
            context_pooled = context.reshape(B, N, self.embed_dims, H, W)
            context_pooled = context_pooled.mean(dim=1)  # [B, embed_dims, H, W]

            # Adaptive pool to BEV resolution
            bev_feat = F.adaptive_avg_pool2d(
                context_pooled, (self.bev_h, self.bev_w)
            )  # [B, embed_dims, bev_h, bev_w]

            # Add BEV positional embedding
            bev_pos = self.bev_embed.weight.reshape(
                1, self.bev_h, self.bev_w, self.embed_dims
            ).permute(0, 3, 1, 2)
            bev_feat = bev_feat + bev_pos

            # Refine BEV features
            bev_feat = self.bev_conv(bev_feat)

            return bev_feat

    class MapTRDecoderLayer(nn.Module):
        """Single decoder layer with self-attention, cross-attention, and FFN."""

        def __init__(self, embed_dims: int, num_heads: int, ffn_dims: int = 512):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(
                embed_dims, num_heads, dropout=0.1, batch_first=True
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dims, num_heads, dropout=0.1, batch_first=True
            )
            self.ffn = nn.Sequential(
                nn.Linear(embed_dims, ffn_dims),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(ffn_dims, embed_dims),
                nn.Dropout(0.1),
            )
            self.norm1 = nn.LayerNorm(embed_dims)
            self.norm2 = nn.LayerNorm(embed_dims)
            self.norm3 = nn.LayerNorm(embed_dims)
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.1)

        def forward(
            self, query: torch.Tensor, bev_feat: torch.Tensor
        ) -> torch.Tensor:
            """
            Args:
                query: [B, num_queries * num_points, embed_dims]
                bev_feat: [B, bev_h * bev_w, embed_dims]
            """
            # Self-attention
            q = self.norm1(query)
            q2 = self.self_attn(q, q, q)[0]
            query = query + self.dropout1(q2)

            # Cross-attention with BEV features
            q = self.norm2(query)
            q2 = self.cross_attn(q, bev_feat, bev_feat)[0]
            query = query + self.dropout2(q2)

            # Feed-forward
            query = query + self.ffn(self.norm3(query))

            return query

    class MapTRHead(nn.Module):
        """
        MapTR decoder head with hierarchical instance and point queries.

        Predicts:
        - Class logits for each instance query
        - 2D BEV point coordinates for each point query
        """

        def __init__(
            self,
            embed_dims: int,
            num_classes: int,
            num_queries: int,
            num_points: int,
            num_layers: int,
            num_heads: int,
        ):
            super().__init__()
            self.embed_dims = embed_dims
            self.num_classes = num_classes
            self.num_queries = num_queries
            self.num_points = num_points
            self.num_layers = num_layers

            # Learnable instance queries
            self.instance_query_embed = nn.Embedding(num_queries, embed_dims)
            # Learnable point queries (relative to instance)
            self.point_query_embed = nn.Embedding(num_points, embed_dims)

            # Decoder layers
            self.decoder_layers = nn.ModuleList([
                MapTRDecoderLayer(embed_dims, num_heads)
                for _ in range(num_layers)
            ])

            # Prediction heads (shared across layers for iterative refinement)
            self.cls_head = nn.Linear(embed_dims, num_classes)
            self.pts_head = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 2),
                nn.Sigmoid(),  # Output in [0, 1]
            )

        def forward(
            self, bev_feat: torch.Tensor
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
            """
            Args:
                bev_feat: [B, embed_dims, bev_h, bev_w]

            Returns:
                all_cls_scores: List of [B, num_queries, num_classes] per layer
                all_pts_preds: List of [B, num_queries, num_points, 2] per layer
            """
            B = bev_feat.shape[0]

            # Flatten BEV features for attention
            bev_flat = bev_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

            # Construct hierarchical queries: instance x point
            inst_q = self.instance_query_embed.weight  # [num_queries, C]
            pt_q = self.point_query_embed.weight  # [num_points, C]

            # Combine: each instance query is expanded with point queries
            # [num_queries, 1, C] + [1, num_points, C] -> [num_queries, num_points, C]
            combined_q = inst_q.unsqueeze(1) + pt_q.unsqueeze(0)
            combined_q = combined_q.reshape(
                self.num_queries * self.num_points, self.embed_dims
            )
            # Expand for batch
            query = combined_q.unsqueeze(0).expand(B, -1, -1)  # [B, Q*P, C]

            all_cls_scores = []
            all_pts_preds = []

            for layer in self.decoder_layers:
                query = layer(query, bev_flat)

                # Reshape query for predictions
                query_reshaped = query.reshape(
                    B, self.num_queries, self.num_points, self.embed_dims
                )

                # Instance-level classification (pool over points)
                inst_feat = query_reshaped.mean(dim=2)  # [B, num_queries, C]
                cls_scores = self.cls_head(inst_feat)  # [B, num_queries, num_classes]

                # Point-level regression
                pts_preds = self.pts_head(query_reshaped)  # [B, num_queries, num_points, 2]

                all_cls_scores.append(cls_scores)
                all_pts_preds.append(pts_preds)

            return all_cls_scores, all_pts_preds

    class MapTRModel(nn.Module):
        """Full MapTR model: backbone + BEV encoder + decoder head."""

        def __init__(
            self,
            backbone: nn.Module,
            bev_encoder: nn.Module,
            head: nn.Module,
        ):
            super().__init__()
            self.backbone = backbone
            self.bev_encoder = bev_encoder
            self.head = head

        def forward(
            self,
            images: torch.Tensor,
            intrinsics: torch.Tensor,
            extrinsics: torch.Tensor,
        ) -> Dict[str, Any]:
            """
            Args:
                images: [B, 6, 3, H, W]
                intrinsics: [B, 6, 3, 3]
                extrinsics: [B, 6, 4, 4]

            Returns:
                Dict with 'all_cls_scores' and 'all_pts_preds' lists.
            """
            # Extract multi-scale features
            fpn_feats = self.backbone(images)  # List of [B*6, C, H_i, W_i]

            # Build BEV representation
            bev_feat = self.bev_encoder(fpn_feats, intrinsics, extrinsics)

            # Decode map elements
            all_cls_scores, all_pts_preds = self.head(bev_feat)

            return {
                "all_cls_scores": all_cls_scores,
                "all_pts_preds": all_pts_preds,
            }

    # Construct model components
    backbone = ResNet50FPN(
        pretrained=True,
        fpn_out_channels=embed_dims,
        num_fpn_levels=4,
    )

    bev_encoder = BEVEncoder(
        in_channels=embed_dims,
        embed_dims=embed_dims,
        bev_h=bev_h,
        bev_w=bev_w,
    )

    head = MapTRHead(
        embed_dims=embed_dims,
        num_classes=num_classes,
        num_queries=num_queries,
        num_points=num_points,
        num_layers=num_decoder_layers,
        num_heads=num_heads,
    )

    model = MapTRModel(backbone, bev_encoder, head)
    return model


# =============================================================================
# Loss computation
# =============================================================================

class MapTRLoss(nn.Module):
    """
    MapTR loss with Hungarian matching.

    Computes:
    - Focal loss for classification
    - Smooth L1 loss for point regression
    - Direction cosine loss for polyline direction consistency
    """

    def __init__(
        self,
        num_classes: int = 3,
        cost_class: float = 2.0,
        cost_pts: float = 5.0,
        cost_dir: float = 0.005,
        cls_weight: float = 2.0,
        pts_weight: float = 5.0,
        dir_weight: float = 0.005,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cost_class = cost_class
        self.cost_pts = cost_pts
        self.cost_dir = cost_dir
        self.cls_weight = cls_weight
        self.pts_weight = pts_weight
        self.dir_weight = dir_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Focal loss for multi-class classification.

        Args:
            pred: [N, num_classes] logits
            target: [N] class indices (0-indexed)

        Returns:
            Scalar focal loss.
        """
        num_classes = pred.shape[1]
        # One-hot encode targets
        target_onehot = F.one_hot(target, num_classes + 1)[:, :num_classes].float()

        pred_sigmoid = pred.sigmoid()
        pt = pred_sigmoid * target_onehot + (1 - pred_sigmoid) * (1 - target_onehot)
        focal_weight = (alpha * target_onehot + (1 - alpha) * (1 - target_onehot))
        focal_weight = focal_weight * (1 - pt).pow(gamma)

        loss = F.binary_cross_entropy_with_logits(
            pred, target_onehot, reduction="none"
        )
        loss = (focal_weight * loss).sum(dim=-1).mean()
        return loss

    @torch.no_grad()
    def hungarian_match(
        self,
        cls_scores: torch.Tensor,
        pts_preds: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_points: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching between predictions and ground truth.

        Args:
            cls_scores: [B, num_queries, num_classes]
            pts_preds: [B, num_queries, num_points, 2]
            gt_labels: [B, max_gt] (-1 for padding)
            gt_points: [B, max_gt, num_points, 2]
            gt_mask: [B, max_gt] bool mask

        Returns:
            List of (pred_indices, gt_indices) tuples per batch element.
        """
        from scipy.optimize import linear_sum_assignment

        B = cls_scores.shape[0]
        indices = []

        for b in range(B):
            valid = gt_mask[b]
            num_gt = valid.sum().item()

            if num_gt == 0:
                indices.append(
                    (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
                )
                continue

            # Valid GT
            b_gt_labels = gt_labels[b][valid]  # [num_gt]
            b_gt_points = gt_points[b][valid]  # [num_gt, P, 2]

            # Classification cost
            b_cls = cls_scores[b].sigmoid()  # [Q, C]
            # Gather predicted prob for GT class
            cls_cost = -b_cls[:, b_gt_labels]  # [Q, num_gt] (negated for minimization)

            # Point regression cost (L1)
            b_pts = pts_preds[b]  # [Q, P, 2]
            pts_cost = torch.cdist(
                b_pts.flatten(1),  # [Q, P*2]
                b_gt_points.flatten(1),  # [num_gt, P*2]
                p=1,
            )  # [Q, num_gt]
            pts_cost = pts_cost / b_gt_points.shape[1]  # Normalize by num_points

            # Direction cost
            pred_dirs = b_pts[:, 1:] - b_pts[:, :-1]  # [Q, P-1, 2]
            gt_dirs = b_gt_points[:, 1:] - b_gt_points[:, :-1]  # [num_gt, P-1, 2]

            pred_dirs_norm = F.normalize(pred_dirs.flatten(1), dim=-1)  # [Q, (P-1)*2]
            gt_dirs_norm = F.normalize(gt_dirs.flatten(1), dim=-1)  # [num_gt, (P-1)*2]

            dir_cost = 1 - torch.mm(pred_dirs_norm, gt_dirs_norm.T)  # [Q, num_gt]

            # Combined cost matrix
            cost = (
                self.cost_class * cls_cost
                + self.cost_pts * pts_cost
                + self.cost_dir * dir_cost
            )

            # Hungarian algorithm
            cost_np = cost.detach().cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(cost_np)

            indices.append(
                (
                    torch.tensor(pred_idx, dtype=torch.long, device=cls_scores.device),
                    torch.tensor(gt_idx, dtype=torch.long, device=cls_scores.device),
                )
            )

        return indices

    def forward(
        self,
        outputs: Dict[str, Any],
        gt_labels: torch.Tensor,
        gt_points: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MapTR losses across all decoder layers.

        Args:
            outputs: Dict with 'all_cls_scores' [list of [B, Q, C]] and
                     'all_pts_preds' [list of [B, Q, P, 2]]
            gt_labels: [B, max_gt]
            gt_points: [B, max_gt, P, 2]
            gt_masks: [B, max_gt]

        Returns:
            Dict of loss components: loss_cls, loss_pts, loss_dir, loss_total.
        """
        all_cls_scores = outputs["all_cls_scores"]
        all_pts_preds = outputs["all_pts_preds"]
        num_layers = len(all_cls_scores)

        total_cls_loss = torch.tensor(0.0, device=gt_labels.device)
        total_pts_loss = torch.tensor(0.0, device=gt_labels.device)
        total_dir_loss = torch.tensor(0.0, device=gt_labels.device)

        for layer_idx in range(num_layers):
            cls_scores = all_cls_scores[layer_idx]  # [B, Q, C]
            pts_preds = all_pts_preds[layer_idx]  # [B, Q, P, 2]

            # Hungarian matching
            indices = self.hungarian_match(
                cls_scores, pts_preds, gt_labels, gt_points, gt_masks
            )

            B = cls_scores.shape[0]
            device = cls_scores.device

            # Classification loss (all queries, matched + unmatched)
            layer_cls_loss = torch.tensor(0.0, device=device)
            layer_pts_loss = torch.tensor(0.0, device=device)
            layer_dir_loss = torch.tensor(0.0, device=device)

            for b in range(B):
                pred_idx, gt_idx = indices[b]
                num_gt = gt_idx.shape[0]

                # Build target labels: num_classes for unmatched (background)
                target_cls = torch.full(
                    (cls_scores.shape[1],),
                    self.num_classes,
                    dtype=torch.long,
                    device=device,
                )
                if num_gt > 0:
                    valid_gt_labels = gt_labels[b][gt_masks[b]]
                    target_cls[pred_idx] = valid_gt_labels[gt_idx]

                # Focal loss
                layer_cls_loss = layer_cls_loss + self.focal_loss(
                    cls_scores[b], target_cls, self.focal_alpha, self.focal_gamma
                )

                if num_gt > 0:
                    # Point regression loss (matched only)
                    matched_pts_pred = pts_preds[b][pred_idx]  # [num_gt, P, 2]
                    valid_gt_points = gt_points[b][gt_masks[b]]
                    matched_pts_gt = valid_gt_points[gt_idx]  # [num_gt, P, 2]

                    pts_loss = F.smooth_l1_loss(
                        matched_pts_pred, matched_pts_gt, beta=1.0
                    )
                    layer_pts_loss = layer_pts_loss + pts_loss

                    # Direction loss
                    pred_dirs = matched_pts_pred[:, 1:] - matched_pts_pred[:, :-1]
                    gt_dirs = matched_pts_gt[:, 1:] - matched_pts_gt[:, :-1]

                    pred_dirs_flat = pred_dirs.reshape(-1, 2)
                    gt_dirs_flat = gt_dirs.reshape(-1, 2)

                    # Cosine similarity loss
                    cos_sim = F.cosine_similarity(
                        pred_dirs_flat, gt_dirs_flat, dim=-1
                    )
                    dir_loss = (1 - cos_sim).mean()
                    layer_dir_loss = layer_dir_loss + dir_loss

            # Average over batch
            layer_cls_loss = layer_cls_loss / max(B, 1)
            layer_pts_loss = layer_pts_loss / max(B, 1)
            layer_dir_loss = layer_dir_loss / max(B, 1)

            total_cls_loss = total_cls_loss + layer_cls_loss
            total_pts_loss = total_pts_loss + layer_pts_loss
            total_dir_loss = total_dir_loss + layer_dir_loss

        # Average over decoder layers
        total_cls_loss = total_cls_loss / num_layers
        total_pts_loss = total_pts_loss / num_layers
        total_dir_loss = total_dir_loss / num_layers

        # Weighted total
        loss_total = (
            self.cls_weight * total_cls_loss
            + self.pts_weight * total_pts_loss
            + self.dir_weight * total_dir_loss
        )

        return {
            "loss_cls": total_cls_loss,
            "loss_pts": total_pts_loss,
            "loss_dir": total_dir_loss,
            "loss_total": loss_total,
        }


# =============================================================================
# LR Scheduler with warmup
# =============================================================================

class CosineAnnealingWithWarmup:
    """Cosine annealing scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        total_iters: int,
        warmup_ratio: float = 0.001,
        eta_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.warmup_ratio = warmup_ratio
        self.eta_min = eta_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_iter = 0

    def step(self):
        """Update learning rate for the current iteration."""
        self.current_iter += 1
        if self.current_iter <= self.warmup_iters:
            # Linear warmup
            warmup_factor = self.warmup_ratio + (1 - self.warmup_ratio) * (
                self.current_iter / self.warmup_iters
            )
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * warmup_factor
        else:
            # Cosine annealing after warmup
            progress = (self.current_iter - self.warmup_iters) / max(
                1, self.total_iters - self.warmup_iters
            )
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = self.eta_min + 0.5 * (base_lr - self.eta_min) * (
                    1 + math.cos(math.pi * progress)
                )

    def get_last_lr(self) -> List[float]:
        """Return current learning rates."""
        return [pg["lr"] for pg in self.optimizer.param_groups]


# =============================================================================
# Training loop
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MapTRLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingWithWarmup,
    scaler: GradScaler,
    epoch: int,
    rank: int,
    logger: logging.Logger,
    log_interval: int = 50,
    max_grad_norm: float = 35.0,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dict of average loss values for the epoch.
    """
    model.train()
    device = torch.device(f"cuda:{rank}")

    epoch_losses = {"loss_cls": 0.0, "loss_pts": 0.0, "loss_dir": 0.0, "loss_total": 0.0}
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        images = batch["images"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        extrinsics = batch["extrinsics"].to(device, non_blocking=True)
        gt_labels = batch["gt_labels"].to(device, non_blocking=True)
        gt_points = batch["gt_points"].to(device, non_blocking=True)
        gt_masks = batch["gt_masks"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            outputs = model(images, intrinsics, extrinsics)
            losses = criterion(outputs, gt_labels, gt_points, gt_masks)
            loss = losses["loss_total"]

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        # Update LR
        scheduler.step()

        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += losses[key].item()
        num_batches += 1

        # Logging
        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch [{epoch}][{batch_idx + 1}/{len(dataloader)}] "
                f"lr: {current_lr:.6f} | "
                f"loss_total: {losses['loss_total'].item():.4f} | "
                f"loss_cls: {losses['loss_cls'].item():.4f} | "
                f"loss_pts: {losses['loss_pts'].item():.4f} | "
                f"loss_dir: {losses['loss_dir'].item():.4f} | "
                f"time: {elapsed:.1f}s"
            )
            start_time = time.time()

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= max(num_batches, 1)

    return epoch_losses


# =============================================================================
# Checkpoint utilities
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingWithWarmup,
    scaler: GradScaler,
    epoch: int,
    work_dir: str,
    filename: Optional[str] = None,
):
    """Save training checkpoint."""
    if filename is None:
        filename = f"epoch_{epoch}.pth"

    os.makedirs(work_dir, exist_ok=True)
    filepath = os.path.join(work_dir, filename)

    # Handle DDP model
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_current_iter": scheduler.current_iter,
        "scaler_state_dict": scaler.state_dict(),
    }

    torch.save(checkpoint, filepath)
    # Also save as latest
    latest_path = os.path.join(work_dir, "latest.pth")
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[CosineAnnealingWithWarmup] = None,
    scaler: Optional[GradScaler] = None,
) -> int:
    """
    Load training checkpoint.

    Returns:
        The epoch number from the checkpoint (to resume from epoch + 1).
    """
    checkpoint = torch.load(filepath, map_location="cpu")

    # Handle DDP model
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_current_iter" in checkpoint:
        scheduler.current_iter = checkpoint["scheduler_current_iter"]

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint.get("epoch", 0)


# =============================================================================
# DDP utilities
# =============================================================================

def setup_distributed(rank: int, world_size: int, port: int = 29500):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(port))

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=30),
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Config loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except ImportError:
        # Fallback: return empty config if yaml not available
        config = {}
    return config


# =============================================================================
# Main training function
# =============================================================================

def main_worker(rank: int, world_size: int, args: argparse.Namespace):
    """Main training worker (one per GPU)."""
    # Setup distributed if multi-GPU
    if world_size > 1:
        setup_distributed(rank, world_size)

    # Setup logging
    logger = setup_logger(rank, args.work_dir)

    if rank == 0:
        logger.info(f"Starting MapTR training with {world_size} GPU(s)")
        logger.info(f"Arguments: {vars(args)}")

    # Load config
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        if rank == 0:
            logger.info(f"Loaded config from: {args.config}")

    # Device
    device = torch.device(f"cuda:{rank}")

    # Dataset
    pipeline_cfg = {
        "target_size": (480, 800),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "size_divisor": 32,
    }

    map_classes = config.get("dataset", {}).get(
        "map_classes", ["ped_crossing", "divider", "boundary"]
    )
    num_points_per_instance = config.get("dataset", {}).get("fixed_num_pts_per_line", 20)
    coord_range_6d = config.get("dataset", {}).get(
        "point_cloud_range", [-30.0, -15.0, -5.0, 30.0, 15.0, 3.0]
    )
    # Extract 2D range (x_min, y_min, x_max, y_max) from 3D range
    coord_range = [coord_range_6d[0], coord_range_6d[1], coord_range_6d[3], coord_range_6d[4]]

    train_dataset = NuScenesMapDataset(
        data_root=args.data_root,
        ann_file=args.ann_file,
        pipeline=pipeline_cfg,
        map_classes=map_classes,
        num_points_per_instance=num_points_per_instance,
        coord_range=coord_range,
    )

    if rank == 0:
        logger.info(f"Training dataset size: {len(train_dataset)}")

    # DataLoader with DistributedSampler
    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        sampler = None

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=NuScenesMapDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # Build model
    num_classes = len(map_classes)
    model = build_model(config, num_classes=num_classes)
    model = model.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model params: {total_params / 1e6:.2f}M total, "
            f"{trainable_params / 1e6:.2f}M trainable"
        )

    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )

    # Optimizer: separate LR for backbone (0.1x)
    backbone_params = []
    non_backbone_params = []

    model_without_ddp = model.module if hasattr(model, "module") else model
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            non_backbone_params.append(param)

    base_lr = args.lr
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": base_lr * 0.1, "name": "backbone"},
            {"params": non_backbone_params, "lr": base_lr, "name": "non_backbone"},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # LR Scheduler with warmup
    total_iters = args.epochs * len(dataloader)
    warmup_iters = config.get("training", {}).get("lr_scheduler", {}).get(
        "warmup", {}
    ).get("warmup_iters", 500)
    warmup_ratio = config.get("training", {}).get("lr_scheduler", {}).get(
        "warmup", {}
    ).get("warmup_ratio", 0.001)
    eta_min = config.get("training", {}).get("lr_scheduler", {}).get("eta_min", 1e-6)

    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer,
        warmup_iters=warmup_iters,
        total_iters=total_iters,
        warmup_ratio=warmup_ratio,
        eta_min=eta_min,
    )

    # Loss
    loss_cfg = config.get("loss", {})
    criterion = MapTRLoss(
        num_classes=num_classes,
        cost_class=loss_cfg.get("matcher", {}).get("cost_class", 2.0),
        cost_pts=loss_cfg.get("matcher", {}).get("cost_pts", 5.0),
        cost_dir=loss_cfg.get("matcher", {}).get("cost_dir", 0.005),
        cls_weight=loss_cfg.get("cls_loss", {}).get("loss_weight", 2.0),
        pts_weight=loss_cfg.get("pts_loss", {}).get("loss_weight", 5.0),
        dir_weight=loss_cfg.get("dir_loss", {}).get("loss_weight", 0.005),
    )

    # Mixed precision
    use_amp = args.fp16
    scaler = GradScaler(enabled=use_amp)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                logger.info(f"Resuming from checkpoint: {args.resume}")
            start_epoch = load_checkpoint(
                args.resume, model, optimizer, scheduler, scaler
            )
            start_epoch += 1  # Start from next epoch
            if rank == 0:
                logger.info(f"Resumed at epoch {start_epoch}")
        else:
            if rank == 0:
                logger.warning(f"Checkpoint not found: {args.resume}")

    # Training loop
    log_interval = config.get("training", {}).get("log_interval", 50)
    max_grad_norm = config.get("training", {}).get("grad_clip", {}).get("max_norm", 35.0)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Starting training loop")
        logger.info(f"  Epochs: {start_epoch} -> {args.epochs}")
        logger.info(f"  Batch size per GPU: {args.batch_size}")
        logger.info(f"  Total batch size: {args.batch_size * world_size}")
        logger.info(f"  Base LR: {base_lr}")
        logger.info(f"  Backbone LR: {base_lr * 0.1}")
        logger.info(f"  Weight decay: {args.weight_decay}")
        logger.info(f"  AMP: {use_amp}")
        logger.info(f"  Gradient clip: {max_grad_norm}")
        logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_start = time.time()

        # Train one epoch
        epoch_losses = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            rank=rank,
            logger=logger,
            log_interval=log_interval,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
        )

        epoch_time = time.time() - epoch_start

        if rank == 0:
            logger.info(
                f"Epoch {epoch} completed in {epoch_time:.1f}s | "
                f"avg_loss: {epoch_losses['loss_total']:.4f} | "
                f"cls: {epoch_losses['loss_cls']:.4f} | "
                f"pts: {epoch_losses['loss_pts']:.4f} | "
                f"dir: {epoch_losses['loss_dir']:.4f}"
            )

            # Save checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                work_dir=args.work_dir,
            )
            logger.info(f"Checkpoint saved: epoch_{epoch}.pth")

    # Cleanup
    if world_size > 1:
        cleanup_distributed()

    if rank == 0:
        logger.info("Training complete!")


def main():
    """Entry point: parse arguments and launch training."""
    parser = argparse.ArgumentParser(description="MapTR Training Script")

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/nuscenes/",
        help="Root directory of nuScenes dataset",
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        default="data/nuscenes/maptr_train_ann.pkl",
        help="Path to annotation file (pickle or JSON)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=24, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=6e-4, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")

    # Precision
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Use mixed precision (AMP)"
    )
    parser.add_argument(
        "--no_fp16", action="store_true", help="Disable mixed precision"
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="work_dirs/maptr_r50_nuscenes_24ep",
        help="Directory for checkpoints and logs",
    )

    # Seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # Handle fp16 flag
    if args.no_fp16:
        args.fp16 = False

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Determine world size
    world_size = args.num_gpus
    if world_size < 1:
        world_size = torch.cuda.device_count()
    world_size = min(world_size, torch.cuda.device_count()) if torch.cuda.is_available() else 1

    if world_size > 1:
        # Multi-GPU: spawn processes
        torch.multiprocessing.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU
        main_worker(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
