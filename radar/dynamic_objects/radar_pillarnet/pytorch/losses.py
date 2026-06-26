"""
Detection losses with velocity regression for RadarPillarNet.

Implements all loss functions needed for training:
- Focal Loss: handles class imbalance in anchor classification
- Smooth L1 Loss: robust regression loss for box parameters
- Direction Classification Loss: BCE for heading disambiguation
- Velocity Regression Loss: Smooth L1 for (vx, vy) prediction

All losses are implemented from scratch without relying on external libraries.

Total loss = cls_weight * focal_loss + reg_weight * smooth_l1_box
           + vel_weight * smooth_l1_vel + dir_weight * bce_dir
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal Loss for dense classification (Lin et al., 2017).

    Addresses class imbalance by down-weighting easy examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        pred: (N, C) raw logits (before sigmoid). C = num_classes.
        target: (N, C) one-hot encoded targets or soft labels in [0, 1].
        alpha: Weighting factor for positive class (default 0.25).
        gamma: Focusing parameter - higher values focus more on hard examples.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (if reduction='mean' or 'sum') or (N, C) per-element loss.
    """
    # Apply sigmoid to get probabilities
    pred_sigmoid = torch.sigmoid(pred)  # (N, C)

    # Compute binary cross entropy component
    # -target * log(p) - (1-target) * log(1-p)
    # Using log-sum-exp trick for numerical stability
    bce = F.binary_cross_entropy_with_logits(
        pred, target, reduction="none"
    )  # (N, C)

    # Compute p_t (probability of correct class)
    p_t = target * pred_sigmoid + (1.0 - target) * (1.0 - pred_sigmoid)

    # Compute alpha_t
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)

    # Focal weight: alpha_t * (1 - p_t)^gamma
    focal_weight = alpha_t * (1.0 - p_t).pow(gamma)

    # Final focal loss
    loss = focal_weight * bce  # (N, C)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Smooth L1 (Huber) loss for robust regression.

    SmoothL1(x) = 0.5 * x^2 / beta   if |x| < beta
                  |x| - 0.5 * beta     otherwise

    Less sensitive to outliers than L2 loss, more stable gradients than L1.

    Args:
        pred: (...) predicted values.
        target: (...) target values, same shape as pred.
        beta: Transition point between L1 and L2 behavior (default 1.0).
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar or element-wise loss depending on reduction.
    """
    diff = torch.abs(pred - target)

    # Piecewise function
    loss = torch.where(
        diff < beta,
        0.5 * diff.pow(2) / beta,
        diff - 0.5 * beta,
    )

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def weighted_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Weighted Smooth L1 loss for per-element importance weighting.

    Args:
        pred: (N, D) predicted values.
        target: (N, D) target values.
        weights: (N,) per-sample weights.
        beta: Smooth L1 transition point.

    Returns:
        Scalar weighted loss.
    """
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta,
        0.5 * diff.pow(2) / beta,
        diff - 0.5 * beta,
    )  # (N, D)

    # Sum over feature dim, then weight
    loss_per_sample = loss.sum(dim=-1)  # (N,)
    weighted_loss = (loss_per_sample * weights).sum()

    # Normalize by number of positive weights
    num_pos = weights.sum().clamp(min=1.0)
    return weighted_loss / num_pos


def direction_classification_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary cross-entropy loss for direction classification.

    Predicts which of two heading bins (0 or pi offset) the object faces.

    Args:
        pred: (N, 2) direction logits (two bins).
        target: (N,) long tensor with direction labels {0, 1}.
        weights: Optional (N,) per-sample weights.

    Returns:
        Scalar direction loss.
    """
    loss = F.cross_entropy(pred, target, reduction="none")  # (N,)

    if weights is not None:
        loss = loss * weights
        num_pos = weights.sum().clamp(min=1.0)
        return loss.sum() / num_pos
    else:
        return loss.mean()


class RadarPillarNetLoss(nn.Module):
    """Combined detection loss for RadarPillarNet training.

    Computes weighted sum of:
    1. Classification loss (focal loss) - handles imbalanced anchors
    2. Box regression loss (smooth L1) - 3D box parameters
    3. Velocity regression loss (smooth L1) - vx, vy prediction
    4. Direction classification loss (BCE) - heading disambiguation

    Total = cls_weight * L_cls + reg_weight * L_reg
          + vel_weight * L_vel + dir_weight * L_dir
    """

    def __init__(
        self,
        num_classes: int = 4,
        cls_weight: float = 1.0,
        reg_weight: float = 2.0,
        vel_weight: float = 0.2,
        dir_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth_l1_beta: float = 1.0,
        code_weights: Optional[list] = None,
    ) -> None:
        """Initialize loss module.

        Args:
            num_classes: Number of object classes.
            cls_weight: Weight for classification loss.
            reg_weight: Weight for box regression loss.
            vel_weight: Weight for velocity regression loss.
            dir_weight: Weight for direction classification loss.
            focal_alpha: Alpha parameter for focal loss.
            focal_gamma: Gamma parameter for focal loss.
            smooth_l1_beta: Beta for smooth L1 transition.
            code_weights: Optional per-dimension weights for box regression.
                Length 7 for [x, y, z, w, l, h, theta]. Default all 1.0.
        """
        super().__init__()

        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.vel_weight = vel_weight
        self.dir_weight = dir_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth_l1_beta = smooth_l1_beta

        if code_weights is None:
            code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.register_buffer(
            "code_weights", torch.tensor(code_weights, dtype=torch.float32)
        )

    def forward(
        self,
        cls_preds: torch.Tensor,
        box_preds: torch.Tensor,
        vel_preds: torch.Tensor,
        dir_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        box_targets: torch.Tensor,
        vel_targets: torch.Tensor,
        dir_targets: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            cls_preds: (B, N, num_classes) classification logits.
            box_preds: (B, N, 7) box regression predictions.
            vel_preds: (B, N, 2) velocity predictions.
            dir_preds: (B, N, 2) direction logits.
            cls_targets: (B, N, num_classes) one-hot classification targets.
            box_targets: (B, N, 7) box regression targets (encoded deltas).
            vel_targets: (B, N, 2) velocity targets (vx, vy).
            dir_targets: (B, N) direction classification targets {0, 1}.
            pos_mask: (B, N) boolean mask for positive (matched) anchors.
            neg_mask: (B, N) boolean mask for negative anchors.

        Returns:
            Dict with keys:
                'cls_loss': scalar classification loss
                'reg_loss': scalar box regression loss
                'vel_loss': scalar velocity regression loss
                'dir_loss': scalar direction classification loss
                'total_loss': weighted sum of all losses
        """
        batch_size = cls_preds.shape[0]

        # --- Classification Loss (all anchors: positive + negative) ---
        # Focal loss operates on positive and negative anchors (ignore anchors excluded)
        cls_mask = pos_mask | neg_mask  # (B, N)
        cls_preds_flat = cls_preds[cls_mask]  # (M, num_classes)
        cls_targets_flat = cls_targets[cls_mask]  # (M, num_classes)

        if cls_preds_flat.shape[0] > 0:
            cls_loss = focal_loss(
                cls_preds_flat,
                cls_targets_flat,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum",
            )
            # Normalize by number of positive anchors
            num_pos = pos_mask.sum().clamp(min=1.0).float()
            cls_loss = cls_loss / num_pos
        else:
            cls_loss = torch.tensor(0.0, device=cls_preds.device, requires_grad=True)

        # --- Box Regression Loss (positive anchors only) ---
        if pos_mask.sum() > 0:
            box_preds_pos = box_preds[pos_mask]  # (P, 7)
            box_targets_pos = box_targets[pos_mask]  # (P, 7)

            # Apply per-dimension code weights
            box_diff = box_preds_pos - box_targets_pos  # (P, 7)
            box_diff_weighted = box_diff * self.code_weights.unsqueeze(0)  # (P, 7)

            reg_loss = smooth_l1_loss(
                box_diff_weighted,
                torch.zeros_like(box_diff_weighted),
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
            reg_loss = reg_loss / pos_mask.sum().clamp(min=1.0).float()
        else:
            reg_loss = torch.tensor(0.0, device=cls_preds.device, requires_grad=True)

        # --- Velocity Regression Loss (positive anchors only) ---
        if pos_mask.sum() > 0:
            vel_preds_pos = vel_preds[pos_mask]  # (P, 2)
            vel_targets_pos = vel_targets[pos_mask]  # (P, 2)

            vel_loss = smooth_l1_loss(
                vel_preds_pos,
                vel_targets_pos,
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
            vel_loss = vel_loss / pos_mask.sum().clamp(min=1.0).float()
        else:
            vel_loss = torch.tensor(0.0, device=cls_preds.device, requires_grad=True)

        # --- Direction Classification Loss (positive anchors only) ---
        if pos_mask.sum() > 0:
            dir_preds_pos = dir_preds[pos_mask]  # (P, 2)
            dir_targets_pos = dir_targets[pos_mask]  # (P,)

            dir_loss = direction_classification_loss(
                dir_preds_pos, dir_targets_pos
            )
        else:
            dir_loss = torch.tensor(0.0, device=cls_preds.device, requires_grad=True)

        # --- Total Loss ---
        total_loss = (
            self.cls_weight * cls_loss
            + self.reg_weight * reg_loss
            + self.vel_weight * vel_loss
            + self.dir_weight * dir_loss
        )

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "vel_loss": vel_loss,
            "dir_loss": dir_loss,
            "total_loss": total_loss,
        }
