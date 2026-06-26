"""Loss functions for PointPillars 3D object detection.

This module implements the loss functions used to train PointPillars networks
for LiDAR-based 3D object detection, including focal loss for classification,
weighted smooth L1 for bounding box regression, and binary cross-entropy for
direction classification.

References:
    - PointPillars: Fast Encoders for Object Detection from Point Clouds
      (Lang et al., 2019)
    - Focal Loss for Dense Object Detection (Lin et al., 2017)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal Loss for dense classification in anchor-based detectors.

    Focal loss addresses class imbalance by down-weighting well-classified
    examples so the model focuses on hard negatives. The loss for a sample
    with ground-truth class y in {0, 1} is:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1 else (1-p), and alpha_t = alpha if y=1 else (1-alpha).

    Args:
        alpha: Balancing factor for the positive class. Default: 0.25.
        gamma: Focusing parameter that reduces loss for well-classified
            examples. Default: 2.0.
        reduction: Specifies the reduction to apply to the output.
            One of 'none', 'mean', or 'sum'. Default: 'none'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"Invalid reduction mode '{reduction}'. "
                "Expected one of 'none', 'mean', 'sum'."
            )
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            pred: Predicted logits of shape (N, ...) before sigmoid activation.
            target: Binary ground-truth labels of shape (N, ...) with values
                in {0, 1}.

        Returns:
            Focal loss tensor. Shape depends on the reduction mode:
            'none' returns element-wise loss, 'mean' and 'sum' return scalars.
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(pred)

        # Binary cross-entropy component (numerically stable)
        # -log(p) when target=1, -log(1-p) when target=0
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # p_t is the model's estimated probability for the true class
        p_t = p * target + (1.0 - p) * (1.0 - target)

        # Focal modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t).pow(self.gamma)

        # Alpha balancing factor: alpha for positives, (1-alpha) for negatives
        alpha_factor = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)

        # Final focal loss
        loss = alpha_factor * modulating_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedSmoothL1Loss(nn.Module):
    """Weighted Smooth L1 Loss for bounding box regression.

    Smooth L1 loss that supports per-element weights and per-dimension
    code weights. This is commonly used for 3D bounding box regression
    where different box parameters (x, y, z, w, l, h, theta) may have
    different importance levels.

    The smooth L1 function is defined as:
        smooth_l1(x) = 0.5 * x^2 / beta   if |x| < beta
                       |x| - 0.5 * beta    otherwise

    Args:
        sigma: Controls the transition point between L1 and L2 regions.
            beta = 1/sigma^2. A larger sigma means a narrower quadratic
            region. Default: 3.0.
        code_weights: Optional list of weights for each box code dimension.
            For example, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5] would
            down-weight the heading angle relative to position and size.
            Default: None (uniform weighting).
        reduction: Specifies the reduction to apply. One of 'none', 'mean',
            or 'sum'. Default: 'none'.
    """

    def __init__(
        self,
        sigma: float = 3.0,
        code_weights: Optional[list] = None,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"Invalid reduction mode '{reduction}'. "
                "Expected one of 'none', 'mean', 'sum'."
            )
        self.sigma = sigma
        self.beta = 1.0 / (sigma ** 2)
        self.reduction = reduction

        if code_weights is not None:
            self.register_buffer(
                "code_weights", torch.tensor(code_weights, dtype=torch.float32)
            )
        else:
            self.code_weights: Optional[Tensor] = None

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute weighted smooth L1 loss.

        Args:
            pred: Predicted box regression values of shape (N, num_codes)
                or (N, H, W, num_codes).
            target: Ground-truth box regression targets with the same shape
                as pred.
            weights: Per-element weights of shape broadcastable to pred.
                Typically a binary mask indicating which anchors are positive.
                Default: None (uniform weight of 1).

        Returns:
            Weighted smooth L1 loss tensor. Shape depends on reduction mode.
        """
        diff = pred - target
        abs_diff = torch.abs(diff)

        # Smooth L1: quadratic for small values, linear for large values
        smooth_l1 = torch.where(
            abs_diff < self.beta,
            0.5 * (diff ** 2) / self.beta,
            abs_diff - 0.5 * self.beta,
        )

        # Apply code weights (per-dimension weighting)
        if self.code_weights is not None:
            # Reshape code_weights to broadcast with smooth_l1
            # code_weights shape: (num_codes,) -> broadcast to (..., num_codes)
            code_weights = self.code_weights.reshape(
                [1] * (smooth_l1.dim() - 1) + [-1]
            )
            smooth_l1 = smooth_l1 * code_weights

        # Apply per-element weights
        if weights is not None:
            smooth_l1 = smooth_l1 * weights

        if self.reduction == "mean":
            return smooth_l1.mean()
        elif self.reduction == "sum":
            return smooth_l1.sum()
        return smooth_l1


class DirectionClassificationLoss(nn.Module):
    """Binary cross-entropy loss for heading direction classification.

    PointPillars uses a two-bin direction classifier to resolve the heading
    ambiguity in bounding box regression (since sin(theta) and sin(theta+pi)
    produce similar regression targets). This loss trains a binary classifier
    that predicts which of the two heading bins the object belongs to.

    Args:
        reduction: Specifies the reduction to apply. One of 'none', 'mean',
            or 'sum'. Default: 'none'.
    """

    def __init__(self, reduction: str = "none") -> None:
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"Invalid reduction mode '{reduction}'. "
                "Expected one of 'none', 'mean', 'sum'."
            )
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute direction classification loss.

        Args:
            pred: Predicted direction logits of shape (N, 2) representing
                the two heading bins.
            target: Ground-truth direction labels of shape (N,) with integer
                values in {0, 1} indicating the correct heading bin.
            weights: Optional per-sample weights of shape (N,) for masking
                out negative/ignore anchors. Default: None.

        Returns:
            Binary cross-entropy loss for direction classification.
        """
        # Convert integer targets to one-hot for cross-entropy
        # pred shape: (N, 2), target shape: (N,) with values 0 or 1
        loss = F.cross_entropy(pred, target.long(), reduction="none")

        # Apply per-sample weights (e.g., only compute for positive anchors)
        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PointPillarsLoss(nn.Module):
    """Combined loss function for PointPillars 3D object detection.

    Aggregates focal loss (classification), weighted smooth L1 (box regression),
    and direction classification loss into a single training objective:

        total_loss = cls_weight * focal_loss
                   + reg_weight * smooth_l1_loss
                   + dir_weight * direction_loss

    All individual losses are normalized by the number of positive anchors
    to stabilize training regardless of batch size or anchor density.

    Args:
        cls_weight: Weight for classification focal loss. Default: 1.0.
        reg_weight: Weight for bounding box regression loss. Default: 2.0.
        dir_weight: Weight for direction classification loss. Default: 0.2.
        focal_alpha: Alpha parameter for focal loss. Default: 0.25.
        focal_gamma: Gamma parameter for focal loss. Default: 2.0.
        smooth_l1_sigma: Sigma parameter for smooth L1 loss. Default: 3.0.
        code_weights: Optional per-dimension weights for box regression codes.
            Default: None.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 2.0,
        dir_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth_l1_sigma: float = 3.0,
        code_weights: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.dir_weight = dir_weight

        self.cls_loss_fn = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, reduction="none"
        )
        self.reg_loss_fn = WeightedSmoothL1Loss(
            sigma=smooth_l1_sigma, code_weights=code_weights, reduction="none"
        )
        self.dir_loss_fn = DirectionClassificationLoss(reduction="none")

    def forward(
        self,
        cls_preds: Tensor,
        box_preds: Tensor,
        dir_preds: Tensor,
        cls_targets: Tensor,
        reg_targets: Tensor,
        reg_weights: Tensor,
        dir_targets: Tensor,
        num_positives: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute the combined PointPillars training loss.

        Args:
            cls_preds: Classification logits of shape (B, num_anchors, num_classes)
                where B is the batch size. These are raw logits before sigmoid.
            box_preds: Box regression predictions of shape (B, num_anchors, 7)
                encoding (x, y, z, w, l, h, theta) residuals.
            dir_preds: Direction classification logits of shape
                (B, num_anchors, 2) for the two heading bins.
            cls_targets: Classification targets of shape (B, num_anchors, num_classes)
                with values in {0, 1}. Positive anchors have 1 in their class dim.
            reg_targets: Box regression targets of shape (B, num_anchors, 7)
                containing the encoded ground-truth residuals.
            reg_weights: Per-anchor regression weights of shape (B, num_anchors)
                that are 1 for positive anchors and 0 otherwise.
            dir_targets: Direction bin targets of shape (B, num_anchors) with
                integer values in {0, 1}.
            num_positives: Number of positive anchors per batch item, shape (B,)
                or scalar. Used for loss normalization.

        Returns:
            Dictionary containing:
                - 'cls_loss': Weighted classification focal loss (scalar).
                - 'reg_loss': Weighted bounding box regression loss (scalar).
                - 'dir_loss': Weighted direction classification loss (scalar).
                - 'total_loss': Sum of all weighted losses (scalar).
        """
        batch_size = cls_preds.shape[0]

        # Normalization factor: total positive anchors across the batch,
        # clamped to at least 1 to avoid division by zero
        normalizer = num_positives.sum().clamp(min=1.0).float()

        # --- Classification Loss ---
        # cls_preds: (B, num_anchors, num_classes)
        # cls_targets: (B, num_anchors, num_classes)
        cls_loss = self.cls_loss_fn(cls_preds, cls_targets)
        cls_loss = cls_loss.sum() / normalizer

        # --- Regression Loss ---
        # box_preds: (B, num_anchors, 7)
        # reg_targets: (B, num_anchors, 7)
        # reg_weights: (B, num_anchors) -> expand to (B, num_anchors, 1)
        reg_weights_expanded = reg_weights.unsqueeze(-1)
        reg_loss = self.reg_loss_fn(box_preds, reg_targets, reg_weights_expanded)
        reg_loss = reg_loss.sum() / normalizer

        # --- Direction Classification Loss ---
        # dir_preds: (B, num_anchors, 2) -> reshape to (B*num_anchors, 2)
        # dir_targets: (B, num_anchors) -> reshape to (B*num_anchors,)
        num_anchors = dir_preds.shape[1]
        dir_preds_flat = dir_preds.reshape(batch_size * num_anchors, 2)
        dir_targets_flat = dir_targets.reshape(batch_size * num_anchors)
        dir_weights_flat = reg_weights.reshape(batch_size * num_anchors)

        dir_loss = self.dir_loss_fn(dir_preds_flat, dir_targets_flat, dir_weights_flat)
        dir_loss = dir_loss.sum() / normalizer

        # --- Combined Loss ---
        weighted_cls_loss = self.cls_weight * cls_loss
        weighted_reg_loss = self.reg_weight * reg_loss
        weighted_dir_loss = self.dir_weight * dir_loss
        total_loss = weighted_cls_loss + weighted_reg_loss + weighted_dir_loss

        return {
            "cls_loss": weighted_cls_loss,
            "reg_loss": weighted_reg_loss,
            "dir_loss": weighted_dir_loss,
            "total_loss": total_loss,
        }
