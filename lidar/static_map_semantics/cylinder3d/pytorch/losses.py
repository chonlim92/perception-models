"""
Loss Functions for Cylinder3D.

Implements:
    - Lovasz-Softmax Loss: A surrogate for IoU optimization via the Lovasz extension
    - Weighted Cross-Entropy Loss: Standard CE with per-class weights and label smoothing
    - Combined Loss: Weighted combination of CE and Lovasz losses

Reference:
    Berman et al., "The Lovasz-Softmax loss: A tractable surrogate for the
    optimization of the intersection-over-union measure in neural networks", CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List
from itertools import filterfalse


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Compute the Lovasz extension gradient.

    Given a sorted binary ground truth vector, computes the gradient of the
    Lovasz extension of the Jaccard (IoU) loss.

    The Lovasz gradient for a sorted error vector is:
        grad_i = (|{j : j >= i}| intersection gt) / |{j : j >= i}| union gt|
               = 1/|union| for items in gt, scaled by position

    More precisely:
        grad[i] = 1 - intersection[i:] / union[i:]

    Args:
        gt_sorted: (P,) sorted binary ground truth (1 = positive, 0 = negative),
                   sorted by decreasing prediction error.

    Returns:
        grad: (P,) Lovasz gradient vector
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()

    # Intersection: cumulative sum of gt from position i to end
    intersection = gts - gt_sorted.float().cumsum(0)

    # Union: (number of items from i to end) + gts - intersection
    union = gts + torch.arange(1, p + 1, device=gt_sorted.device, dtype=torch.float32) - gt_sorted.float().cumsum(0)

    # Jaccard at each position
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)

    # Gradient is the difference of consecutive Jaccard values
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]

    return jaccard


def lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
) -> torch.Tensor:
    """
    Multi-class Lovasz-Softmax loss on flattened predictions.

    Computes the Lovasz extension of the IoU loss for each class,
    then averages over classes.

    Args:
        probas: (P, C) class probabilities at each spatial location
                (after softmax). P = number of spatial locations.
        labels: (P,) ground truth class indices in [0, C-1]
        classes: Which classes to compute loss for:
            - 'present': only classes present in labels (default)
            - 'all': all classes
            - list of int: specific class indices

    Returns:
        loss: scalar Lovasz-Softmax loss
    """
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.shape[1]
    losses = []

    if classes == "present":
        class_to_sum = torch.unique(labels)
    elif classes == "all":
        class_to_sum = torch.arange(C, device=probas.device)
    else:
        class_to_sum = torch.tensor(classes, device=probas.device, dtype=torch.long)

    for c in class_to_sum:
        c = c.item()
        # Binary ground truth for class c
        fg = (labels == c).float()

        if fg.sum() == 0 and classes == "present":
            continue

        if C == 1:
            fg_class = 1.0 - probas[:, 0]
        else:
            fg_class = probas[:, c]

        # Compute errors: 1 - p(correct_class) for positives, p(c) for negatives
        errors = (fg - fg_class).abs()

        # Sort by decreasing errors
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]

        # Compute Lovasz gradient
        grad = lovasz_grad(fg_sorted)

        # Loss for this class
        loss_c = torch.dot(F.relu(errors_sorted), grad)
        losses.append(loss_c)

    if len(losses) == 0:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)

    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovasz-Softmax loss for multi-class semantic segmentation.

    Optimizes a surrogate of the mean IoU by leveraging the Lovasz extension
    of submodular set functions. Unlike cross-entropy which optimizes per-pixel
    accuracy, Lovasz-Softmax directly targets the IoU metric.

    Args:
        classes: Which classes to include in the loss computation:
            - 'present': only classes present in the batch (default)
            - 'all': all classes
            - list of int: specific class indices
        per_sample: If True, compute loss per sample then average.
                    If False, flatten all samples together. Default: False
        ignore_index: Class index to ignore in loss computation. Default: 255
    """

    def __init__(
        self,
        classes: str = "present",
        per_sample: bool = False,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.classes = classes
        self.per_sample = per_sample
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lovasz-Softmax loss.

        Args:
            logits: (N, C) or (B, C, *spatial) raw class logits (before softmax)
            labels: (N,) or (B, *spatial) ground truth class indices

        Returns:
            loss: scalar loss value
        """
        # Convert logits to probabilities
        probas = F.softmax(logits, dim=1)

        # Handle multi-dimensional inputs (e.g., voxel grids)
        if probas.dim() > 2:
            # (B, C, D1, D2, ...) -> (B*D1*D2*..., C)
            B, C = probas.shape[:2]
            probas = probas.permute(0, *range(2, probas.dim()), 1).contiguous()
            probas = probas.view(-1, C)
            labels = labels.view(-1)

        # Filter out ignore_index
        valid_mask = labels != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        vprobas = probas[valid_mask]
        vlabels = labels[valid_mask]

        if self.per_sample:
            # Not applicable after flattening; use flat version
            loss = lovasz_softmax_flat(vprobas, vlabels, classes=self.classes)
        else:
            loss = lovasz_softmax_flat(vprobas, vlabels, classes=self.classes)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted cross-entropy loss with optional label smoothing.

    Applies per-class weights to handle class imbalance (common in LiDAR
    segmentation where road/vegetation dominate) and optional label smoothing
    for regularization.

    Args:
        class_weights: (C,) per-class weights. If None, uniform weights.
        ignore_index: Class index to ignore. Default: 255
        label_smoothing: Label smoothing factor in [0, 1). Default: 0.0
        reduction: Loss reduction mode ('mean', 'sum', 'none'). Default: 'mean'
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            logits: (N, C) or (B, C, *spatial) raw class logits
            labels: (N,) or (B, *spatial) ground truth class indices

        Returns:
            loss: scalar loss value
        """
        if self.label_smoothing > 0:
            return self._label_smoothed_loss(logits, labels)

        # Standard weighted cross-entropy
        if logits.dim() > 2:
            # Reshape for F.cross_entropy: needs (N, C) or (B, C, ...)
            loss = F.cross_entropy(
                logits,
                labels,
                weight=self.class_weights,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )
        else:
            loss = F.cross_entropy(
                logits,
                labels,
                weight=self.class_weights,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )

        return loss

    def _label_smoothed_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy with label smoothing.

        Distributes (label_smoothing / C) probability to all classes and
        (1 - label_smoothing) to the ground truth class.
        """
        # Flatten if needed
        if logits.dim() > 2:
            B, C = logits.shape[:2]
            logits_flat = logits.permute(0, *range(2, logits.dim()), 1).contiguous().view(-1, C)
            labels_flat = labels.view(-1)
        else:
            logits_flat = logits
            labels_flat = labels
            C = logits.shape[1]

        # Filter ignore index
        valid_mask = labels_flat != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_valid = logits_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]
        N = logits_valid.shape[0]
        C = logits_valid.shape[1]

        # Compute log softmax
        log_probs = F.log_softmax(logits_valid, dim=1)

        # Create smoothed target distribution
        smooth_targets = torch.full_like(log_probs, self.label_smoothing / C)
        smooth_targets.scatter_(
            1,
            labels_valid.unsqueeze(1),
            1.0 - self.label_smoothing + self.label_smoothing / C,
        )

        # Compute loss: -sum(target * log_prob)
        loss_per_sample = -(smooth_targets * log_probs).sum(dim=1)

        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights[labels_valid]
            loss_per_sample = loss_per_sample * weights

        if self.reduction == "mean":
            if self.class_weights is not None:
                loss = loss_per_sample.sum() / weights.sum().clamp(min=1e-6)
            else:
                loss = loss_per_sample.mean()
        elif self.reduction == "sum":
            loss = loss_per_sample.sum()
        else:
            loss = loss_per_sample

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted sum of Cross-Entropy and Lovasz-Softmax.

    This combination leverages the strengths of both losses:
        - CE provides stable gradients early in training
        - Lovasz directly optimizes IoU for better segmentation quality

    Total loss = ce_weight * CE_loss + lovasz_weight * Lovasz_loss

    Args:
        num_classes: Number of semantic classes
        ce_weight: Weight for cross-entropy loss. Default: 1.0
        lovasz_weight: Weight for Lovasz-Softmax loss. Default: 1.0
        class_weights: Optional per-class weights for CE loss
        ignore_index: Class index to ignore. Default: 255
        label_smoothing: Label smoothing for CE. Default: 0.0
        lovasz_classes: Class selection for Lovasz ('present' or 'all'). Default: 'present'
    """

    def __init__(
        self,
        num_classes: int = 20,
        ce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
        lovasz_classes: str = "present",
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.num_classes = num_classes

        self.ce_loss = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

        self.lovasz_loss = LovaszSoftmaxLoss(
            classes=lovasz_classes,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        point_logits: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Compute combined loss.

        Can compute loss on both voxel-level and point-level predictions.

        Args:
            logits: (B, C, *spatial) or (N, C) voxel/point logits
            labels: (B, *spatial) or (N,) voxel/point labels
            point_logits: Optional (N, C) point-level logits for additional loss
            point_labels: Optional (N,) point-level labels

        Returns:
            Dictionary containing:
                - 'total_loss': combined loss value
                - 'ce_loss': cross-entropy component
                - 'lovasz_loss': Lovasz-Softmax component
                - 'point_ce_loss': point-level CE (if point_logits provided)
                - 'point_lovasz_loss': point-level Lovasz (if point_logits provided)
        """
        # Voxel-level losses
        ce = self.ce_loss(logits, labels)
        lovasz = self.lovasz_loss(logits, labels)

        total = self.ce_weight * ce + self.lovasz_weight * lovasz

        result = {
            "total_loss": total,
            "ce_loss": ce,
            "lovasz_loss": lovasz,
        }

        # Optional point-level losses
        if point_logits is not None and point_labels is not None:
            point_ce = self.ce_loss(point_logits, point_labels)
            point_lovasz = self.lovasz_loss(point_logits, point_labels)

            point_total = self.ce_weight * point_ce + self.lovasz_weight * point_lovasz
            total = total + point_total

            result["point_ce_loss"] = point_ce
            result["point_lovasz_loss"] = point_lovasz
            result["total_loss"] = total

        return result
