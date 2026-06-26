"""Loss functions for RangeNet++ semantic segmentation.

Implements:
  - WeightedCrossEntropyLoss: Standard CE with per-class weights (inverse frequency).
  - LovaszSoftmaxLoss: Lovasz extension of IoU loss for better boundary performance.
  - CombinedLoss: Weighted sum of WCE + Lovasz.

Reference:
  Lovasz-Softmax: "The Lovász-Softmax loss: A tractable surrogate for the
  optimization of the intersection-over-union measure in neural networks"
  (Berman et al., CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with per-class weights based on inverse frequency.

    Handles the ignore_index for unlabeled pixels (class 0 in SemanticKITTI).
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 0,
    ):
        """
        Args:
            class_weights: (num_classes,) tensor of per-class weights.
                          If None, uses uniform weights.
            ignore_index: Label value to ignore in loss computation.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) predicted logits.
            targets: (B, H, W) ground truth labels (long).

        Returns:
            Scalar loss value.
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="mean",
        )


def compute_class_weights(class_counts: np.ndarray, num_classes: int = 20) -> torch.Tensor:
    """Compute inverse-frequency class weights for SemanticKITTI.

    Args:
        class_counts: Array of per-class point counts (length num_classes).
        num_classes: Total number of classes.

    Returns:
        Tensor of shape (num_classes,) with normalized weights.
    """
    # Avoid division by zero
    counts = np.maximum(class_counts.astype(np.float64), 1.0)
    # Inverse frequency
    weights = 1.0 / np.log(1.02 + counts / counts.sum())
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def get_default_semantickitti_weights(num_classes: int = 20) -> torch.Tensor:
    """Default class weights for SemanticKITTI 20-class setup.

    Approximate inverse-frequency weights based on published class distributions.
    Class 0 (unlabeled) gets weight 0 since it's ignored.
    """
    # Approximate relative frequencies from SemanticKITTI statistics
    # [unlabeled, car, bicycle, motorcycle, truck, other-vehicle, person,
    #  bicyclist, motorcyclist, road, parking, sidewalk, other-ground, building,
    #  fence, vegetation, trunk, terrain, pole, traffic-sign]
    frequencies = np.array([
        0.0,       # 0: unlabeled (ignored)
        0.0389,    # 1: car
        0.0002,    # 2: bicycle
        0.0003,    # 3: motorcycle
        0.0032,    # 4: truck
        0.0022,    # 5: other-vehicle
        0.0008,    # 6: person
        0.0002,    # 7: bicyclist
        0.0001,    # 8: motorcyclist
        0.1540,    # 9: road
        0.0193,    # 10: parking
        0.0590,    # 11: sidewalk
        0.0016,    # 12: other-ground
        0.0802,    # 13: building
        0.0185,    # 14: fence
        0.1870,    # 15: vegetation
        0.0059,    # 16: trunk
        0.0948,    # 17: terrain
        0.0048,    # 18: pole
        0.0014,    # 19: traffic-sign
    ], dtype=np.float64)

    weights = np.zeros(num_classes, dtype=np.float64)
    for i in range(1, num_classes):
        if frequencies[i] > 0:
            weights[i] = 1.0 / np.log(1.02 + frequencies[i])
    # Normalize evaluated classes (1-19) to mean=1
    evaluated = weights[1:]
    if evaluated.sum() > 0:
        evaluated_mean = evaluated[evaluated > 0].mean()
        weights[1:] = evaluated / evaluated_mean
    # Class 0 weight = 0 (ignored)
    weights[0] = 0.0

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# Lovász-Softmax Loss
# ============================================================================


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute the gradient of the Lovász extension w.r.t. sorted errors.

    This implements the piecewise-linear Lovász extension of the Jaccard loss.

    Args:
        gt_sorted: Sorted ground truth indicators (1 for positive, 0 for negative).

    Returns:
        Gradient tensor of the same shape.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()

    # Intersection and union as we include more errors
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union

    # Compute gradient: difference of consecutive Jaccard values
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    return jaccard


def lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
) -> torch.Tensor:
    """Multi-class Lovász-Softmax loss on flattened predictions.

    Args:
        probas: (P, C) tensor of class probabilities (after softmax).
        labels: (P,) tensor of ground truth class labels.
        classes: 'all' to use all classes, 'present' to use only classes
                 present in the batch.

    Returns:
        Scalar loss value.
    """
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.shape[1]
    losses = []

    for c in range(C):
        # Binary classification for class c: foreground (1) vs background (0)
        fg = (labels == c).float()

        if classes == "present" and fg.sum() == 0:
            continue

        if C == 1:
            fg_class_prob = probas[:, 0]
        else:
            fg_class_prob = probas[:, c]

        # Errors: 1 - prob for positives, prob for negatives
        errors = (fg - fg_class_prob).abs()

        # Sort by decreasing error
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]

        # Compute Lovász gradient and loss
        grad = lovasz_grad(fg_sorted)
        loss_c = torch.dot(errors_sorted, grad)
        losses.append(loss_c)

    if len(losses) == 0:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)

    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax loss for multi-class semantic segmentation.

    Optimizes a surrogate of the IoU metric directly, which tends to produce
    better boundary delineation than cross-entropy alone.
    """

    def __init__(self, classes: str = "present", ignore_index: int = 0):
        """
        Args:
            classes: 'present' to only consider classes in the batch,
                     'all' to compute for all classes.
            ignore_index: Label value to exclude from loss computation.
        """
        super().__init__()
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) predicted logits.
            targets: (B, H, W) ground truth labels.

        Returns:
            Scalar loss value.
        """
        B, C, H, W = logits.shape
        probas = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Flatten spatial dimensions
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)
        targets_flat = targets.view(-1)  # (B*H*W,)

        # Remove ignore_index pixels
        valid_mask = targets_flat != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probas = probas[valid_mask]
        targets_flat = targets_flat[valid_mask]

        return lovasz_softmax_flat(probas, targets_flat, classes=self.classes)


class CombinedLoss(nn.Module):
    """Combined loss: weighted sum of Weighted Cross-Entropy and Lovász-Softmax.

    loss = alpha * WCE + beta * Lovász
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 0,
        alpha: float = 1.0,
        beta: float = 1.5,
        lovasz_classes: str = "present",
    ):
        """
        Args:
            class_weights: Per-class weights for CE loss.
            ignore_index: Label to ignore.
            alpha: Weight for cross-entropy loss.
            beta: Weight for Lovász loss.
            lovasz_classes: 'present' or 'all' for Lovász.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.wce = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            ignore_index=ignore_index,
        )
        self.lovasz = LovaszSoftmaxLoss(
            classes=lovasz_classes,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) predicted logits.
            targets: (B, H, W) ground truth labels.

        Returns:
            Scalar combined loss.
        """
        loss_wce = self.wce(logits, targets)
        loss_lovasz = self.lovasz(logits, targets)
        return self.alpha * loss_wce + self.beta * loss_lovasz
