# [IMPLEMENTED BY CLAUDE - was missing]

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for occupancy prediction.

    Addresses class imbalance by down-weighting well-classified examples
    and focusing on hard negatives/positives.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary focal loss.

        Args:
            logits: Raw predictions of shape (B, 1, H, W) or (B, H, W).
            targets: Binary ground truth of shape (B, H, W) with values in {0, 1, ignore_index}.

        Returns:
            Scalar mean focal loss over valid cells.
        """
        # Squeeze channel dim if present
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        # Create valid mask (cells that are not ignore_index)
        valid_mask = targets != self.ignore_index

        # Select only valid cells
        logits_valid = logits[valid_mask]
        targets_valid = targets[valid_mask].float()

        if logits_valid.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute probabilities
        probs = torch.sigmoid(logits_valid)

        # Binary cross-entropy (per element)
        bce = F.binary_cross_entropy_with_logits(logits_valid, targets_valid, reduction='none')

        # p_t: probability of the true class
        p_t = probs * targets_valid + (1 - probs) * (1 - targets_valid)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing: alpha for positive, (1 - alpha) for negative
        alpha_t = self.alpha * targets_valid + (1 - self.alpha) * (1 - targets_valid)

        # Combined loss
        loss = alpha_t * focal_weight * bce

        return loss.mean()


class WCELoss(nn.Module):
    """Weighted Cross-Entropy loss for semantic segmentation.

    Wraps nn.CrossEntropyLoss with optional per-class weights and ignore index.
    """

    def __init__(self, class_weights: torch.Tensor = None, ignore_index: int = 255):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            logits: Predictions of shape (B, C, H, W) where C is the number of classes.
            targets: Ground truth of shape (B, H, W) with integer class labels.

        Returns:
            Scalar cross-entropy loss.
        """
        return self.ce_loss(logits, targets)


class RadarOccupancyLoss(nn.Module):
    """Combined loss for radar occupancy prediction.

    Combines binary focal loss for occupancy prediction with weighted
    cross-entropy for optional semantic segmentation.
    """

    def __init__(
        self,
        occ_weight: float = 1.0,
        sem_weight: float = 0.5,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.occ_weight = occ_weight
        self.sem_weight = sem_weight

        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
        )
        self.wce_loss = WCELoss(
            class_weights=class_weights,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        occ_logits: torch.Tensor,
        occ_target: torch.Tensor,
        sem_logits: torch.Tensor = None,
        sem_target: torch.Tensor = None,
    ) -> dict:
        """Compute combined occupancy and semantic loss.

        Args:
            occ_logits: Occupancy predictions of shape (B, 1, H, W) or (B, H, W).
            occ_target: Binary occupancy ground truth of shape (B, H, W).
            sem_logits: Optional semantic predictions of shape (B, C, H, W).
            sem_target: Optional semantic ground truth of shape (B, H, W).

        Returns:
            Dictionary with keys 'total', 'occupancy', and 'semantic'.
        """
        # Occupancy loss (always computed)
        occ_loss = self.focal_loss(occ_logits, occ_target)

        # Semantic loss (only if both logits and target are provided)
        if sem_logits is not None and sem_target is not None:
            sem_loss = self.wce_loss(sem_logits, sem_target)
        else:
            sem_loss = torch.tensor(0.0, device=occ_logits.device)

        # Weighted combination
        total_loss = self.occ_weight * occ_loss + self.sem_weight * sem_loss

        return {
            'total': total_loss,
            'occupancy': occ_loss,
            'semantic': sem_loss,
        }
