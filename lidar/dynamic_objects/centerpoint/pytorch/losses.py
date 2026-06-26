"""
CenterPoint Loss Functions.

Implements Gaussian focal loss for heatmap supervision, L1 regression loss
at object center locations, and the combined CenterPoint training loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class GaussianFocalLoss(nn.Module):
    """Modified focal loss for dense heatmap prediction.

    For positive locations (where GT Gaussian > 0):
        loss = -(1 - pred)^alpha * log(pred) * weight
        where weight is the Gaussian value at that location.

    For negative locations (where GT Gaussian == 0):
        loss = -(1 - gt)^beta * pred^alpha * log(1 - pred)

    This formulation from CenterNet/CornerNet handles the soft Gaussian targets
    by penalizing negatives less near object centers.

    Args:
        alpha: Focusing parameter for hard example mining (default: 2.0).
        beta: Weight reduction near positive locations for negatives (default: 4.0).
        loss_weight: Scalar weight for this loss component.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        loss_weight: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian focal loss.

        Args:
            pred: (B, C, H, W) predicted heatmap (after sigmoid, values in [0, 1]).
            target: (B, C, H, W) ground truth heatmap with Gaussian peaks.

        Returns:
            Scalar loss value.
        """
        # Clamp predictions to avoid log(0)
        pred = pred.clamp(min=1e-6, max=1.0 - 1e-6)

        # Separate positive and negative locations
        positive_mask = target.eq(1.0)  # Exact center locations (peak of Gaussian)
        near_positive_mask = target.gt(0.0) & target.lt(1.0)  # Near centers (Gaussian tail)
        negative_mask = target.eq(0.0)  # Background

        # Positive loss: at exact Gaussian peaks (target == 1)
        pos_loss = torch.zeros_like(pred)
        if positive_mask.any():
            pos_loss[positive_mask] = (
                -((1.0 - pred[positive_mask]) ** self.alpha) *
                torch.log(pred[positive_mask])
            )

        # Near-positive loss: within Gaussian radius (0 < target < 1)
        # Weighted by the Gaussian value - closer to center gets more weight
        near_pos_loss = torch.zeros_like(pred)
        if near_positive_mask.any():
            near_pos_loss[near_positive_mask] = (
                -((1.0 - pred[near_positive_mask]) ** self.alpha) *
                torch.log(pred[near_positive_mask]) *
                target[near_positive_mask]  # Weight by Gaussian value
            )

        # Negative loss: background (target == 0)
        neg_loss = torch.zeros_like(pred)
        if negative_mask.any():
            neg_loss[negative_mask] = (
                -((1.0 - target[negative_mask]) ** self.beta) *
                (pred[negative_mask] ** self.alpha) *
                torch.log(1.0 - pred[negative_mask])
            )

        # Total loss
        loss = pos_loss + near_pos_loss + neg_loss

        # Normalize by number of positive locations
        num_pos = positive_mask.float().sum() + near_positive_mask.float().sum()
        num_pos = num_pos.clamp(min=1.0)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.sum() / num_pos
        # else: 'none' - return per-element loss

        return loss * self.loss_weight


class RegLoss(nn.Module):
    """L1 regression loss computed only at positive (object center) locations.

    Only computes loss where the mask indicates an object center exists,
    ignoring all background locations.

    Args:
        loss_weight: Scalar weight for this loss component.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, loss_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked L1 regression loss.

        Args:
            pred: (B, C, H, W) predicted regression values.
            target: (B, C, H, W) ground truth regression targets.
            mask: (B, H, W) or (B, 1, H, W) binary mask indicating positive locations.

        Returns:
            Scalar loss value.
        """
        # Ensure mask has channel dimension
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)

        # Expand mask to match prediction channels
        mask = mask.expand_as(pred)  # (B, C, H, W)

        # Compute L1 loss only at masked locations
        num_pos = mask.float().sum().clamp(min=1.0)

        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / num_pos

        return loss * self.loss_weight


class SmoothRegLoss(nn.Module):
    """Smooth L1 (Huber) regression loss at positive locations.

    Args:
        loss_weight: Scalar weight for this loss component.
        beta: Threshold for switching between L1 and L2 behavior.
    """

    def __init__(self, loss_weight: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked Smooth L1 loss.

        Args:
            pred: (B, C, H, W) predictions.
            target: (B, C, H, W) targets.
            mask: (B, H, W) binary mask.

        Returns:
            Scalar loss.
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(pred)

        num_pos = mask.float().sum().clamp(min=1.0)

        diff = torch.abs(pred - target) * mask
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        loss = loss.sum() / num_pos

        return loss * self.loss_weight


class CenterPointLoss(nn.Module):
    """Combined CenterPoint training loss.

    Computes the total loss as a weighted sum of:
    - Heatmap loss: Gaussian focal loss for center detection
    - Offset loss: L1 loss for sub-voxel center offset
    - Height loss: L1 loss for absolute z prediction
    - Dimension loss: L1 loss for log-normalized size
    - Rotation loss: L1 loss for sin/cos of yaw
    - Velocity loss: L1 loss for vx, vy prediction

    Args:
        heatmap_weight: Weight for heatmap focal loss.
        offset_weight: Weight for offset regression loss.
        height_weight: Weight for height regression loss.
        dim_weight: Weight for dimension regression loss.
        rot_weight: Weight for rotation regression loss.
        vel_weight: Weight for velocity regression loss.
        focal_alpha: Alpha parameter for Gaussian focal loss.
        focal_beta: Beta parameter for Gaussian focal loss.
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        offset_weight: float = 2.0,
        height_weight: float = 0.25,
        dim_weight: float = 0.2,
        rot_weight: float = 1.0,
        vel_weight: float = 0.2,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
    ):
        super().__init__()

        self.heatmap_loss_fn = GaussianFocalLoss(
            alpha=focal_alpha, beta=focal_beta, loss_weight=heatmap_weight
        )
        self.offset_loss_fn = RegLoss(loss_weight=offset_weight)
        self.height_loss_fn = RegLoss(loss_weight=height_weight)
        self.dim_loss_fn = RegLoss(loss_weight=dim_weight)
        self.rot_loss_fn = RegLoss(loss_weight=rot_weight)
        self.vel_loss_fn = RegLoss(loss_weight=vel_weight)

    def forward(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute total CenterPoint loss across all tasks.

        Args:
            predictions: List of prediction dicts per task from CenterHead.forward().
                         Each has: 'heatmap', 'offset', 'height', 'dim', 'rot', 'vel'.
            targets: List of target dicts per task from CenterHead.generate_targets().
                     Each has: 'heatmap', 'offset', 'height', 'dim', 'rot', 'vel', 'mask'.

        Returns:
            Dict with individual and total loss values:
            - 'heatmap_loss': sum of heatmap losses across tasks
            - 'offset_loss': sum of offset losses
            - 'height_loss': sum of height losses
            - 'dim_loss': sum of dimension losses
            - 'rot_loss': sum of rotation losses
            - 'vel_loss': sum of velocity losses
            - 'total_loss': weighted sum of all losses
        """
        total_heatmap_loss = torch.tensor(0.0, device=predictions[0]['heatmap'].device)
        total_offset_loss = torch.tensor(0.0, device=predictions[0]['heatmap'].device)
        total_height_loss = torch.tensor(0.0, device=predictions[0]['heatmap'].device)
        total_dim_loss = torch.tensor(0.0, device=predictions[0]['heatmap'].device)
        total_rot_loss = torch.tensor(0.0, device=predictions[0]['heatmap'].device)
        total_vel_loss = torch.tensor(0.0, device=predictions[0]['heatmap'].device)

        num_tasks = len(predictions)

        for task_idx in range(num_tasks):
            pred = predictions[task_idx]
            tgt = targets[task_idx]

            # Heatmap loss (already sigmoided in CenterHead)
            heatmap_loss = self.heatmap_loss_fn(pred['heatmap'], tgt['heatmap'])
            total_heatmap_loss = total_heatmap_loss + heatmap_loss

            # Regression losses (only at positive locations)
            mask = tgt['mask']  # (B, H, W)

            offset_loss = self.offset_loss_fn(pred['offset'], tgt['offset'], mask)
            total_offset_loss = total_offset_loss + offset_loss

            height_loss = self.height_loss_fn(pred['height'], tgt['height'], mask)
            total_height_loss = total_height_loss + height_loss

            dim_loss = self.dim_loss_fn(pred['dim'], tgt['dim'], mask)
            total_dim_loss = total_dim_loss + dim_loss

            rot_loss = self.rot_loss_fn(pred['rot'], tgt['rot'], mask)
            total_rot_loss = total_rot_loss + rot_loss

            vel_loss = self.vel_loss_fn(pred['vel'], tgt['vel'], mask)
            total_vel_loss = total_vel_loss + vel_loss

        # Average over tasks
        total_heatmap_loss = total_heatmap_loss / num_tasks
        total_offset_loss = total_offset_loss / num_tasks
        total_height_loss = total_height_loss / num_tasks
        total_dim_loss = total_dim_loss / num_tasks
        total_rot_loss = total_rot_loss / num_tasks
        total_vel_loss = total_vel_loss / num_tasks

        # Total loss
        total_loss = (
            total_heatmap_loss +
            total_offset_loss +
            total_height_loss +
            total_dim_loss +
            total_rot_loss +
            total_vel_loss
        )

        return {
            'heatmap_loss': total_heatmap_loss,
            'offset_loss': total_offset_loss,
            'height_loss': total_height_loss,
            'dim_loss': total_dim_loss,
            'rot_loss': total_rot_loss,
            'vel_loss': total_vel_loss,
            'total_loss': total_loss,
        }


class IoULoss(nn.Module):
    """IoU-based loss for 3D bounding box regression (optional auxiliary loss).

    Computes axis-aligned BEV IoU between predicted and target boxes at
    positive locations, then uses 1 - IoU as the loss.

    Args:
        loss_weight: Scalar weight.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU loss for matched predicted and target boxes.

        Args:
            pred_boxes: (N, 7+) predicted boxes [x, y, z, w, h, l, yaw, ...].
            target_boxes: (N, 7+) target boxes [x, y, z, w, h, l, yaw, ...].

        Returns:
            Scalar IoU loss (1 - mean IoU).
        """
        if pred_boxes.shape[0] == 0:
            return torch.tensor(0.0, device=pred_boxes.device)

        # BEV IoU (axis-aligned approximation)
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 3] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 5] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 5] / 2

        tgt_x1 = target_boxes[:, 0] - target_boxes[:, 3] / 2
        tgt_x2 = target_boxes[:, 0] + target_boxes[:, 3] / 2
        tgt_y1 = target_boxes[:, 1] - target_boxes[:, 5] / 2
        tgt_y2 = target_boxes[:, 1] + target_boxes[:, 5] / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_area = (
            (inter_x2 - inter_x1).clamp(min=0) *
            (inter_y2 - inter_y1).clamp(min=0)
        )

        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)
        union_area = pred_area + tgt_area - inter_area + 1e-6

        iou = inter_area / union_area
        loss = (1.0 - iou).mean()

        return loss * self.loss_weight
