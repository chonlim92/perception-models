"""
DETR3D Losses: Hungarian matching, focal loss, and L1 regression loss.

Implements the bipartite matching loss used in DETR-style detectors:
1. Hungarian matching to assign predictions to ground truth
2. Focal loss for classification
3. L1 loss for 3D bounding box regression
4. Support for auxiliary losses from intermediate decoder layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Alpha-balanced focal loss for classification.

    Focal loss down-weights well-classified examples and focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        pred: Predicted logits, shape (..., num_classes).
        target: Target class indices, shape (...). Values in [0, num_classes-1]
                for foreground, or num_classes for background.
        alpha: Weighting factor for the rare class (foreground).
        gamma: Focusing parameter that reduces loss for well-classified examples.
        reduction: 'none', 'mean', or 'sum'.

    Returns:
        Focal loss value.
    """
    num_classes = pred.shape[-1]

    # Convert target to one-hot
    # For targets == num_classes (background), all zeros
    target_one_hot = torch.zeros_like(pred)
    valid_mask = target < num_classes
    if valid_mask.any():
        valid_targets = target[valid_mask].long()
        target_one_hot[valid_mask] = F.one_hot(
            valid_targets, num_classes=num_classes
        ).float()

    # Compute probabilities
    pred_sigmoid = torch.sigmoid(pred)

    # Compute focal weight
    # p_t = p if y=1 else (1-p)
    p_t = pred_sigmoid * target_one_hot + (1 - pred_sigmoid) * (1 - target_one_hot)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting
    alpha_t = alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)

    # Binary cross entropy (per class)
    bce = F.binary_cross_entropy_with_logits(pred, target_one_hot, reduction='none')

    # Focal loss
    loss = alpha_t * focal_weight * bce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """L1 loss for bounding box regression.

    Args:
        pred: Predicted values, shape (...).
        target: Target values, shape (...).
        reduction: 'none', 'mean', or 'sum'.

    Returns:
        L1 loss value.
    """
    loss = F.l1_loss(pred, target, reduction=reduction)
    return loss


def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Smooth L1 (Huber) loss for bounding box regression.

    Args:
        pred: Predicted values, shape (...).
        target: Target values, shape (...).
        beta: Threshold for switching between L1 and L2.
        reduction: 'none', 'mean', or 'sum'.

    Returns:
        Smooth L1 loss value.
    """
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)
    return loss


class HungarianMatcher(nn.Module):
    """Hungarian matcher for bipartite matching between predictions and GT.

    Computes a cost matrix considering classification cost, L1 bbox cost,
    and an IoU-like cost, then uses scipy's linear_sum_assignment to find
    the optimal matching.
    """

    def __init__(
        self,
        cls_cost_weight: float = 2.0,
        bbox_cost_weight: float = 5.0,
        iou_cost_weight: float = 2.0,
    ):
        """
        Args:
            cls_cost_weight: Weight for classification cost in matching.
            bbox_cost_weight: Weight for L1 bbox regression cost.
            iou_cost_weight: Weight for IoU-like cost (center distance).
        """
        super().__init__()
        self.cls_cost_weight = cls_cost_weight
        self.bbox_cost_weight = bbox_cost_weight
        self.iou_cost_weight = iou_cost_weight

    @torch.no_grad()
    def forward(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_labels: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute bipartite matching between predictions and ground truth.

        Args:
            cls_scores: Predicted class logits, shape (B, N, num_classes).
            bbox_preds: Predicted bboxes, shape (B, N, 10).
            gt_labels: List of B tensors, each (M_i,) with class indices.
            gt_bboxes: List of B tensors, each (M_i, 10) with GT bbox params.

        Returns:
            List of B tuples (pred_indices, gt_indices) representing the
            matched pairs for each batch element.
        """
        batch_size, num_queries, num_classes = cls_scores.shape

        indices = []
        for b in range(batch_size):
            if gt_labels[b].numel() == 0:
                # No ground truth: return empty matching
                indices.append((
                    torch.tensor([], dtype=torch.long, device=cls_scores.device),
                    torch.tensor([], dtype=torch.long, device=cls_scores.device),
                ))
                continue

            # Get predictions for this batch element
            pred_cls = cls_scores[b]  # (N, num_classes)
            pred_bbox = bbox_preds[b]  # (N, 10)

            # Get GT for this batch element
            tgt_labels = gt_labels[b]  # (M,)
            tgt_bboxes = gt_bboxes[b]  # (M, 10)

            num_gt = tgt_labels.shape[0]

            # Classification cost: negative focal-weighted probability
            pred_prob = pred_cls.sigmoid()  # (N, num_classes)
            # Cost is -probability of the target class
            # Use focal-like weighting
            alpha = 0.25
            gamma = 2.0
            neg_cost = (1 - alpha) * (pred_prob ** gamma) * (
                -(1 - pred_prob + 1e-8).log()
            )
            pos_cost = alpha * ((1 - pred_prob) ** gamma) * (
                -(pred_prob + 1e-8).log()
            )
            cls_cost = pos_cost[:, tgt_labels.long()] - neg_cost[:, tgt_labels.long()]  # (N, M)

            # L1 bbox cost: L1 distance between predicted and GT bboxes
            # Compare all 10 components
            bbox_cost = torch.cdist(pred_bbox, tgt_bboxes, p=1)  # (N, M)

            # IoU-like cost: use center distance as a proxy for 3D IoU
            # (computing true 3D IoU is expensive, center distance is standard in DETR3D)
            pred_centers = pred_bbox[:, :3]  # (N, 3)
            gt_centers = tgt_bboxes[:, :3]  # (M, 3)
            center_dist = torch.cdist(pred_centers, gt_centers, p=2)  # (N, M)

            # Combined cost matrix
            cost_matrix = (
                self.cls_cost_weight * cls_cost +
                self.bbox_cost_weight * bbox_cost +
                self.iou_cost_weight * center_dist
            )

            # Hungarian matching (scipy)
            cost_np = cost_matrix.detach().cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(cost_np)

            indices.append((
                torch.tensor(pred_idx, dtype=torch.long, device=cls_scores.device),
                torch.tensor(gt_idx, dtype=torch.long, device=cls_scores.device),
            ))

        return indices


class DETR3DLoss(nn.Module):
    """Complete DETR3D loss combining Hungarian matching with focal + L1 losses.

    Supports auxiliary losses from intermediate decoder layers.
    """

    def __init__(
        self,
        num_classes: int = 10,
        cls_weight: float = 2.0,
        bbox_weight: float = 0.25,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        aux_loss_weight: float = 1.0,
        matcher_cls_cost: float = 2.0,
        matcher_bbox_cost: float = 5.0,
        matcher_iou_cost: float = 2.0,
    ):
        """
        Args:
            num_classes: Number of foreground classes.
            cls_weight: Weight for classification loss.
            bbox_weight: Weight for bbox regression loss.
            focal_alpha: Alpha parameter for focal loss.
            focal_gamma: Gamma parameter for focal loss.
            aux_loss_weight: Weight multiplier for auxiliary losses.
            matcher_cls_cost: Classification cost weight in Hungarian matcher.
            matcher_bbox_cost: Bbox cost weight in Hungarian matcher.
            matcher_iou_cost: IoU cost weight in Hungarian matcher.
        """
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_loss_weight = aux_loss_weight

        self.matcher = HungarianMatcher(
            cls_cost_weight=matcher_cls_cost,
            bbox_cost_weight=matcher_bbox_cost,
            iou_cost_weight=matcher_iou_cost,
        )

    def _compute_loss_single_layer(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_labels: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for a single decoder layer's predictions.

        Args:
            cls_scores: (B, N, num_classes).
            bbox_preds: (B, N, 10).
            gt_labels: List of B tensors with GT class indices.
            gt_bboxes: List of B tensors with GT bbox params.

        Returns:
            Dict with 'loss_cls' and 'loss_bbox'.
        """
        batch_size, num_queries, _ = cls_scores.shape
        device = cls_scores.device

        # Run Hungarian matching
        indices = self.matcher(cls_scores, bbox_preds, gt_labels, gt_bboxes)

        # Prepare classification targets
        # Default target is num_classes (background)
        cls_targets = torch.full(
            (batch_size, num_queries), self.num_classes,
            dtype=torch.long, device=device
        )

        # Assign matched GT labels
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                cls_targets[b, pred_idx] = gt_labels[b][gt_idx].long()

        # Classification loss (focal loss)
        loss_cls = focal_loss(
            cls_scores.reshape(-1, self.num_classes),
            cls_targets.reshape(-1),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction='mean',
        )

        # Bbox regression loss (only for matched predictions)
        loss_bbox = torch.tensor(0.0, device=device)
        num_pos = 0

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue

            # Get matched predictions and targets
            matched_pred = bbox_preds[b, pred_idx]  # (M, 10)
            matched_gt = gt_bboxes[b][gt_idx]  # (M, 10)

            # L1 loss on all bbox components
            # Weight different components: center(3), size(3), rotation(2), velocity(2)
            center_loss = l1_loss(matched_pred[:, :3], matched_gt[:, :3], reduction='sum')
            size_loss = l1_loss(matched_pred[:, 3:6], matched_gt[:, 3:6], reduction='sum')
            rot_loss = l1_loss(matched_pred[:, 6:8], matched_gt[:, 6:8], reduction='sum')
            vel_loss = l1_loss(matched_pred[:, 8:10], matched_gt[:, 8:10], reduction='sum')

            loss_bbox = loss_bbox + center_loss + size_loss + rot_loss + vel_loss
            num_pos += pred_idx.numel()

        # Normalize by number of positive samples
        if num_pos > 0:
            loss_bbox = loss_bbox / num_pos
        else:
            loss_bbox = loss_bbox * 0.0  # Keep gradient flow

        return {
            'loss_cls': loss_cls * self.cls_weight,
            'loss_bbox': loss_bbox * self.bbox_weight,
        }

    def forward(
        self,
        cls_scores_list: List[torch.Tensor],
        bbox_preds_list: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute total DETR3D loss including auxiliary losses.

        Args:
            cls_scores_list: List of L tensors (one per decoder layer),
                             each (B, N, num_classes).
            bbox_preds_list: List of L tensors (one per decoder layer),
                             each (B, N, 10).
            gt_labels: List of B tensors, each (M_i,) with GT class indices.
            gt_bboxes: List of B tensors, each (M_i, 10) with GT bbox params
                       [cx, cy, cz, w, l, h, sin(yaw), cos(yaw), vx, vy].

        Returns:
            Dictionary of loss components:
                'loss_cls': Final layer classification loss.
                'loss_bbox': Final layer regression loss.
                'loss_cls_aux_0', ..., 'loss_cls_aux_{L-2}': Auxiliary cls losses.
                'loss_bbox_aux_0', ..., 'loss_bbox_aux_{L-2}': Auxiliary bbox losses.
                'total_loss': Sum of all weighted losses.
        """
        num_layers = len(cls_scores_list)
        device = cls_scores_list[0].device

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)

        # Final layer loss (last decoder layer)
        final_losses = self._compute_loss_single_layer(
            cls_scores_list[-1], bbox_preds_list[-1], gt_labels, gt_bboxes
        )
        loss_dict['loss_cls'] = final_losses['loss_cls']
        loss_dict['loss_bbox'] = final_losses['loss_bbox']
        total_loss = total_loss + final_losses['loss_cls'] + final_losses['loss_bbox']

        # Auxiliary losses from intermediate layers
        for layer_idx in range(num_layers - 1):
            aux_losses = self._compute_loss_single_layer(
                cls_scores_list[layer_idx], bbox_preds_list[layer_idx],
                gt_labels, gt_bboxes
            )
            aux_cls_key = f'loss_cls_aux_{layer_idx}'
            aux_bbox_key = f'loss_bbox_aux_{layer_idx}'

            loss_dict[aux_cls_key] = aux_losses['loss_cls'] * self.aux_loss_weight
            loss_dict[aux_bbox_key] = aux_losses['loss_bbox'] * self.aux_loss_weight

            total_loss = total_loss + loss_dict[aux_cls_key] + loss_dict[aux_bbox_key]

        loss_dict['total_loss'] = total_loss

        return loss_dict
