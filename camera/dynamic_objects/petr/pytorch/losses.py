"""
Loss functions for PETR 3D object detection.

Implements Hungarian bipartite matching, focal loss for classification,
L1 loss for bounding box regression, and combined loss computation
with configurable weights.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive examples.
        gamma: Focusing parameter that reduces loss for well-classified examples.
        reduction: Reduction mode ('none', 'mean', 'sum').
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Predicted logits (B, Q, num_classes) or (N, num_classes).
            target: Target class indices (B, Q) or (N,). Use num_classes
                index for background/no-object class.

        Returns:
            Focal loss value.
        """
        num_classes = pred.shape[-1]

        # Convert targets to one-hot
        # target shape: (...,) with values in [0, num_classes]
        # We treat num_classes as the background class index
        target_one_hot = F.one_hot(
            target.long(), num_classes=num_classes + 1
        )[..., :num_classes].float()  # exclude background from one-hot

        # Compute sigmoid probabilities
        pred_sigmoid = pred.sigmoid()

        # Compute focal weight
        p_t = pred_sigmoid * target_one_hot + (1 - pred_sigmoid) * (1 - target_one_hot)
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha weight
        alpha_t = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)

        # Binary cross-entropy loss (per-class)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target_one_hot, reduction="none"
        )

        # Apply focal and alpha weights
        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class L1Loss(nn.Module):
    """Smooth L1 loss (Huber loss) for bounding box regression.

    Args:
        beta: Threshold for switching between L1 and L2. If 0, uses
            standard L1 loss.
        reduction: Reduction mode ('none', 'mean', 'sum').
    """

    def __init__(
        self,
        beta: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute L1 or smooth L1 loss.

        Args:
            pred: Predictions (..., D).
            target: Targets (..., D).
            weight: Optional per-element weights (..., D) or (...,).

        Returns:
            Loss value.
        """
        if self.beta > 0:
            loss = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
        else:
            loss = F.l1_loss(pred, target, reduction="none")

        if weight is not None:
            if weight.dim() < loss.dim():
                weight = weight.unsqueeze(-1)
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground-truth targets.

    Uses the Hungarian algorithm to find optimal one-to-one assignment
    between predicted objects and ground-truth annotations. The matching
    cost combines classification cost and bounding box regression cost.

    Args:
        cls_weight: Weight for classification cost in matching.
        bbox_weight: Weight for L1 bounding box cost in matching.
        iou_weight: Weight for IoU cost in matching (not used for 3D,
            kept for interface compatibility).
    """

    def __init__(
        self,
        cls_weight: float = 2.0,
        bbox_weight: float = 5.0,
        iou_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.iou_weight = iou_weight

    @torch.no_grad()
    def forward(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_labels: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Perform Hungarian matching for a batch.

        Args:
            cls_scores: Predicted class scores (B, Q, num_classes).
            bbox_preds: Predicted bboxes (B, Q, code_size).
            gt_labels: List of ground-truth labels, one tensor per sample.
                Each tensor has shape (num_gt,) with class indices.
            gt_bboxes: List of ground-truth bboxes, one tensor per sample.
                Each tensor has shape (num_gt, code_size).

        Returns:
            List of (pred_indices, gt_indices) tuples for each sample in
            the batch. pred_indices[i] is matched to gt_indices[i].
        """
        batch_size, num_queries = cls_scores.shape[:2]
        device = cls_scores.device

        indices = []

        for b in range(batch_size):
            if gt_labels[b].numel() == 0:
                # No ground truth: empty matching
                indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=device),
                        torch.tensor([], dtype=torch.long, device=device),
                    )
                )
                continue

            # Classification cost: use focal-loss-aware cost
            # -alpha * (1-p)^gamma * log(p) for positive class
            pred_scores = cls_scores[b].sigmoid()  # (Q, num_classes)
            gt_cls = gt_labels[b].long()  # (num_gt,)

            # Gather predicted probabilities for ground-truth classes
            # Cost is negative log-likelihood with focal weighting
            alpha = 0.25
            gamma = 2.0

            # For each query-gt pair, compute focal classification cost
            neg_cost = (
                -(1 - alpha) * (pred_scores ** gamma) * (
                    (1 - pred_scores + 1e-8).log()
                )
            )  # (Q, num_classes)
            pos_cost = (
                -alpha * ((1 - pred_scores) ** gamma) * (
                    (pred_scores + 1e-8).log()
                )
            )  # (Q, num_classes)

            cls_cost = pos_cost[:, gt_cls] - neg_cost[:, gt_cls]  # (Q, num_gt)

            # Bounding box L1 cost
            # bbox_preds[b]: (Q, code_size), gt_bboxes[b]: (num_gt, code_size)
            bbox_cost = torch.cdist(
                bbox_preds[b].float(), gt_bboxes[b].float(), p=1
            )  # (Q, num_gt)

            # Combined cost matrix
            cost_matrix = (
                self.cls_weight * cls_cost + self.bbox_weight * bbox_cost
            )

            # Solve assignment problem
            cost_numpy = cost_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_numpy)

            indices.append(
                (
                    torch.tensor(row_ind, dtype=torch.long, device=device),
                    torch.tensor(col_ind, dtype=torch.long, device=device),
                )
            )

        return indices


class PETRLoss(nn.Module):
    """Combined loss function for PETR 3D object detection.

    Computes classification loss (focal), bounding box regression loss (L1),
    and velocity loss (L1) with Hungarian matching assignment.

    Args:
        num_classes: Number of object classes.
        cls_weight: Weight for classification loss.
        bbox_weight: Weight for bbox regression loss.
        velocity_weight: Weight for velocity loss.
        code_size: Dimension of bbox code.
        match_cls_weight: Classification cost weight for matching.
        match_bbox_weight: Bbox cost weight for matching.
        pc_range: Point cloud range for bbox normalization.
    """

    def __init__(
        self,
        num_classes: int = 10,
        cls_weight: float = 2.0,
        bbox_weight: float = 0.25,
        velocity_weight: float = 0.25,
        code_size: int = 10,
        match_cls_weight: float = 2.0,
        match_bbox_weight: float = 5.0,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.velocity_weight = velocity_weight
        self.code_size = code_size
        self.pc_range = pc_range

        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        self.l1_loss = L1Loss(beta=0.0, reduction="none")

        # Matcher
        self.matcher = HungarianMatcher(
            cls_weight=match_cls_weight,
            bbox_weight=match_bbox_weight,
        )

    def _get_target_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get targets for a single sample after matching.

        Args:
            cls_scores: (Q, num_classes)
            bbox_preds: (Q, code_size)
            gt_labels: (num_gt,)
            gt_bboxes: (num_gt, code_size)

        Returns:
            Tuple of (target_labels, target_bboxes, pos_mask, neg_mask).
        """
        num_queries = cls_scores.shape[0]
        device = cls_scores.device

        # Perform matching for this sample
        indices = self.matcher(
            cls_scores.unsqueeze(0),
            bbox_preds.unsqueeze(0),
            [gt_labels],
            [gt_bboxes],
        )[0]

        pred_indices, gt_indices = indices

        # Initialize targets: background class for all queries
        target_labels = torch.full(
            (num_queries,), self.num_classes, dtype=torch.long, device=device
        )
        target_bboxes = torch.zeros(
            num_queries, self.code_size, dtype=bbox_preds.dtype, device=device
        )

        # Assign matched ground truths
        if len(pred_indices) > 0:
            target_labels[pred_indices] = gt_labels[gt_indices]
            target_bboxes[pred_indices] = gt_bboxes[gt_indices]

        # Positive/negative masks
        pos_mask = target_labels < self.num_classes
        neg_mask = ~pos_mask

        return target_labels, target_bboxes, pos_mask, neg_mask

    def forward(
        self,
        cls_scores_list: List[torch.Tensor],
        bbox_preds_list: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss across all decoder layers.

        Args:
            cls_scores_list: List of classification outputs per decoder layer.
                Each has shape (B, Q, num_classes).
            bbox_preds_list: List of bbox predictions per decoder layer.
                Each has shape (B, Q, code_size).
            gt_labels: Ground-truth labels per sample.
                List of (num_gt_i,) tensors.
            gt_bboxes: Ground-truth bboxes per sample.
                List of (num_gt_i, code_size) tensors.

        Returns:
            Dictionary of loss components:
                'loss_cls': Total classification loss.
                'loss_bbox': Total bounding box regression loss.
                'loss_velocity': Total velocity loss.
                'loss_total': Weighted sum of all losses.
        """
        num_layers = len(cls_scores_list)
        device = cls_scores_list[0].device

        total_cls_loss = torch.tensor(0.0, device=device)
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_vel_loss = torch.tensor(0.0, device=device)

        for layer_idx in range(num_layers):
            cls_scores = cls_scores_list[layer_idx]  # (B, Q, num_classes)
            bbox_preds = bbox_preds_list[layer_idx]  # (B, Q, code_size)
            B, Q, _ = cls_scores.shape

            layer_cls_loss = torch.tensor(0.0, device=device)
            layer_bbox_loss = torch.tensor(0.0, device=device)
            layer_vel_loss = torch.tensor(0.0, device=device)
            num_pos_total = 0

            for b in range(B):
                target_labels, target_bboxes, pos_mask, neg_mask = (
                    self._get_target_single(
                        cls_scores[b], bbox_preds[b], gt_labels[b], gt_bboxes[b]
                    )
                )

                num_pos = pos_mask.sum().item()
                num_pos_total += num_pos

                # Classification loss (all queries)
                cls_loss = self.focal_loss(cls_scores[b], target_labels)
                layer_cls_loss = layer_cls_loss + cls_loss.sum()

                # Bbox regression loss (positive queries only)
                if num_pos > 0:
                    # Separate bbox components
                    pred_bbox_pos = bbox_preds[b][pos_mask]  # (num_pos, code_size)
                    target_bbox_pos = target_bboxes[pos_mask]  # (num_pos, code_size)

                    # Bbox loss on center, size, rotation (indices 0-7)
                    bbox_loss = self.l1_loss(
                        pred_bbox_pos[..., :8], target_bbox_pos[..., :8]
                    )
                    layer_bbox_loss = layer_bbox_loss + bbox_loss.sum()

                    # Velocity loss (indices 8-9: vx, vy)
                    if self.code_size >= 10:
                        vel_loss = self.l1_loss(
                            pred_bbox_pos[..., 8:10], target_bbox_pos[..., 8:10]
                        )
                        layer_vel_loss = layer_vel_loss + vel_loss.sum()

            # Normalize by number of positive samples
            num_pos_total = max(num_pos_total, 1)
            total_cls_loss = total_cls_loss + layer_cls_loss / num_pos_total
            total_bbox_loss = total_bbox_loss + layer_bbox_loss / num_pos_total
            total_vel_loss = total_vel_loss + layer_vel_loss / num_pos_total

        # Average over layers
        total_cls_loss = total_cls_loss / num_layers
        total_bbox_loss = total_bbox_loss / num_layers
        total_vel_loss = total_vel_loss / num_layers

        # Weighted total loss
        loss_total = (
            self.cls_weight * total_cls_loss
            + self.bbox_weight * total_bbox_loss
            + self.velocity_weight * total_vel_loss
        )

        return {
            "loss_cls": total_cls_loss,
            "loss_bbox": total_bbox_loss,
            "loss_velocity": total_vel_loss,
            "loss_total": loss_total,
        }
