"""BEVFormer Losses: Hungarian matching and detection losses.

Implements the bipartite matching strategy from DETR for assigning predicted
object queries to ground-truth objects, along with focal loss for classification
and L1 loss for 3D bounding box regression. Supports auxiliary losses from
intermediate decoder layers with configurable weighting.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

__all__ = ["HungarianMatcher", "BEVFormerLoss"]


class HungarianMatcher(nn.Module):
    """Hungarian Matcher for optimal bipartite matching between predictions and targets.

    Computes a cost matrix based on classification cost (focal-loss-based),
    bounding box L1 cost, and optional IoU cost, then solves the linear sum
    assignment problem to find the optimal one-to-one matching.
    """

    def __init__(
        self,
        cls_cost_weight: float = 2.0,
        bbox_cost_weight: float = 5.0,
        iou_cost_weight: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        """Initialize Hungarian matcher.

        Args:
            cls_cost_weight: Weight for classification cost in the cost matrix.
            bbox_cost_weight: Weight for L1 bounding box cost in the cost matrix.
            iou_cost_weight: Weight for IoU cost (set to 0 to disable).
            focal_alpha: Alpha parameter for focal loss cost computation.
            focal_gamma: Gamma parameter for focal loss cost computation.
        """
        super().__init__()
        self.cls_cost_weight = cls_cost_weight
        self.bbox_cost_weight = bbox_cost_weight
        self.iou_cost_weight = iou_cost_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        assert cls_cost_weight > 0 or bbox_cost_weight > 0 or iou_cost_weight > 0, (
            "At least one cost weight must be positive"
        )

    @torch.no_grad()
    def forward(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_labels: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute optimal bipartite matching between predictions and ground truth.

        Args:
            cls_scores: Predicted classification logits,
                shape (B, num_queries, num_classes).
            bbox_preds: Predicted bounding boxes,
                shape (B, num_queries, num_reg_params).
            gt_labels: List of ground-truth class labels per sample.
                Each tensor has shape (num_gt_i,) with integer class indices.
            gt_bboxes: List of ground-truth bounding boxes per sample.
                Each tensor has shape (num_gt_i, num_reg_params).

        Returns:
            List of tuples (pred_indices, gt_indices) per sample in the batch.
            pred_indices: indices of matched predictions (shape: num_gt_i,)
            gt_indices: indices of matched ground truths (shape: num_gt_i,)
        """
        batch_size, num_queries, num_classes = cls_scores.shape

        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for batch_idx in range(batch_size):
            # Get predictions for this sample
            pred_cls = cls_scores[batch_idx]  # (num_queries, num_classes)
            pred_bbox = bbox_preds[batch_idx]  # (num_queries, num_reg_params)

            # Get ground truth for this sample
            tgt_labels = gt_labels[batch_idx]  # (num_gt,)
            tgt_bboxes = gt_bboxes[batch_idx]  # (num_gt, num_reg_params)

            num_gt = tgt_labels.shape[0]

            if num_gt == 0:
                # No ground truth: return empty indices
                device = pred_cls.device
                indices.append((
                    torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device),
                ))
                continue

            # Classification cost using focal loss formulation
            # Convert logits to probabilities
            pred_probs = torch.sigmoid(pred_cls)  # (num_queries, num_classes)

            # Focal-loss-based cost for classification
            # For matched class c_j of GT j:
            # cost_cls = -alpha * (1 - p)^gamma * log(p) for positive
            #          = -(1 - alpha) * p^gamma * log(1 - p) for negative
            # We compute the negative cost (want to minimize)
            neg_cost = (
                -(1 - self.focal_alpha)
                * (pred_probs ** self.focal_gamma)
                * (-(1 - pred_probs + 1e-8).log())
            )
            pos_cost = (
                -self.focal_alpha
                * ((1 - pred_probs) ** self.focal_gamma)
                * (-(pred_probs + 1e-8).log())
            )

            # Select costs for target classes: (num_queries, num_gt)
            cls_cost = pos_cost[:, tgt_labels] - neg_cost[:, tgt_labels]

            # Bounding box L1 cost: (num_queries, num_gt)
            # Compute pairwise L1 distance between all predictions and all GTs
            bbox_cost = torch.cdist(
                pred_bbox.float(), tgt_bboxes.float(), p=1.0
            )

            # IoU cost (optional, for BEV IoU)
            if self.iou_cost_weight > 0:
                iou_cost = self._compute_bev_iou_cost(pred_bbox, tgt_bboxes)
            else:
                iou_cost = torch.zeros_like(bbox_cost)

            # Total cost matrix
            cost_matrix = (
                self.cls_cost_weight * cls_cost
                + self.bbox_cost_weight * bbox_cost
                + self.iou_cost_weight * iou_cost
            )

            # Solve assignment using scipy (on CPU)
            cost_np = cost_matrix.detach().cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(cost_np)

            indices.append((
                torch.tensor(pred_idx, dtype=torch.long, device=pred_cls.device),
                torch.tensor(gt_idx, dtype=torch.long, device=pred_cls.device),
            ))

        return indices

    @staticmethod
    def _compute_bev_iou_cost(
        pred_bbox: torch.Tensor,
        gt_bbox: torch.Tensor,
    ) -> torch.Tensor:
        """Compute approximate BEV IoU cost between predictions and ground truth.

        Uses axis-aligned BEV bounding boxes (cx, cy, w, l) for a fast
        approximation of rotated IoU cost.

        Args:
            pred_bbox: Predicted boxes, shape (num_queries, num_reg_params).
                Uses params [0]=cx, [1]=cy, [3]=w, [4]=l.
            gt_bbox: Ground truth boxes, shape (num_gt, num_reg_params).

        Returns:
            Negative IoU cost matrix, shape (num_queries, num_gt).
        """
        # Extract BEV parameters (cx, cy, w, l)
        pred_cx, pred_cy = pred_bbox[:, 0], pred_bbox[:, 1]
        pred_w, pred_l = pred_bbox[:, 3].abs(), pred_bbox[:, 4].abs()

        gt_cx, gt_cy = gt_bbox[:, 0], gt_bbox[:, 1]
        gt_w, gt_l = gt_bbox[:, 3].abs(), gt_bbox[:, 4].abs()

        # Convert to corners for axis-aligned IoU
        pred_x1 = pred_cx - pred_w / 2  # (num_queries,)
        pred_x2 = pred_cx + pred_w / 2
        pred_y1 = pred_cy - pred_l / 2
        pred_y2 = pred_cy + pred_l / 2

        gt_x1 = gt_cx - gt_w / 2  # (num_gt,)
        gt_x2 = gt_cx + gt_w / 2
        gt_y1 = gt_cy - gt_l / 2
        gt_y2 = gt_cy + gt_l / 2

        # Pairwise intersection: (num_queries, num_gt)
        inter_x1 = torch.max(pred_x1.unsqueeze(1), gt_x1.unsqueeze(0))
        inter_x2 = torch.min(pred_x2.unsqueeze(1), gt_x2.unsqueeze(0))
        inter_y1 = torch.max(pred_y1.unsqueeze(1), gt_y1.unsqueeze(0))
        inter_y2 = torch.min(pred_y2.unsqueeze(1), gt_y2.unsqueeze(0))

        inter_area = (
            (inter_x2 - inter_x1).clamp(min=0)
            * (inter_y2 - inter_y1).clamp(min=0)
        )

        # Areas
        pred_area = (pred_w * pred_l).unsqueeze(1)  # (num_queries, 1)
        gt_area = (gt_w * gt_l).unsqueeze(0)  # (1, num_gt)

        union_area = pred_area + gt_area - inter_area
        iou = inter_area / (union_area + 1e-8)

        # Return negative IoU as cost (lower is better)
        return -iou


class BEVFormerLoss(nn.Module):
    """BEVFormer detection loss combining focal loss and L1 regression loss.

    Computes:
      - Focal loss for classification with configurable alpha and gamma.
      - L1 loss for all 10 bounding box regression parameters.
      - Auxiliary losses from intermediate decoder layers (same loss, equal weight).

    Loss is normalized by the total number of positive (matched) samples across
    the batch and all decoder layers.
    """

    def __init__(
        self,
        num_classes: int = 10,
        cls_weight: float = 2.0,
        bbox_weight: float = 0.25,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        matcher_cls_cost: float = 2.0,
        matcher_bbox_cost: float = 5.0,
        matcher_iou_cost: float = 0.0,
    ) -> None:
        """Initialize BEVFormer loss.

        Args:
            num_classes: Number of object classes.
            cls_weight: Weight multiplier for classification loss.
            bbox_weight: Weight multiplier for bounding box regression loss.
            focal_alpha: Alpha for focal loss (balances positive/negative).
            focal_gamma: Gamma for focal loss (focuses on hard examples).
            matcher_cls_cost: Classification cost weight for Hungarian matcher.
            matcher_bbox_cost: Bounding box cost weight for Hungarian matcher.
            matcher_iou_cost: IoU cost weight for Hungarian matcher.
        """
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Hungarian matcher for bipartite matching
        self.matcher = HungarianMatcher(
            cls_cost_weight=matcher_cls_cost,
            bbox_cost_weight=matcher_bbox_cost,
            iou_cost_weight=matcher_iou_cost,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

    def forward(
        self,
        cls_scores_list: List[torch.Tensor],
        bbox_preds_list: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss including auxiliary losses from all decoder layers.

        Args:
            cls_scores_list: Classification logits from each decoder layer.
                List of tensors, each with shape (B, num_queries, num_classes).
            bbox_preds_list: Bounding box predictions from each decoder layer.
                List of tensors, each with shape (B, num_queries, num_reg_params).
            gt_labels: Ground-truth class labels per sample in the batch.
                List of length B, each tensor has shape (num_gt_i,).
            gt_bboxes: Ground-truth bounding boxes per sample in the batch.
                List of length B, each tensor has shape (num_gt_i, num_reg_params).

        Returns:
            Dictionary with loss components:
                - "loss_cls": Total classification focal loss (summed over layers).
                - "loss_bbox": Total bounding box L1 loss (summed over layers).
                - "loss_total": Weighted sum of cls and bbox losses.
        """
        num_layers = len(cls_scores_list)
        device = cls_scores_list[0].device

        total_cls_loss = torch.tensor(0.0, device=device)
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_num_positives = 0

        for layer_idx in range(num_layers):
            cls_scores = cls_scores_list[layer_idx]
            bbox_preds = bbox_preds_list[layer_idx]

            # Perform Hungarian matching for this layer
            matched_indices = self.matcher(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
            )

            # Compute losses for this layer
            layer_cls_loss, layer_bbox_loss, num_pos = self._compute_layer_loss(
                cls_scores=cls_scores,
                bbox_preds=bbox_preds,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                matched_indices=matched_indices,
            )

            total_cls_loss = total_cls_loss + layer_cls_loss
            total_bbox_loss = total_bbox_loss + layer_bbox_loss
            total_num_positives += num_pos

        # Normalize by total number of positive samples across all layers
        # Use max(1, ...) to avoid division by zero
        num_positives = max(total_num_positives, 1)
        total_cls_loss = total_cls_loss / num_positives
        total_bbox_loss = total_bbox_loss / num_positives

        # Apply loss weights
        weighted_cls_loss = self.cls_weight * total_cls_loss
        weighted_bbox_loss = self.bbox_weight * total_bbox_loss
        total_loss = weighted_cls_loss + weighted_bbox_loss

        return {
            "loss_cls": weighted_cls_loss,
            "loss_bbox": weighted_bbox_loss,
            "loss_total": total_loss,
        }

    def _compute_layer_loss(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        gt_labels: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute focal loss and L1 loss for a single decoder layer.

        Args:
            cls_scores: Classification logits, shape (B, num_queries, num_classes).
            bbox_preds: Bounding box predictions, shape (B, num_queries, num_reg_params).
            gt_labels: Ground-truth labels per sample, list of (num_gt_i,).
            gt_bboxes: Ground-truth boxes per sample, list of (num_gt_i, num_reg_params).
            matched_indices: List of (pred_idx, gt_idx) tuples from matcher.

        Returns:
            Tuple of (cls_loss, bbox_loss, num_positives).
            Losses are summed (not normalized) -- normalization happens in forward().
        """
        batch_size, num_queries, num_classes = cls_scores.shape
        device = cls_scores.device

        # Build target classification labels
        # Default: all queries are background (class index = num_classes, i.e., no class)
        target_cls = torch.full(
            (batch_size, num_queries),
            self.num_classes,  # background class
            dtype=torch.long,
            device=device,
        )

        # Assign matched predictions their target class
        num_positives = 0
        for batch_idx, (pred_idx, gt_idx) in enumerate(matched_indices):
            if len(pred_idx) > 0:
                target_cls[batch_idx, pred_idx] = gt_labels[batch_idx][gt_idx]
                num_positives += len(pred_idx)

        # Compute focal loss for classification
        cls_loss = self._focal_loss(cls_scores, target_cls)

        # Compute L1 loss for bounding box regression (only for matched predictions)
        bbox_loss = torch.tensor(0.0, device=device)
        for batch_idx, (pred_idx, gt_idx) in enumerate(matched_indices):
            if len(pred_idx) > 0:
                pred_boxes = bbox_preds[batch_idx, pred_idx]  # (num_matched, reg_params)
                target_boxes = gt_bboxes[batch_idx][gt_idx]  # (num_matched, reg_params)
                bbox_loss = bbox_loss + F.l1_loss(
                    pred_boxes, target_boxes, reduction="sum"
                )

        return cls_loss, bbox_loss, num_positives

    def _focal_loss(
        self,
        cls_scores: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss for classification.

        Args:
            cls_scores: Predicted logits, shape (B, num_queries, num_classes).
            targets: Target class indices, shape (B, num_queries).
                Values in [0, num_classes-1] for foreground, num_classes for background.

        Returns:
            Scalar focal loss (summed over all queries and batch).
        """
        batch_size, num_queries, num_classes = cls_scores.shape

        # Create one-hot targets for binary focal loss per class
        # targets: (B, num_queries) with values in [0, num_classes]
        # For background (target == num_classes), all class targets are 0
        target_one_hot = torch.zeros(
            batch_size, num_queries, num_classes,
            dtype=cls_scores.dtype,
            device=cls_scores.device,
        )

        # Set foreground targets
        foreground_mask = targets < num_classes  # (B, num_queries)
        if foreground_mask.any():
            # Gather valid targets
            valid_targets = targets.clamp(max=num_classes - 1)
            target_one_hot.scatter_(
                2, valid_targets.unsqueeze(-1), 1.0
            )
            # Zero out background positions (where target == num_classes)
            background_mask = ~foreground_mask
            target_one_hot[background_mask] = 0.0

        # Compute focal loss element-wise
        pred_probs = torch.sigmoid(cls_scores)  # (B, num_queries, num_classes)

        # Binary cross-entropy components
        # For positive: -alpha * (1-p)^gamma * log(p)
        # For negative: -(1-alpha) * p^gamma * log(1-p)
        p_t = pred_probs * target_one_hot + (1 - pred_probs) * (1 - target_one_hot)
        alpha_t = (
            self.focal_alpha * target_one_hot
            + (1 - self.focal_alpha) * (1 - target_one_hot)
        )
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma

        # Binary cross-entropy (numerically stable)
        bce = F.binary_cross_entropy_with_logits(
            cls_scores, target_one_hot, reduction="none"
        )

        focal_loss = focal_weight * bce

        # Sum over all elements
        return focal_loss.sum()
