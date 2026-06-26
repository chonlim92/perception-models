"""Loss functions for StreamMapNet temporal HD map construction.

This module implements the complete loss computation pipeline for StreamMapNet,
including Hungarian matching for bipartite assignment, focal classification loss,
point-set regression loss, and direction-aware loss for polyline ambiguity.

Reference: StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def focal_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute focal loss for classification.

    Focal loss addresses class imbalance by down-weighting well-classified
    examples and focusing on hard, misclassified ones.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        pred_logits: Predicted logits of shape (N, C) where C is num_classes.
        targets: Ground truth class indices of shape (N,) with values in [0, C-1].
        alpha: Weighting factor for the rare class. Default: 0.25.
        gamma: Focusing parameter that reduces loss for well-classified examples.
            Default: 2.0.
        reduction: Reduction mode - 'mean', 'sum', or 'none'. Default: 'mean'.

    Returns:
        Focal loss tensor, scalar if reduction is 'mean' or 'sum', otherwise
        shape (N,).
    """
    num_classes = pred_logits.shape[-1]
    # Convert logits to probabilities
    probs = F.softmax(pred_logits, dim=-1)

    # One-hot encode targets
    target_one_hot = F.one_hot(targets, num_classes=num_classes).float()

    # Compute focal weight: (1 - p_t)^gamma
    p_t = (probs * target_one_hot).sum(dim=-1)
    focal_weight = (1.0 - p_t) ** gamma

    # Compute cross-entropy per sample
    ce_loss = F.cross_entropy(pred_logits, targets, reduction="none")

    # Apply alpha balancing
    # alpha for positive class, (1 - alpha) for negative/background class
    alpha_t = torch.where(
        targets < num_classes - 1,
        torch.tensor(alpha, device=pred_logits.device, dtype=pred_logits.dtype),
        torch.tensor(1.0 - alpha, device=pred_logits.device, dtype=pred_logits.dtype),
    )

    loss = alpha_t * focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class HungarianMatcher(nn.Module):
    """Hungarian Matcher for bipartite matching between predictions and ground truth.

    Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
    the optimal assignment between predicted map elements and ground truth elements
    that minimizes the total matching cost.

    The cost matrix combines:
        - Classification cost (focal-based)
        - Point-set distance cost (L1)
        - Direction cost (minimum of forward/reverse L1)

    Args:
        cost_class: Weight for classification cost. Default: 2.0.
        cost_pts: Weight for point-set L1 distance cost. Default: 5.0.
        cost_dir: Weight for direction-aware cost. Default: 0.005.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_pts: float = 5.0,
        cost_dir: float = 0.005,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_pts = cost_pts
        self.cost_dir = cost_dir

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_points: torch.Tensor,
        gt_nums: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Perform Hungarian matching between predictions and ground truth.

        Args:
            pred_logits: Predicted classification logits, shape (B, N, num_classes+1).
                The last class is the no-object class.
            pred_points: Predicted point sets, shape (B, N, K, 2).
                K ordered points per element, coordinates normalized to [0,1].
            gt_labels: Ground truth class labels, shape (B, M).
                Padded with -1 for invalid entries.
            gt_points: Ground truth point sets, shape (B, M, K, 2).
                Padded with zeros for invalid entries.
            gt_nums: List of actual number of GT elements per sample in the batch.

        Returns:
            List of tuples (pred_indices, gt_indices) for each sample in the batch,
            where pred_indices and gt_indices are 1D tensors of matched indices.
        """
        batch_size, num_queries = pred_logits.shape[:2]
        device = pred_logits.device

        indices = []

        for b in range(batch_size):
            num_gt = gt_nums[b]

            if num_gt == 0:
                # No ground truth elements - return empty matching
                indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=device),
                        torch.tensor([], dtype=torch.long, device=device),
                    )
                )
                continue

            # Get predictions and ground truth for this sample
            logits = pred_logits[b]  # (N, num_classes+1)
            points = pred_points[b]  # (N, K, 2)
            tgt_labels = gt_labels[b, :num_gt]  # (num_gt,)
            tgt_points = gt_points[b, :num_gt]  # (num_gt, K, 2)

            # --- Classification cost (focal-based) ---
            # Compute softmax probabilities
            probs = F.softmax(logits, dim=-1)  # (N, num_classes+1)
            # Cost is negative log-probability of the target class, weighted by focal term
            # For each query-GT pair, cost = -alpha * (1-p)^gamma * log(p)
            alpha = 0.25
            gamma = 2.0
            # Gather probabilities for target classes: (N, num_gt)
            cost_probs = probs[:, tgt_labels.long()]  # (N, num_gt)
            focal_weight = (1.0 - cost_probs) ** gamma
            cost_class = -alpha * focal_weight * torch.log(cost_probs.clamp(min=1e-8))

            # --- Point-set distance cost (L1) ---
            # points: (N, K, 2), tgt_points: (num_gt, K, 2)
            # Expand for pairwise comparison
            pred_pts_expanded = points.unsqueeze(1).expand(
                -1, num_gt, -1, -1
            )  # (N, num_gt, K, 2)
            tgt_pts_expanded = tgt_points.unsqueeze(0).expand(
                num_queries, -1, -1, -1
            )  # (N, num_gt, K, 2)
            cost_pts = torch.abs(pred_pts_expanded - tgt_pts_expanded).sum(
                dim=(-2, -1)
            )  # (N, num_gt)

            # --- Direction cost ---
            # Polylines can be annotated in either direction
            # Compute L1 for reversed GT point ordering and take minimum
            tgt_pts_reversed = tgt_points.flip(dims=[1])  # (num_gt, K, 2)
            tgt_pts_rev_expanded = tgt_pts_reversed.unsqueeze(0).expand(
                num_queries, -1, -1, -1
            )  # (N, num_gt, K, 2)
            cost_pts_reversed = torch.abs(
                pred_pts_expanded - tgt_pts_rev_expanded
            ).sum(dim=(-2, -1))  # (N, num_gt)
            cost_dir = torch.min(cost_pts, cost_pts_reversed)

            # --- Combined cost matrix ---
            cost_matrix = (
                self.cost_class * cost_class
                + self.cost_pts * cost_pts
                + self.cost_dir * cost_dir
            )

            # Run Hungarian algorithm (on CPU, as scipy requires numpy)
            cost_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            indices.append(
                (
                    torch.tensor(row_ind, dtype=torch.long, device=device),
                    torch.tensor(col_ind, dtype=torch.long, device=device),
                )
            )

        return indices


class DirectionAwareLoss(nn.Module):
    """Direction-aware loss for polyline map elements.

    Polylines can be annotated in either direction (start-to-end or end-to-start).
    This loss computes the L1 distance for both the forward and reversed GT point
    orderings and takes the minimum, resolving the direction ambiguity.

    This is crucial for map elements like lane dividers or road boundaries where
    the annotation direction is arbitrary.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
    ) -> torch.Tensor:
        """Compute direction-aware L1 loss.

        Args:
            pred_points: Predicted points for matched elements, shape (num_matched, K, 2).
            gt_points: Ground truth points for matched elements, shape (num_matched, K, 2).

        Returns:
            Scalar direction-aware loss (mean over matched elements).
        """
        if pred_points.shape[0] == 0:
            return pred_points.sum() * 0.0  # Return zero loss preserving grad

        # Forward direction L1
        loss_forward = torch.abs(pred_points - gt_points).sum(dim=(-2, -1))  # (num_matched,)

        # Reversed direction L1
        gt_points_reversed = gt_points.flip(dims=[1])  # (num_matched, K, 2)
        loss_reversed = torch.abs(pred_points - gt_points_reversed).sum(
            dim=(-2, -1)
        )  # (num_matched,)

        # Take minimum of forward and reversed
        loss = torch.min(loss_forward, loss_reversed)

        return loss.mean()


class StreamMapNetLoss(nn.Module):
    """Complete loss module for StreamMapNet.

    Combines Hungarian matching with classification, point regression, and
    direction-aware losses. Supports auxiliary losses at intermediate decoder
    layers with decreasing weights.

    The loss computation pipeline:
        1. Run Hungarian matching to find optimal prediction-GT assignment
        2. Compute focal classification loss on all predictions
        3. Compute point regression loss (L1) on matched pairs
        4. Compute direction-aware loss on matched pairs
        5. Apply auxiliary losses at each decoder layer

    Args:
        num_classes: Number of map element classes (excluding no-object class).
        cls_weight: Weight for classification loss. Default: 2.0.
        pts_weight: Weight for point regression loss. Default: 5.0.
        dir_weight: Weight for direction-aware loss. Default: 0.005.
        num_pts_per_element: Number of ordered points per map element (K). Default: 20.
        cost_class: Matching cost weight for classification. Default: 2.0.
        cost_pts: Matching cost weight for point distance. Default: 5.0.
        cost_dir: Matching cost weight for direction. Default: 0.005.
    """

    def __init__(
        self,
        num_classes: int,
        cls_weight: float = 2.0,
        pts_weight: float = 5.0,
        dir_weight: float = 0.005,
        num_pts_per_element: int = 20,
        cost_class: float = 2.0,
        cost_pts: float = 5.0,
        cost_dir: float = 0.005,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.pts_weight = pts_weight
        self.dir_weight = dir_weight
        self.num_pts_per_element = num_pts_per_element

        # Hungarian matcher
        self.matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_pts=cost_pts,
            cost_dir=cost_dir,
        )

        # Direction-aware loss
        self.direction_loss = DirectionAwareLoss()

    def _get_gt_nums(self, gt_labels: torch.Tensor) -> List[int]:
        """Get the number of valid GT elements per sample.

        Valid elements have labels >= 0 (padding is -1).

        Args:
            gt_labels: Ground truth labels, shape (B, M), padded with -1.

        Returns:
            List of valid GT counts per sample.
        """
        return [(labels >= 0).sum().item() for labels in gt_labels]

    def _compute_cls_loss(
        self,
        pred_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        gt_nums: List[int],
    ) -> torch.Tensor:
        """Compute focal classification loss for all predictions.

        Matched predictions get their GT class label; unmatched predictions get
        the no-object class (last class index).

        Args:
            pred_logits: Predicted logits, shape (B, N, num_classes+1).
            gt_labels: Ground truth labels, shape (B, M).
            indices: Hungarian matching results.
            gt_nums: Number of valid GTs per sample.

        Returns:
            Scalar classification loss.
        """
        batch_size, num_queries, num_cls = pred_logits.shape
        device = pred_logits.device
        no_object_class = num_cls - 1  # Last class is no-object

        # Build target labels for all predictions
        target_classes = torch.full(
            (batch_size, num_queries),
            no_object_class,
            dtype=torch.long,
            device=device,
        )

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = gt_labels[b, gt_idx].long()

        # Flatten and compute focal loss
        pred_flat = pred_logits.reshape(-1, num_cls)
        target_flat = target_classes.reshape(-1)

        loss = focal_loss(pred_flat, target_flat, alpha=0.25, gamma=2.0, reduction="mean")
        return loss

    def _compute_pts_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute L1 point regression loss on matched pairs.

        Args:
            pred_points: Predicted points, shape (B, N, K, 2).
            gt_points: Ground truth points, shape (B, M, K, 2).
            indices: Hungarian matching results.

        Returns:
            Scalar point regression loss.
        """
        device = pred_points.device
        matched_pred_pts = []
        matched_gt_pts = []

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                matched_pred_pts.append(pred_points[b, pred_idx])  # (n_matched, K, 2)
                matched_gt_pts.append(gt_points[b, gt_idx])  # (n_matched, K, 2)

        if len(matched_pred_pts) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        matched_pred_pts = torch.cat(matched_pred_pts, dim=0)  # (total_matched, K, 2)
        matched_gt_pts = torch.cat(matched_gt_pts, dim=0)  # (total_matched, K, 2)

        # L1 loss normalized by number of points
        loss = F.l1_loss(matched_pred_pts, matched_gt_pts, reduction="mean")
        return loss

    def _compute_dir_loss(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute direction-aware loss on matched pairs.

        Args:
            pred_points: Predicted points, shape (B, N, K, 2).
            gt_points: Ground truth points, shape (B, M, K, 2).
            indices: Hungarian matching results.

        Returns:
            Scalar direction-aware loss.
        """
        device = pred_points.device
        matched_pred_pts = []
        matched_gt_pts = []

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                matched_pred_pts.append(pred_points[b, pred_idx])
                matched_gt_pts.append(gt_points[b, gt_idx])

        if len(matched_pred_pts) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        matched_pred_pts = torch.cat(matched_pred_pts, dim=0)
        matched_gt_pts = torch.cat(matched_gt_pts, dim=0)

        return self.direction_loss(matched_pred_pts, matched_gt_pts)

    def _compute_single_layer_loss(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_points: torch.Tensor,
        gt_nums: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Compute losses for a single decoder layer.

        Args:
            pred_logits: Predicted logits, shape (B, N, num_classes+1).
            pred_points: Predicted points, shape (B, N, K, 2).
            gt_labels: Ground truth labels, shape (B, M).
            gt_points: Ground truth points, shape (B, M, K, 2).
            gt_nums: Number of valid GTs per sample.

        Returns:
            Dict with 'loss_cls', 'loss_pts', 'loss_dir' tensors.
        """
        # Run Hungarian matching
        indices = self.matcher(pred_logits, pred_points, gt_labels, gt_points, gt_nums)

        # Compute individual losses
        loss_cls = self._compute_cls_loss(pred_logits, gt_labels, indices, gt_nums)
        loss_pts = self._compute_pts_loss(pred_points, gt_points, indices)
        loss_dir = self._compute_dir_loss(pred_points, gt_points, indices)

        return {
            "loss_cls": loss_cls,
            "loss_pts": loss_pts,
            "loss_dir": loss_dir,
        }

    def forward(
        self,
        pred_logits_layers: List[torch.Tensor],
        pred_points_layers: List[torch.Tensor],
        gt_labels: torch.Tensor,
        gt_points: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total StreamMapNet loss across all decoder layers.

        Applies auxiliary losses at each decoder layer with decreasing weight:
        - Last layer: weight = 1.0
        - Intermediate layers: weight = 0.5

        Args:
            pred_logits_layers: List of predicted logits per decoder layer,
                each of shape (B, N, num_classes+1). Length = num_decoder_layers.
            pred_points_layers: List of predicted points per decoder layer,
                each of shape (B, N, K, 2). Length = num_decoder_layers.
            gt_labels: Ground truth class labels, shape (B, M).
                Padded with -1 for invalid entries.
            gt_points: Ground truth point sets, shape (B, M, K, 2).
                Padded with zeros for invalid entries.

        Returns:
            Loss dictionary with keys:
                - 'loss_cls': Weighted classification loss (summed across layers).
                - 'loss_pts': Weighted point regression loss (summed across layers).
                - 'loss_dir': Weighted direction-aware loss (summed across layers).
                - 'loss_total': Total combined loss.
        """
        gt_nums = self._get_gt_nums(gt_labels)
        num_layers = len(pred_logits_layers)

        total_loss_cls = torch.tensor(0.0, device=gt_labels.device)
        total_loss_pts = torch.tensor(0.0, device=gt_labels.device)
        total_loss_dir = torch.tensor(0.0, device=gt_labels.device)

        for layer_idx in range(num_layers):
            # Layer weight: 1.0 for last layer, 0.5 for intermediate
            layer_weight = 1.0 if layer_idx == num_layers - 1 else 0.5

            layer_losses = self._compute_single_layer_loss(
                pred_logits=pred_logits_layers[layer_idx],
                pred_points=pred_points_layers[layer_idx],
                gt_labels=gt_labels,
                gt_points=gt_points,
                gt_nums=gt_nums,
            )

            total_loss_cls = total_loss_cls + layer_weight * layer_losses["loss_cls"]
            total_loss_pts = total_loss_pts + layer_weight * layer_losses["loss_pts"]
            total_loss_dir = total_loss_dir + layer_weight * layer_losses["loss_dir"]

        # Apply loss weights
        weighted_cls = self.cls_weight * total_loss_cls
        weighted_pts = self.pts_weight * total_loss_pts
        weighted_dir = self.dir_weight * total_loss_dir

        loss_total = weighted_cls + weighted_pts + weighted_dir

        return {
            "loss_cls": weighted_cls,
            "loss_pts": weighted_pts,
            "loss_dir": weighted_dir,
            "loss_total": loss_total,
        }
