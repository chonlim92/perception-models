"""Loss functions for MapTR: Hungarian matching, Chamfer distance, direction loss.

Implements the hierarchical bipartite matching and composite loss function
for training MapTR models. The loss pipeline:
1. Hungarian matching assigns predictions to ground truth instances
2. Per-instance permutation invariance finds optimal point ordering
3. Combined loss = focal classification + Chamfer point-set + direction consistency

Reference: MapTR: Structured Modeling and Learning for Online Vectorized HD Map
Construction (Liao et al., ICLR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Utility functions
# =============================================================================


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sigmoid focal loss for imbalanced classification.

    Args:
        pred: Predicted logits [*, num_classes].
        target: Ground truth labels [*] (integer class indices) or one-hot [*, num_classes].
        alpha: Balancing factor for positive/negative samples.
        gamma: Focusing parameter to down-weight easy examples.
        reduction: 'none', 'mean', or 'sum'.

    Returns:
        Focal loss scalar or per-element loss depending on reduction.
    """
    num_classes = pred.shape[-1]

    # Convert integer labels to one-hot if needed
    if target.dim() < pred.dim():
        target_one_hot = F.one_hot(target.long(), num_classes).float()
    else:
        target_one_hot = target.float()

    pred_sigmoid = pred.sigmoid()
    # Binary cross-entropy component
    bce = F.binary_cross_entropy_with_logits(pred, target_one_hot, reduction="none")

    # Focal weight: (1 - p_t)^gamma
    p_t = pred_sigmoid * target_one_hot + (1 - pred_sigmoid) * (1 - target_one_hot)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weight
    alpha_t = alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)

    loss = alpha_t * focal_weight * bce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def chamfer_distance(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute symmetric Chamfer distance between two ordered point sets.

    For each point in pred, find the nearest point in GT (and vice versa),
    then average the two directions.

    Args:
        pred_pts: Predicted points [N, 2] or [B, N, 2].
        gt_pts: Ground truth points [M, 2] or [B, M, 2].
        reduction: 'none', 'mean', or 'sum'.

    Returns:
        Chamfer distance (scalar if reduction != 'none').
    """
    if pred_pts.dim() == 2:
        pred_pts = pred_pts.unsqueeze(0)
        gt_pts = gt_pts.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # pred_pts: [B, N, 2], gt_pts: [B, M, 2]
    # Compute pairwise distances: [B, N, M]
    diff = pred_pts.unsqueeze(2) - gt_pts.unsqueeze(1)  # [B, N, M, 2]
    dist_matrix = (diff ** 2).sum(dim=-1)  # [B, N, M]

    # For each predicted point, find nearest GT point
    min_pred_to_gt, _ = dist_matrix.min(dim=2)  # [B, N]
    # For each GT point, find nearest predicted point
    min_gt_to_pred, _ = dist_matrix.min(dim=1)  # [B, M]

    # Symmetric Chamfer: average of both directions
    chamfer_pred = min_pred_to_gt.mean(dim=1)  # [B]
    chamfer_gt = min_gt_to_pred.mean(dim=1)  # [B]
    chamfer = (chamfer_pred + chamfer_gt) / 2.0  # [B]

    if squeeze:
        chamfer = chamfer.squeeze(0)

    if reduction == "mean":
        return chamfer.mean()
    elif reduction == "sum":
        return chamfer.sum()
    return chamfer


# =============================================================================
# Point Set Loss (Chamfer-based)
# =============================================================================


class PointSetLoss(nn.Module):
    """Chamfer distance loss between predicted and ground truth point sets.

    Computes the symmetric nearest-neighbor distance: for each point in set A,
    finds the closest point in set B, and vice versa, then averages both directions.

    Args:
        loss_weight: Scalar weight for this loss component.
        use_l1: If True, use L1 distance; otherwise use squared L2.
    """

    def __init__(self, loss_weight: float = 5.0, use_l1: bool = False):
        super().__init__()
        self.loss_weight = loss_weight
        self.use_l1 = use_l1

    def forward(
        self,
        pred_pts: torch.Tensor,
        gt_pts: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Chamfer distance loss.

        Args:
            pred_pts: Predicted points [B, num_matched, num_points, 2].
            gt_pts: Ground truth points [B, num_matched, num_points, 2].
            valid_mask: Optional mask [B, num_matched] indicating valid pairs.

        Returns:
            Weighted Chamfer distance loss (scalar).
        """
        if pred_pts.numel() == 0:
            return pred_pts.sum() * 0.0

        B, N_matched, N_pts, coord_dim = pred_pts.shape

        # Compute pairwise distances: [B, N_matched, N_pts_pred, N_pts_gt]
        diff = pred_pts.unsqueeze(3) - gt_pts.unsqueeze(2)  # [B, N_matched, N_pts, N_pts, 2]
        if self.use_l1:
            dist_matrix = diff.abs().sum(dim=-1)  # [B, N_matched, N_pts, N_pts]
        else:
            dist_matrix = (diff ** 2).sum(dim=-1)  # [B, N_matched, N_pts, N_pts]

        # Forward direction: for each predicted point, nearest GT
        min_pred_to_gt, _ = dist_matrix.min(dim=3)  # [B, N_matched, N_pts]
        # Backward direction: for each GT point, nearest predicted
        min_gt_to_pred, _ = dist_matrix.min(dim=2)  # [B, N_matched, N_pts]

        # Average both directions, then average over points
        chamfer_per_instance = (
            min_pred_to_gt.mean(dim=2) + min_gt_to_pred.mean(dim=2)
        ) / 2.0  # [B, N_matched]

        if valid_mask is not None:
            # Zero out invalid pairs and normalize by valid count
            chamfer_per_instance = chamfer_per_instance * valid_mask.float()
            num_valid = valid_mask.float().sum().clamp(min=1.0)
            loss = chamfer_per_instance.sum() / num_valid
        else:
            loss = chamfer_per_instance.mean()

        return loss * self.loss_weight


# =============================================================================
# Direction Loss
# =============================================================================


class DirectionLoss(nn.Module):
    """Direction consistency loss penalizing reversed point ordering.

    Computes direction vectors (point[i+1] - point[i]) for both predicted and
    GT point sequences, then penalizes cases where directions are opposite
    (negative cosine similarity).

    This encourages the model to predict points in a consistent ordering
    direction along each map element.

    Args:
        loss_weight: Scalar weight for this loss component.
    """

    def __init__(self, loss_weight: float = 0.005):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        pred_pts: torch.Tensor,
        gt_pts: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute direction consistency loss.

        Args:
            pred_pts: Predicted points [B, num_matched, num_points, 2].
            gt_pts: Ground truth points [B, num_matched, num_points, 2].
            valid_mask: Optional mask [B, num_matched] for valid pairs.

        Returns:
            Weighted direction loss (scalar).
        """
        if pred_pts.numel() == 0 or pred_pts.shape[2] < 2:
            return pred_pts.sum() * 0.0

        # Compute direction vectors: point[i+1] - point[i]
        pred_dirs = pred_pts[:, :, 1:, :] - pred_pts[:, :, :-1, :]  # [B, N, P-1, 2]
        gt_dirs = gt_pts[:, :, 1:, :] - gt_pts[:, :, :-1, :]  # [B, N, P-1, 2]

        # Normalize direction vectors (avoid division by zero)
        pred_norm = pred_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        gt_norm = gt_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pred_dirs_normalized = pred_dirs / pred_norm
        gt_dirs_normalized = gt_dirs / gt_norm

        # Cosine similarity between corresponding direction vectors
        cosine_sim = (pred_dirs_normalized * gt_dirs_normalized).sum(dim=-1)  # [B, N, P-1]

        # Penalize negative cosine similarity (opposing directions)
        # Loss = max(0, -cosine_similarity) = ReLU(-cos)
        direction_loss = F.relu(-cosine_sim)  # [B, N, P-1]

        # Average over direction vectors
        direction_loss_per_instance = direction_loss.mean(dim=2)  # [B, N]

        if valid_mask is not None:
            direction_loss_per_instance = direction_loss_per_instance * valid_mask.float()
            num_valid = valid_mask.float().sum().clamp(min=1.0)
            loss = direction_loss_per_instance.sum() / num_valid
        else:
            loss = direction_loss_per_instance.mean()

        return loss * self.loss_weight


# =============================================================================
# Permutation Loss (find best cyclic shift and direction)
# =============================================================================


class PermutationLoss(nn.Module):
    """Finds the optimal point permutation for each matched prediction-GT pair.

    For closed polylines or polylines where the start vertex is ambiguous,
    tries all cyclic shifts and both traversal directions of the GT points,
    then selects the permutation with minimum point-to-point L1 distance.

    Args:
        try_reverse: Whether to also try reversing the GT point order.
    """

    def __init__(self, try_reverse: bool = True):
        super().__init__()
        self.try_reverse = try_reverse

    @torch.no_grad()
    def find_best_permutation(
        self,
        pred_pts: torch.Tensor,
        gt_pts: torch.Tensor,
    ) -> torch.Tensor:
        """Find optimal permutation of GT points for each pred-GT pair.

        Tries all cyclic shifts (and optionally reversed order) to find the
        assignment that minimizes point-to-point L1 distance.

        Args:
            pred_pts: Predicted points [N_matched, num_points, 2].
            gt_pts: Ground truth points [N_matched, num_points, 2].

        Returns:
            Permuted GT points [N_matched, num_points, 2] with optimal ordering.
        """
        N_matched, num_points, coord_dim = pred_pts.shape
        device = pred_pts.device

        if N_matched == 0:
            return gt_pts.clone()

        best_gt = gt_pts.clone()
        # Compute initial cost (no shift, forward direction)
        best_cost = (pred_pts - gt_pts).abs().sum(dim=-1).sum(dim=-1)  # [N_matched]

        # Try all cyclic shifts in forward direction
        for shift in range(1, num_points):
            shifted_gt = torch.roll(gt_pts, shifts=-shift, dims=1)
            cost = (pred_pts - shifted_gt).abs().sum(dim=-1).sum(dim=-1)  # [N_matched]
            improved = cost < best_cost
            if improved.any():
                best_cost[improved] = cost[improved]
                best_gt[improved] = shifted_gt[improved]

        # Try all cyclic shifts in reverse direction
        if self.try_reverse:
            gt_reversed = gt_pts.flip(dims=[1])
            for shift in range(num_points):
                shifted_gt = torch.roll(gt_reversed, shifts=-shift, dims=1)
                cost = (pred_pts - shifted_gt).abs().sum(dim=-1).sum(dim=-1)
                improved = cost < best_cost
                if improved.any():
                    best_cost[improved] = cost[improved]
                    best_gt[improved] = shifted_gt[improved]

        return best_gt

    def forward(
        self,
        pred_pts: torch.Tensor,
        gt_pts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find best permutation and compute L1 loss with optimal GT ordering.

        Args:
            pred_pts: Predicted points [B, N_matched, num_points, 2].
            gt_pts: Ground truth points [B, N_matched, num_points, 2].

        Returns:
            Tuple of (permutation_loss, permuted_gt):
                - permutation_loss: Mean L1 distance after optimal permutation (scalar).
                - permuted_gt: GT points reordered optimally [B, N_matched, num_points, 2].
        """
        B, N_matched, num_points, coord_dim = pred_pts.shape

        if N_matched == 0:
            return pred_pts.sum() * 0.0, gt_pts

        permuted_gt_list = []
        for b in range(B):
            permuted = self.find_best_permutation(pred_pts[b], gt_pts[b])
            permuted_gt_list.append(permuted)

        permuted_gt = torch.stack(permuted_gt_list, dim=0)  # [B, N_matched, num_points, 2]
        loss = (pred_pts - permuted_gt).abs().mean()

        return loss, permuted_gt


# =============================================================================
# Hungarian Matcher
# =============================================================================


class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground truth using Hungarian algorithm.

    Computes a cost matrix combining classification, point-set, and direction costs,
    then solves the linear assignment problem using scipy's implementation of the
    Hungarian algorithm.

    The matching is performed independently for each sample in the batch.

    Args:
        cost_class: Weight for classification cost component.
        cost_pts: Weight for point-set (Chamfer) cost component.
        cost_dir: Weight for direction consistency cost component.
        focal_alpha: Alpha parameter for focal loss in classification cost.
        focal_gamma: Gamma parameter for focal loss in classification cost.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_pts: float = 5.0,
        cost_dir: float = 0.005,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_pts = cost_pts
        self.cost_dir = cost_dir
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(
        self,
        cls_scores: torch.Tensor,
        pred_pts: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_pts: torch.Tensor,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Perform Hungarian matching for a batch.

        Args:
            cls_scores: Predicted class logits [B, num_queries, num_classes].
            pred_pts: Predicted point coords [B, num_queries, num_points, 2].
            gt_labels: Ground truth class labels, list of [num_gt_i] per sample,
                or padded tensor [B, max_num_gt].
            gt_pts: Ground truth point coords, list of [num_gt_i, num_points, 2]
                per sample, or padded tensor [B, max_num_gt, num_points, 2].
            gt_masks: Optional validity mask [B, max_num_gt] for padded GT.

        Returns:
            List of (pred_indices, gt_indices) tuples for each sample in the batch.
            pred_indices[i] and gt_indices[i] are 1D tensors with the matched indices.
        """
        batch_size, num_queries, num_classes = cls_scores.shape
        _, _, num_points, _ = pred_pts.shape

        indices = []

        for b in range(batch_size):
            # Get valid GT for this sample
            if gt_masks is not None:
                valid = gt_masks[b].bool()
                gt_labels_b = gt_labels[b][valid]  # [num_gt]
                gt_pts_b = gt_pts[b][valid]  # [num_gt, num_points, 2]
            elif isinstance(gt_labels, list):
                gt_labels_b = gt_labels[b]
                gt_pts_b = gt_pts[b]
            else:
                gt_labels_b = gt_labels[b]
                gt_pts_b = gt_pts[b]

            num_gt = gt_labels_b.shape[0]

            if num_gt == 0:
                # No GT: return empty matching
                indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=cls_scores.device),
                        torch.tensor([], dtype=torch.long, device=cls_scores.device),
                    )
                )
                continue

            # --- Classification cost (focal-loss based) ---
            # cls_scores_b: [num_queries, num_classes]
            out_prob = cls_scores[b].sigmoid()  # [num_queries, num_classes]

            # Focal cost: -alpha * (1-p)^gamma * log(p) for positive class
            neg_cost = -(1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (
                -(1 - out_prob + 1e-8).log()
            )
            pos_cost = -self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (
                -(out_prob + 1e-8).log()
            )
            # Select the cost for the target class: [num_queries, num_gt]
            cls_cost = pos_cost[:, gt_labels_b.long()] - neg_cost[:, gt_labels_b.long()]

            # --- Point-set cost (Chamfer distance) ---
            # pred_pts_b: [num_queries, num_points, 2]
            # gt_pts_b: [num_gt, num_points, 2]
            pred_pts_b = pred_pts[b]  # [num_queries, num_points, 2]

            # Compute pairwise Chamfer distance: [num_queries, num_gt]
            # Expand for broadcasting:
            # pred_expanded: [num_queries, 1, num_points, 1, 2]
            # gt_expanded: [1, num_gt, 1, num_points, 2]
            pred_expanded = pred_pts_b.unsqueeze(1).unsqueeze(3)  # [Q, 1, P, 1, 2]
            gt_expanded = gt_pts_b.unsqueeze(0).unsqueeze(2)  # [1, G, 1, P, 2]

            # Pairwise point distances: [Q, G, P_pred, P_gt]
            pairwise_dist = ((pred_expanded - gt_expanded) ** 2).sum(dim=-1)

            # Chamfer: min over gt for each pred point + min over pred for each gt point
            min_pred_to_gt = pairwise_dist.min(dim=3)[0].mean(dim=2)  # [Q, G]
            min_gt_to_pred = pairwise_dist.min(dim=2)[0].mean(dim=2)  # [Q, G]
            pts_cost = (min_pred_to_gt + min_gt_to_pred) / 2.0

            # --- Direction cost ---
            # Compute direction vectors for pred and GT
            pred_dirs = pred_pts_b[:, 1:, :] - pred_pts_b[:, :-1, :]  # [Q, P-1, 2]
            gt_dirs = gt_pts_b[:, 1:, :] - gt_pts_b[:, :-1, :]  # [G, P-1, 2]

            # Normalize
            pred_dirs_norm = pred_dirs / (pred_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6))
            gt_dirs_norm = gt_dirs / (gt_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6))

            # Cosine similarity for all pairs: [Q, G, P-1]
            # pred_dirs_norm: [Q, 1, P-1, 2], gt_dirs_norm: [1, G, P-1, 2]
            cos_sim = (
                pred_dirs_norm.unsqueeze(1) * gt_dirs_norm.unsqueeze(0)
            ).sum(dim=-1)  # [Q, G, P-1]

            # Direction cost: penalize negative cosine (opposite directions)
            dir_cost = F.relu(-cos_sim).mean(dim=2)  # [Q, G]

            # --- Combined cost matrix ---
            cost_matrix = (
                self.cost_class * cls_cost
                + self.cost_pts * pts_cost
                + self.cost_dir * dir_cost
            )

            # Solve assignment problem with Hungarian algorithm
            cost_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            indices.append(
                (
                    torch.tensor(row_ind, dtype=torch.long, device=cls_scores.device),
                    torch.tensor(col_ind, dtype=torch.long, device=cls_scores.device),
                )
            )

        return indices


# =============================================================================
# Main MapTR Loss
# =============================================================================


class MapTRLoss(nn.Module):
    """Combined loss for MapTR training.

    Orchestrates the full loss computation pipeline:
    1. Run Hungarian matching to assign predictions to GT
    2. Compute focal loss for classification on all predictions
    3. Compute point-set loss (Chamfer) for matched pairs
    4. Compute direction loss for matched pairs
    5. Optionally apply permutation-invariant point ordering
    6. Support auxiliary losses from intermediate decoder layers

    Args:
        num_classes: Number of map element classes.
        num_points: Number of points per map element.
        matcher_cfg: Configuration dict for HungarianMatcher.
        cls_weight: Weight for classification loss.
        pts_weight: Weight for point-set (Chamfer) loss.
        dir_weight: Weight for direction loss.
        focal_alpha: Alpha parameter for focal loss.
        focal_gamma: Gamma parameter for focal loss.
        aux_loss_weight: Weight multiplier for auxiliary (intermediate layer) losses.
        use_permutation: Whether to find optimal point permutation before loss.
        try_reverse: Whether to try reversed GT ordering in permutation search.
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_points: int = 20,
        matcher_cfg: Optional[Dict] = None,
        cls_weight: float = 2.0,
        pts_weight: float = 5.0,
        dir_weight: float = 0.005,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        aux_loss_weight: float = 1.0,
        use_permutation: bool = True,
        try_reverse: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.cls_weight = cls_weight
        self.pts_weight = pts_weight
        self.dir_weight = dir_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_loss_weight = aux_loss_weight
        self.use_permutation = use_permutation

        # Build matcher
        if matcher_cfg is None:
            matcher_cfg = {}
        self.matcher = HungarianMatcher(
            cost_class=matcher_cfg.get("cost_class", 2.0),
            cost_pts=matcher_cfg.get("cost_pts", 5.0),
            cost_dir=matcher_cfg.get("cost_dir", 0.005),
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

        # Sub-losses
        self.pts_loss = PointSetLoss(loss_weight=1.0)  # Weight applied via pts_weight
        self.dir_loss = DirectionLoss(loss_weight=1.0)  # Weight applied via dir_weight
        self.permutation_loss = PermutationLoss(try_reverse=try_reverse)

    def _compute_cls_loss(
        self,
        cls_scores: torch.Tensor,
        gt_labels: torch.Tensor,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute focal classification loss.

        Unmatched predictions are assigned a background (num_classes) target.

        Args:
            cls_scores: [B, num_queries, num_classes].
            gt_labels: Padded GT labels [B, max_num_gt] or list of per-sample tensors.
            indices: Matching indices from Hungarian matcher.

        Returns:
            Classification focal loss (scalar).
        """
        B, num_queries, num_classes = cls_scores.shape
        device = cls_scores.device

        # Build target: background class for all, then fill matched
        # Use num_classes as background index for focal loss computation
        target = torch.full(
            (B, num_queries), num_classes, dtype=torch.long, device=device
        )

        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                if isinstance(gt_labels, list):
                    target[b, pred_idx] = gt_labels[b][gt_idx]
                else:
                    target[b, pred_idx] = gt_labels[b][gt_idx]

        # Expand cls_scores to include background class channel
        # Add a zero logit for background class
        bg_logit = torch.zeros(B, num_queries, 1, device=device, dtype=cls_scores.dtype)
        cls_scores_with_bg = torch.cat([cls_scores, bg_logit], dim=-1)  # [B, Q, C+1]

        loss = focal_loss(
            cls_scores_with_bg.reshape(-1, num_classes + 1),
            target.reshape(-1),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="mean",
        )

        return loss

    def _gather_matched_pts(
        self,
        pred_pts: torch.Tensor,
        gt_pts: torch.Tensor,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather matched prediction and GT points using matching indices.

        Args:
            pred_pts: [B, num_queries, num_points, 2].
            gt_pts: [B, max_num_gt, num_points, 2] or list.
            indices: Matching indices.

        Returns:
            Tuple of (matched_pred, matched_gt, valid_mask):
                - matched_pred: [B, max_matched, num_points, 2]
                - matched_gt: [B, max_matched, num_points, 2]
                - valid_mask: [B, max_matched] boolean mask
        """
        B = pred_pts.shape[0]
        device = pred_pts.device
        num_points = pred_pts.shape[2]

        # Find max number of matches across batch
        max_matched = max(len(idx[0]) for idx in indices) if indices else 0
        if max_matched == 0:
            matched_pred = torch.zeros(B, 0, num_points, 2, device=device)
            matched_gt = torch.zeros(B, 0, num_points, 2, device=device)
            valid_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
            return matched_pred, matched_gt, valid_mask

        matched_pred = torch.zeros(B, max_matched, num_points, 2, device=device)
        matched_gt = torch.zeros(B, max_matched, num_points, 2, device=device)
        valid_mask = torch.zeros(B, max_matched, dtype=torch.bool, device=device)

        for b, (pred_idx, gt_idx) in enumerate(indices):
            n = len(pred_idx)
            if n == 0:
                continue
            matched_pred[b, :n] = pred_pts[b][pred_idx]
            if isinstance(gt_pts, list):
                matched_gt[b, :n] = gt_pts[b][gt_idx]
            else:
                matched_gt[b, :n] = gt_pts[b][gt_idx]
            valid_mask[b, :n] = True

        return matched_pred, matched_gt, valid_mask

    def _compute_single_layer_loss(
        self,
        cls_scores: torch.Tensor,
        pred_pts: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_pts: torch.Tensor,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for a single decoder layer's predictions.

        Args:
            cls_scores: [B, num_queries, num_classes].
            pred_pts: [B, num_queries, num_points, 2].
            gt_labels: [B, max_num_gt] or list of per-sample tensors.
            gt_pts: [B, max_num_gt, num_points, 2] or list.
            gt_masks: Optional [B, max_num_gt] validity mask.

        Returns:
            Dict with 'cls_loss', 'pts_loss', 'dir_loss', 'total_loss' keys.
        """
        # Step 1: Hungarian matching
        indices = self.matcher(cls_scores, pred_pts, gt_labels, gt_pts, gt_masks)

        # Step 2: Classification loss
        cls_loss = self._compute_cls_loss(cls_scores, gt_labels, indices)

        # Step 3: Gather matched points
        matched_pred, matched_gt, valid_mask = self._gather_matched_pts(
            pred_pts, gt_pts, indices
        )

        # Step 4: Permutation-invariant ordering (optional)
        if self.use_permutation and matched_pred.shape[1] > 0:
            _, permuted_gt = self.permutation_loss(matched_pred, matched_gt)
        else:
            permuted_gt = matched_gt

        # Step 5: Point-set loss (Chamfer distance)
        pts_loss = self.pts_loss(matched_pred, permuted_gt, valid_mask)

        # Step 6: Direction loss
        dir_loss = self.dir_loss(matched_pred, permuted_gt, valid_mask)

        # Weighted total
        total_loss = (
            self.cls_weight * cls_loss
            + self.pts_weight * pts_loss
            + self.dir_weight * dir_loss
        )

        return {
            "cls_loss": cls_loss,
            "pts_loss": pts_loss,
            "dir_loss": dir_loss,
            "total_loss": total_loss,
        }

    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        gt_labels: torch.Tensor,
        gt_pts: torch.Tensor,
        gt_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute full MapTR loss including auxiliary losses from intermediate layers.

        Args:
            predictions: Dict from MapTRHead.forward() containing:
                - "cls_scores": List of [B, num_queries, num_classes] per decoder layer.
                - "point_coords": List of [B, num_queries, num_points, 2] per decoder layer.
            gt_labels: Ground truth class labels [B, max_num_gt].
            gt_pts: Ground truth point coordinates [B, max_num_gt, num_points, 2].
            gt_masks: Optional validity mask [B, max_num_gt] for padded GT.

        Returns:
            Dict containing:
                - "loss": Total combined loss (scalar).
                - "cls_loss": Classification loss from final layer.
                - "pts_loss": Point-set loss from final layer.
                - "dir_loss": Direction loss from final layer.
                - "aux_loss": Total auxiliary loss from intermediate layers.
        """
        cls_scores_list = predictions["cls_scores"]
        pred_pts_list = predictions["point_coords"]
        num_layers = len(cls_scores_list)

        # Compute loss for the final (last) decoder layer
        final_losses = self._compute_single_layer_loss(
            cls_scores_list[-1], pred_pts_list[-1], gt_labels, gt_pts, gt_masks
        )

        # Compute auxiliary losses from intermediate layers
        aux_total = torch.tensor(0.0, device=cls_scores_list[0].device)
        if num_layers > 1:
            for layer_idx in range(num_layers - 1):
                aux_losses = self._compute_single_layer_loss(
                    cls_scores_list[layer_idx],
                    pred_pts_list[layer_idx],
                    gt_labels,
                    gt_pts,
                    gt_masks,
                )
                aux_total = aux_total + aux_losses["total_loss"]
            aux_total = aux_total / (num_layers - 1)

        # Combined loss
        total_loss = final_losses["total_loss"] + self.aux_loss_weight * aux_total

        return {
            "loss": total_loss,
            "cls_loss": final_losses["cls_loss"],
            "pts_loss": final_losses["pts_loss"],
            "dir_loss": final_losses["dir_loss"],
            "aux_loss": aux_total,
        }
