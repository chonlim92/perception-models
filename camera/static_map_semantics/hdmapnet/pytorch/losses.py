"""
Loss functions for HDMapNet training.

Includes:
- SemanticLoss: Binary cross-entropy with optional focal loss weighting.
- DiscriminativeLoss: Push-pull embedding loss for instance segmentation.
- DirectionLoss: Smooth L1 loss on direction vectors.
- HDMapNetLoss: Combined weighted sum of all losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLoss(nn.Module):
    """Binary cross-entropy loss with optional focal loss for semantic segmentation.

    Focal loss reduces the relative loss for well-classified examples, focusing
    training on hard negatives. This helps with the severe class imbalance in
    HD map segmentation where most BEV pixels are background.
    """

    def __init__(self, use_focal=True, focal_alpha=0.25, focal_gamma=2.0, pos_weight=None):
        """
        Args:
            use_focal: Whether to use focal loss (default True).
            focal_alpha: Balancing factor for focal loss.
            focal_gamma: Focusing parameter for focal loss.
            pos_weight: Optional per-class positive weight tensor.
        """
        super().__init__()
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

    def focal_loss(self, pred, target):
        """Compute focal loss.

        Args:
            pred: Predicted logits (B, C, H, W).
            target: Binary target (B, C, H, W).

        Returns:
            Scalar loss value.
        """
        prob = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Focal modulating factor
        p_t = prob * target + (1 - prob) * (1 - target)
        modulating_factor = (1 - p_t) ** self.focal_gamma

        # Alpha weighting
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)

        focal_loss = alpha_t * modulating_factor * ce_loss
        return focal_loss.mean()

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, num_classes, H, W).
            target: Binary ground truth (B, num_classes, H, W).

        Returns:
            Scalar loss value.
        """
        if self.use_focal:
            return self.focal_loss(pred, target)
        else:
            if self.pos_weight is not None:
                pos_weight = self.pos_weight.to(pred.device)
                # Reshape for broadcasting: (num_classes,) -> (1, num_classes, 1, 1)
                pos_weight = pos_weight.reshape(1, -1, 1, 1)
                return F.binary_cross_entropy_with_logits(
                    pred, target, pos_weight=pos_weight.expand_as(pred)
                )
            return F.binary_cross_entropy_with_logits(pred, target)


class DiscriminativeLoss(nn.Module):
    """Discriminative loss for instance embedding learning.

    Based on "Semantic Instance Segmentation with a Discriminative Loss Function"
    (De Brabandere et al., 2017).

    Three terms:
    - Variance (pull) term: Pull embeddings of same instance toward their mean.
    - Distance (push) term: Push means of different instances apart.
    - Regularization term: Keep instance means close to origin.
    """

    def __init__(self, delta_v=0.5, delta_d=3.0, alpha=1.0, beta=1.0, gamma=0.001):
        """
        Args:
            delta_v: Margin for the variance (pull) term.
            delta_d: Margin for the distance (push) term.
            alpha: Weight for variance term.
            beta: Weight for distance term.
            gamma: Weight for regularization term.
        """
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, embeddings, instance_map, semantic_mask):
        """
        Args:
            embeddings: Predicted embeddings (B, E, H, W) where E is embedding dim.
            instance_map: Instance ID map (B, H, W) with integer instance IDs.
                          Background/unlabeled pixels have ID 0.
            semantic_mask: Binary mask (B, H, W) indicating valid pixels (any class).

        Returns:
            Scalar discriminative loss.
        """
        B, E, H, W = embeddings.shape
        device = embeddings.device

        total_var_loss = torch.tensor(0.0, device=device)
        total_dist_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        valid_batches = 0

        for b in range(B):
            emb = embeddings[b]  # (E, H, W)
            inst = instance_map[b]  # (H, W)
            mask = semantic_mask[b]  # (H, W)

            # Get unique instance IDs (excluding background 0)
            inst_ids = inst.unique()
            inst_ids = inst_ids[inst_ids != 0]

            num_instances = len(inst_ids)
            if num_instances == 0:
                continue

            valid_batches += 1
            means = []

            # Compute instance means and variance loss
            var_loss = torch.tensor(0.0, device=device)
            for inst_id in inst_ids:
                inst_mask = (inst == inst_id) & (mask > 0)  # (H, W)
                num_pixels = inst_mask.sum()
                if num_pixels == 0:
                    continue

                # Extract embeddings for this instance
                inst_emb = emb[:, inst_mask]  # (E, num_pixels)
                mean = inst_emb.mean(dim=1, keepdim=True)  # (E, 1)
                means.append(mean.squeeze(1))

                # Variance term: pull toward mean with hinge
                dist_to_mean = torch.norm(inst_emb - mean, dim=0)  # (num_pixels,)
                hinge = F.relu(dist_to_mean - self.delta_v)
                var_loss = var_loss + (hinge ** 2).mean()

            if len(means) == 0:
                continue

            var_loss = var_loss / len(means)
            total_var_loss = total_var_loss + var_loss

            # Distance term: push different instance means apart
            means_tensor = torch.stack(means, dim=0)  # (num_instances, E)
            dist_loss = torch.tensor(0.0, device=device)

            if num_instances > 1:
                num_pairs = 0
                for i in range(num_instances):
                    for j in range(i + 1, num_instances):
                        dist = torch.norm(means_tensor[i] - means_tensor[j])
                        hinge = F.relu(2 * self.delta_d - dist)
                        dist_loss = dist_loss + hinge ** 2
                        num_pairs += 1
                dist_loss = dist_loss / max(num_pairs, 1)

            total_dist_loss = total_dist_loss + dist_loss

            # Regularization term
            reg_loss = torch.norm(means_tensor, dim=1).mean()
            total_reg_loss = total_reg_loss + reg_loss

        if valid_batches == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        total_var_loss = total_var_loss / valid_batches
        total_dist_loss = total_dist_loss / valid_batches
        total_reg_loss = total_reg_loss / valid_batches

        loss = self.alpha * total_var_loss + self.beta * total_dist_loss + self.gamma * total_reg_loss
        return loss


class DirectionLoss(nn.Module):
    """Direction prediction loss.

    Computes smooth L1 loss on predicted direction vectors, masked to only
    pixels that have valid ground truth direction labels.
    """

    def __init__(self, loss_type="smooth_l1"):
        """
        Args:
            loss_type: 'smooth_l1' or 'l1'.
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred_direction, gt_direction, mask):
        """
        Args:
            pred_direction: Predicted direction vectors (B, 2, H, W).
            gt_direction: Ground truth direction vectors (B, 2, H, W).
            mask: Valid pixel mask (B, H, W) or (B, 1, H, W).

        Returns:
            Scalar direction loss.
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)

        # Expand mask to direction channels
        mask = mask.expand_as(pred_direction).float()

        num_valid = mask.sum().clamp(min=1.0)

        if self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(pred_direction * mask, gt_direction * mask, reduction="sum")
        else:
            loss = F.l1_loss(pred_direction * mask, gt_direction * mask, reduction="sum")

        return loss / num_valid


class HDMapNetLoss(nn.Module):
    """Combined loss for HDMapNet training.

    Weighted sum of semantic, discriminative (instance), and direction losses.
    """

    def __init__(
        self,
        semantic_weight=1.0,
        instance_weight=1.0,
        direction_weight=0.2,
        use_focal=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        delta_v=0.5,
        delta_d=3.0,
        direction_loss_type="smooth_l1",
    ):
        """
        Args:
            semantic_weight: Weight for semantic loss.
            instance_weight: Weight for discriminative loss.
            direction_weight: Weight for direction loss.
            use_focal: Whether to use focal loss for semantics.
            focal_alpha: Focal loss alpha parameter.
            focal_gamma: Focal loss gamma parameter.
            delta_v: Discriminative loss pull margin.
            delta_d: Discriminative loss push margin.
            direction_loss_type: 'smooth_l1' or 'l1'.
        """
        super().__init__()
        self.semantic_weight = semantic_weight
        self.instance_weight = instance_weight
        self.direction_weight = direction_weight

        self.semantic_loss = SemanticLoss(
            use_focal=use_focal, focal_alpha=focal_alpha, focal_gamma=focal_gamma
        )
        self.discriminative_loss = DiscriminativeLoss(delta_v=delta_v, delta_d=delta_d)
        self.direction_loss = DirectionLoss(loss_type=direction_loss_type)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict with keys:
                - 'semantic': (B, num_classes, H, W) logits
                - 'instance': (B, embedding_dim, H, W) embeddings
                - 'direction': (B, 2, H, W) direction vectors
            targets: Dict with keys:
                - 'semantic': (B, num_classes, H, W) binary GT
                - 'instance': (B, H, W) integer instance IDs
                - 'direction': (B, 2, H, W) GT direction vectors

        Returns:
            Dict with 'total' loss and individual loss components.
        """
        # Semantic loss
        sem_loss = self.semantic_loss(predictions["semantic"], targets["semantic"])

        # Create semantic mask for instance and direction losses
        # A pixel is valid if it belongs to any semantic class
        semantic_mask = targets["semantic"].any(dim=1).float()  # (B, H, W)

        # Instance discriminative loss
        inst_loss = self.discriminative_loss(
            predictions["instance"], targets["instance"], semantic_mask
        )

        # Direction loss (only on labeled pixels)
        dir_loss = self.direction_loss(
            predictions["direction"], targets["direction"], semantic_mask
        )

        # Combined loss
        total_loss = (
            self.semantic_weight * sem_loss
            + self.instance_weight * inst_loss
            + self.direction_weight * dir_loss
        )

        return {
            "total": total_loss,
            "semantic": sem_loss,
            "instance": inst_loss,
            "direction": dir_loss,
        }
