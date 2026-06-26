"""
Loss functions for PointNet++ training.

Implements task-specific losses for classification, detection,
and segmentation, including bin-based angle loss and corner loss
for 3D object detection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_box_corners(
    centers: torch.Tensor,
    sizes: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the 8 corners of 3D bounding boxes.

    The box is defined by center (x, y, z), size (w, h, l), and yaw angle
    around the Z-axis (up). The corners are ordered as:
        4 bottom corners followed by 4 top corners.

    Args:
        centers: Box centers, shape (..., 3) - (x, y, z)
        sizes: Box dimensions, shape (..., 3) - (w, h, l) i.e. (dx, dy, dz)
        angles: Yaw angles in radians, shape (..., 1) or (...,)

    Returns:
        corners: 8 corner coordinates, shape (..., 8, 3)
    """
    # Ensure angles is at least 1D in the last dim
    if angles.shape[-1] != 1:
        angles = angles.unsqueeze(-1)

    # Half dimensions
    w = sizes[..., 0:1] / 2.0  # half width (x)
    h = sizes[..., 1:2] / 2.0  # half height (y)
    l = sizes[..., 2:3] / 2.0  # half length (z)

    # Rotation around Z-axis
    cos_a = torch.cos(angles)  # (..., 1)
    sin_a = torch.sin(angles)  # (..., 1)

    # 8 corners in local frame (before rotation):
    # Bottom 4: (-w,-h,-l), (+w,-h,-l), (+w,+h,-l), (-w,+h,-l)
    # Top 4:    (-w,-h,+l), (+w,-h,+l), (+w,+h,+l), (-w,+h,+l)
    # Using x-forward, y-left, z-up convention
    dx = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=-1)  # (..., 8)
    dy = torch.cat([h, -h, -h, h, h, -h, -h, h], dim=-1)  # (..., 8)
    dz = torch.cat([-l, -l, -l, -l, l, l, l, l], dim=-1)  # (..., 8)

    # Apply rotation (yaw around z-axis)
    # x' = x*cos - y*sin
    # y' = x*sin + y*cos
    rotated_dx = dx * cos_a - dy * sin_a
    rotated_dy = dx * sin_a + dy * cos_a
    rotated_dz = dz  # z unchanged by yaw

    # Stack and add center
    corners = torch.stack([rotated_dx, rotated_dy, rotated_dz], dim=-1)  # (..., 8, 3)
    corners = corners + centers.unsqueeze(-2)  # broadcast center

    return corners


class PointNetPPClassificationLoss(nn.Module):
    """
    Classification loss: standard cross-entropy.

    Args:
        label_smoothing: Label smoothing factor (default 0.0)
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Args:
            predictions: Class logits, shape (B, num_classes)
            targets: Ground truth class indices, shape (B,)

        Returns:
            Dictionary with 'total_loss' and 'cls_loss'
        """
        cls_loss = F.cross_entropy(
            predictions, targets, label_smoothing=self.label_smoothing
        )
        return {"total_loss": cls_loss, "cls_loss": cls_loss}


class PointNetPPDetectionLoss(nn.Module):
    """
    Detection loss combining multiple terms:
    - Center regression: Smooth L1
    - Size regression: Smooth L1
    - Angle: Bin classification (cross-entropy) + residual regression (Smooth L1)
    - Class: Cross-entropy
    - Corner loss: L1 between predicted and GT box corners

    Args:
        num_angle_bins: Number of bins for angle prediction
        corner_loss_weight: Weight for the corner loss term
        size_loss_weight: Weight for size regression loss
        center_loss_weight: Weight for center regression loss
        angle_loss_weight: Weight for angle loss
        cls_loss_weight: Weight for classification loss
    """

    def __init__(
        self,
        num_angle_bins: int = 12,
        corner_loss_weight: float = 1.0,
        size_loss_weight: float = 1.0,
        center_loss_weight: float = 2.0,
        angle_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.num_angle_bins = num_angle_bins
        self.corner_loss_weight = corner_loss_weight
        self.size_loss_weight = size_loss_weight
        self.center_loss_weight = center_loss_weight
        self.angle_loss_weight = angle_loss_weight
        self.cls_loss_weight = cls_loss_weight

        # Angle bin boundaries
        self.bin_size = 2 * math.pi / num_angle_bins

    def _angle_to_bin_and_residual(self, angle: torch.Tensor):
        """Convert continuous angle to bin index and residual."""
        # Normalize angle to [0, 2*pi)
        angle = angle % (2 * math.pi)
        bin_idx = (angle / self.bin_size).long()
        bin_idx = torch.clamp(bin_idx, 0, self.num_angle_bins - 1)
        residual = angle - bin_idx.float() * self.bin_size
        # Normalize residual to [-1, 1]
        residual = residual / (self.bin_size / 2.0) - 1.0
        return bin_idx, residual

    def forward(
        self,
        predictions: dict,
        targets: dict,
    ) -> dict:
        """
        Args:
            predictions: Dictionary from DetectionHead with keys:
                'center': (B, N, 3)
                'size': (B, N, 3)
                'angle_cls': (B, N, num_angle_bins)
                'angle_res': (B, N, num_angle_bins)
                'cls_scores': (B, N, num_classes)

            targets: Dictionary with ground truth:
                'center': (B, N, 3) - GT center offsets
                'size': (B, N, 3) - GT sizes
                'angle': (B, N, 1) - GT angles in radians
                'cls': (B, N) - GT class labels (0 for background)
                'mask': (B, N) - Binary mask for valid (foreground) proposals

        Returns:
            Dictionary with individual and total losses
        """
        pred_center = predictions["center"]
        pred_size = predictions["size"]
        pred_angle_cls = predictions["angle_cls"]
        pred_angle_res = predictions["angle_res"]
        pred_cls = predictions["cls_scores"]

        gt_center = targets["center"]
        gt_size = targets["size"]
        gt_angle = targets["angle"]
        gt_cls = targets["cls"]
        mask = targets["mask"]  # (B, N) foreground mask

        # Number of valid proposals
        num_pos = torch.clamp(mask.sum(), min=1.0)

        # --- Classification loss (all proposals) ---
        B, N, C = pred_cls.shape
        cls_loss = F.cross_entropy(
            pred_cls.reshape(B * N, C),
            gt_cls.reshape(B * N),
            reduction="mean",
        )

        # --- Center regression loss (foreground only) ---
        center_loss = F.smooth_l1_loss(
            pred_center * mask.unsqueeze(-1),
            gt_center * mask.unsqueeze(-1),
            reduction="sum",
        ) / num_pos

        # --- Size regression loss (foreground only) ---
        size_loss = F.smooth_l1_loss(
            pred_size * mask.unsqueeze(-1),
            gt_size * mask.unsqueeze(-1),
            reduction="sum",
        ) / num_pos

        # --- Angle loss (foreground only) ---
        gt_angle_squeezed = gt_angle.squeeze(-1)  # (B, N)
        bin_idx, bin_residual = self._angle_to_bin_and_residual(gt_angle_squeezed)

        # Angle bin classification
        angle_cls_loss = F.cross_entropy(
            pred_angle_cls[mask.bool()].reshape(-1, self.num_angle_bins),
            bin_idx[mask.bool()].reshape(-1),
            reduction="mean",
        ) if mask.sum() > 0 else torch.tensor(0.0, device=pred_center.device)

        # Angle residual regression (only for the correct bin)
        if mask.sum() > 0:
            pred_angle_res_selected = torch.gather(
                pred_angle_res[mask.bool()],
                1,
                bin_idx[mask.bool()].unsqueeze(-1),
            ).squeeze(-1)
            gt_residual = bin_residual[mask.bool()]
            angle_res_loss = F.smooth_l1_loss(
                pred_angle_res_selected, gt_residual, reduction="mean"
            )
        else:
            angle_res_loss = torch.tensor(0.0, device=pred_center.device)

        angle_loss = angle_cls_loss + angle_res_loss

        # --- Corner loss (foreground only) ---
        if mask.sum() > 0:
            # Reconstruct predicted angle from bins
            pred_bin = torch.argmax(pred_angle_cls, dim=-1)  # (B, N)
            pred_res = torch.gather(
                pred_angle_res, 2, pred_bin.unsqueeze(-1)
            ).squeeze(-1)
            pred_angle_cont = (
                pred_bin.float() * self.bin_size
                + (pred_res + 1.0) * (self.bin_size / 2.0)
            )

            # Compute corners for predicted and GT boxes (foreground only)
            fg_mask = mask.bool()
            pred_corners = compute_box_corners(
                pred_center[fg_mask],
                pred_size[fg_mask],
                pred_angle_cont[fg_mask].unsqueeze(-1),
            )
            gt_corners = compute_box_corners(
                gt_center[fg_mask],
                gt_size[fg_mask],
                gt_angle[fg_mask],
            )
            corner_loss = F.l1_loss(pred_corners, gt_corners, reduction="mean")
        else:
            corner_loss = torch.tensor(0.0, device=pred_center.device)

        # --- Total loss ---
        total_loss = (
            self.cls_loss_weight * cls_loss
            + self.center_loss_weight * center_loss
            + self.size_loss_weight * size_loss
            + self.angle_loss_weight * angle_loss
            + self.corner_loss_weight * corner_loss
        )

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "center_loss": center_loss,
            "size_loss": size_loss,
            "angle_loss": angle_loss,
            "angle_cls_loss": angle_cls_loss,
            "angle_res_loss": angle_res_loss,
            "corner_loss": corner_loss,
        }


class PointNetPPSegmentationLoss(nn.Module):
    """
    Segmentation loss: per-point cross-entropy with optional class weights.

    Args:
        num_classes: Number of segmentation classes
        class_weights: Optional tensor of per-class weights, shape (num_classes,)
        ignore_index: Class index to ignore in loss computation (default -1)
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Args:
            predictions: Per-point logits, shape (B, N, num_classes)
            targets: Per-point class labels, shape (B, N)

        Returns:
            Dictionary with 'total_loss' and 'seg_loss'
        """
        B, N, C = predictions.shape

        # Reshape for cross_entropy: (B*N, C) and (B*N,)
        pred_flat = predictions.reshape(B * N, C)
        target_flat = targets.reshape(B * N)

        seg_loss = F.cross_entropy(
            pred_flat,
            target_flat,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        return {"total_loss": seg_loss, "seg_loss": seg_loss}
