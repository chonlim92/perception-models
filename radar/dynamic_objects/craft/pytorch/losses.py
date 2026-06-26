"""
Multi-Task Detection Loss for CRAFT (Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer).

Implements loss components for training the CRAFT detection head on nuScenes:
    1. GaussianFocalLoss: Center heatmap classification (CenterPoint-style)
    2. FocalLoss: Per-class detection confidence with class imbalance handling
    3. RegL1Loss: L1 / SmoothL1 regression for bounding box attributes
    4. VelocityLoss: L1 regression for velocity prediction (vx, vy)
    5. CRAFTLoss: Combined multi-task loss with configurable weighting

Loss weighting (from craft_nuscenes.yaml):
    - classification_weight = 1.0
    - bbox_regression_weight = 2.0
    - velocity_weight = 0.2

Detection head specification:
    - num_classes = 10 (nuScenes categories)
    - bbox_code_size = 10 (x, y, z, w, l, h, sin, cos, vx, vy)
    - bias_heatmap = -2.19 (focal loss initialization bias)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_radius(
    height: float, width: float, min_overlap: float = 0.5
) -> float:
    """Compute the minimum Gaussian radius for a bounding box.

    Given an object bounding box of size (height, width), find the smallest radius
    such that a circle of that radius produces an IoU >= min_overlap with the box
    when placed at the box center. Follows the CenterNet/CenterPoint formulation.

    Args:
        height: Object bounding box height in BEV pixels.
        width: Object bounding box width in BEV pixels.
        min_overlap: Minimum required IoU overlap.

    Returns:
        Gaussian radius as a float.
    """
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2.0

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2.0

    return min(r1, r2, r3)


def _generate_gaussian_2d(
    radius: int, sigma: float, device: torch.device = None
) -> torch.Tensor:
    """Generate a 2D Gaussian kernel with the given radius and sigma.

    Args:
        radius: Integer radius of the kernel (kernel size = 2*radius + 1).
        sigma: Standard deviation of the Gaussian.
        device: Device for the output tensor.

    Returns:
        2D Gaussian kernel of shape [2*radius+1, 2*radius+1], peak normalized to 1.
    """
    if device is None:
        device = torch.device("cpu")

    diameter = 2 * radius + 1
    x = torch.arange(0, diameter, dtype=torch.float32, device=device) - radius
    y = x.unsqueeze(1)
    x = x.unsqueeze(0)

    gaussian = torch.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    gaussian[gaussian < torch.finfo(gaussian.dtype).eps * gaussian.max()] = 0.0
    return gaussian


def _draw_gaussian_on_heatmap(
    heatmap: torch.Tensor,
    center: Tuple[int, int],
    radius: int,
    sigma: float,
) -> torch.Tensor:
    """Draw a Gaussian peak onto a heatmap at the specified center location.

    Uses element-wise maximum to handle overlapping Gaussians from nearby objects,
    ensuring the peak value is preserved (no destructive blending).

    Args:
        heatmap: Target heatmap to draw on, shape [H, W].
        center: (x, y) pixel coordinates for the Gaussian center.
        radius: Integer Gaussian radius.
        sigma: Standard deviation of the Gaussian.

    Returns:
        Modified heatmap with the Gaussian drawn (in-place modification, also returned).
    """
    device = heatmap.device
    diameter = 2 * radius + 1
    gaussian = _generate_gaussian_2d(radius, sigma, device=device)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[:2]

    # Compute valid bounds (clipped to heatmap boundaries)
    left = min(x, radius)
    right = min(width - x, radius + 1)
    top = min(y, radius)
    bottom = min(height - y, radius + 1)

    if left + right <= 0 or top + bottom <= 0:
        return heatmap

    # Extract the valid region from the Gaussian kernel
    masked_gaussian = gaussian[
        radius - top: radius + bottom,
        radius - left: radius + right,
    ]

    # Extract the corresponding heatmap region
    masked_heatmap = heatmap[y - top: y + bottom, x - left: x + right]

    # Element-wise maximum (no destructive overwrite)
    torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap


class GaussianFocalLoss(nn.Module):
    """Gaussian Focal Loss for center heatmap prediction (CenterPoint-style).

    Penalizes predictions using a modified focal loss where ground truth heatmaps
    are generated with Gaussian kernels centered at object locations. Positions
    with GT value == 1 are positive samples; positions with GT value < 1 are
    negatively weighted based on how close they are to a positive center.

    This is equivalent to the CornerNet / CenterNet penalty-reduced focal loss:
        Loss_pos = -(1 - p)^alpha * log(p)              for gt == 1
        Loss_neg = -(1 - gt)^beta * p^alpha * log(1-p)  for gt < 1

    Args:
        alpha: Focusing parameter for hard example mining.
        beta: Weighting parameter for positions near object centers (higher beta
              reduces penalty for near-center negatives).
        reduction: Reduction mode ('none', 'mean', 'sum').
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Gaussian focal loss between predicted and target heatmaps.

        Args:
            pred: Predicted heatmap logits (before sigmoid).
                Shape: [B, num_classes, H, W]
            target: Ground truth Gaussian heatmaps (values in [0, 1]).
                Shape: [B, num_classes, H, W]

        Returns:
            Scalar loss value (reduced according to self.reduction).
        """
        pred_sigmoid = torch.clamp(pred.sigmoid(), min=1e-4, max=1.0 - 1e-4)

        # Positive locations: gt == 1
        pos_mask = target.eq(1).float()
        # Negative locations: gt < 1
        neg_mask = target.lt(1).float()

        # Positive loss: -(1 - p)^alpha * log(p)
        pos_loss = -torch.pow(1.0 - pred_sigmoid, self.alpha) * torch.log(pred_sigmoid)
        pos_loss = pos_loss * pos_mask

        # Negative loss: -(1 - gt)^beta * p^alpha * log(1 - p)
        neg_weight = torch.pow(1.0 - target, self.beta)
        neg_loss = -neg_weight * torch.pow(pred_sigmoid, self.alpha) * torch.log(1.0 - pred_sigmoid)
        neg_loss = neg_loss * neg_mask

        # Count number of positive samples for normalization
        num_pos = pos_mask.sum()

        loss = pos_loss.sum() + neg_loss.sum()

        if self.reduction == "mean":
            # Normalize by number of positive samples (at least 1 to avoid div-by-zero)
            loss = loss / torch.clamp(num_pos, min=1.0)
        elif self.reduction == "sum":
            pass  # Already summed
        elif self.reduction == "none":
            loss = pos_loss + neg_loss

        return loss


class FocalLoss(nn.Module):
    """Standard Focal Loss for classification with class imbalance handling.

    Addresses the foreground-background class imbalance in dense object detection
    by down-weighting the loss contribution from easy (well-classified) examples
    and focusing training on hard negatives.

    Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1 else (1-p), and alpha_t = alpha if y=1 else (1-alpha).

    Args:
        alpha: Balancing factor for positive vs negative class. Applied as alpha
               for positive class, (1-alpha) for negative class.
        gamma: Focusing parameter. Higher gamma increases focus on hard examples.
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
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Predicted logits (before sigmoid).
                Shape: [N, num_classes] or [B, num_classes, H, W]
            target: Ground truth class labels (one-hot or integer encoded).
                Shape: same as pred for one-hot, or [N] / [B, H, W] for integer.

        Returns:
            Scalar loss value (reduced according to self.reduction).
        """
        # Handle integer target encoding by converting to one-hot
        if target.dim() != pred.dim():
            num_classes = pred.shape[-1] if pred.dim() == 2 else pred.shape[1]
            if pred.dim() == 2:
                # [N, C] prediction with [N] integer target
                target_one_hot = F.one_hot(target.long(), num_classes).float()
            else:
                # [B, C, H, W] prediction with [B, H, W] integer target
                target_one_hot = F.one_hot(target.long(), num_classes).float()
                # [B, H, W, C] -> [B, C, H, W]
                target_one_hot = target_one_hot.permute(0, 3, 1, 2)
        else:
            target_one_hot = target.float()

        pred_sigmoid = torch.clamp(pred.sigmoid(), min=1e-4, max=1.0 - 1e-4)

        # p_t: probability of the correct class
        p_t = pred_sigmoid * target_one_hot + (1.0 - pred_sigmoid) * (1.0 - target_one_hot)

        # alpha_t: weighting factor
        alpha_t = self.alpha * target_one_hot + (1.0 - self.alpha) * (1.0 - target_one_hot)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        # Binary cross-entropy (per element)
        bce = -torch.log(p_t)

        # Focal loss
        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class RegL1Loss(nn.Module):
    """L1 or Smooth L1 regression loss for bounding box attributes.

    Computes regression loss only at positive (object) locations, ignoring
    background grid positions. Supports both standard L1 and Smooth L1 variants.

    Targets the bounding box code: [dx, dy, dz, w, l, h, sin(yaw), cos(yaw)]
    where dx/dy/dz are sub-voxel center offsets.

    Args:
        smooth: If True, use Smooth L1 (Huber) loss with beta threshold.
                If False, use standard L1 loss.
        beta: Smooth L1 transition threshold (only used when smooth=True).
        reduction: Reduction mode ('none', 'mean', 'sum').
    """

    def __init__(
        self,
        smooth: bool = False,
        beta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute masked regression loss at positive locations.

        Args:
            pred: Predicted regression values.
                Shape: [B, max_objects, code_size] or [B, code_size, H, W]
            target: Ground truth regression values (same shape as pred after gathering).
                Shape: [B, max_objects, code_size]
            mask: Binary mask indicating valid (positive) objects.
                Shape: [B, max_objects] or [B, max_objects, 1]
            index: Optional indices for gathering predictions from spatial maps.
                Shape: [B, max_objects]. If provided, pred is [B, code_size, H, W]
                and will be gathered at the index locations.

        Returns:
            Scalar regression loss.
        """
        if index is not None and pred.dim() == 4:
            # Gather predictions from spatial feature map at object locations
            # pred: [B, C, H, W] -> [B, C, H*W] -> gather at index -> [B, max_objects, C]
            B, C, H, W = pred.shape
            pred_flat = pred.reshape(B, C, H * W)  # [B, C, H*W]
            # index: [B, max_objects] -> [B, 1, max_objects] -> expand to [B, C, max_objects]
            index_expanded = index.unsqueeze(1).expand(B, C, index.shape[1])
            pred = pred_flat.gather(2, index_expanded)  # [B, C, max_objects]
            pred = pred.permute(0, 2, 1)  # [B, max_objects, C]

        # Expand mask to match prediction dimensions
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)  # [B, max_objects, 1]

        # Broadcast mask to match pred shape
        mask = mask.expand_as(pred).float()

        if self.smooth:
            diff = torch.abs(pred * mask - target * mask)
            loss = torch.where(
                diff < self.beta,
                0.5 * diff ** 2 / self.beta,
                diff - 0.5 * self.beta,
            )
        else:
            loss = F.l1_loss(pred * mask, target * mask, reduction="none")

        if self.reduction == "mean":
            # Normalize by number of positive elements
            num_pos = mask.sum()
            return loss.sum() / torch.clamp(num_pos, min=1.0)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class VelocityLoss(nn.Module):
    """L1 regression loss specifically for velocity prediction (vx, vy).

    Computes per-object velocity loss, handling the fact that velocity annotations
    in nuScenes may be invalid (e.g., for stationary objects without Doppler signal).
    Invalid velocity entries are masked out using a separate validity indicator.

    Args:
        reduction: Reduction mode ('none', 'mean', 'sum').
        code_index: Start index of velocity components in the bbox code. For standard
                    nuScenes bbox_code_size=10: [x,y,z,w,l,h,sin,cos,vx,vy] -> index 8.
    """

    def __init__(
        self,
        reduction: str = "mean",
        code_index: int = 8,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.code_index = code_index

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        velocity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity regression loss.

        Args:
            pred: Predicted velocity values [B, max_objects, 2] (vx, vy).
            target: Ground truth velocity values [B, max_objects, 2].
            mask: Binary mask for valid objects [B, max_objects].
            velocity_mask: Optional binary mask for valid velocity annotations
                [B, max_objects]. If None, uses the object mask directly.

        Returns:
            Scalar velocity loss.
        """
        if velocity_mask is not None:
            # Combine object mask with velocity validity mask
            combined_mask = mask.float() * velocity_mask.float()
        else:
            combined_mask = mask.float()

        # Expand mask: [B, max_objects] -> [B, max_objects, 2]
        if combined_mask.dim() == 2:
            combined_mask = combined_mask.unsqueeze(-1).expand_as(pred)

        # L1 loss on velocity components
        loss = F.l1_loss(
            pred * combined_mask,
            target * combined_mask,
            reduction="none",
        )

        if self.reduction == "mean":
            num_pos = combined_mask.sum()
            return loss.sum() / torch.clamp(num_pos, min=1.0)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class HeatmapTargetGenerator(nn.Module):
    """Generates ground truth Gaussian heatmaps from 3D bounding box annotations.

    For each annotated object, a 2D Gaussian blob is drawn on the class-specific
    heatmap channel at the projected BEV center location. The Gaussian radius is
    determined by the object's BEV footprint size.

    This module is used during training to create target heatmaps on-the-fly.

    Args:
        num_classes: Number of object categories (10 for nuScenes).
        bev_height: BEV grid height in pixels.
        bev_width: BEV grid width in pixels.
        point_cloud_range: Spatial extent [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy, vz] for BEV discretization.
        min_radius: Minimum allowed Gaussian radius in pixels.
        min_overlap: Minimum IoU overlap for radius computation.
    """

    def __init__(
        self,
        num_classes: int = 10,
        bev_height: int = 512,
        bev_width: int = 512,
        point_cloud_range: Optional[List[float]] = None,
        voxel_size: Optional[List[float]] = None,
        min_radius: int = 2,
        min_overlap: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.min_radius = min_radius
        self.min_overlap = min_overlap

        if point_cloud_range is None:
            point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if voxel_size is None:
            voxel_size = [0.2, 0.2, 8.0]

        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.x_min = point_cloud_range[0]
        self.y_min = point_cloud_range[1]
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]

    @torch.no_grad()
    def forward(
        self,
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Generate ground truth heatmaps and regression targets.

        Args:
            gt_bboxes: List of ground truth bounding boxes per sample.
                Each tensor shape: [N_i, bbox_code_size] where bbox_code_size=10
                (x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy).
            gt_labels: List of class labels per sample.
                Each tensor shape: [N_i] with integer class indices in [0, num_classes-1].

        Returns:
            Dictionary containing:
                'heatmaps': Gaussian heatmaps [B, num_classes, H, W].
                'indices': Flattened spatial indices for each object [B, max_objects].
                'reg_targets': Regression targets [B, max_objects, bbox_code_size].
                'reg_mask': Binary mask for valid objects [B, max_objects].
                'velocity_mask': Binary mask for valid velocity annotations [B, max_objects].
        """
        batch_size = len(gt_bboxes)
        device = gt_bboxes[0].device if gt_bboxes[0].numel() > 0 else torch.device("cpu")

        # Find maximum objects across batch for padding
        max_objects = max(bbox.shape[0] for bbox in gt_bboxes) if gt_bboxes else 1
        max_objects = max(max_objects, 1)  # At least 1 to avoid empty tensors

        # Initialize outputs
        heatmaps = torch.zeros(
            batch_size, self.num_classes, self.bev_height, self.bev_width,
            device=device, dtype=torch.float32,
        )
        indices = torch.zeros(batch_size, max_objects, device=device, dtype=torch.long)
        bbox_code_size = gt_bboxes[0].shape[-1] if gt_bboxes[0].numel() > 0 else 10
        reg_targets = torch.zeros(
            batch_size, max_objects, bbox_code_size, device=device, dtype=torch.float32,
        )
        reg_mask = torch.zeros(batch_size, max_objects, device=device, dtype=torch.bool)
        velocity_mask = torch.zeros(batch_size, max_objects, device=device, dtype=torch.bool)

        for b in range(batch_size):
            bboxes = gt_bboxes[b]  # [N_i, code_size]
            labels = gt_labels[b]  # [N_i]
            num_objects = bboxes.shape[0]

            if num_objects == 0:
                continue

            for obj_idx in range(num_objects):
                bbox = bboxes[obj_idx]
                cls_id = int(labels[obj_idx].item())

                # Extract BEV center coordinates
                center_x = bbox[0].item()
                center_y = bbox[1].item()

                # Convert world coordinates to BEV pixel coordinates
                px = (center_x - self.x_min) / self.vx
                py = (center_y - self.y_min) / self.vy

                # Check bounds
                if px < 0 or px >= self.bev_width or py < 0 or py >= self.bev_height:
                    continue

                px_int = int(px)
                py_int = int(py)

                # Compute Gaussian radius from object BEV size
                # w, l are at indices 3, 4 in the bbox code
                obj_w = bbox[3].item() / self.vx  # Width in pixels
                obj_l = bbox[4].item() / self.vy  # Length in pixels

                radius = _gaussian_radius(obj_l, obj_w, min_overlap=self.min_overlap)
                radius = max(self.min_radius, int(radius))
                sigma = radius / 3.0  # Standard heuristic: sigma = radius/3

                # Draw Gaussian on the class-specific heatmap channel
                _draw_gaussian_on_heatmap(
                    heatmaps[b, cls_id], center=(px_int, py_int),
                    radius=radius, sigma=sigma,
                )

                # Store flattened spatial index
                flat_index = py_int * self.bev_width + px_int
                indices[b, obj_idx] = flat_index

                # Store regression target (sub-pixel offset + box attributes)
                reg_target = bbox.clone()
                # Replace absolute x, y with sub-pixel offsets
                reg_target[0] = px - px_int  # dx (fractional x offset)
                reg_target[1] = py - py_int  # dy (fractional y offset)
                reg_targets[b, obj_idx] = reg_target

                # Mark as valid
                reg_mask[b, obj_idx] = True

                # Check velocity validity (non-NaN and non-zero-magnitude for annotation)
                vx, vy = bbox[8].item(), bbox[9].item()
                if not (math.isnan(vx) or math.isnan(vy)):
                    velocity_mask[b, obj_idx] = True

        return {
            "heatmaps": heatmaps,
            "indices": indices,
            "reg_targets": reg_targets,
            "reg_mask": reg_mask,
            "velocity_mask": velocity_mask,
        }


class CRAFTLoss(nn.Module):
    """Combined multi-task detection loss for the CRAFT model.

    Aggregates classification, bounding box regression, and velocity losses with
    configurable weighting. Supports auxiliary branch-specific losses for joint
    camera-radar training.

    The total loss is:
        L = cls_weight * L_cls + bbox_weight * L_bbox + velocity_weight * L_vel
            + aux_weight * (L_camera_aux + L_radar_aux)

    Args:
        num_classes: Number of detection categories (10 for nuScenes).
        bbox_code_size: Dimension of the bounding box code vector.
        cls_weight: Weight for classification (heatmap) loss.
        bbox_weight: Weight for bounding box regression loss.
        velocity_weight: Weight for velocity regression loss.
        aux_weight: Weight for auxiliary branch losses (camera-only, radar-only).
        use_gaussian_focal: If True, use GaussianFocalLoss for heatmap. Otherwise
                            use standard FocalLoss.
        focal_alpha: Alpha parameter for focal loss variants.
        focal_gamma: Gamma parameter for focal loss variants.
        smooth_reg: If True, use Smooth L1 for regression instead of L1.
        smooth_beta: Beta threshold for Smooth L1 loss.
        bev_height: BEV grid height for target generation.
        bev_width: BEV grid width for target generation.
        point_cloud_range: Spatial extent for BEV projection.
        voxel_size: Discretization resolution for BEV projection.
    """

    def __init__(
        self,
        num_classes: int = 10,
        bbox_code_size: int = 10,
        cls_weight: float = 1.0,
        bbox_weight: float = 2.0,
        velocity_weight: float = 0.2,
        aux_weight: float = 0.5,
        use_gaussian_focal: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth_reg: bool = False,
        smooth_beta: float = 1.0,
        bev_height: int = 512,
        bev_width: int = 512,
        point_cloud_range: Optional[List[float]] = None,
        voxel_size: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.bbox_code_size = bbox_code_size
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.velocity_weight = velocity_weight
        self.aux_weight = aux_weight

        # Classification loss
        if use_gaussian_focal:
            self.cls_loss_fn = GaussianFocalLoss(
                alpha=focal_gamma,  # GaussianFocalLoss alpha is the focusing param
                beta=4.0,
                reduction="mean",
            )
        else:
            self.cls_loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                reduction="mean",
            )

        # Bounding box regression loss (center offset, z, size, rotation)
        # Velocity is handled separately, so regression covers code indices [0:8]
        self.reg_loss_fn = RegL1Loss(
            smooth=smooth_reg,
            beta=smooth_beta,
            reduction="mean",
        )

        # Velocity loss (vx, vy at code indices [8:10])
        self.velocity_loss_fn = VelocityLoss(
            reduction="mean",
            code_index=8,
        )

        # Target generator
        self.target_generator = HeatmapTargetGenerator(
            num_classes=num_classes,
            bev_height=bev_height,
            bev_width=bev_width,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
        )

        self.use_gaussian_focal = use_gaussian_focal

    def _compute_detection_loss(
        self,
        pred_heatmap: torch.Tensor,
        pred_reg: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute classification and regression losses for a single detection head.

        Args:
            pred_heatmap: Predicted class heatmap [B, num_classes, H, W].
            pred_reg: Predicted regression map [B, bbox_code_size, H, W].
            targets: Dictionary from HeatmapTargetGenerator.forward() containing
                     heatmaps, indices, reg_targets, reg_mask, velocity_mask.

        Returns:
            Dictionary of individual loss components:
                'cls_loss': Classification loss scalar.
                'bbox_loss': Bounding box regression loss scalar.
                'velocity_loss': Velocity regression loss scalar.
        """
        gt_heatmap = targets["heatmaps"]
        indices = targets["indices"]
        reg_targets = targets["reg_targets"]
        reg_mask = targets["reg_mask"]
        velocity_mask = targets["velocity_mask"]

        # Classification loss
        cls_loss = self.cls_loss_fn(pred_heatmap, gt_heatmap)

        # Gather regression predictions at object locations
        B, C, H, W = pred_reg.shape
        pred_reg_flat = pred_reg.reshape(B, C, H * W)  # [B, C, H*W]
        # indices: [B, max_objects] -> [B, 1, max_objects] -> expand to [B, C, max_objects]
        index_expanded = indices.unsqueeze(1).expand(B, C, indices.shape[1])
        pred_reg_gathered = pred_reg_flat.gather(2, index_expanded)  # [B, C, max_objects]
        pred_reg_gathered = pred_reg_gathered.permute(0, 2, 1)  # [B, max_objects, C]

        # Split regression into bbox (indices 0:8) and velocity (indices 8:10)
        bbox_end_idx = self.bbox_code_size - 2  # 8 for standard config
        pred_bbox = pred_reg_gathered[:, :, :bbox_end_idx]  # [B, max_obj, 8]
        pred_vel = pred_reg_gathered[:, :, bbox_end_idx:]   # [B, max_obj, 2]

        target_bbox = reg_targets[:, :, :bbox_end_idx]
        target_vel = reg_targets[:, :, bbox_end_idx:]

        # Bbox regression loss (mask-aware)
        bbox_mask = reg_mask.unsqueeze(-1).expand_as(pred_bbox).float()
        if self.reg_loss_fn.smooth:
            diff = torch.abs(pred_bbox * bbox_mask - target_bbox * bbox_mask)
            bbox_loss_raw = torch.where(
                diff < self.reg_loss_fn.beta,
                0.5 * diff ** 2 / self.reg_loss_fn.beta,
                diff - 0.5 * self.reg_loss_fn.beta,
            )
        else:
            bbox_loss_raw = F.l1_loss(
                pred_bbox * bbox_mask, target_bbox * bbox_mask, reduction="none"
            )
        num_pos = bbox_mask.sum().clamp(min=1.0)
        bbox_loss = bbox_loss_raw.sum() / num_pos

        # Velocity loss (with additional velocity validity masking)
        velocity_loss = self.velocity_loss_fn(
            pred_vel, target_vel, reg_mask, velocity_mask
        )

        return {
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "velocity_loss": velocity_loss,
        }

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute total CRAFT detection loss with all components.

        Args:
            predictions: Model predictions dictionary containing:
                'heatmap': Fused detection heatmap [B, num_classes, H, W].
                'reg': Fused regression map [B, bbox_code_size, H, W].
                'camera_heatmap' (optional): Camera-only heatmap [B, num_classes, H, W].
                'camera_reg' (optional): Camera-only regression [B, bbox_code_size, H, W].
                'radar_heatmap' (optional): Radar-only heatmap [B, num_classes, H, W].
                'radar_reg' (optional): Radar-only regression [B, bbox_code_size, H, W].
            gt_bboxes: List of GT bounding boxes per sample.
                Each [N_i, 10]: (x, y, z, w, l, h, sin, cos, vx, vy).
            gt_labels: List of GT class labels per sample. Each [N_i].

        Returns:
            Dictionary containing:
                'total_loss': Weighted sum of all losses (for backward()).
                'cls_loss': Classification loss (for logging).
                'bbox_loss': Bbox regression loss (for logging).
                'velocity_loss': Velocity loss (for logging).
                'camera_aux_loss': Camera auxiliary loss (for logging, 0 if not present).
                'radar_aux_loss': Radar auxiliary loss (for logging, 0 if not present).
        """
        # Generate ground truth targets
        targets = self.target_generator(gt_bboxes, gt_labels)

        # Main fused detection loss
        main_losses = self._compute_detection_loss(
            pred_heatmap=predictions["heatmap"],
            pred_reg=predictions["reg"],
            targets=targets,
        )

        cls_loss = main_losses["cls_loss"]
        bbox_loss = main_losses["bbox_loss"]
        velocity_loss = main_losses["velocity_loss"]

        # Compute total main loss with weights
        total_loss = (
            self.cls_weight * cls_loss
            + self.bbox_weight * bbox_loss
            + self.velocity_weight * velocity_loss
        )

        # Auxiliary camera branch loss (for joint training)
        camera_aux_loss = torch.tensor(0.0, device=cls_loss.device)
        if "camera_heatmap" in predictions and "camera_reg" in predictions:
            camera_losses = self._compute_detection_loss(
                pred_heatmap=predictions["camera_heatmap"],
                pred_reg=predictions["camera_reg"],
                targets=targets,
            )
            camera_aux_loss = (
                self.cls_weight * camera_losses["cls_loss"]
                + self.bbox_weight * camera_losses["bbox_loss"]
                + self.velocity_weight * camera_losses["velocity_loss"]
            )
            total_loss = total_loss + self.aux_weight * camera_aux_loss

        # Auxiliary radar branch loss (for joint training)
        radar_aux_loss = torch.tensor(0.0, device=cls_loss.device)
        if "radar_heatmap" in predictions and "radar_reg" in predictions:
            radar_losses = self._compute_detection_loss(
                pred_heatmap=predictions["radar_heatmap"],
                pred_reg=predictions["radar_reg"],
                targets=targets,
            )
            radar_aux_loss = (
                self.cls_weight * radar_losses["cls_loss"]
                + self.bbox_weight * radar_losses["bbox_loss"]
                + self.velocity_weight * radar_losses["velocity_loss"]
            )
            total_loss = total_loss + self.aux_weight * radar_aux_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "velocity_loss": velocity_loss.detach(),
            "camera_aux_loss": camera_aux_loss.detach(),
            "radar_aux_loss": radar_aux_loss.detach(),
        }


def build_craft_loss(
    num_classes: int = 10,
    bbox_code_size: int = 10,
    cls_weight: float = 1.0,
    bbox_weight: float = 2.0,
    velocity_weight: float = 0.2,
    aux_weight: float = 0.5,
    use_gaussian_focal: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    smooth_reg: bool = False,
    smooth_beta: float = 1.0,
    bev_height: int = 512,
    bev_width: int = 512,
    point_cloud_range: Optional[List[float]] = None,
    voxel_size: Optional[List[float]] = None,
) -> CRAFTLoss:
    """Factory function to build the CRAFT multi-task detection loss.

    Args:
        num_classes: Number of detection categories (10 for nuScenes).
        bbox_code_size: Bounding box code dimension (10: x,y,z,w,l,h,sin,cos,vx,vy).
        cls_weight: Classification loss weight.
        bbox_weight: Bounding box regression loss weight.
        velocity_weight: Velocity regression loss weight.
        aux_weight: Auxiliary branch loss weight.
        use_gaussian_focal: Whether to use Gaussian focal loss for heatmap.
        focal_alpha: Alpha for focal loss.
        focal_gamma: Gamma for focal loss.
        smooth_reg: Whether to use Smooth L1 for regression.
        smooth_beta: Beta for Smooth L1.
        bev_height: BEV grid height.
        bev_width: BEV grid width.
        point_cloud_range: Spatial extent [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: Discretization resolution [vx, vy, vz].

    Returns:
        Configured CRAFTLoss instance.
    """
    return CRAFTLoss(
        num_classes=num_classes,
        bbox_code_size=bbox_code_size,
        cls_weight=cls_weight,
        bbox_weight=bbox_weight,
        velocity_weight=velocity_weight,
        aux_weight=aux_weight,
        use_gaussian_focal=use_gaussian_focal,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        smooth_reg=smooth_reg,
        smooth_beta=smooth_beta,
        bev_height=bev_height,
        bev_width=bev_width,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
    )


def build_craft_loss_from_config(config: Dict) -> CRAFTLoss:
    """Build CRAFTLoss from a configuration dictionary (e.g., parsed from YAML).

    Expected config structure (matching craft_nuscenes.yaml):
        loss_weights:
            classification: 1.0
            bbox_regression: 2.0
            velocity: 0.2
        detection_head:
            num_classes: 10
            bbox_code_size: 10
            bias_heatmap: -2.19

    Args:
        config: Configuration dictionary.

    Returns:
        Configured CRAFTLoss instance.
    """
    loss_weights = config.get("loss_weights", {})
    detection_head = config.get("detection_head", {})
    model = config.get("model", {})

    cls_weight = loss_weights.get("classification", 1.0)
    bbox_weight = loss_weights.get("bbox_regression", 2.0)
    velocity_weight = loss_weights.get("velocity", 0.2)
    aux_weight = loss_weights.get("auxiliary", 0.5)

    num_classes = detection_head.get("num_classes", 10)
    bbox_code_size = detection_head.get("bbox_code_size", 10)

    point_cloud_range = model.get(
        "point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    )
    voxel_size = model.get("voxel_size", [0.2, 0.2, 8.0])

    bev_height = int(round(
        (point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]
    ))
    bev_width = int(round(
        (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]
    ))

    return build_craft_loss(
        num_classes=num_classes,
        bbox_code_size=bbox_code_size,
        cls_weight=cls_weight,
        bbox_weight=bbox_weight,
        velocity_weight=velocity_weight,
        aux_weight=aux_weight,
        bev_height=bev_height,
        bev_width=bev_width,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
    )


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building CRAFT Loss Module...")
    loss_module = build_craft_loss(
        num_classes=10,
        bbox_code_size=10,
        cls_weight=1.0,
        bbox_weight=2.0,
        velocity_weight=0.2,
        aux_weight=0.5,
        use_gaussian_focal=True,
        bev_height=128,  # Smaller for testing
        bev_width=128,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=[0.8, 0.8, 8.0],  # Coarser grid for testing
    ).to(device)

    print(f"  Loss weights: cls={loss_module.cls_weight}, "
          f"bbox={loss_module.bbox_weight}, vel={loss_module.velocity_weight}")
    print(f"  Auxiliary weight: {loss_module.aux_weight}")
    print(f"  Number of classes: {loss_module.num_classes}")
    print(f"  BBox code size: {loss_module.bbox_code_size}")
    print()

    # Create dummy predictions (simulating model output)
    batch_size = 2
    H, W = 128, 128
    num_classes = 10
    bbox_code_size = 10

    predictions = {
        "heatmap": torch.randn(batch_size, num_classes, H, W, device=device),
        "reg": torch.randn(batch_size, bbox_code_size, H, W, device=device),
        # Auxiliary camera predictions
        "camera_heatmap": torch.randn(batch_size, num_classes, H, W, device=device),
        "camera_reg": torch.randn(batch_size, bbox_code_size, H, W, device=device),
        # Auxiliary radar predictions
        "radar_heatmap": torch.randn(batch_size, num_classes, H, W, device=device),
        "radar_reg": torch.randn(batch_size, bbox_code_size, H, W, device=device),
    }

    # Create dummy ground truth annotations
    # Sample 1: 5 objects
    gt_bboxes_1 = torch.tensor([
        [10.0, 5.0, -1.0, 4.5, 2.0, 1.8, 0.0, 1.0, 3.0, 0.5],   # car
        [-20.0, 15.0, -0.5, 4.2, 1.8, 1.6, 0.5, 0.87, -2.0, 1.0],  # car
        [30.0, -10.0, -1.2, 0.8, 0.6, 1.7, 0.0, 1.0, 1.5, 0.0],  # pedestrian
        [5.0, 25.0, -0.8, 2.0, 0.8, 1.2, 0.71, 0.71, 5.0, 2.0],  # bicycle
        [-15.0, -30.0, -0.3, 10.0, 3.0, 3.5, 1.0, 0.0, 0.0, 0.0],  # truck
    ], device=device)
    gt_labels_1 = torch.tensor([0, 0, 2, 3, 5], device=device)

    # Sample 2: 3 objects
    gt_bboxes_2 = torch.tensor([
        [0.0, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, 1.0, 0.0, 0.0],    # car
        [40.0, 20.0, -0.5, 1.0, 0.5, 1.8, 0.0, 1.0, 2.0, -1.0],  # pedestrian
        [-5.0, 10.0, -0.7, 6.0, 2.5, 2.8, 0.38, 0.92, 8.0, 0.0],  # bus
    ], device=device)
    gt_labels_2 = torch.tensor([0, 2, 7], device=device)

    gt_bboxes = [gt_bboxes_1, gt_bboxes_2]
    gt_labels = [gt_labels_1, gt_labels_2]

    # Forward pass
    print("Computing losses...")
    loss_dict = loss_module(predictions, gt_bboxes, gt_labels)

    print("\nLoss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")

    # Verify gradients flow
    total_loss = loss_dict["total_loss"]
    total_loss.backward()
    print(f"\nGradient check - heatmap grad norm: "
          f"{predictions['heatmap'].grad.norm().item():.6f}")
    print(f"Gradient check - reg grad norm: "
          f"{predictions['reg'].grad.norm().item():.6f}")

    # Test individual loss components
    print("\n--- Individual Loss Component Tests ---")

    # Gaussian Focal Loss
    gfl = GaussianFocalLoss(alpha=2.0, beta=4.0)
    pred_hm = torch.randn(2, 10, 64, 64, device=device)
    target_hm = torch.zeros(2, 10, 64, 64, device=device)
    target_hm[0, 0, 32, 32] = 1.0  # Single positive center
    target_hm[0, 0, 31:34, 31:34] = 0.5  # Gaussian surroundings
    gfl_loss = gfl(pred_hm, target_hm)
    print(f"GaussianFocalLoss: {gfl_loss.item():.4f}")

    # Focal Loss
    fl = FocalLoss(alpha=0.25, gamma=2.0)
    pred_cls = torch.randn(16, 10, device=device)
    target_cls = torch.randint(0, 10, (16,), device=device)
    fl_loss = fl(pred_cls, target_cls)
    print(f"FocalLoss: {fl_loss.item():.4f}")

    # RegL1Loss
    reg_l1 = RegL1Loss(smooth=False)
    pred_r = torch.randn(2, 50, 8, device=device)
    target_r = torch.randn(2, 50, 8, device=device)
    mask_r = torch.zeros(2, 50, device=device)
    mask_r[0, :5] = 1.0
    mask_r[1, :3] = 1.0
    reg_loss = reg_l1(pred_r, target_r, mask_r)
    print(f"RegL1Loss: {reg_loss.item():.4f}")

    # VelocityLoss
    vel_loss_fn = VelocityLoss()
    pred_v = torch.randn(2, 50, 2, device=device)
    target_v = torch.randn(2, 50, 2, device=device)
    mask_v = torch.zeros(2, 50, device=device)
    mask_v[0, :5] = 1.0
    mask_v[1, :3] = 1.0
    vel_mask = mask_v.clone()
    vel_mask[0, 4] = 0.0  # Mark one velocity as invalid
    vel_loss = vel_loss_fn(pred_v, target_v, mask_v, vel_mask)
    print(f"VelocityLoss: {vel_loss.item():.4f}")

    # Config-based construction
    print("\n--- Config-based Construction ---")
    test_config = {
        "loss_weights": {
            "classification": 1.0,
            "bbox_regression": 2.0,
            "velocity": 0.2,
            "auxiliary": 0.5,
        },
        "detection_head": {
            "num_classes": 10,
            "bbox_code_size": 10,
            "bias_heatmap": -2.19,
        },
        "model": {
            "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "voxel_size": [0.2, 0.2, 8.0],
        },
    }
    config_loss = build_craft_loss_from_config(test_config)
    print(f"Config loss - cls_weight: {config_loss.cls_weight}")
    print(f"Config loss - bbox_weight: {config_loss.bbox_weight}")
    print(f"Config loss - velocity_weight: {config_loss.velocity_weight}")
    print(f"Config loss - BEV size: {config_loss.target_generator.bev_height}x"
          f"{config_loss.target_generator.bev_width}")

    print("\nAll checks passed!")
