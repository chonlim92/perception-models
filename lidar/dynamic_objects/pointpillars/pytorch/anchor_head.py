"""SSD-style anchor-based detection head for PointPillars.

This module implements the AnchorHead used in PointPillars for 3D object detection
from LiDAR bird's-eye view (BEV) feature maps. For each spatial location the head
predicts per-anchor class scores, 7-DoF bounding-box regressions, and an optional
binary direction classifier to resolve heading ambiguity.

Reference:
    Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds",
    CVPR 2019.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torchvision.ops import nms as torchvision_nms
except ImportError:
    torchvision_nms = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _limit_period(vals: Tensor, offset: float, period: float) -> Tensor:
    """Limit the angle value to [offset*period, (1-offset)*period].

    Args:
        vals: Tensor of angle values.
        offset: Fractional offset for the lower bound (typically 0.5).
        period: The full angular period (typically 2*pi).

    Returns:
        Angle values wrapped into the specified range.
    """
    return vals - torch.floor(vals / period + offset) * period


def _rotate_points_bev(
    centers_x: Tensor, centers_y: Tensor, dx: Tensor, dy: Tensor, angle: Tensor
) -> Tensor:
    """Compute four BEV corner points of rotated boxes for NMS.

    Args:
        centers_x: (N,) x coordinates of box centres.
        centers_y: (N,) y coordinates of box centres.
        dx: (N,) half-width along local x axis.
        dy: (N,) half-length along local y axis.
        angle: (N,) yaw angle in radians.

    Returns:
        Tensor of shape (N, 4, 2) giving the four corner coordinates.
    """
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # Local corners (half extents)
    corners_x = torch.stack([dx, dx, -dx, -dx], dim=-1)  # (N, 4)
    corners_y = torch.stack([dy, -dy, -dy, dy], dim=-1)  # (N, 4)

    # Rotate
    rot_x = cos_a.unsqueeze(-1) * corners_x - sin_a.unsqueeze(-1) * corners_y
    rot_y = sin_a.unsqueeze(-1) * corners_x + cos_a.unsqueeze(-1) * corners_y

    # Translate
    rot_x = rot_x + centers_x.unsqueeze(-1)
    rot_y = rot_y + centers_y.unsqueeze(-1)

    corners = torch.stack([rot_x, rot_y], dim=-1)  # (N, 4, 2)
    return corners


def _bev_box_to_axis_aligned(boxes_bev: Tensor) -> Tensor:
    """Convert BEV rotated boxes to axis-aligned bounding boxes for NMS fallback.

    Args:
        boxes_bev: (N, 7) boxes with columns (x, y, z, w, l, h, theta).

    Returns:
        (N, 4) axis-aligned boxes as (x1, y1, x2, y2) in BEV.
    """
    x = boxes_bev[:, 0]
    y = boxes_bev[:, 1]
    w = boxes_bev[:, 3]
    l = boxes_bev[:, 4]  # noqa: E741
    theta = boxes_bev[:, 6]

    cos_t = torch.abs(torch.cos(theta))
    sin_t = torch.abs(torch.sin(theta))

    # Bounding extent of rotated rectangle
    half_ext_x = 0.5 * (w * cos_t + l * sin_t)
    half_ext_y = 0.5 * (w * sin_t + l * cos_t)

    x1 = x - half_ext_x
    y1 = y - half_ext_y
    x2 = x + half_ext_x
    y2 = y + half_ext_y

    return torch.stack([x1, y1, x2, y2], dim=-1)


def _iou_2d_axis_aligned(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Compute pairwise IoU between two sets of axis-aligned 2D boxes.

    Args:
        boxes_a: (M, 4) boxes as (x1, y1, x2, y2).
        boxes_b: (N, 4) boxes as (x1, y1, x2, y2).

    Returns:
        (M, N) IoU matrix.
    """
    M = boxes_a.shape[0]
    N = boxes_b.shape[0]

    # Expand for broadcasting: (M, 1, 4) vs (1, N, 4)
    a = boxes_a.unsqueeze(1).expand(M, N, 4)
    b = boxes_b.unsqueeze(0).expand(M, N, 4)

    inter_x1 = torch.max(a[..., 0], b[..., 0])
    inter_y1 = torch.max(a[..., 1], b[..., 1])
    inter_x2 = torch.min(a[..., 2], b[..., 2])
    inter_y2 = torch.min(a[..., 3], b[..., 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    union = area_a + area_b - inter_area
    iou = inter_area / torch.clamp(union, min=1e-6)
    return iou


# ---------------------------------------------------------------------------
# AnchorHead
# ---------------------------------------------------------------------------


class AnchorHead(nn.Module):
    """SSD-style detection head for PointPillars 3D object detection.

    Predicts class scores, 7-DoF bounding-box regressions, and direction
    classification for a set of pre-defined anchors at each spatial location
    of the backbone BEV feature map.

    Args:
        in_channels: Number of input feature channels from the backbone.
        num_classes: Number of object classes to detect (e.g., 3 for Car, Ped, Cyc).
        num_anchors_per_location: Total anchors per spatial cell (e.g., 2 rotations
            per class * num_classes = 6).
        box_code_size: Dimensionality of the box regression target (default 7:
            dx, dy, dz, dw, dl, dh, dtheta).
        use_direction_classifier: Whether to use the binary direction branch.
        direction_offset: Offset angle (radians) for direction classification
            (default pi/4).
        nms_score_threshold: Minimum score to keep a detection before NMS.
        nms_iou_threshold: IoU threshold for NMS suppression.
        nms_pre_max_size: Maximum detections to keep before NMS (top-k by score).
        nms_post_max_size: Maximum detections to keep after NMS.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors_per_location: int,
        box_code_size: int = 7,
        use_direction_classifier: bool = True,
        direction_offset: float = 0.785,
        nms_score_threshold: float = 0.1,
        nms_iou_threshold: float = 0.01,
        nms_pre_max_size: int = 1000,
        nms_post_max_size: int = 300,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location
        self.box_code_size = box_code_size
        self.use_direction_classifier = use_direction_classifier
        self.direction_offset = direction_offset
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_pre_max_size = nms_pre_max_size
        self.nms_post_max_size = nms_post_max_size

        # Classification head: 3x3 conv -> 1x1 conv
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                num_anchors_per_location * num_classes,
                kernel_size=1,
                bias=True,
            ),
        )

        # Box regression head: 3x3 conv -> 1x1 conv
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                num_anchors_per_location * box_code_size,
                kernel_size=1,
                bias=True,
            ),
        )

        # Direction classification head (binary: forward / backward)
        if use_direction_classifier:
            self.dir_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels,
                    num_anchors_per_location * 2,
                    kernel_size=1,
                    bias=True,
                ),
            )
        else:
            self.dir_conv = None

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialize convolutional layers with appropriate distributions."""
        pi = 0.01  # Prior probability for focal-loss style init
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

        # Bias init for classification head (focal loss prior)
        cls_final_conv = self.cls_conv[-1]
        nn.init.constant_(cls_final_conv.bias, -math.log((1 - pi) / pi))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Forward pass through the detection head.

        Args:
            x: Backbone feature map of shape (B, C, H, W).

        Returns:
            cls_preds: Class predictions of shape
                (B, num_anchors_per_location * num_classes, H, W).
            reg_preds: Box regression predictions of shape
                (B, num_anchors_per_location * box_code_size, H, W).
            dir_preds: Direction predictions of shape
                (B, num_anchors_per_location * 2, H, W), or None if the
                direction classifier is disabled.
        """
        cls_preds: Tensor = self.cls_conv(x)
        reg_preds: Tensor = self.reg_conv(x)
        dir_preds: Optional[Tensor] = None
        if self.dir_conv is not None:
            dir_preds = self.dir_conv(x)
        return cls_preds, reg_preds, dir_preds

    # ------------------------------------------------------------------
    # Box decoding
    # ------------------------------------------------------------------

    def decode_boxes(self, reg_preds: Tensor, anchors: Tensor) -> Tensor:
        """Decode regression predictions relative to anchor boxes.

        The encoding follows the standard residual encoding:
            dx = (x_gt - x_a) / diag_a
            dy = (y_gt - y_a) / diag_a
            dz = (z_gt - z_a) / h_a
            dw = log(w_gt / w_a)
            dl = log(l_gt / l_a)
            dh = log(h_gt / h_a)
            dtheta = theta_gt - theta_a

        Args:
            reg_preds: (N, 7) predicted regression deltas.
            anchors: (N, 7) anchor boxes as (x, y, z, w, l, h, theta).

        Returns:
            (N, 7) decoded boxes in absolute coordinates.
        """
        xa, ya, za = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        wa, la, ha = anchors[:, 3], anchors[:, 4], anchors[:, 5]
        theta_a = anchors[:, 6]

        diag_a = torch.sqrt(wa ** 2 + la ** 2)

        dx = reg_preds[:, 0]
        dy = reg_preds[:, 1]
        dz = reg_preds[:, 2]
        dw = reg_preds[:, 3]
        dl = reg_preds[:, 4]
        dh = reg_preds[:, 5]
        dtheta = reg_preds[:, 6]

        x_decoded = dx * diag_a + xa
        y_decoded = dy * diag_a + ya
        z_decoded = dz * ha + za
        w_decoded = torch.exp(dw) * wa
        l_decoded = torch.exp(dl) * la
        h_decoded = torch.exp(dh) * ha
        theta_decoded = dtheta + theta_a

        return torch.stack(
            [x_decoded, y_decoded, z_decoded, w_decoded, l_decoded, h_decoded, theta_decoded],
            dim=-1,
        )

    # ------------------------------------------------------------------
    # Encode boxes (for target generation)
    # ------------------------------------------------------------------

    def encode_boxes(self, gt_boxes: Tensor, anchors: Tensor) -> Tensor:
        """Encode ground-truth boxes relative to anchors for regression targets.

        Args:
            gt_boxes: (N, 7) ground-truth boxes (x, y, z, w, l, h, theta).
            anchors: (N, 7) matched anchor boxes.

        Returns:
            (N, 7) encoded regression targets.
        """
        xa, ya, za = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        wa, la, ha = anchors[:, 3], anchors[:, 4], anchors[:, 5]
        theta_a = anchors[:, 6]

        xg, yg, zg = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        wg, lg, hg = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        theta_g = gt_boxes[:, 6]

        diag_a = torch.sqrt(wa ** 2 + la ** 2)

        dx = (xg - xa) / diag_a
        dy = (yg - ya) / diag_a
        dz = (zg - za) / ha
        dw = torch.log(wg / torch.clamp(wa, min=1e-6))
        dl = torch.log(lg / torch.clamp(la, min=1e-6))
        dh = torch.log(hg / torch.clamp(ha, min=1e-6))
        dtheta = theta_g - theta_a

        return torch.stack([dx, dy, dz, dw, dl, dh, dtheta], dim=-1)

    # ------------------------------------------------------------------
    # NMS post-processing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def nms_post_process(
        self,
        cls_preds: Tensor,
        reg_preds: Tensor,
        dir_preds: Optional[Tensor],
        anchors: Tensor,
    ) -> List[Dict[str, Tensor]]:
        """Decode predictions and apply NMS to produce final detections.

        Args:
            cls_preds: Raw class logits of shape (B, H*W*num_anchors, num_classes).
            reg_preds: Raw regression deltas of shape (B, H*W*num_anchors, 7).
            dir_preds: Raw direction logits of shape (B, H*W*num_anchors, 2) or None.
            anchors: Anchor boxes of shape (H*W*num_anchors, 7) shared across batch,
                or (B, H*W*num_anchors, 7) if per-sample anchors are used.

        Returns:
            List of dicts (one per batch element), each containing:
                - "boxes": (K, 7) final 3D box parameters.
                - "scores": (K,) confidence scores.
                - "labels": (K,) predicted class indices (0-indexed).
        """
        batch_size = cls_preds.shape[0]
        results: List[Dict[str, Tensor]] = []

        for b in range(batch_size):
            cls_logits = cls_preds[b]  # (num_anchors, num_classes)
            reg_deltas = reg_preds[b]  # (num_anchors, 7)
            dir_logits = dir_preds[b] if dir_preds is not None else None

            # Get per-sample anchors
            if anchors.dim() == 3:
                sample_anchors = anchors[b]  # (num_anchors, 7)
            else:
                sample_anchors = anchors  # (num_anchors, 7)

            # Decode boxes
            decoded_boxes = self.decode_boxes(reg_deltas, sample_anchors)

            # Apply direction classifier to correct heading
            if dir_logits is not None:
                dir_labels = torch.argmax(dir_logits, dim=-1)  # (num_anchors,)
                # Wrap theta to [0, pi) first
                period = math.pi
                theta = decoded_boxes[:, 6]
                theta = _limit_period(theta, offset=self.direction_offset / period, period=period)
                # Flip heading for those classified as "backward"
                theta = theta + period * dir_labels.float()
                # Re-wrap
                theta = _limit_period(theta, offset=0.5, period=2 * math.pi)
                decoded_boxes[:, 6] = theta

            # Multi-class score computation
            cls_scores = torch.sigmoid(cls_logits)  # (num_anchors, num_classes)

            # Process per-class and aggregate
            all_boxes: List[Tensor] = []
            all_scores: List[Tensor] = []
            all_labels: List[Tensor] = []

            for cls_idx in range(self.num_classes):
                scores_cls = cls_scores[:, cls_idx]  # (num_anchors,)

                # Score thresholding
                score_mask = scores_cls > self.nms_score_threshold
                if score_mask.sum() == 0:
                    continue

                filtered_scores = scores_cls[score_mask]
                filtered_boxes = decoded_boxes[score_mask]

                # Top-k pre-NMS
                if filtered_scores.shape[0] > self.nms_pre_max_size:
                    topk_scores, topk_inds = torch.topk(
                        filtered_scores, self.nms_pre_max_size
                    )
                    filtered_scores = topk_scores
                    filtered_boxes = filtered_boxes[topk_inds]

                # Compute axis-aligned BEV boxes for NMS
                bev_boxes = _bev_box_to_axis_aligned(filtered_boxes)

                # Run NMS
                if torchvision_nms is not None:
                    keep = torchvision_nms(bev_boxes, filtered_scores, self.nms_iou_threshold)
                else:
                    # Fallback: simple greedy NMS implementation
                    keep = self._greedy_nms(bev_boxes, filtered_scores, self.nms_iou_threshold)

                # Post-NMS top-k
                if keep.numel() > self.nms_post_max_size:
                    keep = keep[: self.nms_post_max_size]

                all_boxes.append(filtered_boxes[keep])
                all_scores.append(filtered_scores[keep])
                all_labels.append(
                    torch.full(
                        (keep.numel(),),
                        cls_idx,
                        dtype=torch.long,
                        device=cls_preds.device,
                    )
                )

            # Concatenate all classes
            if len(all_boxes) > 0:
                final_boxes = torch.cat(all_boxes, dim=0)
                final_scores = torch.cat(all_scores, dim=0)
                final_labels = torch.cat(all_labels, dim=0)
            else:
                final_boxes = torch.zeros((0, self.box_code_size), device=cls_preds.device)
                final_scores = torch.zeros((0,), device=cls_preds.device)
                final_labels = torch.zeros((0,), dtype=torch.long, device=cls_preds.device)

            # Global post-max size limit (across all classes)
            if final_scores.numel() > self.nms_post_max_size:
                topk_scores, topk_inds = torch.topk(
                    final_scores, self.nms_post_max_size
                )
                final_boxes = final_boxes[topk_inds]
                final_scores = topk_scores
                final_labels = final_labels[topk_inds]

            results.append(
                {"boxes": final_boxes, "scores": final_scores, "labels": final_labels}
            )

        return results

    @staticmethod
    def _greedy_nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
        """Simple greedy NMS fallback when torchvision is not available.

        Args:
            boxes: (N, 4) axis-aligned boxes as (x1, y1, x2, y2).
            scores: (N,) confidence scores.
            iou_threshold: Suppression IoU threshold.

        Returns:
            Indices of kept boxes as a 1-D LongTensor.
        """
        order = torch.argsort(scores, descending=True)
        keep: List[int] = []

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        suppressed = torch.zeros(order.numel(), dtype=torch.bool, device=boxes.device)

        for idx in range(order.numel()):
            i = order[idx].item()
            if suppressed[idx]:
                continue
            keep.append(i)

            # Compute IoU of remaining boxes with the selected box
            remaining = order[idx + 1:]
            if remaining.numel() == 0:
                break

            xx1 = torch.clamp(boxes[remaining, 0], min=boxes[i, 0].item())
            yy1 = torch.clamp(boxes[remaining, 1], min=boxes[i, 1].item())
            xx2 = torch.clamp(boxes[remaining, 2], max=boxes[i, 2].item())
            yy2 = torch.clamp(boxes[remaining, 3], max=boxes[i, 3].item())

            inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
            union = areas[i] + areas[remaining] - inter
            iou = inter / torch.clamp(union, min=1e-6)

            suppress_mask = iou > iou_threshold
            suppressed[idx + 1:][suppress_mask] = True

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    # ------------------------------------------------------------------
    # Target assignment
    # ------------------------------------------------------------------

    @torch.no_grad()
    def assign_targets(
        self,
        anchors: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        pos_iou_thr: float = 0.6,
        neg_iou_thr: float = 0.45,
    ) -> Dict[str, Tensor]:
        """Assign ground-truth boxes to anchors based on BEV IoU.

        Uses a max-IoU matching strategy:
        - Anchors with IoU >= pos_iou_thr with any GT are positive.
        - Anchors with IoU < neg_iou_thr with all GTs are negative.
        - Anchors in between are ignored (neither positive nor negative).
        - Each GT is additionally assigned to its highest-IoU anchor to avoid
          unmatched ground-truths.

        Args:
            anchors: (A, 7) all anchor boxes (x, y, z, w, l, h, theta).
            gt_boxes: (G, 7) ground-truth boxes.
            gt_labels: (G,) class labels for each GT box (0-indexed).
            pos_iou_thr: IoU threshold above which an anchor is positive.
            neg_iou_thr: IoU threshold below which an anchor is negative.

        Returns:
            Dictionary containing:
                - "cls_targets": (A,) integer class targets. 0 = background,
                    1..num_classes = foreground classes. -1 = ignore.
                - "reg_targets": (A, 7) regression targets (only meaningful for
                    positives).
                - "reg_weights": (A,) per-anchor regression weights (1.0 for
                    positives, 0.0 otherwise).
                - "dir_targets": (A,) binary direction targets (0 or 1) for positives.
        """
        device = anchors.device
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        # Initialize targets
        cls_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)
        reg_targets = torch.zeros(num_anchors, self.box_code_size, device=device)
        reg_weights = torch.zeros(num_anchors, device=device)
        dir_targets = torch.zeros(num_anchors, dtype=torch.long, device=device)

        if num_gt == 0:
            # All anchors are negative (background)
            return {
                "cls_targets": cls_targets,
                "reg_targets": reg_targets,
                "reg_weights": reg_weights,
                "dir_targets": dir_targets,
            }

        # Compute BEV IoU between anchors and GT boxes using axis-aligned approximation
        anchor_bev = _bev_box_to_axis_aligned(anchors)  # (A, 4)
        gt_bev = _bev_box_to_axis_aligned(gt_boxes)  # (G, 4)
        iou_matrix = _iou_2d_axis_aligned(anchor_bev, gt_bev)  # (A, G)

        # For each anchor, find the best matching GT
        max_iou_per_anchor, matched_gt_indices = iou_matrix.max(dim=1)  # (A,), (A,)

        # For each GT, find the best matching anchor (to ensure every GT is matched)
        max_iou_per_gt, best_anchor_per_gt = iou_matrix.max(dim=0)  # (G,), (G,)

        # Assign positive / negative / ignore
        # Start by marking ignore for everything in the grey zone
        cls_targets[:] = -1  # ignore by default

        # Negatives: max IoU < neg_iou_thr
        neg_mask = max_iou_per_anchor < neg_iou_thr
        cls_targets[neg_mask] = 0

        # Positives: max IoU >= pos_iou_thr
        pos_mask = max_iou_per_anchor >= pos_iou_thr
        cls_targets[pos_mask] = gt_labels[matched_gt_indices[pos_mask]] + 1  # +1 for bg=0

        # Force-assign best anchor for each GT (avoid unmatched GTs)
        for gt_idx in range(num_gt):
            best_anchor_idx = best_anchor_per_gt[gt_idx]
            cls_targets[best_anchor_idx] = gt_labels[gt_idx] + 1
            matched_gt_indices[best_anchor_idx] = gt_idx
            pos_mask[best_anchor_idx] = True

        # Encode regression targets for all positive anchors
        positive_indices = torch.where(pos_mask)[0]
        if positive_indices.numel() > 0:
            matched_gt_for_pos = gt_boxes[matched_gt_indices[positive_indices]]
            pos_anchors = anchors[positive_indices]
            encoded = self.encode_boxes(matched_gt_for_pos, pos_anchors)
            reg_targets[positive_indices] = encoded
            reg_weights[positive_indices] = 1.0

            # Direction targets: based on the GT heading angle
            gt_theta = matched_gt_for_pos[:, 6]
            # Direction label: 0 if angle in [0, pi), 1 if in [pi, 2*pi)
            # after subtracting the direction offset
            dir_theta = gt_theta - self.direction_offset
            dir_theta = _limit_period(dir_theta, offset=0.0, period=2 * math.pi)
            dir_targets[positive_indices] = (dir_theta > math.pi).long()

        return {
            "cls_targets": cls_targets,
            "reg_targets": reg_targets,
            "reg_weights": reg_weights,
            "dir_targets": dir_targets,
        }

    # ------------------------------------------------------------------
    # Convenience: reshape predictions for loss / post-processing
    # ------------------------------------------------------------------

    def reshape_preds(
        self,
        cls_preds: Tensor,
        reg_preds: Tensor,
        dir_preds: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Reshape raw conv outputs to (B, num_anchors, C) format.

        Args:
            cls_preds: (B, A*num_classes, H, W) raw classification output.
            reg_preds: (B, A*box_code_size, H, W) raw regression output.
            dir_preds: (B, A*2, H, W) raw direction output or None.

        Returns:
            cls_preds: (B, H*W*A, num_classes)
            reg_preds: (B, H*W*A, box_code_size)
            dir_preds: (B, H*W*A, 2) or None
        """
        batch_size = cls_preds.shape[0]
        H, W = cls_preds.shape[2], cls_preds.shape[3]
        A = self.num_anchors_per_location

        # cls: (B, A*C_cls, H, W) -> (B, H, W, A, C_cls) -> (B, H*W*A, C_cls)
        cls_preds = cls_preds.view(batch_size, A, self.num_classes, H, W)
        cls_preds = cls_preds.permute(0, 3, 4, 1, 2).contiguous()
        cls_preds = cls_preds.view(batch_size, H * W * A, self.num_classes)

        # reg: (B, A*7, H, W) -> (B, H, W, A, 7) -> (B, H*W*A, 7)
        reg_preds = reg_preds.view(batch_size, A, self.box_code_size, H, W)
        reg_preds = reg_preds.permute(0, 3, 4, 1, 2).contiguous()
        reg_preds = reg_preds.view(batch_size, H * W * A, self.box_code_size)

        # dir: (B, A*2, H, W) -> (B, H, W, A, 2) -> (B, H*W*A, 2)
        if dir_preds is not None:
            dir_preds = dir_preds.view(batch_size, A, 2, H, W)
            dir_preds = dir_preds.permute(0, 3, 4, 1, 2).contiguous()
            dir_preds = dir_preds.view(batch_size, H * W * A, 2)

        return cls_preds, reg_preds, dir_preds
