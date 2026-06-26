"""
CRAFT 3D Detection Head.

Anchor-free, CenterPoint-style detection head for predicting 3D bounding boxes
from fused bird's-eye-view (BEV) feature maps. Designed for the CRAFT radar-camera
fusion model targeting nuScenes-style multi-class detection.

Architecture:
    Fused BEV features [B, 256, H, W]
        -> Shared convolution backbone (2-3 layers)
        -> Separate prediction heads:
            - Heatmap: [B, num_classes, H, W] (focal-loss-ready)
            - Regression: [B, 8, H, W] (offset, height, size, rotation)
            - Velocity: [B, 2, H, W] (vx, vy in m/s)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + ReLU helper block.

    A standard convolution building block used throughout the detection head.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        kernel_size: Spatial size of the convolution kernel.
        stride: Convolution stride.
        padding: Zero-padding added to both sides of the input.
            If None, padding is set to kernel_size // 2 for 'same' output size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: conv -> batchnorm -> relu."""
        return self.relu(self.bn(self.conv(x)))


class SeparateHead(nn.Module):
    """A separate prediction branch for one task (heatmap, regression, or velocity).

    Consists of a stack of N convolutional layers followed by a final 1x1 convolution
    that projects to the desired number of output channels.

    Args:
        in_channels: Number of input channels from the shared backbone.
        hidden_channels: Number of channels in the intermediate conv layers.
        out_channels: Number of output prediction channels.
        num_conv_layers: Number of intermediate 3x3 conv layers before the final 1x1.
        use_bias_init: If True, initialize the final layer bias to -2.19 (for focal loss
            heatmap heads). This corresponds to an initial foreground probability of
            approximately 0.01.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_conv_layers: int = 2,
        use_bias_init: bool = False,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        for i in range(num_conv_layers):
            ch_in = in_channels if i == 0 else hidden_channels
            layers.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.conv_layers = nn.Sequential(*layers)

        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)

        self._init_weights(use_bias_init)

    def _init_weights(self, use_bias_init: bool) -> None:
        """Initialize weights using Kaiming normal and optionally set bias for focal loss."""
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        nn.init.kaiming_normal_(self.final_conv.weight, mode="fan_out", nonlinearity="relu")
        if use_bias_init:
            # Initialize bias to -2.19 so that initial sigmoid output ~ 0.01
            # This is the standard initialization for focal loss heatmap heads.
            nn.init.constant_(self.final_conv.bias, -2.19)
        else:
            nn.init.constant_(self.final_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through intermediate conv layers and final 1x1 projection."""
        x = self.conv_layers(x)
        x = self.final_conv(x)
        return x


class CRAFTDetectionHead(nn.Module):
    """Anchor-free CenterPoint-style 3D detection head for the CRAFT model.

    Processes fused BEV features and produces per-pixel predictions for object
    centers (heatmap), bounding box parameters (regression), and velocity.

    Args:
        in_channels: Number of input channels from the fused BEV feature map.
            Default is 256 to match standard CRAFT backbone output.
        shared_channels: Number of channels in the shared conv backbone.
        num_classes: Number of object classes (10 for nuScenes: car, truck,
            construction_vehicle, bus, trailer, barrier, motorcycle, bicycle,
            pedestrian, traffic_cone).
        head_hidden_channels: Number of hidden channels in each separate head.
        num_head_conv_layers: Number of intermediate conv layers in each separate head.
        num_shared_conv_layers: Number of shared backbone conv layers.
    """

    def __init__(
        self,
        in_channels: int = 256,
        shared_channels: int = 256,
        num_classes: int = 10,
        head_hidden_channels: int = 64,
        num_head_conv_layers: int = 2,
        num_shared_conv_layers: int = 3,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Shared backbone: processes fused BEV features
        shared_layers: List[nn.Module] = []
        for i in range(num_shared_conv_layers):
            ch_in = in_channels if i == 0 else shared_channels
            shared_layers.append(ConvBlock(ch_in, shared_channels, kernel_size=3))
        self.shared_backbone = nn.Sequential(*shared_layers)

        # Heatmap head: [B, num_classes, H, W] - center probability per class
        self.heatmap_head = SeparateHead(
            in_channels=shared_channels,
            hidden_channels=head_hidden_channels,
            out_channels=num_classes,
            num_conv_layers=num_head_conv_layers,
            use_bias_init=True,  # Focal loss initialization
        )

        # Regression head: [B, 8, H, W]
        # Channels: offset_x(1), offset_y(1), height_z(1), log_w(1), log_l(1), log_h(1), sin_yaw(1), cos_yaw(1)
        self.regression_head = SeparateHead(
            in_channels=shared_channels,
            hidden_channels=head_hidden_channels,
            out_channels=8,
            num_conv_layers=num_head_conv_layers,
            use_bias_init=False,
        )

        # Velocity head: [B, 2, H, W] - vx, vy in m/s
        self.velocity_head = SeparateHead(
            in_channels=shared_channels,
            hidden_channels=head_hidden_channels,
            out_channels=2,
            num_conv_layers=num_head_conv_layers,
            use_bias_init=False,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through shared backbone and separate prediction heads.

        Args:
            x: Fused BEV feature map of shape [B, in_channels, H, W].

        Returns:
            Dictionary with keys:
                'heatmap': [B, num_classes, H, W] after sigmoid activation
                'regression': [B, 8, H, W] raw regression predictions
                'velocity': [B, 2, H, W] velocity predictions in m/s
        """
        shared_features = self.shared_backbone(x)

        heatmap = torch.sigmoid(self.heatmap_head(shared_features))
        regression = self.regression_head(shared_features)
        velocity = self.velocity_head(shared_features)

        return {
            "heatmap": heatmap,
            "regression": regression,
            "velocity": velocity,
        }


def _gather_feature(feat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather feature values at specified spatial indices.

    Args:
        feat: Feature tensor of shape [B, C, H, W].
        indices: Linear indices into the H*W spatial grid, shape [B, K].

    Returns:
        Gathered features of shape [B, K, C].
    """
    batch_size, channels, height, width = feat.shape
    num_indices = indices.shape[1]

    # Reshape feature to [B, C, H*W] then transpose to [B, H*W, C]
    feat_flat = feat.view(batch_size, channels, height * width).permute(0, 2, 1)

    # Expand indices to [B, K, C] for gathering
    indices_expanded = indices.unsqueeze(2).expand(-1, -1, channels)

    # Gather: [B, K, C]
    gathered = feat_flat.gather(1, indices_expanded)
    return gathered


def decode_predictions(
    head_outputs: Dict[str, torch.Tensor],
    voxel_size: float = 0.2,
    x_min: float = -51.2,
    y_min: float = -51.2,
    score_threshold: float = 0.1,
    top_k: int = 500,
    nms_kernel_size: int = 3,
) -> List[Dict[str, torch.Tensor]]:
    """Decode raw detection head outputs into 3D bounding boxes.

    Performs peak detection via max-pooling NMS on the heatmap, selects top-K
    detections, and converts regression values to absolute 3D bounding box
    coordinates using the BEV grid parameters.

    Args:
        head_outputs: Dictionary from CRAFTDetectionHead.forward() with keys
            'heatmap', 'regression', 'velocity'.
        voxel_size: Size of each BEV grid cell in meters.
        x_min: Minimum x coordinate of the BEV grid in meters.
        y_min: Minimum y coordinate of the BEV grid in meters.
        score_threshold: Minimum confidence score to keep a detection.
        top_k: Maximum number of detections to keep per sample.
        nms_kernel_size: Kernel size for max-pooling-based NMS on the heatmap.

    Returns:
        List of detection dictionaries (one per batch sample), each containing:
            'boxes_3d': [N, 7] tensor (x, y, z, w, l, h, yaw) in world coordinates
            'scores': [N] confidence scores
            'labels': [N] class indices (0-indexed)
            'velocity': [N, 2] velocity (vx, vy) in m/s
    """
    heatmap = head_outputs["heatmap"]       # [B, C, H, W]
    regression = head_outputs["regression"]  # [B, 8, H, W]
    velocity = head_outputs["velocity"]      # [B, 2, H, W]

    batch_size, num_classes, height, width = heatmap.shape

    # Max-pooling NMS: suppress non-peak locations
    padding = nms_kernel_size // 2
    heatmap_pool = F.max_pool2d(heatmap, kernel_size=nms_kernel_size, stride=1, padding=padding)
    # Keep only locations that are local maxima
    heatmap_nms = heatmap * (heatmap_pool == heatmap).float()

    results: List[Dict[str, torch.Tensor]] = []

    for b in range(batch_size):
        # Flatten heatmap across classes and spatial dims: [C * H * W]
        heatmap_flat = heatmap_nms[b].view(num_classes, -1)  # [C, H*W]

        # Find top-K across all classes
        # First flatten to [C * H * W]
        heatmap_all = heatmap_flat.view(-1)  # [C * H * W]
        num_candidates = min(top_k, heatmap_all.shape[0])
        topk_scores, topk_inds = torch.topk(heatmap_all, num_candidates)

        # Filter by score threshold
        valid_mask = topk_scores >= score_threshold
        topk_scores = topk_scores[valid_mask]
        topk_inds = topk_inds[valid_mask]

        if topk_scores.numel() == 0:
            results.append({
                "boxes_3d": torch.zeros((0, 7), device=heatmap.device, dtype=heatmap.dtype),
                "scores": torch.zeros((0,), device=heatmap.device, dtype=heatmap.dtype),
                "labels": torch.zeros((0,), device=heatmap.device, dtype=torch.long),
                "velocity": torch.zeros((0, 2), device=heatmap.device, dtype=heatmap.dtype),
            })
            continue

        # Decode indices into class, row, col
        topk_classes = topk_inds // (height * width)  # class index
        spatial_inds = topk_inds % (height * width)   # linear spatial index
        topk_rows = spatial_inds // width              # row (y grid index)
        topk_cols = spatial_inds % width               # col (x grid index)

        # Gather regression and velocity at peak locations
        # regression[b]: [8, H, W] -> flatten spatial -> [8, H*W]
        reg_flat = regression[b].view(8, -1)  # [8, H*W]
        reg_vals = reg_flat[:, spatial_inds].T  # [N, 8]

        vel_flat = velocity[b].view(2, -1)  # [2, H*W]
        vel_vals = vel_flat[:, spatial_inds].T  # [N, 2]

        # Decode regression values
        offset_x = reg_vals[:, 0]   # sub-voxel x offset
        offset_y = reg_vals[:, 1]   # sub-voxel y offset
        height_z = reg_vals[:, 2]   # z center
        log_w = reg_vals[:, 3]      # log(width)
        log_l = reg_vals[:, 4]      # log(length)
        log_h = reg_vals[:, 5]      # log(height)
        sin_yaw = reg_vals[:, 6]    # sin(yaw)
        cos_yaw = reg_vals[:, 7]    # cos(yaw)

        # Convert to absolute coordinates
        x = (topk_cols.float() + offset_x) * voxel_size + x_min
        y = (topk_rows.float() + offset_y) * voxel_size + y_min
        z = height_z
        w = torch.exp(log_w)
        l = torch.exp(log_l)
        h = torch.exp(log_h)
        yaw = torch.atan2(sin_yaw, cos_yaw)

        # Stack into boxes: [N, 7] -> (x, y, z, w, l, h, yaw)
        boxes_3d = torch.stack([x, y, z, w, l, h, yaw], dim=1)

        results.append({
            "boxes_3d": boxes_3d,
            "scores": topk_scores,
            "labels": topk_classes,
            "velocity": vel_vals,
        })

    return results


def _boxes_to_bev_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Convert 3D boxes to BEV corner representation for IoU computation.

    Args:
        boxes: [N, 7] tensor (x, y, z, w, l, h, yaw).

    Returns:
        BEV corners of shape [N, 4, 2] representing the four corners
        of each box projected onto the ground plane.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 3]  # width (along x in local frame)
    l = boxes[:, 4]  # length (along y in local frame)
    yaw = boxes[:, 6]

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # Half dimensions
    hw = w / 2.0
    hl = l / 2.0

    # Corner offsets in local frame (4 corners)
    # Order: front-left, front-right, rear-right, rear-left
    dx = torch.stack([hw, hw, -hw, -hw], dim=1)   # [N, 4]
    dy = torch.stack([hl, -hl, -hl, hl], dim=1)   # [N, 4]

    # Rotate corners
    corners_x = cos_yaw.unsqueeze(1) * dx - sin_yaw.unsqueeze(1) * dy + x.unsqueeze(1)
    corners_y = sin_yaw.unsqueeze(1) * dx + cos_yaw.unsqueeze(1) * dy + y.unsqueeze(1)

    corners = torch.stack([corners_x, corners_y], dim=2)  # [N, 4, 2]
    return corners


def _polygon_area(corners: torch.Tensor) -> torch.Tensor:
    """Compute area of convex polygons using the shoelace formula.

    Args:
        corners: [N, M, 2] tensor of polygon vertices (ordered).

    Returns:
        Areas of shape [N].
    """
    n_vertices = corners.shape[1]
    area = torch.zeros(corners.shape[0], device=corners.device, dtype=corners.dtype)
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        area += corners[:, i, 0] * corners[:, j, 1]
        area -= corners[:, j, 0] * corners[:, i, 1]
    return torch.abs(area) / 2.0


def _compute_bev_iou_axis_aligned(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute approximate BEV IoU using axis-aligned bounding box approximation.

    For rotated boxes, this computes the IoU of their axis-aligned bounding
    envelopes in BEV. This is a fast approximation suitable for NMS where
    exact rotated IoU is not strictly necessary.

    Args:
        boxes_a: [N, 7] tensor (x, y, z, w, l, h, yaw).
        boxes_b: [M, 7] tensor (x, y, z, w, l, h, yaw).

    Returns:
        IoU matrix of shape [N, M].
    """
    # Get BEV corners for axis-aligned envelope
    corners_a = _boxes_to_bev_corners(boxes_a)  # [N, 4, 2]
    corners_b = _boxes_to_bev_corners(boxes_b)  # [M, 4, 2]

    # Compute axis-aligned bounding envelope
    min_a = corners_a.min(dim=1).values  # [N, 2]
    max_a = corners_a.max(dim=1).values  # [N, 2]
    min_b = corners_b.min(dim=1).values  # [M, 2]
    max_b = corners_b.max(dim=1).values  # [M, 2]

    # Compute areas of axis-aligned envelopes
    area_a = (max_a[:, 0] - min_a[:, 0]) * (max_a[:, 1] - min_a[:, 1])  # [N]
    area_b = (max_b[:, 0] - min_b[:, 0]) * (max_b[:, 1] - min_b[:, 1])  # [M]

    # Compute intersection
    # min_a: [N, 2], min_b: [M, 2] -> need pairwise comparison
    inter_min_x = torch.max(min_a[:, 0:1], min_b[:, 0:1].T)  # [N, M]
    inter_min_y = torch.max(min_a[:, 1:2], min_b[:, 1:2].T)  # [N, M]
    inter_max_x = torch.min(max_a[:, 0:1], max_b[:, 0:1].T)  # [N, M]
    inter_max_y = torch.min(max_a[:, 1:2], max_b[:, 1:2].T)  # [N, M]

    inter_w = (inter_max_x - inter_min_x).clamp(min=0)
    inter_h = (inter_max_y - inter_min_y).clamp(min=0)
    inter_area = inter_w * inter_h  # [N, M]

    # Union
    union_area = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter_area  # [N, M]

    iou = inter_area / union_area.clamp(min=1e-6)
    return iou


def nms_bev(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.3,
) -> torch.Tensor:
    """Bird's Eye View Non-Maximum Suppression.

    Performs standard greedy NMS using BEV IoU (ignoring height dimension).
    Boxes are sorted by score in descending order, and overlapping boxes
    with IoU above the threshold are suppressed.

    Args:
        boxes: [N, 7] tensor of 3D bounding boxes (x, y, z, w, l, h, yaw).
        scores: [N] tensor of confidence scores.
        iou_threshold: IoU threshold for suppression. Boxes with BEV IoU
            above this value relative to a higher-scoring box are removed.

    Returns:
        Tensor of kept indices (sorted by descending score).
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    num_boxes = boxes.shape[0]

    # Sort by score descending
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_boxes = boxes[sorted_indices]

    # Compute pairwise BEV IoU for all sorted boxes
    iou_matrix = _compute_bev_iou_axis_aligned(sorted_boxes, sorted_boxes)  # [N, N]

    # Greedy NMS
    keep_mask = torch.ones(num_boxes, dtype=torch.bool, device=boxes.device)

    for i in range(num_boxes):
        if not keep_mask[i]:
            continue
        # Suppress all lower-scoring boxes that overlap with box i
        overlap = iou_matrix[i, i + 1:]
        suppress_mask = overlap > iou_threshold
        keep_mask[i + 1:][suppress_mask] = False

    # Return original indices of kept boxes
    kept_indices = sorted_indices[keep_mask]
    return kept_indices


def decode_and_nms(
    head_outputs: Dict[str, torch.Tensor],
    voxel_size: float = 0.2,
    x_min: float = -51.2,
    y_min: float = -51.2,
    score_threshold: float = 0.1,
    top_k: int = 500,
    nms_iou_threshold: float = 0.3,
    nms_kernel_size: int = 3,
) -> List[Dict[str, torch.Tensor]]:
    """Full detection pipeline: decode predictions then apply BEV NMS.

    Combines decode_predictions and nms_bev into a single convenience function.

    Args:
        head_outputs: Dictionary from CRAFTDetectionHead.forward().
        voxel_size: BEV grid cell size in meters.
        x_min: Minimum x coordinate of BEV grid.
        y_min: Minimum y coordinate of BEV grid.
        score_threshold: Minimum detection score.
        top_k: Maximum detections before NMS.
        nms_iou_threshold: IoU threshold for BEV NMS.
        nms_kernel_size: Kernel size for heatmap max-pooling NMS.

    Returns:
        List of detection dictionaries (one per batch sample) after NMS,
        each containing 'boxes_3d', 'scores', 'labels', 'velocity'.
    """
    raw_detections = decode_predictions(
        head_outputs=head_outputs,
        voxel_size=voxel_size,
        x_min=x_min,
        y_min=y_min,
        score_threshold=score_threshold,
        top_k=top_k,
        nms_kernel_size=nms_kernel_size,
    )

    nms_results: List[Dict[str, torch.Tensor]] = []

    for det in raw_detections:
        boxes = det["boxes_3d"]
        scores = det["scores"]
        labels = det["labels"]
        vel = det["velocity"]

        if boxes.shape[0] == 0:
            nms_results.append(det)
            continue

        # Apply NMS per class for better results
        all_kept_indices: List[torch.Tensor] = []
        unique_labels = labels.unique()

        for cls_id in unique_labels:
            cls_mask = labels == cls_id
            cls_indices = torch.where(cls_mask)[0]
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            kept_local = nms_bev(cls_boxes, cls_scores, iou_threshold=nms_iou_threshold)
            all_kept_indices.append(cls_indices[kept_local])

        if len(all_kept_indices) > 0:
            kept = torch.cat(all_kept_indices, dim=0)
            # Sort by score descending
            kept_scores = scores[kept]
            sort_order = torch.argsort(kept_scores, descending=True)
            kept = kept[sort_order]

            nms_results.append({
                "boxes_3d": boxes[kept],
                "scores": scores[kept],
                "labels": labels[kept],
                "velocity": vel[kept],
            })
        else:
            nms_results.append({
                "boxes_3d": torch.zeros((0, 7), device=boxes.device, dtype=boxes.dtype),
                "scores": torch.zeros((0,), device=boxes.device, dtype=boxes.dtype),
                "labels": torch.zeros((0,), device=boxes.device, dtype=torch.long),
                "velocity": torch.zeros((0, 2), device=boxes.device, dtype=boxes.dtype),
            })

    return nms_results
