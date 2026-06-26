"""
CenterPoint: Center-based 3D Object Detection from LiDAR Point Clouds.

Full model assembly including voxelization, 3D sparse backbone, BEV backbone,
center head with heatmap predictions, and optional two-stage refinement.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import spconv for sparse convolutions; fall back to dense if unavailable
try:
    import spconv.pytorch as spconv

    SPCONV_AVAILABLE = True
except ImportError:
    try:
        import spconv

        SPCONV_AVAILABLE = True
    except ImportError:
        SPCONV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------


def gaussian_focal_loss(
    pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0
) -> torch.Tensor:
    """Modified focal loss for heatmap training.

    Args:
        pred: Predicted heatmap (sigmoid applied), shape [B, C, H, W].
        target: Ground truth heatmap with Gaussian peaks, shape [B, C, H, W].
        alpha: Focusing parameter for positive samples.
        beta: Focusing parameter for negative samples.

    Returns:
        Scalar loss value.
    """
    pred = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)

    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
    neg_loss = (
        -((1 - target) ** beta) * (pred**alpha) * torch.log(1 - pred) * neg_mask
    )

    num_pos = pos_mask.sum().clamp(min=1.0)
    loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted L1 loss for regression targets.

    Args:
        pred: Predicted values, shape [N, D].
        target: Ground truth values, shape [N, D].
        weights: Per-sample weights, shape [N] or [N, 1].

    Returns:
        Scalar loss value.
    """
    loss = F.l1_loss(pred, target, reduction="none")
    if weights is not None:
        if weights.dim() == 1:
            weights = weights.unsqueeze(-1)
        loss = loss * weights
    num_valid = weights.sum().clamp(min=1.0) if weights is not None else max(pred.shape[0], 1)
    return loss.sum() / num_valid


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _nms_heatmap(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Apply max-pooling based NMS on heatmap to find local peaks."""
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_predictions(
    heatmap: torch.Tensor,
    regression: Dict[str, torch.Tensor],
    score_threshold: float = 0.1,
    max_detections: int = 500,
    point_cloud_range: Optional[List[float]] = None,
    voxel_size: Optional[List[float]] = None,
    feature_map_stride: int = 8,
) -> List[Dict[str, torch.Tensor]]:
    """Decode center head predictions into 3D bounding boxes.

    Args:
        heatmap: Predicted heatmap after sigmoid, shape [B, num_classes, H, W].
        regression: Dict of regression maps, each shape [B, C, H, W].
            Expected keys: 'offset_2d', 'height', 'dim_3d', 'rotation_sincos', 'velocity'.
        score_threshold: Minimum confidence to keep a detection.
        max_detections: Maximum number of detections per sample.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy, vz].
        feature_map_stride: Total stride from input to feature map.

    Returns:
        List of dicts (one per batch sample), each containing:
            'boxes': [N, 9] (x, y, z, w, l, h, yaw, vx, vy)
            'scores': [N]
            'labels': [N]
    """
    batch_size, num_classes, H, W = heatmap.shape
    device = heatmap.device

    # Default range and voxel size (nuScenes-like)
    if point_cloud_range is None:
        point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    if voxel_size is None:
        voxel_size = [0.075, 0.075, 0.2]

    # Apply NMS on heatmap
    heatmap_nms = _nms_heatmap(heatmap)

    results = []
    for b in range(batch_size):
        scores_all = []
        boxes_all = []
        labels_all = []

        for cls_id in range(num_classes):
            cls_heatmap = heatmap_nms[b, cls_id]  # [H, W]
            # Find top-k peaks
            flat = cls_heatmap.view(-1)
            num_candidates = min(max_detections, flat.shape[0])
            topk_scores, topk_inds = torch.topk(flat, num_candidates)

            # Filter by threshold
            mask = topk_scores > score_threshold
            topk_scores = topk_scores[mask]
            topk_inds = topk_inds[mask]

            if topk_scores.shape[0] == 0:
                continue

            # Convert flat indices to 2D
            topk_ys = (topk_inds // W).float()
            topk_xs = (topk_inds % W).float()

            # Gather regression values at peak locations
            def _gather(feat_map: torch.Tensor) -> torch.Tensor:
                """Gather feature values at topk locations. feat_map: [C, H, W]."""
                C = feat_map.shape[0]
                inds_expanded = topk_inds.unsqueeze(0).expand(C, -1)
                feat_flat = feat_map.view(C, -1)
                return feat_flat.gather(1, inds_expanded).t()  # [N, C]

            # Offset (sub-voxel refinement)
            offset_2d = _gather(regression["offset_2d"][b])  # [N, 2]
            height = _gather(regression["height"][b])  # [N, 1]
            dim_3d = _gather(regression["dim_3d"][b])  # [N, 3]
            rot_sincos = _gather(regression["rotation_sincos"][b])  # [N, 2]
            velocity = _gather(regression["velocity"][b])  # [N, 2]

            # Compute world coordinates
            xs = (topk_xs + offset_2d[:, 0]) * feature_map_stride * voxel_size[0] + point_cloud_range[0]
            ys = (topk_ys + offset_2d[:, 1]) * feature_map_stride * voxel_size[1] + point_cloud_range[1]
            zs = height[:, 0]

            # Dimensions (exp to ensure positive)
            dims = dim_3d.exp()

            # Yaw from sin/cos
            yaw = torch.atan2(rot_sincos[:, 0], rot_sincos[:, 1])

            # Compose boxes: [x, y, z, w, l, h, yaw, vx, vy]
            boxes = torch.stack(
                [xs, ys, zs, dims[:, 0], dims[:, 1], dims[:, 2], yaw, velocity[:, 0], velocity[:, 1]],
                dim=-1,
            )

            scores_all.append(topk_scores)
            boxes_all.append(boxes)
            labels_all.append(torch.full((topk_scores.shape[0],), cls_id, device=device, dtype=torch.long))

        if len(scores_all) > 0:
            scores_cat = torch.cat(scores_all, dim=0)
            boxes_cat = torch.cat(boxes_all, dim=0)
            labels_cat = torch.cat(labels_all, dim=0)

            # Keep top max_detections overall
            if scores_cat.shape[0] > max_detections:
                top_scores, top_inds = torch.topk(scores_cat, max_detections)
                scores_cat = top_scores
                boxes_cat = boxes_cat[top_inds]
                labels_cat = labels_cat[top_inds]

            results.append({"boxes": boxes_cat, "scores": scores_cat, "labels": labels_cat})
        else:
            results.append(
                {
                    "boxes": torch.zeros((0, 9), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), device=device, dtype=torch.long),
                }
            )

    return results


# ---------------------------------------------------------------------------
# VoxelEncoder (Mean VFE)
# ---------------------------------------------------------------------------


class VoxelEncoder(nn.Module):
    """Mean Voxel Feature Encoding.

    Averages all point features within each voxel to produce a single
    feature vector per occupied voxel.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        """
        Args:
            in_channels: Number of input point features (x, y, z, intensity).
            out_channels: Number of output features per voxel.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.linear = None

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_num_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            voxel_features: [num_voxels, max_points_per_voxel, in_channels]
            voxel_num_points: [num_voxels] number of valid points in each voxel.

        Returns:
            features: [num_voxels, out_channels]
        """
        # Create mask for valid points
        max_points = voxel_features.shape[1]
        mask = torch.arange(max_points, device=voxel_features.device).unsqueeze(0)
        mask = mask < voxel_num_points.unsqueeze(1)  # [num_voxels, max_points]
        mask = mask.unsqueeze(-1).float()  # [num_voxels, max_points, 1]

        # Mean over valid points
        points_sum = (voxel_features * mask).sum(dim=1)  # [num_voxels, in_channels]
        num_points_clamped = voxel_num_points.clamp(min=1).unsqueeze(-1).float()
        features = points_sum / num_points_clamped  # [num_voxels, in_channels]

        if self.linear is not None:
            features = self.linear(features)

        return features


# ---------------------------------------------------------------------------
# SparseBackbone3D
# ---------------------------------------------------------------------------


class _DenseConv3dBlock(nn.Module):
    """Dense 3D convolution block as fallback when spconv is not available."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SparseBackbone3D(nn.Module):
    """3D sparse convolutional backbone.

    Uses spconv SparseConv3d and SubMConv3d if available, otherwise
    falls back to dense 3D convolutions.

    Architecture:
        - Input conv (submanifold)
        - Series of stages, each with submanifold convs followed by
          a strided sparse conv for downsampling (stride 2 in spatial dims).
    """

    def __init__(
        self,
        in_channels: int = 4,
        channels: List[int] = None,
        layers: List[int] = None,
        sparse_shape: List[int] = None,
    ):
        """
        Args:
            in_channels: Input feature channels per voxel.
            channels: List of output channels for each stage.
            layers: Number of submanifold conv layers per stage.
            sparse_shape: [Z, Y, X] spatial shape of the sparse tensor.
        """
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128]
        if layers is None:
            layers = [2, 2, 2, 2]
        if sparse_shape is None:
            sparse_shape = [40, 1440, 1440]

        self.sparse_shape = sparse_shape
        self.use_sparse = SPCONV_AVAILABLE
        self.num_stages = len(channels)
        self.out_channels = channels[-1]

        if self.use_sparse:
            self._build_sparse(in_channels, channels, layers)
        else:
            self._build_dense(in_channels, channels, layers)

    def _build_sparse(self, in_channels: int, channels: List[int], layers: List[int]):
        """Build spconv-based sparse layers."""
        # Initial submanifold conv
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, channels[0], 3, padding=1, bias=False, indice_key="subm0"),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.stages = nn.ModuleList()
        prev_ch = channels[0]
        for stage_idx, (out_ch, num_layers) in enumerate(zip(channels, layers)):
            stage_blocks = []
            # Submanifold conv blocks
            for layer_idx in range(num_layers):
                ch_in = prev_ch if layer_idx == 0 else out_ch
                stage_blocks.append(
                    spconv.SubMConv3d(ch_in, out_ch, 3, padding=1, bias=False, indice_key=f"subm{stage_idx + 1}")
                )
                stage_blocks.append(nn.BatchNorm1d(out_ch))
                stage_blocks.append(nn.ReLU(inplace=True))

            # Strided sparse conv for downsampling (stride 2 in all dims)
            stage_blocks.append(
                spconv.SparseConv3d(out_ch, out_ch, 3, stride=2, padding=1, bias=False, indice_key=f"spconv{stage_idx}")
            )
            stage_blocks.append(nn.BatchNorm1d(out_ch))
            stage_blocks.append(nn.ReLU(inplace=True))

            self.stages.append(spconv.SparseSequential(*stage_blocks))
            prev_ch = out_ch

    def _build_dense(self, in_channels: int, channels: List[int], layers: List[int]):
        """Build dense 3D conv layers as fallback."""
        self.input_conv = _DenseConv3dBlock(in_channels, channels[0], kernel_size=3, stride=1)

        self.stages = nn.ModuleList()
        prev_ch = channels[0]
        for out_ch, num_layers in zip(channels, layers):
            stage_blocks = []
            for layer_idx in range(num_layers):
                ch_in = prev_ch if layer_idx == 0 else out_ch
                stage_blocks.append(_DenseConv3dBlock(ch_in, out_ch, kernel_size=3, stride=1))
            # Downsample
            stage_blocks.append(_DenseConv3dBlock(out_ch, out_ch, kernel_size=3, stride=2))
            self.stages.append(nn.Sequential(*stage_blocks))
            prev_ch = out_ch

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Args:
            voxel_features: [num_voxels, C] encoded voxel features.
            voxel_coords: [num_voxels, 4] (batch_idx, z, y, x).
            batch_size: Number of samples in the batch.

        Returns:
            Dense 4D tensor of shape [B, C, Y', X'] after collapsing Z.
        """
        if self.use_sparse:
            return self._forward_sparse(voxel_features, voxel_coords, batch_size)
        else:
            return self._forward_dense(voxel_features, voxel_coords, batch_size)

    def _forward_sparse(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        input_sp = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )
        x = self.input_conv(input_sp)
        for stage in self.stages:
            x = stage(x)

        # Convert to dense and collapse Z dimension
        dense = x.dense()  # [B, C, Z', Y', X']
        B, C, Z, Y, X = dense.shape
        # Reshape: collapse Z into channel dimension, then use a simple reshape
        bev = dense.reshape(B, C * Z, Y, X)
        return bev

    def _forward_dense(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        # Scatter voxel features into dense 3D grid
        Z, Y, X = self.sparse_shape
        C = voxel_features.shape[1]
        device = voxel_features.device

        dense_input = torch.zeros(batch_size, C, Z, Y, X, device=device)
        batch_idx = voxel_coords[:, 0].long()
        z_idx = voxel_coords[:, 1].long()
        y_idx = voxel_coords[:, 2].long()
        x_idx = voxel_coords[:, 3].long()
        dense_input[batch_idx, :, z_idx, y_idx, x_idx] = voxel_features

        x = self.input_conv(dense_input)
        for stage in self.stages:
            x = stage(x)

        # Collapse Z dimension
        B, C_out, Z_out, Y_out, X_out = x.shape
        bev = x.reshape(B, C_out * Z_out, Y_out, X_out)
        return bev


# ---------------------------------------------------------------------------
# BEVBackbone
# ---------------------------------------------------------------------------


class BEVBackbone(nn.Module):
    """Multi-scale 2D convolutional backbone operating on BEV features.

    Two stages of convolution blocks with stride-1 and stride-2 convolutions,
    followed by upsampling and concatenation to form a feature pyramid.
    """

    def __init__(
        self,
        in_channels: int = 128,
        layer_nums: List[int] = None,
        layer_strides: List[int] = None,
        num_filters: List[int] = None,
        upsample_strides: List[int] = None,
        num_upsample_filters: List[int] = None,
    ):
        """
        Args:
            in_channels: Number of input BEV channels.
            layer_nums: Number of conv layers per block.
            layer_strides: Stride of the first conv in each block.
            num_filters: Output channels for each block.
            upsample_strides: Upsample factor for each block.
            num_upsample_filters: Output channels after upsampling.
        """
        super().__init__()
        if layer_nums is None:
            layer_nums = [5, 5]
        if layer_strides is None:
            layer_strides = [1, 2]
        if num_filters is None:
            num_filters = [128, 256]
        if upsample_strides is None:
            upsample_strides = [1, 2]
        if num_upsample_filters is None:
            num_upsample_filters = [256, 256]

        assert len(layer_nums) == len(layer_strides) == len(num_filters)
        assert len(num_filters) == len(upsample_strides) == len(num_upsample_filters)

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        cur_channels = in_channels
        for idx in range(len(layer_nums)):
            block_layers = []
            # First conv with specified stride
            block_layers.append(
                nn.Conv2d(cur_channels, num_filters[idx], 3, stride=layer_strides[idx], padding=1, bias=False)
            )
            block_layers.append(nn.BatchNorm2d(num_filters[idx]))
            block_layers.append(nn.ReLU(inplace=True))

            # Remaining stride-1 convs
            for _ in range(layer_nums[idx] - 1):
                block_layers.append(
                    nn.Conv2d(num_filters[idx], num_filters[idx], 3, stride=1, padding=1, bias=False)
                )
                block_layers.append(nn.BatchNorm2d(num_filters[idx]))
                block_layers.append(nn.ReLU(inplace=True))

            self.blocks.append(nn.Sequential(*block_layers))
            cur_channels = num_filters[idx]

            # Upsample (deconv) block
            if upsample_strides[idx] >= 1:
                stride = upsample_strides[idx]
                deblock = nn.Sequential(
                    nn.ConvTranspose2d(
                        num_filters[idx],
                        num_upsample_filters[idx],
                        stride,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_upsample_filters[idx]),
                    nn.ReLU(inplace=True),
                )
            else:
                stride = int(round(1.0 / upsample_strides[idx]))
                deblock = nn.Sequential(
                    nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False),
                    nn.BatchNorm2d(num_upsample_filters[idx]),
                    nn.ReLU(inplace=True),
                )
            self.deblocks.append(deblock)

        self.out_channels = sum(num_upsample_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BEV feature map [B, C, H, W].

        Returns:
            Concatenated multi-scale features [B, out_channels, H', W'].
        """
        ups = []
        for i, (block, deblock) in enumerate(zip(self.blocks, self.deblocks)):
            x = block(x)
            ups.append(deblock(x))

        # Align spatial sizes to the largest
        target_h = max(u.shape[2] for u in ups)
        target_w = max(u.shape[3] for u in ups)
        aligned = []
        for u in ups:
            if u.shape[2] != target_h or u.shape[3] != target_w:
                u = F.interpolate(u, size=(target_h, target_w), mode="bilinear", align_corners=False)
            aligned.append(u)

        return torch.cat(aligned, dim=1)


# ---------------------------------------------------------------------------
# CenterHead
# ---------------------------------------------------------------------------


class CenterHead(nn.Module):
    """Center-based detection head.

    Predicts per-class heatmaps and shared regression branches for
    offset, height, 3D dimensions, rotation (sin/cos), and velocity.
    Each task group has its own heatmap head but shares regression heads.
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_classes: int = 10,
        tasks: Optional[List[Dict]] = None,
        common_heads: Optional[Dict[str, int]] = None,
        head_conv: int = 64,
    ):
        """
        Args:
            in_channels: Input feature channels from BEV backbone.
            num_classes: Total number of object classes.
            tasks: List of task dicts, each with 'num_class' and 'class_names'.
                   If None, a single task with all classes is used.
            common_heads: Dict mapping regression branch name to output channels.
            head_conv: Intermediate conv channels in head networks.
        """
        super().__init__()

        if tasks is None:
            tasks = [{"num_class": num_classes, "class_names": [f"class_{i}" for i in range(num_classes)]}]
        if common_heads is None:
            common_heads = {
                "offset_2d": 2,
                "height": 1,
                "dim_3d": 3,
                "rotation_sincos": 2,
                "velocity": 2,
            }

        self.tasks = tasks
        self.common_heads = common_heads
        self.num_classes = num_classes

        # Build task-specific heatmap heads
        self.heatmap_heads = nn.ModuleList()
        for task in tasks:
            num_cls = task["num_class"]
            head = nn.Sequential(
                nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=False),
                nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_cls, 1),
            )
            # Initialize heatmap bias for stable training
            head[-1].bias.data.fill_(-2.19)  # -log((1 - 0.1) / 0.1)
            self.heatmap_heads.append(head)

        # Build shared regression heads
        self.regression_heads = nn.ModuleDict()
        for name, out_ch in common_heads.items():
            self.regression_heads[name] = nn.Sequential(
                nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=False),
                nn.BatchNorm2d(head_conv),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, out_ch, 1),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: BEV features [B, C, H, W].

        Returns:
            Dict with:
                'heatmap': [B, total_classes, H, W] (after sigmoid)
                + one key per regression branch: [B, out_ch, H, W]
        """
        # Heatmap predictions (concatenate all tasks)
        heatmaps = []
        for hm_head in self.heatmap_heads:
            heatmaps.append(hm_head(x))
        heatmap = torch.cat(heatmaps, dim=1)

        # Regression predictions
        result = {"heatmap": torch.sigmoid(heatmap), "heatmap_raw": heatmap}
        for name, reg_head in self.regression_heads.items():
            result[name] = reg_head(x)

        return result


# ---------------------------------------------------------------------------
# TwoStageHead
# ---------------------------------------------------------------------------


class TwoStageHead(nn.Module):
    """Optional two-stage refinement head.

    Extracts features at predicted box locations via bilinear sampling,
    then refines box parameters with an MLP.
    """

    def __init__(self, in_channels: int = 512, hidden_channels: int = 256, num_refine_params: int = 9):
        """
        Args:
            in_channels: Input feature channels from BEV backbone.
            hidden_channels: Hidden layer size in the refinement MLP.
            num_refine_params: Number of box parameters to refine
                (x, y, z, w, l, h, yaw, vx, vy).
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_refine_params = num_refine_params

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_refine_params),
        )

        # Initialize last layer to zero so initial refinement is identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        bev_features: torch.Tensor,
        box_centers_2d: torch.Tensor,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            bev_features: [B, C, H, W] BEV feature map.
            box_centers_2d: [B, N, 2] normalized box center coordinates in [-1, 1].
            spatial_shape: (H, W) of the BEV feature map.

        Returns:
            refinements: [B, N, num_refine_params] residual box adjustments.
        """
        B, N, _ = box_centers_2d.shape

        # Create grid for bilinear sampling: [B, N, 1, 2]
        grid = box_centers_2d.unsqueeze(2)  # [B, N, 1, 2]

        # Sample features at box locations
        sampled = F.grid_sample(
            bev_features, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # [B, C, N, 1]
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, C]

        # Refine via MLP
        refinements = self.mlp(sampled)  # [B, N, num_refine_params]
        return refinements


# ---------------------------------------------------------------------------
# Voxelization Utility
# ---------------------------------------------------------------------------


class Voxelizer(nn.Module):
    """Point cloud voxelization.

    Assigns each point to a voxel based on spatial location, aggregates
    points per voxel up to a maximum count.
    """

    def __init__(
        self,
        voxel_size: List[float],
        point_cloud_range: List[float],
        max_num_points_per_voxel: int = 10,
        max_voxels: int = 60000,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.max_voxels = max_voxels

        # Compute grid size
        pc_range = np.array(point_cloud_range)
        vs = np.array(voxel_size)
        self.grid_size = np.round((pc_range[3:] - pc_range[:3]) / vs).astype(np.int64)

    @torch.no_grad()
    def forward(
        self, points_batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Args:
            points_batch: List of [N_i, F] tensors, one per sample in the batch.

        Returns:
            voxel_features: [total_voxels, max_points, F]
            voxel_coords: [total_voxels, 4] (batch_idx, z, y, x)
            voxel_num_points: [total_voxels]
            batch_size: int
        """
        batch_size = len(points_batch)
        device = points_batch[0].device
        num_features = points_batch[0].shape[1]

        all_voxel_features = []
        all_voxel_coords = []
        all_voxel_num_points = []

        pc_range = torch.tensor(self.point_cloud_range[:3], device=device, dtype=torch.float32)
        voxel_size = torch.tensor(self.voxel_size, device=device, dtype=torch.float32)
        grid_size_tensor = torch.tensor(
            [self.grid_size[2], self.grid_size[1], self.grid_size[0]], device=device, dtype=torch.long
        )  # X, Y, Z

        for batch_idx, points in enumerate(points_batch):
            # Filter points within range
            mask_x = (points[:, 0] >= self.point_cloud_range[0]) & (points[:, 0] < self.point_cloud_range[3])
            mask_y = (points[:, 1] >= self.point_cloud_range[1]) & (points[:, 1] < self.point_cloud_range[4])
            mask_z = (points[:, 2] >= self.point_cloud_range[2]) & (points[:, 2] < self.point_cloud_range[5])
            mask = mask_x & mask_y & mask_z
            points = points[mask]

            # Compute voxel indices
            coords = ((points[:, :3] - pc_range) / voxel_size).long()  # [N, 3] -> x, y, z indices
            # Clamp to valid range
            coords[:, 0] = coords[:, 0].clamp(0, self.grid_size[0] - 1)
            coords[:, 1] = coords[:, 1].clamp(0, self.grid_size[1] - 1)
            coords[:, 2] = coords[:, 2].clamp(0, self.grid_size[2] - 1)

            # Hash voxel coordinates for grouping
            voxel_hash = (
                coords[:, 2] * self.grid_size[1] * self.grid_size[0]
                + coords[:, 1] * self.grid_size[0]
                + coords[:, 0]
            )

            # Get unique voxels
            unique_hashes, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
            num_voxels = min(unique_hashes.shape[0], self.max_voxels)

            # Allocate output tensors
            voxel_features = torch.zeros(
                num_voxels, self.max_num_points_per_voxel, num_features, device=device
            )
            voxel_num_points = torch.zeros(num_voxels, device=device, dtype=torch.long)

            # Fill voxels (take first max_voxels unique voxels)
            for v_idx in range(num_voxels):
                point_mask = inverse_indices == v_idx
                pts_in_voxel = points[point_mask]
                num_pts = min(pts_in_voxel.shape[0], self.max_num_points_per_voxel)
                voxel_features[v_idx, :num_pts] = pts_in_voxel[:num_pts]
                voxel_num_points[v_idx] = num_pts

            # Recover voxel coordinates from hashes
            selected_hashes = unique_hashes[:num_voxels]
            z_coords = selected_hashes // (self.grid_size[1] * self.grid_size[0])
            remainder = selected_hashes % (self.grid_size[1] * self.grid_size[0])
            y_coords = remainder // self.grid_size[0]
            x_coords = remainder % self.grid_size[0]

            batch_col = torch.full((num_voxels,), batch_idx, device=device, dtype=torch.long)
            voxel_coords = torch.stack([batch_col, z_coords, y_coords, x_coords], dim=1)

            all_voxel_features.append(voxel_features)
            all_voxel_coords.append(voxel_coords)
            all_voxel_num_points.append(voxel_num_points)

        voxel_features_cat = torch.cat(all_voxel_features, dim=0)
        voxel_coords_cat = torch.cat(all_voxel_coords, dim=0)
        voxel_num_points_cat = torch.cat(all_voxel_num_points, dim=0)

        return voxel_features_cat, voxel_coords_cat, voxel_num_points_cat, batch_size


# ---------------------------------------------------------------------------
# CenterPoint Model
# ---------------------------------------------------------------------------


class CenterPoint(nn.Module):
    """CenterPoint: Center-based 3D Object Detection from LiDAR Point Clouds.

    End-to-end model comprising:
        1. Voxelization
        2. Voxel Feature Encoding (Mean VFE)
        3. 3D Sparse Backbone
        4. BEV Collapse (flatten Z into channels)
        5. 2D BEV Backbone
        6. Center Head (heatmap + regression)
        7. (Optional) Two-stage refinement
    """

    def __init__(
        self,
        voxel_size: List[float] = None,
        point_cloud_range: List[float] = None,
        max_num_points_per_voxel: int = 10,
        max_voxels: int = 60000,
        num_point_features: int = 4,
        backbone_3d_channels: List[int] = None,
        backbone_3d_layers: List[int] = None,
        bev_in_channels: Optional[int] = None,
        bev_layer_nums: List[int] = None,
        bev_layer_strides: List[int] = None,
        bev_num_filters: List[int] = None,
        bev_upsample_strides: List[int] = None,
        bev_upsample_filters: List[int] = None,
        num_classes: int = 10,
        tasks: Optional[List[Dict]] = None,
        common_heads: Optional[Dict[str, int]] = None,
        head_conv: int = 64,
        two_stage_enabled: bool = False,
        two_stage_hidden: int = 256,
        score_threshold: float = 0.1,
        max_detections: int = 500,
    ):
        super().__init__()

        # Defaults
        if voxel_size is None:
            voxel_size = [0.075, 0.075, 0.2]
        if point_cloud_range is None:
            point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        if backbone_3d_channels is None:
            backbone_3d_channels = [16, 32, 64, 128]
        if backbone_3d_layers is None:
            backbone_3d_layers = [2, 2, 2, 2]
        if bev_layer_nums is None:
            bev_layer_nums = [5, 5]
        if bev_layer_strides is None:
            bev_layer_strides = [1, 2]
        if bev_num_filters is None:
            bev_num_filters = [128, 256]
        if bev_upsample_strides is None:
            bev_upsample_strides = [1, 2]
        if bev_upsample_filters is None:
            bev_upsample_filters = [256, 256]

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.two_stage_enabled = two_stage_enabled

        # Compute sparse shape [Z, Y, X]
        pc_range = np.array(point_cloud_range)
        vs = np.array(voxel_size)
        grid_size = np.round((pc_range[3:] - pc_range[:3]) / vs).astype(np.int64)
        sparse_shape = [int(grid_size[2]), int(grid_size[1]), int(grid_size[0])]  # Z, Y, X

        # 1. Voxelizer
        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points_per_voxel=max_num_points_per_voxel,
            max_voxels=max_voxels,
        )

        # 2. Voxel Feature Encoder
        self.voxel_encoder = VoxelEncoder(
            in_channels=num_point_features, out_channels=num_point_features
        )

        # 3. 3D Sparse Backbone
        self.sparse_backbone = SparseBackbone3D(
            in_channels=num_point_features,
            channels=backbone_3d_channels,
            layers=backbone_3d_layers,
            sparse_shape=sparse_shape,
        )

        # Determine BEV input channels:
        # After all downsampling stages (each stage halves spatial), Z dimension shrinks
        # The final channels are backbone_3d_channels[-1] * (Z after all downsampling)
        num_stages = len(backbone_3d_channels)
        z_after_backbone = sparse_shape[0]
        for _ in range(num_stages):
            z_after_backbone = math.ceil(z_after_backbone / 2)
        bev_input_channels = backbone_3d_channels[-1] * z_after_backbone

        if bev_in_channels is not None:
            bev_input_channels = bev_in_channels

        # 4. BEV Backbone
        self.bev_backbone = BEVBackbone(
            in_channels=bev_input_channels,
            layer_nums=bev_layer_nums,
            layer_strides=bev_layer_strides,
            num_filters=bev_num_filters,
            upsample_strides=bev_upsample_strides,
            num_upsample_filters=bev_upsample_filters,
        )

        # 5. Center Head
        head_in_channels = self.bev_backbone.out_channels
        self.center_head = CenterHead(
            in_channels=head_in_channels,
            num_classes=num_classes,
            tasks=tasks,
            common_heads=common_heads,
            head_conv=head_conv,
        )

        # 6. Optional Two-Stage Head
        if two_stage_enabled:
            self.two_stage_head = TwoStageHead(
                in_channels=head_in_channels,
                hidden_channels=two_stage_hidden,
                num_refine_params=9,
            )
        else:
            self.two_stage_head = None

        # Compute feature map stride for decoding
        # Each 3D backbone stage does stride-2, BEV backbone first block stride
        self.feature_map_stride = 2 ** num_stages * bev_layer_strides[0]

    def forward(self, points_batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            points_batch: List of [N_i, F] point cloud tensors (one per sample).

        Returns:
            Dict containing:
                'heatmap': [B, num_classes, H, W]
                'offset_2d': [B, 2, H, W]
                'height': [B, 1, H, W]
                'dim_3d': [B, 3, H, W]
                'rotation_sincos': [B, 2, H, W]
                'velocity': [B, 2, H, W]
                'refinements': [B, N, 9] (if two_stage_enabled)
        """
        # Step 1: Voxelize
        voxel_features, voxel_coords, voxel_num_points, batch_size = self.voxelizer(points_batch)

        # Step 2: Encode voxel features
        encoded_features = self.voxel_encoder(voxel_features, voxel_num_points)

        # Step 3: 3D sparse backbone
        bev_features = self.sparse_backbone(encoded_features, voxel_coords, batch_size)

        # Step 4: 2D BEV backbone
        bev_features = self.bev_backbone(bev_features)

        # Step 5: Center head predictions
        predictions = self.center_head(bev_features)

        # Step 6: Optional two-stage refinement
        if self.two_stage_enabled and self.two_stage_head is not None:
            # Decode first-stage predictions to get box centers
            with torch.no_grad():
                decoded = decode_predictions(
                    predictions["heatmap"],
                    {k: predictions[k] for k in self.center_head.common_heads},
                    score_threshold=self.score_threshold,
                    max_detections=self.max_detections,
                    point_cloud_range=self.point_cloud_range,
                    voxel_size=self.voxel_size,
                    feature_map_stride=self.feature_map_stride,
                )

            # Normalize box centers to [-1, 1] for grid_sample
            B = batch_size
            H, W = bev_features.shape[2], bev_features.shape[3]
            refined_boxes = []

            for b in range(B):
                boxes = decoded[b]["boxes"]  # [N, 9]
                if boxes.shape[0] > 0:
                    # Convert world coords to normalized feature map coords
                    cx = (boxes[:, 0] - self.point_cloud_range[0]) / (
                        self.point_cloud_range[3] - self.point_cloud_range[0]
                    ) * 2 - 1
                    cy = (boxes[:, 1] - self.point_cloud_range[1]) / (
                        self.point_cloud_range[4] - self.point_cloud_range[1]
                    ) * 2 - 1
                    centers_2d = torch.stack([cx, cy], dim=-1).unsqueeze(0)  # [1, N, 2]

                    refinement = self.two_stage_head(
                        bev_features[b : b + 1], centers_2d, (H, W)
                    )  # [1, N, 9]
                    refined = boxes + refinement.squeeze(0)
                    refined_boxes.append(refined)
                else:
                    refined_boxes.append(boxes)

            predictions["refined_boxes"] = refined_boxes

        return predictions

    def forward_train(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Training forward pass that computes losses.

        Args:
            batch: Dict containing:
                'points': List of [N_i, F] point tensors.
                'heatmap_targets': [B, num_classes, H, W] Gaussian heatmap targets.
                'reg_targets': [B, num_reg, H, W] regression targets.
                'reg_mask': [B, H, W] binary mask of valid regression locations.
                'reg_indices': [B, max_objs] flat indices of target locations.
                'reg_values': [B, max_objs, D] regression target values.
                'num_objects': [B] number of valid objects per sample.

        Returns:
            Dict with 'heatmap_loss', 'reg_loss', 'total_loss'.
        """
        points_batch = batch["points"]
        predictions = self.forward(points_batch)

        # Heatmap loss
        heatmap_pred = predictions["heatmap"]
        heatmap_target = batch["heatmap_targets"].to(heatmap_pred.device)
        hm_loss = gaussian_focal_loss(heatmap_pred, heatmap_target)

        # Regression loss
        reg_mask = batch["reg_mask"].to(heatmap_pred.device)  # [B, H, W]
        reg_indices = batch["reg_indices"].to(heatmap_pred.device)  # [B, max_objs]
        reg_values = batch["reg_values"].to(heatmap_pred.device)  # [B, max_objs, D]
        num_objects = batch["num_objects"].to(heatmap_pred.device)  # [B]

        B = heatmap_pred.shape[0]
        total_reg_loss = torch.tensor(0.0, device=heatmap_pred.device)
        total_objects = 0

        # Gather regression predictions at target locations
        reg_branch_names = list(self.center_head.common_heads.keys())
        for b in range(B):
            n_obj = num_objects[b].item()
            if n_obj == 0:
                continue

            indices = reg_indices[b, :n_obj]  # [n_obj]
            targets = reg_values[b, :n_obj]  # [n_obj, D]

            # Concatenate all regression branch predictions
            reg_preds = []
            for name in reg_branch_names:
                feat = predictions[name][b]  # [C, H, W]
                C = feat.shape[0]
                feat_flat = feat.view(C, -1)  # [C, H*W]
                gathered = feat_flat[:, indices.long()].t()  # [n_obj, C]
                reg_preds.append(gathered)

            reg_pred_cat = torch.cat(reg_preds, dim=-1)  # [n_obj, total_reg_channels]

            # Truncate targets to match prediction channels if needed
            pred_ch = reg_pred_cat.shape[-1]
            target_ch = targets.shape[-1]
            if target_ch > pred_ch:
                targets = targets[:, :pred_ch]
            elif pred_ch > target_ch:
                reg_pred_cat = reg_pred_cat[:, :target_ch]

            total_reg_loss = total_reg_loss + F.l1_loss(reg_pred_cat, targets, reduction="sum")
            total_objects += n_obj

        reg_loss = total_reg_loss / max(total_objects, 1)

        # Total loss
        total_loss = hm_loss + reg_loss

        losses = {
            "heatmap_loss": hm_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
        }

        # Two-stage refinement loss (if applicable)
        if self.two_stage_enabled and "refined_boxes" in predictions and "box_targets" in batch:
            box_targets = batch["box_targets"]  # List of [N_i, 9]
            refine_loss = torch.tensor(0.0, device=heatmap_pred.device)
            n_refined = 0
            for b in range(B):
                if predictions["refined_boxes"][b].shape[0] > 0 and box_targets[b].shape[0] > 0:
                    # Simple L1 between refined and target (assumes matched)
                    n_match = min(predictions["refined_boxes"][b].shape[0], box_targets[b].shape[0])
                    refine_loss = refine_loss + F.l1_loss(
                        predictions["refined_boxes"][b][:n_match],
                        box_targets[b][:n_match].to(heatmap_pred.device),
                        reduction="sum",
                    )
                    n_refined += n_match
            refine_loss = refine_loss / max(n_refined, 1)
            losses["refine_loss"] = refine_loss
            losses["total_loss"] = total_loss + 0.5 * refine_loss

        return losses

    def forward_test(self, batch: Dict) -> List[Dict[str, torch.Tensor]]:
        """Inference forward pass that returns decoded predictions.

        Args:
            batch: Dict containing:
                'points': List of [N_i, F] point tensors.

        Returns:
            List of dicts (one per sample) with 'boxes', 'scores', 'labels'.
        """
        points_batch = batch["points"]
        predictions = self.forward(points_batch)

        # Decode predictions
        regression = {k: predictions[k] for k in self.center_head.common_heads}
        results = decode_predictions(
            predictions["heatmap"],
            regression,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            feature_map_stride=self.feature_map_stride,
        )

        # Apply two-stage refinement if available
        if self.two_stage_enabled and "refined_boxes" in predictions:
            for b in range(len(results)):
                if predictions["refined_boxes"][b].shape[0] > 0:
                    results[b]["boxes"] = predictions["refined_boxes"][b]

        return results


# ---------------------------------------------------------------------------
# Build from Config
# ---------------------------------------------------------------------------


def build_model_from_config(config) -> CenterPoint:
    """Build a CenterPoint model from a configuration object/dict.

    Expected config structure:
        config.model.voxel_size: List[float]
        config.model.point_cloud_range: List[float]
        config.model.backbone_3d.channels: List[int]
        config.model.backbone_3d.layers: List[int]
        config.model.bev_backbone.in_channels: int
        config.model.bev_backbone.layer_nums: List[int]
        config.model.bev_backbone.layer_strides: List[int]
        config.model.head.num_classes: int
        config.model.head.tasks: List[Dict]
        config.model.head.common_heads: Dict[str, int]
        config.model.two_stage.enabled: bool

    The config can be either a nested object with attribute access or a nested dict.
    """

    def _get(obj, key, default=None):
        """Access config value supporting both attribute and dict access."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    model_cfg = _get(config, "model", config)

    # Voxelization
    voxel_size = _get(model_cfg, "voxel_size", [0.075, 0.075, 0.2])
    point_cloud_range = _get(model_cfg, "point_cloud_range", [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
    max_num_points_per_voxel = _get(model_cfg, "max_num_points_per_voxel", 10)
    max_voxels = _get(model_cfg, "max_voxels", 60000)
    num_point_features = _get(model_cfg, "num_point_features", 4)

    # Backbone 3D
    backbone_3d_cfg = _get(model_cfg, "backbone_3d", {})
    backbone_3d_channels = _get(backbone_3d_cfg, "channels", [16, 32, 64, 128])
    backbone_3d_layers = _get(backbone_3d_cfg, "layers", [2, 2, 2, 2])

    # BEV Backbone
    bev_cfg = _get(model_cfg, "bev_backbone", {})
    bev_in_channels = _get(bev_cfg, "in_channels", None)
    bev_layer_nums = _get(bev_cfg, "layer_nums", [5, 5])
    bev_layer_strides = _get(bev_cfg, "layer_strides", [1, 2])
    bev_num_filters = _get(bev_cfg, "num_filters", [128, 256])
    bev_upsample_strides = _get(bev_cfg, "upsample_strides", [1, 2])
    bev_upsample_filters = _get(bev_cfg, "upsample_filters", [256, 256])

    # Head
    head_cfg = _get(model_cfg, "head", {})
    num_classes = _get(head_cfg, "num_classes", 10)
    tasks = _get(head_cfg, "tasks", None)
    common_heads = _get(head_cfg, "common_heads", None)
    head_conv = _get(head_cfg, "head_conv", 64)

    # Two-stage
    two_stage_cfg = _get(model_cfg, "two_stage", {})
    two_stage_enabled = _get(two_stage_cfg, "enabled", False)
    two_stage_hidden = _get(two_stage_cfg, "hidden_channels", 256)

    # Inference
    score_threshold = _get(model_cfg, "score_threshold", 0.1)
    max_detections = _get(model_cfg, "max_detections", 500)

    model = CenterPoint(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points_per_voxel=max_num_points_per_voxel,
        max_voxels=max_voxels,
        num_point_features=num_point_features,
        backbone_3d_channels=backbone_3d_channels,
        backbone_3d_layers=backbone_3d_layers,
        bev_in_channels=bev_in_channels,
        bev_layer_nums=bev_layer_nums,
        bev_layer_strides=bev_layer_strides,
        bev_num_filters=bev_num_filters,
        bev_upsample_strides=bev_upsample_strides,
        bev_upsample_filters=bev_upsample_filters,
        num_classes=num_classes,
        tasks=tasks,
        common_heads=common_heads,
        head_conv=head_conv,
        two_stage_enabled=two_stage_enabled,
        two_stage_hidden=two_stage_hidden,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

    return model
