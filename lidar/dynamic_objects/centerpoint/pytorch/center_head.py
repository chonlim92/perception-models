"""
CenterPoint Detection Head.

Implements center-based 3D object detection head that predicts object centers
as heatmap peaks and regresses bounding box attributes at center locations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


def gaussian_radius(
    height: float, width: float, min_overlap: float = 0.5
) -> float:
    """Compute minimum Gaussian radius for an object given its BEV size.

    Based on the CornerNet paper: given a minimum IoU overlap, compute
    the smallest Gaussian radius such that the detection is still valid.

    Args:
        height: Object height in BEV pixels.
        width: Object width in BEV pixels.
        min_overlap: Minimum required IoU overlap.

    Returns:
        Minimum Gaussian radius.
    """
    a1 = 1.0
    b1 = -(height + width)
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (-b1 + sq1) / (2.0 * a1)

    a2 = 4.0
    b2 = -2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (-b2 + sq2) / (2.0 * a2)

    a3 = 4.0 * min_overlap
    b3 = 2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (-b3 + sq3) / (2.0 * a3)

    return min(r1, r2, r3)


def draw_gaussian(
    heatmap: torch.Tensor,
    center: Tuple[int, int],
    radius: int,
    k: float = 1.0,
) -> torch.Tensor:
    """Draw a 2D Gaussian on the heatmap at the specified center.

    Args:
        heatmap: (H, W) tensor to draw the Gaussian on.
        center: (x, y) integer center coordinates in the heatmap.
        radius: Gaussian radius (determines sigma = radius / 3).
        k: Peak value of the Gaussian.

    Returns:
        heatmap: Modified heatmap with Gaussian drawn (in-place).
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0  # 3-sigma rule: 99.7% within diameter

    # Generate 2D Gaussian kernel
    x = torch.arange(0, diameter, dtype=torch.float32, device=heatmap.device)
    y = x.unsqueeze(1)
    x0, y0 = radius, radius
    gaussian = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    gaussian = gaussian * k

    # Compute valid bounds
    height, width = heatmap.shape
    cx, cy = int(center[0]), int(center[1])

    left = min(cx, radius)
    right = min(width - cx - 1, radius)
    top = min(cy, radius)
    bottom = min(height - cy - 1, radius)

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return heatmap

    # Extract regions
    heatmap_region = heatmap[cy - top:cy + bottom + 1, cx - left:cx + right + 1]
    gaussian_region = gaussian[
        radius - top:radius + bottom + 1,
        radius - left:radius + right + 1
    ]

    # Element-wise maximum (don't overwrite larger existing values)
    torch.maximum(heatmap_region, gaussian_region, out=heatmap_region)

    return heatmap


class SeparateHead(nn.Module):
    """Shared convolution trunk with separate output branches for each attribute.

    Args:
        in_channels: Number of input feature channels.
        heads: Dictionary mapping head name to output channels, e.g.
               {'heatmap': 3, 'offset': 2, 'height': 1, 'dim': 3, 'rot': 2, 'vel': 2}.
        head_conv: Number of channels in the shared/intermediate conv layers.
        num_conv: Number of shared conv layers before branching.
        final_kernel: Kernel size for the final prediction layer.
    """

    def __init__(
        self,
        in_channels: int,
        heads: Dict[str, int],
        head_conv: int = 64,
        num_conv: int = 2,
        final_kernel: int = 1,
    ):
        super().__init__()
        self.heads = heads

        # Build separate branch for each head
        for head_name, num_output in heads.items():
            layers = []
            prev_ch = in_channels
            for i in range(num_conv):
                layers.append(nn.Conv2d(prev_ch, head_conv, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(head_conv))
                layers.append(nn.ReLU(inplace=True))
                prev_ch = head_conv

            # Final prediction layer
            layers.append(
                nn.Conv2d(head_conv, num_output, kernel_size=final_kernel,
                          padding=final_kernel // 2, bias=True)
            )
            self.add_module(head_name, nn.Sequential(*layers))

        # Initialize final layers
        self._init_weights()

    def _init_weights(self):
        for head_name in self.heads:
            branch = getattr(self, head_name)
            # Initialize final conv layer
            final_layer = branch[-1]
            if isinstance(final_layer, nn.Conv2d):
                nn.init.kaiming_normal_(final_layer.weight, mode='fan_out')
                if final_layer.bias is not None:
                    if head_name == 'heatmap':
                        # Initialize heatmap bias to a negative value (focal loss convention)
                        nn.init.constant_(final_layer.bias, -2.19)
                    else:
                        nn.init.constant_(final_layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input features.

        Returns:
            Dictionary of predictions: {head_name: (B, num_output, H, W)}.
        """
        outputs = {}
        for head_name in self.heads:
            branch = getattr(self, head_name)
            outputs[head_name] = branch(x)
        return outputs


class CenterHead(nn.Module):
    """CenterPoint detection head with task-specific sub-heads.

    Each task handles a group of object classes. For each task, the head predicts:
    - heatmap: Center probability as Gaussian peaks
    - offset: Sub-voxel x, y offset from BEV grid center
    - height: Absolute z center height
    - dim: Log-normalized width, height, length
    - rot: sin(yaw), cos(yaw)
    - vel: vx, vy velocity components

    Args:
        in_channels: Number of BEV feature channels.
        tasks: List of dicts, each with 'num_classes' and 'class_names'.
               E.g., [{'num_classes': 1, 'class_names': ['car']},
                      {'num_classes': 2, 'class_names': ['truck', 'bus']}]
        head_conv: Intermediate conv channels in SeparateHead.
        num_conv: Number of intermediate conv layers.
        common_heads: Dict defining regression heads, e.g.
                      {'offset': 2, 'height': 1, 'dim': 3, 'rot': 2, 'vel': 2}.
        share_conv: Whether to use a shared convolution before task-specific heads.
        share_conv_channels: Channels for shared conv layer.
    """

    def __init__(
        self,
        in_channels: int = 256,
        tasks: Optional[List[Dict]] = None,
        head_conv: int = 64,
        num_conv: int = 2,
        common_heads: Optional[Dict[str, int]] = None,
        share_conv: bool = True,
        share_conv_channels: int = 64,
    ):
        super().__init__()

        if tasks is None:
            tasks = [
                {'num_classes': 1, 'class_names': ['car']},
                {'num_classes': 2, 'class_names': ['truck', 'bus']},
                {'num_classes': 2, 'class_names': ['pedestrian', 'cyclist']},
            ]

        if common_heads is None:
            common_heads = {
                'offset': 2,
                'height': 1,
                'dim': 3,
                'rot': 2,
                'vel': 2,
            }

        self.tasks = tasks
        self.common_heads = common_heads
        self.in_channels = in_channels

        # Optional shared convolution
        if share_conv:
            self.shared_conv = nn.Sequential(
                nn.Conv2d(in_channels, share_conv_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(share_conv_channels),
                nn.ReLU(inplace=True),
            )
            task_in_channels = share_conv_channels
        else:
            self.shared_conv = None
            task_in_channels = in_channels

        # Build a SeparateHead for each task
        self.task_heads = nn.ModuleList()
        for task in tasks:
            heads = dict(common_heads)
            heads['heatmap'] = task['num_classes']
            self.task_heads.append(
                SeparateHead(
                    in_channels=task_in_channels,
                    heads=heads,
                    head_conv=head_conv,
                    num_conv=num_conv,
                )
            )

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) BEV feature map.

        Returns:
            List of prediction dicts, one per task. Each dict has keys:
            'heatmap', 'offset', 'height', 'dim', 'rot', 'vel'.
        """
        if self.shared_conv is not None:
            x = self.shared_conv(x)

        predictions = []
        for task_head in self.task_heads:
            pred = task_head(x)
            # Apply sigmoid to heatmap for probability
            pred['heatmap'] = torch.sigmoid(pred['heatmap'])
            predictions.append(pred)

        return predictions

    def decode_predictions(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        topk: int = 100,
        score_threshold: float = 0.1,
        voxel_size: List[float] = [0.075, 0.075],
        point_cloud_range: List[float] = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        output_stride: int = 8,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode heatmap peaks and regressions into 3D bounding boxes.

        Args:
            predictions: List of task prediction dicts from forward().
            topk: Number of top detections per task.
            score_threshold: Minimum score threshold for detections.
            voxel_size: [vx, vy] BEV voxel size in meters.
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
            output_stride: Stride factor from input to BEV feature map.

        Returns:
            List of decoded detection dicts per batch sample. Each has:
            - 'boxes': (K, 9) [x, y, z, w, h, l, yaw, vx, vy]
            - 'scores': (K,) confidence scores
            - 'labels': (K,) class labels (global across tasks)
        """
        batch_size = predictions[0]['heatmap'].shape[0]
        effective_voxel_x = voxel_size[0] * output_stride
        effective_voxel_y = voxel_size[1] * output_stride

        batch_results = []
        for b in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []
            class_offset = 0

            for task_idx, pred in enumerate(predictions):
                heatmap = pred['heatmap'][b]  # (num_classes, H, W)
                offset = pred['offset'][b]    # (2, H, W)
                height = pred['height'][b]    # (1, H, W)
                dim = pred['dim'][b]          # (3, H, W)
                rot = pred['rot'][b]          # (2, H, W)
                vel = pred['vel'][b]          # (2, H, W)

                num_classes, H, W = heatmap.shape

                # Apply NMS-like max pooling to find peaks
                heatmap_pooled = F.max_pool2d(
                    heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1
                ).squeeze(0)
                # Keep only local maxima
                heatmap_peaks = heatmap * (heatmap == heatmap_pooled).float()

                # Flatten and get topk
                heatmap_flat = heatmap_peaks.view(num_classes, -1)  # (C, H*W)
                topk_scores, topk_flat_idx = heatmap_flat.view(-1).topk(
                    min(topk, heatmap_flat.numel())
                )

                # Filter by score threshold
                valid_mask = topk_scores > score_threshold
                topk_scores = topk_scores[valid_mask]
                topk_flat_idx = topk_flat_idx[valid_mask]

                if topk_scores.numel() == 0:
                    class_offset += num_classes
                    continue

                # Decode flat indices to class, y, x
                topk_class = topk_flat_idx // (H * W)
                topk_spatial = topk_flat_idx % (H * W)
                topk_y = topk_spatial // W
                topk_x = topk_spatial % W

                # Gather regression values at peak locations
                gather_idx_2d = topk_spatial.unsqueeze(0).expand(2, -1)  # (2, K)
                offset_vals = offset.view(2, -1).gather(1, gather_idx_2d)  # (2, K)
                height_vals = height.view(1, -1).gather(1, topk_spatial.unsqueeze(0))  # (1, K)
                dim_vals = dim.view(3, -1).gather(1, topk_spatial.unsqueeze(0).expand(3, -1))  # (3, K)
                rot_vals = rot.view(2, -1).gather(1, gather_idx_2d)  # (2, K)
                vel_vals = vel.view(2, -1).gather(1, gather_idx_2d)  # (2, K)

                # Decode center positions in world coordinates
                # x = (grid_x + offset_x) * effective_voxel_x + x_min
                # y = (grid_y + offset_y) * effective_voxel_y + y_min
                cx = (topk_x.float() + offset_vals[0]) * effective_voxel_x + point_cloud_range[0]
                cy = (topk_y.float() + offset_vals[1]) * effective_voxel_y + point_cloud_range[1]
                cz = height_vals[0]  # Absolute z height

                # Decode dimensions (log-normalized: actual = exp(predicted))
                w = torch.exp(dim_vals[0])
                h = torch.exp(dim_vals[1])
                l = torch.exp(dim_vals[2])

                # Decode rotation
                sin_yaw = rot_vals[0]
                cos_yaw = rot_vals[1]
                yaw = torch.atan2(sin_yaw, cos_yaw)

                # Velocity
                vx = vel_vals[0]
                vy = vel_vals[1]

                # Assemble boxes: (K, 9) [x, y, z, w, h, l, yaw, vx, vy]
                boxes = torch.stack([cx, cy, cz, w, h, l, yaw, vx, vy], dim=1)

                # Labels (global class index)
                labels = topk_class + class_offset

                all_boxes.append(boxes)
                all_scores.append(topk_scores)
                all_labels.append(labels)

                class_offset += num_classes

            if all_boxes:
                batch_results.append({
                    'boxes': torch.cat(all_boxes, dim=0),
                    'scores': torch.cat(all_scores, dim=0),
                    'labels': torch.cat(all_labels, dim=0),
                })
            else:
                device = predictions[0]['heatmap'].device
                batch_results.append({
                    'boxes': torch.zeros(0, 9, device=device),
                    'scores': torch.zeros(0, device=device),
                    'labels': torch.zeros(0, dtype=torch.long, device=device),
                })

        return batch_results

    def generate_targets(
        self,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        feature_map_size: Tuple[int, int],
        voxel_size: List[float] = [0.075, 0.075],
        point_cloud_range: List[float] = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        output_stride: int = 8,
        min_overlap: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate training targets (heatmaps and regression targets).

        Args:
            gt_boxes: List of (Ni, 9) ground truth boxes per sample
                      [x, y, z, w, h, l, yaw, vx, vy].
            gt_labels: List of (Ni,) class labels per sample.
            feature_map_size: (H, W) of the BEV feature map.
            voxel_size: BEV voxel size.
            point_cloud_range: Point cloud range.
            output_stride: Stride from input to feature map.
            min_overlap: Minimum overlap for Gaussian radius computation.

        Returns:
            List of target dicts per task with:
            'heatmap': (B, num_classes, H, W)
            'offset': (B, 2, H, W)
            'height': (B, 1, H, W)
            'dim': (B, 3, H, W)
            'rot': (B, 2, H, W)
            'vel': (B, 2, H, W)
            'mask': (B, H, W) indicator of valid regression targets
        """
        H, W = feature_map_size
        batch_size = len(gt_boxes)
        device = gt_boxes[0].device
        effective_voxel_x = voxel_size[0] * output_stride
        effective_voxel_y = voxel_size[1] * output_stride

        task_targets = []
        class_offset = 0

        for task_idx, task in enumerate(self.tasks):
            num_classes = task['num_classes']

            heatmaps = torch.zeros(batch_size, num_classes, H, W, device=device)
            offsets = torch.zeros(batch_size, 2, H, W, device=device)
            heights = torch.zeros(batch_size, 1, H, W, device=device)
            dims = torch.zeros(batch_size, 3, H, W, device=device)
            rots = torch.zeros(batch_size, 2, H, W, device=device)
            vels = torch.zeros(batch_size, 2, H, W, device=device)
            masks = torch.zeros(batch_size, H, W, device=device)

            for b in range(batch_size):
                boxes = gt_boxes[b]   # (N, 9)
                labels = gt_labels[b]  # (N,)

                # Filter boxes for this task
                task_mask = (labels >= class_offset) & (labels < class_offset + num_classes)
                task_boxes = boxes[task_mask]
                task_labels = labels[task_mask] - class_offset

                for i in range(task_boxes.shape[0]):
                    box = task_boxes[i]
                    cls_id = task_labels[i].long()

                    # Compute center in BEV grid coordinates
                    cx_grid = (box[0] - point_cloud_range[0]) / effective_voxel_x
                    cy_grid = (box[1] - point_cloud_range[1]) / effective_voxel_y

                    # Integer grid position
                    cx_int = int(cx_grid.item())
                    cy_int = int(cy_grid.item())

                    if cx_int < 0 or cx_int >= W or cy_int < 0 or cy_int >= H:
                        continue

                    # Compute Gaussian radius based on object BEV size
                    box_w_pixels = box[3].item() / effective_voxel_x
                    box_l_pixels = box[5].item() / effective_voxel_y
                    radius = max(0, int(gaussian_radius(box_l_pixels, box_w_pixels, min_overlap)))

                    # Draw Gaussian on heatmap
                    draw_gaussian(heatmaps[b, cls_id], (cx_int, cy_int), radius)

                    # Regression targets at center location
                    offsets[b, 0, cy_int, cx_int] = cx_grid - cx_int  # sub-voxel x offset
                    offsets[b, 1, cy_int, cx_int] = cy_grid - cy_int  # sub-voxel y offset
                    heights[b, 0, cy_int, cx_int] = box[2]  # absolute z
                    dims[b, 0, cy_int, cx_int] = torch.log(box[3].clamp(min=1e-5))  # log(w)
                    dims[b, 1, cy_int, cx_int] = torch.log(box[4].clamp(min=1e-5))  # log(h)
                    dims[b, 2, cy_int, cx_int] = torch.log(box[5].clamp(min=1e-5))  # log(l)
                    rots[b, 0, cy_int, cx_int] = torch.sin(box[6])  # sin(yaw)
                    rots[b, 1, cy_int, cx_int] = torch.cos(box[6])  # cos(yaw)
                    vels[b, 0, cy_int, cx_int] = box[7]  # vx
                    vels[b, 1, cy_int, cx_int] = box[8]  # vy
                    masks[b, cy_int, cx_int] = 1.0

            task_targets.append({
                'heatmap': heatmaps,
                'offset': offsets,
                'height': heights,
                'dim': dims,
                'rot': rots,
                'vel': vels,
                'mask': masks,
            })
            class_offset += num_classes

        return task_targets
