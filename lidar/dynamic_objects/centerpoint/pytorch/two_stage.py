"""
CenterPoint Two-Stage Refinement Module.

Implements second-stage refinement that extracts BEV features at predicted
center locations and refines bounding box predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class PointFeatureExtractor(nn.Module):
    """Extract multi-scale BEV features at predicted center locations.

    Uses bilinear interpolation (grid_sample) to extract features at arbitrary
    (non-integer) BEV coordinates from multiple feature map scales.

    Args:
        bev_channels: List of channel dimensions for each BEV feature scale.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy] base voxel size.
        output_strides: List of output strides for each feature map scale.
    """

    def __init__(
        self,
        bev_channels: List[int] = [128, 256],
        point_cloud_range: List[float] = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        voxel_size: List[float] = [0.075, 0.075],
        output_strides: List[int] = [8, 16],
    ):
        super().__init__()
        self.bev_channels = bev_channels
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.output_strides = output_strides
        self.out_channels = sum(bev_channels)

    def normalize_coords(
        self,
        centers: torch.Tensor,
        feature_map_size: Tuple[int, int],
        stride: int,
    ) -> torch.Tensor:
        """Normalize world coordinates to [-1, 1] for grid_sample.

        Args:
            centers: (N, 2) world coordinates (x, y).
            feature_map_size: (H, W) spatial size of the feature map.
            stride: Stride of the feature map.

        Returns:
            normalized: (N, 2) coordinates in [-1, 1] range.
        """
        x_min = self.point_cloud_range[0]
        y_min = self.point_cloud_range[1]
        x_max = self.point_cloud_range[3]
        y_max = self.point_cloud_range[4]

        # Normalize x to [-1, 1]
        norm_x = 2.0 * (centers[:, 0] - x_min) / (x_max - x_min) - 1.0
        # Normalize y to [-1, 1]
        norm_y = 2.0 * (centers[:, 1] - y_min) / (y_max - y_min) - 1.0

        return torch.stack([norm_x, norm_y], dim=1)

    def forward(
        self,
        bev_features: List[torch.Tensor],
        centers: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features at center locations from multi-scale BEV maps.

        Args:
            bev_features: List of (B, Ci, Hi, Wi) BEV feature maps at different scales.
            centers: (N, 2) world coordinates (x, y) of predicted centers.
            batch_indices: (N,) batch index for each center.

        Returns:
            features: (N, sum(Ci)) concatenated multi-scale features per center.
        """
        device = centers.device
        N = centers.shape[0]
        all_features = []

        for scale_idx, (feat_map, stride) in enumerate(zip(bev_features, self.output_strides)):
            B, C, H, W = feat_map.shape

            # Normalize coordinates for this scale
            norm_coords = self.normalize_coords(centers, (H, W), stride)

            # grid_sample expects (B, H_out, W_out, 2) grid
            # We'll process per-batch for correct indexing
            scale_features = torch.zeros(N, C, dtype=feat_map.dtype, device=device)

            for b in range(B):
                mask = batch_indices == b
                if not mask.any():
                    continue

                batch_coords = norm_coords[mask]  # (Nb, 2)
                Nb = batch_coords.shape[0]

                # Reshape for grid_sample: (1, 1, Nb, 2) - treat as 1xNb spatial grid
                grid = batch_coords.view(1, 1, Nb, 2)

                # Extract features using bilinear interpolation
                # feat_map[b:b+1] is (1, C, H, W)
                sampled = F.grid_sample(
                    feat_map[b:b+1],
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False,
                )  # (1, C, 1, Nb)

                scale_features[mask] = sampled.squeeze(0).squeeze(1).t()  # (Nb, C)

            all_features.append(scale_features)

        # Concatenate features from all scales
        return torch.cat(all_features, dim=1)  # (N, sum(Ci))


class RefinementHead(nn.Module):
    """MLP head that refines first-stage predictions.

    Takes extracted BEV features and first-stage predictions, then outputs
    refined offsets for center, size, rotation, and confidence.

    Args:
        in_channels: Input feature dimension (from PointFeatureExtractor).
        hidden_channels: Hidden layer dimensions.
        num_classes: Number of object classes for confidence prediction.
    """

    def __init__(
        self,
        in_channels: int = 384,
        hidden_channels: List[int] = [256, 128],
        num_classes: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Shared MLP trunk
        layers = []
        prev_ch = in_channels
        for h_ch in hidden_channels:
            layers.append(nn.Linear(prev_ch, h_ch))
            layers.append(nn.BatchNorm1d(h_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
            prev_ch = h_ch
        self.shared_mlp = nn.Sequential(*layers)

        # Output heads
        self.offset_head = nn.Linear(prev_ch, 3)    # dx, dy, dz refinement
        self.size_head = nn.Linear(prev_ch, 3)      # dw, dh, dl refinement
        self.rot_head = nn.Linear(prev_ch, 2)       # sin(dyaw), cos(dyaw) refinement
        self.conf_head = nn.Linear(prev_ch, num_classes)  # refined confidence per class

        self._init_weights()

    def _init_weights(self):
        for module in [self.offset_head, self.size_head, self.rot_head]:
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.conf_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.conf_head.bias, 0.0)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (N, C) extracted features for each proposal.

        Returns:
            Dict with:
            - 'offset': (N, 3) center offset refinements [dx, dy, dz]
            - 'size': (N, 3) size refinements [dw, dh, dl]
            - 'rot': (N, 2) rotation refinement [sin(dyaw), cos(dyaw)]
            - 'confidence': (N, num_classes) refined class confidence
        """
        x = self.shared_mlp(features)

        return {
            'offset': self.offset_head(x),
            'size': self.size_head(x),
            'rot': self.rot_head(x),
            'confidence': self.conf_head(x),
        }


class CenterPointTwoStage(nn.Module):
    """CenterPoint two-stage detection pipeline.

    Combines first-stage predictions from CenterHead with a second-stage
    refinement network that extracts BEV features at predicted centers
    and refines the predictions.

    Args:
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy] base voxel size.
        bev_channels: List of channel dims for multi-scale BEV features.
        output_strides: List of output strides for each BEV scale.
        hidden_channels: Hidden dims for refinement MLP.
        num_classes: Total number of object classes.
        nms_iou_threshold: IoU threshold for NMS after refinement.
        score_threshold: Minimum score for final detections.
    """

    def __init__(
        self,
        point_cloud_range: List[float] = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        voxel_size: List[float] = [0.075, 0.075],
        bev_channels: List[int] = [128, 256],
        output_strides: List[int] = [8, 16],
        hidden_channels: List[int] = [256, 128],
        num_classes: int = 5,
        nms_iou_threshold: float = 0.7,
        score_threshold: float = 0.1,
    ):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold

        self.feature_extractor = PointFeatureExtractor(
            bev_channels=bev_channels,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            output_strides=output_strides,
        )

        self.refinement_head = RefinementHead(
            in_channels=self.feature_extractor.out_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
        )

    def forward(
        self,
        bev_features: List[torch.Tensor],
        first_stage_predictions: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            bev_features: List of multi-scale BEV feature maps [(B, C1, H1, W1), ...].
            first_stage_predictions: List of decoded predictions per batch sample
                                     from CenterHead.decode_predictions().

        Returns:
            List of refined detection dicts per batch sample:
            - 'boxes': (K, 9) refined [x, y, z, w, h, l, yaw, vx, vy]
            - 'scores': (K,) refined confidence
            - 'labels': (K,) class labels
        """
        batch_size = bev_features[0].shape[0]
        refined_results = []

        for b in range(batch_size):
            pred = first_stage_predictions[b]
            boxes = pred['boxes']        # (K, 9) [x, y, z, w, h, l, yaw, vx, vy]
            scores = pred['scores']      # (K,)
            labels = pred['labels']      # (K,)

            if boxes.shape[0] == 0:
                refined_results.append(pred)
                continue

            # Extract centers (x, y)
            centers = boxes[:, :2]  # (K, 2)
            batch_indices = torch.full(
                (centers.shape[0],), b, dtype=torch.long, device=centers.device
            )

            # Extract multi-scale features at center locations
            features = self.feature_extractor(bev_features, centers, batch_indices)

            # Refinement
            refinements = self.refinement_head(features)

            # Apply refinements to first-stage predictions
            refined_boxes = boxes.clone()

            # Center refinement
            refined_boxes[:, 0] += refinements['offset'][:, 0]
            refined_boxes[:, 1] += refinements['offset'][:, 1]
            refined_boxes[:, 2] += refinements['offset'][:, 2]

            # Size refinement (multiplicative in log-space)
            refined_boxes[:, 3] *= torch.exp(refinements['size'][:, 0])
            refined_boxes[:, 4] *= torch.exp(refinements['size'][:, 1])
            refined_boxes[:, 5] *= torch.exp(refinements['size'][:, 2])

            # Rotation refinement
            sin_orig = torch.sin(boxes[:, 6])
            cos_orig = torch.cos(boxes[:, 6])
            sin_delta = refinements['rot'][:, 0]
            cos_delta = refinements['rot'][:, 1]
            # Angle addition: sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
            #                  cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
            new_sin = sin_orig * cos_delta + cos_orig * sin_delta
            new_cos = cos_orig * cos_delta - sin_orig * sin_delta
            refined_boxes[:, 6] = torch.atan2(new_sin, new_cos)

            # Velocity stays unchanged (first-stage prediction)
            # refined_boxes[:, 7:9] already has vx, vy from clone

            # Refined confidence: combine first-stage score with refinement
            refined_conf = torch.sigmoid(refinements['confidence'])  # (K, num_classes)
            # Use the confidence for the predicted class
            labels_clamped = labels.clamp(0, self.num_classes - 1).long()
            class_conf = refined_conf[torch.arange(labels_clamped.shape[0], device=labels.device), labels_clamped]
            refined_scores = scores * class_conf

            # Filter by score threshold
            valid = refined_scores > self.score_threshold
            refined_results.append({
                'boxes': refined_boxes[valid],
                'scores': refined_scores[valid],
                'labels': labels[valid],
            })

        return refined_results

    def forward_train(
        self,
        bev_features: List[torch.Tensor],
        proposals: List[Dict[str, torch.Tensor]],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        iou_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with ground truth assignment.

        Args:
            bev_features: Multi-scale BEV features.
            proposals: First-stage decoded proposals per sample.
            gt_boxes: List of (Mi, 9) ground truth boxes per sample.
            gt_labels: List of (Mi,) ground truth labels per sample.
            iou_threshold: IoU threshold for positive assignment.

        Returns:
            Dict of training losses for the refinement stage.
        """
        batch_size = bev_features[0].shape[0]
        all_features = []
        all_offset_targets = []
        all_size_targets = []
        all_rot_targets = []
        all_conf_targets = []

        for b in range(batch_size):
            pred = proposals[b]
            boxes = pred['boxes']
            labels = pred['labels']

            if boxes.shape[0] == 0:
                continue

            centers = boxes[:, :2]
            batch_indices = torch.full(
                (centers.shape[0],), b, dtype=torch.long, device=centers.device
            )

            # Extract features
            features = self.feature_extractor(bev_features, centers, batch_indices)
            all_features.append(features)

            # Assign proposals to ground truth using center distance
            gt = gt_boxes[b]  # (M, 9)
            if gt.shape[0] == 0:
                # All proposals are negative
                all_offset_targets.append(torch.zeros(boxes.shape[0], 3, device=boxes.device))
                all_size_targets.append(torch.zeros(boxes.shape[0], 3, device=boxes.device))
                all_rot_targets.append(torch.zeros(boxes.shape[0], 2, device=boxes.device))
                all_conf_targets.append(torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device))
                continue

            # Compute distance between proposals and GT centers
            prop_centers = boxes[:, :3]  # (K, 3)
            gt_centers = gt[:, :3]  # (M, 3)
            dist_matrix = torch.cdist(prop_centers, gt_centers)  # (K, M)

            # Assign each proposal to nearest GT (simple distance-based)
            min_dist, gt_assignment = dist_matrix.min(dim=1)  # (K,)

            # Positive mask: proposals within threshold distance
            # Use a simple distance threshold related to GT size
            assigned_gt = gt[gt_assignment]  # (K, 9)
            size_threshold = torch.sqrt(
                assigned_gt[:, 3] ** 2 + assigned_gt[:, 5] ** 2
            ) * iou_threshold
            positive_mask = min_dist < size_threshold

            # Regression targets for positive proposals
            offset_targets = assigned_gt[:, :3] - boxes[:, :3]
            size_targets = torch.log(
                assigned_gt[:, 3:6].clamp(min=1e-5) / boxes[:, 3:6].clamp(min=1e-5)
            )
            # Rotation target: delta angle
            delta_yaw = assigned_gt[:, 6] - boxes[:, 6]
            rot_targets = torch.stack([torch.sin(delta_yaw), torch.cos(delta_yaw)], dim=1)

            # Confidence target: 1 for positive, 0 for negative
            conf_targets = positive_mask.long()

            all_offset_targets.append(offset_targets)
            all_size_targets.append(size_targets)
            all_rot_targets.append(rot_targets)
            all_conf_targets.append(conf_targets)

        if not all_features:
            device = bev_features[0].device
            return {
                'refine_offset_loss': torch.tensor(0.0, device=device),
                'refine_size_loss': torch.tensor(0.0, device=device),
                'refine_rot_loss': torch.tensor(0.0, device=device),
                'refine_conf_loss': torch.tensor(0.0, device=device),
            }

        # Concatenate all proposals
        all_features = torch.cat(all_features, dim=0)
        all_offset_targets = torch.cat(all_offset_targets, dim=0)
        all_size_targets = torch.cat(all_size_targets, dim=0)
        all_rot_targets = torch.cat(all_rot_targets, dim=0)
        all_conf_targets = torch.cat(all_conf_targets, dim=0)

        # Forward through refinement head
        refinements = self.refinement_head(all_features)

        # Compute losses
        # Offset loss (only for positives)
        positive_mask = all_conf_targets > 0
        num_pos = positive_mask.sum().clamp(min=1).float()

        offset_loss = F.smooth_l1_loss(
            refinements['offset'][positive_mask],
            all_offset_targets[positive_mask],
            reduction='sum'
        ) / num_pos

        size_loss = F.smooth_l1_loss(
            refinements['size'][positive_mask],
            all_size_targets[positive_mask],
            reduction='sum'
        ) / num_pos

        rot_loss = F.smooth_l1_loss(
            refinements['rot'][positive_mask],
            all_rot_targets[positive_mask],
            reduction='sum'
        ) / num_pos

        # Confidence loss (binary cross entropy)
        # Use first column as positive indicator
        conf_logits = refinements['confidence'][:, 0]  # Simplify to binary
        conf_loss = F.binary_cross_entropy_with_logits(
            conf_logits, all_conf_targets.float(), reduction='mean'
        )

        return {
            'refine_offset_loss': offset_loss,
            'refine_size_loss': size_loss,
            'refine_rot_loss': rot_loss,
            'refine_conf_loss': conf_loss,
        }
