"""DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries.

This module implements the full DETR3D model, loss function, and post-processor
for 3D object detection from multi-camera images in autonomous driving.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .backbone import ResNet101FPN
from .decoder import DETR3DTransformerDecoder


class DETR3D(nn.Module):
    """DETR3D model for multi-camera 3D object detection.

    The model processes multi-view camera images through a shared backbone,
    then uses a transformer decoder with 3D-to-2D feature sampling to detect
    3D objects. Detection heads predict class logits and 3D bounding box
    parameters (cx, cy, cz, w, l, h, sin, cos, vx, vy).

    Args:
        num_classes: Number of object classes.
        embed_dims: Dimension of the transformer embeddings.
        num_heads: Number of attention heads in the transformer.
        ffn_dims: Dimension of the feed-forward network.
        num_layers: Number of transformer decoder layers.
        num_queries: Number of object queries.
        dropout: Dropout rate.
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        code_size: Dimension of the bounding box code (default 10).
        pretrained_backbone: Whether to use pretrained backbone weights.
        fpn_out_channels: Number of output channels for FPN.
        frozen_backbone_stages: Number of backbone stages to freeze.
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_dims: int = 1024,
        num_layers: int = 6,
        num_queries: int = 900,
        dropout: float = 0.1,
        pc_range: Optional[List[float]] = None,
        code_size: int = 10,
        pretrained_backbone: bool = True,
        fpn_out_channels: int = 256,
        frozen_backbone_stages: int = 1,
    ):
        super().__init__()

        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.code_size = code_size
        self.pc_range = pc_range

        # Backbone: shared across all camera views
        self.backbone = ResNet101FPN(
            pretrained=pretrained_backbone,
            fpn_out_channels=fpn_out_channels,
            frozen_stages=frozen_backbone_stages,
        )

        # Transformer decoder with built-in queries and reference points
        self.decoder = DETR3DTransformerDecoder(
            embed_dims=embed_dims,
            num_heads=num_heads,
            ffn_dims=ffn_dims,
            num_layers=num_layers,
            dropout=dropout,
            num_queries=num_queries,
            pc_range=pc_range,
        )

        # Detection heads (shared across all decoder layers for auxiliary losses)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_classes),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, code_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize detection head weights."""
        for module in [self.cls_head, self.reg_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        # Bias initialization for classification head (focal loss prior)
        prior_prob = 0.01
        bias_value = -torch.log(
            torch.tensor((1 - prior_prob) / prior_prob)
        ).item()
        self.cls_head[-1].bias.data.fill_(bias_value)

    def _extract_multi_view_features(
        self, images: torch.Tensor
    ) -> List[torch.Tensor]:
        """Extract multi-scale features from multi-view images.

        Processes each camera view independently through the shared backbone
        and concatenates features along the camera dimension.

        Args:
            images: Multi-camera images of shape (B, num_cams, 3, H, W).

        Returns:
            List of multi-scale feature tensors. Each tensor has shape
            (B, num_cams, C, H_i, W_i) for scale i.
        """
        batch_size, num_cams = images.shape[:2]

        # Reshape to process all views through backbone at once
        # (B * num_cams, 3, H, W)
        imgs_flat = images.flatten(0, 1)

        # Extract multi-scale features: list of (B*num_cams, C, H_i, W_i)
        multi_scale_feats = self.backbone.get_multi_scale_features(imgs_flat)

        # Reshape back to separate batch and camera dimensions
        # Each: (B, num_cams, C, H_i, W_i)
        multi_scale_feats_reshaped = []
        for feat in multi_scale_feats:
            _, c, h, w = feat.shape
            feat_reshaped = feat.view(batch_size, num_cams, c, h, w)
            multi_scale_feats_reshaped.append(feat_reshaped)

        return multi_scale_feats_reshaped

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of DETR3D.

        Args:
            images: Multi-camera images, shape (B, num_cams, 3, H, W).
            intrinsics: Camera intrinsic matrices, shape (B, num_cams, 3, 3).
            extrinsics: Camera extrinsic matrices, shape (B, num_cams, 4, 4).
            image_shape: Tuple of (H, W) representing input image dimensions.

        Returns:
            Dictionary containing:
                - 'pred_logits': Classification logits (B, num_queries, num_classes).
                - 'pred_boxes': Bounding box predictions (B, num_queries, code_size).
                - 'aux_outputs': List of dicts with same keys from intermediate layers.
        """
        # Extract multi-scale features from all camera views
        multi_scale_features = self._extract_multi_view_features(images)

        # Transformer decoder with 3D-to-2D feature sampling
        # The decoder internally manages queries and reference points
        query_outputs, intermediate_outputs, intermediate_ref_points = (
            self.decoder(
                multi_scale_features, intrinsics, extrinsics, image_shape
            )
        )

        # Apply detection heads to final decoder output
        pred_logits = self.cls_head(query_outputs)
        pred_boxes = self.reg_head(query_outputs)

        outputs = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
        }

        # Auxiliary outputs from intermediate decoder layers
        aux_outputs = []
        for intermediate_out in intermediate_outputs:
            aux_logits = self.cls_head(intermediate_out)
            aux_boxes = self.reg_head(intermediate_out)
            aux_outputs.append(
                {
                    "pred_logits": aux_logits,
                    "pred_boxes": aux_boxes,
                }
            )
        outputs["aux_outputs"] = aux_outputs

        return outputs


class DETR3DLoss(nn.Module):
    """Loss function for DETR3D with Hungarian matching.

    Computes the total loss as a combination of:
    - Focal loss for classification
    - L1 loss for bounding box regression
    - Auxiliary losses from intermediate decoder layers

    Args:
        num_classes: Number of object classes.
        code_size: Dimension of the bounding box code.
        cls_weight: Weight for classification loss.
        reg_weight: Weight for regression loss.
        focal_alpha: Alpha parameter for focal loss.
        focal_gamma: Gamma parameter for focal loss.
    """

    def __init__(
        self,
        num_classes: int = 10,
        code_size: int = 10,
        cls_weight: float = 2.0,
        reg_weight: float = 0.25,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.code_size = code_size
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def _hungarian_matching(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform Hungarian matching between predictions and ground truth.

        Args:
            pred_logits: Predicted class logits (num_queries, num_classes).
            pred_boxes: Predicted boxes (num_queries, code_size).
            gt_labels: Ground truth labels (num_gt,).
            gt_boxes: Ground truth boxes (num_gt, code_size).

        Returns:
            Tuple of (matched_pred_indices, matched_gt_indices).
        """
        num_queries = pred_logits.shape[0]
        num_gt = gt_labels.shape[0]

        if num_gt == 0:
            return (
                torch.tensor([], dtype=torch.long, device=pred_logits.device),
                torch.tensor([], dtype=torch.long, device=pred_logits.device),
            )

        # Classification cost: use focal-loss-based cost
        pred_probs = pred_logits.sigmoid()  # (num_queries, num_classes)
        # Gather the probabilities for the GT classes
        # cost_cls shape: (num_queries, num_gt)
        alpha = self.focal_alpha
        gamma = self.focal_gamma

        # Negative focal cost for each query-gt pair
        pred_probs_gt = pred_probs[:, gt_labels]  # (num_queries, num_gt)
        neg_cost_cls = (
            -(1 - alpha)
            * (pred_probs_gt**gamma)
            * torch.log(1 - pred_probs_gt + 1e-8)
        )
        pos_cost_cls = (
            -alpha
            * ((1 - pred_probs_gt) ** gamma)
            * torch.log(pred_probs_gt + 1e-8)
        )
        cost_cls = pos_cost_cls - neg_cost_cls  # (num_queries, num_gt)

        # Regression cost: L1 distance between predicted and GT boxes
        # (num_queries, num_gt)
        cost_reg = torch.cdist(pred_boxes, gt_boxes, p=1)

        # Total cost matrix
        cost_matrix = (
            self.cls_weight * cost_cls + self.reg_weight * cost_reg
        )

        # Solve assignment with scipy
        cost_np = cost_matrix.detach().cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(cost_np)

        return (
            torch.tensor(row_indices, dtype=torch.long, device=pred_logits.device),
            torch.tensor(col_indices, dtype=torch.long, device=pred_logits.device),
        )

    def _focal_loss(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sigmoid focal loss for classification.

        Args:
            pred_logits: Predicted logits (num_queries, num_classes).
            target_labels: Target class indices (num_queries,). Use
                num_classes as the background/no-object class.

        Returns:
            Scalar focal loss.
        """
        num_queries, num_classes = pred_logits.shape

        # Create one-hot targets; background class maps to all-zeros
        target_one_hot = torch.zeros_like(pred_logits)
        foreground_mask = target_labels < num_classes
        if foreground_mask.any():
            target_one_hot[foreground_mask] = F.one_hot(
                target_labels[foreground_mask], num_classes
            ).float()

        pred_probs = pred_logits.sigmoid()
        # Binary cross entropy per class
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target_one_hot, reduction="none"
        )

        # Focal modulation
        p_t = pred_probs * target_one_hot + (1 - pred_probs) * (
            1 - target_one_hot
        )
        alpha_t = self.focal_alpha * target_one_hot + (
            1 - self.focal_alpha
        ) * (1 - target_one_hot)
        focal_weight = alpha_t * ((1 - p_t) ** self.focal_gamma)

        loss = (focal_weight * bce).sum() / max(foreground_mask.sum().item(), 1)
        return loss

    def _regression_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        matched_pred_indices: torch.Tensor,
        matched_gt_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L1 regression loss for matched predictions.

        Args:
            pred_boxes: Predicted boxes (num_queries, code_size).
            target_boxes: GT boxes (num_gt, code_size).
            matched_pred_indices: Indices of matched predictions.
            matched_gt_indices: Indices of matched ground truths.

        Returns:
            Scalar L1 regression loss.
        """
        if len(matched_pred_indices) == 0:
            return pred_boxes.sum() * 0.0

        matched_pred = pred_boxes[matched_pred_indices]
        matched_gt = target_boxes[matched_gt_indices]

        loss = F.l1_loss(matched_pred, matched_gt, reduction="mean")
        return loss

    def _loss_single_layer(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for a single decoder layer output.

        Args:
            pred_logits: Predictions (B, num_queries, num_classes).
            pred_boxes: Predictions (B, num_queries, code_size).
            targets: List of dicts per batch element with 'labels' and 'boxes'.

        Returns:
            Dict with 'loss_cls' and 'loss_reg' tensors.
        """
        batch_size = pred_logits.shape[0]
        num_queries = pred_logits.shape[1]
        device = pred_logits.device

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            b_logits = pred_logits[b]  # (num_queries, num_classes)
            b_boxes = pred_boxes[b]  # (num_queries, code_size)
            gt_labels = targets[b]["labels"]  # (num_gt,)
            gt_boxes = targets[b]["boxes"]  # (num_gt, code_size)

            # Hungarian matching
            matched_pred_idx, matched_gt_idx = self._hungarian_matching(
                b_logits, b_boxes, gt_labels, gt_boxes
            )

            # Build target labels for all queries (background = num_classes)
            target_labels = torch.full(
                (num_queries,),
                self.num_classes,
                dtype=torch.long,
                device=device,
            )
            if len(matched_pred_idx) > 0:
                target_labels[matched_pred_idx] = gt_labels[matched_gt_idx]

            # Classification loss
            cls_loss = self._focal_loss(b_logits, target_labels)
            total_cls_loss = total_cls_loss + cls_loss

            # Regression loss (only on matched pairs)
            reg_loss = self._regression_loss(
                b_boxes, gt_boxes, matched_pred_idx, matched_gt_idx
            )
            total_reg_loss = total_reg_loss + reg_loss

        # Average over batch
        total_cls_loss = total_cls_loss / batch_size
        total_reg_loss = total_reg_loss / batch_size

        return {
            "loss_cls": total_cls_loss,
            "loss_reg": total_reg_loss,
        }

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute the total DETR3D loss.

        Args:
            predictions: Model output dict with keys:
                - 'pred_logits': (B, num_queries, num_classes)
                - 'pred_boxes': (B, num_queries, code_size)
                - 'aux_outputs': list of dicts with same keys
            targets: List of dicts (one per batch element), each containing:
                - 'labels': (num_gt,) class indices
                - 'boxes': (num_gt, code_size) box parameters

        Returns:
            Dictionary of losses:
                - 'loss_cls': Classification focal loss.
                - 'loss_reg': Bounding box L1 regression loss.
                - 'loss_cls_aux_i': Auxiliary classification loss for layer i.
                - 'loss_reg_aux_i': Auxiliary regression loss for layer i.
                - 'total_loss': Weighted sum of all losses.
        """
        # Final layer losses
        losses = self._loss_single_layer(
            predictions["pred_logits"],
            predictions["pred_boxes"],
            targets,
        )

        loss_dict = {
            "loss_cls": self.cls_weight * losses["loss_cls"],
            "loss_reg": self.reg_weight * losses["loss_reg"],
        }

        # Auxiliary losses from intermediate decoder layers
        if "aux_outputs" in predictions:
            for i, aux_output in enumerate(predictions["aux_outputs"]):
                aux_losses = self._loss_single_layer(
                    aux_output["pred_logits"],
                    aux_output["pred_boxes"],
                    targets,
                )
                loss_dict[f"loss_cls_aux_{i}"] = (
                    self.cls_weight * aux_losses["loss_cls"]
                )
                loss_dict[f"loss_reg_aux_{i}"] = (
                    self.reg_weight * aux_losses["loss_reg"]
                )

        # Total loss
        loss_dict["total_loss"] = sum(loss_dict.values())

        return loss_dict


class DETR3DPostProcessor(nn.Module):
    """Post-processor for DETR3D inference.

    Applies sigmoid to classification logits, filters detections by a score
    threshold, and returns the top-k results.

    Args:
        num_classes: Number of object classes.
        score_threshold: Minimum confidence score for a detection.
        top_k: Maximum number of detections to return per sample.
        pc_range: Point cloud range for denormalizing box predictions.
    """

    def __init__(
        self,
        num_classes: int = 10,
        score_threshold: float = 0.1,
        top_k: int = 300,
        pc_range: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.top_k = top_k
        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.register_buffer(
            "pc_range", torch.tensor(pc_range, dtype=torch.float32)
        )

    @torch.no_grad()
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process model predictions for inference.

        Args:
            predictions: Model output dict with keys:
                - 'pred_logits': (B, num_queries, num_classes)
                - 'pred_boxes': (B, num_queries, code_size)

        Returns:
            List of dicts (one per batch element), each containing:
                - 'scores': (num_detections,) confidence scores.
                - 'labels': (num_detections,) predicted class indices.
                - 'boxes': (num_detections, code_size) predicted boxes.
        """
        pred_logits = predictions["pred_logits"]  # (B, Q, C)
        pred_boxes = predictions["pred_boxes"]  # (B, Q, code_size)

        batch_size = pred_logits.shape[0]

        # Apply sigmoid to get class probabilities
        pred_scores = pred_logits.sigmoid()  # (B, Q, C)

        results = []
        for b in range(batch_size):
            scores = pred_scores[b]  # (Q, C)
            boxes = pred_boxes[b]  # (Q, code_size)

            # Get maximum score and corresponding class per query
            max_scores, max_labels = scores.max(dim=-1)  # (Q,), (Q,)

            # Filter by score threshold
            valid_mask = max_scores > self.score_threshold
            valid_scores = max_scores[valid_mask]
            valid_labels = max_labels[valid_mask]
            valid_boxes = boxes[valid_mask]

            # Select top-k by score
            num_valid = valid_scores.shape[0]
            if num_valid > self.top_k:
                topk_scores, topk_indices = valid_scores.topk(
                    self.top_k, sorted=True
                )
                topk_labels = valid_labels[topk_indices]
                topk_boxes = valid_boxes[topk_indices]
            else:
                # Sort by score descending
                sorted_indices = valid_scores.argsort(descending=True)
                topk_scores = valid_scores[sorted_indices]
                topk_labels = valid_labels[sorted_indices]
                topk_boxes = valid_boxes[sorted_indices]

            # Denormalize box center predictions (cx, cy, cz) from [0,1] to pc_range
            denorm_boxes = topk_boxes.clone()
            pc_range = self.pc_range
            # cx: index 0
            denorm_boxes[:, 0] = (
                denorm_boxes[:, 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
            )
            # cy: index 1
            denorm_boxes[:, 1] = (
                denorm_boxes[:, 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
            )
            # cz: index 2
            denorm_boxes[:, 2] = (
                denorm_boxes[:, 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
            )

            results.append(
                {
                    "scores": topk_scores,
                    "labels": topk_labels,
                    "boxes": denorm_boxes,
                }
            )

        return results
