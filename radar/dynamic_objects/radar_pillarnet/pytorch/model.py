"""
Main RadarPillarNet model combining all components.

RadarPillarNet is a single-stage 3D object detector for automotive radar point clouds.
It adapts the PointPillars architecture with radar-specific modifications:
- Larger pillars (0.4m) to handle radar sparsity
- Radar features: RCS, radial velocity, time delta
- Multi-sweep accumulation for temporal context
- Velocity regression head for tracking support

Full pipeline:
    Input dict -> PillarEncoder -> PillarScatter -> RadarBEVBackbone -> RadarAnchorHead -> Detections

Input dict keys:
    'pillars': (B, max_pillars, max_points_per_pillar, 9)
    'pillar_indices': (B, max_pillars, 3)
    'num_points_per_pillar': (B, max_pillars)

Output dict keys:
    'cls_preds': (B, N, num_classes) classification logits
    'box_preds': (B, N, 7) box regression deltas
    'vel_preds': (B, N, 2) velocity predictions
    'dir_preds': (B, N, 2) direction classification logits
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from .pillar_encoder import PillarEncoder, PillarScatter
from .backbone import RadarBEVBackbone
from .heads import RadarAnchorHead, AnchorConfig


def _init_weights(module: nn.Module) -> None:
    """Initialize model weights using Kaiming initialization for conv layers
    and constant initialization for batch norm layers.

    Args:
        module: PyTorch module to initialize.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


class RadarPillarNet(nn.Module):
    """RadarPillarNet: Single-stage 3D object detector for radar point clouds.

    Combines pillar encoding, BEV feature extraction, and anchor-based detection
    into a single end-to-end trainable model. Designed for automotive radar with
    multi-sweep accumulation and velocity estimation.

    Architecture:
        PillarEncoder (PointNet) -> PillarScatter (sparse-to-dense)
        -> RadarBEVBackbone (multi-scale conv + FPN) -> RadarAnchorHead (detection)
    """

    def __init__(
        self,
        # Pillar encoder config
        in_channels: int = 9,
        pillar_feat_channels: int = 64,
        x_range: Tuple[float, float] = (-51.2, 51.2),
        y_range: Tuple[float, float] = (-51.2, 51.2),
        z_range: Tuple[float, float] = (-5.0, 3.0),
        pillar_size: Tuple[float, float, float] = (0.4, 0.4, 8.0),
        max_points_per_pillar: int = 20,
        max_pillars: int = 12000,
        # Backbone config
        layer_nums: Optional[List[int]] = None,
        layer_strides: Optional[List[int]] = None,
        num_filters: Optional[List[int]] = None,
        upsample_strides: Optional[List[int]] = None,
        num_upsample_filters: Optional[List[int]] = None,
        # Head config
        num_classes: int = 4,
        anchor_configs: Optional[List[AnchorConfig]] = None,
        nms_threshold: float = 0.2,
        score_threshold: float = 0.1,
        max_detections: int = 300,
    ) -> None:
        """Initialize RadarPillarNet.

        Args:
            in_channels: Number of input features per point (default 9).
            pillar_feat_channels: Output channels from pillar encoder (default 64).
            x_range: Detection range in x (default [-51.2, 51.2] meters).
            y_range: Detection range in y (default [-51.2, 51.2] meters).
            z_range: Detection range in z (default [-5.0, 3.0] meters).
            pillar_size: Pillar dimensions [dx, dy, dz] (default [0.4, 0.4, 8.0] m).
            max_points_per_pillar: Max points per pillar (default 20).
            max_pillars: Max non-empty pillars (default 12000).
            layer_nums: Conv layers per backbone block (default [3, 5, 5]).
            layer_strides: Strides per backbone block (default [1, 2, 2]).
            num_filters: Channels per backbone block (default [64, 128, 256]).
            upsample_strides: FPN upsampling strides (default [1, 2, 4]).
            num_upsample_filters: FPN output channels (default [128, 128, 128]).
            num_classes: Number of detection classes (default 4).
            anchor_configs: Anchor configurations. Uses defaults if None.
            nms_threshold: NMS IoU threshold (default 0.2).
            score_threshold: Minimum detection score (default 0.1).
            max_detections: Maximum detections per sample (default 300).
        """
        super().__init__()

        # Set defaults
        if layer_nums is None:
            layer_nums = [3, 5, 5]
        if layer_strides is None:
            layer_strides = [1, 2, 2]
        if num_filters is None:
            num_filters = [64, 128, 256]
        if upsample_strides is None:
            upsample_strides = [1, 2, 4]
        if num_upsample_filters is None:
            num_upsample_filters = [128, 128, 128]

        # Compute grid dimensions
        grid_x = int(round((x_range[1] - x_range[0]) / pillar_size[0]))
        grid_y = int(round((y_range[1] - y_range[0]) / pillar_size[1]))

        # Feature map size after backbone (stride=1 for first block preserves spatial)
        # Since first block has stride 1, feature map size = grid size
        feature_map_size = (grid_x, grid_y)

        # Store config
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.pillar_size = pillar_size
        self.max_pillars = max_pillars
        self.max_points_per_pillar = max_points_per_pillar
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.num_classes = num_classes

        # Build modules
        self.pillar_encoder = PillarEncoder(
            in_channels=in_channels,
            out_channels=pillar_feat_channels,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            pillar_size=pillar_size,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
        )

        self.pillar_scatter = PillarScatter(
            in_channels=pillar_feat_channels,
            grid_x=grid_x,
            grid_y=grid_y,
        )

        self.backbone = RadarBEVBackbone(
            in_channels=pillar_feat_channels,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
        )

        backbone_out_channels = self.backbone.out_channels  # 384

        self.head = RadarAnchorHead(
            in_channels=backbone_out_channels,
            num_classes=num_classes,
            anchor_configs=anchor_configs,
            feature_map_size=feature_map_size,
            point_range=[x_range[0], y_range[0], z_range[0],
                         x_range[1], y_range[1], z_range[1]],
            nms_threshold=nms_threshold,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )

        # Initialize weights
        self.apply(_init_weights)

        # Re-initialize classification head bias for focal loss
        # Prior probability for positive class ~ 0.01
        prior_prob = 0.01
        bias_init = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.head.conv_cls.bias, bias_init)

    def forward(
        self, batch_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the full detection pipeline.

        Args:
            batch_dict: Dictionary containing:
                'pillars': (B, max_pillars, max_points_per_pillar, 9) pillar features
                'pillar_indices': (B, max_pillars, 3) grid indices [batch_idx, x, y]
                'num_points_per_pillar': (B, max_pillars) point counts per pillar

        Returns:
            Dictionary containing:
                'cls_preds': (B, N, num_classes) classification logits
                'box_preds': (B, N, 7) box regression deltas
                'vel_preds': (B, N, 2) velocity predictions (vx, vy)
                'dir_preds': (B, N, 2) direction classification logits
            where N = H * W * num_anchors_per_location
        """
        pillars = batch_dict["pillars"]  # (B, P, N, 9)
        pillar_indices = batch_dict["pillar_indices"]  # (B, P, 3)
        num_points = batch_dict["num_points_per_pillar"]  # (B, P)

        batch_size = pillars.shape[0]

        # Step 1: Encode pillars with PointNet
        # Input: (B, max_pillars, max_points_per_pillar, 9)
        # Output: (B, max_pillars, 64)
        pillar_features = self.pillar_encoder(pillars, num_points)

        # Step 2: Scatter pillar features to BEV pseudo-image
        # Input: (B, max_pillars, 64) + (B, max_pillars, 3) indices
        # Output: (B, 64, grid_x, grid_y)
        bev_image = self.pillar_scatter(pillar_features, pillar_indices, batch_size)

        # Step 3: Extract multi-scale features with backbone + FPN
        # Input: (B, 64, H, W)
        # Output: (B, 384, H, W)
        bev_features = self.backbone(bev_image)

        # Step 4: Predict detections with anchor head
        # Input: (B, 384, H, W)
        # Output: dict with cls_preds, box_preds, dir_preds, vel_preds
        predictions = self.head(bev_features)

        return predictions

    @torch.no_grad()
    def predict(
        self, batch_dict: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """Run inference with NMS post-processing.

        Full forward pass followed by anchor decoding, direction correction,
        and class-specific NMS to produce final detections.

        Args:
            batch_dict: Same format as forward() input.

        Returns:
            List of dicts (one per batch element), each containing:
                'boxes': (K, 7) decoded 3D bounding boxes [x, y, z, w, l, h, theta]
                'scores': (K,) detection confidence scores
                'labels': (K,) class labels (0-indexed)
                'velocities': (K, 2) predicted velocities [vx, vy] in m/s
        """
        self.eval()

        # Run forward pass
        predictions = self.forward(batch_dict)

        # Post-process with NMS
        results = self.head.predict(
            cls_preds=predictions["cls_preds"],
            box_preds=predictions["box_preds"],
            dir_preds=predictions["dir_preds"],
            vel_preds=predictions["vel_preds"],
        )

        return results

    def get_config(self) -> Dict:
        """Return model configuration as a dictionary.

        Returns:
            Dictionary with all model hyperparameters.
        """
        return {
            "in_channels": self.pillar_encoder.in_channels,
            "pillar_feat_channels": self.pillar_encoder.out_channels,
            "x_range": self.x_range,
            "y_range": self.y_range,
            "z_range": self.z_range,
            "pillar_size": self.pillar_size,
            "max_points_per_pillar": self.max_points_per_pillar,
            "max_pillars": self.max_pillars,
            "grid_size": (self.grid_x, self.grid_y),
            "num_classes": self.num_classes,
            "backbone_out_channels": self.backbone.out_channels,
        }
