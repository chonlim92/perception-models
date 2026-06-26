"""Main MapTR model combining backbone, BEV transform, decoder, and prediction heads.

This module provides the top-level MapTR and MapTRv2 model classes that wire together
all components into a single nn.Module for end-to-end training and inference.

Pipeline: Multi-cam images -> ResNet50+FPN -> GKT (BEV) -> MapDecoder -> MapTRHead

Reference: MapTR: Structured Modeling and Learning for Online Vectorized HD Map
Construction (Liao et al., ICLR 2023)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import ResNet50FPN
from gkt import GKT
from map_decoder import MapDecoder
from heads import MapTRHead


class MapTR(nn.Module):
    """MapTR: End-to-end vectorized HD map construction from multi-camera images.

    Architecture overview:
    1. ResNet50 + FPN backbone extracts multi-scale features from each camera
    2. GKT (Geometry-guided Kernel Transformer) projects features into BEV space
    3. MapDecoder uses hierarchical queries to decode map element instances
    4. MapTRHead predicts class labels and ordered point coordinates

    Args:
        num_cameras: Number of surround-view cameras.
        num_classes: Number of map element classes.
        num_queries: Number of instance queries (map elements to detect).
        num_points: Number of points per map element polyline.
        embed_dims: Embedding dimension used throughout the model.
        backbone_pretrained: Whether to load ImageNet pretrained backbone weights.
        fpn_out_channels: FPN output channel dimension.
        num_fpn_levels: Number of FPN feature pyramid levels.
        bev_h: BEV grid height.
        bev_w: BEV grid width.
        bev_x_range: BEV x-axis range in meters (min, max).
        bev_y_range: BEV y-axis range in meters (min, max).
        bev_z_range: Height range for z-anchor sampling (min, max).
        gkt_num_heads: Number of attention heads in GKT.
        gkt_num_points: Number of deformable sampling points in GKT.
        gkt_num_z_anchors: Number of z-anchor heights in GKT.
        gkt_num_layers: Number of GKT transformer layers.
        decoder_num_heads: Number of attention heads in decoder.
        decoder_ffn_dims: FFN hidden dimension in decoder.
        decoder_num_layers: Number of decoder transformer layers.
        decoder_dropout: Dropout rate in decoder.
        share_head: Whether to share prediction heads across decoder layers.
        use_iterative_refinement: Whether to use iterative coordinate refinement.
    """

    def __init__(
        self,
        num_cameras: int = 6,
        num_classes: int = 3,
        num_queries: int = 50,
        num_points: int = 20,
        embed_dims: int = 256,
        backbone_pretrained: bool = True,
        fpn_out_channels: int = 256,
        num_fpn_levels: int = 4,
        bev_h: int = 200,
        bev_w: int = 100,
        bev_x_range: Tuple[float, float] = (-30.0, 30.0),
        bev_y_range: Tuple[float, float] = (-15.0, 15.0),
        bev_z_range: Tuple[float, float] = (-5.0, 3.0),
        gkt_num_heads: int = 8,
        gkt_num_points: int = 8,
        gkt_num_z_anchors: int = 4,
        gkt_num_layers: int = 3,
        decoder_num_heads: int = 8,
        decoder_ffn_dims: int = 1024,
        decoder_num_layers: int = 6,
        decoder_dropout: float = 0.1,
        share_head: bool = True,
        use_iterative_refinement: bool = True,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_points = num_points
        self.embed_dims = embed_dims
        self.num_fpn_levels = num_fpn_levels

        # --- Backbone: ResNet50 + FPN ---
        self.backbone = ResNet50FPN(
            pretrained=backbone_pretrained,
            fpn_out_channels=fpn_out_channels,
            num_fpn_levels=num_fpn_levels,
        )

        # --- BEV Transform: Geometry-guided Kernel Transformer ---
        # GKT input channels: all FPN levels have fpn_out_channels
        gkt_input_channels = [fpn_out_channels] * min(num_fpn_levels, 3)
        self.gkt = GKT(
            embed_dim=embed_dims,
            bev_h=bev_h,
            bev_w=bev_w,
            num_heads=gkt_num_heads,
            num_points=gkt_num_points,
            num_z_anchors=gkt_num_z_anchors,
            num_layers=gkt_num_layers,
            ffn_dim=embed_dims * 4,
            dropout=0.1,
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
            bev_z_range=bev_z_range,
            input_feat_channels=gkt_input_channels,
        )

        # --- Map Decoder ---
        self.map_decoder = MapDecoder(
            embed_dims=embed_dims,
            num_heads=decoder_num_heads,
            ffn_dims=decoder_ffn_dims,
            num_layers=decoder_num_layers,
            num_queries=num_queries,
            num_points=num_points,
            dropout=decoder_dropout,
            activation="relu",
            self_attn_mask_type="none",
            return_intermediate=True,
        )

        # --- Prediction Heads ---
        self.head = MapTRHead(
            embed_dims=embed_dims,
            num_classes=num_classes,
            num_queries=num_queries,
            num_points=num_points,
            num_decoder_layers=decoder_num_layers,
            share_head_across_layers=share_head,
            use_iterative_refinement=use_iterative_refinement,
        )

    def extract_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from multi-camera images using backbone.

        Args:
            images: Multi-camera images [B, N_cams, 3, H, W].

        Returns:
            List of multi-scale FPN features, each [B * N_cams, C, H_i, W_i].
        """
        return self.backbone(images)

    def transform_to_bev(
        self,
        multi_scale_feats: List[torch.Tensor],
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Transform perspective features to BEV representation using GKT.

        Args:
            multi_scale_feats: List of [B * N_cams, C, H_i, W_i] feature maps.
            camera_intrinsics: Camera intrinsic matrices [B, N_cams, 3, 3].
            camera_extrinsics: World-to-camera extrinsic matrices [B, N_cams, 4, 4].

        Returns:
            BEV features [B, C, bev_h, bev_w].
        """
        return self.gkt(multi_scale_feats, camera_intrinsics, camera_extrinsics)

    def forward(
        self,
        images: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """Full forward pass: images to map element predictions.

        Args:
            images: Multi-camera images [B, N_cams, 3, H, W].
            camera_intrinsics: Camera intrinsic matrices [B, N_cams, 3, 3].
            camera_extrinsics: World-to-camera extrinsic matrices [B, N_cams, 4, 4].

        Returns:
            Dict containing:
                - "cls_scores": List of class logits [B, num_queries, num_classes]
                  per decoder layer.
                - "point_coords": List of point coordinates [B, num_queries, num_points, 2]
                  per decoder layer (normalized to [0, 1]).
                - "bev_features": BEV feature map [B, C, bev_h, bev_w].
        """
        # Step 1: Extract multi-scale features from backbone
        multi_scale_feats = self.extract_features(images)

        # Step 2: Transform to BEV using GKT
        bev_features = self.transform_to_bev(
            multi_scale_feats, camera_intrinsics, camera_extrinsics
        )

        # Step 3: Decode map elements using MapDecoder
        decoder_outputs, reference_points = self.map_decoder(bev_features)

        # Step 4: Predict classes and points using heads
        predictions = self.head(decoder_outputs, reference_points)

        # Include BEV features for potential auxiliary supervision
        predictions["bev_features"] = bev_features

        return predictions

    def inference(
        self,
        images: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        score_threshold: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """Run inference and return final predictions with score thresholding.

        Args:
            images: Multi-camera images [B, N_cams, 3, H, W].
            camera_intrinsics: [B, N_cams, 3, 3].
            camera_extrinsics: [B, N_cams, 4, 4].
            score_threshold: Minimum confidence to keep a prediction.

        Returns:
            Dict with 'scores', 'labels', 'points', 'mask' from the final layer.
        """
        # Get full forward pass outputs
        multi_scale_feats = self.extract_features(images)
        bev_features = self.transform_to_bev(
            multi_scale_feats, camera_intrinsics, camera_extrinsics
        )
        decoder_outputs, reference_points = self.map_decoder(bev_features)

        # Use the head's predict method for post-processed results
        results = self.head.predict(
            decoder_outputs, reference_points, score_threshold=score_threshold
        )
        return results


class DenseBEVHead(nn.Module):
    """Auxiliary dense BEV segmentation head for MapTRv2.

    Predicts per-pixel BEV semantic segmentation as auxiliary supervision
    to improve BEV feature quality.

    Args:
        in_channels: Input BEV feature channels.
        num_classes: Number of segmentation classes.
        hidden_channels: Hidden layer channels.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        hidden_channels: int = 128,
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """Predict dense BEV segmentation.

        Args:
            bev_features: BEV feature map [B, C, bev_h, bev_w].

        Returns:
            Segmentation logits [B, num_classes, bev_h, bev_w].
        """
        return self.conv_layers(bev_features)


class MapTRv2(MapTR):
    """MapTRv2: Enhanced MapTR with auxiliary one-to-many matching and decoupled attention.

    Improvements over MapTR:
    1. Decoupled self-attention: point queries within the same instance attend
       to each other but not across instances (via block-diagonal attention mask).
    2. One-to-many auxiliary matching: additional set of queries that use one-to-many
       matching for denser supervision during training (disabled at inference).
    3. Dense BEV prediction head: auxiliary per-pixel BEV segmentation for improved
       feature quality.

    Args:
        All MapTR args, plus:
        use_decoupled_attn: Whether to use decoupled self-attention in decoder.
        one_to_many_num_groups: Number of one-to-many query groups for auxiliary matching.
        use_dense_bev_head: Whether to add auxiliary dense BEV segmentation.
        dense_bev_num_classes: Number of classes for dense BEV segmentation.
    """

    def __init__(
        self,
        num_cameras: int = 6,
        num_classes: int = 3,
        num_queries: int = 50,
        num_points: int = 20,
        embed_dims: int = 256,
        backbone_pretrained: bool = True,
        fpn_out_channels: int = 256,
        num_fpn_levels: int = 4,
        bev_h: int = 200,
        bev_w: int = 100,
        bev_x_range: Tuple[float, float] = (-30.0, 30.0),
        bev_y_range: Tuple[float, float] = (-15.0, 15.0),
        bev_z_range: Tuple[float, float] = (-5.0, 3.0),
        gkt_num_heads: int = 8,
        gkt_num_points: int = 8,
        gkt_num_z_anchors: int = 4,
        gkt_num_layers: int = 3,
        decoder_num_heads: int = 8,
        decoder_ffn_dims: int = 1024,
        decoder_num_layers: int = 6,
        decoder_dropout: float = 0.1,
        share_head: bool = True,
        use_iterative_refinement: bool = True,
        # MapTRv2-specific
        use_decoupled_attn: bool = True,
        one_to_many_num_groups: int = 6,
        use_dense_bev_head: bool = True,
        dense_bev_num_classes: int = 3,
    ):
        # Store v2-specific params before calling super().__init__
        self._use_decoupled_attn = use_decoupled_attn
        self._one_to_many_num_groups = one_to_many_num_groups
        self._use_dense_bev_head = use_dense_bev_head
        self._dense_bev_num_classes = dense_bev_num_classes

        super().__init__(
            num_cameras=num_cameras,
            num_classes=num_classes,
            num_queries=num_queries,
            num_points=num_points,
            embed_dims=embed_dims,
            backbone_pretrained=backbone_pretrained,
            fpn_out_channels=fpn_out_channels,
            num_fpn_levels=num_fpn_levels,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_x_range=bev_x_range,
            bev_y_range=bev_y_range,
            bev_z_range=bev_z_range,
            gkt_num_heads=gkt_num_heads,
            gkt_num_points=gkt_num_points,
            gkt_num_z_anchors=gkt_num_z_anchors,
            gkt_num_layers=gkt_num_layers,
            decoder_num_heads=decoder_num_heads,
            decoder_ffn_dims=decoder_ffn_dims,
            decoder_num_layers=decoder_num_layers,
            decoder_dropout=decoder_dropout,
            share_head=share_head,
            use_iterative_refinement=use_iterative_refinement,
        )

        # Override decoder with decoupled attention if requested
        if use_decoupled_attn:
            self.map_decoder = MapDecoder(
                embed_dims=embed_dims,
                num_heads=decoder_num_heads,
                ffn_dims=decoder_ffn_dims,
                num_layers=decoder_num_layers,
                num_queries=num_queries,
                num_points=num_points,
                dropout=decoder_dropout,
                activation="relu",
                self_attn_mask_type="decoupled",
                return_intermediate=True,
            )

        # One-to-many auxiliary decoder for denser training supervision
        self.one_to_many_num_groups = one_to_many_num_groups
        if one_to_many_num_groups > 0:
            aux_num_queries = num_queries * one_to_many_num_groups
            self.aux_decoder = MapDecoder(
                embed_dims=embed_dims,
                num_heads=decoder_num_heads,
                ffn_dims=decoder_ffn_dims,
                num_layers=decoder_num_layers,
                num_queries=aux_num_queries,
                num_points=num_points,
                dropout=decoder_dropout,
                activation="relu",
                self_attn_mask_type="decoupled" if use_decoupled_attn else "none",
                return_intermediate=True,
            )
            self.aux_head = MapTRHead(
                embed_dims=embed_dims,
                num_classes=num_classes,
                num_queries=aux_num_queries,
                num_points=num_points,
                num_decoder_layers=decoder_num_layers,
                share_head_across_layers=share_head,
                use_iterative_refinement=use_iterative_refinement,
            )

        # Dense BEV segmentation head
        if use_dense_bev_head:
            self.dense_bev_head = DenseBEVHead(
                in_channels=embed_dims,
                num_classes=dense_bev_num_classes,
                hidden_channels=embed_dims // 2,
            )

    def forward(
        self,
        images: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """Forward pass with MapTRv2 auxiliary outputs.

        Args:
            images: Multi-camera images [B, N_cams, 3, H, W].
            camera_intrinsics: Camera intrinsic matrices [B, N_cams, 3, 3].
            camera_extrinsics: World-to-camera extrinsic matrices [B, N_cams, 4, 4].

        Returns:
            Dict containing:
                - "cls_scores": List per decoder layer [B, num_queries, num_classes].
                - "point_coords": List per decoder layer [B, num_queries, num_points, 2].
                - "bev_features": BEV feature map [B, C, bev_h, bev_w].
                - "aux_cls_scores": (training) One-to-many auxiliary class predictions.
                - "aux_point_coords": (training) One-to-many auxiliary point predictions.
                - "dense_bev_seg": (if enabled) Dense BEV segmentation logits.
        """
        # Step 1: Extract multi-scale features from backbone
        multi_scale_feats = self.extract_features(images)

        # Step 2: Transform to BEV using GKT
        bev_features = self.transform_to_bev(
            multi_scale_feats, camera_intrinsics, camera_extrinsics
        )

        # Step 3: Primary decoder (one-to-one matching)
        decoder_outputs, reference_points = self.map_decoder(bev_features)

        # Step 4: Primary prediction heads
        predictions = self.head(decoder_outputs, reference_points)
        predictions["bev_features"] = bev_features

        # Step 5: One-to-many auxiliary decoder (training only)
        if self.training and self.one_to_many_num_groups > 0:
            aux_decoder_outputs, aux_ref_pts = self.aux_decoder(bev_features)
            aux_predictions = self.aux_head(aux_decoder_outputs, aux_ref_pts)
            predictions["aux_cls_scores"] = aux_predictions["cls_scores"]
            predictions["aux_point_coords"] = aux_predictions["point_coords"]

        # Step 6: Dense BEV segmentation (if enabled)
        if self._use_dense_bev_head:
            dense_seg = self.dense_bev_head(bev_features)
            predictions["dense_bev_seg"] = dense_seg

        return predictions

    def inference(
        self,
        images: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        score_threshold: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """Inference uses only the primary decoder (one-to-one), same as MapTR.

        Args:
            images: Multi-camera images [B, N_cams, 3, H, W].
            camera_intrinsics: [B, N_cams, 3, 3].
            camera_extrinsics: [B, N_cams, 4, 4].
            score_threshold: Minimum confidence to keep a prediction.

        Returns:
            Dict with 'scores', 'labels', 'points', 'mask'.
        """
        multi_scale_feats = self.extract_features(images)
        bev_features = self.transform_to_bev(
            multi_scale_feats, camera_intrinsics, camera_extrinsics
        )
        decoder_outputs, reference_points = self.map_decoder(bev_features)
        results = self.head.predict(
            decoder_outputs, reference_points, score_threshold=score_threshold
        )
        return results


# =============================================================================
# Factory function
# =============================================================================


def build_model(
    model_type: str = "MapTR",
    num_cameras: int = 6,
    num_classes: int = 3,
    num_queries: int = 50,
    num_points: int = 20,
    embed_dims: int = 256,
    backbone_pretrained: bool = True,
    bev_h: int = 200,
    bev_w: int = 100,
    bev_x_range: Tuple[float, float] = (-30.0, 30.0),
    bev_y_range: Tuple[float, float] = (-15.0, 15.0),
    bev_z_range: Tuple[float, float] = (-5.0, 3.0),
    decoder_num_layers: int = 6,
    # MapTRv2-specific
    use_decoupled_attn: bool = True,
    one_to_many_num_groups: int = 6,
    use_dense_bev_head: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function to build MapTR or MapTRv2 models.

    Args:
        model_type: "MapTR" or "MapTRv2".
        num_cameras: Number of surround-view cameras.
        num_classes: Number of map element classes.
        num_queries: Number of instance queries.
        num_points: Number of points per polyline.
        embed_dims: Embedding dimension.
        backbone_pretrained: Whether to use pretrained backbone.
        bev_h: BEV grid height.
        bev_w: BEV grid width.
        bev_x_range: BEV x-axis range in meters.
        bev_y_range: BEV y-axis range in meters.
        bev_z_range: Height range for z-anchor sampling.
        decoder_num_layers: Number of decoder layers.
        use_decoupled_attn: (MapTRv2) Whether to use decoupled attention.
        one_to_many_num_groups: (MapTRv2) Number of one-to-many groups.
        use_dense_bev_head: (MapTRv2) Whether to add dense BEV head.
        **kwargs: Additional keyword arguments forwarded to model constructor.

    Returns:
        Instantiated model (MapTR or MapTRv2).

    Raises:
        ValueError: If model_type is not recognized.
    """
    common_kwargs = dict(
        num_cameras=num_cameras,
        num_classes=num_classes,
        num_queries=num_queries,
        num_points=num_points,
        embed_dims=embed_dims,
        backbone_pretrained=backbone_pretrained,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_x_range=bev_x_range,
        bev_y_range=bev_y_range,
        bev_z_range=bev_z_range,
        decoder_num_layers=decoder_num_layers,
    )
    common_kwargs.update(kwargs)

    if model_type == "MapTR":
        return MapTR(**common_kwargs)
    elif model_type == "MapTRv2":
        return MapTRv2(
            **common_kwargs,
            use_decoupled_attn=use_decoupled_attn,
            one_to_many_num_groups=one_to_many_num_groups,
            use_dense_bev_head=use_dense_bev_head,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: 'MapTR', 'MapTRv2'."
        )
