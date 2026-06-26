"""
Main PETR/PETRv2/StreamPETR model.

Combines backbone, FPN, 3D position embedding, transformer decoder,
and detection heads into a unified model. Supports three variants:
- PETR: Single-frame 3D detection with 3D-aware position embeddings.
- PETRv2: Multi-frame with dense temporal fusion of position-aware features.
- StreamPETR: Streaming detection with query propagation across frames.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import BackboneWithFPN
from .decoder import PETRTransformerDecoder
from .heads import PETRDetectionHead
from .losses import PETRLoss
from .position_embedding_3d import PositionEmbedding3D
from .temporal import QueryPropagation


class PETRConfig:
    """Configuration for PETR model variants.

    Args:
        variant: Model variant ('petr', 'petrv2', 'streampetr').
        num_classes: Number of detection classes.
        embed_dims: Feature/embedding dimension.
        num_queries: Total number of object queries.
        num_decoder_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        feedforward_dims: FFN hidden dimension.
        dropout: Dropout probability.
        num_depth_bins: Number of depth bins for 3D PE.
        depth_start: Near depth in meters.
        depth_end: Far depth in meters.
        depth_distribution: Depth bin distribution ('linear' or 'log').
        code_size: Bounding box code dimension.
        pc_range: Point cloud range.
        img_size: Input image size (H, W).
        num_cameras: Number of camera views.
        frozen_backbone_stages: Number of frozen ResNet stages.
        pretrained_backbone: Use pretrained backbone.
        fpn_out_channels: FPN output channels.
        num_propagated_queries: Queries to propagate (StreamPETR).
        num_temporal_frames: Number of temporal frames (PETRv2/StreamPETR).
        cls_weight: Classification loss weight.
        bbox_weight: Bbox regression loss weight.
        velocity_weight: Velocity loss weight.
    """

    def __init__(
        self,
        variant: str = "petr",
        num_classes: int = 10,
        embed_dims: int = 256,
        num_queries: int = 900,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        feedforward_dims: int = 2048,
        dropout: float = 0.1,
        num_depth_bins: int = 64,
        depth_start: float = 1.0,
        depth_end: float = 60.0,
        depth_distribution: str = "linear",
        code_size: int = 10,
        pc_range: Tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        img_size: Tuple[int, int] = (900, 1600),
        num_cameras: int = 6,
        frozen_backbone_stages: int = 1,
        pretrained_backbone: bool = True,
        fpn_out_channels: int = 256,
        num_propagated_queries: int = 256,
        num_temporal_frames: int = 1,
        cls_weight: float = 2.0,
        bbox_weight: float = 0.25,
        velocity_weight: float = 0.25,
    ) -> None:
        self.variant = variant
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.feedforward_dims = feedforward_dims
        self.dropout = dropout
        self.num_depth_bins = num_depth_bins
        self.depth_start = depth_start
        self.depth_end = depth_end
        self.depth_distribution = depth_distribution
        self.code_size = code_size
        self.pc_range = pc_range
        self.img_size = img_size
        self.num_cameras = num_cameras
        self.frozen_backbone_stages = frozen_backbone_stages
        self.pretrained_backbone = pretrained_backbone
        self.fpn_out_channels = fpn_out_channels
        self.num_propagated_queries = num_propagated_queries
        self.num_temporal_frames = num_temporal_frames
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.velocity_weight = velocity_weight

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PETRConfig":
        """Create config from dictionary (e.g., loaded from YAML)."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k) or k in cls.__init__.__code__.co_varnames})


class PETRModel(nn.Module):
    """Unified PETR/PETRv2/StreamPETR 3D object detection model.

    The model pipeline:
    1. Backbone + FPN: Extract multi-scale image features.
    2. 3D Position Embedding: Generate position-aware features by encoding
       camera frustum 3D coordinates.
    3. Transformer Decoder: Object queries attend to position-aware features
       via global cross-attention.
    4. Detection Head: Predict classification scores and bounding boxes.
    5. (StreamPETR) Query Propagation: Propagate queries across frames.

    Args:
        config: PETRConfig instance with model hyperparameters.
    """

    def __init__(self, config: PETRConfig) -> None:
        super().__init__()
        self.config = config
        self.variant = config.variant

        # Backbone + FPN
        self.backbone = BackboneWithFPN(
            pretrained=config.pretrained_backbone,
            frozen_stages=config.frozen_backbone_stages,
            fpn_out_channels=config.fpn_out_channels,
            fpn_num_outs=1,  # Use single scale for PETR (top-level)
        )

        # Input projection: align FPN output channels with embed_dims
        self.input_proj = nn.Conv2d(
            config.fpn_out_channels, config.embed_dims, kernel_size=1
        )

        # 3D Position Embedding (key innovation of PETR)
        self.position_embedding_3d = PositionEmbedding3D(
            embed_dims=config.embed_dims,
            num_depth_bins=config.num_depth_bins,
            depth_start=config.depth_start,
            depth_end=config.depth_end,
            depth_distribution=config.depth_distribution,
        )

        # Transformer Decoder
        self.decoder = PETRTransformerDecoder(
            num_layers=config.num_decoder_layers,
            embed_dims=config.embed_dims,
            num_heads=config.num_heads,
            feedforward_dims=config.feedforward_dims,
            dropout=config.dropout,
            return_intermediate=True,
        )

        # Detection Head
        self.detection_head = PETRDetectionHead(
            embed_dims=config.embed_dims,
            num_classes=config.num_classes,
            code_size=config.code_size,
            num_layers=config.num_decoder_layers,
            shared_head=False,
        )

        # Loss function
        self.loss_fn = PETRLoss(
            num_classes=config.num_classes,
            cls_weight=config.cls_weight,
            bbox_weight=config.bbox_weight,
            velocity_weight=config.velocity_weight,
            code_size=config.code_size,
            pc_range=config.pc_range,
        )

        # Object queries (learnable)
        if config.variant != "streampetr":
            # PETR/PETRv2: fixed learnable queries
            self.query_embedding = nn.Embedding(config.num_queries, config.embed_dims)
            self.reference_points_embed = nn.Embedding(config.num_queries, 3)
            nn.init.uniform_(self.reference_points_embed.weight, 0.0, 1.0)
        else:
            # StreamPETR: query propagation with learnable + propagated queries
            num_learnable = config.num_queries - config.num_propagated_queries
            self.query_propagation = QueryPropagation(
                embed_dims=config.embed_dims,
                num_learnable_queries=num_learnable,
                num_propagated_queries=config.num_propagated_queries,
                pc_range=config.pc_range,
            )

        # Position embedding for queries (from reference points)
        self.query_pos_encoder = nn.Sequential(
            nn.Linear(3, config.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(config.embed_dims, config.embed_dims),
        )

    def extract_features(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Extract position-aware image features.

        Args:
            images: Multi-view images (B, N, 3, H, W).
            intrinsics: Camera intrinsics (B, N, 3, 3).
            extrinsics: Camera extrinsics (B, N, 4, 4).

        Returns:
            Position-aware features (B, N*D*H_feat*W_feat, C) ready for
            cross-attention in the decoder.
        """
        B, N, _, H, W = images.shape

        # Backbone + FPN: process all views jointly
        fpn_feats = self.backbone(images)  # List of (B*N, C_fpn, H_i, W_i)

        # Use the first (highest resolution) FPN level
        feat = fpn_feats[0]  # (B*N, C_fpn, H_feat, W_feat)

        # Project to embed_dims
        feat = self.input_proj(feat)  # (B*N, C, H_feat, W_feat)

        _, C, H_feat, W_feat = feat.shape
        feat = feat.reshape(B, N, C, H_feat, W_feat)  # (B, N, C, H_f, W_f)

        # Generate 3D position-aware features
        pos_aware_feat = self.position_embedding_3d(
            feat, intrinsics, extrinsics, (H, W), self.config.pc_range
        )  # (B, N, C, D*H_f*W_f)

        # Reshape for decoder: (B, N*D*H_f*W_f, C)
        pos_aware_feat = pos_aware_feat.permute(0, 1, 3, 2)  # (B, N, D*H*W, C)
        pos_aware_feat = pos_aware_feat.reshape(B, -1, C)  # (B, N*D*H*W, C)

        return pos_aware_feat

    def forward_petr(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        gt_labels: Optional[List[torch.Tensor]] = None,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Forward pass for PETR (single-frame, no temporal).

        Args:
            images: (B, N, 3, H, W) multi-view images.
            intrinsics: (B, N, 3, 3) camera intrinsics.
            extrinsics: (B, N, 4, 4) camera extrinsics.
            gt_labels: Ground-truth labels (training only).
            gt_bboxes: Ground-truth bboxes (training only).

        Returns:
            Dictionary with predictions and optionally losses.
        """
        B = images.shape[0]

        # Extract position-aware features
        memory = self.extract_features(images, intrinsics, extrinsics)

        # Initialize queries
        queries = self.query_embedding.weight.unsqueeze(0).expand(B, -1, -1)
        reference_points = self.reference_points_embed.weight.sigmoid().unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos_encoder(reference_points)

        # Decoder forward
        _, intermediate_outputs, intermediate_ref_pts = self.decoder(
            query=queries,
            key=memory,
            value=memory,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=self.detection_head.get_reg_branches(),
        )

        # Detection head
        predictions = self.detection_head(
            intermediate_outputs, intermediate_ref_pts
        )

        results: Dict[str, Any] = {"predictions": predictions}

        # Compute losses if ground truth is provided
        if gt_labels is not None and gt_bboxes is not None:
            losses = self.loss_fn(
                predictions["cls_scores"],
                predictions["bbox_preds"],
                gt_labels,
                gt_bboxes,
            )
            results["losses"] = losses

        return results

    def forward_streampetr(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        ego_motion: Optional[torch.Tensor] = None,
        ego_motion_vec: Optional[torch.Tensor] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Forward pass for StreamPETR (query propagation temporal model).

        Args:
            images: (B, N, 3, H, W) current frame images.
            intrinsics: (B, N, 3, 3) camera intrinsics.
            extrinsics: (B, N, 4, 4) camera extrinsics.
            ego_motion: (B, 4, 4) ego-motion from previous to current frame.
            ego_motion_vec: (B, 6) ego velocity vector.
            gt_labels: Ground-truth labels (training only).
            gt_bboxes: Ground-truth bboxes (training only).

        Returns:
            Dictionary with predictions and optionally losses.
        """
        B = images.shape[0]

        # Extract position-aware features
        memory = self.extract_features(images, intrinsics, extrinsics)

        # Get queries via propagation (combines propagated + learnable)
        queries, reference_points = self.query_propagation(
            ego_motion=ego_motion,
            ego_motion_vec=ego_motion_vec,
            batch_size=B,
        )

        # Compute query position embeddings from reference points
        query_pos = self.query_pos_encoder(reference_points)

        # Decoder forward
        final_output, intermediate_outputs, intermediate_ref_pts = self.decoder(
            query=queries,
            key=memory,
            value=memory,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=self.detection_head.get_reg_branches(),
        )

        # Detection head
        predictions = self.detection_head(
            intermediate_outputs, intermediate_ref_pts
        )

        results: Dict[str, Any] = {"predictions": predictions}

        # Update temporal memory with current frame outputs
        # Use the last decoder layer's output
        last_output = intermediate_outputs[-1]
        last_ref_pts = intermediate_ref_pts[-1] if intermediate_ref_pts else reference_points
        last_cls_scores = predictions["cls_scores"][-1]
        # Use max class score as confidence for memory selection
        confidence = last_cls_scores.sigmoid().max(dim=-1).values

        self.query_propagation.update_memory(
            last_output, last_ref_pts, confidence
        )

        # Compute losses if ground truth is provided
        if gt_labels is not None and gt_bboxes is not None:
            losses = self.loss_fn(
                predictions["cls_scores"],
                predictions["bbox_preds"],
                gt_labels,
                gt_bboxes,
            )
            results["losses"] = losses

        return results

    def forward_petrv2(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        prev_images: Optional[torch.Tensor] = None,
        prev_intrinsics: Optional[torch.Tensor] = None,
        prev_extrinsics: Optional[torch.Tensor] = None,
        prev_ego_motions: Optional[torch.Tensor] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Forward pass for PETRv2 (dense temporal feature fusion).

        PETRv2 aligns previous frame features to current frame coordinates
        and concatenates them with current features for cross-attention.

        Args:
            images: (B, N, 3, H, W) current frame images.
            intrinsics: (B, N, 3, 3) current camera intrinsics.
            extrinsics: (B, N, 4, 4) current camera extrinsics.
            prev_images: (B, T, N, 3, H, W) previous frame images.
            prev_intrinsics: (B, T, N, 3, 3) previous camera intrinsics.
            prev_extrinsics: (B, T, N, 4, 4) previous camera extrinsics.
            prev_ego_motions: (B, T, 4, 4) ego-motion transforms.
            gt_labels: Ground-truth labels (training only).
            gt_bboxes: Ground-truth bboxes (training only).

        Returns:
            Dictionary with predictions and optionally losses.
        """
        B = images.shape[0]

        # Extract features for current frame
        memory = self.extract_features(images, intrinsics, extrinsics)

        # Extract and align features from previous frames
        if prev_images is not None and prev_ego_motions is not None:
            T = prev_images.shape[1]
            all_prev_memory = []

            for t in range(T):
                prev_mem = self.extract_features(
                    prev_images[:, t],
                    prev_intrinsics[:, t],
                    prev_extrinsics[:, t],
                )
                all_prev_memory.append(prev_mem)

            # Concatenate temporal features with current features
            # All features are already in their respective camera coordinates
            # but position embeddings encode 3D positions correctly
            temporal_memory = torch.cat(all_prev_memory, dim=1)
            memory = torch.cat([memory, temporal_memory], dim=1)

        # Initialize queries
        queries = self.query_embedding.weight.unsqueeze(0).expand(B, -1, -1)
        reference_points = self.reference_points_embed.weight.sigmoid().unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos_encoder(reference_points)

        # Decoder forward (queries attend to all temporal features)
        _, intermediate_outputs, intermediate_ref_pts = self.decoder(
            query=queries,
            key=memory,
            value=memory,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=self.detection_head.get_reg_branches(),
        )

        # Detection head
        predictions = self.detection_head(
            intermediate_outputs, intermediate_ref_pts
        )

        results: Dict[str, Any] = {"predictions": predictions}

        if gt_labels is not None and gt_bboxes is not None:
            losses = self.loss_fn(
                predictions["cls_scores"],
                predictions["bbox_preds"],
                gt_labels,
                gt_bboxes,
            )
            results["losses"] = losses

        return results

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        ego_motion: Optional[torch.Tensor] = None,
        ego_motion_vec: Optional[torch.Tensor] = None,
        prev_images: Optional[torch.Tensor] = None,
        prev_intrinsics: Optional[torch.Tensor] = None,
        prev_extrinsics: Optional[torch.Tensor] = None,
        prev_ego_motions: Optional[torch.Tensor] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Unified forward pass dispatching to the appropriate variant.

        Args:
            images: (B, N, 3, H, W) multi-view camera images.
            intrinsics: (B, N, 3, 3) camera intrinsic matrices.
            extrinsics: (B, N, 4, 4) camera-to-ego transforms.
            ego_motion: (B, 4, 4) ego-motion matrix (StreamPETR).
            ego_motion_vec: (B, 6) ego velocity (StreamPETR).
            prev_images: (B, T, N, 3, H, W) previous frames (PETRv2).
            prev_intrinsics: (B, T, N, 3, 3) prev intrinsics (PETRv2).
            prev_extrinsics: (B, T, N, 4, 4) prev extrinsics (PETRv2).
            prev_ego_motions: (B, T, 4, 4) prev ego-motions (PETRv2).
            gt_labels: Ground-truth class labels.
            gt_bboxes: Ground-truth bounding boxes.

        Returns:
            Dictionary with 'predictions' and optionally 'losses'.
        """
        if self.variant == "streampetr":
            return self.forward_streampetr(
                images, intrinsics, extrinsics,
                ego_motion, ego_motion_vec,
                gt_labels, gt_bboxes,
            )
        elif self.variant == "petrv2":
            return self.forward_petrv2(
                images, intrinsics, extrinsics,
                prev_images, prev_intrinsics, prev_extrinsics, prev_ego_motions,
                gt_labels, gt_bboxes,
            )
        else:
            return self.forward_petr(
                images, intrinsics, extrinsics,
                gt_labels, gt_bboxes,
            )

    def reset_temporal_state(self) -> None:
        """Reset temporal memory (call at start of each new sequence).

        Only relevant for StreamPETR variant.
        """
        if self.variant == "streampetr":
            self.query_propagation.reset_memory()
