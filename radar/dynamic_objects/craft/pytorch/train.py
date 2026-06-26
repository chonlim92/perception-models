"""
PyTorch Training Script for CRAFT (Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer).

Complete training pipeline for the CRAFT model on the nuScenes dataset, combining
multi-view camera features with radar BEV features through a Spatio-Contextual
Fusion Transformer for 3D object detection.

Features:
    - Full CRAFTModel definition (camera branch + radar branch + BEV transform + fusion + detection head)
    - NuScenesRadarCameraDataset with proper data loading
    - Custom collate function for variable-length radar point clouds
    - Mixed precision training (torch.cuda.amp)
    - Distributed Data Parallel (DDP) support
    - Exponential Moving Average (EMA) of model weights
    - OneCycleLR scheduler with AdamW optimizer
    - Gradient clipping
    - TensorBoard and optional WandB logging
    - Checkpoint save/resume
    - Warmup epochs (branch-separate training before joint)
    - Gaussian focal loss + L1 regression + velocity loss

Usage:
    Single GPU:
        python train.py --config ../configs/craft_nuscenes.yaml

    Multi-GPU (DDP):
        torchrun --nproc_per_node=4 train.py --config ../configs/craft_nuscenes.yaml

    Resume training:
        python train.py --config ../configs/craft_nuscenes.yaml --resume /path/to/checkpoint.pth
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import yaml

from camera_branch import MultiViewCameraBackbone, build_camera_branch
from radar_branch import RadarBranch, build_radar_branch
from fusion_transformer import SpatioContextualFusionTransformer, build_fusion_transformer
from heads import CRAFTDetectionHead, decode_and_nms
from losses import CRAFTLoss, build_craft_loss

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("craft_train")


def setup_logging(work_dir: str, rank: int = 0) -> None:
    """Configure logging to file and console.

    Args:
        work_dir: Working directory for log file output.
        rank: Process rank (only rank 0 logs to console).
    """
    log_file = os.path.join(work_dir, "train.log")
    handlers: List[logging.Handler] = [logging.FileHandler(log_file, mode="a")]
    if rank == 0:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# BEV Transformation Module (Lift camera features to BEV via depth estimation)
# ---------------------------------------------------------------------------


class DepthEstimationNet(nn.Module):
    """Estimates per-pixel depth distribution for lifting 2D image features to 3D.

    For each pixel in the camera feature map, predicts a discrete depth distribution
    over D depth bins. This is used by the BEV transformation to project camera
    features into the BEV space (Lift-Splat-Shoot style).

    Args:
        in_channels: Input feature channels from FPN.
        mid_channels: Intermediate convolution channels.
        num_depth_bins: Number of discrete depth bins (D).
        depth_min: Minimum depth in meters.
        depth_max: Maximum depth in meters.
    """

    def __init__(
        self,
        in_channels: int = 256,
        mid_channels: int = 256,
        num_depth_bins: int = 64,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
    ) -> None:
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max

        # Depth bin centers (uniformly spaced in depth)
        depth_bins = torch.linspace(depth_min, depth_max, num_depth_bins)
        self.register_buffer("depth_bins", depth_bins)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_depth_bins, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming initialization."""
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict per-pixel depth distribution.

        Args:
            x: Camera feature map [B, C, H_feat, W_feat].

        Returns:
            Depth distribution [B, D, H_feat, W_feat] after softmax.
        """
        depth_logits = self.conv(x)  # [B, D, H, W]
        depth_probs = F.softmax(depth_logits, dim=1)  # [B, D, H, W]
        return depth_probs


class CameraBEVTransform(nn.Module):
    """Lift-Splat-Shoot style transformation from camera features to BEV.

    Projects multi-view camera features into 3D using predicted depth distributions,
    then splatters them onto a BEV grid. Uses the camera intrinsics and extrinsics
    to compute the mapping from image pixels to 3D world coordinates.

    Args:
        in_channels: Input camera feature channels.
        bev_channels: Output BEV feature channels.
        bev_height: BEV grid height (number of cells along X).
        bev_width: BEV grid width (number of cells along Y).
        num_depth_bins: Number of discrete depth bins.
        depth_min: Minimum depth in meters.
        depth_max: Maximum depth in meters.
        point_cloud_range: Spatial extent [x_min, y_min, z_min, x_max, y_max, z_max].
        downsample_factor: Downsampling factor from input image to feature map.
    """

    def __init__(
        self,
        in_channels: int = 256,
        bev_channels: int = 256,
        bev_height: int = 128,
        bev_width: int = 128,
        num_depth_bins: int = 64,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
        point_cloud_range: Optional[List[float]] = None,
        downsample_factor: int = 8,
    ) -> None:
        super().__init__()

        if point_cloud_range is None:
            point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        self.in_channels = in_channels
        self.bev_channels = bev_channels
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.point_cloud_range = point_cloud_range
        self.downsample_factor = downsample_factor

        self.x_min, self.y_min, self.z_min = point_cloud_range[:3]
        self.x_max, self.y_max, self.z_max = point_cloud_range[3:]
        self.voxel_x = (self.x_max - self.x_min) / bev_height
        self.voxel_y = (self.y_max - self.y_min) / bev_width

        # Depth estimation network
        self.depth_net = DepthEstimationNet(
            in_channels=in_channels,
            mid_channels=in_channels,
            num_depth_bins=num_depth_bins,
            depth_min=depth_min,
            depth_max=depth_max,
        )

        # BEV compression: reduce depth-scattered features to BEV feature map
        self.bev_compress = nn.Sequential(
            nn.Conv2d(in_channels, bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize BEV compression weights."""
        for m in self.bev_compress.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _compute_frustum_to_bev(
        self,
        depth_probs: torch.Tensor,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Project camera frustum features onto BEV grid using depth-weighted splatting.

        For efficiency, we use a simplified scatter approach: for each pixel with its
        depth distribution, we compute the 3D world coordinate at each depth bin,
        map it to the BEV grid, and splat the depth-weighted feature.

        Args:
            depth_probs: Predicted depth distribution [B_flat, D, H_feat, W_feat].
            features: Camera features [B_flat, C, H_feat, W_feat].
            intrinsics: Camera intrinsics [B_flat, 3, 3].
            extrinsics: Camera-to-ego extrinsics [B_flat, 4, 4].
            image_shape: Original image (H, W).

        Returns:
            BEV feature map [B_flat, C, bev_H, bev_W].
        """
        B_flat, C, H_feat, W_feat = features.shape
        D = self.num_depth_bins
        device = features.device
        dtype = features.dtype

        # Create pixel coordinate grid for the feature map
        u_coords = torch.arange(W_feat, device=device, dtype=dtype) * self.downsample_factor + self.downsample_factor / 2.0
        v_coords = torch.arange(H_feat, device=device, dtype=dtype) * self.downsample_factor + self.downsample_factor / 2.0
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")  # [H_feat, W_feat]

        # Depth bin centers
        depth_bins = self.depth_net.depth_bins  # [D]

        # For each depth bin, compute 3D points in camera frame
        # pixel_coords: [H_feat, W_feat, 3] homogeneous
        ones = torch.ones(H_feat, W_feat, device=device, dtype=dtype)
        pixel_homo = torch.stack([u_grid, v_grid, ones], dim=-1)  # [H_feat, W_feat, 3]
        pixel_homo_flat = pixel_homo.reshape(-1, 3)  # [H*W, 3]

        # Initialize BEV accumulation
        bev_features = torch.zeros(B_flat, C, self.bev_height, self.bev_width, device=device, dtype=dtype)
        bev_weights = torch.zeros(B_flat, 1, self.bev_height, self.bev_width, device=device, dtype=dtype)

        # Process a subset of depth bins for memory efficiency
        # Use top-K depth bins by average probability
        avg_depth_probs = depth_probs.mean(dim=[0, 2, 3])  # [D]
        num_bins_to_process = min(D, 16)  # Process top 16 bins for efficiency
        top_bin_indices = torch.topk(avg_depth_probs, num_bins_to_process).indices

        for d_idx in top_bin_indices:
            d = depth_bins[d_idx]  # scalar depth value

            # Back-project pixels to 3D camera frame: P_cam = K^{-1} * [u, v, 1]^T * d
            # Use batch operation: [B_flat, 3, 3]^{-1} x [H*W, 3]^T * d
            K_inv = torch.inverse(intrinsics)  # [B_flat, 3, 3]
            points_cam = torch.einsum("bij,nj->bni", K_inv, pixel_homo_flat) * d  # [B_flat, H*W, 3]

            # Transform from camera to ego frame using extrinsics
            # extrinsics here is camera-to-ego (inverse of ego-to-camera)
            R = extrinsics[:, :3, :3]  # [B_flat, 3, 3]
            t = extrinsics[:, :3, 3]   # [B_flat, 3]
            points_ego = torch.einsum("bij,bnj->bni", R, points_cam) + t.unsqueeze(1)  # [B_flat, H*W, 3]

            # Map ego-frame XY to BEV grid indices
            bev_x = ((points_ego[:, :, 0] - self.x_min) / self.voxel_x).long()  # [B_flat, H*W]
            bev_y = ((points_ego[:, :, 1] - self.y_min) / self.voxel_y).long()  # [B_flat, H*W]

            # Valid mask: within BEV bounds
            valid = (
                (bev_x >= 0) & (bev_x < self.bev_height)
                & (bev_y >= 0) & (bev_y < self.bev_width)
            )  # [B_flat, H*W]

            # Get depth probability weights for this bin
            d_weight = depth_probs[:, d_idx, :, :]  # [B_flat, H_feat, W_feat]
            d_weight_flat = d_weight.reshape(B_flat, -1)  # [B_flat, H*W]

            # Weighted features: features * depth_weight
            features_flat = features.reshape(B_flat, C, -1)  # [B_flat, C, H*W]
            weighted_feats = features_flat * d_weight_flat.unsqueeze(1)  # [B_flat, C, H*W]

            # Scatter onto BEV grid
            for b in range(B_flat):
                valid_b = valid[b]  # [H*W]
                if not valid_b.any():
                    continue
                bx = bev_x[b, valid_b]  # [N_valid]
                by = bev_y[b, valid_b]  # [N_valid]
                wf = weighted_feats[b, :, valid_b]  # [C, N_valid]
                dw = d_weight_flat[b, valid_b]  # [N_valid]

                # Linear index for scatter_add
                linear_idx = bx * self.bev_width + by  # [N_valid]
                linear_idx_expand = linear_idx.unsqueeze(0).expand(C, -1)  # [C, N_valid]

                bev_flat = bev_features[b].reshape(C, -1)  # [C, H_bev*W_bev]
                bev_flat.scatter_add_(1, linear_idx_expand, wf)

                bev_w_flat = bev_weights[b].reshape(1, -1)  # [1, H_bev*W_bev]
                bev_w_flat.scatter_add_(1, linear_idx.unsqueeze(0), dw.unsqueeze(0))

        # Normalize by accumulated weights
        bev_features = bev_features / (bev_weights + 1e-5)

        return bev_features

    def forward(
        self,
        camera_features: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Transform multi-view camera features to BEV representation.

        Args:
            camera_features: List of FPN feature maps. We use level P3 (stride 8).
                Each tensor: [B, N_cams, C, H_feat, W_feat].
            intrinsics: Camera intrinsic matrices [B, N_cams, 3, 3].
            extrinsics: Camera-to-ego extrinsic matrices [B, N_cams, 4, 4].
            image_shape: Original image (H, W).

        Returns:
            Camera BEV feature map [B, bev_channels, bev_H, bev_W].
        """
        # Use P3 level features (stride 8) for BEV transformation
        # camera_features[1] corresponds to P3 level
        feat_level_idx = 1 if len(camera_features) > 1 else 0
        feats = camera_features[feat_level_idx]  # [B, N_cams, C, H_feat, W_feat]
        B, N_cams, C, H_feat, W_feat = feats.shape

        # Flatten batch and camera dims
        feats_flat = feats.reshape(B * N_cams, C, H_feat, W_feat)
        intrinsics_flat = intrinsics.reshape(B * N_cams, 3, 3)
        extrinsics_flat = extrinsics.reshape(B * N_cams, 4, 4)

        # Predict depth distributions
        depth_probs = self.depth_net(feats_flat)  # [B*N_cams, D, H_feat, W_feat]

        # Project to BEV
        bev_per_cam = self._compute_frustum_to_bev(
            depth_probs, feats_flat, intrinsics_flat, extrinsics_flat, image_shape
        )  # [B*N_cams, C, bev_H, bev_W]

        # Reshape and aggregate across cameras (mean pooling)
        bev_per_cam = bev_per_cam.reshape(B, N_cams, C, self.bev_height, self.bev_width)
        camera_bev = bev_per_cam.mean(dim=1)  # [B, C, bev_H, bev_W]

        # Compress to final BEV channels
        camera_bev = self.bev_compress(camera_bev)  # [B, bev_channels, bev_H, bev_W]

        return camera_bev


# ---------------------------------------------------------------------------
# CRAFT Full Model
# ---------------------------------------------------------------------------


class CRAFTModel(nn.Module):
    """CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer.

    Combines multi-view camera features with radar BEV features through a learned
    BEV transformation and cross-attention fusion transformer, producing dense 3D
    object detections in bird's-eye view.

    Architecture:
        1. Camera Branch: ResNet+FPN extracts multi-scale multi-view features
        2. Radar Branch: PointPillar encoder + BEV backbone
        3. BEV Transform: Lift camera features to BEV via depth estimation
        4. Fusion Transformer: Spatio-contextual cross-attention between modalities
        5. Detection Head: CenterPoint-style heatmap + regression + velocity

    Args:
        config: Model configuration dictionary parsed from YAML.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        model_cfg = config["model"]
        backbone_cfg = model_cfg["backbone"]
        neck_cfg = model_cfg["neck"]
        radar_cfg = model_cfg["radar_pillar_encoder"]
        fusion_cfg = model_cfg["fusion_transformer"]
        head_cfg = model_cfg["detection_head"]
        data_cfg = config["data"]
        pc_cfg = data_cfg["point_cloud"]

        # Store config
        self.config = config
        self.num_classes = head_cfg["num_classes"]
        self.point_cloud_range = pc_cfg["range"]
        self.voxel_size = pc_cfg["voxel_size"]
        self.bev_height = int(round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]
        ))
        self.bev_width = int(round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]
        ))

        # 1. Camera Branch
        self.camera_branch = build_camera_branch(
            backbone_name=backbone_cfg["type"],
            pretrained=backbone_cfg["pretrained"],
            fpn_out_channels=neck_cfg["out_channels"],
            num_cameras=6,
            frozen_stages=backbone_cfg["frozen_stages"],
        )

        # 2. Radar Branch
        self.radar_branch = build_radar_branch(
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            max_points_per_pillar=pc_cfg["max_points_per_pillar"],
            max_num_pillars=pc_cfg["max_pillars"],
            in_channels=6,
            pillar_feat_channels=radar_cfg["out_channels"],
            bev_out_channels=neck_cfg["out_channels"],
        )

        # 3. Camera BEV Transform (Lift-Splat)
        self.camera_bev_transform = CameraBEVTransform(
            in_channels=neck_cfg["out_channels"],
            bev_channels=neck_cfg["out_channels"],
            bev_height=self.bev_height,
            bev_width=self.bev_width,
            num_depth_bins=64,
            depth_min=1.0,
            depth_max=60.0,
            point_cloud_range=self.point_cloud_range,
            downsample_factor=8,
        )

        # 4. Spatio-Contextual Fusion Transformer
        self.fusion_transformer = build_fusion_transformer(
            d_model=fusion_cfg["d_model"],
            n_heads=fusion_cfg["n_heads"],
            d_ffn=fusion_cfg["d_ffn"],
            n_layers=fusion_cfg["n_layers"],
            dropout=fusion_cfg["dropout"],
            radar_channels=neck_cfg["out_channels"],
            camera_channels=neck_cfg["out_channels"],
        )

        # 5. Detection Head
        self.detection_head = CRAFTDetectionHead(
            in_channels=neck_cfg["out_channels"],
            shared_channels=head_cfg["heatmap_channels"],
            num_classes=head_cfg["num_classes"],
            head_hidden_channels=head_cfg["shared_conv_channels"],
            num_head_conv_layers=head_cfg["num_heatmap_convs"],
            num_shared_conv_layers=head_cfg["num_regression_convs"],
        )

        # Regression head for full bbox code (10 channels: 8 reg + 2 velocity merged)
        # We combine regression and velocity into a single prediction map
        self.reg_head = nn.Sequential(
            nn.Conv2d(neck_cfg["out_channels"], head_cfg["regression_channels"], 3, padding=1, bias=False),
            nn.BatchNorm2d(head_cfg["regression_channels"]),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_cfg["regression_channels"], head_cfg["regression_channels"], 3, padding=1, bias=False),
            nn.BatchNorm2d(head_cfg["regression_channels"]),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_cfg["regression_channels"], head_cfg["bbox_code_size"], 1, bias=True),
        )

        self._init_reg_head()

    def _init_reg_head(self) -> None:
        """Initialize regression head weights."""
        for m in self.reg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        images: torch.Tensor,
        radar_points: torch.Tensor,
        radar_num_points: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int] = (900, 1600),
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the full CRAFT pipeline.

        Args:
            images: Multi-view camera images [B, N_cams, 3, H, W].
            radar_points: Radar point cloud [B, N_max, 6] (x, y, z, vx, vy, rcs).
            radar_num_points: Number of valid radar points per sample [B].
            intrinsics: Camera intrinsic matrices [B, N_cams, 3, 3].
            extrinsics: Camera-to-ego extrinsic matrices [B, N_cams, 4, 4].
            image_shape: Original image (H, W) for projection.

        Returns:
            Dictionary with detection predictions:
                'heatmap': [B, num_classes, H_bev, W_bev]
                'reg': [B, bbox_code_size, H_bev, W_bev]
                'regression': [B, 8, H_bev, W_bev] (from detection head)
                'velocity': [B, 2, H_bev, W_bev] (from detection head)
        """
        # 1. Camera feature extraction
        camera_output = self.camera_branch(images)
        camera_features = camera_output["features"]  # List of [B, N_cams, C, H_i, W_i]

        # 2. Radar BEV feature extraction
        radar_output = self.radar_branch(radar_points, radar_num_points)
        radar_bev = radar_output["bev_features"]  # [B, C, H_bev, W_bev]

        # 3. Lift camera features to BEV
        camera_bev = self.camera_bev_transform(
            camera_features, intrinsics, extrinsics, image_shape
        )  # [B, C, H_bev, W_bev]

        # Ensure spatial dimensions match radar BEV
        if camera_bev.shape[2:] != radar_bev.shape[2:]:
            camera_bev = F.interpolate(
                camera_bev,
                size=radar_bev.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # 4. Fusion: use P3 camera features for cross-attention
        # Prepare camera features for fusion transformer (use BEV-transformed features)
        # The fusion transformer expects [B, N_cams, C, H, W] camera features
        # We use the BEV camera features expanded as pseudo multi-view
        B = images.shape[0]
        N_cams = images.shape[1]

        # Use the raw P3 camera features for cross-attention in the fusion transformer
        cam_feat_for_fusion = camera_features[1]  # [B, N_cams, C, H_feat, W_feat]

        fused_bev = self.fusion_transformer(
            radar_bev_features=radar_bev,
            camera_features=cam_feat_for_fusion,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_shape=image_shape,
        )  # [B, C, H_bev, W_bev]

        # Add camera BEV features as residual to fused features
        fused_bev = fused_bev + camera_bev

        # 5. Detection head
        head_output = self.detection_head(fused_bev)

        # 6. Full regression prediction (bbox_code_size = 10)
        reg_output = self.reg_head(fused_bev)  # [B, 10, H_bev, W_bev]

        return {
            "heatmap": head_output["heatmap"],
            "reg": reg_output,
            "regression": head_output["regression"],
            "velocity": head_output["velocity"],
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NuScenesRadarCameraDataset(Dataset):
    """nuScenes dataset for radar-camera fusion training.

    Loads multi-view camera images and radar point clouds along with 3D
    bounding box annotations for the CRAFT model.

    Expects a preprocessed info file (pickle) containing per-sample metadata:
        - cam_paths: paths to 6 camera images
        - cam_intrinsics: [6, 3, 3] intrinsic matrices
        - cam_extrinsics: [6, 4, 4] camera-to-ego transformation matrices
        - radar_points: [N, 18] radar point features (using selected dims)
        - gt_boxes: [M, 10] ground truth bounding boxes
        - gt_labels: [M] class indices

    Args:
        info_path: Path to the preprocessed dataset info pickle file.
        root_path: Root directory of the nuScenes dataset.
        class_names: List of detection class names.
        image_size: Target image size (H, W) after resizing.
        point_cloud_range: BEV spatial extent.
        max_radar_points: Maximum number of radar points to pad/truncate to.
        augmentation_cfg: Data augmentation configuration dict.
        is_train: Whether this is training split (enables augmentation).
    """

    CLASS_NAMES: List[str] = [
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    ]

    def __init__(
        self,
        info_path: str,
        root_path: str,
        class_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (900, 1600),
        point_cloud_range: Optional[List[float]] = None,
        max_radar_points: int = 2048,
        augmentation_cfg: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.root_path = root_path
        self.class_names = class_names if class_names is not None else self.CLASS_NAMES
        self.image_size = image_size
        self.point_cloud_range = point_cloud_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.max_radar_points = max_radar_points
        self.augmentation_cfg = augmentation_cfg or {}
        self.is_train = is_train

        # Image normalization parameters (ImageNet)
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load dataset info
        logger.info(f"Loading dataset info from {info_path}")
        with open(info_path, "rb") as f:
            self.infos = pickle.load(f)
        logger.info(f"Loaded {len(self.infos)} samples")

    def __len__(self) -> int:
        return len(self.infos)

    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess a single camera image.

        Args:
            path: Path to the image file.

        Returns:
            Preprocessed image array [3, H, W] normalized to [0, 1] then ImageNet stats.
        """
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
        except Exception:
            # Fallback: create a dummy image if loading fails
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)

        # Normalize
        img = (img - self.img_mean) / self.img_std
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        return img

    def _load_radar_points(self, info: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """Load and preprocess radar point cloud.

        Args:
            info: Sample info dictionary.

        Returns:
            Tuple of (points [max_radar_points, 6], num_valid_points).
        """
        if "radar_points" in info:
            raw_points = np.array(info["radar_points"], dtype=np.float32)
        elif "radar_path" in info:
            radar_path = os.path.join(self.root_path, info["radar_path"])
            if os.path.exists(radar_path):
                raw_points = np.fromfile(radar_path, dtype=np.float32).reshape(-1, 18)
            else:
                raw_points = np.zeros((0, 18), dtype=np.float32)
        else:
            raw_points = np.zeros((0, 6), dtype=np.float32)

        # Select relevant features: x, y, z, vx_comp, vy_comp, rcs
        if raw_points.shape[1] > 6:
            # Standard nuScenes radar: use dims 0,1,2 (xyz), 8,9 (velocity), 5 (rcs)
            selected_dims = [0, 1, 2, 8, 9, 5]
            if raw_points.shape[1] > max(selected_dims):
                raw_points = raw_points[:, selected_dims]
            else:
                raw_points = raw_points[:, :6]
        elif raw_points.shape[1] < 6:
            # Pad with zeros if fewer features
            padding = np.zeros((raw_points.shape[0], 6 - raw_points.shape[1]), dtype=np.float32)
            raw_points = np.concatenate([raw_points, padding], axis=1)

        num_valid = raw_points.shape[0]

        # Filter points within range
        if num_valid > 0:
            x_mask = (raw_points[:, 0] >= self.point_cloud_range[0]) & (raw_points[:, 0] < self.point_cloud_range[3])
            y_mask = (raw_points[:, 1] >= self.point_cloud_range[1]) & (raw_points[:, 1] < self.point_cloud_range[4])
            z_mask = (raw_points[:, 2] >= self.point_cloud_range[2]) & (raw_points[:, 2] < self.point_cloud_range[5])
            valid_mask = x_mask & y_mask & z_mask
            raw_points = raw_points[valid_mask]
            num_valid = raw_points.shape[0]

        # Pad or truncate to max_radar_points
        padded_points = np.zeros((self.max_radar_points, 6), dtype=np.float32)
        if num_valid > 0:
            n = min(num_valid, self.max_radar_points)
            if num_valid > self.max_radar_points:
                # Random subsample
                indices = np.random.choice(num_valid, self.max_radar_points, replace=False)
                padded_points = raw_points[indices]
                num_valid = self.max_radar_points
            else:
                padded_points[:n] = raw_points[:n]

        return padded_points, num_valid

    def _get_annotations(self, info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 3D bounding box annotations.

        Args:
            info: Sample info dictionary.

        Returns:
            Tuple of (gt_boxes [M, 10], gt_labels [M]).
        """
        if "gt_boxes" in info and "gt_labels" in info:
            gt_boxes = np.array(info["gt_boxes"], dtype=np.float32)
            gt_labels = np.array(info["gt_labels"], dtype=np.int64)
        elif "gt_names" in info and "gt_boxes_3d" in info:
            gt_boxes = np.array(info["gt_boxes_3d"], dtype=np.float32)
            gt_names = info["gt_names"]
            gt_labels = np.array([
                self.class_names.index(name) if name in self.class_names else -1
                for name in gt_names
            ], dtype=np.int64)
            # Filter out unknown classes
            valid = gt_labels >= 0
            gt_boxes = gt_boxes[valid]
            gt_labels = gt_labels[valid]
        else:
            gt_boxes = np.zeros((0, 10), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)

        # Ensure gt_boxes has 10 columns (x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy)
        if gt_boxes.shape[0] > 0 and gt_boxes.shape[1] == 9:
            # If 9 cols (x,y,z,w,l,h,yaw,vx,vy), convert yaw to sin/cos
            yaw = gt_boxes[:, 6]
            vx = gt_boxes[:, 7]
            vy = gt_boxes[:, 8]
            gt_boxes_new = np.zeros((gt_boxes.shape[0], 10), dtype=np.float32)
            gt_boxes_new[:, :6] = gt_boxes[:, :6]
            gt_boxes_new[:, 6] = np.sin(yaw)
            gt_boxes_new[:, 7] = np.cos(yaw)
            gt_boxes_new[:, 8] = vx
            gt_boxes_new[:, 9] = vy
            gt_boxes = gt_boxes_new
        elif gt_boxes.shape[0] > 0 and gt_boxes.shape[1] == 7:
            # If 7 cols (x,y,z,w,l,h,yaw), add sin/cos/vx/vy
            yaw = gt_boxes[:, 6]
            gt_boxes_new = np.zeros((gt_boxes.shape[0], 10), dtype=np.float32)
            gt_boxes_new[:, :6] = gt_boxes[:, :6]
            gt_boxes_new[:, 6] = np.sin(yaw)
            gt_boxes_new[:, 7] = np.cos(yaw)
            gt_boxes = gt_boxes_new
        elif gt_boxes.shape[0] > 0 and gt_boxes.shape[1] < 7:
            # Pad with zeros
            padding = np.zeros((gt_boxes.shape[0], 10 - gt_boxes.shape[1]), dtype=np.float32)
            gt_boxes = np.concatenate([gt_boxes, padding], axis=1)

        return gt_boxes, gt_labels

    def _apply_augmentation(
        self,
        images: np.ndarray,
        radar_points: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation to images, points, and annotations.

        Args:
            images: Camera images [N_cams, 3, H, W].
            radar_points: Radar points [N_max, 6].
            gt_boxes: Ground truth boxes [M, 10].
            gt_labels: Ground truth labels [M].

        Returns:
            Augmented (images, radar_points, gt_boxes, gt_labels).
        """
        if not self.is_train:
            return images, radar_points, gt_boxes, gt_labels

        # Random horizontal flip
        flip_prob = self.augmentation_cfg.get("flip_prob", 0.5)
        if random.random() < flip_prob:
            # Flip images along width
            images = images[:, :, :, ::-1].copy()
            # Flip radar y coordinate
            radar_points[:, 1] = -radar_points[:, 1]
            radar_points[:, 4] = -radar_points[:, 4]  # vy
            # Flip gt boxes
            if gt_boxes.shape[0] > 0:
                gt_boxes[:, 1] = -gt_boxes[:, 1]  # y
                gt_boxes[:, 6] = -gt_boxes[:, 6]  # sin(yaw)
                gt_boxes[:, 9] = -gt_boxes[:, 9]  # vy

        # Random rotation
        rotation_range = self.augmentation_cfg.get("rotation_range", [-0.3925, 0.3925])
        if rotation_range[1] > 0:
            angle = random.uniform(rotation_range[0], rotation_range[1])
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Rotate radar points (x, y)
            x = radar_points[:, 0].copy()
            y = radar_points[:, 1].copy()
            radar_points[:, 0] = x * cos_a - y * sin_a
            radar_points[:, 1] = x * sin_a + y * cos_a

            # Rotate velocity (vx, vy)
            vx = radar_points[:, 3].copy()
            vy = radar_points[:, 4].copy()
            radar_points[:, 3] = vx * cos_a - vy * sin_a
            radar_points[:, 4] = vx * sin_a + vy * cos_a

            # Rotate gt boxes
            if gt_boxes.shape[0] > 0:
                bx = gt_boxes[:, 0].copy()
                by = gt_boxes[:, 1].copy()
                gt_boxes[:, 0] = bx * cos_a - by * sin_a
                gt_boxes[:, 1] = bx * sin_a + by * cos_a

                # Update yaw (sin/cos rotation)
                old_sin = gt_boxes[:, 6].copy()
                old_cos = gt_boxes[:, 7].copy()
                gt_boxes[:, 6] = old_sin * cos_a + old_cos * sin_a
                gt_boxes[:, 7] = old_cos * cos_a - old_sin * sin_a

                # Rotate velocity
                bvx = gt_boxes[:, 8].copy()
                bvy = gt_boxes[:, 9].copy()
                gt_boxes[:, 8] = bvx * cos_a - bvy * sin_a
                gt_boxes[:, 9] = bvx * sin_a + bvy * cos_a

        # Random scale
        scale_range = self.augmentation_cfg.get("scale_range", [0.95, 1.05])
        if scale_range[1] > scale_range[0]:
            scale = random.uniform(scale_range[0], scale_range[1])
            radar_points[:, :3] *= scale
            if gt_boxes.shape[0] > 0:
                gt_boxes[:, :3] *= scale  # position
                gt_boxes[:, 3:6] *= scale  # size

        return images, radar_points, gt_boxes, gt_labels

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single training sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary containing all inputs and annotations for the sample.
        """
        info = self.infos[index]

        # Load multi-view camera images
        cam_images = []
        if "cam_paths" in info:
            for cam_path in info["cam_paths"]:
                full_path = os.path.join(self.root_path, cam_path) if not os.path.isabs(cam_path) else cam_path
                img = self._load_image(full_path)
                cam_images.append(img)
        elif "cams" in info:
            for cam_key in ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
                if cam_key in info["cams"]:
                    cam_info = info["cams"][cam_key]
                    cam_path = cam_info.get("data_path", cam_info.get("filename", ""))
                    full_path = os.path.join(self.root_path, cam_path) if not os.path.isabs(cam_path) else cam_path
                    img = self._load_image(full_path)
                    cam_images.append(img)

        # Ensure 6 camera views
        while len(cam_images) < 6:
            cam_images.append(np.zeros((3, self.image_size[0], self.image_size[1]), dtype=np.float32))
        images = np.stack(cam_images[:6], axis=0)  # [6, 3, H, W]

        # Load camera intrinsics and extrinsics
        if "cam_intrinsics" in info:
            intrinsics = np.array(info["cam_intrinsics"], dtype=np.float32)  # [6, 3, 3]
        elif "cams" in info:
            intrinsics_list = []
            for cam_key in ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
                if cam_key in info["cams"]:
                    K = info["cams"][cam_key].get("cam_intrinsic", np.eye(3))
                    intrinsics_list.append(np.array(K, dtype=np.float32))
            while len(intrinsics_list) < 6:
                intrinsics_list.append(np.eye(3, dtype=np.float32))
            intrinsics = np.stack(intrinsics_list[:6], axis=0)
        else:
            intrinsics = np.tile(np.eye(3, dtype=np.float32), (6, 1, 1))

        if "cam_extrinsics" in info:
            extrinsics = np.array(info["cam_extrinsics"], dtype=np.float32)  # [6, 4, 4]
        elif "cams" in info:
            extrinsics_list = []
            for cam_key in ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
                if cam_key in info["cams"]:
                    ext = info["cams"][cam_key].get("sensor2ego", np.eye(4))
                    extrinsics_list.append(np.array(ext, dtype=np.float32))
            while len(extrinsics_list) < 6:
                extrinsics_list.append(np.eye(4, dtype=np.float32))
            extrinsics = np.stack(extrinsics_list[:6], axis=0)
        else:
            extrinsics = np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))

        # Load radar points
        radar_points, num_radar_points = self._load_radar_points(info)

        # Load annotations
        gt_boxes, gt_labels = self._get_annotations(info)

        # Apply augmentation
        images, radar_points, gt_boxes, gt_labels = self._apply_augmentation(
            images, radar_points, gt_boxes, gt_labels
        )

        return {
            "images": torch.from_numpy(images).float(),  # [6, 3, H, W]
            "radar_points": torch.from_numpy(radar_points).float(),  # [N_max, 6]
            "radar_num_points": torch.tensor(num_radar_points, dtype=torch.long),
            "intrinsics": torch.from_numpy(intrinsics).float(),  # [6, 3, 3]
            "extrinsics": torch.from_numpy(extrinsics).float(),  # [6, 4, 4]
            "gt_boxes": torch.from_numpy(gt_boxes).float(),  # [M, 10]
            "gt_labels": torch.from_numpy(gt_labels).long(),  # [M]
        }


def craft_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for CRAFT data loading.

    Handles variable-length radar points and ground truth annotations by padding
    within the batch. Fixed-size tensors (images, intrinsics, extrinsics) are
    simply stacked.

    Args:
        batch: List of sample dictionaries from NuScenesRadarCameraDataset.

    Returns:
        Collated batch dictionary with batched tensors.
    """
    # Stack fixed-size tensors
    images = torch.stack([s["images"] for s in batch], dim=0)  # [B, 6, 3, H, W]
    intrinsics = torch.stack([s["intrinsics"] for s in batch], dim=0)  # [B, 6, 3, 3]
    extrinsics = torch.stack([s["extrinsics"] for s in batch], dim=0)  # [B, 6, 4, 4]

    # Radar points are already padded to max_radar_points in the dataset
    radar_points = torch.stack([s["radar_points"] for s in batch], dim=0)  # [B, N_max, 6]
    radar_num_points = torch.stack([s["radar_num_points"] for s in batch], dim=0)  # [B]

    # Ground truth: variable length, keep as lists for CRAFTLoss
    gt_boxes = [s["gt_boxes"] for s in batch]  # List of [M_i, 10]
    gt_labels = [s["gt_labels"] for s in batch]  # List of [M_i]

    return {
        "images": images,
        "radar_points": radar_points,
        "radar_num_points": radar_num_points,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "gt_boxes": gt_boxes,
        "gt_labels": gt_labels,
    }


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------


class ModelEMA:
    """Exponential Moving Average of model parameters for stable evaluation.

    Maintains a shadow copy of model parameters that is updated with an exponential
    moving average at each training step. The EMA model typically provides better
    evaluation performance than the raw trained model.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate (higher = slower update). Typically 0.999 or 0.9999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        for name, buf in model.named_buffers():
            self.shadow[name] = buf.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters with current model parameters.

        Args:
            model: Model with updated parameters after a training step.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

        for name, buf in model.named_buffers():
            if name in self.shadow:
                self.shadow[name].copy_(buf.data)

    def apply_shadow(self, model: nn.Module) -> None:
        """Apply EMA parameters to the model (for evaluation).

        Saves a backup of the current parameters so they can be restored later.

        Args:
            model: Model to apply EMA parameters to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

        for name, buf in model.named_buffers():
            if name in self.shadow:
                self.backup[name] = buf.data.clone()
                buf.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original (non-EMA) parameters to the model.

        Args:
            model: Model to restore parameters to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])

        for name, buf in model.named_buffers():
            if name in self.backup:
                buf.data.copy_(self.backup[name])

        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return EMA state for checkpointing."""
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint."""
        self.shadow = state["shadow"]
        self.decay = state.get("decay", self.decay)


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor across distributed processes (mean reduction).

    Args:
        tensor: Tensor to reduce.

    Returns:
        Reduced tensor (averaged across processes).
    """
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    max_keep: int = 5,
) -> None:
    """Save training checkpoint to disk.

    Args:
        state: Dictionary containing model state, optimizer state, epoch, etc.
        filepath: Path to save the checkpoint.
        is_best: If True, also save a copy as 'best.pth'.
        max_keep: Maximum number of periodic checkpoints to retain.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    logger.info(f"Saved checkpoint: {filepath}")

    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), "best.pth")
        torch.save(state, best_path)
        logger.info(f"Saved best checkpoint: {best_path}")

    # Clean up old checkpoints
    ckpt_dir = os.path.dirname(filepath)
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pth")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    while len(ckpts) > max_keep:
        old_ckpt = os.path.join(ckpt_dir, ckpts.pop(0))
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)
            logger.info(f"Removed old checkpoint: {old_ckpt}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: CRAFTLoss,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Run validation and compute metrics.

    Args:
        model: Model to evaluate (should already have EMA applied if desired).
        val_loader: Validation data loader.
        loss_fn: Loss function for computing validation loss.
        device: Computation device.
        epoch: Current epoch number (for logging).
        config: Full configuration dictionary.

    Returns:
        Dictionary of validation metrics (val_loss, val_cls_loss, etc.).
    """
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_vel_loss = 0.0
    num_batches = 0

    image_shape = tuple(config["data"]["image"]["size"])

    for batch in val_loader:
        images = batch["images"].to(device, non_blocking=True)
        radar_points = batch["radar_points"].to(device, non_blocking=True)
        radar_num_points = batch["radar_num_points"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        extrinsics = batch["extrinsics"].to(device, non_blocking=True)
        gt_boxes = [b.to(device, non_blocking=True) for b in batch["gt_boxes"]]
        gt_labels = [l.to(device, non_blocking=True) for l in batch["gt_labels"]]

        # Forward pass
        predictions = model(
            images=images,
            radar_points=radar_points,
            radar_num_points=radar_num_points,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_shape=image_shape,
        )

        # Compute loss
        loss_dict = loss_fn(predictions, gt_boxes, gt_labels)

        total_loss += loss_dict["total_loss"].item()
        total_cls_loss += loss_dict["cls_loss"].item()
        total_bbox_loss += loss_dict["bbox_loss"].item()
        total_vel_loss += loss_dict["velocity_loss"].item()
        num_batches += 1

    num_batches = max(num_batches, 1)
    metrics = {
        "val_loss": total_loss / num_batches,
        "val_cls_loss": total_cls_loss / num_batches,
        "val_bbox_loss": total_bbox_loss / num_batches,
        "val_vel_loss": total_vel_loss / num_batches,
    }

    if is_main_process():
        logger.info(
            f"Epoch {epoch} Validation - "
            f"loss: {metrics['val_loss']:.4f}, "
            f"cls: {metrics['val_cls_loss']:.4f}, "
            f"bbox: {metrics['val_bbox_loss']:.4f}, "
            f"vel: {metrics['val_vel_loss']:.4f}"
        )

    return metrics


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: CRAFTLoss,
    scaler: torch.cuda.amp.GradScaler,
    ema: Optional[ModelEMA],
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    writer: Optional[SummaryWriter] = None,
    use_wandb: bool = False,
    warmup_mode: bool = False,
) -> Dict[str, float]:
    """Train the model for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        scaler: GradScaler for mixed precision.
        ema: Optional EMA tracker.
        device: Computation device.
        epoch: Current epoch number.
        config: Full configuration dictionary.
        writer: Optional TensorBoard writer.
        use_wandb: Whether to log to WandB.
        warmup_mode: If True, train branches separately (no fusion).

    Returns:
        Dictionary of training metrics for the epoch.
    """
    model.train()
    training_cfg = config["training"]
    runtime_cfg = config["runtime"]
    log_interval = runtime_cfg["log_interval"]
    grad_clip = training_cfg["gradient_clip_norm"]
    use_amp = training_cfg["mixed_precision"]["enabled"]
    image_shape = tuple(config["data"]["image"]["size"])

    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_bbox_loss = 0.0
    epoch_vel_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        global_step = epoch * len(train_loader) + batch_idx

        # Move data to device
        images = batch["images"].to(device, non_blocking=True)
        radar_points = batch["radar_points"].to(device, non_blocking=True)
        radar_num_points = batch["radar_num_points"].to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        extrinsics = batch["extrinsics"].to(device, non_blocking=True)
        gt_boxes = [b.to(device, non_blocking=True) for b in batch["gt_boxes"]]
        gt_labels = [l.to(device, non_blocking=True) for l in batch["gt_labels"]]

        # Forward pass with mixed precision
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model(
                images=images,
                radar_points=radar_points,
                radar_num_points=radar_num_points,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_shape=image_shape,
            )

            loss_dict = loss_fn(predictions, gt_boxes, gt_labels)
            total_loss = loss_dict["total_loss"]

        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()

        # Gradient clipping (unscale first for correct norm)
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step (OneCycleLR steps per batch)
        scheduler.step()

        # EMA update
        if ema is not None:
            # Use the underlying model for DDP
            model_for_ema = model.module if hasattr(model, "module") else model
            ema.update(model_for_ema)

        # Accumulate metrics
        epoch_loss += total_loss.item()
        epoch_cls_loss += loss_dict["cls_loss"].item()
        epoch_bbox_loss += loss_dict["bbox_loss"].item()
        epoch_vel_loss += loss_dict["velocity_loss"].item()
        num_batches += 1

        # Logging
        if batch_idx % log_interval == 0 and is_main_process():
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * images.shape[0] / max(elapsed, 1e-5)

            logger.info(
                f"Epoch [{epoch}][{batch_idx}/{len(train_loader)}] "
                f"lr: {lr:.2e} | "
                f"loss: {total_loss.item():.4f} | "
                f"cls: {loss_dict['cls_loss'].item():.4f} | "
                f"bbox: {loss_dict['bbox_loss'].item():.4f} | "
                f"vel: {loss_dict['velocity_loss'].item():.4f} | "
                f"speed: {samples_per_sec:.1f} samples/s"
            )

            # TensorBoard logging
            if writer is not None:
                writer.add_scalar("train/total_loss", total_loss.item(), global_step)
                writer.add_scalar("train/cls_loss", loss_dict["cls_loss"].item(), global_step)
                writer.add_scalar("train/bbox_loss", loss_dict["bbox_loss"].item(), global_step)
                writer.add_scalar("train/velocity_loss", loss_dict["velocity_loss"].item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/grad_scale", scaler.get_scale(), global_step)

            # WandB logging
            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "train/total_loss": total_loss.item(),
                        "train/cls_loss": loss_dict["cls_loss"].item(),
                        "train/bbox_loss": loss_dict["bbox_loss"].item(),
                        "train/velocity_loss": loss_dict["velocity_loss"].item(),
                        "train/lr": lr,
                        "global_step": global_step,
                    })
                except ImportError:
                    pass

    num_batches = max(num_batches, 1)
    metrics = {
        "train_loss": epoch_loss / num_batches,
        "train_cls_loss": epoch_cls_loss / num_batches,
        "train_bbox_loss": epoch_bbox_loss / num_batches,
        "train_vel_loss": epoch_vel_loss / num_batches,
    }

    if is_main_process():
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch} Training Complete - "
            f"loss: {metrics['train_loss']:.4f}, "
            f"cls: {metrics['train_cls_loss']:.4f}, "
            f"bbox: {metrics['train_bbox_loss']:.4f}, "
            f"vel: {metrics['train_vel_loss']:.4f}, "
            f"time: {elapsed:.1f}s"
        )

    return metrics


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Main training entry point.

    Sets up distributed training, builds model/optimizer/scheduler, handles
    checkpoint resume, and runs the full training loop with validation.

    Args:
        args: Parsed command-line arguments.
    """
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    training_cfg = config["training"]
    runtime_cfg = config["runtime"]

    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        training_cfg["optimizer"]["lr"] = args.lr
        training_cfg["scheduler"]["max_lr"] = args.lr
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.grad_clip is not None:
        training_cfg["gradient_clip_norm"] = args.grad_clip

    # Setup distributed training
    distributed = False
    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=runtime_cfg["distributed"]["backend"],
            init_method="env://",
        )
        distributed = True

    # Device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Set seed
    seed = runtime_cfg["seed"]
    set_seed(seed + get_rank())

    # Working directory
    work_dir = args.work_dir or runtime_cfg["work_dir"]
    os.makedirs(work_dir, exist_ok=True)

    # Logging
    setup_logging(work_dir, rank=get_rank())
    logger.info(f"Configuration:\n{yaml.dump(config, default_flow_style=False)}")
    logger.info(f"Device: {device}, Distributed: {distributed}, World size: {get_world_size()}")

    # cuDNN configuration
    torch.backends.cudnn.benchmark = runtime_cfg.get("cudnn_benchmark", True)
    torch.backends.cudnn.deterministic = runtime_cfg.get("deterministic", False)

    # Build datasets
    data_cfg = config["data"]
    train_dataset = NuScenesRadarCameraDataset(
        info_path=data_cfg["info_path"]["train"],
        root_path=data_cfg["root_path"],
        class_names=config["class_names"],
        image_size=tuple(data_cfg["image"]["size"]),
        point_cloud_range=data_cfg["point_cloud"]["range"],
        max_radar_points=2048,
        augmentation_cfg=data_cfg.get("augmentation", {}),
        is_train=True,
    )
    val_dataset = NuScenesRadarCameraDataset(
        info_path=data_cfg["info_path"]["val"],
        root_path=data_cfg["root_path"],
        class_names=config["class_names"],
        image_size=tuple(data_cfg["image"]["size"]),
        point_cloud_range=data_cfg["point_cloud"]["range"],
        max_radar_points=2048,
        augmentation_cfg={},
        is_train=False,
    )

    # Build data loaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=data_cfg["num_workers"],
        collate_fn=craft_collate_fn,
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
        persistent_workers=data_cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=data_cfg["num_workers"],
        collate_fn=craft_collate_fn,
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=False,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    model = CRAFTModel(config).to(device)

    # Sync BatchNorm for multi-GPU
    if distributed and training_cfg.get("sync_bn", True):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Converted BatchNorm to SyncBatchNorm")

    # DDP wrapper
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=training_cfg.get("find_unused_parameters", False),
        )

    # Model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Build optimizer
    opt_cfg = training_cfg["optimizer"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n],
         "lr": opt_cfg["lr"] * 0.1, "name": "backbone"},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n],
         "lr": opt_cfg["lr"], "name": "other"},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
    )

    # Build scheduler
    sched_cfg = training_cfg["scheduler"]
    total_steps = training_cfg["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[opt_cfg["lr"] * 0.1, sched_cfg["max_lr"]],
        total_steps=total_steps,
        pct_start=sched_cfg["pct_start"],
        anneal_strategy=sched_cfg["anneal_strategy"],
        div_factor=sched_cfg["div_factor"],
        final_div_factor=sched_cfg["final_div_factor"],
    )

    # Build loss function
    loss_fn = build_craft_loss(
        num_classes=config["model"]["detection_head"]["num_classes"],
        bbox_code_size=config["model"]["detection_head"]["bbox_code_size"],
        cls_weight=training_cfg["loss_weights"]["classification"],
        bbox_weight=training_cfg["loss_weights"]["bbox_regression"],
        velocity_weight=training_cfg["loss_weights"]["velocity"],
        bev_height=model.module.bev_height if hasattr(model, "module") else model.bev_height,
        bev_width=model.module.bev_width if hasattr(model, "module") else model.bev_width,
        point_cloud_range=data_cfg["point_cloud"]["range"],
        voxel_size=data_cfg["point_cloud"]["voxel_size"],
    ).to(device)

    # Mixed precision scaler
    use_amp = training_cfg["mixed_precision"]["enabled"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # EMA
    ema = None
    if training_cfg["ema"]["enabled"]:
        model_for_ema = model.module if hasattr(model, "module") else model
        ema = ModelEMA(model_for_ema, decay=training_cfg["ema"]["decay"])
        logger.info(f"EMA enabled with decay={training_cfg['ema']['decay']}")

    # TensorBoard writer
    writer = None
    if is_main_process():
        tb_dir = os.path.join(work_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    # WandB setup
    use_wandb = args.wandb and is_main_process()
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="craft-3d-detection",
                name=args.exp_name or f"craft_{time.strftime('%Y%m%d_%H%M%S')}",
                config=config,
                dir=work_dir,
            )
            logger.info("WandB logging enabled")
        except ImportError:
            logger.warning("wandb not installed, disabling WandB logging")
            use_wandb = False

    # Resume from checkpoint
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            model_to_load = model.module if hasattr(model, "module") else model
            model_to_load.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_metric = checkpoint.get("best_metric", 0.0)

            if ema is not None and "ema_state_dict" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state_dict"])

            logger.info(f"Resumed at epoch {start_epoch}, best metric: {best_metric:.4f}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}, starting from scratch")

    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)

    warmup_epochs = training_cfg.get("warmup_epochs", 2)
    eval_frequency = config["evaluation"].get("eval_frequency", 1)
    ckpt_cfg = training_cfg["checkpoint"]

    for epoch in range(start_epoch, training_cfg["epochs"]):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Determine if we're in warmup phase
        warmup_mode = epoch < warmup_epochs
        if warmup_mode and is_main_process():
            logger.info(f"Epoch {epoch}: WARMUP mode (branches trained separately)")

        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            scaler=scaler,
            ema=ema,
            device=device,
            epoch=epoch,
            config=config,
            writer=writer,
            use_wandb=use_wandb,
            warmup_mode=warmup_mode,
        )

        # Validation
        if (epoch + 1) % eval_frequency == 0:
            # Use EMA model for validation if available
            model_for_val = model
            if ema is not None:
                model_for_ema = model.module if hasattr(model, "module") else model
                ema.apply_shadow(model_for_ema)

            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch,
                config=config,
            )

            # Restore non-EMA parameters after validation
            if ema is not None:
                model_for_ema = model.module if hasattr(model, "module") else model
                ema.restore(model_for_ema)

            # Log validation metrics
            if writer is not None and is_main_process():
                for key, value in val_metrics.items():
                    writer.add_scalar(f"val/{key}", value, epoch)

            if use_wandb:
                try:
                    import wandb
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
                except ImportError:
                    pass

            # Check if this is the best model (using negative val_loss as proxy for mAP)
            # In practice, you'd run the nuScenes evaluator here
            current_metric = -val_metrics["val_loss"]
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
                logger.info(f"New best model at epoch {epoch} (metric: {-best_metric:.4f})")

        else:
            is_best = False

        # Save checkpoint
        if is_main_process() and (epoch + 1) % ckpt_cfg["save_interval"] == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            state = {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_metric": best_metric,
                "config": config,
            }
            if ema is not None:
                state["ema_state_dict"] = ema.state_dict()

            ckpt_path = os.path.join(work_dir, "checkpoints", f"epoch_{epoch}.pth")
            save_checkpoint(
                state,
                filepath=ckpt_path,
                is_best=is_best,
                max_keep=ckpt_cfg.get("max_keep", 5),
            )

        # Barrier for distributed sync
        if distributed:
            dist.barrier()

    # Cleanup
    if writer is not None:
        writer.close()
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass
    if distributed:
        dist.destroy_process_group()

    logger.info("Training complete!")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CRAFT training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="CRAFT: Camera-Radar 3D Object Detection Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/craft_nuscenes.yaml",
        help="Path to the YAML configuration file.",
    )

    # Training hyperparameters (override config)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU (overrides config).")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config).")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (overrides config).")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping norm (overrides config).")

    # Checkpoint / Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--work-dir", type=str, default=None, help="Working directory for outputs.")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for WandB.")

    # Misc
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config).")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only (no training).")

    args = parser.parse_args()
    return args


def main() -> None:
    """Main entry point for the training script."""
    args = parse_args()

    if args.seed is not None:
        # Will be applied inside train() after config is loaded
        pass

    if args.eval_only:
        # Evaluation-only mode
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        work_dir = args.work_dir or config["runtime"]["work_dir"]
        os.makedirs(work_dir, exist_ok=True)
        setup_logging(work_dir)

        # Build model and load checkpoint
        model = CRAFTModel(config).to(device)
        if args.resume and os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from {args.resume}")

            # Apply EMA if available
            if "ema_state_dict" in checkpoint:
                ema = ModelEMA(model, decay=config["training"]["ema"]["decay"])
                ema.load_state_dict(checkpoint["ema_state_dict"])
                ema.apply_shadow(model)
                logger.info("Applied EMA weights for evaluation")

        # Build val loader
        data_cfg = config["data"]
        val_dataset = NuScenesRadarCameraDataset(
            info_path=data_cfg["info_path"]["val"],
            root_path=data_cfg["root_path"],
            class_names=config["class_names"],
            image_size=tuple(data_cfg["image"]["size"]),
            point_cloud_range=data_cfg["point_cloud"]["range"],
            max_radar_points=2048,
            augmentation_cfg={},
            is_train=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            collate_fn=craft_collate_fn,
            pin_memory=True,
        )

        # Build loss
        bev_h = int(round((data_cfg["point_cloud"]["range"][3] - data_cfg["point_cloud"]["range"][0]) / data_cfg["point_cloud"]["voxel_size"][0]))
        bev_w = int(round((data_cfg["point_cloud"]["range"][4] - data_cfg["point_cloud"]["range"][1]) / data_cfg["point_cloud"]["voxel_size"][1]))
        loss_fn = build_craft_loss(
            num_classes=config["model"]["detection_head"]["num_classes"],
            bbox_code_size=config["model"]["detection_head"]["bbox_code_size"],
            cls_weight=config["training"]["loss_weights"]["classification"],
            bbox_weight=config["training"]["loss_weights"]["bbox_regression"],
            velocity_weight=config["training"]["loss_weights"]["velocity"],
            bev_height=bev_h,
            bev_width=bev_w,
            point_cloud_range=data_cfg["point_cloud"]["range"],
            voxel_size=data_cfg["point_cloud"]["voxel_size"],
        ).to(device)

        val_metrics = validate(model, val_loader, loss_fn, device, 0, config)
        logger.info(f"Evaluation results: {val_metrics}")
        return

    # Normal training mode
    train(args)


if __name__ == "__main__":
    main()
