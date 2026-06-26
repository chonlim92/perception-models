"""
CRAFT Inference Pipeline.

Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer.

Complete inference pipeline including:
    - BoundingBox3D dataclass with format conversions (corners, dict, nuScenes)
    - Post-processing (heatmap NMS, decoding, circle NMS)
    - CRAFTInference class for single/batch/file-based prediction
    - Visualization utilities (BEV projection, camera overlay)
    - Throughput benchmarking
    - CLI with infer/benchmark/export subcommands
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)


# ==============================================================================
# BoundingBox3D Dataclass
# ==============================================================================


@dataclass
class BoundingBox3D:
    """3D bounding box representation for detected objects.

    Attributes:
        center_x: X coordinate of box center in ego frame (meters).
        center_y: Y coordinate of box center in ego frame (meters).
        center_z: Z coordinate of box center in ego frame (meters).
        width: Box width along the local x-axis (meters).
        length: Box length along the local y-axis (meters).
        height: Box height along the local z-axis (meters).
        yaw: Rotation angle around the z-axis (radians), 0 = facing +x.
        velocity_x: Velocity along global x-axis (m/s).
        velocity_y: Velocity along global y-axis (m/s).
        score: Detection confidence score in [0, 1].
        class_id: Integer class index (0-indexed).
        class_name: Human-readable class name string.
    """

    center_x: float
    center_y: float
    center_z: float
    width: float
    length: float
    height: float
    yaw: float
    velocity_x: float
    velocity_y: float
    score: float
    class_id: int
    class_name: str

    def to_corners(self) -> np.ndarray:
        """Compute the 8 corners of the 3D bounding box.

        Returns:
            Array of shape (8, 3) with corner coordinates in the ego frame.
            Corner ordering:
                0-3: bottom face (z = center_z - height/2)
                4-7: top face (z = center_z + height/2)
                Within each face: front-left, front-right, rear-right, rear-left.
        """
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        hw = self.width / 2.0
        hl = self.length / 2.0
        hh = self.height / 2.0

        # Corner offsets in the local frame (x-right, y-forward convention)
        # front-left, front-right, rear-right, rear-left
        dx = np.array([hw, hw, -hw, -hw])
        dy = np.array([hl, -hl, -hl, hl])

        # Rotate to global frame
        corners_x = cos_yaw * dx - sin_yaw * dy + self.center_x
        corners_y = sin_yaw * dx + cos_yaw * dy + self.center_y

        # Bottom and top faces
        z_bottom = self.center_z - hh
        z_top = self.center_z + hh

        corners = np.zeros((8, 3), dtype=np.float64)
        corners[0:4, 0] = corners_x
        corners[0:4, 1] = corners_y
        corners[0:4, 2] = z_bottom
        corners[4:8, 0] = corners_x
        corners[4:8, 1] = corners_y
        corners[4:8, 2] = z_top

        return corners

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary.

        Returns:
            Dictionary with all box attributes.
        """
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "center_z": self.center_z,
            "width": self.width,
            "length": self.length,
            "height": self.height,
            "yaw": self.yaw,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    def to_nuscenes_format(self) -> Dict[str, Any]:
        """Convert to nuScenes detection submission format.

        Returns:
            Dictionary matching the nuScenes detection evaluation format:
                - translation: [x, y, z] center position
                - size: [width, length, height] in meters
                - rotation: [w, x, y, z] quaternion
                - velocity: [vx, vy] in m/s
                - detection_name: class name string
                - detection_score: confidence score
                - attribute_name: default empty string
        """
        # Convert yaw to quaternion (rotation around z-axis)
        # q = [cos(yaw/2), 0, 0, sin(yaw/2)]
        half_yaw = self.yaw / 2.0
        qw = math.cos(half_yaw)
        qx = 0.0
        qy = 0.0
        qz = math.sin(half_yaw)

        return {
            "translation": [self.center_x, self.center_y, self.center_z],
            "size": [self.width, self.length, self.height],
            "rotation": [qw, qx, qy, qz],
            "velocity": [self.velocity_x, self.velocity_y],
            "detection_name": self.class_name,
            "detection_score": self.score,
            "attribute_name": "",
        }


# ==============================================================================
# Post-Processing Functions
# ==============================================================================

NUSCENES_CLASS_NAMES: List[str] = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


def nms_bev_heatmap(
    heatmap: torch.Tensor, kernel_size: int = 3
) -> torch.Tensor:
    """Apply max-pool based NMS to the heatmap.

    Suppresses non-peak locations by keeping only pixels that are local maxima
    within a kernel_size x kernel_size neighborhood.

    Args:
        heatmap: Detection heatmap [B, num_classes, H, W] with values in [0, 1].
        kernel_size: Size of the max-pooling kernel for peak detection.

    Returns:
        Heatmap with non-peak locations zeroed out, same shape as input.
    """
    padding = kernel_size // 2
    heatmap_max = F.max_pool2d(
        heatmap, kernel_size=kernel_size, stride=1, padding=padding
    )
    # Keep only positions that are local maxima
    keep_mask = (heatmap_max == heatmap).float()
    return heatmap * keep_mask


def decode_heatmap_to_boxes(
    heatmap: torch.Tensor,
    regression: torch.Tensor,
    velocity: torch.Tensor,
    score_threshold: float = 0.1,
    max_detections: int = 500,
    nms_kernel: int = 3,
    voxel_size: float = 0.2,
    x_min: float = -51.2,
    y_min: float = -51.2,
    class_names: Optional[List[str]] = None,
) -> List[List[BoundingBox3D]]:
    """Decode network output tensors into lists of BoundingBox3D objects.

    Performs heatmap NMS, selects top-K peaks, decodes regression parameters
    into 3D bounding box coordinates, and filters by score threshold.

    Args:
        heatmap: Detection heatmap [B, num_classes, H, W] after sigmoid.
        regression: Regression output [B, 8, H, W] with channels:
            (offset_x, offset_y, z, log_w, log_l, log_h, sin_yaw, cos_yaw).
        velocity: Velocity prediction [B, 2, H, W] (vx, vy in m/s).
        score_threshold: Minimum confidence to keep a detection.
        max_detections: Maximum number of detections per sample.
        nms_kernel: Kernel size for heatmap max-pool NMS.
        voxel_size: BEV grid cell size in meters.
        x_min: Minimum x coordinate of BEV grid.
        y_min: Minimum y coordinate of BEV grid.
        class_names: List of class name strings (length = num_classes).

    Returns:
        List of lists of BoundingBox3D, one inner list per batch sample.
    """
    if class_names is None:
        class_names = NUSCENES_CLASS_NAMES

    # Apply heatmap NMS
    heatmap_nms = nms_bev_heatmap(heatmap, kernel_size=nms_kernel)

    batch_size, num_classes, height, width = heatmap_nms.shape
    results: List[List[BoundingBox3D]] = []

    for b in range(batch_size):
        sample_boxes: List[BoundingBox3D] = []

        # Flatten heatmap across classes and spatial dimensions
        heatmap_flat = heatmap_nms[b].view(-1)  # [num_classes * H * W]

        # Select top-K candidates
        num_candidates = min(max_detections, heatmap_flat.numel())
        topk_scores, topk_inds = torch.topk(heatmap_flat, num_candidates)

        # Filter by score threshold
        valid_mask = topk_scores >= score_threshold
        topk_scores = topk_scores[valid_mask]
        topk_inds = topk_inds[valid_mask]

        if topk_scores.numel() == 0:
            results.append(sample_boxes)
            continue

        # Decode linear indices to (class, row, col)
        spatial_size = height * width
        topk_classes = topk_inds // spatial_size
        spatial_inds = topk_inds % spatial_size
        topk_rows = spatial_inds // width
        topk_cols = spatial_inds % width

        # Gather regression values at peak locations
        reg_flat = regression[b].view(8, -1)  # [8, H*W]
        reg_vals = reg_flat[:, spatial_inds].T  # [N, 8]

        vel_flat = velocity[b].view(2, -1)  # [2, H*W]
        vel_vals = vel_flat[:, spatial_inds].T  # [N, 2]

        # Decode regression parameters
        offset_x = reg_vals[:, 0]
        offset_y = reg_vals[:, 1]
        z_center = reg_vals[:, 2]
        log_w = reg_vals[:, 3]
        log_l = reg_vals[:, 4]
        log_h = reg_vals[:, 5]
        sin_yaw = reg_vals[:, 6]
        cos_yaw = reg_vals[:, 7]

        # Convert to absolute coordinates
        cx = (topk_cols.float() + offset_x) * voxel_size + x_min
        cy = (topk_rows.float() + offset_y) * voxel_size + y_min
        cz = z_center
        w = torch.exp(log_w).clamp(max=50.0)
        l = torch.exp(log_l).clamp(max=50.0)
        h = torch.exp(log_h).clamp(max=10.0)
        yaw = torch.atan2(sin_yaw, cos_yaw)

        vx = vel_vals[:, 0]
        vy = vel_vals[:, 1]

        # Convert to BoundingBox3D objects
        for i in range(topk_scores.shape[0]):
            cls_id = int(topk_classes[i].item())
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

            box = BoundingBox3D(
                center_x=float(cx[i].item()),
                center_y=float(cy[i].item()),
                center_z=float(cz[i].item()),
                width=float(w[i].item()),
                length=float(l[i].item()),
                height=float(h[i].item()),
                yaw=float(yaw[i].item()),
                velocity_x=float(vx[i].item()),
                velocity_y=float(vy[i].item()),
                score=float(topk_scores[i].item()),
                class_id=cls_id,
                class_name=cls_name,
            )
            sample_boxes.append(box)

        results.append(sample_boxes)

    return results


def circle_nms_bev(
    boxes: List[BoundingBox3D],
    radius: float = 4.0,
) -> List[BoundingBox3D]:
    """Apply circular Non-Maximum Suppression in BEV space.

    For each pair of boxes, if their BEV center distance is less than
    the specified radius, the lower-scoring box is suppressed. This is
    simpler and faster than IoU-based NMS for 3D detection.

    Args:
        boxes: List of BoundingBox3D detections, assumed sorted by score descending.
        radius: Suppression radius in meters. Boxes within this BEV distance
            of a higher-scoring box are removed.

    Returns:
        Filtered list of BoundingBox3D after suppression.
    """
    if len(boxes) == 0:
        return []

    # Sort by score descending (should already be sorted, but ensure)
    boxes_sorted = sorted(boxes, key=lambda b: b.score, reverse=True)

    kept: List[BoundingBox3D] = []
    suppressed = [False] * len(boxes_sorted)
    radius_sq = radius * radius

    for i in range(len(boxes_sorted)):
        if suppressed[i]:
            continue
        kept.append(boxes_sorted[i])

        # Suppress all lower-scoring boxes within radius
        for j in range(i + 1, len(boxes_sorted)):
            if suppressed[j]:
                continue
            dx = boxes_sorted[i].center_x - boxes_sorted[j].center_x
            dy = boxes_sorted[i].center_y - boxes_sorted[j].center_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < radius_sq:
                suppressed[j] = True

    return kept


def convert_to_nuscenes_submission(
    predictions: Dict[str, List[BoundingBox3D]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert all predictions to nuScenes detection submission format.

    Args:
        predictions: Dictionary mapping sample tokens (str) to lists of
            BoundingBox3D detections for that sample.
        metadata: Optional metadata dict to include in the submission
            (e.g., model name, version, description).

    Returns:
        Complete nuScenes submission dictionary with keys:
            - "meta": metadata about the submission
            - "results": mapping from sample_token -> list of detection dicts
    """
    if metadata is None:
        metadata = {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": True,
            "use_map": False,
            "use_external": False,
        }

    results: Dict[str, List[Dict[str, Any]]] = {}

    for sample_token, boxes in predictions.items():
        sample_results: List[Dict[str, Any]] = []
        for box in boxes:
            det = box.to_nuscenes_format()
            det["sample_token"] = sample_token
            sample_results.append(det)
        results[sample_token] = sample_results

    return {
        "meta": metadata,
        "results": results,
    }


# ==============================================================================
# CRAFT Model Definition (for inference loading)
# ==============================================================================


class CRAFTModel(nn.Module):
    """CRAFT full model assembly for inference.

    Combines camera branch, radar branch, fusion transformer, and detection head.
    This class provides the same interface as the training model but is kept
    self-contained so inference.py can be used standalone.

    Args:
        config: Model configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        model_cfg = config.get("model", config)

        # Import component modules
        from .camera_branch import build_camera_branch
        from .radar_branch import build_radar_branch
        from .fusion_transformer import build_fusion_transformer
        from .heads import CRAFTDetectionHead

        # Camera branch
        camera_cfg = model_cfg.get("camera", {})
        self.camera_branch = build_camera_branch(
            backbone_name=camera_cfg.get("backbone", "resnet50"),
            pretrained=camera_cfg.get("pretrained", False),
            fpn_out_channels=camera_cfg.get("fpn_channels", 256),
            num_cameras=camera_cfg.get("num_cameras", 6),
            frozen_stages=camera_cfg.get("frozen_stages", 1),
        )

        # Radar branch
        radar_cfg = model_cfg.get("radar", {})
        self.radar_branch = build_radar_branch(
            point_cloud_range=radar_cfg.get(
                "point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            ),
            voxel_size=radar_cfg.get("voxel_size", [0.2, 0.2, 8.0]),
            max_points_per_pillar=radar_cfg.get("max_points_per_pillar", 20),
            max_num_pillars=radar_cfg.get("max_num_pillars", 30000),
            in_channels=radar_cfg.get("in_channels", 6),
            pillar_feat_channels=radar_cfg.get("pillar_feat_channels", 128),
            bev_out_channels=radar_cfg.get("bev_out_channels", 256),
        )

        # Fusion transformer
        fusion_cfg = model_cfg.get("fusion", {})
        self.fusion_transformer = build_fusion_transformer(
            d_model=fusion_cfg.get("d_model", 256),
            n_heads=fusion_cfg.get("n_heads", 8),
            d_ffn=fusion_cfg.get("d_ffn", 1024),
            n_layers=fusion_cfg.get("n_layers", 6),
            dropout=fusion_cfg.get("dropout", 0.0),
            radar_channels=fusion_cfg.get("radar_channels", 256),
            camera_channels=fusion_cfg.get("camera_channels", 256),
        )

        # Detection head
        head_cfg = model_cfg.get("head", {})
        self.detection_head = CRAFTDetectionHead(
            in_channels=head_cfg.get("in_channels", 256),
            shared_channels=head_cfg.get("shared_channels", 256),
            num_classes=head_cfg.get("num_classes", 10),
            head_hidden_channels=head_cfg.get("hidden_channels", 64),
            num_head_conv_layers=head_cfg.get("num_conv_layers", 2),
            num_shared_conv_layers=head_cfg.get("num_shared_conv_layers", 3),
        )

        # Store config
        self.config = config
        self.num_classes = head_cfg.get("num_classes", 10)

    def forward(
        self,
        images: torch.Tensor,
        radar_points: torch.Tensor,
        num_points: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the full CRAFT model.

        Args:
            images: Multi-view camera images [B, 6, 3, H, W].
            radar_points: Radar point clouds [B, N_max, 6].
            num_points: Valid point counts [B].
            intrinsics: Camera intrinsic matrices [B, 6, 3, 3].
            extrinsics: Ego-to-camera extrinsic matrices [B, 6, 4, 4].

        Returns:
            Dictionary with detection head outputs:
                'heatmap': [B, num_classes, H_bev, W_bev]
                'regression': [B, 8, H_bev, W_bev]
                'velocity': [B, 2, H_bev, W_bev]
        """
        # Camera feature extraction
        camera_output = self.camera_branch(images)
        camera_features = camera_output["features"]

        # Use the P3 (stride-8) level as the primary camera feature
        # Shape: [B, 6, 256, H/8, W/8]
        cam_feats = camera_features[1]

        # Radar feature extraction
        radar_output = self.radar_branch(radar_points, num_points)
        radar_bev_features = radar_output["bev_features"]  # [B, 256, H_bev, W_bev]

        # Fusion (with or without calibration matrices)
        if intrinsics is not None and extrinsics is not None:
            _, _, _, img_h, img_w = images.shape
            fused_features = self.fusion_transformer(
                radar_bev_features=radar_bev_features,
                camera_features=cam_feats,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_shape=(img_h, img_w),
            )
        else:
            # Without calibration, use radar features directly
            # (fallback: skip cross-attention from camera)
            fused_features = radar_bev_features

        # Detection head
        head_outputs = self.detection_head(fused_features)

        return head_outputs


# ==============================================================================
# CRAFTInference Class
# ==============================================================================


class CRAFTInference:
    """High-level inference wrapper for the CRAFT model.

    Handles model loading, warmup, single-sample and batch prediction, and
    file-based inference from disk. Manages device placement and timing.

    Args:
        model_path: Path to the model checkpoint (.pth file).
        config: Configuration dictionary or path to YAML config file.
        score_threshold: Minimum detection confidence score.
        max_detections: Maximum number of detections per sample.
        nms_kernel: Kernel size for heatmap max-pool NMS.
        circle_nms_radius: Radius for circle NMS in BEV (meters). Set to 0 to disable.
        device: Device string ('cuda', 'cuda:0', 'cpu', etc.).
    """

    def __init__(
        self,
        model_path: str,
        config: Any,
        score_threshold: float = 0.3,
        max_detections: int = 500,
        nms_kernel: int = 3,
        circle_nms_radius: float = 4.0,
        device: Optional[str] = None,
    ) -> None:
        # Resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load config
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError(f"config must be a dict or path to YAML, got {type(config)}")

        self.model_path = model_path
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.nms_kernel = nms_kernel
        self.circle_nms_radius = circle_nms_radius

        # Resolve class names
        self.class_names = self.config.get("class_names", NUSCENES_CLASS_NAMES)

        # BEV grid parameters
        data_cfg = self.config.get("data", {})
        self.voxel_size = data_cfg.get("voxel_size", 0.2)
        self.bev_range = data_cfg.get("bev_range", [-51.2, -51.2, 51.2, 51.2])
        self.x_min = self.bev_range[0] if len(self.bev_range) >= 4 else -51.2
        self.y_min = self.bev_range[1] if len(self.bev_range) >= 4 else -51.2

        # Load model
        self.model = self._load_model()
        self.model.eval()

        logger.info(
            "CRAFTInference initialized: device=%s, threshold=%.2f, max_det=%d",
            self.device,
            self.score_threshold,
            self.max_detections,
        )

    def _load_model(self) -> CRAFTModel:
        """Load model from checkpoint.

        Supports checkpoint dicts with keys 'model_state_dict' or 'state_dict',
        as well as raw state dicts.

        Returns:
            Loaded and configured CRAFTModel on the target device.
        """
        logger.info("Loading model from %s", self.model_path)

        model = CRAFTModel(self.config)

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if model was saved with DataParallel
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                cleaned_state_dict[key[7:]] = value
            else:
                cleaned_state_dict[key] = value

        model.load_state_dict(cleaned_state_dict, strict=False)
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model loaded: %.2fM parameters", total_params / 1e6)

        return model

    def _warmup(self, num_passes: int = 3) -> None:
        """Run dummy forward passes to warm up the model and CUDA kernels.

        Args:
            num_passes: Number of dummy forward passes to execute.
        """
        logger.info("Warming up model with %d passes...", num_passes)

        model_cfg = self.config.get("model", self.config)
        data_cfg = self.config.get("data", {})
        img_h = data_cfg.get("image_height", 256)
        img_w = data_cfg.get("image_width", 704)

        with torch.no_grad():
            dummy_images = torch.randn(
                1, 6, 3, img_h, img_w, device=self.device
            )
            dummy_radar = torch.randn(1, 100, 6, device=self.device)
            dummy_num_points = torch.tensor([100], device=self.device)

            for _ in range(num_passes):
                self.model(dummy_images, dummy_radar, dummy_num_points)

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

        logger.info("Warmup complete.")

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        radar_points: torch.Tensor,
        num_points: torch.Tensor,
        calibration: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[List[BoundingBox3D], float]:
        """Run inference on a single sample.

        Args:
            images: Camera images [1, 6, 3, H, W] or [6, 3, H, W].
            radar_points: Radar points [1, N, 6] or [N, 6].
            num_points: Number of valid points [1] or scalar.
            calibration: Optional dict with 'intrinsics' [1, 6, 3, 3] and
                'extrinsics' [1, 6, 4, 4].

        Returns:
            Tuple of (list of BoundingBox3D detections, inference time in ms).
        """
        # Ensure batch dimension
        if images.dim() == 4:
            images = images.unsqueeze(0)
        if radar_points.dim() == 2:
            radar_points = radar_points.unsqueeze(0)
        if num_points.dim() == 0:
            num_points = num_points.unsqueeze(0)

        # Move to device
        images = images.to(self.device)
        radar_points = radar_points.to(self.device)
        num_points = num_points.to(self.device)

        intrinsics = None
        extrinsics = None
        if calibration is not None:
            intrinsics = calibration["intrinsics"].to(self.device)
            extrinsics = calibration["extrinsics"].to(self.device)
            if intrinsics.dim() == 3:
                intrinsics = intrinsics.unsqueeze(0)
            if extrinsics.dim() == 3:
                extrinsics = extrinsics.unsqueeze(0)

        # Synchronize before timing
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        start_time = time.perf_counter()

        # Forward pass
        outputs = self.model(images, radar_points, num_points, intrinsics, extrinsics)

        # Synchronize after forward pass
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        # Decode predictions
        batch_boxes = decode_heatmap_to_boxes(
            heatmap=outputs["heatmap"],
            regression=outputs["regression"],
            velocity=outputs["velocity"],
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            nms_kernel=self.nms_kernel,
            voxel_size=self.voxel_size,
            x_min=self.x_min,
            y_min=self.y_min,
            class_names=self.class_names,
        )

        # Apply circle NMS if enabled
        boxes = batch_boxes[0]
        if self.circle_nms_radius > 0 and len(boxes) > 0:
            boxes = circle_nms_bev(boxes, radius=self.circle_nms_radius)

        end_time = time.perf_counter()
        time_ms = (end_time - start_time) * 1000.0

        return boxes, time_ms

    @torch.no_grad()
    def predict_batch(
        self, batch_dict: Dict[str, torch.Tensor]
    ) -> Tuple[List[List[BoundingBox3D]], float]:
        """Run inference on a batch of samples.

        Args:
            batch_dict: Dictionary with keys:
                'images': [B, 6, 3, H, W]
                'radar_points': [B, N_max, 6]
                'num_points': [B]
                'intrinsics' (optional): [B, 6, 3, 3]
                'extrinsics' (optional): [B, 6, 4, 4]

        Returns:
            Tuple of (list of box lists per sample, total inference time in ms).
        """
        images = batch_dict["images"].to(self.device)
        radar_points = batch_dict["radar_points"].to(self.device)
        num_points = batch_dict["num_points"].to(self.device)

        intrinsics = None
        extrinsics = None
        if "intrinsics" in batch_dict:
            intrinsics = batch_dict["intrinsics"].to(self.device)
        if "extrinsics" in batch_dict:
            extrinsics = batch_dict["extrinsics"].to(self.device)

        # Synchronize before timing
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        start_time = time.perf_counter()

        # Forward pass
        outputs = self.model(images, radar_points, num_points, intrinsics, extrinsics)

        # Synchronize after forward pass
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        # Decode predictions
        batch_boxes = decode_heatmap_to_boxes(
            heatmap=outputs["heatmap"],
            regression=outputs["regression"],
            velocity=outputs["velocity"],
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            nms_kernel=self.nms_kernel,
            voxel_size=self.voxel_size,
            x_min=self.x_min,
            y_min=self.y_min,
            class_names=self.class_names,
        )

        # Apply circle NMS per sample if enabled
        if self.circle_nms_radius > 0:
            for i in range(len(batch_boxes)):
                if len(batch_boxes[i]) > 0:
                    batch_boxes[i] = circle_nms_bev(
                        batch_boxes[i], radius=self.circle_nms_radius
                    )

        end_time = time.perf_counter()
        time_ms = (end_time - start_time) * 1000.0

        return batch_boxes, time_ms

    @torch.no_grad()
    def predict_from_files(
        self, sample_dir: str
    ) -> Tuple[List[BoundingBox3D], float]:
        """Load sensor data from disk and run inference.

        Expected directory structure:
            sample_dir/
                images/          - 6 camera images (front.png, front_left.png, etc.)
                radar_points.npy - Radar points array [N, 6]
                calibration.json - Intrinsics and extrinsics (optional)

        Args:
            sample_dir: Path to the sample directory.

        Returns:
            Tuple of (list of BoundingBox3D detections, inference time in ms).
        """
        sample_path = Path(sample_dir)

        # Load camera images
        image_dir = sample_path / "images"
        camera_names = [
            "front", "front_left", "front_right",
            "back", "back_left", "back_right",
        ]

        images_list = []
        for cam_name in camera_names:
            # Try common image extensions
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = image_dir / f"{cam_name}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                raise FileNotFoundError(
                    f"Camera image not found for '{cam_name}' in {image_dir}"
                )

            # Load image using numpy (avoid PIL dependency for raw loading)
            try:
                from PIL import Image
                img = Image.open(str(img_path)).convert("RGB")
                img_array = np.array(img, dtype=np.float32) / 255.0
            except ImportError:
                import cv2
                img_bgr = cv2.imread(str(img_path))
                img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Transpose to CHW format
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
            images_list.append(img_tensor)

        images = torch.stack(images_list, dim=0)  # [6, 3, H, W]

        # Resize if needed
        data_cfg = self.config.get("data", {})
        target_h = data_cfg.get("image_height", 256)
        target_w = data_cfg.get("image_width", 704)
        _, _, h, w = images.shape
        if h != target_h or w != target_w:
            images = F.interpolate(
                images, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        # Load radar points
        radar_file = sample_path / "radar_points.npy"
        if not radar_file.exists():
            raise FileNotFoundError(f"Radar points file not found: {radar_file}")
        radar_np = np.load(str(radar_file)).astype(np.float32)
        radar_points = torch.from_numpy(radar_np)  # [N, 6]
        num_points = torch.tensor([radar_points.shape[0]], dtype=torch.long)

        # Load calibration if available
        calibration = None
        calib_file = sample_path / "calibration.json"
        if calib_file.exists():
            with open(calib_file, "r") as f:
                calib_data = json.load(f)
            intrinsics = torch.tensor(
                calib_data["intrinsics"], dtype=torch.float32
            )  # [6, 3, 3]
            extrinsics = torch.tensor(
                calib_data["extrinsics"], dtype=torch.float32
            )  # [6, 4, 4]
            calibration = {
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
            }

        # Run prediction
        return self.predict(images, radar_points, num_points, calibration)


# ==============================================================================
# Visualization Functions
# ==============================================================================


def project_boxes_to_image(
    boxes: List[BoundingBox3D],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image: np.ndarray,
    line_width: int = 2,
) -> np.ndarray:
    """Project 3D bounding boxes onto a camera image.

    Draws the 3D box wireframe projected onto the image plane using the given
    camera calibration parameters.

    Args:
        boxes: List of BoundingBox3D detections to project.
        intrinsic: Camera intrinsic matrix [3, 3].
        extrinsic: Ego-to-camera extrinsic matrix [4, 4].
        image: Input image as numpy array [H, W, 3] in BGR or RGB (uint8).
        line_width: Width of projected box edges in pixels.

    Returns:
        Annotated image with projected boxes drawn, same format as input.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping image projection.")
        return image

    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Class color map (BGR)
    colors = [
        (0, 255, 0),    # car - green
        (0, 200, 255),  # truck - orange
        (0, 128, 255),  # construction_vehicle - dark orange
        (255, 0, 0),    # bus - blue
        (255, 128, 0),  # trailer - light blue
        (128, 128, 128),  # barrier - gray
        (0, 0, 255),    # motorcycle - red
        (255, 0, 255),  # bicycle - magenta
        (0, 255, 255),  # pedestrian - yellow
        (128, 0, 128),  # traffic_cone - purple
    ]

    # Box edge connectivity: 12 edges of a cuboid
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]

    for box in boxes:
        corners_3d = box.to_corners()  # (8, 3)

        # Transform to camera frame
        ones = np.ones((8, 1), dtype=np.float64)
        corners_homo = np.concatenate([corners_3d, ones], axis=1)  # (8, 4)
        corners_cam = (extrinsic @ corners_homo.T).T  # (8, 4)
        corners_cam_xyz = corners_cam[:, :3]  # (8, 3)

        # Check if any corner is behind camera
        depths = corners_cam_xyz[:, 2]
        if np.all(depths <= 0):
            continue

        # Project to image plane
        corners_proj = (intrinsic @ corners_cam_xyz.T).T  # (8, 3)
        corners_2d = corners_proj[:, :2] / corners_proj[:, 2:3].clip(min=1e-5)  # (8, 2)

        # Get color for this class
        color = colors[box.class_id % len(colors)]

        # Draw edges
        for i_start, i_end in edges:
            # Only draw if both endpoints are in front of camera
            if depths[i_start] <= 0 or depths[i_end] <= 0:
                continue

            pt1 = (int(round(corners_2d[i_start, 0])), int(round(corners_2d[i_start, 1])))
            pt2 = (int(round(corners_2d[i_end, 0])), int(round(corners_2d[i_end, 1])))

            # Skip if both points are far outside the image
            if (pt1[0] < -w or pt1[0] > 2 * w or pt1[1] < -h or pt1[1] > 2 * h):
                if (pt2[0] < -w or pt2[0] > 2 * w or pt2[1] < -h or pt2[1] > 2 * h):
                    continue

            cv2.line(annotated, pt1, pt2, color, thickness=line_width)

        # Draw label
        # Find the topmost visible corner for label placement
        visible_mask = depths > 0
        if visible_mask.any():
            visible_2d = corners_2d[visible_mask]
            top_idx = np.argmin(visible_2d[:, 1])
            label_pos = (
                int(round(visible_2d[top_idx, 0])),
                int(round(visible_2d[top_idx, 1])) - 5,
            )
            label_text = f"{box.class_name} {box.score:.2f}"
            cv2.putText(
                annotated,
                label_text,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    return annotated


def visualize_bev(
    boxes: List[BoundingBox3D],
    bev_range: Tuple[float, float, float, float] = (-51.2, -51.2, 51.2, 51.2),
    canvas_size: int = 800,
    background_color: Tuple[int, int, int] = (40, 40, 40),
    ego_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Render detected boxes in a Bird's Eye View visualization.

    Creates a top-down view of the ego vehicle surroundings with detected
    bounding boxes drawn as oriented rectangles.

    Args:
        boxes: List of BoundingBox3D detections to visualize.
        bev_range: (x_min, y_min, x_max, y_max) extent in meters.
        canvas_size: Size of the output square image in pixels.
        background_color: BGR background color.
        ego_color: BGR color for the ego vehicle marker.

    Returns:
        BEV visualization as numpy array [canvas_size, canvas_size, 3] in BGR.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, returning blank canvas.")
        canvas = np.full((canvas_size, canvas_size, 3), background_color[0], dtype=np.uint8)
        return canvas

    x_min, y_min, x_max, y_max = bev_range
    x_range = x_max - x_min
    y_range = y_max - y_min

    canvas = np.full((canvas_size, canvas_size, 3), background_color[0], dtype=np.uint8)
    canvas[:, :, 0] = background_color[0]
    canvas[:, :, 1] = background_color[1]
    canvas[:, :, 2] = background_color[2]

    # Helper to convert world coords to pixel coords
    def world_to_pixel(wx: float, wy: float) -> Tuple[int, int]:
        px = int((wx - x_min) / x_range * canvas_size)
        py = int((y_max - wy) / y_range * canvas_size)  # Flip y for image coords
        return px, py

    # Draw grid lines
    grid_spacing = 10.0  # meters
    grid_color = (60, 60, 60)
    x_val = x_min
    while x_val <= x_max:
        p1 = world_to_pixel(x_val, y_min)
        p2 = world_to_pixel(x_val, y_max)
        cv2.line(canvas, p1, p2, grid_color, 1)
        x_val += grid_spacing
    y_val = y_min
    while y_val <= y_max:
        p1 = world_to_pixel(x_min, y_val)
        p2 = world_to_pixel(x_max, y_val)
        cv2.line(canvas, p1, p2, grid_color, 1)
        y_val += grid_spacing

    # Draw ego vehicle as a cross at origin
    ego_px, ego_py = world_to_pixel(0, 0)
    cv2.drawMarker(
        canvas,
        (ego_px, ego_py),
        ego_color,
        cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )

    # Draw range circles
    for r in [10, 20, 30, 40, 50]:
        radius_px = int(r / x_range * canvas_size)
        cv2.circle(canvas, (ego_px, ego_py), radius_px, (50, 50, 50), 1)

    # Class color map (BGR)
    colors = [
        (0, 255, 0),    # car
        (0, 200, 255),  # truck
        (0, 128, 255),  # construction_vehicle
        (255, 0, 0),    # bus
        (255, 128, 0),  # trailer
        (128, 128, 128),  # barrier
        (0, 0, 255),    # motorcycle
        (255, 0, 255),  # bicycle
        (0, 255, 255),  # pedestrian
        (128, 0, 128),  # traffic_cone
    ]

    # Draw each box as an oriented rectangle
    for box in boxes:
        corners_3d = box.to_corners()  # (8, 3)
        # Use bottom face corners (0-3) for BEV
        bev_corners = corners_3d[:4, :2]  # (4, 2) - x, y only

        # Convert to pixel coordinates
        pts = []
        for cx, cy in bev_corners:
            px, py = world_to_pixel(cx, cy)
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)

        color = colors[box.class_id % len(colors)]

        # Draw filled polygon with transparency effect
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        # Draw heading direction (front edge is thicker)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), color, thickness=3)

        # Draw velocity arrow if significant
        vel_magnitude = math.sqrt(box.velocity_x ** 2 + box.velocity_y ** 2)
        if vel_magnitude > 0.5:
            center_px, center_py = world_to_pixel(box.center_x, box.center_y)
            # Scale velocity for visualization (1 m/s = 5 pixels)
            vel_scale = 5.0 / x_range * canvas_size
            arrow_end_x = center_px + int(box.velocity_x * vel_scale)
            arrow_end_y = center_py - int(box.velocity_y * vel_scale)
            cv2.arrowedLine(
                canvas,
                (center_px, center_py),
                (arrow_end_x, arrow_end_y),
                color,
                thickness=1,
                tipLength=0.3,
            )

        # Draw class label
        center_px, center_py = world_to_pixel(box.center_x, box.center_y)
        label = f"{box.class_name[:3]} {box.score:.1f}"
        cv2.putText(
            canvas,
            label,
            (center_px - 15, center_py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )

    # Draw legend
    y_offset = 20
    for i, name in enumerate(NUSCENES_CLASS_NAMES):
        color = colors[i % len(colors)]
        cv2.rectangle(canvas, (10, y_offset - 10), (25, y_offset + 2), color, -1)
        cv2.putText(
            canvas,
            name,
            (30, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        y_offset += 18

    return canvas


def save_visualization(
    boxes: List[BoundingBox3D],
    output_dir: str,
    sample_name: str = "sample",
    images: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    extrinsics: Optional[np.ndarray] = None,
    bev_range: Tuple[float, float, float, float] = (-51.2, -51.2, 51.2, 51.2),
) -> List[str]:
    """Save annotated visualization images to disk.

    Generates and saves:
        - BEV top-down view with all detections
        - Camera images with projected boxes (if camera data provided)

    Args:
        boxes: List of BoundingBox3D detections.
        output_dir: Directory to save visualization images.
        sample_name: Base name for output files.
        images: Optional camera images [N_cams, H, W, 3] as uint8 BGR.
        intrinsics: Optional camera intrinsics [N_cams, 3, 3].
        extrinsics: Optional camera extrinsics [N_cams, 4, 4].
        bev_range: BEV visualization extent (x_min, y_min, x_max, y_max).

    Returns:
        List of paths to saved visualization files.
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV required for visualization. Install with: pip install opencv-python")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files: List[str] = []

    # Save BEV visualization
    bev_img = visualize_bev(boxes, bev_range=bev_range)
    bev_file = str(output_path / f"{sample_name}_bev.png")
    cv2.imwrite(bev_file, bev_img)
    saved_files.append(bev_file)
    logger.info("Saved BEV visualization: %s", bev_file)

    # Save camera projections if available
    if images is not None and intrinsics is not None and extrinsics is not None:
        camera_names = [
            "front", "front_left", "front_right",
            "back", "back_left", "back_right",
        ]
        num_cams = min(images.shape[0], len(camera_names))

        for cam_idx in range(num_cams):
            cam_img = images[cam_idx]
            cam_intrinsic = intrinsics[cam_idx]
            cam_extrinsic = extrinsics[cam_idx]

            annotated = project_boxes_to_image(
                boxes, cam_intrinsic, cam_extrinsic, cam_img
            )
            cam_file = str(output_path / f"{sample_name}_{camera_names[cam_idx]}.png")
            cv2.imwrite(cam_file, annotated)
            saved_files.append(cam_file)

        logger.info("Saved %d camera projections", num_cams)

    return saved_files


# ==============================================================================
# Throughput Benchmark
# ==============================================================================


def benchmark_throughput(
    model_path: str,
    config: Any,
    batch_size: int = 1,
    num_iterations: int = 100,
    warmup: int = 10,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Benchmark model throughput and latency.

    Runs the model repeatedly with synthetic data and measures timing statistics.

    Args:
        model_path: Path to the model checkpoint.
        config: Configuration dict or path to YAML.
        batch_size: Batch size for inference.
        num_iterations: Number of timed iterations.
        warmup: Number of warmup iterations (not timed).
        device: Device string.

    Returns:
        Dictionary with timing statistics:
            - mean_latency_ms: Mean latency per batch
            - median_latency_ms: Median latency per batch
            - p95_latency_ms: 95th percentile latency
            - p99_latency_ms: 99th percentile latency
            - throughput_fps: Throughput in frames per second (samples/sec)
            - batch_size: Batch size used
            - num_iterations: Number of timed iterations
            - device: Device used
    """
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device

    torch_device = torch.device(device_str)

    # Load config
    if isinstance(config, (str, Path)):
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config

    # Create inference engine
    engine = CRAFTInference(
        model_path=model_path,
        config=cfg,
        device=device_str,
    )

    # Generate synthetic data
    data_cfg = cfg.get("data", {})
    img_h = data_cfg.get("image_height", 256)
    img_w = data_cfg.get("image_width", 704)
    max_radar_points = data_cfg.get("max_radar_points", 500)

    dummy_images = torch.randn(batch_size, 6, 3, img_h, img_w, device=torch_device)
    dummy_radar = torch.randn(batch_size, max_radar_points, 6, device=torch_device)
    dummy_num_points = torch.full(
        (batch_size,), max_radar_points, dtype=torch.long, device=torch_device
    )

    batch_dict = {
        "images": dummy_images,
        "radar_points": dummy_radar,
        "num_points": dummy_num_points,
    }

    # Warmup
    logger.info("Running %d warmup iterations...", warmup)
    for _ in range(warmup):
        engine.predict_batch(batch_dict)

    # Timed iterations
    logger.info("Running %d timed iterations (batch_size=%d)...", num_iterations, batch_size)
    latencies: List[float] = []

    for i in range(num_iterations):
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

        start = time.perf_counter()
        engine.predict_batch(batch_dict)

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # ms

    # Compute statistics
    latencies_np = np.array(latencies)
    mean_latency = float(np.mean(latencies_np))
    median_latency = float(np.median(latencies_np))
    p95_latency = float(np.percentile(latencies_np, 95))
    p99_latency = float(np.percentile(latencies_np, 99))

    # Throughput: total samples / total time
    total_time_sec = float(np.sum(latencies_np)) / 1000.0
    total_samples = num_iterations * batch_size
    throughput_fps = total_samples / total_time_sec

    results = {
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "throughput_fps": throughput_fps,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "device": device_str,
    }

    # Print results
    print("\n" + "=" * 60)
    print("CRAFT Model Throughput Benchmark Results")
    print("=" * 60)
    print(f"  Device:             {device_str}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Iterations:         {num_iterations}")
    print(f"  Image size:         {img_h} x {img_w}")
    print(f"  Radar points:       {max_radar_points}")
    print("-" * 60)
    print(f"  Mean latency:       {mean_latency:.2f} ms")
    print(f"  Median latency:     {median_latency:.2f} ms")
    print(f"  P95 latency:        {p95_latency:.2f} ms")
    print(f"  P99 latency:        {p99_latency:.2f} ms")
    print(f"  Throughput:         {throughput_fps:.1f} FPS")
    print(f"  Per-sample latency: {mean_latency / batch_size:.2f} ms")
    print("=" * 60 + "\n")

    return results


# ==============================================================================
# TorchScript Export
# ==============================================================================


def export_torchscript(
    model_path: str,
    config: Any,
    output_dir: str,
    device: Optional[str] = None,
) -> str:
    """Export CRAFT model to TorchScript format.

    Traces the model with dummy inputs to produce a TorchScript module
    that can be loaded without Python.

    Args:
        model_path: Path to the model checkpoint.
        config: Configuration dict or path to YAML.
        output_dir: Directory to save the exported model.
        device: Device to use for tracing.

    Returns:
        Path to the exported TorchScript file.
    """
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device

    torch_device = torch.device(device_str)

    # Load config
    if isinstance(config, (str, Path)):
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config

    # Build and load model
    model = CRAFTModel(cfg)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value

    model.load_state_dict(cleaned, strict=False)
    model = model.to(torch_device)
    model.eval()

    # Create dummy inputs for tracing
    data_cfg = cfg.get("data", {})
    img_h = data_cfg.get("image_height", 256)
    img_w = data_cfg.get("image_width", 704)
    max_radar_points = data_cfg.get("max_radar_points", 500)

    dummy_images = torch.randn(1, 6, 3, img_h, img_w, device=torch_device)
    dummy_radar = torch.randn(1, max_radar_points, 6, device=torch_device)
    dummy_num_points = torch.tensor([max_radar_points], dtype=torch.long, device=torch_device)

    # Trace the model
    logger.info("Tracing model with TorchScript...")
    try:
        traced_model = torch.jit.trace(
            model,
            (dummy_images, dummy_radar, dummy_num_points),
            check_trace=False,
        )
    except Exception as e:
        logger.warning("Tracing failed, attempting scripting: %s", str(e))
        traced_model = torch.jit.script(model)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    export_file = str(output_path / "craft_model.pt")
    traced_model.save(export_file)

    # Verify by loading
    loaded = torch.jit.load(export_file, map_location=torch_device)
    with torch.no_grad():
        test_output = loaded(dummy_images, dummy_radar, dummy_num_points)
    assert "heatmap" in test_output, "Exported model output missing 'heatmap' key"

    file_size_mb = os.path.getsize(export_file) / (1024 * 1024)
    logger.info("Model exported to: %s (%.1f MB)", export_file, file_size_mb)
    print(f"\nExported TorchScript model: {export_file}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Device: {device_str}")
    print(f"  Input shapes: images=[1,6,3,{img_h},{img_w}], radar=[1,{max_radar_points},6]")

    return export_file


# ==============================================================================
# CLI
# ==============================================================================


def _create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="craft_inference",
        description="CRAFT Camera-Radar 3D Object Detection - Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single sample inference
  python inference.py infer --model checkpoint.pth --config config.yaml --input ./sample_dir --output ./results

  # Throughput benchmark
  python inference.py benchmark --model checkpoint.pth --config config.yaml --batch-size 4 --iterations 200

  # Export to TorchScript
  python inference.py export --model checkpoint.pth --config config.yaml --output-dir ./exported
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- infer subcommand ----
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on a single sample",
        description="Load sensor data from disk and run the CRAFT detection model.",
    )
    infer_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    infer_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    infer_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to sample directory (containing images/ and radar_points.npy)",
    )
    infer_parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory for results and visualizations (default: ./output)",
    )
    infer_parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum detection score threshold (default: 0.3)",
    )
    infer_parser.add_argument(
        "--max-detections",
        type=int,
        default=500,
        help="Maximum number of detections (default: 500)",
    )
    infer_parser.add_argument(
        "--nms-kernel",
        type=int,
        default=3,
        help="Heatmap NMS kernel size (default: 3)",
    )
    infer_parser.add_argument(
        "--circle-nms-radius",
        type=float,
        default=4.0,
        help="Circle NMS radius in meters (default: 4.0, 0 to disable)",
    )
    infer_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save visualization images",
    )
    infer_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (default: auto-detect, e.g., 'cuda:0' or 'cpu')",
    )

    # ---- benchmark subcommand ----
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Measure model throughput and latency",
        description="Run throughput benchmarks with synthetic data.",
    )
    bench_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    bench_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    bench_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarking (default: 1)",
    )
    bench_parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timed iterations (default: 100)",
    )
    bench_parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    bench_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for benchmarking (default: auto-detect)",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )

    # ---- export subcommand ----
    export_parser = subparsers.add_parser(
        "export",
        help="Export model to TorchScript",
        description="Trace or script the model to TorchScript format.",
    )
    export_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    export_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default="./exported",
        help="Directory to save exported model (default: ./exported)",
    )
    export_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for export tracing (default: auto-detect)",
    )

    return parser


def _run_infer(args: argparse.Namespace) -> None:
    """Execute the 'infer' subcommand."""
    logger.info("Starting inference...")

    engine = CRAFTInference(
        model_path=args.model,
        config=args.config,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        nms_kernel=args.nms_kernel,
        circle_nms_radius=args.circle_nms_radius,
        device=args.device,
    )

    # Run warmup
    engine._warmup(num_passes=3)

    # Run inference
    boxes, time_ms = engine.predict_from_files(args.input)

    # Print results
    print(f"\nInference Results:")
    print(f"  Input: {args.input}")
    print(f"  Time: {time_ms:.1f} ms")
    print(f"  Detections: {len(boxes)}")
    print()

    if len(boxes) > 0:
        print(f"  {'Class':<20} {'Score':>6} {'X':>7} {'Y':>7} {'Z':>7} {'W':>5} {'L':>5} {'H':>5}")
        print(f"  {'-'*20} {'-----':>6} {'------':>7} {'------':>7} {'------':>7} {'----':>5} {'----':>5} {'----':>5}")
        for box in boxes[:20]:  # Show top 20
            print(
                f"  {box.class_name:<20} {box.score:>6.3f} "
                f"{box.center_x:>7.2f} {box.center_y:>7.2f} {box.center_z:>7.2f} "
                f"{box.width:>5.2f} {box.length:>5.2f} {box.height:>5.2f}"
            )
        if len(boxes) > 20:
            print(f"  ... and {len(boxes) - 20} more detections")

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detections as JSON
    detections_file = str(output_path / "detections.json")
    detections_data = {
        "input": args.input,
        "inference_time_ms": time_ms,
        "num_detections": len(boxes),
        "score_threshold": args.score_threshold,
        "detections": [box.to_dict() for box in boxes],
    }
    with open(detections_file, "w") as f:
        json.dump(detections_data, f, indent=2)
    print(f"\n  Detections saved to: {detections_file}")

    # Generate visualizations if requested
    if args.visualize:
        saved_files = save_visualization(
            boxes=boxes,
            output_dir=args.output,
            sample_name="inference",
            bev_range=(-51.2, -51.2, 51.2, 51.2),
        )
        for f_path in saved_files:
            print(f"  Visualization saved: {f_path}")


def _run_benchmark(args: argparse.Namespace) -> None:
    """Execute the 'benchmark' subcommand."""
    results = benchmark_throughput(
        model_path=args.model,
        config=args.config,
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        warmup=args.warmup,
        device=args.device,
    )

    # Optionally save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


def _run_export(args: argparse.Namespace) -> None:
    """Execute the 'export' subcommand."""
    export_torchscript(
        model_path=args.model,
        config=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )


def main() -> None:
    """Main entry point for the CRAFT inference CLI."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "infer":
        _run_infer(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    elif args.command == "export":
        _run_export(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
