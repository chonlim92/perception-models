"""
CRAFT Evaluation Script.

Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer.

Computes the nuScenes Detection Score (NDS) and per-class metrics on the
validation set, including:
    - mAP at center-distance thresholds [0.5, 1.0, 2.0, 4.0] meters
    - mATE (mean Average Translation Error)
    - mASE (mean Average Scale Error)
    - mAOE (mean Average Orientation Error)
    - mAVE (mean Average Velocity Error)
    - mAAE (mean Average Attribute Error)
    - NDS = (5 * mAP + sum(max(1 - metric, 0) for 5 TP metrics)) / 10

Supports modality ablation: camera-only, radar-only, and fused inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup: allow importing from the same package directory (train.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Allow running from the craft project root as well
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# Import model components from sibling modules
from heads import CRAFTDetectionHead, decode_predictions, nms_bev
from fusion_transformer import SpatioContextualFusionTransformer, build_fusion_transformer
from camera_branch import MultiViewCameraBackbone  # type: ignore[import-not-found]
from radar_branch import RadarBranch, RadarBEVBackbone, RadarPillarEncoder  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("craft.evaluate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUSCENES_CLASSES: List[str] = [
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

AP_DISTANCE_THRESHOLDS: List[float] = [0.5, 1.0, 2.0, 4.0]

# Default BEV grid parameters
BEV_X_MIN: float = -51.2
BEV_X_MAX: float = 51.2
BEV_Y_MIN: float = -51.2
BEV_Y_MAX: float = 51.2
BEV_RESOLUTION: float = 0.2
BEV_SIZE: int = 512


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class Detection3D:
    """Single 3D detection result."""

    center: np.ndarray  # shape (3,): x, y, z in ego frame
    size: np.ndarray  # shape (3,): width, length, height
    yaw: float  # orientation in radians
    velocity: np.ndarray  # shape (2,): vx, vy in m/s
    score: float  # detection confidence
    class_id: int  # 0-indexed class label
    class_name: str  # human-readable class name
    attribute: str = ""  # nuScenes attribute (e.g. "vehicle.moving")


@dataclass
class GroundTruth3D:
    """Single ground truth 3D annotation."""

    center: np.ndarray  # shape (3,): x, y, z in ego frame
    size: np.ndarray  # shape (3,): width, length, height
    yaw: float  # orientation in radians
    velocity: np.ndarray  # shape (2,): vx, vy in m/s
    class_id: int  # 0-indexed class label
    class_name: str  # human-readable class name
    num_lidar_pts: int = 0  # for filtering
    num_radar_pts: int = 0  # for filtering
    attribute: str = ""


@dataclass
class ClassMetrics:
    """Per-class evaluation metrics."""

    class_name: str
    ap: float = 0.0  # average precision (mean over distance thresholds)
    ap_per_threshold: Dict[float, float] = field(default_factory=dict)
    ate: float = float("inf")  # average translation error (meters)
    ase: float = float("inf")  # average scale error (1 - IoU_3D)
    aoe: float = float("inf")  # average orientation error (radians)
    ave: float = float("inf")  # average velocity error (m/s)
    aae: float = float("inf")  # average attribute error (1 - accuracy)
    num_gt: int = 0
    num_pred: int = 0


@dataclass
class EvalResults:
    """Complete evaluation results."""

    mAP: float = 0.0
    mATE: float = 0.0
    mASE: float = 0.0
    mAOE: float = 0.0
    mAVE: float = 0.0
    mAAE: float = 0.0
    NDS: float = 0.0
    per_class: Dict[str, ClassMetrics] = field(default_factory=dict)
    total_gt: int = 0
    total_pred: int = 0
    eval_time_seconds: float = 0.0
    modality: str = "fused"


# ---------------------------------------------------------------------------
# CRAFT Model Wrapper (for evaluation)
# ---------------------------------------------------------------------------
class CRAFTModel(nn.Module):
    """CRAFT model that combines camera branch, radar branch, fusion transformer,
    and detection head.

    This is the evaluation-mode model wrapper. It mirrors the training definition
    and loads from the same checkpoint format.
    """

    def __init__(
        self,
        num_classes: int = 10,
        backbone_type: str = "resnet50",
        fpn_out_channels: int = 256,
        radar_in_channels: int = 6,
        radar_pillar_feat_channels: int = 128,
        radar_bev_channels: int = 256,
        fusion_d_model: int = 256,
        fusion_n_heads: int = 8,
        fusion_n_layers: int = 6,
        fusion_d_ffn: int = 1024,
        fusion_dropout: float = 0.1,
        head_in_channels: int = 256,
        head_shared_channels: int = 256,
        head_hidden_channels: int = 64,
        point_cloud_range: Optional[List[float]] = None,
        voxel_size: Optional[List[float]] = None,
        max_pillars: int = 30000,
        max_points_per_pillar: int = 20,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        if point_cloud_range is None:
            point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if voxel_size is None:
            voxel_size = [0.2, 0.2, 8.0]

        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        # Camera branch (MultiViewCameraBackbone: ResNet + FPN)
        self.camera_branch = MultiViewCameraBackbone(
            backbone_name=backbone_type,
            fpn_out_channels=fpn_out_channels,
            pretrained=False,  # will load from checkpoint
        )

        # Radar branch (RadarBranch: PillarEncoder + BEV Backbone)
        self.radar_branch = RadarBranch(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_points_per_pillar=max_points_per_pillar,
            max_num_pillars=max_pillars,
            in_channels=radar_in_channels,
            pillar_feat_channels=radar_pillar_feat_channels,
            bev_out_channels=radar_bev_channels,
        )

        # Fusion transformer
        self.fusion_transformer = build_fusion_transformer(
            d_model=fusion_d_model,
            n_heads=fusion_n_heads,
            n_layers=fusion_n_layers,
            d_ffn=fusion_d_ffn,
            dropout=fusion_dropout,
            radar_channels=radar_bev_channels,
            camera_channels=fpn_out_channels,
        )

        # Detection head
        self.detection_head = CRAFTDetectionHead(
            in_channels=head_in_channels,
            shared_channels=head_shared_channels,
            num_classes=num_classes,
            head_hidden_channels=head_hidden_channels,
        )

    def forward(
        self,
        images: torch.Tensor,
        radar_points: torch.Tensor,
        num_radar_points: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_shape: Tuple[int, int] = (900, 1600),
        radar_properties: Optional[torch.Tensor] = None,
        modality: str = "fused",
    ) -> Dict[str, torch.Tensor]:
        """Forward inference pass.

        Args:
            images: Multi-view camera images [B, N_cams, 3, H, W].
            radar_points: Radar point clouds [B, N_max, C] (x, y, z, vx, vy, rcs).
            num_radar_points: Number of valid radar points per sample [B].
            intrinsics: Camera intrinsics [B, N_cams, 3, 3].
            extrinsics: Camera extrinsics [B, N_cams, 4, 4].
            image_shape: Original image size (H, W).
            radar_properties: Optional radar properties [B, H*W, 4].
            modality: One of "fused", "camera_only", "radar_only".

        Returns:
            Detection head outputs: heatmap, regression, velocity.
        """
        B = images.shape[0]
        N_cams = images.shape[1]

        # Camera features
        if modality in ("fused", "camera_only"):
            # MultiViewCameraBackbone returns {'features': [P2, P3, P4, P5]}
            # Each Pi has shape [B, N_cams, C, H_i, W_i]
            camera_out = self.camera_branch(images)
            # Use the highest-resolution feature (P2) for fusion
            camera_features = camera_out["features"][0]  # [B, N_cams, C, H_feat, W_feat]
        else:
            # Zero out camera features for radar-only
            camera_features = torch.zeros(
                B, N_cams, 256, image_shape[0] // 4, image_shape[1] // 4,
                device=images.device, dtype=images.dtype,
            )

        # Radar features
        if modality in ("fused", "radar_only"):
            # RadarBranch returns {'bev_features': [B, C, H, W], 'pillar_coords': ...}
            radar_out = self.radar_branch(radar_points, num_radar_points)
            radar_bev = radar_out["bev_features"]  # [B, C, H_bev, W_bev]
        else:
            # Zero out radar features for camera-only
            radar_bev = torch.zeros(
                B, 256, BEV_SIZE, BEV_SIZE,
                device=images.device, dtype=images.dtype,
            )

        # Fusion
        fused_bev = self.fusion_transformer(
            radar_bev_features=radar_bev,
            camera_features=camera_features,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_shape=image_shape,
            radar_properties=radar_properties,
        )

        # Detection head
        outputs = self.detection_head(fused_bev)
        return outputs


# ---------------------------------------------------------------------------
# EMA Model Wrapper
# ---------------------------------------------------------------------------
class EMAModel:
    """Exponential Moving Average wrapper that stores shadow parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def apply_shadow(self) -> None:
        """Apply EMA shadow weights to the model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
# Post-Processing: Decode Heatmap Predictions
# ---------------------------------------------------------------------------
def decode_model_predictions(
    head_outputs: Dict[str, torch.Tensor],
    score_threshold: float = 0.1,
    max_detections: int = 500,
    nms_kernel_size: int = 3,
    nms_iou_threshold: float = 0.2,
    voxel_size: float = BEV_RESOLUTION,
    x_min: float = BEV_X_MIN,
    y_min: float = BEV_Y_MIN,
) -> List[List[Detection3D]]:
    """Decode raw network outputs into Detection3D objects.

    Performs max-pool NMS on heatmap, extracts top-K peaks, gathers regression
    and velocity at those locations, and applies per-class BEV NMS.

    Args:
        head_outputs: Dict with 'heatmap' [B, C, H, W], 'regression' [B, 8, H, W],
            'velocity' [B, 2, H, W].
        score_threshold: Minimum confidence to keep a detection.
        max_detections: Maximum number of detections per sample.
        nms_kernel_size: Kernel size for heatmap max-pooling NMS.
        nms_iou_threshold: IoU threshold for BEV NMS.
        voxel_size: BEV grid resolution in meters.
        x_min: BEV grid x-axis minimum in meters.
        y_min: BEV grid y-axis minimum in meters.

    Returns:
        List of detection lists (one per batch sample). Each detection is a
        Detection3D dataclass.
    """
    heatmap = head_outputs["heatmap"]  # [B, C, H, W] (already sigmoid)
    regression = head_outputs["regression"]  # [B, 8, H, W]
    velocity = head_outputs["velocity"]  # [B, 2, H, W]

    batch_size, num_classes, height, width = heatmap.shape
    device = heatmap.device

    # Max-pool NMS on heatmap
    padding = nms_kernel_size // 2
    heatmap_pool = F.max_pool2d(
        heatmap, kernel_size=nms_kernel_size, stride=1, padding=padding
    )
    heatmap_nms = heatmap * (heatmap_pool == heatmap).float()

    all_detections: List[List[Detection3D]] = []

    for b in range(batch_size):
        sample_dets: List[Detection3D] = []

        # Flatten across classes and spatial dims
        heatmap_flat = heatmap_nms[b].view(-1)  # [C * H * W]
        num_candidates = min(max_detections * 2, heatmap_flat.shape[0])
        topk_scores, topk_inds = torch.topk(heatmap_flat, num_candidates)

        # Filter by score threshold
        valid_mask = topk_scores >= score_threshold
        topk_scores = topk_scores[valid_mask]
        topk_inds = topk_inds[valid_mask]

        if topk_scores.numel() == 0:
            all_detections.append(sample_dets)
            continue

        # Decode indices: class, row, col
        topk_classes = topk_inds // (height * width)
        spatial_inds = topk_inds % (height * width)
        topk_rows = spatial_inds // width
        topk_cols = spatial_inds % width

        # Gather regression and velocity
        reg_flat = regression[b].view(8, -1)  # [8, H*W]
        reg_vals = reg_flat[:, spatial_inds].T  # [N, 8]

        vel_flat = velocity[b].view(2, -1)  # [2, H*W]
        vel_vals = vel_flat[:, spatial_inds].T  # [N, 2]

        # Decode to absolute coordinates
        offset_x = reg_vals[:, 0]
        offset_y = reg_vals[:, 1]
        center_z = reg_vals[:, 2]
        log_w = reg_vals[:, 3]
        log_l = reg_vals[:, 4]
        log_h = reg_vals[:, 5]
        sin_yaw = reg_vals[:, 6]
        cos_yaw = reg_vals[:, 7]

        cx = (topk_cols.float() + offset_x) * voxel_size + x_min
        cy = (topk_rows.float() + offset_y) * voxel_size + y_min
        cz = center_z
        w = torch.exp(log_w)
        l = torch.exp(log_l)
        h = torch.exp(log_h)
        yaw = torch.atan2(sin_yaw, cos_yaw)

        # Build boxes for NMS: [N, 7]
        boxes_3d = torch.stack([cx, cy, cz, w, l, h, yaw], dim=1)
        scores_tensor = topk_scores
        labels_tensor = topk_classes

        # Per-class NMS
        kept_indices: List[int] = []
        for cls_id in range(num_classes):
            cls_mask = labels_tensor == cls_id
            if cls_mask.sum() == 0:
                continue
            cls_indices = torch.where(cls_mask)[0]
            cls_boxes = boxes_3d[cls_mask]
            cls_scores = scores_tensor[cls_mask]

            nms_keep = nms_bev(cls_boxes, cls_scores, iou_threshold=nms_iou_threshold)
            kept_indices.extend(cls_indices[nms_keep].cpu().tolist())

        # Sort by score and limit
        if not kept_indices:
            all_detections.append(sample_dets)
            continue

        kept_indices_tensor = torch.tensor(kept_indices, device=device, dtype=torch.long)
        kept_scores = scores_tensor[kept_indices_tensor]
        sort_order = torch.argsort(kept_scores, descending=True)
        kept_indices_tensor = kept_indices_tensor[sort_order]

        # Limit to max_detections
        if kept_indices_tensor.shape[0] > max_detections:
            kept_indices_tensor = kept_indices_tensor[:max_detections]

        # Convert to Detection3D
        for idx in kept_indices_tensor.cpu().tolist():
            det = Detection3D(
                center=np.array([
                    cx[idx].item(), cy[idx].item(), cz[idx].item()
                ], dtype=np.float64),
                size=np.array([
                    w[idx].item(), l[idx].item(), h[idx].item()
                ], dtype=np.float64),
                yaw=yaw[idx].item(),
                velocity=np.array([
                    vel_vals[idx, 0].item(), vel_vals[idx, 1].item()
                ], dtype=np.float64),
                score=scores_tensor[idx].item(),
                class_id=labels_tensor[idx].item(),
                class_name=NUSCENES_CLASSES[labels_tensor[idx].item()],
            )
            sample_dets.append(det)

        all_detections.append(sample_dets)

    return all_detections


# ---------------------------------------------------------------------------
# Metric Computation Functions
# ---------------------------------------------------------------------------
def _center_distance(det: Detection3D, gt: GroundTruth3D) -> float:
    """Compute Euclidean center distance in BEV (x-y plane)."""
    dx = det.center[0] - gt.center[0]
    dy = det.center[1] - gt.center[1]
    return math.sqrt(dx * dx + dy * dy)


def _translation_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """Compute 3D translation error (Euclidean distance in 3D)."""
    diff = det.center - gt.center
    return float(np.linalg.norm(diff))


def _scale_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """Compute scale error as 1 - 3D IoU approximation.

    Uses the volume ratio approximation (min/max of volumes) as a proxy
    for 3D IoU since exact rotated 3D IoU is expensive to compute.
    """
    det_vol = float(np.prod(det.size))
    gt_vol = float(np.prod(gt.size))
    if det_vol <= 0 or gt_vol <= 0:
        return 1.0
    # Approximate IoU via min(vol) / max(vol)
    iou_approx = min(det_vol, gt_vol) / max(det_vol, gt_vol)
    return 1.0 - iou_approx


def _yaw_diff(yaw1: float, yaw2: float) -> float:
    """Compute the absolute angular difference, wrapped to [0, pi]."""
    diff = abs(yaw1 - yaw2)
    # Wrap to [0, 2*pi] then take minimum with pi complement
    diff = diff % (2.0 * math.pi)
    if diff > math.pi:
        diff = 2.0 * math.pi - diff
    return diff


def _orientation_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """Compute orientation error (angle difference in radians)."""
    return _yaw_diff(det.yaw, gt.yaw)


def _velocity_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """Compute velocity error (L2 distance between velocity vectors)."""
    diff = det.velocity - gt.velocity
    return float(np.linalg.norm(diff))


def _attribute_error(det: Detection3D, gt: GroundTruth3D) -> float:
    """Compute attribute classification error (0 if match, 1 if mismatch).

    If attributes are not available, returns 1.0 (worst case).
    """
    if not det.attribute or not gt.attribute:
        return 1.0
    return 0.0 if det.attribute == gt.attribute else 1.0


def compute_ap_single_class(
    detections: List[Detection3D],
    ground_truths: List[GroundTruth3D],
    distance_threshold: float,
) -> Tuple[float, List[Tuple[Detection3D, GroundTruth3D]]]:
    """Compute Average Precision for a single class at a single distance threshold.

    Uses the nuScenes matching criterion: a detection is a true positive if its
    BEV center distance to the closest unmatched ground truth is below the
    threshold. Detections are sorted by confidence (descending).

    Args:
        detections: All detections for this class, across all samples.
        ground_truths: All ground truths for this class, across all samples.
        distance_threshold: Maximum center distance for a true positive match.

    Returns:
        Tuple of (AP value, list of matched (detection, ground_truth) pairs for
        TP error computation).
    """
    if len(ground_truths) == 0:
        return 0.0, []

    if len(detections) == 0:
        return 0.0, []

    # Sort detections by score (descending)
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)

    # Track which GTs have been matched
    gt_matched = [False] * len(ground_truths)
    tp_list: List[bool] = []
    matched_pairs: List[Tuple[Detection3D, GroundTruth3D]] = []

    for det in sorted_dets:
        # Find closest unmatched GT
        best_dist = float("inf")
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            dist = _center_distance(det, gt)
            if dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx

        if best_dist <= distance_threshold and best_gt_idx >= 0:
            tp_list.append(True)
            gt_matched[best_gt_idx] = True
            matched_pairs.append((det, ground_truths[best_gt_idx]))
        else:
            tp_list.append(False)

    # Compute precision-recall curve
    num_gt = len(ground_truths)
    tp_cumsum = np.cumsum(tp_list).astype(np.float64)
    fp_cumsum = np.cumsum([not tp for tp in tp_list]).astype(np.float64)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / num_gt

    # Compute AP using all-points interpolation (nuScenes style)
    # Append sentinel values
    recalls_interp = np.concatenate([[0.0], recalls, [1.0]])
    precisions_interp = np.concatenate([[1.0], precisions, [0.0]])

    # Make precision monotonically decreasing (right to left)
    for i in range(len(precisions_interp) - 2, -1, -1):
        precisions_interp[i] = max(precisions_interp[i], precisions_interp[i + 1])

    # Find points where recall changes
    recall_change_indices = np.where(
        recalls_interp[1:] != recalls_interp[:-1]
    )[0]

    # Sum rectangular areas under the PR curve
    ap = float(np.sum(
        (recalls_interp[recall_change_indices + 1] - recalls_interp[recall_change_indices])
        * precisions_interp[recall_change_indices + 1]
    ))

    return ap, matched_pairs


def compute_ap(
    all_detections: List[List[Detection3D]],
    all_ground_truths: List[List[GroundTruth3D]],
    class_id: int,
    distance_thresholds: List[float] = AP_DISTANCE_THRESHOLDS,
) -> ClassMetrics:
    """Compute per-class AP at multiple distance thresholds and TP error metrics.

    Args:
        all_detections: List of detection lists (one per sample).
        all_ground_truths: List of ground truth lists (one per sample).
        class_id: Class index to evaluate.
        distance_thresholds: List of center distance thresholds.

    Returns:
        ClassMetrics with AP and TP error metrics for this class.
    """
    class_name = NUSCENES_CLASSES[class_id]

    # Collect all detections and GTs for this class across samples
    class_dets: List[Detection3D] = []
    class_gts: List[GroundTruth3D] = []

    for sample_dets in all_detections:
        for det in sample_dets:
            if det.class_id == class_id:
                class_dets.append(det)

    for sample_gts in all_ground_truths:
        for gt in sample_gts:
            if gt.class_id == class_id:
                class_gts.append(gt)

    metrics = ClassMetrics(
        class_name=class_name,
        num_gt=len(class_gts),
        num_pred=len(class_dets),
    )

    if len(class_gts) == 0:
        metrics.ap = 0.0
        metrics.ap_per_threshold = {t: 0.0 for t in distance_thresholds}
        return metrics

    # Compute AP at each distance threshold
    all_matched_pairs: List[Tuple[Detection3D, GroundTruth3D]] = []
    ap_values: Dict[float, float] = {}

    for threshold in distance_thresholds:
        ap, matched = compute_ap_single_class(class_dets, class_gts, threshold)
        ap_values[threshold] = ap
        # Use matches from the most permissive threshold for TP errors
        if threshold == max(distance_thresholds):
            all_matched_pairs = matched

    metrics.ap = float(np.mean(list(ap_values.values())))
    metrics.ap_per_threshold = ap_values

    # Compute TP error metrics from matched pairs at the largest threshold
    if len(all_matched_pairs) > 0:
        translation_errors: List[float] = []
        scale_errors: List[float] = []
        orientation_errors: List[float] = []
        velocity_errors: List[float] = []
        attribute_errors: List[float] = []

        for det, gt in all_matched_pairs:
            translation_errors.append(_translation_error(det, gt))
            scale_errors.append(_scale_error(det, gt))
            orientation_errors.append(_orientation_error(det, gt))
            velocity_errors.append(_velocity_error(det, gt))
            attribute_errors.append(_attribute_error(det, gt))

        metrics.ate = float(np.mean(translation_errors))
        metrics.ase = float(np.mean(scale_errors))
        metrics.aoe = float(np.mean(orientation_errors))
        metrics.ave = float(np.mean(velocity_errors))
        metrics.aae = float(np.mean(attribute_errors))

    return metrics


def compute_nds(per_class_metrics: Dict[str, ClassMetrics]) -> EvalResults:
    """Compute the nuScenes Detection Score (NDS) from per-class metrics.

    NDS = (5 * mAP + sum(max(1 - metric, 0) for each TP metric)) / 10

    The five TP metrics are: mATE, mASE, mAOE, mAVE, mAAE.

    Args:
        per_class_metrics: Dictionary mapping class names to ClassMetrics.

    Returns:
        EvalResults with aggregate metrics.
    """
    results = EvalResults()
    results.per_class = per_class_metrics

    # Compute mAP (mean over classes)
    ap_values: List[float] = []
    ate_values: List[float] = []
    ase_values: List[float] = []
    aoe_values: List[float] = []
    ave_values: List[float] = []
    aae_values: List[float] = []

    total_gt = 0
    total_pred = 0

    for cls_name, cls_metrics in per_class_metrics.items():
        ap_values.append(cls_metrics.ap)
        total_gt += cls_metrics.num_gt
        total_pred += cls_metrics.num_pred

        # Only include TP metrics for classes that have matched TPs
        if cls_metrics.ate < float("inf"):
            ate_values.append(cls_metrics.ate)
        if cls_metrics.ase < float("inf"):
            ase_values.append(cls_metrics.ase)
        if cls_metrics.aoe < float("inf"):
            aoe_values.append(cls_metrics.aoe)
        if cls_metrics.ave < float("inf"):
            ave_values.append(cls_metrics.ave)
        if cls_metrics.aae < float("inf"):
            aae_values.append(cls_metrics.aae)

    results.mAP = float(np.mean(ap_values)) if ap_values else 0.0
    results.mATE = float(np.mean(ate_values)) if ate_values else 1.0
    results.mASE = float(np.mean(ase_values)) if ase_values else 1.0
    results.mAOE = float(np.mean(aoe_values)) if aoe_values else 1.0
    results.mAVE = float(np.mean(ave_values)) if ave_values else 1.0
    results.mAAE = float(np.mean(aae_values)) if aae_values else 1.0
    results.total_gt = total_gt
    results.total_pred = total_pred

    # NDS formula: (5 * mAP + sum(max(1 - metric, 0))) / 10
    tp_scores = (
        max(1.0 - results.mATE, 0.0)
        + max(1.0 - results.mASE, 0.0)
        + max(1.0 - results.mAOE, 0.0)
        + max(1.0 - results.mAVE, 0.0)
        + max(1.0 - results.mAAE, 0.0)
    )
    results.NDS = (5.0 * results.mAP + tp_scores) / 10.0

    return results


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed. Install with: pip install pyyaml")
        raise

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model_from_config(config: Dict[str, Any], device: torch.device) -> CRAFTModel:
    """Build CRAFT model from configuration dictionary.

    Args:
        config: Configuration dictionary (from YAML).
        device: Target device.

    Returns:
        CRAFTModel instance moved to the specified device.
    """
    model_cfg = config.get("model", {})
    head_cfg = model_cfg.get("detection_head", {})
    fusion_cfg = model_cfg.get("fusion_transformer", {})
    radar_cfg = model_cfg.get("radar_pillar_encoder", {})
    backbone_cfg = model_cfg.get("backbone", {})

    model = CRAFTModel(
        num_classes=head_cfg.get("num_classes", 10),
        backbone_type=backbone_cfg.get("type", "resnet50"),
        fpn_out_channels=model_cfg.get("neck", {}).get("out_channels", 256),
        radar_in_channels=radar_cfg.get("in_channels", 6),
        radar_pillar_feat_channels=128,
        radar_bev_channels=256,
        fusion_d_model=fusion_cfg.get("d_model", 256),
        fusion_n_heads=fusion_cfg.get("n_heads", 8),
        fusion_n_layers=fusion_cfg.get("n_layers", 6),
        fusion_d_ffn=fusion_cfg.get("d_ffn", 1024),
        fusion_dropout=fusion_cfg.get("dropout", 0.1),
        head_in_channels=head_cfg.get("in_channels", 256),
        head_shared_channels=head_cfg.get("in_channels", 256),
        head_hidden_channels=head_cfg.get("shared_conv_channels", 64),
        point_cloud_range=radar_cfg.get(
            "point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        ),
        voxel_size=radar_cfg.get("voxel_size", [0.2, 0.2, 8.0]),
        max_pillars=radar_cfg.get("max_pillars", 30000),
        max_points_per_pillar=radar_cfg.get("max_points_per_pillar", 20),
    )

    model = model.to(device)
    return model


def load_checkpoint(
    model: CRAFTModel,
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> CRAFTModel:
    """Load model weights from checkpoint.

    Supports both regular and EMA checkpoints. If EMA weights are available
    and use_ema=True, the EMA weights are loaded.

    Args:
        model: CRAFTModel instance to load weights into.
        checkpoint_path: Path to the checkpoint file.
        device: Target device.
        use_ema: Whether to use EMA weights if available.

    Returns:
        Model with loaded weights.
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if use_ema and "ema_state_dict" in checkpoint:
            state_dict = checkpoint["ema_state_dict"]
            logger.info("Using EMA weights from checkpoint")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Report checkpoint metadata if available
        if "epoch" in checkpoint:
            logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if "best_metric" in checkpoint:
            logger.info(f"  Best metric: {checkpoint['best_metric']:.4f}")
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if model was saved with DataParallel/DDP
    cleaned_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "") if key.startswith("module.") else key
        cleaned_state_dict[clean_key] = value

    # Load with strict=False to handle minor mismatches gracefully
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        logger.warning(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    logger.info("Checkpoint loaded successfully")
    return model


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for variable-size radar data.

    Handles padding of radar point clouds and stacking of fixed-size tensors.

    Args:
        batch: List of sample dictionaries from the dataset.

    Returns:
        Batched dictionary with padded tensors.
    """
    keys = batch[0].keys()
    collated: Dict[str, Any] = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        if isinstance(values[0], torch.Tensor):
            # Attempt to stack; if shapes differ, pad to maximum
            shapes = [v.shape for v in values]
            if all(s == shapes[0] for s in shapes):
                collated[key] = torch.stack(values, dim=0)
            else:
                # Pad to maximum size along each dimension
                max_shape = [max(s[d] for s in shapes) for d in range(len(shapes[0]))]
                padded = []
                for v in values:
                    pad_widths: List[int] = []
                    for d in range(len(max_shape) - 1, -1, -1):
                        pad_widths.extend([0, max_shape[d] - v.shape[d]])
                    padded.append(F.pad(v, pad_widths))
                collated[key] = torch.stack(padded, dim=0)
        elif isinstance(values[0], np.ndarray):
            collated[key] = np.stack(values, axis=0)
        elif isinstance(values[0], (list, tuple)):
            collated[key] = values  # Keep as list of lists
        else:
            collated[key] = values

    return collated


def run_evaluation(
    model: CRAFTModel,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.1,
    max_detections: int = 500,
    nms_iou_threshold: float = 0.2,
    modality: str = "fused",
    class_names: List[str] = NUSCENES_CLASSES,
) -> EvalResults:
    """Run full evaluation on the validation set.

    Iterates over the dataloader, runs inference, decodes predictions,
    collects ground truths, and computes NDS metrics.

    Args:
        model: CRAFT model in eval mode.
        dataloader: Validation DataLoader.
        device: Compute device.
        score_threshold: Detection confidence threshold.
        max_detections: Maximum detections per sample.
        nms_iou_threshold: BEV NMS IoU threshold.
        modality: One of "fused", "camera_only", "radar_only".
        class_names: List of class names.

    Returns:
        EvalResults with complete metrics.
    """
    model.eval()
    logger.info(f"Running evaluation in '{modality}' mode...")
    logger.info(f"  Score threshold: {score_threshold}")
    logger.info(f"  Max detections: {max_detections}")
    logger.info(f"  NMS IoU threshold: {nms_iou_threshold}")

    all_detections: List[List[Detection3D]] = []
    all_ground_truths: List[List[GroundTruth3D]] = []

    start_time = time.time()
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move inputs to device
            images = batch["images"].to(device)  # [B, N_cams, 3, H, W]
            radar_points = batch["radar_points"].to(device)
            num_radar_points = batch["num_radar_points"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            extrinsics = batch["extrinsics"].to(device)

            image_shape = (
                batch.get("image_height", 900),
                batch.get("image_width", 1600),
            )
            if isinstance(image_shape[0], torch.Tensor):
                image_shape = (image_shape[0].item(), image_shape[1].item())

            radar_properties = batch.get("radar_properties", None)
            if radar_properties is not None:
                radar_properties = radar_properties.to(device)

            # Forward pass
            head_outputs = model(
                images=images,
                radar_points=radar_points,
                num_radar_points=num_radar_points,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_shape=image_shape,
                radar_properties=radar_properties,
                modality=modality,
            )

            # Decode predictions
            batch_detections = decode_model_predictions(
                head_outputs=head_outputs,
                score_threshold=score_threshold,
                max_detections=max_detections,
                nms_iou_threshold=nms_iou_threshold,
            )
            all_detections.extend(batch_detections)

            # Extract ground truths
            gt_boxes_batch = batch.get("gt_boxes", batch.get("gt_bboxes", []))
            gt_labels_batch = batch.get("gt_labels", [])

            batch_size = images.shape[0]
            for sample_idx in range(batch_size):
                sample_gts: List[GroundTruth3D] = []

                if isinstance(gt_boxes_batch, torch.Tensor):
                    gt_boxes = gt_boxes_batch[sample_idx]
                    gt_labels = gt_labels_batch[sample_idx]
                elif isinstance(gt_boxes_batch, (list, tuple)) and len(gt_boxes_batch) > sample_idx:
                    gt_boxes = gt_boxes_batch[sample_idx]
                    gt_labels = gt_labels_batch[sample_idx]
                else:
                    gt_boxes = torch.zeros((0, 10))
                    gt_labels = torch.zeros((0,), dtype=torch.long)

                if isinstance(gt_boxes, torch.Tensor):
                    gt_boxes_np = gt_boxes.cpu().numpy()
                    gt_labels_np = gt_labels.cpu().numpy()
                else:
                    gt_boxes_np = np.array(gt_boxes)
                    gt_labels_np = np.array(gt_labels)

                for obj_idx in range(gt_boxes_np.shape[0]):
                    box = gt_boxes_np[obj_idx]
                    label = int(gt_labels_np[obj_idx])

                    # box format: [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy]
                    yaw = math.atan2(float(box[6]), float(box[7]))
                    gt_obj = GroundTruth3D(
                        center=np.array([box[0], box[1], box[2]], dtype=np.float64),
                        size=np.array([box[3], box[4], box[5]], dtype=np.float64),
                        yaw=yaw,
                        velocity=np.array([box[8], box[9]], dtype=np.float64),
                        class_id=label,
                        class_name=class_names[label] if label < len(class_names) else "unknown",
                    )
                    sample_gts.append(gt_obj)

                all_ground_truths.append(sample_gts)

            num_samples += batch_size

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"  Processed {num_samples} samples "
                    f"({elapsed:.1f}s, {num_samples / elapsed:.1f} samples/s)"
                )

    elapsed_total = time.time() - start_time
    logger.info(
        f"Inference complete: {num_samples} samples in {elapsed_total:.1f}s "
        f"({num_samples / max(elapsed_total, 1e-6):.1f} samples/s)"
    )

    # Compute per-class metrics
    logger.info("Computing per-class metrics...")
    per_class_metrics: Dict[str, ClassMetrics] = {}

    for cls_id, cls_name in enumerate(class_names):
        cls_metrics = compute_ap(
            all_detections=all_detections,
            all_ground_truths=all_ground_truths,
            class_id=cls_id,
        )
        per_class_metrics[cls_name] = cls_metrics

    # Compute NDS
    results = compute_nds(per_class_metrics)
    results.eval_time_seconds = elapsed_total
    results.modality = modality

    return results


def run_modality_ablation(
    model: CRAFTModel,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.1,
    max_detections: int = 500,
    nms_iou_threshold: float = 0.2,
    class_names: List[str] = NUSCENES_CLASSES,
) -> Dict[str, EvalResults]:
    """Run modality ablation study: evaluate camera-only, radar-only, and fused.

    Args:
        model: CRAFT model in eval mode.
        dataloader: Validation DataLoader.
        device: Compute device.
        score_threshold: Detection confidence threshold.
        max_detections: Maximum detections per sample.
        nms_iou_threshold: BEV NMS IoU threshold.
        class_names: List of class names.

    Returns:
        Dictionary mapping modality name to EvalResults.
    """
    modalities = ["camera_only", "radar_only", "fused"]
    ablation_results: Dict[str, EvalResults] = {}

    for modality in modalities:
        logger.info(f"\n{'='*60}")
        logger.info(f"MODALITY ABLATION: {modality.upper()}")
        logger.info(f"{'='*60}")

        results = run_evaluation(
            model=model,
            dataloader=dataloader,
            device=device,
            score_threshold=score_threshold,
            max_detections=max_detections,
            nms_iou_threshold=nms_iou_threshold,
            modality=modality,
            class_names=class_names,
        )
        ablation_results[modality] = results

        logger.info(f"  mAP: {results.mAP:.4f}")
        logger.info(f"  NDS: {results.NDS:.4f}")
        logger.info(f"  mATE: {results.mATE:.4f}")
        logger.info(f"  mASE: {results.mASE:.4f}")
        logger.info(f"  mAOE: {results.mAOE:.4f}")
        logger.info(f"  mAVE: {results.mAVE:.4f}")
        logger.info(f"  mAAE: {results.mAAE:.4f}")

    # Print comparison table
    logger.info(f"\n{'='*60}")
    logger.info("MODALITY ABLATION SUMMARY")
    logger.info(f"{'='*60}")
    header = f"{'Modality':<15} {'mAP':>7} {'NDS':>7} {'mATE':>7} {'mASE':>7} {'mAOE':>7} {'mAVE':>7} {'mAAE':>7}"
    logger.info(header)
    logger.info("-" * len(header))
    for modality, results in ablation_results.items():
        row = (
            f"{modality:<15} "
            f"{results.mAP:>7.4f} "
            f"{results.NDS:>7.4f} "
            f"{results.mATE:>7.4f} "
            f"{results.mASE:>7.4f} "
            f"{results.mAOE:>7.4f} "
            f"{results.mAVE:>7.4f} "
            f"{results.mAAE:>7.4f}"
        )
        logger.info(row)

    return ablation_results


# ---------------------------------------------------------------------------
# Results Formatting and Output
# ---------------------------------------------------------------------------
def format_results_table(results: EvalResults) -> str:
    """Format evaluation results as a human-readable table.

    Args:
        results: EvalResults to format.

    Returns:
        Multi-line formatted string.
    """
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f"CRAFT Evaluation Results ({results.modality} modality)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Overall Metrics:")
    lines.append(f"  NDS:  {results.NDS:.4f}")
    lines.append(f"  mAP:  {results.mAP:.4f}")
    lines.append(f"  mATE: {results.mATE:.4f} m")
    lines.append(f"  mASE: {results.mASE:.4f}")
    lines.append(f"  mAOE: {results.mAOE:.4f} rad")
    lines.append(f"  mAVE: {results.mAVE:.4f} m/s")
    lines.append(f"  mAAE: {results.mAAE:.4f}")
    lines.append("")
    lines.append(f"  Total GT objects:   {results.total_gt}")
    lines.append(f"  Total predictions:  {results.total_pred}")
    lines.append(f"  Eval time:          {results.eval_time_seconds:.1f}s")
    lines.append("")

    # Per-class breakdown
    lines.append("-" * 70)
    header = (
        f"{'Class':<22} {'AP':>6} {'ATE':>6} {'ASE':>6} "
        f"{'AOE':>6} {'AVE':>6} {'AAE':>6} {'#GT':>5} {'#Pred':>6}"
    )
    lines.append(header)
    lines.append("-" * 70)

    for cls_name, cls_metrics in results.per_class.items():
        ate_str = f"{cls_metrics.ate:.3f}" if cls_metrics.ate < float("inf") else "  N/A"
        ase_str = f"{cls_metrics.ase:.3f}" if cls_metrics.ase < float("inf") else "  N/A"
        aoe_str = f"{cls_metrics.aoe:.3f}" if cls_metrics.aoe < float("inf") else "  N/A"
        ave_str = f"{cls_metrics.ave:.3f}" if cls_metrics.ave < float("inf") else "  N/A"
        aae_str = f"{cls_metrics.aae:.3f}" if cls_metrics.aae < float("inf") else "  N/A"

        row = (
            f"{cls_name:<22} "
            f"{cls_metrics.ap:>6.3f} "
            f"{ate_str:>6} "
            f"{ase_str:>6} "
            f"{aoe_str:>6} "
            f"{ave_str:>6} "
            f"{aae_str:>6} "
            f"{cls_metrics.num_gt:>5d} "
            f"{cls_metrics.num_pred:>6d}"
        )
        lines.append(row)

    lines.append("-" * 70)

    # AP per distance threshold
    lines.append("")
    lines.append("AP per Distance Threshold:")
    thresh_header = f"{'Class':<22}" + "".join(f" {t:>5.1f}m" for t in AP_DISTANCE_THRESHOLDS)
    lines.append(thresh_header)
    lines.append("-" * (22 + 7 * len(AP_DISTANCE_THRESHOLDS)))

    for cls_name, cls_metrics in results.per_class.items():
        vals = "".join(
            f" {cls_metrics.ap_per_threshold.get(t, 0.0):>6.3f}"
            for t in AP_DISTANCE_THRESHOLDS
        )
        lines.append(f"{cls_name:<22}{vals}")

    lines.append("=" * 70)
    return "\n".join(lines)


def results_to_dict(results: EvalResults) -> Dict[str, Any]:
    """Convert EvalResults to a JSON-serializable dictionary.

    Args:
        results: EvalResults to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    output: Dict[str, Any] = {
        "overall": {
            "NDS": results.NDS,
            "mAP": results.mAP,
            "mATE": results.mATE,
            "mASE": results.mASE,
            "mAOE": results.mAOE,
            "mAVE": results.mAVE,
            "mAAE": results.mAAE,
        },
        "metadata": {
            "modality": results.modality,
            "total_gt": results.total_gt,
            "total_pred": results.total_pred,
            "eval_time_seconds": results.eval_time_seconds,
        },
        "per_class": {},
    }

    for cls_name, cls_metrics in results.per_class.items():
        cls_dict: Dict[str, Any] = {
            "AP": cls_metrics.ap,
            "AP_per_threshold": {
                str(k): v for k, v in cls_metrics.ap_per_threshold.items()
            },
            "ATE": cls_metrics.ate if cls_metrics.ate < float("inf") else None,
            "ASE": cls_metrics.ase if cls_metrics.ase < float("inf") else None,
            "AOE": cls_metrics.aoe if cls_metrics.aoe < float("inf") else None,
            "AVE": cls_metrics.ave if cls_metrics.ave < float("inf") else None,
            "AAE": cls_metrics.aae if cls_metrics.aae < float("inf") else None,
            "num_gt": cls_metrics.num_gt,
            "num_pred": cls_metrics.num_pred,
        }
        output["per_class"][cls_name] = cls_dict

    return output


def save_results(
    results: EvalResults,
    output_path: str,
    ablation_results: Optional[Dict[str, EvalResults]] = None,
) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: Primary evaluation results.
        output_path: Path to save the JSON file.
        ablation_results: Optional ablation results to include.
    """
    output: Dict[str, Any] = results_to_dict(results)

    if ablation_results is not None:
        output["ablation"] = {}
        for modality, abl_results in ablation_results.items():
            output["ablation"][modality] = results_to_dict(abl_results)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_path}")


def generate_nuscenes_submission(
    all_detections: List[List[Detection3D]],
    sample_tokens: List[str],
    output_path: str,
    meta: Optional[Dict[str, str]] = None,
) -> None:
    """Generate nuScenes evaluation submission JSON file.

    Creates the standard nuScenes submission format with detection results
    per sample token.

    Args:
        all_detections: Detections per sample (one list per sample).
        sample_tokens: nuScenes sample tokens corresponding to each sample.
        output_path: Path to save the submission JSON.
        meta: Optional metadata dictionary (model name, description, etc.).
    """
    if meta is None:
        meta = {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": True,
            "use_map": False,
            "use_external": False,
        }

    submission: Dict[str, Any] = {
        "meta": meta,
        "results": {},
    }

    for sample_idx, (sample_dets, token) in enumerate(zip(all_detections, sample_tokens)):
        sample_results: List[Dict[str, Any]] = []

        for det in sample_dets:
            result_entry = {
                "sample_token": token,
                "translation": det.center.tolist(),
                "size": det.size.tolist(),
                "rotation": _yaw_to_quaternion(det.yaw),
                "velocity": det.velocity.tolist(),
                "detection_name": det.class_name,
                "detection_score": det.score,
                "attribute_name": det.attribute if det.attribute else _default_attribute(det.class_name),
            }
            sample_results.append(result_entry)

        submission["results"][token] = sample_results

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    logger.info(f"nuScenes submission saved to: {output_path}")
    logger.info(f"  Total samples: {len(sample_tokens)}")
    total_dets = sum(len(d) for d in all_detections)
    logger.info(f"  Total detections: {total_dets}")


def _yaw_to_quaternion(yaw: float) -> List[float]:
    """Convert yaw angle to quaternion [w, x, y, z] (nuScenes convention).

    The quaternion represents a rotation about the z-axis (vertical).

    Args:
        yaw: Yaw angle in radians.

    Returns:
        Quaternion as [w, x, y, z].
    """
    half_yaw = yaw / 2.0
    w = math.cos(half_yaw)
    x = 0.0
    y = 0.0
    z = math.sin(half_yaw)
    return [w, x, y, z]


def _default_attribute(class_name: str) -> str:
    """Return the default nuScenes attribute for a given class.

    Args:
        class_name: Detection class name.

    Returns:
        Default attribute string.
    """
    attribute_map = {
        "car": "vehicle.moving",
        "truck": "vehicle.moving",
        "construction_vehicle": "vehicle.moving",
        "bus": "vehicle.moving",
        "trailer": "vehicle.moving",
        "barrier": "",
        "motorcycle": "cycle.with_rider",
        "bicycle": "cycle.with_rider",
        "pedestrian": "pedestrian.moving",
        "traffic_cone": "",
    }
    return attribute_map.get(class_name, "")


# ---------------------------------------------------------------------------
# Dataset placeholder (for when train.py's NuScenesRadarCameraDataset is not available)
# ---------------------------------------------------------------------------
class NuScenesEvalDataset(torch.utils.data.Dataset):
    """Minimal nuScenes evaluation dataset.

    Loads pre-processed sample info from pickle files and returns structured
    batches for model inference. This is a fallback if the full dataset class
    from train.py is not importable.

    Args:
        data_root: Path to nuScenes data root directory.
        info_path: Path to the info pickle file (e.g., nuscenes_infos_val.pkl).
        point_cloud_range: Spatial range for radar.
        voxel_size: Voxel size for pillar encoding.
        image_size: Target image size (H, W).
        max_pillars: Maximum number of pillars.
        max_points_per_pillar: Maximum points per pillar.
    """

    def __init__(
        self,
        data_root: str,
        info_path: str,
        point_cloud_range: Optional[List[float]] = None,
        voxel_size: Optional[List[float]] = None,
        image_size: Tuple[int, int] = (900, 1600),
        max_pillars: int = 30000,
        max_points_per_pillar: int = 20,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.max_pillars = max_pillars
        self.max_points_per_pillar = max_points_per_pillar

        if point_cloud_range is None:
            point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if voxel_size is None:
            voxel_size = [0.2, 0.2, 8.0]

        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        # Load info file
        import pickle

        logger.info(f"Loading dataset info from: {info_path}")
        with open(info_path, "rb") as f:
            self.infos = pickle.load(f)

        if isinstance(self.infos, dict):
            self.infos = self.infos.get("infos", self.infos.get("data_list", []))

        logger.info(f"  Loaded {len(self.infos)} samples")

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and preprocess a single sample.

        Returns a dictionary with:
            - images: [N_cams, 3, H, W]
            - radar_points: [N_max, C] raw radar points
            - num_radar_points: scalar tensor with count of valid points
            - intrinsics: [N_cams, 3, 3]
            - extrinsics: [N_cams, 4, 4]
            - gt_boxes: [N_gt, 10]
            - gt_labels: [N_gt]
            - sample_token: str
        """
        info = self.infos[idx]
        sample: Dict[str, Any] = {}

        # Load camera images
        cam_names = [
            "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        ]
        images: List[torch.Tensor] = []
        intrinsics_list: List[torch.Tensor] = []
        extrinsics_list: List[torch.Tensor] = []

        cams_info = info.get("cams", info.get("images", {}))

        for cam_name in cam_names:
            cam_info = cams_info.get(cam_name, {})

            # Load image
            img_path = cam_info.get("data_path", cam_info.get("img_path", ""))
            if img_path and os.path.exists(os.path.join(self.data_root, img_path)):
                try:
                    from PIL import Image
                    from torchvision import transforms

                    img = Image.open(os.path.join(self.data_root, img_path)).convert("RGB")
                    transform = transforms.Compose([
                        transforms.Resize(self.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ])
                    img_tensor = transform(img)
                except Exception:
                    img_tensor = torch.zeros(3, self.image_size[0], self.image_size[1])
            else:
                img_tensor = torch.zeros(3, self.image_size[0], self.image_size[1])

            images.append(img_tensor)

            # Camera intrinsics
            intrinsic = cam_info.get(
                "cam_intrinsic",
                cam_info.get("intrinsic", np.eye(3)),
            )
            if isinstance(intrinsic, list):
                intrinsic = np.array(intrinsic)
            intrinsics_list.append(torch.tensor(intrinsic, dtype=torch.float32))

            # Camera extrinsics (sensor2ego or sensor2lidar)
            extrinsic = cam_info.get(
                "sensor2ego",
                cam_info.get("extrinsic", np.eye(4)),
            )
            if isinstance(extrinsic, list):
                extrinsic = np.array(extrinsic)
            if extrinsic.shape == (3, 4):
                extrinsic_4x4 = np.eye(4)
                extrinsic_4x4[:3, :] = extrinsic
                extrinsic = extrinsic_4x4
            extrinsics_list.append(torch.tensor(extrinsic, dtype=torch.float32))

        sample["images"] = torch.stack(images, dim=0)  # [6, 3, H, W]
        sample["intrinsics"] = torch.stack(intrinsics_list, dim=0)  # [6, 3, 3]
        sample["extrinsics"] = torch.stack(extrinsics_list, dim=0)  # [6, 4, 4]

        # Load radar points
        radar_info = info.get("radars", info.get("radar", {}))
        radar_points_all: List[np.ndarray] = []

        radar_names = [
            "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
            "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
        ]

        for radar_name in radar_names:
            r_info = radar_info.get(radar_name, {})
            r_path = r_info.get("data_path", r_info.get("pts_path", ""))
            if r_path and os.path.exists(os.path.join(self.data_root, r_path)):
                try:
                    pts = np.fromfile(
                        os.path.join(self.data_root, r_path), dtype=np.float32
                    )
                    # Radar point format varies; assume 18 features per point
                    n_features = 18
                    if pts.size % n_features == 0:
                        pts = pts.reshape(-1, n_features)
                    else:
                        pts = pts.reshape(-1, pts.size // max(pts.size // n_features, 1))
                    radar_points_all.append(pts)
                except Exception:
                    pass

        if radar_points_all:
            all_pts = np.concatenate(radar_points_all, axis=0)
        else:
            all_pts = np.zeros((0, 6), dtype=np.float32)

        # Filter points within range and prepare for RadarBranch
        all_pts_filtered = self._filter_points_in_range(all_pts)
        num_valid = len(all_pts_filtered)

        # Pad to fixed size for batching (RadarBranch handles voxelization internally)
        max_radar_pts = self.max_pillars * 2  # generous upper bound
        padded_pts = np.zeros((max_radar_pts, all_pts_filtered.shape[1] if num_valid > 0 else 6), dtype=np.float32)
        if num_valid > 0:
            n_keep = min(num_valid, max_radar_pts)
            padded_pts[:n_keep] = all_pts_filtered[:n_keep]

        sample["radar_points"] = torch.tensor(padded_pts, dtype=torch.float32)
        sample["num_radar_points"] = torch.tensor(min(num_valid, max_radar_pts), dtype=torch.long)

        # Ground truth annotations
        gt_boxes_raw = info.get("gt_boxes", info.get("ann", {}).get("gt_boxes_3d", np.zeros((0, 10))))
        gt_labels_raw = info.get("gt_names", info.get("ann", {}).get("gt_labels_3d", np.zeros((0,))))

        if isinstance(gt_boxes_raw, np.ndarray):
            gt_boxes_tensor = torch.tensor(gt_boxes_raw, dtype=torch.float32)
        elif isinstance(gt_boxes_raw, list):
            gt_boxes_tensor = torch.tensor(np.array(gt_boxes_raw), dtype=torch.float32)
        else:
            gt_boxes_tensor = torch.zeros((0, 10), dtype=torch.float32)

        if isinstance(gt_labels_raw, (list, np.ndarray)):
            # Convert string labels to integer indices if needed
            if len(gt_labels_raw) > 0 and isinstance(gt_labels_raw[0], str):
                name_to_id = {name: i for i, name in enumerate(NUSCENES_CLASSES)}
                gt_labels_int = [name_to_id.get(n, -1) for n in gt_labels_raw]
                gt_labels_tensor = torch.tensor(gt_labels_int, dtype=torch.long)
            else:
                gt_labels_tensor = torch.tensor(
                    np.array(gt_labels_raw), dtype=torch.long
                )
        else:
            gt_labels_tensor = torch.zeros((0,), dtype=torch.long)

        # Filter out invalid labels
        valid_label_mask = gt_labels_tensor >= 0
        sample["gt_boxes"] = gt_boxes_tensor[valid_label_mask] if valid_label_mask.any() else torch.zeros((0, 10))
        sample["gt_labels"] = gt_labels_tensor[valid_label_mask] if valid_label_mask.any() else torch.zeros((0,), dtype=torch.long)

        # Sample token for submission
        sample["sample_token"] = info.get("token", info.get("sample_token", f"sample_{idx:06d}"))
        sample["image_height"] = self.image_size[0]
        sample["image_width"] = self.image_size[1]

        return sample

    def _filter_points_in_range(self, points: np.ndarray) -> np.ndarray:
        """Filter radar points to those within the configured point cloud range.

        Args:
            points: Raw radar points [N, C].

        Returns:
            Filtered points [M, C] where M <= N.
        """
        if len(points) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        pc_range = self.point_cloud_range

        # Ensure at least 3 spatial columns exist for filtering
        if points.shape[1] < 3:
            return points

        x_mask = (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3])
        y_mask = (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4])
        z_mask = (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
        valid = x_mask & y_mask & z_mask

        filtered = points[valid]

        # If the point format has more than 6 features, truncate to standard
        # radar features: [x, y, z, vx, vy, rcs]
        if filtered.shape[1] > 6:
            filtered = filtered[:, :6]

        return filtered


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CRAFT 3D Object Detection - Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth or .pt file)",
    )

    # Optional arguments
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/nuscenes",
        help="Path to the nuScenes dataset root directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. If not specified, uses default config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./eval_results.json",
        help="Path to save evaluation results JSON",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="Minimum detection score threshold",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=500,
        help="Maximum number of detections per sample",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.2,
        help="BEV NMS IoU threshold",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="fused",
        choices=["fused", "camera_only", "radar_only"],
        help="Input modality to use for evaluation",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run modality ablation study (camera-only, radar-only, fused)",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Do not use EMA weights even if available in checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0', 'cpu'). Auto-detects if not specified.",
    )
    parser.add_argument(
        "--submission",
        type=str,
        default=None,
        help="Path to save nuScenes submission JSON (for official evaluation)",
    )
    parser.add_argument(
        "--info-path",
        type=str,
        default=None,
        help="Path to info pickle file. Overrides config/data-root based path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Device selection
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    else:
        # Try default config path relative to script
        default_config = _SCRIPT_DIR.parent / "configs" / "craft_nuscenes.yaml"
        if default_config.exists():
            config = load_config(str(default_config))
            logger.info(f"Using default config: {default_config}")
        else:
            # Minimal default configuration
            config = {
                "model": {
                    "backbone": {"type": "resnet50"},
                    "neck": {"out_channels": 256},
                    "radar_pillar_encoder": {
                        "in_channels": 18,
                        "out_channels": 64,
                        "voxel_size": [0.2, 0.2, 8.0],
                        "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                        "max_pillars": 30000,
                        "max_points_per_pillar": 20,
                    },
                    "fusion_transformer": {
                        "d_model": 256,
                        "n_heads": 8,
                        "n_layers": 6,
                        "d_ffn": 1024,
                        "dropout": 0.1,
                    },
                    "detection_head": {
                        "num_classes": 10,
                        "in_channels": 256,
                        "shared_conv_channels": 64,
                    },
                },
                "data": {
                    "root_path": args.data_root,
                    "image": {"size": [900, 1600]},
                    "point_cloud": {
                        "range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                        "voxel_size": [0.2, 0.2, 8.0],
                        "max_pillars": 30000,
                        "max_points_per_pillar": 20,
                    },
                    "num_workers": 4,
                },
                "evaluation": {
                    "score_threshold": 0.1,
                    "nms_threshold": 0.2,
                    "max_detections": 500,
                },
                "class_names": NUSCENES_CLASSES,
            }
            logger.info("Using default built-in configuration")

    # Override config with CLI arguments where applicable
    eval_cfg = config.get("evaluation", {})
    data_cfg = config.get("data", {})
    class_names = config.get("class_names", NUSCENES_CLASSES)

    score_threshold = args.score_threshold
    max_detections = args.max_detections
    nms_threshold = args.nms_threshold

    # Build model
    logger.info("Building CRAFT model...")
    model = build_model_from_config(config, device)

    # Load checkpoint
    model = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        use_ema=not args.no_ema,
    )
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")

    # Build dataset and dataloader
    data_root = args.data_root or data_cfg.get("root_path", "/data/nuscenes")

    if args.info_path is not None:
        info_path = args.info_path
    else:
        info_paths = data_cfg.get("info_path", {})
        if isinstance(info_paths, dict):
            info_path = info_paths.get(args.split, "")
        else:
            info_path = str(info_paths)

        if not info_path:
            info_path = os.path.join(data_root, f"nuscenes_infos_{args.split}.pkl")

    pc_cfg = data_cfg.get("point_cloud", {})
    img_cfg = data_cfg.get("image", {})
    image_size = tuple(img_cfg.get("size", [900, 1600]))

    logger.info(f"Building evaluation dataset (split={args.split})...")
    dataset = NuScenesEvalDataset(
        data_root=data_root,
        info_path=info_path,
        point_cloud_range=pc_cfg.get("range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        voxel_size=pc_cfg.get("voxel_size", [0.2, 0.2, 8.0]),
        image_size=image_size,
        max_pillars=pc_cfg.get("max_pillars", 30000),
        max_points_per_pillar=pc_cfg.get("max_points_per_pillar", 20),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    logger.info(f"  Dataset size: {len(dataset)} samples")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Num workers: {args.num_workers}")

    # Run evaluation
    ablation_results: Optional[Dict[str, EvalResults]] = None

    if args.ablation:
        ablation_results = run_modality_ablation(
            model=model,
            dataloader=dataloader,
            device=device,
            score_threshold=score_threshold,
            max_detections=max_detections,
            nms_iou_threshold=nms_threshold,
            class_names=class_names,
        )
        # Use fused results as primary
        results = ablation_results["fused"]
    else:
        results = run_evaluation(
            model=model,
            dataloader=dataloader,
            device=device,
            score_threshold=score_threshold,
            max_detections=max_detections,
            nms_iou_threshold=nms_threshold,
            modality=args.modality,
            class_names=class_names,
        )

    # Print results
    results_table = format_results_table(results)
    print("\n" + results_table)

    # Save results to JSON
    save_results(results, args.output, ablation_results)

    # Generate nuScenes submission if requested
    if args.submission is not None:
        logger.info("Generating nuScenes submission format...")
        # Re-run inference to collect detections with sample tokens
        all_detections: List[List[Detection3D]] = []
        sample_tokens: List[str] = []

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(device)
                radar_points = batch["radar_points"].to(device)
                num_radar_points = batch["num_radar_points"].to(device)
                intrinsics = batch["intrinsics"].to(device)
                extrinsics = batch["extrinsics"].to(device)

                image_shape = (
                    batch.get("image_height", 900),
                    batch.get("image_width", 1600),
                )
                if isinstance(image_shape[0], torch.Tensor):
                    image_shape = (image_shape[0].item(), image_shape[1].item())

                radar_properties = batch.get("radar_properties", None)
                if radar_properties is not None:
                    radar_properties = radar_properties.to(device)

                head_outputs = model(
                    images=images,
                    radar_points=radar_points,
                    num_radar_points=num_radar_points,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    image_shape=image_shape,
                    radar_properties=radar_properties,
                    modality=args.modality,
                )

                batch_dets = decode_model_predictions(
                    head_outputs=head_outputs,
                    score_threshold=score_threshold,
                    max_detections=max_detections,
                    nms_iou_threshold=nms_threshold,
                )
                all_detections.extend(batch_dets)

                # Collect sample tokens
                tokens = batch.get("sample_token", [])
                if isinstance(tokens, list):
                    sample_tokens.extend(tokens)
                else:
                    sample_tokens.extend([f"sample_{i}" for i in range(images.shape[0])])

        generate_nuscenes_submission(
            all_detections=all_detections,
            sample_tokens=sample_tokens,
            output_path=args.submission,
            meta={
                "use_camera": True,
                "use_lidar": False,
                "use_radar": True,
                "use_map": False,
                "use_external": False,
            },
        )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
