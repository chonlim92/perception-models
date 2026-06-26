"""
Single-sample inference script for PointNet++ 3D object detection.

Supports:
- Loading a model from checkpoint
- Preprocessing a single .bin point cloud file
- Running forward pass and post-processing (NMS)
- Returning boxes for visualization (e.g. with Open3D)
"""

import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .model import PointNetPPDetection, PointNetPPClassification, PointNetPPSegmentation


def load_model(
    checkpoint_path: str,
    task: str = "detection",
    num_classes: int = 4,
    in_channels: int = 4,
    num_angle_bins: int = 12,
    num_seg_classes: int = 20,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Load a PointNet++ model from a checkpoint file.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        task: One of 'detection', 'classification', 'segmentation'
        num_classes: Number of classes
        in_channels: Number of input channels
        num_angle_bins: Number of angle bins (detection only)
        num_seg_classes: Number of segmentation classes
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if task == "detection":
        model = PointNetPPDetection(
            num_classes=num_classes,
            in_channels=in_channels,
            num_angle_bins=num_angle_bins,
        )
    elif task == "classification":
        model = PointNetPPClassification(
            num_classes=num_classes,
            in_channels=in_channels,
        )
    elif task == "segmentation":
        model = PointNetPPSegmentation(
            num_seg_classes=num_seg_classes,
            in_channels=in_channels,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def preprocess_point_cloud(
    bin_path: str,
    npoints: int = 16384,
    point_range: Optional[List[float]] = None,
    channels: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess a single .bin point cloud file.

    Args:
        bin_path: Path to the .bin file (KITTI format: N x 4 float32)
        npoints: Number of points to subsample to
        point_range: [xmin, ymin, zmin, xmax, ymax, zmax] spatial filter
        channels: Number of channels in the .bin file (default 4)

    Returns:
        xyz: (1, npoints, 3) tensor
        features: (1, npoints, C-3) tensor (e.g., intensity)
    """
    if point_range is None:
        point_range = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]

    # Load raw point cloud
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, channels)

    # Filter by range
    xmin, ymin, zmin, xmax, ymax, zmax = point_range
    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )
    points = points[mask]

    # Subsample
    n = points.shape[0]
    if n == 0:
        points = np.zeros((npoints, channels), dtype=np.float32)
    elif n >= npoints:
        indices = np.random.choice(n, npoints, replace=False)
        points = points[indices]
    else:
        indices = np.concatenate([
            np.arange(n),
            np.random.choice(n, npoints - n, replace=True),
        ])
        points = points[indices]

    # Split into xyz and features
    xyz = torch.from_numpy(points[:, :3]).float().unsqueeze(0)  # (1, npoints, 3)
    features = torch.from_numpy(points[:, 3:]).float().unsqueeze(0)  # (1, npoints, C-3)

    return xyz, features


def decode_detections(
    predictions: Dict,
    score_threshold: float = 0.3,
    num_angle_bins: int = 12,
) -> Dict:
    """
    Decode raw model outputs into interpretable detections.

    Args:
        predictions: Dictionary from model forward pass containing:
            'center': (1, N, 3) center offsets
            'size': (1, N, 3) predicted sizes
            'angle_cls': (1, N, num_angle_bins) angle bin logits
            'angle_res': (1, N, num_angle_bins) angle residuals
            'cls_scores': (1, N, num_classes) class logits
            'proposal_xyz': (1, N, 3) proposal locations

        score_threshold: Minimum confidence to keep a detection
        num_angle_bins: Number of angle bins used in training

    Returns:
        Dictionary with:
            'boxes': (K, 7) array [x, y, z, w, l, h, yaw]
            'scores': (K,) confidence scores
            'labels': (K,) predicted class indices (1-indexed)
    """
    bin_size = 2 * math.pi / num_angle_bins

    # Remove batch dimension
    center_offset = predictions["center"][0]  # (N, 3)
    size_pred = predictions["size"][0]  # (N, 3)
    angle_cls = predictions["angle_cls"][0]  # (N, num_angle_bins)
    angle_res = predictions["angle_res"][0]  # (N, num_angle_bins)
    cls_scores = predictions["cls_scores"][0]  # (N, num_classes)
    proposal_xyz = predictions["proposal_xyz"][0]  # (N, 3)

    # Compute class probabilities (exclude background class 0)
    cls_probs = F.softmax(cls_scores, dim=-1)  # (N, num_classes)
    # Best foreground class and score per proposal
    fg_probs = cls_probs[:, 1:]  # exclude background
    max_scores, pred_classes = fg_probs.max(dim=-1)  # (N,)
    pred_classes = pred_classes + 1  # shift back to 1-indexed

    # Decode angle from bins
    angle_bin_idx = torch.argmax(angle_cls, dim=-1)  # (N,)
    angle_residual = torch.gather(
        angle_res, 1, angle_bin_idx.unsqueeze(-1)
    ).squeeze(-1)  # (N,)
    pred_angle = (
        angle_bin_idx.float() * bin_size
        + (angle_residual + 1.0) * (bin_size / 2.0)
    )

    # Decode center (add offset to proposal location)
    pred_center = proposal_xyz + center_offset  # (N, 3)

    # Filter by score threshold
    valid = max_scores > score_threshold
    pred_center = pred_center[valid].detach().cpu().numpy()
    pred_size = size_pred[valid].detach().cpu().numpy()
    pred_angle_np = pred_angle[valid].detach().cpu().numpy()
    scores = max_scores[valid].detach().cpu().numpy()
    labels = pred_classes[valid].detach().cpu().numpy()

    # Assemble boxes: [x, y, z, w, l, h, yaw]
    boxes = np.column_stack([
        pred_center,
        np.abs(pred_size),  # sizes should be positive
        pred_angle_np[:, np.newaxis],
    ])  # (K, 7)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


def nms_3d(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    3D Non-Maximum Suppression using BEV IoU.

    Performed per-class: only suppress boxes of the same class.

    Args:
        boxes: (K, 7) array [x, y, z, w, l, h, yaw]
        scores: (K,) confidence scores
        labels: (K,) class labels
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered (boxes, scores, labels) after NMS
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    keep_mask = np.ones(len(boxes), dtype=bool)
    unique_labels = np.unique(labels)

    for cls in unique_labels:
        cls_mask = labels == cls
        cls_indices = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]

        # Sort by score descending
        order = np.argsort(-cls_scores)
        cls_indices = cls_indices[order]
        cls_boxes = cls_boxes[order]

        # BEV IoU for NMS
        bev_boxes = cls_boxes[:, [0, 1, 3, 4, 6]]  # x, y, w, l, yaw

        from .evaluate import compute_bev_iou

        iou_matrix = compute_bev_iou(bev_boxes, bev_boxes)

        n = len(cls_indices)
        suppressed = np.zeros(n, dtype=bool)

        for i in range(n):
            if suppressed[i]:
                continue
            for j in range(i + 1, n):
                if suppressed[j]:
                    continue
                if iou_matrix[i, j] > iou_threshold:
                    suppressed[j] = True
                    keep_mask[cls_indices[j]] = False

    return boxes[keep_mask], scores[keep_mask], labels[keep_mask]


def prepare_visualization(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = None,
) -> List[Dict]:
    """
    Prepare detection results for Open3D visualization.

    Returns a list of box dictionaries ready to be rendered as
    Open3D LineSet objects.

    Args:
        boxes: (K, 7) [x, y, z, w, l, h, yaw]
        scores: (K,) confidence scores
        labels: (K,) class labels (1-indexed)
        class_names: Optional list mapping class index to name

    Returns:
        List of dictionaries with 'corners', 'label', 'score', 'color'
    """
    if class_names is None:
        class_names = ["Background", "Car", "Pedestrian", "Cyclist"]

    # Color map per class (RGB, 0-1)
    class_colors = {
        "Car": [0.0, 1.0, 0.0],       # Green
        "Pedestrian": [1.0, 0.0, 0.0],  # Red
        "Cyclist": [0.0, 0.0, 1.0],     # Blue
    }
    default_color = [1.0, 1.0, 0.0]  # Yellow

    vis_boxes = []

    for i in range(len(boxes)):
        x, y, z, w, l, h, yaw = boxes[i]

        # Compute 8 corners
        cos_a = np.cos(yaw)
        sin_a = np.sin(yaw)

        hw, hl, hh = w / 2.0, l / 2.0, h / 2.0

        # Local corners
        corners_local = np.array([
            [-hw, -hl, -hh],
            [hw, -hl, -hh],
            [hw, hl, -hh],
            [-hw, hl, -hh],
            [-hw, -hl, hh],
            [hw, -hl, hh],
            [hw, hl, hh],
            [-hw, hl, hh],
        ])

        # Rotation matrix (yaw around z)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ])

        corners = corners_local @ R.T + np.array([x, y, z])

        cls_idx = int(labels[i])
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else "Unknown"
        color = class_colors.get(cls_name, default_color)

        vis_boxes.append({
            "corners": corners,  # (8, 3)
            "label": cls_name,
            "score": float(scores[i]),
            "color": color,
            "box_params": boxes[i],
        })

    return vis_boxes


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    bin_path: str,
    npoints: int = 16384,
    point_range: Optional[List[float]] = None,
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.1,
    num_angle_bins: int = 12,
    device: str = "cuda",
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Run full inference pipeline on a single point cloud file.

    Steps:
    1. Load and preprocess point cloud
    2. Forward pass through model
    3. Decode predictions
    4. Apply NMS
    5. Prepare visualization data

    Args:
        model: PointNet++ model in eval mode
        bin_path: Path to .bin point cloud file
        npoints: Number of points for subsampling
        point_range: Spatial filter [xmin, ymin, zmin, xmax, ymax, zmax]
        score_threshold: Minimum detection confidence
        nms_iou_threshold: NMS IoU threshold
        num_angle_bins: Number of angle bins
        device: Device string
        class_names: List of class names for output

    Returns:
        Dictionary with:
            'boxes': (K, 7) final detection boxes
            'scores': (K,) detection scores
            'labels': (K,) class labels
            'vis_boxes': List of visualization-ready box dicts
            'num_input_points': Number of raw points loaded
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # 1. Preprocess
    xyz, features = preprocess_point_cloud(
        bin_path, npoints=npoints, point_range=point_range
    )
    xyz = xyz.to(device_obj)
    features = features.to(device_obj)

    # 2. Forward pass
    extra_feat = features if features.shape[-1] > 0 else None
    predictions = model(xyz, extra_feat)

    # 3. Decode
    detections = decode_detections(
        predictions,
        score_threshold=score_threshold,
        num_angle_bins=num_angle_bins,
    )

    # 4. NMS
    boxes, scores, labels = nms_3d(
        detections["boxes"],
        detections["scores"],
        detections["labels"],
        iou_threshold=nms_iou_threshold,
    )

    # 5. Visualization
    vis_boxes = prepare_visualization(boxes, scores, labels, class_names)

    # Count raw input points
    raw_points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "vis_boxes": vis_boxes,
        "num_input_points": raw_points.shape[0],
        "num_detections": len(boxes),
    }


def main():
    """Standalone inference entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="PointNet++ Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--bin_file", type=str, required=True,
                        help="Path to .bin point cloud file")
    parser.add_argument("--task", type=str, default="detection",
                        choices=["detection", "classification", "segmentation"])
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--npoints", type=int, default=16384)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--nms_threshold", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load model
    model = load_model(
        args.checkpoint,
        task=args.task,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        device=args.device,
    )

    if args.task == "detection":
        # Run inference
        results = run_inference(
            model,
            args.bin_file,
            npoints=args.npoints,
            score_threshold=args.score_threshold,
            nms_iou_threshold=args.nms_threshold,
            device=args.device,
        )

        print(f"Input points: {results['num_input_points']}")
        print(f"Detections: {results['num_detections']}")
        for vb in results["vis_boxes"]:
            print(
                f"  {vb['label']}: score={vb['score']:.3f}, "
                f"center=({vb['box_params'][0]:.1f}, "
                f"{vb['box_params'][1]:.1f}, {vb['box_params'][2]:.1f})"
            )

    elif args.task == "classification":
        device_obj = torch.device(
            args.device if torch.cuda.is_available() else "cpu"
        )
        xyz, features = preprocess_point_cloud(args.bin_file, npoints=args.npoints)
        xyz = xyz.to(device_obj)
        features = features.to(device_obj)
        extra_feat = features if features.shape[-1] > 0 else None

        logits = model(xyz, extra_feat)
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
        print(f"Predicted class: {pred_class}, confidence: {confidence:.4f}")

    elif args.task == "segmentation":
        device_obj = torch.device(
            args.device if torch.cuda.is_available() else "cpu"
        )
        xyz, features = preprocess_point_cloud(args.bin_file, npoints=args.npoints)
        xyz = xyz.to(device_obj)
        features = features.to(device_obj)
        extra_feat = features if features.shape[-1] > 0 else None

        seg_logits = model(xyz, extra_feat)
        seg_pred = torch.argmax(seg_logits, dim=-1)  # (1, N)
        unique_classes, counts = torch.unique(seg_pred, return_counts=True)
        print("Segmentation results:")
        for cls, cnt in zip(unique_classes.tolist(), counts.tolist()):
            print(f"  Class {cls}: {cnt} points")


if __name__ == "__main__":
    main()
