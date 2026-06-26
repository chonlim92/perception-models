"""
Inference script for PETR/PETRv2/StreamPETR.

Supports single-frame and streaming (temporal) inference, NMS post-processing,
3D bounding box output, and visualization helpers that project 3D boxes
onto camera images.
"""

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import yaml

from .model import PETRConfig, PETRModel

logger = logging.getLogger(__name__)


class PETRInference:
    """Inference wrapper for PETR/PETRv2/StreamPETR models.

    Handles model loading, preprocessing, inference, post-processing (NMS),
    and optional visualization. For StreamPETR, maintains query memory
    across sequential frames.

    Args:
        config_path: Path to YAML configuration file.
        checkpoint_path: Path to trained model checkpoint.
        device: Inference device ('cuda' or 'cpu').
        score_threshold: Minimum confidence for output detections.
        nms_threshold: 3D NMS distance threshold (BEV center distance).
        max_detections: Maximum number of detections to output.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        score_threshold: float = 0.3,
        nms_threshold: float = 2.0,
        max_detections: int = 300,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        model_config = self.config.get("model", {})
        self.petr_config = PETRConfig(**model_config)
        self.pc_range = self.petr_config.pc_range
        self.is_streaming = self.petr_config.variant == "streampetr"

        # Build and load model
        self.model = PETRModel(self.petr_config).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        # Frame counter for streaming
        self._frame_count = 0

        logger.info(
            f"Initialized {self.petr_config.variant} inference on {self.device}"
        )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    def reset(self) -> None:
        """Reset temporal state (call at start of new sequence)."""
        self.model.reset_temporal_state()
        self._frame_count = 0

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        ego_motion: Optional[torch.Tensor] = None,
        ego_motion_vec: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """Run inference on a single frame (or frame batch).

        Args:
            images: Multi-view images (B, N_cams, 3, H, W) or (N_cams, 3, H, W).
                Values in [0, 1] range.
            intrinsics: Camera intrinsics (B, N_cams, 3, 3) or (N_cams, 3, 3).
            extrinsics: Camera-to-ego transforms (B, N_cams, 4, 4) or (N_cams, 4, 4).
            ego_motion: Ego-motion from prev to current (B, 4, 4).
                Required for StreamPETR after first frame.
            ego_motion_vec: Ego velocity vector (B, 6).
                Required for StreamPETR after first frame.

        Returns:
            Dictionary with:
                'boxes_3d': (N_det, 10) array [cx,cy,cz,w,l,h,sin,cos,vx,vy].
                'scores': (N_det,) confidence scores.
                'labels': (N_det,) predicted class indices.
        """
        # Add batch dimension if needed
        if images.dim() == 4:
            images = images.unsqueeze(0)
            intrinsics = intrinsics.unsqueeze(0)
            extrinsics = extrinsics.unsqueeze(0)
            if ego_motion is not None:
                ego_motion = ego_motion.unsqueeze(0)
            if ego_motion_vec is not None:
                ego_motion_vec = ego_motion_vec.unsqueeze(0)

        # Move to device
        images = images.to(self.device)
        intrinsics = intrinsics.to(self.device)
        extrinsics = extrinsics.to(self.device)
        if ego_motion is not None:
            ego_motion = ego_motion.to(self.device)
        if ego_motion_vec is not None:
            ego_motion_vec = ego_motion_vec.to(self.device)

        # For StreamPETR first frame, no ego-motion
        if self.is_streaming and self._frame_count == 0:
            ego_motion = None
            ego_motion_vec = None

        # Forward pass
        outputs = self.model(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            ego_motion=ego_motion,
            ego_motion_vec=ego_motion_vec,
        )

        self._frame_count += 1

        # Post-process predictions
        predictions = outputs["predictions"]
        cls_scores = predictions["cls_scores"][-1]  # Last decoder layer
        bbox_preds = predictions["bbox_preds"][-1]

        # Process first sample in batch
        results = self._post_process(cls_scores[0], bbox_preds[0])

        return results

    def _post_process(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """Post-process raw model predictions.

        Applies score thresholding, NMS, and returns final detections.

        Args:
            cls_scores: (Q, num_classes) classification logits.
            bbox_preds: (Q, code_size) bounding box predictions.

        Returns:
            Dictionary with filtered detections.
        """
        # Get per-query max scores and labels
        scores = cls_scores.sigmoid()  # (Q, num_classes)
        max_scores, pred_labels = scores.max(dim=-1)  # (Q,)

        # Score threshold filtering
        keep_mask = max_scores > self.score_threshold
        kept_scores = max_scores[keep_mask]
        kept_labels = pred_labels[keep_mask]
        kept_boxes = bbox_preds[keep_mask]

        if kept_scores.numel() == 0:
            return {
                "boxes_3d": np.zeros((0, 10), dtype=np.float32),
                "scores": np.zeros(0, dtype=np.float32),
                "labels": np.zeros(0, dtype=np.int64),
            }

        # Convert to numpy for NMS
        scores_np = kept_scores.cpu().numpy()
        labels_np = kept_labels.cpu().numpy()
        boxes_np = kept_boxes.cpu().numpy()

        # Apply class-wise 3D NMS (BEV center distance based)
        keep_indices = self._nms_3d(boxes_np, scores_np, labels_np)

        # Limit to max detections
        if len(keep_indices) > self.max_detections:
            # Sort by score and take top-k
            sort_idx = np.argsort(-scores_np[keep_indices])
            keep_indices = keep_indices[sort_idx[: self.max_detections]]

        return {
            "boxes_3d": boxes_np[keep_indices],
            "scores": scores_np[keep_indices],
            "labels": labels_np[keep_indices],
        }

    def _nms_3d(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Apply 3D NMS based on BEV center distance.

        For each class, suppress detections whose BEV center is within
        nms_threshold of a higher-scoring detection.

        Args:
            boxes: (N, 10) bounding boxes.
            scores: (N,) confidence scores.
            labels: (N,) class labels.

        Returns:
            Array of indices to keep.
        """
        keep_indices = []
        unique_labels = np.unique(labels)

        for cls_id in unique_labels:
            cls_mask = labels == cls_id
            cls_indices = np.where(cls_mask)[0]
            cls_scores = scores[cls_mask]
            cls_boxes = boxes[cls_mask]

            # Sort by score descending
            sort_idx = np.argsort(-cls_scores)
            cls_indices = cls_indices[sort_idx]
            cls_boxes = cls_boxes[sort_idx]

            suppressed = np.zeros(len(cls_indices), dtype=bool)

            for i in range(len(cls_indices)):
                if suppressed[i]:
                    continue
                keep_indices.append(cls_indices[i])

                # Suppress nearby lower-scoring detections
                center_i = cls_boxes[i, :2]  # BEV center (x, y)
                for j in range(i + 1, len(cls_indices)):
                    if suppressed[j]:
                        continue
                    center_j = cls_boxes[j, :2]
                    dist = np.linalg.norm(center_i - center_j)
                    if dist < self.nms_threshold:
                        suppressed[j] = True

        return np.array(keep_indices, dtype=np.int64)

    @torch.no_grad()
    def predict_sequence(
        self,
        image_sequence: List[torch.Tensor],
        intrinsics_sequence: List[torch.Tensor],
        extrinsics_sequence: List[torch.Tensor],
        ego_motion_sequence: List[Optional[torch.Tensor]],
        ego_motion_vec_sequence: List[Optional[torch.Tensor]],
    ) -> List[Dict[str, np.ndarray]]:
        """Run streaming inference on a sequence of frames.

        For StreamPETR, maintains query memory across frames for
        temporal reasoning.

        Args:
            image_sequence: List of (N_cams, 3, H, W) tensors, one per frame.
            intrinsics_sequence: List of (N_cams, 3, 3) tensors.
            extrinsics_sequence: List of (N_cams, 4, 4) tensors.
            ego_motion_sequence: List of (4, 4) ego-motion tensors (None for first).
            ego_motion_vec_sequence: List of (6,) velocity tensors (None for first).

        Returns:
            List of detection dictionaries, one per frame.
        """
        self.reset()
        results = []

        for frame_idx in range(len(image_sequence)):
            result = self.predict(
                images=image_sequence[frame_idx],
                intrinsics=intrinsics_sequence[frame_idx],
                extrinsics=extrinsics_sequence[frame_idx],
                ego_motion=ego_motion_sequence[frame_idx],
                ego_motion_vec=ego_motion_vec_sequence[frame_idx],
            )
            results.append(result)

        return results


def project_3d_box_to_image(
    box_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    img_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Project a 3D bounding box onto a camera image.

    Computes the 8 corners of the 3D box, transforms them to the camera
    frame, and projects them to 2D pixel coordinates.

    Args:
        box_3d: (10,) array [cx, cy, cz, w, l, h, sin_yaw, cos_yaw, vx, vy].
        intrinsics: (3, 3) camera intrinsic matrix.
        extrinsics: (4, 4) camera-to-ego transform (cam2ego).
            We need ego-to-camera, so we invert this.
        img_size: (H, W) image dimensions.

    Returns:
        (8, 2) array of projected corner pixel coordinates, or None if
        the box is behind the camera.
    """
    cx, cy, cz = box_3d[0], box_3d[1], box_3d[2]
    w, l, h = box_3d[3], box_3d[4], box_3d[5]
    sin_yaw, cos_yaw = box_3d[6], box_3d[7]
    yaw = np.arctan2(sin_yaw, cos_yaw)

    # Generate 8 corners of the box in ego frame
    # Box centered at (cx, cy, cz) with dimensions (w, l, h)
    dx = w / 2
    dy = l / 2
    dz = h / 2

    corners_local = np.array(
        [
            [dx, dy, dz],
            [dx, dy, -dz],
            [dx, -dy, dz],
            [dx, -dy, -dz],
            [-dx, dy, dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [-dx, -dy, -dz],
        ],
        dtype=np.float32,
    )  # (8, 3)

    # Rotation matrix for yaw
    R_yaw = np.array(
        [
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Rotate and translate corners to ego frame
    corners_ego = (R_yaw @ corners_local.T).T  # (8, 3)
    corners_ego[:, 0] += cx
    corners_ego[:, 1] += cy
    corners_ego[:, 2] += cz

    # Transform from ego to camera frame
    # extrinsics is cam2ego, so ego2cam = inv(cam2ego)
    T_ego2cam = np.linalg.inv(extrinsics)

    # Convert to homogeneous
    corners_homo = np.hstack(
        [corners_ego, np.ones((8, 1), dtype=np.float32)]
    )  # (8, 4)

    # Transform to camera frame
    corners_cam = (T_ego2cam @ corners_homo.T).T[:, :3]  # (8, 3)

    # Check if all corners are in front of camera
    if np.any(corners_cam[:, 2] <= 0):
        # At least one corner behind camera - partial visibility
        # Filter to only corners in front
        valid = corners_cam[:, 2] > 0
        if not np.any(valid):
            return None

    # Project to image plane: p = K @ [x, y, z]^T / z
    corners_2d = intrinsics @ corners_cam.T  # (3, 8)
    corners_2d = corners_2d[:2, :] / (corners_2d[2:3, :] + 1e-8)  # (2, 8)
    corners_2d = corners_2d.T  # (8, 2)

    # Check if projected points are within image bounds
    H, W = img_size
    in_image = (
        (corners_2d[:, 0] >= 0)
        & (corners_2d[:, 0] < W)
        & (corners_2d[:, 1] >= 0)
        & (corners_2d[:, 1] < H)
    )
    if not np.any(in_image):
        return None

    return corners_2d


def draw_3d_boxes_on_image(
    image: np.ndarray,
    boxes_3d: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Draw projected 3D bounding boxes on a camera image.

    Args:
        image: (H, W, 3) uint8 image array (BGR or RGB).
        boxes_3d: (N, 10) 3D bounding boxes.
        scores: (N,) confidence scores.
        labels: (N,) class indices.
        intrinsics: (3, 3) camera intrinsics.
        extrinsics: (4, 4) camera-to-ego transform.
        class_names: Optional list of class names for labels.
        score_threshold: Minimum score to draw.

    Returns:
        Image with drawn boxes (H, W, 3) uint8.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping visualization")
        return image

    img = image.copy()
    H, W = img.shape[:2]

    # Color palette for different classes
    colors = [
        (255, 0, 0),    # car - blue
        (0, 255, 0),    # truck - green
        (0, 0, 255),    # construction_vehicle - red
        (255, 255, 0),  # bus - cyan
        (255, 0, 255),  # trailer - magenta
        (0, 255, 255),  # barrier - yellow
        (128, 0, 255),  # motorcycle - purple
        (255, 128, 0),  # bicycle - orange
        (0, 128, 255),  # pedestrian - sky blue
        (128, 255, 0),  # traffic_cone - lime
    ]

    # Edge connectivity for drawing box wireframe
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]

    for i in range(boxes_3d.shape[0]):
        if scores[i] < score_threshold:
            continue

        corners_2d = project_3d_box_to_image(
            boxes_3d[i], intrinsics, extrinsics, (H, W)
        )

        if corners_2d is None:
            continue

        label_idx = int(labels[i]) % len(colors)
        color = colors[label_idx]

        # Draw edges
        for start, end in edges:
            pt1 = tuple(corners_2d[start].astype(int))
            pt2 = tuple(corners_2d[end].astype(int))

            # Only draw if both points are in image
            if (
                0 <= pt1[0] < W
                and 0 <= pt1[1] < H
                and 0 <= pt2[0] < W
                and 0 <= pt2[1] < H
            ):
                cv2.line(img, pt1, pt2, color, 2)

        # Draw label text
        top_center = corners_2d[:, 1].min()
        left_center = corners_2d[:, 0].mean()
        text_pos = (int(left_center), int(max(0, top_center - 5)))

        if class_names is not None and int(labels[i]) < len(class_names):
            text = f"{class_names[int(labels[i])]}: {scores[i]:.2f}"
        else:
            text = f"cls{int(labels[i])}: {scores[i]:.2f}"

        cv2.putText(
            img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    return img


def visualize_detections(
    images: np.ndarray,
    detections: Dict[str, np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    output_path: str,
    class_names: Optional[List[str]] = None,
) -> None:
    """Visualize 3D detections projected onto all camera views.

    Args:
        images: (N_cams, H, W, 3) uint8 images.
        detections: Detection dict with 'boxes_3d', 'scores', 'labels'.
        intrinsics: (N_cams, 3, 3) camera intrinsics.
        extrinsics: (N_cams, 4, 4) camera extrinsics.
        output_path: Path to save visualization image.
        class_names: Class name list for labeling.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping visualization")
        return

    N_cams = images.shape[0]
    vis_images = []

    for cam_idx in range(N_cams):
        vis_img = draw_3d_boxes_on_image(
            image=images[cam_idx],
            boxes_3d=detections["boxes_3d"],
            scores=detections["scores"],
            labels=detections["labels"],
            intrinsics=intrinsics[cam_idx],
            extrinsics=extrinsics[cam_idx],
            class_names=class_names,
        )
        vis_images.append(vis_img)

    # Arrange in 2x3 grid (standard nuScenes camera layout)
    H, W = vis_images[0].shape[:2]
    grid = np.zeros((2 * H, 3 * W, 3), dtype=np.uint8)

    # Top row: FRONT_LEFT, FRONT, FRONT_RIGHT
    if N_cams >= 3:
        grid[0:H, 0:W] = vis_images[2]       # FRONT_LEFT
        grid[0:H, W:2*W] = vis_images[0]     # FRONT
        grid[0:H, 2*W:3*W] = vis_images[1]   # FRONT_RIGHT

    # Bottom row: BACK_LEFT, BACK, BACK_RIGHT
    if N_cams >= 6:
        grid[H:2*H, 0:W] = vis_images[4]     # BACK_LEFT
        grid[H:2*H, W:2*W] = vis_images[3]   # BACK
        grid[H:2*H, 2*W:3*W] = vis_images[5] # BACK_RIGHT

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, grid)
    logger.info(f"Visualization saved to {output_path}")


def main() -> None:
    """Entry point for inference script."""
    parser = argparse.ArgumentParser(description="PETR/StreamPETR Inference")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--data_root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./inference_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.3,
        help="Detection score threshold",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save visualizations"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Inference device"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Initialize inference engine
    engine = PETRInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
    )

    # Load data (using dataset for convenience)
    from .dataset import DETECTION_CLASSES, NuScenesDataset

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    data_config = config.get("data", {})

    dataset = NuScenesDataset(
        data_root=args.data_root,
        ann_file=data_config.get("val_ann_file"),
        split="val",
        num_cameras=model_config.get("num_cameras", 6),
        img_size=tuple(model_config.get("img_size", [900, 1600])),
        num_temporal_frames=model_config.get("num_temporal_frames", 0),
        augmentation=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    # Run inference
    engine.reset()
    total_time = 0.0

    for idx in range(len(dataset)):
        sample = dataset[idx]

        start_time = time.time()
        detections = engine.predict(
            images=sample["images"],
            intrinsics=sample["intrinsics"],
            extrinsics=sample["extrinsics"],
            ego_motion=sample.get("ego_motion"),
            ego_motion_vec=sample.get("ego_motion_vec"),
        )
        elapsed = time.time() - start_time
        total_time += elapsed

        all_results.append(detections)

        if idx % 100 == 0:
            logger.info(
                f"Processed {idx + 1}/{len(dataset)} samples, "
                f"FPS: {(idx + 1) / total_time:.1f}, "
                f"Detections: {len(detections['scores'])}"
            )

        # Save visualization for first few samples
        if args.visualize and idx < 20:
            images_np = (sample["images"].permute(0, 2, 3, 1).numpy() * 255).astype(
                np.uint8
            )
            vis_path = os.path.join(args.output_dir, f"vis_{idx:06d}.jpg")
            visualize_detections(
                images=images_np,
                detections=detections,
                intrinsics=sample["intrinsics"].numpy(),
                extrinsics=sample["extrinsics"].numpy(),
                output_path=vis_path,
                class_names=DETECTION_CLASSES,
            )

    # Summary
    avg_fps = len(dataset) / total_time
    avg_dets = np.mean([len(r["scores"]) for r in all_results])
    logger.info(f"Inference complete: {len(dataset)} samples")
    logger.info(f"Average FPS: {avg_fps:.1f}")
    logger.info(f"Average detections per frame: {avg_dets:.1f}")

    # Save results
    results_path = os.path.join(args.output_dir, "detections.npz")
    np.savez_compressed(
        results_path,
        boxes=[r["boxes_3d"] for r in all_results],
        scores=[r["scores"] for r in all_results],
        labels=[r["labels"] for r in all_results],
    )
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
