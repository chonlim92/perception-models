"""
Online inference module with tracking for CenterPoint 3D object detection.

Supports single-frame and sequence modes with BEV visualization,
timing/FPS reporting, and multi-object tracking.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml

from .dataset import voxelize_points
from .model import build_model_from_config
from .tracker import CenterPointTracker

logger = logging.getLogger(__name__)


class CenterPointInference:
    """CenterPoint 3D object detection with online tracking."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        """
        Initialize CenterPoint inference pipeline.

        Args:
            config_path: Path to YAML configuration file.
            checkpoint_path: Path to model checkpoint (.pth).
            device: Inference device ('cuda' or 'cpu').
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.tracker = None
        self.frame_count = 0
        self.total_time = 0.0

        self.load_model()
        self._init_tracker()

    def load_model(self) -> None:
        """Load model checkpoint and set to eval mode."""
        logger.info("Building model from config: %s", self.config_path)
        self.model = build_model_from_config(self.config)

        logger.info("Loading checkpoint: %s", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully on %s", self.device)

    def _init_tracker(self) -> None:
        """Initialize the multi-object tracker."""
        tracker_config = self.config.get("tracker", {})
        self.tracker = CenterPointTracker(
            max_age=tracker_config.get("max_age", 3),
            min_hits=tracker_config.get("min_hits", 1),
            score_threshold=tracker_config.get("score_threshold", 0.1),
        )
        logger.info("Tracker initialized")

    def preprocess(self, points: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Voxelize point cloud for model input.

        Args:
            points: Raw point cloud array of shape (N, 4+) with columns
                    [x, y, z, intensity, ...].

        Returns:
            Dictionary with voxelized tensors ready for model forward pass.
        """
        voxel_config = self.config.get("voxelization", {})
        voxel_size = voxel_config.get("voxel_size", [0.075, 0.075, 0.2])
        point_cloud_range = voxel_config.get(
            "point_cloud_range", [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        )
        max_points_per_voxel = voxel_config.get("max_points_per_voxel", 10)
        max_voxels = voxel_config.get("max_voxels", 60000)

        voxels, coordinates, num_points_per_voxel = voxelize_points(
            points,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )

        batch_dict = {
            "voxels": torch.from_numpy(voxels).float().to(self.device),
            "coordinates": torch.from_numpy(
                np.pad(coordinates, ((0, 0), (1, 0)), mode="constant", constant_values=0)
            ).int().to(self.device),
            "num_points_per_voxel": torch.from_numpy(num_points_per_voxel)
            .int()
            .to(self.device),
            "batch_size": 1,
        }

        return batch_dict

    def detect(self, points: np.ndarray) -> List[Dict]:
        """
        Run full detection pipeline on a point cloud.

        Args:
            points: Raw point cloud array of shape (N, 4+).

        Returns:
            List of detection dicts, each containing:
                - 'box': [x, y, z, dx, dy, dz, yaw] (7,)
                - 'score': float confidence score
                - 'label': int class label
                - 'velocity': [vx, vy] (2,) estimated velocity
        """
        batch_dict = self.preprocess(points)

        with torch.no_grad():
            predictions = self.model(batch_dict)

        detections = self._decode_predictions(predictions)
        return detections

    def _decode_predictions(self, predictions: Dict) -> List[Dict]:
        """
        Decode raw model predictions into detection dicts and apply NMS.

        Args:
            predictions: Raw model output dictionary.

        Returns:
            List of filtered detection dicts after NMS.
        """
        if "final_boxes" in predictions:
            boxes = predictions["final_boxes"][0].cpu().numpy()
            scores = predictions["final_scores"][0].cpu().numpy()
            labels = predictions["final_labels"][0].cpu().numpy()
        elif "boxes" in predictions:
            boxes = predictions["boxes"][0].cpu().numpy()
            scores = predictions["scores"][0].cpu().numpy()
            labels = predictions["labels"][0].cpu().numpy()
        else:
            heatmap = predictions.get("heatmap", [None])[0]
            reg = predictions.get("reg", [None])[0]
            height = predictions.get("height", [None])[0]
            dim = predictions.get("dim", [None])[0]
            rot = predictions.get("rot", [None])[0]
            vel = predictions.get("vel", [None])[0]

            boxes, scores, labels = self._decode_heatmap(
                heatmap, reg, height, dim, rot, vel
            )

        if len(boxes) == 0:
            return []

        score_threshold = self.config.get("inference", {}).get("score_threshold", 0.1)
        mask = scores >= score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        if len(boxes) == 0:
            return []

        keep = self.nms_bev(boxes, scores, threshold=0.1)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        detections = []
        for i in range(len(boxes)):
            det = {
                "box": boxes[i][:7],
                "score": float(scores[i]),
                "label": int(labels[i]),
                "velocity": boxes[i][7:9] if boxes[i].shape[0] >= 9 else np.zeros(2),
            }
            detections.append(det)

        return detections

    def _decode_heatmap(
        self,
        heatmap: torch.Tensor,
        reg: torch.Tensor,
        height: torch.Tensor,
        dim: torch.Tensor,
        rot: torch.Tensor,
        vel: Optional[torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode CenterPoint heatmap-based predictions into boxes.

        Args:
            heatmap: (C, H, W) class heatmap.
            reg: (2, H, W) center offset regression.
            height: (1, H, W) height prediction.
            dim: (3, H, W) dimension prediction (dx, dy, dz).
            rot: (2, H, W) rotation prediction (sin, cos).
            vel: (2, H, W) velocity prediction or None.

        Returns:
            Tuple of (boxes, scores, labels) numpy arrays.
        """
        voxel_config = self.config.get("voxelization", {})
        point_cloud_range = voxel_config.get(
            "point_cloud_range", [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        )
        feature_map_stride = self.config.get("model", {}).get("feature_map_stride", 8)
        voxel_size = voxel_config.get("voxel_size", [0.075, 0.075, 0.2])

        heatmap_np = torch.sigmoid(heatmap).cpu().numpy()
        num_classes, feat_h, feat_w = heatmap_np.shape

        k = min(500, feat_h * feat_w)
        heatmap_flat = heatmap_np.reshape(num_classes, -1)

        all_boxes = []
        all_scores = []
        all_labels = []

        for cls_id in range(num_classes):
            cls_scores = heatmap_flat[cls_id]
            top_k_indices = np.argsort(cls_scores)[::-1][:k]

            for idx in top_k_indices:
                score = cls_scores[idx]
                if score < 0.1:
                    break

                yi = idx // feat_w
                xi = idx % feat_w

                cx = (xi + reg[0, yi, xi].cpu().item()) * feature_map_stride * voxel_size[0] + point_cloud_range[0]
                cy = (yi + reg[1, yi, xi].cpu().item()) * feature_map_stride * voxel_size[1] + point_cloud_range[1]
                cz = height[0, yi, xi].cpu().item()

                dx = dim[0, yi, xi].cpu().item()
                dy = dim[1, yi, xi].cpu().item()
                dz = dim[2, yi, xi].cpu().item()

                sin_val = rot[0, yi, xi].cpu().item()
                cos_val = rot[1, yi, xi].cpu().item()
                yaw = np.arctan2(sin_val, cos_val)

                if vel is not None:
                    vx = vel[0, yi, xi].cpu().item()
                    vy = vel[1, yi, xi].cpu().item()
                    box = np.array([cx, cy, cz, dx, dy, dz, yaw, vx, vy])
                else:
                    box = np.array([cx, cy, cz, dx, dy, dz, yaw, 0.0, 0.0])

                all_boxes.append(box)
                all_scores.append(score)
                all_labels.append(cls_id)

        if len(all_boxes) == 0:
            return np.zeros((0, 9)), np.zeros(0), np.zeros(0, dtype=np.int32)

        return (
            np.array(all_boxes),
            np.array(all_scores),
            np.array(all_labels, dtype=np.int32),
        )

    def nms_bev(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.1,
    ) -> np.ndarray:
        """
        BEV (Bird's Eye View) IoU-based Non-Maximum Suppression.

        Computes IoU in the BEV plane (x-y) using rotated bounding boxes.

        Args:
            boxes: (N, 7+) array with [x, y, z, dx, dy, dz, yaw, ...].
            scores: (N,) confidence scores.
            threshold: IoU threshold for suppression.

        Returns:
            Array of indices to keep.
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        cx = boxes[:, 0]
        cy = boxes[:, 1]
        dx = boxes[:, 3]
        dy = boxes[:, 4]
        yaw = boxes[:, 6]

        order = np.argsort(-scores)
        keep = []

        suppressed = np.zeros(len(boxes), dtype=bool)

        for i in range(len(order)):
            idx = order[i]
            if suppressed[idx]:
                continue
            keep.append(idx)

            for j in range(i + 1, len(order)):
                jdx = order[j]
                if suppressed[jdx]:
                    continue

                iou = self._bev_iou_single(
                    cx[idx], cy[idx], dx[idx], dy[idx], yaw[idx],
                    cx[jdx], cy[jdx], dx[jdx], dy[jdx], yaw[jdx],
                )

                if iou >= threshold:
                    suppressed[jdx] = True

        return np.array(keep, dtype=np.int64)

    def _bev_iou_single(
        self,
        cx1: float, cy1: float, dx1: float, dy1: float, yaw1: float,
        cx2: float, cy2: float, dx2: float, dy2: float, yaw2: float,
    ) -> float:
        """
        Compute BEV IoU between two rotated boxes using cv2.rotatedRectangleIntersection.

        Args:
            cx1, cy1, dx1, dy1, yaw1: Center, size, and yaw of box 1.
            cx2, cy2, dx2, dy2, yaw2: Center, size, and yaw of box 2.

        Returns:
            IoU value in [0, 1].
        """
        rect1 = ((cx1, cy1), (dx1, dy1), np.degrees(yaw1))
        rect2 = ((cx2, cy2), (dx2, dy2), np.degrees(yaw2))

        ret, points = cv2.rotatedRectangleIntersection(rect1, rect2)

        if ret == cv2.INTERSECT_NONE or points is None:
            return 0.0

        points = points.reshape(-1, 2)
        intersection_area = cv2.contourArea(np.array(points, dtype=np.float32))

        area1 = dx1 * dy1
        area2 = dx2 * dy2
        union_area = area1 + area2 - intersection_area

        if union_area <= 0:
            return 0.0

        return float(intersection_area / union_area)

    def track(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detection dicts from detect().

        Returns:
            List of tracked object dicts, each containing:
                - 'track_id': int unique track identifier
                - 'box': [x, y, z, dx, dy, dz, yaw] (7,)
                - 'score': float confidence
                - 'label': int class label
                - 'velocity': [vx, vy] (2,)
                - 'age': int number of frames this track has existed
        """
        tracked_objects = self.tracker.update(detections)
        return tracked_objects

    def run_frame(self, point_cloud_path: str) -> Dict:
        """
        Full pipeline for one frame: load, detect, track.

        Args:
            point_cloud_path: Path to point cloud file (.bin or .npy).

        Returns:
            Dictionary with:
                - 'detections': list of raw detections
                - 'tracks': list of tracked objects with IDs
                - 'frame_time': processing time in seconds
                - 'fps': instantaneous FPS
        """
        t_start = time.perf_counter()

        points = self._load_point_cloud(point_cloud_path)

        detections = self.detect(points)

        tracks = self.track(detections)

        t_end = time.perf_counter()
        frame_time = t_end - t_start

        self.frame_count += 1
        self.total_time += frame_time

        fps = 1.0 / frame_time if frame_time > 0 else 0.0
        avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0.0

        logger.info(
            "Frame %d: %d detections, %d tracks | %.1f ms (%.1f FPS, avg %.1f FPS)",
            self.frame_count,
            len(detections),
            len(tracks),
            frame_time * 1000,
            fps,
            avg_fps,
        )

        return {
            "detections": detections,
            "tracks": tracks,
            "frame_time": frame_time,
            "fps": fps,
            "points": points,
        }

    def run_sequence(self, frame_paths: List[str]) -> List[Dict]:
        """
        Process a sequence of frames with tracking.

        Args:
            frame_paths: Ordered list of point cloud file paths.

        Returns:
            List of per-frame result dicts from run_frame().
        """
        logger.info("Processing sequence of %d frames", len(frame_paths))
        self.frame_count = 0
        self.total_time = 0.0
        self.tracker.reset()

        results = []
        for path in frame_paths:
            result = self.run_frame(path)
            results.append(result)

        avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0.0
        logger.info(
            "Sequence complete: %d frames, avg %.1f FPS (%.1f ms/frame)",
            self.frame_count,
            avg_fps,
            (self.total_time / self.frame_count * 1000) if self.frame_count > 0 else 0,
        )

        return results

    def _load_point_cloud(self, path: str) -> np.ndarray:
        """
        Load point cloud from file.

        Supports .bin (float32, reshaped to Nx4) and .npy formats.

        Args:
            path: Path to point cloud file.

        Returns:
            Point cloud array of shape (N, 4+).
        """
        ext = Path(path).suffix.lower()
        if ext == ".bin":
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        elif ext == ".npy":
            points = np.load(path)
        elif ext == ".pcd":
            points = self._load_pcd(path)
        else:
            raise ValueError(f"Unsupported point cloud format: {ext}")

        logger.debug("Loaded %d points from %s", len(points), path)
        return points

    def _load_pcd(self, path: str) -> np.ndarray:
        """
        Load ASCII/binary PCD file (basic support).

        Args:
            path: Path to .pcd file.

        Returns:
            Point cloud array of shape (N, 4).
        """
        with open(path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                header_lines.append(line)
                if line.startswith("DATA"):
                    break

            num_points = 0
            data_type = "ascii"
            for line in header_lines:
                if line.startswith("POINTS"):
                    num_points = int(line.split()[1])
                if line.startswith("DATA"):
                    data_type = line.split()[1].lower()

            if data_type == "ascii":
                points = []
                for _ in range(num_points):
                    line = f.readline().decode("ascii", errors="ignore").strip()
                    if line:
                        vals = [float(v) for v in line.split()]
                        if len(vals) >= 4:
                            points.append(vals[:4])
                        elif len(vals) == 3:
                            points.append(vals + [0.0])
                return np.array(points, dtype=np.float32) if points else np.zeros((0, 4), dtype=np.float32)
            else:
                data = f.read()
                points = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
                return points[:num_points]

    def visualize_bev(
        self,
        points: np.ndarray,
        detections: List[Dict],
        tracks: List[Dict],
        output_path: str,
        bev_range: float = 54.0,
        bev_resolution: float = 0.1,
    ) -> np.ndarray:
        """
        Create BEV visualization image with detections and track IDs.

        Args:
            points: Point cloud array (N, 4+).
            detections: List of detection dicts.
            tracks: List of tracked object dicts with 'track_id'.
            output_path: Path to save the output image.
            bev_range: Spatial range in meters (symmetric around origin).
            bev_resolution: Meters per pixel.

        Returns:
            BEV image as numpy array (H, W, 3).
        """
        img_size = int(2 * bev_range / bev_resolution)
        bev_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Draw point cloud
        valid_mask = (
            (points[:, 0] > -bev_range) & (points[:, 0] < bev_range) &
            (points[:, 1] > -bev_range) & (points[:, 1] < bev_range)
        )
        valid_points = points[valid_mask]

        px = ((valid_points[:, 0] + bev_range) / bev_resolution).astype(np.int32)
        py = ((valid_points[:, 1] + bev_range) / bev_resolution).astype(np.int32)

        px = np.clip(px, 0, img_size - 1)
        py = np.clip(py, 0, img_size - 1)

        bev_img[py, px] = (100, 100, 100)

        # Color palette for different classes
        class_colors = [
            (0, 255, 0),    # Green - car
            (255, 255, 0),  # Cyan - truck
            (0, 0, 255),    # Red - pedestrian
            (255, 0, 255),  # Magenta - cyclist
            (0, 165, 255),  # Orange - other
            (255, 0, 0),    # Blue
            (128, 255, 0),  # Lime
            (0, 255, 255),  # Yellow
        ]

        # Draw detections as rotated boxes (thin lines)
        for det in detections:
            box = det["box"]
            label = det.get("label", 0)
            color = class_colors[label % len(class_colors)]
            self._draw_rotated_box(bev_img, box, color, bev_range, bev_resolution, thickness=1)

        # Draw tracked objects as thicker boxes with IDs
        for trk in tracks:
            box = trk["box"]
            track_id = trk["track_id"]
            label = trk.get("label", 0)
            color = class_colors[label % len(class_colors)]

            self._draw_rotated_box(bev_img, box, color, bev_range, bev_resolution, thickness=2)

            # Draw track ID text
            cx_px = int((box[0] + bev_range) / bev_resolution)
            cy_px = int((box[1] + bev_range) / bev_resolution)
            cx_px = np.clip(cx_px, 0, img_size - 1)
            cy_px = np.clip(cy_px, 0, img_size - 1)

            text = f"ID:{track_id}"
            cv2.putText(
                bev_img, text, (cx_px + 5, cy_px - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Draw velocity arrow
            velocity = trk.get("velocity", np.zeros(2))
            if np.linalg.norm(velocity) > 0.5:
                vx_px = int(velocity[0] / bev_resolution * 5)
                vy_px = int(velocity[1] / bev_resolution * 5)
                end_pt = (cx_px + vx_px, cy_px + vy_px)
                cv2.arrowedLine(
                    bev_img, (cx_px, cy_px), end_pt,
                    (0, 200, 200), 1, tipLength=0.3,
                )

        # Draw ego vehicle marker at center
        center = img_size // 2
        cv2.drawMarker(
            bev_img, (center, center), (0, 0, 255),
            cv2.MARKER_CROSS, 20, 2,
        )

        # Add frame info
        info_text = f"Dets: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(
            bev_img, info_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Save output
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, bev_img)
        logger.debug("BEV visualization saved to %s", output_path)

        return bev_img

    def _draw_rotated_box(
        self,
        img: np.ndarray,
        box: np.ndarray,
        color: Tuple[int, int, int],
        bev_range: float,
        bev_resolution: float,
        thickness: int = 1,
    ) -> None:
        """
        Draw a rotated bounding box on the BEV image.

        Args:
            img: BEV image array.
            box: [x, y, z, dx, dy, dz, yaw] box parameters.
            color: BGR color tuple.
            bev_range: Spatial range in meters.
            bev_resolution: Meters per pixel.
            thickness: Line thickness.
        """
        cx, cy = box[0], box[1]
        dx, dy = box[3], box[4]
        yaw = box[6]

        # Compute rotated corners
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        half_dx = dx / 2.0
        half_dy = dy / 2.0

        corners = np.array([
            [-half_dx, -half_dy],
            [half_dx, -half_dy],
            [half_dx, half_dy],
            [-half_dx, half_dy],
        ])

        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners = corners @ rotation.T

        corners[:, 0] += cx
        corners[:, 1] += cy

        # Convert to pixel coordinates
        corners_px = ((corners + bev_range) / bev_resolution).astype(np.int32)

        img_size = img.shape[0]
        corners_px = np.clip(corners_px, 0, img_size - 1)

        # Draw box edges
        for i in range(4):
            pt1 = tuple(corners_px[i])
            pt2 = tuple(corners_px[(i + 1) % 4])
            cv2.line(img, pt1, pt2, color, thickness)

        # Draw heading indicator (front of box)
        front_center = ((corners_px[1] + corners_px[2]) / 2).astype(np.int32)
        box_center = ((corners_px[0] + corners_px[2]) / 2).astype(np.int32)
        cv2.line(img, tuple(box_center), tuple(front_center), color, thickness + 1)

    def get_timing_stats(self) -> Dict:
        """
        Get timing statistics for the inference session.

        Returns:
            Dictionary with timing info:
                - 'frame_count': total frames processed
                - 'total_time': total processing time in seconds
                - 'avg_fps': average FPS
                - 'avg_frame_time_ms': average time per frame in ms
        """
        avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0.0
        avg_frame_time = (self.total_time / self.frame_count * 1000) if self.frame_count > 0 else 0.0

        return {
            "frame_count": self.frame_count,
            "total_time": self.total_time,
            "avg_fps": avg_fps,
            "avg_frame_time_ms": avg_frame_time,
        }


def main():
    """CLI entry point for CenterPoint inference."""
    parser = argparse.ArgumentParser(
        description="CenterPoint 3D Object Detection - Online Inference with Tracking"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to point cloud file (.bin/.npy) or directory of files for sequence mode",
    )
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory for results and visualizations",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Enable BEV visualization output",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    os.makedirs(args.output, exist_ok=True)

    logger.info("Initializing CenterPoint inference pipeline")
    inference = CenterPointInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    input_path = Path(args.input)

    if input_path.is_file():
        # Single-frame mode
        logger.info("Running single-frame inference on: %s", input_path)
        result = inference.run_frame(str(input_path))

        if args.visualize:
            vis_path = os.path.join(args.output, "bev_detection.png")
            inference.visualize_bev(
                result["points"], result["detections"], result["tracks"], vis_path
            )

        # Print results
        print(f"\n{'='*60}")
        print(f"CenterPoint Inference Results")
        print(f"{'='*60}")
        print(f"Input: {input_path}")
        print(f"Detections: {len(result['detections'])}")
        print(f"Tracks: {len(result['tracks'])}")
        print(f"Frame time: {result['frame_time']*1000:.1f} ms ({result['fps']:.1f} FPS)")
        print(f"{'='*60}")

        for trk in result["tracks"]:
            box = trk["box"]
            print(
                f"  Track ID {trk['track_id']:3d} | "
                f"class={trk['label']} score={trk['score']:.2f} | "
                f"pos=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}) "
                f"size=({box[3]:.1f}, {box[4]:.1f}, {box[5]:.1f}) "
                f"yaw={box[6]:.2f} | "
                f"vel=({trk['velocity'][0]:.1f}, {trk['velocity'][1]:.1f})"
            )

    elif input_path.is_dir():
        # Sequence mode
        extensions = [".bin", ".npy", ".pcd"]
        frame_paths = sorted([
            str(p) for p in input_path.iterdir()
            if p.suffix.lower() in extensions
        ])

        if not frame_paths:
            logger.error("No point cloud files found in %s", input_path)
            return

        logger.info("Running sequence inference on %d frames from: %s", len(frame_paths), input_path)
        results = inference.run_sequence(frame_paths)

        if args.visualize:
            vis_dir = os.path.join(args.output, "bev_visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            for i, result in enumerate(results):
                vis_path = os.path.join(vis_dir, f"frame_{i:06d}.png")
                inference.visualize_bev(
                    result["points"], result["detections"], result["tracks"], vis_path
                )

        # Print timing summary
        stats = inference.get_timing_stats()
        print(f"\n{'='*60}")
        print(f"CenterPoint Sequence Inference Summary")
        print(f"{'='*60}")
        print(f"Input directory: {input_path}")
        print(f"Frames processed: {stats['frame_count']}")
        print(f"Total time: {stats['total_time']:.2f} s")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"Average frame time: {stats['avg_frame_time_ms']:.1f} ms")
        if args.visualize:
            print(f"Visualizations saved to: {vis_dir}")
        print(f"{'='*60}")

        # Print per-frame track counts
        total_tracks = sum(len(r["tracks"]) for r in results)
        unique_ids = set()
        for r in results:
            for trk in r["tracks"]:
                unique_ids.add(trk["track_id"])
        print(f"Total track instances: {total_tracks}")
        print(f"Unique track IDs: {len(unique_ids)}")

    else:
        logger.error("Input path does not exist: %s", input_path)
        return


if __name__ == "__main__":
    main()
