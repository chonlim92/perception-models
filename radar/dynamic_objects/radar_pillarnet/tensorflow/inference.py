"""
RadarPillarNet inference script.

Features:
- Load model from weights, checkpoint, or SavedModel
- Single-frame and batch inference
- TF-TRT (TensorRT) optimization option
- Pre/post-processing pipeline
- NMS via tf.image.combined_non_max_suppression
- Output to JSON and CSV formats
- Timing measurement with warmup
- Throughput benchmarking
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from model import RadarPillarNet, DEFAULT_CONFIG, build_radar_pillarnet


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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BoundingBox3D:
    """3D bounding box detection result."""

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": [self.center_x, self.center_y, self.center_z],
            "size": [self.width, self.length, self.height],
            "yaw": self.yaw,
            "velocity": [self.velocity_x, self.velocity_y],
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    def to_csv_row(self) -> List[str]:
        return [
            self.class_name,
            f"{self.score:.4f}",
            f"{self.center_x:.3f}",
            f"{self.center_y:.3f}",
            f"{self.center_z:.3f}",
            f"{self.width:.3f}",
            f"{self.length:.3f}",
            f"{self.height:.3f}",
            f"{np.degrees(self.yaw):.2f}",
            f"{self.velocity_x:.3f}",
            f"{self.velocity_y:.3f}",
        ]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class RadarPreprocessor:
    """
    Preprocess raw radar point cloud into pillar format for model input.

    Takes raw radar points (x, y, z, RCS, vr, dt) and converts to:
    - pillar_features: (max_pillars, max_points, 9) with augmented features
    - pillar_mask: (max_pillars, max_points) validity mask
    - pillar_coords: (max_pillars, 2) grid coordinates
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.x_min = config["x_min"]
        self.x_max = config["x_max"]
        self.y_min = config["y_min"]
        self.y_max = config["y_max"]
        self.z_min = config["z_min"]
        self.z_max = config["z_max"]
        self.pillar_x = config["pillar_x_size"]
        self.pillar_y = config["pillar_y_size"]
        self.grid_x = config["grid_x"]
        self.grid_y = config["grid_y"]
        self.max_pillars = config["max_pillars"]
        self.max_points = config["max_points_per_pillar"]

    def process(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert raw radar points to pillar representation.

        Args:
            points: (N, 6+) array with columns [x, y, z, RCS, vr, dt, ...]
        Returns:
            dict with pillar_features, pillar_mask, pillar_coords
        """
        # Filter points within range
        mask_x = (points[:, 0] >= self.x_min) & (points[:, 0] < self.x_max)
        mask_y = (points[:, 1] >= self.y_min) & (points[:, 1] < self.y_max)
        mask_z = (points[:, 2] >= self.z_min) & (points[:, 2] < self.z_max)
        valid = mask_x & mask_y & mask_z
        points = points[valid]

        if len(points) == 0:
            return {
                "pillar_features": np.zeros((self.max_pillars, self.max_points, 9), dtype=np.float32),
                "pillar_mask": np.zeros((self.max_pillars, self.max_points), dtype=np.float32),
                "pillar_coords": np.zeros((self.max_pillars, 2), dtype=np.int32),
            }

        # Compute grid indices for each point
        ix = ((points[:, 0] - self.x_min) / self.pillar_x).astype(np.int32)
        iy = ((points[:, 1] - self.y_min) / self.pillar_y).astype(np.int32)

        # Clip to valid range
        ix = np.clip(ix, 0, self.grid_x - 1)
        iy = np.clip(iy, 0, self.grid_y - 1)

        # Group points by pillar
        pillar_ids = iy * self.grid_x + ix  # unique pillar ID
        unique_pillars, inverse_idx = np.unique(pillar_ids, return_inverse=True)
        n_pillars = min(len(unique_pillars), self.max_pillars)

        # If too many pillars, randomly select subset
        if len(unique_pillars) > self.max_pillars:
            selected = np.random.choice(len(unique_pillars), self.max_pillars, replace=False)
            selected.sort()
        else:
            selected = np.arange(len(unique_pillars))

        # Allocate output arrays
        pillar_features = np.zeros((self.max_pillars, self.max_points, 9), dtype=np.float32)
        pillar_mask = np.zeros((self.max_pillars, self.max_points), dtype=np.float32)
        pillar_coords = np.zeros((self.max_pillars, 2), dtype=np.int32)

        for out_idx, pillar_sel_idx in enumerate(selected):
            pillar_id = unique_pillars[pillar_sel_idx]
            point_mask = inverse_idx == pillar_sel_idx
            pillar_points = points[point_mask]

            # Limit points per pillar
            if len(pillar_points) > self.max_points:
                choice = np.random.choice(len(pillar_points), self.max_points, replace=False)
                pillar_points = pillar_points[choice]

            n_pts = len(pillar_points)

            # Compute pillar center
            x_center = pillar_points[:, 0].mean()
            y_center = pillar_points[:, 1].mean()
            z_center = pillar_points[:, 2].mean()

            # Build features: [x, y, z, RCS, vr, dt, x_c, y_c, z_c]
            # where x_c = x - x_center, etc.
            features = np.zeros((n_pts, 9), dtype=np.float32)
            features[:, 0:6] = pillar_points[:, 0:6]  # x, y, z, RCS, vr, dt
            features[:, 6] = pillar_points[:, 0] - x_center  # x_c
            features[:, 7] = pillar_points[:, 1] - y_center  # y_c
            features[:, 8] = pillar_points[:, 2] - z_center  # z_c

            pillar_features[out_idx, :n_pts] = features
            pillar_mask[out_idx, :n_pts] = 1.0

            # Grid coordinates
            grid_ix = int(pillar_id % self.grid_x)
            grid_iy = int(pillar_id // self.grid_x)
            pillar_coords[out_idx] = [grid_ix, grid_iy]

        return {
            "pillar_features": pillar_features,
            "pillar_mask": pillar_mask,
            "pillar_coords": pillar_coords,
        }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def nms_bev(
    boxes_3d: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    velocities: np.ndarray,
    iou_threshold: float = 0.2,
    max_detections: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    BEV NMS using tf.image.combined_non_max_suppression.

    Args:
        boxes_3d: (N, 7) decoded boxes [x, y, z, w, l, h, yaw]
        scores: (N,) confidence scores
        labels: (N,) class IDs
        velocities: (N, 2) predicted velocities
        iou_threshold: NMS IoU threshold
        max_detections: maximum output detections
    Returns:
        Filtered (boxes, scores, labels, velocities)
    """
    if len(boxes_3d) == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 2), dtype=np.float32),
        )

    # Convert 3D boxes to 2D BEV boxes for NMS: [y1, x1, y2, x2]
    cx = boxes_3d[:, 0]
    cy = boxes_3d[:, 1]
    bw = boxes_3d[:, 3]
    bl = boxes_3d[:, 4]

    bev_boxes = np.stack([
        cy - bl / 2,  # y1
        cx - bw / 2,  # x1
        cy + bl / 2,  # y2
        cx + bw / 2,  # x2
    ], axis=-1)  # (N, 4)

    # Use TensorFlow NMS
    nms_indices = tf.image.non_max_suppression(
        tf.constant(bev_boxes, dtype=tf.float32),
        tf.constant(scores, dtype=tf.float32),
        max_output_size=max_detections,
        iou_threshold=iou_threshold,
    ).numpy()

    return (
        boxes_3d[nms_indices],
        scores[nms_indices],
        labels[nms_indices],
        velocities[nms_indices],
    )


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------


class RadarPillarNetInference:
    """
    Production inference engine for RadarPillarNet.

    Supports loading from:
    - Keras weights (.h5 / .weights.h5)
    - TF checkpoint directory
    - SavedModel directory
    - TF-TRT optimized SavedModel
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.3,
        nms_iou_threshold: float = 0.2,
        max_detections: int = 500,
        use_trt: bool = False,
        trt_precision: str = "FP16",
        device: str = "gpu",
    ) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.use_trt = use_trt
        self.trt_precision = trt_precision

        # Device setup
        if device == "cpu":
            tf.config.set_visible_devices([], "GPU")

        # Preprocessor
        self.preprocessor = RadarPreprocessor(self.config)

        # Load model
        self.model = self._load_model(model_path)

        # Warmup
        self._warmup()

    def _load_model(self, model_path: str) -> Any:
        """Load model from various formats."""
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb")):
            if self.use_trt:
                return self._load_trt_model(model_path)
            print(f"[INFO] Loading SavedModel from: {model_path}")
            loaded = tf.saved_model.load(model_path)
            self._serve_fn = loaded.signatures.get("serving_default")
            if self._serve_fn is None:
                self._serve_fn = loaded.__call__
            return loaded
        else:
            print(f"[INFO] Building model and loading weights from: {model_path}")
            model = build_radar_pillarnet(config=self.config)

            # Build with dummy
            dummy = self._create_dummy_input(batch_size=1)
            _ = model(dummy, training=False)

            # Load weights
            if model_path.endswith(".h5") or model_path.endswith(".weights.h5"):
                model.load_weights(model_path)
            else:
                checkpoint = tf.train.Checkpoint(model=model)
                latest = tf.train.latest_checkpoint(model_path)
                if latest:
                    checkpoint.restore(latest).expect_partial()
                else:
                    raise ValueError(f"No valid model found at: {model_path}")

            print(f"[INFO] Model loaded: {model.count_params():,} parameters")
            self._serve_fn = None
            return model

    def _load_trt_model(self, saved_model_path: str) -> Any:
        """Load and optimize model with TF-TRT."""
        try:
            from tensorflow.python.compiler.tensorrt import trt_convert as trt

            print(f"[INFO] Converting SavedModel to TF-TRT ({self.trt_precision})...")

            precision_map = {
                "FP32": trt.TrtPrecisionMode.FP32,
                "FP16": trt.TrtPrecisionMode.FP16,
                "INT8": trt.TrtPrecisionMode.INT8,
            }

            conversion_params = trt.TrtConversionParams(
                precision_mode=precision_map.get(self.trt_precision, trt.TrtPrecisionMode.FP16),
                max_workspace_size_bytes=1 << 30,  # 1 GB
                maximum_cached_engines=1,
            )

            converter = trt.TrtGraphConverterV2(
                input_saved_model_dir=saved_model_path,
                conversion_params=conversion_params,
            )
            converter.convert()

            # Build with dummy input shape for optimization
            max_p = self.config["max_pillars"]
            max_pts = self.config["max_points_per_pillar"]

            def input_fn():
                yield {
                    "pillar_features": tf.zeros([1, max_p, max_pts, 9], dtype=tf.float32),
                    "pillar_mask": tf.zeros([1, max_p, max_pts], dtype=tf.float32),
                    "pillar_coords": tf.zeros([1, max_p, 2], dtype=tf.int32),
                }

            converter.build(input_fn=input_fn)

            # Save TRT-optimized model
            trt_output_dir = saved_model_path + "_trt"
            converter.save(trt_output_dir)
            print(f"[INFO] TF-TRT model saved to: {trt_output_dir}")

            # Load optimized model
            loaded = tf.saved_model.load(trt_output_dir)
            self._serve_fn = loaded.signatures.get("serving_default")
            return loaded

        except ImportError:
            print("[WARN] TensorRT not available. Loading standard SavedModel.")
            loaded = tf.saved_model.load(saved_model_path)
            self._serve_fn = loaded.signatures.get("serving_default")
            return loaded

    def _create_dummy_input(self, batch_size: int = 1) -> Dict[str, tf.Tensor]:
        """Create dummy input for model warmup."""
        cfg = self.config
        return {
            "pillar_features": tf.zeros([batch_size, cfg["max_pillars"], cfg["max_points_per_pillar"], 9]),
            "pillar_mask": tf.zeros([batch_size, cfg["max_pillars"], cfg["max_points_per_pillar"]]),
            "pillar_coords": tf.zeros([batch_size, cfg["max_pillars"], 2], dtype=tf.int32),
        }

    def _warmup(self, num_warmup: int = 3) -> None:
        """Warm up model with dummy inference."""
        print("[INFO] Warming up model...")
        dummy = self._create_dummy_input(batch_size=1)
        for _ in range(num_warmup):
            self._run_model(dummy)
        print("[INFO] Warmup complete")

    def _run_model(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Run model forward pass."""
        if self._serve_fn is not None:
            # SavedModel serving function
            result = self._serve_fn(**inputs)
            return result
        else:
            # Keras model
            return self.model(inputs, training=False)

    @tf.function(reduce_retracing=True)
    def _inference_step(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """TF function for optimized inference."""
        if isinstance(self.model, tf.keras.Model):
            return self.model(inputs, training=False)
        else:
            return self._serve_fn(**inputs)

    def predict_raw(
        self,
        radar_points: np.ndarray,
    ) -> Tuple[List[BoundingBox3D], float]:
        """
        Run inference from raw radar point cloud.

        Args:
            radar_points: (N, 6+) array [x, y, z, RCS, vr, dt, ...]
        Returns:
            (detections, inference_time_ms)
        """
        # Preprocess
        preprocess_start = time.perf_counter()
        pillar_data = self.preprocessor.process(radar_points)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000.0

        # Add batch dimension
        inputs = {
            "pillar_features": tf.constant(pillar_data["pillar_features"][np.newaxis], dtype=tf.float32),
            "pillar_mask": tf.constant(pillar_data["pillar_mask"][np.newaxis], dtype=tf.float32),
            "pillar_coords": tf.constant(pillar_data["pillar_coords"][np.newaxis], dtype=tf.int32),
        }

        # Inference
        infer_start = time.perf_counter()
        predictions = self._inference_step(inputs)
        # Force synchronization
        for v in predictions.values():
            _ = v.numpy()
        infer_time = (time.perf_counter() - infer_start) * 1000.0

        # Post-process
        post_start = time.perf_counter()
        detections = self._postprocess(predictions)
        post_time = (time.perf_counter() - post_start) * 1000.0

        total_time = preprocess_time + infer_time + post_time
        return detections, total_time

    def predict_pillars(
        self,
        pillar_features: np.ndarray,
        pillar_mask: np.ndarray,
        pillar_coords: np.ndarray,
    ) -> Tuple[List[BoundingBox3D], float]:
        """
        Run inference from pre-computed pillar data.

        Args:
            pillar_features: (max_pillars, max_pts, 9) or (B, max_pillars, max_pts, 9)
            pillar_mask: (max_pillars, max_pts) or batched
            pillar_coords: (max_pillars, 2) or batched
        Returns:
            (detections, inference_time_ms)
        """
        # Add batch dim if needed
        if pillar_features.ndim == 3:
            pillar_features = pillar_features[np.newaxis]
        if pillar_mask.ndim == 2:
            pillar_mask = pillar_mask[np.newaxis]
        if pillar_coords.ndim == 2:
            pillar_coords = pillar_coords[np.newaxis]

        inputs = {
            "pillar_features": tf.constant(pillar_features, dtype=tf.float32),
            "pillar_mask": tf.constant(pillar_mask, dtype=tf.float32),
            "pillar_coords": tf.constant(pillar_coords, dtype=tf.int32),
        }

        start = time.perf_counter()
        predictions = self._inference_step(inputs)
        for v in predictions.values():
            _ = v.numpy()
        infer_time = (time.perf_counter() - start) * 1000.0

        detections = self._postprocess(predictions)
        return detections, infer_time

    def predict_batch(
        self,
        batch_inputs: Dict[str, np.ndarray],
    ) -> Tuple[List[List[BoundingBox3D]], float]:
        """
        Run batch inference.

        Args:
            batch_inputs: dict with batched numpy arrays
        Returns:
            (list of detection lists, total_time_ms)
        """
        inputs = {
            k: tf.constant(v, dtype=tf.int32 if k == "pillar_coords" else tf.float32)
            for k, v in batch_inputs.items()
        }

        start = time.perf_counter()
        predictions = self._inference_step(inputs)
        for v in predictions.values():
            _ = v.numpy()
        infer_time = (time.perf_counter() - start) * 1000.0

        # Post-process each batch element
        batch_size = predictions["cls_preds"].shape[0]
        all_detections: List[List[BoundingBox3D]] = []

        for b in range(batch_size):
            single_preds = {k: v[b:b+1] for k, v in predictions.items()}
            dets = self._postprocess(single_preds)
            all_detections.append(dets)

        return all_detections, infer_time

    def _postprocess(self, predictions: Dict[str, tf.Tensor]) -> List[BoundingBox3D]:
        """
        Post-process model predictions to BoundingBox3D list.

        Applies sigmoid, anchor decoding, score filtering, and NMS.
        """
        cls_preds = predictions["cls_preds"].numpy()   # (B, H, W, A, C)
        box_preds = predictions["box_preds"].numpy()   # (B, H, W, A, 7)
        vel_preds = predictions["vel_preds"].numpy()   # (B, H, W, A, 2)
        dir_preds = predictions["dir_preds"].numpy()   # (B, H, W, A, 2)

        cfg = self.config
        b = 0  # Process first batch element

        h, w = cls_preds.shape[1], cls_preds.shape[2]
        num_anchors = cls_preds.shape[3]
        num_classes = cls_preds.shape[4]

        # Generate anchors
        anchors = self._generate_anchors_np(h, w)  # (H, W, A, 7)

        # Flatten
        cls_flat = cls_preds[b].reshape(-1, num_classes)  # (H*W*A, C)
        box_flat = box_preds[b].reshape(-1, 7)
        vel_flat = vel_preds[b].reshape(-1, 2)
        dir_flat = dir_preds[b].reshape(-1, 2)
        anchors_flat = anchors.reshape(-1, 7)

        # Sigmoid scores
        scores_all = 1.0 / (1.0 + np.exp(-cls_flat))  # sigmoid
        max_scores = scores_all.max(axis=-1)
        max_labels = scores_all.argmax(axis=-1)

        # Score threshold filter
        valid = max_scores > self.score_threshold
        if not np.any(valid):
            return []

        scores_valid = max_scores[valid]
        labels_valid = max_labels[valid]
        box_valid = box_flat[valid]
        vel_valid = vel_flat[valid]
        dir_valid = dir_flat[valid]
        anchors_valid = anchors_flat[valid]

        # Decode boxes
        decoded = self._decode_boxes_np(box_valid, anchors_valid)

        # Apply direction correction
        dir_labels = dir_valid.argmax(axis=-1)
        decoded[:, 6] += dir_labels * np.pi
        decoded[:, 6] = np.arctan2(np.sin(decoded[:, 6]), np.cos(decoded[:, 6]))

        # NMS
        boxes_nms, scores_nms, labels_nms, vels_nms = nms_bev(
            decoded, scores_valid, labels_valid, vel_valid,
            iou_threshold=self.nms_iou_threshold,
            max_detections=self.max_detections,
        )

        # Convert to BoundingBox3D
        detections: List[BoundingBox3D] = []
        for i in range(len(boxes_nms)):
            box = boxes_nms[i]
            label = int(labels_nms[i])
            class_name = NUSCENES_CLASSES[label] if 0 <= label < len(NUSCENES_CLASSES) else "unknown"

            det = BoundingBox3D(
                center_x=float(box[0]),
                center_y=float(box[1]),
                center_z=float(box[2]),
                width=float(box[3]),
                length=float(box[4]),
                height=float(box[5]),
                yaw=float(box[6]),
                velocity_x=float(vels_nms[i, 0]),
                velocity_y=float(vels_nms[i, 1]),
                score=float(scores_nms[i]),
                class_id=label,
                class_name=class_name,
            )
            detections.append(det)

        return detections

    def _generate_anchors_np(self, h: int, w: int) -> np.ndarray:
        """Generate anchors in numpy for post-processing."""
        cfg = self.config
        x_min, x_max = cfg["x_min"], cfg["x_max"]
        y_min, y_max = cfg["y_min"], cfg["y_max"]
        num_anchors = cfg["num_anchors_per_location"]

        x_res = (x_max - x_min) / w
        y_res = (y_max - y_min) / h

        xs = np.arange(w, dtype=np.float32) * x_res + x_min + x_res / 2
        ys = np.arange(h, dtype=np.float32) * y_res + y_min + y_res / 2

        grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W)

        anchors = np.zeros((h, w, num_anchors, 7), dtype=np.float32)
        anchor_dims = np.array([[4.7, 2.0, 1.7], [4.7, 2.0, 1.7]], dtype=np.float32)
        anchor_yaws = np.array([0.0, np.pi / 2], dtype=np.float32)
        anchor_z = -1.0

        for a in range(num_anchors):
            anchors[:, :, a, 0] = grid_x
            anchors[:, :, a, 1] = grid_y
            anchors[:, :, a, 2] = anchor_z
            anchors[:, :, a, 3] = anchor_dims[a, 0]
            anchors[:, :, a, 4] = anchor_dims[a, 1]
            anchors[:, :, a, 5] = anchor_dims[a, 2]
            anchors[:, :, a, 6] = anchor_yaws[a]

        return anchors

    def _decode_boxes_np(self, box_preds: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Decode box predictions relative to anchors (numpy)."""
        xa, ya, za = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        wa, la, ha = anchors[:, 3], anchors[:, 4], anchors[:, 5]
        yaw_a = anchors[:, 6]

        diag = np.sqrt(wa ** 2 + la ** 2)

        dx, dy, dz = box_preds[:, 0], box_preds[:, 1], box_preds[:, 2]
        dw, dl, dh = box_preds[:, 3], box_preds[:, 4], box_preds[:, 5]
        dyaw = box_preds[:, 6]

        x = xa + dx * diag
        y = ya + dy * diag
        z = za + dz * ha
        w = wa * np.exp(dw)
        l = la * np.exp(dl)
        h = ha * np.exp(dh)
        yaw = yaw_a + dyaw

        return np.stack([x, y, z, w, l, h, yaw], axis=-1)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def save_detections_json(
    detections: List[BoundingBox3D],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save detections to JSON file."""
    output = {
        "num_detections": len(detections),
        "detections": [d.to_dict() for d in detections],
    }
    if metadata:
        output["metadata"] = metadata

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def save_detections_csv(
    detections: List[BoundingBox3D],
    output_path: str,
) -> None:
    """Save detections to CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class", "score", "x", "y", "z", "width", "length", "height",
            "yaw_deg", "vx", "vy",
        ])
        for det in detections:
            writer.writerow(det.to_csv_row())


# ---------------------------------------------------------------------------
# Throughput benchmarking
# ---------------------------------------------------------------------------


def benchmark_throughput(
    model_path: str,
    config: Optional[Dict[str, Any]] = None,
    batch_size: int = 1,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_trt: bool = False,
    device: str = "gpu",
) -> Dict[str, float]:
    """
    Measure inference throughput and latency.

    Returns dict with timing statistics.
    """
    model_config = {**DEFAULT_CONFIG, **(config or {})}
    cfg = model_config

    engine = RadarPillarNetInference(
        model_path=model_path,
        config=model_config,
        use_trt=use_trt,
        device=device,
    )

    # Synthetic batch
    batch_inputs = {
        "pillar_features": np.random.randn(
            batch_size, cfg["max_pillars"], cfg["max_points_per_pillar"], 9
        ).astype(np.float32),
        "pillar_mask": np.ones(
            (batch_size, cfg["max_pillars"], cfg["max_points_per_pillar"])
        ).astype(np.float32),
        "pillar_coords": np.random.randint(
            0, cfg["grid_x"], (batch_size, cfg["max_pillars"], 2)
        ).astype(np.int32),
    }

    # Warmup
    print(f"[INFO] Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        engine.predict_batch(batch_inputs)

    # Timed iterations
    print(f"[INFO] Running {num_iterations} timed iterations (batch_size={batch_size})...")
    latencies: List[float] = []

    for _ in range(num_iterations):
        _, elapsed_ms = engine.predict_batch(batch_inputs)
        latencies.append(elapsed_ms)

    latencies_arr = np.array(latencies)
    total_samples = num_iterations * batch_size
    total_time_s = np.sum(latencies_arr) / 1000.0

    stats = {
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "device": device,
        "trt_enabled": use_trt,
        "mean_latency_ms": float(np.mean(latencies_arr)),
        "median_latency_ms": float(np.median(latencies_arr)),
        "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
        "p99_latency_ms": float(np.percentile(latencies_arr, 99)),
        "min_latency_ms": float(np.min(latencies_arr)),
        "max_latency_ms": float(np.max(latencies_arr)),
        "std_latency_ms": float(np.std(latencies_arr)),
        "throughput_fps": total_samples / total_time_s,
    }

    print(f"\n{'=' * 50}")
    print(f"{'RadarPillarNet Benchmark Results':^50}")
    print(f"{'=' * 50}")
    print(f"  Device:           {device}" + (" + TRT" if use_trt else ""))
    print(f"  Batch size:       {batch_size}")
    print(f"  Mean latency:     {stats['mean_latency_ms']:.2f} ms")
    print(f"  Median latency:   {stats['median_latency_ms']:.2f} ms")
    print(f"  P95 latency:      {stats['p95_latency_ms']:.2f} ms")
    print(f"  P99 latency:      {stats['p99_latency_ms']:.2f} ms")
    print(f"  Throughput:       {stats['throughput_fps']:.1f} FPS")
    print(f"{'=' * 50}")

    return stats


# ---------------------------------------------------------------------------
# Single-sample inference from file
# ---------------------------------------------------------------------------


def infer_from_file(
    model_path: str,
    input_path: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
    output_format: str = "json",
    score_threshold: float = 0.3,
    use_trt: bool = False,
) -> List[BoundingBox3D]:
    """
    Run inference on a single radar point cloud file.

    Supported input formats:
    - .npz with 'points' key (N, 6+) or pre-computed pillars
    - .bin binary point cloud (N, 6 float32)
    - .npy numpy array

    Args:
        model_path: path to model
        input_path: path to radar data file
        output_path: path to save results
        config: model config overrides
        output_format: "json" or "csv"
        score_threshold: detection threshold
        use_trt: enable TF-TRT optimization
    Returns:
        List of detected 3D boxes
    """
    model_config = {**DEFAULT_CONFIG, **(config or {})}

    engine = RadarPillarNetInference(
        model_path=model_path,
        config=model_config,
        score_threshold=score_threshold,
        use_trt=use_trt,
    )

    # Load input data
    input_file = Path(input_path)
    if input_file.suffix == ".npz":
        data = np.load(str(input_file))
        if "points" in data:
            # Raw point cloud
            points = data["points"]
            detections, elapsed_ms = engine.predict_raw(points)
        elif "pillar_features" in data:
            # Pre-computed pillars
            detections, elapsed_ms = engine.predict_pillars(
                data["pillar_features"],
                data["pillar_mask"],
                data["pillar_coords"],
            )
        else:
            raise ValueError(f"Unsupported .npz format. Keys: {list(data.keys())}")
    elif input_file.suffix == ".bin":
        points = np.fromfile(str(input_file), dtype=np.float32).reshape(-1, 6)
        detections, elapsed_ms = engine.predict_raw(points)
    elif input_file.suffix == ".npy":
        points = np.load(str(input_file))
        detections, elapsed_ms = engine.predict_raw(points)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")

    # Print results
    print(f"\n[INFO] Detected {len(detections)} objects in {elapsed_ms:.1f} ms")
    print(f"{'Class':<20s} | {'Score':>6s} | {'Position (x,y,z)':>20s} | {'Size (w,l,h)':>18s} | {'Yaw':>6s}")
    print("-" * 85)
    for det in detections[:20]:  # Print top 20
        print(
            f"  {det.class_name:<20s} | {det.score:6.3f} | "
            f"({det.center_x:6.1f}, {det.center_y:6.1f}, {det.center_z:5.1f}) | "
            f"({det.width:5.1f}, {det.length:5.1f}, {det.height:4.1f}) | "
            f"{np.degrees(det.yaw):6.1f}"
        )
    if len(detections) > 20:
        print(f"  ... and {len(detections) - 20} more detections")

    # Save results
    metadata = {
        "input_file": str(input_path),
        "model_path": model_path,
        "inference_time_ms": elapsed_ms,
        "score_threshold": score_threshold,
    }

    if output_format == "csv":
        save_detections_csv(detections, output_path)
    else:
        save_detections_json(detections, output_path, metadata=metadata)

    print(f"[INFO] Results saved to: {output_path}")
    return detections


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RadarPillarNet inference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single-sample inference
    infer_parser = subparsers.add_parser("infer", help="Run inference on a single file")
    infer_parser.add_argument("--model", type=str, required=True, help="Model path (weights/checkpoint/SavedModel)")
    infer_parser.add_argument("--input", type=str, required=True, help="Input radar data file (.npz/.bin/.npy)")
    infer_parser.add_argument("--output", type=str, default="./output/detections.json", help="Output path")
    infer_parser.add_argument("--format", type=str, default="json", choices=["json", "csv"], help="Output format")
    infer_parser.add_argument("--score-threshold", type=float, default=0.3, help="Score threshold")
    infer_parser.add_argument("--use-trt", action="store_true", help="Enable TF-TRT optimization")
    infer_parser.add_argument("--config", type=str, default=None, help="Model config JSON path")

    # Batch inference
    batch_parser = subparsers.add_parser("batch", help="Run batch inference on a directory")
    batch_parser.add_argument("--model", type=str, required=True, help="Model path")
    batch_parser.add_argument("--input-dir", type=str, required=True, help="Directory with radar data files")
    batch_parser.add_argument("--output-dir", type=str, default="./output/batch", help="Output directory")
    batch_parser.add_argument("--format", type=str, default="json", choices=["json", "csv"])
    batch_parser.add_argument("--score-threshold", type=float, default=0.3)
    batch_parser.add_argument("--use-trt", action="store_true")
    batch_parser.add_argument("--config", type=str, default=None)

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run throughput benchmark")
    bench_parser.add_argument("--model", type=str, required=True, help="Model path")
    bench_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    bench_parser.add_argument("--iterations", type=int, default=100, help="Timed iterations")
    bench_parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    bench_parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    bench_parser.add_argument("--use-trt", action="store_true")
    bench_parser.add_argument("--config", type=str, default=None)
    bench_parser.add_argument("--output", type=str, default=None, help="Save benchmark results JSON")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config
    ext_config = None
    if hasattr(args, "config") and args.config:
        with open(args.config, "r") as f:
            ext_config = json.load(f)
        ext_config = ext_config.get("model", ext_config)

    if args.command == "infer":
        infer_from_file(
            model_path=args.model,
            input_path=args.input,
            output_path=args.output,
            config=ext_config,
            output_format=args.format,
            score_threshold=args.score_threshold,
            use_trt=args.use_trt,
        )

    elif args.command == "batch":
        model_config = {**DEFAULT_CONFIG, **(ext_config or {})}
        engine = RadarPillarNetInference(
            model_path=args.model,
            config=model_config,
            score_threshold=args.score_threshold,
            use_trt=args.use_trt,
        )

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all radar data files
        input_files = sorted(
            list(input_dir.glob("*.npz"))
            + list(input_dir.glob("*.bin"))
            + list(input_dir.glob("*.npy"))
        )
        print(f"[INFO] Processing {len(input_files)} files...")

        total_time = 0.0
        total_dets = 0

        for i, input_file in enumerate(input_files):
            # Load data
            if input_file.suffix == ".npz":
                data = np.load(str(input_file))
                if "points" in data:
                    dets, elapsed = engine.predict_raw(data["points"])
                else:
                    dets, elapsed = engine.predict_pillars(
                        data["pillar_features"],
                        data["pillar_mask"],
                        data["pillar_coords"],
                    )
            elif input_file.suffix == ".bin":
                points = np.fromfile(str(input_file), dtype=np.float32).reshape(-1, 6)
                dets, elapsed = engine.predict_raw(points)
            else:
                points = np.load(str(input_file))
                dets, elapsed = engine.predict_raw(points)

            total_time += elapsed
            total_dets += len(dets)

            # Save output
            out_name = input_file.stem + (".csv" if args.format == "csv" else ".json")
            out_path = str(output_dir / out_name)
            if args.format == "csv":
                save_detections_csv(dets, out_path)
            else:
                save_detections_json(dets, out_path)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(input_files)} | "
                      f"Avg: {total_time/(i+1):.1f}ms | "
                      f"Dets: {total_dets}")

        avg_time = total_time / max(len(input_files), 1)
        print(f"\n[INFO] Batch inference complete:")
        print(f"  Total files:    {len(input_files)}")
        print(f"  Total dets:     {total_dets}")
        print(f"  Avg time/frame: {avg_time:.1f} ms")
        print(f"  Throughput:     {1000.0/max(avg_time, 1e-6):.1f} FPS")

    elif args.command == "benchmark":
        stats = benchmark_throughput(
            model_path=args.model,
            config=ext_config,
            batch_size=args.batch_size,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
            use_trt=args.use_trt,
            device=args.device,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"[INFO] Benchmark results saved to: {args.output}")

    else:
        print("Usage: python inference.py {infer|batch|benchmark} [options]")
        print("Run with --help for details.")
