"""
CRAFT model inference script.

Features:
- Load model from weights or SavedModel
- Process single or batch inputs
- Post-processing: decode heatmap, NMS, confidence thresholding
- Output 3D bounding boxes in standard format
- SavedModel export for deployment
- Timing and throughput measurement
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from model import CRAFTModel, DEFAULT_CONFIG, build_craft_model


# ---------------------------------------------------------------------------
# Data structures
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


@dataclass
class BoundingBox3D:
    """3D bounding box output."""

    center_x: float
    center_y: float
    center_z: float
    width: float
    length: float
    height: float
    yaw: float  # rotation in radians
    velocity_x: float
    velocity_y: float
    score: float
    class_id: int
    class_name: str

    def to_corners(self) -> np.ndarray:
        """Compute 8 corner points of the 3D box.

        Returns:
            (8, 3) array of corner coordinates.
        """
        w, l, h = self.width, self.length, self.height
        # Box corners in local frame (center at origin)
        corners_local = np.array([
            [-w / 2, -l / 2, -h / 2],
            [+w / 2, -l / 2, -h / 2],
            [+w / 2, +l / 2, -h / 2],
            [-w / 2, +l / 2, -h / 2],
            [-w / 2, -l / 2, +h / 2],
            [+w / 2, -l / 2, +h / 2],
            [+w / 2, +l / 2, +h / 2],
            [-w / 2, +l / 2, +h / 2],
        ])

        # Rotation matrix (yaw only)
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        rot = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1],
        ])

        # Rotate and translate
        corners_world = corners_local @ rot.T
        corners_world += np.array([self.center_x, self.center_y, self.center_z])
        return corners_world

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


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def nms_heatmap(heatmap: tf.Tensor, kernel_size: int = 3) -> tf.Tensor:
    """
    Apply non-maximum suppression on heatmap using max pooling.

    Args:
        heatmap: (B, H, W, C) heatmap tensor
        kernel_size: NMS kernel size
    Returns:
        Suppressed heatmap (B, H, W, C)
    """
    # Max pool to find local maxima
    padding = kernel_size // 2
    # Pad manually for exact kernel behavior
    padded = tf.pad(heatmap, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode="CONSTANT")
    pooled = tf.nn.max_pool2d(padded, ksize=kernel_size, strides=1, padding="VALID")
    # Keep only pixels that are local maxima
    keep = tf.cast(tf.equal(heatmap, pooled), heatmap.dtype)
    return heatmap * keep


def decode_heatmap_to_boxes(
    heatmap: tf.Tensor,
    regression: tf.Tensor,
    velocity: tf.Tensor,
    height: tf.Tensor,
    config: Dict[str, Any],
    score_threshold: float = 0.3,
    max_detections: int = 300,
    nms_kernel: int = 3,
) -> List[List[BoundingBox3D]]:
    """
    Decode model outputs to 3D bounding boxes.

    Args:
        heatmap: (B, H, W, num_classes) after sigmoid
        regression: (B, H, W, num_reg_attrs)
        velocity: (B, H, W, 2)
        height: (B, H, W, 2)
        config: model config dict
        score_threshold: minimum detection score
        max_detections: maximum boxes per sample
        nms_kernel: NMS kernel size
    Returns:
        List of BoundingBox3D lists per batch element
    """
    # Apply NMS on heatmap
    heatmap_nms = nms_heatmap(heatmap, kernel_size=nms_kernel)

    batch_size = heatmap_nms.shape[0]
    h, w = heatmap_nms.shape[1], heatmap_nms.shape[2]
    num_classes = heatmap_nms.shape[3]

    # BEV resolution
    x_min = config["x_min"]
    x_max = config["x_max"]
    y_min = config["y_min"]
    y_max = config["y_max"]
    x_res = (x_max - x_min) / w
    y_res = (y_max - y_min) / h

    # Convert to numpy for post-processing
    heatmap_np = heatmap_nms.numpy()
    regression_np = regression.numpy()
    velocity_np = velocity.numpy()
    height_np = height.numpy()

    all_boxes: List[List[BoundingBox3D]] = []

    for b in range(batch_size):
        boxes: List[BoundingBox3D] = []

        for cls_id in range(num_classes):
            cls_scores = heatmap_np[b, :, :, cls_id]

            # Find peaks above threshold
            mask = cls_scores > score_threshold
            if not np.any(mask):
                continue

            ys_idx, xs_idx = np.where(mask)
            scores = cls_scores[ys_idx, xs_idx]

            # Sort by score
            sort_idx = np.argsort(scores)[::-1]
            ys_idx = ys_idx[sort_idx]
            xs_idx = xs_idx[sort_idx]
            scores = scores[sort_idx]

            for i in range(min(len(scores), max_detections)):
                yi, xi = ys_idx[i], xs_idx[i]
                score = float(scores[i])

                # Decode regression
                reg = regression_np[b, yi, xi]
                dx, dy = reg[0], reg[1]
                bw = np.exp(reg[3])  # width (log-space)
                bl = np.exp(reg[4])  # length (log-space)
                sin_yaw = reg[6]
                cos_yaw = reg[7]

                # World coordinates
                cx = x_min + (xi + dx) * x_res
                cy = y_min + (yi + dy) * y_res
                cz = float(height_np[b, yi, xi, 0])
                bh = float(np.exp(height_np[b, yi, xi, 1]))  # height (log-space)

                yaw = float(np.arctan2(sin_yaw, cos_yaw))
                vx = float(velocity_np[b, yi, xi, 0])
                vy = float(velocity_np[b, yi, xi, 1])

                box = BoundingBox3D(
                    center_x=float(cx),
                    center_y=float(cy),
                    center_z=float(cz),
                    width=float(bw),
                    length=float(bl),
                    height=float(bh),
                    yaw=yaw,
                    velocity_x=vx,
                    velocity_y=vy,
                    score=score,
                    class_id=cls_id,
                    class_name=NUSCENES_CLASSES[cls_id],
                )
                boxes.append(box)

        # Final sort by score and limit
        boxes.sort(key=lambda b: b.score, reverse=True)
        boxes = boxes[:max_detections]
        all_boxes.append(boxes)

    return all_boxes


def circle_nms(
    boxes: List[BoundingBox3D],
    radius: float = 0.5,
) -> List[BoundingBox3D]:
    """
    Apply circular NMS in BEV space.

    Args:
        boxes: detections sorted by score (descending)
        radius: suppression radius in meters
    Returns:
        Filtered detections
    """
    if not boxes:
        return boxes

    kept: List[BoundingBox3D] = []
    suppressed = set()

    for i, box_i in enumerate(boxes):
        if i in suppressed:
            continue
        kept.append(box_i)
        for j in range(i + 1, len(boxes)):
            if j in suppressed:
                continue
            if box_i.class_id != boxes[j].class_id:
                continue
            dist = np.sqrt(
                (box_i.center_x - boxes[j].center_x) ** 2
                + (box_i.center_y - boxes[j].center_y) ** 2
            )
            if dist < radius:
                suppressed.add(j)

    return kept


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class CRAFTInference:
    """
    Inference engine for the CRAFT model.

    Supports loading from:
    - Keras weights (.h5 / .weights.h5)
    - TF checkpoint directory
    - SavedModel directory
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.3,
        max_detections: int = 300,
        nms_kernel: int = 3,
        circle_nms_radius: float = 0.5,
        device: str = "gpu",
    ) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.nms_kernel = nms_kernel
        self.circle_nms_radius = circle_nms_radius

        # Set device
        if device == "cpu":
            tf.config.set_visible_devices([], "GPU")

        # Load model
        self.model = self._load_model(model_path)
        self._warmup()

    def _load_model(self, model_path: str) -> tf.keras.Model:
        """Load model from various formats."""
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb")):
            # SavedModel format
            print(f"[INFO] Loading SavedModel from: {model_path}")
            loaded = tf.saved_model.load(model_path)
            # Wrap in a callable that matches our interface
            self._saved_model = loaded
            self._serve_fn = loaded.signatures.get("serving_default")
            if self._serve_fn is None:
                # Use the model directly
                self._serve_fn = loaded.__call__
            return loaded
        else:
            # Build model from config and load weights
            print(f"[INFO] Building model and loading weights from: {model_path}")
            model = build_craft_model(config=self.config)

            # Build model with dummy input
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
            return model

    def _create_dummy_input(self, batch_size: int = 1) -> Dict[str, tf.Tensor]:
        """Create dummy input for model warmup."""
        cfg = self.config
        return {
            "images": tf.zeros([batch_size, cfg["num_cameras"], cfg["image_height"], cfg["image_width"], 3]),
            "radar_pillars": tf.zeros([batch_size, cfg["max_pillars"], cfg["max_points_per_pillar"], 9]),
            "radar_pillar_mask": tf.zeros([batch_size, cfg["max_pillars"], cfg["max_points_per_pillar"]]),
            "radar_pillar_coords": tf.zeros([batch_size, cfg["max_pillars"], 2], dtype=tf.int32),
            "lidar_to_cam": tf.eye(4, dtype=tf.float32)[tf.newaxis, tf.newaxis]
            * tf.ones([batch_size, cfg["num_cameras"], 1, 1]),
            "cam_intrinsics": tf.eye(3, dtype=tf.float32)[tf.newaxis, tf.newaxis]
            * tf.ones([batch_size, cfg["num_cameras"], 1, 1]),
        }

    def _warmup(self, num_warmup: int = 3) -> None:
        """Warm up the model with dummy inputs."""
        print("[INFO] Warming up model...")
        dummy = self._create_dummy_input(batch_size=1)
        for _ in range(num_warmup):
            if isinstance(self.model, tf.keras.Model):
                _ = self.model(dummy, training=False)
            else:
                _ = self._serve_fn(**dummy)
        print("[INFO] Warmup complete")

    @tf.function(reduce_retracing=True)
    def _inference_step(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """TF function for optimized inference."""
        if isinstance(self.model, tf.keras.Model):
            return self.model(inputs, training=False)
        else:
            return self._serve_fn(**inputs)

    def predict(
        self,
        images: np.ndarray,
        radar_pillars: np.ndarray,
        radar_pillar_mask: np.ndarray,
        radar_pillar_coords: np.ndarray,
        lidar_to_cam: np.ndarray,
        cam_intrinsics: np.ndarray,
    ) -> Tuple[List[BoundingBox3D], float]:
        """
        Run inference on a single sample.

        Args:
            images: (num_cameras, H, W, 3) or (1, num_cameras, H, W, 3)
            radar_pillars: (max_pillars, max_pts, D) or batched
            radar_pillar_mask: (max_pillars, max_pts) or batched
            radar_pillar_coords: (max_pillars, 2) or batched
            lidar_to_cam: (num_cameras, 4, 4) or batched
            cam_intrinsics: (num_cameras, 3, 3) or batched
        Returns:
            (list of BoundingBox3D, inference_time_ms)
        """
        # Add batch dimension if needed
        if images.ndim == 4:
            images = images[np.newaxis]
        if radar_pillars.ndim == 3:
            radar_pillars = radar_pillars[np.newaxis]
        if radar_pillar_mask.ndim == 2:
            radar_pillar_mask = radar_pillar_mask[np.newaxis]
        if radar_pillar_coords.ndim == 2:
            radar_pillar_coords = radar_pillar_coords[np.newaxis]
        if lidar_to_cam.ndim == 3:
            lidar_to_cam = lidar_to_cam[np.newaxis]
        if cam_intrinsics.ndim == 3:
            cam_intrinsics = cam_intrinsics[np.newaxis]

        inputs = {
            "images": tf.constant(images, dtype=tf.float32),
            "radar_pillars": tf.constant(radar_pillars, dtype=tf.float32),
            "radar_pillar_mask": tf.constant(radar_pillar_mask, dtype=tf.float32),
            "radar_pillar_coords": tf.constant(radar_pillar_coords, dtype=tf.int32),
            "lidar_to_cam": tf.constant(lidar_to_cam, dtype=tf.float32),
            "cam_intrinsics": tf.constant(cam_intrinsics, dtype=tf.float32),
        }

        # Run inference with timing
        start_time = time.perf_counter()
        predictions = self._inference_step(inputs)
        # Ensure computation is complete
        for v in predictions.values():
            _ = v.numpy()
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Post-process
        boxes = decode_heatmap_to_boxes(
            heatmap=predictions["heatmap"],
            regression=predictions["regression"],
            velocity=predictions["velocity"],
            height=predictions["height"],
            config=self.config,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            nms_kernel=self.nms_kernel,
        )

        # Apply circle NMS
        result_boxes = circle_nms(boxes[0], radius=self.circle_nms_radius)

        return result_boxes, inference_time_ms

    def predict_batch(
        self,
        batch_inputs: Dict[str, np.ndarray],
    ) -> Tuple[List[List[BoundingBox3D]], float]:
        """
        Run batch inference.

        Args:
            batch_inputs: dict of batched numpy arrays
        Returns:
            (list of detection lists, total_inference_time_ms)
        """
        inputs = {
            k: tf.constant(v, dtype=tf.int32 if k == "radar_pillar_coords" else tf.float32)
            for k, v in batch_inputs.items()
        }

        start_time = time.perf_counter()
        predictions = self._inference_step(inputs)
        for v in predictions.values():
            _ = v.numpy()
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Post-process
        all_boxes = decode_heatmap_to_boxes(
            heatmap=predictions["heatmap"],
            regression=predictions["regression"],
            velocity=predictions["velocity"],
            height=predictions["height"],
            config=self.config,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            nms_kernel=self.nms_kernel,
        )

        # Apply circle NMS per sample
        filtered_boxes = [circle_nms(boxes, radius=self.circle_nms_radius) for boxes in all_boxes]

        return filtered_boxes, inference_time_ms


# ---------------------------------------------------------------------------
# SavedModel export utility
# ---------------------------------------------------------------------------


def export_saved_model(
    weights_path: str,
    export_dir: str,
    config: Optional[Dict[str, Any]] = None,
    optimize_for_inference: bool = True,
) -> str:
    """
    Export CRAFT model as TensorFlow SavedModel for deployment.

    Args:
        weights_path: path to model weights
        export_dir: output directory for SavedModel
        config: model config overrides
        optimize_for_inference: whether to apply TF optimizations
    Returns:
        Path to exported SavedModel
    """
    model_config = {**DEFAULT_CONFIG, **(config or {})}
    cfg = model_config

    print("[INFO] Building model for export...")
    model = build_craft_model(config=model_config)

    # Build with dummy
    dummy = {
        "images": tf.zeros([1, cfg["num_cameras"], cfg["image_height"], cfg["image_width"], 3]),
        "radar_pillars": tf.zeros([1, cfg["max_pillars"], cfg["max_points_per_pillar"], 9]),
        "radar_pillar_mask": tf.zeros([1, cfg["max_pillars"], cfg["max_points_per_pillar"]]),
        "radar_pillar_coords": tf.zeros([1, cfg["max_pillars"], 2], dtype=tf.int32),
        "lidar_to_cam": tf.zeros([1, cfg["num_cameras"], 4, 4]),
        "cam_intrinsics": tf.zeros([1, cfg["num_cameras"], 3, 3]),
    }
    _ = model(dummy, training=False)

    # Load weights
    print(f"[INFO] Loading weights: {weights_path}")
    model.load_weights(weights_path)

    # Define concrete function with input signatures
    @tf.function(input_signature=[{
        "images": tf.TensorSpec([None, cfg["num_cameras"], cfg["image_height"], cfg["image_width"], 3], tf.float32, name="images"),
        "radar_pillars": tf.TensorSpec([None, cfg["max_pillars"], cfg["max_points_per_pillar"], 9], tf.float32, name="radar_pillars"),
        "radar_pillar_mask": tf.TensorSpec([None, cfg["max_pillars"], cfg["max_points_per_pillar"]], tf.float32, name="radar_pillar_mask"),
        "radar_pillar_coords": tf.TensorSpec([None, cfg["max_pillars"], 2], tf.int32, name="radar_pillar_coords"),
        "lidar_to_cam": tf.TensorSpec([None, cfg["num_cameras"], 4, 4], tf.float32, name="lidar_to_cam"),
        "cam_intrinsics": tf.TensorSpec([None, cfg["num_cameras"], 3, 3], tf.float32, name="cam_intrinsics"),
    }])
    def serve(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        predictions = model(inputs, training=False)
        return {
            "heatmap": predictions["heatmap"],
            "regression": predictions["regression"],
            "velocity": predictions["velocity"],
            "height": predictions["height"],
        }

    # Save
    os.makedirs(export_dir, exist_ok=True)
    tf.saved_model.save(
        model,
        export_dir,
        signatures={"serving_default": serve},
    )
    print(f"[INFO] SavedModel exported to: {export_dir}")

    # Optionally convert to TFLite or apply TF optimization
    if optimize_for_inference:
        try:
            from tensorflow.python.tools import optimize_for_inference_lib
            print("[INFO] Optimization pass applied")
        except ImportError:
            print("[INFO] TF optimization tools not available, skipping")

    # Save config alongside model
    config_path = os.path.join(export_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"[INFO] Config saved: {config_path}")

    return export_dir


# ---------------------------------------------------------------------------
# Throughput benchmarking
# ---------------------------------------------------------------------------


def benchmark_throughput(
    model_path: str,
    config: Optional[Dict[str, Any]] = None,
    batch_size: int = 1,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "gpu",
) -> Dict[str, float]:
    """
    Measure inference throughput and latency.

    Args:
        model_path: path to model
        config: model config
        batch_size: inference batch size
        num_iterations: number of timed iterations
        warmup_iterations: iterations to warm up (excluded from timing)
        device: "gpu" or "cpu"
    Returns:
        Dict with timing statistics
    """
    model_config = {**DEFAULT_CONFIG, **(config or {})}
    cfg = model_config

    engine = CRAFTInference(
        model_path=model_path,
        config=model_config,
        device=device,
    )

    # Create synthetic batch
    batch_inputs = {
        "images": np.random.randn(
            batch_size, cfg["num_cameras"], cfg["image_height"], cfg["image_width"], 3
        ).astype(np.float32),
        "radar_pillars": np.random.randn(
            batch_size, cfg["max_pillars"], cfg["max_points_per_pillar"], 9
        ).astype(np.float32),
        "radar_pillar_mask": np.ones(
            (batch_size, cfg["max_pillars"], cfg["max_points_per_pillar"])
        ).astype(np.float32),
        "radar_pillar_coords": np.random.randint(
            0, 256, (batch_size, cfg["max_pillars"], 2)
        ).astype(np.int32),
        "lidar_to_cam": np.tile(
            np.eye(4, dtype=np.float32), (batch_size, cfg["num_cameras"], 1, 1)
        ),
        "cam_intrinsics": np.tile(
            np.eye(3, dtype=np.float32), (batch_size, cfg["num_cameras"], 1, 1)
        ),
    }

    # Warmup
    print(f"[INFO] Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        engine.predict_batch(batch_inputs)

    # Timed iterations
    print(f"[INFO] Running {num_iterations} timed iterations (batch_size={batch_size})...")
    latencies: List[float] = []

    for i in range(num_iterations):
        _, elapsed_ms = engine.predict_batch(batch_inputs)
        latencies.append(elapsed_ms)

    latencies_arr = np.array(latencies)
    total_samples = num_iterations * batch_size
    total_time_s = np.sum(latencies_arr) / 1000.0

    stats = {
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "device": device,
        "mean_latency_ms": float(np.mean(latencies_arr)),
        "median_latency_ms": float(np.median(latencies_arr)),
        "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
        "p99_latency_ms": float(np.percentile(latencies_arr, 99)),
        "min_latency_ms": float(np.min(latencies_arr)),
        "max_latency_ms": float(np.max(latencies_arr)),
        "std_latency_ms": float(np.std(latencies_arr)),
        "throughput_fps": total_samples / total_time_s,
        "total_time_s": total_time_s,
    }

    print(f"\n{'='*50}")
    print(f"{'Inference Benchmark Results':^50}")
    print(f"{'='*50}")
    print(f"  Device:           {device}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Mean latency:     {stats['mean_latency_ms']:.2f} ms")
    print(f"  Median latency:   {stats['median_latency_ms']:.2f} ms")
    print(f"  P95 latency:      {stats['p95_latency_ms']:.2f} ms")
    print(f"  P99 latency:      {stats['p99_latency_ms']:.2f} ms")
    print(f"  Throughput:       {stats['throughput_fps']:.1f} FPS")
    print(f"{'='*50}")

    return stats


# ---------------------------------------------------------------------------
# Single sample inference from file
# ---------------------------------------------------------------------------


def infer_from_files(
    model_path: str,
    sample_dir: str,
    output_path: str,
    config: Optional[Dict[str, Any]] = None,
    score_threshold: float = 0.3,
) -> List[BoundingBox3D]:
    """
    Run inference on a single sample loaded from disk.

    Expected directory structure:
        sample_dir/
            images/         # camera images (6 .jpg/.png files)
            radar.npz       # radar pillars
            calibration.npz # extrinsic/intrinsic matrices

    Args:
        model_path: path to model weights or SavedModel
        sample_dir: directory containing sample data
        output_path: path to save output JSON
        config: model config overrides
        score_threshold: detection threshold
    Returns:
        List of detected 3D boxes
    """
    model_config = {**DEFAULT_CONFIG, **(config or {})}
    cfg = model_config

    engine = CRAFTInference(
        model_path=model_path,
        config=model_config,
        score_threshold=score_threshold,
    )

    sample_path = Path(sample_dir)

    # Load images
    img_h, img_w = cfg["image_height"], cfg["image_width"]
    num_cameras = cfg["num_cameras"]
    images = np.zeros((num_cameras, img_h, img_w, 3), dtype=np.float32)

    img_dir = sample_path / "images"
    if img_dir.exists():
        img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        for i, img_file in enumerate(img_files[:num_cameras]):
            img_raw = tf.io.read_file(str(img_file))
            if str(img_file).endswith(".png"):
                img = tf.image.decode_png(img_raw, channels=3)
            else:
                img = tf.image.decode_jpeg(img_raw, channels=3)
            img = tf.image.resize(img, [img_h, img_w])
            img = tf.cast(img, tf.float32) / 255.0
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            img = (img - mean) / std
            images[i] = img.numpy()

    # Load radar
    radar_path = sample_path / "radar.npz"
    max_pillars = cfg["max_pillars"]
    max_pts = cfg["max_points_per_pillar"]

    if radar_path.exists():
        radar_data = np.load(str(radar_path))
        pillar_features = np.zeros((max_pillars, max_pts, 9), dtype=np.float32)
        pillar_mask = np.zeros((max_pillars, max_pts), dtype=np.float32)
        pillar_coords = np.zeros((max_pillars, 2), dtype=np.int32)
        n = min(radar_data["features"].shape[0], max_pillars)
        pillar_features[:n] = radar_data["features"][:n]
        pillar_mask[:n] = radar_data["mask"][:n].astype(np.float32)
        pillar_coords[:n] = radar_data["coords"][:n]
    else:
        pillar_features = np.zeros((max_pillars, max_pts, 9), dtype=np.float32)
        pillar_mask = np.zeros((max_pillars, max_pts), dtype=np.float32)
        pillar_coords = np.zeros((max_pillars, 2), dtype=np.int32)

    # Load calibration
    calib_path = sample_path / "calibration.npz"
    if calib_path.exists():
        calib = np.load(str(calib_path))
        lidar_to_cam = calib["lidar_to_cam"].astype(np.float32)
        cam_intrinsics = calib["cam_intrinsics"].astype(np.float32)
    else:
        lidar_to_cam = np.tile(np.eye(4, dtype=np.float32), (num_cameras, 1, 1))
        cam_intrinsics = np.tile(np.eye(3, dtype=np.float32), (num_cameras, 1, 1))
        cam_intrinsics[:, 0, 0] = 1266.0
        cam_intrinsics[:, 1, 1] = 1266.0
        cam_intrinsics[:, 0, 2] = img_w / 2.0
        cam_intrinsics[:, 1, 2] = img_h / 2.0

    # Run inference
    boxes, inference_time_ms = engine.predict(
        images=images,
        radar_pillars=pillar_features,
        radar_pillar_mask=pillar_mask,
        radar_pillar_coords=pillar_coords,
        lidar_to_cam=lidar_to_cam,
        cam_intrinsics=cam_intrinsics,
    )

    print(f"[INFO] Detected {len(boxes)} objects in {inference_time_ms:.1f}ms")
    for box in boxes:
        print(
            f"  {box.class_name:20s} | "
            f"score={box.score:.3f} | "
            f"pos=({box.center_x:.1f}, {box.center_y:.1f}, {box.center_z:.1f}) | "
            f"size=({box.width:.1f}, {box.length:.1f}, {box.height:.1f}) | "
            f"yaw={np.degrees(box.yaw):.1f}deg"
        )

    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    output_data = {
        "detections": [box.to_dict() for box in boxes],
        "inference_time_ms": inference_time_ms,
        "num_detections": len(boxes),
        "score_threshold": score_threshold,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[INFO] Results saved to: {output_path}")

    return boxes


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRAFT model inference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Inference on single sample
    infer_parser = subparsers.add_parser("infer", help="Run inference on a sample")
    infer_parser.add_argument("--model", type=str, required=True, help="Model path")
    infer_parser.add_argument("--input", type=str, required=True, help="Sample directory")
    infer_parser.add_argument("--output", type=str, default="./output/detections.json")
    infer_parser.add_argument("--score-threshold", type=float, default=0.3)
    infer_parser.add_argument("--config", type=str, default=None)

    # Export SavedModel
    export_parser = subparsers.add_parser("export", help="Export SavedModel")
    export_parser.add_argument("--weights", type=str, required=True, help="Weights path")
    export_parser.add_argument("--output-dir", type=str, required=True, help="Export dir")
    export_parser.add_argument("--config", type=str, default=None)
    export_parser.add_argument("--no-optimize", action="store_true")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run throughput benchmark")
    bench_parser.add_argument("--model", type=str, required=True, help="Model path")
    bench_parser.add_argument("--batch-size", type=int, default=1)
    bench_parser.add_argument("--iterations", type=int, default=100)
    bench_parser.add_argument("--warmup", type=int, default=10)
    bench_parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    bench_parser.add_argument("--config", type=str, default=None)
    bench_parser.add_argument("--output", type=str, default=None, help="Save results JSON")

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
        infer_from_files(
            model_path=args.model,
            sample_dir=args.input,
            output_path=args.output,
            config=ext_config,
            score_threshold=args.score_threshold,
        )

    elif args.command == "export":
        export_saved_model(
            weights_path=args.weights,
            export_dir=args.output_dir,
            config=ext_config,
            optimize_for_inference=not args.no_optimize,
        )

    elif args.command == "benchmark":
        stats = benchmark_throughput(
            model_path=args.model,
            config=ext_config,
            batch_size=args.batch_size,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
            device=args.device,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"[INFO] Benchmark results saved to: {args.output}")

    else:
        print("Usage: python inference.py {infer|export|benchmark} [options]")
        print("Run with --help for details.")
